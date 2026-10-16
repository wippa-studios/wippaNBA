"""Compare the feature contract training uses with the one inference feeds.

This exists to catch a silent misalignment: XGBoost and Keras are fed a
positional matrix, so a frame with the right columns in the WRONG order raises
nothing and predicts wrong. ``src/Utils/features.py`` validates that at the
inference boundary; this reports the same disagreement from the training side.

WHY THE DROP LISTS ARE IMPORTED RATHER THAN WRITTEN HERE. This script used to
declare its own copy of the trainer drop list, and the copy had drifted from the
six trainers that call ``training_common.drop_columns_for``:

  * it kept ``OU`` in the moneyline list, which every moneyline trainer drops
    (the closing total is a totals-model feature);
  * it omitted ``GameDate`` and both pandas ``index`` columns, which every
    trainer drops;
  * it named ``TEAM_ID``/``TEAM_ID.1``, columns ``Create_Games`` strips before
    writing the table, so they matched nothing and ``117 - 10`` never
    reconciled.

Six of its ten entries were still correct, which is the worse shape: the checker
agreed with the trainers on most of the contract and disagreed exactly where a
permutation could hide. So both lists below are now the trainer function's own
return values, and ``Tests/test_feature_alignment_checker.py`` pins the equality
-- list AND order -- so the copy cannot start drifting again.

WHAT WAS WRONG, against the live 117-column ``dataset_2012-26`` table: the two
lists produced 109 and 110 features where the trainers produce 106 and 107, and
no committed model has either width. The TOTALS list did not drop the closing
total ``OU``, so ``load_live_frames`` re-attaching ``OU`` for the totals frame
below was reported as an EXTRA column on every live run, and ``OU`` -- the 104th
input, sitting between the away-team block and ``OU-Cover`` -- was missing from
the expectation the frame was checked against. Neither list dropped ``index``,
``index.1`` or ``GameDate``, three phantom inputs. Both named ``TEAM_ID`` and
``TEAM_ID.1``, which are not columns of that table at all.
"""

import argparse
import sqlite3
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    from keras.models import load_model
except ImportError:
    load_model = None

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
MODEL_DIR = BASE_DIR / "Models"
TRAINING_DIR = BASE_DIR / "src" / "Train-Models"

# `src/Train-Models` contains a HYPHEN, so it can never be an importable package
# (renaming it is tracked as follow-up work). It goes on sys.path instead and its
# modules import under plain top-level names, which is the bootstrap
# `training_common`'s own docstring describes for the trainers, the one
# `scripts/evaluate.py` uses, and the one `Tests/test_game_date.py` and
# `Tests/test_training_layer.py` already use. `training_common` itself puts the
# repository root on sys.path in turn, so importing it here is self-sufficient
# whatever the working directory.
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

import training_common  # noqa: E402

# THE FEATURE CONTRACT, TAKEN FROM THE TRAINERS.
#
# `drop_columns_for(MONEYLINE_TARGET)` removes the closing total OU, because a
# moneyline model is not allowed to see it; `drop_columns_for(TOTALS_TARGET)`
# keeps it, which is why `load_live_frames` copies the moneyline frame and
# re-attaches OU for the totals frame below. The two frames therefore have
# different widths and this script compares each against its own list.
#
# `include_team_ids` is left at its default, matching the four XGBoost and NN
# trainers whose committed artifacts live in Models/. The two logistic
# regression trainers pass True; Create_Games drops TEAM_ID/TEAM_ID.1 before
# writing the table, so that entry matches nothing on any dataset the repository
# ships and the feature list is identical either way. See
# `training_common.drop_columns_for` for the full argument.
DROP_COLUMNS_ML = training_common.drop_columns_for(training_common.MONEYLINE_TARGET)
DROP_COLUMNS_UO = training_common.drop_columns_for(training_common.TOTALS_TARGET)


def get_table_columns(dataset_name):
    # Checked through the trainer's own guard rather than by letting sqlite3
    # raise `no such table`: an unknown name now reports the tables that DO
    # exist, and a same-day-join (leaky) table is refused by name instead of
    # being measured as though it were sound.
    dataset_name = training_common.resolve_dataset_table(dataset_name)
    with sqlite3.connect(f"file:{DATASET_DB}?mode=ro", uri=True) as con:
        df = pd.read_sql_query(f'SELECT * FROM "{dataset_name}" LIMIT 1', con)
    return list(df.columns)


def get_training_columns(dataset_name, drop_columns):
    cols = get_table_columns(dataset_name)
    return [col for col in cols if col not in drop_columns]


def load_frame_columns(path):
    df = pd.read_csv(path)
    return list(df.columns)


def load_live_frames():
    sys.path.insert(0, str(BASE_DIR))
    from main import create_todays_games_data, load_schedule, DATA_URL, TODAYS_GAMES_URL
    from src.Utils.tools import get_json_data, to_data_frame, get_todays_games_json, create_todays_games

    stats_json = get_json_data(DATA_URL)
    df = to_data_frame(stats_json)
    schedule_df = load_schedule()
    today = datetime.today()
    games = create_todays_games(get_todays_games_json(TODAYS_GAMES_URL))
    data, todays_games_uo, frame_ml, _, _ = create_todays_games_data(
        games, df, None, schedule_df, today
    )
    if frame_ml is None or frame_ml.empty:
        raise RuntimeError("No games found to build inference frames.")
    frame_uo = frame_ml.copy()
    frame_uo["OU"] = np.asarray(todays_games_uo, dtype=float)
    return frame_ml, frame_uo


def find_latest_model(pattern):
    candidates = list(MODEL_DIR.rglob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def describe_model_input(model_path, label):
    if model_path is None:
        print(f"{label} model: not found")
        return None
    if load_model is None:
        print(f"{label} model: keras not installed")
        return None
    model = load_model(str(model_path), compile=False)
    input_features = model.input_shape[-1]
    print(f"{label} model input features: {input_features} ({model_path})")
    return input_features


def describe_xgb_input(model_path, label):
    if model_path is None:
        print(f"{label} model: not found")
        return None
    if xgb is None:
        print(f"{label} model: xgboost not installed")
        return None
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    feature_count = booster.num_features()
    print(f"{label} model input features: {feature_count} ({model_path})")
    return feature_count


def compare_columns(label, train_cols, infer_cols):
    train_set = set(train_cols)
    infer_set = set(infer_cols)
    missing = sorted(train_set - infer_set)
    extra = sorted(infer_set - train_set)
    print(f"{label} columns: train={len(train_cols)} infer={len(infer_cols)}")
    print(f"{label} missing: {len(missing)}; extra: {len(extra)}")
    if missing:
        print(f"{label} missing sample: {', '.join(missing[:20])}")
    if extra:
        print(f"{label} extra sample: {', '.join(extra[:20])}")


def main():
    parser = argparse.ArgumentParser(description="Compare training vs inference feature sizes.")
    # The trainer's constant, not a literal. The default used to be
    # "dataset_2012-26_new", a table that does not exist in Data/dataset.sqlite,
    # so a bare `python scripts/check_feature_alignment.py` died inside sqlite3
    # before printing a single number. Same drift this file was fixed for.
    parser.add_argument(
        "--dataset",
        default=training_common.DEFAULT_DATASET,
        help="Dataset table name (default: the trainer's DEFAULT_DATASET).",
    )
    parser.add_argument("--frame-ml", help="CSV path to inference frame_ml.")
    parser.add_argument("--frame-uo", help="CSV path to inference frame_uo.")
    parser.add_argument("--use-live", action="store_true", help="Build inference frames from live data.")
    parser.add_argument("--nn-ml-model", help="Path to NN ML model.")
    parser.add_argument("--nn-uo-model", help="Path to NN OU model.")
    parser.add_argument("--xgb-ml-model", help="Path to XGBoost ML model.")
    parser.add_argument("--xgb-uo-model", help="Path to XGBoost OU model.")
    args = parser.parse_args()

    train_cols_ml = get_training_columns(args.dataset, DROP_COLUMNS_ML)
    train_cols_uo = get_training_columns(args.dataset, DROP_COLUMNS_UO)
    print(f"Training columns ML: {len(train_cols_ml)}")
    print(f"Training columns OU: {len(train_cols_uo)}")

    frame_ml = frame_uo = None
    if args.use_live:
        frame_ml, frame_uo = load_live_frames()
    if args.frame_ml:
        frame_ml = pd.read_csv(args.frame_ml)
    if args.frame_uo:
        frame_uo = pd.read_csv(args.frame_uo)

    if frame_ml is not None:
        compare_columns("ML", train_cols_ml, list(frame_ml.columns))
    if frame_uo is not None:
        compare_columns("OU", train_cols_uo, list(frame_uo.columns))

    nn_ml_model = Path(args.nn_ml_model) if args.nn_ml_model else find_latest_model("Trained-Model-ML-*")
    nn_ou_model = Path(args.nn_uo_model) if args.nn_uo_model else find_latest_model("Trained-Model-OU-*")
    describe_model_input(nn_ml_model, "NN ML")
    describe_model_input(nn_ou_model, "NN OU")

    xgb_ml_model = Path(args.xgb_ml_model) if args.xgb_ml_model else find_latest_model("XGBoost_*_ML*.json")
    xgb_ou_model = Path(args.xgb_uo_model) if args.xgb_uo_model else find_latest_model("XGBoost_*_UO*.json")
    describe_xgb_input(xgb_ml_model, "XGB ML")
    describe_xgb_input(xgb_ou_model, "XGB OU")


if __name__ == "__main__":
    main()
