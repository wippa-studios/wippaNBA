import copy
import json
import math
import re
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
from colorama import Fore, Style, init, deinit
from keras.models import load_model
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc

try:
    from src.Utils import features
except ImportError:
    features = None

init()

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "Models"
# Both neural trainers write here (issue #16). `MODEL_DIR` is still searched
# for pre-#16 artifacts, but those have no preprocessor sidecar and are
# filtered out by `_select_best_model`.
NN_MODEL_DIR = MODEL_DIR / "NN_Models"
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"

# The one dataset table inference reads, shared with the trainers rather than
# spelled out here. src/Train-Models has a HYPHEN and can never be imported as a
# package, so its shared modules only resolve when the directory is on sys.path
# -- the same convention the six training scripts use. This used to hard-code
# "dataset_2012-24_new", a same-day-join table that carries the target inside
# its features, while XGBoost_Runner named a different one. One name, one source.
TRAIN_MODELS_DIR = BASE_DIR / "src" / "Train-Models"
if str(TRAIN_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_MODELS_DIR))
try:
    from training_common import DEFAULT_DATASET as DATASET_TABLE
    from training_common import OU_PICK_LABELS
except ImportError as _exc:  # pragma: no cover - only in a broken checkout
    # Deliberately NOT a second literal. A fallback name is a name that can drift
    # from the trainers' one, which is the whole defect being fixed here.
    raise RuntimeError(
        f"Could not load the shared dataset default from {TRAIN_MODELS_DIR}. "
        "Inference reads the same table the trainers train on, so guessing a name "
        "here would reintroduce exactly the disagreement this import removes."
    ) from _exc

ML_PATTERN = re.compile(r"Trained-Model-ML-(\d+(?:\.\d+)?)")
OU_PATTERN = re.compile(r"Trained-Model-OU-(\d+(?:\.\d+)?)")

# ---------------------------------------------------------------------------
# Machine-readable contract with Flask/app.py.
#
# After the human-readable output, this runner prints exactly ONE extra line,
# and it is the last line it prints:
#
#     __NBA_PREDICTIONS_JSON__ {"schema":1,"games":[{...},{...}]}
#
# Flask/app.py parses that one line to build the web UI; everything printed
# before it is CLI output and may be reworded freely. The sentinel, the single
# strict-JSON document that follows it, and the keys below are a contract
# between the two files: change one side, change the other.
#
# Keys per game:
#   home_team, away_team            team names as the pipeline received them
#   home_confidence, away_confidence  win probability per team, percent 0-100
#   ou_pick                          "OVER", "UNDER", or "PUSH" (a void total)
#   ou_value, ou_confidence         the totals line and the model's edge on it
#   home_team_ev, away_team_ev       expected value per team, percent points
#   home_team_odds, away_team_odds   American odds as ints, null when missing
#   home_kelly_fraction, away_kelly_fraction  only when -kc was passed
#
# Every numeric field is a JSON number or null: a non-finite float becomes null
# because json.dumps would otherwise emit a bare NaN/Infinity literal, which is
# not valid JSON and breaks a strict parser.
#
# This block is deliberately duplicated in src/Predict/XGBoost_Runner.py. The two
# runners are separate entry points with no shared predict-side module, and the
# two must emit the identical line for Flask to work with either; keep them in
# step when either changes.
# ---------------------------------------------------------------------------
PREDICTIONS_SENTINEL = "__NBA_PREDICTIONS_JSON__ "
PREDICTIONS_SCHEMA_VERSION = 1

_model = None
_ou_model = None
_ml_preprocessor = None
_ou_preprocessor = None
_ml_calibrator = None
_ou_calibrator = None


def _json_safe(value):
    """`value` as something json.dumps can write, with non-finite floats as None.

    numpy scalars are not JSON types, and json.dumps writes NaN/Infinity as bare
    literals that no strict parser accepts, so both are normalised here instead of
    at each call site.
    """
    if value is None or isinstance(value, (str, bool)):
        return value
    if hasattr(value, "item") and not isinstance(value, (int, float)):
        # numpy scalars and other array-likes expose .item(); plain ints and
        # floats do not need it.
        try:
            value = value.item()
        except (AttributeError, ValueError):
            return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    return str(value)


def emit_predictions(games):
    """Print the single machine-readable line documented above, and nothing else.

    allow_nan=False is the backstop: _json_safe has already turned every
    non-finite float into None, so this can only fire if that ever stops being
    true -- and raising here is much better than shipping a payload that Flask's
    strict parser rejects and renders as no data at all.
    """
    payload = {
        "schema": PREDICTIONS_SCHEMA_VERSION,
        "games": [_json_safe(game) for game in games],
    }
    print(PREDICTIONS_SENTINEL + json.dumps(payload, allow_nan=False, separators=(",", ":")))


def _list_model_candidates(prefix):
    candidates = []
    for base in (MODEL_DIR, NN_MODEL_DIR):
        if not base.exists():
            continue
        for path in base.glob(f"{prefix}*"):
            if path.is_dir():
                candidates.append(path)
                continue
            if path.suffix in {".keras", ".h5"}:
                candidates.append(path)
    return candidates


def _sidecar_path(model_path, suffix):
    """The artifact stored beside a model, e.g. ``<stem>_preprocessor.pkl``.

    One naming rule for both sidecars, shared by the trainers (which write
    them) and this runner (which reads them), so a model can never be paired
    with another run's scaler.
    """
    model_path = Path(model_path)
    return model_path.with_name(f"{model_path.stem}_{suffix}")


def _has_preprocessor(model_path):
    return _sidecar_path(model_path, "preprocessor.pkl").exists()


def _select_best_model(prefix, pattern):
    candidates = _list_model_candidates(prefix)
    if not candidates:
        raise FileNotFoundError(f"No model found for prefix {prefix} in {MODEL_DIR}")

    # A model without a preprocessor was trained under the old per-row L2
    # normalisation, which no current artifact can reproduce. Prefer the
    # models that can actually be served, and if none can, say so instead of
    # silently applying a different transform than the one training used.
    usable = [path for path in candidates if _has_preprocessor(path)]
    if not usable:
        raise FileNotFoundError(
            f"No usable model for prefix {prefix}: every candidate in "
            f"{MODEL_DIR} / {NN_MODEL_DIR} predates feature scaling. Retrain "
            "with src/Train-Models/NN_Model_ML.py or NN_Model_UO.py; the trainer "
            "saves the scaler beside the model (issue #16)."
        )
    candidates = usable

    def score(path):
        # The accuracy is parsed out of the FILE NAME, not read from the artifact:
        # a saved Keras model stores no accuracy, and a fresh clone gives every file
        # the same mtime. It is unverified, and here it outranks the mtime outright.
        match = pattern.search(path.name)
        accuracy = float(match.group(1)) if match else 0.0
        if not (0.0 <= accuracy <= 100.0):
            accuracy = 0.0
        return (accuracy, path.stat().st_mtime)

    return max(candidates, key=score)


def _load_calibrator(model_path):
    """The saved calibrator, or None when this model was trained without one."""
    path = _sidecar_path(model_path, "calibration.pkl")
    if not path.exists():
        return None
    return joblib.load(path)


def _load_models():
    global _model, _ou_model
    global _ml_preprocessor, _ou_preprocessor, _ml_calibrator, _ou_calibrator
    if _model is None:
        ml_path = _select_best_model("Trained-Model-ML-", ML_PATTERN)
        _model = load_model(str(ml_path), compile=False)
        _ml_preprocessor = joblib.load(_sidecar_path(ml_path, "preprocessor.pkl"))
        _ml_calibrator = _load_calibrator(ml_path)
    if _ou_model is None:
        ou_path = _select_best_model("Trained-Model-OU-", OU_PATTERN)
        _ou_model = load_model(str(ou_path), compile=False)
        _ou_preprocessor = joblib.load(_sidecar_path(ou_path, "preprocessor.pkl"))
        _ou_calibrator = _load_calibrator(ou_path)


def _predict_matrix(model, calibrator, matrix):
    """Probabilities for a whole matrix, calibrated when the run saved one."""
    if calibrator is not None:
        return np.asarray(calibrator.predict_proba(matrix), dtype=float)
    return np.asarray(model.predict(matrix, verbose=0), dtype=float)


def _dataset_columns():
    """Raw dataset column order, or None when the shipped database is unavailable."""
    if not DATASET_DB.exists():
        return None
    try:
        # Read-only: inference never writes, and a read-write handle on the
        # shipped database is one stray statement away from corrupting it.
        con = sqlite3.connect(f"file:{DATASET_DB}?mode=ro", uri=True)
        try:
            return [row[1] for row in con.execute(f'PRAGMA table_info("{DATASET_TABLE}")')]
        finally:
            con.close()
    except sqlite3.Error:
        return None


def _uo_drop_columns():
    """The shared totals drop list, whichever of its two published names is present.

    Deliberately not the NN totals variant: NN_Model_UO keeps OU, like
    XGBoost_Model_UO, so both totals models share one contract.
    """
    if features is None:
        return None
    for name in ("UO_DROP_COLUMNS", "UO_LABEL_DROP_COLUMNS"):
        drop_columns = getattr(features, name, None)
        if drop_columns:
            return drop_columns
    return None


def _expected_uo_columns(frame_ml):
    """Column order the totals network was trained on: the moneyline order with the total line in place.

    The totals trainer keeps OU (the pre-game total is known before tip-off), so the
    totals network takes one more input than the moneyline network and OU belongs in
    the middle of the frame. Appending it would feed every trailing feature to the
    wrong input, so build the order explicitly.
    """
    ml_columns = list(frame_ml.columns)

    drop_columns = _uo_drop_columns()
    dataset_columns = _dataset_columns() if drop_columns else None
    if dataset_columns:
        expected = list(features.expected_feature_order(dataset_columns, drop_columns))
        if set(expected) == set(ml_columns) | {"OU"} and len(expected) == len(ml_columns) + 1:
            return expected

    # Local fallback when the shared helper or the database is missing: place OU by
    # its neighbour in the training order rather than at a fixed index, so a wider
    # dataset cannot rot the position.
    if "Days-Rest-Home" in ml_columns:
        expected = list(ml_columns)
        expected.insert(expected.index("Days-Rest-Home"), "OU")
        return expected

    print(
        Fore.YELLOW
        + "Warning: could not derive the totals feature order; appending OU and it may be misaligned."
        + Style.RESET_ALL
    )
    return ml_columns + ["OU"]


def _coerce_odds(value):
    """American odds as an int, or None when the line is missing or unusable.

    Manual odds arrive from main.py as strings (input()), while a sportsbook without
    a moneyline for a team yields None (SbrOddsProvider), so both shapes reach here.
    """
    if value is None:
        return None
    try:
        odds = int(float(value))
    except (TypeError, ValueError):
        return None
    return odds if odds else None


def _kelly_fraction(odds, model_prob):
    """Kelly fraction as a number, or None when the line is missing."""
    if odds is None:
        return None
    return kc.calculate_kelly_criterion(odds, model_prob)


def _bankroll_fraction(odds, model_prob):
    """Kelly fraction exactly as it is printed, or "" when the line is missing."""
    fraction = _kelly_fraction(odds, model_prob)
    if fraction is None:
        return ""
    return ' Fraction of Bankroll: ' + str(fraction) + '%'


def nn_runner(frame_ml, todays_games_uo, games, home_team_odds, away_team_odds, kelly_criterion):
    """Score today's slate with the saved moneyline and totals networks.

    ``frame_ml`` arrives RAW from main.py. The train-fitted imputation and
    ``StandardScaler`` are loaded from the chosen artifacts and applied here,
    because refitting them on today's one-row slate would reproduce none of the
    training transform. The old code divided each row by its own L2 norm, a
    per-row gain that no trainer used (issue #16).
    """
    _load_models()

    ml_matrix = _ml_preprocessor.transform(frame_ml)
    ml_predictions_array = _predict_matrix(_model, _ml_calibrator, ml_matrix)

    uo_columns = _expected_uo_columns(frame_ml)
    frame_uo = copy.deepcopy(frame_ml)
    frame_uo['OU'] = np.asarray(todays_games_uo, dtype=float)
    # Select by name in the training order: the network was fitted on the dataset
    # column order, which puts OU in the middle rather than at the end.
    frame_uo = frame_uo[uo_columns]
    if features is not None:
        features.validate_feature_frame(frame_uo, uo_columns, "neural network totals inference frame")
    uo_matrix = _ou_preprocessor.transform(frame_uo)
    ou_predictions_array = _predict_matrix(_ou_model, _ou_calibrator, uo_matrix)

    count = 0
    board = []
    for game in games:
        home_team = game[0]
        away_team = game[1]
        winner = int(np.argmax(ml_predictions_array[count]))
        under_over = int(np.argmax(ou_predictions_array[count]))
        winner_confidence = ml_predictions_array[count]
        un_confidence = ou_predictions_array[count]
        if winner == 1:
            winner_confidence = round(winner_confidence[0][1] * 100, 1)
            if under_over == 0:
                un_confidence = round(ou_predictions_array[count][0] * 100, 1)
                print(
                    Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                    Fore.MAGENTA + OU_PICK_LABELS[under_over] + ' ' + Style.RESET_ALL + str(todays_games_uo[
                                                                        count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
            elif under_over == 1:
                un_confidence = round(ou_predictions_array[count][1] * 100, 1)
                print(
                    Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                    Fore.BLUE + OU_PICK_LABELS[under_over] + ' ' + Style.RESET_ALL + str(todays_games_uo[
                                                                    count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
            else:
                un_confidence = round(ou_predictions_array[count][2] * 100, 1)
                print(
                    Fore.GREEN + home_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ' vs ' + Fore.RED + away_team + Style.RESET_ALL + ': ' +
                    Fore.YELLOW + OU_PICK_LABELS[under_over] + ' ' + Style.RESET_ALL + str(todays_games_uo[
                                                                       count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
        else:
            winner_confidence = round(winner_confidence[0][0] * 100, 1)
            if under_over == 0:
                un_confidence = round(ou_predictions_array[count][0] * 100, 1)
                print(
                    Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ': ' +
                    Fore.MAGENTA + OU_PICK_LABELS[under_over] + ' ' + Style.RESET_ALL + str(todays_games_uo[
                                                                        count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
            elif under_over == 1:
                un_confidence = round(ou_predictions_array[count][1] * 100, 1)
                print(
                    Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ': ' +
                    Fore.BLUE + OU_PICK_LABELS[under_over] + ' ' + Style.RESET_ALL + str(todays_games_uo[
                                                                    count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
            else:
                un_confidence = round(ou_predictions_array[count][2] * 100, 1)
                print(
                    Fore.RED + home_team + Style.RESET_ALL + ' vs ' + Fore.GREEN + away_team + Style.RESET_ALL + Fore.CYAN + f" ({winner_confidence}%)" + Style.RESET_ALL + ': ' +
                    Fore.YELLOW + OU_PICK_LABELS[under_over] + ' ' + Style.RESET_ALL + str(todays_games_uo[
                                                                       count]) + Style.RESET_ALL + Fore.CYAN + f" ({un_confidence}%)" + Style.RESET_ALL)
        # Both teams' confidences go in the record, not just the winner's: the card
        # in Flask/templates/index.html compares the two to pick a lean.
        board.append(
            {
                'home_team': home_team,
                'away_team': away_team,
                'home_confidence': round(ml_predictions_array[count][1] * 100, 1),
                'away_confidence': round(ml_predictions_array[count][0] * 100, 1),
                'ou_pick': OU_PICK_LABELS[under_over],
                'ou_value': todays_games_uo[count],
                'ou_confidence': un_confidence,
            }
        )
        count += 1
    if kelly_criterion:
        print("------------Expected Value & Kelly Criterion-----------")
    else:
        print("---------------------Expected Value--------------------")
    count = 0
    for game in games:
        home_team = game[0]
        away_team = game[1]
        ev_home = ev_away = 0
        home_odds = _coerce_odds(home_team_odds[count])
        away_odds = _coerce_odds(away_team_odds[count])
        if home_odds is not None and away_odds is not None:
            ev_home = float(
                Expected_Value.expected_value(ml_predictions_array[count][1], home_odds))
            ev_away = float(
                Expected_Value.expected_value(ml_predictions_array[count][0], away_odds))
        expected_value_colors = {'home_color': Fore.GREEN if ev_home > 0 else Fore.RED,
                                 'away_color': Fore.GREEN if ev_away > 0 else Fore.RED}
        # Kelly is only ever used when -kc was passed, and it needs a real moneyline,
        # so both the arithmetic and the crash live behind this guard.
        bankroll_fraction_home = bankroll_fraction_away = ''
        home_kelly = away_kelly = None
        if kelly_criterion:
            home_kelly = _kelly_fraction(home_odds, ml_predictions_array[count][1])
            away_kelly = _kelly_fraction(away_odds, ml_predictions_array[count][0])
            bankroll_fraction_home = _bankroll_fraction(home_odds, ml_predictions_array[count][1])
            bankroll_fraction_away = _bankroll_fraction(away_odds, ml_predictions_array[count][0])

        print(home_team + ' EV: ' + expected_value_colors['home_color'] + str(ev_home) + Style.RESET_ALL + (
            bankroll_fraction_home if kelly_criterion else ''))
        print(away_team + ' EV: ' + expected_value_colors['away_color'] + str(ev_away) + Style.RESET_ALL + (
            bankroll_fraction_away if kelly_criterion else ''))

        if count < len(board):
            record = board[count]
            record['home_team_ev'] = ev_home
            record['away_team_ev'] = ev_away
            record['home_team_odds'] = home_odds
            record['away_team_odds'] = away_odds
            if kelly_criterion:
                record['home_kelly_fraction'] = home_kelly
                record['away_kelly_fraction'] = away_kelly
        count += 1

    # The one machine-readable line, printed last so it is the last thing this
    # runner says.
    emit_predictions(board)

    deinit()
