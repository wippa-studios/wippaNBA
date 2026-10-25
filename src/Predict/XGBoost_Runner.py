import json
import math
import re
import sqlite3
import sys
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from colorama import Fore, Style, init, deinit
from src.Utils import Expected_Value
from src.Utils import Kelly_Criterion as kc

try:
    from src.Utils import features
except ImportError:
    features = None


init()

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "Models" / "XGBoost_Models"
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"

# The one dataset table inference reads, shared with the trainers rather than
# spelled out here. src/Train-Models has a HYPHEN and can never be imported as a
# package, so its shared modules only resolve when the directory is on sys.path
# -- the same convention the six training scripts use. This used to hard-code
# "dataset_2012-26" here while NN_Runner hard-coded a table that does not exist,
# and neither trainer's default matched the other's. One name, one source.
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

ACCURACY_PATTERN = re.compile(r"XGBoost_(\d+(?:\.\d+)?)%_")

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
# ---------------------------------------------------------------------------
PREDICTIONS_SENTINEL = "__NBA_PREDICTIONS_JSON__ "
PREDICTIONS_SCHEMA_VERSION = 1

xgb_ml = None
xgb_uo = None
xgb_ml_calibrator = None
xgb_uo_calibrator = None


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


def _select_model_path(kind):
    candidates = list(MODEL_DIR.glob(f"*{kind}*.json"))
    if not candidates:
        raise FileNotFoundError(f"No XGBoost {kind} model found in {MODEL_DIR}")

    def score(path):
        # The accuracy is parsed out of the FILE NAME, not read from the artifact:
        # a booster stores no accuracy, and a fresh clone gives every file the same
        # mtime, so the number below decides the winner. It is unverified.
        match = ACCURACY_PATTERN.search(path.name)
        accuracy = float(match.group(1)) if match else 0.0
        return (path.stat().st_mtime, accuracy)

    return max(candidates, key=score)


def _load_calibrator(model_path):
    calibration_path = model_path.with_name(f"{model_path.stem}_calibration.pkl")
    if not calibration_path.exists():
        print(
            Fore.YELLOW
            + f"Warning: no calibration sidecar next to {model_path.name}; "
            f"predictions from this model are UNCALIBRATED."
            + Style.RESET_ALL
        )
        return None
    try:
        return joblib.load(calibration_path)
    except Exception:
        return None


def _load_models():
    global xgb_ml, xgb_uo, xgb_ml_calibrator, xgb_uo_calibrator
    if xgb_ml is None:
        ml_path = _select_model_path("ML")
        xgb_ml = xgb.Booster()
        xgb_ml.load_model(str(ml_path))
        xgb_ml_calibrator = _load_calibrator(ml_path)
    if xgb_uo is None:
        uo_path = _select_model_path("UO")
        xgb_uo = xgb.Booster()
        xgb_uo.load_model(str(uo_path))
        xgb_uo_calibrator = _load_calibrator(uo_path)


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
    """The shared totals drop list, whichever of its two published names is present."""
    if features is None:
        return None
    for name in ("UO_DROP_COLUMNS", "UO_LABEL_DROP_COLUMNS"):
        drop_columns = getattr(features, name, None)
        if drop_columns:
            return drop_columns
    return None


def _expected_uo_columns(frame_ml):
    """Column order the totals model was trained on: the moneyline order with the total line in place.

    The totals trainer keeps OU (the pre-game total is known before tip-off), so the
    totals model sees one more column than the moneyline model and OU belongs in the
    middle of the frame. Appending it would keep the count right and silently feed
    every trailing feature to the wrong position, so build the order explicitly.
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
    return " Fraction of Bankroll: " + str(fraction) + "%"


def _predict_probs(model, data, calibrator=None):
    if calibrator is not None:
        return calibrator.predict_proba(data)
    return model.predict(xgb.DMatrix(data))


def _format_game_line(home_team, away_team, winner_is_home, winner_confidence, under_over, ou_value, ou_confidence):
    winner_team = home_team if winner_is_home else away_team
    loser_team = away_team if winner_is_home else home_team
    winner_color = Fore.GREEN if winner_is_home else Fore.RED
    loser_color = Fore.RED if winner_is_home else Fore.GREEN
    # The totals head has three classes. Class 2 is a push: the total landed
    # exactly on the line, so the bet is void. It is never a side. Mapping it to
    # OVER -- which the runner used to do by treating "not UNDER" as OVER --
    # both invents a bet and reports the push probability as that bet's edge.
    # See docs/totals_target.md.
    ou_label = OU_PICK_LABELS[under_over]
    ou_color = (Fore.MAGENTA, Fore.BLUE, Fore.YELLOW)[under_over]
    return (
        f"{winner_color}{winner_team}{Style.RESET_ALL}"
        f"{Fore.CYAN} ({winner_confidence}%)"
        f"{Style.RESET_ALL} vs {loser_color}{loser_team}{Style.RESET_ALL}: "
        f"{ou_color}{ou_label} {Style.RESET_ALL}{ou_value}"
        f"{Style.RESET_ALL}{Fore.CYAN} ({ou_confidence}%)"
        f"{Style.RESET_ALL}"
    )


def _print_expected_value(
    games,
    ml_predictions_array,
    home_team_odds,
    away_team_odds,
    kelly_criterion,
    board=None,
):
    """Print the expected-value block and, when `board` is given, fill it in.

    The EV/odds/Kelly numbers are taken from the same locals that are printed, so
    the machine-readable line can never drift from the human-readable one. `board`
    is the list of per-game records built by xgb_runner; one entry per game, in
    the same order as `games`.
    """
    if kelly_criterion:
        print("------------Expected Value & Kelly Criterion-----------")
    else:
        print("---------------------Expected Value--------------------")
    for idx, game in enumerate(games):
        home_team, away_team = game
        ev_home = ev_away = 0
        home_odds = _coerce_odds(home_team_odds[idx])
        away_odds = _coerce_odds(away_team_odds[idx])
        if home_odds is not None and away_odds is not None:
            ev_home = float(
                Expected_Value.expected_value(
                    ml_predictions_array[idx][1],
                    home_odds,
                )
            )
            ev_away = float(
                Expected_Value.expected_value(
                    ml_predictions_array[idx][0],
                    away_odds,
                )
            )
        expected_value_colors = {
            "home_color": Fore.GREEN if ev_home > 0 else Fore.RED,
            "away_color": Fore.GREEN if ev_away > 0 else Fore.RED,
        }
        # Kelly is only ever used when -kc was passed, and it needs a real moneyline,
        # so both the arithmetic and the crash live behind this guard.
        bankroll_fraction_home = bankroll_fraction_away = ""
        home_kelly = away_kelly = None
        if kelly_criterion:
            home_kelly = _kelly_fraction(home_odds, ml_predictions_array[idx][1])
            away_kelly = _kelly_fraction(away_odds, ml_predictions_array[idx][0])
            bankroll_fraction_home = _bankroll_fraction(home_odds, ml_predictions_array[idx][1])
            bankroll_fraction_away = _bankroll_fraction(away_odds, ml_predictions_array[idx][0])

        print(
            home_team
            + " EV: "
            + expected_value_colors["home_color"]
            + str(ev_home)
            + Style.RESET_ALL
            + (bankroll_fraction_home if kelly_criterion else "")
        )
        print(
            away_team
            + " EV: "
            + expected_value_colors["away_color"]
            + str(ev_away)
            + Style.RESET_ALL
            + (bankroll_fraction_away if kelly_criterion else "")
        )

        if board is not None and idx < len(board):
            record = board[idx]
            record["home_team_ev"] = ev_home
            record["away_team_ev"] = ev_away
            record["home_team_odds"] = home_odds
            record["away_team_odds"] = away_odds
            if kelly_criterion:
                record["home_kelly_fraction"] = home_kelly
                record["away_kelly_fraction"] = away_kelly


def xgb_runner(data, todays_games_uo, frame_ml, games, home_team_odds, away_team_odds, kelly_criterion):
    _load_models()

    uo_columns = _expected_uo_columns(frame_ml)
    frame_uo = frame_ml.copy()
    frame_uo["OU"] = np.asarray(todays_games_uo, dtype=float)
    # Select by name in the training order: the model was fitted on the dataset
    # column order, which puts OU in the middle rather than at the end.
    frame_uo = frame_uo[uo_columns]
    if features is not None:
        features.validate_feature_frame(frame_uo, uo_columns, "XGBoost totals inference frame")

    try:
        ml_predictions_array = _predict_probs(xgb_ml, data, xgb_ml_calibrator)
        ou_predictions_array = _predict_probs(
            xgb_uo,
            frame_uo.values.astype(float),
            xgb_uo_calibrator,
        )

        board = []
        for idx, game in enumerate(games):
            home_team, away_team = game
            winner = int(np.argmax(ml_predictions_array[idx]))
            under_over = int(np.argmax(ou_predictions_array[idx]))
            winner_confidence = round(ml_predictions_array[idx][winner] * 100, 1)
            ou_confidence = round(ou_predictions_array[idx][under_over] * 100, 1)

            print(
                _format_game_line(
                    home_team,
                    away_team,
                    winner_is_home=(winner == 1),
                    winner_confidence=winner_confidence,
                    under_over=under_over,
                    ou_value=todays_games_uo[idx],
                    ou_confidence=ou_confidence,
                )
            )

            # Both teams' confidences go in the record, not just the winner's: the
            # card in Flask/templates/index.html compares the two to pick a lean.
            board.append(
                {
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_confidence": round(ml_predictions_array[idx][1] * 100, 1),
                    "away_confidence": round(ml_predictions_array[idx][0] * 100, 1),
                    "ou_pick": OU_PICK_LABELS[under_over],
                    "ou_value": todays_games_uo[idx],
                    "ou_confidence": ou_confidence,
                }
            )

        _print_expected_value(
            games,
            ml_predictions_array,
            home_team_odds,
            away_team_odds,
            kelly_criterion,
            board=board,
        )
        # The one machine-readable line, printed last so it is the last thing this
        # runner says. Nothing below here may print.
        emit_predictions(board)
    finally:
        deinit()
