"""The one definition of the feature contract shared by build and inference.

XGBoost and Keras are fed a positional matrix. Nothing carries a column name
past the drop step: ``df.drop(columns=[...]).to_numpy()`` throws the names away,
so the model only ever sees "column 41 is 0.504". A frame with the right
columns in the WRONG order therefore raises nothing -- the splits still fire,
the probabilities still sum to one, and every prediction is confidently wrong.

That is why the drop lists here are computed from
``training_common.drop_columns_for`` instead of being retyped, and why the
ordering rule lives here rather than in the trainers and runners. One
definition, one order, checked at the boundary. A copy that is "mostly right" is
worse than one that is obviously wrong, because it does not look like something
to re-check.
"""

import sys
from pathlib import Path

# The drop lists below are the trainers' own return values, not copies of them.
# src/Train-Models has a hyphen and can never be a package, so it goes on
# sys.path and training_common imports under its plain top-level name. Importing
# features must not depend on the caller having done that already: the runners
# import this module before their own path setup.
_TRAIN_MODELS_DIR = Path(__file__).resolve().parents[1] / "Train-Models"
if str(_TRAIN_MODELS_DIR) not in sys.path:
    sys.path.insert(0, str(_TRAIN_MODELS_DIR))

from training_common import (  # noqa: E402
    GAME_DATE_COLUMN,
    MONEYLINE_TARGET,
    TOTALS_TARGET,
    drop_columns_for,
)

# Columns that identify a team rather than describe its play. ``Create_Games``
# strips TEAM_ID and TEAM_ID.1 before writing the dataset and keeps TEAM_NAME /
# TEAM_NAME.1, which the trainers then drop. The inference path only has the
# names, never the ids, so it strips both suffixes.
TEAM_FEATURE_DROP = ("TEAM_ID", "TEAM_NAME")

# The real game date, written by ``Create_Games`` alongside the snapshot Date.
#
# It exists because the two stopped being the same column: the leak-free join
# makes ``Date`` the as-of SNAPSHOT date (strictly before the game), while
# ``Data/OddsData.sqlite`` stores the game date. A positional model has no use
# for either -- they are keys, not measurements -- so every drop list below
# removes GameDate, exactly as it removes Date and Date.1. Leaving it in would
# make the moneyline feature count 107 and the totals count 108, and the
# committed boosters in Models/ were trained on 106 and 107.
#
# Do NOT also add "GameDate.1": no step in the build produces a suffixed copy,
# so the entry would be dead weight in a list that is read as the contract.
# ``GAME_DATE_COLUMN`` is re-exported from training_common so there is one
# spelling of the column name.

# The four names are kept because the runners and tests refer to them by name.
# Every one is computed; none is a literal list. OU stays in the totals input --
# it is the feature, and its dataset position is part of the contract -- and the
# moneyline list drops it.
UO_LABEL_DROP_COLUMNS = drop_columns_for(TOTALS_TARGET)
ML_DROP_COLUMNS = drop_columns_for(MONEYLINE_TARGET)
NN_UO_DROP_COLUMNS = drop_columns_for(TOTALS_TARGET)
NN_ML_DROP_COLUMNS = drop_columns_for(MONEYLINE_TARGET)

# The NN moneyline list used to be written with "OU" before "OU-Cover", and a
# comment asserted that deny-list order was irrelevant. It is: drop() removes by
# membership, so the surviving columns keep their dataset order either way. The
# order difference is now gone at the source as well -- the list is the trainer
# function's own, not a transcription -- and Tests/test_drop_columns.py pins the
# invariance rather than trusting the comment.


def expected_feature_order(dataset_columns, drop_columns):
    """The ordered feature names a model is fed.

    Mirrors ``df.drop(columns=drop_columns, errors="ignore")``: absent entries
    are ignored, and every surviving column keeps its position in
    ``dataset_columns``. The dataset order is the contract.
    """
    return [column for column in dataset_columns if column not in set(drop_columns)]


def describe_order_mismatch(expected, actual, limit=5):
    """Human-readable list of the first positions where two orders disagree."""
    mismatches = []
    for position, (want, got) in enumerate(zip(expected, actual)):
        if want != got:
            mismatches.append(f"position {position}: expected {want!r}, got {got!r}")
        if len(mismatches) >= limit:
            mismatches.append("...")
            break
    return mismatches


def validate_feature_frame(frame, expected, context="inference frame"):
    """Return True when ``frame``'s columns match ``expected`` exactly.

    Raises ``ValueError`` naming what went wrong otherwise. Missing and extra
    columns are reported first because they are the common cause; when the two
    sets agree but the order does not, that is reported as a MISORDERING, which
    is the failure that would otherwise pass silently and predict wrong.
    """
    actual = list(frame.columns)

    missing = [column for column in expected if column not in actual]
    extra = [column for column in actual if column not in expected]

    if missing or extra:
        details = []
        if missing:
            details.append(
                f"missing {len(missing)}: {', '.join(map(str, missing[:10]))}"
            )
        if extra:
            details.append(f"extra {len(extra)}: {', '.join(map(str, extra[:10]))}")
        raise ValueError(
            f"{context} does not match the trained feature contract "
            f"({len(actual)} columns, {len(expected)} expected). " + "; ".join(details)
        )

    if actual != expected:
        mismatches = describe_order_mismatch(expected, actual)
        raise ValueError(
            f"{context} has the right columns in the WRONG order "
            f"({len(actual)} columns, {len(expected)} expected) -- a positional "
            f"model would score these against the wrong splits and predict "
            f"confidently wrong values without error. " + "; ".join(mismatches)
        )

    return True
