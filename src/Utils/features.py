"""The one definition of the feature contract shared by build and inference.

XGBoost and Keras are fed a positional matrix. Nothing carries a column name
past the drop step: ``df.drop(columns=[...]).to_numpy()`` throws the names away,
so the model only ever sees "column 41 is 0.504". A frame with the right
columns in the WRONG order therefore raises nothing -- the splits still fire,
the probabilities still sum to one, and every prediction is confidently wrong.

That is why the drop lists and the ordering rule live here instead of being
retyped in the trainers, in ``Create_Games`` and in the runners. One definition,
one order, checked at the boundary.
"""

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
GAME_DATE_COLUMN = "GameDate"

# Verbatim from src/Train-Models/XGBoost_Model_UO.py (DROP_COLUMNS). OU is
# deliberately NOT dropped here: it is the model input, and its position in the
# dataset -- between the away-team block and OU-Cover -- is part of the contract.
UO_LABEL_DROP_COLUMNS = [
    "index",
    "Score",
    "Home-Team-Win",
    "TEAM_NAME",
    "Date",
    "index.1",
    "TEAM_NAME.1",
    "Date.1",
    GAME_DATE_COLUMN,
    "OU-Cover",
]

# Verbatim from src/Train-Models/XGBoost_Model_ML.py (DROP_COLUMNS).
ML_DROP_COLUMNS = [
    "index",
    "Score",
    "Home-Team-Win",
    "TEAM_NAME",
    "Date",
    "index.1",
    "TEAM_NAME.1",
    "Date.1",
    GAME_DATE_COLUMN,
    "OU-Cover",
    "OU",
]

# src/Train-Models/NN_Model_UO.py was previously recorded here as a discrepancy:
# it dropped "OU" and then re-added it via ``X["OU"] = ou_values`` after the drop,
# so the NN totals model received OU as the LAST column while the XGBoost totals
# model received it at its dataset position. That made two totals models disagree
# about the order of one dataset.
#
# That is now fixed at the source. NN_Model_UO no longer drops "OU" and no longer
# re-appends it, so it matches UO_LABEL_DROP_COLUMNS exactly and both totals
# models receive OU in the same position.
#
# CONSEQUENCE: the committed artifact
# Models/NN_Models/Trained-Model-OU-49.2_*.keras was trained against the OLD
# trainer, which fed OU last. It has 107 inputs either way, so nothing raised --
# but its column order is now wrong and it must be RETRAINED before it is used.
# Do not silently point NN_Runner at it expecting correct totals.
NN_UO_DROP_COLUMNS = [
    "index",
    "Score",
    "Home-Team-Win",
    "TEAM_NAME",
    "Date",
    "index.1",
    "TEAM_NAME.1",
    "Date.1",
    GAME_DATE_COLUMN,
    "OU-Cover",
]

# src/Train-Models/NN_Model_ML.py drops these two in the opposite order to
# ML_DROP_COLUMNS. Order in the deny-list is irrelevant to the result -- the
# surviving columns keep their dataset order either way -- so it is noted, not
# normalised.
NN_ML_DROP_COLUMNS = [
    "index",
    "Score",
    "Home-Team-Win",
    "TEAM_NAME",
    "Date",
    "index.1",
    "TEAM_NAME.1",
    "Date.1",
    GAME_DATE_COLUMN,
    "OU",
    "OU-Cover",
]


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
