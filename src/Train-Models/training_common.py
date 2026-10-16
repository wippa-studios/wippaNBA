"""Settings every trainer in this directory needs, defined once.

`src/Train-Models` contains a HYPHEN, so it is not an importable package and can
never be imported as one. The scripts are executed from inside this directory --
`cd src/Train-Models && python -m XGBoost_Model_ML`, or
`python src/Train-Models/XGBoost_Model_ML.py` from the repo root -- and in both
cases this directory is `sys.path[0]`. A sibling module is therefore importable
under its plain top-level name (`from training_common import ...`) with no
`sys.path` manipulation of any kind.

What belongs here is what all six entry points agree on and used to each spell
out again: where the data lives, which column each target lives in, how many
classes each target carries, and how a DROP_COLUMNS list is derived from those
two facts.

DROP_COLUMNS was previously written out six times, and the six copies had
already drifted. It is load-bearing: it IS the column order the models are fed,
so dropping one column that another trainer keeps does not raise, it silently
permutes every input after it. That is exactly what had happened -- the two
totals trainers disagreed about whether the closing total `OU` is a feature, so
the runner fed them a column order neither had trained on.

`format_param` rounds and dot-escapes a hyperparameter value for the model
filename. The XGBoost and NN trainers use it; the logistic regression trainers
never name a saved model, so they do not.
"""

import sqlite3
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"

DATE_COLUMN = "Date"

# The two prediction targets and how many classes each one carries. The
# moneyline is home/away, a binary label; the totals label is under/over/push,
# three classes.
MONEYLINE_TARGET = "Home-Team-Win"
TOTALS_TARGET = "OU-Cover"
MONEYLINE_NUM_CLASSES = 2
TOTALS_NUM_CLASSES = 3

# The closing total. It is known before tip-off, so it is a legitimate feature
# for the totals trainers and never wanted by the moneyline trainers.
CLOSING_TOTAL_COLUMN = "OU"

# The real game date, written by src/Process-Data/Create_Games.py next to the
# snapshot Date. It is a join key and never a feature: see drop_columns_for.
# The predictors sort on DATE_COLUMN, not on this one.
GAME_DATE_COLUMN = "GameDate"


def num_classes_for(target_column):
    """The number of classes ``target_column`` carries."""
    if target_column == MONEYLINE_TARGET:
        return MONEYLINE_NUM_CLASSES
    return TOTALS_NUM_CLASSES


def drop_columns_for(target_column, include_team_ids=False):
    """The DROP_COLUMNS list for one trainer.

    Three groups go, for every trainer: the labels and the score, the raw
    team-name and date keys that are joined in rather than learned from, and the
    duplicate index columns pandas emits when the dataset was assembled. Both
    labels are always dropped, so this list never depends on which of the two
    targets the caller is training.

    ``include_team_ids`` additionally drops TEAM_ID / TEAM_ID.1. Only the two
    logistic regression scripts ask for it; the XGBoost and NN trainers leave
    the ids in. That asymmetry predates this module and is reproduced verbatim
    rather than tidied away here. On the datasets that ship with the repository
    those two columns are absent, so `drop(columns=..., errors="ignore")` and
    the membership test in the XGBoost trainers both make it a no-op today.

    The closing total is dropped for a moneyline trainer and kept for a totals
    trainer. Every totals model must agree on that, because the runner feeds
    them all the same column order and an input width none of them trained on
    is worse than a worse model.

    ``GAME_DATE_COLUMN`` is dropped here for the same reason Date is. The
    leak-free join in Create_Games makes ``Date`` the as-of SNAPSHOT date --
    strictly before the game -- while ``Data/OddsData.sqlite`` stores the game
    date, so the two are one day apart and no longer interchangeable. The
    trainers still sort by ``Date`` (it is the honest as-of marker) and
    training_reporting joins to the odds on ``GameDate`` instead; neither of
    those makes the column a measurement, so it is dropped. Datasets written
    before the column existed simply do not have it and
    ``drop(columns=..., errors="ignore")`` makes the entry a no-op.
    """
    columns = ["index", "Score", MONEYLINE_TARGET, "TEAM_NAME"]
    if include_team_ids:
        columns.append("TEAM_ID")
    columns += ["Date", "index.1", "TEAM_NAME.1"]
    if include_team_ids:
        columns.append("TEAM_ID.1")
    columns.append("Date.1")
    columns.append(GAME_DATE_COLUMN)
    columns.append(TOTALS_TARGET)
    if target_column == MONEYLINE_TARGET:
        columns.append(CLOSING_TOTAL_COLUMN)
    return columns


def load_dataset(dataset_name):
    with sqlite3.connect(DATASET_DB) as con:
        return pd.read_sql_query(f'SELECT * FROM "{dataset_name}"', con)


def format_param(value, precision=3):
    formatted = f"{value:.{precision}f}" if isinstance(value, float) else str(value)
    return formatted.replace(".", "p")
