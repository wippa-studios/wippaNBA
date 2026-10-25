"""Settings every trainer in this directory needs, defined once.

`src/Train-Models` contains a HYPHEN, so it is not an importable package and can
never be imported as one. The scripts are executed from inside this directory --
`cd src/Train-Models && python -m XGBoost_Model_ML`, or
`python src/Train-Models/XGBoost_Model_ML.py` from the repo root -- and in both
cases this directory is `sys.path[0]`. A sibling module is therefore importable
under its plain top-level name (`from training_common import ...`).

The three invocation styles put different directories on sys.path, and the one
thing they do NOT agree on is whether the repository ROOT is reachable -- it is
not, when the script is run from inside this directory. So this module puts the
root there itself, once, below. That is what lets every module here reach
`src.Utils.*` regardless of how it was started; the entry points keep their own
insertion of this directory because the sibling imports still need it.

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

DEFAULT_DATASET, STABLE_SORT_KIND, DEFAULT_TRIALS, DEFAULT_FOLD_ROWS and the
split/selection helpers are the same story at every other level: each was defined
in two to eight places, and the copies had drifted or were individually wrong.

`format_param` rounds and dot-escapes a hyperparameter value for the model
filename. The XGBoost and NN trainers use it; the logistic regression trainers
never name a saved model, so they do not.
"""

import contextlib
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"

# The repository root goes on sys.path here, once, so that any module in this
# directory can reach ``src.Utils.*``.
#
# This directory is executed three different ways -- ``python -m
# src.Train-Models.X`` from the root, ``python src/Train-Models/X.py`` from the
# root, and ``cd src/Train-Models && python -m X`` -- and the three put different
# directories on sys.path. The first two have the root (so ``src`` resolves); the
# third does not. Every entry point already worked around that by inserting its
# own directory, which is the opposite fix: it makes the SIBLING modules resolve
# and leaves the repo-root modules unreachable. One insertion, of the root, covers
# all three, and the entry points keep their own insertion because the sibling
# modules still need it.
#
# `src/Train-Models` has a HYPHEN and can never be a package, so this asymmetry is
# permanent; renaming it is tracked as follow-up work.
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# The one and only dataset table any entry point reads by default. It used to be
# spelled out in each of the six trainers and both runners, and the six copies
# had already drifted: the two logistic regression trainers pointed at
# `dataset_2012-26_new` and the two neural network trainers at
# `dataset_2012-24_new`. Neither name exists in Data/dataset.sqlite, so the first
# pair died with sqlite3.OperationalError before any guard could run, and the
# second pair silently pointed at a same-day-join build that carries the target
# inside its features. There is exactly one leak-free table in the file and this
# constant names it. See resolve_dataset_table for the guard that replaces the
# silence.
DEFAULT_DATASET = "dataset_2012-26"

# Tables that are known to carry same-day-join leakage. Listing them lets
# resolve_dataset_table refuse one by name instead of quietly training on the
# answer, which is what the neural network trainers used to do.
LEAKED_DATASET_TABLES = (
    "dataset_2012-23",
    "dataset_2012-24",
    "dataset_2012-24_new",
)

DATE_COLUMN = "Date"

# pandas' sort_values default is quicksort, which is NOT stable: rows that share
# a `Date` can come out in an order that depends on nothing reproducible. Rows
# sharing a snapshot date are the same day's slate, they are not independent
# observations, and letting them straddle a train/test boundary makes the split
# unreproducible from the stored table alone. scripts/evaluate.py already sorts
# with kind="mergesort" for exactly this reason; these three were the outliers.
STABLE_SORT_KIND = "mergesort"

# One NBA regular season is 1230 games, and the model is deployed on a rolling
# 62-day window. A validation fold much wider than a season measures regimes the
# model is never used in, which is why the fold geometry is configurable and
# capped at one season by default. See walk_forward_folds.
DEFAULT_FOLD_ROWS = 1230

# HOW MANY CONFIGS THE SEARCH MAY TRY. Fifteen, not the hundred the XGBoost
# trainers used to default to, and not the fifty the logistic ones and the
# networks used to default to.
#
# WHY, REPRODUCIBLE. On this dataset, over 12 sampled configurations, the
# standard deviation of the 5-fold CV score ACROSS configurations is 0.0046
# nats, while the entire true range of HELD-OUT log loss across the same twelve
# configurations is 0.0009 nats. The search was ranking noise, by a factor of
# about five. The expected distance from the mean CV score to the best of N is
# 0.0075 nats at N=15 and 0.0086 nats at N=100 -- so a hundred trials buy
# 0.0011 nats. ``validation.selection_bias_estimate`` agrees in standard-error
# units: 2.50 SE at 100 trials against 1.74 SE at 15, on 1,735 held-out rows.
#
# (Issue #7 quoted 8.2x-8.6x; that was two independent measurements of the same
# ratio, and the figure above is a third with a different and smaller sample of
# configurations. The order of magnitude is the same and so is the conclusion.)
#
# Ten to fifteen trials is the entire signal. ROADMAP.md lists "more
# hyperparameter search without selection correction" as an explicit non-goal for
# exactly this reason.
DEFAULT_TRIALS = 15

# The two prediction targets and how many classes each one carries. The
# moneyline is home/away, a binary label; the totals label is under/over/push,
# three classes. Push is kept as its own class on purpose: it is a void bet, not
# a side, and the decision layer must not fold it into over. The reasoning, the
# measured class counts and the one measurement that would reopen the choice are
# in docs/totals_target.md.
MONEYLINE_TARGET = "Home-Team-Win"
TOTALS_TARGET = "OU-Cover"
MONEYLINE_NUM_CLASSES = 2
TOTALS_NUM_CLASSES = 3

# The totals head's class index -> what the runners may call it. Class 2 is a
# push, a void bet, and deliberately has no side. Both runners share this tuple
# so neither can reintroduce the old "not under, therefore over" mapping.
OU_PICK_LABELS = ("UNDER", "OVER", "PUSH")

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


def available_dataset_tables():
    """Every table in Data/dataset.sqlite, sorted.

    Returns an empty list when the database is absent, so the caller reports a
    missing file rather than a missing table and the two are not confused.
    """
    if not DATASET_DB.exists():
        return []
    # contextlib.closing closes the handle. `with sqlite3.connect(...)` does
    # not: that context manager wraps a transaction, so it guarantees the
    # commit and guarantees nothing about closing. These are read-only
    # connections, so there is no transaction to commit and closing alone is
    # the whole fix. The handle would otherwise survive until the garbage
    # collector happened to run, which is how it went unnoticed - a leaked
    # handle is invisible until a long run exhausts the file-descriptor
    # limit. The Process-Data modules already use this idiom throughout.
    with contextlib.closing(sqlite3.connect(f"file:{DATASET_DB}?mode=ro", uri=True)) as con:
        return sorted(
            row[0]
            for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        )


def resolve_dataset_table(name):
    """Check ``name`` against the database and return it, or raise with the list.

    Replaces the two failure modes the old per-entry-point literals had:

      * a table that does not exist raised ``sqlite3.OperationalError`` from
        inside pandas, which names neither the file it looked in nor the tables
        that were available;
      * a table that DOES exist but was built by the same-day join trained on its
        own answer with no warning at all.

    So this refuses both, and the message says what to pass instead.
    """
    available = available_dataset_tables()
    if not available:
        raise FileNotFoundError(
            f"No dataset database at {DATASET_DB}. Build it first with "
            "`python -m src.Process-Data.Create_Games`."
        )
    if name not in available:
        raise LookupError(
            f"No table named {name!r} in {DATASET_DB.name}. "
            f"Available tables: {', '.join(available)}. "
            f"Pass --dataset with one of those, or omit it to use "
            f"{DEFAULT_DATASET!r}."
        )
    if name in LEAKED_DATASET_TABLES:
        raise LookupError(
            f"{name!r} in {DATASET_DB.name} is a same-day-join build: its stats "
            "snapshot includes the game being predicted, so its features carry "
            "the label. Use "
            f"{DEFAULT_DATASET!r}, or rebuild this table with "
            "`python -m src.Process-Data.Create_Games`."
        )
    return name


def load_dataset(dataset_name=DEFAULT_DATASET):
    """Read one dataset table, with the table name checked first.

    The connection is opened read-only (``mode=ro``) because nothing in a
    training run ever needs to write, and a trainer holding a read-write
    handle on the shipped database is one stray CREATE away from corrupting it.
    """
    table = resolve_dataset_table(dataset_name)
    with contextlib.closing(sqlite3.connect(f"file:{DATASET_DB}?mode=ro", uri=True)) as con:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', con)


def impute_split(train, *others):
    """Median-impute ``train`` and every held-out frame in ``others``.

    The median is computed on ``train`` alone and reused for the rest, so no
    held-out statistic reaches training. It is computed once even for several
    held-out frames, and the first return value is always the TRAINING frame, so
    ``train_filled, *rest = impute_split(train, test)`` cannot be unpacked the
    wrong way round.

    Fitting the imputer on the joined frame lets held-out statistics reach
    training. It is latent on the shipped dataset -- there are no NaN cells in
    the feature columns -- but 14 of the 106 columns would move their fill value
    by more than 1% if the medians were computed over all rows, and a latent leak
    is still a leak. Same shape as scripts/evaluate.py's impute_fit_transform.

    Returns one frame per argument, in the order given. The medians are NOT
    returned, so a caller cannot reuse them somewhere it did not split.
    """
    frames = (train,) + tuple(others)
    cleaned = [frame.replace([np.inf, -np.inf], np.nan) for frame in frames]
    medians = cleaned[0].median(numeric_only=True)
    return tuple(frame.fillna(medians).fillna(0) for frame in cleaned)


def resolve_fold_rows(n_rows, n_splits, fold_rows=None):
    """Validation-fold width in rows, capped at one season by default.

    ``TimeSeriesSplit(n_splits=5)`` on this dataset made every validation block
    2,603 rows -- measured: 716, 720, 990, 679 and 728 days -- and the five
    blocks together ran from 2014-05-10 to 2024-11-08, more than ten years,
    against a 62-day deployment window. Four folds in five measured regimes the
    model will never be used in, and at C=1.0 the per-fold log loss standard
    deviation was already 0.0044 before any hyperparameter search happened.

    Passing ``fold_rows`` uses it as given. Passing None (the default) takes the
    smaller of "an even share of the data" and one regular season, which is the
    widest window whose length the model is ever deployed over.
    """
    if n_splits < 1:
        raise ValueError(f"n_splits must be at least 1, got {n_splits}.")
    if fold_rows is not None:
        width = int(fold_rows)
        if width < 1:
            raise ValueError(f"fold_rows must be a positive row count, got {fold_rows!r}.")
        return width
    even_share = n_rows // (n_splits + 1)
    return max(1, min(even_share, DEFAULT_FOLD_ROWS))


def snap_cut_to_date_boundary(dates, position):
    """Move a row-position cut FORWARD until it sits between two dates.

    A chronological split cuts at a row position, and the dataset holds whole
    slates -- mean 6.45 rows share one snapshot date, the largest group is 40. A
    cut that lands inside such a group puts part of one day's games in training
    and the rest in the test set. They played on the same night, against the same
    injury report and the same slate, so they are not independent observations and
    the split is not reproducible from a row count alone.

    The cut only ever moves FORWARD, never backward: moving it back would put
    later rows into training, which is a leak of a far worse kind. The cost is
    that the test set is up to (largest group - 1) rows smaller than asked for,
    which is the honest price and is reported rather than hidden.

    Returns ``position`` unchanged when it is already at a boundary or already at
    either end of the frame.
    """
    values = pd.Series(dates).to_numpy()
    n_rows = len(values)
    if position <= 0 or position >= n_rows:
        return int(position)
    position = int(position)
    while position < n_rows and values[position - 1] == values[position]:
        position += 1
    return position


def walk_forward_folds(n_rows, n_splits, fold_rows=None, min_train_rows=None, dates=None):
    """Positional ``(train_idx, val_idx)`` walk-forward folds over sorted rows.

    Same shape as ``sklearn.model_selection.TimeSeriesSplit`` -- expanding
    training window, fixed-width validation blocks at the tail, nothing later
    than the rows it is scored against -- but the validation width is
    ``resolve_fold_rows`` rather than ``n_rows // (n_splits + 1)``, so a fold can
    be sized to a deployment window instead of a tenth of the history.

    Passing ``dates`` snaps each validation block's start to a date-group
    boundary (see ``snap_cut_to_date_boundary``); the block then ends one group
    later, so its width varies by at most the largest group size. Without
    ``dates`` the blocks are exactly ``fold_rows`` wide.

    ``purged_walk_forward_split`` in src/Utils/validation.py is deliberately NOT
    used here. It was measured on this dataset and is a no-op: ``embargo_pct=0.01``
    over the 4,522-day training span resolves to a 45-day embargo, and swapping
    ``TimeSeriesSplit`` for the purged split moved mean CV log loss by +0.0006 to
    +0.0016 across three configurations. It is a fine module; it is not the fix
    for fold geometry, and swapping it in would look like a fix while changing
    nothing.
    """
    width = resolve_fold_rows(n_rows, n_splits, fold_rows)
    first_val_start = n_rows - n_splits * width
    if first_val_start < 1:
        raise ValueError(
            f"{n_rows} rows cannot supply {n_splits} validation folds of {width} "
            "rows each with a training set in front of them. Reduce --splits, "
            "reduce --fold-rows, or supply more history."
        )
    floor = 1 if min_train_rows is None else int(min_train_rows)
    folds = []
    for split in range(n_splits):
        start = first_val_start + split * width
        stop = start + width
        if dates is not None:
            start = snap_cut_to_date_boundary(dates, start)
            stop = snap_cut_to_date_boundary(dates, stop)
        stop = min(stop, n_rows)
        if start >= stop:
            raise ValueError(
                f"fold {split}: snapping its boundaries to date-group edges left an "
                f"EMPTY validation block at row {start}. Use a wider --fold-rows; "
                "a validation fold narrower than the largest same-date group cannot "
                "be placed between two dates."
            )
        folds.append((np.arange(start), np.arange(start, stop)))
    if floor > 1 and any(len(train) < floor for train, _ in folds):
        raise ValueError(
            f"the earliest fold would train on fewer than {floor} rows. Lower "
            "--min-train-rows or reduce --splits."
        )
    return folds


def recency_weighted_fold_mean(losses, half_life_folds=1.0):
    """Combine per-fold losses, weighting recent folds above old ones.

    WHY THIS REPLACES THE FLAT MEAN. The flat mean of five folds spread over ten
    years weights a 2014 fold exactly as heavily as a 2024 one, and this model is
    deployed on the most recent 62 days. A configuration that fits 2014 well and
    2024 badly wins the flat mean; the reverse is the configuration worth having.

    Weights are geometric in fold age, so the LAST fold carries the most weight
    by construction and the score stays a weighted average of the same per-fold
    losses -- nothing new is computed, only re-weighted. With ``half_life_folds``
    at its default of 1.0 and five folds the weights are
    [0.0625, 0.125, 0.25, 0.5, 1.0] and the newest fold carries 52%.

    Both directions are tested in ``Tests/test_training_layer.py``: with quality
    improving over the folds the weighted mean is BETTER than the flat one, and
    with quality decaying it is WORSE. That is the property; a re-scaling that
    happened to flatter one direction would not be.

    Returns (weighted_mean, per_fold_losses, weights). The per-fold values are
    returned so the caller can print the last fold's loss beside it -- the two
    numbers disagreeing is the signal, and hiding one of them is what let the flat
    mean go unnoticed.
    """
    values = [float(value) for value in losses]
    if not values:
        return float("inf"), [], []
    half_life = float(half_life_folds)
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError(f"half_life_folds must be a positive number, got {half_life_folds!r}.")
    ages = np.arange(len(values) - 1, -1, -1, dtype=float)
    weights = 0.5 ** (ages / half_life)
    total = float(weights.sum())
    weights = weights / total
    weighted = float(np.dot(np.asarray(values), weights))
    return weighted, values, weights.tolist()


def calibration_choices(argv=None):
    """The ``--calibration`` choices, widened when the opt-in flag is present.

    Isotonic is NOT offered by default: on this dataset it measured +0.0035 nats,
    worse than no calibration at all, because it fits one free parameter per
    training point against a 1,562-row segment. ``resolve_calibration_method``
    refuses it there twice over -- once for the missing opt-in and once for the
    segment size -- but a user should not have to read a traceback to find out.

    argparse fixes ``choices`` before it looks at any other flag, so a boolean
    opt-in cannot widen a NEIGHBOURING flag's choices from inside the parser.
    This pre-scans ``argv`` for the opt-in, which is the only way to express
    "off unless you also say this" as a pair of flags. Passing ``argv``
    explicitly is what makes it testable.
    """
    if argv is None:
        argv = sys.argv[1:]
    if "--allow-isotonic" in list(argv):
        return ("sigmoid", "isotonic", "none")
    return ("sigmoid", "none")


def report_search_selection_risk(n_configs_tried, metric_name, n_test_rows, standard_error):
    """Print src/Utils/validation.report_selection_risk, or say why it could not.

    The correction has existed since the validation layer landed and was never
    called from any trainer, so every printed "Best val log loss" was a
    winner's-curse statistic presented as a measurement. It is now printed next
    to that line by all six entry points.

    Wrapped rather than called directly for the same reason report_closing_line
    is wrapped: a reporting aid must never kill a training run. If the import
    cannot be made -- an unusual invocation style, or the validation layer
    deleted -- it says so on one line and returns None. A missing correction is
    visible; a crash at the end of a two-hour search is not.
    """
    try:
        from src.Utils.validation import report_selection_risk
    except Exception as exc:  # pragma: no cover - import-failure path
        print(
            f"Selection risk not reported: could not load src/Utils/validation.py "
            f"({exc or type(exc).__name__})."
        )
        return None
    return report_selection_risk(
        n_configs_tried, metric_name, n_test_rows, standard_error
    )


def format_param(value, precision=3):
    formatted = f"{value:.{precision}f}" if isinstance(value, float) else str(value)
    return formatted.replace(".", "p")
