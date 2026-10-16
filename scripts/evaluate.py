"""Reproducible measurement suite for the moneyline models.

Every number this project has ever claimed about its own accuracy or its own
profitability lived in a throwaway script outside the repository. This is the
in-repo replacement: one command, one dataset table, one chronological split,
and a printed provenance line beside every figure.

    python scripts/evaluate.py                     # full suite, logistic regression
    python scripts/evaluate.py --model gbdt        # gradient boosting instead
    python scripts/evaluate.py --model none        # baselines only, no training
    python scripts/evaluate.py --skip-odds         # no odds-dependent metric

Four things this file is deliberately strict about.

1. The leakage gate runs FIRST and gates everything else. The project's original
   headline (68.9% accuracy) was target leakage: the stats snapshot named D
   contained the result of the game played on D, so W, L, GP, W_PCT, PTS and
   every *_RANK column carried the answer. A leak-free build joins against a
   snapshot that PREDATES the game, so its `Date` column is the day BEFORE the
   game the row describes, and a closing price for that game is dated strictly
   later. `run_integrity_gate` proves that here before a single accuracy is
   printed, and suppresses every predictive metric unless the verdict is PASS
   (all checks ran) or PASS_NO_ODDS (the user passed --skip-odds, so check 2
   could not run). FAIL, UNVERIFIED and UNKNOWN all print nothing but the
   evidence.

2. Nothing is reported without its provenance: table name, row counts, split
   boundary, seed, feature count, join match rate and date alignment all travel
   with the figures. A number whose method is unknown is the exact failure this
   harness exists to prevent.

3. Odds degradation never raises and never guesses. A missing database, a join
   that matches fewer than MIN_MATCHED_ODDS_ROWS rows, a sparse join or an
   unusable price prints `ROI not available: <reason>` and carries on. A missing
   ROI is fine; a wrong ROI is not.

4. Data/ is read-only, and sqlite is opened `mode=ro` throughout.

`src/Train-Models/training_reporting.py` supplies the American-odds arithmetic
and the closing-price loader, exactly as the trainers use it. That directory
contains a hyphen so it is not a package: it is loaded by file path after being
placed on sys.path, which is what makes its own plain
`from training_common import BASE_DIR` resolve. If that import ever fails the
arithmetic is replicated locally and the substitution is printed rather than
made silently.
"""

import argparse
import base64
import importlib.util
import json
import re
import sqlite3
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_DB = BASE_DIR / "Data" / "dataset.sqlite"
ODDS_DB = BASE_DIR / "Data" / "OddsData.sqlite"
TEAM_DB = BASE_DIR / "Data" / "TeamData.sqlite"
TRAINING_DIR = BASE_DIR / "src" / "Train-Models"

MONEYLINE_TARGET = "Home-Team-Win"
DATE_COLUMN = "Date"
# The real game date. The leak fix made `Date` mean the as-of SNAPSHOT date
# (game date minus one, sometimes much more across a season boundary), while the
# odds ledger stores the GAME date. Joining on `Date` matches 9 of 1722 rows;
# joining on `GameDate` matches all 1722. Never "simplify" this back to an
# offset -- the gap is not a constant.
GAME_DATE_COLUMN = "GameDate"
HOME_TEAM = "TEAM_NAME"
AWAY_TEAM = "TEAM_NAME.1"

# Mirrors training_common.drop_columns_for(MONEYLINE_TARGET, include_team_ids=True)
# so this harness trains on the feature set the trainers do. TEAM_ID was dropped
# from the dataset during the audit; the entries stay because the trainer list
# keeps them and `errors="ignore"` makes an absent column a no-op.
DROP_COLUMNS = [
    "index",
    "Score",
    MONEYLINE_TARGET,
    HOME_TEAM,
    "TEAM_ID",
    DATE_COLUMN,
    "index.1",
    AWAY_TEAM,
    "TEAM_ID.1",
    "Date.1",
    # The real game date, added when the leak fix changed `Date` to mean the
    # as-of snapshot date. It is an identifier, not a feature, and it is what the
    # odds join must use -- see GAME_DATE_COLUMN.
    GAME_DATE_COLUMN,
    "OU-Cover",
    "OU",
]

DEFAULT_DATASET = "dataset_2012-26"
DEFAULT_TEST_SIZE = 0.1
DEFAULT_SEED = 42
DEFAULT_MODEL = "logreg"

# Rows sampled for the leakage gate. Each distinct snapshot date costs one small
# sqlite query, so this is the knob that trades gate cost for confidence. Every
# count the gate reports is out of this sample, never out of the whole table.
GATE_SAMPLE_ROWS = 400

# A same-day join is the historical defect and fails the gate outright. The
# tolerance only guards the case where the sample is dominated by one alignment
# while a handful of rows still join same-day.
SAME_DAY_JOIN_TOLERANCE = 0.01

# The audit's leak detector: if a rule reading only W on date D and W on date
# D-1 beats the always-home base rate by this much, the snapshots carry results.
RULE_EDGE_TOLERANCE = 0.05

# Fewest priced test rows worth an ROI figure, matching training_reporting.
MIN_MATCHED_ODDS_ROWS = 50

# Fraction of the test set that must carry a closing line before an ROI is
# scored. The 2024-25 odds table stops one day before that season ends, so a
# literally complete join is not achievable on this data: the last sixteen test
# games have no price to bet at. Below this floor the join counts as partial and
# no figure is printed at all; at or above it, the unmatched rows are reported
# alongside the figure rather than hidden.
MIN_JOIN_COVERAGE = 0.95

EDGE_THRESHOLDS = (0.00, 0.02, 0.05, 0.10)
# Candidate alignments between a dataset row's snapshot date and the game's own
# date. A leak-free build stores the snapshot date, so the game sits one day
# later; a same-day build stores the game date itself.
ODDS_DATE_OFFSETS = (0, 1)
FLAT_STAKE = 100.0

SEASON_PATTERN = re.compile(r"^(?:odds_)?(\d{4}-\d{2})(?:_new)?$")
ODDS_TABLE_CANDIDATES = (
    "odds_{season}_new",
    "odds_{season}",
    "{season}_new",
    "{season}",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def fmt(value, spec=".4f"):
    """Format a number for printing, or `n/a` when there is not one.

    Every figure passes through here. A NaN reaching the output is a defect in
    the arithmetic, and a silent `nan` in a CI log is how such a defect hides.
    """
    if value is None:
        return "n/a"
    if isinstance(value, str):
        return value
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return format(number, spec)


def fmt_count(value):
    """An integer count for printing, or `n/a`."""
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(number):
        return "n/a"
    return str(int(number))


def open_readonly(path):
    """A read-only sqlite connection, or None when the file is absent."""
    if not path.exists():
        return None
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def rule(title):
    """A section heading. Printed, never logged: this file is the report."""
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def load_training_reporting():
    """Load src/Train-Models/training_reporting.py by file path, or None.

    The directory holds a hyphen, so it can never be imported as a package. It
    goes on sys.path for the duration so the module's own
    `from training_common import BASE_DIR` resolves the way it does when the
    trainers are executed from inside that directory.
    """
    path = TRAINING_DIR / "training_reporting.py"
    if not path.exists():
        return None
    if str(TRAINING_DIR) not in sys.path:
        sys.path.insert(0, str(TRAINING_DIR))
    spec = importlib.util.spec_from_file_location("training_reporting", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local_american_odds_profit(odds, won):
    """American-odds profit on a flat $100 stake; the fallback arithmetic.

    A winning bet returns `odds` when the price is positive and 10000/abs(odds)
    when it is negative. A losing bet loses the stake, and so does a price of
    exactly zero, which is not a real price and cannot win.
    """
    if not won:
        return -FLAT_STAKE
    odds = float(odds)
    if odds > 0:
        return odds
    if odds < 0:
        return 10000.0 / abs(odds)
    return -FLAT_STAKE


def local_implied_probability(odds):
    """Vig-free win probability implied by one American price, or None."""
    try:
        odds = float(odds)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(odds) or odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def resolve_odds_arithmetic():
    """(profit_fn, implied_fn, source_label) from the shared reporting module."""
    try:
        module = load_training_reporting()
    except Exception as exc:
        print(f"Could not import training_reporting.py ({exc}); using local arithmetic.")
        module = None
    if module is None:
        return local_american_odds_profit, local_implied_probability, "local fallback"
    return (
        module.american_odds_profit,
        module.american_odds_implied_probability,
        "training_reporting.american_odds_profit",
    )


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------


def load_dataset(table_name):
    """The dataset table as a frame, or None when it cannot be read."""
    con = open_readonly(DATASET_DB)
    if con is None:
        print(f"ERROR: {DATASET_DB} is missing.")
        return None
    try:
        return pd.read_sql_query(f'SELECT * FROM "{table_name}"', con)
    except Exception as exc:
        print(f"ERROR: could not read table {table_name} from {DATASET_DB.name}: {exc}")
        return None
    finally:
        con.close()


def load_closing_odds():
    """Closing moneyline prices, deduped to one row per game.

    One table per season is chosen with the same candidate precedence as
    src/Process-Data/Create_Games.py, so the price used here is the price that
    built the dataset. Overlapping seasons mean a naive concatenation double
    counts every matchup, so the frame is deduplicated on (Date, Home, Away) and
    both counts are reported.

    Returns (frame_or_None, raw_row_count).
    """
    con = open_readonly(ODDS_DB)
    if con is None:
        return None, 0
    frames = []
    try:
        existing = {
            row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        seasons = {m.group(1) for m in map(SEASON_PATTERN.match, existing) if m}
        for season in sorted(seasons):
            for template in ODDS_TABLE_CANDIDATES:
                table = template.format(season=season)
                if table in existing:
                    frames.append(
                        pd.read_sql_query(
                            f'SELECT Date, Home, Away, ML_Home, ML_Away FROM "{table}"', con
                        )
                    )
                    break
    finally:
        con.close()

    if not frames:
        return None, 0

    odds = pd.concat(frames, ignore_index=True)
    raw_rows = len(odds)
    odds = odds.rename(columns={"Home": HOME_TEAM, "Away": AWAY_TEAM})
    odds[DATE_COLUMN] = pd.to_datetime(odds[DATE_COLUMN], errors="coerce").dt.strftime("%Y-%m-%d")
    odds["ML_Home"] = pd.to_numeric(odds["ML_Home"], errors="coerce")
    odds["ML_Away"] = pd.to_numeric(odds["ML_Away"], errors="coerce")
    odds = odds.dropna(subset=[DATE_COLUMN, HOME_TEAM, AWAY_TEAM, "ML_Home", "ML_Away"])
    odds = odds.drop_duplicates(subset=[DATE_COLUMN, HOME_TEAM, AWAY_TEAM], keep="first")
    if odds.empty:
        return None, raw_rows
    return odds, raw_rows


class SnapshotReader:
    """Team stat snapshots from Data/TeamData.sqlite, one table per date.

    Each table holds every team's season-to-date record as of that date. Rows
    are memoised because a gate sample holds many rows sharing a date.
    """

    def __init__(self):
        self.con = open_readonly(TEAM_DB)
        self.dates = []
        if self.con is not None:
            self.dates = sorted(
                row[0]
                for row in self.con.execute("SELECT name FROM sqlite_master WHERE type='table'")
            )
        self.index = {date: position for position, date in enumerate(self.dates)}
        self.cache = {}

    def available(self):
        return self.con is not None and bool(self.dates)

    def record(self, date, team):
        """(GP, W) for one team on one snapshot date, or None when unavailable."""
        date = self.key(date)
        if date not in self.index:
            return None
        if date not in self.cache:
            try:
                frame = pd.read_sql_query(f'SELECT TEAM_NAME, GP, W FROM "{date}"', self.con)
            except Exception:
                self.cache[date] = {}
                return None
            self.cache[date] = {row[0]: (row[1], row[2]) for row in frame.itertuples(index=False)}
        return self.cache[date].get(team)

    def previous_date(self, date):
        """The newest snapshot date strictly before `date`, or None."""
        position = self.index.get(self.key(date))
        if position is None or position == 0:
            return None
        return self.dates[position - 1]

    @staticmethod
    def key(date):
        """Snapshot tables are named YYYY-MM-DD; dataset dates arrive as timestamps."""
        if isinstance(date, str):
            return date[:10]
        try:
            return pd.Timestamp(date).strftime("%Y-%m-%d")
        except (TypeError, ValueError):
            return None

    def close(self):
        if self.con is not None:
            self.con.close()


# ---------------------------------------------------------------------------
# frame preparation
# ---------------------------------------------------------------------------


def prepare_frame(df):
    """Sort by date and split features from the label.

    The sort is stable, so rows sharing a date keep their stored order and the
    split boundary is reproducible from the stored table alone.
    """
    data = df.copy()
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
    data = data.sort_values(DATE_COLUMN, kind="mergesort").reset_index(drop=True)
    y = data[MONEYLINE_TARGET].astype(int).to_numpy()
    X = data.drop(columns=DROP_COLUMNS, errors="ignore")
    return data, X, y


def chronological_split(X, y, test_size):
    """Train on the first (1 - test_size) of the sorted frame, test on the tail."""
    split_at = int(len(X) * (1.0 - test_size))
    return X.iloc[:split_at], X.iloc[split_at:], y[:split_at], y[split_at:]


def impute_fit_transform(train, test):
    """Median-impute both frames using the TRAIN medians only.

    Fitting the imputer on the whole frame would let test-set statistics reach
    training, which is the same class of defect the gate exists to catch.
    """
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    medians = train.median(numeric_only=True)
    return train.fillna(medians).fillna(0), test.fillna(medians).fillna(0)


def build_model(name, seed):
    """The requested estimator. Class weighting stays off: it costs calibration."""
    if name == "gbdt":
        return GradientBoostingClassifier(
            random_state=seed,
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
        )
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, random_state=seed),
    )


def bootstrap_metric_errors(y_test, proba, seed=42, n_bootstrap=2000):
    """Bootstrap standard errors for accuracy, log loss, Brier and AUC.

    These were the four headline numbers the harness printed as bare point
    estimates, which made them impossible to act on: a change that moved log
    loss by 0.0037 could not be told from a change that moved it by 0.0000.
    Seed noise alone reaches 0.0039 on this test window, so a point estimate
    without an interval is not a measurement.

    The interval is per-row, so it understates uncertainty for rows sharing a
    game night; a clustered variant is worth adding if runs are ever compared
    across more than a handful of configurations.
    """
    if len(proba) < 2:
        return {}
    predicted = (np.asarray(proba) >= 0.5).astype(int)
    y_array = np.asarray(y_test)
    rng = np.random.default_rng(seed)
    n_rows = len(y_array)
    draws = {"accuracy": [], "log_loss": [], "brier": [], "auc": []}
    binary = set(np.unique(y_array).tolist()) >= {0, 1}
    for _ in range(n_bootstrap):
        rows = rng.integers(0, n_rows, n_rows)
        yb, pb, predb = y_array[rows], np.asarray(proba)[rows], predicted[rows]
        draws["accuracy"].append(float((predb == yb).mean()))
        if binary and len(np.unique(yb)) >= 2:
            clipped = np.clip(pb, 1e-15, 1 - 1e-15)
            draws["log_loss"].append(
                float(-np.mean(yb * np.log(clipped) + (1 - yb) * np.log(1 - clipped)))
            )
            draws["brier"].append(float(np.mean((pb - yb) ** 2)))
            draws["auc"].append(float(roc_auc_score(yb, pb)))
        else:
            for key in ("log_loss", "brier", "auc"):
                draws[key].append(np.nan)
    out = {}
    for key, values in draws.items():
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        out[key] = float(values.std(ddof=1)) if values.size > 1 else None
    return out


def ledger_path():
    return Path(__file__).resolve().parents[1] / "runs" / "ledger.jsonl"


def append_ledger_line(record):
    """Append one JSON object per run to runs/ledger.jsonl.

    JSONL rather than SQLite because .gitignore matches *.sqlite, so a
    runs.sqlite would be silently ignored by git; and a binary ledger is not
    diffable. One line per run keeps merges trivial.

    test_predictions_zb64 is the load-bearing field: without the stored
    prediction vector a ledger records numbers but cannot recompute the PAIRED
    delta between two runs, which is the only comparison with enough resolution
    to see a real change (paired SE 0.0019 against an unpaired 0.0082).
    """
    path = ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def load_ledger_run(run_id):
    if not ledger_path().exists():
        return None
    for line in ledger_path().read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("run_id") == run_id or run_id in (record.get("run_id", ""),):
            return record
    return None


def dataset_fingerprint(table_name):
    """content_sha256 of a built table, or None when there is no manifest."""
    manifest_path = DATASET_DB.parent / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return None
    return manifest.get("tables", {}).get(table_name, {}).get("content_sha256")


def compare_to_baseline(y_test, proba, run_id):
    """Paired bootstrap of this run's log loss against a stored baseline run."""
    record = load_ledger_run(run_id)
    if record is None or not record.get("test_predictions_zb64"):
        return None
    try:
        from src.Utils.metrics import bootstrap_paired_delta
    except ImportError:
        return None
    stored = np.frombuffer(
        zlib.decompress(base64.b64decode(record["test_predictions_zb64"])), dtype=np.float32
    )
    if stored.size != len(proba):
        return {
            "error": (
                f"baseline run {run_id} has {stored.size} stored predictions but this "
                f"run scored {len(proba)} rows; the test windows differ, so no paired "
                "comparison is valid"
            )
        }
    return {
        "baseline_run_id": record.get("run_id"),
        "baseline_log_loss": record.get("metrics", {}).get("log_loss"),
        **bootstrap_paired_delta(y_test, stored.astype(float), np.asarray(proba),
                                 labels=[0, 1], n_bootstrap=2000),
    }


def score_model(y_test, proba):
    """Accuracy, log loss, Brier and AUC, each guarded against a degenerate set."""
    predicted = (np.asarray(proba) >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, predicted) if len(proba) else None,
        "log_loss": None,
        "brier": None,
        "auc": None,
        "mean_probability": float(np.mean(proba)) if len(proba) else None,
    }
    if len(proba) and set(np.unique(y_test).tolist()) >= {0, 1}:
        metrics["log_loss"] = log_loss(y_test, np.asarray(proba), labels=[0, 1])
        metrics["brier"] = brier_score_loss(y_test, np.asarray(proba))
        metrics["auc"] = roc_auc_score(y_test, np.asarray(proba))
    return metrics


# ---------------------------------------------------------------------------
# odds alignment, joins and betting arithmetic
# ---------------------------------------------------------------------------


def join_rows_to_odds(rows, odds, offset_days):
    """Attach closing prices, joining on the real game date.

    Prefers the explicit `GameDate` column, which is what the dataset now
    carries. That join matches 1722 of 1722 held-out rows, where a join on the
    as-of `Date` matches 9. The offset is only a fallback for older tables that
    predate the column (`dataset_2012-23`, `dataset_2012-24`), whose `Date` is
    still the game date.

    The row number travels through the join and the result is re-sorted on it,
    so prices stay aligned with the y_test positions they are scored against.
    """
    has_game_date = GAME_DATE_COLUMN in rows.columns
    key_date = GAME_DATE_COLUMN if has_game_date else DATE_COLUMN
    keys = rows[[key_date, HOME_TEAM, AWAY_TEAM]].rename(columns={key_date: DATE_COLUMN})
    keys = keys.reset_index(drop=True).copy()
    keys["_row"] = np.arange(len(keys))
    if not has_game_date:
        shifted = pd.to_datetime(keys[DATE_COLUMN], errors="coerce") + pd.Timedelta(days=offset_days)
        keys[DATE_COLUMN] = shifted.dt.strftime("%Y-%m-%d")
    matched = keys.merge(
        odds, on=[DATE_COLUMN, HOME_TEAM, AWAY_TEAM], how="inner", validate="many_to_one"
    )
    return matched.sort_values("_row").reset_index(drop=True)


def alignment_match_counts(rows, odds):
    """How many rows find a closing price under each candidate alignment."""
    return {offset: len(join_rows_to_odds(rows, odds, offset)) for offset in ODDS_DATE_OFFSETS}


def market_home_probabilities(matched, implied_fn):
    """De-vigged home win probability per matched row, or None when unusable.

    The two implied probabilities are normalised against each other, so the
    book's margin is removed before the model is compared with it.
    """
    home = [implied_fn(value) for value in matched["ML_Home"]]
    away = [implied_fn(value) for value in matched["ML_Away"]]
    if any(value is None for value in home) or any(value is None for value in away):
        return None
    home = np.asarray(home, dtype=float)
    away = np.asarray(away, dtype=float)
    total = home + away
    if not np.isfinite(total).all() or (total <= 0).any():
        return None
    return home / total


def flat_bet(prices, picks, truth, profit_fn):
    """(roi_percent, roi_stderr, hit_rate, bets, total_profit) for a flat stake.

    The standard error travels with the ROI on purpose. A flat-bet ROI over a
    few hundred bets has a standard error measured in percentage points, so a
    figure quoted without one is indistinguishable from a claim.
    """
    profits = []
    wins = 0
    for price, pick, outcome in zip(prices, picks, truth):
        won = bool(pick == outcome)
        wins += won
        profits.append(profit_fn(price, won))
    bets = len(profits)
    if bets == 0:
        return None, None, None, 0, 0.0
    total_profit = float(sum(profits))
    roi = 100.0 * total_profit / (FLAT_STAKE * bets)
    stderr = None
    if bets > 1:
        spread = float(np.std(np.asarray(profits, dtype=float), ddof=1))
        stderr = 100.0 * spread / (FLAT_STAKE * np.sqrt(bets))
    return roi, stderr, wins / bets, bets, total_profit


# ---------------------------------------------------------------------------
# the leakage gate
# ---------------------------------------------------------------------------


def sample_rows(data, sample_size, seed):
    """A reproducible random sample of dataset rows for the gate."""
    if sample_size >= len(data):
        return gate_frame(data)
    rng = np.random.default_rng(seed)
    positions = np.sort(rng.choice(len(data), size=sample_size, replace=False))
    return gate_frame(data.iloc[positions])


def gate_frame(data):
    """The handful of columns the gate reads, renamed so positional access is safe.

    The dataset carries columns like `TEAM_NAME.1` and `index`, which are not
    valid identifiers; `itertuples` renames those behind your back. Selecting
    and renaming up front keeps every row tuple in a documented order.
    """
    frame = data[
        [DATE_COLUMN, HOME_TEAM, "GP", "W", AWAY_TEAM, "GP.1", "W.1", MONEYLINE_TARGET]
    ].copy()
    frame.columns = ["date", "home", "home_gp", "home_w", "away", "away_gp", "away_w", "label"]
    return frame


def check_snapshot_provenance(rows, snapshots):
    """Do the row's GP/W come from the snapshot its Date names?

    A leak-free row must agree with the stats table it points at; a row whose
    features disagree with its own Date cannot be traced, so nothing downstream
    of it can be trusted either.
    """
    checked = 0
    mismatched = 0
    missing = 0
    examples = []
    for date, home, home_gp, home_w, away, away_gp, away_w, _ in rows.itertuples(
        index=False, name=None
    ):
        home_record = snapshots.record(date, home)
        away_record = snapshots.record(date, away)
        if home_record is None or away_record is None:
            missing += 1
            continue
        checked += 1
        agrees = all(
            abs(float(record) - float(stored)) < 1e-6
            for record, stored in (
                (home_record[0], home_gp),
                (home_record[1], home_w),
                (away_record[0], away_gp),
                (away_record[1], away_w),
            )
        )
        if not agrees:
            mismatched += 1
            if len(examples) < 3:
                examples.append(
                    f"{date} {home} vs {away}: row GP/W "
                    f"({home_gp}, {home_w} | {away_gp}, {away_w}) vs snapshot "
                    f"({home_record[0]}, {home_record[1]} | "
                    f"{away_record[0]}, {away_record[1]})"
                )
    return {"checked": checked, "mismatched": mismatched, "missing": missing, "examples": examples}


def check_pre_game_join(rows, odds):
    """Is the game's own date strictly AFTER the row's snapshot date?

    A leak-free build stores the snapshot date, so the game it describes sits
    later and joins a closing price dated later. A build that stores the game
    date joined the same-day snapshot, and that snapshot carries the result of
    the very game being predicted whenever the stats were pulled after tip-off
    -- which is what turned 48% into 68.9%.

    The odds ledger is the only in-repo record of when a game was actually
    played, so it is what this check trusts. It reads prices, not team stats, so
    it says nothing about the feature values themselves; that is check 1 and 3.
    """
    rows = rows.rename(columns={"date": DATE_COLUMN, "home": HOME_TEAM, "away": AWAY_TEAM})
    counts = alignment_match_counts(rows, odds)
    best_offset = max(counts, key=lambda offset: counts[offset])
    return {
        "counts": counts,
        "offset": best_offset,
        "coverage": counts[best_offset] / len(rows) if len(rows) else 0.0,
        "same_day": counts.get(0, 0),
        "checked": len(rows),
    }


def check_increment_rule(rows, snapshots):
    """Score the audit's leak detector against these rows.

    `pred = 1 if W_home[D] - W_home[D-1] > W_away[D] - W_away[D-1] else 0` recovers
    the label almost perfectly when each team's W on D already contains the game
    on D. The two W values on D are the ones the model is actually trained on;
    the ones on D-1 come from the previous TeamData snapshot, because the
    dataset has no per-day history to read them from. On clean data both
    increments are unrelated to the label and the rule collapses to roughly the
    away-win rate, so a rule that beats the always-home base rate by a clear
    margin is the leak's own signature. It needs no odds, so it still runs under
    --skip-odds.
    """
    scored = 0
    correct = 0
    labels = []
    for date, home, _, home_w, away, _, away_w, label in rows.itertuples(
        index=False, name=None
    ):
        previous = snapshots.previous_date(date)
        if previous is None:
            continue
        home_prev = snapshots.record(previous, home)
        away_prev = snapshots.record(previous, away)
        if home_prev is None or away_prev is None:
            continue
        scored += 1
        label = int(label)
        labels.append(label)
        predicted = 1 if (home_w - home_prev[1]) > (away_w - away_prev[1]) else 0
        correct += int(predicted == label)
    return {
        "scored": scored,
        "accuracy": correct / scored if scored else None,
        "base_rate": float(np.mean(labels)) if labels else None,
    }


def run_integrity_gate(data, odds, seed, use_odds):
    """Prove the join is leak-free, or refuse to report accuracy at all.

    Returns (verdict, detail). The verdict is one of PASS, PASS_NO_ODDS, FAIL,
    UNVERIFIED or UNKNOWN, and it is the single switch the rest of the report
    obeys: only PASS and PASS_NO_ODDS print predictive metrics. FAIL means a
    check found evidence of leakage, UNVERIFIED means a check could not run, and
    UNKNOWN means TeamData itself is missing so nothing could be checked at all.
    """
    rule("1. DATASET INTEGRITY GATE (leakage)")
    sample = sample_rows(data, GATE_SAMPLE_ROWS, seed)
    print(f"Sampled {len(sample)} of {len(data)} dataset rows (seed {seed}).")

    snapshots = SnapshotReader()
    if not snapshots.available():
        print(
            f"Gate not run: {TEAM_DB.name} is missing or holds no dated snapshot tables, "
            "so no row can be traced to the stats it was built from."
        )
        print("ROI not available: the leakage gate could not be run.")
        return "UNKNOWN", {"sample": len(sample)}

    try:
        print(
            f"Team snapshot tables available: {len(snapshots.dates)} "
            f"({snapshots.dates[0]} .. {snapshots.dates[-1]})"
        )

        provenance = check_snapshot_provenance(sample, snapshots)
        print(
            f"  [1] snapshot provenance: {provenance['checked']} rows checked, "
            f"{provenance['mismatched']} disagree with the snapshot their Date names, "
            f"{provenance['missing']} not found"
        )
        for example in provenance["examples"]:
            print(f"        {example}")

        pre_game = None
        if use_odds and odds is not None and not odds.empty:
            pre_game = check_pre_game_join(sample, odds)
            counts = ", ".join(
                f"offset {offset:+d}d -> {pre_game['counts'][offset]}"
                for offset in sorted(pre_game["counts"])
            )
            print(
                f"  [2] pre-game join: {pre_game['checked']} rows checked against the odds "
                f"ledger ({counts}); best coverage {pre_game['coverage'] * 100:.1f}% at "
                f"offset {pre_game['offset']:+d} day(s)"
            )
        else:
            why = "--skip-odds" if not use_odds else f"no usable odds tables in {ODDS_DB.name}"
            print(f"  [2] pre-game join: NOT VERIFIED ({why})")

        increment = check_increment_rule(sample, snapshots)
        print(
            f"  [3] W-increment leak rule: {increment['scored']} rows scored, accuracy "
            f"{fmt(increment['accuracy'])} against an always-home base rate of "
            f"{fmt(increment['base_rate'])}"
        )
    finally:
        snapshots.close()

    failures = []
    if provenance["checked"] == 0:
        failures.append(
            "snapshot provenance could not be evaluated on any sampled row, so the "
            "dataset's features cannot be tied to any dated stats at all"
        )
    if provenance["mismatched"]:
        failures.append(
            f"{provenance['mismatched']} of {provenance['checked']} sampled rows carry GP/W "
            "that do not match the TeamData snapshot their Date names"
        )
    if pre_game is None:
        if use_odds:
            failures.append(
                "the pre-game join could not be evaluated because the odds ledger holds no "
                "usable table, so the alignment is unmeasured rather than clean"
            )
    elif pre_game["coverage"] >= 0.5 and pre_game["offset"] == 0:
        failures.append(
            f"{pre_game['same_day']} of {pre_game['checked']} sampled rows "
            f"({pre_game['same_day'] / pre_game['checked'] * 100:.1f}%) join a closing price "
            "dated the SAME DAY as the row's Date: this dataset was built against the "
            "same-day snapshot, which carries the result of the game it is predicting"
        )
    elif pre_game["same_day"] / max(pre_game["checked"], 1) > SAME_DAY_JOIN_TOLERANCE:
        failures.append(
            f"{pre_game['same_day']} of {pre_game['checked']} sampled rows join a same-day "
            f"closing price, more than the {SAME_DAY_JOIN_TOLERANCE * 100:.1f}% tolerance"
        )
    if increment["scored"] == 0:
        failures.append(
            "the W-increment leak rule could not be evaluated on any sampled row, so the "
            "snapshot series was never actually inspected"
        )
    if (
        increment["accuracy"] is not None
        and increment["base_rate"] is not None
        and increment["accuracy"] > increment["base_rate"] + RULE_EDGE_TOLERANCE
    ):
        failures.append(
            f"the W-increment rule reaches {increment['accuracy'] * 100:.1f}% against a "
            f"{increment['base_rate'] * 100:.1f}% base rate: the snapshots contain the "
            "results of the games being predicted"
        )

    detail = {
        "sample": len(sample),
        "provenance_checked": provenance["checked"],
        "provenance_mismatched": provenance["mismatched"],
        "pre_game_offset": None if pre_game is None else pre_game["offset"],
        "pre_game_same_day": None if pre_game is None else pre_game["same_day"],
        "rule_accuracy": increment["accuracy"],
        "rule_base_rate": increment["base_rate"],
    }

    if failures:
        # "Could not check" and "checked and found leakage" are different
        # verdicts and must not share one banner: only the second says anything
        # about the dataset itself.
        unverified_only = all("could not be evaluated" in failure for failure in failures)
        print()
        print("!" * 78)
        if unverified_only:
            print("!! LEAKAGE GATE COULD NOT VERIFY THIS DATASET -- NO ACCURACY, LOG")
            print("!! LOSS, BRIER, AUC OR ROI IS REPORTED. Nothing was found wrong; the")
            print("!! evidence needed to clear it is missing.")
        else:
            print("!! LEAKAGE GATE FAILED -- NO ACCURACY, LOG LOSS, BRIER, AUC OR ROI IS")
            print("!! REPORTED FOR THIS DATASET. Those numbers would be meaningless.")
        for failure in failures:
            print(f"!!   - {failure}")
        print("!" * 78)
        return ("UNVERIFIED" if unverified_only else "FAIL"), detail
    if pre_game is None:
        print(
            "Gate PASSED on the checks that could run (snapshot provenance and the "
            "W-increment rule). The pre-game join was NOT verified."
        )
        return "PASS_NO_ODDS", detail
    print(
        f"Gate PASSED: {len(sample)} rows sampled; each matches the snapshot its Date "
        f"names, the closing price for every one sits {pre_game['offset']:+d} day(s) after "
        "that date, and the W-increment rule does not beat the base rate."
    )
    return "PASS", detail


# ---------------------------------------------------------------------------
# report sections
# ---------------------------------------------------------------------------


def build_odds_join(test_rows, odds):
    """Match the test rows to closing prices, or explain why no ROI is possible.

    Returns (matched_frame_or_None, offset, matched_count, reason). `reason` is
    empty when the join can be scored; otherwise it explains, in one clause, why
    no figure may be printed from it.
    """
    counts = alignment_match_counts(test_rows, odds)
    offset = max(counts, key=lambda key: counts[key])
    matched_count = counts[offset]
    total = len(test_rows)
    coverage = matched_count / total if total else 0.0
    print(
        "Odds join alignment: "
        + ", ".join(f"{key:+d}d -> {value} of {total}" for key, value in sorted(counts.items()))
        + f"; using {offset:+d} day(s) at {coverage * 100:.1f}% coverage."
    )
    if matched_count < MIN_MATCHED_ODDS_ROWS:
        return None, offset, matched_count, (
            f"only {matched_count} of {total} test rows matched a closing odds line, fewer "
            f"than the {MIN_MATCHED_ODDS_ROWS} needed to report an ROI"
        )
    if coverage < MIN_JOIN_COVERAGE:
        return None, offset, matched_count, (
            f"the odds join matched {matched_count} of {total} test rows "
            f"({coverage * 100:.1f}%), under the {MIN_JOIN_COVERAGE * 100:.0f}% coverage "
            "floor; a sparse join cannot be scored"
        )
    reason = ""
    if matched_count < total:
        reason = (
            f"the join is partial -- {matched_count} of {total} test rows "
            f"({coverage * 100:.1f}%) carry a closing line. Those {total - matched_count} "
            "games have no price in the odds database and cannot be bet at one, so every "
            "closing-line figure below is computed on the priced rows only"
        )
    return join_rows_to_odds(test_rows, odds, offset), offset, matched_count, reason


def report_baselines(y_test, priced_fraction):
    """Always-home, majority class, and how much of the test set has a price."""
    rule("2. BASELINES")
    total = len(y_test)
    home_rate = float(np.mean(y_test)) if total else None
    majority_class = "home" if (home_rate or 0.0) >= 0.5 else "away"
    majority_accuracy = home_rate if majority_class == "home" else (
        None if home_rate is None else 1.0 - home_rate
    )
    print(f"Test rows: {total}")
    print(f"Always bet home:              accuracy {fmt(home_rate)}")
    print(f"Majority class ({majority_class:<4}):            accuracy {fmt(majority_accuracy)}")
    if priced_fraction is None:
        print("Test rows with a closing line: n/a (odds skipped)")
    else:
        print(
            "Test rows with a closing line: "
            f"{fmt(priced_fraction)} ({int(round(priced_fraction * total))} of {total})"
        )
    return {
        "always_home": home_rate,
        "majority_class": majority_class,
        "majority_accuracy": majority_accuracy,
        "priced_fraction": priced_fraction,
    }


def report_model(model_name, seed, X_train, y_train, X_test, y_test):
    """Fit on the training slice and score the held-out tail."""
    rule(f"3. TRAINED MODEL ({model_name})")
    if model_name == "none":
        print("--model none: no estimator was fitted, by request.")
        print("Accuracy: n/a (no model trained)")
        print("Log loss: n/a (no model trained)")
        print("Brier:     n/a (no model trained)")
        print("AUC:       n/a (no model trained)")
        return None
    model = build_model(model_name, seed)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = score_model(y_test, proba)
    metrics["estimator"] = type(model).__name__
    print(
        f"Estimator {metrics['estimator']} (random_state={seed}) fitted on "
        f"{len(X_train)} rows, scored on {len(X_test)} held-out rows."
    )
    print(f"Accuracy: {fmt(metrics['accuracy'])}")
    print(f"Log loss: {fmt(metrics['log_loss'])}")
    print(f"Brier:     {fmt(metrics['brier'])}")
    print(f"AUC:       {fmt(metrics['auc'])}")
    print(f"Mean P(home win): {fmt(metrics['mean_probability'])}")
    print("Probabilities are raw model output; no calibrator is fitted.")
    metrics["proba"] = proba
    return metrics


def report_market_baseline(matched, truth, profit_fn, implied_fn, offset):
    """Accuracy and flat-bet ROI of always backing the closing-line favourite."""
    rule("4. MARKET BASELINE (always bet the closing-line favourite)")
    probabilities = market_home_probabilities(matched, implied_fn)
    if probabilities is None:
        print("Market baseline not available: a matched row carries an unusable price.")
        return None
    picks = np.where(probabilities >= 0.5, 1, 0)
    prices = np.where(picks == 1, matched["ML_Home"].to_numpy(), matched["ML_Away"].to_numpy())
    roi, stderr, hit_rate, bets, profit = flat_bet(prices, picks, truth, profit_fn)
    print(f"Priced test rows: {bets} (odds joined at snapshot date {offset:+d} day)")
    print(f"Favourite accuracy: {fmt(accuracy_score(truth, picks))}")
    print(f"Favourite hit rate: {fmt(hit_rate)}")
    print(
        f"Favourite ROI:      {fmt(roi, '+.2f')}% +/- {fmt(stderr, '.2f')} pp (1 s.e.), "
        f"{fmt(profit, '+.2f')} on {bets} flat $100 bets"
    )
    return {
        "accuracy": accuracy_score(truth, picks),
        "roi": roi,
        "stderr": stderr,
        "bets": bets,
        "hit_rate": hit_rate,
    }


def report_model_roi(matched, truth, proba, profit_fn, implied_fn):
    """Flat-bet ROI at the closing line across a sweep of edge thresholds."""
    rule("5. MODEL ROI AT THE CLOSING LINE")
    if proba is None:
        print("ROI not available: no model was trained (--model none).")
        return {}
    probabilities = market_home_probabilities(matched, implied_fn)
    if probabilities is None:
        print("ROI not available: a matched row carries an unusable closing price.")
        return {}
    model_probability = np.asarray(proba)[matched["_row"].to_numpy()]
    picks = np.where(model_probability >= 0.5, 1, 0)
    market_for_pick = np.where(picks == 1, probabilities, 1.0 - probabilities)
    edge = model_probability - market_for_pick
    home_prices = matched["ML_Home"].to_numpy()
    away_prices = matched["ML_Away"].to_numpy()
    print(
        f"Rule: back the model's pick when P(model) - P(de-vigged market) > threshold, "
        f"flat ${FLAT_STAKE:.0f} at the closing price, over {len(matched)} priced rows."
    )
    results = {}
    for threshold in EDGE_THRESHOLDS:
        selected = edge > threshold
        if not selected.any():
            print(f"  edge > {threshold:.2f}: 0 bets, so there is no ROI to report.")
            results[threshold] = {"roi": None, "stderr": None, "bets": 0, "hit_rate": None}
            continue
        prices = np.where(
            picks[selected] == 1, home_prices[selected], away_prices[selected]
        )
        roi, stderr, hit_rate, bets, profit = flat_bet(
            prices, picks[selected], truth[selected], profit_fn
        )
        print(
            f"  edge > {threshold:.2f}: {bets} bets, hit rate {fmt(hit_rate)}, profit "
            f"{fmt(profit, '+.2f')}, ROI {fmt(roi, '+.2f')}% +/- {fmt(stderr, '.2f')} pp (1 s.e.)"
        )
        results[threshold] = {"roi": roi, "stderr": stderr, "bets": bets, "hit_rate": hit_rate}
    return results


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def build_parser():
    parser = argparse.ArgumentParser(
        description="Reproducible evaluation of the moneyline model and its closing-line ROI."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset table name.")
    parser.add_argument(
        "--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Fraction held out."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=["logreg", "gbdt", "none"],
        help="Estimator to train; `none` reports baselines only.",
    )
    parser.add_argument(
        "--skip-odds", action="store_true", help="Skip every odds-dependent metric."
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="run_id from runs/ledger.jsonl to compare against. Produces a paired "
             "bootstrap of the log-loss difference on the same rows, which is the "
             "only comparison precise enough to see a real change.",
    )
    parser.add_argument(
        "--no-ledger",
        action="store_true",
        help="Do not append this run to runs/ledger.jsonl.",
    )
    parser.add_argument(
        "--allow-stale-dataset",
        action="store_true",
        help="Proceed when Data/manifest.json does not match the live table.",
    )
    return parser


def describe_dataset(data, X, train_rows, test_rows):
    """Provenance for the split: sizes, boundaries and feature count."""
    print(f"Dataset rows: {len(data)}")
    print(f"Date range:    {data[DATE_COLUMN].min().date()} .. {data[DATE_COLUMN].max().date()}")
    print(
        f"Train rows:    {len(train_rows)} "
        f"({train_rows[DATE_COLUMN].min().date()} .. {train_rows[DATE_COLUMN].max().date()})"
    )
    print(
        f"Test rows:     {len(test_rows)} "
        f"({test_rows[DATE_COLUMN].min().date()} .. {test_rows[DATE_COLUMN].max().date()})"
    )
    print(
        f"Feature columns: {X.shape[1]} (dropped {len(DROP_COLUMNS)} label and key columns, "
        "TEAM_ID absent from this table)"
    )


def main():
    args = build_parser().parse_args()

    rule(f"EVALUATION HARNESS -- {DATASET_DB.name}::{args.dataset}")
    print(
        f"Options: dataset={args.dataset} test_size={args.test_size} seed={args.seed} "
        f"model={args.model} skip_odds={args.skip_odds}"
    )

    # Provenance gate: refuse to report metrics against a table that is not the
    # one the manifest describes. Without this, "which build produced this
    # table" is unanswerable and two runs can be compared as though they saw the
    # same data.
    fingerprint = dataset_fingerprint(args.dataset)
    if fingerprint is None:
        print(
            "Dataset fingerprint: no Data/manifest.json entry for "
            f"{args.dataset}. Metrics will be reported without provenance."
        )
    else:
        print(f"Dataset fingerprint: {fingerprint[:16]}")

    df = load_dataset(args.dataset)
    if df is None or df.empty:
        if df is not None:
            print(f"No rows found for dataset {args.dataset}.")
        return 2

    profit_fn, implied_fn, arithmetic_source = resolve_odds_arithmetic()
    print(f"American-odds arithmetic: {arithmetic_source}")

    data, X, y = prepare_frame(df)
    X_train, X_test, y_train, y_test = chronological_split(X, y, args.test_size)
    if len(X_train) < 2 or len(X_test) < 1:
        print(
            f"ERROR: --test-size {args.test_size} leaves {len(X_train)} train rows and "
            f"{len(X_test)} test rows; there is nothing to measure."
        )
        return 2
    X_train, X_test = impute_fit_transform(X_train, X_test)
    train_rows = data.iloc[: len(X_train)]
    test_rows = data.iloc[len(X_train):].reset_index(drop=True)
    describe_dataset(data, X, train_rows, test_rows)

    odds = None
    odds_raw_rows = 0
    if args.skip_odds:
        print(f"Odds: skipped (--skip-odds); {ODDS_DB.name} was not opened.")
    else:
        odds, odds_raw_rows = load_closing_odds()
        if odds is None:
            print(f"Odds: {ODDS_DB.name} is absent or holds no season tables.")
        else:
            print(
                f"Odds: {len(odds)} deduped games from {odds_raw_rows} rows across the season "
                f"tables of {ODDS_DB.name} (deduped on Date, Home, Away)."
            )

    verdict, gate = run_integrity_gate(data, odds, args.seed, not args.skip_odds)
    accuracy_allowed = verdict in ("PASS", "PASS_NO_ODDS")

    matched = None
    offset = None
    matched_count = 0
    market = None
    roi_results = {}
    model_metrics = None
    join_note = "odds metrics were skipped (--skip-odds)"
    priced_fraction = None
    if odds is not None and not odds.empty:
        matched, offset, matched_count, join_note = build_odds_join(test_rows, odds)
        priced_fraction = matched_count / len(test_rows) if len(test_rows) else None
        if join_note:
            print(f"NOTE: {join_note}.")

    baselines = report_baselines(y_test, priced_fraction)

    if not accuracy_allowed:
        rule("3. TRAINED MODEL")
        print("Suppressed: the leakage gate failed, so no model metric is reported.")
        rule("4. MARKET BASELINE (always bet the closing-line favourite)")
        print("Suppressed: the leakage gate failed, so no accuracy is reported.")
        rule("5. MODEL ROI AT THE CLOSING LINE")
        print("ROI not available: the leakage gate failed.")
        model_metrics = None
    else:
        model_metrics = report_model(args.model, args.seed, X_train, y_train, X_test, y_test)
        if matched is None:
            rule("4. MARKET BASELINE (always bet the closing-line favourite)")
            print(f"Market baseline not available: {join_note}.")
            rule("5. MODEL ROI AT THE CLOSING LINE")
            print(f"ROI not available: {join_note}.")
        else:
            truth = y_test[matched["_row"].to_numpy()]
            market = report_market_baseline(matched, truth, profit_fn, implied_fn, offset)
            roi_results = report_model_roi(
                matched,
                truth,
                model_metrics["proba"] if model_metrics else None,
                profit_fn,
                implied_fn,
            )

    summary = [
        f"dataset={args.dataset}",
        f"dataset_rows={len(data)}",
        f"train_rows={len(X_train)}",
        f"test_rows={len(X_test)}",
        f"test_size={args.test_size}",
        f"seed={args.seed}",
        f"model={args.model}",
        f"feature_columns={X.shape[1]}",
        f"leakage_gate={verdict}",
        f"gate_rows_checked={fmt_count(gate.get('sample'))}",
        f"gate_provenance_checked={fmt_count(gate.get('provenance_checked'))}",
        f"gate_provenance_mismatched={fmt_count(gate.get('provenance_mismatched'))}",
        f"gate_odds_date_offset={fmt_count(gate.get('pre_game_offset'))}",
        f"gate_same_day_joins={fmt_count(gate.get('pre_game_same_day'))}",
        f"gate_rule_accuracy={fmt(gate.get('rule_accuracy'))}",
        f"baseline_always_home_accuracy={fmt(baselines['always_home'])}",
        f"baseline_majority_class={baselines['majority_class']}",
        f"baseline_majority_class_accuracy={fmt(baselines['majority_accuracy'])}",
        f"baseline_test_rows_with_closing_line={fmt(baselines['priced_fraction'])}",
        f"odds_deduped_games={len(odds) if odds is not None else 0}",
        f"odds_raw_rows={odds_raw_rows}",
        f"odds_date_offset_days={offset if offset is not None else 'n/a'}",
        f"odds_join_matched={matched_count}",
        f"odds_join_of_test_rows={len(X_test)}",
    ]
    model_errors = None
    if model_metrics is None:
        summary += [
            "model_accuracy=n/a",
            "model_log_loss=n/a",
            "model_brier=n/a",
            "model_auc=n/a",
        ]
    else:
        model_errors = bootstrap_metric_errors(
            y_test, model_metrics["proba"], seed=args.seed
        )
        summary += [
            f"model_accuracy={fmt(model_metrics['accuracy'])}",
            f"model_log_loss={fmt(model_metrics['log_loss'])}",
            f"model_brier={fmt(model_metrics['brier'])}",
            f"model_auc={fmt(model_metrics['auc'])}",
        ]
        for key in ("accuracy", "log_loss", "brier", "auc"):
            error = (model_errors or {}).get(key)
            summary.append(f"model_{key}_se={fmt(error)}")
    if market is None:
        summary += [
            "market_favourite_accuracy=n/a",
            "market_favourite_roi_pct=n/a",
            "market_favourite_bets=n/a",
        ]
    else:
        summary += [
            f"market_favourite_accuracy={fmt(market['accuracy'])}",
            f"market_favourite_roi_pct={fmt(market['roi'], '+.2f')}",
            f"market_favourite_roi_stderr_pp={fmt(market['stderr'], '.2f')}",
            f"market_favourite_bets={market['bets']}",
        ]
    for threshold in EDGE_THRESHOLDS:
        entry = roi_results.get(threshold)
        if not entry or not entry["bets"]:
            summary += [
                f"model_roi_pct_edge_{threshold:.2f}=n/a",
                f"model_bets_edge_{threshold:.2f}={entry['bets'] if entry else 'n/a'}",
            ]
            continue
        summary += [
            f"model_roi_pct_edge_{threshold:.2f}={fmt(entry['roi'], '+.2f')}",
            f"model_roi_stderr_pp_edge_{threshold:.2f}={fmt(entry['stderr'], '.2f')}",
            f"model_bets_edge_{threshold:.2f}={entry['bets']}",
            f"model_hit_rate_edge_{threshold:.2f}={fmt(entry['hit_rate'])}",
        ]

    rule("STANDARD ERRORS")
    if not model_errors:
        print("No model was scored, so no intervals are available.")
    else:
        print("Bootstrap standard errors on the held-out rows. A change smaller than")
        print("these is not a change. Seed noise alone reaches ~0.0039 log loss here.")
        for key in ("accuracy", "log_loss", "brier", "auc"):
            error = model_errors.get(key)
            if error is None:
                continue
            value = model_metrics.get(key)
            if value is None:
                continue
            print(f"  {key:9s} = {value:.5f}  +/- {error:.5f}")

    run_id = None
    if model_metrics is not None:
        if args.baseline:
            rule(f"PAIRED COMPARISON vs {args.baseline}")
            comparison = compare_to_baseline(y_test, model_metrics["proba"], args.baseline)
            if comparison is None:
                print(
                    f"No ledger entry for {args.baseline} with stored predictions. "
                    "Run evaluate.py once to create one."
                )
            elif "error" in comparison:
                print(f"Comparison invalid: {comparison['error']}")
            else:
                print(
                    f"Baseline {comparison['baseline_run_id']} log loss "
                    f"{comparison['baseline_log_loss']:.5f}"
                )
                print(f"  delta (baseline - this run) = {comparison['delta']:+.5f} nats  "
                      f"SE = {comparison['standard_error']:.5f}")
                print(f"  95% CI [{comparison['low']:+.5f}, {comparison['high']:+.5f}]")
                low, high = comparison["low"], comparison["high"]
                if low > 0:
                    print("  -> THIS RUN IS BETTER (significantly; interval excludes zero)")
                elif high < 0:
                    print("  -> THIS RUN IS WORSE (significantly; interval excludes zero)")
                elif abs(comparison["delta"]) < 1e-9:
                    print("  -> identical predictions; nothing to compare")
                else:
                    print("  -> INDISTINGUISHABLE (interval crosses zero: not a real change)")
        else:
            rule("PAIRED COMPARISON")
            print(
                "No --baseline given, so this run was not compared with any other.\n"
                "A point estimate alone cannot distinguish a real change from noise:\n"
                "the smallest resolvable paired log-loss change is ~0.0037."
            )

        if not args.no_ledger:
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            append_ledger_line({
                "run_id": run_id,
                "started_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                "dataset": args.dataset,
                "dataset_rows": len(data),
                "dataset_fingerprint": fingerprint,
                "split": {
                    "train_rows": len(X_train), "test_rows": len(X_test),
                    "test_size": args.test_size,
                    "train_max_date": str(train_rows[DATE_COLUMN].max()),
                    "test_min_date": str(test_rows[DATE_COLUMN].min()),
                    "test_max_date": str(test_rows[DATE_COLUMN].max()),
                },
                "model": args.model,
                "seed": args.seed,
                "feature_columns": int(X.shape[1]),
                "metrics": {
                    key: model_metrics.get(key)
                    for key in ("accuracy", "log_loss", "brier", "auc")
                },
                "metric_standard_errors": model_errors,
                "baseline_run_id": args.baseline,
                "test_predictions_zb64": base64.b64encode(
                    zlib.compress(np.asarray(model_metrics["proba"], dtype=np.float32))
                ).decode("ascii"),
            })
            print(f"\nAppended run {run_id} to {ledger_path().relative_to(BASE_DIR)}.")

    rule("SUMMARY")
    print(
        "One key=value per metric; `n/a` means the figure was deliberately not computed, "
        "and the reason is printed above."
    )
    for line in summary:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
