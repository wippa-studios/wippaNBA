"""Calibration of an already-fitted model, and the closing-line ROI report.

Shared by the four scikit-learn trainers -- the two XGBoost and the two logistic
regression scripts. The two neural network trainers do not use either half: they
have no calibrator and no closing-line report.

`make_calibrated_estimator` wraps a model that has already been fitted without
refitting it. `report_closing_line` prints the market baseline and the realised
ROI against the closing line for whichever model produced the picks.

CALIBRATION IS GATED, AND ITS EFFECT IS MEASURED
================================================
Two things changed here after measuring what the calibrators actually do on the
held-out set (1,722 rows):

* Isotonic is off by default and refused below ``MIN_CALIBRATION_ROWS``. Isotonic
  regression fits one free parameter per training point, so it needs far more
  rows than the calibration segment has. Measured here it was **+0.0035 nats --
  actively worse than doing nothing**, which is what overfitting 1,562 rows of a
  two-parameter calibration segment looks like.

* Sigmoid is kept, but its effect is now reported rather than assumed. Measured on
  the moneyline held-out set it moved log loss by -0.00008 +/- 0.00101 on one run
  and +0.00214 on an independent replication: mechanically a near-constant
  intercept shift (correlation 0.999 with the raw scores, mean shift -0.015 to
  -0.020), i.e. nothing, within noise. Re-measured on the current table with
  `Logistic_Regression_ML.py --trials 5`: **-0.00019 +/- 0.00088**, 95% CI
  [-0.00200, +0.00153] -- an interval that includes zero, which is now a printed
  measurement rather than an assumption baked into a score.

  On the TOTALS model the same sigmoid is NOT a no-op: -0.0104 +/- 0.0034, an
  interval excluding zero. Three classes with a 1% push tail distort the
  probability simplex in a way a two-parameter sigmoid genuinely repairs. Which is
  the point of reporting the number instead of deciding in advance that
  calibration does nothing.

  Expected calibration error is deliberately NOT the reported effect. At this
  sample size ECE runs 0.021-0.028, inside the 0.0276 mean / 0.0391 p95 band a
  PERFECTLY calibrated model produces, so ECE movement at this n is not evidence
  of anything.
"""

import re
import sqlite3

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

from src.Utils import metrics

from training_common import BASE_DIR

ODDS_DB = BASE_DIR / "Data" / "OddsData.sqlite"

MIN_MATCHED_ODDS_ROWS = 50

# Isotonic needs enough rows to estimate one free parameter per knot without
# overfitting them. This dataset's calibration segment is 1,547 rows and
# isotonic still cost +0.0035 nats, so the floor is deliberately well above the
# segment size here: the point is that asking for isotonic is a decision that
# has to be made out loud, at a size where the evidence exists.
MIN_CALIBRATION_ROWS = 2000

# The default --calibration choices. Isotonic has to be asked for with
# --allow-isotonic, which widens them; see
# ``training_common.calibration_choices`` for why that cannot be done inside the
# parser, and ``resolve_calibration_method`` below for the size gate.
CALIBRATION_CHOICES = ("sigmoid", "none")

# Re-exported so the trainers get it from a sibling module that already
# imported src.Utils.metrics, rather than each of them reaching across to the
# repository root and depending on import order within their own file.
log_loss_standard_error = metrics.log_loss_standard_error


def resolve_calibration_method(method, n_calib_rows, allow_isotonic=False):
    """Return the calibrator to use, or raise naming what to change.

    ``method="isotonic"`` raises unless BOTH the caller opted in with
    ``allow_isotonic`` AND the calibration segment is at least
    ``MIN_CALIBRATION_ROWS`` rows. Either gate alone would be enough to stop the
    accident; requiring both means the decision is deliberate and evidenced.
    """
    if method != "isotonic":
        return method
    if not allow_isotonic:
        raise ValueError(
            "Isotonic calibration is not offered by default: on this dataset it "
            "measured +0.0035 nats, worse than no calibration at all, because it "
            f"fits one free parameter per training point. Pass --allow-isotonic "
            f"and use at least {MIN_CALIBRATION_ROWS} calibration rows to insist "
            f"(this run has {n_calib_rows})."
        )
    if n_calib_rows < MIN_CALIBRATION_ROWS:
        raise ValueError(
            f"Isotonic calibration needs at least {MIN_CALIBRATION_ROWS} "
            f"calibration rows; this run has {n_calib_rows}. It fits one free "
            "parameter per training point and will overfit anything smaller."
        )
    return method


def report_calibration_effect(
    y_true, raw_proba, calibrated_proba, labels=None, n_calib_rows=None, method="none"
):
    """Print what the calibrator did, with a standard error on it.

    Returns the dict from ``metrics.bootstrap_paired_delta`` so a caller or a
    test can assert on the numbers rather than on stdout. The reported delta is
    calibrated minus raw: NEGATIVE means the calibrator helped.

    Also prints the calibration segment size next to the choice, because
    "sigmoid" on 1,547 rows and "sigmoid" on 150,000 rows are different claims
    wearing the same name, and the segment size is what tells them apart.
    """
    verdict = metrics.bootstrap_paired_delta(
        y_true, calibrated_proba, raw_proba, labels=labels
    )
    segment = "n/a" if n_calib_rows is None else str(n_calib_rows)
    print(
        f"Calibration: {method}, fit on {segment} rows. "
        f"Effect on held-out log loss (calibrated - raw): "
        f"{verdict['delta']:+.5f} +/- {verdict['standard_error']:.5f} "
        f"(bootstrap SE, {verdict['n_bootstrap']} resamples, 95% CI "
        f"[{verdict['low']:+.5f}, {verdict['high']:+.5f}])"
    )
    lower, upper = verdict["low"], verdict["high"]
    crosses_zero = lower <= 0.0 <= upper
    print(
        "  -> the interval includes zero, so this calibrator's effect is not "
        "distinguishable from none." if crosses_zero else
        "  -> the interval excludes zero, so this calibrator's effect is real "
        f"({'improves' if verdict['delta'] < 0 else 'hurts'} log loss)."
    )
    return verdict


SEASON_PATTERN = re.compile(r"^(?:odds_)?(\d{4}-\d{2})(?:_new)?$")
# Same candidate precedence as src/Process-Data/Create_Games.py, so the closing
# line reported here is the same line that built the dataset.
ODDS_TABLE_CANDIDATES = (
    "odds_{season}_new",
    "odds_{season}",
    "{season}_new",
    "{season}",
)

# WHICH dataset column joins to the odds date.
#
# These are two different dates and the difference is a bug fix, not a quirk.
# Data/OddsData.sqlite stores the date a game was PLAYED on. The dataset's
# ``Date`` used to be that same date, and an exact-date merge was correct.
# It is not any more: src/Process-Data/Create_Games.py now joins each game to
# the newest COMPLETE team-stats snapshot STRICTLY BEFORE the game, because the
# NBA ``leaguedashteamstats`` endpoint treats ``DateTo`` as inclusive, so the
# snapshot named D already contained the game played on D and every ``*_RANK``
# derived from it. That fix kept ``Date`` as the honest as-of marker and moved
# the trainers onto ``GameDate``, the real game date, which Create_Games now
# writes beside it.
#
# So: join on GameDate. An exact merge on Date now misses all but a handful of
# rows by one day. Do NOT "simplify" this into ``Date + timedelta(days=1)`` --
# the offset is a guess about the snapshot spacing, it silently rots if
# ``Date`` ever becomes the game date again, and it is wrong on the first and
# last game of every season, where no prior snapshot exists or the gap is
# larger than a day. The explicit column needs none of that.
GAME_DATE_COLUMN = "GameDate"
SNAPSHOT_DATE_COLUMN = "Date"


def make_calibrated_estimator(estimator, method):
    """Wrap an already-fitted model in a probability calibrator.

    Version fork: scikit-learn removed ``cv="prefit"`` in 1.6 in favour of
    ``FrozenEstimator``, and raises InvalidParameterError on modern versions.
    requirements.txt pins scikit-learn==1.3.1, where "prefit" is valid and
    FrozenEstimator does not exist, so that pin selects the legacy branch today.
    """
    try:
        from sklearn.frozen import FrozenEstimator
        return CalibratedClassifierCV(FrozenEstimator(estimator), method=method)
    except ImportError:
        return CalibratedClassifierCV(estimator, method=method, cv="prefit")


def american_odds_profit(odds, won):
    """Profit on a flat $100 stake priced at American ``odds``.

    American odds pay ``odds`` dollars of profit when positive and
    10000 / abs(odds) when negative; a losing bet loses the stake either way.
    """
    if not won:
        return -100.0
    odds = float(odds)
    if odds > 0:
        return odds
    if odds < 0:
        return 10000.0 / abs(odds)
    return -100.0


def american_odds_implied_probability(odds):
    """Vig-free win probability implied by American ``odds``, None if unusable."""
    odds = float(odds)
    if odds == 0:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)


def load_closing_odds():
    """Closing moneyline prices for every season the odds database holds.

    One table per season, resolved with the candidate precedence used by
    src/Process-Data/Create_Games.py. Returns None when the database is absent
    or holds no season tables, so the caller reports a missing ROI instead of
    failing.
    """
    if not ODDS_DB.exists():
        return None

    with sqlite3.connect(ODDS_DB) as con:
        existing = {
            row[0] for row in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        seasons = {match.group(1) for match in map(SEASON_PATTERN.match, existing) if match}
        frames = []
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

    if not frames:
        return None

    odds = pd.concat(frames, ignore_index=True)
    odds = odds.rename(columns={"Home": "TEAM_NAME", "Away": "TEAM_NAME.1"})
    odds["Date"] = pd.to_datetime(odds["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    odds["ML_Home"] = pd.to_numeric(odds["ML_Home"], errors="coerce")
    odds["ML_Away"] = pd.to_numeric(odds["ML_Away"], errors="coerce")
    odds = odds.dropna(subset=["Date", "TEAM_NAME", "TEAM_NAME.1", "ML_Home", "ML_Away"])
    # One row per game: a duplicated key would silently double count a matchup.
    return odds.drop_duplicates(subset=["Date", "TEAM_NAME", "TEAM_NAME.1"], keep="first")


def odds_join_date_column(rows):
    """The dataset column the odds join should key on, and whether it fell back.

    ``GameDate`` whenever the table has it. ``Date`` otherwise, with a warning:
    dataset_2012-23 and dataset_2012-24 predate the leak-free join and so have
    no GameDate column, and on those tables ``Date`` IS the game date, so the
    join is still right -- the caller just has to be told which convention it
    is reading. Silence there would let an old table look like a broken one.
    """
    if GAME_DATE_COLUMN in rows.columns:
        return GAME_DATE_COLUMN, False
    return SNAPSHOT_DATE_COLUMN, True


def match_test_rows_to_odds(rows, odds):
    """Join test rows to the closing line on date plus both team names.

    The date comes from ``odds_join_date_column`` -- GameDate, or Date on a
    dataset old enough to predate it. Never by positional arithmetic. The row
    number is carried through and the result re-sorted on it, so the joined
    prices stay aligned with the y_test array they are scored against.
    """
    date_column, fell_back = odds_join_date_column(rows)
    if fell_back:
        print(
            f"Warning: these test rows carry no {GAME_DATE_COLUMN} column, so the "
            f"closing-line join falls back to {SNAPSHOT_DATE_COLUMN}. That is correct "
            "for a dataset built before the leak-free join, where Date is still the "
            "game date. Rebuild the dataset to get an exact join."
        )
    keys = rows[[date_column, "TEAM_NAME", "TEAM_NAME.1"]].reset_index(drop=True)
    keys = keys.rename(columns={date_column: SNAPSHOT_DATE_COLUMN})
    keys[SNAPSHOT_DATE_COLUMN] = pd.to_datetime(
        keys[SNAPSHOT_DATE_COLUMN], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    keys = keys.rename_axis("_row").reset_index()
    matched = keys.merge(odds, on=[SNAPSHOT_DATE_COLUMN, "TEAM_NAME", "TEAM_NAME.1"], how="inner")
    return matched.sort_values("_row").reset_index(drop=True)


def report_closing_line(rows, home_win_truth, moneyline_picks=None):
    """Print the market baseline and the realised ROI against the closing line.

    ``rows`` are the test rows, ``home_win_truth`` the home-win label for them
    and ``moneyline_picks`` the model's own 0/1 home/away picks -- None for a
    totals model, which has no closing over/under price to be bet at here.

    Every failure mode -- no odds database, no season tables, too few matched
    rows, a partial join, an unusable price -- prints an explicit reason and
    returns. Nothing here raises: a missing ROI is fine, a wrong ROI is not, so
    no ROI is ever reported from a partial join.
    """
    if "Home-Team-Win" not in rows.columns:
        reason = "the test rows carry no Home-Team-Win column"
    else:
        reason = None
        try:
            odds = load_closing_odds()
            if odds is None or odds.empty:
                raise LookupError(f"no closing odds tables in {ODDS_DB.name}")
            matched = match_test_rows_to_odds(rows, odds)
        except Exception as exc:
            # A reporting aid must never kill a training run.
            reason = str(exc) or exc.__class__.__name__

    if reason is None:
        if len(matched) < MIN_MATCHED_ODDS_ROWS:
            reason = (
                f"only {len(matched)} test rows matched a closing odds line, fewer "
                f"than the {MIN_MATCHED_ODDS_ROWS} needed to report an ROI"
            )
        elif len(matched) != len(rows):
            reason = (
                f"the odds join matched {len(matched)} of {len(rows)} test rows; a "
                "partial join cannot be scored"
            )

    if reason is not None:
        print(f"Market baseline not available: {reason}.")
        print(f"ROI not available: {reason}.")
        return

    print(f"Closing-line join: {len(matched)} of {len(rows)} test rows matched a price.")

    home_prob = matched["ML_Home"].map(american_odds_implied_probability)
    away_prob = matched["ML_Away"].map(american_odds_implied_probability)
    usable = (home_prob.notna() & away_prob.notna()).to_numpy()
    if not usable.all():
        print(
            f"Market baseline not available: {int((~usable).sum())} test rows carry an "
            "unusable closing price."
        )
        print(
            f"ROI not available: {int((~usable).sum())} test rows carry an unusable "
            "closing price."
        )
        return

    positions = matched["_row"].to_numpy()
    truth = home_win_truth[positions]
    favourite = np.where(home_prob.to_numpy() >= away_prob.to_numpy(), 1, 0)
    baseline = accuracy_score(truth, favourite)
    # Named label: in the two UO scripts this is a moneyline reference, not a
    # baseline for the totals accuracy printed above it.
    print(
        "Market baseline (always bet the closing-line favourite, scored on "
        f"Home-Team-Win): {baseline:.4f}"
    )

    if moneyline_picks is None:
        print(
            f"ROI not available: {ODDS_DB.name} prices the moneyline only "
            "(ML_Home/ML_Away); it carries no closing over/under price."
        )
        return

    picks = moneyline_picks[positions]
    prices = np.where(picks == 1, matched["ML_Home"].to_numpy(), matched["ML_Away"].to_numpy())
    profit = sum(
        american_odds_profit(price, won=(pick == outcome))
        for price, pick, outcome in zip(prices, picks, truth)
    )
    roi = 100.0 * profit / (100.0 * len(prices))
    print(
        f"ROI of the model's own picks at the closing line: {roi:+.2f}% "
        f"({profit:+.2f} on {len(prices)} bets of $100)"
    )
