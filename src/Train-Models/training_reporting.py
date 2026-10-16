"""Calibration of an already-fitted model, and the closing-line ROI report.

Shared by the four scikit-learn trainers -- the two XGBoost and the two logistic
regression scripts. The two neural network trainers do not use either half: they
have no calibrator and no closing-line report.

`make_calibrated_estimator` wraps a model that has already been fitted without
refitting it. `report_closing_line` prints the market baseline and the realised
ROI against the closing line for whichever model produced the picks.
"""

import re
import sqlite3

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

from training_common import BASE_DIR

ODDS_DB = BASE_DIR / "Data" / "OddsData.sqlite"

MIN_MATCHED_ODDS_ROWS = 50
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
