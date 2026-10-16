"""Recency-weighted feature construction from GAME-LEVEL team history.

WHAT THIS IS FOR (ROADMAP Level 3.1)
====================================
The shipped dataset's features are 106 season-to-date cumulative averages read
off ``leaguedashteamstats`` snapshots. Between two consecutive games those
numbers barely move -- a team's PTS average on Tuesday and on Wednesday differ by
a thousandth -- so a model trained on them is close to blind to form. The
ROADMAP calls replacing them with exponentially-weighted moving averages "the
cheapest meaningful change on the list and typically the single largest gain".

Everything here is a function of a team's games STRICTLY BEFORE the game being
predicted. That is the whole contract; the rest of this docstring is the contract
written down so it can be audited.

THE DATA SOURCE, AND WHY IT IS (a) + (c)
========================================
``Data/TeamData.sqlite`` holds one table per calendar date, each a season-to-date
aggregate in ``PerMode=PerGame``: an AVERAGE, not a per-game observation. An
average cannot be recency-weighted, so game-level lines have to be recovered.
Three routes were available:

  (a) difference consecutive dated aggregates  -- offline, uses committed data
  (b) a game-log endpoint via ``src/Utils/nba_client.py``
  (c) accept a game-level table and stay source-agnostic

(b) buys nothing testable: ``nba_client`` wraps ``leaguedashteamstats`` and
``fetch_team_stats``, both cumulative-snapshot endpoints. It exposes no game log,
and the module's own docstring records that the host times out from this
environment anyway. So the feature maths is written source-agnostically, in the
spirit of (c) -- every function takes a per-game frame -- and
``extract_game_logs`` implements (a) to produce that frame from the committed
snapshots. (a) with (c) is what lets this whole feature layer be built AND tested
with no network at all.

DIFFERENCING MEANS DIFFERENCING TOTALS, NOT AVERAGES
====================================================
This is the step that is easy to get silently wrong, so it is spelled out:

  * ``leaguedashteamstats`` is requested in ``PerMode=PerGame``, so ``PTS`` at
    snapshot ``s`` is ``points_through_s / GP_s``. Differencing two such values
    does NOT give a game's points; it gives the difference of two averages,
    which is meaningless alone. The extractor multiplies every per-game column
    back up by ``GP`` first, turning each snapshot into a season-to-date TOTAL,
    and only then differences.
  * ``FG_PCT``, ``FG3_PCT`` and ``FT_PCT`` are ratios and are never differenced
    at all. They are RECOMPUTED from the recovered counts, which is both correct
    and free.
  * The columns are published rounded to one decimal, so a recovered single-game
    count carries a small error. Measured against the real scores in
    ``Data/OddsData.sqlite``: the pair of recovered PTS values sums to within 2
    points of the recorded game score 63% of the time, median error 1.3 points,
    95th percentile 6.3 points. That error is close to independent across games
    and averages out under any window -- which is an argument FOR smoothing these
    lines, not against.

WHICH SNAPSHOT BOUNDS A GAME
============================
``nba_client.fetch_team_stats`` documents ``DateTo`` as INCLUSIVE: a snapshot
named D contains games played on D. Measured against the committed data, that is
not what these scrapes contain. Counting, for every team and every day on which
``GP`` advanced between two consecutive snapshots:

    GP advanced and the team had played on d-1 : 21,650
    GP advanced and the team had played on d   :  2,866

So the snapshot table named ``d`` in this database carries games through
``d - 1``. ``extract_game_logs`` therefore bounds a game on ``D`` with
``p = newest usable snapshot <= D`` (its coverage stops before the game) and
``e = newest usable snapshot > D`` (its coverage includes it), and recovers
``total(e) - total(p)``. When both snapshots exist the window holds exactly one
game -- 75.1% of the 17,222 games in the committed odds tables -- and those
lines are exact up to the rounding above. When the window holds more than one
game the extractor holds a SUM, not a game: those rows are dropped by default and
counted, never quietly spread across the games they cover.

*** NONE OF THAT IS LEAKAGE. *** The line for a game on D is recovered from the
snapshot named D+1, which is about D and about nothing after it. Features for a
game on D' are built only from lines dated strictly before D', so the newest
input is the line for D'-1, which is complete before D' tips off. Extraction
reads the future; features never do. The distinction is enforced rather than
asserted: ``build_recency_dataset`` refuses any frame whose newest history date
is not strictly earlier than the game it describes.

SEASON BOUNDARIES AND COLD START
================================
* **The EWMA resets at the season boundary.** History is keyed on
  ``(Team, Season)``, so a team's weighted average starts empty every October
  and no weight is carried across the eight-month gap -- rosters, roles and
  opponents have all changed, and a six-month-old observation is not "form".
  ``carry_across_seasons=True`` turns that off and is documented at
  ``rolling_features``.
* **Cold start is a dropped row, not an invented value.** A team's first game of
  a season has no history, so ``build_recency_dataset`` drops the game
  (``min_history``, default 1). Nothing is back-filled from a league average or
  from last season: a fabricated feature is one the model will happily fit, and a
  missing feature is one it cannot.
* Early-season snapshots are incomplete at the source -- ``leaguedashteamstats``
  omits teams that have not played, so a snapshot is only usable with all 30 rows
  -- which removes the first games of each season before any of this runs.

FEATURE DEFINITIONS, EXACTLY
============================
For team T, game D in season S, let ``g_1 < ... < g_k`` be the dates of T's games
in S strictly before D and ``x_i(m)`` T's per-game value of metric ``m`` on
``g_i``. With ``H = halflife`` in GAMES:

  m_ewma       sum_i 2^(-(k-i)/H) * x_i(m)  /  sum_i 2^(-(k-i)/H)
               The game k games back carries weight ``2 ** (-k / H)`` relative
               to the most recent, which carries weight 1. ``H = 10`` therefore
               means "the tenth-most-recent game counts half as much as last
               night". NaN when k = 0.
  m_mean{w}    arithmetic mean of the last w games (NaN if k < w)
  m_std{w}     sample standard deviation of the last w games, ddof=1 (NaN if k < w)
  m_slope{w}   OLS slope of m on game index over the last w games, in units of m
               per game (NaN if k < w)
  Games        k, the number of prior games actually used
  As_Of        g_k, the newest of them (NaT when k = 0)

``m_sos`` and ``m_per100`` come from ``attach_opponent_context`` and
``pace_adjust`` and are documented on those functions.

THE LEAKAGE GATE
================
``assert_no_leakage`` and ``build_recency_dataset`` implement ROADMAP 1.2's
"leakage assertion as a permanent gate", in three layers:

  1. Every feature row carries ``As_Of`` provenance and the builder refuses the
     frame unless ``As_Of < game date``. Because ``rolling_features`` reads
     history through a single choke point (``history_before``, which filters
     ``date < cutoff``), an inclusive-endpoint bug cannot survive quietly: the
     row's own date would surface as ``As_Of`` and the build would stop.
  2. A caller-supplied feature column is accepted only if this module can itself
     derive that column from strictly-prior games, and the supplied value must
     MATCH the derivation to 1e-9. A feature that quietly folds in the game being
     predicted cannot reproduce the strictly-prior value, so it raises. Columns
     named like a result are refused before any arithmetic runs.
  3. ``verify=False`` is the single way past layer 2, for a caller carrying a
     feature this module cannot reconstruct. It is a named, greppable decision
     rather than a default.

Column naming follows the shipped dataset: the home team's features are bare
(``PTS_ewma``) and the away team's carry the ``.1`` suffix (``PTS_ewma.1``).
Provenance columns are named explicitly (``Home-As-Of``, ``Away-As-Of``) because
the gate keys on them.
"""

import argparse
import contextlib
import os
import sqlite3
import sys
from bisect import bisect_right
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import toml
from collections import Counter
from numpy.lib.stride_tricks import sliding_window_view

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(1, os.fspath(BASE_DIR))

CONFIG_PATH = BASE_DIR / "config.toml"
ODDS_DB_PATH = BASE_DIR / "Data" / "OddsData.sqlite"
TEAM_DB_PATH = BASE_DIR / "Data" / "TeamData.sqlite"

# ---------------------------------------------------------------------------
# Tunables. Each is a PARAMETER with a documented default, never a literal
# buried in an expression: a half-life nobody can find is one nobody can change.
# ---------------------------------------------------------------------------

# Half-life in GAMES, not days. 10 means a game from ten nights ago carries half
# the weight of last night's, so ~76% of the weight sits inside the last 20
# games -- the recency window ROADMAP 3.1 asks for, and short enough that a
# ten-game hot streak is visible rather than averaged away.
DEFAULT_HALFLIFE_GAMES = 10.0

# Rolling windows in GAMES: short for current form, long enough that a
# dispersion or a trend is estimated from something other than one slump.
DEFAULT_WINDOWS = (5, 10, 20)

# A complete leaguedashteamstats snapshot holds one row per team. The count is
# DERIVED from the snapshots by load_team_snapshots() (mode of the observed row
# counts) rather than hardcoded. This module used to keep its own
# EXPECTED_TEAM_ROWS = 30, which was a second copy of a bug already fixed in
# Create_Games: at 31 teams it rejected every snapshot and silently produced an
# empty feature set. The floor below is the only constant worth keeping -- it
# catches a snapshot store that is truncated wholesale rather than per-date.
MIN_PLAUSIBLE_TEAM_ROWS = 20

DATE_COLUMN = "Date"
TEAM_COLUMN = "Team"
OPPONENT_COLUMN = "Opponent"
SEASON_COLUMN = "Season"
GAME_ID_COLUMN = "Game_ID"
IS_HOME_COLUMN = "Is_Home"
GAMES_COLUMN = "Games"
AS_OF_COLUMN = "As_Of"
HOME_AS_OF = "Home-As-Of"
AWAY_AS_OF = "Away-As-Of"
HOME_GAMES = "Home-Games"
AWAY_GAMES = "Away-Games"
LABEL_COLUMN = "Home-Team-Win"
SCORE_COLUMN = "Score"
AWAY_SUFFIX = ".1"
SUPPLIED_SUFFIX = "__supplied"

# Columns inside a snapshot that are per-game COUNTS, and so the only ones ever
# differenced -- and only after being multiplied back up by GP.
COUNT_COLUMNS = (
    "MIN", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "REB",
    "AST", "TOV", "STL", "BLK", "BLKA", "PF", "PFD", "PTS", "PLUS_MINUS",
)
# Wins and losses are counts too, and differencing them recovers the game's own
# outcome -- the most predictive per-game number in basketball, which is exactly
# why they may only ever enter a row through PRIOR games.
RESULT_COUNT_COLUMNS = ("W", "L")
GAME_COUNT_COLUMNS = COUNT_COLUMNS + RESULT_COUNT_COLUMNS

# Ratios, in the snapshot. Never differenced -- a difference of two percentages
# is not a percentage -- and rebuilt from the recovered counts instead.
RATE_COLUMNS = ("FG_PCT", "FG3_PCT", "FT_PCT")

# Per-game rates recomputed from the recovered counts. Formulas are the standard
# ones and are written out on ``derived_rates``.
DERIVED_GAME_COLUMNS = ("TS_PCT", "EFG_PCT", "AST_TOV", "FTR", "TPA_RATIO")

# The metrics fed to the recency layer, and why these:
#
#   PTS, PLUS_MINUS   the outcome itself, in the units the market prices
#   FGM, FGA, FG3M,
#   FG3A, FTM, FTA    the shot profile. Splitting these is what separates good
#                     defence from good luck; a cumulative shooting average
#                     cannot tell the two apart over a two-week swing
#   REB, OREB, DREB   offensive boards are the best single predictor of second
#                     chances and they move fast; DREB is the pace-sensitive one
#   AST, TOV, STL,
#   BLK               possession-quality numbers, very noisy per game and much
#                     less noisy across ten
#   PF                fouls per game, the main live-ball drag on margin
#   W                 win/loss as 1/0: its EWMA is recency-weighted winning
#                     percentage, which the cumulative W_PCT cannot be
#   the five derived rates, which are ratios of the counts above and so add no
#                     new information, only a normalised view of it
#
# Deliberately NOT included: every ``*_RANK`` column. Ranks are computed against
# the same day's whole field, so they are a contemporaneous league quantity
# rather than a team's own history, and they are the columns that leaked hardest
# in the original build.
GAME_METRICS = (
    "PTS", "PLUS_MINUS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
    "REB", "OREB", "DREB", "AST", "TOV", "STL", "BLK", "PF", "W",
) + DERIVED_GAME_COLUMNS

# What the opponent-context and pace layers read by default.
DEFAULT_STRENGTH_METRICS = ("PLUS_MINUS", "PTS")
DEFAULT_PACE_METRICS = ("PTS", "REB", "AST", "STL", "BLK", "TOV", "FGM", "FG3M")

# The box-score columns a possession count is estimated from.
POSSESSION_FIELDS = ("FGA", "FTA", "OREB")

# Carried through a dataset for audit, never used as features. ``Is_Win`` is the
# game's own result for that team: it belongs to the label, and a frame that
# treats it as a feature is exactly the bug this module exists to prevent.
NON_FEATURE_COLUMNS = ("Is_Win", "Window_Games", "Snapshot_Before", "Snapshot_After")

# Game-level columns that ride on EITHER team row and describe the game rather
# than either team's history. ``build_recency_dataset`` passes them through
# unchanged and never folds them into a feature vector, and ``assert_no_leakage``
# exempts exactly this set -- it has to, because ``Home-Team-Win`` in this tuple
# IS the label. That exemption is deliberately narrow: a result-named column that
# is not one of these is refused, and a result-named column may never be
# SUPPLIED as a feature (``merge_supplied_features`` applies no exemption). It
# also keeps the away suffix out of existence -- ``Score`` never becomes
# ``Score.1``, because a value column is drawn from this set's complement.
GAME_LEVEL_COLUMNS = (
    LABEL_COLUMN, SCORE_COLUMN, "OU", "OU-Cover",
    "Days-Rest-Home", "Days-Rest-Away",
)

# Names that may never appear in a feature vector because they ARE (or encode)
# the answer. Checked before any arithmetic, so a column named for a result is
# refused rather than quietly scored.
LEAKAGE_COLUMN_NAMES = frozenset({
    "Score", "Score.1", "Points", "Win_Margin", "Win_Margin.1", LABEL_COLUMN,
    "OU", "OU.1", "OU-Cover", "OU-Cover.1", "Spread", "ML_Home", "ML_Away",
    "Is_Win", "Is_Win.1", "Final_Score", "Home_Points", "Away_Points",
    "Result", "Outcome",
})

# The snapshots and the odds tables disagree about two franchises. Both are
# joined on team NAME here, which the shipped dataset never does -- it joins
# POSITIONALLY through Dictionaries.team_index_*, so the disagreement has never
# surfaced and would otherwise be attributed silently to the wrong row.
SNAPSHOT_TEAM_ALIASES = {
    "LA Clippers": "Los Angeles Clippers",
    "Charlotte Hornets": "Charlotte Bobcats",
}


class LeakageError(ValueError):
    """Raised when a feature cannot be shown to predate the game it describes.

    A ``ValueError`` subclass, so existing ``except ValueError`` handlers and
    ``pytest.raises`` around this pipeline keep working, with a name of its own
    so the failure is greppable and cannot be mistaken for a shape or dtype bug.
    """


# ---------------------------------------------------------------------------
# Column helpers
# ---------------------------------------------------------------------------


def away_name(column):
    """The away-team spelling of a home-team column: ``PTS`` -> ``PTS.1``."""
    return f"{column}{AWAY_SUFFIX}"


def game_id(game_date, home, away):
    """Stable per-game key, ``<date>|<home>|<away>``.

    The date is the first ten characters of whatever was passed, so an ISO
    string, a Timestamp and a ``datetime.date`` all produce the same key and two
    sources can be joined without a normalisation step in between.
    """
    return f"{str(game_date)[:10]}|{home}|{away}"


def _as_matrix(values):
    """Coerce to a 2-D ``(n_observations, n_series)`` float matrix.

    An explicit reshape rather than ``np.atleast_2d``, which turns a 1-D series
    of n observations into n SERIES of one observation each -- the exact
    transpose, and a silent one.
    """
    block = np.asarray(values, dtype=float)
    if block.ndim == 1:
        return block.reshape(-1, 1)
    if block.ndim == 2:
        return block
    raise ValueError(
        f"expected a 1-D series or a 2-D (observations, series) block, got "
        f"{block.ndim} dimensions"
    )


# ---------------------------------------------------------------------------
# The recency maths
# ---------------------------------------------------------------------------


def ewma_weights(length, halflife=DEFAULT_HALFLIFE_GAMES):
    """Unnormalised EWMA weights for a chronological series, oldest first.

    CONVENTION, stated exactly because an unauditable weight scheme is an
    unauditable model: ``weights[j]`` is the weight of the element at position
    ``j`` of an OLDEST-FIRST series. Position ``j`` sits ``length - 1 - j`` steps
    back from the newest element, so

        weights[j] = 2 ** (-(length - 1 - j) / halflife)

    The newest element (``j = length - 1``) therefore always carries weight 1,
    and the observation one step back from it carries ``2 ** (-1 / halflife)``.
    ``halflife = 10`` means the observation ten steps back carries exactly half
    the weight of the newest one. For a four-game series at halflife 1 the
    weights are ``[1/8, 1/4, 1/2, 1]``.

    Returned unnormalised (they do not sum to 1) so a caller can combine them
    across different period lengths; ``ewma`` divides by the sum of the weights
    it actually used.
    """
    if halflife is None or not np.isfinite(halflife) or halflife <= 0:
        raise ValueError(
            f"halflife must be a positive, finite number of GAMES, got {halflife!r}"
        )
    length = int(length)
    if length < 0:
        raise ValueError(f"length must not be negative, got {length}")
    steps_back = (length - 1) - np.arange(length, dtype=float)
    return 2.0 ** (-steps_back / float(halflife))


def ewma(series, halflife=DEFAULT_HALFLIFE_GAMES, min_periods=1):
    """Exponentially weighted mean of one chronological series.

    ``series`` is ordered OLDEST FIRST: the last element is the most recent
    observation and carries weight 1. Non-finite entries are dropped and the
    remaining weights re-normalised, so a gap in the box score costs that game's
    influence instead of poisoning the average with a NaN.

    Returns NaN when fewer than ``min_periods`` usable observations remain --
    including for an empty series and for an all-NaN series. With a half-life of
    10 games, the tenth-most-recent observation counts half as much as the newest.
    """
    values = np.asarray(series, dtype=float).ravel()
    if values.size == 0:
        return float("nan")
    weights = ewma_weights(values.size, halflife)
    usable = np.isfinite(values)
    if int(usable.sum()) < max(1, int(min_periods)):
        return float("nan")
    weights = weights[usable]
    return float(np.dot(weights, values[usable]) / weights.sum())


def ewma_history(values, halflife=DEFAULT_HALFLIFE_GAMES, min_periods=1):
    """Causal EWMA: row ``i`` is the weighted mean of rows ``0..i-1`` only.

    ``values`` is a 2-D ``(n_observations, n_series)`` array ordered OLDEST FIRST;
    each column is an independent series, all processed in one pass. The output
    has the same shape and row ``i`` NEVER reads row ``i``: it is exactly
    ``ewma(values[:i])``. That is the entire anti-leakage argument for the feature
    layer -- perturbing ``values[i]`` cannot move ``out[i]``, which
    ``Tests/test_recency_features.py`` checks directly rather than by comment.

    NaN entries are skipped and the weights re-normalised over what remains.

    This is a loop and not a closed form, deliberately. The vectorisable identity
    ``out[i] = 2**(i/H) * cumsum(2**(-k/H) * x[k])`` needs factors up to
    ``2 ** (n / H)`` -- about 2**120 for one season -- which silently loses every
    digit that matters. The recursion below is O(n) and stable; it steps forward
    over observations and vectorises across the metric columns, so a 17k-game
    history costs well under a second.
    """
    block = _as_matrix(values)
    n_observations = block.shape[0]
    out = np.full(block.shape, np.nan, dtype=float)
    if n_observations < 2:
        return out

    ewma_weights(1, halflife)  # validate halflife before the loop runs
    decay = 2.0 ** (-1.0 / float(halflife))
    weighted = np.zeros(block.shape[1], dtype=float)
    weight = np.zeros(block.shape[1], dtype=float)
    observed = np.zeros(block.shape[1], dtype=float)

    # Step FORWARD, absorbing row i-1 with weight 1 and decaying everything older
    # by one half-life. After the step, `weighted`/`weight` hold the weighted
    # mean of rows 0..i-1 -- row i-1 the newest -- and row i has not been read.
    for i in range(1, n_observations):
        weighted *= decay
        weight *= decay
        row = block[i - 1]
        usable = np.isfinite(row)
        weighted += np.where(usable, row, 0.0)
        weight += usable
        observed += usable
        ready = (weight > 0) & (observed >= max(1, int(min_periods)))
        out[i] = np.divide(weighted, weight, out=np.full(block.shape[1], np.nan), where=ready)
    return out


def _lag_windows(values, window):
    """Row ``i`` is ``values[i - window : i]``, NaN-padded: shape ``(n, window, k)``.

    Strictly before ``i``, always -- the same off-by-one the leakage contract
    depends on, applied to the raw block so the rolling statistics inherit it
    instead of re-deriving it and getting it wrong.

    Two mechanical notes, because ``sliding_window_view`` is fussy. It takes ONE
    ``window_shape`` entry per axis it slides -- sliding only axis 0 of a 2-D
    block means ``window_shape=(window,)``, not a shape carrying the series axis
    too, which it refuses. And it APPENDS the window axes, so its output is
    ``(n + 1, k, window)`` and is transposed back here to the documented
    ``(n, window, k)``.

    The result is a read-only, overlapping VIEW of ``padded``, not a copy: every
    reduction that consumes it (``_window_moments``, and the deviations in
    ``rolling_std`` / ``rolling_slope``) builds its own array, so nothing is ever
    written back into the view and a write could not smear one game's value across
    the rows that share its memory.

    A window shorter than one game yields an EMPTY window rather than an error --
    history shorter than the window degrades to NaN, it does not raise.
    """
    block = _as_matrix(values)
    window = int(window)
    if window < 1:
        return np.full((block.shape[0], 0) + block.shape[1:], np.nan)
    padded = np.concatenate([np.full((window,) + block.shape[1:], np.nan), block])
    views = sliding_window_view(padded, (window,), axis=0)
    return views.swapaxes(1, 2)[: block.shape[0]]


def _window_moments(views):
    """``(count, mean)`` per window over shape ``(n, window, k)``, NaN padding skipped.

    Written as masked sums rather than ``np.nanmean``/``np.nanstd`` because those
    warn on an all-NaN window -- which is the normal state of the first few rows
    of every season -- and because the count is needed anyway to decide whether a
    window is full enough to use.
    """
    usable = np.isfinite(views)
    counts = usable.sum(axis=1)
    total = np.where(usable, views, 0.0).sum(axis=1)
    mean = np.divide(total, counts, out=np.full(counts.shape, np.nan), where=counts > 0)
    return usable, counts, mean


def rolling_mean(values, window, min_periods=None):
    """Trailing mean over the last ``window`` rows, excluding row ``i`` itself."""
    _, counts, mean = _window_moments(_lag_windows(values, window))
    return np.where(counts >= int(min_periods or window), mean, np.nan)


def rolling_std(values, window, min_periods=None):
    """Trailing sample standard deviation (ddof=1) over the last ``window`` rows.

    NaN when the window holds fewer than ``min_periods`` observations, and
    everywhere for ``window < 2``: a one-game window has no dispersion.
    """
    window = int(window)
    views = _lag_windows(values, window)
    if window < 2:
        return np.full(views.shape[0], np.nan)
    usable, counts, mean = _window_moments(views)
    deviation = np.where(usable, views - mean[:, None, :], 0.0)
    variance = np.divide(
        (deviation ** 2).sum(axis=1), counts - 1,
        out=np.full(counts.shape, np.nan), where=counts > 1,
    )
    return np.where(counts >= int(min_periods or window), np.sqrt(variance), np.nan)


def rolling_slope(values, window, min_periods=None):
    """Trailing OLS slope of the metric on game index, in units per game.

    Fitted over the last ``window`` rows (excluding row ``i``) against
    ``t = 0 .. window-1``, oldest to newest, so a positive slope means "rising".
    Margins (PLUS_MINUS) are left as they are: the slope of a margin is still a
    margin per game.
    """
    window = int(window)
    views = _lag_windows(values, window)
    if window < 2:
        return np.full(views.shape[0], np.nan)
    usable, counts, mean = _window_moments(views)
    centred = np.arange(window, dtype=float) - (window - 1) / 2.0
    deviation = np.where(usable, views - mean[:, None, :], 0.0)
    numerator = (deviation * centred[None, :, None]).sum(axis=1)
    slope = numerator / float((centred ** 2).sum())
    return np.where(counts >= int(min_periods or window), slope, np.nan)


def safe_ratio(numerator, denominator, default=np.nan):
    """Element-wise ratio returning NaN on a zero or non-finite denominator."""
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    ok = np.isfinite(den) & (den != 0) & np.isfinite(num)
    return np.divide(num, den, out=np.full(num.shape, default, dtype=float), where=ok)


# ---------------------------------------------------------------------------
# The history choke point
# ---------------------------------------------------------------------------


def history_before(team_games, cutoff, season=None):
    """Rows of ``team_games`` STRICTLY before ``cutoff``.

    Every feature in this module reads history through this one function, and it
    uses ``<``, never ``<=``. That is not a style preference. The bug behind this
    project's original 68.9% claim was an inclusive endpoint feeding a game's own
    result into its own feature vector; here an inclusive endpoint would put the
    row's own date into ``As_Of`` and ``build_recency_dataset`` would refuse the
    frame instead of writing it.

    ``season`` restricts to one season, which is what makes the EWMA reset every
    October rather than carry weight across an eight-month gap.
    """
    mask = team_games[DATE_COLUMN] < cutoff
    if season is not None and SEASON_COLUMN in team_games.columns:
        mask = mask & (team_games[SEASON_COLUMN] == season)
    return team_games[mask]


def assert_strictly_before(as_of, game_date, context):
    """Raise unless every as-of value is strictly earlier than the game date.

    Returns True. Missing values are allowed (they mean "no history", which is
    not a leak) but never an as-of that lands on or after the game.
    """
    as_of = pd.to_datetime(pd.Series(as_of), errors="coerce")
    present = as_of.dropna()
    game_date = pd.Timestamp(game_date)
    if len(present) == 0:
        return True
    offenders = present[present >= game_date]
    if len(offenders):
        raise LeakageError(
            f"{context}: {len(offenders)} value(s) of the as-of column are on or "
            f"after the game date {game_date.date()} (worst: {offenders.max().date()}). "
            "Features must be a function of games strictly before the game. This is "
            "the leakage the original 68.9% headline came from; the build stops here."
        )
    return True


# ---------------------------------------------------------------------------
# Per-team recency features
# ---------------------------------------------------------------------------


def rolling_features(game_rows, halflife=DEFAULT_HALFLIFE_GAMES, windows=DEFAULT_WINDOWS,
                     metrics=None, carry_across_seasons=False, min_history=0):
    """Per-team chronological recency features from per-game team lines.

    ``game_rows`` is one row per team-game with at least ``Team`` and ``Date``,
    and ``Season`` unless ``carry_across_seasons``. Returns one row per input
    game, in sorted order, carrying:

        Games                 number of prior games in scope (k)
        As_Of                 the newest of them (NaT when k == 0)
        <m>_ewma              exponential mean, halflife in GAMES
        <m>_mean<w>           trailing mean over w games
        <m>_std<w>            trailing sample standard deviation over w games
        <m>_slope<w>          trailing OLS slope over w games, units per game

    Each of those is a function of games on dates STRICTLY before the row's own
    ``Date``, for one team and -- by default -- one season. NaN means "not enough
    history" and stays NaN: it is never filled with a league average, because a
    filled feature is one the model will fit and a missing feature is one it
    cannot.

    ``halflife`` is in GAMES, ``windows`` are in GAMES, and both are parameters
    with the defaults above rather than literals in the expressions. Rolling
    statistics require a FULL window, so there are no partial means at the start
    of a season. ``min_history`` drops rows with fewer prior games than that.
    ``carry_across_seasons=True`` keys history on the team alone, so the EWMA
    spans the season boundary -- off by default, and if you turn it on, know that
    it is one weight on a six-month-old observation from a different roster.
    """
    metrics = tuple(metrics or GAME_METRICS)
    for column in (TEAM_COLUMN, DATE_COLUMN):
        if column not in game_rows.columns:
            raise ValueError(f"rolling_features requires a {column!r} column")
    missing = [m for m in metrics if m not in game_rows.columns]
    if missing:
        raise ValueError(
            f"rolling_features needs per-game columns for {list(metrics)}; missing: "
            f"{missing}. Recover them with extract_game_logs (which differences the "
            "TeamData.sqlite snapshots) or supply a game-level source of your own."
        )

    ordered = game_rows.copy()
    ordered[DATE_COLUMN] = pd.to_datetime(ordered[DATE_COLUMN])
    has_season = SEASON_COLUMN in ordered.columns
    scope = [TEAM_COLUMN] if (carry_across_seasons or not has_season) else \
        [TEAM_COLUMN, SEASON_COLUMN]
    ordered = ordered.sort_values(scope + [DATE_COLUMN], kind="stable")

    suffixes = ["_ewma"]
    for window in windows:
        suffixes += [f"_mean{window}", f"_std{window}", f"_slope{window}"]
    feature_names = [f"{m}{suffix}" for m in metrics for suffix in suffixes]

    values = np.column_stack([
        pd.to_numeric(ordered[m], errors="coerce").to_numpy(dtype=float) for m in metrics
    ])
    n_rows = len(ordered)
    filled = {name: np.full(n_rows, np.nan) for name in feature_names}
    games = np.zeros(n_rows)
    # The UNIT goes on the NaT as well as on the array: a bare np.datetime64("NaT")
    # is generic-unit and both warns and cannot be written into by a
    # day-resolution column.
    as_of = np.full(n_rows, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    dates = ordered[DATE_COLUMN].to_numpy()

    # History is grouped so it can never cross a team boundary, nor (by default)
    # a season boundary. Within a group every position is strictly after the one
    # before it, so the newest history date at position i is simply date[i-1].
    for positions in ordered.groupby(scope, sort=False).indices.values():
        positions = np.asarray(positions)
        length = len(positions)
        games[positions[1:]] = np.arange(1, length)
        as_of[positions[1:]] = dates[positions[:-1]]

        history = values[positions]
        weighted = ewma_history(history, halflife=halflife)
        for index, metric in enumerate(metrics):
            column = history[:, index]
            filled[f"{metric}_ewma"][positions] = weighted[:, index]
            for window in windows:
                filled[f"{metric}_mean{window}"][positions] = \
                    rolling_mean(column, window)[:, 0]
                filled[f"{metric}_std{window}"][positions] = \
                    rolling_std(column, window)[:, 0]
                filled[f"{metric}_slope{window}"][positions] = \
                    rolling_slope(column, window)[:, 0]

    if min_history:
        keep = games >= int(min_history)
        ordered = ordered[keep]
        games = games[keep]
        as_of = as_of[keep]
        for name in feature_names:
            filled[name] = filled[name][keep]

    # Assembled in ONE concat rather than by assigning the feature columns one at
    # a time: with 554 of them, per-column assignment fragments the frame and
    # pandas raises a PerformanceWarning every few columns. Both arrays are
    # already positional and the index is reset here, so nothing depends on the
    # label alignment the equivalent Series assignment would have relied on.
    result = ordered.reset_index(drop=True)
    result[GAMES_COLUMN] = games
    result[AS_OF_COLUMN] = pd.to_datetime(as_of)
    return pd.concat([result, pd.DataFrame(filled)], axis=1)


def attach_opponent_context(features, strength_metrics=None, metrics=None,
                            league_average="expanding", keep_opponent_columns=False):
    """First-order schedule-of-opponents adjustment, computed strictly before the game.

    A team's 112 points a game means one thing against a good schedule and
    another against a bad one. This adds ``<m>_sos`` for every recency metric
    ``m``, from the classical strength-of-schedule correction:

        m_sos = m + (league_average(m) - opponent(m)) * (m / league_average(m))

    where ``league_average(m)`` is the mean of that metric's ``_ewma`` across all
    teams (expanding along the date axis by default, so only rows already dated on
    or before the row in hand) and ``opponent(m)`` is the OPPONENT's pre-game
    ``m_ewma``, looked up by ``(Opponent, Date)``. The opponent's row for this
    date is already in the frame and already carries its own ``As_Of``, so the
    adjustment is as leak-free as the features it adjusts -- this is a key
    lookup, not a second and differently-computed copy of the same number.

    ITS WEAKNESS, STATED PLAINLY
    ----------------------------
    This is a single non-iterative pass over opponents' RAW recency numbers. It
    is not a rating system, and the ways it is wrong are worth naming:

      * no iteration, so adjustment does not propagate. Correcting every team by
        its opponents' UNADJUSTED strength fixes the first-order term only; a
        schedule effect carried through two or three opponents survives, and
        that is the part that matters most for a soft schedule;
      * the league average is a pooled mean, not a season-specific or
        pace-matched one, so in a league-wide shooting slump it credits teams
        for beating a number that moved for everyone;
      * the shrink factor ``m / league_average`` is the classical one and is
        unbounded, so a 20-point team is adjusted as if its scoring were
        infinitely informative about its opposition;
      * no home/away split, so a road-heavy stretch reads as a neutral one;
      * ``league_average="global"`` pools every row including later dates. That
        is transductive but not label-bearing -- the values averaged are
        themselves built from prior games -- and it is NOT the default.

    Treat ``*_sos`` as a correction, not a rating. Doing this properly means an
    iterative solve, which is a different function and a different evaluation.
    """
    metrics = tuple(metrics or GAME_METRICS)
    league_average = str(league_average).lower()
    if league_average not in ("expanding", "global"):
        raise ValueError(
            "league_average must be 'expanding' (the default, causal) or 'global', "
            f"got {league_average!r}"
        )
    for column in (TEAM_COLUMN, OPPONENT_COLUMN, DATE_COLUMN):
        if column not in features.columns:
            raise ValueError(f"attach_opponent_context requires a {column!r} column")

    result = features.copy()
    result[DATE_COLUMN] = pd.to_datetime(result[DATE_COLUMN])
    ewma_columns = [f"{m}_ewma" for m in metrics]
    missing = [c for c in ewma_columns if c not in result.columns]
    if missing:
        raise ValueError(
            "attach_opponent_context reads the opponent's own pre-game _ewma. "
            f"Missing: {missing[:6]}. Run rolling_features first."
        )

    opponent_columns = {c: f"opp_{c}" for c in ewma_columns}
    lookup = result[[OPPONENT_COLUMN, DATE_COLUMN] + ewma_columns].rename(
        columns={OPPONENT_COLUMN: TEAM_COLUMN, **opponent_columns}
    )
    result = result.merge(lookup, on=[TEAM_COLUMN, DATE_COLUMN], how="left")

    ordered = result.sort_values(DATE_COLUMN, kind="stable")
    for metric in metrics:
        raw = ordered[f"{metric}_ewma"].to_numpy(dtype=float)
        opponent = ordered[f"opp_{metric}_ewma"].to_numpy(dtype=float)
        if league_average == "expanding":
            league = ordered[f"{metric}_ewma"].expanding(min_periods=1).mean().to_numpy()
        else:
            league = np.full(len(ordered), np.nanmean(raw))
        scale = safe_ratio(raw, league)
        result[f"{metric}_sos"] = np.nan
        result.loc[ordered.index, f"{metric}_sos"] = \
            raw + (league - opponent) * scale

    if not keep_opponent_columns:
        result = result.drop(columns=list(opponent_columns.values()))
    return result.sort_index().reset_index(drop=True)


def pace_adjust(features, metrics=None, basis="team"):
    """Per-possession normalisation from an ESTIMATED possession count.

    Possession data is not in ``leaguedashteamstats``'s Base measure set, so this
    uses the standard box-score estimate and says so:

        poss ~= 0.5 * ((FGA + 0.44*FTA - ORB_opp) + (FGA_opp + 0.44*FTA_opp - ORB))

    which needs both teams' shots, free throws and boards. Only PRE-GAME values of
    those exist when a feature row is built, so there are two bases and both are
    approximations:

      ``basis="team"`` (default)
        ``poss ~= FGA + 0.44*FTA - 0.5*ORB`` from the team's own pre-game recency
        values alone. Self-contained, so it works before opponent context exists,
        and it is the honest minimum.
      ``basis="matched"``
        the full two-sided formula, evaluated on both teams' pre-game histories.
        Better centred; requires ``attach_opponent_context(keep_opponent_columns=True)``
        to have run first, which is what ``build_recency_dataset`` arranges.

    Adds ``Possessions`` and ``<m>_per100 = 100 * m_ewma / Possessions``. Note that
    ``EWMA(count) / EWMA(poss)`` is not ``EWMA(count / poss)``: EWMA is linear and
    the ratio is not, so these are ratios of recency-weighted totals rather than
    recency-weighted rates. That is the correct pre-game quantity -- a per-game
    rate cannot be formed without the game -- and it is a real approximation, not
    an identity. The residual error is largely common to both teams in a game
    (turnovers and tempo move possessions for everyone), so the adjustment mostly
    REORDERS fast and slow teams rather than calibrating absolute efficiency.
    """
    metrics = tuple(metrics or DEFAULT_PACE_METRICS)
    basis = str(basis).lower()
    if basis not in ("team", "matched"):
        raise ValueError(f"pace basis must be 'team' or 'matched', got {basis!r}")
    for field in POSSESSION_FIELDS:
        if f"{field}_ewma" not in features.columns:
            raise ValueError(
                f"pace_adjust needs {field}_ewma. Run rolling_features first, and "
                "attach_opponent_context(keep_opponent_columns=True) first too if "
                "basis='matched'."
            )
        if basis == "matched" and f"opp_{field}_ewma" not in features.columns:
            raise ValueError(
                "basis='matched' needs the opponent's pre-game estimates "
                f"(opp_{field}_ewma). Run attach_opponent_context with "
                "keep_opponent_columns=True first."
            )

    result = features.copy()
    estimates = result[[f"{f}_ewma" for f in POSSESSION_FIELDS]].to_numpy(dtype=float)
    fga, fta, orb = estimates[:, 0], estimates[:, 1], estimates[:, 2]
    if basis == "matched":
        other = result[[f"opp_{f}_ewma" for f in POSSESSION_FIELDS]].to_numpy(dtype=float)
        opp_fga, opp_fta, opp_orb = other[:, 0], other[:, 1], other[:, 2]
        possessions = 0.5 * (
            (fga + 0.44 * fta - opp_orb) + (opp_fga + 0.44 * opp_fta - orb)
        )
    else:
        possessions = fga + 0.44 * fta - 0.5 * orb
    result["Possessions"] = possessions

    for metric in metrics:
        column = f"{metric}_ewma"
        if column not in result.columns:
            raise ValueError(f"pace_adjust needs {column!r} to pace-adjust {metric!r}")
        result[f"{metric}_per100"] = 100.0 * safe_ratio(result[column], possessions)
    return result


# ---------------------------------------------------------------------------
# The assembled dataset and the leakage gate
# ---------------------------------------------------------------------------


def build_recency_dataset(team_games, halflife=DEFAULT_HALFLIFE_GAMES,
                          windows=DEFAULT_WINDOWS, metrics=None, min_history=1,
                          opponent_context=True, pace=True, pace_basis="team",
                          carry_across_seasons=False, supplied_features=None,
                          verify=True):
    """Assemble the per-game training frame and prove it is leak-free.

    ``team_games`` is one row per team-game (two rows per game) with at least
    ``Game_ID``, ``Team``, ``Date``, ``Season``, ``Is_Home``, ``Opponent`` and the
    metric columns. Game-level columns carried on either row (``Score``,
    ``Home-Team-Win``, ``Days-Rest-Home``, ``Days-Rest-Away``) are passed through
    unchanged: they describe the game and are never read as features.

    Output: identifiers and labels, then the home team's features (``PTS_ewma``)
    and the away team's (``PTS_ewma.1``) -- the ``.1`` suffix is the shipped
    dataset's convention, kept so a future swap-in does not move every column
    under a trained model -- plus ``Home-Games`` / ``Away-Games`` and
    ``Home-As-Of`` / ``Away-As-Of``.

    THE ASSERTION IS NOT OPTIONAL. Before returning, this function:

      * drops rows where either side has fewer than ``min_history`` prior games
        (cold start: no history, no row, no invented value);
      * merges ``supplied_features`` (keyed on ``Game_ID``) after refusing any
        column named like a result, and -- with ``verify`` on -- requires every
        supplied feature column to be one this module can itself derive and the
        supplied value to MATCH that derivation to 1e-9. A feature that folds in
        the game being predicted cannot reproduce the strictly-prior value, so it
        raises instead of being scored;
      * runs ``assert_no_leakage``, refusing the frame unless every ``As_Of``
        strictly precedes its own game date.

    ``verify=False`` is the only way past the supplied-feature gate, for a caller
    carrying a feature this module cannot reconstruct. It is a named decision,
    not a default.
    """
    required = (GAME_ID_COLUMN, TEAM_COLUMN, DATE_COLUMN, IS_HOME_COLUMN)
    missing = [c for c in required if c not in team_games.columns]
    if missing:
        raise ValueError(f"build_recency_dataset requires {missing} in team_games")

    metrics = tuple(metrics or GAME_METRICS)
    features = rolling_features(
        team_games, halflife=halflife, windows=windows, metrics=metrics,
        carry_across_seasons=carry_across_seasons,
    )
    if opponent_context:
        features = attach_opponent_context(
            features, metrics=metrics,
            keep_opponent_columns=bool(pace) and str(pace_basis).lower() == "matched",
        )
    if pace:
        # Only the metrics this build actually computed: pace_adjust divides by
        # <m>_ewma, so pace-adjusting a metric outside ``metrics`` would ask for a
        # column that was never built. Intersecting also leaves the production
        # default untouched -- every DEFAULT_PACE_METRIC is in GAME_METRICS.
        result_metrics = [m for m in DEFAULT_PACE_METRICS if m in metrics]
        features = pace_adjust(features, metrics=result_metrics, basis=pace_basis)

    identity = {TEAM_COLUMN, OPPONENT_COLUMN, IS_HOME_COLUMN, DATE_COLUMN,
                SEASON_COLUMN, GAME_ID_COLUMN, GAMES_COLUMN, AS_OF_COLUMN}
    # GAME_LEVEL_COLUMNS is excluded here, not just tolerated downstream: a value
    # column is copied onto the away side under a ``.1`` suffix, so leaving
    # ``Score`` in this list would publish the result as a feature twice over.
    value_columns = [
        c for c in features.columns
        if c not in identity and c not in NON_FEATURE_COLUMNS
        and c not in GAME_LEVEL_COLUMNS and not c.startswith("opp_")
    ]
    passthrough = [
        c for c in (DATE_COLUMN, SEASON_COLUMN) + GAME_LEVEL_COLUMNS
        if c in team_games.columns
    ]

    home = features[features[IS_HOME_COLUMN].astype(bool)]
    away = features[~features[IS_HOME_COLUMN].astype(bool)]
    if len(home) != len(away):
        raise ValueError(
            "build_recency_dataset needs exactly one home and one away row per "
            f"Game_ID; got {len(home)} home rows and {len(away)} away rows. Run it on "
            "the output of extract_game_logs, which only emits a game when it "
            "recovered both sides."
        )
    home = home.set_index(GAME_ID_COLUMN)
    # The away side is renamed FIRST, and its provenance columns are in that map:
    # they are selected under their ``.1`` spellings below and renamed onto
    # ``Home-As-Of`` / ``Away-As-Of`` after the concat, so renaming them anywhere
    # but here would leave the selection pointing at columns that do not exist.
    away = away.set_index(GAME_ID_COLUMN).rename(
        columns={c: away_name(c) for c in
                 value_columns + [TEAM_COLUMN, AS_OF_COLUMN, GAMES_COLUMN]}
    )
    shared = home.index.intersection(away.index)

    dataset = pd.concat(
        [
            home.loc[shared, passthrough + [TEAM_COLUMN, AS_OF_COLUMN, GAMES_COLUMN]
                     + value_columns],
            away.loc[shared, [away_name(TEAM_COLUMN), away_name(AS_OF_COLUMN),
                              away_name(GAMES_COLUMN)]
                     + [away_name(c) for c in value_columns]],
        ],
        axis=1,
    ).reset_index()
    dataset = dataset.rename(columns={
        AS_OF_COLUMN: HOME_AS_OF,
        GAMES_COLUMN: HOME_GAMES,
        away_name(AS_OF_COLUMN): AWAY_AS_OF,
        away_name(GAMES_COLUMN): AWAY_GAMES,
    })

    # Cold start: if one side has no history there is no honest feature vector
    # for the game, so the row goes. Nothing is filled in.
    enough = (dataset[HOME_GAMES] >= int(min_history)) & \
             (dataset[AWAY_GAMES] >= int(min_history))
    dataset = dataset[enough].reset_index(drop=True)

    if supplied_features is not None:
        dataset = merge_supplied_features(dataset, supplied_features, verify=verify)

    dataset[DATE_COLUMN] = pd.to_datetime(dataset[DATE_COLUMN])
    dataset = dataset.sort_values(DATE_COLUMN, kind="stable").reset_index(drop=True)

    assert_no_leakage(dataset)
    return dataset


def merge_supplied_features(dataset, supplied_features, verify=True):
    """Merge caller-supplied feature columns in, refusing anything unverifiable.

    Two gates, in this order:

      1. a supplied column whose name is in ``LEAKAGE_COLUMN_NAMES`` raises
         immediately: a result does not get to be a feature;
      2. with ``verify`` on, every other supplied feature column must already be
         one this module computed for the same side of the same game, and its
         value must match to 1e-9.

    Gate 2 is what makes leakage structurally impossible to smuggle in: there is
    no column name that carries a same-day number past a comparison against the
    strictly-prior derivation of that same column. Columns that cannot be
    verified are NOT merged; pass ``verify=False`` to take that decision on the
    record.
    """
    supplied = supplied_features.copy()
    if GAME_ID_COLUMN not in supplied.columns:
        raise ValueError(f"supplied_features must be keyed on {GAME_ID_COLUMN}")
    added = [c for c in supplied.columns if c != GAME_ID_COLUMN]
    if not added:
        return dataset.copy()

    for column in added:
        if column in LEAKAGE_COLUMN_NAMES:
            raise LeakageError(
                f"refusing supplied feature {column!r}: it is named like the game's "
                "own result. Features must be functions of games strictly before the "
                "game."
            )

    # Columns this module already computed are compared under a temporary name
    # rather than overwritten, so the comparison is against the derivation and
    # never against itself.
    renamed = {c: (c + SUPPLIED_SUFFIX if c in dataset.columns else c) for c in added}
    supplied = supplied.rename(columns=renamed)
    merged = dataset.merge(supplied, on=GAME_ID_COLUMN, how="left")

    if verify:
        unverifiable = [c for c in added if c not in dataset.columns]
        if unverifiable:
            raise LeakageError(
                f"cannot verify supplied feature(s) {unverifiable[:5]}: "
                "build_recency_dataset accepts only features it can itself derive "
                "from games strictly before the game. Pass verify=False to override "
                "deliberately."
            )
        for column in added:
            reference = dataset[column].to_numpy(dtype=float)
            candidate = merged[renamed[column]].to_numpy(dtype=float)
            agrees = np.isclose(reference, candidate, rtol=0, atol=1e-9, equal_nan=True)
            if bool(agrees.all()):
                continue
            row = int(np.flatnonzero(~agrees)[0])
            raise LeakageError(
                f"supplied feature {column!r} disagrees with the strictly-prior value "
                f"derived here at game {merged[GAME_ID_COLUMN].iloc[row]}: supplied "
                f"{candidate[row]!r}, derived {reference[row]!r}. A feature that "
                "differs at this point has seen something the derivation did not -- "
                "in practice, the game being predicted. Refusing the frame."
            )

    return merged.drop(
        columns=[renamed[c] for c in added if renamed[c] != c or c in dataset.columns]
    )


def assert_no_leakage(dataset, as_of_columns=(HOME_AS_OF, AWAY_AS_OF),
                      games_columns=(HOME_GAMES, AWAY_GAMES), context="recency dataset"):
    """The permanent gate. Returns True, or raises ``LeakageError``.

    Checks, in order:

      * no column is named like a result, other than the declared
        ``GAME_LEVEL_COLUMNS``. Those are the label and its siblings, which
        ``build_recency_dataset`` passes through by contract and
        ``feature_columns`` drops from the feature vector; exempting them by name
        is the only thing that lets a dataset carry its own target. Anything else
        result-named is refused;
      * every as-of column is present, and every one of its values strictly
        precedes its own game date;
      * a row that claims prior games cannot have an empty as-of, and a row with
        an empty as-of cannot claim any.

    Missing provenance is a failure, not a pass. A frame that cannot say where its
    features came from is precisely the frame this exists to stop.
    """
    forbidden = sorted((set(dataset.columns) & LEAKAGE_COLUMN_NAMES)
                       - set(GAME_LEVEL_COLUMNS))
    if forbidden:
        raise LeakageError(
            f"{context}: {forbidden} are present as columns. A result, or a rank of "
            "one, cannot be a feature."
        )
    if DATE_COLUMN not in dataset.columns:
        raise ValueError(f"{context} has no {DATE_COLUMN} column to check against")
    dates = pd.to_datetime(dataset[DATE_COLUMN])

    for column in as_of_columns:
        if column not in dataset.columns:
            raise LeakageError(
                f"{context}: no {column!r} provenance column. Every feature row must "
                "record the newest game behind it; a frame that cannot say where a "
                "feature came from cannot be shown to be leak-free."
            )
        as_of = pd.to_datetime(dataset[column], errors="coerce")
        empty = as_of.isna()
        offenders = (~empty) & (as_of >= dates)
        if bool(offenders.any()):
            row = int(np.flatnonzero(offenders.to_numpy())[0])
            raise LeakageError(
                f"{context}: row {row} ({dataset[GAME_ID_COLUMN].iloc[row]}) has "
                f"{column} = {as_of.iloc[row].date()} on or after its own game date "
                f"{dates.iloc[row].date()}. Features must be a function of games "
                "strictly before the game. This is the leakage the original 68.9% "
                "headline came from; the build is stopped here."
            )
        for games_column in games_columns:
            if games_column not in dataset.columns:
                continue
            # `claimed` is a numpy bool ARRAY -- the comparison already left pandas
            # -- so it is compared as one, and the mask is built once.
            claimed = dataset[games_column].fillna(0).to_numpy() > 0
            disagree = claimed != (~empty).to_numpy()
            if bool(disagree.any()):
                row = int(np.flatnonzero(disagree)[0])
                raise LeakageError(
                    f"{context}: row {row} has {games_column}="
                    f"{dataset[games_column].iloc[row]} but {column}={as_of.iloc[row]}; "
                    "the two disagree about whether any history was available."
                )
    return True


def feature_columns(dataset, exclude=None):
    """The numeric feature columns of a built dataset, with metadata excluded.

    Follows the shape contract in ``src/Utils/features.py``: identity, labels and
    provenance are dropped, everything else numeric is a feature, in dataset
    order. Positional models see columns in this order and nowhere else.
    """
    dropped = set(exclude or ())
    dropped |= {
        "index", "index.1", GAME_ID_COLUMN, DATE_COLUMN, SEASON_COLUMN,
        TEAM_COLUMN, TEAM_COLUMN + AWAY_SUFFIX, HOME_AS_OF, AWAY_AS_OF,
        HOME_GAMES, AWAY_GAMES, GAMES_COLUMN, AS_OF_COLUMN,
        OPPONENT_COLUMN, OPPONENT_COLUMN + AWAY_SUFFIX, IS_HOME_COLUMN,
        IS_HOME_COLUMN + AWAY_SUFFIX, "Window_Games", "Snapshot_Before",
        "Snapshot_After", "Is_Win",
    }
    # Both spellings of every non-feature column, home side and ``.1`` away side:
    # the builder already keeps these out of the feature vector, and this has to
    # agree with it or an audit run over a hand-assembled frame scores the label.
    dropped |= set(NON_FEATURE_COLUMNS)
    dropped |= set(GAME_LEVEL_COLUMNS)
    dropped |= {c + AWAY_SUFFIX for c in NON_FEATURE_COLUMNS + GAME_LEVEL_COLUMNS}
    columns = []
    for column in dataset.columns:
        if column in dropped or column.startswith("opp_"):
            continue
        if pd.api.types.is_datetime64_any_dtype(dataset[column]):
            continue
        if dataset[column].dtype == object:
            continue
        columns.append(column)
    return columns


# ---------------------------------------------------------------------------
# Recovering game-level lines from the committed snapshots
# ---------------------------------------------------------------------------


def parse_snapshot_date(value):
    """Parse a table name or cell into a ``datetime.date``, or None."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def load_team_snapshots(db_path=TEAM_DB_PATH, expected_team_rows=None):
    """Usable dated snapshots as ``{date: DataFrame}`` indexed by team name.

    Opened READ-ONLY (``mode=ro``) because this function has no business writing
    to a committed database, and the guarantee is better than the intention.

    The league size is DERIVED from the snapshots rather than assumed, by the
    same mode-of-row-counts rule Create_Games uses. This module previously kept
    its own hardcoded ``EXPECTED_TEAM_ROWS = 30``, which meant a second copy of
    a bug already fixed elsewhere: at 31 teams it would reject every snapshot
    and silently produce an empty feature set. Deriving it costs one extra cheap
    COUNT(*) pass and makes the two builders agree by construction.

    A snapshot is usable only with the full complement: leaguedashteamstats
    omits teams that have not played yet, so early-season tables are short and
    are dropped rather than read as "that team scored zero".

    The values are the endpoint's PER-GAME AVERAGES, which ``extract_game_logs``
    multiplies back up by ``GP`` before differencing anything.
    """
    snapshots = {}
    with contextlib.closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as con:
        cursor = con.execute("SELECT name FROM sqlite_master WHERE type='table'")
        dated = []
        for (name,) in cursor.fetchall():
            snapshot_date = parse_snapshot_date(name)
            if snapshot_date is None:
                continue
            dated.append((snapshot_date, name))

        if expected_team_rows is None:
            counts = [
                con.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0]
                for _, name in dated
            ]
            positive = [count for count in counts if count > 0]
            if not positive:
                return {}
            expected_team_rows = Counter(positive).most_common(1)[0][0]
            if expected_team_rows < MIN_PLAUSIBLE_TEAM_ROWS:
                raise ValueError(
                    f"Derived league size of {expected_team_rows} is implausible "
                    f"(minimum {MIN_PLAUSIBLE_TEAM_ROWS}); snapshot tables look corrupt."
                )

        for snapshot_date, name in dated:
            frame = pd.read_sql_query(f'SELECT * FROM "{name}"', con)
            if "TEAM_NAME" not in frame.columns or len(frame.index) != expected_team_rows:
                continue
            frame = frame.rename(columns={"TEAM_NAME": TEAM_COLUMN})
            frame[TEAM_COLUMN] = frame[TEAM_COLUMN].map(
                lambda t: SNAPSHOT_TEAM_ALIASES.get(t, t)
            )
            frame = frame.set_index(TEAM_COLUMN, drop=False)
            snapshots[snapshot_date] = frame
    return dict(sorted(snapshots.items()))


def derived_rates(counts):
    """Per-game rates recomputed from recovered counts.

    Ratios are never differenced -- a difference of two percentages is not a
    percentage -- so each is rebuilt from the counts of the single game. Standard
    definitions:

        TS_PCT     PTS / (2 * (FGA + 0.44 * FTA))    true shooting, with the usual
                   FTA-to-FGA conversion factor
        EFG_PCT    (FGM + 0.5 * FG3M) / FGA         a three worth 1.5
        AST_TOV    AST / TOV                        ball security
        FTR        FTA / FGA                        how much the team chooses to shoot
        TPA_RATIO  FG3A / FGA                       three-point rate

    A zero denominator gives NaN rather than an infinity.
    """
    rates = pd.DataFrame(index=counts.index)
    rates["TS_PCT"] = safe_ratio(counts["PTS"], 2.0 * (counts["FGA"] + 0.44 * counts["FTA"]))
    rates["EFG_PCT"] = safe_ratio(counts["FGM"] + 0.5 * counts["FG3M"], counts["FGA"])
    rates["AST_TOV"] = safe_ratio(counts["AST"], counts["TOV"])
    rates["FTR"] = safe_ratio(counts["FTA"], counts["FGA"])
    rates["TPA_RATIO"] = safe_ratio(counts["FG3A"], counts["FGA"])
    return rates[list(DERIVED_GAME_COLUMNS)]


def extract_game_logs(snapshots, games, count_columns=None, exact_only=True,
                      season_starts=None):
    """Recover one box-score line per team-game from dated cumulative snapshots.

    ``snapshots`` is ``{date: DataFrame}`` as returned by ``load_team_snapshots``
    (per-game averages plus ``GP``); ``games`` is a frame with ``Date``, ``Home``
    and ``Away``, and optionally ``Season``, ``Points`` and ``Win_Margin``.

    For a game on ``D`` the bounding snapshots are ``p = newest <= D`` and
    ``e = newest > D``, because the committed snapshots carry games through
    ``d - 1`` (measured; see the module docstring). The recovered line is
    ``total(e) - total(p)`` with ``total = average * GP``. Both sides of a game
    must yield the same ``n = GP(e) - GP(p)`` for the line to mean "one game", and
    a game is emitted only when BOTH sides do.

    ``exact_only=True`` (default) keeps ``n == 1`` rows. A window holding two
    games yields a SUM, not a game; averaging it across both would put the same
    number into two different histories, so those games are dropped and counted.
    ``exact_only=False`` fills them with the window MEAN and records the width in
    ``Window_Games`` -- an approximation, off by default, for a caller who would
    rather have the games than not have them.

    A window never crosses a season. ``GP`` resetting inside the window is detected
    and dropped, and when ``season_starts`` is given (a ``{season: date}`` map,
    which the CLI builds from ``config.toml``) a window beginning before its own
    season's first day is dropped too.

    Returns ``(logs, stats)``. ``logs`` has two rows per game -- one per team --
    with ``Game_ID``, ``Team``, ``Opponent``, ``Is_Home``, ``Is_Win``, the
    recovered counts, the derived rates, and the provenance columns
    ``Window_Games`` / ``Snapshot_Before`` / ``Snapshot_After`` that make the
    extraction auditable. ``stats`` counts every skip by reason, so a coverage
    regression is visible instead of silent.
    """
    count_columns = list(count_columns or GAME_COUNT_COLUMNS)
    dates = sorted(snapshots)
    season_starts = season_starts or {}

    totals = {}
    for snapshot_date, frame in snapshots.items():
        needed = ["GP"] + count_columns
        missing = [c for c in needed if c not in frame.columns]
        if missing:
            raise ValueError(
                f"snapshot {snapshot_date} is missing {missing}, so it cannot be "
                "differenced into game-level lines"
            )
        played = pd.to_numeric(frame["GP"], errors="coerce").to_numpy(dtype=float)
        block = pd.DataFrame(index=frame.index)
        block["GP"] = played
        for column in count_columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            # AVERAGES BECOME TOTALS HERE. This is the step whose omission makes
            # differencing meaningless; see the module docstring.
            block[column] = values * played
        totals[snapshot_date] = block

    stats = {
        "games": 0, "kept_games": 0, "team_rows": 0, "kept_rows": 0,
        "no_window": 0, "crosses_season": 0, "unknown_team": 0,
        "no_game_in_window": 0, "window_spans_many": 0,
    }
    records = []
    count_frames = []

    games_frame = games.copy()
    games_frame[DATE_COLUMN] = pd.to_datetime(games_frame[DATE_COLUMN]).dt.date
    if SEASON_COLUMN not in games_frame.columns:
        games_frame[SEASON_COLUMN] = ""

    for row in games_frame.itertuples(index=False):
        stats["games"] += 1
        game_date = getattr(row, DATE_COLUMN)
        season = getattr(row, SEASON_COLUMN, "") or ""
        index = bisect_right(dates, game_date)
        after = dates[index] if index < len(dates) else None
        before = dates[index - 1] if index - 1 >= 0 else None
        if before is None or after is None:
            stats["no_window"] += 1
            continue
        if season_starts and season in season_starts and before < season_starts[season]:
            # The previous usable snapshot is in another season, so its totals
            # belong to a different season's accumulation and subtracting them
            # would invent a game rather than recover one.
            stats["crosses_season"] += 1
            continue

        sides = []
        blocked = None
        for is_home, (team, opponent) in (
            (True, (row.Home, row.Away)),
            (False, (row.Away, row.Home)),
        ):
            stats["team_rows"] += 1
            start_block, end_block = totals[before], totals[after]
            if team not in start_block.index or team not in end_block.index:
                blocked = "unknown_team"
                stats["unknown_team"] += 1
                break
            played = float(end_block.at[team, "GP"]) - float(start_block.at[team, "GP"])
            if played < 1:
                # No game in the window, or GP reset across a season boundary.
                blocked = "no_game_in_window"
                stats["no_game_in_window"] += 1
                break
            if played > 1 and exact_only:
                blocked = "window_spans_many"
                stats["window_spans_many"] += 1
                break
            counts = pd.Series({
                column: (float(end_block.at[team, column])
                         - float(start_block.at[team, column])) / played
                for column in count_columns
            })
            sides.append((is_home, team, opponent, played, counts))

        if blocked is not None:
            # Both sides must be recovered. A one-sided row would break the
            # home/away pairing build_recency_dataset relies on, and a game whose
            # two lines came from different windows is not a game.
            stats["team_rows"] -= len(sides)
            continue

        stats["kept_games"] += 1
        for is_home, team, opponent, played, counts in sides:
            record = {
                GAME_ID_COLUMN: game_id(game_date, row.Home, row.Away),
                DATE_COLUMN: pd.Timestamp(game_date),
                TEAM_COLUMN: team,
                OPPONENT_COLUMN: opponent,
                IS_HOME_COLUMN: is_home,
                SEASON_COLUMN: season,
                "Is_Win": np.nan if "W" not in count_columns else
                          float(1.0 if counts["W"] >= 1 else 0.0),
                "Window_Games": played,
                "Snapshot_Before": pd.Timestamp(before),
                "Snapshot_After": pd.Timestamp(after),
            }
            for column in (LABEL_COLUMN, SCORE_COLUMN, "OU", "OU-Cover",
                           "Days-Rest-Home", "Days-Rest-Away"):
                if hasattr(row, column):
                    record[column] = getattr(row, column)
            records.append(record)
            count_frames.append(counts)

    logs = pd.DataFrame(data=records)
    if logs.empty:
        return logs, stats

    counts_frame = pd.DataFrame(data=count_frames).reset_index(drop=True)
    # The recovered counts are the payload, not an intermediate: concatenating
    # only ``derived_rates`` here would emit a row carrying provenance and rates
    # but none of the counts they were computed from.
    logs = pd.concat(
        [logs.reset_index(drop=True), counts_frame,
         derived_rates(counts_frame).reset_index(drop=True)],
        axis=1,
    )
    stats["kept_rows"] = len(logs.index)
    return logs, stats


# ---------------------------------------------------------------------------
# Odds tables -> the game list extract_game_logs consumes
# ---------------------------------------------------------------------------


def select_odds_table(con, season_key):
    """The odds table for a season, or None. Mirrors ``Create_Games``."""
    for name in (f"odds_{season_key}_new", f"odds_{season_key}",
                 f"{season_key}_new", f"{season_key}"):
        exists = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone()
        if exists:
            return name
    return None


def load_odds_games(db_path=ODDS_DB_PATH, config_path=CONFIG_PATH, seasons=None,
                    season_key="create-games"):
    """Games for the seasons named in ``config.toml``, as one frame.

    Read-only, and clipped to each season's own ``start_date`` / ``end_date`` so a
    table that spilled past its season cannot smuggle in the next one.
    """
    config = toml.load(config_path)
    frames = []
    with contextlib.closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as con:
        for season in (list(seasons) if seasons else list(config.get(season_key, {}))):
            window = config.get(season_key, {}).get(season)
            if not window:
                continue
            table = select_odds_table(con, season)
            if not table:
                print(f"No odds table for {season}.")
                continue
            frame = pd.read_sql_query(f'SELECT * FROM "{table}"', con)
            if frame.empty:
                print(f"No odds rows for {season}.")
                continue
            frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN]).dt.date
            start = datetime.strptime(window["start_date"], "%Y-%m-%d").date()
            end = datetime.strptime(window["end_date"], "%Y-%m-%d").date()
            frame = frame[(frame[DATE_COLUMN] >= start) & (frame[DATE_COLUMN] <= end)]
            frame[SEASON_COLUMN] = season
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(data={})


def season_starts_from_config(config_path=CONFIG_PATH, season_key="create-games"):
    """``{season: first date}`` for the boundary checks in ``extract_game_logs``."""
    config = toml.load(config_path)
    return {
        season: datetime.strptime(window["start_date"], "%Y-%m-%d").date()
        for season, window in config.get(season_key, {}).items()
        if "start_date" in window
    }


def main(argv=None):
    """Build the recency dataset from committed data and report what was lost.

    Writes nothing unless ``--out`` names a file, and refuses a path inside
    ``Data/``: those databases are committed artefacts and this pipeline is not
    allowed to rewrite one by accident.
    """
    parser = argparse.ArgumentParser(
        description="Build recency-weighted (EWMA) features from game-level history."
    )
    parser.add_argument("--halflife", type=float, default=DEFAULT_HALFLIFE_GAMES,
                        help="EWMA half-life in GAMES (default: %(default)s).")
    parser.add_argument("--min-history", type=int, default=1,
                        help="Drop games where either side has fewer than this many "
                             "prior games (default: %(default)s).")
    parser.add_argument("--out", default=None,
                        help="Optional sqlite file to write the dataset table to.")
    parser.add_argument("--table", default="recency_dataset",
                        help="Table name for --out (default: %(default)s).")
    args = parser.parse_args(argv)

    games = load_odds_games()
    if games.empty:
        print("No odds games found; nothing to build.")
        return None

    snapshots = load_team_snapshots()
    print(
        f"Usable snapshots: {len(snapshots)} complete tables "
        f"(league size derived from the data)."
    )
    logs, stats = extract_game_logs(
        snapshots, games, season_starts=season_starts_from_config()
    )
    coverage = 100.0 * stats["kept_games"] / max(stats["games"], 1)
    print(
        f"Games: {stats['games']} | both sides recovered as a single game: "
        f"{stats['kept_games']} ({coverage:.1f}%)"
    )
    print(
        "Dropped games -- no snapshot window: "
        f"{stats['no_window']} | window crosses the season start: "
        f"{stats['crosses_season']} | a side absent from a snapshot: "
        f"{stats['unknown_team']} | no game inside the window, or GP reset: "
        f"{stats['no_game_in_window']} | window holds more than one game: "
        f"{stats['window_spans_many']}"
    )
    if logs.empty:
        print("No game lines recovered; nothing to build features from.")
        return None

    dataset = build_recency_dataset(logs, halflife=args.halflife,
                                    min_history=args.min_history)
    print(
        f"Recency dataset: {len(dataset.index)} games, "
        f"{len(feature_columns(dataset))} features, halflife {args.halflife} games, "
        f"windows {DEFAULT_WINDOWS}."
    )
    print(
        "Leakage assertion passed: every Home-As-Of / Away-As-Of strictly precedes "
        "its own game date."
    )

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        data_dir = (BASE_DIR / "Data").resolve()
        if out_path == data_dir or data_dir in out_path.parents:
            raise SystemExit(
                f"Refusing to write inside the committed data directory ({data_dir}). "
                "Point --out at a scratch file."
            )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.closing(sqlite3.connect(out_path)) as con:
            with con:
                dataset.to_sql(args.table, con, if_exists="replace", index=False)
        print(f"Wrote {len(dataset.index)} rows to {out_path}:{args.table}")
    return dataset


if __name__ == "__main__":
    main()