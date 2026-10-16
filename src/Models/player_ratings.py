"""Player-based ratings: build per-player impact, sum rating x EXPECTED minutes.

This is the model the project spec calls "likely your biggest edge", so it is
built to be *falsifiable* rather than to look good. Three things it does that a
naive player model does not:

1. **Every rating is fitted strictly before the game it scores.** A rating that
   has seen the game it is predicting is not a rating. Two separate leaks are
   closed and both are asserted in ``Tests/test_player_ratings.py``:

   * the *rating weights* (a ridge fit on team net rating) are refitted on each
     walk-forward training block, never on the block being scored;
   * the *player statistics* behind each rating are exponentially weighted
     means over that player's own games **strictly before** the current one, so
     a player's rating for game t never contains his game-t box score. Using
     the game-t box score is the most common way a "player model" silently
     cheats, because the box score is the most convenient input available.

2. **Minutes are EXPECTED, not actual.** See "The availability gap" below.

3. **The incremental result is reported with a bootstrap CI, whatever it is.**
   A previous RAPM attempt in this repo scored AUC 0.6969 standalone and added
   *nothing* on top of the team feature set (delta -0.01000, CI
   [-0.02407, +0.00494]). That is the honest prior. This module exists to
   settle whether a correctly specified player model is an edge or a null.

WHAT THIS IS NOT
----------------
**This is not true RAPM.** Regular Adjusted Plus-Minus needs stint-level
on-court/off-court splits (``data/nba/rotations/`` GameRotation). That directory
is empty for this dataset -- the downloader in this run does not request it --
so no stint data exists. What is fitted instead is documented in
:class:`PlusMinusRatings`: a plus-minus rating built from
``LeagueDashPlayerGameLog`` PLUS_MINUS, which is a genuine *on-court* team
plus-minus (it varies within a team-game, verified) but is not stint-resolved
and carries no opponent-strength or location adjustment of its own. Every
number produced here should be read as "a plus-minus-based approximation",
never as RAPM.

THE AVAILABILITY GAP
--------------------
The dataset records a player-game row only when the player PLAYED. There is no
pre-game injury report, and no ``summary/`` directory either. So a "player
model" can be built two ways and they are not the same model:

* **actual minutes** -- weight each player by the minutes they actually logged.
  This knows the answer to "who is playing tonight", which is not knowable
  before tip-off. It is target leakage dressed as a feature.
* **expected minutes** -- weight each player by ``P(plays) * E[min | plays]``,
  where the availability probability is estimated from pre-game information
  only (recent minutes, appearances, days since last game, rest, 3-in-4). A
  player expected to be out still gets weight if he is expected to be a
  rotation player; that weight is exactly the modelling error a real bettor
  cannot avoid.

:func:`availability_gap` measures the distance between the two. That distance
is the cost of the availability problem, and it is the number worth quoting --
the standalone AUC is not.

SIGN CONVENTION
---------------
:func:`src.DataProviders.nba_data.paired_bootstrap_delta` returns
``loss(a) - loss(b)``. **A positive delta means ``b`` is better.** Every delta
in this module's docstrings and in :func:`availability_gap` /
:func:`incremental_test` follows that convention and says so at the call site.

LITERATURE
----------
Vandenbruaene et al. (2022, *J Sports Econ* 23(7)) surveyed 628 published NBA
prediction implementations and never rejected unprofitability once corrected
for multiple testing (their hurdle is ``|z| > 3``). Nothing in this module
should be described as profitable, and a standalone AUC of 0.70 is well inside
the range that study shows is reachable by chance across 628 attempts.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from src.DataProviders import nba_data as D

__all__ = [
    "RatingConfig",
    "BoxScoreRatings",
    "PlusMinusRatings",
    "ExpectedMinutes",
    "PlayerPanel",
    "build_panel",
    "run_walk_forward",
    "availability_gap",
    "incremental_test",
    "team_feature_matrix",
]


# --------------------------------------------------------------------------- #
# Feature lists
# --------------------------------------------------------------------------- #

#: PlayerGameLogs Advanced -- on-court, already pace-adjusted, no future leak
#: because these are per-game rows and every value is lagged before use.
ADVANCED_FEATURES = (
    "TS_PCT", "USG_PCT", "EFG_PCT", "AST_PCT", "REB_PCT", "TM_TOV_PCT",
)

#: PlayerGameLogs Usage -- share of team production while the player was on
#: court. Scale-free, so they need no pace correction.
USAGE_FEATURES = (
    "PCT_AST", "PCT_REB", "PCT_TOV", "PCT_FTA", "PCT_FG3A", "PCT_BLK",
)

#: The box-score block. Used by both rating methods; ``PlusMinusRatings`` adds
#: the trailing plus-minus term on top.
BOX_FEATURES = ADVANCED_FEATURES + USAGE_FEATURES

#: The trailing plus-minus term, per 48 minutes of on-court time. The trailing
#: (lagged) form is `t_pm48`; see `_f`.
PM_FEATURE = "pm48"

#: League-average pace used when a team has no trailing pace yet (first few
#: games of a season). 99 is the long-run NBA figure and this only affects rows
#: in the opening games of October.
FALLBACK_PACE = 99.0

#: Home-court advantage in points. Fixed, not fitted, so it cannot absorb
#: signal from the same season it is evaluated on.
D_HCA_POINTS = 2.2

#: Clamp on trailing plus-minus per 48 minutes. See `load_player_box`.
PM48_CLIP = 60.0


def _f(name: str) -> str:
    """Name of the *trailing* (strictly-before) version of a player feature."""
    return f"t_{name}"


TRAILING_BOX = tuple(_f(f) for f in BOX_FEATURES)
TRAILING_PM = _f(PM_FEATURE)

#: Columns of the candidate panel that are known before tip-off.
MINUTES_FEATURES = (
    "t_MIN", "t_MIN_last5", "t_apps", "days_since_app", "team_games_since",
    "rest_days", "in_3_of_4", "is_home", f"t_USG_PCT",
)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def _paths(pattern: str) -> list[str]:
    return sorted(glob.glob(os.path.join(D.DATA_ROOT, pattern)))


def _parse_iso(s: pd.Series) -> pd.Series:
    """player gamelogs carry `2024-04-14T00:00:00`; the box logs carry `2024-04-14`."""
    s = s.astype(str)
    return pd.to_datetime(s.str.slice(0, 10), format="%Y-%m-%d", errors="coerce")


@lru_cache(maxsize=None)
def load_plus_minus() -> pd.DataFrame:
    """Per-game ON-COURT plus-minus, from ``game_logs/league_player_*.csv``.

    ``LeagueDashPlayerGameLog.PLUS_MINUS`` is the player's own on-court team
    differential, not the team's game total: within a single team-game the
    values differ across players (verified in this dataset -- 0% of team-games
    give all players the same value). That is the signal a plus-minus rating
    needs, but see the module docstring: with no stint data this is a
    plus-minus *approximation*, not RAPM.

    This table is not wrapped by ``src.DataProviders.nba_data``, so the three
    documented traps are applied here explicitly: ``GAME_ID`` is read as ``str``,
    the date is parsed from the ISO prefix, and only the 30 real franchises and
    regular season are kept.
    """
    files = _paths("game_logs/league_player_*.csv")
    if not files:
        raise FileNotFoundError(
            f"no game_logs/league_player_*.csv under {D.DATA_ROOT}")
    df = pd.concat([pd.read_csv(f, dtype={"GAME_ID": str}, low_memory=False)
                    for f in files], ignore_index=True)
    df["GAME_DATE"] = _parse_iso(df["GAME_DATE"])
    df = df[df["TEAM_ABBREVIATION"].isin(D.REAL_TEAMS)].copy()
    if "SEASON_TYPE" in df.columns:
        df = df[df["SEASON_TYPE"] == "Regular Season"].copy()
    return (df[["GAME_ID", "PLAYER_ID", "TEAM_ABBREVIATION", "MIN", "PLUS_MINUS"]]
            .drop_duplicates(subset=["GAME_ID", "PLAYER_ID"])
            .reset_index(drop=True))


@lru_cache(maxsize=None)
def load_player_box() -> pd.DataFrame:
    """One row per player per game: MIN, box-score shares, and on-court +/-.

    Joins PlayerGameLogs Advanced + Usage (via ``nba_data``) to the plus-minus
    table. Rows exist only for players who PLAYED -- that absence is the
    availability problem this module is built around, not an oversight.
    """
    adv = D.load_player_advanced()[[
        "GAME_ID", "PLAYER_ID", "TEAM_ABBREVIATION", "GAME_DATE", "MIN",
        *ADVANCED_FEATURES]]
    # `nba_data._read_measures` does NOT parse dates (only `load_team_games`
    # does), so this table still carries the raw `2024-04-14T00:00:00` string.
    adv = adv.assign(GAME_DATE=_parse_iso(adv["GAME_DATE"]))
    usg = D.load_player_usage()[[
        "GAME_ID", "PLAYER_ID", *USAGE_FEATURES]].drop_duplicates(
        subset=["GAME_ID", "PLAYER_ID"])
    df = adv.merge(usg, on=["GAME_ID", "PLAYER_ID"], how="left")
    pm = (load_plus_minus()
          .drop(columns=["TEAM_ABBREVIATION", "MIN"])
          .rename(columns={"PLUS_MINUS": "pm"}))
    df = df.merge(pm, on=["GAME_ID", "PLAYER_ID"], how="left")
    # A row exists only because the player played; treat <1 minute as a
    # non-appearance so that "played" means the same thing everywhere.
    df["MIN"] = df["MIN"].where(df["MIN"] >= 1.0, 0.0)
    df["played"] = (df["MIN"] > 0).astype(int)
    # Trailing-time rate. A one-minute appearance in a 40-point blowout can
    # carry |PLUS_MINUS| > 30, which is real on-court data but not a rate worth
    # extrapolating to 48 minutes, so the ratio is clipped. Real bench players
    # sit far inside this bound; the clamp only touches garbage-time rows.
    pm48 = np.where(df["MIN"] > 0,
                    df["pm"] * 48.0 / df["MIN"].replace(0, np.nan), 0.0)
    df["pm48"] = np.clip(np.nan_to_num(pm48, posinf=0.0, neginf=0.0),
                         -PM48_CLIP, PM48_CLIP)
    return df.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Trailing (strictly-before) player state
# --------------------------------------------------------------------------- #

def trailing_player_state(box: pd.DataFrame, halflife: float = 8.0,
                          min_appearances: int = 1) -> pd.DataFrame:
    """Attach each player's strictly-prior EWMA state to their game rows.

    The shift-then-weight pattern is the one ``nba_data._ewm_by_team``
    documents and sanctions: shift WITHIN each player first (so player A's
    first game does not inherit player B's state), then weight. Nothing in the
    output can depend on the row it sits on.

    ``t_MIN_last5`` is the unweighted mean of the player's last five PREVIOUS
    minutes; it is a deliberately short window next to the EWMA, because minutes
    respond to a suspension or a new role faster than an EWMA with an 8-game
    halflife can.

    The reset_index is load-bearing. `pd.concat(..., axis=1)` aligns on the
    index, so concatenating a `reset_index(drop=True)` frame against EWMA
    results that kept the shuffled sort index silently interleaves two players'
    histories: every trailing column comes out numerically plausible and
    completely wrong. The assertion at the bottom is the guard against that
    coming back.
    """
    df = (box.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"])
              .reset_index(drop=True))
    keys = df["PLAYER_ID"]
    frames = []
    for col in (*BOX_FEATURES, "MIN", "pm48"):
        shifted = df.groupby("PLAYER_ID", sort=False)[col].shift(1)
        ewm = (shifted.groupby(keys, sort=False)
               .ewm(halflife=halflife, adjust=True).mean()
               .reset_index(level=0, drop=True))
        frames.append(ewm.rename(_f(col)))
    out = pd.concat([df, *frames], axis=1)

    out["t_apps"] = df.groupby("PLAYER_ID", sort=False).cumcount().astype(float)
    prev_min = df.groupby("PLAYER_ID", sort=False)["MIN"].shift(1)
    prev_min = prev_min.where(prev_min > 0)      # a 0-minute row is a DNP
    out["t_MIN_last5"] = (prev_min.groupby(keys, sort=False)
                          .rolling(5, min_periods=1).mean()
                          .reset_index(level=0, drop=True))
    # Before a player has any history there is no rating to report.
    thin = out["t_apps"] < min_appearances
    trail = [c for c in out.columns if c.startswith("t_") and c != "t_apps"]
    out.loc[thin, trail] = np.nan

    # Invariants that a mis-aligned concat would break.
    first = out["t_apps"] == 0
    assert (out.loc[first, "t_MIN"].isna()).all(), \
        "a player's first appearance has trailing minutes; state is misaligned"
    assert (out.loc[~first, "t_MIN"].notna()).all(), \
        "a player with history has no trailing minutes"
    return (out.sort_values(["GAME_DATE", "GAME_ID", "PLAYER_ID"])
              .reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Candidate panel: who COULD play, with what pre-game information
# --------------------------------------------------------------------------- #

@dataclass
class PlayerPanel:
    """The pre-game decision surface: one row per (team, game, candidate).

    A *candidate* is a player who appeared for that team in the previous
    ``pool_window`` team-games. Nobody who has never appeared for the team is a
    candidate -- a player who has been on the roster for two years and played
    once does not get pre-game weight, and neither does a rookie whose debut is
    unknowable in advance. This is a deliberate, stated approximation of "who
    will be in the rotation tonight", and it is the honest side of the
    availability gap: it still cannot see injuries.
    """
    rows: pd.DataFrame          # candidate rows
    team_games: pd.DataFrame    # one row per (GAME_ID, team), the aggregation key


@lru_cache(maxsize=None)
def _cached_state(halflife: float) -> pd.DataFrame:
    """Trailing player state for the whole dataset, cached per halflife.

    Only ever called with the default arguments in this repo; the cache exists
    so that `report()`, `availability_gap()` and `incremental_test()` can each
    ask for the panel without paying for the 14 exponentially weighted means
    every time.
    """
    return trailing_player_state(load_player_box(), halflife=halflife)


def build_panel(games: pd.DataFrame, pool_window: int = 10,
                halflife: float = 8.0) -> PlayerPanel:
    """Build the candidate panel. Every column is strictly before its game."""
    box = _cached_state(halflife).copy()
    keep_ids = set(games["GAME_ID"])
    box = box[box["GAME_ID"].isin(keep_ids)].copy()
    if box.empty:
        raise ValueError("no player game rows overlap the requested games")

    # ---- team-game index, ordered by date so "previous" is well defined ----
    tg = (box[["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION"]]
          .drop_duplicates()
          .rename(columns={"TEAM_ABBREVIATION": "team"})
          .sort_values(["team", "GAME_DATE", "GAME_ID"])
          .reset_index(drop=True))
    tg["team_game_rank"] = tg.groupby("team").cumcount()
    rank_of = dict(zip(zip(tg["GAME_ID"], tg["team"]), tg["team_game_rank"]))

    # ---- candidate pairs -------------------------------------------------
    # A player who appeared at team-game ranks r_1 < r_2 < ... is a candidate
    # for every rank in (r_j, r_j + pool_window], because their most recent
    # appearance at that point is within the window.
    app = box[["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "PLAYER_ID"]].copy()
    app["team_game_rank"] = pd.Series(
        [rank_of[(gid, tm)] for gid, tm
         in zip(app["GAME_ID"], app["TEAM_ABBREVIATION"])],
        index=app.index, dtype=int)
    app = app.dropna(subset=["team_game_rank"])
    app = app.drop_duplicates(subset=["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID"])
    # `tg` is sorted by (team, date, id), so within a team the row order IS the
    # rank order and a candidate rank maps to a row position by indexing. Doing
    # it this way instead of a boolean mask per group turns an O(groups x rows)
    # scan into an O(rows) take, which is the difference between a 97-second
    # panel build and a couple of seconds.
    pos = {t: g.index.to_numpy() for t, g in tg.groupby("team", sort=False)}
    gid = tg["GAME_ID"].to_numpy()
    gdate = tg["GAME_DATE"].to_numpy()
    rank = tg["team_game_rank"].to_numpy()
    n_by_team = {t: len(p) for t, p in pos.items()}
    parts_pid, parts_team, parts_gid, parts_date, parts_rank = [], [], [], [], []
    for (team, pid), sub in app.groupby(
            ["TEAM_ABBREVIATION", "PLAYER_ID"], sort=False):
        n = n_by_team[team]
        ranks = np.unique(sub["team_game_rank"].to_numpy())
        cand = np.unique(np.concatenate(
            [np.arange(r + 1, min(r + 1 + pool_window, n)) for r in ranks]))
        if cand.size == 0:
            continue
        sel = pos[team][cand]
        k = len(sel)
        parts_pid.append(np.full(k, pid))
        parts_team.append(np.full(k, team))
        parts_gid.append(gid[sel])
        parts_date.append(gdate[sel])
        parts_rank.append(rank[sel])
    cand = pd.DataFrame({
        "PLAYER_ID": np.concatenate(parts_pid),
        "team": np.concatenate(parts_team),
        "GAME_ID": np.concatenate(parts_gid),
        "GAME_DATE": np.concatenate(parts_date),
        "team_game_rank": np.concatenate(parts_rank),
    })

    # ---- attach the player's state as of their LAST PRIOR appearance ------
    # merge_asof with allow_exact_matches=False is the strictly-before join,
    # and it operates on (PLAYER_ID, team) so a mid-season trade starts a fresh
    # history rather than borrowing the old team's usage. The date column is
    # renamed only AFTER the join, because merge_asof needs both sides to name
    # the key the same thing.
    right = (box[["GAME_DATE", "GAME_ID", "PLAYER_ID", "TEAM_ABBREVIATION",
                  *TRAILING_BOX, TRAILING_PM, "t_MIN", "t_MIN_last5", "t_apps"]]
             .drop_duplicates(subset=["GAME_DATE", "PLAYER_ID",
                                      "TEAM_ABBREVIATION"])
             .rename(columns={"TEAM_ABBREVIATION": "team",
                              "GAME_ID": "_last_game_id"}))
    # The merge key column in the output carries the LEFT date, so the right
    # side's date has to travel as a separate column to be readable.
    right["_last_game_date"] = right["GAME_DATE"]
    left = cand.sort_values("GAME_DATE", kind="mergesort")
    rows = pd.merge_asof(
        left, right.sort_values("GAME_DATE", kind="mergesort"),
        left_on="GAME_DATE", right_on="GAME_DATE",
        by=["PLAYER_ID", "team"], direction="backward",
        allow_exact_matches=False)
    rows = rows[rows["_last_game_date"].notna()].copy()
    rows["days_since_app"] = (rows["GAME_DATE"] - rows["_last_game_date"]).dt.days

    # How many of the team's games have gone by since this player last suited
    # up. Uses the APPEARANCE rank of that last game, not its date, because a
    # team can play while a player sits out.
    app_rank = app.set_index(["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID"])[
        "team_game_rank"]
    last_key = pd.MultiIndex.from_arrays(
        [rows["_last_game_id"], rows["team"], rows["PLAYER_ID"]])
    rows["team_games_since"] = (rows["team_game_rank"].to_numpy()
                                - app_rank.reindex(last_key).to_numpy())

    # ---- did the candidate actually play, and for how long? ---------------
    actual = box[["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID", "MIN", "played"]]
    idx = pd.MultiIndex.from_frame(actual[["GAME_ID", "TEAM_ABBREVIATION",
                                          "PLAYER_ID"]].rename(
        columns={"TEAM_ABBREVIATION": "team"}))
    amap = actual.set_index(idx)
    akey = pd.MultiIndex.from_arrays(
        [rows["GAME_ID"], rows["team"], rows["PLAYER_ID"]])
    rows["MIN"] = amap["MIN"].reindex(akey).to_numpy()
    rows["played"] = amap["played"].reindex(akey).fillna(0).astype(int).to_numpy()

    # ---- schedule context, known before tip-off --------------------------
    rt = D.rest_and_travel(games)
    gctx = pd.concat([
        games[["GAME_ID", "home"]].assign(
            team=lambda d: d["home"],
            rest_days=rt["home_rest_days"].to_numpy(),
            in_3_of_4=rt["home_in_3_of_4"].to_numpy(),
            is_home=1.0),
        games[["GAME_ID", "away"]].assign(
            team=lambda d: d["away"],
            rest_days=rt["away_rest_days"].to_numpy(),
            in_3_of_4=rt["away_in_3_of_4"].to_numpy(),
            is_home=0.0),
    ], ignore_index=True)
    rows = rows.merge(gctx, on=["GAME_ID", "team"], how="left")

    rows["rest_days"] = rows["rest_days"].fillna(3.0)
    rows["in_3_of_4"] = rows["in_3_of_4"].fillna(0.0)
    keep_cols = (["GAME_ID", "GAME_DATE", "team", "PLAYER_ID", "team_game_rank",
                  "played", "MIN", *TRAILING_BOX, TRAILING_PM, "t_MIN",
                  "t_MIN_last5", "t_apps", "days_since_app",
                  "team_games_since", "rest_days", "in_3_of_4", "is_home"])
    rows = rows[keep_cols].sort_values(
        ["GAME_DATE", "GAME_ID", "team", "PLAYER_ID"]).reset_index(drop=True)

    # A game with no candidate rows cannot be aggregated at all; report it
    # rather than silently emitting a NaN margin.
    covered = rows.groupby("GAME_ID").size()
    missing = len(games) - len(covered)
    if missing:
        import warnings
        warnings.warn(f"{missing} games have no usable candidate rows")
    return PlayerPanel(rows=rows, team_games=tg)


# --------------------------------------------------------------------------- #
# Expected minutes
# --------------------------------------------------------------------------- #

def _standardise(train: np.ndarray, *others: np.ndarray):
    mu = np.nanmean(train, axis=0)
    sd = np.nanstd(train, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    return (mu, sd,
            lambda a: np.nan_to_num((a - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0))


@dataclass
class ExpectedMinutes:
    """Two-part expected-minutes model: ``P(plays)`` x ``E[MIN | plays]``.

    The availability probability is the whole point. A starter who is probable
    to be out still receives a non-zero, *smaller* weight; a bench player
    expected to be a healthy scratch gets a small weight too. Both errors are
    the ones a real forecaster cannot avoid without an injury feed.
    """
    mu: np.ndarray = field(default_factory=lambda: np.zeros(len(MINUTES_FEATURES)))
    sd: np.ndarray = field(default_factory=lambda: np.ones(len(MINUTES_FEATURES)))
    play_coef: np.ndarray | None = None
    play_intercept: float = 0.0
    min_coef: np.ndarray | None = None
    min_intercept: float = 0.0

    def _x(self, rows: pd.DataFrame) -> np.ndarray:
        a = rows[list(MINUTES_FEATURES)].to_numpy(dtype=float)
        return np.nan_to_num((a - self.mu) / self.sd, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, rows: pd.DataFrame, iters: int = 60) -> "ExpectedMinutes":
        from sklearn.linear_model import LogisticRegression, Ridge
        x = rows[list(MINUTES_FEATURES)].to_numpy(dtype=float)
        mu = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
        self.mu, self.sd = mu, sd
        z = np.nan_to_num((x - mu) / sd, nan=0.0)
        y = rows["played"].to_numpy(dtype=int)
        if y.min() == y.max():                      # degenerate training block
            self.play_coef = np.zeros(z.shape[1])
            self.play_intercept = 10.0 if y.mean() > 0.5 else -10.0
        else:
            lr = LogisticRegression(C=1.0, max_iter=400, solver="lbfgs")
            lr.fit(z, y)
            self.play_coef = lr.coef_.ravel().copy()
            self.play_intercept = float(lr.intercept_[0])
        played_rows = rows[rows["played"] == 1]
        if len(played_rows) < 50:
            self.min_coef = np.zeros(z.shape[1])
            self.min_intercept = 25.0
        else:
            zp = self._x(played_rows)
            rg = Ridge(alpha=10.0, fit_intercept=True)
            rg.fit(zp, played_rows["MIN"].to_numpy(dtype=float))
            self.min_coef = rg.coef_.copy()
            self.min_intercept = float(rg.intercept_)
        return self

    def predict_play_prob(self, rows: pd.DataFrame) -> np.ndarray:
        z = self._x(rows)
        return 1.0 / (1.0 + np.exp(-(z @ self.play_coef + self.play_intercept)))

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        """E[minutes] = P(plays) * E[MIN | plays], floored at 0."""
        p = self.predict_play_prob(rows)
        m = np.clip(self._x(rows) @ self.min_coef + self.min_intercept, 0.0, 48.0)
        return np.clip(p * m, 0.0, 48.0)


# --------------------------------------------------------------------------- #
# Rating models
# --------------------------------------------------------------------------- #

@dataclass
class RatingConfig:
    halflife: float = 8.0             # EWMA halflife, in a player's own games
    pool_window: int = 10             # team-games back that define the pool
    ridge_alpha: float = 25.0         # ridge penalty on the rating weights
    pace_halflife: float = 10.0       # matches nba_data.rolling_team_features
    min_minutes: float = 0.0          # drop candidate weights below this


def _weighted_ridge(A: np.ndarray, y: np.ndarray, w: np.ndarray,
                    alpha: float, penalise_intercept: bool = False
                    ) -> np.ndarray:
    """Ridge with per-row weights. Solves (A'WA + alpha P) b = A'Wy.

    Column 0 of `A` is the intercept. Shrinking it is a silent way to bias a
    fit towards the origin, so it is left unpenalised unless asked otherwise.
    """
    Aw = A * w[:, None]
    G = Aw.T @ A
    diag = np.full(G.shape[0], alpha)
    if not penalise_intercept:
        diag[0] = 0.0
    G.flat[:: G.shape[0] + 1] += diag
    b = Aw.T @ (y * w)
    try:
        return np.linalg.solve(G, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(G, b, rcond=None)[0]


class _TeamRatingBase:
    """Shared machinery: aggregate candidate ratings into a team net rating.

    The aggregation is the point of the whole exercise and it is one line::

        net_per_100[t] = sum_i  w_i * r_i          w_i = E[MIN_i] / 240

    ``w`` sums to ~1, so ``net_per_100`` is a minutes-weighted average of the
    players' per-48 impact rates, i.e. a team net rating per 100 possessions in
    the units the target was built in. Both weights and ratings are strictly
    pre-game.
    """

    features: tuple[str, ...] = ()

    def __init__(self, cfg: RatingConfig | None = None):
        self.cfg = cfg or RatingConfig()
        self.coef_: np.ndarray | None = None
        self.mu_: np.ndarray | None = None
        self.sd_: np.ndarray | None = None
        self.y_mean_: float = 0.0
        self.hca_: float = 2.2

    # -- feature extraction ------------------------------------------------
    def _design(self, rows: pd.DataFrame) -> np.ndarray:
        x = np.column_stack([np.ones(len(rows))] +
                            [rows[c].to_numpy(dtype=float) for c in self.features])
        return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    def _standardise(self, A: np.ndarray) -> np.ndarray:
        """Centre the FEATURE columns; leave the intercept column at exactly 1.

        Centring column 0 too would turn it into a constant zero, which strips
        the model of any ability to sit at the mean of the target rather than
        at the origin. That is not cosmetic: with a degenerate intercept the
        whole constant offset of the target ends up unexplained, and every
        prediction is off by the target mean.
        """
        out = np.empty_like(A)
        out[:, 0] = 1.0
        out[:, 1:] = (A[:, 1:] - self.mu_) / self.sd_
        return out

    def _weights(self, rows: pd.DataFrame, minutes_mode: str,
                 em: "ExpectedMinutes | None") -> np.ndarray:
        if minutes_mode == "actual":
            w = rows["MIN"].to_numpy(dtype=float)
        elif minutes_mode == "expected":
            if em is None:
                raise ValueError("minutes_mode='expected' needs a fitted "
                                 "ExpectedMinutes model")
            w = em.predict(rows)
        elif minutes_mode == "expected_no_availability":
            # Ablation: the availability *probability* is replaced by 1. Every
            # candidate gets full weight, which is what a model that believed
            # the whole rotation is healthy would believe.
            if em is None:
                raise ValueError("minutes_mode='expected_no_availability' needs "
                                 "a fitted ExpectedMinutes model")
            w = np.clip(em._x(rows) @ em.min_coef + em.min_intercept, 0.0, 48.0)
        else:
            raise ValueError(f"unknown minutes_mode {minutes_mode!r}")
        w = np.where(w > self.cfg.min_minutes, w, 0.0)
        return w / 240.0

    # -- fit / score -------------------------------------------------------
    def fit(self, team_rows: pd.DataFrame, minutes_mode: str,
            em: "ExpectedMinutes | None" = None) -> "_TeamRatingBase":
        g = team_rows["_team_group"]
        idx = pd.factorize(g, sort=True)[0]
        n_groups = idx.max() + 1
        X = self._design(team_rows)
        W = self._weights(team_rows, minutes_mode, em)
        A = np.zeros((n_groups, X.shape[1]))
        np.add.at(A, idx, X * W[:, None])
        self.mu_ = A[:, 1:].mean(axis=0)
        self.sd_ = A[:, 1:].std(axis=0)
        self.sd_ = np.where(self.sd_ > 1e-9, self.sd_, 1.0)
        Az = self._standardise(A)
        # The design is per TEAM-GAME; the target column arrives per candidate
        # row, so it has to be collapsed to one value per group first.
        vals = team_rows["net_per_100"].to_numpy(dtype=float)
        y = (pd.Series(vals).groupby(idx).first()
             .reindex(np.arange(n_groups)).to_numpy())
        if not np.isfinite(y).all():
            raise ValueError("a team-game has no finite net_per_100 target")
        self.y_mean_ = float(y.mean())
        self.coef_ = _weighted_ridge(
            Az, y, np.ones(n_groups), self.cfg.ridge_alpha)
        return self

    def team_net(self, team_rows: pd.DataFrame, minutes_mode: str,
                 em: "ExpectedMinutes | None" = None) -> np.ndarray:
        idx = pd.factorize(team_rows["_team_group"], sort=True)[0]
        X = self._design(team_rows)
        W = self._weights(team_rows, minutes_mode, em)
        A = np.zeros((idx.max() + 1, X.shape[1]))
        np.add.at(A, idx, X * W[:, None])
        return self._standardise(A) @ self.coef_

    def player_rating(self, rows: pd.DataFrame) -> np.ndarray:
        """Per-player impact rate, in the units of the target (net per 100).

        `rating_i = intercept + sum_k coef_k * (x_ik - mu_k) / sd_k`, and the
        team aggregate is `sum_i (E[MIN_i] / 240) * rating_i`. The two are the
        same number, which is asserted in
        `Tests/test_player_ratings.py::TestHandComputedRotation`.
        """
        X = self._design(rows)
        return self._standardise(X) @ self.coef_


class BoxScoreRatings(_TeamRatingBase):
    """(a) BOX-SCORE REGRESSION.

    Team net rating per 100 is regressed on the team's minutes-weighted
    trailing box-score profile. Each player contributes
    ``(E[MIN_i]/240) * theta' x_i``, so the fitted ``theta`` IS the per-player
    rating: a linear scorecard saying how many net-rating points a player of
    that profile is worth per 48 minutes.

    No plus-minus enters this model. That is what makes it a clean control for
    :class:`PlusMinusRatings`: if the plus-minus signal is carrying both, the
    box-score-only model should be much weaker on its own.
    """
    features = TRAILING_BOX


class PlusMinusRatings(_TeamRatingBase):
    """(b) PLUS-MINUS-BASED APPROXIMATION -- **not true RAPM**.

    The classic RAPM regression puts one coefficient per (player, stint) and
    needs ``GameRotation`` to build the design matrix. ``data/nba/rotations/``
    is empty in this dataset, so there are no stints. What this fits instead is
    the closest thing the available tables support::

        net_per_100[t] = sum_i  (E[MIN_i]/240) * (a + b * pm48_i + gamma' x_i)

    ``pm48_i`` is the player's trailing exponentially weighted ON-COURT plus-
    minus per 48 minutes, taken from ``LeagueDashPlayerGameLog``, which does
    vary within a team-game and so is genuinely on-court. ``b`` is a single
    global trust parameter: how much of a trailing plus-minus is real player
    impact rather than opponent, venue and teammate noise.

    What is missing versus RAPM, stated plainly:

    * **stint on/off separation** -- a player's plus-minus here includes
      garbage-time minutes and is never differenced against the same player on
      the bench;
    * **opponent and venue adjustment** -- carried only by the small ``gamma``
      control block, not by the rating;
    * **per-stint parameters** -- RAPM's whole advantage over a single global
      coefficient is that a player's rating is estimated separately in each of
      the 120 ordered stint positions;
    * **possession weighting** -- minutes are used where RAPM uses possessions.

    So the per-player numbers produced here should be called adjusted
    plus-minus at team level, never RAPM. They are still worth fitting: the
    coefficient ``b`` is a directly interpretable statement about how much
    trailing plus-minus signal survives, which is a real measurement.
    """
    features = (TRAILING_PM,) + TRAILING_BOX

    def plus_minus_trust(self) -> float:
        """The fitted ``b``: points of net rating per unit of trailing +/-."""
        return float(self.coef_[1])


def make_rating_model(method: str, cfg: RatingConfig | None = None):
    if method == "boxscore":
        return BoxScoreRatings(cfg)
    if method in ("rapm", "plusminus"):
        return PlusMinusRatings(cfg)
    raise ValueError(f"unknown method {method!r}")


# --------------------------------------------------------------------------- #
# Targets and pace
# --------------------------------------------------------------------------- #

def team_targets(games: pd.DataFrame,
                 cfg: RatingConfig | None = None) -> pd.DataFrame:
    """Per (game, team): points, opponent points, trailing pace, net target.

    ``net_per_100`` is the model's regression target. Home-court advantage is
    removed before dividing, using a fixed 2.2 points, so the same model can be
    fit on both sides of every game without a role dummy. The prediction puts
    ``hca`` back:

        predicted_margin = net_home - net_away + hca

    Pace is the team's own *trailing* pace, never the realised pace of the game
    being predicted.
    """
    cfg = cfg or RatingConfig()
    pace = D.rolling_team_features(games, halflife=cfg.pace_halflife)
    long = pd.concat([
        games[["GAME_ID", "home", "home_pts", "away_pts"]].assign(
            team=lambda d: d["home"], pts=lambda d: d["home_pts"],
            opp_pts=lambda d: d["away_pts"], is_home=1.0,
            pace=pace["home_ewm_pace"].to_numpy()),
        games[["GAME_ID", "away", "home_pts", "away_pts"]].assign(
            team=lambda d: d["away"], pts=lambda d: d["away_pts"],
            opp_pts=lambda d: d["home_pts"], is_home=0.0,
            pace=pace["away_ewm_pace"].to_numpy()),
    ], ignore_index=True)
    long["pace"] = long["pace"].fillna(FALLBACK_PACE).clip(60.0, 130.0)
    diff = long["pts"] - long["opp_pts"]
    # strip HCA: the home side's observed edge is partly the building.
    adj = np.where(long["is_home"] > 0.5, diff - D_HCA_POINTS, diff + D_HCA_POINTS)
    long["net_per_100"] = adj / long["pace"] * 100.0
    long["diff_pts"] = diff
    return long


def _calibrate(margins: np.ndarray, y: np.ndarray, iters: int = 100
               ) -> tuple[float, float]:
    """Logistic ``P(home) = sigmoid(a * margin + b)``, Newton from a sane start.

    Two parameters on thousands of rows: this is about as much calibration as
    the data supports, and more would start fitting noise.
    """
    a, b = 0.11, 0.0
    for _ in range(iters):
        z = a * margins + b
        p = 1.0 / (1.0 + np.exp(-z))
        w = np.clip(p * (1 - p), 1e-9, None)
        r = y - p
        h = np.array([[np.sum(w * margins * margins), np.sum(w * margins)],
                      [np.sum(w * margins), np.sum(w)]])
        g = np.array([np.sum(w * margins * r), np.sum(w * r)])
        try:
            step = np.linalg.solve(h + 1e-9 * np.eye(2), g)
        except np.linalg.LinAlgError:
            break
        a, b = a + step[0], b + step[1]
        if abs(step[0]) < 1e-10 and abs(step[1]) < 1e-10:
            break
    return float(a), float(b)


# --------------------------------------------------------------------------- #
# Walk-forward driver
# --------------------------------------------------------------------------- #

def metric_standard_errors(y_true, p_pred, n_boot: int = 1000,
                           seed: int = 0) -> dict:
    """Nonparametric bootstrap standard errors for the headline metrics.

    A log-loss difference of 0.002 on ~6,600 rows is inside one standard
    error of zero, so every delta in this module is reported next to the
    scale of the noise it has to beat. The paired delta in
    ``paired_bootstrap_delta`` is the more powerful test and is what the
    conclusions rest on; these SEs are for judging an absolute number.
    """
    from sklearn.metrics import (roc_auc_score, brier_score_loss, log_loss,
                                 accuracy_score)
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p_pred, dtype=float), 1e-6, 1 - 1e-6)
    n = len(y)
    rng = np.random.default_rng(seed)
    out: dict = {}

    def stats(idx):
        return (
            log_loss(y[idx], p[idx], labels=[0, 1]),
            brier_score_loss(y[idx], p[idx]),
            accuracy_score(y[idx], (p[idx] > 0.5).astype(int)),
            roc_auc_score(y[idx], p[idx]) if len(np.unique(y[idx])) > 1 else np.nan,
        )

    point = stats(np.arange(n))
    draws = np.empty((n_boot, 4))
    for i in range(n_boot):
        draws[i] = stats(rng.integers(0, n, size=n))
    for name, col in zip(("log_loss", "brier", "accuracy", "auc"), range(4)):
        out[name] = {"point": float(point[col]),
                     "se": float(np.nanstd(draws[:, col], ddof=1)),
                     "n": int(n)}
    return out


@dataclass
class WalkForwardResult:
    method: str
    minutes_mode: str
    y: np.ndarray
    probs: np.ndarray
    margins: np.ndarray
    net_home: np.ndarray
    net_away: np.ndarray
    game_index: np.ndarray
    calibration: dict
    split_table: list[dict]
    fit_report: dict

    def metrics(self) -> dict:
        m = D.evaluate(self.y, self.probs)
        m.pop("calibration", None)
        m["calibration"] = self.calibration
        m["standard_errors"] = metric_standard_errors(self.y, self.probs)
        return m


def _subset(rows: pd.DataFrame, tgt: pd.DataFrame, games: pd.DataFrame,
            mask_games: np.ndarray):
    """Candidate rows + team-games restricted to the games in `mask_games`.

    Returns `(candidate_rows, team_game_table)`, with the target and the role
    flag already joined onto the candidate rows and a dense `_team_group` id on
    both, so `fit` and `team_net` aggregate over the same partition.

    A team-game with NO candidate rows -- which happens for the first games of
    a windowed slice, before any player has a prior appearance for that team --
    is dropped from the returned table. It cannot be scored (there is no
    pre-game player information for it) and leaving it in makes the group ids
    in the two frames disagree, which shows up as an out-of-bounds index far
    away from the cause.
    """
    sel_ids = set(games.loc[mask_games, "GAME_ID"])
    r = rows[rows["GAME_ID"].isin(sel_ids)]
    t = tgt[tgt["GAME_ID"].isin(sel_ids)]
    key = pd.MultiIndex.from_arrays([r["GAME_ID"], r["team"]])
    tkey = pd.MultiIndex.from_arrays([t["GAME_ID"], t["team"]])
    if not key.isin(tkey).all():
        # A candidate row with no matching team-game would silently become a
        # NaN target, which the ridge then absorbs as "this team had no
        # signal" instead of raising. Refuse instead.
        raise ValueError(
            f"{int((~key.isin(tkey)).sum())} candidate rows have no team-game "
            "target; the panel and the game table disagree")
    # Drop the team-games that no candidate row covers, then renumber densely
    # so the ids are 0..n-1 in the SAME order in both frames.
    grp0 = pd.Series(pd.factorize(tkey, sort=True)[0], index=tkey)
    present = np.sort(np.unique(grp0.reindex(key).dropna().to_numpy(dtype=int)))
    remap = pd.Series(np.arange(len(present)), index=present)
    new_t = remap.reindex(grp0.reindex(tkey).to_numpy()).to_numpy()
    t = t.assign(_team_group=new_t)
    t = t[t["_team_group"].notna()].copy()
    t["_team_group"] = t["_team_group"].astype(int)
    new_r = remap.reindex(grp0.reindex(key).to_numpy()).to_numpy()
    lookup = pd.Series(t["net_per_100"].to_numpy(), index=t["_team_group"])
    r = r.assign(net_per_100=lookup.reindex(new_r).to_numpy(),
                 _team_group=new_r)
    assert r["_team_group"].notna().all()
    assert int(r["_team_group"].max()) == len(t) - 1
    return r, t


def _margins_for(games: pd.DataFrame, r: pd.DataFrame, t: pd.DataFrame,
                 mask_games: np.ndarray, model: "_TeamRatingBase",
                 minutes_mode: str, em: ExpectedMinutes | None) -> np.ndarray:
    """Predicted margin for the games selected by ``mask_games``, using a
    model that was fitted on strictly earlier games."""
    net = model.team_net(r, minutes_mode, em)
    # `net` comes back ordered by GROUP CODE; `t` is in its own row order
    # (every home team, then every away team). Indexing by the code is what
    # makes the two line up -- assigning positionally silently pairs each
    # team's rating with another team's.
    t = t.assign(_net=net[t["_team_group"].to_numpy().astype(int)])
    assert t["_net"].notna().all(), "a team-game lost its predicted net rating"
    home = t[t["is_home"] > 0.5].set_index("GAME_ID")["_net"]
    away = t[t["is_home"] < 0.5].set_index("GAME_ID")["_net"]
    order = games.loc[mask_games, "GAME_ID"]
    return ((home - away + D_HCA_POINTS).reindex(order).to_numpy(dtype=float),
            home.reindex(order).to_numpy(dtype=float),
            away.reindex(order).to_numpy(dtype=float))


def _fit_ratings(rows: pd.DataFrame, tgt: pd.DataFrame, mask_games: np.ndarray,
                 games: pd.DataFrame, method: str, minutes_mode: str,
                 cfg: RatingConfig, em: ExpectedMinutes | None):
    r, _ = _subset(rows, tgt, games, mask_games)
    model = make_rating_model(method, cfg)
    model.fit(r, minutes_mode, em)
    return model, r


def run_walk_forward(method: str = "rapm", minutes_mode: str = "expected",
                     games: pd.DataFrame | None = None,
                     n_splits: int = 5, cfg: RatingConfig | None = None,
                     panel: PlayerPanel | None = None,
                     inner_splits: int = 2) -> WalkForwardResult:
    """Walk-forward by date. Returns out-of-sample probabilities.

    For each split:

    1. fit :class:`ExpectedMinutes` on the TRAINING rows only;
    2. inside the training block, run a nested walk-forward and fit the
       two-parameter logistic on the resulting OUT-OF-sample margins -- so the
       calibration is not fitted on the in-sample margins it will be judged on;
    3. refit the rating weights on the whole training block and score the test
       block.

    Nothing in the loop reads the test block, at any depth.
    """
    cfg = cfg or RatingConfig()
    g = D.build_games() if games is None else games
    if panel is None:
        panel = build_panel(g, pool_window=cfg.pool_window, halflife=cfg.halflife)
    rows, tgt = panel.rows, team_targets(g, cfg)

    splits = D.walk_forward_split(g, n_splits=n_splits)
    probs_all, y_all, marg_all, gidx_all, table = [], [], [], [], []
    nh_all, na_all = [], []
    cal_report: dict = {}
    last_model, last_rows = None, None

    for k, (tr, te) in enumerate(splits):
        tr_mask = np.zeros(len(g), bool); tr_mask[tr] = True
        te_mask = np.zeros(len(g), bool); te_mask[te] = True
        tr_ids = set(g.loc[tr_mask, "GAME_ID"])
        em = ExpectedMinutes().fit(rows[rows["GAME_ID"].isin(tr_ids)])

        # (2) nested walk-forward for the calibration slope
        inner = D.walk_forward_split(g.iloc[tr].reset_index(drop=True),
                                     n_splits=inner_splits, initial_frac=0.5)
        base = g.index[tr]
        cal_m, cal_y = [], []
        for itr, ite in inner:
            i_tr = base[itr]; i_te = base[ite]
            im = np.zeros(len(g), bool); im[i_tr] = True
            jm = np.zeros(len(g), bool); jm[i_te] = True
            inner_ids = set(g.loc[im, "GAME_ID"])
            iem = ExpectedMinutes().fit(rows[rows["GAME_ID"].isin(inner_ids)])
            imodel, _ = _fit_ratings(rows, tgt, im, g, method, minutes_mode,
                                     cfg, iem)
            jr, jt = _subset(rows, tgt, g, jm)
            m, _, _ = _margins_for(g, jr, jt, jm, imodel, minutes_mode, iem)
            ok = np.isfinite(m)
            cal_m.append(m[ok])
            cal_y.append(g.loc[jm, "y"].to_numpy()[ok])
        cm = np.concatenate(cal_m) if cal_m else np.zeros(0)
        cy = np.concatenate(cal_y) if cal_y else np.zeros(0)
        if len(cm) < 200:
            a, b = 0.11, 0.0
        else:
            a, b = _calibrate(cm, cy)
        cal_report = {"a": a, "b": b, "n_calibration_rows": int(len(cm)),
                      "calibration_logloss": float(D.evaluate(cy, 1/(1+np.exp(
                          -(a*cm+b))))["log_loss"]) if len(cm) else float("nan")}

        # (3) refit on the full training block, score the test block
        model, train_rows = _fit_ratings(rows, tgt, tr_mask, g, method,
                                         minutes_mode, cfg, em)
        er, et = _subset(rows, tgt, g, te_mask)
        m, nh, na = _margins_for(g, er, et, te_mask, model, minutes_mode, em)
        # A game with no pre-game candidate rows cannot be scored by a player
        # model at all. Drop it explicitly and say how many, rather than
        # letting a NaN margin become a NaN probability and poison the metrics.
        ok = np.isfinite(m)
        n_dropped = int((~ok).sum())
        positions = np.flatnonzero(te_mask)[ok]
        m, nh, na = m[ok], nh[ok], na[ok]
        y = g.loc[te_mask, "y"].to_numpy()[ok]
        p = 1.0 / (1.0 + np.exp(-np.clip(a * m + b, -30, 30)))
        probs_all.append(p); y_all.append(y); marg_all.append(m)
        nh_all.append(nh); na_all.append(na)
        gidx_all.append(positions)
        last_model, last_rows = model, train_rows
        table.append({
            "split": k, "n_train": int(tr_mask.sum()), "n_test": int(ok.sum()),
            "n_dropped_no_candidates": n_dropped,
            "train_max_date": str(g.loc[tr_mask, "GAME_DATE"].max().date()),
            "test_min_date": str(g.loc[te_mask, "GAME_DATE"].min().date()),
            "test_max_date": str(g.loc[te_mask, "GAME_DATE"].max().date()),
            "calib_a": a, "calib_b": b,
            "log_loss": float(D.evaluate(y, p)["log_loss"]),
        })

    res = WalkForwardResult(
        method=method, minutes_mode=minutes_mode,
        y=np.concatenate(y_all), probs=np.concatenate(probs_all),
        margins=np.concatenate(marg_all),
        net_home=np.concatenate(nh_all), net_away=np.concatenate(na_all),
        game_index=np.concatenate(gidx_all),
        calibration=cal_report, split_table=table, fit_report={})
    if last_model is not None and last_rows is not None:
        r = last_rows.copy()
        r["rating"] = last_model.player_rating(r)
        agg = (r.assign(w=r["MIN"] / 240.0).groupby("PLAYER_ID")
               .apply(lambda d: float(np.average(d["rating"], weights=np.maximum(d["w"], 1e-6))),
                      include_groups=False)
               .sort_values(ascending=False))
        res.fit_report = {
            "coef_": last_model.coef_.tolist(),
            "features": ["intercept"] + list(last_model.features),
            "top_players": agg.head(20).round(4).to_dict(),
            "n_players_rated": int(len(agg)),
        }
    return res


# --------------------------------------------------------------------------- #
# The availability gap
# --------------------------------------------------------------------------- #

def availability_gap(games: pd.DataFrame | None = None, method: str = "rapm",
                      cfg: RatingConfig | None = None,
                      panel: PlayerPanel | None = None) -> dict:
    """How much does using ACTUAL minutes instead of EXPECTED minutes buy?

    Both models are identical except for the weight on each player, and both
    are walk-forward. The comparison is a paired bootstrap on the SAME test
    rows, so the shared variance cancels.

    SIGN: ``paired_bootstrap_delta(y, p_leaked, p_honest)`` returns
    ``loss(leaked) - loss(honest)``. **A positive delta means the honest,
    expected-minutes model is BETTER.** A large negative delta means actual
    minutes were doing real work -- work a bettor cannot do before tip-off.
    """
    g = D.build_games() if games is None else games
    cfg = cfg or RatingConfig()
    out = {"method": method}
    runs = {}
    for mode in ("actual", "expected"):
        runs[mode] = run_walk_forward(method, mode, games=g, cfg=cfg, panel=panel)
    # Align the two runs on the same test rows before comparing.
    a, b = runs["actual"], runs["expected"]
    order_a = np.argsort(a.game_index, kind="stable")
    order_b = np.argsort(b.game_index, kind="stable")
    assert (a.game_index[order_a] == b.game_index[order_b]).all(), \
        "the two minutes modes scored different test rows"
    idx = a.game_index[order_a]
    y = a.y[order_a]
    pa, pb = a.probs[order_a], b.probs[order_b]
    d = D.paired_bootstrap_delta(y, pa, pb)
    ea, eb = D.evaluate(y, pa), D.evaluate(y, pb)
    out.update({
        "n": int(len(y)),
        "leaked_actual_minutes": {
            "log_loss": float(ea["log_loss"]), "brier": float(ea["brier"]),
            "auc": float(ea["auc"]),
            "standard_errors": metric_standard_errors(y, pa),
        },
        "honest_expected_minutes": {
            "log_loss": float(eb["log_loss"]), "brier": float(eb["brier"]),
            "auc": float(eb["auc"]),
            "standard_errors": metric_standard_errors(y, pb),
        },
        # positive => honest is better; negative => actual minutes helped
        "delta_log_loss_leaked_minus_honest": d["delta"],
        "ci": [d["ci_lo"], d["ci_hi"]],
        "p_positive": d["p_positive"],
        "ci_excludes_zero": bool(d["ci_lo"] * d["ci_hi"] > 0),
    })
    return out


# --------------------------------------------------------------------------- #
# Incremental test on top of the team feature set
# --------------------------------------------------------------------------- #

def team_feature_matrix(games: pd.DataFrame | None = None) -> pd.DataFrame:
    """The team-only feature set this player model has to beat on its own terms."""
    g = D.build_games() if games is None else games
    roll = D.rolling_team_features(g)
    rest = D.rest_and_travel(g)
    # `rolling_team_features` keeps GAME_DATE in its output (it is only GAME_ID
    # and the two team columns that get dropped), so select numerics rather
    # than assuming every column is a feature.
    X = pd.concat([roll[[c for c in roll.columns
                         if pd.api.types.is_numeric_dtype(roll[c])]]
                   .reset_index(drop=True),
                   rest[[c for c in rest.columns
                         if pd.api.types.is_numeric_dtype(rest[c])]]
                   .reset_index(drop=True)], axis=1).astype(float)
    # A constant column carries no information and does upset solvers.
    keep = [c for c in X.columns if X[c].nunique(dropna=True) > 1]
    return X[keep]


def _logistic_fit(X: np.ndarray, y: np.ndarray, C: float = 1.0):
    from sklearn.linear_model import LogisticRegression
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    Z = np.nan_to_num((X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    lr = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    lr.fit(Z, y)
    return (mu, sd, lr)


def _logistic_pred(model, X: np.ndarray) -> np.ndarray:
    mu, sd, lr = model
    Z = np.nan_to_num((X - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)
    return lr.predict_proba(Z)[:, 1]


def incremental_test(games: pd.DataFrame | None = None, method: str = "rapm",
                     minutes_mode: str = "expected",
                     cfg: RatingConfig | None = None,
                     panel: PlayerPanel | None = None,
                     n_splits: int = 5,
                     player_splits: int = 10) -> dict:
    """Standalone player-model metrics, then the delta on top of team features.

    Two separate questions, deliberately not mixed:

    * **standalone** -- the player model with no team feature at all. This is
      the number that looks most exciting and is least useful.
    * **incremental** -- team features alone vs team features + the player
      model's aggregated output, on exactly the same rows.

    ``player_splits`` sets how finely the player model's own walk-forward is
    cut. It defaults to 10 rather than 5 because the rating model benefits from
    a later training cutoff, and the evaluation game set (the second half of
    the timeline) is the same either way. It does not change which games are
    scored, only how much data each of their fits sees.

    SIGN: ``paired_bootstrap_delta(y, p_team, p_team_plus_player)`` returns
    ``loss(team) - loss(team+player)``. **A positive delta means adding the
    player features IMPROVED log loss.** A CI straddling zero is a null, and a
    null here is a result, not a failure to report.
    """
    g = D.build_games() if games is None else games
    cfg = cfg or RatingConfig()
    # The player margin only exists out-of-sample, and the walk-forward only
    # produces it for the SECOND HALF of the timeline. Scoring the logistic on
    # the full game table would therefore hand it a feature that is NaN
    # throughout its own training folds and non-NaN at prediction time -- a
    # train/serve mismatch that can only understate the feature. So the
    # comparison is restricted to the games the player model actually covers,
    # and the logistic is then walk-forwarded over THAT subset.
    res = run_walk_forward(method, minutes_mode, games=g, cfg=cfg,
                           panel=panel, n_splits=player_splits)
    order = np.sort(res.game_index)
    sub = g.iloc[order].reset_index(drop=True)
    X = team_feature_matrix(g).iloc[order].reset_index(drop=True)
    y = sub["y"].to_numpy()

    marg = res.margins[np.argsort(res.game_index, kind="stable")]
    feats = pd.DataFrame({
        "pm_margin": marg,
        "pm_home_net": res.net_home[np.argsort(res.game_index, kind="stable")],
        "pm_away_net": res.net_away[np.argsort(res.game_index, kind="stable")],
    })
    assert np.isfinite(feats.to_numpy()).all(), \
        "player features are missing on some scored games"

    both = pd.concat([X, feats], axis=1)
    p_team = np.full(len(sub), np.nan)
    p_both = np.full(len(sub), np.nan)
    fold = []
    for k, (tr, tei) in enumerate(D.walk_forward_split(sub, n_splits=n_splits)):
        trm = np.zeros(len(sub), bool); trm[tr] = True
        tem = np.zeros(len(sub), bool); tem[tei] = True
        if tem.sum() < 50 or trm.sum() < 500:
            continue
        m1 = _logistic_fit(X[trm].to_numpy(), y[trm])
        p_team[tem] = _logistic_pred(m1, X[tem].to_numpy())
        m2 = _logistic_fit(both[trm].to_numpy(), y[trm])
        p_both[tem] = _logistic_pred(m2, both[tem].to_numpy())
        fold.append({"split": k, "n_test": int(tem.sum())})

    ok = np.isfinite(p_team) & np.isfinite(p_both)
    yy = y[ok]
    pt, pb = p_team[ok], p_both[ok]
    d = D.paired_bootstrap_delta(yy, pt, pb)
    sa = D.evaluate(res.y, res.probs)
    sb = D.evaluate(yy, pt)
    sc = D.evaluate(yy, pb)
    return {
        "method": method, "minutes_mode": minutes_mode, "n": int(ok.sum()),
        "n_player_splits": player_splits,
        "standalone": {"log_loss": sa["log_loss"], "brier": sa["brier"],
                       "auc": sa["auc"], "accuracy": sa["accuracy"],
                       "calibration": sa["calibration"]},
        "team_only": {"log_loss": sb["log_loss"], "brier": sb["brier"],
                      "auc": sb["auc"], "accuracy": sb["accuracy"],
                      "calibration": sb["calibration"]},
        "team_plus_player": {"log_loss": sc["log_loss"], "brier": sc["brier"],
                             "auc": sc["auc"], "accuracy": sc["accuracy"],
                             "calibration": sc["calibration"]},
        # positive => adding the player features improved log loss
        "delta_log_loss": d["delta"], "ci": [d["ci_lo"], d["ci_hi"]],
        "ci_excludes_zero": bool(d["ci_lo"] * d["ci_hi"] > 0),
        "p_positive": d["p_positive"], "folds": fold,
        "split_table": res.split_table, "fit_report": res.fit_report,
        "n_player_features": int(feats.shape[1]),
    }


# --------------------------------------------------------------------------- #
# Cross-family sign check
# --------------------------------------------------------------------------- #

def _logistic_sign(X: np.ndarray, y: np.ndarray, col: int) -> float:
    """Standardised logistic coefficient on one column."""
    mu, sd, lr = _logistic_fit(X, y)
    return float(lr.coef_.ravel()[col]) / sd[col]


def _marginal_sign(p: np.ndarray, z: np.ndarray) -> float:
    """Mean predicted probability above the column median minus below it.

    Model-agnostic, so it is comparable across families in a way a tree's
    feature importance is not. A positive number means the model reads a
    larger player-model margin as a larger chance of a home win.
    """
    hi = p[z > np.median(z)].mean()
    lo = p[z <= np.median(z)].mean()
    return float(hi - lo)


def cross_family_sign_check(games: pd.DataFrame | None = None,
                            method: str = "rapm",
                            cfg: RatingConfig | None = None,
                            panel: PlayerPanel | None = None,
                            n_splits: int = 5,
                            player_splits: int = 10) -> dict:
    """Do model families agree on the SIGN of the player-derived feature?

    A previous cross-family disagreement test in this suite found that
    gradient boosting REVERSES the sign of player-derived features relative to
    a linear model. That is worth re-checking rather than assuming fixed,
    because the two failure modes look identical from the log loss alone: a
    feature that carries no information can be given either sign by either
    family.

    For each family this reports

    * ``delta_log_loss`` -- team features vs team + player, paired bootstrap,
      **positive means adding the player features helped**;
    * ``sign_pm_margin`` -- the standardised coefficient for the linear model;
    * ``marginal_gap_pm_margin`` -- mean predicted probability above the
      feature's median minus below it, for every family including the linear
      one, so the signs are directly comparable.

    Disagreement between the marginal gaps is reported explicitly as
    ``sign_disagreement``. It is not averaged away: two families reading the
    same feature in opposite directions is a finding about the feature, not
    something to smooth over with an ensemble.

    Like :func:`incremental_test`, this scores only the games the player model
    covers, so the feature is present in every training fold.
    """
    g = D.build_games() if games is None else games
    cfg = cfg or RatingConfig()
    res = run_walk_forward(method, "expected", games=g, cfg=cfg,
                           panel=panel, n_splits=player_splits)
    perm = np.argsort(res.game_index, kind="stable")
    order = res.game_index[perm]
    sub = g.iloc[order].reset_index(drop=True)
    X = team_feature_matrix(g).iloc[order].reset_index(drop=True)
    marg = res.margins[perm]
    both = X.copy()
    both["pm_margin"] = marg

    y_all = sub["y"].to_numpy()
    n = len(sub)
    out: dict = {"method": method, "n_test": int(n), "families": {}}

    families = ["logistic", "hist_gb"]
    try:
        import xgboost  # noqa: F401
        families.append("xgboost")
    except ImportError:
        out["xgboost_unavailable"] = True

    for fam in families:
        p_team = np.full(n, np.nan)
        p_both = np.full(n, np.nan)
        lin_signs, tree_signs = [], []
        for tr, tei in D.walk_forward_split(sub, n_splits=n_splits):
            trm = np.zeros(n, bool); trm[tr] = True
            tem = np.zeros(n, bool); tem[tei] = True
            if tem.sum() < 50 or trm.sum() < 500:
                continue
            a, b, s_lin, s_tree = _family_pair(
                fam, X[trm].to_numpy(float), y_all[trm],
                both[trm].to_numpy(float), X[tem].to_numpy(float),
                both[tem].to_numpy(float))
            p_team[tem], p_both[tem] = a, b
            if np.isfinite(s_lin):
                lin_signs.append(s_lin)
            tree_signs.append(s_tree)
        ok = np.isfinite(p_team) & np.isfinite(p_both)
        d = D.paired_bootstrap_delta(y_all[ok], p_team[ok], p_both[ok])
        out["families"][fam] = {
            "team_only_log_loss": float(D.evaluate(y_all[ok], p_team[ok])["log_loss"]),
            "team_plus_player_log_loss": float(D.evaluate(y_all[ok], p_both[ok])["log_loss"]),
            "delta_log_loss": d["delta"], "ci": [d["ci_lo"], d["ci_hi"]],
            "sign_pm_margin": (float(np.mean(lin_signs)) if lin_signs
                               else float("nan")),
            "marginal_gap_pm_margin": float(np.mean(tree_signs)),
        }

    signs = [v["marginal_gap_pm_margin"] for v in out["families"].values()]
    out["sign_disagreement"] = bool(min(signs) < 0 < max(signs))
    out["signs"] = {f: v["marginal_gap_pm_margin"] for f, v in out["families"].items()}
    return out


def _family_pair(fam: str, Xtr: np.ndarray, ytr: np.ndarray,
                 Btr: np.ndarray, Xte: np.ndarray, Bte: np.ndarray):
    """Fit a family on two feature sets and report both its predictions and
    its reading of the added ``pm_margin`` column (the last column of B).

    ``Xtr`` is the team feature set, ``Btr`` the team + player set. Both are
    standardised on their own training statistics before fitting.
    """
    def build(X, y, C):
        mu = np.nanmean(X, axis=0)
        sd = np.nanstd(X, axis=0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
        Z = np.nan_to_num((X - mu) / sd, nan=0.0)
        if fam == "logistic":
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(Z, y)
            return m, mu, sd, Z
        if fam == "hist_gb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            m = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=4,
                l2_regularization=1.0, random_state=0).fit(Z, y)
            return m, mu, sd, Z
        import xgboost as xgb
        m = xgb.XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=0, eval_metric="logloss",
            tree_method="hist", verbosity=0).fit(Z, y)
        return m, mu, sd, Z

    def predict(model, mu, sd, X):
        Z = np.nan_to_num((X - mu) / sd, nan=0.0)
        return model.predict_proba(Z)[:, 1]

    m1, mu1, sd1, _ = build(Xtr, ytr, 1.0)
    m2, mu2, sd2, _ = build(Btr, ytr, 1.0)
    p1 = predict(m1, mu1, sd1, Xte)
    p2 = predict(m2, mu2, sd2, Bte)
    col = Btr.shape[1] - 1
    if fam == "logistic":
        s_lin = float(m2.coef_.ravel()[col]) / sd2[col]
    else:
        s_lin = float("nan")
    s_tree = _marginal_sign(p2, Bte[:, col])
    return p1, p2, s_lin, s_tree


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def report() -> dict:
    """Run everything and print a report. This is the module's `__main__`."""
    g = D.build_games()
    cfg = RatingConfig()
    print(f"games {len(g):,}   {g.GAME_DATE.min().date()} .. "
          f"{g.GAME_DATE.max().date()}   home win rate {g.y.mean():.4f}")
    panel = build_panel(g, pool_window=cfg.pool_window, halflife=cfg.halflife)
    rows = panel.rows
    print(f"\ncandidate panel: {len(rows):,} rows over "
          f"{rows.GAME_ID.nunique():,} games "
          f"({len(rows)/rows.GAME_ID.nunique():.1f} candidates/game), "
          f"{rows.PLAYER_ID.nunique():,} players")
    print(f"candidates who played: {rows.played.mean():.4f}")

    out: dict = {}
    for method in ("boxscore", "rapm"):
        print(f"\n=== standalone: {method} / expected minutes ===")
        r = run_walk_forward(method, "expected", games=g, cfg=cfg, panel=panel)
        m = r.metrics()
        se = m["standard_errors"]
        print(f"  n={int(m['n'])}  log_loss={m['log_loss']:.5f} "
              f"(se {se['log_loss']['se']:.5f})  "
              f"brier={m['brier']:.5f} (se {se['brier']['se']:.5f})")
        print(f"  auc={m['auc']:.4f} (se {se['auc']['se']:.4f})  "
              f"accuracy={m['accuracy']:.4f} (se {se['accuracy']['se']:.4f})")
        print(f"  calibration a={r.calibration['a']:.4f} "
              f"b={r.calibration['b']:.4f} on "
              f"{r.calibration['n_calibration_rows']} out-of-sample rows")
        if method == "rapm":
            print(f"  plus-minus trust coefficient b = "
                  f"{r.fit_report['coef_'][1]:.4f}")
        top = list(r.fit_report["top_players"].items())[:8]
        print("  top players by final-block rating: " +
              ", ".join(f"{k}({v:+.3f})" for k, v in top
                        if np.isfinite(v)))
        print("  per-split log loss: " +
              ", ".join(f"{s['log_loss']:.5f}" for s in r.split_table))
        out[f"standalone_{method}"] = m

    print("\n=== availability gap: actual vs expected minutes ===")
    gap = availability_gap(games=g, cfg=cfg, panel=panel)
    la, lb = gap["leaked_actual_minutes"], gap["honest_expected_minutes"]
    print(f"  actual   (LEAKED): log_loss={la['log_loss']:.5f} "
          f"(se {la['standard_errors']['log_loss']['se']:.5f})  "
          f"brier={la['brier']:.5f}  auc={la['auc']:.4f}")
    print(f"  expected (honest): log_loss={lb['log_loss']:.5f} "
          f"(se {lb['standard_errors']['log_loss']['se']:.5f})  "
          f"brier={lb['brier']:.5f}  auc={lb['auc']:.4f}")
    print(f"  delta = loss(leaked) - loss(honest) = "
          f"{gap['delta_log_loss_leaked_minus_honest']:+.5f} "
          f"CI [{gap['ci'][0]:+.5f}, {gap['ci'][1]:+.5f}]  "
          f"(positive would mean the honest model is better)")
    print(f"  CI excludes zero: {gap['ci_excludes_zero']}")
    out["availability_gap"] = gap

    print("\n=== incremental: team features vs team + player ===")
    for method in ("boxscore", "rapm"):
        inc = incremental_test(games=g, method=method, cfg=cfg, panel=panel)
        print(f"  {method}: standalone ll={inc['standalone']['log_loss']:.5f} "
              f"auc={inc['standalone']['auc']:.4f}")
        print(f"           team-only ll={inc['team_only']['log_loss']:.5f} "
              f"auc={inc['team_only']['auc']:.4f}")
        print(f"           +player    ll={inc['team_plus_player']['log_loss']:.5f} "
              f"auc={inc['team_plus_player']['auc']:.4f}")
        print(f"           delta = loss(team) - loss(team+player) = "
              f"{inc['delta_log_loss']:+.5f} "
              f"CI [{inc['ci'][0]:+.5f}, {inc['ci'][1]:+.5f}]  "
              f"(positive => player features helped)")
        verdict = ("CI EXCLUDES ZERO" if inc["ci"][0] * inc["ci"][1] > 0
                   else "CI STRADDLES ZERO -> null")
        print(f"           {verdict}")
        out[f"incremental_{method}"] = inc

    print("\n=== cross-family sign of the player-derived margin ===")
    cs = cross_family_sign_check(games=g, cfg=cfg, panel=panel)
    for fam, v in cs["families"].items():
        print(f"  {fam:9s} delta={v['delta_log_loss']:+.5f} "
              f"CI [{v['ci'][0]:+.5f}, {v['ci'][1]:+.5f}]  "
              f"marginal sign on pm_margin={v['marginal_gap_pm_margin']:+.4f}")
    print(f"  sign disagreement between families: {cs['sign_disagreement']}")
    out["cross_family"] = cs

    print("\nReminder: paired_bootstrap_delta returns loss(a) - loss(b). "
          "A positive delta means the SECOND model is better. All deltas "
          "above follow that convention.\nReminder: Vandenbruaene et al. "
          "(2022) surveyed 628 implementations and never rejected "
          "unprofitability after multiple-testing correction. None of this is "
          "a demonstrated edge.")
    return out


if __name__ == "__main__":
    report()