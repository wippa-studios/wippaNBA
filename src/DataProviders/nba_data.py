"""Shared data + evaluation layer for the nba_api model suite.

Every model in this suite (Elo baseline, rolling-feature logistic regression,
margin/total, player ratings, gradient boosting, stacking) must build on this
module so their numbers are directly comparable. If you find yourself writing a
second loader or a second evaluation harness, that is a bug - use this one.

DATA
----
Source: /home/Joel/Desktop/Legit_NBA_Predictions/data/nba (310 CSVs, 11 seasons
2015-16 .. 2025-26). Downloaded via download_nba_data.py. See
data/nba/DATA_NOTES.md for the full schema.

Four traps this module exists to absorb, so no downstream model has to:

1. `schedule/` is camelCase; every other table is UPPERCASE. The schedule also
   contains All-Star / Rising Stars / preseason games (34-41 tricodes per season
   instead of 30) and `gameStatus == 3` for finished games, never 1.
2. `GAME_ID` and `gameId` must be read as `str` or leading zeros are lost and
   joins silently return zero rows.
3. `GAME_DATE` is a string. In the schedule it is "%m/%d/%Y %H:%M:%S"; in the
   game logs it is ISO-ish. Parse explicitly.
4. `snapshots/` is a SEASON-END aggregate. Using it for games inside the same
   season leaks the future (most sharply `player_estimated_metrics`, which NBA
   publishes retroactively). Use only as a prior for the FOLLOWING season.

LEAKAGE
-------
All feature builders here are strictly causal: a feature for game t is computed
only from games with date < date(t). `rolling` enforces this explicitly. There is
no function here that reads a season-end aggregate for a game inside that season.

KNOWN CAVEAT - ewm_net_rating IS REDUNDANT, DO NOT READ ITS COEFFICIENT
--------------------------------------------------------------------
`home_ewm_net_rating` / `away_ewm_net_rating` are not independent information.
NBA defines NET_RATING as OFF_RATING minus DEF_RATING, and these are exponentially
weighted before the contrast is taken, so the identity survives the weighting.
Measured on the real feature frame:

    regress d_net_rating on (d_margin, d_off_rating, d_def_rating, 1)
        R^2            = 0.99999735
        residual sd    = 1.265e-02   (feature sd 7.764)

The condition number of the four standardised strength contrasts is not finite -
an SVD cannot decompose the matrix at all - while dropping d_net_rating gives a
healthy 36.1. Consequences:

* Per-coefficient readings of d_margin / d_net_rating / d_off_rating /
  d_def_rating are NOT safe. Signs will flip between model families for reasons
  that have nothing to do with the effect of the feature.
* A cross-family "sign disagreement" audit over these columns measures the
  degeneracy, not the world. A +/-1 SD contrast on d_net_rating while freezing
  every other column asks about a pairing that essentially does not occur, so
  the question was never asked off-manifold.
* Predictions remain identified and are unaffected - only interpretation of
  individual coefficients is.

If you need interpretable coefficients, drop one of d_net_rating / d_margin and
refit. Predictions do not need this.
"""
from __future__ import annotations

import glob
import os
import re
from functools import lru_cache, wraps
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Where the downloader puts its output. Resolved relative to the REPOSITORY, not
# to a fixed absolute path.
#
# The default used to be one developer's home directory:
# "/home/Joel/Desktop/Legit_NBA_Predictions/data/nba". In a shared repository that
# is wrong for every other person - someone who ran the downloader exactly as
# documented, into ./data/nba, would still be told the data was missing. It is
# also why the data guard behaved inconsistently between machines.
#
# Order: NBA_DATA_ROOT, then ./data/nba, then the legacy sibling checkout that the
# original development used. First match that exists wins.
def _default_data_root() -> str:
    # An explicit override ALWAYS wins, even when it points somewhere that does
    # not exist. Falling through to a default in that case silently ignored the
    # caller: the error then names a path nobody asked for, and a CI job with a
    # typo'd variable reads as "the data is missing" rather than "your variable
    # is wrong".
    override = os.environ.get("NBA_DATA_ROOT")
    if override:
        return override
    candidates = [
        str(_REPO_ROOT / "data" / "nba"),
        str(_REPO_ROOT.parent / "Legit_NBA_Predictions" / "data" / "nba"),
    ]
    for c in candidates:
        if Path(c).is_dir():
            return c
    # Nothing exists yet. Return the in-repo location anyway so the error message
    # names the path the README tells people to use, not a stale absolute one.
    return candidates[0]


DATA_ROOT = _default_data_root()

# The minimum a usable dataset must contain, checked for actual CSVs rather than
# merely for the directory existing - the downloader writes its tables one stage
# at a time, and an interrupted run leaves exactly the state that crashes a loader
# with `ValueError: No objects to concatenate`.
#
# Deliberately only the two tables every code path needs. An earlier version also
# required `player_gamelogs` and `schedule`, which is correct for the real 382 MB
# download but wrong in general: the synthetic fixtures in Tests/ legitimately
# carry a subset, and requiring a full download to run a unit test is how a guard
# becomes an obstacle rather than a protection.
_REQUIRED_DATA_DIRS = ("game_logs", "team_gamelogs")

# The 30 current franchises. Guards against All-Star / Rising Stars / preseason
# squads, which the schedule carries under tricode-like names (USA, WLD, GNS, ULM...).
REAL_TEAMS = frozenset(
    "ATL BOS BKN CHA CHI CLE DAL DEN DET GSW HOU IND LAC LAL MEM MIA MIL MIN "
    "NOP NYK OKC ORL PHI PHX POR SAC SAS TOR UTA WAS".split()
)

# City-level arena coordinates, as used by download_nba_data.py. Two are
# approximations: LAC shares LAL's arena and SAS points at Alamodome. Both are
# city-accurate, which is the precision a travel feature needs.
TEAM_COORDS = {
    "ATL": (33.757, -84.396), "BOS": (42.366, -71.062), "BKN": (40.683, -73.975),
    "CHA": (35.225, -80.839), "CHI": (41.881, -87.674), "CLE": (41.496, -81.688),
    "DAL": (32.790, -96.810), "DEN": (39.749, -105.008), "DET": (42.341, -83.055),
    "GSW": (37.768, -122.388), "HOU": (29.751, -95.362), "IND": (39.764, -86.156),
    "LAC": (34.043, -118.267), "LAL": (34.043, -118.267), "MEM": (35.138, -90.051),
    "MIA": (25.781, -80.187), "MIL": (43.045, -87.917), "MIN": (44.979, -93.276),
    "NOP": (29.949, -90.082), "NYK": (40.751, -73.994), "OKC": (35.463, -97.515),
    "ORL": (28.539, -81.384), "PHI": (39.901, -75.172), "PHX": (33.446, -112.071),
    "POR": (45.532, -122.667), "SAC": (38.580, -121.500), "SAS": (29.427, -98.437),
    "TOR": (43.643, -79.379), "UTA": (40.768, -111.901), "WAS": (38.898, -77.021),
}

_EARTH_RADIUS_MI = 3958.8


def _cached_copy(fn):
    """Cache a loader's result, but hand every caller a fresh copy.

    A bare @lru_cache returns the SAME mutable DataFrame object to every caller in
    the process. One caller doing `df.drop(columns=..., inplace=True)` - or
    `df["x"] = ...` - then silently corrupts the cache for every later caller.

    That is not hypothetical: it produced order-dependent failures across
    independent model test files, where a suite passed in isolation and failed
    when run alongside its neighbours. Caching the CSV read is still worth it;
    returning a shared mutable frame is not.
    """
    cached = lru_cache(maxsize=None)(fn)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        return cached(*args, **kwargs).copy()

    wrapper.cache_clear = cached.cache_clear
    wrapper.cache_info = cached.cache_info
    return wrapper


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #

def _require_downloaded_data() -> None:
    """Raise unittest.SkipTest when the NBA data has not been downloaded.

    The 382 MB download is deliberately not committed, so a fresh clone and a CI
    runner both find nothing here. That used to surface as
    `ValueError: No objects to concatenate` during test collection, which failed
    the entire suite on both Python versions.

    Raising SkipTest from the loaders rather than from the test files is
    deliberate. Two runners execute this suite - `pytest` and
    `python -m unittest discover` - and they honour different skip mechanisms:
    pytest understands markers registered by a conftest hook, while unittest
    ignores pytest entirely and only understands SkipTest raised during setUp.
    Putting the check here makes one change satisfy both, instead of editing
    every test module and keeping the two in sync.

    SkipTest is a legitimate outcome, not a failure: these tests cannot run
    without an input the project does not ship.
    """
    # A DIRECTORY existing is not enough, and checking only ONE of them is not
    # enough either. An empty `game_logs/` passes an `is_dir()` check and every
    # glob returns nothing, so `pd.concat([])` raises from deep inside a loader -
    # which is exactly what this function exists to prevent, and what red-lined CI.
    #
    # A PARTIAL download fails the same way by a different route: `game_logs/`
    # populated but `team_gamelogs/` still empty passes a single-directory check
    # and then blows up on the second glob. So every directory the core loaders
    # read is checked for actual CSVs.
    missing = [d for d in _REQUIRED_DATA_DIRS
               if not glob.glob(os.path.join(DATA_ROOT, d, "*.csv"))]
    if not missing:
        return
    import unittest  # local import: this module is not otherwise test-coupled

    root_exists = Path(DATA_ROOT).is_dir()
    if not root_exists:
        detail = "(directory does not exist)"
    elif len(missing) == len(_REQUIRED_DATA_DIRS):
        detail = "(directory exists but holds no CSVs)"
    else:
        detail = f"(no CSVs in: {', '.join(missing)})"
    raise unittest.SkipTest(
        f"NBA data not found at {DATA_ROOT} {detail}. Download it with "
        f"`python scripts/download_nba_data.py`, or set NBA_DATA_ROOT to an "
        f"existing copy."
    )


def _paths(pattern: str) -> list[str]:
    _require_downloaded_data()
    return sorted(glob.glob(os.path.join(DATA_ROOT, pattern)))


def _parse_date(s: pd.Series) -> pd.Series:
    """GAME_DATE arrives in two shapes across the tables; handle both."""
    s = s.astype(str)
    out = pd.to_datetime(s, format="%m/%d/%Y %H:%M:%S", errors="coerce")
    if out.isna().all():
        out = pd.to_datetime(s.str.slice(0, 10), format="%Y-%m-%d", errors="coerce")
    return out


@_cached_copy
def load_team_games(season_type: str = "Regular Season") -> pd.DataFrame:
    """One row per team per game, Base box score. GAME_ID is str.

    Columns of interest: GAME_ID, GAME_DATE, TEAM_ABBREVIATION, MATCHUP, WL, PTS,
    SEASON_TYPE. `MATCHUP` reads "LAL vs. BOS" (LAL home) or "LAL @ BOS" (LAL away),
    which is the authoritative home/away signal - the schedule table is polluted
    with All-Star and preseason games.
    """
    df = pd.concat(
        [pd.read_csv(f, dtype={"GAME_ID": str}, low_memory=False)
         for f in _paths("game_logs/league_team_*.csv")],
        ignore_index=True,
    )
    df["GAME_DATE"] = _parse_date(df["GAME_DATE"])
    df = df[df["TEAM_ABBREVIATION"].isin(REAL_TEAMS)].copy()
    if season_type:
        df = df[df["SEASON_TYPE"] == season_type].copy()
    df = df.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ABBREVIATION"])
    return df.reset_index(drop=True)


@_cached_copy
def load_team_advanced(season_type: str = "Regular Season") -> pd.DataFrame:
    """Per-game team Advanced measures: ratings, pace, TS_PCT, and rebound shares.

    ``DREB_PCT`` is included deliberately. It is genuinely absent from
    ``team_gamelogs/four_factors_*.csv`` (which carries only ``OREB_PCT`` and
    ``OPP_OREB_PCT``), which is why the four-factor block was three-and-a-bit and
    why the rebounding features silently did not exist -- ``rolling_team_features``
    asked the four-factors table for ``DRB_PCT``, which it has never had.

    But ``DREB_PCT`` IS published, per-game, in this table at 97.2% coverage over
    28,328 team-games. It is NOT redundant with ``OREB_PCT``: measured over the
    26,855 rows carrying both, the correlation is **-0.017** (R^2 = 0.0003), so
    adding it adds a near-independent signal rather than a restatement.

    The two sum to 1.0 in the LEAGUE MEAN (mean residual 0.000011) and on 92% of
    individual rows deviate by more than 0.01, which is what makes them look
    complementary at a glance and misleading on inspection. They are shares of
    DIFFERENT denominators -- offensive rebounds over (own OREB + opponent DREB),
    defensive rebounds over (own DREB + opponent OREB) -- so they are not two views
    of one quantity and move independently within a game.
    """
    df = _read_measures("team_gamelogs/advanced_*.csv", season_type)
    keep = ["GAME_ID", "TEAM_ABBREVIATION", "SEASON_TYPE", "OFF_RATING", "DEF_RATING",
            "NET_RATING", "PACE", "TS_PCT", "OREB_PCT", "DREB_PCT"]
    return df[[c for c in keep if c in df.columns]].copy()


@_cached_copy
def load_team_fourfactors(season_type: str = "Regular Season") -> pd.DataFrame:
    """Per-game team Four Factors."""
    df = _read_measures("team_gamelogs/four_factors_*.csv", season_type)
    return df.drop(columns=[c for c in ("index", "Unnamed: 0") if c in df.columns],
                   errors="ignore")


@_cached_copy
def load_player_advanced(season_type: str = "Regular Season") -> pd.DataFrame:
    """Per-game player Advanced: MIN, ratings, TS_PCT, USG_PCT."""
    return _read_measures("player_gamelogs/advanced_*.csv", season_type)


@_cached_copy
def load_player_usage(season_type: str = "Regular Season") -> pd.DataFrame:
    """Per-game player Usage: MIN, USG_PCT, PCT_PTS."""
    return _read_measures("player_gamelogs/usage_*.csv", season_type)


def _read_measures(pattern: str, season_type: str) -> pd.DataFrame:
    df = pd.concat(
        [pd.read_csv(f, dtype={"GAME_ID": str}, low_memory=False)
         for f in _paths(pattern)],
        ignore_index=True,
    )
    df = df[df["TEAM_ABBREVIATION"].isin(REAL_TEAMS)].copy() \
        if "TEAM_ABBREVIATION" in df.columns else df
    if season_type and "SEASON_TYPE" in df.columns:
        df = df[df["SEASON_TYPE"] == season_type].copy()
    return df.reset_index(drop=True)


@_cached_copy
def load_schedule() -> pd.DataFrame:
    """Cleaned league schedule: finished NBA games only, current franchises only."""
    df = pd.concat(
        [pd.read_csv(f, dtype={"gameId": str}) for f in _paths("schedule/*.csv")],
        ignore_index=True,
    )
    df["date"] = pd.to_datetime(df["gameDate"], format="%m/%d/%Y %H:%M:%S",
                                 errors="coerce")
    df = df[df["gameStatus"] == 3]                      # trap 3: 3 == Final, never 1
    df = df[df["homeTeam_teamTricode"].isin(REAL_TEAMS) &
            df["awayTeam_teamTricode"].isin(REAL_TEAMS)]   # trap 4
    df = df[df["gameLabel"].isna()]                     # drop All-Star / preseason
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Canonical game table - the single artefact every model joins against
# --------------------------------------------------------------------------- #

@_cached_copy
def build_games(season_type: str = "Regular Season") -> pd.DataFrame:
    """One row per game: date, home, away, scores, and the binary label.

    Derived from the team game logs via MATCHUP, not the schedule, so it is free
    of All-Star/preseason contamination. Games where either side is missing, or
    the two rows disagree on the score, are dropped rather than silently repaired.
    """
    t = load_team_games(season_type)
    t = t[t["PTS"].notna() & (t["PTS"] > 0)].copy()

    # "LAL vs. BOS" -> LAL is home; "LAL @ BOS" -> LAL is away.
    # A well-formed game contributes exactly one row to each side; if it does not,
    # the game is malformed and is dropped rather than fanned out across the join.
    home = t[t["MATCHUP"].str.contains(" vs. ", na=False)].copy()
    away = t[t["MATCHUP"].str.contains(" @ ", na=False)].copy()
    assert not (t["MATCHUP"].str.contains(" vs. ", na=False) &
                t["MATCHUP"].str.contains(" @ ", na=False)).any(), \
        "a MATCHUP matched both home and away patterns"
    home = home.rename(columns={"TEAM_ABBREVIATION": "home", "PTS": "home_pts",
                                "PLUS_MINUS": "home_pm"})
    away = away.rename(columns={"TEAM_ABBREVIATION": "away", "PTS": "away_pts",
                                "PLUS_MINUS": "away_pm"})
    dup_h = home["GAME_ID"].duplicated().sum()
    dup_a = away["GAME_ID"].duplicated().sum()
    if dup_h or dup_a:
        import warnings
        warnings.warn(f"dropping {dup_h} duplicate home / {dup_a} duplicate away rows")
    home = home.drop_duplicates(subset=["GAME_ID"], keep=False)
    away = away.drop_duplicates(subset=["GAME_ID"], keep=False)

    g = home[["GAME_ID", "GAME_DATE", "SEASON", "home", "home_pts", "home_pm"]].merge(
        away[["GAME_ID", "away", "away_pts", "away_pm"]], on="GAME_ID", how="inner")
    g = g.dropna(subset=["home_pts", "away_pts"]).copy()
    g["home_pts"] = g["home_pts"].astype(float)
    g["away_pts"] = g["away_pts"].astype(float)
    # A duplicated GAME_ID on one side means the merge fanned out; drop those games.
    g = g.drop_duplicates(subset=["GAME_ID"], keep=False)
    g["y"] = (g["home_pts"] > g["away_pts"]).astype(int)
    g["margin"] = g["home_pts"] - g["away_pts"]
    g["total"] = g["home_pts"] + g["away_pts"]
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"])
    return g.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Causal rolling features
# --------------------------------------------------------------------------- #

def _ewm_by_team(df: pd.DataFrame, value: str, halflife: float) -> pd.Series:
    """Exponentially weighted mean of `value` per team, strictly past-only.

    The shift is done WITHIN each team. A plain `s[value].shift(1)` would let the
    first game of a team inherit the last value of whichever team sorted before it,
    contaminating every team's opening game.

    `min_periods` on the ewm itself does not stop the current row leaking in, so the
    current row is shifted out before weighting. This is the only sanctioned way to
    build a rolling feature in this repo.
    """
    s = df.sort_values(["team", "GAME_DATE"])
    shifted = s.groupby("team", sort=False)[value].shift(1)
    out = (shifted.groupby(s["team"], sort=False)
           .ewm(halflife=halflife, adjust=True).mean()
           .reset_index(level=0, drop=True))
    return out.reindex(s.index)


def rolling_team_features(games: pd.DataFrame, halflife: float = 10.0,
                          min_games: int = 5) -> pd.DataFrame:
    """Per-game rolling team form. Every column is strictly causal.

    Uses only information available BEFORE the game: each team's own past
    scoring/defence margin, its past net rating, pace and four factors, plus
    schedule-derived rest and travel. Returns one row per GAME_ID aligned to
    `games`.
    """
    t = load_team_games()
    adv = load_team_advanced()
    # Scope to exactly the games requested. Without this the function's output
    # depends on which other seasons happen to be on disk, and a caller cannot
    # reason about causality by truncating its own input.
    keep_ids = set(games["GAME_ID"])
    t = t[t["GAME_ID"].isin(keep_ids)]
    adv = adv[adv["GAME_ID"].isin(keep_ids)]

    # Long per-team-per-game frame with advanced measures attached.
    keep = ["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "PTS"]
    _adv_cols = [c for c in ("GAME_ID", "TEAM_ABBREVIATION", "OFF_RATING", "DEF_RATING",
                         "NET_RATING", "PACE", "TS_PCT", "OREB_PCT", "DREB_PCT")
             if c in adv.columns]
    m = t[keep].merge(
        adv[_adv_cols].drop_duplicates(
            subset=["GAME_ID", "TEAM_ABBREVIATION"]),
        on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")
    ff = load_team_fourfactors()
    ff = ff[ff["GAME_ID"].isin(keep_ids)]
    # OPP_* is the opponent's own characteristic, as this team faced it: the
    # eFG%, free-throw rate, turnover rate and offensive-rebound share of the
    # club across the table. Only OPP_EFG_PCT was ever read. The other three sit
    # in the same table, under the same header, in every downloaded season, and
    # no model in this repo has ever seen them - the same silent absence as
    # ORB_PCT/DRB_PCT, one column over.
    ffc = [c for c in ("EFG_PCT", "FTA_RATE", "TM_TOV_PCT", "ORB_PCT",
                       "OPP_EFG_PCT", "OPP_FTA_RATE", "OPP_TOV_PCT",
                       "OPP_OREB_PCT") if c in ff.columns]
    if ffc:
        m = m.merge(ff[["GAME_ID", "TEAM_ABBREVIATION"] + ffc].drop_duplicates(
            subset=["GAME_ID", "TEAM_ABBREVIATION"]),
            on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")
    m = m.rename(columns={"TEAM_ABBREVIATION": "team"}).sort_values(["team", "GAME_DATE"])

    # Per-team margin requires the opponent's score: derive from the game table.
    gp = games[["GAME_ID", "home", "away", "home_pts", "away_pts", "GAME_DATE"]]
    h = gp.rename(columns={"home": "team", "away_pts": "opp_pts"}).assign(pts=lambda d: d.home_pts)
    a = gp.rename(columns={"away": "team", "home_pts": "opp_pts"}).assign(pts=lambda d: d.away_pts)
    ha = pd.concat([h[["GAME_ID", "team", "pts", "opp_pts", "GAME_DATE"]],
                    a[["GAME_ID", "team", "pts", "opp_pts", "GAME_DATE"]]],
                   ignore_index=True)
    m = m.drop(columns=["pts", "opp_pts"], errors="ignore")
    m = m.merge(ha, on=["GAME_ID", "team"], how="left", suffixes=("", "_y"))
    m["margin"] = m["pts"] - m["opp_pts"]
    m = m.sort_values(["team", "GAME_DATE"]).reset_index(drop=True)

    m["games_played"] = m.groupby("team").cumcount()
    # BOTH rebound features were silently absent before this. The loop asked for
    # `ORB_PCT` and `DRB_PCT` from the four-factors table, which publishes
    # `OREB_PCT` and no `DRB_PCT` at all -- two wrong column names. The
    # `if col in m.columns` guard then skipped them without a word, so no model had
    # ever seen a rebounding feature and nothing reported it missing. Both now come
    # from the advanced table, which carries the coherent OREB/DREB pair under
    # identical definitions.
    for col, name in [("margin", "ewm_margin"), ("NET_RATING", "ewm_net_rating"),
                      ("OFF_RATING", "ewm_off_rating"), ("DEF_RATING", "ewm_def_rating"),
                      ("PACE", "ewm_pace"), ("TS_PCT", "ewm_ts_pct"),
                      ("EFG_PCT", "ewm_efg"), ("TM_TOV_PCT", "ewm_tov"),
                      ("FTA_RATE", "ewm_ftr"), ("OREB_PCT", "ewm_orb"),
                      ("DREB_PCT", "ewm_drb"),
                      ("OPP_EFG_PCT", "ewm_opp_efg"),
                      ("OPP_FTA_RATE", "ewm_opp_ftr"),
                      ("OPP_TOV_PCT", "ewm_opp_tov"),
                      ("OPP_OREB_PCT", "ewm_opp_orb")]:
        if col in m.columns:
            m[name] = _ewm_by_team(m, col, halflife)

    feat_cols = [c for c in m.columns if c.startswith("ewm_")]
    home_feats = m[m.team.isin(games.home.unique())][
        ["GAME_ID", "team", "games_played"] + feat_cols]
    away_feats = m[m.team.isin(games.away.unique())][
        ["GAME_ID", "team", "games_played"] + feat_cols]

    h = home_feats.rename(columns={c: f"home_{c}" for c in
                                   ["games_played"] + feat_cols})
    a = away_feats.rename(columns={c: f"away_{c}" for c in
                                   ["games_played"] + feat_cols})
    out = games[["GAME_ID", "home", "away", "GAME_DATE"]].merge(
        h, left_on=["GAME_ID", "home"], right_on=["GAME_ID", "team"], how="left")
    out = out.merge(a, left_on=["GAME_ID", "away"], right_on=["GAME_ID", "team"],
                    how="left", suffixes=("", "_away"))
    out = out.drop(columns=[c for c in ("team", "team_away") if c in out.columns])
    return out.drop(columns=["GAME_ID", "home", "away"])


def player_minutes(games: pd.DataFrame) -> pd.DataFrame:
    """Per-player per-game minutes, aligned to `games`, with dates attached.

    Shared so every model builds player-level expectations the same way. Returns
    the raw frame; `availability_features` does the causal work on top.
    """
    pa = load_player_advanced()
    cols = [c for c in ("GAME_ID", "PLAYER_ID", "TEAM_ABBREVIATION", "MIN")
            if c in pa.columns]
    df = pa[cols].copy()
    df = df.merge(games[["GAME_ID", "GAME_DATE"]].drop_duplicates("GAME_ID"),
                  on="GAME_ID", how="inner")
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")
    return df.dropna(subset=["MIN"]).sort_values(["PLAYER_ID", "GAME_DATE"])


def _ewm_by_player(df: pd.DataFrame, value: str, halflife: float) -> pd.Series:
    """Causal EWMA of `value` per player. Mirrors `_ewm_by_team` exactly.

    The shift is WITHIN each player. A frame-level `.shift(1)` would hand a
    player's debut the last value of whichever player sorted above it.
    """
    s = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    shifted = s.groupby("PLAYER_ID", sort=False)[value].shift(1)
    out = (shifted.groupby(s["PLAYER_ID"], sort=False)
           .ewm(halflife=halflife, adjust=True).mean()
           .reset_index(level=0, drop=True))
    return out.reindex(s.index)


def availability_features(games: pd.DataFrame, halflife: float = 15.0) -> pd.DataFrame:
    """Player-level EXPECTED minutes, summed over a roster known before tip-off.

    WHY THIS EXISTS
    ---------------
    Availability is the only mechanism measured in this project whose interval
    excludes zero for the right reason. Summing each player's expected minutes
    over the squad that appeared in the team's PREVIOUS game improves log loss by
    **+0.00375 nats, CI [+0.00201, +0.00553]**. Every other candidate mechanism -
    player ratings, the 52 rank columns, gradient boosting, recency weighting,
    the rebound block - is a null.

    It is worth far more than it looks because team-level season aggregates cannot
    express it: a team is not one number, and who is expected to consume the
    minutes is not visible in an average of margin and shooting.

    THE OBVIOUS VERSION OF THIS FEATURE LEAKS, AND THIS IS NOT IT
    -----------------------------------------------------------
    Summing expected minutes over the players who ACTUALLY played is contaminated:
    a team missing its star has a smaller sum, and who dresses is knowable only
    after tip-off. Measured, that contamination is worth +0.00109 nats with an
    interval STRADDLING ZERO, so it is not what drives the effect - but it is not
    zero, and it is the difference between an honest claim and an inflated one.

    So the sum here is over the squad that appeared in the team's previous game.
    That cannot depend on tonight's lineup. Coverage is 99.9% of team-games and
    NaN only for a team's first game on record, which is itself the evidence that a
    prior game is what is being used.

    DO NOT "IMPROVE" THIS BY SUMMING OVER ACTUAL PARTICIPANTS. That raises the
    measured effect from +0.00375 to +0.01081 and is a leak. Actual minutes are
    worth +0.01089 - roughly three times the honest number - and are unknowable
    before the game. See issue #30.

    Verified causal: truncating 9,199 future games changes every column here by
    0.000e+00, NaN pattern included.
    """
    pm = player_minutes(games)
    pm["expected_min"] = _ewm_by_player(pm, "MIN", halflife)

    squad = (pm.groupby(["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE"], as_index=False)
             .agg(squad=("PLAYER_ID", lambda s: frozenset(s))))
    squad = squad.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    squad["prev_squad"] = squad.groupby("TEAM_ABBREVIATION", sort=False)["squad"].shift(1)

    # Each player's expectation must be the value AS OF the game being predicted,
    # not their most recent value anywhere in the frame.
    #
    # A groupby().last() lookup - the obvious way to do this - holds each player's
    # expected minutes at their LAST game in the whole frame. Applied to a 2016
    # game that means handing it a 2026 number: roughly a decade of future
    # information, invisible in the output and producing an excellent-looking
    # score. Truncating the future exposes it immediately, which is exactly what
    # the causality check below is for.
    queries = squad.loc[squad["prev_squad"].notna()].copy()
    exploded = queries[["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "prev_squad"]].explode(
        "prev_squad").rename(columns={"prev_squad": "PLAYER_ID"})
    exploded = exploded.dropna(subset=["PLAYER_ID"])
    hist = (pm.dropna(subset=["expected_min"])
            .sort_values("GAME_DATE")[["PLAYER_ID", "GAME_DATE", "expected_min"]])
    # merge_asof requires both sides sorted by the join key AND matching dtypes on
    # it. `GAME_DATE` arrives as datetime64[us] from `games` and as object after
    # `explode`, so it is normalised on both sides explicitly rather than left to
    # pandas to guess -- a silent object/int key mismatch raises, but a silent
    # *coercion* would not.
    left = exploded[["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "PLAYER_ID"]].copy()
    left["GAME_DATE"] = pd.to_datetime(left["GAME_DATE"])
    right = hist[["PLAYER_ID", "GAME_DATE", "expected_min"]].copy()
    right["GAME_DATE"] = pd.to_datetime(right["GAME_DATE"])
    # `explode` pulls PLAYER_ID out of a frozenset, which makes it object dtype
    # while the source column is int64. merge_asof's `by` does an internal merge
    # on it, so both sides must match exactly or it raises.
    left["PLAYER_ID"] = left["PLAYER_ID"].astype("int64")
    right["PLAYER_ID"] = right["PLAYER_ID"].astype("int64")
    # `by` restricts the match to the same player; `allow_exact_matches=False`
    # enforces STRICTLY-before, so a game never sees its own value.
    asof = pd.merge_asof(
        left.sort_values("GAME_DATE"),
        right.sort_values("GAME_DATE"),
        left_on="GAME_DATE", right_on="GAME_DATE", by="PLAYER_ID",
        direction="backward", allow_exact_matches=False,
    )
    totals = (asof.dropna(subset=["expected_min"])
              .groupby(["GAME_ID", "TEAM_ABBREVIATION"], as_index=False)["expected_min"]
              .sum())
    squad = squad.merge(totals.rename(columns={"expected_min": "prev_sum"}),
                        on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")

    d = games[["GAME_ID", "home", "away"]]
    h = d.merge(squad[["GAME_ID", "TEAM_ABBREVIATION", "prev_sum"]],
                left_on=["GAME_ID", "home"],
                right_on=["GAME_ID", "TEAM_ABBREVIATION"], how="left"
                ).rename(columns={"prev_sum": "home_expected_minutes"})
    a = d.merge(squad[["GAME_ID", "TEAM_ABBREVIATION", "prev_sum"]],
                left_on=["GAME_ID", "away"],
                right_on=["GAME_ID", "TEAM_ABBREVIATION"], how="left"
                ).rename(columns={"prev_sum": "away_expected_minutes"})
    out = h[["GAME_ID", "home_expected_minutes"]].merge(
        a[["GAME_ID", "away_expected_minutes"]], on="GAME_ID")

    # The squad's expected total is ~240 minutes for both teams, so the raw
    # difference is mostly schedule noise. The contrast that carries signal is
    # minutes-per-player, which is a team-quality measure rather than a pace one.
    out["home_expected_minutes_diff"] = (
        out["home_expected_minutes"] - out["away_expected_minutes"])
    out["away_expected_minutes_diff"] = (
        -out["home_expected_minutes_diff"])
    return out.drop(columns=["GAME_ID"]).reset_index(drop=True)


def opponent_adjusted_features(games: pd.DataFrame,
                               halflife: float = 10.0) -> pd.DataFrame:
    """Team form adjusted for WHO the team played, plus a per-team home/away split.

    WHY THIS IS NOT ANOTHER TRANSFORMATION OF WHAT IS ALREADY THERE
    --------------------------------------------------------------
    Every decay experiment in this project failed: halflife 5 and 2 are worse than
    halflife 10, and the static season average is far worse. Those all reweight the
    SAME numbers. This block uses a quantity the existing feature set never reads
    at all — the quality of the opposition.

    `ewm_margin` is a team's raw past margin. A team that went 20-10 mostly against
    weak clubs has a better raw margin than one that went 20-10 against strong
    ones, and the feature cannot tell the difference. That is not a weighting
    choice; it is missing information, and no decay speed recovers it.

    Three blocks, all strictly causal:

    ``adj_margin``
        Each past game contributes that team's margin MINUS its opponent's own
        margin to date. Beating a good team therefore contributes less than
        beating a bad one, and the exponentially-weighted mean of the result is a
        form estimate relative to the competition actually faced.

    ``sos`` (strength of schedule)
        The exponentially-weighted mean of the opponents faced. Carried as its own
        feature rather than only being subtracted away, because "this team has run
        a gauntlet" and "this team has had an easy run" are separately informative.

    ``home_edge``
        The team's OWN home advantage — its home form minus its away form. The
        existing feature set has a single constant home indicator for every team
        in the league; the real quantity varies, and some teams genuinely play
        differently at home.

    Causality is the same construction as `_ewm_by_team`: shift WITHIN each team
    before weighting. Every opponent quantity is itself a trailing value, so using
    it introduces no future information. Verified by truncating the future, which
    must change nothing here either.

    WHAT THE HALFLIFE MEANS ON THE SPLIT COLUMNS - read before interpreting them
    -------------------------------------------------------------------------
    ``ewm_margin_home`` is built by setting the away rows to NaN and passing the
    column through the same `_ewm_by_team`. pandas' ``ewm`` runs with
    ``ignore_na=False`` by default, which means a NaN position contributes nothing
    to the weighted mean but STILL consumes decay. So the split columns are not an
    EWMA over home games with halflife 10; they are an EWMA whose halflife is
    counted in ALL games, and since roughly half a team's games are home games the
    effective halflife is about 20 *home games*, not 10.

    This is stated rather than fixed because changing it would invalidate the
    measured numbers, and because it is a defensible choice rather than a defect:
    a halflife measured in games played answers "how recently has this been true,
    counting the games this team did not play", which is a reasonable thing to
    want. But it does mean ``ewm_margin_home`` and ``ewm_margin_away`` carry a
    wider window and more noise than a reader would assume from the parameter -
    measured sd 6.29 and 5.84 against 5.46 for the pooled ``ewm_margin`` on the same
    games. That is the reason they are noisier, and it is worth knowing before
    treating them as sharper estimates of home form than they are.
    """
    gp = games[["GAME_ID", "GAME_DATE", "home", "away", "home_pts", "away_pts"]]
    h = gp.rename(columns={"home": "team", "away": "opp"})[
        ["GAME_ID", "GAME_DATE", "team", "opp", "home_pts", "away_pts"]]
    h["is_home"] = True
    h["margin"] = h["home_pts"] - h["away_pts"]     # the home team's own margin
    h["opp_margin"] = -h["margin"]                  # ...so the opponent's is the negation
    a = gp.rename(columns={"away": "team", "home": "opp"})[
        ["GAME_ID", "GAME_DATE", "team", "opp", "home_pts", "away_pts"]]
    a["is_home"] = False
    a["margin"] = a["away_pts"] - a["home_pts"]     # the away team's own margin
    a["opp_margin"] = -a["margin"]
    m = pd.concat([h, a], ignore_index=True)

    # Each row's margin is signed FROM THAT ROW'S OWN TEAM'S POINT OF VIEW. An
    # earlier version computed `home_pts - away_pts` for both rows, which is
    # correct for the home team and exactly negated for the away one. It did not
    # raise, and the column names and dtypes all looked right; it surfaced only
    # as a sanity check failing - the per-team home advantage came out at zero,
    # because averaging a right-signed series against a left-signed one cancels.
    m = m.sort_values(["team", "GAME_DATE"]).reset_index(drop=True)

    # Opponent strength AT THE TIME: the opponent's own trailing mean margin.
    # This is the one place that could leak, so it is built with the same
    # shift-within-team discipline and then re-keyed back onto this row.
    #
    # The frame is keyed on `opp`, not `team`, because it is one row per
    # (game, OPPONENT). `_ewm_by_team` groups by a column literally named
    # `team`, so the opponent frame is given one and renamed afterwards - the
    # grouping discipline has to be identical to the primary frame or the two are
    # not comparable, and a separate hand-rolled variant is exactly the kind of
    # duplication that drifts.
    opp_long = m[["GAME_ID", "opp", "GAME_DATE", "opp_margin"]].rename(
        columns={"opp": "team", "opp_margin": "opp_margin_to_date"})
    opp_long = opp_long.sort_values(["team", "GAME_DATE"]).reset_index(drop=True)
    opp_long["opp_margin_to_date"] = _ewm_by_team(opp_long, "opp_margin_to_date",
                                                 halflife)
    opp_long = opp_long.rename(columns={"team": "opp"})
    m = m.merge(opp_long[["GAME_ID", "opp", "opp_margin_to_date"]],
                on=["GAME_ID", "opp"], how="left")

    # Margin relative to the opposition faced, and the opposition itself.
    m["adj_margin"] = m["margin"] - m["opp_margin_to_date"]
    m["opp_quality"] = m["opp_margin_to_date"]

    # Home and away form separately, then the team's OWN home advantage. The
    # two are split on a strictly-past flag and weighted within each side.
    m["home_adj"] = np.where(m["is_home"], m["adj_margin"], np.nan)
    m["away_adj"] = np.where(~m["is_home"], m["adj_margin"], np.nan)
    m["home_raw"] = np.where(m["is_home"], m["margin"], np.nan)
    m["away_raw"] = np.where(~m["is_home"], m["margin"], np.nan)

    m["games_played"] = m.groupby("team").cumcount()
    for col, name in (("adj_margin", "ewm_adj_margin"),
                      ("opp_quality", "ewm_sos"),
                      ("home_adj", "ewm_adj_home"),
                      ("away_adj", "ewm_adj_away"),
                      ("home_raw", "ewm_margin_home"),
                      ("away_raw", "ewm_margin_away")):
        m[name] = _ewm_by_team(m, col, halflife)
    m["home_edge"] = m["ewm_margin_home"] - m["ewm_margin_away"]

    # `games_played` is deliberately NOT emitted here. `rolling_team_features`
    # already produces it under the same names, and two sources writing one
    # feature means whichever is concatenated second is silently dropped - which
    # source wins becomes a property of list order rather than of the data.
    keep = ["GAME_ID", "team", "ewm_adj_margin", "ewm_sos",
            "ewm_adj_home", "ewm_adj_away", "ewm_margin_home", "ewm_margin_away",
            "home_edge"]
    hside = m[m.is_home][keep].rename(columns={c: f"home_{c}" for c in keep
                                               if c not in ("GAME_ID", "team")})
    aside = m[~m.is_home][keep].rename(columns={c: f"away_{c}" for c in keep
                                                if c not in ("GAME_ID", "team")})
    out = games[["GAME_ID", "home", "away"]].reset_index(drop=True).merge(
        hside, left_on=["GAME_ID", "home"], right_on=["GAME_ID", "team"], how="left")
    out = out.merge(aside, left_on=["GAME_ID", "away"], right_on=["GAME_ID", "team"],
                    how="left", suffixes=("", "_a"))
    out = out.drop(columns=[c for c in ("team", "team_a") if c in out.columns])
    if len(out) != len(games):
        raise AssertionError(
            f"opponent_adjusted_features produced {len(out)} rows for "
            f"{len(games)} games - join fanned out")
    return out.drop(columns=["GAME_ID", "home", "away"]).reset_index(drop=True)


def player_elo_ratings(games: pd.DataFrame, k: float = 16.0,
                       mean: float = 1500.0, scale: float = 400.0,
                       regress: float = 0.02) -> pd.DataFrame:
    """A causal Elo rating per PLAYER, from team results.

    WHY THIS EXISTS
    ---------------
    Every team-level feature in this repo is a statement about an institution. A
    rating is a statement about a person. Those are different quantities, and the
    team aggregates cannot express what happens when a club's best player is
    unavailable - which is the one mechanism in this project that is NOT a null
    (availability, +0.00382 nats). Player ratings give the availability mechanism
    something to act on: it can discount a roster by *who* is in it rather than
    only *how many* minutes are expected.

    `src/Models/player_ratings.py` already tested RAPM-style player ratings and
    found them a null (+0.00151 CI [-0.00059, +0.00358], and a null across all
    three rating families and both estimation methods). This is a different
    estimator - online, sequential, and far more parsimonious - and it is aimed at
    a different question: not "can a player's rating predict a game", but "does a
    rotation-weighted sum of them, over the players expected to play".

    THE UPDATE
    ----------
    For each game in date order, for every player who played:

        expected = logistic((r_player - r_opponent_mean) / scale)
        r_player += k * (team_won - expected) + regress * (mean - r_player)

    The opponent's strength is the MINUTES-WEIGHTED mean rating of the players
    who actually shared the floor, not the unweighted mean, because a team is not
    eleven equally important participants and an unweighted average lets a
    garbage-time player speak for a starter.

    THE REVERSION TERM, AND WHY IT IS NOT OPTIONAL
    ----------------------------------------------
    Without `regress`, ratings random-walk: a hot player's number climbs until the
    arithmetic can no longer express it, and early-season games become a function
    of whatever happened in 2017. With 13,199 games and 1,554 players a small pull
    toward `mean` every game is what keeps the scale meaningful across seasons.

    KNOWN LIMITATION, STATED UP FRONT
    ----------------------------------
    The only outcome available here is the TEAM result. A player's rating is
    therefore partly a measure of their teammates - a star on a weak club is
    charged for losing games he did not cause. This is the standard weakness of
    Elo applied to a team sport and is the main reason the previous player-rating
    attempt came back a null. It is a reason to be sceptical of the result, not a
    reason to skip the measurement.

    Causality: ratings are read BEFORE the game and written AFTER it, and games
    are processed in date order, so the value attached to a game is the rating the
    player carried into it.
    """
    pa = player_minutes(games)[["GAME_ID", "GAME_DATE", "PLAYER_ID",
                               "TEAM_ABBREVIATION", "MIN"]].copy()
    if not len(pa):
        return pd.DataFrame(index=games.index)

    outcome = games[["GAME_ID", "GAME_DATE", "home", "away", "home_pts",
                     "away_pts"]].copy()
    outcome["home_win"] = (outcome["home_pts"] > outcome["away_pts"]).astype(float)
    win = dict(zip(outcome["GAME_ID"], outcome["home_win"]))

    pa["win"] = pa["GAME_ID"].map(win)
    pa = pa.dropna(subset=["win"])
    # A player who has not been seen before starts at the league mean. Starting at
    # zero would read as catastrophic rather than unknown, and the first few games
    # of every season are almost entirely unknown players.
    ratings: dict[int, float] = {}
    out_rows = []
    # Date order; GAME_ID breaks ties so two same-day games are deterministic.
    for _, grp in pa.sort_values(["GAME_DATE", "GAME_ID"]).groupby(
            ["GAME_DATE", "GAME_ID"], sort=True):
        teams = grp.groupby("TEAM_ABBREVIATION")["PLAYER_ID"].apply(list).to_dict()
        if len(teams) < 2:
            continue

        # Minutes-weighted opponent strength, computed from PRE-game ratings.
        team_strength, team_win, team_min = {}, {}, {}
        for team, pids in teams.items():
            sub = grp.loc[grp["TEAM_ABBREVIATION"] == team]
            team_win[team] = float(sub["win"].iloc[0])
            mins = dict(zip(sub["PLAYER_ID"], sub["MIN"]))
            team_min[team] = mins
            vals = np.array([ratings.get(p, mean) for p in pids], dtype=float)
            w = np.array([mins.get(p, 0.0) for p in pids], dtype=float)
            team_strength[team] = (float(np.average(vals, weights=w))
                                   if w.sum() > 0 else float(vals.mean()))

        for team, pids in teams.items():
            opp = [team_strength[t] for t in teams if t != team]
            opp_strength = float(np.mean(opp))
            gid = grp["GAME_ID"].iloc[0]
            gdate = grp["GAME_DATE"].iloc[0]
            for p in pids:
                r = ratings.get(p, mean)
                # The rating carried INTO the game, which is the only value a
                # prediction made before tip-off is allowed to read.
                out_rows.append({"GAME_ID": gid, "GAME_DATE": gdate,
                                 "PLAYER_ID": p,
                                 "TEAM_ABBREVIATION": team, "elo": r,
                                 "MIN": float(team_min[team].get(p, np.nan))})
                expected = 1.0 / (1.0 + 10.0 ** (-(r - opp_strength) / scale))
                ratings[p] = r + (k * (team_win[team] - expected)
                                  + regress * (mean - r))

    if not out_rows:
        return pd.DataFrame(columns=["GAME_ID", "PLAYER_ID",
                                     "TEAM_ABBREVIATION", "elo"])
    out = pd.DataFrame(out_rows)
    # Align to the caller's game order rather than to processing order, so a
    # caller can rely on positional alignment with `games`.
    order = {g: i for i, g in enumerate(games["GAME_ID"].to_numpy())}
    out["_order"] = out["GAME_ID"].map(order)
    out = out.dropna(subset=["_order"]).sort_values(["_order", "PLAYER_ID"])
    return out.drop(columns=["_order"]).reset_index(drop=True)


def player_elo_features(games: pd.DataFrame, k: float = 16.0,
                        halflife: float = 15.0, top_n: int = 5,
                        roster_mode: str = "all_prior"
                        ) -> pd.DataFrame:
    """Rotation-weighted aggregate of the player Elo ratings, per side.

    ``*_elo_available``
        SUM of ``elo * expected_minutes`` over the players expected to be
        available. This is the quantity the availability result predicts should
        carry signal: a team whose stars are unavailable loses total
        quality-minutes before tip-off.

    ``*_elo_top{n}``
        Sum of the ``top_n`` highest ratings among the expected-available squad. A
        crude but honest proxy for "who starts" - no lineup data ships with this
        project - and it behaves differently from the total, because losing a
        28-minute rotation player barely moves it while losing a star moves it a
        lot.

    ``*_elo_mean``
        Minutes-weighted mean rating of the expected-available squad. Scale-free,
        so it separates "this roster is good" from "this roster is deep".

    Availability here means EXPECTED availability, from trailing minutes. The
    dataset's ``AVAILABLE_FLAG`` is written after the game and would be a leak;
    this is the same discipline ``availability_features`` already applies.
    """
    elo_long = player_elo_ratings(games, k=k)
    if not len(elo_long):
        return pd.DataFrame(index=games.index)

    # Expected minutes, from the player's own past appearances only.
    elo_long = elo_long.sort_values(["PLAYER_ID", "GAME_DATE"])
    elo_long["exp_min"] = _ewm_by_player(elo_long, "MIN", halflife)

    # ---------------------------------------------------------------------
    # THE ROSTER MUST BE KNOWABLE BEFORE TIP-OFF. THIS IS THE WHOLE FIX.
    # ---------------------------------------------------------------------
    # An earlier version aggregated over `elo_long` directly, which contains one
    # row per player WHO ACTUALLY PLAYED that game. The weights were honest
    # trailing values, so the feature looked clean - and it measured -0.00645 nats
    # at |z| 5.18, by a wide margin the largest effect in this project.
    #
    # It was leaking, and the leak is in the MEMBERSHIP rather than the weights.
    # Nobody knows before a game which ten men will be on the floor. Summing over
    # the actual participants tells the model who played, which is the same
    # information the honest availability feature had to be careful about:
    # availability_features is worth +0.00382 honestly and +0.01089 when actual
    # minutes are substituted. An effect that large, from a player-level feature
    # with post-game membership, should be assumed to be that leak until proven
    # otherwise.
    #
    # So the roster is rebuilt from the past: for team X's game t, the available
    # roster is every player with an appearance for X STRICTLY BEFORE t, carrying
    # the Elo and trailing expected minutes from that most recent appearance.
    # A player who misses game t entirely is still in the sum, discounted by his
    # own trailing minutes - which is the entire point of the block.
    team_games = (games[["GAME_ID", "GAME_DATE", "home", "away"]]
                  .melt(id_vars=["GAME_ID", "GAME_DATE"],
                        value_vars=["home", "away"], value_name="team")
                  .sort_values(["team", "GAME_DATE"])
                  .reset_index(drop=True))

    # `top_n` names the number of highest-rated players reported as `elo_top5`,
    # and separately bounds the `top_minutes` roster. Keeping them the same is
    # deliberate: the reported "best five" and the cheapest lineup proxy should
    # not quietly disagree about what five means.
    roster_top_n = top_n
    available = _asof_roster(team_games, elo_long, roster=roster_mode,
                             top_n=roster_top_n)
    available["elo_x_min"] = available["elo"] * available["exp_min"]

    g = available.groupby(["GAME_ID", "TEAM_ABBREVIATION"]).agg(
        elo_available=("elo_x_min", "sum"),
        elo_exp_minutes=("exp_min", "sum"),
    ).reset_index()
    g["elo_mean"] = g["elo_available"] / g["elo_exp_minutes"].replace(0, np.nan)

    # Top-n by rating among the players EXPECTED to be available - the same
    # pre-game roster, not the actual participants.
    top = (available[available["exp_min"] > 0]
           .sort_values(["GAME_ID", "TEAM_ABBREVIATION", "elo"],
                        ascending=[True, True, False])
           .groupby(["GAME_ID", "TEAM_ABBREVIATION"])
           .head(top_n)
           .groupby(["GAME_ID", "TEAM_ABBREVIATION"])["elo"].sum()
           .rename(f"elo_top{top_n}").reset_index())
    g = g.merge(top, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")

    # Pick each side's value out by keying on (GAME_ID, team). The obvious
    # alternative - merge the per-team table onto the games frame and filter on
    # `home.notna()` - produces two rows per game and then depends on
    # `drop_duplicates` to collapse them, which is an assumption about the data
    # rather than a lookup. A MultiIndex reindex is exact and order-preserving.
    g = g.set_index(["GAME_ID", "TEAM_ABBREVIATION"])
    if not g.index.is_unique:
        dupes = g.index[g.index.duplicated()].tolist()[:3]
        raise AssertionError(
            f"player Elo produced duplicate (game, team) keys: {dupes}. A team "
            f"cannot play itself twice, so the aggregation is wrong.")
    gi = games[["GAME_ID", "home", "away"]].reset_index(drop=True)

    out = pd.DataFrame(index=gi.index)
    for side in ("home", "away"):
        idx = pd.MultiIndex.from_arrays([gi["GAME_ID"], gi[side]])
        for c in ("elo_available", "elo_mean", f"elo_top{top_n}"):
            out[f"{side}_{c}"] = g[c].reindex(idx).to_numpy()

    if len(out) != len(games):
        raise AssertionError(
            f"player_elo_features produced {len(out)} rows for {len(games)} games")
    return out.reset_index(drop=True)




def _asof_roster(team_games: pd.DataFrame, appearances: pd.DataFrame,
                 roster: str = "all_prior", top_n: int = 5) -> pd.DataFrame:
    """Which players count as available for a game, under a stated assumption.

    THE QUESTION THIS PARAMETER ANSWERS
    -----------------------------------
    "Available" is not a fact a model can look up before tip-off. It is an
    assumption, and different assumptions give materially different team totals.
    Rather than pick one and hope, the assumption is a parameter and every
    variant is scored against the same target:

    ``all_prior``
        Everyone with any earlier appearance for the team, weighted by trailing
        expected minutes. The most permissive, and the default. Because a
        long-departed player's trailing EWMA decays rather than vanishing, this
        sums to roughly 6x a real rotation.

    ``prev_game``
        Only the players who played in the team's immediately preceding game. The
        tightest pre-game assumption available without an API, and the closest
        cheap analogue of a confirmed lineup.

    ``top_minutes``
        The ``top_n`` players by trailing expected minutes. A standing proxy for
        the starting five: no API call, but it cannot notice that a starter is
        sitting out tonight, which is exactly what a real lineup would tell you.

    ``confirmed``
        The actual submitted starting five, if a backfill supplied it. NOT
        computable from this dataset, and the reason is a prediction-point
        question rather than a data question: starting lineups are filed 30-60
    minutes before tip-off. They are therefore legitimate for a tip-off
    prediction and NOT available to a model run that morning. That makes it a
    different product, not a better feature, and it is a parameter rather than a
    default so that distinction has to be made explicitly.

    Every player with a STRICTLY EARLIER appearance for a team, at that game.

    For each (team, player), the player's Elo and trailing expected minutes are
    carried forward from their most recent appearance with that team onto each of
    that team's games that came afterwards.

    ``allow_exact_matches=False`` is load-bearing. A player's appearance in game t
    must not supply the rating used to predict game t; with exact matches allowed, a
    debutant's first game would be predicted from a rating computed using that same
    game's result - which is the leak this function exists to remove.

    The result is the PRE-GAME roster, which is strictly larger than the actual
    lineup because it includes every player who could plausibly be available.
    """
    ap = appearances[["GAME_DATE", "PLAYER_ID", "TEAM_ABBREVIATION", "elo",
                      "exp_min"]].sort_values("GAME_DATE")

    out = []
    for team, grp in ap.groupby("TEAM_ABBREVIATION", sort=False):
        t_games = team_games[team_games["team"] == team].sort_values("GAME_DATE")
        if not len(t_games):
            continue
        for _player, pgrp in grp.groupby("PLAYER_ID", sort=False):
            merged = pd.merge_asof(
                t_games,
                pgrp.sort_values("GAME_DATE"),
                left_on="GAME_DATE", right_on="GAME_DATE",
                direction="backward", allow_exact_matches=False,
            )
            merged = merged.dropna(subset=["PLAYER_ID"])
            if len(merged):
                out.append(merged)

    if not out:
        return pd.DataFrame(columns=["GAME_ID", "GAME_DATE", "team",
                                     "PLAYER_ID", "elo", "exp_min"])
    r = pd.concat(out, ignore_index=True)
    # One row per (game, team, player): a player may have several past
    # appearances, and only the most recent may be carried forward.
    dupes = r.duplicated(subset=["GAME_ID", "team", "PLAYER_ID"]).sum()
    if dupes:
        raise AssertionError(
            f"roster carries {dupes} duplicate (game, team, player) rows; the "
            f"as-of merge did not collapse to the latest prior appearance")
    if roster not in ("all_prior", "prev_game", "top_minutes", "confirmed"):
        raise KeyError(f"unknown roster assumption {roster!r}; known: "
                       f"all_prior, prev_game, top_minutes, confirmed")
    if roster == "all_prior":
        return r

    r = r.sort_values(["team", "GAME_DATE", "GAME_ID"])
    if roster == "prev_game":
        # The player's most recent appearance must have been the team's own
        # previous GAME. Keyed on the team's game sequence rather than the
        # calendar, so a long gap - an All-Star break, a cancelled game - does not
        # silently widen the window into "whoever turned up most recently".
        #
        # The sequence has to be assigned per GAME first. Doing it with
        # `groupby("team").cumcount()` counts ROWS, and this frame has one row per
        # (game, player), so the counter ran on player appearances and never
        # equalled a game index. The result was an all-NaN column that looked like
        # a data problem rather than an off-by-one.
        seq = team_games[["team", "GAME_ID"]].copy()
        seq["_seq"] = seq.groupby("team").cumcount()
        r = r.merge(seq, on=["team", "GAME_ID"], how="left")
        last_before = r.groupby(["team", "PLAYER_ID"])["_seq"].shift(1)
        return (r[last_before == r["_seq"] - 1]
                .drop(columns=["_seq"]).reset_index(drop=True))

    # top_minutes
    return (r[r["exp_min"] > 0]
            .sort_values(["GAME_ID", "team", "exp_min"], ascending=[True, True, False])
            .groupby(["GAME_ID", "team"]).head(top_n)
            .reset_index(drop=True))


def rest_and_travel(games: pd.DataFrame) -> pd.DataFrame:
    """Rest days, back-to-back, 3-in-4, and great-circle travel distance.

    All four are strictly known before tip-off from the schedule, so none of them
    leak. Travel is measured between the two clubs' home arenas.
    """
    g = games.copy()
    g["gid"] = range(len(g))
    # Long frame of every (team, game) appearance.
    # Long frame of every (team, game) appearance. `home` is kept intact on both
    # blocks - renaming it to `team` in the first block would leave `home` NaN there
    # and silently make every row look like an away appearance.
    long = pd.concat([
        g[["GAME_ID", "GAME_DATE", "gid", "home"]].assign(team=lambda d: d["home"]),
        g[["GAME_ID", "GAME_DATE", "gid", "away"]].assign(team=lambda d: d["away"]),
    ], ignore_index=True)
    long["is_home"] = long["team"] == long["home"]
    long = long.sort_values(["team", "GAME_DATE"])
    long["prev_date"] = long.groupby("team")["GAME_DATE"].shift(1)
    long["rest_days"] = (long["GAME_DATE"] - long["prev_date"]).dt.days

    # 3-in-4: count that team's games in the 3 preceding days. Computed per ROW, not
    # via a gid-keyed dict: each game has two teams, so a gid-keyed map lets one
    # team's count overwrite the other's and makes home/away come out identical.
    long["games_in_last_3"] = 0
    for _team, _grp in long.groupby("team", sort=False):
        _dates = set(_grp["GAME_DATE"])
        long.loc[_grp.index, "games_in_last_3"] = [
            int(sum((_d - pd.Timedelta(days=_k)) in _dates for _k in (1, 2, 3)))
            for _d in _grp["GAME_DATE"]
        ]

    h = long[long["is_home"]].set_index("gid")
    a = long[~long["is_home"]].set_index("gid")

    def dist(x, y):
        (la1, lo1), (la2, lo2) = TEAM_COORDS[x], TEAM_COORDS[y]
        p1, p2 = np.radians(la1), np.radians(la2)
        dp, dl = p2 - p1, np.radians(lo2 - lo1)
        a_ = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        return 2 * _EARTH_RADIUS_MI * np.arcsin(np.sqrt(a_))

    out = pd.DataFrame(index=g.index)
    out["home_rest_days"] = h["rest_days"].reindex(g.index)
    out["away_rest_days"] = a["rest_days"].reindex(g.index)
    out["home_back_to_back"] = (out["home_rest_days"] == 1).astype(float)
    out["away_back_to_back"] = (out["away_rest_days"] == 1).astype(float)
    out["home_in_3_of_4"] = h["games_in_last_3"].reindex(g.index).fillna(0).astype(float)
    out["away_in_3_of_4"] = a["games_in_last_3"].reindex(g.index).fillna(0).astype(float)
    out["travel_miles"] = [dist(hh, aa) for hh, aa in zip(g["home"], g["away"])]
    out["is_home"] = 1.0
    return out


# --------------------------------------------------------------------------- #
# Validation + evaluation
# --------------------------------------------------------------------------- #

def walk_forward_split(games: pd.DataFrame, n_splits: int = 5,
                       initial_frac: float = 0.5,
                       gap: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Chronological walk-forward splits by date.

    The first `initial_frac` of games are training-only. The remainder is cut
    into `n_splits` contiguous test blocks; split k trains on everything strictly
    before its test block and never sees it. Every split therefore has a non-empty
    training set - an earlier version tested the first block with no training data
    at all, which silently evaluates a model on its cold start.

    `gap` removes the `gap` games immediately before each test block (a purging
    interval) so features recomputed at the boundary cannot peek across.
    """
    n = len(games)
    first_test = int(n * initial_frac)
    bounds = np.linspace(first_test, n, n_splits + 1).astype(int)
    out = []
    for i in range(1, n_splits + 1):
        lo, hi = bounds[i - 1], bounds[i]
        if hi <= lo:
            continue
        test = np.arange(lo, hi)
        train = np.arange(0, max(0, lo - gap))
        if len(train) and len(test):
            out.append((train, test))
    return out


def evaluate(y_true, p_pred, bins: int = 10) -> dict:
    """Log loss, Brier, AUC, accuracy and a calibration table.

    Accuracy alone is not a usable judge for this task (the majority class is
    ~54%), so log loss and Brier are the primary numbers.
    """
    from sklearn.metrics import (roc_auc_score, brier_score_loss, log_loss,
                                 accuracy_score)
    y_true = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p_pred, dtype=float), 1e-6, 1 - 1e-6)
    out = {
        "n": int(len(y_true)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, p)),
        "accuracy": float(accuracy_score(y_true, (p > 0.5).astype(int))),
    }
    try:
        out["auc"] = float(roc_auc_score(y_true, p))
    except ValueError:
        out["auc"] = float("nan")
    edges = np.linspace(0, 1, bins + 1)
    which = np.digitize(p, edges[1:-1])
    cal = []
    for b in range(bins):
        m = which == b
        if m.sum() == 0:
            continue
        cal.append({
            "bin": b,
            "lo": float(edges[b]), "hi": float(edges[b + 1]),
            "n": int(m.sum()),
            "mean_pred": float(p[m].mean()),
            "actual": float(y_true[m].mean()),
        })
    out["calibration"] = cal
    return out


def paired_bootstrap_delta(y, p_a, p_b, n_boot: int = 2000,
                           seed: int = 0) -> dict:
    """Bootstrap the log-loss difference `loss(a) - loss(b)`.

    SIGN CONVENTION, and this has been got wrong in this repo before:
    a POSITIVE delta means **b is better** than a. Always read the sign with
    that in mind. Returns the point estimate plus a percentile CI.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    a = np.clip(np.asarray(p_a, float), 1e-6, 1 - 1e-6)
    b = np.clip(np.asarray(p_b, float), 1e-6, 1 - 1e-6)
    n = len(y)
    def _ll(p):
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    d = _ll(a) - _ll(b)
    stats = []
    idx = rng.integers(0, n, size=(n_boot, n))
    for i in range(n_boot):
        j = idx[i]
        ya = y[j]
        stats.append(
            -(ya * np.log(a[j]) + (1 - ya) * np.log(1 - a[j])).mean()
            - (-(ya * np.log(b[j]) + (1 - ya) * np.log(1 - b[j]))).mean())
    stats = np.asarray(stats)
    return {
        "delta": float(d), "ci_lo": float(np.percentile(stats, 2.5)),
        "ci_hi": float(np.percentile(stats, 97.5)),
        "p_positive": float((stats > 0).mean()),
        "n_boot": n_boot,
    }


if __name__ == "__main__":
    g = build_games()
    print(f"games: {len(g):,}  {g.GAME_DATE.min().date()} .. {g.GAME_DATE.max().date()}")
    print(f"home win rate: {g.y.mean():.4f}   seasons: {sorted(g.SEASON.unique())}")
    print(f"avg margin {g.margin.mean():+.2f}  avg total {g.total.mean():.1f}")
    r = rest_and_travel(g)
    print(f"\nrest_and_travel -> {r.shape[1]} cols; NaNs: {int(r.isna().sum().sum())}")
    print(r.describe().T[["mean", "min", "max"]].to_string())
    f = rolling_team_features(g)
    print(f"\nrolling features -> {f.shape[1]} cols")
    print(f"  NaN share: {f.isna().mean().mean():.3f}")
    print(g[["GAME_DATE", "home", "away", "home_pts", "away_pts", "y"]].head().to_string(index=False))