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
"""
from __future__ import annotations

import glob
import os
import re
from functools import lru_cache, wraps

import numpy as np
import pandas as pd

DATA_ROOT = os.environ.get(
    "NBA_DATA_ROOT", "/home/Joel/Desktop/Legit_NBA_Predictions/data/nba"
)

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

def _paths(pattern: str) -> list[str]:
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
    """Per-game team Advanced measures: OFF_RATING, DEF_RATING, NET_RATING, PACE, TS_PCT."""
    df = _read_measures("team_gamelogs/advanced_*.csv", season_type)
    keep = ["GAME_ID", "TEAM_ABBREVIATION", "SEASON_TYPE", "OFF_RATING", "DEF_RATING",
            "NET_RATING", "PACE", "TS_PCT"]
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
    m = t[keep].merge(
        adv[["GAME_ID", "TEAM_ABBREVIATION", "OFF_RATING", "DEF_RATING",
             "NET_RATING", "PACE", "TS_PCT"]].drop_duplicates(
            subset=["GAME_ID", "TEAM_ABBREVIATION"]),
        on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")
    ff = load_team_fourfactors()
    ff = ff[ff["GAME_ID"].isin(keep_ids)]
    ffc = [c for c in ("EFG_PCT", "FTA_RATE", "TM_TOV_PCT", "OPP_EFG_PCT",
                       "ORB_PCT", "DRB_PCT") if c in ff.columns]
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
    for col, name in [("margin", "ewm_margin"), ("NET_RATING", "ewm_net_rating"),
                      ("OFF_RATING", "ewm_off_rating"), ("DEF_RATING", "ewm_def_rating"),
                      ("PACE", "ewm_pace"), ("TS_PCT", "ewm_ts_pct"),
                      ("EFG_PCT", "ewm_efg"), ("TM_TOV_PCT", "ewm_tov"),
                      ("FTA_RATE", "ewm_ftr"), ("ORB_PCT", "ewm_orb"),
                      ("DRB_PCT", "ewm_drb")]:
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