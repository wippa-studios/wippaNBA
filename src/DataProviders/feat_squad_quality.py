"""Squad-quality features: what the PLAYERS who are expected to play are worth.

WHY THIS EXISTS
---------------
The 43-column E14 frame reads team-level boxes. Every rating in it is a mean
over a team's own past games, so it can say how a team has been playing and
nothing about who is producing it. `availability_features` established the
mechanism that escapes that: aggregate over the squad that appeared in the
team's PREVIOUS game, which is a set that is finished before the game being
predicted. That single change is worth +0.0039 nats and is the largest honest
effect in the project.

This module asks the next question along the same axis. Availability sums
EXPECTED MINUTES over that squad, which is a workload, not a quality. The
quantity a box score cannot express is: how good were the men who played last
game, and how concentrated is the load on them.

THE MEMBERSHIP TRAP
-------------------
This project has been caught by the membership version of leakage twice, most
notably player_elo at |z| 5.18. The rule here is absolute and mechanical:

  A row may only enter the aggregate for game t if that row's existence was
  known before t. The previous game's squad qualifies. Tonight's squad does not,
  and neither does "everyone who has ever played for this team" unless the
  roster is rebuilt as an as-of lookup (`all_prior`), where a player who missed
  the last month still carries his own trailing value rather than dropping out.

Every column is assembled by `pd.merge_asof(..., allow_exact_matches=False)` on
the PLAYER_ID stream, so a game can never see its own value, and the roster is
built by `shift(1)` over the team's own game sequence. `Tests/test_leakage_gate.py`
plus `scripts/round3_verify.py` check both mechanically.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _c in (ROOT, ROOT / "src"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from src.DataProviders import nba_data as nd  # noqa: E402

# Per-player quality measures. Each is a per-game box column that already exists
# in the loaded player frame; NONE of it is read anywhere in the E14 frame.
# Chosen for being about the PLAYER rather than the TEAM:
#   NET_RATING - on-court impact, already possession-adjusted upstream
#   PIE        - share of team production while on the floor
#   TS_PCT     - individual scoring efficiency
#   USG_PCT    - how much of the offence goes through him (star concentration)
#   AST_TO     - playmaking
QUALITY_COLS = ("NET_RATING", "PIE", "TS_PCT", "USG_PCT", "AST_TO")


def _ewm_shift(df: pd.DataFrame, col: str, halflife: float) -> pd.Series:
    """Causal EWMA of `col` per player: shift WITHIN player, then weight."""
    s = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    shifted = s.groupby("PLAYER_ID", sort=False)[col].shift(1)
    out = (shifted.groupby(s["PLAYER_ID"], sort=False)
           .ewm(halflife=halflife, adjust=True).mean()
           .reset_index(level=0, drop=True))
    return out.reindex(s.index)


def _player_streams(games: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """One row per player appearance, with every value already causal.

    The shift happens inside `_ewm_shift`, so the value attached to a player's
    appearance in game t is built ONLY from that player's games strictly before
    t. No post-game box score reaches its own feature.
    """
    pa = nd.load_player_advanced()
    keep = ["PLAYER_ID", "GAME_ID", "TEAM_ABBREVIATION", "MIN",
            *QUALITY_COLS]
    df = pa[keep].copy()
    df = df.merge(games[["GAME_ID", "GAME_DATE"]].drop_duplicates("GAME_ID"),
                  on="GAME_ID", how="inner")
    for c in ["MIN", *QUALITY_COLS]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    df["exp_min"] = _ewm_shift(df, "MIN", halflife)
    for c in QUALITY_COLS:
        df[f"q_{c}"] = _ewm_shift(df, c, halflife)
    return df


def _prev_squad_roster(streams: pd.DataFrame) -> pd.DataFrame:
    """The squad that appeared in each team-game, and the one before it.

    `shift(1)` on the TEAM's own date-ordered game sequence. The set for game t
    is complete before t starts, which is the whole reason it is allowed.
    """
    per_game = (streams[["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "PLAYER_ID"]]
                .drop_duplicates(["TEAM_ABBREVIATION", "GAME_ID", "PLAYER_ID"]))
    per_game = (per_game.groupby(["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE"],
                                 as_index=False)
                .agg(squad=("PLAYER_ID", lambda s: frozenset(s))))
    per_game = per_game.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    per_game["prev_squad"] = per_game.groupby("TEAM_ABBREVIATION", sort=False)["squad"].shift(1)
    return per_game


def _asof_attach(queries: pd.DataFrame, hist: pd.DataFrame,
                 cols: list[str]) -> pd.DataFrame:
    """Attach each queried player's most recent strictly-earlier values.

    `allow_exact_matches=False` is the enforcement, not a convention: a player
    who played in the previous game still cannot see that game's own box score.
    """
    left = queries.copy()
    left["GAME_DATE"] = pd.to_datetime(left["GAME_DATE"])
    right = hist.copy()
    right["GAME_DATE"] = pd.to_datetime(right["GAME_DATE"])
    left["PLAYER_ID"] = left["PLAYER_ID"].astype("int64")
    right["PLAYER_ID"] = right["PLAYER_ID"].astype("int64")
    return pd.merge_asof(
        left.sort_values("GAME_DATE"),
        right.sort_values("GAME_DATE"),
        left_on="GAME_DATE", right_on="GAME_DATE", by="PLAYER_ID",
        direction="backward", allow_exact_matches=False,
    )


def _explode_prev_squad(per_game: pd.DataFrame) -> pd.DataFrame:
    q = per_game.loc[per_game["prev_squad"].notna()].copy()
    e = q[["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "prev_squad"]].explode(
        "prev_squad").rename(columns={"prev_squad": "PLAYER_ID"})
    return e.dropna(subset=["PLAYER_ID"])


def squad_quality_features(games: pd.DataFrame, halflife: float = 15.0) -> pd.DataFrame:
    """Per-side squad-quality contrasts over the PREVIOUS game's squad.

    Columns, for `side` in (home, away):

    ``{side}_squad_n``            players in the previous squad
    ``{side}_sq_exp_minutes``      sum of their expected minutes (~240, so it is
                                   schedule noise; reported for the ratio below)
    ``{side}_sq_min_per_player``  exp_minutes / squad_n. Depth of the rotation.
    ``{side}_sq_{q}``             minutes-weighted MEAN of q over the squad.
    ``{side}_sq_{q}_top3``        mean of the three highest-q players in the
                                   squad by expected minutes. "Is the good
                                   player here", separated from "is the bench
                                   here", which the weighted mean confounds.
    ``{side}_sq_{q}_diff``        ``away_`` minus ``home_``. Mirrored, matching
                                   the convention availability_features uses.
    """
    streams = _player_streams(games, halflife)
    per_game = _prev_squad_roster(streams)
    exploded = _explode_prev_squad(per_game)
    val_cols = ["exp_min", *[f"q_{c}" for c in QUALITY_COLS]]
    attached = _asof_attach(
        exploded[["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "PLAYER_ID"]],
        streams[["PLAYER_ID", "GAME_DATE", *val_cols]],
        val_cols,
    )

    g = attached.groupby(["GAME_ID", "TEAM_ABBREVIATION"], as_index=False).agg(
        sq_exp_minutes=("exp_min", "sum"),
        sq_n=("PLAYER_ID", "nunique"),
    )
    for c in QUALITY_COLS:
        attached[f"w_{c}"] = attached["exp_min"] * attached[f"q_{c}"]
    sums = (attached.groupby(["GAME_ID", "TEAM_ABBREVIATION"], as_index=False)
            [[f"w_{c}" for c in QUALITY_COLS]].sum())
    g = g.merge(sums, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")
    for c in QUALITY_COLS:
        g[f"sq_{c}"] = g[f"w_{c}"] / g["sq_exp_minutes"].replace(0, np.nan)

    # Top-3 by expected minutes, so "the best three available" is not diluted by
    # a deep bench. Ties break on q descending, which is deterministic.
    top = (attached.dropna(subset=["exp_min"])
           .sort_values(["GAME_ID", "TEAM_ABBREVIATION", "exp_min", "PLAYER_ID"],
                        ascending=[True, True, False, True])
           .groupby(["GAME_ID", "TEAM_ABBREVIATION"]).head(3))
    for c in QUALITY_COLS:
        t3 = (top.groupby(["GAME_ID", "TEAM_ABBREVIATION"])[f"q_{c}"].mean()
              .rename(f"sq_{c}_top3").reset_index())
        g = g.merge(t3, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")

    g["sq_min_per_player"] = g["sq_exp_minutes"] / g["sq_n"].replace(0, np.nan)
    g = g.set_index(["GAME_ID", "TEAM_ABBREVIATION"])
    if not g.index.is_unique:
        raise AssertionError(
            f"squad_quality produced duplicate (game, team) keys: "
            f"{g.index[g.index.duplicated()].tolist()[:3]}")

    gi = games[["GAME_ID", "home", "away"]].reset_index(drop=True)
    out = pd.DataFrame(index=gi.index)
    carry = ["sq_n", "sq_exp_minutes", "sq_min_per_player",
             *[f"sq_{c}" for c in QUALITY_COLS],
             *[f"sq_{c}_top3" for c in QUALITY_COLS]]
    for side in ("home", "away"):
        idx = pd.MultiIndex.from_arrays([gi["GAME_ID"], gi[side]])
        for c in carry:
            out[f"{side}_{c}"] = g[c].reindex(idx).to_numpy()

    for c in carry:
        if c == "sq_n":          # a count, not a quality: no contrast
            continue
        out[f"home_{c}_diff"] = out[f"home_{c}"] - out[f"away_{c}"]
        out[f"away_{c}_diff"] = -out[f"home_{c}_diff"]

    if len(out) != len(games):
        raise AssertionError(f"{len(out)} rows for {len(games)} games")
    return out.reset_index(drop=True)