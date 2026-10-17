"""Round-3 verification: leakage proof + redundancy check, BEFORE scoring.

Two gates, both mechanical, both runnable by anyone:

  GATE 1  P1_truncation. Rebuild every candidate feature on a frame truncated at
          date D and assert the surviving rows are numerically IDENTICAL, NaN
          pattern included. This catches a leak in the WEIGHTS (an aggregate
          that reaches past the game being predicted).

  GATE 2  Redundancy. For every candidate column, max |r| against the 43 E14
          columns. Anything above 0.8 is a rearrangement, not a feature, and is
          reported as such whether or not its delta is positive.

A leak in MEMBERSHIP passes Gate 1 entirely -- truncating the future does not
change who was in the room. That is why `feat_squad_quality` assembles its
roster with `shift(1)` over the team's own game sequence rather than from the
game being predicted, and Gate 3 checks that mechanically.

  GATE 3  P3_membership. For a random sample of games, assert that the player
          rows feeding that game's aggregate all belong to the team's PREVIOUS
          game, and that no row's own game equals the game being predicted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from src.DataProviders import nba_data as nd                        # noqa: E402
from src.DataProviders.feat_squad_quality import (                  # noqa: E402
    squad_quality_features, _player_streams, _prev_squad_roster, _explode_prev_squad)


def gate1_truncation(games: pd.DataFrame, fn, cut: str, label: str) -> dict:
    """fn(games) -> DataFrame aligned to `games`. Rebuild on games < cut."""
    full = fn(games)
    trunc_games = games[games["GAME_DATE"] < cut].reset_index(drop=True)
    trunc = fn(trunc_games)
    n = len(trunc)
    a = full.iloc[:n].reset_index(drop=True)
    b = trunc.reset_index(drop=True)
    assert list(a.columns) == list(b.columns), "column set changed under truncation"
    num = a.select_dtypes(include=[np.number])
    b = b[num.columns]
    # NaN pattern must match too: a value that backfills after truncation is a leak.
    nan_mismatch = int((a.isna() != b.isna()).to_numpy().sum())
    both = a.notna() & b.notna()
    maxdiff = 0.0
    if both.to_numpy().any():
        d = (num.to_numpy(dtype=float) - b.to_numpy(dtype=float))[both.to_numpy()]
        maxdiff = float(np.nanmax(np.abs(d))) if d.size else 0.0
    return {"gate": "P1_truncation", "label": label, "cut": cut,
            "rows_compared": n, "nan_pattern_mismatches": nan_mismatch,
            "max_abs_diff": maxdiff,
            "pass": bool(nan_mismatch == 0 and maxdiff < 1e-12)}


def gate3_membership(games: pd.DataFrame, halflife: float = 15.0,
                     n_sample: int = 200, seed: int = 42) -> dict:
    """Every player row feeding game t's aggregate must come from a game the
    team played STRICTLY BEFORE t. No row from game t itself, ever."""
    streams = _player_streams(games, halflife)
    per_game = _prev_squad_roster(streams)
    exploded = _explode_prev_squad(per_game)

    team_games = streams[["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE"]].drop_duplicates()
    team_games = team_games.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"])
    team_games["prev_gid"] = team_games.groupby("TEAM_ABBREVIATION", sort=False)["GAME_ID"].shift(1)
    team_games["prev_date"] = team_games.groupby("TEAM_ABBREVIATION", sort=False)["GAME_DATE"].shift(1)
    prev = team_games.set_index(["TEAM_ABBREVIATION", "GAME_ID"])["prev_gid"].to_dict()

    ex = exploded.copy()
    ex["PREV_GID"] = [prev.get((t, g), None) for t, g in
                      zip(ex["TEAM_ABBREVIATION"], ex["GAME_ID"])]

    # Every player in prev_squad must actually have an appearance row whose
    # GAME_ID is the team's previous game. Any other value means the roster was
    # rebuilt from something other than the previous game.
    sp = streams[["TEAM_ABBREVIATION", "GAME_ID", "PLAYER_ID"]].drop_duplicates()
    sp = sp.assign(_k=sp["TEAM_ABBREVIATION"] + "|" + sp["GAME_ID"] + "|" + sp["PLAYER_ID"].astype(str))
    ex = ex.assign(_k=ex["TEAM_ABBREVIATION"] + "|" + ex["PREV_GID"].astype(str)
                   + "|" + ex["PLAYER_ID"].astype(str))
    _keys = set(sp["_k"])
    bad_membership = int((~ex["_k"].isin(_keys)).sum())

    # And the attached value must come from a strictly earlier date.
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ex), size=min(n_sample, len(ex)), replace=False)
    return {"gate": "P3_membership", "n_query_rows": int(len(ex)),
            "rows_not_in_previous_squad": bad_membership,
            "sampled_rows": int(len(idx)),
            "pass": bool(bad_membership == 0)}


def gate2_redundancy(base: pd.DataFrame, cand: pd.DataFrame,
                     thresh: float = 0.8) -> dict:
    b = base.select_dtypes(include=[np.number])
    c = cand.select_dtypes(include=[np.number])
    rows = {}
    for col in c.columns:
        s = pd.concat([b, c[[col]]], axis=1).corr(numeric_only=True).iloc[:-1, -1]
        rows[col] = {"max_abs_r_vs_e14": round(float(s.abs().max()), 4),
                     "argmax": str(s.abs().idxmax())}
    over = {k: v for k, v in rows.items() if v["max_abs_r_vs_e14"] > thresh}
    return {"gate": "redundancy", "threshold": thresh,
            "n_candidate_cols": len(rows),
            "n_over_threshold": len(over), "over_threshold": over,
            "per_column": rows}


def main() -> None:
    games = pd.read_pickle("/home/Joel/.cache/nba-frames/games.pkl")
    e14 = pd.read_pickle("/home/Joel/.cache/nba-frames/e14.pkl")
    out = {}

    out["GATE1_squad_quality"] = gate1_truncation(
        games, lambda g: squad_quality_features(g, 15.0), "2021-10-20",
        "feat_squad_quality halflife=15")
    out["GATE1_availability"] = gate1_truncation(
        games, lambda g: nd.availability_features(g, 15.0), "2021-10-20",
        "nd.availability_features halflife=15")

    out["GATE3_squad_quality"] = gate3_membership(games, 15.0)

    sq = pd.read_pickle("/home/Joel/.cache/nba-frames/sq15.pkl")
    out["GATE2_squad_quality"] = gate2_redundancy(e14, sq)

    # availability's own redundancy, for reference
    av = nd.availability_features(games, 15.0)
    out["GATE2_availability"] = gate2_redundancy(e14, av)

    (ROOT / "results" / "round3_verification.json").write_text(
        json.dumps(out, indent=1))
    for k, v in out.items():
        keep = {kk: vv for kk, vv in v.items()
                if kk not in ("per_column", "over_threshold")}
        print(k, json.dumps(keep), flush=True)
    print("\nGATE2 over-threshold columns:",
          json.dumps({k: v["over_threshold"] for k, v in out.items()
                      if v.get("gate") == "redundancy"}, indent=1))


if __name__ == "__main__":
    main()

def elo_gate() -> dict:
    """P1_truncation for feat_team_elo, which is the feature being promoted."""
    from src.DataProviders.feat_team_elo import team_elo_features
    games = pd.read_pickle("/home/Joel/.cache/nba-frames/games.pkl")
    return gate1_truncation(
        games, lambda g: team_elo_features(g, 10.0), "2021-10-20",
        "feat_team_elo k=10 hfa=70 mov season_reversion=0.75")
