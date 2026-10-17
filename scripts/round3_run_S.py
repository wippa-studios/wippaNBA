"""Round 3, block S: squad-quality. Pre-registered arms, then run. Nothing is
scored that is not in the list below.

WHY (the one-sentence hypothesis)
--------------------------------
The 43-column frame is entirely team-level trailing means, so it cannot say who
is producing a team's recent form. `availability_features` proved that escaping
that limit pays (+0.0039 nats) by aggregating over the squad from the team's
PREVIOUS game. Squad quality asks the same question about worth rather than
workload: the minutes-weighted mean, and separately the top-3 mean, of each
player's own causal-EWM impact/shooting/usage/playmaking.

The star-concentration split (top3 vs weighted mean) is the part that is not a
rearrangement of a team average. A minutes-weighted mean cannot separate "the
good player is here" from "the bench is here"; a top-3 mean can. That is a real
structural difference, not a correlation of two versions of margin.

NON-FINDINGS THIS MUST NOT REPEAT
---------------------------------
  player_elo over ACTUAL participants leaked at |z| 5.18 and was worth +0.00065
  once the membership was repaired. The same membership discipline is applied
  here (GATE 3 in round3_verify.py) and the roster is the previous game's squad.
  The 52 rank columns, gradient boosting, PCA(20) and recency weighting are all
  existing nulls and are not re-run.

COVERAGE: derived entirely from player_gamelogs/advanced_*.csv joined to games,
with the previous-game squad. Same 99.9%-of-team-games coverage as availability;
NaN only for a team's first game on record.

ACCEPTANCE: |z| > 2.535 (Bonferroni over this block's 4 arms at family-wise
0.05), measured against the 43-column frozen bar AND against the
E14+availability model, because a gain that is smaller than the availability
gain already in hand is not worth the columns it costs.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from scripts.round3_common import (                      # noqa: E402
    load_cached, oof_fixed, compare, log3, delta_vs, scored_mask)
from src.DataProviders import nba_data as nd              # noqa: E402

E14_BAR = 0.618307
AV_LOG_LOSS = 0.614372          # R2_availability_on_e14, reproduced below
REDUNDANT = {"NET_RATING"}      # GATE2: |r| 0.92 against ewm_net_rating


def prune(df: pd.DataFrame, redundant: set[str]) -> pd.DataFrame:
    keep = [c for c in df.columns
            if not any(c.endswith("_" + r) or f"_{r}_" in c for r in redundant)]
    out = df[keep]
    return out.loc[:, out.nunique(dropna=False) > 1].reset_index(drop=True)


def build_frame(e14: pd.DataFrame, av: pd.DataFrame, sq: pd.DataFrame,
                cols: list[str]) -> pd.DataFrame:
    parts = [e14, av, sq[[c for c in sq.columns if c in cols]]]
    X = pd.concat([p.reset_index(drop=True) for p in parts], axis=1)
    dup = [c for c in X.columns if list(X.columns).count(c) > 1]
    assert not dup, f"duplicate columns: {dup}"
    return X.loc[:, X.nunique(dropna=False) > 1].reset_index(drop=True)


def main() -> None:
    t0 = time.time()
    games, e14 = load_cached()
    y = games["y"].to_numpy()

    av = nd.availability_features(games, 15.0)
    sq = pd.read_pickle("/tmp/opencode/nba/sq15.pkl")
    sq_lean = prune(sq, REDUNDANT)
    contrasts = [c for c in sq_lean.columns if c.endswith("_diff")]
    star = [c for c in sq_lean.columns
            if ("_top3" in c) or c.endswith("_sq_n") or c.endswith("_sq_min_per_player")]
    star_lean = prune(sq[star], REDUNDANT)

    print(f"sq_lean {sq_lean.shape[1]} cols | contrasts {len(contrasts)} | "
          f"star {len(star)} | avail {av.shape[1]}", flush=True)

    # ---- references, both reproduced here -------------------------------
    # nd.evaluate does NOT mask NaN, and walk-forward OOF is NaN before the first
    # test block. Score only the scored rows, or the metric silently chokes.
    def ll(p):
        return compare(y, p, p)["log_loss"]

    p_bar = oof_fixed(e14.to_numpy(dtype=float), y, games, C=1.0)
    assert abs(ll(p_bar) - E14_BAR) < 1e-6, \
        f"bar did not reproduce: {ll(p_bar)}"
    X_av = build_frame(e14, av, sq, [])
    p_av = oof_fixed(X_av.to_numpy(dtype=float), y, games, C=1.0)
    av_ll = ll(p_av)
    print(f"BAR {ll(p_bar):.6f} | AVAIL {av_ll:.6f} "
          f"(recorded {AV_LOG_LOSS})", flush=True)
    log3({"run_id": "S0_reference_e14_availability", "block": "S", "kind": "REFERENCE",
          "n_features": X_av.shape[1], "log_loss": av_ll,
          "note": "E14 + availability_features(halflife=15); reproduces R2_availability_on_e14"})

    arms = [
        ("S1_sq_quality_only",      build_frame(e14, sq_lean.iloc[:, :0], sq, list(sq_lean.columns))),
        ("S2_sq_on_availability",  build_frame(e14, av, sq, list(sq_lean.columns))),
        ("S3_sq_contrasts_only",   build_frame(e14, av, sq, contrasts)),
        ("S4_star_concentration",  build_frame(e14, av, sq, list(star_lean.columns))),
    ]

    results = []
    for run_id, X in arms:
        p = oof_fixed(X.to_numpy(dtype=float), y, games, C=1.0)
        vs_bar = compare(y, p, p_bar)
        vs_av = compare(y, p, p_av)
        rec = {"run_id": run_id, "block": "S", "kind": "PRE_REGISTERED",
               "n_features": int(X.shape[1]),
               "vs_frozen_bar_e14": vs_bar, "vs_availability_model": vs_av}
        log3(rec)
        results.append(rec)
        print(f"{run_id:24s} cols={X.shape[1]:3d} ll={vs_bar['log_loss']:.6f} "
              f"z_bar={vs_bar['abs_z']:.2f} z_av={vs_av['abs_z']:.2f}", flush=True)

    print(f"\ntotal {time.time() - t0:.1f}s")
    print("\nVERDICT (|z| > 2.535 against the availability model is the bar that "
          "matters -- anything weaker than the gain already in hand is not worth "
          "the columns):")
    keep = [r for r in results if r["vs_availability_model"]["abs_z"] > 2.535]
    print("  arms clearing the bar:", [r["run_id"] for r in keep] or "NONE")


if __name__ == "__main__":
    main()