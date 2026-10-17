"""Round 3, block A follow-up: does concentration add anything OVER availability?

WHY THIS FILE EXISTS
--------------------
The eight pre-registered arms in round3_avail.py all beat the bar and A4 came out
on top at +0.004911. But that is the MAXIMUM of eight arms scored on the same
6,600 games, so the ranking is selection-biased upward. The claim worth making is
not "A4 is best" - it is "does the concentration column earn its place next to the
column we already had".

That is a different comparison and it needs its own paired test:

    A4 vs A1, not A4 vs baseline.

A4 - A1 = the value of `av_prev_top5_share_diff` given `av_prev_sum_diff` is
already present. If that interval straddles zero, the honest answer is "no
meaningful difference, keep the single column" - and the concentration idea joins
the round-2 dead-end list even though its arm looked like the winner.

Every arm is also re-scored against the bar here so the numbers are comparable to
the published table, and each is reported with its per-fold deltas, because the
primary model is deterministic and per-fold spread is the only real uncertainty.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from src.DataProviders import nba_data as nd          # noqa: E402
import round2_common as rc                            # noqa: E402
import round3_avail as ra                             # noqa: E402

JSONL = ROOT / "results" / "round3_avail_followup.jsonl"
BOOT_SEED = 42
N_BOOT = 4000

S = "av_prev_sum_diff"
C = "av_prev_top5_share_diff"

# (run_id, columns beyond the 43-col base)
ARMS = [
    ("A1_sum_only", [S]),
    ("A4_sum_plus_concentration", [S, C]),
    ("A2_sum_plus_depth", [S, "av_prev_n_diff"]),
    ("A3_sum_plus_minpp", [S, "av_prev_min_per_player_diff"]),
    ("A5_sum_plus_prodpp", [S, "av_prev_prod_per_player_diff"]),
]


def _finite(y, *ps):
    m = np.isfinite(y.astype(float))
    for p in ps:
        m &= np.isfinite(p)
    return m


def paired(y, a, b, folds, name_a, name_b) -> dict:
    """log_loss(a) - log_loss(b), positive => b is better."""
    m = _finite(y, a, b)
    d = rc.delta_vs(y[m], a[m], b[m], seed=BOOT_SEED, n_boot=N_BOOT)
    pf = []
    for _tr, te in folds:
        k = np.isfinite(a[te]) & np.isfinite(b[te])
        pf.append(round(rc.score(y[te][k], a[te][k])["log_loss"]
                        - rc.score(y[te][k], b[te][k])["log_loss"], 6))
    gains = -np.array(pf)
    return {
        "comparison": f"{name_b} minus {name_a} (positive = {name_b} better)",
        "improvement_nats": round(d["improvement_nats"], 6),
        "ci_lo": round(d["ci_lo"], 6), "ci_hi": round(d["ci_hi"], 6),
        "abs_z": round(d["abs_z"], 3),
        "per_fold_delta": pf,
        "fold_gain_mean": round(float(gains.mean()), 6),
        "fold_gain_sd": round(float(gains.std(ddof=1)), 6),
        "folds_where_second_wins": int((gains > 0).sum()),
        "clears_fold_spread": bool(abs(gains.mean()) > gains.std(ddof=1)),
        "ci_excludes_zero": bool(d["ci_lo"] > 0 or d["ci_hi"] < 0),
    }


def main() -> None:
    t0 = time.time()
    g = nd.build_games()
    y = g["y"].to_numpy(int)
    base = rc.build_e14_frame(g)
    blk = ra.availability_block(g)
    folds = nd.walk_forward_split(g, n_splits=5, initial_frac=0.5)
    print(f"[{time.time()-t0:.0f}s] base={base.shape} block={blk.shape}", flush=True)

    p_base = rc.walk_forward_oof(base.to_numpy(float), y, g)
    preds = {"BAR_baseline": p_base}
    for run_id, cols in ARMS:
        X = pd.concat([base, blk[cols].reset_index(drop=True)], axis=1)
        X = X.loc[:, X.nunique(dropna=False) > 1]
        preds[run_id] = rc.walk_forward_oof(X.to_numpy(float), y, g)
        print(f"[{time.time()-t0:5.0f}s] fitted {run_id} ({X.shape[1]} cols)",
              flush=True)

    def ll(key):
        m = _finite(y, preds[key])
        return round(rc.score(y[m], preds[key][m])["log_loss"], 6)

    print("\n== each arm vs the 0.618307 bar ==")
    rows = []
    for run_id, _c in ARMS:
        r = paired(y, preds[run_id], p_base, folds, run_id, "BAR")
        rows.append({"run_id": run_id, "log_loss": ll(run_id), **r})
        print(f"  {run_id:26s} ll={ll(run_id):.6f} imp={r['improvement_nats']:+.6f} "
              f"z={r['abs_z']:.2f} folds={r['folds_where_second_wins']}/5 "
              f"foldsd={r['fold_gain_sd']:.6f} clears={r['clears_fold_spread']}",
              flush=True)

    print("\n== THE QUESTION: does each extra column beat sum_diff ALONE? ==")
    print("   (this is the comparison the pre-registered ranking could not make)")
    head = [
        ("A4_sum_plus_concentration", "A1_sum_only"),
        ("A2_sum_plus_depth", "A1_sum_only"),
        ("A3_sum_plus_minpp", "A1_sum_only"),
        ("A5_sum_plus_prodpp", "A1_sum_only"),
    ]
    verdicts = []
    for two, one in head:
        r = paired(y, preds[two], preds[one], folds, two, one)
        claim = ("ADDS something real" if (r["ci_excludes_zero"] and r["clears_fold_spread"])
                 else "no meaningful difference over the single column")
        verdicts.append({"added_column_arm": two, "reference": one, **r,
                         "verdict": claim})
        print(f"  {two:26s} vs {one:12s} imp={r['improvement_nats']:+.6f} "
              f"ci=[{r['ci_lo']:+.6f},{r['ci_hi']:+.6f}] z={r['abs_z']:.2f} "
              f"folds={r['folds_where_second_wins']}/5 "
              f"foldsd={r['fold_gain_sd']:.6f} -> {claim}", flush=True)

    for r in rows:
        with JSONL.open("a") as fh:
            fh.write(json.dumps({"run_id": r["run_id"], "block": "A_vs_bar", **r}) + "\n")
    for v in verdicts:
        with JSONL.open("a") as fh:
            fh.write(json.dumps({"run_id": v["added_column_arm"],
                                 "block": "A_vs_single_column", **v}) + "\n")

    print(f"\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()