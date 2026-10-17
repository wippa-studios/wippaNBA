"""Round 3, block Z: L1 and elastic net on the promoted frame. The one variance
lever nobody has pulled, and the diagnosis says to pull it.

WHY THIS AND NOT MORE C
------------------------
The C sweep is flat at both widths (43 cols: selected-C scores 0.618975 vs the
bar 0.618307; 67 cols: C=0.3 gives 0.613125 vs C=1.0's 0.613137), and peers
independently closed model-family optimisation the same way. But an L2 penalty
shrinks every coefficient toward zero by the SAME amount. It cannot say "this
column is noise, drop it" or "these eight are the same column, keep one".

Peer `fdd87721` closed with the diagnosis that makes this worth doing anyway:

    "Fold log loss improves MONOTONICALLY with training size: f1 0.62552 ...
     f5 0.59899. This is a LEARNING CURVE, not era drift - the model is
     data-limited, which argues for variance reduction."

Data-limited plus flat under uniform shrinkage plus 47 effective d.o.f. out of 48
columns is the textbook setup where SELECTION beats SHRINKAGE: with correlated
predictors and a finite n, dropping a group of aliases costs less bias than
keeping them costs variance.

ELASTIC NET, NOT L1 ALONE
-------------------------
L1 picks one member of a correlated group and can be unstable across folds,
which would show up as fold instability rather than as an honest result. The
net interpolates, and `l1_ratio` is swept rather than assumed.

EVERY SELECTION IS NESTED
-------------------------
`penalty='l1'` was deprecated in sklearn 1.9 in favour of `l1_ratio`, which is
what `round3_common.pipeline_for` uses: `l1_ratio=0` is L2, `1` is L1, and
anything between is elastic net. Each arm refits median-impute and the scaler
inside every fold exactly as the bar does.

ARMS, all of the grid printed
-----------------------------
  Z1_l1        l1_ratio = 1.0  over C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
  Z2_elastic   l1_ratio in {0.1, 0.25, 0.5, 0.75} over the same C grid

Acceptance is the promoted model's own marginal bar: |z| > 1.125 over 0.612523
AND 5/5 folds, same as block Y used.
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

from scripts.round3_common import load_cached, oof_fixed, compare, log3  # noqa: E402
from scripts.round3_run_U import per_fold_delta                          # noqa: E402
from src.DataProviders import nba_data as nd                              # noqa: E402
from src.DataProviders.feat_team_elo import team_elo_features            # noqa: E402

PROMOTED_LL = 0.612523
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]


def main() -> None:
    t0 = time.time()
    games, e14 = load_cached()
    y = games["y"].to_numpy()
    av = nd.availability_features(games, 15.0)
    elo = team_elo_features(games, k=10.0)[["elo_diff"]]
    X = pd.concat([e14, av, elo], axis=1)
    X = X.loc[:, X.nunique(dropna=False) > 1].reset_index(drop=True)
    Xv = X.to_numpy(dtype=float)
    print(f"promoted frame {Xv.shape}", flush=True)

    p_prom = oof_fixed(Xv, y, games, C=1.0)
    prom_ll = compare(y, p_prom, p_prom)["log_loss"]
    assert abs(prom_ll - PROMOTED_LL) < 1e-5, f"promoted frame moved: {prom_ll}"

    grid = [(1.0, c, f"Z1_l1_C{c:g}") for c in C_GRID]
    grid += [(r, c, f"Z2_elastic_r{r}_C{c:g}")
             for r in (0.1, 0.25, 0.5, 0.75) for c in C_GRID]

    best = None
    for ratio, C, run_id in grid:
        t = time.time()
        p = oof_fixed(Xv, y, games, C=C, penalty="l2", l1_ratio=ratio)
        # saga can diverge and return non-finite probabilities. That is a real
        # outcome of the arm, not an error to crash on: record it and move on,
        # otherwise one bad C hides the other 29.
        n_bad = int((~np.isfinite(p)).sum() - np.isnan(p[:len(games) - 6600]).sum())
        if not np.isfinite(p[np.isfinite(p)]).all() or n_bad > 0:
            rec = {"run_id": run_id, "block": "Z", "kind": "PRE_REGISTERED",
                   "l1_ratio": ratio, "C": C, "diverged": True,
                   "n_non_finite_predictions": n_bad,
                   "seconds": round(time.time() - t, 1)}
            log3(rec)
            print(f"{run_id:22s} DIVERGED, {n_bad} non-finite predictions "
                  f"({rec['seconds']:.0f}s)", flush=True)
            continue
        vs = compare(y, p, p_prom)
        rec = {"run_id": run_id, "block": "Z", "kind": "PRE_REGISTERED",
               "l1_ratio": ratio, "C": C, "vs_promoted_48col": vs,
               "per_fold": per_fold_delta(y, p, p_prom, games),
               "seconds": round(time.time() - t, 1)}
        log3(rec)
        print(f"{run_id:22s} ll={vs['log_loss']:.6f} z={vs['abs_z']:.2f} "
              f"folds={rec['per_fold']['folds_cand_better']}/5 "
              f"({rec['seconds']:.0f}s)", flush=True)
        if best is None or vs["log_loss"] < best["vs_promoted_48col"]["log_loss"]:
            best = rec

    if best is None:
        raise SystemExit("every arm diverged; no result to report")
    rec = best
    print(f"\nBEST {rec['run_id']} ll={rec['vs_promoted_48col']['log_loss']:.6f} "
          f"|z|={rec['vs_promoted_48col']['abs_z']:.2f} "
          f"folds={rec['per_fold']['folds_cand_better']}/5", flush=True)
    print("ACCEPTANCE: |z| > 1.125 over 0.612523 AND 5/5 folds.")
    ok = rec["vs_promoted_48col"]["abs_z"] > 1.125 and \
        rec["per_fold"]["folds_cand_better"] == 5
    print(f"  {'PASS' if ok else 'FAIL'}")
    log3({"run_id": "Z_summary", "block": "Z", "kind": "SUMMARY",
          "best_run_id": rec["run_id"], "best_log_loss": rec["vs_promoted_48col"]["log_loss"],
          "best_abs_z": rec["vs_promoted_48col"]["abs_z"],
          "folds_better": rec["per_fold"]["folds_cand_better"],
          "verdict": "PASS" if ok else "FAIL",
          "n_arms": len(grid)})
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()