"""Round 3, block U: combine the survivors, and check the gain is not one fold.

Where round 3 stands before this script runs
---------------------------------------------
  frozen bar            E14, 43 cols                0.618307
  availability          + 4 cols                    0.614544
  squad-quality         + 20 contrast cols          0.613137
  elo                   + 3 contrast cols (on av)   0.612492

Squad quality and Elo are complementary: each is worth about +0.002 on top of
availability alone, neither is worth its columns by itself, and together they are
worth about +0.004. That pattern is the signature of two estimators carrying
partly independent information about the same games - the model side rather than
a coincidence, but it needs the two checks below before it can be called a
result.

THE TWO CHECKS, BOTH PRE-REGISTERED
-----------------------------------
  1. FOLD STABILITY. The walk-forward is 5 disjoint test blocks. A pooled
     bootstrap CI can be narrow because 6600 rows overlap heavily between
     neighbouring games, not because the effect is real. Report the per-fold
     log-loss DELTA for the candidate minus the availability model. If the sign
     flips in any fold, the pooled number is a fold-average artefact and is
     reported as such. A gain smaller than fold-to-fold spread is not a gain.

  2. SEED STABILITY. LogisticRegression(lbfgs) on this problem is effectively
     deterministic, so a seed sweep would measure floating-point noise and
     would be theatre. Instead the row-ORDER perturbation is used: the walk
     forward is re-run on the games frame sorted by a seeded random key that
     preserves nothing except membership, and the same delta is recomputed. This
     catches an order-dependent artefact, which is a real risk for a frame whose
     features are EWMs over row order within team.

ACCEPTANCE
----------
|z| > 2.535 against the availability model (Bonferroni over this block's 3
arms), AND the same sign in all 5 folds. Failing either, it is logged as "no
meaningful difference" and not promoted.
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

from scripts.round3_common import load_cached, oof_fixed, compare, log3  # noqa: E402
from src.DataProviders import nba_data as nd                              # noqa: E402
from src.DataProviders.feat_team_elo import team_elo_features            # noqa: E402
from scripts.round3_run_S import prune, REDUNDANT, build_frame            # noqa: E402


def per_fold_delta(y: np.ndarray, p_c: np.ndarray, p_b: np.ndarray,
                   games: pd.DataFrame) -> dict:
    """Log-loss delta per walk-forward test block. NEGATIVE = candidate better."""
    out = {}
    for k, (tr, te) in enumerate(nd.walk_forward_split(games)):
        m = np.isfinite(p_c[te]) & np.isfinite(p_b[te])
        if not m.any():
            continue
        yy = y[te][m]
        a = np.clip(p_c[te][m], 1e-6, 1 - 1e-6)
        b = np.clip(p_b[te][m], 1e-6, 1 - 1e-6)
        ll = lambda p: -(yy * np.log(p) + (1 - yy) * np.log(1 - p)).mean()
        out[f"fold_{k}"] = {"n": int(m.sum()),
                            "delta_cand_minus_avail": round(float(ll(a) - ll(b)), 6),
                            "cand_better": bool(ll(a) < ll(b))}
    out["folds_cand_better"] = sum(1 for v in out.values()
                                   if isinstance(v, dict) and v["cand_better"])
    out["n_folds"] = sum(1 for v in out.values() if isinstance(v, dict))
    vals = [v["delta_cand_minus_avail"] for v in out.values() if isinstance(v, dict)]
    out["fold_delta_sd"] = round(float(np.std(vals, ddof=1)), 6) if len(vals) > 1 else None
    out["pooled_delta"] = round(float(np.mean(vals)), 6)
    return out


def main() -> None:
    t0 = time.time()
    games, e14 = load_cached()
    y = games["y"].to_numpy()
    av = nd.availability_features(games, 15.0)
    sq = pd.read_pickle("/tmp/opencode/nba/sq15.pkl")
    sq_lean = prune(sq, REDUNDANT)
    sq_contrast = [c for c in sq_lean.columns if c.endswith("_diff")]
    elo = team_elo_features(games, k=10.0)

    X_av = build_frame(e14, av, sq, [])
    p_av = oof_fixed(X_av.to_numpy(dtype=float), y, games, C=1.0)
    av_ll = compare(y, p_av, p_av)["log_loss"]
    print(f"REFERENCE availability {av_ll:.6f}", flush=True)

    # Build each candidate explicitly so the column provenance is obvious.
    def frame(extra: list[pd.DataFrame]) -> pd.DataFrame:
        X = pd.concat([e14.reset_index(drop=True), av.reset_index(drop=True)]
                      + [d.reset_index(drop=True) for d in extra], axis=1)
        dup = [c for c in X.columns if list(X.columns).count(c) > 1]
        assert not dup, f"duplicate columns: {dup}"
        return X.loc[:, X.nunique(dropna=False) > 1].reset_index(drop=True)

    cands = {
        "U1_av_plus_elo":            frame([elo]),
        "U2_av_plus_elo_plus_sq":    frame([sq[sq_contrast], elo]),
        "U3_av_plus_elo3_plus_sq10": frame([sq[sq_contrast[:10]], elo[elo_diff_only()]]),
    }
    results = {}
    for run_id, X in cands.items():
        p = oof_fixed(X.to_numpy(dtype=float), y, games, C=1.0)
        vs_av = compare(y, p, p_av)
        pf = per_fold_delta(y, p, p_av, games)
        rec = {"run_id": run_id, "block": "U", "kind": "PRE_REGISTERED",
               "n_features": int(X.shape[1]), "vs_availability_model": vs_av,
               "per_fold": pf}
        log3(rec)
        results[run_id] = rec
        print(f"{run_id:26s} cols={X.shape[1]:3d} ll={vs_av['log_loss']:.6f} "
              f"z={vs_av['abs_z']:.2f} folds_better={pf['folds_cand_better']}/{pf['n_folds']} "
              f"fold_sd={pf['fold_delta_sd']}", flush=True)

    # ---- check 2: row-order perturbation -------------------------------
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(games))
    g2 = games.iloc[perm].reset_index(drop=True)
    e14b = e14.iloc[perm].reset_index(drop=True)
    av2 = nd.availability_features(g2, 15.0)
    elo2 = team_elo_features(g2, k=10.0)
    Xa2 = pd.concat([e14b, av2], axis=1)
    Xc2 = pd.concat([e14b, av2, elo2], axis=1)
    p_av2 = oof_fixed(Xa2.to_numpy(dtype=float), g2["y"].to_numpy(), g2, C=1.0)
    p_c2 = oof_fixed(Xc2.to_numpy(dtype=float), g2["y"].to_numpy(), g2, C=1.0)
    ord_check = compare(g2["y"].to_numpy(), p_c2, p_av2)
    log3({"run_id": "U4_row_order_perturbation", "block": "U",
          "kind": "PRE_REGISTERED_ROBUSTNESS",
          "reading": "same arms, games frame permuted by seeded RNG. An "
                     "order-dependent artefact shows up here as a moved delta.",
          "vs_availability_model": ord_check,
          "columns": list(Xc2.columns[-6:])})
    print(f"ORDER-PERTURBED  z={ord_check['abs_z']:.2f} "
          f"ll={ord_check['log_loss']:.6f} delta={ord_check['improvement_nats']:+.6f}",
          flush=True)

    print(f"\ntotal {time.time() - t0:.1f}s")
    print("\nACCEPTANCE needs |z| > 2.535 vs availability AND 5/5 folds:")
    for k, v in results.items():
        ok = v["vs_availability_model"]["abs_z"] > 2.535 and \
            v["per_fold"]["folds_cand_better"] == v["per_fold"]["n_folds"]
        print(f"  {k:26s} {'PASS' if ok else 'FAIL'}")


def cols_of(cols: list[str], sq: pd.DataFrame) -> list[str]:
    return cols


def elo_diff_only() -> list[str]:
    return ["elo_diff", "elo_diff_adj"]


if __name__ == "__main__":
    main()