"""Issue #42: do the three strongest models disagree, and does a blend help?

Pre-registered in ``results/experiments.jsonl`` as ``I42-prereg-ensemble``
before this script ran.

MEMBERS
-------
* ``e14`` -- the 43-column logistic frame, the frozen 0.61831 bar.
* ``margin`` -- ``MarginTotalModel`` win probability, Phi(m_hat / sigma).
* ``elo`` -- logistic regression on the two pre-game Elo ratings plus a home
  flag (the already-reported 0.6234 model).

WEIGHTS
-------
The four primary arms are fixed before the run: equal weights, and the three
equal two-model averages.  No weight is fitted on a scored row.  The full 0.1
simplex (66 configurations) is printed as a **descriptive** diagnostic; its
maximum is not selected and is not a result.

Run:  python scripts/issue42_ensemble.py
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

from DataProviders import nba_data as nd          # noqa: E402
from Models import baselines as bl                # noqa: E402
from Models import ensemble as en                 # noqa: E402
from Models import margin_total as mt             # noqa: E402
from Models import measurement as ms              # noqa: E402

JSONL = ROOT / "results" / "experiments.jsonl"
OUT_JSON = ROOT / "results" / "issue42_ensemble.json"
OUT_MD = ROOT / "results" / "issue42_ensemble.md"
N_BOOT = 2000
SEED = 42
NAMES = ("e14", "margin", "elo")
FIXED_ARMS = {
    "equal_weights": {"e14": 1 / 3, "margin": 1 / 3, "elo": 1 / 3},
    "logreg_elo": {"e14": 0.5, "margin": 0.0, "elo": 0.5},
    "logreg_margin": {"e14": 0.5, "margin": 0.5, "elo": 0.0},
    "margin_elo": {"e14": 0.0, "margin": 0.5, "elo": 0.5},
}


def metrics(y, p) -> dict:
    m = nd.evaluate(np.asarray(y, dtype=int), np.asarray(p, dtype=float))
    return {k: (int(v) if k == "n" else float(v))
            for k, v in m.items() if k in ("n", "log_loss", "brier",
                                           "accuracy", "auc")}


def margin_oof(games: pd.DataFrame, data: mt.Dataset) -> np.ndarray:
    y_margin = games["margin"].to_numpy(dtype=float)
    y_total = games["total"].to_numpy(dtype=float)
    p = np.full(len(games), np.nan)
    for tr, te in nd.walk_forward_split(games):
        model = mt.MarginTotalModel(sigma_method="oof").fit(
            data.X_margin.iloc[tr], y_margin[tr],
            data.X_total.iloc[tr], y_total[tr])
        p[te] = model.predict_win_prob(data.X_margin.iloc[te])
    return p


def elo_oof(frame: pd.DataFrame) -> tuple[np.ndarray, float]:
    splits = nd.walk_forward_split(frame)
    k_table = bl.tune_elo_k(frame, splits)
    tuned_k = float(k_table.iloc[0]["k"])
    fit = bl.elo_rating_pass(frame, k=tuned_k, same_day="batched")
    p = np.full(len(frame), np.nan)
    for tr, te in splits:
        p[te] = bl.loglinear_elo_predictions(frame, tr, te, fit)
    return p, tuned_k


def main() -> int:
    t0 = time.time()
    games = nd.build_games()
    n = len(games)
    y = games["y"].to_numpy(dtype=int)

    _, p_e14, mask_e14 = ms.oof_e14(games)
    p_e14 = np.asarray(p_e14, dtype=float)
    if not np.isfinite(p_e14[mask_e14]).all():
        raise AssertionError("E14 OOF has missing scored rows")

    data = mt.build_dataset(games)
    p_margin = margin_oof(games, data)

    frame = bl.load_baseline_frame()
    if len(frame) != n or not (frame["GAME_ID"].astype(str).to_numpy()
                               == games["GAME_ID"].astype(str).to_numpy()).all():
        raise AssertionError("baseline frame is not aligned to the shared games")
    p_elo, tuned_k = elo_oof(frame)

    _, p_can, _ = ms.canonical_oof(games)
    p_can = np.asarray(p_can, dtype=float)

    common = mask_e14 & np.isfinite(p_margin) & np.isfinite(p_elo)
    if not np.isfinite(p_can[common]).all():
        raise AssertionError("canonical OOF is missing a row in the common mask")
    yy = y[common]
    probs = {"e14": p_e14[common], "margin": p_margin[common],
             "elo": p_elo[common]}
    ref = probs["e14"]

    member_metrics = {name: metrics(yy, p) for name, p in probs.items()}
    member_metrics["canonical (reference, not a member)"] = metrics(
        yy, p_can[common])
    member_deltas = {
        name: ms.delta_stats(yy, p, ref, n_boot=N_BOOT, seed=SEED)
        for name, p in probs.items() if name != "e14"
    }

    disagreement = en.disagreement_table(yy, probs)

    arm_rows: dict[str, dict] = {}
    for name, weights in FIXED_ARMS.items():
        p = en.blend(probs, weights)
        p_full = np.full(n, np.nan)
        p_full[common] = p
        arm_rows[name] = {
            "weights": weights,
            "metrics": metrics(yy, p),
            "delta_vs_e14": ms.delta_stats(yy, p, ref, n_boot=N_BOOT, seed=SEED),
            "per_fold_vs_e14": ms.per_fold_delta(y, p_full, p_e14, games),
        }

    grid: list[dict] = []
    best = None
    for weights in en.simplex_grid(NAMES, step=0.1):
        p = en.blend(probs, weights)
        ll = ms.log_loss(yy, p)
        row = {"weights": weights, "log_loss": float(ll),
               "delta_vs_e14": float(ms.log_loss(yy, ref) - ll)}
        grid.append(row)
        if best is None or ll < best["log_loss"]:
            best = row

    primary = arm_rows["equal_weights"]
    d = primary["delta_vs_e14"]
    promoted = bool(d["delta_candidate_minus_reference"] > 0 and d["abs_z"] > 3)
    verdict = (
        "PROMOTED" if promoted else
        "NULL: the fixed equal-weight ensemble does not clear the E14 bar at "
        "|z| > 3. The members are not near-duplicates, but the blend still "
        "does not beat the best member.")

    payload = {
        "issue": 42,
        "protocol_ref": "I42-prereg-ensemble",
        "generated_utc": ms.utc_now(),
        "runtime_seconds": round(time.time() - t0, 1),
        "n_scored": int(common.sum()),
        "elo_tuned_k_reused_from_baselines": tuned_k,
        "members": member_metrics,
        "member_deltas_vs_e14": member_deltas,
        "disagreement": disagreement,
        "fixed_arms": arm_rows,
        "simplex_grid_step": 0.1,
        "simplex_grid": grid,
        "grid_max_is_descriptive_only": best,
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    L: list[str] = []
    a = L.append
    a("=" * 78)
    a("ISSUE 42 -- ENSEMBLE DIAGNOSTIC AND FIXED-WEIGHT BLENDS")
    a("=" * 78)
    a(f"n scored {int(common.sum())}; Elo member reuses the baseline tuned "
      f"K={tuned_k:.0f} (recorded model 0.6234)")
    a("")
    a("MEMBERS")
    a(f"{'model':<40} {'logloss':>9} {'brier':>8} {'auc':>7} {'acc':>7}")
    for name, m in member_metrics.items():
        a(f"{name:<40} {m['log_loss']:>9.6f} {m['brier']:>8.6f} "
          f"{m['auc']:>7.4f} {m['accuracy']:>7.4f}")
    a("")
    a("PAIRWISE DISAGREEMENT (0.5 threshold)")
    for pair, row in disagreement.items():
        left, right = [s.strip() for s in pair.split(" vs ")]
        a(f"{pair:<24} n={row['n_disagree']:>4} "
          f"rate={row['disagreement_rate']:.4f} "
          f"hit({left})={row[f'{left}_hit_rate_on_disagreement']:.4f} "
          f"hit({right})={row[f'{right}_hit_rate_on_disagreement']:.4f}")
    a("")
    a("FIXED ARMS (positive delta = ensemble better than E14)")
    a(f"{'arm':<18} {'logloss':>9} {'brier':>8} {'auc':>7} {'acc':>7} "
      f"{'delta':>10} {'CI':>24} {'|z|':>5}")
    for name, row in arm_rows.items():
        m, d = row["metrics"], row["delta_vs_e14"]
        a(f"{name:<18} {m['log_loss']:>9.6f} {m['brier']:>8.6f} "
          f"{m['auc']:>7.4f} {m['accuracy']:>7.4f} "
          f"{d['delta_candidate_minus_reference']:>+10.6f} "
          f"[{d['ci_lo']:+.6f},{d['ci_hi']:+.6f}] {d['abs_z']:>5.2f}")
    a("")
    a("FULL 0.1 SIMPLEX (descriptive; no point selected)")
    a(f"{'w_e14':>5} {'w_margin':>8} {'w_elo':>6} {'logloss':>9} "
      f"{'delta_vs_e14':>13}")
    for row in grid:
        w = row["weights"]
        a(f"{w['e14']:>5.1f} {w['margin']:>8.1f} {w['elo']:>6.1f} "
          f"{row['log_loss']:>9.6f} {row['delta_vs_e14']:>+13.6f}")
    a("")
    a(f"grid maximum (descriptive only): "
      f"{best['weights']} logloss {best['log_loss']:.6f}")
    a(f"verdict: {verdict}")
    text = "\n".join(L)
    print(text)
    OUT_MD.write_text(text + "\n")

    ms.append_jsonl(JSONL, {
        "name": "I42-run-ensemble",
        "status": "executed",
        "declared_utc": ms.utc_now(),
        "experiment_type": "ensemble diagnostic and fixed-weight result (issue #42)",
        "protocol_ref": "I42-prereg-ensemble",
        "n_scored": int(common.sum()),
        "members": member_metrics,
        "fixed_arms": {k: {"weights": v["weights"],
                           "log_loss": v["metrics"]["log_loss"],
                           "delta_vs_e14": v["delta_vs_e14"]}
                       for k, v in arm_rows.items()},
        "primary_arm": "equal_weights",
        "primary_delta_vs_e14": primary["delta_vs_e14"],
        "grid_max_descriptive": best,
        "verdict": verdict,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
