"""H3 (theory dossier): season-blocked, strictly causal recalibration of the
canonical model. Pre-registered in results/experiments.jsonl as
``H3-prereg-recalibration`` BEFORE this script was run. That entry governs.

THE HYPOTHESIS
--------------
Season-blocked, strictly causal recalibration of the canonical model's
out-of-fold probabilities - isotonic, Platt, and a regime-aware home-term
shrink (home-court advantage measurably shrank from ~2.74 to ~1.75 pts after
2020, Pelechrinis 2023) - improves held-out log loss by >= 0.003 nats at zero
new information.

WHAT IS CANONICAL HERE
----------------------
`src/Models/logreg_rolling.py`: RollingLogreg (C=1.0, L2, lbfgs) on
``load_design()`` features resolved through ``DEFAULT_MODELS["full"]`` (22
contrast columns incl. ``home_adv``), scored on the shared
``walk_forward_split(n_splits=5, initial_frac=0.5, gap=0)`` - 6,600 out-of-fold
rows. Pooled OOF log loss 0.61957 in this session. The model is NOT refit,
retuned or re-featured here: the probabilities are produced ONCE and calibrated
post-hoc.

THE FAILURE MODE THIS FILE IS BUILT AROUND
-----------------------------------------
The repo documents a pipeline that fit its calibrator on the same tail used for
early stopping and validation. Here, for each scored season block S, every
fitted calibrator sees ONLY the OOF rows of the immediately preceding season
block S-1 - never a row of S, never a label of S. The first block (the partial
2020-21 tail) has no in-region predecessor and passes through identity. This is
declared in the pre-registration ("fit_slices") and asserted here: no variant
may fit and score the same rows.

PRIOR, STATED RATHER THAN HIDDEN
--------------------------------
results/round3_calib.jsonl found nested calibrators a null on the E14 frame
(temperature -0.0005, platt -0.0008, beta -0.0008, isotonic -0.0080 worse).
That run differs in model/frame, fit slicing and has no regime variant. The
prior is AGAINST H3. A null here is a complete result.

Run:  python scripts/h3_recalibration.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from DataProviders import nba_data as nd          # noqa: E402
from Models import logreg_rolling as lr           # noqa: E402
from Utils import metrics as um                   # noqa: E402

JSONL = ROOT / "results" / "experiments.jsonl"
OUT = ROOT / "results" / "h3_recalibration.json"

N_SPLITS = 5
INITIAL_FRAC = 0.5
C_MODEL = 1.0
N_BOOT = 2000
BOOT_SEEDS = (42, 137, 2718)
# FIXED ex ante from Pelechrinis (2023): post-2020 / pre-2020 home margin in
# points. No sweep over this constant - it is a pre-registered single value.
RHO = 1.75 / 2.74

VARIANTS = ("isotonic_prevseason", "platt_prevseason",
            "home_regime_shrink", "home_intercept_adj")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #

def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


def metrics(y, p) -> dict:
    """Log loss, Brier, AUC, accuracy + ECE (uniform-10 and quantile-10)."""
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    m = nd.evaluate(y, p)
    return {
        "n": int(len(y)),
        "log_loss": float(m["log_loss"]),
        "brier": float(m["brier"]),
        "auc": float(m["auc"]),
        "accuracy": float(m["accuracy"]),
        "ece_uniform10": float(um.expected_calibration_error(y, p, n_bins=10,
                                                             strategy="uniform")),
        "ece_quantile10": float(um.expected_calibration_error(y, p, n_bins=10,
                                                              strategy="quantile")),
    }


def bootstrap(y, p_before, p_after, seed) -> dict:
    """loss(before) - loss(after); POSITIVE = the recalibration improved.

    Same sign convention as nba_data.paired_bootstrap_delta, which is where the
    number comes from. |z| from the CI width, as feature_block_ablation does.
    """
    d = nd.paired_bootstrap_delta(y, p_before, p_after,
                                  n_boot=N_BOOT, seed=seed)
    se = (d["ci_hi"] - d["ci_lo"]) / (2 * 1.959964)
    return {
        "seed": seed, "n_boot": N_BOOT,
        "delta": float(d["delta"]),
        "ci_lo": float(d["ci_lo"]), "ci_hi": float(d["ci_hi"]),
        "p_positive": float(d["p_positive"]),
        "se": float(se),
        "z_vs_zero": float(d["delta"] / se) if se > 0 else float("nan"),
        # the pre-registered minimum effect is 0.003 nats
        "z_vs_target_0p003": float((d["delta"] - 0.003) / se) if se > 0
        else float("nan"),
    }


# --------------------------------------------------------------------------- #
# The canonical OOF stream (produced once; nothing below refits the model)
# --------------------------------------------------------------------------- #

def canonical_oof(X: pd.DataFrame, games: pd.DataFrame, y: np.ndarray,
                  features: list[str]):
    """p[te] and the fold model's home_adv coefficient h[te] per OOF row."""
    p = np.full(len(y), np.nan)
    h = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)
    splits = nd.walk_forward_split(games, n_splits=N_SPLITS,
                                   initial_frac=INITIAL_FRAC)
    home_terms = {}
    for k, (tr, te) in enumerate(splits, start=1):
        m = lr.RollingLogreg(C=C_MODEL, features=features).fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])
        hk = float(m.home_edge_logodds())
        h[te] = hk
        fold_id[te] = k
        home_terms[k] = hk
    return p, h, fold_id, splits, home_terms


# --------------------------------------------------------------------------- #
# Calibrators. Each fitted variant sees ONLY (p, y) of the PREVIOUS season
# block. home_regime_shrink has no fit slice at all (zero parameters).
# --------------------------------------------------------------------------- #

def cal_isotonic(p_fit, y_fit, p_block):
    from sklearn.isotonic import IsotonicRegression
    m = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    m.fit(p_fit, y_fit)
    return np.asarray(m.predict(p_block)).ravel(), "isotonic"


def cal_platt(p_fit, y_fit, p_block):
    from sklearn.linear_model import LogisticRegression
    z = _logit(p_fit).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=5000).fit(z, y_fit)
    a = float(m.coef_[0][0])
    b = float(m.intercept_[0])
    return _sigmoid(a * _logit(p_block) + b), {"a": a, "b": b}


def cal_home_intercept(p_fit, y_fit, p_block):
    """Intercept-only MLE: z' = z + gamma, gamma solves sum(sigmoid(z+g) - y) = 0.

    Bisection on a monotone score function: no optimiser, no tuning, and it
    cannot fail to converge on a well-posed fit block.
    """
    z = _logit(p_fit)
    y = np.asarray(y_fit).astype(float)

    def score(g):
        return float(np.sum(_sigmoid(z + g) - y))

    lo, hi = -10.0, 10.0
    if score(lo) > 0 or score(hi) < 0:
        return p_block, {"gamma": float("nan"), "note": "degenerate fit block"}
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if score(mid) < 0:
            lo = mid
        else:
            hi = mid
    g = 0.5 * (lo + hi)
    return _sigmoid(_logit(p_block) + g), {"gamma": g}


def apply_variants(p, h, y, blocks, prev_of):
    """p_after per variant over the whole OOF stream, plus fitted params."""
    out = {v: np.full(len(p), np.nan) for v in VARIANTS}
    params = {v: [] for v in VARIANTS}
    for bi, rows in enumerate(blocks):
        has_prev = prev_of[bi] is not None
        prev = blocks[prev_of[bi]] if has_prev else None
        for v in VARIANTS:
            if v == "home_regime_shrink":
                # zero-parameter regime shrink; no fit slice exists
                out[v][rows] = _sigmoid(_logit(p[rows]) - (1 - RHO) * h[rows])
                params[v].append({"block_index": bi, "params": f"rho={RHO:.4f}"})
                continue
            if not has_prev:
                out[v][rows] = p[rows]                    # identity, declared
                params[v].append({"block_index": bi, "params": "identity "
                                  "(no preceding season block)"})
                continue
            p_fit, y_fit = p[prev], y[prev]
            if v == "isotonic_prevseason":
                q, prm = cal_isotonic(p_fit, y_fit, p[rows])
            elif v == "platt_prevseason":
                q, prm = cal_platt(p_fit, y_fit, p[rows])
            else:
                q, prm = cal_home_intercept(p_fit, y_fit, p[rows])
            out[v][rows] = q
            params[v].append({"block_index": bi, "n_fit": int(len(prev)),
                              "params": prm})
    return out, params


def main() -> int:
    t0 = time.time()
    g = nd.build_games()
    y = g["y"].to_numpy(dtype=int)
    X = lr.load_design(with_availability=False)
    features = lr.prune_features(X, lr.resolve_features(lr.DEFAULT_MODELS["full"]),
                                 context="h3")
    print(f"[{time.time()-t0:.0f}s] games {len(g):,}  design {X.shape}  "
          f"canonical features {len(features)}", flush=True)

    p, h, fold_id, splits, home_terms = canonical_oof(X, g, y, features)
    scored = ~np.isnan(p)
    assert scored.sum() == sum(len(te) for _, te in splits)
    idx = np.flatnonzero(scored)
    print(f"[{time.time()-t0:.0f}s] canonical OOF: {len(idx):,} rows  "
          f"home_adv per fold: "
          + "  ".join(f"f{k}={v:.4f}" for k, v in sorted(home_terms.items())),
          flush=True)

    before = metrics(y[idx], p[idx])
    print(f"[{time.time()-t0:.0f}s] BEFORE (identity): log_loss "
          f"{before['log_loss']:.5f}  brier {before['brier']:.5f}  "
          f"ece_u {before['ece_uniform10']:.4f}  "
          f"ece_q {before['ece_quantile10']:.4f}", flush=True)

    # ---- season blocks, strictly chronological -----------------------------
    seasons = g["SEASON"].to_numpy()[idx]
    dates = g["GAME_DATE"].to_numpy()[idx]
    order = sorted(set(seasons.tolist()), key=lambda s: dates[seasons == s].min())
    blocks, block_meta = [], []
    for s in order:
        rows = idx[seasons == s]
        blocks.append(rows)
        block_meta.append({"season": str(s), "n": int(len(rows)),
                           "first": str(np.min(g["GAME_DATE"].to_numpy()[rows]))[:10],
                           "last": str(np.max(g["GAME_DATE"].to_numpy()[rows]))[:10]})
    prev_of = {i: (i - 1 if i > 0 else None) for i in range(len(blocks))}
    # THE GUARD for the documented failure mode: a fitted calibrator's fit rows
    # and score rows must be disjoint. Asserted, not assumed.
    for i, rows in enumerate(blocks):
        if prev_of[i] is None:
            continue
        prev = blocks[prev_of[i]]
        assert not np.intersect1d(rows, prev).size, \
            "calibrator fit slice overlaps its score slice"

    p_after, params = apply_variants(p, h, y, blocks, prev_of)
    print(f"[{time.time()-t0:.0f}s] season blocks: "
          + "  ".join(f"{b['season']}({b['n']})" for b in block_meta), flush=True)

    # ---- scoring ----------------------------------------------------------
    results = {}
    for v in VARIANTS:
        q = p_after[v]
        assert np.isfinite(q[idx]).all()
        after = metrics(y[idx], q[idx])
        per_block = []
        for bi, rows in enumerate(blocks):
            mb = metrics(y[rows], p[rows])
            ma = metrics(y[rows], q[rows])
            per_block.append({
                "season": block_meta[bi]["season"], "n": int(len(rows)),
                "log_loss_before": mb["log_loss"], "log_loss_after": ma["log_loss"],
                "brier_before": mb["brier"], "brier_after": ma["brier"],
                "ece_uniform10_before": mb["ece_uniform10"],
                "ece_uniform10_after": ma["ece_uniform10"],
                "improvement_nats": mb["log_loss"] - ma["log_loss"],
                "fitted": params[v][bi],
            })
        boots = [bootstrap(y[idx], p[idx], q[idx], seed) for seed in BOOT_SEEDS]
        results[v] = {"before": before, "after": after,
                      "improvement_nats": before["log_loss"] - after["log_loss"],
                      "bootstrap": boots, "per_block": per_block}
        b0 = boots[0]
        print(f"[{time.time()-t0:6.0f}s] {v:22s} ll {after['log_loss']:.6f} "
              f"imp {b0['delta']:+.6f} CI [{b0['ci_lo']:+.6f},{b0['ci_hi']:+.6f}] "
              f"z0 {b0['z_vs_zero']:+.2f} z_target {b0['z_vs_target_0p003']:+.2f} "
              f"ece_u {after['ece_uniform10']:.4f}", flush=True)

    # ---- selection cost over the 4 variants (best-of-N is a maximum of N) --
    imps = np.array([results[v]["bootstrap"][0]["delta"] for v in VARIANTS])
    selection = {"best_variant": str(VARIANTS[int(np.argmax(imps))]),
                 "best_minus_median": float(imps.max() - np.median(imps)),
                 "best_minus_worst": float(imps.max() - imps.min()),
                 "all_improvements": {v: float(results[v]["bootstrap"][0]["delta"])
                                      for v in VARIANTS}}

    # ---- verdict against the pre-registered rule --------------------------
    passed = []
    for v in VARIANTS:
        b0 = results[v]["bootstrap"][0]
        ok = (b0["delta"] >= 0.003 and b0["ci_lo"] > 0 and abs(b0["z_vs_zero"]) > 3)
        if ok:
            passed.append(v)
    verdict = {
        "rule": "some variant reaches pooled improvement >= 0.003 nats with 95% "
                "CI excluding zero AND |z| > 3 (H3-prereg-recalibration)",
        "variants_passing": passed,
        "decision": "H3 SUCCEEDS" if passed else
                    "H3 FAILS its pre-registered rule: no variant reaches "
                    "0.003 nats with |z| > 3. Recalibration is a null / "
                    "sub-bar at zero new information.",
        "multiplicity": "4 variant looks -> Bonferroni 0.0125 one-sided; "
                        "the |z| > 3 bar is stricter and governs",
        "selection_cost": selection,
        "max_abs_z_over_variants": float(max(abs(results[v]["bootstrap"][0]["z_vs_zero"])
                                             for v in VARIANTS)),
    }

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pre_registration": "results/experiments.jsonl :: H3-prereg-recalibration",
        "canonical": {"model": "RollingLogreg C=1.0 on load_design() "
                               "resolve_features('full')",
                      "n_features": len(features),
                      "n_splits": N_SPLITS, "initial_frac": INITIAL_FRAC,
                      "oof_rows": int(len(idx)),
                      "home_adv_per_fold": home_terms,
                      "before": before},
        "season_blocks": block_meta,
        "rho_home_regime_shrink": RHO,
        "variants": results,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {OUT}", flush=True)

    # ---- append one record per variant + a verdict record ------------------
    ts = datetime.now(timezone.utc).isoformat()
    with JSONL.open("a") as fh:
        for v in VARIANTS:
            b0 = results[v]["bootstrap"][0]
            after = results[v]["after"]
            fh.write(json.dumps({
                "name": f"H3-run-{v}",
                "status": "executed",
                "declared_utc": ts,
                "experiment_type": "post-hoc recalibration result (H3)",
                "protocol_ref": "H3-prereg-recalibration",
                "variant": v,
                "fit_slice": "immediately-preceding season block (identity for "
                             "the first block; home_regime_shrink has no fit slice)",
                "n_scored": int(len(idx)),
                "log_loss_before": round(before["log_loss"], 6),
                "log_loss_after": round(after["log_loss"], 6),
                "brier_before": round(before["brier"], 6),
                "brier_after": round(after["brier"], 6),
                "ece_uniform10_before": round(before["ece_uniform10"], 6),
                "ece_uniform10_after": round(after["ece_uniform10"], 6),
                "improvement_nats": round(b0["delta"], 6),
                "ci_lo": round(b0["ci_lo"], 6), "ci_hi": round(b0["ci_hi"], 6),
                "se": round(b0["se"], 6),
                "z_vs_zero": round(b0["z_vs_zero"], 3),
                "z_vs_target_0p003": round(b0["z_vs_target_0p003"], 3),
                "bootstrap_seeds": list(BOOT_SEEDS),
                "bootstrap_replication_improvements": [
                    round(b["delta"], 6) for b in results[v]["bootstrap"]],
                "per_block_improvement_nats": {
                    pb["season"]: round(pb["improvement_nats"], 6)
                    for pb in results[v]["per_block"]},
                "artifact": "results/h3_recalibration.json",
            }) + "\n")
        fh.write(json.dumps({
            "name": "H3-verdict",
            "status": "executed",
            "declared_utc": ts,
            "experiment_type": "verdict_record",
            "protocol_ref": "H3-prereg-recalibration",
            "decision_rule": verdict["rule"],
            "decision": verdict["decision"],
            "variants_passing": verdict["variants_passing"],
            "max_abs_z_over_variants": verdict["max_abs_z_over_variants"],
            "selection_cost": verdict["selection_cost"],
            "multiplicity": verdict["multiplicity"],
            "leakage_check": "passed: each fitted calibrator's fit rows and "
                             "score rows are disjoint season blocks (asserted in "
                             "code); no calibrator sees a label from the block it "
                             "scores; model probabilities produced once, "
                             "post-hoc calibration only",
        }) + "\n")
    print(f"[{time.time()-t0:.0f}s] appended {len(VARIANTS) + 1} records to "
          f"results/experiments.jsonl", flush=True)
    print(f"\n{verdict['decision']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
