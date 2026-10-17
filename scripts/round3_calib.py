"""Round 3, block CAL: probability calibration, nested honestly.

WHY
---
The 67-feature ablation's calibration table (results/feature_block_ablation.json)
shows the model is mildly OVERCONFIDENT in both tails: it predicts 0.842 where
0.829 actually happens, 0.748 where 0.737 happens, 0.165 where 0.155 happens.
That is a pure calibration error and it costs log loss directly. Log loss is the
only one of the four headline metrics that punishes it.

This is a lever nobody has pulled in this repo: twelve feature arms, zero
calibration arms. And it needs NO new data - it is free.

THE TRAP, AND WHY THIS FILE IS NESTED
-------------------------------------
Fitting a calibrator on the out-of-fold predictions and then scoring those same
OOF predictions is leakage of the mildest and most seductive kind. The
calibrator sees every label it is then graded on. It will look like a win and it
is not one. On 6,600 rows a 2-parameter Platt fit will happily shave 0.002 nats
off a number that does not move on future data.

So the calibrator is fitted on an INNER walk-forward OOF computed from the outer
fold's training rows only, and is then applied to the outer test fold:

    for each outer fold:
        inner 3-split walk-forward over the TRAINING rows -> inner OOF preds
        fit calibrator on (inner OOF preds, training labels)
        fit the model on the full TRAINING rows -> predict outer test
        apply the calibrator to the outer test predictions

The calibrator never sees the outer test block, and never sees a label from it.
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

JSONL = ROOT / "results" / "round3_calib.jsonl"
BOOT_SEED = 42
N_BOOT = 2000
N_INNER = 3


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))


# --------------------------------------------------------------------------- #
# Calibrators. Each is fit(raw_inner_pred, y_inner) and applied(raw_pred).
# --------------------------------------------------------------------------- #
def cal_none(p_fit, y_fit, p):
    return p, "identity"


def cal_temperature(p_fit, y_fit, p):
    """One parameter: p' = sigmoid(logit(p)/T). Shrinks toward 0.5.

    T > 1 tames overconfidence, T < 1 sharpens. Fit by direct 1-D search on the
    inner log loss - a convex problem, so this is not an optimiser in disguise.
    """
    z = _logit(p_fit)
    best_T, best_ll = 1.0, np.inf
    for T in np.exp(np.linspace(np.log(0.4), np.log(4.0), 121)):
        q = _sigmoid(z / T)
        ll = -np.mean(y_fit * np.log(q) + (1 - y_fit) * np.log(1 - q))
        if ll < best_ll:
            best_ll, best_T = ll, T
    return _sigmoid(_logit(p) / best_T), best_T


def cal_platt(p_fit, y_fit, p):
    """Two parameters: p' = sigmoid(a*logit(p) + b). a also corrects sharpness."""
    from sklearn.linear_model import LogisticRegression
    z = _logit(p_fit).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=5000).fit(z, y_fit)
    a = float(m.coef_[0][0])
    b = float(m.intercept_[0])
    return _sigmoid(a * _logit(p) + b), (a, b)


def cal_beta(p_fit, y_fit, p):
    """Three parameters (Kull et al.): a*logit(p) + b*logit(1-p) + c.

    The 3-term form separates a slope, a curvature and an intercept, which is what
    a two-parameter Platt cannot do when the miscalibration is U-shaped.
    """
    from sklearn.linear_model import LogisticRegression
    z = np.column_stack([_logit(p_fit), _logit(1 - p_fit)])
    m = LogisticRegression(C=1e6, max_iter=5000).fit(z, y_fit)
    a, b = (float(v) for v in m.coef_[0])
    c = float(m.intercept_[0])
    return _sigmoid(a * _logit(p) + b * _logit(1 - p) + c), (a, b, c)


def cal_isotonic(p_fit, y_fit, p):
    from sklearn.isotonic import IsotonicRegression
    m = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    m.fit(p_fit, y_fit)
    return np.asarray(m.predict(p)).ravel(), "isotonic"


CALIBRATORS = {
    "none": cal_none,
    "temperature": cal_temperature,
    "platt": cal_platt,
    "beta": cal_beta,
    "isotonic": cal_isotonic,
}


# --------------------------------------------------------------------------- #
def inner_oof(X, y, n_inner=N_INNER):
    """Walk-forward inner OOF over an already-chronological block of rows."""
    n = len(y)
    p = np.full(n, np.nan)
    # Same shape as the outer protocol: the first half is training-only.
    first = int(n * 0.5)
    bounds = np.linspace(first, n, n_inner + 1).astype(int)
    for i in range(1, n_inner + 1):
        lo, hi = bounds[i - 1], bounds[i]
        if hi <= lo:
            continue
        m = rc.e14_pipeline()
        m.fit(X[:lo], y[:lo])
        p[lo:hi] = m.predict_proba(X[lo:hi])[:, 1]
    return p


def run_calibrated(X, y, g, folds, calib) -> tuple[np.ndarray, list]:
    """Out-of-fold predictions, each block calibrated by a calibrator that never
    saw that block."""
    p = np.full(len(y), np.nan)
    fitted = []
    for tr, te in folds:
        m = rc.e14_pipeline()
        m.fit(X[tr], y[tr])
        raw_te = m.predict_proba(X[te])[:, 1]
        pin = inner_oof(X[tr], y[tr])
        ok = np.isfinite(pin)
        if ok.sum() < 200:
            p[te] = raw_te
            fitted.append({"n_train_inner": int(ok.sum()), "params": "SKIPPED"})
            continue
        cal, params = CALIBRATORS[calib](pin[ok], y[tr][ok], raw_te)
        p[te] = np.clip(cal, 1e-6, 1 - 1e-6)
        fitted.append({"n_train_inner": int(ok.sum()), "params": str(params)})
    return p, fitted


def _finite(y, *ps):
    """Mask to the scored region; `walk_forward_oof` NaNs the cold-start prefix
    and `nd.evaluate` / `paired_bootstrap_delta` reject NaN."""
    m = np.isfinite(y.astype(float))
    for p in ps:
        m &= np.isfinite(p)
    return m


def evaluate(y, p, p_base, folds) -> dict:
    m = _finite(y, p, p_base)
    s = rc.score(y[m], p[m])
    d = rc.delta_vs(y[m], p[m], p_base[m], seed=BOOT_SEED, n_boot=N_BOOT)
    per_fold = []
    for _tr, te in folds:
        m = np.isfinite(p[te]) & np.isfinite(p_base[te])
        ll_c = rc.score(y[te][m], p[te][m])["log_loss"]
        ll_b = rc.score(y[te][m], p_base[te][m])["log_loss"]
        per_fold.append({"n": int(m.sum()), "cand": round(ll_c, 6),
                         "base": round(ll_b, 6), "delta": round(ll_c - ll_b, 6)})
    deltas = np.array([f["delta"] for f in per_fold])
    gains = -deltas
    return {
        "log_loss": round(s["log_loss"], 6), "brier": round(s["brier"], 6),
        "auc": round(s["auc"], 6), "accuracy": round(s["accuracy"], 6),
        "improvement_nats": round(d["improvement_nats"], 6),
        "ci_lo": round(d["ci_lo"], 6), "ci_hi": round(d["ci_hi"], 6),
        "abs_z": round(d["abs_z"], 3),
        "per_fold_delta": [round(float(v), 6) for v in deltas],
        "fold_gain_mean": round(float(gains.mean()), 6),
        "fold_gain_sd": round(float(gains.std(ddof=1)), 6),
        "folds_where_cal_wins": int((gains > 0).sum()),
        "clears_fold_spread": bool(abs(gains.mean()) > gains.std(ddof=1)),
    }


def main() -> None:
    t0 = time.time()
    g = nd.build_games()
    y = g["y"].to_numpy(int)
    frame = rc.build_e14_frame(g)
    X = frame.to_numpy(float)
    folds = nd.walk_forward_split(g, n_splits=5, initial_frac=0.5)
    print(f"[{time.time()-t0:.0f}s] frame={frame.shape} folds={len(folds)}", flush=True)

    p_base = rc.walk_forward_oof(X, y, g)
    sm = _finite(y, p_base)
    s = rc.score(y[sm], p_base[sm])
    with JSONL.open("a") as fh:
        fh.write(json.dumps({"run_id": "R0_baseline_e14", "block": "REFERENCE",
                             "n_features": frame.shape[1], **s}) + "\n")
    print(f"[{time.time()-t0:.0f}s] BASELINE log_loss={s['log_loss']:.6f} "
          f"(expect 0.618307)", flush=True)

    rows = []
    for name in CALIBRATORS:
        t1 = time.time()
        try:
            p, fitted = run_calibrated(X, y, g, folds, name)
            ev = evaluate(y, p, p_base, folds)
            rows.append((name, ev, fitted))
            print(f"[{time.time()-t0:6.0f}s] {name:12s} ll={ev['log_loss']:.6f} "
                  f"imp={ev['improvement_nats']:+.6f} |z|={ev['abs_z']:.2f} "
                  f"foldgain={ev['fold_gain_mean']:+.6f}±{ev['fold_gain_sd']:.6f} "
                  f"wins={ev['folds_where_cal_wins']}/5 "
                  f"clears={ev['clears_fold_spread']}", flush=True)
            print(f"{'':20s}params per fold: {fitted}", flush=True)
            with JSONL.open("a") as fh:
                fh.write(json.dumps({"run_id": f"CAL_{name}", "block": "CAL",
                                     "calibrator": name, "fitted": fitted, **ev,
                                     "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                              time.gmtime())}) + "\n")
        except Exception as exc:  # a crash is a result
            import traceback
            print(f"[{time.time()-t0:6.0f}s] {name:12s} FAILED\n{traceback.format_exc()}",
                  flush=True)
            with JSONL.open("a") as fh:
                fh.write(json.dumps({"run_id": f"CAL_{name}", "kind": "FAILED",
                                     "error": traceback.format_exc()}) + "\n")

    print("\n== the deliberately-illegal version, for contrast ==")
    # Fit on the OOF predictions themselves and score those same predictions.
    # NOT a result. Reported so the honest number above can be read against it.
    for name in ("temperature", "platt", "isotonic"):
        p, _f = run_calibrated(X, y, g, folds, name)   # honest, to get the raw stream
        raw = np.full(len(y), np.nan)
        for tr, te in folds:
            m = rc.e14_pipeline().fit(X[tr], y[tr])
            raw[te] = m.predict_proba(X[te])[:, 1]
        ok = np.isfinite(raw)
        cheat, params = CALIBRATORS[name](raw[ok], y[ok], raw[ok])
        ll = rc.score(y[ok], cheat)["log_loss"]
        honest = next((r[1]["log_loss"] for r in rows if r[0] == name), float("nan"))
        print(f"  {name:12s} in-sample-on-OOF log_loss={ll:.6f}  "
              f"honest={honest:.6f}  inflation={honest - ll:+.6f}  params={params}",
              flush=True)

    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()