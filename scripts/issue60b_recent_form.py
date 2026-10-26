"""Issue #60 follow-on: recent-form win-count windows, decided on 2025-26 only.

Not the streak test (I60).  This asks whether a team's WIN COUNT over its last
5 or last 10 games entering tonight carries signal beyond the canonical EWM
--- whether a team that is 4-from-5 beats a team that is 1-from-5, at every
bracket of both windows.  The DECISION is the same single last-season
holdout as I60: train = everything before 2025-10-21 (11,974 rows), test =
the 1,225 games of 2025-26.  No whole-frame number is computed or reported.

Two PRIMARY arms, each judged independently at the full |z| > 3 bar:
form5 (canonical 26 + the full >=1..>=5 ladder on both sides, 36 cols) and
form10 (canonical 26 + the >=1..>=10 ladders, 46 cols).  The pre-registered
bracket lists (the six 5-game forms and eleven 10-game forms) are released as
cumulative per-side ladders, which have the same span as the one-hot level
sets: the "4-from-5 vs 1-from-5" question is answered by the contrast between
the two sides' ladders, and every bracket possibility is in the model.

Pre-registered in ``results/experiments.jsonl`` as
``I60b-prereg-recent-form`` before this module ran.

Run::

    OMP_NUM_THREADS=1 python scripts/issue60b_recent_form.py
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

from DataProviders import nba_data as nd        # noqa: E402
from Models import logreg_rolling as lr         # noqa: E402
from Models import measurement as ms            # noqa: E402

OUT_JSON = ROOT / "results" / "issue60b_recent_form.json"
OUT_MD = ROOT / "results" / "issue60b_recent_form.md"
JSONL = ROOT / "results" / "experiments.jsonl"

OPENER = pd.Timestamp("2025-10-21")
N_BOOT = 2000
SEED = 42
WINDOWS = (5, 10)

_CORE_METRICS = ("n", "log_loss", "brier", "accuracy", "auc")


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    out = nd.evaluate(y, p)
    return {k: (int(out[k]) if k == "n" else float(out[k]))
            for k in _CORE_METRICS}


def delta(y: np.ndarray, p_cand: np.ndarray, p_ref: np.ndarray) -> dict:
    d = ms.delta_stats(y, p_cand, p_ref, n_boot=N_BOOT, seed=SEED)
    d["bootstrap_seed"] = SEED
    return d


def fit_predict(X: pd.DataFrame, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
                features: list[str]) -> np.ndarray:
    m = lr.RollingLogreg(C=1.0, features=features).fit(X.iloc[tr], y[tr])
    return m.predict_proba(X.iloc[te])


def main() -> int:
    t0 = time.time()
    games = nd.build_games().reset_index(drop=True)
    g_dates = pd.to_datetime(games["GAME_DATE"])
    tr = (g_dates < OPENER).to_numpy()
    te = (g_dates >= OPENER).to_numpy()
    y = games["y"].to_numpy(dtype=int)
    y_tr, y_te = y[tr], y[te]
    n_test = int(te.sum())
    if n_test != 1225:
        raise AssertionError(f"expected 1,225 test rows, got {n_test}")

    X_all = lr.load_design(games=games).reset_index(drop=True)
    rf = nd.recent_form_features(games).reset_index(drop=True)

    features_full = lr.resolve_features(lr.DEFAULT_MODELS["full"])
    recipes = {W: lr.resolve_features(lr.DEFAULT_MODELS[f"full_form{W}"])
               for W in WINDOWS}
    assert all(len(recipes[W]) == 26 + 2 * W for W in WINDOWS)

    p_ref = fit_predict(X_all, y, tr, te, features_full)
    ref_m = metrics(y_te, p_ref)

    arms = {}
    for W in WINDOWS:
        ladders = lr.form_brackets(rf, W)
        Xc = pd.concat([X_all, ladders], axis=1).reset_index(drop=True)
        p = fit_predict(Xc, y, tr, te, recipes[W])
        arms[W] = {"features": recipes[W][26:], "log_loss":
                   float(nd.evaluate(y_te, p)["log_loss"]),
                   "metrics": metrics(y_te, p),
                   "delta": delta(y_te, p, p_ref)}

    # Positive control: availability on vs off, same split (I60 measured
    # +0.009247, |z| 4.93; this run re-reports it for the same power light).
    avail_cols = lr.resolve_features("availability")
    p_noavail = fit_predict(X_all, y, tr, te,
                            [f for f in features_full if f not in avail_cols])
    d_avail = delta(y_te, p_ref, p_noavail)

    # Per-bracket coefficients (train fit, descriptive) for both windows.
    coef = {}
    for W in WINDOWS:
        ladders = lr.form_brackets(rf, W)
        Xc = pd.concat([X_all, ladders], axis=1).reset_index(drop=True)
        m = lr.RollingLogreg(C=1.0, features=recipes[W]).fit(Xc.iloc[tr], y_tr)
        c = m.coefficients()["coef"]
        coef[W] = {k: float(c[k]) for k in recipes[W][26:]}

    # Sensitivity (descriptive; cannot promote).
    def arm(extra_cols: pd.DataFrame, feats: list[str], label: str) -> dict:
        Xc = pd.concat([X_all, extra_cols], axis=1).reset_index(drop=True)
        p = fit_predict(Xc, y, tr, te, feats)
        da = delta(y_te, p, p_ref)
        return {"log_loss": float(nd.evaluate(y_te, p)["log_loss"]),
                "delta": da}

    f5 = lr.form_brackets(rf, 5)
    f10 = lr.form_brackets(rf, 10)
    sens = {
        "form5_and_form10": arm(pd.concat([f5, f10], axis=1),
                                features_full + list(f5.columns)
                                + list(f10.columns), "both"),
        "form5_home_only": arm(f5[["home_form5_ge%i" % k for k in range(1, 6)]],
                               features_full + [f"home_form5_ge{k}"
                                                for k in range(1, 6)],
                               "f5_home"),
        "form5_away_only": arm(f5[["away_form5_ge%i" % k for k in range(1, 6)]],
                               features_full + [f"away_form5_ge{k}"
                                                for k in range(1, 6)],
                               "f5_away"),
    }

    verdicts = {}
    for W in WINDOWS:
        d = arms[W]["delta"]
        pt, az, lo, hi = (d["delta_candidate_minus_reference"], d["abs_z"],
                          d["ci_lo"], d["ci_hi"])
        if pt > 0 and az > 3:
            verdicts[W] = (f"form{W} CLEARS THE BAR ON 2025-26: +{pt:.6f} "
                           f"nats, |z| {az:.2f} on the {n_test} test rows.")
        elif pt > 0:
            verdicts[W] = (f"form{W} NULL: point {pt:+.6f} nats, CI "
                           f"[{lo:+.6f}, {hi:+.6f}], |z| {az:.2f} -- interval "
                           "includes zero; not promoted.")
        else:
            verdicts[W] = (f"form{W} NULL/NEGATIVE: point {pt:+.6f} nats, CI "
                           f"[{lo:+.6f}, {hi:+.6f}], |z| {az:.2f}.")

    payload = {
        "issue": "60-followon",
        "protocol_ref": "I60b-prereg-recent-form",
        "generated_utc": ms.utc_now(),
        "runtime_seconds": round(time.time() - t0, 1),
        "decision_protocol": ("single last-season holdout: train < 2025-10-21, "
                              "test = 2025-26 season; no whole-frame number"),
        "n_train": int(tr.sum()),
        "n_test": n_test,
        "windows": list(WINDOWS),
        "metrics": {"canonical_26_holdout": ref_m},
        "arms": {str(W): {"log_loss": arms[W]["log_loss"],
                          "metrics": arms[W]["metrics"],
                          "bracket_columns": arms[W]["features"],
                          "delta": arms[W]["delta"]} for W in WINDOWS},
        "positive_control_availability_on_minus_off": d_avail,
        "per_side_bracket_coefficients": {str(W): coef[W] for W in WINDOWS},
        "sensitivity": sens,
        "verdicts": {str(W): verdicts[W] for W in WINDOWS},
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ms.append_jsonl(JSONL, {
        "name": "I60b-run-recent-form",
        "protocol_ref": "I60b-prereg-recent-form",
        "declared_utc": ms.utc_now(),
        "experiment_type": "issue #60 follow-on: recent-form windows (5/10), 2025-26 holdout",
        "n_train": int(tr.sum()),
        "n_test": n_test,
        "metrics": payload["metrics"],
        "arms": payload["arms"],
        "positive_control_availability_on_minus_off": d_avail,
        "sensitivity": sens,
        "verdicts": verdicts,
        "status": "executed",
    })

    L = ["=" * 78,
         "ISSUE 60 FOLLOW-ON -- RECENT-FORM WINDOWS (5 / 10), 2025-26 HOLDOUT",
         "=" * 78]
    a = L.append
    a(f"train {int(tr.sum())} rows (< {OPENER.date()}); test {n_test} rows (2025-26)")
    a("")
    a(f"{'model':<26} {'log loss':>10} {'brier':>9} {'accuracy':>9} {'auc':>7}")
    a(f"{'canonical 26 (holdout)':<26} {ref_m['log_loss']:>10.6f} "
      f"{ref_m['brier']:>9.6f} {ref_m['accuracy']:>9.4f} {ref_m['auc']:>7.4f}")
    for W in WINDOWS:
        m = arms[W]["metrics"]
        a(f"{f'+ form{W} ladders ({26+2*W})':<26} {m['log_loss']:>10.6f} "
          f"{m['brier']:>9.6f} {m['accuracy']:>9.4f} {m['auc']:>7.4f}")
    a("")
    for W in WINDOWS:
        d = arms[W]["delta"]
        a(f"form{W}: delta {d['delta_candidate_minus_reference']:+.6f} "
          f"CI [{d['ci_lo']:+.6f}, {d['ci_hi']:+.6f}] |z| {d['abs_z']:.2f}")
    a(f"positive control (avail on vs off): "
      f"{d_avail['delta_candidate_minus_reference']:+.6f} "
      f"|z| {d_avail['abs_z']:.2f}")
    a("")
    a("per-side bracket coefficients (train fit, descriptive):")
    for W in WINDOWS:
        a(f"  form{W}:")
        for k, v in coef[W].items():
            a(f"    {k:<20} {v:+.4f}")
    a("")
    a("sensitivity (descriptive, cannot promote):")
    for label, row in sens.items():
        a(f"  {label:<18} log loss {row['log_loss']:.6f}  "
          f"delta {row['delta']['delta_candidate_minus_reference']:+.6f}  "
          f"|z| {row['delta']['abs_z']:.2f}")
    a("")
    for W in WINDOWS:
        a(verdicts[W])
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())