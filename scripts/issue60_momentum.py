"""Issue #60 (revised): momentum streak brackets, decided on 2025-26 only.

The question is whether the market over-reacts to momentum: a team entering
tonight's game on a winning streak is overbet (hot), and a losing streak
underbet (cold), so pre-game streak information should add signal.  The
DECISION is a single last-season holdout, per the issue thread: a feature
that does not work on the 2025-26 season is not viable, regardless of what
the other eleven seasons say, so **no whole-frame walk-forward number is
computed or reported**.

WHY BRACKETS, NOT A COUNT
-------------------------
The momentum claim is threshold-shaped (Byrnes & Farinella's 56.5% is a
>=4-streak claim).  A linear streak-count would force one slope over a
relationship that switches on at a threshold; cumulative indicators
(>=2/3/4/6) let the model treat each threshold separately.  Eight level
columns (home and away), matching the schedule group's "a side's own state
is a level fact" rule.

THE HOLDOUT
-----------
train = every build_games() row with GAME_DATE < 2025-10-21 (11,974 rows);
test  = the 1,225 rows of the 2025-26 season.  BOTH models are fit on the
train rows only and predict the same test rows, with the canonical train-only
FeaturePipeline.  delta = loss(reference) - loss(candidate) on the test rows;
promotion requires delta > 0 with |z| > 3 on 2,000 paired bootstrap resamples
(seed 42).  A positive control (the proven availability block vs its absence
on the same split) calibrates how much a 1,225-game holdout can resolve.

Pre-registered in ``results/experiments.jsonl`` as
``I60-prereg-momentum-streaks`` before this module ran.

Run::

    OMP_NUM_THREADS=1 python scripts/issue60_momentum.py
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

OUT_JSON = ROOT / "results" / "issue60_momentum.json"
OUT_MD = ROOT / "results" / "issue60_momentum.md"
JSONL = ROOT / "results" / "experiments.jsonl"

#: The 2025-26 season opener: the holdout boundary, fixed ex ante.
OPENER = pd.Timestamp("2025-10-21")
N_BOOT = 2000
SEED = 42

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
    """Canonical-contract logistic on the train block, predictions on te."""
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
    mom = lr.momentum_brackets(
        nd.streak_features(games).reset_index(drop=True))
    X_cand = pd.concat([X_all, mom], axis=1).reset_index(drop=True)

    features_full = lr.resolve_features(lr.DEFAULT_MODELS["full"])
    features_mom = lr.resolve_features(lr.DEFAULT_MODELS["full_momentum"])
    if not (len(features_full) == 26 and len(features_mom) == 34
            and features_mom[:26] == features_full):
        raise AssertionError("full_momentum must be canonical 26 PLUS 8")

    # Reference and candidate on the same train/test rows.
    p_ref = fit_predict(X_all, y, tr, te, features_full)
    p_cand = fit_predict(X_cand, y, tr, te, features_mom)

    ref_m = metrics(y_te, p_ref)
    cand_m = metrics(y_te, p_cand)
    d = delta(y_te, p_cand, p_ref)

    # Positive control: canonical 26 (availability ON) vs same minus the four
    # availability columns (22), same split.  Calibrates resolvability.
    avail_cols = lr.resolve_features("availability")
    feats_noavail = [f for f in features_full if f not in avail_cols]
    p_noavail = fit_predict(X_all, y, tr, te, feats_noavail)
    d_avail = delta(y_te, p_ref, p_noavail)   # ON minus OFF
    avail_m = metrics(y_te, p_noavail)

    # Per-bracket coefficients, read descriptively from the train fit.
    m_cand = lr.RollingLogreg(C=1.0, features=features_mom).fit(
        X_cand.iloc[tr], y_tr)
    coef = m_cand.coefficients()["coef"]

    # Sensitivity arms (descriptive; cannot promote).
    def arm(extra_cols: pd.DataFrame, label: str) -> dict:
        Xa = pd.concat([X_all, extra_cols], axis=1).reset_index(drop=True)
        feats = features_full + list(extra_cols.columns)
        p_a = fit_predict(Xa, y, tr, te, feats)
        da = delta(y_te, p_a, p_ref)
        return {"log_loss": float(nd.evaluate(y_te, p_a)["log_loss"]),
                "delta": da}

    sens = {
        "home_only": arm(mom[["home_streak_ge2", "home_streak_ge3",
                              "home_streak_ge4", "home_streak_ge6"]],
                          "home_only"),
        "away_only": arm(mom[["away_streak_ge2", "away_streak_ge3",
                              "away_streak_ge4", "away_streak_ge6"]],
                          "away_only"),
        "loss_streaks": arm(lr.momentum_loss_brackets(
            nd.streak_features(games).reset_index(drop=True)), "loss_streaks"),
    }

    point, az, ci_lo, ci_hi = (d["delta_candidate_minus_reference"],
                               d["abs_z"], d["ci_lo"], d["ci_hi"])
    if point > 0 and az > 3:
        verdict = (
            "MOMENTUM CLEARS THE BAR ON 2025-26: the streak brackets improve "
            f"log loss by {point:+.6f} nats on the {n_test} last-season test "
            f"rows (CI [{ci_lo:+.6f}, {ci_hi:+.6f}], |z| {az:.2f}). The "
            "market does over-react to streaks at this threshold; promotion is "
            "the candidate model on the season holdout."
        )
    elif point > 0 and ci_lo > 0:
        verdict = (
            "NOT PROMOTED on 2025-26 (sub-threshold positive): the streak "
            f"brackets are {point:+.6f} nats better than the canonical model "
            f"on the {n_test} test rows (CI [{ci_lo:+.6f}, {ci_hi:+.6f}], "
            f"|z| {az:.2f}). The interval is above zero but below the |z| > 3 "
            "bar; positive is not promoted."
        )
    elif point > 0:
        verdict = (
            "NULL on 2025-26: the streak brackets' point estimate is "
            f"{point:+.6f} nats against the canonical model (CI [{ci_lo:+.6f}, "
            f"{ci_hi:+.6f}], |z| {az:.2f}). The interval includes zero, so "
            "there is no evidence the brackets help on the season that would "
            "actually be bet; positive is not promoted."
        )
    else:
        if ci_hi < 0:
            tail = ("The interval lies entirely below zero: the brackets "
                    "actively hurt on the season that would actually be bet.")
        else:
            tail = ("The interval includes zero, so there is no evidence the "
                    "brackets help on the season that would actually be bet.")
        verdict = (
            "NULL/NEGATIVE on 2025-26: the streak brackets do not beat the "
            f"canonical model; point estimate {point:+.6f} nats "
            f"(CI [{ci_lo:+.6f}, {ci_hi:+.6f}], |z| {az:.2f}). {tail}"
        )

    payload = {
        "issue": 60,
        "protocol_ref": "I60-prereg-momentum-streaks",
        "generated_utc": ms.utc_now(),
        "runtime_seconds": round(time.time() - t0, 1),
        "decision_protocol": "single last-season holdout: train < 2025-10-21, "
                             "test = 2025-26 season; no whole-frame number",
        "n_train": int(tr.sum()),
        "n_test": n_test,
        "brackets": list(lr.MOMENTUM_BRACKETS),
        "momentum_columns": features_mom[26:],
        "metrics": {"canonical_26_holdout": ref_m,
                    "plus_momentum_34_holdout": cand_m,
                    "availability_off_22_holdout": avail_m},
        "delta_momentum_vs_canonical": d,
        "positive_control_availability_on_minus_off": d_avail,
        "per_bracket_coefficients": {
            k: float(v) for k, v in coef.items()
            if k.startswith(("home_streak", "away_streak"))
        },
        "sensitivity": sens,
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ms.append_jsonl(JSONL, {
        "name": "I60-run-momentum-streaks",
        "protocol_ref": "I60-prereg-momentum-streaks",
        "declared_utc": ms.utc_now(),
        "experiment_type": "issue #60: momentum streak brackets, 2025-26 holdout",
        "n_train": int(tr.sum()),
        "n_test": n_test,
        "metrics": payload["metrics"],
        "delta_momentum_vs_canonical": d,
        "positive_control_availability_on_minus_off": d_avail,
        "sensitivity": sens,
        "verdict": verdict,
        "status": "executed",
    })

    L = ["=" * 78,
         "ISSUE 60 -- MOMENTUM STREAK BRACKETS, 2025-26 HOLDOUT",
         "=" * 78]
    a = L.append
    a(f"train {int(tr.sum())} rows (< {OPENER.date()}); test {n_test} rows (2025-26)")
    a(f"brackets {list(lr.MOMENTUM_BRACKETS)}; columns "
      f"{features_full[26:] or 'canonical 26'} + 8 momentum")
    a("")
    a(f"{'model':<28} {'log loss':>10} {'brier':>9} {'accuracy':>9} {'auc':>7}")
    for name, m in (("canonical 26 (holdout)", ref_m),
                    ("+ momentum 34 (holdout)", cand_m),
                    ("availability off 22 (control)", avail_m)):
        a(f"{name:<28} {m['log_loss']:>10.6f} {m['brier']:>9.6f} "
          f"{m['accuracy']:>9.4f} {m['auc']:>7.4f}")
    a("")
    a(f"delta (momentum - canonical)  {point:+.6f} "
      f"CI [{ci_lo:+.6f}, {ci_hi:+.6f}] |z| {az:.2f}")
    a(f"positive control (avail on vs off): "
      f"{d_avail['delta_candidate_minus_reference']:+.6f} "
      f"|z| {d_avail['abs_z']:.2f}")
    a("")
    a("per-bracket coefficients (train fit, descriptive):")
    for k, v in payload["per_bracket_coefficients"].items():
        a(f"  {k:<22} {v:+.4f}")
    a("")
    a("sensitivity (descriptive, cannot promote):")
    for label, row in sens.items():
        a(f"  {label:<12} log loss {row['log_loss']:.6f}  "
          f"delta {row['delta']['delta_candidate_minus_reference']:+.6f}  "
          f"|z| {row['delta']['abs_z']:.2f}")
    a("")
    a(verdict)
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())