"""Issue #70: meta-replication of the recent nulls on the 2024-25 holdout.

Every recent candidate was decided on 2025-26 alone: momentum streak brackets
(+0.000260, |z| 0.29), the five- and ten-game recent-form ladders (-0.000620,
|z| 0.64; -0.000764, |z| 0.63), and the time-zone pair (+0.000035, |z| 0.28
full-frame in #56).  A null on one season is a fact about that season as much
as about the frame.  This battery asks the stability question: are the same
arms null on **2024-25** under the identical train/test protocol?

DECISION (pre-registered I70-prereg-null-meta-replication): train = everything
before 2024-10-22 (10,749 rows); test = the 1,225 games of 2024-25.  Reference
= the canonical 26 fit on train.  Four candidate arms, each the canonical 26
plus its recorded block.  Each arm is read at |z| > 3 on 2,000 paired bootstrap
resamples (seed 42); no arm is promoted by this battery -- a clearing arm is
season-dependent evidence and triggers a follow-up combined test.  Positive
control: availability on vs off on the same split; if it does not clear, every
null is "underpowered", not evidence.

Run::

    OMP_NUM_THREADS=1 python scripts/issue70_null_replication.py
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

OUT_JSON = ROOT / "results" / "issue70_null_replication.json"
OUT_MD = ROOT / "results" / "issue70_null_replication.md"
JSONL = ROOT / "results" / "experiments.jsonl"

OPENER = pd.Timestamp("2024-10-22")
OPENER_END = pd.Timestamp("2025-10-21")
N_BOOT = 2000
SEED = 42
RECORDED_CANONICAL = 0.615663649
DRIFT_TOLERANCE = 1e-6

#: Recorded 2025-26 (or #56 full-frame) readings, quoted from their artifacts.
PRIOR = {
    "momentum": {"delta": 0.000260, "abs_z": 0.29, "season": "2025-26",
                 "source": "results/issue60_momentum.json"},
    "form5": {"delta": -0.000620, "abs_z": 0.64, "season": "2025-26",
              "source": "results/issue60b_recent_form.json"},
    "form10": {"delta": -0.000764, "abs_z": 0.63, "season": "2025-26",
               "source": "results/issue60b_recent_form.json"},
    "tz": {"delta": 0.000035, "abs_z": 0.28, "season": "full frame (#56)",
           "source": "results/issue56_tz_travel.json"},
}

_CORE = ("n", "log_loss", "brier", "accuracy", "auc")


def metrics(y, p):
    out = nd.evaluate(y, p)
    return {k: (int(out[k]) if k == "n" else float(out[k])) for k in _CORE
            if k in out}


def delta(y, p_c, p_r):
    d = ms.delta_stats(y, p_c, p_r, n_boot=N_BOOT, seed=SEED)
    d["bootstrap_seed"] = SEED
    return d


def fit_predict(X, y, tr, te, features):
    m = lr.RollingLogreg(C=1.0, features=features).fit(X.iloc[tr], y[tr])
    return m.predict_proba(X.iloc[te])


def main() -> int:
    t0 = time.time()
    games = nd.build_games().reset_index(drop=True)
    dates = pd.to_datetime(games["GAME_DATE"])
    tr = (dates < OPENER).to_numpy()
    te = ((dates >= OPENER) & (dates < OPENER_END)).to_numpy()
    y = games["y"].to_numpy(dtype=int)
    y_tr, y_te = y[tr], y[te]
    n_test = int(te.sum())
    if n_test != 1225:
        raise AssertionError(f"expected 1,225 test rows, got {n_test}")

    X_all = lr.load_design(games=games).reset_index(drop=True)
    features_full = lr.resolve_features(lr.DEFAULT_MODELS["full"])
    if len(features_full) != 26:
        raise AssertionError("canonical must be 26 columns")

    # Reproduction anchor: the recorded canonical walk-forward bar.
    _, p_can, mask_can = ms.canonical_oof(games=games)
    reproduced = float(ms.log_loss(y[mask_can], p_can[mask_can]))
    drift = reproduced - RECORDED_CANONICAL
    reproduced_ok = abs(drift) <= DRIFT_TOLERANCE
    print(f"canonical reproduced {reproduced:.9f} "
          f"(recorded {RECORDED_CANONICAL:.9f}, drift {drift:+.2e}, "
          f"ok={reproduced_ok})")

    # Reference on the holdout.
    p_ref = fit_predict(X_all, y, tr, te, features_full)
    ref_m = metrics(y_te, p_ref)

    # Four pre-registered arms.
    blocks: dict[str, tuple[pd.DataFrame, list[str]]] = {}

    mom = lr.momentum_brackets(
        nd.streak_features(games).reset_index(drop=True))
    blocks["momentum"] = (mom,
                          lr.resolve_features(lr.DEFAULT_MODELS["full_momentum"]))

    rf = nd.recent_form_features(games).reset_index(drop=True)
    for w in (5, 10):
        ladders = lr.form_brackets(rf, w)
        blocks[f"form{w}"] = (
            ladders, lr.resolve_features(lr.DEFAULT_MODELS[f"full_form{w}"]))

    # The tz columns already live in load_design; the arm selects them.
    blocks["tz"] = (None,
                    lr.resolve_features(lr.DEFAULT_MODELS["full_tz"]))

    arms = {}
    for name, (block, feats) in blocks.items():
        if block is None:
            Xc = X_all
        else:
            Xc = pd.concat([X_all, block], axis=1).reset_index(drop=True)
        if not feats[:26] == features_full:
            raise AssertionError(f"{name} must be canonical 26 PLUS a block")
        p = fit_predict(Xc, y, tr, te, feats)
        arms[name] = {
            "features_added": len(feats) - 26,
            "log_loss": float(nd.evaluate(y_te, p)["log_loss"]),
            "metrics": metrics(y_te, p),
            "delta": delta(y_te, p, p_ref),
        }

    # Positive control: availability ON vs OFF on the same split.
    avail_cols = lr.resolve_features("availability")
    p_noavail = fit_predict(
        X_all, y, tr, te, [f for f in features_full if f not in avail_cols])
    d_ctrl = delta(y_te, p_ref, p_noavail)

    control_ok = bool(d_ctrl["delta_candidate_minus_reference"] > 0
                      and d_ctrl["abs_z"] > 3)

    verdicts = {}
    for name, arm in arms.items():
        d = arm["delta"]
        point, az = d["delta_candidate_minus_reference"], d["abs_z"]
        if point > 0 and az > 3:
            verdicts[name] = (
                f"SEASON-DEPENDENT: {name} reads {point:+.6f} nats, |z| "
                f"{az:.2f} on 2024-25 -- it clears the bar on a season where "
                f"the recorded result ({PRIOR[name]['delta']:+.6f}, |z| "
                f"{PRIOR[name]['abs_z']:.2f}) was null. Not promoted by this "
                "battery; a combined two-season test is the follow-up.")
        else:
            qualifier = ("replicates" if control_ok else "underpowered")
            verdicts[name] = (
                f"NULL ({qualifier}): {name} reads {point:+.6f} nats, |z| "
                f"{az:.2f} on 2024-25, consistent with the recorded "
                f"{PRIOR[name]['delta']:+.6f} (|z| {PRIOR[name]['abs_z']:.2f}).")

    clearing = [n for n in arms
                if arms[n]["delta"]["abs_z"] > 3
                and arms[n]["delta"]["delta_candidate_minus_reference"] > 0]
    if not control_ok:
        meta = ("INCONCLUSIVE: the availability positive control did not clear "
                "(|z| > 3) on 2024-25, so this season cannot resolve an "
                "availability-sized effect. The nulls above are underpowered, "
                "not evidence of stability.")
    elif clearing:
        meta = ("SEASON-DEPENDENT: " + ", ".join(clearing) + " clear(s) on "
                "2024-25 where the recorded season was null. The two seasons "
                "disagree; the family needs a combined two-season test before "
                "any reading of 'nothing works'.")
    else:
        meta = ("STABLE: all four arms are null on 2024-25 and on the recorded "
                "season, with the availability positive control resolving on "
                "both. The recent nulls describe the frame, not one season.")

    payload = {
        "issue": 70,
        "protocol_ref": "I70-prereg-null-meta-replication",
        "generated_utc": ms.utc_now(),
        "runtime_seconds": round(time.time() - t0, 1),
        "n_train": int(tr.sum()),
        "n_test": n_test,
        "reproduction": {
            "canonical_log_loss": reproduced,
            "recorded": RECORDED_CANONICAL,
            "drift": drift,
            "within_tolerance": bool(reproduced_ok),
            "n_scored": int(mask_can.sum()),
        },
        "reference": {"log_loss": ref_m["log_loss"], "metrics": ref_m},
        "arms": arms,
        "prior_2025_26": PRIOR,
        "positive_control_availability_on_minus_off": d_ctrl,
        "control_clears": control_ok,
        "verdicts": verdicts,
        "verdict": meta,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ms.append_jsonl(JSONL, {
        "name": "I70-run-null-meta-replication",
        "protocol_ref": "I70-prereg-null-meta-replication",
        "declared_utc": ms.utc_now(),
        "experiment_type": ("issue #70: momentum/form/tz nulls, 2024-25 "
                            "holdout"),
        "n_train": int(tr.sum()),
        "n_test": n_test,
        "reproduction": payload["reproduction"],
        "reference": payload["reference"],
        "arms": arms,
        "positive_control": d_ctrl,
        "verdicts": verdicts,
        "verdict": meta,
        "status": "executed",
    })

    L = ["=" * 78,
         "ISSUE 70 -- NULL META-REPLICATION, 2024-25 HOLDOUT",
         "=" * 78]
    a = L.append
    a(f"train {int(tr.sum())} rows (< {OPENER.date()}); test {n_test} rows "
      "(2024-25)")
    a(f"canonical reproduction {reproduced:.9f} (recorded "
      f"{RECORDED_CANONICAL:.9f}, drift {drift:+.2e}, ok={reproduced_ok})")
    a(f"reference (canonical 26) log loss {ref_m['log_loss']:.6f}")
    a("")
    for name, arm in arms.items():
        d = arm["delta"]
        a(f"{name:<9} +{arm['features_added']:>2} cols  "
          f"log loss {arm['log_loss']:.6f}  "
          f"delta {d['delta_candidate_minus_reference']:+.6f} "
          f"CI [{d['ci_lo']:+.6f}, {d['ci_hi']:+.6f}] |z| {d['abs_z']:.2f}")
    a("")
    a(f"availability control (on - off) "
      f"{d_ctrl['delta_candidate_minus_reference']:+.6f} "
      f"|z| {d_ctrl['abs_z']:.2f}  clears={control_ok}")
    a("")
    for name in arms:
        a(verdicts[name])
    a("")
    a(meta)
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
