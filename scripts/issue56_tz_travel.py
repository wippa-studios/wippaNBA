"""Issue #56: time-zone / travel-quantity features on the canonical logistic.

Tests whether two pre-game schedule-geometry columns help the canonical model:
``tz_offset_diff`` (|standard-time offset| between the two clubs' home cities)
and ``away_tz_crossed`` (zones the away team crosses to this game from its
previous same-season arena, else its home city).  These are the two named
features in issue #56, motivated by Langborg (2023): teams crossing 2+ time
zones win ~41% and spreads fail to centre such games.

WHY THIS IS A SEPARATE RECIPE, NOT A NEW CANONICAL
--------------------------------------------------
The canonical model is pinned to ``DEFAULT_MODELS["full"]`` (26 columns,
0.615663649).  The candidate is ``DEFAULT_MODELS["full_tz"]`` = the canonical 26
PLUS the two columns, in that order (28).  The ``travel_tz`` feature group is
opt-in so nothing about the canonical frame changes; this script only adds a
second model on an extended frame, exactly like the availability and injury
arms did.

SCORING
-------
``measurement.canonical_oof`` fits the reference on the same folds; delta is
``loss(canonical) - loss(tz)``, so **positive means the block is better**.
Promotion requires a positive delta with |z| > 3 on 2,000 paired bootstrap
resamples (seed 42), the project's standing bar.  Pre-registered in
``results/experiments.jsonl`` as ``I56-prereg-tz-travel`` before this module
ran.  The two sensitivity arms (each single column) are descriptive only.

Stage 2 (market-anchored stack) is DECLARED in the pre-registration but only
runs if the model bar clears: a block that dies at the closing line is a
betting null even if the model is better.  If the model bar does not clear,
this run IS the result.

Run::

    OMP_NUM_THREADS=1 python scripts/issue56_tz_travel.py
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

OUT_JSON = ROOT / "results" / "issue56_tz_travel.json"
OUT_MD = ROOT / "results" / "issue56_tz_travel.md"
JSONL = ROOT / "results" / "experiments.jsonl"

#: The recorded post-#36 canonical pooled OOF log loss.
RECORDED_CANONICAL = 0.615663649

N_BOOT = 2000
SEED = 42

#: Descriptive robustness only; cannot promote (see the pre-registration).
SENSITIVITY_DROP: tuple[tuple[str, dict], ...] = (
    ("tz_offset_only", {"away_tz_crossed"}),
    ("crossed_only", {"tz_offset_diff"}),
)

_CORE_METRICS = ("n", "log_loss", "brier", "accuracy", "auc")


def oof_tz(X: pd.DataFrame, y: np.ndarray, games: pd.DataFrame,
           features: list[str], n_splits: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward logistic probabilities on the given feature list, plus the
    scored-row mask.  Same folds and train-only FeaturePipeline as the
    canonical model; row order of ``games`` is load-bearing and never re-sorted.
    """
    y = np.asarray(y, dtype=int)
    p = np.full(len(y), np.nan)
    for tr, te in nd.walk_forward_split(games, n_splits=n_splits):
        # RollingLogreg applies the train-only FeaturePipeline itself, exactly
        # like the canonical model; nothing here touches the test block.
        model = lr.RollingLogreg(C=1.0, features=features).fit(X.iloc[tr], y[tr])
        p[te] = model.predict_proba(X.iloc[te])
    mask = np.isfinite(p)
    return p, mask


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    out = nd.evaluate(y, p)
    return {k: (int(out[k]) if k == "n" else float(out[k]))
            for k in _CORE_METRICS}


def _delta(y: np.ndarray, p_candidate: np.ndarray,
           p_reference: np.ndarray, mask: np.ndarray) -> dict:
    d = ms.delta_stats(y[mask], p_candidate[mask], p_reference[mask],
                       n_boot=N_BOOT, seed=SEED)
    d["bootstrap_seed"] = SEED
    return d


def main() -> int:
    t0 = time.time()
    games = nd.build_games().reset_index(drop=True)
    y = games["y"].to_numpy(dtype=int)
    X_all = lr.load_design(games=games)

    features_full = lr.resolve_features(lr.DEFAULT_MODELS["full"])
    features_tz = lr.resolve_features(lr.DEFAULT_MODELS["full_tz"])
    if X_all.shape[0] != len(games):
        raise AssertionError("design not aligned to games")
    if not features_tz[:26] == features_full:
        raise AssertionError("full_tz must be the canonical 26 PLUS two")

    # Reference: the canonical model via the shared harness (reproduces the
    # recorded number rather than assuming it).
    _, p_ref, mask_ref = ms.canonical_oof(games=games)
    p_tz, mask_tz = oof_tz(X_all, y, games, features_tz)
    if not np.array_equal(mask_ref, mask_tz):
        raise AssertionError("reference and candidate scored different rows")
    mask = mask_ref

    ref_metrics = _metrics(y[mask], p_ref[mask])
    tz_metrics = _metrics(y[mask], p_tz[mask])
    delta = _delta(y, p_tz, p_ref, mask)
    per_fold = ms.per_fold_delta(y, p_tz, p_ref, games)

    sensitivity = {}
    for label, drop in SENSITIVITY_DROP:
        feats = [f for f in features_tz if f not in drop]
        p_s, mask_s = oof_tz(X_all, y, games, feats)
        if not np.array_equal(mask_s, mask):
            raise AssertionError(f"sensitivity {label} scored other rows")
        d = _delta(y, p_s, p_ref, mask)
        sensitivity[label] = {
            "features_dropped": sorted(drop),
            "n_features": len(feats),
            "log_loss": float(nd.evaluate(y[mask], p_s[mask])["log_loss"]),
            "delta": d,
        }

    drift = ref_metrics["log_loss"] - RECORDED_CANONICAL
    reproduced = abs(drift) <= 1e-6
    point = delta["delta_candidate_minus_reference"]
    az = delta["abs_z"]
    ci_lo, ci_hi = delta["ci_lo"], delta["ci_hi"]

    if not reproduced:
        verdict = (
            "FAILED REPRODUCTION: the canonical reference came back "
            f"{ref_metrics['log_loss']:.9f}, not {RECORDED_CANONICAL:.9f} "
            f"(drift {drift:+.2e}). The tz delta is reported for diagnosis "
            "only."
        )
    elif point > 0 and az > 3:
        verdict = (
            "TZ BLOCK CLEARS THE CANONICAL MODEL: the two-column time-zone "
            f"block improves log loss by {point:+.6f} nats on the same "
            f"{int(mask.sum())} walk-forward rows (CI [{ci_lo:+.6f}, "
            f"{ci_hi:+.6f}], |z| {az:.2f}), better in "
            f"{per_fold['n_folds_candidate_better']}/{per_fold['n_folds']} "
            "folds. Per the pre-registration, stage 2 (the market-anchored "
            "stack against the closing line) is now the decision: a block that "
            "dies at the close is a betting null."
        )
    elif point > 0:
        if ci_lo > 0:
            verdict = (
                "NOT PROMOTED (sub-threshold positive): the tz block is "
                f"{point:+.6f} nats better than the canonical model (CI "
                f"[{ci_lo:+.6f}, {ci_hi:+.6f}], |z| {az:.2f}, better in "
                f"{per_fold['n_folds_candidate_better']}/{per_fold['n_folds']} "
                "folds). The interval is entirely above zero but does not "
                "clear |z| > 3; stage 2 is not run."
            )
        else:
            verdict = (
                "NULL: the tz block's point estimate is "
                f"{point:+.6f} nats against the canonical model (CI "
                f"[{ci_lo:+.6f}, {ci_hi:+.6f}], |z| {az:.2f}, better in "
                f"{per_fold['n_folds_candidate_better']}/{per_fold['n_folds']} "
                "folds). The interval includes zero, so this is not evidence "
                "of a difference either way; positive is not promoted, and "
                "stage 2 is not run."
            )
    else:
        verdict = (
            "NULL/NEGATIVE: the tz block does not beat the canonical model. "
            f"Its point estimate is {point:+.6f} nats (CI [{ci_lo:+.6f}, "
            f"{ci_hi:+.6f}], |z| {az:.2f}), better in "
            f"{per_fold['n_folds_candidate_better']}/{per_fold['n_folds']} "
            "folds. "
            + ("The interval is above zero only noise below the bar; it does "
               "not clear |z| > 3." if ci_lo > 0 else
               "The interval includes zero, so this is not evidence of a "
               "difference either way; it is certainly not an improvement.")
        )

    payload = {
        "issue": 56,
        "protocol_ref": "I56-prereg-tz-travel",
        "generated_utc": ms.utc_now(),
        "runtime_seconds": round(time.time() - t0, 1),
        "population": "build_games(), all seasons",
        "n_games": int(len(games)),
        "n_scored": int(mask.sum()),
        "frame_columns": features_tz,
        "tz_columns": features_tz[26:],
        "recorded_canonical_log_loss": RECORDED_CANONICAL,
        "reproduced_canonical_log_loss": ref_metrics["log_loss"],
        "reproduction_drift": drift,
        "reproduced_within_tolerance": bool(reproduced),
        "metrics": {"canonical": ref_metrics, "tz_candidate": tz_metrics},
        "delta_tz_vs_canonical": delta,
        "per_fold": per_fold,
        "sensitivity": sensitivity,
        "stage_2_market_stack": "RUN IF AND ONLY IF the model bar cleared "
                                "(|z| > 3 positive); this run did not run it "
                                "unless the verdict says otherwise.",
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ms.append_jsonl(JSONL, {
        "name": "I56-run-tz-travel",
        "protocol_ref": "I56-prereg-tz-travel",
        "declared_utc": ms.utc_now(),
        "experiment_type": "issue #56: time-zone / travel features (canonical 26 + 2)",
        "n_scored": int(mask.sum()),
        "reproduced_canonical_log_loss": ref_metrics["log_loss"],
        "reproduction_drift": drift,
        "metrics": payload["metrics"],
        "delta_tz_vs_canonical": delta,
        "per_fold": per_fold,
        "sensitivity": sensitivity,
        "verdict": verdict,
        "status": "executed",
    })

    L = ["=" * 78,
         "ISSUE 56 -- TIME-ZONE / TRAVEL FEATURES ON THE CANONICAL FRAME",
         "=" * 78]
    a = L.append
    a(f"games {len(games)}; scored {int(mask.sum())}; canonical columns "
      f"{len(features_full)}; tz columns {features_tz[26:]}")
    a(f"canonical reproduced {ref_metrics['log_loss']:.9f} "
      f"(recorded {RECORDED_CANONICAL:.9f}, drift {drift:+.2e})")
    a("")
    a(f"{'model':<24} {'log loss':>10} {'brier':>9} {'accuracy':>9} {'auc':>7}")
    for name, met in (("canonical (26)", ref_metrics),
                      ("+ tz block (28)", tz_metrics)):
        a(f"{name:<24} {met['log_loss']:>10.6f} {met['brier']:>9.6f} "
          f"{met['accuracy']:>9.4f} {met['auc']:>7.4f}")
    a("")
    a(f"delta (tz - canonical)      {point:+.6f} "
      f"CI [{ci_lo:+.6f}, {ci_hi:+.6f}] |z| {az:.2f}")
    a(f"folds better: {per_fold['n_folds_candidate_better']}/"
      f"{per_fold['n_folds']}")
    a("")
    a("sensitivity (descriptive, cannot promote):")
    for label, row in sensitivity.items():
        a(f"  {label:<18} n={row['n_features']} log loss {row['log_loss']:.6f} "
          f"delta {row['delta']['delta_candidate_minus_reference']:+.6f} "
          f"|z| {row['delta']['abs_z']:.2f}")
    a("")
    a(verdict)
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())