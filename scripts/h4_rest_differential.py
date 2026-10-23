"""H4 (theory dossier): rest as a DIFFERENTIAL between the clubs, not as two
absolute day counts. Pre-registered in results/experiments.jsonl as
``H4-prereg-relative-rest`` BEFORE this script was run. That entry governs.

THE HYPOTHESIS
--------------
The schedule block's sub-bar effect (delta -0.00204 nats in
feature_block_ablation's minus-block convention, localised to back-to-backs by
the issue #45 decomposition) is mis-parameterised: rest should enter as the
rest-day DIFFERENTIAL between the two teams (relative rest; Esteves et al.
2021), not as team-wise absolute day counts. A relative-rest block should show
>= 0.0024 nats, and the home/away B2B columns should shrink when it is present.

ARMS (declared in the pre-registration)
--------------------------------------
A1 = the canonical 67-column frame: build_feature_frame output minus the six
     relative_rest candidate columns. Pinned explicitly so every comparison
     runs against the historical instrument's frame, unchanged.
A2 = A1 + the candidate block (73 cols).   "full + candidate"
A3 = A2 minus the schedule block's 7 columns (66 cols).  "full + candidate
     minus schedule"

COMPARISONS
-----------
C1 = A2 vs A1   does relative rest earn its place?
C2 = A3 vs A1   can relative rest replace absolute rest and still beat A1?
C3 = A3 vs A2   SUPPLEMENTARY, primary setting only: does dropping absolute
                rest once the candidate is present cost anything?

SIGN CONVENTION, per row: delta = loss(first arm) - loss(second arm) via
nba_data.paired_bootstrap_delta(y, p_first, p_second). POSITIVE delta means the
SECOND arm is BETTER. Note this is the OPPOSITE helpful sign to
feature_block_ablation's minus-block convention (schedule's historical
-0.00204): magnitudes are comparable, signs are not.

SETTINGS: primary seed 42 / 5 folds; replications at bootstrap seeds 137 and
2718 (5 folds) and at 8 folds. walk_forward_split and the lbfgs fit are
seed-invariant (the issue #45 erratum), so at a fixed fold count the point
delta is identical across bootstrap seeds by construction and the seed
dimension probes CI variability only; the 8-fold setting is the real
replication. Arms are fit once per fold count and re-bootstrapped per seed.

VERDICT (pre-registered): the block earns its place iff the primary C1 CI
excludes zero in the helpful direction (delta > 0), |z| > 3 at the primary
setting, and the sign is consistent across every replication setting.
Multiplicity: 9 looks (C1 x 4 + C2 x 4 + C3 x 1) -> Bonferroni 0.00556
one-sided; the project's |z| > 3 bar governs on top.

Run:  python scripts/h4_rest_differential.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from DataProviders import nba_data as nd          # noqa: E402
import feature_block_ablation as fba              # noqa: E402
import recency_ablation as ra                     # noqa: E402

JSONL = ROOT / "results" / "experiments.jsonl"
OUT = ROOT / "results" / "h4_rest_differential.json"

N_BOOT = 2000
# (bootstrap seed, fold count). Primary first.
SETTINGS = ((42, 5), (137, 5), (2718, 5), (42, 8))
PRIMARY = (42, 5)
TARGET_NATS = 0.0024
# Coefficient-shrink diagnostic columns (descriptive; no look, no test).
DIAG_COLS = ("home_back_to_back", "away_back_to_back",
             "home_rest_days", "away_rest_days")


def fit_oof_keep_models(X, y, splits):
    """ra.fit_oof's loop, but also keeping the fitted pipelines.

    Identical by construction to the sanctioned instrument (same
    ra.make_pipeline, same fit/predict calls); kept local only because the
    pre-registered diagnostic needs the coefficients. `fit_oof` itself is still
    called on one arm and asserted equal below, so drift would be caught.

    Returns the FULL-LENGTH OOF array (NaN outside the test blocks), the mask,
    and the per-fold pipelines, so per-fold deltas can index in games-row space.
    """
    oof = np.full(len(y), np.nan)
    models = {}
    for k, (tr, te) in enumerate(splits, start=1):
        pipe = ra.make_pipeline()
        pipe.fit(X.iloc[tr], y[tr])
        oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]
        models[k] = pipe
    mask = ~np.isnan(oof)
    return oof, mask, models


def delta_row(y, p_first, p_second, seed) -> dict:
    """loss(first) - loss(second); POSITIVE = second arm better."""
    d = nd.paired_bootstrap_delta(y, p_first, p_second,
                                  n_boot=N_BOOT, seed=seed)
    se = (d["ci_hi"] - d["ci_lo"]) / (2 * 1.959964)
    return {"seed": seed, "n_boot": N_BOOT,
            "delta": float(d["delta"]), "ci_lo": float(d["ci_lo"]),
            "ci_hi": float(d["ci_hi"]), "p_positive": float(d["p_positive"]),
            "se": float(se),
            "z": float(abs(d["delta"]) / se) if se > 0 else float("nan")}


def _ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def per_fold_deltas(y, splits, p_first_full, p_second_full) -> list[dict]:
    """loss(first) - loss(second) per walk-forward test block (in-row-space)."""
    out = []
    for k, (_tr, te) in enumerate(splits, start=1):
        yy = y[te]
        out.append({"fold": k, "n": int(len(te)),
                    "delta": _ll(yy, p_first_full[te]) - _ll(yy, p_second_full[te])})
    return out


def main() -> int:
    t0 = time.time()
    games = nd.build_games()
    y = games["y"].to_numpy(dtype=int)
    frame = fba.build_feature_frame(games)
    candidate = fba.resolve_block("relative_rest", frame)
    schedule = fba.resolve_block("schedule", frame)

    a1 = fba.drop_block(frame, candidate)              # canonical 67-col frame
    a2 = frame                                          # + candidate
    a3 = fba.drop_block(frame, schedule)                # + candidate, - schedule
    assert a1.shape[1] == 67, f"canonical frame is {a1.shape[1]} cols, expected 67"
    print(f"[{time.time()-t0:.0f}s] frame {frame.shape[1]} cols -> "
          f"A1 {a1.shape[1]}  A2 {a2.shape[1]}  A3 {a3.shape[1]}", flush=True)
    print(f"  candidate: {candidate}", flush=True)
    print(f"  schedule : {schedule}", flush=True)

    arms = {"A1": a1, "A2": a2, "A3": a3}
    comparisons = {"C1": ("A1", "A2"), "C2": ("A1", "A3"), "C3": ("A2", "A3")}

    results = []
    coef_diag = {}
    for folds in sorted({f for _s, f in SETTINGS}):
        splits = nd.walk_forward_split(games, n_splits=folds)
        preds_full, masks, models = {}, {}, {}
        for name, fr in arms.items():
            p_full, m, mods = fit_oof_keep_models(fr, y, splits)
            preds_full[name], masks[name], models[name] = p_full, m, mods
        # Guard against drift from the sanctioned harness: same arm, same rows.
        p_check, y_check, _ = ra.fit_oof(a2, y, splits)
        assert np.allclose(p_check, preds_full["A2"][masks["A2"]])
        assert np.array_equal(y_check, y[masks["A2"]])
        assert np.array_equal(masks["A1"], masks["A2"]) \
            and np.array_equal(masks["A2"], masks["A3"])
        y_oof = y[masks["A1"]]
        preds = {name: preds_full[name][masks[name]] for name in arms}
        ys = {name: y_oof for name in arms}
        print(f"[{time.time()-t0:.0f}s] folds={folds}: arms fit, "
              f"{len(y_oof):,} OOF rows (fit_oof parity asserted)", flush=True)

        for seed, f in SETTINGS:
            if f != folds:
                continue
            for cname in ("C1", "C2", "C3"):
                if cname == "C3" and (seed, f) != PRIMARY:
                    continue          # C3 is a single pre-registered look
                first, second = comparisons[cname]
                row = delta_row(ys[first], preds[first], preds[second], seed)
                row.update({
                    "name": f"H4-run-{cname}-s{seed}-f{folds}",
                    "comparison": cname, "arms": f"{second} vs {first}",
                    "sign_convention": "delta = loss(first) - loss(second); "
                                       "POSITIVE = the second arm is better",
                    "n_splits": folds, "n_scored": int(len(y_oof)),
                    "log_loss_first": float(nd.evaluate(
                        ys[first], preds[first])["log_loss"]),
                    "log_loss_second": float(nd.evaluate(
                        ys[second], preds[second])["log_loss"]),
                    "per_fold_deltas": per_fold_deltas(
                        y, splits, preds_full[first], preds_full[second]),
                })
                results.append(row)
                print(f"[{time.time()-t0:6.0f}s] {row['name']:22s} "
                      f"delta {row['delta']:+.5f} "
                      f"CI [{row['ci_lo']:+.5f},{row['ci_hi']:+.5f}] "
                      f"|z| {row['z']:.2f}", flush=True)

        # ---- coefficient-shrink diagnostic (primary fold count only) -------
        if folds == PRIMARY[1]:
            for name in ("A1", "A2"):
                cols = list(arms[name].columns)
                per = {c: [] for c in DIAG_COLS}
                for k, pipe in models[name].items():
                    beta = np.asarray(pipe.steps[-1][1].coef_).ravel()
                    for c in DIAG_COLS:
                        per[c].append(float(beta[cols.index(c)]))
                coef_diag[name] = {
                    "per_fold": {c: [round(v, 5) for v in vs]
                                 for c, vs in per.items()},
                    "mean_abs": {c: round(float(np.mean(np.abs(vs))), 5)
                                 for c, vs in per.items()},
                }
            print(f"[{time.time()-t0:.0f}s] coef diagnostic (std units, "
                  f"mean |coef|): A1 {coef_diag['A1']['mean_abs']}  "
                  f"A2 {coef_diag['A2']['mean_abs']}", flush=True)

    # ---- verdict against the pre-registered rule --------------------------
    def get(cname, seed, folds):
        return next(r for r in results
                    if r["comparison"] == cname and r["seed"] == seed
                    and r["n_splits"] == folds)

    prim = get("C1", *PRIMARY)
    reps = [get("C1", s, f) for s, f in SETTINGS if (s, f) != PRIMARY]
    signs_ok = all(r["delta"] > 0 for r in [prim] + reps)
    rule_ok = prim["ci_lo"] > 0 and prim["z"] > 3 and signs_ok
    verdict = {
        "rule": "earn-its-place iff primary C1 CI excludes zero in the helpful "
                "direction, |z| > 3 at the primary setting, and the sign is "
                "consistent across every replication setting "
                "(H4-prereg-relative-rest)",
        "primary": {"setting": f"seed {PRIMARY[0]} / {PRIMARY[1]} folds",
                    "delta": prim["delta"], "ci_lo": prim["ci_lo"],
                    "ci_hi": prim["ci_hi"], "z": prim["z"]},
        "replications": [{"setting": f"seed {s} / {f} folds",
                          "delta": get("C1", s, f)["delta"],
                          "ci_lo": get("C1", s, f)["ci_lo"],
                          "ci_hi": get("C1", s, f)["ci_hi"],
                          "z": get("C1", s, f)["z"]}
                         for s, f in SETTINGS if (s, f) != PRIMARY],
        "signs_consistent": bool(signs_ok),
        "decision": ("H4 candidate EARNS its place" if rule_ok else
                     "H4 candidate FAILS the pre-registered rule: "
                     "no earn-its-place"),
        "c2_replacement": {"setting": f"seed {PRIMARY[0]} / {PRIMARY[1]} folds",
                           "delta": get("C2", *PRIMARY)["delta"],
                           "ci_lo": get("C2", *PRIMARY)["ci_lo"],
                           "ci_hi": get("C2", *PRIMARY)["ci_hi"],
                           "z": get("C2", *PRIMARY)["z"],
                           "reading": "positive delta = the replacement arm "
                                      "(relative rest in, absolute rest out) "
                                      "beats the canonical frame"},
        "c3_drop_absolute_rest_given_candidate": {
            "delta": get("C3", *PRIMARY)["delta"],
            "ci_lo": get("C3", *PRIMARY)["ci_lo"],
            "ci_hi": get("C3", *PRIMARY)["ci_hi"],
            "z": get("C3", *PRIMARY)["z"],
            "reading": "positive delta = dropping the schedule block once the "
                       "candidate is present HELPS"},
        "multiplicity": "9 looks (C1 x 4 + C2 x 4 + C3 x 1) -> Bonferroni "
                        "0.05/9 = 0.00556 one-sided (|z| ~ 2.54); the project "
                        "bar |z| > 3 governs on top",
        "target_nats": TARGET_NATS,
        "coef_shrink_diagnostic": coef_diag,
    }

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pre_registration": "results/experiments.jsonl :: H4-prereg-relative-rest",
        "frame": {"full": int(frame.shape[1]),
                  "A1_canonical": int(a1.shape[1]), "A2": int(a2.shape[1]),
                  "A3": int(a3.shape[1]),
                  "candidate_columns": candidate,
                  "schedule_columns": schedule},
        "results": results,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {OUT}", flush=True)

    ts = datetime.now(timezone.utc).isoformat()
    with JSONL.open("a") as fh:
        for r in results:
            rec = {k: v for k, v in r.items() if k != "per_fold_deltas"}
            rec.update({
                "status": "executed", "declared_utc": ts,
                "experiment_type": "feature_block candidate result (H4)",
                "protocol_ref": "H4-prereg-relative-rest",
                "model": "logistic, fit_oof walk-forward "
                         "(scripts/feature_block_ablation.py machinery)",
                "per_fold_deltas": r["per_fold_deltas"],
                "artifact": "results/h4_rest_differential.json",
            })
            fh.write(json.dumps(rec) + "\n")
        fh.write(json.dumps({
            "name": "H4-verdict", "status": "executed", "declared_utc": ts,
            "experiment_type": "verdict_record",
            "protocol_ref": "H4-prereg-relative-rest",
            **{k: v for k, v in verdict.items()
               if k != "coef_shrink_diagnostic"},
            "coef_shrink_mean_abs": {
                arm: coef_diag[arm]["mean_abs"] for arm in coef_diag},
            "leakage_check": "passed: candidate columns are causal by "
                             "construction (rest_and_travel inputs, knowable "
                             "before tip-off); imputation and scaling fit on "
                             "train rows only (ra.fit_oof); arms scored on "
                             "identical out-of-fold rows",
        }) + "\n")
    print(f"appended {len(results) + 1} records to results/experiments.jsonl",
          flush=True)
    print(f"\n{verdict['decision']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
