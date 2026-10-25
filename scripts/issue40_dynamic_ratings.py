"""Issue #40: dynamic team-strength ratings, scored walk-forward.

Pre-registered in ``results/experiments.jsonl`` as
``I40-prereg-dynamic-ratings`` before this script ran.

TWO MODELS, ONE STATE-EVOLUTION IDEA
------------------------------------
* ``glicko`` -- per-team rating and rating deviation, updated game by game.
  Inactivity widens the RD; games narrow it.  The RD enters the prediction, so
  an uncertain team is shrunk toward a coin flip.
* ``state_space_bt`` -- latent Bradley-Terry strength with a random-walk prior
  and an extended Kalman filter.  The state variance is the uncertainty term.

Both carry a **fixed** 70-Elo-point home edge outside the team rating.  No
parameter is tuned on a scored row.  The bar is the same-row canonical logistic
model; the recorded 0.6190 is also reported.

Run:  python scripts/issue40_dynamic_ratings.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from DataProviders import nba_data as nd          # noqa: E402
from Models import dynamic_ratings as dr          # noqa: E402
from Models import measurement as ms              # noqa: E402

JSONL = ROOT / "results" / "experiments.jsonl"
OUT_JSON = ROOT / "results" / "issue40_dynamic_ratings.json"
OUT_MD = ROOT / "results" / "issue40_dynamic_ratings.md"
N_BOOT = 2000
SEED = 42
RECORDED_CANONICAL = 0.6190
RECORDED_E14 = 0.6183071193165273


def metrics(y, p) -> dict:
    m = nd.evaluate(np.asarray(y, dtype=int), np.asarray(p, dtype=float))
    return {k: (int(v) if k == "n" else float(v))
            for k, v in m.items() if k in ("n", "log_loss", "brier",
                                           "accuracy", "auc")}


def summary(name: str, p: np.ndarray, mask: np.ndarray, y: np.ndarray,
            reference: np.ndarray, games) -> dict:
    yy = y[mask]
    pp = p[mask]
    return {
        "model": name,
        "metrics": metrics(yy, pp),
        "delta_vs_reference": ms.delta_stats(
            yy, pp, reference[mask], n_boot=N_BOOT, seed=SEED),
        "per_fold_vs_reference": ms.per_fold_delta(y, p, reference, games),
    }


def main() -> int:
    t0 = time.time()
    games = nd.build_games()
    y = games["y"].to_numpy(dtype=int)

    _, p_e14, mask_e14 = ms.oof_e14(games)
    _, p_can, _ = ms.canonical_oof(games)
    p_e14 = np.asarray(p_e14, dtype=float)
    p_can = np.asarray(p_can, dtype=float)
    if not np.isfinite(p_can[mask_e14]).all():
        raise AssertionError("canonical OOF does not cover the E14 scored rows")

    print(f"running Glicko over {len(games)} games ...")
    glicko = dr.fit_predict_glicko(games)
    print(f"running state-space BT over {len(games)} games ...")
    ss = dr.fit_predict_state_space_bt(games)

    predictions = {"glicko": glicko["p"], "state_space_bt": ss["p"]}
    rows: dict[str, dict] = {}
    for name, p in predictions.items():
        if not np.isfinite(p[mask_e14]).all():
            raise AssertionError(f"{name} is missing a scored row")
        rows[name] = summary(name, p, mask_e14, y, p_can, games)
        rows[name]["uncertainty"] = (
            {"rd_home_mean": float(np.nanmean(glicko["rd_home"])),
             "rd_away_mean": float(np.nanmean(glicko["rd_away"])),
             "rd_home_median": float(np.nanmedian(glicko["rd_home"])),
             "rd_min": float(np.nanmin(glicko["rd_home"])),
             "rd_max": float(np.nanmax(glicko["rd_home"]))}
            if name == "glicko" else
            {"var_home_mean": float(np.nanmean(ss["var_home"])),
             "var_away_mean": float(np.nanmean(ss["var_away"])),
             "var_home_median": float(np.nanmedian(ss["var_home"])),
             "var_min": float(np.nanmin(ss["var_home"])),
             "var_max": float(np.nanmax(ss["var_home"]))})
        rows[name]["mean_prediction"] = float(np.mean(np.asarray(p)[mask_e14]))
        rows[name]["base_rate"] = float(np.mean(y[mask_e14]))

    # E14 on the same rows, for the second comparison the issue asks for.
    e14_metrics = metrics(y[mask_e14], p_e14[mask_e14])
    can_metrics = metrics(y[mask_e14], p_can[mask_e14])
    d_vs_e14 = {
        name: ms.delta_stats(y[mask_e14], p[mask_e14], p_e14[mask_e14],
                             n_boot=N_BOOT, seed=SEED)
        for name, p in predictions.items()
    }

    verdicts = {}
    for name, row in rows.items():
        d = row["delta_vs_reference"]
        verdicts[name] = (
            "PROMOTED" if d["delta_candidate_minus_reference"] > 0
            and d["abs_z"] > 3 else
            f"NULL: {name} does not beat the same-row canonical model at "
            f"|z| > 3 (delta "
            f"{d['delta_candidate_minus_reference']:+.6f}, |z| {d['abs_z']:.2f}).")

    payload = {
        "issue": 40,
        "protocol_ref": "I40-prereg-dynamic-ratings",
        "generated_utc": ms.utc_now(),
        "runtime_seconds": round(time.time() - t0, 1),
        "n_scored": int(mask_e14.sum()),
        "reference": {
            "canonical_same_rows": can_metrics,
            "e14_same_rows": e14_metrics,
            "recorded_canonical": RECORDED_CANONICAL,
            "recorded_e14": RECORDED_E14,
        },
        "models": rows,
        "delta_vs_e14": d_vs_e14,
        "configs": {
            "glicko": glicko["config"].__dict__,
            "state_space_bt": {k: v for k, v in ss["config"].__dict__.items()},
        },
        "verdicts": verdicts,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    L: list[str] = []
    a = L.append
    a("=" * 78)
    a("ISSUE 40 -- DYNAMIC RATINGS, WALK-FORWARD")
    a("=" * 78)
    a(f"n scored {int(mask_e14.sum())}; home edge fixed at 70 Elo points, "
      "never learned into team strength")
    a("")
    a(f"{'model':<16} {'logloss':>9} {'brier':>8} {'auc':>7} {'acc':>7} "
      f"{'mean p':>7}")
    a(f"{'canonical':<16} {can_metrics['log_loss']:>9.6f} "
      f"{can_metrics['brier']:>8.6f} {can_metrics['auc']:>7.4f} "
      f"{can_metrics['accuracy']:>7.4f} "
      f"{float(np.mean(p_can[mask_e14])):>7.4f}")
    a(f"{'e14':<16} {e14_metrics['log_loss']:>9.6f} "
      f"{e14_metrics['brier']:>8.6f} {e14_metrics['auc']:>7.4f} "
      f"{e14_metrics['accuracy']:>7.4f} "
      f"{float(np.mean(p_e14[mask_e14])):>7.4f}")
    for name, row in rows.items():
        m = row["metrics"]
        a(f"{name:<16} {m['log_loss']:>9.6f} {m['brier']:>8.6f} "
          f"{m['auc']:>7.4f} {m['accuracy']:>7.4f} "
          f"{row['mean_prediction']:>7.4f}")
    a("")
    a("DELTAS (positive = dynamic model better)")
    for name, row in rows.items():
        d = row["delta_vs_reference"]
        de = d_vs_e14[name]
        a(f"{name:<16} vs canonical {d['delta_candidate_minus_reference']:+.6f} "
          f"[{d['ci_lo']:+.6f},{d['ci_hi']:+.6f}] |z| {d['abs_z']:.2f}; "
          f"vs E14 {de['delta_candidate_minus_reference']:+.6f} "
          f"[{de['ci_lo']:+.6f},{de['ci_hi']:+.6f}] |z| {de['abs_z']:.2f}")
        a(f"{'':<16} folds better than canonical: "
          f"{row['per_fold_vs_reference']['n_folds_candidate_better']}"
          f"/{row['per_fold_vs_reference']['n_folds']}")
    a("")
    a("UNCERTAINTY TERM")
    for name, row in rows.items():
        a(f"{name:<16} {row['uncertainty']}")
    a("")
    for name, v in verdicts.items():
        a(f"verdict {name}: {v}")
    text = "\n".join(L)
    print(text)
    OUT_MD.write_text(text + "\n")

    ms.append_jsonl(JSONL, {
        "name": "I40-run-dynamic-ratings",
        "status": "executed",
        "declared_utc": ms.utc_now(),
        "experiment_type": "dynamic rating results (issue #40)",
        "protocol_ref": "I40-prereg-dynamic-ratings",
        "n_scored": int(mask_e14.sum()),
        "models": {name: {"metrics": row["metrics"],
                          "delta_vs_canonical": row["delta_vs_reference"],
                          "delta_vs_e14": d_vs_e14[name],
                          "uncertainty": row["uncertainty"]}
                   for name, row in rows.items()},
        "canonical_same_rows": can_metrics,
        "e14_same_rows": e14_metrics,
        "verdicts": verdicts,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
