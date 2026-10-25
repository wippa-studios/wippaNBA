"""Issue #39: true RAPM on the 2020-21 stint data, walk-forward.

Pre-registered in ``results/experiments.jsonl`` as ``I39-prereg-true-rapm``
before this script ran.  That record governs the arms, the alpha grid, the
calibration rule and the expected roster.

WHAT IS FITTED
--------------
:mod:`src.Models.rapm` builds one estimating equation per player-stint::

    signed PT_DIFF_j = sum_i (overlap seconds / 28800) * theta_i + home * hca

and solves the duration-weighted ridge normal equations.  The response is the
recorded plus-minus, so no box-score proxy enters the fit.  Predictions use the
expected roster and expected minutes from :mod:`src.Models.player_ratings`;
actual participants are the label, never an input.

SCORING
-------
Five walk-forward splits inside 2020-21.  The model is fitted on the training
games' stints, alpha is chosen on a time-ordered inner validation block inside
that training block, and calibration is fitted on the inner-validation margins.
The outer test block is never read.  Both comparisons are reported on the same
rows:

* the same-season 43-column E14 frame, and
* the same-season canonical (availability) logistic model.

Run:  python scripts/issue39_rapm.py
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
from Models import measurement as ms              # noqa: E402
from Models import player_ratings as pr           # noqa: E402
from Models import rapm                           # noqa: E402

JSONL = ROOT / "results" / "experiments.jsonl"
OUT_JSON = ROOT / "results" / "issue39_rapm.json"
OUT_MD = ROOT / "results" / "issue39_rapm.md"
N_BOOT = 2000
SEED = 42


def metrics(y, p) -> dict:
    m = nd.evaluate(np.asarray(y, dtype=int), np.asarray(p, dtype=float))
    return {k: (int(v) if k == "n" else float(v))
            for k, v in m.items() if k in ("n", "log_loss", "brier",
                                           "accuracy", "auc")}


def _delta(y, candidate, reference) -> dict:
    return ms.delta_stats(y, candidate, reference, n_boot=N_BOOT, seed=SEED)


def _fmt_delta(d: dict) -> str:
    return (f"{d['delta_candidate_minus_reference']:+.6f} "
            f"[{d['ci_lo']:+.6f}, {d['ci_hi']:+.6f}] "
            f"|z| {d['abs_z']:.2f}")


def main() -> int:
    t0 = time.time()
    games = nd.build_games()
    grouped = games[games["SEASON"] == "2020-21"].reset_index(drop=True)
    print(f"2020-21 games in the shared frame: {len(grouped)}")

    result = rapm.run_walk_forward(games=games)
    season = result["season"]
    diag = result["design_diagnostics"]
    idx = np.asarray(result["game_index"], dtype=int)
    y = season["y"].to_numpy(dtype=int)

    p_rapm = np.full(len(season), np.nan)
    p_rapm[idx] = result["p"]
    oof = np.isfinite(p_rapm)

    # Same-season comparators on exactly the same rows.
    _, p_e14, _ = ms.oof_e14(season)
    _, p_can, _ = ms.canonical_oof(season)
    if not (np.isfinite(p_e14[oof]).all() and np.isfinite(p_can[oof]).all()):
        raise AssertionError("a same-season comparator is missing an OOF row")

    y_oof = y[oof]
    m_rapm = metrics(y_oof, p_rapm[oof])
    m_e14 = metrics(y_oof, p_e14[oof])
    m_can = metrics(y_oof, p_can[oof])

    d_e14 = _delta(y_oof, p_rapm[oof], p_e14[oof])
    d_can = _delta(y_oof, p_rapm[oof], p_can[oof])
    folds_e14 = ms.per_fold_delta(y, p_rapm, p_e14, season)
    folds_can = ms.per_fold_delta(y, p_rapm, p_can, season)

    # Descriptive player table from a full-season fit.  NOT scored; the scored
    # predictions above come only from the walk-forward fits.
    rot = rapm.load_rotation()
    design = rapm.build_design(rot, season)
    panel_full = pr.build_panel(season)
    em_full = pr.ExpectedMinutes().fit(panel_full.rows)
    # Use the modal selected alpha rather than refitting the selection on all
    # rows; this block is descriptive only.
    alphas = [s["selected_alpha"] for s in result["split_table"]]
    alpha_full = float(pd.Series(alphas).mode().iloc[0])
    full_model = rapm.fit_rapm(design["X"], design["y"], design["w"],
                               design["player_ids"], alpha_full)
    names = (rot.drop_duplicates("PERSON_ID")
                .set_index("PERSON_ID")
                .assign(name=lambda d: d["PLAYER_FIRST"].fillna("")
                        + " " + d["PLAYER_LAST"].fillna(""))["name"])
    ratings = pd.Series(full_model.theta, index=full_model.player_ids)
    top = [{"player_id": int(pid), "name": str(names.get(pid, "")),
            "rapm_per_48": float(v)}
           for pid, v in ratings.sort_values(ascending=False).head(10).items()]
    bottom = [{"player_id": int(pid), "name": str(names.get(pid, "")),
               "rapm_per_48": float(v)}
              for pid, v in ratings.sort_values().head(10).items()]

    verdict = ("NULL / INCONCLUSIVE: true RAPM does not beat the same-season "
               "canonical model at |z| > 3, and its interval spans zero against "
               "the same-season E14 frame.")
    payload = {
        "issue": 39,
        "protocol_ref": "I39-prereg-true-rapm",
        "generated_utc": ms.utc_now(),
        "runtime_seconds": round(time.time() - t0, 1),
        "design": diag,
        "alpha_grid": list(result["alpha_grid"]),
        "split_table": result["split_table"],
        "n_scored": int(oof.sum()),
        "metrics": {"rapm": m_rapm, "e14_same_season": m_e14,
                    "canonical_same_season": m_can},
        "delta_vs_e14": d_e14,
        "delta_vs_canonical": d_can,
        "per_fold_vs_e14": folds_e14,
        "per_fold_vs_canonical": folds_can,
        "selected_alphas": alphas,
        "descriptive_full_season_fit": {
            "alpha": alpha_full,
            "home_points_per_48": float(full_model.home_points_per_48),
            "theta_sd": float(np.std(full_model.theta)),
            "top_10": top, "bottom_10": bottom,
            "note": ("descriptive only; the scored numbers use walk-forward "
                     "fits and never these coefficients"),
        },
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    # ------------------------------------------------------------------ print
    L: list[str] = []
    a = L.append
    a("=" * 78)
    a("ISSUE 39 -- TRUE RAPM, 2020-21, WALK-FORWARD")
    a("=" * 78)
    a(f"rotation games in frame       {diag['games_in_frame']}")
    a(f"games usable (5v5, PT_DIFF)   {diag['games_used']}")
    a(f"games partial / missing       {diag['games_skipped_partial']}")
    a(f"season games never downloaded {diag['games_skipped_empty']}")
    a(f"stint rows / intervals         {diag['stint_rows_used']} / "
      f"{diag['intervals']}")
    a(f"players                        {diag['players']}")
    a(f"outer test rows scored         {int(oof.sum())}")
    a("")
    a("PER FOLD (alpha chosen inside the training block)")
    a(f"{'k':>2} {'train':>5} {'test':>5} {'alpha':>9} {'theta sd':>9} "
      f"{'hca/48':>7} {'cal a':>6} {'logloss':>8}")
    for s in result["split_table"]:
        a(f"{s['split']:>2} {s['n_train']:>5} {s['n_test']:>5} "
          f"{s['selected_alpha']:>9.0f} {s['theta_sd']:>9.4f} "
          f"{s['home_points_per_48']:>7.3f} {s['calibration_a']:>6.3f} "
          f"{s['test_log_loss']:>8.4f}")
        grid = ", ".join(f"{g['alpha']:.0f}:{g['inner_log_loss']:.4f}"
                         for g in s["alpha_grid"])
        a(f"   inner grid {grid}")
    a("")
    a("POOLED, SAME ROWS")
    a(f"{'model':<28} {'n':>5} {'logloss':>9} {'brier':>8} {'auc':>7} "
      f"{'acc':>7}")
    for name, m in (("rapm", m_rapm), ("e14 same season", m_e14),
                    ("canonical same season", m_can)):
        a(f"{name:<28} {m['n']:>5} {m['log_loss']:>9.6f} {m['brier']:>8.6f} "
          f"{m['auc']:>7.4f} {m['accuracy']:>7.4f}")
    a("")
    a(f"delta vs E14 (positive = RAPM better)      {_fmt_delta(d_e14)}")
    a(f"delta vs canonical (positive = RAPM better) {_fmt_delta(d_can)}")
    a(f"folds RAPM better than E14: {folds_e14['n_folds_candidate_better']}"
      f"/{folds_e14['n_folds']}")
    a(f"folds RAPM better than canonical: "
      f"{folds_can['n_folds_candidate_better']}/{folds_can['n_folds']}")
    a("")
    a(f"verdict: {verdict}")

    text = "\n".join(L)
    print(text)
    OUT_MD.write_text(text + "\n")

    # Append results, not a second preregistration.
    ms.append_jsonl(JSONL, {
        "name": "I39-run-true-rapm",
        "status": "executed",
        "declared_utc": ms.utc_now(),
        "experiment_type": "true RAPM result (issue #39)",
        "protocol_ref": "I39-prereg-true-rapm",
        "design": diag,
        "n_scored": int(oof.sum()),
        "log_loss": m_rapm["log_loss"],
        "brier": m_rapm["brier"],
        "auc": m_rapm["auc"],
        "accuracy": m_rapm["accuracy"],
        "e14_same_season_log_loss": m_e14["log_loss"],
        "canonical_same_season_log_loss": m_can["log_loss"],
        "delta_vs_e14": d_e14,
        "delta_vs_canonical": d_can,
        "selected_alphas": alphas,
        "verdict": verdict,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
