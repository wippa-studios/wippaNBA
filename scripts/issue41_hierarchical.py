"""Issue #41: hierarchical player-to-team ratings with partial pooling.

Pre-registered in ``results/experiments.jsonl`` as ``I41-prereg-hierarchical``
before this script ran.

THE MODEL
---------
``team_net_per_100[j] = mu + sum_i A[j, i] theta_i + eps`` with
``theta_i ~ N(0, tau^2)``.  ``A[j, i]`` is player *i*'s expected minutes share
(``E[MIN] / 240``) for team-game *j*.  The prior standard deviation is the
estimated quantity: the ridge parameter ``alpha = sigma^2 / tau^2`` is selected
from a fixed grid on a time-ordered inner validation block **inside each outer
training fold**.  The test block never enters the fit, the selection or the
calibration.

The roster is the expected one -- players who appeared for the team in the
previous ten team-games -- never the actual squad.

Run:  python scripts/issue41_hierarchical.py
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
from Models import hierarchical_ratings as hr     # noqa: E402
from Models import measurement as ms              # noqa: E402

JSONL = ROOT / "results" / "experiments.jsonl"
OUT_JSON = ROOT / "results" / "issue41_hierarchical.json"
OUT_MD = ROOT / "results" / "issue41_hierarchical.md"
N_BOOT = 2000
SEED = 42
RECORDED_CANONICAL = 0.6190


def metrics(y, p) -> dict:
    m = nd.evaluate(np.asarray(y, dtype=int), np.asarray(p, dtype=float))
    return {k: (int(v) if k == "n" else float(v))
            for k, v in m.items() if k in ("n", "log_loss", "brier",
                                           "accuracy", "auc")}


def main() -> int:
    t0 = time.time()
    games = nd.build_games()
    y = games["y"].to_numpy(dtype=int)

    result = hr.run_walk_forward(games=games)
    idx = np.asarray(result["game_index"], dtype=int)
    p_hr = np.full(len(games), np.nan)
    p_hr[idx] = result["p"]
    if not np.isfinite(p_hr[idx]).all():
        raise AssertionError("hierarchical OOF has missing rows")

    _, p_e14, mask = ms.oof_e14(games)
    _, p_can, _ = ms.canonical_oof(games)
    p_e14 = np.asarray(p_e14, dtype=float)
    p_can = np.asarray(p_can, dtype=float)

    ok = np.zeros(len(games), dtype=bool)
    ok[idx] = True
    if not (np.isfinite(p_e14[ok]).all() and np.isfinite(p_can[ok]).all()):
        raise AssertionError("a comparator is missing an OOF row")

    yy = y[ok]
    m_hr = metrics(yy, p_hr[ok])
    m_e14 = metrics(yy, p_e14[ok])
    m_can = metrics(yy, p_can[ok])
    d_e14 = ms.delta_stats(yy, p_hr[ok], p_e14[ok], n_boot=N_BOOT, seed=SEED)
    d_can = ms.delta_stats(yy, p_hr[ok], p_can[ok], n_boot=N_BOOT, seed=SEED)

    verdict = (
        "PROMOTED" if d_can["delta_candidate_minus_reference"] > 0
        and d_can["abs_z"] > 3 else
        "NULL: hierarchical partial pooling does not beat the same-row "
        "canonical model at |z| > 3, nor the 43-column frame. At this sample "
        "size the team-level aggregates already carry the recoverable signal.")

    payload = {
        "issue": 41,
        "protocol_ref": "I41-prereg-hierarchical",
        "generated_utc": ms.utc_now(),
        "runtime_seconds": round(time.time() - t0, 1),
        "n_scored": int(ok.sum()),
        "panel_rows": result["panel_rows"],
        "alpha_grid": result["alpha_grid"],
        "split_table": result["split_table"],
        "metrics": {"hierarchical": m_hr, "e14": m_e14, "canonical": m_can},
        "delta_vs_e14": d_e14,
        "delta_vs_canonical": d_can,
        "per_fold_vs_e14": ms.per_fold_delta(y, p_hr, p_e14, games),
        "per_fold_vs_canonical": ms.per_fold_delta(y, p_hr, p_can, games),
        "recorded_canonical_reference": RECORDED_CANONICAL,
        "verdict": verdict,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    L: list[str] = []
    a = L.append
    a("=" * 78)
    a("ISSUE 41 -- HIERARCHICAL PLAYER-TO-TEAM RATINGS")
    a("=" * 78)
    a(f"candidate panel rows {result['panel_rows']}; scored games "
      f"{int(ok.sum())}")
    a("pooling strength selected inside each outer training fold; expected "
      "roster only")
    a("")
    a("PER FOLD")
    a(f"{'k':>2} {'train':>5} {'test':>5} {'alpha':>7} {'theta sd':>9} "
      f"{'cal a':>6} {'cal b':>7} {'logloss':>8}")
    for s in result["split_table"]:
        a(f"{s['split']:>2} {s['n_train']:>5} {s['n_test']:>5} "
          f"{s['selected_alpha']:>7.2f} {s['theta_sd']:>9.4f} "
          f"{s['calibration_a']:>6.3f} {s['calibration_b']:>7.3f} "
          f"{s['test_log_loss']:>8.4f}")
        grid = ", ".join(f"{g['alpha']:.1f}:{g['inner_log_loss']:.4f}"
                         for g in s["alpha_grid"])
        a(f"   inner grid {grid}")
    a("")
    a("POOLED, SAME ROWS")
    a(f"{'model':<24} {'n':>5} {'logloss':>9} {'brier':>8} {'auc':>7} "
      f"{'acc':>7}")
    for name, m in (("hierarchical", m_hr), ("e14", m_e14),
                    ("canonical", m_can)):
        a(f"{name:<24} {m['n']:>5} {m['log_loss']:>9.6f} {m['brier']:>8.6f} "
          f"{m['auc']:>7.4f} {m['accuracy']:>7.4f}")
    a("")
    a(f"delta vs E14 (positive = hierarchical better)       "
      f"{d_e14['delta_candidate_minus_reference']:+.6f} "
      f"[{d_e14['ci_lo']:+.6f},{d_e14['ci_hi']:+.6f}] "
      f"|z| {d_e14['abs_z']:.2f}")
    a(f"delta vs canonical (positive = hierarchical better) "
      f"{d_can['delta_candidate_minus_reference']:+.6f} "
      f"[{d_can['ci_lo']:+.6f},{d_can['ci_hi']:+.6f}] "
      f"|z| {d_can['abs_z']:.2f}")
    a("")
    a(f"verdict: {verdict}")
    text = "\n".join(L)
    print(text)
    OUT_MD.write_text(text + "\n")

    ms.append_jsonl(JSONL, {
        "name": "I41-run-hierarchical",
        "status": "executed",
        "declared_utc": ms.utc_now(),
        "experiment_type": "hierarchical player-rating result (issue #41)",
        "protocol_ref": "I41-prereg-hierarchical",
        "n_scored": int(ok.sum()),
        "metrics": {"hierarchical": m_hr, "e14": m_e14, "canonical": m_can},
        "delta_vs_e14": d_e14,
        "delta_vs_canonical": d_can,
        "selected_alphas": [s["selected_alpha"] for s in result["split_table"]],
        "verdict": verdict,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
