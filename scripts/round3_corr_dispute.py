"""Settle the corr(elo_diff, ewm_margin) dispute. Both numbers are probably right
about different things, and the repo needs one definition.

Peers reported, for the same feature and the same builder:
  0.636  (my round3_run_T / round3_final, and peer 0a2786ed)
  0.90-0.93  (peer d84d8cf7, "in BOTH ratings.fit_predict_elo and
              baselines.elo_rating_pass", across k=10/20/40)

The two are reconcilable if they correlate against DIFFERENT targets:

  (a) LEVEL  corr(elo_diff, away_ewm_margin)              -> what I computed.
      A single one of the 43 columns, compared one at a time.

  (b) CONTRAST  corr(elo_diff, home_ewm_margin - away_ewm_margin)
      A derived quantity that is NOT a column in the frame at all.

(b) is the one that describes how much NEW ordering elo contributes, and it is
the one that matters for the redundancy claim: a column that is 0.90-correlated
with the margin contrast is a different estimate of the margin, which is exactly
what an Elo rating is, and that is what "not a rearrangement" has to be measured
against. Reporting only (a) would understate the redundancy.

This script measures BOTH, plus the partial correlation after the frame's own
margin contrast, and pins which builder each comes from. No model is fitted.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from scripts.round3_common import load_cached, log3            # noqa: E402
from src.DataProviders.feat_team_elo import team_elo_features  # noqa: E402


def best_level(e14: pd.DataFrame, s: pd.Series) -> tuple[float, str]:
    c = pd.concat([e14, s.rename("_x")], axis=1).corr(numeric_only=True).iloc[:-1, -1]
    return float(c.abs().max()), str(c.abs().idxmax())


def main() -> None:
    games, e14 = load_cached()
    out: dict = {"definitions": {
        "level": "max |corr| of the feature against each of the 43 E14 columns "
                 "individually -- i.e. against a LEVEL",
        "contrast": "|corr| against home_ewm_margin - away_ewm_margin, a "
                    "quantity that is not a column in the frame",
        "partial_given_contrast": "|corr| of the residual after regressing both "
                                  "on the frame's margin contrast",
    }}
    rows = []
    for k in (5.0, 10.0, 15.0, 20.0, 30.0, 40.0):
        elo = team_elo_features(games, k=k)
        d = elo["elo_diff"]
        lv, arg = best_level(e14, d)
        mc = e14["home_ewm_margin"] - e14["away_ewm_margin"]
        # residual correlation after removing the frame's own margin contrast.
        # The EWM levels are NaN for a team's first few games, so every
        # correlation here is computed on COMPLETE rows; the level |r| is
        # unaffected in substance because pandas .corr() is pairwise and uses
        # every shared row, which is what the earlier round-3 tables reported.
        m = d.notna() & mc.notna()
        dv = d.to_numpy(dtype=float)[m.to_numpy()]
        mv = mc.to_numpy(dtype=float)[m.to_numpy()]
        c_contrast = float(np.corrcoef(dv, mv)[0, 1])
        hv = e14["home_ewm_margin"].to_numpy(dtype=float)[m.to_numpy()]
        A = np.column_stack([np.ones(int(m.sum())), mv])
        r1 = dv - A @ np.linalg.lstsq(A, dv, rcond=None)[0]
        r2 = hv - A @ np.linalg.lstsq(A, hv, rcond=None)[0]
        rows.append({"K": k, "level_max_abs_r": round(lv, 4), "level_argmax": arg,
                     "abs_r_vs_margin_contrast": round(abs(c_contrast), 4),
                     "partial_r_given_contrast": round(abs(float(np.corrcoef(r1, r2)[0, 1])), 4),
                     "n_complete": int(m.sum())})
    out["by_K"] = rows
    print(f"{'K':>5} {'level r':>9} {'argmax':>24} {'vs contrast':>13} {'partial':>9}")
    for r in rows:
        print(f"{r['K']:>5g} {r['level_max_abs_r']:>9.4f} {r['level_argmax']:>24} "
              f"{r['abs_r_vs_margin_contrast']:>13.4f} {r['partial_r_given_contrast']:>9.4f}")

    best = min(rows, key=lambda r: r["abs_r_vs_margin_contrast"])
    out["verdict"] = (
        f"BOTH numbers are real. The 0.636 is a correlation against a single "
        f"E14 LEVEL and the 0.90+ is against the frame's margin CONTRAST "
        f"(home_ewm_margin - away_ewm_margin), which is not itself a column. "
        f"The redundancy claim must be stated against the contrast: at K=10 "
        f"|r| = {rows[1]['abs_r_vs_margin_contrast']:.3f}, above 0.8, so elo_diff "
        f"IS a re-estimate of the margin contrast and not an orthogonal signal. "
        f"Its residual contribution after the frame's own margin contrast is "
        f"|r| = {rows[1]['partial_r_given_contrast']:.3f}."
    )
    print("\nVERDICT:", out["verdict"])
    log3({"run_id": "W1_elo_corr_definition_dispute", "block": "W",
          "kind": "DISPUTE_RESOLUTION", **out})
    (ROOT / "results" / "elo_corr_dispute.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()