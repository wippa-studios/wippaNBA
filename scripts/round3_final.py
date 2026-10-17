"""Round 3, final consolidation. Two jobs, both of which fix things above.

JOB 1  the order-robustness check, done correctly
-------------------------------------------------
U4 permuted the `games` rows and then called `walk_forward_split`, which slices
by ROW INDEX. Permuting therefore broke the split geometry: the "test" blocks
were no longer later than the training blocks, which is why U4's absolute log
loss moved from 0.612528 to 0.623113 and its delta collapsed. That number
measures a broken split, not an order-dependent artefact, and it is superseded
rather than reported.

The honest version keeps the chronology and permutes only WITHIN a date, which
is the ordering the features could conceivably depend on: team EWMs sort by
(date, team) within team, and `fit_predict_elo` sorts internally by
(date, home, away) with mergesort. If within-date order mattered, the delta
would move here. It should not, and if it does not, the claim "these features
are order-invariant" is earned rather than asserted.

JOB 2  the promotable configuration, with its degrees of freedom stated
---------------------------------------------------------------------
T3 was reported as "E14 + availability + 3 elo contrast columns" and carried
49 columns. Two of those three columns were `elo_diff` and `elo_diff_adj =
elo_diff + 70`. A logistic regression with an intercept cannot separate those:
they are one degree of freedom, not two. So the candidate is

    E14 (43) + availability (4) + elo_diff (1)  = 48 columns, 47 d.o.f.
                                  - away_expected_minutes_diff (perfectly
                                    collinear with its home twin) = 46 d.o.f.

The number is unchanged; what was wrong was the description. Corrected here, with
per-fold deltas against the availability model, because 5/5 is the standard the
pre-registration set and it is the thing that has to be re-earned.
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

from scripts.round3_common import load_cached, oof_fixed, compare, log3  # noqa: E402
from src.DataProviders import nba_data as nd                              # noqa: E402
from src.DataProviders.feat_team_elo import team_elo_features            # noqa: E402
from scripts.round3_run_U import per_fold_delta                          # noqa: E402


def assemble(e14: pd.DataFrame, av: pd.DataFrame,
             extra: list[pd.DataFrame]) -> pd.DataFrame:
    X = pd.concat([e14.reset_index(drop=True), av.reset_index(drop=True)]
                  + [d.reset_index(drop=True) for d in extra], axis=1)
    dup = [c for c in X.columns if list(X.columns).count(c) > 1]
    assert not dup, f"duplicate columns: {dup}"
    return X.loc[:, X.nunique(dropna=False) > 1].reset_index(drop=True)


def main() -> None:
    t0 = time.time()
    games, e14 = load_cached()
    y = games["y"].to_numpy()
    av = nd.availability_features(games, 15.0)
    elo = team_elo_features(games, k=10.0)

    X_av = assemble(e14, av, [])
    p_av = oof_fixed(X_av.to_numpy(dtype=float), y, games, C=1.0)
    X_final = assemble(e14, av, [elo[["elo_diff"]]])
    p_fin = oof_fixed(X_final.to_numpy(dtype=float), y, games, C=1.0)

    vs_av = compare(y, p_fin, p_av)
    pf = per_fold_delta(y, p_fin, p_av, games)
    p_bar = oof_fixed(e14.to_numpy(dtype=float), y, games, C=1.0)
    vs_bar = compare(y, p_fin, p_bar)

    collinear = []
    b = X_final.corr(numeric_only=True).abs()
    for i, a in enumerate(X_final.columns):
        for c in X_final.columns[i + 1:]:
            if b.loc[a, c] > 1 - 1e-12:
                collinear.append([a, c, round(float(b.loc[a, c]), 12)])

    rec = {"run_id": "V1_final_48col", "block": "V", "kind": "FINAL",
           "n_columns": int(X_final.shape[1]),
           "effective_dof": int(X_final.shape[1]) - len(collinear),
           "perfectly_collinear_pairs": collinear,
           "vs_availability_model": vs_av, "per_fold": pf,
           "vs_frozen_bar_e14": vs_bar,
           "note": "elo_diff_adj was dropped: elo_diff + 70 is not separable "
                   "from elo_diff when the model has an intercept. T3's '3 "
                   "columns' were 2 columns carrying 1 degree of freedom."}
    log3(rec)
    print(json.dumps({k: v for k, v in rec.items()
                      if k != "per_fold"}, indent=1), flush=True)
    print("per_fold:", json.dumps(pf, indent=1), flush=True)

    # ---- JOB 1: within-date order perturbation, chronology preserved ----
    rng = np.random.default_rng(11)
    # Reorder WITHIN each date only. Chronology - the thing
    # walk_forward_split slices on - is preserved exactly.
    order = np.lexsort((rng.random(len(games)),
                        games["GAME_DATE"].to_numpy()))
    g2 = games.iloc[order].reset_index(drop=True)
    e14_2 = e14.iloc[order].reset_index(drop=True)
    assert g2["GAME_DATE"].is_monotonic_increasing, "jitter broke chronology"
    # Re-derive features on the reordered frame rather than assuming the frame
    # builder is order-invariant; that is the whole point of the test.
    av2 = nd.availability_features(g2, 15.0)
    elo2 = team_elo_features(g2, k=10.0)
    Xa2 = assemble(e14_2, av2, [])
    Xc2 = assemble(e14_2, av2, [elo2[["elo_diff"]]])
    p_av2 = oof_fixed(Xa2.to_numpy(dtype=float), g2["y"].to_numpy(), g2, C=1.0)
    p_c2 = oof_fixed(Xc2.to_numpy(dtype=float), g2["y"].to_numpy(), g2, C=1.0)
    ord2 = compare(g2["y"].to_numpy(), p_c2, p_av2)
    log3({"run_id": "V2_within_date_order_perturbation", "block": "V",
          "kind": "ROBUSTNESS",
          "supersedes": "U4_row_order_perturbation (invalid: permuting rows "
                        "broke walk_forward_split's chronological geometry)",
          "reading": "within-date game order jittered, chronology preserved. "
                     "Team EWMs sort by (team, date) and fit_predict_elo sorts "
                     "internally by (date, home, away), so the delta should not move.",
          "vs_availability_model": ord2,
          "natural_order_delta": vs_av["improvement_nats"],
          "delta_shift": round(ord2["improvement_nats"] - vs_av["improvement_nats"], 8)})
    print("WITHIN-DATE ORDER:", json.dumps(ord2), flush=True)

    # ---- K robustness on the FINAL single-column configuration ----------
    for k in (5.0, 10.0, 15.0, 20.0, 30.0):
        ek = team_elo_features(games, k=k)
        pk = oof_fixed(assemble(e14, av, [ek[["elo_diff"]]]).to_numpy(dtype=float),
                        y, games, C=1.0)
        r = compare(y, pk, p_av)
        log3({"run_id": f"V3_elo_diff_k{k:g}", "block": "V", "kind": "ROBUSTNESS",
              "K": k, "vs_availability_model": r,
              "vs_frozen_bar_e14": compare(y, pk, p_bar),
              "per_fold": per_fold_delta(y, pk, p_av, games)})
        print(f"K={k:<5g} ll={r['log_loss']:.6f} z_av={r['abs_z']:.2f}", flush=True)
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()