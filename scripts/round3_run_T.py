"""Round 3, block T: team Elo. Pre-registered arms, then run.

HYPOTHESIS
----------
The 43-column frame has no rating. Every strength estimate in it is a linear
EWM of the last ~10 box scores. Elo is a margin-weighted recursive update with
its own forgetting schedule. Two estimators of team strength that disagree in
form should combine; the standalone Elo already scores 0.6234 against the
frame's 0.6183, so this is combining two comparable things, not bolting on a
weak one.

PRE-REGISTERED ARMS
-------------------
  T1_elo_on_e14             E14 + 6 elo columns, against the FROZEN bar
  T2_elo_on_availability    E14 + availability + elo, against the availability
                            model. This is the delta that decides anything: a
                            gain smaller than the availability gain already in
                            hand is not worth the columns.
  T3_elo_diff_only          E14 + availability + the two CONTRAST columns only.
                            Variance control: 4 columns instead of 8.
  T4_elo_k20                E14 + availability + elo at the untuned K=20
                            (FiveThirtyEight default). A K the project never
                            chose by validation, as a check that the result is
                            not an artefact of the tuned K.

ACCEPTANCE
----------
|z| > 2.535 (Bonferroni, 4 arms, family-wise 0.05) against the availability
model. Anything less is "no meaningful difference" and is logged that way.

GATES
-----
Redundancy: max |r| of each elo column against the 43 E14 columns. `elo_diff`
is EXPECTED to be highly correlated with the EWM margin contrast - if it comes
back above 0.95 the arm is a rearrangement and is labelled so regardless of
its delta.
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
from scripts.round3_run_S import prune, REDUNDANT, build_frame            # noqa: E402

E14_BAR = 0.618307


def redundancy(base: pd.DataFrame, cand: pd.DataFrame) -> dict:
    b = base.select_dtypes(include=[np.number])
    c = cand.select_dtypes(include=[np.number])
    rows = {}
    for col in c.columns:
        s = pd.concat([b, c[[col]]], axis=1).corr(numeric_only=True).iloc[:-1, -1]
        rows[col] = {"max_abs_r_vs_e14": round(float(s.abs().max()), 4),
                     "argmax": str(s.abs().idxmax())}
    return rows


def main() -> None:
    t0 = time.time()
    games, e14 = load_cached()
    y = games["y"].to_numpy()
    av = nd.availability_features(games, 15.0)
    sq = pd.read_pickle("/tmp/opencode/nba/sq15.pkl")
    sq_lean = prune(sq, REDUNDANT)

    elo = team_elo_features(games, k=10.0)
    elo20 = team_elo_features(games, k=20.0)
    print("elo cols", list(elo.columns), flush=True)
    print("REDUNDANCY vs E14:", json.dumps(redundancy(e14, elo), indent=1), flush=True)

    p_bar = oof_fixed(e14.to_numpy(dtype=float), y, games, C=1.0)
    assert abs(compare(y, p_bar, p_bar)["log_loss"] - E14_BAR) < 1e-6, "bar moved"
    p_av = oof_fixed(build_frame(e14, av, sq, []).to_numpy(dtype=float), y, games, C=1.0)

    diff_cols = ["elo_diff", "elo_diff_adj", "elo_edge"]
    arms = [
        ("T1_elo_on_e14",          build_frame(e14, sq.iloc[:, :0], elo, list(elo.columns))),
        ("T2_elo_on_availability", build_frame(e14, av, elo, list(elo.columns))),
        ("T3_elo_diff_only",       build_frame(e14, av, elo, diff_cols)),
        ("T4_elo_k20",             build_frame(e14, av, elo20, [c for c in elo20.columns
                                                               if c != "home_home_edge_elo_resid"])),
    ]
    for run_id, X in arms:
        p = oof_fixed(X.to_numpy(dtype=float), y, games, C=1.0)
        rec = {"run_id": run_id, "block": "T", "kind": "PRE_REGISTERED",
               "n_features": int(X.shape[1]),
               "vs_frozen_bar_e14": compare(y, p, p_bar),
               "vs_availability_model": compare(y, p, p_av)}
        log3(rec)
        print(f"{run_id:24s} cols={X.shape[1]:3d} ll={rec['vs_frozen_bar_e14']['log_loss']:.6f} "
              f"z_bar={rec['vs_frozen_bar_e14']['abs_z']:.2f} "
              f"z_av={rec['vs_availability_model']['abs_z']:.2f}", flush=True)
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()