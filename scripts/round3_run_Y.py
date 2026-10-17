"""Round 3, block Y: squad continuity + load decay. Gated, then scored.

PRE-REGISTERED ARMS
-------------------
  Y1_continuity_only      E14 + availability + elo_diff + the 8 contrast columns
  Y2_depth_load_decay     E14 + availability + elo_diff + squad_depth and
                          load_decay only (4 columns) - the two quantities
                          `availability_features` cannot express, nothing else
  Y3_continuity_no_elo    E14 + availability + continuity, no elo. Isolates the
                          block from the block it would otherwise be stacked on.

REFERENCE
---------
The promoted 48-column frame, 0.612523, which itself only clears |z| 2.535
against the frozen 43-column bar and carries +0.0020 at |z| 1.12 over the
availability model. So the bar for a NEW block is not "beat the bar" - it is
"beat the already-promoted model by more than its own marginal effect", i.e.
more than |z| 1.12 on the same kind of comparison.

REDUNDANCY, DONE RIGHT THIS TIME
--------------------------------
Block W measured `elo_diff` against the 43 E14 columns one at a time, got
0.636, and was wrong: against the margin CONTRAST
(`home_ewm_margin - away_ewm_margin`) it is 0.89. A column-by-column scan cannot
see a derived quantity that is not itself a column.

So every candidate here is checked against a battery of DERIVED contrasts that
the frame does not contain but that any competitor column would resemble:
    margin        home_ewm_margin - away_ewm_margin
    net_rating    home_ewm_net_rating - away_ewm_net_rating
    off/def       (home_ewm_off_rating - away_ewm_off_rating)
                  + (home_ewm_def_rating - away_ewm_def_rating)
    scoring       (home_ewm_efg - away_ewm_efg) + (home_ewm_ts_pct - away_ewm_ts_pct)
    opp_efg       away_ewm_opp_efg - home_ewm_opp_efg
    hfa_split     (home_ewm_margin_home - home_ewm_margin_away)
                  - (away_ewm_margin_home - away_ewm_margin_away)
    rest          home_rest_days - away_rest_days
Any |r| > 0.8 against ANY of these is a rearrangement and is reported as one.
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
from scripts.round3_verify import gate1_truncation                       # noqa: E402
from scripts.round3_run_U import per_fold_delta                          # noqa: E402
from src.DataProviders import nba_data as nd                              # noqa: E402
from src.DataProviders.feat_team_elo import team_elo_features            # noqa: E402
from src.DataProviders.feat_squad_continuity import (                     # noqa: E402
    squad_continuity_features)

PROMOTED_LL = 0.612523


def contrast_battery(e14: pd.DataFrame) -> dict[str, pd.Series]:
    h, a = e14["home_ewm_margin"], e14["away_ewm_margin"]
    return {
        "margin": h - a,
        "net_rating": e14["home_ewm_net_rating"] - e14["away_ewm_net_rating"],
        "off_minus_def": (e14["home_ewm_off_rating"] - e14["away_ewm_off_rating"])
                         + (e14["home_ewm_def_rating"] - e14["away_ewm_def_rating"]),
        "scoring": (e14["home_ewm_efg"] - e14["away_ewm_efg"])
                   + (e14["home_ewm_ts_pct"] - e14["away_ewm_ts_pct"]),
        "opp_efg": e14["away_ewm_opp_efg"] - e14["home_ewm_opp_efg"],
        "hfa_split": (e14["home_ewm_margin_home"] - e14["home_ewm_margin_away"])
                     - (e14["away_ewm_margin_home"] - e14["away_ewm_margin_away"]),
        "rest": e14["home_rest_days"] - e14["away_rest_days"],
    }


def redundancy(e14: pd.DataFrame, cand: pd.DataFrame, thresh: float = 0.8) -> dict:
    bat = contrast_battery(e14)
    cols = list(e14.select_dtypes(include=[np.number]).columns) + list(bat)
    ref = pd.concat([e14.select_dtypes(include=[np.number]),
                     pd.DataFrame(bat)], axis=1)
    ref = ref.loc[:, ~ref.columns.duplicated()]
    rows = {}
    for col in cand.select_dtypes(include=[np.number]).columns:
        s = pd.concat([ref, cand[[col]]], axis=1).corr(numeric_only=True).iloc[:-1, -1]
        rows[col] = {"max_abs_r": round(float(s.abs().max()), 4),
                     "argmax": str(s.abs().idxmax()),
                     "is_derived_contrast": bool(s.abs().idxmax() in bat)}
    over = {k: v for k, v in rows.items() if v["max_abs_r"] > thresh}
    return {"threshold": thresh, "n_battery": len(bat),
            "n_over_threshold": len(over), "over_threshold": over, "per_column": rows}


def assemble(*dfs: pd.DataFrame) -> pd.DataFrame:
    X = pd.concat([d.reset_index(drop=True) for d in dfs], axis=1)
    dup = [c for c in X.columns if list(X.columns).count(c) > 1]
    assert not dup, f"duplicate columns: {dup}"
    return X.loc[:, X.nunique(dropna=False) > 1].reset_index(drop=True)


def main() -> None:
    t0 = time.time()
    games, e14 = load_cached()
    y = games["y"].to_numpy()

    # ---- GATES FIRST -------------------------------------------------
    sc = pd.read_pickle("/home/Joel/.cache/nba-frames/sc15.pkl")
    gate = gate1_truncation(games, lambda g: squad_continuity_features(g, 15.0),
                            "2021-10-20", "feat_squad_continuity halflife=15")
    log3({"run_id": "Y0_truncation_gate", "block": "Y", "kind": "LEAKAGE_GATE", **gate})
    print("GATE1:", json.dumps(gate), flush=True)
    if not gate["pass"]:
        raise SystemExit("truncation gate failed; refusing to score")

    red = redundancy(e14, sc)
    print("REDUNDANCY over threshold:", json.dumps(red["over_threshold"], indent=1), flush=True)
    log3({"run_id": "Y0_redundancy", "block": "Y", "kind": "REDUNDANCY",
          "n_over_threshold": red["n_over_threshold"],
          "over_threshold": red["over_threshold"], "per_column": red["per_column"]})

    # ---- references --------------------------------------------------
    av = nd.availability_features(games, 15.0)
    elo = team_elo_features(games, k=10.0)[["elo_diff"]]
    p_bar = oof_fixed(e14.to_numpy(dtype=float), y, games, C=1.0)
    X_prom = assemble(e14, av, elo)
    p_prom = oof_fixed(X_prom.to_numpy(dtype=float), y, games, C=1.0)
    prom_ll = compare(y, p_prom, p_prom)["log_loss"]
    assert abs(prom_ll - PROMOTED_LL) < 1e-5, f"promoted frame moved: {prom_ll}"
    print(f"BAR {compare(y, p_bar, p_bar)['log_loss']:.6f} | PROMOTED {prom_ll:.6f}",
          flush=True)

    diffs = [c for c in sc.columns if c.endswith("_diff")]
    depth_decay = [c for c in sc.columns
                   if c.endswith(("_depth", "_load_decay", "_depth_diff",
                                  "_load_decay_diff"))]
    arms = {
        "Y1_continuity_only":   assemble(e14, av, elo, sc[diffs]),
        "Y2_depth_load_decay":  assemble(e14, av, elo, sc[depth_decay]),
        "Y3_continuity_no_elo": assemble(e14, av, sc[diffs]),
    }
    results = {}
    for run_id, X in arms.items():
        p = oof_fixed(X.to_numpy(dtype=float), y, games, C=1.0)
        vs_prom = compare(y, p, p_prom)
        pf = per_fold_delta(y, p, p_prom, games)
        rec = {"run_id": run_id, "block": "Y", "kind": "PRE_REGISTERED",
               "n_features": int(X.shape[1]), "vs_promoted_48col": vs_prom,
               "per_fold": pf}
        log3(rec)
        results[run_id] = rec
        print(f"{run_id:22s} cols={X.shape[1]:3d} ll={vs_prom['log_loss']:.6f} "
              f"z={vs_prom['abs_z']:.2f} folds={pf['folds_cand_better']}/{pf['n_folds']} "
              f"sd={pf['fold_delta_sd']}", flush=True)

    print(f"\ntotal {time.time() - t0:.1f}s")
    print("ACCEPTANCE: must beat the PROMOTED model's own marginal bar, i.e. "
          "|z| > 1.12 over 0.612523, and hold 5/5 folds.")
    for k, v in results.items():
        ok = v["vs_promoted_48col"]["abs_z"] > 1.125 and \
            v["per_fold"]["folds_cand_better"] == v["per_fold"]["n_folds"]
        print(f"  {k:22s} {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()