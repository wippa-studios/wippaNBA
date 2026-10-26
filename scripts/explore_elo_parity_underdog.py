"""Does betting the underdog make positive ROI when the Elo gap is under 100?

EXPLORATORY (user-directed, 2026-10-27) - not a pre-registered family. The
hypothesis: in near-Elo-parity games the underdog is underpriced. Measured on
both markets the repo holds:

* **ML at best-of-books** (`odds_ml_bestbooks`, OddsPortal best prices),
  side = the market underdog, and separately the Elo-lower side;
* **ATS at a flat -110** (single-book close, `effective_line` convention),
  side = the spread underdog (L_eff < 0), with the favourite side and the
  Elo-agrees/Elo-disagrees split reported.

Elo: the issue-#81 pass (`baselines.elo_rating_pass`, k=10, HFA=70,
season_reversion=0.75, margin-weighted, same-day batched); the gap is
home + 70 - away, i.e. the adjusted pre-game gap. Discovery 2015-16..2024-25,
holdout 2025-26.

Usage:  python scripts/explore_elo_parity_underdog.py
        [--out results/exploratory/elo_parity_underdog.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

import edge_search as es                       # noqa: E402
from Models import baselines as bl             # noqa: E402
from DataProviders import nba_data as nd       # noqa: E402


def elo_gap_frame() -> pd.DataFrame:
    games = nd.build_games().reset_index(drop=True)
    gg = games.rename(columns={"y": "home_win", "GAME_DATE": "date"})
    fit = bl.elo_rating_pass(gg, k=10.0, home_advantage=70.0, same_day="batched",
                             use_margin=True, season_reversion=0.75)
    return pd.DataFrame({"GAME_ID": games["GAME_ID"],
                         "elo_gap": fit.games["rating_home_before"].to_numpy() + 70.0
                                    - fit.games["rating_away_before"].to_numpy()})


def _roi(profit: np.ndarray) -> dict:
    se = float(profit.std(ddof=1) / np.sqrt(len(profit)))
    return {"n": int(len(profit)), "roi": round(float(profit.mean()), 4),
            "z": round(float(profit.mean() / se), 2)}


def analyse() -> dict:
    eg = elo_gap_frame()

    j, _ = es.load_bestprice_frame()
    j = j.merge(eg, on="GAME_ID", how="left")
    j["coherent"] = (1 / j["O1"] + 1 / j["O2"]) >= 1
    ml = {"market_underdog": {}, "elo_underdog": {}}
    for gap_label, gx in (("|gap| < 100", j[np.abs(j["elo_gap"]) < 100]),
                          ("|gap| < 50", j[np.abs(j["elo_gap"]) < 50])):
        dog_home = ~gx["fav_home"].to_numpy()
        O_dog = np.where(dog_home, gx["O1"], gx["O2"]).astype(float)
        won = np.where(dog_home, gx["y"] == 1, gx["y"] == 0)
        ml["market_underdog"][gap_label] = {}
        for split in ("discovery", "holdout"):
            m = (gx["split"] == split).to_numpy()
            ml["market_underdog"][gap_label][split] = _roi(
                np.where(won[m], O_dog[m] - 1, -1.0))
        coh = (gx["coherent"]).to_numpy()
        for split in ("discovery", "holdout"):
            m = ((gx["split"] == split) & coh).to_numpy()
            ml["market_underdog"][gap_label][f"{split}_coherent"] = _roi(
                np.where(won[m], O_dog[m] - 1, -1.0))
    # the Elo-lower side, |gap| < 100, both splits
    close = j[np.abs(j["elo_gap"]) < 100]
    elo_dog_home = (close["elo_gap"] < 0).to_numpy()
    O_edog = np.where(elo_dog_home, close["O1"], close["O2"]).astype(float)
    won_edog = np.where(elo_dog_home, close["y"] == 1, close["y"] == 0)
    ml["elo_underdog"]["|gap| < 100"] = {
        "both_splits": _roi(np.where(won_edog, O_edog - 1, -1.0)),
        "elo_side_is_market_underdog_share": round(
            float((elo_dog_home == ~close["fav_home"].to_numpy()).mean()), 4),
    }

    f, _ = es.priced_frame()
    f = f.merge(eg, on="GAME_ID", how="left")
    f = f[f["L_eff"] != 0].copy()
    f["elo_dog_home"] = f["elo_gap"] < 0
    f["mkt_dog_home"] = f["L_eff"] < 0
    f["agree"] = f["elo_dog_home"] == f["mkt_dog_home"]
    ats = {}
    for gap_label, gx in (("|gap| < 100", f[np.abs(f["elo_gap"]) < 100]),
                          ("|gap| < 50", f[np.abs(f["elo_gap"]) < 50])):
        dog_home = (gx["L_eff"] < 0).to_numpy()
        won_dog = np.where(dog_home, gx["margin"] > gx["L_eff"], gx["margin"] < gx["L_eff"])
        ats[gap_label] = {
            "underdog": {}, "favourite": {},
        }
        for split in ("discovery", "holdout", "both"):
            m = (gx["split"] == split).to_numpy() if split != "both" else np.ones(len(gx), bool)
            ats[gap_label]["underdog"][split] = _roi(np.where(won_dog[m], 100 / 110, -1.0))
            ats[gap_label]["favourite"][split] = _roi(np.where(~won_dog[m], 100 / 110, -1.0))
    for a, lab in ((True, "elo_agrees_with_market"), (False, "elo_disagrees_with_market")):
        x = f[(np.abs(f["elo_gap"]) < 100) & (f["agree"] == a)]
        dog_home = (x["L_eff"] < 0).to_numpy()
        won_dog = np.where(dog_home, x["margin"] > x["L_eff"], x["margin"] < x["L_eff"])
        ats[lab] = {
            "n": int(len(x)),
            "underdog_both_splits": _roi(np.where(won_dog, 100 / 110, -1.0)),
            "favourite_both_splits": _roi(np.where(~won_dog, 100 / 110, -1.0)),
        }

    return {
        "generated": datetime.now(timezone.utc).isoformat(),
        "exploratory": True,
        "hypothesis": "in near-Elo-parity games (|adjusted gap| < 100) the underdog is underpriced",
        "elo_definition": "baselines.elo_rating_pass(k=10, HFA=70, reversion=0.75, margin-weighted); gap = home + 70 - away",
        "ml_best_of_books": ml,
        "ats_single_book_close": ats,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "exploratory" / "elo_parity_underdog.json")
    args = ap.parse_args()
    out = analyse()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    u = out["ats_single_book_close"]["|gap| < 100"]["underdog"]
    print(f"ATS underdog |gap|<100: discovery {u['discovery']['roi']:+.2%} "
          f"(z {u['discovery']['z']}), holdout {u['holdout']['roi']:+.2%} "
          f"(z {u['holdout']['z']})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
