"""Family 7: lay bets at the moneyline odds (issue #95, pre-registered).

Laying team X at decimal odds O with backer's stake 1:
    profit = +1 if X loses, -(O - 1) if X wins.

The pre-registered identity: lay ROI = -back ROI at the same odds, so this
family re-expresses the measured favourite-overbet result (family 3: backing
the favourite at best-of-books prices is significantly negative in every
probability bucket). The new questions are the coherent-filter behaviour, the
commission load, and the gap to the sportsbook fallback (betting the opposite
side at its own vig-reduced price).

Prices: `odds_ml_bestbooks` (OddsPortal best-of-books decimal moneylines).
Outcomes: `build_games()`. Discovery 2015-16..2024-25, holdout 2025-26.
Bar |z| > 3.0 (Bonferroni floor for 12 arms).

Usage:  python scripts/backtest_lay_rules.py
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

BAR_Z = 3.0
COMMISSION = (0.02, 0.05)


def lay_profit(j: pd.DataFrame, lay_side: np.ndarray) -> np.ndarray:
    """Profit per unit of backer's stake when laying `lay_side` at its own odds.

    `lay_side` is True where the HOME team is laid. +1 when the laid team
    loses, -(O - 1) when it wins.
    """
    O_lay = np.where(lay_side, j["O1"], j["O2"]).astype(float)
    laid_won = np.where(lay_side, j["y"] == 1, j["y"] == 0)
    return np.where(laid_won, -(O_lay - 1.0), 1.0)


def settle_lay(j: pd.DataFrame, mask: np.ndarray, lay_home: bool | np.ndarray) -> dict:
    if np.isscalar(lay_home):
        lay_home = np.full(len(j), bool(lay_home))
    lay_home = np.asarray(lay_home, dtype=bool)
    raw = j[mask]
    keep = (raw["O1"].notna() & raw["O2"].notna()).to_numpy()
    sub = raw[keep]
    sides = lay_home[mask][keep]
    profit = lay_profit(sub, sides)
    out = es.boot(profit)
    fav_loses = ~np.where(sides, sub["y"] == 1, sub["y"] == 0)
    out["win_rate"] = float(fav_loses.mean()) if len(fav_loses) else float("nan")
    out["mean_lay_odds"] = float(np.where(sides, sub["O1"], sub["O2"]).mean())
    for c in COMMISSION:
        # exchange commission c on net market winnings (the +1 backer stake
        # when the lay is successful); losses carry no commission
        adj = np.where(profit > 0, profit - c * np.minimum(profit, 1.0), profit)
        out[f"commission_{int(c*100)}pct"] = es.boot(adj)
    coh = sub["coherent"].fillna(False).to_numpy()
    if coh.any():
        out["coherent"] = es.boot(lay_profit(sub[coh], sides[coh]))
    return out


def analyse(j: pd.DataFrame) -> tuple[list[dict], dict]:
    fav_home = j["fav_home"].to_numpy()
    devig_fav = np.maximum(j["p_fav_devig"], 1 - j["p_fav_devig"]).to_numpy(dtype=float)
    fav = np.minimum(j["O1"], j["O2"]).to_numpy(dtype=float)
    fav_won = np.where(fav_home, j["y"] == 1, j["y"] == 0)

    arms = []
    buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
               (0.70, 0.75), (0.75, 0.80), (0.80, 1.01)]
    for lo, hi in buckets:
        m = (devig_fav >= lo) & (devig_fav < hi)
        arms.append((f"L{len(arms)+1}: lay the favourite {lo:.2f}-{hi:.2f}",
                     m, fav_home))
    arms.append(("L8: lay the favourite priced <= 1.40", fav <= 1.40, fav_home))
    arms.append(("L9: lay the favourite priced <= 1.30", fav <= 1.30, fav_home))
    arms.append(("L10: lay the underdog (any price)", ~fav_home, ~fav_home))
    arms.append(("L11: lay the home favourite priced <= 1.30",
                 (fav <= 1.30) & fav_home, fav_home))
    arms.append(("L12: lay the road favourite priced <= 1.30",
                 (fav <= 1.30) & ~fav_home, fav_home))

    arm_rows = []
    for label, mask, lay_home in arms:
        d = settle_lay(j, mask & (j["split"] == "discovery").to_numpy(), lay_home)
        h = settle_lay(j, mask & (j["split"] == "holdout").to_numpy(), lay_home)
        if d["n"] and h["n"] and d["roi"] > 0 and abs(d["z"]) > BAR_Z and h["roi"] > 0:
            verdict = "candidate (see mirror-identity caveat)"
        elif d["n"] and d["roi"] > 0 and abs(d["z"]) > BAR_Z:
            verdict = "discovery-only"
        elif d["n"] and h["n"] and d["roi"] > 0 and h["roi"] > 0:
            verdict = "same sign, underpowered"
        else:
            verdict = "no evidence"
        arm_rows.append({"label": label, "kind": "ml_lay", "discovery": d,
                         "holdout": h, "verdict": verdict})

    # descriptive: the sportsbook fallback for the chalk cells - back the dog
    # at its own price, on the same games as L9.
    m9 = (fav <= 1.30) & (j["split"] == "discovery").to_numpy()
    dog_home = ~fav_home
    O_dog = np.where(dog_home, j["O1"], j["O2"]).astype(float)
    dog_won = np.where(dog_home, j["y"] == 1, j["y"] == 0)
    dog_profit = np.where(dog_won[m9], O_dog[m9] - 1.0, -1.0)
    lay_profit9 = lay_profit(j[m9], fav_home[m9])
    refs = {
        "sportsbook fallback for L9 (back the dog at its own price)":
            es.boot(dog_profit),
        "L9 lay (same games, for comparison)": es.boot(lay_profit9),
        "lay-advantage over the dog bet, mean per backer stake":
            float(lay_profit9.mean() - dog_profit.mean()),
    }
    return arm_rows, refs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "edge_search" / "lay_rules.json")
    args = ap.parse_args()

    j, frame_stats = es.load_bestprice_frame()
    j = j.copy()
    j["coherent"] = (1 / j["O1"] + 1 / j["O2"]) >= 1
    arm_rows, refs = analyse(j)

    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "family": "lay_rules",
        "issue": 95,
        "prereg": "E07-prereg-lay_rules",
        "n_arms": len(arm_rows),
        "bar_z": BAR_Z,
        "mirror_identity": "lay ROI = -back ROI at the same odds; candidates are "
                           "re-expressions of measured back results",
        "population": frame_stats,
        "arms": arm_rows,
        "references": refs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    run = {
        "name": "E07-run-lay_rules",
        "experiment_type": "search loop family 7: lay_rules",
        "declared_utc": payload["generated"],
        "meta": {"family": "lay_rules", "n_arms": len(arm_rows), "bar_z": BAR_Z,
                 "population": frame_stats},
        "arms": arm_rows,
        "references": refs,
        "status": "executed",
    }
    with open(ROOT / "results" / "experiments.jsonl", "a") as fh:
        fh.write(json.dumps(run, sort_keys=True) + "\n")

    print(f"lay_rules: {len(arm_rows)} arms, priced {frame_stats['priced_games']:,} "
          f"(discovery {frame_stats['discovery']:,}, holdout {frame_stats['holdout']:,}), "
          f"bar |z| > {BAR_Z}")
    for a in arm_rows:
        d, h = a["discovery"], a["holdout"]
        print(f"{a['label']:46s} disc n {d['n']:5d} {d['roi']:+8.2%} z{d['z']:+6.2f} "
              f"| hold n {h['n']:4d} {h['roi']:+8.2%} z{h['z']:+6.2f} | {a['verdict']}")
    for k, v in refs.items():
        if isinstance(v, dict):
            print(f"  ref {k}: n {v['n']:,} roi {v['roi']:+.2%} z {v['z']:+.2f}")
        else:
            print(f"  ref {k}: {v:+.4f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
