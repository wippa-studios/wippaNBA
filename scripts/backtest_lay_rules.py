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
from backtest_user_rules import build_team_features, assemble  # noqa: E402

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
    mean_odds = float(np.where(sides, sub["O1"], sub["O2"]).mean())
    out["mean_lay_odds"] = mean_odds
    out["mean_liability"] = float(mean_odds - 1.0)
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


def rules_grid(F: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Family 8: the lay grid beyond favourites (24 arms, issue #97).

    Every arm lays a named side at its own best-of-books price. L13-L16 are
    the "no favs" near-pick'em lays; L17-L22 lay underdogs by bucket and
    price; L23-L26 lay streak/letdown teams; L27-L30 lay fatigued teams;
    L31-L32 are the venue lays; L33-L36 are price-band and form-gated lays.
    """
    fh, fa = F["home_form10"], F["away_form10"]
    home_rest, away_rest = F["home_rest"], F["away_rest"]
    home_b2b, away_b2b = F["home_b2b"], F["away_b2b"]
    O1, O2 = F["O1"], F["O2"]
    devig_home = np.where(F["fav_home"], F["p_fav_devig"], 1 - F["p_fav_devig"])
    devig_dog = 1 - devig_home
    dog_home = ~F["fav_home"]
    O_dog = np.where(dog_home, O1, O2)
    O_fav = np.where(dog_home, O2, O1)
    arms = []

    def add(label, mask, lay_home):
        if np.isscalar(lay_home):
            lay_home = np.full(len(F), bool(lay_home))
        if isinstance(mask, pd.Series):
            mask = mask.fillna(False).to_numpy()
        arms.append((label, np.asarray(mask, dtype=bool), np.asarray(lay_home, dtype=bool)))

    tight = (O1 - O2).abs() <= 0.20
    add("L13 lay the away side in near-pick'em (devig home 0.50-0.55)",
        (devig_home >= 0.50) & (devig_home < 0.55), False)
    add("L14 lay the home side in near-pick'em (devig home 0.45-0.50)",
        (devig_home >= 0.45) & (devig_home < 0.50), True)
    add("L15 lay the home side in tight-price games", tight, True)
    add("L16 lay the away side in tight-price games", tight, False)
    add("L17 lay the underdog, devig-dog 0.20-0.30",
        (devig_dog >= 0.20) & (devig_dog < 0.30), dog_home)
    add("L18 lay the underdog, devig-dog 0.30-0.40",
        (devig_dog >= 0.30) & (devig_dog < 0.40), dog_home)
    add("L19 lay the underdog, devig-dog 0.40-0.50",
        (devig_dog >= 0.40) & (devig_dog < 0.50), dog_home)
    add("L20 lay underdogs priced >= 2.50", (O_dog >= 2.50), dog_home)
    add("L21 lay underdogs priced >= 3.00", (O_dog >= 3.00), dog_home)
    add("L22 lay underdogs priced >= 4.00", (O_dog >= 4.00), dog_home)
    r23_home = F["home_win_streak"] >= 3
    r23_away = F["away_win_streak"] >= 3
    add("L23 lay the team on a 3+ game win streak", r23_home | r23_away, r23_home)
    r24_home = F["home_win_streak"] >= 4
    r24_away = F["away_win_streak"] >= 4
    add("L24 lay the team on a 4+ game win streak", r24_home | r24_away, r24_home)
    r25_home = F["home_prev_margin"] >= 15
    r25_away = F["away_prev_margin"] >= 15
    add("L25 lay the team off a 15+ point win", r25_home | r25_away, r25_home)
    r26_home = F["home_prev_margin"] >= 20
    r26_away = F["away_prev_margin"] >= 20
    add("L26 lay the team off a 20+ point win", r26_home | r26_away, r26_home)
    add("L27 lay the B2B team (either venue)", (home_b2b == 1) | (away_b2b == 1),
        away_b2b == 1)
    add("L28 lay the away team on a road B2B",
        (away_b2b == 1) & (F["away_prev_is_home"] == 0), True)
    r29_home = F["home_games_in_last_5"] == 3
    r29_away = F["away_games_in_last_5"] == 3
    add("L29 lay the team on its 4th game in 6 nights", r29_home | r29_away, r29_home)
    r30_home = home_rest <= away_rest - 1
    r30_away = away_rest <= home_rest - 1
    add("L30 lay the team with >=1 day less rest", r30_home | r30_away, r30_home)
    add("L31 lay the road team (any game)", np.ones(len(F), bool), False)
    add("L32 lay the home team (any game)", np.ones(len(F), bool), True)
    add("L33 lay the road favourite priced 1.30-1.70",
        (F["O2"] < F["O1"]) & (O2 >= 1.30) & (O2 <= 1.70), False)
    add("L34 lay the 7-3+ team vs a 3-7- opponent",
        ((fh >= 7) & (fa <= 3)) | ((fa >= 7) & (fh <= 3)),
        (fh >= 7) & (fa <= 3))
    add("L35 lay the 6-4+ team vs a 4-6- opponent",
        ((fh >= 6) & (fa <= 4)) | ((fa >= 6) & (fh <= 4)),
        (fh >= 6) & (fa <= 4))
    add("L36 lay the 8-2+ team vs a 2-8- opponent",
        ((fh >= 8) & (fa <= 2)) | ((fa >= 8) & (fh <= 2)),
        (fh >= 8) & (fa <= 2))
    assert len(arms) == 24, len(arms)
    return arms


def away_fav_expr(F):
    return F["O2"] < F["O1"]


def run_lay_grid(F: pd.DataFrame, out: Path) -> dict:
    """Family 8: the 24-arm lay grid (issue #97) — the expansion beyond favourites."""
    F = F[F["split"] != "other"].reset_index(drop=True)
    priced = F.dropna(subset=["O1", "O2"]).reset_index(drop=True)
    arm_rows = []
    for label, mask, lay_home in rules_grid(priced):
        d = settle_lay(priced, (mask & (priced["split"] == "discovery").to_numpy()), lay_home)
        h = settle_lay(priced, (mask & (priced["split"] == "holdout").to_numpy()), lay_home)
        if d["n"] and h["n"] and d["roi"] > 0 and abs(d["z"]) > BAR_Z and h["roi"] > 0:
            verdict = "candidate (exchange-only; vig-capture + mirror caveat)"
        elif d["n"] and d["roi"] > 0 and abs(d["z"]) > BAR_Z:
            verdict = "discovery-only"
        elif d["n"] and h["n"] and d["roi"] > 0 and h["roi"] > 0:
            verdict = "same sign, underpowered"
        else:
            verdict = "no evidence"
        arm_rows.append({"label": label, "kind": "ml_lay", "discovery": d,
                         "holdout": h, "verdict": verdict})
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "family": "lay_grid",
        "issue": 97,
        "prereg": "E08-prereg-lay_grid",
        "n_arms": len(arm_rows),
        "bar_z": BAR_Z,
        "mirror_note": "every lay arm is the exact mirror of backing the same "
                       "team at the same price; surpluses are vig-capture net "
                       "of the mirror, executable only on an exchange",
        "population": {"priced_games": int(len(priced)),
                       "discovery": int((priced["split"] == "discovery").sum()),
                       "holdout": int((priced["split"] == "holdout").sum())},
        "arms": arm_rows,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    run = {
        "name": "E08-run-lay_grid",
        "experiment_type": "search loop family 8: lay_grid",
        "declared_utc": payload["generated"],
        "meta": {"family": "lay_grid", "n_arms": len(arm_rows), "bar_z": BAR_Z,
                 "population": payload["population"]},
        "arms": arm_rows,
        "status": "executed",
    }
    with open(ROOT / "results" / "experiments.jsonl", "a") as fh:
        fh.write(json.dumps(run, sort_keys=True) + "\n")
    print(f"lay_grid: {len(arm_rows)} arms, priced {len(priced):,} "
          f"(discovery {payload['population']['discovery']:,}, "
          f"holdout {payload['population']['holdout']:,}), bar |z| > {BAR_Z}")
    for a in arm_rows:
        d, h = a["discovery"], a["holdout"]
        print(f"{a['label']:54s} disc n {d['n']:5d} {d['roi']:+8.2%} z{d['z']:+6.2f} "
              f"| hold n {h['n']:4d} {h['roi']:+8.2%} | {a['verdict']}")
    print(f"wrote {out}")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="lay_rules",
                    choices=["lay_rules", "lay_grid"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if args.out is None:
        args.out = ROOT / "results" / "edge_search" / f"{args.family}.json"

    if args.family == "lay_grid":
        L, g = build_team_features()
        F = assemble(L, g)
        F["fav_home"] = F["O1"] <= F["O2"]
        F["p_fav_devig"] = np.where(
            F["fav_home"], (1 / F["O1"]) / (1 / F["O1"] + 1 / F["O2"]),
            (1 / F["O2"]) / (1 / F["O1"] + 1 / F["O2"]))
        F["coherent"] = (1 / F["O1"] + 1 / F["O2"]) >= 1
        F["split"] = np.where(F["SEASON"] < es.HOLDOUT, "discovery",
                              np.where(F["SEASON"] == es.HOLDOUT, "holdout", "other"))
        run_lay_grid(F, args.out)
        return 0

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
