"""What does a team that wins its 4th straight game look like before tip-off?

EXPLORATORY (user-directed, 2026-10-27) - not a pre-registered family. Every
number below is computed strictly from PRE-GAME state; the game's own result
enters only as the label. The base population is every team-game where the
team enters on exactly a 3-game win streak (streaks reset at season
boundaries, so the 3 wins are same-season).

Questions answered:
1. How often does the 4th win happen, and is the streak priced? (calibration
   against the best-of-books moneyline in odds_ml_bestbooks)
2. Which pre-game features separate the teams that extend from the teams that
   stop at three? (Cohen's d, winners vs losers - exploratory, multiplicity
   applies: ~19 features, one comparison each)
3. Subgroup scan - also exploratory; with ~16 cuts one 2-sigma cell is
   expected by chance.

Prices: `odds_ml_bestbooks` (OddsPortal best-of-books decimal moneylines),
joined side-aware on the ET-normalised date. Outcomes come from
`build_games()`, never from the scrape.

Usage:  python scripts/explore_streak4.py [--out results/exploratory/streak4_profile.json]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

import closing_line as cl                      # noqa: E402
from DataProviders import nba_data as nd       # noqa: E402
from DataProviders.feat_team_elo import team_elo_features  # noqa: E402

PRICE_TABLE = "odds_ml_bestbooks"


def build_frame() -> pd.DataFrame:
    g = nd.build_games().reset_index(drop=True)
    sf = nd.streak_features(g)
    rt = nd.rest_and_travel(g)
    rf = nd.recent_form_features(g)
    elo = team_elo_features(g)

    rows = []
    for side in ("home", "away"):
        o = "away" if side == "home" else "home"
        rows.append(pd.DataFrame({
            "GAME_ID": g["GAME_ID"], "SEASON": g["SEASON"],
            "Date": pd.to_datetime(g["GAME_DATE"]).dt.strftime("%Y-%m-%d"),
            "team": g[side], "opp": g[o],
            "is_home": 1 if side == "home" else 0,
            "won": ((g["home_pts"] > g["away_pts"]) if side == "home"
                    else (g["away_pts"] > g["home_pts"])),
            "margin_for": ((g["home_pts"] - g["away_pts"]) if side == "home"
                           else (g["away_pts"] - g["home_pts"])),
            "win_streak": sf[f"{side}_streak_enter"],
            "opp_win_streak": sf[f"{o}_streak_enter"],
            "rest": rt[f"{side}_rest_days"], "opp_rest": rt[f"{o}_rest_days"],
            "b2b": rt[f"{side}_back_to_back"], "opp_b2b": rt[f"{o}_back_to_back"],
            "in3of4": rt[f"{side}_in_3_of_4"], "opp_in3of4": rt[f"{o}_in_3_of_4"],
            "travel_miles": rt["travel_miles"],
            "tz_crossed": rt["away_tz_crossed"] if side == "away" else 0.0,
            "elo_self": elo[f"{side}_elo"], "elo_opp": elo[f"{o}_elo"],
            "form10_self": rf[f"{side}_form10_enter"],
            "form10_opp": rf[f"{o}_form10_enter"],
        }))
    L = pd.concat(rows, ignore_index=True).sort_values(["team", "Date"]).reset_index(drop=True)
    L["gp_season"] = L.groupby(["team", "SEASON"]).cumcount()
    grp = L.groupby(["team", "SEASON"])
    pm = pd.concat([grp["margin_for"].shift(i) for i in (1, 2, 3)], axis=1)
    oe = pd.concat([grp["elo_opp"].shift(i) for i in (1, 2, 3)], axis=1)
    L["streak_margin"] = pm.mean(axis=1)
    L["streak_opp_elo"] = oe.mean(axis=1)

    three = L[L["win_streak"] == 3].copy()
    mapping = cl.tricode_to_full_name()
    mapping.pop("_alias", None)
    three["team_full"] = three["team"].map(mapping)
    three["opp_full"] = three["opp"].map(mapping)
    three["Home_key"] = np.where(three["is_home"] == 1, three["team_full"], three["opp_full"])
    three["Away_key"] = np.where(three["is_home"] == 1, three["opp_full"], three["team_full"])

    con = sqlite3.connect(f"file:{cl.ODDS_DB}?mode=ro", uri=True)
    try:
        o = pd.read_sql(f"SELECT Season, Date, Home, Away, O1, O2 FROM '{PRICE_TABLE}'", con)
    finally:
        con.close()
    o = o.rename(columns={"Season": "SEASON"}).drop_duplicates(["Date", "Home", "Away"])
    three = three.merge(o[["SEASON", "Date", "Home", "Away", "O1", "O2"]],
                        left_on=["SEASON", "Date", "Home_key", "Away_key"],
                        right_on=["SEASON", "Date", "Home", "Away"], how="left")
    three["O_self"] = np.where(three["is_home"] == 1, three["O1"], three["O2"])
    three["O_opp"] = np.where(three["is_home"] == 1, three["O2"], three["O1"])
    three["implied"] = 1 / three["O_self"]
    three["devig"] = (1 / three["O_self"]) / (1 / three["O_self"] + 1 / three["O_opp"])
    three["is_fav"] = three["O_self"] <= three["O_opp"]
    three["coherent"] = (1 / three["O1"] + 1 / three["O2"]) >= 1
    return three.reset_index(drop=True)


def cohen_d(x: pd.Series, y: pd.Series) -> float:
    return float((x.mean() - y.mean())
                 / np.sqrt((x.std(ddof=1) ** 2 + y.std(ddof=1) ** 2) / 2))


def analyse(j: pd.DataFrame) -> dict:
    n = len(j)
    out = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "exploratory": True,
        "population": "team-games entering on exactly a 3-game win streak, 2015-16..2025-26",
        "n": n,
        "overall_win_rate": round(float(j["won"].mean()), 4),
        "by_venue": {str(k): {"n": int(len(x)), "win_rate": round(float(x["won"].mean()), 4)}
                     for k, x in j.groupby("is_home")},
        "by_season": {str(k): round(float(x["won"].mean()), 4)
                      for k, x in j.groupby("SEASON")},
        "calibration": [],
        "betting": {},
        "feature_profile": [],
        "subgroups": {},
    }

    priced = j.dropna(subset=["implied"]).copy()
    priced["bucket"] = pd.cut(priced["devig"], [0.2, 0.4, 0.5, 0.6, 0.7, 0.9])
    for b, x in priced.groupby("bucket", observed=True):
        out["calibration"].append({
            "devig_bucket": str(b), "n": int(len(x)),
            "implied": round(float(x["devig"].mean()), 4),
            "actual": round(float(x["won"].mean()), 4),
            "roi_betting_team": round(float(np.where(x["won"], x["O_self"] - 1, -1).mean()), 4),
        })
    prof = np.where(priced["won"], priced["O_self"] - 1, -1.0)
    se = float(prof.std(ddof=1) / np.sqrt(len(prof)))
    out["betting"]["all_priced"] = {"n": int(len(priced)), "roi": round(float(prof.mean()), 4),
                                    "z": round(float(prof.mean() / se), 2)}
    coh = priced[priced["coherent"]]
    profc = np.where(coh["won"], coh["O_self"] - 1, -1.0)
    sec = float(profc.std(ddof=1) / np.sqrt(len(profc)))
    out["betting"]["coherent_only"] = {"n": int(len(coh)), "roi": round(float(profc.mean()), 4),
                                       "z": round(float(profc.mean() / sec), 2)}
    out["betting"]["implied_mean"] = round(float(priced["devig"].mean()), 4)
    out["betting"]["actual_mean"] = round(float(priced["won"].mean()), 4)

    W, L = j[j["won"]], j[~j["won"]]
    feats = ["is_home", "rest", "opp_rest", "b2b", "opp_b2b", "in3of4", "opp_in3of4",
             "travel_miles", "tz_crossed", "elo_self", "elo_opp", "form10_self",
             "form10_opp", "opp_win_streak", "streak_margin", "streak_opp_elo",
             "gp_season", "devig", "is_fav"]
    profile = []
    for f in feats:
        x, y = W[f].dropna().astype(float), L[f].dropna().astype(float)
        if len(x) < 30 or x.std(ddof=1) == 0:
            continue
        profile.append({"feature": f, "mean_won": round(float(x.mean()), 3),
                        "mean_lost": round(float(y.mean()), 3),
                        "d": round(cohen_d(x, y), 3)})
    profile.sort(key=lambda r: -abs(r["d"]))
    out["feature_profile"] = profile
    out["d_significance_note"] = (
        f"SE of d ≈ {np.sqrt(2 / (n / 2)):.3f}; with ~{len(feats)} comparisons, "
        "|d| > ~0.13 survives a 99% single-comparison test; multiplicity applies")

    cuts = [
        ("home", j["is_home"] == 1), ("away", j["is_home"] == 0),
        ("opponent on B2B", j["opp_b2b"] == 1),
        ("own B2B", j["b2b"] == 1),
        ("opponent entering win streak >= 2", j["opp_win_streak"] >= 2),
        ("opponent entering win streak <= 1", j["opp_win_streak"] < 2),
        ("streak margin >= median", j["streak_margin"] >= j["streak_margin"].median()),
        ("streak margin < median", j["streak_margin"] < j["streak_margin"].median()),
        ("streak opponents >= median elo", j["streak_opp_elo"] >= j["streak_opp_elo"].median()),
        ("streak opponents < median elo", j["streak_opp_elo"] < j["streak_opp_elo"].median()),
        ("own rest <= 1", j["rest"] <= 1), ("own rest >= 2", j["rest"] >= 2),
        ("opponent rest <= 1", j["opp_rest"] <= 1), ("opponent rest >= 2", j["opp_rest"] >= 2),
        ("market favourite", j["is_fav"] == 1), ("market underdog", j["is_fav"] == 0),
    ]
    for name, mask in cuts:
        x = j[mask]
        if len(x) < 40:
            continue
        out["subgroups"][name] = {"n": int(len(x)),
                                  "win_rate": round(float(x["won"].mean()), 3)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results" / "exploratory" / "streak4_profile.json")
    args = ap.parse_args()
    j = build_frame()
    out = analyse(j)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"n {out['n']} | overall win rate {out['overall_win_rate']:.4f} "
          f"| priced ROI {out['betting']['all_priced']['roi']:+.4f} "
          f"(coherent {out['betting']['coherent_only']['roi']:+.4f})")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
