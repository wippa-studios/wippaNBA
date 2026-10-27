"""Backtest 13 user-supplied form/schedule rules — family 5, pre-registered.

The rules arrived before any number was computed (issue #91, ledger
`E05-prereg-user_rules`), which makes this a genuine pre-registration. Every
rule uses only odds, schedule and form; every feature is strictly pre-game.
Settlement is the best-of-books decimal moneyline for the backed side
(O-1 on a win, -1 on a loss, no pushes), with the coherent-row ROI reported
alongside as the family-3 artifact diagnostic.

Rule definitions are pinned in the ledger record and in
`results/edge_search/user_rules.json`; the implementation below is the
executable form. Discovery 2015-16..2024-25, holdout 2025-26, bar |z| > 3.0.

Usage:  python scripts/backtest_user_rules.py
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
import edge_search as es                       # noqa: E402
from DataProviders import nba_data as nd       # noqa: E402

PRICE_TABLE = "odds_ml_bestbooks"
BAR_Z = 3.0


def entering_run(won: pd.Series, target: bool) -> np.ndarray:
    """Consecutive-run length of `target` immediately BEFORE each row."""
    out = np.zeros(len(won), dtype=int)
    run = 0
    for i, w in enumerate(won.to_numpy()):
        out[i] = run
        run = run + 1 if bool(w) == target else 0
    return out


def build_team_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-team-game causal features (long frame) and the per-game price frame."""
    g = nd.build_games().reset_index(drop=True)
    g["Date"] = pd.to_datetime(g["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    sf = nd.streak_features(g)
    rt = nd.rest_and_travel(g)
    rf = nd.recent_form_features(g)

    tlogs = nd.load_team_games("Regular Season")
    min_map = {(r.TEAM_ABBREVIATION, r.GAME_ID): float(r.MIN)
               for r in tlogs.itertuples()}

    rows = []
    for side in ("home", "away"):
        o = "away" if side == "home" else "home"
        rows.append(pd.DataFrame({
            "GAME_ID": g["GAME_ID"], "SEASON": g["SEASON"], "Date": g["Date"],
            "team": g[side], "opp": g[o], "is_home": int(side == "home"),
            "won": ((g["home_pts"] > g["away_pts"]) if side == "home"
                    else (g["away_pts"] > g["home_pts"])),
            "margin_for": ((g["home_pts"] - g["away_pts"]) if side == "home"
                           else (g["away_pts"] - g["home_pts"])),
            "form10": rf[f"{side}_form10_enter"],
            "rest": rt[f"{side}_rest_days"], "b2b": rt[f"{side}_back_to_back"],
            "opp_rest": rt[f"{o}_rest_days"], "opp_b2b": rt[f"{o}_back_to_back"],
            "loss_streak": sf[f"{side}_loss_streak_enter"],
            "win_streak": sf[f"{side}_streak_enter"],
            "opp_loss_streak": sf[f"{o}_loss_streak_enter"],
            "in3of4": rt[f"{side}_in_3_of_4"],
        }))
    L = pd.concat(rows, ignore_index=True).sort_values(["team", "Date"]).reset_index(drop=True)
    L["MIN"] = [min_map.get((t, gid), np.nan)
                for t, gid in zip(L["team"], L["GAME_ID"])]

    L["gp_season"] = L.groupby(["team", "SEASON"]).cumcount()
    grp = L.groupby(["team", "SEASON"], sort=False)
    L["prev_margin"] = grp["margin_for"].shift(1)
    L["prev_is_home"] = grp["is_home"].shift(1)
    L["prev_MIN"] = grp["MIN"].shift(1)
    L["season_win_pct"] = (grp["won"].transform(lambda s: s.shift(1).expanding().mean()))

    # 4-in-6: prior games within the previous 5 calendar days
    L["games_in_last_5"] = 0
    d_num = (pd.to_datetime(L["Date"]) - pd.Timestamp("2000-01-01")).dt.days.to_numpy()
    for team, tdf in L.groupby("team", sort=False):
        d = d_num[tdf.index.to_numpy()]
        order = np.argsort(d, kind="stable")
        d_sorted = d[order]
        lo = np.searchsorted(d_sorted, d - 5, side="left")
        hi = np.searchsorted(d_sorted, d, side="left")
        L.loc[tdf.index, "games_in_last_5"] = (hi - lo)[np.argsort(order, kind="stable")]

    # venue-split entering streaks (within season)
    L["home_win_run_enter"] = 0
    L["road_loss_run_enter"] = 0
    L["away_run_enter"] = 0
    for (team, season), tdf in L.groupby(["team", "SEASON"], sort=False):
        tdf = tdf.sort_values("Date")
        home_rows = tdf[tdf["is_home"] == 1]
        road_rows = tdf[tdf["is_home"] == 0]
        L.loc[home_rows.index, "home_win_run_enter"] = entering_run(home_rows["won"], True)
        L.loc[road_rows.index, "road_loss_run_enter"] = entering_run(road_rows["won"], False)
        L.loc[tdf.index, "away_run_enter"] = entering_run(tdf["is_home"], False)

    # opponent road form: last 10 road games entering (per team, within season)
    L["road_games_enter"] = 0
    L["road_wins_last10"] = np.nan
    for (team, season), tdf in L.groupby(["team", "SEASON"], sort=False):
        road = tdf[tdf["is_home"] == 0].sort_values("Date")
        if not len(road):
            continue
        wins = road["won"].to_numpy()
        counts = np.arange(1, len(road) + 1)
        csum = np.cumsum(wins)
        enter_n = counts - 1                     # road games before this one
        enter_wins = csum - wins                 # road wins before this one
        # wins in the previous min(10, enter_n) road games:
        last10 = np.full(len(road), np.nan)
        for i in range(len(road)):
            k = min(10, i)
            if k > 0:
                last10[i] = wins[max(0, i - 10):i].sum()
        L.loc[road.index, "road_games_enter"] = enter_n
        L.loc[road.index, "road_wins_last10"] = last10

    # H2H: last 4 meetings between the pair (any venue, all prior)
    g["pair"] = ["".join(sorted([h, a])) for h, a in zip(g["home"], g["away"])]
    g["h2h_home_wins_last4"] = 0
    g["h2h_n"] = 0
    for pair, pdf in g.groupby("pair", sort=False):
        pdf = pdf.sort_values("Date")
        winners = np.where(pdf["home_pts"].to_numpy() > pdf["away_pts"].to_numpy(),
                           pdf["home"].to_numpy(), pdf["away"].to_numpy())
        for i in range(len(pdf)):
            prior = winners[max(0, i - 4):i]
            g.loc[pdf.index[i], "h2h_n"] = len(prior)
            if len(prior) == 4:
                g.loc[pdf.index[i], "h2h_home_wins_last4"] = int((prior == pdf["home"].iloc[i]).sum())

    return L, g


def assemble(L: pd.DataFrame, g: pd.DataFrame) -> pd.DataFrame:
    """Pivot team features to a per-game frame and attach best-of-books prices."""
    piv = L.pivot(index="GAME_ID", columns="is_home",
                  values=["form10", "rest", "b2b", "opp_rest", "opp_b2b", "loss_streak",
                          "win_streak", "prev_margin", "prev_is_home", "prev_MIN",
                          "season_win_pct", "games_in_last_5", "home_win_run_enter",
                          "road_loss_run_enter", "away_run_enter", "road_games_enter",
                          "road_wins_last10"])
    F = pd.DataFrame(index=piv.index)
    for feat in piv.columns.levels[0]:
        F[f"home_{feat}"] = piv[feat][1]
        F[f"away_{feat}"] = piv[feat][0]
    F = g.set_index("GAME_ID")[["SEASON", "Date", "home", "away", "y",
                                "h2h_home_wins_last4", "h2h_n"]].join(F)
    mapping = cl.tricode_to_full_name()
    mapping.pop("_alias", None)
    F["Home_key"] = F["home"].map(mapping)
    F["Away_key"] = F["away"].map(mapping)
    con = sqlite3.connect(f"file:{cl.ODDS_DB}?mode=ro", uri=True)
    try:
        o = pd.read_sql(f"SELECT Season, Date, Home, Away, O1, O2 FROM '{PRICE_TABLE}'", con)
    finally:
        con.close()
    o = o.rename(columns={"Season": "SEASON"}).drop_duplicates(["Date", "Home", "Away"])
    F = F.reset_index().merge(o, left_on=["SEASON", "Date", "Home_key", "Away_key"],
                              right_on=["SEASON", "Date", "Home", "Away"], how="left")
    F["coherent"] = (1 / F["O1"] + 1 / F["O2"]) >= 1
    F["split"] = np.where(F["SEASON"] < es.HOLDOUT, "discovery",
                          np.where(F["SEASON"] == es.HOLDOUT, "holdout", "other"))
    return F


def rules_13(F: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """(label, mask, bet_home) for the 13 family-5 rules."""
    fh, fa = F["home_form10"], F["away_form10"]
    home_rest, away_rest = F["home_rest"], F["away_rest"]
    home_b2b, away_b2b = F["home_b2b"], F["away_b2b"]
    O1, O2 = F["O1"], F["O2"]
    home_fav = O1 < O2
    away_fav = O2 < O1
    out = []

    def add(label, mask, bet_home):
        if np.isscalar(bet_home):
            bet_home = np.full(len(F), bool(bet_home))
        out.append((label, mask.fillna(False).to_numpy(), np.asarray(bet_home, dtype=bool)))

    add("R1 home fav 7-3+ vs 3-7- away on B2B",
        home_fav & (fh >= 7) & (fa <= 3) & (away_b2b == 1), True)
    add("R2 road fav 8-2+ vs home 3-7-",
        away_fav & (fa >= 8) & (fh <= 3), False)
    add("R3 home dog 6-4+ vs fav 4-6-",
        (O1 > O2) & (fh >= 6) & (fa <= 4), True)
    add("R4 home 4+ straight home wins vs opp 4+ straight road losses",
        (F["home_home_win_run_enter"] >= 4) & (F["away_road_loss_run_enter"] >= 4), True)
    add("R5 home fav <= 1.40, 7-3+, opp road form <= 2-8",
        (O1 <= 1.40) & (fh >= 7) & (F["away_road_games_enter"] >= 10)
        & (F["away_road_wins_last10"] <= 2), True)
    rested_home = (home_rest >= 2) & (away_b2b == 1) & (fh >= 5)
    rested_away = (away_rest >= 2) & (home_b2b == 1) & (fa >= 5)
    add("R6 rested (2+d) 5-5+ team vs B2B opponent",
        rested_home | rested_away, rested_home)
    add("R7 home off 15+ pt loss vs opp road B2B",
        (F["home_prev_margin"] <= -15) & (away_b2b == 1) & (F["away_prev_is_home"] == 0), True)
    add("R8 home opp of road fav on 4th game in 6 nights",
        away_fav & (F["away_games_in_last_5"] == 3), True)
    add("R9 home returning from 4+ game road trip, 6-4+",
        (F["home_away_run_enter"] >= 4) & (fh >= 6), True)
    add("R10 home opp of road-B2B team off a 2OT+ game",
        (away_b2b == 1) & (F["away_prev_is_home"] == 0) & (F["away_prev_MIN"] >= 250), True)
    add("R11 home on 3-game losing streak, season win% > .550",
        (F["home_loss_streak"] == 3) & (F["home_season_win_pct"] > 0.55), True)
    add("R12 home off 20+ pt loss, 6-4+",
        (F["home_prev_margin"] <= -20) & (fh >= 6), True)
    add("R13 home dog that won 3 of last 4 meetings",
        (O1 > O2) & (F["h2h_n"] >= 4) & (F["h2h_home_wins_last4"] >= 3), True)
    return out


def rules_50(F: pd.DataFrame) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The family-5 grid expanded to 50 systematic rules (issue #93).

    R1-R13 are the family-5 rules unchanged (their numbers must reproduce
    family 5 exactly - deterministic). R14-R50 are the systematic expansion:
    schedule events (B2B, road B2B, 4-in-6, road trips, 2OT legs, letdowns)
    crossed with form gates (none / 5-5+ / 6-4+) and venue/price roles.
    """
    arms = rules_13(F)
    fh, fa = F["home_form10"], F["away_form10"]
    home_rest, away_rest = F["home_rest"], F["away_rest"]
    home_b2b, away_b2b = F["home_b2b"], F["away_b2b"]
    O1, O2 = F["O1"], F["O2"]
    home_fav = O1 < O2
    away_fav = O2 < O1
    home_dog = O1 > O2
    rest_diff = F["home_rest"] - F["away_rest"]
    opp_prev_won = F["away_prev_margin"] > 0      # away team won last night
    opp_prev_lost = F["away_prev_margin"] < 0

    def add(label, mask, bet_home):
        if np.isscalar(bet_home):
            bet_home = np.full(len(F), bool(bet_home))
        arms.append((label, mask.fillna(False).to_numpy(), np.asarray(bet_home, dtype=bool)))

    # --- opponent B2B, rested side (any venue) ---
    r14_home = (away_b2b == 1) & (home_rest >= 2)
    r14_away = (home_b2b == 1) & (away_rest >= 2)
    add("R14 opp B2B, backed rest>=2 (either venue)", r14_home | r14_away, r14_home)
    add("R15 opp B2B, backed rest>=2, backed 6-4+", r14_home & (fh >= 6), True)
    add("R16 opp B2B, backed rest>=3", (away_b2b == 1) & (home_rest >= 3), True)
    # --- road B2B ---
    add("R17 opp away on road B2B (plain)", (away_b2b == 1) & (F["away_prev_is_home"] == 0), True)
    add("R18 opp away on road B2B, home 5-5+", (away_b2b == 1) & (F["away_prev_is_home"] == 0) & (fh >= 5), True)
    add("R19 opp away on road B2B, home 6-4+", (away_b2b == 1) & (F["away_prev_is_home"] == 0) & (fh >= 6), True)
    add("R20 opp away on road B2B, home rest>=2",
        (away_b2b == 1) & (F["away_prev_is_home"] == 0) & (home_rest >= 2), True)
    # --- 4-in-6 ---
    r21_home_opp = (F["away_games_in_last_5"] == 3) & (home_rest >= 2)
    r21_away_opp = (F["home_games_in_last_5"] == 3) & (away_rest >= 2)
    add("R21 opp on 4-in-6, backed rest>=2 (either venue)",
        r21_home_opp | r21_away_opp, r21_home_opp)
    add("R22 opp on 4-in-6, backed 5-5+ (either venue)",
        ((F["away_games_in_last_5"] == 3) & (fh >= 5)) |
        ((F["home_games_in_last_5"] == 3) & (fa >= 5)),
        (F["away_games_in_last_5"] == 3))
    add("R23 opp on 4-in-6 playing away (plain)", F["away_games_in_last_5"] == 3, True)
    add("R24 opp away on road B2B and 4-in-6",
        (away_b2b == 1) & (F["away_prev_is_home"] == 0) & (F["away_games_in_last_5"] == 3), True)
    # --- B2B previous-game splits ---
    add("R25 opp away on B2B that won last night; bet home",
        (away_b2b == 1) & (F["away_prev_is_home"] == 0) & opp_prev_won, True)
    add("R26 opp away on B2B that lost last night; bet home",
        (away_b2b == 1) & (F["away_prev_is_home"] == 0) & opp_prev_lost, True)
    # --- road trips ---
    add("R27 home returning from 4+ road trip, opp rest>=2",
        (F["home_away_run_enter"] >= 4) & (away_rest >= 2), True)
    add("R28 home returning from 4+ road trip (plain)",
        F["home_away_run_enter"] >= 4, True)
    # --- streaks ---
    add("R29 home win streak>=3 vs opp loss streak>=3",
        (F["home_win_streak"] >= 3) & (F["away_loss_streak"] >= 3), True)
    r30_home = F["home_win_streak"] >= 4
    r30_away = F["away_win_streak"] >= 4
    add("R30 any team win streak>=4", r30_home | r30_away, r30_home)
    # --- losing streaks ---
    add("R31 home loss streak==3, season win%>.500",
        (F["home_loss_streak"] == 3) & (F["home_season_win_pct"] > 0.50), True)
    add("R32 both teams on 3+ game losing streaks; bet home",
        (F["home_loss_streak"] >= 3) & (F["away_loss_streak"] >= 3), True)
    # --- bounce-back / letdown ---
    add("R33 home off a 10+ pt loss (plain)", F["home_prev_margin"] <= -10, True)
    add("R34 home off a 10+ pt loss, 5-5+", (F["home_prev_margin"] <= -10) & (fh >= 5), True)
    r35_home = F["home_prev_margin"] >= 15   # home won big last game -> fade home, bet away
    r35_away = F["away_prev_margin"] >= 15   # away won big last game -> bet home
    add("R35 opp off a 15+ pt win (letdown, either venue)", r35_home | r35_away, r35_away)
    # --- road favourite on a B2B ---
    add("R36 away fav on B2B; bet home (plain)", away_fav & (away_b2b == 1), True)
    add("R37 away fav on B2B, home 5-5+", away_fav & (away_b2b == 1) & (fh >= 5), True)
    # --- dog + schedule ---
    add("R38 home dog vs opp B2B", home_dog & (away_b2b == 1), True)
    add("R39 home dog vs opp on 4-in-6", home_dog & (F["away_games_in_last_5"] == 3), True)
    big_dog_home = (O1 >= 2.50)
    big_dog_away = (O2 >= 2.50)
    add("R40 dog >=2.50 vs opp B2B",
        (big_dog_home & (away_b2b == 1)) | (big_dog_away & (home_b2b == 1)), big_dog_home)
    # --- price roles ---
    add("R41 favourite <=1.40 (plain)", O1 <= 1.40, True)
    add("R42 favourite <=1.30", O1 <= 1.30, True)
    add("R43 favourite <=1.40 vs opp B2B", (O1 <= 1.40) & (away_b2b == 1), True)
    add("R44 road underdogs (plain)", (O2 > O1), False)
    add("R45 road underdogs 5-5+", (O2 > O1) & (fa >= 5), False)
    add("R46 home underdogs 5-5+", (O1 > O2) & (fh >= 5), True)
    near_form = (fh - fa).abs() <= 1
    r47_home = home_dog & near_form
    r47_away = away_fav & near_form   # away team is the underdog when the home side is fav
    add("R47 underdog when form10 differs by <=1 (either venue)",
        r47_home | r47_away, r47_home)
    add("R48 rest edge >=1 day, both 5-5+",
        (rest_diff >= 1) & (fh >= 5) & (fa >= 5), True)
    add("R49 opp away on 3rd+ consecutive road game",
        (F["away_away_run_enter"] >= 3), True)
    add("R50 opp away on 4+ consecutive road game",
        (F["away_away_run_enter"] >= 4), True)
    assert len(arms) == 50, len(arms)
    return arms


FAMILIES = {
    "user_rules": rules_13,
    "user_rules_50": rules_50,
}

def family_defs():
    return {
        "user_rules": {"ledger": "E05", "issue": 91, "n": 13, "bar": 3.0},
        "user_rules_50": {"ledger": "E06", "issue": 93, "n": 50, "bar": 3.0},
    }


def settle(F: pd.DataFrame, mask: np.ndarray, bet_home) -> dict:
    """Settle the rows selected by `mask` at the best-of-books ML.

    `bet_home` is a scalar or a boolean array aligned with F. Rows without
    both prices are dropped, and the side array is filtered in lock-step so
    the alignment can never drift.
    """
    if np.isscalar(bet_home):
        bet_home = np.full(len(F), bool(bet_home))
    bet_home = np.asarray(bet_home, dtype=bool)
    raw = F[mask]
    sides = bet_home[mask]
    keep = (raw["O1"].notna() & raw["O2"].notna()).to_numpy()
    sub = raw[keep]
    sides = sides[keep]
    O = np.where(sides, sub["O1"].to_numpy(dtype=float),
                 sub["O2"].to_numpy(dtype=float))
    won = np.where(sides, sub["y"] == 1, sub["y"] == 0)
    profit = np.where(won, O - 1.0, -1.0)
    out = es.boot(profit)
    out["win_rate"] = float(won.mean()) if len(won) else float("nan")
    coh = sub["coherent"].fillna(False).to_numpy()
    if coh.any():
        Oc = np.where(sides[coh], sub.loc[coh, "O1"].to_numpy(dtype=float),
                      sub.loc[coh, "O2"].to_numpy(dtype=float))
        wc = np.where(sides[coh], sub.loc[coh, "y"] == 1, sub.loc[coh, "y"] == 0)
        pc = np.where(wc, Oc - 1.0, -1.0)
        out["coherent"] = es.boot(pc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="user_rules",
                    choices=["user_rules", "user_rules_50"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--preregister", action="store_true")
    args = ap.parse_args()
    if args.out is None:
        args.out = ROOT / "results" / "edge_search" / f"{args.family}.json"
    if args.preregister:
        raise SystemExit("the pre-registration lives in results/experiments.jsonl "
                         "(E05-prereg-user_rules / E06-prereg-user_rules_50); "
                         "nothing to append here")

    L, g = build_team_features()
    F = assemble(L, g)
    F = F[F["split"] != "other"].reset_index(drop=True)
    priced = F.dropna(subset=["O1", "O2"])
    arm_rows, overlap = [], {}
    rule_fn = {"user_rules": rules_13, "user_rules_50": rules_50}[args.family]
    for label, mask, bet_home in rule_fn(F):
        full = pd.Series(mask, index=F.index)
        d = settle(F, (mask & (F["split"] == "discovery").to_numpy()), bet_home)
        h = settle(F, (mask & (F["split"] == "holdout").to_numpy()), bet_home)
        verdict = ("candidate" if d["n"] and h["n"] and d["roi"] > 0
                   and abs(d["z"]) > BAR_Z and h["roi"] > 0 else
                   "discovery-only" if d["n"] and d["roi"] > 0 and abs(d["z"]) > BAR_Z else
                   "same sign, underpowered" if d["n"] and h["n"] and d["roi"] > 0 and h["roi"] > 0
                   else "no evidence")
        sides = np.asarray(bet_home)[mask]
        n_home = int(sides.sum())
        side_desc = (f"home x{n_home} / away x{int(sides.size - n_home)}"
                     if sides.size and not np.isscalar(sides[0]) or isinstance(sides, np.ndarray)
                     and sides.size and len(set(sides.tolist())) > 1
                     else ("home" if (sides.size and bool(sides[0])) else "away"))
        arm_rows.append({"label": label, "kind": "ml_dec", "bet_side": side_desc,
                         "discovery": d, "holdout": h, "verdict": verdict})
        overlap[label] = set(F.index[mask])
    labels = [r["label"] for r in arm_rows]
    ov = {}
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            inter = len(overlap[a] & overlap[b])
            if inter:
                ov[f"{a} ∩ {b}"] = inter
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "family": args.family,
        "issue": {"user_rules": 91, "user_rules_50": 93}[args.family],
        "prereg": "E05-prereg-user_rules",
        "n_arms": len(arm_rows),
        "bar_z": BAR_Z,
        "population": {"priced_games": int(len(priced)),
                       "discovery": int((priced["split"] == "discovery").sum()),
                       "holdout": int((priced["split"] == "holdout").sum())},
        "arms": arm_rows,
        "rule_overlap_games": ov,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    ledger = {"user_rules": "E05", "user_rules_50": "E06"}[args.family]
    run = {
        "name": f"{ledger}-run-{args.family}",
        "experiment_type": f"search loop family: {args.family}",
        "declared_utc": payload["generated"],
        "meta": {"family": args.family, "n_arms": len(arm_rows), "bar_z": BAR_Z,
                 "population": payload["population"]},
        "arms": arm_rows,
        "status": "executed",
    }
    with open(ROOT / "results" / "experiments.jsonl", "a") as fh:
        fh.write(json.dumps(run, sort_keys=True) + "\n")

    print(f"{args.family}: {len(arm_rows)} arms, priced {payload['population']['priced_games']:,} "
          f"(discovery {payload['population']['discovery']:,}, "
          f"holdout {payload['population']['holdout']:,}), bar |z| > {BAR_Z}")
    for r in arm_rows:
        d, h = r["discovery"], r["holdout"]
        dc = d.get("coherent", {})
        print(f"{r['label']:56s} disc n {d['n']:4d} {d['roi']:+7.2%} z{d['z']:+5.2f} "
              f"| hold n {h['n']:4d} {h['roi']:+7.2%} | {r['verdict']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
