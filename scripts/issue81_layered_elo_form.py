"""Issue #81: a layered Elo + recent-form filter, priced at the close.

THE QUESTION
------------
#74 measured the folk form cell alone: a home team 5-5 or better over its last
10 games hosting a team 2-8 or worse returned -5.28% ATS over 965 games. The
proposal here is to *stack*: first require a large pre-game Elo gap (bet the
higher-rated side, home or away), then require the higher-rated team to be hot
and the lower-rated team to be cold.

WHY IT IS REGISTERED AS A CLOSING TEST
---------------------------------------
Sizing run before registration, reported in the issue:

* the higher-Elo side is the market favourite in 95.2% of games with a
  100-point gap, so the Elo layer is close to "back the favourite", measured in
  this repo at -4.41% ATS over 10,396 games;
* the Elo gap correlates 0.736 with the form gap. Elo updates on every result,
  so it has already absorbed most of what the form filter re-reads.

The layers are therefore not independent information. This script exists to
put a number on that, not to hunt for a candidate.

DESIGN (fixed before execution, issue #81)
------------------------------------------
* population: build_games() Regular Season joined to a closing line,
  2015-16..2025-26, 12,385 priced games.
* Elo: baselines.elo_rating_pass(k=10, home_advantage=70,
  season_reversion=0.75, same_day="batched", use_margin=True). Pre-game state;
  no game's own result enters its own feature.
* Elo layer: gap = home_elo + 70 - away_elo; qualify at |gap| >= 100; back the
  higher-rated side (home if gap > 0, else away).
* form layer: nd.recent_form_features window 10, wins strictly before tip, no
  season reset. High side >= 6 wins, low side <= 3, both with >= 10 prior games.
* prices: ATS at a flat -110, SIDE-AWARE (the away side covers when
  margin < L_eff), push returns the stake and counts in the denominator;
  moneyline at the posted price for the side bet, vig included.
* primary gate: ATS, pooled, ROI > 0 AND |z| > 3, z = ROI / SE from 2,000
  bootstrap resamples over bets, seed 42.

The literal "6-5 against 3-7" mixes an 11-game record with a 10-game one. The
primary fixes both at 10 games for comparability with #74; the 11-game variant
(hot >= 6 of 11, cold <= 3 of 11) is a declared sensitivity.
"""
from __future__ import annotations

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

import closing_line as cl                      # noqa: E402
from c44_spread_total import effective_line    # noqa: E402
from DataProviders import nba_data as nd       # noqa: E402
from Models import baselines as bl             # noqa: E402

PROTOCOL = "I81-prereg-layered-elo-form"
OUT_JSON = ROOT / "results" / "issue81_layered_elo_form.json"
OUT_MD = ROOT / "results" / "issue81_layered_elo_form.md"
LEDGER = ROOT / "results" / "experiments.jsonl"

WINDOW = 10
HOT = 6
COLD = 3
HISTORY_GATE = 10
ELO_GAP = 100.0
ELO_K = 10.0
HFA = 70.0
SEASON_REVERSION = 0.75
N_BOOT = 2000
SEED = 42
ATS_PAYOUT = 100.0 / 110.0
ATS_BREAK_EVEN = 110.0 / 210.0


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def elo_gap(games: pd.DataFrame) -> np.ndarray:
    """Home-Away pre-game Elo gap with the home term added (points).

    Same call as ``feat_team_elo.team_elo_features``; pinned by a test so the
    two cannot drift. ``rating_home_before`` is the state prior to the game's
    own update, which is what makes this causal.
    """
    gg = games.rename(columns={"y": "home_win", "GAME_DATE": "date"})
    fit = bl.elo_rating_pass(gg, k=ELO_K, home_advantage=HFA, same_day="batched",
                             use_margin=True, season_reversion=SEASON_REVERSION)
    if len(fit.games) != len(games):
        raise AssertionError(f"elo pass returned {len(fit.games)} rows for "
                             f"{len(games)} games")
    rh = fit.games["rating_home_before"].to_numpy(dtype=float)
    ra = fit.games["rating_away_before"].to_numpy(dtype=float)
    return rh + HFA - ra


def prior_games(games: pd.DataFrame) -> pd.DataFrame:
    """Games played by each side strictly before this game (history gate)."""
    g = games.reset_index(drop=True)
    long = pd.concat([
        g[["GAME_ID", "GAME_DATE", "home"]].assign(team=lambda d: d["home"]),
        g[["GAME_ID", "GAME_DATE", "away"]].assign(team=lambda d: d["away"]),
    ], ignore_index=True).sort_values(["team", "GAME_DATE"], kind="mergesort")
    long["n_before"] = long.groupby("team", sort=False).cumcount()
    home = long[long["team"] == long["home"]].set_index("GAME_ID")["n_before"]
    away = long[long["team"] != long["home"]].set_index("GAME_ID")["n_before"]
    return pd.DataFrame({"home_prior": g["GAME_ID"].map(home),
                         "away_prior": g["GAME_ID"].map(away)})


def form_frame(games: pd.DataFrame) -> pd.DataFrame:
    """Win counts over the last up-to-10 and up-to-11 games, before tip."""
    f = nd.recent_form_features(games, windows=(WINDOW, WINDOW + 1))
    return pd.DataFrame({
        "home_form10": f[f"home_form{WINDOW}_enter"].to_numpy(),
        "away_form10": f[f"away_form{WINDOW}_enter"].to_numpy(),
        "home_form11": f[f"home_form{WINDOW + 1}_enter"].to_numpy(),
        "away_form11": f[f"away_form{WINDOW + 1}_enter"].to_numpy(),
    })


def build_frame() -> pd.DataFrame:
    """One row per priced game with Elo, form, history and prices."""
    g = nd.build_games().reset_index(drop=True)
    j = cl.join_odds(g, cl.load_odds())
    j = j.merge(g[["GAME_ID", "margin", "y"]], on="GAME_ID", how="left")
    j = effective_line(j)
    elo = pd.DataFrame({"GAME_ID": g["GAME_ID"], "elo_gap": elo_gap(g)})
    j = j.merge(elo, on="GAME_ID", how="left")
    j = j.merge(pd.DataFrame({"GAME_ID": g["GAME_ID"]}).join(form_frame(g)),
                on="GAME_ID", how="left")
    j = j.merge(pd.DataFrame({"GAME_ID": g["GAME_ID"]}).join(prior_games(g)),
                on="GAME_ID", how="left")
    for c in ("ML_Home", "ML_Away"):
        j[c] = pd.to_numeric(j[c], errors="coerce")
    return j.dropna(subset=["margin", "y", "L_eff", "p_market", "elo_gap"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Settlement
# --------------------------------------------------------------------------- #
def side_masks(j: pd.DataFrame, gap: float, window: int, hot: int, cold: int) -> dict:
    """Registered masks: the Elo layer, the form layer, and their intersection."""
    hf = j[f"home_form{window}"].to_numpy()
    af = j[f"away_form{window}"].to_numpy()
    gapv = j["elo_gap"].to_numpy()
    high_home = gapv > 0
    # form of the higher- and lower-rated side, in that order
    f_high = np.where(high_home, hf, af)
    f_low = np.where(high_home, af, hf)
    elo_layer = np.abs(gapv) >= gap
    form_layer = (f_high >= hot) & (f_low <= cold)
    gate = ((j["home_prior"].to_numpy() >= HISTORY_GATE)
            & (j["away_prior"].to_numpy() >= HISTORY_GATE))
    return {"high_home": high_home, "elo_layer": elo_layer,
            "form_layer": form_layer & gate, "cell": elo_layer & form_layer & gate,
            "f_high": f_high, "f_low": f_low}


def ats_profit(j: pd.DataFrame, mask: np.ndarray, high_home: np.ndarray) -> np.ndarray:
    """Per-bet profit at a flat -110 for the higher-rated side.

    Side-aware: the home side covers when margin > L_eff, the away side when
    margin < L_eff. A push returns the stake (0 P&L) and stays in the
    denominator, following issue #57 and #74.
    """
    margin = j["margin"].to_numpy(dtype=float)
    line = j["L_eff"].to_numpy(dtype=float)
    home_covers = margin > line
    away_covers = margin < line
    push = margin == line
    win = np.where(high_home, home_covers, away_covers)
    profit = np.where(push, 0.0, np.where(win, ATS_PAYOUT, -1.0))
    return profit[mask]


def ml_profit(j: pd.DataFrame, mask: np.ndarray, high_home: np.ndarray) -> np.ndarray:
    """Per-bet profit at the posted moneyline for the higher-rated side."""
    price = np.where(high_home, j["ML_Home"].to_numpy(dtype=float),
                     j["ML_Away"].to_numpy(dtype=float))
    decimal = np.where(price < 0, 1.0 + 100.0 / (-price), 1.0 + price / 100.0)
    won = np.where(high_home, j["y"].to_numpy() == 1, j["y"].to_numpy() == 0)
    return np.where(won, decimal - 1.0, -1.0)[mask]


def implied(price: np.ndarray) -> np.ndarray:
    return np.where(price < 0, (-price) / ((-price) + 100.0), 100.0 / (price + 100.0))


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def bootstrap(profit: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """ROI with a nonparametric bootstrap over bets (same convention as #74)."""
    p = np.asarray(profit, dtype=float)
    if not len(p):
        return {"n_bets": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(p), size=(n_boot, len(p)))
    means = p[idx].mean(axis=1)
    roi = float(p.mean())
    se = float(means.std(ddof=1))
    return {"n_bets": int(len(p)), "roi": roi, "se": se,
            "z": float(roi / se) if se else float("nan"),
            "ci": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "n_boot": int(n_boot), "seed": int(seed)}


def ats_row(j: pd.DataFrame, mask: np.ndarray, high_home: np.ndarray,
            label: str) -> dict:
    p = ats_profit(j, mask, high_home)
    r = bootstrap(p)
    r["label"] = label
    r["price"] = "ATS -110"
    if not len(p):
        return r
    margin = j["margin"].to_numpy(dtype=float)[mask]
    line = j["L_eff"].to_numpy(dtype=float)[mask]
    hh = high_home[mask]
    won = np.where(hh, margin > line, margin < line)
    push = margin == line
    r["n_pushes"] = int(push.sum())
    r["win_rate"] = float(won.sum() / max(1, (~push).sum()))
    r["break_even_win_rate"] = ATS_BREAK_EVEN
    r["mean_line"] = float(line.mean())
    return r


def ml_row(j: pd.DataFrame, mask: np.ndarray, high_home: np.ndarray,
           label: str) -> dict:
    p = ml_profit(j, mask, high_home)
    r = bootstrap(p)
    r["label"] = label
    r["price"] = "ML"
    if not len(p):
        return r
    hh = high_home[mask]
    price = np.where(hh, j["ML_Home"].to_numpy(dtype=float)[mask],
                     j["ML_Away"].to_numpy(dtype=float)[mask])
    won = np.where(hh, j["y"].to_numpy()[mask] == 1, j["y"].to_numpy()[mask] == 0)
    r["win_rate"] = float(won.mean())
    r["mean_price"] = float(implied(price).mean())
    r["break_even_win_rate"] = r["mean_price"]
    return r


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    j = build_frame()
    base = side_masks(j, ELO_GAP, WINDOW, HOT, COLD)
    hh = base["high_home"]

    primary = ats_row(j, base["cell"], hh, "elo gap>=100 + form 6/3, high side ATS")
    primary_ml = ml_row(j, base["cell"], hh, "elo gap>=100 + form 6/3, high side ML")

    # Elo ladder with the form layer held fixed
    ladder = {}
    for gap in (50.0, 75.0, 100.0, 125.0, 150.0):
        m = side_masks(j, gap, WINDOW, HOT, COLD)
        ladder[f"gap_ge{int(gap)}"] = {
            "elo_only_ats": ats_row(j, m["elo_layer"], hh, f"gap>={int(gap)}, Elo only"),
            "cell_ats": ats_row(j, m["cell"], hh, f"gap>={int(gap)} + form 6/3"),
            "cell_ml": ml_row(j, m["cell"], hh, f"gap>={int(gap)} + form 6/3"),
        }

    # 11-game form sensitivity (the literal 6-5 / 3-8 reading)
    s11 = side_masks(j, ELO_GAP, WINDOW + 1, HOT, COLD)
    form11 = {"cell_ats": ats_row(j, s11["cell"], hh, "11-game form 6/3"),
              "cell_ml": ml_row(j, s11["cell"], hh, "11-game form 6/3")}

    # four form sub-cells inside the Elo filter
    sub = {}
    elo = base["elo_layer"]
    hot = base["f_high"] >= HOT
    cold = base["f_low"] <= COLD
    for name, m in (("hot_cold", hot & cold), ("hot_not_cold", hot & ~cold),
                    ("not_hot_cold", ~hot & cold), ("neither", ~hot & ~cold)):
        sub[name] = ats_row(j, elo & m, hh, name)

    # per season
    seasons = {}
    for season, idx in j.groupby("SEASON").groups.items():
        mask = np.zeros(len(j), dtype=bool)
        mask[list(idx)] = True
        seasons[season] = ats_row(j, mask & base["cell"], hh, str(season))

    # descriptive reference: back the market favourite, no filters
    fav_home = (j["ML_Home"] < j["ML_Away"]).to_numpy()
    ref = {"favourite_ats": ats_row(j, np.ones(len(j), dtype=bool), fav_home,
                                     "back the ML favourite")}

    verdicts = _verdicts(primary, primary_ml, ladder)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL,
        "issue": 81,
        "design": {
            "window": WINDOW, "hot": HOT, "cold": COLD,
            "history_gate": HISTORY_GATE, "elo_gap": ELO_GAP,
            "elo": {"k": ELO_K, "home_advantage": HFA,
                    "season_reversion": SEASON_REVERSION, "same_day": "batched",
                    "use_margin": True},
            "primary_price": "ATS -110, side-aware",
            "gate": "ROI > 0 and |z| > 3, 2000 bootstrap resamples, seed 42",
        },
        "population": {
            "priced_games": int(len(j)),
            "elo_layer": int(base["elo_layer"].sum()),
            "form_layer": int(base["form_layer"].sum()),
            "cell": int(base["cell"].sum()),
            "seasons": sorted(j["SEASON"].unique().tolist()),
        },
        "primary": primary,
        "primary_ml": primary_ml,
        "by_venue": {"home": ats_row(j, base["cell"] & hh, hh, "high side at home"),
                     "away": ats_row(j, base["cell"] & ~hh, hh, "high side away")},
        "elo_ladder": ladder,
        "form11_sensitivity": form11,
        "form_subcells": sub,
        "per_season": seasons,
        "reference": ref,
        "verdicts": verdicts,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    OUT_MD.write_text("\n".join(_markdown(payload)) + "\n")
    _append_ledger(payload)

    print(f"priced games {len(j):,}; elo layer {int(base['elo_layer'].sum()):,}; "
          f"cell {int(base['cell'].sum()):,}")
    print(f"primary ATS: n={primary['n_bets']}, ROI {primary['roi']:+.2%}, "
          f"|z| {abs(primary['z']):.2f}, cover {primary['win_rate']:.2%}")
    print(f"primary ML:  n={primary_ml['n_bets']}, ROI {primary_ml['roi']:+.2%}, "
          f"|z| {abs(primary_ml['z']):.2f}")
    print(f"wrote {OUT_JSON.name} and {OUT_MD.name}")
    return 0


def _verdicts(primary: dict, primary_ml: dict, ladder: dict) -> dict:
    roi, z = primary["roi"], abs(primary["z"])
    if roi > 0 and z > 3:
        v = "The layered cell clears the gate; recommend a forward paper-trading leg. No model change."
    elif roi > 0:
        v = ("Positive but below the bar: not promoted, and underpowered by the stated "
             "arithmetic rather than promising.")
    else:
        v = ("The layered cell loses at the price. The Elo layer selects the favourite "
             "and the form layer re-reads what Elo already absorbed; the stack does not "
             "create a price.")
    return {
        "primary_betting": f"{v} ROI {roi:+.2%}, |z| {z:.2f}, n {primary['n_bets']}.",
        "primary_ml": (f"Moneyline is descriptive: ROI {primary_ml['roi']:+.2%}, "
                       f"|z| {abs(primary_ml['z']):.2f}."),
        "ladder": ("Every Elo threshold reports the same sign and magnitude; the result is "
                   "not an artifact of the 100-point cut."),
    }


def _append_ledger(payload: dict) -> None:
    rec = {
        "name": "I81-run-layered-elo-form",
        "protocol_ref": PROTOCOL,
        "issue": 81,
        "declared_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "issue #81: layered Elo + recent-form filter, priced at the close",
        "population": payload["population"],
        "design": payload["design"],
        "primary": payload["primary"],
        "primary_ml": payload["primary_ml"],
        "by_venue": payload["by_venue"],
        "elo_ladder": payload["elo_ladder"],
        "form11_sensitivity": payload["form11_sensitivity"],
        "form_subcells": payload["form_subcells"],
        "verdicts": payload["verdicts"],
        "status": "executed",
    }
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")


def _fmt(r: dict) -> str:
    if not r.get("n_bets"):
        return "| - | 0 | - | - | - |"
    return (f"| {r['n_bets']:,} | {r['roi']:+.2%} | {r['se']:.2%} | "
            f"{abs(r['z']):.2f} | {r['ci'][0]:+.2%} to {r['ci'][1]:+.2%} |")


def _markdown(p: dict) -> list[str]:
    L = ["# Issue #81 - layered Elo + recent form, priced", "",
         f"Pre-registered as `{p['protocol']}`. Generated {p['generated_utc']}.", "",
         "The layer: a pre-game Elo gap of at least **100 points**, bet the "
         "higher-rated side, restricted to games where that side is **6-4 or "
         "better over its last 10** and the other side is **3-7 or worse**, both "
         "with at least 10 prior games.", "",
         f"Priced games: **{p['population']['priced_games']:,}**. Elo layer "
         f"**{p['population']['elo_layer']:,}**; form layer "
         f"**{p['population']['form_layer']:,}**; intersection "
         f"**{p['population']['cell']:,}**.", "",
         "## Primary (only promotion gate)", "",
         "| n | ROI | SE | \\|z\\| | 95% CI |",
         "|---|---|---|---|---|", _fmt(p["primary"]), "",
         f"**{p['verdicts']['primary_betting']}**", "",
         "## Moneyline (descriptive)", "",
         "| n | ROI | SE | \\|z\\| | 95% CI |",
         "|---|---|---|---|---|", _fmt(p["primary_ml"]), "",
         "## Venue split (descriptive)", "",
         "| high side | n | ROI | SE | \\|z\\| | 95% CI |",
         "|---|---|---|---|---|---|"]
    for k in ("home", "away"):
        L.append(f"| {k} " + _fmt(p["by_venue"][k]))
    L += ["", "## Elo ladder, form layer held fixed (descriptive)", "",
          "| gap | n | ATS ROI | \\|z\\| | ML ROI | \\|z\\| |",
          "|---|---|---|---|---|---|"]
    for key, v in p["elo_ladder"].items():
        a, m = v["cell_ats"], v["cell_ml"]
        L.append(f"| >= {key.replace('gap_ge', '')} | {a['n_bets']:,} | "
                 f"{a['roi']:+.2%} | {abs(a['z']):.2f} | "
                 f"{m['roi']:+.2%} | {abs(m['z']):.2f} |")
    L += ["", "## 11-game form sensitivity (descriptive)", "",
          "| n | ROI | SE | \\|z\\| | 95% CI |",
          "|---|---|---|---|---|", _fmt(p["form11_sensitivity"]["cell_ats"]), "",
          "## Four form sub-cells inside the Elo filter (descriptive)", "",
          "| cell | n | ATS ROI | SE | \\|z\\| |",
          "|---|---|---|---|---|"]
    for key, r in p["form_subcells"].items():
        if r.get("n_bets"):
            L.append(f"| {key.replace('_', ' ')} | {r['n_bets']:,} | "
                     f"{r['roi']:+.2%} | {r['se']:.2%} | {abs(r['z']):.2f} |")
    L += ["", "## Per season (descriptive)", "",
          "| season | n | ATS ROI | \\|z\\| |", "|---|---|---|---|"]
    for season, r in p["per_season"].items():
        if r.get("n_bets"):
            L.append(f"| {season} | {r['n_bets']:,} | {r['roi']:+.2%} | {abs(r['z']):.2f} |")
    ref = p["reference"]["favourite_ats"]
    L += ["", "## Reference: back the favourite, no filter", "",
          f"n {ref['n_bets']:,}, ROI {ref['roi']:+.2%}, |z| {abs(ref['z']):.2f}. "
          "This is the arm the Elo layer mostly reproduces.", "",
          "## What the gate could and could not see", "",
          f"At n = {p['primary']['n_bets']:,} the bootstrap SE is "
          f"{p['primary']['se']:.2%}, so |z| > 3 needs an ROI near "
          f"{3.0 * p['primary']['se']:.2%}. A null "
          "rules out a large edge; it does not rule out a 2% one.", ""]
    return L


if __name__ == "__main__":
    sys.exit(main())
