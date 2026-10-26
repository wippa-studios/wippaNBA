"""Issue #59: is the market less efficient in the first weeks of each season?

The weakest claim in the research-links file: early-season lines are
systematically mispriced because books anchor on prior-season form (the one
cited study is a week-1 TOTALS result, n=60-65).  This measures it directly on
the project's moneylines, on the existing market comparison (the join of
build_games() with closing lines), split by how deep each game is into the
season.

DEFINITION AND DECISION (pre-registered I59-prereg-early-season)
----------------------------------------------------------------
For each game, season_depth = min(home_n, away_n) = the fewer of the two
teams' games played THIS season strictly before tonight (pre-game,
truncation-safe).  A game is EARLY iff season_depth <= 15.  The claim has legs
iff the MARKET log loss is measurably WORSE early: (early_market_ll -
late_market_ll) > 0 with |z| > 3 on 2,000 bootstrap resamples (seed 42).  A
flat gap closes the claim: markets can be efficient on day 1 too.

The model and market numbers are the same arms the market comparison already
reports (0.59766 vs 0.62034); the whole-frame values are reprinted as a
reproduction check.  The betting-P&L split reuses the exact parameter-free
rule whose overall result is the recorded -8.23% over 4,828 bets.

Run::

    OMP_NUM_THREADS=1 python scripts/issue59_early_season.py
"""
from __future__ import annotations

import importlib.util
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

from DataProviders import nba_data as nd            # noqa: E402
from Models import logreg_rolling as lr             # noqa: E402
from Models import measurement as ms                # noqa: E402


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


cl = _load_script("closing_line")
eb = _load_script("edge_betting")

OUT_JSON = ROOT / "results" / "issue59_early_season.json"
OUT_MD = ROOT / "results" / "issue59_early_season.md"
JSONL = ROOT / "results" / "experiments.jsonl"

EARLY_N = 15
N_BOOT = 2000
SEED = 42

RECORDED_MARKET_LL = 0.59766
RECORDED_MODEL_AV_LL = 0.62034


def _clip(p):
    return np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)


def mean_ll(y, p) -> float:
    p = _clip(p)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def boot_gap(y_e, p_e, y_l, p_l) -> dict:
    """Bootstrap the (early minus late) log-loss gap; positive = early worse."""
    rng = np.random.default_rng(SEED)
    ne, nl = len(y_e), len(y_l)
    if not (len(y_e) == len(p_e) and len(y_l) == len(p_l)):
        raise ValueError("y and p must be same length within each segment")
    vals = np.empty(N_BOOT)
    for i in range(N_BOOT):
        ie = rng.integers(0, ne, size=ne)
        il = rng.integers(0, nl, size=nl)
        vals[i] = (mean_ll(y_e[ie], p_e[ie]) - mean_ll(y_l[il], p_l[il]))
    se = float(vals.std(ddof=1))
    mean = float(vals.mean())
    return {
        "gap_early_minus_late": mean,
        "se": se,
        "ci": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))],
        "z": (mean / se if se > 0 else float("nan")),
        "p_positive": float((vals > 0).mean()),
        "n_boot": N_BOOT,
        "bootstrap_seed": SEED,
    }


def main() -> int:
    t0 = time.time()
    games = nd.build_games().reset_index(drop=True)
    odds = cl.load_odds()
    joined = cl.join_odds(games, odds).reset_index(drop=True)
    splits = nd.walk_forward_split(games, n_splits=5)
    p_av, y = cl.model_probs(games, splits, with_availability=True)

    sgp = lr.season_games_played(games)
    g = games[["GAME_ID", "y"]].copy()
    g["p_model"] = p_av
    g["home_n"] = sgp["home_n"].to_numpy()
    g["away_n"] = sgp["away_n"].to_numpy()
    joined = joined.merge(g, on="GAME_ID", how="left")
    # The comparison subset: games with BOTH a closing line and a model
    # prediction (the recorded 5,788-game market comparison).
    joined = joined.dropna(subset=["p_model", "p_market", "y"]).reset_index(drop=True)
    joined["season_depth"] = np.minimum(joined["home_n"], joined["away_n"])

    y_j = joined["y"].to_numpy().astype(int)
    p_mkt = joined["p_market"].to_numpy(float)
    p_mod = joined["p_model"].to_numpy(float)
    # The AWAY price comes from the odds table (p_away_raw), NOT 1 - p_raw_home:
    # the two raw prices sum to the overround, and using 1 - p_raw_home flatters
    # every away bet by the vig (the historical +7.00% control bug).
    p_raw_h = joined["p_market_raw"].to_numpy(float)
    p_raw_a = joined["p_away_raw"].to_numpy(float)
    depth = joined["season_depth"].to_numpy()

    # Whole-frame reproduction.
    repro_mkt = mean_ll(y_j, p_mkt)
    repro_mod = mean_ll(y_j, p_mod)

    def run(cut_early_mask, label: str) -> dict:
        m = cut_early_mask
        out = {
            "label": label,
            "n_early": int(m.sum()),
            "n_late": int((~m).sum()),
            "market_ll_early": mean_ll(y_j[m], p_mkt[m]),
            "market_ll_late": mean_ll(y_j[~m], p_mkt[~m]),
            "model_ll_early": mean_ll(y_j[m], p_mod[m]),
            "model_ll_late": mean_ll(y_j[~m], p_mod[~m]),
            "market_gap_early_minus_late": boot_gap(
                y_j[m], p_mkt[m], y_j[~m], p_mkt[~m]),
        }
        out["model_gap_early_minus_late"] = boot_gap(
            y_j[m], p_mod[m], y_j[~m], p_mod[~m])
        # market-minus-model efficiency gap, early vs late (early is "what the
        # model loses to"): positive means the market looks worse early.
        g_early = mean_ll(y_j[m], p_mkt[m]) - mean_ll(y_j[m], p_mod[m])
        g_late = mean_ll(y_j[~m], p_mkt[~m]) - mean_ll(y_j[~m], p_mod[~m])
        out["market_minus_model_early"] = g_early
        out["market_minus_model_late"] = g_late
        out["mkt_minus_model_gap_early_minus_late"] = g_early - g_late
        # The exact parameter-free bet rule, split.
        for part, mask in (("early", m), ("late", ~m)):
            r = eb.settle(p_mod[mask], p_raw_h[mask], p_raw_a[mask],
                          y_j[mask], margin=0.0)
            out[f"bet_rule_{part}"] = {k: r[k] for k in
                                       ("bets", "roi", "se", "z",
                                        "home_bets", "away_bets")}
        return out

    primary = run(joined["season_depth"].to_numpy() <= EARLY_N,
                  f"season_depth <= {EARLY_N}")
    # Combined (all rows) reproduction of the recorded breakeven rule: the
    # recorded headline is -8.23% over 4,828 bets for this exact rule.
    bet_all = eb.settle(p_mod, p_raw_h, p_raw_a, y_j, margin=0.0)
    sens = {
        "cutoff_10": run(depth <= 10, "season_depth <= 10"),
        "cutoff_20": run(depth <= 20, "season_depth <= 20"),
        "early_calendar_12": run(calendar_early(joined, games),
                                 "first ~3 calendar weeks"),
    }

    g = primary["market_gap_early_minus_late"]
    point = g["gap_early_minus_late"]
    if point > 0 and g["z"] is not None and abs(g["z"]) > 3:
        verdict = (
            "CLAIM HAS LEGS: the market is measurably worse early -- "
            f"+{point:.5f} nats early-vs-late (|z| {abs(g['z']):.2f}). The "
            "first-weeks inefficiency is real on this sample and needs a "
            "targeted early-season market test."
        )
    elif point > 0:
        verdict = (
            "FLAT: the market log loss is "
            f"{point:+.5f} nats worse early (|z| {abs(g['z']):.2f}, CI "
            f"{g['ci']}), below the |z| > 3 bar. The claim is not supported; "
            "any early-season effect is noise on this sample."
        )
    else:
        verdict = (
            "CLOSED: the market is "
            f"{'no better' if abs(point) < 1e-4 else f'{point:+.5f} nats '}"
            f"late-vs-early (|z| {abs(g['z']):.2f}). No early-season "
            "inefficiency; markets are efficient in week one too."
        )

    payload = {
        "issue": 59,
        "protocol_ref": "I59-prereg-early-season",
        "generated_utc": ms.utc_now(),
        "runtime_seconds": round(time.time() - t0, 1),
        "early_definition": f"min(home_n, away_n) <= {EARLY_N} (both teams "
                            "fewer than EARLY_N games into the season)",
        "n_games_with_line": int(len(joined)),
        "reproduction": {
            "market_log_loss": repro_mkt,
            "model_plus_availability_log_loss": repro_mod,
            "recorded_market_log_loss": RECORDED_MARKET_LL,
            "recorded_model_plus_availability_log_loss": RECORDED_MODEL_AV_LL,
            "market_drift": repro_mkt - RECORDED_MARKET_LL,
            "model_drift": repro_mod - RECORDED_MODEL_AV_LL,
        },
        "primary": primary,
        "bet_rule_all_reproduction": {k: bet_all[k] for k in
                                      ("bets", "roi", "z", "home_bets",
                                       "away_bets")},
        "sensitivity": sens,
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ms.append_jsonl(JSONL, {
        "name": "I59-run-early-season",
        "protocol_ref": "I59-prereg-early-season",
        "declared_utc": ms.utc_now(),
        "experiment_type": "issue #59: early-season market drift",
        "n_games_with_line": int(len(joined)),
        "reproduction": payload["reproduction"],
        "primary": primary,
        "sensitivity": sens,
        "verdict": verdict,
        "status": "executed",
    })

    def fmt(r):
        g = r["market_gap_early_minus_late"]
        return (f"  n early {r['n_early']:>4} late {r['n_late']:>5} | "
                f"market ll early {r['market_ll_early']:.5f} late "
                f"{r['market_ll_late']:.5f} | gap {g['gap_early_minus_late']:+.5f} "
                f"(|z| {abs(float(g['z'])):.2f}) | bet early "
                f"{r['bet_rule_early']['roi']*100:+.2f}% (n "
                f"{r['bet_rule_early']['bets']}) late "
                f"{r['bet_rule_late']['roi']*100:+.2f}% (n "
                f"{r['bet_rule_late']['bets']})")

    L = ["=" * 78,
         "ISSUE 59 -- EARLY-SEASON MARKET DRIFT",
         "=" * 78]
    a = L.append
    a(f"games with line {len(joined)}; reproduction: market {repro_mkt:.5f} "
      f"(recorded {RECORDED_MARKET_LL}), model+avail {repro_mod:.5f} "
      f"(recorded {RECORDED_MODEL_AV_LL})")
    a(f"combined breakeven rule: {bet_all['bets']:,} bets, "
      f"ROI {bet_all['roi']*100:+.2f}% (|z| {abs(bet_all['z']):.2f}) "
      f"[recorded 4,828 bets / -8.23%]")
    a(f"primary ('season_depth <= {EARLY_N}'):")
    a(fmt(primary))
    a("sensitivity:")
    for s in sens.values():
        a(fmt(s))
    a("")
    a(verdict)
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


def calendar_early(joined: pd.DataFrame, games: pd.DataFrame) -> np.ndarray:
    """First ~3 calendar weeks of each season, by the league's first game."""
    gd = pd.to_datetime(games["GAME_DATE"])
    season_start = games.groupby("SEASON")["GAME_DATE"].transform("min")
    started = pd.to_datetime(season_start)
    joined_date = pd.to_datetime(joined["GAME_DATE"])
    season_map = games.set_index("GAME_ID")["SEASON"]
    j_season = joined["GAME_ID"].map(season_map)
    j_start = games.groupby("SEASON")["GAME_DATE"].min().reindex(j_season)
    weeks = (joined_date - pd.to_datetime(j_start.to_numpy())).dt.days <= 21
    return weeks.to_numpy()


if __name__ == "__main__":
    raise SystemExit(main())