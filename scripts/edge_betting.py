"""Bet where the estimate disagrees with the PRICE. A pre-specified rule.

The previous market comparison scored log loss over every game and then bet on an
absolute-confidence threshold. Both under-test the thing that matters.

Scoring every game answers "is this a good forecaster?" Betting only where the
model disagrees with the price answers "is there money in my disagreement?" Those
are different questions, and the second is the one that pays.

THE RULE, AND WHY IT NEEDS NO TUNING
------------------------------------
Bet one unit on a side when the model's probability for it EXCEEDS the probability
implied by the price actually offered. Expected profit per unit staked is

    p * (1/p_raw - 1) - (1 - p)  =  p / p_raw  -  1

which is positive exactly when p > p_raw. So "bet when the model beats the price"
is not a strategy chosen by looking at results -- it is the breakeven point, fixed
by arithmetic. There is nothing here to overfit.

That is the whole reason this script is worth running separately: every threshold
sweep reported elsewhere is a multiple-comparisons problem, and this rule has zero
free parameters.

`p_raw` is the RAW implied probability including vig, not the de-vigged one.
De-vigging first would hand the model a fairer-looking price than the one actually
posted, and the vig is exactly what a bettor pays. The overround here is +4.13%
(mean overround this run prints; reproduce: `python scripts/edge_betting.py`),
so the bar is roughly four points above the market's own estimate -- the model has
to be wrong by more than the book to be worth backing.

WHAT WOULD AND WOULD NOT SHOW UP HERE
--------------------------------------
* If the model were a better forecaster AND disagreed usefully, this would find it.
* If the model merely reproduced the market, this finds nothing, correctly: every
  bet would sit right at breakeven and the average would be the vig, negative.
* If the model is worse than the market, this should LOSE. That is the control. A
  strategy that profits here while the model trails the market at |z| = 7.28
  (model + availability) / 8.18 (model) - reproduce: `python scripts/closing_line.py`
  - would be evidence of a bug, not of an edge.

Run:  python scripts/edge_betting.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT, ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

import importlib.util  # noqa: E402

from DataProviders import nba_data as nd  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "closing_line", ROOT / "scripts" / "closing_line.py")
cl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cl)

SEED = 42
N_SPLITS = 5
Z_BAR = 3.0


def settle(p_model: np.ndarray, p_raw_home: np.ndarray, p_raw_away: np.ndarray,
           y: np.ndarray, margin: float = 0.0, detail: bool = False) -> dict:
    """Bet whichever side has positive edge over its own posted price.

    Profit per unit staked, betting side s at implied probability q = p_raw:
        win -> (1/q - 1);  lose -> -1
        expected = p/q - 1, so positive exactly when p > q.

    BOTH sides are considered, independently. That is the correction, and it is the
    whole difference between testing this idea and not testing it.

    An earlier version took the model's PREFERRED side and then asked whether that
    side had edge. That silently discards every game where the favourite has no
    edge but the underdog does -- which is precisely backing a team
    the market has underrated. HISTORICAL MEASUREMENT (pre-refactor dataset): on
    the data this was written against that was **1,928 games**, more than the
    1,553 the old rule bet on - it was not a detail, it was most of the
    opportunity. On the current run the split is 2,437 home vs 2,391 away bets
    (reproduce: `python scripts/edge_betting.py`); either way, both sides must be
    considered.

    At most one side can have positive edge in a game, because the two raw prices
    sum to the overround (~1.0404) rather than to 1, so the edges cannot both
    clear zero. That is verified by a test rather than assumed.

    BOTH prices come from the data. `p_raw_home + p_raw_away` is the overround, NOT
    1. Using `1 - p_raw_home` for the away side understates it by the vig, which
    flatters every away bet - and that bug made the control report a spurious
    +7.00% profit before it was found.

    Returns per-bet detail when `detail=True`, for the calibration diagnostic.
    """
    p_home = p_model
    p_away = 1.0 - p_model

    edge_home = p_home - p_raw_home
    edge_away = p_away - p_raw_away

    take_home = edge_home >= edge_away
    side_p = np.where(take_home, p_home, p_away)
    side_q = np.where(take_home, p_raw_home, p_raw_away)
    edge = np.where(take_home, edge_home, edge_away)

    bet = edge > margin
    won = np.where(take_home, y, 1 - y).astype(float)
    stake = np.where(bet, side_q, 0.0)
    profit = np.where(bet, won * (1.0 / side_q - 1.0) - (1.0 - won), 0.0)

    n = int(bet.sum())
    if n == 0:
        return {"bets": 0, "roi": float("nan"), "se": float("nan"),
                "total_staked": 0.0, "total_profit": 0.0,
                "home_bets": 0, "away_bets": 0}
    staked = float(stake.sum())
    total_profit = float(profit.sum())
    roi = total_profit / staked

    rng = np.random.default_rng(SEED)
    idx = np.flatnonzero(bet)
    draws = np.empty(2000)
    for i in range(2000):
        pick = rng.choice(idx, size=len(idx), replace=True)
        draws[i] = profit[pick].sum() / stake[pick].sum()
    se = float(draws.std(ddof=1))
    out = {
        "bets": n,
        "home_bets": int((bet & take_home).sum()),
        "away_bets": int((bet & ~take_home).sum()),
        "mean_edge": float(edge[bet].mean()),
        "total_staked": staked,
        "total_profit": total_profit,
        "roi": roi,
        "se": se,
        "z": roi / se if se > 0 else float("nan"),
        "win_rate": float(won[bet].mean()),
        "avg_price": float(side_q[bet].mean()),
        "implied_win_rate": float(side_p[bet].mean()),
    }
    if detail:
        out["_bets"] = {
            "edge": edge[bet], "profit": profit[bet], "stake": stake[bet],
            "won": won[bet], "price": side_q[bet], "p_model": side_p[bet],
        }
    return out


def calibration(per_bet: dict, n_bins: int = 5) -> list[dict]:
    """Is the model's stated edge related to its realised result?

    A model claiming a +9 point edge and losing money is not necessarily a
    paradox: it can mean the edge is real on average but the model is
    OVERCONFIDENT exactly where it is most confident. That is worth knowing and is
    not worth trading on its own - see the note in the module docstring.
    """
    edge = per_bet["edge"]
    profit = per_bet["profit"]
    stake = per_bet["stake"]
    if len(edge) == 0:
        return []
    qs = np.quantile(edge, np.linspace(0, 1, n_bins + 1))
    qs[-1] += 1e-12
    rows = []
    for i in range(n_bins):
        m = (edge >= qs[i]) & (edge < qs[i + 1])
        if m.sum() == 0:
            continue
        rows.append({
            "edge_lo": float(edge[m].min()), "edge_hi": float(edge[m].max()),
            "n": int(m.sum()),
            "mean_edge": float(edge[m].mean()),
            "realised_roi": float(profit[m].sum() / stake[m].sum()),
        })
    return rows


def margin_sweep(p_model, p_raw_home, p_raw_away, y,
                 margins=(0.0, 0.02, 0.04, 0.06)):
    """The parameter-free rule, then a few margins. ALL reported, none selected.

    Delegates to `settle` so there is exactly ONE implementation of the rule.
    An earlier version carried its own inline copy and kept the old
    favourite-only behaviour after `settle` was corrected, so the sweep silently
    described a different strategy from the headline number. Duplicated logic
    drifts; this is the third time in this repo that it has, after the drop lists
    in #14/#28 and the availability feature in #30.
    """
    return {m: settle(p_model, p_raw_home, p_raw_away, y, margin=m)
            for m in margins}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/edge_betting.json")
    args = ap.parse_args()

    if not cl.ODDS_DB.exists():
        print("odds database absent; cannot settle bets")
        return 1

    games = nd.build_games()
    odds = cl.load_odds()
    joined = cl.join_odds(games, odds)

    splits = nd.walk_forward_split(games, n_splits=N_SPLITS)
    p_base, y_all = cl.model_probs(games, splits, with_availability=False)
    p_avail, _ = cl.model_probs(games, splits, with_availability=True)

    row_of = {gid: i for i, gid in enumerate(games["GAME_ID"].to_numpy())}
    idx = joined["GAME_ID"].map(row_of).to_numpy()
    # The joined frame comes straight from `join_odds`, which has no model columns;
    # they are attached here rather than assumed to be there. Same GAME_ID -> row
    # lookup as the closing-line script, because an index alignment that is wrong
    # here is wrong identically there.
    scored = joined.assign(y=y_all[idx],
                           p_model=p_base[idx],
                           p_model_avail=p_avail[idx])
    scored = scored.dropna(subset=["y", "p_model"]).reset_index(drop=True)

    y = scored["y"].to_numpy().astype(int)
    p_raw = scored["p_market_raw"].to_numpy()
    p_raw_away = scored["p_away_raw"].to_numpy()
    p_devig = scored["p_market"].to_numpy()
    pm = scored["p_model"].to_numpy()
    pa = scored["p_model_avail"].to_numpy()
    n = len(y)

    print(f"settled on {n:,} games with a prediction and a posted price\n")
    over = (p_raw + p_raw_away).mean()
    print(f"  mean overround {over - 1:.4%}  (home+away raw probabilities)")
    print("  (the model must beat the RAW price, vig included, or it is not a bet)\n")

    print(f"  {'strategy':34s} {'bets':>6s} {'home':>6s} {'away':>6s} {'ROI':>9s} {'SE':>8s} {'|z|':>6s}")
    print("  " + "-" * 82)
    results = {}
    for label, p in (("model, breakeven rule", pm),
                     ("model + availability, breakeven", pa)):
        r = settle(p, p_raw, p_raw_away, y, detail=True)
        results[label] = r
        cal = calibration(r.pop("_bets"))
        r["calibration"] = cal
        print(f"  {label:34s} {r['bets']:>6,} {r['home_bets']:>6,} {r['away_bets']:>6,} "
              f"{r['roi']:>+8.2%} {r['se']:>7.2%} {abs(r['z']):>6.2f}")

    best_label = "model + availability, breakeven"
    r = settle(pa, p_raw, p_raw_away, y, detail=True)
    cal = calibration(r.pop("_bets"))

    print(f"\n  EDGE CALIBRATION ({best_label}) - stated edge vs realised ROI")
    print(f"  {'stated edge':>18s} {'n':>6s} {'mean edge':>10s} {'realised ROI':>13s}")
    print("  " + "-" * 50)
    for row in cal:
        print(f"  {row['edge_lo']:>7.3f}-{row['edge_hi']:<10.3f} {row['n']:>6,} "
              f"{row['mean_edge']:>+10.4f} {row['realised_roi']:>+12.2%}")
    print("\n  A model claiming edge and losing money is not a paradox - it can mean")
    print("  the edge is real on average but the model is OVERCONFIDENT where it is")
    print("  most confident. That is a calibration fact, not a strategy: acting on it")
    print("  would be selecting on the very statistic just computed.")

    ctrl = settle(p_devig, p_raw, p_raw_away, y)
    print(f"\n  CONTROL  market's own P vs the price   {ctrl['bets']:>6,}  ", end="")
    if ctrl["bets"] == 0:
        print("no bets - correct")
        print("           The de-vigged market probability never exceeds the raw")
        print("           posted price, because de-vigging divides the overround")
        print("           away. So the control places no bets at all, which is the")
        print("           expected result and a check on the pricing arithmetic.")
        print("           (An earlier version priced the away side as 1-p_home and")
        print("           made this control 'profit' +7.00%. It was a bug.)")
    else:
        print(f"\n  CONTROL  market's own P vs the price   {ctrl['bets']:>6,} "
              f"{'':>9s} {ctrl['roi']:>+8.2%} {ctrl['se']:>7.2%} {abs(ctrl['z']):>6.2f}")
        print("           (a near-market estimate should place no bets at all)")
    results["control_market_vs_price"] = ctrl

    print(f"\n  MARGIN SWEEP - all reported, none selected")
    print(f"  {'margin':>8s} {'model bets':>11s} {'ROI':>9s} {'|z|':>6s}   "
          f"{'avail bets':>11s} {'ROI':>9s} {'|z|':>6s}")
    print("  " + "-" * 78)
    sw_base = margin_sweep(pm, p_raw, p_raw_away, y)
    sw_avail = margin_sweep(pa, p_raw, p_raw_away, y)
    for m in sw_base:
        b, a = sw_base[m], sw_avail[m]
        tag = "  <- pre-specified" if m == 0.0 else ""
        def _fmt(d):
            if not d.get("bets"):
                return f"{'no bets':>11s} {'-':>9s} {'-':>6s}"
            return (f"{d['bets']:>11,} {d['roi']:>+8.2%} {abs(d['z']):>6.2f}")
        print(f"  {m:>8.2f} {_fmt(b)}   {_fmt(a)}{tag}")

    print(f"\nVERDICT  (bar: |z| > {Z_BAR})")
    pre = results["model + availability, breakeven"]
    print(f"\n  Both sides were considered: {pre['home_bets']:,} home bets and "
          f"{pre['away_bets']:,} away bets out of {n:,} games.")
    if abs(pre.get("z", 0)) >= Z_BAR:
        print(f"  The pre-specified breakeven rule reaches the bar at "
              f"|z| = {abs(pre['z']):.2f}.")
    else:
        print(f"  The pre-specified rule sits at |z| = {abs(pre.get('z', 0)):.2f}, "
              f"well under the bar.")
        print("  It is also the ONLY row here with no free parameter, so there is")
        print("  nothing to have been selected on.")
    print("\n  This is the strongest form of the selective-betting question: a rule")
    print("  fixed by arithmetic rather than by a threshold search. Betting only")
    print("  where the model beats the posted price does not produce an edge.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_settled": int(n), "z_bar": Z_BAR,
        "results": results,
        "margin_sweep_model": sw_base,
        "margin_sweep_model_availability": sw_avail,
    }, indent=2, default=float) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())