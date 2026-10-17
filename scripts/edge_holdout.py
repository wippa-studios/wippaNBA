"""Is the +13.54% ROI a real edge, or a threshold picked by looking at the data?

WHY THIS SCRIPT EXISTS
----------------------
`scripts/closing_line.py` prints a threshold sweep in which two cells look like an
edge:

    threshold 0.55   model +11.76% |z| 2.22   model + availability +13.54% |z| 2.54
    threshold 0.60   model  +6.19% |z| 1.02   model + availability  +3.52% |z| 0.59
    threshold 0.65   model  -2.11% |z| 0.30   model + availability  +0.07% |z| 0.01

Reporting the best of those as a result would be exactly the error this project
exists to avoid. Both sit below the |z| > 3 bar adopted from Vandenbruaene et al.
(2022, J Sports Econ 23(7)), whose null of unprofitability survived 628
implementations. And `scripts/edge_betting.py` measures a PARAMETER-FREE rule and
returns -10.80% over MORE bets. One of those numbers is selection.

THE TEST
--------
A threshold chosen by looking at a dataset is part of that dataset. The only way
to know whether the choice generalises is to choose it somewhere else and score
it somewhere else. So: leave-one-season-out.

    For each season S:
        1. Pick the best threshold on the other ten seasons.
        2. Apply that threshold to season S, which had no part in choosing it.
        3. Pool the eleven out-of-sample results.

If the sweep found a real edge, the in-sample winner is a stable threshold and
the pooled out-of-sample ROI stays positive. If it is the maximum of a sweep over
noise, a different threshold wins every fold and the pooled result lands near
zero or below - which is what "selection" means here.

Seasons are the right unit because they are the largest genuinely independent
blocks in this dataset: different rosters, different schedules, and a full year
apart in time.

THE RULE BEING TESTED
---------------------
This replicates `closing_line.roi()` exactly, including its two properties that
are easy to miss and both matter:

  * It bets the HOME SIDE ONLY. `scripts/edge_betting.py` backs whichever side
    has the edge and returns -10.80%. A home-only rule is a materially different
    bet, and its number is not comparable with that one's.
  * `stake = p_model - p_raw` can be NEGATIVE, and negative stakes are dropped
    rather than laid. So the effective rule is "bet home wherever the model is at
    least `threshold` AND beats the posted price", staked in proportion to the
    edge.

Both are reproduced rather than improved, because the question is whether THAT
number was selection - not whether a better rule could be found. Finding a better
rule is the search this project has already declined to run.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

import closing_line as cl  # noqa: E402
from DataProviders import nba_data as nd  # noqa: E402

SEED = 42
THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70)


def bets(p_model: np.ndarray, p_raw: np.ndarray, y: np.ndarray,
         threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Stake and per-bet profit. Faithful to closing_line.roi(), including the
    drop-the-negative-stakes behaviour."""
    stake = np.where(p_model >= threshold, p_model - p_raw, 0.0)
    stake = np.where(stake > 0, stake, 0.0)
    won = (y == 1).astype(float)
    profit = np.where(stake > 0, (won - p_raw) / p_raw * stake, 0.0)
    return stake, profit


def roi(p_model, p_raw, y, threshold) -> tuple[float, int]:
    stake, profit = bets(p_model, p_raw, y, threshold)
    n = int((stake > 0).sum())
    if n == 0:
        return float("nan"), 0
    return float(profit.sum() / stake.sum()), n


def main() -> int:
    games = nd.build_games()
    splits = nd.walk_forward_split(games, n_splits=5)
    p_avail, y = cl.model_probs(games, splits, with_availability=True)

    odds = cl.load_odds()
    joined = cl.join_odds(games, odds)
    if not len(joined):
        print("NO OVERLAP - cannot test anything.")
        return 1

    # The alignment closing_line.main() uses, including the GAME_ID lookup and the
    # reason it is a lookup rather than searchsorted.
    row_of = {gid: i for i, gid in enumerate(games["GAME_ID"].to_numpy())}
    idx = joined["GAME_ID"].map(row_of).to_numpy()
    d = joined.assign(
        y=y[idx], p_model=p_avail[idx],
        SEASON=games["SEASON"].to_numpy()[idx],
    ).dropna(subset=["p_model"]).reset_index(drop=True)

    pm = d["p_model"].to_numpy(float)
    praw = d["p_market_raw"].to_numpy(float)
    yy = d["y"].to_numpy().astype(int)
    seasons = d["SEASON"].to_numpy()

    print(f"games with both a prediction and a closing line: {len(d):,}")
    print(f"seasons: {', '.join(sorted(set(seasons)))}\n")

    print("=" * 72)
    print("IN-SAMPLE sweep - the headline number, on all 11 seasons at once")
    print("=" * 72)
    print(f"  {'threshold':>10s}  {'bets':>6s}  {'ROI':>9s}")
    for t in THRESHOLDS:
        r, n = roi(pm, praw, yy, t)
        print(f"  {t:>10.2f}  {n:>6d}  {r * 100:>+8.2f}%")

    print()
    print("=" * 72)
    print("PER-SEASON at threshold 0.55 - is the sign stable or does it flip?")
    print("=" * 72)
    print(f"  {'season':>10s}  {'games':>6s}  {'bets':>6s}  {'ROI':>9s}")
    pos = 0
    scored = 0
    for s in sorted(set(seasons)):
        m = seasons == s
        r, n = roi(pm[m], praw[m], yy[m], 0.55)
        if np.isfinite(r):
            scored += 1
            pos += r > 0
        shown = f"{r * 100:>+8.2f}%" if np.isfinite(r) else "      n/a"
        print(f"  {s:>10s}  {int(m.sum()):>6d}  {n:>6d}  {shown:>9s}")
    print(f"\n  positive in {pos} of {scored} seasons that placed any bet.")
    print("  Under the null of no edge, roughly half. That is the whole test.")

    print()
    print("=" * 72)
    print("LEAVE-ONE-SEASON-OUT - threshold chosen on 10, scored on the 11th")
    print("=" * 72)
    picks = []
    pooled_stake = np.zeros(len(d))
    pooled_profit = np.zeros(len(d))
    print(f"  {'held out':>10s}  {'picked':>8s}  {'bets':>6s}  {'ROI':>9s}")
    for s in sorted(set(seasons)):
        hold = seasons == s
        train = ~hold
        scores = {t: roi(pm[train], praw[train], yy[train], t)[0] for t in THRESHOLDS}
        scores = {k: v for k, v in scores.items() if np.isfinite(v)}
        if not scores:
            continue
        best = max(scores, key=scores.get)
        picks.append(best)
        st, pf = bets(pm[hold], praw[hold], yy[hold], best)
        pooled_stake[hold] = st
        pooled_profit[hold] = pf
        r, n = roi(pm[hold], praw[hold], yy[hold], best)
        shown = f"{r * 100:>+8.2f}%" if np.isfinite(r) else "      n/a"
        print(f"  {s:>10s}  {best:>8.2f}  {n:>6d}  {shown:>9s}")

    placed = pooled_stake > 0
    if placed.any():
        pooled = float(pooled_profit[placed].sum() / pooled_stake[placed].sum())
        gm, gt = pooled_profit[placed], pooled_stake[placed]
        rng = np.random.default_rng(SEED)
        draws = [gm[rng.integers(0, len(gm), len(gm))].sum()
                 / gt[rng.integers(0, len(gt), len(gt))].sum()
                 for _ in range(2000)]
        se = float(np.std(draws, ddof=1))
        z = pooled / se if se > 0 else float("nan")
        print(f"\n  POOLED OUT-OF-SAMPLE: {int(placed.sum()):,} bets  "
              f"ROI {pooled * 100:+.2f}%  SE {se * 100:.2f}%  |z| {z:.2f}")
        print(f"  against the in-sample headline of "
              f"{roi(pm, praw, yy, 0.55)[0] * 100:+.2f}% at a FIXED 0.55.")

    print(f"\n  threshold picked per fold: {picks}")
    print(f"  distinct values: {sorted(set(picks))}")
    if len(set(picks)) > 1:
        print("  The winner changes between folds. A real parameter is the same")
        print("  number on different data; one that has to be re-chosen each time")
        print("  was fitting the fold, not describing the market.")
    else:
        print("  The same threshold won every fold, which is necessary but not")
        print("  sufficient - the pooled number above is what decides it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())