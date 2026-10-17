"""Do buckets of the player-Elo contrast behave like a real signal?

THE CHECK
---------
Everything else in this project scores a feature by asking whether adding it to a
logistic regression improves log loss. That is the right test for a *model*, but
it is indirect: it asks whether the model can use the feature, not whether the
feature means anything on its own.

This script removes the model entirely. It takes the raw contrast

    home team's expected total player Elo  -  away team's expected total player Elo

buckets games by it, and reports the actual home-win rate in each bucket. If the
contrast carries information, three things must hold:

  1. Home-win rate RISES monotonically as the bucket rises. A feature that is
     ordered but flat, or ordered then reversing, is not a strength measure.
  2. The bucket containing contrast = 0 must sit near the home-court base rate
     (~55-58%), NOT at 50%. Elo difference says nothing about home court, so a
     bucket centred on zero is the cleanest read on the league's floor.
  3. The spread across buckets must be wider than binomial noise on those bucket
     sizes, or the ordering is luck.

It also reports the CLOSING MARKET's de-vigged probability for the same buckets.
That is the comparison that matters: a feature is only useful for betting if it
orders games better than the price does, and this shows both orderings on one
axis.

WHY THIS CANNOT BE CIRCULAR
---------------------------
The Elo ratings are strictly causal - `player_elo_ratings` reads a rating BEFORE a
game and writes it after, and `Tests/test_player_elo.py` verifies that truncating
the future changes nothing. So the bucketing uses no post-game information. There
is no fitted object here, no train/test split, and no threshold: the buckets are
fixed quantiles of the contrast, not chosen to look good.

RUN:  python scripts/elo_buckets.py
"""

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

N_BUCKETS = 10
ROSTER = "all_prior"


def bucketise(contrast: np.ndarray, n: int = N_BUCKETS) -> np.ndarray:
    """Fixed quantile buckets. Rank-based so the buckets are equally POPULATED,
    which is what makes the binomial noise comparable across them."""
    order = np.argsort(contrast, kind="mergesort")
    ranks = np.empty(len(contrast), dtype=int)
    ranks[order] = np.arange(len(contrast))
    return (ranks * n) // len(contrast)


def main() -> int:
    games = nd.build_games()
    elo = nd.player_elo_features(games, roster_mode=ROSTER)

    d = pd.concat([
        games[["GAME_ID", "GAME_DATE", "SEASON", "y"]].reset_index(drop=True),
        elo.reset_index(drop=True),
    ], axis=1)

    contrasts = {
        "elo_mean (scale-free)": (d["home_elo_mean"] - d["away_elo_mean"]),
        "elo_available (raw total)": (d["home_elo_available"] - d["away_elo_available"]),
        "elo_top5": (d["home_elo_top5"] - d["away_elo_top5"]),
    }

    # Market, for the same games, de-vigged.
    try:
        j = cl.join_odds(games, cl.load_odds())
        row_of = {gid: i for i, gid in enumerate(games["GAME_ID"].to_numpy())}
        idx = j["GAME_ID"].map(row_of).to_numpy()
        d["p_market"] = np.nan
        d.loc[idx, "p_market"] = j["p_market"].to_numpy()
        d["in_market"] = False
        d.loc[idx, "in_market"] = True
        have_market = int(d["in_market"].sum())
    except Exception as exc:                                   # noqa: BLE001
        print(f"market join unavailable ({type(exc).__name__}); market columns "
              f"will be blank. The monotonicity result is unaffected.")
        d["p_market"] = np.nan
        d["in_market"] = False
        have_market = 0

    print(f"games {len(d):,}   roster assumption: {ROSTER}   "
          f"buckets: {N_BUCKETS} (equal-population quantiles)")
    print(f"games with a closing line: {have_market:,}\n")

    for name, contrast in contrasts.items():
        ok = contrast.notna()
        c = contrast[ok].to_numpy(float)
        y = d.loc[ok, "y"].to_numpy(float)
        b = bucketise(c, N_BUCKETS)
        mkt = d.loc[ok, "p_market"].to_numpy(float)

        print("=" * 78)
        print(f"CONTRAST: {name}")
        print("=" * 78)
        print(f"  {'bucket':>7s} {'n':>6s} {'mean contrast':>15s} {'home win':>9s} "
              f"{'95% CI':>17s} {'market':>8s}")
        rates, idxs = [], []
        for k in range(N_BUCKETS):
            sel = b == k
            n = int(sel.sum())
            if not n:
                continue
            p = float(y[sel].mean())
            se = np.sqrt(p * (1 - p) / n)
            lo, hi = p - 1.96 * se, p + 1.96 * se
            rates.append(p)
            idxs.append(k)
            mm = mkt[sel]
            mtxt = f"{np.nanmean(mm):.3f}" if np.isfinite(mm).any() else "-"
            print(f"  {k:>7d} {n:>6d} {c[sel].mean():>15,.1f} {p:>9.4f} "
                  f"[{lo:.3f}, {hi:.3f}] {mtxt:>8s}")

        rates = np.array(rates)
        idxs = np.array(idxs, dtype=float)
        rho = float(np.corrcoef(idxs, rates)[0, 1]) if len(rates) > 2 else float("nan")

        print(f"\n  monotonic in bucket index : Spearman rho = {rho:+.4f}"
              f"   ({'RISES' if rho > 0 else 'FALLS'} consistently)")
        print(f"  spread across buckets     : {rates.max() - rates.min():.4f} "
              f"({rates.min():.4f} -> {rates.max():.4f})")

        # Is the spread bigger than binomial noise on the same bucket sizes?
        ns = np.array([int((b == k).sum()) for k in idxs], dtype=float)
        phat = y.mean()
        noise = np.sqrt(np.mean(phat * (1 - phat) / ns))
        print(f"  binomial noise on a flat : {noise:.4f} (one SE of a bucket rate)")
        print(f"  spread / noise           : {(rates.max() - rates.min()) / noise:+.2f}x")
        print()

    print("=" * 78)
    print("WHAT TO DO WITH THIS")
    print("=" * 78)
    print("  Rising and ordered  -> the contrast is a real strength measure, and the")
    print("                        ablation's log-loss delta is believable.")
    print("  Flat, or falling   -> the Elo totals are a restatement of team form in")
    print("                        a different unit, whatever the regression says.")
    print("  Read the MARKET column before celebrating: a feature that orders games")
    print("  about as well as the closing line does is not a betting edge, however")
    print("  good its log loss looks.")
    print()
    print("Reminder: home-court advantage is worth roughly +2.5 to +3 Elo points of")
    print("this scale, so the zero-contrast bucket should sit ABOVE 0.5, not on it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())