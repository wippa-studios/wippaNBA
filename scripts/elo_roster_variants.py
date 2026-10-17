"""Compare roster assumptions for the player-Elo block.

"Available" is not a fact a pre-game model can look up. It is an assumption, and
different assumptions give materially different team totals. Rather than pick one,
the assumption is a parameter of `nba_data.player_elo_features` and every variant
is built here so the numbers sit side by side.

The absolute totals are NOT comparable across constructions: `all_prior` sums over
roughly 6.4x a real rotation, because a long-departed player's trailing EWMA decays
without ever reaching zero. Only `elo_mean` is scale-comparable, and only the
predictive value of any of them is comparable at all - which is what
`scripts/feature_block_ablation.py` scores.

`confirmed` (the real submitted starting five) is listed for completeness and is
NOT computable here. Starting lineups are filed 30-60 minutes before tip-off, so
they belong to a tip-off prediction point and not to a model run that morning.
That is a prediction-point decision, not a data gap, and it is recorded as a
parameter so the distinction has to be made explicitly rather than by accident.

RUN: python scripts/elo_roster_variants.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from DataProviders import nba_data as nd  # noqa: E402

VARIANTS = ("all_prior", "prev_game", "top_minutes")


def main() -> int:
    games = nd.build_games()
    late_at = int(len(games) * 0.7)

    print(f"games {len(games):,}")
    print(f"construction  assumption                                 "
          f"elo_avail(mean)  elo_mean  rows/rot  coverage")
    print("-" * 96)

    frames = {}
    for mode in VARIANTS:
        t0 = time.time()
        f = nd.player_elo_features(games, roster_mode=mode)
        frames[mode] = f
        late = f.iloc[late_at:]

        # How many player-rows does this construction sum over, per team-game?
        ap = nd.player_elo_ratings(games)
        ap = ap.sort_values(["PLAYER_ID", "GAME_DATE"])
        ap["exp_min"] = nd._ewm_by_player(ap, "MIN", 15.0)
        tg = (games[["GAME_ID", "GAME_DATE", "home", "away"]]
              .melt(id_vars=["GAME_ID", "GAME_DATE"], value_vars=["home", "away"],
                    value_name="team")
              .sort_values(["team", "GAME_DATE"]).reset_index(drop=True))
        roster = nd._asof_roster(tg, ap, roster=mode)
        per_rot = len(roster) / (len(games) * 2)

        note = {
            "all_prior": "everyone with any earlier appearance",
            "prev_game": "only the previous game's participants",
            "top_minutes": "top 5 by trailing expected minutes",
        }[mode]
        print(f"  {mode:<12s} {note:<41s} "
              f"{late.home_elo_available.mean():>12,.0f} "
              f"{late.home_elo_mean.mean():>9.1f} "
              f"{per_rot:>8.2f} {late.home_elo_available.notna().mean() * 100:>7.1f}%"
              f"  ({time.time() - t0:.0f}s)")

    print()
    print("How similar are the constructions to each other?")
    print("  If the choice of assumption barely moves the feature, the question is")
    print("  moot. If it moves a lot, the assumption has to be stated with every")
    print("  number that depends on it.")
    print()
    print(f"  {'pair':<34s} {'corr(elo_mean)':>15s} {'corr(elo_available)':>20s}")
    print("  " + "-" * 70)
    for i, a in enumerate(VARIANTS):
        for b in VARIANTS[i + 1:]:
            fa, fb = frames[a].iloc[late_at:], frames[b].iloc[late_at:]
            c1 = fa["home_elo_mean"].corr(fb["home_elo_mean"])
            c2 = fa["home_elo_available"].corr(fb["home_elo_available"])
            print(f"  {a + ' vs ' + b:<34s} {c1:>15.4f} {c2:>20.4f}")

    print()
    print("`confirmed` - the actual submitted starting five - is not in this table.")
    print("Starting lineups are filed 30-60 minutes before tip-off, so they are a")
    print("legitimate input to a TIP-OFF prediction and an impossible one for a model")
    print("run that morning. Fetching them would not improve this model; it would")
    print("be a different model with a different prediction time.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())