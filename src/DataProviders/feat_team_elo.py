"""Team Elo ratings as E14 features.

WHY THIS EXISTS
---------------
`feature_block_ablation.json` lists thirteen feature blocks. None of them is a
team rating. The frame's every strength estimate is an exponentially weighted
mean of the last ~10 box scores, which is a LINEAR function of recent margins.
Elo is a recursive multiplicative update weighted by margin of victory: it
moves a team hard after a 30-point loss and barely at all after a 2-point win,
and it forgets on its own schedule through `decay_half_life_days` and
`season_reversion`.

So the two carry partly different information about the same games, which is
the only reason it is worth adding an estimator the frame already half-contains.
`baselines_results.md` reports the standalone Elo at log loss 0.6234 against the
E14 frame's 0.6183: comparable accuracy from a completely different estimator,
which is exactly the profile of something worth combining rather than replacing.

LEAKAGE
-------
`fit_predict_elo` is an online recursion and `rating_home_before` is the state
PRIOR to each game's own update, so no game's result reaches its own feature.
The one thing that needs saying rather than assuming: within a walk-forward test
block the ratings keep updating from that block's earlier results. That is
legitimate under this project's deployment convention - tonight's prediction may
use last night's results - and it is the SAME convention the E14 EWMs already
use, since `rolling_team_features` shifts over the whole frame and a test row
therefore also sees its block's earlier games. The two blocks are measured on
equal terms.

`decay_half_life_days` reads the calendar, not the result, so truncation cannot
change it for a surviving row. `scripts/round3_verify.py::gate1_truncation`
proves it mechanically rather than trusting the argument.

NOT A NEW RESULT
----------------
This is the team-level version of a mechanism the project already measured at
+0.00065 once its membership was repaired (`player_elo`, honest value). The
difference here is that there is no membership question at all: a team rating
depends only on games that team played.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _c in (ROOT, ROOT / "src"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from src.DataProviders import nba_data as nd      # noqa: E402

HFA = 70.0        # the project's tuned home-court term, points


def team_elo_features(games: pd.DataFrame, k: float = 10.0,
                      home_advantage: float = HFA,
                      season_reversion: float = 0.75) -> pd.DataFrame:
    """One causal Elo pass; pre-game ratings as per-side features.

    Columns:
      ``home_elo``/``away_elo``   pre-game rating, points, ~N(0, 104)
      ``home_elo_adj``/``away_elo_adj``  the same with the home term added to
            the home side, so the two levels are directly comparable
      ``elo_diff``      home minus away, RAW contrast
      ``elo_diff_adj``  home+70 minus away, the home-advantage-adjusted contrast.
            NOT separable from ``elo_diff`` in a model with an intercept -- it is
            ``elo_diff`` plus a constant. It is kept because the contrast reads
            more clearly against a fixed home term, not because it adds a degree
            of freedom. The promoted 48-column frame uses ``elo_diff`` alone.
      ``elo_edge``      ``elo_diff_adj`` minus the home team's own
            ``home_home_edge``, i.e. how much of the rating gap the frame's
            home-advantage estimate has NOT already explained. NOT part of any
            scored arm in round 3; reported here so the next reader does not
            re-derive it.

    K=10 is the value `runs/baselines_results.md` selected on the last 20% of
    each training block. Test rows never entered that choice, and the sweep is
    published in that file, so it is not re-tuned here.
    """
    from src.Models import baselines as bl

    gg = games.rename(columns={"y": "home_win", "GAME_DATE": "date"})
    fit = bl.elo_rating_pass(gg, k=k, home_advantage=home_advantage,
                             same_day="batched", use_margin=True,
                             season_reversion=season_reversion)
    gf = fit.games
    if len(gf) != len(games):
        raise AssertionError(f"elo pass returned {len(gf)} rows for {len(games)} games")

    rh = gf["rating_home_before"].to_numpy(dtype=float)
    ra = gf["rating_away_before"].to_numpy(dtype=float)
    out = pd.DataFrame({
        "home_elo": rh, "away_elo": ra,
        "home_elo_adj": rh + home_advantage, "away_elo_adj": ra,
        "elo_diff": rh - ra,
        "elo_diff_adj": rh + home_advantage - ra,
    })

    # The frame already carries a home term for the EWM margin. Reporting the
    # residual against it says whether Elo adds anything beyond a better estimate
    # of the same gap, or a second, partly independent one.
    #
    # `home_home_edge` is emitted by `opponent_adjusted_features`, NOT by
    # `rolling_team_features`. An earlier version of this module looked in the
    # wrong builder, found nothing, and silently emitted no `elo_edge` at all
    # while its docstring promised the column -- a docstring that lied about the
    # output, which is worse than a missing feature. The `assert` below is the
    # fix: the column is either there or the build fails.
    oa = nd.opponent_adjusted_features(games, halflife=10.0)
    if "home_home_edge" not in oa.columns:
        raise AssertionError(
            "opponent_adjusted_features no longer emits home_home_edge; "
            "feat_team_elo.elo_edge depends on it and must be revisited")
    edge = oa["home_home_edge"].to_numpy(dtype=float)
    out["elo_edge"] = out["elo_diff_adj"].to_numpy() - edge

    return out.reset_index(drop=True)