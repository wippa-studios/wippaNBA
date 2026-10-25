"""Contract tests for the issue #60 momentum streak features.

Pins, from the strongest claim down:

* the STREAK SEMANTICS on synthetic schedules: consecutive wins count, a loss
  resets, a season change resets, and the current game's outcome never enters
  its own streak (strictly-before causality);
* the LOSS-streak mirror;
* TRUNCATION INVARIANCE on the real data (the schedule leak class);
* the BRACKET SHAPING (cumulative >=K indicators, monotone);
* the RECIPE: ``full_momentum`` = the canonical 26 PLUS the 8 brackets, and the
  canonical 26 is untouched.
"""

import unittest

import numpy as np
import pandas as pd

from src.DataProviders import nba_data as nd
from src.Models import logreg_rolling as lr


def mini_games(rows):
    df = pd.DataFrame(rows, columns=["GAME_ID", "GAME_DATE", "SEASON",
                                     "home", "away", "home_pts", "away_pts",
                                     "y"])
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


GSW_W = lambda d, s="2015-16", pid=None: (f"{s}-{pid or d}", d, s, "GSW", "CHA",
                                           115.0, 90.0, 1)
CHAW = lambda d, s="2015-16", pid=None: (f"{s}-{pid or d}", d, s, "CHA", "GSW",
                                          115.0, 90.0, 0)
GSW_L = lambda d, s="2015-16", pid=None: (f"{s}-{pid or d}", d, s, "GSW", "CHA",
                                          90.0, 115.0, 0)


class TestStreakSemantics(unittest.TestCase):
    def test_consecutive_wins_count_and_current_game_is_excluded(self):
        g = mini_games([
            GSW_W("2020-01-02", pid=1),
            GSW_W("2020-01-03", pid=2),
            GSW_W("2020-01-04", pid=3),
        ])
        sf = nd.streak_features(g).reset_index(drop=True)
        # Home (GSW) enters game 1 at 0, game 2 at 1, game 3 at 2.
        self.assertEqual(list(sf["home_streak_enter"]), [0, 1, 2])
        # The away side (CHA) loses every game: loss streaks 0,1,2; wins 0.
        self.assertEqual(list(sf["away_streak_enter"]), [0, 0, 0])
        self.assertEqual(list(sf["away_loss_streak_enter"]), [0, 1, 2])

    def test_a_loss_resets_the_win_streak(self):
        g = mini_games([
            GSW_W("2020-01-02"), GSW_L("2020-01-03"), GSW_W("2020-01-04"),
        ])
        sf = nd.streak_features(g).reset_index(drop=True)
        self.assertEqual(list(sf["home_streak_enter"]), [0, 1, 0])
        self.assertEqual(list(sf["home_loss_streak_enter"]), [0, 0, 1])

    def test_a_season_change_resets_the_streak(self):
        # GSW ends 2015-16 with a win, but the 2016-17 opener enters at 0:
        # the off-season is not momentum.
        g = mini_games([
            GSW_W("2016-04-10", "2015-16", pid=1),
            GSW_W("2016-10-20", "2016-17", pid=2),
        ])
        sf = nd.streak_features(g).reset_index(drop=True)
        self.assertEqual(list(sf["home_streak_enter"]), [0, 0])

    def test_win_and_loss_streaks_are_mutually_exclusive(self):
        # W W L W: the streak enters 0,1,2 on the first three; the loss on
        # game 3 only breaks it for game 4 (a game's own result never enters
        # its own streak). At every row at most one of win/loss can be > 0.
        g = mini_games([
            GSW_W("2020-01-02"), GSW_W("2020-01-03"),
            GSW_L("2020-01-04"), GSW_W("2020-01-05"),
        ])
        sf = nd.streak_features(g).reset_index(drop=True)
        self.assertEqual(list(sf["home_streak_enter"]), [0, 1, 2, 0])
        self.assertEqual(list(sf["home_loss_streak_enter"]), [0, 0, 0, 1])
        cross = sf["home_streak_enter"] * sf["home_loss_streak_enter"]
        self.assertEqual(cross.sum(), 0.0)


class TestStreakTruncation(unittest.TestCase):
    def test_streaks_are_truncation_invariant(self):
        g = nd.build_games().reset_index(drop=True)
        n = 3000
        full = nd.streak_features(g).reset_index(drop=True)
        cut = nd.streak_features(g.iloc[:n].reset_index(drop=True)
                                 ).reset_index(drop=True)
        for col in ("home_streak_enter", "away_streak_enter",
                    "home_loss_streak_enter", "away_loss_streak_enter"):
            self.assertTrue(
                (full[col].iloc[:n].reset_index(drop=True) == cut[col]).all(),
                col)
        self.assertEqual(int(full.isna().sum().sum()), 0)


class TestBrackets(unittest.TestCase):
    def _b(self, streak):
        enter = pd.DataFrame({"home_streak_enter": [streak],
                              "away_streak_enter": [0]})
        return lr.momentum_brackets(enter)

    def test_brackets_are_cumulative_thresholds(self):
        b7 = self._b(7)
        for c in ("home_streak_ge2", "home_streak_ge3",
                  "home_streak_ge4", "home_streak_ge6"):
            self.assertEqual(b7.loc[0, c], 1.0, c)
        b5 = self._b(5)
        self.assertEqual(b5.loc[0, "home_streak_ge2"], 1.0)
        self.assertEqual(b5.loc[0, "home_streak_ge3"], 1.0)
        self.assertEqual(b5.loc[0, "home_streak_ge4"], 1.0)
        self.assertEqual(b5.loc[0, "home_streak_ge6"], 0.0)
        b0 = self._b(0)
        self.assertEqual(list(b0[["home_streak_ge2"]].iloc[0]), [0.0])

    def test_brackets_are_monotone_in_streak(self):
        g = nd.build_games().reset_index(drop=True)
        mom = lr.momentum_brackets(nd.streak_features(g).reset_index(drop=True))
        for k_hi, k_lo in ((3, 2), (4, 3), (6, 4)):
            self.assertTrue(
                (mom[f"home_streak_ge{k_hi}"] <= mom[f"home_streak_ge{k_lo}"]).all())
            self.assertTrue(
                (mom[f"away_streak_ge{k_hi}"] <= mom[f"away_streak_ge{k_lo}"]).all())


class TestMomentumRecipe(unittest.TestCase):
    def test_full_momentum_is_canonical_plus_eight(self):
        full = lr.resolve_features(lr.DEFAULT_MODELS["full"])
        fm = lr.resolve_features(lr.DEFAULT_MODELS["full_momentum"])
        self.assertEqual(len(full), 26)
        self.assertEqual(len(fm), 34)
        self.assertEqual(fm[:26], full)
        self.assertEqual(
            fm[26:],
            [f"{s}_streak_ge{k}"
             for s in ("home", "away") for k in lr.MOMENTUM_BRACKETS])

    def test_canonical_recipe_is_unchanged(self):
        full = lr.resolve_features(lr.DEFAULT_MODELS["full"])
        self.assertFalse(any("streak" in c for c in full))


if __name__ == "__main__":
    unittest.main()