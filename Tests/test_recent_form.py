"""Contract tests for the issue #60 follow-on recent-form window features.

Pins, from the strongest claim down:

* WINDOW COUNTING SEMANTICS on synthetic schedules: wins over the last up-to-W
  games entering tonight, strictly-before, windows crossing seasons (NOT
  reset), partial windows, and the sliding-window accounting itself;
* TRUNCATION INVARIANCE on the real data;
* the BRACKET LADDERS (cumulative >=K, full ladder, both sides);
* the RECIPES: ``full_form5`` / ``full_form10`` = canonical 26 PLUS the ladders,
  and the canonical 26 untouched.
"""

import unittest

import pandas as pd

from src.DataProviders import nba_data as nd
from src.Models import logreg_rolling as lr


def mini_games(rows):
    df = pd.DataFrame(rows, columns=["GAME_ID", "GAME_DATE", "SEASON",
                                     "home", "away", "home_pts", "away_pts"])
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def W(d, season, pid, home, ah):
    return (f"{season}-{pid}", d, season, home, ah, 0.0, 0.0)


def home_game(d, season, pid, home_won: bool, home_pts=110.0, away_pts=100.0):
    """GSW home; if home_won the away side loses it."""
    if home_won:
        return (f"{season}-{pid}", d, season, "GSW", "CHA", home_pts, away_pts)
    return (f"{season}-{pid}", d, season, "GSW", "CHA", away_pts, home_pts)


class TestFormWindows(unittest.TestCase):
    def test_win_count_over_last_up_to_5_strictly_before(self):
        # GSW home five straight days: wins, wins, loss, win, win.
        # Entering counts (window 5, no season reset):
        #  game1: 0 ; g2: 1 ; g3: 2 ; g4: 0 (3rd was a loss) ; g5: 1
        g = mini_games([
            home_game("2020-01-01", "2019-20", 1, True),
            home_game("2020-01-02", "2019-20", 2, True),
            home_game("2020-01-03", "2019-20", 3, False),
            home_game("2020-01-04", "2019-20", 4, True),
            home_game("2020-01-05", "2019-20", 5, True),
        ])
        rf = nd.recent_form_features(g, windows=(5,)).reset_index(drop=True)
        # WINS in the last up-to-5, entering tonight: a loss reduces the count,
        # it does not erase the earlier wins in the window.
        self.assertEqual(list(rf["home_form5_enter"]), [0, 1, 2, 2, 3])
        # Away (CHA) mirrors: it lost g1,g2,g4,g5 and won g3.
        self.assertEqual(list(rf["away_form5_enter"]), [0, 0, 0, 1, 1])

    def test_window_slides_past_W_games(self):
        # Six straight wins: counts entering are 0,1,2,3,4,5 (window 5).
        g = mini_games([home_game(f"2020-01-0{i}", "2019-20", i, True)
                        for i in range(1, 7)])
        rf = nd.recent_form_features(g, windows=(5,)).reset_index(drop=True)
        self.assertEqual(list(rf["home_form5_enter"]), [0, 1, 2, 3, 4, 5])

    def test_windows_cross_seasons_deliberately(self):
        # GSW wins the last 2015-16 game, then wins the first 2016-17 game.
        # Recent FORM does NOT reset at the season (unlike streaks): the opener
        # enters the window carrying the prior season's win.
        g = mini_games([
            home_game("2016-04-10", "2015-16", 1, True),
            home_game("2016-10-20", "2016-17", 2, True),
            home_game("2016-10-21", "2016-17", 3, True),
        ])
        rf = nd.recent_form_features(g, windows=(5,)).reset_index(drop=True)
        self.assertEqual(list(rf["home_form5_enter"]), [0, 1, 2])

    def test_partial_window_uses_available_games(self):
        # Only two prior games: the count is out of 2, never more.
        g = mini_games([
            home_game("2020-01-01", "2019-20", 1, True),
            home_game("2020-01-02", "2019-20", 2, True),
        ])
        rf = nd.recent_form_features(g, windows=(5,)).reset_index(drop=True)
        self.assertEqual(list(rf["home_form5_enter"]), [0, 1])
        self.assertEqual(int(rf.isna().sum().sum()), 0)

    def test_ten_game_window_counts_ten(self):
        # Ten games: nine straight wins then a loss before the 11th.
        g = mini_games([home_game(f"2020-01-0{i}", "2019-20", i, True)
                        for i in range(1, 10)] +
                       [home_game("2020-01-10", "2019-20", 10, False),
                        home_game("2020-01-11", "2019-20", 11, True)])
        rf = nd.recent_form_features(g, windows=(10,)).reset_index(drop=True)
        # Entering win counts over the last up-to-10: 0..9 across the nine
        # wins and the loss, then game 11 enters with the last 10 prior games
        # (nine wins + one loss) = 9.
        self.assertEqual(list(rf["home_form10_enter"]),
                         [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9])


class TestFormTruncation(unittest.TestCase):
    def test_form_windows_are_truncation_invariant(self):
        g = nd.build_games().reset_index(drop=True)
        n = 3000
        full = nd.recent_form_features(g).reset_index(drop=True)
        cut = nd.recent_form_features(g.iloc[:n].reset_index(drop=True)
                                      ).reset_index(drop=True)
        for col in full.columns:
            self.assertTrue((full[col].iloc[:n].reset_index(drop=True)
                             == cut[col]).all(), col)
        self.assertEqual(int(full.isna().sum().sum()), 0)


class TestFormBrackets(unittest.TestCase):
    def test_ladder_is_full_and_cumulative(self):
        counts = pd.DataFrame({"home_form5_enter": [0.0, 3.0, 5.0],
                               "away_form5_enter": [1.0, 2.0, 0.0]})
        br = lr.form_brackets(counts, 5)
        self.assertEqual(list(br.columns),
                         [f"{s}_form5_ge{k}"
                          for s in ("home", "away") for k in range(1, 6)])
        # count 0 -> all 0; count 3 -> ge1..3 = 1, ge4/ge5 = 0; count 5 -> all 1.
        self.assertEqual(list(br.loc[0][:5]), [0.0] * 5)
        self.assertEqual(list(br.loc[1][:5]), [1.0, 1.0, 1.0, 0.0, 0.0])
        self.assertEqual(list(br.loc[2][:5]), [1.0] * 5)
        # monotone within each side
        for side in ("home", "away"):
            for k_hi, k_lo in ((5, 4), (4, 3), (3, 2), (2, 1)):
                self.assertTrue(
                    (br[f"{side}_form5_ge{k_hi}"]
                     <= br[f"{side}_form5_ge{k_lo}"]).all())


class TestFormRecipe(unittest.TestCase):
    def test_recipes_are_canonical_plus_ladders(self):
        full = lr.resolve_features(lr.DEFAULT_MODELS["full"])
        f5 = lr.resolve_features(lr.DEFAULT_MODELS["full_form5"])
        f10 = lr.resolve_features(lr.DEFAULT_MODELS["full_form10"])
        self.assertEqual(len(full), 26)
        self.assertEqual(len(f5), 36)
        self.assertEqual(len(f10), 46)
        self.assertEqual(f5[:26], full)
        self.assertEqual(f10[:26], full)
        self.assertEqual(f5[26:], [f"{s}_form5_ge{k}"
                                   for s in ("home", "away")
                                   for k in range(1, 6)])
        self.assertEqual(f10[26:], [f"{s}_form10_ge{k}"
                                    for s in ("home", "away")
                                    for k in range(1, 11)])

    def test_canonical_recipe_unchanged(self):
        full = lr.resolve_features(lr.DEFAULT_MODELS["full"])
        self.assertFalse(any("form5" in c or "form10" in c for c in full))


if __name__ == "__main__":
    unittest.main()