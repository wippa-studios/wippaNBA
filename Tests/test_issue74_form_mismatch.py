"""Contract tests for issue #74's form-mismatch spot.

The spot is a simple rule, but four pieces are easy to get wrong and would
silently change the measurement:

* the form window must use STRICTLY PRIOR games, and the cross-season helper
  must equal the shared builder that I60b used (so the two definitions cannot
  drift);
* the within-season variant must actually reset (it is the sensitivity);
* a 2-0 team in October must not satisfy "2-8 or worse": the history mask is
  part of the rule;
* ATS settlement must treat the line as the home-cover threshold, void a push,
  and price every bet at -110; ML settlement must use the posted price, not a
  de-vigged probability.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import issue74_form_mismatch as i74  # noqa: E402
from DataProviders import nba_data as nd  # noqa: E402


def tiny_games(season1_wins=10, season2_wins=0, games_per_season=10):
    """Two teams, two seasons, a known winner in every game.

    Team A wins the first ``season1_wins`` games of season 1 and the first
    ``season2_wins`` games of season 2; B wins the rest.  A is home on odd
    game numbers, so both sides' form columns get exercised.
    """
    rows = []
    gid = 0
    for season, wins in (("S1", season1_wins), ("S2", season2_wins)):
        for i in range(games_per_season):
            gid += 1
            a_home = i % 2 == 0
            a_won = i < wins
            home, away = ("A", "B") if a_home else ("B", "A")
            home_pts, away_pts = ((110, 100) if a_won == a_home else (100, 110))
            rows.append({
                "GAME_ID": f"{season}-{i:02d}",
                "GAME_DATE": pd.Timestamp("2024-10-01") + pd.Timedelta(
                    days=(0 if season == "S1" else 200) + i),
                "SEASON": season,
                "home": home, "away": away,
                "home_pts": home_pts, "away_pts": away_pts,
            })
    return pd.DataFrame(rows)


class TestFormWindows(unittest.TestCase):

    def test_cross_season_helper_matches_the_shared_builder(self):
        games = tiny_games()
        got = i74.form_wins_entering(games, window=10, reset_seasons=False)
        expected = nd.recent_form_features(games, windows=(10,))
        np.testing.assert_array_equal(
            got["home_form10_enter"].to_numpy(),
            expected["home_form10_enter"].to_numpy())
        np.testing.assert_array_equal(
            got["away_form10_enter"].to_numpy(),
            expected["away_form10_enter"].to_numpy())

    def test_within_season_does_not_carry_last_season_form(self):
        games = tiny_games(season1_wins=10, season2_wins=0)
        cross = i74.form_wins_entering(games, window=10, reset_seasons=False)
        within = i74.form_wins_entering(games, window=10, reset_seasons=True)
        # First season-2 game: A is home (i=0) and lost every season-1 game?
        # A won all of season 1, so the cross-season window is 10 wins and the
        # within-season window is 0.
        first_s2 = games.index[games["SEASON"] == "S2"][0]
        self.assertEqual(cross.loc[first_s2, "home_form10_enter"], 10.0)
        self.assertEqual(within.loc[first_s2, "home_form10_enter"], 0.0)

    def test_form_uses_only_prior_games(self):
        games = tiny_games()
        before = i74.form_wins_entering(games, window=10)
        flipped = games.copy()
        last = flipped.index[-1]
        flipped.loc[last, ["home_pts", "away_pts"]] = (
            flipped.loc[last, ["away_pts", "home_pts"]].to_numpy())
        after = i74.form_wins_entering(flipped, window=10)
        # Changing the last game's result cannot change the last game's
        # entering form (or any earlier row's).
        np.testing.assert_array_equal(
            before["home_form10_enter"].to_numpy(),
            after["home_form10_enter"].to_numpy())
        np.testing.assert_array_equal(
            before["away_form10_enter"].to_numpy(),
            after["away_form10_enter"].to_numpy())


class TestRules(unittest.TestCase):

    def frame(self, h, a, hp, ap):
        form = pd.DataFrame({"home_form10_enter": h, "away_form10_enter": a})
        history = pd.DataFrame({"home_gp_enter": hp, "away_gp_enter": ap})
        return i74.spot_masks(form, history)

    def test_exact_mirror_and_gap_cells(self):
        masks = self.frame([5, 4, 5, 2, 6], [2, 2, 3, 5, 0],
                           [10] * 5, [10] * 5)
        np.testing.assert_array_equal(
            masks["exact"], [True, False, False, False, True])
        np.testing.assert_array_equal(
            masks["mirror"], [False, False, False, True, False])
        np.testing.assert_array_equal(
            masks["gap_ge2"], [True, True, True, False, True])
        np.testing.assert_array_equal(
            masks["gap_ge4"], [False, False, False, False, True])

    def test_a_short_history_cannot_satisfy_the_cold_cell(self):
        # 2-0 is "<= 2 wins", but it is not 2-8.
        masks = self.frame([5], [0], [10], [2])
        self.assertFalse(bool(masks["exact"][0]))
        masks = self.frame([5], [0], [10], [10])
        self.assertTrue(bool(masks["exact"][0]))


class TestSettlement(unittest.TestCase):

    def test_ats_uses_the_home_cover_threshold_and_voids_pushes(self):
        margin = np.array([5.0, 6.0, 4.0])
        line = np.array([5.0, 5.0, 5.0])       # home favoured by 5
        out = i74.settle_ats(np.array([True, True, True]), margin, line, "home")
        self.assertEqual(out["n_bets"], 3)
        self.assertEqual(out["n_pushes"], 1)
        self.assertAlmostEqual(out["win_rate"], 0.5)   # 6 covers, 4 does not
        self.assertAlmostEqual(out["total_profit"],
                               (1.0 / i74.P_PRICE - 1.0) - 1.0)

    def test_away_settles_on_the_other_side_of_the_line(self):
        margin = np.array([5.0, 4.0])
        line = np.array([5.0, 5.0])
        out = i74.settle_ats(np.array([False, True]), margin, line, "away")
        self.assertEqual(out["n_bets"], 1)
        self.assertEqual(out["n_pushes"], 0)   # the push row was not bet
        self.assertAlmostEqual(out["win_rate"], 1.0)   # 4 < 5

    def test_ml_uses_the_posted_price(self):
        out = i74.settle_ml(np.array([True, True]), np.array([1, 0]),
                            np.array([0.5, 0.5]), np.array([0.5, 0.5]),
                            "home")
        self.assertAlmostEqual(out["roi"], 0.0)
        out = i74.settle_ml(np.array([True, True]), np.array([1, 0]),
                            np.array([0.25, 0.25]), np.array([0.75, 0.75]),
                            "home")
        self.assertAlmostEqual(out["roi"], 1.0)        # +3 and -1 over two bets

    def test_roi_bootstrap_is_deterministic(self):
        profit = np.array([1.0, -1.0, 1.0, 1.0, -1.0])
        stake = np.ones(5)
        a = i74._roi(profit, stake)
        b = i74._roi(profit, stake)
        self.assertEqual(a["se"], b["se"])
        self.assertEqual(a["z"], b["z"])


if __name__ == "__main__":
    unittest.main()
