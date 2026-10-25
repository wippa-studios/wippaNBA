"""Tests for `src/Models/dynamic_ratings.py`.

The two properties that make a rating "dynamic" are also the two easiest to
fake: state that updates after the result, and an uncertainty term that actually
changes the prediction.  Both are pinned here.

* **Causality.** Flipping a future result must leave every earlier prediction
  bit-identical.  A model that recomputes a batch fit would fail this.
* **Uncertainty is used.** Two teams with the same rating gap but different RD
  (or state variance) must get different probabilities, and the less certain
  matchup must be closer to 0.5.
"""

import unittest

import numpy as np
import pandas as pd

from src.Models import dynamic_ratings as dr


def league(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = ["A", "B", "C", "D"]
    rows = []
    for i in range(n):
        home = teams[i % 4]
        away = teams[(i + 1) % 4]
        rows.append({
            "GAME_ID": f"00224{i:05d}",
            "GAME_DATE": pd.Timestamp("2024-11-01") + pd.Timedelta(days=i),
            "home": home,
            "away": away,
            "y": int(rng.random() < 0.55),
        })
    return pd.DataFrame(rows)


class TestGlicko(unittest.TestCase):
    def test_future_result_cannot_move_an_earlier_prediction(self):
        games = league(24)
        base = dr.fit_predict_glicko(games)
        flipped = games.copy()
        # Flip the second-to-last result.  The last prediction must move; the
        # predictions before the flipped game must not.
        flipped.loc[len(flipped) - 2, "y"] = 1 - flipped.loc[len(flipped) - 2, "y"]
        moved = dr.fit_predict_glicko(flipped)
        np.testing.assert_array_equal(base["p"][:-2], moved["p"][:-2])
        self.assertNotEqual(base["p"][-1], moved["p"][-1])

    def test_rd_widens_with_inactivity_and_is_capped(self):
        model = dr.GlickoRating()
        model.rating["A"] = 1500.0
        model.rd["A"] = 80.0
        model.last_date["A"] = pd.Timestamp("2024-11-01")
        model.inflate_inactivity("A", pd.Timestamp("2024-11-11"))
        self.assertGreater(model.rd["A"], 80.0)
        model.inflate_inactivity("A", pd.Timestamp("2026-01-01"))
        self.assertLessEqual(model.rd["A"], model.config.max_rd)

    def test_same_date_does_not_inflate(self):
        model = dr.GlickoRating()
        model.rating["A"] = 1500.0
        model.rd["A"] = 80.0
        model.last_date["A"] = pd.Timestamp("2024-11-01")
        model.inflate_inactivity("A", pd.Timestamp("2024-11-01"))
        self.assertEqual(model.rd["A"], 80.0)

    def test_home_edge_is_separate_from_team_rating(self):
        no_edge = dr.GlickoRating(dr.GlickoConfig(hfa=0.0))
        self.assertAlmostEqual(no_edge.prob_home("A", "B"), 0.5)
        home_edge = dr.GlickoRating(dr.GlickoConfig(hfa=70.0))
        self.assertGreater(home_edge.prob_home("A", "B"), 0.5)
        self.assertEqual(home_edge.rating, {})

    def test_more_uncertain_matchup_is_closer_to_a_coin_flip(self):
        sure = dr.GlickoRating()
        sure.rating.update({"A": 1600.0, "B": 1500.0})
        sure.rd.update({"A": 30.0, "B": 30.0})
        unsure = dr.GlickoRating()
        unsure.rating.update({"A": 1600.0, "B": 1500.0})
        unsure.rd.update({"A": 350.0, "B": 350.0})
        self.assertGreater(sure.prob_home("A", "B"), unsure.prob_home("A", "B"))
        self.assertGreater(unsure.prob_home("A", "B"), 0.5)

    def test_predictions_are_aligned_to_the_caller_row_order(self):
        games = league(12)
        base = dr.fit_predict_glicko(games)
        shuffled = games.sample(frac=1.0, random_state=5).reset_index(drop=True)
        out = dr.fit_predict_glicko(shuffled)
        for i, gid in enumerate(games["GAME_ID"]):
            j = int(shuffled.index[shuffled["GAME_ID"] == gid][0])
            self.assertAlmostEqual(out["p"][j], base["p"][i])
        self.assertTrue(np.isfinite(out["p"]).all())


class TestStateSpaceBT(unittest.TestCase):
    def test_future_result_cannot_move_an_earlier_prediction(self):
        games = league(24)
        base = dr.fit_predict_state_space_bt(games)
        flipped = games.copy()
        flipped.loc[len(flipped) - 2, "y"] = 1 - flipped.loc[len(flipped) - 2, "y"]
        moved = dr.fit_predict_state_space_bt(flipped)
        np.testing.assert_array_equal(base["p"][:-2], moved["p"][:-2])
        self.assertNotEqual(base["p"][-1], moved["p"][-1])

    def test_variance_shrinks_after_a_game_and_grows_with_inactivity(self):
        model = dr.StateSpaceBT()
        model.mu["A"] = 0.0
        model.var["A"] = 1.0
        model.last_date["A"] = pd.Timestamp("2024-11-01")
        before = model.var["A"]
        model.update("A", "B", 1)
        after_game = model.var["A"]
        self.assertLess(after_game, before)
        model.last_date["A"] = pd.Timestamp("2024-11-01")
        model.inflate_inactivity("A", pd.Timestamp("2024-12-01"))
        self.assertGreater(model.var["A"], after_game)

    def test_home_edge_separate_and_uncertainty_used(self):
        no_edge = dr.StateSpaceBT(dr.StateSpaceConfig(hfa_elo_points=0.0))
        self.assertAlmostEqual(no_edge.prob_home("A", "B"), 0.5)
        sure = dr.StateSpaceBT()
        sure.mu.update({"A": 0.4, "B": 0.0})
        sure.var.update({"A": 0.01, "B": 0.01})
        unsure = dr.StateSpaceBT()
        unsure.mu.update({"A": 0.4, "B": 0.0})
        unsure.var.update({"A": 1.0, "B": 1.0})
        self.assertGreater(sure.prob_home("A", "B"), unsure.prob_home("A", "B"))
        self.assertGreater(unsure.prob_home("A", "B"), 0.5)

    def test_uncertainty_columns_are_pre_game(self):
        out = dr.fit_predict_state_space_bt(league(10))
        self.assertTrue(np.isfinite(out["var_home"]).all())
        self.assertTrue((out["var_home"] >= out["config"].min_variance).all())


if __name__ == "__main__":
    unittest.main()
