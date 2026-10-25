"""Tests for `src/Models/ensemble.py`.

Three small, checkable properties:

* a blend is the arithmetic mean it claims to be, and refuses silent
  renormalisation;
* the simplex grid enumerates every point and nothing else (66 for three
  models at step 0.1);
* the disagreement table counts the right rows and the conditional hit rates
  are computed on those rows only.
"""

import unittest

import numpy as np

from src.Models import ensemble as en


class TestBlend(unittest.TestCase):
    def test_equal_weights_are_the_arithmetic_mean(self):
        p = {"a": np.array([0.2, 0.8]), "b": np.array([0.4, 0.6])}
        out = en.blend(p, {"a": 0.5, "b": 0.5})
        np.testing.assert_allclose(out, [0.3, 0.7])

    def test_weights_must_sum_to_one(self):
        p = {"a": np.array([0.2]), "b": np.array([0.8])}
        with self.assertRaises(ValueError):
            en.blend(p, {"a": 0.2, "b": 0.2})

    def test_negative_weight_rejected(self):
        p = {"a": np.array([0.2]), "b": np.array([0.8])}
        with self.assertRaises(ValueError):
            en.blend(p, {"a": 1.5, "b": -0.5})

    def test_weight_names_must_match(self):
        p = {"a": np.array([0.2]), "b": np.array([0.8])}
        with self.assertRaises(ValueError):
            en.blend(p, {"a": 1.0})


class TestSimplexGrid(unittest.TestCase):
    def test_three_models_step_tenth_has_66_points(self):
        grid = en.simplex_grid(["a", "b", "c"], step=0.1)
        self.assertEqual(len(grid), 66)
        self.assertEqual(len({tuple(sorted(g.items())) for g in grid}), 66)
        for g in grid:
            self.assertAlmostEqual(sum(g.values()), 1.0)
            self.assertTrue(all(v >= 0 for v in g.values()))

    def test_step_must_divide_one(self):
        with self.assertRaises(ValueError):
            en.simplex_grid(["a", "b"], step=0.3)


class TestDisagreement(unittest.TestCase):
    def test_counts_and_hit_rates_on_disagreement_rows(self):
        y = np.array([1, 1, 0, 0])
        p = {
            "a": np.array([0.9, 0.6, 0.4, 0.1]),
            "b": np.array([0.9, 0.4, 0.6, 0.1]),
        }
        table = en.disagreement_table(y, p)
        row = table["a vs b"]
        # Rows 2 and 3 (0-based 1 and 2) are the disagreements.
        self.assertEqual(row["n_disagree"], 2)
        self.assertEqual(row["disagreement_rate"], 0.5)
        # a predicts 1 and 0 there: both right. b predicts 0 and 1: both wrong.
        self.assertEqual(row["a_hit_rate_on_disagreement"], 1.0)
        self.assertEqual(row["b_hit_rate_on_disagreement"], 0.0)

    def test_no_disagreement_is_nan_not_zero(self):
        y = np.array([1, 0, 1, 0])
        p = {"a": np.array([0.9, 0.1, 0.9, 0.1]),
             "b": np.array([0.8, 0.2, 0.8, 0.2])}
        row = en.disagreement_table(y, p)["a vs b"]
        self.assertEqual(row["n_disagree"], 0)
        self.assertTrue(np.isnan(row["a_hit_rate_on_disagreement"]))


if __name__ == "__main__":
    unittest.main()
