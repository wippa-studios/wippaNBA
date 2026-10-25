"""Tests for `src/Models/measurement.py`.

The shared harness is the reference every remaining issue is scored against, so
the properties pinned here are the ones a wrong harness would silently break:

1. **The E14 column contract.** The 43 columns are frozen in a recorded order.
   A reordered frame still fits and still produces a plausible number, so the
   order is asserted rather than assumed.
2. **The delta sign.** ``positive = candidate better`` is the opposite of the
   raw bootstrap field in one place; the conversion happens in ``delta_stats``
   and is checked against a hand-computed log loss.
3. **Per-fold arithmetic.** The fold summary must agree with the pooled metric
   on a case where the answer is known by construction.
"""

import unittest

import numpy as np
import pandas as pd

from src.DataProviders import nba_data as nd
from src.Models import measurement as ms


def synthetic_games(n: int = 200) -> pd.DataFrame:
    dates = pd.Timestamp("2024-10-01") + pd.to_timedelta(np.arange(n) // 4, unit="D")
    return pd.DataFrame({
        "GAME_ID": [f"00224{i:05d}" for i in range(n)],
        "GAME_DATE": dates,
        "y": np.tile([1, 0], n // 2),
    })


class TestE14Contract(unittest.TestCase):
    def test_columns_are_unique_and_frozen(self):
        self.assertEqual(len(ms.E14_COLUMNS), 43)
        self.assertEqual(len(set(ms.E14_COLUMNS)), 43)
        self.assertEqual(ms.E14_COLUMNS[0], "home_games_played")
        self.assertEqual(ms.E14_COLUMNS[-1], "away_home_edge")

    def test_built_frame_matches_the_recorded_order(self):
        games = nd.build_games().iloc[:800].reset_index(drop=True)
        frame = ms.build_e14_frame(games)
        self.assertEqual(list(frame.columns), ms.E14_COLUMNS)
        self.assertEqual(len(frame), len(games))
        self.assertTrue(np.isfinite(frame.to_numpy(dtype=float)).any())

    def test_design_returns_the_frame_and_labels_aligned(self):
        games = nd.build_games().iloc[:600].reset_index(drop=True)
        g, X, y = ms.e14_design(games)
        self.assertEqual(len(X), len(g))
        self.assertEqual(list(X.columns), ms.E14_COLUMNS)
        self.assertTrue(np.array_equal(y, g["y"].to_numpy(dtype=int)))


class TestDeltaSign(unittest.TestCase):
    def test_candidate_better_is_positive(self):
        y = np.array([1, 1, 0, 0] * 25)
        good = np.array([0.9, 0.8, 0.2, 0.1] * 25)
        bad = np.array([0.6, 0.6, 0.4, 0.4] * 25)
        d = ms.delta_stats(y, good, bad, n_boot=200)
        self.assertGreater(d["delta_candidate_minus_reference"], 0)
        self.assertGreater(d["z"], 0)
        # Hand-computed: loss(bad) - loss(good), per-row mean.
        lp_good = -(y * np.log(good) + (1 - y) * np.log(1 - good)).mean()
        lp_bad = -(y * np.log(bad) + (1 - y) * np.log(1 - bad)).mean()
        self.assertAlmostEqual(d["delta_candidate_minus_reference"],
                               lp_bad - lp_good, places=12)

    def test_identical_models_give_zero(self):
        y = np.array([1, 0] * 50)
        p = np.array([0.7, 0.3] * 50)
        d = ms.delta_stats(y, p, p, n_boot=100)
        self.assertAlmostEqual(d["delta_candidate_minus_reference"], 0.0)
        self.assertEqual(d["abs_z"], 0.0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            ms.delta_stats(np.array([1, 0]), np.array([0.5]), np.array([0.5, 0.5]))


class TestPerFoldDelta(unittest.TestCase):
    def test_fold_arithmetic_agrees_with_hand_computation(self):
        games = synthetic_games(100)
        y = games["y"].to_numpy()
        good = np.where(y == 1, 0.8, 0.2)
        bad = np.full(100, 0.5)
        out = ms.per_fold_delta(y, good, bad, games, n_splits=2)
        self.assertEqual(out["n_folds"], 2)
        folds = [out[f"fold_{k}"] for k in range(2)]
        expected = np.mean([f["delta"] for f in folds])
        self.assertAlmostEqual(out["pooled_mean"], expected)
        for f in folds:
            self.assertGreater(f["delta"], 0)

    def test_oof_logistic_scores_only_the_test_window(self):
        rng = np.random.default_rng(3)
        games = synthetic_games(300)
        X = pd.DataFrame(rng.normal(size=(300, 3)))
        X.columns = ["a", "b", "c"]
        X.iloc[:150, 0] += 3.0 * (games["y"].to_numpy()[:150] - 0.5)
        p, mask = ms.oof_logistic(X, games["y"].to_numpy(), games, n_splits=3)
        self.assertEqual(mask.sum(), 150)
        self.assertTrue(np.isnan(p[:150]).all())
        self.assertTrue(np.isfinite(p[mask]).all())
        self.assertGreater(nd.evaluate(games["y"].to_numpy()[mask],
                                       p[mask])["auc"], 0.5)


if __name__ == "__main__":
    unittest.main()
