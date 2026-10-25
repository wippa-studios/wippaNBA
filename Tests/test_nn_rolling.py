"""Contract tests for the M01 neural arm.

These are deliberately small and synthetic.  They pin the things that would
make the reported number meaningless if they broke: the OOF function scores
exactly the walk-forward test rows, it is deterministic for a fixed seed, and
the seed average really is the mean of the per-seed predictions.  The real
measurement lives in ``results/nn_rolling.json``; it is not run here.
"""

import unittest

import numpy as np
import pandas as pd

from src.Models import nn_rolling as nn

ARCH = {
    "hidden_layer_sizes": (8,),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "batch_size": 16,
    "learning_rate_init": 1e-3,
    "max_iter": 30,
    "tol": 1e-6,
    "n_iter_no_change": 5,
    "early_stopping": False,
    "shuffle": True,
}


def _data(n=120, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)),
                     columns=[f"f{i}" for i in range(p)])
    y = (X["f0"] + 0.5 * X["f1"] + rng.normal(size=n) > 0).astype(int)
    games = pd.DataFrame({
        "GAME_ID": np.arange(n),
        "GAME_DATE": pd.date_range("2020-01-01", periods=n),
    })
    return X, y.to_numpy(), games


class TestOofMlp(unittest.TestCase):
    def test_scores_exactly_the_test_rows_and_is_deterministic(self):
        X, y, games = _data()
        p1, mask1, per1 = nn.oof_mlp(X, y, games, seeds=(7,), arch=ARCH,
                                     n_splits=2)
        p2, mask2, per2 = nn.oof_mlp(X, y, games, seeds=(7,), arch=ARCH,
                                     n_splits=2)
        # first 50% train-only, last 50% split into two contiguous test blocks
        self.assertEqual(int(mask1.sum()), 60)
        np.testing.assert_array_equal(mask1, mask2)
        self.assertTrue(np.isfinite(p1[mask1]).all())
        self.assertTrue(np.isnan(p1[~mask1]).all())
        np.testing.assert_array_equal(per1[7], per2[7])
        np.testing.assert_array_equal(p1, p2)

    def test_seed_average_is_the_mean_of_the_seed_predictions(self):
        X, y, games = _data()
        p, mask, per = nn.oof_mlp(X, y, games, seeds=(1, 2), arch=ARCH,
                                  n_splits=2)
        expected = np.vstack([per[1][mask], per[2][mask]]).mean(axis=0)
        np.testing.assert_allclose(p[mask], expected, rtol=0, atol=0)

    def test_refuses_misaligned_rows(self):
        X, y, games = _data()
        with self.assertRaises(ValueError):
            nn.oof_mlp(X.iloc[:-5], y, games, seeds=(1,), arch=ARCH)


if __name__ == "__main__":
    unittest.main()
