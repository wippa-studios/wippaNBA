"""The moneyline head is a sigmoid, not a two-row softmax.

``multi:softprob`` with ``num_class=2`` and ``binary:logistic`` span the same
function class -- softmax over two logits is the sigmoid of their difference --
but the softmax parameterisation fits one tree per class per round, so it spends
two trees on a binary question. These tests pin the switch and the contract the
rest of the pipeline depends on: ``predict_proba`` returns two columns whatever
the objective returns, and the two objectives produce the same loss quantity.

The last class re-runs a fixed configuration on the shipped dataset. It is not a
baseline in the "pin the number" sense; it asserts that the objective switch does
not move held-out log loss by more than a hundredth, which is the claim #15
makes. The full 15-trial search comparison is recorded in the issue.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TRAIN_MODELS = ROOT / "src" / "Train-Models"
if str(TRAIN_MODELS) not in sys.path:
    sys.path.insert(0, str(TRAIN_MODELS))

_spec = importlib.util.spec_from_file_location(
    "boosting_training", TRAIN_MODELS / "boosting_training.py"
)
boosting_training = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(boosting_training)

BoosterWrapper = boosting_training.BoosterWrapper
predict_probabilities = boosting_training.predict_probabilities
prepare_data = boosting_training.prepare_data
sample_params = boosting_training.sample_params
split_train_test = boosting_training.split_train_test
train_model = boosting_training.train_model


def _binary_toy(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    logits = 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.4 * X[:, 2]
    y = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))
    return X, y


def _three_class_toy(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    scores = np.column_stack((X[:, 0], X[:, 1], -X[:, 0] - X[:, 1] + 0.3 * X[:, 2]))
    y = scores.argmax(axis=1)
    return X, y


def _softmax_variant(params):
    """The pre-#15 parameter block, for a like-for-like comparison."""
    variant = dict(params)
    variant.pop("objective", None)
    variant.pop("num_class", None)
    variant.update(
        objective="multi:softprob",
        num_class=2,
        eval_metric=["mlogloss", "merror"],
    )
    return variant


class TestObjectiveSelection(unittest.TestCase):
    def test_two_classes_use_the_sigmoid(self):
        params, _ = sample_params(np.random.default_rng(7), seed=42, num_classes=2)
        self.assertEqual(params["objective"], "binary:logistic")
        self.assertNotIn("num_class", params)
        # Same order as the multiclass pair: xgboost early-stops on the LAST
        # metric, so this keeps "error" as the stopping signal.
        self.assertEqual(params["eval_metric"], ["logloss", "error"])

    def test_three_classes_keep_the_softmax(self):
        params, _ = sample_params(np.random.default_rng(7), seed=42, num_classes=3)
        self.assertEqual(params["objective"], "multi:softprob")
        self.assertEqual(params["num_class"], 3)
        self.assertEqual(params["eval_metric"], ["mlogloss", "merror"])

    def test_the_objective_is_the_only_thing_that_changes(self):
        """A search trial must mean the same thing for both trainers.

        If switching the head also perturbed depth/eta/etc, the two searches
        would no longer be comparable and the moneyline could not inherit the
        measured behaviour of the shared sampler.
        """
        binary, _ = sample_params(np.random.default_rng(11), seed=42, num_classes=2)
        softmax, _ = sample_params(np.random.default_rng(11), seed=42, num_classes=3)
        for key in set(binary) - {"objective", "num_class", "eval_metric"}:
            self.assertEqual(binary[key], softmax[key], f"{key} moved with the head")
        self.assertEqual(set(softmax) - set(binary), {"num_class"})


class TestProbabilityContract(unittest.TestCase):
    def test_binary_output_is_widened_to_two_columns(self):
        X, y = _binary_toy()
        params, _ = sample_params(np.random.default_rng(3), seed=42, num_classes=2)
        model = train_model(X[:2000], y[:2000], X[2000:], y[2000:], params, 60, 2)

        single = model.predict(xgb.DMatrix(X[2000:]))
        self.assertEqual(single.ndim, 1, "binary:logistic should return one column")
        probabilities = predict_probabilities(model, X[2000:], 2)

        self.assertEqual(probabilities.shape, (1000, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(probabilities[:, 1], single)
        np.testing.assert_allclose(probabilities[:, 0], 1.0 - single)

    def test_the_wrapper_exposes_the_same_two_columns(self):
        X, y = _binary_toy(seed=2)
        params, _ = sample_params(np.random.default_rng(4), seed=42, num_classes=2)
        model = train_model(X[:2000], y[:2000], X[2000:], y[2000:], params, 60, 2)

        wrapper = BoosterWrapper(model, 2)
        self.assertEqual(list(wrapper.classes_), [0, 1])
        probabilities = wrapper.predict_proba(X[2000:])
        self.assertEqual(probabilities.shape, (1000, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_multiclass_output_is_left_alone(self):
        X, y = _three_class_toy()
        params, _ = sample_params(np.random.default_rng(5), seed=42, num_classes=3)
        model = train_model(X[:2000], y[:2000], X[2000:], y[2000:], params, 60, 3)

        probabilities = predict_probabilities(model, X[2000:], 3)
        self.assertEqual(probabilities.shape, (1000, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)
        np.testing.assert_allclose(
            probabilities, model.predict(xgb.DMatrix(X[2000:])), atol=1e-6
        )


class TestBinaryFitsHalfTheTrees(unittest.TestCase):
    def test_one_tree_per_round_against_two(self):
        X, y = _binary_toy(n=1200, seed=4)
        params, _ = sample_params(np.random.default_rng(6), seed=42, num_classes=2)
        dtrain = xgb.DMatrix(X, label=y)
        rounds = 20

        binary = xgb.train(params, dtrain, num_boost_round=rounds)
        softmax = xgb.train(_softmax_variant(params), dtrain, num_boost_round=rounds)

        self.assertEqual(len(binary.get_dump()), rounds)
        self.assertEqual(len(softmax.get_dump()), rounds * 2)


class TestTheLossQuantityIsTheSame(unittest.TestCase):
    def test_binary_log_loss_equals_two_column_mlogloss(self):
        """``mlogloss`` on [[1-p, p]] is a long-hand way of writing ``logloss``."""
        p = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
        y = np.array([0, 0, 1, 1, 1])
        binary = log_loss(y, p, labels=[0, 1])
        softmax = log_loss(y, np.column_stack((1.0 - p, p)), labels=[0, 1])
        self.assertAlmostEqual(binary, softmax, places=12)


def _dataset_available():
    try:
        import training_common  # noqa: F401
    except Exception:
        return False
    return training_common.DATASET_DB.exists()


@unittest.skipUnless(_dataset_available(), "shipped dataset is absent")
class TestHeldOutMetricsDoNotMove(unittest.TestCase):
    """A fixed configuration, fitted both ways, on the shipped dataset.

    This is deliberately one configuration and one chronological split. It is
    not the 15-trial search -- that would add minutes to the suite -- but it
    fails if the sigmoid head silently produces a differently scaled model.
    """

    @classmethod
    def setUpClass(cls):
        import training_common as tc

        frame = tc.load_dataset()
        X, y, _features, data = prepare_data(
            frame, tc.MONEYLINE_TARGET, tc.drop_columns_for(tc.MONEYLINE_TARGET)
        )
        dates = data[tc.DATE_COLUMN]
        X_train, y_train, X_test, y_test = split_train_test(X, y, dates=dates)
        params, _ = sample_params(np.random.default_rng(42), seed=42, num_classes=2)

        cls.y_test = y_test
        cls.binary = predict_probabilities(
            train_model(X_train, y_train, X_test, y_test, params, 150, 2), X_test, 2
        )
        cls.softmax = predict_probabilities(
            train_model(
                X_train, y_train, X_test, y_test, _softmax_variant(params), 150, 2
            ),
            X_test,
            2,
        )

    def test_log_loss_agrees_to_within_a_hundredth(self):
        binary = log_loss(self.y_test, self.binary, labels=[0, 1])
        softmax = log_loss(self.y_test, self.softmax, labels=[0, 1])
        self.assertLess(binary, 0.69, "the sigmoid head is no better than chance")
        self.assertLess(
            abs(binary - softmax),
            0.01,
            f"held-out log loss moved: binary={binary:.4f} softmax={softmax:.4f}",
        )

    def test_accuracy_agrees_to_within_three_points(self):
        binary = accuracy_score(self.y_test, self.binary.argmax(axis=1))
        softmax = accuracy_score(self.y_test, self.softmax.argmax(axis=1))
        self.assertLess(abs(binary - softmax), 0.03)


if __name__ == "__main__":
    unittest.main()
