"""The early-stopping segment and the calibration segment must be disjoint.

XGBoost early-stops on the LAST entry of ``evals``. If the calibration rows are
also passed as ``dval``, the boosting round is chosen to minimise loss on exactly
the rows the sigmoid is then fitted to, and the calibrator learns a slope from
in-sample-optimal scores rather than from a draw from the distribution it will be
applied to.

These tests pin the four-segment split and, more importantly, assert on the actual
row indices rather than on the shape of the return value, because a shape check
would pass even if the slices were handed out in the wrong order.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# boosting_training does `from training_common import ...`, and the directory's
# hyphen means it can never be a package - the trainers rely on this being on the
# path, so the test does too.
TRAIN_MODELS = ROOT / "src" / "Train-Models"
if str(TRAIN_MODELS) not in sys.path:
    sys.path.insert(0, str(TRAIN_MODELS))

_spec = importlib.util.spec_from_file_location(
    "boosting_training", ROOT / "src" / "Train-Models" / "boosting_training.py"
)
boosting_training = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(boosting_training)

split_train_es_calib = boosting_training.split_train_es_calib


def _frame(n, seed=0):
    """A frame where every row carries its own index, so slices are traceable."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {"row_id": np.arange(n), "f1": rng.normal(size=n), "f2": rng.normal(size=n)}
    )
    y = pd.Series(rng.integers(0, 2, size=n))
    return X, y


class TestFourSegmentsAreDisjoint(unittest.TestCase):
    def test_indices_partition_the_frame_in_order(self):
        n = 10_000
        X, y = _frame(n)
        X_tr, y_tr, X_es, y_es, X_ca, y_ca = split_train_es_calib(X, y)

        train_ids = set(X_tr.row_id)
        es_ids = set(X_es.row_id)
        calib_ids = set(X_ca.row_id)

        self.assertEqual(train_ids & es_ids, set(), "train and early-stop overlap")
        self.assertEqual(train_ids & calib_ids, set(), "train and calibration overlap")
        self.assertEqual(es_ids & calib_ids, set(), "early-stop and calibration overlap")

        self.assertEqual(train_ids | es_ids | calib_ids, set(range(n)))
        self.assertLess(max(train_ids), min(es_ids), "segments are not chronological")
        self.assertLess(max(es_ids), min(calib_ids), "calibration is not the last segment")

    def test_every_segment_carries_its_own_labels(self):
        X, y = _frame(4000, seed=1)
        X_tr, y_tr, X_es, y_es, X_ca, y_ca = split_train_es_calib(X, y)
        for name, Xi, yi in (("train", X_tr, y_tr), ("es", X_es, y_es), ("calib", X_ca, y_ca)):
            self.assertEqual(len(Xi), len(yi), f"{name}: X and y lengths differ")
            self.assertGreater(len(Xi), 0, f"{name} segment is empty")

    def test_default_sizes_reserve_a_non_trivial_early_stop_segment(self):
        X, y = _frame(20_000)
        _, _, X_es, _, X_ca, _ = split_train_es_calib(X, y)
        self.assertGreater(len(X_es), 500, "early-stop segment is implausibly small")
        self.assertGreater(len(X_ca), 1000, "calibration segment is implausibly small")
        self.assertLess(len(X_es), len(X_ca), "calibration should stay larger than early-stop")

    def test_segment_sizes_are_configurable(self):
        X, y = _frame(20_000)
        small = split_train_es_calib(X, y, es_size=0.01, calib_size=0.10)
        large = split_train_es_calib(X, y, es_size=0.20, calib_size=0.10)
        self.assertLess(len(small[2]), len(large[2]), "es_size did not take effect")
        self.assertLess(len(large[0]), len(small[0]), "a larger es_size should shrink train")


class TestBoundariesAreSnappedToDates(unittest.TestCase):
    def test_no_boundary_lands_inside_a_date_group(self):
        # Many rows share each date, which is exactly the case a raw row-count cut
        # gets wrong: it slices a same-date group in half.
        n = 12_000
        rng = np.random.default_rng(2)
        X, y = _frame(n, seed=2)
        dates = pd.Series(
            np.repeat(pd.date_range("2015-10-27", periods=600, freq="D"), 20)
        )
        X_tr, _, X_es, _, X_ca, _ = split_train_es_calib(X, y, dates=dates)

        for label, left, right in (
            ("train|early-stop", X_tr, X_es),
            ("early-stop|calibration", X_es, X_ca),
        ):
            last_in_left = left.row_id.iloc[-1]
            first_in_right = right.row_id.iloc[0]
            self.assertNotEqual(
                dates.iloc[first_in_right], dates.iloc[last_in_left],
                f"{label} boundary falls inside a same-date group",
            )


class TestMisconfigurationIsRejected(unittest.TestCase):
    def test_empty_input_raises(self):
        X, y = _frame(0)
        with self.assertRaises(ValueError):
            split_train_es_calib(X, y)

    def test_segments_that_leave_no_training_data_raise(self):
        X, y = _frame(5000)
        with self.assertRaises(ValueError):
            split_train_es_calib(X, y, es_size=0.6, calib_size=0.5)

    def test_out_of_range_fractions_raise(self):
        X, y = _frame(5000)
        for kwargs in ({"es_size": 0}, {"calib_size": 1.0}, {"es_size": 1.5}):
            with self.assertRaises(ValueError):
                split_train_es_calib(X, y, **kwargs)


class TestCalibrationIsNeverTheValidationSet(unittest.TestCase):
    def test_calibration_rows_are_absent_from_early_stopping(self):
        """The specific regression: X_calib used to be handed to train_model as X_val."""
        X, y = _frame(15_000, seed=3)
        _, _, X_es, _, X_ca, _ = split_train_es_calib(X, y)
        self.assertFalse(
            set(X_es.row_id) & set(X_ca.row_id),
            "calibration rows must never appear in the early-stopping segment",
        )

    def test_train_model_early_stops_on_the_last_eval_entry(self):
        """Guard the assumption this fix rests on.

        If xgboost's behaviour changed so that it early-stopped on some entry other
        than the last, this whole split would be solving a different problem. Assert
        the shape of the call site instead of trusting the comment.
        """
        src = (ROOT / "src" / "Train-Models" / "boosting_training.py").read_text()
        self.assertIn("evals=[(dtrain, \"train\"), (dval, \"val\")]", src)
        self.assertIn("early_stopping_rounds=60", src)


if __name__ == "__main__":
    unittest.main()