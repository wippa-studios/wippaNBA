"""The neural path after #16: train-fitted scaling, calibration, one model dir.

The defects this file pins, in the order issue #16 lists them:

* the network was the only model in the repo that did not use a
  ``StandardScaler`` -- it divided each ROW by its own L2 norm, injecting a
  per-game magnitude gain that is uncorrelated with the outcome;
* it had no calibration segment and no closing-line ROI report, so raw softmax
  output was priced directly;
* a fresh run wrote ``Models/Trained-Model-*.keras`` while the runner looked in
  ``Models/NN_Models/`` too, leaving two indistinguishable generations;
* ``best_*_temp.keras`` survived an aborted run.

Every test runs with no network. The two trainer tests fit a tiny network on a
synthetic frame (one epoch, one trial) because the point is the plumbing --
scaler, split, calibrator, temp cleanup -- rather than a model worth keeping.
"""

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import pytest

# These tests exercise the real TensorFlow path, so skip cleanly when the
# dependency is absent rather than failing collection. CI installs it.
pytest.importorskip("tensorflow")

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_MODELS_DIR = REPO_ROOT / "src" / "Train-Models"
for _path in (REPO_ROOT, TRAIN_MODELS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import neural_training  # noqa: E402
import training_common  # noqa: E402
from src.Predict import NN_Runner  # noqa: E402


def load_trainer(name):
    """Import a trainer by path, with its real TensorFlow dependency."""
    path = TRAIN_MODELS_DIR / name
    spec = importlib.util.spec_from_file_location(f"test_{name[:-3].lower()}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_frame(n_rows=80):
    """A frame with the columns the trainer and the ROI reporter touch.

    Dates are whole slates of four rows, so the split's date-boundary snap has
    something to snap to. The label alternates so every segment carries both
    classes and a sigmoid calibrator can be fitted.
    """
    dates = pd.date_range("2020-01-01", periods=n_rows // 4, freq="D").repeat(4)
    rng = np.random.default_rng(7)
    return pd.DataFrame({
        "Date": dates,
        "GameDate": dates + pd.Timedelta(days=1),
        "F1": rng.normal(size=n_rows),
        "F2": rng.normal(size=n_rows),
        "F3": rng.normal(size=n_rows),
        "Home-Team-Win": np.arange(n_rows) % 2,
        "TEAM_NAME": "HOME",
        "TEAM_NAME.1": "AWAY",
        "OU": rng.normal(220.0, 8.0, n_rows),
        "Score": rng.integers(180, 260, n_rows),
    })


class TestChronologicalSegments(unittest.TestCase):
    def test_four_segments_are_disjoint_and_in_date_order(self):
        frame = synthetic_frame(120)
        X = frame[["F1", "F2", "F3"]]
        y = frame["Home-Team-Win"].to_numpy()

        X_train, y_train, X_es, y_es, X_calib, y_calib, X_test, y_test = \
            neural_training.split_train_es_calib_test(
                X, y, val_size=0.1, calib_size=0.1, test_size=0.1,
                dates=frame["Date"],
            )

        sizes = [len(X_train), len(X_es), len(X_calib), len(X_test)]
        self.assertTrue(all(size > 0 for size in sizes), sizes)
        self.assertEqual(sum(sizes), len(X))

        dates = frame["Date"]
        train_end = len(X_train)
        es_end = train_end + len(X_es)
        calib_end = es_end + len(X_calib)
        self.assertLess(dates.iloc[train_end - 1], dates.iloc[train_end])
        self.assertLess(dates.iloc[es_end - 1], dates.iloc[es_end])
        self.assertLess(dates.iloc[calib_end - 1], dates.iloc[calib_end])
        self.assertEqual(
            pd.concat([X_train, X_es, X_calib, X_test]).reset_index(drop=True).equals(
                X.reset_index(drop=True)
            ),
            True,
        )

    def test_an_impossible_split_is_refused(self):
        X = pd.DataFrame({"a": range(10)})
        y = np.arange(10)
        with self.assertRaises(ValueError):
            neural_training.split_train_es_calib_test(X, y, 0.5, 0.4, 0.2)

    def test_the_legacy_three_way_split_is_still_consistent(self):
        X = pd.DataFrame({"a": range(20)})
        y = np.arange(20)
        X_train, X_val, X_test, y_train, y_val, y_test = \
            neural_training.split_time_series(X, y)
        self.assertEqual(len(X_train) + len(X_val) + len(X_test), 20)


class TestScalerReplacesRowNormalisation(unittest.TestCase):
    def test_to_model_matrix_returns_a_train_fitted_preprocessor(self):
        train = pd.DataFrame({"a": [0.0, 2.0, 4.0], "b": [1.0, 2.0, 3.0]})
        held_out = pd.DataFrame({"a": [100.0], "b": [100.0]})

        preprocessor, (train_matrix, held_out_matrix) = \
            neural_training.to_model_matrix(train, held_out)

        self.assertIsInstance(preprocessor, training_common.FeatureScaler)
        self.assertEqual(preprocessor.feature_names, ["a", "b"])
        self.assertAlmostEqual(float(train_matrix[:, 0].mean()), 0.0, places=12)
        self.assertAlmostEqual(float(train_matrix[:, 0].std(ddof=0)), 1.0, places=12)
        expected = (100.0 - preprocessor.scaler.mean_[0]) / preprocessor.scaler.scale_[0]
        self.assertAlmostEqual(float(held_out_matrix[0, 0]), float(expected), places=12)

    def test_the_preprocessor_round_trips_through_joblib(self):
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        preprocessor, _ = training_common.fit_feature_scaler(train)
        frame = pd.DataFrame({"b": [10.0], "a": [10.0]})

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "preprocessor.pkl"
            joblib.dump(preprocessor, path)
            loaded = joblib.load(path)

        np.testing.assert_allclose(loaded.transform(frame), preprocessor.transform(frame))


class TestKerasCalibrationAdapter(unittest.TestCase):
    def test_predict_proba_matches_the_network(self):
        import tensorflow as tf

        rng = np.random.default_rng(0)
        X = rng.normal(size=(32, 3)).astype("float32")
        y = np.arange(32) % 2
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(4, activation="relu", input_shape=(3,)),
            tf.keras.layers.Dense(2, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
        model.fit(X, y, epochs=1, verbose=0)

        wrapper = neural_training.KerasProbabilityWrapper(model, 2)
        np.testing.assert_allclose(
            wrapper.predict_proba(X[:4]), model.predict(X[:4], verbose=0), rtol=1e-6
        )
        self.assertEqual(list(wrapper.classes_), [0, 1])
        self.assertIs(wrapper.fit(X, y), wrapper)


class TestRunnerUsesTheSavedArtifacts(unittest.TestCase):
    def _touch_model(self, directory, name, with_preprocessor=False):
        path = Path(directory) / name
        path.write_bytes(b"not a real keras model")
        if with_preprocessor:
            NN_Runner._sidecar_path(path, "preprocessor.pkl").write_bytes(b"stub")
        return path

    def test_a_model_without_a_preprocessor_is_not_selected_over_one_with(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._touch_model(tmp, "Trained-Model-ML-70.0.keras", with_preprocessor=False)
            usable = self._touch_model(tmp, "Trained-Model-ML-50.0.keras", with_preprocessor=True)
            with mock.patch.object(NN_Runner, "MODEL_DIR", Path(tmp)), \
                    mock.patch.object(NN_Runner, "LEGACY_MODEL_DIR", Path(tmp) / "legacy"):
                selected = NN_Runner._select_best_model("Trained-Model-ML-", NN_Runner.ML_PATTERN)

        self.assertEqual(selected, usable)

    def test_legacy_only_artifacts_produce_an_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._touch_model(tmp, "Trained-Model-ML-64.6.keras", with_preprocessor=False)
            with mock.patch.object(NN_Runner, "MODEL_DIR", Path(tmp)), \
                    mock.patch.object(NN_Runner, "LEGACY_MODEL_DIR", Path(tmp) / "legacy"):
                with self.assertRaises(FileNotFoundError) as caught:
                    NN_Runner._select_best_model("Trained-Model-ML-", NN_Runner.ML_PATTERN)

        self.assertIn("predates feature scaling", str(caught.exception))

    def test_calibrated_probabilities_are_used_when_present(self):
        class Model:
            def predict(self, matrix, verbose=0):
                return np.tile([0.25, 0.75], (len(matrix), 1))

        class Calibrator:
            def predict_proba(self, matrix):
                return np.tile([0.40, 0.60], (len(matrix), 1))

        matrix = np.zeros((3, 2))
        np.testing.assert_allclose(
            NN_Runner._predict_matrix(Model(), None, matrix),
            np.tile([0.25, 0.75], (3, 1)),
        )
        np.testing.assert_allclose(
            NN_Runner._predict_matrix(Model(), Calibrator(), matrix),
            np.tile([0.40, 0.60], (3, 1)),
        )


class TestTrainerEndToEnd(unittest.TestCase):
    """Run the moneyline entry point on a synthetic frame, one trial, one epoch."""

    TINY_PARAMS = {
        "units": [4],
        "dropout": 0.0,
        "batch_norm": False,
        "l2": 0.0,
        "lr": 1e-3,
        "batch_size": 8,
        "epochs": 1,
    }

    def _run(self, trainer, tmp, argv):
        output = io.StringIO()
        with mock.patch.object(sys, "argv", [argv[0]] + argv[1:]), \
                mock.patch.object(trainer, "MODEL_DIR", Path(tmp)), \
                mock.patch.object(trainer, "load_dataset", return_value=synthetic_frame()), \
                mock.patch.object(trainer, "sample_params", return_value=dict(self.TINY_PARAMS)), \
                contextlib.redirect_stdout(output):
            trainer.main()
        return output.getvalue()

    def test_a_run_saves_the_scaler_calibrator_and_no_temp_file(self):
        trainer = load_trainer("NN_Model_ML.py")
        with tempfile.TemporaryDirectory() as tmp:
            output = self._run(trainer, tmp, ["NN_Model_ML.py", "--trials", "1"])

            self.assertFalse((Path(tmp) / "best_ml_temp.keras").exists())
            models = list(Path(tmp).glob("Trained-Model-ML-*.keras"))
            self.assertEqual(len(models), 1, output)
            preprocessor_path = NN_Runner._sidecar_path(models[0], "preprocessor.pkl")
            calibration_path = NN_Runner._sidecar_path(models[0], "calibration.pkl")
            self.assertTrue(preprocessor_path.exists(), output)
            self.assertTrue(calibration_path.exists(), output)

            preprocessor = joblib.load(preprocessor_path)
            transformed = preprocessor.transform(synthetic_frame()[["F1", "F2", "F3"]])
            self.assertEqual(transformed.shape, (80, 3))
            self.assertTrue(np.isfinite(transformed).all())
            # Applied to the whole frame the mean is not zero -- the scaler was
            # fit on the training segment only, which is the point. What it must
            # not do is put every row on the unit sphere (the old L2 transform).
            self.assertFalse(np.allclose(np.linalg.norm(transformed, axis=1), 1.0))

            # The calibrator is what the runner will load, so it must survive
            # joblib's round trip with its embedded network intact.
            calibrator = joblib.load(calibration_path)
            probabilities = calibrator.predict_proba(np.zeros((2, transformed.shape[1])))
            self.assertEqual(probabilities.shape, (2, 2))

            self.assertIn("Saved preprocessor:", output)
            self.assertIn("Calibration: sigmoid", output)
            self.assertIn("ROI", output)
            self.assertIn("Configurations tried: 1", output)

    def test_a_failure_after_the_temp_write_still_removes_the_temp_file(self):
        trainer = load_trainer("NN_Model_ML.py")
        with tempfile.TemporaryDirectory() as tmp:
            temp_path = Path(tmp) / "best_ml_temp.keras"

            def explode(*args, **kwargs):
                # The temp file exists at the moment the failure happens, which
                # is the state an aborted run used to leave behind.
                temp_path.write_bytes(b"partial")
                raise RuntimeError("simulated crash")

            with mock.patch.object(sys, "argv", ["NN_Model_ML.py", "--trials", "1"]), \
                    mock.patch.object(trainer, "MODEL_DIR", Path(tmp)), \
                    mock.patch.object(trainer, "load_dataset", return_value=synthetic_frame()), \
                    mock.patch.object(trainer, "sample_params", return_value=dict(self.TINY_PARAMS)), \
                    mock.patch.object(trainer, "build_model", side_effect=explode), \
                    contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(RuntimeError):
                    trainer.main()

            self.assertFalse(temp_path.exists(), "an aborted run left its scratch model behind")


if __name__ == "__main__":
    unittest.main()
