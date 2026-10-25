"""Model filenames carry the target and accuracy, never the tuned configuration.

Every trainer used to append the entire best-of-N search result to the artifact
name: depth, eta, subsample, column fractions, min-child-weight, gamma, delta
step, max-bin, lambda, alpha and the boosting round. That published the recipe by
default, and a future configuration with a real edge would have been published
the same way without anyone deciding to (#18).

The name now keeps only what discovery needs -- the ML/UO kind and the held-out
accuracy the runners parse -- and the configuration goes into a gitignored
``<stem>_manifest.json``. These tests pin both halves of that contract and the
runner globs that must keep matching.
"""
import fnmatch
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAIN_MODELS = ROOT / "src" / "Train-Models"
for _path in (ROOT, TRAIN_MODELS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import training_common  # noqa: E402
from src.Predict import XGBoost_Runner  # noqa: E402


class TestXGBoostNames(unittest.TestCase):
    def test_only_the_kind_and_accuracy_remain(self):
        for kind in ("ML", "UO"):
            name = training_common.xgboost_model_filename(kind, 0.6404)
            self.assertEqual(Path(name).stem.split("_"), ["XGBoost", "64.0%", kind])

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            training_common.xgboost_model_filename("OU", 0.5)

    def test_the_runner_can_still_parse_the_accuracy(self):
        name = training_common.xgboost_model_filename("ML", 0.6404)
        match = XGBoost_Runner.ACCURACY_PATTERN.search(name)
        self.assertIsNotNone(match, f"{name} no longer matches the runner pattern")
        self.assertEqual(match.group(1), "64.0")
        self.assertIn("ML", name)


class TestNeuralNames(unittest.TestCase):
    def test_only_the_kind_and_accuracy_remain(self):
        for kind in ("ML", "OU"):
            name = training_common.neural_model_filename(kind, 0.6462)
            self.assertEqual(
                Path(name).stem.split("-"), ["Trained", "Model", kind, "64.6"]
            )

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError):
            training_common.neural_model_filename("UO", 0.5)

    def test_the_runner_can_still_parse_the_accuracy(self):
        from src.Predict import NN_Runner

        name = training_common.neural_model_filename("ML", 0.6462)
        match = NN_Runner.ML_PATTERN.search(name)
        self.assertIsNotNone(match, f"{name} no longer matches the runner pattern")
        self.assertEqual(match.group(1), "64.6")


class TestTheTrainersUseTheHelpers(unittest.TestCase):
    """A source check, because these four mains are too heavy to run here."""

    def test_no_trainer_builds_a_name_from_parameters(self):
        expected = {
            "XGBoost_Model_ML.py": "xgboost_model_filename",
            "XGBoost_Model_UO.py": "xgboost_model_filename",
            "NN_Model_ML.py": "neural_model_filename",
            "NN_Model_UO.py": "neural_model_filename",
        }
        for filename, helper in expected.items():
            source = (TRAIN_MODELS / filename).read_text()
            self.assertIn(f"model_name = {helper}(", source)
            self.assertNotIn("format_param", source)


class TestManifest(unittest.TestCase):
    def test_it_is_written_beside_the_artifact_as_json(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "XGBoost_64.0%_ML.json"
            model_path.write_text("{}")
            payload = {
                "params": {"max_depth": 3, "eta": 0.07},
                "target": "Home-Team-Win",
                "n_features": 106,
            }
            manifest_path = training_common.write_run_manifest(model_path, payload)

            self.assertEqual(
                manifest_path.name, "XGBoost_64.0%_ML_manifest.json"
            )
            self.assertEqual(manifest_path.parent, model_path.parent)
            self.assertEqual(json.loads(manifest_path.read_text()), payload)

    def test_manifests_are_not_committed(self):
        ignored = (ROOT / ".gitignore").read_text()
        self.assertIn("*_manifest.json", ignored)
        self.assertTrue(fnmatch.fnmatch("XGBoost_64.0%_ML_manifest.json", "*_manifest.json"))
        self.assertFalse(
            fnmatch.fnmatch("Data/manifest.json", "*_manifest.json"),
            "the dataset manifest must stay trackable",
        )

    def test_the_committed_artifacts_have_no_parameter_tokens(self):
        pattern = (
            r"_(md|eta|sub|col|cbl|cbn|mcw|g|mds|mb|l|a|nb|d|u|dr|bn|l2|lr|bs)\d"
        )
        import re

        for path in (ROOT / "Models").rglob("*"):
            if path.is_file() and path.suffix in {".json", ".keras"}:
                self.assertIsNone(
                    re.search(pattern, path.name),
                    f"committed artifact publishes its configuration: {path.name}",
                )


if __name__ == "__main__":
    unittest.main()
