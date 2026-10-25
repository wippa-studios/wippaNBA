"""One drop list, computed, not four copies of it.

``training_common.drop_columns_for`` is the trainers' definition. ``src/Utils/
features.py`` used to carry four "verbatim" transcriptions, and
``scripts/evaluate.py`` a fifth; one of them had already drifted out of order.
A positional model cannot see a wrong order, so the copies are compared here by
identity of value, order included.

The last class also pins the property the old NN comment asserted but did not
test: dropping is membership-based, so permuting the deny-list cannot change the
surviving column order.
"""
import ast
import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_MODELS = ROOT / "src" / "Train-Models"
for _path in (ROOT, TRAIN_MODELS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import training_common  # noqa: E402
from src.Utils import features  # noqa: E402


class TestEveryMirrorIsTheCanonicalList(unittest.TestCase):
    def test_moneyline_lists(self):
        canonical = training_common.drop_columns_for(
            training_common.MONEYLINE_TARGET
        )
        self.assertEqual(features.ML_DROP_COLUMNS, canonical)
        self.assertEqual(features.NN_ML_DROP_COLUMNS, canonical)

    def test_totals_lists(self):
        canonical = training_common.drop_columns_for(training_common.TOTALS_TARGET)
        self.assertEqual(features.UO_LABEL_DROP_COLUMNS, canonical)
        self.assertEqual(features.NN_UO_DROP_COLUMNS, canonical)

    def test_the_closing_total_is_the_only_difference_between_the_targets(self):
        moneyline = training_common.drop_columns_for(
            training_common.MONEYLINE_TARGET
        )
        totals = training_common.drop_columns_for(training_common.TOTALS_TARGET)
        self.assertIn("OU", moneyline)
        self.assertNotIn("OU", totals)


class TestEvaluateUsesTheSameFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location(
            "evaluate_under_test", ROOT / "scripts" / "evaluate.py"
        )
        cls.evaluate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.evaluate)

    def test_its_drop_list_is_the_trainers(self):
        self.assertEqual(
            self.evaluate.DROP_COLUMNS,
            training_common.drop_columns_for(
                training_common.MONEYLINE_TARGET, include_team_ids=True
            ),
        )


class TestDenyListOrderCannotMatter(unittest.TestCase):
    def test_permuting_the_list_leaves_the_feature_order_alone(self):
        columns = [
            "TEAM_NAME",
            "index",
            "OU",
            "home_x",
            "Home-Team-Win",
            "OU-Cover",
            "Date",
            "away_x",
        ]
        frame = pd.DataFrame(columns=columns)
        canonical = features.ML_DROP_COLUMNS

        self.assertEqual(
            features.expected_feature_order(frame.columns, canonical),
            features.expected_feature_order(frame.columns, list(reversed(canonical))),
            "drop() is membership-based; a deny-list reorder changed the output",
        )

    def test_the_keras_and_booster_lists_are_identical(self):
        """The NN path does not get a differently ordered deny-list any more."""
        self.assertEqual(
            features.NN_ML_DROP_COLUMNS,
            features.ML_DROP_COLUMNS,
        )
        self.assertEqual(
            features.NN_UO_DROP_COLUMNS,
            features.UO_LABEL_DROP_COLUMNS,
        )


class TestNoLiteralDropListSurvives(unittest.TestCase):
    """The scan catches a re-introduced copy, not just the ones known today."""

    def test_src_and_scripts_contain_no_list_literal_drop_columns(self):
        offenders = []
        paths = list((ROOT / "src").rglob("*.py")) + list(
            (ROOT / "scripts").rglob("*.py")
        )
        for path in paths:
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                for target in targets:
                    if (
                        isinstance(target, ast.Name)
                        and target.id.endswith("DROP_COLUMNS")
                        and isinstance(node.value, (ast.List, ast.Tuple))
                    ):
                        offenders.append(
                            f"{path.relative_to(ROOT)}:{node.lineno} {target.id}"
                        )
        self.assertEqual(
            offenders,
            [],
            "a drop list was re-declared instead of computed: " + ", ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
