"""A push is a void, never a side.

The totals head has three classes: under (0), over (1) and push (2). Both
runners used to treat "not under" as over, so a push-class prediction was
printed as an OVER bet carrying P(push) as its confidence. These tests pin the
label contract the decision layer reads; the reasoning is in
docs/totals_target.md.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TRAIN_MODELS = ROOT / "src" / "Train-Models"
if str(TRAIN_MODELS) not in sys.path:
    sys.path.insert(0, str(TRAIN_MODELS))

from src.Predict.XGBoost_Runner import _format_game_line  # noqa: E402
from training_common import (  # noqa: E402
    OU_PICK_LABELS,
    TOTALS_NUM_CLASSES,
)


class TestTotalsLabels(unittest.TestCase):
    def test_one_label_per_class(self):
        self.assertEqual(len(OU_PICK_LABELS), TOTALS_NUM_CLASSES)

    def test_push_has_its_own_label(self):
        self.assertEqual(OU_PICK_LABELS[0], "UNDER")
        self.assertEqual(OU_PICK_LABELS[1], "OVER")
        self.assertEqual(OU_PICK_LABELS[2], "PUSH")
        self.assertEqual(len(set(OU_PICK_LABELS)), 3, "two classes share a label")

    def test_a_push_prediction_is_not_printed_as_a_side(self):
        line = _format_game_line(
            "BOS",
            "NYK",
            winner_is_home=True,
            winner_confidence=61.0,
            under_over=2,
            ou_value=221.5,
            ou_confidence=4.1,
        )
        self.assertIn("PUSH", line)
        self.assertNotIn("OVER", line)
        self.assertNotIn("UNDER", line)

    def test_the_sides_keep_their_own_labels(self):
        for class_index, label in ((0, "UNDER"), (1, "OVER")):
            line = _format_game_line(
                "BOS",
                "NYK",
                winner_is_home=False,
                winner_confidence=55.0,
                under_over=class_index,
                ou_value=221.5,
                ou_confidence=58.0,
            )
            self.assertIn(label, line)
            self.assertNotIn("PUSH", line)


if __name__ == "__main__":
    unittest.main()
