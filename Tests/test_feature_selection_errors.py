"""An empty feature list is a mistake, not an answer.

`prune_features` intersects a declared feature list against the columns the
shared loaders actually produced. Dropping *some* columns is normal — the spec
asks for features the data may not supply, and the honest answer is the
intersection.

Dropping *all* of them is different. It means the caller passed something the
function cannot interpret, almost always feature GROUP names where column names
belong:

    prune_features(X, ["rating", "schedule"])   -> []   # silently, before this

No group name happens to also be a column name, so every entry misses and the
function returned an empty list with no error. That empty list then flowed into
`FeaturePipeline.fit` on a zero-column frame, where pandas raised a `ValueError`
from inside `hash_pandas_object` — naming neither the group that was requested
nor the function that discarded it.

These tests pin both halves of the fix: the error at the point of the mistake,
and an actionable message at the point where the empty frame would otherwise
have exploded.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT, ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from Models import logreg_rolling as lr  # noqa: E402


def _frame(n=8, cols=("net_diff", "home_ewm_margin", "away_ewm_margin")):
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {c: rng.normal(size=n) for c in cols},
        columns=list(cols),
    )


class TestEmptyFromNonEmptyIsAnError(unittest.TestCase):
    """The specific defect: a group name where a column name belongs."""

    def test_group_names_are_rejected_with_an_explanation(self):
        X = _frame()
        with self.assertRaises(lr.FeatureSelectionError) as ctx:
            lr.prune_features(X, ["rating", "schedule"])
        msg = str(ctx.exception)
        # The message has to be actionable, not merely an exception.
        self.assertIn("rating", msg, "must name the feature that was requested")
        self.assertIn("resolve_features", msg,
                      "must point at the function that DOES take group names")
        self.assertIn("zero columns", msg,
                      "must say what the consequence was")

    def test_an_unrecognised_column_is_also_rejected_not_silently_dropped(self):
        X = _frame()
        with self.assertRaises(lr.FeatureSelectionError):
            lr.prune_features(X, ["definitely_not_a_column"])

    def test_a_typo_in_one_of_several_is_still_an_error_when_nothing_matches(self):
        """One bad name among many good ones prunes normally.

        The guard fires on the EMPTY case only. Pruning a single absent column out
        of a long list is the documented behaviour and must not start raising --
        that would break every caller relying on partial availability.
        """
        X = _frame()
        kept = lr.prune_features(X, ["net_diff", "not_a_column"])
        self.assertEqual(kept, ["net_diff"])

    def test_context_is_echoed_so_the_error_points_at_the_call_site(self):
        X = _frame()
        with self.assertRaises(lr.FeatureSelectionError) as ctx:
            lr.prune_features(X, ["rating"], context="walk_forward_eval()")
        self.assertIn("walk_forward_eval()", str(ctx.exception))

    def test_the_error_offers_the_columns_that_do_exist(self):
        X = _frame()
        with self.assertRaises(lr.FeatureSelectionError) as ctx:
            lr.prune_features(X, ["rating"])
        self.assertIn("net_diff", str(ctx.exception),
                      "must show what the frame actually offers")


class TestEmptyInputIsNotAnError(unittest.TestCase):
    """Asking for nothing is asking for nothing. That is the caller's business."""

    def test_empty_request_returns_empty_without_raising(self):
        X = _frame()
        self.assertEqual(lr.prune_features(X, []), [])

    def test_empty_request_on_an_empty_frame_is_still_fine(self):
        self.assertEqual(lr.prune_features(pd.DataFrame(), []), [])

    def test_pruning_everything_from_a_real_request_still_raises(self):
        """The distinction that matters: non-empty in, empty out is a mistake.

        An empty frame with a non-empty request is the bug. An empty request is
        not. Both are covered above; this pins that they cannot be confused.
        """
        with self.assertRaises(lr.FeatureSelectionError):
            lr.prune_features(pd.DataFrame(), ["net_diff"])


class TestZeroColumnFrameIsRefusedInOurOwnVoice(unittest.TestCase):
    """The second half of the fix: the downstream failure is actionable."""

    def test_fit_on_a_zero_column_frame_names_the_requested_columns(self):
        y = np.array([0, 1] * 4)
        pipe = lr.FeaturePipeline(columns=["net_diff", "home_ewm_margin"])
        empty = pd.DataFrame(index=range(8))
        with self.assertRaises(lr.FeatureSelectionError) as ctx:
            pipe.fit(empty, y)
        msg = str(ctx.exception)
        self.assertIn("net_diff", msg,
                      "must name the columns the pipeline wanted")
        self.assertNotIn("hash_pandas_object", msg,
                         "must not surface pandas' internal machinery")

    def test_fit_with_no_configured_columns_still_refuses_a_zero_column_frame(self):
        pipe = lr.FeaturePipeline()
        with self.assertRaises(lr.FeatureSelectionError):
            pipe.fit(pd.DataFrame(index=range(8)), np.array([0, 1] * 4))

    def test_the_error_is_a_valueerror_subclass(self):
        """Existing `except ValueError` around a fit must keep working."""
        self.assertTrue(issubclass(lr.FeatureSelectionError, ValueError))

    def test_a_normal_frame_still_fits(self):
        """The guard must not fire on ordinary input."""
        X = _frame()
        y = np.array([0, 1] * 4)
        pipe = lr.FeaturePipeline(columns=list(X.columns))
        pipe.fit(X, y)
        self.assertTrue(np.isfinite(pipe.medians_.to_numpy()).all())


class TestTheResolvedPathIsUnaffected(unittest.TestCase):
    """resolve_features -> prune_features is the correct sequence and still works."""

    def test_resolve_then_prune_against_a_frame_holding_those_columns(self):
        groups = lr.feature_groups(with_rebounds=False)
        names = list(groups)
        if not names:
            self.skipTest("no feature groups defined")
        resolved = lr.resolve_features(names)
        # Build a frame that actually contains the resolved columns.
        X = pd.DataFrame(np.random.default_rng(1).normal(size=(6, len(resolved))),
                         columns=resolved)
        kept = lr.prune_features(X, resolved)
        self.assertEqual(kept, resolved,
                         "the documented sequence must be a no-op, not an error")


if __name__ == "__main__":
    unittest.main()