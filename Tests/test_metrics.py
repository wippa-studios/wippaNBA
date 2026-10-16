"""Closed-form tests for `src/Utils/metrics.py`.

Every expected value in this file is worked out by hand in the comment above
it, not copied from a library, so a wrong implementation cannot agree with a
wrong test. Where scikit-learn is mentioned it is as a cross-check that lives
outside the repo (there is a scratch script under ~/.cache/exp), never as the
source of a number here.

The regression test at the bottom is the one that matters. The bug that
motivated this module was inverse-frequency class weighting pushing the model's
mean P(home) to 0.5079 against a true base rate of 0.5694 -- 6.15 points low --
while the accuracy number barely moved. `TestMiscalibratedBySixPoints` builds
exactly that shape and asserts the calibration metrics scream about it.
"""

import math
import unittest

import numpy as np

from src.Utils import metrics


FLOAT_EPSILON = np.finfo(np.float64).eps


class TestLogLoss(unittest.TestCase):
    """mean(-log p) with the extremes clipped, never infinite."""

    def test_two_row_closed_form(self):
        # -(log(0.9) + log(0.8)) / 2 = -log(0.72) / 2 = 0.16425203
        loss = metrics.log_loss_score([1, 0], [0.9, 0.2])
        self.assertAlmostEqual(loss, 0.16425203, places=8)
        self.assertAlmostEqual(loss, -math.log(0.72) / 2, places=12)

    def test_coin_flip_costs_log_two(self):
        # A coin flip is log(2) on the binary target and nothing else.
        loss = metrics.log_loss_score([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5])
        self.assertAlmostEqual(loss, math.log(2), places=12)

    def test_constant_prediction_equals_binary_entropy(self):
        # base rate 0.6, constant 0.5:
        # -(0.6*log(0.5) + 0.4*log(0.5)) = -log(0.5) = log(2)
        y_true = [1] * 6 + [0] * 4
        loss = metrics.log_loss_score(y_true, [0.5] * 10)
        self.assertAlmostEqual(loss, math.log(2), places=12)

    def test_confident_and_correct_is_finite_and_near_zero(self):
        # p=1.0 is clipped to 1-eps, so -log(1-eps) ~ 2.2e-16 rather than inf.
        loss = metrics.log_loss_score([1, 0], [1.0, 0.0])
        self.assertTrue(math.isfinite(loss))
        self.assertLess(loss, 1e-12)

    def test_confident_and_wrong_is_large_but_finite(self):
        # Both terms become -log(eps) = 36.043653, and that is the whole answer.
        # An unguarded implementation returns inf here, which is the bug this
        # guards against: one saturated prediction would poison a whole report.
        loss = metrics.log_loss_score([1, 0], [0.0, 1.0])
        self.assertTrue(math.isfinite(loss))
        self.assertAlmostEqual(loss, -math.log(FLOAT_EPSILON), places=9)

    def test_multiclass_closed_form(self):
        # -(log(0.7) + log(0.8) + log(0.5)) / 3
        # = -(-0.35667494 - 0.22314355 - 0.69314718) / 3 = 0.42432189
        y_true = [0, 1, 2]
        y_prob = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]]
        loss = metrics.log_loss_score(y_true, y_prob)
        self.assertAlmostEqual(loss, 0.42432189, places=8)

    def test_multiclass_one_hot_is_perfect(self):
        y_prob = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        loss = metrics.log_loss_score([0, 1, 2], y_prob)
        self.assertTrue(math.isfinite(loss))
        self.assertLess(loss, 1e-12)

    def test_labels_decide_which_column_is_the_positive_class(self):
        # columns are ["away", "home"]; p is P(home), so the loss is
        # -(log(0.8) + log(1 - 0.3)) / 2 = -(log(0.8) + log(0.7)) / 2 = 0.28990925
        loss = metrics.log_loss_score(
            ["home", "away"], [0.8, 0.3], labels=["away", "home"]
        )
        self.assertAlmostEqual(loss, 0.28990925, places=8)

    def test_labels_reordering_actually_changes_the_score(self):
        # A 1-D vector is always P of the LAST class in labels, which is what
        # scikit-learn does. With labels=[0, 1] that is the positive class:
        # row 0 (y=0) is scored on 1 - 0.2 = 0.8 and row 1 (y=1) on 0.8, so
        # the loss is -(log(0.8) + log(0.8)) / 2 = -log(0.8) = 0.22314355.
        # With labels=[1, 0] the vector now means P(label 0), so both rows are
        # scored on 0.2 and the loss is -log(0.2) = 1.60943791. If the labels
        # argument were decorative the two numbers would be equal.
        as_is = metrics.log_loss_score([0, 1], [0.2, 0.8], labels=[0, 1])
        swapped = metrics.log_loss_score([0, 1], [0.2, 0.8], labels=[1, 0])
        self.assertAlmostEqual(as_is, 0.22314355, places=8)
        self.assertAlmostEqual(swapped, 1.60943791, places=8)
        self.assertNotAlmostEqual(as_is, swapped, places=3)

    def test_single_class_still_scores_against_the_binary_layout(self):
        # Every row is class 1, so the loss is just -mean(log p).
        loss = metrics.log_loss_score([1, 1, 1, 1], [0.6, 0.7, 0.8, 0.9])
        self.assertAlmostEqual(
            loss, -(math.log(0.6) + math.log(0.7) + math.log(0.8) + math.log(0.9)) / 4,
            places=12,
        )

    def test_empty_input_is_nan_not_a_crash(self):
        loss = metrics.log_loss_score([], [])
        self.assertTrue(math.isnan(loss))


class TestBrierScore(unittest.TestCase):
    """Two shapes, two definitions, and the factor of two between them."""

    def test_binary_closed_form(self):
        # (0.1^2 + 0.1^2 + 0.2^2 + 0.3^2) / 4 = (0.01+0.01+0.04+0.09)/4 = 0.0375
        score = metrics.brier_score([0, 1, 1, 0], [0.1, 0.9, 0.8, 0.3])
        self.assertAlmostEqual(score, 0.0375, places=12)

    def test_perfect_predictions_score_zero(self):
        self.assertAlmostEqual(
            metrics.brier_score([0, 1, 1, 0], [0.0, 1.0, 1.0, 0.0]), 0.0, places=12
        )

    def test_multiclass_sums_over_classes_and_averages_over_rows(self):
        # row 0 -> (0.8-1)^2 + 0.1^2 + 0.1^2 = 0.04 + 0.01 + 0.01 = 0.06
        # row 1 -> 0.2^2 + (0.7-1)^2 + 0.1^2 = 0.04 + 0.09 + 0.01 = 0.14
        # row 2 -> 0.2^2 + 0.2^2 + (0.6-1)^2 = 0.04 + 0.04 + 0.16 = 0.24
        # mean = (0.06 + 0.14 + 0.24) / 3 = 0.44 / 3 = 0.14666667
        y_prob = [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1], [0.2, 0.2, 0.6]]
        score = metrics.brier_score([0, 1, 2], y_prob)
        self.assertAlmostEqual(score, 0.44 / 3, places=12)

    def test_two_column_matrix_is_twice_the_binary_form(self):
        # Documented convention: 2-D input always means the multi-class form,
        # even at two columns, so it is exactly 2x the 1-D answer here.
        y_true = [0, 1, 1, 0]
        y_prob = [[0.9, 0.1], [0.1, 0.9], [0.2, 0.8], [0.7, 0.3]]
        binary = metrics.brier_score(y_true, [0.1, 0.9, 0.8, 0.3])
        matrix = metrics.brier_score(y_true, y_prob)
        self.assertAlmostEqual(matrix, 2 * binary, places=12)

    def test_single_class_brier_is_the_mean_squared_shortfall(self):
        # mean((p - 1)^2) = (0.16 + 0.09 + 0.04 + 0.01) / 4 = 0.30 / 4 = 0.075
        score = metrics.brier_score([1, 1, 1, 1], [0.6, 0.7, 0.8, 0.9])
        self.assertAlmostEqual(score, 0.075, places=12)

    def test_empty_input_is_nan(self):
        self.assertTrue(math.isnan(metrics.brier_score([], [])))


class TestCalibrationCurveData(unittest.TestCase):
    """The one binning implementation everything else is a view over."""

    def test_uniform_bin_edges_are_equal_width(self):
        edges, predicted, observed, counts = metrics.reliability_curve(
            [0, 1], [0.05, 0.95], n_bins=10
        )
        self.assertEqual(edges.shape, (11,))
        self.assertAlmostEqual(float(edges[0]), 0.0, places=12)
        self.assertAlmostEqual(float(edges[10]), 1.0, places=12)
        self.assertEqual(predicted.shape, (10,))
        self.assertEqual(counts.tolist(), [1, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    def test_empty_bins_are_nan_not_zero(self):
        # Eight of the ten bins hold nothing. An empty bin has no accuracy, and
        # 0.0 there would read as "confidently wrong" -- a different claim.
        edges, predicted, observed, counts = metrics.reliability_curve(
            [0, 1], [0.05, 0.95], n_bins=10
        )
        empty = counts == 0
        self.assertEqual(int(empty.sum()), 8)
        self.assertTrue(np.all(np.isnan(predicted[empty])))
        self.assertTrue(np.all(np.isnan(observed[empty])))
        self.assertAlmostEqual(float(predicted[0]), 0.05, places=12)
        self.assertAlmostEqual(float(observed[9]), 1.0, places=12)

    def test_hard_zero_and_one_land_in_the_first_and_last_bins(self):
        # searchsorted puts 0.0 in bin 0 and would send 1.0 past the end, so the
        # clip has to catch it. A saturated model must still report.
        _edges, _predicted, _observed, counts = metrics.reliability_curve(
            [0, 1], [0.0, 1.0], n_bins=10
        )
        self.assertEqual(int(counts[0]), 1)
        self.assertEqual(int(counts[9]), 1)

    def test_multiclass_matrix_is_pooled_one_versus_rest(self):
        # 3 rows x 3 classes = 9 confidence pairs. Uniform 10 bins put 0.1 and
        # 0.2 in the bottom bins and 0.7/0.8 near the top; no class is left out
        # and no row is counted twice.
        curve = metrics.calibration_curve_data(
            [0, 1, 2],
            [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]],
            n_bins=10,
        )
        self.assertEqual(curve["total"], 9)
        self.assertEqual(int(curve["counts"].sum()), 9)
        # Values present: 0.1 three times, 0.2 twice, then 0.3, 0.5, 0.7, 0.8.
        self.assertEqual(int(curve["counts"][1]), 3)  # the three 0.1s
        self.assertEqual(int(curve["counts"][2]), 2)  # the two 0.2s
        self.assertEqual(int(curve["counts"][8]), 1)  # the 0.8

    def test_quantile_bins_are_equal_mass(self):
        # 20 predictions, 5 bins -> exactly 4 predictions per bin.
        y_true = [1] * 4 + [0] * 16
        y_prob = [0.6] * 4 + [0.5] * 16
        curve = metrics.calibration_curve_data(
            y_true, y_prob, n_bins=5, strategy="quantile"
        )
        self.assertEqual(curve["counts"].tolist(), [4, 4, 4, 4, 4])

    def test_quantile_with_fewer_rows_than_bins_leaves_gaps(self):
        # 3 predictions over 10 equal-mass bins: bins 0, 3 and 6 get one each
        # and the other seven stay empty. Counts must still add up.
        curve = metrics.calibration_curve_data(
            [1, 0, 1], [0.2, 0.5, 0.8], n_bins=10, strategy="quantile"
        )
        self.assertEqual(int(curve["counts"].sum()), 3)
        self.assertEqual(int((curve["counts"] == 0).sum()), 7)
        self.assertEqual(int((curve["counts"] > 0).sum()), 3)

    def test_quantile_edges_are_non_decreasing(self):
        curve = metrics.calibration_curve_data(
            [0, 1, 1, 0], [0.1, 0.1, 0.9, 0.9], n_bins=5, strategy="quantile"
        )
        edges = curve["bin_edges"]
        self.assertEqual(edges.shape, (6,))
        self.assertTrue(np.all(np.diff(edges) >= 0))


class TestExpectedCalibrationError(unittest.TestCase):
    """Hand-computed bin gaps, no library involved."""

    def test_perfectly_calibrated_extremes_score_zero(self):
        # Two occupied bins and both exactly right: [0.0, 0.1) holds two
        # predictions of 0.0 on two losses, [0.9, 1.0] holds two predictions of
        # 1.0 on two wins. Every gap is 0, so the ECE is 0.
        score = metrics.expected_calibration_error(
            [1, 1, 0, 0], [1, 1, 0, 0], n_bins=10
        )
        self.assertAlmostEqual(score, 0.0, places=12)

    def test_constant_predictions_give_the_closed_form(self):
        # base rate 0.5, every prediction 0.4 -> |0.5 - 0.4| = 0.1 exactly,
        # whatever the binning, because every occupied bin carries the same
        # gap. This is the cheapest sanity check on a collapsed model.
        #
        # The labels are interleaved on purpose. Equal-mass binning splits by
        # RANK, so a label-sorted array would hand bin 0 the first two rows of
        # y_true and produce an observed frequency of 1.0 in the first bins.
        # With the composition held constant across the rows the closed form
        # holds for both strategies, which is what makes this a test of the
        # metric rather than of the order of the input.
        y_true = [1, 0] * 5
        self.assertAlmostEqual(
            metrics.expected_calibration_error(y_true, [0.4] * 10, n_bins=10),
            0.1, places=12,
        )
        self.assertAlmostEqual(
            metrics.expected_calibration_error(
                y_true, [0.4] * 10, n_bins=5, strategy="quantile"
            ),
            0.1, places=12,
        )
        self.assertAlmostEqual(
            metrics.maximum_calibration_error(y_true, [0.4] * 10, n_bins=10),
            0.1, places=12,
        )

    def test_quantile_hand_computation(self):
        # 20 rows over 5 equal-mass bins, 4 per bin. Bins 0-3 hold p=0.5 with
        # no outcomes (gap 0.5), bin 4 holds p=0.6 with all four (gap 0.4).
        # ECE = (16 * 0.5 + 4 * 0.4) / 20 = (8 + 1.6) / 20 = 0.48
        y_true = [1] * 4 + [0] * 16
        y_prob = [0.6] * 4 + [0.5] * 16
        score = metrics.expected_calibration_error(
            y_true, y_prob, n_bins=5, strategy="quantile"
        )
        self.assertAlmostEqual(score, 0.48, places=12)

    def test_both_strategies_agree_on_a_pure_split(self):
        # Every bin here holds only wins or only losses at the predicted rate,
        # so equal-width and equal-mass binning both score 0.1.
        y_true = [1] * 4 + [0] * 6
        y_prob = [0.9] * 4 + [0.1] * 6
        uniform = metrics.expected_calibration_error(y_true, y_prob, n_bins=5)
        quantile = metrics.expected_calibration_error(
            y_true, y_prob, n_bins=5, strategy="quantile"
        )
        self.assertAlmostEqual(uniform, 0.1, places=12)
        self.assertAlmostEqual(quantile, 0.1, places=12)

    def test_uniform_pooling_hides_misgrouping_that_quantile_exposes(self):
        # 10 rows in one equal-width bin [0.5, 0.6): mean predicted 0.535,
        # 5 of 10 observed -> gap 0.035. Split into equal-mass bins of one
        # each, the 0.52 predictions are all losses (gap 0.52) and the 0.55
        # predictions are all wins (gap 0.45):
        #   quantile ECE = (5 * 0.52 + 5 * 0.45) / 10 = (2.6 + 2.25) / 10 = 0.485
        y_true = [1] * 5 + [0] * 5
        y_prob = [0.55] * 5 + [0.52] * 5
        uniform = metrics.expected_calibration_error(y_true, y_prob, n_bins=10)
        quantile = metrics.expected_calibration_error(
            y_true, y_prob, n_bins=10, strategy="quantile"
        )
        self.assertAlmostEqual(uniform, 0.035, places=12)
        self.assertAlmostEqual(quantile, 0.485, places=12)
        self.assertGreater(quantile, uniform)

    def test_empty_bins_never_contribute(self):
        # Two occupied bins, each 0.05 out, and eight bins holding nothing:
        # (1 * 0.05 + 1 * 0.05) / 2 = 0.05. The empty bins are not zeros, so
        # they cannot dilute the answer either.
        y_true = [0, 1]
        y_prob = [0.05, 0.95]
        self.assertAlmostEqual(
            metrics.expected_calibration_error(y_true, y_prob, n_bins=10),
            0.05, places=12,
        )
        self.assertAlmostEqual(
            metrics.expected_calibration_error(y_true, y_prob, n_bins=10),
            metrics.expected_calibration_error(
                y_true, y_prob, n_bins=100
            ),
            places=12,
        )

    def test_empty_input_is_nan(self):
        self.assertTrue(math.isnan(metrics.expected_calibration_error([], [])))

    def test_agrees_with_the_curve_it_is_built_from(self):
        rng = np.random.default_rng(7)
        y_true = (rng.random(200) < 0.55).astype(int).tolist()
        y_prob = np.clip(rng.normal(0.55, 0.12, 200), 0.0, 1.0).tolist()
        curve = metrics.calibration_curve_data(y_true, y_prob, n_bins=8)
        filled = curve["counts"] > 0
        gaps = np.abs(
            curve["observed_frequency"][filled] - curve["mean_predicted"][filled]
        )
        expected = float(
            np.sum(curve["counts"][filled] * gaps) / curve["total"]
        )
        self.assertAlmostEqual(
            metrics.expected_calibration_error(y_true, y_prob, n_bins=8),
            expected, places=12,
        )


class TestMaximumCalibrationError(unittest.TestCase):
    """The worst bin, and the guarantee it dominates the average."""

    def test_worst_single_gap_is_reported(self):
        # Bin [0.5, 0.6) holds five predictions of 0.55 with zero wins and five
        # of 0.52 with five wins. Uniform: both in one bin, mean 0.535,
        # observed 0.5 -> gap 0.035.
        y_true = [1] * 5 + [0] * 5
        y_prob = [0.55] * 5 + [0.52] * 5
        self.assertAlmostEqual(
            metrics.maximum_calibration_error(y_true, y_prob, n_bins=10),
            0.035, places=12,
        )
        # Quantile: one prediction per bin, so the worst bin is a lone
        # prediction of 0.52 that never happened -> gap 0.52.
        self.assertAlmostEqual(
            metrics.maximum_calibration_error(
                y_true, y_prob, n_bins=10, strategy="quantile"
            ),
            0.52, places=12,
        )

    def test_maximum_is_never_below_the_average(self):
        # The ECE is a count-weighted mean of exactly the gaps the maximum is
        # taken over, so this inequality is structural, not incidental.
        rng = np.random.default_rng(11)
        y_true = (rng.random(300) < 0.5).astype(int).tolist()
        y_prob = np.clip(rng.normal(0.5, 0.2, 300), 0.0, 1.0).tolist()
        for strategy in ("uniform", "quantile"):
            average = metrics.expected_calibration_error(
                y_true, y_prob, n_bins=10, strategy=strategy
            )
            worst = metrics.maximum_calibration_error(
                y_true, y_prob, n_bins=10, strategy=strategy
            )
            self.assertGreaterEqual(worst + 1e-12, average)
            self.assertLessEqual(worst, 1.0)

    def test_perfectly_calibrated_is_zero(self):
        self.assertAlmostEqual(
            metrics.maximum_calibration_error([1, 1, 0, 0], [1, 1, 0, 0]),
            0.0, places=12,
        )

    def test_empty_input_is_nan(self):
        self.assertTrue(math.isnan(metrics.maximum_calibration_error([], [])))


class TestReliabilityCurve(unittest.TestCase):
    """The tuple unpack, and its agreement with the dict."""

    def test_returns_the_four_arrays_in_order(self):
        edges, predicted, observed, counts = metrics.reliability_curve(
            [0, 1], [0.05, 0.95], n_bins=10
        )
        self.assertEqual(len(edges), 11)
        self.assertEqual(len(predicted), 10)
        self.assertEqual(len(observed), 10)
        self.assertEqual(len(counts), 10)
        self.assertEqual(int(counts.sum()), 2)

    def test_unpacks_the_curve_data_verbatim(self):
        rng = np.random.default_rng(3)
        y_true = (rng.random(150) < 0.6).astype(int).tolist()
        y_prob = np.clip(rng.normal(0.6, 0.1, 150), 0.0, 1.0).tolist()
        curve = metrics.calibration_curve_data(y_true, y_prob, n_bins=6)
        edges, predicted, observed, counts = metrics.reliability_curve(
            y_true, y_prob, n_bins=6
        )
        np.testing.assert_array_equal(edges, curve["bin_edges"])
        np.testing.assert_array_equal(predicted, curve["mean_predicted"])
        np.testing.assert_array_equal(observed, curve["observed_frequency"])
        np.testing.assert_array_equal(counts, curve["counts"])

    def test_empty_input_returns_zero_counts(self):
        edges, predicted, observed, counts = metrics.reliability_curve([], [], n_bins=5)
        self.assertEqual(edges.shape, (6,))
        self.assertEqual(counts.tolist(), [0, 0, 0, 0, 0])
        self.assertTrue(np.all(np.isnan(predicted)))
        self.assertTrue(np.all(np.isnan(observed)))


class TestProbabilitySummary(unittest.TestCase):
    """The report a human reads."""

    def test_returns_a_string_and_does_not_print(self):
        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()
        with redirect_stdout(buffer):
            report = metrics.probability_summary([1, 0], [0.8, 0.2], n_bins=5)
        self.assertIsInstance(report, str)
        self.assertEqual(buffer.getvalue(), "")
        self.assertIn("Calibration summary", report)

    def test_names_every_required_figure(self):
        report = metrics.probability_summary([1, 0, 1, 0], [0.9, 0.1, 0.9, 0.1])
        for fragment in (
            "base rate",
            "mean predicted",
            "calibration gap",
            "expected calibration err",
            "maximum calibration err",
            "log loss",
            "brier score",
            "reliability table",
        ):
            self.assertIn(fragment, report)

    def test_printed_figures_match_the_metric_functions(self):
        y_true = [1, 0, 1, 0, 1, 0]
        y_prob = [0.7, 0.2, 0.8, 0.1, 0.6, 0.3]
        report = metrics.probability_summary(y_true, y_prob, n_bins=5)
        self.assertIn(
            f"{metrics.expected_calibration_error(y_true, y_prob, n_bins=5):.4f}",
            report,
        )
        self.assertIn(
            f"{metrics.maximum_calibration_error(y_true, y_prob, n_bins=5):.4f}",
            report,
        )
        self.assertIn(f"{metrics.log_loss_score(y_true, y_prob):.4f}", report)
        self.assertIn(f"{metrics.brier_score(y_true, y_prob):.4f}", report)

    def test_counts_appear_on_the_table(self):
        report = metrics.probability_summary([1, 0], [0.05, 0.95], n_bins=10)
        self.assertIn("[0.000, 0.100)", report)
        self.assertIn("[0.900, 1.000)", report)
        self.assertIn("total", report)

    def test_under_confident_gap_is_called_out_with_its_direction(self):
        y_true = [1] * 6 + [0] * 4
        report = metrics.probability_summary(y_true, [0.4] * 10, n_bins=5)
        self.assertIn("under-confident", report)
        self.assertIn("20.00 points", report)

    def test_over_confident_gap_is_called_out_too(self):
        y_true = [1] * 6 + [0] * 4
        report = metrics.probability_summary(y_true, [0.8] * 10, n_bins=5)
        self.assertIn("over-confident", report)
        self.assertIn("20.00 points", report)

    def test_single_class_input_is_flagged_not_crashed(self):
        report = metrics.probability_summary([1, 1, 1, 1], [0.6, 0.7, 0.8, 0.9])
        self.assertIn("only one class is present", report)
        self.assertIn("log loss", report)

    def test_empty_input_is_flagged_not_crashed(self):
        report = metrics.probability_summary([], [], n_bins=5)
        self.assertIn("no rows to score", report)
        self.assertIn("nan", report)

    def test_multiclass_report_lists_every_class(self):
        y_true = [0, 1, 2, 0]
        y_prob = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.6, 0.3, 0.1]]
        report = metrics.probability_summary(y_true, y_prob, n_bins=5)
        self.assertIn("per-class base rate", report)
        self.assertIn("class 0", report)
        self.assertIn("class 2", report)
        self.assertIn("brier score", report)

    def test_uniform_binning_carries_a_sparseness_note(self):
        report = metrics.probability_summary([1, 0, 1, 0], [0.55, 0.52, 0.51, 0.53])
        self.assertIn("quantile", report)


class TestEdgeCases(unittest.TestCase):
    """The documented degenerate-versus-broken contract, one test per rule."""

    def test_nan_probability_raises(self):
        with self.assertRaises(ValueError):
            metrics.expected_calibration_error([1, 0], [float("nan"), 0.2])
        with self.assertRaises(ValueError):
            metrics.log_loss_score([1, 0], [0.5, float("nan")])
        with self.assertRaises(ValueError):
            metrics.brier_score([1, 0], [0.5, float("nan")])

    def test_infinite_probability_raises(self):
        with self.assertRaises(ValueError):
            metrics.expected_calibration_error([1, 0], [0.5, float("inf")])
        with self.assertRaises(ValueError):
            metrics.log_loss_score([1, 0], [float("-inf"), 0.2])

    def test_probability_outside_zero_to_one_raises(self):
        with self.assertRaises(ValueError):
            metrics.brier_score([1, 0], [0.5, 1.4])
        with self.assertRaises(ValueError):
            metrics.log_loss_score([1, 0], [0.5, -0.1])

    def test_unnormalised_matrix_raises_and_is_not_silently_fixed(self):
        # Renormalising would hide a column-order bug for seasons.
        with self.assertRaises(ValueError):
            metrics.brier_score([0, 1], [[0.7, 0.2], [0.1, 0.2]])

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            metrics.log_loss_score([1, 0, 1], [0.5, 0.5])
        with self.assertRaises(ValueError):
            metrics.brier_score([1, 0], [0.5])

    def test_three_class_target_against_1d_probabilities_is_named(self):
        with self.assertRaises(ValueError) as caught:
            metrics.log_loss_score([0, 1, 2], [0.3, 0.4, 0.5])
        self.assertIn("three-class", str(caught.exception))

    def test_label_with_no_column_raises(self):
        with self.assertRaises(ValueError) as caught:
            metrics.brier_score(["neither"], [[0.5, 0.5]])
        self.assertIn("neither", str(caught.exception))

    def test_column_count_mismatch_hints_at_labels(self):
        with self.assertRaises(ValueError) as caught:
            # A 3-column matrix where y_true only ever shows two classes (the
            # push never happened) -- recoverable, but only if the caller says
            # which order the columns are in.
            metrics.brier_score([0, 1], [[0.5, 0.3, 0.2], [0.2, 0.6, 0.2]])
        self.assertIn("labels", str(caught.exception))

    def test_two_dimensional_y_true_raises(self):
        with self.assertRaises(ValueError):
            metrics.brier_score([[1, 0], [0, 1]], [0.5, 0.5])

    def test_column_vector_y_true_is_accepted_as_one_dimensional(self):
        # predict_proba()[:, [1]] arrives as (n, 1); treating it the same as a
        # flat vector is deliberate, and matches scikit-learn.
        loss = metrics.log_loss_score([[1], [0]], [0.9, 0.2])
        self.assertAlmostEqual(loss, metrics.log_loss_score([1, 0], [0.9, 0.2]),
                               places=12)

    def test_bad_n_bins_raises(self):
        with self.assertRaises(ValueError):
            metrics.expected_calibration_error([1, 0], [0.5, 0.5], n_bins=0)
        with self.assertRaises(ValueError):
            metrics.expected_calibration_error([1, 0], [0.5, 0.5], n_bins=2.5)

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            metrics.expected_calibration_error([1, 0], [0.5, 0.5], strategy="isotonic")

    def test_single_class_predictions_are_still_scored(self):
        # Every outcome was a home win: log loss and Brier are well defined and
        # the layout stays [0, 1] regardless of what appeared in y_true.
        loss = metrics.log_loss_score([1, 1, 1, 1], [0.6, 0.7, 0.8, 0.9])
        self.assertAlmostEqual(loss, -np.log([0.6, 0.7, 0.8, 0.9]).mean(), places=12)
        self.assertAlmostEqual(metrics.brier_score([1, 1, 1, 1], [0.6, 0.7, 0.8, 0.9]),
                               0.075, places=12)


class TestMiscalibratedBySixPoints(unittest.TestCase):
    """The regression this module was written for.

    The real numbers: mean P(home) came out at 0.5079 against a true base rate
    of 0.5694 -- 6.15 points low -- because inverse-frequency class weighting
    rebalanced the training set while the sigmoid calibrator was fit on
    unweighted labels. Accuracy did not notice. These tests pin the fact that
    these metrics do, and they pin it with numbers worked out by hand.

    Two fixtures:

    ``real_defect_pair``     the literal figures from the defect: a flat 0.5079
                             prediction against a 0.5694 base rate. The ECE is
                             the gap itself, 0.0615, by the closed form for
                             constant predictions.
    ``shifted_fixture``      five buckets that are each EXACTLY calibrated --
                             a bucket predicted 0.70 that wins 70 times out of
                             100, and so on -- and then every probability
                             moved down by 0.0615. This isolates the level error
                             from the ranking, because no prediction crosses
                             0.5, so thresholded accuracy does not move at all
                             while the calibration error appears in full.
    """

    BASE_RATE = 0.5694
    MEAN_PREDICTED = 0.5079
    GAP = MEAN_PREDICTED - BASE_RATE  # -0.0615
    LEVEL_SHIFT = 0.0615
    # (predicted probability, wins in that 100-row bucket). Each bucket is
    # calibrated by construction: a bucket predicted at 0.70 wins 70 times.
    BUCKETS = ((0.90, 90), (0.80, 80), (0.70, 70), (0.30, 30), (0.20, 20))

    def real_defect_pair(self):
        """10,000 rows with a true 0.5694 base rate, predicted flat at 0.5079.

        The wins are spread evenly rather than listed first, for the reason
        given in ``calibrated_fixture``: every prediction here is identical, so
        a rank-based binner splits the whole array into consecutive slices and a
        label-sorted array would put 5,694 wins in the first slices.
        """
        return self._spread_out(5694, 10000), [self.MEAN_PREDICTED] * 10000

    def calibrated_fixture(self):
        """500 rows, five buckets, perfectly calibrated. Mean 0.58, base rate 0.58.

        Wins are SPREAD EVENLY through each bucket instead of being listed
        first. That matters for more than tidiness: equal-mass binning splits a
        block of identical predictions by input order, so a bucket written as
        [1, 1, ..., 0, 0] would hand its first half to one bin and its second
        half to another, and the half full of wins would read as a calibration
        error that is not in the model at all. With the wins spread evenly, each
        half of a bucket holds exactly half its wins, so every bin -- uniform or
        quantile -- reproduces the bucket's own win rate to the digit.
        """
        y_true, y_prob = [], []
        for probability, wins in self.BUCKETS:
            y_true.extend(self._spread_out(wins, 100))
            y_prob.extend([probability] * 100)
        return y_true, y_prob

    @staticmethod
    def _spread_out(wins, length):
        """`length` labels with exactly `wins` of them 1, evenly distributed.

        A 1 is placed wherever ``position * wins // length`` advances, so each
        contiguous half of the result holds `wins / 2` of them (for the even
        counts used here) and the block looks homogeneous to a rank-based
        binner.
        """
        return [
            1 if (position * wins) // length > ((position - 1) * wins) // length else 0
            for position in range(1, length + 1)
        ]

    def shifted_fixture(self):
        """The same 500 rows with every probability moved down by 6.15 points."""
        y_true, y_prob = self.calibrated_fixture()
        return y_true, [probability - self.LEVEL_SHIFT for probability in y_prob]

    # -- the literal defect numbers ----------------------------------------

    def test_the_defect_fixture_is_six_points_low(self):
        y_true, y_prob = self.real_defect_pair()
        self.assertAlmostEqual(np.mean(y_true), self.BASE_RATE, places=10)
        self.assertAlmostEqual(np.mean(y_prob), self.MEAN_PREDICTED, places=10)
        self.assertAlmostEqual(np.mean(y_prob) - np.mean(y_true), self.GAP, places=10)

    def test_constant_underconfidence_scores_exactly_the_gap(self):
        # Constant predictions put every row in one bin, so the ECE collapses to
        # |base rate - predicted| = |0.5694 - 0.5079| = 0.0615 for either
        # binning strategy. It is exact rather than approximate: the ECE is a
        # weighted mean of identical gaps, and a mean of identical numbers is
        # that number.
        y_true, y_prob = self.real_defect_pair()
        for strategy in ("uniform", "quantile"):
            for n_bins in (5, 10, 20):
                self.assertAlmostEqual(
                    metrics.expected_calibration_error(
                        y_true, y_prob, n_bins=n_bins, strategy=strategy
                    ),
                    self.LEVEL_SHIFT, places=10,
                )
                # The maximum is the WORST bin, so it can only be at least the
                # average, and it carries that bin's own sampling noise on top:
                # 10,000 rows over 20 bins is 500 rows a bin, and a bin that
                # lands 285 wins instead of 284.7 reads 0.5700, a gap of
                # 0.0621. Within a thousandth of the level error, never under it.
                worst = metrics.maximum_calibration_error(
                    y_true, y_prob, n_bins=n_bins, strategy=strategy
                )
                self.assertGreaterEqual(worst, self.LEVEL_SHIFT - 1e-12)
                self.assertLessEqual(worst, self.LEVEL_SHIFT + 0.001)

    def test_the_report_names_the_direction_and_the_size(self):
        y_true, y_prob = self.real_defect_pair()
        report = metrics.probability_summary(y_true, y_prob, n_bins=10)
        self.assertIn("under-confident by 6.15 points", report)
        self.assertIn(f"{self.BASE_RATE:.4f}", report)
        self.assertIn(f"{self.MEAN_PREDICTED:.4f}", report)

    # -- the same defect isolated from the ranking --------------------------

    def test_the_calibrated_control_really_is_calibrated(self):
        y_true, y_prob = self.calibrated_fixture()
        self.assertAlmostEqual(np.mean(y_true), 0.58, places=12)
        self.assertAlmostEqual(np.mean(y_prob), 0.58, places=12)
        for strategy in ("uniform", "quantile"):
            self.assertAlmostEqual(
                metrics.expected_calibration_error(
                    y_true, y_prob, n_bins=10, strategy=strategy
                ),
                0.0, places=12,
            )
            self.assertAlmostEqual(
                metrics.maximum_calibration_error(
                    y_true, y_prob, n_bins=10, strategy=strategy
                ),
                0.0, places=12,
            )

    def test_shifting_the_level_moves_calibration_and_not_accuracy(self):
        # This is the assertion the whole module exists for. Accuracy is
        # identical to four decimal places before and after, because no bucket
        # crosses the 0.5 threshold (0.70 -> 0.6385 stays above it, 0.30 ->
        # 0.2385 stays below it). The ECE, the maximum calibration error, the
        # log loss and the Brier score all move.
        y_true, before = self.calibrated_fixture()
        _same, after = self.shifted_fixture()

        accuracy = lambda probabilities: np.mean(  # noqa: E731
            [(p >= 0.5) == bool(t) for p, t in zip(probabilities, y_true)]
        )
        # 90 + 80 + 70 correct on the high buckets, 70 + 80 on the low ones.
        self.assertAlmostEqual(accuracy(before), 0.78, places=12)
        self.assertAlmostEqual(accuracy(after), 0.78, places=12)

        self.assertAlmostEqual(np.mean(after), 0.58 - self.LEVEL_SHIFT, places=12)
        for strategy in ("uniform", "quantile"):
            self.assertAlmostEqual(
                metrics.expected_calibration_error(
                    y_true, after, n_bins=10, strategy=strategy
                ),
                self.LEVEL_SHIFT, places=12,
            )
            self.assertAlmostEqual(
                metrics.maximum_calibration_error(
                    y_true, after, n_bins=10, strategy=strategy
                ),
                self.LEVEL_SHIFT, places=12,
            )
        self.assertGreater(metrics.log_loss_score(y_true, after),
                           metrics.log_loss_score(y_true, before))
        self.assertGreater(metrics.brier_score(y_true, after),
                           metrics.brier_score(y_true, before))

    def test_every_occupied_bin_shows_the_same_six_point_gap(self):
        # Each bucket was calibrated on its own, so a uniform shift moves every
        # one of them by exactly 6.15 points. That is the signature of a level
        # error rather than a shape error, and it is what the reliability table
        # exists to let a human see.
        y_true, y_prob = self.shifted_fixture()
        _edges, predicted, observed, counts = metrics.reliability_curve(
            y_true, y_prob, n_bins=10
        )
        filled = counts > 0
        self.assertEqual(int(filled.sum()), 5)
        gaps = np.abs(observed[filled] - predicted[filled])
        for gap in gaps:
            self.assertAlmostEqual(gap, self.LEVEL_SHIFT, places=12)

    def test_the_unshifted_brier_score_is_the_mean_bucket_variance(self):
        # Per bucket the mean of (p - y)^2 over its 100 rows is
        # p*(1 - p)^2 + (1 - p)*p^2 = p*(1 - p). So the overall score is the
        # mean of p(1-p) over the five buckets:
        # (0.09 + 0.16 + 0.21 + 0.21 + 0.16) / 5 = 0.83 / 5 = 0.166.
        y_true, y_prob = self.calibrated_fixture()
        self.assertAlmostEqual(metrics.brier_score(y_true, y_prob), 0.166, places=12)

    def test_the_unshifted_log_loss_is_the_mean_binary_entropy(self):
        # Each bucket contributes H(p) = -(p*log(p) + (1-p)*log(1-p)):
        # H(0.9) 0.32508297 + H(0.8) 0.50040242 + H(0.7) 0.61086430
        # + H(0.3) 0.61086430 + H(0.2) 0.50040242 = 2.54761641 / 5
        y_true, y_prob = self.calibrated_fixture()
        self.assertAlmostEqual(metrics.log_loss_score(y_true, y_prob),
                               0.50952328, places=8)

    def test_an_under_confident_model_hides_a_genuine_plus_ev_bet(self):
        # -110 is a decimal price of 1 + 100/110 = 1.909090..., so break-even is
        # 1 / 1.90909 = 0.523810. The truth is 0.5694 and the model says 0.5079:
        # 0.5079 * 1.90909 - 1 = -0.0304 (reported as -EV, so never taken) while
        # 0.5694 * 1.90909 - 1 = +0.0871 (a real 8.7% edge). Six points of
        # calibration error is the difference between betting and not betting.
        decimal_odds = 1 + 100 / 110
        break_even = 1 / decimal_odds
        self.assertAlmostEqual(break_even, 0.523810, places=6)
        self.assertGreater(self.BASE_RATE, break_even)
        self.assertLess(self.MEAN_PREDICTED, break_even)
        self.assertAlmostEqual(self.MEAN_PREDICTED * decimal_odds - 1,
                               -0.03037273, places=8)
        self.assertAlmostEqual(self.BASE_RATE * decimal_odds - 1,
                               0.08703636, places=8)
        # And the gap between the two is the size of the mistake in money:
        # 0.0615 * 1.90909 = 0.11740909 of expected return, on one bet.
        self.assertAlmostEqual(
            (self.BASE_RATE - self.MEAN_PREDICTED) * decimal_odds,
            0.11740909, places=8,
        )
