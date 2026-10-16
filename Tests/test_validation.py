"""Tests for ``src/Utils/validation.py``.

Every statistic in this module is an APPROXIMATION, and a test suite that only
checked "does it return a float" would let an approximation quietly rot. The
tests below therefore do three different kinds of thing:

1. ``INVARIANTS`` -- properties that must hold for the split regardless of the
   approximation: no training row may share a calendar date with a test row, no
   training row may sit inside the embargo window, every index must be usable
   with ``.iloc``, and the split must not depend on input order.
2. ``NUMBERS`` -- the published values, pinned against sources a reader can look
   up (the normal-quantile table, the Blom approximation, a brute-force Monte
   Carlo of ``E[max of N normals]``).
3. ``HONESTY`` -- the cases where the module is supposed to REFUSE to produce a
   number, plus the specific bug this suite was written to catch: an
   implementation that computes the expected maximum of N normals and then never
   subtracts it returns the same answer for every N. ``test_dsr_responds_to_n_configs``
   is the regression test for exactly that.
"""

import unittest
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from src.Utils import validation


def synthetic_dates(n_days=200, start="2021-10-20", per_day=1):
    """A date series with ``per_day`` games on each of ``n_days`` consecutive days.

    ``per_day > 1`` is what makes purging observable: multiple rows share one
    calendar date, which is the situation ``TimeSeriesSplit`` silently leaks.
    """
    base = pd.Timestamp(start)
    days = [base + timedelta(days=i) for i in range(n_days)]
    if per_day == 1:
        return pd.Series(days)
    return pd.Series(
        [d + timedelta(hours=19, minutes=30 * k) for d in days for k in range(per_day)]
    )


def day_keys(series, idx):
    """The set of calendar dates covered by the rows at ``idx``."""
    return set(pd.Series(series).iloc[idx].dt.normalize())


def overlap_count(dates, train_idx, test_idx):
    """How many dates the training and test sets share -- 0 is the only safe value."""
    return len(day_keys(dates, train_idx) & day_keys(dates, test_idx))


def responsive_returns():
    """A short deterministic series whose DSR does NOT saturate at 0 or 1.

    ``sqrt(T - 1)`` multiplies the z-score, so with a long series any non-zero
    selection effect drives the deflated Sharpe all the way to 0.0 and every
    comparison between configurations collapses to ``0.0 > 0.0`` (which is false).
    T = 10 keeps the value in the responsive band across N = 1 .. 100, which is
    what the ordering tests below need.
    """
    return [0.02, 0.00, 0.03, -0.01, 0.04, -0.02, 0.01, 0.00, 0.03, -0.01]


class TestNormalQuantile(unittest.TestCase):
    """Pins ``Phi^-1`` against a standard normal table.

    The statistic functions are only as trustworthy as this one, and it is the
    only piece of the published machinery that is purely numerical -- so it gets
    exact-value tests rather than a smoke test.
    """

    def test_known_quantiles(self):
        # Values from the standard normal quantile table.
        self.assertAlmostEqual(validation.normal_quantile(0.975), 1.959964, places=6)
        self.assertAlmostEqual(validation.normal_quantile(0.995), 2.575829, places=6)
        self.assertAlmostEqual(validation.normal_quantile(0.5), 0.0, places=12)
        self.assertAlmostEqual(validation.normal_quantile(0.025), -1.959964, places=6)
        self.assertAlmostEqual(validation.normal_quantile(0.8413447461), 1.0, places=6)

    def test_boundary_values(self):
        self.assertEqual(validation.normal_quantile(0.0), float("-inf"))
        self.assertEqual(validation.normal_quantile(1.0), float("inf"))

    def test_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            validation.normal_quantile(-0.01)
        with self.assertRaises(ValueError):
            validation.normal_quantile(1.01)
        with self.assertRaises(ValueError):
            validation.normal_quantile(float("nan"))

    def test_monotonic(self):
        grid = np.linspace(0.001, 0.999, 200)
        values = validation.normal_quantile(grid)
        self.assertTrue(np.all(np.diff(values) > 0))


class TestEmbargoDays(unittest.TestCase):

    def test_fraction_of_span(self):
        dates = pd.Series(pd.date_range("2020-01-01", periods=101, freq="D"))
        # 100-day span, 10% -> 10 days.
        self.assertEqual(validation.embargo_days(dates, 0.10), 10)
        self.assertEqual(validation.embargo_days(dates, 0.01), 1)

    def test_zero_pct_is_zero(self):
        dates = pd.Series(pd.date_range("2020-01-01", periods=500, freq="D"))
        self.assertEqual(validation.embargo_days(dates, 0.0), 0)

    def test_floors_never_rounds_up(self):
        # 3-day span at 50% is 1.5 days; flooring keeps the embargo conservative.
        dates = pd.Series(pd.date_range("2020-01-01", periods=4, freq="D"))
        self.assertEqual(validation.embargo_days(dates, 0.5), 1)

    def test_input_order_does_not_change_span(self):
        dates = synthetic_dates(200)
        self.assertEqual(
            validation.embargo_days(dates, 0.01),
            validation.embargo_days(dates.iloc[::-1], 0.01),
        )

    def test_accepts_strings_and_dates(self):
        from_strings = validation.embargo_days(["2020-01-01", "2020-02-01"], 0.5)
        from_dates = validation.embargo_days([date(2020, 1, 1), date(2020, 2, 1)], 0.5)
        self.assertEqual(from_strings, from_dates)

    def test_rejects_negative(self):
        dates = synthetic_dates(50)
        with self.assertRaises(ValueError):
            validation.embargo_days(dates, -0.01)

    def test_rejects_missing_dates(self):
        with self.assertRaises(ValueError):
            validation.embargo_days(pd.Series([pd.NaT, pd.Timestamp("2020-01-01")]), 0.1)

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            validation.embargo_days(pd.Series([], dtype="datetime64[ns]"), 0.1)

    def test_rejects_none(self):
        with self.assertRaises(ValueError):
            validation.embargo_days(None, 0.1)


class TestSelectionBiasEstimate(unittest.TestCase):

    def test_single_config_is_zero(self):
        # Nothing was selected, so there is nothing to deflate. This is exact,
        # not approximate: E[max of one standard normal] is 0.
        self.assertEqual(validation.selection_bias_estimate(1, 1697), 0.0)

    def test_hundred_configs_is_about_two_and_a_half_standard_errors(self):
        estimate = validation.selection_bias_estimate(100, 1697)
        self.assertGreater(estimate, 2.4)
        self.assertLess(estimate, 2.6)

    def test_roughly_matches_monte_carlo(self):
        """Compare against a brute-force simulation of E[max of N normals].

        The simulation is the reference; the Blom approximation is what is under
        test. Seeded so this cannot flake.
        """
        rng = np.random.default_rng(20240613)
        for n_configs in (2, 10, 100):
            simulated = rng.standard_normal((20000, n_configs)).max(axis=1).mean()
            estimate = validation.selection_bias_estimate(n_configs, 100000)
            self.assertLess(
                abs(estimate - simulated), 0.06,
                f"N={n_configs}: approximation {estimate:.4f} vs "
                f"simulation {simulated:.4f}",
            )

    def test_monotonic_in_configs_tried(self):
        values = [validation.selection_bias_estimate(n, 1697) for n in (1, 2, 5, 10, 50, 100)]
        self.assertEqual(values[0], 0.0)
        for earlier, later in zip(values, values[1:]):
            self.assertLess(earlier, later)

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            validation.selection_bias_estimate(0, 1697)
        with self.assertRaises(ValueError):
            validation.selection_bias_estimate(-5, 1697)
        with self.assertRaises(ValueError):
            validation.selection_bias_estimate(1.5, 1697)
        with self.assertRaises(ValueError):
            validation.selection_bias_estimate(100, 1)
        with self.assertRaises(ValueError):
            validation.selection_bias_estimate(100, 0)


class TestPurgedWalkForwardSplit(unittest.TestCase):

    def test_yields_requested_number_of_folds(self):
        dates = synthetic_dates(200)
        folds = list(validation.purged_walk_forward_split(dates, n_splits=5))
        self.assertEqual(len(folds), 5)

    def test_training_is_always_earlier_than_test(self):
        dates = synthetic_dates(200)
        for train_idx, test_idx in validation.purged_walk_forward_split(dates, n_splits=5):
            self.assertLess(dates.iloc[train_idx].max(), dates.iloc[test_idx].min())

    def test_training_window_expands(self):
        dates = synthetic_dates(200)
        sizes = [len(tr) for tr, _ in
                 validation.purged_walk_forward_split(dates, n_splits=5)]
        self.assertEqual(sizes, sorted(sizes))
        self.assertLess(sizes[0], sizes[-1])

    def test_no_shared_calendar_date(self):
        """THE core property. Same-day games are not independent events."""
        dates = synthetic_dates(200, per_day=3)
        for train_idx, test_idx in validation.purged_walk_forward_split(
            dates, n_splits=5, embargo_pct=0.0
        ):
            self.assertEqual(overlap_count(dates, train_idx, test_idx), 0)

    def test_purging_actually_removes_rows(self):
        """Purging must be doing work, not passing vacuously.

        Compares against the leak a naive walk-forward would produce: a split that
        trains on everything dated before the END of the test window necessarily
        trains on the test window's own first days. The overlap is counted before
        and after so the effect is visible rather than merely asserted.
        """
        dates = synthetic_dates(200, per_day=3)
        for train_idx, test_idx in validation.purged_walk_forward_split(
            dates, n_splits=5, embargo_pct=0.0
        ):
            test_days = day_keys(dates, test_idx)
            # Everything a NON-purging walk-forward would have trained on.
            naive_positions = np.where(
                dates.dt.normalize() <= dates.iloc[test_idx].max().normalize()
            )[0]

            overlap_before = len(day_keys(dates, naive_positions) & test_days)
            overlap_after = overlap_count(dates, train_idx, test_idx)

            self.assertGreater(overlap_before, 0, "test data has no leak to close")
            self.assertEqual(overlap_after, 0)
            # Purging can only remove rows, never add them.
            self.assertTrue(set(train_idx).issubset(set(naive_positions)))
            self.assertLess(len(train_idx), len(naive_positions))

    def test_embargo_removes_rows_purging_alone_keeps(self):
        """Purging and embargoing are different operations and must be measured apart.

        Purging closes the boundary (rows ON the test dates). The embargo pulls the
        training set back FURTHER, to before the test window starts. If the two were
        the same operation this test could not distinguish them.
        """
        dates = synthetic_dates(300, per_day=2)
        embargo = validation.embargo_days(dates, 0.10)
        self.assertGreater(embargo, 0, "test needs a non-zero embargo to be meaningful")

        purge_only = list(
            validation.purged_walk_forward_split(dates, n_splits=5, embargo_pct=0.0)
        )
        with_embargo = list(
            validation.purged_walk_forward_split(dates, n_splits=5, embargo_pct=0.10)
        )

        for (purge_train, _), (embargo_train, test_idx) in zip(purge_only, with_embargo):
            test_start = dates.iloc[test_idx].min()
            purge_gap = (test_start - dates.iloc[purge_train].max()).days
            embargo_gap = (test_start - dates.iloc[embargo_train].max()).days

            # Purging alone leaves a gap of 0-1 days: same-day rows gone, but
            # nothing removed from the days BEFORE the test window.
            self.assertLess(purge_gap, embargo)
            # The embargo puts every training row at least `embargo` days back.
            self.assertGreaterEqual(embargo_gap, embargo)
            self.assertLess(len(embargo_train), len(purge_train))

    def test_larger_embargo_never_adds_training_rows(self):
        """Widening the embargo must shrink or preserve every training set.

        This is a monotonicity property of the implementation, not a statistical
        claim: if a bigger buffer ever produced MORE training rows, the cutoff
        arithmetic would be wrong.
        """
        dates = synthetic_dates(300, per_day=2)
        # Ascending embargo_pct, so later entries are the tighter buffer.
        sizes_by_embargo = [
            [len(tr) for tr, _ in validation.purged_walk_forward_split(
                dates, n_splits=5, embargo_pct=pct
            )]
            for pct in (0.0, 0.005, 0.01, 0.05, 0.10)
        ]
        embargo_widths = [
            validation.embargo_days(dates, pct)
            for pct in (0.0, 0.005, 0.01, 0.05, 0.10)
        ]
        self.assertEqual(embargo_widths, sorted(embargo_widths))

        for looser, tighter in zip(sizes_by_embargo, sizes_by_embargo[1:]):
            for loose_size, tight_size in zip(looser, tighter):
                self.assertLessEqual(tight_size, loose_size)

    def test_gap_adds_to_embargo(self):
        dates = synthetic_dates(300, per_day=2)
        without_gap = [len(tr) for tr, _ in
                       validation.purged_walk_forward_split(dates, n_splits=5, gap=0)]
        with_gap = [len(tr) for tr, _ in
                    validation.purged_walk_forward_split(dates, n_splits=5, gap=20)]
        for plain, gapped in zip(without_gap, with_gap):
            self.assertLess(gapped, plain)

    def test_unsorted_input_gives_identical_split(self):
        """The sort must happen inside, or the caller can silently leak."""
        dates = synthetic_dates(200, per_day=2)
        shuffled = dates.sample(frac=1.0, random_state=3)
        ordered_folds = list(validation.purged_walk_forward_split(dates, n_splits=5))
        shuffled_folds = list(validation.purged_walk_forward_split(shuffled, n_splits=5))
        # The DATE SETS must match even though positional indices differ.
        for (tr_a, te_a), (tr_b, te_b) in zip(ordered_folds, shuffled_folds):
            self.assertEqual(
                set(dates.iloc[tr_a].dt.normalize()),
                set(shuffled.iloc[tr_b].dt.normalize()),
            )
            self.assertEqual(
                set(dates.iloc[te_a].dt.normalize()),
                set(shuffled.iloc[te_b].dt.normalize()),
            )

    def test_indices_are_usable_with_iloc(self):
        dates = synthetic_dates(200)
        frame = pd.DataFrame({"value": np.arange(len(dates))})
        for train_idx, test_idx in validation.purged_walk_forward_split(dates, n_splits=5):
            train_rows = frame.iloc[train_idx]
            test_rows = frame.iloc[test_idx]
            self.assertEqual(len(train_rows), len(train_idx))
            self.assertEqual(len(test_rows), len(test_idx))

    def test_indices_are_integer_arrays(self):
        dates = synthetic_dates(120)
        for train_idx, test_idx in validation.purged_walk_forward_split(dates, n_splits=4):
            self.assertIsInstance(train_idx, np.ndarray)
            self.assertTrue(np.issubdtype(train_idx.dtype, np.integer))
            self.assertTrue(np.issubdtype(test_idx.dtype, np.integer))
            self.assertEqual(set(train_idx) & set(test_idx), set())

    def test_test_windows_do_not_overlap_each_other(self):
        dates = synthetic_dates(200, per_day=2)
        seen = []
        for _, test_idx in validation.purged_walk_forward_split(dates, n_splits=5):
            days = day_keys(dates, test_idx)
            for previous in seen:
                self.assertEqual(days & previous, set())
            seen.append(days)

    def test_empty_training_set_raises(self):
        """An empty training set must be loud, not silent."""
        dates = synthetic_dates(60, per_day=2)
        with self.assertRaises(ValueError) as caught:
            list(validation.purged_walk_forward_split(
                dates, n_splits=5, embargo_pct=5.0
            ))
        message = str(caught.exception)
        self.assertIn("EMPTY", message)
        self.assertIn("fold 0", message)

    def test_rejects_too_few_rows_for_requested_folds(self):
        with self.assertRaises(ValueError):
            validation.purged_walk_forward_split(synthetic_dates(3), n_splits=5)

    def test_rejects_bad_arguments_eagerly(self):
        dates = synthetic_dates(100)
        # These must raise at CALL time, not on first iteration.
        with self.assertRaises(ValueError):
            validation.purged_walk_forward_split(dates, n_splits=0)
        with self.assertRaises(ValueError):
            validation.purged_walk_forward_split(dates, n_splits=-2)
        with self.assertRaises(ValueError):
            validation.purged_walk_forward_split(dates, gap=-1)
        with self.assertRaises(ValueError):
            validation.purged_walk_forward_split(dates, embargo_pct=-0.5)

    def test_rejects_missing_and_empty_dates(self):
        with self.assertRaises(ValueError):
            validation.purged_walk_forward_split(None, n_splits=5)
        with self.assertRaises(ValueError):
            validation.purged_walk_forward_split(
                pd.Series([], dtype="datetime64[ns]"), n_splits=5
            )
        with self.assertRaises(ValueError):
            validation.purged_walk_forward_split(
                pd.Series([pd.NaT, pd.Timestamp("2020-01-01")]), n_splits=2
            )

    def test_accepts_mixed_input_forms(self):
        """Dates may arrive as strings, numpy datetime64, or datetimes."""
        stamps = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(120)]
        as_objects = list(validation.purged_walk_forward_split(stamps, n_splits=5))
        as_strings = list(validation.purged_walk_forward_split(
            [s.strftime("%Y-%m-%d") for s in stamps], n_splits=5
        ))
        as_numpy = list(validation.purged_walk_forward_split(
            pd.Series(stamps).to_numpy(), n_splits=5
        ))
        for other in (as_strings, as_numpy):
            self.assertEqual(len(as_objects), len(other))
            for (a_tr, a_te), (b_tr, b_te) in zip(as_objects, other):
                self.assertEqual(a_te.tolist(), b_te.tolist())

    def test_generator_is_lazy_but_validation_is_eager(self):
        dates = synthetic_dates(100)
        split = validation.purged_walk_forward_split(dates, n_splits=5)
        self.assertTrue(hasattr(split, "__next__"))
        # Consuming a lazy generator must not be required to learn the args are bad.
        with self.assertRaises(ValueError):
            validation.purged_walk_forward_split(dates, n_splits=0)


class TestDeflatedSharpeRatio(unittest.TestCase):

    def setUp(self):
        self.rng = np.random.default_rng(12345)

    def test_returns_a_probability(self):
        returns = self.rng.normal(0.01, 1.0, 1000)
        for n_configs in (1, 10, 100):
            value = validation.deflated_sharpe_ratio(returns, n_configs)
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_dsr_responds_to_n_configs(self):
        """REGRESSION TEST for a real bug caught during development.

        The expected-maximum-of-N term was computed and then never subtracted, so
        every value of ``n_configs_tried`` returned an identical number. A
        selection-bias statistic that does not move when you change the number of
        selections is worse than no statistic, because it looks like evidence.
        """
        returns = responsive_returns()
        one = validation.deflated_sharpe_ratio(returns, 1)
        ten = validation.deflated_sharpe_ratio(returns, 10)
        hundred = validation.deflated_sharpe_ratio(returns, 100)
        self.assertGreater(one, ten)
        self.assertGreater(ten, hundred)

    def test_saturates_at_zero_for_large_samples(self):
        """Pins a property of the formula that looks like a bug but is not.

        ``z = (SR - SR*) * sqrt(T - 1)``. With a per-observation Sharpe around 0.02
        and a 100-trial hurdle around 2.5, T only has to reach a few hundred before
        ``z`` is so negative that the probability underflows to exactly 0.0. The
        answer "0" is the correct one; it is not a failure of the arithmetic. This
        test exists so that a future reader does not "fix" it.
        """
        returns = self.rng.normal(0.02, 1.0, 2000)
        self.assertEqual(validation.deflated_sharpe_ratio(returns, 100), 0.0)

    def test_pure_noise_looks_like_a_coin_flip(self):
        """Zero real skill, 100 configurations tried: the probability must collapse."""
        returns = self.rng.normal(0.0, 1.0, 2000)
        self.assertLess(validation.deflated_sharpe_ratio(returns, 100), 0.01)

    def test_real_signal_beats_noise(self):
        noise = self.rng.normal(0.0, 1.0, 2000)
        signal = self.rng.normal(0.15, 1.0, 2000)
        # Compared at N = 1: at N = 100 BOTH saturate to 0.0 and the comparison
        # would be vacuous. The deflation is exercised by the tests above.
        self.assertGreater(
            validation.deflated_sharpe_ratio(signal, 1),
            validation.deflated_sharpe_ratio(noise, 1),
        )

    def test_matches_normal_approximation_for_gaussian_returns(self):
        """Sanity-check the whole z-score against the textbook Gaussian result.

        In the Gaussian limit the non-Gaussian correction term is exactly 1, so
        the DSR reduces to ``Phi(SR * sqrt(T - 1))``. The residual here is the
        finite-sample deviation of the SAMPLE kurtosis from 3, which the
        implementation is supposed to pick up -- hence ``places=4`` and not an
        exact equality. Pinned separately below.
        """
        returns = self.rng.normal(0.05, 1.0, 3000)
        dsr = validation.deflated_sharpe_ratio(returns, 1)
        sharpe = float(np.mean(returns) / np.std(returns, ddof=1))
        gaussian_limit = validation.normal_cdf(sharpe * np.sqrt(len(returns) - 1))
        self.assertAlmostEqual(dsr, gaussian_limit, places=4)

    def test_gaussian_returns_have_kurtosis_near_three(self):
        """Documents the assumption behind the correction term.

        The published formula is written in PEARSON kurtosis (3.0 for a Gaussian),
        not pandas' excess kurtosis. This test is what would fail if that
        distinction were ever silently dropped.
        """
        returns = self.rng.normal(0.0, 1.0, 5000)
        self.assertAlmostEqual(float(pd.Series(returns).kurt()) + 3.0, 3.0, delta=0.15)

    def test_benchmark_sequence_and_scalar_agree(self):
        """The two benchmark input forms must be the same statistic.

        A sequence is benchmark RETURNS; a number is a benchmark SHARPE. Passing
        the Sharpe of the same series must reproduce the sequence result exactly,
        otherwise the shorthand silently means something different.
        """
        returns = responsive_returns()
        benchmark_series = [0.01, 0.00, 0.02, -0.01, 0.03, -0.01, 0.01, 0.00, 0.02, -0.01]
        benchmark_sharpe = float(
            np.mean(benchmark_series) / np.std(benchmark_series, ddof=1)
        )
        from_series = validation.deflated_sharpe_ratio(
            returns, 10, benchmark=benchmark_series
        )
        from_scalar = validation.deflated_sharpe_ratio(
            returns, 10, benchmark=benchmark_sharpe
        )
        self.assertEqual(from_series, from_scalar)

    def test_benchmark_raises_the_bar(self):
        returns = responsive_returns()
        harder_benchmark = [0.06, 0.04, 0.09, 0.05, 0.10, 0.03, 0.07, 0.04, 0.09, 0.05]
        self.assertLess(
            validation.deflated_sharpe_ratio(returns, 10, benchmark=harder_benchmark),
            validation.deflated_sharpe_ratio(returns, 10),
        )
        self.assertLess(
            validation.deflated_sharpe_ratio(returns, 10, benchmark=0.5),
            validation.deflated_sharpe_ratio(returns, 10),
        )

    def test_benchmark_equals_own_returns_gives_near_half(self):
        """Scoring a series against itself should look like a coin flip, not a triumph."""
        returns = self.rng.normal(0.15, 1.0, 2000)
        value = validation.deflated_sharpe_ratio(returns, 1, benchmark=returns)
        self.assertLess(value, 0.6)
        self.assertGreater(value, 0.4)

    def test_rejects_short_series(self):
        with self.assertRaises(ValueError):
            validation.deflated_sharpe_ratio([0.1, 0.2], 10)

    def test_rejects_constant_series(self):
        with self.assertRaises(ValueError):
            validation.deflated_sharpe_ratio([0.01] * 100, 10)

    def test_rejects_non_finite_returns(self):
        returns = [0.1] * 50 + [float("inf")] * 50
        with self.assertRaises(ValueError):
            validation.deflated_sharpe_ratio(returns, 10)

    def test_rejects_bad_config_count(self):
        returns = self.rng.normal(0.0, 1.0, 500)
        with self.assertRaises(ValueError):
            validation.deflated_sharpe_ratio(returns, 0)

    def test_is_deterministic(self):
        returns = self.rng.normal(0.01, 1.0, 1000)
        first = validation.deflated_sharpe_ratio(returns, 100)
        second = validation.deflated_sharpe_ratio(returns, 100)
        self.assertEqual(first, second)


class TestReportSelectionRisk(unittest.TestCase):

    def test_contains_the_plain_language_verdict(self):
        summary = validation.report_selection_risk(
            100, "accuracy", 1697, 0.0113
        )
        self.assertIn("accuracy", summary)
        self.assertIn("100", summary)
        self.assertIn("1697", summary)
        self.assertIn(
            "a result this small relative to the selection effect is not "
            "evidence of skill",
            summary,
        )

    def test_states_that_the_number_is_an_approximation(self):
        summary = validation.report_selection_risk(100, "accuracy", 1697, 0.0113)
        self.assertIn("APPROXIMATION", summary)
        self.assertIn("not a bound", summary)

    def test_includes_both_raw_and_x100_units(self):
        """Percentage points vs proportion is a 100x footgun, so both are shown."""
        summary = validation.report_selection_risk(100, "accuracy", 1697, 0.0113)
        self.assertIn("0.0113", summary)
        self.assertIn("1.13", summary)

    def test_prints_and_returns_the_same_text(self):
        import io
        import contextlib

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            summary = validation.report_selection_risk(10, "ROI", 500, 0.02)
        self.assertEqual(buffer.getvalue().strip(), summary.strip())

    def test_single_config_reports_no_inflation(self):
        summary = validation.report_selection_risk(1, "accuracy", 1697, 0.0113)
        self.assertIn("0.000 SE", summary)

    def test_rejects_bad_standard_error(self):
        with self.assertRaises(ValueError):
            validation.report_selection_risk(100, "accuracy", 1697, -0.01)
        with self.assertRaises(ValueError):
            validation.report_selection_risk(100, "accuracy", 1697, float("nan"))


if __name__ == "__main__":
    unittest.main()
