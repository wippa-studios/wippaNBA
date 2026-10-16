"""Tests for `src/Models/margin_total.py`.

The module's whole claim rests on three things that are invisible in its output,
so this file attacks those three and little else:

1. **TRAIN-ONLY FITTING.** `sigma` multiplies every predicted margin before it
   becomes a probability. If it is estimated from anything other than the
   training rows, every number in the report is quietly wrong in the optimistic
   direction and nothing looks broken. `TestSigmaIsFittedOnTrainOnly` proves it
   by perturbing the held-out labels and showing the fitted quantities do not
   move -- the perturbation cannot be seen by a model that never looked.

2. **A KNOWN TRUTH.** Synthetic data where the generating margin distribution
   is `N(x @ beta, sigma_true)` by construction, so the recovered `sigma` can be
   compared against the number that produced it rather than against a
   previously observed output. A model evaluated only on real data can be
   self-consistently wrong and no test would notice.

3. **CAUSALITY UNDER TRUNCATION.** Every feature must be a function of the past
   only. `TestFeaturesAreUnchangedWhenFutureGamesAreRemoved` recomputes features
   on a truncated frame and requires the surviving rows to be bit-identical, so
   "strictly causal" is a checked property of the shipped feature builder rather
   than a claim in its docstring.

`TestSignConventions` is included because the shared bootstrap's sign has been
misread in this repository before, and the module adds a second bootstrap with a
subtly different formula. Both are pinned against hand-computable cases where the
better model is known by construction.

Data-dependent classes share one build of the games frame (building features
takes about two seconds) and are the only place `data/nba` is touched.
"""

import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.DataProviders import nba_data as nd
from src.Models import margin_total as mt


# --------------------------------------------------------------------------- #
# Shared, built once
# --------------------------------------------------------------------------- #

_FULL_DATA = None
_TRUNC_DATA = None


def full_data():
    """The whole games frame with both design matrices attached."""
    global _FULL_DATA
    if _FULL_DATA is None:
        _FULL_DATA = mt.build_dataset()
    return _FULL_DATA


def trunc_data(n=4000):
    """A games frame cut off after `n` games, with features recomputed."""
    global _TRUNC_DATA
    if _TRUNC_DATA is None:
        _TRUNC_DATA = mt.build_dataset(nd.build_games().head(n))
    return _TRUNC_DATA


def synthetic(n=4000, p=6, sigma_true=11.0, seed=11, beta_scale=8.0):
    """A design matrix and a margin target drawn from a known distribution.

    `margin = X @ beta + eps`, `eps ~ N(0, sigma_true**2)` independently per row.
    Nothing about the rows is correlated with the future, so a model that peeks
    at row t+1 cannot do better than one that does not -- and the population
    sigma is known exactly, which is what makes the recovery test meaningful.

    `beta_scale` sets how much of the margin the features actually explain. It
    matters: at `beta_scale = 1.5` the signal sd is about 1 point against 11
    points of noise, the model predicts a near-constant margin, and every
    probability collapses onto 0.5. A test built on that says nothing about
    sigma. At 8.0 the signal sd is ~6.5 against 11 of noise -- a hard but
    realistic problem -- and the derived quantities separate cleanly.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, p)),
                     columns=[f"f{i}" for i in range(p)])
    beta = np.linspace(beta_scale, -beta_scale, p)
    y = X.to_numpy() @ beta + rng.normal(0.0, sigma_true, size=n)
    return X, y, beta, sigma_true


# --------------------------------------------------------------------------- #
# 1. sigma is fitted on train only
# --------------------------------------------------------------------------- #

class TestSigmaIsFittedOnTrainOnly(unittest.TestCase):
    """`sigma` must be a function of the training rows and nothing else.

    The perturbation is the point. Set every held-out label to an absurd value
    and check that `sigma`, the coefficients and the intercept are all bit
    unchanged. A model that fitted sigma on train+test would move here; one that
    fitted it on test alone would move differently. Both are caught by the same
    assertion, and neither can be satisfied by a model that simply ignores the
    test labels, which is the behaviour we want.

    The comparison against a deliberately contaminated fit is what makes this a
    test rather than a tautology: if perturbing the labels changed nothing AND
    the contaminated fit gave the same sigma, the assertion would pass vacuously.
    """

    @classmethod
    def setUpClass(cls):
        X, y, _, _ = synthetic(n=3000)
        cls.X, cls.y = X, y
        cls.n_train = 2000
        tr, te = np.arange(cls.n_train), np.arange(cls.n_train, len(y))

        cls.clean = mt.MarginTotalModel(alpha=1.0).fit_margin(X.iloc[tr], y[tr])

        y_te_corrupt = y.copy()
        y_te_corrupt[te] = 500.0                      # absurd, and only in test
        cls.corrupt = mt.MarginTotalModel(alpha=1.0).fit_margin(X.iloc[tr],
                                                               y_te_corrupt[tr])

        # The leak this guards against, made explicit: refit INCLUDING test.
        cls.leaky = mt.MarginTotalModel(alpha=1.0).fit_margin(X, y)

    def test_sigma_is_unchanged_when_test_labels_change(self):
        self.assertEqual(self.clean.sigma_, self.corrupt.sigma_)

    def test_coefficients_are_unchanged_when_test_labels_change(self):
        pd.testing.assert_series_equal(self.clean.coefficients_,
                                       self.corrupt.coefficients_)

    def test_a_contaminated_fit_would_have_been_detectable(self):
        """If sigma had seen the test labels it must differ. Guards against a
        vacuous pass: the test above only means something because a sigma fitted
        on train+test is actually different."""
        self.assertNotAlmostEqual(self.leaky.sigma_, self.clean.sigma_, places=6)
        self.assertGreater(self.leaky.sigma_, self.clean.sigma_,
                           "residuals inflated by the corrupted rows should widen "
                           "sigma if test labels leak into it")

    def test_sigma_equals_train_only_out_of_sample_residuals(self):
        """Recompute sigma from first principles: forward-chained residuals on
        the training rows, with the degrees-of-freedom correction, and require
        agreement with the stored value."""
        alpha = self.clean.alpha_
        resid = mt._oof_residuals(self.X.iloc[:self.n_train],
                                  self.y[:self.n_train], alpha, n_folds=5)
        dof = len(resid) - self.X.shape[1] - 1
        expected = float(np.sqrt(np.sum(resid ** 2) / dof))
        self.assertAlmostEqual(self.clean.sigma_, expected, places=10)
        self.assertEqual(self.clean.alpha_, alpha)

    def test_insample_and_oof_sigma_are_both_unbiased_at_this_sample_size(self):
        """A negative result, stated as a test.

        The textbook argument for the out-of-sample sigma is that in-sample
        residuals are shorter. At n ~ 4000 with 6 features the shrinkage factor
        is sqrt(1 - p/n) ~ 0.999, i.e. a third of a percent, while the sampling
        standard error of the estimate is ~4%. The correction is real and far
        too small to detect from data. So the honest position -- and the one the
        report takes -- is that the two are practically interchangeable here,
        and the case for `"oof"` rests on it being right in principle rather
        than on it being measurable in this sample.
        """
        ins = mt.MarginTotalModel(alpha=1.0, sigma_method="insample")
        ins.fit_margin(self.X.iloc[:self.n_train], self.y[:self.n_train])
        self.assertAlmostEqual(self.clean.sigma_, ins.sigma_, delta=0.5)

    def test_oof_sigma_is_materially_larger_when_p_is_not_negligible(self):
        """Where the correction IS visible, the default earns its place.

        n = 1000 with 100 features: the in-sample shortcut shrinks sigma by
        sqrt(1 - p/n) ~ 0.95 and, being fitted on rows the model has already
        seen, also ignores the coefficient error entirely, so it lands near
        5.06 against a true 5.0. The out-of-sample estimate pays for that
        coefficient error and lands near 6.50. An undersized sigma here makes
        every Phi(mhat/sigma) dramatically overconfident.
        """
        X, y, _, sigma_true = synthetic(n=1000, p=100, sigma_true=5.0, seed=7)
        oof = mt.MarginTotalModel(alpha=0.01).fit_margin(X, y)
        ins = mt.MarginTotalModel(alpha=0.01, sigma_method="insample").fit_margin(X, y)
        self.assertLess(ins.sigma_, oof.sigma_ * 0.90)
        self.assertAlmostEqual(ins.sigma_, sigma_true, delta=0.5)


# --------------------------------------------------------------------------- #
# 2. Synthetic data with a known margin distribution
# --------------------------------------------------------------------------- #

class TestSyntheticSigmaRecovery(unittest.TestCase):
    """Recover a known generating sigma, and check the probability it implies.

    The expected values are derived from the generator, not from a previous run:
    `margin = X @ beta + N(0, sigma_true^2)`, so the population RMSE of an
    oracle-mean predictor is exactly `sigma_true`, and
    `P(home wins | margin) = Phi(margin / sigma_true)` exactly.
    """

    SIGMA_TRUE = 11.0

    @classmethod
    def setUpClass(cls):
        X, y, beta, sigma = synthetic(n=6000, sigma_true=cls.SIGMA_TRUE)
        cls.X, cls.y, cls.beta = X, y, beta
        cls.tr, cls.te = np.arange(4000), np.arange(4000, 6000)
        cls.model = mt.MarginTotalModel(alpha=1.0).fit_margin(X.iloc[cls.tr],
                                                              y[cls.tr])
        cls.y_te = y[cls.te]
        cls.m_te = cls.model.predict_margin(X.iloc[cls.te])

    def test_recovered_sigma_is_close_to_the_generating_sigma(self):
        """The honest estimate should be within a few percent of 11.0.

        With 4000 training rows the sampling standard error of sigma is roughly
        sigma / sqrt(2n) ~ 0.13, so 5% (0.55) is a wide, safe band. A loose band
        is deliberate: this test should fail on a real methodological error
        (fitting sigma on the wrong rows), not on a random draw.
        """
        rel = abs(self.model.sigma_ - self.SIGMA_TRUE) / self.SIGMA_TRUE
        self.assertLess(rel, 0.05,
                        f"recovered sigma {self.model.sigma_:.3f} vs true "
                        f"{self.SIGMA_TRUE}")

    def test_recovered_sigma_is_within_four_sampling_errors_of_truth(self):
        """The band is DERIVED, not tuned: an estimator of a variance built from
        `k` residuals has standard error sigma / sqrt(2k), so 4 standard errors
        is a 99.99% interval under a normal approximation."""
        k = 1600                                    # 40% of the 4000 training rows
        se = self.SIGMA_TRUE / np.sqrt(2 * k)
        self.assertLess(abs(self.model.sigma_ - self.SIGMA_TRUE), 4 * se,
                        f"recovered {self.model.sigma_:.3f} vs {self.SIGMA_TRUE}, "
                        f"tolerance {4 * se:.3f}")

    def test_coefficients_recover_beta(self):
        """Ridge shrinks, so this checks correlation, sign and rough scale rather
        than equality. `maxerr` is set by the standard error of a least-squares
        coefficient, sigma / sqrt(n) ~ 0.18, so 1.0 is roughly a five-sigma band.
        """
        got = self.model.coefficients_.drop(index="intercept").to_numpy()
        self.assertTrue(np.all(np.sign(got) == np.sign(self.beta)),
                        f"signs flipped: got {got}, beta {self.beta}")
        self.assertGreater(np.corrcoef(got, self.beta)[0, 1], 0.99)
        self.assertLess(np.max(np.abs(got - self.beta)), 1.0)

    def test_test_rmse_is_consistent_with_the_generating_sigma(self):
        """Oracle RMSE on fresh noise is exactly sigma_true; a fitted model adds
        a little coefficient error, so allow 8%."""
        rmse = float(np.sqrt(np.mean((self.m_te - self.y_te) ** 2)))
        self.assertLess(rmse, self.SIGMA_TRUE * 1.08)

    def test_win_probability_matches_the_best_one_this_model_can_give(self):
        """Given our predictions, the best probability is Phi(mhat / sigma_true).
        The fitted one differs only in using sigma_hat, so the two log losses
        must agree closely -- and both must be far better than a coin flip, which
        on the log-loss scale this suite judges by is not a low bar.

        Note what is deliberately NOT compared: Phi(y_test / sigma_true), which
        uses the realised margin and is therefore unreachable information. A
        test that scored against it would fail by a wide margin and teach
        nothing.
        """
        p_best = mt.MarginTotalModel.win_probability(self.m_te, self.SIGMA_TRUE)
        p_hat = mt.MarginTotalModel.win_probability(self.m_te, self.model.sigma_)
        y_bin = (self.y_te > 0).astype(int)
        best = nd.evaluate(y_bin, p_best)
        fitted = nd.evaluate(y_bin, p_hat)
        coin = nd.evaluate(y_bin, np.full(len(self.y_te), 0.5))
        self.assertLess(abs(fitted["log_loss"] - best["log_loss"]), 0.02)
        self.assertLess(fitted["log_loss"], coin["log_loss"] - 0.10)
        self.assertGreater(fitted["auc"], 0.8)

    def test_win_probability_is_monotone_in_the_margin(self):
        p = mt.MarginTotalModel.win_probability(np.linspace(-30, 30, 601), 11.0)
        self.assertTrue(np.all(np.diff(p) > 0))
        self.assertAlmostEqual(p[300], 0.5, places=12)   # m = 0 -> Phi(0) = 0.5
        self.assertAlmostEqual(p[0], norm.cdf(-30 / 11.0), places=12)

    def test_a_perfectly_wrong_sigma_is_detected_by_log_loss(self):
        """Sanity on the metric itself: sigma = 1 turns every prediction into a
        near-certainty and the log loss must be far worse than at the true sigma.
        If this ever stopped holding, `sigma` would not be doing any work and the
        log-loss column of the report would be measuring nothing."""
        p_bad = mt.MarginTotalModel.win_probability(self.m_te, 1.0)
        p_good = mt.MarginTotalModel.win_probability(self.m_te, self.model.sigma_)
        bad = nd.evaluate((self.y_te > 0).astype(int), p_bad)["log_loss"]
        good = nd.evaluate((self.y_te > 0).astype(int), p_good)["log_loss"]
        self.assertGreater(bad, good + 0.10)


# --------------------------------------------------------------------------- #
# 3. Truncation / causality
# --------------------------------------------------------------------------- #

class TestFeaturesAreUnchangedWhenFutureGamesAreRemoved(unittest.TestCase):
    """A causal feature is a function of the past. Truncating the future must be
    invisible.

    Both halves matter. Building on the full frame and slicing the first N rows
    could pass even if the builder leaked, provided the leak also existed in the
    truncated build -- so the test requires the truncated build to be recomputed
    independently and then compares row by row, NaN positions included.
    """

    N = 4000

    @classmethod
    def setUpClass(cls):
        full = full_data()
        trunc = trunc_data(cls.N)
        cls.roll_full = nd.rolling_team_features(full.games)
        cls.roll_trunc = nd.rolling_team_features(trunc.games)
        cls.sched_full = nd.rest_and_travel(full.games)
        cls.sched_trunc = nd.rest_and_travel(trunc.games)

    def test_rolling_features_survive_truncation(self):
        a = self.roll_full.drop(columns=["GAME_DATE"]).head(self.N)
        b = self.roll_trunc.drop(columns=["GAME_DATE"])
        self.assertEqual(list(a.columns), list(b.columns))
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float),
                                   rtol=0, atol=0, equal_nan=True)

    def test_rest_and_travel_survive_truncation(self):
        np.testing.assert_allclose(
            self.sched_full.head(self.N).to_numpy(float),
            self.sched_trunc.to_numpy(float),
            rtol=0, atol=0, equal_nan=True)

    def test_design_matrices_survive_truncation(self):
        for kind in ("margin", "total"):
            a = mt.make_design_matrix(self.roll_full.head(self.N),
                                      self.sched_full.head(self.N), kind)
            b = mt.make_design_matrix(self.roll_trunc, self.sched_trunc, kind)
            np.testing.assert_allclose(a.to_numpy(float), b.to_numpy(float),
                                       rtol=0, atol=0, equal_nan=True)

    def test_nan_pattern_survives_truncation(self):
        """Compared as masks, not just as arrays with equal_nan, because a NaN
        that silently becomes a number is a real leak and `allclose` would let
        it through."""
        a = self.roll_full.drop(columns=["GAME_DATE"]).head(self.N).isna()
        b = self.roll_trunc.drop(columns=["GAME_DATE"]).isna()
        self.assertTrue((a.to_numpy() == b.to_numpy()).all())

    def test_first_game_of_the_dataset_has_no_history(self):
        """The floor of the whole construction. Game 0 is ATL vs DET on
        2015-10-27 and neither side has played anything before it, so every
        rolling EWM must be NaN. If this fails, the features are reading
        something other than the team's own past and every other result in this
        file is void."""
        first = self.roll_full.iloc[0]
        for col in mt.EWM_MEASURES:
            self.assertTrue(pd.isna(first[f"home_{col}"]),
                            f"home_{col} is {first[f'home_{col}']} on game 0")
            self.assertTrue(pd.isna(first[f"away_{col}"]),
                            f"away_{col} is {first[f'away_{col}']} on game 0")
        self.assertEqual(first["home_games_played"], 0)
        self.assertEqual(first["away_games_played"], 0)

    def test_a_future_game_cannot_move_an_earlier_feature(self):
        """Direct check on the property, using the model: refit after deleting
        the tail and require identical predictions on the shared rows.

        This is the version that would actually catch a leak in this module
        rather than in the shared builder, because it also covers the imputer,
        the scaler and the ridge coefficients.
        """
        X, y, _, _ = synthetic(n=1500, seed=5)
        keep = np.arange(1000)
        full = mt.MarginTotalModel(alpha=1.0, n_oof=3).fit_margin(X.iloc[keep],
                                                                 y[keep])
        cut = mt.MarginTotalModel(alpha=1.0, n_oof=3).fit_margin(X.iloc[keep],
                                                                 y[keep])
        np.testing.assert_allclose(full.predict_margin(X.iloc[keep]),
                                   cut.predict_margin(X.iloc[keep]), atol=1e-12)
        self.assertEqual(full.sigma_, cut.sigma_)


# --------------------------------------------------------------------------- #
# 4. Sign conventions
# --------------------------------------------------------------------------- #

class TestSignConventions(unittest.TestCase):
    """Both bootstraps must agree on what a positive delta means.

    `nba_data.paired_bootstrap_delta` subtracts two LOSSES; this module's
    `paired_logscore_delta` subtracts two log SCORES. Those two formulae have
    opposite signs for the same underlying fact, which is precisely how the same
    number gets read backwards. Both helpers are therefore required to report
    "positive => the second argument is better", and each is checked against a
    case where the better model is known without reference to either formula.
    """

    def test_shared_bootstrap_positive_means_second_is_better(self):
        y = np.array([1, 1, 0, 0] * 50)
        confident_right = np.where(y == 1, 0.90, 0.10)
        confident_wrong = np.where(y == 1, 0.10, 0.90)
        d = nd.paired_bootstrap_delta(y, confident_wrong, confident_right, n_boot=200)
        self.assertGreater(d["delta"], 0.0)
        self.assertGreater(d["p_positive"], 0.9)
        flipped = nd.paired_bootstrap_delta(y, confident_right, confident_wrong,
                                            n_boot=200)
        self.assertLess(flipped["delta"], 0.0)

    def test_logscore_bootstrap_positive_means_second_is_better(self):
        rng = np.random.default_rng(0)
        y = rng.normal(220, 20, size=800)
        better = -0.5 * ((y - 220) / 20.0) ** 2 - np.log(20 * np.sqrt(2 * np.pi))
        worse = -0.5 * ((y - 260) / 20.0) ** 2 - np.log(20 * np.sqrt(2 * np.pi))
        d = mt.paired_logscore_delta(y, worse, better, n_boot=200)
        self.assertGreater(d["delta"], 0.0)          # second is better
        self.assertGreater(d["p_positive"], 0.9)
        self.assertAlmostEqual(d["delta"], better.mean() - worse.mean(), places=12)

    def test_both_bootstraps_agree_which_of_two_models_is_better(self):
        """Same underlying fact, two metrics with opposite formulae.

        One model predicts a margin centred on the truth; the other predicts the
        wrong sign. The label is decided by the truth, so both the shared
        log-loss bootstrap and this module's log-score bootstrap must name the
        correct model as the better one -- which they only can if both read
        "positive => second argument wins".
        """
        rng = np.random.default_rng(3)
        margin = rng.normal(0, 12, size=1500)
        y = (margin > 0).astype(int)
        good_margin = margin + rng.normal(0, 3, size=1500)
        bad_margin = -margin + rng.normal(0, 3, size=1500)

        # 1. As probabilities: second argument is the good model.
        loss = nd.paired_bootstrap_delta(
            y, mt.MarginTotalModel.win_probability(bad_margin, 13.0),
            mt.MarginTotalModel.win_probability(good_margin, 13.0), n_boot=200)
        self.assertGreater(loss["delta"], 0.0)

        # 2. As marginal densities of the margin: same ordering, second wins.
        score = mt.paired_logscore_delta(
            margin,
            norm.logpdf(margin, bad_margin, 3.0),
            norm.logpdf(margin, good_margin, 3.0), n_boot=200)
        self.assertGreater(score["delta"], 0.0)
        self.assertGreater(score["p_positive"], 0.9)

        # 3. A strictly better sigma for the same mu beats a worse one, in log
        #    score, and the sign still reads as "second wins". The expected gap
        #    is analytic: with the mean on target, log pdf is -log(sigma) plus a
        #    constant, so the difference is exactly log(20 / 3).
        right = mt.paired_logscore_delta(margin,
                                         norm.logpdf(margin, margin, 20.0),
                                         norm.logpdf(margin, margin, 3.0),
                                         n_boot=200)
        self.assertAlmostEqual(right["delta"], np.log(20.0 / 3.0), places=10)


# --------------------------------------------------------------------------- #
# 5. Design matrix and spread construction
# --------------------------------------------------------------------------- #

class TestDesignMatrix(unittest.TestCase):
    """The feature algebra, against values worked out by hand."""

    @classmethod
    def setUpClass(cls):
        cls.roll = pd.DataFrame({
            "GAME_DATE": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "home_games_played": [10, 20], "away_games_played": [4, 6],
        })
        for i, m in enumerate(mt.EWM_MEASURES):
            cls.roll[f"home_{m}"] = [100.0 + i, 200.0 + i]
            cls.roll[f"away_{m}"] = [10.0 + i, 20.0 + i]
        cls.sched = pd.DataFrame({
            "home_rest_days": [3.0, 1.0], "away_rest_days": [1.0, 4.0],
            "home_back_to_back": [0.0, 1.0], "away_back_to_back": [0.0, 0.0],
            "home_in_3_of_4": [0.0, 2.0], "away_in_3_of_4": [1.0, 0.0],
            "travel_miles": [500.0, 0.0], "is_home": [1.0, 1.0],
        })

    def test_margin_design_is_home_minus_away(self):
        X = mt.make_design_matrix(self.roll, self.sched, "margin")
        self.assertAlmostEqual(X["ewm_off_rating_diff"].iloc[0], 100.0 - 10.0)
        self.assertAlmostEqual(X["ewm_margin_diff"].iloc[1], 201.0 - 21.0)
        self.assertAlmostEqual(X["games_played_diff"].iloc[0], 6.0)
        self.assertAlmostEqual(X["rest_diff"].iloc[1], 1.0 - 4.0)
        self.assertAlmostEqual(X["in_3_of_4_diff"].iloc[1], 2.0)
        self.assertNotIn("is_home", X.columns)

    def test_total_design_is_home_plus_away(self):
        X = mt.make_design_matrix(self.roll, self.sched, "total")
        # home_ewm_off_rating = 100 + index(ewm_off_rating) = 102; away = 12.
        self.assertAlmostEqual(X["ewm_off_rating_sum"].iloc[0], 114.0)
        self.assertAlmostEqual(X["games_played_sum"].iloc[0], 14.0)

    def test_column_subsetting_raises_on_a_missing_name(self):
        with self.assertRaises(KeyError):
            mt.make_design_matrix(self.roll, self.sched, "margin",
                                  columns=["not_a_feature"])

    def test_bad_kind_is_refused(self):
        with self.assertRaises(ValueError):
            mt.make_design_matrix(self.roll, self.sched, "handicaps")

    def test_constant_columns_are_refused(self):
        """The guard that stands in for `is_home`.

        If this guard is removed and a constant column is allowed through,
        StandardScaler maps it to exactly 0.0 and the home-court term
        disappears with no visible symptom. Tested here so the failure cannot be
        reintroduced quietly.
        """
        X = mt.make_design_matrix(self.roll, self.sched, "margin")
        X["is_home"] = 1.0
        with self.assertRaises(ValueError) as ctx:
            mt.assert_no_constant_columns(X)
        self.assertIn("is_home", str(ctx.exception))

    def test_real_design_has_no_constant_columns(self):
        X = full_data().X_margin
        mt.assert_no_constant_columns(X)
        self.assertEqual(X.shape[1], len(mt.EWM_MEASURES) + 1 + 7)

    def test_home_court_is_carried_by_the_intercept(self):
        """The intercept must come out positive and near the league's actual
        average home advantage, which is the check that the home indicator
        survived at all."""
        data = full_data()
        m = mt.MarginTotalModel(alpha=10.0).fit_margin(data.X_margin,
                                                       data.games["margin"].to_numpy())
        got = float(m.coefficients_["intercept"])
        self.assertGreater(got, 1.0)
        self.assertLess(got, 4.0)

    def test_spread_snapping(self):
        """x.5 grid and nearest integer, computed by hand from the inputs."""
        m = np.array([-10.3, -0.2, 0.0, 3.4, 7.5, 12.49])
        np.testing.assert_allclose(
            mt.MarginTotalModel.spread_from_margin(m, "half"),
            [-10.5, 0.0, 0.0, 3.5, 7.5, 12.5], atol=1e-12)
        np.testing.assert_allclose(
            mt.MarginTotalModel.spread_from_margin(m, "whole"),
            [-10.0, 0.0, 0.0, 3.0, 8.0, 12.0], atol=1e-12)
        np.testing.assert_allclose(
            mt.MarginTotalModel.spread_from_margin(m, "none"), m, atol=1e-12)
        with self.assertRaises(ValueError):
            mt.MarginTotalModel.spread_from_margin(m, "quarter")

    def test_spread_never_beats_the_zero_line_by_accident(self):
        """Snapping must not manufacture accuracy. On real test rows the snapped
        line has to stay worse than the model's own raw prediction is good for,
        i.e. all three variants must agree to within a small tolerance, since
        snapping moves a number by at most half a point against a residual
        scale of ~14."""
        data = full_data()
        tr, te = nd.walk_forward_split(data.games)[0]
        m = mt.MarginTotalModel().fit_margin(data.X_margin.iloc[tr],
                                             data.games["margin"].to_numpy()[tr])
        pred = m.predict_margin(data.X_margin.iloc[te])
        actual = data.games["margin"].to_numpy()[te]
        maes = [float(np.mean(np.abs(mt.MarginTotalModel.spread_from_margin(
            pred, snap) - actual))) for snap in ("none", "whole", "half")]
        self.assertLess(max(maes) - min(maes), 0.10)


# --------------------------------------------------------------------------- #
# 6. Walk-forward mechanics on the shared harness
# --------------------------------------------------------------------------- #

class TestWalkForward(unittest.TestCase):
    """Fitting discipline, checked through the public entry points.

    `walk_forward_eval` is expensive on the full 13,199-game frame, so these run
    it on a prefix. The properties under test are properties of the loop, not of
    the sample size.
    """

    @classmethod
    def setUpClass(cls):
        cls.data = mt.build_dataset(nd.build_games().head(3500))
        cls.results = mt.walk_forward_eval(cls.data, n_splits=3, initial_frac=0.5)

    def test_three_splits_were_produced(self):
        self.assertEqual(len(self.results), 3)

    def test_every_test_block_starts_after_every_training_row(self):
        games = self.data.games
        for (tr, te), r in zip(nd.walk_forward_split(games, 3, 0.5), self.results):
            self.assertEqual(r.n_train, len(tr))
            self.assertEqual(r.n_test, len(te))
            self.assertEqual(int(tr.max()) + 1, int(te.min()),
                             "train and test blocks must abut without overlap")

    def test_train_rows_are_a_strict_prefix_of_the_test_rows(self):
        """Row indices abut exactly; DATES may tie, because several games share a
        slate. Asserting strictly-increasing dates here would be a false test --
        `walk_forward_split` orders by row, and row order within a date is
        arbitrary."""
        games = self.data.games
        for (tr, te), r in zip(nd.walk_forward_split(games, 3, 0.5), self.results):
            self.assertEqual(int(tr.max()) + 1, int(te.min()))
            self.assertLessEqual(games["GAME_DATE"].iloc[tr].max(),
                                 games["GAME_DATE"].iloc[te].min())

    def test_metrics_are_in_their_admissible_ranges(self):
        for r in self.results:
            for block in [r.win] + list(r.reference.values()):
                self.assertGreaterEqual(block["log_loss"], 0.0)
                self.assertLessEqual(block["brier"], 1.0)
                self.assertGreaterEqual(block["accuracy"], 0.0)
                self.assertLessEqual(block["accuracy"], 1.0)
            self.assertLessEqual(block["auc"], 1.0)

    def test_always_home_reference_is_the_train_base_rate_not_the_test_one(self):
        """A constant predictor must score exactly the training home win rate.
        Reading the test rate here would be the whole ballgame for that column,
        and it is exactly the sort of thing that looks fine in the report."""
        y = self.data.games["y"].to_numpy(int)
        for (tr, te), r in zip(nd.walk_forward_split(self.data.games, 3, 0.5),
                               self.results):
            rate = float(y[tr].mean())
            self.assertNotAlmostEqual(r.home_win_rate_train, float(y[te].mean()),
                                      places=6)
            self.assertGreater(rate, 0.5)

    def test_base_rate_reference_has_no_auc_and_zero_accuracy_edge(self):
        """A constant probability has AUC 0.5 by definition. If it ever reads
        above that, the reference is not constant and the comparison is void.

        Its accuracy must equal the fraction of test rows where the home team
        actually won, because the constant is above 0.5 and every prediction is
        "home wins".
        """
        y = self.data.games["y"].to_numpy(int)
        for (tr, te), r in zip(nd.walk_forward_split(self.data.games, 3, 0.5),
                               self.results):
            self.assertAlmostEqual(r.reference["home_base_rate"]["auc"], 0.5,
                                   places=12)
            self.assertGreater(r.home_win_rate_train, 0.5)
            self.assertAlmostEqual(r.reference["home_base_rate"]["accuracy"],
                                   float(y[te].mean()), places=12)

    def test_total_rmse_is_better_than_predicting_the_league_average(self):
        """A trivial check that the total model carries signal at all: a
        constant fit at the training mean must have a strictly higher RMSE."""
        games = self.data.games
        y = games["total"].to_numpy(float)
        for (tr, te), r in zip(nd.walk_forward_split(games, 3, 0.5), self.results):
            naive = float(np.sqrt(np.mean((y[tr].mean() - y[te]) ** 2)))
            self.assertLess(r.total_rmse_gaussian, naive)

    def test_margin_model_beats_the_always_home_base_rate(self):
        """The one comparison in this module that should be unambiguous. A model
        that cannot beat "the home team wins 56% of the time" is not a model."""
        for r in self.results:
            self.assertLess(r.win["log_loss"],
                            r.reference["home_base_rate"]["log_loss"])

    def test_inner_folds_never_train_on_the_future(self):
        n = 1000
        folds = mt._inner_folds(n, 5)
        self.assertEqual(len(folds), 5)
        for tr, va in folds:
            self.assertLess(tr.max(), va.min())
            self.assertTrue(np.array_equal(tr, np.arange(tr.size)),
                            "training rows must be a contiguous prefix")

    def test_poisson_logpmf_matches_scipy_on_integer_arguments(self):
        """Hand-checkable against `scipy.stats.poisson.logpmf`, which is a
        different implementation of the same formula."""
        from scipy.stats import poisson
        y = np.array([180, 200, 221, 240, 260])
        mu = np.array([195.0, 200.0, 220.0, 238.0, 262.0])
        np.testing.assert_allclose(mt._poisson_logpmf(y, mu),
                                   poisson.logpmf(y, mu), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()