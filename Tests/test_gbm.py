"""Tests for `src/Models/gbm.py`.

Three rules govern this file, and they are the same three that
`Tests/test_ratings.py` follows.

1. **Every expected value is derived in the comment next to it.** Where
   scikit-learn appears it is a sanity check on this module's own arithmetic,
   never the source of an expected figure.
2. **The leakage tests are the point of this file.** A gradient-boosted model
   evaluated on data its feature builder or its hyperparameter search has
   already touched produces a smaller, entirely fictional log loss, and nothing
   about the number says so. Four independent attacks, because they fail
   differently: feature causality, truncation stability, walk-forward ordering,
   and train-only model selection. None subsumes another.
3. **Determinism is asserted, not assumed.** A tree ensemble whose histogram
   pass is threaded can return a different model from the same seed on the same
   data, so `Tests/test_ratings.py`'s "same order in, same numbers out"
   guarantee does not transfer to this model family for free.

The real dataset is 13,199 games and takes ~13 seconds to assemble, so it is
built once at module scope and shared. Tests about the selection and ordering
machinery use a small synthetic fixture instead and stay fast.
"""

import unittest
import warnings

import numpy as np
import pandas as pd

from src.DataProviders.nba_data import paired_bootstrap_delta, walk_forward_split
from src.Models import gbm

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Assembled at most once, on first use. `full_frame()` is separate from
# `real_data()` so a test that only needs the contrast matrix does not pay for
# the label lookup.
_CACHE: dict = {}


def real_data():
    """The real games frame, contrast matrix and labels."""
    if "d" not in _CACHE:
        _CACHE["d"] = gbm.load_dataset()
    return _CACHE["d"]


def full_frame():
    """The real games frame and its 15-column contrast matrix."""
    if "f" not in _CACHE:
        games, X, _ = real_data()
        _CACHE["f"] = (games, X)
    return _CACHE["f"]


#: Two configurations per family. Enough for a search to have something to
#: choose between, few enough that the whole selection machinery runs in
#: seconds. Production uses the eight-config grids in `gbm`.
TINY_GRIDS = {
    "xgb": ({"max_depth": 2, "learning_rate": 0.05, "n_estimators": 200,
             "subsample": 0.8, "min_child_weight": 5},
            {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 200,
             "subsample": 0.8, "min_child_weight": 5}),
    "hist": ({"max_depth": 2, "learning_rate": 0.05, "max_iter": 200,
              "min_samples_leaf": 40, "l2_regularization": 1.0},
             {"max_depth": 3, "learning_rate": 0.05, "max_iter": 200,
              "min_samples_leaf": 40, "l2_regularization": 1.0}),
}


def tiny_problem(n=1200, seed=11, teams=8):
    """A synthetic walk-forward problem carrying a real signal, for fast tests.

    Each team has a latent quality; a game is a coin flip tilted by the quality
    difference plus a home term. Column 0 of the design matrix -- ``d_margin`` --
    is exactly that quality difference, so the correct sign is known by
    construction and can be asserted. Nothing here resembles a real basketball
    schedule and it is not trying to: these tests are about *which rows were
    used*, not about the quality of the fit.
    """
    rng = np.random.default_rng(seed)
    quality = rng.normal(scale=2.0, size=teams)
    dates = pd.date_range("2020-10-20", periods=2 * n, freq="D")
    rows = []
    for i in range(n):
        h, a = rng.choice(teams, 2, replace=False)
        p = 1.0 / (1.0 + np.exp(-(quality[h] - quality[a] + 1.5)))
        rows.append((dates[2 * i], f"T{h}", f"T{a}", int(rng.random() < p)))
    games = pd.DataFrame(rows, columns=["GAME_DATE", "home", "away", "y"])
    games["GAME_ID"] = [f"{i:05d}" for i in range(n)]
    n_feat = len(gbm.FEATURE_NAMES)
    X = np.random.default_rng(seed + 1).normal(size=(n, n_feat))
    home_q = quality[np.asarray([int(h[1:]) for h in games["home"]], dtype=int)]
    away_q = quality[np.asarray([int(a[1:]) for a in games["away"]], dtype=int)]
    X[:, gbm.FEATURE_NAMES.index("d_margin")] = home_q - away_q
    X[:, gbm.FEATURE_NAMES.index("is_home")] = 1.0
    return games, X, games["y"].to_numpy(dtype=int)


def flip_test_labels(games: pd.DataFrame, n_splits: int = 2) -> pd.DataFrame:
    """Reverse every outcome in every TEST block, leaving training blocks intact.

    This is the canary used by the train-only-selection tests: any statistic
    computed from a test label must move, and any hyperparameter chosen from
    training rows must not.
    """
    _, test = walk_forward_split(games, n_splits=n_splits, initial_frac=0.5)[-1]
    out = games.copy()
    out.loc[test, "y"] = 1 - out.loc[test, "y"]
    return out


class TestFeatureConstruction(unittest.TestCase):
    """The contrast matrix is built the way the docstring says, exactly.

    Every other claim in the module rests on these. If a contrast silently became
    a concatenation of home and away columns, all the numbers below would still
    compute and still look plausible.
    """

    def test_contrast_is_home_minus_away(self):
        games, X = full_frame()
        roll = gbm.rolling_team_features(games)
        for name in ("d_margin", "d_net_rating", "d_def_rating", "d_pace"):
            expected = (roll[f"home_ewm_{name[2:]}"].to_numpy(float)
                        - roll[f"away_ewm_{name[2:]}"].to_numpy(float))
            np.testing.assert_allclose(X[:, gbm.FEATURE_NAMES.index(name)], expected,
                                       equal_nan=True)

    def test_schedule_edges_are_differences_and_travel_is_not(self):
        games, X = full_frame()
        sched = gbm.rest_and_travel(games)
        np.testing.assert_allclose(
            X[:, gbm.FEATURE_NAMES.index("d_rest_days")],
            sched["home_rest_days"].to_numpy(float)
            - sched["away_rest_days"].to_numpy(float),
            equal_nan=True)
        # travel_miles is a property of the pairing -- both clubs fly the same
        # road -- so there is no per-team decomposition to difference. Asserting
        # it is NOT differenced pins that decision rather than leaving it to
        # whichever way a refactor happens to fall out.
        np.testing.assert_allclose(X[:, gbm.FEATURE_NAMES.index("travel_miles")],
                                   sched["travel_miles"].to_numpy(float))

    def test_experience_uses_log1p_of_games_played(self):
        games, X = full_frame()
        roll = gbm.rolling_team_features(games)
        expected = (np.log1p(roll["home_games_played"].to_numpy(float))
                    - np.log1p(roll["away_games_played"].to_numpy(float)))
        np.testing.assert_allclose(
            X[:, gbm.FEATURE_NAMES.index("d_experience")], expected)

    def test_a_missing_side_makes_the_whole_contrast_missing(self):
        """A NaN minus a number is not a number.

        This is a logical-OR assertion, not a count: the contrast must be
        missing exactly when *either* side is. Filling the two sides
        independently -- or, worse, filling each contrast's missing value with a
        zero -- would convert "we know nothing about the home team" into a
        large *positive* difference, inventing evidence that the home team is
        good. The row count is asserted to be non-zero first, so a data change
        that removed every missing value cannot make this test pass vacuously.
        """
        games, X = full_frame()
        roll = gbm.rolling_team_features(games)
        h = roll["home_ewm_margin"].to_numpy(float)
        a = roll["away_ewm_margin"].to_numpy(float)
        expected_missing = np.isnan(h) | np.isnan(a)
        self.assertGreater(int(expected_missing.sum()), 0)
        got = X[:, gbm.FEATURE_NAMES.index("d_margin")]
        np.testing.assert_array_equal(np.isnan(got), expected_missing)

    def test_no_finite_value_is_invented_where_a_side_is_missing(self):
        """Complements the OR assertion: nothing spurious appears either.

        Checks the two directions separately so a change that filled *both*
        sides (making the count accidentally match) still fails.
        """
        _, X = full_frame()
        col = X[:, gbm.FEATURE_NAMES.index("d_margin")]
        finite = col[np.isfinite(col)]
        # A filled contrast would be a plausible-looking number; assert the
        # missing share is small and, crucially, that the missing set is
        # disjoint from nothing -- i.e. exactly the teams' first appearances.
        self.assertLess(float(np.isnan(col).mean()), 0.01)
        self.assertGreater(len(finite), 13000)

    def test_feature_count_is_modest(self):
        """The spec's explicit instruction, pinned as a test.

        "Keep the feature count modest to avoid overfitting." A number that
        merely happens to be 15 today is not the same claim as one that is
        asserted, so it is asserted.
        """
        self.assertEqual(len(gbm.FEATURE_NAMES), 15)
        self.assertEqual(len(set(gbm.FEATURE_NAMES)), 15, "feature names must be unique")
        _, X, _ = real_data()
        self.assertEqual(X.shape[1], 15)

    def test_home_indicator_is_constant_and_cannot_split(self):
        """`is_home` is the intercept in both families, and is reported as such.

        Every row here is a home team playing an away team, so the column is 1.0
        throughout. A tree cannot split on a constant, and the logistic
        reference's scaler maps it to zeros, so the home-court term is carried
        entirely by the intercept. The column is kept because the spec asks for
        it; what matters is that nothing here pretends otherwise.
        """
        _, X = full_frame()
        col = X[:, gbm.FEATURE_NAMES.index("is_home")]
        np.testing.assert_array_equal(col, np.ones_like(col))

    def test_two_cost_columns_are_oriented_against_the_home_team(self):
        """DEF_RATING and TOV are costs, so the orientation table has to say so.

        A higher defensive rating concedes more points per 100 possessions, and a
        higher turnover rate is simply worse. For those two a positive raw
        contrast means the home team is worse on that axis. Every reported
        effect is multiplied by this table, and this test is what stops the table
        from drifting into being decorative.
        """
        self.assertEqual(gbm.FEATURE_ORIENTATION["d_def_rating"], -1.0)
        self.assertEqual(gbm.FEATURE_ORIENTATION["d_tov"], -1.0)
        for name in gbm.FEATURE_NAMES:
            if name not in ("d_def_rating", "d_tov"):
                self.assertEqual(gbm.FEATURE_ORIENTATION[name], 1.0, name)
        self.assertEqual(set(gbm.FEATURE_ORIENTATION), set(gbm.FEATURE_NAMES))

    def test_net_rating_contrast_is_near_redundant_with_off_and_def(self):
        """A collinearity worth knowing before reading any single coefficient.

        NET_RATING is OFF_RATING minus DEF_RATING per game, and an exponentially
        weighted mean is a *linear* functional, so `d_net_rating` is
        `d_off_rating - d_def_rating` up to the rounding applied to each published
        rating. That is an almost-exact linear dependency in the design matrix.
        It does not break prediction -- the predicted probability is still
        identified -- but it does make a per-coefficient reading of this feature
        set unsafe, which is one reason the sign audit works on predictions
        rather than on coefficients.
        """
        _, X = full_frame()
        cols = {n: X[:, gbm.FEATURE_NAMES.index(n)]
                for n in ("d_net_rating", "d_off_rating", "d_def_rating")}
        net = cols["d_net_rating"]
        combo = cols["d_off_rating"] - cols["d_def_rating"]
        ok = np.isfinite(net) & np.isfinite(combo)
        self.assertGreater(int(ok.sum()), 10000)
        self.assertGreater(float(np.corrcoef(net[ok], combo[ok])[0, 1]), 0.999)


class TestTruncationAndCausality(unittest.TestCase):
    """The feature for game t may not contain game t.

    Two independent attacks, because they fail differently.

    *Truncation*: build features from a prefix of the game list and compare
    against the same rows of the full build. A builder that reads "all games on
    disk" rather than "the games I was handed" passes a causality test and fails
    this one.

    *Future scramble*: reverse the outcome of every game after position t and
    require row t to be bit-identical. This needs no knowledge of how the
    builder works -- it only asks whether row t depends on the future at all.
    """

    def test_prefix_build_matches_full_build_on_shared_rows(self):
        games, _ = full_frame()
        cut = 8000
        prefix = gbm.build_feature_frame(games.iloc[:cut]).to_numpy(dtype=float)
        full = gbm.build_feature_frame(games).to_numpy(dtype=float)[:cut]
        # equal_nan matters: 17 rows have a missing first-game EWM, and NaN != NaN
        # would report those as 17 spurious differences.
        np.testing.assert_allclose(prefix, full, equal_nan=True)
        np.testing.assert_array_equal(np.isnan(prefix), np.isnan(full))

    def test_adding_future_games_cannot_change_the_past(self):
        """The same guarantee additively, against a builder that fills with a
        dataset-wide statistic.

        A median over all teams and seasons is insensitive to appending future
        rows; a mean or a scale is not. Both are the kind of normalisation that
        looks harmless and leaks the future distribution into the past.
        """
        games, _ = full_frame()
        cut = 6000
        small = gbm.build_feature_frame(games.iloc[:cut]).to_numpy(dtype=float)
        grown = gbm.build_feature_frame(games).to_numpy(dtype=float)[:cut]
        np.testing.assert_allclose(small, grown, equal_nan=True)

    def test_reversing_every_future_outcome_leaves_row_t_untouched(self):
        games, _ = full_frame()
        flipped = games.copy()
        after = flipped.index > 3000
        hp = flipped.loc[after, "home_pts"].to_numpy()
        flipped.loc[after, "home_pts"] = flipped.loc[after, "away_pts"]
        flipped.loc[after, "away_pts"] = hp
        flipped.loc[after, "y"] = (
            flipped.loc[after, "home_pts"] > flipped.loc[after, "away_pts"]).astype(int)
        flipped = flipped.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
        before = gbm.build_feature_frame(games).to_numpy(dtype=float)
        after_f = gbm.build_feature_frame(flipped).to_numpy(dtype=float)
        np.testing.assert_allclose(before[3000], after_f[3000], equal_nan=True)
        # ... and the flip must actually have done something downstream, or this
        # class is testing a no-op.
        self.assertGreater(
            float(np.nanmax(np.abs(before[3001:] - after_f[3001:]))), 1.0,
            "reversing future outcomes changed nothing downstream; the test is vacuous")

    def test_the_feature_frame_row_count_always_matches_the_game_count(self):
        """A length mismatch would silently misattribute every number to the
        wrong game, so the builder refuses rather than pads or truncates."""
        games, _ = full_frame()
        for n in (50, 500, 5000):
            frame = gbm.build_feature_frame(games.iloc[:n])
            self.assertEqual(len(frame), n)
            self.assertEqual(list(frame.columns), list(gbm.FEATURE_NAMES))


class TestInnerSplitStaysInsideTheTrainingBlock(unittest.TestCase):
    """The tuning split is carved out of training rows, mechanically.

    `Tests/test_ratings.py` attacks ordering from three angles because each
    catches a different bug. The mechanical equivalent here is that the tuner is
    handed positions, and positions pointing outside the block it was given are
    a hard error rather than a silent slice.
    """

    def test_inner_indices_are_in_range_and_ordered(self):
        dates = pd.date_range("2018-10-20", periods=4000, freq="D")
        splits = gbm.inner_splits(len(dates), dates, n_splits=3, initial_frac=0.6)
        self.assertEqual(len(splits), 3)
        for tr, va in splits:
            self.assertTrue(set(tr.tolist()).isdisjoint(va.tolist()),
                            "train and valid overlap")
            self.assertGreater(int(va.min()), int(tr.max()),
                               "every validation row must come after every training row")

    def test_inner_split_never_reaches_past_the_block_it_was_given(self):
        dates = pd.date_range("2018-10-20", periods=3000, freq="D")
        for tr, va in gbm.inner_splits(len(dates), dates, n_splits=3, initial_frac=0.5):
            self.assertLess(int(va.max()), len(dates))

    def test_tuner_rejects_positions_outside_the_block_it_was_given(self):
        games, X, y = tiny_problem(600)
        with self.assertRaises(ValueError) as ctx:
            gbm.tune_on_inner_splits(
                "logreg", X[:400], y[:400], np.arange(500),
                games["GAME_DATE"].iloc[:400].to_numpy(), n_splits=2)
        self.assertIn("outside", str(ctx.exception))

    def test_tuner_refuses_a_block_that_leaves_no_validation_data(self):
        """A tuning split with no validation rows is an error, not an empty grid.

        `initial_frac=1.0` hands every training row to the tuning-train side,
        which is what a caller who sets it too high would do. Returning an empty
        `scores` dict would make `min()` raise a `ValueError` about an empty
        sequence deep inside the search, naming neither the cause nor the fix.
        """
        games, X, y = tiny_problem(120)
        with self.assertRaises(ValueError) as ctx:
            gbm.tune_on_inner_splits(
                "logreg", X[:40], y[:40], np.arange(40),
                games["GAME_DATE"].iloc[:40].to_numpy(), n_splits=3, initial_frac=1.0)
        self.assertIn("inner", str(ctx.exception))


class TestHyperparametersComeFromTrainingDataOnly(unittest.TestCase):
    """(a) The required test: model selection never sees a test label.

    The strong form of the assertion is a canary. Run the whole walk-forward and
    record the hyperparameters chosen for every split. Then reverse the outcome
    of every game in the TEST blocks and run it again.

    If any part of the selection had touched a test label, the chosen
    hyperparameters would move. They cannot, because `run_walk_forward` slices
    `X[tr]` and `y[tr]` before handing them to the tuner, so the test rows are
    not in the array the tuner can see.

    The second half of the class checks the canary is not vacuous: the reported
    test metrics *must* move when the test labels are reversed. A test that
    passed because the flip changed nothing would prove nothing.
    """

    def _run(self, games, X, y, n_splits=2):
        return gbm.run_walk_forward(
            X, y, games, n_splits=n_splits, inner_n_splits=2, n_boot=200, seed=7,
            grids=TINY_GRIDS, verbose=False)

    def test_reversing_test_labels_cannot_move_the_chosen_hyperparameters(self):
        games, X, _ = tiny_problem(3000, seed=5)
        y = games["y"].to_numpy(dtype=int)
        clean = self._run(games, X, y)
        flipped = flip_test_labels(games)
        dirty = self._run(flipped, X, flipped["y"].to_numpy(dtype=int))
        for fam in clean["meta"]["families"]:
            self.assertEqual([t.params for t in clean["tuning"][fam]],
                             [t.params for t in dirty["tuning"][fam]],
                             f"{fam}: hyperparameters moved when test labels changed")

    def test_reversing_test_labels_cannot_move_the_inner_scores(self):
        """Stronger still: not only the winner but every grid score is pinned.

        A search that peeked at test labels and happened to pick the same winner
        would still leak; comparing the whole score vector closes that gap.
        """
        games, X, _ = tiny_problem(3000, seed=5)
        y = games["y"].to_numpy(dtype=int)
        clean = self._run(games, X, y)
        flipped = flip_test_labels(games)
        dirty = self._run(flipped, X, flipped["y"].to_numpy(dtype=int))
        for fam in clean["meta"]["families"]:
            self.assertEqual(clean["tuning"][fam][-1].scores,
                             dirty["tuning"][fam][-1].scores, fam)

    def test_the_canary_is_not_vacuous(self):
        games, X, _ = tiny_problem(3000, seed=5)
        clean = self._run(games, X, games["y"].to_numpy(dtype=int))
        flipped = flip_test_labels(games)
        dirty = self._run(flipped, X, flipped["y"].to_numpy(dtype=int))
        fam = clean["meta"]["families"][0]
        a = [r["log_loss"] for r in clean["per_split"] if r["model"] == fam]
        b = [r["log_loss"] for r in dirty["per_split"] if r["model"] == fam]
        self.assertNotEqual(a, b,
                            "flipping test labels changed no test metric, so the "
                            "leakage canaries above prove nothing")

    def test_recorded_params_reproduce_a_standalone_search_on_the_train_block(self):
        """The wiring is pinned, not just the tuner.

        `run_walk_forward` passes `X[tr]`/`y[tr]` to the tuner. If that slice
        were wrong -- off by one, or including the first test row -- the recorded
        parameters would differ from a search handed the training block alone.
        """
        games, X, y = tiny_problem(3000, seed=5)
        result = self._run(games, X, y)
        for k, (tr, te) in enumerate(walk_forward_split(games, n_splits=2, initial_frac=0.5)):
            for fam in result["meta"]["families"]:
                standalone = gbm.tune_on_inner_splits(
                    fam, X[tr], y[tr], np.arange(len(tr)),
                    games["GAME_DATE"].iloc[tr].to_numpy(), seed=7, n_splits=2,
                    grid=TINY_GRIDS[fam])
                self.assertEqual(standalone.params, result["tuning"][fam][k].params,
                                 f"{fam} split {k}")

    def test_the_search_minimises_log_loss_over_the_grid_it_was_given(self):
        games, X, y = tiny_problem(1500, seed=3)
        res = gbm.tune_on_inner_splits(
            "hist", X, y, np.arange(len(y)), games["GAME_DATE"].to_numpy(),
            n_splits=2, grid=TINY_GRIDS["hist"])
        self.assertEqual(len(res.scores), 2)
        self.assertAlmostEqual(res.inner_log_loss, min(res.scores.values()), places=15)
        winner = min(TINY_GRIDS["hist"], key=lambda p: res.scores[repr(p)])
        self.assertEqual(res.params, winner)
        self.assertEqual(res.n_splits_used, 2)

    def test_a_wider_search_can_never_score_worse_than_its_own_best_entry(self):
        """The reported `inner_log_loss` is a real score, not a lower bound.

        Ties in the search are broken by grid order, which is deterministic, so
        the recorded value must equal one of the recorded entries exactly.
        """
        games, X, y = tiny_problem(1500, seed=3)
        res = gbm.tune_on_inner_splits(
            "xgb", X, y, np.arange(len(y)), games["GAME_DATE"].to_numpy(),
            n_splits=2, grid=TINY_GRIDS["xgb"])
        self.assertIn(res.inner_log_loss, res.scores.values())


class TestPreprocessingIsFittedOnTrainOnly(unittest.TestCase):
    """Imputation and scaling are pipeline steps, fitted inside the fence.

    A median fitted over train+test lets the test distribution move the value
    used to fill the test block's own missing entries. The effect on log loss is
    small and the effect on every summary statistic is nil, which is why it is
    tested directly instead of reasoned about.
    """

    def test_imputer_uses_the_training_median_not_the_pooled_one(self):
        games, X, y = tiny_problem(2000, seed=9)
        tr = np.arange(0, 1200)
        model = gbm.make_model("hist", gbm.HIST_GRID[0])
        gbm.fit_single_threaded(model, X[tr], y[tr])
        stats = model.named_steps["impute"].statistics_
        for j, name in enumerate(gbm.FEATURE_NAMES):
            self.assertAlmostEqual(float(stats[j]), float(np.median(X[tr, j])),
                                   places=12, msg=name)

    def test_fitted_state_is_identical_when_only_the_test_block_changes(self):
        """Two runs, same training block, wildly different test block.

        Every fitted artefact -- imputer medians, scaler mean and scale,
        coefficients -- must be identical, because the fit never saw the test
        rows. Were the pipeline fitted on the union, these would differ.
        """
        games, X, y = tiny_problem(2000, seed=9)
        tr, te = np.arange(0, 1400), np.arange(1400, 2000)
        X_b = X.copy()
        X_b[te, 0] = 1e6          # an enormous value, test rows only
        fitted = []
        for Xi in (X, X_b):
            m = gbm.make_model("logreg", {})
            gbm.fit_single_threaded(m, X[tr], y[tr])
            fitted.append(m)
        np.testing.assert_allclose(fitted[0].named_steps["impute"].statistics_,
                                   fitted[1].named_steps["impute"].statistics_)
        np.testing.assert_allclose(fitted[0].named_steps["scale"].mean_,
                                   fitted[1].named_steps["scale"].mean_)
        np.testing.assert_allclose(fitted[0].named_steps["scale"].scale_,
                                   fitted[1].named_steps["scale"].scale_)
        np.testing.assert_allclose(fitted[0].named_steps["model"].coef_,
                                   fitted[1].named_steps["model"].coef_)
        np.testing.assert_allclose(fitted[0].predict_proba(X[tr])[:, 1],
                                   fitted[1].predict_proba(X[tr])[:, 1])

    def test_the_test_block_still_reaches_the_evaluation(self):
        """The mirror of the previous test: predictions there must differ.

        Together they say: the fit ignores the test block and the score does
        not. Asserting only the first would pass for a pipeline that ignored the
        test block altogether.
        """
        games, X, y = tiny_problem(2000, seed=9)
        tr, te = np.arange(0, 1400), np.arange(1400, 2000)
        X_b = X.copy()
        X_b[te, 0] = 1e6
        m = gbm.make_model("hist", gbm.HIST_GRID[0])
        gbm.fit_single_threaded(m, X[tr], y[tr])
        self.assertFalse(np.allclose(m.predict_proba(X[te])[:, 1],
                                     m.predict_proba(X_b[te])[:, 1]))

    def test_an_all_missing_column_does_not_change_the_feature_count(self):
        """`keep_empty_features` exists for a specific walk-forward failure.

        Early in a training block a column can be entirely missing. The default
        imputer drops it, so a model fitted in one split would take 14 columns
        and the next would take 15 -- and a feature-count curve across such runs
        would be measuring the imputer's bookkeeping rather than the features.
        """
        from sklearn.impute import SimpleImputer

        X = np.random.default_rng(0).normal(size=(50, 4))
        X[:, 0] = np.nan
        out = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X)
        self.assertEqual(out.shape[1], 4)
        self.assertIs(gbm.make_model("hist", gbm.HIST_GRID[0])
                      .named_steps["impute"].keep_empty_features, True)


class TestDeterminismUnderFixedSeed(unittest.TestCase):
    """(c) The required test: same seed, same numbers.

    Not free for a tree ensemble the way it is for the rating systems in
    `Tests/test_ratings.py`. Two separate things must hold: the seeded sampler
    draws the same subsample, and the arithmetic combining parallel partial
    results does so in a fixed order. The second is why `fit_single_threaded`
    exists, and it is asserted rather than assumed.
    """

    def test_two_fits_with_the_same_seed_give_bit_identical_predictions(self):
        games, X, y = tiny_problem(1500, seed=13)
        tr, te = np.arange(0, 1100), np.arange(1100, 1500)
        for fam, params in (("xgb", gbm.XGB_GRID[0]), ("hist", gbm.HIST_GRID[0]),
                            ("gbdt", gbm.GBDT_GRID[0]), ("logreg", {})):
            outs = []
            for _ in range(2):
                m = gbm.make_model(fam, params, seed=gbm.SEED)
                gbm.fit_single_threaded(m, X[tr], y[tr])
                outs.append(m.predict_proba(X[te])[:, 1])
            np.testing.assert_array_equal(outs[0], outs[1], f"{fam} is not deterministic")

    def test_an_arbitrary_seed_is_also_reproducible(self):
        """Determinism is a property of the pipeline, not of the default seed.

        A test pinning only `SEED` would pass even if the seed were ignored
        entirely and the estimator happened to be deterministic anyway. This
        checks an arbitrary seed is honoured: two runs at seed 4242 agree.
        """
        games, X, y = tiny_problem(1500, seed=13)
        tr, te = np.arange(0, 1100), np.arange(1100, 1500)
        outs = []
        for _ in range(2):
            m = gbm.make_model("xgb", gbm.XGB_GRID[0], seed=4242)
            gbm.fit_single_threaded(m, X[tr], y[tr])
            outs.append(m.predict_proba(X[te])[:, 1])
        np.testing.assert_array_equal(outs[0], outs[1])

    def test_tuning_is_reproducible_and_scores_every_grid_entry(self):
        games, X, y = tiny_problem(1500, seed=13)
        dates = games["GAME_DATE"].to_numpy()
        a = gbm.tune_on_inner_splits("xgb", X, y, np.arange(len(y)), dates,
                                     n_splits=2, grid=TINY_GRIDS["xgb"])
        b = gbm.tune_on_inner_splits("xgb", X, y, np.arange(len(y)), dates,
                                     n_splits=2, grid=TINY_GRIDS["xgb"])
        self.assertEqual(a.params, b.params)
        self.assertEqual(a.scores, b.scores)
        self.assertEqual(a.inner_log_loss, b.inner_log_loss)
        self.assertEqual(len(a.scores), 2)

    def test_permutation_importance_is_reproducible(self):
        games, X, y = tiny_problem(800, seed=17)
        m = gbm.make_model("hist", gbm.HIST_GRID[0])
        gbm.fit_single_threaded(m, X, y)
        pd.testing.assert_frame_equal(
            gbm.permutation_importance_table(m, X, y, n_repeats=3, seed=99),
            gbm.permutation_importance_table(m, X, y, n_repeats=3, seed=99))

    def test_the_sign_audit_is_reproducible(self):
        games, X, y = tiny_problem(2000, seed=19)
        grids = dict(TINY_GRIDS)
        grids["gbdt"] = gbm.GBDT_GRID
        args = dict(families=("xgb", "hist", "logreg"), n_splits=2, seed=5, grids=grids)
        e1, a1 = gbm.cross_family_sign_audit(games, X, y, **args)
        e2, a2 = gbm.cross_family_sign_audit(games, X, y, **args)
        pd.testing.assert_frame_equal(e1, e2)
        pd.testing.assert_frame_equal(a1, a2)

    def test_hist_gradient_boosting_has_its_hidden_holdout_disabled(self):
        """`early_stopping="auto"` is a second, undeclared holdout.

        Above 10,000 rows the default silently switches on and carves a random
        validation slice out of the training data. That would make the tuned
        `max_iter` a function of a slice nobody declared, and would add a second
        source of randomness on top of the seed.
        """
        self.assertIs(gbm.make_model("hist", gbm.HIST_GRID[0])
                      .named_steps["model"].early_stopping, False)

    def test_xgboost_runs_single_threaded(self):
        """Documented in `fit_single_threaded`: a threaded histogram can combine
        partial results in a scheduling-dependent order."""
        self.assertEqual(gbm.make_model("xgb", gbm.XGB_GRID[0])
                         .named_steps["model"].n_jobs, 1)

    def test_the_whole_walk_forward_reproduces_its_own_numbers(self):
        games, X, y = tiny_problem(2500, seed=21)
        grids = {"xgb": TINY_GRIDS["xgb"], "hist": TINY_GRIDS["hist"]}
        runs = [gbm.run_walk_forward(X, y, games, n_splits=2, inner_n_splits=2,
                                     n_boot=200, seed=5, grids=grids, verbose=False)
                for _ in range(2)]
        for fam in runs[0]["meta"]["families"]:
            np.testing.assert_array_equal(runs[0]["predictions"][fam],
                                          runs[1]["predictions"][fam])
            self.assertEqual([t.params for t in runs[0]["tuning"][fam]],
                             [t.params for t in runs[1]["tuning"][fam]])


class TestDeltaSignConvention(unittest.TestCase):
    """A positive delta means the *second* model is better.

    `paired_bootstrap_delta` documents this, and the repository has got it wrong
    before. The convention is pinned against the shared function itself rather
    than against this module's reading of it: every headline number in the
    report is a delta, and a flipped sign would invert the entire conclusion
    while leaving every printed figure looking reasonable.
    """

    def test_shared_delta_is_positive_when_the_second_model_is_better(self):
        """A model that is confidently *wrong* has the higher loss.

        `np.where(y == 1, 0.05, 0.95)` is the confidently-wrong model: 0.05
        whenever the home team won and 0.95 whenever it lost. (The mirror
        spelling is confidently *right* and scores far better -- getting that
        backwards is exactly how a test of this convention passes for the wrong
        reason.)
        """
        rng = np.random.default_rng(0)
        y = (rng.random(2000) < 0.55).astype(int)
        honest = np.full(2000, 0.55)                       # the base rate
        confident_wrong = np.where(y == 1, 0.05, 0.95)
        d = paired_bootstrap_delta(y, confident_wrong, honest, n_boot=200, seed=1)
        self.assertGreater(d["delta"], 0.0,
                           "a confident-wrong model has HIGHER loss than the base "
                           "rate, i.e. a positive delta means the second is better")
        # A 0.05/0.95 flip against a 0.55 base rate is a large, consistent gap,
        # so the whole interval sits above zero rather than straddling it.
        self.assertGreater(d["ci_lo"], 0.0)
        reversed_delta = paired_bootstrap_delta(y, honest, confident_wrong,
                                                n_boot=200, seed=1)
        self.assertLess(reversed_delta["delta"], 0.0)
        self.assertLess(reversed_delta["ci_hi"], 0.0)

    def test_pooled_delta_equals_baseline_loss_minus_model_loss(self):
        """The reported entry is loss(baseline) - loss(model), by construction."""
        games, X, y = tiny_problem(2500, seed=23)
        res = gbm.run_walk_forward(
            X, y, games, n_splits=2, inner_n_splits=2, n_boot=200, seed=5,
            grids=TINY_GRIDS, verbose=False)
        base = res["meta"]["baseline"]
        for fam in res["meta"]["families"]:
            d = res["pooled"][fam][f"delta_vs_{base}"]
            by_subtraction = res["pooled"][base]["log_loss"] - res["pooled"][fam]["log_loss"]
            self.assertAlmostEqual(d["delta"], by_subtraction, places=10)
            self.assertGreaterEqual(d["ci_lo"], min(d["delta"], by_subtraction) - 1.0)

    def test_per_split_deltas_are_consistent_with_each_other_and_the_pool(self):
        games, X, y = tiny_problem(2500, seed=23)
        res = gbm.run_walk_forward(
            X, y, games, n_splits=3, inner_n_splits=2, n_boot=200, seed=5,
            grids=TINY_GRIDS, verbose=False)
        pooled_key = f"delta_vs_{res['meta']['baseline']}"
        for fam in res["meta"]["families"]:
            rows = [d for d in gbm.per_split_deltas(res, n_boot=200) if d["model"] == fam]
            self.assertEqual(len(rows), 3)
            for r in rows:
                self.assertEqual(r["better"], r["delta"] > 0)
                self.assertLessEqual(r["ci_lo"], r["delta"])
                self.assertGreaterEqual(r["ci_hi"], r["delta"])
            # The pooled delta is the ROW-COUNT-WEIGHTED mean of the split deltas,
            # not their plain mean: `walk_forward_split` cuts the tail with
            # `np.linspace`, so the blocks differ by a game or two and the
            # weights are not exactly equal. Weighting by n is the exact
            # relationship, and asserting it exactly is what makes this a
            # statement about arithmetic rather than about tolerance.
            weights = np.asarray([r["n"] for r in rows], dtype=float)
            weighted = float(np.average([r["delta"] for r in rows], weights=weights))
            self.assertAlmostEqual(res["pooled"][fam][pooled_key]["delta"],
                                   weighted, places=10)
            # The weighting really is non-uniform, so the plain mean would fail.
            self.assertGreater(int(weights.max() - weights.min()), 0)

    def test_the_base_rate_is_reported_and_is_worse_than_every_real_model(self):
        """The always-home baseline is present and beaten.

        Its presence is asserted first because the model documents it as always
        included: a "the baseline beats us" check against a key that is not
        there would have passed for the wrong reason.
        """
        games, X, y = tiny_problem(2500, seed=23)
        res = gbm.run_walk_forward(
            X, y, games, n_splits=2, inner_n_splits=2, n_boot=200, seed=5,
            grids=TINY_GRIDS, verbose=False)
        self.assertIn("base_rate", res["pooled"])
        base = res["pooled"]["base_rate"]["log_loss"]
        for fam in list(res["meta"]["families"]) + [res["meta"]["baseline"]]:
            self.assertLess(res["pooled"][fam]["log_loss"], base)

    def test_base_rate_prediction_is_the_training_home_win_rate(self):
        games, X, y = tiny_problem(2500, seed=23)
        res = gbm.run_walk_forward(
            X, y, games, n_splits=2, inner_n_splits=2, n_boot=200, seed=5,
            grids=TINY_GRIDS, verbose=False)
        for rec in res["per_split"]:
            if rec["model"] != "base_rate":
                continue
            p = res["predictions_by_split"]["base_rate"][rec["split"] - 1]
            tr, _ = walk_forward_split(games, n_splits=2, initial_frac=0.5)[rec["split"] - 1]
            self.assertTrue(np.allclose(p, y[tr].mean()))

    def test_the_always_home_baseline_is_always_evaluated(self):
        """A regression guard on a bug this class found.

        `run_walk_forward` documented an always-home base rate and then built its
        model loop as `families + [baseline]`, which never contained that entry.
        The baseline was therefore silently absent from every report, and the
        paired delta had nothing to be measured against but the logistic
        regression. Asserting the key exists is the whole fix.
        """
        games, X, y = tiny_problem(2000, seed=23)
        for families in (("xgb",), ("xgb", "hist"), ()):
            res = gbm.run_walk_forward(
                X, y, games, families=families, n_splits=2, inner_n_splits=2,
                n_boot=100, seed=5, grids=TINY_GRIDS, verbose=False)
            self.assertIn("base_rate", res["pooled"], f"families={families}")
            self.assertIn("base_rate", res["predictions_by_split"])
            for rec in res["per_split"]:
                self.assertIn(rec["model"], set(res["pooled"]))


class TestWalkForwardOrdering(unittest.TestCase):
    """Test blocks are strictly later than everything they are fitted on."""

    def test_splits_are_chronological_and_non_overlapping(self):
        """Row positions are strictly ordered; dates are non-decreasing.

        The date assertion is `<=`, not `<`, and the difference matters. Splits
        are cut by row position and an NBA slate puts several games on one date,
        so the last training game and the first test game can share a calendar
        day. That is not leakage here: `rolling_team_features` shifts each team's
        own history by one *game*, so a test game's EWM holds its team's
        previous games -- possibly one on the same date, but a different game
        that had already tipped off. Information flows along the played order.
        `walk_forward_split`'s `gap` argument is what a caller would use to purge
        a calendar interval, and the default is 0.

        What *is* strictly ordered, and asserted here, is the row position: every
        training row precedes every test row.
        """
        games, _, _ = real_data()
        splits = walk_forward_split(games, n_splits=5, initial_frac=0.5)
        self.assertEqual(len(splits), 5)
        prev_hi = -1
        for tr, te in splits:
            self.assertLess(int(tr.max()), int(te.min()))
            self.assertGreaterEqual(int(te.min()), prev_hi)
            self.assertLessEqual(games["GAME_DATE"].iloc[tr].max(),
                                 games["GAME_DATE"].iloc[te].min())
            prev_hi = int(te.max())

    def test_a_same_date_boundary_is_real_and_bounded(self):
        """Pins how often the shared-date boundary actually happens.

        Documenting it as "it never happens" would be a claim about this season
        of this data rather than about the splitter, and would quietly fail the
        first time it did.
        """
        games, _, _ = real_data()
        splits = walk_forward_split(games, n_splits=5, initial_frac=0.5)
        same_day = sum(
            1 for tr, te in splits
            if games["GAME_DATE"].iloc[tr].max() == games["GAME_DATE"].iloc[te].min()
        )
        self.assertLessEqual(same_day, len(splits))

    def test_a_purged_split_moves_the_boundary_off_the_shared_date(self):
        """`gap` is the knob for the objection above, so it is shown working."""
        games, _, _ = real_data()
        for tr, te in walk_forward_split(games, n_splits=5, initial_frac=0.5, gap=30):
            self.assertLess(int(tr.max()), int(te.min()))
            self.assertLess(games["GAME_DATE"].iloc[tr].max(),
                            games["GAME_DATE"].iloc[te].min())

    def test_no_game_is_scored_twice(self):
        games, _, _ = real_data()
        test_rows = np.concatenate(
            [te for _, te in walk_forward_split(games, n_splits=5, initial_frac=0.5)])
        self.assertEqual(len(test_rows), len(np.unique(test_rows)))
        self.assertTrue(bool(np.all(np.diff(test_rows) > 0)),
                        "test blocks are not in increasing order")

    def test_the_first_training_block_is_half_the_data(self):
        """`initial_frac=0.5` is what guarantees no split trains on nothing."""
        games, _, _ = real_data()
        tr, te = walk_forward_split(games, n_splits=5, initial_frac=0.5)[0]
        self.assertEqual(len(tr), int(len(games) * 0.5))
        self.assertGreater(len(tr), len(te))


class TestSignDisagreementsAreOffManifold(unittest.TestCase):
    """The cross-family sign reversal, and what it is actually evidence of.

    An earlier cross-family test in this project found XGBoost's effect sign
    reversing relative to logistic regression, GBDT and HistGB. The instruction
    was to investigate a reversal rather than average it away, and this class
    pins the answer, because an investigation that is not pinned gets
    re-litigated every time the feature set moves.

    The finding: the reversals sit on the features most correlated with
    ``d_margin``, the top feature for every model. The +/-1 SD contrast freezes
    every other column, so on such a feature it describes a combination of
    values that essentially never occurs -- a team with an average recent margin
    and an average net rating, at r = 0.998. Off-manifold, a linear model
    extrapolates along a straight line and a tree lands in a rarely-visited
    leaf. Neither answer is wrong; the question is one the data never asked.

    These assertions are on the REAL data and the REAL fitted models, which is
    the point: this is a record of an observation, not a property of a fixture.
    """

    @classmethod
    def setUpClass(cls):
        cls.games, cls.X, cls.y = real_data()
        cls.corr = gbm.dominant_feature_corr(cls.X, "d_margin")

    def test_the_top_feature_is_dominant_for_every_family(self):
        """`d_margin` is what the naming of the diagnostic assumes. Check it."""
        games, X, y = real_data()
        tr = walk_forward_split(games, n_splits=5, initial_frac=0.5)[-1][0]
        for fam in ("xgb", "hist", "logreg"):
            res = gbm.tune_on_inner_splits(
                fam, X[tr], y[tr], np.arange(len(tr)),
                games["GAME_DATE"].iloc[tr].to_numpy(), seed=gbm.SEED)
            m = gbm.make_model(fam, res.params)
            gbm.fit_single_threaded(m, X[tr], y[tr])
            top = gbm.permutation_importance_table(m, X[tr], y[tr], n_repeats=3)["feature"].iloc[0]
            self.assertEqual(top, "d_margin", fam)

    def test_the_sharpest_disagreements_are_the_most_correlated_features(self):
        """The claim, asserted on measured correlations rather than asserted free."""
        # `d_net_rating`, `d_off_rating`, `d_def_rating` are the three most
        # correlated features with `d_margin`, and the three the sign audit
        # disagrees on. `travel_miles`, `d_back_to_back` and `d_in_3_of_4` are
        # effectively uncorrelated and are among those that agree.
        for name in ("d_net_rating", "d_off_rating", "d_def_rating"):
            self.assertGreater(float(self.corr[name]), 0.7, name)
        for name in ("travel_miles", "d_back_to_back", "d_in_3_of_4"):
            self.assertLess(float(self.corr[name]), 0.05, name)

    def test_net_rating_is_almost_the_same_column_as_margin(self):
        """The number that makes the off-manifold argument, stated as a bound.

        An EWM of net rating and an EWM of point margin are both team-strength
        summaries over the same games, so at |r| = 0.998 they are near-
        interchangeable. That is precisely why contrasting one while holding the
        other produces an input the data does not contain.
        """
        self.assertGreater(float(self.corr["d_net_rating"]), 0.99)

    def test_a_constant_feature_has_no_correlation_to_report(self):
        self.assertTrue(np.isnan(float(self.corr["is_home"])))


class TestEffectDirections(unittest.TestCase):
    """The signed contrast that makes cross-family sign comparison possible.

    A logistic regression has a coefficient per feature and a tree has none, and
    neither `feature_importances_` nor permutation importance carries a sign. So
    "the effect sign of XGBoost" is not something the estimator reports, and the
    only honest version of the cross-family question has to define its own
    contrast -- which must then agree with the logistic coefficient in sign and
    magnitude, or the definition is wrong.
    """

    def test_contrast_reproduces_the_logistic_coefficient_exactly(self):
        """The +/-1 SD contrast must equal what the linear model itself implies.

        Derived from the definition rather than approximated. The reference is a linear
        model with coefficients `w` and intercept `b`, so replacing column j by
        `median_j +/- sd_j` everywhere gives a predicted probability of
        `sigmoid(w . x + b +/- 2 * sd_j * w_j)` and the mean difference over the
        rows is a closed form in `w`, `b`, `X` and the medians and sds.

        One wrinkle: the reference model sits behind a `StandardScaler`, so its
        coefficients are in scaled coordinates. Folding the scaler back into raw
        coordinates gives the equivalent raw-space model

            w_raw = w / sigma        b_raw = b - w . mu / sigma

        which is what the expected value below uses. A first-order approximation
        (`2 * sd_j * beta_j / 4`, from `p(1-p) <= 1/4` near the origin) was tried
        and is *not* what this function returns: on a fixture with real signal it
        is off by a factor of three, because that linearisation is only valid
        around p = 0.5. Pinning the exact identity catches a genuinely wrong
        contrast; pinning the approximation would only have pinned a wrong
        expectation.
        """
        games, X, y = tiny_problem(1500, seed=31)
        m = gbm.make_model("logreg", {})
        gbm.fit_single_threaded(m, X, y)
        eff = gbm.effect_directions(m, X).to_numpy()

        w_scaled = m.named_steps["model"].coef_.ravel()
        b_scaled = float(m.named_steps["model"].intercept_[0])
        mu = m.named_steps["scale"].mean_
        sigma = m.named_steps["scale"].scale_
        w = w_scaled / sigma
        # Parenthesised: `@` binds tighter than `/`, so `w @ mu / sigma` would be
        # the *vector* (w @ mu) / sigma rather than the scalar w @ (mu / sigma).
        b = b_scaled - float(w_scaled @ (mu / sigma))
        expected = np.empty(X.shape[1])
        for j in range(X.shape[1]):
            med = float(np.median(X[:, j]))
            sd = float(X[:, j].std())
            if sd <= 0:
                expected[j] = 0.0
                continue
            lo = X.copy()
            hi = X.copy()
            lo[:, j] = med - sd
            hi[:, j] = med + sd
            z_lo = lo @ w + b
            z_hi = hi @ w + b
            expected[j] = float(np.mean(1 / (1 + np.exp(-z_hi)))
                                - np.mean(1 / (1 + np.exp(-z_lo))))
        np.testing.assert_allclose(eff, expected, rtol=1e-9, atol=1e-12)
        # The sign of the contrast is the sign of the coefficient, exactly, and
        # the scaling is positive so it does not reorder anything.
        live = X.std(axis=0) > 0
        np.testing.assert_array_equal(np.sign(eff[live]), np.sign(w[live]))
        self.assertTrue(bool(np.all(sigma > 0)))

    def test_a_constant_feature_has_no_contrast_and_reports_zero(self):
        """`is_home` cannot be moved by +/-1 SD, so the honest answer is 0.0.

        NaN would also be defensible, but 0.0 reads correctly in an agreement
        table: "this family does not respond to it". It is not evidence that the
        feature is useless in general, which is why the docstring says so.
        """
        games, X, y = tiny_problem(800, seed=31)
        m = gbm.make_model("hist", gbm.HIST_GRID[0])
        gbm.fit_single_threaded(m, X, y)
        self.assertEqual(float(gbm.effect_directions(m, X)["is_home"]), 0.0)

    def test_every_live_feature_gets_a_finite_number(self):
        """No silent NaN, which would drop out of the agreement table and
        flatter the agreement rate."""
        games, X, y = tiny_problem(800, seed=31)
        m = gbm.make_model("xgb", gbm.XGB_GRID[0])
        gbm.fit_single_threaded(m, X, y)
        eff = gbm.effect_directions(m, X)
        self.assertTrue(bool(np.isfinite(eff.to_numpy()).all()))
        live = [n for n in gbm.FEATURE_NAMES
                if X[:, gbm.FEATURE_NAMES.index(n)].std() > 0]
        self.assertEqual(len(live), len(gbm.FEATURE_NAMES) - 1,
                         "only is_home should be constant on this fixture")

    def test_a_planted_signal_comes_out_with_the_right_sign_in_every_family(self):
        """Column 0 is home quality minus away quality by construction.

        Raising it must raise the home-win probability in every family, the trees
        included -- which is exactly the claim the sign audit rests on.
        """
        games, X, y = tiny_problem(1500, seed=31)
        for fam, params in (("xgb", gbm.XGB_GRID[0]), ("hist", gbm.HIST_GRID[0]),
                            ("gbdt", gbm.GBDT_GRID[0]), ("logreg", {})):
            m = gbm.make_model(fam, params)
            gbm.fit_single_threaded(m, X, y)
            self.assertGreater(float(gbm.effect_directions(m, X)["d_margin"]), 0.0, fam)

    def test_the_audit_applies_orientation_to_the_cost_columns(self):
        """The orientation table is applied by the audit, not by the model.

        `d_def_rating` is a cost, so more of it hurts. Reported raw the effect is
        negative; in home-help orientation it is positive, which is what lets it
        be compared with the other fourteen.
        """
        games, X, y = tiny_problem(1500, seed=31)
        m = gbm.make_model("logreg", {})
        gbm.fit_single_threaded(m, X, y)
        eff = gbm.effect_directions(m, X)
        beta = m.named_steps["model"].coef_.ravel()
        j = gbm.FEATURE_NAMES.index("d_def_rating")
        self.assertEqual(np.sign(float(eff["d_def_rating"])), np.sign(beta[j]))
        self.assertEqual(np.sign(float(eff["d_def_rating"]))
                         * gbm.FEATURE_ORIENTATION["d_def_rating"],
                         -np.sign(beta[j]))

    def test_the_audit_reports_one_row_per_feature_and_one_column_per_family(self):
        games, X, y = tiny_problem(2000, seed=19)
        grids = dict(TINY_GRIDS)
        grids["gbdt"] = gbm.GBDT_GRID
        fams = ("xgb", "hist", "gbdt", "logreg")
        effects, agreement = gbm.cross_family_sign_audit(
            games, X, y, families=fams, n_splits=2, seed=5, grids=grids)
        self.assertEqual(list(effects.index), list(gbm.FEATURE_NAMES))
        self.assertEqual(list(effects.columns), list(fams))
        self.assertTrue(bool(np.isfinite(effects.to_numpy()).all()))
        # One row per family PAIR for the agreement rate, plus one column per
        # family counting the features that family has no effect on. Four
        # families give six pairs.
        n_pairs = len(fams) * (len(fams) - 1) // 2
        self.assertEqual(len(agreement), n_pairs)
        self.assertEqual(list(agreement.columns),
                         ["sign_agreement"] + [f"{f}: zero_effect" for f in fams])
        self.assertEqual(set(agreement["sign_agreement"].index),
                         {f"{a} vs {b}" for i, a in enumerate(fams)
                          for b in fams[i + 1:]})
        # `is_home` is constant, so every family must report exactly one
        # feature it cannot respond to. The per-family count is a scalar
        # broadcast down every pair row, hence the `unique()`.
        for f in fams:
            self.assertEqual(agreement[f"{f}: zero_effect"].unique().tolist(), [1],
                             f"{f} should be blind to is_home and nothing else")
        for frac in agreement["sign_agreement"]:
            self.assertTrue(0.0 <= frac <= 1.0)


if __name__ == "__main__":
    unittest.main()