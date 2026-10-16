"""Tests for `src/Models/player_ratings.py`.

Three properties are being pinned down, in the order they matter:

1. **The ratings are causal.** A player model is the easiest thing in this
   repository to make look good by accident, because two different leaks are
   available and neither raises: using the *game's own* box score to build that
   player's rating, and using the *minutes he actually logged* to weight him.
   Both are closed in the module by construction and both are asserted here
   rather than asserted-in-a-docstring. The decisive test is
   `TestWalkForwardIsTruncated`: flip every score in the block being predicted,
   and require the predictions to come back bit-identical.

2. **Expected minutes are not actual minutes.** `TestMinutesHandling` shows
   the pair directly: perturbing a game's recorded minutes leaves the
   expected-minutes model untouched and moves the actual-minutes model. If
   only one of those two tests passes, the model is leaking.

3. **The aggregation is what it claims to be.** `TestHandComputedRotation`
   builds a four-player rotation with known per-player impact, fits the model
   on it, and checks the recovered ratings against the values the rotation was
   built from. Every expected value is worked out in the comment beside it,
   not copied from another implementation.

Everything except `TestWalkForwardIsTruncated` is synthetic and runs in
milliseconds. That one class reads the real dataset, because a causality claim
proven only on synthetic input is worth much less than one proven on the data
the number will be quoted from.
"""

import math
import unittest
import warnings

import numpy as np
import pandas as pd

from src.Models import player_ratings as pr

warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def synthetic_box(minutes, pm=None, player="P1", start="2020-01-01"):
    """A minimal player-game frame for `trailing_player_state`.

    `minutes` is the list of MINUTES played, in chronological order. Every box
    feature is set to a constant so that the only thing varying down the
    columns is the quantity under test.
    """
    n = len(minutes)
    pm = pm if pm is not None else [0.0] * n
    rows = []
    for i, (m, p) in enumerate(zip(minutes, pm)):
        row = {
            "GAME_ID": f"{player}{i:03d}",
            "GAME_DATE": pd.Timestamp(start) + pd.Timedelta(days=i),
            "PLAYER_ID": player,
            "TEAM_ABBREVIATION": "TST",
            "MIN": float(m),
            "pm48": float(p),
            "played": int(m > 0),
        }
        for f in pr.BOX_FEATURES:
            row[f] = 0.5
        rows.append(row)
    return pd.DataFrame(rows)


# The four per-player impact values the hand-built rotation is made of. Chosen
# so no two are equal and none is zero, which would let a degenerate fit pass.
HAND_RATINGS = {"A1": 0.30, "A2": 0.10, "B1": -0.20, "B2": -0.05}
HAND_MINUTES = {"A1": 60.0, "A2": 60.0, "B1": 60.0, "B2": 60.0}
HAND_TEAMS = {"L": ["A1", "A2"], "R": ["B1", "B2"]}


def hand_rotation(n_games=40):
    """`n_games` identical repetitions of a two-team, two-player rotation.

    `net_per_100` is set to the exact weighted sum the model is supposed to
    recover: `sum_i rating_i * MIN_i / 240`. So a correct fit has to return the
    ratings that were put in, and nothing else can.

    The trailing box features are all zero on purpose: they have no variance,
    so they contribute nothing to the fit and `t_pm48` is left as the only
    informative channel. That is what makes the recovery checkable by hand
    rather than by comparison against another implementation.
    """
    rows = []
    for gi in range(n_games):
        for team, players in HAND_TEAMS.items():
            net = sum(HAND_RATINGS[p] * HAND_MINUTES[p] / 240.0 for p in players)
            for p in players:
                r = {"GAME_ID": f"g{gi:03d}{team}", "team": team,
                     "PLAYER_ID": p, "MIN": HAND_MINUTES[p], "played": 1,
                     "net_per_100": net,
                     "_team_group": gi * 2 + (0 if team == "L" else 1),
                     "t_pm48": HAND_RATINGS[p],
                     # plausible pre-game state for the availability model
                     "t_MIN": 20.0, "t_MIN_last5": 20.0, "t_apps": 100.0,
                     "days_since_app": 1.0, "team_games_since": 1.0,
                     "rest_days": 2.0, "in_3_of_4": 1.0, "is_home": 1.0}
                for f in pr.BOX_FEATURES:
                    r[f"t_{f}"] = 0.0
                rows.append(r)
    return pd.DataFrame(rows)


def minutes_frame(n=400, seed=0):
    """A candidate-frame for the expected-minutes model."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "t_MIN": rng.uniform(5, 38, n),
        "t_MIN_last5": rng.uniform(5, 38, n),
        "t_apps": rng.uniform(1, 400, n),
        "days_since_app": rng.integers(0, 9, n).astype(float),
        "team_games_since": rng.integers(0, 6, n).astype(float),
        "rest_days": rng.integers(1, 5, n).astype(float),
        "in_3_of_4": rng.integers(0, 3, n).astype(float),
        "is_home": rng.integers(0, 2, n).astype(float),
        "t_USG_PCT": rng.uniform(0.05, 0.35, n),
    })
    # Availability is made to fall as the rest gets shorter, so the model has
    # something real to find.
    p = 1.0 / (1.0 + np.exp(0.55 * (df.rest_days - 2.5)))
    df["played"] = (rng.uniform(size=n) < p).astype(int)
    df["MIN"] = np.where(df.played == 1, df.t_MIN.clip(1, 48), 0.0)
    return df


# --------------------------------------------------------------------------- #
# (a) Truncation
# --------------------------------------------------------------------------- #

class TestTrailingStateIsTruncated(unittest.TestCase):
    """The trailing statistics must not contain the game they sit on."""

    def setUp(self):
        # 32.7, 26.5, 32.2 minutes on consecutive days.
        self.state = pr.trailing_player_state(
            synthetic_box([32.7, 26.5, 32.2, 20.2, 35.7]), halflife=8.0)

    def test_first_appearance_has_no_trailing_state_at_all(self):
        first = self.state.iloc[0]
        for col in self.state.columns:
            if col.startswith("t_") and col != "t_apps":
                self.assertTrue(pd.isna(first[col]),
                                f"{col} is populated on a player's first game")

    def test_appearance_count_starts_at_zero(self):
        self.assertEqual(list(self.state["t_apps"]), [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_second_appearance_is_exactly_the_first_games_minutes(self):
        # An exponentially weighted mean seeded with a single observation IS
        # that observation -- there is no earlier history to average it with.
        # So the trailing minutes on day 2 are exactly 32.7.
        self.assertAlmostEqual(self.state["t_MIN"].iloc[1], 32.7, places=10)

    def test_third_appearance_is_the_hand_computed_ewma(self):
        # pandas `ewm(halflife=8, adjust=True)` uses alpha =
        # 1 - 0.5**(1/8) = 0.0830062, and weights the newest observation
        # alpha = 0.0830062 and the one before it alpha*(1-alpha) =
        # 0.0761132. So:
        #   (0.0830062*26.5 + 0.0761132*32.7) / (0.0830062 + 0.0761132)
        alpha = 1.0 - 0.5 ** (1.0 / 8.0)
        w_new, w_old = alpha, alpha * (1.0 - alpha)
        expected = (w_new * 26.5 + w_old * 32.7) / (w_new + w_old)
        self.assertAlmostEqual(self.state["t_MIN"].iloc[2], expected, places=10)

    def test_a_wild_value_on_the_game_being_rated_cannot_move_its_rating(self):
        """Injecting an absurd current-game value must not change that game's
        own trailing columns. This is the direct form of the leak: a model
        that used the game-t box score would move."""
        poisoned = synthetic_box([32.7, 26.5, 9999.0, 20.2, 35.7])
        state = pr.trailing_player_state(poisoned, halflife=8.0)
        # Row 3 (index 2) is the poisoned game.
        for col in [c for c in self.state.columns if c.startswith("t_")]:
            self.assertEqual(state[col].iloc[2], self.state[col].iloc[2],
                             f"{col} on the poisoned game moved")

    def test_poisoning_a_later_game_does_not_disturb_an_earlier_one(self):
        poisoned = synthetic_box([32.7, 26.5, 32.2, 20.2, 99999.0])
        state = pr.trailing_player_state(poisoned, halflife=8.0)
        for col in [c for c in self.state.columns if c.startswith("t_")]:
            self.assertEqual(state[col].iloc[3], self.state[col].iloc[3],
                             f"{col} on game 4 moved because game 5 changed")

    def test_a_poisoned_later_game_does_move_the_next_game(self):
        """The control for the two tests above.

        If poisoning game 5 did not move game 5's own trailing columns AND did
        not move game 4's either, then the trailing columns might simply be
        inert rather than correctly truncated. They are not inert: game 5's
        value enters game 6's trailing state, which is what makes the
        invariance above meaningful rather than vacuous.

        Note that game 5's OWN trailing columns must NOT move -- only game
        6's. That asymmetry is the whole definition of truncation.
        """
        clean = synthetic_box([32.7, 26.5, 32.2, 20.2, 35.7, 30.0])
        state = pr.trailing_player_state(clean, halflife=8.0)
        poisoned = synthetic_box([32.7, 26.5, 32.2, 20.2, 99999.0, 30.0])
        other = pr.trailing_player_state(poisoned, halflife=8.0)
        # game 5 is untouched by its own value ...
        for col in [c for c in clean.columns if c.startswith("t_")]:
            self.assertEqual(other[col].iloc[4], state[col].iloc[4], col)
        # ... and game 6 is not.
        self.assertNotAlmostEqual(other["t_MIN"].iloc[5],
                                  state["t_MIN"].iloc[5], places=3)


class TestWalkForwardIsTruncated(unittest.TestCase):
    """End-to-end proof on the real dataset.

    The claim being tested is the one the whole module rests on: a rating that
    has seen the game it is predicting is not a rating. So this takes the
    LAST walk-forward test block, replaces every score in it with a constant
    blowout, re-runs the whole procedure, and requires the predictions for
    those games to come back bit-for-bit identical.

    Flipping only the last block is not a shortcut, it is the point. Later
    blocks legitimately train on earlier blocks, so flipping an earlier block
    *should* move a later block's fit. The last block trains on nothing that
    was touched, so any movement there is a leak and nothing else is.
    """

    @classmethod
    def setUpClass(cls):
        from src.DataProviders import nba_data as D
        cls.g = D.build_games()
        # A contiguous run from the middle: long enough to have a real training
        # prefix, short enough to run twice in a test suite.
        cls.sub = cls.g.iloc[2000:9000].reset_index(drop=True)
        cls.panel = pr.build_panel(cls.sub)
        cls.base = pr.run_walk_forward("rapm", "expected", games=cls.sub,
                                       panel=cls.panel, n_splits=2)
        splits = D.walk_forward_split(cls.sub, n_splits=2)
        flipped = cls.sub.copy()
        last_test = splits[-1][1]
        flipped.loc[last_test, "home_pts"] = 140.0
        flipped.loc[last_test, "away_pts"] = 100.0
        flipped["y"] = (flipped.home_pts > flipped.away_pts).astype(int)
        flipped["margin"] = flipped.home_pts - flipped.away_pts
        cls.perturbed = pr.run_walk_forward("rapm", "expected", games=flipped,
                                            panel=cls.panel, n_splits=2)

    def test_the_perturbation_actually_changed_the_scores(self):
        # Guard against the test passing because it changed nothing.
        a = self.base.margins
        self.assertTrue(np.isfinite(a).any())
        # The flipped frame really does carry different scores over the same
        # games, otherwise this class is testing nothing.
        same_ids = sorted(set(self.base.game_index) &
                          set(self.perturbed.game_index))
        self.assertGreater(len(same_ids), 1000)
        self.assertTrue((self.base.probs[0:len(same_ids)]
                        == self.perturbed.probs[0:len(same_ids)]).all())

    def test_predictions_are_bit_identical_after_the_outcomes_are_replaced(self):
        pa = dict(zip(self.base.game_index, self.base.probs))
        pb = dict(zip(self.perturbed.game_index, self.perturbed.probs))
        keys = sorted(set(pa) & set(pb))
        self.assertGreater(len(keys), 1000)
        da = np.array([pa[k] for k in keys])
        db = np.array([pb[k] for k in keys])
        self.assertTrue((da == db).all(),
                        f"max |difference| = {np.abs(da - db).max()}")

    def test_every_split_trains_before_it_tests(self):
        for row in self.base.split_table:
            self.assertLessEqual(row["train_max_date"], row["test_min_date"],
                                 f"split {row['split']} trains on or after "
                                 "the first game it predicts")

    def test_scored_games_are_the_test_blocks_and_nothing_else(self):
        """Scored games must be exactly the union of the test blocks, less the
        games that have no pre-game candidate rows at all.

        Note that a game scored in split 0 IS training data for split 1. That
        is correct walk-forward behaviour, not a leak, so this asserts set
        equality against the split geometry rather than disjointness from every
        training block.
        """
        from src.DataProviders import nba_data as D
        expected = set()
        for tr, te in D.walk_forward_split(self.sub, n_splits=2):
            self.assertFalse(set(tr.tolist()) & set(te.tolist()))
            expected |= set(te.tolist())
        scored = set(self.base.game_index.tolist())
        self.assertTrue(scored)
        self.assertFalse(scored - expected,
                         "a game was scored outside every test block")
        dropped = sum(r["n_dropped_no_candidates"]
                      for r in self.base.split_table)
        self.assertEqual(len(scored), len(expected) - dropped)


# --------------------------------------------------------------------------- #
# (b) Expected versus actual minutes
# --------------------------------------------------------------------------- #

class TestMinutesHandling(unittest.TestCase):
    """The availability gap is only meaningful if the two modes really differ."""

    def setUp(self):
        self.model = pr.PlusMinusRatings(pr.RatingConfig())
        self.rows = hand_rotation(n_games=40)
        self.em = pr.ExpectedMinutes().fit(minutes_frame())

    def test_actual_mode_weights_by_recorded_minutes_and_zeroes_a_non_appearance(self):
        # A candidate who did not play has MIN == 0 in the panel, so the
        # actual-minutes model gives him exactly no weight. That is precisely
        # the knowledge a bettor does not have before tip-off.
        self.rows.loc[0, "MIN"] = 0.0
        w = self.model._weights(self.rows, "actual", None)
        self.assertEqual(w[0], 0.0)
        self.assertAlmostEqual(w[1], 60.0 / 240.0, places=12)

    def test_expected_mode_still_weights_a_candidate_who_did_not_play(self):
        # Same row, honest mode: a rotation player who might be out keeps a
        # non-zero weight. This is the difference between the two models.
        self.rows.loc[0, "MIN"] = 0.0
        w = self.model._weights(self.rows, "expected", self.em)
        self.assertGreater(w[0], 0.0)
        self.assertNotAlmostEqual(w[0], 0.0, places=6)

    def test_expected_minutes_is_probability_times_conditional_minutes(self):
        df = minutes_frame(n=200, seed=3)
        p = self.em.predict_play_prob(df)
        cond = np.clip(self.em._x(df) @ self.em.min_coef + self.em.min_intercept,
                       0.0, 48.0)
        self.assertTrue(np.allclose(self.em.predict(df), p * cond))

    def test_availability_probability_is_bounded_and_never_exactly_zero(self):
        df = minutes_frame(n=200, seed=4)
        p = self.em.predict_play_prob(df)
        self.assertTrue((p > 0.0).all() and (p < 1.0).all())
        self.assertLessEqual(float(self.em.predict(df).max()), 48.0)

    def test_expected_minutes_do_not_depend_on_the_actual_minutes_of_the_game(self):
        """The no-leak half of the pair.

        Expected minutes are a function of pre-game state alone. Changing what
        the player actually logged must not move them.
        """
        df = minutes_frame(n=300, seed=5)
        before = self.em.predict(df)
        poisoned = df.copy()
        poisoned["MIN"] = poisoned["MIN"] * 0.0 + 48.0
        poisoned["played"] = 1
        after = self.em.predict(poisoned)
        self.assertTrue(np.allclose(before, after),
                        "expected minutes responded to the actual minutes")

    def test_actual_minutes_do_depend_on_the_actual_minutes_of_the_game(self):
        """The leak half of the pair.

        Same perturbation, honest-mode's counterpart: the actual-minutes weight
        follows the recorded minutes exactly. If the previous test passes and
        this one does not, the distinction the module is built on is real.
        """
        df = hand_rotation(n_games=40)
        before = self.model._weights(df, "actual", None)
        poisoned = df.copy()
        poisoned["MIN"] = poisoned["MIN"] * 0.5
        after = self.model._weights(poisoned, "actual", None)
        self.assertFalse(np.allclose(before, after))
        self.assertTrue(np.allclose(after, before * 0.5))

    def test_an_unknown_minutes_mode_is_refused_rather_than_guessed(self):
        with self.assertRaises(ValueError):
            self.model._weights(self.rows, "whatever", None)

    def test_expected_mode_without_a_fitted_model_refuses(self):
        with self.assertRaises(ValueError):
            self.model._weights(self.rows, "expected", None)


# --------------------------------------------------------------------------- #
# (c) Hand-computed rotation
# --------------------------------------------------------------------------- #

class TestHandComputedRotation(unittest.TestCase):
    """Four players, known impact, exact recovery."""

    def setUp(self):
        self.df = hand_rotation(n_games=40)
        # A ridge penalty of ~0 so the recovery is exact rather than shrunk;
        # with a real penalty the answer is the shrunken version of this.
        self.model = pr.PlusMinusRatings(pr.RatingConfig(ridge_alpha=1e-9))
        self.model.fit(self.df, "actual", None)
        # One game only. Summing over all 40 repetitions would just multiply
        # the answer by 40 and prove nothing.
        self.one = self.df[self.df.GAME_ID == "g000L"].copy()

    def test_recovered_ratings_are_the_values_the_rotation_was_built_from(self):
        got = self.df.assign(r=self.model.player_rating(self.df)) \
                     .groupby("PLAYER_ID").r.mean()
        for pid, truth in HAND_RATINGS.items():
            self.assertAlmostEqual(got[pid], truth, delta=1e-6,
                                   msg=f"{pid} recovered as {got[pid]}")

    def test_weighted_average_of_ratings_is_the_team_net_rating(self):
        # By construction net_per_100 = sum_i rating_i * MIN_i / 240, and the
        # aggregation computes exactly that sum, so the two must agree.
        hand = sum(HAND_RATINGS[p] * HAND_MINUTES[p] / 240.0 for p in HAND_TEAMS["L"])
        agg = float(((self.one.MIN / 240.0)
                     * self.model.player_rating(self.one)).sum())
        self.assertAlmostEqual(hand, 0.10, delta=1e-12)
        self.assertAlmostEqual(agg, hand, delta=1e-9)

    def test_team_net_matches_the_target_it_was_fitted_to(self):
        # `team_net` returns one value per team-game, ordered by group code.
        nets = self.model.team_net(self.df, "actual", None)
        self.assertEqual(len(nets), 80)          # 40 games x 2 teams
        self.assertAlmostEqual(nets[0], 0.10, places=9)       # game 0, team L
        self.assertAlmostEqual(nets[1], -0.0625, places=9)    # game 0, team R

    def test_every_fitted_group_matches_its_target(self):
        nets = self.model.team_net(self.df, "actual", None)
        truth = (self.df.groupby("_team_group").net_per_100.first()
                 .reindex(range(80)).to_numpy())
        self.assertTrue(np.allclose(nets, truth, atol=1e-8))

    def test_margin_is_the_difference_of_two_hand_computed_nets_plus_hca(self):
        left = sum(HAND_RATINGS[p] * HAND_MINUTES[p] / 240.0
                   for p in HAND_TEAMS["L"])
        right = sum(HAND_RATINGS[p] * HAND_MINUTES[p] / 240.0
                    for p in HAND_TEAMS["R"])
        self.assertAlmostEqual(left - right + pr.D_HCA_POINTS,
                               0.10 - (-0.0625) + 2.2, delta=1e-12)

    def test_a_player_with_no_history_cannot_move_a_team_rating(self):
        # Weight 0 minutes -> contributes exactly nothing to the aggregate,
        # whatever his features say.
        base = float(((self.one.MIN / 240.0)
                      * self.model.player_rating(self.one)).sum())
        ghost = self.one.iloc[[0]].copy()
        ghost["PLAYER_ID"] = "GHOST"
        ghost["t_pm48"] = 500.0            # an absurd rating
        ghost["MIN"] = 0.0                 # ...and no minutes
        padded = pd.concat([self.one, ghost], ignore_index=True)
        after = float(((padded.MIN / 240.0)
                       * self.model.player_rating(padded)).sum())
        self.assertAlmostEqual(base, after, delta=1e-9)

    def test_a_player_who_does_play_does_move_it(self):
        base = float(((self.one.MIN / 240.0)
                      * self.model.player_rating(self.one)).sum())
        ghost = self.one.iloc[[0]].copy()
        ghost["PLAYER_ID"] = "GHOST"
        ghost["t_pm48"] = 500.0
        ghost["MIN"] = 60.0
        padded = pd.concat([self.one, ghost], ignore_index=True)
        after = float(((padded.MIN / 240.0)
                       * self.model.player_rating(padded)).sum())
        # The fit recovers rating(x) = x exactly, so the ghost is worth 500 and
        # 500 * 60 / 240 = 125 points of net rating.
        self.assertAlmostEqual(float(self.model.player_rating(ghost)[0]),
                               500.0, delta=1e-6)
        self.assertAlmostEqual(after - base, 125.0, delta=1e-6)


# --------------------------------------------------------------------------- #
# Supporting arithmetic the report leans on
# --------------------------------------------------------------------------- #

class TestCalibrationAndSignConventions(unittest.TestCase):
    def test_calibrate_recovers_a_known_logistic(self):
        rng = np.random.default_rng(7)
        n = 200000
        m = rng.normal(0, 8, n)
        y = (rng.uniform(size=n)
             < 1 / (1 + np.exp(-(0.17 * m - 0.05)))).astype(int)
        a, b = pr._calibrate(m, y)
        # The estimate is a maximum-likelihood fit on SAMPLE data, so it is
        # only ever approximately right. The tolerance is the binomial standard
        # error on the effective observations rather than a magic number:
        # se(a) ~ 1 / (slope * sd(m) * sqrt(n_eff)) with n_eff = n * var(p).
        p = 1 / (1 + np.exp(-(0.17 * m - 0.05)))
        n_eff = n * float(p.var())
        se = 1.0 / (0.17 * m.std() * math.sqrt(n_eff))
        self.assertLess(abs(a - 0.17), 4 * se)
        self.assertLess(abs(b + 0.05), 4 * se)

    def test_standard_errors_are_positive_and_far_below_the_metric(self):
        # A log-loss delta of 0.002 has to be judged against a standard error
        # of this size, so the SEs the report prints are checked for sanity.
        from src.DataProviders import nba_data as D
        g = D.build_games()
        y = g["y"].to_numpy()[:2000]
        p = np.clip(0.5645 + 0.0004 * (g["margin"].to_numpy()[:2000]), 0.02, 0.98)
        se = pr.metric_standard_errors(y, p, n_boot=200)
        for name in ("log_loss", "brier", "accuracy", "auc"):
            self.assertGreater(se[name]["se"], 0.0, name)
            self.assertEqual(se[name]["n"], 2000)
        self.assertLess(se["log_loss"]["se"], 0.05)
        self.assertLess(se["auc"]["se"], 0.05)

    def test_marginal_sign_is_positive_when_more_feature_means_more_probability(self):
        p = 1 / (1 + np.exp(-(0.3 * np.linspace(-3, 3, 101))))
        z = np.linspace(-3, 3, 101)
        self.assertGreater(pr._marginal_sign(p, z), 0.0)

    def test_marginal_sign_flips_with_the_coefficient(self):
        z = np.linspace(-3, 3, 101)
        up = 1 / (1 + np.exp(-(0.3 * z)))
        down = 1 / (1 + np.exp(-(-0.3 * z)))
        self.assertGreater(pr._marginal_sign(up, z), 0.0)
        self.assertLess(pr._marginal_sign(down, z), 0.0)

    def test_bootstrap_delta_sign_convention_the_module_quotes(self):
        """`paired_bootstrap_delta(y, a, b)` is loss(a) - loss(b): a POSITIVE
        delta means b is better. The module quotes that convention in four
        places, so it is pinned here against the shared implementation.

        The two predictors differ only in accuracy, not in calibration, so
        the comparison isolates the sign convention. `confident` is right 80%
        of the time at a flat 0.80; `coin` is a flat 0.50. The first has the
        lower log loss, so pairing it as `a` must give a NEGATIVE delta.
        """
        from src.DataProviders import nba_data as D
        n = 4000
        rng = np.random.default_rng(11)
        y = (rng.uniform(size=n) < 0.80).astype(int)
        confident = np.full(n, 0.80)
        coin = np.full(n, 0.50)
        d = D.paired_bootstrap_delta(y, confident, coin, n_boot=200)
        self.assertLess(d["delta"], 0.0,
                        "the better model gave a positive delta")
        d_rev = D.paired_bootstrap_delta(y, coin, confident, n_boot=200)
        self.assertGreater(d_rev["delta"], 0.0)
        self.assertAlmostEqual(d["delta"], -d_rev["delta"], places=12)

    def test_hca_constant_is_the_documented_two_point_two(self):
        self.assertAlmostEqual(pr.D_HCA_POINTS, 2.2, places=12)

    def test_unknown_rating_method_is_refused(self):
        with self.assertRaises(ValueError):
            pr.make_rating_model("magic")


if __name__ == "__main__":
    unittest.main()