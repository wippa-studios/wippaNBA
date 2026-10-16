"""Tests for `src/Models/ratings.py`.

Two rules govern the numbers in this file, and they are the same two rules
`Tests/test_metrics.py` follows.

1. **Every expected value is derived in the comment next to it**, from the
   formula in the module, not copied from a library. Where scikit-learn appears
   it is a sanity check on the module's own arithmetic, never the source of an
   expected figure.
2. **The ordering tests are the point of this file.** A rating system that
   processes its games in the wrong order still produces confident, plausible,
   entirely wrong numbers, and nothing about the output says so. Three separate
   tests attack it from different directions, and the reasoning for why each is
   necessary is in the class docstrings.

The module's central data claim is also tested here rather than trusted:
`Points` in `OddsData.sqlite` is the *combined* score, not the winning team's
score. See `TestScoresFromOdds`.
"""

import math
import unittest

import numpy as np
import pandas as pd

from src.Models import ratings


# Elo's classic curve, spelled out so a test can reproduce any probability by
# hand instead of calling back into the implementation it is checking.
def elo_prob(diff, scale=400.0):
    """P(home wins) for a rating difference of `diff`, matching EloRating."""
    return 1.0 / (1.0 + 10.0 ** (-diff / scale))


def season_dates(n, start="2024-10-22"):
    """`n` distinct, monotonically increasing ISO dates, ~3 games apart.

    Built by adding a real timedelta rather than by formatting day counters:
    ``2024-02-30`` is what you get from the naive arithmetic, and pandas rejects
    it. The step of three days roughly matches an NBA slate, so a fixture that
    wants a realistic season gets one.
    """
    first = pd.Timestamp(start)
    return [(first + pd.Timedelta(days=3 * i)).strftime("%Y-%m-%d") for i in range(n)]


def canonical(rows, *, dates=None, margins=None, rest=None):
    """Build a minimal canonical games frame for the tests below.

    `dates` defaults to distinct increasing dates. A single repeated date works
    for most tests, but it makes SRS's time-ordered calibration holdout
    degenerate when a fixture needs many games, so the default is always a real
    spread.
    """
    n = len(rows)
    if dates is None:
        dates = season_dates(n)
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "home": [r[0] for r in rows],
            "away": [r[1] for r in rows],
            "home_win": [float(r[2]) for r in rows],
            "margin": np.asarray(margins if margins is not None else [np.nan] * n, dtype=float),
            "days_rest_home": np.asarray(
                rest[:, 0] if rest is not None else [np.nan] * n, dtype=float
            ),
            "days_rest_away": np.asarray(
                rest[:, 1] if rest is not None else [np.nan] * n, dtype=float
            ),
        }
    )


class TestScoresFromOdds(unittest.TestCase):
    """`Points` is the combined score, not the winning team's score.

    The intuitive reading of an odds table -- "Points is what the winner
    scored, so the loser scored `Points - Win_Margin`" -- is wrong here, and
    wrong in a way that would quietly double every margin fed to a
    margin-of-victory multiplier. The scraper is explicit
    (`src/Process-Data/Get_Odds_Data.py`):

        "Points": game["away_score"] + game["home_score"],
        "Win_Margin": game["home_score"] - game["away_score"],

    and the arithmetic below is the check that the data agrees.
    """

    def test_split_reconstructs_both_scores(self):
        # A 118-104 home win: Points = 222, Win_Margin = +14.
        # home = (222 + 14) / 2 = 118, away = (222 - 14) / 2 = 104.
        home, away = ratings.scores_from_odds(222, 14)
        self.assertEqual((home, away), (118.0, 104.0))

    def test_away_win_has_negative_margin(self):
        # A 97-110 away win: Points = 207, Win_Margin = -13.
        # home = (207 - 13) / 2 = 97, away = (207 + 13) / 2 = 110.
        home, away = ratings.scores_from_odds(207, -13)
        self.assertEqual((home, away), (97.0, 110.0))

    def test_scores_always_sum_to_points(self):
        for points, margin in [(222, 14), (207, -13), (300, 0), (180, 30)]:
            home, away = ratings.scores_from_odds(points, margin)
            self.assertAlmostEqual(home + away, float(points))

    def test_combined_reading_is_the_only_one_that_lands_in_range(self):
        # The losing hypothesis ("Points is the winner's score") is testable on
        # a real row: Denver beat the Lakers by 12 on 2023-10-24 with
        # Points = 226. Under the combined reading the split is
        # (226 + 12) / 2 = 119 and 107 -- both plausible team scores.
        # Under the winner's-score reading the "loser" would have scored
        # 226 - 12 = 214, which no NBA team has ever done.
        home, away = ratings.scores_from_odds(226, 12)
        self.assertEqual((home, away), (119.0, 107.0))
        self.assertLess(max(home, away), 200)

    def test_rejects_input_that_admits_no_integer_split(self):
        # Points +/- Win_Margin must be even for real box scores. An odd value
        # means the two columns are not the pair this function assumes, and
        # guessing anyway would produce half-point "scores".
        with self.assertRaises(ValueError):
            ratings.scores_from_odds(221, 14)

    def test_is_vectorised_over_arrays(self):
        home, away = ratings.scores_from_odds(np.array([222.0, 207.0]), np.array([14.0, -13.0]))
        np.testing.assert_allclose(home, [118.0, 97.0])
        np.testing.assert_allclose(away, [104.0, 110.0])


class TestEqualRatingsAreExactlyFair(unittest.TestCase):
    """A coin-flip matchup must produce a coin flip, exactly.

    Not "approximately" and not "to six decimal places": the arithmetic is
    `1 / (1 + 10**(-0/400))`, and `10**(-0.0)` is exactly `1.0`, so the result
    is exactly `0.5`. Any home-court advantage, rest term or margin multiplier
    configured into the rating would break this, which is the point -- it is the
    shortest possible statement that nothing is being silently added.
    """

    def test_single_rating_predicts_exactly_one_half(self):
        elo = ratings.EloRating(ratings.EloConfig(home_advantage=0.0))
        self.assertEqual(elo.prob_home("A", "B"), 0.5)
        self.assertEqual(elo.prob_home("B", "A"), 0.5)

    def test_default_config_predicts_exactly_one_half_when_hca_is_zero(self):
        # Same thing, but reached through the default config path.
        config = ratings.EloConfig()
        elo = ratings.EloRating(ratings.EloConfig(home_advantage=0.0, k=config.k))
        self.assertEqual(elo.rating("A"), elo.rating("B"))
        self.assertEqual(elo.prob_home("A", "B"), 0.5)

    def test_first_game_of_a_season_is_a_coin_flip(self):
        # Through fit_predict_elo: the very first row's prediction must come
        # from two untouched ratings.
        games = canonical([("A", "B", 1)], dates=["2024-10-22"])
        fit = ratings.fit_predict_elo(games, home_advantage=0.0, use_margin=False)
        self.assertEqual(fit.prob_home[0], 0.5)

    def test_home_advantage_shifts_the_probability_the_expected_amount(self):
        # A 70-point home advantage is 70/400 = 0.175 decades of the curve, so
        # 10**(-0.175) = 0.6683439 and P(home) = 1 / (1 + 0.6683439) =
        # 0.5993968.
        elo = ratings.EloRating(ratings.EloConfig(home_advantage=70.0))
        self.assertAlmostEqual(elo.prob_home("A", "B"), elo_prob(70.0), places=15)
        self.assertAlmostEqual(10.0 ** (-0.175), 0.6683439, places=7)
        self.assertAlmostEqual(elo.prob_home("A", "B"), 0.5993968, places=7)

    def test_rest_adjustment_breaks_the_coin_flip_only_when_supplied(self):
        # No rest given: still exactly 0.5. Rest given: the 3-point-per-day
        # coefficient times a two-day edge is +6 rating points.
        elo = ratings.EloRating(ratings.EloConfig(home_advantage=0.0, rest_coefficient=3.0))
        self.assertEqual(elo.prob_home("A", "B", days_rest_home=1, days_rest_away=1), 0.5)
        self.assertAlmostEqual(
            elo.prob_home("A", "B", days_rest_home=3, days_rest_away=1), elo_prob(6.0), places=15
        )


class TestHandComputedTwoGameOrdering(unittest.TestCase):
    """The ordering guarantee, asserted against arithmetic done by hand.

    Two games, two teams, no home court, no margin, no season change:

    game 1  A home vs B, A wins.
            both ratings 0, so diff = 0, P(A wins) = 1/(1 + 10**0) = 0.5
            delta = K * (1 - 0.5) = 20 * 0.5 = +10
            => A = +10, B = -10

    game 2  B home vs A, B loses.
            diff = B - A = -10 - 10 = -20
            P(B wins) = 1 / (1 + 10**(20/400)) = 1 / (1 + 10**0.05)
                      = 1 / (1 + 1.1220185) = 0.4712494
            delta = 20 * (0 - 0.4712494) = -9.4249888
            => B = -10 - 9.4249888 = -19.4249888
            => A = +10 + 9.4249888 = +19.4249888

    If the implementation updated before predicting, game 1's stored probability
    would be 1/(1+10**(20/400)) = 0.5287556 instead of 0.5 and every later
    number would shift with it. The first assertion is the one that fails first.
    """

    games = canonical(
        [("A", "B", 1), ("B", "A", 0)],
        dates=["2024-01-01", "2024-01-02"],
    )

    def test_game_one_is_predicted_from_untouched_ratings(self):
        fit = ratings.fit_predict_elo(self.games, home_advantage=0.0, use_margin=False)
        self.assertEqual(fit.games.loc[0, "prob_home"], 0.5)
        self.assertEqual(fit.games.loc[0, "rating_home_before"], 0.0)
        self.assertEqual(fit.games.loc[0, "rating_away_before"], 0.0)

    def test_game_one_update_is_ten_points(self):
        fit = ratings.fit_predict_elo(self.games, home_advantage=0.0, use_margin=False)
        self.assertEqual(fit.games.loc[0, "delta"], 10.0)
        self.assertEqual(fit.games.loc[0, "rating_home_after"], 10.0)

    def test_game_two_sees_game_one_but_not_itself(self):
        fit = ratings.fit_predict_elo(self.games, home_advantage=0.0, use_margin=False)
        # diff = -20 exactly, from B's -10 against A's +10.
        self.assertEqual(fit.games.loc[1, "effective_diff"], -20.0)
        self.assertEqual(fit.games.loc[1, "rating_home_before"], -10.0)
        self.assertEqual(fit.games.loc[1, "rating_away_before"], 10.0)

    def test_game_two_probability_matches_the_closed_form(self):
        fit = ratings.fit_predict_elo(self.games, home_advantage=0.0, use_margin=False)
        expected = 1.0 / (1.0 + 10.0 ** (20.0 / 400.0))
        self.assertAlmostEqual(fit.games.loc[1, "prob_home"], expected, places=15)
        self.assertAlmostEqual(10.0**0.05, 1.1220185, places=7)
        self.assertAlmostEqual(fit.games.loc[1, "prob_home"], 0.4712494, places=7)

    def test_final_ratings_match_the_hand_computation(self):
        fit = ratings.fit_predict_elo(self.games, home_advantage=0.0, use_margin=False)
        expected = 10.0 + 20.0 * (1.0 / (1.0 + 10.0 ** (20.0 / 400.0)))
        self.assertAlmostEqual(fit.ratings["A"], expected, places=12)
        self.assertAlmostEqual(fit.ratings["B"], -expected, places=12)

    def test_update_is_never_visible_in_the_same_row(self):
        """`rating_home_after` must differ from `rating_home_before` on a win,
        while `prob_home` is computed from the *before* value."""
        fit = ratings.fit_predict_elo(self.games, home_advantage=0.0, use_margin=False)
        row = fit.games.loc[0]
        self.assertNotEqual(row["rating_home_before"], row["rating_home_after"])
        # The probability is exactly what the before-ratings imply.
        self.assertEqual(
            row["prob_home"],
            elo_prob(row["rating_home_before"] - row["rating_away_before"] + 0.0),
        )


class TestPredictionsPrecedeUpdates(unittest.TestCase):
    """The leak that matters is visible directly: a prediction that depends on
    its own outcome.

    `EloRating` exposes `predict`-then-`update` as separate calls rather than
    one fused `fit_predict`, which is deliberate -- it is what makes this test
    expressible. A fused method would hide the ordering inside an
    implementation nobody can reach from a test.

    The shape asserted here is general: take two games with identical pre-game
    ratings, one won by the home team and one lost. Under correct ordering both
    predictions are identical. Under update-before-predict the won game's
    probability is strictly higher, because the rating already moved. A system
    that orders correctly cannot produce that separation, and one that does not
    always will.
    """

    def _leaked_probabilities(self, games, config):
        """Walk `games` updating before predicting -- the wrong order."""
        elo = ratings.EloRating(config)
        out = []
        for _, row in games.iterrows():
            diff = elo.effective_diff(row["home"], row["away"])
            elo.update(row["home"], row["away"], row["home_win"], effective_diff=diff)
            out.append(elo.prob_home(row["home"], row["away"]))
        return np.asarray(out)

    def test_correct_order_gives_identical_predictions_for_equal_pre_game_state(self):
        # Two games on the same date between disjoint team pairs. Neither game
        # can move the other's ratings, so both are predicted from untouched
        # ratings and both must be exactly 0.5 -- whatever their results, and
        # whichever order they are processed in.
        games = canonical(
            [("A", "B", 1), ("C", "D", 0)],
            dates=["2024-01-01", "2024-01-01"],
        )
        fit = ratings.fit_predict_elo(games, home_advantage=0.0, use_margin=False)
        self.assertEqual(fit.prob_home[0], 0.5)
        self.assertEqual(fit.prob_home[1], 0.5)

    def test_the_outcome_does_not_change_the_prediction_it_followed(self):
        """Two schedules identical except for the outcome of the LAST game.

        Every game before it must be predicted identically, because no earlier
        prediction may depend on any later result. (Varying an *earlier*
        outcome would legitimately change later predictions, so the varying game
        has to be last for this to test the guarantee rather than the ordinary
        behaviour of an updating model.)
        """
        shared = [("A", "B", 1), ("B", "A", 1)]
        dates = ["2024-01-01", "2024-01-02", "2024-01-03"]
        won = canonical(shared + [("C", "D", 1)], dates=dates)
        lost = canonical(shared + [("C", "D", 0)], dates=dates)
        a = ratings.fit_predict_elo(won, home_advantage=0.0, use_margin=False)
        b = ratings.fit_predict_elo(lost, home_advantage=0.0, use_margin=False)
        np.testing.assert_allclose(a.prob_home[:2], b.prob_home[:2], rtol=0, atol=0)
        self.assertEqual(list(a.prob_home[:2]), list(b.prob_home[:2]))
        # The last game's own prediction is the same too: C and D are both
        # still untouched when it is predicted.
        self.assertEqual(a.prob_home[2], b.prob_home[2])
        self.assertEqual(a.prob_home[2], 0.5)

    def test_leaked_prediction_is_a_strict_function_of_the_games_own_outcome(self):
        # Game 1: home wins. Game 2: home loses. Both start from zero ratings.
        games = canonical(
            [("A", "B", 1), ("C", "D", 0)],
            dates=["2024-01-01", "2024-01-02"],
        )
        config = ratings.EloConfig(home_advantage=0.0, mov_multiplier=False)
        correct = ratings.fit_predict_elo(games, config=config)
        leaked = self._leaked_probabilities(games, config)

        # Correctly ordered: both games are coin flips, so the two predictions
        # are equal -- the outcome has no influence on them at all.
        self.assertEqual(correct.prob_home[0], correct.prob_home[1])
        # Leaked: the won game is predicted above 0.5 and the lost game below.
        self.assertGreater(leaked[0], 0.5)
        self.assertLess(leaked[1], 0.5)
        self.assertGreater(leaked[0], leaked[1])

    def test_self_leak_improves_the_in_sample_fit_and_nothing_else(self):
        """Why the leak is dangerous, stated precisely rather than as folklore.

        Updating before predicting leaves the rating *recurrence* untouched --
        the same games are applied in the same order either way -- so the final
        ratings, and therefore any held-out prediction built from them, are
        bit-for-bit identical. The leak buys a better in-sample log loss and
        costs nothing held out. That is exactly why it survives review: the
        number a reviewer checks goes down.

        This is why "reverse the update and watch the log loss degrade" is NOT
        a sufficient test, and this test pins the correct understanding so a
        future change cannot quietly rely on it.
        """
        rng = np.random.default_rng(20240101)
        rows = []
        teams = [f"T{i}" for i in range(8)]
        for i in range(600):
            home, away = rng.choice(teams, 2, replace=False)
            rows.append((home, away, int(rng.random() < 0.55)))
        games = canonical(rows, dates=[f"2024-{1 + i // 200:02d}-{1 + i % 28:02d}" for i in range(600)])
        games = games.sort_values(["date", "home", "away"]).reset_index(drop=True)

        # season_reversion disabled so the two walks differ only by the leak.
        config = ratings.EloConfig(home_advantage=0.0, season_reversion=1.0, mov_multiplier=False)
        correct = ratings.fit_predict_elo(games, config=config)
        leaked = self._leaked_probabilities(games, config)

        y = games["home_win"].to_numpy()
        correct_ll = ratings.score_predictions(y, correct.prob_home)["log_loss"]
        leaked_ll = ratings.score_predictions(y, leaked)["log_loss"]
        # The leak flatters the fit it is measured on.
        self.assertLess(leaked_ll, correct_ll)

        # And it leaves the ratings -- the only thing a held-out prediction
        # depends on -- exactly as they were.
        for team, value in correct.ratings.items():
            self.assertAlmostEqual(
                ratings.EloRating(config).rating(team) * 0.0
                + self._walk_final(games, config, team),
                value,
                places=12,
            )

    def _walk_final(self, games, config, team):
        elo = ratings.EloRating(config)
        for _, row in games.iterrows():
            elo.update(row["home"], row["away"], row["home_win"])
        return elo.ratings[team]


class TestChronologicalOrderBeatsReversed(unittest.TestCase):
    """The ordering violation that actually costs held-out accuracy.

    The previous class shows that self-leak is in-sample-only. To move a
    held-out number the *fit window* has to be walked out of order, which
    changes the ratings the window ends on. `fit_predict_elo` cannot be made to
    do this -- it sorts, so feeding it reversed rows is a no-op (asserted
    below). A real bug does it by hand-rolling the loop, so the test does too.

    Sequence, chosen so the information sits at the END of the window:

    games 1-40   B (home) beats A  -> A's rating falls to about -161
    games 41-60  A (home) beats B  -> A's rating recovers to about +61
    test 61-80   A beats B again, 20 times

    Walking the window forwards leaves A rated above zero, because the last
    twenty games said so. Walking it backwards leaves A rated well below zero,
    because the last twenty games *processed* are the ones from the start of the
    window. The test window goes to the forward reading, so the reversed walk
    gets it wrong -- by a log loss of 0.40 against 2.00, and accuracy 1.00
    against 0.00.
    """

    def _window(self):
        rows = [("B", "A", 1)] * 40 + [("A", "B", 1)] * 20
        dates = [f"2023-{1 + i // 20:02d}-{1 + i % 20:02d}" for i in range(60)]
        return canonical(rows, dates=dates)

    def _walk(self, frame):
        elo = ratings.EloRating(ratings.EloConfig(home_advantage=0.0))
        for _, row in frame.iterrows():
            elo.update(row["home"], row["away"], row["home_win"])
        return elo

    def test_reversed_fit_window_is_materially_worse_held_out(self):
        window = self._window()
        forwards = self._walk(window)
        backwards = self._walk(window.iloc[::-1].reset_index(drop=True))

        y = np.ones(20)
        p_forward = np.full(20, forwards.prob_home("A", "B"))
        p_backward = np.full(20, backwards.prob_home("A", "B"))

        ll_forward = ratings.score_predictions(y, p_forward)["log_loss"]
        ll_backward = ratings.score_predictions(y, p_backward)["log_loss"]

        self.assertGreater(forwards.ratings["A"], 0.0)
        self.assertLess(backwards.ratings["A"], 0.0)
        # A wide, not marginal, separation: ~0.40 against ~2.00.
        self.assertLess(ll_forward, ll_backward)
        self.assertAlmostEqual(ll_forward, 0.40170, places=4)
        self.assertAlmostEqual(ll_backward, 1.99500, places=4)

    def test_fit_predict_elo_is_immune_because_it_always_sorts(self):
        # Feeding the reversed window to the real entry point is a no-op, so
        # this class has to walk EloRating directly. Asserted so that if the
        # sort is ever removed, this test says why it mattered.
        window = self._window()
        forwards = ratings.fit_predict_elo(window, home_advantage=0.0, use_margin=False)
        backwards = ratings.fit_predict_elo(
            window.iloc[::-1].reset_index(drop=True), home_advantage=0.0, use_margin=False
        )
        self.assertAlmostEqual(forwards.ratings["A"], backwards.ratings["A"], places=12)
        self.assertAlmostEqual(forwards.ratings["B"], backwards.ratings["B"], places=12)


class TestWithinDateOrderIsIrrelevant(unittest.TestCase):
    """Shuffling the order of players within a date cannot move a number.

    A real schedule has several games on a date. Which of them is processed
    first is an implementation detail, and if it leaked into the output then the
    "same" dataset would produce different predictions on different runs. The
    sort key is `(date, home, away)`: the second and third components are
    arbitrary but total and deterministic, so row order in the input is
    discarded.
    """

    def _schedule(self):
        rows = [
            ("Cleveland Cavaliers", "Washington Wizards", 1),
            ("Miami Heat", "Boston Celtics", 0),
            ("Los Angeles Lakers", "Dallas Mavericks", 1),
            ("Indiana Pacers", "Toronto Raptors", 0),
            ("Phoenix Suns", "Miami Heat", 1),
        ]
        return canonical(rows, dates=["2024-11-05"] * 5)

    def test_predictions_are_identical_under_many_same_date_shuffles(self):
        games = self._schedule()
        baseline = ratings.fit_predict_elo(games, home_advantage=70.0, use_margin=False)
        rng = np.random.default_rng(7)
        for _ in range(50):
            shuffled = games.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
            other = ratings.fit_predict_elo(shuffled, home_advantage=70.0, use_margin=False)
            # Same games in the same internally-determined order.
            pd.testing.assert_frame_equal(
                baseline.games.sort_values(["home", "away"]).reset_index(drop=True),
                other.games.sort_values(["home", "away"]).reset_index(drop=True),
            )

    def test_shuffling_does_not_change_the_sorted_output_order(self):
        games = self._schedule()
        baseline = ratings.fit_predict_elo(games, home_advantage=70.0, use_margin=False)
        shuffled = ratings.fit_predict_elo(
            games.sample(frac=1.0, random_state=99), home_advantage=70.0, use_margin=False
        )
        self.assertEqual(list(baseline.games["home"]), list(shuffled.games["home"]))
        self.assertEqual(list(baseline.games["away"]), list(shuffled.games["away"]))

    def test_games_are_ordered_by_date_first(self):
        games = canonical(
            [("A", "Z", 1), ("A", "B", 0)],
            dates=["2024-03-01", "2024-01-01"],
        )
        fit = ratings.fit_predict_elo(games, use_margin=False)
        # The later-dated game was played second, so the early game is row 0.
        self.assertEqual(list(fit.games["date"].dt.strftime("%Y-%m-%d")), ["2024-01-01", "2024-03-01"])

    def test_shuffling_across_dates_is_also_irrelevant(self):
        games = self._schedule()
        dates = ["2024-11-05", "2024-11-06", "2024-11-07", "2024-11-08", "2024-11-09"]
        staggered = canonical(
            list(zip(games["home"], games["away"], games["home_win"])), dates=dates
        )
        base = ratings.fit_predict_elo(staggered, use_margin=False)
        other = ratings.fit_predict_elo(
            staggered.sample(frac=1.0, random_state=3), use_margin=False
        )
        np.testing.assert_allclose(base.prob_home, other.prob_home)


class TestMarginOfVictoryScalesTheUpdate(unittest.TestCase):
    """A 20-point win must move ratings more than a 1-point win.

    The formula is FiveThirtyEight's:
    `mov_mult = (|margin| / max(points_per_elo * |diff|, mov_floor)) ** 0.5`,
    clamped. With the two teams level the denominator hits the 1.0 floor, so

        1-point win   -> (1 / 1) ** 0.5 = 1.000
        20-point win  -> (20 / 1) ** 0.5 = 4.472, clamped to mov_max = 2.5

    The clamp is the other half of the assertion: without it a single 40-point
    blowout between evenly-matched teams would move ratings 6.3x a normal step,
    and one game would dominate a season.
    """

    def test_multiplier_is_larger_for_a_bigger_win(self):
        elo = ratings.EloRating(ratings.EloConfig(home_advantage=0.0))
        one = elo.margin_multiplier(1.0, 0.0)
        twenty = elo.margin_multiplier(20.0, 0.0)
        self.assertAlmostEqual(one, 1.0, places=12)
        self.assertAlmostEqual(twenty, ratings.EloConfig().mov_max, places=12)
        self.assertGreater(twenty, one)

    def test_multiplier_grows_monotonically_with_the_margin(self):
        elo = ratings.EloRating(ratings.EloConfig(home_advantage=0.0))
        values = [elo.margin_multiplier(float(m), 0.0) for m in range(1, 31)]
        self.assertEqual(values, sorted(values))

    def test_a_blowout_wins_move_ratings_further_than_a_narrow_win(self):
        # Both games start from identical, level ratings and identical
        # predictions, and both are won by the home team. The only difference
        # is the margin, so any difference in the update is the multiplier's.
        config = ratings.EloConfig(home_advantage=0.0, k=20.0)
        narrow = canonical([("A", "B", 1)], dates=["2024-01-01"], margins=[1.0])
        blowout = canonical([("A", "B", 1)], dates=["2024-01-01"], margins=[25.0])
        delta_narrow = ratings.fit_predict_elo(narrow, config=config).games.loc[0, "delta"]
        delta_blowout = ratings.fit_predict_elo(blowout, config=config).games.loc[0, "delta"]
        self.assertEqual(delta_narrow, 10.0)  # 20 * 1.0 * (1 - 0.5)
        self.assertAlmostEqual(delta_blowout, 20.0 * 2.5 * 0.5, places=12)
        self.assertGreater(delta_blowout, delta_narrow)

    def test_an_upset_is_damped_relative_to_an_expected_result(self):
        """The downward half of the multiplier, which is the half that matters.

        A one-point win between level teams is routine, so the multiplier is
        exactly 1.0. The same one-point win in a game the ratings had at a
        200-point advantage is a huge upset: the expected margin is
        0.0325 * 200 = 6.5 points, so the multiplier is
        (1 / 6.5) ** 0.5 = 0.3921, and the favoured team's rating should barely
        move.

        This is the assertion that would fail if ``mov_min`` were left at 1.0 --
        a lower clamp of 1.0 forbids the multiplier from ever falling, which
        silently disables the entire reason for dividing by the expected
        margin.
        """
        elo = ratings.EloRating(ratings.EloConfig(home_advantage=0.0))
        routine = elo.margin_multiplier(1.0, 0.0)      # expected margin hits the floor
        upset = elo.margin_multiplier(1.0, 200.0)      # expected margin is 6.5
        self.assertAlmostEqual(routine, 1.0, places=12)
        self.assertAlmostEqual(upset, (1.0 / 6.5) ** 0.5, places=12)
        self.assertAlmostEqual(upset, 0.3922323, places=7)
        self.assertLess(upset, routine)
        # A heavy favourite winning comfortably is *reinforced* the same way.
        self.assertGreater(elo.margin_multiplier(30.0, 200.0), routine)

    def test_multiplier_is_never_below_the_lower_clamp(self):
        elo = ratings.EloRating(ratings.EloConfig(home_advantage=0.0))
        for margin in (1.0, 2.0, 5.0):
            for diff in (0.0, 100.0, 400.0, 1000.0):
                self.assertGreaterEqual(
                    elo.margin_multiplier(margin, diff), ratings.EloConfig().mov_min
                )

    def test_multiplier_is_one_when_disabled_or_margin_missing(self):
        config = ratings.EloConfig(home_advantage=0.0, mov_multiplier=False)
        elo = ratings.EloRating(config)
        self.assertEqual(elo.margin_multiplier(40.0, 0.0), 1.0)
        enabled = ratings.EloRating(ratings.EloConfig(home_advantage=0.0))
        self.assertEqual(enabled.margin_multiplier(float("nan"), 0.0), 1.0)

    def test_multiplier_never_exceeds_the_clamp(self):
        elo = ratings.EloRating(ratings.EloConfig(home_advantage=0.0))
        for margin in (5.0, 30.0, 60.0, 500.0):
            self.assertLessEqual(elo.margin_multiplier(margin, 0.0), ratings.EloConfig().mov_max)

    def test_multiplier_denominator_uses_the_pre_update_difference(self):
        """The denominator must not be able to see the game's own result.

        The same level ratings give multiplier 2.5 for a 25-point win. If the
        implementation recomputed the difference *after* the update it would be
        working from a moved rating and produce a different number for the same
        inputs, which is the leak in miniature.
        """
        elo = ratings.EloRating(ratings.EloConfig(home_advantage=0.0))
        self.assertAlmostEqual(elo.margin_multiplier(25.0, 0.0), 2.5, places=12)
        elo.ratings["A"] = 10.0  # A has already been moved by something else
        elo.ratings["B"] = -10.0
        # Now the expected margin is 0.0325 * 20 = 0.65, floored to 1.0, so the
        # 25-point win is still 5 ** 0.5 clamped to 2.5. Deterministic either way.
        self.assertAlmostEqual(elo.margin_multiplier(25.0, elo.effective_diff("A", "B")), 2.5, places=12)

    def test_games_without_a_margin_still_fit_and_use_multiplier_one(self):
        # The 17k-row dataset carries the game total and the winner but not the
        # score split, so most rows have no recoverable margin. Elo must run
        # without them rather than refusing.
        games = canonical([("A", "B", 1), ("B", "A", 0)], dates=["2024-01-01", "2024-01-02"])
        fit = ratings.fit_predict_elo(games, home_advantage=0.0, use_margin=True)
        self.assertEqual(list(fit.games["mov_multiplier"]), [1.0, 1.0])
        self.assertEqual(fit.prob_home[0], 0.5)


class TestRatingSumIsPreserved(unittest.TestCase):
    """Elo updates are antisymmetric, so the league average never drifts.

    Adding `delta` to one team and subtracting it from the other leaves the sum
    over all teams unchanged. That is what keeps home-court advantage from
    being silently absorbed into whichever team happened to be measured
    against, and it is a property worth asserting rather than assuming.
    """

    def test_sum_is_invariant_across_a_long_random_run(self):
        rng = np.random.default_rng(31337)
        teams = [f"T{i}" for i in range(10)]
        rows = []
        for i in range(500):
            home, away = rng.choice(teams, 2, replace=False)
            rows.append((home, away, int(rng.random() < 0.5)))
        games = canonical(rows, dates=[f"2024-{1 + i // 250:02d}-{1 + i % 28:02d}" for i in range(500)])
        margins = rng.integers(1, 40, 500).astype(float)
        games["margin"] = margins
        fit = ratings.fit_predict_elo(games, home_advantage=70.0, use_margin=True)
        self.assertAlmostEqual(sum(fit.ratings.values()), 0.0, places=9)
        # And the update really was the antisymmetric pair it claims to be.
        self.assertAlmostEqual(
            fit.games.loc[0, "rating_home_after"] - fit.games.loc[0, "rating_home_before"],
            fit.games.loc[0, "delta"],
            places=12,
        )

    def test_season_reversion_and_decay_also_preserve_the_sum(self):
        rows = [("A", "B", 1), ("A", "B", 1), ("A", "B", 0)]
        games = canonical(rows, dates=["2024-01-01", "2024-01-02", "2025-01-02"])
        fit = ratings.fit_predict_elo(
            games,
            home_advantage=0.0,
            use_margin=False,
            config=ratings.EloConfig(
                home_advantage=0.0, season_reversion=0.5, decay_half_life_days=100.0
            ),
        )
        self.assertAlmostEqual(sum(fit.ratings.values()), 0.0, places=9)


class TestSeasonCarryOver(unittest.TestCase):
    """Ratings shrink at a season boundary but do not reset to zero.

    `nba_season` labels October onwards with the current year and January to
    September with the previous one, so 2024-10-22 is season 2024 and 2024-04-10
    is season 2023. `season_reversion` scales every rating when the label
    changes.
    """

    def test_season_labels_are_the_starting_year(self):
        self.assertEqual(ratings.nba_season(pd.Timestamp("2024-10-22")), 2024)
        self.assertEqual(ratings.nba_season(pd.Timestamp("2025-01-15")), 2024)
        self.assertEqual(ratings.nba_season(pd.Timestamp("2023-04-10")), 2022)
        self.assertEqual(ratings.nba_season(pd.Timestamp("2023-06-30")), 2022)

    def test_carryover_scales_ratings_at_the_boundary(self):
        # Game 1 (season 2023) leaves A at +10 from a single 20-point K win
        # with level ratings. Game 2 is 300 days later, season 2024, so A must
        # enter game 2 at 0.5 * 10 = 5.
        games = canonical(
            [("A", "B", 1), ("B", "A", 0)],
            dates=["2023-11-01", "2024-11-01"],
        )
        fit = ratings.fit_predict_elo(
            games,
            use_margin=False,
            config=ratings.EloConfig(
                home_advantage=0.0, season_reversion=0.5, decay_half_life_days=None
            ),
        )
        self.assertEqual(fit.games.loc[0, "season"], 2023)
        self.assertEqual(fit.games.loc[1, "season"], 2024)
        self.assertEqual(fit.games.loc[0, "rating_home_after"], 10.0)
        self.assertEqual(fit.games.loc[1, "rating_home_before"], -5.0)
        self.assertEqual(fit.games.loc[1, "rating_away_before"], 5.0)

    def test_no_carryover_between_dates_inside_one_season(self):
        games = canonical(
            [("A", "B", 1), ("B", "A", 0)],
            dates=["2024-01-05", "2024-01-10"],
        )
        fit = ratings.fit_predict_elo(
            games,
            use_margin=False,
            config=ratings.EloConfig(home_advantage=0.0, season_reversion=0.5),
        )
        self.assertEqual(fit.games.loc[1, "rating_home_before"], -10.0)

    def test_reversion_of_one_is_a_no_op(self):
        games = canonical(
            [("A", "B", 1), ("B", "A", 0)],
            dates=["2023-11-01", "2024-11-01"],
        )
        fit = ratings.fit_predict_elo(
            games,
            use_margin=False,
            config=ratings.EloConfig(home_advantage=0.0, season_reversion=1.0),
        )
        self.assertEqual(fit.games.loc[1, "rating_home_before"], -10.0)

    def test_season_key_can_be_a_column_name(self):
        games = canonical(
            [("A", "B", 1), ("B", "A", 0)], dates=["2024-01-05", "2024-01-10"]
        )
        games["my_season"] = ["2024-25", "2024-25"]
        fit = ratings.fit_predict_elo(
            games,
            use_margin=False,
            config=ratings.EloConfig(home_advantage=0.0, season_key="my_season"),
        )
        self.assertEqual(list(fit.games["season"]), ["2024-25", "2024-25"])


class TestDecay(unittest.TestCase):
    """A rating from four years ago should not still be arguing.

    With `decay_half_life_days = 30`, a rating multiplied by `0.5 ** (30/30)`
    halves after 30 days and is a twentieth of itself after 130 days. The point
    is recency, not a cosmetic smoothing: it stops a 2012 rating from
    dominating a 2025 one even when no roster change is detectable from
    win/loss alone.
    """

    def test_ratings_halve_after_exactly_one_half_life(self):
        games = canonical(
            [("A", "B", 1), ("B", "A", 0)],
            dates=["2024-01-01", "2024-01-31"],
        )
        fit = ratings.fit_predict_elo(
            games,
            use_margin=False,
            config=ratings.EloConfig(
                home_advantage=0.0, season_reversion=1.0, decay_half_life_days=30.0
            ),
        )
        # 30 days elapsed, so B enters game 2 at -10 * 0.5 = -5.
        self.assertEqual(fit.games.loc[1, "rating_home_before"], -5.0)
        self.assertEqual(fit.games.loc[1, "rating_away_before"], 5.0)

    def test_long_gap_erases_a_rating(self):
        games = canonical(
            [("A", "B", 1), ("B", "A", 0)],
            dates=["2015-01-01", "2025-01-01"],
        )
        fit = ratings.fit_predict_elo(
            games,
            use_margin=False,
            config=ratings.EloConfig(
                home_advantage=0.0, season_reversion=1.0, decay_half_life_days=30.0
            ),
        )
        # 3652 days is 121.7 half-lives: 2 ** -121.7 is ~3e-37.
        self.assertEqual(fit.games.loc[1, "prob_home"], 0.5)
        self.assertAlmostEqual(fit.games.loc[1, "rating_home_before"], 0.0, places=30)

    def test_same_day_games_are_not_decayed(self):
        games = canonical(
            [("A", "B", 1), ("B", "A", 0)], dates=["2024-01-01", "2024-01-01"]
        )
        fit = ratings.fit_predict_elo(
            games,
            use_margin=False,
            config=ratings.EloConfig(
                home_advantage=0.0, season_reversion=1.0, decay_half_life_days=30.0
            ),
        )
        self.assertEqual(fit.games.loc[1, "rating_home_before"], -10.0)

    def test_decay_disabled_by_default(self):
        self.assertIsNone(ratings.EloConfig().decay_half_life_days)


class TestRestAdjustment(unittest.TestCase):
    """Rest is game context, so it belongs in the expectation.

    A well-rested team is a better bet for tonight and not a permanently better
    team. The module keeps rest out of the stored ratings and puts it into the
    pre-game difference instead, which means it feeds the update indirectly:
    a team that wins *while rested* is less surprising, so its rating moves
    less. That is the desired behaviour -- crediting a rest advantage to skill
    is how a rating system starts to believe its own injuries -- but it does
    mean the stored ratings differ between a rested schedule and a level one,
    which is why the assertion at the bottom is about the direction of the
    effect rather than about equality.
    """

    def test_rest_edge_shifts_the_prediction(self):
        config = ratings.EloConfig(home_advantage=0.0, rest_coefficient=3.0)
        elo = ratings.EloRating(config)
        self.assertEqual(elo.effective_diff("A", "B"), 0.0)
        self.assertEqual(elo.effective_diff("A", "B", days_rest_home=4, days_rest_away=1), 9.0)
        self.assertEqual(elo.effective_diff("A", "B", days_rest_home=1, days_rest_away=4), -9.0)

    def test_missing_rest_is_ignored_not_treated_as_zero(self):
        elo = ratings.EloRating(ratings.EloConfig(home_advantage=0.0))
        self.assertEqual(elo.effective_diff("A", "B", days_rest_home=None, days_rest_away=None), 0.0)
        self.assertEqual(elo.effective_diff("A", "B", days_rest_home=5.0, days_rest_away=None), 0.0)

    def test_rest_damps_the_update_for_a_rested_winner(self):
        """Two schedules, same games, opposite rest edges.

        Rested: the home team wins while rested, so the win was expected and
        the rating moves less. Level: no rest information, so the same win was
        more surprising and the rating moves more. Both predictions are also
        ordered the same way, which is the part a user actually sees.
        """
        rows = [("A", "B", 1), ("B", "A", 0)]
        dates = ["2024-01-01", "2024-01-02"]
        rested = canonical(rows, dates=dates, rest=np.array([[4.0, 1.0], [1.0, 4.0]]))
        level = canonical(rows, dates=dates, rest=np.array([[2.0, 2.0], [2.0, 2.0]]))
        a = ratings.fit_predict_elo(rested, use_rest=True)
        b = ratings.fit_predict_elo(level, use_rest=True)

        # Rested home team is predicted to win more often...
        self.assertGreater(a.prob_home[0], b.prob_home[0])
        # ...but its rating gains less from doing so.
        self.assertLess(a.games.loc[0, "delta"], b.games.loc[0, "delta"])
        self.assertLess(a.ratings["A"], b.ratings["A"])

    def test_rest_is_off_unless_requested(self):
        games = canonical(
            [("A", "B", 1)],
            dates=["2024-01-01"],
            rest=np.array([[7.0, 1.0]]),
        )
        without = ratings.fit_predict_elo(games, use_rest=False, home_advantage=0.0)
        self.assertEqual(without.prob_home[0], 0.5)
        with_rest = ratings.fit_predict_elo(games, use_rest=True, home_advantage=0.0)
        self.assertGreater(with_rest.prob_home[0], 0.5)


class TestNormalizeGames(unittest.TestCase):
    """Column mapping for this repository's two table shapes.

    The important behaviour is the refusal. `dataset_2012-26` carries both
    `Date` (a prior-snapshot date) and `GameDate` (the real one), and a rating
    system fed the snapshot date processes the season out of order with nothing
    in the output to show for it. Silently picking the wrong one is the failure
    this guards.
    """

    def test_reads_dataset_shaped_columns(self):
        raw = pd.DataFrame(
            {
                "TEAM_NAME": ["Cleveland Cavaliers", "Miami Heat"],
                "TEAM_NAME.1": ["Washington Wizards", "Boston Celtics"],
                "GameDate": ["2012-10-30", "2012-10-30"],
                "Home-Team-Win": [1.0, 0.0],
                "Score": [178.0, 227.0],
                "Days-Rest-Home": [10.0, 10.0],
                "Days-Rest-Away": [3.0, 3.0],
            }
        )
        games = ratings.normalize_games(raw)
        self.assertEqual(list(games["home"]), ["Cleveland Cavaliers", "Miami Heat"])
        self.assertEqual(list(games["away"]), ["Washington Wizards", "Boston Celtics"])
        self.assertEqual(list(games["home_win"]), [1.0, 0.0])
        self.assertEqual(list(games["days_rest_away"]), [3.0, 3.0])
        # A dataset row has the total and the winner but not the score split.
        self.assertTrue(bool(games["margin"].isna().all()))

    def test_reads_odds_shaped_columns(self):
        raw = pd.DataFrame(
            {
                "Date": ["2023-10-24"],
                "Home": ["Denver Nuggets"],
                "Away": ["Los Angeles Lakers"],
                "Points": [226],
                "Win_Margin": [12],
                "Days_Rest_Home": [7],
                "Days_Rest_Away": [7],
            }
        )
        games = ratings.normalize_games(raw)
        self.assertEqual(list(games["home_win"]), [1.0])
        self.assertEqual(list(games["margin"]), [12.0])

    def test_outcome_is_derived_from_the_margin_when_no_label_exists(self):
        raw = pd.DataFrame(
            {
                "Date": ["2023-10-24", "2023-10-25"],
                "Home": ["Denver Nuggets", "Golden State Warriors"],
                "Away": ["Los Angeles Lakers", "Phoenix Suns"],
                "Win_Margin": [12, -4],
            }
        )
        games = ratings.normalize_games(raw)
        self.assertEqual(list(games["home_win"]), [1.0, 0.0])

    def test_refuses_the_snapshot_date_when_the_real_one_is_available(self):
        raw = pd.DataFrame(
            {
                "TEAM_NAME": ["A"],
                "TEAM_NAME.1": ["B"],
                "Date": ["2024-11-04"],
                "GameDate": ["2024-11-05"],
                "Home-Team-Win": [1.0],
            }
        )
        with self.assertRaises(KeyError) as ctx:
            ratings.normalize_games(raw, date_col="Date")
        self.assertIn("GameDate", str(ctx.exception))

    def test_prefers_game_date_over_date_automatically(self):
        raw = pd.DataFrame(
            {
                "TEAM_NAME": ["A"],
                "TEAM_NAME.1": ["B"],
                "Date": ["2024-11-04"],
                "GameDate": ["2024-11-05"],
                "Home-Team-Win": [1.0],
            }
        )
        games = ratings.normalize_games(raw)
        self.assertEqual(games.loc[0, "date"], "2024-11-05")

    def test_aliases_merge_relocations_without_leaking(self):
        raw = pd.DataFrame(
            {
                "TEAM_NAME": ["LA Clippers", "Los Angeles Clippers"],
                "TEAM_NAME.1": ["Utah Jazz", "Utah Jazz"],
                "GameDate": ["2024-11-05", "2014-11-05"],
                "Home-Team-Win": [1.0, 1.0],
            }
        )
        merged = ratings.normalize_games(raw)
        self.assertEqual(list(merged["home"]), ["LA Clippers", "LA Clippers"])
        verbatim = ratings.normalize_games(raw, alias=None)
        self.assertEqual(
            list(verbatim["home"]), ["LA Clippers", "Los Angeles Clippers"]
        )

    def test_every_alias_pair_occupies_disjoint_eras(self):
        """Guards the merge itself: if two names in an alias pair were ever
        live at the same time, merging them would be a data bug, not history."""
        eras = {
            "Charlotte Bobcats": ("2012-11-02", "2014-10-29"),
            "Charlotte Hornets": ("2014-11-01", "2026-01-07"),
            "New Jersey Nets": ("2012-11-03", "2012-11-03"),
            "Brooklyn Nets": ("2012-11-05", "2026-01-07"),
            "New Orleans Hornets": ("2012-10-31", "2013-10-30"),
            "New Orleans Pelicans": ("2013-11-01", "2026-01-07"),
            "Los Angeles Clippers": ("2012-10-31", "2015-10-29"),
            "LA Clippers": ("2015-10-31", "2026-01-07"),
        }
        for old, new in ratings.DEFAULT_TEAM_ALIASES.items():
            old_start, old_end = eras[old]
            new_start, new_end = eras[new]
            self.assertLess(old_end, new_start, f"{old} overlaps {new}")

    def test_rows_with_no_recorded_result_are_dropped(self):
        # The odds tables store Points == 0 and Win_Margin == 0 for games whose
        # result was never scraped. Reading those as 0-0 games would add eight
        # bogus ties.
        raw = pd.DataFrame(
            {
                "Date": ["2023-10-24", "2024-03-28", "2024-03-28"],
                "Home": ["Denver Nuggets", "Atlanta Hawks", "New Orleans Pelicans"],
                "Away": ["Los Angeles Lakers", "Boston Celtics", "Milwaukee Bucks"],
                "Points": [226, 0, 0],
                "Win_Margin": [12, 0, 0],
            }
        )
        games = ratings.normalize_games(raw)
        self.assertEqual(len(games), 1)
        self.assertEqual(games.loc[0, "home"], "Denver Nuggets")

    def test_require_margin_refuses_a_frame_that_has_none(self):
        raw = pd.DataFrame(
            {
                "TEAM_NAME": ["A"],
                "TEAM_NAME.1": ["B"],
                "GameDate": ["2024-11-05"],
                "Home-Team-Win": [1.0],
                "Score": [220.0],
            }
        )
        with self.assertRaises(ValueError) as ctx:
            ratings.normalize_games(raw, require_margin=True)
        self.assertIn("margin", str(ctx.exception))
        # Without the requirement it is allowed through, with a NaN margin.
        self.assertEqual(len(ratings.normalize_games(raw)), 1)

    def test_missing_required_columns_are_reported_by_name(self):
        with self.assertRaises(KeyError):
            ratings.normalize_games(pd.DataFrame({"foo": [1]}))

    def test_round_trips_through_fit_predict_elo(self):
        # Three real rows from odds_2023-24_new, home advantage switched off so
        # the multiplier's denominator falls to the 1.0 floor and the numbers
        # are checkable by hand:
        #   margin +12 -> (12/1) ** 0.5 = 3.4641 -> clamped to mov_max 2.5
        #   margin  -4 -> (4/1)  ** 0.5 = 2.0000 (inside both clamps)
        #   margin +23 -> (23/1) ** 0.5 = 4.7958 -> clamped to 2.5
        raw = pd.DataFrame(
            {
                "Date": ["2023-10-24", "2023-10-25", "2023-10-26"],
                "Home": ["Denver Nuggets", "Golden State Warriors", "Indiana Pacers"],
                "Away": ["Los Angeles Lakers", "Phoenix Suns", "Washington Wizards"],
                "Points": [226, 212, 263],
                "Win_Margin": [12, -4, 23],
            }
        )
        fit = ratings.fit_predict_elo(
            ratings.normalize_games(raw), home_advantage=0.0, use_margin=True
        )
        self.assertEqual(len(fit.games), 3)
        multipliers = list(fit.games["mov_multiplier"])
        for got, expected in zip(multipliers, [2.5, 2.0, 2.5]):
            self.assertAlmostEqual(got, expected, places=12)


class TestBradleyTerry(unittest.TestCase):
    """The batch control for the Elo ordering guarantee.

    Bradley-Terry sees every game at once, so it has no ordering to get wrong.
    That makes it the right instrument for two checks: that the ordering
    guarantee is doing real work in Elo, and that the solver finds the true
    maximum rather than stopping somewhere plausible.
    """

    def test_perfectly_mirrored_data_gives_exactly_zero_strengths(self):
        # A home win and the mirror-image away win, same team winning both
        # times. There is no strength difference to find, so with the home term
        # pinned the two strengths are exactly zero.
        #
        # The home term cannot be pinned this way. Mirrored results carry no
        # information about home court at all -- the likelihood is flat in
        # `hca` -- so its unpenalised MLE is unbounded. That is a property of
        # the data, not of the solver, and it is why `l2` exists; asserted
        # separately below.
        games = canonical(
            [("A", "B", 1), ("B", "A", 1)], dates=["2024-01-01", "2024-01-02"]
        )
        fit = ratings.fit_bradley_terry(games, fit_home_advantage=False, l2=0.0)
        self.assertAlmostEqual(fit.strengths["A"], 0.0, places=12)
        self.assertAlmostEqual(fit.strengths["B"], 0.0, places=12)
        self.assertAlmostEqual(fit.prob_home("A", "B"), 0.5, places=12)

    def test_mirrored_data_leaves_home_advantage_unidentified(self):
        # Documented above: perfectly mirrored results say nothing about home
        # court, so the likelihood is flat in that direction and the unpenalised
        # iteration walks off along it -- from 11.2 at ten iterations to 24.2
        # at sixty, still climbing. `converged` reports stationarity, which a
        # flat direction satisfies, so the flag alone is not enough; the ridge
        # is what keeps the parameter finite.
        games = canonical(
            [("A", "B", 1), ("B", "A", 1)], dates=["2024-01-01", "2024-01-02"]
        )
        short = ratings.fit_bradley_terry(games, l2=0.0, max_iter=10)
        long = ratings.fit_bradley_terry(games, l2=0.0, max_iter=60)
        penalised = ratings.fit_bradley_terry(games, l2=1e-3, max_iter=60)
        # Still running away, and reporting that it has stopped.
        self.assertGreater(long.home_advantage, short.home_advantage)
        self.assertTrue(long.converged)
        # The prior pulls it back to something finite and sane.
        self.assertLess(penalised.home_advantage, long.home_advantage)
        self.assertLess(abs(penalised.home_advantage), 10.0)

    def test_a_team_that_wins_every_game_rates_above_one_that_loses_every_game(self):
        games = canonical(
            [("A", "B", 1)] * 30 + [("B", "A", 0)] * 30,
            dates=[f"2024-{1 + i // 28:02d}-{1 + i % 28:02d}" for i in range(60)],
        )
        fit = ratings.fit_bradley_terry(games, l2=1e-6)
        self.assertGreater(fit.strengths["A"], fit.strengths["B"])
        self.assertGreater(fit.prob_home("A", "B"), 0.99)
        self.assertLess(fit.prob_home("B", "A"), 0.01)

    def test_strengths_sum_to_zero(self):
        rng = np.random.default_rng(4242)
        teams = [f"T{i}" for i in range(8)]
        rows = []
        for _ in range(2000):
            home, away = rng.choice(teams, 2, replace=False)
            rows.append((home, away, int(rng.random() < 0.6)))
        games = canonical(rows, dates=["2024-01-01"] * 2000)
        fit = ratings.fit_bradley_terry(games)
        self.assertAlmostEqual(sum(fit.strengths.values()), 0.0, places=10)

    def test_fit_is_completely_order_independent(self):
        """The batch fit's defining property, asserted rather than described."""
        rng = np.random.default_rng(11)
        teams = [f"T{i}" for i in range(6)]
        rows = []
        for _ in range(1200):
            home, away = rng.choice(teams, 2, replace=False)
            rows.append((home, away, int(rng.random() < 0.55)))
        games = canonical(rows, dates=["2024-01-01"] * 1200)
        straight = ratings.fit_bradley_terry(games)
        for seed in (1, 2, 3):
            shuffled = ratings.fit_bradley_terry(
                games.sample(frac=1.0, random_state=seed)
            )
            for team in straight.teams:
                self.assertAlmostEqual(
                    straight.strengths[team], shuffled.strengths[team], places=12
                )
            self.assertAlmostEqual(
                straight.home_advantage, shuffled.home_advantage, places=12
            )

    def test_home_advantage_is_positive_when_home_wins_more(self):
        rng = np.random.default_rng(5)
        teams = [f"T{i}" for i in range(10)]
        rows = []
        for _ in range(3000):
            home, away = rng.choice(teams, 2, replace=False)
            # 0.65 from home regardless of who it is: a genuine home edge.
            rows.append((home, away, int(rng.random() < 0.65)))
        games = canonical(rows, dates=["2024-01-01"] * 3000)
        fit = ratings.fit_bradley_terry(games)
        self.assertGreater(fit.home_advantage, 0.0)
        self.assertAlmostEqual(fit.home_advantage, math.log(0.65 / 0.35), places=1)

    def test_can_be_pinned_to_no_home_advantage(self):
        games = canonical(
            [("A", "B", 1), ("B", "A", 1)], dates=["2024-01-01", "2024-01-02"]
        )
        fit = ratings.fit_bradley_terry(games, fit_home_advantage=False, l2=0.0)
        self.assertEqual(fit.home_advantage, 0.0)

    def test_solver_converges_to_the_likelihood_maximum(self):
        """The stationarity check that a wrong Hessian would fail.

        At the optimum the score equations hold: the strength gradient is zero.
        Checked numerically here because that is the property a sign error in
        the Hessian destroys while leaving the output looking entirely
        reasonable -- the earlier version of this solver reported a "fit" with
        a home advantage of -1e13 and a gradient norm of 1e3.
        """
        rng = np.random.default_rng(2024)
        teams = [f"T{i}" for i in range(5)]
        truth = {t: rng.normal(0, 0.7) for t in teams}
        rows = []
        for _ in range(2500):
            home, away = rng.choice(teams, 2, replace=False)
            p = 1.0 / (1.0 + np.exp(-(truth[home] - truth[away] + 0.35)))
            rows.append((home, away, int(rng.random() < p)))
        games = canonical(rows, dates=["2024-01-01"] * 2500)
        fit = ratings.fit_bradley_terry(games, l2=0.0, tol=1e-11)
        _SCORES: dict[str, float] = {}

        self.assertTrue(fit.converged, f"did not converge: grad_norm={fit.grad_norm}")
        self.assertLess(fit.grad_norm, 1e-8)
        # Score equations: dL/ds_i = SUM_{i home} (y - p) - SUM_{i away} (y - p)
        # is zero at the optimum, and so is dL/dhca = SUM (y - p). These use
        # the home team's `y` on both sides with the sign flipped for away
        # appearances -- writing them as (1 - p) instead is the easy mistake,
        # and it reports a residual of hundreds on a perfectly converged fit.
        home_score = 0.0
        for _, row in games.iterrows():
            y = float(row["home_win"])
            p = fit.prob_home(row["home"], row["away"])
            home_score += y - p
            for team in (row["home"], row["away"]):
                contribution = (y - p) if row["home"] == team else -(y - p)
                _SCORES[team] = _SCORES.get(team, 0.0) + contribution
        for team, score in _SCORES.items():
            self.assertAlmostEqual(score, 0.0, places=7, msg=f"score equation for {team}")
        self.assertAlmostEqual(home_score, 0.0, places=7, msg="home-court score equation")

    def test_recovers_known_strengths_in_odds_units(self):
        """A large synthetic league should come back within sampling error.

        Six teams with strengths spread over about +/-1.5 in log-odds and 6000
        games. Sampling noise on ~1000 games per team puts the standard error
        per strength at roughly 0.1, so 0.35 is a generous bar that a wrong
        sign, a wrong identification or a missing cross term cannot clear.
        """
        rng = np.random.default_rng(99)
        teams = [f"T{i}" for i in range(6)]
        truth = np.linspace(-1.5, 1.5, 6)
        index = {t: i for i, t in enumerate(teams)}
        rows = []
        for _ in range(6000):
            home, away = rng.choice(6, 2, replace=False)
            p = 1.0 / (1.0 + np.exp(-(truth[home] - truth[away] + 0.4)))
            rows.append((teams[home], teams[away], int(rng.random() < p)))
        games = canonical(rows, dates=["2024-01-01"] * 6000)
        fit = ratings.fit_bradley_terry(games, l2=0.0)
        for team in teams:
            self.assertAlmostEqual(fit.strengths[team], truth[index[team]], delta=0.35)
        self.assertAlmostEqual(fit.home_advantage, 0.4, delta=0.15)

    def test_rejects_an_unknown_team(self):
        games = canonical([("A", "B", 1), ("B", "A", 1)], dates=["2024-01-01", "2024-01-02"])
        with self.assertRaises(KeyError):
            ratings.fit_bradley_terry(games, teams=["A"])

    def test_predictions_on_a_frozen_fit_are_repeatable(self):
        games = canonical(
            [("A", "B", 1)] * 10 + [("B", "C", 1)] * 10,
            dates=[f"2024-01-{1 + i:02d}" for i in range(20)],
        )
        fit = ratings.fit_bradley_terry(games)
        first = [fit.prob_home("A", "C") for _ in range(5)]
        second = [fit.prob_home("A", "C") for _ in range(5)]
        self.assertEqual(first, second)


class TestSRS(unittest.TestCase):
    """The point-differential system.

    SRS predicts a margin, not a win probability, so two things have to be
    right for it to be usable: the offense/defense signs (a big win must raise
    offense *and* lower defense) and a calibration that turns a margin into a
    probability.
    """

    def _league(self, teams=4, reps=3):
        """A round robin of `teams` teams, `reps` games per ordered pairup.

        Margins come from a genuine additive SRS league -- separate offense and
        defense parameters -- plus noise, so there is an additive model to
        recover.

        `teams=4` or more is required for that to be possible. In a three-team
        round robin the system ``o[h] - d[a] = m`` has a two-dimensional null
        space, not one, so the offense/defense split is not identified and no
        solver can recover the generated values. Measured here: rank 5 of 6
        parameters for three teams against rank 11 of 12 for six.
        """
        names = [chr(ord("A") + i) for i in range(teams)]
        offense = {t: v for t, v in zip(names, np.linspace(6.0, -6.0, teams))}
        defense = {t: v for t, v in zip(names, np.linspace(-5.0, 5.0, teams))}
        rng = np.random.default_rng(2024)
        rows, mg = [], []
        for _ in range(reps):
            for home in names:
                for away in names:
                    if home == away:
                        continue
                    margin = offense[home] - defense[away] + rng.uniform(-2.0, 2.0)
                    rows.append((home, away, int(margin > 0)))
                    mg.append(margin)
        return canonical(rows, dates=season_dates(len(rows)), margins=mg)

    def test_defense_is_positive_is_good_and_ordered(self):
        """`defense` is points ALLOWED above average, so higher = worse.

        This is the sign convention that a reader has to know to interpret the
        column at all. In the fixture A is the best team: highest offense AND
        most negative defense.
        """
        fit = ratings.fit_srs(self._league())
        offense = fit.offense
        defense = fit.defense
        best_offense = max(offense, key=offense.get)
        best_defense = min(defense, key=defense.get)
        self.assertEqual(best_offense, best_defense)
        self.assertEqual(best_offense, "A")   # offense +6, defense -5
        self.assertEqual(min(offense, key=offense.get), "D")  # offense -6

    def test_net_rating_is_offense_minus_defense(self):
        """The arithmetic identity, and that the ratings are mean-zero.

        `net` is the conventional SRS number and `defense` is points allowed
        above average, so `net = offense - defense` with a POSITIVE defense
        meaning worse. The two are not negatives of each other -- they are
        separate parameters, and the fit separates them only up to a shared
        constant, which the solver pins to zero.
        """
        fit = ratings.fit_srs(self._league())
        for team in fit.offense:
            self.assertAlmostEqual(
                fit.net_rating[team],
                fit.offense[team] - fit.defense[team],
                places=12,
                msg=team,
            )
        self.assertAlmostEqual(sum(fit.offense.values()), 0.0, places=10)
        self.assertAlmostEqual(sum(fit.defense.values()), 0.0, places=10)

    def test_net_and_matchup_forms_are_different_expressions(self):
        """The distinction SRSResult's docstring claims, on a 6-team league.

        Four teams are not enough to pin the offense/defense split: in a
        four-team round robin the system ``o[h] - d[a] = m`` still has more than
        one null direction, so the recovered net rating order can invert. Six
        teams -- rank 11 of 12 parameters, one null direction -- is the smallest
        fixture here that recovers the generated ordering.

        The two predictor forms -- net-rating difference and offense-only
        difference -- do not agree, which is the reason the convention is stated
        in the docstring rather than left to the reader.
        """
        games = self._league(teams=6, reps=3)
        fit = ratings.fit_srs(games)

        net = fit.net_rating
        self.assertEqual(
            sorted(net, key=net.get, reverse=True),
            ["A", "B", "C", "D", "E", "F"],
        )
        for team in fit.teams:
            self.assertAlmostEqual(
                net[team], fit.offense[team] - fit.defense[team], places=12
            )
            # 3 reps x (6-1) home games + 3 reps x (6-1) away games.
            self.assertEqual(fit.games_played[team], 30)
        self.assertAlmostEqual(sum(fit.offense.values()), 0.0, places=10)
        # The league mean margin is removed with the ratings, not re-added.
        self.assertAlmostEqual(fit.home_advantage, 0.0, places=12)

        self.assertNotAlmostEqual(
            fit.predicted_margin("A", "B"),
            fit.predicted_matchup_margin("A", "B"),
            places=6,
        )

    def test_symmetric_league_leaves_only_the_home_advantage(self):
        # Every team wins half its games, so every rating is zero and both
        # predictor forms reduce to the home advantage -- which is the correct
        # answer, not a degenerate one.
        games = canonical(
            [("A", "B", 1), ("B", "A", 1), ("A", "C", 1), ("C", "A", 1),
             ("B", "C", 1), ("C", "B", 1)],
            dates=[f"2024-01-{i + 1:02d}" for i in range(6)],
            margins=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        )
        fit = ratings.fit_srs(games)
        for team in fit.teams:
            self.assertAlmostEqual(fit.net_rating[team], 0.0, places=9)
        # Both forms collapse to the explicit intercept.
        self.assertAlmostEqual(fit.predicted_margin("A", "B"), 0.0, places=9)
        self.assertAlmostEqual(fit.predicted_matchup_margin("A", "B"), 0.0, places=9)
        with_hca = ratings.fit_srs(games, home_advantage=3.0)
        self.assertAlmostEqual(with_hca.predicted_margin("A", "B"), 3.0, places=9)
        self.assertAlmostEqual(with_hca.predicted_matchup_margin("A", "B"), 3.0, places=9)

    def test_weakest_team_is_negative_and_strongest_positive(self):
        # Ratings are mean-zero by construction, so in a league where one team
        # is clearly best and one clearly worst the ends must straddle zero.
        fit = ratings.fit_srs(self._league())
        net = fit.net_rating
        self.assertGreater(net["A"], 0.0)
        self.assertLess(net["D"], 0.0)
        self.assertGreater(net["A"], net["B"])
        self.assertGreater(net["B"], net["C"])
        self.assertGreater(net["C"], net["D"])

    def test_regression_to_the_mean_shrinks_a_one_game_record(self):
        """The regression is what makes it a rating and not an average.

        Team Z wins its only game by 50. Divided by one game that would be a
        +50 rating; regressed at 1/games it is halved every game, so after four
        games of a realistic margin it is nowhere near +50.
        """
        games = canonical(
            [("Z", "A", 1), ("Z", "A", 1), ("Z", "A", 1), ("Z", "A", 1)],
            dates=["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            margins=[50.0, 2.0, 2.0, 2.0],
        )
        fit = ratings.fit_srs(games)
        self.assertEqual(fit.games_played["Z"], 4)
        self.assertLess(fit.net_rating["Z"], 50.0 / 4)

    def test_single_game_is_centred_to_zero(self):
        """One game, A home B, A wins by 12.

        The league mean margin is 12 -- it is the only game there is -- so the
        scoring deviation is zero and every rating is zero. That is the
        mean-centring working, not a failure: a one-game league has no relative
        strength to measure. Only the home advantage survives.
        """
        games = canonical([("A", "B", 1)], dates=["2024-01-01"], margins=[12.0])
        fit = ratings.fit_srs(games)
        for team in ("A", "B"):
            self.assertAlmostEqual(fit.offense[team], 0.0, places=9)
            self.assertAlmostEqual(fit.net_rating[team], 0.0, places=9)
        # The home edge is REMOVED with the league mean rather than re-added, so
        # the default intercept is 0.0 and the predicted margin is 0 too.
        self.assertAlmostEqual(fit.home_advantage, 0.0, places=12)
        self.assertAlmostEqual(fit.predicted_margin("A", "B"), 0.0, places=9)
        # Passing it explicitly puts it back, exactly once.
        self.assertAlmostEqual(
            ratings.fit_srs(games, home_advantage=12.0).predicted_margin("A", "B"),
            12.0,
            places=9,
        )

    def test_home_advantage_defaults_to_zero_not_the_mean_margin(self):
        """The league mean margin is already removed by the centring.

        Subtracting the league mean removes the home edge as well -- the mean NBA
        margin is about +2.0 points precisely because home teams win more often
        -- so defaulting the intercept to mean(margin) counts it twice. On real
        data that double count inflated predicted margins by ~2 points and drove
        the calibration to a log loss of 13.3 before it was fixed.
        """
        games = canonical(
            [("A", "B", 1), ("B", "A", 1)], dates=["2024-01-01", "2024-01-02"],
            margins=[12.0, 18.0],
        )
        self.assertAlmostEqual(float(np.mean([12.0, 18.0])), 15.0, places=9)
        self.assertAlmostEqual(ratings.fit_srs(games).home_advantage, 0.0, places=12)
        self.assertAlmostEqual(
            ratings.fit_srs(games, home_advantage=2.5).home_advantage, 2.5, places=12
        )

    def test_margin_scale_turns_a_margin_into_a_probability(self):
        games = self._league()
        fit = ratings.fit_srs(games)
        margin = fit.predicted_margin("A", "B")
        expected = 1.0 / (1.0 + np.exp(-(fit.margin_scale * (margin - fit.margin_center))))
        self.assertAlmostEqual(fit.prob_home("A", "B"), expected, places=12)

    def test_calibration_is_finite_and_positive(self):
        """The calibration slope must be a usable number.

        Three calibrations each looked plausible and were wrong, and the guard
        is against all of them: calibrating on the REALISED margin (which
        determines the label exactly in this data, so the slope runs to 1.26 and
        saturates the sigmoid), calibrating on the in-sample fitted margin (about
        2x too steep), and letting the Newton step escape to |k| ~ 1e12 when the
        Hessian underflows. A positive finite slope, with probabilities inside
        the unit interval, rules out every one of them.
        """
        rng = np.random.default_rng(4711)
        teams = [f"T{i}" for i in range(8)]
        # Persistent per-team strength and an ADDITIVE margin, so there is an
        # additive model to recover. The outcome is the margin's sign, as in the
        # real data (sign(margin) == home_win on 3067/3067 odds rows). Drawing
        # the outcome independently instead produces a league whose ratings are
        # uncorrelated with its results, and the honest out-of-sample
        # calibration then correctly returns a slope near zero.
        power = {t: v for t, v in zip(teams, rng.normal(0, 1.5, len(teams)))}
        rows, margins = [], []
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                for _ in range(8):
                    margin = 2.0 + (power[home] - power[away]) + rng.uniform(-2.0, 2.0)
                    rows.append((home, away, int(margin > 0)))
                    margins.append(margin)
        games = canonical(
            rows,
            dates=season_dates(len(rows)),
            margins=margins,
        )
        # Interleave, as a real season is. The construction above groups games by
        # matchup, which would put the calibration's time-ordered holdout inside
        # a single pair and leave it with nothing to predict.
        games = games.sample(frac=1.0, random_state=7).reset_index(drop=True)
        fit = ratings.fit_srs(games)
        probs = np.asarray([fit.prob_home(h, a) for h, a in zip(games["home"], games["away"])])
        # Realisable, and not near either of the two failure slopes.
        self.assertGreater(fit.margin_scale, 0.0)
        self.assertTrue(np.isfinite(fit.margin_scale))
        # Bounded away from the two failure values, without pinning an exact one.
        self.assertLess(fit.margin_scale, 2.0)
        self.assertGreater(float(np.std(probs)), 1e-3)

    def test_better_predicted_margin_gives_a_higher_probability(self):
        """A rating recovers the ordering of a league with known strengths.

        Six teams on a linspace from -6 to +6, every pairing six times,
        interleaved so the calibration's time-ordered holdout is not one matchup.

        The margin must be ADDITIVE -- ``margin = offense[h] - defense[a]`` -- or
        there is no additive model to recover. So the margin is drawn as
        ``base + strength[h] - strength[a]``, which is what SRS assumes, and the
        outcome is its sign, matching the real data where
        ``sign(margin) == Home-Team-Win`` on 2986/2986 merged rows.

        An earlier version of this fixture used ``base + 0.5 * |gap|``, which is
        a function of the *absolute* gap and so is not additive at all; the
        rating correctly returned a spread of 0.7 where the truth was 12.
        """
        rng = np.random.default_rng(808)
        teams = [f"T{i}" for i in range(6)]
        # Separate offense and defense parameters. SRS models a margin as
        # `offense[home] - defense[away]`, so a fixture built from a single
        # "strength" per team is an offense-only league -- and it is not
        # recoverable, because the two readings of the same margin differ by a
        # constant. Generating `margin = o[home] - d[away]` directly is the
        # league SRS is actually specified for.
        offense = {t: v for t, v in zip(teams, np.linspace(-3.0, 3.0, 6))}
        defense = {t: v for t, v in zip(teams, np.linspace(2.0, -2.0, 6))}
        rows, margins = [], []
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                for _ in range(6):
                    margin = (
                        offense[home] - defense[away] + rng.uniform(-1.0, 1.0)
                    )
                    rows.append((home, away, int(margin > 0)))
                    margins.append(margin)
        games = canonical(rows, dates=season_dates(len(rows)), margins=margins)
        games = games.sample(frac=1.0, random_state=3).reset_index(drop=True)

        fit = ratings.fit_srs(games)
        # Offense must come back as generated: T5 best at +3, T0 worst at -3.
        ordered = sorted(fit.offense, key=fit.offense.get, reverse=True)
        self.assertEqual(ordered, ["T5", "T4", "T3", "T2", "T1", "T0"])
        for stronger, weaker in zip(ordered, ordered[1:]):
            self.assertGreater(fit.offense[stronger], fit.offense[weaker])
        self.assertAlmostEqual(fit.offense["T5"], 3.0, delta=0.6)
        self.assertAlmostEqual(fit.offense["T0"], -3.0, delta=0.6)
        # Defense comes back reversed, as generated: T0 +2, T5 -2.
        self.assertAlmostEqual(fit.defense["T5"], -2.0, delta=0.6)
        self.assertAlmostEqual(fit.defense["T0"], 2.0, delta=0.6)
        # The probability must increase with the predicted margin. On this
        # fixture the margins determine the winner deterministically, so the
        # maximum-likelihood slope is steep and both ends saturate at 0 and 1
        # -- correct behaviour for that league, and the reason the real-data
        # fit is checked separately. So the MARGIN ordering is asserted here,
        # and the probability ordering is asserted only where it is strict.
        self.assertGreater(fit.predicted_margin("T5", "T0"), fit.predicted_margin("T5", "T3"))
        self.assertGreater(fit.predicted_margin("T5", "T3"), fit.predicted_margin("T0", "T5"))
        self.assertGreaterEqual(fit.prob_home("T5", "T3"), fit.prob_home("T0", "T5"))
        self.assertGreater(fit.margin_scale, 0.0)
        self.assertTrue(np.isfinite(fit.margin_scale))
        for team in teams:
            for other in teams:
                if team != other:
                    self.assertTrue(0.0 <= fit.prob_home(team, other) <= 1.0)

    def test_probability_increases_with_predicted_margin(self):
        """The monotonicity, on a league where nothing saturates.

        A league whose margins do NOT determine the winner gives a realistic
        slope. Eight teams, twelve games per ordered pairup, uniform noise
        large enough that the better team loses about a third of the time.
        """
        rng = np.random.default_rng(4242)
        teams = [f"T{i}" for i in range(8)]
        offense = {t: v for t, v in zip(teams, rng.normal(0, 2.0, 8))}
        defense = {t: v for t, v in zip(teams, rng.normal(0, 2.0, 8))}
        rows, margins = [], []
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                for _ in range(12):
                    margin = offense[home] - defense[away] + rng.uniform(-8.0, 8.0)
                    rows.append((home, away, int(margin > 0)))
                    margins.append(margin)
        fit = ratings.fit_srs(canonical(rows, dates=season_dates(len(rows)), margins=margins))
        # Probabilities span a usable range and are strictly ordered in margin.
        probs = [
            fit.prob_home(h, a)
            for h in teams
            for a in teams
            if h != a
        ]
        self.assertGreater(float(np.std(probs)), 0.05)
        self.assertGreater(min(probs), 1e-3)
        self.assertLess(max(probs), 1.0 - 1e-3)
        pairs = sorted(
            (
                (fit.predicted_margin(h, a), fit.prob_home(h, a))
                for h in teams
                for a in teams
                if h != a
            )
        )
        for (m_lo, p_lo), (m_hi, p_hi) in zip(pairs, pairs[1:]):
            if m_hi > m_lo:
                self.assertGreaterEqual(p_hi, p_lo)

    def test_needs_a_margin_and_says_so(self):
        games = canonical([("A", "B", 1)], dates=["2024-01-01"]).drop(columns=["margin"])
        with self.assertRaises(KeyError):
            ratings.fit_srs(games)

    def test_a_frame_of_all_nan_margins_is_rejected(self):
        # Different failure from the one above: the column is there but carries
        # no information. Either way it must raise rather than fit zeroes.
        games = canonical([("A", "B", 1), ("B", "A", 1)], dates=["2024-01-01", "2024-01-02"])
        self.assertTrue(bool(games["margin"].isna().all()))
        with self.assertRaises(ValueError):
            ratings.fit_srs(games)

    def test_drops_games_without_a_margin(self):
        # The second game has no margin, so C appears only there and is never
        # fitted at all -- the roster is derived from the games that survive.
        games = canonical(
            [("A", "B", 1), ("B", "C", 1)],
            dates=["2024-01-01", "2024-01-02"],
            margins=[10.0, np.nan],
        )
        fit = ratings.fit_srs(games)
        self.assertEqual(set(fit.teams), {"A", "B"})
        self.assertEqual(fit.games_played["A"], 1)
        self.assertEqual(fit.games_played["B"], 1)
        self.assertNotIn("C", fit.offense)


class TestScorePredictions(unittest.TestCase):
    """The four comparison numbers, worked out by hand.

    Every system in the comparison is scored by this one function, so a bug
    here would shift all of them together and be invisible.
    """

    def test_two_row_case_is_exact(self):
        # y = [1, 0], p = [0.8, 0.4]
        # log loss = -(log 0.8 + log 0.6) / 2 = (0.2231436 + 0.5108256) / 2
        metrics = ratings.score_predictions([1, 0], [0.8, 0.4])
        self.assertAlmostEqual(metrics["log_loss"], 0.3669846, places=7)
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["brier"], 0.1, places=12)
        self.assertAlmostEqual(metrics["auc"], 1.0, places=12)

    def test_a_coin_flip_costs_log_two(self):
        # -log(0.5) = log(2) = 0.6931472 for every game, whatever the outcome.
        for y in ([1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 1]):
            metrics = ratings.score_predictions(y, [0.5] * 4)
            self.assertAlmostEqual(metrics["log_loss"], math.log(2.0), places=12)
            self.assertAlmostEqual(metrics["brier"], 0.25, places=12)

    def test_inverted_ranking_scores_auc_of_zero(self):
        metrics = ratings.score_predictions([1, 0], [0.2, 0.8])
        self.assertAlmostEqual(metrics["auc"], 0.0, places=12)
        self.assertEqual(metrics["accuracy"], 0.0)

    def test_ties_are_ranked_by_average_rank(self):
        # p = [0.5, 0.5] for y = [1, 0]: the two games are indistinguishable, so
        # each gets rank 1.5 and AUC is exactly 0.5.
        metrics = ratings.score_predictions([1, 0], [0.5, 0.5])
        self.assertAlmostEqual(metrics["auc"], 0.5, places=12)

    def test_auc_agrees_with_sklearn_on_a_random_case(self):
        # A cross-check, not a source of expected values: the Mann-Whitney
        # form computed above must match an independent implementation.
        from sklearn.metrics import roc_auc_score

        rng = np.random.default_rng(17)
        for _ in range(20):
            y = rng.integers(0, 2, 250)
            if len(np.unique(y)) < 2:
                continue
            p = np.round(rng.random(250), 2)  # force ties
            self.assertAlmostEqual(
                ratings.score_predictions(y, p)["auc"],
                float(roc_auc_score(y, p)),
                places=12,
            )

    def test_single_class_outcomes_leave_auc_undefined(self):
        metrics = ratings.score_predictions([1, 1, 1], [0.5, 0.6, 0.7])
        self.assertIsNone(metrics["auc"])
        self.assertIsNotNone(metrics["log_loss"])

    def test_empty_input_is_handled(self):
        metrics = ratings.score_predictions([], [])
        self.assertEqual(metrics, {"log_loss": None, "accuracy": None, "auc": None, "brier": None})


class TestLogisticStability(unittest.TestCase):
    """The sigmoid must not overflow at the extremes.

    `1 / (1 + exp(-x))` returns exactly 0.0 once `exp(-x)` exceeds the largest
    double, which happens below `x = -710`. A confidently-wrong prediction then
    makes a log loss of infinity rather than a large finite number. Games do
    produce rating differences that large once a team has been dominant for a
    decade, so this is a live path.

    At `x = -720`: `exp(720) = 5.5e312`, past the 1.8e308 ceiling, so the naive
    form overflows to `inf` and returns 0.0. The sign-split form returns
    `exp(-720) = 2.0e-313`, which is representable.
    """

    def test_large_negative_argument_does_not_return_zero(self):
        # The naive form, with the overflow it provokes, suppressed so the
        # suite stays warning-clean.
        with np.errstate(over="ignore"):
            self.assertEqual(1.0 / (1.0 + np.exp(720.0)), 0.0)
        value = float(ratings.logistic(-720.0))
        self.assertGreater(value, 0.0)
        self.assertAlmostEqual(value, float(np.exp(-720.0)), places=320)

    def test_large_positive_argument_saturates_at_one(self):
        self.assertAlmostEqual(float(ratings.logistic(800.0)), 1.0, places=15)

    def test_matches_the_naive_form_in_the_safe_range(self):
        for x in (-700.0, -30.0, -1.0, 0.0, 1.0, 30.0, 700.0):
            self.assertAlmostEqual(
                float(ratings.logistic(x)), 1.0 / (1.0 + np.exp(-x)), places=14
            )

    def test_is_vectorised(self):
        values = np.asarray(ratings.logistic(np.array([-720.0, 0.0, 800.0])))
        self.assertEqual(values.shape, (3,))
        self.assertGreater(values[0], 0.0)
        self.assertEqual(values[1], 0.5)


class TestInputValidation(unittest.TestCase):
    """Errors that name the problem instead of returning a wrong number."""

    def test_empty_games_is_rejected(self):
        empty = canonical([], dates=[])
        with self.assertRaises(ValueError):
            ratings.fit_predict_elo(empty)
        with self.assertRaises(ValueError):
            ratings.fit_bradley_terry(empty)
        with self.assertRaises(ValueError):
            ratings.fit_srs(empty)

    def test_missing_columns_are_named(self):
        frame = pd.DataFrame({"home": ["A"], "away": ["B"]})
        with self.assertRaises(KeyError) as ctx:
            ratings.fit_predict_elo(frame)
        self.assertIn("date", str(ctx.exception))

    def test_unparseable_dates_are_rejected(self):
        games = canonical([("A", "B", 1), ("B", "A", 0)], dates=["2024-01-01", "2024-01-02"])
        games["date"] = ["not a date", "also not a date"]
        with self.assertRaises(ValueError):
            ratings.fit_predict_elo(games)

    def test_string_dates_are_parsed_with_the_supplied_format(self):
        games = canonical([("A", "B", 1), ("B", "A", 0)], dates=["2024-01-01", "2024-01-02"])
        games["date"] = ["01/02/2024", "02/01/2024"]
        fit = ratings.fit_predict_elo(
            games, date_format="%d/%m/%Y", home_advantage=0.0, use_margin=False
        )
        self.assertEqual(fit.games.loc[0, "date"], pd.Timestamp("2024-01-02"))

    def test_bad_season_key_type_is_reported(self):
        games = canonical([("A", "B", 1)], dates=["2024-01-01"])
        with self.assertRaises(TypeError):
            ratings.fit_predict_elo(games, config=ratings.EloConfig(season_key=17))

    def test_unknown_season_column_is_reported(self):
        games = canonical([("A", "B", 1)], dates=["2024-01-01"])
        with self.assertRaises(KeyError):
            ratings.fit_predict_elo(games, config=ratings.EloConfig(season_key="nope"))

    def test_nba_season_rejects_a_non_date(self):
        with self.assertRaises(TypeError):
            ratings.nba_season("2024-01-01")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()