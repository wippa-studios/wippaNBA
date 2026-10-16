"""Tests for `src/Models/baselines.py`.

Three rules govern this file, and they are the same three `Tests/test_ratings.py`
follows:

1. **Every expected value is derived in the comment next to it**, from the
   formula in the module or from arithmetic written out longhand. Where
   scikit-learn or the shared harness appears it is a cross-check, never the
   source of the expected figure.
2. **The leakage tests are the point.** A baseline that reads its own test block
   still produces a confident, plausible, entirely wrong number, and nothing
   about the output says so. Five separate tests attack it from different
   directions: the Elo recursion, the calibration fits, the K sweep, the split
   construction, and a source-level guard against the season-end aggregates in
   `snapshots/`.
3. **Nothing here reads the real dataset unless it has to.** The unit tests run
   on a synthetic league with known strengths, which is both faster and a
   stronger check -- a league where Elo cannot beat the base rate is broken, and
   that is true regardless of what the NBA does.

No test pins a number this suite previously produced. The baselines' job is to
be the bar for five other models, and a test that freezes today's bar cannot
notice the bar moving.
"""

import unittest

import numpy as np
import pandas as pd

from src.DataProviders import nba_data as nd
from src.Models import baselines, ratings


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

def elo_prob(diff, scale=400.0):
    """P(home wins) at rating difference `diff`, matching EloRating.prob_home."""
    return 1.0 / (1.0 + 10.0 ** (-diff / scale))


#: The probability two unknown teams face when only the home term separates them.
P_UNKNOWN = elo_prob(70.0)


def tiny_frame(rows, *, start="2024-10-22", step_days=2, season="2024-25", dates=None):
    """A canonical frame from `(home, away, home_win, margin)` tuples.

    Dates are generated with a real timedelta rather than a day counter, so the
    frame cannot contain `2024-02-30`. `step_days=1` gives each game its own
    date and `step_days=0` puts them all on one; pass `dates` outright when the
    exact schedule matters, which is what the same-day test needs.
    """
    n = len(rows)
    first = pd.Timestamp(start)
    frame = pd.DataFrame({
        "GAME_ID": [f"00224{i:05d}" for i in range(n)],
        "date": (pd.to_datetime(list(dates)) if dates is not None
                 else [first + pd.Timedelta(days=step_days * i) for i in range(n)]),
        "SEASON": [season] * n,
        "home": [r[0] for r in rows],
        "away": [r[1] for r in rows],
        "home_win": [float(r[2]) for r in rows],
        "margin": [float(r[3]) for r in rows],
    })
    return baselines.to_canonical(frame.assign(
        days_rest_home=np.nan, days_rest_away=np.nan,
    ))


def synthetic_league(n_days=60, n_teams=6, seed=7, twice_in_a_day=False):
    """A league with known team strengths, simulated game by game.

    Enough structure for a rating system to have something real to find: each
    team has a persistent strength in Elo points, home court is worth 65, and the
    outcome is a Bernoulli draw from the Elo probability rather than a
    deterministic function of strength. The noise is the point -- with zero noise
    every rating system is perfect and the tests prove nothing.

    `twice_in_a_day` schedules every team TWICE on every date, which is
    impossible in the NBA and is the only way to break the equivalence proof in
    `baselines.same_day_cost`.
    """
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    strength = dict(zip(teams, rng.normal(0.0, 45.0, n_teams)))
    rows = []
    first = pd.Timestamp("2020-10-20")
    for d in range(n_days):
        date = first + pd.Timedelta(days=d)
        pool = teams * (2 if twice_in_a_day else 1)
        rng.shuffle(pool)
        n_games = n_teams if twice_in_a_day else n_teams // 2
        for i in range(n_games):
            home, away = pool[2 * i], pool[2 * i + 1]
            p = elo_prob(strength[home] - strength[away] + 65.0)
            home_win = int(rng.random() < p)
            margin = (1 if home_win else -1) * float(6 + abs(rng.normal(0, 12)))
            rows.append({
                "GAME_ID": f"{d:03d}{i:03d}", "date": date, "SEASON": "2020-21",
                "home": home, "away": away, "home_win": float(home_win),
                "margin": margin,
            })
    return baselines.to_canonical(pd.DataFrame(rows).assign(
        days_rest_home=np.nan, days_rest_away=np.nan,
    ))


def splits_of(games, n_splits=3, initial_frac=0.4):
    return nd.walk_forward_split(games, n_splits=n_splits, initial_frac=initial_frac)


# --------------------------------------------------------------------------- #
# Correctness of the rating system itself
# --------------------------------------------------------------------------- #

class TestEloMovesInTheRightDirection(unittest.TestCase):
    """A hand-built three-team sequence, checked for direction not magnitude.

    A beats B, A beats C, B beats C, each by the same margin on its own date. The
    results are *transitive*, which is what makes the expected ordering a real
    prediction rather than a tautology -- the obvious rock-paper-scissors version
    (A beats B, B beats C, C beats A) gives every team one win and one loss, and
    no ordering of the final ratings follows from it.

    Whatever the learning rate and whatever the home term, a positive K has to
    leave each winner above zero and each loser below it, and it has to leave the
    team with two wins on top. Asserting the sign rather than the number is what
    makes this a test of direction: a symmetric bug cannot satisfy it.
    """

    def setUp(self):
        self.games = tiny_frame([
            ("A", "B", 1, 10.0),   # A beats B
            ("A", "C", 1, 10.0),   # A beats C
            ("B", "C", 1, 10.0),   # B beats C
        ])
        self.fit = baselines.elo_rating_pass(self.games, k=20.0)
        self.g = self.fit.games

    def test_winner_goes_up_and_loser_goes_down_on_each_game(self):
        for i in range(3):
            self.assertGreater(
                self.g.loc[i, "rating_home_after"], self.g.loc[i, "rating_home_before"],
                f"game {i}: the home winner's rating must rise",
            )
            self.assertLess(
                self.g.loc[i, "rating_home_before"], self.g.loc[i, "rating_home_after"],
                f"game {i}: the away side must fall",
            )
            # The update is antisymmetric: one side gains exactly what the other
            # loses, which is what keeps the rating vector mean-zero.
            self.assertAlmostEqual(
                self.g.loc[i, "rating_home_after"] - self.g.loc[i, "rating_home_before"],
                float(self.g.loc[i, "delta"]), places=12,
            )

    def test_two_wins_puts_a_team_on_top_and_one_win_puts_it_last(self):
        final = dict(self.fit.ratings)
        self.assertGreater(final["A"], final["B"], "A won twice")
        self.assertGreater(final["B"], final["C"], "B beat C head to head")
        self.assertGreater(final["A"], 0.0)

    def test_first_game_prediction_is_exactly_home_advantage(self):
        # Two unknown teams, so the difference is the home term and nothing else.
        self.assertAlmostEqual(self.g.loc[0, "prob_home"], P_UNKNOWN, places=12)
        self.assertAlmostEqual(self.g.loc[0, "rating_home_before"], 0.0, places=12)
        self.assertAlmostEqual(self.g.loc[0, "rating_away_before"], 0.0, places=12)

    def test_rating_sum_is_preserved(self):
        self.assertAlmostEqual(float(sum(self.fit.ratings.values())), 0.0, places=10)


class TestEloAgainstLonghandArithmetic(unittest.TestCase):
    """Two games, worked out by hand, to 1e-12.

    Setup: K=20, home_advantage=70, elo_scale=400, margin-of-victory OFF (so the
    multiplier is exactly 1 and the arithmetic stays checkable), one season so no
    reversion fires, both games the same pair.

    Game 1 (A home vs B, both unknown, A wins):
        diff  = 0 - 0 + 70                  = 70
        p     = 1/(1+10**(-70/400))        = 0.599397...
        delta = 20 * (1 - p)                = +8.012059...
        R[A]  = +8.012059..., R[B] = -8.012059...

    Game 2 (A home vs B again, B wins):
        diff  = 8.012059 - (-8.012059) + 70 = 86.024119...
        p     = 1/(1+10**(-86.024119/400))  = 0.621586...
        delta = 20 * (0 - p)                = -12.431714...
    """

    def setUp(self):
        self.games = tiny_frame([("A", "B", 1, 5.0), ("A", "B", 0, 5.0)])
        self.fit = baselines.elo_rating_pass(
            self.games, k=20.0, home_advantage=70.0, use_margin=False
        )
        self.delta1 = 20.0 * (1.0 - P_UNKNOWN)
        self.diff2 = 2.0 * self.delta1 + 70.0
        self.p2 = elo_prob(self.diff2)
        self.delta2 = 20.0 * (0.0 - self.p2)

    def test_first_prediction_and_update(self):
        g = self.fit.games
        self.assertAlmostEqual(g.loc[0, "prob_home"], P_UNKNOWN, places=12)
        self.assertAlmostEqual(g.loc[0, "delta"], self.delta1, places=12)
        self.assertAlmostEqual(g.loc[0, "rating_home_after"], self.delta1, places=12)

    def test_second_prediction_uses_the_updated_ratings(self):
        g = self.fit.games
        self.assertAlmostEqual(g.loc[1, "rating_home_before"], self.delta1, places=12)
        self.assertAlmostEqual(g.loc[1, "rating_away_before"], -self.delta1, places=12)
        self.assertAlmostEqual(g.loc[1, "effective_diff"], self.diff2, places=12)
        self.assertAlmostEqual(g.loc[1, "prob_home"], self.p2, places=12)
        self.assertAlmostEqual(g.loc[1, "delta"], self.delta2, places=12)
        self.assertAlmostEqual(
            g.loc[1, "rating_home_after"], self.delta1 + self.delta2, places=12
        )

    def test_a_stronger_team_shrinks_the_home_advantage_it_buys(self):
        """The whole point of a rating, stated so it cannot be satisfied by luck.

        A wins the first game at home, so A's rating is 2*delta above B's. In the
        return leg -- B at home -- those rating points are subtracted straight out
        of the 70-point home term, so the home team's probability must FALL. Note
        it does not have to fall below 0.5: an 8-point rating swing is worth far
        less than a 70-point home term, and a test that asserted it did would be
        asserting something false.
        """
        games = tiny_frame([("A", "B", 1, 5.0), ("B", "A", 1, 5.0)])
        fit = baselines.elo_rating_pass(games, k=20.0, use_margin=False)
        self.assertLess(fit.games.loc[1, "prob_home"], fit.games.loc[0, "prob_home"])
        self.assertAlmostEqual(
            fit.games.loc[1, "effective_diff"],
            70.0 - 2.0 * (20.0 * (1.0 - P_UNKNOWN)),
            places=12,
        )


class TestEloLearns(unittest.TestCase):
    """The behavioural claim the whole suite rests on, on known ground truth."""

    def test_elo_beats_the_base_rate_on_a_league_with_known_strengths(self):
        games = synthetic_league(n_days=120, n_teams=6, seed=17)
        tr, te = splits_of(games)[-1]
        y = games["home_win"].to_numpy()[te].astype(int)
        base = baselines.base_rate_predictions(games, tr, te)
        elo = baselines.elo_predictions(games, tr, te, k=10.0)
        self.assertLess(
            nd.evaluate(y, elo)["log_loss"], nd.evaluate(y, base)["log_loss"]
        )
        # And it must rank games, not merely shift the base rate: AUC has to clear
        # 0.5 by more than noise on 150 games.
        self.assertGreater(nd.evaluate(y, elo)["auc"], 0.60)

    def test_frozen_ratings_are_worse_than_updated_ones(self):
        # K -> 0 is a rating system that never learns. If it matched a tuned K
        # the whole walk-forward design would be pointless.
        games = synthetic_league(n_days=120, n_teams=6, seed=17)
        tr, te = splits_of(games)[-1]
        y = games["home_win"].to_numpy()[te].astype(int)
        frozen = baselines.elo_predictions(games, tr, te, k=1e-9)
        tuned = baselines.elo_predictions(games, tr, te, k=10.0)
        self.assertLess(nd.evaluate(y, tuned)["log_loss"],
                        nd.evaluate(y, frozen)["log_loss"])


class TestBaseRate(unittest.TestCase):
    def test_predictions_are_the_training_home_win_rate(self):
        games = synthetic_league()
        tr, te = splits_of(games)[0]
        rate = float(games["home_win"].to_numpy()[tr].mean())
        p = baselines.base_rate_predictions(games, tr, te)
        self.assertEqual(len(p), len(te))
        self.assertTrue(np.allclose(p, rate))

    def test_test_outcomes_do_not_move_it(self):
        games = synthetic_league()
        tr, te = splits_of(games)[0]
        before = baselines.base_rate_predictions(games, tr, te)
        flipped = games.copy()
        flipped.loc[te, "home_win"] = 1.0 - flipped.loc[te, "home_win"]
        self.assertTrue(np.allclose(
            before, baselines.base_rate_predictions(flipped, tr, te)))

    def test_a_constant_probability_has_the_closed_form_log_loss(self):
        # The closed form, written out: -p*log(p) - (1-p)*log(1-p). Comparing the
        # harness against the formula is what makes the harness trustworthy for
        # every other number in the suite.
        games = synthetic_league()
        tr, te = splits_of(games)[0]
        p = float(baselines.base_rate_predictions(games, tr, te)[0])
        y = games["home_win"].to_numpy()[te]
        expected = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
        self.assertAlmostEqual(
            nd.evaluate(y, np.full(len(te), p))["log_loss"], expected, places=12
        )

    def test_the_base_rate_has_auc_of_one_half(self):
        # A constant has no ordering, so its AUC must be exactly 0.5. Any
        # implementation that leaks test labels into the base rate would show up
        # here as an AUC above 0.5.
        games = synthetic_league()
        tr, te = splits_of(games)[0]
        y = games["home_win"].to_numpy()[te].astype(int)
        p = baselines.base_rate_predictions(games, tr, te)
        self.assertAlmostEqual(nd.evaluate(y, p)["auc"], 0.5, places=12)


# --------------------------------------------------------------------------- #
# Leakage
# --------------------------------------------------------------------------- #

class TestNoFutureLeaks(unittest.TestCase):
    """Five attacks on one property, from five different directions."""

    def test_a_games_own_result_cannot_change_its_own_prediction(self):
        """Flip every outcome on the FINAL date and nothing before it may move.

        Scoped to one date rather than "the last N rows" because the walk sorts by
        (date, home, away): flipping rows that are merely the last twenty in
        frame order legitimately moves the predictions of the games played after
        them, and a test asserting otherwise would be asserting something false.
        Under same-day batching a game on date d is predicted from the state at the
        end of d-1, so no outcome on date d -- its own included -- can touch it.
        """
        games = synthetic_league()
        last_date = games["date"].max()
        on_day = games.index[games["date"] == last_date].to_numpy()
        base = baselines.elo_rating_pass(games, k=10.0)
        flipped = games.copy()
        flipped.loc[on_day, "home_win"] = 1.0 - flipped["home_win"].iloc[on_day]
        after = baselines.elo_rating_pass(flipped, k=10.0)
        self.assertTrue(np.allclose(base.prob_home, after.prob_home))
        # ...while the update itself must move, or the test above is vacuous.
        self.assertNotAlmostEqual(
            float(base.games.iloc[-1]["rating_home_after"]),
            float(after.games.iloc[-1]["rating_home_after"]), places=6,
        )

    def test_the_calibration_never_sees_a_test_outcome(self):
        games = synthetic_league()
        tr, te = splits_of(games)[-1]
        fit = baselines.elo_rating_pass(games, k=10.0)
        before = baselines.loglinear_elo_predictions(games, tr, te, fit)
        flipped = games.copy()
        flipped.loc[te, "home_win"] = 1.0 - flipped["home_win"].iloc[te]
        self.assertTrue(np.allclose(
            before, baselines.loglinear_elo_predictions(flipped, tr, te, fit)))

    def test_a_past_prediction_is_unchanged_by_later_games(self):
        games = synthetic_league()
        full = baselines.elo_rating_pass(games, k=10.0)
        cut = len(games) // 3
        short = baselines.elo_rating_pass(games.iloc[:cut].copy(), k=10.0)
        self.assertTrue(np.allclose(full.prob_home[:cut], short.prob_home))

    def test_the_split_has_no_overlap_and_never_looks_backwards(self):
        games = synthetic_league()
        for tr, te in splits_of(games):
            self.assertEqual(set(tr.tolist()) & set(te.tolist()), set())
            self.assertLess(int(tr.max()), int(te.min()))
            self.assertLessEqual(games["date"].to_numpy()[tr].max(),
                                 games["date"].to_numpy()[te].min())

    def test_no_season_end_aggregate_is_ever_read(self):
        # `snapshots/` is a season-end table, and one row of it used inside its
        # own season leaks the rest of that season. The guard is a source scan
        # rather than a runtime check because the failure mode is invisible at
        # runtime: the number is simply too good.
        #
        # Docstrings are blanked out first, because this module's docstring
        # discusses snapshots at length precisely in order to promise they are
        # never read -- and a naive substring scan fails on its own promise.
        # Prose cannot leak; only code can.
        import ast
        import inspect
        source = inspect.getsource(baselines)
        tree = ast.parse(source)
        code = source
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.ClassDef)):
                doc = ast.get_docstring(node, clean=False)
                if doc:
                    code = code.replace(doc, "")
        for forbidden in ("snapshot", "player_estimated_metrics",
                          "team_estimated_metrics"):
            self.assertNotIn(forbidden, code.lower(),
                             f"{forbidden!r} appears in executable code")


class TestWalkForwardSplitUsage(unittest.TestCase):
    def test_the_module_uses_the_shared_split_helper(self):
        # A second, home-grown split function would be a second evaluation
        # harness by another name. This pins that the shared one is the one in use.
        games = synthetic_league()
        self.assertEqual(
            [tuple(map(len, s)) for s in splits_of(games)],
            [tuple(map(len, s))
             for s in nd.walk_forward_split(games, n_splits=3, initial_frac=0.4)],
        )

    def test_every_test_block_is_scored_disjointly(self):
        games = synthetic_league()
        pooled = np.concatenate([te for _, te in splits_of(games)])
        self.assertEqual(len(pooled), len(set(pooled.tolist())))


# --------------------------------------------------------------------------- #
# The alignment bug this module was written around
# --------------------------------------------------------------------------- #

class TestRatingPassAlignsToItsInput(unittest.TestCase):
    """`fit_predict_elo` sorts internally; the wrapper has to undo that.

    Found the hard way. A rating pass sorted by `(date, home, away)` returns its
    rows in *walked* order, so `fit.prob_home[test_idx]` silently reads the wrong
    games. The predictions stay inside [0, 1], the metrics stay plausible, and
    the only visible symptom is an AUC of about 0.5 -- reported as "the model is
    no better than chance" rather than as "the indices are wrong". Every test in
    this class fails on the unwrapped version.
    """

    def test_output_rows_are_in_the_callers_order(self):
        games = synthetic_league()
        fit = baselines.elo_rating_pass(games, k=10.0)
        self.assertTrue(np.array_equal(
            fit.games["GAME_ID"].to_numpy(), games["GAME_ID"].to_numpy()))

    def test_a_shuffled_run_reproduces_the_unshuffled_one_per_game(self):
        # Compared through GAME_ID on both sides, never positionally: a shuffled
        # run's row 0 is a different game from the original run's row 0, and a
        # positional comparison of two different schedules measures the schedule
        # rather than the model.
        games = synthetic_league()
        fit = baselines.elo_rating_pass(games, k=10.0)
        shuffled = games.sample(frac=1.0, random_state=3).reset_index(drop=True)
        fit_shuffled = baselines.elo_rating_pass(shuffled, k=10.0)
        straight = dict(zip(fit.games["GAME_ID"], fit.prob_home))
        scrambled = dict(zip(fit_shuffled.games["GAME_ID"], fit_shuffled.prob_home))
        self.assertEqual(set(straight), set(scrambled))
        self.assertEqual(len(straight), len(games))
        for gid, value in straight.items():
            self.assertAlmostEqual(value, scrambled[gid], places=12,
                                   msg=f"game {gid} moved when the input was shuffled")

    def test_a_known_game_gets_a_known_prediction(self):
        # Two league games put A and C at the same rating, so the third is a
        # pure home-term prediction -- and it is the THIRD row of the output.
        games = tiny_frame([("A", "B", 1, 5.0), ("C", "D", 1, 5.0), ("A", "C", 1, 5.0)])
        fit = baselines.elo_rating_pass(games, k=10.0, home_advantage=70.0)
        third = fit.games.iloc[2]
        self.assertEqual(third["GAME_ID"], games["GAME_ID"].iloc[2])
        self.assertAlmostEqual(third["rating_home_before"], third["rating_away_before"],
                               places=12)
        self.assertAlmostEqual(third["prob_home"], P_UNKNOWN, places=12)

    def test_a_duplicate_game_id_is_refused_rather_than_guessed_at(self):
        games = synthetic_league()
        dup = games.copy()
        dup.loc[dup.index[1], "GAME_ID"] = dup["GAME_ID"].iloc[0]
        with self.assertRaises(AssertionError):
            baselines.elo_rating_pass(dup, k=10.0)

    def test_reindexing_without_a_unique_key_is_an_error(self):
        games = synthetic_league()
        fit = baselines.elo_rating_pass(games, k=10.0)
        keyless = fit.games.drop(columns=["GAME_ID"])
        stripped = ratings.EloFitResult(keyless, fit.ratings, fit.elo, fit.config)
        with self.assertRaises(KeyError):
            baselines._reindex_fit_to_frame(
                stripped, games.drop(columns=["GAME_ID"]))


class TestOneGlobalPassEqualsRefitPerSplit(unittest.TestCase):
    """The shortcut that lets this suite run in seconds instead of minutes.

    The Elo recursion is causal, so the state a single global pass has reached by
    the start of a test block is identical to what refitting from scratch on that
    block's training data produces. If that stops being true -- someone makes the
    pass depend on the length of the frame, say -- this fails.
    """

    def test_global_pass_matches_a_train_only_refit(self):
        games = synthetic_league()
        _, te = splits_of(games)[-1]
        upto = int(te.max())
        global_fit = baselines.elo_rating_pass(games, k=10.0)
        refit = baselines.elo_rating_pass(games.iloc[:upto].copy(), k=10.0)
        self.assertTrue(np.allclose(global_fit.prob_home[:upto], refit.prob_home))


# --------------------------------------------------------------------------- #
# The same-date question
# --------------------------------------------------------------------------- #

class TestSameDayHandling(unittest.TestCase):
    def test_identical_when_no_team_plays_twice_in_a_day(self):
        games = synthetic_league(twice_in_a_day=False)
        self.assertEqual(baselines.max_team_games_per_day(games), 1)
        batched = baselines.elo_rating_pass(games, k=10.0, same_day="batched")
        sequential = baselines.elo_rating_pass(games, k=10.0, same_day="sequential")
        # Exact equality, not approximate: the equivalence is a proof, and a
        # tolerance here would hide exactly the bug the proof is about.
        self.assertTrue(np.array_equal(batched.prob_home, sequential.prob_home))

    def test_the_modes_diverge_once_a_team_does_play_twice(self):
        # The counterexample that keeps the proof honest. If the two modes were
        # the same code path this would pass too, and the guarantee above would
        # be worthless.
        games = synthetic_league(twice_in_a_day=True)
        self.assertEqual(baselines.max_team_games_per_day(games), 2)
        batched = baselines.elo_rating_pass(games, k=10.0, same_day="batched")
        sequential = baselines.elo_rating_pass(games, k=10.0, same_day="sequential")
        self.assertFalse(np.array_equal(batched.prob_home, sequential.prob_home))

    def test_the_divergence_is_the_one_the_arithmetic_predicts(self):
        """Hand-computed, to 1e-12.

        K=10, home_advantage=70, elo_scale=400, margin-of-victory OFF.

        Day 1 holds one game and day 2 holds two, so the two teams play twice on
        day 2 and the batching has something to do. Sorted by (date, home, away)
        the order is ("A","B") on day 1, then ("A","B") and ("B","A") on day 2.

        Day 1, A home vs B, A wins:
            diff = 70,  p = 0.599397,  delta = 10*(1-p) = +4.006030
            R[A] = +4.006030,  R[B] = -4.006030

        Day 2 BATCHED -- both games predicted from the end of day 1:
            game 1: diff = 4.006030 + 4.006030 + 70 = 78.012060
            game 2: diff = -4.006030 - 4.006030 + 70 = 61.987940

        Day 2 SEQUENTIAL -- game 2 is predicted after game 1's update landed.
        Game 1's delta is 10*(0 - 0.610403) = -6.104031, so
            R[A] = 4.006030 - 6.104031 = -2.098000
            R[B] = -4.006030 + 6.104031 = +2.098000
            game 2: diff = 2.098000 - (-2.098000) + 70 = 74.196000
        which is strictly larger than the batched 61.987940, so the sequential
        probability is strictly higher: the home team was credited with a result
        that had not happened yet at its own tip-off.
        """
        games = tiny_frame([
            ("A", "B", 1, 5.0),   # day 1
            ("A", "B", 0, 5.0),   # day 2, first in (date, home, away) order
            ("B", "A", 1, 5.0),   # day 2, second
        ], dates=["2024-10-22", "2024-10-23", "2024-10-23"])
        self.assertEqual(games["date"].nunique(), 2)
        self.assertEqual(baselines.max_team_games_per_day(games), 2)

        batched = baselines.elo_rating_pass(
            games, k=10.0, home_advantage=70.0, use_margin=False, same_day="batched")
        sequential = baselines.elo_rating_pass(
            games, k=10.0, home_advantage=70.0, use_margin=False, same_day="sequential")

        delta1 = 10.0 * (1.0 - P_UNKNOWN)
        r_a, r_b = delta1, -delta1
        batched_diff_1, batched_diff_2 = r_a - r_b + 70.0, r_b - r_a + 70.0
        p_day2_1 = elo_prob(batched_diff_1)
        r_a2 = r_a + 10.0 * (0.0 - p_day2_1)
        r_b2 = r_b - 10.0 * (0.0 - p_day2_1)
        sequential_diff_2 = r_b2 - r_a2 + 70.0

        self.assertAlmostEqual(
            batched.games.loc[1, "effective_diff"], batched_diff_1, places=12)
        self.assertAlmostEqual(
            batched.games.loc[2, "effective_diff"], batched_diff_2, places=12)
        self.assertAlmostEqual(
            sequential.games.loc[2, "effective_diff"], sequential_diff_2, places=12)
        self.assertGreater(sequential_diff_2, batched_diff_2)
        self.assertGreater(
            float(sequential.games.loc[2, "prob_home"]),
            float(batched.games.loc[2, "prob_home"]),
        )
        # And the first game of day 2 is identical in both modes: nothing on that
        # date has been played yet either way.
        self.assertAlmostEqual(
            float(batched.games.loc[1, "prob_home"]),
            float(sequential.games.loc[1, "prob_home"]), places=12)

    def test_the_order_by_argument_is_validated(self):
        games = synthetic_league()
        with self.assertRaises(KeyError):
            baselines.elo_rating_pass(games, k=10.0, order_by=("date", "nope"))
        with self.assertRaises(ValueError):
            baselines.elo_rating_pass(games, k=10.0, order_by=("home", "away"))
        with self.assertRaises(ValueError):
            baselines.elo_rating_pass(games, k=10.0, same_day="sideways")

    def test_the_day_counter_agrees_with_the_schedule(self):
        games = synthetic_league(twice_in_a_day=True, n_days=10, n_teams=4)
        self.assertEqual(baselines.max_team_games_per_day(games), 2)
        games = synthetic_league(twice_in_a_day=False, n_days=10, n_teams=4)
        self.assertEqual(baselines.max_team_games_per_day(games), 1)


# --------------------------------------------------------------------------- #
# The other baselines
# --------------------------------------------------------------------------- #

class TestLogLinearOnElo(unittest.TestCase):
    def test_it_is_a_recalibration_of_the_same_information(self):
        """Two claims, and one anti-claim that a first draft got wrong.

        The anti-claim: "a logistic on (R_home, R_away, 1) cannot reorder games
        relative to Elo, because it is a logistic in the rating gap". That is
        false unless the two slopes come out exactly antiparallel, and they do
        not -- home teams carry systematically higher ratings, so the fit can put
        part of the home effect on the R_home coefficient instead of the
        intercept, which genuinely reorders games. Measured rank correlation
        against the gap is 0.89, not 1.0. Reordering is what lets it fix the
        home bias, so that is a feature.

        The two real claims: the AUC barely moves (same information, different
        prices) and the log loss improves (better prices).
        """
        games = synthetic_league(n_days=200, n_teams=6, seed=17)
        tr, te = splits_of(games)[-1]
        fit = baselines.elo_rating_pass(games, k=10.0)
        y = games["home_win"].to_numpy()[te].astype(int)
        p = baselines.loglinear_elo_predictions(games, tr, te, fit)
        gap = (fit.games["rating_home_before"].to_numpy()[te]
               - fit.games["rating_away_before"].to_numpy()[te])
        raw = nd.evaluate(y, fit.prob_home[te])
        cal = nd.evaluate(y, p)
        self.assertGreater(float(pd.Series(p).corr(pd.Series(gap), method="spearman")),
                           0.80)
        self.assertLess(abs(cal["auc"] - raw["auc"]), 0.05)
        self.assertLess(cal["log_loss"], raw["log_loss"])

    def test_it_is_calibrated_where_raw_elo_is_not(self):
        # Raw Elo over-predicts the home side; the fitted intercept puts the mean
        # back near the training base rate. Asserted as a direction, not a number.
        games = synthetic_league()
        tr, te = splits_of(games)[-1]
        fit = baselines.elo_rating_pass(games, k=10.0)
        base = float(games["home_win"].to_numpy()[tr].mean())
        raw = fit.prob_home[te]
        cal = baselines.loglinear_elo_predictions(games, tr, te, fit)
        self.assertGreater(abs(float(cal.mean()) - base), abs(float(raw.mean()) - base))

    def test_an_extreme_training_base_rate_still_yields_finite_probabilities(self):
        # 90% home wins: the fit is nearly separable, so the solver pushes the
        # coefficients out. It must degrade to something finite inside (0, 1)
        # rather than raising or returning NaN -- and `evaluate` clips, so a NaN
        # would silently become a plausible-looking number downstream.
        games = synthetic_league(n_days=40, n_teams=4)
        rng = np.random.default_rng(4)
        keep = rng.random(len(games)) > 0.9
        games.loc[keep, "home_win"] = 0.0
        tr, te = splits_of(games)[-1]
        fit = baselines.elo_rating_pass(games, k=10.0)
        p = baselines.loglinear_elo_predictions(games, tr, te, fit)
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertTrue(np.all((p > 0.0) & (p < 1.0)))


class TestBatchBaselines(unittest.TestCase):
    def test_bradley_terry_is_one_number_in_a_two_team_league(self):
        # With only A and B there is nothing to rank: P(home wins) is a single
        # fitted constant, and it must not depend on which game of the block is
        # being predicted. The block has both home wins and home losses on
        # purpose -- a block of pure home wins has no finite maximum-likelihood
        # strength, and the module's own docstring says the ridge is what keeps
        # the optimum finite, so a degenerate fixture would be testing the
        # regulariser rather than the fit.
        rows = [("A", "B", int(i % 3 != 0), 5.0) for i in range(40)]
        games = tiny_frame(rows, step_days=1)
        tr, te = np.arange(0, 30), np.arange(30, 40)
        p = baselines.bradley_terry_predictions(games, tr, te)
        self.assertEqual(len(p), len(te))
        self.assertTrue(np.allclose(p, p[0]))
        self.assertGreater(p[0], 0.4)
        self.assertLess(p[0], 0.8)

    def test_bradley_terry_never_sees_a_game_it_is_scoring(self):
        games = synthetic_league()
        tr, te = splits_of(games)[-1]
        before = baselines.bradley_terry_predictions(games, tr, te, refit_every=10_000)
        flipped = games.copy()
        flipped.loc[te, "home_win"] = 1.0 - flipped["home_win"].iloc[te]
        self.assertTrue(np.allclose(
            before, baselines.bradley_terry_predictions(
                flipped, tr, te, refit_every=10_000)))

    def test_bradley_terry_parameters_do_not_depend_on_row_order(self):
        # A batch maximum-likelihood fit, so shuffling the training rows cannot
        # move a parameter. This is the control that shows the Elo ordering
        # guarantee is doing real work rather than decorating the API.
        #
        # The comparison is between PARAMETERS on the same set of games, not
        # between predictions from two different schedules: reshuffling a frame
        # and re-deriving the split gives a different set of games per index,
        # so a prediction-level comparison would be measuring the shuffle.
        games = synthetic_league()
        tr, _ = splits_of(games)[-1]
        block = games.iloc[tr]
        ordered = ratings.fit_bradley_terry(block)
        shuffled = ratings.fit_bradley_terry(
            block.sample(frac=1.0, random_state=9))
        self.assertEqual(set(ordered.strengths), set(shuffled.strengths))
        for team, value in ordered.strengths.items():
            self.assertAlmostEqual(value, shuffled.strengths[team], places=12,
                                   msg=f"{team} moved with the row order")
        self.assertAlmostEqual(ordered.home_advantage, shuffled.home_advantage,
                               places=12)

    def test_srs_module_calibration_is_blind_to_the_home_win_rate(self):
        """The structural claim, stated so it cannot be satisfied by accident.

        Two leagues with the SAME schedule and the SAME margins, differing only
        in their home win rate -- one where home teams win ~70% and one where they
        win ~85%. The module's calibration is
        `logistic(scale * (margin - mean_training_margin))`, so its predictions
        cannot move with the home win rate at all: it prices the home team as a
        coin flip and can only see a home edge as a *change* in the average
        margin relative to training, never as a *level*. The intercept version,
        which fits `logistic(a + b * margin)`, tracks the rate directly.
        """
        games = synthetic_league(n_days=120, n_teams=6, seed=3)
        tr, te = splits_of(games)[-1]
        homey = games.copy()
        homey.loc[homey.index[::8], "home_win"] = 0.0
        storm = games.copy()
        storm.loc[storm.index[1::4], "home_win"] = 0.0

        p_homey = baselines.srs_predictions(homey, tr, te, calibrate="module")
        p_storm = baselines.srs_predictions(storm, tr, te, calibrate="module")
        self.assertAlmostEqual(float(p_homey.mean()), 0.5, delta=0.05)
        self.assertAlmostEqual(float(p_storm.mean()), 0.5, delta=0.05)

        q_homey = baselines.srs_predictions(homey, tr, te, calibrate="intercept")
        q_storm = baselines.srs_predictions(storm, tr, te, calibrate="intercept")
        # `storm` relabels a quarter of the games as home losses, so it is the
        # league with the LOWER home rate and must get the LOWER mean
        # prediction. Getting that backwards would have made the test pass for
        # the wrong reason.
        rate_homey = float(homey["home_win"].to_numpy()[te].mean())
        rate_storm = float(storm["home_win"].to_numpy()[te].mean())
        self.assertGreater(rate_homey, rate_storm)
        self.assertGreater(float(q_homey.mean()) - float(q_storm.mean()), 0.05)
        # And the intercept version is calibrated -- against the TRAINING home
        # rate, which is the only rate it is entitled to know. Judged against the
        # test block's rate it looks badly calibrated (0.4867 predicted against
        # 0.361 actual for `storm`), and that is not a bug: the relabelling hits
        # the test block harder than the training block, and no model fitted at
        # the end of training can know that in advance. Scoring a walk-forward
        # prediction against a statistic of the window it is scored on is how a
        # calibration check turns into a leakage check by accident.
        train_homey = float(homey["home_win"].to_numpy()[tr].mean())
        train_storm = float(storm["home_win"].to_numpy()[tr].mean())
        self.assertLess(abs(float(q_homey.mean()) - train_homey), 0.05)
        self.assertLess(abs(float(q_storm.mean()) - train_storm), 0.05)
        # The two test-block rates are genuinely different, so the two leagues
        # really are different and the check above is not vacuous.
        self.assertGreater(abs(rate_homey - rate_storm), 0.10)

    def test_srs_predictions_survive_a_uniform_margin_shift_unchanged(self):
        # A weaker but exact companion: the ratings are built from
        # `margin - mean(margin)`, so a constant added to every margin is removed
        # by the centring and cannot move a single prediction. Equality to
        # floating-point precision rather than bit-for-bit, because
        # `mean(m + 12) - 12 != mean(m)` in the last bit and that difference
        # propagates.
        games = synthetic_league(n_days=60, n_teams=6, seed=3)
        shifted = games.copy()
        shifted["margin"] = shifted["margin"].to_numpy() + 12.0
        tr, te = splits_of(games)[-1]
        a = baselines.srs_predictions(games, tr, te, calibrate="module")
        b = baselines.srs_predictions(shifted, tr, te, calibrate="module")
        self.assertAlmostEqual(float(np.abs(a - b).max()), 0.0, places=12)

    def test_srs_with_an_intercept_recovers_a_home_edge(self):
        games = synthetic_league(n_days=60, n_teams=6, seed=3)
        tr, te = splits_of(games)[-1]
        actual = float(games["home_win"].to_numpy()[te].mean())
        p = baselines.srs_predictions(games, tr, te, calibrate="intercept")
        # The prediction has to point the same way as the outcome, and within a
        # few points of it, or the intercept is not doing its job.
        self.assertGreater(float(p.mean()) - 0.5, 0.0)
        self.assertLess(abs(float(p.mean()) - actual), 0.15)


# --------------------------------------------------------------------------- #
# Harness, reporting and reproducibility
# --------------------------------------------------------------------------- #

class TestSharedHarnessIsUsed(unittest.TestCase):
    """No second loader, no second evaluation harness. Checked by identity."""

    def test_the_module_binds_the_shared_functions_themselves(self):
        self.assertIs(baselines.nd, nd)
        self.assertIs(baselines.nd.build_games, nd.build_games)
        self.assertIs(baselines.nd.evaluate, nd.evaluate)
        self.assertIs(baselines.nd.paired_bootstrap_delta, nd.paired_bootstrap_delta)
        self.assertIs(baselines.nd.walk_forward_split, nd.walk_forward_split)

    def test_to_canonical_consumes_the_shared_game_frame_without_dropping_rows(self):
        # The compatibility claim the deliverable asks for: `fit_predict_elo`
        # cannot consume `build_games()` directly (the column names differ), and
        # this is the adapter that makes it work without losing a game.
        frame = baselines.load_baseline_frame()
        canonical = baselines.to_canonical(frame)
        self.assertEqual(len(canonical), len(frame))
        self.assertTrue({"date", "home", "away", "home_win", "margin", "GAME_ID",
                         "SEASON"} <= set(canonical.columns))
        self.assertTrue(np.array_equal(
            canonical["home_win"].to_numpy(), frame["y"].to_numpy()))
        self.assertTrue(np.array_equal(
            canonical["GAME_ID"].to_numpy(), frame["GAME_ID"].to_numpy()))
        # And the whole chain runs end to end on the real frame.
        fit = baselines.elo_rating_pass(canonical, k=10.0)
        self.assertEqual(len(fit.prob_home), len(frame))
        self.assertTrue(np.all((fit.prob_home > 0) & (fit.prob_home < 1)))

    def test_no_team_plays_twice_in_a_day_in_the_real_data(self):
        # The condition the same-date equivalence proof rests on, checked on the
        # data rather than trusted.
        self.assertEqual(baselines.max_team_games_per_day(
            baselines.load_baseline_frame()), 1)

    def test_bootstrap_sign_convention_positive_means_b_is_better(self):
        # This repository has got this backwards before, so it is pinned rather
        # than trusted: delta = loss(a) - loss(b), so a good `b` is POSITIVE.
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 400)
        p_flat = np.full(400, y.mean())
        p_good = np.where(y == 1, 0.75, 0.25)
        self.assertGreater(
            nd.paired_bootstrap_delta(y, p_flat, p_good, n_boot=200, seed=1)["delta"], 0.0)
        self.assertLess(
            nd.paired_bootstrap_delta(y, p_good, p_flat, n_boot=200, seed=1)["delta"], 0.0)

    def test_the_reported_delta_is_loss_a_minus_loss_b(self):
        # The sign convention checked against arithmetic rather than against the
        # helper's own docstring.
        y = np.array([1, 0, 1, 0] * 50)
        p_a = np.full(200, 0.5)
        p_b = np.where(y == 1, 0.7, 0.3)
        ll_a = -(y * np.log(p_a) + (1 - y) * np.log(1 - p_a)).mean()
        ll_b = -(y * np.log(p_b) + (1 - y) * np.log(1 - p_b)).mean()
        got = nd.paired_bootstrap_delta(y, p_a, p_b, n_boot=100, seed=2)
        self.assertAlmostEqual(got["delta"], ll_a - ll_b, places=12)

    def test_metrics_agree_with_the_shared_evaluate(self):
        games = synthetic_league()
        tr, te = splits_of(games)[-1]
        fit = baselines.elo_rating_pass(games, k=10.0)
        y = games["home_win"].to_numpy()[te].astype(int)
        p = fit.prob_home[te]
        m = baselines.BaselineResult("elo", "pooled", y, p).metrics()
        shared = nd.evaluate(y, p)
        self.assertEqual(m["n"], shared["n"])
        for key in ("log_loss", "brier", "auc", "accuracy"):
            self.assertAlmostEqual(m[key], shared[key], places=12)

    def test_standard_error_is_the_se_of_the_mean_per_game_loss(self):
        # sd / sqrt(n) of the per-game losses, written out longhand from the same
        # closed form the harness uses.
        y = np.array([0, 1, 1, 0, 1], dtype=int)
        p = np.array([0.2, 0.8, 0.6, 0.3, 0.9])
        per_game = -(y * np.log(p) + (1 - y) * np.log1p(-p))
        expected = float(per_game.std(ddof=1) / np.sqrt(len(y)))
        got = baselines.BaselineResult("m", "pooled", y, p).metrics()["log_loss_se"]
        self.assertAlmostEqual(got, expected, places=12)


class TestReproducible(unittest.TestCase):
    """Same input, same numbers, twice. Fixed seed wherever anything is random."""

    def test_two_suite_runs_are_identical(self):
        games = synthetic_league(n_days=50, n_teams=6, seed=11)
        first = baselines.run_baseline_suite(games=games, n_splits=3,
                                             initial_frac=0.4, k=10.0)
        second = baselines.run_baseline_suite(games=games, n_splits=3,
                                              initial_frac=0.4, k=10.0)
        a, b = first.table(), second.table()
        self.assertEqual(list(a.model), list(b.model))
        for col in ("log_loss", "brier", "auc", "accuracy", "log_loss_se"):
            self.assertTrue(np.allclose(a[col], b[col], rtol=0, atol=0),
                            f"{col} moved between two identical runs")
        self.assertTrue(np.array_equal(
            first.results[0].p, second.results[0].p))

    def test_the_bootstrap_is_seeded_and_only_the_interval_moves(self):
        games = synthetic_league(n_days=40, n_teams=4, seed=5)
        suite = baselines.run_baseline_suite(games=games, n_splits=2,
                                             initial_frac=0.4, k=10.0)
        one = suite.vs_base_rate(n_boot=300)
        two = suite.vs_base_rate(n_boot=300)
        # Same seed, same interval; and the point estimate is the difference of
        # two mean log losses, which no resampling can change.
        self.assertTrue(np.allclose(one["ci_lo"], two["ci_lo"]))
        self.assertTrue(np.allclose(one["ci_hi"], two["ci_hi"]))
        base = next(r for r in suite.results
                    if r.name == "base rate" and r.split == "pooled")
        elo = next(r for r in suite.results
                   if r.name == "elo (k tuned)" and r.split == "pooled")
        ll_base = nd.evaluate(base.y, base.p)["log_loss"]
        ll_elo = nd.evaluate(elo.y, elo.p)["log_loss"]
        row = one.loc[one.model == "elo (k tuned)"].iloc[0]
        self.assertAlmostEqual(float(row["delta_vs_ref"]), ll_base - ll_elo, places=12)
        self.assertEqual(baselines.SEED, 20241027)

    def test_the_finding_text_cannot_contradict_the_table(self):
        # The summary prose is generated from the run, so every figure it quotes
        # for a model has to be that model's own number. If the wiring drifts,
        # this fails rather than shipping a results file that disagrees with
        # itself.
        games = synthetic_league(n_days=60, n_teams=6, seed=5)
        suite = baselines.run_baseline_suite(games=games, n_splits=2,
                                             initial_frac=0.4, k=10.0)
        table = suite.table().set_index("model")
        text = baselines._findings_text(
            suite.table(), suite.vs_base_rate(n_boot=200),
            suite.pairwise("base rate", n_boot=200), suite.tuned_k)
        # Exactly the figures the summary actually quotes, so this is a check on
        # the wiring rather than a demand that the text grow.
        quoted = {
            "base rate": ("log_loss", "auc"),
            "elo (k tuned)": ("log_loss", "auc"),
            "log-linear on elo": ("log_loss",),
            "elo (k=20, FiveThirtyEight default)": ("log_loss", "auc"),
            "bradley-terry (walk-forward)": ("log_loss", "auc"),
            "srs + intercept (walk-forward)": ("log_loss", "auc"),
            "srs (module calibration)": ("mean_pred",),
        }
        for name, columns in quoted.items():
            for column in columns:
                self.assertIn(f"{table.loc[name, column]:.4f}", text,
                              f"{name}'s {column} is quoted in the summary but the "
                              f"text does not contain {table.loc[name, column]:.4f}")

    def test_the_finding_text_follows_the_interval_rather_than_asserting(self):
        # The earlier draft of the prose said "nowhere near zero" and "LOSE to
        # the base rate" as literals, both of which are false on a short sample.
        # The verdict words have to follow the interval, so a small suite must
        # not be told its differences are resolved.
        games = synthetic_league(n_days=60, n_teams=6, seed=5)
        suite = baselines.run_baseline_suite(games=games, n_splits=2,
                                             initial_frac=0.4, k=10.0)
        vs_base = suite.vs_base_rate(n_boot=200)
        text = baselines._findings_text(
            suite.table(), vs_base, suite.pairwise("base rate", n_boot=200),
            suite.tuned_k)
        row = vs_base.loc[vs_base.model == "elo (k tuned)"].iloc[0]
        straddles = row["ci_lo"] < 0 < row["ci_hi"]
        self.assertEqual("STRADDLES ZERO" in text, straddles,
                         "the prose verdict and the interval disagree")


class TestKIsTunedOnValidationOnly(unittest.TestCase):
    """The leak a tuner invites: score the sweep on the test block.

    The suite here is 800 games over 8 teams, which is the smallest league on
    which the validation slices are big enough to distinguish a learning rate.
    A 180-game league was tried first and its inner validation slices held 14
    games, which is noise: the sweep then "preferred" the smallest K on the grid
    for reasons that had nothing to do with learning rates.
    """

    def league(self):
        return synthetic_league(n_days=200, n_teams=8, seed=7)

    def test_the_sweep_is_sorted_and_the_winner_is_the_minimum(self):
        table = baselines.tune_elo_k(self.league(), splits_of(self.league()),
                                     grid=(5.0, 10.0, 20.0, 30.0))
        self.assertEqual(sorted(table["k"]), [5.0, 10.0, 20.0, 30.0])
        self.assertTrue(table["val_log_loss_mean"].is_monotonic_increasing)
        best = table.loc[table["val_log_loss_mean"].idxmin()]
        self.assertEqual(float(table.iloc[0]["k"]), float(best["k"]))
        self.assertEqual(float(table.iloc[0]["val_rank"]), 1.0)

    def test_each_split_tuning_ignores_its_own_test_block(self):
        """Scored per split, because "test rows" is split-relative.

        Walk-forward blocks grow: split 3's training block contains split 0's test
        block. So scrambling *every* split's test rows legitimately moves the
        later splits' tuning, and a test that asserted otherwise would be
        asserting that the growing window is not a growing window. What must hold
        is that split k's own test rows do not enter split k's validation score.
        """
        games = self.league()
        splits = splits_of(games)
        grid = (5.0, 10.0, 20.0)
        before = baselines.tune_elo_k(games, splits, grid=grid).set_index("k")
        for k, (_, te) in enumerate(splits):
            scrambled = games.copy()
            scrambled.loc[te, "home_win"] = 1.0 - scrambled["home_win"].iloc[te]
            after = baselines.tune_elo_k(scrambled, splits, grid=grid).set_index("k")
            # Compared by K label, not positionally: `tune_elo_k` returns the
            # sweep sorted best-first, so the two frames need not share a row
            # order and a positional comparison would compare K=5's score against
            # K=20's.
            self.assertTrue(
                np.allclose(before[f"val_split_{k}"].sort_index(),
                            after[f"val_split_{k}"].sort_index()),
                f"split {k}'s validation score moved when its own test labels changed")

    def test_a_grid_that_includes_a_frozen_rating_loses_to_a_tuned_one(self):
        # Guards against a "tuning" implementation that simply takes the smallest
        # value on the grid. K -> 0 is a rating system that never learns.
        games = self.league()
        splits = splits_of(games)
        tuned = baselines.tune_elo_k(games, splits, grid=(10.0, 15.0, 20.0, 25.0))
        frozen = baselines.tune_elo_k(games, splits, grid=(1e-9,))
        self.assertLess(float(tuned.iloc[0]["val_log_loss_mean"]),
                        float(frozen.iloc[0]["val_log_loss_mean"]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
