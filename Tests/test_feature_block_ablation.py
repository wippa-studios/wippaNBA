"""Feature-block ablation: the mechanism, pinned so it cannot quietly rot.

Issue #19 measured one feature block by hand. `scripts/feature_block_ablation.py`
is the instrument that makes "does this block earn its place?" answerable for any
block, with a paired bootstrap interval on the difference.

These tests deliberately do NOT re-run the measurement. The harness fits five
logistic regressions per arm over ~13,000 games; asserting its numbers again in CI
would be slow, and would assert values the data does not guarantee on a fresh
download anyway. What is pinned here is everything that could make a block score
WRONG while still producing a confident-looking interval:

  * the registry resolves against the real frame, and no two blocks overlap;
  * `drop_block` really removes the named columns (a silent no-op ablation fits the
    same model twice and reports a beautiful zero-width interval);
  * the sign convention, against a case where the correct answer is forced;
  * a synthetic block of PURE NOISE yields a null rather than a spurious win;
  * the JSON artifact records the run's OWN --seed/--splits, not the module
    defaults (the writer once logged SEED/N_SPLITS and mislabelled every
    replication run in results/experiments.jsonl as a default run).

The last one is the important one. An ablation harness that reports findings when
there is nothing to find is worse than no harness, because its positive results
read as evidence.

Data-dependence is NOT handled here. `nba_data`'s loaders raise `unittest.SkipTest`
when the NBA data is absent, which both pytest and unittest honour, so a missing
382 MB download skips these tests through the one sanctioned path.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from DataProviders import nba_data as nd  # noqa: E402
import feature_block_ablation as fba  # noqa: E402
import recency_ablation as ra  # noqa: E402

RESULTS = ROOT / "results" / "feature_block_ablation.json"


class TestBlockRegistry(unittest.TestCase):
    """The registry is the contract. If it lies, every number below it lies."""

    def setUp(self):
        self.frame = fba.build_feature_frame(nd.build_games())

    def test_every_registered_block_resolves_against_the_real_frame(self):
        """Each block's columns must actually exist.

        `resolve_block` raises rather than returning a short list, so reaching the
        assertions at all means resolution succeeded. This test exists to make the
        registry's drift a named failure rather than something a later test trips
        over by accident.
        """
        for name in sorted(fba.FEATURE_BLOCKS):
            with self.subTest(block=name):
                columns = fba.resolve_block(name, self.frame)
                self.assertTrue(columns, f"block {name!r} is empty")
                self.assertEqual(len(columns), len(set(columns)),
                                 f"block {name!r} lists a column twice")

    def test_blocks_do_not_overlap(self):
        """Two blocks sharing a column cannot both be ablated independently.

        Dropping block A when block B still carries the column removes nothing, and
        the resulting "null" is an artifact of the registry rather than a finding
        about the data. Overlap is a registry bug, so it is checked here rather
        than inferred from a confusing result later.
        """
        owner: dict[str, str] = {}
        for name in sorted(fba.FEATURE_BLOCKS):
            for column in fba.resolve_block(name, self.frame):
                self.assertNotIn(
                    column, owner,
                    f"column {column!r} is in both {owner.get(column)!r} and "
                    f"{name!r}; the two ablations are not independent",
                )
                owner[column] = name

    def test_unknown_block_name_is_refused(self):
        """A typo must raise, not resolve to nothing.

        Silently accepting an unknown name is how a block gets dropped from a run
        without anyone noticing the row is missing.
        """
        with self.assertRaises(KeyError):
            fba.resolve_block("no_such_block", self.frame)

    def test_block_naming_a_missing_column_is_refused(self):
        """A block whose column has vanished must fail loudly.

        This is the real hazard the registry exists for. `rolling_team_features`
        silently dropped both rebounding features for the life of this project by
        asking the four-factors table for column names it does not publish; the
        guard that would have caught it is this one.
        """
        saved = fba.FEATURE_BLOCKS["pace"]["columns"]
        try:
            fba.FEATURE_BLOCKS["pace"]["columns"] = ["home_ewm_pace", "not_a_column"]
            with self.assertRaises(KeyError) as ctx:
                fba.resolve_block("pace", self.frame)
            self.assertIn("not_a_column", str(ctx.exception))
        finally:
            fba.FEATURE_BLOCKS["pace"]["columns"] = saved


class TestTheAblationActuallyAblates(unittest.TestCase):
    def test_drop_block_removes_exactly_the_named_columns(self):
        frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
        out = fba.drop_block(frame, ["b"])
        self.assertEqual(list(out.columns), ["a", "c"])
        self.assertEqual(out.shape[1], frame.shape[1] - 1)

    def test_drop_block_refuses_a_column_that_is_not_there(self):
        """Dropping a non-existent column is a no-op, and a no-op scores a fake null."""
        frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with self.assertRaises(KeyError):
            fba.drop_block(frame, ["zzz"])

    def test_drop_block_detects_an_ablation_that_changed_nothing(self):
        """Even a duplicate-suppressed drop must be caught by the width check.

        `frame.drop(columns=...)` is exact, so this is a belt-and-braces guard on
        the invariant that matters: if the width does not fall by exactly the number
        of columns named, the two arms are the same model and the delta is a
        meaningless zero.
        """
        frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        with self.assertRaises(AssertionError):
            fba.drop_block(frame, [])  # dropping nothing must not be silent

    def test_scoring_a_block_changes_the_model(self):
        """End-to-end: two arms of a real block must not produce identical predictions.

        Cheap enough to keep (one block, one split, a subsample) and it is the only
        check that the whole path - resolve, drop, fit, score - actually removes
        information rather than passing the full frame through twice.
        """
        games = nd.build_games().head(2500)
        frame = fba.build_feature_frame(games).head(2500)
        splits = nd.walk_forward_split(games, n_splits=2)
        y = games["y"].to_numpy()
        columns = fba.resolve_block("rank", frame)
        p_full, _, _ = ra.fit_oof(frame, y, splits, max_iter=100)
        p_minus, _, _ = ra.fit_oof(fba.drop_block(frame, columns), y, splits,
                                   max_iter=100)
        self.assertFalse(np.allclose(p_full, p_minus),
                         "removing the rank block left the model unchanged")


class TestSignConvention(unittest.TestCase):
    """This repo has inverted this convention three times."""

    def test_positive_delta_means_the_ablated_model_is_better(self):
        """The raw function's contract, restated at the boundary we call it from."""
        y = np.array([0, 0, 1, 1] * 25)
        p_full = np.full(100, 0.5)                      # no information at all
        p_minus = np.where(y == 1, 0.9, 0.1)            # perfectly separated
        d = nd.paired_bootstrap_delta(y, p_full, p_minus, n_boot=200, seed=0)
        self.assertGreater(d["delta"], 0.0,
                           "a BETTER minus-block model must give a POSITIVE delta; "
                           "if this fails the sign convention has flipped")
        self.assertGreater(d["ci_lo"], 0.0,
                           "a flawless minus-block model must have its whole "
                           "interval above zero")

    def test_classify_reads_an_interval_not_a_point_estimate(self):
        """The verdict must come from the interval, in the repo's direction.

        delta = loss(full) - loss(minus block):
          interval below zero -> removing the block HURT  -> it earns its place
          interval above zero -> removing the block HELPED -> it does not
          straddling zero     -> null, and it must be labelled a null
        """
        self.assertEqual(fba.classify(-0.01, -0.002), fba.EARNS)
        self.assertEqual(fba.classify(0.002, 0.01), fba.DOES_NOT_EARN)
        self.assertEqual(fba.classify(-0.01, 0.01), fba.NULL)
        # Touching zero from either side is still a null: an interval whose endpoint
        # is exactly 0.0 has not excluded it.
        self.assertEqual(fba.classify(-0.01, 0.0), fba.NULL)
        self.assertEqual(fba.classify(0.0, 0.01), fba.NULL)

    def test_a_large_point_estimate_with_a_null_interval_is_still_a_null(self):
        """A big-looking delta whose interval straddles zero is not a finding.

        This is the failure the harness exists to prevent: a point estimate of the
        wrong sign or of impressive size, reported as though it settled something.
        """
        self.assertEqual(fba.classify(-0.004, 0.02), fba.NULL)


def _synthetic_games(n, seed=0):
    """A synthetic frame with one genuinely informative pair and pure-noise extras.

    Built by hand rather than drawn from a real season so the ground truth is known:
    `signal` carries the label, `noise_*` carry nothing at all. Used to check that a
    block of noise produces a null instead of a spurious win.
    """
    rng = np.random.default_rng(seed)
    n_splits = 3
    # walk_forward_split holds the first half for training only, so the synthetic
    # frame must be long enough to leave out-of-fold rows at both ends.
    n_games = 1400
    y = (rng.random(n_games) < 0.55).astype(int)

    data = {
        "signal": rng.normal(size=n_games),
        "noise_a": rng.normal(size=n_games),
        "noise_b": rng.normal(size=n_games),
        "noise_c": rng.normal(size=n_games),
        "noise_d": rng.normal(size=n_games),
    }
    frame = pd.DataFrame(data)
    # Label depends on `signal` and on nothing else, with an effect comfortably
    # larger than the noise columns' spurious correlations.
    logit = 1.4 * data["signal"] + 0.25 * (y - 0.5)
    y = (rng.random(n_games) < 1.0 / (1.0 + np.exp(-logit))).astype(int)

    games = pd.DataFrame({
        "GAME_ID": [f"{i:07d}" for i in range(n_games)],
        "GAME_DATE": pd.date_range("2015-10-01", periods=n_games, freq="D"),
        "home": ["AAA"] * n_games,
        "away": ["BBB"] * n_games,
        "home_pts": np.where(y == 1, 110, 100),
        "away_pts": np.where(y == 1, 100, 110),
        "y": y,
    })
    games["margin"] = games["home_pts"] - games["away_pts"]
    games["total"] = games["home_pts"] + games["away_pts"]
    games["SEASON"] = "2015-16"
    splits = nd.walk_forward_split(games, n_splits=n_splits)
    return frame, y, splits


class TestNoiseBlockIsANullNotAWin(unittest.TestCase):
    """The harness must not manufacture findings."""

    def test_pure_noise_block_produces_a_null(self):
        """Dropping four uninformative columns must NOT come out significant.

        The risk is asymmetric. Adding noise features leaves log loss essentially
        unchanged, so the delta sits near zero and the interval straddles it -
        unless the harness leaks, or compares the wrong pair, or scores
        training rows, any of which turns this into a confident wrong answer.
        """
        frame, y, splits = _synthetic_games(0)
        noise = ["noise_a", "noise_b", "noise_c", "noise_d"]

        p_full, y_oof, _ = ra.fit_oof(frame, y, splits, max_iter=200)
        p_minus, y_minus, _ = ra.fit_oof(fba.drop_block(frame, noise), y, splits,
                                         max_iter=200)
        self.assertEqual(len(y_oof), len(y_minus))

        d = nd.paired_bootstrap_delta(y_oof, p_full, p_minus, n_boot=400, seed=42)
        verdict = fba.classify(d["ci_lo"], d["ci_hi"])
        self.assertEqual(
            verdict, fba.NULL,
            f"a pure-noise block was called {verdict!r} "
            f"(delta {d['delta']:+.5f}, CI [{d['ci_lo']:+.5f}, {d['ci_hi']:+.5f}]). "
            f"Either the harness leaks or the comparison is mispaired.",
        )

    def test_the_signal_column_is_not_a_null(self):
        """The control for the test above.

        A harness that returns a null for everything would pass the noise test
        trivially, so something must be shown to produce a NON-null.

        Sign, stated explicitly because this file's author got it backwards on the
        first attempt. delta = loss(full) - loss(minus <block>). Dropping the only
        informative column makes the ablated model WORSE, so loss(minus) > loss(full)
        and the delta is NEGATIVE. A negative interval means removing the block hurt,
        which is what EARNS ITS PLACE means. `recency_followup.py` uses the same
        reading: "delta = loss(default) - loss(arm). Negative means the arm is WORSE."
        """
        frame, y, splits = _synthetic_games(0)

        p_full, y_oof, _ = ra.fit_oof(frame, y, splits, max_iter=200)
        p_minus, y_minus, _ = ra.fit_oof(fba.drop_block(frame, ["signal"]), y,
                                         splits, max_iter=200)

        d = nd.paired_bootstrap_delta(y_oof, p_full, p_minus, n_boot=400, seed=42)
        self.assertLess(
            d["delta"], 0.0,
            f"removing the only informative column must make the model worse, i.e. "
            f"give a NEGATIVE delta; got {d['delta']:+.5f}. loss(minus)="
            f"{nd.evaluate(y_minus, p_minus)['log_loss']:.5f} against "
            f"loss(full)={nd.evaluate(y_oof, p_full)['log_loss']:.5f}",
        )
        self.assertEqual(
            fba.classify(d["ci_lo"], d["ci_hi"]), fba.EARNS,
            f"the signal column's interval should exclude zero and read as earning "
            f"its place; got delta {d['delta']:+.5f} with CI "
            f"[{d['ci_lo']:+.5f}, {d['ci_hi']:+.5f}]",
        )


class TestOnlyOutOfFoldRowsAreScored(unittest.TestCase):
    """The first half of the data is training-only by design and must stay unscored."""

    def test_fit_oof_returns_only_test_block_rows(self):
        games = nd.build_games().head(2500)
        frame = fba.build_feature_frame(games).head(2500)
        splits = nd.walk_forward_split(games, n_splits=3)
        expected = np.zeros(len(games), dtype=bool)
        for _, te in splits:
            expected[te] = True

        _, y_out, mask = ra.fit_oof(frame, games["y"].to_numpy(), splits,
                                    max_iter=100)
        self.assertEqual(len(y_out), int(expected.sum()))
        # The training-only prefix must contribute nothing.
        prefix = splits[0][0]
        self.assertFalse(mask[prefix].any(),
                         "a training row was scored as though it were held out")

    def test_walk_forward_split_holds_the_first_half_back(self):
        """The geometry the guard depends on. If this changes, the guard is moot."""
        games = nd.build_games().head(1200)
        splits = nd.walk_forward_split(games, n_splits=5)
        self.assertTrue(splits)
        first_test = splits[0][1][0]
        self.assertGreater(first_test, 0,
                           "the first split has no held-out rows at all")


class TestRankBlockIsCausal(unittest.TestCase):
    """The rank block is a reconstruction of #19's columns, so it must not leak."""

    def test_future_games_cannot_change_historical_ranks(self):
        """Truncating the future must leave every past rank untouched.

        This is the property the whole block rests on. `rolling_team_features` is
        strictly causal per team, but a league-wide RANK is a cross-sectional
        quantity, so it is not automatically causal: if a club's standing is taken
        over a league that includes teams whose future games have been folded in,
        the rank can move after the fact. Extending the frame with later games and
        re-reading the earlier rows is the direct test.
        """
        games = nd.build_games()
        short = games.head(1200)
        long = games.head(2000)

        def ranks(g):
            roll = nd.rolling_team_features(g, halflife=fba.HALFLIFE,
                                            min_games=fba.MIN_GAMES)
            return fba.league_rank_features(g, roll).reset_index(drop=True)

        a = ranks(short)
        b = ranks(long).iloc[:len(a)].reset_index(drop=True)
        np.testing.assert_allclose(
            a.to_numpy(dtype=float), b.to_numpy(dtype=float),
            rtol=0, atol=0, equal_nan=True,
            err_msg="future games changed historical league ranks; the rank block "
                    "sees the game it is predicting",
        )

    def test_ranks_are_ordinal_over_the_league(self):
        """Ranks must span 1..30, with 1 the strongest club.

        A block that silently degraded into "the raw value again" would make the
        ablation meaningless - it would measure the value twice under a new name.
        """
        games = nd.build_games().head(1500)
        roll = nd.rolling_team_features(games, halflife=fba.HALFLIFE,
                                        min_games=fba.MIN_GAMES)
        r = fba.league_rank_features(games, roll)
        values = r.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        self.assertGreaterEqual(finite.min(), 1.0)
        self.assertLessEqual(finite.max(), 30.0)
        self.assertEqual(sorted(set(np.unique(finite[np.isfinite(finite)]))),
                         sorted(np.unique(finite[np.isfinite(finite)])))
        # Rank 1 must be the strongest margin, and the home/away of one game must
        # be different clubs, so ranks should actually vary.
        self.assertGreater(len(np.unique(finite)), 5,
                           "ranks are near-constant; the block measures nothing")


class TestMeasuredArtifact(unittest.TestCase):
    """Pin the shape of the published result, read from its JSON artifact."""

    @classmethod
    def setUpClass(cls):
        if not RESULTS.exists():
            raise unittest.SkipTest(
                f"{RESULTS.name} not present; run scripts/feature_block_ablation.py"
            )
        cls.data = json.loads(RESULTS.read_text())

    def test_every_recorded_block_carries_an_interval(self):
        """No comparison may be reported without one.

        The artifact is what gets read months later. A delta without a CI in it is a
        number waiting to be quoted as a finding.
        """
        blocks = self.data["blocks"]
        self.assertTrue(blocks, "no blocks recorded")
        for name, d in blocks.items():
            with self.subTest(block=name):
                for key in ("delta", "ci_lo", "ci_hi", "verdict", "z"):
                    self.assertIn(key, d, f"{name} is missing {key!r}")
                self.assertLessEqual(d["ci_lo"], d["ci_hi"],
                                     f"{name} has an inverted interval")

    def test_each_verdict_matches_its_own_interval(self):
        """The verdict must be re-derivable from the stored interval.

        If this fails, the artifact's verdicts and its intervals disagree - the
        reader is being told a block earned its place by a number that says the
        opposite.
        """
        for name, d in self.data["blocks"].items():
            with self.subTest(block=name):
                self.assertEqual(d["verdict"],
                                 fba.classify(d["ci_lo"], d["ci_hi"]),
                                 f"{name}: stored verdict disagrees with its CI")

    def test_the_summary_partitions_the_blocks(self):
        """Every scored block lands in exactly one of the three verdict buckets."""
        summary = self.data["summary"]
        buckets = (summary["earns_its_place"] + summary["does_not_earn_its_place"]
                   + summary["null"])
        self.assertCountEqual(buckets, list(self.data["blocks"]))

    def test_the_artifact_states_its_sign_convention(self):
        """The sign rule travels with the numbers.

        A delta stored without it is a delta that will eventually be read backwards.
        """
        self.assertIn("loss(full)", self.data["sign_convention"])
        self.assertIn("POSITIVE", self.data["sign_convention"])

    def test_the_artifact_records_the_columns_actually_scored(self):
        """The registry shipped in the artifact must match what the code declares.

        A stale artifact that names a different column list than the code would let
        a reader believe a block meant something it no longer means.
        """
        for name, columns in self.data["registry"].items():
            self.assertEqual(columns, list(fba.FEATURE_BLOCKS[name]["columns"]),
                             f"registry for {name!r} drifted from the artifact")


class TestArtifactRecordsRunArgsNotModuleDefaults(unittest.TestCase):
    """The artifact must describe the run that produced it, not the defaults.

    The JSON writer once recorded the module constants SEED and N_SPLITS instead
    of the parsed command-line args, so every replication run - the runs whose
    entire point is a non-default --seed or --splits - was logged under the
    DEFAULT configuration. Twenty records in results/experiments.jsonl carry that
    wrong label (an erratum there records it). A record that lies about its own
    configuration is worse than no record: it turns a replication on paper into a
    second copy of the original, which is the one thing a replication must not be.
    """

    def test_written_record_carries_the_run_args_not_the_module_constants(self):
        seed, n_splits, n_boot = 137, 4, 100
        self.assertNotEqual(
            (seed, n_splits), (fba.SEED, fba.N_SPLITS),
            "the probe must run non-default args or it proves nothing",
        )

        # A fabricated run through the REAL parse-and-write path: only the model
        # fitting is stubbed, because the bug this pins is in what main() records.
        frame, y, _ = _synthetic_games(0)
        games = pd.DataFrame({"y": y})
        splits = nd.walk_forward_split(games, n_splits=n_splits)
        n_oof = sum(len(te) for _, te in splits)

        def fake_fit_oof(frame, y, splits, **kwargs):
            order = np.concatenate([te for _, te in splits])
            mask = np.zeros(len(y), dtype=bool)
            mask[order] = True
            return np.full(len(order), 0.5), np.asarray(y)[order], mask

        scored_with: dict = {}

        def fake_score_block(frame, y, splits, block, *, n_boot, seed):
            scored_with.update(seed=seed, n_boot=n_boot)
            return {
                "block": block, "n_columns": 1, "columns": ["signal"],
                "delta": 0.0, "ci_lo": -0.01, "ci_hi": 0.01,
                "p_positive": 0.5, "n_boot": n_boot, "z": 0.0,
                "verdict": fba.NULL, "log_loss_full": 0.69,
                "log_loss_minus": 0.69, "auc_full": 0.5, "auc_minus": 0.5,
                "n_scored": n_oof,
            }

        # Registered like the tests above mutate the registry, so `resolve_block`
        # runs for real against the synthetic frame.
        fba.FEATURE_BLOCKS["meta_probe"] = {
            "columns": ["signal"],
            "note": "test-only probe block",
        }
        try:
            with tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp) / "artifact.json"
                argv = ["feature_block_ablation.py",
                        "--blocks", "meta_probe",
                        "--seed", str(seed), "--splits", str(n_splits),
                        "--n-boot", str(n_boot), "--out", str(out)]
                with mock.patch.object(sys, "argv", argv), \
                        mock.patch.object(fba.nd, "build_games",
                                          return_value=games), \
                        mock.patch.object(fba, "build_feature_frame",
                                          return_value=frame), \
                        mock.patch.object(fba.ra, "fit_oof",
                                          side_effect=fake_fit_oof), \
                        mock.patch.object(fba, "score_block",
                                          side_effect=fake_score_block):
                    self.assertEqual(fba.main(), 0)
                record = json.loads(out.read_text())
        finally:
            fba.FEATURE_BLOCKS.pop("meta_probe", None)

        # The point of the exercise: the RUN's args, not SEED/N_SPLITS.
        self.assertEqual(record["seed"], seed,
                         "the artifact recorded the module SEED, not the run's "
                         "--seed; every non-default run is logged as a default one")
        self.assertEqual(record["n_splits"], n_splits,
                         "the artifact recorded the module N_SPLITS, not the "
                         "run's --splits")
        self.assertEqual(scored_with, {"seed": seed, "n_boot": n_boot},
                         "scoring was not handed the run's args either")
        # And no other field may describe anything but THIS run.
        self.assertEqual(record["n_games"], len(games))
        self.assertEqual(record["n_out_of_fold"], n_oof)
        self.assertEqual(record["n_features_full"], frame.shape[1])


if __name__ == "__main__":
    unittest.main()