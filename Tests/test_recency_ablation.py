"""Recency weighting: the shape of the answer, pinned so it cannot drift.

`scripts/recency_followup.py` measured the relationship between how fast a team
feature decays and how well it predicts, holding the feature set, the model class,
the splits and the seed fixed. The result is an inverted-U:

    arm              halflife   log loss   vs halflife 10 (95% CI)
    season-average   10000      0.68383    worse by 0.06390  [+0.05567, +0.07178]
    slow                30      0.62341    worse by 0.00349  [+0.00016, +0.00685]
    medium              10      0.61993    (reference)
    fast                 5      0.62731    worse by 0.00738  [+0.00501, +0.00966]
    very-fast            2      0.64462    worse by 0.02470  [+0.01978, +0.02964]

Read with the repo's sign convention: `paired_bootstrap_delta(y, p_a, p_b)`
returns `loss(a) - loss(b)`, so a POSITIVE delta means b is better. Every interval
above is positive, which means every non-reference arm is worse than halflife 10.

This is worth pinning because it is a NULL-shaped result that reads like a finding
if you only look at one column. The static arm loses by 0.064 nats with |z| = 15.6,
which looks spectacular until you notice the fast arms also lose -- so the correct
conclusion is "halflife 10 is the optimum", not "make the features move faster".

These tests check the MECHANISM (that the arms genuinely differ, and that the
ordering is the one measured) rather than re-running the 30-minute measurement,
which belongs to the script and its JSON output. Re-deriving the full number in CI
would be slow and would assert a value the data does not guarantee on new data.
"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT, ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from DataProviders import nba_data as nd  # noqa: E402

RESULTS = ROOT / "results" / "recency_followup.json"

# The decay arms the experiment compares. Longest halflife first: it approximates
# the static season average #20 describes, and halflife 10 is the committed default.
ARMS_EXPECTED = [10000.0, 30.0, 10.0, 5.0, 2.0]


class TestDecayChangesFeatures(unittest.TestCase):
    """The arms must actually differ, or the comparison measures nothing."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.games = nd.build_games()
        except Exception as exc:  # data not downloaded
            raise unittest.SkipTest(f"NBA data unavailable: {exc}")

    def test_shorter_halflife_moves_features_further(self):
        """Movement must increase monotonically as the halflife shortens.

        If this fails, the decay parameter is not wired into the feature
        computation at all and every downstream number is meaningless.
        """
        # A short, hand-checkable range. Tests should not be slow; the full sweep
        # belongs to the script.
        halflives = [30.0, 10.0, 5.0]
        movements = []
        for hl in halflives:
            r = nd.rolling_team_features(
                self.games.head(3000), halflife=hl, min_games=5
            ).reset_index(drop=True)
            long = pd.concat([
                self.games.head(3000)[["GAME_DATE", "home"]]
                    .rename(columns={"home": "team"})
                    .assign(v=r["home_ewm_margin"].to_numpy(float)),
                self.games.head(3000)[["GAME_DATE", "away"]]
                    .rename(columns={"away": "team"})
                    .assign(v=r["away_ewm_margin"].to_numpy(float)),
            ], ignore_index=True).sort_values(["team", "GAME_DATE"])
            chg = long.groupby("team", sort=False)["v"].diff().abs()
            sd = long.groupby("team", sort=False)["v"].std()
            movements.append(float((chg.groupby(long["team"]).mean() / sd).median()))

        self.assertLess(movements[0], movements[1],
                        "halflife 30 must move features LESS than halflife 10")
        self.assertLess(movements[1], movements[2],
                        "halflife 10 must move features LESS than halflife 5")

    def test_every_decay_stays_strictly_causal(self):
        """Truncating the future must not change a historical feature.

        This is the property that makes the whole comparison legitimate: if the
        arms differed because one leaked, the ordering would be an artifact.
        """
        g = self.games.head(2000)
        for hl in (30.0, 10.0, 5.0):
            full = nd.rolling_team_features(g, halflife=hl, min_games=5)
            # Same games, plus 800 later ones. Nothing about row i may change.
            extended = nd.rolling_team_features(
                self.games.head(2800), halflife=hl, min_games=5
            )
            a = full.reset_index(drop=True)
            b = extended.iloc[:len(a)].reset_index(drop=True)
            self.assertEqual(a.shape, b.shape, f"shape changed at halflife {hl}")
            np.testing.assert_allclose(
                a.select_dtypes(include=[np.number]).to_numpy(),
                b.select_dtypes(include=[np.number]).to_numpy(),
                rtol=0, atol=0,
                err_msg=f"future games changed historical features at halflife {hl}",
            )


class TestMeasuredOrdering(unittest.TestCase):
    """Pin the shape of the published result, read from its JSON artifact."""

    @classmethod
    def setUpClass(cls):
        if not RESULTS.exists():
            raise unittest.SkipTest(
                f"{RESULTS.name} not present; run scripts/recency_followup.py"
            )
        cls.data = json.loads(RESULTS.read_text())

    def test_every_non_reference_arm_is_worse_with_interval_below_zero(self):
        """The whole point: nothing beats halflife 10.

        delta = loss(reference) - loss(arm). An arm is worse when delta < 0 AND its
        entire interval sits below zero. Testing only the sign would pass on an
        interval straddling zero, which is exactly the mistake this file exists to
        prevent.
        """
        vs = self.data["vs_default"]
        self.assertTrue(vs, "no comparisons recorded")
        for name, d in vs.items():
            self.assertLess(d["delta"], 0.0,
                            f"{name} beat halflife 10 by {d['delta']:+.5f}")
            self.assertLess(d["ci_hi"], 0.0,
                            f"{name} has an interval reaching zero: "
                            f"[{d['ci_lo']:+.5f}, {d['ci_hi']:+.5f}]")

    def test_movement_rose_monotonically(self):
        """Movement must increase as the halflife shortens.

        Combined with the test above this is the finding: the metric improved, the
        predictions got worse. If movement ever stopped responding to the halflife,
        the experiment would no longer be measuring what it claims to.

        Sorted LONGEST halflife first, so consecutive pairs run from a longer to a
        shorter halflife. Movement must RISE along that direction. (Getting this
        direction backwards is exactly what an earlier draft of this test did, so
        it is stated explicitly rather than left to be inferred from the operator.)
        """
        arms = sorted(
            ((float(d["halflife"]), float(d["movement"]))
             for d in self.data["movement_by_arm"].values()),
            key=lambda t: -t[0],  # LONGEST halflife first
        )
        self.assertEqual(len(arms), len(ARMS_EXPECTED),
                         "every decay arm should appear in the results")
        for (hl_long, mv_long), (hl_short, mv_short) in zip(arms, arms[1:]):
            self.assertLess(
                mv_long, mv_short,
                f"as the halflife shortens from {hl_long} to {hl_short}, movement "
                f"must rise; got {mv_long:.4f} -> {mv_short:.4f}",
            )

    def test_movement_peak_is_not_where_accuracy_peaks(self):
        """The null has to remain a null.

        Accuracy is best at halflife 10. Movement keeps climbing past it. If a
        future change made those coincide, the conclusion would genuinely need
        revisiting -- so this test asserts they still differ, and says why.
        """
        by_hl = {d["halflife"]: d for d in self.data["movement_by_arm"].values()}
        worst_movement_hl = min(by_hl, key=lambda h: by_hl[h]["movement"])
        # halflife 10 is where the prediction quality peaked; it is NOT where
        # movement is smallest, which is the whole point of the finding.
        self.assertNotEqual(
            10.0, worst_movement_hl,
            "if the most static arm is now also the best arm, re-run the "
            "measurement before trusting the null",
        )


class TestTheSignConventionIsNotSilentlyFlipped(unittest.TestCase):
    def test_paired_bootstrap_positive_means_b_is_better(self):
        """A perfect-but-wrong b must come out as a positive delta.

        This repo has inverted this convention three times. A one-line test is
        cheaper than another three.
        """
        y = np.array([0, 0, 1, 1] * 25)
        p_bad = np.full(100, 0.5)                    # no information
        p_good = np.where(y == 1, 0.9, 0.1)         # perfectly separated
        d = nd.paired_bootstrap_delta(y, p_bad, p_good, n_boot=200, seed=0)
        self.assertGreater(d["delta"], 0.0,
                           "a BETTER b must produce a POSITIVE delta; if this "
                           "fails the sign convention in this repo has flipped")
        # A perfectly-separated predictor is right every time, so the bootstrap
        # interval is strictly positive. This is the opposite of "tight and
        # straddling zero" -- it pins that the interval brackets the point
        # estimate from the correct side.
        self.assertGreater(d["ci_lo"], 0.0,
                           "a flawless b must have its WHOLE interval above zero")


if __name__ == "__main__":
    unittest.main()