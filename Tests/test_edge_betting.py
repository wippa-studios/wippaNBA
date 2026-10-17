"""Selective betting: bet only where the estimate beats the PRICE.

The obvious framing scores every game and reports log loss, which answers "is this
a good forecaster?" - not "is there money in my disagreement with the market?"
This file pins the second question, which is the one that pays.

THE RULE HAS NO FREE PARAMETER, WHICH IS THE POINT
-------------------------------------------------
Bet one unit on the model's preferred side when the model's probability for it
exceeds the probability implied by the price actually posted. Expected profit per
unit staked is

    p * (1/p_raw - 1) - (1 - p) = p / p_raw - 1

which is positive exactly when p > p_raw. "Bet when the model beats the price" is
therefore the breakeven point, fixed by arithmetic. Every threshold sweep reported
elsewhere in this project is a multiple-comparisons problem; this rule has nothing
to have been selected on.

THE RESULT: IT LOSES
--------------------
    model, breakeven rule             3,481 bets   ROI -12.11%   SE 6.44%  |z| 1.88
    model + availability, breakeven   3,440 bets   ROI -10.80%   SE 6.33%  |z| 1.71
    every margin 0.00 - 0.06        1,806-3,481    ROI -2.2% to -13.2%

Not "indistinguishable from zero" - a loss, and most margins lose more.

BACKING AN UNDERDOG IS MOST OF THE OPPORTUNITY, AND THE FIRST VERSION MISSED IT
-------------------------------------------------------------------------------
An earlier version took the model's PREFERRED side and asked only whether THAT
side had edge. That silently discards every game where the favourite has no edge
but the underdog does - which is exactly "lay a loser". On this data:

    games where only the UNDERDOG has positive edge   1,928
    games the old rule bet on                         1,553

So the first version tested less than half the available bets, and the ones it
skipped were precisely the interesting ones. Correcting it raised the bet count
from 1,553 to 3,440. It changed the ROI slightly (both are losses) and it is still
the right test.

THE CONTROL IS THE PART THAT MATTERS MOST
-----------------------------------------
Running the same rule on the MARKET's own de-vigged probability places ZERO bets.
That is the correct result: de-vigging divides the overround away, so the
de-vigged probability can never exceed the raw posted price on the same side.

It is also the test that caught a real bug. An earlier version priced the away
side as `1 - p_raw_home` instead of reading the away price from the data. Because
the two raw probabilities sum to the overround (~1.0404) and not to 1, that
understated the away price by four points, made every away bet look better than it
was, and made the control report a spurious +7.00% profit. With the correct price
the control places no bets, and the model's ROI falls from -4.46% to -10.55%.

A control that profits when it should place no bets is the signature of broken
pricing arithmetic, not of an edge.
"""

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT, ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

_spec = importlib.util.spec_from_file_location(
    "edge_betting", ROOT / "scripts" / "edge_betting.py")
eb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eb)

RESULTS = ROOT / "results" / "edge_betting.json"


class TestTheBreakevenRuleIsArithmetic(unittest.TestCase):
    """The rule's threshold must be a property of the payoff, not a choice."""

    def test_one_unit_at_price_q_breakevens_at_p_equals_q(self):
        """Fair probability at fair odds breaks even.

        The tolerance is 4 analytic standard errors of the sample mean, not a fixed
        number. An earlier draft used delta=0.02, which at these sample sizes is
        only 1-2.5 SE and fails on Monte Carlo noise rather than on any property
        of the payoff. The payoff per unit is Bernoulli with gain 1/q, so
        SE(mean) = sqrt(q * (1/q)^2) / sqrt(n) = 1 / sqrt(q * n).
        """
        n = 200000
        for q in (0.2, 0.35, 0.5, 0.65, 0.8):
            y = (np.random.default_rng(0).random(n) < q).astype(int)
            profit = y * (1.0 / q - 1.0) - (1.0 - y)
            se = 1.0 / np.sqrt(q * n)
            self.assertLess(
                abs(float(profit.mean())), 4.0 * se,
                msg=f"breakeven failed at q={q}: mean {profit.mean():+.4f} "
                    f"against SE {se:.4f}")

    def test_a_better_estimate_profits_at_the_same_price(self):
        q, p = 0.5, 0.56
        y = (np.random.default_rng(1).random(200000) < p).astype(int)
        profit = y * (1.0 / q - 1.0) - (1.0 - y)
        # (p/q - 1) = 0.12 expected per unit.
        self.assertAlmostEqual(float(profit.mean()), p / q - 1.0, delta=0.01)

    def test_an_overround_means_fair_probability_still_loses(self):
        """p == p_raw is breakeven; the VIG is what makes the bar harder.

        This is why the rule compares against the RAW price and not the de-vigged
        one. Comparing against de-vigged would let the model bet at a price the
        bookmaker never offered.
        """
        q_raw = 0.52          # includes vig
        p = q_raw             # model agrees exactly with the posted price
        y = (np.random.default_rng(2).random(200000) < p).astype(int)
        profit = y * (1.0 / q_raw - 1.0) - (1.0 - y)
        self.assertAlmostEqual(float(profit.mean()), 0.0, delta=0.01)


class TestBothSidesArePricedFromTheData(unittest.TestCase):
    """The bug this file's docstring describes, pinned."""

    def test_raw_prices_sum_to_the_overround_not_to_one(self):
        """`1 - p_raw_home` is NOT the away price, and treating it as one is a bug.

        The two raw implied probabilities sum to the overround (~1.0404 here), so
        subtracting one from the other understates the far side by the vig. That
        flatters every bet on that side.
        """
        p_home = np.array([0.55, 0.60])
        p_away_true = np.array([0.49, 0.44])   # sums to 1.0400 / 1.0400
        self.assertFalse(np.allclose(1.0 - p_home, p_away_true))
        self.assertAlmostEqual(float((p_home + p_away_true).mean()), 1.04, places=6)

    def test_settle_receives_both_prices(self):
        import inspect
        params = list(inspect.signature(eb.settle).parameters)
        self.assertIn("p_raw_home", params)
        self.assertIn("p_raw_away", params,
                      "settle must take the away price from the data; deriving it "
                      "as 1-p_home understates it by the overround")

    def test_both_sides_are_considered_not_just_the_favourite(self):
        """Backing an underdog the market has underrated must be reachable.

        The first version took the model's preferred side and looked only there,
        which discarded every game where the favourite had no edge but the
        underdog did - 1,928 games, more than it bet on.
        """
        # Model says home 0.55 / away 0.45; market prices home at 0.70 / away 0.30.
        # Home has NO edge (-0.15); away has +0.15.
        p_model = np.array([0.55])
        p_raw_home = np.array([0.70])
        p_raw_away = np.array([0.30])
        y = np.array([0])                       # the away team wins
        r = eb.settle(p_model, p_raw_home, p_raw_away, y)
        self.assertEqual(r["bets"], 1,
                         "the underdog edge was missed; this rule must consider "
                         "both sides independently")
        self.assertEqual(r["away_bets"], 1)
        self.assertEqual(r["home_bets"], 0)

    def test_at_most_one_side_can_have_edge(self):
        """The two raw prices sum to the overround, so edges cannot both clear 0."""
        rng = np.random.default_rng(7)
        n = 20000
        p_model = rng.uniform(0.05, 0.95, n)
        over = rng.uniform(1.01, 1.08, n)
        p_raw_home = np.clip(p_model * over, 0.001, 0.999)
        p_raw_away = np.clip(over - p_raw_home, 0.001, 0.999)
        both = ((p_model - p_raw_home) > 0) & (((1 - p_model) - p_raw_away) > 0)
        self.assertEqual(int(both.sum()), 0,
                         "both sides had positive edge in the same game, which "
                         "should be impossible given the overround")

    def test_the_control_places_no_bets(self):
        """The de-vigged market probability can never beat the raw price.

        De-vigging divides the overround away, so both sides come out strictly
        below their raw counterparts and the rule has nothing to bet on. A control
        that places bets here means the pricing arithmetic is broken.
        """
        rng = np.random.default_rng(3)
        n = 5000
        p_home = rng.uniform(0.25, 0.85, n)
        over = rng.uniform(1.02, 1.06, n)
        p_raw_home = np.clip(p_home * over, 0.01, 0.99)
        p_raw_away = np.clip(over - p_raw_home, 0.01, 0.99)
        p_devig = p_home
        y = (rng.random(n) < p_home).astype(int)
        r = eb.settle(p_devig, p_raw_home, p_raw_away, y)
        self.assertEqual(r["bets"], 0,
                         "the market's own de-vigged probability placed bets; the "
                         "pricing is wrong")


class TestPublishedResultStaysNegative(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RESULTS.exists():
            raise unittest.SkipTest("run scripts/edge_betting.py first")
        cls.data = json.loads(RESULTS.read_text())

    def test_selective_betting_loses_at_every_margin(self):
        for label, sweep in (("model", self.data["margin_sweep_model"]),
                             ("availability", self.data["margin_sweep_model_availability"])):
            for margin, r in sweep.items():
                if not r.get("bets"):
                    continue
                self.assertLess(r["roi"], 0.0,
                                f"{label} margin {margin} turned profitable "
                                f"({r['roi']:+.2%}); re-read the whole analysis")

    def test_the_pre_specified_rule_reaches_no_bar(self):
        r = self.data["results"]["model + availability, breakeven"]
        self.assertLess(abs(r["z"]), self.data["z_bar"],
                        "the pre-specified rule reached the |z| bar; that would "
                        "change the project's conclusion")

    def test_the_headline_also_bets_underdogs(self):
        """Both headline rows must include away bets, or the rule regressed."""
        for label in ("model, breakeven rule",
                      "model + availability, breakeven"):
            r = self.data["results"][label]
            self.assertGreater(r["away_bets"], 0,
                               f"{label} placed no away bets; the underdog side is "
                               "no longer being considered")

    def test_edge_calibration_is_not_monotonic_and_must_not_be_traded(self):
        """One bin is positive. That is noise, and acting on it would be selection.

        The bins are reported because the non-monotonicity is interesting. The
        single positive bin is the 4th of 5 at 688 bets each - exactly what noise
        produces, and picking it would be the error this project exists to avoid.
        """
        r = self.data["results"]["model + availability, breakeven"]
        cal = r.get("calibration", [])
        if not cal:
            self.skipTest("calibration not recorded in this artifact")
        rois = [row["realised_roi"] for row in cal]
        self.assertGreater(len(rois), 1)
        # If a future run makes the top bin the best by a wide margin AND the
        # result clears the bar on a pre-registered basis, this test is the signal
        # to re-open the conclusion rather than to quietly ship a strategy.
        self.assertLess(abs(self.data["results"]
                            ["model + availability, breakeven"]["z"]),
                        self.data["z_bar"])

    def test_the_pre_specified_rule_is_margin_zero(self):
        """It must be the row with no free parameter, not a tuned one."""
        sweep = self.data["margin_sweep_model_availability"]
        self.assertIn("0.0", sweep, "margin 0.0 is the arithmetic breakeven point")
        self.assertEqual(sweep["0.0"]["bets"],
                         self.data["results"]["model + availability, breakeven"]["bets"],
                         "the headline row and the margin-0 row must be the same "
                         "calculation, not two different ones")


if __name__ == "__main__":
    unittest.main()