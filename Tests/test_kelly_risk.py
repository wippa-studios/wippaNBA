"""Tests for the fractional-Kelly risk controls in ``Kelly_Criterion``.

READ THIS BEFORE CHANGING THE FUNCTIONS TESTED HERE
==================================================

These tests cover the POLICY layer added above ``calculate_kelly_criterion``
-- de-vig, shrinkage, fractional Kelly, caps and slate exposure. They do not
re-test the arithmetic underneath, which is owned by
``Tests/test_kelly_criterion.py`` and pinned there.

The single most important property of this file is that it is a CHECK on the
policy, not a description of it. Almost every assertion here is a number
computed by hand from the textbook definition and written down in the test
before the implementation was allowed to produce it. If a change to the risk
controls flips one of these, that is a deliberate policy decision that needs
an explicit update to the expected value and the comment beside it -- not an
accident to be papered over.

The frozen behaviour of ``american_to_decimal`` (net odds, not decimal odds)
and ``calculate_kelly_criterion`` is deliberately not exercised for its own
sake here; see ``Tests/test_kelly_american_odds.py``. Several tests below do
depend on that rounding (``-110`` is net odds ``0.91``, so full Kelly at
``p = 0.55`` is ``5.55`` and not the exact ``5.50``), and those dependencies
are spelled out at the point of use.
"""

import math
import unittest

from src.Utils import Kelly_Criterion as kc


class TestImpliedProbability(unittest.TestCase):
    """The vig-inclusive market price, as a probability."""

    def test_common_prices(self):
        # 1 / (1 + b), computed off the RAW price so it does not inherit
        # american_to_decimal's round(x, 2). -110 is 11/21 exactly; -500 is a
        # FAVOURITE at 1/1.2, so its implied probability is high, not low --
        # net odds are not probabilities and the two must not be confused.
        self.assertEqual(kc.implied_probability(-110), 0.52381)
        self.assertEqual(kc.implied_probability(110), 0.47619)
        self.assertEqual(kc.implied_probability(-150), 0.6)
        self.assertEqual(kc.implied_probability(150), 0.4)
        self.assertEqual(kc.implied_probability(-500), 0.833333)
        self.assertEqual(kc.implied_probability(400), 0.2)

    def test_relationship_to_american_to_decimal_is_one_over_one_plus_net(self):
        # implied = 1 / (1 + b) where b is the NET odds american_to_decimal
        # returns. The two disagree by more than the rounding tolerance,
        # which is exactly why implied_probability re-derives the price.
        # For -110 the true probability is 1/1.909090... = 0.523810 while
        # the rounded net odds give 1/1.91 = 0.523560 -- a gap of 0.00025,
        # the same order as the edges being hunted. The rounding pushes the
        # net odds UP, so the reciprocal path UNDERSTATES the market.
        rounded_path = 1.0 / (1.0 + kc.american_to_decimal(-110))
        self.assertAlmostEqual(kc.implied_probability(-110), rounded_path, delta=0.001)
        self.assertGreater(kc.implied_probability(-110), rounded_path)

    def test_mirrored_pairs_sum_to_exactly_one(self):
        # Exact algebra: 1/(1+b) + 1/(1+1/b) = 1. This is why size_bet's
        # counterpart inference is the identity when the caller passes
        # implied_probability() back in, and it is asserted so that identity
        # is a known property rather than a surprise.
        for odds in (100, 110, 150, 200, 300, 500):
            self.assertAlmostEqual(
                kc.implied_probability(odds) + kc.implied_probability(-odds), 1.0, places=6
            )

    def test_realistic_two_way_book_carries_vig_above_one(self):
        # A real book does NOT quote mirrored prices. -110 against +105 is
        # 0.523810 + 0.487805 = 1.011615, a 1.16% margin. This is the number
        # devig_two_way exists to remove, and it is why a single quoted price
        # cannot be treated as a probability.
        self.assertAlmostEqual(
            kc.implied_probability(-110) + kc.implied_probability(105), 1.011615, places=6
        )
        self.assertGreater(kc.implied_probability(-110) + kc.implied_probability(105), 1.0)

    def test_probability_is_always_a_probability(self):
        for american in range(-600, 601, 7):
            if american == 0:
                continue
            value = kc.implied_probability(american)
            self.assertGreater(value, 0.0)
            self.assertLess(value, 1.0)

    def test_zero_odds_raise_zero_division_error(self):
        # Same non-validation as american_to_decimal. Callers filter prices.
        with self.assertRaises(ZeroDivisionError):
            kc.implied_probability(0)


class TestDevigTwoWay(unittest.TestCase):
    """Removing the book margin from a pair of quoted prices."""

    def test_proportional_normalises_the_pair_to_one(self):
        fair_a, fair_b = kc.devig_two_way(0.55, 0.50)
        self.assertAlmostEqual(fair_a + fair_b, 1.0, places=6)
        self.assertAlmostEqual(fair_a, 0.55 / 1.05, places=6)
        self.assertAlmostEqual(fair_b, 0.50 / 1.05, places=6)

    def test_proportional_preserves_the_ratio_between_the_sides(self):
        # Dividing both sides by the same total cannot change their ratio.
        # That property is what "proportional" means, and it is also its
        # weakness: a book that shades longshots harder is not fixed by it.
        quoted_a, quoted_b = 0.62, 0.44
        fair_a, fair_b = kc.devig_two_way(quoted_a, quoted_b)
        self.assertAlmostEqual(fair_a / fair_b, quoted_a / quoted_b, places=5)

    def test_shin_removes_more_from_the_longshot_than_proportional(self):
        # The favourite-longshot direction, asserted directly. Both methods
        # give the favourite a fair probability; Shin gives it MORE of the
        # total, which is the entire point of using it when FLS bias is the
        # concern: the favourite ends up at 0.525 rather than 0.52381, and
        # the longshot at 0.475 rather than 0.47619.
        #
        # Neither fair number may exceed its own QUOTED number, though --
        # removing a 1.05 overround from a 0.55 quote cannot leave 0.55. Shin
        # moves mass BETWEEN the two sides, it does not create it.
        prop_a, prop_b = kc.devig_two_way(0.55, 0.50)
        shin_a, shin_b = kc.devig_two_way(0.55, 0.50, method="shin")
        self.assertAlmostEqual(shin_a + shin_b, 1.0, places=6)
        self.assertGreater(shin_a, prop_a)
        self.assertLess(shin_b, prop_b)
        # ... and both sides still come in BELOW their own quoted numbers,
        # because removing a 1.05 overround from 0.55 and 0.50 cannot leave
        # either one higher. Shin moves mass BETWEEN the two sides, it does
        # not create it.
        self.assertLess(shin_a, 0.55)
        self.assertLess(shin_b, 0.50)

    def test_shin_is_symmetric_under_swapping_the_sides(self):
        shin_a, shin_b = kc.devig_two_way(0.55, 0.50, method="shin")
        swapped_a, swapped_b = kc.devig_two_way(0.50, 0.55, method="shin")
        self.assertAlmostEqual(shin_a, swapped_b, places=6)
        self.assertAlmostEqual(shin_b, swapped_a, places=6)

    def test_both_methods_fall_back_to_identity_on_a_vig_free_pair(self):
        # A pair summing to exactly 1.0 has no margin. Shin's informed-fraction
        # root does not exist there, so it must not crash and must not invent
        # a margin.
        for method in ("proportional", "shin"):
            fair_a, fair_b = kc.devig_two_way(0.48, 0.52, method=method)
            self.assertAlmostEqual(fair_a, 0.48, places=6)
            self.assertAlmostEqual(fair_b, 0.52, places=6)

    def test_unknown_method_raises_rather_than_defaulting(self):
        with self.assertRaises(ValueError):
            kc.devig_two_way(0.55, 0.50, method="power")
        with self.assertRaises(ValueError):
            kc.devig_two_way(0.55, 0.50, method=None)

    def test_zero_pair_raises(self):
        with self.assertRaises(ValueError):
            kc.devig_two_way(0.0, 0.0)


class TestShrinkProbability(unittest.TestCase):
    """Linear shrinkage of a model probability toward the market's."""

    def test_endpoints_are_the_two_inputs(self):
        self.assertEqual(kc.shrink_probability(0.55, 0.48, 0.0), 0.55)
        self.assertEqual(kc.shrink_probability(0.55, 0.48, 1.0), 0.48)

    def test_half_weight_is_the_plain_midpoint(self):
        self.assertEqual(kc.shrink_probability(0.55, 0.48, 0.5), 0.515)

    def test_shrinking_moves_strictly_toward_the_market(self):
        # The only property that matters operationally: if the model already
        # agrees with the market, nothing moves; if it disagrees, the result
        # lands strictly between the two and never outside.
        for weight in (0.1, 0.25, 0.5, 0.75, 0.9):
            shrunk = kc.shrink_probability(0.60, 0.50, weight)
            self.assertGreater(shrunk, 0.50)
            self.assertLess(shrunk, 0.60)

    def test_full_weight_removes_the_entire_edge(self):
        # Shrinking all the way to the market must leave no edge at all --
        # this is the property that makes shrinkage a risk control rather
        # than a haircut.
        market = kc.implied_probability(-110)
        shrunk = kc.shrink_probability(0.55, market, 1.0)
        self.assertAlmostEqual(shrunk, market, places=6)
        self.assertLessEqual(kc.calculate_kelly_criterion(-110, shrunk), 0.06)

    def test_weight_outside_zero_to_one_raises(self):
        for weight in (-0.1, 1.1, 2, float("nan"), float("inf"), None):
            with self.assertRaises(ValueError):
                kc.shrink_probability(0.55, 0.48, weight)

    def test_probabilities_outside_zero_to_one_are_clamped(self):
        self.assertEqual(kc.shrink_probability(1.4, 0.48, 1.0), 0.48)
        self.assertEqual(kc.shrink_probability(0.55, -0.2, 1.0), 0.0)

    def test_non_finite_probability_falls_back_to_a_coin_flip(self):
        # nan and inf have no clamped value, so both become 0.5. Each side
        # of the interpolation is checked separately, against a market that
        # is far enough from 0.5 that the fallback cannot be confused with
        # either input.
        for bad in (float("nan"), float("inf"), float("-inf")):
            # weight 0.0 exposes the MODEL fallback
            self.assertEqual(kc.shrink_probability(bad, 0.30, 0.0), 0.5)
            # weight 1.0 exposes the MARKET fallback
            self.assertEqual(kc.shrink_probability(0.55, bad, 1.0), 0.5)


class TestFractionalKelly(unittest.TestCase):
    """Full Kelly scaled by a fraction. Same arithmetic, smaller stake."""

    def test_quarter_kelly_is_a_quarter_of_full(self):
        # Hand check: full Kelly at -110 with p = 0.55 uses net odds 0.91, so
        #   100 * (0.91 * 0.55 - 0.45) / 0.91 = 100 * 0.0505 / 0.91 = 5.54945
        # which the frozen arithmetic rounds to 5.55. A quarter of that is
        # 1.3875. (The exact textbook value would be 5.50; the difference is
        # american_to_decimal's load-bearing round(x, 2).)
        self.assertEqual(kc.calculate_kelly_criterion(-110, 0.55), 5.55)
        self.assertEqual(kc.fractional_kelly(-110, 0.55), 1.3875)
        self.assertEqual(kc.fractional_kelly(-110, 0.55, 0.25), 1.3875)

    def test_fraction_of_one_is_exactly_full_kelly(self):
        for american, prob in ((-110, 0.6), (400, 0.35), (-500, 0.85), (150, 0.62)):
            self.assertEqual(
                kc.fractional_kelly(american, prob, 1.0),
                kc.calculate_kelly_criterion(american, prob),
            )

    def test_fraction_scales_linearly(self):
        # Half of full Kelly must be exactly twice a quarter of it, and
        # smaller fractions must produce smaller stakes. Hand check: full
        # Kelly at -110 with p = 0.62 is
        #   100 * (0.91 * 0.62 - 0.38) / 0.91 = 100 * 0.1842 / 0.91 = 20.2418
        # -> 20.24 frozen-rounded, so a quarter is 5.06 and a half is 10.12.
        full = kc.calculate_kelly_criterion(-110, 0.62)
        self.assertEqual(full, 20.24)
        self.assertEqual(kc.fractional_kelly(-110, 0.62, 0.25), 5.06)
        self.assertEqual(kc.fractional_kelly(-110, 0.62, 0.5), 10.12)
        self.assertAlmostEqual(
            kc.fractional_kelly(-110, 0.62, 0.5),
            2 * kc.fractional_kelly(-110, 0.62, 0.25),
            places=6,
        )
        self.assertLess(kc.fractional_kelly(-110, 0.62, 0.1), kc.fractional_kelly(-110, 0.62, 0.25))
        self.assertLess(kc.fractional_kelly(-110, 0.62, 0.25), full)

    def test_never_returns_a_negative_stake(self):
        # No edge: full Kelly floors at 0, so every fraction does too.
        for american, prob in ((-110, 0.4), (-110, 0.1), (400, 0.15), (-500, 0.2)):
            for fraction in (0.1, 0.25, 0.5, 1.0):
                self.assertGreaterEqual(kc.fractional_kelly(american, prob, fraction), 0)

    def test_fraction_must_be_in_zero_exclusive_to_one(self):
        # 0.0 is rejected on purpose: "bet nothing" is a decision the caller
        # should make, not something a sizing helper should absorb.
        for fraction in (0.0, -0.25, 1.01, 2, float("nan"), float("inf"), None):
            with self.assertRaises(ValueError):
                kc.fractional_kelly(-110, 0.55, fraction)

    def test_out_of_range_probability_is_clamped_not_rejected(self):
        # Clamping is a documented last resort. A caller passing 55 means
        # 55%; here that clamps to 1.0 (a certainty) and a certainty at any
        # real price is a maximum stake, which is the correct consequence.
        self.assertEqual(
            kc.fractional_kelly(-110, 1.5), kc.fractional_kelly(-110, 1.0)
        )
        self.assertEqual(kc.fractional_kelly(-110, -0.5), 0)

    def test_non_finite_probability_becomes_a_coin_flip(self):
        # 0.5 against -110 is exactly the break-even of the vig, so no bet.
        for bad in (float("nan"), float("inf"), float("-inf")):
            self.assertEqual(kc.fractional_kelly(-110, bad), 0)


class TestSizeBet(unittest.TestCase):
    """The composed policy function. Every step must be reconstructible."""

    def test_worked_example_by_hand(self):
        # size_bet(-110, model 0.55, market 0.5238, quarter Kelly, 5% cap)
        #
        # 1. market 0.5238 is de-vigged against the counterpart inferred from
        #    the -110 price (1 - 0.523810 = 0.476190). 0.5238 + 0.476190 =
        #    0.999990, so proportional de-vig gives 0.5238 / 0.999990 =
        #    0.523805. This is the identity in all but the fifth decimal,
        #    because a mirrored pair carries no overround.
        # 2. shrink_weight is 0, so the model probability is untouched: 0.55.
        # 3. full Kelly = 100 * (0.91 * 0.55 - 0.45) / 0.91 = 5.54945 -> 5.55
        # 4. a quarter of 5.55 is 1.3875
        # 5. 1.3875 is under the 5.0 cap, so the cap does not bind.
        result = kc.size_bet(-110, 0.55, market_prob=0.5238, fraction=0.25)

        self.assertAlmostEqual(result["market_prob_vig_free"], 0.523805, places=6)
        self.assertEqual(result["probability_used"], 0.55)
        self.assertEqual(result["raw_kelly_pct"], 5.55)
        self.assertEqual(result["fractional_kelly_pct"], 1.3875)
        self.assertEqual(result["stake_pct"], 1.3875)
        self.assertFalse(result["capped"])
        self.assertFalse(result["no_edge"])

    def test_full_kelly_alone_would_have_bet_five_and_a_half_percent(self):
        # The whole point of the change, stated as a number: the same edge
        # under the old policy is 5.55% of bankroll on one line.
        self.assertEqual(kc.calculate_kelly_criterion(-110, 0.55), 5.55)
        self.assertEqual(kc.size_bet(-110, 0.55)["stake_pct"], 1.3875)

    def test_shrinking_to_the_market_removes_the_edge(self):
        # shrink_weight 1.0 replaces the model probability with the market's,
        # and the market price is by definition close to fair after a de-vig,
        # so the stake collapses toward zero. 0.523805 at -110 is a 0.05%
        # full-Kelly edge, a quarter of which is 0.0125.
        result = kc.size_bet(-110, 0.55, market_prob=0.5238, shrink_weight=1.0)
        self.assertEqual(result["probability_used"], 0.523805)
        self.assertEqual(result["raw_kelly_pct"], 0.05)
        self.assertEqual(result["stake_pct"], 0.0125)

    def test_partial_shrinkage_lands_between_model_and_market(self):
        # The full chain, hand-checked in three steps, including the de-vig
        # -- which is NOT a no-op here and that is the point of asserting it:
        #
        # 1. the supplied 0.50 is below the 0.523810 the -110 price implies,
        #    so the pair (0.50, 1 - 0.523810) = (0.50, 0.476190) sums to
        #    0.976190, i.e. it is an UNDERROUND. Proportional de-vig
        #    renormalises it up: 0.50 / 0.976190 = 0.512195.
        # 2. quarter-weight shrinkage toward that: 0.75 * 0.60 + 0.25 *
        #    0.512195 = 0.45 + 0.128049 = 0.578049.
        # 3. a market number that disagrees with the offered price IS what
        #    makes the de-vig step do work.
        result = kc.size_bet(-110, 0.60, market_prob=0.50, shrink_weight=0.25)
        self.assertAlmostEqual(result["market_prob_vig_free"], 0.512195, places=6)
        self.assertAlmostEqual(result["probability_used"], 0.578049, places=6)
        self.assertLess(result["probability_used"], result["model_prob"])

    def test_shrinkage_reduces_the_stake_monotonically(self):
        stakes = [
            kc.size_bet(-110, 0.60, market_prob=0.50, shrink_weight=w)["stake_pct"]
            for w in (0.0, 0.25, 0.5, 0.75, 1.0)
        ]
        self.assertEqual(stakes, sorted(stakes, reverse=True))
        self.assertGreater(stakes[0], stakes[-1])

    def test_cap_binds_and_is_reported(self):
        # -110 at p = 0.90 is a huge edge: full Kelly is
        # 100 * (0.91 * 0.9 - 0.1) / 0.91 = 79.01. A 5% per-bet cap must
        # bind, and the audit dict must say so rather than returning a
        # quiet 5.0.
        result = kc.size_bet(-110, 0.90, fraction=1.0, max_bet_pct=5.0)
        self.assertEqual(result["raw_kelly_pct"], 79.01)
        self.assertEqual(result["fractional_kelly_pct"], 79.01)
        self.assertEqual(result["stake_pct"], 5.0)
        self.assertTrue(result["capped"])
        self.assertEqual(result["cap_pct"], 5.0)

    def test_cap_reports_not_capped_when_it_just_escapes(self):
        result = kc.size_bet(-110, 0.90, fraction=0.25, max_bet_pct=5.0)
        # quarter of 79.01 is 19.7525, still over the cap
        self.assertEqual(result["fractional_kelly_pct"], 19.7525)
        self.assertTrue(result["capped"])
        result = kc.size_bet(-110, 0.90, fraction=0.25, max_bet_pct=25.0)
        self.assertEqual(result["stake_pct"], 19.7525)
        self.assertFalse(result["capped"])

    def test_zero_cap_bets_nothing(self):
        result = kc.size_bet(-110, 0.90, max_bet_pct=0.0)
        self.assertEqual(result["stake_pct"], 0.0)
        self.assertTrue(result["capped"])

    def test_no_edge_is_reported_and_stakes_zero(self):
        result = kc.size_bet(-110, 0.45)
        self.assertEqual(result["stake_pct"], 0)
        self.assertTrue(result["no_edge"])
        self.assertFalse(result["capped"])

    def test_shrinkage_without_a_market_prob_raises(self):
        # A risk control that silently does nothing is worse than none,
        # because the caller believes it is applied.
        with self.assertRaises(ValueError):
            kc.size_bet(-110, 0.55, shrink_weight=0.25)

    def test_zero_shrinkage_without_a_market_prob_is_explicit_and_allowed(self):
        result = kc.size_bet(-110, 0.55, shrink_weight=0.0)
        self.assertIsNone(result["market_prob_vig_inclusive"])
        self.assertIsNone(result["market_prob_vig_free"])
        self.assertEqual(result["stake_pct"], 1.3875)

    def test_audit_dict_exposes_every_intermediate_step(self):
        # Silent magic is a defect in this repo. If a key disappears, a step
        # became unobservable, and that is a regression even if the stake is
        # unchanged.
        result = kc.size_bet(-110, 0.55, market_prob=0.5238, fraction=0.25)
        self.assertEqual(
            sorted(result.keys()),
            sorted([
                "stake_pct", "raw_kelly_pct", "fractional_kelly_pct", "cap_pct",
                "capped", "no_edge", "probability_used", "model_prob",
                "market_prob_vig_inclusive", "market_prob_vig_free",
                "fraction", "shrink_weight",
            ]),
        )
        self.assertEqual(result["model_prob"], 0.55)
        self.assertEqual(result["market_prob_vig_inclusive"], 0.5238)
        self.assertEqual(result["fraction"], 0.25)
        self.assertEqual(result["shrink_weight"], 0.0)

    def test_stake_is_never_negative_and_never_exceeds_the_cap(self):
        for american in (-600, -300, -110, 100, 150, 400, 1000):
            for prob in (0.05, 0.2, 0.5, 0.8, 0.99):
                for fraction in (0.1, 0.25, 1.0):
                    for cap in (0.0, 1.0, 5.0):
                        result = kc.size_bet(american, prob, fraction=fraction, max_bet_pct=cap)
                        self.assertGreaterEqual(result["stake_pct"], 0)
                        self.assertLessEqual(result["stake_pct"], cap)

    def test_agrees_with_the_component_functions_it_composes(self):
        # The composition must not quietly re-derive anything: each stage has
        # to be the function it claims to call.
        result = kc.size_bet(-300, 0.72, market_prob=0.30, fraction=0.5, max_bet_pct=50.0)
        self.assertEqual(result["model_prob"], 0.72)
        self.assertEqual(result["market_prob_vig_inclusive"], 0.30)
        self.assertAlmostEqual(
            result["market_prob_vig_free"],
            kc.devig_two_way(0.30, 1.0 - kc.implied_probability(-300))[0],
            places=6,
        )
        self.assertEqual(result["probability_used"], result["model_prob"])
        self.assertEqual(
            result["raw_kelly_pct"], kc.calculate_kelly_criterion(-300, 0.72)
        )
        self.assertEqual(
            result["stake_pct"], kc.fractional_kelly(-300, 0.72, 0.5)
        )

    def test_fraction_and_shrink_weight_are_validated(self):
        with self.assertRaises(ValueError):
            kc.size_bet(-110, 0.55, fraction=0.0)
        with self.assertRaises(ValueError):
            kc.size_bet(-110, 0.55, fraction=1.5)
        with self.assertRaises(ValueError):
            kc.size_bet(-110, 0.55, market_prob=0.5238, shrink_weight=1.5)

    def test_non_finite_cap_is_treated_as_zero(self):
        result = kc.size_bet(-110, 0.55, max_bet_pct=float("nan"))
        self.assertEqual(result["cap_pct"], 0.0)
        self.assertEqual(result["stake_pct"], 0.0)


class TestCorrelatedExposureCap(unittest.TestCase):
    """The 'eight bets at 5% is not eight independent bets' control."""

    def test_slate_over_the_cap_is_scaled_down_to_it(self):
        # Eight 5% bets sum to 40%. A 15% cap must scale every one by
        # 15/40 = 0.375, giving 1.875 each and a total of exactly 15.
        slate = [5.0] * 8
        capped = kc.correlated_exposure_cap(slate, 15.0)
        for stake in capped:
            self.assertAlmostEqual(stake, 1.875, places=9)
        self.assertAlmostEqual(sum(capped), 15.0, places=9)

    def test_slate_under_the_cap_is_returned_unchanged(self):
        slate = [2.0, 1.5, 1.0, 0.5]
        capped = kc.correlated_exposure_cap(slate, 15.0)
        self.assertEqual(capped, slate)
        self.assertEqual(sum(capped), 5.0)

    def test_slate_exactly_at_the_cap_is_unchanged(self):
        slate = [5.0, 5.0, 5.0]
        self.assertEqual(kc.correlated_exposure_cap(slate, 15.0), slate)

    def test_scaling_preserves_relative_ranking(self):
        # Proportional scaling shrinks the big line and the small line by the
        # same factor. That is the documented weakness, asserted so it stays
        # documented rather than becoming folklore.
        slate = [12.0, 1.0]
        capped = kc.correlated_exposure_cap(slate, 6.5)
        self.assertAlmostEqual(capped[0] / capped[1], 12.0, places=9)
        self.assertLess(capped[1], capped[0])

    def test_total_never_exceeds_the_cap(self):
        import random
        generator = random.Random(20241016)
        for _ in range(500):
            slate = [generator.uniform(0.0, 9.0) for _ in range(generator.randint(1, 12))]
            cap = generator.uniform(0.0, 40.0)
            capped = kc.correlated_exposure_cap(slate, cap)
            self.assertLessEqual(sum(capped), cap + 1e-9)
            for stake in capped:
                self.assertGreaterEqual(stake, 0.0)

    def test_empty_slate(self):
        self.assertEqual(kc.correlated_exposure_cap([], 10.0), [])

    def test_zero_total_needs_no_scaling(self):
        # There is no direction to distribute into; a zero cap must not
        # produce a division by zero.
        self.assertEqual(kc.correlated_exposure_cap([0.0, 0.0], 0.0), [0.0, 0.0])
        self.assertEqual(kc.correlated_exposure_cap([0.0, 0.0], 5.0), [0.0, 0.0])

    def test_zero_cap_flattens_everything(self):
        self.assertEqual(kc.correlated_exposure_cap([5.0, 3.0, 2.0], 0.0), [0.0, 0.0, 0.0])

    def test_negative_and_non_finite_entries_are_zeroed_not_propagated(self):
        capped = kc.correlated_exposure_cap([-4.0, float("nan"), 6.0], 6.0)
        self.assertEqual(capped, [0.0, 0.0, 6.0])
        for stake in capped:
            self.assertGreaterEqual(stake, 0.0)
            self.assertTrue(math.isfinite(stake))

    def test_negative_cap_is_treated_as_zero(self):
        self.assertEqual(kc.correlated_exposure_cap([5.0, 5.0], -10.0), [0.0, 0.0])

    def test_composes_with_size_bet_output(self):
        # The realistic use: size eight bets, cap the slate. Each 5% line is
        # individually within the per-bet cap, and only the slate cap catches
        # the aggregate.
        stakes = [kc.size_bet(-110, 0.55, max_bet_pct=5.0)["stake_pct"] for _ in range(8)]
        self.assertEqual(stakes, [1.3875] * 8)
        self.assertLessEqual(max(stakes), 5.0)
        capped = kc.correlated_exposure_cap(stakes, 10.0)
        self.assertAlmostEqual(sum(capped), 10.0, places=9)
        self.assertLess(max(capped), max(stakes))


if __name__ == "__main__":
    unittest.main()