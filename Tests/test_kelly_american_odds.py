"""Behaviour pins for ``src/Utils/Kelly_Criterion.american_to_decimal``.

READ THIS BEFORE CHANGING THE FUNCTION
======================================
``american_to_decimal`` is misnamed and its docstring is wrong. It does NOT
convert American odds to decimal (European) odds. It returns **net odds**,
which is the true decimal odds minus 1:

    +110  ->  returns 1.1     true decimal odds are 2.1
    -110  ->  returns 0.91    true decimal odds are 1.909...
    +100  ->  returns 1.0     true decimal odds are 2.0

So a caller that does what the name promises and feeds this straight into a
Kelly or EV formula is silently off by 1.0 on every single line. Today the only
caller in the tree is ``calculate_kelly_criterion`` in this same module, which
absorbs the difference, which is the ONLY reason the bug is invisible. Any
second caller will get a wrong value.

These tests therefore pin the ACTUAL numeric behaviour, not the documented
intent, so that a refactor of this function is caught loudly rather than
silently changing the money-management numbers. They are characterization
tests: they assert what the code does today.

The behaviour is deliberately NOT corrected here. Changing the return value is
owned by a separate task; this file only documents and freezes it.

One more quirk worth knowing: the ``round(..., 2)`` on the return value is
load-bearing for downstream arithmetic. That rounding is why
``calculate_kelly_criterion(-110, .6)`` yields 16.04 rather than 16.00 (the
value you get from the unrounded net odds). Existing tests in
``Tests/test_kelly_criterion.py`` assert the rounded 16.04, so that behaviour
is intentionally left untouched by this file as well.
"""

import unittest

from src.Utils import Kelly_Criterion as kc


class TestAmericanToDecimal(unittest.TestCase):
    """Pin the real numeric output of american_to_decimal."""

    def test_negative_odds_return_net_odds_not_decimal_odds(self):
        # Negative (favourite) side: 100 / abs(odds).
        # True decimal odds for -110 are 1.909...; this returns 0.91.
        self.assertEqual(kc.american_to_decimal(-110), 0.91)
        self.assertEqual(kc.american_to_decimal(-150), 0.67)
        self.assertEqual(kc.american_to_decimal(-200), 0.5)
        self.assertEqual(kc.american_to_decimal(-300), 0.33)
        self.assertEqual(kc.american_to_decimal(-500), 0.2)

    def test_positive_odds_return_net_odds_not_decimal_odds(self):
        # Positive (underdog) side: odds / 100.
        # True decimal odds for +110 are 2.1; this returns 1.1.
        self.assertEqual(kc.american_to_decimal(110), 1.1)
        self.assertEqual(kc.american_to_decimal(150), 1.5)
        self.assertEqual(kc.american_to_decimal(200), 2.0)
        self.assertEqual(kc.american_to_decimal(250), 2.5)
        self.assertEqual(kc.american_to_decimal(275), 2.75)
        self.assertEqual(kc.american_to_decimal(400), 4.0)
        self.assertEqual(kc.american_to_decimal(550), 5.5)
        self.assertEqual(kc.american_to_decimal(1000), 10.0)

    def test_net_odds_is_always_true_decimal_minus_one(self):
        # The defining relationship, across the range of realistic moneylines.
        # If someone "fixes" this to return true decimal odds, every assertion
        # here flips to the +1 value and the change is impossible to miss.
        for american in (-500, -300, -200, -150, -110, -101, -100,
                         100, 101, 110, 150, 200, 250, 275, 400, 550, 1000):
            if american >= 100:
                true_decimal = 1 + (american / 100)
            else:
                true_decimal = 1 + (100 / abs(american))
            self.assertEqual(kc.american_to_decimal(american),
                             round(true_decimal - 1, 2))

    def test_even_money_sides_are_asymmetric_with_the_real_game(self):
        # +100 and -100 are both even money in American odds (true decimal 2.0),
        # and this function collapses both to 1.0 -- which happens to be the
        # correct NET odds. Pinned so the collapse is not mistaken for a bug in
        # the test data.
        self.assertEqual(kc.american_to_decimal(100), 1.0)
        self.assertEqual(kc.american_to_decimal(-100), 1.0)

    def test_short_odds_take_the_negative_side_branch(self):
        # The `else` branch fires for ANY value below 100, including positive
        # ones. +50 is not a standard American price (true decimal 1.5, net
        # 0.5), yet it returns 2.0 as though it were -50. Pinned because this is
        # the one input class where the "net odds" description does not even
        # hold -- it is a third, separate wrong answer.
        self.assertEqual(kc.american_to_decimal(50), 2.0)
        self.assertEqual(kc.american_to_decimal(-50), 2.0)
        self.assertEqual(kc.american_to_decimal(99), 1.01)
        self.assertEqual(kc.american_to_decimal(101), 1.01)

    def test_result_is_rounded_to_two_decimal_places(self):
        self.assertEqual(kc.american_to_decimal(-101), 0.99)
        self.assertEqual(kc.american_to_decimal(-111), 0.9)
        self.assertEqual(kc.american_to_decimal(333), 3.33)

    def test_extreme_short_odds_do_not_raise(self):
        # -1 / +1 hit 100 / abs(1). No validation, no exception.
        self.assertEqual(kc.american_to_decimal(-1), 100.0)
        self.assertEqual(kc.american_to_decimal(1), 100.0)

    def test_zero_odds_raise_zero_division_error(self):
        # There is no input validation at all: a 0 propagates into
        # 100 / abs(0) and blows up. Callers are expected to filter this.
        with self.assertRaises(ZeroDivisionError):
            kc.american_to_decimal(0)

    def test_single_caller_compensates_for_the_off_by_one(self):
        # calculate_kelly_criterion is the sole caller in the tree and is the
        # reason the misnamed return value has never surfaced. Its formula
        # divides by whatever american_to_decimal produced, so it works on
        # net odds today. Asserted here to make the coupling explicit: if
        # american_to_decimal is ever corrected to true decimal odds, this
        # expectation changes and the compensating algebra must be revisited.
        self.assertEqual(kc.calculate_kelly_criterion(-110, .6), 16.04)
        self.assertEqual(kc.calculate_kelly_criterion(-110, .4), 0)