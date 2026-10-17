"""The ROI denominator must be the money actually staked.

THE BUG THIS EXISTS FOR
-----------------------
`scripts/closing_line.py:roi()` used to compute

    total_staked = stake.sum()          # includes NEGATIVE stakes
    roi          = profit.sum() / total_staked

where `stake = p_model - p_raw`. That expression is negative whenever the market
is the more confident one, which at a 0.55 threshold is most of the slate. Those
games are NOT bets - the rule skips them - but they were counted into the
denominator anyway.

On the real data at threshold 0.55:

    positive stakes 812, sum +69.22   <- money actually at risk
    negative stakes 1382, sum -117.10 <- not bets at all
    stake.sum() = -47.88              <- the denominator as coded
    total profit = -6.48              <- a LOSS
    profit / sum(positive) = -9.37%   correct
    profit / stake.sum()  = +13.54%   as it was reported

A loss divided by a negative total reports as a profit. The headline number was
**+13.54% ROI on a strategy that lost 9.37%**, with the sign inverted by the
arithmetic rather than by anything about the betting.

WHY IT SURVIVED SO LONG
----------------------
The wrong answer was the one that looked good, and it looked good in a way that
survives a quick sanity check: the bet COUNT was correct (812), the stakes were
individually sensible, and every intermediate value was a plausible number. Only
the ratio was wrong. A best-of-three threshold sweep sitting on top of it made
+11.76% and +13.54% the two best cells of six, which is what a selection effect
looks like and is why it was initially written off as multiplicity rather than
arithmetic.

The same mistake was inside the bootstrap (`st = stake[pick]`), so the standard
error was computed over the wrong population as well. A wrong ROI with a
plausible-looking SE is harder to catch than either alone.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from closing_line import roi  # noqa: E402


class TestTheDenominatorIsMoneyStaked(unittest.TestCase):
    def test_a_game_where_the_price_is_better_is_not_money_at_risk(self):
        """The whole bug. Model 0.60, price says 0.65: no bet, and the negative
        difference must not reduce the amount at risk."""
        p_model = np.array([0.60])
        p_raw = np.array([0.65])
        y = np.array([1])
        out = roi(p_model, p_raw, y, threshold=0.55)
        self.assertEqual(out["bets"], 0, "this game must not count as a bet")
        self.assertTrue(np.isnan(out["roi"]),
                        "with no bets placed there is no ROI to report")

    def test_a_losing_strategy_reports_a_loss(self):
        """The headline case. Four games, model consistently wrong, and one of
        them has the price better so it must not enter the denominator."""
        p_model = np.array([0.60, 0.62, 0.58, 0.70])
        p_raw = np.array([0.55, 0.57, 0.65, 0.68])
        y = np.array([0, 0, 1, 0])          # the model is wrong on the first two
        out = roi(p_model, p_raw, y, threshold=0.55)
        self.assertEqual(out["bets"], 3, "the 0.58-vs-0.65 game is not a bet")
        self.assertLess(out["roi"], 0.0,
                        "a strategy that loses money must report a negative ROI")

    def test_a_winning_strategy_reports_a_profit(self):
        """The mirror image, so the fix is not merely sign-flipping the answer."""
        p_model = np.array([0.70, 0.72, 0.66])
        p_raw = np.array([0.55, 0.60, 0.58])
        y = np.array([1, 1, 1])
        out = roi(p_model, p_raw, y, threshold=0.55)
        self.assertEqual(out["bets"], 3)
        self.assertGreater(out["roi"], 0.0)

    def test_the_return_is_roi_over_staked_not_over_a_signed_sum(self):
        """Recompute by hand and require agreement to 1e-12."""
        p_model = np.array([0.60, 0.75, 0.58, 0.80, 0.55])
        p_raw = np.array([0.55, 0.70, 0.65, 0.75, 0.50])
        y = np.array([1, 0, 1, 1, 0])
        thr = 0.55
        out = roi(p_model, p_raw, y, threshold=thr)

        placed = np.where((p_model >= thr) & (p_model - p_raw > 0))
        stake = p_model[placed] - p_raw[placed]
        won = y[placed].astype(float)
        profit = (won - p_raw[placed]) / p_raw[placed] * stake
        self.assertAlmostEqual(out["bets"], len(placed[0]), places=12)
        self.assertAlmostEqual(out["roi"], profit.sum() / stake.sum(), places=12)


class TestTheSignCannotBeInverted(unittest.TestCase):
    def test_the_denominator_is_positive_whenever_bets_are_placed(self):
        """If this ever fails, the ROI sign is once again at the mercy of a sum
        that can go negative. Asserted structurally rather than on one example."""
        rng = np.random.default_rng(0)
        for trial in range(200):
            p_raw = rng.uniform(0.35, 0.65, size=40)
            p_model = np.clip(p_raw + rng.normal(0, 0.08, size=40), 0.01, 0.99)
            y = rng.integers(0, 2, size=40)
            stake = np.where(p_model >= 0.55, p_model - p_raw, 0.0)
            placed = stake > 0
            if not placed.any():
                continue
            total = stake[placed].sum()
            self.assertGreater(total, 0.0,
                               f"trial {trial}: placed stakes sum to {total}")
            out = roi(p_model, p_raw, y, threshold=0.55)
            if not np.isnan(out["roi"]):
                self.assertGreater(out["roi"], -1.0,
                                   f"trial {trial}: ROI {out['roi']} below -100%")

    def test_roi_and_profit_always_agree_in_sign(self):
        """The invariant the bug violated: the sign of the ROI is the sign of the
        total profit, full stop."""
        rng = np.random.default_rng(7)
        for _ in range(200):
            p_raw = rng.uniform(0.35, 0.65, size=30)
            p_model = np.clip(p_raw + rng.normal(0, 0.1, size=30), 0.01, 0.99)
            y = rng.integers(0, 2, size=30)
            thr = float(rng.choice([0.5, 0.55, 0.6, 0.65]))
            out = roi(p_model, p_raw, y, threshold=thr)
            stake = np.where(p_model >= thr, p_model - p_raw, 0.0)
            placed = stake > 0
            if not placed.any() or np.isnan(out["roi"]):
                continue
            won = y.astype(float)
            # Mask BEFORE summing. Writing this without the mask reproduced the
            # original bug inside the test that exists to catch it: summing over
            # the unmasked column folded negative stakes into the reference
            # profit, so the assertion compared the fix against the bug and
            # failed for the right reason at the wrong layer.
            profit = float(np.sum(np.where(placed,
                                           (won - p_raw) / p_raw * stake, 0.0)))
            if profit > 0:
                self.assertGreater(out["roi"], 0.0, "profit but reported a loss")
            elif profit < 0:
                self.assertLess(out["roi"], 0.0, "loss but reported a profit")


if __name__ == "__main__":
    unittest.main()