"""Availability is the first effect in this project that is not a null.

Everything else measured here has come back a null: player ratings (+0.00052),
the 52 rank columns (0.0018, interval crossing zero), gradient boosting (loses to
logistic regression), recency weighting (halflife 10 is already optimal), and the
rebound block (+0.00040, interval crossing zero).

Availability is the exception, and it is also the effect with the most dangerous
history here: substituting ACTUAL minutes for expected minutes changes log loss by
~0.01 nats, and ACTUAL minutes are known only after the game. That number is a
leak, and a leak that produces excellent numbers is worse than no model at all.

These tests pin the bounded claim, which is much smaller than the leak:

    roster size ONLY (leak control)     +0.00109   CI [-0.00073, +0.00299]  straddles zero
    STRICT-ROSTER expected minutes      +0.00382   CI [+0.00230, +0.00547]  EXCLUDES ZERO
    expected over ACTUAL squad          +0.01081   EXCLUDES ZERO
    ACTUAL minutes (full leak)          +0.01089   EXCLUDES ZERO

The strictly-past arm sums each player's expected minutes over the squad that
appeared in that team's PREVIOUS game. It cannot depend on who dresses tonight,
so it carries no post-hoc knowledge.

The leak control exists because the obvious feature is subtly contaminated:
summing expected minutes over the players who ACTUALLY played makes a team missing
its star score lower, which is knowledge you only have after tip-off. The control
isolates roster SIZE, carrying no minutes information at all, and it does NOT
reproduce the gain. That is the test that makes the bounded claim trustworthy.
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

RESULTS = ROOT / "results" / "availability.json"


class TestStrictRosterIsStrictlyPast(unittest.TestCase):
    """The feature must not know who dresses tonight."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.games = nd.build_games()
        except Exception as exc:
            raise unittest.SkipTest(f"NBA data unavailable: {exc}")

    def test_roster_comes_from_the_previous_game_not_this_one(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "avail_script", ROOT / "scripts" / "availability.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        sub = self.games.head(1500)
        strict = mod.strict_roster_expectation(sub)
        self.assertEqual(len(strict), len(sub))
        # Coverage must be high but NOT total: a team's first game has no previous
        # game, so the very first entries are NaN. If this were 100%, the "previous"
        # game is somehow this game.
        self.assertGreater(strict["home_sexp"].notna().mean(), 0.9)
        self.assertLess(strict["home_sexp"].notna().mean(), 1.0,
                        "a roster known before tip-off must be unavailable for a "
                        "team's first game on record")

    def test_truncating_the_future_changes_nothing(self):
        """The property the whole claim rests on.

        The tolerance is 1e-9 rather than exact equality, and the reason is worth
        stating: the sum is a `merge_asof` lookup over a frame whose LENGTH differs
        between the two calls, so the order floats are summed in differs and the
        result carries ~1e-14 of rounding. Observed maximum drift is 5.68e-14.

        Exact `== 0` here would fail on that noise while passing on a genuine leak
        of any size below the tolerance - so the check is a tight bound, and the
        NaN pattern is compared exactly, because a cell that becomes finite when
        future data is added is a leak with no tolerance that could excuse it.
        """
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "avail_script", ROOT / "scripts" / "availability.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        sub = self.games.head(3000)
        a = mod.strict_roster_expectation(sub).reset_index(drop=True)
        b = mod.strict_roster_expectation(
            self.games.head(4500)).head(len(sub)).reset_index(drop=True)
        tol = 1e-9
        for c in ("home_sexp", "away_sexp", "sexp_diff"):
            x = a[c].to_numpy(float)
            y = b[c].to_numpy(float)
            # A NaN pattern that changes under truncation is a leak outright.
            self.assertTrue((np.isfinite(x) == np.isfinite(y)).all(),
                            f"{c}: finiteness pattern changed when future games "
                            f"were added, which is a leak")
            m = np.isfinite(x)
            self.assertLess(float(np.nanmax(np.abs(x[m] - y[m]))), tol,
                            f"{c} moved by more than {tol} when future games were added")

    def test_expected_minutes_never_uses_the_current_game(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "avail_script", ROOT / "scripts" / "availability.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        pm = mod.player_minutes()
        pm["expected_min"] = mod._ewm_by_player(pm, "MIN", 15.0)
        both = pm.dropna(subset=["expected_min"])
        # An expectation built from the current game would equal it far more often
        # than a trailing average should. 0.01% is the observed rate; a large jump
        # means the shift stopped working.
        same = float((np.abs(both["expected_min"] - both["MIN"]) < 1e-9).mean())
        self.assertLess(same, 0.01,
                        f"{same:.2%} of expectations equal the current game's "
                        "minutes, which means the current game leaked in")


class TestTheBoundedClaimStaysBounded(unittest.TestCase):
    """Read the published numbers and assert the claim has not drifted."""

    @classmethod
    def setUpClass(cls):
        if not RESULTS.exists():
            raise unittest.SkipTest("run scripts/availability.py first")
        cls.data = json.loads(RESULTS.read_text())

    def test_the_strict_roster_effect_excludes_zero(self):
        d = self.data["arms"]["strict_roster_expected"]
        self.assertGreater(d["delta"], 0.0,
                           "the strictly-past availability effect was positive")
        self.assertGreater(d["ci_lo"], 0.0,
                           "the whole interval must sit above zero for this to be "
                           "a finding rather than a null")

    def test_the_leak_control_does_not_exclude_zero(self):
        """The control must stay a null, or the strict arm is not trustworthy.

        If roster size alone ever becomes significant, the strict-roster number is
        measuring "how many people dressed" rather than player quality, and the
        bounded claim collapses.
        """
        d = self.data["arms"]["control_roster_size_only"]
        self.assertLess(d["ci_lo"], 0.0,
                        "the roster-size control became significant; re-examine "
                        "the strict-roster claim before quoting it")
        self.assertGreater(d["ci_hi"], 0.0)

    def test_the_honest_share_is_a_minority_of_the_leak(self):
        """35% is the honest share. It must never be reported as the whole.

        The leaky arm is three times more impressive and entirely unattainable.
        Asserting the ratio keeps the two from being conflated.
        """
        arms = self.data["arms"]
        honest = arms["strict_roster_expected"]["delta"]
        leak = arms["full_leak_actual_minutes"]["delta"]
        share = honest / leak
        self.assertGreater(share, 0.0)
        self.assertLess(share, 0.5,
                        "the strictly-past share is still a minority of the leak; "
                        "if this changed the writeup needs revisiting")

    def test_the_full_leak_is_still_much_larger(self):
        arms = self.data["arms"]
        self.assertGreater(arms["full_leak_actual_minutes"]["delta"],
                           2 * arms["strict_roster_expected"]["delta"],
                           "if the leak is no longer clearly larger, the framing "
                           "of 'most of this is unavailable' is stale")


class TestSignConvention(unittest.TestCase):
    def test_positive_delta_means_b_is_better(self):
        """Availability is now the largest measured effect, so this matters more.

        Reading its sign backwards would turn a real finding into a null.
        """
        y = np.array([0, 0, 1, 1] * 25)
        d = nd.paired_bootstrap_delta(y, np.full(100, 0.5),
                                      np.where(y == 1, 0.9, 0.1),
                                      n_boot=200, seed=0)
        self.assertGreater(d["delta"], 0.0)


if __name__ == "__main__":
    unittest.main()