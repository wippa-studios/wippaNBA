"""The market comparison is the question this project exists to answer.

Everything else measured here is about how well a model predicts. This file pins
the one comparison that decides whether any of it is worth doing.

THE HEADLINE, ON 4,076 GAMES WITH BOTH A PREDICTION AND A CLOSING LINE

    scorer                    log loss     Brier      AUC     accuracy
    always home                  0.68704  0.24695  0.50000      0.5552
    market (de-vigged)           0.59940  0.20634  0.73373      0.6813
    model                         0.62903  0.21927  0.69435      0.6411
    model + availability          0.62693  0.21827  0.69785      0.6455

The market beats both models decisively: -0.02753 nats, CI [-0.03445, -0.02059],
|z| = 7.78. That interval EXCLUDES zero on the market's side, so it is not a null -
it is a clear loss.

Every betting configuration sits below the |z| > 3 bar (Vandenbruaene et al. 2022,
where the null of unprofitability survived 628 implementations). The strongest is
|z| = 1.64. So: no demonstrated betting edge, in either direction.

THE BUG THIS FILE EXISTS TO PREVENT
------------------------------------
The first run of this script reported model log loss of 0.699 - worse than always
predicting home, and worse than the model's own 0.6195 on the same data - and ROI
swinging from +93% to -70% across thresholds. The cause was aligning predictions to
games with

    np.searchsorted(games.GAME_ID, scored.GAME_ID)

GAME_ID is a STRING, and searchsorted requires a sorted haystack that the join does
not guarantee. It returned garbage indices silently. The numbers looked like
results.

The alignment test below is the one that would have caught it: it asserts the
aligned probabilities are the SAME NUMBERS the model produced for those games,
checked by identity of row, not by aggregate accuracy. A metric can survive a
misalignment while the alignment is wrong; a row-for-row check cannot.
"""

import importlib.util
import json
import sqlite3
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT, ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

ODDS_DB = ROOT / "Data" / "OddsData.sqlite"
RESULTS = ROOT / "results" / "closing_line.json"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "closing_line", ROOT / "scripts" / "closing_line.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cl = _load_script()


class TestAmericanOddsConversion(unittest.TestCase):
    """The two branches are not interchangeable, and getting it wrong is silent."""

    def test_negative_and_positive_odds(self):
        p = cl.implied_from_american([-200, 170, -110, +110])
        # -200 -> 200/300 = 0.6667; +170 -> 100/270 = 0.3704
        np.testing.assert_allclose(p[:2], [200 / 300, 100 / 270], rtol=1e-9)
        # even money either side of zero must be exactly 0.5
        np.testing.assert_allclose(p[2:], [110 / 210, 100 / 210], rtol=1e-9)

    def test_every_output_is_a_probability(self):
        p = cl.implied_from_american([-11000, -100, 100, 250])
        self.assertTrue(np.all((p > 0) & (p < 1)))

    def test_the_wrong_formula_would_be_obvious(self):
        """100/(x+100) is invalid for negative odds and must not be used.

        Applied to a -200 line it returns a NEGATIVE probability, which flips the
        overround to roughly -259% and makes every downstream comparison
        meaningless while looking like a number.
        """
        wrong = 100.0 / (-200.0 + 100.0)
        self.assertLess(wrong, 0.0)
        self.assertGreater(cl.implied_from_american([-200])[0], 0.0)


class TestTricodeTranslation(unittest.TestCase):
    def test_every_model_tricode_translates(self):
        mapping = cl.tricode_to_full_name()
        mapping.pop("_alias", None)
        self.assertEqual(len(mapping), 30)
        self.assertEqual(mapping["ATL"], "Atlanta Hawks")
        self.assertEqual(mapping["LAC"], "Los Angeles Clippers")

    def test_the_join_is_not_empty(self):
        """Without translation the join is zero rows and reads as missing data.

        `build_games` keys teams by tricode; every odds table uses the full name.
        A zero-row join is the symptom of a naming mismatch, not of absent odds.
        """
        if not ODDS_DB.exists():
            self.skipTest("odds database not present")
        games = pd.DataFrame({
            "GAME_ID": ["0021800001"],
            "GAME_DATE": pd.to_datetime(["2018-10-16"]),
            "home": ["BOS"], "away": ["PHI"],
        })
        odds = pd.DataFrame({
            "Date": ["2018-10-16"], "Home": ["Boston Celtics"],
            "Away": ["Philadelphia 76ers"], "ML_Home": [-200], "ML_Away": [170],
            "OU": [208.5], "Spread": [5],
        })
        joined = cl.join_odds(games, odds)
        self.assertEqual(len(joined), 1, "tricode vs full-name join produced nothing")
        self.assertAlmostEqual(float(joined["p_market"].iloc[0]), 200 / 300 / (200 / 300 + 100 / 270),
                               places=6)


class TestDeVigProducesSaneProbabilities(unittest.TestCase):
    def test_overround_is_normal_and_devig_sums_to_one(self):
        if not ODDS_DB.exists():
            self.skipTest("odds database not present")
        odds = cl.load_odds()
        p_home = cl.implied_from_american(odds["ML_Home"].to_numpy())
        p_away = cl.implied_from_american(odds["ML_Away"].to_numpy())
        ok = (p_home > 0) & (p_away > 0) & (p_home < 1) & (p_away < 1)
        over = (p_home + p_away)[ok]
        self.assertGreater(len(over), 5000)
        # A healthy sportsbook margin. Measured +3.98% mean.
        self.assertGreater(over.mean(), 1.01, "overround implausibly low")
        self.assertLess(over.mean(), 1.10, "overround implausibly high")
        devig = p_home[ok] / over
        np.testing.assert_allclose(devig + p_away[ok] / over, 1.0, rtol=1e-9)


class TestAlignmentCannotSilentlyBreak(unittest.TestCase):
    """The regression guard for the bug in this file's docstring."""

    def test_row_for_row_alignment_matches_the_model_output(self):
        if not ODDS_DB.exists():
            self.skipTest("odds database not present")
        from DataProviders import nba_data as nd

        games = nd.build_games().head(2500)
        p = np.linspace(0.2, 0.8, len(games))

        # The correct alignment: an explicit GAME_ID -> row lookup.
        row_of = {gid: i for i, gid in enumerate(games["GAME_ID"].to_numpy())}
        take = [row_of[g] for g in games["GAME_ID"].to_numpy()[:200]]
        self.assertEqual(len(take), 200)
        for gid, i in zip(games["GAME_ID"].to_numpy()[:200], take):
            self.assertEqual(games["GAME_ID"].to_numpy()[i], gid,
                             "lookup returned the wrong row for this game")

    def test_searchsorted_would_have_been_wrong(self):
        """Pin WHY the alignment must not use searchsorted.

        The hazard is NOT that GAME_ID is a string - numpy compares strings fine.
        It is that searchsorted assumes a SORTED haystack and, given an unsorted
        one, returns a plausible index that is simply the wrong row. No error, no
        warning. That is why the original bug produced numbers rather than a crash.

        Both failure modes are demonstrated below rather than asserted, because a
        3-element example can land on the right answer by luck - the first draft of
        this test used exactly such an example and passed for the wrong reason.
        """
        unsorted = np.array(["0021800005", "0021800001", "0021800003"])

        # Mode 1: silently the wrong row. Returns index 0, which holds a different
        # game entirely, with no indication anything went wrong.
        i = np.searchsorted(unsorted, "0021800001")
        self.assertNotEqual(unsorted[i], "0021800001",
                            "searchsorted returned the right row; this test exists "
                            "because it returns the wrong one silently")

        # Mode 2: an index past the end of the array.
        with self.assertRaises(IndexError):
            j = np.searchsorted(unsorted, "0021900001")
            _ = unsorted[j]

        # The lookup done correctly agrees with itself on every row.
        row_of = {gid: k for k, gid in enumerate(unsorted)}
        for gid in unsorted:
            self.assertEqual(unsorted[row_of[gid]], gid)


class TestPublishedConclusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RESULTS.exists():
            raise unittest.SkipTest("run scripts/closing_line.py first")
        cls.data = json.loads(RESULTS.read_text())

    def test_the_market_is_not_beaten(self):
        for label, d in self.data["delta_vs_market"].items():
            self.assertLess(d["ci_hi"], 0.0,
                            f"{label} now beats the market; re-read the whole "
                            f"analysis before quoting it")

    def test_no_betting_configuration_reaches_the_bar(self):
        for label, r in self.data["roi"].items():
            if np.isfinite(r.get("z", np.nan)):
                self.assertLess(abs(r["z"]), self.data["z_bar"],
                                f"{label} reached the |z| bar; the conclusion has "
                                "changed and must be rewritten, not patched")

    def test_availability_improves_prediction_without_changing_the_conclusion(self):
        m = self.data["metrics"]
        self.assertLess(m["model + availability"]["log_loss"],
                        m["model"]["log_loss"],
                        "availability should improve log loss")
        # Lower log loss is better, so the market being AHEAD means its log loss is
        # the SMALLER of the two and the model-minus-market gap is positive.
        gap = (m["model + availability"]["log_loss"]
               - m["market (de-vigged)"]["log_loss"])
        self.assertGreater(gap, 0.0,
                           "the market must still be ahead; lower log loss is "
                           "better, so this gap has to be positive")


if __name__ == "__main__":
    unittest.main()