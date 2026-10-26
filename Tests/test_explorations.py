"""Schema pins for the committed exploratory artifacts.

These are measurement records, not load-bearing code, so the tests only pin
that the committed artifacts keep the fields the docs quote — rebuilding them
requires the live crawl sources and is intentionally not part of the suite.
"""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestStreak4Artifact(unittest.TestCase):
    def test_schema_and_headline_numbers(self):
        r = json.loads((ROOT / "results" / "exploratory" / "streak4_profile.json").read_text())
        self.assertTrue(r["exploratory"])
        self.assertEqual(r["n"], 1596)
        self.assertAlmostEqual(r["overall_win_rate"], 0.5827, places=3)
        self.assertEqual(round(r["by_venue"]["1"]["win_rate"], 3), 0.665)
        self.assertEqual(round(r["by_venue"]["0"]["win_rate"], 3), 0.5)
        self.assertGreater(len(r["calibration"]), 3)
        self.assertGreater(len(r["feature_profile"]), 10)
        # the docs' headline: opponent Elo is the top non-market separator
        top = [f["feature"] for f in r["feature_profile"][:3]]
        self.assertIn("elo_opp", top)
        self.assertIn("devig", top)
        # and the streak's own construction carries nothing
        by_name = {f["feature"]: f["d"] for f in r["feature_profile"]}
        self.assertLess(abs(by_name["streak_opp_elo"]), 0.05)


class TestEloParityArtifact(unittest.TestCase):
    def test_schema_and_headline_numbers(self):
        r = json.loads((ROOT / "results" / "exploratory" / "elo_parity_underdog.json").read_text())
        self.assertTrue(r["exploratory"])
        ats = r["ats_single_book_close"]["|gap| < 100"]["underdog"]
        self.assertLess(ats["discovery"]["roi"], -0.05)
        self.assertLess(ats["discovery"]["z"], -5)
        ml = r["ml_best_of_books"]["market_underdog"]["|gap| < 100"]
        self.assertGreater(ml["discovery"]["roi"], 0)
        self.assertLess(ml["discovery_coherent"]["roi"], ml["discovery"]["roi"])
        dis = r["ats_single_book_close"]["elo_disagrees_with_market"]
        self.assertGreater(dis["favourite_both_splits"]["roi"],
                           dis["underdog_both_splits"]["roi"])


if __name__ == "__main__":
    unittest.main()
