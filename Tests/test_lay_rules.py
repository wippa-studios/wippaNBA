"""Schema pins for the family-7 artifact (lay bets, #95)."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestLayRulesArtifact(unittest.TestCase):
    def test_schema_and_mirror_identity(self):
        r = json.loads((ROOT / "results" / "edge_search" / "lay_rules.json").read_text())
        self.assertEqual(r["family"], "lay_rules")
        self.assertEqual(r["n_arms"], 12)
        self.assertEqual(r["prereg"], "E07-prereg-lay_rules")
        self.assertTrue(r["mirror_identity"].startswith("lay ROI = -back ROI"))

        by = {a["label"].split(":")[0]: a for a in r["arms"]}
        # exact mirrors of the measured back arms
        f3 = json.loads((ROOT / "results" / "edge_search" / "bestprice_ml.json").read_text())
        fav080 = [a for a in f3["arms"] if "0.80-1.01" in a["label"]
                  and "favourite" in a["label"]][0]
        self.assertAlmostEqual(by["L7"]["discovery"]["roi"],
                               -fav080["discovery"]["roi"], places=12)
        f6 = json.loads((ROOT / "results" / "edge_search" / "user_rules_50.json").read_text())
        r42 = [a for a in f6["arms"] if a["label"].startswith("R42")][0]
        self.assertAlmostEqual(by["L11"]["discovery"]["roi"],
                               -r42["discovery"]["roi"], places=12)

        # the two formal candidates are the known mirrors
        cands = [a["label"].split(":")[0] for a in r["arms"]
                 if a["verdict"].startswith("candidate")]
        self.assertEqual(sorted(cands), ["L11", "L7"])

        # commission must reduce, never increase, the lay ROI
        for a in r["arms"]:
            d = a["discovery"]
            if "commission_5pct" in d:
                self.assertLessEqual(d["commission_5pct"]["roi"], d["roi"])

    def test_the_run_is_in_the_ledger_in_order(self):
        recs = [json.loads(line) for line in
                (ROOT / "results" / "experiments.jsonl").read_text().splitlines()]
        names = [r.get("name") for r in recs]
        for n in ("E07-prereg-lay_rules", "E07-run-lay_rules",
                  "E07-diagnostics-lay_rules"):
            self.assertIn(n, names)
        idx = [names.index(n) for n in ("E07-prereg-lay_rules", "E07-run-lay_rules",
                                        "E07-diagnostics-lay_rules")]
        self.assertEqual(idx, sorted(idx))


class TestLayGridArtifact(unittest.TestCase):
    def test_schema_and_headline_numbers(self):
        r = json.loads((ROOT / "results" / "edge_search" / "lay_grid.json").read_text())
        self.assertEqual(r["family"], "lay_grid")
        self.assertEqual(r["n_arms"], 24)
        self.assertEqual(r["prereg"], "E08-prereg-lay_grid")
        self.assertEqual(r["population"]["discovery"], 11902)
        self.assertEqual(r["population"]["holdout"], 1204)
        # no candidate: the rule requires discovery ROI > 0 AND |z| > bar
        # AND holdout ROI > 0
        cleared = [a for a in r["arms"] if a["discovery"]["roi"] > 0
                   and abs(a["discovery"]["z"]) > r["bar_z"]
                   and a["holdout"]["roi"] > 0]
        self.assertEqual(cleared, [])
        # the one |z|>3 result is the significant negative on long lays:
        # L22 fails the discovery-ROI > 0 leg of the candidate rule
        neg = [a for a in r["arms"] if a["discovery"]["z"] < -r["bar_z"]]
        self.assertEqual(len(neg), 1)
        self.assertTrue(neg[0]["label"].startswith("L22"))
        self.assertLess(neg[0]["discovery"]["roi"], 0)
        # commission never helps a lay
        for a in r["arms"]:
            d = a["discovery"]
            if "commission_5pct" in d:
                self.assertLessEqual(d["commission_5pct"]["roi"], d["roi"])
        # liability scales with price: L22's liability >> L7-style chalk lays
        by = {a["label"].split()[0]: a for a in r["arms"]}
        self.assertGreater(by["L22"]["discovery"]["mean_liability"], 4.0)

    def test_the_run_is_in_the_ledger_in_order(self):
        recs = [json.loads(line) for line in
                (ROOT / "results" / "experiments.jsonl").read_text().splitlines()]
        names = [r.get("name") for r in recs]
        for n in ("E08-prereg-lay_grid", "E08-run-lay_grid",
                  "E08-diagnostics-lay_grid"):
            self.assertIn(n, names)
        idx = [names.index(n) for n in ("E08-prereg-lay_grid", "E08-run-lay_grid",
                                        "E08-diagnostics-lay_grid")]
        self.assertEqual(idx, sorted(idx))


if __name__ == "__main__":
    unittest.main()
