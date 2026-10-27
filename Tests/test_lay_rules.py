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


if __name__ == "__main__":
    unittest.main()
