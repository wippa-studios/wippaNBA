"""Schema pins for the family-5 artifact (13 user-supplied rules, #91)."""
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class TestUserRulesArtifact(unittest.TestCase):
    def test_schema_and_headline_numbers(self):
        r = json.loads((ROOT / "results" / "edge_search" / "user_rules.json").read_text())
        self.assertEqual(r["family"], "user_rules")
        self.assertEqual(r["n_arms"], 13)
        self.assertEqual(r["prereg"], "E05-prereg-user_rules")
        self.assertEqual(r["bar_z"], 3.0)
        self.assertEqual(len(r["arms"]), 13)
        # the run happened under the registered split
        self.assertEqual(r["population"]["discovery"], 11902)
        self.assertEqual(r["population"]["holdout"], 1204)
        # no candidate: no arm cleared the bar with both splits positive
        cleared = [a for a in r["arms"]
                   if abs(a["discovery"]["z"]) > r["bar_z"] and a["holdout"]["roi"] > 0]
        self.assertEqual(cleared, [])
        # the docs' headline: R8 fails the holdout, R12/R13 are same-sign
        by_label = {a["label"].split()[0]: a for a in r["arms"]}
        self.assertGreater(by_label["R8"]["discovery"]["roi"], 0.05)
        self.assertLess(by_label["R8"]["holdout"]["roi"], 0)
        for key in ("R12", "R13"):
            self.assertGreater(by_label[key]["discovery"]["roi"], 0)
            self.assertGreater(by_label[key]["holdout"]["roi"], 0)
            self.assertEqual(by_label[key]["verdict"], "same sign, underpowered")

    def test_the_run_is_in_the_ledger(self):
        recs = [json.loads(line) for line in
                (ROOT / "results" / "experiments.jsonl").read_text().splitlines()]
        names = {r.get("name") for r in recs}
        self.assertIn("E05-prereg-user_rules", names)
        self.assertIn("E05-run-user_rules", names)
        # the pre-registration must precede the run
        order = [i for i, r in enumerate(recs)
                 if r.get("name") in ("E05-prereg-user_rules", "E05-run-user_rules")]
        self.assertEqual(sorted(order), order)


class TestUserRules50Artifact(unittest.TestCase):
    def test_schema_and_consistency_with_family_5(self):
        r50 = json.loads((ROOT / "results" / "edge_search" / "user_rules_50.json").read_text())
        self.assertEqual(r50["family"], "user_rules_50")
        self.assertEqual(r50["n_arms"], 50)
        r13 = json.loads((ROOT / "results" / "edge_search" / "user_rules.json").read_text())
        by13 = {a["label"]: (a["discovery"], a["holdout"]) for a in r13["arms"]}
        for a in r50["arms"]:
            if a["label"].split()[0] in {f"R{i}" for i in range(1, 14)}:
                d13, h13 = by13[a["label"]]
                self.assertEqual(a["discovery"], d13, a["label"])
                self.assertEqual(a["holdout"], h13, a["label"])
        # no candidate among the 50
        cleared = [a for a in r50["arms"]
                   if abs(a["discovery"]["z"]) > r50["bar_z"] and a["holdout"]["roi"] > 0]
        self.assertEqual(cleared, [])
        # the one |z|>3 result is the significant negative on heavy chalk
        neg = [a for a in r50["arms"] if a["discovery"]["z"] < -r50["bar_z"]]
        self.assertEqual(len(neg), 1)
        self.assertTrue(neg[0]["label"].startswith("R42"))
        # the ledger carries prereg -> run -> diagnostics in order
        recs = [json.loads(line) for line in
                (ROOT / "results" / "experiments.jsonl").read_text().splitlines()]
        names = [r.get("name") for r in recs]
        for n in ("E06-prereg-user_rules_50", "E06-run-user_rules_50",
                  "E06-diagnostics-user_rules_50"):
            self.assertIn(n, names)
        idx = [names.index(n) for n in ("E06-prereg-user_rules_50",
                                        "E06-run-user_rules_50",
                                        "E06-diagnostics-user_rules_50")]
        self.assertEqual(idx, sorted(idx))


if __name__ == "__main__":
    unittest.main()
