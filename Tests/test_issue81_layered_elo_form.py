"""Tests for issue #81: the layered Elo + recent-form filter.

The properties that carry weight here:

* ATS settlement is SIDE-AWARE. An away bet covers when the home margin is
  *below* the home-cover threshold; getting this backwards would silently
  invert every away arm, which is half the population.
* Form and history counts are strictly pre-game. A team's own result must not
  enter its own entering count, or the filter is looking at the game it bets.
* The Elo constants cannot drift from ``feat_team_elo``. The script duplicates
  the call rather than importing the feature builder, so a default change in
  one place has to fail here rather than split the two silently.
"""
import importlib.util
import inspect
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "issue81_layered_elo_form.py"
SPEC = importlib.util.spec_from_file_location("issue81", MODULE_PATH)
i81 = importlib.util.module_from_spec(SPEC)
sys.modules["issue81"] = i81
SPEC.loader.exec_module(i81)


def _row(n=10, roi=-0.05, se=0.02, z=-2.5, ci=(-0.09, -0.01)):
    return {"n_bets": n, "roi": roi, "se": se, "z": z, "ci": list(ci),
            "price": "ATS -110", "n_pushes": 0, "win_rate": 0.5,
            "break_even_win_rate": i81.ATS_BREAK_EVEN, "mean_line": 5.0}


def _payload():
    r = _row()
    return {
        "generated_utc": "2026-10-26T00:00:00+00:00", "protocol": i81.PROTOCOL,
        "population": {"priced_games": 12385, "elo_layer": 6735,
                       "form_layer": 4000, "cell": 1893,
                       "seasons": ["2015-16", "2025-26"]},
        "primary": r, "primary_ml": dict(r, price="ML", mean_price=0.8),
        "by_venue": {"home": r, "away": r},
        "elo_ladder": {"gap_ge100": {"elo_only_ats": r, "cell_ats": r, "cell_ml": r}},
        "form11_sensitivity": {"cell_ats": r, "cell_ml": r},
        "form_subcells": {"hot_cold": r, "neither": r},
        "per_season": {"2025-26": r},
        "reference": {"favourite_ats": r},
        "verdicts": {"primary_betting": "x", "primary_ml": "y", "ladder": "z"},
    }


class TestSettlement(unittest.TestCase):
    def test_away_side_is_settled_against_the_home_line(self):
        j = pd.DataFrame({"margin": [-6.0, -4.0, -5.0],
                          "L_eff": [-5.0, -5.0, -5.0],
                          "ML_Home": [-110.0] * 3, "ML_Away": [-110.0] * 3,
                          "y": [0, 0, 0]})
        high_home = np.array([False, False, False])
        p = i81.ats_profit(j, np.ones(3, dtype=bool), high_home)
        # -6 < -5: the away side covers. -4 > -5: it does not. -5 is a push.
        self.assertAlmostEqual(p[0], i81.ATS_PAYOUT)
        self.assertEqual(p[1], -1.0)
        self.assertEqual(p[2], 0.0)

    def test_home_side_is_the_complement(self):
        j = pd.DataFrame({"margin": [-6.0, -4.0, -5.0],
                          "L_eff": [-5.0, -5.0, -5.0],
                          "ML_Home": [-110.0] * 3, "ML_Away": [-110.0] * 3,
                          "y": [0, 1, 1]})
        high_home = np.array([True, True, True])
        p = i81.ats_profit(j, np.ones(3, dtype=bool), high_home)
        self.assertEqual(p[0], -1.0)
        self.assertAlmostEqual(p[1], i81.ATS_PAYOUT)
        self.assertEqual(p[2], 0.0)

    def test_moneyline_decimal_conversion(self):
        j = pd.DataFrame({"ML_Home": [-200.0, -200.0], "ML_Away": [150.0, 150.0],
                          "y": [1, 0]})
        home = i81.ml_profit(j, np.array([True, True]), np.array([True, True]))
        self.assertAlmostEqual(home[0], 0.5)     # -200 wins 0.5 per unit
        away = i81.ml_profit(j, np.array([True, True]), np.array([False, False]))
        self.assertAlmostEqual(away[1], 1.5)     # +150 wins 1.5 per unit
        self.assertEqual(home[1], -1.0)
        self.assertEqual(away[0], -1.0)


class TestLayers(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame({
            "elo_gap": [120.0, -120.0, 120.0, 40.0],
            "home_form10": [8, 8, 8, 8],
            "away_form10": [1, 1, 1, 1],
            "home_prior": [20, 20, 9, 20],
            "away_prior": [20, 20, 20, 20],
        })

    def test_the_higher_rated_side_is_the_one_filtered(self):
        m = i81.side_masks(self._frame(), 100.0, 10, 6, 3)
        # row 0: high side is home, form 8; row 1: high side is away, form 1.
        self.assertEqual(m["high_home"].tolist(), [True, False, True, True])
        self.assertEqual(m["cell"].tolist(), [True, False, False, False])

    def test_history_gate_blocks_short_samples(self):
        m = i81.side_masks(self._frame(), 100.0, 10, 6, 3)
        self.assertFalse(m["cell"][2])

    def test_elo_threshold_is_inclusive(self):
        j = pd.DataFrame({"elo_gap": [100.0, 99.99],
                          "home_form10": [8, 8], "away_form10": [1, 1],
                          "home_prior": [20, 20], "away_prior": [20, 20]})
        m = i81.side_masks(j, 100.0, 10, 6, 3)
        self.assertTrue(m["elo_layer"][0])
        self.assertFalse(m["elo_layer"][1])


class TestCausality(unittest.TestCase):
    def _games(self):
        rows = []
        for gid, (hp, ap) in enumerate([(110, 100), (105, 99), (98, 104)], start=1):
            rows.append({"GAME_ID": f"g{gid}", "GAME_DATE": f"2025-10-{gid:02d}",
                         "home": "A", "away": "B", "home_pts": hp, "away_pts": ap})
        return pd.DataFrame(rows)

    def test_form_entering_excludes_the_game_itself(self):
        f = i81.form_frame(self._games())
        # A wins games 1 and 2; entering counts are 0, 1, 2.
        self.assertEqual(f["home_form10"].tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(f["away_form10"].tolist(), [0.0, 0.0, 0.0])

    def test_prior_games_excludes_the_game_itself(self):
        p = i81.prior_games(self._games())
        self.assertEqual(p["home_prior"].tolist(), [0, 1, 2])
        self.assertEqual(p["away_prior"].tolist(), [0, 1, 2])


class TestConstants(unittest.TestCase):
    def test_elo_constants_match_the_feature_builder(self):
        from DataProviders.feat_team_elo import HFA, team_elo_features
        sig = inspect.signature(team_elo_features)
        self.assertEqual(i81.ELO_K, sig.parameters["k"].default)
        self.assertEqual(i81.HFA, sig.parameters["home_advantage"].default)
        self.assertEqual(i81.SEASON_REVERSION,
                         sig.parameters["season_reversion"].default)
        self.assertEqual(i81.HFA, HFA)


class TestStatistics(unittest.TestCase):
    def test_bootstrap_is_seeded(self):
        p = np.array([1.0, -1.0] * 100)
        a = i81.bootstrap(p, n_boot=300, seed=42)
        b = i81.bootstrap(p, n_boot=300, seed=42)
        self.assertEqual(a["roi"], b["roi"])
        self.assertEqual(a["se"], b["se"])
        self.assertAlmostEqual(a["roi"], 0.0)

    def test_verdicts_follow_the_registered_rule(self):
        neg = i81._verdicts({"roi": -0.05, "z": -2.0, "n_bets": 100},
                            {"roi": -0.04, "z": -1.0, "n_bets": 100}, {})
        self.assertIn("loses at the price", neg["primary_betting"])
        sub = i81._verdicts({"roi": 0.02, "z": 1.0, "n_bets": 100},
                            {"roi": 0.01, "z": 0.5, "n_bets": 100}, {})
        self.assertIn("not promoted", sub["primary_betting"])
        won = i81._verdicts({"roi": 0.07, "z": 3.5, "n_bets": 100},
                            {"roi": 0.05, "z": 2.0, "n_bets": 100}, {})
        self.assertIn("clears the gate", won["primary_betting"])


class TestArtifact(unittest.TestCase):
    def test_markdown_escapes_pipes_in_the_z_column(self):
        md = "\n".join(i81._markdown(_payload()))
        self.assertIn(r"\|z\|", md)
        self.assertNotIn("| |z| |", md)

    def test_markdown_reports_every_registered_block(self):
        md = "\n".join(i81._markdown(_payload()))
        for heading in ("Primary", "Moneyline", "Venue split", "Elo ladder",
                        "11-game form", "sub-cells", "Per season",
                        "Reference"):
            self.assertIn(heading, md)


if __name__ == "__main__":
    unittest.main()
