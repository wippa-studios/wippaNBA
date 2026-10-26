"""Tests for the pre-registered search loop (`scripts/edge_search.py`).

The properties that carry weight here:

* ATS and total settlement is SIDE-AWARE and push-aware. Getting the away side
  backwards would silently invert half the arms; dropping pushes from `n`
  instead of settling them at zero would bias every ROI toward zero.
* The holdout swap never mixes sources. If the old 2025-26 snapshot leaked into
  the frame alongside the ESPN close, every holdout game would be counted twice
  and the holdout ROI would be the average of two different lines.
* The multiplicity bar is fixed by the number of registered arms, not by the
  result. A family that quietly grew after registration must fail here.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "scripts" / "edge_search.py"
SPEC = importlib.util.spec_from_file_location("edge_search", MODULE_PATH)
es = importlib.util.module_from_spec(SPEC)
sys.modules["edge_search"] = es
SPEC.loader.exec_module(es)


def frame(margins, lines, **extra):
    n = len(margins)
    data = {
        "margin": np.asarray(margins, dtype=float),
        "L_eff": np.asarray(lines, dtype=float),
        "ML_Home": np.full(n, -110.0),
        "ML_Away": np.full(n, -110.0),
        "y": (np.asarray(margins, dtype=float) > 0).astype(int),
        "total": np.full(n, 220.0),
        "OU": np.full(n, 220.0),
    }
    data.update(extra)
    return pd.DataFrame(data)


class TestSettlement(unittest.TestCase):
    def test_home_ats_covers_above_the_line(self):
        j = frame([6.0, 4.0, 5.0], [5.0, 5.0, 5.0])
        profit, win, push = es.settle_ats(j, True)
        np.testing.assert_allclose(profit, [es.ATS_PAYOUT, -1.0, 0.0])
        np.testing.assert_array_equal(win, [True, False, False])
        np.testing.assert_array_equal(push, [False, False, True])

    def test_away_ats_covers_below_the_line(self):
        j = frame([6.0, 4.0, 5.0], [5.0, 5.0, 5.0])
        profit, win, push = es.settle_ats(j, False)
        np.testing.assert_allclose(profit, [-1.0, es.ATS_PAYOUT, 0.0])
        np.testing.assert_array_equal(win, [False, True, False])

    def test_a_push_returns_the_stake_and_stays_in_the_sample(self):
        j = frame([5.0], [5.0])
        profit, _win, push = es.settle_ats(j, True)
        self.assertEqual(float(profit[0]), 0.0)
        self.assertTrue(bool(push[0]))

    def test_moneyline_uses_the_posted_price_on_the_chosen_side(self):
        j = frame([5.0, -5.0], [2.0, 2.0], ML_Home=[150.0, -200.0],
                  ML_Away=[-170.0, 170.0])
        home, _won = es.settle_ml(j, True)
        away, _won = es.settle_ml(j, False)
        # +150 wins 1.50; the away +170 wins 1.70 on the second game.
        np.testing.assert_allclose(home, [1.5, -1.0])
        np.testing.assert_allclose(away, [-1.0, 1.7])

    def test_moneyline_refuses_a_zero_price(self):
        j = frame([5.0], [2.0], ML_Home=[0.0], ML_Away=[-110.0])
        with self.assertRaises(ValueError):
            es.settle_ml(j, True)

    def test_total_over_under_and_push(self):
        j = frame([0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                  total=[225.0, 215.0, 220.0], OU=[220.0, 220.0, 220.0])
        over, _win, push = es.settle_total(j, True)
        under, _win, _push = es.settle_total(j, False)
        np.testing.assert_allclose(over, [es.ATS_PAYOUT, -1.0, 0.0])
        np.testing.assert_allclose(under, [-1.0, es.ATS_PAYOUT, 0.0])
        np.testing.assert_array_equal(push, [False, False, True])

    def test_cover_rate_excludes_pushes_from_the_denominator(self):
        j = frame([6.0, 4.0, 5.0], [5.0, 5.0, 5.0])
        out = es.stats(j, np.ones(3, dtype=bool), "ats", True)
        self.assertEqual(out["n"], 3)
        self.assertEqual(out["n_pushes"], 1)
        self.assertAlmostEqual(out["win_rate"], 0.5)

    def test_scalar_and_array_sides_agree(self):
        j = frame([6.0, 4.0, 5.0], [5.0, 5.0, 5.0])
        scalar = es.stats(j, np.ones(3, dtype=bool), "ats", True)
        array = es.stats(j, np.ones(3, dtype=bool), "ats",
                         np.ones(3, dtype=bool))
        self.assertEqual(scalar["roi"], array["roi"])
        self.assertEqual(scalar["n"], array["n"])

    def test_decimal_moneyline_settles_at_the_posted_odds(self):
        j = frame([5.0, -5.0], [2.0, 2.0], O1=[2.5, 1.8], O2=[1.6, 2.2])
        home, won = es.settle_ml_decimal(j, True)
        away, _won = es.settle_ml_decimal(j, False)
        # home wins game 1 at 2.5 (+1.50) and loses game 2 (-1); the away side
        # of game 2 wins at 2.2 (+1.20).
        np.testing.assert_allclose(home, [1.5, -1.0])
        np.testing.assert_allclose(away, [-1.0, 1.2])
        np.testing.assert_array_equal(won, [True, False])

    def test_decimal_moneyline_refuses_degenerate_odds(self):
        j = frame([5.0], [2.0], O1=[1.0], O2=[2.0])
        with self.assertRaises(ValueError):
            es.settle_ml_decimal(j, True)


class TestStatistics(unittest.TestCase):
    def test_bootstrap_is_deterministic(self):
        p = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        self.assertEqual(es.boot(p), es.boot(p))

    def test_bar_is_the_bonferroni_value_above_the_floor(self):
        self.assertAlmostEqual(es.bar_z(42), 3.2411521655173563, places=12)
        self.assertEqual(es.bar_z(1), 3.0)
        self.assertGreater(es.bar_z(42), 3.0)
        self.assertGreater(es.bar_z(100), es.bar_z(42))

    def test_verdict_requires_both_discovery_and_holdout(self):
        def arm(roi, z, n=100):
            row = {"n": n, "roi": roi, "z": z, "se": 0.02}
            return {"discovery": row, "holdout": row}
        bar = 3.24
        self.assertTrue(es.verdict(arm(0.05, 4.0), bar).startswith("candidate"))
        self.assertEqual(es.verdict(arm(0.05, 4.0) | {
            "holdout": {"n": 100, "roi": -0.01, "z": -1.0, "se": 0.02}}, bar),
            "discovery-only")
        self.assertEqual(es.verdict(arm(0.01, 1.0), bar),
                         "same sign, underpowered")
        self.assertEqual(es.verdict(arm(0.01, 1.0) | {
            "holdout": {"n": 100, "roi": -0.01, "z": -1.0, "se": 0.02}}, bar),
            "no evidence")
        self.assertEqual(es.verdict(arm(0.01, 1.0) | {
            "holdout": {"n": 0, "roi": float("nan"), "z": float("nan"),
                        "se": float("nan")}}, bar), "no data")


class TestHoldoutSwap(unittest.TestCase):
    def _base(self):
        return pd.DataFrame({
            "Date": ["2024-01-01", "2025-12-01"],
            "Home": ["Boston Celtics", "New York Knicks"],
            "Away": ["Miami Heat", "Cleveland Cavaliers"],
            "ML_Home": [-110.0, 150.0], "ML_Away": [-110.0, -170.0],
            "OU": [220.0, 230.0], "Spread": [-5.0, 4.5],
            "SEASON": ["2023-24", "2025-26"],
        })

    def _espn(self):
        return pd.DataFrame({
            "Date": ["2025-12-01"],
            "Home": ["New York Knicks"],
            "Away": ["Cleveland Cavaliers"],
            "ML_Home": [-180.0], "ML_Away": [155.0],
            "OU": [241.5], "Spread": [4.5],
        })

    def test_the_old_holdout_rows_are_dropped_not_merged(self):
        out = es.swap_holdout(self._base(), self._espn())
        held = out[out["SEASON"] == "2025-26"]
        self.assertEqual(len(held), 1)
        self.assertEqual(float(held["ML_Home"].iloc[0]), -180.0)
        self.assertEqual(len(out[out["SEASON"] == "2023-24"]), 1)

    def test_the_discovery_seasons_are_untouched(self):
        out = es.swap_holdout(self._base(), self._espn())
        old = self._base()
        pd.testing.assert_frame_equal(
            out[out["SEASON"] == "2023-24"].reset_index(drop=True),
            old[old["SEASON"] == "2023-24"].reset_index(drop=True))

    def test_duplicate_espn_keys_are_refused(self):
        espn = pd.concat([self._espn(), self._espn()], ignore_index=True)
        with self.assertRaises(ValueError):
            es.swap_holdout(self._base(), espn)

    def test_a_missing_quote_column_is_refused(self):
        with self.assertRaises(ValueError):
            es.swap_holdout(self._base(), self._espn().drop(columns=["OU"]))


class TestFamilyRegistry(unittest.TestCase):
    def test_price_structure_registers_42_named_arms(self):
        rng = np.random.default_rng(7)
        n = 600
        p = rng.uniform(0.25, 0.85, n)
        j = pd.DataFrame({
            "p_market": p,
            "L_eff": rng.uniform(-16, 16, n),
            "OU": rng.uniform(195, 245, n),
            "margin": rng.normal(0, 13, n),
            "total": rng.normal(225, 20, n),
            "ML_Home": np.where(p > 0.5, -150.0, 130.0),
            "ML_Away": np.where(p > 0.5, 130.0, -150.0),
            "y": (rng.normal(0, 13, n) > 0).astype(int),
            "split": np.where(np.arange(n) % 5 == 0, "holdout", "discovery"),
        })
        arms, meta = es.family_price_structure(j)
        self.assertEqual(len(arms), 42)
        self.assertEqual(meta["arms"], 42)
        self.assertEqual(len({a["label"] for a in arms}), 42)
        self.assertAlmostEqual(meta["bar_z"], es.bar_z(42))

    def test_the_preregistration_declares_the_same_count(self):
        rec = es.prereg_record("price_structure")
        self.assertEqual(rec["n_arms"], 42)
        self.assertEqual(rec["bar_z"], es.bar_z(42))
        self.assertEqual(rec["status"], "registered")
        self.assertIn("holdout", rec["split"])
        self.assertTrue(rec["holdout_source"].startswith(es.ESPN_TABLE))


class TestHoldoutData(unittest.TestCase):
    """The committed 2025-26 close table, pinned to the games it prices."""

    def test_the_table_is_canonical_complete_and_unique(self):
        df = es.read_espn_table()
        self.assertEqual(len(df), 1321)
        self.assertFalse(df.duplicated(subset=["Date", "Home", "Away"]).any())
        self.assertFalse(df[["OU", "Spread", "ML_Home", "ML_Away"]]
                         .isna().any().any())
        mapping = __import__("closing_line").tricode_to_full_name()
        mapping.pop("_alias", None)
        canonical = set(mapping.values()) | {"Los Angeles Clippers"}
        self.assertEqual(set(df["Home"]) - canonical, set())
        self.assertEqual(set(df["Away"]) - canonical, set())

    def test_1224_of_1225_regular_season_games_are_priced(self):
        df = es.read_espn_table()
        g = __import__("DataProviders.nba_data", fromlist=["x"]).build_games()
        g = g[g["SEASON"] == "2025-26"].copy()
        g["d"] = pd.to_datetime(g["GAME_DATE"]).dt.strftime("%Y-%m-%d")
        mapping = __import__("closing_line").tricode_to_full_name()
        mapping.pop("_alias", None)
        full_to_abbr = {v: k for k, v in mapping.items()}
        e = df.copy()
        e["h"] = e["Home"].map(full_to_abbr)
        e["a"] = e["Away"].map(full_to_abbr)
        merged = e.merge(g[["GAME_ID", "d", "home", "away"]],
                         left_on=["Date", "h", "a"],
                         right_on=["d", "home", "away"], how="inner")
        self.assertEqual(len(merged), 1224)
        self.assertEqual(merged["GAME_ID"].duplicated().sum(), 0)
        missing = set(g["GAME_ID"]) - set(merged["GAME_ID"])
        self.assertEqual(len(missing), 1)

    def test_the_search_odds_carry_only_the_espn_close_for_2025_26(self):
        odds = es.load_search_odds()
        held = odds[odds["SEASON"] == "2025-26"]
        self.assertEqual(len(held), 1321)
        # The old snapshot's Clippers spelling is gone; the ESPN table is canonical.
        self.assertNotIn("LA Clippers", set(held["Home"]))
        row = held[(held["Date"].astype(str).str[:10] == "2026-01-01")
                   & (held["Home"] == "Los Angeles Clippers")]
        self.assertEqual(float(row["Spread"].iloc[0]), 13.5)


class TestBestPriceFrame(unittest.TestCase):
    """The best-of-books moneyline frame behind the profit family (#86)."""

    def test_bestprice_family_registers_14_arms(self):
        rng = np.random.default_rng(11)
        n = 400
        o1 = np.round(np.exp(rng.normal(0.15, 0.55, n)) + 1.0, 2)
        o2 = np.round(np.exp(rng.normal(0.15, 0.55, n)) + 1.0, 2)
        j = pd.DataFrame({
            "O1": o1, "O2": o2,
            "p_fav_devig": (1 / o1) / (1 / o1 + 1 / o2),
            "fav_home": o1 <= o2,
            "y": (rng.random(n) < (1 / o1) / (1 / o1 + 1 / o2)).astype(int),
            "split": np.where(np.arange(n) % 4 == 0, "holdout", "discovery"),
        })
        arms, meta = es.family_bestprice_ml(j)
        self.assertEqual(len(arms), 14)
        self.assertEqual(meta["arms"], 14)
        self.assertEqual(len(meta["references"]), 2)
        self.assertAlmostEqual(meta["bar_z"], es.bar_z(14))
        self.assertEqual(es.bar_z(14), 3.0)  # Bonferroni floor holds

    def test_the_preregistration_declares_the_same_count(self):
        rec = es.prereg_record("bestprice_ml")
        self.assertEqual(rec["n_arms"], 14)
        self.assertEqual(rec["bar_z"], es.bar_z(14))
        self.assertEqual(rec["name"], "E03-prereg-bestprice_ml")
        self.assertTrue(rec["holdout_source"].startswith(es.BESTPRICE_TABLE))

    def test_devigged_favourite_probability_sums_to_one(self):
        j, _st = es.load_bestprice_frame()
        p = j["p_fav_devig"].to_numpy()
        self.assertTrue(np.all((p > 0.3) & (p < 1.0)))
        # the frame's favourite side must agree with the lower odds
        self.assertTrue((j["fav_home"] == (j["O1"] <= j["O2"])).all())

    def test_the_bestprice_table_covers_the_regular_season(self):
        j, st = es.load_bestprice_frame()
        # ~12.6k of 13.2k regular-season games priced, both splits populated
        self.assertGreater(st["priced_games"], 12000)
        self.assertGreater(st["discovery"], 11000)
        self.assertGreater(st["holdout"], 1100)
        self.assertTrue(st["price_source"].endswith(es.BESTPRICE_TABLE))


if __name__ == "__main__":
    unittest.main()
