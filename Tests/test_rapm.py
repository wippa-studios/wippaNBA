"""Tests for `src/Models/rapm.py`.

The point of this module is that it fits on **stints**, not on a box-score
proxy, and that it predicts from the **expected** roster.  Those claims are
mechanical and are pinned with a hand-built game:

* the design entry for a stint is the overlap with each on-floor interval,
  divided by 28800, signed by side;
* a noiseless stint fit recovers the coefficients that generated it;
* changing actual minutes in the panel cannot move a prediction, while changing
  expected minutes must.
"""

import unittest

import numpy as np
import pandas as pd
from scipy import sparse

from src.Models import rapm


def toy_game() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One game with a substitution at the halfway point."""
    rows = []
    for pid in (10, 11, 12, 13):
        rows.append({"GAME_ID": 22000001, "TEAM_SIDE": "HOME",
                     "PERSON_ID": pid, "IN_TIME_REAL": 0.0,
                     "OUT_TIME_REAL": 28800.0, "PT_DIFF": 2.0})
    rows.append({"GAME_ID": 22000001, "TEAM_SIDE": "HOME",
                 "PERSON_ID": 14, "IN_TIME_REAL": 0.0,
                 "OUT_TIME_REAL": 14400.0, "PT_DIFF": 1.0})
    rows.append({"GAME_ID": 22000001, "TEAM_SIDE": "HOME",
                 "PERSON_ID": 15, "IN_TIME_REAL": 14400.0,
                 "OUT_TIME_REAL": 28800.0, "PT_DIFF": -1.0})
    for pid in (20, 21, 22, 23, 24):
        rows.append({"GAME_ID": 22000001, "TEAM_SIDE": "AWAY",
                     "PERSON_ID": pid, "IN_TIME_REAL": 0.0,
                     "OUT_TIME_REAL": 28800.0, "PT_DIFF": -2.0})
    rotation = pd.DataFrame(rows)
    games = pd.DataFrame({
        "GAME_ID": ["0022000001"], "SEASON": ["2020-21"],
        "home": ["H"], "away": ["A"], "y": [1], "margin": [5.0],
    })
    return rotation, games


class TestNormaliseStints(unittest.TestCase):
    def test_reversed_clock_columns_are_min_max_not_a_negative_stint(self):
        rot = pd.DataFrame({
            "GAME_ID": [1, 1], "PERSON_ID": [10, 11],
            "IN_TIME_REAL": [100.0, 50.0], "OUT_TIME_REAL": [50.0, 50.0],
            "PT_DIFF": [3.0, np.nan],
        })
        out = rapm._normalise_stints(rot)
        self.assertEqual(len(out), 1)
        self.assertEqual(float(out.iloc[0]["start"]), 50.0)
        self.assertEqual(float(out.iloc[0]["end"]), 100.0)

    def test_missing_plus_minus_is_dropped(self):
        rot = pd.DataFrame({
            "GAME_ID": [1], "PERSON_ID": [10],
            "IN_TIME_REAL": [0.0], "OUT_TIME_REAL": [100.0],
            "PT_DIFF": [np.nan],
        })
        self.assertTrue(rapm._normalise_stints(rot).empty)


class TestDesign(unittest.TestCase):
    def setUp(self):
        self.rotation, self.games = toy_game()
        self.design = rapm.build_design(self.rotation, self.games)
        self.pidx = {int(p): i for i, p in
                     enumerate(self.design["player_ids"])}

    def test_diagnostics_and_shape(self):
        d = self.design["diagnostics"]
        self.assertEqual(d["games_used"], 1)
        self.assertEqual(d["games_skipped_partial"], 0)
        self.assertEqual(d["intervals"], 2)
        self.assertEqual(d["stint_rows_used"], 11)
        # Eleven players (six home, five away) plus the home indicator.
        self.assertEqual(self.design["X"].shape, (11, 12))

    def test_home_indicator_and_signed_overlap_for_a_full_game_stint(self):
        # Find the stint row for H player 10: 0-28800, overlapping both halves.
        rows = np.flatnonzero(self.design["game_id"] == "0022000001")
        home_col = self.design["home_col"]
        player_ids = self.design["player_ids"]
        target = None
        for r in rows:
            row = self.design["X"].getrow(r).toarray().ravel()
            # The full-game home stints have home indicator 1.0 and away
            # players all -1.0; use player 10's identity to select one.
            if np.isclose(row[home_col], 1.0) and np.isclose(
                    row[self.pidx[10]], 1.0):
                target = row
                break
        self.assertIsNotNone(target)
        self.assertEqual(float(target[self.pidx[10]]), 1.0)
        # Teammates who played the whole game share the full overlap.
        for pid in (11, 12, 13):
            self.assertEqual(float(target[self.pidx[pid]]), 1.0)
        # The substituted players each played half a game.
        self.assertEqual(float(target[self.pidx[14]]), 0.5)
        self.assertEqual(float(target[self.pidx[15]]), 0.5)
        # Every away player is on the floor for the whole game, against.
        for pid in (20, 21, 22, 23, 24):
            self.assertEqual(float(target[self.pidx[pid]]), -1.0)
        self.assertEqual(float(target[home_col]), 1.0)

    def test_weights_are_stint_durations(self):
        w = self.design["w"]
        self.assertEqual(float(np.min(w)), 14400.0)
        self.assertEqual(float(np.max(w)), 28800.0)
        self.assertEqual(float(np.sum(w)), 4 * 28800 + 14400 + 14400
                         + 4 * 28800 + 28800)


class TestFit(unittest.TestCase):
    def test_noiseless_fit_recovers_generating_coefficients(self):
        true_theta = np.array([2.0, -1.0])
        true_home = 3.0
        X = sparse.csr_matrix(np.array([
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ]))
        y = X @ np.r_[true_theta, true_home]
        w = np.ones(3)
        model = rapm.fit_rapm(X, y, w, np.array([10, 11]), alpha=0.0)
        np.testing.assert_allclose(model.theta, true_theta)
        self.assertAlmostEqual(model.home_points_per_48, true_home)


class TestPredictionUsesExpectedRoster(unittest.TestCase):
    def setUp(self):
        # The toy design is rank-deficient for individual players, so build a
        # prediction model directly with known effects.
        self.model = rapm.RapmModel(
            player_ids=np.array([10, 11]), theta=np.array([2.0, -1.0]),
            home_points_per_48=3.0, alpha=0.0, config=rapm.RapmConfig())
        self.panel = pd.DataFrame({
            "GAME_ID": ["0022000001"] * 4,
            "team": ["H", "H", "A", "A"],
            "PLAYER_ID": [10, 11, 20, 21],
            "is_home": [1.0, 1.0, 0.0, 0.0],
            "MIN": [30.0, 30.0, 30.0, 30.0],
        })

    def test_margin_matches_hand_computation(self):
        em = _FakeEM({10: 24.0, 11: 24.0, 20: 24.0, 21: 24.0})
        margin, home, away = rapm.predict_margins(
            self.panel, em, self.model, ["0022000001"])
        # 24/240 * 2 + 24/240 * (-1) = 0.1; away players are unseen -> 0.
        self.assertAlmostEqual(float(home[0]), 0.1)
        self.assertAlmostEqual(float(away[0]), 0.0)
        self.assertAlmostEqual(float(margin[0]), 3.1)

    def test_actual_minutes_cannot_move_the_prediction(self):
        em = _FakeEM({10: 24.0, 11: 24.0, 20: 24.0, 21: 24.0})
        base, _, _ = rapm.predict_margins(
            self.panel, em, self.model, ["0022000001"])
        perturbed = self.panel.copy()
        perturbed["MIN"] = 0.0
        moved, _, _ = rapm.predict_margins(
            perturbed, em, self.model, ["0022000001"])
        self.assertAlmostEqual(float(base[0]), float(moved[0]))

    def test_expected_minutes_must_move_the_prediction(self):
        base_em = _FakeEM({10: 24.0, 11: 24.0, 20: 24.0, 21: 24.0})
        base, _, _ = rapm.predict_margins(
            self.panel, base_em, self.model, ["0022000001"])
        wide_em = _FakeEM({10: 48.0, 11: 48.0, 20: 24.0, 21: 24.0})
        moved, _, _ = rapm.predict_margins(
            self.panel, wide_em, self.model, ["0022000001"])
        self.assertNotAlmostEqual(float(base[0]), float(moved[0]))


class _FakeEM:
    def __init__(self, minutes):
        self.minutes = minutes

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return np.asarray([self.minutes.get(int(p), 0.0)
                           for p in rows["PLAYER_ID"]], dtype=float)


if __name__ == "__main__":
    unittest.main()
