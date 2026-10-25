"""Tests for `src/Models/hierarchical_ratings.py`.

The model is a random-effects regression, so the properties worth pinning are
mechanical rather than numerical:

* **Shrinkage is real.** Raising alpha must reduce the spread of the player
  effects and pull team predictions toward the league mean.
* **Expected minutes, not actual minutes.** The design reads expected minutes
  from the fitted ``ExpectedMinutes`` object and ignores the ``MIN`` column that
  happens to sit in the panel.  Perturbing ``MIN`` must leave the design
  bit-identical.
* **Unseen players are the prior mean.** A player absent from the training
  index contributes zero, not an error and not an arbitrary number.
"""

import unittest

import numpy as np
import pandas as pd

from src.Models import hierarchical_ratings as hr


class FakeEM:
    """A deterministic expected-minutes model keyed by player id."""

    def __init__(self, minutes: dict[int, float]):
        self.minutes = minutes
        self.seen_frames: list[pd.DataFrame] = []

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        self.seen_frames.append(rows.copy())
        return np.asarray([self.minutes.get(int(p), 0.0)
                           for p in rows["PLAYER_ID"]], dtype=float)


def toy_rows() -> tuple[pd.DataFrame, pd.DataFrame, FakeEM, dict[int, float]]:
    games = [("G1", "A", "B"), ("G2", "C", "D")]
    players = {team: list(range(base, base + 5))
               for team, base in (("A", 100), ("B", 200), ("C", 300), ("D", 400))}
    rows = []
    for gid, home, away in games:
        for team, is_home in ((home, 1.0), (away, 0.0)):
            for pid in players[team]:
                rows.append({"GAME_ID": gid, "team": team, "PLAYER_ID": pid,
                             "is_home": is_home, "MIN": 24.0 + pid % 7})
    tgt = pd.DataFrame({
        "GAME_ID": ["G1", "G1", "G2", "G2"],
        "team": ["A", "B", "C", "D"],
        "net_per_100": [5.0, -5.0, 2.0, -2.0],
    })
    minutes = {pid: 30.0 for team in players.values() for pid in team}
    return pd.DataFrame(rows), tgt, FakeEM(minutes), minutes


class TestDesign(unittest.TestCase):
    def test_expected_minutes_are_used_and_actual_minutes_are_not(self):
        rows, tgt, em, _ = toy_rows()
        keys, A, pids, codes, weights = hr._design(rows, em)
        self.assertEqual(A.shape[0], 4)          # two games x two teams
        self.assertEqual(A.shape[1], 20)         # twenty players
        np.testing.assert_allclose(np.asarray(A.sum(axis=1)).ravel(),
                                   [5 * 30 / 240] * 4)
        perturbed = rows.copy()
        perturbed["MIN"] = 0.0
        _, A2, _, _, _ = hr._design(perturbed, em)
        np.testing.assert_array_equal(A.toarray(), A2.toarray())

    def test_player_codes_are_contiguous_and_ordered(self):
        rows, _, em, _ = toy_rows()
        _, _, pids, codes, _ = hr._design(rows, em)
        self.assertTrue((np.diff(pids) > 0).all())
        self.assertEqual(codes.max(), len(pids) - 1)


class TestShrinkage(unittest.TestCase):
    def test_larger_alpha_shrinks_theta_and_predictions(self):
        rows, tgt, em, _ = toy_rows()
        small = hr.fit_player_skills(rows, tgt, em, alpha=0.1)
        large = hr.fit_player_skills(rows, tgt, em, alpha=1000.0)
        self.assertLess(np.std(large.theta), np.std(small.theta))
        m_small, _, _ = hr.predict_team_net(rows, ["G1", "G2"], small)
        m_large, _, _ = hr.predict_team_net(rows, ["G1", "G2"], large)
        self.assertLess(np.std(m_large), np.std(m_small))

    def test_small_alpha_recovers_the_team_targets(self):
        rows, tgt, em, _ = toy_rows()
        # alpha must be tiny relative to the design's own X'X scale here:
        # 0.1 is already enough shrinkage to halve a four-team fit.
        model = hr.fit_player_skills(rows, tgt, em, alpha=1e-6)
        margin, home, away = hr.predict_team_net(rows, ["G1"], model)
        # Target difference is 10 points per 100; home court adds 2.2.
        self.assertAlmostEqual(float(margin[0]), 12.2, delta=0.6)
        self.assertGreater(float(home[0]), float(away[0]))


class TestPrediction(unittest.TestCase):
    def test_output_follows_the_requested_game_order(self):
        rows, tgt, em, _ = toy_rows()
        model = hr.fit_player_skills(rows, tgt, em, alpha=1.0)
        m_ab, _, _ = hr.predict_team_net(rows, ["G1", "G2"], model)
        m_ba, _, _ = hr.predict_team_net(rows, ["G2", "G1"], model)
        self.assertAlmostEqual(float(m_ab[0]), float(m_ba[1]))
        self.assertAlmostEqual(float(m_ab[1]), float(m_ba[0]))

    def test_unseen_player_contributes_zero(self):
        rows, tgt, em, minutes = toy_rows()
        model = hr.fit_player_skills(rows, tgt, em, alpha=1.0)
        base, _, _ = hr.predict_team_net(rows, ["G1"], model)
        extra = rows.copy()
        extra = pd.concat([extra, pd.DataFrame([{
            "GAME_ID": "G1", "team": "A", "PLAYER_ID": 999,
            "is_home": 1.0, "MIN": 0.0}])], ignore_index=True)
        changed, _, _ = hr.predict_team_net(extra, ["G1"], model)
        self.assertAlmostEqual(float(base[0]), float(changed[0]))

    def test_expected_minutes_change_the_prediction(self):
        rows, tgt, em, minutes = toy_rows()
        model = hr.fit_player_skills(rows, tgt, em, alpha=1.0)
        base, _, _ = hr.predict_team_net(rows, ["G1"], model)
        shifted = dict(minutes)
        for pid in (100, 101, 102, 103, 104):
            shifted[pid] = 48.0
        model.em = FakeEM(shifted)
        changed, _, _ = hr.predict_team_net(rows, ["G1"], model)
        self.assertNotAlmostEqual(float(base[0]), float(changed[0]))


class TestLabels(unittest.TestCase):
    def test_player_ratings_are_sorted(self):
        rows, tgt, em, _ = toy_rows()
        model = hr.fit_player_skills(rows, tgt, em, alpha=1.0)
        s = model.player_ratings()
        self.assertTrue((np.diff(s.to_numpy()) <= 0).all())
        self.assertEqual(len(s), 20)


if __name__ == "__main__":
    unittest.main()
