"""Contract tests for issue #72's late-scratch census.

The census itself is a descriptive artifact, but two pieces of selection logic
are easy to get wrong and were wrong on the first development run:

* at a tied report label, ``select_pre_tip`` takes the MOST severe
  designation (lowest weight), not the least severe;
* a later report replaces an earlier one, even when the later designation is
  less severe.

These tests pin both, plus the Wilson helper and the season label.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import issue72_late_scratch as i72  # noqa: E402
from DataProviders import injury_features as inf  # noqa: E402


def _panel(rows):
    df = pd.DataFrame(rows)
    df["label_ts"] = pd.to_datetime(df["label_ts"])
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["tip_ts"] = pd.to_datetime(df["tip_ts"])
    df["weight"] = df["status"].map(inf.STATUS_WEIGHTS)
    return df


class TestWilson(unittest.TestCase):

    def test_zero_successes_has_a_zero_lower_bound(self):
        lo, hi = i72.wilson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)
        self.assertLess(hi, 0.5)

    def test_half_contains_the_point_estimate(self):
        lo, hi = i72.wilson(5, 10)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)

    def test_empty_is_nan(self):
        lo, hi = i72.wilson(0, 0)
        self.assertTrue(pd.isna(lo))
        self.assertTrue(pd.isna(hi))


class TestStatusSelection(unittest.TestCase):

    def test_tied_label_takes_the_most_severe_status(self):
        panel = _panel([
            # Same player, same label, three designations.  Questionable is
            # the most severe; the first development run picked Available.
            {"GAME_ID": "g1", "PLAYER_ID": 1, "id": 1,
             "game_date": "2021-11-01", "tip_ts": "2021-11-01T22:00:00",
             "label_ts": "2021-11-01T17:00:00", "status": "Available"},
            {"GAME_ID": "g1", "PLAYER_ID": 1, "id": 2,
             "game_date": "2021-11-01", "tip_ts": "2021-11-01T22:00:00",
             "label_ts": "2021-11-01T17:00:00", "status": "Questionable"},
            {"GAME_ID": "g1", "PLAYER_ID": 1, "id": 3,
             "game_date": "2021-11-01", "tip_ts": "2021-11-01T22:00:00",
             "label_ts": "2021-11-01T17:00:00", "status": "Probable"},
        ])
        got = i72.status_at(panel, 1.0)
        self.assertEqual(len(got), 1)
        self.assertEqual(got.iloc[0]["status"], "Questionable")

    def test_a_later_report_wins_even_if_less_severe(self):
        panel = _panel([
            {"GAME_ID": "g1", "PLAYER_ID": 1, "id": 1,
             "game_date": "2021-11-01", "tip_ts": "2021-11-01T22:00:00",
             "label_ts": "2021-11-01T17:00:00", "status": "Questionable"},
            {"GAME_ID": "g1", "PLAYER_ID": 1, "id": 2,
             "game_date": "2021-11-01", "tip_ts": "2021-11-01T22:00:00",
             "label_ts": "2021-11-01T20:00:00", "status": "Available"},
        ])
        got = i72.status_at(panel, 1.0)
        self.assertEqual(got.iloc[0]["status"], "Available")

    def test_a_report_after_the_cutoff_is_excluded(self):
        panel = _panel([
            {"GAME_ID": "g1", "PLAYER_ID": 1, "id": 1,
             "game_date": "2021-11-01", "tip_ts": "2021-11-01T22:00:00",
             "label_ts": "2021-11-01T17:00:00", "status": "Questionable"},
            {"GAME_ID": "g1", "PLAYER_ID": 1, "id": 2,
             "game_date": "2021-11-01", "tip_ts": "2021-11-01T22:00:00",
             "label_ts": "2021-11-01T21:30:00", "status": "Out"},
        ])
        got = i72.status_at(panel, 1.0)  # cutoff 21:00
        self.assertEqual(got.iloc[0]["status"], "Questionable")


class TestSeasonLabel(unittest.TestCase):

    def test_july_starts_a_new_season_and_june_stays(self):
        dates = pd.to_datetime(["2026-06-30", "2026-07-01", "2026-10-01"])
        got = i72.season_of(pd.Series(dates)).tolist()
        self.assertEqual(got, [2025, 2026, 2026])

    def test_label_is_the_two_year_form(self):
        self.assertEqual(i72.season_label(2025), "2025-26")


if __name__ == "__main__":
    unittest.main()
