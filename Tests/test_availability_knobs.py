"""Contract tests for issue #71's availability-block knobs.

The grid changes two things the recorded block never exposed: WHICH roster is
summed (the previous game's squad vs the season's cumulative union) and HOW
MANY players are summed (all vs the top N expected minutes). These tests pin:

* the default is bit-identical to the recorded construction;
* the season union only ever adds players from EARLIER games of the SAME
  season (and is NaN on a team's first game of a season);
* top_n keeps the N largest expected minutes, not the first N rows;
* truncating the future does not change either variant.
"""

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from src.DataProviders import nba_data as nd


# G0 exists so every player has one prior appearance and a materialised
# expectation from G1 onward (a player's expectation is NaN on their debut).
GAMES = [
    ("GM1", "2025-09-26", "T1", "T2"),
    ("G0", "2025-09-28", "T1", "T2"),
    ("G1", "2025-10-01", "T1", "T2"),
    ("G2", "2025-10-03", "T1", "T2"),
    ("G3", "2025-10-05", "T1", "T2"),
    ("G4", "2026-07-01", "T1", "T2"),   # a new season
]

MINUTES = {
    #        GM1   G0    G1    G2    G3    G4
    "A": (30.0, 30.0, 30.0, 30.0, 30.0, 30.0),
    "B": (20.0, 20.0, 20.0, 20.0, 20.0, 20.0),
    "C": (40.0, 40.0, 40.0, None, None, None),   # sits G2/G3
    "D": (25.0, 25.0, 25.0, 25.0, 25.0, 25.0),
    "E": (15.0, 15.0, 15.0, 15.0, 15.0, 15.0),
    "F": (5.0, 5.0, None, None, 5.0, 5.0),        # sits G1/G2
}
TEAM = {"A": "T1", "B": "T1", "C": "T1", "D": "T2", "E": "T2", "F": "T2"}
PLAYER_ID = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
ROSTERS = {
    "GM1": ("A", "B", "C", "D", "E", "F"),
    "G0": ("A", "B", "C", "D", "E", "F"),
    "G1": ("A", "B", "C", "D", "E"),
    "G2": ("A", "B", "D", "E"),
    "G3": ("A", "B", "D", "E", "F"),
    "G4": ("A", "D"),
}


def _games() -> pd.DataFrame:
    g = pd.DataFrame(GAMES, columns=["GAME_ID", "GAME_DATE", "home", "away"])
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"])
    return g


def _player_minutes(_games_arg=None) -> pd.DataFrame:
    rows = []
    order = [x[0] for x in GAMES]
    for gid, date, _, _ in GAMES:
        for name in ROSTERS[gid]:
            mins = MINUTES[name][order.index(gid)]
            if mins is None:
                continue
            rows.append({"GAME_ID": gid, "PLAYER_ID": PLAYER_ID[name],
                         "TEAM_ABBREVIATION": TEAM[name], "MIN": mins,
                         "GAME_DATE": pd.to_datetime(date)})
    return pd.DataFrame(rows)


def _features(**kwargs) -> pd.DataFrame:
    with mock.patch.object(nd, "player_minutes", side_effect=_player_minutes):
        return nd.availability_features(_games(), **kwargs)


class TestAvailabilityKnobs(unittest.TestCase):
    def test_default_is_the_recorded_construction(self):
        a = _features()
        b = _features(squad_mode="prior_game", top_n=None)
        pd.testing.assert_frame_equal(a, b)

    def test_prior_game_roster_is_the_previous_game(self):
        out = _features()
        # G1: T1's prior roster is G0 {A,B,C}.
        self.assertAlmostEqual(out.loc[2, "home_expected_minutes"], 90.0)
        # G2: T1's prior roster is G1 {A,B,C} -- C still expected to play.
        self.assertAlmostEqual(out.loc[3, "home_expected_minutes"], 90.0)
        # G3: T1's prior roster is G2 {A,B}, so C drops out.
        self.assertAlmostEqual(out.loc[4, "home_expected_minutes"], 50.0)
        # A team's first game has no prior roster.
        self.assertTrue(np.isnan(out.loc[0, "home_expected_minutes"]))

    def test_season_average_adds_earlier_season_appearances(self):
        out = _features(squad_mode="season_average")
        # G3: the union of GM1/G0/G1/G2 is still {A,B,C}, so C stays in
        # (90 vs the prior-game arm's 50).
        self.assertAlmostEqual(out.loc[4, "home_expected_minutes"], 90.0)
        # G4 starts a new season: no earlier games in it, so no roster.
        self.assertTrue(np.isnan(out.loc[5, "home_expected_minutes"]))
        self.assertTrue(np.isnan(out.loc[5, "away_expected_minutes"]))

    def test_top_n_keeps_the_largest_expectations(self):
        out = _features(top_n=2)
        # G1 roster {A30,B20,C40} -> top two are C and A.
        self.assertAlmostEqual(out.loc[2, "home_expected_minutes"], 70.0)
        # G2 roster {A30,B20,C40} -> same top two; G3 {A30,B20} is unchanged.
        self.assertAlmostEqual(out.loc[3, "home_expected_minutes"], 70.0)
        self.assertAlmostEqual(out.loc[4, "home_expected_minutes"], 50.0)
        # A top_n larger than the roster is the same as no cap.
        big = _features(top_n=50)
        base = _features()
        pd.testing.assert_frame_equal(big, base)

    def test_season_union_is_truncation_invariant(self):
        k = 5
        g = _games()
        short_games = g.iloc[:k].reset_index(drop=True)
        with mock.patch.object(nd, "player_minutes", side_effect=_player_minutes):
            short = nd.availability_features(short_games,
                                             squad_mode="season_average")
            full = nd.availability_features(g, squad_mode="season_average")
        # Adding a FUTURE game (G4) must not move G0-G3.  The NaN pattern is
        # compared exactly; finite cells to float tolerance.
        self.assertTrue((np.isfinite(short.to_numpy(float))
                         == np.isfinite(full.iloc[:k].to_numpy(float))).all())
        np.testing.assert_allclose(
            short.to_numpy(float), full.iloc[:k].to_numpy(float),
            rtol=0, atol=1e-9, equal_nan=True)

    def test_bad_knobs_are_rejected(self):
        with self.assertRaises(ValueError):
            _features(squad_mode="actual_lineup")
        with self.assertRaises(ValueError):
            _features(top_n=0)


if __name__ == "__main__":
    unittest.main()
