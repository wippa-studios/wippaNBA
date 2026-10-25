"""Contract tests for the issue #56 time-zone / travel features.

Four things are pinned, from the strongest claim down:

* TRUNCATION INVARIANCE of both columns: deleting every game after game t
  leaves the row for t unchanged (the schedule leak class).  Inherited from
  ``rest_and_travel``'s same-frame ``shift`` construction, asserted explicitly
  here so the two new columns cannot regress to a lookahead.
* THE SEMANTICS on synthetic schedules where the true geometry is known: the
  ``|offset|`` equation for a hand-picked matchup, the season-opener origin,
  the same-zone consecutive-road-game case, and the cross-season boundary.
* THE TABLE: ``TEAM_TZ_OFFSET`` is complete over REAL_TEAMS and pins the
  boundary cities where a longitude projection would lie (OKC/SAS Central,
  not Mountain).
* THE RECIPE: ``full_tz`` is the canonical 26 columns PLUS exactly these two,
  and the canonical 26 is unchanged.
"""

import unittest

import numpy as np
import pandas as pd

from src.DataProviders import nba_data as nd
from src.DataProviders.nba_data import TEAM_TZ_OFFSET, REAL_TEAMS
from src.Models import logreg_rolling as lr


def mini_games(rows):
    """A tiny games frame with just the columns rest_and_travel reads."""
    return pd.DataFrame(rows, columns=["GAME_ID", "GAME_DATE", "SEASON",
                                       "home", "away"])


class TestTzTable(unittest.TestCase):
    def test_table_covers_every_real_team(self):
        self.assertEqual(set(TEAM_TZ_OFFSET), set(REAL_TEAMS),
                         "TEAM_TZ_OFFSET must name every REAL_TEAMS club")

    def test_boundary_cities_are_their_true_zones_not_longitude(self):
        # The whole point of the explicit table: these would be Mountain if
        # derived from longitude.
        self.assertEqual(TEAM_TZ_OFFSET["OKC"], -6)   # Central
        self.assertEqual(TEAM_TZ_OFFSET["SAS"], -6)   # Central
        self.assertEqual(TEAM_TZ_OFFSET["DAL"], -6)   # Central
        # Pacific / Mountain / Eastern spot checks.
        self.assertEqual(TEAM_TZ_OFFSET["GSW"], -8)
        self.assertEqual(TEAM_TZ_OFFSET["LAL"], -8)
        self.assertEqual(TEAM_TZ_OFFSET["POR"], -8)
        self.assertEqual(TEAM_TZ_OFFSET["DEN"], -7)
        self.assertEqual(TEAM_TZ_OFFSET["UTA"], -7)
        self.assertEqual(TEAM_TZ_OFFSET["PHX"], -7)
        self.assertEqual(TEAM_TZ_OFFSET["BOS"], -5)
        self.assertEqual(TEAM_TZ_OFFSET["CHA"], -5)


class TestTzSemantics(unittest.TestCase):
    def test_offset_diff_is_absolute_home_minus_away(self):
        g = mini_games([
            (1, "2015-11-01", "2015-16", "CHA", "GSW"),   # ET vs PT -> 3
            (2, "2015-11-02", "2015-16", "DEN", "HOU"),   # MT vs CT -> 1
            (3, "2015-11-03", "2015-16", "BOS", "NYK"),   # ET vs ET -> 0
        ])
        g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"])
        rt = nd.rest_and_travel(g).reset_index(drop=True)
        self.assertEqual(rt.loc[0, "tz_offset_diff"], 3.0)
        self.assertEqual(rt.loc[1, "tz_offset_diff"], 1.0)
        self.assertEqual(rt.loc[2, "tz_offset_diff"], 0.0)

    def test_away_crossed_season_opener_uses_home_city(self):
        # CHA's first game of the season is away at GSW: origin is its home
        # city (ET), venue is PT -> 3 zones.
        g = mini_games([
            (1, "2015-11-01", "2015-16", "GSW", "CHA"),
        ])
        g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"])
        rt = nd.rest_and_travel(g).reset_index(drop=True)
        self.assertEqual(rt.loc[0, "away_tz_crossed"], 3.0)

    def test_away_crossed_same_zone_consecutive_road_games_is_zero(self):
        # CHA opens at GSW (PT, 3 crossed), then plays a SECOND road game at
        # SAC (also PT): origin is the PREVIOUS GAME'S arena, so 0 crossed.
        g = mini_games([
            (1, "2015-11-01", "2015-16", "GSW", "CHA"),
            (2, "2015-11-05", "2015-16", "SAC", "CHA"),
        ])
        g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"])
        rt = nd.rest_and_travel(g).reset_index(drop=True)
        self.assertEqual(rt.loc[1, "away_tz_crossed"], 0.0)

    def test_away_crossed_crosses_zones_between_consecutive_venues(self):
        # CHA at DEN (MT) then at GSW (PT): origin is DEN's mountain arena, so
        # the second game crosses 1 zone, not 3.
        g = mini_games([
            (1, "2015-11-01", "2015-16", "DEN", "CHA"),
            (2, "2015-11-05", "2015-16", "GSW", "CHA"),
        ])
        g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"])
        rt = nd.rest_and_travel(g).reset_index(drop=True)
        self.assertEqual(rt.loc[1, "away_tz_crossed"], 1.0)

    def test_cross_season_previous_game_is_not_travel(self):
        # BOS ends 2015-16 away at GSW; its first 2016-17 game is away at MEM.
        # The offseason gap is NOT travel: origin is BOS's home city (ET), so
        # MEM (CT) is 1 zone, not BOS-to-GSW-to-MEM.
        g = mini_games([
            (1, "2016-05-01", "2015-16", "GSW", "BOS"),
            (2, "2016-10-20", "2016-17", "MEM", "BOS"),
        ])
        g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"])
        rt = nd.rest_and_travel(g).reset_index(drop=True)
        self.assertEqual(rt.loc[1, "away_tz_crossed"], 1.0)

    def test_columns_are_never_nan(self):
        g = nd.build_games().reset_index(drop=True)
        rt = nd.rest_and_travel(g).reset_index(drop=True)
        self.assertEqual(int(rt["tz_offset_diff"].isna().sum()), 0)
        self.assertEqual(int(rt["away_tz_crossed"].isna().sum()), 0)

    def test_both_columns_are_truncation_invariant(self):
        g = nd.build_games().reset_index(drop=True)
        n = 3000  # well inside the first fold
        full = nd.rest_and_travel(g).reset_index(drop=True)
        cut = nd.rest_and_travel(g.iloc[:n].reset_index(drop=True)
                                 ).reset_index(drop=True)
        pd.testing.assert_series_equal(
            full["tz_offset_diff"].iloc[:n].reset_index(drop=True),
            cut["tz_offset_diff"], check_exact=True)
        pd.testing.assert_series_equal(
            full["away_tz_crossed"].iloc[:n].reset_index(drop=True),
            cut["away_tz_crossed"], check_exact=True)


class TestTzRecipe(unittest.TestCase):
    def test_full_tz_is_canonical_plus_exactly_two(self):
        full = lr.resolve_features(lr.DEFAULT_MODELS["full"])
        tz = lr.resolve_features(lr.DEFAULT_MODELS["full_tz"])
        self.assertEqual(len(full), 26)
        self.assertEqual(len(tz), 28)
        self.assertEqual(tz[:26], full)
        self.assertEqual(tz[26:], ["tz_offset_diff", "away_tz_crossed"])

    def test_canonical_recipe_is_unchanged(self):
        full = lr.resolve_features(lr.DEFAULT_MODELS["full"])
        self.assertNotIn("tz_offset_diff", full)
        self.assertNotIn("away_tz_crossed", full)

    def test_design_has_the_two_columns(self):
        g = nd.build_games().reset_index(drop=True)
        X = lr.load_design(games=g)
        self.assertIn("tz_offset_diff", X.columns)
        self.assertIn("away_tz_crossed", X.columns)


if __name__ == "__main__":
    unittest.main()