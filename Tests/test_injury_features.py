"""Issue #37: identity, the pre-tip gate, and the weighted availability block.

The three things that can go silently wrong, each pinned here:

1. **The clock.** The report prints ``07:30 (ET)`` with no AM/PM. A naive
   24-hour parse puts every evening tip at 7:30 a.m., which makes every
   afternoon report "after tip" and empties the feature without raising.
2. **The gate.** A game at 19:30 must use the 17:00 report even when a 20:00
   report exists, and a game at 22:00 may use the 20:00 report. Getting this
   backwards is a leak that looks like a better feature.
3. **The discount.** Re-weighting must be a no-op by default (the pre-#37
   block is unchanged), linear in the weight, and scoped to the prior-game
   squad. A weight that leaks onto the wrong game or the wrong player is the
   failure mode the whole feature exists to avoid.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT, ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from DataProviders import injury_features as inf  # noqa: E402
from DataProviders import nba_data as nd  # noqa: E402


class TestClock(unittest.TestCase):

    def test_evening_times_are_pm_without_a_suffix(self):
        self.assertEqual(inf.parse_tip_minutes("07:30 (ET)"), 19 * 60 + 30)
        self.assertEqual(inf.parse_tip_minutes("10:00 (ET)"), 22 * 60)
        self.assertEqual(inf.parse_tip_minutes("03:30 (ET)"), 15 * 60 + 30)

    def test_noon_and_explicit_suffixes(self):
        self.assertEqual(inf.parse_tip_minutes("12:00 (ET)"), 12 * 60)
        self.assertEqual(inf.parse_tip_minutes("7:30 PM"), 19 * 60 + 30)
        self.assertEqual(inf.parse_tip_minutes("12:15 AM"), 15)

    def test_tbd_is_not_a_time(self):
        self.assertIsNone(inf.parse_tip_minutes("TBD"))
        self.assertIsNone(inf.parse_tip_minutes(None))


class TestNameKeys(unittest.TestCase):

    def test_last_first_and_first_last_are_the_same_key(self):
        self.assertEqual(inf.name_key("Carter-Williams, Michael"),
                         inf.name_key("Michael Carter-Williams"))
        self.assertEqual(inf.name_key("Brown Jr., Chaundee"),
                         inf.name_key("Chaundee Brown Jr."))

    def test_punctuation_and_accents_do_not_split_a_player(self):
        self.assertEqual(inf.name_key("Jokić, Nikola"),
                         inf.name_key("Nikola Jokic"))
        self.assertEqual(inf.name_key("O'Neale, Royce"),
                         inf.name_key("Royce ONeale"))

    def test_different_players_do_not_collide(self):
        self.assertNotEqual(inf.name_key("Curry, Stephen"),
                            inf.name_key("Curry, Seth"))


class TestSelectPreTip(unittest.TestCase):

    def setUp(self):
        self.games = pd.DataFrame({
            "GAME_ID": ["g1", "g2", "g3"],
            "GAME_DATE": pd.to_datetime(
                ["2021-11-01", "2021-11-01", "2021-11-02"]),
            "home": ["MIL", "LAL", "BOS"],
            "away": ["BKN", "GSW", "NYK"],
            "y": [1, 0, 1],
        })
        self.snaps = pd.DataFrame({
            "id": [1, 2],
            "report_date": pd.to_datetime(["2021-11-01", "2021-11-01"]),
            "report_time_label": ["05PM", "08PM"],
            "label_ts": pd.to_datetime(
                ["2021-11-01T17:00:00", "2021-11-01T20:00:00"]),
            "snapshot_ts_utc": ["2021-11-01T21:00:00+00:00",
                                "2021-11-02T00:00:00+00:00"],
        })
        rows = []
        for gid, matchup, tip, label, status in (
                ("g1", "BKN@MIL", "07:30 (ET)", "2021-11-01T17:00:00", "Out"),
                ("g1", "BKN@MIL", "07:30 (ET)", "2021-11-01T20:00:00", "Available"),
                ("g2", "GSW@LAL", "10:00 (ET)", "2021-11-01T20:00:00", "Questionable")):
            rows.append({
                "id": len(rows) + 1, "snapshot_id": 1 if "17:" in label else 2,
                "line_no": len(rows), "snapshot_ts_et": label,
                "game_date": pd.Timestamp("2021-11-01"), "game_time": tip,
                "matchup": matchup, "team": "Home Team", "player": "Doe, John",
                "status": status, "reason": "", "label_ts": pd.Timestamp(label),
                "PLAYER_ID": 1, "weight": inf.STATUS_WEIGHTS[status],
                "team_abbr": gid == "g1" and "MIL" or "LAL",
                "match_method": "exact", "ambiguous": False,
                "player_team": gid == "g1" and "MIL" or "LAL",
            })
        self.events = pd.DataFrame(rows)

    def test_an_early_game_cannot_use_a_report_published_after_its_tip(self):
        d, cov = inf.select_pre_tip(self.events, self.snaps, self.games)
        g1 = d[d["GAME_ID"] == "g1"].iloc[0]
        self.assertEqual(str(g1["status_label_ts"]), "2021-11-01 17:00:00")
        self.assertEqual(float(g1["weight"]), 0.0)

    def test_a_late_game_uses_the_later_report(self):
        d, cov = inf.select_pre_tip(self.events, self.snaps, self.games)
        g2 = d[d["GAME_ID"] == "g2"].iloc[0]
        self.assertEqual(str(g2["status_label_ts"]), "2021-11-01 20:00:00")
        self.assertEqual(float(g2["weight"]), 0.5)

    def test_no_snapshot_is_reported_as_uncovered_not_as_no_injury(self):
        d, cov = inf.select_pre_tip(self.events, self.snaps, self.games)
        row = cov[cov["GAME_ID"] == "g3"].iloc[0]
        self.assertFalse(bool(row["has_snapshot"]))
        self.assertTrue(pd.isna(row["snapshot_label"]))
        self.assertNotIn("g3", set(d["GAME_ID"]))

    def test_the_buffer_can_exclude_a_report_close_to_tip(self):
        d, cov = inf.select_pre_tip(self.events, self.snaps, self.games,
                                    buffer_minutes=400)
        row = cov[cov["GAME_ID"] == "g2"].iloc[0]
        self.assertFalse(bool(row["has_snapshot"]))


class TestWeightedAvailability(unittest.TestCase):
    """The block is unchanged by default and linear in a same-game weight."""

    @classmethod
    def setUpClass(cls):
        try:
            games = nd.build_games()
        except Exception as exc:
            raise unittest.SkipTest(f"NBA data unavailable: {exc}")
        cls.games = games.head(600).reset_index(drop=True)
        cls.base = nd.availability_features(cls.games)
        cls.pm = nd.player_minutes(cls.games)

    def _a_squad_player(self, game_id, team):
        sub = self.pm[self.pm["GAME_ID"] == game_id]
        sub = sub[sub["TEAM_ABBREVIATION"] == team]
        self.assertGreater(len(sub), 0)
        return int(sub["PLAYER_ID"].iloc[0])

    def test_default_is_exactly_the_pre_issue_block(self):
        again = nd.availability_features(self.games, weights=None)
        pd.testing.assert_frame_equal(self.base, again)

    def test_a_zero_weight_removes_that_players_expected_minutes(self):
        i = 300
        gid = self.games.loc[i, "GAME_ID"]
        team = self.games.loc[i, "home"]
        pid = self._a_squad_player(gid, team)
        w = pd.DataFrame({"GAME_ID": [gid], "PLAYER_ID": [pid], "weight": [0.0]})
        adj = nd.availability_features(self.games, weights=w)
        removed = (self.base.loc[i, "home_expected_minutes"]
                   - adj.loc[i, "home_expected_minutes"])
        self.assertGreater(removed, 0.0)
        self.assertLess(removed, 48.0)
        # The other side of the same game is untouched.
        self.assertAlmostEqual(self.base.loc[i, "away_expected_minutes"],
                               adj.loc[i, "away_expected_minutes"], places=9)

    def test_the_effect_is_linear_in_the_weight(self):
        i = 300
        gid = self.games.loc[i, "GAME_ID"]
        pid = self._a_squad_player(gid, self.games.loc[i, "home"])
        full = nd.availability_features(
            self.games,
            weights=pd.DataFrame({"GAME_ID": [gid], "PLAYER_ID": [pid],
                                  "weight": [0.0]}))
        half = nd.availability_features(
            self.games,
            weights=pd.DataFrame({"GAME_ID": [gid], "PLAYER_ID": [pid],
                                  "weight": [0.5]}))
        removed_full = (self.base.loc[i, "home_expected_minutes"]
                        - full.loc[i, "home_expected_minutes"])
        removed_half = (self.base.loc[i, "home_expected_minutes"]
                        - half.loc[i, "home_expected_minutes"])
        self.assertAlmostEqual(removed_full / 2.0, removed_half, places=9)

    def test_a_weight_for_a_player_not_in_the_game_is_a_no_op(self):
        w = pd.DataFrame({"GAME_ID": [self.games.loc[300, "GAME_ID"]],
                          "PLAYER_ID": [999999999], "weight": [0.0]})
        adj = nd.availability_features(self.games, weights=w)
        pd.testing.assert_frame_equal(self.base, adj)

    def test_a_duplicate_weight_is_refused(self):
        gid = self.games.loc[300, "GAME_ID"]
        pid = self._a_squad_player(gid, self.games.loc[300, "home"])
        w = pd.DataFrame({"GAME_ID": [gid, gid], "PLAYER_ID": [pid, pid],
                          "weight": [0.0, 0.5]})
        with self.assertRaises(ValueError):
            nd.availability_features(self.games, weights=w)

    def test_a_weight_above_one_is_refused(self):
        gid = self.games.loc[300, "GAME_ID"]
        pid = self._a_squad_player(gid, self.games.loc[300, "home"])
        w = pd.DataFrame({"GAME_ID": [gid], "PLAYER_ID": [pid], "weight": [1.5]})
        with self.assertRaises(ValueError):
            nd.availability_features(self.games, weights=w)


class TestIdentityResolution(unittest.TestCase):

    def _pa(self):
        return pd.DataFrame({
            "PLAYER_ID": [1, 2, 3],
            "PLAYER_NAME": ["Michael Carter-Williams", "Michael Carter-Williams",
                            "Svi Mykhailiuk"],
            "NICKNAME": ["Michael", "Michael", "Svi"],
            "SEASON": ["2021-22", "2021-22", "2021-22"],
            "TEAM_ABBREVIATION": ["ORL", "HOU", "TOR"],
            "TEAM_NAME": ["Orlando Magic", "Houston Rockets", "Toronto Raptors"],
            "GAME_DATE": pd.to_datetime(
                ["2021-11-01", "2021-11-01", "2021-11-01"]),
        })

    def _events(self, team):
        return pd.DataFrame({
            "id": [1],
            "game_date": pd.to_datetime(["2021-11-01"]),
            "game_time": ["07:30 (ET)"],
            "matchup": ["BOS@ORL"],
            "team": [team],
            "player": ["Carter-Williams, Michael"],
            "status": ["Out"],
        })

    def test_the_team_cell_breaks_a_same_name_tie(self):
        r = inf.resolve_players(self._events("Orlando Magic"), self._pa())
        self.assertEqual(int(r.loc[0, "PLAYER_ID"]), 1)
        self.assertEqual(r.loc[0, "match_method"], "exact")

    def test_different_team_names_the_other_player(self):
        r = inf.resolve_players(self._events("Houston Rockets"), self._pa())
        self.assertEqual(int(r.loc[0, "PLAYER_ID"]), 2)

    def test_a_nickname_bridges_the_report_and_the_roster(self):
        ev = self._events("Toronto Raptors")
        ev["player"] = "Mykhailiuk, Svi"
        r = inf.resolve_players(ev, self._pa())
        self.assertEqual(int(r.loc[0, "PLAYER_ID"]), 3)
        self.assertIn(r.loc[0, "match_method"], ("exact", "nickname"))

    def test_an_unknown_player_stays_unresolved(self):
        ev = self._events("Orlando Magic")
        ev["player"] = "Nobody, Real"
        r = inf.resolve_players(ev, self._pa())
        self.assertTrue(pd.isna(r.loc[0, "PLAYER_ID"]))
        self.assertEqual(r.loc[0, "match_method"], "unresolved")


if __name__ == "__main__":
    unittest.main()
