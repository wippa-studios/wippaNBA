"""Contract tests for issue #69's early-designation helper.

Pins the things that would make the trajectory arm meaningless if broken:
the EARLY label is the FIRST same-day pre-tip slot (not the latest), a
single-slot day supplies no early label (so no trajectory), and the helper
agrees with the authoritative ``select_pre_tip`` on the FINAL column where
both exist.
"""

import unittest
from pathlib import Path

import pandas as pd

from src.DataProviders import injury_features as inf


def _mini_store():
    """Two snapshots (17:00, 20:00, ET wall-clock, tz-naive) for one date."""
    reports = [
        ("2025-11-01", "2025-11-01 17:00", "Injury-Report_2025-11-01_05PM.pdf"),
        ("2025-11-01", "2025-11-01 20:00", "Injury-Report_2025-11-01_08PM.pdf"),
    ]
    snaps = pd.DataFrame(reports, columns=["report_date", "label_ts",
                                           "report_time_label"])
    # label_ts is the ET WALL-CLOCK as stored (tz-naive), exactly like the
    # live store: the report label is compared against ET tip minutes.
    snaps["label_ts"] = pd.to_datetime(snaps["label_ts"])
    snaps["report_date"] = pd.to_datetime(snaps["report_date"]).dt.normalize()
    rows = [
        # snapshot 1 (17:00): P1 Probable, P2 Questionable
        (101, 1, "2025-11-01 17:00", "2025-11-01", "8:30 PM", "GSW@CHI",
         "Golden State", "Stephen", "stephen", "Probable", None),
        (102, 1, "2025-11-01 17:00", "2025-11-01", "8:30 PM", "GSW@CHI",
         "Golden State", "Kevin", "kevin", "Questionable", None),
        # snapshot 2 (20:00): P1 Out (downgraded), P2 Available (upgraded)
        (103, 2, "2025-11-01 20:00", "2025-11-01", "8:30 PM", "GSW@CHI",
         "Golden State", "Stephen", "stephen", "Out", None),
        (104, 2, "2025-11-01 20:00", "2025-11-01", "8:30 PM", "GSW@CHI",
         "Golden State", "Kevin", "kevin", "Available", None),
    ]
    events = pd.DataFrame(rows, columns=["id", "snapshot_id", "ts",
                                         "game_date", "game_time", "matchup",
                                         "team", "player", "player_key",
                                         "status", "reason"])
    events["label_ts"] = pd.to_datetime(events["ts"])   # ET labels, naive
    events["snapshot_ts_utc"] = events["label_ts"]
    events["game_date"] = pd.to_datetime(events["game_date"]).dt.normalize()
    return snaps, events


class TestEarlyDesignation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        snaps, raw = _mini_store()
        cls.snaps = snaps
        cls.resolved = inf.resolve_players(raw)
        # The picker under test does not depend on name resolution; give the
        # two fixture players stable IDs and abbreviations (what the player-aux
        # table produces in production), then drop the NAs.
        id_map = {"Stephen": 1, "Kevin": 2}
        cls.resolved["PLAYER_ID"] = cls.resolved["player"].map(id_map)
        cls.resolved["team_abbr"] = "GSW"
        cls.resolved = cls.resolved.dropna(subset=["PLAYER_ID"])
        cls.resolved["PLAYER_ID"] = cls.resolved["PLAYER_ID"].astype("int64")
        cls.games = pd.DataFrame({
            "GAME_ID": ["G1"],
            "GAME_DATE": pd.to_datetime(["2025-11-01"]),
            "home": ["CHI"],
            "away": ["GSW"],
            "team_home": ["Golden State"],
        })
        # tip at 20:30 ET -> a 20:00 slot is 30 min pre-tip, 17:00 is earlier
        cls.games["tip_minutes"] = 20 * 60 + 30
        cls.games["game_date"] = pd.to_datetime("2025-11-01")
        cls.games["matchup"] = "GSW@CHI"

    def test_earliest_picks_the_first_same_day_slot(self):
        d = inf.earliest_pre_tip_designation(self.resolved, self.snaps,
                                             self.games, buffer_minutes=15)
        by = d.set_index("PLAYER_ID")["weight_early"].to_dict()
        self.assertAlmostEqual(by[1], 0.75)   # Stephen at 17:00: Probable
        self.assertAlmostEqual(by[2], 0.5)    # Kevin at 17:00: Questionable

    def test_final_picker_disagrees_where_trajectory_moved(self):
        d = inf.earliest_pre_tip_designation(self.resolved, self.snaps,
                                             self.games, buffer_minutes=15)
        fin = inf.select_pre_tip(self.resolved, self.snaps, self.games,
                                 buffer_minutes=15)[0]
        fin = fin.set_index("PLAYER_ID")["weight"]
        early = d.set_index("PLAYER_ID")["weight_early"]
        # delta = final - early: P1 -0.75 (Probable->Out), P2 +0.5 (Q->A).
        self.assertAlmostEqual(fin[1] - early[1], -0.75)
        self.assertAlmostEqual(fin[2] - early[2], 0.5)

    def test_no_usable_pre_tip_slot_has_no_early_label(self):
        # A day whose only snapshot is AFTER the 15-minute gate (20:30+ ET)
        # supplies no early label: nothing is usable.
        reports = [
            ("2025-11-01", "2025-11-01 21:30", "Injury-Report_2025-11-01_09_30PM.pdf"),
        ]
        snaps = pd.DataFrame(reports, columns=["report_date", "label_ts",
                                               "report_time_label"])
        snaps["label_ts"] = pd.to_datetime(snaps["label_ts"])
        snaps["report_date"] = pd.to_datetime(snaps["report_date"]).dt.normalize()
        d = inf.earliest_pre_tip_designation(self.resolved, snaps,
                                             self.games, buffer_minutes=15)
        self.assertEqual(len(d), 0)


if __name__ == "__main__":
    unittest.main()