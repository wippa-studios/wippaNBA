"""Tests for the dataset manifest and the loud-failure build paths (issues #3, #12)."""
import importlib.util
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CREATE_GAMES_PATH = REPO / "src" / "Process-Data" / "Create_Games.py"
SPEC = importlib.util.spec_from_file_location("create_games_manifest", CREATE_GAMES_PATH)
create_games = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(create_games)


def _frame(n=3):
    return pd.DataFrame(
        {
            "Date": ["2024-01-01"] * n,
            "GameDate": ["2024-01-02"] * n,
            "Score": [210.0, 205.0, 199.0],
            "Home-Team-Win": [1, 0, 1],
            "OU-Cover": [1, 0, 2],
        }
    )


class TestDatasetManifest(unittest.TestCase):
    """The manifest exists so a metrics run can be tied to the build it used."""

    def _write(self, tmpdir, frame, expected_team_rows=30, usable=(3,), sizes=None):
        create_games.OUTPUT_DB_PATH = Path(tmpdir) / "out.sqlite"
        create_games.write_manifest(
            frame,
            stats={"kept": len(frame)},
            expected_team_rows=expected_team_rows,
            usable_snapshots=range(usable[0]),
            team_table_sizes=sizes if sizes is not None else {"2024-01-01": 30},
            odds_tables_used={"2023-24": ["odds_2023-24_new", "2023-24"]},
        )
        return json.loads((Path(tmpdir) / "manifest.json").read_text())

    def test_manifest_records_fingerprint_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = self._write(tmpdir, _frame())
            entry = manifest["tables"][create_games.OUTPUT_TABLE]
            self.assertEqual(entry["rows"], 3)
            self.assertEqual(len(entry["content_sha256"]), 64)
            self.assertIn("base_rate", entry["labels"]["Home-Team-Win"])
            self.assertEqual(entry["build"]["expected_team_rows"], 30)
            self.assertEqual(entry["build"]["odds_tables_used"]["2023-24"],
                             ["odds_2023-24_new", "2023-24"])

    def test_content_hash_differs_between_builds(self):
        """Two builds of the same schema must not fingerprint the same."""
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            first = self._write(a, _frame())["tables"][create_games.OUTPUT_TABLE]
            changed = _frame()
            changed.loc[0, "Score"] = 999.0
            second = self._write(b, changed)["tables"][create_games.OUTPUT_TABLE]
            self.assertNotEqual(first["content_sha256"], second["content_sha256"])

            same_rows = _frame()
            third = self._write(a, same_rows)["tables"][create_games.OUTPUT_TABLE]
            self.assertEqual(first["content_sha256"], third["content_sha256"])

    def test_manifest_tolerates_a_frame_without_a_date_column(self):
        frame = _frame().drop(columns=["Date"])
        with tempfile.TemporaryDirectory() as tmpdir:
            entry = self._write(tmpdir, frame)["tables"][create_games.OUTPUT_TABLE]
            self.assertIsNone(entry["build"]["max_snapshot_lag_days"])

    def test_manifest_is_written_next_to_the_dataset(self):
        """A redirected dataset must redirect the manifest with it."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.sqlite"
            create_games.OUTPUT_DB_PATH = out
            create_games.write_manifest(
                _frame(), {}, 30, range(3), {"2024-01-01": 30}, {}
            )
            self.assertTrue((Path(tmpdir) / "manifest.json").exists())


class TestLoudFailure(unittest.TestCase):
    """A build that produces nothing must not look like a successful rebuild."""

    def test_no_usable_snapshots_raises_instead_of_returning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            teams = Path(tmpdir) / "teams.sqlite"
            odds = Path(tmpdir) / "odds.sqlite"
            out = Path(tmpdir) / "out.sqlite"
            with sqlite3.connect(teams) as con:
                pd.DataFrame({"TEAM_NAME": ["A"], "GP": [1]}).to_sql(
                    "2024-01-01", con, index=False)
            with sqlite3.connect(odds) as con:
                pd.DataFrame({"Date": ["2024-01-02"], "Home": ["A"], "Away": ["B"],
                              "OU": [200.0], "Points": [210], "Win_Margin": [10],
                              "Days_Rest_Home": [1], "Days_Rest_Away": [1]}).to_sql(
                    "2024-25", con, index=False)
            config = Path(tmpdir) / "config.toml"
            create_games.toml.dump(
                {"create-games": {"2024-25": {"start_date": "2024-01-01",
                                               "end_date": "2024-06-01"}}},
                config.open("w"),
            )
            saved = (create_games.CONFIG_PATH, create_games.ODDS_DB_PATH,
                     create_games.TEAMS_DB_PATH, create_games.OUTPUT_DB_PATH)
            try:
                create_games.CONFIG_PATH = config
                create_games.ODDS_DB_PATH = odds
                create_games.TEAMS_DB_PATH = teams
                create_games.OUTPUT_DB_PATH = out
                # One row means the derived league size is 1, below the floor.
                with self.assertRaises(ValueError):
                    create_games.main()
                self.assertFalse(out.exists())
            finally:
                (create_games.CONFIG_PATH, create_games.ODDS_DB_PATH,
                 create_games.TEAMS_DB_PATH, create_games.OUTPUT_DB_PATH) = saved


if __name__ == "__main__":
    unittest.main()