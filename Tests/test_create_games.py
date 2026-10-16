import importlib.util
import sqlite3
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

CREATE_GAMES_PATH = Path(__file__).resolve().parents[1] / "src" / "Process-Data" / "Create_Games.py"
SPEC = importlib.util.spec_from_file_location("create_games", CREATE_GAMES_PATH)
create_games = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(create_games)


class TestCreateGames(unittest.TestCase):

    def test_normalize_date(self):
        self.assertEqual(create_games.normalize_date("2025-01-02"), "2025-01-02")
        self.assertEqual(create_games.normalize_date(datetime(2025, 1, 2, 12, 0, 0)), "2025-01-02")

    def test_get_team_index_map(self):
        self.assertIs(create_games.get_team_index_map("2023-24"), create_games.team_index_current)
        self.assertIs(create_games.get_team_index_map("2015-16"), create_games.team_index_14)
        self.assertIs(create_games.get_team_index_map("bad-key"), create_games.team_index_current)

    def test_select_odds_table(self):
        with sqlite3.connect(":memory:") as con:
            con.execute('CREATE TABLE "odds_2024-25_new" (Date TEXT)')
            con.execute('CREATE TABLE "2023-24" (Date TEXT)')
            self.assertEqual(create_games.select_odds_table(con, "2024-25"), "odds_2024-25_new")
            self.assertEqual(create_games.select_odds_table(con, "2023-24"), "2023-24")
            self.assertIsNone(create_games.select_odds_table(con, "2022-23"))

    def test_build_game_features(self):
        team_df = pd.DataFrame(
            {
                "TEAM_ID": list(range(30)),
                "TEAM_NAME": [f"Team {i}" for i in range(30)],
                "STAT_A": list(range(30)),
            }
        )
        index_map = {"Team A": 0, "Team B": 1}
        features = create_games.build_game_features(team_df, "Team A", "Team B", index_map)
        self.assertIsNotNone(features)
        self.assertIn("TEAM_NAME", features.index)
        self.assertIn("TEAM_NAME.1", features.index)

    def test_fetch_team_table_missing(self):
        with sqlite3.connect(":memory:") as con:
            result = create_games.fetch_team_table(con, "2025-01-02")
        self.assertIsNone(result)

    def test_main_writes_dataset(self):
        config = {
            "create-games": {
                "2023-24": {
                    "start_date": "2023-10-23",
                    "end_date": "2024-04-28",
                }
            }
        }
        odds_rows = [
            {
                "Date": "2025-01-02",
                "Home": "Team A",
                "Away": "Team B",
                "OU": 210.5,
                "Points": 205.0,
                "Win_Margin": 5,
                "Days_Rest_Home": 2,
                "Days_Rest_Away": 4,
            }
        ]
        team_df = pd.DataFrame(
            {
                "TEAM_ID": list(range(30)),
                "TEAM_NAME": [f"Team {i}" for i in range(30)],
                "STAT_A": list(range(30)),
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            odds_path = Path(tmpdir) / "odds.sqlite"
            teams_path = Path(tmpdir) / "teams.sqlite"
            out_path = Path(tmpdir) / "out.sqlite"

            with sqlite3.connect(odds_path) as con:
                pd.DataFrame(odds_rows).to_sql("2023-24", con, if_exists="replace", index=False)
            with sqlite3.connect(teams_path) as con:
                # The pre-game snapshot, strictly BEFORE the 2025-01-02 game.
                # A same-day snapshot must NOT be used: the NBA endpoint treats
                # DateTo as inclusive, so the table named 2025-01-02 contains the
                # result of the very game being predicted. Joining it was the
                # target leakage behind the project's original 68.9% figure, so
                # Create_Games now reaches back to the newest snapshot strictly
                # before the game date.
                team_df.to_sql("2025-01-01", con, if_exists="replace", index=False)
                # A same-day snapshot is present too, and MUST be ignored.
                team_df.assign(STAT_A=[v + 1000 for v in team_df["STAT_A"]]).to_sql(
                    "2025-01-02", con, if_exists="replace", index=False
                )

            original_config_path = create_games.CONFIG_PATH
            original_odds_path = create_games.ODDS_DB_PATH
            original_teams_path = create_games.TEAMS_DB_PATH
            original_output_path = create_games.OUTPUT_DB_PATH
            original_output_table = create_games.OUTPUT_TABLE
            original_team_map = create_games.TEAM_INDEX_BY_SEASON

            try:
                create_games.CONFIG_PATH = Path(tmpdir) / "config.toml"
                create_games.ODDS_DB_PATH = odds_path
                create_games.TEAMS_DB_PATH = teams_path
                create_games.OUTPUT_DB_PATH = out_path
                create_games.OUTPUT_TABLE = "dataset_test"
                create_games.TEAM_INDEX_BY_SEASON = {"2023-24": {"Team A": 0, "Team B": 1}}

                create_games.toml.dump(config, create_games.CONFIG_PATH.open("w"))
                create_games.main()

                with sqlite3.connect(out_path) as con:
                    df = pd.read_sql_query('SELECT * FROM "dataset_test"', con)
            finally:
                create_games.CONFIG_PATH = original_config_path
                create_games.ODDS_DB_PATH = original_odds_path
                create_games.TEAMS_DB_PATH = original_teams_path
                create_games.OUTPUT_DB_PATH = original_output_path
                create_games.OUTPUT_TABLE = original_output_table
                create_games.TEAM_INDEX_BY_SEASON = original_team_map

        self.assertEqual(len(df.index), 1)
        self.assertIn("Score", df.columns)
        self.assertIn("OU-Cover", df.columns)

    def test_main_never_uses_a_same_day_snapshot(self):
        """A game must be joined against stats from strictly before its own date.

        Regression test for target leakage. The NBA `leaguedashteamstats` endpoint
        treats `DateTo` as inclusive, so the snapshot named D contains the result
        of the game played on D. Joining it let a rule comparing each team's W on
        D against D-1 recover the label 96.11% of the time.
        """
        config = {"create-games": {"2023-24": {"start_date": "2023-10-23", "end_date": "2024-04-28"}}}
        odds_rows = [{"Date": "2025-01-02", "Home": "Team A", "Away": "Team B",
                      "OU": 210.5, "Points": 205.0, "Win_Margin": 5,
                      "Days_Rest_Home": 2, "Days_Rest_Away": 4}]
        prior = pd.DataFrame({"TEAM_ID": list(range(30)), "TEAM_NAME": [f"Team {i}" for i in range(30)],
                              "STAT_A": list(range(30))})
        # Same-day snapshot carries a distinctive value in STAT_A. If any of it
        # leaks into the output, the join regressed back to the leaky behaviour.
        same_day = prior.assign(STAT_A=[v + 1000 for v in prior["STAT_A"]])

        with tempfile.TemporaryDirectory() as tmpdir:
            odds_path = Path(tmpdir) / "odds.sqlite"
            teams_path = Path(tmpdir) / "teams.sqlite"
            out_path = Path(tmpdir) / "out.sqlite"
            with sqlite3.connect(odds_path) as con:
                pd.DataFrame(odds_rows).to_sql("2023-24", con, if_exists="replace", index=False)
            with sqlite3.connect(teams_path) as con:
                prior.to_sql("2025-01-01", con, if_exists="replace", index=False)
                same_day.to_sql("2025-01-02", con, if_exists="replace", index=False)

            saved = (create_games.CONFIG_PATH, create_games.ODDS_DB_PATH, create_games.TEAMS_DB_PATH,
                     create_games.OUTPUT_DB_PATH, create_games.OUTPUT_TABLE, create_games.TEAM_INDEX_BY_SEASON)
            try:
                create_games.CONFIG_PATH = Path(tmpdir) / "config.toml"
                create_games.ODDS_DB_PATH = odds_path
                create_games.TEAMS_DB_PATH = teams_path
                create_games.OUTPUT_DB_PATH = out_path
                create_games.OUTPUT_TABLE = "dataset_test"
                create_games.TEAM_INDEX_BY_SEASON = {"2023-24": {"Team A": 0, "Team B": 1}}
                create_games.toml.dump(config, create_games.CONFIG_PATH.open("w"))
                create_games.main()
                with sqlite3.connect(out_path) as con:
                    df = pd.read_sql_query('SELECT * FROM "dataset_test"', con)
            finally:
                (create_games.CONFIG_PATH, create_games.ODDS_DB_PATH, create_games.TEAMS_DB_PATH,
                 create_games.OUTPUT_DB_PATH, create_games.OUTPUT_TABLE,
                 create_games.TEAM_INDEX_BY_SEASON) = saved

        self.assertEqual(len(df.index), 1)
        # Team A is row 0, Team B row 1, so STAT_A is 0 / 1 from the PRIOR day.
        self.assertEqual(df["STAT_A"].iloc[0], 0)
        self.assertEqual(df["STAT_A.1"].iloc[0], 1)
