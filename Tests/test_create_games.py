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

    def test_select_odds_tables_returns_every_candidate_not_just_the_first(self):
        """All existing candidates come back so the caller can union them.

        Regression test for the precedence bug: the scraper writes to the bare
        season key while the original lookup preferred "odds_<season>_new", so
        for two of the three configured seasons the scraper's output went to a
        table the builder never read -- which also made the --backfill fix
        inert, since backfilled rows landed in the unread table.
        """
        with sqlite3.connect(":memory:") as con:
            con.execute('CREATE TABLE "odds_2024-25_new" (Date TEXT)')
            con.execute('CREATE TABLE "2023-24" (Date TEXT)')
            con.execute('CREATE TABLE "odds_2019-20_new" (Date TEXT)')
            con.execute('CREATE TABLE "2019-20" (Date TEXT)')

            self.assertEqual(
                create_games.select_odds_tables(con, "2024-25"), ["odds_2024-25_new"]
            )
            self.assertEqual(create_games.select_odds_tables(con, "2023-24"), ["2023-24"])
            self.assertEqual(
                create_games.select_odds_tables(con, "2019-20"),
                ["odds_2019-20_new", "2019-20"],
            )
            self.assertEqual(create_games.select_odds_tables(con, "2022-23"), [])

    def test_load_odds_for_season_unions_and_deduplicates(self):
        with sqlite3.connect(":memory:") as con:
            pd.DataFrame({"Date": ["2024-01-01", "2024-01-02"],
                          "Home": ["A", "C"], "Away": ["B", "D"],
                          "Points": [200, 210]}).to_sql("odds_2019-20_new", con, index=False)
            pd.DataFrame({"Date": ["2024-01-01", "2024-01-03"],
                          "Home": ["A", "E"], "Away": ["B", "F"],
                          "Points": [200, 220]}).to_sql("2019-20", con, index=False)

            combined = create_games.load_odds_for_season(
                con, ["odds_2019-20_new", "2019-20"], con
            )
            # 4 rows across the two tables; 2024-01-01 is in both -> 3 games.
            self.assertEqual(len(combined.index), 3)
            self.assertEqual(
                sorted(combined["Date"]), ["2024-01-01", "2024-01-02", "2024-01-03"]
            )

    def test_derive_expected_team_rows_uses_the_mode_not_a_constant(self):
        """League size comes from the data, so 31 teams cannot break it."""
        counts = {f"2024-01-{d:02d}": 30 for d in range(1, 29)}
        counts.update({"2024-02-01": 0, "2024-02-02": 6, "2024-02-03": 28})
        self.assertEqual(create_games.derive_expected_team_rows(counts), 30)

        counts31 = {f"2024-03-{d:02d}": 31 for d in range(1, 20)}
        counts31["2024-04-01"] = 12
        self.assertEqual(create_games.derive_expected_team_rows(counts31), 31)

        # Implausible data must raise rather than silently yield zero snapshots.
        with self.assertRaises(ValueError):
            create_games.derive_expected_team_rows({"2024-01-01": 3})
        with self.assertRaises(ValueError):
            create_games.derive_expected_team_rows({"2024-01-01": 0})

    def test_build_game_features(self):
        team_df = pd.DataFrame(
            {
                "TEAM_ID": list(range(30)),
                "TEAM_NAME": [f"Team {i}" for i in range(30)],
                "STAT_A": list(range(30)),
            }
        )
        index_map = {"Team A": 0, "Team B": 1}
        features = create_games.build_game_features(team_df, "Team A", "Team B", index_map, 30)
        self.assertIsNotNone(features)
        self.assertIn("TEAM_NAME", features.index)
        self.assertIn("TEAM_NAME.1", features.index)

    def test_build_game_features_resolves_by_name_not_position(self):
        """A stale cross-season snapshot must not mis-attribute a team.

        Regression test. The original .iloc[index_map[team]] lookup shifted by
        one when a franchise was missing from the frame, so a Cleveland game was
        written out carrying Dallas's record -- 21 rows measured on
        dataset_2012-26. Resolving by canonical name is immune to row order and
        to franchises joining or leaving.
        """
        # One team absent, exactly as in a real prior-season snapshot.
        names = [f"Team {i}" for i in range(30) if i != 2]
        team_df = pd.DataFrame(
            {"TEAM_ID": range(29), "TEAM_NAME": names, "STAT_A": range(100, 129)}
        )
        stale_map = {f"Team {i}": i for i in range(30)}

        features = create_games.build_game_features(team_df, "Team 5", "Team 9", stale_map, 29)
        self.assertIsNotNone(features)
        self.assertEqual(features["TEAM_NAME"], "Team 5")
        self.assertEqual(features["TEAM_NAME.1"], "Team 9")
        # Team 5 sits at position 4 in this frame (Team 2 is absent), so the
        # correct STAT_A is 104. Positional lookup would have returned 105 --
        # which is Team 6's row, i.e. the wrong team's record entirely.
        self.assertEqual(features["STAT_A"], 104)
        self.assertEqual(features["STAT_A.1"], 108)

    def test_build_game_features_handles_alias_spellings(self):
        team_df = pd.DataFrame(
            {"TEAM_ID": range(30), "TEAM_NAME": [f"Team {i}" for i in range(30)],
             "STAT_A": range(30)}
        )
        df_alias = team_df.copy()
        df_alias.loc[12, "TEAM_NAME"] = "Los Angeles Clippers"
        index_map = {"LA Clippers": 12, "Los Angeles Clippers": 12}
        features = create_games.build_game_features(
            df_alias, "LA Clippers", "Team 3", index_map, 30
        )
        self.assertIsNotNone(features)
        self.assertEqual(features["STAT_A"], 12)

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
