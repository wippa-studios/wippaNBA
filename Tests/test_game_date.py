"""The dataset's game date is an explicit column, not an inferred offset.

`Data/OddsData.sqlite` stores the date a game was PLAYED on. The dataset used to
carry that same date in `Date`, and `training_reporting.match_test_rows_to_odds`
merged on it exactly. Then target leakage was fixed: the NBA
`leaguedashteamstats` endpoint treats `DateTo` as inclusive, so the snapshot
named D already contained the game played on D, and `Create_Games` began joining
each game to the newest COMPLETE snapshot STRICTLY BEFORE it. That made `Date`
the as-of snapshot date -- typically one day earlier -- and every exact-date
odds join missed by a day (9 of 1,722 held-out rows).

The fix is to carry the real game date as its own column, `GameDate`, and join
on that. These tests pin all four things that has to stay true:

  a. `Create_Games` writes the odds row's game date into `GameDate` while `Date`
     still holds the prior snapshot date, and does so as an ISO string.
  b. A join on `GameDate` matches far more rows than a join on `Date`.
  c. A dataset old enough to predate the column still joins, on `Date`, and says
     so out loud.
  d. `GameDate` never becomes a model feature: the moneyline count stays 106 and
     the totals count 107, which is what the committed boosters in `Models/`
     were trained on.

`src/Train-Models` contains a hyphen and cannot be imported as a package, so
`training_common` / `training_reporting` are loaded the way their own docstrings
describe: this directory on `sys.path` and imported under plain top-level names.
"""

import contextlib
import importlib.util
import io
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_MODELS_DIR = REPO_ROOT / "src" / "Train-Models"
CREATE_GAMES_PATH = REPO_ROOT / "src" / "Process-Data" / "Create_Games.py"
DATASET_DB = REPO_ROOT / "Data" / "dataset.sqlite"
REBUILT_TABLE = "dataset_2012-26"

# The counts the committed artifacts in Models/ require. src/Utils/features.py
# asserts column ORDER at inference; this asserts WIDTH at build time.
ML_FEATURE_COUNT = 106
UO_FEATURE_COUNT = 107

for _path in (REPO_ROOT, TRAIN_MODELS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import training_reporting  # noqa: E402
from training_common import (  # noqa: E402
    GAME_DATE_COLUMN,
    MONEYLINE_TARGET,
    TOTALS_TARGET,
    drop_columns_for,
)

from src.Utils import features  # noqa: E402

_CG_SPEC = importlib.util.spec_from_file_location("create_games", CREATE_GAMES_PATH)
create_games = importlib.util.module_from_spec(_CG_SPEC)
_CG_SPEC.loader.exec_module(create_games)


def build_dataset_once():
    """Run Create_Games.main() over two tiny databases; returns the written frame.

    One odds row on 2025-01-02 and a complete 30-team snapshot from 2025-01-01,
    plus the same-day 2025-01-02 snapshot that a leaky join would have used.
    """
    config = {"create-games": {"2023-24": {"start_date": "2023-10-23", "end_date": "2024-04-28"}}}
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
    prior = pd.DataFrame(
        {
            "TEAM_ID": list(range(30)),
            "TEAM_NAME": [f"Team {i}" for i in range(30)],
            "STAT_A": list(range(30)),
            # The real snapshot tables end in a Date column naming themselves;
            # that is the value the dataset's Date/Date.1 come from.
            "Date": "2025-01-01",
        }
    )
    same_day = prior.assign(STAT_A=[v + 1000 for v in prior["STAT_A"]], Date="2025-01-02")

    with tempfile.TemporaryDirectory() as tmpdir:
        odds_path = Path(tmpdir) / "odds.sqlite"
        teams_path = Path(tmpdir) / "teams.sqlite"
        out_path = Path(tmpdir) / "out.sqlite"

        with sqlite3.connect(odds_path) as con:
            pd.DataFrame(odds_rows).to_sql("2023-24", con, if_exists="replace", index=False)
        with sqlite3.connect(teams_path) as con:
            prior.to_sql("2025-01-01", con, if_exists="replace", index=False)
            same_day.to_sql("2025-01-02", con, if_exists="replace", index=False)

        saved = (
            create_games.CONFIG_PATH,
            create_games.ODDS_DB_PATH,
            create_games.TEAMS_DB_PATH,
            create_games.OUTPUT_DB_PATH,
            create_games.OUTPUT_TABLE,
            create_games.TEAM_INDEX_BY_SEASON,
        )
        try:
            create_games.CONFIG_PATH = Path(tmpdir) / "config.toml"
            create_games.ODDS_DB_PATH = odds_path
            create_games.TEAMS_DB_PATH = teams_path
            create_games.OUTPUT_DB_PATH = out_path
            create_games.OUTPUT_TABLE = "dataset_test"
            create_games.TEAM_INDEX_BY_SEASON = {"2023-24": {"Team A": 0, "Team B": 1}}
            create_games.toml.dump(config, create_games.CONFIG_PATH.open("w"))
            with contextlib.redirect_stdout(io.StringIO()):
                create_games.main()
            with sqlite3.connect(out_path) as con:
                return pd.read_sql_query('SELECT * FROM "dataset_test"', con)
        finally:
            (
                create_games.CONFIG_PATH,
                create_games.ODDS_DB_PATH,
                create_games.TEAMS_DB_PATH,
                create_games.OUTPUT_DB_PATH,
                create_games.OUTPUT_TABLE,
                create_games.TEAM_INDEX_BY_SEASON,
            ) = saved


def synthetic_rows(count=60, with_game_date=True):
    """``count`` test rows whose snapshot date trails the game date by a day."""
    rows = []
    for i in range(count):
        game_date = pd.Timestamp("2025-03-01") + pd.Timedelta(days=i)
        snapshot_date = game_date - pd.Timedelta(days=1)
        row = {
            "Date": snapshot_date.strftime("%Y-%m-%d"),
            "TEAM_NAME": f"Home {i}",
            "TEAM_NAME.1": f"Away {i}",
            "Home-Team-Win": i % 2,
        }
        if with_game_date:
            row[GAME_DATE_COLUMN] = game_date.strftime("%Y-%m-%d")
        rows.append(row)
    return pd.DataFrame(rows)


def synthetic_odds(rows, date_column):
    """An odds frame keyed on ``date_column``, priced like load_closing_odds()."""
    return pd.DataFrame(
        {
            "Date": rows[date_column].tolist(),
            "TEAM_NAME": rows["TEAM_NAME"].tolist(),
            "TEAM_NAME.1": rows["TEAM_NAME.1"].tolist(),
            "ML_Home": [-110] * len(rows),
            "ML_Away": [-110] * len(rows),
        }
    )


def dataset_columns(table=REBUILT_TABLE):
    """The stored column order of ``table``, or None when it is unavailable."""
    if not DATASET_DB.exists():
        return None
    with sqlite3.connect(DATASET_DB) as con:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if rows is None:
            return None
        return [row[1] for row in con.execute(f'PRAGMA table_info("{table}")')]


class TestGameDateColumn(unittest.TestCase):
    """(a) GameDate is the odds row's date; Date stays the prior snapshot."""

    @classmethod
    def setUpClass(cls):
        cls.frame = build_dataset_once()

    def test_game_date_is_the_odds_row_date(self):
        self.assertIn(GAME_DATE_COLUMN, self.frame.columns)
        self.assertEqual(self.frame[GAME_DATE_COLUMN].iloc[0], "2025-01-02")

    def test_date_is_the_prior_snapshot_not_the_game_date(self):
        # The 2025-01-01 table is the newest complete snapshot strictly before the
        # 2025-01-02 game. The same-day table must not be used: the endpoint
        # treats DateTo as inclusive, so it holds this game's own result.
        self.assertEqual(self.frame["Date"].iloc[0], "2025-01-01")
        self.assertNotEqual(self.frame["Date"].iloc[0], self.frame[GAME_DATE_COLUMN].iloc[0])

    def test_game_date_is_a_string_and_not_nan(self):
        # Create_Games coerces only the numeric columns to float. A GameDate
        # cast to float would be a column of NaN and the odds join would die
        # quietly, so it has to survive the loop as the ISO string read from
        # the odds table.
        value = self.frame[GAME_DATE_COLUMN].iloc[0]
        self.assertIsInstance(value, str)
        self.assertEqual(pd.to_datetime(value).strftime("%Y-%m-%d"), value)

    def test_game_date_sits_before_the_label_block(self):
        # Placement is part of the contract: GameDate joins the date keys, and
        # every column after it keeps its position relative to the others, so
        # OU still sits between the away-team block and OU-Cover.
        columns = list(self.frame.columns)
        self.assertLess(columns.index(GAME_DATE_COLUMN), columns.index("Score"))
        self.assertLess(columns.index("OU"), columns.index("OU-Cover"))
        self.assertLess(columns.index("TEAM_NAME.1"), columns.index("OU"))

    def test_leak_free_features_still_come_from_the_prior_snapshot(self):
        # STAT_A is 0 / 1 from the 2025-01-01 table; the same-day table carries
        # 1000 / 1001. Guards that adding GameDate did not disturb the join.
        self.assertEqual(self.frame["STAT_A"].iloc[0], 0)
        self.assertEqual(self.frame["STAT_A.1"].iloc[0], 1)


class TestOddsJoinUsesGameDate(unittest.TestCase):
    """(b) joining on GameDate beats joining on Date, by a wide margin."""

    def test_join_on_game_date_matches_every_row(self):
        rows = synthetic_rows()
        odds = synthetic_odds(rows, GAME_DATE_COLUMN)
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            matched = training_reporting.match_test_rows_to_odds(rows, odds)
        self.assertEqual(len(matched), len(rows))
        self.assertNotIn("Warning", captured.getvalue())

    def test_join_on_date_alone_matches_almost_nothing(self):
        # The regression itself: an exact-date merge on the snapshot date.
        rows = synthetic_rows()
        odds = synthetic_odds(rows, GAME_DATE_COLUMN)
        on_date = rows[["Date", "TEAM_NAME", "TEAM_NAME.1"]].merge(
            odds, on=["Date", "TEAM_NAME", "TEAM_NAME.1"], how="inner"
        )
        with contextlib.redirect_stdout(io.StringIO()):
            matched = training_reporting.match_test_rows_to_odds(rows, odds)
        self.assertEqual(len(on_date), 0)
        self.assertGreater(len(matched), len(on_date) * 10)

    def test_join_preserves_row_order_for_truth_alignment(self):
        rows = synthetic_rows()
        odds = synthetic_odds(rows, GAME_DATE_COLUMN)
        with contextlib.redirect_stdout(io.StringIO()):
            matched = training_reporting.match_test_rows_to_odds(rows, odds)
        self.assertEqual(list(matched["_row"]), list(range(len(rows))))
        self.assertEqual(list(matched[GAME_DATE_COLUMN if GAME_DATE_COLUMN in matched else "Date"]),
                         list(rows[GAME_DATE_COLUMN]))


class TestOddsJoinFallback(unittest.TestCase):
    """(c) a dataset without GameDate still joins, on Date, and warns."""

    def test_date_column_is_chosen_when_game_date_is_absent(self):
        rows = synthetic_rows(with_game_date=False)
        column, fell_back = training_reporting.odds_join_date_column(rows)
        self.assertEqual(column, "Date")
        self.assertTrue(fell_back)

    def test_game_date_is_preferred_when_present(self):
        rows = synthetic_rows()
        column, fell_back = training_reporting.odds_join_date_column(rows)
        self.assertEqual(column, GAME_DATE_COLUMN)
        self.assertFalse(fell_back)

    def test_fallback_still_matches_every_row(self):
        # dataset_2012-23 and dataset_2012-24 predate the leak-free join, so on
        # those tables Date IS the game date and the fallback is correct.
        rows = synthetic_rows(with_game_date=False)
        odds = synthetic_odds(rows, "Date")
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            matched = training_reporting.match_test_rows_to_odds(rows, odds)
        self.assertEqual(len(matched), len(rows))
        self.assertIn("Warning", captured.getvalue())
        self.assertIn(GAME_DATE_COLUMN, captured.getvalue())

    def test_fallback_is_quiet_when_the_column_exists(self):
        rows = synthetic_rows()
        odds = synthetic_odds(rows, GAME_DATE_COLUMN)
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            training_reporting.match_test_rows_to_odds(rows, odds)
        self.assertEqual(captured.getvalue(), "")

    def test_module_level_constants_agree(self):
        self.assertEqual(training_reporting.GAME_DATE_COLUMN, GAME_DATE_COLUMN)
        self.assertEqual(training_reporting.SNAPSHOT_DATE_COLUMN, "Date")


class TestGameDateIsNeverAFeature(unittest.TestCase):
    """(d) the moneyline count stays 106 and the totals count stays 107."""

    def test_every_drop_list_removes_game_date(self):
        lists = {
            "drop_columns_for(MONEYLINE)": drop_columns_for(MONEYLINE_TARGET),
            "drop_columns_for(TOTALS)": drop_columns_for(TOTALS_TARGET),
            "drop_columns_for(MONEYLINE, ids)": drop_columns_for(MONEYLINE_TARGET, True),
            "drop_columns_for(TOTALS, ids)": drop_columns_for(TOTALS_TARGET, True),
            "ML_DROP_COLUMNS": features.ML_DROP_COLUMNS,
            "UO_LABEL_DROP_COLUMNS": features.UO_LABEL_DROP_COLUMNS,
            "NN_ML_DROP_COLUMNS": features.NN_ML_DROP_COLUMNS,
            "NN_UO_DROP_COLUMNS": features.NN_UO_DROP_COLUMNS,
        }
        for name, drops in lists.items():
            self.assertIn(GAME_DATE_COLUMN, drops, name)

    def test_no_suffixed_game_date_column_is_produced(self):
        # Nothing in the build writes GameDate.1, so a drop list claiming to
        # remove it would be padding the contract with a fiction.
        columns = dataset_columns()
        if columns is None:
            self.skipTest(f"{DATASET_DB.name} is unavailable")
        self.assertNotIn(f"{GAME_DATE_COLUMN}.1", columns)

    def test_rebuilt_dataset_feature_counts_are_unchanged(self):
        columns = dataset_columns()
        if columns is None:
            self.skipTest(f"{REBUILT_TABLE} is not in {DATASET_DB.name}")
        self.assertIn(GAME_DATE_COLUMN, columns)

        ml_order = features.expected_feature_order(
            columns, drop_columns_for(MONEYLINE_TARGET, include_team_ids=True)
        )
        uo_order = features.expected_feature_order(
            columns, drop_columns_for(TOTALS_TARGET, include_team_ids=True)
        )
        self.assertEqual(len(ml_order), ML_FEATURE_COUNT)
        self.assertEqual(len(uo_order), UO_FEATURE_COUNT)
        self.assertNotIn(GAME_DATE_COLUMN, ml_order)
        self.assertNotIn(GAME_DATE_COLUMN, uo_order)

    def test_published_feature_lists_match_the_trainer_counts(self):
        columns = dataset_columns()
        if columns is None:
            self.skipTest(f"{REBUILT_TABLE} is not in {DATASET_DB.name}")
        self.assertEqual(len(features.expected_feature_order(columns, features.ML_DROP_COLUMNS)),
                         ML_FEATURE_COUNT)
        self.assertEqual(len(features.expected_feature_order(columns, features.UO_LABEL_DROP_COLUMNS)),
                         UO_FEATURE_COUNT)
        self.assertEqual(len(features.expected_feature_order(columns, features.NN_ML_DROP_COLUMNS)),
                         ML_FEATURE_COUNT)
        self.assertEqual(len(features.expected_feature_order(columns, features.NN_UO_DROP_COLUMNS)),
                         UO_FEATURE_COUNT)

    def test_totals_order_is_the_moneyline_order_plus_the_closing_total(self):
        columns = dataset_columns()
        if columns is None:
            self.skipTest(f"{REBUILT_TABLE} is not in {DATASET_DB.name}")
        ml_order = features.expected_feature_order(
            columns, drop_columns_for(MONEYLINE_TARGET, include_team_ids=True)
        )
        uo_order = features.expected_feature_order(
            columns, drop_columns_for(TOTALS_TARGET, include_team_ids=True)
        )
        self.assertEqual(set(uo_order), set(ml_order) | {"OU"})
        self.assertEqual(len(uo_order), len(ml_order) + 1)
        # OU keeps its dataset position, between the away block and OU-Cover,
        # rather than being appended.
        neighbours = [column for column in uo_order if column.endswith("_RANK.1") or column == "Date.1"]
        self.assertLess(uo_order.index("OU"), len(uo_order))
        self.assertGreater(uo_order.index("OU"), uo_order.index(neighbours[-1]))


if __name__ == "__main__":
    unittest.main()