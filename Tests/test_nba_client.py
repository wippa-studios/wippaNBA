"""Tests for ``src/Utils/nba_client.py``.

Every test here runs with NO NETWORK. Two things are being pinned:

1. ``normalise_snapshot``'s discard-vs-repair contract, including the
   exactly-30-rows rule that ``Create_Games.build_game_features`` depends on and
   the alphabetical row order that ``Dictionaries.team_index_current`` encodes
   positionally.
2. ``fetch_pregame_team_stats``'s date arithmetic -- that it really does pass
   ``game_date - 1 day`` to the endpoint -- proven by stubbing the endpoint out
   entirely, because the one-day offset is the only thing standing between this
   repo and a 96.11% label-recovery rate.

The last test asserts the module imports with ``nba_api`` absent from
``sys.modules``. A module-scope ``import nba_api`` would fail CI collection on
any environment that has not installed it, and a collection failure reads like a
test failure.
"""

import subprocess
import sys
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest import mock

import pandas as pd

from src.Utils import nba_client

REPO_ROOT = Path(__file__).resolve().parents[1]

# 30 teams in the exact order the NBA endpoint returns them, which is
# alphabetical by TEAM_NAME -- the same order Dictionaries.team_index_current
# encodes positionally.
TEAM_NAMES = [
    "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
    "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
    "Detroit Pistons", "Golden State Warriors", "Houston Rockets",
    "Indiana Pacers", "Los Angeles Clippers", "Los Angeles Lakers",
    "Memphis Grizzlies", "Miami Heat", "Milwaukee Bucks",
    "Minnesota Timberwolves", "New Orleans Pelicans", "New York Knicks",
    "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers",
    "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings",
    "San Antonio Spurs", "Toronto Raptors", "Utah Jazz", "Washington Wizards",
]


def make_snapshot(team_names=None, shuffle=False):
    """A synthetic but structurally faithful leaguedashteamstats frame."""
    names = TEAM_NAMES if team_names is None else team_names
    frame = pd.DataFrame({
        "TEAM_ID": [1610610000 + i for i in range(len(names))],
        "TEAM_NAME": names,
        "GP": [10 + i for i in range(len(names))],
        "W": [5 + i for i in range(len(names))],
        "L": [5 for _ in names],
        "W_PCT": [0.5] * len(names),
        "PTS": [110.0 + i for i in range(len(names))],
    })
    if shuffle:
        frame = frame.iloc[::-1].reset_index(drop=True)
    return frame


class TestNormaliseSnapshot(unittest.TestCase):

    def test_complete_snapshot_is_accepted(self):
        result = nba_client.normalise_snapshot(make_snapshot(), table_name="2025-01-15")
        self.assertIsNotNone(result)
        self.assertEqual(len(result.index), 30)

    def test_complete_snapshot_gets_team_name_and_date(self):
        result = nba_client.normalise_snapshot(make_snapshot(), table_name="2025-01-15")
        self.assertIn("TEAM_NAME", result.columns)
        self.assertIn("Date", result.columns)
        self.assertTrue((result["Date"] == "2025-01-15").all())

    def test_date_accepts_a_date_object(self):
        result = nba_client.normalise_snapshot(make_snapshot(), table_name=date(2025, 1, 15))
        self.assertTrue((result["Date"] == "2025-01-15").all())

    def test_date_is_absent_when_no_table_name_given(self):
        result = nba_client.normalise_snapshot(make_snapshot())
        self.assertNotIn("Date", result.columns)

    def test_row_order_is_alphabetical_and_matches_team_index_current(self):
        """Positional contract with Dictionaries.team_index_current.

        build_game_features does team_df.iloc[home_index] with that map's value,
        so row order is load-bearing: wrong rows, no exception, wrong features.
        """
        shuffled = make_snapshot(shuffle=True)
        result = nba_client.normalise_snapshot(shuffled, table_name="2025-01-15")
        self.assertEqual(list(result["TEAM_NAME"]), sorted(TEAM_NAMES))
        self.assertEqual(list(result.index), list(range(30)))

    def test_row_order_matches_the_dictionaries_index_map(self):
        from src.Utils.Dictionaries import team_index_current
        result = nba_client.normalise_snapshot(make_snapshot(), table_name="2025-01-15")
        for position, team in enumerate(result["TEAM_NAME"]):
            self.assertEqual(team_index_current[team], position, team)

    def test_column_order_is_identity_first_and_date_last(self):
        result = nba_client.normalise_snapshot(make_snapshot(), table_name="2025-01-15")
        self.assertEqual(list(result.columns)[0:2], ["TEAM_ID", "TEAM_NAME"])
        self.assertEqual(list(result.columns)[-1], "Date")

    def test_column_order_is_deterministic_across_calls(self):
        first = nba_client.normalise_snapshot(make_snapshot(), table_name="2025-01-15")
        second = nba_client.normalise_snapshot(make_snapshot(), table_name="2025-01-16")
        self.assertEqual(list(first.columns), list(second.columns))

    def test_shuffled_rows_and_ordered_rows_agree(self):
        ordered = nba_client.normalise_snapshot(make_snapshot(), table_name="2025-01-15")
        shuffled = nba_client.normalise_snapshot(
            make_snapshot(shuffle=True), table_name="2025-01-15"
        )
        pd.testing.assert_frame_equal(ordered, shuffled)

    def test_short_snapshot_returns_none(self):
        """29 teams must be discarded, not repaired.

        build_game_features returns None for anything that is not exactly 30
        rows, which silently drops the game; catching it here makes it visible.
        """
        result = nba_client.normalise_snapshot(make_snapshot(TEAM_NAMES[:29]))
        self.assertIsNone(result)

    def test_empty_snapshot_returns_none(self):
        self.assertIsNone(nba_client.normalise_snapshot(pd.DataFrame(data={})))

    def test_frame_with_columns_but_no_rows_returns_none(self):
        self.assertIsNone(nba_client.normalise_snapshot(make_snapshot(TEAM_NAMES[:0])))

    def test_missing_team_name_returns_none(self):
        """Documented contract: no TEAM_NAME -> None, never a raised error.

        The column is the join key for every downstream consumer, so inventing it
        would fabricate team identities.
        """
        frame = make_snapshot().drop(columns=["TEAM_NAME"])
        self.assertIsNone(nba_client.normalise_snapshot(frame))

    def test_none_input_returns_none(self):
        self.assertIsNone(nba_client.normalise_snapshot(None))

    def test_duplicate_team_names_return_none(self):
        names = list(TEAM_NAMES)
        names[1] = names[0]
        self.assertIsNone(nba_client.normalise_snapshot(make_snapshot(names)))

    def test_null_team_name_returns_none(self):
        frame = make_snapshot()
        frame.loc[0, "TEAM_NAME"] = None
        self.assertIsNone(nba_client.normalise_snapshot(frame))

    def test_expected_teams_is_honoured(self):
        """A caller mid-migration can accept a different league size."""
        result = nba_client.normalise_snapshot(make_snapshot(TEAM_NAMES[:29]),
                                               expected_teams=29)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.index), 29)

    def test_bad_table_name_raises(self):
        """Caller misuse is loud; bad upstream data is not."""
        with self.assertRaises(ValueError):
            nba_client.normalise_snapshot(make_snapshot(), table_name=12345)


class TestPregameDateArithmetic(unittest.TestCase):
    """Prove the endpoint really receives game_date - 1 day. No network.

    The endpoint class is stubbed at the nba_api import site so the wrapper's own
    argument mapping is exercised but no request is made.
    """

    def _captured_kwargs(self, season="2024-25", **call_kwargs):
        captured = {}

        class FakeLeagueDashTeamStats:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def get_data_frames(self):
                return [make_snapshot()]

        fake_module = mock.MagicMock()
        fake_module.LeagueDashTeamStats = FakeLeagueDashTeamStats

        with mock.patch.dict(
            sys.modules,
            {"nba_api": mock.MagicMock(), "nba_api.stats": mock.MagicMock(),
             "nba_api.stats.endpoints": mock.MagicMock(),
             "nba_api.stats.endpoints.leaguedashteamstats": fake_module},
        ):
            nba_client.fetch_pregame_team_stats(season, **call_kwargs)
        return captured

    def test_mid_season_date_is_backed_off_by_one_day(self):
        kwargs = self._captured_kwargs(game_date=date(2025, 1, 15))
        self.assertEqual(kwargs["date_to_nullable"], "1/14/2025")

    def test_january_first_crosses_the_year_boundary(self):
        kwargs = self._captured_kwargs(game_date=date(2025, 1, 1))
        self.assertEqual(kwargs["date_to_nullable"], "12/31/2024")

    def test_first_of_a_month_crosses_the_month_boundary(self):
        kwargs = self._captured_kwargs(game_date=date(2024, 11, 1))
        self.assertEqual(kwargs["date_to_nullable"], "10/31/2024")

    def test_march_first_in_a_leap_year(self):
        kwargs = self._captured_kwargs(game_date=date(2024, 3, 1))
        self.assertEqual(kwargs["date_to_nullable"], "2/29/2024")

    def test_datetime_is_truncated_to_its_date(self):
        kwargs = self._captured_kwargs(game_date=datetime(2025, 1, 15, 19, 30))
        self.assertEqual(kwargs["date_to_nullable"], "1/14/2025")

    def test_string_game_date_is_parsed_then_shifted(self):
        kwargs = self._captured_kwargs(game_date="1/1/2025")
        self.assertEqual(kwargs["date_to_nullable"], "12/31/2024")

    def test_season_and_defaults_are_forwarded(self):
        kwargs = self._captured_kwargs(game_date=date(2025, 1, 15))
        self.assertEqual(kwargs["season"], "2024-25")
        self.assertEqual(kwargs["measure_type_detailed_defense"], "Base")
        self.assertEqual(kwargs["per_mode_detailed"], "PerGame")
        self.assertEqual(kwargs["season_type_all_star"], "Regular Season")

    def test_timeout_proxy_and_headers_are_forwarded(self):
        kwargs = self._captured_kwargs(
            game_date=date(2025, 1, 15),
            timeout=7,
            proxy="http://proxy.example:8080",
            headers={"Host": "stats.nba.com"},
        )
        self.assertEqual(kwargs["timeout"], 7)
        self.assertEqual(kwargs["proxy"], "http://proxy.example:8080")
        self.assertEqual(kwargs["headers"], {"Host": "stats.nba.com"})

    def test_bad_game_date_raises(self):
        with self.assertRaises(ValueError):
            nba_client.fetch_pregame_team_stats("2024-25", "not-a-date")

    def test_result_is_a_normalisable_snapshot(self):
        class FakeLeagueDashTeamStats:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def get_data_frames(self):
                return [make_snapshot(shuffle=True)]

        fake_module = mock.MagicMock()
        fake_module.LeagueDashTeamStats = FakeLeagueDashTeamStats
        with mock.patch.dict(
            sys.modules,
            {"nba_api": mock.MagicMock(), "nba_api.stats": mock.MagicMock(),
             "nba_api.stats.endpoints": mock.MagicMock(),
             "nba_api.stats.endpoints.leaguedashteamstats": fake_module},
        ):
            raw = nba_client.fetch_pregame_team_stats("2024-25", date(2025, 1, 15))
        result = nba_client.normalise_snapshot(raw, table_name="2025-01-14")
        self.assertEqual(len(result.index), 30)
        self.assertTrue((result["Date"] == "2025-01-14").all())


class TestFetchTeamStats(unittest.TestCase):

    def test_date_to_is_passed_through_inclusive(self):
        """fetch_team_stats does NOT shift the date -- that is the trap."""
        captured = {}

        class FakeLeagueDashTeamStats:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def get_data_frames(self):
                return [make_snapshot()]

        fake_module = mock.MagicMock()
        fake_module.LeagueDashTeamStats = FakeLeagueDashTeamStats
        with mock.patch.dict(
            sys.modules,
            {"nba_api": mock.MagicMock(), "nba_api.stats": mock.MagicMock(),
             "nba_api.stats.endpoints": mock.MagicMock(),
             "nba_api.stats.endpoints.leaguedashteamstats": fake_module},
        ):
            nba_client.fetch_team_stats("2024-25", date_to=date(2025, 1, 15))
        self.assertEqual(captured["date_to_nullable"], "1/15/2025")

    def test_empty_response_yields_an_empty_frame(self):
        class FakeLeagueDashTeamStats:
            def __init__(self, **kwargs):
                pass

            def get_data_frames(self):
                return []

        fake_module = mock.MagicMock()
        fake_module.LeagueDashTeamStats = FakeLeagueDashTeamStats
        with mock.patch.dict(
            sys.modules,
            {"nba_api": mock.MagicMock(), "nba_api.stats": mock.MagicMock(),
             "nba_api.stats.endpoints": mock.MagicMock(),
             "nba_api.stats.endpoints.leaguedashteamstats": fake_module},
        ):
            result = nba_client.fetch_team_stats("2024-25")
        self.assertTrue(result.empty)
        self.assertIsNone(nba_client.normalise_snapshot(result))


class TestFormatDateTo(unittest.TestCase):

    def test_date(self):
        self.assertEqual(nba_client.format_date_to(date(2025, 1, 5)), "1/5/2025")

    def test_datetime_is_truncated(self):
        self.assertEqual(
            nba_client.format_date_to(datetime(2025, 1, 5, 20, 0)), "1/5/2025"
        )

    def test_none_stays_none(self):
        self.assertIsNone(nba_client.format_date_to(None))

    def test_string_passes_through(self):
        self.assertEqual(nba_client.format_date_to("1/5/2025"), "1/5/2025")

    def test_unsupported_type_raises(self):
        with self.assertRaises(ValueError):
            nba_client.format_date_to(20250105)


class TestDescribe(unittest.TestCase):

    def test_reports_the_host_and_the_availability_risk(self):
        info = nba_client.describe()
        self.assertEqual(info["host"], "stats.nba.com")
        self.assertIn("hardcodes", info["host_is_hardcoded"])
        self.assertIn("proxy", info["unreachable_from_datacenters"])
        self.assertIn("RETRYABLE", info["blocks_server_ips"])

    def test_states_the_inclusive_date_to_trap(self):
        self.assertIn("INCLUSIVE", nba_client.describe()["date_to_is_inclusive"])


class TestLazyImport(unittest.TestCase):

    def test_module_imports_without_nba_api_installed(self):
        """The sharp test: block nba_api out of sys.modules, then import.

        Run in a subprocess so the block cannot leak into this process, and so a
        stray sys.modules entry from another test cannot mask a top-level import.
        """
        script = (
            "import sys\n"
            "class Block:\n"
            "    def find_module(self, name, path=None):\n"
            "        return None\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'nba_api' or name.startswith('nba_api.'):\n"
            "            raise ImportError('nba_api is blocked for this test')\n"
            "        return None\n"
            "sys.meta_path.insert(0, Block())\n"
            "import src.Utils.nba_client as m\n"
            "assert 'nba_api' not in sys.modules\n"
            "print('imported OK, nba_api in sys.modules:', 'nba_api' in sys.modules)\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("imported OK", completed.stdout)

    def test_module_namespace_has_no_top_level_nba_api_binding(self):
        """Static check: the source must not import nba_api at module scope."""
        source = (REPO_ROOT / "src" / "Utils" / "nba_client.py").read_text()
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("import nba_api") or stripped.startswith("from nba_api"):
                self.assertTrue(
                    stripped.startswith(("import", "from")) and "    " in line,
                    f"nba_api imported at module scope: {line!r}",
                )
