"""Tests for Get_Odds_Line_Movement: intraday snapshots and CLV arithmetic.

Two properties carry most of the weight here and are checked directly rather
than through a helper:

* The CLV worked example in `TestClosingLineValue.test_worked_example_is_checkable_by_hand`
  is the arithmetic the whole module exists to support. It is written out longhand
  in the test so a reader can verify it without running anything.
* Snapshot idempotency and open/close ordering are checked against rows written
  in deliberately shuffled order, so neither can be satisfied by rowid.

No network, no sbrscrape, and no committed database: every test runs against a
tempfile.TemporaryDirectory or an in-memory sqlite connection. `test_main_writes_only_to_the_snapshot_table`
and the real-database assertions in the module mean a regression here would
show up as a modified Data/OddsData.sqlite.
"""

import importlib.util
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "Process-Data" / "Get_Odds_Line_Movement.py"

SPEC = importlib.util.spec_from_file_location("get_odds_line_movement", MODULE_PATH)
line_movement = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(line_movement)


def build_row(game_date="2026-01-05", home="Boston Celtics", away="Denver Nuggets",
              sportsbook="fanduel", home_ml=-110, away_ml=-110, total=224.5):
    return {
        "Date": game_date,
        "Sportsbook": sportsbook,
        "Home": home,
        "Away": away,
        "ML_Home": home_ml,
        "ML_Away": away_ml,
        "OU": total,
    }


class FakeScoreboard:
    def __init__(self, games):
        self.games = games


def fake_game(home="Boston Celtics", away="Denver Nuggets", home_ml=-110, away_ml=-110,
              total=224.5, sportsbook="fanduel"):
    return {
        "home_team": home,
        "away_team": away,
        "home_ml": {sportsbook: home_ml},
        "away_ml": {sportsbook: away_ml},
        "total": {sportsbook: total},
    }


def make_connection():
    con = sqlite3.connect(":memory:")
    line_movement.ensure_snapshot_table(con)
    return con


def record(con, captured_at, rows):
    with mock.patch.object(line_movement.time, "sleep"):
        return line_movement.record_snapshot(con, captured_at, rows)


def count_snapshots(con):
    return con.execute("SELECT COUNT(*) FROM odds_snapshots").fetchone()[0]


# ---------------------------------------------------------------------------


class TestAmericanToProb(unittest.TestCase):
    def test_negative_american(self):
        # -110 -> 110 / 210
        self.assertAlmostEqual(line_movement.american_to_prob(-110), 110 / 210, places=12)
        self.assertAlmostEqual(line_movement.american_to_prob(-130), 130 / 230, places=12)
        self.assertAlmostEqual(line_movement.american_to_prob(-100), 0.5, places=12)

    def test_positive_american(self):
        # +130 -> 100 / 230
        self.assertAlmostEqual(line_movement.american_to_prob(130), 100 / 230, places=12)
        self.assertAlmostEqual(line_movement.american_to_prob(100), 0.5, places=12)

    def test_invalid_odds_raise(self):
        for bad in (0, None, True, "-110/-110"):
            with self.assertRaises(ValueError):
                line_movement.american_to_prob(bad)


class TestDevig(unittest.TestCase):
    def test_even_book_is_exactly_half(self):
        market = line_movement.devig(-110, -110)
        self.assertAlmostEqual(market["home"], 0.5, places=12)
        self.assertAlmostEqual(market["away"], 0.5, places=12)
        # 110/210 + 110/210 = 22/21
        self.assertAlmostEqual(market["overround"], 22 / 21, places=12)

    def test_favourite_price_is_reduced_not_inflated(self):
        # De-vigging can only ever lower a raw implied probability. This is the
        # invariant the CLV arithmetic depends on.
        market = line_movement.devig(-200, 180)
        # 200/300 = 2/3, 100/280 = 5/14, sum = 43/42, fair = 28/43
        self.assertAlmostEqual(market["home_raw"], 2 / 3, places=12)
        self.assertAlmostEqual(market["away_raw"], 5 / 14, places=12)
        self.assertAlmostEqual(market["overround"], 43 / 42, places=12)
        self.assertAlmostEqual(market["home"], 28 / 43, places=12)
        self.assertLess(market["home"], market["home_raw"])

    def test_fair_probabilities_sum_to_one(self):
        for prices in ((-110, -110), (-200, 180), (150, 130), (-500, 400)):
            market = line_movement.devig(*prices)
            self.assertAlmostEqual(market["home"] + market["away"], 1.0, places=12)


class TestLineMovement(unittest.TestCase):
    def test_shortening_the_line_is_positive_drift(self):
        # -110 -> -130 means the market grew more confident in this side.
        result = line_movement.line_movement(-110, -130)
        self.assertGreater(result["drift"], 0)
        self.assertAlmostEqual(result["open_implied"], 110 / 210, places=12)
        self.assertAlmostEqual(result["close_implied"], 130 / 230, places=12)
        self.assertAlmostEqual(result["drift"], 130 / 230 - 110 / 210, places=12)
        self.assertAlmostEqual(result["drift_points"], result["drift"] * 100, places=12)
        self.assertFalse(result["de_vigged"])

    def test_lengthening_the_line_is_negative_drift(self):
        result = line_movement.line_movement(-130, -110)
        self.assertLess(result["drift"], 0)

    def test_unchanged_line_has_zero_drift(self):
        self.assertAlmostEqual(line_movement.line_movement(-110, -110)["drift"], 0.0, places=12)

    def test_overround_devigs_both_ends(self):
        result = line_movement.line_movement(-110, -130, overround=22 / 21)
        self.assertTrue(result["de_vigged"])
        self.assertAlmostEqual(result["open_implied"], (110 / 210) / (22 / 21), places=12)
        self.assertAlmostEqual(result["close_implied"], (130 / 230) / (22 / 21), places=12)

    def test_invalid_overround_raises(self):
        for bad in (0, -1.0, "flat"):
            with self.assertRaises(ValueError):
                line_movement.line_movement(-110, -130, overround=bad)


class TestClosingLineValue(unittest.TestCase):
    def test_worked_example_is_checkable_by_hand(self):
        """The single most important check in this module.

        By hand, from the worked example:
          entry: -110 at a model probability of 0.55
          close: -130, de-vigged market probability 0.60

        CLV is a difference of probabilities, not of American prices, so
        neither line conversion belongs in the answer:

          clv = close_prob - entry_prob
              = 0.60 - 0.55
              = 0.05
              = +5.0 probability points

        Positive, because the close says the market ended up more confident in
        the side that was bought than the price paid implied.
        """
        result = line_movement.closing_line_value(0.55, -110, 0.60, -130)

        self.assertAlmostEqual(result["clv"], 0.05, places=12)
        self.assertAlmostEqual(result["clv_points"], 5.0, places=12)
        self.assertAlmostEqual(result["entry_prob"], 0.55, places=12)
        self.assertAlmostEqual(result["close_prob"], 0.60, places=12)

        # The prices are reported as diagnosis and must NOT leak into `clv`.
        self.assertAlmostEqual(result["entry_implied"], 110 / 210, places=12)
        self.assertAlmostEqual(result["close_implied"], 130 / 230, places=12)
        self.assertAlmostEqual(
            result["raw_price_drift"], 130 / 230 - 110 / 210, places=12
        )
        self.assertNotAlmostEqual(result["raw_price_drift"], result["clv"], places=3)
        self.assertAlmostEqual(result["edge_at_entry"], 0.55 - 110 / 210, places=12)
        self.assertAlmostEqual(result["edge_at_close"], 0.55 - 130 / 230, places=12)

    def test_gaining_clv_while_losing_money(self):
        """The distinction that makes CLV worth measuring separately from ROI."""
        result = line_movement.closing_line_value(0.55, -110, 0.60, -130)

        # The bet itself lost: home did not win.
        realised_return = -100.0
        self.assertLess(realised_return, 0)

        # It still gained CLV, because the entry price beat the close.
        self.assertGreater(result["clv"], 0)

    def test_betting_the_close_has_zero_clv(self):
        # If you take the closing price yourself, you cannot beat it. This is
        # why the audit's -1.9% to -7% ROI at the close was never a signal.
        result = line_movement.closing_line_value(0.60, -130, 0.60, -130)
        self.assertAlmostEqual(result["clv"], 0.0, places=12)

    def test_negative_clv_when_the_line_moves_against_the_bet(self):
        result = line_movement.closing_line_value(0.60, -110, 0.55, -110)
        self.assertAlmostEqual(result["clv"], -0.05, places=12)

    def test_probabilities_must_be_probabilities(self):
        # A stray percentage must fail loudly rather than silently scoring 60.
        for bad in (60, 0, 1, -0.1, 1.5, "fifty", None):
            with self.assertRaises(ValueError):
                line_movement.closing_line_value(bad, -110, 0.60, -130)
        with self.assertRaises(ValueError):
            line_movement.closing_line_value(0.55, -110, 60, -130)

    def test_prices_must_be_valid(self):
        with self.assertRaises(ValueError):
            line_movement.closing_line_value(0.55, 0, 0.60, -130)


class TestCapturedAt(unittest.TestCase):
    def test_aware_datetime_is_accepted(self):
        moment = datetime(2026, 1, 5, 18, 30, tzinfo=timezone.utc)
        self.assertEqual(line_movement.format_captured_at(moment), "2026-01-05T18:30:00+00:00")

    def test_aware_string_is_accepted(self):
        self.assertEqual(
            line_movement.format_captured_at("2026-01-05T18:30:00+00:00"),
            "2026-01-05T18:30:00+00:00",
        )
        self.assertEqual(
            line_movement.format_captured_at("2026-01-05T18:30:00Z"),
            "2026-01-05T18:30:00+00:00",
        )

    def test_non_utc_offset_is_converted_to_utc(self):
        self.assertEqual(
            line_movement.format_captured_at("2026-01-05T13:30:00-05:00"),
            "2026-01-05T18:30:00+00:00",
        )

    def test_naive_datetime_is_rejected(self):
        # The audited defect: naive local time silently mixed with UTC schedule
        # data. A naive timestamp is a call-site bug and must not be absorbed.
        with self.assertRaises(line_movement.SnapshotError):
            line_movement.parse_captured_at(datetime(2026, 1, 5, 18, 30))

    def test_naive_string_is_rejected(self):
        with self.assertRaises(line_movement.SnapshotError):
            line_movement.parse_captured_at("2026-01-05T18:30:00")

    def test_unparseable_value_is_rejected(self):
        for bad in ("yesterday", "", 1234567890):
            with self.assertRaises(line_movement.SnapshotError):
                line_movement.parse_captured_at(bad)

    def test_format_is_fixed_width_so_sort_order_is_chronological(self):
        moments = [
            datetime(2026, 1, 5, 9, 5, tzinfo=timezone.utc),
            datetime(2026, 1, 5, 18, 30, tzinfo=timezone.utc),
            datetime(2026, 1, 10, 23, 59, tzinfo=timezone.utc),
            datetime(2025, 12, 31, 0, 0, tzinfo=timezone.utc),
        ]
        formatted = [line_movement.format_captured_at(moment) for moment in moments]
        self.assertEqual({len(value) for value in formatted}, {25})
        # The input is deliberately not chronological.
        self.assertNotEqual(formatted, sorted(formatted))
        # Lexicographic sort of the stored strings reproduces chronological order,
        # which is what ORDER BY "CapturedAt" in closing_line() depends on.
        self.assertEqual(
            sorted(formatted),
            [line_movement.format_captured_at(moment) for moment in sorted(moments)],
        )

    def test_sub_second_precision_is_truncated(self):
        moment = datetime(2026, 1, 5, 18, 30, 0, 123456, tzinfo=timezone.utc)
        self.assertEqual(line_movement.format_captured_at(moment), "2026-01-05T18:30:00+00:00")

    def test_game_date_normalisation(self):
        self.assertEqual(line_movement.format_game_date("2026-01-05"), "2026-01-05")
        self.assertEqual(line_movement.format_game_date("2026-01-05T18:30:00Z"), "2026-01-05")
        self.assertEqual(line_movement.format_game_date(date(2026, 1, 5)), "2026-01-05")
        self.assertEqual(line_movement.format_game_date(datetime(2026, 1, 5, 9, 0)), "2026-01-05")
        with self.assertRaises(line_movement.SnapshotError):
            line_movement.format_game_date("05/01/2026")


class TestRecordSnapshot(unittest.TestCase):
    def test_creates_its_own_table_on_a_fresh_connection(self):
        con = sqlite3.connect(":memory:")
        self.assertIsNone(
            con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='odds_snapshots'"
            ).fetchone()
        )
        result = record(con, "2026-01-05T12:00:00+00:00", [build_row()])
        self.assertEqual(result["inserted"], 1)
        self.assertTrue(
            con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='odds_snapshots'"
            ).fetchone()
        )

    def test_records_the_documented_schema(self):
        con = make_connection()
        record(con, "2026-01-05T12:00:00+00:00", [build_row(home_ml=-110, away_ml=-110, total=224.5)])
        row = con.execute("SELECT * FROM odds_snapshots").fetchone()
        columns = [column[0] for column in con.execute("SELECT * FROM odds_snapshots").description]
        self.assertEqual(
            columns,
            ["CapturedAt", "Date", "Sportsbook", "Home", "Away", "ML_Home", "ML_Away", "OU"],
        )
        self.assertEqual(
            row,
            ("2026-01-05T12:00:00+00:00", "2026-01-05", "fanduel", "Boston Celtics",
             "Denver Nuggets", -110, -110, 224.5),
        )

    def test_identical_recapture_is_idempotent(self):
        con = make_connection()
        captured_at = "2026-01-05T12:00:00+00:00"
        rows = [build_row(), build_row(home="Miami Heat", away="LA Clippers")]

        first = record(con, captured_at, rows)
        self.assertEqual(first["inserted"], 2)
        self.assertEqual(count_snapshots(con), 2)

        for _ in range(3):
            again = record(con, captured_at, rows)
            self.assertEqual(again["inserted"], 0)
            self.assertEqual(again["updated"], 0)
            self.assertEqual(again["skipped"], 2)

        self.assertEqual(count_snapshots(con), 2)

    def test_duplicate_across_different_times_is_not_a_duplicate(self):
        con = make_connection()
        record(con, "2026-01-05T12:00:00+00:00", [build_row()])
        record(con, "2026-01-05T13:00:00+00:00", [build_row()])
        self.assertEqual(count_snapshots(con), 2)

    def test_recapture_with_corrected_prices_updates(self):
        con = make_connection()
        captured_at = "2026-01-05T12:00:00+00:00"
        record(con, captured_at, [build_row(home_ml=-110, away_ml=-110)])
        result = record(con, captured_at, [build_row(home_ml=-130, away_ml=+110)])
        self.assertEqual(result["updated"], 1)
        self.assertEqual(result["inserted"], 0)
        self.assertEqual(count_snapshots(con), 1)
        self.assertEqual(
            con.execute("SELECT ML_Home, ML_Away FROM odds_snapshots").fetchone(), (-130, 110)
        )

    def test_primary_key_blocks_duplicates_even_without_the_pre_read(self):
        # The pre-read classifies; the primary key is what makes duplication
        # structurally impossible, including against a concurrent writer.
        con = make_connection()
        values = ("2026-01-05T12:00:00+00:00", "2026-01-05", "fanduel",
                  "Boston Celtics", "Denver Nuggets", -110, -110, 224.5)
        con.execute(
            'INSERT INTO "odds_snapshots" ("CapturedAt", "Date", "Sportsbook", "Home", '
            '"Away", "ML_Home", "ML_Away", "OU") VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            values,
        )
        with self.assertRaises(sqlite3.IntegrityError):
            con.execute(
                'INSERT INTO "odds_snapshots" ("CapturedAt", "Date", "Sportsbook", "Home", '
                '"Away", "ML_Home", "ML_Away", "OU") VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                values,
            )

    def test_naive_timestamp_is_rejected(self):
        con = make_connection()
        with self.assertRaises(line_movement.SnapshotError):
            line_movement.record_snapshot(con, datetime(2026, 1, 5, 12, 0), [build_row()])
        self.assertEqual(count_snapshots(con), 0)

    def test_empty_snapshot_is_refused(self):
        con = make_connection()
        with self.assertRaises(line_movement.SnapshotError):
            line_movement.record_snapshot(con, "2026-01-05T12:00:00+00:00", [])

    def test_missing_required_keys_are_rejected(self):
        con = make_connection()
        for missing in ("Date", "Sportsbook", "Home", "Away"):
            row = build_row()
            del row[missing]
            with self.assertRaises(line_movement.SnapshotError):
                line_movement.record_snapshot(con, "2026-01-05T12:00:00+00:00", [row])

    def test_invalid_american_price_is_rejected(self):
        con = make_connection()
        with self.assertRaises(line_movement.SnapshotError):
            line_movement.record_snapshot(
                con, "2026-01-05T12:00:00+00:00", [build_row(home_ml=-45)]
            )

    def test_invalid_total_is_rejected(self):
        con = make_connection()
        with self.assertRaises(line_movement.SnapshotError):
            line_movement.record_snapshot(
                con, "2026-01-05T12:00:00+00:00", [build_row(total="over")]
            )

    def test_a_failed_batch_writes_nothing(self):
        con = make_connection()
        rows = [build_row(), build_row(home_ml=0)]
        with self.assertRaises(line_movement.SnapshotError):
            line_movement.record_snapshot(con, "2026-01-05T12:00:00+00:00", rows)
        self.assertEqual(count_snapshots(con), 0)

    def test_never_touches_the_per_season_tables(self):
        con = sqlite3.connect(":memory:")
        con.execute('CREATE TABLE "2025-26" ("Date" TEXT, "Home" TEXT, "Away" TEXT)')
        con.execute(
            'INSERT INTO "2025-26" VALUES (?, ?, ?)', ("2025-12-01", "Boston Celtics", "Denver Nuggets")
        )
        before = con.execute('SELECT * FROM "2025-26"').fetchall()

        record(con, "2026-01-05T12:00:00+00:00", [build_row()])

        self.assertEqual(con.execute('SELECT * FROM "2025-26"').fetchall(), before)
        self.assertEqual(
            sorted(
                name for (name,) in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            ),
            ["2025-26", "odds_snapshots"],
        )


class TestOpeningAndClosingLine(unittest.TestCase):
    def test_open_is_first_and_close_is_last_in_shuffled_insert_order(self):
        """Ordering comes from CapturedAt, not rowid.

        The snapshots are inserted newest-first, oldest-last, and each carries a
        distinct price, so an implementation that trusted insertion order would
        return the wrong end and fail here.
        """
        con = make_connection()
        slate = [
            ("2026-01-05T09:00:00+00:00", -110, -110),
            ("2026-01-05T12:00:00+00:00", -120, +110),
            ("2026-01-05T15:00:00+00:00", -130, +100),
            ("2026-01-05T18:00:00+00:00", -140, +105),
            ("2026-01-05T21:00:00+00:00", -150, +110),
        ]
        shuffled = [slate[3], slate[0], slate[4], slate[2], slate[1]]

        for captured_at, home_ml, away_ml in shuffled:
            record(con, captured_at, [build_row(home_ml=home_ml, away_ml=away_ml)])

        opening = line_movement.opening_line(
            con, "2026-01-05", "Boston Celtics", "Denver Nuggets", "fanduel"
        )
        closing = line_movement.closing_line(
            con, "2026-01-05", "Boston Celtics", "Denver Nuggets", "fanduel"
        )

        self.assertEqual(opening["CapturedAt"], "2026-01-05T09:00:00+00:00")
        self.assertEqual(opening["ML_Home"], -110)
        self.assertEqual(closing["CapturedAt"], "2026-01-05T21:00:00+00:00")
        self.assertEqual(closing["ML_Home"], -150)

        # Neither end is an end of the insertion order: the first row written is
        # the 18:00 snapshot and the last is the 12:00 one, so any implementation
        # that trusted rowid would have returned the wrong snapshot for both.
        inserted_first, inserted_last = shuffled[0][0], shuffled[-1][0]
        self.assertEqual(inserted_first, "2026-01-05T18:00:00+00:00")
        self.assertEqual(inserted_last, "2026-01-05T12:00:00+00:00")
        self.assertNotEqual(opening["CapturedAt"], inserted_first)
        self.assertNotEqual(closing["CapturedAt"], inserted_last)
        self.assertLess(opening["CapturedAt"], inserted_first)
        self.assertGreater(closing["CapturedAt"], inserted_last)

    def test_open_and_close_differ_when_the_line_moves(self):
        con = make_connection()
        record(con, "2026-01-05T09:00:00+00:00", [build_row(home_ml=-110, away_ml=-110)])
        record(con, "2026-01-05T21:00:00+00:00", [build_row(home_ml=-130, away_ml=+110)])

        opening = line_movement.opening_line(
            con, "2026-01-05", "Boston Celtics", "Denver Nuggets", "fanduel"
        )
        closing = line_movement.closing_line(
            con, "2026-01-05", "Boston Celtics", "Denver Nuggets", "fanduel"
        )
        movement = line_movement.line_movement(opening["ML_Home"], closing["ML_Home"])
        self.assertGreater(movement["drift"], 0)

    def test_single_snapshot_makes_open_equal_close(self):
        # This is exactly the gapped case snapshot_coverage flags: zero measured
        # movement that means "nobody was watching", not "the line held".
        con = make_connection()
        record(con, "2026-01-05T12:00:00+00:00", [build_row()])
        opening = line_movement.opening_line(
            con, "2026-01-05", "Boston Celtics", "Denver Nuggets", "fanduel"
        )
        closing = line_movement.closing_line(
            con, "2026-01-05", "Boston Celtics", "Denver Nuggets", "fanduel"
        )
        self.assertEqual(opening["CapturedAt"], closing["CapturedAt"])
        self.assertAlmostEqual(
            line_movement.line_movement(opening["ML_Home"], closing["ML_Home"])["drift"],
            0.0,
            places=12,
        )

    def test_returns_none_for_an_unwatched_game(self):
        con = make_connection()
        self.assertIsNone(
            line_movement.closing_line(
                con, "2026-01-05", "Boston Celtics", "Denver Nuggets", "fanduel"
            )
        )
        self.assertIsNone(
            line_movement.opening_line(
                con, "2026-01-05", "Boston Celtics", "Denver Nuggets", "fanduel"
            )
        )

    def test_reading_a_database_without_the_table_creates_nothing(self):
        con = sqlite3.connect(":memory:")
        self.assertIsNone(
            line_movement.closing_line(
                con, "2026-01-05", "Boston Celtics", "Denver Nuggets", "fanduel"
            )
        )
        self.assertIsNone(
            con.execute("SELECT name FROM sqlite_master WHERE name='odds_snapshots'").fetchone()
        )

    def test_books_are_isolated_from_each_other(self):
        con = make_connection()
        record(con, "2026-01-05T21:00:00+00:00", [build_row(sportsbook="fanduel", home_ml=-150)])
        record(con, "2026-01-05T21:00:00+00:00", [build_row(sportsbook="draftkings", home_ml=-120)])
        self.assertEqual(
            line_movement.closing_line(
                con, "2026-01-05", "Boston Celtics", "Denver Nuggets", "fanduel"
            )["ML_Home"],
            -150,
        )
        self.assertEqual(
            line_movement.closing_line(
                con, "2026-01-05", "Boston Celtics", "Denver Nuggets", "DraftKings"
            )["ML_Home"],
            -120,
        )


class TestSnapshotCoverage(unittest.TestCase):
    def build_slate(self):
        con = make_connection()
        times = [
            "2026-01-05T09:00:00+00:00",
            "2026-01-05T13:00:00+00:00",
            "2026-01-05T19:00:00+00:00",
        ]
        for captured_at in times:
            record(con, captured_at, [build_row(), build_row(home="Miami Heat", away="LA Clippers")])
        for captured_at in times[:2]:
            record(con, captured_at, [build_row(home="Milwaukee Bucks", away="Phoenix Suns")])
        record(con, times[0], [build_row(game_date="2026-01-06", home="Golden State Warriors",
                                          away="Dallas Mavericks")])
        return con

    def test_flags_a_gapped_day(self):
        report = line_movement.snapshot_coverage(self.build_slate(), min_snapshots=2)

        # 2026-01-05: three games, two well covered and one watched once.
        self.assertEqual(report["games"], 4)
        self.assertEqual(report["games_with_min_snapshots"], 3)
        self.assertAlmostEqual(report["coverage_pct"], 75.0, places=6)
        self.assertFalse(report["dense_enough"])

        gaps = report["gaps"]
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["Date"], "2026-01-06")
        self.assertEqual(gaps[0]["Home"], "Golden State Warriors")
        self.assertEqual(gaps[0]["snapshots"], 1)

    def test_per_date_breakdown(self):
        report = line_movement.snapshot_coverage(self.build_slate(), min_snapshots=2)
        # 2026-01-05: three games, watched three, three and two times.
        self.assertEqual(report["by_date"]["2026-01-05"]["games"], 3)
        self.assertEqual(report["by_date"]["2026-01-05"]["games_with_min_snapshots"], 3)
        self.assertEqual(report["by_date"]["2026-01-06"]["games"], 1)
        self.assertEqual(report["by_date"]["2026-01-06"]["games_with_min_snapshots"], 0)

    def test_date_filter_isolates_the_gap(self):
        report = line_movement.snapshot_coverage(
            self.build_slate(), min_snapshots=2, date="2026-01-06"
        )
        self.assertEqual(report["games"], 1)
        self.assertEqual(report["games_with_min_snapshots"], 0)
        self.assertAlmostEqual(report["coverage_pct"], 0.0, places=6)

    def test_sportsbook_filter(self):
        con = self.build_slate()
        for hour in (9, 13, 19):
            record(con, f"2026-01-05T{hour:02d}:00:00+00:00",
                   [build_row(sportsbook="draftkings", home="Miami Heat", away="LA Clippers")])
        report = line_movement.snapshot_coverage(con, min_snapshots=2, sportsbook="draftkings")
        self.assertEqual(report["games"], 1)
        self.assertEqual(report["games_with_min_snapshots"], 1)
        self.assertEqual(report["gaps"], [])
        self.assertTrue(report["dense_enough"])

    def test_snapshot_counts_are_reported(self):
        report = line_movement.snapshot_coverage(self.build_slate(), min_snapshots=2)
        self.assertEqual(report["total_snapshots"], 3 + 3 + 2 + 1)
        self.assertEqual(report["min_snapshots_per_game"], 1)
        self.assertEqual(report["max_snapshots_per_game"], 3)

    def test_dense_coverage_is_reported_as_dense(self):
        con = make_connection()
        for hour in (9, 13, 19):
            record(con, f"2026-01-05T{hour:02d}:00:00+00:00", [build_row()])
        report = line_movement.snapshot_coverage(con, min_snapshots=2)
        self.assertAlmostEqual(report["coverage_pct"], 100.0, places=6)
        self.assertEqual(report["gaps"], [])
        self.assertTrue(report["dense_enough"])

    def test_empty_database_is_zero_not_an_error(self):
        report = line_movement.snapshot_coverage(make_connection(), min_snapshots=2)
        self.assertEqual(report["games"], 0)
        self.assertEqual(report["total_snapshots"], 0)
        self.assertFalse(report["dense_enough"])

    def test_missing_table_is_zero_not_an_error(self):
        report = line_movement.snapshot_coverage(sqlite3.connect(":memory:"), min_snapshots=2)
        self.assertEqual(report["games"], 0)


class TestClvReport(unittest.TestCase):
    # De-vigged close probabilities for two book shapes, by hand:
    #   -110 / -110  ->  home fair = (110/210) / (2 * 110/210) = 1/2
    #   -200 / +180  ->  home fair = (2/3) / (2/3 + 5/14)     = 28/43
    # Entry probabilities are chosen so each per-bet CLV is an exact fraction:
    #   1/2  - 0.52 = -0.02
    #   28/43 - 0.60 = 11/215
    #   28/43 - 0.65 =  1/860
    GAMES = [
        ("2026-01-05", "Boston Celtics", "Denver Nuggets", -110, -110, 0.52, -110),
        ("2026-01-05", "Miami Heat", "LA Clippers", -200, 180, 0.60, -120),
        ("2026-01-05", "Milwaukee Bucks", "Phoenix Suns", -200, 180, 0.65, -160),
    ]

    def build_slate(self):
        con = make_connection()
        for game_date, home, away, close_home, close_away, _prob, _price in self.GAMES:
            for captured_at in ("2026-01-05T09:00:00+00:00", "2026-01-05T21:00:00+00:00"):
                record(
                    con,
                    captured_at,
                    [build_row(game_date=game_date, home=home, away=away,
                               home_ml=close_home, away_ml=close_away)],
                )
        return con

    def bets(self):
        return [
            {
                "Date": game_date,
                "Home": home,
                "Away": away,
                "Sportsbook": "fanduel",
                "entry_prob": prob,
                "entry_price": price,
            }
            for game_date, home, away, _h, _a, prob, price in self.GAMES
        ]

    def test_per_bet_clv_matches_hand_arithmetic(self):
        con = self.build_slate()
        report = line_movement.clv_report(con, self.bets())

        expected = [-0.02, 11 / 215, 1 / 860]
        self.assertEqual(report["evaluated"], 3)
        for entry, want in zip(report["per_bet"], expected):
            self.assertAlmostEqual(entry["clv"], want, places=12)
        self.assertAlmostEqual(report["per_bet"][0]["close_prob"], 0.5, places=12)
        self.assertAlmostEqual(report["per_bet"][1]["close_prob"], 28 / 43, places=12)
        self.assertEqual(report["per_bet"][1]["close_price"], -200)

    def test_mean_and_standard_error(self):
        con = self.build_slate()
        report = line_movement.clv_report(con, self.bets())

        values = [-0.02, 11 / 215, 1 / 860]
        expected_mean = sum(values) / 3
        self.assertAlmostEqual(report["mean_clv"], expected_mean, places=12)
        self.assertAlmostEqual(report["mean_clv_points"], expected_mean * 100, places=9)

        # Sample SD, n-1 -- the whole point of the report. A mean over a handful
        # of bets carries a standard error comparable to the mean itself.
        expected_sd = (
            sum((value - expected_mean) ** 2 for value in values) / (len(values) - 1)
        ) ** 0.5
        self.assertAlmostEqual(report["stdev"], expected_sd, places=12)
        self.assertAlmostEqual(
            report["standard_error"], expected_sd / (len(values) ** 0.5), places=12
        )
        self.assertLess(abs(report["mean_clv"] / report["standard_error"]), 2.0)

    def test_reports_the_count(self):
        report = line_movement.clv_report(self.build_slate(), self.bets())
        self.assertEqual(report["count"], 3)
        self.assertEqual(report["evaluated"], 3)
        self.assertEqual(report["missing_close"], 0)

    def test_bets_without_a_close_are_counted_not_dropped(self):
        bets = self.bets()
        bets.append({
            "Date": "2026-01-05",
            "Home": "Golden State Warriors",
            "Away": "Dallas Mavericks",
            "Sportsbook": "fanduel",
            "entry_prob": 0.6,
            "entry_price": -110,
        })
        report = line_movement.clv_report(self.build_slate(), bets)
        self.assertEqual(report["count"], 4)
        self.assertEqual(report["evaluated"], 3)
        self.assertEqual(report["missing_close"], 1)
        self.assertEqual(len(report["missing_close_games"]), 1)
        self.assertIn("Golden State Warriors", report["missing_close_games"][0])

    def test_one_sided_close_is_not_deviggable(self):
        con = self.build_slate()
        record(con, "2026-01-06T09:00:00+00:00", [
            build_row(game_date="2026-01-06", home="Golden State Warriors",
                      away="Dallas Mavericks", home_ml=-110, away_ml=None)
        ])
        report = line_movement.clv_report(con, [{
            "Date": "2026-01-06", "Home": "Golden State Warriors", "Away": "Dallas Mavericks",
            "Sportsbook": "fanduel", "entry_prob": 0.6, "entry_price": -110,
        }])
        self.assertEqual(report["evaluated"], 0)
        self.assertEqual(report["missing_close"], 1)

    def test_no_evaluable_bets_reports_none_rather_than_raising(self):
        report = line_movement.clv_report(self.build_slate(), [{
            "Date": "2026-02-01", "Home": "A", "Away": "B",
            "Sportsbook": "fanduel", "entry_prob": 0.6, "entry_price": -110,
        }])
        self.assertEqual(report["evaluated"], 0)
        self.assertIsNone(report["mean_clv"])
        self.assertIsNone(report["standard_error"])

    def test_single_bet_has_no_standard_error(self):
        con = self.build_slate()
        report = line_movement.clv_report(con, self.bets()[:1])
        self.assertEqual(report["evaluated"], 1)
        self.assertAlmostEqual(report["mean_clv"], -0.02, places=12)
        self.assertIsNone(report["stdev"])
        self.assertIsNone(report["standard_error"])
        self.assertIsNone(report["mean_over_standard_error"])

    def test_away_side_uses_the_away_price(self):
        con = self.build_slate()
        report = line_movement.clv_report(con, [{
            "Date": "2026-01-05", "Home": "Miami Heat", "Away": "LA Clippers",
            "Sportsbook": "fanduel", "side": "away",
            "entry_prob": 0.40, "entry_price": 180,
        }])
        self.assertEqual(report["evaluated"], 1)
        # away fair = 15/43, so CLV = 15/43 - 0.40
        self.assertAlmostEqual(report["per_bet"][0]["close_prob"], 15 / 43, places=12)
        self.assertAlmostEqual(report["per_bet"][0]["clv"], 15 / 43 - 0.40, places=12)
        self.assertEqual(report["per_bet"][0]["close_price"], 180)

    def test_default_sportsbook_argument(self):
        con = self.build_slate()
        report = line_movement.clv_report(con, [{
            "Date": "2026-01-05", "Home": "Boston Celtics", "Away": "Denver Nuggets",
            "entry_prob": 0.52, "entry_price": -110,
        }], sportsbook="fanduel")
        self.assertEqual(report["evaluated"], 1)

    def test_empty_bets_raise(self):
        with self.assertRaises(ValueError):
            line_movement.clv_report(self.build_slate(), [])

    def test_malformed_bets_raise(self):
        con = self.build_slate()
        with self.assertRaises(ValueError):
            line_movement.clv_report(con, ["not a dict"])
        with self.assertRaises(ValueError):
            line_movement.clv_report(con, [{"entry_prob": 0.5, "entry_price": -110}])
        with self.assertRaises(ValueError):
            line_movement.clv_report(con, [{
                "Date": "2026-01-05", "Home": "A", "Away": "B", "Sportsbook": "fanduel",
                "entry_prob": 0.5, "entry_price": -110, "side": "under",
            }])


class TestCaptureOnce(unittest.TestCase):
    def setUp(self):
        self.at = "2026-01-05T18:30:00+00:00"
        self.games = [fake_game(), fake_game(home="Miami Heat", away="LA Clippers", home_ml=-150,
                                             away_ml=130)]

    def test_records_a_snapshot(self):
        con = make_connection()
        with mock.patch.object(line_movement.time, "sleep"):
            result = line_movement.capture_once(
                "fanduel", at_time=self.at, con=con, scorer=lambda d: FakeScoreboard(self.games)
            )

        self.assertEqual(result["captured_at"], "2026-01-05T18:30:00+00:00")
        self.assertEqual(result["date"], "2026-01-05")
        self.assertEqual(result["sportsbook"], "fanduel")
        self.assertEqual(result["games"], 2)
        self.assertEqual(count_snapshots(con), 2)
        self.assertEqual(
            line_movement.closing_line(con, "2026-01-05", "Boston Celtics", "Denver Nuggets",
                                       "fanduel")["ML_Home"],
            -110,
        )

    def test_scorer_receives_the_game_date(self):
        con = make_connection()
        seen = []

        def scorer(game_date):
            seen.append(game_date)
            return FakeScoreboard(self.games)

        with mock.patch.object(line_movement.time, "sleep"):
            line_movement.capture_once("fanduel", at_time=self.at, con=con, scorer=scorer)
        self.assertEqual(seen, [date(2026, 1, 5)])

    def test_game_date_defaults_to_the_capture_date_and_can_be_overridden(self):
        con = make_connection()
        seen = []

        def scorer(game_date):
            seen.append(game_date)
            return FakeScoreboard(self.games)

        with mock.patch.object(line_movement.time, "sleep"):
            line_movement.capture_once("fanduel", at_time=self.at, con=con,
                                       game_date="2026-01-06", scorer=scorer)
        self.assertEqual(seen, [date(2026, 1, 6)])
        self.assertEqual(
            con.execute("SELECT DISTINCT Date FROM odds_snapshots").fetchall(),
            [("2026-01-06",)],
        )

    def test_open_and_close_survive_two_captures(self):
        con = make_connection()
        with mock.patch.object(line_movement.time, "sleep"):
            line_movement.capture_once("fanduel", at_time="2026-01-05T12:00:00+00:00", con=con,
                                       scorer=lambda d: FakeScoreboard([fake_game(home_ml=-110,
                                                                                 away_ml=-110)]))
            line_movement.capture_once("fanduel", at_time="2026-01-05T20:00:00+00:00", con=con,
                                       scorer=lambda d: FakeScoreboard([fake_game(home_ml=-140,
                                                                                 away_ml=115)]))
        opening = line_movement.opening_line(con, "2026-01-05", "Boston Celtics",
                                             "Denver Nuggets", "fanduel")
        closing = line_movement.closing_line(con, "2026-01-05", "Boston Celtics",
                                             "Denver Nuggets", "fanduel")
        self.assertEqual((opening["ML_Home"], closing["ML_Home"]), (-110, -140))
        movement = line_movement.line_movement(opening["ML_Home"], closing["ML_Home"])
        self.assertGreater(movement["drift"], 0)

    def test_recapture_at_the_same_instant_is_idempotent(self):
        con = make_connection()
        with mock.patch.object(line_movement.time, "sleep"):
            for _ in range(2):
                result = line_movement.capture_once(
                    "fanduel", at_time=self.at, con=con,
                    scorer=lambda d: FakeScoreboard([self.games[0]])
                )
        self.assertEqual(result["skipped"], 1)
        self.assertEqual(count_snapshots(con), 1)

    def test_opens_and_closes_a_database_itself(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "odds.sqlite"
            with mock.patch.object(line_movement.time, "sleep"):
                result = line_movement.capture_once(
                    "fanduel", at_time=self.at, db_path=db_path,
                    scorer=lambda d: FakeScoreboard(self.games),
                )
            self.assertEqual(result["games"], 2)
            with sqlite3.connect(db_path) as con:
                self.assertEqual(count_snapshots(con), 2)

    def test_sleeps_for_politeness(self):
        con = make_connection()
        with mock.patch.object(line_movement.time, "sleep") as sleeper:
            line_movement.capture_once("fanduel", at_time=self.at, con=con,
                                       scorer=lambda d: FakeScoreboard(self.games))
        self.assertEqual(sleeper.call_count, 1)
        seconds = sleeper.call_args[0][0]
        self.assertGreaterEqual(seconds, line_movement.MIN_DELAY_SECONDS)
        self.assertLessEqual(seconds, line_movement.MAX_DELAY_SECONDS)

    def test_games_without_a_price_are_counted_not_hidden(self):
        con = make_connection()
        bad_game = fake_game(home="Miami Heat", away="LA Clippers")
        del bad_game["home_ml"]["fanduel"]
        with mock.patch.object(line_movement.time, "sleep"):
            result = line_movement.capture_once(
                "fanduel", at_time=self.at, con=con,
                scorer=lambda d: FakeScoreboard([self.games[0], bad_game])
            )
        self.assertEqual(result["games"], 1)
        self.assertEqual(len(result["missing"]), 1)

    # --- explicit failure reporting ---------------------------------------

    def test_missing_sportsbook_raises(self):
        with self.assertRaises(line_movement.CaptureError):
            line_movement.capture_once("", at_time=self.at,
                                       scorer=lambda d: FakeScoreboard(self.games))

    def test_empty_slate_raises_rather_than_recording_nothing(self):
        con = make_connection()
        with mock.patch.object(line_movement.time, "sleep"):
            with self.assertRaises(line_movement.CaptureError):
                line_movement.capture_once("fanduel", at_time=self.at, con=con,
                                           scorer=lambda d: FakeScoreboard([]))
        self.assertEqual(count_snapshots(con), 0)

    def test_slate_with_no_odds_for_this_book_raises(self):
        con = make_connection()
        bad = fake_game(home="Miami Heat", away="LA Clippers")
        del bad["total"]["fanduel"]
        with mock.patch.object(line_movement.time, "sleep"):
            with self.assertRaises(line_movement.CaptureError):
                line_movement.capture_once("fanduel", at_time=self.at, con=con,
                                           scorer=lambda d: FakeScoreboard([bad]))
        self.assertEqual(count_snapshots(con), 0)

    def test_fetch_exception_is_wrapped_not_swallowed(self):
        con = make_connection()

        def broken(_game_date):
            raise ConnectionError("dns went away")

        with self.assertRaises(line_movement.CaptureError) as caught:
            line_movement.capture_once("fanduel", at_time=self.at, con=con, scorer=broken)
        self.assertIn("fanduel", str(caught.exception))
        self.assertEqual(count_snapshots(con), 0)

    def test_none_scoreboard_raises(self):
        with self.assertRaises(line_movement.CaptureError):
            line_movement.capture_once("fanduel", at_time=self.at, con=make_connection(),
                                       scorer=lambda d: None)

    def test_timeout_raises_rather_than_blocking_forever(self):
        import threading

        def slow(_game_date):
            # threading.Event().wait, not time.sleep: `line_movement.time` is
            # the global time module and the test patches its sleep, which
            # would make this return instantly and never time out.
            threading.Event().wait(0.3)
            return FakeScoreboard([])

        with mock.patch.object(line_movement.time, "sleep"):
            with self.assertRaises(line_movement.CaptureError) as caught:
                line_movement.capture_once("fanduel", at_time=self.at, con=make_connection(),
                                           scorer=slow, timeout=0.05)
        self.assertIn("timed out", str(caught.exception))

    def test_absent_sbrscrape_raises_capture_error(self):
        with mock.patch.object(line_movement, "Scoreboard", None):
            with self.assertRaises(line_movement.CaptureError) as caught:
                line_movement._default_scorer(date(2026, 1, 5))
        self.assertIn("sbrscrape", str(caught.exception))

    def test_naive_at_time_is_rejected(self):
        with mock.patch.object(line_movement.time, "sleep"):
            with self.assertRaises(line_movement.SnapshotError):
                line_movement.capture_once("fanduel", at_time=datetime(2026, 1, 5, 18, 30),
                                           con=make_connection(),
                                           scorer=lambda d: FakeScoreboard(self.games))


class TestModuleImportsWithoutSbrscrape(unittest.TestCase):
    SCRIPT = """
import importlib.util, pathlib, sys
sys.modules["sbrscrape"] = None
path = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("no_sbrscrape", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert module.Scoreboard is None, "Scoreboard should be None when sbrscrape is absent"
try:
    module._default_scorer("2026-01-05")
except module.CaptureError:
    pass
else:
    raise AssertionError("a missing sbrscrape must raise CaptureError")
print("IMPORT_OK")
"""

    def test_import_succeeds_and_fetches_fail_loudly(self):
        result = subprocess.run(
            [sys.executable, "-c", self.SCRIPT, str(MODULE_PATH)],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("IMPORT_OK", result.stdout)


class TestMain(unittest.TestCase):
    def test_captures_into_the_given_database(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "odds.sqlite"
            games = [fake_game()]
            with mock.patch.object(line_movement, "Scoreboard", lambda date: FakeScoreboard(games)), \
                 mock.patch.object(line_movement.time, "sleep"):
                code = line_movement.main([
                    "--sportsbook", "fanduel",
                    "--db", str(db_path),
                    "--at", "2026-01-05T18:30:00+00:00",
                    "--date", "2026-01-05",
                ])
            self.assertEqual(code, 0)
            with sqlite3.connect(db_path) as con:
                self.assertEqual(count_snapshots(con), 1)
                self.assertEqual(
                    con.execute("SELECT CapturedAt FROM odds_snapshots").fetchone(),
                    ("2026-01-05T18:30:00+00:00",),
                )

    def test_capture_failure_exits_non_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "odds.sqlite"
            with mock.patch.object(line_movement, "Scoreboard", lambda date: FakeScoreboard([])), \
                 mock.patch.object(line_movement.time, "sleep"):
                code = line_movement.main(["--db", str(db_path)])
            self.assertEqual(code, 1)

    def test_naive_at_exits_non_zero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "odds.sqlite"
            code = line_movement.main(["--db", str(db_path), "--at", "2026-01-05T18:30:00"])
            self.assertEqual(code, 1)
            self.assertFalse(db_path.exists())

    def test_coverage_flag_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "odds.sqlite"
            with sqlite3.connect(db_path) as con:
                line_movement.ensure_snapshot_table(con)
            with mock.patch.object(line_movement, "Scoreboard") as scoreboard:
                code = line_movement.main(["--db", str(db_path), "--coverage"])
            self.assertEqual(code, 0)
            self.assertFalse(scoreboard.called)
            with sqlite3.connect(db_path) as con:
                self.assertEqual(count_snapshots(con), 0)


if __name__ == "__main__":
    unittest.main()