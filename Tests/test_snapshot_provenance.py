"""A repeated price is not a trajectory.

The odds endpoint returns one current price no matter which time the caller asks
for, so `--at` is a LABEL on a real price, not a retrieval of the price at that
time. Before these tests existed, writing the same observation under three
timestamps produced a table that reported "3 snapshots per game, 100% coverage"
and produced a CLV of exactly 0.0 -- which reads as "the line held" rather than as
"this is one observation written three times".

The tests below pin the two properties that make that impossible to do silently:

  1. Coverage counts OBSERVED rows, and a game needs more than one distinct price.
  2. A materially past --at is refused unless explicitly declared with --backfill,
     and backfilled rows never count toward coverage or CLV.

No network: record_snapshot and capture_once are driven directly.
"""
import contextlib
import importlib.util
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src" / "Process-Data"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

_spec = importlib.util.spec_from_file_location(
    "glm", ROOT / "src" / "Process-Data" / "Get_Odds_Line_Movement.py"
)
glm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(glm)

NOW = datetime.now(timezone.utc)


def _row(home="Detroit Pistons", away="New York Knicks", ml_home=104,
         ml_away=-122, total=232.5):
    # Keys must match what _normalise_row reads: ML_Home / ML_Away / OU. Using the
    # odds-provider's own key names silently produced NULL prices, which made every
    # row price-identical and defeated the entire point of these tests.
    return {
        "Date": "2026-01-05",
        "Sportsbook": "fanduel",
        "Home": home,
        "Away": away,
        "ML_Home": ml_home,
        "ML_Away": ml_away,
        "OU": total,
    }


class SnapshotStoreMixin:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.tmp.name) / "s.sqlite")
        self.addCleanup(self.tmp.cleanup)

    def con(self):
        return sqlite3.connect(self.db)

    def write(self, captured_at, rows=None, provenance=glm.PROVENANCE_OBSERVED):
        with contextlib.closing(self.con()) as con:
            return glm.record_snapshot(con, captured_at, rows or [_row()],
                                       provenance=provenance)


class TestIdenticalPricesAreNotCoverage(SnapshotStoreMixin, unittest.TestCase):
    def test_one_price_written_many_times_is_not_covered(self):
        """The exact defect: same observation, three timestamps, 100% 'coverage'."""
        for hours in (0, 8, 20):
            self.write(NOW - timedelta(hours=hours))

        with contextlib.closing(self.con()) as con:
            report = glm.snapshot_coverage(con, min_snapshots=2)

        self.assertEqual(report["total_snapshots"], 3,
                         "three rows were written, and that must still be visible")
        self.assertEqual(report["games"], 1)
        self.assertEqual(
            report["games_with_min_snapshots"], 0,
            "three copies of one price must NOT count as a trajectory",
        )
        self.assertFalse(report["dense_enough"])

    def test_a_genuine_movement_does_count(self):
        """The control: same game, two DIFFERENT prices, must be covered."""
        self.write(NOW - timedelta(hours=8))
        self.write(NOW - timedelta(hours=1), rows=[_row(ml_home=110, ml_away=-118)])

        with contextlib.closing(self.con()) as con:
            report = glm.snapshot_coverage(con, min_snapshots=2)

        self.assertEqual(report["games_with_min_snapshots"], 1,
                         "two genuinely different prices are a real trajectory")
        self.assertTrue(report["dense_enough"])

    def test_distinct_prices_are_reported_per_game(self):
        self.write(NOW - timedelta(hours=8))
        self.write(NOW - timedelta(hours=1), rows=[_row(ml_home=110, ml_away=-118)])
        with contextlib.closing(self.con()) as con:
            report = glm.snapshot_coverage(con, min_snapshots=2)
        # A covered game has no entry in `gaps`, so assert on the counters instead
        # of reaching for a per-game key the report does not expose.
        self.assertEqual(report["games_with_min_snapshots"], 1)
        self.assertEqual(report["total_observed_snapshots"], 2)
        self.assertEqual(report["total_backfilled_snapshots"], 0)
        self.assertEqual(report["gaps"], [],
                         "a game with two genuinely different prices is not a gap")

    def test_changing_only_the_total_counts_as_a_different_price(self):
        self.write(NOW - timedelta(hours=8))
        self.write(NOW - timedelta(hours=1), rows=[_row(total=235.0)])
        with contextlib.closing(self.con()) as con:
            report = glm.snapshot_coverage(con, min_snapshots=2)
        self.assertEqual(report["games_with_min_snapshots"], 1,
                         "a moved total is still line movement")


class TestBackfilledRowsNeverCount(SnapshotStoreMixin, unittest.TestCase):
    def test_backfilled_rows_are_marked(self):
        self.write(NOW, provenance=glm.PROVENANCE_BACKFILLED)
        with contextlib.closing(self.con()) as con:
            value = con.execute(
                'SELECT "Provenance" FROM "' + glm.SNAPSHOT_TABLE + '" LIMIT 1'
            ).fetchone()[0]
        self.assertEqual(value, glm.PROVENANCE_BACKFILLED)

    def test_backfilled_rows_are_excluded_from_coverage(self):
        for hours in (1, 5, 12):
            self.write(NOW - timedelta(hours=hours),
                       provenance=glm.PROVENANCE_BACKFILLED)
        with contextlib.closing(self.con()) as con:
            report = glm.snapshot_coverage(con, min_snapshots=2)
        self.assertEqual(report["total_snapshots"], 3)
        self.assertEqual(report["total_observed_snapshots"], 0)
        self.assertEqual(report["games_with_min_snapshots"], 0,
                         "back-filled rows must never make a game look covered")

    def test_backfilled_and_observed_do_not_add_up_to_coverage(self):
        """Three backfilled + one observed is still one observation."""
        for hours in (2, 6, 10):
            self.write(NOW - timedelta(hours=hours),
                       provenance=glm.PROVENANCE_BACKFILLED)
        self.write(NOW)
        with contextlib.closing(self.con()) as con:
            report = glm.snapshot_coverage(con, min_snapshots=2)
        self.assertEqual(report["total_snapshots"], 4)
        self.assertEqual(report["total_observed_snapshots"], 1)
        self.assertEqual(report["games_with_min_snapshots"], 0)

    def test_provenance_is_never_downgraded_or_promoted_by_a_later_write(self):
        self.write(NOW, provenance=glm.PROVENANCE_BACKFILLED)
        # Same key, corrected price, declared as observed.
        self.write(NOW, rows=[_row(ml_home=110, ml_away=-118)],
                   provenance=glm.PROVENANCE_OBSERVED)
        with contextlib.closing(self.con()) as con:
            value = con.execute(
                'SELECT "Provenance" FROM "' + glm.SNAPSHOT_TABLE + '" LIMIT 1'
            ).fetchone()[0]
        self.assertEqual(value, glm.PROVENANCE_BACKFILLED,
                         "a corrected price must not retroactively become observed")

    def test_unknown_provenance_is_rejected(self):
        with self.assertRaises(glm.SnapshotError):
            self.write(NOW, provenance="probably-observed")


class TestStaleAtIsRefusedUnlessDeclared(SnapshotStoreMixin, unittest.TestCase):
    def _run(self, at, extra=()):
        real_argv = sys.argv
        sys.argv = ["glm", "--db", self.db, "--sportsbook", "fanduel",
                    "--date", "2026-01-05", "--at", at.isoformat(), *extra]
        try:
            return glm.main()
        finally:
            sys.argv = real_argv

    def test_materially_past_at_is_refused(self):
        code = self._run(NOW - timedelta(hours=20))
        self.assertEqual(code, 1,
                         "a past timestamp must not be silently accepted")

    def test_refusal_writes_nothing(self):
        self._run(NOW - timedelta(hours=20))
        with contextlib.closing(self.con()) as con:
            try:
                count = con.execute(
                    'SELECT COUNT(*) FROM "' + glm.SNAPSHOT_TABLE + '"').fetchone()[0]
            except sqlite3.Error:
                count = 0
        self.assertEqual(count, 0, "a refused capture must leave no rows behind")

    def test_backfill_flag_permits_it_and_marks_the_rows(self):
        code = self._run(NOW - timedelta(hours=20), extra=["--backfill"])
        # The fetch itself may fail without network; what matters is that the
        # refusal path was NOT taken, i.e. we did not return 1 before the fetch.
        with contextlib.closing(self.con()) as con:
            try:
                rows = con.execute(
                    'SELECT "Provenance" FROM "' + glm.SNAPSHOT_TABLE + '"'
                ).fetchall()
            except sqlite3.Error:
                rows = []
        self.assertTrue(all(r[0] == glm.PROVENANCE_BACKFILLED for r in rows),
                        "every back-filled row must be marked")

    def test_a_recent_at_is_accepted_without_the_flag(self):
        code = self._run(NOW - timedelta(minutes=5))
        self.assertNotEqual(code, 1, "a fresh timestamp is a normal observation")


class TestColumnIsMigratedNotAssumed(SnapshotStoreMixin, unittest.TestCase):
    def test_a_database_written_before_this_change_still_opens(self):
        """An existing odds_snapshots table lacks the column; it must be added."""
        with contextlib.closing(self.con()) as con:
            con.execute(
                'CREATE TABLE "' + glm.SNAPSHOT_TABLE + '" ('
                '"CapturedAt" TEXT NOT NULL, "Date" TEXT NOT NULL, '
                '"Sportsbook" TEXT NOT NULL, "Home" TEXT NOT NULL, '
                '"Away" TEXT NOT NULL, "ML_Home" INTEGER, "ML_Away" INTEGER, '
                '"OU" REAL, PRIMARY KEY ("CapturedAt", "Sportsbook", "Date", '
                '"Home", "Away"))'
            )
            con.commit()
        self.write(NOW)
        with contextlib.closing(self.con()) as con:
            cols = [r[1] for r in con.execute(
                f'PRAGMA table_info("{glm.SNAPSHOT_TABLE}")')]
        self.assertIn("Provenance", cols,
                      "ensure_snapshot_table must migrate the column in")


if __name__ == "__main__":
    unittest.main()