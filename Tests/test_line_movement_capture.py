"""Closing-line value needs a capture schedule, not just the arithmetic.

The CLV functions in Get_Odds_Line_Movement.py are correct and have been for a
while. What makes CLV *measurable* is a snapshot TRAJECTORY: one scrape per game
is indistinguishable from a line that never moved, and any CLV computed over
thin coverage is biased toward zero.

These tests pin the two things that make the trajectory accumulate -- multi-book
capture and a way to schedule it -- and pin the refusal to report CLV before
coverage is known. That refusal is the point: the arithmetic being right is not
the same as the measurement being meaningful.

No network access. capture_once is replaced with a recorder.
"""
import importlib.util
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT, ROOT / "src" / "Process-Data"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


def _load():
    spec = importlib.util.spec_from_file_location(
        "glm", ROOT / "src" / "Process-Data" / "Get_Odds_Line_Movement.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


glm = _load()


class TestSportsbookParsing(unittest.TestCase):
    def test_single_book(self):
        self.assertEqual(glm.parse_sportsbooks("fanduel"), ["fanduel"])

    def test_comma_separated_rotation(self):
        self.assertEqual(
            glm.parse_sportsbooks("fanduel,draftkings,betmgm"),
            ["fanduel", "draftkings", "betmgm"],
        )

    def test_all_expands_to_the_known_set_and_is_at_least_two_books(self):
        books = glm.parse_sportsbooks("all")
        self.assertEqual(books, list(glm.KNOWN_SPORTSBOOKS))
        self.assertGreaterEqual(
            len(books), 2,
            "rotation needs at least two books: one book's line is not the market's",
        )

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(
            glm.parse_sportsbooks("  FanDuel , DRAFTKINGS "), ["fanduel", "draftkings"]
        )

    def test_duplicates_collapse_so_a_book_is_not_double_captured(self):
        self.assertEqual(
            glm.parse_sportsbooks("fanduel,fanduel,FANDUEL"), ["fanduel"]
        )

    def test_order_is_preserved_so_a_rotation_is_reproducible(self):
        self.assertEqual(
            glm.parse_sportsbooks("pointsbet,fanduel,wynn"),
            ["pointsbet", "fanduel", "wynn"],
        )

    def test_empty_spec_falls_back_to_the_default_book(self):
        self.assertEqual(glm.parse_sportsbooks(""), [glm.DEFAULT_SPORTSBOOK])


class TestScheduledCaptureExists(unittest.TestCase):
    """Without a schedule the snapshot table stays empty and CLV stays unanswerable."""

    def test_ships_a_systemd_timer_and_service(self):
        contrib = ROOT / "contrib"
        self.assertTrue((contrib / "odds-capture.timer").exists(),
                        "no timer ships, so nothing accumulates")
        self.assertTrue((contrib / "odds-capture.service").exists())

    def test_timer_repeats_rather_than_running_once(self):
        timer = (ROOT / "contrib" / "odds-capture.timer").read_text()
        self.assertIn("OnUnitActiveSec", timer)
        self.assertIn("[Install]", timer)

    def test_service_does_not_mask_failure(self):
        """A capture job that exits 0 while capturing nothing is worse than a broken one."""
        service = (ROOT / "contrib" / "odds-capture.service").read_text()
        self.assertIn("SuccessExitStatus=0", service)
        self.assertNotIn(
            "SuccessExitStatus=0 1", service,
            "the service must not declare a failing capture successful",
        )

    def test_watch_mode_exists_as_a_scheduler_free_alternative(self):
        src = (ROOT / "src" / "Process-Data" / "Get_Odds_Line_Movement.py").read_text()
        self.assertIn('"--watch"', src)
        self.assertIn('"--every"', src)


class TestWatchModeBehaviour(unittest.TestCase):
    """--watch must survive one book failing, and must not exit 0 while broken."""

    def _run_watch(self, books, failures=(), max_passes=1):
        calls = []

        def fake_capture_once(book, at_time=None, game_date=None, con=None, timeout=None):
            # Stop once the requested number of whole passes has completed, by
            # unwinding at the START of the next pass the way Ctrl-C would. Raising
            # from inside the pass instead would abort before `failures` is assigned,
            # which would test the unwind path rather than the exit-code contract.
            if len(calls) >= max_passes * len(books):
                raise KeyboardInterrupt()
            calls.append(book)
            if book in failures:
                raise glm.LineMovementError(f"synthetic failure for {book}")

        real_capture = glm.capture_once
        real_argv = sys.argv
        glm.capture_once = fake_capture_once
        sys.argv = ["glm", "--watch", "--every", "1", "--sportsbook", ",".join(books)]
        try:
            code = glm.main()
        finally:
            glm.capture_once = real_capture
            sys.argv = real_argv
        return code, calls

    def test_watch_captures_every_book_each_pass(self):
        books = ["fanduel", "draftkings"]
        code, calls = self._run_watch(books, max_passes=1)
        self.assertEqual(calls[:2], books,
                         "a pass must cover every configured book")

    def test_one_book_failing_does_not_stop_the_others(self):
        books = ["fanduel", "draftkings"]
        _, calls = self._run_watch(books, failures={"fanduel"}, max_passes=1)
        self.assertIn("draftkings", calls[:2],
                      "a failing book must not prevent the others being captured")

    def test_exit_code_is_non_zero_when_the_final_pass_failed(self):
        code, _ = self._run_watch(["fanduel", "draftkings"], failures={"fanduel"})
        self.assertEqual(code, 1,
                         "a still-broken capture must not exit 0 at shutdown")

    def test_exit_code_is_zero_when_every_book_succeeded(self):
        code, _ = self._run_watch(["fanduel", "draftkings"], max_passes=1)
        self.assertEqual(code, 0)

    def test_non_positive_every_is_rejected(self):
        real_argv = sys.argv
        sys.argv = ["glm", "--watch", "--every", "0", "--sportsbook", "fanduel"]
        try:
            self.assertEqual(glm.main(), 1)
        finally:
            sys.argv = real_argv


class TestCoverageGatesAnyClvClaim(unittest.TestCase):
    def test_snapshot_coverage_exists_and_is_reported(self):
        self.assertTrue(callable(glm.snapshot_coverage))

    def test_min_snapshot_default_is_at_least_two(self):
        """One snapshot cannot be distinguished from a line that never moved."""
        self.assertGreaterEqual(
            glm.DEFAULT_MIN_SNAPSHOTS, 2,
            "a single snapshot is not a trajectory; CLV over it is biased to zero",
        )

    def test_coverage_is_reported_before_capture_in_the_cli(self):
        src = (ROOT / "src" / "Process-Data" / "Get_Odds_Line_Movement.py").read_text()
        self.assertIn("snapshot_coverage(con, min_snapshots=args.min_snapshots)", src)


if __name__ == "__main__":
    unittest.main()