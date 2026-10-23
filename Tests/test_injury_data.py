"""Injury reports: the URL is the timestamp, so point-in-time is answerable.

The NBA publishes an official injury report PDF at an immutable CDN URL that
carries its own publication time. That is what makes "what was known before
tip-off" a question with a real answer rather than a reconstruction: the
snapshot whose label predates the cutoff IS the knowledge of that moment.

These tests pin the four properties the pre-game availability block depends
on, each one because getting it wrong is silent:

  1. URL construction speaks all three filename eras and is DST-safe. The
     filenames are EASTERN-TIME labels; a UTC date near midnight names a
     different report day, and one wrong boundary constant means a backfill
     quietly fetches empty grids.
  2. The parser records what it cannot parse. A malformed row that vanishes
     is exactly the class of bug docs/measurement_bugs.md is about -- the
     numbers still look fine and the missing row is invisible. Every line a
     fixture puts on the page is accounted for by the parser: row, unparsed,
     furniture, header or blank. NOTHING is dropped.
  3. Point-in-time semantics are the "known at 6 p.m. ET" question: the
     latest designation at or before the cutoff, inclusive at the boundary.
     The headline case is a Questionable -> Out late scratch: known at
     5:30 p.m., Questionable; known at 5:47 p.m., Out. A model that reads the
     final status instead has leaked.
  4. Capture is idempotent and resumable. A duplicate fetch is a no-op; a
     recorded absence is never re-probed behind your back; --max-requests can
     always stop a run, and the next run picks up exactly where it stopped.

No network access anywhere: every fetch is a recorded fake. The PDF fixtures
are generated at test time -- minimal, absolutely-positioned text grids that
the real parser reads through pdfplumber, so the geometry layer is exercised
end to end without downloading anything.
"""

import contextlib
import hashlib
import importlib.util
import io
import os
import sqlite3
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT, ROOT / "src" / "Process-Data"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

_spec = importlib.util.spec_from_file_location(
    "gid", ROOT / "src" / "Process-Data" / "Get_Injury_Data.py"
)
gid = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gid)

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Fixture PDFs. Minimal PDF 1.4 with one absolutely-positioned text run per
# cell, WinAnsi-encoded so em/en dashes survive (the reports use "G League –
# Two-Way"). pdfplumber reads char boxes from these exactly as from a real
# report, so the parser's column geometry is tested against real extraction.
# ---------------------------------------------------------------------------

PAGE_W, PAGE_H = 792, 612

HEADER = ["Game Date", "Game Time", "Matchup", "Team", "Player",
          "Current Status", "Reason"]
COLX = [36, 90, 140, 280, 380, 500, 590]


def make_pdf(pages, gap_words=False):
    """pages: list of list of (y, [(x, text)]). Returns PDF bytes.

    gap_words=True emits every word as its own text run with NO space glyphs
    (spaces become pure advances, one Courier space wide): that is exactly how
    the live 2025-26 reports encode text, and it is the only way to pin the
    parser's gap-based word splitting and space reconstruction. The default
    mode writes real space glyphs. Both render identically for Courier, so a
    parse of one must equal a parse of the other.
    """
    char_w = 0.6 * 8  # Courier at the 8pt size the content stream requests
    n_pages = len(pages)
    objs = []
    font_num = 3 + 2 * n_pages
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pages))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode())
    for i, lines in enumerate(pages):
        parts = ["BT /F1 8 Tf"]
        for y, cells in lines:
            for x, text in cells:
                if gap_words:
                    cursor = float(x)
                    runs = []
                    for word in text.split(" "):
                        if word:
                            runs.append((cursor, word))
                        cursor += (len(word) + 1) * char_w
                else:
                    runs = [(float(x), text)]
                for run_x, run_text in runs:
                    esc = (run_text.replace("\\", r"\\")
                           .replace("(", r"\(").replace(")", r"\)"))
                    parts.append(f"1 0 0 1 {run_x} {y} Tm ({esc}) Tj")
        parts.append("ET")
        stream = "\n".join(parts).encode("cp1252")
        objs.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources << /Font << /F1 {font_num} 0 R >> >> "
            f"/Contents {4 + 2 * i} 0 R >>".encode()
        )
        objs.append(
            b"<< /Length " + str(len(stream)).encode()
            + b" >>\nstream\n" + stream + b"\nendstream"
        )
    objs.append(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier "
        b"/Encoding /WinAnsiEncoding >>"
    )

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets[1:]:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


def build_page(rows, title="NBA Official Injury Report",
               header=HEADER, colx=COLX):
    """One fixture page: optional title, a column header, then the rows.

    Each row is a 7-item list in canonical order; "" leaves a cell empty (a
    wrapped continuation, or a grouped-table row that inherits its context).
    Returns the page's (y, [(x, text)]) lines AND the line count, so tests can
    assert the parser's accounting adds up to what was actually on the page.
    """
    lines = []
    y = 580
    if title:
        lines.append((y, [(36, title)]))
        y -= 24
    lines.append((y, list(zip(colx, header))))
    y -= 14
    for row in rows:
        lines.append((y, [(x, t) for x, t in zip(colx, row) if t]))
        y -= 14
    return lines, len(lines)


def fixture_pdf(rows, gap_words=False, **kwargs):
    lines, n_lines = build_page(rows, **kwargs)
    return make_pdf([lines], gap_words=gap_words), n_lines


GAME = "New York at Brooklyn"


def game_row(player, status, reason="", team="Brooklyn Nets",
             game_date="10/23/2026", game_time="7:30 pm ET", matchup=GAME):
    return [game_date, game_time, matchup, team, player, status, reason]


class PdfAccountingMixin:
    """The no-silent-drop invariant, reusable across parser tests."""

    def assertNothingDropped(self, result, total_lines):
        accounted = (
            len(result["rows"]) + len(result["unparsed"]) + len(result["furniture"])
            + result["header_lines"] + result["blank_lines"]
            # A wrapped line is accounted for INSIDE the row it continues --
            # merged, not dropped -- so it counts here too.
            + result["continuation_lines"]
        )
        self.assertEqual(
            accounted, total_lines,
            "every line on the page must be accounted for as a parsed row, an "
            "unparsed row, document furniture, a repeated header or a blank. "
            f"{accounted} of {total_lines} were; the rest were dropped silently."
        )


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


class TestUrlConstruction(unittest.TestCase):
    """The three filename eras, pinned byte for byte."""

    def test_era1_hour_only_label(self):
        self.assertEqual(
            gid.injury_report_url(date(2019, 1, 15), 17),
            "https://ak-static.cms.nba.com/referee/injury/"
            "Injury-Report_2019-01-15_05PM.pdf",
        )

    def test_era2_hour_only_label(self):
        self.assertEqual(
            gid.injury_report_url(date(2023, 2, 14), 0),
            "https://ak-static.cms.nba.com/referee/injury/"
            "Injury-Report_2023-02-14_12AM.pdf",
        )

    def test_era3_minute_label(self):
        self.assertEqual(
            gid.injury_report_url(date(2026, 10, 23), 0),
            "https://ak-static.cms.nba.com/referee/injury/"
            "Injury-Report_2026-10-23_12_00AM.pdf",
        )
        self.assertEqual(
            gid.injury_report_url(date(2026, 10, 23), 17, 45),
            "https://ak-static.cms.nba.com/referee/injury/"
            "Injury-Report_2026-10-23_05_45PM.pdf",
        )

    def test_twelve_hour_clock_edges(self):
        # 12 AM is hour 0, 12 PM is hour 12. Getting either backwards is the
        # classic 12-hour-clock bug and would name the wrong report.
        self.assertEqual(gid.time_token(date(2026, 10, 23), 0, 0), "12_00AM")
        self.assertEqual(gid.time_token(date(2026, 10, 23), 12, 0), "12_00PM")
        self.assertEqual(gid.time_token(date(2026, 10, 23), 13, 5), "01_05PM")
        self.assertEqual(gid.time_token(date(2026, 10, 23), 23, 45), "11_45PM")
        self.assertEqual(gid.time_token(date(2026, 10, 23), 11, 59), "11_59AM")
        self.assertEqual(gid.time_token(date(2026, 10, 23), 0, 30), "12_30AM")

    def test_format_boundary_is_2025_26_opening_night(self):
        self.assertEqual(gid.time_token(date(2025, 10, 20), 23, 0), "11PM")
        self.assertEqual(gid.time_token(date(2025, 10, 21), 23, 0), "11_00PM")

    def test_hour_era_rejects_minutes_rather_than_rounding_them(self):
        with self.assertRaises(gid.InjuryUrlError):
            gid.injury_report_url(date(2023, 2, 14), 17, 30)

    def test_dates_before_the_first_report_are_refused(self):
        with self.assertRaises(gid.InjuryUrlError):
            gid.injury_report_url(date(2018, 11, 30), 17)

    def test_invalid_hour_or_minute_is_refused(self):
        for hour, minute in ((24, 0), (-1, 0), (12, 60), (12, -1)):
            with self.assertRaises(gid.InjuryUrlError):
                gid.injury_report_url(date(2026, 10, 23), hour, minute)
        with self.assertRaises(gid.InjuryUrlError):
            gid.injury_report_url(date(2026, 10, 23), "17")  # not an int

    def test_urls_are_unique_per_slot(self):
        urls = {u for _, u in gid.candidate_urls(date(2026, 10, 23))}
        self.assertEqual(len(urls), 96)
        urls = {u for _, u in gid.candidate_urls(date(2019, 1, 15))}
        self.assertEqual(len(urls), 24)

    def test_filename_inverse_lands_on_the_original_label(self):
        for day, hour, minute in (
            (date(2019, 1, 15), 17, 0),
            (date(2023, 2, 14), 0, 0),
            (date(2026, 10, 23), 0, 0),
            (date(2026, 10, 23), 17, 45),
        ):
            url = gid.injury_report_url(day, hour, minute)
            parsed = gid.parse_report_filename(url)
            self.assertEqual(parsed["report_date"], day)
            self.assertEqual((parsed["hour"], parsed["minute"]), (hour, minute))

    def test_filename_inverse_tolerates_single_digit_hour(self):
        # Recovered Wayback filenames are not all zero-padded; parsing is
        # where drift is tolerated, construction is where it is not.
        parsed = gid.parse_report_filename("Injury-Report_2019-01-15_5PM.pdf")
        self.assertEqual((parsed["hour"], parsed["minute"]), (17, 0))
        self.assertEqual(parsed["time_label"], "05PM")

    def test_filename_inverse_rejects_non_report_names(self):
        for bad in ("Injury-Report_x.pdf", "other.pdf",
                    "Injury-Report_2019-01-15_05PM.txt",
                    "Injury-Report_2019-13-40_05PM.pdf"):
            with self.assertRaises(gid.InjuryUrlError):
                gid.parse_report_filename(bad)


# ---------------------------------------------------------------------------
# Slot enumeration and DST
# ---------------------------------------------------------------------------


class TestSlotsAndDst(unittest.TestCase):
    def test_slot_density_per_era(self):
        self.assertEqual(len(gid.slots_for_date(date(2019, 1, 15))), 24)
        self.assertEqual(len(gid.slots_for_date(date(2023, 2, 14))), 24)
        self.assertEqual(len(gid.slots_for_date(date(2026, 10, 23))), 96)

    def test_slots_are_chronological(self):
        slots = gid.slots_for_date(date(2026, 10, 23))
        self.assertEqual(slots[0], (0, 0))
        self.assertEqual(slots[-1], (23, 45))
        self.assertEqual(slots, sorted(slots))

    def test_slot_filter_rejects_minutes_in_the_hour_era(self):
        with self.assertRaises(gid.InjuryUrlError):
            gid.slots_for_date(date(2023, 2, 14), slots=[(17, 30)])
        self.assertEqual(gid.slots_for_date(date(2023, 2, 14), slots=[(17, 0)]),
                         [(17, 0)])

    def test_labels_convert_to_utc_across_the_est_edt_boundary(self):
        # 2026-10-23 is EDT (UTC-4); 2026-01-15 is EST (UTC-5).
        self.assertEqual(
            gid.et_label_to_utc(date(2026, 10, 23), 17),
            datetime(2026, 10, 23, 21, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(
            gid.et_label_to_utc(date(2026, 1, 15), 17),
            datetime(2026, 1, 15, 22, 0, tzinfo=timezone.utc),
        )

    def test_fall_back_ambiguous_hour_resolves_to_the_first_occurrence(self):
        # 1 a.m. on 2026-11-01 happens twice (EDT then EST). The label must
        # resolve deterministically, and it resolves to the EARLIER instant.
        self.assertEqual(
            gid.et_label_to_utc(date(2026, 11, 1), 1),
            datetime(2026, 11, 1, 5, 0, tzinfo=timezone.utc),
        )

    def test_spring_forward_gap_hour_resolves_after_the_gap(self):
        # 2 a.m. on 2026-03-08 does not exist. The label resolves with the
        # pre-transition (standard) offset: 2:00 EST -> 07:00Z, which is
        # 3:00 EDT on the wall -- immediately after the gap.
        self.assertEqual(
            gid.et_label_to_utc(date(2026, 3, 8), 2),
            datetime(2026, 3, 8, 7, 0, tzinfo=timezone.utc),
        )

    def test_report_day_is_eastern_not_utc(self):
        # 03:30Z is 23:30 ET the DAY BEFORE. A UTC date here would enumerate
        # the wrong report day -- the DST-safe date handling requirement.
        self.assertEqual(
            gid.report_date_for_instant(datetime(2026, 3, 8, 3, 30, tzinfo=timezone.utc)),
            date(2026, 3, 7),
        )
        self.assertEqual(
            gid.report_date_for_instant(datetime(2026, 10, 23, 3, 0, tzinfo=timezone.utc)),
            date(2026, 10, 22),
        )

    def test_cutoff_parses_et_suffix_and_offset_identically(self):
        self.assertEqual(
            gid.parse_cutoff("2026-10-23T18:00:00 ET"),
            gid.parse_cutoff("2026-10-23T18:00:00-04:00"),
        )

    def test_cutoff_rejects_naive_input(self):
        for naive in (datetime(2026, 10, 23, 18, 0),
                      "2026-10-23T18:00:00"):
            with self.assertRaises(gid.InjuryDataError):
                gid.parse_cutoff(naive)


# ---------------------------------------------------------------------------
# Parser fixtures, one per era
# ---------------------------------------------------------------------------


class TestParserEras(PdfAccountingMixin, unittest.TestCase):
    def test_era1_rows_and_the_wrapped_reason_that_must_not_become_a_row(self):
        rows = [
            game_row("Doe, John", "Out", "Right Knee Soreness",
                     game_date="01/15/2019"),
            game_row("Roe, Jane", "Questionable", "Return to Competition",
                     game_date="01/15/2019"),
            # wrapped cell: only the Reason column continues on this line
            ["", "", "", "", "", "", "Reconditioning"],
        ]
        pdf, n_lines = fixture_pdf(rows)
        result = gid.parse_injury_pdf(pdf)

        self.assertIsNone(result["error"])
        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(result["continuation_lines"], 1)
        wrapped = result["rows"][1]
        self.assertEqual(wrapped["reason"], "Return to Competition Reconditioning")
        self.assertEqual(wrapped["player"], "Roe, Jane")
        self.assertEqual(wrapped["status"], "Questionable")
        self.assertEqual(result["rows"][0]["game_date"], "2019-01-15")
        self.assertEqual(result["rows"][0]["game_time"], "7:30 pm ET")
        self.assertNothingDropped(result, n_lines)

    def test_era2_iso_dates_and_the_en_dash_reason_are_preserved(self):
        rows = [
            game_row("Roe, Jane", "Out", "G League \u2013 Two-Way",
                     game_date="2023-02-14", game_time="7:30 pm ET"),
            game_row("Doe, John", "Probable", "Rest",
                     game_date="2023-02-14", game_time="7:30 pm ET"),
        ]
        pdf, n_lines = fixture_pdf(rows)
        result = gid.parse_injury_pdf(pdf)

        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(result["rows"][0]["game_date"], "2023-02-14")
        self.assertEqual(result["rows"][0]["reason"], "G League \u2013 Two-Way")
        self.assertEqual(result["rows"][1]["status"], "Probable")
        self.assertNothingDropped(result, n_lines)

    def test_era3_rows(self):
        rows = [game_row("Doe, John", "Doubtful", "Left Ankle Sprain")]
        pdf, n_lines = fixture_pdf(rows)
        result = gid.parse_injury_pdf(pdf)
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0]["status"], "Doubtful")
        self.assertNothingDropped(result, n_lines)

    def test_malformed_row_is_recorded_with_raw_text_not_dropped(self):
        rows = [
            game_row("Doe, John", "Out", "Right Knee Soreness"),
            # malformed: identity cells present, Player empty
            game_row("", "Out", "Injury/Illness"),
            game_row("Poe, Alex", "Available", ""),
        ]
        pdf, n_lines = fixture_pdf(rows)
        result = gid.parse_injury_pdf(pdf)

        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(len(result["unparsed"]), 1)
        bad = result["unparsed"][0]
        self.assertIn("Player", bad["reason"])
        self.assertIn("Injury/Illness", bad["raw_text"])
        self.assertIn("Brooklyn Nets", bad["raw_text"])
        self.assertNothingDropped(result, n_lines)

    def test_corrupt_game_date_row_is_recorded_not_dropped(self):
        rows = [
            game_row("Doe, John", "Out", "Rest",
                     game_date="TBD not a date"),
        ]
        pdf, n_lines = fixture_pdf(rows)
        result = gid.parse_injury_pdf(pdf)
        self.assertEqual(len(result["rows"]), 0)
        self.assertEqual(len(result["unparsed"]), 1)
        self.assertIn("Game Date", result["unparsed"][0]["reason"])
        self.assertNothingDropped(result, n_lines)

    def test_column_order_drift_still_parses_by_label(self):
        # Era-defensive: "Team" and "Player" swap places and the columns get
        # wider. Cells are placed by their header LABELS' positions, so the
        # data must land in the right fields regardless.
        header = ["Game Date", "Game Time", "Matchup", "Player", "Team",
                  "Current Status", "Reason"]
        colx = [36, 100, 170, 330, 440, 540, 620]
        rows = [["10/23/2026", "7:30 pm ET", GAME,
                 "Doe, John", "Brooklyn Nets", "Out", "Rest"]]
        lines, n_lines = build_page(rows, header=header, colx=colx)
        result = gid.parse_injury_pdf(make_pdf([lines]))

        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(result["rows"][0]["player"], "Doe, John")
        self.assertEqual(result["rows"][0]["team"], "Brooklyn Nets")
        self.assertNothingDropped(result, n_lines)

    def test_grouped_table_rows_inherit_the_game_context(self):
        # A drift shape where the date/time/matchup are printed once per game
        # block instead of on every row. The row must inherit them, and the
        # inherited game_date must be real, not blank.
        rows = [
            game_row("Doe, John", "Out", "Rest"),
            ["", "", "", "New York Knicks", "Roe, Jane", "Questionable",
             "Not With Team"],
        ]
        pdf, n_lines = fixture_pdf(rows)
        result = gid.parse_injury_pdf(pdf)

        self.assertEqual(len(result["rows"]), 2)
        second = result["rows"][1]
        self.assertEqual(second["game_date"], "2026-10-23")
        self.assertEqual(second["matchup"], GAME)
        self.assertEqual(second["player"], "Roe, Jane")
        self.assertNothingDropped(result, n_lines)

    def test_second_page_with_repeated_header_parses_both_pages(self):
        page1, n1 = build_page([game_row("Doe, John", "Out", "Rest")])
        page2, n2 = build_page(
            [game_row("Roe, Jane", "Available", "")],
            title="Injury Report continued",
        )
        result = gid.parse_injury_pdf(make_pdf([page1, page2]))

        self.assertEqual(len(result["rows"]), 2)
        self.assertEqual(result["header_lines"], 2)
        self.assertEqual(result["rows"][1]["player"], "Roe, Jane")
        self.assertNothingDropped(result, n1 + n2)

    def test_titles_are_furniture_not_fake_rows(self):
        rows = [game_row("Doe, John", "Out", "Rest")]
        pdf, n_lines = fixture_pdf(rows)
        result = gid.parse_injury_pdf(pdf)
        self.assertEqual(len(result["rows"]), 1)
        self.assertEqual(len(result["unparsed"]), 0)
        self.assertTrue(result["furniture"])
        self.assertIn("NBA Official Injury Report",
                      result["furniture"][0]["raw_text"])
        self.assertNothingDropped(result, n_lines)

    def test_headerless_document_fails_loudly_and_keeps_every_line(self):
        lines = [(580, [(36, "Some text that is not a table")]),
                 (560, [(36, "Nor is this a row")])]
        result = gid.parse_injury_pdf(make_pdf([lines]))

        self.assertIsNotNone(result["error"])
        self.assertIn("no column header", result["error"])
        self.assertEqual(len(result["rows"]), 0)
        self.assertEqual(len(result["unparsed"]), 2)  # kept, not dropped
        self.assertNothingDropped(result, 2)

    def test_non_pdf_input_is_an_error_not_an_empty_parse(self):
        with self.assertRaises(gid.InjuryParseError):
            gid.parse_injury_pdf(b"<html>cdn error page</html>")
        with self.assertRaises(gid.InjuryParseError):
            gid.parse_injury_pdf(b"")

    def test_pdfplumber_is_imported_lazily(self):
        # The URL, time and store layers must stay usable without the PDF
        # stack -- same rule as nba_client.py and nba_api. Checked in a fresh
        # interpreter, which is also how CI sees it.
        script = (
            "import importlib.util, sys;"
            f"spec = importlib.util.spec_from_file_location('gid', "
            f"{str(ROOT / 'src' / 'Process-Data' / 'Get_Injury_Data.py')!r});"
            "m = importlib.util.module_from_spec(spec);"
            "spec.loader.exec_module(m);"
            "assert 'pdfplumber' not in sys.modules, "
            "'pdfplumber must not be imported at module scope';"
            "print('ok')"
        )
        proc = subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("ok", proc.stdout)


# ---------------------------------------------------------------------------
# Store: idempotency, absences, conflicts
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code, content, headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}


class FakeRequester:
    """A recorded transport. Maps URL -> (status, body); defaults to 404."""

    def __init__(self, mapping=None, sequence=None):
        self.mapping = mapping or {}
        self.sequence = list(sequence or [])
        self.calls = []

    def __call__(self, url, timeout):
        self.calls.append(url)
        if self.sequence:
            status, body = self.sequence.pop(0)
        else:
            status, body = self.mapping.get(url, (404, b""))
        return FakeResponse(status, body)


class ZeroRng:
    """Deterministic pacing: uniform() is only used for sleep jitter."""

    def uniform(self, low, high):
        return 0.0


def no_sleep(_seconds):
    return None


def outcome(url, content=None, http_status=None, fetched_at="2026-10-23T21:45:00+00:00"):
    if content is not None:
        http_status = 200 if http_status is None else http_status
    else:
        http_status = 404 if http_status is None else http_status
    return {"url": url, "content": content, "http_status": http_status,
            "fetched_at": fetched_at}


class StoreTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.db = self.base / "injury.sqlite"
        self.raw = self.base / "raw"
        self.con = gid.open_db(self.db)
        self.addCleanup(self.con.close)
        self.url = gid.injury_report_url(date(2026, 10, 23), 17, 45)
        self.pdf = make_pdf([build_page([game_row("Doe, John", "Out", "Rest")])[0]])

    def snapshot_rows(self):
        return self.con.execute(
            'SELECT * FROM "injury_snapshots" ORDER BY "id"'
        ).fetchall()


class TestStoreIdempotency(StoreTestCase):
    def test_content_is_stored_by_url_and_sha256(self):
        result = gid.store_fetch(self.con, outcome(self.url, self.pdf),
                                 raw_dir=self.raw)
        self.assertEqual(result["status"], "inserted")
        rows = self.snapshot_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sha256"], hashlib.sha256(self.pdf).hexdigest())
        self.assertEqual(rows[0]["http_status"], 200)
        self.assertEqual(rows[0]["parse_status"], "unparsed")
        stored = self.raw / rows[0]["raw_path"]
        self.assertEqual(stored.read_bytes(), self.pdf)
        self.assertTrue(rows[0]["raw_path"].startswith("2026-10-23/"))

    def test_duplicate_fetch_is_a_no_op(self):
        gid.store_fetch(self.con, outcome(self.url, self.pdf), raw_dir=self.raw)
        result = gid.store_fetch(self.con, outcome(self.url, self.pdf),
                                 raw_dir=self.raw)
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(len(self.snapshot_rows()), 1)
        files = [p for p in self.raw.rglob("*") if p.is_file()]
        self.assertEqual(len(files), 1)

    def test_duplicate_fetch_through_capture_makes_no_request(self):
        requester = FakeRequester({self.url: (200, self.pdf)})
        kwargs = dict(
            db_path=self.db, raw_dir=self.raw, requester=requester,
            sleep=no_sleep, rng=ZeroRng(),
            now=datetime(2026, 10, 23, 23, 0, tzinfo=ET),
            slots=[(17, 45)],
        )
        first = gid.capture_days(date(2026, 10, 23), date(2026, 10, 23), **kwargs)
        self.assertEqual(first["fetched"], 1)
        second = gid.capture_days(date(2026, 10, 23), date(2026, 10, 23), **kwargs)
        self.assertEqual(second["requests_made"], 0)
        self.assertEqual(second["already_recorded"], 1)
        self.assertEqual(len(requester.calls), 1)
        self.assertEqual(len(self.snapshot_rows()), 1)

    def test_conflicting_bytes_at_one_url_are_kept_side_by_side(self):
        # The CDN is documented immutable; different bytes at one URL is an
        # anomaly that must SURVIVE as evidence, not overwrite the first copy.
        gid.store_fetch(self.con, outcome(self.url, self.pdf), raw_dir=self.raw)
        other = self.pdf + b"\n% changed"
        result = gid.store_fetch(self.con, outcome(self.url, other), raw_dir=self.raw)
        self.assertEqual(result["status"], "conflict")
        rows = self.snapshot_rows()
        self.assertEqual(len(rows), 2)
        paths = [r["raw_path"] for r in rows]
        self.assertEqual(len(set(paths)), 2, "each version keeps its own file")
        for row in rows:
            self.assertEqual((self.raw / row["raw_path"]).exists(), True)

    def test_missing_bytes_on_disk_are_restored_loudly(self):
        gid.store_fetch(self.con, outcome(self.url, self.pdf), raw_dir=self.raw)
        stored = self.raw / self.snapshot_rows()[0]["raw_path"]
        stored.unlink()
        result = gid.store_fetch(self.con, outcome(self.url, self.pdf),
                                 raw_dir=self.raw)
        self.assertEqual(result["status"], "restored")
        self.assertEqual(stored.read_bytes(), self.pdf)

    def test_404_is_recorded_as_an_absence_not_forgotten(self):
        result = gid.store_fetch(self.con, outcome(self.url, None),
                                 raw_dir=self.raw)
        self.assertEqual(result["status"], "absent")
        rows = self.snapshot_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["sha256"], gid.NO_CONTENT_SHA256)
        self.assertEqual(rows[0]["parse_status"], "missing")
        self.assertIsNone(rows[0]["raw_path"])

    def test_non_pdf_body_is_recorded_but_never_stored_as_data(self):
        result = gid.store_fetch(self.con, outcome(self.url, b"<html>oops</html>"),
                                 raw_dir=self.raw)
        self.assertEqual(result["status"], "not_a_pdf")
        rows = self.snapshot_rows()
        self.assertEqual(rows[0]["parse_status"], "not_a_pdf")
        self.assertEqual(rows[0]["sha256"], gid.NOT_A_PDF_SHA256,
                         "not-a-PDF gets its own sentinel: it is a different "
                         "measurement from a 404 and must not share its row")
        self.assertFalse(any(p.is_file() for p in self.raw.rglob("*")))

    def test_404_and_not_a_pdf_are_distinct_measurements(self):
        gid.store_fetch(self.con, outcome(self.url, None), raw_dir=self.raw)
        other_url = gid.injury_report_url(date(2026, 10, 23), 18, 0)
        gid.store_fetch(self.con, outcome(other_url, b"<html>oops</html>"),
                        raw_dir=self.raw)
        rows = self.snapshot_rows()
        self.assertEqual(len(rows), 2)
        sentinels = {r["sha256"] for r in rows}
        self.assertEqual(sentinels, {gid.NO_CONTENT_SHA256, gid.NOT_A_PDF_SHA256},
                         "one row per fetch outcome, distinct sentinels: "
                         "'records every fetch outcome' must be literally true")

    def test_parse_is_idempotent_and_reparse_replaces_derived_rows_only(self):
        gid.store_fetch(self.con, outcome(self.url, self.pdf), raw_dir=self.raw)
        snap_id = self.snapshot_rows()[0]["id"]
        first = gid.parse_snapshot(self.con, snap_id, raw_dir=self.raw)
        self.assertEqual(first["rows"], 1)
        again = gid.parse_snapshot(self.con, snap_id, raw_dir=self.raw)
        self.assertEqual(again["status"], "already_parsed")
        n_events = self.con.execute(
            'SELECT COUNT(*) FROM "injury_status_events"').fetchone()[0]
        self.assertEqual(n_events, 1, "re-parsing must not duplicate events")
        gid.parse_snapshot(self.con, snap_id, raw_dir=self.raw, reparse=True)
        n_events = self.con.execute(
            'SELECT COUNT(*) FROM "injury_status_events"').fetchone()[0]
        self.assertEqual(n_events, 1)
        self.assertEqual(self.raw.joinpath(self.snapshot_rows()[0]["raw_path"]).read_bytes(),
                         self.pdf, "raw evidence is never touched by a reparse")


class TestAbsenceResumePolicy(StoreTestCase):
    """The retry gate is a function of the SLOT, not of the probe clock.

    The bug this pins: a probe-clock window is invisible to a 3-hourly batch
    (its absences are always older than the window), so every 404 recorded
    during the CDN's 5-20 minute upload lag became a PERMANENT wrong absence
    -- and a point-in-time query would then serve a stale designation for the
    exact late scratches this module exists for. The gate below keys on
    `now - et_label_to_utc(slot)`, which is how upload lag actually behaves.
    """

    def test_absence_for_a_fresh_slot_is_reprobed_even_from_an_old_probe(self):
        # The probe itself is ancient (recorded at the start of the day), but
        # the SLOT is only 30 minutes old: the PDF may simply not have landed
        # yet. This must re-probe -- at any capture cadence.
        gid.store_fetch(
            self.con,
            outcome(self.url, None, fetched_at="2026-10-23T04:00:00+00:00"),
            raw_dir=self.raw,
        )
        slot_utc = datetime(2026, 10, 23, 21, 45, tzinfo=timezone.utc)  # 17:45 ET
        now = slot_utc + timedelta(minutes=30)
        self.assertTrue(gid.fetch_needed(self.con, self.url, now=now),
                        "a 30-minute-old slot's absence is an upload lag, "
                        "not a fact: it must be re-probed")

    def test_absence_for_an_old_slot_is_final_even_if_just_probed(self):
        # The mirror image: the slot is three days old and was probed one
        # second ago. A historical backfill must NOT re-probe it on every
        # pass -- that is what froze throughput at one batch's budget per
        # retry window.
        slot_utc = datetime(2026, 10, 23, 21, 45, tzinfo=timezone.utc)
        now = slot_utc + timedelta(days=3)
        gid.store_fetch(
            self.con,
            outcome(self.url, None, fetched_at=now.isoformat()),
            raw_dir=self.raw,
        )
        self.assertFalse(gid.fetch_needed(self.con, self.url, now=now),
                         "an old slot's absence is the truth about an "
                         "immutable URL: final")

    def test_retry_missing_still_forces_the_issue(self):
        slot_utc = datetime(2026, 10, 23, 21, 45, tzinfo=timezone.utc)
        now = slot_utc + timedelta(days=3)
        gid.store_fetch(
            self.con,
            outcome(self.url, None, fetched_at=now.isoformat()),
            raw_dir=self.raw,
        )
        self.assertTrue(
            gid.fetch_needed(self.con, self.url, retry_missing=True, now=now)
        )

    def test_not_a_pdf_is_reprobed_like_an_absence(self):
        gid.store_fetch(self.con, outcome(self.url, b"<html>lag</html>"),
                        raw_dir=self.raw)
        slot_utc = datetime(2026, 10, 23, 21, 45, tzinfo=timezone.utc)
        self.assertTrue(
            gid.fetch_needed(self.con, self.url, now=slot_utc + timedelta(minutes=30))
        )
        self.assertFalse(
            gid.fetch_needed(self.con, self.url, now=slot_utc + timedelta(days=3))
        )

    def test_content_is_never_reprobed(self):
        gid.store_fetch(self.con, outcome(self.url, self.pdf), raw_dir=self.raw)
        self.assertFalse(
            gid.fetch_needed(self.con, self.url, retry_missing=True)
        )

    def test_unrecorded_slots_are_always_probed(self):
        self.assertTrue(gid.fetch_needed(self.con, self.url))


# ---------------------------------------------------------------------------
# Capture: budget, resumability, backoff
# ---------------------------------------------------------------------------


class TestCaptureResumable(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.db = self.base / "injury.sqlite"
        self.raw = self.base / "raw"

    def capture(self, requester, **kwargs):
        kwargs.setdefault("db_path", self.db)
        kwargs.setdefault("raw_dir", self.raw)
        kwargs.setdefault("sleep", no_sleep)
        kwargs.setdefault("rng", ZeroRng())
        kwargs.setdefault("now", datetime(2026, 10, 23, 23, 0, tzinfo=ET))
        return gid.capture_days(
            kwargs.pop("day_start", date(2026, 10, 23)),
            kwargs.pop("day_end", date(2026, 10, 23)),
            requester=requester, **kwargs,
        )

    def test_budget_stops_a_run_and_the_next_run_resumes_it(self):
        # Three real slots, a two-request budget: the run must stop cleanly,
        # and re-running must fetch ONLY what is left. This is the guard that
        # makes a smoke test unable to become a bulk download.
        slots = [(17, 0), (17, 45), (18, 0)]
        pdf = make_pdf([build_page([game_row("Doe, John", "Out", "Rest")])[0]])
        mapping = {gid.injury_report_url(date(2026, 10, 23), h, m): (200, pdf)
                   for h, m in slots}
        requester = FakeRequester(mapping)

        first = self.capture(requester, slots=slots, max_requests=2)
        self.assertTrue(first["budget_exhausted"])
        self.assertEqual(first["requests_made"], 2)
        self.assertEqual(first["fetched"], 2)

        second = self.capture(requester, slots=slots, max_requests=2)
        self.assertFalse(second["budget_exhausted"])
        self.assertEqual(second["requests_made"], 1, "resumes, does not re-fetch")
        self.assertEqual(second["already_recorded"], 2)
        self.assertEqual(len(requester.calls), 3)

    def test_summary_buckets_account_for_every_request(self):
        # Every stored fetch lands in exactly one bucket: the buckets plus
        # errors must sum to requests_made (no retries in this run).
        slots = [(17, 0), (18, 0)]
        pdf = make_pdf([build_page([game_row("Doe, John", "Out", "Rest")])[0]])
        mapping = {
            gid.injury_report_url(date(2026, 10, 23), 17, 0): (200, pdf),
            gid.injury_report_url(date(2026, 10, 23), 18, 0): (404, b""),
        }
        summary = self.capture(FakeRequester(mapping), slots=slots,
                               max_requests=10)
        accounted = (
            summary["fetched"] + summary["absent"] + summary["not_a_pdf"]
            + summary["conflicts"] + summary["restored"] + summary["skipped"]
            + summary["errors"]
        )
        self.assertEqual(accounted, summary["requests_made"])
        self.assertEqual(summary["requests_made"], 2)
        self.assertIn("skipped", summary)

    def test_future_slots_are_never_probed(self):
        # At 11:00 ET the 23:45 snapshot does not exist yet. Recording it as
        # an absence would poison resumability -- it would never be fetched.
        requester = FakeRequester()
        summary = self.capture(
            requester, slots=[(10, 0), (23, 45)],
            now=datetime(2026, 10, 23, 11, 0, tzinfo=ET),
        )
        self.assertEqual(summary["future_skipped"], 1)
        self.assertEqual(summary["requests_made"], 1)
        self.assertEqual(len(requester.calls), 1)
        self.assertNotIn("23_45PM", requester.calls[0])

    def test_backoff_grows_exponentially_between_attempts(self):
        slept = []
        pdf = make_pdf([build_page([game_row("Doe, John", "Out", "Rest")])[0]])
        requester = FakeRequester(sequence=[(500, b""), (429, b""), (200, pdf)])
        result = gid.fetch_url(
            gid.injury_report_url(date(2026, 10, 23), 17, 0), requester=requester,
            sleep=slept.append,
        )
        self.assertEqual(result["http_status"], 200)
        self.assertEqual(len(requester.calls), 3)
        self.assertEqual(slept, [10.0, 20.0])

    def test_404_is_final_for_one_fetch(self):
        slept = []
        requester = FakeRequester()
        result = gid.fetch_url(gid.injury_report_url(date(2026, 10, 23), 17, 0),
                               requester=requester, sleep=slept.append)
        self.assertEqual(result["http_status"], 404)
        self.assertEqual(len(requester.calls), 1, "an absent slot is not an error")
        self.assertEqual(slept, [])

    def test_transport_exhaustion_raises_rather_than_returning_empty(self):
        def broken(url, timeout):
            raise OSError("connection reset")

        with self.assertRaises(gid.InjuryFetchError):
            gid.fetch_url(gid.injury_report_url(date(2026, 10, 23), 17, 0), requester=broken,
                          sleep=no_sleep)


# ---------------------------------------------------------------------------
# Point-in-time semantics
# ---------------------------------------------------------------------------


class TestPointInTime(unittest.TestCase):
    """The headline question: what was known at 5:47 p.m. ET?

    End to end: fixture PDFs are served by a fake transport, captured into the
    store, parsed, and queried. The 5 p.m. snapshot lists the player
    Questionable; the 5:45 p.m. snapshot flips him to Out (a late scratch).
    A model preparing at 5:47 must see Out; a model preparing at 5:30 must
    see Questionable. Reading the final status at 5:30 is the leak this whole
    pipeline exists to prevent.
    """

    GAME_DAY = date(2026, 10, 23)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.db = self.base / "injury.sqlite"
        self.raw = self.base / "raw"

        early_url = gid.injury_report_url(self.GAME_DAY, 17, 0)
        late_url = gid.injury_report_url(self.GAME_DAY, 17, 45)
        early_pdf, _ = fixture_pdf([
            game_row("Roe, Jane", "Questionable", "Right Hip Soreness",
                     team="New York Knicks"),
        ])
        late_pdf, _ = fixture_pdf([
            game_row("Roe, Jane", "Out", "Right Hip Soreness",
                     team="New York Knicks"),
            game_row("Doe, John", "Available", ""),
        ])
        requester = FakeRequester({
            early_url: (200, early_pdf),
            late_url: (200, late_pdf),
        })
        summary = gid.capture_days(
            self.GAME_DAY, self.GAME_DAY,
            db_path=self.db, raw_dir=self.raw, requester=requester,
            sleep=no_sleep, rng=ZeroRng(),
            now=datetime(2026, 10, 23, 23, 0, tzinfo=ET),
            slots=[(17, 0), (17, 45)],
        )
        assert summary["fetched"] == 2, summary
        self.con = gid.open_db(self.db)
        self.addCleanup(self.con.close)
        gid.parse_all(self.con, raw_dir=self.raw)

    def known(self, cutoff, player="Roe, Jane"):
        return gid.status_known_as_of(self.con, player, self.GAME_DAY, cutoff)

    def test_questionable_to_out_transition_known_at_the_547pm_cutoff(self):
        at_547 = self.known("2026-10-23T17:47:00 ET")
        self.assertIsNotNone(at_547)
        self.assertEqual(at_547["status"], "Out")
        self.assertEqual(at_547["reason"], "Right Hip Soreness")

    def test_the_same_player_reads_questionable_at_the_530pm_cutoff(self):
        at_530 = self.known("2026-10-23T17:30:00 ET")
        self.assertIsNotNone(at_530)
        self.assertEqual(at_530["status"], "Questionable")

    def test_cutoff_at_exactly_the_snapshot_label_is_inclusive(self):
        at_545 = self.known("2026-10-23T17:45:00 ET")
        self.assertEqual(at_545["status"], "Out")

    def test_one_minute_before_the_late_snapshot_sees_the_old_status(self):
        at_544 = self.known("2026-10-23T17:44:00 ET")
        self.assertEqual(at_544["status"], "Questionable")

    def test_a_cutoff_before_any_report_knows_nothing(self):
        at_4pm = self.known("2026-10-23T16:00:00 ET")
        self.assertIsNone(at_4pm)

    def test_a_player_never_listed_knows_nothing(self):
        self.assertIsNone(self.known("2026-10-23T18:00:00 ET", player="Nobody, Some"))

    def test_player_matching_is_case_insensitive(self):
        self.assertEqual(self.known("2026-10-23T18:00:00 ET",
                                    player="roe,  jane")["status"], "Out")

    def test_naive_cutoffs_are_rejected_not_guessed(self):
        with self.assertRaises(gid.InjuryDataError):
            self.known(datetime(2026, 10, 23, 18, 0))

    def test_team_filter_narrows_to_the_right_side(self):
        hit = gid.status_known_as_of(
            self.con, "Roe, Jane", self.GAME_DAY,
            "2026-10-23T18:00:00 ET", team="New York Knicks",
        )
        self.assertEqual(hit["status"], "Out")
        miss = gid.status_known_as_of(
            self.con, "Roe, Jane", self.GAME_DAY,
            "2026-10-23T18:00:00 ET", team="Brooklyn Nets",
        )
        self.assertIsNone(miss)

    def test_history_shows_the_whole_trajectory_oldest_first(self):
        history = gid.status_history(self.con, "Roe, Jane", self.GAME_DAY)
        self.assertEqual([e["status"] for e in history],
                         ["Questionable", "Out"])
        self.assertEqual([e["snapshot_ts_et"] for e in history],
                         ["2026-10-23T17:00:00", "2026-10-23T17:45:00"])

    def test_stored_timestamps_carry_both_the_et_label_and_the_utc_instant(self):
        history = gid.status_history(self.con, "Roe, Jane", self.GAME_DAY)
        self.assertEqual(history[0]["snapshot_ts_et"], "2026-10-23T17:00:00")
        self.assertEqual(history[0]["snapshot_ts_utc"],
                         "2026-10-23T21:00:00+00:00")  # EDT, UTC-4

    def test_nothing_known_is_an_answer_not_an_error(self):
        # 'not yet listed' is exactly what a pre-tip-off model would have
        # seen; it must never be conflated with a failure.
        self.assertIsNone(self.known("2026-10-23T17:20:00 ET",
                                     player="Doe, John"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class TestCli(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.db = str(self.base / "injury.sqlite")
        self.raw = str(self.base / "raw")

    def run_cli(self, argv):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            code = gid.main(argv)
        return code, buffer.getvalue()

    def test_slots_prints_urls_without_touching_the_network(self):
        code, out = self.run_cli(["slots", "--date", "2026-10-23"])
        self.assertEqual(code, 0)
        lines = out.strip().splitlines()
        self.assertEqual(len(lines), 96)
        self.assertIn("Injury-Report_2026-10-23_12_00AM.pdf", lines[0])

    def test_capture_respects_the_request_budget_and_says_so(self):
        pdf = make_pdf([build_page([game_row("Doe, John", "Out", "Rest")])[0]])
        url = gid.injury_report_url(date(2026, 10, 23), 17, 0)
        # The CLI has no injectable transport, so exercise budget accounting
        # through the library instead and pin the CLI's summary formatting on
        # the slots path. (Never a real request in tests.)
        summary = gid.capture_days(
            date(2026, 10, 23), date(2026, 10, 23),
            db_path=self.db, raw_dir=self.raw,
            requester=FakeRequester({url: (200, pdf)}),
            sleep=no_sleep, rng=ZeroRng(), max_requests=1, slots=[(17, 0), (18, 0)],
            now=datetime(2026, 10, 23, 23, 0, tzinfo=ET),
        )
        self.assertTrue(summary["budget_exhausted"])
        self.assertEqual(summary["requests_made"], 1)

    def test_query_prints_the_designation_or_says_nothing_known(self):
        con = gid.open_db(self.db)
        url = gid.injury_report_url(date(2026, 10, 23), 17, 0)
        pdf, _ = fixture_pdf([game_row("Roe, Jane", "Out", "Right Hip Soreness",
                                       team="New York Knicks")])
        gid.store_fetch(con, outcome(url, pdf), raw_dir=self.raw)
        gid.parse_all(con, raw_dir=self.raw)
        con.close()

        code, out = self.run_cli([
            "query", "--player", "Roe, Jane", "--game", "2026-10-23",
            "--cutoff", "2026-10-23T18:00:00 ET", "--db", self.db,
        ])
        self.assertEqual(code, 0)
        self.assertIn("Out", out)
        self.assertIn("Roe, Jane", out)

        code, out = self.run_cli([
            "query", "--player", "Nobody, Some", "--game", "2026-10-23",
            "--cutoff", "2026-10-23T18:00:00 ET", "--db", self.db,
        ])
        self.assertEqual(code, 0)
        self.assertIn("NOTHING KNOWN", out)

    def test_query_on_a_missing_database_fails_loudly(self):
        code, out = self.run_cli([
            "query", "--player", "Roe, Jane", "--game", "2026-10-23",
            "--cutoff", "2026-10-23T18:00:00 ET", "--db", self.db,
        ])
        self.assertEqual(code, 1)
        self.assertIn("FAILED", out)

    def test_bad_arguments_fail_loudly(self):
        code, out = self.run_cli(["capture"])
        self.assertEqual(code, 1)
        self.assertIn("FAILED", out)
        code, out = self.run_cli(["capture", "--date", "23/10/2026"])
        self.assertEqual(code, 1)
        self.assertIn("FAILED", out)

    def test_capture_exits_non_zero_when_requests_fail(self):
        # "Every request failed" must NOT exit 0: a scheduled job that
        # reports success while doing nothing is the failure mode a
        # monitoring exit code exists to catch. No network -- the transport,
        # the retry base and the pacing are swapped out.
        def broken(url, timeout):
            raise OSError("connection refused")

        saved = (gid._default_requester, gid.RETRY_BACKOFF_BASE_SECONDS,
                 gid.MIN_DELAY_SECONDS, gid.MAX_DELAY_SECONDS)
        gid._default_requester = broken
        gid.RETRY_BACKOFF_BASE_SECONDS = 0.0
        gid.MIN_DELAY_SECONDS = 0.0
        gid.MAX_DELAY_SECONDS = 0.0
        try:
            code, out = self.run_cli([
                "capture", "--date", "2026-10-22", "--slot", "00:00",
                "--max-requests", "5", "--db", self.db, "--raw-dir", self.raw,
            ])
        finally:
            (gid._default_requester, gid.RETRY_BACKOFF_BASE_SECONDS,
             gid.MIN_DELAY_SECONDS, gid.MAX_DELAY_SECONDS) = saved
        self.assertEqual(code, 1)
        self.assertIn("FAILED REQUEST", out)


ERA3_COLX = [23, 120, 200, 264, 425, 586, 666]
ERA3_HEADER = ["Game Date", "Game Time", "Matchup", "Team", "Player Name",
               "Current Status", "Reason"]


def era3_line(top, cells):
    """One text line at a page top coordinate, cells in canonical order."""
    return (PAGE_H - top, [(x, t) for x, t in zip(ERA3_COLX, cells) if t])


class TestRealEra3Layout(PdfAccountingMixin, unittest.TestCase):
    """The 2025-26 layout, pinned from the live PDFs themselves.

    Three properties of the real documents break a line-at-a-time parser
    silently, and each is pinned here because getting it wrong produces
    plausible-looking garbage rather than an error:

      * a row is a BAND: the identity line is vertically centred and a long
        reason wraps onto sub-lines ABOVE and BELOW it. Naive parsing gives
        the first half of "Injury/Illness - Right Quadriceps Strain" to the
        row above;
      * Game Date / Game Time / Matchup / Team print ONCE per block and rows
        inherit them -- the wrong team is one skipped fill-forward away;
      * titles and footers scatter words across the Team/Player/Status
        columns ("Injury Report: 10/23/26 12:00 AM" puts "AM" in Status and
        "10/23/26 12:00" in Player) and must be kept out of the rows without
        being dropped.
    """

    def build_doc(self, gap_words=False):
        page1 = [
            (PAGE_H - 45.5, [(288, "Injury Report:"), (437, "10/23/26 12:00 AM")]),
            (PAGE_H - 107.7, list(zip(ERA3_COLX, ERA3_HEADER))),
            era3_line(128.95, ["10/23/2026", "04:00 (ET)", "CHI@MIL",
                               "Chicago Bulls", "Awaka, Tobe", "Probable",
                               "Injury/Illness - Left MCL Sprain"]),
            # identity line, then the reason wraps ABOVE and BELOW it
            era3_line(150.98, ["", "", "", "", "Claxton, Nic", "Out",
                               "Injury/Illness - Left Hamstring Strain"]),
            era3_line(173.02, ["", "", "", "", "", "",
                               "Injury/Illness - Right Quadriceps"]),
            era3_line(180.04, ["", "", "", "", "Collins, Zach", "Probable", ""]),
            era3_line(187.06, ["", "", "", "", "", "", "Strain"]),
            # new team block: only the Team cell changes
            era3_line(387.40, ["", "", "", "Charlotte Hornets", "Boswell, Kylan",
                               "Out", "Injury/Illness - Right Knee Surgery"]),
            (PAGE_H - 538.1, [(398, "Page 1 of"), (438, "6")]),
        ]
        page2 = [
            (PAGE_H - 45.5, [(288, "Injury Report:"), (437, "10/23/26 12:00 AM")]),
            era3_line(82.6, ["", "", "", "", "", "",
                             "Injury/Illness - Right Calf Tightness /"]),
            era3_line(89.6, ["", "", "", "", "Howard, Jett", "Doubtful", ""]),
            era3_line(96.6, ["", "", "", "", "", "", "Soreness"]),
            era3_line(118.7, ["", "", "", "", "Lively II, Dereck", "Out",
                              "Injury/Illness - Right Foot Surgery"]),
            (PAGE_H - 538.1, [(398, "Page 2 of"), (438, "6")]),
        ]
        return make_pdf([page1, page2], gap_words=gap_words), len(page1) + len(page2)

    def setUp(self):
        self.pdf, self.n_lines = self.build_doc()
        self.result = gid.parse_injury_pdf(self.pdf)

    def test_every_row_is_read_out_of_the_band_layout(self):
        rows = {r["player"]: r for r in self.result["rows"]}
        self.assertEqual(len(rows), 6)
        self.assertEqual(rows["Awaka, Tobe"]["team"], "Chicago Bulls")
        self.assertEqual(rows["Awaka, Tobe"]["game_date"], "2026-10-23")
        self.assertEqual(rows["Awaka, Tobe"]["game_time"], "04:00 (ET)")
        self.assertEqual(rows["Awaka, Tobe"]["matchup"], "CHI@MIL")

    def test_the_reason_wrapping_above_the_identity_line_stays_with_its_row(self):
        rows = {r["player"]: r for r in self.result["rows"]}
        self.assertEqual(
            rows["Collins, Zach"]["reason"],
            "Injury/Illness - Right Quadriceps Strain",
        )
        # And the row ABOVE did not absorb the first half of it.
        self.assertEqual(
            rows["Claxton, Nic"]["reason"],
            "Injury/Illness - Left Hamstring Strain",
        )

    def test_top_of_page_wraps_join_the_row_below_not_the_title(self):
        rows = {r["player"]: r for r in self.result["rows"]}
        self.assertEqual(
            rows["Howard, Jett"]["reason"],
            "Injury/Illness - Right Calf Tightness / Soreness",
        )

    def test_block_context_fills_forward_including_across_pages(self):
        rows = {r["player"]: r for r in self.result["rows"]}
        # Team printed once per block: the rows below it inherit it.
        self.assertEqual(rows["Claxton, Nic"]["team"], "Chicago Bulls")
        self.assertEqual(rows["Boswell, Kylan"]["team"], "Charlotte Hornets")
        # Page 2 continues the last block of page 1 (the documents do this).
        self.assertEqual(rows["Howard, Jett"]["team"], "Charlotte Hornets")
        self.assertIn("team", rows["Claxton, Nic"]["inherited"])

    def test_titles_and_footers_are_furniture_not_rows(self):
        self.assertEqual(len(self.result["rows"]), 6)
        raws = [f["raw_text"] for f in self.result["furniture"]]
        self.assertEqual(sum("Page" in r for r in raws), 2)
        self.assertEqual(sum("Injury Report:" in r for r in raws), 2)
        # The title's "10/23/26 12:00" must never appear as a player.
        for row in self.result["rows"]:
            self.assertNotIn("10/23/26", row["player"])

    def test_nothing_on_the_real_layout_is_dropped(self):
        self.assertEqual(self.result["unparsed"], [])
        self.assertNothingDropped(self.result, self.n_lines)

    def test_team_level_not_yet_submitted_lines_are_recorded_not_glued(self):
        # The reports end with team-level lines for the next slate: game
        # context + team + "NOT YET SUBMITTED", no player anywhere. The old
        # nearest-row merge glued all of them onto the last player's reason
        # (this happened to Ben Simmons on every 2026-10-23 snapshot); the
        # parser must record them as what they are and leave rows alone.
        lines = [
            (PAGE_H - 107.7, list(zip(ERA3_COLX, ERA3_HEADER))),
            era3_line(125.71, ["10/23/2026", "04:00 (ET)", "CHI@MIL",
                               "Chicago Bulls", "Simmons, Ben", "Out",
                               "Injury/Illness - Right Achilles Tendon"]),
            era3_line(132.73, ["", "", "", "", "", "", "Repair / Surgery"]),
            era3_line(155.76, ["10/24/2026", "06:00 (ET)", "MIA@CHA",
                               "Miami Heat", "", "", "NOT YET SUBMITTED"]),
            era3_line(178.80, ["", "", "", "Charlotte Hornets", "", "",
                               "NOT YET SUBMITTED"]),
        ]
        result = gid.parse_injury_pdf(make_pdf([lines]))
        self.assertNothingDropped(result, len(lines))

        rows = {r["player"]: r for r in result["rows"]}
        self.assertEqual(list(rows), ["Simmons, Ben"])
        self.assertEqual(rows["Simmons, Ben"]["reason"],
                         "Injury/Illness - Right Achilles Tendon Repair / Surgery")
        slate = [f for f in result["furniture"]
                 if "team-level status line" in f["reason"]]
        self.assertEqual(len(slate), 2)
        self.assertEqual(result["unparsed"], [])

    def test_the_real_layout_parses_identically_without_space_glyphs(self):
        # The live 2025-26 PDFs carry spaces as pure advances, with no space
        # glyphs at all. This runs the SAME document in that mode: header
        # detection (the gap-splitting path in _char_words) and cell text
        # reconstruction (the gap path in _chars_to_text) must both hold, or
        # the live parser quietly stops finding the table.
        spaced_pdf, n_lines = self.build_doc()
        gap_pdf, gap_lines = self.build_doc(gap_words=True)
        spaced = gid.parse_injury_pdf(spaced_pdf)
        gapped = gid.parse_injury_pdf(gap_pdf)

        self.assertEqual(gap_lines, n_lines)
        self.assertIsNone(gapped["error"])
        self.assertEqual(gapped["header_lines"], spaced["header_lines"])
        self.assertEqual(
            [{k: r[k] for k in ("game_date", "game_time", "matchup", "team",
                                "player", "status", "reason")}
             for r in gapped["rows"]],
            [{k: r[k] for k in ("game_date", "game_time", "matchup", "team",
                                "player", "status", "reason")}
             for r in spaced["rows"]],
            "gap-spaced glyphs must reconstruct the same table as real spaces",
        )
        self.assertNothingDropped(gapped, gap_lines)


class TestCrossPageWrap(PdfAccountingMixin, unittest.TestCase):
    def test_a_reason_running_onto_the_next_page_joins_its_row(self):
        page1 = [
            (PAGE_H - 107.7, list(zip(ERA3_COLX, ERA3_HEADER))),
            era3_line(128.95, ["10/23/2026", "04:00 (ET)", "CHI@MIL",
                               "Chicago Bulls", "Doe, John", "Out",
                               "Injury/Illness - Left Foot"]),
        ]
        # Page 2 starts with the tail of that reason, too far from any row on
        # this page to be a local wrap, then carries its own rows.
        page2 = [
            era3_line(52.0, ["", "", "", "", "", "", "Bone Bruise"]),
            era3_line(82.6, ["", "", "", "", "Roe, Jane", "Out", "Rest"]),
            era3_line(104.6, ["", "", "", "", "Poe, Alex", "Probable",
                              "Soreness"]),
        ]
        result = gid.parse_injury_pdf(make_pdf([page1, page2]))

        rows = {r["player"]: r for r in result["rows"]}
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows["Doe, John"]["reason"],
                         "Injury/Illness - Left Foot Bone Bruise")
        self.assertEqual(rows["Roe, Jane"]["reason"], "Rest")
        self.assertEqual(result["continuation_lines"], 1)
        self.assertEqual(result["unparsed"], [])
        self.assertNothingDropped(result, 5)

    def test_two_line_cross_page_tail_joins_in_reading_order(self):
        # The tail of a reason can span TWO lines at the top of the next page;
        # both belong to the previous page's last row, in order.
        page1 = [
            (PAGE_H - 107.7, list(zip(ERA3_COLX, ERA3_HEADER))),
            era3_line(128.95, ["10/23/2026", "04:00 (ET)", "CHI@MIL",
                               "Chicago Bulls", "Doe, John", "Out",
                               "Injury/Illness - Left Foot"]),
        ]
        page2 = [
            era3_line(52.0, ["", "", "", "", "", "", "Bone Bruise"]),
            era3_line(59.0, ["", "", "", "", "", "", "Week To Week"]),
            era3_line(100.0, ["", "", "", "", "Roe, Jane", "Out", "Rest"]),
        ]
        result = gid.parse_injury_pdf(make_pdf([page1, page2]))

        rows = {r["player"]: r for r in result["rows"]}
        self.assertEqual(
            rows["Doe, John"]["reason"],
            "Injury/Illness - Left Foot Bone Bruise Week To Week",
        )
        self.assertEqual(rows["Roe, Jane"]["reason"], "Rest")
        self.assertEqual(result["continuation_lines"], 2)
        self.assertEqual(result["unparsed"], [])
        self.assertNothingDropped(result, 5)



class TestWrapRuns(PdfAccountingMixin, unittest.TestCase):
    """Wrap attribution must not hand a sub-line to a neighbouring row.

    The bug this pins: line-at-a-time nearest-row attribution gives the
    SECOND sub-line of a wrap below a row's identity line to the NEXT row
    (14pt from its own row vs ~8pt from the next in the real geometry) --
    silently truncating one player's reason and inflating another's. Wraps
    are attributed as RUNS (see `_process_page`), and prepends keep reading
    order.
    """

    def parse_rows(self, lines):
        result = gid.parse_injury_pdf(make_pdf([lines]))
        self.assertNothingDropped(result, len(lines))
        return result

    def test_two_sub_line_wrap_below_a_row_stays_with_its_row(self):
        lines = [
            (PAGE_H - 107.7, list(zip(ERA3_COLX, ERA3_HEADER))),
            era3_line(180.04, ["10/23/2026", "04:00 (ET)", "CHI@MIL",
                               "Chicago Bulls", "Doe, John", "Out", ""]),
            era3_line(187.06, ["", "", "", "", "", "", "Left Hamstring"]),
            # 14pt from its own row, 8pt from the next: nearest-line
            # attribution gives this to Roe -- and it is DOE's text.
            era3_line(194.08, ["", "", "", "", "", "", "Strain"]),
            era3_line(202.10, ["", "", "", "", "Roe, Jane", "Out", "Rest"]),
        ]
        rows = {r["player"]: r for r in self.parse_rows(lines)["rows"]}
        self.assertEqual(rows["Doe, John"]["reason"], "Left Hamstring Strain")
        self.assertEqual(rows["Roe, Jane"]["reason"], "Rest")

    def test_two_sub_line_wrap_above_a_row_keeps_reading_order(self):
        lines = [
            (PAGE_H - 107.7, list(zip(ERA3_COLX, ERA3_HEADER))),
            era3_line(150.98, ["10/23/2026", "04:00 (ET)", "CHI@MIL",
                               "Chicago Bulls", "Poe, Alex", "Available",
                               "Rest"]),
            # Closer to the row ABOVE than to the row below; both lines are
            # the reason of the row below, and must prepend in reading order.
            era3_line(164.00, ["", "", "", "", "", "", "Part One"]),
            era3_line(173.02, ["", "", "", "", "", "", "Part Two"]),
            era3_line(180.04, ["", "", "", "", "Doe, John", "Out", "Tail"]),
            era3_line(202.10, ["", "", "", "", "Roe, Jane", "Probable",
                               "Soreness"]),
        ]
        rows = {r["player"]: r for r in self.parse_rows(lines)["rows"]}
        self.assertEqual(rows["Poe, Alex"]["reason"], "Rest")
        self.assertEqual(rows["Doe, John"]["reason"], "Part One Part Two Tail")
        self.assertEqual(rows["Roe, Jane"]["reason"], "Soreness")

    def test_consecutive_two_line_bands_do_not_chain_across_rows(self):
        # The exact pattern from the live 2026-10-23 reports: every reason is
        # a sandwich -- part one ABOVE the identity line, part two BELOW --
        # and consecutive bands sit 22pt apart with 7pt sub-lines. Wrap lines
        # from ADJACENT bands are 22pt apart and must NOT chain into one run:
        # the first draft of run attribution used the median line gap, which
        # on a dense page is the ROW gap, and it glued every part-one onto the
        # row above. These coordinates are the real ones.
        lines = [
            (PAGE_H - 107.7, list(zip(ERA3_COLX, ERA3_HEADER))),
            era3_line(370.14, ["10/23/2026", "04:00 (ET)", "ORL@DET",
                               "Orlando Magic", "Black, Anthony", "Out",
                               "Injury/Illness - Right Ankle Sprain"]),
            era3_line(392.18, ["", "", "", "", "", "",
                               "Injury/Illness - Left Knee Injury"]),
            era3_line(399.20, ["", "", "", "", "Isaac, Jonathan", "Out", ""]),
            era3_line(406.22, ["", "", "", "", "", "", "Management"]),
            era3_line(428.26, ["", "", "", "", "", "",
                               "Injury/Illness - Left Ankle Injury"]),
            era3_line(435.28, ["", "", "", "", "Nelson, Izaiyah", "Out", ""]),
            era3_line(442.30, ["", "", "", "", "", "", "Recovery"]),
        ]
        rows = {r["player"]: r for r in self.parse_rows(lines)["rows"]}
        self.assertEqual(rows["Black, Anthony"]["reason"],
                         "Injury/Illness - Right Ankle Sprain")
        self.assertEqual(rows["Isaac, Jonathan"]["reason"],
                         "Injury/Illness - Left Knee Injury Management")
        self.assertEqual(rows["Nelson, Izaiyah"]["reason"],
                         "Injury/Illness - Left Ankle Injury Recovery")

    def test_equidistant_wrap_resolves_to_the_row_above(self):
        # The documented tie rule: 14pt from the row above, 14pt from the row
        # below. It attaches to the row ABOVE, and that choice is pinned --
        # an unpinned tie is a coin flip wearing a rule's clothes.
        lines = [
            (PAGE_H - 107.7, list(zip(ERA3_COLX, ERA3_HEADER))),
            era3_line(150.0, ["10/23/2026", "04:00 (ET)", "CHI@MIL",
                              "Chicago Bulls", "Doe, John", "Out", ""]),
            era3_line(164.0, ["", "", "", "", "", "", "Middle Part"]),
            era3_line(178.0, ["", "", "", "", "Roe, Jane", "Out", "Rest"]),
        ]
        rows = {r["player"]: r for r in self.parse_rows(lines)["rows"]}
        self.assertEqual(rows["Doe, John"]["reason"], "Middle Part")
        self.assertEqual(rows["Roe, Jane"]["reason"], "Rest")


class TestGapGlyphSpacing(PdfAccountingMixin, unittest.TestCase):
    """The gap-based word path must parse as well as real space glyphs do.

    Every fixture before this one wrote real space glyphs, so the branch that
    splits words on inter-glyph GAPS -- the only thing that works on the live
    2025-26 PDFs -- was executed by no test at all. This pins it on the
    compact fixture; TestRealEra3Layout pins it on the real layout.
    """

    def test_word_gaps_without_space_glyphs_match_spaced_text(self):
        rows = [
            game_row("Doe, John", "Out", "Right Knee Soreness"),
            game_row("Roe, Jane", "Questionable", "G League \u2013 Two-Way"),
        ]
        spaced_pdf, n_lines = fixture_pdf(rows)
        gapped_pdf, gap_lines = fixture_pdf(rows, gap_words=True)
        spaced = gid.parse_injury_pdf(spaced_pdf)
        gapped = gid.parse_injury_pdf(gapped_pdf)

        self.assertEqual(gap_lines, n_lines)
        self.assertIsNone(gapped["error"])
        self.assertEqual(gapped["header_lines"], 1,
                         "header detection is the gap-splitting path: if the "
                         "words merge, the table is never found")
        keys = ("game_date", "game_time", "matchup", "team", "player",
                "status", "reason")
        self.assertEqual(
            [{k: r[k] for k in keys} for r in gapped["rows"]],
            [{k: r[k] for k in keys} for r in spaced["rows"]],
        )
        self.assertEqual(gapped["rows"][1]["reason"], "G League \u2013 Two-Way")
        self.assertNothingDropped(gapped, gap_lines)


class TestParseResilience(PdfAccountingMixin, StoreTestCase):
    """One bad file must never wedge `parse` at the same snapshot forever.

    pdfplumber parses lazily, so page-content errors surface AFTER open();
    a missing raw file used to raise out of parse_snapshot; and parse_all had
    no per-snapshot guard. Any of those aborted every future run at the same
    id while capture kept appending behind it. The failure contract: record
    parse_status='failed' with the reason, keep whatever lines were
    extracted, and carry on with the rest of the batch.
    """

    def snapshot(self, url):
        return self.con.execute(
            'SELECT * FROM "injury_snapshots" WHERE "source_url" = ?', (url,)
        ).fetchone()

    def test_corrupt_pdf_is_failed_and_the_batch_continues(self):
        good1, _ = fixture_pdf([game_row("Doe, John", "Out", "Rest")])
        good2, _ = fixture_pdf([game_row("Roe, Jane", "Probable", "")])
        corrupt = b"%PDF-1.4\n" + b"\x00garbage-not-a-content-stream" * 64
        url1 = gid.injury_report_url(date(2026, 10, 23), 17, 0)
        url2 = gid.injury_report_url(date(2026, 10, 23), 17, 45)
        url3 = gid.injury_report_url(date(2026, 10, 23), 18, 0)
        for url, blob in ((url1, good1), (url2, corrupt), (url3, good2)):
            gid.store_fetch(self.con, outcome(url, blob), raw_dir=self.raw)

        totals = gid.parse_all(self.con, raw_dir=self.raw)

        self.assertEqual(totals["parsed"], 2, "the good snapshots must parse")
        self.assertEqual(totals["failed"], 1)
        self.assertIsNotNone(totals["first_error"])
        n_events = self.con.execute(
            'SELECT COUNT(*) FROM "injury_status_events"').fetchone()[0]
        self.assertEqual(n_events, 2)
        bad = self.snapshot(url2)
        self.assertEqual(bad["parse_status"], "failed")
        self.assertTrue(bad["parse_error"])

    def test_missing_raw_file_is_failed_and_the_batch_continues(self):
        good1, _ = fixture_pdf([game_row("Doe, John", "Out", "Rest")])
        good2, _ = fixture_pdf([game_row("Roe, Jane", "Probable", "")])
        url1 = gid.injury_report_url(date(2026, 10, 23), 17, 0)
        url2 = gid.injury_report_url(date(2026, 10, 23), 17, 45)
        for url, blob in ((url1, good1), (url2, good2)):
            gid.store_fetch(self.con, outcome(url, blob), raw_dir=self.raw)
        (self.raw / self.snapshot(url1)["raw_path"]).unlink()

        totals = gid.parse_all(self.con, raw_dir=self.raw)

        self.assertEqual(totals["failed"], 1)
        self.assertEqual(totals["parsed"], 1)
        gone = self.snapshot(url1)
        self.assertEqual(gone["parse_status"], "failed")
        self.assertIn("missing", gone["parse_error"])

    def test_tampered_raw_bytes_are_refused_not_parsed(self):
        original, _ = fixture_pdf([game_row("Doe, John", "Out", "Rest")])
        tampered, _ = fixture_pdf([game_row("Roe, Jane", "Out", "Not His Row")])
        url = gid.injury_report_url(date(2026, 10, 23), 17, 0)
        gid.store_fetch(self.con, outcome(url, original), raw_dir=self.raw)
        (self.raw / self.snapshot(url)["raw_path"]).write_bytes(tampered)

        result = gid.parse_snapshot(self.con, self.snapshot(url)["id"],
                                    raw_dir=self.raw)

        self.assertEqual(result["status"], "failed")
        self.assertIn("sha256", result["error"])
        n_events = self.con.execute(
            'SELECT COUNT(*) FROM "injury_status_events"').fetchone()[0]
        self.assertEqual(n_events, 0,
                         "bytes we did not hash must not become events under "
                         "a real snapshot id")

    def test_mid_document_failure_keeps_the_lines_it_extracted(self):
        page1, _ = build_page([game_row("Doe, John", "Out", "Rest")])
        page2, _ = build_page([game_row("Roe, Jane", "Probable", "")],
                              title="Injury Report continued")
        pdf = make_pdf([page1, page2])
        url = gid.injury_report_url(date(2026, 10, 23), 17, 0)
        gid.store_fetch(self.con, outcome(url, pdf), raw_dir=self.raw)

        # Fail AFTER page 2's lines are extracted but before they are
        # assembled: that is the shape of a lazy pdfplumber content-stream
        # error, and it is where lines could silently vanish.
        original = gid._process_page
        calls = []

        def flaky(entries, context, carry_row=None):
            calls.append(1)
            if len(calls) > 1:
                raise RuntimeError("simulated corrupt content stream")
            return original(entries, context, carry_row=carry_row)

        gid._process_page = flaky
        try:
            result = gid.parse_snapshot(self.con, self.snapshot(url)["id"],
                                        raw_dir=self.raw)
        finally:
            gid._process_page = original

        self.assertEqual(result["status"], "failed")
        n_events = self.con.execute(
            'SELECT COUNT(*) FROM "injury_status_events"').fetchone()[0]
        self.assertEqual(n_events, 1,
                         "lines extracted before the bad page are kept")
        reasons = [r[0] for r in self.con.execute(
            'SELECT "reason" FROM "injury_unparsed_rows"')]
        self.assertTrue(any("before the page failed" in r for r in reasons))
        self.assertTrue(any("failed to extract" in r for r in reasons))


class TestLabelConflicts(PdfAccountingMixin, StoreTestCase):
    def test_label_content_conflict_is_flagged_not_resolved_silently(self):
        pdf_a, n_lines = fixture_pdf([game_row("Roe, Jane", "Questionable", "Hip")])
        pdf_b, _ = fixture_pdf([game_row("Roe, Jane", "Out", "Hip")])
        self.assertNothingDropped(gid.parse_injury_pdf(pdf_a), n_lines)
        url = gid.injury_report_url(date(2026, 10, 23), 17, 45)
        gid.store_fetch(self.con, outcome(url, pdf_a), raw_dir=self.raw)
        conflict = gid.store_fetch(self.con, outcome(url, pdf_b),
                                   raw_dir=self.raw)
        self.assertEqual(conflict["status"], "conflict")
        gid.parse_all(self.con, raw_dir=self.raw)

        event = gid.status_known_as_of(
            self.con, "Roe, Jane", "2026-10-23", "2026-10-23T18:00:00 ET"
        )
        self.assertIn(event["status"], ("Questionable", "Out"))
        self.assertTrue(event["label_conflict"],
                        "two contents existed for one label: the evidence "
                        "layer disagreed and the caller must be told")
        self.assertEqual(len(event["label_conflict_variants"]), 1)
        self.assertEqual(event["label_conflict_variants"][0]["status"],
                         "Questionable" if event["status"] == "Out" else "Out")

    def test_single_content_has_no_conflict_flag(self):
        pdf, n_lines = fixture_pdf([game_row("Roe, Jane", "Out", "Hip")])
        self.assertNothingDropped(gid.parse_injury_pdf(pdf), n_lines)
        url = gid.injury_report_url(date(2026, 10, 23), 17, 45)
        gid.store_fetch(self.con, outcome(url, pdf), raw_dir=self.raw)
        gid.parse_all(self.con, raw_dir=self.raw)
        event = gid.status_known_as_of(
            self.con, "Roe, Jane", "2026-10-23", "2026-10-23T18:00:00 ET"
        )
        self.assertFalse(event["label_conflict"])
        self.assertEqual(event["label_conflict_variants"], [])


class StreamingResponse:
    """A slow-drip body: streams chunks with a delay, never finishes on time."""

    def __init__(self, chunks=1000, chunk=b"x" * 1024, delay=0.01):
        self.status_code = 200
        self.content = None
        self.headers = {}
        self._chunks = chunks
        self._chunk = chunk
        self._delay = delay
        self.closed = False

    def iter_content(self, chunk_size):
        for _ in range(self._chunks):
            time.sleep(self._delay)
            yield self._chunk

    def close(self):
        self.closed = True


class RedirectRequester:
    """Maps URL -> (status, body, headers); records every call."""

    def __init__(self, mapping):
        self.mapping = mapping
        self.calls = []

    def __call__(self, url, timeout):
        self.calls.append(url)
        status, body, headers = self.mapping.get(url, (404, b"", {}))
        return FakeResponse(status, body, headers)


class TestFetchCaps(unittest.TestCase):
    """The security audit's hostile-response caps (F-1/F-2/F-3).

    The CDN response is untrusted input. A body is streamed under a byte
    budget, redirects are followed manually and re-validated against the CDN
    host on every hop, and one wall-clock deadline covers the whole fetch --
    so a hostile response can neither exhaust memory, nor steer the client at
    an internal host (SSRF), nor stall the batch loop.
    """

    def setUp(self):
        self.url = gid.injury_report_url(date(2026, 10, 23), 17, 0)

    def test_content_length_over_the_cap_is_refused_before_reading(self):
        calls = []

        def oversized(url, timeout):
            calls.append(url)
            return FakeResponse(200, b"x" * 10,
                               headers={"Content-Length": "999999999"})

        with self.assertRaises(gid.FetchRefused) as ctx:
            gid.fetch_url(self.url, requester=oversized, sleep=no_sleep,
                          max_bytes=1024)
        self.assertIn("Content-Length", str(ctx.exception))
        self.assertEqual(len(calls), 1,
                         "refusals are not retried: one call, then stop")

    def test_streamed_body_over_the_budget_is_cut_off(self):
        def dripping(url, timeout):
            return StreamingResponse(chunks=64, chunk=b"x" * 1024, delay=0)

        with self.assertRaises(gid.FetchRefused) as ctx:
            gid.fetch_url(self.url, requester=dripping, sleep=no_sleep,
                          max_bytes=1024)
        self.assertIn("download cap", str(ctx.exception))

    def test_slow_drip_body_cannot_exceed_the_fetch_deadline(self):
        started = time.monotonic()

        def dripping(url, timeout):
            return StreamingResponse(chunks=100000, chunk=b"x" * 1024,
                                     delay=0.01)

        with self.assertRaises(gid.FetchRefused) as ctx:
            gid.fetch_url(self.url, requester=dripping, sleep=no_sleep,
                          deadline_seconds=0.05)
        self.assertIn("deadline", str(ctx.exception))
        self.assertLess(time.monotonic() - started, 2.0,
                        "the wall-clock deadline must stop the read, not "
                        "merely warn about it")

    def test_expired_deadline_refuses_without_any_request(self):
        requester = FakeRequester({self.url: (200, b"%PDF-1.4 x")})
        with self.assertRaises(gid.FetchRefused):
            gid.fetch_url(self.url, requester=requester, sleep=no_sleep,
                          deadline_seconds=0)
        self.assertEqual(requester.calls, [])

    def test_redirect_to_a_non_cdn_host_is_refused_and_never_requested(self):
        target = "http://169.254.169.254/latest/meta-data/"
        requester = RedirectRequester(
            {self.url: (302, b"", {"Location": target})}
        )
        with self.assertRaises(gid.FetchRefused) as ctx:
            gid.fetch_url(self.url, requester=requester, sleep=no_sleep)
        self.assertIn("No request was sent", str(ctx.exception))
        self.assertEqual(requester.calls, [self.url],
                         "the metadata host must never be dialed: the hop is "
                         "re-validated BEFORE the request, not after")

    def test_same_host_redirect_is_followed_and_recorded(self):
        final = gid.injury_report_url(date(2026, 10, 23), 17, 15)
        pdf = b"%PDF-1.4 redirected"
        requester = RedirectRequester({
            self.url: (302, b"", {"Location": final}),
            final: (200, pdf, {}),
        })
        outcome = gid.fetch_url(self.url, requester=requester, sleep=no_sleep)
        self.assertEqual(outcome["final_url"], final)
        self.assertEqual(outcome["content"], pdf)
        self.assertEqual(requester.calls, [self.url, final])

        with tempfile.TemporaryDirectory() as tmp:
            con = gid.open_db(Path(tmp) / "injury.sqlite")
            try:
                gid.store_fetch(con, outcome, raw_dir=Path(tmp) / "raw")
                row = con.execute(
                    'SELECT "source_url", "final_url" FROM "injury_snapshots"'
                ).fetchone()
                self.assertEqual(row["source_url"], self.url)
                self.assertEqual(row["final_url"], final,
                                 "the snapshot records where the bytes came "
                                 "FROM, not just the URL we asked for")
            finally:
                con.close()

    def test_redirect_chains_are_capped(self):
        hops = [gid.CDN_BASE + f"hop{i}.pdf" for i in range(5)]
        mapping = {self.url: (302, b"", {"Location": hops[0]})}
        for i, hop in enumerate(hops):
            nxt = hops[i + 1] if i + 1 < len(hops) else hops[-1]
            mapping[hop] = (302, b"", {"Location": nxt})
        requester = RedirectRequester(mapping)
        with self.assertRaises(gid.FetchRefused) as ctx:
            gid.fetch_url(self.url, requester=requester, sleep=no_sleep)
        self.assertIn("redirects", str(ctx.exception))
        self.assertEqual(len(requester.calls), 1 + gid.MAX_REDIRECT_HOPS)

    def test_redirect_hops_spend_request_budget(self):
        # --max-requests N means at most N OUTBOUND REQUESTS, period --
        # redirects included. This was the budget-multiplication hole.
        final = gid.injury_report_url(date(2026, 10, 23), 17, 15)
        requester = RedirectRequester({
            self.url: (302, b"", {"Location": final}),
            final: (200, b"%PDF-1.4 x", {}),
        })
        budget = gid.RequestBudget(1)
        with self.assertRaises(gid.RequestBudgetExhausted):
            gid.fetch_url(self.url, requester=requester, sleep=no_sleep,
                          budget=budget)
        self.assertEqual(len(requester.calls), 1,
                         "the redirect hop may not be dialed past the budget")
        self.assertEqual(budget.used, 1)

        # With room for the hop, the same chain succeeds on exactly 2.
        requester2 = RedirectRequester({
            self.url: (302, b"", {"Location": final}),
            final: (200, b"%PDF-1.4 x", {}),
        })
        budget2 = gid.RequestBudget(3)
        outcome = gid.fetch_url(self.url, requester=requester2,
                                sleep=no_sleep, budget=budget2)
        self.assertEqual(outcome["final_url"], final)
        self.assertEqual(budget2.used, 2)


class TestParseCaps(StoreTestCase):
    """F-1 parse side: a hostile PDF must not OOM an unattended batch.

    An OOM SIGKILL is the one failure the exception contract cannot catch,
    and parse_all's ordering means it would wedge every batch at the same
    snapshot forever -- so the caps refuse or skip explicitly and keep going.
    """

    def test_oversize_pdf_is_refused_not_opened(self):
        blob = b"%PDF-1.4\n" + b"\x00" * (gid.MAX_PDF_BYTES + 1)
        with self.assertRaises(gid.InjuryParseError) as ctx:
            gid.parse_injury_pdf(blob)
        self.assertIn("parse cap", str(ctx.exception))

    def test_page_count_over_the_cap_is_refused(self):
        page, _ = build_page([game_row("Doe, John", "Out", "Rest")])
        pdf = make_pdf([list(page) for _ in range(gid.MAX_PDF_PAGES + 1)])
        with self.assertRaises(gid.InjuryParseError) as ctx:
            gid.parse_injury_pdf(pdf)
        self.assertIn("page parse cap", str(ctx.exception))

    def test_over_glyph_page_is_recorded_and_other_pages_still_parse(self):
        # BOTH pages over the ceiling: every page is recorded as a marker,
        # nothing crashes, and the document verdict stays loud (nothing was
        # extracted, so 'no column header found' is the truth).
        page1, _ = build_page([game_row("Doe, John", "Out", "Rest")])
        page2, _ = build_page([game_row("Roe, Jane", "Available", "")],
                              title="Injury Report continued")
        pdf = make_pdf([page1, page2])

        original = gid.MAX_PAGE_GLYPHS
        gid.MAX_PAGE_GLYPHS = 5  # every page now exceeds the ceiling
        try:
            result = gid.parse_injury_pdf(pdf)
        finally:
            gid.MAX_PAGE_GLYPHS = original

        self.assertEqual(result["rows"], [])
        markers = [u for u in result["unparsed"]
                   if "glyph ceiling" in u["reason"]]
        self.assertEqual(len(markers), 2)
        self.assertEqual(len(result["unparsed"]), 2,
                         "the capped pages' lines are not extracted; the "
                         "markers ARE the record")
        self.assertIn("no column header", result["error"])

    def test_over_glyph_page_skips_only_that_page(self):
        # Page 1 is small (under the ceiling); page 2 carries five rows and
        # blows through it. The capped page is recorded; page 1 still parses.
        page1, _ = build_page([game_row("Doe, John", "Out", "Rest")])
        page2, _ = build_page(
            [game_row(f"Row{i}, Test", "Available", "Rest") for i in range(5)],
            title="Injury Report continued",
        )
        pdf = make_pdf([page1, page2])

        original = gid.MAX_PAGE_GLYPHS
        gid.MAX_PAGE_GLYPHS = 300
        try:
            result = gid.parse_injury_pdf(pdf)
        finally:
            gid.MAX_PAGE_GLYPHS = original

        self.assertEqual([r["player"] for r in result["rows"]], ["Doe, John"],
                         "other pages still parse")
        self.assertEqual(len([u for u in result["unparsed"]
                              if "glyph ceiling" in u["reason"]]), 1)

    def test_page_failure_marker_does_not_collide_with_the_next_page(self):
        # Regression: the marker used to be written at line_no + 1 WITHOUT
        # advancing the counter, so the next page's first line took the same
        # number and INSERT OR IGNORE silently dropped one of them. Both must
        # survive, with distinct line numbers.
        page1, _ = build_page([game_row("Doe, John", "Out", "Rest")])
        page2, _ = build_page([game_row("Roe, Jane", "Probable", "")],
                              title="Injury Report continued")
        page3, _ = build_page([game_row("Poe, Alex", "Available", "")],
                              title="Injury Report continued")
        pdf = make_pdf([page1, page2, page3])
        url = gid.injury_report_url(date(2026, 10, 23), 17, 0)
        gid.store_fetch(self.con, outcome(url, pdf), raw_dir=self.raw)

        original = gid._process_page
        calls = []

        def flaky(entries, context, carry_row=None):
            calls.append(1)
            if len(calls) == 2:  # exactly page 2 fails
                raise RuntimeError("simulated corrupt content stream")
            return original(entries, context, carry_row=carry_row)

        gid._process_page = flaky
        try:
            result = gid.parse_snapshot(self.con, self.snapshot_rows()[0]["id"],
                                        raw_dir=self.raw)
        finally:
            gid._process_page = original

        self.assertEqual(result["status"], "failed")
        line_nos = [r[0] for r in self.con.execute(
            'SELECT "line_no" FROM "injury_unparsed_rows" ORDER BY "line_no"')]
        self.assertEqual(len(line_nos), len(set(line_nos)),
                         "no two recorded lines may share a line number")
        self.assertEqual(result.get("duplicates", 0), 0,
                         "and nothing was dropped along the way")
        # page1 title(1), page2 title(4), page2 salvaged row(6), the
        # MARKER(7), page3 title(8): the marker and page 3's first line both
        # survive at distinct numbers.
        self.assertEqual(line_nos, [1, 4, 6, 7, 8])
        events = [r[0] for r in self.con.execute(
            'SELECT "player" FROM "injury_status_events"')]
        self.assertEqual(events, ["Doe, John", "Poe, Alex"])

    def test_duplicate_derived_rows_are_counted_not_silently_dropped(self):
        good, _ = fixture_pdf([game_row("Doe, John", "Out", "Rest")])
        url = gid.injury_report_url(date(2026, 10, 23), 17, 0)
        gid.store_fetch(self.con, outcome(url, good), raw_dir=self.raw)

        row_a = {"game_date": "2026-10-23", "game_time": "", "matchup": "A@B",
                 "team": "T", "player": "Doe, John", "status": "Out",
                 "reason": "", "line_no": 5}
        row_b = dict(row_a, player="Roe, Jane")
        fake_result = {"rows": [row_a, row_b], "unparsed": [], "furniture": [],
                       "header_lines": 1, "blank_lines": 0,
                       "continuation_lines": 0, "error": None}

        original = gid.parse_injury_pdf
        gid.parse_injury_pdf = lambda data: fake_result
        try:
            totals = gid.parse_all(self.con, raw_dir=self.raw)
        finally:
            gid.parse_injury_pdf = original

        self.assertEqual(totals["rows"], 2)
        self.assertEqual(totals["unique_violations"], 1,
                         "the duplicate is not written -- and the drop is "
                         "COUNTED in the summary, never silent")
        n_events = self.con.execute(
            'SELECT COUNT(*) FROM "injury_status_events"').fetchone()[0]
        self.assertEqual(n_events, 1)

    def test_record_failure_clears_rows_and_counts_together(self):
        good, _ = fixture_pdf([game_row("Doe, John", "Out", "Rest")])
        url = gid.injury_report_url(date(2026, 10, 23), 17, 0)
        gid.store_fetch(self.con, outcome(url, good), raw_dir=self.raw)
        snap_id = self.snapshot_rows()[0]["id"]
        gid.parse_snapshot(self.con, snap_id, raw_dir=self.raw)
        self.assertEqual(
            self.con.execute('SELECT COUNT(*) FROM "injury_status_events"'
                             ).fetchone()[0], 1)

        # Tamper, then --reparse: the failure path must clear derived rows
        # and counts TOGETHER, not leave old rows under zeroed counts.
        (self.raw / self.snapshot_rows()[0]["raw_path"]).write_bytes(b"%PDF-1.4 junk")
        result = gid.parse_snapshot(self.con, snap_id, raw_dir=self.raw,
                                    reparse=True)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            self.con.execute('SELECT COUNT(*) FROM "injury_status_events"'
                             ).fetchone()[0], 0)
        row = self.snapshot_rows()[0]
        self.assertEqual((row["parsed_rows"], row["unparsed_rows"]), (0, 0))


class TestPathContainment(StoreTestCase):
    """F-7: a database-sourced raw_path must never escape the raw directory."""

    def _tamper_raw_path(self, value):
        self.con.execute(
            'UPDATE "injury_snapshots" SET "raw_path" = ? WHERE "id" = ?',
            (value, self.snapshot_rows()[0]["id"]),
        )

    def test_parse_refuses_a_raw_path_that_escapes_the_raw_dir(self):
        good, _ = fixture_pdf([game_row("Doe, John", "Out", "Rest")])
        gid.store_fetch(self.con, outcome(self.url, good), raw_dir=self.raw)
        self._tamper_raw_path("../../etc/passwd")

        result = gid.parse_snapshot(self.con, self.snapshot_rows()[0]["id"],
                                    raw_dir=self.raw)

        self.assertEqual(result["status"], "failed")
        self.assertIn("path traversal", result["error"])
        n_events = self.con.execute(
            'SELECT COUNT(*) FROM "injury_status_events"').fetchone()[0]
        self.assertEqual(n_events, 0)

    def test_restore_refuses_a_raw_path_that_escapes_the_raw_dir(self):
        good, _ = fixture_pdf([game_row("Doe, John", "Out", "Rest")])
        gid.store_fetch(self.con, outcome(self.url, good), raw_dir=self.raw)
        self._tamper_raw_path("../evil.pdf")
        (self.base / "evil.pdf").unlink(missing_ok=True)

        with self.assertRaises(gid.InjuryStoreError) as ctx:
            gid.store_fetch(self.con, outcome(self.url, good), raw_dir=self.raw)
        self.assertIn("path traversal", str(ctx.exception))
        self.assertFalse((self.base / "evil.pdf").exists(),
                         "no byte may be written outside the raw directory")


if __name__ == "__main__":
    unittest.main()
