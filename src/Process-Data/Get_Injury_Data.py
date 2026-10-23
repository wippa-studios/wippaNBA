"""Official NBA injury reports: capture, parse, and point-in-time availability.

WHY THIS MODULE EXISTS
======================
The pre-game availability feature block needs to know what was knowable BEFORE
tip-off, not what turned out to be true afterwards. The NBA publishes an
official injury report PDF at a deterministic, immutable CDN URL:

    https://ak-static.cms.nba.com/referee/injury/Injury-Report_{YYYY-MM-DD}_{time}.pdf

Every one of those PDFs is a timestamped snapshot of the league's own
designations, and the URL is the timestamp. That makes "what did the market
know at 6 p.m. ET?" an answerable question: fetch the snapshot whose label
predates the cutoff and read it. This module is the data layer for that
question. It captures the PDFs, parses their tables, and stores both in a
snapshot store that supports point-in-time queries.

WHAT IT DOES
============
* URL construction (`injury_report_url`, `time_token`, `slots_for_date`): pure
  functions that speak all three filename eras (see below).
* Capture (`fetch_url`, `capture_days`): fetches snapshot PDFs with polite
  pacing, exponential backoff, a hard request budget, and resume support.
  Raw bytes are stored keyed by (source_url, sha256); a duplicate fetch is a
  no-op. Absent slots are RECORDED, not forgotten, so a resumed backfill does
  not re-probe them.
* Parse (`parse_injury_pdf`, `parse_snapshot`): PDF -> rows, era-defensive.
  Anything that does not parse as a row is RECORDED with its raw text and a
  reason, never dropped. This repository has a documented allergy to silent
  drops (docs/measurement_bugs.md) and an injury row that vanishes is exactly
  the kind of measurement bug that doc is about.
* Store (Data/injury.sqlite): append-only evidence layer. `injury_snapshots`
  records every fetch attempt (200 or 404, content hash, raw path), and
  `injury_status_events` records every parsed designation row. Both timestamps
  are stored: the PDF's own Eastern-Time label AND the UTC instant it denotes.
* Query (`status_known_as_of`): the latest status for a player in a game that
  was known at or before a cutoff instant -- "known at 6 p.m. ET" semantics.

THE THREE FILENAME ERAS
=======================
The filename format and the snapshot cadence both changed over the years.
The URL constructor handles all three, selected by the report date:

  1. Dec 2018 - 2020-21   2-3 snapshots/day, hour-only label
                          e.g. Injury-Report_2019-01-15_05PM.pdf
  2. 2021-22 - 2024-25    hourly, still hour-only
                          e.g. Injury-Report_2023-02-14_12AM.pdf
  3. 2025-26 - now        every 15 minutes, minute in the label
                          e.g. Injury-Report_2026-10-23_12_00AM.pdf

Era 1 and era 2 share a filename format and differ only in cadence, so the
format boundary that matters to the URL is 2025-10-21 (2025-26 opening night);
the 2021-10-19 boundary only affects `slots_for_date` density. The exact
boundary dates are the two constants below. They are the best available
approximation of when the league switched; if a recovered PDF proves a boundary
wrong, change the constant -- every consumer reads it from there.

Era 1 published only 2-3 snapshots per day and which slots those were is not
documented, so slot enumeration for that era deliberately probes the full
24-hour grid and RECORDS the absences. An absent slot is a measurement ("no
report was published at this hour"), not a gap in the download.

TIME HANDLING (DST INCLUDED)
============================
Filenames carry EASTERN-TIME wall-clock labels. Everything here stores both:

  snapshot_ts_et   the PDF's own label, wall time, e.g. "2026-11-01T01:00:00"
  snapshot_ts_utc  the instant that label denotes, fixed-width UTC

The conversion uses zoneinfo "America/New_York", so EDT/EST is handled by the
platform database rather than by a hand-rolled offset. Two labels are
ambiguous, and both are resolved deterministically and pinned by tests:

  * fall-back, 1 a.m. hour occurs twice -> the FIRST occurrence (fold=0,
    still EDT). A cutoff at that label counts the earlier instant.
  * spring-forward, 2 a.m. hour does not exist -> interpreted with the
    pre-transition (standard) offset, which lands the instant immediately
    AFTER the gap.

Naive datetimes are REJECTED on input, never assumed to be UTC. This is the
same rule as Get_Odds_Line_Movement.py and for the same reason: this project
has already paid for a naive-local-time bug once (docs/measurement_bugs.md).
Storage timestamps are fixed-width so lexicographic ordering is chronological.

LABEL TIME IS OPTIMISTIC BY THE UPLOAD LAG. The point-in-time cutoff compares
against the snapshot's LABEL (the report's own timestamp), but the PDF becomes
visible on the CDN a few minutes after its label. "Known at T" therefore
means "published under a label at or before T", which is a few minutes more
optimistic than "actually visible at T". Consumers that need strict
visibility semantics should cut a few minutes EARLIER than the label
boundary (a late scratch at T-5m can be visible at T+2m under a T label).

SNAPSHOT CADENCE SEMANTICS (what the snapshots mean)
====================================================
League-mandated designations land 5 p.m. local the day before the game; a
game-day report lands 11 a.m.-1 p.m. local; then updates keep landing through
tip-off. Late scratches happen after T-30m and only show up in the last
snapshots before the game. A point-in-time query at cutoff T therefore reads
the newest snapshot whose label is <= T -- and if the answer differs from the
final status, that is not a bug, that is the feature.

APPEND-ONLY, AND WHAT THAT MEANS HERE
=====================================
The EVIDENCE layer -- `injury_snapshots` and the raw PDFs under Data/injury_raw
-- is append-only. Nothing in this module issues a DELETE, a DROP or a rewrite
of a stored PDF. A re-fetch of identical bytes is a no-op; a re-fetch of
different bytes at the same URL (the CDN is documented as immutable, so this
is an anomaly) is recorded as an ADDITIONAL row with its own raw file and
reported loudly as a conflict, never silently overwritten.

The DERIVED layer -- `injury_status_events` and `injury_unparsed_rows` -- is a
regenerable view of the PDFs. `parse` skips snapshots that are already parsed;
`parse --reparse` replaces a snapshot's derived rows (and ONLY theirs) in one
transaction so that a parser improvement can be re-run over stored evidence.
That replacement is explicit and loud; the raw bytes are never touched.

RATE LIMITS, AND HOW TO NOT GET BLOCKED
=======================================
* Pacing: 5-10 seconds (random) between requests by default.
* Backoff: exponential on 429/5xx/transport errors, 3 attempts.
* Budget: `--max-requests` (default 20) caps HTTP requests per invocation, and
  the counter includes retries. A smoke test cannot become a bulk download by
  accident. Budget exhaustion is an explicit stop, not a failure: the run
  reports what it did and re-running it resumes.
* Absences: a 404 is recorded, and re-probed only while the SLOT is under
  ABSENT_RETRY_SECONDS old (measured from the report's label, not from the
  probe: upload lag is real and a 3-hourly batch must still catch it), or
  unconditionally with --retry-missing. Older absences are final.
* Future slots are never probed: a slot whose instant has not arrived yet has
  no PDF yet and recording it as absent would poison resumability.

HISTORICAL RECOVERY (NOT IMPLEMENTED HERE -- READ BEFORE BACKFILLING)
====================================================================
Beyond the live page, the Wayback CDX API enumerates archived PDFs back to
Dec 2018:

    https://web.archive.org/cdx/search/cdx?url=ak-static.cms.nba.com/referee/injury/*

Do NOT hammer that endpoint. It is a single public service, not a CDN. The
detached historical backfill this module was built for should enumerate slots
deterministically (this module's own URLs) and use CDX only for gap discovery,
politely. Nothing here calls CDX; capture talks only to ak-static.cms.nba.com.

PDF PARSING DEPENDENCY
======================
pdfplumber is imported LAZILY inside `parse_injury_pdf`, never at module
scope, for the same reason nba_client.py lazily imports nba_api: this module
must stay importable (and its URL/time/store layers testable) on a machine
without the PDF stack. See `parse_injury_pdf` for the error a missing
dependency produces.

INVOCATION
==========
    python -m src.Process-Data.Get_Injury_Data capture --date 2026-10-23 --max-requests 3
    python -m src.Process-Data.Get_Injury_Data capture --from 2026-10-01 --to 2026-10-23
    python -m src.Process-Data.Get_Injury_Data parse
    python -m src.Process-Data.Get_Injury_Data query --player "Doe, John" \
        --game 2026-10-23 --cutoff "2026-10-23T18:00:00-04:00"
    python -m src.Process-Data.Get_Injury_Data slots --date 2026-10-23

(`python src/Process-Data/Get_Injury_Data.py ...` works too; the directory
contains a hyphen, so tests load this file by path.)

No top-level call below fetches anything. `main()` fetches only what the
command line asked for, under the request budget it was given.
"""

import argparse
import contextlib
import hashlib
import os
import random
import re
import sqlite3
import statistics
import sys
import time
from datetime import date as date_cls, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

BASE_DIR = Path(__file__).resolve().parents[2]
# Same header as Get_Odds_Line_Movement.py: this directory has a hyphen, so it
# is not an importable package and this file can only ever be loaded by path or
# via `python -m src.Process-Data.Get_Injury_Data` (runpy resolves the hyphen;
# the `import` statement cannot). The sys.path insert mirrors the sibling
# module so both resolve repo-root imports the same way. noqa: E402 marks the
# imports above as deliberately being pure stdlib -- this module must import
# without pandas, without requests and without pdfplumber.
sys.path.insert(1, os.fspath(BASE_DIR))  # noqa: E402


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

DB_PATH = BASE_DIR / "Data" / "injury.sqlite"
RAW_DIR = BASE_DIR / "Data" / "injury_raw"

# Raw PDFs land at RAW_DIR / "<report_date>/<filename>". The stored `raw_path`
# is RELATIVE to the raw dir so the database stays portable across checkouts;
# parse resolves it against --raw-dir at read time.

# ---------------------------------------------------------------------------
# URLs and eras
# ---------------------------------------------------------------------------

CDN_BASE = "https://ak-static.cms.nba.com/referee/injury/"
FILENAME_TEMPLATE = "Injury-Report_{date}_{time}.pdf"

# Earliest archived reports (Wayback CDX). URL construction before this date
# is refused rather than guessed.
EARLIEST_REPORT_DATE = date_cls(2018, 12, 1)

# 2021-22 opening night: cadence goes from 2-3 snapshots/day to hourly. Same
# filename format, so this boundary only affects slot enumeration density.
HOURLY_ERA_START = date_cls(2021, 10, 19)

# 2025-26 opening night: filenames start carrying MINUTES ("12_00AM"). This is
# the filename-format boundary; below it the label is hour-only ("12AM").
MINUTE_ERA_START = date_cls(2025, 10, 21)

MINUTE_ERA = "minute"       # HH_MM + AM/PM, 15-minute cadence
HOUR_ERA = "hour"           # HH + AM/PM, hourly or sparser

DATE_FORMAT = "%Y-%m-%d"

# ---------------------------------------------------------------------------
# Time storage
# ---------------------------------------------------------------------------

# Fixed width on purpose: lexicographic ORDER BY must equal chronological
# order for the point-in-time query to be correct without parsing in Python.
# Same rule as Get_Odds_Line_Movement.py's CAPTURED_AT_FORMAT.
UTC_FORMAT = "%Y-%m-%dT%H:%M:%S+00:00"
ET_FORMAT = "%Y-%m-%dT%H:%M:%S"

ET_ZONE_NAME = "America/New_York"
_ET_ZONE = None  # resolved lazily; see _et_zone()


# ---------------------------------------------------------------------------
# Network policy
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT_SECONDS = 60
MIN_DELAY_SECONDS = 5.0
MAX_DELAY_SECONDS = 10.0
MAX_ATTEMPTS = 3
# Backoff between retry attempts grows as max_delay * 2**(attempt-1):
# 10s, then 20s. A 429 or a 5xx is the host asking for breathing room.
RETRY_BACKOFF_BASE_SECONDS = MAX_DELAY_SECONDS

# SECURITY AUDIT CAPS (F-1/F-2/F-3). The CDN is a static file host we control
# the URL space of; everything else about a response is untrusted. A hostile
# or corrupt response must not be able to exhaust memory or stall the batch
# loop -- an OOM SIGKILL or a wedged loop is the one failure the exception
# contract cannot catch.
#
#   * a body is READ AS A STREAM under a hard byte budget; a Content-Length
#     over the cap is refused before the first byte is read.
#   * redirects are followed MANUALLY (allow_redirects=False), at most
#     MAX_REDIRECT_HOPS hops, and every hop is re-validated against the
#     injury-report CDN host: no redirect may leave the CDN (SSRF), and no
#     request is ever sent to a host that fails the check.
#   * every hop -- including redirect hops -- spends one unit of the request
#     budget, so `--max-requests N` means at most N outbound requests, period.
#   * one wall-clock deadline covers the whole fetch (all attempts and hops),
#     so a slow-drip body cannot stall a batch past FETCH_DEADLINE_SECONDS.
MAX_DOWNLOAD_BYTES = 25 * 1024 * 1024
MAX_REDIRECT_HOPS = 3
FETCH_DEADLINE_SECONDS = 60

# Parse-side caps (F-1). Same threat model one layer down: a hostile PDF must
# not OOM the parser. Over-size documents are refused up front and recorded as
# failed; an over-glyph page is skipped with its reason recorded and the other
# pages still processed.
MAX_PDF_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 200
MAX_PAGE_GLYPHS = 500_000

# A recorded absence is re-probed while the SLOT is younger than this, not
# while the probe is: the gate is `now - et_label_to_utc(slot) <
# ABSENT_RETRY_SECONDS` (see `fetch_needed`). Upload lag is measured from the
# report's own label -- a 404 five minutes after the label is a PDF that has
# not landed yet, and it must be retried at ANY capture cadence, including a
# 3-hourly batch that would otherwise always find a cold window. Old slots
# become final immediately, so a historical backfill runs at full throughput
# instead of re-probing dead slots every pass.
ABSENT_RETRY_SECONDS = 2 * 3600

# The request budget. Small by default on purpose: a smoke test must not be
# able to become a bulk download by forgetting a flag. The detached historical
# backfill passes --max-requests explicitly.
DEFAULT_MAX_REQUESTS = 20

# ---------------------------------------------------------------------------
# Storage schema
# ---------------------------------------------------------------------------

# Sentinels: a recorded ATTEMPT with no stored PDF. Each distinct failure
# mode gets its own sentinel so "records every fetch outcome" is literally
# true at the row level, and so resume logic can treat them alike.
#   ""            a 404: the slot was probed and no report exists at it.
#   "not-a-pdf"   HTTP 200 whose body did not start with '%PDF' (CDN error
#                 page, HTML interstitial): probed, but not evidence.
# Both are REAL rows -- an absent slot is a measurement, not a gap -- and both
# make UNIQUE (source_url, sha256) structural, so a resumed backfill cannot
# double-record an attempt even across processes. (Transport failures after
# retries are NOT recorded: no HTTP outcome exists to record. They are
# reported and counted as errors, and the slot stays unrecorded, so the next
# run re-probes it.)
NO_CONTENT_SHA256 = ""
NOT_A_PDF_SHA256 = "not-a-pdf"
NO_CONTENT_SENTINELS = (NO_CONTENT_SHA256, NOT_A_PDF_SHA256)

SNAPSHOT_TABLE = "injury_snapshots"
EVENTS_TABLE = "injury_status_events"
UNPARSED_TABLE = "injury_unparsed_rows"

SNAPSHOT_SCHEMA = (
    'CREATE TABLE IF NOT EXISTS "' + SNAPSHOT_TABLE + '" ('
    '"id" INTEGER PRIMARY KEY AUTOINCREMENT,'
    '"source_url" TEXT NOT NULL,'
    '"filename" TEXT NOT NULL,'
    # The label the PDF itself carries: report date and time token from the
    # filename, both Eastern Time.
    '"report_date" TEXT NOT NULL,'
    '"report_time_label" TEXT NOT NULL,'
    '"snapshot_ts_et" TEXT NOT NULL,'
    '"snapshot_ts_utc" TEXT NOT NULL,'
    '"fetched_at" TEXT NOT NULL,'
    '"http_status" INTEGER NOT NULL,'
    '"sha256" TEXT NOT NULL,'
    '"raw_path" TEXT,'
    '"byte_size" INTEGER,'
    # Where the bytes ACTUALLY came from. Normally equal to source_url; after
    # a redirect it is the last hop, so the row records the true provenance
    # instead of only "the URL we asked for".
    '"final_url" TEXT,'
    # Derived annotations describing the CURRENT parse of this snapshot. They
    # are refreshed by `parse` and by `parse --reparse`; the columns are on the
    # snapshot row for reporting, and the derived rows themselves live in
    # injury_status_events / injury_unparsed_rows.
    '"parse_status" TEXT NOT NULL DEFAULT \'unparsed\','
    '"parsed_rows" INTEGER NOT NULL DEFAULT 0,'
    '"unparsed_rows" INTEGER NOT NULL DEFAULT 0,'
    '"parse_error" TEXT,'
    'UNIQUE ("source_url", "sha256")'
    ")"
)

SNAPSHOT_INDEX = (
    'CREATE INDEX IF NOT EXISTS "idx_injury_snapshots_ts" ON "'
    + SNAPSHOT_TABLE + '" ("snapshot_ts_utc")'
)

EVENTS_SCHEMA = (
    'CREATE TABLE IF NOT EXISTS "' + EVENTS_TABLE + '" ('
    '"id" INTEGER PRIMARY KEY AUTOINCREMENT,'
    '"snapshot_id" INTEGER NOT NULL,'
    '"line_no" INTEGER NOT NULL,'
    '"snapshot_ts_et" TEXT NOT NULL,'
    '"snapshot_ts_utc" TEXT NOT NULL,'
    '"game_date" TEXT NOT NULL,'
    '"game_time" TEXT,'
    '"matchup" TEXT NOT NULL,'
    '"team" TEXT NOT NULL,'
    '"player" TEXT NOT NULL,'
    # Casefolded, whitespace-collapsed player name: the point-in-time query
    # keys on this so "Doe, John" == "doe,  john". It is derived at write time
    # so the lookup is indexable and deterministic.
    '"player_key" TEXT NOT NULL,'
    '"status" TEXT NOT NULL,'
    '"reason" TEXT,'
    'UNIQUE ("snapshot_id", "line_no")'
    ")"
)

EVENTS_INDEX = (
    'CREATE INDEX IF NOT EXISTS "idx_injury_events_pit" ON "'
    + EVENTS_TABLE + '" ("player_key", "game_date", "snapshot_ts_utc")'
)

UNPARSED_SCHEMA = (
    'CREATE TABLE IF NOT EXISTS "' + UNPARSED_TABLE + '" ('
    '"id" INTEGER PRIMARY KEY AUTOINCREMENT,'
    '"snapshot_id" INTEGER NOT NULL,'
    '"line_no" INTEGER NOT NULL,'
    '"raw_text" TEXT NOT NULL,'
    '"reason" TEXT NOT NULL,'
    'UNIQUE ("snapshot_id", "line_no")'
    ")"
)

# parse_status values on the snapshot row.
PARSE_UNPARSED = "unparsed"
PARSE_PARSED = "parsed"
PARSE_PARSED_WITH_UNPARSED = "parsed_with_unparsed_rows"
PARSE_FAILED = "failed"
PARSE_MISSING = "missing"       # recorded attempt with no PDF (404 etc.)
PARSE_NOT_A_PDF = "not_a_pdf"   # 200 whose body was not a PDF


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InjuryDataError(RuntimeError):
    """Base for every failure this module raises, so callers can catch one type."""


class InjuryUrlError(InjuryDataError):
    """Raised for a date or label that cannot name a report URL."""


class InjuryFetchError(InjuryDataError):
    """Raised when a fetch fails. Never converted into an empty result."""


class InjuryStoreError(InjuryDataError):
    """Raised when a snapshot cannot be written to or read from the database."""


class InjuryParseError(InjuryDataError):
    """Raised when a PDF cannot be parsed at all (as opposed to row-level noise)."""


class RequestBudgetExhausted(InjuryDataError):
    """Raised inside a run when --max-requests has been spent. A stop signal,
    not a failure: the capture loop turns it into a clean, resumable exit."""


class FetchRefused(InjuryFetchError):
    """A response was refused on purpose and will be refused again: an
    over-cap body, an off-CDN redirect, too many hops, or an exhausted fetch
    deadline. NOT retryable -- retrying a policy refusal just repeats the
    hostile exchange -- so it bypasses the retry loop and surfaces as an
    error in the capture summary."""


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def _et_zone():
    """Resolve America/New_York once, and fail loudly if it is unavailable."""
    global _ET_ZONE
    if _ET_ZONE is None:
        try:
            from zoneinfo import ZoneInfo

            _ET_ZONE = ZoneInfo(ET_ZONE_NAME)
        except Exception as exc:  # pragma: no cover - environment failure
            raise InjuryDataError(
                f"cannot resolve the {ET_ZONE_NAME} timezone ({exc!r}). Injury "
                "report filenames are Eastern-Time labels and converting them "
                "without the platform tz database is exactly the naive-local-time "
                "bug this repository has already paid for once. Install tzdata "
                "(pip install tzdata) on platforms without system zoneinfo."
            ) from exc
    return _ET_ZONE


def et_label_to_utc(day, hour, minute=0):
    """Convert an Eastern-Time wall label (the filename's own time) to UTC.

    DST is delegated to zoneinfo. Deterministic on the two ambiguous labels:

      * fall-back, the repeated 1 a.m. hour resolves to the FIRST occurrence
        (fold=0, still EDT);
      * spring-forward, the missing 2 a.m. hour resolves with the
        pre-transition (standard) offset, i.e. immediately after the gap.

    Both outcomes are pinned by Tests/test_injury_data.py. See the module
    docstring.
    """
    _validate_hour_minute(hour, minute)
    if not isinstance(day, date_cls) or isinstance(day, datetime):
        raise InjuryUrlError(f"day must be a datetime.date, got {day!r}")
    wall = datetime(day.year, day.month, day.day, hour, minute, tzinfo=_et_zone())
    return wall.astimezone(timezone.utc)


def format_utc(moment):
    """Format a timezone-aware instant as fixed-width UTC storage text."""
    return _require_aware(moment).astimezone(timezone.utc).strftime(UTC_FORMAT)


def format_et_label(day, hour, minute=0):
    """Format a report label as fixed-width Eastern wall time (no offset)."""
    et_label_to_utc(day, hour, minute)  # validates
    return datetime(day.year, day.month, day.day, hour, minute).strftime(ET_FORMAT)


def utcnow():
    """The current instant, timezone-aware UTC."""
    return datetime.now(timezone.utc)


def report_date_for_instant(moment):
    """The report date a UTC instant falls on IN EASTERN TIME.

    Not `moment.date()`: at 23:30 ET the UTC date is already tomorrow, and
    using the UTC date would enumerate the wrong day's slots. DST-safe by
    construction (the zone database does the arithmetic).
    """
    return _require_aware(moment).astimezone(_et_zone()).date()


def _require_aware(moment):
    """Return `moment` as an aware UTC datetime, or raise. Naive is rejected."""
    if isinstance(moment, str):
        text = moment[:-1] + "+00:00" if moment.endswith(("Z", "z")) else moment
        try:
            moment = datetime.fromisoformat(text)
        except ValueError as exc:
            raise InjuryDataError(
                f"could not parse timestamp {moment!r}; expected ISO-8601 with a "
                "UTC offset, e.g. 2026-10-23T18:00:00-04:00"
            ) from exc
    if not isinstance(moment, datetime):
        raise InjuryDataError(
            f"timestamp must be a datetime or an ISO-8601 string, got "
            f"{type(moment).__name__}"
        )
    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        raise InjuryDataError(
            f"naive timestamp {moment!r} rejected. Timestamps must carry an "
            "offset. This pipeline was audited for mixing naive local time with "
            "UTC schedule data; do not repeat it (docs/measurement_bugs.md)."
        )
    return moment.astimezone(timezone.utc)


def parse_cutoff(value):
    """Parse a point-in-time cutoff into an aware UTC datetime.

    Accepts:
      * an aware datetime (converted to UTC);
      * an ISO-8601 string with an offset or a trailing Z;
      * "YYYY-MM-DDTHH:MM[:SS] ET" -- the report's own label convention, parsed
        as America/New_York so "known at 6 p.m. ET" means exactly that across
        DST.

    A naive datetime or offset-less string is REJECTED, not assumed to be
    anything. The caller must say which clock they mean.
    """
    if isinstance(value, datetime):
        return _require_aware(value)
    if not isinstance(value, str):
        raise InjuryDataError(
            f"cutoff must be a datetime or a string, got {type(value).__name__}"
        )
    text = value.strip()
    if text.upper().endswith(" ET"):
        stamp = text[:-3].strip()
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S",
                    "%Y-%m-%d %H:%M"):
            try:
                wall = datetime.strptime(stamp, fmt)
            except ValueError:
                continue
            return wall.replace(tzinfo=_et_zone()).astimezone(timezone.utc)
        raise InjuryDataError(
            f"could not parse ET cutoff {value!r}; expected e.g. "
            "'2026-10-23T18:00:00 ET'"
        )
    return _require_aware(text)


# ---------------------------------------------------------------------------
# URL construction (pure)
# ---------------------------------------------------------------------------


def _validate_hour_minute(hour, minute):
    if not isinstance(hour, int) or isinstance(hour, bool) or not 0 <= hour <= 23:
        raise InjuryUrlError(f"hour must be an int in 0..23, got {hour!r}")
    if not isinstance(minute, int) or isinstance(minute, bool) or not 0 <= minute <= 59:
        raise InjuryUrlError(f"minute must be an int in 0..59, got {minute!r}")


def era_for_date(day):
    """HOUR_ERA or MINUTE_ERA for a report date. Pure.

    Era 1 and era 2 share the hour-only filename format; only the cadence
    differs and that is `slots_for_date`'s concern, not this function's.
    """
    if not isinstance(day, date_cls) or isinstance(day, datetime):
        raise InjuryUrlError(f"day must be a datetime.date, got {day!r}")
    if day < EARLIEST_REPORT_DATE:
        raise InjuryUrlError(
            f"{day.isoformat()} is before {EARLIEST_REPORT_DATE.isoformat()}, the "
            "earliest injury report the NBA published (Wayback CDX archives start "
            "Dec 2018). There is no URL to construct."
        )
    return MINUTE_ERA if day >= MINUTE_ERA_START else HOUR_ERA


def time_token(day, hour, minute=0):
    """The filename time token for a report label, e.g. '05PM' or '12_00AM'.

    Hour-only era: minute MUST be 0 and the token is 'HHAM'/'HHPM' with the
    hour zero-padded to two digits (12-hour clock: 00:00 -> '12AM',
    12:00 -> '12PM', 13:05 -> '01_05PM' only in the minute era).

    Minute era: the token is 'HH_MMAM'/'HH_MM PM' with both parts zero-padded.

    Strict on purpose. A rounded label would construct a URL that looks right
    and is wrong; there is exactly one URL per published snapshot.
    """
    _validate_hour_minute(hour, minute)
    era = era_for_date(day)
    suffix = "AM" if hour < 12 else "PM"
    hour12 = hour % 12 or 12
    if era == MINUTE_ERA:
        return f"{hour12:02d}_{minute:02d}{suffix}"
    if minute != 0:
        raise InjuryUrlError(
            f"{day.isoformat()} is in the hour-only filename era "
            f"(before {MINUTE_ERA_START.isoformat()}): the label carries no "
            f"minutes, but minute={minute} was requested. Era 1 and era 2 "
            "snapshots are named like '05PM', not '05_07PM'."
        )
    return f"{hour12:02d}{suffix}"


def injury_report_url(day, hour, minute=0):
    """The immutable CDN URL of one snapshot. Pure; the ONLY url constructor."""
    token = time_token(day, hour, minute)
    filename = FILENAME_TEMPLATE.format(date=day.isoformat(), time=token)
    return CDN_BASE + filename


_FILENAME_MINUTE_RE = re.compile(
    r"^Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{1,2})_(\d{2})(AM|PM)\.pdf$", re.I
)
_FILENAME_HOUR_RE = re.compile(
    r"^Injury-Report_(\d{4}-\d{2}-\d{2})_(\d{1,2})(AM|PM)\.pdf$", re.I
)


def parse_report_filename(name):
    """Inverse of `injury_report_url`'s filename. Era-defensive.

    Accepts both filename eras regardless of the date they carry: recovered
    PDFs from the Wayback CDX predate our era constants and parsing is where
    drift is TOLERATED (and recorded); construction is where it is not.

    Returns a dict: report_date (date), hour (int, 24h), minute (int),
    time_label (the raw token as it appears, uppercased), filename.
    Raises InjuryUrlError on anything that is not a report filename.
    """
    base = str(name).strip().split("/")[-1]
    match = _FILENAME_MINUTE_RE.match(base)
    has_minute = match is not None
    if match:
        date_text, hour_text, minute_text, suffix = match.groups()
        minute = int(minute_text)
    else:
        match = _FILENAME_HOUR_RE.match(base)
        if not match:
            raise InjuryUrlError(
                f"{base!r} is not an injury report filename; expected "
                "'Injury-Report_YYYY-MM-DD_HH AM|PM.pdf' or "
                "'Injury-Report_YYYY-MM-DD_HH_MM AM|PM.pdf'"
            )
        date_text, hour_text, suffix = match.groups()
        minute = 0
    try:
        day = datetime.strptime(date_text, DATE_FORMAT).date()
    except ValueError as exc:
        raise InjuryUrlError(f"{base!r}: {date_text!r} is not a YYYY-MM-DD date") from exc
    hour12 = int(hour_text)
    if not 1 <= hour12 <= 12:
        raise InjuryUrlError(f"{base!r}: hour {hour_text!r} is not 1..12")
    hour = hour12 % 12 + (12 if suffix.upper() == "PM" else 0)
    if has_minute:
        time_label = f"{int(hour_text):02d}_{minute:02d}{suffix.upper()}"
    else:
        time_label = f"{int(hour_text):02d}{suffix.upper()}"
    return {
        "report_date": day,
        "hour": hour,
        "minute": minute,
        "time_label": time_label,
        "filename": base,
    }


def slots_for_date(day, slots=None):
    """The candidate snapshot labels (hour, minute) for one report date.

    Era 3 (minute era): the 96 fifteen-minute slots.
    Era 1 and era 2: the 24 hour slots. Era 1 published only 2-3 of these per
    day, so most are absences by design -- enumeration probes the full grid and
    RECORDS the absences rather than guessing which hours the league used.

    `slots` optionally restricts the result to an explicit list of (hour,
    minute) pairs -- an operator optimising a sparse-era backfill.

    Returned in chronological order (midnight first).
    """
    era = era_for_date(day)
    if slots is None:
        if era == MINUTE_ERA:
            return [(h, m) for h in range(24) for m in (0, 15, 30, 45)]
        return [(h, 0) for h in range(24)]
    chosen = []
    for hour, minute in slots:
        _validate_hour_minute(hour, minute)
        if era == HOUR_ERA and minute != 0:
            raise InjuryUrlError(
                f"slot {hour:02d}:{minute:02d} names a minute but "
                f"{day.isoformat()} is in the hour-only filename era"
            )
        chosen.append((hour, minute))
    return sorted(set(chosen))


def candidate_urls(day, slots=None):
    """(slot, url) pairs for one report date, chronological. Pure."""
    return [(slot, injury_report_url(day, slot[0], slot[1]))
            for slot in slots_for_date(day, slots=slots)]


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------


class RequestBudget:
    """A hard cap on HTTP requests, counted per ATTEMPT including retries.

    Deliberately tiny API: `spend()` returns False when the budget is gone.
    The point is that no code path in this module can issue a request the
    caller did not budget for.
    """

    def __init__(self, max_requests):
        if not isinstance(max_requests, int) or isinstance(max_requests, bool):
            raise InjuryFetchError(f"--max-requests must be an int, got {max_requests!r}")
        if max_requests < 1:
            raise InjuryFetchError(
                f"--max-requests must be at least 1, got {max_requests}"
            )
        self.max_requests = max_requests
        self.used = 0

    @property
    def remaining(self):
        return max(0, self.max_requests - self.used)

    def spend(self):
        if self.remaining == 0:
            return False
        self.used += 1
        return True


def _default_requester(url, timeout):
    """One request with redirects OFF and the body NOT read.

    `allow_redirects=False`: redirects are followed manually in
    `_get_with_redirects`, because every hop must be re-validated against the
    CDN host before a request is sent. `stream=True`: the body is read as a
    stream under a byte budget instead of `response.content`, which would
    happily buffer a hostile gigabyte into memory first and check later.
    Imported here so the module stays light.
    """
    import requests

    return requests.get(url, timeout=timeout, allow_redirects=False, stream=True)


def _response_headers(response):
    return getattr(response, "headers", None) or {}


def _close_response(response):
    closer = getattr(response, "close", None)
    if callable(closer):
        with contextlib.suppress(Exception):
            closer()


def _validate_fetch_url(url, original):
    """Refuse any URL that is not the injury-report CDN. SSRF guard.

    Applied to the ORIGINAL url and to every redirect hop BEFORE the request
    is sent: a redirect may only target the same https CDN host. An off-CDN
    Location is refused with no request leaving this machine.
    """
    parts = urlsplit(url)
    expected = urlsplit(CDN_BASE)
    if parts.scheme != expected.scheme or parts.netloc != expected.netloc:
        raise FetchRefused(
            f"refusing to fetch {url!r} (requested {original!r}): every URL, "
            f"including every redirect hop, must stay on {expected.scheme}://"
            f"{expected.netloc} -- the injury-report CDN. No request was sent."
        )


def _check_deadline(deadline, url):
    if deadline is not None and time.monotonic() >= deadline:
        raise FetchRefused(
            f"fetch deadline exceeded while reading {url}: the response is "
            "stalled or dripping and the wall-clock budget for this fetch is "
            "gone. Refusing to continue."
        )


def _read_body_capped(response, url, max_bytes, deadline):
    """Read the body as a stream under a hard byte budget.

    Content-Length over the cap is refused before the first byte; without a
    Content-Length the stream is cut off at the cap (a chunk past the budget
    raises). The deadline is checked between chunks so a slow-drip body cannot
    stall the batch loop. The response is closed either way.
    """
    headers = _response_headers(response)
    declared = headers.get("Content-Length") or headers.get("content-length")
    if declared is not None:
        try:
            if int(declared) > max_bytes:
                raise FetchRefused(
                    f"{url} declares Content-Length {declared}, over the "
                    f"{max_bytes}-byte download cap. Refusing before reading "
                    "a byte: this is not a plausible injury report."
                )
        except ValueError:
            pass  # a non-numeric length is the host's problem; the stream cap holds
    try:
        iter_content = getattr(response, "iter_content", None)
        if callable(iter_content):
            chunks = []
            total = 0
            for chunk in iter_content(chunk_size=65536):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    raise FetchRefused(
                        f"{url} exceeds the {max_bytes}-byte download cap "
                        f"after {total} bytes. Refusing: the stream was cut "
                        "off and nothing is kept."
                    )
                chunks.append(chunk)
                _check_deadline(deadline, url)
            return b"".join(chunks)
        body = response.content if response.content is not None else b""
        if len(body) > max_bytes:
            raise FetchRefused(
                f"{url} body is {len(body)} bytes, over the {max_bytes}-byte "
                "download cap. Refusing: nothing is kept."
            )
        return bytes(body)
    finally:
        _close_response(response)


def _get_with_redirects(url, requester, timeout, budget, deadline,
                        max_bytes=MAX_DOWNLOAD_BYTES):
    """One GET with MANUAL redirect following. Returns (final_url, status, body).

    Every hop: the target is re-validated against the CDN host, spends one
    unit of request budget (--max-requests N means at most N outbound
    requests, hops included), and is checked against the deadline. At most
    MAX_REDIRECT_HOPS redirects are followed before giving up. An off-CDN
    hop raises FetchRefused without sending anything.
    """
    current = url
    for _hop in range(MAX_REDIRECT_HOPS + 1):
        _validate_fetch_url(current, url)
        if budget is not None and not budget.spend():
            raise RequestBudgetExhausted(
                f"request budget spent before {current} could be fetched"
            )
        _check_deadline(deadline, current)
        response = requester(current, timeout)
        status = int(response.status_code)
        headers = _response_headers(response)
        location = headers.get("Location") or headers.get("location")
        if status in (301, 302, 303, 307, 308) and location:
            _close_response(response)
            # Relative Locations resolve against the current URL; the next
            # hop re-validates the result before any request is sent.
            current = urljoin(current, location)
            continue
        body = _read_body_capped(response, current, max_bytes, deadline)
        return current, status, body
    raise FetchRefused(
        f"more than {MAX_REDIRECT_HOPS} redirects fetching {url} "
        f"(last target {current!r}): giving up rather than following an "
        "endless chain."
    )


def fetch_url(url, requester=None, timeout=REQUEST_TIMEOUT_SECONDS, budget=None,
              sleep=time.sleep, max_attempts=MAX_ATTEMPTS,
              deadline_seconds=FETCH_DEADLINE_SECONDS,
              max_bytes=MAX_DOWNLOAD_BYTES):
    """Fetch one URL. Returns a fetch outcome dict. Never writes anything.

    outcome keys: url (the URL we asked for), final_url (the URL the bytes
    actually came from, after any redirects), fetched_at, http_status, sha256,
    content, content_type, attempts, delays. `sha256` is NO_CONTENT_SHA256
    when no PDF is returned; `content` is then None.

    Retry policy: 429/5xx and transport failures are retryable (the host
    asking for breathing room, or a network wobble -- neither says anything
    about the data). 404 is FINAL for this call: the URL is immutable, so a
    404 is "no snapshot was published here", recorded as an absence.
    FetchRefused (over-cap body, off-CDN redirect, too many hops, deadline)
    is NOT retryable: it bypasses the loop and surfaces as an error.

    Every outbound request -- retries and redirect hops included -- spends one
    unit of `budget` first, so `--max-requests N` is a hard cap on requests,
    not on fetches. `deadline_seconds` is a wall-clock budget for the WHOLE
    call (all attempts and hops), so a slow-drip body cannot stall a batch.
    Retries back off exponentially from RETRY_BACKOFF_BASE_SECONDS (10s, then
    20s). `sleep` is injectable so tests never wait.
    """
    get = requester if requester is not None else _default_requester
    deadline = (time.monotonic() + deadline_seconds
                if deadline_seconds is not None else None)
    delays = []
    attempts = 0
    last_error = None
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        try:
            final_url, status, body = _get_with_redirects(
                url, get, timeout, budget, deadline, max_bytes=max_bytes
            )
        except (RequestBudgetExhausted, FetchRefused):
            raise
        except Exception as exc:
            last_error = f"transport error: {exc!r}"
            _check_deadline(deadline, url)  # no retry past the fetch deadline
            if attempt == max_attempts:
                raise InjuryFetchError(
                    f"{url} failed after {attempts} attempt(s): {last_error}"
                ) from exc
            delay = RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            delays.append(delay)
            sleep(delay)
            continue

        if status == 404:
            return {
                "url": url, "final_url": final_url,
                "fetched_at": format_utc(utcnow()), "http_status": 404,
                "sha256": NO_CONTENT_SHA256, "content": None,
                "content_type": None, "attempts": attempts, "delays": delays,
            }
        if status == 200:
            return {
                "url": url, "final_url": final_url,
                "fetched_at": format_utc(utcnow()), "http_status": 200,
                "sha256": hashlib.sha256(body).hexdigest(), "content": body,
                "content_type": None, "attempts": attempts, "delays": delays,
            }
        # Any other status (429, 5xx, unexpected 4xx) is retryable: it is
        # either the host asking for space or a proxy in the way.
        last_error = f"HTTP {status}"
        if attempt == max_attempts:
            raise InjuryFetchError(
                f"{url} failed after {attempts} attempt(s): {last_error}"
            )
        delay = RETRY_BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
        delays.append(delay)
        sleep(delay)
    raise InjuryFetchError(f"{url} failed after {attempts} attempt(s): {last_error}")


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def ensure_schema(con):
    """Create the injury tables and indexes if they do not exist.

    Additive only. Nothing here drops, alters or rewrites existing rows.
    """
    try:
        con.execute(SNAPSHOT_SCHEMA)
        con.execute(SNAPSHOT_INDEX)
        con.execute(EVENTS_SCHEMA)
        con.execute(EVENTS_INDEX)
        con.execute(UNPARSED_SCHEMA)
        existing = [row[1] for row in con.execute(
            'PRAGMA table_info("' + SNAPSHOT_TABLE + '")'
        )]
        if existing and "final_url" not in existing:
            # Additive migration, mirroring Get_Odds_Line_Movement's
            # Provenance column: a database written before the security audit
            # would otherwise fail every insert with "no column named
            # final_url". Old rows keep NULL ("no redirect known").
            con.execute(
                'ALTER TABLE "' + SNAPSHOT_TABLE + '" ADD COLUMN "final_url" TEXT'
            )
    except sqlite3.Error as exc:
        raise InjuryStoreError(f"could not create injury tables: {exc}") from exc
    return con


def open_db(db_path=None, readonly=False):
    """Open (and unless readonly, initialise) the injury database."""
    target = DB_PATH if db_path is None else Path(db_path)
    try:
        if readonly:
            if not target.exists():
                raise InjuryStoreError(
                    f"injury database {target} does not exist yet. Run "
                    "`capture` (and `parse`) before `query`."
                )
            con = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            con = sqlite3.connect(target)
    except sqlite3.Error as exc:
        raise InjuryStoreError(f"could not open {target}: {exc}") from exc
    con.row_factory = sqlite3.Row
    if not readonly:
        ensure_schema(con)
    return con


def _sha_suffix(filename, sha256):
    """Filename variant for a content conflict: '<stem>__<sha12><ext>'."""
    stem, ext = os.path.splitext(filename)
    return f"{stem}__{sha256[:12]}{ext}"


def _write_raw_file(raw_dir, relative, content):
    """Write raw bytes atomically. Returns the path written."""
    target = Path(raw_dir) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(content)
    os.replace(tmp, target)
    return target


def _contained_raw_path(raw_dir, relative):
    """Resolve a DATABASE-SOURCED raw_path under raw_dir, or None if it escapes.

    Paths we write ourselves at ingest are anchored (report_date/filename from
    a validated URL). This check is for values READ BACK from the database at
    use time: a tampered raw_path must never turn a restore or a parse into an
    arbitrary-file write or read (F-7, path traversal).
    """
    base = Path(raw_dir).resolve()
    target = (Path(raw_dir) / relative).resolve()
    return target if target.is_relative_to(base) else None


def _url_has_stored_content(con, url):
    """True when this URL already has at least one stored PDF version."""
    try:
        row = con.execute(
            'SELECT COUNT(*) AS n FROM "' + SNAPSHOT_TABLE + '"'
            ' WHERE "source_url" = ? AND "sha256" NOT IN (?, ?)',
            (url, NO_CONTENT_SHA256, NOT_A_PDF_SHA256),
        ).fetchone()
    except sqlite3.Error as exc:
        raise InjuryStoreError(f"could not read {SNAPSHOT_TABLE} for {url}: {exc}") from exc
    return row["n"] > 0


def store_fetch(con, outcome, raw_dir=None):
    """Record one fetch outcome. Idempotent: duplicate (url, sha256) is a no-op.

    Returns a dict: status in {'inserted', 'skipped', 'restored', 'conflict',
    'absent', 'not_a_pdf'} and snapshot_id (row id).

      inserted   new content or new absence
      skipped    identical (url, sha256) already recorded -> nothing written
      restored   identical content was recorded but its file was missing on
                 disk and has been rewritten (explicit repair, loudly counted)
      conflict   the same URL returned DIFFERENT bytes. The CDN is documented
                 immutable, so this is an anomaly: the new version is recorded
                 as an additional row with its own hash-suffixed raw file and
                 the caller reports it. Nothing is overwritten.
      absent     a 404: recorded with sha256 NO_CONTENT_SHA256 so a resumed
                 run does not re-probe the slot.
      not_a_pdf  HTTP 200 whose body does not start with '%PDF'. Recorded with
                 sha256 NOT_A_PDF_SHA256 (nothing is written to disk): a CDN
                 error page is not evidence and is retried like an upload lag.

    Every HTTP outcome gets exactly one row, with distinct sentinels for the
    two content-less outcomes -- 404 and not-a-PDF are different measurements
    and must not share a row. (A transport failure after retries has no HTTP
    outcome to record: it is reported as an error and the slot stays
    unrecorded, so the next run re-probes it.)

    The content hash is computed HERE from the bytes, not trusted from the
    caller: the (source_url, sha256) key must describe the stored file, so the
    stored file is what gets hashed.

    The evidence layer is append-only: this function never deletes or rewrites
    a stored PDF or a snapshot row.
    """
    raw_dir = RAW_DIR if raw_dir is None else Path(raw_dir)
    url = outcome["url"]
    parsed = parse_report_filename(url)
    filename = parsed["filename"]  # the URL's own basename: exact, not rebuilt
    content = outcome.get("content")

    is_pdf = content is not None and bytes(content).startswith(b"%PDF")
    if content is not None and not is_pdf:
        # A 200 that is not a PDF (CDN error page, HTML interstitial). Do not
        # keep it as evidence -- it is not the report -- but DO record the
        # attempt so resumability works exactly like a 404.
        content = None
        parse_status = PARSE_NOT_A_PDF
    elif content is not None:
        parse_status = PARSE_UNPARSED
    else:
        parse_status = PARSE_MISSING

    if content is not None:
        sha256 = hashlib.sha256(bytes(content)).hexdigest()
    elif parse_status == PARSE_NOT_A_PDF:
        sha256 = NOT_A_PDF_SHA256
    else:
        sha256 = NO_CONTENT_SHA256

    report_date = parsed["report_date"]
    row = {
        "source_url": url,
        "filename": filename,
        "report_date": report_date.isoformat(),
        "report_time_label": parsed["time_label"],
        "snapshot_ts_et": format_et_label(report_date, parsed["hour"], parsed["minute"]),
        "snapshot_ts_utc": format_utc(
            et_label_to_utc(report_date, parsed["hour"], parsed["minute"])
        ),
        "fetched_at": outcome.get("fetched_at") or format_utc(utcnow()),
        "http_status": int(outcome["http_status"]),
        "sha256": sha256,
        "raw_path": None,
        "byte_size": None,
        "final_url": outcome.get("final_url") or url,
        "parse_status": parse_status,
    }

    try:
        with con:
            existing = con.execute(
                'SELECT "id", "raw_path" FROM "' + SNAPSHOT_TABLE + '"'
                ' WHERE "source_url" = ? AND "sha256" = ?',
                (url, sha256),
            ).fetchone()
            if existing is not None:
                snapshot_id = existing["id"]
                if content is None:
                    return {"status": "skipped", "snapshot_id": snapshot_id}
                stored = existing["raw_path"]
                # Recorded but the bytes may be gone from disk: rewrite them.
                # The stored relative path is authoritative; if it is somehow
                # unset, fall back to the canonical location. It comes FROM
                # THE DATABASE, so it is containment-checked before any use.
                relative = stored or str(Path(row["report_date"]) / filename)
                target = _contained_raw_path(raw_dir, relative)
                if target is None:
                    raise InjuryStoreError(
                        f"stored raw_path {relative!r} for snapshot "
                        f"{snapshot_id} escapes the raw directory "
                        f"({raw_dir}): refusing to touch it (path traversal)"
                    )
                if target.exists():
                    return {"status": "skipped", "snapshot_id": snapshot_id}
                _write_raw_file(raw_dir, relative, bytes(content))
                con.execute(
                    'UPDATE "' + SNAPSHOT_TABLE + '" SET "raw_path" = ?,'
                    ' "byte_size" = ? WHERE "id" = ?',
                    (str(relative), len(content), snapshot_id),
                )
                return {"status": "restored", "snapshot_id": snapshot_id}

            conflict = content is not None and _url_has_stored_content(con, url)
            if content is not None:
                name = _sha_suffix(filename, sha256) if conflict else filename
                relative = str(Path(row["report_date"]) / name)
                _write_raw_file(raw_dir, relative, bytes(content))
                row["raw_path"] = relative
                row["byte_size"] = len(content)
            cursor = con.execute(
                'INSERT INTO "' + SNAPSHOT_TABLE + '" ('
                '"source_url", "filename", "report_date", "report_time_label",'
                '"snapshot_ts_et", "snapshot_ts_utc", "fetched_at", "http_status",'
                ' "sha256", "raw_path", "byte_size", "final_url", "parse_status"'
                ') VALUES ('
                ':source_url, :filename, :report_date, :report_time_label,'
                ':snapshot_ts_et, :snapshot_ts_utc, :fetched_at, :http_status,'
                ' :sha256, :raw_path, :byte_size, :final_url, :parse_status)',
                row,
            )
            snapshot_id = cursor.lastrowid
    except sqlite3.Error as exc:
        raise InjuryStoreError(
            f"could not record fetch of {url} into {SNAPSHOT_TABLE}: {exc}"
        ) from exc

    if content is None:
        status = "absent" if outcome["http_status"] == 404 else "not_a_pdf"
        return {"status": status, "snapshot_id": snapshot_id}
    return {"status": "conflict" if conflict else "inserted",
            "snapshot_id": snapshot_id}


def fetch_needed(con, url, retry_missing=False, now=None):
    """Should this URL be probed again? Resume logic in one place.

    False when content is already recorded. When only absences are recorded,
    the URL is re-probed while the SLOT is younger than ABSENT_RETRY_SECONDS
    -- measured from the report's own label, NOT from when we happened to look
    at it -- unless --retry-missing forces the issue.

    Why the slot instant and not the probe clock: upload lag is a property of
    the report slot. A 404 five minutes after the label is a PDF that has not
    landed yet and must be retried at ANY capture cadence; a probe-clock
    window is invisible to a 3-hourly batch (it always finds absences older
    than the window) and turns every lag-404 into a permanent wrong absence --
    exactly the late scratches this module exists for. Conversely a slot from
    three days ago is final the moment it is recorded, so a historical
    backfill stops re-probing dead slots and runs at full throughput.
    """
    try:
        rows = con.execute(
            'SELECT "sha256" FROM "' + SNAPSHOT_TABLE + '"'
            ' WHERE "source_url" = ?',
            (url,),
        ).fetchall()
    except sqlite3.Error as exc:
        raise InjuryStoreError(f"could not read {SNAPSHOT_TABLE} for {url}: {exc}") from exc
    if not rows:
        return True
    if any(r["sha256"] not in NO_CONTENT_SENTINELS for r in rows):
        return False
    if retry_missing:
        return True
    parsed = parse_report_filename(url)
    slot_utc = et_label_to_utc(parsed["report_date"], parsed["hour"], parsed["minute"])
    moment = _require_aware(now if now is not None else utcnow())
    return (moment - slot_utc).total_seconds() < ABSENT_RETRY_SECONDS


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def capture_days(day_start, day_end, db_path=None, raw_dir=None,
                 max_requests=DEFAULT_MAX_REQUESTS, requester=None,
                 sleep=time.sleep, rng=random, retry_missing=False, slots=None,
                 now=None, timeout=REQUEST_TIMEOUT_SECONDS):
    """Resumable capture of every snapshot slot in [day_start, day_end].

    Enumerates each date's slots (see `slots_for_date`), skips URLs whose
    outcome is already recorded (a young absence is NOT yet final: see
    `fetch_needed` for the slot-based retry gate), never probes slots whose
    instant has not arrived yet, and stops cleanly when the request budget is
    spent. Returns a summary dict:

      days, slots_considered, future_skipped, already_recorded, requests_made,
      fetched, absent, not_a_pdf, conflicts, restored, skipped,
      budget_exhausted, errors, first_error

    The outcome buckets account for every stored fetch: fetched + absent +
    not_a_pdf + conflicts + restored + skipped + errors == requests_made
    whenever no request needed a retry (retries and transport failures spend
    extra attempts by design).

    `budget_exhausted` is progress, not failure: re-running the same command
    resumes exactly where this one stopped, because every recorded slot the
    retry gate considers final is skipped.

    Prints one line per stored outcome and a summary at the end. This is a CLI
    tool's reporting surface; nothing here raises for a missing snapshot.
    """
    if day_end < day_start:
        raise InjuryFetchError(
            f"--to {day_end.isoformat()} is before --from {day_start.isoformat()}"
        )
    raw_dir = RAW_DIR if raw_dir is None else Path(raw_dir)
    budget = RequestBudget(max_requests)
    now_utc = _require_aware(now) if now is not None else utcnow()
    get = requester if requester is not None else _default_requester

    summary = {
        "days": 0, "slots_considered": 0, "future_skipped": 0,
        "already_recorded": 0, "requests_made": 0, "fetched": 0, "absent": 0,
        "not_a_pdf": 0, "conflicts": 0, "restored": 0, "skipped": 0,
        "budget_exhausted": False, "errors": 0, "first_error": None,
    }

    with contextlib.closing(open_db(db_path)) as con:
        day = day_start
        while day <= day_end:
            summary["days"] += 1
            for (hour, minute), url in candidate_urls(day, slots=slots):
                summary["slots_considered"] += 1
                if et_label_to_utc(day, hour, minute) > now_utc:
                    summary["future_skipped"] += 1
                    continue
                if not fetch_needed(con, url, retry_missing=retry_missing, now=now_utc):
                    summary["already_recorded"] += 1
                    continue
                if budget.remaining == 0:
                    summary["budget_exhausted"] = True
                    break
                try:
                    outcome = fetch_url(
                        url, requester=get, timeout=timeout, budget=budget,
                        sleep=sleep,
                    )
                except RequestBudgetExhausted:
                    summary["budget_exhausted"] = True
                    break
                except InjuryFetchError as exc:
                    summary["errors"] += 1
                    summary["first_error"] = summary["first_error"] or str(exc)
                    print(f"FETCH FAILED: {exc}")
                    # Pace after a failure too: a run of errors is not
                    # permission to hammer the host harder.
                    sleep(rng.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
                    continue
                result = store_fetch(con, outcome, raw_dir=raw_dir)
                status = result["status"]
                if status == "conflict":
                    summary["conflicts"] += 1
                elif status == "restored":
                    summary["restored"] += 1
                elif status in ("absent", "not_a_pdf"):
                    summary[status] += 1
                elif status == "skipped":
                    # An outcome already on disk (a concurrent batch won the
                    # race). Counted so the outcome buckets are complete:
                    # fetched + absent + not_a_pdf + conflicts + restored +
                    # skipped + errors == requests_made whenever no request
                    # needed a retry -- retries and transport failures spend
                    # extra attempts by design.
                    summary["skipped"] += 1
                else:
                    summary["fetched"] += 1
                print(
                    f"[{outcome['http_status']}] {url} -> {status} "
                    f"(requests {budget.used}/{budget.max_requests})"
                )
                sleep(rng.uniform(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
            if summary["budget_exhausted"]:
                break
            day += timedelta(days=1)

    # Every budget unit was spent on exactly one HTTP attempt, including
    # retries and failed fetches. One number, one meaning.
    summary["requests_made"] = budget.used
    return summary


def capture_day(day, **kwargs):
    """Capture one report date. Thin wrapper over `capture_days`."""
    return capture_days(day, day, **kwargs)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


# Column identities, in canonical order. Matching is by LABEL, not by
# position, so a report that reorders or renames a column still parses: the
# anchors come from where the label actually sits.
COL_GAME_DATE = 0
COL_GAME_TIME = 1
COL_MATCHUP = 2
COL_TEAM = 3
COL_PLAYER = 4
COL_STATUS = 5
COL_REASON = 6

HEADER_LABELS = {
    COL_GAME_DATE: ("game date",),
    COL_GAME_TIME: ("game time",),
    COL_MATCHUP: ("matchup", "game matchup"),
    COL_TEAM: ("team",),
    COL_PLAYER: ("player",),
    COL_STATUS: ("current status", "status"),
    COL_REASON: ("reason", "reason for status", "comments"),
}

# A header declaration needs the query-critical columns plus a quorum of the
# rest. "team" alone is a word that appears in titles; the SET is the signal.
HEADER_REQUIRED = (COL_GAME_DATE, COL_TEAM, COL_PLAYER, COL_STATUS)
HEADER_QUORUM = 4

# The designation vocabulary, used only by the row-shape heuristic (a line
# whose player cell carries no comma is a row when its status is one of these).
# This is NOT a whitelist on parsing: a comma-named player with a status
# outside this set is parsed and stored verbatim. Drift shows up as data, not
# as a dropped row.
KNOWN_STATUS_KEYS = ("available", "out", "doubtful", "questionable", "probable")


def _load_pdfplumber():
    """Import pdfplumber lazily, with the repo's standard failure message."""
    try:
        import pdfplumber  # noqa: F401
    except ImportError as exc:
        raise InjuryParseError(
            "pdfplumber is required to parse injury report PDFs but is not "
            "installed. Install it with `pip install pdfplumber` (it is listed "
            "in requirements.txt). It is imported lazily so that the URL, time "
            "and store layers stay usable without it."
        ) from exc
    import pdfplumber

    return pdfplumber


def _cluster_lines(chars, y_tolerance=2.0):
    """Group character dicts into text lines by vertical position."""
    lines = []
    for ch in sorted(chars, key=lambda c: (c["top"], c["x0"])):
        if lines and abs(ch["top"] - lines[-1]["ref_top"]) <= y_tolerance:
            lines[-1]["chars"].append(ch)
        else:
            lines.append({"ref_top": ch["top"], "chars": [ch]})
    return [line["chars"] for line in lines]


def _char_words(line_chars, gap_ratio=0.2, gap_floor=1.0):
    """Group a line's chars into words.

    Two signals end a word, and both matter:

      * an explicit whitespace CHARACTER. pdfminer emits the spaces inside a
        text run as real char dicts, so "Game Date" arrives as two words worth
        of glyphs with a space glyph between them. Relying on glyph gaps alone
        merges them, because the gap between "Game"'s last glyph and the space
        glyph is zero.
      * an inter-glyph gap wider than max(gap_floor, gap_ratio * font size).
        The live 2025-26 reports carry NO space glyphs at all -- spaces exist
        only as advances (~0.25 em), while inter-letter side bearings stay
        under ~0.15 em -- so the threshold must scale with the font size or
        every phrase merges into one word (which is exactly what broke header
        detection against the real PDFs before this rule).
    """
    words = []
    for ch in sorted(line_chars, key=lambda c: c["x0"]):
        if not ch["text"] or ch["text"].isspace():
            if words:
                words.append(None)
            continue
        if words and words[-1] is not None:
            prev = words[-1]["chars"][-1]
            gap = ch["x0"] - prev["x1"]
            size = min(float(ch.get("size") or 0), float(prev.get("size") or 0))
            if gap > max(gap_floor, gap_ratio * max(size, 1.0)):
                words.append({"chars": [ch]})
                continue
            words[-1]["chars"].append(ch)
        else:
            words.append({"chars": [ch]})
    return [w for w in words if w is not None]


def _chars_to_text(chars, gap_ratio=0.2, gap_floor=1.0):
    """Join chars into a cell string.

    Whitespace char dicts contribute their own space (see `_char_words` for
    why those exist); a wide inter-glyph GAP adds a space too, for runs that
    carry no space glyph at all. The gap threshold scales with the font size
    for the same reason as in `_char_words` -- the live reports carry no space
    glyphs, and a width-based threshold splits narrow letters ("I njury") in
    fonts whose side bearings exceed it. `_clean_cell` collapses any doubling.
    """
    parts = []
    prev = None
    for ch in sorted(chars, key=lambda c: c["x0"]):
        if prev is not None and ch["text"] and not ch["text"].isspace():
            gap = ch["x0"] - prev["x1"]
            size = min(float(ch.get("size") or 0), float(prev.get("size") or 0))
            if gap > max(gap_floor, gap_ratio * max(size, 1.0)):
                parts.append(" ")
        parts.append(ch["text"])
        if ch["text"] and not ch["text"].isspace():
            prev = ch
        else:
            prev = None  # a space char already separates; do not double it
    return _clean_cell("".join(parts))


def _clean_cell(text):
    """Whitespace-normalise one cell. Unicode is preserved as extracted."""
    return " ".join(text.replace("\xa0", " ").split())


def _normalise_label(text):
    return " ".join(text.lower().replace(".", "").split())


def _match_header(line_chars):
    """If this line is a column header, return {column: anchor_x0}, else None.

    Matching is on the label WORDS at the line's leftmost positions; the
    anchors are the labels' own x0 values. Later columns are then delimited by
    the next anchor to the right, which is why a cell overflowing its column
    (a long "7:30 pm ET" in the Game Time column, say) still lands correctly:
    each CHARACTER is placed by its own position, not by a word-level guess.
    """
    words = _char_words(line_chars)
    tokens = [_normalise_label(_chars_to_text(w["chars"], gap_ratio=0.0))
              for w in words]
    occupied = [False] * len(words)
    anchors = {}
    # Longest aliases first so "current status" wins over "status".
    candidates = sorted(
        ((alias, col) for col, aliases in HEADER_LABELS.items() for alias in aliases),
        key=lambda pair: -len(pair[0].split()),
    )
    for alias, col in candidates:
        alias_tokens = alias.split()
        width = len(alias_tokens)
        for start in range(len(tokens) - width + 1):
            if any(occupied[start:start + width]):
                continue
            if tokens[start:start + width] == alias_tokens:
                anchors[col] = words[start]["chars"][0]["x0"]
                for i in range(start, start + width):
                    occupied[i] = True
                break
    if len(anchors) < HEADER_QUORUM:
        return None
    if any(col not in anchors for col in HEADER_REQUIRED):
        return None
    return anchors


def _assign_cells(line_chars, anchors):
    """Split one text line into column cells by character position.

    Each char goes to the column whose anchor is the greatest x0 at or left of
    the char's CENTER. Chars left of the first anchor join the first column
    (they are the left edge of the date cell). Returns {column: text}.
    """
    bounds = sorted((x0, col) for col, x0 in anchors.items())
    cells = {col: [] for col in anchors}
    for ch in sorted(line_chars, key=lambda c: c["x0"]):
        center = (ch["x0"] + ch["x1"]) / 2.0
        col = bounds[0][1]
        for x0, candidate in bounds:
            if center >= x0:
                col = candidate
            else:
                break
        cells[col].append(ch)
    return {col: _chars_to_text(chars) for col, chars in cells.items() if chars}


def _line_text(line_chars):
    """One line's raw text, for the recorded unparsed/furniture rows."""
    return _chars_to_text(line_chars)


def _parse_game_date(text):
    """Parse a Game Date cell into a date, tolerating era drift in formats.

    MM/DD/YYYY first (the format the reports actually use), then ISO, then
    two-digit year. A day-first reading is offered last purely as a fallback
    for recovered PDFs; it can only ever match when the month-first readings
    failed.
    """
    stamp = text.strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%d/%y", "%d/%m/%Y"):
        with contextlib.suppress(ValueError):
            return datetime.strptime(stamp, fmt).date()
    return None


def _player_key(player):
    """Deterministic lookup key: casefolded, whitespace-collapsed."""
    return " ".join(player.split()).casefold()


def parse_injury_pdf(data):
    """Parse one injury report PDF into rows and explicitly-unparsed lines.

    Returns a dict:

      rows      parsed rows: game_date (ISO), game_time, matchup, team, player,
                status, reason, line_no, inherited (context fields the row did
                not print itself). `reason` may be ''.
      unparsed  [{line_no, raw_text, reason}] -- EVERY line that was ROW-SHAPED
                but could not be parsed is here, with its raw text. Nothing
                that looks like a data row is dropped; a malformed row is
                recorded, which is what docs/measurement_bugs.md demands of a
                parser.
      furniture [{line_no, raw_text, reason}] -- lines that are NOT row-shaped:
                titles, footers, page furniture. Recorded too (nothing is
                dropped silently, ever), but classified separately so a title
                on every page does not read like a parser failure. The two
                buckets are distinguishable in injury_unparsed_rows by their
                `reason`.
      header_lines, blank_lines, continuation_lines  counts
      error     None, or a document-level message. Two things set it: a
                missing column header (`rows` empty, every line in
                `unparsed`), and a mid-document extraction failure -- corrupt
                or truncated pages. In the failure case the lines extracted
                BEFORE the bad page are kept (rows parsed so far, plus the
                rest recorded as unparsed) and the document reports failure:
                loud AND lossless wherever extraction succeeded.

    THE REAL LAYOUT, AND WHY PARSING IS TWO-PASS PER PAGE
    =====================================================
    Verified against live 2025-26 reports (the PDFs themselves are the
    fixtures for this; see Tests/test_injury_data.py::TestRealEra3Layout):

      * A row is a BAND, not a line. The identity cells (player, status) are
        vertically centred, and a long reason wraps onto sub-lines BOTH ABOVE
        and BELOW that centre line, ~7pt away, while rows are ~22pt apart.
        Line-at-a-time parsing attributes the first half of "Injury/Illness -
        Right Quadriceps Strain" to the row ABOVE, and the second sub-line of
        a two-line wrap to the row BELOW. So each page is processed in two
        passes: identity lines become rows; wrap lines are grouped into RUNS
        (consecutive wrap lines with no row line between them) which attach to
        the row nearest the run's interval -- ties to the row above -- and
        merge above the row's centre line PREPEND (in reading order), below
        APPEND. See `_process_page` for the exact rule and its one known
        ambiguity.
      * Context columns (Game Date, Game Time, Matchup, Team) print ONCE per
        block and every row fills forward from the last printed value. The
        game date prints once per document; the team once per team block;
        the time and matchup once per game block. Rows carry no copies.
      * The label is "Player Name", not "Player". Column positions come from
        the header LABELS wherever they sit, never from constants.
      * The Game Time cell reads "04:00 (ET)".
      * Titles and footers ("Injury Report: 10/23/26 12:00 AM", "Page 1 of 6")
        scatter words across the Team/Player/Status columns. They are kept out
        of the rows by SHAPE: a row's player cell is "Last, First" (it carries
        a comma) and its status is a designation. A line with identity-looking
        cells that fails that shape is document furniture -- recorded with its
        raw text, never merged into a player's name.

    DEFENSIVE DETAIL (drift tolerated in every direction)
      * columns are located by the header LABELS' positions, so column order
        and width drift is tolerated ("Team" and "Player" swap places? fine);
      * each CHARACTER is placed by its own x position, so a cell that
        overflows its column does not corrupt the next one;
      * date cells tolerate MM/DD/YYYY, YYYY-MM-DD, MM/DD/YY, DD/MM/YYYY;
      * wrap text merges only into text columns (Reason, and Matchup/Game Time
        when the row already printed them). A wrap run that cannot be
        attributed to a row within a plausible distance is RECORDED as an
        unparsed line rather than glued onto an unrelated row. Known
        ambiguity: wrap lines from ADJACENT rows that sit textually adjacent,
        with no row line between them, form one run and attach together to
        the nearer row (ties to the row above); when it matters, verify with
        `query --history`.

    pdfplumber is imported inside this function, never at module scope.
    """
    pdfplumber = _load_pdfplumber()
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise InjuryParseError(f"parse_injury_pdf needs PDF bytes, got {type(data).__name__}")
    if len(data) > MAX_PDF_BYTES:
        raise InjuryParseError(
            f"PDF is {len(data)} bytes, over the {MAX_PDF_BYTES}-byte parse "
            "cap. Refusing to open it: an OOM kill is the one failure this "
            "module's exception contract cannot catch."
        )
    if not bytes(data).startswith(b"%PDF"):
        raise InjuryParseError(
            "input does not start with '%PDF': not a PDF. A CDN error page "
            "reached the parser; record it as an absence upstream, not as data."
        )

    import io

    rows = []
    unparsed = []
    furniture = []
    preheader = []
    header_lines = 0
    blank_lines = 0
    continuation_lines = 0
    anchors = None
    line_no = 0
    error = None
    carry_row = None  # the last row of the previous page: a reason can wrap
    # across a page break and continue at the top of the next page.
    context = {"game_date_raw": None, "game_date_iso": None,
               "game_time": None, "matchup": None, "team": None}

    try:
        with pdfplumber.open(io.BytesIO(bytes(data))) as pdf:
            pages = list(pdf.pages)
    except Exception as exc:
        raise InjuryParseError(f"pdfplumber could not open the PDF: {exc!r}") from exc

    if not pages:
        raise InjuryParseError("PDF has no pages")
    if len(pages) > MAX_PDF_PAGES:
        raise InjuryParseError(
            f"PDF has {len(pages)} pages, over the {MAX_PDF_PAGES}-page parse "
            "cap. Refusing to process it: a hostile page count must not be "
            "able to stall or exhaust an unattended batch."
        )

    for page_index, page in enumerate(pages, start=1):
        entries = []
        try:
            chars = list(page.chars)
            if len(chars) > MAX_PAGE_GLYPHS:
                # Per-page glyph ceiling. The page's lines are NOT extracted
                # (that is the memory-blowing work this caps); the page is
                # recorded instead of dropped, and the other pages still
                # parse. Deliberately NOT a document-level error: the report
                # is partially readable and the marker says exactly which
                # part is missing.
                line_no += 1
                unparsed.append({
                    "line_no": line_no,
                    "raw_text": (
                        f"<page {page_index} not extracted: "
                        f"{len(chars)} glyphs exceeds the {MAX_PAGE_GLYPHS} "
                        "glyph per-page ceiling>"
                    ),
                    "reason": "page exceeded the per-page glyph ceiling; its "
                              "lines were not extracted (memory bound), other "
                              "pages still processed",
                })
                continue
            page_lines = _cluster_lines(chars)
            for line_chars in page_lines:
                line_no += 1
                text = _line_text(line_chars)
                if not text:
                    blank_lines += 1
                    continue

                header_anchors = _match_header(line_chars)
                if header_anchors is not None:
                    header_lines += 1
                    anchors = header_anchors
                    continue

                if anchors is None:
                    # Before any header on this document: the line cannot be
                    # interpreted at all. Park it; the document-level verdict
                    # at the end decides whether it is furniture or a lost row.
                    preheader.append({"line_no": line_no, "raw_text": text})
                    continue

                cells = _assign_cells(line_chars, anchors)
                if not cells:
                    blank_lines += 1
                    continue

                top = min(ch["top"] for ch in line_chars)
                entry = {"top": top, "cells": cells, "text": text,
                         "line_no": line_no}

                if cells.get(COL_PLAYER) or cells.get(COL_STATUS):
                    if _row_shaped(cells):
                        entry["kind"] = "primary"
                        entries.append(entry)
                    else:
                        furniture.append({
                            "line_no": line_no, "raw_text": text,
                            "reason": "document furniture: identity cells are "
                                      "not a 'Last, First' player and a "
                                      "designation status",
                        })
                    continue

                entry["kind"] = "secondary"
                entries.append(entry)

            page_rows, merged, page_unparsed, page_furniture = _process_page(
                entries, context, carry_row=carry_row
            )
        except Exception as exc:
            # pdfplumber parses LAZILY: a corrupt or truncated content stream
            # surfaces here, at page level, not at open(). One bad page must
            # not wedge the document, and one bad document must not wedge the
            # batch (parse_all guards that too). Degrade explicitly: keep the
            # lines extracted before the failure, record the failure itself,
            # and carry on with the remaining pages.
            error = error or f"page {page_index} failed to parse: {exc!r}"
            # The marker ALLOCATES its own line number (advancing the
            # counter). Writing at line_no + 1 without advancing collided
            # with the next page's first line on UNIQUE (snapshot_id,
            # line_no), and the insert silently dropped one of them.
            line_no += 1
            unparsed.append({
                "line_no": line_no,
                "raw_text": f"<page {page_index} failed to extract: {exc!r}>",
                "reason": "page failed to extract mid-document; previously "
                          "extracted lines were kept",
            })
            for entry in entries:
                unparsed.append({
                    "line_no": entry["line_no"], "raw_text": entry["text"],
                    "reason": "line extracted before the page failed to parse",
                })
            continue
        rows.extend(page_rows)
        unparsed.extend(page_unparsed)
        furniture.extend(page_furniture)
        continuation_lines += merged
        if page_rows:
            carry_row = page_rows[-1]

    if header_lines == 0:
        error = error or "no column header row found in the document"
        # A header-less document degrades to recorded noise, never to a silent
        # empty result: every line the parser saw is kept as unparsed.
        for entry in preheader:
            unparsed.append({
                **entry,
                "reason": "no column header row found in the document; "
                          "line could not be interpreted",
            })
    else:
        for entry in preheader:
            furniture.append({
                **entry,
                "reason": "document furniture: line before the table header",
            })

    return {
        "rows": rows,
        "unparsed": unparsed,
        "furniture": furniture,
        "header_lines": header_lines,
        "blank_lines": blank_lines,
        "continuation_lines": continuation_lines,
        "error": error,
    }


# Wrap text may continue only these columns: the free-text ones. Identity
# columns (Date/Team/Player/Status) are never written by a wrap line -- a
# stray word in them is far more likely a footer than a name.
WRAP_MERGE_COLUMNS = (COL_REASON, COL_MATCHUP, COL_GAME_TIME)

_ROW_KEY_BY_COLUMN = {
    COL_GAME_DATE: "game_date_raw",
    COL_GAME_TIME: "game_time",
    COL_MATCHUP: "matchup",
    COL_TEAM: "team",
    COL_PLAYER: "player",
    COL_STATUS: "status",
    COL_REASON: "reason",
}


def _process_page(entries, context, carry_row=None):
    """Two-pass page assembly: rows first, then wrap-RUN attribution.

    Returns (rows, merged_count, unparsed, furniture).

    A row is an identity line (player/status cells); a wrap line is anything
    else that reached this point.

    Wrap lines are attributed as RUNS, not line by line. Consecutive wrap
    lines -- with no row line between them and no gap wider than ~1.5x the
    page's own median line gap -- form one run and belong to ONE row. The run
    attaches to the row whose identity line is nearest to the run's INTERVAL
    (distance zero inside the interval, else distance to the nearest end);
    ties go to the row above. Line-by-line attribution is provably wrong for
    this: a two-sub-line reason below a row's identity line puts its second
    sub-line closer to the NEXT row (in the real 2025-26 geometry, 14pt from
    its own row vs ~8pt from the next) -- which silently truncated one
    player's reason and inflated another's. Runs also fix the READING ORDER
    of prepends: sub-lines above the identity line join their row's text in
    top-to-bottom order, ahead of the row's own text.

    Known residual ambiguity, recorded rather than hidden: two wrap lines that
    really belong to adjacent rows but sit textually adjacent, with no row
    line between them, form one run and attach together to the nearer row
    (ties to the above). Geometry cannot separate them; `query --history`
    shows the merged text when it matters.

    A wrap run at the TOP of a page that no row on this page can claim
    continues `carry_row`, the previous page's last row: that is what a reason
    running across a page break looks like. Runs that cannot be attributed at
    all are recorded as unparsed, never glued to an unrelated row. Pages carry
    their own vertical coordinates, so cross-page geometry is meaningless.
    """
    rows = []
    unparsed = []
    furniture = []
    merged = 0

    primaries = [e for e in entries if e["kind"] == "primary"]
    secondaries = [e for e in entries if e["kind"] == "secondary"]

    # Pass 1: rows, in reading order, with fill-forward context.
    for entry in primaries:
        row, problem = _row_from_cells(entry["cells"], entry["text"],
                                       entry["line_no"], context)
        if row is None:
            unparsed.append({"line_no": entry["line_no"],
                             "raw_text": entry["text"], "reason": problem})
            entry["row"] = None
        else:
            rows.append(row)
            entry["row"] = row

    if not secondaries:
        return rows, merged, unparsed, furniture

    # Geometry. Two gap scales exist on these pages: the LINE gap inside a row
    # band (wrap sub-lines sit one line below/above the identity line) and the
    # ROW gap between identity lines. A wrap run is adjacent at the LINE scale,
    # so its threshold derives from the TIGHTEST line gap -- the median is
    # dominated by row gaps on dense pages and would chain wraps across whole
    # row bands (this regressed the real 2026-10-23 reports the first time it
    # was written). The cap at half the tightest ROW gap keeps a uniform
    # layout (line gap == row gap) from chaining a run past its own band.
    # A footer or a straggler sits far from every row: the attribution bound
    # stays at the page's median gap.
    tops = sorted(e["top"] for e in entries)
    line_gaps = [b - a for a, b in zip(tops, tops[1:]) if b > a]
    median_gap = statistics.median(line_gaps) if line_gaps else 0.0
    distance_bound = max(15.0, median_gap)
    run_gap = max(2.0, 1.5 * min(line_gaps)) if line_gaps else 2.0
    primary_tops = sorted(p["top"] for p in primaries)
    row_gaps = [b - a for a, b in zip(primary_tops, primary_tops[1:]) if b > a]
    if row_gaps:
        run_gap = min(run_gap, 0.5 * min(row_gaps))

    # Group wrap lines into runs. A run breaks at any row line (an identity
    # line interrupts the text flow) and at gaps wider than run_gap.
    runs = []
    current = []
    prev_top = None
    for entry in sorted(entries, key=lambda e: e["top"]):
        if entry["kind"] == "primary":
            if current:
                runs.append(current)
                current = []
            prev_top = None
            continue
        if current and (entry["top"] - prev_top) > run_gap:
            runs.append(current)
            current = []
        current.append(entry)
        prev_top = entry["top"]
    if current:
        runs.append(current)

    # Attribute each run to one row, then merge its lines in reading order.
    # Prepended parts (wrap text above the row's identity line) are collected
    # and composed BEFORE the row's own text at the end, so a multi-line run
    # above a row keeps its top-to-bottom order.
    prepend = {}  # (id(row), key) -> {"row", "key", "parts"}
    first_primary_top = min((p["top"] for p in primaries), default=None)
    for run in runs:
        run_min = run[0]["top"]
        run_max = run[-1]["top"]

        def run_distance(primary):
            if primary["top"] < run_min:
                return run_min - primary["top"]
            if primary["top"] > run_max:
                return primary["top"] - run_max
            return 0.0

        target = None
        structural = any(
            e["cells"].get(COL_GAME_DATE) or e["cells"].get(COL_TEAM)
            for e in run
        )
        if primaries and not structural:
            # A line carrying a Game Date or Team cell is STRUCTURAL (a slate
            # line, a title): wraps never restate the game date or the team,
            # so occupancy decides that, not distance. Without this guard a
            # sparse page (few rows -> inflated distance bound) glues a title
            # into some row's game_time/matchup cells.
            best = min(
                primaries,
                key=lambda p: (run_distance(p),
                               0 if p["top"] <= run_min else 1),
            )
            # A run next to a row line that FAILED to parse is recorded, not
            # re-homed to a further-away row: its text belongs to the failed
            # row, and gluing it elsewhere corrupts that row's reason.
            if (run_distance(best) <= distance_bound
                    and best.get("row") is not None):
                target = best
        if (target is None and not structural and carry_row is not None
                and all(_wrap_shaped(e) for e in run)):
            # Cross-page wrap: the text runs off the bottom of the previous
            # page and continues here, above this page's first row. It belongs
            # to the previous page's last row, and it CONTINUES that row's
            # text, so it appends. The pseudo-target sits below everything
            # (-inf top) so every line of the tail appends in reading order.
            # (A page's own top-of-page wrap stays within `distance_bound` of
            # its row and took the interval path.)
            if first_primary_top is None or run_max < first_primary_top:
                target = {"row": carry_row, "top": float("-inf")}

        if target is None:
            # Cannot attribute this run to any row. KEEP every line. Three
            # shapes appear here and each is recorded with what it is:
            for entry in run:
                date_text = entry["cells"].get(COL_GAME_DATE, "")
                if date_text and _parse_game_date(date_text) is None:
                    furniture.append({
                        "line_no": entry["line_no"], "raw_text": entry["text"],
                        "reason": "document furniture: non-date text in the "
                                  "Game Date column and no player designation "
                                  "(title-like line); kept, not a row",
                    })
                elif date_text or entry["cells"].get(COL_TEAM):
                    # The reports end with team-level lines for games whose
                    # designations are not in yet: game context + team + a
                    # "NOT YET SUBMITTED" marker, no player. Not row-shaped
                    # (no identity cells) but not furniture in the
                    # title/footer sense either -- structured slate state,
                    # recorded here so nothing is dropped, and deliberately
                    # kept OUT of the rows rather than glued onto some
                    # player's reason (which is what a nearest-row merge did
                    # to the row above). The bucket is `furniture` in the
                    # documented sense: NOT a row-shaped data line.
                    furniture.append({
                        "line_no": entry["line_no"], "raw_text": entry["text"],
                        "reason": "team-level status line: game/team context "
                                  "with no player designation yet (e.g. 'NOT "
                                  "YET SUBMITTED'); kept as evidence, not a "
                                  "player row",
                    })
                elif _wrap_shaped(entry):
                    unparsed.append({
                        "line_no": entry["line_no"], "raw_text": entry["text"],
                        "reason": "wrap-shaped line could not be attributed to "
                                  "any row on this page (possible cross-page "
                                  "wrap)",
                    })
                else:
                    furniture.append({
                        "line_no": entry["line_no"], "raw_text": entry["text"],
                        "reason": "document furniture: text outside the row "
                                  "columns that no row could claim",
                    })
            continue

        row = target["row"]
        for entry in run:  # reading order
            above = entry["top"] < target["top"]
            joined = False
            for col in WRAP_MERGE_COLUMNS:
                value = entry["cells"].get(col)
                if not value:
                    continue
                key = _ROW_KEY_BY_COLUMN[col]
                if col != COL_REASON and not row[key]:
                    # Matchup/Game Time wrap only around text the row printed.
                    continue
                if above:
                    slot = prepend.setdefault(
                        (id(row), key), {"row": row, "key": key, "parts": []}
                    )
                    slot["parts"].append(value)
                else:
                    row[key] = _clean_cell(f"{row[key]} {value}")
                joined = True
            if joined:
                merged += 1
            else:
                furniture.append({
                    "line_no": entry["line_no"], "raw_text": entry["text"],
                    "reason": "document furniture: wrap line landed only in "
                              "identity columns",
                })

    for slot in prepend.values():
        composed = slot["parts"] + [slot["row"][slot["key"]]]
        slot["row"][slot["key"]] = _clean_cell(" ".join(composed))

    return rows, merged, unparsed, furniture


def _wrap_shaped(entry):
    """Does this line carry free-text (wrap) cells rather than identity cells?"""
    return any(entry["cells"].get(col) for col in WRAP_MERGE_COLUMNS)


def _row_shaped(cells):
    """Does this line look like a real data row rather than page furniture?

    The reports' own convention decides this: a player cell is "Last, First"
    (it carries a comma) and a status cell is a designation. Titles and
    footers scatter words into the identity columns ("...12:00" into Player,
    "AM" into Status) but never BOTH of those shapes. This is a heuristic in
    the honest direction: a false negative lands a real row in the recorded
    furniture bucket with its raw text intact; a false positive lands a title
    in the recorded unparsed bucket. Nothing is ever dropped either way.
    """
    player = cells.get(COL_PLAYER, "")
    status = _clean_cell(cells.get(COL_STATUS, "")).lower()
    return "," in player or status in KNOWN_STATUS_KEYS


def _row_from_cells(cells, text, line_no, context):
    """Build one row from its cells plus the fill-forward `context`.

    Context columns (date/time/matchup/team) are printed once per block in
    these documents, so an empty cell takes the last printed value and marks
    the field as `inherited` on the row. The date context advances only when
    the date cell actually PARSES -- a garbage date must never become the
    inherited truth for the rows below it.

    Returns (row, None) or (None, problem). `context` is mutated.
    """
    player = cells.get(COL_PLAYER, "")
    status = cells.get(COL_STATUS, "")
    missing = [name for name, value in (("Player", player),
                                        ("Current Status", status)) if not value]
    if missing:
        return None, f"missing required cell(s): {', '.join(missing)}"

    inherited = []

    date_cell = cells.get(COL_GAME_DATE, "")
    if date_cell:
        game_date = _parse_game_date(date_cell)
        if game_date is None:
            return None, f"Game Date cell {date_cell!r} is not a date"
        context["game_date_raw"] = date_cell
        context["game_date_iso"] = game_date.isoformat()
    elif context["game_date_iso"] is not None:
        inherited.append("game_date")
    game_date_iso = context["game_date_iso"]
    if game_date_iso is None:
        return None, "empty Game Date cell and no earlier row to inherit one from"

    values = {}
    for col, key in ((COL_GAME_TIME, "game_time"), (COL_MATCHUP, "matchup"),
                     (COL_TEAM, "team")):
        cell = cells.get(col, "")
        if cell:
            context[key] = cell
            values[key] = cell
        elif context[key] is not None:
            values[key] = context[key]
            inherited.append(_ROW_KEY_BY_COLUMN[col])
        else:
            values[key] = None

    if values["matchup"] is None:
        return None, "empty Matchup cell and no earlier row to inherit one from"
    if values["team"] is None:
        return None, "empty Team cell and no earlier row to inherit one from"

    return {
        "game_date": game_date_iso,
        "game_time": values["game_time"] or "",
        "matchup": values["matchup"],
        "team": values["team"],
        "player": player,
        "status": status,
        "reason": cells.get(COL_REASON, ""),
        "line_no": line_no,
        "inherited": inherited,
    }, None


# ---------------------------------------------------------------------------
# Parse -> store
# ---------------------------------------------------------------------------


def parse_snapshot(con, snapshot_id, raw_dir=None, reparse=False):
    """Parse one stored PDF into the derived tables. Returns a summary dict.

    Skips snapshots that are already parsed (reporting `already_parsed`) so
    re-running `parse` is a no-op. `reparse=True` replaces that snapshot's
    derived rows -- and only theirs -- in one transaction: the derived layer is
    a regenerable view, the raw PDF and the snapshot row are never touched.

    A row that does not parse is inserted into injury_unparsed_rows with its
    raw text and the reason. `parse_status` on the snapshot becomes one of
    parsed / parsed_with_unparsed_rows / failed -- never a silent empty.

    Failure contract: a missing raw file, on-disk bytes that do not match the
    recorded sha256, or an unparseable PDF each mark the snapshot
    parse_status='failed' with the reason in parse_error and RETURN -- they
    never raise through the batch. One bad file must not wedge `parse`
    forever while capture keeps appending new snapshots behind it.
    """
    raw_dir = RAW_DIR if raw_dir is None else Path(raw_dir)
    try:
        snap = con.execute(
            'SELECT * FROM "' + SNAPSHOT_TABLE + '" WHERE "id" = ?', (snapshot_id,)
        ).fetchone()
    except sqlite3.Error as exc:
        raise InjuryStoreError(
            f"could not read snapshot {snapshot_id}: {exc}"
        ) from exc
    if snap is None:
        raise InjuryStoreError(f"no snapshot with id {snapshot_id}")
    if snap["sha256"] in NO_CONTENT_SENTINELS:
        return {"status": "missing", "snapshot_id": snapshot_id, "rows": 0,
                "unparsed": 0, "duplicates": 0}
    if snap["parse_status"] not in (PARSE_UNPARSED,) and not reparse:
        return {
            "status": "already_parsed", "snapshot_id": snapshot_id,
            "rows": snap["parsed_rows"], "unparsed": snap["unparsed_rows"],
            "duplicates": 0,
        }

    def record_failure(message):
        """Mark the snapshot failed, loudly and losslessly, without raising.

        Counts and derived rows are kept CONSISTENT: a failed snapshot carries
        no derived rows at all (they cannot be trusted as a set -- an earlier
        --reparse may have left rows that the zeroed counts would contradict).
        The evidence layer (snapshot row + raw bytes) is never touched.
        """
        try:
            with con:
                con.execute(
                    'DELETE FROM "' + EVENTS_TABLE + '" WHERE "snapshot_id" = ?',
                    (snapshot_id,),
                )
                con.execute(
                    'DELETE FROM "' + UNPARSED_TABLE + '" WHERE "snapshot_id" = ?',
                    (snapshot_id,),
                )
                con.execute(
                    'UPDATE "' + SNAPSHOT_TABLE + '" SET "parse_status" = ?,'
                    ' "parse_error" = ?, "parsed_rows" = 0, "unparsed_rows" = 0'
                    ' WHERE "id" = ?',
                    (PARSE_FAILED, message, snapshot_id),
                )
        except sqlite3.Error as exc:
            raise InjuryStoreError(
                f"could not record parse failure for snapshot {snapshot_id}: {exc}"
            ) from exc
        return {"status": "failed", "snapshot_id": snapshot_id,
                "error": message, "rows": 0, "unparsed": 0, "duplicates": 0}

    # The raw_path comes FROM THE DATABASE: containment-check it before any
    # filesystem use (F-7, path traversal).
    path = None
    if snap["raw_path"]:
        path = _contained_raw_path(raw_dir, snap["raw_path"])
        if path is None:
            return record_failure(
                f"raw_path {snap['raw_path']!r} for snapshot {snapshot_id} "
                f"escapes the raw directory ({raw_dir}): refusing to read it "
                "(path traversal)."
            )
    if path is None or not path.exists():
        return record_failure(
            f"raw PDF for snapshot {snapshot_id} is missing at {path}. The "
            "database records it; the file is gone. Restore the file or "
            "re-fetch the slot before parsing."
        )
    data = path.read_bytes()
    # The stored bytes must BE the recorded evidence. A file that has been
    # truncated or swapped since ingest is not the report we hashed, and
    # parsing it would write fabricated events under a real snapshot id.
    if hashlib.sha256(data).hexdigest() != snap["sha256"]:
        return record_failure(
            f"raw PDF at {path} does not match the recorded sha256 for "
            f"snapshot {snapshot_id}: the file on disk is not the evidence we "
            "stored. Re-fetch the slot to obtain the original bytes."
        )
    try:
        parsed = parse_injury_pdf(data)
    except InjuryParseError as exc:
        return record_failure(str(exc))

    rows = parsed["rows"]
    unparsed = parsed["unparsed"]
    furniture = parsed["furniture"]
    recorded = unparsed + furniture
    if parsed["error"]:
        # A mid-document extraction failure: lines extracted before the bad
        # page are still written (kept, not dropped), and the snapshot says
        # 'failed' so nobody mistakes a partial table for a complete one.
        parse_status = PARSE_FAILED
    elif unparsed:
        # Row-shaped lines failed to parse: data may be missing, say so on the
        # snapshot. Furniture alone does NOT set this flag -- a title on every
        # page is not a parser failure and must not read like one.
        parse_status = PARSE_PARSED_WITH_UNPARSED
    else:
        parse_status = PARSE_PARSED

    duplicates = 0
    try:
        with con:
            if reparse:
                con.execute(
                    'DELETE FROM "' + EVENTS_TABLE + '" WHERE "snapshot_id" = ?',
                    (snapshot_id,),
                )
                con.execute(
                    'DELETE FROM "' + UNPARSED_TABLE + '" WHERE "snapshot_id" = ?',
                    (snapshot_id,),
                )
            for row in rows:
                try:
                    con.execute(
                        'INSERT INTO "' + EVENTS_TABLE + '" ('
                        '"snapshot_id", "line_no", "snapshot_ts_et",'
                        ' "snapshot_ts_utc", "game_date", "game_time",'
                        ' "matchup", "team", "player", "player_key",'
                        ' "status", "reason"'
                        ') VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                        (
                            snapshot_id, row["line_no"], snap["snapshot_ts_et"],
                            snap["snapshot_ts_utc"], row["game_date"],
                            row["game_time"], row["matchup"], row["team"],
                            row["player"], _player_key(row["player"]),
                            row["status"], row["reason"] or None,
                        ),
                    )
                except sqlite3.IntegrityError:
                    # A duplicate (snapshot_id, line_no). The row already
                    # exists, so nothing is LOST -- but a drop is never
                    # silent: it is counted and surfaced in the summary.
                    duplicates += 1
            for entry in recorded:
                try:
                    con.execute(
                        'INSERT INTO "' + UNPARSED_TABLE + '" ('
                        '"snapshot_id", "line_no", "raw_text", "reason"'
                        ') VALUES (?,?,?,?)',
                        (snapshot_id, entry["line_no"], entry["raw_text"],
                         entry["reason"]),
                    )
                except sqlite3.IntegrityError:
                    duplicates += 1
            con.execute(
                'UPDATE "' + SNAPSHOT_TABLE + '" SET "parse_status" = ?,'
                ' "parsed_rows" = ?, "unparsed_rows" = ?, "parse_error" = ?'
                ' WHERE "id" = ?',
                (parse_status, len(rows), len(recorded), parsed["error"],
                 snapshot_id),
            )
    except sqlite3.Error as exc:
        raise InjuryStoreError(
            f"could not write parsed rows for snapshot {snapshot_id}: {exc}"
        ) from exc

    if duplicates:
        print(
            f"PARSE [snapshot {snapshot_id}]: {duplicates} duplicate derived "
            "row(s) not written (a row with that line number already exists). "
            "Counted and reported: drops here are never silent."
        )

    return {
        "status": parse_status, "snapshot_id": snapshot_id,
        "rows": len(rows), "unparsed": len(unparsed), "furniture": len(furniture),
        "duplicates": duplicates,
    }


def parse_all(con, raw_dir=None, reparse=False, report_date=None):
    """Parse every stored PDF that needs it. Returns aggregate counts.

    `report_date` (a date or 'YYYY-MM-DD') restricts to one day's snapshots.

    Per-snapshot isolation is the point: parse_snapshot's failure contract
    turns the known bad outcomes (missing file, hash mismatch, corrupt PDF)
    into recorded failures, and the guard below catches anything ELSE one
    snapshot can throw so a single bad file can never abort the batch at the
    same id forever while capture keeps appending behind it. Totals carry
    `first_error` for the operator; the run continues either way.
    """
    raw_dir = RAW_DIR if raw_dir is None else Path(raw_dir)
    sql = (
        'SELECT "id", "report_date" FROM "' + SNAPSHOT_TABLE + '"'
        ' WHERE "sha256" NOT IN (?, ?)'
    )
    params = list(NO_CONTENT_SENTINELS)
    if report_date is not None:
        stamp = report_date.isoformat() if isinstance(report_date, date_cls) \
            else str(report_date)
        sql += ' AND "report_date" = ?'
        params.append(stamp)
    sql += ' ORDER BY "snapshot_ts_utc", "id"'
    try:
        targets = con.execute(sql, params).fetchall()
    except sqlite3.Error as exc:
        raise InjuryStoreError(f"could not list snapshots to parse: {exc}") from exc

    totals = {"snapshots": 0, "parsed": 0, "already_parsed": 0, "failed": 0,
              "rows": 0, "unparsed": 0, "furniture": 0,
              "unique_violations": 0, "first_error": None}
    for snap in targets:
        totals["snapshots"] += 1
        try:
            result = parse_snapshot(con, snap["id"], raw_dir=raw_dir,
                                    reparse=reparse)
        except Exception as exc:  # noqa: BLE001 - one file must not wedge all
            totals["failed"] += 1
            totals["first_error"] = totals["first_error"] or (
                f"snapshot {snap['id']}: {exc!r}"
            )
            print(f"PARSE FAILED [snapshot {snap['id']}]: {exc!r}")
            continue
        totals["rows"] += result.get("rows", 0)
        totals["unparsed"] += result.get("unparsed", 0)
        totals["furniture"] += result.get("furniture", 0)
        totals["unique_violations"] += result.get("duplicates", 0)
        if result["status"] == "already_parsed":
            totals["already_parsed"] += 1
        elif result["status"] == "failed":
            totals["failed"] += 1
            totals["first_error"] = totals["first_error"] or (
                f"snapshot {snap['id']}: {result.get('error')}"
            )
        else:
            totals["parsed"] += 1
    return totals


# ---------------------------------------------------------------------------
# Point-in-time query
# ---------------------------------------------------------------------------


# Every selected column is table-qualified: the join gives both tables an "id"
# and an unqualified "id" would be ambiguous -- which sqlite reports as a hard
# error at query time, not as a wrong answer.
_EVENT_COLUMNS = (
    'e."id", e."snapshot_id", e."line_no", e."snapshot_ts_et",'
    ' e."snapshot_ts_utc", e."game_date", e."game_time", e."matchup",'
    ' e."team", e."player", e."player_key", e."status", e."reason"'
)

_EVENT_FROM = (
    ' FROM "' + EVENTS_TABLE + '" AS e'
    ' JOIN "' + SNAPSHOT_TABLE + '" AS s ON s."id" = e."snapshot_id"'
)


def status_known_as_of(con, player, game_date, cutoff_ts, team=None):
    """What was known about `player` for `game_date` at instant `cutoff_ts`.

    The "known at 6 p.m. ET" query: the LATEST designation whose snapshot
    label is at or before the cutoff. Inclusive at the boundary -- a snapshot
    labelled exactly at the cutoff is known AT the cutoff.

    LABEL TIME IS OPTIMISTIC BY THE UPLOAD LAG: the boundary is the report's
    label, but the PDF becomes visible on the CDN a few minutes later. A
    consumer that needs strict "actually visible at T" semantics should cut a
    few minutes EARLIER than the label boundary -- a late scratch at T-5m can
    be visible at T+2m under a T label.

    `player` is matched case-insensitively on the "Last, First" form the report
    uses ("doe, john" finds "Doe, John"). `game_date` is the GAME's date (a
    date, or 'YYYY-MM-DD'); `cutoff_ts` is an aware datetime, an ISO-8601
    string with an offset, or ".... ET" (see `parse_cutoff`).

    Returns the event dict (status, reason, team, matchup, both timestamps,
    source_url) or None when nothing was known by then. None is an answer:
    "the report had not listed the player yet" is exactly the information a
    pre-tip-off model needs.

    CONTENT CONFLICTS ARE FLAGGED, NOT RESOLVED SILENTLY. The CDN is
    documented immutable, so one label should carry one content. If more than
    one stored snapshot covers the same label for this player, the winner is
    still deterministic (newest label, then highest row id) but the dict
    carries `label_conflict: True` and `label_conflict_variants`: every other
    (status, reason, snapshot_id, source_url) that existed for that label.
    Two contents at one label is an anomaly in the evidence layer; a caller
    deciding a lineup needs to know the evidence disagreed.
    """
    key = _player_key(str(player))
    if not key:
        raise InjuryDataError("player must be a non-empty name")
    stamp = game_date.isoformat() if isinstance(game_date, date_cls) else str(game_date)
    try:
        datetime.strptime(stamp, DATE_FORMAT)
    except ValueError as exc:
        raise InjuryDataError(
            f"game_date must be YYYY-MM-DD, got {game_date!r}"
        ) from exc
    cutoff = format_utc(parse_cutoff(cutoff_ts))

    sql = (
        "SELECT " + _EVENT_COLUMNS + ', s."source_url"' + _EVENT_FROM
        + ' WHERE e."player_key" = ? AND e."game_date" = ?'
        + ' AND e."snapshot_ts_utc" <= ?'
    )
    params = [key, stamp, cutoff]
    if team is not None:
        sql += ' AND e."team" = ?'
        params.append(str(team).strip())
    # Newest snapshot first; within one snapshot, the highest row id wins so
    # the answer is deterministic even if a document lists a player twice.
    sql += ' ORDER BY e."snapshot_ts_utc" DESC, e."id" DESC LIMIT 1'
    try:
        row = con.execute(sql, params).fetchone()
        if row is None:
            return None
        event = dict(row)
        others = con.execute(
            "SELECT e.\"snapshot_id\", e.\"status\", e.\"reason\","
            ' s."source_url"' + _EVENT_FROM
            + ' WHERE e."player_key" = ? AND e."game_date" = ?'
            + ' AND e."snapshot_ts_utc" = ? AND e."snapshot_id" != ?'
            + ' ORDER BY e."id"',
            (key, stamp, event["snapshot_ts_utc"], event["snapshot_id"]),
        ).fetchall()
    except sqlite3.Error as exc:
        raise InjuryStoreError(
            f"point-in-time query failed for {player!r}: {exc}"
        ) from exc
    event["label_conflict"] = bool(others)
    event["label_conflict_variants"] = [dict(o) for o in others]
    return event


def status_history(con, player, game_date, team=None):
    """Every known designation for a player in a game, oldest first.

    The full trajectory, which is what makes a Questionable -> Out flip
    auditable rather than invisible. Rows are (event, source_url) exactly as
    stored: two rows at the same label are a content conflict, visible here
    as-is rather than resolved.
    """
    key = _player_key(str(player))
    stamp = game_date.isoformat() if isinstance(game_date, date_cls) else str(game_date)
    sql = (
        "SELECT " + _EVENT_COLUMNS + ', s."source_url"' + _EVENT_FROM
        + ' WHERE e."player_key" = ? AND e."game_date" = ?'
    )
    params = [key, stamp]
    if team is not None:
        sql += ' AND e."team" = ?'
        params.append(str(team).strip())
    sql += ' ORDER BY e."snapshot_ts_utc", e."id"'
    try:
        return [dict(r) for r in con.execute(sql, params).fetchall()]
    except sqlite3.Error as exc:
        raise InjuryStoreError(
            f"history query failed for {player!r}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_date_arg(text, flag):
    try:
        return datetime.strptime(text, DATE_FORMAT).date()
    except ValueError as exc:
        raise InjuryDataError(f"{flag} must be YYYY-MM-DD, got {text!r}") from exc


def _parse_slot_arg(text):
    """Parse one slot spec: 'HH:MM' (24h) or 'H:MMAM'/'H:MM PM'."""
    stamp = text.strip().upper().replace(" ", "")
    match = re.match(r"^(\d{1,2}):(\d{2})(AM|PM)?$", stamp)
    if not match:
        raise InjuryDataError(
            f"slot {text!r} must look like '17:00' or '5:00PM'"
        )
    hour, minute, suffix = int(match.group(1)), int(match.group(2)), match.group(3)
    if suffix:
        if not 1 <= hour <= 12 or minute > 59:
            raise InjuryDataError(f"slot {text!r} is not a valid clock time")
        hour = hour % 12 + (12 if suffix == "PM" else 0)
    if not 0 <= hour <= 23 or minute > 59:
        raise InjuryDataError(f"slot {text!r} is not a valid clock time")
    return (hour, minute)


def main(argv=None):
    """CLI entry point. Additive only: creates its own tables, writes its own rows."""
    parser = argparse.ArgumentParser(
        prog="Get_Injury_Data",
        description=(
            "Capture, parse and query official NBA injury report PDFs. "
            "Snapshots are immutable CDN URLs; the store is append-only and "
            "supports point-in-time ('what was known before tip-off') lookups."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_capture = sub.add_parser(
        "capture", help="Fetch snapshot PDFs by slot enumeration (resumable)."
    )
    p_capture.add_argument("--date", default=None, help="One report date (YYYY-MM-DD).")
    p_capture.add_argument("--from", dest="date_from", default=None,
                           help="Range start (YYYY-MM-DD), inclusive.")
    p_capture.add_argument("--to", dest="date_to", default=None,
                           help="Range end (YYYY-MM-DD), inclusive.")
    p_capture.add_argument("--slot", action="append", default=None,
                           help="Restrict to these clock slots, e.g. --slot 17:00 "
                                "(repeatable). Sparse-era backfill optimisation.")
    p_capture.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS,
                           help="Hard cap on HTTP requests this run, retries "
                                "included (default: %(default)s). Budget "
                                "exhaustion is a clean, resumable stop.")
    p_capture.add_argument("--retry-missing", action="store_true",
                           help="Re-probe slots previously recorded as absent.")
    p_capture.add_argument("--db", default=None, help="SQLite file (default: %s)." % DB_PATH)
    p_capture.add_argument("--raw-dir", default=None,
                           help="Directory for raw PDFs (default: %s)." % RAW_DIR)

    p_parse = sub.add_parser(
        "parse", help="Parse stored PDFs into status events (idempotent)."
    )
    p_parse.add_argument("--date", default=None, help="Restrict to one report date.")
    p_parse.add_argument("--reparse", action="store_true",
                         help="Replace derived rows for snapshots that are "
                              "already parsed (raw PDFs are never touched).")
    p_parse.add_argument("--db", default=None, help="SQLite file (default: %s)." % DB_PATH)
    p_parse.add_argument("--raw-dir", default=None,
                         help="Directory for raw PDFs (default: %s)." % RAW_DIR)

    p_query = sub.add_parser(
        "query", help="Point-in-time: what was known at or before a cutoff."
    )
    p_query.add_argument("--player", required=True,
                         help='Player as "Last, First" (case-insensitive).')
    p_query.add_argument("--game", required=True,
                         help="Game date (YYYY-MM-DD).")
    p_query.add_argument("--team", default=None, help="Optional team filter.")
    p_query.add_argument("--cutoff", required=True,
                         help="Instant, ISO-8601 with offset, or '.... ET'.")
    p_query.add_argument("--history", action="store_true",
                         help="Print the full designation trajectory instead.")
    p_query.add_argument("--db", default=None, help="SQLite file (default: %s)." % DB_PATH)

    p_slots = sub.add_parser(
        "slots", help="Print one date's candidate slots and URLs. No network."
    )
    p_slots.add_argument("--date", required=True, help="Report date (YYYY-MM-DD).")
    p_slots.add_argument("--slot", action="append", default=None,
                         help="Restrict to these clock slots (repeatable).")

    args = parser.parse_args(argv)

    try:
        return _run(args)
    except InjuryDataError as exc:
        print(f"FAILED: {exc}")
        return 1


def _run(args):
    if args.command == "slots":
        day = _parse_date_arg(args.date, "--date")
        slots = [_parse_slot_arg(s) for s in args.slot] if args.slot else None
        for (hour, minute), url in candidate_urls(day, slots=slots):
            print(f"{hour:02d}:{minute:02d}  {url}")
        return 0

    if args.command == "capture":
        if args.date is not None:
            day_start = day_end = _parse_date_arg(args.date, "--date")
        elif args.date_from is not None and args.date_to is not None:
            day_start = _parse_date_arg(args.date_from, "--from")
            day_end = _parse_date_arg(args.date_to, "--to")
        else:
            print("FAILED: pass --date, or both --from and --to")
            return 1
        slots = [_parse_slot_arg(s) for s in args.slot] if args.slot else None
        summary = capture_days(
            day_start, day_end,
            db_path=args.db, raw_dir=args.raw_dir,
            max_requests=args.max_requests, retry_missing=args.retry_missing,
            slots=slots,
        )
        print("SUMMARY " + " ".join(f"{k}={v}" for k, v in summary.items()))
        if summary["budget_exhausted"]:
            print(
                "STOPPED (--max-requests reached). This is progress, not "
                "failure: re-running the same command resumes; every slot the "
                "retry gate considers final is skipped."
            )
        if summary["errors"]:
            # A run where every request failed must NOT exit 0: the daily job
            # that "succeeded" while doing nothing is exactly the failure mode
            # a monitoring exit code exists to catch.
            print(
                f"CAPTURE HAD {summary['errors']} FAILED REQUEST(S): "
                f"{summary['first_error']}"
            )
            return 1
        return 0

    if args.command == "parse":
        with contextlib.closing(open_db(args.db)) as con:
            totals = parse_all(
                con, raw_dir=args.raw_dir, reparse=args.reparse,
                report_date=_parse_date_arg(args.date, "--date") if args.date else None,
            )
        print("SUMMARY " + " ".join(f"{k}={v}" for k, v in totals.items()))
        return 0

    if args.command == "query":
        with contextlib.closing(open_db(args.db, readonly=True)) as con:
            if args.history:
                events = status_history(con, args.player, args.game, team=args.team)
                if not events:
                    print(
                        f"NOTHING KNOWN: no recorded designation for "
                        f"{args.player!r} in the {args.game} game."
                    )
                    return 0
                for event in events:
                    print(_format_event(event))
                return 0
            event = status_known_as_of(
                con, args.player, args.game, args.cutoff, team=args.team
            )
            if event is None:
                print(
                    f"NOTHING KNOWN: at the given cutoff, no injury report had "
                    f"listed {args.player!r} for the {args.game} game. That is "
                    f"an answer, not an error: 'not yet listed' is what a "
                    f"pre-tip-off model would have seen."
                )
                return 0
            print(_format_event(event))
            return 0

    raise InjuryDataError(f"unknown command {args.command!r}")  # pragma: no cover


def _format_event(event):
    text = (
        f"{event['snapshot_ts_et']} ET (={event['snapshot_ts_utc']}): "
        f"{event['player']} -- {event['team']} -- {event['status']}"
        + (f" ({event['reason']})" if event.get("reason") else "")
        + f" | game {event['game_date']} {event['matchup']} | {event['source_url']}"
    )
    if event.get("label_conflict"):
        variants = ", ".join(
            f"{v['status']}({v['reason'] or 'no reason'}) from snapshot "
            f"{v['snapshot_id']}"
            for v in event.get("label_conflict_variants", [])
        )
        text += (
            f" | LABEL CONFLICT: this label also exists with other content: "
            f"{variants}. Two contents at one label is an evidence-layer "
            f"anomaly; verify before acting."
        )
    return text


if __name__ == "__main__":
    sys.exit(main())
