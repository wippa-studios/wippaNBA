"""Intraday odds snapshots and closing-line-value (CLV) measurement.

WHY THIS MODULE EXISTS
======================

This repository used to scrape the day's odds exactly once, from one sportsbook,
at the close, and store a single row per game. That design makes CLV
structurally impossible, not merely unlucky.

The audit measured it: over 1,697 held-out games, betting the model's picks at
the closing line returned -1.9% to -7% ROI, and betting the closing-line
favourite itself returned -5.5%. That is the expected result, not a surprise.
You do not beat a closing line by being *right* about a game -- the closing line
is already the most efficient price in the market, assembled from every piece of
public information. You beat it by having *found the price earlier*, or by
locating a book that mispriced it. Both of those require knowing what the line
did on its way to the close, and one scrape a day throws that information away
before it is ever recorded. ROADMAP Level 4.1 calls this "the highest leverage
change on this level"; this module is the capture half of it.

WHAT IT DOES
============

* Capture half (`capture_once`, `record_snapshot`): appends a timestamped
  snapshot of every live line to a NEW table, `odds_snapshots`, in
  `Data/OddsData.sqlite`. The existing per-season tables are historical data
  and are never read for writing or written to at all.
* Measurement half (`opening_line`, `closing_line`, `line_movement`,
  `closing_line_value`, `clv_report`, `snapshot_coverage`): turns those
  snapshots into an open/close comparison and a CLV number with a standard
  error attached.

THE CAVEAT, PLAINLY
===================

1. **Intraday capture makes CLV measurable. It does not make CLV positive.**
   These are different things and this module only delivers the first. Once the
   snapshots exist you can finally see whether the prices you found early were
   better than the close. The honest answer for a naive feature set may be that
   they were not. A negative CLV result is a real result and should be published,
   not debugged away.

2. **A positive CLV over a small sample is not evidence of an edge.** CLV is
   noisy per bet; the sample standard error in `clv_report` is printed for
   exactly that reason. Treat CLV as a direction, not a number, until the
   standard error is small relative to the mean over a few hundred bets.

3. **The selection effect applies to CLV too, and this project has already paid
   it once.** The audit ran 100 hyperparameter trials and reported the best one
   selected on a single 10% slice; that inflates the winner's result by roughly
   one standard error. The same is true of any CLV number produced by trying
   many entry-time rules (different snapshot hours, different sportsbooks,
   different entry thresholds) and keeping the best. If you sweep 100 entry
   rules and report the best CLV, your number is optimistic by about one
   standard error before you have measured anything. Report the count of
   configurations tried next to every CLV figure.

4. **Coverage has to be dense before any of this is worth reading.** A game with
   one snapshot has an "open" and a "close" that are the same scrape, and its
   CLV is exactly zero by construction -- not because the line never moved. Run
   `snapshot_coverage` before believing any aggregate; it flags those games
   explicitly.

TIME HANDLING
=============

Every timestamp here is timezone-aware UTC, and `parse_captured_at` *rejects*
naive datetimes rather than guessing at one. This is not defensiveness for its
own sake: the existing pipeline was audited for mixing naive local time with
UTC schedule data, which produced days-rest and join errors that looked like
model error for an entire season. A naive timestamp passed to this module is a
bug at the call site, and failing loudly here is how it gets found. Store
`CapturedAt` in the fixed-width format `%Y-%m-%dT%H:%M:%S+00:00` so that
SQLite's lexicographic ordering is identical to chronological ordering --
`closing_line` and `opening_line` rely on `ORDER BY "CapturedAt"` and would be
subtly wrong with a variable-width format like `fromisoformat()` output.
"""

import argparse
import contextlib
import os
import random
import sqlite3
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import date as date_cls, datetime, timezone
from math import sqrt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
# Same header as Get_Odds_Data.py: this directory has a hyphen, so it is not an
# importable package and this file can only ever be loaded by path. The sys.path
# insert mirrors the sibling module so both files resolve repo-root imports the
# same way. noqa: E402 marks the imports below as deliberately after it.
sys.path.insert(1, os.fspath(BASE_DIR))  # noqa: E402

try:
    from sbrscrape import Scoreboard  # noqa: E402
except ImportError:
    # sbrscrape is a hard requirement of the scrape scripts but not of the
    # measurement half. This module must stay importable -- and therefore
    # testable, and usable for CLV over snapshots already on disk -- on a
    # machine that has never installed it. `capture_once` raises a clear
    # CaptureError instead of silently returning nothing.
    Scoreboard = None

DB_PATH = BASE_DIR / "Data" / "OddsData.sqlite"

SNAPSHOT_TABLE = "odds_snapshots"
# Fixed width on purpose. Lexicographic ORDER BY must equal chronological order
# for opening_line()/closing_line() to be correct without parsing in Python.
CAPTURED_AT_FORMAT = "%Y-%m-%dT%H:%M:%S+00:00"
DATE_FORMAT = "%Y-%m-%d"

DEFAULT_SPORTSBOOK = "fanduel"

# Books the SBR scoreboard is known to carry. Rotation across more than one matters
# for CLV: one book's line is not "the" closing line, and a movement at a single book
# may be that book's risk management rather than the market repricing.
#
# This is a default, not a whitelist. An unrecognised key is passed through and the
# fetch simply reports no odds for it, so adding a book needs no code change.
KNOWN_SPORTSBOOKS = (
    "fanduel",
    "draftkings",
    "betmgm",
    "pointsbet",
    "caesars",
    "wynn",
    "bet_rivers_ny",
)


def parse_sportsbooks(spec):
    """Parse --sportsbook into an ordered list of book keys.

    Accepts a comma-separated list, or ``all`` for KNOWN_SPORTSBOOKS. Order is
    preserved so a rotation is reproducible, and duplicates are dropped so
    ``--sportsbook fanduel,fanduel`` does not double-capture.
    """
    text = (spec or DEFAULT_SPORTSBOOK).strip().lower()
    if text == "all":
        candidates = list(KNOWN_SPORTSBOOKS)
    else:
        candidates = [part.strip() for part in text.split(",") if part.strip()]
    seen, ordered = set(), []
    for book in candidates:
        if book not in seen:
            seen.add(book)
            ordered.append(book)
    if not ordered:
        raise LineMovementError(f"--sportsbook {spec!r} names no sportsbook")
    return ordered
DEFAULT_MIN_SNAPSHOTS = 2

# The politeness delay Get_Odds_Data.py already uses between books. Snapshot
# capture makes this more important, not less: it hits the same host far more
# often per day, so hammering it is the fastest way to get rate-limited out of
# the one data source this project has.
MIN_DELAY_SECONDS = 1
MAX_DELAY_SECONDS = 3

# sbrscrape's Scoreboard accepts no timeout argument and calls requests
# internally, so the bound has to be enforced around the call rather than
# inside it. Matches REQUEST_TIMEOUT_SECONDS in src/Utils/tools.py.
REQUEST_TIMEOUT_SECONDS = 60

SNAPSHOT_SELECT = (
    'SELECT "CapturedAt", "Date", "Sportsbook", "Home", "Away", '
    '"ML_Home", "ML_Away", "OU" FROM "' + SNAPSHOT_TABLE + '"'
)

SNAPSHOT_SCHEMA = (
    'CREATE TABLE IF NOT EXISTS "' + SNAPSHOT_TABLE + '" ('
    '"CapturedAt" TEXT NOT NULL,'
    '"Date" TEXT NOT NULL,'
    '"Sportsbook" TEXT NOT NULL,'
    '"Home" TEXT NOT NULL,'
    '"Away" TEXT NOT NULL,'
    '"ML_Home" INTEGER,'
    '"ML_Away" INTEGER,'
    '"OU" REAL,'
    'PRIMARY KEY ("CapturedAt", "Sportsbook", "Date", "Home", "Away")'
    ")"
)

SNAPSHOT_INDEX = (
    'CREATE INDEX IF NOT EXISTS "idx_odds_snapshots_game" ON "' + SNAPSHOT_TABLE + '"'
    ' ("Date", "Sportsbook", "Home", "Away", "CapturedAt")'
)

SNAPSHOT_UPSERT = (
    'INSERT INTO "' + SNAPSHOT_TABLE + '" ('
    '"CapturedAt", "Date", "Sportsbook", "Home", "Away", "ML_Home", "ML_Away", "OU"'
    ') VALUES (:CapturedAt, :Date, :Sportsbook, :Home, :Away, :ML_Home, :ML_Away, :OU) '
    'ON CONFLICT ("CapturedAt", "Sportsbook", "Date", "Home", "Away") DO UPDATE SET '
    '"ML_Home" = excluded."ML_Home", '
    '"ML_Away" = excluded."ML_Away", '
    '"OU" = excluded."OU"'
)


class LineMovementError(RuntimeError):
    """Base for every failure this module raises, so callers can catch one type."""


class SnapshotError(LineMovementError):
    """Raised for a malformed snapshot: naive timestamp, bad odds, missing key."""


class SnapshotStoreError(LineMovementError):
    """Raised when the snapshot cannot be written to or read from the database."""


class CaptureError(LineMovementError):
    """Raised when a live capture fails. Never converted into an empty result."""


# ---------------------------------------------------------------------------
# Time
# ---------------------------------------------------------------------------


def utcnow():
    """The current instant, timezone-aware UTC."""
    return datetime.now(timezone.utc)


def parse_captured_at(value):
    """Return `value` as a timezone-aware UTC datetime, or raise SnapshotError.

    A naive datetime is REJECTED, not assumed to be UTC. See the module
    docstring: naive local time silently mixed with UTC schedule data is a
    defect this project has already paid for.
    """
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        # datetime.fromisoformat only learned to accept the trailing "Z" in
        # Python 3.11. Normalise it to an explicit +00:00 so a UTC timestamp
        # parses identically on every supported interpreter.
        text = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
        try:
            moment = datetime.fromisoformat(text)
        except ValueError as exc:
            raise SnapshotError(
                f"could not parse CapturedAt {value!r}; expected an ISO-8601 timestamp "
                "with a UTC offset, e.g. 2026-01-05T18:30:00+00:00"
            ) from exc
    else:
        raise SnapshotError(
            f"CapturedAt must be a datetime or an ISO-8601 string, got "
            f"{type(value).__name__}"
        )

    if moment.tzinfo is None or moment.tzinfo.utcoffset(moment) is None:
        raise SnapshotError(
            f"naive timestamp {value!r} rejected. CapturedAt must be timezone-aware "
            "UTC. This pipeline was audited for mixing naive local time with UTC "
            "schedule data; do not repeat it."
        )
    return moment.astimezone(timezone.utc)


def format_captured_at(value):
    """Format a captured-at instant into the fixed-width UTC storage form.

    Sub-second precision is truncated rather than stored. A variable-width
    format would break the lexicographic ordering that opening_line() and
    closing_line() depend on, and two captures inside the same second are the
    same observation as far as the line is concerned.
    """
    moment = parse_captured_at(value).replace(microsecond=0)
    return moment.strftime(CAPTURED_AT_FORMAT)


def format_game_date(value):
    """Normalise a game date to 'YYYY-MM-DD'. Accepts date, datetime or string."""
    if isinstance(value, datetime):
        return value.date().strftime(DATE_FORMAT)
    if isinstance(value, date_cls):
        return value.strftime(DATE_FORMAT)
    if isinstance(value, str):
        text = value.strip()
        if len(text) >= 10:
            try:
                return datetime.strptime(text[:10], DATE_FORMAT).strftime(DATE_FORMAT)
            except ValueError:
                pass
    raise SnapshotError(f"could not parse game date {value!r}; expected YYYY-MM-DD")


# ---------------------------------------------------------------------------
# Prices and de-vigging
# ---------------------------------------------------------------------------


def american_to_prob(odds):
    """Raw implied probability from an American line. Vig still included.

    -110 -> 110/210, +130 -> 100/230. A single line cannot be de-vigged: the
    overround only exists relative to the other side of the market. Use
    `devig` with both prices, or pass `overround=` to `line_movement`.
    """
    if odds is None:
        raise ValueError("American odds are required; got None")
    if isinstance(odds, bool):
        raise ValueError("American odds are required; got a bool")
    try:
        value = float(odds)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"American odds must be numeric, got {odds!r}") from exc
    if value == 0:
        raise ValueError("American odds of 0 have no implied probability")
    if value > 0:
        return 100.0 / (value + 100.0)
    return -value / (-value + 100.0)


def devig(home_price, away_price):
    """Proportional de-vig of a two-way moneyline. Returns both sides and the book.

    Proportional (divide each raw probability by the overround) is the simplest
    correct de-vig and is exact for a book quoting a symmetric margin. Shin's
    method is the principled upgrade -- it models the book's margin as
    information rather than as a flat haircut -- but it is a bigger change than
    this capture layer needs, and it is the one to revisit before quoting CLV to
    anyone as a research result.
    """
    home_raw = american_to_prob(home_price)
    away_raw = american_to_prob(away_price)
    overround = home_raw + away_raw
    if overround <= 0:
        raise ValueError(f"degenerate market: -110 / {away_price} sums to {overround}")
    return {
        "home": home_raw / overround,
        "away": away_raw / overround,
        "home_raw": home_raw,
        "away_raw": away_raw,
        "overround": overround,
    }


def line_movement(open_ml, close_ml, overround=None):
    """Implied-probability drift of one side between an opening and a closing line.

    Returns a dict: open_implied, close_implied, drift, drift_points,
    overround, de_vigged.

    `drift` is close minus open, so a POSITIVE drift means the market grew more
    confident in this side over the day (the line shortened).

    `overround` is the book's two-way margin as a float > 1.0 (1.045 for a
    standard -110/-110 book). Pass it to de-vig both ends before comparing --
    omitting it compares two raw implied probabilities that each contain a
    different amount of vig, which is not the same measurement. It defaults to
    1.0 (no de-vig) because a single line carries no overround of its own; the
    returned `de_vigged` flag says which you got, and `devig()` is how you
    compute it from both sides.
    """
    divisor = 1.0
    if overround is not None:
        try:
            divisor = float(overround)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"overround must be numeric, got {overround!r}") from exc
        if divisor <= 0:
            raise ValueError(f"overround must be positive, got {overround!r}")

    open_implied = american_to_prob(open_ml) / divisor
    close_implied = american_to_prob(close_ml) / divisor
    drift = close_implied - open_implied
    return {
        "open_ml": open_ml,
        "close_ml": close_ml,
        "open_implied": open_implied,
        "close_implied": close_implied,
        "drift": drift,
        "drift_points": drift * 100.0,
        "overround": divisor,
        "de_vigged": divisor != 1.0,
    }


def closing_line_value(entry_prob, entry_price, close_prob, close_price):
    """Closing-line value for one bet, in probability points.

    entry_prob   the probability YOU held when you bet (the model's number)
    entry_price  the American line you actually got
    close_prob   the DE-VIGGED market probability for the side you bet, at the
                 close. Supply it; this function does not re-derive it, because
                 de-vigging needs both sides of the close and only one price
                 arrives here. Use `devig(row["ML_Home"], row["ML_Away"])`.
    close_price  the American line at the close

    clv = close_prob - entry_prob, positive meaning you bought a better price
    than the market settled on.

    WHY PROBABILITY AND NOT THE AMERICAN LINE
    ------------------------------------------
    Comparing -110 to -130 tells you the line moved; it does not tell you
    whether you got value, because two books' vigs differ and vig is a function
    of the price itself. De-vig both ends first and the comparison is on a
    single comparable scale: probability.

    This is also why CLV and profit are different questions. A model that takes
    -110 at 55% and sees the line close at -130 gained CLV, because the market
    ended up more confident in the side it bought than the price it paid
    implied -- even though that particular bet can still lose money. Betting the
    closing line cannot show a gain like that: you were at the close, so your
    CLV is zero by construction. That is the entire reason CLV is worth
    measuring separately from ROI.

    Also returned, for diagnosis only and NOT as CLV:
      raw_price_drift  close_price vs entry_price in implied-probability terms
                       with no de-vig. Vig-contaminated; do not report it.
      edge_at_entry / edge_at_close
                       entry_prob minus each end's raw implied probability,
                       i.e. how much edge the model thought it had at each end.
    """
    entry_probability = _as_probability(entry_prob, "entry_prob")
    close_probability = _as_probability(close_prob, "close_prob")

    entry_implied = american_to_prob(entry_price)
    close_implied = american_to_prob(close_price)

    clv = close_probability - entry_probability
    return {
        "clv": clv,
        "clv_points": clv * 100.0,
        "entry_prob": entry_probability,
        "entry_price": entry_price,
        "entry_implied": entry_implied,
        "close_prob": close_probability,
        "close_price": close_price,
        "close_implied": close_implied,
        "raw_price_drift": close_implied - entry_implied,
        "edge_at_entry": entry_probability - entry_implied,
        "edge_at_close": entry_probability - close_implied,
    }


def _as_probability(value, field_name):
    """Coerce and validate a probability, so a stray 60 fails instead of scoring."""
    try:
        probability = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number in (0, 1), got {value!r}") from exc
    if not 0.0 < probability < 1.0:
        raise ValueError(
            f"{field_name} must be a probability strictly between 0 and 1, got "
            f"{value!r}. Pass 0.60, not 60."
        )
    return probability


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


def ensure_snapshot_table(con):
    """Create the snapshot table and its lookup index if they do not exist.

    Additive only: this never drops, alters or reads the per-season odds tables,
    which are the project's committed history. Nothing in this module issues a
    DELETE, a DROP or a table-level UPDATE.
    """
    try:
        con.execute(SNAPSHOT_SCHEMA)
        con.execute(SNAPSHOT_INDEX)
    except sqlite3.Error as exc:
        raise SnapshotStoreError(f"could not create {SNAPSHOT_TABLE}: {exc}") from exc
    return con


def _to_american(value, field_name, game):
    """Validate an American price, or None when the book did not post one."""
    if value is None:
        return None
    try:
        odds = int(value)
    except (TypeError, ValueError) as exc:
        raise SnapshotError(
            f"{field_name} is not an American price for {game}: {value!r}"
        ) from exc
    if odds == 0 or abs(odds) < 100:
        raise SnapshotError(
            f"{field_name} of {odds} for {game} is not a valid American price; "
            "the magnitude must be at least 100"
        )
    return odds


def _to_float(value, field_name, game):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise SnapshotError(f"{field_name} is not numeric for {game}: {value!r}") from exc


def _normalise_row(row):
    """Validate one input row and return it in storage order.

    Sportsbook lives on the row rather than on the function because one capture
    call may legitimately mix books when several are polled together.
    """
    if not isinstance(row, dict):
        raise SnapshotError(f"snapshot row must be a dict, got {type(row).__name__}")

    missing = [
        key for key in ("Date", "Sportsbook", "Home", "Away") if not row.get(key)
    ]
    if missing:
        raise SnapshotError(
            f"snapshot row is missing required key(s) {missing}: {row!r}. "
            "Every row needs Date, Sportsbook, Home and Away."
        )

    home = str(row["Home"]).strip()
    away = str(row["Away"]).strip()
    sportsbook = str(row["Sportsbook"]).strip().lower()
    if not home or not away:
        raise SnapshotError(f"snapshot row has an empty team name: {row!r}")

    game = f"{away} at {home} ({row['Date']}, {sportsbook})"
    return (
        format_game_date(row["Date"]),
        sportsbook,
        home,
        away,
        _to_american(row.get("ML_Home"), "ML_Home", game),
        _to_american(row.get("ML_Away"), "ML_Away", game),
        _to_float(row.get("OU"), "OU", game),
    )


def _fetch_snapshot(con, values):
    """Read one snapshot by its identity key, or None. Never creates anything."""
    cursor = con.execute(
        SNAPSHOT_SELECT
        + ' WHERE "CapturedAt" = :CapturedAt AND "Sportsbook" = :Sportsbook'
        + ' AND "Date" = :Date AND "Home" = :Home AND "Away" = :Away',
        values,
    )
    row = cursor.fetchone()
    if row is None:
        return None
    return dict(zip([column[0] for column in cursor.description], row))


def record_snapshot(con, captured_at, rows):
    """Append one timestamped snapshot of `rows` and return what was written.

    Returns a dict: total, inserted, updated, skipped.

      inserted  new observation tuples
      updated   the same tuple re-recorded with corrected prices
      skipped   the same tuple re-recorded with identical prices, so nothing
                was written at all -- this is the idempotent path

    Idempotency is enforced twice on purpose: the primary key on
    (CapturedAt, Sportsbook, Date, Home, Away) makes duplication structurally
    impossible at the storage layer, and the explicit pre-read lets an identical
    re-record skip the write entirely and be reported as such. The upsert
    statement is race-safe if two capture processes ever land on the same
    timestamp, which is the one case the pre-read alone cannot cover.
    """
    captured_text = format_captured_at(captured_at)
    rows = list(rows)
    if not rows:
        raise SnapshotError(
            f"refusing to record an empty snapshot for {captured_text}; an empty "
            "capture is a failed capture, not a no-op"
        )

    ensure_snapshot_table(con)
    prepared = [_normalise_row(row) for row in rows]

    inserted = 0
    updated = 0
    skipped = 0
    try:
        # contextlib-style `with con` gives the whole snapshot one transaction:
        # a partial write would record a slate that was never actually posted.
        with con:
            for game_date, sportsbook, home, away, home_ml, away_ml, total in prepared:
                # Named parameters, not positional: the upsert's column list and
                # its ON CONFLICT target are written in different orders (the
                # target follows the primary key), and positional binding across
                # two different orderings is a silent data-corruption bug.
                values = {
                    "CapturedAt": captured_text,
                    "Date": game_date,
                    "Sportsbook": sportsbook,
                    "Home": home,
                    "Away": away,
                    "ML_Home": home_ml,
                    "ML_Away": away_ml,
                    "OU": total,
                }
                existing = _fetch_snapshot(con, values)
                con.execute(SNAPSHOT_UPSERT, values)
                if existing is None:
                    inserted += 1
                elif (existing["ML_Home"], existing["ML_Away"], existing["OU"]) == (
                    home_ml,
                    away_ml,
                    total,
                ):
                    skipped += 1
                else:
                    updated += 1
    except sqlite3.Error as exc:
        raise SnapshotStoreError(
            f"could not record the {captured_text} snapshot to {SNAPSHOT_TABLE}: {exc}"
        ) from exc

    return {
        "captured_at": captured_text,
        "total": len(prepared),
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
    }


def opening_line(con, date, home, away, sportsbook):
    """The FIRST snapshot recorded for this game. Returns None if there are none."""
    return _edge_of_game(con, date, home, away, sportsbook, newest=False)


def closing_line(con, date, home, away, sportsbook):
    """The LAST snapshot recorded for this game. Returns None if there are none.

    This is the close, by definition, as long as the capture job keeps running
    through tip-off. That assumption is worth stating rather than hiding: the
    last row is the last thing anyone looked at, and a capture job that died at
    6pm leaves an afternoon line masquerading as the close. `snapshot_coverage`
    is how you notice.
    """
    return _edge_of_game(con, date, home, away, sportsbook, newest=True)


def _edge_of_game(con, date, home, away, sportsbook, newest):
    """Read the newest or oldest snapshot for one game.

    Ordering comes from "CapturedAt", never from rowid, so a backfill that
    writes out of chronological order still reads back correctly.
    """
    order = "DESC" if newest else "ASC"
    try:
        cursor = con.execute(
            SNAPSHOT_SELECT
            + ' WHERE "Date" = ? AND "Sportsbook" = ? AND "Home" = ? AND "Away" = ?'
            + ' ORDER BY "CapturedAt" ' + order + ' LIMIT 1',
            (
                format_game_date(date),
                str(sportsbook).strip().lower(),
                str(home).strip(),
                str(away).strip(),
            ),
        )
        row = cursor.fetchone()
    except sqlite3.Error as exc:
        # A database that predates this module has no snapshot table at all.
        # That is "no snapshots", not a crash -- and a read must never be the
        # thing that creates a table.
        if "no such table" in str(exc):
            return None
        raise SnapshotStoreError(f"could not read {SNAPSHOT_TABLE}: {exc}") from exc
    if row is None:
        return None
    return dict(zip([column[0] for column in cursor.description], row))


def snapshot_coverage(con, min_snapshots=DEFAULT_MIN_SNAPSHOTS, date=None, sportsbook=None):
    """How many games have at least `min_snapshots` observations, and which do not.

    Returns a dict with total_snapshots, games, games_with_min_snapshots,
    coverage_pct, min/max/mean snapshots per game, a per-date breakdown, and
    `gaps` -- the games below the threshold, each with its actual count.

    READ THIS BEFORE QUOTING ANY CLV NUMBER
    ---------------------------------------
    A game captured once has an "open" and a "close" that are the same scrape.
    Its line movement is exactly zero and its CLV is exactly zero -- not because
    the line held, but because nobody was watching. Those games do not dilute a
    CLV average randomly; they bias it toward zero, and they bias it hardest
    exactly where lines move most (early in the day, before injury news).

    `dense_enough` is a necessary condition, not a sufficient one. Even with
    dense capture, a CLV mean over a few hundred bets needs its standard error
    read before it means anything -- see `clv_report`, and see the selection
    effect in the module docstring before you sweep entry rules and report the
    best one.
    """
    try:
        cursor = con.execute(
            'SELECT "Date", "Sportsbook", "Home", "Away", COUNT(*) AS snapshots FROM "'
            + SNAPSHOT_TABLE
            + '" GROUP BY "Date", "Sportsbook", "Home", "Away"'
        )
        grouped = cursor.fetchall()
    except sqlite3.Error as exc:
        if "no such table" in str(exc):
            grouped = []
        else:
            raise SnapshotStoreError(f"could not read {SNAPSHOT_TABLE}: {exc}") from exc

    wanted_date = format_game_date(date) if date is not None else None
    wanted_book = str(sportsbook).strip().lower() if sportsbook is not None else None

    per_game = []
    for game_date, book, home, away, count in grouped:
        if wanted_date is not None and game_date != wanted_date:
            continue
        if wanted_book is not None and book != wanted_book:
            continue
        per_game.append(
            {
                "Date": game_date,
                "Sportsbook": book,
                "Home": home,
                "Away": away,
                "snapshots": count,
            }
        )

    counts = [entry["snapshots"] for entry in per_game]
    games = len(per_game)
    covered = sum(1 for count in counts if count >= min_snapshots)
    coverage_pct = (covered / games * 100.0) if games else 0.0

    by_date = {}
    for entry in per_game:
        bucket = by_date.setdefault(entry["Date"], {"games": 0, "games_with_min_snapshots": 0})
        bucket["games"] += 1
        if entry["snapshots"] >= min_snapshots:
            bucket["games_with_min_snapshots"] += 1

    gaps = [entry for entry in per_game if entry["snapshots"] < min_snapshots]
    mean_count = (sum(counts) / games) if games else 0.0
    dense_enough = bool(games) and coverage_pct >= 95.0 and mean_count >= min_snapshots

    report = {
        "min_snapshots": min_snapshots,
        "total_snapshots": sum(counts),
        "games": games,
        "games_with_min_snapshots": covered,
        "coverage_pct": coverage_pct,
        "min_snapshots_per_game": min(counts) if counts else 0,
        "max_snapshots_per_game": max(counts) if counts else 0,
        "mean_snapshots_per_game": mean_count,
        "by_date": by_date,
        "gaps": gaps,
        "dense_enough": dense_enough,
    }

    if not games:
        print(
            f"No odds snapshots found in {SNAPSHOT_TABLE}. CLV cannot be measured "
            "until the capture job has run."
        )
    else:
        print(
            f"Snapshot coverage: {covered}/{games} games with >= {min_snapshots} "
            f"snapshots ({coverage_pct:.1f}%), mean {mean_count:.1f} per game."
        )
        for gap in gaps:
            print(
                f"  GAP: {gap['Away']} at {gap['Home']} on {gap['Date']} "
                f"({gap['Sportsbook']}) has {gap['snapshots']} snapshot(s)."
            )
        if not dense_enough:
            print(
                "Coverage is not dense. A single snapshot is indistinguishable from "
                "a line that never moved, so CLV computed over these games is biased "
                "toward zero. Do not report it until coverage improves."
            )
    return report


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _default_scorer(game_date):
    """Fetch the day's scoreboard through sbrscrape.

    Resolves the module global at call time, which is also how the test suite
    substitutes a fake book without patching anything else.
    """
    if Scoreboard is None:
        raise CaptureError(
            "sbrscrape is not installed, so live odds cannot be fetched. Install it "
            "(requirements.txt) to capture snapshots, or drive record_snapshot() "
            "directly. Measurement over existing snapshots needs no network."
        )
    return Scoreboard(date=game_date)


def call_with_timeout(func, args, timeout):
    """Run func(*args) on a worker thread and give up waiting after `timeout`.

    sbrscrape's Scoreboard exposes no timeout and issues its own requests, so
    the only place a bound can be enforced is around the call. The bound is
    real for the caller -- this returns control on time -- but the worker thread
    itself cannot be killed, because CPython has no thread cancellation. A
    timed-out scrape may therefore still finish in the background and be
    discarded; that is the intended trade, and it is why the default is a
    generous 60 seconds rather than something aggressive.
    """
    if timeout is None or timeout <= 0:
        return func(*args)

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)
        except FutureTimeoutError as exc:
            future.cancel()
            raise CaptureError(
                f"odds fetch timed out after {timeout}s. The request may still be "
                "in flight upstream; raising rather than recording an empty slate."
            ) from exc
    finally:
        # wait=False so a stuck scrape cannot hold the process open.
        executor.shutdown(wait=False)


def rows_from_scoreboard(scoreboard, sportsbook):
    """Turn an sbrscrape scoreboard into snapshot rows for one sportsbook.

    A book that has not posted a line for a game is a normal event, not an
    error, so it is counted and reported rather than raised. But it is never
    dropped silently: `missing` carries every game and the reason, and an empty
    result is an error the caller raises on.
    """
    games = getattr(scoreboard, "games", None)
    if not games:
        raise CaptureError("scoreboard response carried no games")

    rows = []
    missing = []
    for game in games:
        try:
            rows.append(
                {
                    "Date": None,  # filled in by the caller; the board has no date field
                    "Sportsbook": sportsbook,
                    "Home": game["home_team"],
                    "Away": game["away_team"],
                    "ML_Home": game["home_ml"][sportsbook],
                    "ML_Away": game["away_ml"][sportsbook],
                    "OU": game["total"][sportsbook],
                }
            )
        except KeyError as exc:
            label = f"{game.get('home_team', '?')} vs {game.get('away_team', '?')}"
            missing.append(label)
            print(f"No {sportsbook} odds for {label} (missing {exc}).")

    if not rows:
        raise CaptureError(
            f"{sportsbook} returned {len(games)} games but no usable odds for any "
            f"of them: {', '.join(missing)}. Recording an empty snapshot would look "
            "like a settled slate and is not one."
        )
    return rows, missing


def capture_once(sportsbook, at_time=None, game_date=None, con=None, db_path=None,
                 timeout=REQUEST_TIMEOUT_SECONDS, scorer=None):
    """Fetch the live slate for one sportsbook and record it as a snapshot.

    at_time    the observation instant; defaults to now, UTC. It is the whole
               point of this module, so it is a parameter rather than an
               internal call to datetime.now() buried in a fetch helper.
    game_date  the slate to pull; defaults to at_time's UTC date.
    con        an open connection to reuse. When None, one is opened on
               db_path and closed again (contextlib.closing, as elsewhere).
    timeout    seconds to wait for the fetch before giving up.

    Raises CaptureError on every failure path -- missing sbrscrape, a fetch that
    throws, a fetch that times out, a slate with no games, a slate with no odds
    for this book, an unwritable database. There is no branch that returns an
    empty result to be mistaken for a slate with nothing on it.

    Sleeps a random 1-3 seconds afterwards, matching Get_Odds_Data.py. Snapshot
    capture hammers the same host far more often per day than a once-a-day
    scrape does, so this delay is doing more work here, not less.
    """
    sportsbook = str(sportsbook or "").strip().lower()
    if not sportsbook:
        raise CaptureError("a sportsbook key is required (e.g. fanduel)")

    captured_at = utcnow() if at_time is None else parse_captured_at(at_time)
    slate_date = captured_at.date() if game_date is None else date_cls.fromisoformat(
        format_game_date(game_date)
    )
    fetch = _default_scorer if scorer is None else scorer

    print(f"Fetching {sportsbook} odds for {slate_date} at {format_captured_at(captured_at)}.")
    try:
        scoreboard = call_with_timeout(lambda: fetch(slate_date), (), timeout)
    except CaptureError:
        raise
    except Exception as exc:
        raise CaptureError(f"{sportsbook} fetch failed for {slate_date}: {exc!r}") from exc

    if scoreboard is None:
        raise CaptureError(f"{sportsbook} fetch returned no scoreboard for {slate_date}")

    rows, missing = rows_from_scoreboard(scoreboard, sportsbook)
    for row in rows:
        row["Date"] = slate_date

    if con is not None:
        result = record_snapshot(con, captured_at, rows)
    else:
        target = DB_PATH if db_path is None else db_path
        with contextlib.closing(sqlite3.connect(target)) as owned:
            result = record_snapshot(owned, captured_at, rows)

    result["sportsbook"] = sportsbook
    result["date"] = slate_date.strftime(DATE_FORMAT)
    result["games"] = len(rows)
    result["missing"] = missing
    print(
        f"Snapshot {result['captured_at']} for {result['date']} ({sportsbook}): "
        f"{result['inserted']} inserted, {result['updated']} updated, "
        f"{result['skipped']} unchanged, {len(missing)} game(s) without {sportsbook} odds."
    )

    time.sleep(random.randint(MIN_DELAY_SECONDS, MAX_DELAY_SECONDS))
    return result


# ---------------------------------------------------------------------------
# CLV
# ---------------------------------------------------------------------------


def _close_probability(row, side):
    """De-vig a snapshot's two-way moneyline.

    Returns (fair probability for `side`, the raw American price for `side`).
    Either can be None: a snapshot with a one-sided moneyline cannot be
    de-vigged, and guessing the missing side's price would be inventing data.
    """
    home_ml = row.get("ML_Home")
    away_ml = row.get("ML_Away")
    if home_ml is None or away_ml is None:
        return None, None
    market = devig(home_ml, away_ml)
    close_prob = market["home"] if side == "home" else market["away"]
    close_price = home_ml if side == "home" else away_ml
    return close_prob, close_price


def clv_report(con, bets, sportsbook=None, min_snapshots=DEFAULT_MIN_SNAPSHOTS):
    """Aggregate CLV across a set of bets, with the sample standard error.

    Each bet is a dict with at minimum `entry_prob` and `entry_price`, plus
    `Date`, `Home`, `Away` (and `Sportsbook`, unless the `sportsbook` argument
    supplies it) so the close can be looked up. `side` defaults to "home";
    `entry_prob` and `entry_price` are read as the side you bet.

    Returns a dict: count, evaluated, missing_close, missing_close_games,
    mean_clv, mean_clv_points, stdev, standard_error,
    mean_over_standard_error, per_bet.

    `missing_close` is reported, not hidden. A bet whose close was never
    captured has no CLV, and dropping those silently would quietly select
    exactly the bets that were easiest to observe.

    The standard error is the reason this function exists in this shape. A mean
    CLV of +0.4 points over 12 bets is noise; the same mean over 1,200 is a
    signal worth arguing about. `mean_over_standard_error` makes that ratio
    explicit -- but read the selection-effect warning in the module docstring
    before you sweep entry rules and quote the best one.
    """
    bets = list(bets)
    if not bets:
        raise ValueError("clv_report needs at least one bet; an empty report is not a report")

    per_bet = []
    missing_games = []
    for bet in bets:
        if not isinstance(bet, dict):
            raise ValueError(f"each bet must be a dict, got {type(bet).__name__}")

        side = str(bet.get("side", "home")).strip().lower()
        if side not in ("home", "away"):
            raise ValueError(f"bet side must be 'home' or 'away', got {side!r}")

        game_date = bet.get("Date")
        home = bet.get("Home")
        away = bet.get("Away")
        if not (game_date and home and away):
            raise ValueError(
                f"bet needs Date, Home and Away to look up its close: {bet!r}"
            )

        book = bet.get("Sportsbook", sportsbook)
        close = closing_line(con, game_date, home, away, book)
        label = f"{away} at {home} on {game_date} ({book})"
        if close is None:
            missing_games.append(label)
            per_bet.append({"bet": bet, "clv": None, "reason": "no closing snapshot"})
            continue

        close_prob, close_price = _close_probability(close, side)
        if close_prob is None:
            missing_games.append(label + " (one-sided close)")
            per_bet.append(
                {"bet": bet, "clv": None, "reason": "closing snapshot has a one-sided moneyline"}
            )
            continue

        result = closing_line_value(
            bet["entry_prob"],
            bet["entry_price"],
            close_prob,
            close_price,
        )
        per_bet.append(
            {
                "bet": bet,
                "clv": result["clv"],
                "clv_points": result["clv_points"],
                "entry_prob": result["entry_prob"],
                "entry_price": result["entry_price"],
                "close_prob": result["close_prob"],
                "close_price": result["close_price"],
                "captured_at": close["CapturedAt"],
            }
        )

    values = [entry["clv"] for entry in per_bet if entry["clv"] is not None]
    evaluated = len(values)
    missing_close = len(bets) - evaluated

    report = {
        "count": len(bets),
        "evaluated": evaluated,
        "missing_close": missing_close,
        "missing_close_games": missing_games,
        "mean_clv": None,
        "mean_clv_points": None,
        "stdev": None,
        "standard_error": None,
        "mean_over_standard_error": None,
        "per_bet": per_bet,
    }

    if evaluated == 0:
        print(
            f"CLV report: none of {len(bets)} bet(s) has a closing snapshot. "
            "Capture has not run far enough into the day."
        )
        return report

    mean_clv = statistics.fmean(values)
    report["mean_clv"] = mean_clv
    report["mean_clv_points"] = mean_clv * 100.0

    # statistics.stdev is the sample SD (n-1). With a single bet there is no
    # spread to estimate, so the error columns stay None rather than reporting
    # a fabricated 0.0.
    if evaluated >= 2:
        stdev = statistics.stdev(values)
        standard_error = stdev / sqrt(evaluated)
        report["stdev"] = stdev
        report["standard_error"] = standard_error
        if standard_error > 0:
            report["mean_over_standard_error"] = mean_clv / standard_error

    coverage = snapshot_coverage(con, min_snapshots=min_snapshots)
    report["coverage"] = coverage

    print(
        f"CLV report: {evaluated}/{len(bets)} bet(s) priced against a close, "
        f"{missing_close} missing."
    )
    if missing_close:
        print("  No close captured for: " + "; ".join(missing_games))
    if evaluated >= 2:
        print(
            f"  mean CLV {mean_clv * 100.0:+.3f} points "
            f"(SE {report['standard_error'] * 100.0:.3f}, "
            f"mean/SE {report['mean_over_standard_error']:+.2f})"
        )
    else:
        print("  n < 2: no standard error is estimable. One bet is not a CLV measurement.")
    if report["mean_over_standard_error"] is not None and abs(report["mean_over_standard_error"]) < 2.0:
        print(
            "  |mean/SE| < 2: this is not distinguishable from zero. Before sweeping "
            "entry rules and reporting the best one, note that the best of 100 trials "
            "is optimistic by roughly one standard error."
        )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None):
    """CLI entry point. Additive only: creates its own table, writes its own rows.

    --coverage reports snapshot density without fetching or writing anything.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Record one timestamped odds snapshot. Repeating this through the day "
            "is what makes closing-line value measurable."
        )
    )
    parser.add_argument(
        "--sportsbook",
        default=DEFAULT_SPORTSBOOK,
        help=(
            "Sportsbook key, or a comma-separated list to capture several in one "
            "pass, or 'all' for the known set: %s (default: %%(default)s). "
            "Rotation across books matters because one book's line is not 'the' "
            "closing line." % ", ".join(KNOWN_SPORTSBOOKS)
        ),
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Stay resident and capture every --every seconds instead of exiting. "
             "Equivalent to running this under a systemd --user timer.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=900,
        help="Seconds between passes when --watch is set (default: %(default)s).",
    )
    parser.add_argument(
        "--db",
        default=DB_PATH,
        help="SQLite file holding the odds tables (default: %(default)s).",
    )
    parser.add_argument(
        "--at",
        default=None,
        help=(
            "Observation time, ISO-8601 with a UTC offset, e.g. "
            "2026-01-05T18:30:00+00:00. Defaults to now, UTC. Naive timestamps "
            "are rejected."
        ),
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Slate to fetch. Defaults to the date of --at.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=REQUEST_TIMEOUT_SECONDS,
        help="Seconds to wait for the fetch (default: %(default)s).",
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Report snapshot density instead of capturing. Fetches and writes nothing.",
    )
    parser.add_argument(
        "--min-snapshots",
        type=int,
        default=DEFAULT_MIN_SNAPSHOTS,
        help="Snapshots a game needs to count as covered (default: %(default)s).",
    )
    args = parser.parse_args(argv)

    if args.coverage:
        with contextlib.closing(sqlite3.connect(args.db)) as con:
            snapshot_coverage(con, min_snapshots=args.min_snapshots)
        return 0

    try:
        books = parse_sportsbooks(args.sportsbook)
    except LineMovementError as exc:
        print(f"CAPTURE FAILED: {exc}")
        return 1

    # Validate --at before the database is touched: a bad argument should not
    # leave an empty sqlite file behind as evidence that a job "ran".
    try:
        captured_at = None if args.at is None else parse_captured_at(args.at)
    except LineMovementError as exc:
        print(f"CAPTURE FAILED [{','.join(books)}]: {exc}")
        return 1

    def one_pass():
        """Capture every configured book once. Returns the number that failed."""
        failures = 0
        for book in books:
            try:
                with contextlib.closing(sqlite3.connect(args.db)) as con:
                    capture_once(
                        book,
                        at_time=captured_at,
                        game_date=args.date,
                        con=con,
                        timeout=args.timeout,
                    )
            except LineMovementError as exc:
                # A non-zero exit is the monitoring signal. The daily job that
                # "succeeded" while doing nothing is exactly the failure mode
                # Level 5 is about.
                print(f"CAPTURE FAILED [{book}]: {exc}")
                failures += 1
        return failures

    if not args.watch:
        return 1 if one_pass() else 0

    # --watch: stay resident and capture on an interval.
    if args.every <= 0:
        print("CAPTURE FAILED: --every must be a positive number of seconds")
        return 1
    print(f"Watching {len(books)} book(s) every {args.every}s. Ctrl-C to stop.")
    failures = 0
    try:
        while True:
            failures = one_pass()
            time.sleep(args.every)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())