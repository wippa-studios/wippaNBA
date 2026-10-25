"""Backfill injury-report PDFs for the dates the model actually needs.

The capture module is deliberately polite: 5-10 s between requests, a hard
request budget per invocation, and an append-only store.  Its ``capture``
command, however, walks *calendar* days.  A season has ~150 game dates and
~215 calendar days, so a range capture spends a third of its requests
recording 404s on days when no report can exist.

This wrapper takes the game dates from the shared ``nba_data.build_games()``
frame and calls ``capture_days`` once per date.  Everything else -- pacing,
budget accounting, absence retry policy, raw-PDF provenance -- stays in the
capture module; this file only chooses *which* dates to ask about.

It is resumable in the same way the capture module is: a date whose slots are
already recorded (200 or a final 404) costs no request, and re-running the same
command picks up where it stopped.

It is also *honest about incompleteness*: a date whose request budget runs out
before its slots are all handled (a retryable 403/5xx eating the headroom) is
flagged ``BUDGET_EXHAUSTED`` on the date line, counted in the TOTAL line as
``budget_exhausted_dates``, and makes the exit code nonzero -- a silently
skipped slot is a silent data drop, and historically every one of them was
invisible (see docs/injury_pipeline.md, "The 2025-26 boundary incident").
Each date is granted ``slots x MAX_ATTEMPTS`` of budget headroom (still inside
the run-wide ``--max-requests`` cap) so one retry cannot starve the rest of a
date.

Run::

    python scripts/backfill_injury_reports.py --from 2022-01-16 --to 2022-06-30 \
        --slot 17:00 --slot 20:00 --max-requests 400

The default budget is intentionally small.  Pass ``--max-requests`` explicitly
for a bulk run; the point is that a forgotten flag cannot become a bulk
download.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from DataProviders import nba_data as nd  # noqa: E402

INJURY_MODULE = ROOT / "src" / "Process-Data" / "Get_Injury_Data.py"


def load_injury_module():
    """Load the hyphen-directory module by path, as the tests do."""
    spec = importlib.util.spec_from_file_location(
        "get_injury_data", INJURY_MODULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def game_dates(date_from: str, date_to: str, games=None) -> list:
    """Unique game dates in the inclusive range, chronologically."""
    import pandas as pd

    g = nd.build_games() if games is None else games
    dates = pd.to_datetime(g["GAME_DATE"]).dt.normalize().drop_duplicates()
    lo, hi = pd.Timestamp(date_from), pd.Timestamp(date_to)
    return [d.date() for d in sorted(dates[(dates >= lo) & (dates <= hi)])]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from", dest="date_from", required=True)
    p.add_argument("--to", dest="date_to", required=True)
    p.add_argument("--slot", action="append", default=None,
                   help="Clock slot, e.g. 17:00 (repeatable; default 17:00)")
    p.add_argument("--max-requests", type=int, default=20,
                   help="Hard cap on HTTP requests across the whole run")
    p.add_argument("--db", default=None)
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-parse", action="store_true",
                   help="Capture PDFs only; skip turning them into status events")
    args = p.parse_args(argv)

    slots = args.slot or ["17:00"]
    dates = game_dates(args.date_from, args.date_to)
    if not dates:
        print("no game dates in range")
        return 1

    gid = load_injury_module()
    total_requests = 0
    total_fetched = 0
    total_absent = 0
    total_errors = 0
    total_parsed = 0
    total_parse_failed = 0
    total_budget_exhausted = 0
    for i, day in enumerate(dates, start=1):
        remaining = args.max_requests - total_requests
        if remaining <= 0:
            print(f"STOPPED budget={args.max_requests} at {day} "
                  f"({i - 1}/{len(dates)} dates)")
            break
        summary = gid.capture_days(
            day, day,
            db_path=args.db, raw_dir=args.raw_dir,
            # Per-date headroom of MAX_ATTEMPTS x slots, capped by the run
            # budget: retries (transport wobbles, CDN 403/5xx refusals, both
            # routine) must not starve a date's later slots out of the budget.
            # The run-wide cap still holds: `remaining` is the hard ceiling and
            # every retry spends one unit of it via summary["requests_made"].
            max_requests=min(len(slots) * gid.MAX_ATTEMPTS, remaining),
            slots=[gid._parse_slot_arg(s) for s in slots],
        )
        total_requests += summary["requests_made"]
        total_fetched += summary["fetched"]
        total_absent += summary["absent"]
        total_errors += summary["errors"]
        # budget_exhausted inside ONE date means the daily request budget ran
        # out before the date's slots were all handled: the date was aborted
        # with no evidence recorded for the remaining slots. That is
        # incomplete evidence, so it is flagged and counted, never silent.
        if summary["budget_exhausted"]:
            total_budget_exhausted += 1
        # A captured PDF that is not parsed is not evidence yet. Parsing is
        # idempotent, so doing it per date keeps an interrupted backfill
        # usable and costs one pdfplumber pass over the day's two files.
        if summary["fetched"] and not args.no_parse:
            with contextlib.closing(gid.open_db(args.db)) as con:
                parsed = gid.parse_all(con, raw_dir=args.raw_dir,
                                       report_date=day)
            total_parsed += parsed["parsed"]
            total_parse_failed += parsed["failed"]
        if not args.quiet or summary["fetched"] or summary["errors"] \
                or summary["budget_exhausted"]:
            budget_note = " BUDGET_EXHAUSTED" if summary["budget_exhausted"] else ""
            print(f"{i:>3}/{len(dates)} {day} requests={summary['requests_made']} "
                  f"fetched={summary['fetched']} absent={summary['absent']} "
                  f"errors={summary['errors']}{budget_note}")

    print(f"TOTAL dates={len(dates)} requests={total_requests} "
          f"fetched={total_fetched} absent={total_absent} errors={total_errors} "
          f"budget_exhausted_dates={total_budget_exhausted} "
          f"parsed={total_parsed} parse_failed={total_parse_failed}")
    return 1 if (total_errors or total_parse_failed or total_budget_exhausted) else 0


if __name__ == "__main__":
    raise SystemExit(main())
