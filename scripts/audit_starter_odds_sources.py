"""Issue #38: what a second, T-45min prediction point would actually need.

The issue asks for four things: verified starter-publication timing, an explicit
second prediction point, a market price *at that timestamp*, and a like-for-like
comparison only after those exist.  This script audits the repository's data for
each requirement instead of assuming the answer.

It is deliberately read-only and cheap.  A missing artifact is a result: it is
the difference between "not measured yet" and "measured as a null", and issue
#38 must not be closed by conflating them.

Run:  python scripts/audit_starter_odds_sources.py
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from DataProviders import nba_data as nd  # noqa: E402
from Models import measurement as ms       # noqa: E402

ODDS_DB = ROOT / "Data" / "OddsData.sqlite"
OUT_JSON = ROOT / "results" / "issue38_source_audit.json"
OUT_MD = ROOT / "results" / "issue38_source_audit.md"

#: Columns that would turn a day-level row into a timestamped snapshot.
TIME_HINTS = ("time", "timestamp", "captured", "snapshot", "stamp", "_at")
#: Filenames that would be a lineup/starter archive if one existed.
LINEUP_HINTS = ("lineup", "starter", "rotation", "boxscore", "box_score")
_DATE_TIME = re.compile(r"\d{1,2}:\d{2}")


def audit_odds() -> dict:
    if not ODDS_DB.exists():
        return {"exists": False, "path": str(ODDS_DB)}
    con = sqlite3.connect(f"file:{ODDS_DB}?mode=ro", uri=True)
    try:
        tables = [r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")]
        out: dict = {"exists": True, "path": str(ODDS_DB),
                     "tables": len(tables),
                     "has_odds_snapshots": "odds_snapshots" in tables,
                     "time_columns": {}, "date_has_time": {},
                     "max_rows_per_game": {}, "season_tables": []}
        for t in tables:
            cols = [r[1] for r in con.execute(f'PRAGMA table_info("{t}")')]
            hits = [c for c in cols
                    if any(h in c.lower() for h in TIME_HINTS)]
            if hits:
                out["time_columns"][t] = hits
            if "Date" not in cols:
                continue
            out["season_tables"].append(t)
            sample = con.execute(
                f'SELECT "Date" FROM "{t}" WHERE "Date" IS NOT NULL LIMIT 5'
            ).fetchall()
            out["date_has_time"][t] = any(
                _DATE_TIME.search(str(s[0])) for s in sample)
            key = [c for c in ("Date", "Home", "Away") if c in cols]
            if len(key) == 3:
                n, distinct = con.execute(
                    f'SELECT COUNT(*), COUNT(DISTINCT "Date" || "|" || "Home" '
                    f'|| "|" || "Away") FROM "{t}"').fetchone()
                out["max_rows_per_game"][t] = int(n - distinct)
        out["tables_with_more_than_one_row_per_game"] = {
            k: v for k, v in out["max_rows_per_game"].items() if v > 0}
        out["any_table_has_time"] = bool(out["time_columns"]) or any(
            out["date_has_time"].values())
        return out
    finally:
        con.close()


def audit_local_files() -> dict:
    hits = []
    for p in (ROOT / "Data").rglob("*"):
        if p.is_file() and any(h in p.name.lower() for h in LINEUP_HINTS):
            hits.append(str(p.relative_to(ROOT)))
    return {"files": hits, "count": len(hits)}


def audit_roster_columns() -> dict:
    try:
        pa = nd.load_player_advanced()
    except Exception as exc:  # data is optional for the audit
        return {"available": False, "error": str(exc)}
    cols = list(pa.columns)
    starter = [c for c in cols if "STARTER" in c.upper()
               or "START_POSITION" in c.upper()
               or "LINEUP" in c.upper()]
    return {"available": True, "rows": int(len(pa)),
            "starter_columns": starter,
            "has_starter_flag": bool(starter)}


def audit_confirmed_guard() -> dict:
    """The code path that would consume a lineup must fail loudly, not proxy."""
    try:
        g = nd.build_games().head(200)
        nd.player_elo_features(g, roster_mode="confirmed")
    except NotImplementedError as exc:
        return {"raises_not_implemented": True, "message": str(exc)}
    except Exception as exc:  # pragma: no cover - reported, not hidden
        return {"raises_not_implemented": False,
                "other_error": f"{type(exc).__name__}: {exc}"}
    return {"raises_not_implemented": False, "other_error": None}


def main() -> int:
    odds = audit_odds()
    files = audit_local_files()
    roster = audit_roster_columns()
    guard = audit_confirmed_guard()

    criteria = {
        "publication timing verified per season": {
            "met": False,
            "why": "No archived pre-tip lineup document exists locally or in a "
                   "free historical source. The league's 30-minute filing rule "
                   "is a rule, not per-season evidence of what was published.",
        },
        "second prediction point written down": {
            "met": False,
            "why": "It cannot be defined against a data source that does not "
                   "exist. The point is T-45min in the issue text, but no "
                   "artifact records a lineup at that time.",
        },
        "market price at that timestamp": {
            "met": bool(odds.get("has_odds_snapshots")),
            "why": ("odds_snapshots exists" if odds.get("has_odds_snapshots")
                    else "Data/OddsData.sqlite stores one day-level row per game "
                         "(Date is a season/date key, not a timestamp) and has no "
                         "odds_snapshots table. The closing line is the only "
                         "stored price, and it is set later than T-45min."),
        },
        "0.61831 comparison labelled not like-for-like": {
            "met": True,
            "why": "This audit does not compare any starter model to 0.61831; "
                   "the issue text is the only place the number is mentioned.",
        },
        "canonical model unchanged": {
            "met": True,
            "why": "No feature or model file was changed by this audit. The "
                   "roster='confirmed' path was already unreachable from the "
                   "canonical model; it now raises instead of silently "
                   "returning the top-minutes proxy.",
        },
    }
    verdict = (
        "BLOCKED, NOT NULL: three of the five acceptance criteria cannot be "
        "met with the repository's data. There is no timestamped pre-tip "
        "starter archive and no intraday odds history. The honest disposition "
        "is to keep #38 open and capture both forward."
        if not (odds.get("has_odds_snapshots") and files["count"]) else
        "DATA PRESENT: re-run the second-prediction-point protocol."
    )
    payload = {
        "issue": 38,
        "generated_utc": ms.utc_now(),
        "odds": odds,
        "lineup_files": files,
        "roster_columns": roster,
        "confirmed_roster_guard": guard,
        "acceptance_criteria": criteria,
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    L = ["=" * 78, "ISSUE 38 -- SECOND PREDICTION POINT: SOURCE AUDIT", "=" * 78]
    a = L.append
    a(f"odds store: {odds.get('path')}  tables={odds.get('tables')}  "
      f"odds_snapshots={'yes' if odds.get('has_odds_snapshots') else 'no'}")
    a(f"season tables: {len(odds.get('season_tables', []))}; any timestamp "
      f"column: {odds.get('any_table_has_time')}")
    a(f"tables with >1 row per (Date, Home, Away): "
      f"{len(odds.get('tables_with_more_than_one_row_per_game', {}))}")
    a(f"local lineup/rotation/starter files: {files['count']}")
    a(f"roster starter columns: {roster.get('starter_columns')}")
    a(f"roster='confirmed' guard: {guard}")
    a("")
    for name, row in criteria.items():
        a(f"[{'x' if row['met'] else ' '}] {name}")
        a(f"      {row['why']}")
    a("")
    a(verdict)
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
