"""Backfill a full-season closing-line table from ESPN's public core odds API.

WHY THIS EXISTS
---------------
The shipped `odds_2025-26` table is a single early snapshot: it stops on
2026-01-07 and prices 514 of the season's 1,225 games. As a holdout that is both
small (SE ~4.4%) and not a close. ESPN's summary `pickcenter` block is a close,
but it does not separate open from close and does not say which provider it
came from. The core odds endpoint does:

    /v2/sports/basketball/leagues/nba/events/{event}/competitions/{event}/odds

Each provider record carries `open`, `close`, and `current` sub-objects. This
script reads **close** only, preferring DraftKings (id 100), then ESPN BET
(id 58), then any provider with a non-null close whose name does not contain
"Live".

SIGN CONVENTION
---------------
`Spread` in OddsData.sqlite is the negative of the home team's handicap for
2022-23 onward (`L_eff = Spread`), so positive = the home side gives points.
ESPN reports the home handicap directly, so `Spread = -handicap`.

OUTPUT
------
A new table, default `odds_2025-26_espn`, with the same quote columns as the
other odds tables plus `Provider` and `ESPN_event` for provenance. The old
`odds_2025-26` table is never modified; `scripts/edge_search.py` swaps the
holdout explicitly.

USAGE
-----
    python scripts/backfill_espn_close.py --season 2025-26 \\
        --start 2025-10-21 --end 2026-06-21
    python scripts/backfill_espn_close.py --from-csv /tmp/espn_close.csv

The second form re-imports an existing download and is what the committed table
was built from.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

import closing_line as cl  # noqa: E402

SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba/events"
QUOTE_COLS = ["Date", "Home", "Away", "ML_Home", "ML_Away", "OU", "Spread"]
PROVIDER_RANK = {"100": 0, "58": 1}   # DraftKings, ESPN BET
DEFAULT_TABLE = "odds_2025-26_espn"


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def get(url: str, tries: int = 3):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=20)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            log(f"HTTP {r.status_code} {url}")
        except Exception as exc:  # noqa: BLE001
            log(f"ERR {type(exc).__name__} {url}")
        time.sleep(1.0 * (i + 1))
    return None


def american(text):
    if text is None:
        return None
    s = str(text).strip().replace("+", "")
    try:
        return float(s)
    except ValueError:
        return None


def close_line(event_id: str):
    """One closing-line row for an event, or None if no usable close exists."""
    listing = get(f"{CORE}/{event_id}/competitions/{event_id}/odds")
    if not listing:
        return None
    ranked = []
    for item in listing.get("items") or []:
        prov = item.get("provider") or {}
        pid = str(prov.get("id", ""))
        name = str(prov.get("name", ""))
        rank = PROVIDER_RANK.get(pid, 2)
        if "live" in name.lower():
            rank = 9
        ranked.append((rank, name, item.get("$ref")))
    ranked.sort(key=lambda x: x[0])
    for _rank, name, ref in ranked:
        if not ref:
            continue
        detail = get(ref)
        if not detail:
            continue
        close = detail.get("close") or {}
        home = ((detail.get("homeTeamOdds") or {}).get("close") or {})
        away = ((detail.get("awayTeamOdds") or {}).get("close") or {})
        total = (close.get("total") or {}).get("alternateDisplayValue")
        hsp = (home.get("pointSpread") or {}).get("alternateDisplayValue")
        mlh = (home.get("moneyLine") or {}).get("alternateDisplayValue")
        mla = (away.get("moneyLine") or {}).get("alternateDisplayValue")
        if None in (total, hsp, mlh, mla):
            continue
        try:
            return {"Spread": -float(hsp), "OU": float(total),
                    "ML_Home": american(mlh), "ML_Away": american(mla),
                    "Provider": name}
        except ValueError:
            continue
    return None


def one_date(d: date) -> list[dict]:
    ds = d.strftime("%Y%m%d")
    board = get(f"{SITE}/scoreboard?dates={ds}")
    if not board:
        return []
    rows = []
    for event in board.get("events", []):
        comp = (event.get("competitions") or [{}])[0]
        status = ((comp.get("status") or {}).get("type") or {}).get("name", "")
        if status not in ("STATUS_FINAL", "STATUS_FINAL_OT"):
            continue
        home = away = None
        for t in comp.get("competitors", []):
            name = (t.get("team") or {}).get("displayName")
            if t.get("homeAway") == "home":
                home = name
            elif t.get("homeAway") == "away":
                away = name
        if not home or not away:
            continue
        line = close_line(event["id"])
        if not line:
            continue
        rows.append({"Date": d.strftime("%Y-%m-%d"), "Home": home, "Away": away,
                     **line, "ESPN_event": event["id"]})
        time.sleep(0.02)
    return rows


def fetch(start: date, end: date) -> pd.DataFrame:
    dates = []
    d = start
    while d <= end:
        dates.append(d)
        d += timedelta(days=1)
    log(f"dates {len(dates)} {start}..{end}")
    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futs = {ex.submit(one_date, dd): dd for dd in dates}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                rows.extend(fut.result())
            except Exception as exc:  # noqa: BLE001
                log(f"DATE ERR {futs[fut]} {type(exc).__name__} {exc}")
            if done % 25 == 0:
                log(f"{done}/{len(dates)} dates, {len(rows)} games")
    return pd.DataFrame(rows)


def canonicalise(df: pd.DataFrame) -> pd.DataFrame:
    """Map source team names onto the names `build_games` translates to.

    ESPN uses "LA Clippers"; every other display name already matches nba_api's
    full_name. Unknown names raise rather than silently becoming unmatched rows.
    """
    mapping = cl.tricode_to_full_name()
    mapping.pop("_alias", None)
    canonical = set(mapping.values())
    manual = {"LA Clippers": "Los Angeles Clippers"}
    out = df.copy()
    for col in ("Home", "Away"):
        out[col] = out[col].replace(manual)
        bad = sorted(set(out[col]) - canonical)
        if bad:
            raise ValueError(f"unmapped team name(s) in {col}: {bad}")
    return out


def validate(df: pd.DataFrame, season: str = "2025-26") -> dict:
    """Match the table against the project's own game logs, both season types."""
    from DataProviders import nba_data as nd

    mapping = cl.tricode_to_full_name()
    mapping.pop("_alias", None)
    full_to_abbr = {v: k for k, v in mapping.items()}
    e = df.copy()
    e["h"] = e["Home"].map(full_to_abbr)
    e["a"] = e["Away"].map(full_to_abbr)
    report = {"rows": int(len(e)), "seasons": {}}
    for season_type in ("Regular Season", "Playoffs"):
        g = nd.build_games(season_type).reset_index(drop=True)
        g = g[g["SEASON"] == season]
        g["d"] = pd.to_datetime(g["GAME_DATE"]).dt.strftime("%Y-%m-%d")
        m = e.merge(g[["GAME_ID", "d", "home", "away"]],
                    left_on=["Date", "h", "a"],
                    right_on=["d", "home", "away"], how="inner")
        report["seasons"][season_type] = {
            "games": int(len(g)),
            "matched": int(len(m)),
            "duplicate_matches": int(m["GAME_ID"].duplicated().sum()),
        }
    return report


def write_table(df: pd.DataFrame, db_path: Path, table: str) -> None:
    missing = [c for c in QUOTE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing quote columns {missing}")
    dup = df.duplicated(subset=["Date", "Home", "Away"], keep=False)
    if dup.any():
        raise ValueError(f"{int(dup.sum())} duplicate game keys; refusing to write")
    con = sqlite3.connect(db_path)
    try:
        with con:
            con.execute(f"DROP TABLE IF EXISTS '{table}'")
            df[QUOTE_COLS + [c for c in ("Provider", "ESPN_event")
                             if c in df.columns]].to_sql(table, con, index=False)
    finally:
        con.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="2025-26")
    ap.add_argument("--start", default="2025-10-21")
    ap.add_argument("--end", default="2026-06-21")
    ap.add_argument("--table", default=DEFAULT_TABLE)
    ap.add_argument("--from-csv", type=Path, default=None)
    ap.add_argument("--csv-out", type=Path, default=None)
    ap.add_argument("--db", type=Path, default=cl.ODDS_DB)
    args = ap.parse_args()

    if args.from_csv:
        df = pd.read_csv(args.from_csv)
        log(f"read {len(df)} rows from {args.from_csv}")
    else:
        df = fetch(date.fromisoformat(args.start), date.fromisoformat(args.end))
        if args.csv_out:
            df.to_csv(args.csv_out, index=False)
            log(f"wrote {args.csv_out} ({len(df)} rows)")

    df = df.drop_duplicates(subset=["Date", "Home", "Away"]).reset_index(drop=True)
    df = canonicalise(df).sort_values(["Date", "Home"]).reset_index(drop=True)
    report = validate(df, args.season)
    write_table(df, args.db, args.table)
    log(f"wrote {args.db}::{args.table} ({len(df)} rows)")
    for k, v in report["seasons"].items():
        log(f"  {k}: {v['matched']}/{v['games']} games priced, "
            f"{v['duplicate_matches']} duplicate matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
