import contextlib
import sqlite3
from bisect import bisect_left
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import toml

from src.Utils.Dictionaries import (
    canonical_team_name,
    team_index_07,
    team_index_08,
    team_index_12,
    team_index_13,
    team_index_14,
    team_index_current,
)

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config.toml"
ODDS_DB_PATH = BASE_DIR / "Data" / "OddsData.sqlite"
TEAMS_DB_PATH = BASE_DIR / "Data" / "TeamData.sqlite"
OUTPUT_DB_PATH = BASE_DIR / "Data" / "dataset.sqlite"
OUTPUT_TABLE = "dataset_2012-26"

# leaguedashteamstats returns one row per team, so a complete snapshot has one
# row per team in the league. The count is DERIVED from the data
# (derive_expected_team_rows) rather than hardcoded: hardcoding 30 meant that at
# 31 or 32 teams every snapshot would fail the completeness test, zero usable
# snapshots would be found, and main() would return quietly having republished
# the previous dataset.
MIN_PLAUSIBLE_TEAM_ROWS = 20

TEAM_INDEX_BY_SEASON = {
    "2007-08": team_index_07,
    "2008-09": team_index_08,
    "2009-10": team_index_08,
    "2010-11": team_index_08,
    "2011-12": team_index_08,
    "2012-13": team_index_12,
    "2013-14": team_index_13,
    "2014-15": team_index_14,
    "2015-16": team_index_14,
    "2016-17": team_index_14,
    "2017-18": team_index_14,
    "2018-19": team_index_14,
    "2019-20": team_index_14,
    "2020-21": team_index_14,
    "2021-22": team_index_14,
    "2022-23": team_index_current,
    "2023-24": team_index_current,
    "2024-25": team_index_current,
    "2025-26": team_index_current,
}


def table_exists(con, table_name):
    cursor = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def normalize_date(value):
    if isinstance(value, datetime):
        return value.date().isoformat()
    if hasattr(value, "date"):
        try:
            return value.date().isoformat()
        except Exception:
            pass
    return str(value)


def parse_snapshot_date(value):
    """Parse a value into a datetime.date, or None when it is not a date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return datetime.strptime(normalize_date(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def get_team_index_map(season_key):
    if season_key in TEAM_INDEX_BY_SEASON:
        return TEAM_INDEX_BY_SEASON[season_key]
    try:
        start_year = int(season_key.split("-")[0])
    except (ValueError, IndexError):
        return team_index_current
    return team_index_current if start_year >= 2022 else team_index_14


def team_table_row_count(teams_con, date_str):
    """Rows in the snapshot table named date_str; 0 when the table is absent."""
    if not table_exists(teams_con, date_str):
        return 0
    return teams_con.execute(f'SELECT COUNT(*) FROM "{date_str}"').fetchone()[0]


def derive_expected_team_rows(row_counts):
    """The league size, taken as the modal row count across dated snapshots.

    The mode is robust here: complete snapshots dominate (4,262 of 4,320 tables
    hold exactly 30 rows) while truncated ones scatter across many small
    counts. A hardcoded 30 broke the moment the league changed size.
    """
    counts = [count for count in row_counts.values() if count > 0]
    if not counts:
        raise ValueError("No dated team snapshot tables contain any rows.")
    expected = Counter(counts).most_common(1)[0][0]
    if expected < MIN_PLAUSIBLE_TEAM_ROWS:
        raise ValueError(
            f"Derived league size of {expected} teams is implausible "
            f"(minimum {MIN_PLAUSIBLE_TEAM_ROWS}). The snapshot tables are "
            "probably truncated or corrupt; refusing to build."
        )
    return expected


def scan_team_tables(teams_con):
    """Return (usable snapshot dates ascending, row count per dated table,
    derived expected team rows).

    A snapshot is usable only when it holds the full team complement.
    leaguedashteamstats omits teams that have not played yet, so the first
    couple of weeks of every season come back short; those dates are excluded
    here so find_prior_snapshot steps past them to the newest COMPLETE snapshot
    instead of silently dropping the game. Scanning once at startup also keeps
    the per-game lookup free of the cost of counting rows again.
    """
    cursor = teams_con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    row_counts = {}
    for (name,) in cursor.fetchall():
        if parse_snapshot_date(name) is None:
            continue
        row_counts[name] = team_table_row_count(teams_con, name)

    expected = derive_expected_team_rows(row_counts)
    usable = sorted(
        parse_snapshot_date(name)
        for name, count in row_counts.items()
        if count == expected
    )
    return usable, row_counts, expected


def find_prior_snapshot(usable_snapshots, game_date):
    """Newest usable snapshot STRICTLY before game_date, or None.

    Strictly before is the entire point of this function. The table named
    game_date holds season-to-date totals that are INCLUSIVE of games played on
    that date, so joining a game's odds row onto the same-day table puts that
    game's own result -- and every *_RANK derived from it -- straight into the
    feature vector. There is deliberately no same-day fallback: if nothing
    predates the game there is no feature row to build, and the caller skips it.
    """
    position = bisect_left(usable_snapshots, game_date)
    if position == 0:
        return None
    return usable_snapshots[position - 1]


def fetch_team_table(teams_con, date_str):
    if not table_exists(teams_con, date_str):
        return None
    return pd.read_sql_query(f'SELECT * FROM "{date_str}"', teams_con)


def load_team_table(teams_con, date_str, cache):
    """Read a snapshot table, memoised so one date is read once per build.

    A night of NBA games shares a single prior snapshot, and every game on that
    date would otherwise re-read the same table from disk.
    """
    if date_str not in cache:
        cache[date_str] = fetch_team_table(teams_con, date_str)
    return cache[date_str]


def build_game_features(team_df, home_team, away_team, index_map, expected_team_rows):
    """Home block + away block for one game, resolved BY NAME not by position.

    The original implementation looked the team up in index_map and indexed
    positionally with .iloc[index]. That silently produced the wrong team's
    features whenever the snapshot came from a different season than the game,
    because the alphabetical index shifts when franchises join or leave.
    Measured on dataset_2012-26: 354 rows (2.06%) used a prior season's
    snapshot, and 21 of them (0.12%) carried a completely different team's
    record -- for example a Cleveland game whose feature block was Dallas's
    end-of-season line, because Brooklyn is in team_index_12 but not in the
    2011-12 frame, shifting every later row by one.

    Resolving by canonical name removes the whole class of bug: it is immune to
    row order, to franchises joining or leaving, and to era-specific spellings.
    index_map remains as a fallback for snapshots that have no TEAM_NAME column
    at all.
    """
    if len(team_df.index) != expected_team_rows:
        return None

    def resolve(team):
        """Position of `team` in this snapshot: by name first, position second."""
        if "TEAM_NAME" in team_df.columns:
            for position, name in enumerate(team_df["TEAM_NAME"]):
                if canonical_team_name(name) == canonical_team_name(team):
                    return position
        # Fall back to the season's index map only when the snapshot carries no
        # usable name for this franchise. This is the old, unsafe path and is
        # reached far less often now; keeping it avoids dropping games whose
        # team naming differs from the snapshot's beyond the alias table.
        position = index_map.get(canonical_team_name(team))
        if position is None or position >= len(team_df.index):
            return None
        return position

    home_row = resolve(home_team)
    away_row = resolve(away_team)
    if home_row is None or away_row is None:
        return None

    home_team_series = team_df.iloc[home_row]
    away_team_series = team_df.iloc[away_row]
    return pd.concat([
        home_team_series,
        away_team_series.rename(index={col: f"{col}.1" for col in team_df.columns.values}),
    ])


def is_missing_result(points, win_margin):
    """True for cancelled games and phantom ties -- rows with no real result.

    A postponed or cancelled game carries no score (Points 0) and a zero margin
    is a tie, which basketball cannot produce; both are scraper artefacts that
    were being written out as an away win and an under, i.e. as training losses
    the model could not possibly have predicted. NULL is caught here too: the
    odds rows arrive through pandas, so a missing score can arrive as None or as
    NaN, and comparing it raised a TypeError.

    NOTE: win_margin is compared with 0, not <= 0. A NEGATIVE margin is an
    ordinary away win and is precisely the label we want to keep -- guarding on
    <= 0 would discard every away win in the dataset.
    """
    if pd.isna(points) or pd.isna(win_margin):
        return True
    return points <= 0 or win_margin == 0


def select_odds_tables(odds_con, season_key):
    """Every odds table that exists for this season, most-preferred first.

    This returns a LIST and the caller unions them, because picking the first
    match silently threw away data.

    Get_Odds_Data.py writes to the bare season key ("2023-24"), while the
    original lookup preferred "odds_2023-24_new". For two of the three
    configured seasons those are different tables, so the scraper's output was
    written somewhere the builder never read -- which also made the earlier
    --backfill fix inert, since backfilled rows landed in the unread table.

    The tables are genuinely complementary, not duplicates. Matching on game
    identity alone (Home, Away, ML_Home, ML_Away, Points) and ignoring Date --
    which had to be ignored because the legacy "odds_YYYY-YY" tables store a
    mangled "YYYY-DD-MMDD" that datetime cannot parse -- "2023-24" holds 145
    games that "odds_2023-24_new" does not, while "odds_2023-24_new" holds 85
    that "2023-24" does not.

    Unparseable tables are not filtered here. parse_snapshot_date() drops their
    rows anyway, and skipping them silently would hide a broken scraper.
    """
    candidates = [
        f"odds_{season_key}_new",
        f"odds_{season_key}",
        f"{season_key}_new",
        f"{season_key}",
    ]
    return [name for name in candidates if table_exists(odds_con, name)]


def load_odds_for_season(odds_con, table_names, teams_con):
    """Union the season's odds tables, deduplicated on game identity.

    Dedup is on the whole row rather than (Date, Home, Away) because two
    tables can carry the same game with different line values, and picking one
    arbitrarily would make the result depend on table order. Rows are keyed on
    the parsed game date plus both team names so an unparseable-date row cannot
    collide with a parseable one.
    """
    frames = []
    for table_name in table_names:
        frame = pd.read_sql_query(f'SELECT * FROM "{table_name}"', odds_con)
        if not frame.empty:
            frame = frame.assign(_source_table=table_name)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    dedup_columns = [
        column for column in ("Date", "Home", "Away")
        if column in combined.columns
    ]
    before = len(combined)
    combined = combined.drop_duplicates(subset=dedup_columns, keep="first")
    if before != len(combined):
        print(
            f"  unioned {len(table_names)} table(s); "
            f"{before - len(combined)} duplicate game rows dropped"
        )
    return combined.reset_index(drop=True)


def main():
    config = toml.load(CONFIG_PATH)

    scores = []
    win_margin = []
    ou_values = []
    ou_cover = []
    games = []
    game_dates = []
    days_rest_away = []
    days_rest_home = []

    # Every odds row is accounted for below so a shrink in the dataset is
    # visible instead of silent. "legacy_same_day_drop" is a measurement, not a
    # skip under the leak-free join: it counts the games the old same-day join
    # would have thrown away on the len(team_df) != 30 rule.
    stats = {
        "odds_rows": 0,
        "kept": 0,
        "no_prior_snapshot": 0,
        "unknown_team": 0,
        "missing_result": 0,
        "unusable_snapshot": 0,
        "unparseable_date": 0,
        "legacy_same_day_drop": 0,
    }
    odds_tables_used = {}
    expected_team_rows = None

    # contextlib.closing closes the handle; the sqlite3 context manager only
    # commits, so nesting `with con:` inside it is what keeps the write below
    # atomic while still releasing the file.
    with contextlib.closing(sqlite3.connect(ODDS_DB_PATH)) as odds_con, contextlib.closing(
        sqlite3.connect(TEAMS_DB_PATH)
    ) as teams_con:
        usable_snapshots, team_table_sizes, expected_team_rows = scan_team_tables(teams_con)
        team_table_cache = {}
        print(
            f"Usable team snapshots: {len(usable_snapshots)} complete tables, "
            f"{len(team_table_sizes) - len(usable_snapshots)} truncated or empty "
            f"(league size derived from the data: {expected_team_rows} teams)."
        )
        if usable_snapshots:
            print(
                f"Snapshot range: {usable_snapshots[0]} to {usable_snapshots[-1]}. "
                "Games are joined against the newest snapshot STRICTLY BEFORE their "
                "own date, so no feature row carries its own result."
            )
        else:
            # Previously this fell through to a bare `return`, leaving the
            # previous dataset in place with a message that read like a routine
            # no-op and a zero exit code. A silent total failure is worse than a
            # crash: the next rebuild republishes stale data.
            raise SystemExit(
                "FATAL: no complete team snapshot found. Every dated table in "
                f"Data/TeamData.sqlite holds other than {expected_team_rows} rows. "
                "Refusing to build a dataset. Re-run Get_Team_Data.py first."
            )

        for season_key in config["create-games"].keys():
            print(season_key)
            odds_tables = select_odds_tables(odds_con, season_key)
            if not odds_tables:
                print(f"Missing odds tables for {season_key}.")
                continue
            odds_tables_used[season_key] = odds_tables

            odds_df = load_odds_for_season(odds_con, odds_tables, teams_con)
            if odds_df.empty:
                print(f"No odds data for {season_key}.")
                continue
            print(f"  reading from: {', '.join(odds_tables)}")

            index_map = get_team_index_map(season_key)
            canonical_index = {canonical_team_name(k) for k in index_map}

            for row in odds_df.itertuples(index=False):
                stats["odds_rows"] += 1

                game_date = parse_snapshot_date(row.Date)
                if game_date is None:
                    stats["unparseable_date"] += 1
                    stats["no_prior_snapshot"] += 1
                    continue

                # Leak-free join. The snapshot has to predate the game: the table
                # named game_date already contains that game's result.
                snapshot = find_prior_snapshot(usable_snapshots, game_date)
                if snapshot is None:
                    stats["no_prior_snapshot"] += 1
                    continue

                # What the old same-day join would have done with this row. This
                # is a counterfactual, NOT a shortfall: none of these games are
                # lost, they are joined against an older complete snapshot.
                if team_table_sizes.get(game_date.strftime("%Y-%m-%d"), 0) != expected_team_rows:
                    stats["legacy_same_day_drop"] += 1

                if (canonical_team_name(row.Home) not in canonical_index
                        or canonical_team_name(row.Away) not in canonical_index):
                    stats["unknown_team"] += 1
                    continue

                team_df = load_team_table(teams_con, snapshot.strftime("%Y-%m-%d"), team_table_cache)
                if team_df is None:
                    stats["unusable_snapshot"] += 1
                    continue

                game = build_game_features(
                    team_df, row.Home, row.Away, index_map, expected_team_rows
                )
                if game is None:
                    # Either the snapshot lost rows between the startup scan and
                    # now, or it genuinely does not contain both franchises --
                    # which is the case for a cross-season snapshot, and is
                    # counted rather than silently mis-attributed to another
                    # team.
                    stats["unusable_snapshot"] += 1
                    continue

                # Guard before anything is appended, so a cancelled game can
                # never reach the labels.
                if is_missing_result(row.Points, row.Win_Margin):
                    stats["missing_result"] += 1
                    continue

                scores.append(row.Points)
                ou_values.append(row.OU)
                # The date the game is actually played on, kept alongside the
                # snapshot Date because the two stopped being the same thing
                # when the leak-free join landed (see find_prior_snapshot).
                game_dates.append(game_date.isoformat())
                days_rest_home.append(row.Days_Rest_Home)
                days_rest_away.append(row.Days_Rest_Away)
                win_margin.append(1 if row.Win_Margin > 0 else 0)

                # 2 is a push: the total landed exactly on the line and the bet
                # is void. It stays a distinct class in the totals head; see
                # docs/totals_target.md for why it is not folded into a side.
                if row.Points < row.OU:
                    ou_cover.append(0)
                elif row.Points > row.OU:
                    ou_cover.append(1)
                else:
                    ou_cover.append(2)

                games.append(game)
                stats["kept"] += 1

    print(
        f"Odds rows read: {stats['odds_rows']} | kept: {stats['kept']} | "
        f"skipped, no usable snapshot before the game: {stats['no_prior_snapshot']} "
        f"(of which unparseable date: {stats['unparseable_date']}) | "
        f"skipped, team not in this season's index: {stats['unknown_team']} | "
        f"skipped, cancelled game or 0-margin tie: {stats['missing_result']} | "
        f"skipped, snapshot unreadable or franchise absent: {stats['unusable_snapshot']}"
    )
    print(
        f"COUNTERFACTUAL (not a shortfall): {stats['legacy_same_day_drop']} rows fall on "
        f"a same-day snapshot holding other than {expected_team_rows} rows. The old join "
        "dropped them; this build keeps them, joined against the newest complete snapshot "
        "instead -- which is why some rows carry an older season's record."
    )

    if not games:
        raise SystemExit(
            "FATAL: no game rows produced from "
            f"{stats['odds_rows']} odds rows. Refusing to leave the previous "
            f"{OUTPUT_TABLE} in place as though the rebuild had succeeded."
        )

    season = pd.concat(games, ignore_index=True, axis=1).T
    frame = season.drop(columns=["TEAM_ID", "TEAM_ID.1"], errors="ignore")
    # Column order is part of the model contract. OU sits between the away-team
    # block and OU-Cover, and the UO trainers keep it there, so it must not
    # move: expected_feature_order()/validate_feature_frame() in
    # src/Utils/features.py exist to keep it that way.
    #
    # GameDate is the real game date from the odds row. It is NOT a feature --
    # every drop list in training_common.drop_columns_for and in
    # src/Utils/features.py removes it by name -- and it is written here, ahead
    # of Score, only because it belongs with the date keys. Every column below
    # it keeps its position relative to every other column below it, so the OU
    # contract above is untouched. Do not "tidy" it into the away-team block or
    # drop it as redundant with Date: it is the only column that joins to
    # Data/OddsData.sqlite, whose Date is the game date, while Date is now the
    # prior snapshot date.
    frame["GameDate"] = np.asarray(game_dates)
    frame["Score"] = np.asarray(scores)
    frame["Home-Team-Win"] = np.asarray(win_margin)
    frame["OU"] = np.asarray(ou_values)
    frame["OU-Cover"] = np.asarray(ou_cover)
    frame["Days-Rest-Home"] = np.asarray(days_rest_home)
    frame["Days-Rest-Away"] = np.asarray(days_rest_away)

    # Only the numeric columns are coerced. The identifier and date columns --
    # Date, Date.1 and GameDate all contain "Date", TEAM_NAME contains "TEAM_" --
    # stay as they were read, so GameDate survives as the ISO string the odds
    # join compares against. Coercing it to float would silently produce a
    # column full of NaN and break that join.
    for field in frame.columns.values:
        if "TEAM_" in field or "Date" in field:
            continue
        frame[field] = frame[field].astype(float)

    with contextlib.closing(sqlite3.connect(OUTPUT_DB_PATH)) as con:
        with con:
            frame.to_sql(OUTPUT_TABLE, con, if_exists="replace", index=False)
    print(f"Wrote {len(frame.index)} rows to {OUTPUT_DB_PATH}:{OUTPUT_TABLE}")

    write_manifest(frame, stats, expected_team_rows, usable_snapshots,
                   team_table_sizes, odds_tables_used)


def write_manifest(frame, stats, expected_team_rows, usable_snapshots,
                   team_table_sizes, odds_tables_used):
    """Record what this build was, so a metrics run can be tied to it.

    Data/dataset.sqlite is both tracked in git and matched by the *.sqlite rule
    in .gitignore, so the committed copy and the working copy can diverge with
    `git status` staying silent. A content fingerprint written next to the data
    is what lets scripts/evaluate.py refuse to report metrics against a table
    that is not the one it names.

    content_sha256 is computed over the sorted column list, the row count and a
    pandas hash of the label, GameDate and Score columns -- cheap, and enough to
    tell four builds of the same schema apart without scanning 37 MB.
    """
    import hashlib
    import json
    import subprocess

    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=BASE_DIR,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or None
        git_dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=BASE_DIR,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        git_sha, git_dirty = None, None

    fingerprint_columns = [
        column for column in ("Home-Team-Win", "GameDate", "Score", "OU-Cover")
        if column in frame.columns
    ]
    digest = hashlib.sha256()
    digest.update(("|".join(sorted(frame.columns))).encode())
    digest.update(str(len(frame.index)).encode())
    digest.update(
        pd.util.hash_pandas_object(frame[fingerprint_columns], index=False).values.tobytes()
    )

    game_dates = pd.to_datetime(frame["GameDate"], errors="coerce")
    lag = None
    if "Date" in frame.columns:
        snapshot_dates = pd.to_datetime(frame["Date"], errors="coerce")
        lag = (game_dates - snapshot_dates).dt.days

    manifest = {
        "generated_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "tables": {
            OUTPUT_TABLE: {
                "rows": int(len(frame.index)),
                "cols": int(len(frame.columns)),
                "content_sha256": digest.hexdigest(),
                "column_list_sha256": hashlib.sha256(
                    "|".join(frame.columns).encode()
                ).hexdigest(),
                "game_date_range": [
                    str(game_dates.min().date()), str(game_dates.max().date()),
                ],
                "build": {
                    "odds_tables_used": odds_tables_used,
                    "expected_team_rows": expected_team_rows,
                    "usable_snapshots": len(usable_snapshots),
                    "truncated_snapshots": len(team_table_sizes) - len(usable_snapshots),
                    "snapshot_lag_days_gt_7": int((lag > 7).sum()) if lag is not None else None,
                    "max_snapshot_lag_days": int(lag.max()) if lag is not None and len(lag) else None,
                    "stats": stats,
                },
                "labels": {
                    "Home-Team-Win": {
                        "positives": int(frame["Home-Team-Win"].sum()),
                        "base_rate": float(frame["Home-Team-Win"].mean()),
                    },
                },
            }
        },
    }
    # Derived from OUTPUT_DB_PATH rather than a module global, so that a caller
    # redirecting the dataset (the tests do) redirects the manifest too.
    manifest_path = OUTPUT_DB_PATH.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
