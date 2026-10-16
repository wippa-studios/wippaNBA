import contextlib
import sqlite3
from bisect import bisect_left
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import toml

from src.Utils.Dictionaries import (
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

# leaguedashteamstats returns one row per team, so a complete snapshot has 30.
# It also omits teams that have not played yet, which makes early-season
# snapshots come back short.
EXPECTED_TEAM_ROWS = 30

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


def scan_team_tables(teams_con):
    """Return (usable snapshot dates ascending, row count per dated table).

    A snapshot is usable only when it holds the full 30-team complement.
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

    usable = sorted(
        parse_snapshot_date(name)
        for name, count in row_counts.items()
        if count == EXPECTED_TEAM_ROWS
    )
    return usable, row_counts


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


def build_game_features(team_df, home_team, away_team, index_map):
    home_index = index_map.get(home_team)
    away_index = index_map.get(away_team)
    if home_index is None or away_index is None:
        return None
    if len(team_df.index) != EXPECTED_TEAM_ROWS:
        return None

    home_team_series = team_df.iloc[home_index]
    away_team_series = team_df.iloc[away_index]
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


def select_odds_table(odds_con, season_key):
    candidates = [
        f"odds_{season_key}_new",
        f"odds_{season_key}",
        f"{season_key}_new",
        f"{season_key}",
    ]
    for table_name in candidates:
        if table_exists(odds_con, table_name):
            return table_name
    return None


def main():
    config = toml.load(CONFIG_PATH)

    scores = []
    win_margin = []
    ou_values = []
    ou_cover = []
    games = []
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
        "legacy_same_day_drop": 0,
    }

    # contextlib.closing closes the handle; the sqlite3 context manager only
    # commits, so nesting `with con:` inside it is what keeps the write below
    # atomic while still releasing the file.
    with contextlib.closing(sqlite3.connect(ODDS_DB_PATH)) as odds_con, contextlib.closing(
        sqlite3.connect(TEAMS_DB_PATH)
    ) as teams_con:
        usable_snapshots, team_table_sizes = scan_team_tables(teams_con)
        team_table_cache = {}
        print(
            f"Usable team snapshots: {len(usable_snapshots)} complete tables, "
            f"{len(team_table_sizes) - len(usable_snapshots)} truncated or empty."
        )
        if usable_snapshots:
            print(
                f"Snapshot range: {usable_snapshots[0]} to {usable_snapshots[-1]}. "
                "Games are joined against the newest snapshot STRICTLY BEFORE their "
                "own date, so no feature row carries its own result."
            )

        for season_key in config["create-games"].keys():
            print(season_key)
            odds_table = select_odds_table(odds_con, season_key)
            if not odds_table:
                print(f"Missing odds tables for {season_key}.")
                continue

            odds_df = pd.read_sql_query(f'SELECT * FROM "{odds_table}"', odds_con)
            if odds_df.empty:
                print(f"No odds data for {season_key}.")
                continue

            index_map = get_team_index_map(season_key)

            for row in odds_df.itertuples(index=False):
                stats["odds_rows"] += 1

                game_date = parse_snapshot_date(row.Date)
                if game_date is None:
                    stats["no_prior_snapshot"] += 1
                    continue

                # Leak-free join. The snapshot has to predate the game: the table
                # named game_date already contains that game's result.
                snapshot = find_prior_snapshot(usable_snapshots, game_date)
                if snapshot is None:
                    stats["no_prior_snapshot"] += 1
                    continue

                # What the old same-day join would have done with this row.
                if team_table_sizes.get(game_date.strftime("%Y-%m-%d"), 0) != EXPECTED_TEAM_ROWS:
                    stats["legacy_same_day_drop"] += 1

                if row.Home not in index_map or row.Away not in index_map:
                    stats["unknown_team"] += 1
                    continue

                team_df = load_team_table(teams_con, snapshot.strftime("%Y-%m-%d"), team_table_cache)
                if team_df is None:
                    stats["unusable_snapshot"] += 1
                    continue

                game = build_game_features(team_df, row.Home, row.Away, index_map)
                if game is None:
                    # The scan already filtered on row count, so reaching here
                    # means the table lost rows between the startup scan and now.
                    stats["unusable_snapshot"] += 1
                    continue

                # Guard before anything is appended, so a cancelled game can
                # never reach the labels.
                if is_missing_result(row.Points, row.Win_Margin):
                    stats["missing_result"] += 1
                    continue

                scores.append(row.Points)
                ou_values.append(row.OU)
                days_rest_home.append(row.Days_Rest_Home)
                days_rest_away.append(row.Days_Rest_Away)
                win_margin.append(1 if row.Win_Margin > 0 else 0)

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
        f"skipped, no usable snapshot before the game: {stats['no_prior_snapshot']} | "
        f"skipped, team not in this season's index: {stats['unknown_team']} | "
        f"skipped, cancelled game or 0-margin tie: {stats['missing_result']} | "
        f"skipped, snapshot unreadable: {stats['unusable_snapshot']}"
    )
    print(
        f"Same-day rows that the old join would have dropped on the "
        f"'len(team_df) != {EXPECTED_TEAM_ROWS}' rule: {stats['legacy_same_day_drop']} "
        "(they are no longer dropped for that reason: the join reaches back to "
        "the newest complete snapshot instead, and only games with no complete "
        "snapshot at all are skipped)."
    )

    if not games:
        print("No game rows produced. Check odds and team tables.")
        return

    season = pd.concat(games, ignore_index=True, axis=1).T
    frame = season.drop(columns=["TEAM_ID", "TEAM_ID.1"], errors="ignore")
    # Column order is part of the model contract. OU sits between the away-team
    # block and OU-Cover, and the UO trainers keep it there, so it must not
    # move: expected_feature_order()/validate_feature_frame() in
    # src/Utils/features.py exist to keep it that way.
    frame["Score"] = np.asarray(scores)
    frame["Home-Team-Win"] = np.asarray(win_margin)
    frame["OU"] = np.asarray(ou_values)
    frame["OU-Cover"] = np.asarray(ou_cover)
    frame["Days-Rest-Home"] = np.asarray(days_rest_home)
    frame["Days-Rest-Away"] = np.asarray(days_rest_away)

    for field in frame.columns.values:
        if "TEAM_" in field or "Date" in field:
            continue
        frame[field] = frame[field].astype(float)

    with contextlib.closing(sqlite3.connect(OUTPUT_DB_PATH)) as con:
        with con:
            frame.to_sql(OUTPUT_TABLE, con, if_exists="replace", index=False)
    print(f"Wrote {len(frame.index)} rows to {OUTPUT_DB_PATH}:{OUTPUT_TABLE}")


if __name__ == "__main__":
    main()
