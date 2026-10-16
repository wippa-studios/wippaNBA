import contextlib
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd
import toml

BASE_DIR = Path(__file__).resolve().parents[2]
CONFIG_PATH = BASE_DIR / "config.toml"
DB_PATH = BASE_DIR / "Data" / "OddsData.sqlite"
DATE_FORMAT = "%Y-%m-%d"


def parse_legacy_date(raw_date, base_year, seen_january):
    date_parts = str(raw_date).split('-')
    if len(date_parts) < 3:
        return None, seen_january

    # The modern odds tables already store plain ISO dates (2025-10-05), whose
    # year is explicit. Only the legacy rows carry a 4-digit MMDD tail after the
    # season key, so anything shorter is a real date and is parsed as-is.
    if len(date_parts[2]) <= 2:
        if date_parts[1].startswith("01") and not seen_january:
            seen_january = True
        try:
            return datetime.strptime(str(raw_date).strip()[:10], DATE_FORMAT).date(), seen_january
        except ValueError:
            return None, seen_january

    month = date_parts[2][:2]
    day = date_parts[2][2:]
    if len(month) != 2 or len(day) != 2:
        return None, seen_january

    if month == "01" and not seen_january:
        seen_january = True

    year = int(base_year) + (1 if seen_january else 0)
    date_str = f"{year}-{month}-{day}"
    return datetime.strptime(date_str, DATE_FORMAT).date(), seen_january


def table_exists(con, table_name):
    cursor = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def select_odds_table(con, season_key):
    # Odds table naming drifted over the seasons: the shipped database holds
    # odds_2023-24_new, 2024-25 and odds_2025-26 next to the legacy
    # odds_{season} tables, so the hardcoded name misses the modern seasons.
    # Same candidates as Create_Games.select_odds_table, minus the table this
    # script writes, so a run never re-reads its own output.
    candidates = [
        f"odds_{season_key}",
        f"{season_key}_new",
        f"{season_key}",
    ]
    for table_name in candidates:
        if table_exists(con, table_name):
            return table_name
    return None


def get_columns(con, table_name):
    cursor = con.execute("SELECT name FROM pragma_table_info(?)", (table_name,))
    return [name for (name,) in cursor.fetchall()]


def main():
    config = toml.load(CONFIG_PATH)

    # contextlib.closing closes the handle; the inner `with odds_con` keeps the
    # original commit/rollback semantics for the tables written below.
    with contextlib.closing(sqlite3.connect(DB_PATH)) as odds_con:
        with odds_con:
            for season_key in config["get-data"].keys():
                print(season_key)
                source_table = select_odds_table(odds_con, season_key)
                if source_table is None:
                    print(f"No odds table found for {season_key}.")
                    continue

                # Only the legacy tables carry an "index" column; passing
                # index_col="index" for a modern table raises KeyError.
                index_col = "index" if "index" in get_columns(odds_con, source_table) else None
                odds_df = pd.read_sql_query(
                    f'SELECT * FROM "{source_table}"', odds_con, index_col=index_col
                )
                if odds_df.empty:
                    print(f"No rows found for {source_table}.")
                    continue
                if "Date" not in odds_df.columns:
                    print(f"Missing Date column in {source_table}.")
                    continue

                new_dates = []
                seen_january = False
                base_year = season_key.split("-")[0]

                for raw_date in odds_df["Date"].tolist():
                    parsed_date, seen_january = parse_legacy_date(raw_date, base_year, seen_january)
                    if parsed_date is None:
                        new_dates.append(None)
                        print(f"Skipping invalid date: {raw_date}")
                        continue
                    new_dates.append(str(parsed_date))
                    print(f"Old date = {raw_date} : New date = {parsed_date}")

                odds_df["Date"] = new_dates
                odds_df.drop(odds_df.filter(regex="Unname"), axis=1, inplace=True)
                odds_df.to_sql(f"odds_{season_key}_new", odds_con, if_exists="replace", index=False)


if __name__ == "__main__":
    main()
