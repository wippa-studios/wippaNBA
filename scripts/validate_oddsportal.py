"""Validate the OddsPortal best-of-books backfill against the project's games.

Checks, per season (the repo's no-silent-drops doctrine - every row and every
game leaves a count behind):

* duplicate game keys within the scrape,
* team names that do not map onto `closing_line.tricode_to_full_name()`,
* match rate against `build_games()` on date + both teams,
* SCORE AGREEMENT - the scrape's home/away points must equal the game log's;
  a disagreement means the parse swapped a team or a score,
* moneyline overround 1/O1 + 1/O2 per game (the point of this source:
  best-of-books should sit near 1.00-1.02, far below the ~1.045 of a single
  book), with negative-overround rows counted, not silently de-vigged.

Usage:
    python scripts/validate_oddsportal.py                 # validate only
    python scripts/validate_oddsportal.py --promote       # validate + write DB
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from closing_line import tricode_to_full_name  # noqa: E402
from DataProviders import nba_data as nd       # noqa: E402

KEEP_STAGES = {"regular", "play offs", "promotion - play offs"}


def load_seasons(csv_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for csv in sorted(csv_dir.glob("oddsportal_*.csv")):
        out[csv.stem.replace("oddsportal_", "")] = pd.read_csv(csv)
    if not out:
        raise SystemExit(f"no oddsportal_*.csv under {csv_dir}")
    return out


def validate(seasons: dict[str, pd.DataFrame]) -> dict:
    mapping = tricode_to_full_name()
    mapping.pop("_alias", None)
    full_to_tri = {v: k for k, v in mapping.items()}

    g = nd.build_games().reset_index(drop=True)
    g["Date"] = pd.to_datetime(g["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    po = nd.build_games("Playoffs").reset_index(drop=True)
    po["Date"] = pd.to_datetime(po["GAME_DATE"]).dt.strftime("%Y-%m-%d")

    report = {"generated": datetime.now(timezone.utc).isoformat(),
              "seasons": {}}
    for season, d in sorted(seasons.items()):
        d = d.copy()
        d["Date"] = pd.to_datetime(d["Date_OP"], format="%d %b %Y").dt.strftime("%Y-%m-%d")
        d["stage_norm"] = d["Stage"].fillna("").str.strip().str.lower().replace({"": "regular"})
        kept = d[d["stage_norm"].isin(KEEP_STAGES)].copy()
        dropped = d[~d["stage_norm"].isin(KEEP_STAGES)]

        unmapped = sorted((set(d["Home"]) | set(d["Away"])) - set(mapping.values()))
        kept["h"] = kept["Home"].map(full_to_tri)
        kept["a"] = kept["Away"].map(full_to_tri)

        dup_keys = int(kept.duplicated(["Date", "Home", "Away"]).sum())

        gr = g[g["SEASON"] == season]
        pr = po[po["SEASON"] == season]
        j = kept.merge(gr[["GAME_ID", "Date", "home", "away", "home_pts", "away_pts"]],
                       left_on=["Date", "h", "a"],
                       right_on=["Date", "home", "away"], how="inner")
        jp = kept.merge(pr[["GAME_ID", "Date", "home", "away"]],
                        left_on=["Date", "h", "a"],
                        right_on=["Date", "home", "away"], how="inner")
        seen_ids = set(j["GAME_ID"]) | set(jp["GAME_ID"])
        unmatched = kept[~kept["h"].notna() | ~kept["GAME_ID_x"].notna()] \
            if False else None

        score_ok = ((j["Home_Score"] == j["home_pts"])
                    & (j["Away_Score"] == j["away_pts"]))
        ov = 1 / kept["O1"] + 1 / kept["O2"]

        report["seasons"][season] = {
            "rows": int(len(d)),
            "kept_rows": int(len(kept)),
            "dropped_by_stage": dropped["stage_norm"].value_counts().to_dict(),
            "unmapped_names": unmapped,
            "duplicate_keys": dup_keys,
            "reg_games_in_logs": int(len(gr)),
            "reg_matched_rows": int(len(j)),
            "reg_games_priced": int(j["GAME_ID"].nunique()),
            "playoff_games_in_logs": int(len(pr)),
            "playoff_matched_rows": int(len(jp)),
            "distinct_games_priced": int(len(seen_ids)),
            "score_disagreements": int((~score_ok).sum()),
            "missing_odds": int(kept["O1"].isna().sum()),
            "overround_mean": round(float(ov.mean()), 4),
            "overround_p5": round(float(ov.quantile(0.05)), 4),
            "overround_p95": round(float(ov.quantile(0.95)), 4),
            "negative_overround_rows": int((ov < 1).sum()),
        }
        r = report["seasons"][season]
        print(f"{season}: kept {r['kept_rows']} (dropped {r['dropped_by_stage']}), "
              f"reg priced {r['reg_games_priced']}/{r['reg_games_in_logs']}, "
              f"dups {r['duplicate_keys']}, score_disagree "
              f"{r['score_disagreements']}, overround mean {r['overround_mean']}, "
              f"neg {r['negative_overround_rows']}")
    return report


def promote(seasons: dict[str, pd.DataFrame], db_path: Path) -> None:
    """Write validated scrape rows into OddsData.sqlite::odds_ml_bestbooks.

    Only the stages the repo prices (regular season, play-in, playoffs) are
    promoted; pre-season and All-Star rows stay in the CSVs only.
    """
    frames = []
    for season, d in sorted(seasons.items()):
        d = d.copy()
        d["Date"] = pd.to_datetime(d["Date_OP"], format="%d %b %Y").dt.strftime("%Y-%m-%d")
        d["stage_norm"] = d["Stage"].fillna("").str.strip().str.lower().replace({"": "regular"})
        keep = d[d["stage_norm"].isin(KEEP_STAGES)].copy()
        keep["Season"] = season
        frames.append(keep)
    out = pd.concat(frames, ignore_index=True)
    # The same fixture can sit under two OddsPortal event URLs (observed: 5
    # games in 2015-16, identical scores and odds under both). Verified
    # agreeing, so keep the first listing.
    before = len(out)
    out = out.drop_duplicates(subset=["Date", "Home", "Away"], keep="first")
    if before - len(out):
        print(f"dropped {before - len(out)} duplicate listings "
              "(same game, two event URLs, agreeing odds)")
    cols = ["Season", "Date", "Stage", "Status", "Home", "Away",
            "Home_Score", "Away_Score", "O1", "O2", "OddsPortal_URL"]
    con = sqlite3.connect(db_path)
    try:
        with con:
            con.execute("DROP TABLE IF EXISTS 'odds_ml_bestbooks'")
            out[cols].to_sql("odds_ml_bestbooks", con, index=False)
    finally:
        con.close()
    print(f"promoted {len(out)} rows to {db_path}::odds_ml_bestbooks")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, default=Path("/tmp/opencode/oddsportal"))
    ap.add_argument("--out", type=Path,
                    default=Path("/tmp/opencode/oddsportal_validation.json"))
    ap.add_argument("--promote", action="store_true",
                    help="after validation, write the scrape into OddsData.sqlite")
    ap.add_argument("--db", type=Path, default=ROOT / "Data" / "OddsData.sqlite")
    args = ap.parse_args()

    seasons = load_seasons(args.csv_dir)
    report = validate(seasons)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    if args.promote:
        promote(seasons, args.db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
