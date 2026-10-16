# NBA data download

`download_nba_data.py` builds the dataset every model here trains on. The
canonical copy is **this one**, inside the repository, so the pipeline is version
controlled and reviewable. An earlier working copy lived outside the repo; it is
now superseded and any further change belongs here.

```bash
python scripts/download_nba_data.py --out data/nba          # core, 11 seasons
python scripts/download_nba_data.py --stages core,extras
python scripts/download_nba_data.py --stages rotations --start 2024 --end 2024 --max-games 20
```

See `DATA_NOTES.md` (written into the output directory on first run) for the schema
and its traps. The four that matter most:

- `schedule/` is camelCase; every other table is UPPERCASE. `dtype={'GAME_ID': str}`
  applies to the game logs, not the schedule, which needs `dtype={'gameId': str}`.
- `GAME_DATE` is a string, in two formats across tables. String-sorting it yields an
  inverted range.
- `schedule/` `gameStatus` is `3` (Final). Filtering `== 1` returns zero rows.
- `schedule/` carries All-Star, Rising Stars and preseason games under tricode-like
  names, so a season appears to have 34-41 teams. Filter, or derive home/away from
  the game logs' `MATCHUP` field instead.

## Endpoint health, measured

Per-game endpoints on stats.nba.com, probed with identical headers from one IP:

| endpoint | status |
|---|---|
| `boxscoretraditionalv3`, `boxscoreadvancedv3`, `boxscoreusagev3`, `boxscoretraditionalv2`, `boxscoresummaryv2` | 200, reliable |
| `gamerotation` | ~25% success, HTTP 500 with an empty body; needs a large retry budget |
| `playbyplayv3`, `winprobabilitypbp` | 0 of 12. HTTP 500 with an empty body — currently dead |

Play-by-play therefore falls back to
`cdn.nba.com/static/json/liveData/playbyplay/playbyplay_<GAME_ID>.json`, which serves
**completed** games as well as live ones, so retrospective game state is recoverable.
That host rate-limits aggressively — five retries at 12s intervals after one success
all returned 403 — so the fallback retries with exponential backoff.

## Leakage

`snapshots/`, `style/` and `shots/` are **season-end aggregates**. Using one for a game
inside the same season leaks the future, most sharply `player_estimated_metrics`, which
NBA publishes retroactively. They are priors for the following season only. Build
in-season rolling features from `game_logs/` and `*_gamelogs/`, which are true per-game
rows.

`*_RANK` columns are dropped at download time by design: they encode rank as of pull
time, not as of any game.
