# Data provenance

This page explains the one thing a fresh clone notices first: there is no data
here. It records what the published results were built from, how each artefact
is produced, and what cannot be rebuilt from this repository.

## 1. No data is shipped, on purpose

The repository contains code, scripts, results, and the provenance that says
what the results were built from. The datasets themselves are not committed:

- **Licence.** `LICENSE` is proprietary ("Copyright (c) 2026 wippa studios.
  All rights reserved") and grants no reuse rights. The Limitations section of
  `README.md` (item 6) states the upstream data is not licensed for
  redistribution and that neither this fork nor its upstream grants reuse
  rights. Committing the datasets would redistribute them.
- **Size.** The derived artefacts are large sqlite databases and a
  multi-hundred-file CSV tree. They are build products, not source.
- **Mechanics.** Git tracks exactly one file under `Data/`:
  `Data/manifest.json`. Every `*.sqlite` path is excluded by the `*.sqlite`
  rule in `.gitignore`, and `data/` (the CSV tree) does not exist in a fresh
  clone until `scripts/download_nba_data.py` builds it.

Because of this, `scripts/_data_guard.py` hard-stops any maintained script
that opens a missing derived artefact: it prints one line naming the artefact
and the command that builds it, and exits non-zero. It never creates an empty
database or a dummy value.

## 2. What the published results were produced from

| Artefact | What it held | Where it came from | Rebuilt by a script in this repo? |
|---|---|---|---|
| `Data/dataset.sqlite` | The design matrix: table `dataset_2012-26`, **17,356 rows × 117 columns** (per `Data/manifest.json`), game dates 2012-10-30 → 2026-01-07, label `Home-Team-Win` at base rate 0.5718 (manifest `labels`). The manifest's `build.stats` record 31,515 odds rows seen and 17,356 kept. | Assembled by `Create_Games` (module `wippa_nba.legacy.process_data.Create_Games`) from the two sqlite databases below; the manifest's `build.odds_tables_used` records which odds table fed each season, back to 2012-13. | Yes — `python -m wippa_nba.legacy.process_data.Create_Games` (see §3). |
| `Data/TeamData.sqlite` | Per-day team snapshots: one table per calendar date of scraped team stats (`Get_Data.py` writes a date-named table per fetch). Seasons are configured in `config.toml` under `[get-data]` (2007-08 onward). | Scraped daily from NBA stats endpoints by `Get_Data.py`. | Yes — `python -m wippa_nba.legacy.process_data.Get_Data` (see §3), forward-only. |
| `Data/OddsData.sqlite` | Closing moneyline prices, one table per season key, written by `Get_Odds_Data.py` (which reads live SBR scoreboards through `sbrscrape`). | Scraped from SBR as the seasons ran; the per-season tables are the `odds_*` tables the manifest names. | Yes — `python -m wippa_nba.legacy.process_data.Get_Odds_Data` (see §3), but only forward from the day you run it (see §4). |
| `Data/injury.sqlite` | The append-only injury-report store. Its scale is recorded in `results/issue72_late_scratch.json`: **240,241 designation events across 2,046 snapshots** (also tabled in `docs/late_scratch.md`). | Official pre-tip injury-report PDFs from `ak-static.cms.nba.com`, captured and parsed by `Get_Injury_Data.py` and backfilled per game date by `scripts/backfill_injury_reports.py`. | Yes, forward-only (see §3 and §4). |
| `Data/teams.sqlite` | — | **Not named in `Data/manifest.json`.** The manifest's only table entry is `dataset_2012-26`. The filename `teams.sqlite` appears in this repository only inside `tests/`, as a fixture path. | No — there is no such production artefact in this pipeline. |
| `data/nba` (CSV tree) | The NBA-api data layer behind Pipeline A: **310 CSVs covering 13,199 games** (figures from `README.md`, "Data & engineering" section; the 13,199-game frame is also the `n_games` recorded in `results/availability.json`). | Downloaded from NBA's own endpoints by `scripts/download_nba_data.py` (argparse defaults: `--out data/nba`, `--start 2015`, `--end 2025`; the README describes a core run as 10+ seasons / 13,199 games). | Yes — `python scripts/download_nba_data.py`. |

The train/test split (15,620 train / 1,736 test of the 17,356 rows) and the
headline `scripts/evaluate.py --skip-odds` measurements (0.6584 accuracy,
0.6189 log loss) are not stored in any data file; the committed run
`20261023T065448Z` in `runs/ledger.jsonl` (appended by `scripts/evaluate.py`)
records them, and its `dataset_fingerprint` matches the manifest's
`content_sha256` for `dataset_2012-26`.

## 3. How to rebuild it

The rebuild commands below are the same table `scripts/_data_guard.py` prints
in its guard message (its `BUILD_COMMANDS` mapping), plus the argparse
switches read from each script. **Status: every command was verified by
reading the code (docstring, argparse, and constants); none was executed in
the environment that wrote this page.**

| Artefact | Command | Notes from the code |
|---|---|---|
| `data/nba` | `python scripts/download_nba_data.py` | Resumable; default `--out data/nba`, stages `core` (`--stages core,extras` adds style and shots). Defaults `--start 2015 --end 2025`. Documented in `scripts/README_DATA.md`. |
| `Data/TeamData.sqlite` | `python -m wippa_nba.legacy.process_data.Get_Data` | This exact form is the guard's table entry (incremental update). `--backfill [--season 2025-26]` fetches all missing dates for the seasons configured in `config.toml` `[get-data]` (argparse in `Get_Data.py`). |
| `Data/OddsData.sqlite` | `python -m wippa_nba.legacy.process_data.Get_Odds_Data` | Guard's table entry (incremental). `--backfill [--season KEY] [--sportsbook fanduel]` fetches missing odds dates for the configured seasons (argparse in `Get_Odds_Data.py`). |
| `Data/dataset.sqlite` | `python -m wippa_nba.legacy.process_data.Create_Games` | Reads `Data/OddsData.sqlite` and `Data/TeamData.sqlite` (constants `ODDS_DB_PATH`, `TEAMS_DB_PATH`), writes table `dataset_2012-26` to `Data/dataset.sqlite`. The module-rebuild hint in `src/wippa_nba/legacy/train_models/training_common.py` points to this same command. |
| `Data/injury.sqlite` | `python -m wippa_nba.legacy.process_data.Get_Injury_Data capture --from 2026-10-01 --to 2026-10-23` then `... Get_Injury_Data parse` | Invocation block in the `Get_Injury_Data.py` docstring; capture has a hard `--max-requests` budget (default 20) and 5-10 s pacing. |
| `Data/injury.sqlite` (per game date) | `python scripts/backfill_injury_reports.py --from 2022-01-16 --to 2022-06-30 --slot 17:00 --slot 20:00 --max-requests 400` | Docstring example; `--from`/`--to` are required argparse flags. Resumable, and exits non-zero on budget exhaustion rather than silently skipping a slot. |

## 4. What cannot be rebuilt from this repository

Say it plainly: **a fresh clone cannot reproduce the published numbers.**

- **The SBR odds history cannot be re-fetched.** `Get_Odds_Data.py` reads live
  SBR scoreboards for the dates you run it over. The per-season tables behind
  2012-13 through the present exist only because they were collected as the
  seasons ran. There is no archive of them in the repository and no tool here
  that can retrieve past SBR prices. The OddsPortal best-of-books backfill
  (`scripts/backfill_oddsportal.py`, documented in `docs/edge_search.md`)
  scrapes a third-party historical aggregator: whatever it returns tomorrow is
  whatever OddsPortal still serves, so those results are a historical
  snapshot, not a reproducible build.
- **The injury-report history cannot be re-fetched.** The capture module talks
  only to `ak-static.cms.nba.com`, which serves the current season's reports.
  The `Get_Injury_Data.py` docstring is explicit that historical recovery
  (enumerating archived PDFs through the Wayback CDX API) is *not implemented*
  — "Nothing here calls CDX". The 240,241-event store recorded in
  `results/issue72_late_scratch.json` therefore cannot be re-derived from a
  clone.
- **Consequence for the published numbers.** The design matrix itself is built
  *from* the odds tables: `Data/manifest.json` records `odds_tables_used` back
  to 2012-13, and `Create_Games` consumes `Data/OddsData.sqlite` and
  `Data/TeamData.sqlite`. Every measurement in this project runs on that
  design matrix or on the odds tables. So no published number — the 0.6584
  accuracy / 0.6189 log loss recorded in `runs/ledger.jsonl`, the
  market comparisons in `results/closing_line.json`, the injury-store counts —
  can be re-derived from a clone alone. The one artefact a fresh clone can
  rebuild end to end is the `data/nba` CSV tree, subject to the endpoint
  health measured in `scripts/README_DATA.md` (at measurement time
  `playbyplayv3` and `winprobabilitypbp` returned HTTP 500 on every probe).

## 5. Provenance: `Data/manifest.json`

`Data/manifest.json` is the only data file the repository commits, and it is
the fingerprint of the build that produced the published results. It records:

- when it was generated and from which commit (`generated_utc`, `git_sha`,
  `git_dirty`);
- per table: row and column counts (`dataset_2012-26`: 17,356 × 117), a
  `content_sha256` and `column_list_sha256` for the table contents and column
  order, and the game date range (2012-10-30 → 2026-01-07);
- the build recipe: which odds table backed each season
  (`odds_tables_used`, back to 2012-13), snapshot counts
  (`expected_team_rows`, `usable_snapshots`, `truncated_snapshots`,
  `snapshot_lag_days_gt_7`, `max_snapshot_lag_days`) and the keep/drop stats
  (`odds_rows`, `kept`, `no_prior_snapshot`, `legacy_same_day_drop`, …);
- the label base rate (`Home-Team-Win`: 9,924 positives, 0.5718).

A rebuild is honest only if a fresh manifest over the rebuilt table matches
these hashes; the `content_sha256` is the check.

## 6. The guard and the data README

- `scripts/_data_guard.py` — the shared guard imported by the maintained
  tools. Its message names the missing artefact, the exact rebuild command
  (the `BUILD_COMMANDS` table quoted in §3), and points the reader at
  `scripts/README_DATA.md`.
- `scripts/README_DATA.md` — the downloader's own documentation: usage, the
  measured endpoint-health table, and the schema traps of the `data/nba`
  layer.

Scope note, so the two pages are not mistaken for duplicates:
`scripts/README_DATA.md` documents only the `scripts/download_nba_data.py`
CSV layer. The legacy sqlite pipeline (`Get_Data`, `Get_Odds_Data`,
`Create_Games`), the injury store, and `Data/manifest.json` are documented on
this page. No statement in `scripts/README_DATA.md` contradicts this page;
the difference is scope, and the guard message and this page agree on the
rebuild commands.
