# `scripts/`

Maintained measurement and inspection tools: **21 runnable scripts** plus the
shared `_data_guard.py` module. The one-off experimental runs that produced
this project's historical numbers — the `issue*`, `h*`, `round3_*`,
`explore_*`, `elo_*`, `backtest_*` and audit scripts — live in
[`research/archive/`](../research/README.md) with a per-file classification.
Nothing was deleted; the archived scripts are the provenance of the recorded
results and are kept compiling, but they are not maintained tools.

## What happens when data is absent

**The scripts stop with one clear line and a non-zero exit. This is
deliberate.** The repository ships no data: `Data/` holds only
`manifest.json`, and `data/` does not exist until the downloader builds it.
Every sqlite database and the `data/nba` CSV tree are build products. Before
the guard, a fresh-clone run died with a raw traceback deep inside sqlite3 or
pandas, or quietly produced a zero-row result that looked like a measurement.

The guard is `scripts/_data_guard.py`. A maintained script that needs an
artefact checks it at the point of first use and, when it is absent, prints
exactly this shape to stderr and exits 1:

```
Data/dataset.sqlite is not in this repository.
It is a derived artefact - see scripts/README_DATA.md and the build commands
in tests/_data_guard.py for how to rebuild it.
```

There are no fallbacks, no empty databases, no dummy values: a missing
artefact is a configuration problem, not a measurement result. The same
artefact-to-build-command table lives in `tests/_data_guard.py`, which skips
the tests that need the data; the messages in both places are deliberately
mirrored. A present-but-empty (0-byte) sqlite file is treated as absent, and
`data/nba` is checked for the two sub-directories every loader needs
(`game_logs`, `team_gamelogs`) — an interrupted download leaves exactly the
half-built state that otherwise crashes a loader.

Two producers are deliberately **not** guarded, because a missing output is
their normal cold start: `download_nba_data.py` (creates `data/nba`) and
`backfill_oddsportal.py` (network to scrape CSVs). `feature_block_ablation.py`
keeps its own contract test that drives `main()` with only the model fitting
stubbed, so instead of a pre-check it translates `nba_data`'s own absent-data
signal into the same one-line stop.

## The maintained tools

| script | what it does | data it needs | `results/` file it produces |
|---|---|---|---|
| `evaluate.py` | the reproducible measurement suite (below) | `Data/dataset.sqlite`, `Data/TeamData.sqlite`, `Data/OddsData.sqlite` unless `--skip-odds` | prints; provenance in `runs/ledger.jsonl` |
| `check_feature_alignment.py` | training-vs-inference feature contract check (below) | `Data/dataset.sqlite` | prints |
| `closing_line.py` | does anything beat the closing price? market baseline, model ROI at the close | `Data/OddsData.sqlite`, `data/nba` | `closing_line.json` |
| `edge_betting.py` | the parameter-free breakeven rule: bet only where the model beats the posted price | `Data/OddsData.sqlite`, `data/nba` | `edge_betting.json` |
| `edge_search.py` | pre-registered family scan over the odds archive (`--family`, `--preregister`) | `Data/OddsData.sqlite`, `data/nba` | `edge_search/<family>.json`; ledger appends to `experiments.jsonl` |
| `edge_holdout.py` | is a threshold cell a real edge, or picked by looking? leave-one-season-out | `Data/OddsData.sqlite`, `data/nba` | prints |
| `feature_block_ablation.py` | score any named feature block by paired bootstrap (`--blocks`, `--seed`, `--splits`) | `data/nba` | `feature_block_ablation.json` |
| `availability.py` | is the availability effect attainable honestly | `data/nba` | `availability.json` |
| `recency_ablation.py` | recency-halflife arm comparison | `data/nba` | `recency_ablation.json` |
| `recency_followup.py` | is the recency gain real, or a warm-up artefact | `data/nba` | `recency_followup.json` |
| `run_experiments.py` | the pre-registered experiment grid, stage by stage | `data/nba` | appends to `experiments.jsonl` |
| `experiments.py` | the experiment-harness library `run_experiments.py` imports; ledger writer | `data/nba` | appends to `experiments.jsonl` |
| `c44_spread_total.py` | spread/total family runner (`--prereg` first); exports `effective_line` used by `edge_search.py` | `Data/OddsData.sqlite`, `data/nba` | `c44_spread_total.json`, `c44_report.md` |
| `issue72_late_scratch.py` | late-scratch census from the injury store (contract-tested) | `Data/injury.sqlite`, `data/nba` | `issue72_late_scratch.json`, `.md` |
| `issue74_form_mismatch.py` | form-mismatch spots, priced (contract-tested) | `Data/OddsData.sqlite`, `data/nba` | `issue74_form_mismatch.json`, `.md` |
| `issue81_layered_elo_form.py` | layered Elo + recent-form filter, priced at the close (contract-tested) | `Data/OddsData.sqlite`, `data/nba` | `issue81_layered_elo_form.json`, `.md` |
| `download_nba_data.py` | the data downloader: stats.nba.com -> `data/nba` (no guard: it is the producer) | network | builds `data/nba` |
| `backfill_espn_close.py` | backfill full-season closing lines from ESPN's odds API | network, `data/nba` | writes a table into `Data/OddsData.sqlite` |
| `backfill_oddsportal.py` | scrape best-of-books moneylines from OddsPortal (no guard: network to CSVs) | network | scrape CSVs under `--out-dir` |
| `validate_oddsportal.py` | validate that scrape against the project's games; `--promote` writes `odds_ml_bestbooks` | scrape CSVs, `data/nba` | validation JSON (`--out`); optional `Data/OddsData.sqlite` write |
| `backfill_injury_reports.py` | backfill injury-report PDFs for the dates the model needs | network, `data/nba` | `Data/injury.sqlite` |

## `evaluate.py` — the reproducible measurement suite

One command that produces every number this project claims about itself, with
the method printed next to each figure.

```bash
python scripts/evaluate.py                      # full suite (logistic regression)
python scripts/evaluate.py --model gbdt         # gradient boosting instead
python scripts/evaluate.py --model none         # baselines only, no training
python scripts/evaluate.py --skip-odds          # no odds-dependent metric
```

Flags: `--dataset` (default `dataset_2012-26`), `--test-size` (default `0.1`),
`--seed` (default `42`), `--model {logreg,gbdt,none}`, `--skip-odds`.

It needs `Data/dataset.sqlite` and `Data/TeamData.sqlite` (and
`Data/OddsData.sqlite` unless `--skip-odds`); the guard stops the run before
the gate when one of them is absent.

It prints, in order: the **leakage gate**, **baselines**, the **trained
model** (accuracy, log loss, Brier, AUC), the **market baseline** (accuracy and
flat-bet ROI of always backing the closing-line favourite), the **model ROI at
the closing line** across edge thresholds 0.00 / 0.02 / 0.05 / 0.10, and a
`SUMMARY` block of `key=value` lines for CI to grep.

### Why the gate comes first

The project's original 68.9% accuracy claim was target leakage: the team-stats
snapshot named `D` contained the result of the game played on `D`, so `W`,
`L`, `GP`, `W_PCT`, `PTS` and every `*_RANK` column carried the answer. A
leak-free build joins against a snapshot that predates the game, so its `Date`
column is the day **before** the game the row describes.

The gate samples 400 rows and runs three checks before any accuracy is printed:

1. **Snapshot provenance** — the row's `GP`/`W` must equal the `GP`/`W` in the
   `Data/TeamData.sqlite` table its `Date` names. A row that disagrees cannot be
   traced at all.
2. **Pre-game join** — the game's own date (from the odds ledger, the only
   in-repo record of when a game was played) must fall **after** the row's
   `Date`. A dataset that joins same-day prices was built from the same-day
   snapshot, and that snapshot carries the result whenever the stats were pulled
   after tip-off. This is what fires on `dataset_2012-23`, `dataset_2012-24` and
   `dataset_2012-24_new`.
3. **W-increment leak rule** — the audit's own detector,
   `pred = 1 if W_home[D] - W_home[D-1] > W_away[D] - W_away[D-1] else 0`,
   scored against the label, reading `W` on `D` from the row itself and `W` on
   `D-1` from the previous TeamData snapshot. On clean data it collapses to
   roughly the away-win rate; if it beats the always-home base rate by more than
   5 points, the snapshot carries results. It needs no odds, so it still runs
   under `--skip-odds`.

If a check finds evidence of leakage the harness prints a loud banner and
**suppresses** every accuracy, log loss, Brier, AUC and ROI figure; only the
gate's own evidence, the row counts and the `SUMMARY` line are printed. Exit
code is still 0 — CI should grep the summary rather than rely on the exit
status.

The verdict is one of five values, and only the first two print metrics:

| verdict | meaning |
| --- | --- |
| `PASS` | every check ran and none found leakage |
| `PASS_NO_ODDS` | checks 1 and 3 passed; check 2 was **not** verified (`--skip-odds`) |
| `FAIL` | a check found evidence of leakage |
| `UNVERIFIED` | odds were requested but the database is unusable. Since the data guard was added, an **absent** `OddsData.sqlite` stops the script before the gate, so this verdict now means present-but-unusable |
| `UNKNOWN` | historically: `TeamData.sqlite` missing, nothing checked. The data guard now stops the script before the gate when `TeamData.sqlite` is absent, so this verdict is no longer reachable |

`UNVERIFIED` and `UNKNOWN` print a different banner from `FAIL` on purpose: a
check that could not run says nothing about the dataset, and a banner claiming
leakage was found would be its own kind of false claim.

### The odds join, and why it is shifted by a day

`Data/OddsData.sqlite` holds moneyline **prices** only (`ML_Home`, `ML_Away`).
`OU` and `Spread` are lines, not prices, so no totals bet can be priced from
this database. Overlapping season tables mean a naive concatenation double
counts every matchup, so games are deduplicated on `(Date, Home, Away)` and both
counts are printed.

Because `Date` is the *snapshot* date, the game's closing price is dated one day
later. The harness resolves that alignment at run time (the `Odds join
alignment:` line) and reports it: `+1d -> 1706 of 1722` means 99.1% of the test
window prices up. A leak-free dataset resolves to `+1`; a same-day-join dataset
resolves to `+0`.

### Degradation

`ROI not available: <reason>` is printed — and execution continues with exit
code 0 — when `--skip-odds` is passed, when the join matches fewer than
`MIN_MATCHED_ODDS_ROWS` (50) rows, when a matched row carries an unusable
price, when `--model none` means there are no model picks to price, or when the
gate refused to clear the dataset. Nothing here raises. An absent odds
database is no longer one of these cases: the guard stops the run instead (see
above).

A join that covers less than `MIN_JOIN_COVERAGE` (95%) is treated as a sparse
join and no figure is printed at all. At or above that floor a figure is printed
together with an explicit note naming the unmatched rows — on this data the
2024-25 odds table stops one day before that season ends, so the last sixteen
test games genuinely have no price to bet at. Set `MIN_JOIN_COVERAGE = 1.0` for
the strict reading in which any unmatched row refuses the figure.

### Betting arithmetic

A flat `$100` stake at the closing price. A winning bet profits `odds` when the
American price is positive and `10000 / abs(odds)` when it is negative; a losing
bet loses the stake. Model bets are selected where
`P(model) - P(de-vigged market) > threshold`, the market probability being the
two implied probabilities normalised against each other. `american_odds_profit`
and `american_odds_implied_probability` are imported from
`wippa_nba.legacy.train_models.training_reporting` — the same module the
trainers use, imported normally now that it lives in the package. The
file-path loader this script used while the module sat in `src/Train-Models/`,
its printed substitution note and its local fallback arithmetic are gone: the
fallbacks guarded against an import failure that would take down the script's
other package imports first.

Every ROI is printed with its standard error. A flat-bet ROI over a few hundred
bets has a standard error measured in percentage points, so a figure quoted
without one is indistinguishable from a claim.

### Reading the output

`SUMMARY` is one `key=value` per metric. `n/a` always means the figure was
deliberately not computed and the reason is printed above it. The
`gate_rows_checked`, `odds_join_matched` and `odds_date_offset_days` lines carry
the provenance for the gate and the ROI respectively.

## `check_feature_alignment.py`

Compares the feature width the trained models were fitted on with the width the
inference path feeds them (and the ORDER when `--use-live` builds a frame from
today's games). It is the check for the silent three-column permutation that fed
the totals model `OU` at position 106 having been trained with it at 104.

```bash
python scripts/check_feature_alignment.py --dataset dataset_2012-26
python scripts/check_feature_alignment.py --use-live
```

It opens `Data/dataset.sqlite` read-only; the guard stops the run with one
line when that file is absent.
