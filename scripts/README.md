# `scripts/`

Measurement and inspection tools. Both scripts are read-only: they open the
SQLite databases under `Data/` with `mode=ro` and never write to them.

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
| `UNVERIFIED` | a check could not run: odds were requested but the database is unusable |
| `UNKNOWN` | `TeamData.sqlite` is missing, so nothing could be checked at all |

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
gate refused to clear the dataset. Nothing here raises.

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
`src/Train-Models/training_reporting.py` (loaded by file path, because that
directory contains a hyphen and is not a package); identical local fallbacks are
used if that import ever fails, and the substitution is printed.

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
