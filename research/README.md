# `research/`

Historical one-off experiment scripts, moved out of `scripts/` so the directory
there holds only maintained tools. Nothing here was deleted: every script
compiles, and each one is the provenance of numbers recorded under `results/`
or in `docs/`. They are kept as the historical record, not as tools to run.

The classification of every script that was in `scripts/` is in the table
below. The rules applied:

- **maintained** — parameterised, documented, and/or cited by `README.md`,
  `docs/`, `usage.md`, `reproducing.md`, or pinned by `tests/`. Stays in
  `scripts/`.
- **archive** — a one-off experimental run whose value is the historical
  record. Moved here with `git mv`, with its repo-root path assumptions fixed
  for the extra directory depth (`Path(__file__).resolve().parents[2]`).
- **superseded** — a duplicate or dead script with no result behind it. None
  were found: the one candidate, `experiments.py`, is imported by
  `run_experiments.py` and stays in `scripts/`.

## How to run an archived script

From the repository root:

```bash
python research/archive/issue39_rapm.py
```

Three of the archived round-3 scripts (`round3_avail.py`,
`round3_avail_followup.py`, `round3_calib.py`) import `round2_common`, a
module that no longer exists anywhere in the tree; they compile but cannot
run, and no round-2 module was invented to revive them. Their recorded
outputs are `results/round3_*.jsonl` and `results/round3_summary.md`.

Data needed by any script here is the same as for the maintained tools:
`Data/*.sqlite` and `data/nba` (see `scripts/README_DATA.md`). The explicit
data guard added to the maintained tools was deliberately **not** applied
here — archived scripts keep their original behaviour.

## Classification

### Maintained (stays in `scripts/`)

| script | one-line role | reason it is maintained |
|---|---|---|
| `evaluate.py` | the reproducible measurement suite (leakage gate, baselines, model, market, ROI, SUMMARY) | cited by root `README.md`, `reproducing.md`, `docs/feature_order_contract.md`, `CONTRIBUTING.md` |
| `check_feature_alignment.py` | training-vs-inference feature contract check | cited by `docs/feature_order_contract.md`, `docs/engineering/module_map.md` |
| `closing_line.py` | does anything beat the closing price (market comparison, ROI) | cited by root `README.md`, `usage.md`, `docs/reading_the_results.md`, `docs/measurement_bugs.md`; imported by tests |
| `edge_betting.py` | the parameter-free breakeven betting rule | cited by root `README.md`, `usage.md`, `docs/reading_the_results.md` |
| `edge_search.py` | pre-registered family scan over the odds archive | `docs/edge_search.md` documents it; `tests/integration/test_edge_search.py` pins its settlement logic |
| `edge_holdout.py` | is a threshold cell a real edge (leave-one-season-out) | cited by `reproducing.md`, `docs/measurement_bugs.md` |
| `feature_block_ablation.py` | score any named feature block by paired bootstrap | cited by `usage.md`, `reproducing.md`, `docs/feature_search.md`, `docs/player_elo_leak.md`, `docs/availability_block.md`; `tests/unit/test_feature_block_ablation.py` |
| `availability.py` | is the availability effect attainable honestly | cited by `usage.md`, `docs/availability_block.md` |
| `recency_ablation.py` | recency-halflife arm comparison | cited by `docs/modeling_exploration.md`; imported by `recency_followup.py` |
| `recency_followup.py` | is the recency gain real or warm-up | cited by `usage.md` |
| `run_experiments.py` | pre-registered experiment grid, appends to the ledger | cited by `reproducing.md`, `docs/reading_the_results.md` |
| `experiments.py` | experiment-harness library (`Experiment`, `append`, `build_frame`) + ledger writer | imported by `run_experiments.py`; writes `results/experiments.jsonl` |
| `c44_spread_total.py` | spread/total family runner; exports `effective_line` | `effective_line` is imported by `edge_search.py`; its own results are `results/c44_spread_total.json` / `c44_report.md` |
| `issue72_late_scratch.py` | late-scratch census from the injury store | `docs/late_scratch.md`; `tests/unit/test_issue72_late_scratch.py` pins its selection logic |
| `issue74_form_mismatch.py` | form-mismatch spots, priced | `docs/form_mismatch.md`; `tests/unit/test_issue74_form_mismatch.py` |
| `issue81_layered_elo_form.py` | layered Elo + recent-form filter, priced at the close | `docs/layered_elo_form.md`; `tests/unit/test_issue81_layered_elo_form.py` |
| `download_nba_data.py` | the data downloader (network -> `data/nba`) | cited by root `README.md`, `usage.md`; documented in `scripts/README_DATA.md` |
| `backfill_espn_close.py` | backfill closing lines from ESPN's odds API into `OddsData.sqlite` | cited by `docs/edge_search.md` |
| `backfill_oddsportal.py` | scrape best-of-books moneylines from OddsPortal | cited by `docs/edge_search.md` |
| `validate_oddsportal.py` | validate that scrape against the project's games; `--promote` writes `odds_ml_bestbooks` | companion instrument to `backfill_oddsportal.py` |
| `backfill_injury_reports.py` | backfill injury-report PDFs for the dates the model needs | companion instrument to the injury pipeline (`docs/injury_pipeline.md`) |
| `_data_guard.py` | shared explicit data guard (not a runnable script) | imported by the maintained tools that open data |

### Archived (moved to `research/archive/`)

| script | one-line role | reason |
|---|---|---|
| `audit_starter_odds_sources.py` | issue #38 audit of what a T-45min prediction point needs | one-off issue audit; recorded as `results/issue38_source_audit.{json,md}` |
| `backtest_user_rules.py` | family 5: 13 pre-registered user rules (issue #91) | one-shot pre-registered family; superseded as a workflow by `edge_search.py` |
| `backtest_lay_rules.py` | family 7: lay bets at the moneyline odds (issue #95) | one-shot pre-registered family; superseded as a workflow by `edge_search.py` |
| `elo_buckets.py` | do player-Elo contrast deciles behave like a signal | one-off check; provenance for `docs/player_elo_leak.md` |
| `elo_roster_variants.py` | roster-assumption variants of the player-Elo block | one-off comparison; provenance for `docs/player_elo_leak.md` |
| `explore_elo_parity_underdog.py` | exploratory: underdog ROI at near-Elo parity | labelled exploratory in its own docstring; output in `results/exploratory/` |
| `explore_streak4.py` | exploratory: profile of a team on a 4th straight win | labelled exploratory in its own docstring; output in `results/exploratory/` |
| `h1h2_market_anchor.py` | batch A gate-keepers: market-anchored blend, availability vs price | theory-dossier family; recorded in `docs/experiment_log_2026-10.md`, `docs/market_anchor.md` |
| `h3_recalibration.py` | H3: season-blocked causal recalibration | theory-dossier family; `results/h3_recalibration.json` |
| `h4_rest_differential.py` | H4: rest as a differential, not two day counts | theory-dossier family; `results/h4_rest_differential.json` |
| `h11_absence_context.py` | H11: role-substitution / usage redistribution | theory-dossier family; `results/h11_absence_context.json` |
| `issue37_injury_reports.py` | #37: do timestamped injury designations add signal | issue one-off; `results/issue37_injury.{json,md}` |
| `issue37_full_frame.py` | #37 criterion 4: injury block on the 43-column bar | issue one-off; `results/issue37_injury_full_frame*.json/md` |
| `issue39_rapm.py` | #39: true RAPM on stint data, walk-forward | issue one-off; `results/issue39_rapm.{json,md}` |
| `issue40_dynamic_ratings.py` | #40: glicko / dynamic team-strength ratings | issue one-off; `results/issue40_dynamic_ratings.{json,md}` |
| `issue41_hierarchical.py` | #41: hierarchical player-to-team partial pooling | issue one-off; `results/issue41_hierarchical.{json,md}` |
| `issue42_ensemble.py` | #42: do the strongest models disagree; does a blend help | issue one-off; `results/issue42_ensemble.{json,md}` |
| `issue56_tz_travel.py` | #56: time-zone / travel-quantity features | issue one-off; `results/issue56_tz_travel.{json,md}` |
| `issue57_cover_lines.py` | #57: spread-cover objective at alternate lines | issue one-off; `results/issue57_cover_lines.{json,md}` |
| `issue58_variance_staking.py` | #58: variance-conditioned staking | issue one-off; `results/issue58_variance_staking.{json,md}` |
| `issue59_early_season.py` | #59: is the market less efficient early in the season | issue one-off; `results/issue59_early_season.{json,md}` |
| `issue60_momentum.py` | #60: momentum streak brackets, 2025-26 holdout | issue one-off; `results/issue60_momentum.{json,md}` |
| `issue60b_recent_form.py` | #60 follow-on: recent-form win-count windows | issue one-off; `results/issue60b_recent_form.{json,md}` |
| `issue68_injury_promotion.py` | #68: #37 promotion path on a powered season holdout | issue one-off; `results/issue68_injury_promotion.{json,md}` |
| `issue69_status_trajectory.py` | #69: intraday change in injury designation | issue one-off; `results/issue69_status_trajectory.{json,md}` |
| `issue70_null_replication.py` | #70: meta-replication of recent nulls on 2024-25 | issue one-off; `results/issue70_null_replication.{json,md}` |
| `issue71_availability_sensitivity.py` | #71: is the availability block on a plateau or a grid interior | issue one-off; `results/issue71_availability_sensitivity.{json,md}` |
| `issue73_pre2020_parser_audit.py` | #73: audit pre-2020 injury layouts against the live CDN | issue one-off audit; `results/issue73_pre2020_parser_audit.{json,md}` |
| `round3_common.py` | round-3 harness (bar change stated up front) | historical round experiment; imported by the `round3_run_*` scripts here |
| `round3_avail.py` | round 3 block A: expanded availability block | historical round experiment; `results/round3_avail.jsonl`; imports the removed `round2_common` |
| `round3_avail_followup.py` | does concentration add over availability | historical follow-up; `results/round3_avail_followup.jsonl`; imports the removed `round2_common` |
| `round3_calib.py` | round 3 block CAL: nested probability calibration | historical round experiment; `results/round3_calib.jsonl`; imports the removed `round2_common` |
| `round3_run_S.py` | round 3 block S: squad-quality arms | historical round experiment |
| `round3_run_T.py` | round 3 block T: team Elo arms | historical round experiment |
| `round3_run_U.py` | round 3 block U: combine the survivors | historical round experiment |
| `round3_run_Y.py` | round 3 block Y: squad continuity + load decay | historical round experiment |
| `round3_run_Z.py` | round 3 block Z: L1 / elastic net on the promoted frame | historical round experiment |
| `round3_final.py` | round-3 consolidation: order-robustness done correctly | historical consolidation |
| `round3_corr_dispute.py` | settle the corr(elo_diff, ewm_margin) dispute | one-off dispute resolution; `results/elo_corr_dispute.json` |
| `round3_verify.py` | leakage proof + redundancy gates before scoring | historical gates; referenced by `docs/experiment_log_2026-10.md`, `docs/leakage-gate` prose and `tests/unit/test_leakage_gate.py` comments |

### Superseded

None. The initial candidate, `experiments.py`, turned out to be the harness
library `run_experiments.py` imports, so it is maintained. No script in this
archive is dead code with no result behind it: each has recorded outputs
under `results/` or is cited as provenance in `docs/`.
