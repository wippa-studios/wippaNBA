# `results/` — index of committed result artefacts

Every file here is the recorded output of a run of a script in this
repository. Paths below are given relative to the repository root. Producers
marked *(archive)* live in `research/archive/` and are kept as the historical
record, not as tools to run (see `research/README.md`). Producers marked
*(no script in tree)* have no producing script committed; where that is the
case it is said so, not papered over.

## Top level

| File | What it measures | Produced by | Cited by |
|---|---|---|---|
| `results/availability.json` | Availability-block arms on the 13,199-game frame (per-arm log-loss deltas with CIs; baseline 0.6195) | `scripts/availability.py` | `results_pipeline_a.md`, `docs/availability_block.md` |
| `results/c44_spread_total.json` + `results/c44_report.md` | Pre-registered spread/total family C44: margin- and total-RMSE verdicts, 6 looks, Bonferroni | `scripts/c44_spread_total.py` | report: `docs/experiment_log_2026-10.md`; the JSON itself: not cited |
| `results/closing_line.json` | Market vs model on 5,788 games with closing lines (overround, per-arm log loss) | `scripts/closing_line.py` | `results_pipeline_b.md`, `docs/availability_block.md`, `docs/market_anchor.md`, `docs/reading_the_results.md` |
| `results/edge_betting.json` | The parameter-free breakeven betting rule, settled over 5,788 games | `scripts/edge_betting.py` | Not cited by path; the −8.23% headline and the script are cited in `README.md` (Limitations item 1) |
| `results/elo_corr_dispute.json` | Settles the corr(elo_diff, ewm_margin) level-vs-contrast dispute | *(archive)* `research/archive/round3_corr_dispute.py` | Not cited |
| `results/experiments.jsonl` | The append-only experiment ledger: pre-registrations, runs, verdicts — the spine every other artefact hangs from | `scripts/experiments.py` (library), written by `scripts/run_experiments.py` and the issue scripts | `README.md`, `results_pipeline_a.md`, `reproducing.md`, `audit_and_fixes.md`, `docs/availability_block.md`, `docs/experiment_log_2026-10.md`, `docs/form_mismatch.md`, `docs/injury_pipeline.md`, `docs/late_scratch.md`, `docs/market_anchor.md`, `docs/modeling_exploration.md`, `docs/research_links.md` |
| `results/feature_block_ablation.json` | Paired-bootstrap scoring of named feature blocks (67 features, 6,600 out-of-fold rows) | `scripts/feature_block_ablation.py` | `docs/experiment_log_2026-10.md`, `docs/modeling_exploration.md` |
| `results/h11_absence_context.json` + `results/h11_report.md` | H11 role-substitution / usage redistribution — a null | JSON: *(archive)* `research/archive/h11_absence_context.py`; the report has no script in tree | Both: `docs/availability_block.md`, `docs/experiment_log_2026-10.md` |
| `results/h3_recalibration.json` | H3 season-blocked causal recalibration | *(archive)* `research/archive/h3_recalibration.py` | `docs/experiment_log_2026-10.md` |
| `results/h3h4_report.md` | Run report for the H3 + H4 theory-dossier pair | *(no script in tree)* — pre-registrations live in `results/experiments.jsonl` | `docs/experiment_log_2026-10.md` |
| `results/h4_rest_differential.json` | H4 rest as a differential, not two day counts | *(archive)* `research/archive/h4_rest_differential.py` | `docs/experiment_log_2026-10.md` |
| `results/issue37_injury.json` + `.md`, `results/issue37_injury_2020-21.json` + `.md` | Timestamped injury designations as signal, 2021-22 and 2020-21 seasons | *(archive)* `research/archive/issue37_injury_reports.py` (the 2020-21 variant is not separately mapped in `research/README.md`) | Not cited |
| `results/issue37_injury_full_frame.json` + `.md`, `results/issue37_injury_full_frame_refresh.json` + `.md` | The injury block on the 43-column bar, full 13,199-game frame (and its refreshed rerun) | *(archive)* `research/archive/issue37_full_frame.py` | `results/issue37_injury_full_frame.json`: `docs/injury_pipeline.md` |
| `results/issue38_source_audit.json` + `.md` | Audit of what a second prediction point needs, per odds source | *(archive)* `research/archive/audit_starter_odds_sources.py` | Not cited |
| `results/issue39_rapm.json` + `.md` | True RAPM on stint data, walk-forward | *(archive)* `research/archive/issue39_rapm.py` | Not cited |
| `results/issue40_dynamic_ratings.json` + `.md` | Glicko / dynamic team-strength ratings, walk-forward | *(archive)* `research/archive/issue40_dynamic_ratings.py` | Not cited |
| `results/issue41_hierarchical.json` + `.md` | Hierarchical player-to-team partial pooling | *(archive)* `research/archive/issue41_hierarchical.py` | Not cited |
| `results/issue42_ensemble.json` + `.md` | Do the strongest models disagree; do fixed-weight blends help | *(archive)* `research/archive/issue42_ensemble.py` | Not cited |
| `results/issue56_tz_travel.json` + `.md` | Time-zone / travel-quantity features on the canonical frame | *(archive)* `research/archive/issue56_tz_travel.py` | `docs/research_links.md` |
| `results/issue57_cover_lines.json` + `.md` | Spread-cover betting objective at alternate lines | *(archive)* `research/archive/issue57_cover_lines.py` | `docs/research_links.md` |
| `results/issue58_variance_staking.json` + `.md` | Variance-conditioned staking vs flat breakeven betting | *(archive)* `research/archive/issue58_variance_staking.py` | `docs/research_links.md` |
| `results/issue59_early_season.json` + `.md` | Is the market less efficient early in the season | *(archive)* `research/archive/issue59_early_season.py` | `docs/research_links.md` |
| `results/issue60_momentum.json` + `.md` | Momentum streak brackets, 2025-26 holdout | *(archive)* `research/archive/issue60_momentum.py` | `docs/research_links.md` |
| `results/issue60b_recent_form.json` + `.md` | Recent-form win-count windows, 2025-26 holdout | *(archive)* `research/archive/issue60b_recent_form.py` | `docs/research_links.md` |
| `results/issue68_injury_promotion.json` + `.md` | Injury promotion path on a powered season holdout | *(archive)* `research/archive/issue68_injury_promotion.py` | Not cited |
| `results/issue69_status_trajectory.json` + `.md` | Intraday change in injury designation, 2025-26 holdout | *(archive)* `research/archive/issue69_status_trajectory.py` | Not cited |
| `results/issue70_null_replication.json` + `.md` | Meta-replication of recent nulls on 2024-25 | *(archive)* `research/archive/issue70_null_replication.py` | `docs/research_links.md` |
| `results/issue71_availability_sensitivity.json` + `.md` | Availability effect across a halflife/top-N sensitivity grid | *(archive)* `research/archive/issue71_availability_sensitivity.py` | `docs/availability_block.md` |
| `results/issue72_late_scratch.json` + `.md` | Late-scratch census of the injury store — the artefact recording **240,241 events / 2,046 snapshots** | `scripts/issue72_late_scratch.py` | Both: `docs/late_scratch.md` |
| `results/issue73_pre2020_parser_audit.json` + `.md` | Pre-2020 injury-report layouts audited against the live CDN parser | *(archive)* `research/archive/issue73_pre2020_parser_audit.py` | Not cited |
| `results/issue74_form_mismatch.json` + `.md` | Form-mismatch spots, priced at the close | `scripts/issue74_form_mismatch.py` | `docs/form_mismatch.md` |
| `results/issue81_layered_elo_form.json` + `.md` | Layered Elo + recent-form filter, priced at the close | `scripts/issue81_layered_elo_form.py` | `docs/layered_elo_form.md` |
| `results/nn_rolling.json` + `results/nn_rolling.md` | Feed-forward network on the canonical 26-feature frame (run M01) | `src/wippa_nba/models/nn_rolling.py` (writes `OUT_JSON = results/nn_rolling.json`) | `results_pipeline_a.md`, `models_and_training.md` |
| `results/recency_ablation.json` | Recency-halflife arm comparison on the canonical frame | `scripts/recency_ablation.py` | Not cited |
| `results/recency_followup.json` | Is the recency gain real or warm-up | `scripts/recency_followup.py` | Not cited |
| `results/round3_avail.jsonl`, `results/round3_avail_followup.jsonl`, `results/round3_calib.jsonl` | Round-3 runs: expanded availability block, concentration follow-up, nested calibration | *(archive)* `research/archive/round3_avail.py`, `round3_avail_followup.py`, `round3_calib.py` — these import `round2_common`, which no longer exists in the tree, so the outputs are not reproducible | Not cited |
| `results/round3_availability_summary.md` | Prose summary of the availability/calibration slice of round 3 | *(no script in tree)* — round-3 outputs; `research/archive/round3_common.py` records that these outputs are not reproducible here | Not cited |
| `results/round3_summary.md` | Prose summary of the round-3 team-Elo / squad-quality sweep | *(no script in tree)* — same round-3 provenance as above | `docs/experiment_log_2026-10.md` |

## Subdirectories

**`results/archive/`** — holds `experiments-leaky-frame.jsonl`: the experiment
ledger as it stood on the pre-fix (leaky) frame, kept rather than deleted so
the audit trail of the leaked measurements survives. `results_pipeline_a.md`
cites it by that description.

**`results/edge_search/`** — outputs of the pre-registered family scan over
the odds archive. Families 1/3/4 — `price_structure.*`, `bestprice_ml.*`,
`model_bestprice.*` — are written by `scripts/edge_search.py`; the rest are
*(archive)* producers: `user_rules.json` / `user_rules_50.json` by
`research/archive/backtest_user_rules.py` and `lay_rules.json` /
`lay_grid.json` by `research/archive/backtest_lay_rules.py`.
`docs/edge_search.md` documents the scan and names
`results/edge_search/price_structure.{json,md}` as the authoritative artefact.

**`results/exploratory/`** — two outputs whose own producer docstrings label
them exploratory, per `research/README.md`:
`elo_parity_underdog.json` (from `research/archive/explore_elo_parity_underdog.py`)
and `streak4_profile.json` (from `research/archive/explore_streak4.py`).

**`results/issue45/`** — the #45 schedule-block follow-up: per-cell grids
`grid_s{7,19,42,137,2718}_f{4,5,6,8}.json` plus
`decomposition_s42_f5.json` with its exact driver committed in-file as
`decomposition_driver.py.txt` (its header says it is "not part of any
pipeline"). The runner script itself is not in the tree;
`docs/experiment_log_2026-10.md` documents the run and its
`--out results/issue45/grid_s<S>_f<F>.json` invocation.

## Traceability

Every number in the root `README.md` traces to one of the files indexed here,
directly or through `results/experiments.jsonl`. Two cited numbers do not:

- **The totals-model 0.497 vs 0.504 comparison has no artefact.** It is cited
  in `audit_and_fixes.md` ("It scores **0.497** against **0.504** for always
  [predicting over]") and in `ROADMAP.md`. No file under `results/` holds it
  — the `0.497` occurrences in `results/c44_report.md` (a calibration bucket's
  actual rate) and `results/issue42_ensemble.md` (an Elo hit rate) are
  different quantities.
- **The injury-store count in the prose disagrees with its artefact.**
  `README.md`, `results_pipeline_b.md` and `docs/injury_pipeline.md` say
  **237,436** designation events; the artefact that records the store,
  `results/issue72_late_scratch.json` (and `docs/late_scratch.md`), says
  **240,241**. The artefact is the committed measurement; the prose numbers
  predate it.
