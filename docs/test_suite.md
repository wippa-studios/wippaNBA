# Test suite: 1,104 passing, 0 failing

**Repo:** `/home/Joel/.cache/nba-fix-worktree` (branch `improve-model-v2`)
**Run:** `pytest Tests/` — or `python -m unittest discover -s Tests -t .`

35 files, measured individually:

| test file | tests | what it protects |
|---|---|---|
| `test_ratings.py` | 106 | Elo/SRS/Bradley-Terry arithmetic and walk-forward correctness |
| `test_training_layer.py` | 94 | the sklearn version gate and the totals trainer end to end |
| `test_line_movement.py` | 85 | the odds snapshot store, coverage, and CLV arithmetic |
| `test_metrics.py` | 71 | log loss / Brier / AUC / calibration, cross-checked to scikit-learn |
| `test_validation.py` | 57 | purged walk-forward splits and selection-bias correction |
| `test_gbm.py` | 56 | gradient boosting, including that early stopping does not touch calibration |
| `test_baselines.py` | 54 | Elo tuning, same-day cost, and the base-rate bar |
| `test_kelly_risk.py` | 54 | fractional Kelly, caps, exposure limits |
| `test_logreg_rolling.py` | 54 | the canonical model's features, causality and leakage guards |
| `test_recency_features.py` | 53 | the as-of assertion on recency features |
| `test_margin_total.py` | 41 | margin/total design and the Φ(m̂/σ) conversion |
| `test_nba_client.py` | 39 | HTTP handling and endpoint fallback |
| `test_player_ratings.py` | 33 | player-level ratings and their ablation |
| `test_feature_block_ablation.py` | 22 | the ablation harness, including leak guards |
| `test_feature_alignment_checker.py` | 19 | that the checker validates the trainers' real contract |
| `test_line_movement_capture.py` | 19 | capture scheduling and exit codes |
| `test_game_date.py` | 18 | date alignment across the pipeline |
| `test_snapshot_provenance.py` | 14 | that a repeated price is never counted as a trajectory |
| `test_edge_betting.py` | 13 | the parameter-free betting rule and its pricing control |
| `test_feature_selection_errors.py` | 13 | that an empty feature list is an error, not a crash |
| `test_player_elo.py` | 13 | player Elo: causality, scale, and that `elo_available` is inverted, not ordered |
| `test_opponent_adjusted.py` | 12 | opponent-quality features: causality, and the margin sign convention |
| `test_closing_line.py` | 11 | American-odds conversion, de-vigging, and row alignment |
| `test_create_games.py` | 11 | dataset assembly |
| `test_boosting_splits.py` | 10 | the 4-way split for gradient boosting |
| `test_kelly_american_odds.py` | 9 | price parsing |
| `test_availability.py` | 8 | that availability stays strictly causal and bounded |
| `test_expected_value.py` | 8 | EV arithmetic |
| `test_get_odds_data.py` | 8 | odds scraping |
| `test_roi_denominator.py` | 6 | that ROI is divided by money staked, not by a signed sum |
| `test_data_guard.py` | 7 | that missing *or partial* data skips instead of crashing a loader |
| `test_recency_ablation.py` | 6 | the recency null stays a null |
| `test_dataset_manifest.py` | 5 | dataset provenance |
| `test_get_data.py` | 5 | team-stats scraping |
| `test_kelly_criterion.py` | 5 | Kelly sizing |

## Why each guard exists

**Every guard in this table was added because something was measured to be wrong,
and several exist because a test caught a bug I introduced in the same commit.**
The three largest post-mortems have their own pages:
[the ROI denominator](measurement_bugs.md), [the data
guard](measurement_bugs.md#2-the-data-guard-a-directory-that-exists-is-not-data-in-it),
and [the player-Elo leak](player_elo_leak.md). The rest, in one line each:

- `test_opponent_adjusted.py` exists because the first opponent-quality block computed
  `home_pts - away_pts` for **both** rows of the long frame — right for the home team,
  exactly negated for the away one. Nothing raised and every column looked plausible;
  the per-team home advantage came out at −0.07, because averaging a right-signed
  series against a left-signed one cancels.
- `test_snapshot_provenance.py` exists because `--at` labelled one real price with
  three timestamps, which reported as 100% coverage and a CLV of exactly 0.0 —
  indistinguishable from "the line held".
- `test_closing_line.py` exists because `np.searchsorted` silently returned the
  wrong rows (GAME_ID is a string and the haystack was unsorted), producing a model
  log loss of 0.699 against its own 0.6195 and an ROI swinging +93% to −70%.
- `test_edge_betting.py` exists because pricing the away side as `1 − p_home` rather
  than reading it made a *control* report +7.00% profit where it should place no bets.
- `test_player_elo.py` exists because a per-player Elo block measured **|z| = 5.18**
  and was leaking: it summed over the players who actually played, which nobody can
  know before tip-off. Truncating the future does **not** catch that — the leak is in
  the *membership*, not the weights — so the test asserts something else entirely:
  that the raw total is **inverted** relative to winning, which is what remains once
  the leak is removed.

## Skipping, not failing, without the data

The model suites under `src/Models/` need the downloaded NBA data. When `data/nba`
is absent they **skip** rather than fail, so a fresh clone still gets a meaningful
signal, and the guard lives centrally in `src/DataProviders/nba_data.py` rather than
in each test module — two runners execute this suite (`pytest` and `unittest`) and
they honour different skip mechanisms, so one change satisfies both.

SkipTest is a legitimate outcome here, not a failure: these tests cannot run without
an input the project does not ship.

## CI

CI runs on Python 3.10 and 3.11, plus a matrix job that exercises the totals trainer
on both sides of the scikit-learn 1.7 `multi_class` removal.