# Models, features and retraining

> The model modules, the shared feature layer, how to retrain, and the feature-order contract. Part of the [Legit NBA Predictions README](../README.md).

---

## How models were retrained and added

This is the section a new contributor needs first, because the model files in
`Models/` are **stale**: they were fitted on the leaked dataset, and the totals
artifact additionally carries the old column order — 107 inputs either way, so
nothing raises, it simply predicts wrong. Retrain before using anything.

### The model modules — 6 files, 12 distinct approaches, 18 configurations

Six modules under `src/Models/`, each standalone — `python -m src.Models.<name>` —
and each printing its own numbers, so nothing here depends on trusting this page.

**Six files is not six models, and the distinction matters.** Those six modules
hold **12 distinct model approaches**, scored in **18 configurations** — and the
count is itself part of the argument: "the simplest sensible model wins" only
carries weight if the alternatives were genuinely tried.

| | approaches | configurations |
|---|---|---|
| Scored head-to-head in the [Pipeline A comparison](results_pipeline_a.md#pipeline-a-model-comparison) | 9 | 13 |
| Unique to the [experiment log](results_pipeline_a.md#the-experiment-log-17-pre-registered-runs) | 3 | 5 |
| `player_ratings` methods (features, not standalone predictors) | 2 | 2 |

The nine are: a constant, Elo, SRS, Bradley-Terry, log-linear Elo, margin-and-total
→ Φ, XGBoost, HistGradientBoosting, and logistic regression. Several carry several
configurations — Elo alone appears four times (with and without a movement term,
K=10 and K=20), SRS twice.

Three approaches exist in **no model module at all** and live only in
`scripts/experiments.py`: **random forest**, **Gaussian Naive Bayes + PCA**, and
**logistic + PCA**. They are the least familiar entries and arguably the most
informative — the survey in `docs/repo_survey.md` names Gaussian NB + PCA as
best-in-class elsewhere, and here it scores **worse than always predicting the
favourite**. The other two extra configurations are logistic at `C=0.1` and
`C=0.01`, a walk down the regularisation path rather than new model classes.

| module | what it actually does | why it exists |
|---|---|---|
| `baselines` | Predicts the home win probability from **results alone**: Elo with a tuned K, a home-court-only constant, a logistic fit on Elo, and two batch rating systems (Bradley-Terry, SRS). | The bar. Any model that cannot beat "home wins 55% of the time" is not a model. It does no feature engineering on purpose. |
| `ratings` | Fits rating systems from past results only — the cleanest possible statement of "how much is knowable from what already happened". | Everything else has to beat this. If a model cannot beat Elo, it is reading noise. |
| `margin_total` | Does **not** predict a winner. Predicts the two things a game is made of — the point margin and the combined total — then converts to a win probability with Φ(margin/σ) and reports spread error too. | Tests whether a richer target pays for itself. It also outputs a pick-no-edge line, which is what a bettor actually needs. |
| `logreg_rolling` | One L2-penalised logistic regression on **exponentially-weighted team form**: each team's past margin, net rating, pace and four factors, plus rest days, back-to-backs, 3-in-4 and travel distance — and, since issue #36, the availability block (22 → 26 features). | The model the whole suite is measured against. Deliberately simple and deliberately causal. |
| `gbm` | Gradient boosting (XGBoost and HistGradientBoosting) on the **same features** as the model above, so the only variable is the algorithm. | The project spec expected boosting to win. It does not, and the confidence interval proves it is not a tie. |
| `player_ratings` | Builds **player-level** ratings and adds them to the team prediction. | The spec calls this "likely your biggest edge", so it was built to be falsifiable rather than to look good. Across 2 methods × 3 families it came back a null. |

A seventh mechanism — player-level **expected minutes**, summed over the squad that
played the team's *previous* game — lives in the shared layer rather than in a module
of its own, because it is a feature block and not a model. It is the one mechanism
that survived measurement, and since issue #36 it is part of the canonical model
(22 → 26 features).

Two things are deliberately *not* in that list. **Odds are never a feature.** Every
model above predicts from basketball information alone and is then scored against
the closing line separately, because a model that reads the price cannot be used to
evaluate the price. And the **totals/under-overs models** live under
`src/Train-Models/` in the original pipeline, not here.

### `load_design` — the one place a feature matrix is built

`src/DataProviders/nba_data.py` is the **single shared feature layer**. Every model
in suite A builds on it, so their numbers are comparable; the module's own docstring
says that writing a second loader or a second evaluation harness is a bug. It
supplies the loader, the causal feature builders, the walk-forward splitter, the
metric harness and the paired bootstrap.

`logreg_rolling.load_design(halflife=10.0, min_games=5, with_rebounds=True,
with_home_split=True, with_availability=True, games=None)` assembles those builders
into one design matrix, in order: `build_games()` for the frame,
`rolling_team_features()` for the exponentially-weighted team metrics,
`rest_and_travel()` for the schedule block, `rebound_features()` and
`opponent_adjusted_features()` for the rebound and per-team home/away split blocks,
and finally `assemble_features()`, which derives every **contrast** explicitly rather
than fitting raw home and away levels. `with_availability=True` (the default since
issue #36) appends the shared availability block last, in the block's own order;
`with_availability=False` returns the pre-#36 frame bit-for-bit, which is how every
pre-#36 measurement stays reproducible. `gbm.py`
calls the same function for its comparison frame, which is what makes the boosting
delta in [Results](results_pipeline_a.md#pipeline-a-model-comparison) a measurement of the algorithm
rather than of the features.

### Retraining the original pipeline's models

Four disjoint chronological segments — train, early-stop, calibration, test — and a
fixed seed, so the search measures the model rather than the noise.

```bash
# dataset first (see Usage → B for the full pipeline)
python -m src.Process-Data.Get_Data
python -m src.Process-Data.Get_Odds_Data
python -m src.Process-Data.Create_Games

# then retrain. --trials defaults to 15, not 100; see the note in Usage → B.
python -m src.Train-Models.XGBoost_Model_ML --dataset dataset_2012-26 --trials 15 --splits 5
python -m src.Train-Models.XGBoost_Model_UO --dataset dataset_2012-26 --trials 15 --splits 5
python -m src.Train-Models.Logistic_Regression_ML --dataset dataset_2012-26 --trials 15 --splits 5

# the neural trainers take the same --dataset and --trials. They validate on one
# contiguous slice rather than a fold series, so --splits does not apply, but
# they now take --calib-size for the calibration segment:
python -m src.Train-Models.NN_Model_ML --dataset dataset_2012-26 --trials 15
python -m src.Train-Models.NN_Model_UO --dataset dataset_2012-26 --trials 15
```

Each XGBoost trainer writes three files: the model, a `*_calibration.pkl`, and a
`*_features.json` holding the **exact ordered column list** the matrix was built
from — the model itself is a positional matrix, so nothing inside the `.json` records
that order. The neural trainers write to `Models/NN_Models/` (the directory the
runner reads) and save a `*_preprocessor.pkl` holding the train-fitted imputation
values and `StandardScaler`, plus `*_calibration.pkl` when calibration is on. The
runner applies that saved preprocessor and refuses a model that has none, because
the old per-row L2 normalisation cannot be reproduced from a raw inference frame.

### The feature-order contract

The trainers and the runners must agree on column *order*, not just on column
*names*, and both derive that order from one shared function —
`src/Train-Models/training_common.drop_columns_for()` on the training side,
`src/Utils/features.expected_feature_order()` at the inference boundary.
`validate_feature_frame()` rejects a frame whose columns match as a set but
disagree in sequence: the failure that would otherwise pass silently and predict
wrong. `python scripts/check_feature_alignment.py --use-live` reports the same
disagreement from the training side, and
`Tests/test_feature_alignment_checker.py` pins the equality so the two copies
cannot drift apart again. They did once, and six of ten entries were still correct
while disagreeing exactly where a permutation could hide. The full write-up is in
[docs/feature_order_contract.md](feature_order_contract.md).

---

## How pipeline A works

```
  NBA's own data          Features (26 cols)       Models            Judgement
  ──────────────          ──────────────────       ──────            ────────
  nba_api pull  ──────►   12 form contrasts     ──────►  Elo       ──────►  log loss
  310 CSVs                   (margin, rating,        logistic       Brier
  11 seasons                 pace, 4 factors,        margin/total   AUC
  13,199 games               rebounds) + 2           gradient       calibration
                             matchup + 7 schedule    boosting
                             (rest, back-to-back,
                             3-in-4, travel) + home
                             + 4 availability
                             (expected minutes)
```

1. **Pull the data.** `scripts/download_nba_data.py` fetches game logs, per-game team and
   player measures, schedules and season-end snapshots from `stats.nba.com`. It is
   resumable, and it drops `*_RANK` columns because those encode rank as of download
   time rather than as of any game.
2. **Build features.** `src/DataProviders/nba_data.py` is shared by every model, so
   their numbers are comparable. Each feature for a game is computed **only from
   games that already happened**. That is tested rather than asserted: deleting
   5,199 future games changes no historical feature by more than `0.000e+00`.
3. **Split by date, walk forward.** Train always precedes test, and never overlaps.
   Accuracy alone is useless here — always predicting "home" scores 55% — so log
   loss leads, followed by Brier, AUC and a calibration table.
4. **Run the models.** Each is standalone and prints its own results.
