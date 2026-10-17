# Legit NBA Predictions

[![CI](https://github.com/wippa-studios/Legit-NBA-Predictions/actions/workflows/ci.yml/badge.svg?branch=improve-model-v2)](https://github.com/wippa-studios/Legit-NBA-Predictions/actions/workflows/ci.yml?query=branch%3Aimprove-model-v2)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-3776AB.svg)](https://github.com/wippa-studios/Legit-NBA-Predictions/actions/workflows/ci.yml)
[![Data](https://img.shields.io/badge/data-nba_api-13%2C199%20games-informational.svg)](https://github.com/swar/nba_api)
[![Fork of kyleskom/NBA-Machine-Learning-Sports-Betting](https://img.shields.io/badge/fork%20of-kyleskom%2FNBA--Machine--Learning--Sports--Betting-blueviolet.svg)](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting)
[![No betting edge claimed](https://img.shields.io/badge/prediction%20quality%20only%20%E2%80%94%20no%20betting%20edge%20claimed-lightgrey.svg)](#pipeline-b-the-model-against-the-market)

A fork of [kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting) — 1,744 stars, 569 forks, maintained since 2019.

> **Credit:** all original code, pipelines and model scripts are the work of
> **Kyle Skompinski and contributors** ([github.com/kyleskom](https://github.com/kyleskom)).
> Kyle built a substantial, well-maintained project and did it in good faith. The
> defects below are the ordinary accumulation of years of iteration by one
> maintainer — most are invisible without measuring rather than reading. This fork
> corrects; it does not replace. Corrections, audit and documentation in this fork
> are ours.

---

## Summary

This fork does two things:

1. **It fixes defects it found in the original** (an audit, see [What the audit found](#what-the-audit-found)).
2. **It rebuilds the modelling layer from NBA's own data**, so results can be read against each other honestly (see [Results](#results)).

**There is no demonstrated betting edge here, and we do not claim one.** What follows
is an audit and a set of prediction models, judged on how well they predict — not on
whether they make money.

### Key takeaways

| | |
|---|---|
| **The original 68.9% headline was target leakage** | Season-to-date stats were scraped with the game's own result already included. Rebuilt without the leak, accuracy is ~48% on the affected rows. |
| **The market beats the model decisively** | On 4,076 games with both a prediction and a closing line: market log loss **0.5994** vs the model + availability at **0.6258**, a gap of 0.0264 with the interval entirely on the market's side (|z| = 7.36). That is a loss, not a tie. |
| **Betting selectively does not rescue it** | The parameter-free rule — bet wherever the model beats the posted price, **both sides considered** — places 3,440 bets and returns **−10.80%**. Not one of six margins is positive. |
| **Almost everything measured here is a null** | Player ratings, 52 rank columns, gradient boosting, recency weighting, the rebound block: every interval straddles zero. See [the one thing that is not](#the-one-thing-that-is-not-a-null). |
| **The one big-looking win was a leak** | A per-player Elo block measured **\|z\| = 5.18** — the largest effect ever found here — and it was leaking the real lineup. On a pre-game roster it is +0.00065 and does not earn its place. See [the player-Elo block](#the-player-elo-block-a-strong-result-that-was-a-leak). |
| **The best model is the simplest sensible one** | Logistic regression (log loss 0.6190) beats XGBoost, HistGradientBoosting, random forest, Gaussian NB and margin/total models. Every run is logged in `results/experiments.jsonl`. |
| **The null is the field's finding too** | A survey of ~45 public NBA repos finds the same accuracy ceiling (65–69%), and only two of them state plainly that they lose to the market. See [against other repos](#against-other-repos). |
| **Do not use the committed model files** | They were trained on the leaked dataset and must be retrained. |

### Contents

- [How the repo is organised](#how-the-repo-is-organised)
- [Setup](#setup)
- [Usage](#usage) — NBA-data models · original betting pipeline · Flask app
- [Verifying and testing](#verifying-and-testing)
- [Results](#results)
- [The one thing that is not a null](#the-one-thing-that-is-not-a-null)
- [The feature search](#the-feature-search-four-candidates-pre-registered)
- [The player-Elo block](#the-player-elo-block-a-strong-result-that-was-a-leak)
- [Nine more candidate blocks](#nine-more-candidate-blocks-all-measured-all-null)
- [Against other repos](#against-other-repos)
- [What the audit found](#what-the-audit-found)
- [What this fork changes](#what-this-fork-changes)
- [Tooling added here](#tooling-added-here)
- [Limitations](#limitations) — **read before using any number**
- [Further reading](#further-reading)

---

## How the repo is organised

There are two pipelines side by side. Knowing which one you are using is the most
useful thing to settle first.

| | **A. NBA-data models** (new) | **B. Original betting pipeline** (fixed) |
|---|---|---|
| **Purpose** | Predict game outcomes and measure how good the predictions are | Predict games against sportsbook odds, with Kelly sizing |
| **Data** | NBA's own data via `nba_api` (310 CSVs, 11 seasons, 13,199 games) | Scraped team stats and sportsbook odds |
| **Models** | Elo, logistic regression, margin/total, gradient boosting | XGBoost and neural networks |
| **Judged on** | Log loss, Brier, AUC, calibration | Accuracy, log loss, market baseline, ROI |
| **Entry points** | `scripts/download_nba_data.py`, `python -m src.Models.*` | `main.py`, `src/Process-Data/`, `src/Train-Models/` |

Supporting pieces: a Flask app (`Flask/`), an evaluation harness (`scripts/evaluate.py`),
and shared utilities (`src/Utils/`).

### How pipeline A works

```
  NBA's own data          Features (25 cols)       Models            Judgement
  ──────────────          ──────────────────       ──────            ────────
  nba_api pull  ──────►   20 exponentially       ──────►  Elo       ──────►  log loss
  310 CSVs                   weighted team                   logistic       Brier
  11 seasons                 metrics (margin,                margin/total   AUC
  13,199 games               rating, pace, 4                  gradient       calibration
                             factors) + rebounds              boosting
                          +  7 schedule cols
                             (rest, back-to-back,
                             3-in-4, travel)
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

---

## Setup

Python **3.10 or 3.11**. The ceiling is deliberate: `tensorflow==2.14.0` has no
wheels for 3.12+, and it cannot be bumped without a Keras 3 migration that also
invalidates the committed `.keras` artifacts. `requirements.txt` documents this in
full.

```bash
pip install -r requirements.txt
```

---

## Usage

### A. NBA-data models

```bash
# 1. Download data
python scripts/download_nba_data.py                 # core, 11 seasons
python scripts/download_nba_data.py --stages core,extras   # + tracking, synergy, shot charts

# 2. Run the models (each is standalone)
python -m src.Models.baselines       # the bar everything else must beat
python -m src.Models.logreg_rolling
python -m src.Models.margin_total
python -m src.Models.gbm

# 3. Measure the things that actually matter (each prints a paired bootstrap
#    interval, never a bare point estimate)
python scripts/closing_line.py             # does any of this beat the closing line?
python scripts/availability.py             # is the availability effect attainable honestly?
python scripts/feature_block_ablation.py   # does each feature block earn its place?
python scripts/recency_followup.py         # does recency weighting help?
python scripts/edge_betting.py             # bet only where the model beats the price
```

### B. Original betting pipeline

**Quick start**

```bash
python3 main.py -xgb -odds=fanduel
```

| flag | effect |
|---|---|
| `-xgb` | run the XGBoost models |
| `-nn` | run the neural-network models |
| `-A` | run all models |
| `-kc` | add Kelly Criterion bankroll fractions |
| `-odds=<book>` | fetch odds (`fanduel`, `draftkings`, `betmgm`, `pointsbet`, `caesars`, `wynn`, `bet_rivers_ny`) |

> ⚠️ The committed model files predate every fix in this fork and must be retrained
> before use. See [Limitations](#limitations).

**Data pipeline and training** (run from the repository root)

```bash
python -m src.Process-Data.Get_Data          # scrape daily team stats
python -m src.Process-Data.Get_Odds_Data      # scrape sportsbook odds + results
python -m src.Process-Data.Create_Games       # build the training dataset

python -m src.Train-Models.XGBoost_Model_ML   --dataset dataset_2012-26 --trials 15 --splits 5
python -m src.Train-Models.XGBoost_Model_UO   --dataset dataset_2012-26 --trials 15 --splits 5
```

Fill gaps with `--backfill` (optionally `--season 2025-26`).

`--trials` defaults to **15**, not 100. On this dataset the standard deviation of the
cross-validation score *across configurations* is several times the entire true
spread of held-out performance, so a bigger search ranks noise — and at 100 trials
the expected winner's curse is 2.50 standard errors. Every trainer now prints that
correction next to its own `Best val log loss`, so the number cannot be read without
it. See [ROADMAP.md](ROADMAP.md) Level 6.2.

> **Note on directory names.** `Process-Data` and `Train-Models` contain hyphens and
> so are not importable as packages. All three of these work and are tested:
> `python -m src.Train-Models.XGBoost_Model_ML` ·
> `python src/Train-Models/XGBoost_Model_ML.py` ·
> `cd src/Train-Models && python -m XGBoost_Model_ML`.
> Renaming the directories is tracked as follow-up work.

### Flask app

```bash
export RAPIDAPI_KEY="<your rapidapi key>"   # optional; team/player endpoints 503 without it
cd Flask && flask run
```

Do **not** run `flask --debug run` on any reachable host: debug mode serves
application source, including any credentials, to unauthenticated visitors.

---

## Verifying and testing

### Reproduce every number in this README

Nothing needs manual setup:

```bash
python scripts/evaluate.py
```

It opens with a leakage gate and **refuses to report accuracy at all** if the dataset
is contaminated, so a passing run is itself evidence that the join is clean. Then it
prints the baselines, the trained model, the closing-line favourite, and ROI across
edge thresholds with standard errors.

Useful flags: `--dataset`, `--test-size`, `--model {logreg,gbdt,none}`, `--skip-odds`.
See `scripts/README.md` for the gate's checks and degradation rules.

#### The experiment log

`scripts/run_experiments.py` is the iterative loop, made mechanical so the search
is an artefact rather than a memory. Every experiment is **declared with its
rationale before it is run**, executed on identical walk-forward folds, and
appended to `results/experiments.jsonl`.

```bash
python scripts/run_experiments.py --stage 1        # baselines and regularisation
python scripts/run_experiments.py --stage 2        # feature-block ablations
python scripts/run_experiments.py --stage 3        # RF / HistGB / XGBoost
python scripts/run_experiments.py --stage elo      # player-Elo block
python scripts/run_experiments.py --stage survey   # models named in the repo survey
```

It reports log loss as primary, with Brier, AUC and accuracy secondary and
**explicitly not ranked** — accuracy discards the confidence a bettor needs, and
two models can share it while disagreeing about every individual game. It also
reports the **train-vs-test gap** for every run, so an improvement bought by
memorising the training folds cannot hide, and it prints the **selection cost** of
the winner: the gap between best and median. A best-of-N is the maximum of N draws,
and the gap between best and median is the expected size of that maximum.

`scripts/feature_block_ablation.py` does the same job for feature blocks, with the
paired bootstrap and the Bonferroni accounting:

```bash
python scripts/feature_block_ablation.py --seed 42  --splits 5
python scripts/feature_block_ablation.py --seed 137 --splits 5
python scripts/feature_block_ablation.py --seed 2718 --splits 5
python scripts/feature_block_ablation.py --seed 42 --splits 8   # the fold-artifact check
```

`scripts/edge_holdout.py` answers a narrower question: does a *threshold chosen by
looking at the data* survive being chosen somewhere else? See
[the ROI arm](#a-number-in-the-committed-output-that-looked-good-and-was-a-bug).

### Run the tests

```bash
pytest Tests/                       # or: python -m unittest discover -s Tests -t .
```

Most tests need no network access and no TensorFlow or XGBoost import. The model
suites under `src/Models/` do need the downloaded NBA data described above; when
`data/nba` is absent they **skip** rather than fail, so a fresh clone still gets a
meaningful signal. That guard lives centrally in `src/DataProviders/nba_data.py`,
which raises `unittest.SkipTest` — so there is exactly one place where "needs data"
is decided.

#### Current state: 1,039 passing, 0 failing

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

**Every guard in this table was added because something was measured to be wrong,
and several exist because a test caught a bug I introduced in the same commit.**
Five examples worth knowing about:

- `test_player_elo.py` exists because a per-player Elo block measured **|z| = 5.18**
  and was leaking: it summed over the players who actually played, which nobody can
  know before tip-off. Truncating the future does **not** catch that — the leak is
  in the *membership*, not the weights — so the test asserts something else
  entirely: that the raw total is **inverted** relative to winning, which is what
  remains once the leak is removed. See
  [the player-Elo block](#the-player-elo-block-a-strong-result-that-was-a-leak).
- `test_roi_denominator.py` exists because `closing_line.roi()` divided total
  profit by `stake.sum()`, which included negative stakes, so a strategy losing
  9.37% reported **+13.54%**. A loss divided by a negative total.
- `test_data_guard.py` exists because the guard that skips the suite when the NBA
  data is absent checked for a *directory*, not for files. An empty `game_logs/`
  passed it, every glob then returned nothing, and CI failed with
  `ValueError: No objects to concatenate` — the exact failure the guard was written
  to prevent, reached by the one route it did not cover. A partial download failed
  the same way.
- `test_opponent_adjusted.py` exists because the first version of the
  opponent-quality block computed `home_pts - away_pts` for **both** rows of the
  long frame — right for the home team, exactly negated for the away one. Nothing
  raised and every column looked plausible. It surfaced only as a sanity check
  failing: the per-team home advantage came out at −0.07, because averaging a
  right-signed series against a left-signed one cancels.
- `test_snapshot_provenance.py` exists because `--at` labelled one real price with
  three timestamps, which reported as 100% coverage and a CLV of exactly 0.0 —
  indistinguishable from "the line held".
- `test_closing_line.py` exists because `np.searchsorted` silently returned the
  wrong rows (GAME_ID is a string and the haystack was unsorted), producing a model
  log loss of 0.699 against its own 0.6195 and an ROI swinging +93% to −70%.
- `test_edge_betting.py` exists because pricing the away side as `1 − p_home` rather
  than reading it made a *control* report +7.00% profit where it should place no bets.

CI runs on Python 3.10 and 3.11, plus a matrix job that exercises the totals trainer
on both sides of the scikit-learn 1.7 `multi_class` removal.

---

## Results

### The models, in plain terms

Seven modules. Each is standalone — `python -m src.Models.<name>` — and each prints its
own numbers so nothing here depends on trusting this page.

| module | what it actually does | why it exists |
|---|---|---|
| `baselines` | Predicts the home win probability from **results alone**: Elo with a tuned K, a home-court-only constant, a logistic fit on Elo, and two batch rating systems (Bradley-Terry, SRS). | The bar. Any model that cannot beat "home wins 55% of the time" is not a model. It does no feature engineering on purpose. |
| `ratings` | Fits rating systems from past results only — the cleanest possible statement of "how much is knowable from what already happened". | Everything else has to beat this. If a model cannot beat Elo, it is reading noise. |
| `margin_total` | Does **not** predict a winner. Predicts the two things a game is made of — the point margin and the combined total — then converts to a win probability with Φ(margin/σ) and reports spread error too. | Tests whether a richer target pays for itself. It also outputs a pick-no-edge line, which is what a bettor actually needs. |
| `logreg_rolling` | One L2-penalised logistic regression on **exponentially-weighted team form**: each team's past margin, net rating, pace and four factors, plus rest days, back-to-backs, 3-in-4 and travel distance. | The model the whole suite is measured against. Deliberately simple and deliberately causal. |
| `gbm` | Gradient boosting (XGBoost and HistGradientBoosting) on the **same features** as the model above, so the only variable is the algorithm. | The project spec expected boosting to win. It does not, and the confidence interval proves it is not a tie. |
| `player_ratings` | Builds **player-level** ratings and adds them to the team prediction. | The spec calls this "likely your biggest edge", so it was built to be falsifiable rather than to look good. Across 2 methods × 3 families it came back a null. |
| `availability` (in `nba_data`) | Adds each player's **expected minutes**, summed over the squad that played the team's *previous* game. | The one mechanism that survived measurement. Player-level expectations carry information team-level averages cannot. |

Two things are deliberately *not* in that list. **Odds are never a feature.** Every
model above predicts from basketball information alone and is then scored against
the closing line separately, because a model that reads the price cannot be used to
evaluate the price. And **the totals/under-overs models** live under
`src/Train-Models/` in the original pipeline, not here.

### Pipeline A: model comparison

All models are scored on the **same 6,600 walk-forward test rows** from the same five
splits, so they are directly comparable. Lower log loss and Brier are better; higher
AUC is better. Every figure below was produced by running the module, not copied
forward:

```bash
python -m src.Models.baselines        # Elo, SRS, Bradley-Terry, base rate
python -m src.Models.logreg_rolling    # the canonical model
python -m src.Models.gbm               # boosting, paired against the above
python -m src.Models.margin_total      # margin and total, then Φ(m̂/σ)
python -m src.Models.player_ratings    # player-level ratings
```

| model | log loss ↓ | Brier ↓ | AUC ↑ | accuracy |
|---|---|---|---|---|
| always predict home | 0.6883 | 0.2476 | 0.5010 | 0.5527 |
| Elo, no movement term | 0.6343 | 0.2220 | 0.6912 | 0.6394 |
| Elo (K=20, FiveThirtyEight default) | 0.6300 | 0.2187 | 0.7055 | 0.6594 |
| Elo + rest term | 0.6271 | 0.2182 | 0.7017 | 0.6529 |
| Elo (K=10, tuned on validation) | 0.6260 | 0.2180 | 0.7020 | 0.6512 |
| SRS + intercept (frozen) | 0.6923 | 0.2495 | 0.4867 | 0.5503 |
| SRS + intercept (walk-forward) | 0.6890 | 0.2479 | 0.5020 | 0.5500 |
| Bradley-Terry (walk-forward) | 0.6929 | 0.2494 | 0.5514 | 0.5568 |
| log-linear on Elo | 0.6234 | 0.2169 | 0.7026 | 0.6542 |
| margin + total → Φ(m̂/σ) | 0.6204 | 0.2156 | 0.7074 | 0.6562 |
| XGBoost | 0.6233 | 0.2167 | 0.7030 | 0.6562 |
| HistGradientBoosting | 0.6236 | 0.2169 | 0.7026 | 0.6526 |
| **logistic regression (canonical)** | **0.6190** | **0.2148** | **0.7093** | **0.6567** |

**What the table actually says.** Four results matter more than the ranking.

**Boosting loses, and the interval proves it is not a tie.** Against the canonical
model on identical rows and identical features:

| model | Δ log loss | 95% CI | verdict |
|---|---|---|---|
| XGBoost | −0.00394 | [−0.00557, −0.00229] | worse, significant |
| HistGradientBoosting | −0.00431 | [−0.00601, −0.00255] | worse, significant |

(Δ = loss(logreg) − loss(model), so negative means the booster is worse.) The two GBMs
agree with each other to within 0.0004, so this is not an implementation artifact.

**More complex targets do not pay.** Predicting margin and total, then converting
with Φ(m̂/σ), reaches 0.6204 — indistinguishable from fitting the label directly on
the same features (Δ +0.0006, CI [−0.0026, +0.0038]). Its value is the richer output:
margin RMSE 14.01, total RMSE 18.78, spread MAE 11.0.

**Batch rating systems lose to the base rate.** Bradley-Terry (0.6929) and SRS
(0.6890) both score *worse* than always predicting home (0.6883). The cause is
structural: a batch fit over five seasons reports where a team **was**, while the
next game depends on where it **is**. Removing Elo's movement term costs 0.0083 nats
— more than the entire gap between the best model and the base rate.

**Player ratings are a null.** The project spec calls player ratings "likely your
biggest edge". Built to be falsifiable, it was falsified. Δ = loss(team) −
loss(team + player), so **positive would mean the player features helped**:

| estimate | Δ log loss | 95% CI | verdict |
|---|---|---|---|
| RAPM-style, two-pool | +0.00151 | [−0.00059, +0.00358] | null |
| RAPM-style, shrunk | +0.00038 | [−0.00178, +0.00260] | null |
| logistic | +0.00060 | [−0.00154, +0.00282] | null |
| gradient boosting | −0.00243 | [−0.00815, +0.00295] | null |
| XGBoost | −0.00614 | [−0.01273, +0.00062] | null |

Every interval straddles zero, across both estimation methods and all three rating
families. Player-level ratings add nothing the team-level features do not already
carry — **with one exception**, and it is the important one.

The same module also measures the *availability gap* the honest way. Substituting
leaked actual minutes for expected minutes is worth −0.00975 nats,
CI [−0.01349, −0.00590] — an order of magnitude larger than anything above. See
[the one thing that is not a null](#the-one-thing-that-is-not-a-null): about a third
of that survives honestly, and the rest does not exist before tip-off.

### The one thing that is not a null

Almost every mechanism measured in this project came back flat. These did not.

| mechanism | delta | 95% CI | |
|---|---|---|---|
| **player-level expected minutes** | **+0.00382** | **[+0.00230, +0.00547]** | real, and small |
| *actual minutes (a leak)* | *+0.01089* | *[+0.00746, +0.01432]* | *unknown before tip-off* |
| *roster size alone (leak control)* | *+0.00109* | *[-0.00073, +0.00299]* | *straddles zero* |
| rebound block | +0.00040 | [−0.00026, +0.00104] | null |
| rank columns (52 cols) | +0.00065 | [−0.00039, +0.00166] | null |
| schedule block (7 cols) | −0.00203 | [−0.00358, −0.00051] | **lead, not a finding** |
| per-team home/away split (6 cols) | −0.00089 | [−0.00167, −0.00015] | **replicated, below the bar** |
| player Elo, pre-game roster (6 cols) | +0.00065 | [+0.00007, +0.00126] | **does not earn its place** |
| *player Elo, real lineup (a leak)* | *−0.00645* | *[−0.00890, −0.00402]* | *\|z\| 5.18 — the biggest fake win here* |
| recency weighting (halflife 5, 2) | worse | excludes zero | halflife 10 is already optimal |

**Availability is the first mechanism here whose interval excludes zero for the
right reason.** Summing each player's expected minutes over the squad that played
the team's *previous* game — known before tip-off whatever happens tonight —
improves log loss by 0.00382 nats. Team-level season averages cannot express it,
because a team is not one number.

**Only about a third of it is honestly available.** Substituting *actual* minutes is
worth +0.01089, roughly three times more, and those are known only after the game.
The middle row exists to make that falsifiable rather than asserted: roster size
alone, carrying no minutes information whatsoever, returns +0.00109 with an
interval straddling zero — so the gain is not "how many people dressed". The
remaining two thirds is injury news, in-game injury and ejection, which does not
exist before the game and caps what any model here can achieve.

**The schedule block is a lead, not a finding, and the arithmetic says so.** Ten
blocks are now scored against one full frame. At |z| = 2.59 the one-sided p is
0.005, which sits exactly on a Bonferroni threshold of 0.05/10 for ten looks
rather than comfortably inside it, and it still does not clear the |z| > 3 bar
this project applies to selection effects (Vandenbruaene et al. 2022, *J Sports
Econ* 23(7) — the null of unprofitability survived across 628 implementations).
Rest days, back-to-backs, 3-in-4 and travel distance are the cheapest block in the
set and the only one whose removal has a clearly negative interval, so it is worth
a pre-registered follow-up. It is not a discovery, and presenting it as one would
be exactly the error the rest of this page exists to prevent.

### The feature search: four candidates, pre-registered

Every decay experiment in this project failed, because all of them reweight the
*same* numbers the model already has. Faster and slower windows, ranks, four
factors, rebounds — all transformations. So the next round had to bring
information the frame did not already contain, and the four candidates were
**written into the block registry before any was measured**, so that the results
below cannot be cherry-picked out of a longer search. Reproduce with:

```bash
python scripts/feature_block_ablation.py --seed 42  --splits 5
python scripts/feature_block_ablation.py --seed 137 --splits 5
python scripts/feature_block_ablation.py --seed 2718 --splits 5
python scripts/feature_block_ablation.py --seed 42 --splits 8
```

Δ = loss(full) − loss(minus block), so **negative means the block earns its
place**.

| candidate | the new information | Δ | 95% CI | \|z\| | verdict |
|---|---|---|---|---|---|
| opponent-adjusted form | beating a good club counts for less | −0.00004 | [−0.00014, +0.00005] | 0.92 | null |
| **home/away split** | **each team's *own* home advantage** | **−0.00089** | **[−0.00167, −0.00015]** | **2.29** | **replicated** |
| strength of schedule | quality of opposition faced | −0.00008 | [−0.00015, −0.00000] | 2.01 | **died in replication** |
| opponent characteristics | three `OPP_*` columns never read | +0.00000 | [−0.00061, +0.00060] | 0.01 | null |

**One of four is real, and it is small.** The per-team home/away split is the only
candidate that replicated. Measured −0.00089 nats at three bootstrap seeds
(−0.00089 [−0.00161, −0.00011] and [−0.00166, −0.00017]) and at eight folds
instead of five (−0.00089 [−0.00167, −0.00013]) — the point estimate does not move.
It is worth having, and it is in the canonical model, where it moves log loss from
0.61956 to **0.61897**.

**It is also plainly below the bar, and both readings matter.** |z| ≈ 2.3 does not
clear |z| > 3, and one-sided p ≈ 0.011 does not clear a Bonferroni threshold of
0.005 for ten looks. And the size does not matter: 0.00089 nats is under a
quarter of the availability effect and about a thirtieth of what the closing
market beats this model by. It is a real feature, reported at its real size.

**One caveat on how to read the split columns.** They are built by blanking the
away rows and passing the column through the shared EWMA, and pandas' `ewm` decays
across NaN positions. So `ewm_margin_home` is not an EWMA over home games with
halflife 10 — its halflife counts *all* a team's games, making the effective window
about 20 home games. It is a defensible choice, but it means the split columns are
noisier than a reader would assume (sd 6.29 and 5.84 against 5.46 for the pooled
`ewm_margin`), which is part of why the gain is small. Documented in the function.

**Strength of schedule is the instructive failure.** At five folds it excluded
zero and the script called it a keeper — −0.00008, |z| = 2.01. At eight folds it
became +0.00000 [−0.00008, +0.00009], |z| = 0.11, **a clean null with the sign
flipped**. A result that exists only at one fold count is a property of the fold
count. The script's own verdict text says an exclusion is a candidate for a
second angle rather than a settled result; this is what following that advice
looks like, and it is why that sentence is in the code.

**Opponent-adjusted form was a null, exactly as predicted.** Its correlation with
the `ewm_margin` it adjusts is 0.98 — it is largely a rearrangement of information
the frame already had, written down as a prior before measuring. The prior was
right. What made `home/away split` different is visible in the same diagnostic:
`home_edge` correlates **0.08** with overall form, so it is close to orthogonal to
everything else in the matrix. New information and a rearrangement look identical
in a column list; only the correlation distinguishes them, and only after the fact.

**Opponent characteristics is the silent-absence bug again.** `OPP_FTA_RATE`,
`OPP_TOV_PCT` and `OPP_OREB_PCT` sit in the same table, under the same header, in
every downloaded season — and no model in this repo had ever read them. Only
`OPP_EFG_PCT` was used. It is structurally the same defect as the rebound
percentages that were missing from every model until `src/Utils/features.py` was
asked to name them, and the result is the same: +0.00000. They are now in the
frame and in the registry, so they are measured rather than invisible.

### Compared with the original model

| | data | rows | validation | log loss | AUC |
|---|---|---|---|---|---|
| original logistic regression | scraped odds | 17,356 | chronological 90/10 | 0.6189 | 0.7084 |
| logistic regression (this fork) | NBA game logs | 13,199 | 5× walk-forward | 0.6190 | 0.7093 |

**These two numbers are not a like-for-like comparison, and the near-identical values
are a coincidence rather than a result.** Different games, different coverage,
different seasons, and different split geometry. What can be said is narrower and
still worth saying: a linear model on team strength is hard to beat on this task,
whether the features come from a scrape or from NBA directly, and the models built on
top of it did not beat it here.

### Pipeline B: the model against the market

Measured on **4,076 games** that have both an out-of-fold prediction and a closing
line — `scripts/closing_line.py`. This supersedes an earlier 1,722-game reading on
the legacy dataset.

| scorer | log loss ↓ | Brier ↓ | AUC ↑ | accuracy |
|---|---|---|---|---|
| always bet home | 0.68704 | 0.24695 | 0.5000 | 0.5552 |
| **market (closing ML, de-vigged)** | **0.59940** | **0.20634** | **0.7337** | **0.6813** |
| model | 0.62777 | 0.21868 | 0.6964 | 0.6470 |
| model + availability | 0.62581 | 0.21771 | 0.6998 | 0.6528 |

**The market wins decisively, and this is not a null.** Model + availability trails
by 0.02641 nats, CI [−0.03339, −0.01932], |z| = 7.36 — the interval excludes zero
on the market's side. What an efficient market looks like against a model built on
the same public information. The market row is unchanged from the previous
measurement, which is the useful part: same prices, same games, so the movement
below is entirely the model.

**The feature work moved this number the wrong way, by an amount worth naming.**
Adding the home/away split improved the *canonical* model from 0.61956 to 0.61897,
but this comparison's own scorer went from 0.62693 to 0.62581 for availability and
the plain model went 0.62903 → 0.62777, leaving the gap at 0.02641 against
0.02753 before. The split did not help *here*.

That is not a contradiction, and the reason is in `scripts/closing_line.py`: it
builds its **own** feature frame rather than calling
`logreg_rolling.load_design`, so it feeds the raw per-side columns instead of the
contrasts the canonical model uses, and the sixteen new columns arrive without the
differencing that makes them informative. It is also the fourth time in this repo
that a duplicated feature definition drifted independently — after the drop lists
(#14, #28), the availability block (#30) and `margin_sweep`. The block was added
here explicitly, and the discrepancy is recorded rather than papered over by
reporting only the flattering half.

The honest summary is that the split is worth about 0.0006 nats on the model that
differences its features, and nothing measurable on the one that does not. Either
way the market is 0.026 nats ahead — roughly **7×** the availability effect, the
best prediction improvement this project has found, and about **30×** the split's
own.

#### Being selective does not help — including backing underdogs

A model does not have to be right every game to make money; it has to be right
where it disagrees with the price. The rule below has **no free parameter**, which
is the point: every threshold sweep is a multiple-comparisons problem, and this one
is fixed by arithmetic.

Bet one unit wherever the model's probability for a side exceeds the probability
implied by the price actually posted. Expected profit per unit is
`p·(1/p_raw − 1) − (1 − p) = p/p_raw − 1`, positive exactly when `p > p_raw`.

| strategy | bets | home | away | ROI | SE | \|z\| |
|---|---|---|---|---|---|---|
| model, breakeven rule | 3,481 | 1,552 | 1,929 | −12.11% | ±6.44% | 1.88 |
| model + availability | 3,440 | 1,512 | 1,928 | −10.80% | ±6.33% | 1.71 |
| margin 0.02 | 2,846 | — | — | −12.67% | — | 1.75 |
| margin 0.04 | 2,283 | — | — | −13.20% | — | 1.50 |
| margin 0.06 | 1,806 | — | — | −2.16% | — | 0.21 |

**Both sides are considered independently.** Restricting to the model's preferred
side would silently skip every game where the favourite has no edge but the
underdog does — 1,928 of them, more than half the bets placed. That is what "lay a
loser" means in a two-way market, and excluding it is not a shortcut, it is
skipping the question.

The comparison is against the **raw** price, not the de-vigged one, because the
**+3.98%** overround (p5–p95 2.96%–5.00%) is what a bettor actually pays.

**A control checks the pricing arithmetic.** Running the same rule on the market's
own de-vigged probability places **zero bets** — correct, because de-vigging
divides the overround away. An earlier version of this script priced the away side
as `1 − p_raw_home` instead of reading it from the data, which understated it by
the vig, made every away bet look better than it was, and made that control report
a spurious **+7.00%** profit. A control that profits when it should place no bets
is a broken price, not an edge.

**Conclusion: no demonstrated betting edge, in either direction.** What *is* solid
is the market's side, and the availability result below is a genuine improvement to
*prediction* that a 4% bookmaker already exceeds.

**The player-Elo bucket analysis reaches the same verdict from a new direction, and
that agreement is worth more than either result alone.** `elo_mean` orders games
correctly — decile home-win rate rises monotonically from 0.535 to 0.622, ρ = +0.85
— yet its spread across those buckets is **0.091, against the closing market's
0.099 on the very same games.** A feature that orders games marginally worse than
the price already does is a real feature and not a betting edge. See
[the player-Elo block](#the-player-elo-block-a-strong-result-that-was-a-leak).

#### A number in the committed output that looked good, and was a bug

`scripts/closing_line.py` printed this in an earlier revision:

| threshold | bets | model ROI | \|z\| | model + availability ROI | \|z\| |
|---|---|---|---|---|---|
| 0.55 | 815 | **+11.76%** | 2.22 | **+13.54%** | 2.54 |
| 0.60 | 638 | +6.19% | 1.02 | +3.52% | 0.59 |
| 0.65 | 478 | −2.11% | 0.30 | +0.07% | 0.01 |

**Those two positive numbers were a bug, not a finding, and the bug inverted the
sign.** The corrected figures, every threshold negative or flat:

| threshold | bets | model ROI | \|z\| | model + availability ROI | \|z\| |
|---|---|---|---|---|---|
| 0.55 | 815 | −8.55% | 1.62 | **−9.37%** | 1.78 |
| 0.60 | 638 | −5.02% | 0.86 | −2.74% | 0.46 |
| 0.65 | 478 | +1.76% | 0.25 | −0.06% | 0.01 |

**What was wrong.** The stake is `p_model − p_raw`, which is **negative whenever
the market is the more confident side** — at a 0.55 threshold, 1,382 of 4,076
games. Those are not short positions. They are not bets, and their stake is not
money. But the ROI divided total profit by `stake.sum()`, which summed them:

```
positive stakes   812, sum  +69.22   <- money actually at risk
negative stakes  1382, sum -117.10   <- not bets at all
stake.sum()             -47.88        <- the denominator as coded
total profit            -6.48         <- a LOSS
profit / sum(positive)  -9.37%        correct
profit / stake.sum()   +13.54%        as it was reported
```

A loss divided by a negative total reports as a profit. The same mistake was
inside the bootstrap, so the standard error was computed over the wrong population
too — a wrong ROI with a plausible-looking SE, which is harder to catch than
either alone.

**Why it survived as long as it did.** The bet *count* was right (812), every
individual stake was sensible, and every intermediate number was plausible. Only
the ratio was wrong. Sitting on top of a best-of-six threshold sweep, `+11.76%`
and `+13.54%` became the two best cells — which is exactly what a selection effect
looks like, and why it was initially written off in this file as multiplicity
rather than arithmetic. **That explanation was wrong, and being wrong in the
conservative direction is still wrong.** An independent reimplementation
(`scripts/edge_holdout.py`) computed the same 812 bets and got −9.37% from a
different code path; that disagreement is what found it.

Both the mean and the bootstrap now restrict to placed bets before aggregating.
Pinned by `Tests/test_roi_denominator.py`, whose headline test asserts the
invariant the bug violated: *the sign of the ROI is the sign of the total profit.*

---

### The player-Elo block: a strong result that was a leak

This is the longest entry here because the *shape* of it is the point.

Per-player Elo ratings, totalled over each team's expected-available roster, were
built as a feature block and scored. The first measurement was:

| roster construction | Δ log loss | 95% CI | \|z\| | verdict |
|---|---|---|---|---|
| **over the actual participants** | **−0.00645** | [−0.00890, −0.00402] | **5.18** | the largest effect ever measured here |
| over the pre-game roster | +0.00065 | [+0.00007, +0.00126] | 2.14 | **does not earn its place** |

**That first row was target leakage, and it was nearly published as a result.**
|z| = 5.18 clears this project's bar of 3, clears Bonferroni for eleven looks,
and is larger than every real effect in this repository combined. It came from
summing `rating × expected_minutes` over the players who **actually played**.

The *weights* were honest — trailing EWMAs, strictly causal, verified by truncating
the future. The leak was in the **membership**. Nobody knows before tip-off which
ten men will be on the floor, so a sum over the actual lineup tells the model who
played. It is the same failure mode as substituting actual minutes for expected
ones, which is worth +0.01089 against an honest +0.00382; the size of the
fabrication here sits exactly in that pattern, which is what made it worth
suspecting.

The fix rebuilds the roster from the past: for team X's game *t*, the available
roster is every player with an appearance for X **strictly before t**, carrying the
rating and trailing minutes from that last appearance. A player who misses the
game is still in the sum, discounted by his own trailing minutes — which is the
entire point of the block. Reproduce both numbers with
`python scripts/feature_block_ablation.py --blocks player_elo`.

**Bucketing the contrast explains the mechanism, without needing the model.**
`scripts/elo_buckets.py` removes the regression entirely: it takes the raw
home-minus-away Elo contrast, splits games into ten equal-population buckets, and
reports the actual home-win rate in each. Three columns, three different answers.

| contrast | Spearman ρ vs bucket | win rate, low → high bucket | spread / noise | market's own spread |
|---|---|---|---|---|
| `elo_mean` (scale-free) | **+0.85** | 0.535 → 0.622 | 6.7× | 0.099 |
| `elo_available` (raw total) | **−0.96** | 0.624 → 0.483 | 10.3× | 0.111 |
| `elo_top5` | +0.17 | 0.537 → 0.591 | 4.0× | 0.035 |

Read those three rows together:

1. **`elo_available` runs backwards.** The stronger a team's accumulated
   rating-minutes looks, the *less* it wins. It is not a quality measure at all —
   it is dominated by **departed** players, whose trailing EWMA decays without ever
   reaching zero, so a club that recently traded a star keeps carrying his weight
   while a club that just acquired one is under-counted. It is a lagging indicator
   of roster churn, and it is the column that made the leaky version look like a
   breakthrough.
2. **`elo_mean` is correctly ordered** — and its spread (0.091) is *narrower* than
   the closing market's spread on the same buckets (0.099). A feature that orders
   games slightly worse than the price already does is a real feature and **not a
   betting edge**, which is the whole conclusion of this project arriving from a
   completely new direction.
3. **`elo_top5` is nearly useless** (ρ = +0.17), which is expected: it is a crude
   proxy for a starting five that cannot notice a starter sitting out.

The bucket at zero contrast sits at **0.5585**, not 0.500 — correctly above a coin
flip, because an Elo difference says nothing about home court.

**What is left is honest and small.** The block is retained, behind a
`roster_mode` parameter, because a measured null with a diagnosed mechanism is worth
more than a deleted function. `all_prior`, `prev_game` and `top_minutes` are three
stated answers to "who is available", and the `elo_mean` numbers across them agree
to within a point (1546.4 / 1546.7 / 1548.0) even though their raw totals differ
by 10×. A confirmed starting five would be a fourth answer, but lineups are filed
**30–60 minutes before tip-off** — legitimate for a tip-off prediction and
impossible for a model run that morning. That is a prediction-point decision, not a
data gap, so it is a parameter that stays unfilled rather than a feature that
quietly becomes unavailable.

### Nine more candidate blocks, all measured, all null

`docs/modeling_exploration.md` builds and **scores** nine feature blocks out of the
tables nothing had read — actual travel distance, margin volatility, matchup-specific
deviation, momentum as a second difference, shot-chart mix, league-era effects,
never-read team columns, and two play-type PPP variants. All on the same five
walk-forward splits. Full document in `docs/`.

| block | cols | OOF log loss | vs base | \|z\| |
|---|---|---|---|---|
| actual travel (origin → destination) | 3 | 0.61890 | **+0.00026** | 1.02 |
| margin volatility (trailing sd) | 2 | 0.61889 | **+0.00026** | 1.19 |
| matchup deviation (this pairing) | 3 | 0.61911 | +0.00004 | 0.39 |
| momentum (last 2 vs last 10) | 4 | 0.61919 | −0.00004 | 0.06 |
| road-trip length | 2 | 0.61924 | −0.00009 | 0.68 |
| never-read team columns (28) | 28 | 0.61932 | −0.00017 | 0.22 |
| shot-chart shot mix (49% coverage) | 14 | 0.61952 | −0.00037 | 0.59 |
| league era / season index | 4 | 0.62043 | −0.00128 | 2.25 |
| prior-season play-type PPP (crossed) | 22 | 0.62084 | −0.00169 | 2.32 |
| prior-season play-type PPP (full) | 66 | 0.62763 | −0.00847 | **4.96** |

**The best of ten is +0.00026 at |z| = 1.19.** That is the finding, and the
correlation screen is what makes it one rather than an opinion about ideas: several
of these are about as orthogonal to the existing features as a team statistic can
be — trailing margin volatility at r = 0.04, the last-2-vs-last-10 margin
contrast at r = 0.06 — and they still predict nothing. If a quantity has no
correlation with team strength, schedule state, or opponent quality, and it still
does not pick a winner, it is not a weaker version of team strength. It is nothing.

**One block is worse in a way that is informative.** Prior-season play-type PPP
costs **−0.0085 nats at |z| = 4.96**, monotonically worse as columns are added,
and PCA does not rescue it — while correlating only r ≈ 0.08 with the features
that already carry the signal. The information arrived and a 59-column
unregularised frame could not carry it. That is an architecture problem, not a
data problem, and it is the one place where more work might plausibly pay.

**One candidate killed itself during the work.** The strongest idea — player-level
zone defence crossed with opponent shot mix — turned out to be untestable:
`D_FG_PCT` is populated on the `Overall` category only, and all five zone
categories are 100% NaN. Its first correlation reading (+0.691) had also used the
wrong season key, so it was reading the same leak this project warns about. Both
are recorded in the document rather than quietly dropped.

### Against other repos

`docs/repo_survey.md` compiles ~45 public NBA prediction repositories from their
own published READMEs. It is a useful external check for two reasons: it says what
the field's ceiling looks like, and it says how honestly the field reports.

**This project's accuracy sits inside the reported range, not above it.** Reported
moneyline accuracies in the survey: ~0.69 (whisdev), 0.669 (Pirkn, neural net),
0.6505 (Pirkn, ridge), 0.648 (Pirkn, XGBoost), 0.615 (cmunch1), 0.573 (Exidekat,
combined). This model's canonical accuracy is **0.6567** — mid-pack. The one
repo that reports a baseline alongside its own number, cmunch1, gets 0.615 against
0.580 for "home always wins"; this project's equivalent comparison is 0.6567
against **0.5527**, a wider margin than cmunch1's. So the model is not weak. It is
ordinary, and being ordinary at 0.657 is what the field looks like.

**Two repos in the survey say outright that they lose to the market.** tonyelhabr
states its log loss is worse than Benz's 538 model. NBA_Betting states that
"consistently outperforming Vegas lines remains a challenge", noting the vig. That
is the same conclusion this project reached by measurement, and the fact that it
is the minority view in the field is worth naming rather than hiding behind.

**The survey's §7 observation is the one that matters most.** It notes that *no*
reported accuracy figure across ~45 repos accounts for the vig, typically 4–8% on
a two-way market, and that a 65%-accurate model is roughly break-even only after
deducting it. This project measures the vig directly: **+3.98% mean overround**
(p5–p95 2.96–5.00%) on the closing lines, and a model-to-market gap of 0.026 nats.
So the accuracy figures in the survey — including the 0.69 and 0.669 — are
pre-vig numbers that cannot be read as profitability. Comparing this project's
0.6567 accuracy against those would be the same error the survey warns about.

**The survey's §6.4 finding matches this project's own measurements.** It observes
that feature volume does not track performance: nealmick uses 500+ features per
game across 20,000+ games and reports no accuracy, while the repos with the
strongest numbers use compact hand-engineered sets. That is exactly what happened
here — the 52 rank columns, the rebound block, the three never-read `OPP_*`
columns and the opponent-adjusted form were all null or worse than nothing, and
dropping the `OPP_*` columns very slightly *improved* log loss.

**Three architectures the survey names as winners were tested here, and all three
lose.** Gaussian Naive Bayes + PCA (best model in luke-lite), logistic on 20
principal components (rileypsmith reports 80 PCs beating 10), and ridge. See the
experiment log for the measured numbers. That is consistent with §6.4 rather than
in conflict with it: the winning numbers elsewhere come with vaguer evaluation
discipline than this project uses, and §6.5 records that 13 of 28 game-outcome
repos publish no metrics at all while one 29★ repo has no README whatsoever.

**What this project adds to that literature is not a better model.** It is a
baseline, a held-out protocol, a measured vig, and a comparison against the actual
closing price. Of the ~45 repos surveyed, one (cmunch1) publishes a baseline; the
survey itself identifies it as the exception. Most of the field's conclusions are
unfalsifiable without re-running the code, which is also the survey's §8 first
suggestion.

---

## What the audit found

Six defects, all measured on the committed data. Reproduce with
`python scripts/evaluate.py`.

| # | Defect | Impact |
|---|---|---|
| 1 | [Target leakage](#1-the-689-headline-was-target-leakage) | Headline accuracy and ROI were inflated |
| 2 | [No market baseline](#2-the-model-does-not-beat-the-market) | Model never compared against the market |
| 3 | [Totals model has no signal](#3-the-totals-model-has-no-signal) | Worse than always predicting "over" |
| 4 | [Harmful calibration](#4-calibration-was-actively-harmful) | Probabilities (and expected value) distorted |
| 5 | [Leaked API key](#5-the-web-app-leaked-a-live-api-key) | Live credential committed since 2023 |
| 6 | [Engineering gaps](#6-engineering-gaps) | Untested code, silent failures, bad labels |

### 1. The 68.9% headline was target leakage

`config.toml` scraped season-to-date stats with `DateTo` set to **the game date
itself**. NBA treats `DateTo` as inclusive, so the snapshot named `D` already
contained the result of the game on `D`. Seven columns carried the answer.

A rule using only contaminated columns recovers the label nearly perfectly:

```python
pred = 1 if (W_home[D] - W_home[D-1]) > (W_away[D] - W_away[D-1]) else 0
```

| year | as shipped | rebuilt |
|---|---|---|
| 2024 | 68.0% | 48.1% |
| 2025 | 96.1% | 45.6% |
| 2026 | **100.0%** | 38.5% |
| all | 49.1% | **48.4%** |

The leak is **not uniform**: 2012–2021 was scraped before tip-off and is genuinely
pre-game, while recent seasons were backfilled after the fact. So the chronological
split put mostly-clean data in training and 94.3% poisoned data in the test set.

| | accuracy | log loss | ROI of the model's own picks |
|---|---|---|---|
| as shipped (94% of test rows leaked) | 0.675 | 0.613 | **+4.21%** |
| correctly aligned, same rows | 0.647 | 0.632 | **−1.86%** |

Leakage was worth roughly **+2.8 points of accuracy and +6 points of phantom ROI**.

### 2. The model does not beat the market

The original had no market baseline anywhere. Once one is added (see
[Pipeline B results](#pipeline-b-the-model-against-the-market)), the closing-line
favourite wins outright and the model shows no demonstrable edge in either direction.

### 3. The totals model has no signal

`OU-Cover` is three-class (under / over / push) on a 49.5 / 49.4 / 1.1 split, with the
rare push class up-weighted ~30×. It scores **0.497** against **0.504** for always
predicting "over", and a worse log loss than the class prior.

### 4. Calibration was actively harmful

Inverse-frequency class weighting pushed mean home probability ~6 points off the true
base rate (0.5079 vs 0.5694) while the sigmoid calibrator was fit on **unweighted**
labels — correcting a distortion it had never been validated against. Expected value
is computed straight off those probabilities. The calibration set was also used as
the early-stopping set, the calibrator's fit set, and part of the final validation
fold.

### 5. The web app leaked a live API key

A hardcoded RapidAPI key sat in `Flask/app.py`, committed since 2023, reachable
through two unauthenticated routes. The README also told users to run
`flask --debug run`, whose Werkzeug debugger serves application source — the key
included — on any unhandled exception.

### 6. Engineering gaps

- No CI, no linting, no type checking.
- `pytest` and `unittest` both failed on a clean checkout.
- 74% of production code was in modules no test imported; entry points were at 0% coverage.
- 255 games were silently dropped by a `len(rows) != 30` guard.
- Nine postponed games and one phantom 190–190 tie were labelled as losses.
- Documented commands silently exited 0 while doing nothing.
- Three mutually incompatible days-rest formulas existed across training and inference.

---

## What this fork changes

### Correctness

- `Create_Games` joins each game to the most recent stats snapshot **strictly before**
  the game date, eliminating the leak at its source.
- Postponed, zero-score and phantom-tie games are excluded instead of being labelled
  as losses.
- Days-rest is computed from **date-only UTC values against the team's previous
  game**, in one shared implementation — replacing three divergent ones that included
  the current game and varied with the runner's timezone.
- The UO model receives its `OU` column at the position it was trained on. It was
  being appended, so XGBoost silently scored days-rest as the totals line.
- A shared `src/Utils/features.py` defines the feature contract once, and both
  runners now **assert** the inference frame matches the training column order — a
  right-columns-wrong-order frame is rejected rather than scored.

### Model evaluation

- Class weighting is opt-in and defaults to **off**; probabilities need to be
  calibrated, not balanced.
- Train / early-stop / calibration / test are four disjoint chronological segments.
- Each trainer persists its ordered feature list alongside the model.
- Reported metrics include **log loss, the closing-line favourite baseline, and
  realised ROI** — not accuracy alone.
- The hyperparameter search was ranking on noise: it reseeded per trial, so each
  trial ranked configurations differently rather than finding a better one. The seed
  is now fixed, and the expected winner's curse is printed next to every headline.
- Imputation is fitted on train inside each fold. It was fitted before the split, and
  14 of 106 columns have a median that moves more than 1% as a result.

### Security and operations

- The API key is read from `RAPIDAPI_KEY`; endpoints fail closed with a 503.
- Debug mode is gone; unhandled exceptions return a generic response, logged
  server-side. Raw exception text is no longer returned to clients.
- `player_id` is validated.
- Every `requests.get` and `subprocess` call has a timeout; the per-request
  triple-subprocess spawn is single-flighted and semaphore-capped.
- Upstream fetch failures raise a typed error instead of being swallowed into an
  empty DataFrame three frames downstream.

### Engineering

- CI on Python 3.10–3.11 running both `pytest` and `unittest`.
- `pyproject.toml`, a working `conftest.py`, and a declared Python ceiling.
- `requests`, `numpy` and `joblib` are declared instead of resolving transitively.
- Odds-date holes are backfilled; missing-season runs exit non-zero.
- SQLite connections are closed; `Add_Days_Rest.py` refuses to overwrite a table it
  could not parse.
- Season values come from `config.toml`, not from 19 hand-edited literals.
- Six near-identical training scripts collapsed to shared modules (2255 → 1551 lines)
  with byte-identical output.
- Six trainers disagreed about which dataset table to train on; they now share one
  name, resolved through a function that refuses same-day-join builds by name.

---

## Tooling added here

Each of these is built and tested. None is wired into the runtime yet — that is the
next deliberate step, tracked in [ROADMAP.md](ROADMAP.md).

| Module | What it is for |
|---|---|
| `scripts/evaluate.py` | One command that reproduces every number in this README. Opens with a **leakage gate** that refuses to report accuracy on a contaminated dataset, then reports baselines, model metrics, the closing-line favourite, and ROI with standard errors. Degrades explicitly rather than scoring a partial odds join. |
| `scripts/experiments.py` + `scripts/run_experiments.py` | The iterative experiment loop as an append-only log. Experiments are declared with their rationale **before** being run, executed on identical walk-forward folds, and appended to `results/experiments.jsonl` with log loss, Brier, AUC, the train-vs-test gap, and top features. Prints the selection cost of the winner so a best-of-N cannot masquerade as a result. |
| `scripts/feature_block_ablation.py` | Feature-block scoring by paired bootstrap, with `--seed` and `--splits` so any result can be **replicated** rather than accepted. Ten blocks against one consistent full frame. |
| `scripts/edge_holdout.py` | Leave-one-season-out test of whether a betting threshold survives being chosen on different seasons than it is scored on. Reports per-season ROI stability and whether the in-sample winner is even stable. |
| `src/Utils/metrics.py` | Expected calibration error, maximum calibration error, reliability curves, multi-class Brier, guarded log loss. Cross-checked against scikit-learn to `0.000e+00`. |
| `src/Utils/validation.py` | Purged walk-forward splits with embargo, plus selection-bias and deflated-Sharpe correction. On this dataset 100 hyperparameter trials are worth **~2.5 standard errors** — larger than most effects anyone would claim from it. |
| `src/Utils/Kelly_Criterion.py` | Fractional Kelly, probability shrinkage, two-way de-vig (proportional and Shin), per-bet and slate caps. The audited arithmetic underneath is unchanged. |
| `src/Process-Data/Get_Odds_Line_Movement.py` | Timestamped intraday odds snapshots and closing-line-value arithmetic. |
| `src/Process-Data/Create_Recency_Features.py` | EWMA and rolling features replacing season-to-date cumulative averages, with an as-of assertion that raises on leakage. |
| `src/Utils/nba_client.py` | Thin, lazily-imported `nba_api` wrapper. |

Two of these are worth reading the reasoning for, because both encode a lesson.

**`size_bet` exists because full Kelly is the wrong policy, not wrong maths.** The
underlying formula is correct — it is exactly `f* = (bp−q)/b`. But on a mis-specified
edge, a Monte-Carlo over 20,000 seasons ends at a **median 0.278× bankroll with a
100% chance of a 50% drawdown**; at quarter Kelly, a median 0.946× and no such
drawdown. The honest counterpoint is in the same report: when the edge is exactly
right, full Kelly returns 3.97× against quarter Kelly's 1.83×. Roughly half the
growth when you are right, in exchange for not going broke when you are wrong.

**`closing_line_value` exists because anything measured at the close is not a
signal.** Betting the closing line returns CLV of exactly `0.0` by construction. You
cannot beat a price you already have. An honest limit the module documents: the
historical odds tables are single scrapes, so there is **no intraday history to
recover** and past CLV cannot be computed. That question is permanently unanswerable
for games already played.

---

## Limitations

**Read this before using any number from this repository.**

### About the results

1. **No demonstrated betting edge.** This is now a *measured loss* rather than an
   absence of a result, and the distinction matters. Betting wherever the model beats
   the posted price — a rule with no free parameter, considering both sides — returns
   **−10.80%** over 3,440 bets, and none of six margins is positive. Reproduce with
   `python scripts/edge_betting.py`; the market comparison is
   `python scripts/closing_line.py`.
2. **The model is worse than the market it is trying to beat**, and the gap is now
   measured rather than inferred from accuracy. On 4,076 games with both a prediction
   and a closing line: market log loss **0.59940** against model + availability's
   **0.62581**, a gap of 0.02641 with CI [−0.03339, −0.01932] — the interval excludes
   zero on the market's side, |z| = 7.36.
3. **Accuracy is not profitability, and log loss is closer to profitability than
   accuracy is.** The market has both the best accuracy (0.6813) *and* the best log
   loss (0.59940); a model could beat it on one and still lose money on the other.
   This project's accuracy (0.6567) sits mid-pack among published NBA models
   (~0.615–0.69, `docs/repo_survey.md`), so being ordinary at 0.657 is the field's
   normal — and **no** published figure in that survey accounts for the vig, which
   this one measures directly at +3.98%. Those accuracy numbers cannot be read as
   returns.
4. **The `68.9%` figure is retired.** It is reproduced here only to explain the leak.
   It is not a forecasting result.
5. **Effect sizes here are small, and the honest bar is high.** Everything measured
   is of order 0.001–0.03 nats. This project holds itself to |z| > 3, from
   Vandenbruaene et al. 2022 (*J Sports Econ* 23(7)), where the null of
   unprofitability was never rejected across 628 implementations after
   multiple-testing correction. A single 95% interval is not enough; **ten blocks
   are** now scored against one full frame, and the largest of them reaches |z| = 2.59,
   still short of the bar.
   scored in one ablation and only one excluded zero, which did not survive
   correction.

### About the code and models

6. **The committed model artifacts predate every fix in this fork and must be
   retrained before use.** They were fitted on the leaked dataset above, and
   `Models/NN_Models/Trained-Model-OU-49.2_*.keras` additionally carries the old
   totals column order — it has 107 inputs either way, so nothing will raise, it will
   simply be wrong. Treat everything in `Models/` as untrusted until retrained.
7. **The totals model should not be used.** It does not beat a coin flip or a constant
   prior.
8. **Kelly sizing is full Kelly** with no fractional scaling, bankroll cap,
   simultaneous-exposure limit, or discount for model uncertainty. Full Kelly on a
   marginally-uncertain edge has a high probability of severe drawdown.

### Legal and usage

9. **Not financial advice.** Nothing here is a recommendation to wager. Sports betting
   carries real financial risk; this is an educational data-science project.
10. **Data provenance.** Team stats come from `stats.nba.com` (whose Terms of Use
    restrict redistribution) and odds from a scraped source. Neither is licensed for
    redistribution by this project.
11. **No license.** Neither this fork nor its upstream grants reuse rights.

---

## Further reading

- [ROADMAP.md](ROADMAP.md) — the levels between this working pipeline and a real
  forecasting or betting engine, with the measured case for why model tuning is not
  the bottleneck.
- `scripts/README.md` — what the evaluation harness measures and how it degrades.
