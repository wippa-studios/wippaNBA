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
| **The market beats the model decisively** | On 4,076 games with both a prediction and a closing line: market log loss **0.5994** vs the model's **0.6269**, a gap of 0.0275 with the interval entirely on the market's side (\|z\| = 7.78). That is a loss, not a tie. |
| **Betting selectively does not rescue it** | The parameter-free rule — bet wherever the model beats the posted price, **both sides considered** — places 3,440 bets and returns **−10.80%**. Not one of six margins is positive. |
| **Almost everything measured here is a null** | Player ratings, 52 rank columns, gradient boosting, recency weighting, the rebound block: every interval straddles zero. See [the one thing that is not](#the-one-thing-that-is-not-a-null). |
| **The best model is the simplest sensible one** | Logistic regression (log loss 0.6193) beats XGBoost, HistGradientBoosting and margin/total models. |
| **Do not use the committed model files** | They were trained on the leaked dataset and must be retrained. |

### Contents

- [How the repo is organised](#how-the-repo-is-organised)
- [Setup](#setup)
- [Usage](#usage) — NBA-data models · original betting pipeline · Flask app
- [Verifying and testing](#verifying-and-testing)
- [Results](#results)
- [The one thing that is not a null](#the-one-thing-that-is-not-a-null)
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

#### Current state: 1,001 passing, 0 failing

31 files, measured individually:

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
| `test_closing_line.py` | 11 | American-odds conversion, de-vigging, and row alignment |
| `test_create_games.py` | 11 | dataset assembly |
| `test_boosting_splits.py` | 10 | the 4-way split for gradient boosting |
| `test_kelly_american_odds.py` | 9 | price parsing |
| `test_availability.py` | 8 | that availability stays strictly causal and bounded |
| `test_expected_value.py` | 8 | EV arithmetic |
| `test_get_odds_data.py` | 8 | odds scraping |
| `test_recency_ablation.py` | 6 | the recency null stays a null |
| `test_dataset_manifest.py` | 5 | dataset provenance |
| `test_get_data.py` | 5 | team-stats scraping |
| `test_kelly_criterion.py` | 5 | Kelly sizing |

**Every guard in this table was added because something was measured to be wrong,
and several exist because a test caught a bug I introduced in the same commit.**
Three examples worth knowing about:

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
| **logistic regression (canonical)** | **0.6193** | **0.2150** | **0.7089** | **0.6570** |

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
biggest edge". Across 2 estimation methods × 3 rating families they add nothing the
team-level features do not already carry, and every interval straddles zero. It was
built to be falsifiable, and it was falsified.

### The one thing that is not a null

Almost every mechanism measured in this project came back flat. These did not.

| mechanism | delta | 95% CI | |
|---|---|---|---|
| **player-level expected minutes** | **+0.00382** | **[+0.00230, +0.00547]** | real, and small |
| *actual minutes (a leak)* | *+0.01089* | *[+0.00746, +0.01432]* | *unknown before tip-off* |
| *roster size alone (leak control)* | *+0.00109* | *[-0.00073, +0.00299]* | *straddles zero* |
| rebound block | +0.00040 | [−0.00026, +0.00104] | null |
| rank columns (52 cols) | +0.00065 | [−0.00039, +0.00166] | null |
| schedule block (7 cols) | −0.00194 | [−0.00349, −0.00043] | **lead, not a finding** |
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

**The schedule block is a lead, not a finding, and the arithmetic says so.** Six
blocks were scored. At |z| = 2.49 the two-sided p is 0.0128; a Bonferroni
threshold for six looks is 0.0083, so it **does not survive multiple-comparison
correction**, and it does not clear the |z| > 3 bar this project applies to
selection effects (Vandenbruaene et al. 2022, *J Sports Econ* 23(7) — the null of
unprofitability survived across 628 implementations). Rest days, back-to-backs,
3-in-4 and travel distance are the cheapest block in the set and the only one whose
removal has an interval entirely below zero, so it is worth a pre-registered
follow-up. It is not a discovery, and presenting it as one would be exactly the
error the rest of this page exists to prevent.

### Compared with the original model

| | data | rows | validation | log loss | AUC |
|---|---|---|---|---|---|
| original logistic regression | scraped odds | 17,356 | chronological 90/10 | 0.6189 | 0.7084 |
| logistic regression (this fork) | NBA game logs | 13,199 | 5× walk-forward | 0.6193 | 0.7089 |

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
| model | 0.62903 | 0.21927 | 0.6944 | 0.6411 |
| model + availability | 0.62693 | 0.21827 | 0.6979 | 0.6455 |

**The market wins decisively, and this is not a null.** Model + availability trails
by 0.02753 nats, CI [−0.03445, −0.02059], |z| = 7.78 — the interval excludes zero
on the market's side. What an efficient market looks like against a model built on
the same public information.

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
   and a closing line: market log loss **0.59940** against the model's **0.62693**, a
   gap of 0.02753 with CI [−0.03445, −0.02059] — the interval excludes zero on the
   market's side, |z| = 7.78.
3. **Accuracy is not profitability, and log loss is closer to profitability than
   accuracy is.** The market has both the best accuracy (0.6813) *and* the best log
   loss (0.59940); a model could beat it on one and still lose money on the other.
4. **The `68.9%` figure is retired.** It is reproduced here only to explain the leak.
   It is not a forecasting result.
5. **Effect sizes here are small, and the honest bar is high.** Everything measured
   is of order 0.001–0.03 nats. This project holds itself to |z| > 3, from
   Vandenbruaene et al. 2022 (*J Sports Econ* 23(7)), where the null of
   unprofitability was never rejected across 628 implementations after
   multiple-testing correction. A single 95% interval is not enough; six blocks were
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
