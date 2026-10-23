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
| **The market beats the model decisively** | On 5,788 games with both a prediction and a closing line (re-measured on complete odds data): market log loss **0.59766** vs the model + availability at **0.62034**, a gap of 0.02268 with the interval entirely on the market's side (\|z\| = 7.28). That is a loss, not a tie. |
| **Betting selectively does not rescue it** | The parameter-free rule — bet wherever the model beats the posted price, **both sides considered** — places 4,828 bets and returns **−8.23%**. No margin or threshold cell reaches \|z\| = 3 (strongest |z|: 1.91 margin sweep, 0.62 threshold sweep); the sweep's one positive cell, +3.99%, is \|z\| = 0.46 noise. |
| **Almost everything measured here is a null** | Player ratings, 52 rank columns, gradient boosting, recency weighting, the rebound block: every interval straddles zero. See [the one thing that is not](#the-one-thing-that-is-not-a-null). |
| **The biggest-looking win was a leak** | A per-player Elo block measured **\|z\| = 5.18** — the largest effect ever found here — and it was leaking the real lineup. On a pre-game roster it is +0.00065 and does not earn its place. See [the player-Elo block](#the-player-elo-block-a-strong-result-that-was-a-leak). |
| **The best model is the simplest sensible one** | Logistic regression (log loss 0.6157, canonical since issue #36) beats XGBoost, HistGradientBoosting, random forest, Gaussian NB and margin/total models. Every run is logged in `results/experiments.jsonl`. |
| **The close absorbs everything our model knows** | The market-anchored blend is *adverse*: price + model loses to price alone by 0.00177 nats (\|z\| 3.15), and availability beyond price + model is a tight null whose CI upper bound is 10× below the 0.002-nat minimum effect. See [the market anchor](docs/market_anchor.md). |
| **The null is the field's finding too** | A survey of ~45 public NBA repos finds the same accuracy ceiling (65–69%), and only two of them state plainly that they lose to the market. See [against other repos](#against-other-repos). |
| **Do not use the committed model files** | They were trained on the leaked dataset and must be retrained. |

### Contents

- [What we started with](#what-we-started-with)
- [What we tested](#what-we-tested)
- [What the audit found](#what-the-audit-found) — six numbered defects
- [What this fork changes](#what-this-fork-changes)
- [Tooling added here](#tooling-added-here)
- [The agent swarm](#the-agent-swarm) — how the measurement work was done, and its mistakes
- [How models were retrained and added](#how-models-were-retrained-and-added)
- [How the repo is organised](#how-the-repo-is-organised)
  - [How pipeline A works](#how-pipeline-a-works)
- [Results](#results)
  - [Pipeline A: model comparison](#pipeline-a-model-comparison)
  - [The experiment log](#the-experiment-log-17-pre-registered-runs)
  - [The one thing that is not a null](#the-one-thing-that-is-not-a-null)
  - [The feature search](#the-feature-search-four-candidates-pre-registered)
  - [The player-Elo block](#the-player-elo-block-a-strong-result-that-was-a-leak)
  - [Nine more candidate blocks](#nine-more-candidate-blocks-all-measured-all-null)
  - [Compared with the original model](#compared-with-the-original-model)
  - [Pipeline B: the model against the market](#pipeline-b-the-model-against-the-market)
  - [Stint-level rotations](#stint-level-rotations-the-one-table-nothing-has-read)
  - [Against other repos](#against-other-repos)
- [Setup](#setup)
- [Usage](#usage) — NBA-data models · original betting pipeline · Flask app
- [Verifying and testing](#verifying-and-testing)
- [Limitations](#limitations) — **read before using any number**
- [Further reading](#further-reading)

---

## What we started with

The original repository is [kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting)
— 1,744 stars, 569 forks, maintained since 2019. It scrapes daily NBA team stats,
joins them with sportsbook odds, trains XGBoost, Keras and logistic-regression
models for moneyline and totals, prints expected value and Kelly Criterion stake
sizes for the day's games, and ships a Flask UI for browsing the output.

Its central number is **68.9% accuracy**, and that figure appears nowhere in the
original README as prose. It exists only in a screenshot
(`Screenshots/output.png`) and in model filenames
(`Models/XGBoost_Models/XGBoost_68.9%_ML_*.json`). An unsourced, undated,
unqualified headline accuracy for a product whose entire purpose is placing bets
is what prompted this fork.

**What was actually true of that number:** `config.toml` scraped season-to-date
stats with `DateTo` set to the game date itself. NBA treats `DateTo` as inclusive,
so the snapshot named `D` already contained the result of the game on `D`, and
seven columns carried the answer. Leakage was worth roughly **+2.8 points of
accuracy and +6 points of phantom ROI**. The per-year table, the rule that recovers
the label from contaminated columns alone, and the rest of the audit are in
[What the audit found](#what-the-audit-found).

**What the original model scored honestly**, on a chronological 90/10 split of
17,356 scraped-odds rows: log loss **0.6189**, AUC **0.7084**. That is the real
baseline, and it is a hard one to beat. [This fork's
logistic regression reaches 0.6157 on 13,199 NBA game logs under 5× walk-forward
validation](#compared-with-the-original-model) — but the two are not a like-for-like
comparison, and any similarity between the values is a coincidence rather than a
result. What this repository actually establishes is narrower: **the model loses to the
closing market by 0.02268 nats at |z| = 7.28**, and everything measured on top of it
is either null or, once you look closely, a leak.

Three things were missing from that starting point, and each became a section below: a
**market baseline** (there was none anywhere), an **honest protocol** (the original
split put mostly-clean data in training and 94.3% poisoned data in test), and a
**feature layer nobody had audited** (52 rank columns, a rebound block, and three
`OPP_*` columns that no model had ever read).

---

## What we tested

Everything in this repository is a measurement, and every measurement has an
instrument that can be re-run. Nothing here is an assertion.

| what | instrument | what it decided |
|---|---|---|
| the committed pipeline | `python scripts/evaluate.py` | six defects, behind a leakage gate that **refuses to report accuracy** on a contaminated dataset |
| every model in suite A | `python -m src.Models.<name>` | log loss, Brier, AUC and calibration on the same 6,600 walk-forward rows, five splits |
| 17 candidate designs | `python scripts/run_experiments.py` | which design actually earns its place; every run, declared with its rationale **before** it ran, is in `results/experiments.jsonl` |
| ten feature blocks | `python scripts/feature_block_ablation.py --seed … --splits …` | whether each block earns its place, replicated at three seeds and two fold counts |
| model against the price | `python scripts/closing_line.py` | 5,788 games with both a prediction and a closing line (13,424 odds rows load) |
| the betting rule | `python scripts/edge_betting.py` | 4,828 bets, **−8.23%**, no free parameter, both sides considered |
| the code itself | `pytest Tests/` | **1,287 passing + 104 subtests, 0 failing** across 40 files |

Two rules apply to all of it and are worth stating before the numbers.

**Log loss leads; accuracy is reported and explicitly not ranked.** Always
predicting "home" scores 55%, so accuracy on its own cannot separate models. Log
loss cannot be gamed by a confident-but-wrong forecast.

**Every exclusion has to survive a second look.** The project's own bar is
**\|z\| > 3** (Vandenbruaene et al. 2022, *J Sports Econ* 23(7), where the null of
unprofitability survived 628 implementations), and where ten blocks are scored at
once, a Bonferroni threshold of 0.005 applies on top. Two results in this repository
clear the bar. The larger, |z| = 5.18, was a leak — see
[the player-Elo block](#the-player-elo-block-a-strong-result-that-was-a-leak). The
one that clears it honestly is the availability block now in the canonical model
(|z| = 4.78); see
[the one thing that is not a null](#the-one-thing-that-is-not-a-null).

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

## The agent swarm

Most of the measurement work in this repository was done by a swarm of agents
sharing one machine, communicating through an append-only message log. It is worth
describing plainly, because **how the work was done turned out to matter as much as
what it found** — including the parts that went wrong.

<details>
<summary><b>How it worked, what it found, and what it got wrong</b> (click to expand)</summary>

### How it worked

Agents posted to a shared JSONL bus under `~/Desktop/agents/inbox/`, in five kinds:
`feature`, `config`, `result`, `failure`, `note`. **102 messages.** No daemon, no
database — a crashed writer costs at worst one truncated line.

The design choice that carried the most weight: **`failure` was the largest category,
35 of 102.** The stated rationale is the right one — *"the expensive thing in a sweep
is re-running something that already failed."* Every dead end below was measured once
and never again.

### What it found

**One mechanism works, and it is the obvious one.** Expected player minutes beat the
bar:

| frame | cols | log loss | vs bar |
|---|---|---|---|
| bar (E14, `home_split_only`) | 43 | 0.618307 | — |
| + availability, as shipped | 47 | 0.614544 | +0.003763 |
| **+ availability, one contrast column** | **44** | **0.614337** | **+0.003970** |
| + availability + team Elo (promoted) | 48 | 0.612523 | +0.005784 |

**The win turned out to be a simplification.** `availability_features` emits four
columns, and two of them — `home_expected_minutes_diff` and `away_expected_minutes_diff`
— are *exactly* `d` and `−d`. **Three of the four are redundant**, and the single
contrast column does the whole job in fewer columns.

At |z| = 2.63 that is still **below this project's own bar**, and it is reported that
way. It is the best-measured feature effect here, not a discovery.

### What it closed

| lever | result |
|---|---|
| model family (85 arms, 8 families) | none beat the bar |
| regularisation (C sweep, 1 → 100) | flat; C=1.0 already optimal |
| calibration, nested honestly | null to harmful |
| feature selection (30-arm L1 + elastic-net) | best \|z\| 0.285 |
| bagging, subsampling, halflife averaging | null; nothing clears any bar |
| training window / recency weighting | worse at every setting |
| era drift | dead, measured |
| prior-season play-type (50 cols) | harmful, and harm **scales with column count** |

The last one has a mechanism: 5 columns cost −0.0004, 10 cost −0.0009, 50 cost
−0.0049. Monotone in width, never positive. **That is variance, not information** —
and it fits the learning curve, where fold log loss improves monotonically with
training size.

**The unifying conclusion: the model's error is bias, not variance. Only new
information can help.** That is why every variance-reduction lever is flat and why
the project stopped adding model arms.

### What it fixed

- **Two real bugs in the leakage gate** — the `blind` variant did not blind data read
  from disk, and a missing pre-event declaration made an honest builder look like it
  leaked. Both were found by *trying to use it*, which is how a tool should fail.
- **Its own shims.** An audit shim caught only **16 of 30** deliberately-injected
  leaks; the gap was found and closed.
- **A positive control on everything.** For the availability block the gate reports
  `membership: ok=False, clean=False` on the leaky variant and `clean=True` on the
  honest one, in the same run.

### What it got wrong

Worth recording, because the errors are the same classes this repository already
knows:

1. **A false claim about the odds data, published three times.** *"The database has
   zero rows for 2024-25 and 2025-26; market features are structurally unusable."*
   That was an artefact of the `load_odds` bug, not a property of the data. It was
   retracted by message id — after the fact.
2. **A vacuous robustness check.** A three-seed sweep was reported as replication; the
   estimator is deterministic, so all seeds returned identical numbers and the seed
   axis carried no information.
3. **No supersession.** An append-only bus cannot retract. A reader scanning by
   `kind=failure` hit the false entry first and stopped. Two notes carrying the same
   false claim were never retracted at all.
4. **A headline that was behind the state.** The availability number was published
   against the 43-column bar while a promoted 48-column frame already banked it —
   retracted as a supersession notice.

### What it produced

`f1f8e47f` and `7e3ef22d` — three feature modules, nineteen round-3 scripts, the
experiment logs, and a 283-line summary at `results/round3_summary.md`. Full suite
**1,165 passed + 103 subtests** at those commits, and CI green on both Python
versions.

</details>

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
| Scored head-to-head in the [Pipeline A comparison](#pipeline-a-model-comparison) | 9 | 13 |
| Unique to the [experiment log](#the-experiment-log-17-pre-registered-runs) | 3 | 5 |
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
delta in [Results](#pipeline-a-model-comparison) a measurement of the algorithm
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

# the neural trainers take the same --dataset and --trials, but validate on one
# contiguous slice rather than a fold series, so --splits does not apply:
python -m src.Train-Models.NN_Model_ML --dataset dataset_2012-26 --trials 15
python -m src.Train-Models.NN_Model_UO --dataset dataset_2012-26 --trials 15
```

Each XGBoost trainer writes three files: the model, a `*_calibration.pkl`, and a
`*_features.json` holding the **exact ordered column list** the matrix was built
from — the model itself is a positional matrix, so nothing inside the `.json` records
that order.

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
[docs/feature_order_contract.md](docs/feature_order_contract.md).

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

---

## Results

Every number below was produced by running the module or script named beside it. The
reasoning behind each table — and why the ordering below the top row is the wrong
thing to read — is in [docs/reading_the_results.md](docs/reading_the_results.md).

### Pipeline A: model comparison

All models are scored on the **same 6,600 walk-forward test rows** from the same five
splits, so they are directly comparable. Lower log loss and Brier are better; higher
AUC is better.

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
| **logistic regression (canonical, + availability, 26 feats)** | **0.6157** | **0.2133** | **0.7141** | **0.6573** |
| logistic regression (availability block off — the pre-#36 canonical) | 0.6196 | 0.2151 | 0.7085 | 0.6564 |

**The canonical row is post-#36** (`python -m src.Models.logreg_rolling`, 6,600
pooled out-of-fold rows, five splits): the availability block is in the model —
22 → 26 features — and log loss is **0.6157** (0.61566). The row beneath it is the
same model with `load_design(with_availability=False)`: the pre-#36 canonical,
preserved bit-for-bit at **0.6196** (0.61957), and still the frame every pre-#36
ablation is read against. The value this table carried before issue #36 —
0.6190 (Brier 0.2148, AUC 0.7093, accuracy 0.6567) — is kept as the measurement it
was and not restated; likewise the experiment log's E00 **0.61915** and E14
**0.61831** below, which are experiment-log-frame measurements — E14's is the
43-column `home_split_only` frame, not a stale version of the canonical number.

**Boosting loses, significantly.** Δ = loss(logreg) − loss(model), so negative means
the booster is worse; the two GBMs agree to within 0.0001, so this is not an
implementation artifact. Measured by `python -m src.Models.gbm`, re-run 2026-10-23
and paired against the **post-#36 canonical** (pooled log loss 0.61566):

| model | Δ log loss | 95% CI | verdict |
|---|---|---|---|
| XGBoost | −0.00785 | [−0.01038, −0.00533] | worse, significant |
| HistGradientBoosting | −0.00793 | [−0.01046, −0.00536] | worse, significant |

The sign is stable in every split (XGBoost −0.0102, −0.0013, −0.0122, −0.0055,
−0.0100; HistGB −0.0107, −0.0005, −0.0126, −0.0052, −0.0106). Two superseded
figures are kept as the measurements they were: against the **pre-#36** canonical
the same comparison read **−0.00394** [−0.00557, −0.00229] and **−0.00431**
[−0.00601, −0.00255], and the boosters' own pooled values re-measure at 0.6235 /
0.6236 (the XGBoost row in the table above is the earlier 0.6233 reading).

**Margin and total do not pay either.** Φ(m̂/σ) reaches 0.6204, indistinguishable
from fitting the label directly (Δ +0.0006, CI [−0.0026, +0.0038]); its value is the
richer output — margin RMSE 14.01, total RMSE 18.78, spread MAE 11.0.

**Bradley-Terry (0.6929) and SRS (0.6890) lose to always-predict-home (0.6883)**, for
a structural reason: a batch fit over five seasons reports where a team **was**.
Dropping Elo's movement term costs 0.0083 nats — more than the whole gap between the
best model and the base rate.

**Player ratings are a null.** The spec calls them "likely your biggest edge"; built to
be falsifiable, they were falsified. Δ = loss(team) − loss(team + player), so
**positive would mean the player features helped**:

| estimate | Δ log loss | 95% CI | verdict |
|---|---|---|---|
| RAPM-style, two-pool | +0.00151 | [−0.00059, +0.00358] | null |
| RAPM-style, shrunk | +0.00038 | [−0.00178, +0.00260] | null |
| logistic | +0.00060 | [−0.00154, +0.00282] | null |
| gradient boosting | −0.00243 | [−0.00815, +0.00295] | null |
| XGBoost | −0.00614 | [−0.01273, +0.00062] | null |

The one exception: substituting **leaked** actual minutes for expected ones is worth
−0.00975 nats, CI [−0.01349, −0.00590] — an order of magnitude larger than anything
above. See [the one thing that is not a null](#the-one-thing-that-is-not-a-null).

### The experiment log: 17 pre-registered runs

`scripts/run_experiments.py`, all on identical walk-forward folds (6,600 out-of-fold
rows). Log loss is the ranking metric; accuracy is reported and **not** ranked. Every
run is in `results/experiments.jsonl`; the script prints the selection cost of the
winner, because a best-of-N is the maximum of N draws.

Every figure in this table is an experiment-log-frame measurement — 59-, 53- and
43-column frames — made before issue #36, and kept as the measurements they were.
E14's **0.61831** is the 43-column `home_split_only` frame, not the canonical
model's number; the post-#36 canonical is **0.6157** in
[the Pipeline A comparison](#pipeline-a-model-comparison).

| experiment | feats | model | log loss | vs base | AUC | train−test gap |
|---|---|---|---|---|---|---|
| **E14-home-split-only** | **43** | logistic | **0.61831** | **−0.00084** | 0.7093 | +0.0017 |
| E31-drop-player-elo | 53 | logistic | 0.61835 | −0.00080 | 0.7095 | +0.0013 |
| E13-drop-opp-chars | 53 | logistic | 0.61892 | −0.00023 | 0.7080 | +0.0008 |
| E42-ridge | 59 | L2 logistic (ridge proxy) | 0.61894 | −0.00021 | 0.7078 | +0.0016 |
| E01-baseline-logistic-C0.1 | 59 | logistic C=0.1 | 0.61897 | −0.00018 | 0.7080 | +0.0008 |
| **E00-baseline-logistic** | 59 | logistic | **0.61915** | — | 0.7080 | +0.0003 |
| E12-drop-sos | 57 | logistic | 0.61919 | +0.00004 | 0.7080 | +0.0002 |
| E41-logistic-pca20 | 59 | logistic + PCA(20) | 0.61920 | +0.00005 | 0.7078 | +0.0031 |
| E11-drop-opp-adjusted | 57 | logistic | 0.61923 | +0.00008 | 0.7079 | +0.0002 |
| E10-drop-home-split | 53 | logistic | 0.62006 | **+0.00091** | 0.7068 | +0.0005 |
| E15-schedule-only | 37 | logistic | 0.62020 | +0.00105 | 0.7067 | +0.0016 |
| E20-rf-full | 59 | random forest | 0.62267 | +0.00352 | 0.7042 | **−0.1102** |
| E21-histgb-full | 59 | HistGradientBoosting | 0.62790 | +0.00875 | 0.6952 | **−0.0567** |
| E22-xgb-full | 59 | XGBoost | 0.63081 | +0.01166 | 0.6898 | **−0.0971** |
| E40-gaussian-nb-pca20 | 59 | Gaussian NB + PCA(20) | 0.64389 | +0.02474 | 0.6940 | +0.0044 |

**The best model is a 43-column logistic regression, and 16 of the 59 columns are
what is wrong with the 59-column version.** E14 uses *fewer* features and scores
*better*: the sixteen it drops — opponent-adjusted margin, strength of schedule, the
three never-read `OPP_*` pairs, and the six player-Elo aggregates — are worth
**negative 0.00084 nats**, which is to say they actively cost accuracy.

**E10 removes the per-team home/away split and costs +0.00091** — the largest penalty
of any ablation, and the only one whose sign is clearly the wrong way. That is the
difference between a feature that replicated at three seeds and two fold counts and
one that did not.

**Every tree model loses, and every tree model overfits.** Random forest +0.00352,
HistGB +0.00875, XGBoost +0.01166, with train−test gaps of −0.110, −0.057 and −0.097.
Linear models sit at +0.0003 to +0.0017, which is no overfitting at all.

**`logistic + PCA(20)`, once logged at 0.61605, loses.** That earlier run used a
59-column frame whose extra six columns were the **leaking** player-Elo block; on the
clean frame PCA(20) scores **0.61920**. The contaminated log is archived at
`results/archive/experiments-leaky-frame.jsonl` rather than deleted. `RidgeClassifier`
could not be scored at all — no `predict_proba` means no comparable probability — so
E42 is L2 logistic, the same estimator under a logit link.

**Do not read the winner's margin as an effect.** Best − median across the 17 runs is
−0.00026 nats and the top five are separated by less than 0.0007.

#### The experiment ledger: everything else tried

Everything tried in this month's campaign beyond the 17 frame runs, one row per
experiment, each pre-registered in `results/experiments.jsonl` with its rationale
**before** it ran. The 17 frame runs stay in the table above, kept as
experiment-log-frame measurements; the full chronicle — declared rationale →
result → verdict, the errata, and why each null closed its branch — is
[docs/experiment_log_2026-10.md](docs/experiment_log_2026-10.md).

| experiment | hypothesis | verdict | key number | detail |
|---|---|---|---|---|
| E00–E42 (17 frame runs) | which design earns its place on the 59/53/43-column frames? | none — the 43-col logistic wins; trees overfit | E14 0.61831 vs E00 0.61915 | [table above](#the-experiment-log-17-pre-registered-runs) |
| #45 schedule-block follow-up | the sub-bar lead buys the missing 0.41 of \|z\| at higher replication power | **stable sub-bar lead** — not a fold artifact, not a replication death; stays in the frame | \|z\| 2.55–2.76 over a 20-cell grid (Δ −0.00204 to −0.00212) | [log](docs/experiment_log_2026-10.md#s45--issue-45-schedule-block-follow-up) · `results/issue45/` |
| H3 recalibration | season-blocked recalibration gains ≥ 0.003 nats at zero new information | **null — isotonic/Platt significantly harmful; branch closed** | isotonic −0.0129 nats, \|z\| 4.56 | [log](docs/experiment_log_2026-10.md#h3--recalibration-null-and-two-variants-harmful) |
| H4 relative rest | rest enters as the rest-day differential (Esteves et al. 2021), ≥ 0.0024 nats | **null — B2B coefficients shrink 85% but no gain; branch closed** | primary Δ −0.00011, \|z\| 0.61 | [log](docs/experiment_log_2026-10.md#h4--relative-rest-null) |
| H11 role substitution | absence-context (what kind of absence) adds ≥ 0.002 nats on top of availability | **null with a lean to cost; branch closed** | primary Δ −0.00069 [−0.00153, +0.00020], \|z\| 1.56 | [log](docs/experiment_log_2026-10.md#h11--role-substitution-null-with-a-lean-to-cost) |
| H1a market-anchored blend | price + model beats price alone by ≥ 0.002 nats | **adverse — the blend is worse than price alone** | −0.00177 nats, \|z\| 3.15 (CI excludes zero *against*) | [market anchor](docs/market_anchor.md) |
| H2 availability beyond price | availability adds ≥ 0.002 nats beyond price + model | **tight null — CI upper bound 10× below the minimum effect; branch closed** | Δ −0.00004 [−0.00027, +0.00019], \|z\| 0.31 | [market anchor](docs/market_anchor.md) |
| C44 pace-first | pace-first decomposition beats the monolithic model on both RMSEs | **null at both endpoints; branch closed** | margin \|z\| 0.55, total \|z\| 1.51 | [log](docs/experiment_log_2026-10.md#c44--spread-and-total) |
| C44 Skellam | a proper margin distribution beats Φ(m̂/σ) on margin log score **or** cover loss | **loses margin log score, wins cover — dispersion, not shape** | −0.00401 nats (\|z\| 2.49) vs +0.00415 (\|z\| 14.53) | [log](docs/experiment_log_2026-10.md#c44--spread-and-total) |

Two errata ride with the ledger and are kept append-only: the S45 grid
artifacts carry a seed/fold metadata bug (`S45-erratum-artifact-metadata` maps
each artifact to its true run; every metric is per-run correct), and C44's
first execution had a secondary-endpoint bug whose corrected wide-bucket
comparison reads −0.00362 (\|z\| 2.27) — primary endpoints unaffected. Details
in [docs/experiment_log_2026-10.md](docs/experiment_log_2026-10.md#errata-append-only-the-earlier-rows-stand-as-written).

### The one thing that is not a null

Almost every mechanism measured in this project came back flat. These did not.

| mechanism | delta | 95% CI | |
|---|---|---|---|
| **player-level expected minutes** | **+0.00382** | **[+0.00230, +0.00547]** | real, and small |
| *actual minutes (a leak)* | *+0.01089* | *[+0.00746, +0.01432]* | *unknown before tip-off* |
| *roster size alone (leak control)* | *+0.00109* | *[-0.00073, +0.00299]* | *straddles zero* |
| rebound block | +0.00040 | [−0.00026, +0.00104] | null |
| rank columns (52 cols) | +0.00065 | [−0.00039, +0.00166] | null |
| schedule block (7 cols) | −0.00204 | [−0.00358, −0.00051] | **still a lead, below the bar** |
| per-team home/away split (6 cols) | −0.00089 | [−0.00167, −0.00015] | **replicated, below the bar** |
| player Elo, pre-game roster (6 cols) | +0.00065 | [+0.00007, +0.00126] | **does not earn its place** |
| *player Elo, real lineup (a leak)* | *−0.00645* | *[−0.00890, −0.00402]* | *\|z\| 5.18 — the biggest fake win here* |
| recency weighting (halflife 5, 2) | worse | excludes zero | halflife 10 is already optimal |

**Availability is the first mechanism here whose interval excludes zero for the right
reason.** Summing each player's expected minutes over the squad that played the team's
*previous* game — known before tip-off whatever happens tonight — improves log loss by
0.00382 nats. With issue #36 landed, its measured contribution to the canonical model
is Δ **+0.00390** [0.00237, 0.00557] at **|z| = 4.78** (`python -m
src.Models.logreg_rolling`) — above this project's |z| > 3 bar, and the only effect
measured here that clears it honestly. Three instruments agree on one effect:
**+0.00382** published (`results/availability.json`, the table above), **+0.00399** on
the ablation frame, **+0.00390** in the canonical model. Team-level season averages
cannot express it, because a team is not one number. The block's full history — the
mechanism, the three instruments, the membership-leak contrast behind the
prior-game-squad convention, and issue #36's wiring (22 → 26 features, bit-identical
pre-#36 frame) — is [docs/availability_block.md](docs/availability_block.md); its
boundary is now measured too: it improves forecasting and is fully priced at the
close ([docs/market_anchor.md](docs/market_anchor.md)).

**Only about a third of it is honestly available.** Substituting *actual* minutes is
worth +0.01089, roughly three times more, and those are known only after the game. The
middle row exists to make that falsifiable rather than asserted: roster size alone,
carrying no minutes information whatsoever, returns +0.00109 with an interval
straddling zero — so the gain is not "how many people dressed". The remaining two thirds
is injury news, in-game injury and ejection, which does not exist before the game and
caps what any model here can achieve.

**The schedule block is a stable small effect that does not meet the bar:** Δ −0.00204
[−0.00358, −0.00051] at 5 folds, |z| 2.55–2.76 across 20 cells (4/5/6/8 folds × 5
seeds), never reaching |z| > 3. Any cell reproduces with
`python scripts/feature_block_ablation.py --blocks schedule --seed <S> --splits <F>
--out results/issue45/grid_s<S>_f<F>.json`.† Not a fold artifact, not a replication
death; it stays in the model and stays a lead. At |z| = 2.59 the one-sided p is 0.005,
which sits exactly on a Bonferroni threshold of 0.05/10 for ten looks rather than
comfortably inside it, and it still does not clear the |z| > 3 bar (Vandenbruaene et
al. 2022, *J Sports Econ* 23(7) — the null of unprofitability survived across 628
implementations). It is not a discovery, and presenting it as one would be exactly the
error the rest of this page exists to prevent.

† The grid artifacts' `seed`/`n_splits` fields carried a metadata bug (fixed); `S45-erratum-artifact-metadata` in `results/experiments.jsonl` maps each artifact to its true run.

### The feature search: four candidates, pre-registered

Every decay experiment in this project failed, because all of them reweight the *same*
numbers the model already has. So the next round had to bring information the frame did
not already contain, and the four candidates were **written into the block registry
before any was measured**, so the results below cannot be cherry-picked out of a longer
search.

Δ = loss(full) − loss(minus block), so **negative means the block earns its place**.
Reproduce with `python scripts/feature_block_ablation.py --seed 42 --splits 5` (and
`--seed 137`, `--seed 2718`, `--splits 8` for the replication).

| candidate | the new information | Δ | 95% CI | \|z\| | verdict |
|---|---|---|---|---|---|
| opponent-adjusted form | beating a good club counts for less | −0.00004 | [−0.00014, +0.00005] | 0.92 | null |
| **home/away split** | **each team's *own* home advantage** | **−0.00089** | **[−0.00167, −0.00015]** | **2.29** | **replicated** |
| strength of schedule | quality of opposition faced | −0.00008 | [−0.00015, −0.00000] | 2.01 | **died in replication** |
| opponent characteristics | three `OPP_*` columns never read | +0.00000 | [−0.00061, +0.00060] | 0.01 | null |

**One of four is real, and it is small.** The per-team home/away split is the only
candidate that replicated — the point estimate does not move at three bootstrap seeds or
at eight folds instead of five. The block sits in the canonical frame, and adding it to
the model moves log loss from 0.61956 to **0.61897** (both pre-#36 measurements; the
canonical model's post-#36 number is 0.6157, availability block included). It is also
plainly below the bar: |z| ≈ 2.3 does not clear
|z| > 3, one-sided p ≈ 0.011 does not clear a Bonferroni threshold of 0.005 for ten
looks, and 0.00089 nats is under a quarter of the availability effect and about a
thirtieth of what the closing market beats this model by. A real feature, reported at
its real size.

**The instructive failure is strength of schedule.** At five folds it excluded zero and
the script called it a keeper; at eight folds it became a clean null with the sign
flipped. A result that exists only at one fold count is a property of the fold count.
Why each candidate went the way it did, and the two ways a candidate can fool you —
the fold artifact and the unwritten duplicate — is in
[docs/feature_search.md](docs/feature_search.md).

### The player-Elo block: a strong result that was a leak

Per-player Elo ratings, totalled over each team's expected-available roster, were built
as a feature block and scored. The first measurement was:

| roster construction | Δ log loss | 95% CI | \|z\| | verdict |
|---|---|---|---|---|
| **over the actual participants** | **−0.00645** | [−0.00890, −0.00402] | **5.18** | the largest effect ever measured here |
| over the pre-game roster | +0.00065 | [+0.00007, +0.00126] | 2.14 | **does not earn its place** |

The first row is target leakage, and it was nearly published as a result. |z| = 5.18
clears this project's bar of 3, clears Bonferroni for eleven looks, and is larger than
every real effect in this repository combined. The *weights* were honest — trailing
EWMAs, strictly causal, verified by truncating the future. The leak was in the
**membership**: it summed over the players who actually played, and nobody knows before
tip-off which ten men will be on the floor. On a pre-game roster the same block is
+0.00065 and does not earn its place. Full post-mortem, including the bucket analysis
that shows the raw total runs **backwards** — the stronger a team's accumulated
rating-minutes looks, the *less* it wins:
[docs/player_elo_leak.md](docs/player_elo_leak.md).

### Nine more candidate blocks, all measured, all null

`docs/modeling_exploration.md` builds and **scores** nine feature blocks out of the
tables nothing had read — actual travel distance, margin volatility, matchup-specific
deviation, momentum, shot-chart mix, league-era effects, never-read team columns and
two play-type PPP variants — all on the same five walk-forward splits. Five of the ten,
and the shape of the result:

| block | cols | OOF log loss | vs base | \|z\| |
|---|---|---|---|---|
| actual travel (origin → destination) | 3 | 0.61890 | **+0.00026** | 1.02 |
| margin volatility (trailing sd) | 2 | 0.61889 | **+0.00026** | 1.19 |
| never-read team columns (28) | 28 | 0.61932 | −0.00017 | 0.22 |
| league era / season index | 4 | 0.62043 | −0.00128 | 2.25 |
| prior-season play-type PPP (full) | 66 | 0.62763 | −0.00847 | **4.96** |

**The best of ten is +0.00026 at |z| = 1.19**, and the correlation screen is what makes
that a finding rather than an opinion: several of these are about as orthogonal to the
existing features as a team statistic can be — trailing margin volatility at r = 0.04,
the last-2-vs-last-10 margin contrast at r = 0.06 — and they still predict nothing.
One block is worse in an informative way: prior-season play-type PPP costs **−0.0085
nats at |z| = 4.96**, monotonically worse as columns are added, while correlating only
r ≈ 0.08 with the features that already carry the signal. The information arrived and a
59-column unregularised frame could not carry it. The full table, the correlation
screens, the rejected candidates and the reproduce instructions are in
[docs/modeling_exploration.md](docs/modeling_exploration.md).

### Compared with the original model

| | data | rows | validation | log loss | AUC |
|---|---|---|---|---|---|
| original logistic regression | scraped odds | 17,356 | chronological 90/10 | 0.6189 | 0.7084 |
| logistic regression (this fork, post-#36) | NBA game logs | 13,199 | 5× walk-forward | 0.6157 | 0.7141 |

**These two numbers are not a like-for-like comparison.** Different games, different
coverage, different seasons, different split geometry. Pre-#36 this fork's row read
0.6190 / 0.7093 — near-identical to the original's 0.6189 by coincidence rather than
as a result, and kept in [the Pipeline A comparison](#pipeline-a-model-comparison) as
the measurement it was. What can be said is narrower and still worth saying:
a linear model on team strength is hard to beat on this task, whether the features come
from a scrape or from NBA directly, and the models built on top of it did not beat it
here.

### Pipeline B: the model against the market

Measured on **5,788 games** that have both an out-of-fold prediction and a closing
line — `python scripts/closing_line.py`. **13,424 odds rows load** (was 11,400:
issue #47's 1,880 dropped closing lines are recovered), and the twin table spellings
across nine seasons are reconciled — 11,316 exact duplicates collapsed, conflicting
moneylines refuse to resolve rather than picking a side. This supersedes the
pre-refactor 4,076-game reading and, before that, a 1,722-game reading on the legacy
dataset. The pre-refactor numbers are kept below, labelled, as the post-mortems they
are.

| scorer | log loss ↓ | Brier ↓ | AUC ↑ | accuracy |
|---|---|---|---|---|
| always bet home | 0.68791 | 0.24738 | 0.5000 | 0.5511 |
| **market (closing ML, de-vigged)** | **0.59766** | **0.20573** | **0.7363** | **0.6818** |
| model | 0.62340 | 0.21685 | 0.7034 | 0.6513 |
| model + availability | 0.62034 | 0.21540 | 0.7082 | 0.6543 |

**The market wins decisively, and this is not a null.** Model + availability trails by
**0.02268** nats, CI [0.01638, 0.02859], |z| = 7.28; the plain model trails by
**0.02574** at |z| = 8.18. Each interval sits entirely on the market's side. The gap
is roughly **6×** the availability effect (0.0039 nats, the best prediction
improvement found here) and about **25×** the per-team home/away split's own
(0.00089).

**Does anything we know add to the close? No.** The market-anchored blend — price and
model logits stacked, walk-forward by season on 5,328 scored games — is **worse than
the price alone** by 0.00177 nats at |z| = 3.15 (H1a, adverse), and availability's
increment beyond price + model is a tight null whose CI upper bound is 10× below the
0.002-nat minimum effect (H2). The protocol, the arm table and the
disagreement-decile diagnostic are in
[docs/market_anchor.md](docs/market_anchor.md).

**Pre-refactor, the same comparison read market 0.59940 vs model + availability
0.62581 — a gap of 0.02641 at |z| = 7.36 on 4,076 games**, through the #47-short
loader (11,400 rows) and `closing_line`'s own drifted frame. That is kept as the
defect record it is, not as a current claim. The feature work moved that number the
wrong way, by an amount worth naming: adding
the home/away split improved the *canonical* model from 0.61956 to 0.61897, but this
comparison's own scorer went from 0.62693 to 0.62581 for availability and the plain
model went 0.62903 → 0.62777, leaving the gap at 0.02641 against 0.02753 before.
`scripts/closing_line.py` built its **own** feature frame rather than calling
`logreg_rolling.load_design`, so the sixteen new columns arrived without the
differencing that makes them informative — the fourth time in this repo a duplicated
feature definition drifted independently, after the drop lists (#14, #28), the
availability block (#30) and `margin_sweep`. The re-measured comparison above is built
on the canonical frame instead.

#### Being selective does not help — including backing underdogs

Bet one unit wherever the model's probability for a side exceeds the probability implied
by the price actually posted — **no free parameter**, so no threshold sweep to
cherry-pick from. Expected profit per unit is `p·(1/p_raw − 1) − (1 − p) = p/p_raw − 1`,
positive exactly when `p > p_raw`.

| strategy | bets | home | away | ROI | SE | \|z\| |
|---|---|---|---|---|---|---|
| model, breakeven rule | 4,855 | 2,475 | 2,380 | −10.21% | ±5.33% | 1.91 |
| model + availability, breakeven | 4,828 | 2,437 | 2,391 | −8.23% | ±5.10% | 1.61 |
| margin 0.02 (model) | 4,003 | — | — | −9.68% | ±6.35% | 1.52 |
| margin 0.04 (model) | 3,201 | — | — | −8.23% | ±7.22% | 1.14 |
| margin 0.06 (model) | 2,474 | — | — | −2.55% | ±8.51% | 0.30 |
| margin 0.02 (+ availability) | 3,911 | — | — | −3.97% | ±6.20% | 0.64 |
| margin 0.04 (+ availability) | 3,112 | — | — | −4.63% | ±7.21% | 0.64 |
| margin 0.06 (+ availability) | 2,402 | — | — | +3.99% | ±8.72% | 0.46 |

(Pre-refactor, the same rule on 4,076 games: 3,440 bets at −10.80% for the
availability arm, 3,481 at −12.11% for the model — superseded by the rows above,
kept as the record.)

**The sweep's one positive cell — margin 0.06 with availability, +3.99% — is |z| =
0.46 noise**, reported here precisely so it cannot be quoted as a result. No margin
cell reaches |z| = 3 (strongest 1.91), and the absolute-threshold sweep agrees:
−1.97%, −0.82%, −0.64% at 0.55/0.60/0.65 for the model, −2.86%, +0.15% and −1.44%
with availability (`python scripts/closing_line.py`) — **no cell reaches |z| = 3,
strongest 0.62**. The leave-one-season-out test — a threshold chosen on ten seasons,
scored on the eleventh — pools **937 out-of-sample bets at −2.53% ±4.85%**
(`python scripts/edge_holdout.py`): the in-sample winner does not survive being
chosen somewhere else.

**Both sides are considered independently.** Restricting to the model's preferred side
would silently skip every game where the favourite has no edge but the underdog does —
2,391 of them, half the bets placed (1,928 of 3,440 on the pre-refactor reading).
That is what backing the underdog side means in a two-way market, and excluding it is
skipping the question. The comparison is against the
**raw** price, not the de-vigged one, because the **+4.13%** mean overround on the
settled games is what a bettor pays (`python scripts/edge_betting.py`;
`python scripts/closing_line.py` measures +4.03% over its full join, p5–p95
2.96%–5.84%).

**A control checks the pricing arithmetic.** Running the same rule on the market's own
de-vigged probability places **zero bets** — correct, because de-vigging divides the
overround away. An earlier version priced the away side as `1 − p_raw_home` instead of
reading it, which made that control report a spurious **+7.00%** profit. A control that
profits when it should place no bets is a broken price, not an edge.

**Conclusion: no demonstrated betting edge, in either direction.** What *is* solid is the
market's side, and the availability result above is a genuine improvement to *prediction*
that a 4% bookmaker already exceeds.

**The player-Elo bucket analysis reaches the same verdict from a new direction.** `elo_mean`
orders games correctly — decile home-win rate rises monotonically from 0.535 to 0.622, ρ
= +0.85 — yet its spread across those buckets is **0.091, against the closing market's
0.099 on the very same games.** A feature that orders games marginally worse than the
price already does is a real feature and not a betting edge. See
[the player-Elo block](#the-player-elo-block-a-strong-result-that-was-a-leak).

#### A number in the committed output that looked good, and was a bug

An earlier revision reported **+11.76%** ROI for the model and **+13.54%** for model +
availability at a 0.55 threshold. Both were a bug, and the bug inverted the sign: the
stake is `p_model − p_raw`, which is **negative whenever the market is the more confident
side** — 1,382 of 4,076 games at that threshold — and the ROI divided total profit by
`stake.sum()`, which summed them. A loss of −6.48 over a denominator of −47.88 reported
as a profit. Corrected, every threshold is negative or flat: **−8.55%** and **−9.37%** at
0.55, −5.02% and −2.74% at 0.60, +1.76% and −0.06% at 0.65. Both the mean and the
bootstrap now restrict to placed bets, pinned by `Tests/test_roi_denominator.py`. Full
post-mortem: [docs/measurement_bugs.md](docs/measurement_bugs.md).

Every figure in this section is the pre-refactor 4,076-game measurement, kept as the
post-mortem it is; the re-measured sweeps are the ones reported above.

### Stint-level rotations: the one table nothing has read

`rotations/` held a single season because `GameRotation` died after 25 consecutive
errors. **It answers again** — `IN_TIME_REAL`, `OUT_TIME_REAL`, `PLAYER_PTS`, `PT_DIFF`,
`USG_PCT` per stint. The bank seeded into `data/nba/rotations/` was **47,736 stint rows
across 749 games** — 723 games of 2020-21 and 26 of 2015-16 — and the resumable
backfill (issues #23/#34) is running against it: **51,299 stint rows across 799 of
14,164 games** as of 2026-10-23, and growing. An earlier partial re-download had banked 38,410 rows across
602 games of 2020-21 (3.01 stints per player-game, `PT_DIFF` 99.9% populated) before
NBA began refusing with 60-second timeouts on every subsequent request; that count is
superseded, and kept as the record.

This matters more than any feature block, because `src/Models/player_ratings.py` states
that its plus-minus model *"needs `GameRotation` to build the design matrix"* and is
*"not true RAPM"*. **The RAPM null recorded here is therefore a null for a proxy, not for
RAPM**, and should be read that way until real stint data exists.

799 games is still about 6% of the dataset. A RAPM fitted on that would be thin, and
reporting it as a result would repeat the mistake this page keeps documenting. The
backfill is resumable and is tracked in issues #23/#34; the fix for the timeouts is
probably a longer `--sleep`.

The injury-report side of pre-game availability is being banked alongside it:
`Data/injury.sqlite` holds the league's official designations as append-only
evidence — **119 snapshots / 7,996 status events** as of 2026-10-23, with the live
capture loop running and the historical backfill in flight. Three-era URL handling,
point-in-time semantics, the two bugs caught before damage and the security audit are
in [docs/injury_pipeline.md](docs/injury_pipeline.md).

### Against other repos

`docs/repo_survey.md` compiles ~45 public NBA prediction repositories from their own
published READMEs, and is a useful external check for two reasons: it says what the
field's ceiling looks like, and it says how honestly the field reports.

**This project's accuracy sits inside the reported range, not above it.** Reported
moneyline accuracies in the survey: ~0.69 (whisdev), 0.669 (Pirkn, neural net), 0.6505
(Pirkn, ridge), 0.648 (Pirkn, XGBoost), 0.615 (cmunch1), 0.573 (Exidekat, combined).
This model's canonical accuracy is **0.6573** — mid-pack. The one repo that reports a
baseline alongside its own number, cmunch1, gets 0.615 against 0.580 for "home always
wins"; this project's equivalent comparison is 0.6573 against **0.5527**, a wider margin
than cmunch1's. So the model is not weak. It is ordinary, and being ordinary at 0.657 is
what the field looks like.

**Two repos in the survey say outright that they lose to the market** — tonyelhabr and
NBA_Betting — which is the same conclusion this project reached by measurement. The fact
that it is the minority view in the field is worth naming rather than hiding behind.

**The survey's §7 observation is the one that matters most.** No reported accuracy across
~45 repos accounts for the vig, typically 4–8% on a two-way market, and a 65%-accurate
model is roughly break-even only after deducting it. This project measures the vig
directly: **+4.13% mean overround** (p5–p95 2.96–5.84% across the closing-line join —
+4.03% full-join mean per `closing_line.json` vs `edge_betting.py`'s scored-subset
+4.13%) and a model-to-market gap of 0.02268 nats. So the accuracy
figures in the survey — including the 0.69 and the 0.669 —
are pre-vig numbers that cannot be read as profitability.

**Three architectures the survey names as winners were tested here and all three lose**
(Gaussian Naive Bayes + PCA, logistic on 20 principal components, ridge — see the
[experiment log](#the-experiment-log-17-pre-registered-runs)), which is consistent with
its §6.4 finding that feature volume does not track performance. Of ~45 repos surveyed,
one publishes a baseline; 13 of 28 game-outcome repos publish no metrics at all while one
29★ repo has no README whatsoever.

**What this project adds to that literature is not a better model.** It is a baseline, a
held-out protocol, a measured vig, and a comparison against the actual closing price.

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
> before use. See [Limitations](#limitations) and
> [How models were retrained and added](#how-models-were-retrained-and-added).

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

#### `/games` — the NBA slate

```bash
curl 'http://127.0.0.1:5000/games'
curl 'http://127.0.0.1:5000/games?date=2026-10-21&tz=America/New_York'
```

Server-rendered from `ScoreboardV3`, **one upstream call per 60-second window**
shared across all viewers — the endpoint returns 403 under burst, so the page never
polls from the browser. `?date=` defaults to today; `?tz=` takes any IANA zone and
falls back to a `viewer_tz` cookie, then to UTC.

Three upstream traps this route absorbs, all confirmed against live payloads: nothing
on `ScoreboardV3` says which team is home (only `gameCode`, `YYYYMMDD/AWAYHOME`,
carries it, and `LineScore` orders rows **home first**); `gameEt` is Eastern
wall-clock with a `Z` on it, so only `gameTimeUTC` is used; and `no_games` is a true
answer, not a failure, so a 403, a dead upstream and a partial payload each get their
own status and their own sentence. **Known gap:** the in-progress branch
(`gameStatus == 2`) is modelled from the schema and was **not observed live**. 65
tests cover the route without touching the network; the full write-up is in
[docs/flask_games_route.md](docs/flask_games_route.md).

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
**explicitly not ranked**. It also reports the **train-vs-test gap** for every run,
so an improvement bought by memorising the training folds cannot hide, and it prints
the **selection cost** of the winner: the gap between best and median, which is the
expected size of a maximum of N draws. See
[the experiment log](#the-experiment-log-17-pre-registered-runs).

`scripts/feature_block_ablation.py` does the same job for feature blocks, with the
paired bootstrap and the Bonferroni accounting — `--seed 42/137/2718` and
`--splits 8` are the replication, and see
[the ROI arm](#a-number-in-the-committed-output-that-looked-good-and-was-a-bug) for
`scripts/edge_holdout.py`, which asks whether a threshold chosen by looking at the
data survives being chosen somewhere else.

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

#### Current state: 1,287 passing + 104 subtests, 0 failing

40 files, measured individually — `test_injury_data.py` (110), `test_ratings.py` (106),
`test_training_layer.py` (94), `test_line_movement.py` (85), `test_metrics.py` (71),
and 35 more down to `test_kelly_criterion.py` (5). **Every guard was added because
something was measured to be wrong**, and several because a test caught a bug I
introduced in the same commit. The per-file table, what each one protects, and why
each exists is in [docs/test_suite.md](docs/test_suite.md) (written when the suite
was 35 files; `test_injury_data.py`, `test_leakage_gate.py` and three more have
joined since); the three largest post-mortems are
[the ROI denominator](docs/measurement_bugs.md) (a loss divided by a signed sum
reported as +13.54%), [the data guard](docs/measurement_bugs.md#2-the-data-guard-a-directory-that-exists-is-not-data-in-it)
(a directory that exists is not data in it — the one route past the check that
missed), and [the player-Elo leak](docs/player_elo_leak.md) (a |z| = 5.18 result
that was a function of the answer).

CI runs on Python 3.10 and 3.11, plus a matrix job that exercises the totals trainer
on both sides of the scikit-learn 1.7 `multi_class` removal.

---

## Limitations

**Read this before using any number from this repository.**

### About the results

1. **No demonstrated betting edge.** This is now a *measured loss* rather than an
   absence of a result, and the distinction matters. Betting wherever the model beats
   the posted price — a rule with no free parameter, considering both sides — returns
   **−8.23%** over 4,828 bets (−10.21% over 4,855 without the availability block), and
   no margin or threshold cell reaches |z| = 3 (strongest |z|: 1.91 margin sweep, 0.62 threshold sweep). Reproduce with
   `python scripts/edge_betting.py`; the market comparison is
   `python scripts/closing_line.py`.
2. **The model is worse than the market it is trying to beat**, and the gap is now
   measured rather than inferred from accuracy. On 5,788 games with both a prediction
   and a closing line: market log loss **0.59766** against model + availability's
   **0.62034**, a gap of 0.02268 with CI [0.01638, 0.02859] — the interval excludes
   zero on the market's side, |z| = 7.28.
3. **Accuracy is not profitability, and log loss is closer to profitability than
   accuracy is.** The market has both the best accuracy (0.6818) *and* the best log
   loss (0.59766); a model could beat it on one and still lose money on the other.
   This project's accuracy (0.6573) sits mid-pack among published NBA models
   (~0.615–0.69, `docs/repo_survey.md`), so being ordinary at 0.657 is the field's
   normal — and **no** published figure in that survey accounts for the vig, which
   this one measures directly at +4.13% (`python scripts/edge_betting.py`). Those
   accuracy numbers cannot be read as returns.
4. **The `68.9%` figure is retired.** It is reproduced here only to explain the leak.
   It is not a forecasting result.
5. **Effect sizes here are small, and the honest bar is high.** Everything measured
   is of order 0.001–0.03 nats. This project holds itself to |z| > 3, from
   Vandenbruaene et al. 2022 (*J Sports Econ* 23(7)), where the null of
   unprofitability was never rejected across 628 implementations after
   multiple-testing correction. A single 95% interval is not enough; **ten blocks
   are** now scored against one full frame, and the largest of them reaches |z| = 2.59,
   still short of the bar. The one effect that clears it is the availability block now
   in the canonical model (Δ +0.00390, |z| = 4.78 —
   [the one thing that is not a null](#the-one-thing-that-is-not-a-null)); everything
   else measured here is short of the bar or a leak.

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

- [The market anchor](docs/market_anchor.md) — H1a and H2: why odds are legitimate
  as an input here and only here, the walk-forward stacking protocol, the arm
  table, and the disagreement decile where the market wins 60.1% of head-to-head
  disagreements. The close absorbs everything our model knows, including
  availability.
- [The availability block](docs/availability_block.md) — the one mechanism that
  survived: expected minutes over the prior game's squad, three agreeing
  instruments, the membership-leak contrast that motivates the convention, the
  issue #36 wiring, and the honest boundary (it improves forecasting; it is fully
  priced at the close).
- [The injury-report pipeline](docs/injury_pipeline.md) — point-in-time
  availability as evidence: three-era URL handling, append-only snapshots with
  sha256 provenance, the two bugs caught before damage, the security audit, and
  how to run it.
- [The October 2026 experiment campaign](docs/experiment_log_2026-10.md) — every
  experiment of the month from declared rationale to verdict, the two errata, why
  each null closed its branch, and the round-3 reproducibility finding.
- [The player-Elo leak, in full](docs/player_elo_leak.md) — a |z| = 5.18 result that
  was target leakage, why truncating the future does not catch it, and the bucket
  analysis that shows the raw Elo total runs backwards.
- [Two measurement bugs](docs/measurement_bugs.md) — the ROI denominator that reported
  a loss as a profit, and the data guard that checked for a directory rather than for
  files.
- [The feature search](docs/feature_search.md) — four pre-registered candidates, the
  fold-count artifact that killed one, and the two ways a feature candidate can fool
  you.
- [What else could be modelled on this dataset](docs/modeling_exploration.md) — nine
  further candidate blocks built from tables nothing had read, all scored, all null.
- [Survey of ~45 public NBA prediction repos](docs/repo_survey.md) — the field's
  accuracy ceiling and how honestly it is reported.
- [The test suite](docs/test_suite.md) — what each file protects and the bug each
  guard was added for (the per-file table as written at 35 files).
- [The feature-order contract](docs/feature_order_contract.md) — how training and
  inference agree on column *order*, and the copy that drifted.
- [`/games`, the Flask route](docs/flask_games_route.md) — the three upstream traps it
  absorbs, all confirmed against live payloads.
- [ROADMAP.md](ROADMAP.md) — the levels between this working pipeline and a real
  forecasting or betting engine, with the measured case for why model tuning is not
  the bottleneck.
- `scripts/README.md` — what the evaluation harness measures and how it degrades.
