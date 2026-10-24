# The audit and the fixes

> What the original repository was, what was measured about it, the six defects found, what this fork changed, and the tooling added. Part of the [Legit NBA Predictions README](../README.md).

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
validation](results_pipeline_a.md#compared-with-the-original-model) — but the two are not a like-for-like
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
[the player-Elo block](results_pipeline_a.md#the-player-elo-block-a-strong-result-that-was-a-leak). The
one that clears it honestly is the availability block now in the canonical model
(|z| = 4.78); see
[the one thing that is not a null](results_pipeline_a.md#the-one-thing-that-is-not-a-null).

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
[Pipeline B results](results_pipeline_b.md#pipeline-b-the-model-against-the-market)), the closing-line
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
next deliberate step, tracked in [ROADMAP.md](../ROADMAP.md).

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
