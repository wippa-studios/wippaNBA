<div align="center">

# Legit NBA Predictions

**NBA outcome prediction, measured honestly — as a research project, a betting project, and a data-analytics project in one repository.**

[![CI](https://github.com/wippa-studios/Legit-NBA-Predictions/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/wippa-studios/Legit-NBA-Predictions/actions/workflows/ci.yml?query=branch%3Amaster)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11-3776AB.svg)](https://github.com/wippa-studios/Legit-NBA-Predictions/actions/workflows/ci.yml)
[![Data](https://img.shields.io/badge/data-nba_api-13%2C199%20games-informational.svg)](https://github.com/swar/nba_api)
[![No betting edge claimed](https://img.shields.io/badge/prediction%20quality%20only%20%E2%80%94%20no%20betting%20edge%20claimed-lightgrey.svg)](results_pipeline_b.md)

[Research](#research) ·
[Betting](#betting) ·
[Data &amp; engineering](#data-engineering) ·
[Key findings](#key-findings) ·
[Quick start](#quick-start) ·
[Docs](#docs) ·
[Limitations](#limitations)

</div>

> **There is no demonstrated betting edge here, and we do not claim one.** This is an audit and a set of prediction models, judged on how well they predict — not on whether they make money (they lose, and we measure that). A fork of [kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting) that corrects its defects and rebuilds the modelling layer from NBA's own data; the original project is by Kyle Skompinski and contributors, and this fork corrects rather than replaces.

---

## One repository, three identities

The same project is genuinely read three ways, and each way has different evidence attached. Use the section that matches what you are here for.

### 🔬 The research project {#research}

Pre-registered experiments, an append-only ledger, and a high bar for calling anything real. **Every candidate feature and model is measured against the same frame; the project's bar is |z| > 3** (Vandenbruaene et al. 2022, *J Sports Econ* 23(7)).

- **Two effects clear the bar, and both are about who plays:** player-level *expected minutes* (availability) improves log loss by **+0.00390 nats, |z| 4.78**; the same-game injury multiplier on top of it clears a **forward two-season holdout at +0.003579, |z| 3.85**. The closing line already prices the availability block in.
- **Two headline "wins" were leaks and are documented as such:** season-to-date stats that contained the game's own result (the famous 68.9%), and a per-player Elo block that summed over the lineup which *actually* played (|z| 5.18 → +0.00065 on a pre-game roster).
- **The simple model wins:** an L2 logistic regression (0.6157 log loss) beats XGBoost, HistGradientBoosting, random forest, Gaussian NB, a five-seed neural net and margin/total models. Every run — including the nulls — is in `results/experiments.jsonl`.
- **The field agrees:** ~45 public NBA repos top out at 65–69% accuracy; the external literature lands at roughly this project's log loss and loses betting too. Quotes and links in [the research-notes page](docs/research_links.md).

**Start here:** [Results: Pipeline A](results_pipeline_a.md) · [the October 2026 experiment log](docs/experiment_log_2026-10.md) · [reading the results honestly](docs/reading_the_results.md) · [the reusable leakage gate](docs/leakage_gate.md).

### 🎰 The betting project {#betting}

The same models, scored against the closing line and the posted price — the audit of an original betting pipeline, plus a rebuilt one that cannot claim an edge.

- **The market beats the model decisively:** on 5,788 games with both a prediction and a closing line, market log loss **0.59766** vs model + availability **0.62034** — a gap of 0.02268 nats, interval entirely on the market's side (|z| 7.28).
- **Selective betting does not rescue it:** bet wherever the model beats the posted price (both sides, no free parameters) → **−8.23%** over 4,828 bets; no margin/threshold cell reaches |z| 3.
- **The one real model edge is already in the price:** availability adds +0.00390 nats to the model, then *nothing* after conditioning on the close (tight null at the market-anchored stack). The injury multiplier is measured as a model feature, not at the close.
- **The original pipeline's headline was measured leakage:** the audit found and fixed six defects, including a 68.9% accuracy and ROI inflation from target leakage, a leaked API key, and a totals model with no signal.

**Start here:** [Results: Pipeline B](results_pipeline_b.md) · [why odds are legitimate here](docs/market_anchor.md) · [the totals target](docs/totals_target.md) · [the full audit](audit_and_fixes.md) · [roadmap to a real forecasting engine](ROADMAP.md).

### 🧮 The data &amp; engineering project {#data-engineering}

A single reproducible feature layer over NBA's own data, and the tooling that keeps its numbers trustworthy.

- **13,199 games, one shared design:** `scripts/download_nba_data.py` builds 310 CSVs (10+ seasons) and `src/DataProviders/nba_data.py` assembles **26 causal feature columns** used by every model. No model has a second loader.
- **Causality is tested, not asserted:** deleting 5,199 future games changes no historical feature by more than `0.000e+00`; a reusable leakage gate separates confirmed future-dated leaks from same-date dependencies.
- **Append-only evidence:** the injury-report store holds **2,046 snapshots across 1,037 report dates / 237,436 designation events** (official pre-tip injury reports, point-in-time semantics).
- **1,464 tests, 0 failing, 104 subtests** across 40 files; CI on Python 3.10 & 3.11 and both sides of the scikit-learn 1.7 `multi_class` removal.

**Start here:** [Models, features and retraining](models_and_training.md) · [the availability block](docs/availability_block.md) · [the injury-report pipeline](docs/injury_pipeline.md) · [the feature search](docs/feature_search.md) · [reproducing every number](reproducing.md).

---

## Key findings

| | |
|---|---|
| **The 68.9% headline was target leakage** | Season-to-date stats were scraped with the game's own result included. Rebuilt without the leak, accuracy is ~48% on the affected rows. |
| **The market beats the model decisively** | Market log loss **0.59766** vs model + availability **0.62034** on 5,788 games. Gap 0.02268 nats, \|z\| 7.28 — a loss, not a tie. |
| **Selective betting does not rescue it** | Bet-where-the-model-beats-the-price returns **−8.23%** over 4,828 bets; no margin or threshold cell reaches \|z\| 3. |
| **Who plays is what works** | Player-level *expected minutes* (availability) improves log loss **+0.00390** at \|z\| 4.78; the pre-tip injury multiplier on top of it clears a forward two-season holdout at **+0.003579, \|z\| 3.85**. Availability is priced out at the close; the multiplier is a model result, not a measured bet. |
| **The biggest-looking win was a leak** | A per-player Elo block measured \|z\| 5.18 and was leaking the real lineup; on a pre-game roster it is +0.00065. |
| **The simplest sensible model wins** | Logistic regression (0.6157) leads XGBoost, HistGradientBoosting, random forest, Gaussian NB, a five-seed neural network and margin/total models. |
| **The close absorbs everything our model knows** | Price + model loses to price alone (0.00177 nats, \|z\| 3.15); availability beyond price is a tight null. |
| **Do not use the committed model files** | They were trained on the leaked dataset. Retrain before using anything. |

---

## The two pipelines

| | **A. NBA-data models** (new) | **B. Original betting pipeline** (fixed) |
|---|---|---|
| **Purpose** | Predict outcomes; measure how good the predictions are | Predict games against sportsbook odds, with Kelly sizing |
| **Data** | NBA's own data via `nba_api` (310 CSVs, 13,199 games) | Scraped team stats and sportsbook odds |
| **Models** | Elo, logistic regression, margin/total, boosting, MLP | XGBoost and neural networks |
| **Judged on** | Log loss, Brier, AUC, calibration | Accuracy, log loss, market baseline, ROI |
| **Entry points** | `scripts/download_nba_data.py`, `python -m src.Models.*` | `main.py`, `src/Process-Data/`, `src/Train-Models/` |

Every feature for a game is computed **only from games that already happened**; the leakage gate tests it.

---

## Quick start

Python **3.10 or 3.11** (the ceiling is deliberate: `tensorflow==2.14.0` has no wheels for 3.12+).

```bash
pip install -r requirements.txt

python scripts/evaluate.py                 # reproduce every number; opens with a leakage gate
python scripts/download_nba_data.py        # resumable; core = 10+ seasons / 13,199 games
python -m src.Models.logreg_rolling        # the canonical model (Pipeline A)
python scripts/closing_line.py             # does any of this beat the closing line? (it doesn't)
```

> ⚠️ The model files committed in `Models/` predate every fix in this fork and were trained on the leaked dataset. **Retrain before using anything** — see [Models, features and retraining](models_and_training.md).

---

## Docs

**Research** — the measurement discipline and what it found

| doc | what it covers |
|---|---|
| [Results: Pipeline A](results_pipeline_a.md) | Model comparison, feature blocks, the one thing that is not a null (and its injury add-on) |
| [The October 2026 experiment campaign](docs/experiment_log_2026-10.md) | Every experiment from declared rationale to verdict, plus errata |
| [Reading the results](docs/reading_the_results.md) | Why the ordering below the top row is the wrong thing to read |
| [The reusable leakage gate](docs/leakage_gate.md) | One function, three probes, the three known leaks as fixtures |
| [The player-Elo leak](docs/player_elo_leak.md) | A \|z\| = 5.18 result that was target leakage |
| [Two measurement bugs](docs/measurement_bugs.md) | The ROI denominator and the data guard |
| [Survey of ~45 public NBA repos](docs/repo_survey.md) | The field's accuracy ceiling, and how honestly it is reported |
| [The research links, assessed](docs/research_links.md) | 38 external papers/posts triaged against the measured results |

**Betting** — the market, the rules, and the audit of the original project

| doc | what it covers |
|---|---|
| [Results: Pipeline B](results_pipeline_b.md) | Closing-line comparison, betting rules, the repo survey |
| [The market anchor](docs/market_anchor.md) | Why odds are legitimate as an input only here; the close absorbs everything |
| [The totals target](docs/totals_target.md) | Why `OU-Cover` keeps its push class; the closing price that would justify folding it |
| [The audit and the fixes](audit_and_fixes.md) | The six defects found in the original, and everything this fork changed |
| [ROADMAP](ROADMAP.md) | The levels between this pipeline and a real forecasting engine |

**Data &amp; engineering** — the feature layer, the evidence store, and the tooling

| doc | what it covers |
|---|---|
| [Models, features and retraining](models_and_training.md) | Model modules, `load_design`, retraining commands, the feature-order contract |
| [The availability block](docs/availability_block.md) | The mechanism that survived: expected minutes over the prior game's squad (the membership convention is load-bearing), plus the pre-tip injury multiplier |
| [The injury-report pipeline](docs/injury_pipeline.md) | Point-in-time availability as append-only evidence, the three pre-2020 grammars, and the backfill policy |
| [Late scratches](docs/late_scratch.md) | What a designation is worth by time to tip: Questionable is a coin flip, Available is not the safest state |
| [The feature search](docs/feature_search.md) | Four pre-registered candidates and the fold-count artifact |
| [Modelling exploration](docs/modeling_exploration.md) | Nine further candidate blocks, all scored, all null |
| [Usage](usage.md) | Full commands for both pipelines and the Flask `/games` route |
| [Verifying and testing](reproducing.md) | Reproducing every number, the experiment-log CLI, the test suite |
| [The feature-order contract](docs/feature_order_contract.md) | How training and inference agree on column order |
| [The test suite](docs/test_suite.md) | What each test file protects |
| [`/games` Flask route](docs/flask_games_route.md) | The three upstream traps it absorbs |

---

## Limitations

1. **No demonstrated betting edge — this is a measured loss.** Betting wherever the model beats the posted price returns **−8.23%** over 4,828 bets; no margin or threshold cell reaches \|z\| 3. Reproduce with `python scripts/edge_betting.py`.
2. **The model is worse than the market.** On 5,788 games with both a prediction and a closing line: market **0.59766** vs model + availability **0.62034** (gap 0.02268, CI [0.01638, 0.02859], \|z\| 7.28). Accuracy is not profitability; this project's 0.6573 is the field's normal – and no published figure in the survey accounts for the vig that this one measures at +4.13%.
3. **Effect sizes here are small and the honest bar is high.** Everything measured is of order 0.001–0.03 nats; the effects that clear |z| > 3 are availability and the pre-tip injury multiplier on top of it, and availability is priced in at the close (see [Pipeline A](results_pipeline_a.md)).
4. **The committed model artifacts must be retrained.** They were fitted on the leaked dataset; the neural artifact also carries a stale totals column order. The totals model should not be used.
5. **Kelly sizing here is full Kelly** with no fractional scaling or drawdown control.
6. **Legal:** not financial advice; upstream data is not licensed for redistribution; neither this fork nor its upstream grants reuse rights.

---

## Credit and licence

Original project: [kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting) by Kyle Skompinski and contributors ([github.com/kyleskom](https://github.com/kyleskom)). This fork's audit, corrections, rebuilt modelling layer, measurements and documentation are ours; it corrects rather than replaces. Neither fork currently grants reuse rights (see [Limitations](#limitations)). Contributing: see [CONTRIBUTING.md](CONTRIBUTING.md) — start with [Models, features and retraining](models_and_training.md).