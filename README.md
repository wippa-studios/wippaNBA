# Legit NBA Predictions

A fork of [kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting)
that audits the original work, fixes the defects it found, and reports what the
model actually does when you bet it.

**The short version, before you read anything else:**

- The original project's headline **68.9% moneyline accuracy was caused by target
  leakage** and does not survive a corrected join. Leak-free accuracy is about
  **65.7%** against a **54.7%** home baseline in the held-out window.
- **There is no demonstrated betting edge, and the model is worse than the
  market.** The closing-line favourite wins **68.1%** against the model's 65.7%.
  The model's ROI at the closing line is **+1.5% to +8.3% depending on
  threshold, with standard errors of ±4.4 to ±9.4** — every estimate sits within
  roughly one standard error of zero, so the sign of the result depends on the
  window and the definition used. Nothing here distinguishes skill from noise.
- The totals (over/under) model is **no better than a coin flip** and does not beat
  always predicting "over".

Reproduce every figure below with `python scripts/evaluate.py`.

This fork is therefore a **data-pipeline and modelling project**, not a
profitable-betting system. See [Limitations](#limitations) before using any number
from this repo.

---

## Where this started

[`kyleskom/NBA-Machine-Learning-Sports-Betting`](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting)
began on **2019-11-29** as a personal NBA statistics project and grew into one of
the most-starred sports-analytics repositories on GitHub (**1,744 stars, 569
forks**, 142 commits). It scrapes daily NBA team stats, joins them with sportsbook
odds, trains XGBoost / Keras / logistic-regression models for moneyline and
totals, and prints expected value and Kelly Criterion stake sizes for the day's
games. It also ships a Flask UI for browsing the output.

It is a genuinely substantial piece of work, and it was clearly maintained in good
faith — commits run from "Initial commit" through a substantial
[`Major Overhaul (#502)`](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting/commit/8e36b0b72ecc5082aa8976dff742cdf910b3aa36)
in January 2026. The defects documented below are the ordinary accumulation of
years of iteration by a single maintainer, not carelessness, and most of them are
invisible without measuring rather than reading.

This fork exists because the project's central number — **68.9%** — never appears
in the original README as prose. It exists only in a screenshot
(`Screenshots/output.png`) and in model filenames
(`Models/XGBoost_Models/XGBoost_68.9%_ML_*.json`). An unsourced, undated,
unqualified headline accuracy figure for a product whose entire purpose is placing
bets is the single most consequential documentation gap in the project, and it is
what prompted the audit.

**Attribution:** all original code, data pipelines, and model scripts are the work
of Kyle Skompinski and contributors. This fork makes corrections; it does not
replace the original effort. There is **no LICENSE file in the upstream project**,
which means default copyright applies and no rights to reuse the code are granted
by upstream. This fork is offered for study and comment.

---

## What the audit found

Every number below is reproducible from the committed `Data/*.sqlite` files.
Method and caveats are stated with each claim rather than buried.

### 1. Target leakage — the 68.9% is not a forecasting result

`config.toml` scrapes NBA season-to-date stats with `DateTo` set to **the game
date itself**. The NBA treats `DateTo` as inclusive, so the stats snapshot named
`D` contains the result of the game played on `D`. `Create_Games.py` then joined
that snapshot to the very game it was supposed to predict, so `W`, `L`, `GP`,
`W_PCT`, `PTS`, `PLUS_MINUS` and every `*_RANK` column carried the answer.

A rule using only the contaminated columns recovers the label almost perfectly:

```
pred = 1 if (W_home[D] - W_home[D-1]) > (W_away[D] - W_away[D-1]) else 0
```

When each team's `W` on `D` includes that night's result, the increments encode
the answer outright. Measured on the dataset as shipped versus the dataset this
fork rebuilds, by calendar year:

| year | shipped `dataset_2012-26` | rebuilt | rows |
|---|---|---|---|
| 2013 | 41.9% | 45.6% | 1,310 |
| 2020 | 47.1% | 49.2% | 711 |
| 2023 | 44.0% | 48.9% | 1,259 |
| **2024** | **68.0%** | 48.1% | 1,196 |
| **2025** | **96.1%** | 45.6% | 1,330 |
| **2026** | **100.0%** | 38.5% | 52 |
| all | 49.1% | **48.4%** | 17,166 |

Chance is 50%. The clean years sit slightly *below* 50% because with no leak both
teams' `W` are unchanged, so the rule degenerates into always predicting the away
win — 42% is simply the away-win rate. That degenerate signature is what "no
leakage" looks like.

The leak is **not uniform**, which is what makes it nastier. Stats for 2012–2021
were scraped the morning of `D`, before tip-off, so `DateTo=D` returned data
through `D−1` and they were genuinely pre-game. Recent seasons were backfilled
after the fact and are poisoned. The chronological train/test split therefore
puts mostly-clean data in training and almost entirely poisoned data in the test
set — 3.5% leaked in train/validation versus **94.3% in the held-out 10%**.
`Data/dataset.sqlite` in this fork has been rebuilt through the corrected join.

**Measured impact**, rebuilding every row against the correctly-aligned `D−1`
snapshot and re-running an identical pipeline on identical rows:

| | accuracy | log loss | ROI of the model's own `EV>0` bets |
|---|---|---|---|
| as shipped (94% of test rows leaked) | 0.675 | 0.613 | **+4.21%** |
| correctly aligned (same rows) | 0.647 | 0.632 | **−1.86%** |

Leakage was worth roughly **+2.8 points of accuracy and +6 points of phantom ROI**.

### 2. The model does not beat the market

The original project had no market baseline anywhere. On the **rebuilt, leak-free**
dataset, `scripts/evaluate.py` reports over 1,722 held-out games:

| strategy | accuracy | flat-bet ROI | standard error |
|---|---|---|---|
| model (logistic regression) | 0.657 | +1.5% (edge ≥ 0.00) | ±4.4 |
| model (edge ≥ 0.05) | 0.657 | +5.3% | ±6.0 |
| model (edge ≥ 0.10) | 0.657 | +8.3% | ±9.4 |
| **closing-line favourite** | **0.681** | **−5.46%** | ±1.62 |
| always bet home | 0.547 | — | — |

Model log loss 0.618, Brier 0.215, AUC 0.709. All 1,722 test rows carry a closing
price, so no figure below rests on a partial join.

Two things to read carefully here.

**The market beats the model outright.** The favourite wins 68.1% where the model
wins 65.7%. That is the clearest single result in this document.

**The model's ROI is indistinguishable from zero.** The point estimates run from
+1.5% to +8.3%, but the standard errors run from ±4.4 to ±9.4 — every threshold
sits inside roughly one standard error of zero, and the sample shrinks from 581
bets to 211 as the threshold rises. Earlier measurements on this same data
produced negative figures (−1.9% to −7%); the differences are definitional
(de-vigged edge versus raw `EV>0`, calibrated versus uncalibrated probabilities)
and window dependent. **The honest conclusion is not "the model loses money" — it
is that the sign of the result is not stable, so no edge is demonstrated in
either direction.** A 550-row window returns +25% to +48%; the full window returns
+1% to +8%. That is variance, not skill.

By contrast the favourite's −5.46% ±1.62 **is** statistically solid — it is the
bookmaker's vig, which is what an efficient market is supposed to charge.

### 3. The totals model has no signal

`OU-Cover` is a three-class target (under / over / push) on a 49.5 / 49.4 / 1.1
split, and `compute_sample_weights` up-weights the rare push class by ~30×. On the
held-out split the shipped totals model scores **0.497** three-class accuracy
against **0.504** for always predicting "over", and a worse log loss than simply
predicting the class prior. The reported ~50.1% is a coin flip.

### 4. Probability calibration was actively harmful

Inverse-frequency class weighting pushed the model's mean home probability ~6
points off the true base rate (0.5079 vs a true 0.5694), while the sigmoid
calibrator was fit on **unweighted** labels — so it was correcting a distortion it
had never been validated against. Expected value is computed directly off those
probabilities. Separately, the calibration set was used simultaneously as the
early-stopping set, the calibrator's fit set, and a subset of the final CV
validation fold.

### 5. The web app leaked a live API key

`Flask/app.py` contained a hardcoded third-party RapidAPI key on two lines,
committed since 2023 and reachable through the unauthenticated `/player-stats` and
`/team-data` routes. The README also instructed users to run `flask --debug run`,
whose Werkzeug debugger serves application source — including that key — to any
unauthenticated visitor on any unhandled exception.

### 6. Engineering gaps that let all of this through

- **No CI, no linting, no type checking** of any kind.
- `pytest Tests/` and `unittest discover` both **failed** on a clean checkout.
- **74% of production code** lived in modules no test imported; `main.py` and all
  six training scripts were at 0% coverage.
- **255 games (1.5%)** were silently dropped from the dataset by a `len(rows) != 30`
  guard, unrepairable because backfill treats any table as "present".
- **9 postponed games and 1 phantom 190–190 tie** were labelled as home-team losses.
- The README documented commands that **silently exit 0** while doing nothing
  (`--dataset dataset_2012-26_new` — a table that does not exist).
- `cd src/Process-Data && python -m Create_Games` raises `ModuleNotFoundError`.
- **Three mutually incompatible days-rest formulas** across training and inference.

---

## What this fork changes

**Correctness**
- `Create_Games` joins each game to the most recent stats snapshot **strictly
  before** the game date, eliminating the leak at its source.
- Postponed, zero-score and phantom-tie games are excluded instead of being
  labelled as losses.
- Days-rest is computed from **date-only UTC values against the team's previous
  game**, in one shared implementation — replacing three divergent ones that
  included the current game and varied with the runner's timezone.
- The UO model receives its `OU` column at the position it was trained on. It was
  being appended, so XGBoost silently scored days-rest as the totals line.
- A shared `src/Utils/features.py` defines the feature contract once, and both
  runners now **assert** the inference frame matches the training column order —
  a right-columns-wrong-order frame is rejected rather than scored.

**Model evaluation**
- Class weighting is opt-in and defaults to **off**; probabilities need to be
  calibrated, not balanced.
- Train / early-stop / calibration / test are four disjoint chronological segments.
- Each trainer now persists its ordered feature list alongside the model.
- Reported metrics include **log loss, the closing-line favourite baseline, and
  realised ROI** — not accuracy alone.

**Security & operations**
- The API key is read from `RAPIDAPI_KEY`; the endpoints fail closed with a 503.
- Debug mode is gone; unhandled exceptions return a generic response, logged
  server-side.
- Every `requests.get` and `subprocess` call has a timeout; the per-request
  triple-subprocess spawn is single-flighted and semaphore-capped.
- `player_id` is validated; raw exception text is no longer returned to clients.
- Upstream fetch failures raise a typed error instead of being swallowed into an
  empty DataFrame three frames downstream.

**Engineering**
- CI on Python 3.10–3.11 running both `pytest` and `unittest`.
- `pyproject.toml`, a working `conftest.py`, and a declared Python ceiling.
- `requests`, `numpy` and `joblib` are declared instead of resolving transitively.
- Odds-date holes are backfilled; missing-season runs exit non-zero.
- SQLite connections are closed; `Add_Days_Rest.py` refuses to overwrite a table
  it could not parse.

---

## Requirements

Python **3.10 or 3.11**. The ceiling is deliberate: `tensorflow==2.14.0` has no
wheels for 3.12+, and it cannot be bumped without a Keras 3 migration that also
invalidates the committed `.keras` artifacts. `requirements.txt` documents this in
full.

```bash
pip install -r requirements.txt
```

## Quick start

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

## Data pipeline and training

Run from the repository root:

```bash
python -m src.Process-Data.Get_Data          # scrape daily team stats
python -m src.Process-Data.Get_Odds_Data      # scrape sportsbook odds + results
python -m src.Process-Data.Create_Games       # build the training dataset

python -m src.Train-Models.XGBoost_Model_ML   --dataset dataset_2012-26 --trials 100 --splits 5
python -m src.Train-Models.XGBoost_Model_UO   --dataset dataset_2012-26 --trials 100 --splits 5
```

Fill gaps with `--backfill` (optionally `--season 2025-26`).

> The `Process-Data` and `Train-Models` directory names contain hyphens and are not
> importable as Python packages; they are invoked as paths. Renaming them is
> tracked as follow-up work.

## Tests

```bash
pytest Tests/                       # or: python -m unittest discover -s Tests -t .
```

## Flask app

```bash
export RAPIDAPI_KEY="<your rapidapi key>"   # optional; team/player endpoints 503 without it
cd Flask && flask run
```

Do **not** run `flask --debug run` on any reachable host: debug mode serves
application source, including any credentials, to unauthenticated visitors.

---

## Limitations

Read this before using any number from this repository.

1. **No demonstrated betting edge.** The model's ROI at the closing line is
   +1.5% to +8.3% with standard errors of ±4.4 to ±9.4 — inside one standard
   error of zero at every threshold, with the sign depending on the evaluation
   window. Treat it as no result, not as a profit and not as a loss. The
   measurements are reproducible with `python scripts/evaluate.py`.
2. **The model is worse than the market it is trying to beat.** The closing-line
   favourite wins 68.1% where the model wins 65.7%.
3. **Accuracy is not profitability.** 65.7% accuracy loses money when the market
   prices that accuracy into the line. Only ROI and closing-line value measure
   this product.
4. **The `68.9%` figure is retired.** It is reproduced here only to explain the
   leak. It is not a forecasting result.
5. **The committed model artifacts predate every fix in this fork and must be
   retrained before use.** They were fitted on the leaked dataset above, and
   `Models/NN_Models/Trained-Model-OU-49.2_*.keras` additionally carries the old
   totals column order — it has 107 inputs either way, so nothing will raise, it
   will simply be wrong. Treat everything in `Models/` as untrusted until
   retrained.
6. **The totals model should not be used.** It does not beat a coin flip or a
   constant prior.
7. **Small test set.** ~1.7k held-out rows spanning ~1.2 seasons. Too small to
   resolve a 1–2 point edge, with no way to re-test on new data.
8. **Kelly sizing is full Kelly** with no fractional scaling, bankroll cap,
   simultaneous-exposure limit, or discount for model uncertainty. Full Kelly on a
   marginally-uncertain edge has a high probability of severe drawdown.
9. **Not financial advice.** Nothing here is a recommendation to wager. Sports
   betting carries real financial risk; this is an educational data-science project.
10. **Data provenance.** Team stats come from `stats.nba.com` (whose Terms of Use
   restrict redistribution) and odds from a scraped source. Neither is licensed
   for redistribution by this project.
11. **No license.** Neither this fork nor its upstream grants reuse rights.

## Credit

Original project by [Kyle Skompinski](https://github.com/kyleskom) and
contributors. Corrections, audit and documentation in this fork are ours.
