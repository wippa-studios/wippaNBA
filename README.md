# Legit NBA Predictions

A fork of [kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting) —
1,744 stars, 569 forks, maintained since 2019.

This fork does two things: it **fixes defects it found in the original**, and it
**rebuilds the modelling layer from NBA's own data** so the results can be read
against each other honestly.

**There is no demonstrated betting edge here, and we do not claim one.** What
follows is an audit and a set of prediction models, judged on how well they
predict — not on whether they make money.

> **Credit:** all original code, pipelines and model scripts are the work of
> **Kyle Skompinski and contributors**. Kyle built a substantial, well-maintained
> project and did it in good faith. The defects below are the ordinary
> accumulation of years of iteration by one maintainer — most are invisible
> without measuring rather than reading. This fork corrects; it does not replace.

---

## What this does

Predicts the outcome of NBA games, then measures how good those predictions
actually are.

```
  NBA's own data          Features                 Models            Judgement
  ──────────────          ────────                 ──────            ────────
  nba_api pull  ──────►   20 rolling    ──────►   Elo            ──────►  log loss
  310 CSVs                team metrics            logistic          Brier
  11 seasons              rest days               margin/total      AUC
  13,199 games            back-to-backs           gradient          calibration
                          3-in-4                  boosting
                          travel distance
```

**1. Pull the data.** `download_nba_data.py` fetches game logs, per-game team and
player measures, schedules and season-end snapshots from `stats.nba.com`. It is
resumable, and it drops `*_RANK` columns because those encode rank as of download
time rather than as of any game.

```bash
python download_nba_data.py                          # core, 11 seasons
python download_nba_data.py --stages core,extras     # + tracking, synergy, shot charts
```

**2. Build features.** `src/DataProviders/nba_data.py` is shared by every model,
so their numbers are comparable. Each feature for a game is computed **only from
games that already happened**. That is tested rather than asserted: deleting 5,199
future games changes no historical feature by more than `0.000e+00`.

**3. Split by date, walk forward.** Train always precedes test, and never
overlaps. Accuracy alone is useless here — always predicting "home" scores 55% —
so log loss leads, followed by Brier, AUC and a calibration table.

**4. Run the models.** Each is standalone and prints its own results:

```bash
python -m src.Models.baselines       # the bar everything else must beat
python -m src.Models.logreg_rolling
python -m src.Models.margin_total
python -m src.Models.gbm
```

---

## Results

All models below are scored on the **same 13,199 games** with the **same five
walk-forward splits**, so they are directly comparable to each other. Lower log
loss and Brier are better; higher AUC is better.

| model | log loss ↓ | Brier ↓ | AUC ↑ | accuracy |
|---|---|---|---|---|
| always predict home | 0.6883 | 0.2476 | 0.5010 | 0.5527 |
| Elo (K=10, tuned on validation) | 0.6260 | 0.2180 | 0.7020 | 0.6512 |
| Bradley-Terry | 0.6929 | 0.2494 | 0.5514 | 0.5568 |
| log-linear on Elo | 0.6234 | 0.2169 | 0.7026 | 0.6542 |
| margin + total → Φ(m̂/σ) | 0.6204 | 0.2156 | 0.7074 | — |
| **logistic regression** | **0.6193** | **0.2150** | **0.7089** | **0.6570** |
| XGBoost | 0.6233 | 0.2167 | 0.7030 | 0.6562 |
| HistGradientBoosting | 0.6236 | 0.2169 | 0.7026 | 0.6526 |

**What the table actually says.** Three results matter more than the ranking.

**Boosting loses, and the confidence interval proves it.** XGBoost trails logistic
regression by 0.0039 log loss, CI [−0.0056, −0.0023] — excludes zero. The two GBMs
agree with each other to within 0.0004, so it is not an implementation artifact.
The GBMs do best with *one* feature and never reach the linear model.

**More complex targets do not pay.** Predicting margin and total, then converting
with Φ(m̂/σ), reaches 0.6204 — no better than fitting the label directly on the same
features (Δ +0.0006, CI [−0.0026, +0.0038]). Its value is the richer output: margin
RMSE 14.01, total RMSE 18.78, and spread MAE 10.99 against 12.44 for a
pick-no-edge line.

**Batch rating systems lose to the base rate.** Bradley-Terry (0.6929) and SRS
(0.6890) both score *worse* than always predicting home (0.6883), with CIs
straddling zero. The cause is structural: a batch fit over five seasons reports
where a team **was**, while the next game depends on where it **is**.

### Compared with the original model

| | data | rows | validation | log loss | AUC |
|---|---|---|---|---|---|
| original logistic regression | scraped odds | 17,356 | chronological 90/10 | 0.6189 | 0.7084 |
| logistic regression (this fork) | NBA game logs | 13,199 | 5× walk-forward | 0.6193 | 0.7089 |

**These two numbers are not a like-for-like comparison, and the near-identical
values are a coincidence rather than a result.** Different games, different
coverage, different seasons, and different split geometry. What can be said is
narrower and still worth saying: a linear model on team strength is hard to beat
on this task, whether the features come from a scrape or from NBA directly, and
the models built on top of it did not beat it here.

---

## What the audit found

Six defects, all measured on the committed data. Reproduce with
`python scripts/evaluate.py`.

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

The original had no market baseline anywhere. On rebuilt, leak-free data over
1,722 held-out games:

| strategy | accuracy | flat-bet ROI | SE |
|---|---|---|---|
| model | 0.657 | +1.5% to +8.3% | ±4.4 to ±9.4 |
| **closing-line favourite** | **0.681** | **−5.46%** | ±1.62 |
| always bet home | 0.547 | — | — |

The market wins outright: 68.1% against 65.7%. The model's own ROI is
**indistinguishable from zero** — every threshold sits within about one standard
error, earlier measurements came out negative, and the sign depends on the window.
The honest conclusion is not "the model loses money"; it is that **no edge is
demonstrated in either direction.** The favourite's −5.46% ±1.62 *is* solid — that
is the vig, which is what an efficient market should charge.

### 3. The totals model has no signal

`OU-Cover` is three-class (under / over / push) on a 49.5 / 49.4 / 1.1 split, with
the rare push class up-weighted ~30×. It scores **0.497** against **0.504** for
always predicting "over", and a worse log loss than the class prior.

### 4. Calibration was actively harmful

Inverse-frequency class weighting pushed mean home probability ~6 points off the
true base rate (0.5079 vs 0.5694) while the sigmoid calibrator was fit on
**unweighted** labels — correcting a distortion it had never been validated
against. Expected value is computed straight off those probabilities. The
calibration set was also used as the early-stopping set, the calibrator's fit set,
and part of the final validation fold.

### 5. The web app leaked a live API key

A hardcoded RapidAPI key sat in `Flask/app.py`, committed since 2023, reachable
through two unauthenticated routes. The README also told users to run
`flask --debug run`, whose Werkzeug debugger serves application source — the key
included — on any unhandled exception.

### 6. Engineering gaps

No CI, no linting, no type checking. `pytest` and `unittest` both failed on a clean
checkout. 74% of production code was in modules no test imported; the entry
points were at 0% coverage. 255 games were silently dropped by a `len(rows) != 30`
guard. Nine postponed games and one phantom 190–190 tie were labelled as losses.
Documented commands silently exited 0 while doing nothing. Three mutually
incompatible days-rest formulas existed across training and inference.

---

## What this fork changes

**Correctness**
- `Create_Games` joins each game to the most recent stats snapshot **strictly
  before** the game date, eliminating the leak at its source.
- Postponed, zero-score and phantom-tie games are excluded instead of labelled
  as losses.
- Days-rest is computed from **date-only UTC values against the team's previous
  game**, in one shared implementation — replacing three divergent ones that
  included the current game.
- The UO model receives its `OU` column where it was trained on. It was being
  appended, so XGBoost silently scored days-rest as the totals line.
- A shared feature contract makes both runners **assert** the inference frame
  matches training column order, so right-columns-wrong-order is rejected.

**Model evaluation**
- Class weighting is opt-in and defaults to **off**; probabilities need to be
  calibrated, not balanced.
- Train / early-stop / calibration / test are four disjoint chronological segments.
- Every trainer persists its ordered feature list alongside the model.
- Reported metrics include **log loss, a market baseline, and realised ROI** — not
  accuracy alone.
- The hyperparameter search was ranking on noise: it reseeded per trial, so each
  trial ranked configurations differently rather than finding a better one. Seed is
  now fixed, and the expected winner's curse is printed next to every headline.
- Imputation is fitted on train inside each fold. It was fitted before the split,
  and 14 of 106 columns have a median that moves more than 1% as a result.

**Security & operations**
- The API key comes from `RAPIDAPI_KEY`; endpoints fail closed with a 503.
- Debug mode is gone; unhandled exceptions return a generic response, logged
  server-side.
- Every `requests.get` and `subprocess` call has a timeout; the per-request
  triple-subprocess spawn is single-flighted and semaphore-capped.
- Upstream failures raise a typed error instead of becoming an empty DataFrame
  three frames downstream.

**Engineering**
- CI on Python 3.10–3.11 running both `pytest` and `unittest`.
- Six near-identical training scripts collapsed to shared modules (2255 → 1551
  lines) with byte-identical output.
- Six trainers disagreed about which dataset table to train on; they now share one
  name, resolved through a function that refuses same-day-join builds by name.

---

<details>
<summary>Older fixes carried over from the first round of this audit</summary>

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
- Season values come from `config.toml`, not from 19 hand-edited literals.
- Six near-identical training scripts collapsed to shared modules (2255 → 1551
  lines) with byte-identical output.

</details>

---

## Tooling added here

Each of these is built and tested. None is wired into the runtime yet — that is
the next deliberate step, tracked in [ROADMAP.md](ROADMAP.md).

| Module | What it is for |
|---|---|
| `scripts/evaluate.py` | One command that reproduces every number in this README. Opens with a **leakage gate** that refuses to report accuracy on a contaminated dataset, then reports baselines, model metrics, the closing-line favourite, and ROI with standard errors. Degrades explicitly rather than scoring a partial odds join. |
| `src/Utils/metrics.py` | Expected calibration error, maximum calibration error, reliability curves, multi-class Brier, guarded log loss. Cross-checked against scikit-learn to `0.000e+00`. |
| `src/Utils/validation.py` | Purged walk-forward splits with embargo, plus selection-bias and deflated-Sharpe correction. On this dataset 100 hyperparameter trials are worth **~2.5 standard errors** — larger than most effects anyone would claim from it. |
| `src/Utils/Kelly_Criterion.py` | Fractional Kelly, probability shrinkage, two-way de-vig (proportional and Shin), per-bet and slate caps. The audited arithmetic underneath is unchanged. |
| `src/Process-Data/Get_Odds_Line_Movement.py` | Timestamped intraday odds snapshots and closing-line-value arithmetic. |
| `src/Process-Data/Create_Recency_Features.py` | EWMA and rolling features replacing season-to-date cumulative averages, with an as-of assertion that raises on leakage. |
| `src/Utils/nba_client.py` | Thin, lazily-imported `nba_api` wrapper. |

Two of these are worth reading the reasoning for, because both encode a lesson:

**`size_bet` exists because full Kelly is the wrong policy, not wrong maths.** The
underlying formula is correct — it is exactly `f* = (bp−q)/b`. But on a
mis-specified edge, a Monte-Carlo over 20,000 seasons ends at a **median 0.278×
bankroll with a 100% chance of a 50% drawdown**; at quarter Kelly, a median 0.946×
and no such drawdown. The honest counterpoint is in the same report: when the
edge is exactly right, full Kelly returns 3.97× against quarter Kelly's 1.83×.
Roughly half the growth when you are right, in exchange for not going broke when
you are wrong.

**`closing_line_value` exists because anything measured at the close is not a
signal.** Betting the closing line returns CLV of exactly `0.0` by construction.
You cannot beat a price you already have. An honest limit the module documents:
the historical odds tables are single scrapes, so there is **no intraday history
to recover** and past CLV cannot be computed. That question is permanently
unanswerable for games already played.

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

python -m src.Train-Models.XGBoost_Model_ML   --dataset dataset_2012-26 --trials 15 --splits 5
python -m src.Train-Models.XGBoost_Model_UO   --dataset dataset_2012-26 --trials 15 --splits 5
```

`--trials` defaults to **15**, not 100. On this dataset the standard deviation
of the cross-validation score *across configurations* is several times the entire
true spread of held-out performance, so a bigger search ranks noise — and at 100
trials the expected winner's curse is 2.50 standard errors. Every trainer now
prints that correction next to its own `Best val log loss`, so the number cannot
be read without it. See [ROADMAP.md](ROADMAP.md) Level 6.2.

Fill gaps with `--backfill` (optionally `--season 2025-26`).

> `Process-Data` and `Train-Models` contain hyphens and so are not importable as
> packages. All three of these work and are tested:
> `python -m src.Train-Models.XGBoost_Model_ML` ·
> `python src/Train-Models/XGBoost_Model_ML.py` ·
> `cd src/Train-Models && python -m XGBoost_Model_ML`.
> Renaming the directories is tracked as follow-up work.

## Verifying the numbers in this README

Every performance figure here is reproducible. Nothing needs manual setup:

```bash
python scripts/evaluate.py
```

It opens with a leakage gate and **refuses to report accuracy at all** if the
dataset is contaminated, so a passing run is itself evidence that the join is
clean. Then it prints the baselines, the trained model, the closing-line
favourite, and ROI across edge thresholds with standard errors.

Useful flags: `--dataset`, `--test-size`, `--model {logreg,gbdt,none}`,
`--skip-odds`. See `scripts/README.md` for the gate's checks and degradation
rules.

## Tests

```bash
pytest Tests/                       # or: python -m unittest discover -s Tests -t .
```

Most tests need no network access and no TensorFlow or XGBoost import. The model
suites under `src/Models/` do need the downloaded NBA data described above; when
`data/nba` is absent they skip rather than fail, so a fresh clone still gets a
meaningful signal.

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

## Further reading

- [ROADMAP.md](ROADMAP.md) — the levels between this working pipeline and a real
  forecasting or betting engine, with the measured case for why model tuning is
  not the bottleneck.
- `scripts/README.md` — what the evaluation harness measures and how it degrades.
