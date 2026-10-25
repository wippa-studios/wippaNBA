# Roadmap

Where this project is, what is actually blocking it, and what has to be true
before any claim of being an "elite prediction engine" is meaningful.

Every number in this document was measured from the data committed in this
repository. Where a figure is quoted it names how it was produced. Numbers that
have not been reproduced here are labelled as such.

---

## The short version

Level 0 (correctness) is done. The pipeline is no longer quietly wrong.

The bottleneck is **not** model tuning. It is, in order:

1. There is no reproducible way to measure whether a change helped. Every
   performance figure in this project's history — upstream's and ours — has
   existed as a claim in a README or a filename, not as a command someone can
   re-run.
2. The feature set is season-to-date cumulative averages. That is close to the
   information floor and cannot support a betting edge.
3. Odds are captured once, at the close. The closing line is the hardest price
   in the market to beat.

---

## What "elite" has to mean here

These are two different goals with very different difficulty, and conflating
them is how projects like this one end up with impressive numbers and no money.

**Elite forecasting** — probabilities that are well calibrated and that beat
naive baselines on log loss and Brier score. Achievable, honest, measurable.

**Elite betting** — durable positive ROI after vig. Much harder. Measured here:
backing the **closing-line favourite** on every game returns **−5.5%** ROI over
1,697 held-out rows. The market prices public information very efficiently. The
106 cumulative season-average features in this repo are nowhere near the
information edge needed to beat it.

Pursue both, but hold them to different standards of proof, and never let a
result in the first be reported as a result in the second.

---

## What has already been established

These are measurements, not intentions. Re-running them is the first task below.

| Finding | Measurement |
|---|---|
| The `68.9%` headline was target leakage | Trivial rule on contaminated columns scores 96.1% (2025), 100% (2026); chance is 50% |
| Leak-free accuracy | 0.657 vs a 0.547 always-bet-home baseline, over 1,722 held-out games |
| **No demonstrated edge at the closing line** | Model ROI +1.5% to +8.3% across edge thresholds, standard errors ±4.4 to ±9.4 — inside one SE of zero everywhere, with the sign moving with the evaluation window |
| **The market beats the model** | Closing-line favourite wins 0.681 to the model's 0.657 |
| The favourite loses money, as it must | −5.46% ROI ±1.62 — that is the vig, and it is the one result here that is statistically solid |
| The totals model has no signal | 0.497 accuracy vs 0.504 for always predicting "over" |
| Class weighting broke calibration | mean P(home) 0.5079 vs a true 0.5694; the resulting model scores 0.5324 accuracy — identical to its own base rate — while a +8.7% edge sat unactioned |
| Selection effect is real | 100 hyperparameter trials are worth ~2.50 standard errors at this sample size |

All of the above reproduce with `python scripts/evaluate.py`. The last row
matters for everything below. **Any number produced by searching many
configurations and reporting the best one is optimistic by construction.**
See Level 6.

---

## Level 0 — Correctness ✅ complete

Done in PR #1. Summary, for context:

- Target leakage removed at the join and on the live fetch path; `Data/dataset.sqlite` rebuilt and verified at chance.
- Silent prediction bugs fixed (3-column permutation in the totals model, push-rendered-as-OVER, Flask labelling the winner as "home" and misattributing EV).
- Leaked third-party API key removed; Werkzeug debug source disclosure closed; timeouts on every request and subprocess.
- 81 tests passing under both `pytest` and `unittest`; CI on Python 3.10–3.11.
  (Now 419 — see Levels 1 and 2.)

**Standing hazard:** nothing has been retrained. `xgboost` and TensorFlow are
absent from the review environment, so the XGBoost and NN paths were verified by
construction and AST comparison, never by execution. Every artifact in
`Models/` predates the fixes and is untrusted. `Trained-Model-OU-49.2.keras`
additionally carries the old totals column order and will fail **silently**.

---

## Level 1 — Make measurement reproducible ✅ built, not yet wired

**Why first:** without this, every subsequent level is unfalsifiable. It is also
the cheapest thing on this list by a wide margin, and it is pure profit.

**Done:** `scripts/evaluate.py` reproduces every figure in the README and in this
document. It opens with a leakage gate that refuses to report accuracy on a
contaminated dataset — verified to fire on the old leaky tables (89.5% of rows
join a same-day price) and on a poisoned frame (the W-increment rule reaches
83%). It reports baselines, model metrics, the closing-line favourite, and ROI
across edge thresholds with standard errors, and it degrades explicitly rather
than scoring a partial odds join.

**Not done:** nothing invokes it. It is not in CI, and no training run calls it.
Until that is fixed, a regression in the model can land without anyone noticing
that the number moved. That wiring is the first thing still outstanding here.

Also still outstanding at this level: dataset and model manifests, so a model
file can be traced to the exact dataset, seed and column order that produced it.

### 1.1 Promote the evaluation harness ✅

The scripts that produced the measurements above lived outside the repository and
would have bit-roted immediately. They are now `scripts/evaluate.py`, documented
in `scripts/README.md`.

- The dataset is rebuilt leak-free by construction and its as-of correctness is
  asserted before any accuracy is reported.
- Training is on a chronological split. Reported: **log loss, Brier, AUC,
  accuracy, the closing-line favourite baseline, and ROI with standard errors**.
- A partial odds join is refused rather than scored.

**Gate — met:** `python scripts/evaluate.py` reproduces the figures in this
document from committed data, with no manual steps.

**Remaining:** calibration error and CLV are implemented as modules
(`src/Utils/metrics.py`, `Get_Odds_Line_Movement.py`) but are not yet reported by
the harness.

**Retrospective CLV is struck, not deferred.** It cannot be computed for any game
already played. The historical odds tables are single scrapes, so there is no
intraday history to recover and no CLV figure for those games exists, now or ever.
This is a permanent unanswerability for past games, not a backlog item.

What is *not* struck is prospective CLV, and the capture machinery now ships for it:
`Get_Odds_Line_Movement.py --sportsbook all --watch --every 900` keeps a trajectory
resident, and `contrib/odds-capture.timer` does the same under systemd for anyone who
would rather schedule it than babysit a process. Coverage has to build before any
figure means anything — `snapshot_coverage()` reports density per game, and a game
with one snapshot is indistinguishable from a line that never moved, so CLV over thin
coverage is biased toward zero.

Note what CLV would and would not settle. A positive CLV on a real, prospectively
captured series would be evidence of skill; a negative one would exonerate the
strategy. It is the only measurement available that can separate the two, because
everything measured at the close returns CLV of exactly 0.0 by construction — you
cannot beat a price you already have.

### 1.2 Leakage assertion as a permanent gate

`Tests/test_create_games.py::test_main_never_uses_a_same_day_snapshot` proves the
join does not use a same-day snapshot. Extend the same idea to the live path in
`main.py` (`DateTo` must be yesterday) and to any future data source.

**Gate:** CI fails if any code path can produce a same-day feature snapshot.

### 1.3 Dataset and model manifests

Nothing records which dataset, seed, or column order produced a given model.
The trainers now persist `<model>_features.json`; extend that to a manifest with
dataset table name, row count, build date, seed, git SHA, and a content hash.

**Gate:** given a model file, you can determine exactly what trained it.

---

## Level 2 — Calibration as a first-class metric

**Why:** expected value is computed directly from the output probabilities. A
model that is 65% accurate but miscalibrated produces systematically wrong bet
sizing even when its direction is right. This was already demonstrated — class
weighting moved mean P(home) 6 points off the true base rate while the calibrator
was fit on unweighted data.

### 2.1 Calibration reporting

- Reliability curves and expected calibration error, per model and per horizon.
- Calibration curve on the held-out set, not on the set used to fit the
  calibrator (the old code reused one tail for early stopping, calibration, and
  part of the final CV fold).

### 2.2 Refit calibrators on leak-free data

All committed calibrators were fit on the leaked dataset. Retrain everything.

### 2.3 Ensemble with calibrated outputs

Blend the XGBoost, logistic-regression and NN models on a held-out slice. Only
use an ensemble if it improves **log loss**, not accuracy.

**Gate:** the project's headline number is log loss and Brier, not accuracy.
Accuracy is reported only as a secondary diagnostic.

---

## Level 3 — Features (the real ceiling)

**Why this is the ceiling:** the current 106 features are season-to-date
cumulative averages. In March they say a team is playing .500 without
distinguishing last week from last month. Between consecutive games these
features barely move, so the model is nearly blind to form.

Ordered by expected value per unit of effort.

### 3.1 Recency weighting — do this first

Replace raw cumulative aggregates with exponentially-weighted moving averages
(half-life ~10 games). This is the cheapest meaningful change on the list and
typically the single largest gain.

**Gate:** measurable improvement in log loss on the same chronological split,
confirmed by `scripts/evaluate.py`.

### 3.2 Rolling windows

Last 5 / 10 / 20 games per team, plus home/away splits. Straightforward once 3.1
establishes the pattern.

### 3.3 Context adjustment

Opponent strength (schedule of adjusted net rating) and pace adjustment. Current
features compare raw rates between teams that play at different paces and
different opponents.

### 3.4 Rest and travel — correct, not new

Days-rest is now correct (date-only, UTC, against the previous *game*; upstream
had three mutually incompatible implementations, one of which included the
current game). Extend to travel distance and time-zone changes, which is where
back-to-backs actually bite.

### 3.5 Player-level data — the large project

On/off impact, availability, and lineup data. This is where genuinely modern
edges live and it is a substantially larger undertaking than everything above
combined. Treat it as a separate phase with its own evaluation, not an increment.

**Gate for 3.1–3.4:** each step improves the Level 2 metrics on the same split.

---

## Level 4 — Market integration (where betting edge would come from)

**Why this level exists:** you do not beat the closing line by being *right*.
You beat it by finding the price earlier, or by locating value the market's own
pricing missed. Neither is possible with one scrape of one book at the close.

### 4.1 Capture line movement — highest leverage on this level

Scrape odds repeatedly through the day, not once. The publicly known structural
edge is the gap between the price you could have taken and the close. Without
intraday snapshots, closing-line value is unmeasurable and therefore unachievable.

### 4.2 Multiple books, proper de-vigging

`OddsData.sqlite` stores one book per season. Shop several, de-vig each two-way
market correctly, and model the consensus line rather than a single sportsbook's.

### 4.3 Stale versus sharp lines

A line that has not moved against a strong public team may be available. Requires
4.1's snapshots.

**Gate:** CLV becomes a reported metric, not an aspiration. If CLV is not
positive, say so and stop — that is a valid, publishable result.

---

## Level 5 — Risk and operations

### 5.1 Fractional Kelly with caps

The Kelly implementation in `src/Utils/Kelly_Criterion.py` is mathematically
correct (verified: exactly `f* = (bp−q)/b`; only the function name is misleading).
The problem is the **policy**: it is full Kelly, uncapped.

- Use ¼ Kelly (standard for uncertain edges).
- Cap per-bet and per-day exposure.
- Shrink model probabilities toward the market implied probability by the
  model's *measured* log-loss advantage.
- Cap total exposure across correlated simultaneous bets — a slate of eight
  games all nominally 5% is not eight independent bets.

### 5.2 Scheduling, monitoring, failure modes

A daily job that cannot silently succeed while doing nothing. The upstream
pipeline exited 0 for months during the offseason and nobody noticed; that class
of failure is now fixed but the monitoring around it is not.

### 5.3 Paper-trade before real money

Validate over a full season on paper. Any real deployment should follow months of
shadow operation.

---

## Level 6 — Validation discipline

### 6.1 Purged walk-forward with embargo

Consecutive games share team state, so validation folds are not independent.
Use purged splits with an embargo window.

**Tested here and rejected as the fix.** `purged_walk_forward_split` in
`src/Utils/validation.py` does what it says, but on this dataset it changes
nothing: `embargo_pct=0.01` over the 4,522-day training span resolves to a
45-day embargo, and swapping `TimeSeriesSplit` for the purged split moved mean CV
log loss by **+0.0006 to +0.0016** across three configurations. Fold *geometry* is
a separate problem and is fixed separately: fold width and count are now
configurable, a fold is capped at one season (1,230 rows) rather than at
`n/(splits+1)` — which was 2,603 rows, two seasons, two years — and the search
scores on a recency-weighted mean rather than a flat one.

### 6.2 Correct for multiple testing

The trainers used to run 100 trials and select the best on one slice. That number
is now **15**, and every trainer prints `report_selection_risk`'s selection
correction next to its own `Best val log loss`, so a winner's-curse statistic can
no longer be presented as a measurement.

Measured on the current table, over 12 sampled configurations: the standard
deviation of the 5-fold CV score *across* configurations is **0.0046** while the
entire true range of held-out log loss across the same configurations is
**0.0009** — the search is ranking noise by a factor of about five. The expected
gap between the best and the mean CV score is −0.0075 nats at 15 trials and
−0.0086 at 100, so 100 trials buy **0.0011 nats**. `selection_bias_estimate`
returns 2.50 SE at 100 trials against 1.74 SE at 15, on 1,735 held-out rows.

The remaining half of this level is still open: the deflated Sharpe ratio
(`deflated_sharpe_ratio`) is implemented and tested but is not reported by any
trainer, because it needs a returns series and the trainers report log loss.

### 6.3 Lock the holdout

Once. At the end. No peeking. The current test set is a single contiguous block
of ~1.7k rows spanning ~1.2 seasons — it cannot be re-used repeatedly without
being burned.

---

## Explicit non-goals

Things that look like progress and are not:

- **Raising accuracy.** It is not the product metric and it is not the barrier.
- **Adding more models.** Model count is not the constraint; features and
  measurement are.
- **More hyperparameter search without selection correction.** This makes
  reported numbers *less* trustworthy, not more.
- **Trading real money before Level 5 and the Level 6 gates.** Not financial
  advice. The expected value of this project is a better forecaster and a
  publishable, honestly-negative result — not income.

---

## Open questions

1. **Is a durable public-data edge achievable at all?** Unknown, and Level 4 is
   the experiment. A negative answer here is a legitimate outcome worth writing
   up.
2. **Data availability.** `stats.nba.com` was unreachable from the review
   environment and `data.nba.com` served a year-stale feed. `nba_api` wraps the
   same host, so it bought maintainability and proxy support but did not fix
   availability. A commercial source that serves stats *and* odds would also
   retire the `sbrscrape` scraping — worth evaluating.
3. **Licensing.** Neither this fork nor upstream ships a LICENSE file. That
   constrains reuse and should be settled before the project attracts
   contributors.
