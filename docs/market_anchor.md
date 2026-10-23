# The market anchor: what the closing price already knows

The Batch A gate-keepers (H1a and H2), pre-registered in
`results/experiments.jsonl` :: `H1H2-prereg-market-anchor` (declared
2026-10-23T11:36:08Z, before any arm ran). Artifacts: the appended
`H1H2-run-*`, `H1H2-diag-*` and `H1H2-verdict` rows of
`results/experiments.jsonl`; the run transcript reproduces every figure below.

**Reproduce:**

```bash
.venv/bin/python scripts/h1h2_market_anchor.py --prereg   # declaration only (already logged)
.venv/bin/python scripts/h1h2_market_anchor.py            # the measurement, ~10 s
```

The question this page answers is the alpha question: **is any of the
information in our model not already in the closing price?** The answer is no,
and the measurement says so more precisely than "no" usually gets said.

## Why odds are legitimate as an input here — and only here

The project's rule — *"odds are never a feature"* — governs the
**market-evaluation** models (the betting and ROI arms): a model that reads the
price cannot be used to evaluate the price. That rule stands.

But for pure **outcome prediction** the closing line is a pre-tip-off quantity,
exactly like a starting lineup: nobody at 7 p.m. knows the game's result, and
the close is a public forecast made before it. Using it as an input to a
forecast of the same game is legitimate. This distinction is written into the
pre-registration (*odds_role_note*) so it cannot be quietly re-litigated later:
the blend below is a forecasting experiment, not a betting experiment, and
nothing on this page feeds the ROI arms.

## The protocol

Fixed before measurement, in `H1H2-prereg-market-anchor`:

- **Arms.** `P` = de-vigged closing probability; `M` = canonical model OOF
  home-win probability (`load_design(with_availability=False)`, 22 features);
  `M+A` = the same with the availability block (`with_availability=True`, 26
  features; pooled OOF log loss 0.61566); `P+M` = a logistic stack of
  `[logit(P), logit(M)]`; `P+M+A` = the same stack with the `M+A` logit in
  place of the plain one. The `P+M+A` parameterisation was fixed **ex ante**;
  the residualised-availability variant is declared **not run** (no
  parameterisation selection).
- **De-vig.** Proportional (`Kelly_Criterion.devig_two_way(method='proportional')`),
  fixed ex ante. Shin's de-vig is declared **not run**: choosing the de-vig
  after seeing which one flatters the model is selection.
- **Stacking, strictly walk-forward by season block.** Scored seasons are
  2021-22 through 2025-26; the overlap's first season (2020-21, 460 rows) is
  the initial training block and is never scored. For scored season *k* the
  stack is fit on every overlap row of the seasons before *k* (expanding
  window) and scores *k* only. Features are `logit(p)` with `p` clipped to
  `[1e-6, 1−1e-6]`; `StandardScaler` + `LogisticRegression(C=1.0)` fit on
  train blocks only. Train/test disjointness is asserted on season labels
  **and** dates (max train date < min test date); identical scored rows across
  arms are asserted; only OOF model logits enter the stack.
- **Scored rows:** 5,328 (2021-22 1,230 · 2022-23 1,228 · 2023-24 1,213 ·
  2024-25 1,143 · 2025-26 514), identical across all arms. Alignment is
  checked first: the 5,788-row overlap reproduces `results/closing_line.json`
  (market 0.59766 / model 0.62340 / model + availability 0.62034).
- **Decision rule.** Δ ≥ 0.002 nats (the minimum effect of interest) with the
  95% CI excluding zero and |z| > 3 at the primary setting (seed 42); three
  primary looks → Bonferroni 0.0167 one-sided; the project's |z| > 3 bar
  governs on top. Seeds 137 and 2718 resample the same rows (same point
  estimate, CI probe only) and add no looks.

One run crashed before producing any measurement —
`H1H2-crash-stack-coverage-assert`: an over-strict coverage assertion demanded
finiteness over the 2020-21 training block that is by design never scored. No
arm-vs-arm number was computed or logged; the run was restarted from scratch
under the identical pre-registration.

## The arms

| arm | log loss ↓ | Brier ↓ | AUC ↑ | accuracy |
|---|---|---|---|---|
| **P — de-vigged closing price** | **0.59964** | **0.20658** | **0.73361** | **0.6785** |
| M — model alone | 0.62532 | 0.21772 | 0.69973 | 0.6488 |
| M+A — model + availability | 0.62234 | 0.21628 | 0.70441 | 0.6515 |
| P+M — price + model stacked | 0.60141 | 0.20720 | 0.73202 | 0.6789 |
| P+M+A — price + model + availability | 0.60144 | 0.20722 | 0.73193 | 0.6787 |

## H1a — the blend is adverse: the price alone beats price + model

Δ = loss(P) − loss(P+M); **positive would mean the blend is better**.

| setting | Δ | 95% CI | \|z\| | verdict |
|---|---|---|---|---|
| seed 42 (primary) | **−0.00177** | **[−0.00292, −0.00072]** | **3.15** | **market better** |
| seed 137 | −0.00177 | [−0.00284, −0.00075] | 3.30 | market better |
| seed 2718 | −0.00177 | [−0.00284, −0.00072] | 3.28 | market better |

**H1a verdict: adverse, not null.** The interval excludes zero **against** the
candidate. Adding our model's logit to the price's logit makes the forecast
worse by 0.00177 nats at |z| = 3.15 — significant under the project's own bar,
in the direction opposite to the hypothesis. The pre-registered bar was
"beats price alone by ≥ 0.002 nats"; the measurement is −0.00177.

**The cold-start caveat, and the post-hoc variant that was deliberately not
run.** The harm concentrates in the first scored season: per-season Δ is
2021-22 **−0.00530**, 2022-23 −0.00169, 2023-24 −0.00036, 2024-25 +0.00003,
2025-26 −0.00081. 2021-22 is the cold start — its stack trains on only the
460-row 2020-21 tail, so the fitted weight on the model logit is at its
noisiest exactly where the harm is largest. A post-hoc variant that drops the
cold start, or refits on more history, was **deliberately not run**: choosing
a variant after seeing where the harm sits is selection, the same error the
de-vig and parameterisation declarations above exist to prevent. The verdict
stands as measured.

H1b (descriptive, no decision weight) confirms the machinery works: P+M beats
M alone by **+0.02392** nats [+0.01759, +0.03018], |z| = 7.45 — the stack is
not broken; the price simply has nothing to gain from our logit.

## H2 — availability beyond price: a tight null

Δ = loss(P+M) − loss(P+M+A); positive would mean availability adds to the
blend. Δ = **−0.00004** [−0.00027, +0.00019], |z| = 0.31 (SE 0.00012;
identical point estimates at seeds 137 and 2718).

**H2 verdict: tight null.** The CI upper bound, +0.00019, is **10× below** the
0.002-nat minimum effect of interest. The pre-registration's own sizing guide
had expected SE ≈ 0.0015–0.002 on this overlap ("a true 0.002-nat effect is at
|z| ≈ 1.5, so a null here establishes nothing AT the bar"); the measured
interval came in an order of magnitude tighter because the two arms share
almost all of their structure. So this is not a power failure: at this
resolution the availability block's increment **beyond price + model** is
ruled out at the minimum effect size.

What the null does **not** rule out:

- It says nothing against the block's forecasting value, which is measured
  separately and stands: +0.00390 nats at |z| = 4.78 in the canonical model
  ([the availability block](availability_block.md)).
- Dare et al. (2015) found player absence fully priced at the close on
  1995–2012 data. The NBA changed its injury-report rules in **Dec 2025**
  (15-minute cadence, [the injury pipeline](injury_pipeline.md)), so "fully
  priced" is genuinely open on current data. H2 tests the increment on this
  2020-21–2025-26 overlap only, and the 2025-26 slice is 514 rows.

## The disagreement decile: where model and market part ways

Descriptive diagnostic, no decision weight. Take the top decile of
|logit(M+A) − logit(P)| on the scored rows — the 533 games where model and
market disagree most:

| | top decile | the rest |
|---|---|---|
| model and market on the same side | **57.2%** | 87.4% |
| market's home-win probability | 0.534 | 0.561 |
| log loss, market (P) | **0.5886** | 0.6009 |
| log loss, our model (M+A) | **0.6734** | — |
| log loss, P+M+A | 0.5911 | 0.6026 |

On the 228 games in that decile where the two actually pick different sides,
**the market's side wins 60.1% of the time** (ours 39.9%). The games where we
disagree with the price most are the games where the price is most right —
including availability, which the close has already read.

## Stacking-weight drift

What the walk-forward stack learns per season (fit on prior seasons only;
`testfit` columns are what a fit on the *scored* season would have found —
shown as drift, never used):

| arm | scored season | n train | price coef (train) | model coef (train) | price coef (test-fit) | model coef (test-fit) | model gap |
|---|---|---|---|---|---|---|---|
| P+M | 2021-22 | 460 | 0.914 | +0.271 | 1.113 | −0.149 | −0.420 |
| P+M | 2022-23 | 1,690 | 1.047 | −0.033 | 0.892 | −0.012 | +0.021 |
| P+M | 2023-24 | 2,918 | 0.991 | −0.028 | 1.158 | −0.089 | −0.061 |
| P+M | 2024-25 | 4,131 | 1.042 | −0.044 | 0.938 | +0.099 | +0.143 |
| P+M | 2025-26 | 5,274 | 1.024 | −0.014 | 0.555 | +0.342 | +0.357 |
| P+M+A | 2021-22 | 460 | 0.895 | +0.282 | 1.130 | −0.165 | −0.446 |
| P+M+A | 2022-23 | 1,690 | 1.051 | −0.037 | 0.890 | −0.008 | +0.029 |
| P+M+A | 2023-24 | 2,918 | 0.994 | −0.030 | 1.166 | −0.096 | −0.066 |
| P+M+A | 2024-25 | 4,131 | 1.047 | −0.048 | 0.867 | +0.189 | +0.238 |
| P+M+A | 2025-26 | 5,274 | 1.011 | +0.003 | 0.447 | +0.483 | +0.481 |

The price coefficient sits at ~1.0 in every training block (the stack learns
to trust the price almost fully); the model coefficient is small and signed
the wrong way in all but the cold-start block. The last scored season
(2025-26, 514 rows) is the exception in the test-fit columns — the drift tail,
not a result.

## Verdict

**H1a: the close absorbs everything our model knows, including availability.**
Price + model is *worse* than price alone (−0.00177 nats, |z| 3.15), and
price + model + availability is indistinguishable from price + model (CI upper
bound 10× below the minimum effect). The market comparison in the README —
market 0.59766 vs model + availability 0.62034, a gap of 0.02268 nats at
|z| = 7.28 on 5,788 games — is the same finding from the other side. There is
no demonstrated betting edge here, and this page does not claim one; see
[the limitations](../README.md#limitations).
