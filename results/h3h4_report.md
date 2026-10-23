# H3 + H4 run report (theory dossier)

Pre-registered in `results/experiments.jsonl` BEFORE execution:

* `H3-prereg-recalibration` (declared 2026-10-23T09:14:46Z)
* `H4-prereg-relative-rest` (declared 2026-10-23T09:14:46Z)

Those entries govern; this file is the readable mirror. Every number below was
computed by `scripts/h3_recalibration.py` and `scripts/h4_rest_differential.py`
in this session and is also in `results/h3_recalibration.json` /
`results/h4_rest_differential.json` and the appended `experiments.jsonl` rows.

Reproducing commands:

```sh
cd /home/Joel/Desktop/Legit-NBA-Predictions
.venv/bin/python scripts/h3_recalibration.py      # H3, ~20 s
.venv/bin/python scripts/h4_rest_differential.py  # H4, ~4 min (player-Elo frame build)
.venv/bin/python -m pytest Tests/ -q              # 1219 passed, 104 subtests (12:34)
```

Canonical model (H3's subject): `RollingLogreg` (C=1.0, L2, lbfgs) on
`load_design()` columns resolved through `DEFAULT_MODELS["full"]` (22 contrast
columns incl. `home_adv`), 5 walk-forward splits (`initial_frac=0.5, gap=0`),
6,600 out-of-fold rows. Measured before any calibration:

| | log loss | Brier | AUC | acc | ECE unif-10 | ECE quant-10 |
|---|---|---|---|---|---|---|
| canonical (identity) | 0.61957 | 0.21511 | 0.70851 | 0.65636 | 0.01337 | 0.01514 |

## H3 — recalibration free lunch: **FAILS** (null / actively harmful)

Fit slices (as pre-registered): for scored season block S, every fitted
calibrator sees ONLY the OOF rows of block S−1; the first block (2020-21 tail,
460 rows) passes through identity. Asserted in code that fit and score rows are
disjoint — the repo's documented failure mode (calibrator fit on the scored
tail) cannot occur here.

| variant | log loss | Brier | ECE u-10 | Δ nats (improv.) | 95% CI | z vs 0 | z vs 0.003 |
|---|---|---|---|---|---|---|---|
| isotonic_prevseason | 0.63249 | 0.21812 | 0.0192 | −0.01292 | [−0.01877, −0.00765] | −4.56 | −5.61 |
| platt_prevseason | 0.62145 | 0.21584 | 0.0116 | −0.00188 | [−0.00335, −0.00036] | −2.46 | −6.39 |
| home_regime_shrink | 0.61957 | 0.21513 | 0.0141 | −0.00001 | [−0.00136, +0.00131] | −0.01 | −4.43 |
| home_intercept_adj | 0.62025 | 0.21548 | 0.0065 | −0.00069 | [−0.00181, +0.00044] | −1.19 | −6.41 |

Δ = loss(before) − loss(after); positive would mean the recalibration helped.
Bootstrap seeds 137 / 2718 give identical point deltas (same rows resampled)
and CIs of the same width. Selection cost of the best-of-4: best − median
0.00127 nats, best − worst 0.01292 nats.

Per-season-block improvements (nats), 2020-21 block is identity by construction:

| variant | 2020-21 | 2021-22 | 2022-23 | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|---|---|---|
| isotonic | 0 | −0.0263 | −0.0138 | −0.0093 | −0.0127 | −0.0074 |
| platt | 0 | −0.0022 | −0.0035 | −0.0046 | +0.0006 | −0.0004 |
| home_regime_shrink | +0.0015 | +0.0023 | −0.0033 | +0.0006 | +0.0006 | −0.0009 |
| home_intercept_adj | 0 | +0.0023 | −0.0047 | −0.0016 | +0.0007 | −0.0005 |

Notes from the run:

* The Pelechrinis regime drift IS visible in the fitted model: the canonical
  `home_adv` coefficient falls 0.346 → 0.300 log-odds across the five folds
  (home probability 0.586 → 0.574). Acting on it post-hoc still gains nothing
  (home_regime_shrink Δ −0.00001): the model's fit window already adapts.
* ECE can be bought but log loss pays for it: home_intercept_adj improves ECE
  u-10 from 0.0134 to 0.0065 while losing 0.0007 nats. ECE is not a proper
  scoring rule; log loss is the pre-registered primary.
* Isotonic on one previous season (n ≈ 1230) is destructive, consistent with
  `results/round3_calib.jsonl`'s −0.008 on a different frame/slicing.

**Verdict (pre-registered rule: ≥ 0.003 nats, CI excludes zero, |z| > 3): FAIL.**
No variant reaches +0.003 nats; two of four are significantly negative.
Recalibration of the canonical model is a null (and isotonic is harmful) at zero
new information. This is a complete result; nothing is kept.

## H4 — relative rest, not absolute rest: **FAILS** (null)

Arms: A1 = canonical 67-col ablation frame (pinned explicitly), A2 = A1 +
candidate (73), A3 = A2 − schedule block (66). Candidate = `rest_diff_cap4`,
`b2b_diff`, four Esteves-style dummies — registered as its own block
(`relative_rest`) under new names, so `test_blocks_do_not_overlap` holds and the
`schedule` block keeps its absolute rest columns.

Sign convention: Δ = loss(first arm) − loss(second arm); **positive = the
second arm is better** (the opposite helpful sign to the minus-block convention
of `feature_block_ablation`, where schedule's historical −0.00204 sits).

### C1 — full + candidate vs full (does it earn its place?)

| setting | Δ | 95% CI | \|z\| | per-fold Δ |
|---|---|---|---|---|
| seed 42 / 5 folds (primary) | −0.00011 | [−0.00046, +0.00025] | 0.61 | −0.00108, −0.00022, +0.00044, +0.00028, +0.00003 |
| seed 137 / 5 folds | −0.00011 | [−0.00046, +0.00022] | 0.63 | (same rows; seed probes CI only) |
| seed 2718 / 5 folds | −0.00011 | [−0.00046, +0.00025] | 0.61 | (same rows; seed probes CI only) |
| seed 42 / 8 folds | −0.00012 | [−0.00046, +0.00024] | 0.64 | −0.00028, −0.00142, −0.00028, +0.00018, +0.00036, +0.00036, −0.00034, +0.00049 |

Point estimate is NEGATIVE at every setting: adding the candidate slightly
*hurts*. The 0.0024-nat target is not approached (upper CI bound +0.00025).

### C2 — full + candidate − schedule vs full (can relative rest replace absolute?)

| setting | Δ | 95% CI | \|z\| |
|---|---|---|---|
| seed 42 / 5 folds (primary) | +0.00034 | [−0.00041, +0.00113] | 0.85 |
| seed 137 / 5 folds | +0.00034 | [−0.00047, +0.00117] | 0.80 |
| seed 2718 / 5 folds | +0.00034 | [−0.00043, +0.00115] | 0.83 |
| seed 42 / 8 folds | +0.00035 | [−0.00040, +0.00114] | 0.90 |

Null. The replacement arm's point estimate beats the canonical frame by
0.0003 nats with an interval straddling zero — nothing established either way.

### C3 — drop absolute rest once the candidate is present (supplementary, 1 look)

| setting | Δ | 95% CI | \|z\| |
|---|---|---|---|
| seed 42 / 5 folds | +0.00045 | [−0.00028, +0.00114] | 1.24 |

Null with a lean: dropping `schedule` costs nothing measurable once the
differential is in — consistent with the issue #45 decomposition (absolute
rest_days were nulls) but itself far below the bar.

### Secondary diagnostic (descriptive, no look): B2B coefficients shrink

Standardised |coef|, mean across 5 folds (seed 42 / 5 folds):

| column | A1 (canonical) | A2 (+ candidate) | shrink |
|---|---|---|---|
| home_back_to_back | 0.0509 | 0.0077 | −85% |
| away_back_to_back | 0.1204 | 0.0649 | −46% |
| home_rest_days | 0.0313 | 0.0293 | −6% |
| away_rest_days | 0.0163 | 0.0144 | −12% |

The pre-registered secondary expectation DID occur: the B2B columns collapse
when the relative-rest block is present — the differential soaks up their
weight. But that representational fact buys no log loss.

**Verdict (pre-registered rule: primary CI excludes zero, |z| > 3, signs
consistent): FAIL.** CI straddles zero, |z| 0.61, and the sign is not even
helpful (Δ < 0 at every setting). Multiplicity: 9 looks → Bonferroni 0.00556
one-sided; the project's |z| > 3 bar governs on top. The block does not earn
its place; it stays out of the canonical frame.

## Leaderboard (primary metric: pooled OOF log loss, lower better)

| rank | arm | log loss | Δ vs canonical | verdict |
|---|---|---|---|---|
| 1 | H4 A3: 66-col frame (cand in, absolute rest out) | 0.61911 | +0.00034 (CI straddles 0) | null, not a result |
| 2 | H4 A1 / canonical model | 0.61945–0.61957 | — | the line to beat |
| 3 | H4 A2: + relative_rest block | 0.61956 | −0.00011 | null (lean: worse) |
| 4 | H3 home_regime_shrink | 0.61957 | −0.00001 | null |
| 5 | H3 home_intercept_adj | 0.62025 | −0.00069 | null (lean: worse) |
| 6 | H3 platt_prevseason | 0.62145 | −0.00188 | significantly worse |
| 7 | H3 isotonic_prevseason | 0.63249 | −0.01292 | significantly worse |

Nothing beat the baseline. Both hypotheses are nulls under their own
pre-registered rules.

## Caveats

* **Leakage:** H3's fit/score slices are disjoint season blocks (asserted); the
  model is never refit; no calibrator sees a scored label. H4's candidate is
  causal by construction (schedule-derived, pre-tip-off). Imputation/scaling fit
  on train rows only (`ra.fit_oof`). Leakage checks passed for both.
* **Season drift:** H3's blocks are single seasons (~1,230 rows each); a
  2-parameter Platt fit on one season is noisy, and the 2021-22 block is fit on
  the partial 2020-21 tail (460 rows). This makes the fitted calibrators
  high-variance — which is part of why season-blocked recalibration fails here.
* **Calibration:** the canonical model is mildly overconfident in both tails
  (ECE 0.013); the intercept adjustment halves ECE at a log-loss cost. Proper
  scoring rules say that trade is a loss; if ECE were the product goal the
  answer would differ, but the pre-registration names log loss.
* **Small samples:** H4's effects (±0.0005 nats) are an order of magnitude under
  the 0.0024-nat bar; the SE (~0.0002 in these comparisons) is smaller than the
  sizing guide's 0.00078 because the paired arms share 95%+ of their features.
* **Home-term regime:** the drift (0.346 → 0.300 log-odds) is measured across
  folds, not identified from a counterfactual; the Pelechrinis points→log-odds
  mapping in `home_regime_shrink` assumes a stable margin→logit scale.

## Files touched

* new: `scripts/h3_recalibration.py`, `scripts/h4_rest_differential.py`,
  `results/h3_recalibration.json`, `results/h4_rest_differential.json`, this file
* edited: `scripts/feature_block_ablation.py` (new feature-block code
  `relative_rest_features` + registry entry `relative_rest`; the frame now emits
  the six candidate columns — H4's A1 pins the canonical 67-col base explicitly)
* appended: `results/experiments.jsonl` (2 pre-registration + 5 H3 + 10 H4 rows)
* not touched: `load_design`, the shared layer, odds/closing-line/injury files,
  `requirements.txt`, no commits or pushes
