# Results: Pipeline B (the model against the market)

> Closing-line comparison, betting rules, a measurement bug, stint-level rotations, and the survey of other repos. Part of the [Legit NBA Predictions README](../README.md).

---

## Pipeline B: the model against the market

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
[docs/market_anchor.md](market_anchor.md).

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

### Being selective does not help — including backing underdogs

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
[the player-Elo block](results_pipeline_a.md#the-player-elo-block-a-strong-result-that-was-a-leak).

### A number in the committed output that looked good, and was a bug

An earlier revision reported **+11.76%** ROI for the model and **+13.54%** for model +
availability at a 0.55 threshold. Both were a bug, and the bug inverted the sign: the
stake is `p_model − p_raw`, which is **negative whenever the market is the more confident
side** — 1,382 of 4,076 games at that threshold — and the ROI divided total profit by
`stake.sum()`, which summed them. A loss of −6.48 over a denominator of −47.88 reported
as a profit. Corrected, every threshold is negative or flat: **−8.55%** and **−9.37%** at
0.55, −5.02% and −2.74% at 0.60, +1.76% and −0.06% at 0.65. Both the mean and the
bootstrap now restrict to placed bets, pinned by `Tests/test_roi_denominator.py`. Full
post-mortem: [docs/measurement_bugs.md](measurement_bugs.md).

Every figure in this section is the pre-refactor 4,076-game measurement, kept as the
post-mortem it is; the re-measured sweeps are the ones reported above.

## Stint-level rotations: the one table nothing has read

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
evidence — **2,046 snapshots across 1,037 report dates / 237,436 status events**
as of 2026-10-26, with the live capture loop running and the historical backfill
complete: every game date from January 2020 through the end of 2025-26 is
covered, with zero absences and zero parse failures across every leg (the
2025-26 boundary incident is documented in the injury pipeline, as is the fix
that surfaced its silent drops). The first measured use (issue #37) started as a
null with an uninformative interval; with the completed store the criterion-4
measurement now reads **+0.002632 nats, |z| 4.47** against the availability
block — a |z| > 3 signal reported as non-promoting, because the pre-registered
season protocol (the promotion path) stays null. Three-era URL handling,
point-in-time semantics, the two bugs caught before damage and the
security audit are in [docs/injury_pipeline.md](injury_pipeline.md).

## Against other repos

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
[experiment log](results_pipeline_a.md#the-experiment-log-17-pre-registered-runs)), which is consistent with
its §6.4 finding that feature volume does not track performance. Of ~45 repos surveyed,
one publishes a baseline; 13 of 28 game-outcome repos publish no metrics at all while one
29★ repo has no README whatsoever.

**What this project adds to that literature is not a better model.** It is a baseline, a
held-out protocol, a measured vig, and a comparison against the actual closing price.
