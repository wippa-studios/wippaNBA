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

### Folk spots are priced too: the hot home team against the cold road team

The thread's rule, tested at the price rather than as a win-rate story: a home
team **5-5 or better over its last 10 games**, hosting a team **2-8 or worse**.
The window is the team's last up-to-10 games strictly before tip, crossing
season boundaries; both teams need at least 10 prior games. Pre-registered as
`I74-prereg-form-mismatch-spots`; the single promotion gate was the exact cell
on the home spread at a flat −110, pooled over every priced season (965 of
12,385 games), at |z| > 3.

| bet | n | ROI | SE | \|z\| |
|---|---|---|---|---|
| exact cell, home ATS −110 | 965 | **−5.28%** | 3.09% | 1.71 |
| exact cell, home moneyline | 965 | **−4.12%** | 1.70% | 2.43 |
| mirror (cold home, hot road), away ATS | 948 | **−5.06%** | 3.12% | 1.63 |
| gap ≥ 2, home ATS | 3,741 | **−6.69%** | 1.53% | 4.36 |

The home team wins **79.5%** of the exact cell and the close asks **82.7%**.
That is the result: the win-rate mismatch is real and the price has it. The
reverse spot loses as well — and so does every other side: all eight cells
(both configurations × both sides × both prices) are negative, and the two
sides of the exact spread sum to −8.97%, which is the vig rather than a second
finding.

The gap ≥ 2 cell is the interesting one. Over 3,741 games the hot home team
covers **48.8%** — |z| 4.36 in the negative direction, a real mispricing of
about a third of a spread point. It is still not harvestable: the away side
covers 51.2% and −110 needs 52.4%, so ~1.2 points of mispriced cover
probability sit inside a ~2.4-point vig, and the away side returns about −2.3%.

The 2025-26 slice (47 priced games, odds end 2026-01-07) is +1.55% ATS at
|z| 0.11 and cannot decide anything; the within-season form definition keeps
833 games at −5.51%. A model diagnostic — canonical 26 plus a cell indicator on
the 2025-26 holdout — gives +0.000737 nats at |z| 2.15, a hint that the spot
carries a trace of information, already priced. Full write-up:
[docs/form_mismatch.md](docs/form_mismatch.md).

### Stacking filters does not create a price: Elo plus the form mismatch

The next natural move after #74 was to stack: require a pre-game Elo gap of at
least 100 (bet the higher-rated side, home or away), then require that side to
be 6-4 or better over its last 10 and the other side 3-7 or worse. Registered
as `I81-prereg-layered-elo-form` and run as a **closing test of the form
family**, not a candidate.

Two facts made that the right framing. The higher-Elo side is the market
favourite in **95.2%** of games with a 100-point gap, so the Elo layer is close
to "back the favourite", which returns **−3.66%** ATS over all 12,385 priced
games. And the Elo gap correlates **0.736** with the 10-game form gap: Elo has
already absorbed most of the recent results the form filter re-reads.

| bet | n | ROI | SE | \|z\| |
|---|---|---|---|---|
| primary: high side ATS −110 | 1,886 | **−4.42%** | 2.19% | 2.01 |
| high side moneyline | 1,886 | **−4.06%** | 1.23% | 3.32 |
| Elo layer alone, ATS | 6,735 | −5.11% | — | 4.32 |
| back the favourite, ATS | 12,385 | −3.66% | 0.85% | 4.31 |

The form filter improves the Elo layer by about +0.7 points (−5.11% to
−4.42%), but the cell stays negative and the cell-minus-complement contrast is
+0.96pp at |z| 0.38. The 11-game "6-5 against 3-7" reading is worse (−5.28%,
|z| 2.41). Inside the Elo filter the hot/cold cell is −4.47%, hot/not-cold is
−3.78%, not-hot/cold is −5.51%, and neither is −9.01%: the high-side condition
has a hint (+2.99pp, |z| 1.24), the cold-side condition has nothing (+0.41pp,
|z| 0.18), and all four cells lose. Full write-up:
[docs/layered_elo_form.md](docs/layered_elo_form.md).

### A pre-registered search finds no positive structural cell

The search loop registers a whole family of arms before computing any of them,
reports every cell, and raises the bar with the family size: 42 arms need
**\|z\| > 3.24**, not 1.96. Family 1 (`price_structure`) covers moneyline
favourite/underdog by de-vigged probability, home/away ATS by \|line\|, over/under
by closing total, and dog-vs-favourite ATS. The holdout is a new full-season
2025-26 close backfilled from ESPN (1,224 of 1,225 regular-season games), because
the shipped 2025-26 table is an early snapshot that stops on 2026-01-07.

| arm | n disc | ROI disc | \|z\| | n hold | ROI hold |
|---|---|---|---|---|---|
| away ATS, \|line\| 10.5-12.5 (best positive) | 919 | **+3.52%** | 1.12 | 111 | +3.19% |
| back the underdog ATS | 11,640 | −5.63% | **6.45** | 1,224 | −2.99% |
| ML favourite, de-vigged 0.80+ | 2,240 | −5.26% | **6.21** | 267 | −4.20% |
| under, total 220-230 | 3,797 | −7.27% | **4.79** | 458 | −3.71% |

No positive cell clears the bar. Fourteen arms are negative at \|z\| > 3, which is
the market working rather than a finding: over 11,640 discovery games the dog
returns −5.63% and the favourite −3.30%, a paired gap of just −2.33pp (|z| 1.33),
and the two sides sum to **−8.94%** — the hold. The best positive cell's holdout
interval is −14.0% to +20.4% over 111 games, so it is a point estimate, not
evidence. Full protocol and table: [docs/edge_search.md](docs/edge_search.md).

### Best-of-books prices close the "just shop better" route

The profit objective changed the settlement price: not one book's close, but
the **highest pre-game moneyline across books** — OddsPortal publishes exactly
that, and we backfilled it for every game from 2015-16 to 2025-26 (14,305 rows,
98.6% of the regular season priced, overround ~1% per side instead of ~4.5%).
Registered as family 3 (`bestprice_ml`, 14 arms, bar |z| > 3.0).

| bet | n disc | ROI disc | \|z\| | n hold | ROI hold |
|---|---|---|---|---|---|
| ML underdog, de-vigged 0.80+ (best price) | 2,313 | **+20.41%** | **3.29** | 289 | −3.94% |
| back the underdog, no bucket (reference) | 11,902 | **+4.10%** | 2.39 | 1,204 | −1.62% |
| back the favourite, no bucket (reference) | 11,902 | −1.62% | 2.48 | 1,204 | −1.76% |

One discovery cell cleared the bar and the aggregate underdog reference carried
a **95% interval of [+0.45%, +6.73%]** — entirely positive. Both are snapshot
artifacts: the best-of-both-sides price lets each side come from a different
book at a different moment, and on the 13.7% of rows where the two snapshots
are incoherent (1/O1 + 1/O2 < 1) the "best price" is a stale quote. On
**coherent rows only**, the underdog aggregate collapses from +4.10% to
**−0.14%**, and the dog win rate (32.28%) lands on the implied break-even
(32.10%) to within two tenths of a point. The holdout is negative. Full
diagnostics in the ledger (`E03-diagnostics-bestprice_ml`) and
[docs/edge_search.md](docs/edge_search.md).

The useful residue is the measured size of price improvement: best-of-books
settlement is worth **+2 to +4 points of ROI** over a single book's close on
the same bets. That is the honest ceiling of line-shopping. What remains
between −1% and zero is information, and no pre-registered structural cell in
this repo has found it.

### The model at best-of-books prices: the information route closes too

Family 4 asked the last cheap question with existing data: does the repo's own
model — the one predictor here with a measured edge (availability, +0.00390
nats at \|z\| 4.78) — cross zero when settled at the **best bookable price**
instead of one book's close? Pre-registered rule, no free parameters: bet the
side whose model probability beats the best-price implied probability
(6,530 scored-and-priced games; bar \|z\| > 3.0).

| arm | n disc | ROI disc | \|z\| | n hold | ROI hold |
|---|---|---|---|---|---|
| base model, all rows | 5,326 | +6.02% | 2.29 | 1,204 | −3.45% |
| base model, coherent rows | 4,481 | −2.33% | 1.08 | 923 | −4.00% |
| model + availability, all rows | 5,326 | +6.77% | 2.61 | 1,204 | −0.21% |
| model + availability, coherent rows | 4,481 | −1.46% | 0.68 | 923 | −2.01% |

No arm cleared the bar, the holdout is negative in every arm, and the
discovery surplus has the same incoherent-row signature family 3 exposed: on
coherent prices the model's bets pay the vig. The mechanism is one number —
the model backs the **underdog 68.2%** of the time (its probabilities are
flatter than the market's), and coherent best-price longshots are exactly
fair. The information route is closed with the data this repo holds; what
remains open needs player props, live prices, alternate lines, or a
multi-season opening-line archive.

### Thirteen fan-style rules, pre-registered and settled at best prices

The user contributed 13 form/schedule rules (last-10 spots, B2B fades, rest
edges, road-trip returns, head-to-head dogs) *before* any number was computed
— a genuine pre-registration (#91, family 5). All 13 settled at best-of-books
moneyline prices: **no candidate**. No arm cleared \|z\| 3.0; the two biggest
discovery surpluses are fatigue-flavoured and fail the holdout (road
favourite on its 4th game in 6 nights: +9.4% discovery, **−4.6% holdout**;
fade of a road-B2B team off a 2OT game: +17.0% on 91 discovery bets, −43.9%
on 9 holdout bets). Two arms are same-sign on both splits and underpowered —
home off a 20-point loss (+1.9%/+3.9%) and the home dog with 3 of the last 4
head-to-head wins (+2.5%/+10.8%) — watchlist material only, exactly like the
family-1 pair. Rare-event cells (2OT fades: 9 holdout bets) can never clear
a bar, which is itself a finding.

### The 50-rule grid: one significant result, and it is negative

Family 6 expanded the fan rules to a systematic 50-arm grid — every schedule
event (B2B, road B2B, 4-in-6, road trips, 2OT legs, letdowns) crossed with
form gates and venue/price roles, registered before the run (#93). **No
candidate.** The only \|z\| > 3 result in 50 arms is a **negative**: backing
heavy chalk (≤1.30) at the best-of-books maximum returns **−2.88% (z −3.13)**,
−2.64% on coherent rows, −3.72% on the holdout — heavy favourites are
significantly overbet. Its mirror (the dog side of those games, +3.64%) is
noise-level at z 0.84.

The positive leads died the usual deaths: road dogs plain (+6.6%, z 2.8)
collapse to +0.2% on coherent rows; road dogs 5-5+ collapse to −2.9%; and the
one cell positive on coherent prices — the home opponent of a road favourite
on its 4th game in 6 nights (+9.6%, z 2.1) — loses 4.6% on the holdout. The
most intuitive schedule angle in basketball, fading tired teams, is negative
in *every* variant measured (seven arms), because the market already prices
the fatigue.

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
