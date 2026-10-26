# Form-mismatch spots, priced: the hot home team against the cold road team

The issue thread proposed the folk rule directly: a home team that is **5-5 or
better over its last 10 games**, hosting a team that is **2-8 or worse**, is a
spot.  This page answers it in the only form that pays: not "does the home team
win?" but "does it beat the price?"

**The short version.** The mismatch is real.  The home team wins **79.5%** of
these games, a genuine win-probability mismatch.  The closing moneyline prices
it at **82.7%**, so backing it returns **−4.12%**; on the spread at a flat −110
it returns **−5.28%** (|z| 1.71).  The reverse spot — cold home team, hot road
team — loses too (−5.06% on the road spread).  There is no side of the belief
that pays.  Nothing is promoted; the canonical model is unchanged.

Pre-registration: `I74-prereg-form-mismatch-spots` in
`results/experiments.jsonl`.  Artifacts: `results/issue74_form_mismatch.json` /
`.md`.  (`I74` is the campaign label fixed before the run, not a GitHub issue
number — the pull request is the record for this work.)

## The rule, fixed before the run

| element | definition |
|---|---|
| form window | wins in the team's most recent up-to-10 games **strictly before** tonight, counting back over the full frame with **no season reset** (`nba_data.recent_form_features`, the I60b definition) |
| history gate | both teams must have **≥ 10 prior games** in the frame, so a 2-0 team in October cannot count as "2-8" |
| **H** (primary) | `home_form10 ≥ 5` and `away_form10 ≤ 2` → back the home team |
| **M** (mirror) | `home_form10 ≤ 2` and `away_form10 ≥ 5` → back the away team |
| **G** (ladder) | `gap = home_form10 − away_form10`; back home for gap ≥ 2, ≥ 4, ≥ 6, ≥ 8 |
| prices | closing spread normalised to the home-cover threshold (`c44_spread_total.effective_line`, the pre-2022-23 tables store an unsigned handicap), settled at a flat −110 with a push returning the stake; closing moneyline at the posted raw price, vig included |

**One promotion gate:** H, home ATS, pooled 2015-16..2025-26, at
**|z| > 3** on 2,000 bootstrap resamples (seed 42).  Every other object — the
moneyline, the mirror, the gap ladder, the per-season rows, the 2025-26 slice,
the within-season-form variant — is descriptive and cannot promote.

This is an **explicit multi-season meta-replication**, declared as such.  The
cell is far too rare for a one-season test (47 priced games in 2025-26), and
2025-26 closing lines stop at 2026-01-07.  The pooled test is retrospective
evidence, not a forward claim.

**Power, stated before execution.** At n = 965 and a flat −110 the per-bet
profit standard deviation is about 0.95, so the bootstrap SE of ROI is about
**3.1%**; |z| > 3 therefore needs an ROI of roughly **9%**.  A null here rules
out the large edge the spot is usually sold as.  It does not rule out a 2–3%
edge, and a positive point estimate below the bar is not evidence of one.

## Population

`build_games()` joined to a closing line gives 12,386 games; one 2018-19
pick'em (Washington–Minnesota, `Spread = 'PK'`, moneyline −110/−110) has no
numeric spread and is dropped from **both** price arms, leaving **12,385**.
That row is not in the cell, so the primary population is unaffected; the drop
is recorded as `dropped_no_spread` in the artifact.

The H cell holds **965** priced games — 88, 83, 109, 76, 76, 72, 105, 84, 103,
122 and 47 by season from 2015-16 to 2025-26.

## Primary result: it loses, and the moneyline shows why

| bet | n | ROI | SE | \|z\| | 95% CI | win rate | break-even |
|---|---|---|---|---|---|---|---|
| H, home ATS −110 | 965 | **−5.28%** | 3.09% | 1.71 | −11.30% to +1.05% | 49.6% | 52.38% |
| H, home moneyline | 965 | **−4.12%** | 1.70% | 2.43 | −7.40% to −0.79% | 79.5% | 82.73% |

13 of the 965 spread bets push; the 49.6% win rate is over the 952 graded bets,
against the 52.38% a −110 price requires.

The moneyline row is the whole story.  The home team really does win four games
in five in this spot, and the market asks for 82.7%.  The mismatch in *wins* is
not a mismatch in *price*.  The spread loses a little more than the moneyline:
the same overpricing is on both prices, and the favourite also has to cover a
number set for it.

The other side of these same 965 games loses too; it is in the consolidated
table below, where the arithmetic of the vig is the point.

## The mirror loses too

| bet | n | ROI | SE | \|z\| |
|---|---|---|---|---|
| M, away ATS | 948 | **−5.06%** | 3.12% | 1.63 |
| M, away moneyline | 948 | **−4.12%** | 2.34% | 1.76 |

Cold home team against a hot road team: the road team covers 49.7% of the time
and wins 67.8% at a mean price of 71.0%.  It loses 5% on the spread and 4% on
the moneyline.  Reversing the story does not reverse the result — the market
prices both ends of the form spectrum.

## Every side of the trade

Both configurations were priced on both sides, at both prices.  All eight
cells lose:

| configuration | side | price | n | rate | break-even | ROI | \|z\| |
|---|---|---|---|---|---|---|---|
| hot home / cold road | home | ATS | 965 | 49.58% | 52.38% | **−5.28%** | 1.71 |
| | away | ATS | 965 | 50.42% | 52.38% | −3.69% | 1.20 |
| | home | ML | 965 | 79.48% | 82.73% | **−4.12%** | 2.43 |
| | away | ML | 965 | 20.52% | 21.39% | −6.69% | 0.97 |
| cold home / hot road | home | ATS | 948 | 50.32% | 52.38% | −3.86% | 1.24 |
| | away | ATS | 948 | 49.68% | 52.38% | **−5.06%** | 1.63 |
| | home | ML | 948 | 32.17% | 33.05% | −0.76% | 0.13 |
| | away | ML | 948 | 67.83% | 71.00% | **−4.12%** | 1.76 |

**These are not eight independent results.**  Betting the other side of the
same game is the complement of the same price.  At −110 a coin-flip game
returns −4.55% on each side, −9.09% combined; the exact cell's two spread legs
sum to **−8.97%** and the mirror's to **−8.92%**, which is the vig and nothing
else.  The mirror image of a losing bet is another losing bet.

The informative quantity is the *difference* between the sides.  On the spread
the hot home team is the worse side by 1.6 points (−5.28% against −3.69%), the
same direction as the ladder's |z| 4.36 cover-rate result.  On the moneyline
the ordering reverses, but that comparison is driven by how the price is
distributed across games rather than by one number, so it is not a second
signal.

The least negative cell is the mirror's home moneyline at **−0.76%** (|z| 0.13,
SE 5.70%, n 948).  That is a flat cell inside a losing sample, not a break-even
spot: it sits 1.4 standard errors from a 5% loss.

## The gap ladder: a real, small mispricing that the vig eats

| gap | bets | home ATS ROI | \|z\| | home ML ROI | \|z\| |
|---|---|---|---|---|---|
| ≥ 2 | 3,741 | −6.69% | 4.36 | −5.42% | 5.02 |
| ≥ 4 | 1,482 | −5.78% | 2.35 | −4.87% | 3.54 |
| ≥ 6 | 343 | −4.72% | 0.93 | −1.36% | 0.62 |
| ≥ 8 | 42 | −15.58% | 1.11 | +2.29% | 0.44 |

The broad cell clears |z| > 3 **in the negative direction**: over 3,741 games the
hot home team covers only **48.8%** of the time.  That is a genuine signal — the
market's spread is slightly too generous to the home side in this state, by
roughly a third of a point.

It is not bettable.  The away side covers 51.2%, and −110 needs 52.4%, so the
away side returns about −2.3% (the two sides' ROIs sum to the vig).  The
mispricing, about 1.2 percentage points of cover probability, is less than half
the vig, about 2.4 percentage points per side.  A detectable inefficiency is
not the same as a harvestable one, and this ladder is exactly that distinction.
The wider gaps are thin and noisy; nothing here is monotone.

## Per season

| season | bets | home ATS ROI | home ML ROI |
|---|---|---|---|
| 2015-16 | 88 | −8.78% | −4.26% |
| 2016-17 | 83 | −22.89% | −17.14% |
| 2017-18 | 109 | −0.92% | +0.03% |
| 2018-19 | 76 | +1.79% | −3.50% |
| 2019-20 | 76 | +1.79% | −2.02% |
| 2020-21 | 72 | −4.55% | +5.47% |
| 2021-22 | 105 | −12.64% | −12.98% |
| 2022-23 | 84 | −2.27% | −5.44% |
| 2023-24 | 103 | −9.09% | −2.87% |
| 2024-25 | 122 | +0.97% | −1.63% |
| 2025-26 | 47 | +1.55% | +3.40% |

Seven of eleven seasons are negative on the spread and eight on the moneyline.
The four positive seasons are small and have no common feature; the two worst
(2016-17, 2021-22) are worse than −12%.  This is not a spot with a stable
seasonal core and a few unlucky draws.

## The 2025-26 slice and the within-season variant

The 2025-26 slice is **47** priced cell games: home ATS **+1.55%** (|z| 0.11),
home moneyline **+3.40%** (|z| 0.49).  The confidence interval on the spread
runs from −26.9% to +25.9%.  This slice cannot decide anything, which is why the
primary was declared as a pooled meta-replication rather than a last-season
holdout.

The within-season-form sensitivity — the same rules with the window reset at
each season boundary and 10 prior games required *this* season — keeps **833**
games and returns home ATS **−5.51%** (|z| 1.65), home moneyline **−3.79%**
(|z| 2.15).  Resetting the form definition does not rescue the spot; the
cross-season version the primary used is not what produces the loss.

## Model diagnostic: a hint of information, not a profit

Canonical 26 features plus a single indicator for the H cell, trained on the
11,974 rows before 2025-10-21 and scored on the 1,225 rows of 2025-26 (131 in
the cell): **+0.000737 nats** better than the canonical model, CI
[+0.000056, +0.001399], **|z| 2.15**, indicator coefficient +0.032.  The
positive control on the same split — availability on minus off — reproduces
I60's **+0.009247 nats, |z| 4.93**, so the split and the bootstrap are the ones
that can detect a real effect.

Read the two results together and they are consistent.  There is a *hint* that
the joint form state adds something the canonical features lack (the
indicator's sign says the home team wins a little more often than the model
expects), but it is far below the bar, and the betting test says the market
already knows it: the moneyline is priced 3.2 percentage points above the
realised win rate.  Information the model lacks is not automatically money;
here it is money the market has already taken.

## What this rules out, and what it does not

At 965 bets the bootstrap SE is 3.1%, so the gate needs an ROI near 9%.  The
result rules out the large edge the spot is usually sold as.  It does not rule
out a 2–3% edge, and this run is not evidence for one — the point estimate is
negative, and the confidence interval only barely includes zero.

The interesting part is not the null on the exact cell.  It is that the broad
form-gap cell produces a **|z| = 4.36 negative** result and still cannot be
traded, because the mispricing is smaller than the vig.  That is the honest
shape of most "outside the box" angles: a real, small regularity that the price
has already absorbed to within a fraction of a point.

Reproduce with:

```bash
OMP_NUM_THREADS=1 python scripts/issue74_form_mismatch.py
python -m pytest Tests/test_issue74_form_mismatch.py -q
```

The script writes `results/issue74_form_mismatch.json` and `.md` and appends
`I74-run-form-mismatch` to `results/experiments.jsonl`.  The canonical model,
the feature layer, and the betting rule are unchanged.
