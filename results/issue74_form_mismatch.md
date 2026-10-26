# Issue #74 - form-mismatch spots, priced

Pre-registered as `I74-prereg-form-mismatch-spots`. Generated 2026-10-26T06:15:28.169146+00:00.

The spot: a home team **5-5 or better over its last 10**, hosting a team **2-8 or worse**, both with at least 10 prior games. The question is not whether the home team wins; it is whether it beats the price.

Pooled priced games: **12,385** across 2015-16..2025-26; the cell holds **965**. 2025-26 odds end 2026-01-07.

## Primary (only promotion gate)

| bet | n | ROI | SE | |z| | 95% CI | win rate | break-even |
|---|---|---|---|---|---|---|---|
| exact cell, home ATS -110 | 965 | -5.28% | 3.09% | 1.71 | -11.30% to +1.05% | 49.6% | 52.38% |

**The exact cell LOSES at the price: ROI -5.28% (CI -11.30% to +1.05%), |z| 1.71, over 965 bets. The market has the spot.**

## The same cell at the moneyline, and the mirror

| bet | n | ROI | SE | |z| | win rate | mean price / line |
|---|---|---|---|---|---|---|
| exact home ml | 965 | -4.12% | 1.70% | 2.43 | 79.5% | price 82.73% |
| exact away ats | 965 | -3.69% | 3.08% | 1.20 | 50.4% | line +10.1 |
| exact away ml | 965 | -6.69% | 6.92% | 0.97 | 20.5% | price 21.39% |
| mirror away ats | 948 | -5.06% | 3.12% | 1.63 | 49.7% | line -6.0 |
| mirror away ml | 948 | -4.12% | 2.34% | 1.76 | 67.8% | price 71.00% |
| mirror home ats | 948 | -3.86% | 3.12% | 1.24 | 50.3% | line -6.0 |
| mirror home ml | 948 | -0.76% | 5.70% | 0.13 | 32.2% | price 33.05% |

## Every side of both configurations (descriptive)

| configuration | side | price | n | rate | break-even | ROI | |z| |
|---|---|---|---|---|---|---|---|
| hot home / cold road | home | ATS | 965 | 49.58% | 52.38% | -5.28% | 1.71 |
| hot home / cold road | away | ATS | 965 | 50.42% | 52.38% | -3.69% | 1.20 |
| hot home / cold road | home | ML | 965 | 79.48% | 82.73% | -4.12% | 2.43 |
| hot home / cold road | away | ML | 965 | 20.52% | 21.39% | -6.69% | 0.97 |
| cold home / hot road | home | ATS | 948 | 50.32% | 52.38% | -3.86% | 1.24 |
| cold home / hot road | away | ATS | 948 | 49.68% | 52.38% | -5.06% | 1.63 |
| cold home / hot road | home | ML | 948 | 32.17% | 33.05% | -0.76% | 0.13 |
| cold home / hot road | away | ML | 948 | 67.83% | 71.00% | -4.12% | 1.76 |

The two sides of the exact spread sum to -8.97%; the mirror's two spread sides sum to -8.92%. At -110 a coin-flip game returns -4.55% on each side, -9.09% combined, so a pair of legs summing to about -9% is the vig, not a second finding. Betting the other side of the same games is the complement of the same price and cannot be independent evidence; the informative number is the difference between the sides, which the ladder below measures directly.

## Form-gap ladder (descriptive)

| gap | bets | home ATS ROI | |z| | home ML ROI | |z| |
|---|---|---|---|---|---|
| >= 2 | 3,741 | -6.69% | 4.36 | -5.42% | 5.02 |
| >= 4 | 1,482 | -5.78% | 2.35 | -4.87% | 3.54 |
| >= 6 | 343 | -4.72% | 0.93 | -1.36% | 0.62 |
| >= 8 | 42 | -15.58% | 1.11 | +2.29% | 0.44 |

## Per season (descriptive)

| season | bets | home ATS ROI | home ML ROI |
|---|---|---|---|
| 2015-16 | 88 | -8.78% | -4.26% |
| 2016-17 | 83 | -22.89% | -17.14% |
| 2017-18 | 109 | -0.92% | +0.03% |
| 2018-19 | 76 | +1.79% | -3.50% |
| 2019-20 | 76 | +1.79% | -2.02% |
| 2020-21 | 72 | -4.55% | +5.47% |
| 2021-22 | 105 | -12.64% | -12.98% |
| 2022-23 | 84 | -2.27% | -5.44% |
| 2023-24 | 103 | -9.09% | -2.87% |
| 2024-25 | 122 | +0.97% | -1.63% |
| 2025-26 | 47 | +1.55% | +3.40% |

## 2025-26 slice and within-season-form sensitivity

- 2025-26, 47 priced cell games: home ATS +1.55% (|z| 0.11), home ML +3.40% (|z| 0.49). This slice cannot decide anything.
- Within-season form, 833 cell games: home ATS -5.51% (|z| 1.65), home ML -3.79% (|z| 2.15). Descriptive sensitivity.

## Model diagnostic (2025-26 holdout)

Canonical + the cell indicator, trained on 11,974 rows and scored on 1,225 (131 in the cell). Delta +0.000737 nats, |z| 2.15; indicator coefficient +0.0320. Positive control (availability on minus off): +0.009247 nats, |z| 4.93.

**The cell indicator is null in the model diagnostic: +0.000737 nats (CI +0.000056 to +0.001399), |z| 2.15. Positive, below the bar, not promoted.**

## What a null means

At 965 bets the bootstrap SE is about 3%, so the |z| > 3 gate needs an ROI near 9%. A null here rules out the large edge the spot is usually sold as. It does not rule out a 2-3% edge, and a positive point estimate below the bar is not evidence of one.
