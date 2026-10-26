# Issue #81 - layered Elo + recent form, priced

Pre-registered as `I81-prereg-layered-elo-form`. Generated 2026-10-26T06:27:39.970717+00:00.

The layer: a pre-game Elo gap of at least **100 points**, bet the higher-rated side, restricted to games where that side is **6-4 or better over its last 10** and the other side is **3-7 or worse**, both with at least 10 prior games.

Priced games: **12,385**. Elo layer **6,735**; form layer **2,371**; intersection **1,886**.

## Primary (only promotion gate)

| n | ROI | SE | \|z\| | 95% CI |
|---|---|---|---|---|
| 1,886 | -4.42% | 2.19% | 2.01 | -8.86% to -0.06% |

**The layered cell loses at the price. The Elo layer selects the favourite and the form layer re-reads what Elo already absorbed; the stack does not create a price. ROI -4.42%, |z| 2.01, n 1886.**

## Moneyline (descriptive)

| n | ROI | SE | \|z\| | 95% CI |
|---|---|---|---|---|
| 1,886 | -4.06% | 1.23% | 3.32 | -6.54% to -1.70% |

## Venue split (descriptive)

| high side | n | ROI | SE | \|z\| | 95% CI |
|---|---|---|---|---|---|
| home | 1,147 | -5.81% | 2.78% | 2.09 | -11.04% to -0.07% |
| away | 739 | -2.25% | 3.48% | 0.65 | -9.20% to +4.76% |

## Elo ladder, form layer held fixed (descriptive)

| gap | n | ATS ROI | \|z\| | ML ROI | \|z\| |
|---|---|---|---|---|---|
| >= 50 | 2,178 | -4.34% | 2.14 | -4.40% | 3.66 |
| >= 75 | 2,040 | -4.28% | 2.02 | -4.26% | 3.54 |
| >= 100 | 1,886 | -4.42% | 2.01 | -4.06% | 3.32 |
| >= 125 | 1,731 | -4.19% | 1.87 | -3.88% | 3.12 |
| >= 150 | 1,547 | -4.95% | 2.07 | -3.84% | 3.08 |

## 11-game form sensitivity (descriptive)

| n | ROI | SE | \|z\| | 95% CI |
|---|---|---|---|---|
| 1,885 | -5.28% | 2.19% | 2.41 | -9.49% to -0.90% |

## Four form sub-cells inside the Elo filter (descriptive)

| cell | n | ATS ROI | SE | \|z\| |
|---|---|---|---|---|
| hot cold | 1,893 | -4.47% | 2.17% | 2.06 |
| hot not cold | 2,515 | -3.78% | 1.85% | 2.05 |
| not hot cold | 1,295 | -5.51% | 2.67% | 2.06 |
| neither | 1,032 | -9.01% | 2.90% | 3.11 |

## Per season (descriptive)

| season | n | ATS ROI | \|z\| |
|---|---|---|---|
| 2015-16 | 145 | -11.72% | 1.52 |
| 2016-17 | 190 | -14.98% | 2.21 |
| 2017-18 | 200 | -0.59% | 0.09 |
| 2018-19 | 190 | -6.03% | 0.87 |
| 2019-20 | 179 | -0.15% | 0.02 |
| 2020-21 | 134 | -7.33% | 0.87 |
| 2021-22 | 206 | -0.26% | 0.04 |
| 2022-23 | 136 | -7.95% | 1.01 |
| 2023-24 | 220 | -0.17% | 0.03 |
| 2024-25 | 194 | +0.42% | 0.06 |
| 2025-26 | 92 | -4.55% | 0.45 |

## Reference: back the favourite, no filter

n 12,385, ROI -3.66%, |z| 4.31. This is the arm the Elo layer mostly reproduces.

## What the gate could and could not see

At n = 1,886 the bootstrap SE is 2.19%, so |z| > 3 needs an ROI near 6.57%. A null rules out a large edge; it does not rule out a 2% one.

