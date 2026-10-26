# Search family: bestprice_ml

Generated 2026-10-26T10:45:05.669200+00:00. Pre-registered in `results/experiments.jsonl`.

**Question.** Does any moneyline cell cross zero at the best bookable price?

Discovery **11,902** priced games (2015-16..2024-25); holdout **1,204** (2025-26, odds_ml_bestbooks (OddsPortal best-of-books)). 14 arms; multiplicity bar **|z| > 3.00** (Bonferroni, floor 3.0).

| arm | n disc | ROI disc | \|z\| | cover disc | n hold | ROI hold | \|z\| | cover hold | verdict |
|---|---|---|---|---|---|---|---|---|---|
| ML favourite 0.50-0.55 (best price) | 1,668 | +1.49% | 0.64 | 53.9% | 143 | -14.82% | 1.81 | 44.8% | no evidence |
| ML underdog 0.50-0.55 (best price) | 1,668 | -3.76% | 1.46 | 46.1% | 143 | +16.43% | 1.82 | 55.2% | no evidence |
| ML favourite 0.55-0.60 (best price) | 1,820 | -1.99% | 1.00 | 57.0% | 183 | -1.32% | 0.21 | 56.8% | no evidence |
| ML underdog 0.55-0.60 (best price) | 1,820 | +0.07% | 0.03 | 43.0% | 183 | +1.12% | 0.13 | 43.2% | same sign, underpowered |
| ML favourite 0.60-0.65 (best price) | 1,868 | -3.11% | 1.69 | 61.1% | 157 | +0.82% | 0.13 | 63.1% | no evidence |
| ML underdog 0.60-0.65 (best price) | 1,868 | +2.47% | 0.81 | 38.9% | 157 | -2.09% | 0.20 | 36.9% | no evidence |
| ML favourite 0.65-0.70 (best price) | 1,648 | -1.26% | 0.76 | 67.5% | 164 | +1.35% | 0.26 | 68.9% | no evidence |
| ML underdog 0.65-0.70 (best price) | 1,648 | -1.26% | 0.37 | 32.5% | 164 | -5.16% | 0.48 | 31.1% | no evidence |
| ML favourite 0.70-0.75 (best price) | 1,417 | -1.31% | 0.83 | 72.3% | 146 | +4.88% | 1.07 | 76.7% | no evidence |
| ML underdog 0.70-0.75 (best price) | 1,417 | +1.18% | 0.26 | 27.7% | 146 | -16.10% | 1.33 | 23.3% | no evidence |
| ML favourite 0.75-0.80 (best price) | 1,168 | -1.17% | 0.70 | 77.1% | 122 | -1.21% | 0.25 | 77.0% | no evidence |
| ML underdog 0.75-0.80 (best price) | 1,168 | +3.00% | 0.50 | 22.9% | 122 | +1.32% | 0.08 | 23.0% | same sign, underpowered |
| ML favourite 0.80-1.01 (best price) | 2,313 | -3.03% | 3.42 | 84.0% | 289 | -2.33% | 0.98 | 85.5% | no evidence |
| ML underdog 0.80-1.01 (best price) | 2,313 | +20.41% | 3.29 | 16.0% | 289 | -3.94% | 0.28 | 14.5% | discovery-only |

## Reading the table

`cover` is wins divided by settled bets; pushes are excluded from the denominator but kept in `n` at zero profit. A cell is a *candidate* only when discovery is positive above the multiplicity bar. A *promoted* rule needs the holdout positive as well; with a one-season holdout that is a high bar, and a null holdout does not rescue a discovery cell. Cells below the bar are noise, including the positive ones.

## Frame

`build_games()` 13,199 regular-season games; **13,106** joined to a closing line; discovery 11,902, holdout 1,204. Holdout source: ``.

