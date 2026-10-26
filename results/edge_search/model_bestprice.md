# Search family: model_bestprice

Generated 2026-10-26T11:13:16.327157+00:00. Pre-registered in `results/experiments.jsonl`.

**Question.** Does the repo's model cross zero at the best bookable price?

Discovery **5,326** priced games (2015-16..2024-25); holdout **1,204** (2025-26, odds_ml_bestbooks x closing_line.model_probs). 4 arms; multiplicity bar **|z| > 3.00** (Bonferroni, floor 3.0).

| arm | n disc | ROI disc | \|z\| | cover disc | n hold | ROI hold | \|z\| | cover hold | verdict |
|---|---|---|---|---|---|---|---|---|---|
| base model, all rows | 5,326 | +6.02% | 2.29 | 40.7% | 1,204 | -3.45% | 0.78 | 39.7% | no evidence |
| base model, coherent rows only | 4,481 | -2.33% | 1.08 | 40.4% | 923 | -4.00% | 0.83 | 40.1% | no evidence |
| model + availability, all rows | 5,326 | +6.77% | 2.61 | 41.6% | 1,204 | -0.21% | 0.05 | 41.4% | no evidence |
| model + availability, coherent rows only | 4,481 | -1.46% | 0.68 | 41.4% | 923 | -2.01% | 0.42 | 41.6% | no evidence |

## Reading the table

`cover` is wins divided by settled bets; pushes are excluded from the denominator but kept in `n` at zero profit. A cell is a *candidate* only when discovery is positive above the multiplicity bar. A *promoted* rule needs the holdout positive as well; with a one-season holdout that is a high bar, and a null holdout does not rescue a discovery cell. Cells below the bar are noise, including the positive ones.

## Frame

`build_games()` 13,199 regular-season games; **13,106** joined to a closing line; discovery 5,326, holdout 1,204. Holdout source: ``.

