# Reading the results

**Repo:** `/home/Joel/.cache/nba-fix-worktree` (branch `improve-model-v2`)

The tables live in the README; this is the reasoning behind them. Every figure here
is already printed by a module or script — nothing here is a claim you cannot re-run.

```bash
python -m src.Models.baselines        # Elo, SRS, Bradley-Terry, base rate
python -m src.Models.logreg_rolling    # the canonical model
python -m src.Models.gbm               # boosting, paired against the above
python -m src.Models.margin_total      # margin and total, then Φ(m̂/σ)
python -m src.Models.player_ratings    # player-level ratings
python scripts/run_experiments.py --stage 1
python scripts/feature_block_ablation.py --seed 42 --splits 5
python scripts/closing_line.py
python scripts/edge_betting.py
```

## Why the ranking below the top row is the wrong thing to read

All models are scored on the **same 6,600 walk-forward test rows** from the same five
splits. That is what makes the table comparable at all, and it is why the four
results below matter more than the ordering of the middle of the table.

## Boosting loses, and the interval proves it is not a tie

Against the canonical model on identical rows and identical features — `gbm.py` calls
`logreg_rolling.load_design(halflife=10.0)` for its reference frame, so the algorithm
is the only variable:

| model | Δ log loss | 95% CI | verdict |
|---|---|---|---|
| XGBoost | −0.00394 | [−0.00557, −0.00229] | worse, significant |
| HistGradientBoosting | −0.00431 | [−0.00601, −0.00255] | worse, significant |

Δ = loss(logreg) − loss(model), so negative means the booster is worse. The two GBMs
agree with each other to within 0.0004, so this is not an implementation artifact.

The project spec expected boosting to win by a small margin. It does not, and the
interval is not straddling zero, so this is a loss and not a tie.

## More complex targets do not pay

Predicting margin and total, then converting with Φ(m̂/σ), reaches 0.6204 —
indistinguishable from fitting the label directly on the same features (Δ +0.0006,
CI [−0.0026, +0.0038]). Its value is the richer output: margin RMSE 14.01, total
RMSE 18.78, spread MAE 11.0.

## Batch rating systems lose to the base rate

Bradley-Terry (0.6929) and SRS (0.6890) both score *worse* than always predicting home
(0.6883). The cause is structural, not a tuning failure: a batch fit over five seasons
reports where a team **was**, while the next game depends on where it **is**. Removing
Elo's movement term costs 0.0083 nats — more than the entire gap between the best
model and the base rate.

## Player ratings are a null

The project spec calls player ratings "likely your biggest edge". It was built to be
falsifiable rather than to look good, and it was falsified. Δ = loss(team) −
loss(team + player), so **positive would mean the player features helped**:

| estimate | Δ log loss | 95% CI | verdict |
|---|---|---|---|
| RAPM-style, two-pool | +0.00151 | [−0.00059, +0.00358] | null |
| RAPM-style, shrunk | +0.00038 | [−0.00178, +0.00260] | null |
| logistic | +0.00060 | [−0.00154, +0.00282] | null |
| gradient boosting | −0.00243 | [−0.00815, +0.00295] | null |
| XGBoost | −0.00614 | [−0.01273, +0.00062] | null |

Every interval straddles zero, across both estimation methods and all three rating
families.

## The one exception, and it is the important one

The same module measures the *availability gap* the honest way. Substituting **leaked**
actual minutes for expected ones is worth −0.00975 nats, CI [−0.01349, −0.00590] —
an order of magnitude larger than anything else in the table. The honest version of
that mechanism is +0.00382, and the third of it that survives is the project's only
real effect. See [the one thing that is not a null](../README.md#the-one-thing-that-is-not-a-null).

## Why the winner's margin is not an effect

Best − median across the 17 pre-registered runs is −0.00026 nats, and the top five
are separated by less than 0.0007. A best-of-N is the maximum of N draws; the gap
between best and median is the expected size of that maximum, which is exactly what
`scripts/run_experiments.py` prints. The clear signals in that table are the
*direction and size* of the three tree models and the E10 ablation, not the ordering
among the linear models.

## Why the feature work made the market gap worse

Adding the per-team home/away split improved the *canonical* model from 0.61956 to
0.61897, but `scripts/closing_line.py`'s own scorer went from 0.62693 to 0.62581 for
availability and the plain model went 0.62903 → 0.62777, leaving the gap at 0.02641
against 0.02753 before. The split did not help *here*.

That is not a contradiction. `scripts/closing_line.py` builds its **own** feature frame
rather than calling `logreg_rolling.load_design`, so it feeds the raw per-side columns
instead of the contrasts the canonical model uses, and the sixteen new columns arrive
without the differencing that makes them informative. It is also the fourth time in
this repo that a duplicated feature definition drifted independently — after the drop
lists (#14, #28), the availability block (#30) and `margin_sweep`.

The honest summary is that the split is worth about 0.0006 nats on the model that
differences its features, and nothing measurable on the one that does not. Either way
the market is 0.026 nats ahead — roughly **7×** the availability effect, the best
prediction improvement this project has found, and about **30×** the split's own.

## Why both sides are considered when betting

A model does not have to be right every game to make money; it has to be right where
it disagrees with the price. The rule has **no free parameter**, which is the point:
every threshold sweep is a multiple-comparisons problem, and this one is fixed by
arithmetic. Bet one unit wherever the model's probability for a side exceeds the
probability implied by the price actually posted; expected profit per unit is
`p·(1/p_raw − 1) − (1 − p) = p/p_raw − 1`, positive exactly when `p > p_raw`.

Restricting to the model's preferred side would silently skip every game where the
favourite has no edge but the underdog does — 1,928 of them, more than half the bets
placed. That is what "lay a loser" means in a two-way market, and excluding it is not
a shortcut, it is skipping the question.

The comparison is against the **raw** price, not the de-vigged one, because the
**+3.98%** overround (p5–p95 2.96%–5.00%) is what a bettor actually pays.

## The control, and what it caught

Running the same rule on the market's own de-vigged probability places **zero bets** —
correct, because de-vigging divides the overround away. An earlier version of the
script priced the away side as `1 − p_raw_home` instead of reading it from the data,
which understated it by the vig, made every away bet look better than it was, and made
that control report a spurious **+7.00%** profit. A control that profits when it
should place no bets is a broken price, not an edge.

Full post-mortem: [the ROI denominator bug](measurement_bugs.md).

## Two independent routes to the same conclusion

The player-Elo bucket analysis reaches the market verdict from a completely different
direction. `elo_mean` orders games correctly — decile home-win rate rises
monotonically from 0.535 to 0.622, ρ = +0.85 — yet its spread across those buckets is
**0.091, against the closing market's 0.099 on the very same games.**

That agreement is worth more than either result alone. A feature that orders games
marginally worse than the price already does is a real feature and **not** a betting
edge. Full write-up: [the player-Elo leak](player_elo_leak.md).

## Why "mostly null" is the finding, not a failure

Nearly every mechanism measured here came back flat: player ratings across 2 methods ×
3 families, 52 rank columns, the rebound block, recency weighting, three never-read
`OPP_*` columns, and ten further candidate blocks in
[docs/modeling_exploration.md](modeling_exploration.md). The correlation screen is
what makes that a finding rather than an opinion about ideas — trailing margin
volatility correlates 0.04 with the features that already carry the signal, the
last-2-vs-last-10 margin contrast 0.06, and they still predict nothing.

The project's own bar is |z| > 3 (Vandenbruaene et al. 2022, *J Sports Econ* 23(7),
where the null of unprofitability survived across 628 implementations), and where ten
blocks are scored at once a Bonferroni threshold of 0.005 applies on top. Exactly
one result in this repository clears the bar, and it was a leak. That is worth
saying plainly rather than quietly rounding in our own favour.