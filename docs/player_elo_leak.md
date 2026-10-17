# The player-Elo block: a strong result that was a leak

**Repo:** `/home/Joel/.cache/nba-fix-worktree` (branch `improve-model-v2`)
**Reproduce:** `python scripts/feature_block_ablation.py --blocks player_elo`
**Summary table in the README:** [the player-Elo block](../README.md#the-player-elo-block-a-strong-result-that-was-a-leak)

This is the single most instructive result in this repository, and the reason is
its *shape* rather than its number. A feature block measured |z| = 5.18 — the
largest effect ever found here, past this project's bar of 3, past Bonferroni for
eleven looks, larger than every real effect in the repo combined — and it was
target leakage. The honest version of the same block is +0.00065 and does not
earn its place in the model.

Nothing here is hypothetical: the number was in the README's own results table
before this page existed.

## The measurement

Per-player Elo ratings, totalled over each team's expected-available roster, were
built as a feature block and scored. The first measurement was:

| roster construction | Δ log loss | 95% CI | \|z\| | verdict |
|---|---|---|---|---|
| **over the actual participants** | **−0.00645** | [−0.00890, −0.00402] | **5.18** | the largest effect ever measured here |
| over the pre-game roster | +0.00065 | [+0.00007, +0.00126] | 2.14 | **does not earn its place** |

**That first row was target leakage, and it was nearly published as a result.**
|z| = 5.18 clears this project's bar of 3, clears Bonferroni for eleven looks,
and is larger than every real effect in this repository combined. It came from
summing `rating × expected_minutes` over the players who **actually played**.

The *weights* were honest — trailing EWMAs, strictly causal, verified by truncating
the future. The leak was in the **membership**. Nobody knows before tip-off which
ten men will be on the floor, so a sum over the actual lineup tells the model who
played. It is the same failure mode as substituting actual minutes for expected
ones, which is worth +0.01089 against an honest +0.00382; the size of the
fabrication here sits exactly in that pattern, which is what made it worth
suspecting.

The fix rebuilds the roster from the past: for team X's game *t*, the available
roster is every player with an appearance for X **strictly before t**, carrying the
rating and trailing minutes from that last appearance. A player who misses the
game is still in the sum, discounted by his own trailing minutes — which is the
entire point of the block. Reproduce both numbers with
`python scripts/feature_block_ablation.py --blocks player_elo`.

## Why truncating the future does not catch this

The standard leak guard in this repo deletes future games and checks that no
historical feature moves. That test passes here: the rating weights are trailing
EWMAs and genuinely causal. The leak lives in *which players are in the sum*, and
deleting the future does not change who is in the sum — the lineup is in the
feature because it was in the frame, not because the weights saw the result.

So `Tests/test_player_elo.py` asserts something else entirely: that the raw total
(`elo_available`) is **inverted** relative to winning. See the bucketing below —
the diagnosis, not the arithmetic, is what the test pins.

## Bucketing the contrast explains the mechanism, without needing the model

`scripts/elo_buckets.py` removes the regression entirely: it takes the raw
home-minus-away Elo contrast, splits games into ten equal-population buckets, and
reports the actual home-win rate in each. Three columns, three different answers.

| contrast | Spearman ρ vs bucket | win rate, low → high bucket | spread / noise | market's own spread |
|---|---|---|---|---|
| `elo_mean` (scale-free) | **+0.85** | 0.535 → 0.622 | 6.7× | 0.099 |
| `elo_available` (raw total) | **−0.96** | 0.624 → 0.483 | 10.3× | 0.111 |
| `elo_top5` | +0.17 | 0.537 → 0.591 | 4.0× | 0.035 |

Read those three rows together:

1. **`elo_available` runs backwards.** The stronger a team's accumulated
   rating-minutes looks, the *less* it wins. It is not a quality measure at all —
   it is dominated by **departed** players, whose trailing EWMA decays without ever
   reaching zero, so a club that recently traded a star keeps carrying his weight
   while a club that just acquired one is under-counted. It is a lagging indicator
   of roster churn, and it is the column that made the leaky version look like a
   breakthrough.
2. **`elo_mean` is correctly ordered** — and its spread (0.091) is *narrower* than
   the closing market's spread on the same buckets (0.099). A feature that orders
   games slightly worse than the price already does is a real feature and **not a
   betting edge**, which is the whole conclusion of this project arriving from a
   completely new direction.
3. **`elo_top5` is nearly useless** (ρ = +0.17), which is expected: it is a crude
   proxy for a starting five that cannot notice a starter sitting out.

The bucket at zero contrast sits at **0.5585**, not 0.500 — correctly above a coin
flip, because an Elo difference says nothing about home court.

Reproduce with `python scripts/elo_buckets.py`.

## Roster assumptions are a parameter, not a detail

"Available" is not a fact a pre-game model can look up. It is an assumption, and
different assumptions give materially different team totals, so the assumption is a
parameter of `nba_data.player_elo_features` rather than a line of code:

| `roster_mode` | who counts as available |
|---|---|
| `all_prior` | every player with an appearance strictly before the game |
| `prev_game` | only the squad that played the team's previous game |
| `top_minutes` | the trailing-minutes leaders among those |
| `confirmed` | the real submitted starting five — **not computable here** |

The absolute totals are *not* comparable across constructions: `all_prior` sums
over roughly 6.4× a real rotation, because a long-departed player's trailing EWMA
decays without ever reaching zero. Only `elo_mean` is scale-comparable, and only
the predictive value of any of them is comparable at all — which is what
`scripts/feature_block_ablation.py` scores.

`scripts/elo_roster_variants.py` builds every computable variant so the numbers
sit side by side. The `elo_mean` values across `all_prior`, `prev_game` and
`top_minutes` agree to within a point (1546.4 / 1546.7 / 1548.0) even though their
raw totals differ by 10×.

`confirmed` is the fourth answer and it stays unfilled on purpose. Starting lineups
are filed **30–60 minutes before tip-off** — legitimate for a tip-off prediction
and impossible for a model run that morning. That is a prediction-point decision,
not a data gap, so it is recorded as a parameter rather than a feature that
quietly becomes unavailable.

## What is left

The block is retained, behind the `roster_mode` parameter, because a measured null
with a diagnosed mechanism is worth more than a deleted function. The lesson is
the shape, and it is worth stating plainly: **a large effect that clears every
multiple-testing bar you have can still be a leak, and the way to tell is to ask
what the feature is a function of — not how well it predicts.** A sum over "the
players who played" is a function of the answer.

## Where this shows up elsewhere

- [Limitations](../README.md#limitations) — effect sizes here are of order
  0.001–0.03 nats, and the honest bar is |z| > 3.
- [Against other repos](../README.md#against-other-repos) — the survey's
  ~45-repo accuracy ceiling is the same story from the outside.
- `docs/modeling_exploration.md` — further candidate blocks built from tables
  nothing had read, all null.