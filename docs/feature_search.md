# The feature search: four candidates, pre-registered

**Repo:** `/home/Joel/.cache/nba-fix-worktree` (branch `improve-model-v2`)
**Reproduce:**

```bash
python scripts/feature_block_ablation.py --seed 42  --splits 5
python scripts/feature_block_ablation.py --seed 137 --splits 5
python scripts/feature_block_ablation.py --seed 2718 --splits 5
python scripts/feature_block_ablation.py --seed 42 --splits 8
```

The results table is in the README: [the feature search](../README.md#the-feature-search-four-candidates-pre-registered).
This page is the narrative behind it — why these four, what each one turned out to
be, and the two ways a candidate can fool you.

## Why four, and why written down first

Every decay experiment in this project failed, because all of them reweight the
*same* numbers the model already has. Faster and slower windows, ranks, four
factors, rebounds — all transformations. So the next round had to bring
information the frame did not already contain, and the four candidates were
**written into the block registry before any was measured**, so that the results
cannot be cherry-picked out of a longer search.

Δ = loss(full) − loss(minus block), so **negative means the block earns its
place**.

| candidate | the new information | Δ | 95% CI | \|z\| | verdict |
|---|---|---|---|---|---|
| opponent-adjusted form | beating a good club counts for less | −0.00004 | [−0.00014, +0.00005] | 0.92 | null |
| **home/away split** | **each team's *own* home advantage** | **−0.00089** | **[−0.00167, −0.00015]** | **2.29** | **replicated** |
| strength of schedule | quality of opposition faced | −0.00008 | [−0.00015, −0.00000] | 2.01 | **died in replication** |
| opponent characteristics | three `OPP_*` columns never read | +0.00000 | [−0.00061, +0.00060] | 0.01 | null |

## One of four is real, and it is small

The per-team home/away split is the only candidate that replicated. Measured
−0.00089 nats at three bootstrap seeds (−0.00089 [−0.00161, −0.00011] and
[−0.00166, −0.00017]) and at eight folds instead of five (−0.00089
[−0.00167, −0.00013]) — the point estimate does not move. It is worth having, and
it is in the canonical model, where it moves log loss from 0.61956 to **0.61897**.

**It is also plainly below the bar, and both readings matter.** |z| ≈ 2.3 does not
clear |z| > 3, and one-sided p ≈ 0.011 does not clear a Bonferroni threshold of
0.005 for ten looks. And the size does not matter: 0.00089 nats is under a
quarter of the availability effect and about a thirtieth of what the closing
market beats this model by. It is a real feature, reported at its real size.

### One caveat on how to read the split columns

They are built by blanking the away rows and passing the column through the shared
EWMA, and pandas' `ewm` decays across NaN positions. So `ewm_margin_home` is not
an EWMA over home games with halflife 10 — its halflife counts *all* of a team's
games, making the effective window about 20 home games. It is a defensible choice,
but it means the split columns are noisier than a reader would assume (sd 6.29 and
5.84 against 5.46 for the pooled `ewm_margin`), which is part of why the gain is
small. Documented in the function.

## Strength of schedule is the instructive failure

At five folds it excluded zero and the script called it a keeper — −0.00008,
|z| = 2.01. At eight folds it became +0.00000 [−0.00008, +0.00009], |z| = 0.11,
**a clean null with the sign flipped**. A result that exists only at one fold count
is a property of the fold count.

The script's own verdict text says an exclusion is a candidate for a second angle
rather than a settled result; this is what following that advice looks like, and
it is why that sentence is in the code. It is also the cheapest replication in the
whole project — one flag — and it is the reason `--seed` and `--splits` exist.

## Opponent-adjusted form was a null, exactly as predicted

Its correlation with the `ewm_margin` it adjusts is 0.98 — it is largely a
rearrangement of information the frame already had, written down as a prior before
measuring. The prior was right. What made `home/away split` different is visible
in the same diagnostic: `home_edge` correlates **0.08** with overall form, so it is
close to orthogonal to everything else in the matrix. New information and a
rearrangement look identical in a column list; only the correlation distinguishes
them, and only after the fact.

## Opponent characteristics is the silent-absence bug again

`OPP_FTA_RATE`, `OPP_TOV_PCT` and `OPP_OREB_PCT` sit in the same table, under the
same header, in every downloaded season — and no model in this repo had ever read
them. Only `OPP_EFG_PCT` was used. It is structurally the same defect as the
rebound percentages that were missing from every model until
`src/Utils/features.py` was asked to name them, and the result is the same:
+0.00000. They are now in the frame and in the registry, so they are measured
rather than invisible.

## Two ways a candidate can fool you

Both of the failures above are the same mistake in opposite directions, and they
are worth separating because the second one is much easier to miss:

1. **A fold-count artifact.** Strength of schedule looked real at five folds and
   was a clean null at eight, with the sign flipped. The interval never moved
   outside zero; the *conclusion* did.
2. **An unwritten duplicate.** Opponent-adjusted form was new-looking, and it was
   a rearrangement of `ewm_margin` at r = 0.98. Nothing about the column list
   reveals this; only the correlation screen does.

The third case, and the most expensive, is a feature that is a function of the
answer rather than a rearrangement of it — see
[the player-Elo leak](player_elo_leak.md).