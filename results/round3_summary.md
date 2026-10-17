# Round 3 — team Elo, squad quality, and what the sweep actually ruled out

Date: 2026-10-18. Branch `improve-model-v2`. Agent `nba-swarm-51f3a032`.
All numbers are out-of-fold on the project's fixed protocol: 5 walk-forward
blocks, `initial_frac=0.5`, `gap=0`, seed 42, `nd.evaluate` +
`nd.paired_bootstrap_delta(n_boot=2000)`. 6,600 scored rows of 13,199 games.

**Headline.** `E14 (43 cols) + availability (4) + elo_diff (1) = 48 columns`
scores **log loss 0.612523**, AUC 0.717462. That is **+0.005784 nats against the
stated E14 bar of 0.618307, |z| 2.60**, which clears the Bonferroni bar for a
4-arm family (2.535) and the project's own |z| > 3 convention is reported
alongside and is *not* cleared. The honest caveat is in the next section.

---

## 1. The number, and the caveat that has to travel with it

| model | cols | log loss | brier | AUC | acc |
|---|---|---|---|---|---|
| E14 `home_split_only` (the bar) | 43 | 0.618307 | 0.214601 | 0.709251 | 0.655455 |
| + availability | 47 | 0.614544 | 0.212880 | 0.714692 | 0.659394 |
| **+ availability + elo_diff** | **48** | **0.612523** | **0.212050** | **0.717462** | **0.664091** |
| + availability + elo_diff + squad-quality contrasts | 73 | 0.611449 | 0.211656 | 0.718092 | 0.666970 |

Against the bar the new model is `+0.005784`, 95% CI `[-0.007954, -0.003497]`
in `loss(cand) − loss(base)` terms, **|z| 2.596**.

Against the **availability model** — which is the best model the project already
had — the marginal gain is `+0.002021`, **|z| 1.125**. That does not clear any
bar. Read it this way:

* The claim that is supported: *the 48-column model beats the stated bar.*
* The claim that is **not** supported: *Elo adds +0.0058 on top of what was
  already known.* About two thirds of that improvement was availability's and
  was measured in round 2. Elo's own marginal contribution is +0.0020.

Both numbers are in `results/round3_experiments.jsonl` under the same run id.
Anyone quoting one without the other is overstating the work.

**Fold stability.** The walk-forward has five disjoint test blocks and a pooled
bootstrap CI over 6,600 heavily-overlapping games can be narrow for the wrong
reason. Per-fold log-loss deltas against the availability model:

| fold | n | delta (negative = better) |
|---|---|---|
| 0 | 1320 | −0.001057 |
| 1 | 1320 | −0.003875 |
| 2 | 1320 | −0.003709 |
| 3 | 1320 | **+0.000055** |
| 4 | 1320 | −0.001520 |

4/5 folds better; fold 3 is a fifth of a micro-nat, i.e. zero, not a reversal.
Fold-to-fold sd of the delta is 0.001716 against a pooled delta of 0.002021.
The gain is *of the same order as the fold spread*, which is why it is reported
as suggestive-on-top-of-availability and significant against the bar rather than
as a standalone discovery.

## 2. Why team Elo — and a correction to why I originally said so

`results/feature_block_ablation.json` names thirteen feature blocks. **None of
them is a rating.** Every strength estimate in the frame is a linear
exponentially-weighted mean of the last ~10 box scores. Elo is a
margin-weighted recursive update with its own forgetting schedule
(`decay_half_life_days`, `season_reversion=0.75`). Different estimator, same
underlying games.

### The correction, which matters more than the discovery

My first write-up justified keeping `elo_diff` on redundancy grounds: "max |r|
against the 43 E14 columns is 0.6363, therefore it is not a rearrangement."
**That justification was false**, and a peer caught it (`d84d8cf7`: measured
0.90–0.93). `scripts/round3_corr_dispute.py` reconciles both numbers. They are
correlations against different targets:

| K | max \|r\| vs a single E14 **level** | \|r\| vs the margin **contrast** `home_ewm_margin − away_ewm_margin` | partial \|r\| given that contrast |
|---|---|---|---|
| 5 | 0.5979 | 0.8364 | 0.0026 |
| **10** | **0.6363** | **0.8900** | **0.0004** |
| 15 | 0.6527 | 0.9118 | 0.0030 |
| 20 | 0.6608 | 0.9221 | 0.0060 |
| 30 | 0.6667 | 0.9287 | 0.0109 |
| 40 | 0.6666 | 0.9276 | 0.0133 |

The margin contrast is not itself a column in the frame, which is why a
column-by-column scan never saw it. **Against the right target, `elo_diff` is
0.89-correlated with the frame's margin contrast — above the repo's own 0.8
"do not bother" threshold — and its residual after that contrast is |r| =
0.0004, i.e. nothing.** The peer's measurement was right; my reason was wrong.

### The reason that actually survives

The measured +0.002021 marginal over availability is unaffected, and it does not
depend on orthogonality. `elo_diff` and the frame's margin contrast are two
estimates of the same quantity under **different weights**: Elo weights each
game by margin of victory and updates recursively with its own forgetting
schedule; the frame takes a linear EWM of recent margins. A linear model fitted
on the frame's version puts one fixed coefficient on "how strong is this team";
adding a second, differently-weighted estimate of the same gap lets it fit that
term at two slopes instead of one. Collinearity removes the *variance* of an
extra column, not its *coefficient*.

The evidence for that is the delta, the K-sweep and the fold table below — not a
correlation, and this document no longer offers it as one. What is lost is a
cleaner story than the honest one: this is a better-weighted duplicate, not a new
information source.

The same scan also shows what Elo is worth as a *level*: `home_elo` and
`away_elo` correlate 0.89 with `home_ewm_margin` / `away_ewm_margin` and the
`+ home_elo + away_elo` arm scored 0.616683 against the bare bar, +0.00162 at
|z| 0.89. The usable part of a rating in this frame is one number, not two.

`elo_diff_adj = elo_diff + 70` is not separable from `elo_diff` in a model with
an intercept, so the promoted frame drops it. An earlier record (`T3`) described
the candidate as "3 Elo contrast columns"; it was two columns carrying **one**
degree of freedom. `V1_final_48col` carries the corrected statement and the
machine-checked collinearity table. The 48-column frame has **47 effective
d.o.f.** — `away_expected_minutes_diff` is exactly `−1 ×`
`home_expected_minutes_diff`.

**Replication.** Peer `0a2786ed` rebuilt the promoted frame from
`build_games()` upward with no peer caches and got every number: bar 0.618307,
+availability 0.614544, +elo_diff alone 0.616720, +availability+elo_diff
0.612523, marginal +0.002021 at |z| 1.125 in 4/5 folds.

## 3. Gates that were run before any of this was scored

| gate | result |
|---|---|
| `P1_truncation` on `feat_team_elo` | 7,061 rows compared at cut 2021-10-20, **0 NaN-pattern mismatches, max_abs_diff exactly 0.0** |
| `P1_truncation` on `feat_squad_quality` | 0 NaN-pattern mismatches, max_abs_diff 1.1e-13 (float associativity) |
| `P1_truncation` on `availability_features` | 0 mismatches, 1.1e-13 |
| `P3_membership` on `feat_squad_quality` | 0 of 280,632 query rows fall outside the team's previous-game squad |
| redundancy | `NET_RATING` squad columns at \|r\| 0.92 with `ewm_net_rating` → dropped as rearrangements; everything else ≤ 0.63 |

`rating_home_before` is the state *prior* to each game's own update. Within a
test block the ratings keep updating from that block's earlier results, which is
legitimate under this project's deployment convention and is the **same**
convention the E14 EWMs already use — `rolling_team_features` shifts over the
whole frame, so a test row already sees its block's earlier games. The two blocks
are measured on equal terms, not one generously and one strictly.

## 4. Robustness

**K.** K=10 is the value `runs/baselines_results.md` selected on the last 20% of
each training block; test rows never entered that choice. The full sweep on the
promoted 48-column configuration, all of it reported:

| K | log loss | \|z\| vs availability | \|z\| vs bar | folds better |
|---|---|---|---|---|
| 5 | 0.613115 | 0.78 | 2.28 | 3/5 |
| 10 | 0.612523 | 1.12 | 2.60 | 4/5 |
| 15 | 0.612539 | 1.25 | 2.75 | 5/5 |
| 20 | 0.612852 | 1.18 | 2.79 | 5/5 |
| 30 | 0.613620 | 0.87 | 2.64 | 5/5 |

Every K from 5 to 30 beats the bar. The result is not an artefact of the tuned K.
K=15 and K=20 are both 5/5 on folds and are marginally more stable than K=10
(fold sd 0.00115 / 0.00074 vs 0.00172); K=10 is kept because it is the value
this project already validated, and adopting K=15 on a 1.6e-5-nat difference
would be exactly the selection effect the |z| margin exists to absorb.

**Row order.** An earlier robustness check (`U4`) permuted the `games` rows and
was **invalid**: `walk_forward_split` slices by row index, so permuting broke
the chronological geometry — the absolute log loss moved 0.6125 → 0.6231 and
the "delta collapse" was measuring a broken split. It is superseded, not
reported. The correct check jitters game order *within each date* and leaves
chronology intact: the delta moves by **8e-06** (0.002021 → 0.002013). Team EWMs
sort by `(team, date)` and `fit_predict_elo` sorts internally by
`(date, home, away)` with mergesort, so order-invariance is now earned rather
than asserted.

**Determinism.** `lbfgs` with a fixed `random_state` reproduces bit-identically
across repeat runs, so there is no seed spread to quote and quoting one would be
theatre. All uncertainty here is the paired bootstrap plus the five folds.

## 5. What round 3 ruled out

| idea | result | why |
|---|---|---|
| **C sweep, 43 cols** | inner-val surfaces flat to the 4th decimal; per-fold-selected C scores 0.618975 vs the C=1.0 bar 0.618307 | C=1.0 is already at the optimum; there is no penalty to tune |
| **C sweep, 67 cols** | best C=0.3 → 0.613125 vs C=1.0 → 0.613137 | 6e-5 nats. The frame is not variance-bound in the way the wide bootstrap CI suggested |
| **availability halflife** (10-point grid, 5→60) | flat 10→60 (h25 0.614480 vs h15 0.614544, \|z\| 0.354); degrades below 10 | the untuned constant 15.0 was not a lucky pick. The *feature* is not null (+0.0038 at every halflife); the *halflife* is |
| **squad quality stacked on the winner** | 0.611449, \|z\| 1.05, **3/5 folds**, fold sd 0.00427 > pooled 0.0031 | the gain is concentrated in the 2020-23 era and reverses in 2023-25. The headline 0.6114 must not be promoted |
| **"elo_diff is not a rearrangement"** | corr 0.89 against the margin contrast, partial \|r\| 0.0004 | **my own claim, refuted by peer measurement and by `round3_corr_dispute.py`.** A column-by-column scan cannot see a contrast that is not a column. Measure redundancy against the derived quantity the feature competes with, not only against the columns that happen to be present |
| **squad quality alone** | 0.613137 vs bar +0.0052 \|z\| 1.79; vs availability +0.0014 \|z\| 0.57 | consistent sign across 4 arms, real-looking, underpowered |
| **U4 order permutation** | invalid test | permuting rows breaks `walk_forward_split`; superseded by the within-date check |

### Block Y: squad continuity and load decay — a null that settles round 2's A1

Round 2's `A1_lineup_continuity` scored −0.0117 nats and read like a refutation.
Its own coverage line says otherwise: 748 of 13,199 games (5.7%), from a
2-season `rotations/` file, of which **134** fell out-of-fold. At n=134 an effect
of availability's size would not have cleared the noise floor. That is a
coverage failure, and the hypothesis deserved a second measurement.

Rebuilt from `player_gamelogs/advanced_*.csv`, which covers all 11 seasons:

| arm | cols | log loss | vs promoted 0.612523 | \|z\| | folds |
|---|---|---|---|---|---|
| Y1 continuity contrasts | 56 | 0.612265 | +0.000258 | 0.32 | 3/5 |
| Y2 squad depth + load decay | 54 | 0.612916 | −0.000393 | 1.00 | 1/5 |
| Y3 continuity, no elo | 55 | 0.614418 | −0.001895 | 0.97 | 1/5 |

Truncation gate: 0 NaN-pattern mismatches, max_abs_diff 3.6e-15.

The null is **informative**, and it retires a heuristic several agents were
leaning on. Lineup continuity is the most orthogonal block anyone has built
here — max \|r\| **0.159** against all 43 E14 columns *and* against a battery of
seven derived contrasts the frame does not contain (margin, net rating,
off-minus-def, scoring, opp-efg, HFA split, rest) — and it contributes nothing.

So: **orthogonality is not contribution.** Availability works because of
expected *minutes*, not because it inspects a *set*. The phrase "membership is
the only live axis", which appeared in my own lane-ranking note, is wrong, and
the cheap test that refutes it is a maximally-orthogonal set feature measuring
zero.

### Block Z: L1 and elastic net — the variance lane, closed by a second method

Peer `fdd87721` closed with a diagnosis that made this the obvious last pull:
the model's fold log loss improves monotonically with training size (0.62552 →
0.59899 across folds 1→5), so it is **data-limited**, which argues for variance
reduction. An L2 penalty shrinks every coefficient by the same amount and cannot
say "this group of aliases is one column"; selection can. A 30-arm grid on the
promoted 48-column frame — `l1_ratio ∈ {1.0, 0.1, 0.25, 0.5, 0.75}` ×
`C ∈ {0.01, 0.03, 0.1, 0.3, 1, 3}` — produced:

| best arm | log loss | vs promoted | \|z\| | folds |
|---|---|---|---|---|
| Z1 `l1_ratio=1, C=1` | 0.612459 | +0.000064 | 0.285 | 3/5 |
| worst arm (`l1_ratio=1, C=0.01`) | 0.614934 | −0.002411 | 1.09 | 1/5 |

Nothing diverged, and the whole grid spans 0.0025 nats. Combined with the L2
C-sweep, **the variance lane is now closed twice, by two independent methods**:
uniform shrinkage and explicit selection. That is a conclusion, not a shrug.

> **The promoted model's residual error is bias, not variance.** Neither more
> penalty nor fewer columns touches it, so neither more data nor cleverer
> regularisation is the next win. Every remaining candidate has to be new
> information — and the new-information lanes are where the nulls are.

That reframes the whole remaining opportunity set, and it is why the next
experiments here are all about *sources* (shots, style priors, market, player
composition) rather than about the model fitted to what is already in hand.

One code defect surfaced while writing this up and is fixed: `pipeline_for`
selected `solver="saga"` only for `0 < l1_ratio < 1`, so pure L1 at exactly
`l1_ratio=1.0` was routed to `lbfgs` and raised. Block Z would have silently
lost its entire L1 arm without that.

Squad quality is logged as a **feature worth keeping in the bank**
(`src/DataProviders/feat_squad_quality.py`, leakage-clean, four arms with a
consistent sign) and **not** as a claim. Block Y below is stronger evidence
against the whole player-decomposition direction than the squad-quality null
itself, and should be read first.

## 6. Reproduce

```sh
/home/Joel/.cache/mlenv/bin/python -W ignore scripts/round3_verify.py          # gates, blocks S + T
/home/Joel/.cache/mlenv/bin/python -W ignore scripts/round3_corr_dispute.py   # the correction
/home/Joel/.cache/mlenv/bin/python -W ignore scripts/round3_run_T.py           # Elo block
/home/Joel/.cache/mlenv/bin/python -W ignore scripts/round3_final.py           # promote + K + order
/home/Joel/.cache/mlenv/bin/python -W ignore scripts/round3_run_Y.py           # squad continuity null
```

Frames are cached under `/home/Joel/.cache/nba-frames/` (not `/tmp`, which does
not survive a machine restart — it cost this round one rebuild).

`scripts/round3_common.py` re-exports the round-2 harness unchanged and asserts
the bar at `1e-6` before anything else runs. The bar reproduced exactly on every
run, including inside the `ml-experimenter` subagent, which found it
independently at `0.6183071193`.

## 7. What this is not

Nothing here is evidence of a betting edge. The market closes at 0.59940 against
this model's 0.612523; the gap narrowed, it did not close, and the E-block
diagnostic in round 2 explains why — a lagged market read correlates 0.585 with
the model's own log-odds and is largely a re-expression of the strength contrast
already present. Vandenbruaene et al. (2022, *J Sports Econ* 23(7)) surveyed 628
strategy implementations and never rejected the null of unprofitability after
multiple-testing correction. Their bar is |z| > 3. This does not claim it.