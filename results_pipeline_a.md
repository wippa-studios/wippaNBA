# Results: Pipeline A (NBA-data models)

> Model comparison, the experiment log, feature blocks and the leaks found along the way. Every number was produced by running the module or script named beside it. Part of the [Legit NBA Predictions README](../README.md).

---

## Pipeline A: model comparison

All models are scored on the **same 6,600 walk-forward test rows** from the same five
splits, so they are directly comparable. Lower log loss and Brier are better; higher
AUC is better.

```bash
python -m src.Models.baselines        # Elo, SRS, Bradley-Terry, base rate
python -m src.Models.logreg_rolling    # the canonical model
python -m src.Models.gbm               # boosting, paired against the above
python -m src.Models.nn_rolling        # feed-forward MLP, pre-registered M01
python -m src.Models.margin_total      # margin and total, then Φ(m̂/σ)
python -m src.Models.player_ratings    # player-level ratings
```

| model | log loss ↓ | Brier ↓ | AUC ↑ | accuracy |
|---|---|---|---|---|
| always predict home | 0.6883 | 0.2476 | 0.5010 | 0.5527 |
| Elo, no movement term | 0.6343 | 0.2220 | 0.6912 | 0.6394 |
| Elo (K=20, FiveThirtyEight default) | 0.6300 | 0.2187 | 0.7055 | 0.6594 |
| Elo + rest term | 0.6271 | 0.2182 | 0.7017 | 0.6529 |
| Elo (K=10, tuned on validation) | 0.6260 | 0.2180 | 0.7020 | 0.6512 |
| SRS + intercept (frozen) | 0.6923 | 0.2495 | 0.4867 | 0.5503 |
| SRS + intercept (walk-forward) | 0.6890 | 0.2479 | 0.5020 | 0.5500 |
| Bradley-Terry (walk-forward) | 0.6929 | 0.2494 | 0.5514 | 0.5568 |
| log-linear on Elo | 0.6234 | 0.2169 | 0.7026 | 0.6542 |
| margin + total → Φ(m̂/σ) | 0.6204 | 0.2156 | 0.7074 | 0.6562 |
| XGBoost | 0.6233 | 0.2167 | 0.7030 | 0.6562 |
| HistGradientBoosting | 0.6236 | 0.2169 | 0.7026 | 0.6526 |
| **logistic regression (canonical, + availability, 26 feats)** | **0.6157** | **0.2133** | **0.7141** | **0.6573** |
| neural network (MLP, five-seed average, 26 feats) | 0.6167 | 0.2138 | 0.7122 | 0.6617 |
| logistic regression (availability block off — the pre-#36 canonical) | 0.6196 | 0.2151 | 0.7085 | 0.6564 |

**The canonical row is post-#36** (`python -m src.Models.logreg_rolling`, 6,600
pooled out-of-fold rows, five splits): the availability block is in the model —
22 → 26 features — and log loss is **0.6157** (0.61566). The row beneath it is the
same model with `load_design(with_availability=False)`: the pre-#36 canonical,
preserved bit-for-bit at **0.6196** (0.61957), and still the frame every pre-#36
ablation is read against. The value this table carried before issue #36 —
0.6190 (Brier 0.2148, AUC 0.7093, accuracy 0.6567) — is kept as the measurement it
was and not restated; likewise the experiment log's E00 **0.61915** and E14
**0.61831** below, which are experiment-log-frame measurements — E14's is the
43-column `home_split_only` frame, not a stale version of the canonical number.

**Boosting loses, significantly.** Δ = loss(logreg) − loss(model), so negative means
the booster is worse; the two GBMs agree to within 0.0001, so this is not an
implementation artifact. Measured by `python -m src.Models.gbm`, re-run 2026-10-23
and paired against the **post-#36 canonical** (pooled log loss 0.61566):

| model | Δ log loss | 95% CI | verdict |
|---|---|---|---|
| XGBoost | −0.00785 | [−0.01038, −0.00533] | worse, significant |
| HistGradientBoosting | −0.00793 | [−0.01046, −0.00536] | worse, significant |

The sign is stable in every split (XGBoost −0.0102, −0.0013, −0.0122, −0.0055,
−0.0100; HistGB −0.0107, −0.0005, −0.0126, −0.0052, −0.0106). Two superseded
figures are kept as the measurements they were: against the **pre-#36** canonical
the same comparison read **−0.00394** [−0.00557, −0.00229] and **−0.00431**
[−0.00601, −0.00255], and the boosters' own pooled values re-measure at 0.6235 /
0.6236 (the XGBoost row in the table above is the earlier 0.6233 reading).

**A neural network does not clear the bar either.** `python -m src.Models.nn_rolling`
is the pre-registered **M01** arm: a feed-forward MLP (64, 32) on the **same 26
columns**, the same five walk-forward folds and the same train-only
`FeaturePipeline`, with the mean prediction of seeds 42–46 as the arm. The
canonical reference reproduces at **0.615663649** (recorded 0.615663649, drift
+3.4e-10), so the comparison is on its recorded footing. The MLP reaches
**0.6167** (Brier 0.2138, AUC 0.7122, accuracy 0.6617): accuracy is the one
metric that ticks up, and accuracy is not the judge. Δ = **−0.00099** nats,
CI [−0.00295, +0.00084], **|z| 1.03**, better in 2/5 folds. The interval includes
zero, so this is not evidence of a difference in either direction — and it is
certainly not an improvement.

The per-seed rows are the informative part: every individual seed is **worse**
than the canonical model (−0.00499 to −0.00903, |z| 3.02–5.05), and the five-seed
average is better than any of them (seed delta sd 0.00154). Seed averaging is
cancelling variance, not creating signal; the pre-registered arm is the average,
so the average is the decision. The descriptive sensitivity grid agrees: one
hidden layer of 32 (−0.00507), 128/64 (−0.00521) and stronger L2 (−0.00674) all
lose, and dropping early stopping diverges (log loss 1.18). Those configurations
cannot promote. This arm tests one MLP, not every neural architecture; the legacy
Keras trainers were never wired to this frame. Full output:
[`results/nn_rolling.md`](results/nn_rolling.md).

**Margin and total do not pay either.** Φ(m̂/σ) reaches 0.6204, indistinguishable
from fitting the label directly (Δ +0.0006, CI [−0.0026, +0.0038]); its value is the
richer output — margin RMSE 14.01, total RMSE 18.78, spread MAE 11.0.

**Bradley-Terry (0.6929) and SRS (0.6890) lose to always-predict-home (0.6883)**, for
a structural reason: a batch fit over five seasons reports where a team **was**.
Dropping Elo's movement term costs 0.0083 nats — more than the whole gap between the
best model and the base rate.

**Player ratings are a null.** The spec calls them "likely your biggest edge"; built to
be falsifiable, they were falsified. Δ = loss(team) − loss(team + player), so
**positive would mean the player features helped**:

| estimate | Δ log loss | 95% CI | verdict |
|---|---|---|---|
| RAPM-style, two-pool | +0.00151 | [−0.00059, +0.00358] | null |
| RAPM-style, shrunk | +0.00038 | [−0.00178, +0.00260] | null |
| logistic | +0.00060 | [−0.00154, +0.00282] | null |
| gradient boosting | −0.00243 | [−0.00815, +0.00295] | null |
| XGBoost | −0.00614 | [−0.01273, +0.00062] | null |

The one exception: substituting **leaked** actual minutes for expected ones is worth
−0.00975 nats, CI [−0.01349, −0.00590] — an order of magnitude larger than anything
above. See [the one thing that is not a null](#the-one-thing-that-is-not-a-null).

## The experiment log: 17 pre-registered runs

`scripts/run_experiments.py`, all on identical walk-forward folds (6,600 out-of-fold
rows). Log loss is the ranking metric; accuracy is reported and **not** ranked. Every
run is in `results/experiments.jsonl`; the script prints the selection cost of the
winner, because a best-of-N is the maximum of N draws.

Every figure in this table is an experiment-log-frame measurement — 59-, 53- and
43-column frames — made before issue #36, and kept as the measurements they were.
E14's **0.61831** is the 43-column `home_split_only` frame, not the canonical
model's number; the post-#36 canonical is **0.6157** in
[the Pipeline A comparison](#pipeline-a-model-comparison).

| experiment | feats | model | log loss | vs base | AUC | train−test gap |
|---|---|---|---|---|---|---|
| **E14-home-split-only** | **43** | logistic | **0.61831** | **−0.00084** | 0.7093 | +0.0017 |
| E31-drop-player-elo | 53 | logistic | 0.61835 | −0.00080 | 0.7095 | +0.0013 |
| E13-drop-opp-chars | 53 | logistic | 0.61892 | −0.00023 | 0.7080 | +0.0008 |
| E42-ridge | 59 | L2 logistic (ridge proxy) | 0.61894 | −0.00021 | 0.7078 | +0.0016 |
| E01-baseline-logistic-C0.1 | 59 | logistic C=0.1 | 0.61897 | −0.00018 | 0.7080 | +0.0008 |
| **E00-baseline-logistic** | 59 | logistic | **0.61915** | — | 0.7080 | +0.0003 |
| E12-drop-sos | 57 | logistic | 0.61919 | +0.00004 | 0.7080 | +0.0002 |
| E41-logistic-pca20 | 59 | logistic + PCA(20) | 0.61920 | +0.00005 | 0.7078 | +0.0031 |
| E11-drop-opp-adjusted | 57 | logistic | 0.61923 | +0.00008 | 0.7079 | +0.0002 |
| E10-drop-home-split | 53 | logistic | 0.62006 | **+0.00091** | 0.7068 | +0.0005 |
| E15-schedule-only | 37 | logistic | 0.62020 | +0.00105 | 0.7067 | +0.0016 |
| E20-rf-full | 59 | random forest | 0.62267 | +0.00352 | 0.7042 | **−0.1102** |
| E21-histgb-full | 59 | HistGradientBoosting | 0.62790 | +0.00875 | 0.6952 | **−0.0567** |
| E22-xgb-full | 59 | XGBoost | 0.63081 | +0.01166 | 0.6898 | **−0.0971** |
| E40-gaussian-nb-pca20 | 59 | Gaussian NB + PCA(20) | 0.64389 | +0.02474 | 0.6940 | +0.0044 |

**The best model is a 43-column logistic regression, and 16 of the 59 columns are
what is wrong with the 59-column version.** E14 uses *fewer* features and scores
*better*: the sixteen it drops — opponent-adjusted margin, strength of schedule, the
three never-read `OPP_*` pairs, and the six player-Elo aggregates — are worth
**negative 0.00084 nats**, which is to say they actively cost accuracy.

**E10 removes the per-team home/away split and costs +0.00091** — the largest penalty
of any ablation, and the only one whose sign is clearly the wrong way. That is the
difference between a feature that replicated at three seeds and two fold counts and
one that did not.

**Every tree model loses, and every tree model overfits.** Random forest +0.00352,
HistGB +0.00875, XGBoost +0.01166, with train−test gaps of −0.110, −0.057 and −0.097.
Linear models sit at +0.0003 to +0.0017, which is no overfitting at all.

**`logistic + PCA(20)`, once logged at 0.61605, loses.** That earlier run used a
59-column frame whose extra six columns were the **leaking** player-Elo block; on the
clean frame PCA(20) scores **0.61920**. The contaminated log is archived at
`results/archive/experiments-leaky-frame.jsonl` rather than deleted. `RidgeClassifier`
could not be scored at all — no `predict_proba` means no comparable probability — so
E42 is L2 logistic, the same estimator under a logit link.

**Do not read the winner's margin as an effect.** Best − median across the 17 runs is
−0.00026 nats and the top five are separated by less than 0.0007.

### The experiment ledger: everything else tried

Everything tried in this month's campaign beyond the 17 frame runs, one row per
experiment, each pre-registered in `results/experiments.jsonl` with its rationale
**before** it ran. The 17 frame runs stay in the table above, kept as
experiment-log-frame measurements; the full chronicle — declared rationale →
result → verdict, the errata, and why each null closed its branch — is
[docs/experiment_log_2026-10.md](experiment_log_2026-10.md).

| experiment | hypothesis | verdict | key number | detail |
|---|---|---|---|---|
| E00–E42 (17 frame runs) | which design earns its place on the 59/53/43-column frames? | none — the 43-col logistic wins; trees overfit | E14 0.61831 vs E00 0.61915 | [table above](#the-experiment-log-17-pre-registered-runs) |
| #45 schedule-block follow-up | the sub-bar lead buys the missing 0.41 of \|z\| at higher replication power | **stable sub-bar lead** — not a fold artifact, not a replication death; stays in the frame | \|z\| 2.55–2.76 over a 20-cell grid (Δ −0.00204 to −0.00212) | [log](experiment_log_2026-10.md#s45--issue-45-schedule-block-follow-up) · `results/issue45/` |
| H3 recalibration | season-blocked recalibration gains ≥ 0.003 nats at zero new information | **null — isotonic/Platt significantly harmful; branch closed** | isotonic −0.0129 nats, \|z\| 4.56 | [log](experiment_log_2026-10.md#h3--recalibration-null-and-two-variants-harmful) |
| H4 relative rest | rest enters as the rest-day differential (Esteves et al. 2021), ≥ 0.0024 nats | **null — B2B coefficients shrink 85% but no gain; branch closed** | primary Δ −0.00011, \|z\| 0.61 | [log](experiment_log_2026-10.md#h4--relative-rest-null) |
| H11 role substitution | absence-context (what kind of absence) adds ≥ 0.002 nats on top of availability | **null with a lean to cost; branch closed** | primary Δ −0.00069 [−0.00153, +0.00020], \|z\| 1.56 | [log](experiment_log_2026-10.md#h11--role-substitution-null-with-a-lean-to-cost) |
| H1a market-anchored blend | price + model beats price alone by ≥ 0.002 nats | **adverse — the blend is worse than price alone** | −0.00177 nats, \|z\| 3.15 (CI excludes zero *against*) | [market anchor](market_anchor.md) |
| H2 availability beyond price | availability adds ≥ 0.002 nats beyond price + model | **tight null — CI upper bound 10× below the minimum effect; branch closed** | Δ −0.00004 [−0.00027, +0.00019], \|z\| 0.31 | [market anchor](market_anchor.md) |
| C44 pace-first | pace-first decomposition beats the monolithic model on both RMSEs | **null at both endpoints; branch closed** | margin \|z\| 0.55, total \|z\| 1.51 | [log](experiment_log_2026-10.md#c44--spread-and-total) |
| C44 Skellam | a proper margin distribution beats Φ(m̂/σ) on margin log score **or** cover loss | **loses margin log score, wins cover — dispersion, not shape** | −0.00401 nats (\|z\| 2.49) vs +0.00415 (\|z\| 14.53) | [log](experiment_log_2026-10.md#c44--spread-and-total) |

Two errata ride with the ledger and are kept append-only: the S45 grid
artifacts carry a seed/fold metadata bug (`S45-erratum-artifact-metadata` maps
each artifact to its true run; every metric is per-run correct), and C44's
first execution had a secondary-endpoint bug whose corrected wide-bucket
comparison reads −0.00362 (\|z\| 2.27) — primary endpoints unaffected. Details
in [docs/experiment_log_2026-10.md](experiment_log_2026-10.md#errata-append-only-the-earlier-rows-stand-as-written).

## The one thing that is not a null

Almost every mechanism measured in this project came back flat. These did not.

| mechanism | delta | 95% CI | |
|---|---|---|---|
| **player-level expected minutes** | **+0.00382** | **[+0.00230, +0.00547]** | real, and small |
| *actual minutes (a leak)* | *+0.01089* | *[+0.00746, +0.01432]* | *unknown before tip-off* |
| *roster size alone (leak control)* | *+0.00109* | *[-0.00073, +0.00299]* | *straddles zero* |
| rebound block | +0.00040 | [−0.00026, +0.00104] | null |
| rank columns (52 cols) | +0.00065 | [−0.00039, +0.00166] | null |
| schedule block (7 cols) | −0.00204 | [−0.00358, −0.00051] | **still a lead, below the bar** |
| per-team home/away split (6 cols) | −0.00089 | [−0.00167, −0.00015] | **replicated, below the bar** |
| player Elo, pre-game roster (6 cols) | +0.00065 | [+0.00007, +0.00126] | **does not earn its place** |
| *player Elo, real lineup (a leak)* | *−0.00645* | *[−0.00890, −0.00402]* | *\|z\| 5.18 — the biggest fake win here* |
| recency weighting (halflife 5, 2) | worse | excludes zero | halflife 10 is already optimal |

**Availability is the first mechanism here whose interval excludes zero for the right
reason.** Summing each player's expected minutes over the squad that played the team's
*previous* game — known before tip-off whatever happens tonight — improves log loss by
0.00382 nats. With issue #36 landed, its measured contribution to the canonical model
is Δ **+0.00390** [0.00237, 0.00557] at **|z| = 4.78** (`python -m
src.Models.logreg_rolling`) — above this project's |z| > 3 bar, and the only effect
measured here that clears it honestly. Three instruments agree on one effect:
**+0.00382** published (`results/availability.json`, the table above), **+0.00399** on
the ablation frame, **+0.00390** in the canonical model. Team-level season averages
cannot express it, because a team is not one number. The block's full history — the
mechanism, the three instruments, the membership-leak contrast behind the
prior-game-squad convention, and issue #36's wiring (22 → 26 features, bit-identical
pre-#36 frame) — is [docs/availability_block.md](availability_block.md); its
boundary is now measured too: it improves forecasting and is fully priced at the
close ([docs/market_anchor.md](market_anchor.md)).

**Only about a third of it is honestly available.** Substituting *actual* minutes is
worth +0.01089, roughly three times more, and those are known only after the game. The
middle row exists to make that falsifiable rather than asserted: roster size alone,
carrying no minutes information whatsoever, returns +0.00109 with an interval
straddling zero — so the gain is not "how many people dressed". The remaining two thirds
is injury news, in-game injury and ejection, which does not exist before the game and
caps what any model here can achieve.

**The schedule block is a stable small effect that does not meet the bar:** Δ −0.00204
[−0.00358, −0.00051] at 5 folds, |z| 2.55–2.76 across 20 cells (4/5/6/8 folds × 5
seeds), never reaching |z| > 3. Any cell reproduces with
`python scripts/feature_block_ablation.py --blocks schedule --seed <S> --splits <F>
--out results/issue45/grid_s<S>_f<F>.json`.† Not a fold artifact, not a replication
death; it stays in the model and stays a lead. At |z| = 2.59 the one-sided p is 0.005,
which sits exactly on a Bonferroni threshold of 0.05/10 for ten looks rather than
comfortably inside it, and it still does not clear the |z| > 3 bar (Vandenbruaene et
al. 2022, *J Sports Econ* 23(7) — the null of unprofitability survived across 628
implementations). It is not a discovery, and presenting it as one would be exactly the
error the rest of this page exists to prevent.

† The grid artifacts' `seed`/`n_splits` fields carried a metadata bug (fixed); `S45-erratum-artifact-metadata` in `results/experiments.jsonl` maps each artifact to its true run.

## The feature search: four candidates, pre-registered

Every decay experiment in this project failed, because all of them reweight the *same*
numbers the model already has. So the next round had to bring information the frame did
not already contain, and the four candidates were **written into the block registry
before any was measured**, so the results below cannot be cherry-picked out of a longer
search.

Δ = loss(full) − loss(minus block), so **negative means the block earns its place**.
Reproduce with `python scripts/feature_block_ablation.py --seed 42 --splits 5` (and
`--seed 137`, `--seed 2718`, `--splits 8` for the replication).

| candidate | the new information | Δ | 95% CI | \|z\| | verdict |
|---|---|---|---|---|---|
| opponent-adjusted form | beating a good club counts for less | −0.00004 | [−0.00014, +0.00005] | 0.92 | null |
| **home/away split** | **each team's *own* home advantage** | **−0.00089** | **[−0.00167, −0.00015]** | **2.29** | **replicated** |
| strength of schedule | quality of opposition faced | −0.00008 | [−0.00015, −0.00000] | 2.01 | **died in replication** |
| opponent characteristics | three `OPP_*` columns never read | +0.00000 | [−0.00061, +0.00060] | 0.01 | null |

**One of four is real, and it is small.** The per-team home/away split is the only
candidate that replicated — the point estimate does not move at three bootstrap seeds or
at eight folds instead of five. The block sits in the canonical frame, and adding it to
the model moves log loss from 0.61956 to **0.61897** (both pre-#36 measurements; the
canonical model's post-#36 number is 0.6157, availability block included). It is also
plainly below the bar: |z| ≈ 2.3 does not clear
|z| > 3, one-sided p ≈ 0.011 does not clear a Bonferroni threshold of 0.005 for ten
looks, and 0.00089 nats is under a quarter of the availability effect and about a
thirtieth of what the closing market beats this model by. A real feature, reported at
its real size.

**The instructive failure is strength of schedule.** At five folds it excluded zero and
the script called it a keeper; at eight folds it became a clean null with the sign
flipped. A result that exists only at one fold count is a property of the fold count.
Why each candidate went the way it did, and the two ways a candidate can fool you —
the fold artifact and the unwritten duplicate — is in
[docs/feature_search.md](feature_search.md).

## The player-Elo block: a strong result that was a leak

Per-player Elo ratings, totalled over each team's expected-available roster, were built
as a feature block and scored. The first measurement was:

| roster construction | Δ log loss | 95% CI | \|z\| | verdict |
|---|---|---|---|---|
| **over the actual participants** | **−0.00645** | [−0.00890, −0.00402] | **5.18** | the largest effect ever measured here |
| over the pre-game roster | +0.00065 | [+0.00007, +0.00126] | 2.14 | **does not earn its place** |

The first row is target leakage, and it was nearly published as a result. |z| = 5.18
clears this project's bar of 3, clears Bonferroni for eleven looks, and is larger than
every real effect in this repository combined. The *weights* were honest — trailing
EWMAs, strictly causal, verified by truncating the future. The leak was in the
**membership**: it summed over the players who actually played, and nobody knows before
tip-off which ten men will be on the floor. On a pre-game roster the same block is
+0.00065 and does not earn its place. Full post-mortem, including the bucket analysis
that shows the raw total runs **backwards** — the stronger a team's accumulated
rating-minutes looks, the *less* it wins:
[docs/player_elo_leak.md](player_elo_leak.md).

## Nine more candidate blocks, all measured, all null

`docs/modeling_exploration.md` builds and **scores** nine feature blocks out of the
tables nothing had read — actual travel distance, margin volatility, matchup-specific
deviation, momentum, shot-chart mix, league-era effects, never-read team columns and
two play-type PPP variants — all on the same five walk-forward splits. Five of the ten,
and the shape of the result:

| block | cols | OOF log loss | vs base | \|z\| |
|---|---|---|---|---|
| actual travel (origin → destination) | 3 | 0.61890 | **+0.00026** | 1.02 |
| margin volatility (trailing sd) | 2 | 0.61889 | **+0.00026** | 1.19 |
| never-read team columns (28) | 28 | 0.61932 | −0.00017 | 0.22 |
| league era / season index | 4 | 0.62043 | −0.00128 | 2.25 |
| prior-season play-type PPP (full) | 66 | 0.62763 | −0.00847 | **4.96** |

**The best of ten is +0.00026 at |z| = 1.19**, and the correlation screen is what makes
that a finding rather than an opinion: several of these are about as orthogonal to the
existing features as a team statistic can be — trailing margin volatility at r = 0.04,
the last-2-vs-last-10 margin contrast at r = 0.06 — and they still predict nothing.
One block is worse in an informative way: prior-season play-type PPP costs **−0.0085
nats at |z| = 4.96**, monotonically worse as columns are added, while correlating only
r ≈ 0.08 with the features that already carry the signal. The information arrived and a
59-column unregularised frame could not carry it. The full table, the correlation
screens, the rejected candidates and the reproduce instructions are in
[docs/modeling_exploration.md](modeling_exploration.md).

## Compared with the original model

| | data | rows | validation | log loss | AUC |
|---|---|---|---|---|---|
| original logistic regression | scraped odds | 17,356 | chronological 90/10 | 0.6189 | 0.7084 |
| logistic regression (this fork, post-#36) | NBA game logs | 13,199 | 5× walk-forward | 0.6157 | 0.7141 |

**These two numbers are not a like-for-like comparison.** Different games, different
coverage, different seasons, different split geometry. Pre-#36 this fork's row read
0.6190 / 0.7093 — near-identical to the original's 0.6189 by coincidence rather than
as a result, and kept in [the Pipeline A comparison](#pipeline-a-model-comparison) as
the measurement it was. What can be said is narrower and still worth saying:
a linear model on team strength is hard to beat on this task, whether the features come
from a scrape or from NBA directly, and the models built on top of it did not beat it
here.
