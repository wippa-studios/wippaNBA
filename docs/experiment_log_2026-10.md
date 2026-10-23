# The October 2026 experiment campaign

Every experiment run this month, in order, with its pre-registration, its
result and its verdict — plus the two errata, why each null closed its branch,
and the tree-audit finding that put part of round 3 beyond reproduction. The
append-only ledger is `results/experiments.jsonl`; the readable mirrors are
`results/h3h4_report.md`, `results/h11_report.md` and `results/c44_report.md`.
The compact version of this page is
[the experiment ledger](../README.md#the-experiment-log-17-pre-registered-runs).

## The discipline

Every experiment is **declared before it runs**: hypothesis, arms,
sign convention, decision rule, minimum effect of interest, and the
multiplicity budget are appended to `results/experiments.jsonl` with status
`declared_before_execution`, and the verdict is appended as a
`verdict_record` after. Three rules hold across the campaign:

1. **Seeds resample, they do not replicate.** Seeds 42 / 137 / 2718 bootstrap
   the same rows; identical point estimates across seeds are expected, the CIs
   probe stability, and the seeds add **no** looks.
2. **Multiplicity is counted up front** and a Bonferroni threshold is written
   into each pre-registration; the project's **|z| > 3** bar
   (Vandenbruaene et al. 2022, *J Sports Econ* 23(7)) governs on top of it.
3. **Secondary metrics cannot promote a result.** A primary endpoint that
   fails cannot be rescued by a favourable secondary; a favourable secondary
   is reported as a finding about a different quantity.

Each experiment states its rationale before its result. The rationale is kept
verbatim below precisely where it turned out wrong.

## The day at a glance

| # | experiment | declared | hypothesis in one line | verdict | key number | detail |
|---|---|---|---|---|---|---|
| 1 | S45 schedule-block follow-up (#45) | 08:15 | the sub-bar lead buys 0.41 of \|z\| at higher replication power | **still a lead, below the bar** | \|z\| 2.55–2.76 over 20 cells | [below](#s45--issue-45-schedule-block-follow-up) · `results/issue45/` |
| 2 | H3 recalibration | 09:14 | recalibration at zero new information ≥ +0.003 nats | **null — isotonic/Platt significantly harmful; branch closed** | isotonic −0.0129, \|z\| 4.56 | [below](#h3--recalibration-null-and-two-variants-harmful) · `results/h3_recalibration.json` |
| 3 | H4 relative rest | 09:14 | rest-day differential ≥ +0.0024 nats (Esteves et al. 2021) | **null — B2B coefficients shrink 85%, no gain; branch closed** | primary Δ −0.00011, \|z\| 0.61 | [below](#h4--relative-rest-null) · `results/h4_rest_differential.json` |
| 4 | H11 role substitution | 10:18 | absence-context block ≥ +0.002 nats incremental | **null with a lean to cost; branch closed** | primary Δ −0.00069, \|z\| 1.56 | [below](#h11--role-substitution-null-with-a-lean-to-cost) · `results/h11_absence_context.json` |
| 5 | H1a market-anchored blend | 11:36 | price+model beats price alone ≥ +0.002 nats | **adverse — the blend is worse than price alone** | Δ −0.00177, \|z\| 3.15 | [market anchor](market_anchor.md) |
| 6 | H2 availability beyond price | 11:36 | price+model+availability beats price+model ≥ +0.002 nats | **tight null — CI upper bound 10× below the minimum effect** | Δ −0.00004, \|z\| 0.31 | [market anchor](market_anchor.md) |
| 7 | C44 pace-first (#44) | 12:10 | pace-first decomposition beats the monolithic model on both RMSEs | **null at both endpoints** | margin \|z\| 0.55, total \|z\| 1.51 | [below](#c44--spread-and-total) · `results/c44_report.md` |
| 8 | C44 Skellam (#44) | 12:10 | a proper margin distribution beats Φ(m̂/σ) | **loses margin log score, wins cover — dispersion, not shape** | −0.00401 (\|z\| 2.49) vs +0.00415 (\|z\| 14.53) | [below](#c44--spread-and-total) · `results/c44_report.md` |

The 17 pre-registered frame runs (E00–E42) predate this campaign and are kept
as experiment-log-frame measurements in
[the experiment log](../README.md#the-experiment-log-17-pre-registered-runs).

## S45 — issue #45, schedule-block follow-up

**Declared rationale:** *"The schedule block (7 cols: rest days, back-to-backs,
3-in-4, travel_miles) carries a real, leak-free improvement of about −0.002
nats of log loss; the original measurement (delta −0.00203 [−0.00358,
−0.00051], |z| = 2.59) was labelled 'lead, not a finding' because it misses
the |z| > 3 bar. This follow-up asks whether 0.41 of |z| is obtainable at
higher replication power."*

**Result.** A pre-registered 20-cell grid (folds 4/5/6/8 × seeds
42/137/2718/7/19, n_boot 2000) plus seven leave-one-column-out
decompositions. The point estimate is −0.00204 to −0.00212 in **every** cell,
every CI excludes zero, and |z| runs **2.55–2.76** — zero cells clear |z| > 3.
The decomposition localises the effect to the back-to-back columns:
`home_back_to_back` −0.00079 [−0.00137, −0.00023], |z| 2.71, is the only
column whose interval excludes zero (`away_back_to_back` −0.00092 leans the
same way, |z| 1.61); both `rest_days` and both `in_3_of_4` are nulls and
`travel_miles` is a null at |z| 0.46, as pre-registered.

**Verdict: still a lead — a stable small effect below the bar.** Not a fold
artifact (it holds at 4/5/6/8 folds) and not a replication death (sign stable
in all 20 cells). The block stays in the canonical frame, and the README line
reads exactly this. Even the cherry-picked best-of-20 cell (|z| 2.76) fails
|z| > 3 and fails the follow-up's own Bonferroni (27 looks → 0.00185;
cumulative 37 looks → 0.00135).

## H3 — recalibration: null, and two variants harmful

**Declared rationale:** *"Season-blocked, strictly causal recalibration of the
canonical model's out-of-fold probabilities — isotonic, Platt, and a
regime-aware home-term shrink (home-court advantage measurably shrank from
~2.74 to ~1.75 pts after 2020, Pelechrinis 2023) — improves held-out log loss
by ≥ 0.003 nats at zero new information."* Each fitted calibrator sees only
the previous season block's OOF rows (fit and score rows disjoint, asserted).

| variant | Δ (improvement) | 95% CI | \|z\| vs 0 | verdict |
|---|---|---|---|---|
| isotonic_prevseason | **−0.01292** | [−0.01877, −0.00765] | 4.56 | significantly harmful |
| platt_prevseason | **−0.00188** | [−0.00335, −0.00036] | 2.46 | significantly harmful |
| home_regime_shrink | −0.00001 | [−0.00136, +0.00131] | 0.01 | null |
| home_intercept_adj | −0.00069 | [−0.00181, +0.00044] | 1.19 | null (leans worse) |

**Verdict: FAIL, branch closed.** No variant reaches +0.003 nats; two of four
are significantly negative. Two findings worth keeping from the wreckage: the
home-term regime drift **is** visible in the fitted model (`home_adv` falls
0.346 → 0.300 log-odds across folds) but acting on it post-hoc gains nothing —
the fit window already adapts; and ECE can be bought at log-loss cost
(0.0134 → 0.0065 for −0.0007 nats), which is a loss under any proper scoring
rule and was the pre-registered primary. Recalibration of the canonical model
is closed as a null at zero new information; nothing is kept.

## H4 — relative rest: null

**Declared rationale:** *"The schedule block's sub-bar effect ... is
mis-parameterised: rest should enter as the REST-DAY DIFFERENTIAL between the
two teams (relative rest; Esteves et al. 2021 found relative rest matters more
than absolute), not as team-wise absolute day counts. A relative-rest block
should show ≥ 0.0024 nats and the home/away B2B columns should shrink when it
is present."* Arms: A1 = pinned canonical 67-col frame, A2 = A1 + six
candidate columns (73), A3 = A2 − schedule block (66).

| comparison | Δ (primary, s42/f5) | 95% CI | \|z\| | reading |
|---|---|---|---|---|
| C1: A2 vs A1 (earns its place?) | −0.00011 | [−0.00046, +0.00025] | 0.61 | negative at every setting — leans worse |
| C2: A3 vs A1 (replaces absolute?) | +0.00034 | [−0.00041, +0.00113] | 0.85 | null |
| C3: drop absolute given candidate | +0.00045 | [−0.00028, +0.00114] | 1.24 | null |

**Verdict: FAIL, branch closed.** The pre-registered secondary expectation
**did** occur — standardised |coef| for `home_back_to_back` collapses 0.0509 →
0.0077 (−85%) and `away_back_to_back` 0.1204 → 0.0649 (−46%) when the
differential is present — but that representational fact buys no log loss. The
block does not earn its place and stays out of the canonical frame.

## H11 — role substitution: null, with a lean to cost

**Declared rationale:** *"An absence-context block improves held-out log loss
by ≥ 0.002 nats INCREMENTALLY on top of the canonical frame + the existing
availability block ... The availability block says HOW MUCH is missing; H11
asks whether WHAT KIND of missing value is being absorbed adds anything on
top."* Six columns in three groups (missing-usage concentration, performance
without the missing top, next-man-up absorbency), B1 = pinned canonical +
availability (70 cols) vs B2 = B1 + block (76).

**Result.** Primary Δ = loss(B1) − loss(B2) = **−0.000686**
[−0.001532, +0.000196], |z| 1.56 (B1 0.61546 vs B2 0.61615) — negative at all
four designated settings, with 2025-26 (−0.00284) contributing most of the
lean. No group diagnostic moves in the helpful direction; the "next-man-up"
group is closest to a real signal and it points the wrong way (|z| 1.94).
Truncation invariance is bit-identical on 7,061 rows; membership derives from
prior-game sequence numbers only (the `docs/player_elo_leak.md` trap).

**Verdict: NULL with a lean to a small cost — branch closed.** Nowhere near
the 0.002-nat target; the block does not earn its place. Sanity check that
survives the failure: B1's margin over the pinned base is 0.00399 nats,
reproducing the availability block's published +0.0038 on this frame.

## H1a + H2 — the market-anchored blend

**Declared rationale:** H1a — *"the walk-forward logistic stack of [price
logit, canonical-model logit] beats the de-vigged closing price alone by ≥
0.002 nats ... This is THE alpha question: is any of our information not
already in the close?"*; H2 — *"replacing the plain model logit with the
model+availability logit beats P+M by ≥ 0.002 nats ... Is the availability
block's signal already priced by the close?"* Full protocol, arm table,
cold-start caveat and diagnostics: [the market anchor](market_anchor.md).

**Result.** H1a: **adverse** — P+M is worse than P by 0.00177 nats,
CI [−0.00292, −0.00072], |z| 3.15 (seeds 137/2718 agree). H2: **tight
null** — Δ −0.00004 [−0.00027, +0.00019], |z| 0.31, CI upper bound 10× below
the 0.002-nat minimum effect. H1b (descriptive) confirms the machinery: P+M
beats M alone by +0.02392 nats, |z| 7.45.

**Verdict: both branches closed.** Blending our model into the price is
adverse (with the cold-start caveat on record and the post-hoc variant
deliberately not run); availability's increment beyond price + model is not
established at the minimum effect size. The close absorbs everything our model
knows, including availability.

## C44 — spread and total

**Declared rationale:** C1 — *"a pace-first decomposition (possessions first,
then points-per-possession per side; margin and total derived from the same two
team-point predictions) beats the monolithic margin/total ridge on BOTH margin
RMSE and total RMSE"*; C2 — *"a proper margin distribution beats Φ(m̂/σ): the
SKELLAM on (μ_home, μ_away) ... integer-valued like a margin, game-specific
variance μ_h+μ_a = predicted total, zero additional fitted parameters — scores
better than the single-scale Gaussian on (a) the margin-bucket log score (≥
0.002 nats) OR (b) cover-probability calibration against the closing spread."*

| endpoint | Δ | 95% CI | \|z\| | verdict |
|---|---|---|---|---|
| C1 margin RMSE | +0.0058 | [−0.0153, +0.0260] | 0.55 | null |
| C1 total RMSE | +0.0209 | [−0.0063, +0.0480] | 1.51 | null |
| C2 margin log score (1-pt buckets) | **−0.00401** | [−0.00713, −0.00081] | 2.49 | **decisive loss** |
| C2 cover log loss vs closing spread | **+0.00415** | [+0.00358, +0.00470] | **14.53** | **clears the bar** |

**Verdicts.** C1: pace-first does not beat the monolithic model — **null at
both endpoints**, branch closed. C2: the pre-registered OR rule is met at the
cover endpoint — a proper margin distribution **earns its place** there — but
it **loses** the margin log score outright. The reading is **dispersion, not
shape**: the Skellam's game-specific variance spreads the tails better (cover
probabilities against the closing spread), while its integer shape scores
worse on 1-point margin buckets than the Gaussian Φ(m̂/σ). Issue #44's own
market tests agree with everything else here: the closing line beats the model
on margin MAE (|z| 9.07) and total MAE (|z| 9.02), and the parameter-free ROI
arms at flat −110 are all negative (strongest |z| 2.80 — below the bar).

## Errata (append-only; the earlier rows stand as written)

**S45-erratum-artifact-metadata.** `scripts/feature_block_ablation.py` writes
the module constants `SEED=42` / `N_SPLITS=5` into its JSON artifact instead
of the run's `--seed`/`--splits` arguments, and those fields were read into
the 20 grid records — all of which are named `S45-grid-s42-f5`. Every metric
in those records is per-run correct and matches the run logs in
`results/issue45/logs/`; read each record's true identity from its artifact
path (`grid_s<seed>_f<folds>.json`), mapped in the erratum record. Not fixed in
code at the time — it was outside the change surface — and the mislabelled
records are left in place rather than rewritten; `S45-erratum-artifact-metadata`
supersedes their label fields, and `S45-verdict-corrected` supersedes
`S45-verdict`'s collapsed per-fold fields. The same metadata bug is in
`results/feature_block_ablation.json` (harmless there: that run really was
s42/f5).

**C44-correction-note.** A secondary-metric bug in the first execution:
`skellam_bucket_prob_wide` truncated half-integer bounds with `astype(int)`
and was off by one on the lower bound, so 5-point bucket masses summed to
~1.20 instead of 1. The wide-bucket log score and the bucket-calibration
Skellam column of that run are **wrong**; the corrected wide-5 comparison is
−0.00362 [−0.00667, −0.00043], |z| 2.27. The primary endpoints (1-point margin
log score via `skellam.logpmf`, cover/over probabilities via `sf`, the C1
RMSEs) were computed from exact pmf/cdf values and are **unaffected** — the
primary rows stand as logged. The helper now uses `cdf(5j+2)−cdf(5j−3)` and
`_assert_bucket_partition` enforces the partition.

**H3 re-appended rows.** The H3 reproduction rerun (13:08, after the
`load_design(with_availability=False)` frame pin) re-appended H3 rows to
`results/experiments.jsonl` whose values duplicate rows 50–54's exactly —
numerically identical by design, since that bit-identical match is the
reproduction gate. Append-only is honored: the duplicate rows are kept as
written; they are the reproduction evidence, not new measurements.

## Why each null closed its branch

The campaign tested the four remaining shapes of improvement. Each failed in
the shape's own way:

- **Re-parameterising known information is flat.** H4's relative rest is a
  re-parameterisation of the schedule block — the coefficients move (B2B −85%)
  and the loss does not. Consistent with the swarm's unifying conclusion: the
  model's error is bias, not variance.
- **Post-hoc correction of the output is harmful at zero new information.**
  H3's calibrators fit on the season before and lose; isotonic on one season
  (~1,230 rows) is destructive.
- **Context about *what* is missing adds nothing beyond *how much*.** H11, on
  top of the one block that works.
- **The close already has it.** H1a and H2: the price alone beats every blend,
  and availability's increment beyond price + model is ruled out at the
  0.002-nat minimum effect.

A null is a complete result: each branch above is closed, with its decision
rule, its multiplicity budget and its leakage check on record. The branches
are not pending; they are done.

## Tree audit and the round-3 reproducibility finding

The round-3 tree audit found process debt from prior round-3 work:
`scripts/round2_common.py`, the shared round-2 harness named throughout
`results/round3_summary.md` ("re-exports the round-2 harness unchanged"), was
**never committed** — `git log --all -- scripts/round2_common.py` is empty on
every branch. The round-3 promoted-frame numbers in `results/round3_summary.md`
are therefore **not reproducible in this checkout**.

It is deliberately **not** reimplemented. A substitute would be a different
harness than the one that measured those numbers (the bar was reproduced to
1e-9 against `results/round2_experiments.jsonl` through that exact module), and
silently swapping in a lookalike would turn a documented gap into an
undocumented one. `scripts/round3_common.py` fails loudly with that message
instead — the gap stays loud. Recover the original module (or re-derive the
numbers) before trusting anything downstream of it.

What the audit did produce: **`Tests/test_leakage_gate.py` now exists — 16
pins.** `feat_squad_quality`'s docstring had promised that file plus
`scripts/round3_verify.py` check membership and truncation mechanically; the
file did not exist in repo history. It now pins, fixture-level and with no
network, the leakage properties of the three prior-game-squad builders
(`feat_squad_quality`, `feat_squad_continuity`, `feat_team_elo`): membership
causality (a player debuting tonight contributes nothing to tonight's
features; a player absent tonight still contributes from the prior game),
truncation invariance (bit-identical retained rows, NaN pattern included),
strictly-causal attach (`allow_exact_matches=False`, so game 2 of a
doubleheader cannot reach game 1), and that a game's own result cannot reach
its own ratings. The sin it guards against is
[the player-Elo leak](player_elo_leak.md): cutting the future does not change
who was in the room tonight.

## Reproducing this page

```bash
.venv/bin/python scripts/feature_block_ablation.py --blocks schedule --seed <S> --splits <F> \
    --out results/issue45/grid_s<S>_f<F>.json     # S45 cells
.venv/bin/python scripts/h3_recalibration.py      # H3, ~20 s
.venv/bin/python scripts/h4_rest_differential.py  # H4, ~4 min
.venv/bin/python scripts/h11_absence_context.py   # H11, ~8 min cold
.venv/bin/python scripts/h1h2_market_anchor.py    # H1a/H2, ~10 s
.venv/bin/python scripts/c44_spread_total.py      # C44, from cached predictions
python -m pytest Tests/ -q                        # 1,287 passed, 104 subtests (2026-10-23)
```

Every declaration, run, erratum and verdict is in `results/experiments.jsonl`
in that order, and nothing in that file is rewritten.
