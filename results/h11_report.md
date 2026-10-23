# H11 — role-substitution / usage redistribution: a null

Pre-registration: `results/experiments.jsonl` :: `H11-prereg-absence-context`
(declared 2026-10-23T10:18:18Z, before any run). Artifact:
`results/h11_absence_context.json`. Script: `scripts/h11_absence_context.py`.

## Hypothesis

An `absence_context` block improves held-out log loss by **>= 0.002 nats**
incrementally on top of the canonical frame **+ the existing availability
block** (expected minutes of the prior-game squad, +0.0038 nats [0.0023,
0.0055]), with the 95% CI excluding zero and |z| > 3. The availability block
says *how much* is missing; H11 asks whether *what kind* of missing value is
being absorbed adds anything on top.

Candidate block (6 cols, fixed before measurement):
`{side}_missing_usage_share` (trailing usage-minutes share of the
highest-usage presumed-missing player), `{side}_margin_without_missing_top`
(trailing EWMA of team margin in past games where that player was absent,
shrunk to overall form), `{side}_margin_without_top2_usage` (same over past
games missing >= 1 of the team's current top-2 usage players — "next-man-up"
capacity).

## Config

- Instrument: `feature_block_ablation.build_feature_frame` machinery +
  `recency_ablation.make_pipeline` (impute/scale/logreg C=1.0), walk-forward
  `n_splits x (initial_frac=0.5, gap=0)`, `paired_bootstrap_delta` n_boot 2000.
- B1 = pinned canonical 67-col base + the availability block's 3 measured
  columns (70 cols, `strict_roster_expected` arm of `results/availability.json`).
  B2 = B1 + 6 candidate cols (76).
- Settings: primary s42/f5; replications s137/f5, s2718/f5, s42/f8.
- 7 looks (C1 x 4 + 3 group diagnostics) -> Bonferroni 0.05/7 = 0.0071
  one-sided; project bar |z| > 3 governs.

## Results — C1 (Δ = loss(B1) − loss(B2); POSITIVE = candidate better)

| setting | Δ | 95% CI | \|z\| | log loss B1 | log loss B2 |
|---|---|---|---|---|---|
| s42/f5 (primary) | **−0.000686** | [−0.001532, +0.000196] | 1.56 | 0.615460 | 0.616146 |
| s137/f5 | −0.000686 | [−0.001540, +0.000182] | 1.56 | 0.615460 | 0.616146 |
| s2718/f5 | −0.000686 | [−0.001545, +0.000194] | 1.55 | 0.615460 | 0.616146 |
| s42/f8 | −0.000602 | [−0.001429, +0.000294] | 1.37 | 0.615728 | 0.616330 |

Per-fold Δ (f5): −0.00083, −0.00020, −0.00002, +0.00034, −0.00272.
Per-season Δ (primary): 2020-21 −0.00045; 2021-22 −0.00048; 2022-23 +0.00039;
2023-24 −0.00108; 2024-25 +0.00049; 2025-26 **−0.00284** (the drift tail).

Primary metrics — B1: log loss 0.61546, Brier 0.21332, AUC 0.7135, acc 0.6594.
B2: log loss 0.61615, Brier 0.21353, AUC 0.7132, acc 0.6597.

## Per-feature-group diagnostics (primary setting)

Leave-one-group-out, Δ = loss(B2−g) − loss(B2); POSITIVE = the group helps:

| group | Δ | 95% CI | \|z\| | mean \|coef\| (std units) |
|---|---|---|---|---|
| missing_usage_concentration | −0.00006 | [−0.00054, +0.00048] | 0.22 | 0.034 |
| performance_without_missing_top | −0.00037 | [−0.00102, +0.00032] | 1.09 | 0.088 |
| next_man_up_absorbency | −0.00031 | [−0.00062, +0.00001] | 1.94 | 0.030 |

Nothing moves in the helpful direction. The "without that player" history
carries the largest coefficients but adds no out-of-sample value; the
next-man-up group is the closest to a real (negative) signal — removing it
marginally helps (|z| 1.94, still not a finding).

## Leakage checks — passed

Truncation invariance: rebuilding the block on games cut at 2021-10-20 is
**bit-identical** on 7,061 rows (max |diff| 0.0, NaN pattern equal).
Membership: the presumed-missing set, top-2 usage players and every
carried-forward player value come from team-game sequence numbers strictly
below the row's own; game t's participants are never read (the
`docs/player_elo_leak.md` trap). Imputation/scaling fit on train folds only
(`ra.fit_oof`); identical OOF rows and labels across arms (asserted) plus
parity with `ra.fit_oof` (asserted).

## Verdict against the pre-registered rule

**H11 FAILS its pre-registered rule — the absence-context block is a NULL (with
a lean toward a small cost): the primary Δ is −0.00069, its CI straddles zero,
|z| = 1.56 < 3, the point estimate is negative at all four designated settings,
and it is nowhere near the 0.002-nat target. The block does not earn its place.**

Leaderboard (pooled OOF log loss, this instrument): ablation pinned A1 base
0.61945 → canonical `load_design` 0.61957 (other instrument) → **B1
(canonical + availability) 0.61546** → B2 (B1 + absence_context) 0.61615.
Sanity: B1's margin over the pinned base is 0.00399 nats, reproducing the
availability block's published +0.0038 on this frame.

Caveats: lbfgs hit `max_iter=300` (the frozen instrument config, identical in
both arms and in every prior run — affects the paired comparison not at all);
the 2025-26 season contributes most of the negative lean; the group-3 "at least
one of the top-2 out" reading was pre-registered and is not the dossier's
literal "both out" reading.

## Reproduce

```
.venv/bin/python scripts/h11_absence_context.py --prereg     # declaration only
.venv/bin/python scripts/h11_absence_context.py              # the measurement
```

Frames are cached at `/tmp/opencode/h11/arms.pkl` (keyed on the game set); a
cold build takes ~1 min, the full run ~8 min.
