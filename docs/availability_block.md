# The availability block: the one mechanism that survived

The pre-game availability feature block, its full history, and the boundary
the market-anchored measurement put on it. The README carries the headline
([the one thing that is not a null](../README.md#the-one-thing-that-is-not-a-null));
this page carries the mechanics, the falsification attempts, and the wiring.

**Reproduce:**

```bash
python scripts/availability.py                       # the published instrument
.venv/bin/python scripts/h11_absence_context.py      # the +0.00399 line (row 2): B1 vs pinned base sanity print, recorded in results/experiments.jsonl / results/h11_report.md
python scripts/feature_block_ablation.py --seed 42 --splits 5   # the ablation-frame instrument; its own artifact does not carry the +0.00399
python -m src.Models.logreg_rolling                  # the canonical-model instrument
.venv/bin/python scripts/h1h2_market_anchor.py       # the boundary measurement (H2)
```

## The mechanism

Sum each player's **expected minutes** over the squad that played the team's
*previous* game. Everything in that sum is knowable before tip-off whatever
happens tonight: the previous game's participants are a fact by game time, and
expected minutes are trailing quantities. Team-level season averages cannot
express it — a team is not one number.

The membership convention — the *previous* game's squad, not tonight's — is the
whole lesson of this repository in one line. See
[the player-Elo block](../README.md#the-player-elo-block-a-strong-result-that-was-a-leak):
its weights were honest (trailing EWMAs, strictly causal, verified by
truncating the future) and it still measured |z| = 5.18 target leakage,
because it summed over **the players who actually played**. Nobody knows
before tip-off which ten men will be on the floor, so a feature that reads
tonight's participants is a function of the answer. The prior-game-squad
convention makes membership a pre-game quantity, and that is the difference
between the two results.

## Three instruments, one effect

The same mechanism measured three ways, all agreeing:

| instrument | frame | Δ log loss | notes |
|---|---|---|---|
| `scripts/availability.py` (published) | `results/availability.json` | **+0.00382** [+0.00230, +0.00547] | the table in the README |
| feature-block ablation | pinned A1 base → + availability | **+0.00399** | reproduces the published +0.0038 on this frame (the H11 sanity line) |
| canonical model (issue #36) | `load_design(with_availability=True)` | **+0.00390** [+0.00237, +0.00557] | **\|z\| = 4.78** (Δ reads off the two `logreg_rolling` outputs above; its CI is from the #36 verification record, prints-only) |

Δ = loss(without) − loss(with), so positive means the block helped. |z| = 4.78
was the only effect in this repository that cleared the project's |z| > 3 bar
honestly until issue #68, when the pre-tip injury multiplier built on this same
block cleared a forward two-season holdout at +0.003579 (|z| 3.85) — see
[the injury pipeline](injury_pipeline.md#what-it-measured-68-promoted-on-a-powered-forward-holdout).

## Falsification attempts that failed to falsify it

From `results/availability.json` (same instrument, same rows):

| arm | Δ | 95% CI | reading |
|---|---|---|---|
| expected minutes, prior-game squad | +0.00382 | [+0.00230, +0.00547] | the mechanism |
| *actual minutes (a leak)* | *+0.01089* | *[+0.00746, +0.01432]* | *known only after the game* |
| *roster size alone (leak control)* | *+0.00109* | *[−0.00073, +0.00299]* | *straddles zero* |

The roster-size control exists to make the claim falsifiable: a column carrying
no minutes information whatsoever does nothing, so the gain is not "how many
people dressed". The actual-minutes arm sets the ceiling: the honest mechanism
recovers roughly a third of what the leaked one measures, and the remaining
two thirds is injury news, in-game injury and ejection — information that does
not exist before the game and caps what any pre-game model here can achieve.
(Substituting actual minutes is worth −0.00975 nats in the player-ratings
module's convention, CI [−0.01349, −0.00590] — the largest magnitude measured
anywhere in this project, and a leak.)

## Issue #36: wiring it into the canonical model

Issue #36 landed the block in the shared design matrix, and the wiring is
designed so nothing measured before it moves:

- `load_design(..., with_availability=True)` (the default since #36) appends
  the block last, in the block's own order: **22 → 26 features**. The
  canonical model is the row with it in
  [the Pipeline A comparison](../README.md#pipeline-a-model-comparison) —
  log loss **0.6157** (0.61566), against 0.6196 (0.61957) for the pre-#36
  frame.
- `with_availability=False` returns the pre-#36 frame **bit-for-bit**, which
  is how every pre-#36 measurement stays reproducible — the E-runs, the
  feature-search ablations, the boosting deltas of their time.
- The market arms are bit-identical: the stacking measurement
  ([the market anchor](market_anchor.md)) first reproduces
  `results/closing_line.json` on the 5,788-row overlap (market 0.59766 /
  model 0.62340 / model + availability 0.62034) before it computes anything.
- The mechanism itself was simplified on the way in: `availability_features`
  emits four columns and three of them are redundant (two are exactly `d` and
  `−d`); the single contrast column does the whole job, measured at 44 columns
  against the 43-column bar in
  [the swarm section](../README.md#the-agent-swarm).

## Issue #37: the measured injury extension

The block predicts availability from trailing minutes, and knows nothing about
tonight's designations. Issue #37 wired the official pre-tip injury report into
it as a **same-game multiplier** — `availability_features(..., weights=w)` with
one `(GAME_ID, PLAYER_ID, weight)` row per designated player, `Out` = 0,
`Doubtful` = 0.25, `Questionable` = 0.5, `Probable` = 0.75, `Available` = 1.
The squad, the 15-game player halflife and the causality are the block's own;
only the multiplier is new, and `weights=None` is bit-identical to the pre-#37
block (pinned by a test). The measurement protocol and the coverage numbers are
in [the injury pipeline](injury_pipeline.md#what-it-measured-37-below-the-bar).

The result is a **null with a wide interval**, not a refutation. On the 2021-22
season (615 scored games) the replacement arm is +0.003868
[−0.004013, +0.011707], |z| 0.96; on 2020-21 it is +0.000760, |z| 0.24. The
reason to read neither as evidence against the mechanism is the positive
control: the known +0.0038 availability effect itself measures +0.0048 at
|z| 0.92 on the 2021-22 subset and −0.0012 on 2020-21. A subset that cannot see
the established effect cannot resolve a smaller one. With an SE of 0.0040, the
`|z| > 3` bar needs ~6,000 scored games.

Scored on the full 6,600-row sample against the availability-only frame, the
adjustment was **+0.000818 [+0.000222, +0.001422], |z| 2.67, better in 5 of 5
folds** on the then-two-season store — the closest this had come to a signal,
still below the bar, and the reason to extend the backfill. The backfill is now
complete (every game date from January 2020 through the end of 2025-26), and
the identical arm on the completed store reads **+0.002632
[+0.001471, +0.003779], |z| 4.47, better in 5 of 5 folds** with coverage at
7,206/13,199 games (54.6%) and 73.6% of scored rows carrying a changed weight.
That **cleared the |z| > 3 bar on the criterion measurement** at the time —
reported as a signal, not a promotion, because the original pre-registered
season protocol (2020-21/2021-22) remained null.  Issue #68 then ran the same
arms on a powered forward holdout (train before 2024-10-22; test = 2024-25 plus
2025-26) and the multiplier cleared at **+0.003579, |z| 3.85**; it is now
promoted as evidence.  The canonical model is still unchanged: promotion does
not silently rewire the default frame.  Full numbers in
[the injury pipeline](injury_pipeline.md#what-it-measured-68-promoted-on-a-powered-forward-holdout).

So the block keeps its measured claim and gains a **switchable, now-promoted**
extension: the canonical model remains the 26-column frame described above. The useful transferable finding is the identity-resolution and
point-in-time machinery — 98.9% of 39,524 designations resolve to a player from
the report's own team cell, and a 15-minute pre-tip gate leaves 93% of games
covered — which is what a larger sample would be scored on.

## The honest boundary

The block improves **forecasting**: +0.0039 nats at |z| = 4.78, three
instruments agreeing. It is fully **priced** at the close: replacing `P+M` by
`P+M+A` in the market-anchored stack changes log loss by −0.00004
[−0.00027, +0.00019], a tight null whose CI upper bound is 10× below the
0.002-nat minimum effect
([H2](market_anchor.md#h2--availability-beyond-price-a-tight-null)). Both
statements are results, and neither cancels the other: the availability block
is a real improvement to a model that still loses to the market by 0.02268
nats at |z| = 7.28.

## Where the data comes from

The block reads trailing player logs and the prior game's participants; the
injury-designation side of "who is available" is implemented as an optional
same-game multiplier on top of it ([the injury pipeline](injury_pipeline.md))
but is **not** part of the canonical block's measured claim. The block's signal
is minutes **expectation**, not designation status.
