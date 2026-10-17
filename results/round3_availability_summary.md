# Round 3 — availability block, calibration, and the leakage-trap work

> **Scope note.** This is the *availability/calibration* slice of round 3, not the
> whole of it. For the team-Elo work, the promoted 48-column frame (0.612523) and
> the full set of closed lanes, read `results/round3_summary.md`. Where the two
> overlap, that document is the superseding one: it already carries availability in
> the promoted frame, so the headline here (0.614337) is measured on the older
> 43-column bar and is **behind** current state, not ahead of it. The part of this
> work that is still novel is the *simplification* in §1.

Bar: E14, 43 columns, `LogisticRegression(C=1.0)`, 5 walk-forward folds
(`initial_frac=0.5`, `gap=0`), 6,600 out-of-fold games, **log_loss 0.618307**.
Everything below is out-of-sample within that protocol. Nothing here is fitted on
a test block.

## The one result worth shipping

**Availability, as a single contrast column. 0.618307 → 0.614337 (+0.003970 nats).**

| | log_loss | brier | auc | accuracy |
|---|---|---|---|---|
| E14 baseline | 0.618307 | 0.214601 | 0.709251 | 0.655455 |
| + availability | **0.614337** | **0.212777** | **0.715080** | **0.659848** |

Confirmed three ways: 5 bootstrap seeds (z 2.52–2.69, every CI excludes zero),
all 5 folds positive (+0.0013 … +0.0087, mean 0.00397, sd 0.00297), all four
metrics improve.

**And it is a simplification.** `nd.availability_features` ships *two* columns,
`home_expected_minutes_diff` and `away_expected_minutes_diff`, where the second
is defined as the exact negation of the first (r = −1.0). One column does the
whole job and scores marginally better. The two-column pair was one degree of
freedom wearing two hats.

Column: `av_prev_sum_diff` — expected minutes summed over the squad that played
the team's **previous** game. `scripts/round3_avail.py::availability_block`
reproduces the shipped feature to `maxabs = 0.0`, so this is a verified superset.

## What did not work (the expensive part)

| lever | verdict | best number |
|---|---|---|
| Model family — 85 arms, 8 families | dead | `logreg C=0.3` +0.000012; **all 70 tree arms worse**; blend +0.000575. 0 of 85 survive BH FDR 0.05 |
| Probability calibration — nested | dead, slightly harmful | temperature −0.00054, platt −0.00080, isotonic −0.00798 |
| Roster-shape extensions over availability | dead | concentration +0.00094 (z 0.63), depth +0.00011, min-per-player +0.00016, role-weighted +0.00024 — all CIs straddle zero |
| Training-window recency | dead | last-K-seasons only: K=5 0.619014, K=4 0.621093, K=3 0.622048 vs 0.618307 |
| Round-2 features (12 arms) | dead | fatigue, altitude, travel, shots, tracking, odds — all null or worse |

`C=1.0` was already the right pick. The model was never the problem.

## Two traps, measured rather than assumed

**1. Membership leaks are invisible to future-truncation.** The known-leaky
feature (expected minutes summed over who *actually* played) shifts **0.0000**
under `as_of` truncation — because `expected_min` is already a lagged EWMA, so
deleting the future changes no earlier value. It is caught only by deleting or
scrambling the probe date's *own* player rows.

`src/Utils/leakage_gate.py` does **not** do this: its blind variant cuts the games
frame, but `nd.player_minutes` re-imports those `GAME_ID`s from disk. Anyone
gating a player-level feature with that gate alone is testing nothing.

Getting the test right took three attempts. v1 (truncate, compare prefix) called
the leaky feature clean. v2 (truncate source, compare probe-date rows) flagged the
honest block too — `squad` is built per game from player rows, so deleting a
game's rows removes that game from the table and *everything* goes NaN. v3
scrambles tonight's `PLAYER_ID`s while preserving row structure; honest features
are bit-identical, membership leaks move.

**2. Calibration fitted on the OOF predictions is fiction.** Isotonic scored
**0.613431** in-sample-on-OOF — a +0.0049 nat "win", about the size of the only
real feature effect in this repo. Nested honestly it is **0.626289**, i.e. 0.0080
*worse* than baseline. Inflation +0.0129 nats. Always nest.

## Two bugs worth not rediscovering

- `pandas 3.0`: `groupby(...).head(5)` returns a Series with a **flat** index and
  silently drops the group keys → `KeyError`. And `groupby(...).nlargest(5)` cost
  **256.6 seconds** on a 280k-row frame where the entire rest of the builder takes
  4 seconds. Global `sort_values` + `cumcount` is ~1s and identical (asserted).
- `rc.score` / `rc.delta_vs` reject the NaN prefix before the first test block.
  Round 2 always masked first; an unmasked call dies on the first arm.

## Where the headroom is not

Not the estimator, not calibration, not recency. The 67-feature ablation says only
`home_away_split` and `schedule` earn their place and `player_elo` actively hurts,
and per-fold log loss falls monotonically with training size (0.62552 @ 6599 rows
→ 0.59899 @ 11879; AUC 0.706 → 0.735). That is a learning curve: the model is
**data-limited**, which points at variance reduction and at making the team-level
averages say less about who is actually available.

## Files

| file | what |
|---|---|
| `scripts/round3_avail.py` | the block, the 8 pre-registered arms, the leakage suite + control |
| `scripts/round3_avail_followup.py` | each extension vs the single column, paired |
| `scripts/round3_calib.py` | nested calibration + the in-sample contrast |
| `scripts/round3_models.py` | the 85-arm model-family sweep |
| `results/round3_*.jsonl` | every run, including the crashes |

**Uncommitted** — the worktree's parent repo was wiped by a server restart, so
`git` no longer resolves here. Everything is files-only.