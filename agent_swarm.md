# The agent swarm

> How most of the measurement work was done, what it found and what it got wrong. Part of the [Legit NBA Predictions README](../README.md).

---

## The agent swarm

Most of the measurement work in this repository was done by a swarm of agents
sharing one machine, communicating through an append-only message log. It is worth
describing plainly, because **how the work was done turned out to matter as much as
what it found** — including the parts that went wrong.

<details>
<summary><b>How it worked, what it found, and what it got wrong</b> (click to expand)</summary>

### How it worked

Agents posted to a shared JSONL bus under `~/Desktop/agents/inbox/`, in five kinds:
`feature`, `config`, `result`, `failure`, `note`. **102 messages.** No daemon, no
database — a crashed writer costs at worst one truncated line.

The design choice that carried the most weight: **`failure` was the largest category,
35 of 102.** The stated rationale is the right one — *"the expensive thing in a sweep
is re-running something that already failed."* Every dead end below was measured once
and never again.

### What it found

**One mechanism works, and it is the obvious one.** Expected player minutes beat the
bar:

| frame | cols | log loss | vs bar |
|---|---|---|---|
| bar (E14, `home_split_only`) | 43 | 0.618307 | — |
| + availability, as shipped | 47 | 0.614544 | +0.003763 |
| **+ availability, one contrast column** | **44** | **0.614337** | **+0.003970** |
| + availability + team Elo (promoted) | 48 | 0.612523 | +0.005784 |

**The win turned out to be a simplification.** `availability_features` emits four
columns, and two of them — `home_expected_minutes_diff` and `away_expected_minutes_diff`
— are *exactly* `d` and `−d`. **Three of the four are redundant**, and the single
contrast column does the whole job in fewer columns.

At |z| = 2.63 that is still **below this project's own bar**, and it is reported that
way. It is the best-measured feature effect here, not a discovery.

### What it closed

| lever | result |
|---|---|
| model family (85 arms, 8 families) | none beat the bar |
| regularisation (C sweep, 1 → 100) | flat; C=1.0 already optimal |
| calibration, nested honestly | null to harmful |
| feature selection (30-arm L1 + elastic-net) | best \|z\| 0.285 |
| bagging, subsampling, halflife averaging | null; nothing clears any bar |
| training window / recency weighting | worse at every setting |
| era drift | dead, measured |
| prior-season play-type (50 cols) | harmful, and harm **scales with column count** |

The last one has a mechanism: 5 columns cost −0.0004, 10 cost −0.0009, 50 cost
−0.0049. Monotone in width, never positive. **That is variance, not information** —
and it fits the learning curve, where fold log loss improves monotonically with
training size.

**The unifying conclusion: the model's error is bias, not variance. Only new
information can help.** That is why every variance-reduction lever is flat and why
the project stopped adding model arms.

### What it fixed

- **Two real bugs in the leakage gate** — the `blind` variant did not blind data read
  from disk, and a missing pre-event declaration made an honest builder look like it
  leaked. Both were found by *trying to use it*, which is how a tool should fail.
- **Its own shims.** An audit shim caught only **16 of 30** deliberately-injected
  leaks; the gap was found and closed.
- **A positive control on everything.** For the availability block the gate reports
  `membership: ok=False, clean=False` on the leaky variant and `clean=True` on the
  honest one, in the same run.

### What it got wrong

Worth recording, because the errors are the same classes this repository already
knows:

1. **A false claim about the odds data, published three times.** *"The database has
   zero rows for 2024-25 and 2025-26; market features are structurally unusable."*
   That was an artefact of the `load_odds` bug, not a property of the data. It was
   retracted by message id — after the fact.
2. **A vacuous robustness check.** A three-seed sweep was reported as replication; the
   estimator is deterministic, so all seeds returned identical numbers and the seed
   axis carried no information.
3. **No supersession.** An append-only bus cannot retract. A reader scanning by
   `kind=failure` hit the false entry first and stopped. Two notes carrying the same
   false claim were never retracted at all.
4. **A headline that was behind the state.** The availability number was published
   against the 43-column bar while a promoted 48-column frame already banked it —
   retracted as a supersession notice.

### What it produced

`f1f8e47f` and `7e3ef22d` — three feature modules, nineteen round-3 scripts, the
experiment logs, and a 283-line summary at `results/round3_summary.md`. Full suite
**1,165 passed + 103 subtests** at those commits, and CI green on both Python
versions.

</details>
