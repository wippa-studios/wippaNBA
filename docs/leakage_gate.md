# The reusable leakage gate

`src/Utils/leakage_gate.py` answers one question about any feature builder:

> Does a row's feature value depend on information that was not available when
> that row's event had to be predicted?

It is the reusable artifact issue [#46](https://github.com/wippa-studios/Legit-NBA-Predictions/issues/46)
asked for, and it incorporates the separation issue
[#48](https://github.com/wippa-studios/Legit-NBA-Predictions/issues/48) required:
**a finding that the future was read is kept in a different collection from a
finding that same-date records were read.** The two are not the same claim and
must not share a verdict.

The module imports only `numpy` and `pandas`. It has no dependency on `nba_data`,
the 382 MB download, or this repository's layout. Copy the file into another
project and pass it that project's feature builder.

---

## The three probes

Suppose a probe event happened on date `D`. The gate rebuilds features three
ways and compares the probe row's values against the full-frame build.

| probe | records the rebuild uses | a change means | where it is reported |
|---|---|---|---|
| **`as_of`** | every record dated **on or before `D`** | the feature reads a **later-dated** record. **This is a real leak.** Trust it. | `report.leaks` |
| **`blind`** | records dated **strictly before `D`**, plus the probe event's own row and any record on `D` the caller marked `known_before` | the feature reads **other records sharing its date** — tonight's box score, a same-day snapshot, the other half of a doubleheader. This is the membership shape. It may be a leak or a legitimate pre-game input. **Investigate; do not conclude.** | `report.membership_dependencies` |
| **`self`** (optional) | the same as `blind`, but the probe row is replaced by a stub with its non-identity columns nulled | the feature reads **its own row**. A trailing average does this by construction. Diagnostic only; never a leak verdict. | `report.self_dependencies` |

### Why `blind` keeps the probe event's own row

An earlier design removed the probe row entirely. That caught both membership
leaks and legitimate self-reference, which sounds acceptable until you see what
it does to a clean feature:

```python
d["roll"] = d["val"].rolling(5, min_periods=1).mean()   # trailing, not centred
```

Deleting the probe row deletes the last input of its own trailing window, so the
feature moves. The old `blind` flagged it every time. A gate that cries wolf is a
gate whose users learn to ignore it, and the false alarm makes the genuine
membership findings it was built for easier to dismiss.

Keeping the event's own row fixes that without weakening the membership probe: a
feature that reads **itself and earlier records** is untouched, while a feature
that reads **other records dated `D`** still moves. The probe that asks about the
event's own row survives as the separate, optional `self` diagnostic.

### What `known_before` is for

Some records dated `D` are genuinely known before the event: the published
schedule, a rest-day calculation, a pre-game rating. Others are not: tonight's
box score, a snapshot pulled after tip-off, the result of the other game in a
doubleheader.

If the frame has a boolean column for this, pass `known_before_col`. Rows dated
`D` with `True` survive the blind rebuild; rows with `False` do not. Without the
column, only the probe event's own row survives that date, which is the
conservative default.

> Mark what was available **before the first event on that date**. Do not mark a
> snapshot `True` because it was *created* that morning; mark it on whether it
> contains the event's own result.

---

## Usage

```python
import pandas as pd
from src.Utils.leakage_gate import audit_future_dependence


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one numeric row per event, indexed by the event's row_key."""
    ...


report = audit_future_dependence(
    build_features,
    frame,                       # the raw record universe
    date_col="game_date",
    row_key="game_id",
    n_probe=200,                 # at most one probe per distinct date
    known_before_col="known_before",   # optional
    include_self_probe=True,     # optional diagnostic
    self_keep_cols=("home", "away"),   # what the stub keeps
    seed=0,
)

print(report.summary())

for finding in report.leaks:
    print("LEAK", finding.feature, finding.changed, finding.probes)

for finding in report.membership_dependencies:
    print("INVESTIGATE", finding.feature, finding.examples)
```

The feature frame must be numeric (booleans are fine) and indexed by row keys
that exist in the input frame. The gate compares the NaN pattern exactly: a cell
that backfills under truncation is a finding, not a rounding artefact. Float
noise is absorbed by `tol` (default `1e-9`, three orders of magnitude above the
`~1e-12` summation-order noise measured in this repository and far below any
real move).

### Reading the report

| property | meaning |
|---|---|
| `report.leaks` | `as_of` findings. A feature demonstrably reads a later-dated record. |
| `report.membership_dependencies` | `blind`-only findings. A feature reads other records sharing its date. **Not proof of leakage.** |
| `report.self_dependencies` | `self`-only findings. A feature reads its own row. Expected for self-referential features. |
| `report.clean` | `True` only when there are no leaks **and** no membership dependencies to investigate. |
| `report.future_dependence_free` | The narrower claim: no feature reads a later-dated record. |
| `report.inconclusive` | No probe could be rebuilt. `clean` is false. |
| `report.skipped` | Probes whose rebuild raised, with the exception text. |

`report.summary()` states the distinction in words. It never calls a blind
finding a leak:

```
Leakage gate: 7 feature rows, 6 probe(s) over 6 date(s), 1 feature column(s), tol=1e-09
  as_of  (later records)       leaks: none
  blind  (same-date records)   membership dependencies: 1
    - player_elo_actual: changed on 6/6 probes, max |delta| NaN transition
        e.g. g1 (2024-11-01): 8320 -> NaN
  self   (own event row)       diagnostics: 1 (expected for self-referential features)
    - rolling_pre: changed on 6/6 probes, max |delta| 0.8
Verdict: NO CONFIRMED LEAK, but 1 membership dependency(ies) to investigate.
         A blind finding is not proof of leakage.
```

---

## The fixtures: a gate must be shown to fire

`Tests/test_leakage_gate_artifact.py` pins six fixtures with no download and no
network. Three are the known-fatal patterns in this project's history; three are
their honest counterparts.

| fixture | what it encodes | result |
|---|---|---|
| `headline_leak` | the **68.9% headline**: each game joins the snapshot dated the same day, which already contains the result | `blind` finding |
| `headline_clean` | the same feature against each team's most recent **prior** snapshot | passes |
| `elo_actual` | **player-Elo summed over the players who actually played** — the \|z\| = 5.18 membership leak | `blind` finding |
| `elo_prior` | the same ratings over the previous game's squad | passes |
| `availability_actual` | **expected minutes over tonight's actual squad** — the contaminated construction | `blind` finding |
| `availability_strict` | expected minutes over the previous game's squad — the shipped block | passes |

Two more pin the #48 properties:

| fixture | what it pins | result |
|---|---|---|
| `rolling_pre` | a strictly causal trailing mean that reads **its own row's pre-game value** | passes `blind`; appears only in `self_dependencies` |
| `tomorrows_label` | a feature that reads the **next game's result** | `as_of` **leak**; not a membership finding |

The `rest_and_travel` construction is also a passing control: it reads the
published schedule, so neither probe moves it.

Run them:

```bash
pytest Tests/test_leakage_gate_artifact.py -q
```

---

## What the gate cannot see

1. **An outcome stored as a column on the event's own row.** The blind probe
   keeps that row, so it cannot tell an outcome column from a pre-game input.
   Model the outcome as a separate dated record (a snapshot or box-score row),
   which is how the three known leaks here were built, or use the `self` probe
   and ask which of the row's own columns are knowable at prediction time.
2. **A leak in the label's construction.** The gate audits the feature builder,
   not how `y` was assembled. Train on a wrong label and every probe passes.
3. **Statistical leakage through the evaluation protocol.** Fitting an imputer,
   a scaler or a shrinkage parameter on held-out rows is invisible here. That is
   a split-discipline bug, not a feature-build bug.
4. **A dependency that is causal but unattainable.** A same-date record can be
   genuinely known before tip-off (the schedule) or unknowable (tonight's box
   score). The gate reports the dependency; the caller has to rule on whether the
   record was available. That is what `known_before_col` is for.

So do not describe a clean report as proof that a pipeline is leak-free. It is
proof that the three questions above came back negative on the dates probed.

---

## Cost and CI

One full build plus two builds per probe date (three with `include_self_probe`),
capped by `n_probe`. Probe dates are sampled deterministically from `seed`, one
row per date, so the same frame and seed produce the same audit. On the
repository's own data, point it at the three builders named in
`Tests/test_leakage_gate.py` with a small `n_probe`; on a synthetic fixture, the
whole suite runs in seconds.

CI runs `Tests/test_leakage_gate_artifact.py` as a test of the gate itself. The
three known leaks are the regression tests; the clean fixtures are the
anti-vacuity controls. If a future change makes the gate stop firing on the
player-Elo membership case, CI says so.

---

## Pointing it at another repository

1. Copy `src/Utils/leakage_gate.py` (it has no imports from this repository).
2. Write a `build_features(frame)` that returns the repo's feature frame, indexed
   by an event id.
3. Decide what `known_before` means for that repo. If it has no such concept,
   omit the column: the blind rebuild keeps only the probe event's own row.
4. Run it with `n_probe` small enough to afford, and read `.leaks` first.

The survey of ~45 public NBA repositories did not run the gate on any of them:
each needs its own feature builder, and a README does not expose one. The
`Gate` column in [`repo_survey.md`](repo_survey.md) records that honestly rather
than guessing from prose.

Related: [#46](https://github.com/wippa-studios/Legit-NBA-Predictions/issues/46),
[#48](https://github.com/wippa-studios/Legit-NBA-Predictions/issues/48),
[#32](https://github.com/wippa-studios/Legit-NBA-Predictions/issues/32).
