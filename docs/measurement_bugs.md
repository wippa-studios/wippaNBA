# Two measurement bugs worth reading

**Repo:** `/home/Joel/.cache/nba-fix-worktree` (branch `improve-model-v2`)

Both bugs below produced a number that looked fine. One reported a losing strategy
as a 13.54% profit; the other made a whole test suite crash in the way the guard
was written to prevent. Neither would have been found by reading the code — both
were found by a disagreement between two implementations of the same question.

Each one is pinned by a test whose name is the bug. A third section lists the
smaller guards that came from the same process.

---

## 1. The ROI denominator: a loss divided by a negative total

`scripts/closing_line.py` printed this in an earlier revision:

| threshold | bets | model ROI | \|z\| | model + availability ROI | \|z\| |
|---|---|---|---|---|---|
| 0.55 | 815 | **+11.76%** | 2.22 | **+13.54%** | 2.54 |
| 0.60 | 638 | +6.19% | 1.02 | +3.52% | 0.59 |
| 0.65 | 478 | −2.11% | 0.30 | +0.07% | 0.01 |

**Those two positive numbers were a bug, not a finding, and the bug inverted the
sign.** The corrected figures, every threshold negative or flat:

| threshold | bets | model ROI | \|z\| | model + availability ROI | \|z\| |
|---|---|---|---|---|---|
| 0.55 | 815 | −8.55% | 1.62 | **−9.37%** | 1.78 |
| 0.60 | 638 | −5.02% | 0.86 | −2.74% | 0.46 |
| 0.65 | 478 | +1.76% | 0.25 | −0.06% | 0.01 |

### What was wrong

The stake is `p_model − p_raw`, which is **negative whenever the market is the
more confident side** — at a 0.55 threshold, 1,382 of 4,076 games. Those are not
short positions. They are not bets, and their stake is not money. But the ROI
divided total profit by `stake.sum()`, which summed them:

```
positive stakes   812, sum  +69.22   <- money actually at risk
negative stakes  1382, sum -117.10   <- not bets at all
stake.sum()             -47.88        <- the denominator as coded
total profit            -6.48         <- a LOSS
profit / sum(positive)  -9.37%        correct
profit / stake.sum()   +13.54%        as it was reported
```

A loss divided by a negative total reports as a profit. The same mistake was
inside the bootstrap, so the standard error was computed over the wrong population
too — a wrong ROI with a plausible-looking SE, which is harder to catch than
either alone.

### Why it survived as long as it did

The bet *count* was right (812), every individual stake was sensible, and every
intermediate number was plausible. Only the ratio was wrong. Sitting on top of a
best-of-six threshold sweep, `+11.76%` and `+13.54%` became the two best cells —
which is exactly what a selection effect looks like, and why it was initially
written off in the README as multiplicity rather than arithmetic.

**That explanation was wrong, and being wrong in the conservative direction is
still wrong.** An independent reimplementation (`scripts/edge_holdout.py`) computed
the same 812 bets and got −9.37% from a different code path; that disagreement is
what found it.

The lesson is the disagreement, not the arithmetic. Two implementations of the
same measurement agreeing is evidence; one implementation agreeing with itself is
not. Where a number is load-bearing enough to change a conclusion, it is worth
computing twice by different routes.

### The fix

Both the mean and the bootstrap now restrict to placed bets before aggregating.
Pinned by `Tests/test_roi_denominator.py`, whose headline test asserts the
invariant the bug violated: *the sign of the ROI is the sign of the total profit.*

---

## 2. The data guard: a directory that exists is not data in it

`_require_downloaded_data()` in `src/DataProviders/nba_data.py` raises
`unittest.SkipTest` so that a fresh clone and a CI runner — neither of which has
the 382 MB NBA download — skip the model test suites instead of failing them. CI
on `master` was red for two consecutive commits anyway, every error identical:

```
ValueError: No objects to concatenate
  nba_data.load_team_games -> pd.concat([])
```

**The guard checked for a *directory*, not for *files*.** An empty
`game_logs/` passes an `is_dir()` check, every subsequent glob returns nothing,
and `pd.concat` is handed an empty list — the exact failure the guard was written
to prevent, reached by the one route it did not cover.

A **partial** download failed the same way by a different route: `game_logs/`
populated but `team_gamelogs/` still empty passed a single-directory check and then
blew up on the second glob. So checking only one directory was not enough either.

Two defects, both fixed and both pinned in `Tests/test_data_guard.py`:

1. the guard tested for a directory rather than for files;
2. `DATA_ROOT` defaulted to one developer's home directory,
   `/home/Joel/Desktop/Legit_NBA_Predictions/data/nba`. Someone who ran the
   downloader exactly as the README says, into `./data/nba`, was told the data was
   missing.

The guard now globs `*.csv` in every directory the core loaders read, and reports
which ones are empty in the skip message:

```
NBA data not found at <root> (no CSVs in: game_logs, team_gamelogs).
Download it with `python scripts/download_nba_data.py`, or set NBA_DATA_ROOT to
an existing copy.
```

Every scenario in the test runs in a **subprocess**: `DATA_ROOT` is resolved at
import time, so reloading the module in-process to change it is fragile and
order-dependent — an earlier draft of that test did exactly that and failed three
ways. A subprocess is also simply how CI sees it: a fresh interpreter, an
environment variable, and a directory that may or may not exist.

### Why the check lives in the shared module and not in the tests

Two runners execute this suite — `pytest` and `python -m unittest discover` — and
they honour different skip mechanisms. pytest understands markers registered by a
conftest hook; unittest ignores pytest entirely and only understands `SkipTest`
raised during `setUp`. Putting the check in `src/DataProviders/nba_data.py` makes
one change satisfy both, instead of editing every test module and keeping the two
in sync.

SkipTest is a legitimate outcome here, not a failure: these tests cannot run
without an input the project does not ship.

---

## What the two have in common

Neither bug was found by reading the code, and neither produced an obviously wrong
intermediate value. Both were found by something disagreeing:

| bug | what disagreed |
|---|---|
| ROI denominator | `edge_holdout.py` recomputed the same 812 bets from a different code path and got −9.37% instead of +13.54% |
| data guard | CI, which has no downloaded data, failed in the exact way the guard existed to prevent |

That is the reason both are pinned by tests rather than fixed once. See also
[the player-Elo leak](player_elo_leak.md) for the third one, where nothing
disagreed — the number was simply a function of the answer.