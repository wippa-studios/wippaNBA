# Verifying and testing

> How to reproduce every number and run the test suite. Part of the [Legit NBA Predictions README](../README.md).

---

## Reproduce every number in this README

Nothing needs manual setup:

```bash
python scripts/evaluate.py
```

It opens with a leakage gate and **refuses to report accuracy at all** if the dataset
is contaminated, so a passing run is itself evidence that the join is clean. Then it
prints the baselines, the trained model, the closing-line favourite, and ROI across
edge thresholds with standard errors.

Useful flags: `--dataset`, `--test-size`, `--model {logreg,gbdt,none}`, `--skip-odds`.
See `scripts/README.md` for the gate's checks and degradation rules.

### The experiment log

`scripts/run_experiments.py` is the iterative loop, made mechanical so the search
is an artefact rather than a memory. Every experiment is **declared with its
rationale before it is run**, executed on identical walk-forward folds, and
appended to `results/experiments.jsonl`.

```bash
python scripts/run_experiments.py --stage 1        # baselines and regularisation
python scripts/run_experiments.py --stage 2        # feature-block ablations
python scripts/run_experiments.py --stage 3        # RF / HistGB / XGBoost
python scripts/run_experiments.py --stage elo      # player-Elo block
python scripts/run_experiments.py --stage survey   # models named in the repo survey
```

It reports log loss as primary, with Brier, AUC and accuracy secondary and
**explicitly not ranked**. It also reports the **train-vs-test gap** for every run,
so an improvement bought by memorising the training folds cannot hide, and it prints
the **selection cost** of the winner: the gap between best and median, which is the
expected size of a maximum of N draws. See
[the experiment log](results_pipeline_a.md#the-experiment-log-17-pre-registered-runs).

`scripts/feature_block_ablation.py` does the same job for feature blocks, with the
paired bootstrap and the Bonferroni accounting — `--seed 42/137/2718` and
`--splits 8` are the replication, and see
[the ROI arm](results_pipeline_b.md#a-number-in-the-committed-output-that-looked-good-and-was-a-bug) for
`scripts/edge_holdout.py`, which asks whether a threshold chosen by looking at the
data survives being chosen somewhere else.

## Run the tests

```bash
pytest Tests/                       # or: python -m unittest discover -s Tests -t .
```

Most tests need no network access and no TensorFlow or XGBoost import. The model
suites under `src/Models/` do need the downloaded NBA data described above; when
`data/nba` is absent they **skip** rather than fail, so a fresh clone still gets a
meaningful signal. That guard lives centrally in `src/DataProviders/nba_data.py`,
which raises `unittest.SkipTest` — so there is exactly one place where "needs data"
is decided.

### Current state: 1,287 passing + 104 subtests, 0 failing

40 files, measured individually — `test_injury_data.py` (110), `test_ratings.py` (106),
`test_training_layer.py` (94), `test_line_movement.py` (85), `test_metrics.py` (71),
and 35 more down to `test_kelly_criterion.py` (5). **Every guard was added because
something was measured to be wrong**, and several because a test caught a bug I
introduced in the same commit. The per-file table, what each one protects, and why
each exists is in [docs/test_suite.md](test_suite.md) (written when the suite
was 35 files; `test_injury_data.py`, `test_leakage_gate.py` and three more have
joined since); the three largest post-mortems are
[the ROI denominator](measurement_bugs.md) (a loss divided by a signed sum
reported as +13.54%), [the data guard](measurement_bugs.md#2-the-data-guard-a-directory-that-exists-is-not-data-in-it)
(a directory that exists is not data in it — the one route past the check that
missed), and [the player-Elo leak](player_elo_leak.md) (a |z| = 5.18 result
that was a function of the answer).

CI runs on Python 3.10 and 3.11, plus a matrix job that exercises the totals trainer
on both sides of the scikit-learn 1.7 `multi_class` removal.
