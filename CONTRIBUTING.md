# Contributing

## What this project is

An honest research pipeline for NBA game prediction. Its central claim is a
negative one: after fixing a target leak, the model has **no demonstrated
edge** at the closing line. Every ROI figure it reports is within about one
standard error of zero, and the harness prints those standard errors so the
claim can be checked rather than taken on trust.

Please read `README.md` before changing anything. Most of what looks like an
obvious improvement in this codebase is not one, and several have been measured
already.

## Rules for what may be committed

This repository is public and has **no LICENSE file**, so default copyright
applies and no reuse rights are granted. Keep it that way: adding a permissive
license would grant rights the maintainer has not chosen to grant.

### Never commit a configuration that shows an edge

If a model configuration ever produces a ROI whose confidence interval excludes
zero, **do not commit it** — not as a file, not in a filename, not as a ledger
line, not in an issue or a pull request.

This is not hypothetical caution. The repository already publishes search
outcomes in filenames:

```
Models/XGBoost_Models/XGBoost_68.9%_ML_md3_eta0p077_sub0p741_col0p891_cbl0p874_...json
                   ^d3  ^eta      ^sub        ^col        ^cbl
```

Those particular values were found on the *leaky* dataset, so they encode no
edge. But they establish the convention, and the convention would publish a
real one automatically. This is tracked as issue #18.

What is safe to publish:

| Publishable | Not publishable |
|---|---|
| Model artifacts (`.keras`, XGBoost `.json`) — these give predictions, not method | The search space and objective that found a winning configuration |
| Quality bars and acceptance criteria | Specific feature formulas or tuned hyperparameters of a working model |
| Measurement infrastructure | Run ledgers for profitable configurations |
| Negative results, including failed experiments | Which configuration produced the best ROI |

A local override channel already exists: `.env` and `local_settings.py` are
gitignored. Use them.

### Keep the evaluation harness public — for now

`scripts/evaluate.py` is, functionally, an edge-searching engine: it takes a
model, scores it against the closing line across confidence thresholds, and
reports standard errors. Combined with a dataset and a search space, that is
close to a complete recipe for hunting an edge.

It is published deliberately. **There is no demonstrated edge to protect**, so
transparency buys credibility that matters more than the hypothetical. That
calculation changes the moment a configuration shows a non-zero ROI outside its
standard error — at that point, stop, and revisit this section before
publishing anything further about the search.

## Before you open a pull request

- `pytest` and `python -m unittest discover` both pass on Python 3.10 and 3.11.
- No entry point crashes. Five of eleven used to: three referenced a table that
  does not exist, and three defaulted to a same-day-join table that still
  carried the leakage this project exists to remove. Run them.
- Every metric you report has a standard error attached. A point estimate
  without an interval is not a result here.
- Any change to the dataset is accompanied by a regenerated
  `Data/manifest.json`, and the row count and column list are reported. Column
  **order** is part of the model contract; `src/Utils/features.py` exists to
  enforce it.
- New features carry an as-of assertion proving they cannot see the game they
  predict. `Create_Games.py` joins strictly *before* the game date for exactly
  this reason.

## Things that have already been measured, so you do not have to

- **Recency (EWMA) features do not help the moneyline target.** The block is
  sound — `PTS_ewma` correlates +0.462 with a team's actual next-game points —
  but merging it *significantly degrades* the model in every configuration
  tested. Full numbers in issue #20.
- **The 52 league-wide `*_RANK` columns are redundant.** Dropping them changes
  log loss by +0.0018 ± 0.0016, i.e. not significantly, in either direction.
  Issue #19.
- **100 hyperparameter trials are ~8x noisier than the signal they resolve.**
  10–15 trials is the entire signal. Issue #7.
- **The sigmoid calibrator does nothing** (−0.00008 on one run, +0.00214 on a
  replication — both inside noise) and isotonic is actively worse. Issue #8.
- **Purged walk-forward CV is a no-op here**; the real defect is fold geometry.
  Issue #9.

If you propose one of these, bring a measurement that beats the paired
bootstrap, not an argument.