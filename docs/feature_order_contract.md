# The feature-order contract

**Repo:** `/home/Joel/.cache/nba-fix-worktree` (branch `improve-model-v2`)

XGBoost and Keras are fed a **positional matrix**. A frame with the right columns
in the wrong order raises nothing and predicts wrong, so "the columns match" is
not a sufficient check — the *order* is the contract, and this is how one
definition of it is kept in one place.

## The three pieces

| piece | file | role |
|---|---|---|
| the trainers' drop list | `src/Train-Models/training_common.drop_columns_for()` | what every trainer removes from the dataset |
| the inference-side mirror | `src/Utils/features.expected_feature_order()` | the ordered column names a model is fed |
| the check | `src/Utils/features.validate_feature_frame()` | rejects a frame that matches as a set but disagrees in sequence |

`expected_feature_order()` mirrors `df.drop(columns=drop_columns,
errors="ignore")`: absent entries are ignored, and every surviving column keeps its
position in the dataset. The dataset order is the contract.

The four lists in `src/Utils/features.py` (`ML_DROP_COLUMNS`,
`UO_LABEL_DROP_COLUMNS`, and the two `NN_*` names) are **computed from
`drop_columns_for` at import**, not transcribed. `scripts/evaluate.py` derives its
list the same way. The NN moneyline list used to order `OU` before `OU-Cover`;
dropping is membership-based, so the surviving order was identical either way, and
`Tests/test_drop_columns.py` now pins that invariance instead of asserting it in a
comment. It also fails if any module under `src/` or `scripts/` re-declares a
`*DROP_COLUMNS` list literal.

`validate_feature_frame()` reports missing and extra columns first, because those
are the common cause. When the two sets agree but the order does not, it reports a
**MISORDERING** — the failure that would otherwise pass silently and predict wrong.
Both runners (`src/Predict/XGBoost_Runner.py`, `src/Predict/NN_Runner.py`) call it
at the inference boundary.

## What the drop list covers

Three groups go, for every trainer:

- the labels and the score;
- the raw team-name and date keys that are joined in rather than learned from;
- the duplicate index columns pandas emits when the dataset was assembled.

Both labels are always dropped, so the list never depends on which of the two
targets the caller is training. `GAME_DATE_COLUMN` goes for the same reason `Date`
does: the leak-free join in `Create_Games` makes `Date` the as-of **snapshot**
date — strictly before the game — while `Data/OddsData.sqlite` stores the game
date, so the two are one day apart and no longer interchangeable. The trainers
still sort by `Date`, because it is the honest as-of marker.

**The closing total is dropped for a moneyline trainer and kept for a totals
trainer.** Every totals model must agree on that, because the runner feeds them all
the same column order and an input width none of them trained on is worse than a
worse model.

`include_team_ids` additionally drops `TEAM_ID` / `TEAM_ID.1`. Only the two logistic
regression scripts ask for it; the XGBoost and NN trainers leave the ids in. That
asymmetry predates this module and is reproduced verbatim rather than tidied away
here. On the datasets that ship with the repository those two columns are absent, so
`drop(columns=..., errors="ignore")` and the membership test in the XGBoost trainers
are both a no-op today.

## Why the persisted list exists

Each trainer writes the ordered column list next to the model:
`Models/XGBoost_Models/<model stem>_features.json`. The model file itself is a
positional matrix, so **nothing inside the `.json` records which columns produced
it** — the sibling file is the only record.

## Checking it yourself

```bash
python scripts/check_feature_alignment.py --use-live
```

This reports the same disagreement from the **training** side, against live frames
rather than the shipped dataset. Flags: `--dataset`, `--frame-ml`, `--frame-uo`,
`--use-live`, `--nn-ml-model`, `--nn-uo-model`, `--xgb-ml-model`, `--xgb-uo-model`.

Its default dataset is `training_common.DEFAULT_DATASET`, not a literal. The
default used to be `dataset_2012-26_new`, a table that does not exist in
`Data/dataset.sqlite`, so a bare `python scripts/check_feature_alignment.py` died
inside `sqlite3` before printing a single number. Same drift this script was fixed
for.

## The copy that drifted

This script used to declare its own copy of the trainer drop list, and the copy had
drifted from the six trainers that call `training_common.drop_columns_for`:

- it kept `OU` in the moneyline list, which every moneyline trainer drops (the
  closing total is a totals-model feature);
- it omitted `GameDate` and both pandas `index` columns, which every trainer drops;
- it named `TEAM_ID` / `TEAM_ID.1`, columns `Create_Games` strips before writing the
  table, so they matched nothing and `117 - 10` never reconciled.

**Six of its ten entries were still correct, which is the worse shape:** the checker
agreed with the trainers on most of the contract and disagreed exactly where a
permutation could hide. Both lists are now the trainer function's own return values,
and `Tests/test_feature_alignment_checker.py` pins the equality — list **and** order
— so the copy cannot start drifting again.

What was wrong against the live 117-column `dataset_2012-26` table: the two lists
produced 109 and 110 features where the trainers produce 106 and 107, and **no
committed model has either width**.