"""Shared machinery for the two logistic regression trainers.

`Logistic_Regression_ML.py` (moneyline) and `Logistic_Regression_UO.py` (totals)
differ only in their target column, their class count and a handful of solver
settings. Everything in this module is common to both.

Two notes on what is NOT here:

* These scripts split and scale a pandas frame, so the split helpers return
  frame slices via ``.iloc``. The XGBoost pair cuts the same boundaries out of a
  numpy matrix instead; those are genuinely different functions and live in
  ``boosting_training``.
* These scripts never persist a model. There is no MODEL_DIR and no joblib.dump
  here or in either entry point, so they have no features JSON and must not
  grow one.

Imputation is NOT done in ``prepare_data`` any more. It used to be, which meant
the medians were computed over the joined frame and the held-out rows therefore
reached training. It is latent on the shipped dataset -- there are no NaN cells
in the 106 feature columns -- but 14 of those columns would move their fill
value by more than 1% if the medians were computed on all rows. Use
``training_common.impute_split(train, *others)`` after every split instead; see
its docstring for why it returns the frames and not the medians.

The split helpers take an optional ``dates``. Pass the sorted frame's ``Date``
column and every cut is moved forward to a date-group boundary, so no split ever
separates two games played on the same night. Omit it and the cuts are plain row
counts, which is what these helpers did before and which is wrong on this data:
mean 6.45 rows share one snapshot date and the largest group is 40, and the 90%
cut lands inside one of them.
"""

import inspect

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from training_common import DATE_COLUMN, STABLE_SORT_KIND, snap_cut_to_date_boundary


def _snap(position, dates):
    """Move a split point to a date-group boundary, if the dates were supplied."""
    if dates is None:
        return int(position)
    return snap_cut_to_date_boundary(dates, position)


def prepare_data(df, target_column, drop_columns):
    """Sort by snapshot date and split features from the label.

    The sort is a stable merge sort. pandas' default is quicksort, which is NOT
    stable: rows sharing a snapshot date came out in a different order under it,
    and a train/test boundary landing inside a same-date group means the split
    cannot be reproduced from the stored table alone. Mean group size is 6.45
    rows per snapshot date and the largest is 40.

    Returns the frame UN-imputed. See the module docstring.
    """
    data = df.copy()
    if DATE_COLUMN in data.columns:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        data = data.sort_values(DATE_COLUMN, kind=STABLE_SORT_KIND)
    y = data[target_column].astype(int).to_numpy()
    X = data.drop(columns=drop_columns, errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y, data


def split_time_series(X, y, val_size=0.1, test_size=0.1, dates=None):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    val_start = int(n * (1 - val_size - test_size))
    test_start = int(n * (1 - test_size))
    val_start, test_start = _snap(val_start, dates), _snap(test_start, dates)
    X_train, y_train = X.iloc[:val_start], y[:val_start]
    X_test, y_test = X.iloc[test_start:], y[test_start:]
    return X_train, X_test, y_train, y_test


def split_train_test(X, y, test_size=0.1, dates=None):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    test_start = _snap(int(n * (1 - test_size)), dates)
    return X.iloc[:test_start], y[:test_start], X.iloc[test_start:], y[test_start:]


def split_train_calib(X, y, calib_size=0.1, dates=None):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    calib_start = _snap(int(n * (1 - calib_size)), dates)
    return X.iloc[:calib_start], y[:calib_start], X.iloc[calib_start:], y[calib_start:]


def compute_sample_weights(y, num_classes, mode):
    """Inverse-frequency sample weights for "inverse", None for the other modes.

    "balanced" is handed to scikit-learn as class_weight instead -- the same
    n / (num_classes * count) rebalancing, expressed inside the estimator, so
    it needs no sample weights here. "none" needs neither.

    "inverse" is the RAW 1 / count, which is roughly 1000x smaller in absolute
    terms than "balanced" for a dataset this size. Logistic regression penalises
    against those weights directly, so the small scale effectively multiplies C
    and over-regularises: use "balanced" for the same rebalancing at a sane
    scale. The distinction matters here in a way it does not for XGBoost, whose
    objective is scale invariant.
    """
    if mode != "inverse":
        return None
    counts = np.bincount(y, minlength=num_classes)
    class_weights = {cls: (1.0 / count if count else 1.0) for cls, count in enumerate(counts)}
    return np.array([class_weights[label] for label in y])


def build_class_weight(mode):
    """The scikit-learn class_weight argument for the chosen --class-weights mode."""
    return "balanced" if mode == "balanced" else None


def supports_multi_class_parameter():
    """Whether this scikit-learn still accepts LogisticRegression(multi_class=...).

    Version fork, following the convention already set by
    ``training_reporting.make_calibrated_estimator``: inspect the signature and
    build the kwargs, rather than branching on a version number.

    scikit-learn deprecated ``multi_class`` in 1.5 and REMOVED it in 1.7. The
    removal is not a deprecation warning -- the parameter is gone from the
    signature, so passing it is a plain TypeError. It was always redundant
    anyway: from 1.5 the choice is inferred from the problem, and ``saga``/
    ``lbfgs`` both use the multinomial loss for three or more classes.
    ``Logistic_Regression_UO.py`` passed it unconditionally, so that script could
    not run at all on a modern scikit-learn -- ``TypeError: __init__() got an
    unexpected keyword argument 'multi_class'``.
    """
    return "multi_class" in inspect.signature(LogisticRegression.__init__).parameters


def logistic_regression_kwargs(C, penalty, solver, random_state=None, num_classes=2):
    """Constructor kwargs for a LogisticRegression, version-safe.

    ``multi_class`` is included only where the installed scikit-learn still
    takes it (see ``supports_multi_class_parameter``) and only where it is
    meaningful: on two classes the default "auto" already means "binary" and
    saying "multinomial" changed nothing but noise.

    ``random_state`` is returned unconditionally. It only seeds the solvers that
    use randomness, but leaving it unset means a result that cannot be
    reproduced -- and ``saga`` (the UO script's L1 solver) is randomised by
    default, so the two scripts' L1 models were silently unreproducible.
    """
    kwargs = {
        "C": C,
        "penalty": penalty,
        "solver": solver,
        "random_state": random_state,
    }
    if num_classes > 2 and supports_multi_class_parameter():
        kwargs["multi_class"] = "multinomial"
    return kwargs
