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
"""

import numpy as np
import pandas as pd

from training_common import DATE_COLUMN


def prepare_data(df, target_column, drop_columns):
    data = df.copy()
    if DATE_COLUMN in data.columns:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        data = data.sort_values(DATE_COLUMN)
    y = data[target_column].astype(int).to_numpy()
    X = data.drop(columns=drop_columns, errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X, y, data


def split_time_series(X, y, val_size=0.1, test_size=0.1):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    val_start = int(n * (1 - val_size - test_size))
    test_start = int(n * (1 - test_size))
    X_train, y_train = X.iloc[:val_start], y[:val_start]
    X_test, y_test = X.iloc[test_start:], y[test_start:]
    return X_train, X_test, y_train, y_test


def split_train_test(X, y, test_size=0.1):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    test_start = int(n * (1 - test_size))
    return X.iloc[:test_start], y[:test_start], X.iloc[test_start:], y[test_start:]


def split_train_calib(X, y, calib_size=0.1):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    calib_start = int(n * (1 - calib_size))
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
