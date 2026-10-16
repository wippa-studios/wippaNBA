"""Shared machinery for the two XGBoost trainers.

`XGBoost_Model_ML.py` (moneyline) and `XGBoost_Model_UO.py` (totals) differ only
in their target column, their class count and the ``_ML`` / ``_UO`` infix in the
model filename. Everything in this module is common to both.

`NUM_CLASSES` is a parameter here rather than a module constant precisely
because it is one of the things that differs between the two entry points.

Note the split helpers cut a numpy matrix, not a frame: unlike the logistic
regression trainers, the boosters are fed a positional float matrix whose column
order is written next to the saved model as a features JSON.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit

from training_common import DATE_COLUMN


def prepare_data(df, target_column, drop_columns):
    data = df.copy()
    if DATE_COLUMN in data.columns:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        data = data.sort_values(DATE_COLUMN)
    y = data[target_column].astype(int).to_numpy()
    # This list IS the column order the model is fed, so X is selected from it
    # rather than recomputed: the two cannot drift apart, and the list is what
    # gets written next to the saved model for inference to check against.
    feature_names = [column for column in data.columns if column not in drop_columns]
    X = data[feature_names].astype(float).to_numpy()
    return X, y, feature_names, data


def split_train_test(X, y, test_size=0.1):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    test_start = int(n * (1 - test_size))
    return X[:test_start], y[:test_start], X[test_start:], y[test_start:]


def split_train_calib(X, y, calib_size=0.1):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    calib_start = int(n * (1 - calib_size))
    return X[:calib_start], y[:calib_start], X[calib_start:], y[calib_start:]


def compute_sample_weights(y, num_classes, mode):
    """Training-time class weights, or None when the model is left unweighted.

    "balanced" is scikit-learn's definition, total / (num_classes * count), which
    is what this script has always applied. "inverse" is the raw 1 / count. The
    two differ only by the constant total / num_classes, so both rebalance the
    classes; they exist as separate choices so the shipped behaviour can be
    selected by name instead of by remembering which formula it was.
    """
    if mode == "none":
        return None
    counts = np.bincount(y, minlength=num_classes)
    total = len(y)
    class_weights = {}
    for cls, count in enumerate(counts):
        if not count:
            class_weights[cls] = 1.0
        elif mode == "inverse":
            class_weights[cls] = 1.0 / count
        else:
            class_weights[cls] = total / (num_classes * count)
    return np.array([class_weights[label] for label in y])


def sample_params(rng, seed, num_classes):
    eta = 10 ** rng.uniform(np.log10(0.003), np.log10(0.3))
    params = {
        "max_depth": int(rng.integers(2, 13)),
        "eta": float(eta),
        "subsample": float(rng.uniform(0.5, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "colsample_bylevel": float(rng.uniform(0.5, 1.0)),
        "colsample_bynode": float(rng.uniform(0.5, 1.0)),
        "min_child_weight": int(rng.integers(1, 21)),
        "gamma": float(rng.uniform(0.0, 10.0)),
        "max_delta_step": int(rng.integers(0, 11)),
        "max_bin": int(rng.integers(128, 1025)),
        "lambda": float(10 ** rng.uniform(np.log10(0.1), np.log10(10.0))),
        "alpha": float(10 ** rng.uniform(np.log10(0.01), np.log10(5.0))),
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": ["mlogloss", "merror"],
        "seed": seed,
        "tree_method": "hist",
    }
    num_boost_round = int(rng.integers(300, 2501))
    return params, num_boost_round


def train_model(
    X_train, y_train, X_val, y_val, params, num_boost_round, num_classes, class_weights="none"
):
    weights = compute_sample_weights(y_train, num_classes=num_classes, mode=class_weights)
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights)
    dval = xgb.DMatrix(X_val, label=y_val)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=60,
        verbose_eval=False,
    )
    return model


class BoosterWrapper:
    def __init__(self, booster, num_class):
        self.booster = booster
        self.classes_ = np.arange(num_class)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return self.booster.predict(xgb.DMatrix(X))


def walk_forward_cv_loss(
    X, y, params, num_boost_round, n_splits, num_classes, class_weights="none"
):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    losses = []
    for train_idx, val_idx in tscv.split(X):
        model = train_model(
            X[train_idx],
            y[train_idx],
            X[val_idx],
            y[val_idx],
            params,
            num_boost_round,
            num_classes,
            class_weights,
        )
        val_probs = model.predict(xgb.DMatrix(X[val_idx]))
        loss = log_loss(y[val_idx], val_probs, labels=list(range(num_classes)))
        losses.append(loss)
    return float(np.mean(losses)) if losses else None
