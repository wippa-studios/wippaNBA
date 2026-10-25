"""Shared machinery for the two XGBoost trainers.

`XGBoost_Model_ML.py` (moneyline) and `XGBoost_Model_UO.py` (totals) differ only
in their target column, their class count and the ``_ML`` / ``_UO`` infix in the
model filename. Everything in this module is common to both.

`NUM_CLASSES` is a parameter here rather than a module constant precisely
because it is one of the things that differs between the two entry points.

Note the split helpers cut a numpy matrix, not a frame: unlike the logistic
regression trainers, the boosters are fed a positional float matrix whose column
order is written next to the saved model as a features JSON.

TWO DIMENSIONS ARE NOT SEARCHED ANY MORE
----------------------------------------
``num_boost_round`` was sampled from U{300, 2500} per trial. It was pure noise:
``train_model`` already passes ``early_stopping_rounds=60``, so the validation
fold decides when to stop and the sampled cap only ever mattered for the trials
that never early-stopped. Two trials that differed only in the cap were scored
as different configurations, which is the same defect as varying the seed.

``max_delta_step`` was sampled from 0-10 per trial. It exists to control the
weight update for extreme class imbalance, and there is no extreme class
imbalance here -- ``--class-weights`` defaults to ``none`` and the moneyline
label splits 57/43. It is therefore sampled only when class weighting is
actually switched on, where it does have a job.

Both are noise dimensions: at this sample size the selection effect of a search
is about 2.5 standard errors, and a dimension that carries no signal only makes
the winner's curse worse.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss

from training_common import (
    DATE_COLUMN,
    STABLE_SORT_KIND,
    recency_weighted_fold_mean,
    snap_cut_to_date_boundary,
    walk_forward_folds,
)

# The boosting budget is a CAP, not a search dimension. 2500 was the top of the
# range that used to be sampled; keeping the top means early stopping is what
# resolves the budget in every trial, instead of the sampled cap deciding it for
# the trials that happened not to stop early.
DEFAULT_NUM_BOOST_ROUND = 2500


def _snap(position, dates):
    """Move a split point to a date-group boundary, if the dates were supplied.

    Plain row counts put whole slates on both sides of a cut; mean 6.45 rows share
    one snapshot date here and the largest group is 40. See
    ``training_common.snap_cut_to_date_boundary``.
    """
    if dates is None:
        return int(position)
    return snap_cut_to_date_boundary(dates, position)


def prepare_data(df, target_column, drop_columns):
    """Sort by snapshot date and select the positional feature matrix.

    The sort is a stable merge sort, not pandas' default quicksort, which is not
    stable: rows sharing a snapshot date (a same-day slate, mean 6.45 rows, max
    40) came out in a different order under it, so a train/test boundary could
    land inside a group and the split could not be reproduced from the stored
    table alone.
    """
    data = df.copy()
    if DATE_COLUMN in data.columns:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        data = data.sort_values(DATE_COLUMN, kind=STABLE_SORT_KIND)
    y = data[target_column].astype(int).to_numpy()
    # This list IS the column order the model is fed, so X is selected from it
    # rather than recomputed: the two cannot drift apart, and the list is what
    # gets written next to the saved model for inference to check against.
    feature_names = [column for column in data.columns if column not in drop_columns]
    X = data[feature_names].astype(float).to_numpy()
    return X, y, feature_names, data


def split_train_test(X, y, test_size=0.1, dates=None):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    test_start = _snap(int(n * (1 - test_size)), dates)
    return X[:test_start], y[:test_start], X[test_start:], y[test_start:]


def split_train_calib(X, y, calib_size=0.1, dates=None):
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    calib_start = _snap(int(n * (1 - calib_size)), dates)
    return X[:calib_start], y[:calib_start], X[calib_start:], y[calib_start:]


def split_train_es_calib(X, y, es_size=0.05, calib_size=0.1, dates=None):
    """Four disjoint chronological segments: train | early-stop | calibration.

    Why the early-stopping segment must not be the calibration segment
    ------------------------------------------------------------------
    XGBoost early-stops on the LAST entry of ``evals``. The trainers previously
    passed the calibration rows as ``X_val``, so ``dval`` *was* ``dcalib`` and the
    boosting round was chosen to minimise loss on exactly the rows the sigmoid was
    then fitted to. The calibrator therefore saw the best-scoring slice of the model
    rather than a draw from the distribution it is applied to, and any slope it
    learned was fitted to in-sample-optimal scores.

    Ordering is train -> early-stop -> calibration, each contiguous in time, with
    every boundary snapped to a date edge. Calibration comes last so the calibrator
    still sees the most recent data available to it.

    The cost is real and was measured before paying it: refitting the logistic
    reference on 90% of train_val instead of 100% moved test log loss by +0.0007
    nats for 1,395 fewer training rows. That is what a dedicated early-stop segment
    costs, and it is smaller than the distortion it removes.

    Returns (X_train, y_train, X_es, y_es, X_calib, y_calib).
    """
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    if not 0 < es_size < 1 or not 0 < calib_size < 1:
        raise ValueError("es_size and calib_size must each be a fraction of (0, 1).")
    if es_size + calib_size >= 1:
        raise ValueError(
            f"es_size ({es_size}) + calib_size ({calib_size}) leaves no training data."
        )
    calib_start = _snap(int(n * (1 - calib_size)), dates)
    es_start = _snap(int(calib_start * (1 - es_size)), dates)
    if es_start <= 0:
        raise ValueError(
            f"Not enough rows for the requested segments: n={n}, es_size={es_size}, "
            f"calib_size={calib_size}."
        )
    return (
        X[:es_start], y[:es_start],
        X[es_start:calib_start], y[es_start:calib_start],
        X[calib_start:], y[calib_start:],
    )


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


def sample_params(rng, seed, num_classes, class_weights="none"):
    """One random configuration.

    ``seed`` is the RUN's seed and must be the same for every trial. Passing
    ``seed + trial`` -- which is what both entry points used to do -- made two
    trials that differed only in seed look like two different configurations to
    the search, so the search selected on seed luck. The seed is a fixed property
    of the run, not a hyperparameter.

    ``max_delta_step`` is included only when class weighting is on; see the
    module docstring.

    Two classes use ``binary:logistic``, not ``multi:softprob`` with
    ``num_class=2``. Softmax over two logits is algebraically the sigmoid of
    their difference, so the two objectives span the same function class, but
    softmax fits one tree per class per round (two trees) while the sigmoid fits
    one. ``logloss`` on the single probability equals ``mlogloss`` on the
    two-column matrix exactly, so the selection criterion is unchanged. The
    eval-metric list keeps the same order as the multiclass pair so early
    stopping still reads the same quantity: xgboost stops on the LAST entry.
    """
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
        "max_bin": int(rng.integers(128, 1025)),
        "lambda": float(10 ** rng.uniform(np.log10(0.1), np.log10(10.0))),
        "alpha": float(10 ** rng.uniform(np.log10(0.01), np.log10(5.0))),
        "seed": seed,
        "tree_method": "hist",
    }
    if num_classes == 2:
        params["objective"] = "binary:logistic"
        params["eval_metric"] = ["logloss", "error"]
    else:
        params["objective"] = "multi:softprob"
        params["num_class"] = num_classes
        params["eval_metric"] = ["mlogloss", "merror"]
    if class_weights != "none":
        params["max_delta_step"] = int(rng.integers(0, 11))
    else:
        params["max_delta_step"] = 0
    return params, DEFAULT_NUM_BOOST_ROUND


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


def predict_probabilities(booster, X, num_classes):
    """Predict a probability matrix, whatever shape the objective returns.

    ``binary:logistic`` returns one column (P(class 1)); every other objective
    used here returns one column per class. The rest of the pipeline indexes
    ``[:, 1]`` and hands the matrix to ``log_loss``/calibrators, so the binary
    case is widened to two columns with ``1 - p`` in the first.
    """
    raw = booster.predict(xgb.DMatrix(X))
    if num_classes == 2:
        if raw.ndim == 1:
            raw = np.column_stack((1.0 - raw, raw))
        elif raw.shape[1] == 1:
            raw = np.column_stack((1.0 - raw[:, 0], raw[:, 0]))
    return raw


class BoosterWrapper:
    def __init__(self, booster, num_class):
        self.booster = booster
        self.classes_ = np.arange(num_class)
        self.num_classes_ = num_class

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return predict_probabilities(self.booster, X, self.num_classes_)


def walk_forward_cv_loss(
    X,
    y,
    params,
    num_boost_round,
    n_splits,
    num_classes,
    class_weights="none",
    fold_rows=None,
    half_life_folds=1.0,
    dates=None,
):
    """Recency-weighted walk-forward log loss over the training block.

    Returns ``(weighted_mean, per_fold_losses)``. Both are wanted: the weighted
    mean is what the search selects on, and the per-fold list is what gets
    printed beside it. Per-fold log loss standard deviation across the old
    five-fold geometry was already 0.0044 at a single fixed configuration, before
    any hyperparameter search was involved, so a single number without its folds
    is not interpretable.

    Folds come from ``training_common.walk_forward_folds``, whose windows are
    capped at one season by default and are configurable. The returned mean
    weights recent folds above old ones; see
    ``training_common.recency_weighted_fold_mean`` for why. Passing ``dates``
    snaps every fold boundary to a date-group boundary.
    """
    folds = walk_forward_folds(
        len(X), n_splits, fold_rows=fold_rows, dates=dates
    )
    losses = []
    for train_idx, val_idx in folds:
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
        val_probs = predict_probabilities(model, X[val_idx], num_classes)
        losses.append(
            log_loss(y[val_idx], val_probs, labels=list(range(num_classes)))
        )
    if not losses:
        return None, []
    weighted, per_fold, _weights = recency_weighted_fold_mean(
        losses, half_life_folds=half_life_folds
    )
    return weighted, per_fold
