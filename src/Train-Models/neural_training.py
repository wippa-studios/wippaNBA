"""Shared machinery for the two neural network trainers.

`NN_Model_ML.py` (moneyline) and `NN_Model_UO.py` (totals) differ only in their
target column, their class count, the `_ML` / `_UO` infix in the model filename
and the scratch file the best-so-far model is parked in. Everything in this
module is common to both.

This is deliberately its own module rather than a fold of `boosting_training`:
the architecture, the normalisation and the split (a real validation slice, not
a calibration slice) have nothing in common with a gradient boosted tree, and
pretending otherwise would only hide the differences.

``prepare_data`` no longer imputes. It used to fill NaN with the median of the
WHOLE frame before the split, which lets held-out statistics reach training, and
it also collapsed the frame to a normalised numpy matrix in one step -- so the
imputation could not be moved after the split without being unpacked first.
``prepare_data`` now returns the feature FRAME; ``to_model_matrix`` does the
imputation, the standardisation and the cast, and the two entry points call it
once per split segment.

NORMALISATION (issue #16)
========================
This module used to call Keras's per-row L2 normalise, which divides
each row by its own L2 norm. That is not "scale the features"; it is a
per-row gain. Raw row norms on the shipped dataset span 249.7-333.6 (CV 5%),
and after normalisation the first layer sees ``(w·x_i)/||x_i||``, so two games
with the same feature direction differ by up to 32% purely from magnitude --
a quantity correlated with the outcome at r = -0.023, i.e. noise. The fix is a
``StandardScaler`` fit on TRAIN rows only (``training_common.fit_feature_scaler``),
saved beside the model and applied unchanged at inference.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from training_common import (
    DATE_COLUMN,
    STABLE_SORT_KIND,
    fit_feature_scaler,
    snap_cut_to_date_boundary,
)


def prepare_data(df, target_column, drop_columns):
    """Sort by snapshot date and select the feature frame. Does NOT impute.

    The sort is a stable merge sort. pandas' default quicksort is not stable, and
    rows sharing a snapshot date (a same-day slate; mean 6.45 rows, max 40) came
    out in a different order under it, so a split boundary could land inside a
    group and the split could not be reproduced from the stored table alone.

    Returns ``(X, y, data)``. ``data`` is the full sorted frame; the caller
    passes ``data["Date"]`` to ``split_time_series`` so both cuts land on
    date-group boundaries.

    ``OU`` is kept in the totals trainers' drop_columns' complement, so it lands
    in X at its dataset position; re-appending it here would push it to the
    end and permute the inputs.
    """
    data = df.copy()
    if DATE_COLUMN in data.columns:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        data = data.sort_values(DATE_COLUMN, kind=STABLE_SORT_KIND)
    y = data[target_column].astype(int).to_numpy()
    X = data.drop(columns=drop_columns, errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan)
    return X, y, data


def to_model_matrix(X_train, *others):
    """Impute and standardise every frame with TRAIN statistics only.

    ``X_train`` is first; every other frame is held out. The fill values and the
    ``StandardScaler`` mean/scale are computed once, on ``X_train`` alone, and
    reused for the rest, so no held-out statistic reaches training. Call this
    once per split segment, never on the joined frame.

    Returns ``(preprocessor, matrices)``: the fitted
    :class:`training_common.FeatureScaler` (save this beside the model; the
    runner loads it instead of normalising) and one standardised matrix per
    argument, in the order given.
    """
    return fit_feature_scaler(X_train, *others)


def split_train_es_calib_test(
    X, y, val_size=0.1, calib_size=0.1, test_size=0.1, dates=None
):
    """Four disjoint chronological segments: train | early-stop | calib | test.

    The validation slice chooses the epoch (and the trial), and the calibration
    slice fits the probability calibrator. They must not be the same rows: a
    calibrator fitted on the rows that chose the best epoch is fitted to the
    model's best-scoring slice, not to the distribution it is applied to. The
    scikit-learn trainers already made this split for the same reason
    (``boosting_training.split_train_es_calib``); the neural path previously had
    no calibration segment at all.

    Cuts are snapped forward to a date boundary when ``dates`` is given, so a
    single night's slate never straddles two segments. Returns features and
    labels interleaved::

        X_train, y_train, X_es, y_es, X_calib, y_calib, X_test, y_test
    """
    for name, value in (
        ("val_size", val_size),
        ("calib_size", calib_size),
        ("test_size", test_size),
    ):
        if not (0 < value < 1):
            raise ValueError(f"{name} must be in (0, 1).")
    if val_size + calib_size + test_size >= 1:
        raise ValueError("val_size + calib_size + test_size must be < 1.")

    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    test_start = int(n * (1 - test_size))
    calib_start = int(n * (1 - test_size - calib_size))
    val_start = int(n * (1 - test_size - calib_size - val_size))
    if dates is not None:
        val_start = snap_cut_to_date_boundary(dates, val_start)
        calib_start = snap_cut_to_date_boundary(dates, calib_start)
        test_start = snap_cut_to_date_boundary(dates, test_start)
    if not (0 < val_start < calib_start < test_start < n):
        raise ValueError(
            "the requested split leaves an empty segment after snapping to date "
            f"boundaries: {val_start}, {calib_start}, {test_start}, n={n}"
        )
    return (
        X[:val_start], y[:val_start],
        X[val_start:calib_start], y[val_start:calib_start],
        X[calib_start:test_start], y[calib_start:test_start],
        X[test_start:], y[test_start:],
    )


def split_time_series(X, y, val_size=0.1, test_size=0.1, dates=None):
    """Legacy three-way split, kept for callers outside this repository.

    The trainers use :func:`split_train_es_calib_test`; a model with no separate
    calibration segment cannot be calibrated honestly.
    """
    if not (0 < val_size < 1) or not (0 < test_size < 1):
        raise ValueError("val_size and test_size must be in (0, 1).")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1.")
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    val_start = int(n * (1 - val_size - test_size))
    test_start = int(n * (1 - test_size))
    if dates is not None:
        val_start = snap_cut_to_date_boundary(dates, val_start)
        test_start = snap_cut_to_date_boundary(dates, test_start)
    X_train, y_train = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:test_start], y[val_start:test_start]
    X_test, y_test = X[test_start:], y[test_start:]
    return X_train, X_val, X_test, y_train, y_val, y_test


class KerasProbabilityWrapper:
    """``predict_proba`` adapter so a fitted Keras model can be calibrated.

    ``CalibratedClassifierCV`` requires an estimator implementing ``fit`` and
    ``predict_proba``; a Keras model implements ``predict``. ``fit`` is a no-op
    because the network is already fitted -- the same contract as
    ``boosting_training.BoosterWrapper``. ``classes_`` is what sklearn reads to
    validate the label range and to order the output columns.
    """

    def __init__(self, model, num_classes):
        self.model = model
        self.num_classes_ = int(num_classes)
        self.classes_ = np.arange(self.num_classes_)

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return np.asarray(self.model.predict(X, verbose=0))



def sample_params(rng):
    depth = int(rng.integers(2, 7))
    width = int(rng.choice([64, 96, 128, 160, 192, 256]))
    units = [max(32, int(width / (2 ** i))) for i in range(depth)]
    return {
        "units": units,
        "dropout": float(rng.uniform(0.0, 0.4)),
        "batch_norm": bool(rng.integers(0, 2)),
        "l2": float(10 ** rng.uniform(np.log10(1e-6), np.log10(1e-4))),
        "lr": float(10 ** rng.uniform(np.log10(5e-5), np.log10(5e-3))),
        "batch_size": int(rng.choice([64, 128, 256])),
        "epochs": int(rng.integers(40, 121)),
    }


def build_model(input_dim, params, num_classes):
    reg = tf.keras.regularizers.l2(params["l2"]) if params["l2"] > 0 else None
    inputs = tf.keras.Input(shape=(input_dim,))
    x = inputs
    for units in params["units"]:
        x = tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=reg)(x)
        if params["batch_norm"]:
            x = tf.keras.layers.BatchNormalization()(x)
        if params["dropout"] > 0:
            x = tf.keras.layers.Dropout(params["dropout"])(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)
