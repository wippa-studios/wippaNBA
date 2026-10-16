"""Shared machinery for the two neural network trainers.

`NN_Model_ML.py` (moneyline) and `NN_Model_UO.py` (totals) differ only in their
target column, their class count, the `_ML` / `_UO` infix in the model filename
and the scratch file the best-so-far model is parked in. Everything in this
module is common to both.

This is deliberately its own module rather than a fold of `boosting_training`:
the architecture, the normalisation and the split (a real validation slice, not
a calibration slice) have nothing in common with a gradient boosted tree, and
pretending otherwise would only hide the differences.
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from training_common import DATE_COLUMN


def prepare_data(df, target_column, drop_columns):
    data = df.copy()
    if DATE_COLUMN in data.columns:
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN], errors="coerce")
        data = data.sort_values(DATE_COLUMN)
    y = data[target_column].astype(int).to_numpy()
    # `OU` is kept in the totals trainers' drop_columns' complement, so it lands
    # in X at its dataset position; re-appending it here would push it to the
    # end and permute the inputs.
    X = data.drop(columns=drop_columns, errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    X = X.to_numpy(dtype=float)
    X = tf.keras.utils.normalize(X, axis=1)
    return X, y


def split_time_series(X, y, val_size=0.1, test_size=0.1):
    if not (0 < val_size < 1) or not (0 < test_size < 1):
        raise ValueError("val_size and test_size must be in (0, 1).")
    if val_size + test_size >= 1:
        raise ValueError("val_size + test_size must be < 1.")
    n = len(X)
    if n == 0:
        raise ValueError("Empty dataset.")
    val_start = int(n * (1 - val_size - test_size))
    test_start = int(n * (1 - test_size))
    X_train, y_train = X[:val_start], y[:val_start]
    X_val, y_val = X[val_start:test_start], y[val_start:test_start]
    X_test, y_test = X[test_start:], y[test_start:]
    return X_train, X_val, X_test, y_train, y_val, y_test


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
