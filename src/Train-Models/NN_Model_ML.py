import argparse

import numpy as np
import tensorflow as tf

import os
import sys

# This directory is named with a hyphen, so it is not an importable package.
# The shared modules below are siblings, which means they only resolve when
# this script runs with the repository root on sys.path -- e.g.
# `python -m src.Train-Models.NN_Model_ML`. Adding the script's own directory
# keeps `python src/Train-Models/NN_Model_ML.py` working too.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imported as a MODULE as well as by name: `training_common.DATE_COLUMN` is read at
# run time to snap the split points to date-group boundaries, and mixing the two
# spellings in one file is how the two get out of step.
import training_common

from neural_training import (
    build_model,
    prepare_data,
    sample_params,
    split_time_series,
    to_model_matrix,
)
from training_common import (
    BASE_DIR,
    DEFAULT_DATASET,
    DEFAULT_TRIALS,
    MONEYLINE_NUM_CLASSES,
    MONEYLINE_TARGET,
    drop_columns_for,
    load_dataset,
    neural_model_filename,
    report_search_selection_risk,
    write_run_manifest,
)


MODEL_DIR = BASE_DIR / "Models"

NUM_CLASSES = MONEYLINE_NUM_CLASSES
TARGET_COLUMN = MONEYLINE_TARGET
# OU is dropped here on purpose: it is the label-adjacent closing line, and this
# moneyline trainer does not want it. See training_common.drop_columns_for.
DROP_COLUMNS = drop_columns_for(TARGET_COLUMN)


def add_search_arguments(parser):
    """The shared flags, minus the fold geometry.

    These networks validate on ONE contiguous slice (see
    ``neural_training.split_time_series``), not on a walk-forward fold series, so
    ``--fold-rows`` and ``--fold-half-life`` do not apply here. ``--trials`` and
    ``--dataset`` are the same shared values as everywhere else.

    These models have no calibrator, so ``--calibration`` is deliberately absent:
    there is nothing to choose.
    """
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET, help="Dataset table name."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help=(
            "Number of parameter trials. Default is 15, not 50: measured on this "
            "dataset, CV noise exceeds the true spread of test performance by 8x, "
            "so a larger search only buys selection bias."
        ),
    )
    return parser


def build_parser():
    """The fully-populated parser, so its defaults can be inspected."""
    parser = argparse.ArgumentParser(description="Train NN moneyline model (random search).")
    add_search_arguments(parser)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation size.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test size.")
    return parser


def main():
    args = build_parser().parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.optimizer.set_experimental_options({"remapping": False})
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    if df.empty:
        print(f"No rows found for dataset {args.dataset}.")
        return

    X, y, data = prepare_data(df, TARGET_COLUMN, DROP_COLUMNS)
    # `dates` makes both cuts land on a date-group boundary: the dataset holds
    # whole slates (mean 6.45 rows per snapshot date, max 40), and a cut through
    # one of them puts part of a single night's games on each side.
    X_train, X_val, X_test, y_train, y_val, y_test = split_time_series(
        X,
        y,
        val_size=args.val_size,
        test_size=args.test_size,
        dates=data[training_common.DATE_COLUMN],
    )
    # Imputed with the TRAIN medians only, then normalised. This used to happen
    # inside prepare_data, on the joined frame, which let the validation and test
    # rows reach training through the median.
    X_train, X_val, X_test = to_model_matrix(X_train, X_val, X_test)
    rng = np.random.default_rng(args.seed)
    best = {"val_loss": float("inf"), "params": None}
    temp_path = MODEL_DIR / "best_ml_temp.keras"

    for trial in range(1, args.trials + 1):
        tf.keras.backend.clear_session()
        params = sample_params(rng)
        model = build_model(X_train.shape[1], params, NUM_CLASSES)
        model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=params["lr"]),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6),
        ]
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            callbacks=callbacks,
            verbose=0,
        )
        val_loss = float(np.min(history.history["val_loss"]))
        if not np.isfinite(val_loss):
            print(f"Trial {trial}/{args.trials}: val loss invalid, skipping")
            continue
        print(f"Trial {trial}/{args.trials}: val loss {val_loss:.4f}")

        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["params"] = params
            model.save(temp_path)

    if best["params"] is None or not temp_path.exists():
        print("No valid model trained.")
        return

    best_model = tf.keras.models.load_model(temp_path, compile=False)
    best_model.compile(
        optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=best["params"]["lr"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    loss, accuracy = best_model.evaluate(X_test, y_test, verbose=0)
    print(f"Best val loss: {best['val_loss']:.4f}")
    print(f"Configurations tried: {args.trials} on one validation fold of {len(X_val)} rows.")
    # These networks report categorical crossentropy, which is log loss up to a
    # constant, so the standard error below is the per-row standard deviation of
    # that same quantity over the held-out rows.
    report_search_selection_risk(
        n_configs_tried=args.trials,
        metric_name="held-out categorical crossentropy (nats)",
        n_test_rows=len(y_test),
        standard_error=_crossentropy_standard_error(best_model, X_test, y_test),
    )
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test loss: {loss:.4f}")

    params = best["params"]
    # The filename carries the target and the held-out accuracy, nothing else.
    # The winning configuration goes into the manifest below, which is
    # gitignored. The runner's prefix glob still finds this artifact.
    model_name = neural_model_filename("ML", accuracy)
    model_path = MODEL_DIR / model_name
    best_model.save(model_path)
    temp_path.unlink(missing_ok=True)
    print(f"Saved model: {model_path}")

    manifest_path = write_run_manifest(model_path, {
        "schema": 1,
        "artifact": model_path.name,
        "kind": "moneyline",
        "target": TARGET_COLUMN,
        "accuracy": float(accuracy),
        "test_loss": float(loss),
        "best_val_loss": float(best["val_loss"]),
        "n_features": int(X_train.shape[1]),
        "params": params,
        "dataset": args.dataset,
        "seed": args.seed,
        "trials": args.trials,
        "val_size": args.val_size,
        "test_size": args.test_size,
    })
    print(f"Saved run manifest: {manifest_path}")


def _crossentropy_standard_error(model, X_test, y_test):
    """Per-row standard deviation of the network's loss, divided by sqrt(n).

    The networks score with ``sparse_categorical_crossentropy``, and the
    selection-risk correction needs a standard error in the same units as the
    metric it corrects. Taking it straight from the per-row losses keeps the two
    from drifting apart.
    """
    probabilities = np.asarray(model.predict(X_test, verbose=0))
    probabilities = np.clip(probabilities, np.finfo(np.float64).eps, 1.0)
    rows = np.arange(probabilities.shape[0])
    losses = -np.log(probabilities[rows, np.asarray(y_test, dtype=int)])
    if losses.size < 2:
        return float("nan")
    return float(np.std(losses, ddof=1) / np.sqrt(losses.size))


if __name__ == "__main__":
    main()