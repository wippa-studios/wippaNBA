import argparse

import joblib
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

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
    KerasProbabilityWrapper,
    build_model,
    prepare_data,
    sample_params,
    split_train_es_calib_test,
    to_model_matrix,
)
from training_common import (
    BASE_DIR,
    DEFAULT_DATASET,
    DEFAULT_TRIALS,
    MONEYLINE_NUM_CLASSES,
    MONEYLINE_TARGET,
    calibration_choices,
    drop_columns_for,
    load_dataset,
    neural_model_filename,
    report_search_selection_risk,
    write_run_manifest,
)
from training_reporting import (
    MIN_CALIBRATION_ROWS,
    log_loss_standard_error,
    make_calibrated_estimator,
    report_calibration_effect,
    report_closing_line,
    resolve_calibration_method,
)


# The networks live beside their own artifacts, not directly in Models/. The
# runner used to glob Models/*.keras and Models/NN_Models/*.keras together, so a
# fresh run wrote an artifact into a directory it did not read from and left the
# two indistinguishable (issue #16). One directory, the one the runner prefers.
MODEL_DIR = BASE_DIR / "Models" / "NN_Models"

NUM_CLASSES = MONEYLINE_NUM_CLASSES
TARGET_COLUMN = MONEYLINE_TARGET
LABELS = [0, 1]
# OU is dropped here on purpose: it is the label-adjacent closing line, and this
# moneyline trainer does not want it. See training_common.drop_columns_for.
DROP_COLUMNS = drop_columns_for(TARGET_COLUMN)


def add_search_arguments(parser):
    """The shared flags, minus the fold geometry.

    These networks validate on ONE contiguous slice (see
    ``neural_training.split_train_es_calib_test``), not on a walk-forward fold
    series, so ``--fold-rows`` and ``--fold-half-life`` do not apply here.
    ``--trials`` and ``--dataset`` are the same shared values as everywhere else.

    Calibration is the same choice as the scikit-learn trainers: sigmoid by
    default, isotonic only on request and only with enough rows. The network
    gets its own calibration segment, separate from the validation slice that
    chooses the epoch (issue #16).
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
    parser.add_argument(
        "--calibration",
        default="sigmoid",
        choices=list(calibration_choices()),
        help="Calibration method for probabilities.",
    )
    parser.add_argument(
        "--allow-isotonic",
        action="store_true",
        help=(
            "Permit --calibration isotonic. Off by default: isotonic measured "
            "+0.0035 nats here, worse than no calibration, and needs at least "
            f"{MIN_CALIBRATION_ROWS} calibration rows."
        ),
    )
    return parser


def build_parser():
    """The fully-populated parser, so its defaults can be inspected."""
    parser = argparse.ArgumentParser(description="Train NN moneyline model (random search).")
    add_search_arguments(parser)
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation (early-stop) size.")
    parser.add_argument("--calib-size", type=float, default=0.1, help="Calibration size.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test size.")
    return parser


def main():
    args = build_parser().parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    # TensorFlow is pinned to 2.14.0 in requirements.txt, and this call plus
    # ``optimizers.legacy`` below are why that pin is load-bearing: Keras 3
    # (TF 2.16+) removed both. Migrating means re-baselining every number this
    # project reports, not just renaming an optimizer, so it is deliberately a
    # separate decision (issue #16 allows the documented pin).
    tf.config.optimizer.set_experimental_options({"remapping": False})
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    if df.empty:
        print(f"No rows found for dataset {args.dataset}.")
        return

    X, y, data = prepare_data(df, TARGET_COLUMN, DROP_COLUMNS)
    # `dates` makes every cut land on a date-group boundary: the dataset holds
    # whole slates (mean 6.45 rows per snapshot date, max 40), and a cut through
    # one of them puts part of a single night's games on each side.
    X_train, y_train, X_es, y_es, X_calib, y_calib, X_test, y_test = \
        split_train_es_calib_test(
            X,
            y,
            val_size=args.val_size,
            calib_size=args.calib_size,
            test_size=args.test_size,
            dates=data[training_common.DATE_COLUMN],
        )
    # Fit imputation + StandardScaler on TRAIN rows only, then apply that exact
    # transform to the held-out segments. This replaced per-row L2
    # normalisation; see neural_training's module docstring.
    preprocessor, (X_train, X_es, X_calib, X_test) = to_model_matrix(
        X_train, X_es, X_calib, X_test
    )
    rng = np.random.default_rng(args.seed)
    best = {"val_loss": float("inf"), "params": None}
    temp_path = MODEL_DIR / "best_ml_temp.keras"
    # A killed process cannot run a finally block, so a scratch file from an
    # earlier aborted run is removed before this one can be mistaken for it.
    temp_path.unlink(missing_ok=True)

    try:
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
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=10, restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
                ),
            ]
            model.fit(
                X_train,
                y_train,
                validation_data=(X_es, y_es),
                epochs=params["epochs"],
                batch_size=params["batch_size"],
                callbacks=callbacks,
                verbose=0,
            )
            # The loss of the RESTORED weights, not min(history["val_loss"]).
            # EarlyStopping already restores the best epoch, so this is the
            # number the saved model actually has; reporting the history's
            # minimum was a second selection presented as a measurement.
            val_loss, _ = model.evaluate(X_es, y_es, verbose=0)
            val_loss = float(val_loss)
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

        # Load WITHOUT a compile config: this model is only used for predict()
        # and is embedded in the saved calibrator. A Keras pickle re-runs
        # compile_from_config on load, and TF 2.14's legacy Adam cannot be
        # rebuilt that way (`AttributeError: 'Adam' object has no attribute
        # 'build'`), so an uncompiled model is the artifact that round-trips.
        best_model = tf.keras.models.load_model(temp_path, compile=False)

        raw_probabilities = np.asarray(best_model.predict(X_test, verbose=0))
        calibration_method = resolve_calibration_method(
            args.calibration, len(X_calib), allow_isotonic=args.allow_isotonic
        )
        calibrator = None
        if calibration_method == "none":
            probabilities = raw_probabilities
        else:
            calibrator = make_calibrated_estimator(
                KerasProbabilityWrapper(best_model, NUM_CLASSES), calibration_method
            )
            calibrator.fit(X_calib, y_calib)
            probabilities = np.asarray(calibrator.predict_proba(X_test))

        y_pred = np.argmax(probabilities, axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        test_loss = log_loss(y_test, probabilities, labels=list(range(NUM_CLASSES)))

        print(f"Best val loss: {best['val_loss']:.4f}")
        print(
            f"Configurations tried: {args.trials} on one validation fold of "
            f"{len(X_es)} rows, with a separate {len(X_calib)}-row calibration "
            f"segment."
        )
        # The networks report categorical crossentropy, which is log loss up to
        # a constant, so the standard error below is in the same units as the
        # metric it corrects.
        report_search_selection_risk(
            n_configs_tried=args.trials,
            metric_name="held-out log loss",
            n_test_rows=len(y_test),
            standard_error=log_loss_standard_error(
                y_test, probabilities, labels=LABELS
            ),
        )
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Test log loss: {test_loss:.4f}")
        print(f"Test ROC AUC: {roc_auc_score(y_test, probabilities[:, 1]):.4f}")
        report_calibration_effect(
            y_test,
            raw_probabilities,
            probabilities,
            labels=LABELS,
            n_calib_rows=len(X_calib),
            method=calibration_method,
        )

        # The test rows are the tail of the sorted frame, taken by the length of
        # the test matrix rather than by recomputing the split point, so the
        # rows cannot drift out of step with y_test.
        test_rows = data.tail(len(X_test))
        report_closing_line(
            test_rows,
            home_win_truth=test_rows["Home-Team-Win"].astype(int).to_numpy(),
            moneyline_picks=y_pred,
        )

        params = best["params"]
        # The filename carries the target and the held-out accuracy, nothing
        # else. The winning configuration goes into the manifest below, which is
        # gitignored. The runner's prefix glob still finds this artifact.
        model_name = neural_model_filename("ML", accuracy)
        model_path = MODEL_DIR / model_name
        best_model.save(model_path)
        print(f"Saved model: {model_path}")

        # Inference must apply the SAME imputation and scaler as training. It
        # cannot refit them on a one-row slate, and falling back to per-row
        # normalisation would silently feed the network a different function.
        preprocessor_path = model_path.with_name(f"{model_path.stem}_preprocessor.pkl")
        joblib.dump(preprocessor, preprocessor_path)
        print(f"Saved preprocessor: {preprocessor_path}")

        calibration_path = None
        if calibrator is not None:
            calibration_path = model_path.with_name(f"{model_path.stem}_calibration.pkl")
            joblib.dump(calibrator, calibration_path)
            print(f"Saved calibration: {calibration_path}")

        manifest_path = write_run_manifest(model_path, {
            "schema": 1,
            "artifact": model_path.name,
            "kind": "moneyline",
            "target": TARGET_COLUMN,
            "accuracy": float(accuracy),
            "test_loss": float(test_loss),
            "best_val_loss": float(best["val_loss"]),
            "n_features": int(X_train.shape[1]),
            "params": params,
            "dataset": args.dataset,
            "seed": args.seed,
            "trials": args.trials,
            "val_size": args.val_size,
            "calib_size": args.calib_size,
            "calibration_rows": int(len(X_calib)),
            "test_size": args.test_size,
            "calibration": calibration_method,
            "preprocessor": preprocessor_path.name,
            "calibrator": calibration_path.name if calibration_path else None,
        })
        print(f"Saved run manifest: {manifest_path}")
    finally:
        # Covers every exit: no valid trial, an exception mid-save, and success.
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
