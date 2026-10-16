import argparse

import numpy as np
import tensorflow as tf

import os
import sys

# This directory is named with a hyphen, so it is not an importable package.
# The shared modules below are siblings, which means they only resolve when
# this script runs with the repository root on sys.path -- e.g.
# `python -m src.Train-Models.Logistic_Regression_ML`. Adding the script's own
# directory keeps `python src/Train-Models/Logistic_Regression_ML.py` working too.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from neural_training import build_model, prepare_data, sample_params, split_time_series
from training_common import (
    BASE_DIR,
    TOTALS_NUM_CLASSES,
    TOTALS_TARGET,
    drop_columns_for,
    format_param,
    load_dataset,
)


MODEL_DIR = BASE_DIR / "Models"

DEFAULT_DATASET = "dataset_2012-24_new"
NUM_CLASSES = TOTALS_NUM_CLASSES
TARGET_COLUMN = TOTALS_TARGET
# OU stays out of this list on purpose: the pre-game total is known before tip-off,
# and this totals trainer must agree with XGBoost_Model_UO.DROP_COLUMNS, which keeps
# it too, or the runner feeds the networks an input width and order neither trained on.
# See training_common.drop_columns_for.
DROP_COLUMNS = drop_columns_for(TARGET_COLUMN)


def main():
    parser = argparse.ArgumentParser(description="Train NN totals model (random search).")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset table name.")
    parser.add_argument("--trials", type=int, default=50, help="Number of parameter trials.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation size.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Test size.")
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    tf.config.optimizer.set_experimental_options({"remapping": False})
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    if df.empty:
        print(f"No rows found for dataset {args.dataset}.")
        return

    X, y = prepare_data(df, TARGET_COLUMN, DROP_COLUMNS)
    X_train, X_val, X_test, y_train, y_val, y_test = split_time_series(
        X, y, val_size=args.val_size, test_size=args.test_size
    )
    rng = np.random.default_rng(args.seed)
    best = {"val_loss": float("inf"), "params": None}
    temp_path = MODEL_DIR / "best_uo_temp.keras"

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
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test loss: {loss:.4f}")

    params = best["params"]
    model_name = (
        f"Trained-Model-OU-{accuracy * 100:.1f}"
        f"_d{len(params['units'])}"
        f"_u{params['units'][0]}"
        f"_dr{format_param(params['dropout'])}"
        f"_bn{int(params['batch_norm'])}"
        f"_l2{format_param(params['l2'])}"
        f"_lr{format_param(params['lr'])}"
        f"_bs{params['batch_size']}.keras"
    )
    model_path = MODEL_DIR / model_name
    best_model.save(model_path)
    temp_path.unlink(missing_ok=True)
    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
