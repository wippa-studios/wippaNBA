import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import os
import sys

# This directory is named with a hyphen, so it is not an importable package.
# The shared modules below are siblings, which means they only resolve when
# this script runs with the repository root on sys.path -- e.g.
# `python -m src.Train-Models.Logistic_Regression_ML`. Adding the script's own
# directory keeps `python src/Train-Models/Logistic_Regression_ML.py` working too.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logistic_training import (
    build_class_weight,
    compute_sample_weights,
    prepare_data,
    split_train_calib,
    split_train_test,
)
from training_common import (
    MONEYLINE_NUM_CLASSES,
    MONEYLINE_TARGET,
    drop_columns_for,
    load_dataset,
)
from training_reporting import make_calibrated_estimator, report_closing_line


DEFAULT_DATASET = "dataset_2012-26_new"
NUM_CLASSES = MONEYLINE_NUM_CLASSES
TARGET_COLUMN = MONEYLINE_TARGET
# The logistic regression trainers drop the team ids as well; the XGBoost and NN
# trainers do not. See training_common.drop_columns_for.
DROP_COLUMNS = drop_columns_for(TARGET_COLUMN, include_team_ids=True)


def main():
    parser = argparse.ArgumentParser(description="Train logistic regression moneyline model.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset table name.")
    parser.add_argument("--trials", type=int, default=50, help="Number of hyperparameter trials.")
    parser.add_argument("--splits", type=int, default=5, help="Walk-forward CV splits.")
    parser.add_argument(
        "--calibration",
        default="sigmoid",
        choices=["sigmoid", "isotonic", "none"],
        help="Calibration method for probabilities.",
    )
    # Default "none" because the product prices expected value straight off these
    # probabilities: rebalancing the training classes pushes them off the true base
    # rate and calibration matters more than balanced accuracy here. This script
    # used to hard-code class_weight="balanced".
    parser.add_argument(
        "--class-weights",
        default="none",
        choices=["none", "inverse", "balanced"],
        help="Training-time class weighting; rebalancing costs probability calibration.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    df = load_dataset(args.dataset)
    if df.empty:
        print(f"No rows found for dataset {args.dataset}.")
        return

    X, y, data = prepare_data(df, TARGET_COLUMN, DROP_COLUMNS)
    X_train_val, y_train_val, X_test, y_test = split_train_test(X, y)

    rng = np.random.default_rng(args.seed)
    best = {
        "val_loss": float("inf"),
        "params": None,
    }

    for trial in range(1, args.trials + 1):
        C = float(10 ** rng.uniform(np.log10(0.01), np.log10(10.0)))
        penalty = rng.choice(["l2", "l1"])
        solver = "liblinear" if penalty == "l1" else "lbfgs"

        base_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight=build_class_weight(args.class_weights),
                C=C,
                penalty=penalty,
                solver=solver,
            ),
        )

        tscv = TimeSeriesSplit(n_splits=args.splits)
        losses = []
        for train_idx, val_idx in tscv.split(X_train_val):
            base_model.fit(
                X_train_val.iloc[train_idx],
                y_train_val[train_idx],
                logisticregression__sample_weight=compute_sample_weights(
                    y_train_val[train_idx], NUM_CLASSES, args.class_weights
                ),
            )
            val_proba = base_model.predict_proba(X_train_val.iloc[val_idx])
            loss = log_loss(y_train_val[val_idx], val_proba, labels=[0, 1])
            losses.append(loss)
        val_loss = float(np.mean(losses)) if losses else float("inf")
        print(f"Trial {trial}/{args.trials}: val log loss {val_loss:.4f}")

        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["params"] = {"C": C, "penalty": penalty, "solver": solver}

    if best["params"] is None:
        print("No valid model trained.")
        return

    X_train, y_train, X_calib, y_calib = split_train_calib(X_train_val, y_train_val)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=3000,
            class_weight=build_class_weight(args.class_weights),
            C=best["params"]["C"],
            penalty=best["params"]["penalty"],
            solver=best["params"]["solver"],
        ),
    )
    model.fit(
        X_train,
        y_train,
        logisticregression__sample_weight=compute_sample_weights(
            y_train, NUM_CLASSES, args.class_weights
        ),
    )

    if args.calibration == "none":
        calibrated = model
    else:
        calibrated = make_calibrated_estimator(model, args.calibration)
        calibrated.fit(X_calib, y_calib)

    y_pred = calibrated.predict(X_test)
    y_proba = calibrated.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)

    print(f"Best val log loss: {best['val_loss']:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test log loss: {loss:.4f}")
    # The test rows are the tail of the sorted frame, taken by the length of the
    # test frame rather than by recomputing the split point, so the rows cannot
    # drift out of step with y_test.
    test_rows = data.tail(len(X_test))
    report_closing_line(
        test_rows,
        home_win_truth=test_rows["Home-Team-Win"].astype(int).to_numpy(),
        moneyline_picks=y_pred,
    )
    print("Classification report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
