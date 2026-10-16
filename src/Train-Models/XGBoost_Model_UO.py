import argparse
import json

import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

import os
import sys

# This directory is named with a hyphen, so it is not an importable package.
# The shared modules below are siblings, which means they only resolve when
# this script runs with the repository root on sys.path -- e.g.
# `python -m src.Train-Models.Logistic_Regression_ML`. Adding the script's own
# directory keeps `python src/Train-Models/Logistic_Regression_ML.py` working too.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from boosting_training import (
    BoosterWrapper,
    prepare_data,
    sample_params,
    split_train_calib,
    split_train_test,
    train_model,
    walk_forward_cv_loss,
)
from training_common import (
    BASE_DIR,
    TOTALS_NUM_CLASSES,
    TOTALS_TARGET,
    drop_columns_for,
    format_param,
    load_dataset,
)
from training_reporting import make_calibrated_estimator, report_closing_line


MODEL_DIR = BASE_DIR / "Models" / "XGBoost_Models"

DEFAULT_DATASET = "dataset_2012-26"
NUM_CLASSES = TOTALS_NUM_CLASSES
TARGET_COLUMN = TOTALS_TARGET
# OU stays out of this list on purpose, so it is a feature here exactly as it is
# for NN_Model_UO: the runner feeds every totals model the same input width and
# order. See training_common.drop_columns_for.
DROP_COLUMNS = drop_columns_for(TARGET_COLUMN)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost totals model with walk-forward CV.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Dataset table name.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--trials", type=int, default=100, help="Number of parameter trials.")
    parser.add_argument("--splits", type=int, default=5, help="Walk-forward CV splits.")
    parser.add_argument(
        "--calibration",
        default="sigmoid",
        choices=["sigmoid", "isotonic", "none"],
        help="Calibration method for probabilities.",
    )
    # Default "none" because the product prices expected value straight off these
    # probabilities: rebalancing the training classes distorts them in the same
    # way it distorts the moneyline model's base rate, and calibration matters
    # more than balanced accuracy here.
    parser.add_argument(
        "--class-weights",
        default="none",
        choices=["none", "inverse", "balanced"],
        help="Training-time class weighting; rebalancing costs probability calibration.",
    )
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    if df.empty:
        print(f"No rows found for dataset {args.dataset}.")
        return

    X, y, feature_names, data = prepare_data(df, TARGET_COLUMN, DROP_COLUMNS)
    X_train_val, y_train_val, X_test, y_test = split_train_test(X, y)

    rng = np.random.default_rng(args.seed)
    best = {
        "val_loss": float("inf"),
        "params": None,
        "num_boost_round": None,
    }

    for trial in range(1, args.trials + 1):
        params, num_boost_round = sample_params(rng, seed=args.seed + trial, num_classes=NUM_CLASSES)
        val_loss = walk_forward_cv_loss(
            X_train_val,
            y_train_val,
            params,
            num_boost_round,
            args.splits,
            NUM_CLASSES,
            args.class_weights,
        )
        if val_loss is None:
            continue

        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["params"] = params
            best["num_boost_round"] = num_boost_round

        print(f"Trial {trial}/{args.trials}: val log loss {val_loss:.4f}")

    if best["params"] is None:
        print("No valid parameter set found.")
        return

    X_train, y_train, X_calib, y_calib = split_train_calib(X_train_val, y_train_val)
    best_model = train_model(
        X_train,
        y_train,
        X_calib,
        y_calib,
        best["params"],
        best["num_boost_round"],
        NUM_CLASSES,
        args.class_weights,
    )

    calibrator = None
    if args.calibration == "none":
        probabilities = best_model.predict(xgb.DMatrix(X_test))
    else:
        calibrator = make_calibrated_estimator(
            BoosterWrapper(best_model, NUM_CLASSES), args.calibration
        )
        calibrator.fit(X_calib, y_calib)
        probabilities = calibrator.predict_proba(X_test)

    y_pred = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    test_loss = log_loss(y_test, probabilities, labels=list(range(NUM_CLASSES)))

    print(f"Best val log loss: {best['val_loss']:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test log loss: {test_loss:.4f}")

    # The test rows are the tail of the sorted frame, taken by the length of the
    # test matrix rather than by recomputing the split point, so the rows cannot
    # drift out of step with y_test.
    test_rows = data.tail(len(X_test))
    report_closing_line(
        test_rows,
        home_win_truth=test_rows["Home-Team-Win"].astype(int).to_numpy(),
        moneyline_picks=None,
    )

    params = best["params"]
    model_name = (
        f"XGBoost_{accuracy * 100:.1f}%_UO"
        f"_md{params['max_depth']}"
        f"_eta{format_param(params['eta'])}"
        f"_sub{format_param(params['subsample'])}"
        f"_col{format_param(params['colsample_bytree'])}"
        f"_cbl{format_param(params['colsample_bylevel'])}"
        f"_cbn{format_param(params['colsample_bynode'])}"
        f"_mcw{params['min_child_weight']}"
        f"_g{format_param(params['gamma'])}"
        f"_mds{params['max_delta_step']}"
        f"_mb{params['max_bin']}"
        f"_l{format_param(params['lambda'])}"
        f"_a{format_param(params['alpha'])}"
        f"_nb{best['num_boost_round']}.json"
    )
    model_path = MODEL_DIR / model_name
    best_model.save_model(str(model_path))
    print(f"Saved model: {model_path}")

    # The model is a positional matrix, so nothing in the .json records which
    # columns produced it. This is the exact list X was built from.
    features_path = MODEL_DIR / f"{model_path.stem}_features.json"
    features_path.write_text(json.dumps(feature_names, indent=2))
    print(f"Saved feature order: {features_path}")

    if calibrator is not None:
        calibration_path = MODEL_DIR / f"{model_path.stem}_calibration.pkl"
        joblib.dump(calibrator, calibration_path)
        print(f"Saved calibration: {calibration_path}")


if __name__ == "__main__":
    main()
