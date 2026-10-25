import argparse
import json

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

import os
import sys

# This directory is named with a hyphen, so it is not an importable package.
# The shared modules below are siblings, which means they only resolve when
# this script runs with the repository root on sys.path -- e.g.
# `python -m src.Train-Models.XGBoost_Model_ML`. Adding the script's own
# directory keeps `python src/Train-Models/XGBoost_Model_ML.py` working too.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imported as a MODULE as well as by name: `training_common.DATE_COLUMN` is read at
# run time to snap the split points to date-group boundaries, and mixing the two
# spellings in one file is how the two get out of step.
import training_common

from boosting_training import (
    BoosterWrapper,
    predict_probabilities,
    prepare_data,
    sample_params,
    split_train_es_calib,
    split_train_test,
    train_model,
    walk_forward_cv_loss,
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
    report_search_selection_risk,
    write_run_manifest,
    xgboost_model_filename,
)
from training_reporting import (
    MIN_CALIBRATION_ROWS,
    log_loss_standard_error,
    make_calibrated_estimator,
    report_calibration_effect,
    report_closing_line,
    resolve_calibration_method,
)


MODEL_DIR = BASE_DIR / "Models" / "XGBoost_Models"

NUM_CLASSES = MONEYLINE_NUM_CLASSES
TARGET_COLUMN = MONEYLINE_TARGET
LABELS = [0, 1]
# OU stays out of this list on purpose, so it is a feature here exactly as it is
# for XGBoost_Model_UO and NN_Model_UO: the runner feeds every totals model the
# same input width and order. See training_common.drop_columns_for.
DROP_COLUMNS = drop_columns_for(TARGET_COLUMN)


def add_search_arguments(parser):
    """The flags every trainer shares. See Logistic_Regression_ML for the why.

    `--trials` used to default to 100 here. It is 15 now:
    `selection_bias_estimate` returns 2.50 standard errors at 100 trials against
    1.74 at 15, on this sample size, and the measured gap between the best and the
    mean CV score grows by 0.0011 nats between 15 and 100.
    """
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET, help="Dataset table name."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help=(
            "Number of parameter trials. Default is 15, not 100: measured here, "
            "CV noise exceeds the true spread of test performance by 8x, and 100 "
            "trials buy a 2.50 SE winner's curse."
        ),
    )
    parser.add_argument(
        "--splits", type=int, default=5, help="Walk-forward CV folds (default 5)."
    )
    parser.add_argument(
        "--fold-rows",
        type=int,
        default=0,
        help=(
            "Validation rows per fold. 0 (default) caps a fold at one NBA season "
            "(1230 rows) rather than at n/(splits+1), which was two seasons."
        ),
    )
    parser.add_argument(
        "--fold-half-life",
        type=float,
        default=1.0,
        help=(
            "Fold half-life for the recency-weighted CV mean. 1.0 (default) "
            "gives the newest fold 52% of the weight."
        ),
    )
    parser.add_argument(
        "--calibration",
        default="sigmoid",
        # calibration_choices() reads argv, because argparse fixes `choices`
        # before it looks at any other flag. See its docstring.
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
    parser = argparse.ArgumentParser(description="Train XGBoost moneyline model with walk-forward CV.")
    add_search_arguments(parser)
    # Default "none" because the product prices expected value straight off these
    # probabilities: rebalancing the training classes pushes the base rate off
    # (measured mean P(home) 0.508 vs a true 0.569) and calibration matters more
    # than balanced accuracy here.
    parser.add_argument(
        "--class-weights",
        default="none",
        choices=["none", "inverse", "balanced"],
        help="Training-time class weighting; rebalancing costs probability calibration.",
    )
    return parser


def main():
    args = build_parser().parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset)
    if df.empty:
        print(f"No rows found for dataset {args.dataset}.")
        return

    X, y, feature_names, data = prepare_data(df, TARGET_COLUMN, DROP_COLUMNS)
    # `dates` makes every cut land on a date-group boundary: the dataset holds
    # whole slates (mean 6.45 rows per snapshot date, max 40), and a cut through
    # one of them puts part of a single night's games on each side.
    snapshot_dates = data[training_common.DATE_COLUMN]
    X_train_val, y_train_val, X_test, y_test = split_train_test(
        X, y, dates=snapshot_dates
    )
    fold_rows = args.fold_rows if args.fold_rows > 0 else None

    rng = np.random.default_rng(args.seed)
    best = {
        "val_loss": float("inf"),
        "params": None,
        "num_boost_round": None,
        "folds": [],
    }

    for trial in range(1, args.trials + 1):
        # seed=args.seed + trial made every trial a different configuration even
        # when only the seed differed, so the search selected on seed luck. The
        # seed is a fixed property of the run, not a hyperparameter.
        params, num_boost_round = sample_params(
            rng,
            seed=args.seed,
            num_classes=NUM_CLASSES,
            class_weights=args.class_weights,
        )
        val_loss, per_fold = walk_forward_cv_loss(
            X_train_val,
            y_train_val,
            params,
            num_boost_round,
            args.splits,
            NUM_CLASSES,
            args.class_weights,
            fold_rows=fold_rows,
            half_life_folds=args.fold_half_life,
            dates=snapshot_dates.iloc[: len(X_train_val)],
        )
        if val_loss is None:
            continue

        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["params"] = params
            best["num_boost_round"] = num_boost_round
            best["folds"] = per_fold

        fold_text = " ".join(f"{value:.4f}" for value in per_fold)
        print(
            f"Trial {trial}/{args.trials}: val log loss {val_loss:.4f} "
            f"[recency-weighted; folds {fold_text}; last fold {per_fold[-1]:.4f}]"
        )

    if best["params"] is None:
        print("No valid parameter set found.")
        return

    # Four segments, not three. XGBoost early-stops on the last eval entry, so
    # passing dcalib as dval would choose the boosting round on exactly the rows the
    # sigmoid is then fitted to. See boosting_training.split_train_es_calib.
    X_train, y_train, X_es, y_es, X_calib, y_calib = split_train_es_calib(
        X_train_val, y_train_val, dates=snapshot_dates.iloc[: len(X_train_val)]
    )
    best_model = train_model(
        X_train,
        y_train,
        X_es,
        y_es,
        best["params"],
        best["num_boost_round"],
        NUM_CLASSES,
        args.class_weights,
    )

    calibration_method = resolve_calibration_method(
        args.calibration, len(X_calib), allow_isotonic=args.allow_isotonic
    )
    raw_probabilities = predict_probabilities(best_model, X_test, NUM_CLASSES)
    calibrator = None
    if calibration_method == "none":
        probabilities = raw_probabilities
    else:
        calibrator = make_calibrated_estimator(
            BoosterWrapper(best_model, NUM_CLASSES), calibration_method
        )
        calibrator.fit(X_calib, y_calib)
        probabilities = calibrator.predict_proba(X_test)

    y_pred = np.argmax(probabilities, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    test_loss = log_loss(y_test, probabilities, labels=list(range(NUM_CLASSES)))

    print(f"Best val log loss: {best['val_loss']:.4f}")
    fold_count = len(best["folds"])
    print(
        f"Configurations tried: {args.trials} over {fold_count} validation folds "
        f"(recency-weighted mean, not a flat mean)."
    )
    report_search_selection_risk(
        n_configs_tried=args.trials,
        metric_name="held-out log loss",
        n_test_rows=len(y_test),
        standard_error=log_loss_standard_error(y_test, probabilities, labels=LABELS),
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

    # The test rows are the tail of the sorted frame, taken by the length of the
    # test matrix rather than by recomputing the split point, so the rows cannot
    # drift out of step with y_test.
    test_rows = data.tail(len(X_test))
    report_closing_line(
        test_rows,
        home_win_truth=test_rows["Home-Team-Win"].astype(int).to_numpy(),
        moneyline_picks=y_pred,
    )

    params = best["params"]
    # The filename carries the target and the held-out accuracy, nothing else.
    # The winning configuration goes into the manifest below, which is
    # gitignored: a filename is published by default, a manifest is not. The
    # runner globs on the ML/UO infix here, so this remains discoverable.
    model_name = xgboost_model_filename("ML", accuracy)
    model_path = MODEL_DIR / model_name
    best_model.save_model(str(model_path))
    print(f"Saved model: {model_path}")

    manifest_path = write_run_manifest(model_path, {
        "schema": 1,
        "artifact": model_path.name,
        "kind": "moneyline",
        "target": TARGET_COLUMN,
        "objective": params["objective"],
        "eval_metric": params["eval_metric"],
        "accuracy": float(accuracy),
        "test_log_loss": float(test_loss),
        "best_val_log_loss": float(best["val_loss"]),
        "num_boost_round": int(best["num_boost_round"]),
        "n_features": len(feature_names),
        "params": params,
        "dataset": args.dataset,
        "seed": args.seed,
        "trials": args.trials,
        "splits": args.splits,
        "fold_rows": fold_rows,
        "class_weights": args.class_weights,
        "calibration": calibration_method,
    })
    print(f"Saved run manifest: {manifest_path}")

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