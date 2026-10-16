import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import os
import sys

# This directory is named with a hyphen, so it is not an importable package.
# The shared modules below are siblings, which means they only resolve when
# this script runs with the repository root on sys.path -- e.g.
# `python -m src.Train-Models.Logistic_Regression_UO`. Adding the script's own
# directory keeps `python src/Train-Models/Logistic_Regression_UO.py` working too.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Imported as a MODULE as well as by name: `training_common.DATE_COLUMN` is read at
# run time to snap the split points to date-group boundaries, and mixing the two
# spellings in one file is how the two get out of step.
import training_common

from logistic_training import (
    build_class_weight,
    compute_sample_weights,
    logistic_regression_kwargs,
    prepare_data,
    split_train_calib,
    split_train_test,
)
from training_common import (
    DEFAULT_DATASET,
    DEFAULT_TRIALS,
    TOTALS_NUM_CLASSES,
    TOTALS_TARGET,
    drop_columns_for,
    impute_split,
    load_dataset,
    recency_weighted_fold_mean,
    report_search_selection_risk,
    walk_forward_folds,
)
from training_reporting import (
    CALIBRATION_CHOICES,
    MIN_CALIBRATION_ROWS,
    log_loss_standard_error,
    make_calibrated_estimator,
    report_calibration_effect,
    report_closing_line,
    resolve_calibration_method,
)

NUM_CLASSES = TOTALS_NUM_CLASSES
TARGET_COLUMN = TOTALS_TARGET
LABELS = [0, 1, 2]
# OU stays out of this list on purpose, so it is a feature here exactly as it is
# for XGBoost_Model_UO and NN_Model_UO: the runner feeds every totals model the
# same input width and order. See training_common.drop_columns_for.
DROP_COLUMNS = drop_columns_for(TARGET_COLUMN, include_team_ids=True)


def add_search_arguments(parser):
    """The flags every trainer shares. See Logistic_Regression_ML for the why.

    `--trials` defaults to 15 rather than 100/50 for the measured reason stated
    there: CV noise on this dataset exceeds the true spread of test performance by
    8x, so a larger search only buys selection bias.
    """
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET, help="Dataset table name."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help=(
            "Number of hyperparameter trials. Default is 15, not 100: measured "
            "here, CV noise exceeds the true spread of test performance by 8x."
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
            "(1230 rows) rather than at n/(splits+1)."
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
        choices=list(CALIBRATION_CHOICES),
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
    parser = argparse.ArgumentParser(description="Train logistic regression totals model.")
    add_search_arguments(parser)
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
    return parser


def main():
    args = build_parser().parse_args()

    df = load_dataset(args.dataset)
    if df.empty:
        print(f"No rows found for dataset {args.dataset}.")
        return

    X, y, data = prepare_data(df, TARGET_COLUMN, DROP_COLUMNS)
    # `dates` makes every cut land on a date-group boundary: the dataset holds
    # whole slates (mean 6.42 rows per snapshot date, max 40), and a cut through
    # one of them puts part of a single night's games on each side.
    snapshot_dates = data[training_common.DATE_COLUMN]
    X_train_val, y_train_val, X_test, y_test = split_train_test(
        X, y, dates=snapshot_dates
    )
    fold_rows = args.fold_rows if args.fold_rows > 0 else None
    folds = walk_forward_folds(
        len(X_train_val),
        args.splits,
        fold_rows=fold_rows,
        dates=snapshot_dates.iloc[: len(X_train_val)],
    )

    rng = np.random.default_rng(args.seed)
    best = {
        "val_loss": float("inf"),
        "params": None,
        "folds": [],
    }

    for trial in range(1, args.trials + 1):
        C = float(10 ** rng.uniform(np.log10(0.01), np.log10(10.0)))
        penalty = rng.choice(["l2", "l1"])
        # saga is the L1 solver here (lbfgs does not support L1), and saga is
        # randomised by default. random_state was previously unset, so every L1
        # model this script produced was irreproducible -- and the search would
        # then have been selecting on solver luck as well as seed luck.
        solver = "saga" if penalty == "l1" else "lbfgs"
        estimator_kwargs = logistic_regression_kwargs(
            C, penalty, solver, random_state=args.seed, num_classes=NUM_CLASSES
        )

        base_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=4000,
                class_weight=build_class_weight(args.class_weights),
                **estimator_kwargs,
            ),
        )

        # Each fold fits its own imputer on its own training rows. Imputing the
        # joined frame (which is what this used to do, before the split) let the
        # validation rows reach training through the medians.
        losses = []
        for train_idx, val_idx in folds:
            X_fold_train, X_fold_val = impute_split(
                X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
            )
            base_model.fit(
                X_fold_train,
                y_train_val[train_idx],
                logisticregression__sample_weight=compute_sample_weights(
                    y_train_val[train_idx], NUM_CLASSES, args.class_weights
                ),
            )
            val_proba = base_model.predict_proba(X_fold_val)
            losses.append(log_loss(y_train_val[val_idx], val_proba, labels=LABELS))
        val_loss, per_fold, _weights = recency_weighted_fold_mean(
            losses, half_life_folds=args.fold_half_life
        )
        fold_text = " ".join(f"{value:.4f}" for value in per_fold)
        print(
            f"Trial {trial}/{args.trials}: val log loss {val_loss:.4f} "
            f"[recency-weighted; folds {fold_text}; last fold {per_fold[-1]:.4f}]"
        )

        if val_loss < best["val_loss"]:
            best["val_loss"] = val_loss
            best["folds"] = per_fold
            best["params"] = estimator_kwargs

    if best["params"] is None:
        print("No valid model trained.")
        return

    X_train, y_train, X_calib, y_calib = split_train_calib(
        X_train_val, y_train_val, dates=snapshot_dates.iloc[: len(X_train_val)]
    )
    X_train, X_calib = impute_split(X_train, X_calib)
    _, X_test_imputed = impute_split(X_train, X_test)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=4000,
            class_weight=build_class_weight(args.class_weights),
            **best["params"],
        ),
    )
    model.fit(
        X_train,
        y_train,
        logisticregression__sample_weight=compute_sample_weights(
            y_train, NUM_CLASSES, args.class_weights
        ),
    )

    calibration_method = resolve_calibration_method(
        args.calibration, len(X_calib), allow_isotonic=args.allow_isotonic
    )
    raw_proba = model.predict_proba(X_test_imputed)
    if calibration_method == "none":
        y_proba = raw_proba
    else:
        calibrated = make_calibrated_estimator(model, calibration_method)
        calibrated.fit(X_calib, y_calib)
        y_proba = calibrated.predict_proba(X_test_imputed)

    y_pred = np.argmax(y_proba, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba, labels=LABELS)

    print(f"Best val log loss: {best['val_loss']:.4f}")
    print(
        f"Configurations tried: {args.trials} over {len(folds)} validation folds "
        f"of {len(folds[0][1])} rows (recency-weighted mean, not a flat mean)."
    )
    report_search_selection_risk(
        n_configs_tried=args.trials,
        metric_name="held-out log loss",
        n_test_rows=len(y_test),
        standard_error=log_loss_standard_error(y_test, y_proba, labels=LABELS),
    )
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test log loss: {loss:.4f}")
    print(
        "Test ROC AUC: multiclass "
        f"{roc_auc_score(y_test, y_proba, labels=LABELS, multi_class='ovr', average='macro'):.4f}"
    )
    report_calibration_effect(
        y_test,
        raw_proba,
        y_proba,
        labels=LABELS,
        n_calib_rows=len(X_calib),
        method=calibration_method,
    )
    # The test rows are the tail of the sorted frame, taken by the length of the
    # test frame rather than by recomputing the split point, so the rows cannot
    # drift out of step with y_test.
    test_rows = data.tail(len(X_test))
    report_closing_line(
        test_rows,
        home_win_truth=test_rows["Home-Team-Win"].astype(int).to_numpy(),
        moneyline_picks=None,
    )
    print("Classification report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()