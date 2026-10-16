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
# `python -m src.Train-Models.Logistic_Regression_ML`. Adding the script's own
# directory keeps `python src/Train-Models/Logistic_Regression_ML.py` working too.
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
    MONEYLINE_NUM_CLASSES,
    MONEYLINE_TARGET,
    calibration_choices,
    drop_columns_for,
    impute_split,
    load_dataset,
    recency_weighted_fold_mean,
    report_search_selection_risk,
    walk_forward_folds,
)
from training_reporting import (
    MIN_CALIBRATION_ROWS,
    log_loss_standard_error,
    make_calibrated_estimator,
    report_calibration_effect,
    report_closing_line,
    resolve_calibration_method,
)

NUM_CLASSES = MONEYLINE_NUM_CLASSES
TARGET_COLUMN = MONEYLINE_TARGET
# The logistic regression trainers drop the team ids as well; the XGBoost and NN
# trainers do not. See training_common.drop_columns_for.
DROP_COLUMNS = drop_columns_for(TARGET_COLUMN, include_team_ids=True)
LABELS = [0, 1]


def add_search_arguments(parser):
    """The flags every trainer shares, in one place.

    `--trials` defaults to 15, not 100. Measured on the current table over 12
    sampled configurations: the standard deviation of the 5-fold CV score ACROSS
    configurations is 0.0046 nats, while the entire true range of held-out log
    loss across the same twelve is 0.0009 -- the search was ranking noise, by a
    factor of about five. `selection_bias_estimate` returns 2.50 SE at 100 trials
    against 1.74 SE at 15. Ten to fifteen trials IS essentially the entire signal;
    a hundred buys about 0.0011 nats. See `training_common.DEFAULT_TRIALS` for
    the full derivation, and ROADMAP.md Level 6.2, which lists "more
    hyperparameter search without selection correction" as an explicit non-goal.

    `--fold-rows` and `--splits` together set the validation geometry; see
    training_common.walk_forward_folds for why the width is capped at one
    season and why the score is recency-weighted.
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
            "(1230 rows) rather than at n/(splits+1), which was two seasons and "
            "two years per fold."
        ),
    )
    parser.add_argument(
        "--fold-half-life",
        type=float,
        default=1.0,
        help=(
            "Fold half-life for the recency-weighted CV mean. 1.0 (default) "
            "gives the newest fold 52% of the weight; larger flattens it toward "
            "the old unweighted mean."
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
    parser = argparse.ArgumentParser(description="Train logistic regression moneyline model.")
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
    # whole slates (mean 6.45 rows per snapshot date, max 40), and a cut through
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
        solver = "liblinear" if penalty == "l1" else "lbfgs"
        # random_state is pinned for every solver, not just the randomised ones:
        # an unpinned seed makes a "best" configuration irreproducible, and the
        # search would then be selecting on solver luck as well as seed luck.
        estimator_kwargs = logistic_regression_kwargs(
            C, penalty, solver, random_state=args.seed, num_classes=NUM_CLASSES
        )

        base_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
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
            max_iter=3000,
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
        calibrated = model
        y_proba = model.predict_proba(X_test_imputed)
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
    print(f"Test ROC AUC: {roc_auc_score(y_test, y_proba[:, 1]):.4f}")
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
        moneyline_picks=y_pred,
    )
    print("Classification report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
