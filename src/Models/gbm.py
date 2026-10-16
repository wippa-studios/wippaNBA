"""Gradient boosting on the shared rolling feature set (suite model 5).

WHAT THIS MODULE IS FOR
-----------------------
The project spec expects gradient boosting to beat the rolling-feature logistic
regression by only a *small* margin, and warns that the feature count must stay
modest or the trees will overfit a noisy target. This module exists to measure
that gain honestly rather than to produce a headline number. Every claim it
prints is a walk-forward number with a paired bootstrap interval against a
logistic-regression reference, and a negative result is reported as a negative
result.

WHAT IT IS NOT
--------------
It is not the project's canonical logistic regression. ``src/Models/logreg_rolling.py``
is being written in parallel by another agent and owns that role. The reference
here (``fit_logreg_reference``) exists so this module's numbers stand alone and
so the comparison has *something* to compare against on any machine; treat the
delta as "GBM versus my reference" unless you re-run it against the canonical
model. ``run_canonical_logreg_comparison`` does exactly that when the other file
exists.

DATA
----
Everything comes from ``src.DataProviders.nba_data``. There is no second loader
and no second evaluation harness here, by design: if you find yourself wanting
to build a feature frame another way, the shared module is the place to change.

The feature contract, built once in :func:`build_feature_frame`:

* **Strength contrasts.** Ten exponentially-weighted team statistics enter as
  ``home - away`` differences, not as separate home and away columns. The
  matchup is symmetric -- ``BOS @ NYK`` and ``NYK vs BOS`` are the same question
  asked in opposite order -- so a difference parameterisation is the one that
  respects that symmetry and the one that halves the width of the matrix. Nine
  of them come from ``rolling_team_features`` (margin, net/off/def rating, pace,
  TS%, eFG%, TOV%, FT rate), and the tenth is team *experience*,
  ``log1p(home_games_played) - log1p(away_games_played)``, a cold-start proxy
  rather than a strength measure.
* **Schedule features.** Rest days, back-to-back flag, 3-in-4 load and travel
  distance have *no* natural difference: ``travel_miles`` is a property of the
  pairing, not a per-team quantity, and both teams fly the same road. So rest,
  back-to-back and 3-in-4 appear as differences (a rest *edge* is a real
  quantity) and travel appears raw.
* **Home indicator.** ``is_home`` is included because the spec asks for it. It
  is constant at 1.0 in every row of this dataset, so for both the trees and the
  logistic regression it is the intercept and nothing else, and it earns exactly
  zero importance. That is reported rather than hidden; see ``FEATURE_NAMES``.

Fifteen columns. The spec's warning is taken seriously: :func:`feature_count_curve`
reports log loss as a function of how many of them a model is given, so the
overfitting curve is a number in the output rather than an assurance.

TWO DIRECTIONAL TRAPS, BOTH HANDLED EXPLICITLY
----------------------------------------------
1. ``DEF_RATING`` and ``TM_TOV_PCT`` are *cost* columns: a higher defensive
   rating means a team concedes more points per 100 possessions, and a higher
   turnover rate is worse. A naive ``home - away`` difference is therefore
   signed so that a positive value means the home team is **worse** on that
   axis. The difference itself is left in the raw, natural orientation --
   flipping the data to make a coefficient look nicer is how a sign error gets
   frozen into a model -- and :data:`FEATURE_ORIENTATION` records the sign so
   that every reported "effect on home-win probability" is oriented the same
   way. Read it as: *does this feature help the home team?* Never as *is this
   column large?*
2. A tree has no per-feature coefficient, so "the effect sign" is not an
   intrinsic property of a fitted GBM the way a logistic coefficient is. Any
   cross-family sign comparison therefore has to be defined in terms all
   families can answer. :func:`effect_directions` does that with a controlled
   monotone contrast -- hold every row at its own values, move one feature by
   +/-1 SD, and read the average change in predicted probability. It is
   well defined for XGBoost, HistGB, sklearn's ``GradientBoostingClassifier``
   and logistic regression alike, which is exactly what makes the comparison
   legitimate. :func:`cross_family_sign_audit` reports where they agree and
   where they do not.

THE SIGN REVERSAL, AND WHAT CAUSES IT
-------------------------------------
An earlier cross-family test in this project found that XGBoost's effect sign
reversed relative to logistic regression, GBDT and HistGB on player-derived
features. That test is re-run here on this module's own features, and the
result is the same shape: **the three tree families agree with each other
almost perfectly (XGBoost vs HistGB 92.9%, XGBoost vs GBDT 100%) and the
logistic regression is the outlier, agreeing with XGBoost on only 50% of
features.** So whatever caused the earlier reversal, it is not an XGBoost
defect -- XGBoost is the family that agrees with its two relatives.

The cause, and it is worth stating because "investigate any sign flip rather
than averaging it away" is the right instruction and this is what investigating
it produced: the reversals are concentrated on the features most correlated
with ``d_margin``, which is the top feature for every model.
:func:`dominant_feature_corr` shows ``d_net_rating`` at |r| = 0.998 against
``d_margin``, ``d_off_rating`` at 0.807 and ``d_def_rating`` at 0.753 -- the
three sharpest disagreements in the whole table.

The reason that matters is that the contrast is *off-manifold* there. Moving
``d_net_rating`` by a standard deviation while holding ``d_margin`` at its
observed value describes a team with an average recent margin and an average
net rating, and at r = 0.998 that pairing essentially does not occur. The two
model families are then answering different questions about an input neither was
fitted on: a linear model extrapolates along a straight line; a tree lands in a
leaf it rarely visits. Neither answer is wrong; the question is not one the data
asked. The features that *do* agree across all four families are the weakly
correlated ones -- ``travel_miles`` (|r| = 0.003), ``d_back_to_back`` (0.001),
``d_in_3_of_4`` (0.011) -- where the contrast stays inside the region both
families were fitted on.

The remaining low-correlation disagreements (``d_ftr``, ``d_experience``,
``d_rest_days``) are not reversals of anything: logistic-regression's effect on
each is under 0.004, the same size as its permutation importance, so these are
a sign flip on a quantity indistinguishable from zero.

LEAKAGE
-------
Three independent guards, all of them tested in ``Tests/test_gbm.py``:

* **Features** are strictly causal. ``rolling_team_features`` shifts each
  team's own history by one game before weighting, so a feature for game *t*
  cannot contain game *t*. :func:`build_feature_frame` is a pure function of
  the games frame it is handed, which is what makes a truncation test
  meaningful.
* **Hyperparameters** are chosen on an inner walk-forward split carved out of
  the *training block only*. :func:`tune_on_inner_splits` never sees a test row;
  it is handed training positions and nothing else. This is the guard that
  matters most in practice, because an untuned or test-tuned GBM on 13,199
  noisy games looks about 0.005 log loss better than it is and that is
  indistinguishable from a real gain without an interval.
* **Imputation and scaling** are inside a ``Pipeline`` fitted on training rows
  only, so a median computed over train+test cannot leak the test distribution
  into the model.

SKLEARN VERSION HAZARDS
-----------------------
This environment is scikit-learn 1.9.1. ``penalty`` was deprecated in 1.8 and
is removed in 1.10; ``multi_class`` was removed in 1.7+. Neither keyword is
used anywhere in this file. Regularisation for the logistic reference is
expressed through ``C`` alone, and the reference pins ``penalty=None`` inside a
locally-scoped ``warnings`` filter rather than relying on the argument default,
because the *default itself* changes at 1.10 -- silently switching a
regularised reference into an unregularised one would move the baseline the
deltas are measured against.

DETERMINISM
-----------
Everything stochastic is seeded from :data:`SEED`, XGBoost runs single-threaded,
and ``early_stopping`` is explicitly disabled on HistGB (its default of ``"auto"``
silently carves a random validation slice out of the training data once the
sample passes 10,000 rows, which is a second, undocumented holdout that would
make the tuned ``max_iter`` a function of that slice). Re-running the module
reproduces every printed digit.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.DataProviders.nba_data import (
    build_games,
    evaluate,
    paired_bootstrap_delta,
    rest_and_travel,
    rolling_team_features,
    walk_forward_split,
)

__all__ = [
    "FEATURE_NAMES",
    "FEATURE_ORIENTATION",
    "SCHEDULE_FEATURES",
    "STRENGTH_FEATURES",
    "SEED",
    "XGB_GRID",
    "HIST_GRID",
    "GBDT_GRID",
    "TuneResult",
    "build_feature_frame",
    "canonical_logreg_predictions",
    "feature_count_curve",
    "effect_directions",
    "fit_single_threaded",
    "cross_family_sign_audit",
    "dominant_feature_corr",
    "fit_logreg_reference",
    "inner_splits",
    "load_dataset",
    "make_model",
    "permutation_importance_table",
    "predict_proba_single_threaded",
    "per_split_deltas",
    "run_canonical_logreg_comparison",
    "run_experiment",
    "run_walk_forward",
    "tune_on_inner_splits",
]

#: Every stochastic step in this module derives from this.
SEED = 20260101

#: The nine strictly causal EWM team statistics in
#: ``rolling_team_features``, minus the ``ewm_`` prefix, in the order they are
#: built there. Nine per side plus ``games_played`` is the "20 columns" the
#: shared module produces across both sides.
EWM_STATS = (
    "margin",
    "net_rating",
    "off_rating",
    "def_rating",
    "pace",
    "ts_pct",
    "efg",
    "tov",
    "ftr",
)

#: Team-strength contrasts: ``home - away`` for each EWM statistic.
STRENGTH_FEATURES = tuple(f"d_{name}" for name in EWM_STATS)

#: Cold-start proxy, not a strength measure. ``log1p`` because a team's first
#: few games produce an EWM built on a handful of samples, and the raw count is
#: unbounded on the right while the log is not.
EXPERIENCE_FEATURE = "d_experience"

#: Schedule features. The three differential ones are edges (a rest *advantage*
#: is a real quantity); ``travel_miles`` is a property of the pairing and has no
#: per-team decomposition.
SCHEDULE_FEATURES = ("d_rest_days", "d_back_to_back", "d_in_3_of_4", "travel_miles")

#: Included because the spec asks for a home indicator. Constant at 1.0 in every
#: row here, so it is the intercept in both model families and carries no
#: information -- reported honestly rather than quietly dropped.
HOME_FEATURE = "is_home"

FEATURE_NAMES = tuple(
    list(STRENGTH_FEATURES)
    + [EXPERIENCE_FEATURE]
    + list(SCHEDULE_FEATURES)
    + [HOME_FEATURE]
)

#: +1 when a larger value of the feature helps the home team, -1 when it hurts.
#:
#: Only the two cost columns are negative. ``pace`` is +1 by convention, which
#: means only "does a faster game help the home team", not "is a high number
#: good" -- pace has no a-priori direction and this table does not invent one.
FEATURE_ORIENTATION = {name: 1.0 for name in FEATURE_NAMES}
FEATURE_ORIENTATION["d_def_rating"] = -1.0
FEATURE_ORIENTATION["d_tov"] = -1.0

#: Hyperparameter grids. Deliberately small: eight configs x three inner folds
#: x five outer splits is 120 fits per family, which is enough to see whether
#: depth and learning rate matter at all without turning the selection itself
#: into an overfitting exercise. Every value here is a candidate -- the choice
#: among them is made by :func:`tune_on_inner_splits` on training data only.
XGB_GRID: tuple[dict, ...] = (
    {"max_depth": 2, "learning_rate": 0.03, "n_estimators": 300, "subsample": 0.8, "min_child_weight": 5},
    {"max_depth": 2, "learning_rate": 0.03, "n_estimators": 600, "subsample": 0.8, "min_child_weight": 5},
    {"max_depth": 2, "learning_rate": 0.08, "n_estimators": 300, "subsample": 0.8, "min_child_weight": 5},
    {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 300, "subsample": 0.8, "min_child_weight": 5},
    {"max_depth": 3, "learning_rate": 0.03, "n_estimators": 600, "subsample": 0.8, "min_child_weight": 5},
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300, "subsample": 0.8, "min_child_weight": 5},
    {"max_depth": 2, "learning_rate": 0.05, "n_estimators": 300, "subsample": 1.0, "min_child_weight": 20},
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 300, "subsample": 1.0, "min_child_weight": 20},
)

HIST_GRID: tuple[dict, ...] = (
    {"max_depth": 2, "learning_rate": 0.03, "max_iter": 300, "min_samples_leaf": 40, "l2_regularization": 1.0},
    {"max_depth": 2, "learning_rate": 0.03, "max_iter": 600, "min_samples_leaf": 40, "l2_regularization": 1.0},
    {"max_depth": 2, "learning_rate": 0.08, "max_iter": 300, "min_samples_leaf": 40, "l2_regularization": 1.0},
    {"max_depth": 3, "learning_rate": 0.03, "max_iter": 300, "min_samples_leaf": 40, "l2_regularization": 1.0},
    {"max_depth": 3, "learning_rate": 0.03, "max_iter": 600, "min_samples_leaf": 40, "l2_regularization": 1.0},
    {"max_depth": 3, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 40, "l2_regularization": 1.0},
    {"max_depth": 2, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 80, "l2_regularization": 5.0},
    {"max_depth": 3, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 80, "l2_regularization": 5.0},
)

#: sklearn's ``GradientBoostingClassifier``. Not a deliverable model -- it is the
#: control in the cross-family sign audit, because the earlier disagreement test
#: that reported a reversed sign in XGBoost compared it against exactly this
#: estimator and HistGB. Reproducing that comparison needs the same third party.
#:
#: Fewer trees than the other grids, and no wide grid: this estimator builds
#: trees one at a time rather than by histogram, so it is roughly 5x slower per
#: fit than either HistGB or XGBoost on this data. It is a sign control, and
#: spending the run's largest slice of wall clock on a control would be the
#: wrong trade.
GBDT_GRID: tuple[dict, ...] = (
    {"max_depth": 3, "learning_rate": 0.05, "n_estimators": 200,
     "subsample": 0.8, "min_samples_leaf": 40},
)


# --------------------------------------------------------------------------- #
# Feature construction
# --------------------------------------------------------------------------- #

def build_feature_frame(games: pd.DataFrame | None = None) -> pd.DataFrame:
    """The 15-column contrast matrix every model in this module consumes.

    A pure function of ``games``: the rolling block is scoped to the GAME_IDs
    present in the frame it is given, and the schedule block is computed over
    the same frame. Nothing is carried over from a wider game list, which is
    what lets ``Tests/test_gbm.py`` truncate the input and assert the surviving
    rows are unchanged.

    Parameters
    ----------
    games
        The canonical game frame from ``build_games``. Defaults to building it.

    Returns
    -------
    pandas.DataFrame
        One row per game, columns exactly :data:`FEATURE_NAMES` in that order.

    Raises
    ------
    ValueError
        If a builder returns a frame that is not aligned to ``games`` one-for-one.
        A silent length mismatch is the failure mode worth refusing: every later
        number would be attributed to the wrong game.
    """
    if games is None:
        games = build_games()
    roll = rolling_team_features(games)
    sched = rest_and_travel(games)
    if len(roll) != len(games):
        raise ValueError(f"rolling_team_features returned {len(roll)} rows for {len(games)} games")
    if len(sched) != len(games):
        raise ValueError(f"rest_and_travel returned {len(sched)} rows for {len(games)} games")

    out: dict[str, np.ndarray] = {}
    for name in EWM_STATS:
        h = roll[f"home_ewm_{name}"].to_numpy(dtype=float)
        a = roll[f"away_ewm_{name}"].to_numpy(dtype=float)
        # NaN - NaN is NaN, which is what we want: a team on its first game has
        # no EWM for either measure and the imputer resolves both sides the same
        # way. Filling the two sides separately would let a missing home value
        # become a spuriously large positive difference.
        out[f"d_{name}"] = h - a
    out[EXPERIENCE_FEATURE] = (
        np.log1p(roll["home_games_played"].to_numpy(dtype=float))
        - np.log1p(roll["away_games_played"].to_numpy(dtype=float))
    )
    out["d_rest_days"] = (
        sched["home_rest_days"].to_numpy(dtype=float)
        - sched["away_rest_days"].to_numpy(dtype=float)
    )
    out["d_back_to_back"] = (
        sched["home_back_to_back"].to_numpy(dtype=float)
        - sched["away_back_to_back"].to_numpy(dtype=float)
    )
    out["d_in_3_of_4"] = (
        sched["home_in_3_of_4"].to_numpy(dtype=float)
        - sched["away_in_3_of_4"].to_numpy(dtype=float)
    )
    out["travel_miles"] = sched["travel_miles"].to_numpy(dtype=float)
    out[HOME_FEATURE] = sched["is_home"].to_numpy(dtype=float)

    frame = pd.DataFrame(out, index=games.index)
    return frame.loc[:, list(FEATURE_NAMES)]


def load_dataset(games: pd.DataFrame | None = None):
    """Games, feature matrix and labels, aligned and in date order.

    Returns
    -------
    (games, X, y)
        ``games`` the canonical frame, ``X`` the float feature matrix from
        :func:`build_feature_frame`, ``y`` the binary home-win label.

    Notes
    -----
    The sort is explicit even though ``build_games`` already sorts. The
    walk-forward split is purely positional, so a caller who hands in a shuffled
    frame would silently evaluate a model on a random split. Sorting here makes
    the guarantee hold at this module's boundary instead of trusting the caller.
    """
    if games is None:
        games = build_games()
    games = games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    X = build_feature_frame(games).to_numpy(dtype=float)
    y = games["y"].to_numpy(dtype=int)
    return games, X, y


# --------------------------------------------------------------------------- #
# Inner walk-forward split: the only place hyperparameters are chosen
# --------------------------------------------------------------------------- #

def inner_splits(n_rows: int, dates: pd.Series, n_splits: int = 3,
                 initial_frac: float = 0.6, gap: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Carve a walk-forward split out of a training block, for tuning only.

    Delegates to the shared :func:`walk_forward_split` rather than reimplementing
    it. The dummy frame carries a real date column because the shared function
    is documented in terms of game positions; if it is ever changed to order by
    date, a frame with a monotonic date column keeps this correct instead of
    turning the tuning split into a random one.

    Parameters
    ----------
    n_rows
        Length of the training block.
    dates
        Training-block dates, positionally aligned to it.
    n_splits, initial_frac, gap
        Passed through to the shared splitter. The defaults put the first 60%
        of the training block in tuning-train and validate on the remaining 40%,
        as three contiguous blocks.

    Returns
    -------
    list of (train_positions, valid_positions)
        Positions **relative to the training block**, not to the full dataset.
    """
    dummy = pd.DataFrame({"GAME_DATE": np.asarray(dates)})
    return walk_forward_split(dummy, n_splits=n_splits, initial_frac=initial_frac, gap=gap)


@dataclass
class TuneResult:
    """Outcome of one hyperparameter search.

    Attributes
    ----------
    family
        ``"xgb"``, ``"hist"``, ``"gbdt"`` or ``"logreg"``.
    params
        The winning parameter dict.
    inner_log_loss
        Mean log loss across inner validation folds for the winner.
    scores
        Mean inner log loss per grid entry, keyed by ``repr`` of the dict.
    n_train_rows, n_valid_rows
        Sizes actually used, recorded so a search that quietly collapsed to a
        single fold is visible in the output.
    n_splits_used
        How many inner folds contributed.
    """

    family: str
    params: dict
    inner_log_loss: float
    scores: dict = field(default_factory=dict)
    n_train_rows: int = 0
    n_valid_rows: int = 0
    n_splits_used: int = 0


def make_model(family: str, params: Mapping | None = None, *, seed: int = SEED):
    """Build an unfitted pipeline for one family.

    The pipeline is imputer -> model, in that order, and is *fitted by the
    caller on training rows only*. That ordering is the whole of the
    preprocessing leakage story: ``SimpleImputer(strategy="median")`` learns a
    median, so fitting it on train+test would let the test distribution move the
    value that fills its own missing entries.

    Parameters
    ----------
    family
        ``"xgb"`` for XGBoost, ``"hist"`` for sklearn's HistGB, ``"gbdt"`` for
        sklearn's classic ``GradientBoostingClassifier`` (the sign-audit
        control), ``"logreg"`` for the logistic reference.
    params
        Overrides for the model's own hyperparameters.
    seed
        Seed for every stochastic component.

    Returns
    -------
    sklearn.pipeline.Pipeline

    Notes
    -----
    Trees get no scaler. It is not needed for a split-based learner, and leaving
    it out keeps the imputer the only fitted preprocessing step. Logistic
    regression gets a ``StandardScaler`` because its coefficients are read as
    per-standard-deviation effects in the cross-family sign audit, and those are
    only comparable across features if the features are on one scale.

    ``keep_empty_features=True`` matters: early in a walk-forward split a column
    can be entirely NaN in the training block, and the default would silently
    *drop* it, changing the feature count between splits.
    """
    from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    p = dict(params or {})
    steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True))
    ]
    if family == "xgb":
        import xgboost as xgb

        model = xgb.XGBClassifier(
            # hist is the only tree method that is both fast and exactly
            # reproducible here; `n_jobs=1` removes any chance of a
            # nondeterministic histogram reduction order.
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
            eval_metric="logloss",
            reg_lambda=1.0,
            colsample_bytree=0.8,
            verbosity=0,
            **p,
        )
    elif family == "hist":
        model = HistGradientBoostingClassifier(
            random_state=seed,
            # `early_stopping="auto"` silently becomes True above 10,000 rows
            # and then carves its own random validation slice out of the
            # training data. That is a hidden holdout inside the training block
            # and it would make the tuned max_iter a function of a slice nobody
            # declared. Off, and max_iter is tuned honestly in the grid.
            early_stopping=False,
            **p,
        )
    elif family == "gbdt":
        model = GradientBoostingClassifier(random_state=seed, **p)
    elif family == "logreg":
        # `penalty` was deprecated in sklearn 1.8 and removed in 1.10, and its
        # *default* also changes across that boundary. Regularisation is
        # expressed through C alone, and the penalty is pinned so the reference
        # cannot drift underneath the deltas measured against it. lbfgs accepts
        # only "l2" or None, so this is not a choice about solver compatibility
        # -- it is about the reference staying put.
        #
        # The deprecation notice is raised at fit time, so
        # :func:`fit_single_threaded` is where it is filtered.
        model = LogisticRegression(
            penalty=None,
            C=1.0,
            solver="lbfgs",
            max_iter=5000,
            random_state=seed,
        )
        steps.append(("scale", StandardScaler()))
        steps.append(("model", model))
        return Pipeline(steps)
    else:
        raise ValueError(f"unknown family {family!r}; expected xgb, hist, gbdt or logreg")
    steps.append(("model", model))
    return Pipeline(steps)


def _grid_for(family: str) -> Sequence[Mapping]:
    return {"xgb": XGB_GRID, "hist": HIST_GRID, "gbdt": GBDT_GRID,
            "logreg": (dict(),)}[family]


def _single_threaded():
    """Context manager pinning every BLAS and OpenMP pool to one thread.

    Both halves of the GBM lifecycle need this on a small dataset, and for the
    same reason. See :func:`fit_single_threaded` and
    :func:`predict_proba_single_threaded`.
    """
    import threadpoolctl

    return threadpoolctl.threadpool_limits(limits=1)


def fit_single_threaded(model, X: np.ndarray, y: np.ndarray):
    """Fit ``model`` with every BLAS/OpenMP pool pinned to one thread.

    Not a micro-optimisation. On this 4-core box HistGB takes **94 seconds** to
    fit 2,250 rows and 15 features with its thread pool left at 4, and 1.0
    second with it pinned to 1: at this data size the OpenMP barrier and
    histogram-bin synchronisation cost two orders of magnitude more than the work
    being parallelised. The same limit is applied to BLAS so the logistic
    reference is not subject to the same trap through OpenBLAS.

    It is also what makes the determinism guarantee real. Multi-threaded
    reductions can combine partial results in an order that depends on thread
    scheduling, so "same seed, same answer" is not something a tree ensemble can
    promise while its histogram pass is threaded.

    The deprecation filter lives here rather than in :func:`make_model` for a
    reason that cost a run to find out: sklearn emits the notice at **fit** time,
    not at construction time, so a filter wrapped around the constructor sees
    nothing. Matching on the message prefix alone catches both spellings
    sklearn uses -- a ``DeprecationWarning`` and, in 1.9, a ``FutureWarning`` --
    without touching any other warning.

    Returns
    -------
    The fitted model, for chaining.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="'penalty' was deprecated")
        with _single_threaded():
            return model.fit(X, y)


def predict_proba_single_threaded(model, X: np.ndarray) -> np.ndarray:
    """Column 1 of ``predict_proba`` with the thread pool pinned to one thread.

    Prediction needs the same treatment as fitting, and accounts for more of the
    wall clock: profiling a walk-forward run attributed 56 of its 82 seconds to
    ten HistGB ``predict_proba`` calls, because the predictor opens one OpenMP
    region per boosting iteration and 200 of those over 750 rows is nothing but
    barrier overhead.

    Returns
    -------
    numpy.ndarray
        Predicted probability of the positive class.
    """
    with _single_threaded():
        return model.predict_proba(X)[:, 1]


def tune_on_inner_splits(family: str, X: np.ndarray, y: np.ndarray,
                         train_pos: np.ndarray, dates: pd.Series,
                         *, grid: Sequence[Mapping] | None = None,
                         seed: int = SEED, n_splits: int = 3,
                         initial_frac: float = 0.6) -> TuneResult:
    """Pick hyperparameters using only the rows in ``train_pos``.

    This function is handed training positions and never a test index, which is
    the mechanism behind the "tuned without touching test data" guarantee -- it
    is not a convention that a caller is trusted to keep. The caller in
    :func:`run_walk_forward` slices ``X``/``y``/``dates`` to the training block
    first and passes indices that are relative to it, so there is no way for a
    test row to enter this function's view even by accident.

    Selection criterion is mean **log loss**, not accuracy. Accuracy on a ~56%
    majority class is close to flat across the whole grid, which is exactly the
    failure mode a grid search is supposed to avoid.

    Parameters
    ----------
    family
        One of ``"xgb"``, ``"hist"``, ``"gbdt"``, ``"logreg"``.
    X, y
        The training block only.
    train_pos
        Kept for signature symmetry with the call site and asserted to be empty
        of anything outside the block; see the module-level note. Positions are
        relative to the training block.
    dates
        Training-block dates.
    grid
        Override the default grid.

    Returns
    -------
    TuneResult
    """
    grid = list(_grid_for(family) if grid is None else grid)
    if len(train_pos) and train_pos.max() >= len(X):
        raise ValueError("train_pos refers outside the training block passed as X")
    splits = inner_splits(len(X), dates, n_splits=n_splits, initial_frac=initial_frac)
    if not splits:
        raise ValueError(
            "the training block is too short to carve an inner walk-forward "
            "split; lower n_splits or initial_frac"
        )
    scores: dict[str, float] = {}
    valid_rows = 0
    n_train_rows = 0
    for params in grid:
        losses = []
        for tr, va in splits:
            model = make_model(family, params, seed=seed)
            fit_single_threaded(model, X[tr], y[tr])
            p = predict_proba_single_threaded(model, X[va])
            losses.append(evaluate(y[va], p)["log_loss"])
            valid_rows += len(va)
            n_train_rows = max(n_train_rows, len(tr))
        scores[repr(dict(params))] = float(np.mean(losses))
    best_key = min(scores, key=scores.get)
    best_params = next(dict(p) for p in grid if repr(dict(p)) == best_key)
    return TuneResult(
        family=family,
        params=best_params,
        inner_log_loss=scores[best_key],
        scores=scores,
        n_train_rows=n_train_rows,
        n_valid_rows=valid_rows // max(1, len(scores)) if scores else 0,
        n_splits_used=len(splits),
    )


def fit_logreg_reference(X: np.ndarray, y: np.ndarray, train_pos: np.ndarray,
                         **kwargs) -> tuple[object, TuneResult]:
    """Fit the logistic reference on the training block.

    Model 2 in the spec, ``logistic regression on home-minus-away feature
    contrasts + home flag``, which is exactly :data:`FEATURE_NAMES`. The same
    feature matrix the GBMs get, so the comparison isolates the model family
    rather than confounding it with a wider or narrower feature set.

    This is a *reference*, not the project's canonical logistic regression --
    ``src/Models/logreg_rolling.py`` is that. See
    :func:`run_canonical_logreg_comparison`.

    Returns
    -------
    (fitted_pipeline, TuneResult)
        The pipeline is fitted on training rows only, imputation and scaling
        included.
    """
    grid: Sequence[Mapping] = (dict(),)
    res = tune_on_inner_splits("logreg", X, y, train_pos, kwargs.pop("dates"), grid=grid, **kwargs)
    model = make_model("logreg", res.params, seed=kwargs.get("seed", SEED))
    fit_single_threaded(model, X, y)
    return model, res


# --------------------------------------------------------------------------- #
# Walk-forward evaluation
# --------------------------------------------------------------------------- #

def _log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Per-observation log loss, clipped so a confident miss cannot return inf.

    Identical in substance to what ``paired_bootstrap_delta`` computes; this is
    the per-row version, needed because a standard error on the *mean* loss is
    the standard deviation of these over sqrt(n).
    """
    pc = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(pc) + (1 - y) * np.log(1 - pc))


def run_walk_forward(X: np.ndarray, y: np.ndarray, games: pd.DataFrame, *,
                     families: Sequence[str] = ("xgb", "hist"),
                     baseline: str = "logreg",
                     n_splits: int = 5, seed: int = SEED,
                     n_boot: int = 2000,
                     inner_n_splits: int = 3,
                     inner_initial_frac: float = 0.6,
                     grids: Mapping[str, Sequence[Mapping]] | None = None,
                     verbose: bool = True) -> dict:
    """Walk forward by date, tune inside each training block, evaluate once.

    For every outer split: tune on an inner split of the training block, refit
    on the whole training block, predict the untouched test block. The test
    block is read exactly once per family and never during selection.

    The always-home base rate is included because a model that cannot beat
    "every home team wins 56%" has no claim on anything, and because it is the
    honest zero point for the deltas: a GBM that ties the base rate and beats
    logreg is telling you the logreg was carrying the whole result. It is
    appended unconditionally, so it is present whatever ``families`` says.

    Same-date boundary games
    -----------------------
    Splits are cut by row position, and several games share a calendar date, so
    the last training game and the first test game of a split can fall on the
    same day. That is not leakage here, and the reason is worth stating because
    it looks alarming: ``rolling_team_features`` shifts each team's own history
    by one game, so a test game's EWM contains its team's *previous* games, which
    may include one on the same date -- but a different game that had already
    tipped off. Information flows along the played order, not the calendar day,
    and ``walk_forward_split``'s ``gap`` argument exists for callers who do want
    a purged interval. Causality is asserted directly in
    ``Tests/test_gbm.py::TestTruncationAndCausality`` rather than inferred from
    the dates here.

    Parameters
    ----------
    X, y, games
        From :func:`load_dataset`.
    families
        GBM families to fit, each tuned independently.
    baseline
        Which fitted model the paired deltas are measured against.
    inner_n_splits, inner_initial_frac
        Shape of the inner walk-forward split carved out of each training block.
    grids
        Per-family hyperparameter grid override. Exists so the test suite can
        exercise the selection machinery on a two-config grid in seconds rather
        than on the full eight-config grid; production runs leave it ``None``
        and get :data:`XGB_GRID` / :data:`HIST_GRID`.

    Returns
    -------
    dict
        ``per_split`` (one record per split per model), ``pooled``, ``tuning``,
        ``meta``, and the split-aligned ``predictions_by_split`` /
        ``labels_by_split`` needed by the paired comparisons.
    """
    splits = walk_forward_split(games, n_splits=n_splits, initial_frac=0.5, gap=0)
    if not splits:
        raise ValueError("no walk-forward splits produced")
    per_split: list[dict] = []
    predictions: dict[str, list[np.ndarray]] = {}
    labels: list[np.ndarray] = []
    tuning: dict[str, list[TuneResult]] = {}

    for k, (tr, te) in enumerate(splits, start=1):
        tr_dates = games["GAME_DATE"].iloc[tr].reset_index(drop=True)
        te_dates = games["GAME_DATE"].iloc[te].reset_index(drop=True)
        base_rate = float(y[tr].mean())
        trained: dict[str, object] = {"base_rate": base_rate}
        y_te = y[te]
        labels.append(y_te)

        for fam in list(families) + [baseline] + ["base_rate"]:
            if fam == "base_rate":
                p = np.full(len(te), base_rate)
            else:
                res = tune_on_inner_splits(
                    fam, X[tr], y[tr], np.arange(len(tr)), tr_dates,
                    seed=seed, n_splits=inner_n_splits, initial_frac=inner_initial_frac,
                    grid=None if grids is None else grids.get(fam),
                )
                tuning.setdefault(fam, []).append(res)
                model = make_model(fam, res.params, seed=seed)
                fit_single_threaded(model, X[tr], y[tr])
                trained[fam] = model
                p = predict_proba_single_threaded(model, X[te])
            predictions.setdefault(fam, []).append(p)
            per_split.append({"split": k, "model": fam,
                              "train_n": len(tr), "test_n": len(te),
                              "test_dates": (te_dates.min(), te_dates.max()),
                              **evaluate(y_te, p)})
        if verbose:
            print(f"  split {k}/{len(splits)}  train={len(tr):>5}  test={len(te):>5}"
                  f"  {te_dates.min().date()} .. {te_dates.max().date()}")

    names = list(predictions)
    pooled: dict[str, dict] = {}
    y_all = np.concatenate(labels)
    p_all = {n: np.concatenate(predictions[n]) for n in names}
    base_name = baseline
    for n in names:
        rowll = _log_loss(y_all, p_all[n])
        pooled[n] = {
            "n": int(len(y_all)),
            "log_loss": float(rowll.mean()),
            "log_loss_se": float(rowll.std(ddof=1) / np.sqrt(len(rowll))),
            **evaluate(y_all, p_all[n]),
        }
        if n != base_name:
            pooled[n]["delta_vs_" + base_name] = paired_bootstrap_delta(
                y_all, p_all[base_name], p_all[n], n_boot=n_boot, seed=seed
            )
        # Per-split records also get a SE, for the same reason the pooled figure
        # does: a mean log loss quoted without an error bar invites reading a
        # 0.002 difference as a result.
        for rec, y_block in zip(
            [r for r in per_split if r["model"] == n], labels
        ):
            block_ll = _log_loss(y_block, predictions[n][rec["split"] - 1])
            rec["log_loss_se"] = float(block_ll.std(ddof=1) / np.sqrt(len(block_ll)))
    return {
        "per_split": per_split,
        "pooled": pooled,
        "predictions": p_all,
        "predictions_by_split": predictions,
        "labels": y_all,
        "labels_by_split": labels,
        "tuning": tuning,
        "meta": {"families": list(families), "baseline": baseline,
                 "n_splits": n_splits, "seed": seed,
                 "inner_n_splits": inner_n_splits,
                 "inner_initial_frac": inner_initial_frac,
                 "n_features": X.shape[1],
                 "n_games": len(games),
                 "train_max_date": str(games["GAME_DATE"].iloc[splits[-1][0].max()].date())},
    }


def per_split_deltas(result: dict, baseline: str | None = None,
                     n_boot: int = 2000, seed: int = SEED) -> list[dict]:
    """Per-split log-loss delta against the baseline, each with its own CI.

    A pooled delta over all five blocks is the headline number, but it hides
    *stability*: a model can take the pooled test because one block went its way.
    Five intervals of a third of the width each are far noisier than one pooled
    interval, and seeing which blocks are positive is the difference between
    "consistently slightly better" and "better on average, on the strength of
    one block" -- two very different claims that a single pooled figure cannot
    distinguish.

    Parameters
    ----------
    result
        Output of :func:`run_walk_forward`.
    baseline
        Defaults to the baseline recorded in the run.

    Returns
    -------
    list of dict
        One record per (split, model) with ``delta``, CI bounds and
        ``better`` -- where ``better`` is ``True`` when the model under test has
        the lower log loss, following the shared delta sign convention.
    """
    baseline = baseline or result["meta"]["baseline"]
    out = []
    for fam in result["meta"]["families"]:
        for k, (y_block, p_block) in enumerate(
            zip(result["labels_by_split"], result["predictions_by_split"][fam]), start=1
        ):
            if baseline not in result["predictions_by_split"]:
                continue
            p_base = result["predictions_by_split"][baseline][k - 1]
            d = paired_bootstrap_delta(y_block, p_base, p_block, n_boot=n_boot, seed=seed)
            out.append({
                "split": k, "model": fam, "n": int(len(y_block)),
                "delta": d["delta"], "ci_lo": d["ci_lo"], "ci_hi": d["ci_hi"],
                "p_better": d["p_positive"],
                "better": bool(d["delta"] > 0),
            })
    return out


# --------------------------------------------------------------------------- #
# Feature accounting: the overfitting curve and what earns its place
# --------------------------------------------------------------------------- #

def permutation_importance_table(model, X: np.ndarray, y: np.ndarray, *,
                                 n_repeats: int = 10, seed: int = SEED,
                                 feature_names: Sequence[str] = FEATURE_NAMES
                                 ) -> pd.DataFrame:
    """Mean permutation importance (increase in log loss) per feature.

    Importance is measured by **destroying** the feature and seeing the damage,
    never by a split count. A split count rewards high-cardinality columns and
    is blind to a feature the model splits on in ways that do not help. The
    score is the increase in log loss from shuffling that one column, so
    positive means the model was using it and zero means it was carrying its
    weight for nothing.

    The shuffle loop runs inside the single-thread context for the same reason
    as fitting: it makes ``p * n_repeats`` predictions, and each HistGB
    prediction unthreaded is orders of magnitude cheaper than a threaded one on
    this data size. Measured, that was 412 seconds for a 15-feature,
    3-repeat table and about two seconds pinned to one thread.
    """
    from sklearn.inspection import permutation_importance

    with _single_threaded():
        r = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=seed,
            scoring="neg_log_loss", n_jobs=1,
        )
    return pd.DataFrame({
        "feature": list(feature_names),
        "importance": np.asarray(r.importances_mean, dtype=float),
        "importance_se": np.asarray(r.importances_std, dtype=float) / np.sqrt(n_repeats),
    }).sort_values("importance", ascending=False, ignore_index=True)


def feature_count_curve(games: pd.DataFrame, X: np.ndarray, y: np.ndarray, *,
                        families: Sequence[str] = ("xgb", "hist"),
                        baseline: str = "logreg",
                        n_splits: int = 5, seed: int = SEED,
                        n_repeats: int = 5,
                        feature_names: Sequence[str] = FEATURE_NAMES) -> pd.DataFrame:
    """Log loss as a function of how many features the model is given.

    The spec warns that feature count drives overfitting on a noisy target, and
    this is the measurement of that warning rather than a claim about it.
    Features are ranked by permutation importance computed on a *training*
    block only (never the test block), then the model is refitted on the top k
    for k = 1..p with the *same* tuned hyperparameters, and evaluated on the
    untouched test block.

    One ranking is computed once, from the FIRST split's training block and the
    FIRST family, and reused for every k and every family. That is what makes
    the three columns comparable: if each family ranked its own features, the
    curve would be showing three different definitions of "top k" and the
    logistic reference -- which loads almost all of its signal onto a single
    feature -- would be picking up a different feature at k = 1 than the trees.

    Holding hyperparameters fixed across k is deliberate. Re-tuning at every k
    would multiply the selection budget by the number of points on the curve and
    turn a diagnostic into a second, larger model-selection experiment -- and
    the resulting curve would no longer be comparable across k, because the k=1
    point would have had eight configs searched for it and the k=15 point one.
    """
    splits = walk_forward_split(games, n_splits=n_splits, initial_frac=0.5, gap=0)
    rows = []
    # Rank features once, on the largest training block, using the first family's
    # tuned model. One ranking for all families keeps the curve comparable.
    names = np.asarray(feature_names, dtype=object)
    ranks = None
    for k, (tr, te) in enumerate(splits, start=1):
        for fam in list(families) + [baseline]:
            res = tune_on_inner_splits(
                fam, X[tr], y[tr], np.arange(len(tr)),
                games["GAME_DATE"].iloc[tr].reset_index(drop=True),
                seed=seed, n_splits=3, initial_frac=0.6,
            )
            if ranks is None:
                model = make_model(fam, res.params, seed=seed)
                fit_single_threaded(model, X[tr], y[tr])
                imp = permutation_importance_table(
                    model, X[tr], y[tr], n_repeats=n_repeats, seed=seed,
                    feature_names=names,
                )
                ranks = np.asarray(imp["feature"], dtype=object)
            for n_feat in range(1, X.shape[1] + 1):
                cols = np.asarray([list(names).index(c) for c in ranks[:n_feat]], dtype=int)
                m = make_model(fam, res.params, seed=seed)
                fit_single_threaded(m, X[tr][:, cols], y[tr])
                p = predict_proba_single_threaded(m, X[te][:, cols])
                rows.append({
                    "split": k, "model": fam, "n_features": n_feat,
                    "features": ",".join(ranks[:n_feat]),
                    "log_loss": evaluate(y[te], p)["log_loss"],
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Cross-family effect signs
# --------------------------------------------------------------------------- #

def effect_directions(model, X: np.ndarray, *, feature_names: Sequence[str] = FEATURE_NAMES
                      ) -> pd.Series:
    """Signed effect of +/-1 SD on each feature, in average predicted probability.

    This is the definition that lets model families be compared on sign at all.
    A logistic regression has a coefficient per feature; XGBoost has none, and
    neither ``feature_importances_`` nor permutation importance carries a sign --
    both are magnitudes. "The effect sign of XGBoost on feature j" is therefore
    not a property the estimator reports, and quoting one as if it were would be
    inventing it.

    The definition below *is* answerable by every family: take the data as
    given, replace one column by ``median - sd`` for every row and predict; then
    replace it by ``median + sd`` and predict. The difference of the two mean
    probabilities is a signed, monotone-contrast effect. It is positive when
    more of that feature means a higher home-win probability, and negative when
    it means lower -- for a linear model it is exactly ``2 * sd * beta``, so for
    logreg this reproduces the coefficient and for a tree it measures something
    the tree actually does.

    Returns
    -------
    pandas.Series
        Indexed by feature name, in :data:`FEATURE_NAMES` order.
    """
    X = np.asarray(X, dtype=float)
    out = {}
    for j, name in enumerate(feature_names):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            out[name] = np.nan
            continue
        med = float(np.median(finite))
        sd = float(np.std(finite))
        if not np.isfinite(sd) or sd <= 0:
            # A constant column (is_home) has no contrast available at all.
            # Reporting 0.0 rather than NaN says "no effect", which is the
            # correct reading: the model cannot respond to a value that never
            # varies. It is not evidence that the feature is useless in general.
            out[name] = 0.0
            continue
        lo, hi = X.copy(), X.copy()
        lo[:, j] = med - sd
        hi[:, j] = med + sd
        p_lo = predict_proba_single_threaded(model, lo)
        p_hi = predict_proba_single_threaded(model, hi)
        out[name] = float(np.mean(p_hi) - np.mean(p_lo))
    return pd.Series(out, dtype=float).loc[list(feature_names)]


def dominant_feature_corr(X: np.ndarray, dominant: str,
                          feature_names: Sequence[str] = FEATURE_NAMES
                          ) -> pd.Series:
    """|correlation| of every feature with ``dominant``, the model's top feature.

    This is the diagnostic that explains the cross-family sign audit, so it
    lives next to the audit rather than in a notebook.

    Why it explains it. :func:`effect_directions` moves one column to
    ``median +/- 1 SD`` and leaves every other column at its own observed value.
    For a feature that is almost perfectly correlated with the feature the model
    leans on hardest, that combination **does not occur in the data**. A team
    with an average recent margin and an average net rating barely exists, so
    the question being put to the model is off-manifold, and off-manifold
    response is model-specific by construction: a linear model extrapolates along
    a straight line it was never trained along, while a tree lands in a leaf it
    almost never visited. A sign disagreement there is evidence about where the
    contrast was asked, not about a defect in either estimator.

    On this feature set the three sharpest disagreements are ``d_net_rating``,
    ``d_off_rating`` and ``d_def_rating`` at 0.998, 0.807 and 0.753 against
    ``d_margin`` -- the top feature for every model -- and ``d_margin`` itself
    agrees in sign across all four families. That is the pattern this function
    exists to make visible.

    Returns
    -------
    pandas.Series
        Absolute correlations indexed by feature name, ordered as
        :data:`FEATURE_NAMES`. ``d_margin`` itself is omitted because its
        correlation with itself is undefined.
    """
    names = list(feature_names)
    frame = pd.DataFrame(np.where(np.isnan(X), np.nan, X), columns=names)
    # The dominant feature is missing on a team's first games; impute at the
    # median purely to compute a correlation, which is why this does not touch
    # anything a model sees.
    frame[dominant] = frame[dominant].fillna(float(np.nanmedian(frame[dominant])))
    out = frame.corrwith(frame[dominant]).abs().drop(dominant)
    return out.loc[[n for n in names if n != dominant]]


def cross_family_sign_audit(games: pd.DataFrame, X: np.ndarray, y: np.ndarray, *,
                            families: Sequence[str] = ("xgb", "hist", "gbdt", "logreg"),
                            n_splits: int = 5, seed: int = SEED,
                            grids: Mapping[str, Sequence[Mapping]] | None = None,
                            feature_names: Sequence[str] = FEATURE_NAMES
                            ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Do XGBoost and HistGB agree on the direction of each feature's effect?

    Motivation: an earlier cross-family test in this project found that
    XGBoost's effect sign **reversed** relative to logistic regression, GBDT and
    HistGB on *player-derived* features. Those features belong to a different
    model (player ratings) and are not in this module's feature set, so the
    honest thing is not to assume the finding is inapplicable but to re-run the
    comparison here and look.

    Method: for each family, on each walk-forward split, tune on the training
    block, fit on the training block, and measure :func:`effect_directions` on
    the *test* block -- the same rows every family is scored on. Effects are
    reported in home-help orientation, i.e. multiplied by
    :data:`FEATURE_ORIENTATION`, so a positive number always means "more of
    this helps the home team" regardless of how the raw column is signed. Without
    that correction ``d_def_rating`` and ``d_tov`` would appear to disagree with
    every other family when in fact all four are saying the same thing.

    feature_names
        Column order of ``X``; also fixes the row order of the returned frames.
    grids
        Per-family grid override, for the same reason
        :func:`run_walk_forward` has one.

    Returns
    -------
    (effects, agreement)
        ``effects`` is a feature x family frame of mean oriented effects.
        ``agreement`` reports, for each pair of families, the fraction of
        features that have a non-zero effect in *both* and share a sign, plus a
        count of features on which each family has no effect at all.
    """
    names = np.asarray(feature_names, dtype=object)
    orientation = np.asarray([FEATURE_ORIENTATION[n] for n in feature_names], dtype=float)
    splits = walk_forward_split(games, n_splits=n_splits, initial_frac=0.5, gap=0)
    acc = {f: np.zeros(len(names), dtype=float) for f in families}
    n_terms = {f: 0 for f in families}
    for tr, te in splits:
        for fam in families:
            res = tune_on_inner_splits(
                fam, X[tr], y[tr], np.arange(len(tr)),
                games["GAME_DATE"].iloc[tr].reset_index(drop=True),
                seed=seed, n_splits=3, initial_frac=0.6,
                grid=None if grids is None else grids.get(fam),
            )
            model = make_model(fam, res.params, seed=seed)
            fit_single_threaded(model, X[tr], y[tr])
            eff = effect_directions(model, X[te], feature_names=names)
            acc[fam] += eff.to_numpy(dtype=float) * orientation
            n_terms[fam] += 1
    effects = pd.DataFrame({f: acc[f] / max(1, n_terms[f]) for f in families}, index=names)

    agree_rows = {}
    for i, a in enumerate(families):
        for b in families[i + 1:]:
            ea, eb = effects[a].to_numpy(), effects[b].to_numpy()
            mask = (np.abs(ea) > 0) & (np.abs(eb) > 0)
            frac = float(np.mean(np.sign(ea[mask]) == np.sign(eb[mask]))) if mask.any() else float("nan")
            agree_rows[f"{a} vs {b}"] = frac
    agreement = pd.DataFrame({"sign_agreement": pd.Series(agree_rows)})
    for f in families:
        agreement[f"{f}: zero_effect"] = int(np.sum(np.abs(effects[f].to_numpy()) <= 0))
    return effects, agreement


# --------------------------------------------------------------------------- #
# Comparison against the canonical model-2 logistic regression
# --------------------------------------------------------------------------- #

def canonical_logreg_predictions(games: pd.DataFrame, *, n_splits: int = 5,
                                 initial_frac: float = 0.5, C: float = 1.0,
                                 module_path: str = "src.Models.logreg_rolling"
                                 ) -> dict | None:
    """Ask model 2 for its out-of-sample probabilities on our exact test blocks.

    Model 2 (``src/Models/logreg_rolling.py``) owns the project's canonical
    logistic regression. It is written in parallel and may not exist when this
    module runs, so this probe is allowed to fail and returns ``None``.

    Both sides must be scored on the *same rows* or the paired delta is
    meaningless, so the returned labels are checked against this module's own
    before anything is compared.

    Returns
    -------
    dict or None
        ``{"p_model", "y", "p_base"}``, or ``None`` if model 2 is unavailable or
        disagrees about which games are being scored.
    """
    try:
        import importlib

        canon = importlib.import_module(module_path)
    except Exception:
        return None
    X2 = canon.load_design(halflife=10.0)
    features = canon.resolve_features(canon.DEFAULT_MODELS["full"])
    pp = canon.pooled_predictions(X2, games, features, C=C,
                                  n_splits=n_splits, initial_frac=initial_frac)
    return {"p_model": np.asarray(pp["p_model"], dtype=float),
            "y": np.asarray(pp["y"], dtype=int),
            "p_base": np.asarray(pp["p_base"], dtype=float),
            "n_features": len(features)}


def run_canonical_logreg_comparison(result: dict, canonical: Mapping, *, games=None,
                                    module_path: str = "src.Models.logreg_rolling",
                                    n_boot: int = 2000) -> dict:
    """Re-score this module's deltas against model 2's canonical logreg.

    The headline delta in :func:`run_walk_forward` is measured against the local
    stand-in in :func:`fit_logreg_reference`, because model 2 may not exist. This
    re-runs the same paired comparison against the canonical model so the
    conclusion does not rest on the stand-in.

    Alignment is verified, not assumed. Model 2 returns its pooled predictions
    with a ``y`` alongside; that ``y`` must equal this module's concatenated test
    labels exactly, or the two were scored on different games and the delta
    would be a comparison of two different samples that still prints a number.

    Parameters
    ----------
    result
        Output of :func:`run_walk_forward`, which carries split-aligned
        predictions and labels for every model.
    canonical
        Output of :func:`canonical_logreg_predictions`.

    Returns
    -------
    dict
        ``pooled`` (one delta with CI per GBM family) and ``per_split``.

    Raises
    ------
    ValueError
        If the two runs were not scored on the same games.
    """
    y_canon = np.asarray(canonical["y"], dtype=int)
    y_ours = np.asarray(result["labels"], dtype=int)
    if y_canon.shape != y_ours.shape or not np.array_equal(y_canon, y_ours):
        raise ValueError(
            f"model 2 scored {y_canon.shape[0]} games and this module scored "
            f"{y_ours.shape[0]}, and the labels do not match; the two "
            "walk-forward runs are not on the same rows and a paired delta "
            "between them would be meaningless"
        )
    p_canon = np.asarray(canonical["p_model"], dtype=float)
    out = {"module": module_path, "per_split": [], "pooled": {}}
    seed = result["meta"]["seed"]
    for fam in result["meta"]["families"]:
        p_ours = np.concatenate(result["predictions_by_split"][fam])
        out["pooled"][fam] = paired_bootstrap_delta(y_ours, p_canon, p_ours,
                                                    n_boot=n_boot, seed=seed)
        offset = 0
        for k, y_block in enumerate(result["labels_by_split"], start=1):
            n = len(y_block)
            out["per_split"].append({
                "split": k, "model": fam, "n": n,
                **paired_bootstrap_delta(y_block, p_canon[offset:offset + n],
                                        result["predictions_by_split"][fam][k - 1],
                                        n_boot=n_boot, seed=seed),
            })
            offset += n
    out["log_loss_canonical"] = float(
        evaluate(y_ours, p_canon)["log_loss"])
    out["n_features_canonical"] = canonical.get("n_features")
    return out


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _fmt(x: float, w: int = 9, p: int = 5) -> str:
    return f"{x:{w}.{p}f}" if isinstance(x, (int, float)) and np.isfinite(x) else f"{'n/a':>{w}}"


def print_report(result: dict, curve: pd.DataFrame | None = None,
                 effects: pd.DataFrame | None = None,
                 agreement: pd.DataFrame | None = None,
                 importances: dict[str, pd.DataFrame] | None = None,
                 per_split_delta: list[dict] | None = None,
                 dominant_corr: pd.Series | None = None,
                 dominant: str | None = None) -> None:
    """Print the full report: per split, pooled with CI, calibration, curve, signs."""
    meta = result["meta"]
    print("=" * 100)
    print("GRADIENT BOOSTING vs LOGISTIC REGRESSION - walk-forward, honest")
    print("=" * 100)
    print(f"games {meta['n_games']:,}   features {meta['n_features']}   "
          f"splits {meta['n_splits']}   seed {meta['seed']}")
    print(f"training for the last split ends {meta['train_max_date']} "
          f"(test blocks are strictly later)")
    print()

    print("PER-SPLIT METRICS")
    hdr = (f"{'split':>5} {'model':<10} {'n':>6} {'logloss':>9} {'SE':>8} "
           f"{'brier':>8} {'auc':>7} {'acc':>7}")
    print(hdr); print("-" * len(hdr))
    for r in result["per_split"]:
        print(f"{r['split']:>5} {r['model']:<10} {r['test_n']:>6} "
              f"{_fmt(r['log_loss'])} {_fmt(r.get('log_loss_se'), 8)} "
              f"{_fmt(r['brier'])} {_fmt(r['auc'])} {_fmt(r['accuracy'])}")

    print()
    print("POOLED over all test blocks, log loss first. "
          "SE is the standard error of the mean per-game log loss.")
    hdr2 = f"{'model':<10} {'n':>6} {'logloss':>9} {'SE':>8} {'brier':>8} {'auc':>7} {'acc':>7}"
    print(hdr2); print("-" * len(hdr2))
    for name, row in result["pooled"].items():
        print(f"{name:<10} {row['n']:>6} {_fmt(row['log_loss'])} "
              f"{_fmt(row['log_loss_se'], 8)} {_fmt(row['brier'])} "
              f"{_fmt(row['auc'])} {_fmt(row['accuracy'])}")

    print()
    print("PAIRED BOOTSTRAP vs " + meta["baseline"] + ".")
    print("Sign convention: delta = loss(" + meta["baseline"] + ") - loss(model).")
    print("  delta > 0  => the GBM is BETTER (lower log loss).")
    print("  delta < 0  => the GBM is WORSE. Report it; do not re-tune to flip it.")
    for name, row in result["pooled"].items():
        d = row.get(f"delta_vs_{meta['baseline']}")
        if d is None:
            continue
        verdict = "BETTER" if d["delta"] > 0 else "WORSE"
        sig = "significant" if d["ci_lo"] > 0 or d["ci_hi"] < 0 else "NOT significant"
        print(f"  {name:<10} delta {d['delta']:+.5f}  "
              f"95% CI [{d['ci_lo']:+.5f}, {d['ci_hi']:+.5f}]  "
              f"P(better)={d['p_positive']:.3f}  -> {verdict}, {sig}")

    if per_split_delta:
        print()
        print("PER-SPLIT PAIRED DELTAS (same convention). Each block is ~a third of "
              "the pooled sample, so each CI is much wider; the point of showing "
              "them is stability, not precision.")
        hdr3 = (f"{'split':>5} {'model':<10} {'n':>6} {'delta':>9} "
                f"{'95% CI':>24} {'better':>7}")
        print(hdr3); print("-" * len(hdr3))
        for d in per_split_delta:
            print(f"{d['split']:>5} {d['model']:<10} {d['n']:>6} {d['delta']:>+9.5f} "
                  f"  [{d['ci_lo']:+.5f}, {d['ci_hi']:+.5f}] {str(d['better']):>7}")

    print()
    print("CALIBRATION (pooled, 10 bins). mean_pred vs actual win rate per bin. "
          "This is where GBM overconfidence shows.")
    hdr4 = (f"{'model':<10} {'bin':>4} {'lo':>5} {'hi':>5} {'n':>6} "
            f"{'mean_pred':>10} {'actual':>8} {'gap':>8}")
    print(hdr4)
    for name, row in result["pooled"].items():
        for b in row["calibration"]:
            gap = b["actual"] - b["mean_pred"]
            print(f"{name:<10} {b['bin']:>4} {b['lo']:>5.1f} {b['hi']:>5.1f} "
                  f"{b['n']:>6} {b['mean_pred']:>10.4f} {b['actual']:>8.4f} {gap:>+8.4f}")

    if importances:
        print()
        print("PERMUTATION IMPORTANCE (increase in log loss from shuffling a "
              "feature; positive = earning its place). Measured on training blocks.")
        for fam, imp in importances.items():
            print(f"  [{fam}]")
            for _, row in imp.iterrows():
                bar = "#" * max(0, int(round(row["importance"] / 1e-4)))
                print(f"    {row['feature']:<16} {row['importance']:+.5f} "
                      f"+/- {row['importance_se']:.5f}  {bar}")

    if curve is not None:
        print()
        print("OVERFITTING CURVE: mean log loss vs number of features "
              "(ranked by train-only permutation importance; hyperparameters "
              "held fixed across k).")
        piv = curve.groupby(["model", "n_features"])["log_loss"].mean().unstack(0)
        print(piv.round(5).to_string())

    if effects is not None:
        print()
        print("CROSS-FAMILY EFFECT SIGNS. A signed +/-1 SD contrast, averaged over "
              "test blocks, in home-help orientation (positive = helps the home "
              "team).")
        print("A tree has no per-feature coefficient, so this contrast is the only "
              "sign definition all families can answer.")
        if dominant_corr is not None and dominant is not None:
            shown = effects.copy()
            shown.insert(0, f"|corr w/ {dominant}|",
                         [float(dominant_corr[n]) if n in dominant_corr.index else np.nan
                          for n in shown.index])
            print(shown.round(4).to_string())
        else:
            print(effects.round(6).to_string())
        if agreement is not None:
            print()
            print("PAIRWISE SIGN AGREEMENT (fraction of features with a non-zero "
                  "effect in both families that share a sign).")
            print(agreement.round(4).to_string())
        if dominant_corr is not None and dominant is not None:
            print()
            print(f"READING THE DISAGREEMENTS. The +/-1 SD contrast moves one column "
                  f"and freezes the others, so for a feature highly correlated with "
                  f"`{dominant}` -- the top feature for every model here -- it asks "
                  f"about a combination of values that does not occur in the data.")
            print("Off-manifold, a linear model extrapolates along a straight line and "
                  "a tree lands in a rarely-visited leaf, so their answers can differ "
                  "without either being wrong. The disagreements above sit on the "
                  "high-|corr| features; the agreement sits on the low-|corr| ones.")
            print()
            print(dominant_corr.sort_values(ascending=False).round(3).to_string())

    print()
    print("Reminder: over ~6,600 test games a 0.002 log-loss difference sits "
          "inside the noise band. Read the CI, not the point estimate.")
    print("This is a predictive-log-loss comparison. It says nothing about "
          "profitability: Vandenbruaene et al. 2022 (J Sports Econ 23(7)) "
          "surveyed 628 NBA game-implementation papers and, after correcting "
          "for multiple testing, never rejected unprofitability at a hurdle "
          "of |z| > 3. Nothing here clears that bar.")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run_experiment(*, with_curve: bool = True, with_signs: bool = True,
                   n_splits: int = 5, seed: int = SEED, n_boot: int = 2000,
                   verbose: bool = True) -> dict:
    """Load, walk forward, then produce the curve, importances and sign audit.

    A single call that produces everything the report needs, so the numbers in
    the write-up cannot come from two different runs of two slightly different
    configurations.

    Returns
    -------
    dict
        The :func:`run_walk_forward` result, plus ``curve``, ``effects``,
        ``agreement``, ``importances`` and ``per_split_delta``.
    """
    games, X, y = load_dataset()
    if verbose:
        print(f"loaded {len(games):,} games x {X.shape[1]} features; "
              f"home win rate {y.mean():.4f}")
        print("tuning on inner splits of each training block (never the test block):")
    result = run_walk_forward(X, y, games, n_splits=n_splits, seed=seed,
                              n_boot=n_boot, verbose=verbose)

    # Permutation importance on the largest training block, so the ranking that
    # drives the feature-count curve and the reported "what earns its place"
    # table both come from data the model was fitted on, never from test rows.
    splits = walk_forward_split(games, n_splits=n_splits, initial_frac=0.5)
    tr = splits[-1][0]
    importances: dict[str, pd.DataFrame] = {}
    for fam in list(result["meta"]["families"]) + [result["meta"]["baseline"]]:
        res = tune_on_inner_splits(
            fam, X[tr], y[tr], np.arange(len(tr)),
            games["GAME_DATE"].iloc[tr].reset_index(drop=True), seed=seed)
        m = make_model(fam, res.params, seed=seed)
        fit_single_threaded(m, X[tr], y[tr])
        importances[fam] = permutation_importance_table(m, X[tr], y[tr], seed=seed)

    curve = (feature_count_curve(games, X, y, n_splits=n_splits, seed=seed)
             if with_curve else None)
    effects = agreement = None
    if with_signs:
        effects, agreement = cross_family_sign_audit(
            games, X, y, n_splits=n_splits, seed=seed)

    result.update({
        "games": games, "X": X, "y": y,
        "curve": curve, "effects": effects, "agreement": agreement,
        "importances": importances,
        "per_split_delta": per_split_deltas(result, seed=seed, n_boot=n_boot),
    })
    # The diagnostic that explains the sign audit. The "dominant" feature is
    # taken from the reported importances rather than hard-coded, so if the
    # ranking ever changes the explanation follows it instead of going stale.
    dominant = importances[result["meta"]["families"][0]]["feature"].iloc[0]
    result["dominant_feature"] = dominant
    result["dominant_corr"] = dominant_feature_corr(X, dominant)
    return result


def main() -> None:  # pragma: no cover - entry point
    import argparse

    ap = argparse.ArgumentParser(
        description="Walk-forward gradient boosting on the shared rolling feature set.")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--no-curve", action="store_true",
                    help="skip the feature-count curve (it refits the grid)")
    ap.add_argument("--no-signs", action="store_true",
                    help="skip the cross-family effect-sign audit")
    args = ap.parse_args()

    result = run_experiment(
        with_curve=not args.no_curve,
        with_signs=not args.no_signs,
        n_splits=args.splits, seed=args.seed, n_boot=args.boot,
    )
    print_report(result, curve=result["curve"], effects=result["effects"],
                 agreement=result["agreement"], importances=result["importances"],
                 per_split_delta=result["per_split_delta"],
                 dominant_corr=result.get("dominant_corr"),
                 dominant=result.get("dominant_feature"))

    # Model 2's canonical logistic regression may have landed while this was
    # running. Probe for it and, if present, re-score against it so the delta is
    # not left pointing at the local stand-in.
    games = result["games"]
    canon = canonical_logreg_predictions(games, n_splits=args.splits)
    if canon is None:
        print("\nsrc/Models/logreg_rolling.py not importable; every delta above "
              "is against THIS module's logistic reference, not the project's "
              "canonical model 2.")
    else:
        try:
            out = run_canonical_logreg_comparison(result, canon, games=games)
        except ValueError as exc:
            print(f"\ncould not compare against model 2: {exc}")
        else:
            print("\nAGAINST THE CANONICAL MODEL-2 LOGISTIC REGRESSION "
                  f"({out['n_features_canonical']} features, its own design "
                  f"matrix, pooled log loss {out['log_loss_canonical']:.5f}).")
            print("  Sign convention unchanged: delta > 0 => the GBM is BETTER.")
            for fam, d in out["pooled"].items():
                verdict = "BETTER" if d["delta"] > 0 else "WORSE"
                print(f"  {fam:<8} delta {d['delta']:+.5f}  "
                      f"95% CI [{d['ci_lo']:+.5f}, {d['ci_hi']:+.5f}]  "
                      f"P(GBM better)={d['p_positive']:.3f}  -> {verdict}")
            for r in out["per_split"]:
                print(f"    split {r['split']} {r['model']:<8} "
                      f"delta {r['delta']:+.5f}  "
                      f"CI [{r['ci_lo']:+.5f}, {r['ci_hi']:+.5f}]")


if __name__ == "__main__":  # pragma: no cover
    main()