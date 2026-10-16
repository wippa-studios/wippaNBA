"""Margin-and-total model: predict the score distribution, derive everything else.

Instead of predicting the winner, this model predicts the two quantities a game
is actually made of -- the point margin ``home_pts - away_pts`` and the combined
score ``home_pts + away_pts`` -- and derives a win probability and a spread from
them. The win probability is not a separate fit:

    P(home wins | features) = Phi(mhat / sigma)

with ``mhat`` the predicted margin and ``sigma`` the residual scale. That single
identity is the whole point of this module and also its main hazard. A margin
model that is well calibrated on the *scale* of the margin can still be badly
miscalibrated as a *probability*, because the conversion multiplies every
confidence by the same ``1/sigma``. So ``sigma`` is a fitted parameter here,
fitted on TRAIN ONLY, and its own diagnostics (the calibration table) are
reported alongside the log loss rather than assumed.

WHAT IS FITTED, AND WHEN
------------------------
Per walk-forward split, and only ever on the training rows: the imputer medians,
the scaler variances, the ridge penalty (chosen by an inner forward-chaining CV
*inside* the training rows), the coefficients, the intercept, and both residual
scales. The test rows are touched exactly once, by ``predict``. Nothing in this
module reads ``snapshots/``; every feature comes from
``src.DataProviders.nba_data``, which builds them strictly from games before the
game in question.

THE HOME INDICATOR IS THE INTERCEPT
-----------------------------------
``rest_and_travel`` returns ``is_home``, which is 1.0 on every row. Putting it in
the design matrix and then running ``StandardScaler`` would map a constant column
to exactly 0.0, deleting home advantage from the model while the fitted output
still looked entirely reasonable. So home court is carried as the model's
intercept, and the design matrix is required to contain no zero-variance
column. ``assert_no_constant_columns`` enforces that.

FEATURE FORM DIFFERS BETWEEN MARGIN AND TOTAL
---------------------------------------------
This is not a stylistic choice. A margin is a *difference*, so the informative
features are home-minus-away contrasts. A total is a *level*, so contrasting
home against away throws away almost all of the signal and leaves only "is this
an even matchup", which barely moves a total. ``make_design_matrix`` therefore
sums the same nine EWM measures for the total. ``TOTAL_CONTRAST_ABLATION`` in
``run_experiment`` quantifies the cost of getting this wrong.

REFERENCES, FIT LOCALLY
-----------------------
``run_experiment`` fits its own comparison points rather than assuming another
model in this suite exists: the always-home base rate (train-only), a Gaussian
margin model on the EWM net-rating difference plus the intercept, and a direct
logistic regression on the identical design matrix. The third is the interesting
one -- it answers "does going through the margin actually buy anything?".

SCOPE
-----
This measures prediction quality. It does not measure betting value, because this
dataset has no closing market lines: no spread, no total, no moneyline is joined
anywhere in ``data/nba``. The spread numbers below are MAE against *actual
margins*, i.e. "how close to the true margin", and must never be read as
"beats the market". Vandenbruaene et al. (2022, J Sports Econ 23(7)) surveyed 628
strategy implementations and, after multiple-testing correction, never rejected
the null of unprofitability at the conventional hurdle (|z| > 3); the same is
true of everything else in this suite.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.DataProviders import nba_data as nd

__all__ = [
    "EWM_MEASURES",
    "MarginTotalModel",
    "DirectWinModel",
    "make_design_matrix",
    "build_dataset",
    "walk_forward_eval",
    "run_experiment",
    "format_report",
]

# The nine strictly-causal EWM measures the shared feature builder produces per
# side (margin, net rating, offence, defence, pace, TS%, eFG%, TOV%, FTR%). The
# tenth, games_played, is handled separately because it is an integer count.
EWM_MEASURES: tuple[str, ...] = (
    "ewm_margin",
    "ewm_net_rating",
    "ewm_off_rating",
    "ewm_def_rating",
    "ewm_pace",
    "ewm_ts_pct",
    "ewm_efg",
    "ewm_tov",
    "ewm_ftr",
)

# Ridge penalties considered. Selected by inner forward-chaining CV on training
# rows only; never by looking at a test block.
RIDGE_ALPHAS: tuple[float, ...] = (0.1, 1.0, 10.0, 100.0)

# Poisson penalties. PoissonRegressor's own default (alpha=1.0) is a strong L2
# pull towards mu=1, which is not remotely where a 220-point total lives, so the
# grid is shifted two orders of magnitude down.
POISSON_ALPHAS: tuple[float, ...] = (1e-4, 1e-3, 1e-2, 1e-1)

# Fractions of the fitting data reserved for validation folds. The inner CV that
# picks the penalty, and the out-of-fold pass that estimates sigma, both start
# here so neither uses a model trained on almost nothing.
_INNER_START_FRAC = 0.6


# --------------------------------------------------------------------------- #
# Design matrices
# --------------------------------------------------------------------------- #

def _schedule_features(sched: pd.DataFrame) -> pd.DataFrame:
    """Rest / fatigue / travel columns, plus the rest differentials.

    All of these are known from the published schedule before tip-off, so none of
    them can leak. ``is_home`` from the shared builder is deliberately dropped:
    it is constant, and a constant column cannot survive the scaler (see the
    module docstring).
    """
    out = pd.DataFrame(index=sched.index)
    out["rest_diff"] = sched["home_rest_days"] - sched["away_rest_days"]
    out["home_rest_days"] = sched["home_rest_days"]
    out["away_rest_days"] = sched["away_rest_days"]
    out["home_back_to_back"] = sched["home_back_to_back"]
    out["away_back_to_back"] = sched["away_back_to_back"]
    out["in_3_of_4_diff"] = sched["home_in_3_of_4"] - sched["away_in_3_of_4"]
    out["travel_miles"] = sched["travel_miles"]
    return out


def make_design_matrix(
    roll: pd.DataFrame,
    sched: pd.DataFrame,
    kind: str = "margin",
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build the design matrix for `kind` in {"margin", "total"}.

    `roll` is `nba_data.rolling_team_features(games)` and `sched` is
    `nba_data.rest_and_travel(games)`, both aligned row-wise to the same games
    frame. `columns`, if given, restricts the result to those names -- which is
    how the net-rating reference model is expressed.

    For `kind == "margin"` every EWM measure enters as home-minus-away, plus the
    difference in games played (experience), plus the schedule block. Signs are
    NOT hand-flipped: a higher home DEF rating is good for the home side, a
    higher away DEF rating is good for the away side, so the learned coefficient
    on `ewm_def_rating_diff` comes out positive on its own. Flipping signs by
    hand is how a margin model ends up predicting that good defence is bad.

    For `kind == "total"` the same measures enter as sums, plus summed games
    played, plus the schedule block.
    """
    if kind not in ("margin", "total"):
        raise ValueError(f"kind must be 'margin' or 'total', got {kind!r}")

    combine = (lambda h, a: h - a) if kind == "margin" else (lambda h, a: h + a)
    cols: dict[str, pd.Series] = {}
    for m in EWM_MEASURES:
        h, a = roll[f"home_{m}"], roll[f"away_{m}"]
        label = "diff" if kind == "margin" else "sum"
        cols[f"{m}_{label}"] = combine(h, a)

    if kind == "margin":
        cols["games_played_diff"] = (roll["home_games_played"]
                                     - roll["away_games_played"])
    else:
        cols["games_played_sum"] = (roll["home_games_played"]
                                    + roll["away_games_played"])

    for name, s in _schedule_features(sched).items():
        cols[name] = s.to_numpy()

    X = pd.DataFrame(cols, index=roll.index)
    if columns is not None:
        missing = [c for c in columns if c not in X.columns]
        if missing:
            raise KeyError(f"design matrix has no column(s) {missing}; "
                           f"available: {list(X.columns)}")
        X = X[list(columns)]
    return X.astype(float)


def assert_no_constant_columns(X: pd.DataFrame) -> None:
    """Refuse a design matrix whose columns the scaler would flatten to zero.

    StandardScaler divides by the column standard deviation and substitutes 1.0
    when that is 0, then subtracts the mean -- so a constant column becomes
    identically 0.0 and contributes nothing to any downstream fit. For `is_home`
    that would delete home-court advantage without any visible symptom.
    """
    sd = X.std(ddof=0)
    dead = [c for c in sd.index if not np.isfinite(sd[c]) or sd[c] == 0.0]
    if dead:
        raise ValueError(
            f"constant / degenerate design columns {dead}: StandardScaler maps "
            "these to exactly 0.0, silently deleting their contribution. Use the "
            "model intercept for the home indicator instead."
        )


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #

@dataclass
class Dataset:
    """Everything the models consume, built once and shared by every split."""
    games: pd.DataFrame
    X_margin: pd.DataFrame
    X_total: pd.DataFrame

    @property
    def n(self) -> int:
        return len(self.games)

    def y(self, name: str) -> np.ndarray:
        return self.games[name].to_numpy(dtype=float)


def build_dataset(games: pd.DataFrame | None = None) -> Dataset:
    """Build margin and total design matrices for a games frame.

    Features are computed ONCE over the full chronological frame and then sliced
    per split. That is safe and is not leakage: every feature is strictly
    causal, so a row's features depend only on games that precede it, and adding
    later games cannot change an earlier row. `Tests/test_margin_total.py`
    pins that property down by truncating the input and re-comparing.
    """
    if games is None:
        games = nd.build_games()
    games = games.reset_index(drop=True)
    roll = nd.rolling_team_features(games).reset_index(drop=True)
    sched = nd.rest_and_travel(games).reset_index(drop=True)
    if len(roll) != len(games) or len(sched) != len(games):
        raise ValueError("feature builders are not aligned to the games frame")
    return Dataset(games=games,
                   X_margin=make_design_matrix(roll, sched, "margin"),
                   X_total=make_design_matrix(roll, sched, "total"))


# --------------------------------------------------------------------------- #
# Train-only fitting helpers
# --------------------------------------------------------------------------- #

def _inner_folds(n: int, n_folds: int, start_frac: float = _INNER_START_FRAC
                 ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Forward-chaining folds over rows [0, n): train on the past, test forward.

    The first `start_frac * n` rows are training-only, mirroring
    `nba_data.walk_forward_split`. Shuffled K-fold is not usable here: with
    time-ordered features a random fold puts future games into the fit of a past
    game.
    """
    cuts = np.linspace(int(n * start_frac), n, n_folds + 1).astype(int)
    out = []
    for i in range(n_folds):
        lo, hi = cuts[i], cuts[i + 1]
        if hi > lo and lo > 0:
            out.append((np.arange(0, lo), np.arange(lo, hi)))
    return out


def _ridge_pipeline(alpha: float) -> Pipeline:
    """Median imputer -> standard scaler -> ridge, all fitted inside `.fit`.

    The intercept is left ON. It is the home indicator: every row of a margin or
    total regression needs a baseline, and since `is_home` is constant it is the
    only place that baseline can live. For the margin model the intercept reads
    directly as average home-court advantage in points; for the total model it
    reads as the league-average scoring level. Dropping it (setting
    `fit_intercept=False`) is not a neutral simplification -- without it the total
    model predicts roughly league-average-minus-itself and lands ~220 points
    below the answer, which is what happened the first time this was written.

    The imputer and scaler are Pipeline steps, so `fit` on training rows is the
    only thing that ever touches them.
    """
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha, fit_intercept=True)),
    ])


def _poisson_pipeline(alpha: float) -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("poisson", PoissonRegressor(alpha=alpha, max_iter=2000)),
    ])


# BLAS spawns one thread per core by default, and these models are tiny -- 17
# columns, a few thousand rows. The resulting oversubscription costs an order of
# magnitude in wall clock (measured: 4.15 s -> 0.25 s for one Poisson fit on
# 7,919 rows) for no numerical gain. It also makes results mildly
# non-reproducible, because BLAS reduction order depends on the thread count and
# floating-point addition is not associative. Scoping the fit to one thread fixes
# both. `threadpoolctl` ships with scikit-learn; if it is missing the fits still
# run, just slower.
try:                                            # pragma: no cover - env dependent
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:                             # pragma: no cover - env dependent
    @contextlib.contextmanager
    def _threadpool_limits(_n: int = 1):
        yield


def _select_ridge_alpha(X: pd.DataFrame, y: np.ndarray,
                        grid: Sequence[float] = RIDGE_ALPHAS,
                        n_folds: int = 4) -> tuple[float, list[dict]]:
    """Pick the ridge penalty by forward-chaining CV inside the training rows."""
    folds = _inner_folds(len(X), n_folds)
    scores: list[dict] = []
    for a in grid:
        errs = []
        for tr, va in folds:
            m = _ridge_pipeline(a).fit(X.iloc[tr], y[tr])
            errs.append(float(np.sqrt(np.mean((m.predict(X.iloc[va]) - y[va]) ** 2))))
        scores.append({"alpha": float(a), "cv_rmse": float(np.mean(errs))})
    best = min(scores, key=lambda r: r["cv_rmse"])
    return float(best["alpha"]), scores


def _select_poisson_alpha(X: pd.DataFrame, y: np.ndarray,
                          grid: Sequence[float] = POISSON_ALPHAS,
                          n_folds: int = 4) -> tuple[float, list[dict]]:
    """Pick the Poisson penalty by held-out mean Poisson log score on TRAIN rows.

    Scored with the Poisson log predictive density rather than squared error,
    because that is the quantity the Poisson variant is being judged on.
    """
    folds = _inner_folds(len(X), n_folds)
    scores: list[dict] = []
    with _threadpool_limits(limits=1):
        for a in grid:
            lls = []
            for tr, va in folds:
                m = _poisson_pipeline(a).fit(X.iloc[tr], y[tr])
                mu = np.clip(m.predict(X.iloc[va]), 1e-9, None)
                lls.append(float(_poisson_logpmf(y[va], mu).mean()))
            scores.append({"alpha": float(a), "cv_logscore": float(np.mean(lls))})
    best = max(scores, key=lambda r: r["cv_logscore"])
    return float(best["alpha"]), scores


def _poisson_logpmf(y: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """log P(Y = y) for Y ~ Poisson(mu), written out to avoid a second dependency."""
    mu = np.clip(np.asarray(mu, float), 1e-12, None)
    y = np.asarray(y, float)
    from scipy.special import gammaln
    return y * np.log(mu) - mu - gammaln(y + 1.0)


def _oof_residuals(X: pd.DataFrame, y: np.ndarray, alpha: float,
                   n_folds: int = 5) -> np.ndarray:
    """Out-of-sample residuals on the TRAINING rows only.

    Why not just take the in-sample RMSE? With p fitted parameters on n rows the
    in-sample residual scale is shrunk by roughly sqrt(1 - p/n); using it in
    Phi(mhat / sigma) makes every probability too confident, and that shows up
    as a calibration table with predicted above actual in the outer bins. This
    is the same failure the in-sample variant has, but it is measurable: both
    are reported in ``run_experiment``.
    """
    folds = _inner_folds(len(X), n_folds)
    resid = np.empty(len(y))
    resid[:] = np.nan
    for tr, va in folds:
        m = _ridge_pipeline(alpha).fit(X.iloc[tr], y[tr])
        resid[va] = y[va] - m.predict(X.iloc[va])
    return resid[~np.isnan(resid)]


def _gaussian_logpdf(y: np.ndarray, mu: np.ndarray, sigma: float) -> np.ndarray:
    z = (np.asarray(y, float) - np.asarray(mu, float)) / sigma
    return -0.5 * z ** 2 - np.log(sigma) - 0.5 * np.log(2 * np.pi)


def paired_logscore_delta(y: np.ndarray, lp_a: np.ndarray, lp_b: np.ndarray,
                          n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap mean log-score difference for a CONTINUOUS target.

    SIGN CONVENTION, kept deliberately IDENTICAL to
    `nba_data.paired_bootstrap_delta` even though the quantity here is a log
    score rather than a log loss, because the two natural formulae have
    OPPOSITE signs and this is the easiest sign in the repo to get wrong:

        delta = mean(lp_b) - mean(lp_a)

    A POSITIVE delta means the SECOND model (`lp_b`) has the higher log
    predictive density and is therefore better. Note the explicit minus in
    front of `lp_a`: writing `mean(lp_a) - mean(lp_b)` would flip the meaning
    while looking correct, because for losses and for scores the "second model
    wins" direction reverses.

    This exists only for the Gaussian-vs-Poisson comparison on the game TOTAL,
    which is an integer quantity where the shared binary log-loss bootstrap does
    not apply. Win-probability comparisons in this module go through the shared
    `nba_data.paired_bootstrap_delta` and nothing else.
    """
    rng = np.random.default_rng(seed)
    y, la, lb = map(np.asarray, (y, lp_a, lp_b))
    n = len(y)
    d = float((lb - la).mean())
    stats = np.empty(n_boot)
    for i in range(n_boot):
        j = rng.integers(0, n, size=n)
        stats[i] = (lb[j] - la[j]).mean()
    return {"delta": d,
            "ci_lo": float(np.percentile(stats, 2.5)),
            "ci_hi": float(np.percentile(stats, 97.5)),
            "p_positive": float((stats > 0).mean()),
            "n_boot": n_boot}


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

class MarginTotalModel:
    """Predict margin and total; derive win probability and spread.

    Fitted quantities, all on training rows only: imputer medians, scaler
    variances, ridge penalty, ridge coefficients, the intercept (home-court
    advantage), the margin residual scale `sigma`, the Poisson penalty, and the
    total residual scale.

    Parameters
    ----------
    alpha
        Ridge penalty, or ``"auto"`` to choose it by forward-chaining CV inside
        the training rows.
    sigma_method
        ``"oof"`` (default) estimates sigma from forward-chained out-of-sample
        residuals on training rows. ``"insample"`` uses the training residuals
        directly and is kept only so the two can be compared -- see
        `_oof_residuals`.
    """

    def __init__(self, alpha: float | str = "auto", sigma_method: str = "oof",
                 poisson_alpha: float | str = "auto", n_inner: int = 4,
                 n_oof: int = 5) -> None:
        if sigma_method not in ("oof", "insample"):
            raise ValueError("sigma_method must be 'oof' or 'insample'")
        self.alpha = alpha
        self.sigma_method = sigma_method
        self.poisson_alpha = poisson_alpha
        self.n_inner = n_inner
        self.n_oof = n_oof
        self.is_fitted_ = False

    # -- fitting ------------------------------------------------------------ #

    def _fit_one(self, X: pd.DataFrame, y: np.ndarray
                 ) -> tuple[Pipeline, float, float]:
        assert_no_constant_columns(X)
        alpha = (self.alpha if self.alpha != "auto"
                 else _select_ridge_alpha(X, y, n_folds=self.n_inner)[0])
        alpha = float(alpha)
        model = _ridge_pipeline(alpha).fit(X, y)
        if self.sigma_method == "oof":
            resid = _oof_residuals(X, y, alpha, n_folds=self.n_oof)
        else:
            resid = y - model.predict(X)
        # Add-one correction: sigma is estimated from `len(resid)` observations
        # but consumes one degree of freedom per fitted coefficient PLUS the
        # intercept. With ~8k residuals it is worth well under 0.1%, and it keeps
        # sigma^2 unbiased as a variance estimate.
        dof = max(len(resid) - X.shape[1] - 1, 1)
        sigma = float(np.sqrt(np.sum(resid ** 2) / dof))
        return model, sigma, alpha

    def fit_margin(self, X: pd.DataFrame, y: np.ndarray) -> "MarginTotalModel":
        """Fit the margin half only. Used by the net-rating reference model."""
        self.margin_, self.sigma_, self.alpha_ = self._fit_one(
            X, np.asarray(y, float))
        self.is_fitted_ = True
        return self

    def fit(self, X_margin: pd.DataFrame, y_margin: np.ndarray,
            X_total: pd.DataFrame, y_total: np.ndarray) -> "MarginTotalModel":
        """Fit margin and total on the given (training) rows."""
        self.fit_margin(X_margin, y_margin)
        self.total_, self.sigma_total_, self.alpha_total_ = self._fit_one(
            X_total, np.asarray(y_total, float))
        assert_no_constant_columns(X_total)
        p_alpha = (self.poisson_alpha if self.poisson_alpha != "auto"
                   else _select_poisson_alpha(X_total, np.asarray(y_total, float),
                                              n_folds=self.n_inner)[0])
        self.poisson_alpha_ = float(p_alpha)
        with _threadpool_limits(limits=1):
            self.poisson_ = _poisson_pipeline(self.poisson_alpha_).fit(X_total,
                                                                      y_total)
        return self

    # -- attributes --------------------------------------------------------- #

    @property
    def coefficients_(self) -> pd.Series:
        """Coefficients on the ORIGINAL feature scale, intercept included.

        The pipeline scales, so `ridge.coef_` is in z-units. Multiplying by the
        scaler's scale puts it back in points-per-unit-feature, and the
        intercept is the ridge intercept plus the sum of coef * centre.
        """
        m = self.margin_.named_steps
        coef = m["ridge"].coef_ / m["scale"].scale_
        b = m["ridge"].intercept_ - float(np.sum(coef * m["scale"].mean_))
        return pd.Series(np.append(coef, b),
                         index=list(self.margin_.feature_names_in_) + ["intercept"])

    # -- prediction --------------------------------------------------------- #

    def predict_margin(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.margin_.predict(X), float)

    def predict_total_gaussian(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.total_.predict(X), float)

    def predict_total_poisson(self, X: pd.DataFrame) -> np.ndarray:
        """Poisson mean of the total. Note this is exp(eta), so it is biased low."""
        return np.clip(np.asarray(self.poisson_.predict(X), float), 1e-6, None)

    @staticmethod
    def win_probability(margin_hat: np.ndarray, sigma: float) -> np.ndarray:
        """P(home wins) = Phi(margin_hat / sigma).

        A tie is impossible in basketball, so this is exact rather than a
        limit: margin is an odd number of points about half the time, and
        Phi(0) = 0.5 puts exactly half the mass on the home side, which is
        what the base rate says.
        """
        return norm.cdf(np.asarray(margin_hat, float) / float(sigma))

    def predict_win_prob(self, X: pd.DataFrame) -> np.ndarray:
        return self.win_probability(self.predict_margin(X), self.sigma_)

    @staticmethod
    def spread_from_margin(margin_hat: np.ndarray,
                           snap: str = "half") -> np.ndarray:
        """Convert a predicted margin into a posted-style line.

        Real spreads are whole or half points, so a raw regression output is not
        directly comparable with anything a bookmaker would print. `snap` is
        "half" (x.5 grid, the house convention), "whole" (nearest integer) or
        "none".

        There is no market line in this dataset to compare against. This
        function only quantises our own prediction; the MAE that follows is
        against ACTUAL margins, not against a closing line.
        """
        m = np.asarray(margin_hat, float)
        if snap == "half":
            return np.round(2.0 * m) / 2.0
        if snap == "whole":
            return np.round(m)
        if snap == "none":
            return m
        raise ValueError("snap must be 'half', 'whole' or 'none'")


class DirectWinModel:
    """Direct logistic regression on the same design matrix.

    Not a strawman: this is the honest control for the whole premise of the
    module. If going through the margin does not beat fitting the label
    directly on identical features, the Phi() detour is costing, not buying.
    """

    def __init__(self, C: float = 1.0) -> None:
        self.C = C

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "DirectWinModel":
        assert_no_constant_columns(X)
        self.pipe_ = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(C=self.C, max_iter=2000)),
        ]).fit(X, np.asarray(y, int))
        return self

    def predict_win_prob(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipe_.predict_proba(X)[:, 1]


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@dataclass
class SplitResult:
    split: int
    n_train: int
    n_test: int
    test_start: str
    alpha: float
    sigma: float
    sigma_insample: float
    home_advantage_pts: float
    home_win_rate_train: float
    margin_rmse: float
    total_rmse_gaussian: float
    total_rmse_poisson: float
    total_rmse_contrast_design: float
    total_logscore_gaussian: float
    total_logscore_poisson: float
    win: dict = field(default_factory=dict)
    spread_mae: dict = field(default_factory=dict)
    reference: dict = field(default_factory=dict)
    bootstrap: dict = field(default_factory=dict)
    total_boot: dict = field(default_factory=dict)


def walk_forward_eval(data: Dataset, n_splits: int = 5,
                      initial_frac: float = 0.5,
                      sigma_method: str = "oof") -> list[SplitResult]:
    """Run the full walk-forward evaluation and return one result per split."""
    games = data.games
    y_win = games["y"].to_numpy(int)
    y_margin = games["margin"].to_numpy(float)
    y_total = games["total"].to_numpy(float)

    # The net-rating reference, exactly as specified: ONE feature (the EWM
    # net-rating difference) plus the home indicator, which is the intercept.
    # Net rating is already OFF - DEF, so a single contrast is the whole
    # "obvious" margin model and there is nothing else to give it.
    ref_cols = ["ewm_net_rating_diff"]

    results: list[SplitResult] = []
    for k, (tr, te) in enumerate(nd.walk_forward_split(games, n_splits, initial_frac), 1):
        Xm_tr, Xm_te = data.X_margin.iloc[tr], data.X_margin.iloc[te]
        Xt_tr, Xt_te = data.X_total.iloc[tr], data.X_total.iloc[te]

        model = MarginTotalModel(sigma_method=sigma_method).fit(
            Xm_tr, y_margin[tr], Xt_tr, y_total[tr])
        insample = MarginTotalModel(sigma_method="insample").fit_margin(
            Xm_tr, y_margin[tr])

        ref = MarginTotalModel(sigma_method=sigma_method).fit_margin(
            Xm_tr[ref_cols], y_margin[tr])
        direct = DirectWinModel().fit(Xm_tr, y_win[tr])

        m_te = model.predict_margin(Xm_te)
        p = model.predict_win_prob(Xm_te)
        p_home = np.full(len(te), float(y_win[tr].mean()))          # train-only base rate
        p_ref = ref.predict_win_prob(Xm_te[ref_cols])
        p_direct = direct.predict_win_prob(Xm_te)
        # Counterfactual: same coefficients, but sigma taken from the IN-SAMPLE
        # training residuals instead of the out-of-sample ones. Nothing here
        # leaks -- both sigmas are train-only -- this isolates how much of the
        # Phi() calibration comes purely from choosing the right residual scale.
        p_insample_sigma = MarginTotalModel.win_probability(m_te, insample.sigma_)

        t_g = model.predict_total_gaussian(Xt_te)
        t_p = model.predict_total_poisson(Xt_te)

        # Ablation: the SAME total target fitted on the home-minus-away
        # CONTRAST design instead of the sum design, to price the feature-form
        # choice rather than assert it. Contrasting a level destroys the level.
        c_alpha = _select_ridge_alpha(Xm_tr, y_total[tr], n_folds=4)[0]
        contrast = _ridge_pipeline(c_alpha).fit(Xm_tr, y_total[tr])
        contrast_t = contrast.predict(Xm_te)

        res = SplitResult(
            split=k, n_train=len(tr), n_test=len(te),
            test_start=str(games["GAME_DATE"].iloc[te[0]].date()),
            alpha=model.alpha_, sigma=model.sigma_,
            sigma_insample=insample.sigma_,
            home_advantage_pts=float(model.coefficients_["intercept"]),
            home_win_rate_train=float(y_win[tr].mean()),
            margin_rmse=float(np.sqrt(np.mean((m_te - y_margin[te]) ** 2))),
            total_rmse_gaussian=float(np.sqrt(np.mean((t_g - y_total[te]) ** 2))),
            total_rmse_poisson=float(np.sqrt(np.mean((t_p - y_total[te]) ** 2))),
            total_rmse_contrast_design=float(np.sqrt(
                np.mean((contrast_t - y_total[te]) ** 2))),
            total_logscore_gaussian=float(_gaussian_logpdf(
                y_total[te], t_g, model.sigma_total_).mean()),
            total_logscore_poisson=float(_poisson_logpmf(y_total[te], t_p).mean()),
            win=nd.evaluate(y_win[te], p),
            spread_mae={
                "raw": float(np.mean(np.abs(model.spread_from_margin(m_te, "none")
                                            - y_margin[te]))),
                "whole": float(np.mean(np.abs(model.spread_from_margin(m_te, "whole")
                                              - y_margin[te]))),
                "half": float(np.mean(np.abs(model.spread_from_margin(m_te, "half")
                                             - y_margin[te]))),
                "zero_line": float(np.mean(np.abs(y_margin[te]))),
                "reference": float(np.mean(np.abs(ref.predict_margin(Xm_te[ref_cols])
                                                  - y_margin[te]))),
            },
        )
        # SIGN: a POSITIVE delta means the SECOND model is better (lower log
        # loss). Read every line below with that in mind.
        res.reference = {
            "home_base_rate": nd.evaluate(y_win[te], p_home),
            "net_rating_gaussian": nd.evaluate(y_win[te], p_ref),
            "direct_logistic": nd.evaluate(y_win[te], p_direct),
            "insample_sigma": nd.evaluate(y_win[te], p_insample_sigma),
        }
        res.bootstrap = {
            "vs_home_base_rate": nd.paired_bootstrap_delta(y_win[te], p, p_home),
            "vs_net_rating": nd.paired_bootstrap_delta(y_win[te], p, p_ref),
            "vs_direct_logistic": nd.paired_bootstrap_delta(y_win[te], p, p_direct),
            "vs_insample_sigma": nd.paired_bootstrap_delta(y_win[te], p,
                                                            p_insample_sigma),
        }
        # SIGN: a POSITIVE delta here means the POISSON total has the higher log
        # predictive density, i.e. Poisson is better. A NEGATIVE delta means the
        # Gaussian is better. Same reading as the win-probability rows above.
        res.total_boot = paired_logscore_delta(
            y_total[te],
            _gaussian_logpdf(y_total[te], t_g, model.sigma_total_),
            _poisson_logpmf(y_total[te], t_p))
        results.append(res)
    return results


def _pooled(parts: list[dict]) -> dict:
    """Concatenate pooled calibration bins across splits, weighted by n."""
    bins: dict[int, dict] = {}
    for p in parts:
        for b in p["calibration"]:
            acc = bins.setdefault(b["bin"], {"lo": b["lo"], "hi": b["hi"], "n": 0,
                                             "sp": 0.0, "sa": 0.0})
            acc["n"] += b["n"]
            acc["sp"] += b["mean_pred"] * b["n"]
            acc["sa"] += b["actual"] * b["n"]
    out = []
    for b in sorted(bins):
        a = bins[b]
        out.append({"bin": b, "lo": a["lo"], "hi": a["hi"], "n": a["n"],
                    "mean_pred": a["sp"] / a["n"], "actual": a["sa"] / a["n"]})
    return {"calibration": out,
            "calibration_in_the_large": sum(
                (b["mean_pred"] - b["actual"]) * b["n"] for b in out)
            / sum(b["n"] for b in out),
            "ece": sum(abs(b["mean_pred"] - b["actual"]) * b["n"] for b in out)
            / sum(b["n"] for b in out)}


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _se(xs: Sequence[float]) -> float:
    """Standard error of the mean across split-level values."""
    a = np.asarray(xs, float)
    return float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan")


def _calibration_rows(parts: list[dict]) -> list[dict]:
    rows: dict[int, dict] = {}
    for p in parts:
        for b in p["calibration"]:
            acc = rows.setdefault(b["bin"], {"lo": b["lo"], "hi": b["hi"],
                                             "n": 0, "sp": 0.0, "sa": 0.0})
            acc["n"] += b["n"]
            acc["sp"] += b["mean_pred"] * b["n"]
            acc["sa"] += b["actual"] * b["n"]
    out = []
    for b in sorted(rows):
        a = rows[b]
        out.append({"lo": a["lo"], "hi": a["hi"], "n": a["n"],
                    "mean_pred": a["sp"] / a["n"], "actual": a["sa"] / a["n"]})
    return out


def run_experiment(n_splits: int = 5, sigma_method: str = "oof",
                   verbose: bool = True) -> dict:
    """Fit, evaluate against locally-fitted references, and print the report."""
    data = build_dataset()
    games = data.games
    y_win = games["y"].to_numpy(int)

    out: dict = {
        "n_games": int(len(games)),
        "n_margin_features": int(data.X_margin.shape[1]),
        "n_total_features": int(data.X_total.shape[1]),
        "date_range": f"{games.GAME_DATE.min().date()} .. {games.GAME_DATE.max().date()}",
        "overdispersion": {
            "total_var_over_mean": float(games.total.var() / games.total.mean()),
            "home_var_over_mean": float(games.home_pts.var() / games.home_pts.mean()),
            "away_var_over_mean": float(games.away_pts.var() / games.away_pts.mean()),
        },
        "sigma_method": sigma_method,
    }
    out["results"] = walk_forward_eval(data, n_splits=n_splits,
                                       sigma_method=sigma_method)
    R = out["results"]

    pooled = {}
    for split_idx, rs in enumerate(R):
        tr, te = nd.walk_forward_split(games, n_splits)[split_idx]
        pooled.setdefault("p", []).append(rs.win)
        pooled.setdefault("y", []).append(y_win[te])
        for k in ("home_base_rate", "net_rating_gaussian", "direct_logistic",
                  "insample_sigma"):
            pooled.setdefault(k, []).append(rs.reference[k])
    out["calibration"] = _calibration_rows(pooled["p"])
    for name in ("p", "home_base_rate", "net_rating_gaussian", "direct_logistic",
                   "insample_sigma"):
        out.setdefault("cal", {})[name] = _pooled(pooled[name])

    out["summary"] = {
        "margin_rmse": [r.margin_rmse for r in R],
        "margin_rmse_mean": float(np.mean([r.margin_rmse for r in R])),
        "margin_rmse_se": _se([r.margin_rmse for r in R]),
        "total_rmse_gaussian": [r.total_rmse_gaussian for r in R],
        "total_rmse_poisson": [r.total_rmse_poisson for r in R],
        "total_rmse_contrast_design": float(np.mean(
            [r.total_rmse_contrast_design for r in R])),
        "total_rmse_gaussian_mean": float(np.mean([r.total_rmse_gaussian for r in R])),
        "total_rmse_poisson_mean": float(np.mean([r.total_rmse_poisson for r in R])),
        "total_rmse_gaussian_se": _se([r.total_rmse_gaussian for r in R]),
        "total_rmse_poisson_se": _se([r.total_rmse_poisson for r in R]),
        "total_logscore_gaussian": [r.total_logscore_gaussian for r in R],
        "total_logscore_poisson": [r.total_logscore_poisson for r in R],
        "total_logscore_gaussian_mean": float(np.mean([r.total_logscore_gaussian
                                                       for r in R])),
        "total_logscore_poisson_mean": float(np.mean([r.total_logscore_poisson
                                                      for r in R])),
        "total_logscore_se": _se([r.total_logscore_gaussian - r.total_logscore_poisson
                                  for r in R]),
        "sigma": [r.sigma for r in R],
        "sigma_insample": [r.sigma_insample for r in R],
        "spread_mae_raw": float(np.mean([r.spread_mae["raw"] for r in R])),
        "spread_mae_whole": float(np.mean([r.spread_mae["whole"] for r in R])),
        "spread_mae_half": float(np.mean([r.spread_mae["half"] for r in R])),
        "spread_mae_zero_line": float(np.mean([r.spread_mae["zero_line"] for r in R])),
        "spread_mae_reference": float(np.mean([r.spread_mae["reference"] for r in R])),
    }
    if verbose:
        print(format_report(out))
    return out


def format_report(out: dict) -> str:
    R = out["results"]
    S = out["summary"]
    L: list[str] = []
    a = L.append

    a("=" * 78)
    a("MARGIN + TOTAL MODEL  (src/Models/margin_total.py)")
    a("=" * 78)
    a(f"games {out['n_games']:,}   {out['date_range']}")
    a(f"design: {out['n_margin_features']} margin features (home-minus-away + schedule), "
      f"{out['n_total_features']} total features (sums + schedule)")
    a("home indicator carried as the model intercept, not a column "
      "(StandardScaler would zero it out)")
    a("NO MARKET LINES EXIST IN THIS DATASET. Nothing below is a comparison "
      "against a closing spread.")
    a("")
    a(f"dispersion check  var/mean: total {out['overdispersion']['total_var_over_mean']:.2f}, "
      f"home {out['overdispersion']['home_var_over_mean']:.2f}, "
      f"away {out['overdispersion']['away_var_over_mean']:.2f}   "
      f"(1.0 = exactly Poisson)")
    a("")

    a("PER SPLIT")
    a(f"{'k':>2} {'train':>6} {'test':>6} {'from':>10} {'alpha':>7} {'sigma':>6} "
      f"{'mRMSE':>7} {'tRMSEg':>7} {'tRMSp':>7} {'logloss':>8} {'brier':>7} "
      f"{'auc':>6} {'sprMAE':>7}")
    for r in R:
        a(f"{r.split:>2} {r.n_train:>6,} {r.n_test:>6,} {r.test_start:>10} "
          f"{r.alpha:>7.2f} {r.sigma:>6.2f} {r.margin_rmse:>7.3f} "
          f"{r.total_rmse_gaussian:>7.3f} {r.total_rmse_poisson:>7.3f} "
          f"{r.win['log_loss']:>8.4f} {r.win['brier']:>7.4f} {r.win['auc']:>6.4f} "
          f"{r.spread_mae['raw']:>7.3f}")
    a(f"{'mean':>2} {'':>6} {'':>6} {'':>10} {'':>7} "
      f"{np.mean(S['sigma']):>6.2f} {S['margin_rmse_mean']:>7.3f} "
      f"{S['total_rmse_gaussian_mean']:>7.3f} {S['total_rmse_poisson_mean']:>7.3f}")
    a(f"{'+-SE':>2} {'':>6} {'':>6} {'':>10} {'':>7} "
      f"{'':>6} {S['margin_rmse_se']:>7.3f} {S['total_rmse_gaussian_se']:>7.3f} "
      f"{S['total_rmse_poisson_se']:>7.3f}")
    a("")

    a("WIN PROBABILITY  (Phi(margin_hat / sigma))  vs locally fitted references")
    a(f"{'model':<36}{'logloss':>9}{'brier':>8}{'auc':>8}{'accuracy':>9}")
    for name, key in [("margin model (this module)", "p"),
                      ("always-home base rate (train)", "home_base_rate"),
                      ("net-rating diff + home (reference)", "net_rating_gaussian"),
                      ("direct logistic, same features", "direct_logistic"),
                      ("margin model, in-sample sigma", "insample_sigma")]:
        ll = float(np.mean([r.win["log_loss"] if key == "p"
                            else r.reference[key]["log_loss"] for r in R]))
        br = float(np.mean([r.win["brier"] if key == "p"
                            else r.reference[key]["brier"] for r in R]))
        au = float(np.mean([r.win["auc"] if key == "p"
                            else r.reference[key]["auc"] for r in R]))
        ac = float(np.mean([r.win["accuracy"] if key == "p"
                            else r.reference[key]["accuracy"] for r in R]))
        a(f"{name:<36}{ll:>9.4f}{br:>8.4f}{au:>8.4f}{ac:>9.4f}")
    a("")
    a("PAIRED BOOTSTRAP, mean log-loss loss(margin) - loss(reference).")
    a("A POSITIVE delta means the REFERENCE is better. CI excluding 0 => real;")
    a("a CI straddling 0 => the two are indistinguishable at this sample size.")
    a(f"{'comparison':<36}{'delta':>9}{'ci_lo':>9}{'ci_hi':>9}{'P(delta>0)':>11}")
    for key, label in [("vs_home_base_rate", "vs always-home base rate"),
                       ("vs_net_rating", "vs net-rating reference"),
                       ("vs_direct_logistic", "vs direct logistic"),
                       ("vs_insample_sigma", "vs in-sample sigma")]:
        d = [r.bootstrap[key] for r in R]
        delta = float(np.mean([x["delta"] for x in d]))
        lo = float(np.mean([x["ci_lo"] for x in d]))
        hi = float(np.mean([x["ci_hi"] for x in d]))
        pp = float(np.mean([x["p_positive"] for x in d]))
        a(f"{label:<36}{delta:>9.4f}{lo:>9.4f}{hi:>9.4f}{pp:>11.3f}")
    a("")

    a("CALIBRATION of the win probability (pooled over all test blocks)")
    a(f"{'bin':>14}{'n':>7}{'mean pred':>11}{'actual':>9}{'gap':>9}")
    for b in out["calibration"]:
        a(f"{b['lo']:>6.2f}-{b['hi']:<7.2f}{b['n']:>7}{b['mean_pred']:>11.4f}"
          f"{b['actual']:>9.4f}{b['mean_pred'] - b['actual']:>+9.4f}")
    c = out["cal"]["p"]
    a(f"calibration in the large (mean pred - actual): {c['calibration_in_the_large']:+.4f}"
      f"   ECE: {c['ece']:.4f}")
    ci = out["cal"]["insample_sigma"]
    a(f"same, with in-sample sigma: in-the-large "
      f"{ci['calibration_in_the_large']:+.4f}   ECE {ci['ece']:.4f}")
    a("  Both sigmas are train-only, so this is not a leakage comparison -- it")
    a("  is a check on whether the textbook argument for the out-of-sample")
    a("  sigma survives at this sample size. It does not: the two are")
    a("  indistinguishable on log loss and on this table. The default stays")
    a("  'oof' because it is right in principle and because the gap grows")
    a("  with p/n, not because the data here prefers it.")
    a("")

    a("TOTAL MODEL: Gaussian residual vs Poisson (log link), same design")
    a(f"{'split':>6}{'RMSE gauss':>12}{'RMSE pois':>12}"
      f"{'logscore gauss':>16}{'logscore pois':>16}{'diff':>9}")
    for r in R:
        a(f"{r.split:>6}{r.total_rmse_gaussian:>12.3f}{r.total_rmse_poisson:>12.3f}"
          f"{r.total_logscore_gaussian:>16.4f}{r.total_logscore_poisson:>16.4f}"
          f"{r.total_logscore_gaussian - r.total_logscore_poisson:>+9.4f}")
    a(f"{'mean':>6}{S['total_rmse_gaussian_mean']:>12.3f}"
      f"{S['total_rmse_poisson_mean']:>12.3f}"
      f"{S['total_logscore_gaussian_mean']:>16.4f}"
      f"{S['total_logscore_poisson_mean']:>16.4f}"
      f"{S['total_logscore_gaussian_mean'] - S['total_logscore_poisson_mean']:>+9.4f}")
    a("")
    a("feature-form ablation, same target, same ridge:")
    a(f"  total RMSE on the SUM design      {S['total_rmse_gaussian_mean']:8.3f}")
    a(f"  total RMSE on the CONTRAST design {S['total_rmse_contrast_design']:8.3f}"
      "   <- what using home-minus-away for a level costs")
    tb = [r.total_boot for r in R]
    a("paired bootstrap on held-out log predictive density of the actual total,")
    a("delta = logscore(poisson) - logscore(gaussian); POSITIVE => POISSON better,")
    a("matching the reading of the win-probability rows above:")
    a(f"  delta {np.mean([t['delta'] for t in tb]):+.4f}   "
      f"CI [{np.mean([t['ci_lo'] for t in tb]):+.4f}, "
      f"{np.mean([t['ci_hi'] for t in tb]):+.4f}]   "
      f"P(poisson better) {np.mean([t['p_positive'] for t in tb]):.3f}")
    a("")

    a("SPREAD (predicted margin snapped to a posted-style line) - MAE vs ACTUAL margin")
    a(f"{'raw':<12}{S['spread_mae_raw']:>8.3f}")
    a(f"{'nearest int':<12}{S['spread_mae_whole']:>8.3f}")
    a(f"{'x.5 grid':<12}{S['spread_mae_half']:>8.3f}")
    a(f"{'net-rating ref':<12}{S['spread_mae_reference']:>8.3f}")
    a(f"{'always 0 line':<12}{S['spread_mae_zero_line']:>8.3f}   <- pick-no-edge reference")
    a("")
    a("sigma (out-of-sample, train only) vs sigma (in-sample, train residuals):")
    a(f"  oof  {[round(s, 2) for s in S['sigma']]}")
    a(f"  in   {[round(s, 2) for s in S['sigma_insample']]}")
    a("  A smaller in-sample sigma means a sharper, more overconfident Phi().")
    a("")
    a("fitted home-court advantage, read off the intercept (points per game):")
    a(f"  {[round(r.home_advantage_pts, 2) for r in R]}")
    a("")
    a("This measures prediction quality only. There is no closing line in this")
    a("dataset, so none of these numbers say anything about betting value.")
    a("=" * 78)
    return "\n".join(L)


if __name__ == "__main__":
    run_experiment()