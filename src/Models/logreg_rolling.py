"""Logistic regression on causal rolling team features.

MODEL 2 of the suite. MODEL 1 (Elo / home-court) is the bar to beat; this module
fits a single L2-penalised logistic regression on exponentially-weighted team
form, schedule state and a home indicator, and reports honestly whether it gets
there.

WHAT IS AND IS NOT HERE
-----------------------
The loader, the causal feature builders, the walk-forward splitter, the metric
harness and the paired bootstrap all come from ``src.DataProviders.nba_data``.
This module contains no second loader and no second evaluation harness. The only
thing built here is the MODEL: the contrast matrix, a train-only imputer/
standardiser, the regularised fit, and the reporting around it.

DESIGN DECISIONS THAT MATTER
----------------------------
1. CONTRASTS, NOT LEVELS. The label is "did the home team win", so almost every
   informative quantity is a home-minus-away difference. Fitting raw home and
   away *levels* for the same quantity double-counts team strength: the sum
   (home + away)/2 is a league-wide weather term that predicts nothing about who
   wins, and it inflates variance. Differences are built explicitly in
   ``assemble_features``; the only levels kept are ``pace_mean`` (total
   expected pace, which is symmetric and genuinely informative about the game,
   not about either side) and the schedule levels, where "the away team is
   travelling 2000 miles" is a level effect.

2. TRAIN-ONLY IMPUTATION AND STANDARDISATION. ``FeaturePipeline`` fits medians
   (for NaN, which is real here: a team's first game has no EWM) and means/stds
   on the training rows alone. This is the recurring bug in this repo - imputing
   or standardising before the split leaks the test block's distribution. The
   pipeline makes that structurally hard to get wrong and
   ``Tests/test_logreg_rolling.py`` pins it.

3. HOME IS A RAW 1.0 COLUMN, NOT A STANDARDISED ONE. It has zero variance, so
   standardising it is a division by zero, and centring it would make its
   coefficient unidentifiable and fold the home edge into the intercept. Left as
   a literal 1.0 it doubles as the intercept reference: the intercept is the
   league-average home log-odds and ``home_adv`` is the home-court effect.

4. THE OFF/DEF COEFFICIENTS ARE BARELY IDENTIFIED IN THE MAIN MODEL. The rating
   block contains the near-exact linear dependency
   ``net_diff == off_diff - def_diff`` - exact algebra, but the underlying data
   only holds it to about 0.1, because the NBA publishes NET_RATING,
   OFF_RATING and DEF_RATING as independent rounded columns and an EWM
   accumulates the rounding over roughly a half-life. The design therefore
   cannot separate the three, and L2 happily splits the near-unidentifiable
   direction between them: predictions are unaffected, individual coefficients
   are not interpretable. The ``signs`` group drops ``net_diff`` and fits
   ``off_diff`` + ``def_diff`` alone, which IS identified - that variant is what
   the coefficient sanity check is run on.

   The matchup pair is a genuinely DIFFERENT contrast and is not part of this
   problem. Writing it out, because it is easy to get backwards:

       matchup_home = home_off - away_def      ("my offence vs their defence")
       matchup_away = away_off - home_def
       sum         = home_net + away_net       (a SUM of levels - a league term)
       difference  = (home_off + home_def) - (away_off + away_def)
                                                       (total-rating difference)

   so the pair adds the total-quality contrast that the difference form lacks,
   which is why ``rating+matchup+home`` is a distinct row in the ablation and
   not a duplicate of ``rating+home``. Both of those readings of the matchup
   terms are likewise near-exact (~0.1), since the left side is built from OFF
   and DEF and the right side from the separately published NET.
   ``Tests/test_logreg_rolling.py`` asserts all of this against the real data,
   including the 0.1 bound, so the claim cannot quietly rot.

5. HALF-LIFE IS A REPORTED AVERAGE, NOT A TUNED NUMBER. Features are strictly
   causal, so they are computed once over the full 13,199-game history and then
   walk-forward split; a feature for game t cannot see game t. ``halflife=5``
   and ``halflife=15`` are both reported per split. Picking the better one by
   reading the test blocks is tuning on the test set - the module reports the
   per-split numbers for both and, where a headline number needs a choice, the
   choice is made by an inner walk-forward inside the training rows only.

CAVEATS THAT BELONG IN EVERY TABLE PRODUCED BY THIS MODULE
----------------------------------------------------------
* Accuracy is reported but is not a judge. The home team wins 56.4% of the time
  here, so "always pick home" scores 0.564.
* This measures PREDICTION QUALITY. It says nothing about betting value. See
  Vandenbruaene et al. 2022 (J Sports Econ 23(7)), who surveyed 628 published
  NBA implementations and never rejected unprofitability once the hurdle was
  corrected for multiple testing (|z| > 3). Nothing here is a profit claim.
* Per-split log-loss differences of ~0.002 are inside the noise band; see
  ``paired_bootstrap_delta`` for whether a given delta separates from zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.DataProviders import nba_data as nd

# --------------------------------------------------------------------------- #
# Feature recipes
# --------------------------------------------------------------------------- #

# (name, home column, away column) -> home column minus away column.
RATING_PAIRS: list[tuple[str, str, str]] = [
    ("net_diff", "home_ewm_net_rating", "away_ewm_net_rating"),
    ("margin_diff", "home_ewm_margin", "away_ewm_margin"),
    ("off_diff", "home_ewm_off_rating", "away_ewm_off_rating"),
    ("def_diff", "home_ewm_def_rating", "away_ewm_def_rating"),
    ("ts_diff", "home_ewm_ts_pct", "away_ewm_ts_pct"),
    ("efg_diff", "home_ewm_efg", "away_ewm_efg"),
    ("tov_diff", "home_ewm_tov", "away_ewm_tov"),
    ("ftr_diff", "home_ewm_ftr", "away_ewm_ftr"),
    ("pace_diff", "home_ewm_pace", "away_ewm_pace"),
]

# The two matchup terms: "my offence against their defence". These are the only
# places a team's offence is paired with the OPPONENT's defence rather than with
# the opponent's offence, and they are what makes the coefficient signs readable.
MATCHUP_PAIRS: list[tuple[str, str, str]] = [
    ("matchup_home", "home_ewm_off_rating", "away_ewm_def_rating"),
    ("matchup_away", "away_ewm_off_rating", "home_ewm_def_rating"),
]

# Rebound percentages. ``rolling_team_features`` cannot produce these: it asks
# the four-factors table for ``ORB_PCT``/``DRB_PCT`` while that table actually
# ships ``OREB_PCT``/``DREB_PCT``, so the columns silently never appear. See
# ``rebound_features`` for how the gap is filled without a second loader.
REBOUND_PAIRS: list[tuple[str, str, str]] = [
    ("orb_diff", "home_ewm_orb", "away_ewm_orb"),
    ("drb_diff", "home_ewm_drb", "away_ewm_drb"),
]

# The raw rolling columns that get shrunk toward a prior when one is supplied.
SHRINKABLE: list[str] = [
    "ewm_net_rating", "ewm_margin", "ewm_off_rating", "ewm_def_rating",
    "ewm_pace", "ewm_ts_pct", "ewm_efg", "ewm_tov", "ewm_ftr",
]

# The home indicator. Constant 1.0 by construction: see note 3 in the docstring.
HOME_COLUMN = "home_adv"

# Never standardised, never centred. Standardising a zero-variance column is a
# division by zero; centring it destroys the home-court coefficient.
NO_SCALE: frozenset[str] = frozenset({HOME_COLUMN})

# The C grid. sklearn's LogisticRegression minimises ``C * SUM_i loss_i +
# 0.5*||w||^2``, so C is not a per-sample inverse penalty: with ~10k training
# rows the summed loss is ~6200, and a "small" C of 0.01 still applies a penalty
# of 1/0.01 = 100 against a loss curvature of order 10k * 0.22 = 2200 - i.e.
# almost nothing. A grid that starts at 0.01 looks like a regularisation path
# and is in fact a flat line. The grid has to reach down to ~1e-5 before the
# penalty is doing anything visible.
C_GRID = (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 100.0)

REST_CAP = 7.0          # days; beyond ~7 a team is simply rested (or it is October)
REST_PIVOT = 4.0        # subtracted so a normally-rested team sits at 0


def _group_rating(with_rebounds: bool = True) -> list[str]:
    # pace_mean is the one symmetric LEVEL kept on purpose: the average of the
    # two teams' pace says how the game will be played (and, through pace-
    # adjusted margin, how it usually ends), which neither side's pace alone
    # says. Every other entry here is a difference.
    cols = [n for n, _, _ in RATING_PAIRS] + ["pace_mean"]
    if with_rebounds:
        cols += [n for n, _, _ in REBOUND_PAIRS]
    return cols


def feature_groups(with_rebounds: bool = True) -> dict[str, list[str]]:
    """Named feature sets. The ablation slices these, not the raw column list."""
    return {
        # Pure team-strength contrasts. Everything the model knows about who is
        # better, nothing about the schedule.
        "rating": _group_rating(with_rebounds),
        # Offence-versus-defence matchup terms.
        "matchup": [n for n, _, _ in MATCHUP_PAIRS],
        # Rest / back-to-back / 3-in-4 at each end, plus travel. Levels, not
        # differences: "this side is travelling 2000 miles" is a level fact.
        "schedule": [
            "rest_home", "rest_away", "b2b_home", "b2b_away",
            "three_home", "three_away", "travel_k",
        ],
        # The same four facts as contrasts. Reported as a separate row because
        # levels and differences answer different questions and the L2 penalty
        # splits a shared direction between them if both are present at once.
        "schedule_diff": [
            "rest_diff", "b2b_diff", "three_diff", "travel_k",
        ],
        # Off and defence WITHOUT net_diff, so the two coefficients are
        # identified. Only meaningful as a coefficient read, never as a
        # performance row.
        "signs": ["off_diff", "def_diff"],
        "home": [HOME_COLUMN],
    }


# Feature set for each reported model. ``DEFAULT_MODELS`` is the results table.
DEFAULT_MODELS: dict[str, list[str]] = {
    "rating+home": ["rating", "home"],
    "rating+matchup+home": ["rating", "matchup", "home"],
    "rating+schedule+home": ["rating", "schedule", "home"],
    "full": ["rating", "matchup", "schedule", "home"],
    "full_sched_diff": ["rating", "matchup", "schedule_diff", "home"],
    "net+home": ["net_diff", HOME_COLUMN],
}


def resolve_features(names: list[str] | str, with_rebounds: bool = True) -> list[str]:
    """Turn group names, named model recipes, or literal column names into an
    ordered column list.

    Accepts, in any mix: a group from ``feature_groups`` (``"rating"``), a
    recipe from ``DEFAULT_MODELS`` (``"full"``), or a bare column (``"net_diff"``).
    """
    if isinstance(names, str):
        names = [names]
    groups = feature_groups(with_rebounds)
    known = {c for cols in groups.values() for c in cols}
    known |= {n for n, _, _ in RATING_PAIRS + MATCHUP_PAIRS + REBOUND_PAIRS}

    def _expand(n: str) -> list[str]:
        if n in groups:
            return groups[n]
        if n in DEFAULT_MODELS:
            return _expand_list(DEFAULT_MODELS[n])
        if n in known:
            return [n]
        raise KeyError(f"unknown feature, group or model {n!r}; "
                       f"known groups: {sorted(groups)}; "
                       f"known models: {sorted(DEFAULT_MODELS)}")

    def _expand_list(ns: list[str]) -> list[str]:
        out: list[str] = []
        for n in ns:
            for c in _expand(n):
                if c not in out:
                    out.append(c)
        return out

    return _expand_list(list(names))


# --------------------------------------------------------------------------- #
# Preprocessing - train-only, and only ever train-only
# --------------------------------------------------------------------------- #

class FeaturePipeline:
    """Median-impute then z-score. Statistics are fitted on the rows given to
    ``fit`` and are never updated afterwards.

    Kept deliberately explicit (rather than a ``ColumnTransformer``) because the
    whole point is that a caller can see, in one place, that nothing here touches
    the test rows. ``fit`` records a fingerprint of what it saw so a test can
    assert the provenance of every parameter.
    """

    def __init__(self, columns: list[str] | None = None,
                 no_scale: frozenset[str] = NO_SCALE):
        self.columns = list(columns) if columns is not None else None
        self.no_scale = frozenset(no_scale)

    def fit(self, X: pd.DataFrame, y=None) -> "FeaturePipeline":
        X = self._check(X, reset=True)
        # Median is taken per column over the training rows. ``skipna`` is on, so
        # a training column that is entirely NaN gets NaN back; that is a bug in
        # the feature set, not something to paper over with 0, so it is filled
        # with 0.0 and flagged rather than silently producing inf.
        self.medians_ = X.median(numeric_only=True)
        all_nan = [c for c in X.columns if bool(X[c].isna().all())]
        self.degenerate_columns_ = all_nan
        self.medians_ = self.medians_.fillna(0.0)
        self.no_scale_ = self.no_scale & set(X.columns)

        self.means_ = {}
        self.scales_ = {}
        for c in X.columns:
            col = X[c].to_numpy(dtype=float)
            if c in self.no_scale_:
                # Passed through untouched, so a constant column stays constant
                # and its coefficient stays identifiable.
                self.means_[c] = 0.0
                self.scales_[c] = 1.0
                continue
            if c in all_nan:
                # Nothing to average and nothing to divide by. Left on the
                # identity transform rather than letting nanmean/nanstd return
                # NaN and poison every downstream prediction.
                self.means_[c] = 0.0
                self.scales_[c] = 1.0
                continue
            mu = float(np.nanmean(col))
            sd = float(np.nanstd(col))
            self.means_[c] = mu
            # A constant column has nothing to standardise against. Leave it on
            # its natural scale rather than dividing by ~0, which would blow the
            # column up into pure noise.
            self.scales_[c] = sd if sd > 1e-12 else 1.0

        self.n_fit_rows_ = int(len(X))
        self.fit_index_hash_ = pd.util.hash_pandas_object(
            X.reset_index(drop=True), index=False).sum()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "means_"):
            raise RuntimeError("FeaturePipeline.transform called before fit")
        X = self._check(X, reset=False)
        out = pd.DataFrame(index=X.index)
        for c in X.columns:
            med = self.medians_.get(c, 0.0)
            v = X[c].to_numpy(dtype=float)
            # Imputation happens BEFORE scaling, so a missing entry lands on the
            # training mean (0 after centring) rather than somewhere arbitrary.
            v = np.where(np.isnan(v), float(med), v)
            out[c] = (v - self.means_[c]) / self.scales_[c]
        return out

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _check(self, X: pd.DataFrame, reset: bool) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"expected a DataFrame, got {type(X).__name__}")
        if self.columns is not None:
            if reset:
                missing = [c for c in self.columns if c not in X.columns]
                if missing:
                    raise KeyError(f"missing feature columns: {missing}")
                X = X[self.columns]
            else:
                missing = [c for c in self.columns if c not in X.columns]
                if missing:
                    raise KeyError(f"missing feature columns: {missing}")
                X = X[self.columns]
        elif reset:
            X = X.copy()
        return X

    def params(self) -> pd.DataFrame:
        """Fitted preprocessing parameters, for reporting and for tests."""
        return pd.DataFrame({
            "median": self.medians_,
            "mean": pd.Series(self.means_),
            "scale": pd.Series(self.scales_),
        })

    def __repr__(self) -> str:
        n = "None" if self.columns is None else len(self.columns)
        return f"<FeaturePipeline columns={n} fitted={hasattr(self, 'means_')}>"


# --------------------------------------------------------------------------- #
# The model
# --------------------------------------------------------------------------- #

class RollingLogreg:
    """L2 logistic regression on contrast features. Small and explicit.

    INTERCEPT HANDLING, which is not a detail. Every game has exactly one home
    team, so ``home_adv`` is the constant 1.0. That makes it linearly dependent
    on the intercept: with ``fit_intercept=True`` the two fight, the solver
    splits the home effect arbitrarily between them, and the reported
    ``home_adv`` coefficient comes out at exactly zero - which looks like "the
    model found no home-court effect" when it actually found one and put it in
    the intercept. So when ``home_adv`` is in the feature set the intercept is
    switched OFF and ``home_adv`` *is* the intercept: its coefficient is the
    league-average home log-odds, directly comparable with the other
    coefficients and directly interpretable as the home-court term.

    Pass ``fit_intercept`` explicitly to override.
    """

    def __init__(self, C: float = 1.0, features: list[str] | None = None,
                 max_iter: int = 2000, penalty: str = "l2",
                 fit_intercept: bool | None = None):
        self.C = float(C)
        self.features = list(features) if features is not None else None
        self.max_iter = max_iter
        self.penalty = penalty
        if fit_intercept is None:
            fit_intercept = not (self.features is not None
                                 and HOME_COLUMN in self.features)
        self.fit_intercept = bool(fit_intercept)

    def fit(self, X: pd.DataFrame, y) -> "RollingLogreg":
        y = np.asarray(y).astype(int)
        self.pipeline_ = FeaturePipeline(self.features).fit(X, y)
        Xs = self.pipeline_.transform(X)
        self.model_ = _logreg(C=self.C, max_iter=self.max_iter,
                              penalty=self.penalty,
                              fit_intercept=self.fit_intercept)
        Xn = Xs.to_numpy(dtype=float)
        self.model_.fit(Xn, y)
        # Cached for the (approximate) Wald SEs in `coefficients()`. The design
        # matrix is small and thrown away with the model.
        self.model_._last_X = Xn
        self.n_features_ = Xs.shape[1]
        self.n_train_ = int(len(y))
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.pipeline_.transform(X)
        return self.model_.predict_proba(Xs.to_numpy(dtype=float))[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) > 0.5).astype(int)

    def decision_function(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self.pipeline_.transform(X)
        return self.model_.decision_function(Xs.to_numpy(dtype=float))

    def coefficients(self) -> pd.DataFrame:
        """Coefficients in STANDARDISED units, with a standard error.

        The SE is the OBSERVED INFORMATION one. sklearn minimises
        ``C * sum_i loss_i(w) + 0.5 * ||w||^2``, whose Hessian at the fitted
        weights is ``C * X' W X + I`` with ``W = diag(p_i (1 - p_i))``. Ignoring
        W - as a naive ``(X'X)^-1`` does - overstates the information in a
        binary design by roughly a factor of 1/0.22 and shrinks every SE to about
        half its true size, which is the direction that manufactures false
        confidence.

        Two honest caveats: this is still the conditional SE, and the L2 term
        biases coefficients toward zero without that bias appearing in the SE, so
        these are for reading signs and relative magnitudes, never for claiming
        significance. The walk-forward deltas are the significance test.
        """
        names = self.pipeline_.columns
        beta = np.asarray(self.model_.coef_)[0]
        Xs = self.model_._last_X
        p = self.model_.predict_proba(Xs)[:, 1]
        w = np.clip(p * (1.0 - p), 1e-10, None)
        A = self.C * (Xs.T * w) @ Xs + np.eye(Xs.shape[1])
        cov = np.linalg.pinv(A)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0, beta / se, np.nan)
        return pd.DataFrame({
            "feature": names,
            "coef": beta,
            "se": se,
            "z": z,
            # Per-SD effect in odds-ratio form. Readable for the signs.
            "odds_ratio_per_sd": np.exp(beta),
        }).set_index("feature")

    def intercept(self) -> float:
        """The separate intercept, or 0.0 when ``home_adv`` is acting as it."""
        if not self.fit_intercept:
            return 0.0
        return float(self.model_.intercept_[0])

    def home_edge_logodds(self) -> float:
        """The home-court effect in log-odds, however this model represents it.

        With ``home_adv`` in the feature set the intercept is switched off and
        this is its coefficient; otherwise it is the intercept. Either way the
        number to sanity-check is POSITIVE and near log(0.564 / 0.436) = 0.257,
        which is the entire league-wide home advantage.
        """
        if not self.fit_intercept:
            names = list(self.pipeline_.columns)
            return float(np.asarray(self.model_.coef_)[0][names.index(HOME_COLUMN)])
        return float(self.model_.intercept_[0])

    def home_edge_prob(self) -> float:
        return float(1.0 / (1.0 + np.exp(-self.home_edge_logodds())))

    def __repr__(self) -> str:
        return (f"<RollingLogreg C={self.C:g} n_features={getattr(self, 'n_features_', None)}"
                f" n_train={getattr(self, 'n_train_', None)}>")


def _logreg(C: float, max_iter: int, penalty: str, fit_intercept: bool = True):
    """sklearn's LogisticRegression, isolated so tests can monkeypatch it and so
    the solver choice is one edit rather than a search.

    ``penalty`` is deliberately NOT passed for "l2": sklearn deprecated that
    argument in 1.8 and L2 is the default, so naming it only produces a
    FutureWarning on every single fit. An explicit non-default penalty still
    goes through, for anyone who wants it.
    """
    from sklearn.linear_model import LogisticRegression
    kw = {}
    if penalty != "l2":
        kw["penalty"] = penalty
    return LogisticRegression(
        C=C, solver="lbfgs", max_iter=max_iter, fit_intercept=fit_intercept,
        tol=1e-8, **kw,
    )


# --------------------------------------------------------------------------- #
# Feature assembly
# --------------------------------------------------------------------------- #

def season_games_played(games: pd.DataFrame) -> pd.DataFrame:
    """Games this team has played IN THIS SEASON before this game. 0 on a team's
    season opener.

    The rolling builder's ``games_played`` is a career cumcount across all
    seasons, so it reads ~800 in October of the tenth season. That is useless for
    deciding how much to trust a team's early-season form, which is the entire
    reason a prior is worth having.
    """
    g = games.reset_index(drop=True).copy()
    g["home_n"] = g.groupby(["SEASON", "home"], sort=False).cumcount()
    g["away_n"] = g.groupby(["SEASON", "away"], sort=False).cumcount()
    return g[["home_n", "away_n"]]


def rest_features(games: pd.DataFrame, rest_df: pd.DataFrame) -> pd.DataFrame:
    """Schedule features from the shared rest/travel builder.

    Two deviations from the raw columns, both forced by the data and both stated
    rather than buried:

    * Rest is capped at ``REST_CAP`` days. The raw column reaches 288 because it
      measures days since a team's last game ACROSS the offseason, so the first
      game of every October is scored as "287 days of rest". That is a season
      indicator wearing a rest indicator's clothes, and it would be picked up by
      any regularisation path as if it meant something.
    * Travel is divided by 1000 so its coefficient is per 1000 miles and does not
      need a scientific-notation footnote.
    """
    out = pd.DataFrame(index=games.index)
    h_rest = np.clip(rest_df["home_rest_days"].to_numpy(dtype=float), 1.0, REST_CAP)
    a_rest = np.clip(rest_df["away_rest_days"].to_numpy(dtype=float), 1.0, REST_CAP)
    out["rest_home"] = h_rest - REST_PIVOT
    out["rest_away"] = a_rest - REST_PIVOT
    out["rest_diff"] = h_rest - a_rest
    out["b2b_home"] = rest_df["home_back_to_back"].to_numpy(dtype=float)
    out["b2b_away"] = rest_df["away_back_to_back"].to_numpy(dtype=float)
    out["b2b_diff"] = out["b2b_home"] - out["b2b_away"]
    out["three_home"] = rest_df["home_in_3_of_4"].to_numpy(dtype=float)
    out["three_away"] = rest_df["away_in_3_of_4"].to_numpy(dtype=float)
    out["three_diff"] = out["three_home"] - out["three_away"]
    out["travel_k"] = rest_df["travel_miles"].to_numpy(dtype=float) / 1000.0
    return out


def assemble_features(games: pd.DataFrame, roll: pd.DataFrame,
                      rest: pd.DataFrame) -> pd.DataFrame:
    """Build the model matrix. PURE: takes the shared builders' outputs as plain
    frames and derives every contrast explicitly.

    No file, cache or global is read here, which is what lets the tests drive it
    with synthetic data where the true relationship is known by construction.
    """
    n = len(games)
    for nm, df in (("roll", roll), ("rest", rest)):
        if len(df) != n:
            raise ValueError(f"{nm} has {len(df)} rows, expected {n}")
    roll = roll.reset_index(drop=True)
    out = pd.DataFrame(index=games.reset_index(drop=True).index)

    for name, hc, ac in RATING_PAIRS + MATCHUP_PAIRS:
        if hc in roll.columns and ac in roll.columns:
            out[name] = (roll[hc].to_numpy(dtype=float)
                         - roll[ac].to_numpy(dtype=float))

    # The one symmetric level worth keeping: expected total pace. Neither side's
    # pace alone says anything about who wins, but the pair does say something
    # about how the game will be played.
    if "home_ewm_pace" in roll.columns:
        out["pace_mean"] = 0.5 * (roll["home_ewm_pace"].to_numpy(dtype=float)
                                  + roll["away_ewm_pace"].to_numpy(dtype=float))

    for name, hc, ac in REBOUND_PAIRS:
        if hc in roll.columns and ac in roll.columns:
            out[name] = (roll[hc].to_numpy(dtype=float)
                         - roll[ac].to_numpy(dtype=float))

    sched = rest_features(games.reset_index(drop=True), rest.reset_index(drop=True))
    for c in sched.columns:
        out[c] = sched[c].to_numpy(dtype=float)

    # The home indicator, always 1.0. Present so the model has an explicit
    # home-court term to read, rather than one hidden in the intercept.
    out[HOME_COLUMN] = 1.0
    return out


def rebound_features(games: pd.DataFrame, halflife: float = 10.0) -> pd.DataFrame:
    """ORB%/DRB% exponentially-weighted form, home_/away_ aligned to ``games``.

    Why this exists: ``rolling_team_features`` asks the four-factors table for
    ``ORB_PCT`` and ``DRB_PCT``, but that table ships ``OREB_PCT`` and
    ``DREB_PCT``. The names do not match, the ``if col in m.columns`` guard
    skips both, and the spec's fourth factor silently never appears in the
    feature set. This fills the gap by calling the SHARED loader
    (``load_team_advanced``) and the SHARED causal EWM primitive
    (``_ewm_by_team``) - not by re-implementing either. Same data, same
    causality guarantee (shift-within-team, then weight), two extra columns.

    Rebound DIFFERENCES are near-useless on their own anyway - offensive
    rebounding barely moves win probability - so this is reported as an ablation,
    not in the headline feature set.
    """
    _ewm = getattr(nd, "_ewm_by_team", None)
    if _ewm is None:                       # shared module was refactored
        return pd.DataFrame(index=games.index)
    keep_ids = set(games["GAME_ID"])
    # ORB% lives in the four-factors table. DREB% is NOT reachable from any PUBLIC
    # shared loader: `load_team_fourfactors` has OREB_PCT but no DREB_PCT, and
    # `load_team_advanced` strips OREB_PCT/DREB_PCT down to eight columns at
    # construction. So DREB% is dropped rather than read behind the loader's
    # back. One of the spec's four factors is therefore genuinely unavailable,
    # which is worth saying out loud rather than papering over.
    ff = nd.load_team_fourfactors()
    name_map = {}
    for src, dst in (("OREB_PCT", "ewm_orb"), ("DREB_PCT", "ewm_drb")):
        if src in ff.columns:
            name_map[src] = dst
    if not name_map:
        return pd.DataFrame(index=games.index)
    cols = list(name_map)
    ff = ff[ff["GAME_ID"].isin(keep_ids)][["GAME_ID", "TEAM_ABBREVIATION"] + cols]
    # The shared builder dedups before merging the four-factors table; without
    # that the same fan-out bug it guards against reappears here (26398 rows out
    # of 13199 games, i.e. exactly doubled).
    ff = ff.drop_duplicates(subset=["GAME_ID", "TEAM_ABBREVIATION"])
    dates = games[["GAME_ID", "GAME_DATE"]].drop_duplicates("GAME_ID")
    m = ff.merge(dates, on="GAME_ID", how="left")
    m = m.rename(columns={"TEAM_ABBREVIATION": "team"}).sort_values(["team", "GAME_DATE"])

    for c, nm in name_map.items():
        m[nm] = _ewm(m, c, halflife)

    feat = ["GAME_ID", "team"] + list(name_map.values())
    h = m[m.team.isin(games.home.unique())][feat].rename(
        columns={c: f"home_{c}" for c in name_map.values()})
    a = m[m.team.isin(games.away.unique())][feat].rename(
        columns={c: f"away_{c}" for c in name_map.values()})
    # Both joins MUST key on the team as well as the game. Almost every team
    # appears as both home and away somewhere in the sample, so filtering by
    # `home.unique()` / `away.unique()` leaves BOTH rows of every game in each
    # frame; joining on GAME_ID alone then doubles the output (26398 rows from
    # 13199 games) and every contrast below silently averages both teams in.
    out = games[["GAME_ID", "home", "away"]].reset_index(drop=True).merge(
        h, left_on=["GAME_ID", "home"], right_on=["GAME_ID", "team"], how="left")
    out = out.merge(a, left_on=["GAME_ID", "away"], right_on=["GAME_ID", "team"],
                    how="left", suffixes=("", "_a"))
    out = out.drop(columns=[c for c in ("team", "team_a") if c in out.columns])
    if len(out) != len(games):
        raise AssertionError(f"rebound_features produced {len(out)} rows for "
                             f"{len(games)} games - join fanned out")
    return out.drop(columns=["GAME_ID", "home", "away"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Prior-season shrinkage
# --------------------------------------------------------------------------- #

def prior_season_features(games: pd.DataFrame, halflife: float = 10.0,
                          min_games: int = 10) -> pd.DataFrame:
    """End-of-PRIOR-season EWM values, keyed by (SEASON, team).

    The row keyed ``(S, team)`` holds that team's state at the END OF THE LAST
    SEASON BEFORE S - never its own season. Getting that off-by-one is
    catastrophic and invisible, because the resulting number looks perfectly
    reasonable: an October 2015 game would receive BOS's end-of-2015-16 rating,
    which is roughly a full season of future information, and the resulting
    scores would look great. So the implementation walks the seasons in order
    and emits season S's rows from a carry that was frozen BEFORE S started, and
    ``Tests/test_logreg_rolling.py`` pins it against an independently
    recomputed value.

    Teams with no earlier season get NaN, which ``shrink_to_prior`` resolves to
    the league mean. ``prior_n`` is the number of games behind the PRIOR value
    (i.e. games played last season), which is what tells you how much the prior
    is worth.
    """
    g = games.reset_index(drop=True)
    seasons = sorted(g["SEASON"].unique())
    roll = nd.rolling_team_features(g, halflife=halflife, min_games=min_games)
    gp = season_games_played(g)
    g = g.assign(home_n=gp["home_n"].to_numpy(), away_n=gp["away_n"].to_numpy())

    out_rows = []
    # Frozen at the end of the previously-processed season. Starts empty, so
    # the first season emitted carries nothing - which is the truth.
    carry: dict[str, dict] = {}
    for s in seasons:
        idx = np.flatnonzero((g["SEASON"] == s).to_numpy())
        last = {}
        for i in idx:
            last[g["home"].iat[i]] = i
            last[g["away"].iat[i]] = i

        # Emit using the carry from BEFORE this season was touched.
        for team in last:
            prev = carry.get(team, {})
            row = {"SEASON": s, "team": team, "prior_n": int(prev.get("prior_n", 0))}
            for col in SHRINKABLE:
                row[col] = float(prev.get(col, np.nan))
            out_rows.append(row)

        # Only now, with season s fully processed, refreeze the carry.
        for team, i in last.items():
            is_home = g["home"].iat[i] == team
            side = "home_" if is_home else "away_"
            entry = {"prior_n": int(g["home_n"].iat[i] if is_home else g["away_n"].iat[i])}
            for col in SHRINKABLE:
                key = side + col
                entry[col] = float(roll[key].iat[i]) if key in roll.columns else np.nan
            carry[team] = entry

    return pd.DataFrame(out_rows)


def league_prior_table(prior: pd.DataFrame) -> pd.DataFrame:
    """Per-season league mean of the prior-season values.

    This is the fallback target for a team with no prior at all (expansion, or
    the first season in the data). It is a causal scalar: every input row is an
    end-of-ANOTHER-season number, and the mean is taken across teams within the
    target season's own key, so it contains no information from the season being
    predicted. For the very first season in the data every input is NaN and the
    mean is NaN, which ``shrink_to_prior`` then treats as 0.0 - the correct
    answer, since league-average net rating is zero by construction.
    """
    cols = [c for c in prior.columns if c not in ("SEASON", "team", "prior_n")]
    return prior.groupby("SEASON")[cols].mean().reset_index()


def shrink_to_prior(roll: pd.DataFrame, games: pd.DataFrame,
                    prior: pd.DataFrame | None,
                    k: float, cols: list[str] | None = None) -> pd.DataFrame:
    """Regress each team's in-season EWM toward its prior-season value.

        shrunk = (n * ewm + k * prior) / (n + k)

    with ``n`` the team's games played THIS SEASON and ``k`` the prior's assumed
    precision in pseudo-games. ``k = 0`` disables shrinkage entirely, which is
    the "no prior" arm. ``n = 0`` (a season opener) returns the prior exactly.
    A team with no prior at all shrinks toward the league mean instead, which is
    the only honest fallback - the alternative, leaving it at NaN, hands the
    imputer the job and biases the whole column.

    The league-mean fallback is built from ``prior`` for the SAME season, so it
    also contains no information from the season being predicted.
    """
    cols = list(cols or SHRINKABLE)
    out = roll.reset_index(drop=True).copy()
    if k <= 0 or prior is None:
        return out

    g = games.reset_index(drop=True)
    gp = season_games_played(g)
    n_home = gp["home_n"].to_numpy(dtype=float)
    n_away = gp["away_n"].to_numpy(dtype=float)

    pr = prior.set_index(["SEASON", "team"])
    lg = league_prior_table(prior).set_index("SEASON")
    teams_h = g["home"].to_numpy()
    teams_a = g["away"].to_numpy()
    seasons = g["SEASON"].to_numpy()

    for col in cols:
        for side, teams, n in (("home_", teams_h, n_home), ("away_", teams_a, n_away)):
            key = side + col
            if key not in out.columns:
                continue
            cur = out[key].to_numpy(dtype=float)
            target = np.empty(len(g), dtype=float)
            for i in range(len(g)):
                try:
                    target[i] = pr.at[(seasons[i], teams[i]), col]
                except KeyError:
                    target[i] = lg.at[seasons[i], col] if seasons[i] in lg.index else np.nan
            target = np.where(np.isnan(target), 0.0, target)
            # A NaN in-season value means the team has no in-season history; the
            # prior is strictly better information than NaN here, and the
            # imputer would replace it with a median that is not.
            cur = np.where(np.isnan(cur), 0.0, cur)
            out[key] = (n * cur + k * target) / (n + k)
    return out


# --------------------------------------------------------------------------- #
# Experiment harness
# --------------------------------------------------------------------------- #

def prune_features(X: pd.DataFrame, features: list[str]) -> list[str]:
    """Drop declared features this build of the shared module cannot supply.

    ``feature_groups`` names what the SPEC wants; the shared loaders decide what
    is actually reachable. Reporting the intersection is honest, and keeps a
    missing column from becoming a KeyError deep inside a fit.
    """
    return [f for f in features if f in X.columns]


def load_design(halflife: float = 10.0, min_games: int = 5,
                with_prior: bool = False, prior_k: float = 0.0,
                with_rebounds: bool = True,
                games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Everything from the shared module, assembled into one design matrix."""
    g = nd.build_games() if games is None else games
    roll = nd.rolling_team_features(g, halflife=halflife, min_games=min_games)
    rest = nd.rest_and_travel(g)
    if with_rebounds:
        reb = rebound_features(g, halflife=halflife)
        if len(reb.columns):
            roll = pd.concat(
                [roll.reset_index(drop=True), reb.reset_index(drop=True)], axis=1)
    if with_prior and prior_k > 0:
        roll = shrink_to_prior(roll, g, prior_season_features(g, halflife=halflife),
                               prior_k)
    X = assemble_features(g, roll, rest)
    return X


def constant_baseline(y_train, n_test: int) -> np.ndarray:
    """'Always bet the home team', with the base rate fitted on TRAIN only."""
    p = float(np.mean(np.asarray(y_train).astype(int)))
    p = min(max(p, 1e-6), 1 - 1e-6)
    return np.full(n_test, p)


def walk_forward_eval(X: pd.DataFrame, games: pd.DataFrame, features: list[str],
                      C: float = 1.0, n_splits: int = 5, initial_frac: float = 0.5,
                      gap: int = 0,
                      include_baselines: bool = True) -> pd.DataFrame:
    """Per-split metrics for one feature set at one C.

    Refits EVERYTHING - pipeline included - inside each split. There is no state
    carried across splits, which is the whole point.
    """
    y = games["y"].to_numpy(dtype=int)
    features = prune_features(X, features)
    rows = []
    preds: dict[str, np.ndarray] = {}
    for k, (tr, te) in enumerate(nd.walk_forward_split(
            games, n_splits=n_splits, initial_frac=initial_frac, gap=gap), start=1):
        m = RollingLogreg(C=C, features=features).fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])
        preds[f"split{k}"] = p
        r = nd.evaluate(y[te], p)
        rows.append({"split": k, "model": "model", "C": C, **{
            key: r[key] for key in ("n", "log_loss", "brier", "accuracy", "auc")}})
        if include_baselines:
            pb = constant_baseline(y[tr], len(te))
            rb = nd.evaluate(y[te], pb)
            rows.append({"split": k, "model": "always_home", "C": np.nan, **{
                key: rb[key] for key in ("n", "log_loss", "brier", "accuracy", "auc")}})
    return pd.DataFrame(rows)


def regularisation_path(X: pd.DataFrame, games: pd.DataFrame, features: list[str],
                        C_grid=C_GRID,
                        n_splits: int = 5, initial_frac: float = 0.5) -> pd.DataFrame:
    """Metrics across the C grid, per split. Report the whole path, not the
    argmin - the argmin over these test blocks is itself an optimistic number."""
    y = games["y"].to_numpy(dtype=int)
    features = prune_features(X, features)
    splits = nd.walk_forward_split(games, n_splits=n_splits, initial_frac=initial_frac)
    rows = []
    for k, (tr, te) in enumerate(splits, start=1):
        for C in C_grid:
            m = RollingLogreg(C=C, features=features).fit(X.iloc[tr], y[tr])
            r = nd.evaluate(y[te], m.predict_proba(X.iloc[te]))
            rows.append({"split": k, "C": C, **{key: r[key] for key in
                        ("log_loss", "brier", "auc", "accuracy")}})
    return pd.DataFrame(rows)


def select_C_inner(X: pd.DataFrame, games: pd.DataFrame, features: list[str],
                   C_grid=C_GRID, n_inner: int = 3,
                   initial_frac: float = 0.5) -> float:
    """Pick C using ONLY the rows given here, via an inner walk-forward.

    The point is that the caller passes the OUTER split's training rows and
    nothing else, so the outer test block cannot influence the choice. Inside,
    the split is delegated to the shared ``walk_forward_split`` rather than
    re-derived here - two hand-rolled index expressions is how a harness ends up
    quietly disagreeing with the one everything else reports against.

    Returns the C with the best mean inner log loss. Ties go to the SMALLER C
    (the grid is ascending and ``<`` is strict), which is the conservative
    choice: when the path is flat, prefer the more shrunk model.
    """
    y = games["y"].to_numpy(dtype=int)
    features = prune_features(X, features)
    inner = nd.walk_forward_split(games, n_splits=n_inner,
                                  initial_frac=initial_frac)
    if not inner:
        return float(C_grid[0])
    best_C, best_ll = float(C_grid[0]), np.inf
    for C in C_grid:
        lls = []
        for a, b in inner:
            m = RollingLogreg(C=C, features=features).fit(X.iloc[a], y[a])
            p = np.clip(m.predict_proba(X.iloc[b]), 1e-9, 1 - 1e-9)
            lls.append(-np.mean(np.log(p) * y[b] + np.log(1 - p) * (1 - y[b])))
        if lls and float(np.mean(lls)) < best_ll:
            best_ll, best_C = float(np.mean(lls)), float(C)
    return best_C


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def bootstrap_metric_se(y, p, groups=None, n_boot: int = 500, seed: int = 0) -> dict:
    """Percentile SE for log loss / Brier / AUC, resampling GAMES within the
    given group (default: the game date).

    Resampling games i.i.d. would treat two teams on the same night as
    independent, and the walk-forward blocks are serially correlated anyway, so
    both of those make a naive SE smaller than it should be. The date-clustered
    bootstrap is the honest cheap version; it still does not fix the
    cross-block dependence, which is why every per-split number is reported
    separately rather than pooled into one headline.
    """
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    rng = np.random.default_rng(seed)
    if groups is None:
        groups = np.arange(len(y))
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_by_g = [np.flatnonzero(groups == g) for g in uniq]

    def _stats(ii):
        yy, pp = y[ii], p[ii]
        ll = -np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp))
        br = float(np.mean((pp - yy) ** 2))
        try:
            au = roc_auc_score(yy, pp)
        except ValueError:
            au = np.nan
        return ll, br, au

    draws = np.empty((n_boot, 3))
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        ii = np.concatenate([idx_by_g[k] for k in pick])
        draws[b] = _stats(ii)
    out = {}
    for j, nm in enumerate(("log_loss", "brier", "auc")):
        out[f"{nm}_se"] = float(np.nanstd(draws[:, j], ddof=1))
    out["n_boot"] = n_boot
    return out


def pooled_predictions(X: pd.DataFrame, games: pd.DataFrame, features: list[str],
                       C: float = 1.0, n_splits: int = 5,
                       initial_frac: float = 0.5) -> dict:
    """Out-of-sample predictions for every game after the training window,
    concatenated in date order. Used for pooled metrics and for the paired
    bootstrap against a baseline."""
    y = games["y"].to_numpy(dtype=int)
    features = prune_features(X, features)
    p_model = np.full(len(games), np.nan)
    p_base = np.full(len(games), np.nan)
    for tr, te in nd.walk_forward_split(games, n_splits=n_splits,
                                        initial_frac=initial_frac):
        m = RollingLogreg(C=C, features=features).fit(X.iloc[tr], y[tr])
        p_model[te] = m.predict_proba(X.iloc[te])
        p_base[te] = constant_baseline(y[tr], len(te))
    keep = ~np.isnan(p_model)
    return {"y": y[keep], "p_model": p_model[keep], "p_base": p_base[keep],
            "dates": games["GAME_DATE"].to_numpy()[keep]}


def format_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    return df.to_string(index=False, float_format=lambda v: floatfmt.format(v))


def calibration_table(cal: list[dict]) -> str:
    rows = [f"  [{c['lo']:.1f},{c['hi']:.1f})  n={c['n']:5d}  "
            f"pred={c['mean_pred']:.4f}  actual={c['actual']:.4f}  "
            f"gap={c['mean_pred'] - c['actual']:+.4f}" for c in cal]
    return "\n".join(rows)


def describe_results(model_metrics: pd.DataFrame, path: pd.DataFrame | None = None,
                     coef: pd.DataFrame | None = None,
                     deltas: dict | None = None,
                     bootstrap: dict | None = None,
                     cal: list[dict] | None = None) -> None:
    """Print a report block. Kept as one function so the same numbers appear
    wherever this module is run from."""
    print("\n" + "=" * 78)
    print("PER-SPLIT METRICS")
    print("=" * 78)
    print(format_table(model_metrics))
    for c in ("log_loss", "brier", "auc"):
        if c in model_metrics.columns:
            agg = model_metrics.groupby("model")[c].agg(["mean", "std", "count"])
            agg["se"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
            print(f"\n{c} across splits (mean +/- SE across {int(agg['count'].iloc[0])} splits):")
            print(agg[["mean", "se"]].to_string(float_format=lambda v: f"{v:.4f}"))
    if path is not None:
        print("\n" + "=" * 78)
        print("REGULARISATION PATH (test-block log loss by C)")
        print("=" * 78)
        piv = path.pivot_table(index="C", values="log_loss", aggfunc="mean")
        piv["sd_across_splits"] = path.groupby("C")["log_loss"].std()
        print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
        print("NOTE: the minimum of this curve is chosen on the test blocks and is")
        print("therefore optimistic. The honest C is picked inside the training rows")
        print("only (select_C_inner) and reported separately.")
    if coef is not None:
        print("\n" + "=" * 78)
        print("COEFFICIENTS (standardised units, `signs` model: off/def without")
        print("net_diff, so the two are identified)")
        print("=" * 78)
        print(format_table(coef.reset_index()))
    if deltas:
        print("\n" + "=" * 78)
        print("PAIRED BOOTSTRAP vs always_home")
        print("=" * 78)
        print("paired_bootstrap_delta(y, a, b) returns loss(a) - loss(b). A POSITIVE")
        print("delta means the SECOND argument is BETTER; equivalently a negative")
        print("delta here means the MODEL is better. Every delta below is quoted as")
        print("logloss(model) - logloss(always_home), so read NEGATIVE as good.")
        for name, d in deltas.items():
            sig = "" if (d["ci_lo"] <= 0 <= d["ci_hi"]) else "  (excludes zero)"
            better = "model better" if d["delta"] < 0 else "baseline better"
            print(f"  {name:<28} delta={d['delta']:+.5f}  "
                  f"95% CI [{d['ci_lo']:+.5f}, {d['ci_hi']:+.5f}]  "
                  f"P(delta>0)={d['p_positive']:.3f}  -> {better}{sig}")
    if bootstrap:
        print("\nPooled out-of-sample metric SEs (date-clustered bootstrap):")
        for k, v in bootstrap.items():
            print(f"  {k}: log_loss_se={v['log_loss_se']:.5f}  "
                  f"brier_se={v['brier_se']:.5f}  auc_se={v['auc_se']:.5f}")
    if cal:
        print("\nCalibration (pooled out-of-sample):")
        print(calibration_table(cal))


def main(argv: list[str] | None = None) -> int:
    """Run the full walk-forward evaluation and print the report."""
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--halflife", type=float, default=10.0)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--initial-frac", type=float, default=0.5)
    p.add_argument("--ablate", action="store_true",
                   help="run every feature-set ablation")
    p.add_argument("--path", action="store_true", help="print the C path")
    p.add_argument("--signs", action="store_true",
                   help="print identified off/def coefficients")
    p.add_argument("--calibration", action="store_true")
    args = p.parse_args(argv)

    g = nd.build_games()
    y = g["y"].to_numpy(dtype=int)
    print(f"games={len(g):,}  home win rate={y.mean():.4f}  "
          f"{g.GAME_DATE.min().date()} .. {g.GAME_DATE.max().date()}")

    X = load_design(halflife=args.halflife)
    print(f"design matrix: {X.shape[0]} x {X.shape[1]}")

    kw = dict(n_splits=args.splits, initial_frac=args.initial_frac)
    metrics = walk_forward_eval(X, g, resolve_features(DEFAULT_MODELS["full"]),
                                C=args.C, **kw)

    path = regularisation_path(X, g, resolve_features(DEFAULT_MODELS["full"]),
                               **kw) if args.path else None
    coef = None
    if args.signs:
        f = resolve_features(["signs", "home"])
        tr, te = nd.walk_forward_split(g, n_splits=args.splits,
                                       initial_frac=args.initial_frac)[-1]
        coef = RollingLogreg(C=args.C, features=f).fit(X.iloc[tr], y[tr]).coefficients()

    deltas, boot, cal = None, None, None
    pp = pooled_predictions(X, g, resolve_features(DEFAULT_MODELS["full"]),
                            C=args.C, **kw)
    base = nd.evaluate(pp["y"], pp["p_base"])
    model = nd.evaluate(pp["y"], pp["p_model"])
    deltas = {"full vs always_home":
              nd.paired_bootstrap_delta(pp["y"], pp["p_model"], pp["p_base"])}
    boot = {"full": bootstrap_metric_se(pp["y"], pp["p_model"],
                                        pp["dates"].astype("datetime64[D]").astype(str)),
            "always_home": bootstrap_metric_se(pp["y"], pp["p_base"],
                                               pp["dates"].astype("datetime64[D]").astype(str))}
    cal = model["calibration"] if args.calibration else None

    print("\nPooled out-of-sample (all post-training-window games, concatenated):")
    print(f"  full         log_loss={model['log_loss']:.4f}  brier={model['brier']:.4f}  "
          f"auc={model['auc']:.4f}  acc={model['accuracy']:.4f}  n={model['n']}")
    print(f"  always_home  log_loss={base['log_loss']:.4f}  brier={base['brier']:.4f}  "
          f"auc={base['auc']:.4f}  acc={base['accuracy']:.4f}  n={base['n']}")

    describe_results(metrics, path=path, coef=coef, deltas=deltas,
                     bootstrap=boot, cal=cal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())