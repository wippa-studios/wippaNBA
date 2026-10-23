"""Pace-first decomposition: predict the parts of a score, not the score itself.

Pre-registered in results/experiments.jsonl as ``C44-prereg-spread-total`` BEFORE
this module was ever fit (``python scripts/c44_spread_total.py --prereg``).
That entry governs. This module is the C1 arm of issue #44.

THE HYPOTHESIS (C1)
-------------------
``margin_total`` predicts the margin and the total directly, from two separate
design matrices (home-minus-away contrasts for the margin, sums for the total).
Nothing ties the two predictions together: the fitted margin and the fitted
total can imply a negative away score and nothing complains. C1 asks whether
predicting the GAME MECHANICS first - possessions, then points-per-possession
for each side - produces better margins AND better totals than the monolithic
ridge, on identical walk-forward rows:

    poss_hat        = f(pace, rest, travel, experience)
    ppp_home_hat    = g(home offence, away defence, ...)
    ppp_away_hat    = g(away offence, home defence, ...)
    mu_home         = poss_hat * ppp_home_hat
    mu_away         = poss_hat * ppp_away_hat
    margin_hat      = mu_home - mu_away
    total_hat       = mu_home + mu_away

"Keeping the correlation" is structural, not statistical: margin and total are
derived from the SAME two team-point predictions, so they are mutually
consistent by construction and share the possessions error the way real scores
do. The monolithic model's two ridges share nothing.

TARGETS (fixed here, ex ante)
-----------------------------
Per-game possessions use the Dean Oliver box-score estimate on the shared team
game logs (``nba_data.load_team_games``), which carry FGA/OREB/TOV/FTA per team:

    poss_team = FGA - OREB + TOV + 0.44 * FTA
    poss_game = mean(poss_home_team, poss_away_team)

This is an OUTCOME-derived training target (like points), never a feature, so
it is no more a leak than the score itself. Points-per-possession for a side is
that side's points divided by ``poss_game``.

DESIGN MATRICES (fixed here, ex ante; no variants)
--------------------------------------------------
The pace model sees both teams' pace and the schedule block (11 columns). The
per-side efficiency model is MIRRORED (same 17 columns with the side/opponent
roles swapped), so home and away share one functional form and differ only in
their data and their intercept - the intercept is where home-court efficiency
lives, exactly as ``margin_total`` carries home court as the intercept (a
constant ``is_home`` column would be flattened to zero by the scaler).

Every feature is the shared causal layer: ``nba_data.rolling_team_features``
(halflife 10, min_games 5 - the same builders ``margin_total`` consumes) and
``nba_data.rest_and_travel``. Nothing is fitted outside a training fold except
the strictly-causal feature builders themselves, which are truncation-invariant.

FITTING
-------
Per walk-forward split, on training rows only: median imputer, standard scaler,
ridge with an intercept. The penalty is chosen per model by forward-chaining CV
inside the training rows over the SAME grid and the SAME helper
(``margin_total._select_ridge_alpha``) the monolithic model uses - imported,
not reimplemented, so C1 cannot win by a friendlier tuning convention. The
possessions prediction is clipped to [60, 140] (a declared physical bound; a
ridge extrapolating possessions to 40 or 400 is a numerical accident, not a
prediction) and nothing else is clipped.

NO DISPERSION IS FIT HERE. C1 is scored on margin RMSE and total RMSE only;
the distributional question is C2's, on the monolithic model's predictions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.DataProviders import nba_data as nd
from src.Models import margin_total as mt

__all__ = [
    "POSS_FEATURES",
    "PPP_FEATURES",
    "POSS_CLIP",
    "possessions_target",
    "build_targets",
    "design_poss",
    "design_ppp",
    "PaceFirstModel",
]

# The two teams' pace, experience and the schedule block. Known before tip-off.
POSS_FEATURES: tuple[str, ...] = (
    "home_ewm_pace", "away_ewm_pace",
    "home_games_played", "away_games_played",
    "home_rest_days", "away_rest_days",
    "home_back_to_back", "away_back_to_back",
    "home_in_3_of_4", "away_in_3_of_4",
    "travel_miles",
)

# Per-side efficiency, mirrored across sides: {s} is the side being predicted,
# {o} its opponent. Home offence against away defence and vice versa - the
# matchup is the point, which is why the opponent's DEF rating is in the side's
# own model. in_3_of_4_diff and travel_miles are the same schedule contrasts
# ``margin_total._schedule_features`` uses.
PPP_FEATURES: tuple[str, ...] = (
    "{s}_ewm_off_rating", "{o}_ewm_def_rating",
    "{s}_ewm_ts_pct", "{s}_ewm_efg", "{s}_ewm_tov", "{s}_ewm_ftr",
    "{s}_ewm_margin", "{o}_ewm_margin",
    "{s}_ewm_pace", "{o}_ewm_pace",
    "{s}_games_played",
    "{s}_rest_days", "{o}_rest_days",
    "{s}_back_to_back", "{o}_back_to_back",
    "in_3_of_4_diff", "travel_miles",
)

# Physical bound on the predicted possessions count. Declared, not tuned.
POSS_CLIP = (60.0, 140.0)


def possessions_target(games: pd.DataFrame) -> pd.Series:
    """Dean Oliver box-score possessions per game, aligned to `games` rows.

    poss_team = FGA - OREB + TOV + 0.44 * FTA on the team game log; poss_game is
    the mean of the two teams' estimates (they differ slightly because the
    formula counts team possessions asymmetrically). NaN if either side's box
    score is missing - a game without a box score cannot train this target and
    must fail loudly rather than be silently filled.
    """
    tg = nd.load_team_games()
    tg = tg[tg["GAME_ID"].isin(set(games["GAME_ID"]))].copy()
    for col in ("FGA", "OREB", "TOV", "FTA"):
        if col not in tg.columns:
            raise ValueError(f"team game log has no {col}; possessions cannot "
                             "be estimated from this data")
    tg["poss_team"] = (tg["FGA"] - tg["OREB"] + tg["TOV"]
                       + 0.44 * tg["FTA"]).astype(float)
    per_game = tg.groupby("GAME_ID")["poss_team"].mean()
    out = games["GAME_ID"].map(per_game)
    out.index = games.index
    if out.isna().any():
        raise ValueError(f"{int(out.isna().sum())} game(s) have no possession "
                         "estimate; refusing to fill them")
    return out.rename("poss")


def build_targets(games: pd.DataFrame) -> pd.DataFrame:
    """poss / ppp_home / ppp_away per game, aligned to `games` rows."""
    poss = possessions_target(games)
    out = pd.DataFrame(index=games.index)
    out["poss"] = poss
    out["ppp_home"] = games["home_pts"].astype(float) / poss
    out["ppp_away"] = games["away_pts"].astype(float) / poss
    return out


def design_poss(roll: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """Possessions design matrix: exactly POSS_FEATURES, in order, nothing else."""
    cols = {}
    for c in POSS_FEATURES:
        if c in roll.columns:
            cols[c] = roll[c]
        elif c in sched.columns:
            cols[c] = sched[c]
        else:
            raise KeyError(f"neither roll nor sched has {c!r}")
    return pd.DataFrame(cols, index=roll.index).astype(float)


def design_ppp(roll: pd.DataFrame, sched: pd.DataFrame,
               side: str) -> pd.DataFrame:
    """Per-side efficiency design matrix, mirrored across home/away.

    ``in_3_of_4_diff`` is the schedule contrast ``margin_total.
    _schedule_features`` uses (home minus away) and is built here because it
    exists in neither builder verbatim.
    """
    o = "away" if side == "home" else "home"
    cols = {}
    for name in PPP_FEATURES:
        c = name.format(s=side, o=o)
        if c == "in_3_of_4_diff":
            cols[c] = sched["home_in_3_of_4"] - sched["away_in_3_of_4"]
        elif c in roll.columns:
            cols[c] = roll[c]
        elif c in sched.columns:
            cols[c] = sched[c]
        else:
            raise KeyError(f"neither roll nor sched has {c!r}")
    return pd.DataFrame(cols, index=roll.index).astype(float)


@dataclass
class PaceFirstModel:
    """Three ridges: possessions, home efficiency, away efficiency.

    ``alpha="auto"`` selects each ridge's penalty by forward-chaining CV inside
    the training rows (``margin_total._select_ridge_alpha``, the monolithic
    model's own grid and helper). Fitted quantities are train-only by
    construction: the pipeline steps fit inside ``.fit`` and the fit is only
    ever called on a training block.
    """

    alpha: float | str = "auto"
    n_inner: int = 4

    def fit(self, X_poss: pd.DataFrame, y_poss: np.ndarray,
            X_ppp_home: pd.DataFrame, y_ppp_home: np.ndarray,
            X_ppp_away: pd.DataFrame, y_ppp_away: np.ndarray
            ) -> "PaceFirstModel":
        self.poss_, self.alpha_poss_ = self._fit_one(X_poss, y_poss)
        self.ppp_home_, self.alpha_ppp_home_ = self._fit_one(X_ppp_home,
                                                             y_ppp_home)
        self.ppp_away_, self.alpha_ppp_away_ = self._fit_one(X_ppp_away,
                                                             y_ppp_away)
        return self

    def _fit_one(self, X: pd.DataFrame, y: np.ndarray):
        mt.assert_no_constant_columns(X)
        alpha = (self.alpha if self.alpha != "auto"
                 else mt._select_ridge_alpha(X, np.asarray(y, float),
                                             n_folds=self.n_inner)[0])
        alpha = float(alpha)
        return mt._ridge_pipeline(alpha).fit(X, np.asarray(y, float)), alpha

    def predict(self, X_poss: pd.DataFrame, X_ppp_home: pd.DataFrame,
                X_ppp_away: pd.DataFrame) -> dict[str, np.ndarray]:
        """poss_hat, ppp_hat per side, and the derived team points / margin / total.

        The possessions clip is the only post-processing. mu_home and mu_away
        are NOT clipped: a negative team-point prediction is information about
        the fit, and hiding it inside a clip is how a broken decomposition
        looks reasonable in a report.
        """
        poss_hat = np.clip(self.poss_.predict(X_poss), *POSS_CLIP)
        ppp_home = np.asarray(self.ppp_home_.predict(X_ppp_home), float)
        ppp_away = np.asarray(self.ppp_away_.predict(X_ppp_away), float)
        mu_home = poss_hat * ppp_home
        mu_away = poss_hat * ppp_away
        return {
            "poss_hat": np.asarray(poss_hat, float),
            "ppp_home_hat": ppp_home,
            "ppp_away_hat": ppp_away,
            "mu_home": mu_home,
            "mu_away": mu_away,
            "margin_hat": mu_home - mu_away,
            "total_hat": mu_home + mu_away,
        }
