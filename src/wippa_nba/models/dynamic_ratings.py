"""Dynamic team-strength ratings: Glicko and a state-space Bradley-Terry filter.

The diagnosis behind issue #40 is that a batch fit describes where a team *was*
over a window, not what it is now.  Refitting the same static model on a
growing window is not a dynamic model.  These two systems carry a state that
moves game to game:

* :func:`fit_predict_glicko` -- a Glicko-style online rating with a per-team
  rating deviation (RD).  The RD widens during inactivity and narrows as games
  are observed; the prediction shrinks an uncertain matchup toward a coin flip.
* :func:`fit_predict_state_space_bt` -- a latent Bradley-Terry strength with a
  random-walk prior, updated by an extended Kalman filter.  The state variance
  is the uncertainty term the issue asks for.

Both keep home court **outside** the team rating.  The home advantage is a
fixed, pre-registered number of rating points (70, the repository's Elo
default), never learned into a team's strength.  No parameter is tuned on the
scored rows; the only data-dependent quantities are the online updates
themselves, which use a game only after its prediction has been recorded.

Sign convention: probabilities are P(home wins).  The filters are causal by
construction, and the tests in ``Tests/test_dynamic_ratings.py`` pin that a
future result cannot move an earlier prediction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

__all__ = [
    "GlickoConfig",
    "GlickoRating",
    "StateSpaceConfig",
    "StateSpaceBT",
    "fit_predict_glicko",
    "fit_predict_state_space_bt",
]


def _sigmoid(x: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -35, 35)))


@dataclass(frozen=True)
class GlickoConfig:
    """Fixed Glicko settings, declared before the run.

    ``hfa`` is in the same rating-point units as ``scale`` (400 is the classic
    doubling constant).  ``rd_increase_per_day`` is the inactivity inflation;
    it is applied to the RD before the game is predicted, so a team returning
    from a long layoff is correctly treated as less certain.
    """

    initial_rating: float = 1500.0
    initial_rd: float = 350.0
    hfa: float = 70.0
    scale: float = 400.0
    rd_increase_per_day: float = 10.0
    min_rd: float = 30.0
    max_rd: float = 350.0


class GlickoRating:
    """Online Glicko-style team ratings with an explicit home-court term."""

    def __init__(self, config: GlickoConfig | None = None) -> None:
        self.config = config or GlickoConfig()
        self.rating: dict[str, float] = {}
        self.rd: dict[str, float] = {}
        self.last_date: dict[str, pd.Timestamp] = {}

    def _get(self, team: str) -> tuple[float, float]:
        return (self.rating.get(team, self.config.initial_rating),
                self.rd.get(team, self.config.initial_rd))

    def inflate_inactivity(self, team: str, date: pd.Timestamp) -> None:
        """Widen RD for the days since this team's previous game."""
        last = self.last_date.get(team)
        if last is None:
            return
        days = max(0.0, float((date - last).days))
        mu, rd = self._get(team)
        rd = np.sqrt(rd ** 2 + (self.config.rd_increase_per_day ** 2) * days)
        self.rating[team] = mu
        self.rd[team] = float(np.clip(rd, self.config.min_rd, self.config.max_rd))

    def prob_home(self, home: str, away: str) -> float:
        """P(home wins) from the pre-game state, including both RDs."""
        mu_h, rd_h = self._get(home)
        mu_a, rd_a = self._get(away)
        q = np.log(10.0) / self.config.scale
        combined = np.sqrt(rd_h ** 2 + rd_a ** 2)
        g = 1.0 / np.sqrt(1.0 + 3.0 * q ** 2 * combined ** 2 / np.pi ** 2)
        diff = mu_h + self.config.hfa - mu_a
        return float(1.0 / (1.0 + 10.0 ** (-g * diff / self.config.scale)))

    def update(self, home: str, away: str, home_win: int) -> None:
        """Apply the Glicko update after the result is known."""
        mu_h, rd_h = self._get(home)
        mu_a, rd_a = self._get(away)
        q = np.log(10.0) / self.config.scale

        def _update_one(mu: float, rd: float, mu_opp: float, rd_opp: float,
                        score: int, diff: float) -> tuple[float, float]:
            g = 1.0 / np.sqrt(1.0 + 3.0 * q ** 2 * rd_opp ** 2 / np.pi ** 2)
            expected = 1.0 / (1.0 + 10.0 ** (-g * diff / self.config.scale))
            expected = float(np.clip(expected, 1e-6, 1 - 1e-6))
            d2 = 1.0 / (q ** 2 * g ** 2 * expected * (1.0 - expected))
            denom = 1.0 / rd ** 2 + 1.0 / d2
            new_mu = mu + q / denom * g * (score - expected)
            new_rd = np.sqrt(1.0 / denom)
            return float(new_mu), float(np.clip(new_rd, self.config.min_rd,
                                                 self.config.max_rd))

        diff_h = mu_h + self.config.hfa - mu_a
        diff_a = mu_a - (mu_h + self.config.hfa)
        nh, rh = _update_one(mu_h, rd_h, mu_a, rd_a, int(home_win), diff_h)
        na, ra = _update_one(mu_a, rd_a, mu_h, rd_h, int(1 - home_win), diff_a)
        self.rating[home], self.rd[home] = nh, rh
        self.rating[away], self.rd[away] = na, ra


def _ordered_games(games: pd.DataFrame) -> pd.DataFrame:
    if "game_date" in games.columns:
        date_col = "game_date"
    elif "GAME_DATE" in games.columns:
        date_col = "GAME_DATE"
    elif "date" in games.columns:
        date_col = "date"
    else:
        raise KeyError("games needs GAME_DATE, date, or game_date")
    if "y" in games.columns:
        y_col = "y"
    elif "home_win" in games.columns:
        y_col = "home_win"
    else:
        raise KeyError("games needs y or home_win")
    out = games.copy()
    out["_date"] = pd.to_datetime(out[date_col])
    out["_y"] = out[y_col].astype(int)
    # Same-date games are batched: all predictions come from the state at the
    # end of the previous date, then the whole slate updates.  This is the
    # conservative convention used by the Elo baseline.
    return (out.sort_values(["_date", "home", "away"], kind="mergesort")
               .reset_index(drop=False).rename(columns={"index": "_source_index"}))


def fit_predict_glicko(games: pd.DataFrame,
                       config: GlickoConfig | None = None) -> dict:
    """Run one causal Glicko pass and return pre-game probabilities.

    The returned ``p`` is aligned to the caller's row order.  ``rating_home``
    and ``rating_away`` are the pre-game means; ``rd_home``/``rd_away`` are the
    pre-game deviations.  Every one of those quantities is recorded before the
    row's own update.
    """
    cfg = config or GlickoConfig()
    ordered = _ordered_games(games)
    model = GlickoRating(cfg)
    n = len(ordered)
    p = np.full(n, np.nan)
    mu_h = np.full(n, np.nan)
    mu_a = np.full(n, np.nan)
    rd_h = np.full(n, np.nan)
    rd_a = np.full(n, np.nan)

    for date, slate in ordered.groupby("_date", sort=True):
        # Predict the whole slate from the state at the start of the date.
        for i, row in slate.iterrows():
            model.inflate_inactivity(row["home"], date)
            model.inflate_inactivity(row["away"], date)
            p[i] = model.prob_home(row["home"], row["away"])
            mu_h[i], rd_h[i] = model._get(row["home"])
            mu_a[i], rd_a[i] = model._get(row["away"])
        # Then update, once per game, after every prediction on the date.
        for _, row in slate.iterrows():
            model.update(row["home"], row["away"], int(row["_y"]))
            model.last_date[row["home"]] = date
            model.last_date[row["away"]] = date

    out = pd.DataFrame({
        "p": p, "rating_home": mu_h, "rating_away": mu_a,
        "rd_home": rd_h, "rd_away": rd_a,
        "_source_index": ordered["_source_index"].to_numpy(),
    }).sort_values("_source_index")
    return {
        "p": out["p"].to_numpy(),
        "rating_home": out["rating_home"].to_numpy(),
        "rating_away": out["rating_away"].to_numpy(),
        "rd_home": out["rd_home"].to_numpy(),
        "rd_away": out["rd_away"].to_numpy(),
        "ratings": dict(model.rating),
        "rd": dict(model.rd),
        "config": cfg,
    }


@dataclass(frozen=True)
class StateSpaceConfig:
    """Fixed random-walk Bradley-Terry settings, declared before the run."""

    initial_mu: float = 0.0
    initial_variance: float = 1.0
    process_variance_per_day: float = 0.02
    hfa_elo_points: float = 70.0
    elo_scale: float = 400.0
    observation_variance: float = 1.0
    min_variance: float = 1e-6

    @property
    def hfa_logit(self) -> float:
        return self.hfa_elo_points / self.elo_scale * np.log(10.0)


class StateSpaceBT:
    """Latent BT strengths on a continuous timeline, EKF-updated per game."""

    def __init__(self, config: StateSpaceConfig | None = None) -> None:
        self.config = config or StateSpaceConfig()
        self.mu: dict[str, float] = {}
        self.var: dict[str, float] = {}
        self.last_date: dict[str, pd.Timestamp] = {}

    def _get(self, team: str) -> tuple[float, float]:
        return (self.mu.get(team, self.config.initial_mu),
                self.var.get(team, self.config.initial_variance))

    def inflate_inactivity(self, team: str, date: pd.Timestamp) -> None:
        last = self.last_date.get(team)
        if last is None:
            return
        days = max(0.0, float((date - last).days))
        mu, var = self._get(team)
        self.mu[team] = mu
        self.var[team] = float(var + self.config.process_variance_per_day * days)

    def prob_home(self, home: str, away: str) -> float:
        mu_h, var_h = self._get(home)
        mu_a, var_a = self._get(away)
        # Probit-style integration of the latent difference plus logistic
        # observation noise.  The uncertainty term therefore widens the
        # probability toward 0.5 for a team whose state is poorly known.
        total_var = var_h + var_a
        scale = np.sqrt(1.0 + (np.pi ** 2 / 8.0) * total_var)
        return float(_sigmoid((mu_h - mu_a + self.config.hfa_logit) / scale))

    def update(self, home: str, away: str, home_win: int) -> None:
        mu_h, var_h = self._get(home)
        mu_a, var_a = self._get(away)
        p = self.prob_home(home, away)
        innovation = int(home_win) - p
        s = var_h + var_a + self.config.observation_variance
        k_h = var_h / s
        k_a = -var_a / s
        self.mu[home] = mu_h + k_h * innovation
        self.mu[away] = mu_a + k_a * innovation
        self.var[home] = max(self.config.min_variance, var_h - k_h * var_h)
        self.var[away] = max(self.config.min_variance, var_a + k_a * var_a)


def fit_predict_state_space_bt(games: pd.DataFrame,
                               config: StateSpaceConfig | None = None) -> dict:
    """Run one causal EKF-BT pass and return pre-game probabilities."""
    cfg = config or StateSpaceConfig()
    ordered = _ordered_games(games)
    model = StateSpaceBT(cfg)
    n = len(ordered)
    p = np.full(n, np.nan)
    mu_h = np.full(n, np.nan)
    mu_a = np.full(n, np.nan)
    var_h = np.full(n, np.nan)
    var_a = np.full(n, np.nan)

    for date, slate in ordered.groupby("_date", sort=True):
        for i, row in slate.iterrows():
            model.inflate_inactivity(row["home"], date)
            model.inflate_inactivity(row["away"], date)
            p[i] = model.prob_home(row["home"], row["away"])
            mu_h[i], var_h[i] = model._get(row["home"])
            mu_a[i], var_a[i] = model._get(row["away"])
        for _, row in slate.iterrows():
            model.update(row["home"], row["away"], int(row["_y"]))
            model.last_date[row["home"]] = date
            model.last_date[row["away"]] = date

    out = pd.DataFrame({
        "p": p, "mu_home": mu_h, "mu_away": mu_a,
        "var_home": var_h, "var_away": var_a,
        "_source_index": ordered["_source_index"].to_numpy(),
    }).sort_values("_source_index")
    return {
        "p": out["p"].to_numpy(),
        "mu_home": out["mu_home"].to_numpy(),
        "mu_away": out["mu_away"].to_numpy(),
        "var_home": out["var_home"].to_numpy(),
        "var_away": out["var_away"].to_numpy(),
        "mu": dict(model.mu),
        "variance": dict(model.var),
        "config": cfg,
    }
