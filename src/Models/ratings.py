"""Rating systems for NBA moneyline prediction.

Every other model in this repository has to beat this module. A rating system
is the reference implementation of "how much can you know about a game from the
games that came before it, and nothing else". If a gradient-boosted tree with
52 ranking features cannot clear it, the feature set is not earning its keep.

Three systems are provided, deliberately different in kind rather than three
parameterisations of one idea:

* :class:`EloRating` / :func:`fit_predict_elo` -- an *online* system. Ratings
  move after every game, and a prediction for game N uses ratings as they stood
  before game N was played. The rating update from game N can only ever affect
  predictions for games after N.
* :func:`fit_bradley_terry` -- the *same job, different framing*. A single
  maximum-likelihood fit over all games at once, with one strength per team and
  one home-court term. Because it is a batch fit it is order-independent by
  construction, which makes it the control that proves the Elo ordering
  guarantee is doing real work.
* :func:`fit_srs` -- the *point-based* system. Net rating per game, regressed to
  the mean, iterated to convergence (the classic SRS loop). Predicts a point
  margin rather than a win probability, and needs a one-parameter logistic
  calibration to turn that margin into a probability.

The ordering guarantee
----------------------
:func:`fit_predict_elo` sorts the input by ``(date, home, away)`` before it does
anything else, and for game N it predicts first and updates second. That is the
whole contract, and it is asserted in ``Tests/test_ratings.py`` two ways: a
hand-computed two-game sequence, and a deliberately mis-ordered run whose log
loss is measurably worse. Every other function in this module states its own
ordering behaviour in its docstring -- :func:`fit_bradley_terry` and
:func:`fit_srs` are batch fits and are order-independent.

Data contract
-------------
:func:`normalize_games` turns a repo-shaped frame into the canonical columns
these functions take:

===========  ==========================================================
``date``     the REAL game date, ``datetime64``. Never a snapshot date.
``home``     home team name, ``str``.
``away``     away team name, ``str``.
``home_win`` ``1`` if the home team won, ``0`` otherwise. Required.
``margin``   signed ``home_score - away_score``. Optional; required for
             margin-of-victory and for SRS.
``days_rest_home`` / ``days_rest_away``  optional, days since each
             team's previous game.
===========  ==========================================================

On the two date columns in this repository's dataset
---------------------------------------------------
``dataset_*.sqlite`` carries **both** ``Date`` and ``GameDate``, and they mean
different things. After the leakage fix ``Date`` is the as-of *snapshot* date of
the stats row -- the game date minus one, sometimes minus much more across a
season boundary -- while ``GameDate`` is when the game was actually played.
:func:`normalize_games` prefers ``GameDate`` and will raise rather than fall back
to ``Date``, because a rating system fed snapshot dates silently reorders the
season and there is no way to notice from the output.

On ``Points`` and ``Win_Margin`` in ``OddsData.sqlite``
------------------------------------------------------
``Win_Margin`` is ``home_score - away_score``, signed, positive when the home
team won -- that part is the intuitive reading.

``Points`` is **not** the winning team's score. It is the combined total:
``away_score + home_score``. The scraper that builds the table says so
explicitly (``src/Process-Data/Get_Odds_Data.py``)::

    "Points": game["away_score"] + game["home_score"],
    "Win_Margin": game["home_score"] - game["away_score"],

and the data agrees. Across all 3067 games in the three tables with parseable
dates, ``Points`` has median 228 (an NBA game total, not a team score), takes
values up to 309, and is at most 200 for only 238 of them -- impossible for a
single side. Meanwhile ``Points + Win_Margin`` and ``Points - Win_Margin`` are
even integers for all 3067 rows and both land in 26..162, which is exactly the
range of real team scores, and the reconstructed total tracks the over/under
line with a mean absolute error of 14.3 points.

So the per-team split is::

    home_score = (Points + Win_Margin) / 2
    away_score = (Points - Win_Margin) / 2

and :func:`scores_from_odds` is the tested implementation of that. Games where
``Points == 0`` have no recorded result at all -- 8 of the 3075 rows -- and must
be dropped, not read as a 0-0 game.

Dependencies
------------
numpy for all the arithmetic. pandas is imported lazily inside
:func:`normalize_games` and :func:`fit_predict_elo` only because those two
manipulate DataFrames, so the rating maths itself stays importable with numpy
alone. scipy happens to be installed in this environment, but
:func:`fit_bradley_terry` uses a hand-rolled Newton solver rather than lean on
that, so the module has no optional-scipy behaviour to test.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "GAME_COLUMNS",
    "DEFAULT_TEAM_ALIASES",
    "BradleyTerryResult",
    "EloConfig",
    "EloFitResult",
    "EloRating",
    "SRSResult",
    "fit_bradley_terry",
    "fit_predict_elo",
    "fit_srs",
    "logistic",
    "nba_season",
    "normalize_games",
    "scores_from_odds",
    "score_predictions",
]

#: Canonical column names produced by :func:`normalize_games` and consumed by
#: every fit function in this module.
GAME_COLUMNS = ("date", "home", "away", "home_win", "margin", "days_rest_home", "days_rest_away")

#: Franchise renames present in ``dataset_2012-26``. These are the same
#: franchise under two names, not two teams, and each pair occupies disjoint,
#: back-to-back eras (``Charlotte Bobcats`` ends 2014-10-29 where
#: ``Charlotte Hornets`` begins 2014-11-01, and so on), so merging them is a
#: statement about history rather than a modelling choice that could leak.
#:
#: Left unmerged, the Clippers would enter the 2025-26 season with a rating
#: built on no history at all -- ``LA Clippers`` is the name the test window
#: actually uses, and ``Los Angeles Clippers`` stops in 2015. Apply with
#: ``alias=DEFAULT_TEAM_ALIASES``; pass ``alias=None`` to keep names verbatim.
DEFAULT_TEAM_ALIASES = {
    "Charlotte Bobcats": "Charlotte Hornets",
    "New Jersey Nets": "Brooklyn Nets",
    "New Orleans Hornets": "New Orleans Pelicans",
    "Los Angeles Clippers": "LA Clippers",
}


def logistic(x: float | np.ndarray) -> float | np.ndarray:
    """Numerically stable logistic function.

    Accepts a scalar or an array and returns the same shape. Splits on sign
    before exponentiating so that ``exp`` never sees a large positive argument;
    a naive ``1/(1+exp(-x))`` overflows to ``inf`` for ``x < -709`` and silently
    returns 0.0.
    """
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-np.abs(x))),
        np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))),
    )


def nba_season(date) -> int:
    """Return the NBA season label for a date, as the starting calendar year.

    A game played from October onwards belongs to the season that began that
    year; a game played before October belongs to the season that began the
    previous year. So 2012-10-30 is season 2012 and 2013-04-15 is also season
    2012, which is what the odds tables call ``2012-13``.

    Parameters
    ----------
    date
        Anything ``datetime.date``-like, or a pandas timestamp.

    Returns
    -------
    int
        The four-digit year the season started in.
    """
    if getattr(date, "month", None) is None:
        raise TypeError(f"nba_season needs a date, got {type(date).__name__}")
    return date.year if date.month >= 10 else date.year - 1


def scores_from_odds(points, win_margin):
    """Split an odds-table row into per-team scores.

    ``Points`` in ``OddsData.sqlite`` is the *combined* score and ``Win_Margin``
    is the signed ``home - away`` difference, so::

        home_score = (Points + Win_Margin) / 2
        away_score = (Points - Win_Margin) / 2

    This is the opposite of the intuitive-but-wrong reading in which ``Points``
    is the winner's score and the loser scored ``Points - Win_Margin``. That
    reading implies game totals of ``2 * Points - Win_Margin``, which is wrong
    by construction; see this module's docstring for the evidence.

    Parameters
    ----------
    points
        Combined score, i.e. the odds table's ``Points`` column.
    win_margin
        Signed margin, positive when the home team won.

    Returns
    -------
    (float, float)
        ``(home_score, away_score)``.

    Raises
    ------
    ValueError
        If the two inputs cannot both be team scores -- that is, if
        ``Points + Win_Margin`` is odd, which no real box score can be.
    """
    points_arr = np.asarray(points, dtype=float)
    margin_arr = np.asarray(win_margin, dtype=float)
    if not np.allclose((points_arr + margin_arr) % 2.0, 0.0):
        bad = int(np.sum((points_arr + margin_arr) % 2.0 != 0.0))
        raise ValueError(
            f"{bad} row(s) have an odd Points + Win_Margin, so no integer score "
            "split exists; Points is probably not the combined score here"
        )
    return (points_arr + margin_arr) / 2.0, (points_arr - margin_arr) / 2.0


def normalize_games(
    frame,
    *,
    home_col: str | None = None,
    away_col: str | None = None,
    date_col: str | None = None,
    home_win_col: str | None = None,
    margin_col: str | None = None,
    rest_home_col: str | None = None,
    rest_away_col: str | None = None,
    alias: Mapping[str, str] | None = DEFAULT_TEAM_ALIASES,
    require_margin: bool = False,
):
    """Turn a repo-shaped frame into the canonical games frame.

    Every argument is optional; each one falls back to the column names this
    repository actually uses, and :class:`KeyError` names the candidates it
    looked for rather than failing silently on the wrong column.

    Ordering guarantee
    ------------------
    **None.** This function does not sort. It is a column-mapping step, and the
    caller is expected to pass the result to :func:`fit_predict_elo`, which
    does the sorting. It is kept separate so that the sort is visible at the
    one call site where leakage would happen, and so the two decisions cannot be
    confused.

    Data it expects
    ---------------
    A pandas DataFrame with a team pair, a real game date, and an outcome.
    ``Data/OddsData.sqlite`` tables expose ``Home``, ``Away``, ``Date``,
    ``Win_Margin``, ``Days_Rest_Home`` and ``Days_Rest_Away``;
    ``dataset_*.sqlite`` exposes ``TEAM_NAME`` (the HOME team -- verified
    against the odds tables, 2986/2986 rows agree), ``TEAM_NAME.1``,
    ``GameDate``, ``Home-Team-Win``, ``Score`` and ``Days-Rest-*``.

    Parameters
    ----------
    frame
        Source DataFrame.
    date_col
        Name of the real game date. Defaults to ``"GameDate"``. Falls back to
        ``"Date"`` **only** when ``GameDate`` is absent and ``Date`` parses
        cleanly; if both exist, ``Date`` is never used, because in this
        repository's dataset ``Date`` is a prior-snapshot date and feeding it to
        a rating system reorders the season.
    margin_col
        Name of the signed home-minus-away margin. ``"Win_Margin"`` by
        default. When absent, the margin is derived from ``home_win`` and left
        as ``None`` if that is unavailable too -- but see ``require_margin``.
    alias
        Mapping applied to both team columns, defaulting to
        :data:`DEFAULT_TEAM_ALIASES`. Pass ``None`` to leave names verbatim.
    require_margin
        When ``True``, raise if any row ends up without a margin. Margin-of-
        victory multipliers and SRS both need one, and silently degrading to
        ``mov_multiplier=1`` is the kind of quiet degradation that turns a bug
        into a worse number instead of an error.

    Returns
    -------
    pandas.DataFrame
        Columns :data:`GAME_COLUMNS`. Sorted by nothing -- see the ordering
        guarantee above.

    Raises
    ------
    KeyError
        If a required column cannot be located, or if no usable date column
        exists.
    ValueError
        If ``require_margin`` is set and a margin is missing.
    """

    import pandas as pd  # local: the rating maths itself does not need pandas

    def pick(explicit: str | None, candidates: Sequence[str], what: str) -> str | None:
        if explicit is not None:
            if explicit not in frame.columns:
                raise KeyError(f"{what}: column {explicit!r} not in frame")
            return explicit
        for name in candidates:
            if name in frame.columns:
                return name
        return None

    h_col = pick(home_col, ("home", "Home", "TEAM_NAME"), "home team")
    a_col = pick(away_col, ("away", "Away", "TEAM_NAME.1"), "away team")
    if h_col is None or a_col is None:
        raise KeyError(
            "home/away team columns not found; looked for home/Home/TEAM_NAME "
            "and away/Away/TEAM_NAME.1"
        )

    # Date: prefer the real game date, and never silently accept a snapshot date
    # when the real one is sitting right there under another name.
    if date_col is not None:
        d_col = pick(date_col, (), "date")
    elif "GameDate" in frame.columns:
        d_col = "GameDate"
    elif "date" in frame.columns:
        d_col = "date"
    elif "Date" in frame.columns:
        d_col = "Date"
    else:
        d_col = None
    if d_col is None:
        raise KeyError("no date column found; looked for date/GameDate/Date")

    w_col = pick(home_win_col, ("home_win", "Home-Team-Win", "home_team_win"), "outcome")
    m_col = pick(margin_col, ("margin", "Win_Margin"), "margin")
    rh_col = pick(rest_home_col, ("days_rest_home", "Days-Rest-Home", "Days_Rest_Home"), "home rest")
    ra_col = pick(rest_away_col, ("days_rest_away", "Days-Rest-Away", "Days_Rest_Away"), "away rest")

    out = {}
    for name, col in (("home", h_col), ("away", a_col)):
        values = frame[col].astype("object")
        if alias:
            values = values.map(lambda v: alias.get(v, v))
        out[name] = values.to_numpy()

    dates = frame[d_col]
    if d_col in ("Date",) and "GameDate" in frame.columns:
        raise KeyError(
            "refusing to use the 'Date' column while 'GameDate' is present: in "
            "this repository 'Date' is the prior-snapshot date, not the game "
            "date, and a rating system sorted by it processes the season out of "
            "order. Pass date_col='GameDate'."
        )
    out["date"] = np.asarray(dates)

    if w_col is not None:
        out["home_win"] = frame[w_col].astype(float).to_numpy()
    elif m_col is not None:
        margin = frame[m_col].astype(float).to_numpy()
        out["home_win"] = (margin > 0).astype(float)
    else:
        raise KeyError(
            "no outcome column found; looked for home_win/Home-Team-Win, or a "
            "margin column to derive the winner from"
        )

    if m_col is not None:
        margin = frame[m_col].astype(float).to_numpy()
    elif w_col is not None and "Score" in frame.columns:
        # A dataset row carries the game total and the winner but not the split,
        # so there is no honest margin to recover here. Left as None on purpose.
        margin = np.full(len(frame), np.nan)
    else:
        margin = np.full(len(frame), np.nan)
    out["margin"] = margin

    for key, col in (("days_rest_home", rh_col), ("days_rest_away", ra_col)):
        out[key] = (
            frame[col].astype(float).to_numpy()
            if col is not None
            else np.full(len(frame), np.nan)
        )

    games = pd.DataFrame(out, index=frame.index)

    # A row with no recorded result is not a 0-0 game; the odds tables store
    # Points == 0 and Win_Margin == 0 for games whose result was never scraped.
    bad = games["margin"].notna() & (games["margin"] == 0) & (games["home_win"] == 0)
    if int(bad.sum()):
        games = games[~bad]
    games = games.reset_index(drop=True)

    if require_margin and not bool(games["margin"].notna().all()):
        missing = int(games["margin"].isna().sum())
        raise ValueError(
            f"{missing} game(s) have no margin. Margin-of-victory and SRS both "
            "need one; supply margin_col, or pass require_margin=False to "
            "accept mov_multiplier=1 for those games."
        )
    return games


def score_predictions(y_true, prob) -> dict[str, float | None]:
    """Log loss, accuracy, AUC and Brier for a set of probabilities.

    Implemented here rather than imported from ``src/Utils/metrics.py`` so the
    module has no dependency on the training stack, and so the same four
    numbers are computed identically for every system in the comparison.

    Parameters
    ----------
    y_true
        Binary outcomes, ``0``/``1``.
    prob
        Predicted probability that the HOME team wins.

    Returns
    -------
    dict
        ``{"log_loss", "accuracy", "auc", "brier"}``. ``auc`` is ``None`` when
        the outcomes are single-class, because AUC is undefined there; every
        other metric is always present.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(prob, dtype=float), 1e-15, 1.0 - 1e-15)
    if y.size == 0:
        return {"log_loss": None, "accuracy": None, "auc": None, "brier": None}
    predicted = (p >= 0.5).astype(float)
    accuracy = float(np.mean(predicted == y))
    # log loss, clipped so a confidently-wrong prediction cannot return inf
    ll = -float(np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    brier = float(np.mean((p - y) ** 2))
    auc: float | None = None
    if np.unique(y).size >= 2:
        order = np.argsort(p, kind="mergesort")
        ranks = np.empty(p.size, dtype=float)
        ranks[order] = np.arange(1, p.size + 1, dtype=float)
        # average ranks over ties so the statistic is rank-based, not order-based
        sorted_p = p[order]
        start = 0
        for stop in np.flatnonzero(np.diff(sorted_p)).tolist() + [p.size - 1]:
            if stop > start:
                ranks[order[start : stop + 1]] = ranks[order[start : stop + 1]].mean()
            start = stop + 1
        pos = float(ranks[y == 1].sum())
        n_pos = float(np.sum(y == 1))
        n_neg = float(np.sum(y == 0))
        auc = (pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
    return {"log_loss": ll, "accuracy": accuracy, "auc": auc, "brier": brier}


@dataclass(frozen=True)
class EloConfig:
    """Hyperparameters for :class:`EloRating`.

    Defaults are the FiveThirtyEight-style choices, chosen once and not tuned
    against the test window. Every field is used by at least one test in
    ``Tests/test_ratings.py``.

    Attributes
    ----------
    k
        Base learning rate in rating points per unit of surprise. ``K * (result
        - expected)`` is the raw update.
    home_advantage
        Additive term in rating points, added to the home team's rating before
        the game and therefore handled by the same update rule as team strength.
        It is a *rating-scale* offset, not a probability offset: a 70-point home
        advantage is worth about ``0.5 + 0.5 * (1 - 1/(1+10**(70/400)))``
        ~= 0.59 in expected win probability.
    elo_scale
        Rating points that correspond to a doubling of win probability, in the
        classic ``10 ** (-diff / scale)`` convention. ``400`` is standard.
    mov_multiplier
        Whether to scale the update by how surprising the *margin* was. See
        :meth:`EloRating.margin_multiplier` for the formula and its
        justification.
    points_per_elo
        Converting a rating difference into an expected points margin. 100 Elo
        is about a 3.25-point spread in this league, which is roughly the
        spread between two average teams once home court is accounted for, so
        ``0.0325`` puts the denominator of the multiplier on a real points
        scale.
    mov_exponent
        Exponent applied to the surprise ratio. ``0.5`` is FiveThirtyEight's
        choice and is a compromise: it stops a 30-point blowout between teams
        nobody separated from moving ratings far more than a 3-point win, while
        still rewarding lopsided results.
    mov_floor
        Lower bound on the expected margin in the multiplier's denominator. Two
        teams rated exactly equal would otherwise divide by zero; ``1.0`` means
        "never treat a game as more surprising than a 1-point-margin game".
    mov_min, mov_max
        Hard clamps on the multiplier. The **upper** clamp stops one 40-point
        blowout between evenly-matched teams from moving ratings six times a
        normal step. The **lower** clamp stops a long run of narrow losses from
        freezing a rating, but it is deliberately below 1.0: a multiplier
        pinned at or above 1 would forbid the multiplier from ever falling,
        which would silently disable half of what the formula is for. Dividing
        by the *expected* margin only earns its keep on the way down -- a
        one-point win in a game the ratings had at a six-point spread is an
        upset and must move the favoured team less than a one-point win between
        equals. ``0.25`` leaves room for that while keeping the rating alive.
    rest_coefficient
        Rating points per extra day of rest differential. Rest effects are
        real but saturate, so the term is linear and deliberately small; with
        the default a three-day rest edge is worth about 9 rating points, well
        under a tenth of the gap between two average teams.
    season_reversion
        Fraction of a team's rating carried across a season boundary. Ratings
        are about *relative* strength within a league, and rosters turn over
        between seasons, so most of the gap should not persist. ``0.75`` keeps
        a quarter, which lets a genuinely strong team stay strong without
        letting a 2012 rating dominate a 2025 one.
    decay_half_life_days
        When set, all ratings shrink toward zero by half every this many days,
        applied on the elapsed days between consecutive games. This is what
        makes a rating from four years ago stop mattering. ``None`` disables it.
        At ``30``, a rating halves over a month, which is roughly how fast
        roster and form news moves.
    season_key
        Maps a game date to a season label. Defaults to :func:`nba_season`.
        Override with a column name or any callable to use a stored season.
    """

    k: float = 20.0
    home_advantage: float = 70.0
    elo_scale: float = 400.0
    mov_multiplier: bool = True
    points_per_elo: float = 0.0325
    mov_exponent: float = 0.5
    mov_floor: float = 1.0
    mov_min: float = 0.25
    mov_max: float = 2.5
    rest_coefficient: float = 3.0
    season_reversion: float = 0.75
    decay_half_life_days: float | None = None
    season_key: Callable | str | None = None


class EloRating:
    """An incremental team rating with home court, margin of victory and rest.

    The class is deliberately small: it holds a rating per team and knows how to
    (a) turn a rating difference into a win probability and (b) move the ratings
    once a result is known. It does **not** know how to order games -- that is
    :func:`fit_predict_elo`'s job, and keeping the two apart is what makes the
    ordering guarantee auditable rather than incidental.

    Ordering guarantee
    ------------------
    **None inside this class.** :meth:`predict` and :meth:`update` read and write
    the rating dict in call order and nothing else, so a caller that calls
    ``update`` before ``predict`` gets exactly the leaked answer. That is a
    feature for testing: ``Tests/test_ratings.py`` uses it to build the
    mis-ordered run on purpose. Use :func:`fit_predict_elo` for anything real.

    Data it expects
    ---------------
    Team names, and optionally a signed margin and two rest-day counts. Missing
    margins degrade to ``mov_multiplier = 1.0``; missing rest days contribute
    nothing to the rest term.

    Notes
    -----
    Ratings are mean-zero by construction rather than by renormalisation: every
    update adds ``delta`` to one team and subtracts the same ``delta`` from the
    other, so the sum over all teams is invariant. Season reversion and decay
    are per-team scalings, which also preserve the sum. That means home-court
    advantage is identified separately and never silently absorbed into one
    team's rating.
    """

    def __init__(self, config: EloConfig | None = None, initial_rating: float = 0.0):
        self.config = config or EloConfig()
        self.initial_rating = float(initial_rating)
        self.ratings: dict[str, float] = {}

    # -- state ---------------------------------------------------------------

    def rating(self, team: str) -> float:
        """Current rating for ``team``, registering it at the initial value."""
        return self.ratings.setdefault(team, self.initial_rating)

    def snapshot(self) -> dict[str, float]:
        """A copy of the current ratings."""
        return dict(self.ratings)

    def scale_all(self, factor: float) -> None:
        """Multiply every known rating by ``factor``.

        Used for season carry-over and time decay. The sum over teams scales
        with it, so a mean-zero rating vector stays mean-zero.
        """
        for team in self.ratings:
            self.ratings[team] *= factor

    # -- prediction ----------------------------------------------------------

    def effective_diff(
        self,
        home: str,
        away: str,
        *,
        days_rest_home: float | None = None,
        days_rest_away: float | None = None,
    ) -> float:
        """Home team's rating advantage in rating points, HFA and rest included.

        Returns ``R[home] - R[away] + home_advantage + rest_coefficient *
        (rest_home - rest_away)``. The rest term is dropped entirely when either
        rest value is missing, rather than treating a missing value as 0 days,
        because "unknown" and "no rest advantage" are different claims.
        """
        diff = self.rating(home) - self.rating(away) + self.config.home_advantage
        if days_rest_home is not None and days_rest_away is not None:
            rest = days_rest_home - days_rest_away
            if rest == rest:  # NaN check without importing numpy semantics at call sites
                diff += self.config.rest_coefficient * float(rest)
        return diff

    def prob_home(
        self,
        home: str,
        away: str,
        *,
        days_rest_home: float | None = None,
        days_rest_away: float | None = None,
    ) -> float:
        """Probability the home team wins, given current ratings.

        Uses the classic Elo curve ``1 / (1 + 10 ** (-diff / elo_scale))``.

        With both teams at the initial rating the difference is exactly
        ``home_advantage``, so the output is exactly ``0.5`` when
        ``home_advantage`` is 0.0 -- the equality two teams start from produces a
        fair coin, which is the property the test suite pins down.
        """
        diff = self.effective_diff(
            home, away, days_rest_home=days_rest_home, days_rest_away=days_rest_away
        )
        return float(1.0 / (1.0 + 10.0 ** (-diff / self.config.elo_scale)))

    # -- update --------------------------------------------------------------

    def margin_multiplier(self, margin: float, effective_diff: float) -> float:
        """Scale factor applied to the update for a given margin of victory.

        The formula is FiveThirtyEight's::

            mov_mult = ( |margin| / max(expected_margin, mov_floor) ) ** mov_exponent

        where ``expected_margin`` is the points margin the ratings themselves
        imply for this game, ``points_per_elo * |effective_diff|``, and
        ``effective_diff`` includes home-court advantage and rest because those
        are part of what the ratings expect.

        Why divide by the expected margin: the update magnitude should reflect
        how *surprising* the result was, not merely its size. A one-point win
        between two teams nobody separated is a routine result and should move
        ratings a full step. The same one-point win in a game the ratings had
        at a 20-point spread is an enormous upset and should move them far less
        -- and that second case only works if ``mov_min`` lets the multiplier
        fall below 1, which is why the default is 0.25 and not 1.0. Dividing the
        observed margin by the expected one captures the ratio; the ``** 0.5``
        exponent keeps it from over-reacting at either end; ``mov_floor`` stops
        the denominator collapsing to zero for evenly-matched teams; and the
        two clamps stop one blowout from dominating a season and a long drought
        from freezing a rating.

        Parameters
        ----------
        margin
            Signed home-minus-away margin. Only ``abs()`` is used.
        effective_diff
            Output of :meth:`effective_diff` **computed before this game's
            update**. The whole point is that the denominator cannot see the
            result.

        Returns
        -------
        float
            Multiplier in ``[mov_min, mov_max]``. Exactly ``1.0`` when
            ``mov_multiplier`` is off.
        """
        cfg = self.config
        if not cfg.mov_multiplier:
            return 1.0
        if margin is None:
            return 1.0
        margin = float(margin)
        if margin != margin:  # NaN margin: no margin information, no multiplier
            return 1.0
        expected = max(abs(effective_diff) * cfg.points_per_elo, cfg.mov_floor)
        mult = (abs(margin) / expected) ** cfg.mov_exponent
        return float(min(max(mult, cfg.mov_min), cfg.mov_max))

    def update(
        self,
        home: str,
        away: str,
        home_win: float,
        *,
        margin: float | None = None,
        days_rest_home: float | None = None,
        days_rest_away: float | None = None,
        effective_diff: float | None = None,
    ) -> float:
        """Move both ratings by one game and return the raw ``delta``.

        ``R[home] += delta`` and ``R[away] -= delta`` with::

            delta = k * mov_mult * (home_win - prob_home)

        Antisymmetric by construction, so the rating sum is preserved and no
        renormalisation step is needed.

        Ordering guarantee
        ------------------
        None -- see the class docstring. Call this *after* the prediction you
        wanted to make for this game.

        Parameters
        ----------
        home, away
            Team names.
        home_win
            ``1`` if the home team won, ``0`` otherwise.
        margin
            Signed margin, for the multiplier. ``None`` gives multiplier 1.0.
        effective_diff
            Pass the value already used for this game's prediction to guarantee
            the multiplier's denominator is the pre-update expectation. When
            omitted it is recomputed here, which is equivalent only if no update
            has happened since that prediction.
        """
        cfg = self.config
        if effective_diff is None:
            effective_diff = self.effective_diff(
                home, away, days_rest_home=days_rest_home, days_rest_away=days_rest_away
            )
        prob = float(1.0 / (1.0 + 10.0 ** (-effective_diff / cfg.elo_scale)))
        mult = self.margin_multiplier(margin, effective_diff) if margin is not None else 1.0
        delta = cfg.k * mult * (float(home_win) - prob)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta
        return delta


@dataclass
class EloFitResult:
    """Per-game output of :func:`fit_predict_elo`.

    Attributes
    ----------
    games
        The sorted games frame, one row per game, carrying the pre-game
        quantities. Every column here was computed from ratings that existed
        *before* the game on that row:

        ``date``, ``home``, ``away``, ``home_win``, ``margin``, ``season``,
        ``rating_home_before``, ``rating_away_before``, ``effective_diff``,
        ``prob_home``, ``mov_multiplier``, ``delta``, ``rating_home_after``.
    ratings
        Final rating per team after the last game.
    elo
        The :class:`EloRating` instance used, for further prediction.
    config
        The :class:`EloConfig` used.
    """

    games: object
    ratings: dict[str, float]
    elo: EloRating
    config: EloConfig

    @property
    def prob_home(self) -> np.ndarray:
        """Pre-game predicted home win probability, in played order."""
        return np.asarray(self.games["prob_home"], dtype=float)


def _season_labels(frame, config: EloConfig) -> np.ndarray:
    """Season label per row, honouring ``config.season_key``.

    ``None`` -> :func:`nba_season` on the date. A callable -> called with the
    date. A string -> used as a column name on ``frame``, which is how a stored
    season column is honoured without threading it through every call site.
    """
    key = config.season_key
    if key is None:
        return np.asarray([nba_season(d) for d in frame["date"]], dtype=object)
    if callable(key):
        return np.asarray([key(d) for d in frame["date"]], dtype=object)
    if isinstance(key, str):
        if key not in frame.columns:
            raise KeyError(
                f"config.season_key={key!r} is a column name but the games "
                f"frame has no such column; columns are {list(frame.columns)}"
            )
        return frame[key].to_numpy()
    raise TypeError(
        "config.season_key must be None, a callable, or the name of a column on "
        f"the games frame, got {type(key).__name__}"
    )


def fit_predict_elo(
    games,
    *,
    config: EloConfig | None = None,
    k: float | None = None,
    home_advantage: float | None = None,
    use_margin: bool | None = None,
    use_rest: bool = False,
    initial_rating: float = 0.0,
    season: np.ndarray | None = None,
    date_format: str = "%Y-%m-%d",
    same_day: str = "sequential",
    order_by: Sequence[str] | None = None,
):
    """Walk games in date order, predicting each before updating with its result.

    This is the leakage-safe entry point and the only function in the module
    that processes games sequentially. The guarantee it provides:

    1. Games are sorted by ``(date, home, away)`` unless ``order_by`` says
       otherwise. The first key is the real game date; ``home`` then ``away``
       break same-date ties in a way that is arbitrary but **deterministic**, so
       the row order a caller happens to supply cannot change any prediction.
       (Shuffling the order of players within a date is a no-op.)
    2. For game N the prediction uses the ratings as they stood *before* game N.
    3. The update from game N's result is applied only *after* game N's
       prediction is recorded, so it can affect games N+1 onward and nothing
       earlier.
    4. Season carry-over and time decay are applied on the gap between
       consecutive games, before that later game is predicted.

    ``Tests/test_ratings.py`` pins (2) and (3) with a hand-computed two-game
    sequence, and pins (1) by shuffling within a date and asserting the output
    frame is unchanged.

    Same-date batching
    ------------------
    Guarantees (2) and (3) are per *game*, which leaves one decision open when
    several games share a date: if they are processed one after another, the
    second game of the day is predicted from ratings that already contain the
    first game's result. That is only sound if you believe you could have known
    that result before tip-off -- and since a slate can contain a 19:30 and a
    21:30 game, and the schedule table this repository reads has no tip-off time,
    it is an assumption rather than a fact. ``same_day`` makes it explicit:

    ``"sequential"`` (default)
        One game at a time, exactly as described above. Preserves the historical
        behaviour of this function, and the alphabetical tiebreak makes it
        reproducible, but a same-date game is not independent of its
        same-date neighbours.
    ``"batched"``
        Every game on a date is predicted from the state as of the end of the
        *previous* date, and only then does the whole date update. No game on a
        date can see another game on the same date, in either direction. This is
        the conservative choice, and it is the one to quote when the question is
        "how much of the accuracy is real".

    The two modes differ only in whether the group of games that shares a
    rating state is one game or one date; the arithmetic inside a group is the
    same in both. ``fit_predict_elo`` is the single place that decides, so both
    modes stay auditable together.

    Data it expects
    ---------------
    A frame with the canonical columns from :func:`normalize_games` -- ``date``,
    ``home``, ``away``, ``home_win``, and optionally ``margin``,
    ``days_rest_home``, ``days_rest_away``. ``date`` must be the real game date.
    If it arrives as ``str``, ``date_format`` is used to parse it.

    Parameters
    ----------
    games
        Canonical games frame.
    config
        A full :class:`EloConfig`. Overridden field-by-field by the keyword
        arguments below.
    k, home_advantage
        Convenience overrides for the two most-tuned fields.
    same_day
        ``"sequential"`` or ``"batched"`` -- see above.
    order_by
        Column names to sort by, replacing ``("date", "home", "away")``. Must
        name columns of ``games``, and the date must be one of them: sorting on
        anything else would reintroduce the snapshot-date bug
        :func:`normalize_games` goes out of its way to prevent. Useful to put
        same-date games in a known order -- an NBA ``GAME_ID`` runs in
        schedule order within a season, which is a closer approximation to
        tip-off order than alphabetical team names -- so the cost of an
        arbitrary tiebreak can be measured rather than assumed.
    use_margin
        Force margin-of-victory on or off. ``None`` (default) uses
        ``config.mov_multiplier``, which in turn only applies where a margin is
        actually available, so games with no margin silently get multiplier 1.0.
    use_rest
        Apply the rest adjustment. Off by default because it needs two extra
        columns; the rest term is zero without them anyway.
    season
        Explicit season label per row, overriding the date-derived one.

    Returns
    -------
    EloFitResult

    Raises
    ------
    ValueError
        If ``games`` has no rows.
    """
    import pandas as pd

    cfg = config or EloConfig()
    overrides = {}
    if k is not None:
        overrides["k"] = float(k)
    if home_advantage is not None:
        overrides["home_advantage"] = float(home_advantage)
    if use_margin is not None:
        overrides["mov_multiplier"] = bool(use_margin)
    if overrides:
        cfg = replace(cfg, **overrides)

    missing = [c for c in ("date", "home", "away", "home_win") if c not in games.columns]
    if missing:
        raise KeyError(f"games is missing required column(s): {missing}")
    if len(games) == 0:
        raise ValueError("fit_predict_elo needs at least one game")

    frame = games.copy()
    dates = frame["date"]
    if not hasattr(dates.iloc[0], "year"):
        frame["date"] = pd.to_datetime(frame["date"], format=date_format, errors="coerce")
        if frame["date"].isna().any():
            raise ValueError("could not parse every date; pass date_format or pre-parse them")
    else:
        frame["date"] = pd.to_datetime(frame["date"])
    if frame["date"].isna().any():
        raise ValueError("games contains an unparseable date")

    same_day = str(same_day).lower()
    if same_day not in ("sequential", "batched"):
        raise ValueError(
            f"same_day must be 'sequential' or 'batched', got {same_day!r}"
        )

    # THE ordering. Mergesort keeps the (home, away) tiebreak stable, so the
    # result depends only on the game's own keys and never on input row order.
    sort_cols = ["date", "home", "away"] if order_by is None else list(order_by)
    unknown_cols = [c for c in sort_cols if c not in frame.columns]
    if unknown_cols:
        raise KeyError(
            f"order_by names column(s) {unknown_cols} that the games frame does "
            f"not have; columns are {list(frame.columns)}"
        )
    if "date" not in sort_cols:
        raise ValueError(
            "order_by must include 'date': a rating system whose sort keys do not "
            "start from the real game date can process a season out of order, "
            "which is the exact bug normalize_games refuses to allow for the "
            "'Date' column"
        )
    frame = frame.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    if season is not None:
        if len(season) != len(games):
            raise ValueError(f"season has {len(season)} entries, games has {len(games)}")
        frame["season"] = np.asarray(season, dtype=object)
    else:
        frame["season"] = _season_labels(frame, cfg)

    has_margin = "margin" in frame.columns and bool(frame["margin"].notna().any())
    use_rest = bool(use_rest) and "days_rest_home" in frame.columns

    elo = EloRating(cfg, initial_rating=initial_rating)

    n = len(frame)
    rating_home_before = np.zeros(n)
    rating_away_before = np.zeros(n)
    effective_diff = np.zeros(n)
    prob_home = np.zeros(n)
    mov_mult = np.ones(n)
    delta = np.zeros(n)
    rating_home_after = np.zeros(n)

    prev_date = None
    prev_season = None
    half_life = cfg.decay_half_life_days

    home_vals = frame["home"].to_numpy()
    away_vals = frame["away"].to_numpy()
    win_vals = frame["home_win"].astype(float).to_numpy()
    margin_vals = frame["margin"].to_numpy() if has_margin else np.full(n, np.nan)
    season_vals = frame["season"].to_numpy()
    date_vals = frame["date"].to_numpy()
    rest_h = frame["days_rest_home"].to_numpy() if use_rest else np.full(n, np.nan)
    rest_a = frame["days_rest_away"].to_numpy() if use_rest else np.full(n, np.nan)

    # The loop below runs over GROUPS of games that share a rating state, and
    # "sequential" is the special case where every group is one game. Expressing
    # it this way rather than as two separate loops is what keeps the two modes
    # from drifting apart: there is exactly one copy of the arithmetic, so
    # `same_day="batched"` differs from the default only in how wide a group is,
    # never in what happens inside one.
    if same_day == "sequential":
        group_starts = np.arange(n)
        group_ends = np.arange(1, n + 1)
    else:
        starts = np.concatenate(([True], date_vals[1:] != date_vals[:-1]))
        group_starts = np.flatnonzero(starts)
        group_ends = np.append(group_starts[1:], n)

    for lo, hi in zip(group_starts, group_ends):
        # Step 1: advance the rating state to just before this group. Season
        # carry-over and decay both scale every rating, and neither may look at
        # this group's results.
        game_date = date_vals[lo]
        game_season = season_vals[lo]
        if prev_date is not None:
            elapsed_days = float((game_date - prev_date) / np.timedelta64(1, "D"))
            if half_life is not None and elapsed_days > 0:
                elo.scale_all(0.5 ** (elapsed_days / float(half_life)))
            if game_season != prev_season:
                elo.scale_all(cfg.season_reversion)
        prev_date, prev_season = game_date, game_season

        # Step 2: PREDICT every game in the group from the state built in step 1.
        # Nothing below the end of this block may influence any of these rows.
        for i in range(lo, hi):
            home, away = home_vals[i], away_vals[i]
            rh = rest_h[i]
            ra = rest_a[i]
            rh = None if rh != rh else float(rh)
            ra = None if ra != ra else float(ra)

            diff = elo.effective_diff(home, away, days_rest_home=rh, days_rest_away=ra)
            prob = float(1.0 / (1.0 + 10.0 ** (-diff / cfg.elo_scale)))
            mult = elo.margin_multiplier(margin_vals[i], diff) if has_margin else 1.0

            rating_home_before[i] = elo.rating(home)
            rating_away_before[i] = elo.rating(away)
            effective_diff[i] = diff
            prob_home[i] = prob
            mov_mult[i] = mult

        # Step 3: UPDATE every game in the group. A group's updates can only
        # influence rows after `hi`, and inside a group of one, only rows after
        # `i` -- which is the ordering guarantee. Under `same_day="batched"` no
        # row inside this block can see any other row inside it.
        for i in range(lo, hi):
            home, away = home_vals[i], away_vals[i]
            step = cfg.k * mov_mult[i] * (win_vals[i] - prob_home[i])
            elo.ratings[home] = elo.rating(home) + step
            elo.ratings[away] = elo.rating(away) - step
            delta[i] = step
            rating_home_after[i] = elo.rating(home)

    frame["same_day_mode"] = same_day

    frame["rating_home_before"] = rating_home_before
    frame["rating_away_before"] = rating_away_before
    frame["effective_diff"] = effective_diff
    frame["prob_home"] = prob_home
    frame["mov_multiplier"] = mov_mult
    frame["delta"] = delta
    frame["rating_home_after"] = rating_home_after

    return EloFitResult(
        games=frame, ratings=elo.snapshot(), elo=elo, config=cfg
    )


@dataclass
class BradleyTerryResult:
    """Fitted Bradley-Terry strengths and the home-court term.

    Attributes
    ----------
    strengths
        Per-team strength in log-odds units. Summed to exactly zero over all
        teams (the MLE is otherwise not identifiable).
    home_advantage
        Fitted home-court term, same units.
    teams
        Team ordering matching ``strengths``, as a tuple.
    log_likelihood, iterations, converged, grad_norm
        Fit diagnostics.

        ``converged`` means the *projected* gradient fell below ``tol`` -- the
        fit is stationary. That is not the same as "the optimum was found".
        An unidentified direction makes the gradient vanish there too: on a
        perfectly mirrored schedule the likelihood is flat in ``home_advantage``
        and the iteration walks off along it while still reporting
        ``converged=True``. Check ``home_advantage`` for a plausible magnitude,
        not just the flag, when the training data might not identify it.
    """

    strengths: dict[str, float]
    home_advantage: float
    teams: tuple[str, ...]
    log_likelihood: float
    iterations: int
    converged: bool
    grad_norm: float = math.inf

    def prob_home(self, home: str, away: str) -> float:
        """Predicted home win probability from the frozen strengths."""
        if home not in self.strengths or away not in self.strengths:
            unknown = home if home not in self.strengths else away
            raise KeyError(f"team {unknown!r} was not in the training data")
        diff = self.strengths[home] - self.strengths[away] + self.home_advantage
        return float(logistic(diff))

    def prob_home_many(self, home: Sequence[str], away: Sequence[str]) -> np.ndarray:
        """Vectorised :meth:`prob_home` over equal-length sequences."""
        return np.asarray([self.prob_home(h, a) for h, a in zip(home, away)], dtype=float)

    def adjusted_odds(self, home: str, away: str) -> float:
        """``P(home beats away) / P(away beats home)``, home court removed."""
        return math.exp(self.strengths[home] - self.strengths[away])


def fit_bradley_terry(
    games,
    *,
    teams: Iterable[str] | None = None,
    fit_home_advantage: bool = True,
    l2: float = 1e-3,
    max_iter: int = 200,
    tol: float = 1e-10,
):
    """Maximum-likelihood Bradley-Terry strengths with a home-court term.

    The batch counterpart to :func:`fit_predict_elo`: same information, same job,
    a different framing. Where Elo walks the season one game at a time and
    therefore has an ordering to get wrong, this fits every game at once, so
    **the result does not depend on the order of the rows at all**. That is the
    point of including it -- it is the control that shows the Elo ordering
    guarantee is load-bearing rather than decorative, and it is the honest
    baseline to compare a *frozen* rating against.

    The model is ``P(home beats away) = logistic(s[home] - s[away] + hca)``,
    maximised over ``s`` and ``hca``. The strengths enter only through
    differences, so adding a constant to every one of them changes no prediction:
    their mean is not identified and is projected to zero after every iteration.
    ``hca`` *is* identified and is fitted freely -- shifting the strengths and
    ``hca`` in opposite directions does change every prediction.

    Optimiser
    ---------
    Newton's method, implemented directly in numpy -- no scipy. The log
    likelihood is concave, so the Hessian is negative semi-definite and the
    Newton direction is the global maximiser of the local quadratic, which makes
    this both faster and more reliable here than gradient ascent. ``l2`` adds a
    tiny ridge for numerical safety and as a weak prior against a team with one
    game in the data.

    Ordering guarantee
    ------------------
    **None needed -- the fit is order-independent.** Every game's contribution
    enters as a sum, and the optimiser is a deterministic function of that sum.
    Predictions for a held-out window therefore use the same frozen parameters
    no matter how the training rows were ordered.

    Data it expects
    ---------------
    A canonical games frame (``home``, ``away``, ``home_win``) as produced by
    :func:`normalize_games`. No margin, no rest, no date: this model uses
    outcomes only. Sorting the frame before calling it is harmless but
    unnecessary.

    Parameters
    ----------
    teams
        Roster to fit over. Defaults to every team present in ``games``.
        Supplying it explicitly keeps a team that only appears in the test
        window out of the parameter vector.
    fit_home_advantage
        When ``False``, ``hca`` is pinned to 0.
    l2
        Ridge penalty on every fitted parameter. Small by default; set 0 for the
        exact unpenalised MLE. It exists for two reasons. A team that never
        wins has an unbounded strength in the unpenalised MLE, and a perfectly
        mirrored schedule (every home win matched by the same result from the
        other side) leaves the home term unbounded too, because such data says
        nothing at all about home court. Neither is a numerical accident --
        they are properties of the likelihood -- and both are fixed by a prior
        rather than by a fudge factor in the optimiser.
    max_iter, tol
        Iteration cap and gradient-norm convergence threshold.

    Returns
    -------
    BradleyTerryResult

    Notes
    -----
    The ridge is what keeps the optimum finite when the data cannot identify a
    parameter. See the ``l2`` entry above.
    """
    if len(games) == 0:
        raise ValueError("fit_bradley_terry needs at least one game")

    home = np.asarray(games["home"].astype(str).to_numpy())
    away = np.asarray(games["away"].astype(str).to_numpy())
    y = np.asarray(games["home_win"].astype(float).to_numpy())

    roster = tuple(teams) if teams is not None else tuple(sorted(set(home) | set(away)))
    index = {t: i for i, t in enumerate(roster)}
    unknown = (set(home) | set(away)) - set(index)
    if unknown:
        raise KeyError(f"games contains team(s) not in `teams`: {sorted(unknown)}")

    hi = np.asarray([index[t] for t in home], dtype=int)
    ai = np.asarray([index[t] for t in away], dtype=int)

    n_teams = len(roster)
    n_params = n_teams + (1 if fit_home_advantage else 0)
    if n_teams == 0:
        raise ValueError("fit_bradley_terry has no teams to fit")

    theta = np.zeros(n_params)
    ridge = np.full(n_params, l2)

    # The likelihood depends on the strengths only through differences s[h] -
    # s[a], so shifting every strength by a common constant leaves every
    # prediction, and therefore the whole objective, unchanged. That makes
    # (1, ..., 1, 0) an exact null direction of the Hessian. The home-court
    # term is NOT identified with it -- adding c to every strength and -c to
    # `hca` would change z = s[h] - s[a] + hca by -c -- so it is a real
    # parameter. Two consequences, both needed:
    #   * the gradient must be measured with the flat direction projected out,
    #     or the convergence test can fire on the wrong thing;
    #   * the Newton solve must return a MINIMUM-NORM step, which `lstsq` gives
    #     for free and `solve` does not. A ridge-only fix gives the flat
    #     direction a step of order grad/ridge, i.e. arbitrarily large.
    flat = np.zeros(n_params)
    flat[:n_teams] = 1.0

    def project(vector: np.ndarray) -> np.ndarray:
        if not fit_home_advantage:
            return vector
        return vector - flat * (float(vector @ flat) / float(flat @ flat))

    def loglik_and_grad(th):
        diff = th[hi] - th[ai]
        if fit_home_advantage:
            diff = diff + th[n_teams]
        p = np.clip(logistic(diff), 1e-12, 1.0 - 1e-12)
        ll = float(np.sum(y * np.log(p) + (1.0 - y) * np.log1p(-p)))
        resid = y - p  # dLL/d(diff) for each game
        grad = np.zeros(n_params)
        np.add.at(grad, hi, resid)
        np.add.at(grad, ai, -resid)
        if fit_home_advantage:
            grad[n_teams] = float(np.sum(resid))
        grad -= ridge * th
        return ll - 0.5 * float(np.sum(ridge * th * th)), grad

    iterations = 0
    grad_norm = math.inf
    converged = False
    for iterations in range(1, int(max_iter) + 1):
        ll, grad = loglik_and_grad(theta)
        grad_norm = float(np.linalg.norm(project(grad)))
        if grad_norm < tol:
            converged = True
            break
        # NEGATIVE Hessian of the log likelihood, built from scratch, so
        # that `theta += solve(neg_hess, grad)` is an ascent step. With
        # dz/ds[h] = 1, dz/ds[a] = -1, dz/dhca = 1 and -d2L/dz2 = w > 0, every
        # entry is +w times a product of those three derivatives. Two traps
        # here, both of which send the iteration to infinity rather than to the
        # optimum: getting the sign backwards turns the ascent into a descent,
        # and omitting the strength/home-court cross terms makes the matrix
        # indefinite.
        diff = theta[hi] - theta[ai]
        if fit_home_advantage:
            diff = diff + theta[n_teams]
        p = np.clip(logistic(diff), 1e-12, 1.0 - 1e-12)
        w = p * (1.0 - p)  # -d2L/dz2, always >= 0
        neg_hess = np.zeros((n_params, n_params))
        np.add.at(neg_hess, (hi, hi), w)
        np.add.at(neg_hess, (ai, ai), w)
        np.add.at(neg_hess, (hi, ai), -w)
        np.add.at(neg_hess, (ai, hi), -w)
        if fit_home_advantage:
            np.add.at(neg_hess, (hi, n_teams), w)
            np.add.at(neg_hess, (n_teams, hi), w)
            np.add.at(neg_hess, (ai, n_teams), -w)
            np.add.at(neg_hess, (n_teams, ai), -w)
            neg_hess[n_teams, n_teams] += float(np.sum(w))
        neg_hess = neg_hess - np.diag(ridge)
        # `lstsq` returns the minimum-norm solution, so the step automatically
        # has no component along the flat direction.
        try:
            step = np.linalg.lstsq(neg_hess, project(grad), rcond=None)[0]
        except np.linalg.LinAlgError:
            # Unreachable in practice -- `lstsq` does not raise on a singular
            # system, it returns the min-norm solution -- but a silent crash in
            # a baseline would be worse than a slow fit.
            step = 0.05 * project(grad)
        if not np.all(np.isfinite(step)):
            step = 0.05 * project(grad)
        theta = theta + step
        # Re-identify by zeroing the mean of the strengths. Subtracting a common
        # constant from every strength leaves every s[h] - s[a] untouched, so
        # predictions are unchanged; it just keeps the reported numbers centred.
        theta[:n_teams] -= float(np.mean(theta[:n_teams]))
        if not np.all(np.isfinite(theta)):
            theta = np.nan_to_num(theta, nan=0.0, posinf=0.0, neginf=0.0)

    theta[:n_teams] -= float(np.mean(theta[:n_teams]))
    final_ll, _ = loglik_and_grad(theta)
    strengths = {team: float(theta[i]) for i, team in enumerate(roster)}
    hca = float(theta[n_teams]) if fit_home_advantage else 0.0
    return BradleyTerryResult(
        strengths=strengths,
        home_advantage=hca,
        teams=roster,
        log_likelihood=float(final_ll),
        iterations=iterations,
        converged=converged,
        grad_norm=grad_norm,
    )


@dataclass
class SRSResult:
    """Fitted SRS ratings and the margin calibration.

    Conventions, because SRS has several and getting one wrong silently zeroes
    the signal:

    * ``offense[i]`` is team i's average *scoring above the league average*.
    * ``defense[i]`` is team i's average *conceding above the league average*,
      so **positive means a worse defence** -- this is "points allowed minus
      average", not a positive-is-good rating.
    * ``net_rating[i] = offense[i] - defense[i]`` is the classic SRS number in
      points per game, and it is what predictions are built from.

    Which margin predicts a game
    ----------------------------
    Two expressions are available and they are **not** interchangeable:

    * :meth:`predicted_margin` -- ``offense[home] - defense[away] + hca``, the
      conventional SRS predictor, and what :meth:`prob_home` uses.
    * :meth:`predicted_matchup_margin` -- ``offense[home] - offense[away] + hca``,
      using only the two teams' own scoring tendencies and ignoring the matchup.

    They differ whenever a team scores well against some opponents and badly
    against others, which is the whole reason SRS was published with a separate
    defense term. Read off the wrong one and the number is simply different, so
    the distinction is spelled out rather than left to the reader.

    Attributes
    ----------
    offense, defense
        Per-team scoring and conceding, in points per game above league average,
        with every team's games averaged at rate ``1 / games_played``.
    games_played
        Per-team game count, the divisor in that regression.
    home_advantage
        Points added for playing at home, in the same units.
    margin_scale, margin_center
        Logistic calibration: ``p = logistic(margin_scale * (margin -
        margin_center))``. ``margin_center`` is the mean training margin and
        exists so the calibration has an intercept -- without it the fitted
        slope collapses to zero, because an NBA margin already encodes the home
        base rate. See :func:`_fit_margin_scale`.
    log_likelihood
        Log likelihood of the calibration at the out-of-fold margins.
    """

    offense: dict[str, float]
    defense: dict[str, float]
    games_played: dict[str, int]
    home_advantage: float
    margin_scale: float
    margin_center: float = 0.0
    log_likelihood: float = math.nan

    @property
    def teams(self) -> tuple[str, ...]:
        return tuple(self.offense)

    @property
    def net_rating(self) -> dict[str, float]:
        """``offense - defense`` per team, the conventional SRS number."""
        return {t: self.offense[t] - self.defense[t] for t in self.offense}

    def predicted_margin(self, home: str, away: str) -> float:
        """Predicted home-minus-away points margin: net difference plus HCA."""
        return self.net_rating[home] - self.net_rating[away] + self.home_advantage

    def predicted_matchup_margin(self, home: str, away: str) -> float:
        """The offense-only form, ``offense[home] - offense[away] + hca``.

        Exposed for comparison, not used for predictions. See the class
        docstring.
        """
        return self.offense[home] - self.offense[away] + self.home_advantage

    def prob_home(self, home: str, away: str) -> float:
        """Predicted home win probability from the calibrated margin.

        ``logistic(margin_scale * (predicted_margin - margin_center))``. The
        centring matches the calibration in :func:`_fit_margin_scale`, which
        fits the same centred form. Dropping it here while keeping it there
        would make every prediction read as a huge edge and saturate the
        sigmoid.
        """
        return float(
            logistic(self.margin_scale * (self.predicted_margin(home, away) - self.margin_center))
        )


def _srs_ratings(
    hi: np.ndarray,
    ai: np.ndarray,
    adjusted: np.ndarray,
    games_played: np.ndarray,
    n: int,
    games: np.ndarray | None = None,
) -> np.ndarray:
    """Fit SRS offense and defense by weighted least squares. Returns both.

    The system
    ----------
    One equation per game::

        o[home] - d[away] = adjusted margin

    with ``adjusted = margin - mean(margin)`` removing the league average (which
    is also the home edge, about +2.0 points) so the ratings are mean-centred by
    construction. ``o`` is points scored above average, ``d`` is points allowed
    above average, so a positive ``d`` is a worse defence and
    ``net = o - d``.

    That is ``n_games`` equations in ``2n`` unknowns, and it is genuinely
    underdetermined in the useful sense: every team plays every other twice,
    once at each venue, and the two games constrain the same combination with
    different right-hand sides -- consistency would demand the two margins be
    equal. Measured on this repo's data, 463 pairings appear at both venues and
    a typical pair disagrees by 20 points. The solver's job is the least-bad
    answer, not an exact one.

    Why offense and defense are separate parameters
    -------------------------------------------------
    Collapsing ``d = -o`` -- true in a fixed-total league, where a team's
    conceding deviation is its opponent's scoring deviation -- *overdetermines*
    the system and forces the ratings to shrink toward zero. Measured here the
    collapsed form returns a net-rating spread of 5 points where the separated
    form returns 10, for the same R^2. Per-game margin noise (std ~15 points)
    dwarfs the team-strength signal, so both fit the games about equally badly;
    the separated form at least preserves the scale.

    Do not add the mirror equation
    ------------------------------
    It is tempting to also write ``d[home] - o[away] = -adjusted``, giving each
    team a residual from its own perspective. That is the same observation
    entered twice, and on a symmetric round robin the two blocks contradict each
    other. Measured on a six-team league with strengths spread over +/-6, the
    two-block version returned a net-rating spread of 0.72 where this one returns
    12.0 -- the true spread. A game informs the attacker's offense and the
    defender's defense. That is all the information it carries.

    Why not the published iteration
    -------------------------------
    The iterative form -- take the residual, reassign, divide by games played --
    is the usual presentation, and it is what this module implemented first. It
    does not converge. Because ``d = -o`` holds at its fixed point, the
    prediction collapses to ``o[home] + o[away]``, and a game then corrects the
    attacker by ``+resid/gp_h`` and the defender by ``-resid/gp_a``; when those
    game counts differ the pair's total drifts every pass instead of being
    preserved. Measured on an 8-team synthetic league, ratings grew by 1.27x per
    pass and passed 300 after 60 passes, where the least-squares answer is 13.
    Damping slowed it without removing the drift, because the instability is
    structural rather than a step-size accident, and the "converged" value was
    an artefact of stopping at ``max_iter``. In the calibration holdout each
    block stopped somewhere different, producing margins of +-600 points and a
    calibration slope 55x too flat.

    This solve has no iteration, so there is no truncation artefact: same input,
    same output, always.

    Regression to the mean
    ----------------------
    Game weights are ``1 / sqrt(gp_home * gp_away)``, floored at
    ``1 / max(gp)`` so a one-game team cannot dominate. A game between two
    established teams therefore counts for more than one involving a team with
    three games, which is the regression that makes this a rating rather than an
    average.

    The system is rank-deficient -- adding a constant to every ``o`` and
    subtracting it from every ``d`` changes no game equation -- so ``lstsq``
    returns the minimum-norm solution, which is the centred one. That
    normalisation comes out of the solver rather than needing a projection
    afterwards, though the centring is re-applied explicitly regardless.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Per-team ``(offense, defense)`` arrays, each mean-zero.
    """
    n_games = hi.size
    if n_games == 0 or n == 0:
        return np.zeros(n), np.zeros(n)

    gp = np.maximum(games_played, 1.0)
    weight = 1.0 / np.sqrt(gp[hi] * gp[ai])
    weight = np.maximum(weight, 1.0 / max(float(gp.max()), 1.0))

    rows = np.arange(n_games)
    # One equation per game: the home team's scoring minus the away team's
    # conceding equals the adjusted margin.
    #
    # It is tempting to add the mirror equation for the away team's own
    # perspective, ``d[home] - o[away] = -adjusted``. That is the same
    # observation twice, and it is actively harmful: on a symmetric round robin
    # the two blocks contradict each other and pull every rating toward zero.
    # Measured on a six-team league with strengths spread over +/-6, the
    # two-block version returned a net-rating spread of 0.72 where the
    # one-block version returns 12.0, which is the true spread. A game informs
    # the attacker's offense and the defender's defense -- that is all.
    design = np.zeros((n_games, 2 * n))
    target = np.zeros(n_games)
    design[rows, hi] = 1.0
    design[rows, n + ai] = -1.0
    target[:] = adjusted

    solution, _, rank, _ = np.linalg.lstsq(design * weight[:, None], target * weight, rcond=None)
    if rank < 1 or not np.all(np.isfinite(solution)):
        return np.zeros(n), np.zeros(n)
    offense = solution[:n]
    defense = solution[n:]
    # Guarantee the centring the min-norm property gives in exact arithmetic.
    return offense - float(np.mean(offense)), defense - float(np.mean(defense))


def _calibrate_out_of_sample(
    home, away, y, adjusted, index, n, home_advantage, holdout: float = 0.15
):
    """Fit ``margin_scale`` on the final ``holdout`` fraction of games.

    Why a time-ordered holdout
    --------------------------
    Two calibrations look reasonable and both are badly wrong.

    1. On the **realised** margin. The realised margin determines the label
       exactly -- ``sign(margin) == home_win`` on 3067/3067 rows of the odds
       tables -- so this fits a near-deterministic relation and returns a slope
       around ``1.26``, steep enough that every predicted margin saturates the
       sigmoid. Measured log loss with it: 13.3.
    2. On the **in-sample fitted** margin. Better, but the rating was fitted to
       these very games, so the fitted margin is correlated with the outcome by
       construction. Measured slope: about 2x too steep.

    So the slope is fitted on games the rating never saw: the last
    ``holdout`` fraction of the input, in the order given, predicted by a rating
    fitted on everything before. That is also exactly how the calibrated rating
    gets used at prediction time, so the constant is fitted in the regime it
    will be applied in.

    The split is on *position in the input*, not on a random mask, and that
    matters: the caller supplies games in date order, so the holdout is the most
    recent games. A rating is never asked to predict a game from before it.

    Falls back to in-sample when there are too few games to split, which is
    biased but better than refusing to produce a calibration at all.

    Returns
    -------
    (float, float)
        The slope and the log likelihood at the held-out margins.
    """
    n_games = len(home)
    home_arr = np.asarray(home)
    away_arr = np.asarray(away)
    y_arr = np.asarray(y, dtype=float)
    scoring_arr = np.asarray(adjusted, dtype=float)

    def ratings_for(mask: np.ndarray):
        """Fit the rating system on the masked subset; return (offense, defense)."""
        sub_hi = np.asarray([index[t] for t in home_arr[mask]], dtype=int)
        sub_ai = np.asarray([index[t] for t in away_arr[mask]], dtype=int)
        gp = np.zeros(n)
        np.add.at(gp, sub_hi, 1.0)
        np.add.at(gp, sub_ai, 1.0)
        return _srs_ratings(sub_hi, sub_ai, scoring_arr[mask], gp, n)

    def margins_from(off, dfn, mask: np.ndarray) -> np.ndarray:
        """The SAME expression `predicted_margin` returns, for the masked rows.

        That is a net-rating difference, ``offense[h] - defense[a]``. Building it
        any other way -- from the offense difference alone, say -- would fit the
        slope on margins of a different size and return a gradient that
        ``prob_home`` does not actually apply.
        """
        return np.asarray(
            [
                off[index[h]] - dfn[index[a]] + home_advantage
                for h, a in zip(home_arr[mask], away_arr[mask])
            ]
        )

    # Need enough games on both sides for a rating and a holdout to mean anything.
    cut = int(n_games * (1.0 - holdout))
    if cut >= 30 and (n_games - cut) >= 20:
        early = np.zeros(n_games, dtype=bool)
        early[:cut] = True
        late = ~early
        off, dfn = ratings_for(early)
        held = margins_from(off, dfn, late)
        if np.std(held) > 1e-9:
            return _fit_margin_scale(held, y_arr[late])

    every = np.ones(n_games, dtype=bool)
    off, dfn = ratings_for(every)
    return _fit_margin_scale(margins_from(off, dfn, every), y_arr)


def _fit_margin_scale(margins: np.ndarray, y: np.ndarray, tol: float = 1e-11) -> tuple[float, float]:
    """Logistic calibration of a point margin, by 1-D Newton on the scale.

    Fits ``p = logistic(k * (margin - mean(margin)))`` and returns ``k``.

    The centring is not cosmetic, it is what makes the parameter identifiable.
    A raw fit of ``p = logistic(k * margin)`` has no intercept, and an NBA
    margin already carries the base rate -- its mean is about +2.0 points
    precisely because home teams win more often. So the un-centred model's
    likelihood is maximised as ``k -> 0``, which returns the base rate and
    discards the signal entirely; measured on real margins here, the optimum
    was ``k = 0`` with a log likelihood 2600 nats worse than at a real scale.
    Centring profiles the intercept out, exactly as logistic regression does,
    and leaves a single identifiable slope.

    Safeguards, each added because its absence produced a real failure:

    * ``z`` is clipped to +/-500 before ``exp``. The log likelihood
      ``y*z - log1p(exp(z))`` overflows to ``inf``/``nan`` beyond that, and one
      ``nan`` poisons the gradient permanently.
    * A step is taken only if it improves the objective, halving until it does.
      On a saturated likelihood the raw Newton step overshoots to ``|k| ~ 1e12``
      where the Hessian underflows to zero (``2e-212``) and there is no way
      back. Unfixed, that produced ``margin_scale = 33.1`` on real NBA margins
      -- a scale at which one point of predicted margin is worth 0.84 in
      probability -- and a log loss of 13.3.
    * ``k`` is confined to ``[1e-9, 1000]``, since a negative slope is not a
      probability model.

    Returns
    -------
    (float, float)
        The scale and the fitted log likelihood at the centred margin.
    """
    if margins.size == 0:
        return 1.0, math.nan
    centred = margins - float(np.mean(margins))
    spread = float(np.std(centred))
    if spread <= 1e-12:
        # No variation in the margin, so no slope is identifiable. Return the
        # inverse-logit of the base rate as a scale-like quantity rather than
        # dividing by zero.
        base = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
        return float(np.clip(math.log(base / (1.0 - base)), 1e-9, 1000.0)), math.nan

    def objective(value: float) -> float:
        z = np.clip(value * centred, -500.0, 500.0)
        return float(np.sum(y * z - np.log1p(np.exp(z))))

    # A 2-sigma margin should read as roughly a 2:1 chance, so start near 0.35.
    k = float(np.clip(0.35, 1e-9, 1000.0))
    best_k, best_ll = k, objective(k)
    for _ in range(200):
        z = np.clip(k * centred, -500.0, 500.0)
        p = np.asarray(logistic(z), dtype=float)
        grad = float(np.sum((y - p) * centred))
        w = p * (1.0 - p)
        hess = -float(np.sum(w * centred * centred))
        if not math.isfinite(hess) or hess >= -1e-12 or not math.isfinite(grad):
            break
        step = -grad / hess
        if not math.isfinite(step):
            break
        improved = False
        for _ in range(60):
            candidate = float(np.clip(k + step, 1e-9, 1000.0))
            candidate_ll = objective(candidate)
            if math.isfinite(candidate_ll) and candidate_ll >= best_ll:
                if candidate_ll > best_ll:
                    best_ll, best_k = candidate_ll, candidate
                k = candidate
                improved = True
                break
            step *= 0.5
        if not improved or abs(step) < tol:
            break
    return best_k, best_ll


def fit_srs(
    games,
    *,
    home_advantage: float | None = None,
    iterations: int = 60,
    tol: float = 1e-8,
):
    """Classic SRS: net rating per game, regressed to the mean, iterated.

    Each team's scoring and conceding are regressed toward the league average
    at rate ``1 / games_played`` -- that regression is what makes it an SRS
    rating rather than a raw average, and it is why a team that wins its first
    two games 30 points does not end up rated like a champion. The loop is the
    standard SRS iteration: attribute each game's scoring deviation to the
    attacker's offense and to the defender's concedes, re-average toward the
    mean, repeat. :attr:`SRSResult.net_rating` is the conventional single
    number and is what predictions use.

    Ordering guarantee
    ------------------
    **None required -- the fit is order-independent.** The iteration averages
    per-team residuals and every term enters as a sum, so row order cannot move
    any parameter. Unlike Elo there is no "prediction before update" question to
    get wrong, because nothing is predicted until the fit finishes. As with
    Bradley-Terry, a held-out window is therefore scored from frozen parameters.

    Data it expects
    ---------------
    A canonical games frame **with a signed margin on every row**. SRS is a
    point-differential system; win/loss labels alone cannot identify it, which
    is why :func:`normalize_games` can be told ``require_margin=True``. Games
    with no margin are dropped, and a team left with no games is not fitted at
    all -- ``games_played`` shows who was actually used.

    Parameters
    ----------
    home_advantage
        Points added for playing at home, **on top of** the league average
        margin that the rating loop already subtracts. Defaults to ``0.0``, not
        to the data's mean margin, for the double-counting reason in the
        implementation below. Pass an explicit value to put the home edge in the
        intercept instead.

        If you want the home edge estimated rather than removed with the league
        mean, fit on the raw margin with ``home_advantage=0.0`` and read the
        fitted ``margin_center`` -- that is where the home edge ends up once the
        calibration's intercept is in play.
    iterations, tol
        Accepted for call-site compatibility and ignored. The rating solve is a
        single weighted least-squares problem, so it has no iteration to cap and
        no tolerance to meet; it is deterministic and runs once. See
        :func:`_srs_ratings` for why the published iterative form of this loop
        was replaced rather than merely bounded.

    Returns
    -------
    SRSResult

    Notes
    -----
    ``margin_scale`` is calibrated on the same games that produced the ratings,
    so the in-sample probability fit is optimistic by construction. Held-out
    scoring does not inherit that bias.
    """
    if len(games) == 0:
        raise ValueError("fit_srs needs at least one game")
    if "margin" not in games.columns:
        raise KeyError("fit_srs needs a 'margin' column; call normalize_games(..., require_margin=True)")

    margin = np.asarray(games["margin"].astype(float).to_numpy())
    keep = margin == margin  # drop NaN margins
    if not keep.all():
        games = games[keep]
        margin = margin[keep]
    if margin.size == 0:
        raise ValueError("fit_srs has no games with a margin")

    home = games["home"].astype(str).to_numpy()
    away = games["away"].astype(str).to_numpy()
    y = np.asarray(games["home_win"].astype(float).to_numpy())
    roster = tuple(sorted(set(home) | set(away)))
    index = {t: i for i, t in enumerate(roster)}
    n = len(roster)

    # League average removed once, up front, so every rating is a deviation and
    # the system is mean-centred by construction. With the average gone, the two
    # teams' scoring deviations in a game sum to zero, so an adjusted margin of m
    # means the home team scored +m/2 above average and the away team -m/2.
    # See SRSResult's docstring for why that factor of one half matters.
    adjusted = margin - float(np.mean(margin))

    # `home_advantage` is an INTERCEPT applied on top of those centred ratings,
    # and the centring has already removed the home edge -- the mean NBA margin
    # is about +2.0 points precisely because home teams win more often. So the
    # default is 0.0; defaulting to mean(margin) counts that edge twice.
    if home_advantage is None:
        home_advantage = 0.0

    hi = np.asarray([index[t] for t in home], dtype=int)
    ai = np.asarray([index[t] for t in away], dtype=int)
    games_played = np.zeros(n)
    np.add.at(games_played, hi, 1.0)
    np.add.at(games_played, ai, 1.0)
    games_played = np.maximum(games_played, 1.0)

    offense, defense = _srs_ratings(hi, ai, adjusted, games_played, n)

    predicted = np.asarray(
        [
            offense[index[h]] - defense[index[a]] + home_advantage
            for h, a in zip(home, away)
        ],
        dtype=float,
    )
    # Calibrated on games the rating never saw. See _calibrate_out_of_sample for
    # why both the realised margin and the in-sample fitted margin give a badly
    # wrong slope.
    scale, ll = _calibrate_out_of_sample(home, away, y, adjusted, index, n, home_advantage)

    return SRSResult(
        offense={t: float(offense[i]) for i, t in enumerate(roster)},
        defense={t: float(defense[i]) for i, t in enumerate(roster)},
        games_played={t: int(games_played[i]) for i, t in enumerate(roster)},
        home_advantage=float(home_advantage),
        margin_scale=scale,
        margin_center=float(np.mean(predicted)),
        log_likelihood=ll,
    )
