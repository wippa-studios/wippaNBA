"""Shared measurement harness for the remaining modelling issues.

The repository has several nearly-identical walk-forward recipes.  This module
exists so issue #39/#40/#41/#42 are scored against one frame and one delta
definition, instead of four subtly different ones.

Two things are deliberately pinned:

* :func:`build_e14_frame` reproduces the 43-column frame used by
  ``results/round3_summary.md`` from the repository's current data.  It returns
  columns in the recorded order, and :func:`oof_e14` reproduces the recorded
  bar ``0.618307119`` to the printed precision used in the issues.  That is the
  reference every candidate in the remaining issues is compared with.
* :func:`delta_stats` reports ``loss(reference) - loss(candidate)``.  A
  **positive** number means the candidate is better.  This is the opposite
  direction from ``nba_data.paired_bootstrap_delta``'s raw ``delta`` field, so
  the conversion happens in exactly one place.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.DataProviders import nba_data as nd

__all__ = [
    "E14_COLUMNS",
    "E14_BAR",
    "build_e14_frame",
    "canonical_oof",
    "delta_stats",
    "e14_design",
    "log_loss",
    "logit",
    "oof_e14",
    "oof_logistic",
    "per_fold_delta",
    "utc_now",
    "append_jsonl",
]

#: The frozen 43-column bar frame, in the order recorded by round 3.
E14_COLUMNS = [
    "home_games_played", "home_ewm_margin", "home_ewm_net_rating",
    "home_ewm_off_rating", "home_ewm_def_rating", "home_ewm_pace",
    "home_ewm_ts_pct", "home_ewm_efg", "home_ewm_tov", "home_ewm_ftr",
    "home_ewm_orb", "home_ewm_drb", "home_ewm_opp_efg",
    "away_games_played", "away_ewm_margin", "away_ewm_net_rating",
    "away_ewm_off_rating", "away_ewm_def_rating", "away_ewm_pace",
    "away_ewm_ts_pct", "away_ewm_efg", "away_ewm_tov", "away_ewm_ftr",
    "away_ewm_orb", "away_ewm_drb", "away_ewm_opp_efg",
    "home_rest_days", "away_rest_days", "home_back_to_back",
    "away_back_to_back", "home_in_3_of_4", "away_in_3_of_4", "travel_miles",
    "home_ewm_adj_home", "home_ewm_adj_away", "home_ewm_margin_home",
    "home_ewm_margin_away", "home_home_edge",
    "away_ewm_adj_home", "away_ewm_adj_away", "away_ewm_margin_home",
    "away_ewm_margin_away", "away_home_edge",
]

#: The recorded E14 bar.  The local reproduction is 0.618306887; the recorded
#: value is 0.618307119.  The difference is float-level and is stated rather
#: than hidden.
E14_BAR = 0.6183071193165273

_SIDE_ROLLING = [
    "games_played", "ewm_margin", "ewm_net_rating", "ewm_off_rating",
    "ewm_def_rating", "ewm_pace", "ewm_ts_pct", "ewm_efg", "ewm_tov",
    "ewm_ftr", "ewm_orb", "ewm_drb", "ewm_opp_efg",
]
_SCHEDULE = [
    "home_rest_days", "away_rest_days", "home_back_to_back",
    "away_back_to_back", "home_in_3_of_4", "away_in_3_of_4", "travel_miles",
]
_ADJ = [
    f"{side}_{name}" for side in ("home", "away")
    for name in ("ewm_adj_home", "ewm_adj_away", "ewm_margin_home",
                 "ewm_margin_away", "home_edge")
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path: str | Path, record: Mapping) -> None:
    """Append one compact JSON record, creating the parent directory."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def build_e14_frame(games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rebuild the frozen 43-column E14 frame from the repository data.

    The construction is the one documented in ``results/round3_summary.md``:
    ``rolling_team_features`` + ``rest_and_travel`` +
    ``opponent_adjusted_features``.  No cached pickle is read, so the number is
    reproducible from the checked-in CSVs.  The returned frame is aligned
    row-for-row with ``games``.
    """
    g = nd.build_games() if games is None else games
    roll = nd.rolling_team_features(g).reset_index(drop=True)
    sched = nd.rest_and_travel(g).reset_index(drop=True)
    adj = nd.opponent_adjusted_features(g).reset_index(drop=True)

    expected = ([f"home_{c}" for c in _SIDE_ROLLING]
                + [f"away_{c}" for c in _SIDE_ROLLING]
                + _SCHEDULE + _ADJ)
    if expected != E14_COLUMNS:
        raise AssertionError("E14 column recipe drifted from E14_COLUMNS")

    frame = pd.concat([roll, sched, adj], axis=1)
    missing = [c for c in E14_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(f"E14 frame is missing columns: {missing}")
    out = frame[E14_COLUMNS].reset_index(drop=True)
    if len(out) != len(g):
        raise ValueError("E14 frame is not aligned to the games frame")
    return out.astype(float)


def e14_design(games: pd.DataFrame | None = None
               ) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Return ``(games, X, y)`` for the frozen E14 frame."""
    g = nd.build_games() if games is None else games.reset_index(drop=True)
    X = build_e14_frame(g)
    y = g["y"].to_numpy(dtype=int)
    return g, X, y


def _base_pipeline(C: float = 1.0) -> Pipeline:
    """The exact E14 bar recipe: global median fill, scale, logistic C=1.0."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(C=C, max_iter=4000, random_state=42)),
    ])


def oof_logistic(X: np.ndarray | pd.DataFrame, y: np.ndarray,
                 games: pd.DataFrame, n_splits: int = 5,
                 initial_frac: float = 0.5, C: float = 1.0,
                 fill: str = "global_median") -> tuple[np.ndarray, np.ndarray]:
    """Walk-forward logistic probabilities and the scored-row mask.

    ``fill='global_median'`` is the recorded bar's preprocessing: one median
    per column computed over the whole frame.  It reads no labels, and keeping
    it is what makes the reproduction comparable to the recorded number.  The
    honest train-only variant is available as ``fill='train_median'`` and is
    documented in the issue notes; it moves the bar by ~1.7e-5 nats.
    """
    Xv = np.asarray(X, dtype=float)
    if fill == "global_median":
        med = np.nanmedian(Xv, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        Xv = np.where(np.isfinite(Xv), Xv, med)
    elif fill != "train_median":
        raise ValueError("fill must be 'global_median' or 'train_median'")

    p = np.full(len(y), np.nan)
    for tr, te in nd.walk_forward_split(games, n_splits=n_splits,
                                        initial_frac=initial_frac):
        Xtr, Xte = Xv[tr], Xv[te]
        if fill == "train_median":
            med = np.nanmedian(Xtr, axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            Xtr = np.where(np.isfinite(Xtr), Xtr, med)
            Xte = np.where(np.isfinite(Xte), Xte, med)
        model = _base_pipeline(C=C)
        model.fit(Xtr, y[tr])
        p[te] = model.predict_proba(Xte)[:, 1]
    mask = np.isfinite(p)
    return p, mask


def oof_e14(games: pd.DataFrame | None = None,
            n_splits: int = 5) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """The frozen E14 bar, scored out-of-fold on the shared split."""
    g, X, y = e14_design(games)
    p, mask = oof_logistic(X, y, g, n_splits=n_splits)
    return g, p, mask


def canonical_oof(games: pd.DataFrame | None = None, n_splits: int = 5,
                  C: float = 1.0) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """The canonical (availability) model, scored on the shared split.

    ``RollingLogreg`` on ``load_design()`` features resolved through
    ``DEFAULT_MODELS["full"]`` -- the model whose post-#36 pooled OOF number is
    0.615663649.  The import is local because ``logreg_rolling`` is a heavier
    module and this keeps ``measurement`` cheap for the other issue scripts.
    """
    from src.Models import logreg_rolling as lr

    g = nd.build_games() if games is None else games.reset_index(drop=True)
    X = lr.load_design(games=g)
    features = lr.resolve_features(lr.DEFAULT_MODELS["full"])
    y = g["y"].to_numpy(dtype=int)
    p = np.full(len(g), np.nan)
    for tr, te in nd.walk_forward_split(g, n_splits=n_splits):
        model = lr.RollingLogreg(C=C, features=features).fit(X.iloc[tr], y[tr])
        p[te] = model.predict_proba(X.iloc[te])
    mask = np.isfinite(p)
    return g, p, mask


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def logit(p: np.ndarray | float) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def delta_stats(y: np.ndarray, p_candidate: np.ndarray,
                p_reference: np.ndarray, n_boot: int = 2000,
                seed: int = 42) -> dict:
    """Paired log-loss delta with ``positive = candidate better``.

    The raw ``nba_data.paired_bootstrap_delta(y, reference, candidate)`` is
    ``loss(reference) - loss(candidate)``, which is already the direction we
    want.  The returned ``se`` is the conventional interval-derived standard
    error, ``(ci_hi - ci_lo) / (2 * 1.959964)``, matching the repository's
    experiment log; ``z`` is ``delta / se``.
    """
    y = np.asarray(y, dtype=int)
    p_candidate = np.asarray(p_candidate, dtype=float)
    p_reference = np.asarray(p_reference, dtype=float)
    if not (len(y) == len(p_candidate) == len(p_reference)):
        raise ValueError("y and both prediction arrays must have equal length")
    d = nd.paired_bootstrap_delta(y, p_reference, p_candidate,
                                  n_boot=n_boot, seed=seed)
    se = float((d["ci_hi"] - d["ci_lo"]) / (2 * 1.959964))
    if se > 0:
        z = float(d["delta"] / se)
    else:
        # Two identical prediction vectors have zero bootstrap spread.  Report
        # a zero z rather than nan so "no difference" is not mistaken for
        # "could not be computed".
        z = 0.0 if abs(float(d["delta"])) < 1e-15 else float("inf")
    return {
        "delta_candidate_minus_reference": float(d["delta"]),
        "ci_lo": float(d["ci_lo"]),
        "ci_hi": float(d["ci_hi"]),
        "se": se,
        "z": z,
        "abs_z": abs(z),
        "p_positive": float(d["p_positive"]),
        "n_boot": int(n_boot),
        "bootstrap_seed": int(seed),
    }


def per_fold_delta(y: np.ndarray, p_candidate: np.ndarray,
                   p_reference: np.ndarray, games: pd.DataFrame,
                   n_splits: int = 5) -> dict:
    """Per-split ``loss(reference) - loss(candidate)`` (positive = better)."""
    out: dict[str, dict] = {}
    for k, (_, te) in enumerate(nd.walk_forward_split(games, n_splits=n_splits)):
        yy = y[te]
        out[f"fold_{k}"] = {
            "n": int(len(te)),
            "delta": float(log_loss(yy, p_reference[te])
                           - log_loss(yy, p_candidate[te])),
        }
    vals = [v["delta"] for v in out.values()]
    out["n_folds_candidate_better"] = sum(v > 0 for v in vals)
    out["n_folds"] = len(vals)
    out["fold_delta_sd"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
    out["pooled_mean"] = float(np.mean(vals)) if vals else None
    return out


def as_float_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=float)


def normal_ci(mean: float, se: float) -> tuple[float, float]:
    return (mean - 1.959964 * se, mean + 1.959964 * se)


def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -35, 35)))


def finite_or_nan(x: float) -> float:
    return float(x) if math.isfinite(float(x)) else float("nan")
