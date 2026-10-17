"""Iterative modelling experiment harness.

WHAT THIS IS FOR
----------------
An append-only experiment log. Every run appends one record to
`results/experiments.jsonl` with its feature set, model, metric, train-vs-test gap
and top features, so the set of things that were tried is a permanent artefact
rather than a memory.

WHY A HARNESS INSTEAD OF A NOTEBOOK
-----------------------------------
The failure this project has to avoid is reporting the winner of a search without
the losers. A harness makes the search mechanical: define experiments, run them
all, append them all, and let the ranking fall out. Selecting after the fact is
much harder when the discarded runs are already on disk next to the kept ones.

THE METRIC
----------
Primary is **log loss**. It is the project's established metric - it is what every
delta in the README is denominated in, it is what the market comparison uses, and
it is the scoring rule appropriate for a probability forecast that will be turned
into a bet. Accuracy is reported but never used to rank: it throws away the
confidence that a bettor needs, and on this task two models can share it while
disagreeing sharply about every individual game. Brier and AUC come along as
secondary.

LEAKAGE DISCIPLINE
------------------
* The split is `nba_data.walk_forward_split`, five folds, identical for every
  experiment, so every number is comparable with every other.
* Every transform - imputation, scaling - lives INSIDE a `Pipeline` fitted within
  each fold. Nothing is fitted on the union of train and test.
* Feature builders are causal by construction: they take `games` and read only
  strictly-past information, which `Tests/test_opponent_adjusted.py` and
  `Tests/test_logreg_rolling.py` verify by truncating the future.
* Train-vs-test gap is reported for every experiment precisely so that an
  improvement bought by memorising the training set cannot hide.

REPORTING RULE
--------------
Every configuration that is run is reported. A bare best-of-N is not a result; it
is the maximum of N draws, and the harness prints the selection cost of the
winner so that number is visible rather than implied.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from DataProviders import nba_data as nd  # noqa: E402

SEED = 42
N_SPLITS = 5
HALFLIFE = 10.0
LOG_PATH = ROOT / "results" / "experiments.jsonl"


@dataclass
class Experiment:
    name: str
    features: str
    model: str
    rationale: str
    #: columns to drop from the base frame, used to express "base minus X"
    drop: tuple[str, ...] = field(default_factory=tuple)
    params: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Feature frames. Each returns a numeric frame aligned to `games`.
# --------------------------------------------------------------------------

_BASE_CACHE: dict = {}


def _base(games: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Every block the shared layer can produce, named.

    Cached per process. `build_frame` runs once per experiment and
    `player_elo_features` re-executes the whole rating loop plus an as-of merge
    over 281k player-games each time - roughly two and a half minutes a call.
    Across seventeen experiments that is most of the wall clock spent rebuilding
    an identical frame.

    The key is the frame's identity and shape, so a different slice is a different
    key rather than a silently wrong answer. Frames come back as copies, because
    `build_frame` selects and drops columns on them and a shared mutable frame is
    precisely the hazard `_cached_copy` documents.
    """
    key = (id(games), len(games),
           str(games["GAME_ID"].iloc[0]) if len(games) else "",
           str(games["GAME_ID"].iloc[-1]) if len(games) else "")
    if key not in _BASE_CACHE:
        _BASE_CACHE[key] = {
            "roll": nd.rolling_team_features(games, halflife=HALFLIFE,
                                             min_games=5),
            "schedule": nd.rest_and_travel(games),
            "home_split": nd.opponent_adjusted_features(games,
                                                       halflife=HALFLIFE),
            "player_elo": nd.player_elo_features(games, halflife=HALFLIFE),
        }
    return {k: v.copy() for k, v in _BASE_CACHE[key].items()}


_HOME_SPLIT_COLS = (
    "home_ewm_margin_home", "away_ewm_margin_home",
    "home_ewm_margin_away", "away_ewm_margin_away",
    "home_home_edge", "away_home_edge",
)
_OPP_ADJ_COLS = ("home_ewm_adj_margin", "away_ewm_adj_margin")
_SOS_COLS = ("home_ewm_sos", "away_ewm_sos")
_OPP_CHAR_COLS = ("home_ewm_opp_ftr", "away_ewm_opp_ftr",
                  "home_ewm_opp_tov", "away_ewm_opp_tov",
                  "home_ewm_opp_orb", "away_ewm_opp_orb")
_PLAYER_ELO_COLS = ("home_elo_available", "away_elo_available",
                    "home_elo_mean", "away_elo_mean",
                    "home_elo_top5", "away_elo_top5")

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    # name -> columns DROPPED from the full block set
    "shared_full": (),
    "no_home_split": _HOME_SPLIT_COLS,
    "no_opp_adjusted": _OPP_ADJ_COLS,
    "no_sos": _SOS_COLS,
    "no_opp_chars": _OPP_CHAR_COLS,
    "no_player_elo": _PLAYER_ELO_COLS,
    "home_split_only": (_OPP_ADJ_COLS + _SOS_COLS + _OPP_CHAR_COLS
                        + _PLAYER_ELO_COLS),
    "schedule_only": (_HOME_SPLIT_COLS + _OPP_ADJ_COLS + _SOS_COLS
                      + _OPP_CHAR_COLS + _PLAYER_ELO_COLS),
}


def build_frame(games: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    if feature_set not in FEATURE_SETS:
        raise KeyError(f"unknown feature set {feature_set!r}; "
                       f"known: {sorted(FEATURE_SETS)}")
    drop = set(FEATURE_SETS[feature_set])
    keep = lambda d: d.select_dtypes(include=[np.number])  # noqa: E731
    blocks = _base(games)
    order = ("roll", "schedule", "home_split", "player_elo")
    parts = [keep(blocks[n]).reset_index(drop=True) for n in order]
    frame = pd.concat(parts, axis=1)

    dupes = sorted({c for c in frame.columns if list(frame.columns).count(c) > 1})
    if dupes:
        raise AssertionError(f"two builders emitted the same column: {dupes}")

    frame = frame.drop(columns=[c for c in drop if c in frame.columns])
    frame = frame.loc[:, frame.nunique(dropna=False) > 1]
    if len(frame) != len(games):
        raise AssertionError(f"frame has {len(frame)} rows, expected {len(games)}")
    return frame.reset_index(drop=True)


# --------------------------------------------------------------------------
# Models. A Pipeline so that scaling and imputation are fitted per fold.
# --------------------------------------------------------------------------

def build_model(name: str, seed: int = SEED):
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  RandomForestClassifier)
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    if name == "logistic":
        return Pipeline([("i", SimpleImputer(strategy="median")),
                         ("s", StandardScaler()),
                         ("c", LogisticRegression(C=1.0, max_iter=2000,
                                                  random_state=seed))])
    if name == "logistic_C0.1":
        return Pipeline([("i", SimpleImputer(strategy="median")),
                         ("s", StandardScaler()),
                         ("c", LogisticRegression(C=0.1, max_iter=2000,
                                                  random_state=seed))])
    if name == "rf":
        return Pipeline([("i", SimpleImputer(strategy="median")),
                         ("c", RandomForestClassifier(
                             n_estimators=400, min_samples_leaf=20,
                             random_state=seed, n_jobs=-1))])
    if name == "ridge":
        # L2 logistic IS ridge regression under a logit link, and it is the
        # only version of that idea that can emit a probability.
        # `RidgeClassifier` was tried first and has no `predict_proba`, which
        # makes it unscoreable on this project's primary metric - a linear model
        # that cannot produce a probability cannot be compared to one that can.
        # Stronger L2 than E01's C=0.1, so the regularisation path is genuinely
        # different rather than the same point re-measured.
        return Pipeline([("i", SimpleImputer(strategy="median")),
                         ("s", StandardScaler()),
                         ("c", LogisticRegression(C=0.01, max_iter=2000,
                                                  random_state=seed))])
    if name == "gnb_pca":
        # Named as the best model in luke-lite/NBA-Prediction-Modeling (53 stars),
        # from a survey of ~45 public NBA repos. Included because it is a genuinely
        # different hypothesis class from every other entry: it assumes features
        # are conditionally INDEPENDENT given the class, which no linear model does
        # and which is plainly false for correlated team statistics. Worth failing
        # explicitly rather than assuming.
        from sklearn.decomposition import PCA
        from sklearn.naive_bayes import GaussianNB
        return Pipeline([("i", SimpleImputer(strategy="median")),
                         ("s", StandardScaler()),
                         ("p", PCA(n_components=20, random_state=seed)),
                         ("c", GaussianNB())])
    if name == "logistic_pca":
        # rileypsmith reports 80 principal components beating 10. Scaled to this
        # frame's ~59 columns, 20 is the comparable choice.
        from sklearn.decomposition import PCA
        return Pipeline([("i", SimpleImputer(strategy="median")),
                         ("s", StandardScaler()),
                         ("p", PCA(n_components=20, random_state=seed)),
                         ("c", LogisticRegression(C=1.0, max_iter=2000,
                                                  random_state=seed))])
    if name == "histgb":
        return Pipeline([("i", SimpleImputer(strategy="median")),
                         ("c", HistGradientBoostingClassifier(
                             max_depth=3, max_iter=300, learning_rate=0.05,
                             l2_regularization=1.0, random_state=seed))])
    if name == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise SystemExit("xgboost is not installed in this interpreter") from exc
        return Pipeline([("i", SimpleImputer(strategy="median")),
                         ("c", XGBClassifier(
                             n_estimators=400, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, random_state=seed,
                             n_jobs=-1, eval_metric="logloss"))])
    raise KeyError(f"unknown model {name!r}")


def feature_importance(model, columns: list[str], top: int = 8
                       ) -> list[tuple[str, float]]:
    """Coefficients or importances, whichever the final estimator exposes."""
    est = model.steps[-1][1]
    if hasattr(est, "coef_"):
        w = np.abs(est.coef_).ravel()
    elif hasattr(est, "feature_importances_"):
        w = np.asarray(est.feature_importances_).ravel()
    else:
        return []
    order = np.argsort(w)[::-1][:top]
    return [(columns[i], float(w[i])) for i in order]


def run_one(exp: Experiment, games: pd.DataFrame, splits) -> dict:
    from sklearn.metrics import log_loss

    t0 = time.time()
    frame = build_frame(games, exp.features)
    X = frame.to_numpy(dtype=float)
    columns = list(frame.columns)
    y = games["y"].to_numpy(int)

    oof = np.full(len(y), np.nan)
    train_scores, importances = [], []
    for tr, te in splits:
        m = build_model(exp.model, **exp.params)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        # In-sample score on the training half only, for the overfitting check.
        p_tr = m.predict_proba(X[tr])[:, 1]
        train_scores.append(log_loss(y[tr], np.clip(p_tr, 1e-9, 1 - 1e-9),
                                     labels=[0, 1]))
        importances.append(feature_importance(m, columns))

    scored = ~np.isnan(oof)
    m_oof = nd.evaluate(y[scored], oof[scored])
    train_loss = float(np.mean(train_scores))
    gap = train_loss - m_oof["log_loss"]

    # Average importance across folds, then rank.
    if importances and importances[0]:
        acc: dict[str, list[float]] = {}
        for fold in importances:
            for name, val in fold:
                acc.setdefault(name, []).append(val)
        ranked = sorted(((k, float(np.mean(v))) for k, v in acc.items()),
                        key=lambda kv: -kv[1])[:8]
    else:
        ranked = []

    return {
        "name": exp.name,
        "features": exp.features,
        "model": exp.model,
        "rationale": exp.rationale,
        "n_features": len(columns),
        "log_loss": round(m_oof["log_loss"], 5),
        "brier": round(m_oof["brier"], 5),
        "auc": round(m_oof["auc"], 5),
        "accuracy": round(m_oof["accuracy"], 4),
        "train_log_loss": round(train_loss, 5),
        "train_test_gap": round(gap, 5),
        "oof_rows": int(scored.sum()),
        "top_features": ranked,
        "seconds": round(time.time() - t0, 1),
    }


def append(rec: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def load_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    return [json.loads(line) for line in LOG_PATH.read_text().splitlines()
            if line.strip()]