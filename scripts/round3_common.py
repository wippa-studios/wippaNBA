"""Round-3 harness. Reuses round2_common verbatim for geometry and scoring.

What round 3 changes about the BAR, and why that is stated up front
------------------------------------------------------------------
Round 2's preregistration fixed `LogisticRegression(C=1.0)` and said "no C
sweep, because changing it changes the bar". That is a real constraint on
COMPARABILITY, not an argument against the sweep: the bar is a model
configuration, and model optimisation is a legitimate axis. So round 3 does
both of the following and reports them as separate numbers:

  BAR  (frozen)      E14 @ C=1.0, 43 cols. Reproduced to 1e-9 against
                     results/round2_experiments.jsonl record R0_baseline_e14.
  CAND (re-tuned)    The same 43 columns, with C chosen per fold on that
                     fold's TRAINING rows only, via an inner time-ordered
                     validation block. Never touches test rows.

Every delta in round 3 is reported against BOTH, because a gain that only
appears once C is retuned is a gain that would not have been available to a
user still on C=1.0, and a gain measured against the frozen bar alone would be
overstated by the retuning itself.

SEED / SPLITS / METRIC are untouched: 5 walk-forward folds, initial_frac=0.5,
gap=0, seed 42, nd.evaluate + nd.paired_bootstrap_delta(n_boot=2000).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from scripts.round2_common import (              # noqa: E402,F401
    SEED, N_SPLITS, HALFLIFE,
    build_e14_frame, e14_pipeline, score, delta_vs,
    walk_forward_oof, verify_sign_convention, append,
)
from src.DataProviders import nba_data as nd    # noqa: E402

JSONL3 = ROOT / "results" / "round3_experiments.jsonl"


# --------------------------------------------------------------------------- #
# Inner validation: time-ordered tail of the TRAIN block only
# --------------------------------------------------------------------------- #
def inner_val_split(tr: np.ndarray, games: pd.DataFrame,
                    frac: float = 0.20) -> tuple[np.ndarray, np.ndarray]:
    """Split TRAIN indices into fit/validate by TIME, keeping the tail as val.

    A random inner split would let a validation row's near neighbours sit in
    the fit set, and every feature in this frame is a trailing EWM with a
    ~10-game halflife -- neighbouring rows share almost all of their inputs.
    That inflates the inner score and selects too small a C. Chronological is
    the only honest inner split for this frame.
    """
    d = games.iloc[tr]["GAME_DATE"].to_numpy()
    order = np.argsort(d, kind="stable")
    tr_sorted = tr[order]
    n_val = max(50, int(round(frac * len(tr_sorted))))
    return tr_sorted[:-n_val], tr_sorted[-n_val:]


def pipeline_for(C: float, penalty: str = "l2", seed: int = SEED,
                 l1_ratio: float | None = None):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    # sklearn 1.9 deprecates `penalty=` in favour of `l1_ratio`:
    # l1_ratio=0 is L2, 1 is L1, a float in between is elastic net.
    _RATIO = {"l2": 0.0, "l1": 1.0}
    ratio = l1_ratio if l1_ratio is not None else _RATIO[penalty]
    # lbfgs only implements the pure-L2 objective. Anything with l1_ratio > 0
    # has to go through saga, INCLUDING the pure-L1 case at exactly 1.0 -- the
    # `0 < ratio < 1` version of this silently sent l1_ratio=1.0 to lbfgs and
    # raised "Solver lbfgs supports only 'l2' or None penalties, got l1".
    solver = "saga" if ratio > 0.0 else "lbfgs"
    kw = dict(C=C, max_iter=4000, random_state=seed,
              l1_ratio=ratio, solver=solver)
    return Pipeline([("i", SimpleImputer(strategy="median")),
                     ("s", StandardScaler()),
                     ("c", LogisticRegression(**kw))])


C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def oof_fixed(X: np.ndarray, y: np.ndarray, games: pd.DataFrame,
              C: float = 1.0, penalty: str = "l2",
              l1_ratio: float | None = None,
              n_splits: int = N_SPLITS, seed: int = SEED) -> np.ndarray:
    p = np.full(len(y), np.nan)
    for tr, te in nd.walk_forward_split(games, n_splits=n_splits,
                                        initial_frac=0.5):
        m = pipeline_for(C, penalty, seed, l1_ratio)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


def oof_inner_selected(X: np.ndarray, y: np.ndarray, games: pd.DataFrame,
                       grid: list[float] | None = None,
                       penalty: str = "l2",
                       l1_ratio: float | None = None,
                       n_splits: int = N_SPLITS, seed: int = SEED
                       ) -> tuple[np.ndarray, list, list]:
    """Per-fold inner validation over `grid`. Returns OOF p, chosen C per fold,
    and each fold's inner validation loss for every C (the full grid is always
    printed -- no partial sweeps)."""
    grid = grid or C_GRID
    p = np.full(len(y), np.nan)
    chosen: list = []
    trace: list = []
    for k, (tr, te) in enumerate(nd.walk_forward_split(
            games, n_splits=n_splits, initial_frac=0.5)):
        fit_i, val_i = inner_val_split(tr, games)
        losses = []
        for C in grid:
            m = pipeline_for(C, penalty, seed, l1_ratio)
            m.fit(X[fit_i], y[fit_i])
            pv = m.predict_proba(X[val_i])[:, 1]
            losses.append(float(nd.evaluate(y[val_i], pv)["log_loss"]))
        j = int(np.argmin(losses))
        chosen.append({"fold": k, "C": grid[j],
                       "inner_val_log_loss": round(losses[j], 6),
                       "n_val": int(len(val_i)), "n_fit": int(len(fit_i))})
        m = pipeline_for(grid[j], penalty, seed, l1_ratio)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
        trace.append({"fold": k, "C": grid[j],
                      "inner_grid_log_loss": {str(c): round(l, 6)
                                              for c, l in zip(grid, losses)}})
    return p, chosen, trace


def scored_mask(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return np.isfinite(p)


def compare(y: np.ndarray, p: np.ndarray, p_base: np.ndarray,
            seed: int = SEED) -> dict:
    m = scored_mask(y, p) & scored_mask(y, p_base)
    d = delta_vs(y[m], p[m], p_base[m], seed=seed)
    s = score(y[m], p[m])
    return {"n": int(m.sum()), "log_loss": round(s["log_loss"], 6),
            "brier": round(s["brier"], 6), "auc": round(s["auc"], 6),
            "accuracy": round(s["accuracy"], 6),
            "improvement_nats": round(d["improvement_nats"], 6),
            "ci_lo": round(d["ci_lo"], 6), "ci_hi": round(d["ci_hi"], 6),
            "abs_z": round(d["abs_z"], 3)}


def log3(rec: dict) -> None:
    rec = dict(rec)
    rec["ts_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    JSONL3.parent.mkdir(parents=True, exist_ok=True)
    with JSONL3.open("a") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")
    print(json.dumps({k: v for k, v in rec.items()
                      if k not in ("inner_trace",)}, default=str)[:1400], flush=True)


def load_cached():
    g = pd.read_pickle("/home/Joel/.cache/nba-frames/games.pkl")
    f = pd.read_pickle("/home/Joel/.cache/nba-frames/e14.pkl")
    return g, f


if __name__ == "__main__":
    print(verify_sign_convention())