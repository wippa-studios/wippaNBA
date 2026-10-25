"""Diagnostics and fixed-weight blends for the three strongest models.

Issue #42 asks a question that is easy to get backwards: before optimising an
ensemble, measure whether the members actually disagree.  A blend of three
near-duplicate models cannot help, and a blend of models that disagree on
different games might help even when their pooled scores are close.

This module is deliberately small:

* :func:`disagreement_table` reports pairwise disagreement rates and each
  model's hit rate on the disagreement rows.
* :func:`simplex_grid` enumerates every weight vector on a fixed grid.
* :func:`blend` is a weighted arithmetic mean of probabilities.

Weight selection is **not** done here.  The issue's primary arms use weights
fixed before the run (equal weights and two-model averages).  The grid is a
printed diagnostic; choosing its maximum would be a test-set fit and is
reported as such.
"""

from __future__ import annotations

from itertools import combinations
from typing import Mapping, Sequence

import numpy as np

__all__ = ["blend", "disagreement_table", "simplex_grid"]


def blend(probabilities: Mapping[str, np.ndarray],
          weights: Mapping[str, float]) -> np.ndarray:
    """Weighted arithmetic mean of probability vectors.

    Weights must be non-negative and sum to 1 (within float tolerance).  The
    check is explicit because a silently unnormalised blend is just a badly
    calibrated model with no record of why.
    """
    names = list(probabilities)
    if set(weights) != set(names):
        raise ValueError("weights must name exactly the probability keys")
    w = np.asarray([weights[n] for n in names], dtype=float)
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0, atol=1e-12):
        raise ValueError("weights must be non-negative and sum to one")
    p = np.zeros(len(next(iter(probabilities.values()))), dtype=float)
    for name, weight in zip(names, w):
        p += weight * np.asarray(probabilities[name], dtype=float)
    return p


def disagreement_table(y: np.ndarray,
                       probabilities: Mapping[str, np.ndarray]) -> dict:
    """Pairwise disagreement and conditional hit rates.

    ``hit_rate`` is the share of disagreement rows on which a model's 0.5
    threshold happens to match the label.  It is a descriptive statistic, not
    a significance test: two models can disagree on 50 rows and produce a
    difference of 12 that is pure noise.  The point of the table is to show
    whether the members are near-duplicates, not to rank them.
    """
    y = np.asarray(y, dtype=int)
    names = list(probabilities)
    pred = {n: (np.asarray(probabilities[n]) > 0.5).astype(int) for n in names}
    rows: dict[str, dict] = {}
    for a, b in combinations(names, 2):
        disagree = pred[a] != pred[b]
        n = int(disagree.sum())
        row = {
            "n_disagree": n,
            "disagreement_rate": float(n / len(y)),
            "n_total": int(len(y)),
        }
        for name in (a, b):
            if n:
                row[f"{name}_hit_rate_on_disagreement"] = float(
                    (pred[name][disagree] == y[disagree]).mean())
                row[f"{name}_mean_pred_on_disagreement"] = float(
                    np.asarray(probabilities[name])[disagree].mean())
            else:
                row[f"{name}_hit_rate_on_disagreement"] = float("nan")
                row[f"{name}_mean_pred_on_disagreement"] = float("nan")
        rows[f"{a} vs {b}"] = row
    return rows


def simplex_grid(names: Sequence[str], step: float = 0.1) -> list[dict[str, float]]:
    """Every weight vector on the ``step`` simplex, in lexicographic order.

    For three models and ``step=0.1`` this is 66 configurations.  No
    configuration is hidden: issue #42 explicitly requires the whole grid.
    """
    if step <= 0 or step > 1:
        raise ValueError("step must be in (0, 1]")
    units = int(round(1.0 / step))
    if not np.isclose(units * step, 1.0):
        raise ValueError("step must divide 1 exactly")

    out: list[dict[str, float]] = []

    def rec(prefix: list[int], remaining: int) -> None:
        if len(prefix) == len(names) - 1:
            vals = prefix + [remaining]
            out.append({name: float(v * step) for name, v in zip(names, vals)})
            return
        for v in range(remaining + 1):
            rec(prefix + [v], remaining - v)

    rec([], units)
    return out
