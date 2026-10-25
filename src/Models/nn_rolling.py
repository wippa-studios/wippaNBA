"""Feed-forward neural network on the canonical 26-feature frame.

MODEL M01 of the suite.  The canonical model is an L2-penalised logistic
regression; this arm asks whether a small multilayer perceptron can learn
non-linear interactions that logistic regression cannot represent, on the
**identical** frame, folds, preprocessing contract and evaluation harness.

WHY AN MLP AND NOT THE LEGACY KERAS MODELS
------------------------------------------
``src/Train-Models/NN_Model_ML.py`` is the original pipeline's Keras trainer.
It expects the scraped odds dataset, writes a different artifact contract and
was never wired to the Pipeline A walk-forward split.  This arm is deliberately
the smallest honest neural-network test: a scikit-learn ``MLPClassifier`` on the
same 26 columns.  It inherits the canonical ``FeaturePipeline`` (train-only
median impute and z-score), so the only variable is the model class.  A deeper
Keras arm would be a separate pre-registration, not a replacement for this one.

PRE-REGISTERED, NOT TUNED ON THE TEST BLOCK
-------------------------------------------
``results/experiments.jsonl`` carries the declaration ``M01-prereg-nn-rolling``
written before this module ran.  The primary arm is the mean prediction of five
seeds; a single favourable seed cannot promote.  The architecture, preprocessing
and decision rule are fixed there.  The sensitivity configurations are
descriptive only and are reported whether they win or lose.

SCORING
-------
``measurement.canonical_oof`` fits the reference on the same folds; ``delta``
is ``loss(canonical) - loss(MLP)``, so **positive means the network is better**.
Promotion requires a positive seed-averaged delta with ``|z| > 3`` on 2,000
paired bootstrap resamples (seed 42), the project's standing bar.

Run::

    OMP_NUM_THREADS=1 python -m src.Models.nn_rolling
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from src.DataProviders import nba_data as nd
from src.Models import logreg_rolling as lr
from src.Models import measurement as ms

ROOT = Path(__file__).resolve().parents[2]
OUT_JSON = ROOT / "results" / "nn_rolling.json"
OUT_MD = ROOT / "results" / "nn_rolling.md"
JSONL = ROOT / "results" / "experiments.jsonl"

#: The post-#36 canonical pooled OOF log loss recorded in results_pipeline_a.md.
RECORDED_CANONICAL = 0.615663649

#: Fixed before execution (M01-prereg-nn-rolling).
PRIMARY_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

#: Fixed before execution.  ``max_iter`` is a cap; early stopping normally ends
#: the fit sooner.  15% of each training block is used for early stopping -- an
#: inner holdout, never a test row.
PRIMARY_ARCH: dict = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "batch_size": 64,
    "learning_rate_init": 1e-3,
    "max_iter": 600,
    "tol": 1e-6,
    "n_iter_no_change": 20,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "shuffle": True,
}

#: Descriptive robustness only; cannot promote (see the pre-registration).
SENSITIVITY: tuple[dict, ...] = (
    {"label": "one_hidden_32", "hidden_layer_sizes": (32,), "alpha": 1e-4,
     "early_stopping": True},
    {"label": "wide_128_64", "hidden_layer_sizes": (128, 64), "alpha": 1e-4,
     "early_stopping": True},
    {"label": "strong_l2_1e-2", "hidden_layer_sizes": (64, 32), "alpha": 1e-2,
     "early_stopping": True},
    {"label": "no_early_stopping", "hidden_layer_sizes": (64, 32),
     "alpha": 1e-4, "early_stopping": False, "max_iter": 400},
)

_CORE_METRICS = ("n", "log_loss", "brier", "accuracy", "auc")


def _estimator(seed: int, arch: dict) -> MLPClassifier:
    return MLPClassifier(**{**arch, "random_state": int(seed)})


def oof_mlp(X: pd.DataFrame, y: np.ndarray, games: pd.DataFrame,
            seeds: tuple[int, ...] = PRIMARY_SEEDS,
            arch: dict | None = None, n_splits: int = 5
            ) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """Walk-forward MLP probabilities, their scored-row mask, and per-seed OOF.

    ``X`` must be the canonical 26-column frame in canonical order.  The
    preprocessing object is fitted on each training block and never on the test
    block; the same folds are ``nba_data.walk_forward_split``'s, i.e. the ones
    the canonical model and every other suite model use.  Row order is
    load-bearing: no caller may re-sort ``games`` between building ``X`` and
    calling this function.
    """
    arch = dict(PRIMARY_ARCH if arch is None else arch)
    features = list(X.columns)
    y = np.asarray(y, dtype=int)
    if len(X) != len(games) or len(y) != len(games):
        raise ValueError("X, y and games must have the same number of rows")
    p_by_seed = {int(s): np.full(len(y), np.nan) for s in seeds}
    for tr, te in nd.walk_forward_split(games, n_splits=n_splits):
        prep = lr.FeaturePipeline(columns=features).fit(X.iloc[tr])
        Xtr = prep.transform(X.iloc[tr]).to_numpy(dtype=float)
        Xte = prep.transform(X.iloc[te]).to_numpy(dtype=float)
        for seed in seeds:
            est = _estimator(seed, arch)
            # Single-threaded BLAS keeps repeated fits bit-identical on this
            # data; the arm is seed-averaged, so it must not also be a lottery
            # over thread scheduling.
            with threadpool_limits(limits=1, user_api="blas"):
                est.fit(Xtr, y[tr])
                p_by_seed[int(seed)][te] = est.predict_proba(Xte)[:, 1]
    masks = [np.isfinite(p) for p in p_by_seed.values()]
    if not all(np.array_equal(masks[0], m) for m in masks[1:]):
        raise AssertionError("seeds scored different rows")
    p_mean = np.vstack(list(p_by_seed.values())).mean(axis=0)
    return p_mean, masks[0], p_by_seed


def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    out = nd.evaluate(y, p)
    return {k: (int(out[k]) if k == "n" else float(out[k]))
            for k in _CORE_METRICS}


def _delta(y: np.ndarray, p_candidate: np.ndarray,
           p_reference: np.ndarray, mask: np.ndarray) -> dict:
    d = ms.delta_stats(y[mask], p_candidate[mask], p_reference[mask],
                       n_boot=2000, seed=42)
    d["bootstrap_seed"] = 42
    return d


def main() -> int:  # pragma: no cover - entry point
    t0 = time.time()
    games = nd.build_games().reset_index(drop=True)
    y = games["y"].to_numpy(dtype=int)
    X_all = lr.load_design(games=games)
    features = lr.resolve_features(lr.DEFAULT_MODELS["full"])
    X = X_all[features].reset_index(drop=True)
    if X.shape[1] != 26:
        raise AssertionError(f"expected the canonical 26 columns, got {X.shape[1]}")

    # Reference: the canonical model, same frame, same folds, same
    # train-only FeaturePipeline.
    _, p_ref, mask_ref = ms.canonical_oof(games=games)

    p_nn, mask_nn, per_seed = oof_mlp(X, y, games)
    if not np.array_equal(mask_ref, mask_nn):
        raise AssertionError("reference and MLP scored different rows")

    mask = mask_ref
    ref_metrics = _metrics(y[mask], p_ref[mask])
    nn_metrics = _metrics(y[mask], p_nn[mask])
    delta = _delta(y, p_nn, p_ref, mask)
    per_fold = ms.per_fold_delta(y, p_nn, p_ref, games)

    seed_rows = {}
    for seed, p in per_seed.items():
        d = _delta(y, p, p_ref, mask)
        seed_rows[str(seed)] = {
            "log_loss": float(nd.evaluate(y[mask], p[mask])["log_loss"]),
            "delta": d,
        }
    seed_deltas = np.array([r["delta"]["delta_candidate_minus_reference"]
                            for r in seed_rows.values()], dtype=float)
    seed_losses = np.array([r["log_loss"] for r in seed_rows.values()],
                           dtype=float)

    sensitivity = {}
    for cfg in SENSITIVITY:
        arch = {k: v for k, v in cfg.items() if k != "label"}
        p_s, mask_s, _ = oof_mlp(X, y, games, seeds=(42,), arch=arch)
        if not np.array_equal(mask_s, mask):
            raise AssertionError(f"sensitivity {cfg['label']} scored other rows")
        d = _delta(y, p_s, p_ref, mask)
        sensitivity[cfg["label"]] = {
            "config": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in arch.items()},
            "log_loss": float(nd.evaluate(y[mask], p_s[mask])["log_loss"]),
            "delta": d,
        }

    drift = ref_metrics["log_loss"] - RECORDED_CANONICAL
    reproduced = abs(drift) <= 1e-6
    if not reproduced:
        verdict = (
            "FAILED REPRODUCTION: the canonical reference came back "
            f"{ref_metrics['log_loss']:.9f}, not {RECORDED_CANONICAL:.9f} "
            f"(drift {drift:+.2e}). The MLP delta is reported for diagnosis "
            "only."
        )
    elif delta["delta_candidate_minus_reference"] > 0 and delta["abs_z"] > 3:
        verdict = (
            "NN CLEARS THE CANONICAL MODEL: the five-seed MLP beats the "
            f"canonical logistic by {delta['delta_candidate_minus_reference']:+.6f} "
            f"nats on the same {int(mask.sum())} walk-forward rows "
            f"(CI [{delta['ci_lo']:+.6f}, {delta['ci_hi']:+.6f}], |z| "
            f"{delta['abs_z']:.2f}), better in "
            f"{per_fold['n_folds_candidate_better']}/{per_fold['n_folds']} "
            "folds. This is a new model class clearing the |z| > 3 bar, not a "
            "feature-block result."
        )
    elif delta["delta_candidate_minus_reference"] > 0:
        verdict = (
            "NOT PROMOTED: the five-seed MLP is "
            f"{delta['delta_candidate_minus_reference']:+.6f} nats better than "
            f"the canonical model (CI [{delta['ci_lo']:+.6f}, "
            f"{delta['ci_hi']:+.6f}], |z| {delta['abs_z']:.2f}, better in "
            f"{per_fold['n_folds_candidate_better']}/{per_fold['n_folds']} "
            "folds), but below the pre-registered |z| > 3 bar. Positive is not "
            "promoted."
        )
    else:
        verdict = (
            "NULL/NEGATIVE: the five-seed MLP does not beat the canonical "
            f"model. Its point estimate is "
            f"{delta['delta_candidate_minus_reference']:+.6f} nats "
            f"(CI [{delta['ci_lo']:+.6f}, {delta['ci_hi']:+.6f}], |z| "
            f"{delta['abs_z']:.2f}), better in "
            f"{per_fold['n_folds_candidate_better']}/{per_fold['n_folds']} "
            "folds. The interval includes zero, so this is not evidence of a "
            "difference either way; it is certainly not an improvement. A "
            "non-linear model class adds no measurable signal on this frame."
        )

    payload = {
        "issue": "M01",
        "protocol_ref": "M01-prereg-nn-rolling",
        "generated_utc": ms.utc_now(),
        "runtime_seconds": round(time.time() - t0, 1),
        "population": "build_games(), all seasons",
        "n_games": int(len(games)),
        "n_scored": int(mask.sum()),
        "frame_columns": features,
        "architecture": {k: (list(v) if isinstance(v, tuple) else v)
                         for k, v in PRIMARY_ARCH.items()},
        "seeds": list(PRIMARY_SEEDS),
        "recorded_canonical_log_loss": RECORDED_CANONICAL,
        "reproduced_canonical_log_loss": ref_metrics["log_loss"],
        "reproduction_drift": drift,
        "reproduced_within_tolerance": bool(reproduced),
        "metrics": {"canonical": ref_metrics, "mlp_seed_average": nn_metrics},
        "delta_mlp_vs_canonical": delta,
        "per_fold": per_fold,
        "per_seed": seed_rows,
        "seed_dispersion": {
            "delta_mean": float(seed_deltas.mean()),
            "delta_sd": float(seed_deltas.std(ddof=1)),
            "log_loss_mean": float(seed_losses.mean()),
            "log_loss_sd": float(seed_losses.std(ddof=1)),
            "best_seed_delta": float(seed_deltas.max()),
            "worst_seed_delta": float(seed_deltas.min()),
        },
        "sensitivity": sensitivity,
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ms.append_jsonl(JSONL, {
        "name": "M01-run-nn-rolling",
        "protocol_ref": "M01-prereg-nn-rolling",
        "declared_utc": ms.utc_now(),
        "experiment_type": (
            "new model class: feed-forward neural network on the canonical "
            "26-feature frame (Pipeline A)"
        ),
        "n_scored": int(mask.sum()),
        "metrics": payload["metrics"],
        "reproduced_canonical_log_loss": ref_metrics["log_loss"],
        "reproduction_drift": drift,
        "delta_mlp_vs_canonical": delta,
        "per_fold": per_fold,
        "per_seed": seed_rows,
        "seed_dispersion": payload["seed_dispersion"],
        "sensitivity": sensitivity,
        "verdict": verdict,
        "status": "executed",
    })

    L = ["=" * 78,
         "M01 -- FEED-FORWARD NETWORK ON THE CANONICAL 26-FEATURE FRAME",
         "=" * 78]
    a = L.append
    a(f"games {len(games)}; scored {int(mask.sum())}; "
      f"canonical columns {X.shape[1]}")
    a(f"architecture {PRIMARY_ARCH['hidden_layer_sizes']}; "
      f"seeds {list(PRIMARY_SEEDS)}; alpha {PRIMARY_ARCH['alpha']}")
    a(f"canonical reproduced {ref_metrics['log_loss']:.9f} "
      f"(recorded {RECORDED_CANONICAL:.9f}, drift {drift:+.2e})")
    a("")
    a(f"{'model':<28} {'log loss':>10} {'brier':>9} {'accuracy':>9} {'auc':>7}")
    for name, met in (("canonical logistic", ref_metrics),
                      ("MLP, 5-seed average", nn_metrics)):
        a(f"{name:<28} {met['log_loss']:>10.6f} {met['brier']:>9.6f} "
          f"{met['accuracy']:>9.4f} {met['auc']:>7.4f}")
    a("")
    a(f"delta (MLP - canonical)      "
      f"{delta['delta_candidate_minus_reference']:+.6f} "
      f"CI [{delta['ci_lo']:+.6f}, {delta['ci_hi']:+.6f}] "
      f"|z| {delta['abs_z']:.2f}")
    a(f"folds better: {per_fold['n_folds_candidate_better']}/"
      f"{per_fold['n_folds']}; seed delta sd "
      f"{payload['seed_dispersion']['delta_sd']:.6f}")
    a("")
    a("per seed (descriptive):")
    for seed, row in seed_rows.items():
        a(f"  {seed:>2}  log loss {row['log_loss']:.6f}  "
          f"delta {row['delta']['delta_candidate_minus_reference']:+.6f}  "
          f"|z| {row['delta']['abs_z']:.2f}")
    a("")
    a("sensitivity (descriptive, cannot promote):")
    for label, row in sensitivity.items():
        a(f"  {label:<20} log loss {row['log_loss']:.6f}  "
          f"delta {row['delta']['delta_candidate_minus_reference']:+.6f}  "
          f"|z| {row['delta']['abs_z']:.2f}")
    a("")
    a(verdict)
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0 if reproduced else 1


if __name__ == "__main__":
    raise SystemExit(main())
