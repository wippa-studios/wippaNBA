"""Batch C / issue #44: spread and total markets, tested at last.

Pre-registered in results/experiments.jsonl as ``C44-prereg-spread-total``
BEFORE any measurement (``python scripts/c44_spread_total.py --prereg``).
That entry governs. This script refuses to run without it.

WHAT ISSUE #44 ASKS
-------------------
Everything measured in this repo so far concerns the moneyline. The margin
model (``src/Models/margin_total.py``: margin RMSE 14.01, total RMSE 18.78,
spread MAE 11.0) already predicts the two quantities the other two markets are
made of - by how much, and how many points - and neither the spread nor the
total has ever been evaluated against its closing line. This run evaluates both,
and tests two pre-registered hypotheses about the model itself:

C1  pace-first decomposition beats the monolithic margin/total model.
    Predict possessions first, then points-per-possession per side (see
    ``src/Models/pace_first.py``), derive margin and total from the two team
    points. Target: margin RMSE AND total RMSE better than ``margin_total``'s
    on identical walk-forward rows, CI excluding zero, |z| > 3.

C2  a proper margin distribution beats Phi(mhat / sigma).
    Primary distribution: the SKELLAM on (mu_home, mu_away) =
    ((t_hat + m_hat)/2, (t_hat - m_hat)/2) from ``margin_total``'s own monolithic
    predictions - the margin as a difference of two scoring counts, with a
    game-specific variance (mu_h + mu_a = the predicted total) instead of one
    global sigma. ZERO additional fitted parameters: every quantity is either
    the monolithic model's train-only fit or a Poisson rate. The alternative
    (a quantile / dispersion model, e.g. a Gaussian margin with a predicted
    per-game scale) is named NOT-RUN: each fitted dispersion layer is another
    train-only parameter and another look, and this family is already at its
    look budget. Scored on (a) the margin-bucket log score and (b) cover
    probabilities against the closing spread and over probabilities against the
    closing total. Pre-registered minimum: >= 0.002 nats on the margin log score
    OR a demonstrable calibration improvement on cover probabilities, either
    with the 95% CI excluding zero and |z| > 3.

DATA INVENTORY (measured before this declaration; documented here)
-----------------------------------------------------------------
* ``Data/OddsData.sqlite`` carries per-game CLOSING SPREADS AND TOTALS, not
  just moneylines: 13,424 reconciled rows over 2015-16..2025-26 (twin
  spellings collapsed by ``scripts/closing_line.py::load_odds`` - reused here,
  never reimplemented), every row with OU and Spread non-null. C2's market
  scope is therefore FULL: no reduction to C2b.
* The spread's sign convention CHANGES BY ERA and is fixed here from the
  inventory diagnostics (cover rate under each convention, frac(spread > 0),
  corr(spread, margin)): 2022-23..2025-26 store the signed expected HOME
  margin (cover iff margin > Spread; cover rate 47.6-52.0%, mean residual
  -0.34..+0.24); 2015-16..2021-22 store the UNSIGNED handicap of the favourite
  (spread >= 0 in 96-98% of rows; cover iff margin > +Spread when the home side
  is the de-vigged moneyline favourite, else margin > -Spread; cover rate
  47.3-51.1% after resolution vs 39.5-43.0% naively). Fixed per row as L_eff
  before any arm is scored.
* Corrupt-quote exclusion, fixed ex ante: rows with |Spread| > 40 or OU outside
  [150, 290] are excluded from market scoring and counted (13 rows on the
  shipped database; e.g. 2019-20 Spread=216.0 / OU=1105.0).
* Pushes (margin == L_eff, total == OU) have no binary outcome and are excluded
  from cover / over scoring, counted and reported.
* Per-quarter scores DO NOT EXIST in the banked data: ``scripts/
  download_nba_data.py`` defines a ``summary/`` LineScore stage but
  ``data/nba/summary/`` was never downloaded. The margin distribution is
  therefore GAME-level only - no quarter decomposition is run and none is
  claimed.
* The odds tables carry no juice. Spread and total bets are settled at a flat
  -110 (p_raw = 110/210), stated as an assumption, not measured.

MARKET TESTS (issue #44's acceptance criteria, all reported)
------------------------------------------------------------
MAE and RMSE of the model's margin vs the closing spread AS A MARGIN FORECAST
(the line is itself a prediction of the margin: |L_eff - margin| vs
|m_hat - margin| on identical rows), the same for the total vs OU, calibration
of P(cover) and P(over) in 10 bins, and ROI of the parameter-free rule (bet the
side the model makes more likely whenever its probability beats the posted
price - both sides considered, no threshold; ``scripts/closing_line.py::roi``
reused so the corrected denominator applies) for the spread and the total. The
moneyline prior carries over: the moneyline rule returned -10.80% over 3,440
bets; "efficient" is the prior here too.

MULTIPLICITY
------------
6 looks in the Batch C family: C1 x 2 endpoints (margin RMSE, total RMSE),
C2 x 2 endpoints (margin log score, cover log loss), market x 2 (spread ROI,
total ROI, on the baseline Phi(m/sigma) arm). Bonferroni 0.05/6 = 0.0083
one-sided. The project's |z| > 3 bar (~0.00135 one-sided) is stricter and
governs. Seeds 137 / 2718 are replications of the same rows (CI variability
only, issue #45 erratum) and cannot promote a result. Secondary and
descriptive metrics (5-point-bucket log score, Brier, AUC, ECE, per-season
breakdowns, total-over scores, Skellam ROI, line-as-forecast comparisons) carry
no decision weight and add no looks.

SIGN CONVENTIONS (the repo's, restated because two loss types meet here)
-----------------------------------------------------------------------
Loss deltas: delta = loss(baseline) - loss(candidate); POSITIVE = candidate is
better (``nba_data.paired_bootstrap_delta``). RMSE deltas: delta =
RMSE(baseline) - RMSE(candidate); POSITIVE = candidate is better. Log-score
deltas are expressed as log-LOSS deltas so the sign is the same everywhere.
SE = (ci_hi - ci_lo) / (2 * 1.959964); |z| = delta / SE.

Run:  python scripts/c44_spread_total.py --prereg   # declaration only
      python scripts/c44_spread_total.py            # measurement (needs the above)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, skellam

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from src.DataProviders import nba_data as nd            # noqa: E402
from src.Models import margin_total as mt               # noqa: E402
from src.Models import pace_first as pf                 # noqa: E402

# scripts/closing_line.py is imported, never modified and never reimplemented:
# its load_odds twin-spelling reconciliation, its join accounting and its roi()
# (with the fixed betting denominator) are all reused as-is.
_spec = importlib.util.spec_from_file_location(
    "closing_line", ROOT / "scripts" / "closing_line.py")
cl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cl)
roi = cl.roi   # reused, not reimplemented: the corrected betting denominator

JSONL = ROOT / "results" / "experiments.jsonl"
OUT_JSON = ROOT / "results" / "c44_spread_total.json"
OUT_MD = ROOT / "results" / "c44_report.md"
CACHE = Path("/tmp/opencode/c44/preds.pkl")

PREREG_NAME = "C44-prereg-spread-total"
SEEDS = (42, 137, 2718)
PRIMARY_SEED = 42
N_SPLITS = 5
INITIAL_FRAC = 0.5
N_BOOT = 2000
Z_BAR = 3.0
MIN_EFFECT_NATS = 0.002
N_LOOKS = 6
BONFERRONI = 0.05 / N_LOOKS

P_PRICE = 110.0 / 210.0          # flat -110, both sides; no juice in the data
SPREAD_MAX_ABS = 40.0            # corrupt-quote bounds, fixed ex ante
OU_BOUNDS = (150.0, 290.0)
SIGNED_ERA_FROM = "2022-23"      # signed expected-home-margin from here on

PROB_FLOOR = 1e-15               # numerical floor on bucket probabilities


# --------------------------------------------------------------------------- #
# Pre-registration
# --------------------------------------------------------------------------- #
def prereg_entry() -> dict:
    return {
        "name": PREREG_NAME,
        "status": "declared_before_execution",
        "declared_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "Batch C (issue #44): pre-registered measurement of "
                           "margin-distribution models on the spread and total "
                           "markets",
        "hypothesis": {
            "C1_pace_first": "A pace-first decomposition (possessions first, "
                "then points-per-possession per side; margin and total derived "
                "from the same two team-point predictions) beats the monolithic "
                "margin/total ridge on BOTH margin RMSE and total RMSE over "
                "identical walk-forward test rows: delta > 0, 95% CI excluding "
                "zero, |z| > 3. Baseline is margin_total run as-is (its "
                "documented 14.01 / 18.78 must reproduce).",
            "C2_distributional_margin": "A proper margin distribution beats "
                "Phi(mhat/sigma): the SKELLAM on (mu_home, mu_away) = "
                "((t_hat+m_hat)/2, (t_hat-m_hat)/2) from margin_total's own "
                "predictions - integer-valued like a margin, game-specific "
                "variance mu_h+mu_a = predicted total, zero additional fitted "
                "parameters - scores better than the single-scale Gaussian on "
                "(a) the margin-bucket log score (>= 0.002 nats, CI excluding "
                "zero, |z| > 3) OR (b) cover-probability calibration against "
                "the closing spread (cover log-loss delta, CI excluding zero, "
                "|z| > 3). Total-over probabilities vs the closing total are "
                "scored and reported but carry no decision weight.",
        },
        "arms": {
            "baseline": "margin_total.MarginTotalModel() with default "
                "parameters (alpha auto, sigma oof, poisson auto), fit per "
                "walk-forward split on training rows only, module unmodified.",
            "C1": "pace_first.PaceFirstModel() (3 ridges: possessions, home "
                "ppp, away ppp), same folds, same ridge grid and the same "
                "inner-CV selector imported from margin_total.",
            "C2": "Skellam(mu_home, mu_away) over the baseline's (m_hat, t_hat) "
                "vs the baseline's own N(m_hat, sigma^2). The comparison "
                "isolates distribution form: same means, same folds, same rows.",
            "named_not_run": "quantile / dispersion model (Gaussian margin with "
                "a predicted per-game scale); Skellam over the C1 arm's team "
                "points; any threshold sweep. Not run: each is an extra fitted "
                "dispersion layer and/or an extra look.",
        },
        "parameterization_fixed_ex_ante": {
            "splits": "nba_data.walk_forward_split(n_splits=5, "
                "initial_frac=0.5, gap=0); identical rows across every arm",
            "features": "the shared causal builders only "
                "(nba_data.rolling_team_features halflife 10 min_games 5, "
                "nba_data.rest_and_travel); pace_first.possessions_target is an "
                "outcome-derived TRAINING TARGET (Dean Oliver box-score "
                "possessions), never a feature",
            "poss_clip": list(pf.POSS_CLIP),
            "skellam_rates": "mu = clip((t_hat +/- m_hat)/2, 0.01, None); "
                "t_hat = margin_total predict_total_gaussian, m_hat = "
                "predict_margin",
            "margin_buckets": "primary: exact integer margin (bucket = one "
                "point; Gaussian arm scores Phi((k+0.5-m)/sigma) - "
                "Phi((k-0.5-m)/sigma), Skellam arm scores its pmf at k); "
                "robustness (no decision weight): 5-point buckets "
                "[5j-2.5, 5j+2.5]",
            "cover_rule": "cover iff margin > L_eff; K = floor(L_eff)+1; "
                "P(cover) = P(margin >= K) under each arm's own distribution; "
                "pushes (margin == L_eff) excluded, counted",
            "L_eff": "2022-23..2025-26: Spread (signed expected home margin). "
                "2015-16..2021-22: unsigned favourite handicap - +Spread if the "
                "home side is the de-vigged moneyline favourite else -Spread. "
                "Fixed from the inventory cover-rate diagnostics before any "
                "arm ran",
            "market_rows": "walk-forward test rows with a closing line, "
                f"|Spread| <= {SPREAD_MAX_ABS}, OU in {list(OU_BOUNDS)}; "
                "identical rows across arms",
            "roi_rule": "closing_line.roi() reused with threshold=0.5 at a flat "
                "p_raw = 110/210: bet the side the model makes more likely "
                "whenever its probability beats the posted price; both sides "
                "considered; no free threshold; stake = edge over the price "
                "(roi()'s own convention)",
        },
        "metrics": "C1: pooled per-game margin RMSE and total RMSE (delta vs "
            "baseline, paired bootstrap over games, 2000 resamples) + per-split "
            "means for parity with 14.01/18.78 + MAE. C2 (a): per-game margin "
            "log score (log loss in nats) on identical rows. C2 (b): cover log "
            "loss + Brier + 10-bin calibration/ECE for both arms. Secondary: "
            "5-point-bucket log score, total-over log loss and calibration, "
            "per-season breakdowns of every primary endpoint, spread/total MAE "
            "and RMSE vs the closing line as a forecaster, ROI. SE = "
            "(ci_hi-ci_lo)/(2*1.959964); |z| = delta/SE.",
        "decision_rule": {
            "C1": "succeeds iff BOTH margin-RMSE and total-RMSE deltas are > 0 "
                "with 95% CI excluding zero and |z| > 3 at the primary setting "
                "(seed 42, 5 folds), with the sign stable at seeds 137 and "
                "2718. Otherwise 'sub-bar lead' (CI excludes zero, |z| <= 3) "
                "or 'null' (CI straddles zero).",
            "C2": "succeeds iff (a) margin-log-score delta >= 0.002 nats with "
                "CI excluding zero and |z| > 3, OR (b) cover-log-loss delta > 0 "
                "with CI excluding zero and |z| > 3, at the primary setting, "
                "sign stable at the replication seeds.",
            "market": "an ROI at |z| > 3 would be a finding; the prior is "
                "efficient (ROI <= 0). A null is a complete result.",
        },
        "multiplicity": f"{N_LOOKS} looks in the Batch C family (C1 x 2, C2 x 2, "
            f"market ROI x 2) -> Bonferroni 0.05/{N_LOOKS} = "
            f"{BONFERRONI:.4f} one-sided. The |z| > 3 bar (~0.00135 one-sided) "
            "is stricter and governs. Replication seeds resample the same rows "
            "and add no looks; secondary metrics cannot promote a result.",
        "leakage_plan": "All features strictly pre-tip-off via the shared "
            "causal builders (shift-within-team, truncation-invariant). Every "
            "fitted quantity (imputer, scaler, ridge penalty and coefficients, "
            "sigma, sigma_total, the Poisson fit) is fit on TRAIN rows only, "
            "inside the walk-forward loop; dispersion parameters are train-only "
            "by construction and the Skellam arm fits NOTHING beyond them. The "
            "odds columns (Spread/OU/ML) are used only as market lines and "
            "prices to score against, never as features; the odds tables' own "
            "outcome columns (Points/Win_Margin) are never consumed - outcomes "
            "come from nba_data.build_games. Identical scored rows across arms "
            "is asserted in code.",
        "runtime_note": "Both arms fit per fold with inner CV; roughly 5-15 "
            "minutes total, cached under /tmp/opencode/c44/. No search, no "
            "sweep, no variants - selection cost is zero because nothing is "
            "selected.",
    }


# --------------------------------------------------------------------------- #
# Inventory (data documentation, not measurement)
# --------------------------------------------------------------------------- #
def inventory() -> dict:
    rep: dict = {}
    con_ = cl.ODDS_DB
    rep["odds_db"] = str(con_)
    stats: dict = {}
    odds = cl.load_odds(stats=stats)
    rep["odds_rows_reconciled"] = int(len(odds))
    rep["odds_rows_expected"] = int(stats["rows_expected"])
    rep["odds_rows_collapsed"] = int(stats["rows_collapsed"])
    per_season = {}
    for s, g in odds.groupby("SEASON"):
        sp = pd.to_numeric(g["Spread"], errors="coerce")
        ou = pd.to_numeric(g["OU"], errors="coerce")
        per_season[s] = {
            "rows": int(len(g)),
            "spread_non_null": int(sp.notna().sum()),
            "ou_non_null": int(ou.notna().sum()),
            "frac_spread_positive": round(float((sp > 0).mean()), 4),
            "spread_min": float(sp.min()), "spread_max": float(sp.max()),
        }
    rep["per_season_tables"] = per_season
    rep["quarter_level_scores"] = (
        "NOT BANKED: data/nba/summary/ (BoxScoreSummaryV2 -> LineScore) was "
        "never downloaded; scripts/download_nba_data.py defines the stage but "
        "no summary/ directory or LineScore CSV exists. No per-quarter margins "
        "are available; the margin distribution is game-level only.")
    rep["juice"] = ("no price columns beyond ML_Home/ML_Away exist; spread and "
                    "total bets are settled at a flat -110 (p_raw = 110/210)")
    return rep


def effective_line(joined: pd.DataFrame) -> pd.DataFrame:
    """L_eff per row: the home-cover threshold under the era convention.

    2022-23..2025-26: Spread is the signed expected home margin -> L_eff = Spread.
    2015-16..2021-22: Spread is the unsigned handicap of the favourite ->
    L_eff = +Spread if the de-vigged moneyline favours the home side, else
    -Spread. (p_market >= 0.5 -> home favourite; a pick'em has Spread 0 and the
    branch does not matter.)
    """
    out = joined.copy()
    sp = pd.to_numeric(out["Spread"], errors="coerce")
    signed = out["SEASON"] >= SIGNED_ERA_FROM
    fav_home = out["p_market"].astype(float) >= 0.5
    out["L_eff"] = np.where(signed, sp, np.where(fav_home, sp, -sp))
    out["era_signed"] = signed
    return out


# --------------------------------------------------------------------------- #
# Distributional scoring helpers
# --------------------------------------------------------------------------- #
def gaussian_bucket_logprob_exact(k: np.ndarray, m: np.ndarray,
                                  sigma: float) -> np.ndarray:
    """log P(margin = k) under N(m, sigma^2) discretised to integer points."""
    upper = norm.cdf((k + 0.5 - m) / sigma)
    lower = norm.cdf((k - 0.5 - m) / sigma)
    return np.log(np.clip(upper - lower, PROB_FLOOR, None))


def gaussian_bucket_prob_wide(k: np.ndarray, m: np.ndarray, sigma: float
                              ) -> np.ndarray:
    """P(5-point bucket) under N(m, sigma^2): bucket [5j-2.5, 5j+2.5]."""
    j = np.floor((k + 2.5) / 5.0)
    return np.clip(norm.cdf((5 * j + 2.5 - m) / sigma)
                   - norm.cdf((5 * j - 2.5 - m) / sigma), PROB_FLOOR, None)


def skellam_bucket_prob_wide(k: np.ndarray, mu_h: np.ndarray,
                             mu_a: np.ndarray) -> np.ndarray:
    """P(5-point bucket [5j-2, 5j+2]) under the Skellam.

    Integer arithmetic is written as cdf(5j+2) - cdf(5j-3), NOT via
    ``astype(int)`` on the half-integer bounds: ``astype(int)`` truncates toward
    zero, which maps -7.5 to -7 and silently widens every negative bucket, and
    the first version of this helper had an additional off-by-one on the lower
    bound. The two errors summed to ~1.20 of probability mass across buckets
    (a partition must sum to 1) and inflated the wide-bucket log score; caught
    by ``_assert_bucket_partition`` below, added with the fix.
    """
    j = np.floor((k + 2.5) / 5.0)
    hi = (5 * j + 2).astype(int)     # largest integer inside the bucket
    lo = (5 * j - 3).astype(int)     # largest integer below the bucket
    return np.clip(skellam.cdf(hi, mu_h, mu_a) - skellam.cdf(lo, mu_h, mu_a),
                   PROB_FLOOR, None)


def _assert_bucket_partition(mu_h: np.ndarray, mu_a: np.ndarray, m: np.ndarray,
                             sigma: np.ndarray, n: int = 50) -> None:
    """Bucket probabilities must partition the space: their sum must be 1."""
    for i in range(min(n, len(mu_h))):
        js = np.arange(-25, 26)
        kk = 5 * js
        s = float(skellam_bucket_prob_wide(kk, np.repeat(mu_h[i], len(kk)),
                                           np.repeat(mu_a[i], len(kk))).sum())
        g = float(gaussian_bucket_prob_wide(kk, np.repeat(m[i], len(kk)),
                                            np.repeat(sigma[i], len(kk))).sum())
        if abs(s - 1.0) > 1e-6 or abs(g - 1.0) > 1e-6:
            raise AssertionError(
                f"margin-bucket probabilities do not partition "
                f"(skellam {s:.6f}, gaussian {g:.6f} at row {i}); "
                "refusing to score a log loss from a broken partition")


def cover_prob_gaussian(L_eff: np.ndarray, m: np.ndarray,
                        sigma: np.ndarray | float) -> np.ndarray:
    """P(margin > L_eff) = P(margin >= floor(L_eff)+1) under the discretised Gaussian."""
    K = np.floor(L_eff) + 1.0
    return 1.0 - norm.cdf((K - 0.5 - m) / sigma)


def cover_prob_skellam(L_eff: np.ndarray, mu_h: np.ndarray,
                       mu_a: np.ndarray) -> np.ndarray:
    K = np.floor(L_eff).astype(int) + 1
    return skellam.sf(K - 1, mu_h, mu_a)


def over_prob_gaussian(OU: np.ndarray, t_hat: np.ndarray,
                       sigma_total: np.ndarray | float) -> np.ndarray:
    K = np.floor(OU) + 1.0
    return 1.0 - norm.cdf((K - 0.5 - t_hat) / sigma_total)


def over_prob_poisson(OU: np.ndarray, t_hat: np.ndarray) -> np.ndarray:
    from scipy.stats import poisson
    K = np.floor(OU).astype(int) + 1
    return poisson.sf(K - 1, np.clip(t_hat, 1e-6, None))


# --------------------------------------------------------------------------- #
# Bootstraps
# --------------------------------------------------------------------------- #
def _summary(delta: float, draws: np.ndarray) -> dict:
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    se = (hi - lo) / (2 * 1.959964)
    return {"delta": float(delta), "ci_lo": lo, "ci_hi": hi, "se": se,
            "z": float(delta / se) if se > 0 else float("nan"),
            "p_positive": float((draws > 0).mean()), "n_boot": len(draws)}


def boot_loss_delta(l_a: np.ndarray, l_b: np.ndarray, seed: int) -> dict:
    """delta = mean(l_a - l_b); POSITIVE = the second arm loses less."""
    a, b = np.asarray(l_a, float), np.asarray(l_b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    n = len(d)
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        draws[i] = d[rng.integers(0, n, n)].mean()
    return _summary(float(d.mean()), draws)


def boot_rmse_delta(se_a: np.ndarray, se_b: np.ndarray, seed: int) -> dict:
    """delta = RMSE(a) - RMSE(b) on paired per-game squared errors; POSITIVE = b better."""
    a, b = np.asarray(se_a, float), np.asarray(se_b, float)
    delta = float(np.sqrt(a.mean()) - np.sqrt(b.mean()))
    rng = np.random.default_rng(seed)
    n = len(a)
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        j = rng.integers(0, n, n)
        draws[i] = np.sqrt(a[j].mean()) - np.sqrt(b[j].mean())
    return _summary(delta, draws)


def boot_mae_delta(ae_a: np.ndarray, ae_b: np.ndarray, seed: int) -> dict:
    """delta = MAE(a) - MAE(b); POSITIVE = b better."""
    return boot_loss_delta(ae_a, ae_b, seed)


def binary_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, int)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return {"n": int(len(y)), "log_loss": float(ll.mean()),
            "brier": float(((p - y) ** 2).mean())}


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    which = np.digitize(p, edges[1:-1])
    rows = []
    for b in range(bins):
        m = which == b
        if not m.any():
            continue
        rows.append({"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]),
                     "n": int(m.sum()), "mean_pred": float(p[m].mean()),
                     "actual": float(y[m].mean())})
    n = len(y)
    return {"calibration": rows,
            "ece": float(sum(abs(r["mean_pred"] - r["actual"]) * r["n"]
                             for r in rows) / n),
            "calibration_in_the_large": float(p.mean() - y.mean())}


# --------------------------------------------------------------------------- #
# Arms: fit once per fold, predict once per test row
# --------------------------------------------------------------------------- #
def fit_predict(games: pd.DataFrame, reuse_cache: bool = True) -> dict:
    if reuse_cache and CACHE.exists():
        with CACHE.open("rb") as fh:
            cached = pickle.load(fh)
        if cached.get("n") == len(games):
            print(f"[cache] predictions loaded from {CACHE}", flush=True)
            return cached

    data = mt.build_dataset(games)
    assert data.games["GAME_ID"].tolist() == games["GAME_ID"].tolist(), \
        "build_dataset reordered the games frame; row alignment would break"
    targets = pf.build_targets(data.games)
    roll = nd.rolling_team_features(data.games).reset_index(drop=True)
    sched = nd.rest_and_travel(data.games).reset_index(drop=True)

    X_poss = pf.design_poss(roll, sched)
    X_ppp_home = pf.design_ppp(roll, sched, "home")
    X_ppp_away = pf.design_ppp(roll, sched, "away")

    y_margin = data.games["margin"].to_numpy(float)
    y_total = data.games["total"].to_numpy(float)

    splits = nd.walk_forward_split(data.games, N_SPLITS, INITIAL_FRAC)
    out = {
        "n": len(data.games),
        "splits": [(len(tr), len(te),
                    str(data.games["GAME_DATE"].iloc[te[0]].date())) for tr, te in splits],
        "row": np.full(len(data.games), -1, dtype=int),
        "fold": np.full(len(data.games), -1, dtype=int),
    }
    keys = ("m_hat", "t_hat", "sigma", "sigma_total", "pf_margin", "pf_total",
            "pf_poss", "pf_ppp_home", "pf_ppp_away", "poss_hat")
    for k in keys:
        out[k] = np.full(len(data.games), np.nan)

    for k, (tr, te) in enumerate(splits):
        t0 = time.time()
        # Baseline: margin_total run AS-IS (default parameters, unmodified module).
        model = mt.MarginTotalModel().fit(
            data.X_margin.iloc[tr], y_margin[tr],
            data.X_total.iloc[tr], y_total[tr])
        out["m_hat"][te] = model.predict_margin(data.X_margin.iloc[te])
        out["t_hat"][te] = model.predict_total_gaussian(data.X_total.iloc[te])
        out["sigma"][te] = model.sigma_
        out["sigma_total"][te] = model.sigma_total_

        # C1: pace-first, same folds, same ridge selector.
        pfm = pf.PaceFirstModel().fit(
            X_poss.iloc[tr], targets["poss"].to_numpy()[tr],
            X_ppp_home.iloc[tr], targets["ppp_home"].to_numpy()[tr],
            X_ppp_away.iloc[tr], targets["ppp_away"].to_numpy()[tr])
        pred = pfm.predict(X_poss.iloc[te], X_ppp_home.iloc[te],
                           X_ppp_away.iloc[te])
        out["pf_margin"][te] = pred["margin_hat"]
        out["pf_total"][te] = pred["total_hat"]
        out["pf_poss"][te] = pred["poss_hat"]
        out["pf_ppp_home"][te] = pred["ppp_home_hat"]
        out["pf_ppp_away"][te] = pred["ppp_away_hat"]
        out["row"][te] = te
        out["fold"][te] = k + 1
        print(f"[fold {k+1}/{len(splits)}] train {len(tr):,} test {len(te):,} "
              f"from {out['splits'][k][2]}  sigma {model.sigma_:.3f} "
              f"sigma_total {model.sigma_total_:.3f}  "
              f"alphas m {model.alpha_:.2f} t {model.alpha_total_:.2f} "
              f"pf poss {pfm.alpha_poss_:.2f} "
              f"({time.time()-t0:.1f}s)", flush=True)

    # Identical scored rows across arms: every scored row has both arms.
    scored = out["row"] >= 0
    assert np.isfinite(out["m_hat"][scored]).all() and \
        np.isfinite(out["pf_margin"][scored]).all(), "an arm left NaN predictions"
    out["scored_mask"] = scored
    out["games"] = data.games
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    with CACHE.open("wb") as fh:
        pickle.dump(out, fh)
    print(f"[cache] predictions written to {CACHE}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
def measure(out: dict) -> dict:
    games = out["games"]
    mask = out["scored_mask"]
    idx = np.flatnonzero(mask)
    y_margin = games["margin"].to_numpy(float)[idx]
    y_total = games["total"].to_numpy(float)[idx]
    season = games["SEASON"].to_numpy()[idx]

    m_hat = out["m_hat"][idx]
    t_hat = out["t_hat"][idx]
    sigma = out["sigma"][idx]
    sigma_total = out["sigma_total"][idx]
    pf_m = out["pf_margin"][idx]
    pf_t = out["pf_total"][idx]

    R: dict = {"n_games": int(len(games)), "n_scored": int(len(idx)),
               "splits": out["splits"]}

    # ---- C1: margin / total RMSE ---------------------------------------- #
    se_m_base = (m_hat - y_margin) ** 2
    se_m_pf = (pf_m - y_margin) ** 2
    se_t_base = (t_hat - y_total) ** 2
    se_t_pf = (pf_t - y_total) ** 2
    R["C1"] = {}
    for seed in SEEDS:
        primary = seed == PRIMARY_SEED
        R["C1"].setdefault("margin_rmse", {})[str(seed)] = {
            "primary_look": primary,
            "baseline_rmse": float(np.sqrt(se_m_base.mean())),
            "candidate_rmse": float(np.sqrt(se_m_pf.mean())),
            "baseline_mae": float(np.abs(m_hat - y_margin).mean()),
            "candidate_mae": float(np.abs(pf_m - y_margin).mean()),
            **boot_rmse_delta(se_m_base, se_m_pf, seed)}
        R["C1"].setdefault("total_rmse", {})[str(seed)] = {
            "primary_look": primary,
            "baseline_rmse": float(np.sqrt(se_t_base.mean())),
            "candidate_rmse": float(np.sqrt(se_t_pf.mean())),
            "baseline_mae": float(np.abs(t_hat - y_total).mean()),
            "candidate_mae": float(np.abs(pf_t - y_total).mean()),
            **boot_rmse_delta(se_t_base, se_t_pf, seed)}
    # per-split parity with the README's 14.01 / 18.78
    per_split = []
    for k, (tr, te) in enumerate(nd.walk_forward_split(games, N_SPLITS,
                                                       INITIAL_FRAC), 1):
        mm = np.isin(idx, te)
        per_split.append({
            "split": k, "n": int(mm.sum()),
            "margin_rmse_baseline": float(np.sqrt(se_m_base[mm].mean())),
            "margin_rmse_pace_first": float(np.sqrt(se_m_pf[mm].mean())),
            "total_rmse_baseline": float(np.sqrt(se_t_base[mm].mean())),
            "total_rmse_pace_first": float(np.sqrt(se_t_pf[mm].mean())),
        })
    R["C1"]["per_split"] = per_split
    R["C1"]["margin_rmse_baseline_split_mean"] = float(np.mean(
        [p["margin_rmse_baseline"] for p in per_split]))
    R["C1"]["total_rmse_baseline_split_mean"] = float(np.mean(
        [p["total_rmse_baseline"] for p in per_split]))

    # ---- C2 (a): margin log score --------------------------------------- #
    k_int = np.round(y_margin).astype(int)   # margins are integers; round guards float repr
    ll_gauss = gaussian_bucket_logprob_exact(k_int, m_hat, sigma)
    ll_sk = skellam.logpmf(k_int, np.clip((t_hat + m_hat) / 2, 0.01, None),
                           np.clip((t_hat - m_hat) / 2, 0.01, None))
    loss_g, loss_s = -ll_gauss, -ll_sk
    R["C2_margin_logscore"] = {}
    for seed in SEEDS:
        R["C2_margin_logscore"][str(seed)] = {
            "primary_look": seed == PRIMARY_SEED,
            "baseline_log_loss": float(loss_g.mean()),
            "skellam_log_loss": float(loss_s.mean()),
            **boot_loss_delta(loss_g, loss_s, seed)}
    # robustness: 5-point buckets (no decision weight)
    mu_h = np.clip((t_hat + m_hat) / 2, 0.01, None)
    mu_a = np.clip((t_hat - m_hat) / 2, 0.01, None)
    _assert_bucket_partition(mu_h, mu_a, m_hat, sigma)
    wide_g = -np.log(gaussian_bucket_prob_wide(k_int, m_hat, sigma))
    wide_s = -np.log(skellam_bucket_prob_wide(k_int, mu_h, mu_a))
    R["C2_margin_logscore"]["wide5_secondary"] = {
        "baseline_log_loss": float(wide_g.mean()),
        "skellam_log_loss": float(wide_s.mean()),
        **boot_loss_delta(wide_g, wide_s, PRIMARY_SEED)}

    # distribution calibration on 5-point buckets (descriptive)
    j = np.floor((k_int + 2.5) / 5.0).astype(int)
    cal = []
    for jj in np.unique(j):
        m_row = j == jj
        # predicted mass of THIS bucket for EVERY row, then averaged:
        bounds_lo, bounds_hi = 5 * jj - 2.5, 5 * jj + 2.5
        pg_all = np.clip(norm.cdf((bounds_hi - m_hat) / sigma)
                         - norm.cdf((bounds_lo - m_hat) / sigma), 0, 1)
        ps_all = np.clip(skellam.cdf(5 * jj + 2, mu_h, mu_a)
                         - skellam.cdf(5 * jj - 3, mu_h, mu_a), 0, 1)
        cal.append({"bucket": int(jj), "range": [int(5*jj-2), int(5*jj+2)],
                    "n": int(m_row.sum()), "observed_freq": float(m_row.mean()),
                    "gauss_mean_mass": float(pg_all.mean()),
                    "skellam_mean_mass": float(ps_all.mean())})
    R["C2_bucket_calibration"] = cal

    # ---- market rows ---------------------------------------------------- #
    joined = cl.join_odds(games, cl.load_odds())
    row_of = {gid: i for i, gid in enumerate(games["GAME_ID"].to_numpy())}
    joined["row"] = joined["GAME_ID"].map(row_of)
    joined = joined[joined["row"].isin(set(idx.tolist()))].copy()
    joined["row"] = joined["row"].astype(int)
    joined = effective_line(joined)
    joined["margin"] = games["margin"].to_numpy()[joined["row"].to_numpy()]
    joined["total"] = games["total"].to_numpy()[joined["row"].to_numpy()]
    joined["season_g"] = games["SEASON"].to_numpy()[joined["row"].to_numpy()]
    # arm predictions, aligned to joined rows via the scored-row position map
    pos = {int(r): i for i, r in enumerate(idx)}
    at = np.array([pos[int(r)] for r in joined["row"].to_numpy()])
    for name, arr in (("m_hat", m_hat), ("t_hat", t_hat), ("sigma", sigma),
                      ("sigma_total", sigma_total), ("mu_h", mu_h),
                      ("mu_a", mu_a), ("pf_margin", pf_m), ("pf_total", pf_t)):
        joined[name] = arr[at]
    R["market_rows"] = {
        "joined_scored": int(len(joined)),
        "signed_era_rows": int(joined["era_signed"].sum()),
        "unsigned_era_rows": int((~joined["era_signed"]).sum()),
    }

    sp = pd.to_numeric(joined["Spread"], errors="coerce").to_numpy(float)
    ou = pd.to_numeric(joined["OU"], errors="coerce").to_numpy(float)
    good_sp = np.abs(sp) <= SPREAD_MAX_ABS
    good_ou = (ou >= OU_BOUNDS[0]) & (ou <= OU_BOUNDS[1])
    R["market_rows"]["spread_rows_sane"] = int(good_sp.sum())
    R["market_rows"]["spread_rows_excluded_out_of_bounds"] = int((~good_sp).sum())
    R["market_rows"]["total_rows_sane"] = int(good_ou.sum())
    R["market_rows"]["total_rows_excluded_out_of_bounds"] = int((~good_ou).sum())

    # ---- C2 (b): cover probabilities ------------------------------------ #
    L = joined["L_eff"].to_numpy(float)
    y_cover_all = joined["margin"].to_numpy(float) - L
    push = y_cover_all == 0
    keep = good_sp & ~push
    R["market_rows"]["cover_pushes_excluded"] = int((good_sp & push).sum())
    y_cover = (y_cover_all[keep] > 0).astype(int)
    p_cov_g = cover_prob_gaussian(L[keep], joined["m_hat"].to_numpy()[keep],
                                  joined["sigma"].to_numpy()[keep])
    p_cov_s = cover_prob_skellam(L[keep], joined["mu_h"].to_numpy()[keep],
                                 joined["mu_a"].to_numpy()[keep])
    R["C2_cover"] = {}
    for seed in SEEDS:
        R["C2_cover"][str(seed)] = {
            "primary_look": seed == PRIMARY_SEED,
            "baseline": binary_metrics(y_cover, p_cov_g),
            "skellam": binary_metrics(y_cover, p_cov_s),
            **boot_loss_delta(
                -(y_cover * np.log(np.clip(p_cov_g, 1e-6, 1-1e-6))
                  + (1 - y_cover) * np.log(np.clip(1 - p_cov_g, 1e-6, 1-1e-6))),
                -(y_cover * np.log(np.clip(p_cov_s, 1e-6, 1-1e-6))
                  + (1 - y_cover) * np.log(np.clip(1 - p_cov_s, 1e-6, 1-1e-6))),
                seed)}
    R["C2_cover"]["calibration_gaussian"] = calibration_table(y_cover, p_cov_g)
    R["C2_cover"]["calibration_skellam"] = calibration_table(y_cover, p_cov_s)
    # sensitivity: continuous-Phi cover probability (descriptive)
    p_cov_g_cont = 1.0 - norm.cdf((L[keep] - joined["m_hat"].to_numpy()[keep])
                                  / joined["sigma"].to_numpy()[keep])
    R["C2_cover"]["sensitivity_continuous_phi"] = binary_metrics(y_cover,
                                                                 p_cov_g_cont)

    # ---- total-over (secondary) ----------------------------------------- #
    y_over_all = joined["total"].to_numpy(float) - ou
    push_o = y_over_all == 0
    keep_o = good_ou & ~push_o
    R["market_rows"]["over_pushes_excluded"] = int((good_ou & push_o).sum())
    y_over = (y_over_all[keep_o] > 0).astype(int)
    p_over_g = over_prob_gaussian(ou[keep_o], joined["t_hat"].to_numpy()[keep_o],
                                  joined["sigma_total"].to_numpy()[keep_o])
    p_over_p = over_prob_poisson(ou[keep_o], joined["t_hat"].to_numpy()[keep_o])
    R["total_over"] = {
        "gaussian": binary_metrics(y_over, p_over_g),
        "poisson_t_hat_secondary": binary_metrics(y_over, p_over_p),
        "calibration_gaussian": calibration_table(y_over, p_over_g),
        "delta_vs_poisson_secondary": boot_loss_delta(
            -(y_over * np.log(np.clip(p_over_p, 1e-6, 1-1e-6))
              + (1 - y_over) * np.log(np.clip(1 - p_over_p, 1e-6, 1-1e-6))),
            -(y_over * np.log(np.clip(p_over_g, 1e-6, 1-1e-6))
              + (1 - y_over) * np.log(np.clip(1 - p_over_g, 1e-6, 1-1e-6))),
            PRIMARY_SEED)}

    # ---- issue #44: line as a forecaster vs the model -------------------- #
    line_ae = np.abs(L[keep] - joined["margin"].to_numpy()[keep])
    for label, pred in (("margin_total", joined["m_hat"].to_numpy()[keep]),
                        ("pace_first", joined["pf_margin"].to_numpy()[keep])):
        ae = np.abs(pred - joined["margin"].to_numpy()[keep])
        R.setdefault("spread_forecast", {})[label] = {
            "model_mae": float(ae.mean()), "line_mae": float(line_ae.mean()),
            "model_rmse": float(np.sqrt(((pred - joined["margin"].to_numpy()[keep])**2).mean())),
            "line_rmse": float(np.sqrt(((L[keep] - joined["margin"].to_numpy()[keep])**2).mean())),
            "delta_mae_line_minus_model": boot_mae_delta(line_ae, ae, PRIMARY_SEED)}
    line_ae_t = np.abs(ou[keep_o] - joined["total"].to_numpy()[keep_o])
    ae_t = np.abs(joined["t_hat"].to_numpy()[keep_o]
                  - joined["total"].to_numpy()[keep_o])
    ae_t_pf = np.abs(joined["pf_total"].to_numpy()[keep_o]
                     - joined["total"].to_numpy()[keep_o])
    R["total_forecast"] = {
        "line_mae": float(line_ae_t.mean()), "line_rmse": float(np.sqrt(
            ((ou[keep_o] - joined["total"].to_numpy()[keep_o])**2).mean())),
        "margin_total_mae": float(ae_t.mean()),
        "margin_total_rmse": float(np.sqrt(((joined["t_hat"].to_numpy()[keep_o]
                                             - joined["total"].to_numpy()[keep_o])**2).mean())),
        "pace_first_mae": float(ae_t_pf.mean()),
        "pace_first_rmse": float(np.sqrt(((joined["pf_total"].to_numpy()[keep_o]
                                           - joined["total"].to_numpy()[keep_o])**2).mean())),
        "delta_mae_line_minus_margin_total": boot_mae_delta(
            line_ae_t, ae_t, PRIMARY_SEED)}

    # ---- ROI at the closing price (reuse roi(); threshold 0.5, -110) ----- #
    def _roi(p_side: np.ndarray, y_side: np.ndarray) -> dict:
        p = np.clip(p_side, 1e-6, 1 - 1e-6)
        p_pick = np.maximum(p, 1 - p)
        y_pick = np.where(p >= 0.5, y_side, 1 - y_side)
        r = roi(p_pick, P_PRICE, y_pick, threshold=0.5)
        return {k: v for k, v in r.items()}

    R["roi"] = {
        "price": P_PRICE,
        "spread_gaussian": _roi(p_cov_g, y_cover),
        "spread_skellam_secondary": _roi(p_cov_s, y_cover),
        "total_gaussian": _roi(p_over_g, y_over),
    }

    # ---- per-season breakdown of the primary endpoints ------------------- #
    seas = []
    for s in sorted(set(season)):
        mrow = season == s
        entry = {"season": s, "n": int(mrow.sum())}
        entry["margin_rmse_baseline"] = float(np.sqrt(se_m_base[mrow].mean()))
        entry["margin_rmse_pace_first"] = float(np.sqrt(se_m_pf[mrow].mean()))
        entry["total_rmse_baseline"] = float(np.sqrt(se_t_base[mrow].mean()))
        entry["total_rmse_pace_first"] = float(np.sqrt(se_t_pf[mrow].mean()))
        entry["margin_logloss_gaussian"] = float(loss_g[mrow].mean())
        entry["margin_logloss_skellam"] = float(loss_s[mrow].mean())
        sm = joined["season_g"].to_numpy()[keep] == s
        if sm.any():
            entry["cover_logloss_gaussian"] = float(binary_metrics(
                y_cover[sm], p_cov_g[sm])["log_loss"])
            entry["cover_logloss_skellam"] = float(binary_metrics(
                y_cover[sm], p_cov_s[sm])["log_loss"])
            entry["n_cover"] = int(sm.sum())
        seas.append(entry)
    R["per_season"] = seas

    # ---- team-point sanity for the decomposition (descriptive) ---------- #
    R["pace_first_diagnostics"] = {
        "poss_hat_mean": float(out["pf_poss"][idx].mean()),
        "poss_true_mean": float(pf.build_targets(games)["poss"].to_numpy()[idx].mean()),
        "ppp_home_hat_mean": float(out["pf_ppp_home"][idx].mean()),
        "ppp_away_hat_mean": float(out["pf_ppp_away"][idx].mean()),
        "share_mu_away_negative": float((out["pf_total"][idx] - out["pf_margin"][idx] < 0).mean()),
    }
    return R


# --------------------------------------------------------------------------- #
# Verdict + logging
# --------------------------------------------------------------------------- #
def verdicts(R: dict) -> dict:
    def _rule(block: dict, minimum: float | None = None) -> str:
        d, z = block["delta"], abs(block["z"])
        lo, hi = block["ci_lo"], block["ci_hi"]
        ok_min = (d >= minimum) if minimum is not None else (d > 0)
        if ok_min and lo > 0 and z > Z_BAR:
            return "clears the bar"
        if lo > 0 and z <= Z_BAR:
            return "sub-bar lead (CI excludes zero, |z| <= 3)"
        if hi < 0:
            return "decisive loss"
        return "null (CI straddles zero)"

    c1_m = R["C1"]["margin_rmse"][str(PRIMARY_SEED)]
    c1_t = R["C1"]["total_rmse"][str(PRIMARY_SEED)]
    c2_a = R["C2_margin_logscore"][str(PRIMARY_SEED)]
    c2_b = R["C2_cover"][str(PRIMARY_SEED)]
    v = {
        "rule": "C1 succeeds iff BOTH margin-RMSE and total-RMSE deltas clear "
                "delta>0, CI excluding zero, |z| > 3 at seed 42 with sign "
                "stable at 137/2718; C2 succeeds iff the margin log score "
                "gains >= 0.002 nats (same bars) OR cover log loss improves "
                "(same bars). |z| > 3 governs the 6-look family (Bonferroni "
                f"{BONFERRONI:.4f}).",
        "C1_margin_rmse": _rule(c1_m),
        "C1_total_rmse": _rule(c1_t),
        "C2_margin_logscore": _rule(c2_a, MIN_EFFECT_NATS),
        "C2_cover": _rule(c2_b),
    }
    c1_ok = (v["C1_margin_rmse"] == "clears the bar"
             and v["C1_total_rmse"] == "clears the bar")
    # sign stability at replications
    c1_stable = all(R["C1"]["margin_rmse"][s]["delta"] > 0
                    and R["C1"]["total_rmse"][s]["delta"] > 0 for s in map(str, SEEDS))
    c2_stable = all(R["C2_margin_logscore"][s]["delta"] > 0 for s in map(str, SEEDS)) \
        or all(R["C2_cover"][s]["delta"] > 0 for s in map(str, SEEDS))
    v["C1_verdict"] = ("pace-first decomposition BEATS the monolithic model"
                       if c1_ok and c1_stable else
                       "pace-first decomposition does NOT beat the monolithic model"
                       + ("" if c1_stable else " (sign unstable across seeds)"))
    c2_ok = (v["C2_margin_logscore"] == "clears the bar"
             or v["C2_cover"] == "clears the bar")
    v["C2_verdict"] = ("a proper margin distribution EARNs its place"
                       if c2_ok and c2_stable else
                       "a proper margin distribution does NOT earn its place"
                       + ("" if c2_stable else " (sign unstable across seeds)"))
    v["C1_details"] = {"margin": v["C1_margin_rmse"], "total": v["C1_total_rmse"]}
    v["C2_details"] = {"margin_logscore": v["C2_margin_logscore"],
                       "cover": v["C2_cover"]}
    return v


def log_rows(R: dict, V: dict, inv: dict, correction_note: str | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    rows = []

    def add(name, experiment_type, payload):
        rows.append({"name": name, "status": "executed", "declared_utc": now,
                     "experiment_type": experiment_type,
                     "protocol_ref": PREREG_NAME, **payload})

    for seed in SEEDS:
        prim = seed == PRIMARY_SEED
        m = R["C1"]["margin_rmse"][str(seed)]
        add(f"C44-run-C1-margin-s{seed}", "pace-first vs monolithic, margin RMSE",
            {"hypothesis": "C1", "primary_look": prim, "seed": seed,
             "arms": "pace_first vs margin_total",
             "sign_convention": "delta = RMSE(baseline) - RMSE(candidate); "
                                "POSITIVE = the candidate is better",
             "n_boot": N_BOOT, "n_scored": R["n_scored"], **m})
        t = R["C1"]["total_rmse"][str(seed)]
        add(f"C44-run-C1-total-s{seed}", "pace-first vs monolithic, total RMSE",
            {"hypothesis": "C1", "primary_look": prim, "seed": seed,
             "arms": "pace_first vs margin_total",
             "sign_convention": "delta = RMSE(baseline) - RMSE(candidate); "
                                "POSITIVE = the candidate is better",
             "n_boot": N_BOOT, "n_scored": R["n_scored"], **t})
        a = R["C2_margin_logscore"][str(seed)]
        add(f"C44-run-C2-logscore-s{seed}", "Skellam vs Gaussian, margin log score",
            {"hypothesis": "C2", "primary_look": prim, "seed": seed,
             "arms": "skellam vs Phi(mhat/sigma) gaussian",
             "sign_convention": "delta = logloss(gaussian) - logloss(skellam); "
                                "POSITIVE = the Skellam is better",
             "n_boot": N_BOOT, "n_scored": R["n_scored"], **a})
        b = R["C2_cover"][str(seed)]
        add(f"C44-run-C2-cover-s{seed}", "Skellam vs Gaussian, cover-probability "
            "log loss against the closing spread",
            {"hypothesis": "C2", "primary_look": prim, "seed": seed,
             "arms": "skellam vs Phi(mhat/sigma) gaussian",
             "sign_convention": "delta = logloss(gaussian) - logloss(skellam); "
                                "POSITIVE = the Skellam is better",
             "n_boot": N_BOOT, "n_scored": b["baseline"]["n"],
             "baseline": b["baseline"], "skellam": b["skellam"],
             **{k: b[k] for k in ("delta", "ci_lo", "ci_hi", "se", "z",
                                  "p_positive")}})
    add("C44-run-market-spread-roi", "spread-market ROI at the closing price "
        "(parameter-free rule)", {"hypothesis": "market", "primary_look": True,
        "arms": "Phi(mhat/sigma) cover probabilities at a flat -110",
        "roi": R["roi"]["spread_gaussian"], "z_bar": Z_BAR})
    add("C44-run-market-total-roi", "total-market ROI at the closing price "
        "(parameter-free rule)", {"hypothesis": "market", "primary_look": True,
        "arms": "Gaussian total at a flat -110",
        "roi": R["roi"]["total_gaussian"], "z_bar": Z_BAR})
    rows.append({"name": "C44-verdict", "status": "executed", "declared_utc": now,
                 "experiment_type": "verdict_record", "protocol_ref": PREREG_NAME,
                 **V, "inventory_rows": R["market_rows"]})
    if correction_note:
        rows.append({"name": "C44-correction-note", "status": "executed",
                     "declared_utc": now,
                     "experiment_type": "correction record (append-only; the "
                                       "earlier rows stand as written)",
                     "protocol_ref": PREREG_NAME,
                     "note": correction_note,
                     "corrected_wide5": R["C2_margin_logscore"]["wide5_secondary"]})
    # Append-only, but never duplicate a record that is already on the log: a
    # re-run after a bug fix must not silently double the look count.
    existing = set()
    for line in JSONL.open():
        try:
            existing.add(json.loads(line).get("name"))
        except json.JSONDecodeError:
            continue
    fresh = [r for r in rows if r["name"] not in existing]
    skipped = [r["name"] for r in rows if r["name"] in existing]
    with JSONL.open("a") as fh:
        for r in fresh:
            fh.write(json.dumps(r, default=float) + "\n")
    print(f"appended {len(fresh)} measurement rows to {JSONL}"
          + (f" (skipped already-logged: {skipped})" if skipped else ""),
          flush=True)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def format_report(R: dict, V: dict, inv: dict,
                  correction_note: str | None = None) -> str:
    L: list[str] = []
    a = L.append
    a("# Batch C / issue #44 - spread and total markets (results/c44_spread_total.json)")
    a("")
    a(f"Pre-registration: results/experiments.jsonl :: {PREREG_NAME} "
      "(written before any arm ran). "
      f"{N_LOOKS} looks, Bonferroni {BONFERRONI:.4f} one-sided; the |z| > 3 bar governs.")
    if correction_note:
        a(f"> Correction on record: {correction_note}")
    a("")
    a("## Inventory")
    a(f"- Odds DB: {inv['odds_rows_reconciled']:,} reconciled rows "
      f"({inv['odds_rows_expected']:,} counted across every table spelling, "
      f"{inv['odds_rows_collapsed']:,} collapsed as duplicate spellings) over "
      "2015-16..2025-26, every row carrying Spread and OU. "
      "**Spread and total lines exist -> C2 runs at full scope.**")
    a(f"- Per-season rows / spread / OU coverage: " + "; ".join(
        f"{s} {v['rows']:,}/{v['spread_non_null']:,}/{v['ou_non_null']:,}"
        f" (frac(spread>0) {v['frac_spread_positive']:.2f})"
        for s, v in inv["per_season_tables"].items()))
    a(f"- Spread sign convention: 2022-23..2025-26 signed expected home margin; "
      "2015-16..2021-22 unsigned favourite handicap resolved by the de-vigged "
      "moneyline favourite (L_eff; inventory cover rates 47-52% after "
      "resolution vs 39-43% naive).")
    a(f"- Quarter-level scores: {inv['quarter_level_scores']}")
    a(f"- Juice: {inv['juice']}")
    a(f"- Market rows scored: {R['market_rows']}")
    a("")
    a("## C1 - pace-first decomposition vs the monolithic margin/total model")
    a("delta = RMSE(baseline) - RMSE(candidate); POSITIVE = pace-first better.")
    for label, key in (("margin RMSE", "margin_rmse"), ("total RMSE", "total_rmse")):
        a(f"### {label}")
        a("| seed | look | baseline | pace-first | delta | 95% CI | SE | \\|z\\| |")
        a("|---|---|---|---|---|---|---|---|")
        for seed in SEEDS:
            b = R["C1"][key][str(seed)]
            a(f"| {seed} | {'primary' if b['primary_look'] else 'replication'} "
              f"| {b['baseline_rmse']:.4f} | {b['candidate_rmse']:.4f} "
              f"| {b['delta']:+.4f} | [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] "
              f"| {b['se']:.4f} | {abs(b['z']):.2f} |")
        b0 = R["C1"][key][str(PRIMARY_SEED)]
        a(f"MAE at the primary look: baseline {b0['baseline_mae']:.4f} vs "
          f"pace-first {b0['candidate_mae']:.4f}.")
    a("")
    a("Per-split parity with the module's own report (README: margin 14.01, "
      "total 18.78):")
    a(f"- baseline split-mean margin RMSE {R['C1']['margin_rmse_baseline_split_mean']:.3f}, "
      f"total RMSE {R['C1']['total_rmse_baseline_split_mean']:.3f}")
    a("| split | n | margin base | margin pf | total base | total pf |")
    a("|---|---|---|---|---|---|")
    for p in R["C1"]["per_split"]:
        a(f"| {p['split']} | {p['n']:,} | {p['margin_rmse_baseline']:.3f} "
          f"| {p['margin_rmse_pace_first']:.3f} | {p['total_rmse_baseline']:.3f} "
          f"| {p['total_rmse_pace_first']:.3f} |")
    a("")
    a(f"Decomposition diagnostics (descriptive): {R['pace_first_diagnostics']}")
    a("")
    a("## C2 - Skellam margin distribution vs Phi(mhat/sigma)")
    a("delta = logloss(gaussian) - logloss(skellam); POSITIVE = Skellam better.")
    a("### (a) margin-bucket log score (1-point buckets, nats)")
    a("| seed | look | gaussian | skellam | delta | 95% CI | SE | \\|z\\| |")
    a("|---|---|---|---|---|---|---|---|")
    for seed in SEEDS:
        b = R["C2_margin_logscore"][str(seed)]
        a(f"| {seed} | {'primary' if b['primary_look'] else 'replication'} "
          f"| {b['baseline_log_loss']:.5f} | {b['skellam_log_loss']:.5f} "
          f"| {b['delta']:+.5f} | [{b['ci_lo']:+.5f}, {b['ci_hi']:+.5f}] "
          f"| {b['se']:.5f} | {abs(b['z']):.2f} |")
    w = R["C2_margin_logscore"]["wide5_secondary"]
    a(f"5-point buckets (robustness, no decision weight): gaussian "
      f"{w['baseline_log_loss']:.5f} vs skellam {w['skellam_log_loss']:.5f}, "
      f"delta {w['delta']:+.5f} [{w['ci_lo']:+.5f}, {w['ci_hi']:+.5f}], "
      f"|z| {abs(w['z']):.2f}.")
    a("")
    a("### (b) cover probabilities vs the closing spread "
      f"({R['C2_cover'][str(PRIMARY_SEED)]['baseline']['n']:,} rows, pushes excluded)")
    a("| seed | look | logloss gaussian | logloss skellam | delta | 95% CI | SE | \\|z\\| |")
    a("|---|---|---|---|---|---|---|---|")
    for seed in SEEDS:
        b = R["C2_cover"][str(seed)]
        a(f"| {seed} | {'primary' if b['primary_look'] else 'replication'} "
          f"| {b['baseline']['log_loss']:.5f} | {b['skellam']['log_loss']:.5f} "
          f"| {b['delta']:+.5f} | [{b['ci_lo']:+.5f}, {b['ci_hi']:+.5f}] "
          f"| {b['se']:.5f} | {abs(b['z']):.2f} |")
    for label, key in (("gaussian", "calibration_gaussian"),
                       ("skellam", "calibration_skellam")):
        c = R["C2_cover"][key]
        a(f"Calibration, {label}: ECE {c['ece']:.4f}, "
          f"calibration-in-the-large {c['calibration_in_the_large']:+.4f}")
        for r in c["calibration"]:
            a(f"  [{r['lo']:.1f},{r['hi']:.1f}) n={r['n']:,} pred "
              f"{r['mean_pred']:.3f} actual {r['actual']:.3f}")
    a(f"Sensitivity (continuous Phi, descriptive): "
      f"{R['C2_cover']['sensitivity_continuous_phi']}")
    a("")
    a("### margin-bucket distribution calibration (mean predicted mass vs observed)")
    a("| bucket | n | observed | gauss mass | skellam mass |")
    a("|---|---|---|---|---|")
    for r in R["C2_bucket_calibration"]:
        a(f"| {r['range']} | {r['n']:,} | {r['observed_freq']:.4f} "
          f"| {r['gauss_mean_mass']:.4f} | {r['skellam_mean_mass']:.4f} |")
    a("")
    a("### total-over vs the closing total (secondary, no decision weight)")
    tg, tp = R["total_over"]["gaussian"], R["total_over"]["poisson_t_hat_secondary"]
    d = R["total_over"]["delta_vs_poisson_secondary"]
    a(f"over log loss: gaussian total {tg['log_loss']:.5f} (Brier {tg['brier']:.5f}); "
      f"Poisson(t_hat) {tp['log_loss']:.5f} (Brier {tp['brier']:.5f}); "
      f"delta(poisson - gaussian loss) {d['delta']:+.5f} "
      f"[{d['ci_lo']:+.5f}, {d['ci_hi']:+.5f}] (POSITIVE = gaussian better).")
    c = R["total_over"]["calibration_gaussian"]
    a(f"Gaussian-total over-calibration: ECE {c['ece']:.4f}, "
      f"in-the-large {c['calibration_in_the_large']:+.4f}")
    for r in c["calibration"]:
        a(f"  [{r['lo']:.1f},{r['hi']:.1f}) n={r['n']:,} pred "
          f"{r['mean_pred']:.3f} actual {r['actual']:.3f}")
    a("")
    a("## Issue #44 market tests (the closing line as a forecaster, and ROI)")
    a("MAE / RMSE against ACTUAL margins and totals, model vs closing line, "
      "same rows (delta = MAE(line) - MAE(model); POSITIVE = model better):")
    for label, r in R["spread_forecast"].items():
        d = r["delta_mae_line_minus_model"]
        a(f"- spread/margin [{label}]: model MAE {r['model_mae']:.3f} RMSE "
          f"{r['model_rmse']:.3f} vs line MAE {r['line_mae']:.3f} RMSE "
          f"{r['line_rmse']:.3f}; delta {d['delta']:+.4f} "
          f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}], |z| {abs(d['z']):.2f}")
    d = R["total_forecast"]["delta_mae_line_minus_margin_total"]
    a(f"- totals [margin_total]: model MAE {R['total_forecast']['margin_total_mae']:.3f} "
      f"RMSE {R['total_forecast']['margin_total_rmse']:.3f} vs line MAE "
      f"{R['total_forecast']['line_mae']:.3f} RMSE {R['total_forecast']['line_rmse']:.3f}; "
      f"delta {d['delta']:+.4f} [{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}], "
      f"|z| {abs(d['z']):.2f}")
    a(f"- totals [pace_first]: model MAE {R['total_forecast']['pace_first_mae']:.3f} "
      f"RMSE {R['total_forecast']['pace_first_rmse']:.3f}")
    a("")
    a(f"ROI, parameter-free rule at a flat -110 (p_raw={P_PRICE:.5f}), both sides "
      "considered, closing_line.roi() reused (threshold 0.5):")
    for label, r in R["roi"].items():
        if label == "price":
            continue
        if r.get("bets"):
            a(f"- {label}: {r['bets']:,} bets, ROI {r['roi']*100:+.2f}% "
              f"SE {r['se']*100:.2f}%, |z| {abs(r.get('z', float('nan'))):.2f}"
              + ("  <- PASTES THE BAR" if abs(r.get("z", 0)) >= Z_BAR
                 else "  (below the |z|>3 bar)"))
        else:
            a(f"- {label}: no bets placed")
    a("")
    a("## Per-season (primary endpoints)")
    a("| season | n | margin base | margin pf | total base | total pf "
      "| logloss G | logloss Sk | cover n | cover G | cover Sk |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for e in R["per_season"]:
        a(f"| {e['season']} | {e['n']:,} | {e['margin_rmse_baseline']:.3f} "
          f"| {e['margin_rmse_pace_first']:.3f} | {e['total_rmse_baseline']:.3f} "
          f"| {e['total_rmse_pace_first']:.3f} "
          f"| {e['margin_logloss_gaussian']:.4f} | {e['margin_logloss_skellam']:.4f} "
          f"| {e.get('n_cover', 0):,} | {e.get('cover_logloss_gaussian', float('nan')):.4f} "
          f"| {e.get('cover_logloss_skellam', float('nan')):.4f} |")
    a("")
    a("## Verdicts")
    a(f"- C1: **{V['C1_verdict']}** (margin: {V['C1_margin_rmse']}; "
      f"total: {V['C1_total_rmse']})")
    a(f"- C2: **{V['C2_verdict']}** (margin log score: {V['C2_margin_logscore']}; "
      f"cover: {V['C2_cover']})")
    a("")
    a("Reproduce:")
    a("```")
    a(".venv/bin/python scripts/c44_spread_total.py --prereg   # declaration (already logged)")
    a(".venv/bin/python scripts/c44_spread_total.py            # full measurement")
    a("```")
    a("Nothing in src/Models/margin_total.py, scripts/closing_line.py or the "
      "shared data layer was modified; every fitted quantity is train-only; "
      "no threshold sweep, no variants (selection cost: zero).")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prereg", action="store_true",
                    help="append the pre-registration to "
                         "results/experiments.jsonl and exit")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--correction-note", default=None,
                    help="append a correction record to results/experiments.jsonl "
                         "explaining what changed and why")
    args = ap.parse_args()

    if args.prereg:
        rec = prereg_entry()
        with JSONL.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"appended pre-registration {rec['name']!r} to {JSONL}")
        return 0

    declared = any(PREREG_NAME in line for line in JSONL.open())
    if not declared:
        raise SystemExit(f"results/experiments.jsonl has no {PREREG_NAME} "
                         "pre-registration; run with --prereg first")

    t0 = time.time()
    inv = inventory()
    games = nd.build_games()
    out = fit_predict(games, reuse_cache=not args.no_cache)
    R = measure(out)
    R["inventory"] = inv
    V = verdicts(R)

    report = format_report(R, V, inv, correction_note=args.correction_note)
    OUT_MD.write_text(report + "\n")
    OUT_JSON.write_text(json.dumps({"prereg": PREREG_NAME, "verdicts": V,
                                    "results": R}, indent=2, default=float) + "\n")
    log_rows(R, V, inv, correction_note=args.correction_note)
    print(report)
    print(f"\nwrote {OUT_MD} and {OUT_JSON}   ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
