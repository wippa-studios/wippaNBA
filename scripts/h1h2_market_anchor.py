"""Batch A gate-keepers: does anything we measure beat the closing price?

H1 (market-anchored blend) and H2 (availability beyond the price). These are the
two questions that scope everything else in the project, because a feature that
does not improve on the closing line is a feature the market already prices.

THE ODDS-AS-INPUT DISTINCTION, ON THE RECORD
--------------------------------------------
The project's "odds are never a feature" rule governs the *market-evaluation*
models - the ones whose purpose is to detect mispricing in the odds themselves
(scripts/edge_betting.py and friends). This script is the OTHER kind of
experiment: it predicts GAME OUTCOMES, and the closing line is a pre-tip-off
quantity, so using it as an INPUT is legitimate and is exactly what a bettor
anchoring on the price does. H1 and H2 ask what our basketball information adds
to the price, so the price must be in the model. The two roles are distinct and
this file keeps them distinct.

THE ARMS (pre-registered, identical scored rows)
-----------------------------------------------
  P       price only: the de-vigged closing probability (proportional de-vig).
  M       the canonical model's out-of-fold home-win probability
          (load_design(with_availability=False)).
  M+A     the same with the availability block (with_availability=True), the
          project's one confirmed non-null mechanism.
  P+M     logistic stack of [price logit, M logit].
  P+M+A   logistic stack of [price logit, M+A logit].

The stack weights are fit STRICTLY WALK-FORWARD BY SEASON BLOCK: for each scored
season, on every overlap row of the seasons before it, never on the scored
season itself (the H3 lesson: a calibrator fit on the tail it scores is
leakage). Train/test disjointness is asserted in code, on both season labels
and dates. Scaler and estimator are fit on train blocks only.

THE HYPOTHESES (decision rules fixed before any run)
---------------------------------------------------
  H1a  P+M beats P by >= 0.002 nats, 95% CI excluding zero, |z| > 3.
       "Does basketball information add anything the close does not contain?"
  H1b  P+M beats M alone (descriptive - shows the blend machinery works).
  H2   P+M+A beats P+M by >= 0.002 nats, CI excluding zero, |z| > 3.
       "Is the availability signal already priced by the close?" (Dare et al.
       2015 found absence fully priced at the close on 1995-2012 data; the NBA
       changed its injury-report rules in Dec 2025, so this is genuinely open.)

Three primary looks -> Bonferroni 0.05/3 = 0.0167 one-sided (|z| ~ 2.13). The
project's |z| > 3 bar (Vandenbruaene et al. 2022) is stricter and governs.

THE DE-VIG, FIXED EX ANTE
-------------------------
PROPORTIONAL (p / (p_home + p_away)) - the same quantity closing_line.join_odds
stores as `p_market`, so every number here is comparable with
results/closing_line.json. Kelly_Criterion.devig_two_way offers Shin's
informed-trader de-vig as well; it is NOT used and NOT run here, because
choosing the de-vig after seeing which one flatters the model is selection.
Shin moves mass toward favourites relative to proportional; if anything the
proportional choice leaves the favourite-longshot skew in the price, which
makes the P arm's job harder, not easier.

Run:  python scripts/h1h2_market_anchor.py --prereg   # declaration only
      python scripts/h1h2_market_anchor.py            # the measurement
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from DataProviders import nba_data as nd          # noqa: E402
import closing_line as cl                         # noqa: E402  (read-only)

JSONL = ROOT / "results" / "experiments.jsonl"
CACHE_DIR = Path("/tmp/opencode/h1h2")

SEED = 42                      # primary bootstrap seed
REPLICATION_SEEDS = (137, 2718)
N_BOOT = 2000
MIN_EFFECT = 0.002             # nats, pre-registered minimum effect of interest
Z_BAR = 3.0                    # Vandenbruaene et al. 2022
BONFERRONI_ALPHA = 0.05 / 3    # three primary looks, one-sided
LOGIT_CLIP = 1e-6
HALFLIFE = cl.HALFLIFE         # 10.0 - the canonical model's config, consumed
MIN_GAMES = cl.MIN_GAMES       # 5     - not re-tuned here
N_SPLITS = cl.N_SPLITS         # 5     - the model OOF splits, consumed

#: The stacking blocks. The overlap's first season is the initial training
#: block (it cannot be scored - nothing precedes it to fit on); every later
#: season is scored, one block at a time, with an expanding train window.
INITIAL_SEASON = "2020-21"
SCORED_SEASONS = ("2021-22", "2022-23", "2023-24", "2024-25", "2025-26")

PREREG_NAME = "H1H2-prereg-market-anchor"

#: results/closing_line.json, reproduced as an ALIGNMENT check (the searchsorted
#: misalignment failure mode: a misaligned join produces confident nonsense).
REFERENCE = {
    "n_overlap": 5788,
    "market_log_loss": 0.5976574083334246,
    "model_log_loss": 0.6233982796244897,
    "model_avail_log_loss": 0.6203386376508432,
}


# --------------------------------------------------------------------------- #
# Pre-registration (appended by --prereg, BEFORE any run)
# --------------------------------------------------------------------------- #
def prereg_entry() -> dict:
    return {
        "name": PREREG_NAME,
        "status": "declared_before_execution",
        "declared_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "market-anchored blend (theory Batch A gate-keepers "
                           "H1 and H2): does the basketball information add "
                           "anything the closing price does not already contain?",
        "hypothesis": {
            "H1a_primary": "The walk-forward logistic stack of [price logit, "
                           "canonical-model logit] (P+M) beats the de-vigged "
                           "closing price alone (P) by >= 0.002 nats held-out "
                           "log loss, with the paired-bootstrap 95% CI "
                           "excluding zero and |z| > 3. This is THE alpha "
                           "question: is any of our information not already in "
                           "the close?",
            "H1b_descriptive": "P+M beats the model alone (M) - expected large "
                               "(the market alone beats M by 0.0227 nats, "
                               "|z| 7.28 on the 5,788-game overlap). No "
                               "decision weight: it shows the blend machinery "
                               "works.",
            "H2_primary": "Replacing the plain model logit with the "
                          "model+availability logit (P+M+A) beats P+M by >= "
                          "0.002 nats, CI excluding zero, |z| > 3. Is the "
                          "availability block's signal already priced by the "
                          "close? Dare et al. (2015) found player absence "
                          "fully priced at the close on 1995-2012 data; the "
                          "NBA changed its injury-report rules in Dec 2025, "
                          "so this is genuinely open on this data.",
        },
        "odds_role_note": "The closing line is a PRE-TIP-OFF quantity and is "
                          "legitimate as an INPUT for outcome prediction here. "
                          "The project's 'odds are never a feature' rule "
                          "governs the MARKET-EVALUATION models (betting/ROI "
                          "arms) only. On record so the distinction cannot be "
                          "quietly re-litigated later.",
        "arms": {
            "P": "de-vigged closing probability (proportional de-vig, "
                 "closing_line.join_odds' p_market)",
            "M": "canonical model OOF home-win probability, "
                 "load_design(with_availability=False) + the OOF machinery of "
                 "closing_line.model_probs (walk_forward_split, 5 splits)",
            "M+A": "the same with with_availability=True (the project's one "
                   "confirmed mechanism; pooled OOF log loss 0.61566)",
            "P+M": "logistic stack of [logit(P), logit(M)]",
            "P+M+A": "logistic stack of [logit(P), logit(M+A)] - the "
                     "'model+availability logit in place of the plain one' "
                     "parameterization, FIXED here ex ante. The residualized "
                     "availability variant is declared NOT run (no "
                     "parameterization selection).",
        },
        "de_vig": "PROPORTIONAL, fixed ex ante (Kelly_Criterion "
                  "devig_two_way(method='proportional'), identical to "
                  "join_odds' p_market up to rounding). Shin's de-vig is "
                  "declared NOT run: choosing the de-vig after seeing which "
                  "one flatters the model is selection.",
        "stacking_protocol": "Strictly walk-forward BY SEASON BLOCK. Scored "
                             "seasons " + ", ".join(SCORED_SEASONS) + "; the "
                             "overlap's first season (" + INITIAL_SEASON + ", "
                             "460 rows) is the initial training block and is "
                             "NOT scored. For scored season k the stack is fit "
                             "on every overlap row of the seasons before k "
                             "(expanding window) and scores k only. Features "
                             "logit(p) = log(p/(1-p)) with p clipped to "
                             "[1e-6, 1-1e-6] as in nba_data.evaluate. Pipeline "
                             "StandardScaler + LogisticRegression(C=1.0, "
                             "max_iter=300, random_state=42); imputation is "
                             "unnecessary (all logits finite by construction) "
                             "and the scaler + estimator are fit on train "
                             "blocks only. Train/test disjointness is ASSERTED "
                             "on season labels and on dates (max train date < "
                             "min test date). Identical scored rows across "
                             "arms are ASSERTED (single scored frame, all-arm "
                             "finite).",
        "comparisons": "delta = loss(first arm) - loss(second arm); POSITIVE = "
                       "the second arm is better (nba_data.paired_bootstrap_"
                       "delta's convention, stated in every row). H1a: "
                       "loss(P) - loss(P+M). H1b: loss(M) - loss(P+M). H2: "
                       "loss(P+M) - loss(P+M+A).",
        "decision_rule": "H1a succeeds iff delta >= 0.002 nats (the "
                         "pre-registered minimum effect), the 95% CI excludes "
                         "zero in the helpful direction, AND |z| > 3. H2 is "
                         "governed by the same rule on its own delta. "
                         "Otherwise: 'sub-bar lead' if the CI excludes zero "
                         "with |z| <= 3; 'null' if the CI straddles zero; "
                         "'priced / market better' if the CI excludes zero "
                         "against the candidate. A null is a complete and "
                         "valuable result. H1b is descriptive only.",
        "multiplicity": "3 primary looks (H1a, H1b, H2) -> Bonferroni "
                        "0.05/3 = 0.0167 one-sided (|z| ~ 2.13). The project "
                        "bar |z| > 3 governs on top. The bootstrap seeds do "
                        "not add looks (same rows resampled). Per-season "
                        "deltas, the stacking-weight drift table and the "
                        "residual diagnostic are descriptive, carry no "
                        "decision weight and add no looks.",
        "settings": "Primary: bootstrap seed 42 (n_boot 2000). Replications: "
                    "seeds 137 and 2718. The stack fits and the walk-forward "
                    "splits are deterministic (lbfgs, fixed random_state), so "
                    "the seed dimension probes CI variability only; deltas are "
                    "expected identical across seeds and any non-identity is "
                    "itself a bug signal.",
        "metrics": "Primary: held-out log loss on the scored rows. Also Brier, "
                   "ROC-AUC, accuracy and the nba_data.evaluate calibration "
                   "table per arm; SE = (ci_hi - ci_lo)/(2*1.959964); |z| = "
                   "delta/SE. Per-season log-loss deltas for every hypothesis "
                   "(season drift read). Stacking-weight drift: walk-forward "
                   "(train) weights vs a diagnostic-only refit on the scored "
                   "block (NEVER used for scoring) - do the weights drift, is "
                   "the model weight ~ 0 in later seasons?",
        "sizing": "SE on this overlap is ~0.0015-0.002 nats for a paired "
                  "log-loss delta; the pre-registered minimum effect (0.002 "
                  "nats) sits AT or below |z| ~ 1-1.5 at that SE, so this "
                  "design can only confirm effects meaningfully larger than "
                  "0.002 nats (~0.005-0.006 at the wide end of the SE range). "
                  "A null at 0.002 nats therefore does NOT establish that "
                  "there is nothing there - it establishes nothing at the bar. "
                  "This power caveat is fixed before the run.",
        "causal_convention": "Everything used to score a game is derived from "
                             "information strictly before it: the closing line "
                             "is pre-tip-off; the model logits are OUT-OF-FOLD "
                             "from walk_forward_split (each block's model "
                             "trained on games strictly before the block); the "
                             "stack for a scored season is fit only on earlier "
                             "seasons' rows and labels. Availability is the "
                             "prior-game squad's expected minutes (never "
                             "tonight's participants).",
        "leakage_checks_asserted_in_code": "(1) ALIGNMENT: the pipeline must "
                             "reproduce results/closing_line.json on the "
                             "5,788-game overlap (market 0.597657, model "
                             "0.623398, model+availability 0.620339) within "
                             "1e-6 or it raises - the searchsorted "
                             "misalignment failure mode. (2) Stacking "
                             "train/test disjointness on seasons AND dates. "
                             "(3) Scaler/estimator fit on train blocks only. "
                             "(4) Identical scored rows across arms (single "
                             "frame, all arms finite, asserted). (5) No "
                             "in-sample model predictions enter the stack "
                             "(OOF logits only).",
        "prior_evidence": "AGAINST our alpha: the market beats the canonical "
                          "model by 0.0257 nats |z| 8.18 and beats "
                          "model+availability by 0.0227 nats |z| 7.28 "
                          "(results/closing_line.json); Vandenbruaene et al. "
                          "2022 found no post-correction profitability across "
                          "628 implementations. FOR even trying: a blend can "
                          "add a little even when each arm alone loses, and "
                          "the availability block is the project's one "
                          "confirmed mechanism (+0.0038 nats over the model) "
                          "whose increment beyond the PRICE has never been "
                          "measured. Dare et al. 2015 (absence fully priced, "
                          "1995-2012) is the prior for H2's null; the Dec "
                          "2025 NBA injury-report rule change is the reason "
                          "it is re-opened.",
        "overlap_note": "H1/H2 stacking arms are new; nothing in "
                        "results/experiments.jsonl measures a market-anchored "
                        "blend as of this declaration. This is not a re-run of "
                        "H3 (recalibration: its lesson - never fit a "
                        "calibrator on the tail it scores - is enforced "
                        "here), nor of H4/H11 (feature blocks). Concurrent "
                        "agents are editing Tests/, src/DataProviders/feat_* "
                        "docstrings, scripts/round3_common.py and the injury "
                        "module; this script consumes load_odds/join_odds/"
                        "model_probs READ-ONLY and touches only itself and "
                        "the append-only log.",
        "variants_declared_not_run": "Shin de-vig; residualized-availability "
                        "parameterization; any threshold/ROI sweep (the "
                        "estimator question only - profitability is "
                        "closing_line.py's job and is not re-opened here); "
                        "any hyperparameter or block-length search. If a "
                        "variant is run later it must carry its selection "
                        "cost.",
        "runtime_note": "load_design x 2 arms + 10 OOF logistic fits: ~1-3 "
                        "min, cached under /tmp/opencode/h1h2/ keyed on the "
                        "game set; stacking fits are seconds.",
    }


# --------------------------------------------------------------------------- #
# Model logits (consumed from closing_line, cached, never re-derived)
# --------------------------------------------------------------------------- #
def _cache_key(games: pd.DataFrame, with_availability: bool) -> str:
    h = hashlib.md5()
    h.update("|".join(map(str, games["GAME_ID"].to_numpy())).encode())
    h.update(f"|avail={with_availability}|hl={HALFLIFE}|mg={MIN_GAMES}"
             f"|splits={N_SPLITS}".encode())
    return h.hexdigest()


def cached_model_probs(games: pd.DataFrame, with_availability: bool):
    """closing_line.model_probs, memoised on the game set. Read-only use."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f = CACHE_DIR / f"probs_{_cache_key(games, with_availability)}.npz"
    if f.exists():
        z = np.load(f)
        return z["p"], z["y"]
    splits = nd.walk_forward_split(games, n_splits=N_SPLITS)
    p, y = cl.model_probs(games, splits, with_availability=with_availability)
    np.savez_compressed(f, p=p, y=y)
    return p, y


# --------------------------------------------------------------------------- #
# The walk-forward stacking
# --------------------------------------------------------------------------- #
def to_logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), LOGIT_CLIP, 1 - LOGIT_CLIP)
    return np.log(p / (1 - p))


def fit_stack(X: np.ndarray, y: np.ndarray):
    """StandardScaler + LogisticRegression, fit on TRAIN rows only."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([("s", StandardScaler()),
                     ("c", LogisticRegression(C=1.0, max_iter=300,
                                              random_state=SEED))])
    pipe.fit(X, y)
    return pipe


def raw_scale_weights(pipe) -> dict:
    """Standardised-fit coefficients expressed on the RAW logit scale."""
    sc = pipe.named_steps["s"]
    clf = pipe.named_steps["c"]
    coef = clf.coef_[0] / sc.scale_
    intercept = float(clf.intercept_[0] - np.sum(clf.coef_[0] * sc.mean_ / sc.scale_))
    return {"coef": [float(c) for c in coef], "intercept": intercept}


def walk_forward_stack(scored: pd.DataFrame, model_col: str,
                       name: str) -> tuple[np.ndarray, list[dict]]:
    """Season-block walk-forward stacking of [price logit, model_col logit].

    Returns the out-of-block stacked probability (NaN nowhere - every scored
    season is scored) and a per-block weight table. The diagnostic test-block
    refit in the table is NEVER used for scoring.
    """
    feats = ["logit_price", model_col]
    seasons = [s for s in SCORED_SEASONS if (scored["SEASON"] == s).any()]
    scorable = scored["SEASON"].isin(SCORED_SEASONS).to_numpy()
    out = np.full(len(scored), np.nan)
    table = []
    for k, s in enumerate(seasons):
        test = scored["SEASON"] == s
        train = scored["SEASON"].isin(seasons[:k]) | (scored["SEASON"] == INITIAL_SEASON)
        if k == 0:
            train = scored["SEASON"] == INITIAL_SEASON
        tr, te = scored.loc[train], scored.loc[test]
        # Disjointness, asserted on both axes (the H3 lesson).
        assert not (set(tr["SEASON"]) & set(te["SEASON"])), \
            f"{name}: train/test seasons overlap for {s}"
        assert len(tr) and len(te), f"{name}: empty train or test block for {s}"
        assert tr["GAME_DATE"].max() < te["GAME_DATE"].min(), \
            f"{name}: train rows exist on/after {s}'s first game"
        pipe = fit_stack(tr[feats].to_numpy(), tr["y"].to_numpy())
        out[test.to_numpy()] = pipe.predict_proba(te[feats].to_numpy())[:, 1]
        w_tr = raw_scale_weights(pipe)
        # Diagnostic ONLY: what the weights would be if fit on the scored
        # block. The gap to the train weights is the drift read. Nothing about
        # these enters a prediction.
        pipe_te = fit_stack(te[feats].to_numpy(), te["y"].to_numpy())
        w_te = raw_scale_weights(pipe_te)
        table.append({
            "scored_season": s,
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "train_seasons": sorted(set(tr["SEASON"])),
            "train_max_date": str(tr["GAME_DATE"].max().date()),
            "test_min_date": str(te["GAME_DATE"].min().date()),
            "train_price_coef": w_tr["coef"][0],
            "train_model_coef": w_tr["coef"][1],
            "train_intercept": w_tr["intercept"],
            "testfit_price_coef": w_te["coef"][0],
            "testfit_model_coef": w_te["coef"][1],
            "testfit_intercept": w_te["intercept"],
            "gap_price_coef": w_te["coef"][0] - w_tr["coef"][0],
            "gap_model_coef": w_te["coef"][1] - w_tr["coef"][1],
        })
    # Every SCORED row gets exactly one out-of-block prediction; the initial
    # train block is by design never scored (it has no preceding season to fit
    # on) and must stay NaN - which is what the second assertion pins.
    assert np.isfinite(out[scorable]).all(), f"{name}: stack left unscored rows"
    assert np.isnan(out[~scorable]).all(), \
        f"{name}: the train-only block leaked a score"
    return out, table


# --------------------------------------------------------------------------- #
# Scoring helpers
# --------------------------------------------------------------------------- #
def ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, float), LOGIT_CLIP, 1 - LOGIT_CLIP)
    y = np.asarray(y).astype(int)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def compare(y, p_a, p_b, seed) -> dict:
    d = nd.paired_bootstrap_delta(y, p_a, p_b, n_boot=N_BOOT, seed=seed)
    se = (d["ci_hi"] - d["ci_lo"]) / (2 * 1.959964)
    z = d["delta"] / se if se > 0 else float("nan")
    return {**d, "se": se, "z": z}


ARM_COLUMN = {"P": "p_market", "M": "p_model", "M+A": "p_model_avail",
              "P+M": "p_P+M", "P+M+A": "p_P+M+A"}


def per_season_delta(scored: pd.DataFrame, arm_a: str, arm_b: str) -> list[dict]:
    rows = []
    for s in SCORED_SEASONS:
        d = scored.loc[scored["SEASON"] == s]
        if not len(d):
            continue
        y = d["y"].to_numpy().astype(int)
        rows.append({"season": s, "n": int(len(d)),
                     "delta": ll(y, d[ARM_COLUMN[arm_a]].to_numpy())
                              - ll(y, d[ARM_COLUMN[arm_b]].to_numpy())})
    return rows


def log(rec: dict) -> None:
    with JSONL.open("a") as fh:
        fh.write(json.dumps(rec, default=float) + "\n")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prereg", action="store_true",
                    help="append the pre-registration to "
                         "results/experiments.jsonl and exit")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    if args.prereg:
        rec = prereg_entry()
        log(rec)
        print(f"appended pre-registration {rec['name']!r} to {JSONL}")
        return 0

    # The pre-registration must exist BEFORE any measurement is produced.
    declared = [json.loads(l)["name"] for l in JSONL.open() if l.strip()]
    if PREREG_NAME not in declared:
        raise SystemExit(f"results/experiments.jsonl has no {PREREG_NAME} "
                         "pre-registration; run with --prereg first")
    if args.no_cache:
        for f in CACHE_DIR.glob("probs_*.npz"):
            f.unlink()

    t0 = time.time()
    prereg_utc = [json.loads(l)["declared_utc"] for l in JSONL.open()
                  if PREREG_NAME in l][0]

    # ---- pipeline, consumed read-only from closing_line --------------------
    games = nd.build_games()
    odds = cl.load_odds()
    joined = cl.join_odds(games, odds)
    row_of = {gid: i for i, gid in enumerate(games["GAME_ID"].to_numpy())}
    idx = joined["GAME_ID"].map(row_of).to_numpy()

    p_base, y_games = cached_model_probs(games, with_availability=False)
    p_avail, _ = cached_model_probs(games, with_availability=True)

    ov = joined.assign(y=y_games[idx], p_model=p_base[idx],
                       p_model_avail=p_avail[idx],
                       SEASON=games["SEASON"].to_numpy()[idx])
    ov = ov.dropna(subset=["p_model", "p_model_avail"]).reset_index(drop=True)

    # ---- leakage check 1: ALIGNMENT against results/closing_line.json ------
    ref = {"n_overlap": len(ov),
           "market_log_loss": ll(ov["y"], ov["p_market"]),
           "model_log_loss": ll(ov["y"], ov["p_model"]),
           "model_avail_log_loss": ll(ov["y"], ov["p_model_avail"])}
    for k, tol in (("n_overlap", 0), ("market_log_loss", 1e-6),
                   ("model_log_loss", 1e-6), ("model_avail_log_loss", 1e-6)):
        if abs(ref[k] - REFERENCE[k]) > tol:
            raise RuntimeError(
                f"ALIGNMENT CHECK FAILED: {k} = {ref[k]!r} but "
                f"results/closing_line.json says {REFERENCE[k]!r}. A "
                "misaligned join produces confident nonsense (the "
                "searchsorted failure mode); refusing to score.")
    print(f"alignment check passed: overlap {ref['n_overlap']:,} rows reproduce "
          f"closing_line.json (market {ref['market_log_loss']:.6f} / model "
          f"{ref['model_log_loss']:.6f} / model+avail "
          f"{ref['model_avail_log_loss']:.6f})")

    # ---- the scored frame: identical rows across every arm -----------------
    scored = ov.loc[ov["SEASON"].isin(SCORED_SEASONS)].copy()
    scored = scored.sort_values(["GAME_DATE"]).reset_index(drop=True)
    scored["logit_price"] = to_logit(scored["p_market"].to_numpy())
    scored["logit_model"] = to_logit(scored["p_model"].to_numpy())
    scored["logit_model_avail"] = to_logit(scored["p_model_avail"].to_numpy())
    init_rows = ov.loc[ov["SEASON"] == INITIAL_SEASON].copy()
    init_rows["logit_price"] = to_logit(init_rows["p_market"].to_numpy())
    init_rows["logit_model"] = to_logit(init_rows["p_model"].to_numpy())
    init_rows["logit_model_avail"] = to_logit(init_rows["p_model_avail"].to_numpy())
    stack_pool = pd.concat([init_rows, scored], ignore_index=True)

    p_pm, w_pm = walk_forward_stack(stack_pool, "logit_model", "P+M")
    p_pma, w_pma = walk_forward_stack(stack_pool, "logit_model_avail", "P+M+A")
    scored["p_P+M"] = p_pm[-len(scored):]
    scored["p_P+M+A"] = p_pma[-len(scored):]

    arms = {
        "P": scored["p_market"].to_numpy(),
        "M": scored["p_model"].to_numpy(),
        "M+A": scored["p_model_avail"].to_numpy(),
        "P+M": scored["p_P+M"].to_numpy(),
        "P+M+A": scored["p_P+M+A"].to_numpy(),
    }
    # leakage check 4: identical scored rows across arms, all finite
    y = scored["y"].to_numpy().astype(int)
    for k, v in arms.items():
        assert len(v) == len(y) and np.isfinite(v).all(), f"arm {k} is ragged/NaN"
        assert (v > 0).all() and (v < 1).all(), f"arm {k} leaves [0,1]"
    n = len(scored)
    print(f"\nscored rows: {n:,} (identical across all arms; the overlap's "
          f"{len(init_rows):,} {INITIAL_SEASON} rows are stacking TRAIN ONLY)")
    print("  per season  " + "  ".join(
        f"{s} {int((scored['SEASON'] == s).sum()):,}" for s in SCORED_SEASONS))
    print(f"  [{time.time() - t0:.0f}s]")

    # ---- arm table ---------------------------------------------------------
    print(f"\n{'arm':8s} {'log loss':>10s} {'Brier':>8s} {'AUC':>8s} {'acc':>7s}")
    print("-" * 46)
    metrics = {}
    for k, v in arms.items():
        e = nd.evaluate(y, v)
        metrics[k] = e
        print(f"{k:8s} {e['log_loss']:>10.5f} {e['brier']:>8.5f} "
              f"{e['auc']:>8.5f} {e['accuracy']:>7.4f}")

    print("\nCALIBRATION  mean_pred / actual by decile")
    for k in ("P", "P+M", "P+M+A"):
        print(f"  {k:6s} " + " ".join(
            f"{c['mean_pred']:.3f}/{c['actual']:.3f}(n{c['n']})"
            for c in metrics[k]["calibration"]))

    # ---- hypotheses --------------------------------------------------------
    print("\nARM-vs-ARM  (delta = loss(first) - loss(second); POSITIVE = the "
          "second arm is better)")
    print(f"{'comparison':22s} {'delta':>9s} {'CI95':>21s} {'|z|':>6s}  seed")
    print("-" * 70)
    hyps = {
        "H1a": ("P", "P+M", True),
        "H1b": ("M", "P+M", False),
        "H2": ("P+M", "P+M+A", True),
    }
    stack_desc = {
        "H1a": "walk-forward season-block logistic stack of "
               "[logit(price), logit(M)]; scaler+LR(C=1.0) fit on prior "
               "seasons' rows only; M logits OOF from "
               "closing_line.model_probs(with_availability=False)",
        "H1b": "P+M arm as in H1a; M arm is the bare OOF model "
               "(model_probs(with_availability=False)), no fitting here",
        "H2": "walk-forward season-block logistic stack of "
              "[logit(price), logit(M+A)]; scaler+LR(C=1.0) fit on prior "
              "seasons' rows only; M+A logits OOF from "
              "closing_line.model_probs(with_availability=True)",
    }
    results = {}
    for name, (a, b, primary) in hyps.items():
        rows = []
        for seed in (SEED, *REPLICATION_SEEDS):
            d = compare(y, arms[a], arms[b], seed=seed)
            d.update({"seed": seed, "first": a, "second": b})
            rows.append(d)
            flag = ("PASS" if (primary and d["delta"] >= MIN_EFFECT
                               and d["ci_lo"] > 0 and abs(d["z"]) > Z_BAR)
                    else "descriptive" if not primary
                    else "FAIL")
            print(f"{name} {a} vs {b:9s} {d['delta']:+9.5f} "
                  f"[{d['ci_lo']:+9.5f},{d['ci_hi']:+9.5f}] {abs(d['z']):6.2f}  "
                  f"{d['seed']:<4d} {flag}")
        results[name] = rows
    print(f"\nreference deltas on the FULL {ref['n_overlap']:,}-row overlap "
          "(context, from closing_line.json): market beats model by 0.0257 "
          "nats |z| 8.18, beats model+availability by 0.0227 nats |z| 7.28.")

    # per-season deltas (descriptive)
    per_season = {name: per_season_delta(scored, a, b)
                  for name, (a, b, _) in hyps.items()}
    print("\nPER-SEASON DELTAS  (positive = second arm better)")
    for name in hyps:
        print(f"  {name}:  " + "  ".join(
            f"{r['season']} {r['delta']:+.4f} (n {r['n']})"
            for r in per_season[name]))

    # ---- stacking weight drift --------------------------------------------
    print("\nSTACKING WEIGHTS (raw logit scale). train = the walk-forward fit "
          "actually used; testfit = diagnostic refit on the scored block, "
          "NEVER used for scoring.")
    print(f"{'arm':6s} {'season':8s} {'n_tr':>5s} {'price_tr':>9s} {'model_tr':>9s} "
          f"{'price_te':>9s} {'model_te':>9s} {'gap_model':>9s}")
    print("-" * 78)
    weights = {"P+M": w_pm, "P+M+A": w_pma}
    for arm_name, tab in weights.items():
        for r in tab:
            print(f"{arm_name:6s} {r['scored_season']:8s} {r['n_train']:>5,} "
                  f"{r['train_price_coef']:>9.3f} {r['train_model_coef']:>9.3f} "
                  f"{r['testfit_price_coef']:>9.3f} {r['testfit_model_coef']:>9.3f} "
                  f"{r['gap_model_coef']:>+9.3f}")

    # ---- residual diagnostic (no decision weight) --------------------------
    d_logit = scored["logit_model_avail"] - scored["logit_price"]
    top = np.abs(d_logit.to_numpy()) >= np.quantile(np.abs(d_logit.to_numpy()), 0.9)
    rest = ~top
    y_top, y_rest = y[top], y[rest]
    model_side = (scored["p_model_avail"].to_numpy() > 0.5).astype(int)
    market_side = (scored["p_market"].to_numpy() > 0.5).astype(int)
    diag = {
        "definition": "top decile of |logit(M+A) - logit(P)| on the scored "
                      "rows (the games where model and market disagree most)",
        "n_top": int(top.sum()), "n_rest": int(rest.sum()),
        "season_mix_top": {s: int((scored.loc[top, "SEASON"] == s).sum())
                           for s in SCORED_SEASONS},
        "mean_p_market_top": float(scored.loc[top, "p_market"].mean()),
        "mean_p_market_rest": float(scored.loc[rest, "p_market"].mean()),
        "market_favours_home_share_top": float((market_side[top] == 1).mean()),
        "market_favours_home_share_rest": float((market_side[rest] == 1).mean()),
        "model_market_same_side_share_top": float(
            (model_side[top] == market_side[top]).mean()),
        "model_market_same_side_share_rest": float(
            (model_side[rest] == market_side[rest]).mean()),
        "home_win_rate_top": float(y_top.mean()),
        "home_win_rate_rest": float(y_rest.mean()),
        "log_loss_P_top": ll(y_top, scored["p_market"].to_numpy()[top]),
        "log_loss_P_rest": ll(y_rest, scored["p_market"].to_numpy()[rest]),
        "log_loss_P+M+A_top": ll(y_top, scored["p_P+M+A"].to_numpy()[top]),
        "log_loss_P+M+A_rest": ll(y_rest, scored["p_P+M+A"].to_numpy()[rest]),
        "log_loss_M+A_top": ll(y_top, scored["p_model_avail"].to_numpy()[top]),
    }
    disagree = top & (model_side != market_side)
    diag["n_top_disagree_on_side"] = int(disagree.sum())
    if disagree.any():
        diag["win_rate_of_model_side_when_disagreeing"] = float(
            np.where(y[disagree] == 1, model_side[disagree] == 1,
                     model_side[disagree] == 0).mean())
        diag["win_rate_of_market_side_when_disagreeing"] = 1 - diag[
            "win_rate_of_model_side_when_disagreeing"]
    print("\nRESIDUAL DIAGNOSTIC (descriptive, no decision weight)")
    for k, v in diag.items():
        print(f"  {k}: {v}")

    # ---- verdicts ----------------------------------------------------------
    def classify(rows):
        d = rows[0]
        if d["delta"] >= MIN_EFFECT and d["ci_lo"] > 0 and abs(d["z"]) > Z_BAR:
            return ("ESTABLISHED", f"delta {d['delta']:+.5f} nats >= "
                    f"{MIN_EFFECT}, CI [{d['ci_lo']:+.5f}, {d['ci_hi']:+.5f}] "
                    f"excludes zero, |z| {abs(d['z']):.2f} > {Z_BAR}")
        if d["ci_lo"] > 0 and abs(d["z"]) > Z_BAR:
            return ("below minimum effect", f"CI excludes zero and |z| "
                    f"{abs(d['z']):.2f} > {Z_BAR} but delta {d['delta']:+.5f} "
                    f"< {MIN_EFFECT} nats")
        if d["ci_lo"] > 0:
            return ("sub-bar lead", f"CI [{d['ci_lo']:+.5f}, {d['ci_hi']:+.5f}] "
                    f"excludes zero but |z| {abs(d['z']):.2f} <= {Z_BAR}")
        if d["ci_hi"] < 0:
            return ("market better", f"CI [{d['ci_lo']:+.5f}, "
                    f"{d['ci_hi']:+.5f}] excludes zero AGAINST the candidate, "
                    f"|z| {abs(d['z']):.2f}")
        return ("NULL", f"CI [{d['ci_lo']:+.5f}, {d['ci_hi']:+.5f}] straddles "
                f"zero; nothing established")

    v1a, r1a = classify(results["H1a"])
    v2, r2 = classify(results["H2"])
    d1b = results["H1b"][0]
    sent1a = (f"H1a VERDICT: {v1a.upper()} - {r1a}. P+M does NOT beat the "
              f"closing price at the pre-registered bar."
              if v1a != "ESTABLISHED" else
              f"H1a VERDICT: ESTABLISHED - {r1a}. Basketball information adds "
              f"signal beyond the close.")
    sent2 = (f"H2 VERDICT: {v2.upper()} - {r2}. Availability's increment beyond "
             f"price+model is not established."
             if v2 != "ESTABLISHED" else
             f"H2 VERDICT: ESTABLISHED - {r2}. Availability adds signal the "
             f"close does not price.")
    sent1b = (f"H1b (descriptive): P+M beats M alone by {d1b['delta']:+.5f} "
              f"nats, CI [{d1b['ci_lo']:+.5f}, {d1b['ci_hi']:+.5f}], "
              f"|z| {abs(d1b['z']):.2f} - the blend machinery works.")
    print(f"\n{sent1a}\n{sent2}\n{sent1b}")

    leakage_check = ("passed: alignment reproduced closing_line.json on the "
                     f"{ref['n_overlap']:,}-row overlap; stacking train/test "
                     "disjoint on seasons AND dates (asserted per block); "
                     "scaler+estimator fit on train blocks only; identical "
                     "scored rows across arms (asserted, all arms finite); "
                     "only OOF model logits enter the stack")

    # ---- append-only log ---------------------------------------------------
    for name, (a, b, primary) in hyps.items():
        for r in results[name]:
            log({
                "name": f"H1H2-run-{name}-s{r['seed']}",
                "status": "executed",
                "declared_utc": prereg_utc,
                "experiment_type": "market-anchored blend result (Batch A "
                                   "gate-keeper " + name + ")",
                "protocol_ref": PREREG_NAME,
                "hypothesis": name,
                "primary_look": primary and r["seed"] == SEED,
                "arms": f"{r['second']} vs {r['first']}",
                "sign_convention": "delta = loss(" + r["first"] + ") - loss("
                                   + r["second"] + "); POSITIVE = the second "
                                   "arm is better",
                "seed": r["seed"], "n_boot": N_BOOT,
                "n_scored": n,
                "delta": r["delta"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                "se": r["se"], "z": r["z"],
                "log_loss_first": metrics[r["first"]]["log_loss"],
                "log_loss_second": metrics[r["second"]]["log_loss"],
                "arm_metrics": {k: {kk: vv for kk, vv in metrics[k].items()
                                    if kk != "calibration"} for k in (r["first"], r["second"])},
                "per_season_deltas": per_season[name],
                "decision_rule": "delta >= 0.002 nats AND CI excludes zero "
                                 "AND |z| > 3 (" + str(Z_BAR) + ") at seed "
                                 + str(SEED) + "; Bonferroni 0.0167 one-sided "
                                 "across 3 primary looks; |z| > 3 governs",
                "multiplicity": "3 primary looks -> Bonferroni 0.0167 "
                                "one-sided; the |z| > 3 bar governs",
                "model": stack_desc[name],
                "de_vig": "proportional (fixed ex ante)",
                "leakage_check": leakage_check,
            })
    log({
        "name": "H1H2-diag-stacking-weights", "status": "executed",
        "declared_utc": prereg_utc,
        "experiment_type": "stacking-weight drift diagnostic (descriptive)",
        "protocol_ref": PREREG_NAME,
        "weights": weights,
        "note": "train_* = walk-forward fit used for scoring; testfit_* = "
                "diagnostic refit on the scored block, NEVER used for "
                "scoring. gap = testfit - train.",
    })
    log({
        "name": "H1H2-diag-residual", "status": "executed",
        "declared_utc": prereg_utc,
        "experiment_type": "residual diagnostic (descriptive, no decision "
                           "weight)",
        "protocol_ref": PREREG_NAME,
        "diagnostic": diag,
    })
    log({
        "name": "H1H2-verdict", "status": "executed",
        "declared_utc": prereg_utc,
        "experiment_type": "verdict_record",
        "protocol_ref": PREREG_NAME,
        "rule": "H1a/H2 succeed iff delta >= 0.002 nats AND the 95% CI "
                "excludes zero AND |z| > 3 at the primary setting (seed 42); "
                "3 primary looks -> Bonferroni 0.0167 one-sided; |z| > 3 "
                "governs",
        "H1a": {"verdict": v1a, "detail": r1a,
                "delta": results["H1a"][0]["delta"],
                "ci": [results["H1a"][0]["ci_lo"], results["H1a"][0]["ci_hi"]],
                "z": results["H1a"][0]["z"]},
        "H2": {"verdict": v2, "detail": r2,
               "delta": results["H2"][0]["delta"],
               "ci": [results["H2"][0]["ci_lo"], results["H2"][0]["ci_hi"]],
               "z": results["H2"][0]["z"]},
        "H1b_descriptive": {"delta": results["H1b"][0]["delta"],
                            "ci": [results["H1b"][0]["ci_lo"],
                                   results["H1b"][0]["ci_hi"]],
                            "z": results["H1b"][0]["z"]},
        "sentences": [sent1a, sent2, sent1b],
        "replications": {name: [{"seed": r["seed"], "delta": r["delta"],
                                 "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                                 "z": r["z"]} for r in results[name][1:]]
                         for name in hyps},
        "per_season_deltas": per_season,
        "n_scored": n,
        "arm_metrics": metrics,
        "alignment_reference": ref,
        "leakage_check": leakage_check,
        "sizing_caveat": "SE ~ 0.0015-0.002 nats on this overlap; a true "
                         "0.002-nat effect is at/below |z| ~ 1.5, so a null "
                         "here establishes nothing AT the bar, not nothing "
                         "at all",
    })
    print(f"\nappended {len(hyps) * 3 + 3} rows to {JSONL}  "
          f"[{time.time() - t0:.0f}s total]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
