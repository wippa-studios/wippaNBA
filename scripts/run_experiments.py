"""Run the pre-registered experiment grid and print the results table.

Every experiment below is declared here BEFORE any of them is run, with the reason
it is expected to help. That ordering is the point: a list written after the
results is a summary, not a search log, and it cannot rule out the possibility that
the winner was picked from a longer list of tries.

Reporting rules, applied to the output:

  * Every experiment is printed. There is no top-1 display.
  * The selection cost of the winner is printed explicitly - the gap between the
    best run and the median run - because a best-of-N is the maximum of N draws.
  * Log loss is the ranking metric. Accuracy is shown but never used to pick.
  * The train/test gap is shown for every run, so a win bought by memorising the
    training folds cannot hide behind an average.

Run:  python scripts/run_experiments.py
      python scripts/run_experiments.py --stage 2
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from DataProviders import nba_data as nd  # noqa: E402
from experiments import (Experiment, N_SPLITS, append, build_frame,  # noqa: E402
                         load_log, run_one)

# ===========================================================================
# THE PRE-REGISTERED GRID
#
# Rationales are one line each and were written before the run. Several predict
# NOTHING - that is intentional. A pre-registered null is a result; an unplanned
# one is noise with a story attached.
# ===========================================================================

STAGE_1 = [
    Experiment(
        name="E00-baseline-logistic",
        features="shared_full",
        model="logistic",
        rationale="Reference. Every shared block, strongest regularisation-free "
                  "linear model. Nothing here is new; this is the line to beat.",
    ),
    Experiment(
        name="E01-baseline-logistic-C0.1",
        features="shared_full",
        model="logistic_C0.1",
        rationale="Expect a small win or a null: 28+ mildly collinear columns on "
                  "13k rows overfit the walk-forward folds, so stronger L2 should "
                  "help if overfitting is the binding constraint.",
    ),
]

STAGE_2 = [
    Experiment(
        name="E10-drop-home-split",
        features="no_home_split",
        model="logistic",
        rationale="Ablation of the one replicated feature finding. Removing it "
                  "should COST log loss; if it does not, it was not real.",
    ),
    Experiment(
        name="E11-drop-opp-adjusted",
        features="no_opp_adjusted",
        model="logistic",
        rationale="Ablation of the pre-registered null. Expect no change, which "
                  "confirms the 0.98 correlation reading.",
    ),
    Experiment(
        name="E12-drop-sos",
        features="no_sos",
        model="logistic",
        rationale="Ablation of the block that died in replication. Expect no "
                  "change at five folds too - the artifact was only visible at eight.",
    ),
    Experiment(
        name="E13-drop-opp-chars",
        features="no_opp_chars",
        model="logistic",
        rationale="Ablation of the never-read OPP_* columns. Measured +0.00000 as "
                  "a null, so this should confirm it inside the model.",
    ),
    Experiment(
        name="E14-home-split-only",
        features="home_split_only",
        model="logistic",
        rationale="Expect a clear loss: schedule and team form are the bulk of the "
                  "signal, and removing them should cost far more than any block "
                  "ablation. This is the control that proves the harness can "
                  "detect a real feature.",
    ),
    Experiment(
        name="E15-schedule-only",
        features="schedule_only",
        model="logistic",
        rationale="Expect a large loss. If this beats E00 the harness is broken.",
    ),
]

STAGE_3 = [
    Experiment(
        name="E20-rf-full",
        features="shared_full",
        model="rf",
        rationale="Expect WORSE log loss than logistic: the data is 13k rows of "
                  "smooth team-strength signal, and boosting already lost to "
                  "logistic by a significant interval. RF's axis-aligned splits "
                  "should suit it even less.",
    ),
    Experiment(
        name="E21-histgb-full",
        features="shared_full",
        model="histgb",
        rationale="Expect worse than logistic, consistent with the existing "
                  "HistGB-vs-logistic measurement of -0.00431 [-0.00601, -0.00255].",
    ),
    Experiment(
        name="E22-xgb-full",
        features="shared_full",
        model="xgb",
        rationale="Expect worse than logistic, consistent with the existing XGB "
                  "measurement of -0.00394 [-0.00557, -0.00229].",
    ),
]

STAGE_4 = [
    Experiment(
        name="E30-player-elo",
        features="shared_full",
        model="logistic",
        rationale="Reference for the Elo arm. Identical to E00, so any movement is "
                  "attributable to the six player-Elo columns and nothing else.",
    ),
    Experiment(
        name="E31-drop-player-elo",
        features="no_player_elo",
        model="logistic",
        rationale="The ablation. A player's Elo from TEAM results is largely a "
                  "measure of the team they play for, so this may simply restate "
                  "ewm_margin on a different scale - which is why the block is "
                  "correlated against it BEFORE being scored. Correlations: "
                  "elo_available 0.16, elo_mean 0.09, elo_top5 0.07 against "
                  "home_ewm_margin. Low, so new information rather than a rescaling.",
    ),
    Experiment(
        name="E32-player-elo-C0.1",
        features="shared_full",
        model="logistic_C0.1",
        rationale="If the Elo columns add correlated signal, stronger L2 should "
                  "recover a little of whatever regularisation is being spent on "
                  "them. Expect a null; recorded so the direction is on the record.",
    ),
]

# ===========================================================================
# STAGE 5 - models named by an external survey of ~45 public NBA repos.
#
# Taken from `docs/repo_survey.md`, a compile of published READMEs. The point is
# not to adopt someone else's result but to FAIL these explicitly: they are the
# architectures other people report winning, and this project's own track record
# (boosting loses to logistic by a significant interval) says at least some of
# them should lose here too. An untried rival is an untested claim.
# ===========================================================================

STAGE_5 = [
    Experiment(
        name="E40-gaussian-nb-pca20",
        features="shared_full",
        model="gnb_pca",
        rationale="Named BEST by luke-lite/NBA-Prediction-Modeling (53 stars). A "
                  "different hypothesis class from every other entry: assumes "
                  "features are conditionally independent given the class, which "
                  "is plainly false for correlated team statistics. Expect WORSE "
                  "than logistic, and worth failing out loud.",
    ),
    Experiment(
        name="E41-logistic-pca20",
        features="shared_full",
        model="logistic_pca",
        rationale="rileypsmith reports 80 PCs beating 10. The question here is "
                  "narrower and testable: does unsupervised compression onto 20 "
                  "components help, help or hurt versus keeping all 59 columns "
                  "in the logistic? Expect a null - PCA discards exactly the "
                  "small-variance columns that a regularised model was about to "
                  "down-weight anyway.",
    ),
    Experiment(
        name="E42-ridge",
        features="shared_full",
        model="ridge",
        rationale="A linear model that is not logistic, as a check that the "
                  "logistic's advantage is about the loss function rather than "
                  "about being linear at all. Ridge is worse calibrated than "
                  "logistic by construction, so expect a higher log loss even if "
                  "its accuracy is competitive - which is the point of ranking on "
                  "log loss rather than accuracy.",
    ),
]

STAGES = {"1": STAGE_1, "2": STAGE_2, "3": STAGE_3, "4": STAGE_4, "5": STAGE_5,
          "all": STAGE_1 + STAGE_2 + STAGE_3,
          "elo": STAGE_4, "survey": STAGE_5}


def main() -> int:
    args = sys.argv[1:]
    stage = "all"
    for a in args:
        if a in STAGES:
            stage = a
    exps = STAGES[stage]

    games = nd.build_games()
    splits = nd.walk_forward_split(games, n_splits=N_SPLITS)
    print(f"games {len(games):,}  folds {len(splits)}  "
          f"train {[len(t) for t, _ in splits]}  test {[len(v) for _, v in splits]}")
    print(f"metric: LOG LOSS (primary). Brier/AUC/accuracy secondary, not ranked.\n")
    print(f"{len(exps)} pre-registered experiments in stage '{stage}'\n")

    results = []
    base = None
    for exp in exps:
        rec = run_one(exp, games, splits)
        if exp.name.startswith("E00"):
            base = rec["log_loss"]
        results.append(rec)
        append(rec)
        delta = "" if base is None else f"{rec['log_loss'] - base:+.5f}"
        flag = ""
        if rec["train_test_gap"] > 0.10:
            flag = "  <-- LARGE GAP"
        print(f"  {rec['name']:<32s} n={rec['n_features']:>3d}  "
              f"LL {rec['log_loss']:.5f}  d={delta:>9s}  "
              f"gap {rec['train_test_gap']:+.4f}{flag}")

    print()
    print("=" * 104)
    print("RESULTS")
    print("=" * 104)
    print(f"  {'experiment':<32s} {'feats':>5s} {'model':<14s} "
          f"{'log loss':>9s} {'vs base':>9s} {'Brier':>7s} {'AUC':>7s} "
          f"{'acc':>6s} {'gap':>7s}")
    print("  " + "-" * 100)
    for r in results:
        d = "" if base is None else f"{r['log_loss'] - base:+.5f}"
        print(f"  {r['name']:<32s} {r['n_features']:>5d} {r['model']:<14s} "
              f"{r['log_loss']:>9.5f} {d:>9s} {r['brier']:>7.4f} "
              f"{r['auc']:>7.4f} {r['accuracy']:>6.4f} {r['train_test_gap']:>+7.4f}")

    ll = np.array([r["log_loss"] for r in results])
    best = results[int(np.argmin(ll))]
    print()
    print(f"  best by log loss: {best['name']}  ({best['log_loss']:.5f})")
    print(f"  selection cost   : best - median = "
          f"{ll.min() - float(np.median(ll)):+.5f} nats")
    print(f"                    best - worst   = "
          f"{ll.min() - ll.max():+.5f} nats over {len(ll)} runs")
    print("  The gap between best and median is the expected size of a maximum")
    print("  of that many noisy draws. It is not evidence of a real difference.")
    print()
    print("  Top features by magnitude (mean across folds):")
    for name, val in best["top_features"]:
        print(f"    {name:<34s} {val:.4f}")

    log = load_log()
    print(f"\n  {len(log)} experiments now in results/experiments.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())