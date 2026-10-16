"""Does recency weighting buy anything, or is the ceiling the information itself?

Issue #20 measured that the committed feature set is nearly static: between
consecutive games a team's points, shooting, rebounds and record move by roughly
one-eighth of a seasonal standard deviation. A model reading those features
reads a slowly-drifting season average, and cannot see form, injuries or lineups
because none of them are in its inputs.

That is a claim about the FEATURES, and it is testable without rebuilding
anything. ``rolling_team_features`` computes the same feature set at any decay
speed. A very long halflife puts every past game at nearly equal weight, which
reproduces the static season average. A short halflife makes the features
responsive. Holding the feature set, the model class, the splits and the seed
fixed, the only thing that varies is how fast the features move.

    arm              halflife   interpretation
    season-average   10000      every past game weighted ~equally
    slow              30        form over roughly a month
    medium            10        the committed default
    fast               5        form over the last week
    very-fast          2        last few games

If recency were the missing ingredient, log loss would fall monotonically as the
halflife shortens. If the ceiling is information rather than responsiveness, the
curve is flat and every difference sits inside the bootstrap interval.

THE ANSWER MAY BE NULL, AND THAT IS A RESULT
--------------------------------------------
This script reports a paired bootstrap interval rather than a point estimate,
because the differences here are of order 0.001-0.002 nats and a point estimate
on that scale is noise dressed as a finding. Issue #20 explicitly accepts a null
outcome: "the point is that it be measured rather than assumed."

Sign convention, which has been got wrong in this repo before:
``paired_bootstrap_delta(y, p_a, p_b)`` returns ``loss(a) - loss(b)``, so a
POSITIVE delta means **b is better**.

Run:  python scripts/recency_ablation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT, ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from DataProviders import nba_data as nd  # noqa: E402

# Long enough that the exponential weights are effectively uniform, which
# reproduces the static season average #20 describes.
ARMS = [
    ("season-average", 10000.0),
    ("slow", 30.0),
    ("medium", 10.0),
    ("fast", 5.0),
    ("very-fast", 2.0),
]

SEED = 42
N_SPLITS = 5


def feature_block(games: pd.DataFrame, halflife: float) -> pd.DataFrame:
    """The feature set, at one decay speed.

    Identical columns for every arm - only the decay differs. `rest_and_travel`
    is schedule-derived and does not depend on the halflife, so it is identical
    across arms by construction rather than by luck.
    """
    roll = nd.rolling_team_features(games, halflife=halflife, min_games=5)
    sched = nd.rest_and_travel(games)
    # Select numeric columns explicitly rather than dropping by name. `roll` and
    # `sched` each carry a GAME_DATE, and concatenating frames with a datetime
    # column hands the imputer a DateTime64DType it cannot promote to float --
    # a loud failure, but one that is avoided by asking for what we want rather
    # than by maintaining a list of things to delete.
    keep = lambda d: d.select_dtypes(include=[np.number])  # noqa: E731
    block = pd.concat(
        [keep(roll).reset_index(drop=True), keep(sched).reset_index(drop=True)],
        axis=1,
    )
    # A feature identical in every row cannot be predicted from and only makes
    # the design matrix singular; drop it rather than let one coefficient be
    # unidentifiable.
    block = block.loc[:, block.nunique(dropna=False) > 1]
    assert len(block) == len(games), "feature block lost rows"
    return block.reset_index(drop=True)


def run_arm(games, splits, halflife, C=1.0, max_iter=300):
    """Out-of-fold predictions for one decay speed, same splits as every other arm."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X = feature_block(games, halflife)
    y = games["y"].to_numpy()
    oof = np.full(len(y), np.nan)

    for tr, te in splits:
        pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=C, max_iter=max_iter,
                                       random_state=SEED)),
        ])
        # Imputation and scaling are fit on TRAIN ONLY. Fitting them on the whole
        # frame would let a test block's median and spread inform the model's
        # representation of it -- a leak that is invisible in the output and small
        # enough to look like noise.
        pipe.fit(X.iloc[tr], y[tr])
        oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]

    # Only the walk-forward test blocks are out-of-sample. The first half of the
    # data is training-only by design, so scoring it would score the model on its
    # own training rows. Every arm is scored on the SAME rows, so the paired
    # comparison stays valid regardless of which subset that is.
    mask = ~np.isnan(oof)
    if not mask.any():
        raise AssertionError("no out-of-fold predictions were produced")
    return oof[mask], y[mask], mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="results/recency_ablation.json")
    args = ap.parse_args()

    games = nd.build_games()
    splits = nd.walk_forward_split(games, n_splits=N_SPLITS)
    y = games["y"].to_numpy()
    print(f"games {len(games):,}  splits {len(splits)}  "
          f"train sizes {[len(tr) for tr, _ in splits]}")
    print(f"test sizes  {[len(te) for _, te in splits]}")
    print(f"out-of-fold rows: {sum(len(te) for _, te in splits):,}\n")

    preds, metrics = {}, {}
    for name, hl in ARMS:
        p, yy, mask = run_arm(games, splits, hl)
        preds[name] = p
        # Every arm covers the same out-of-fold rows, so this y is the reference
        # labels for the paired comparison below.
        y = yy
        m = nd.evaluate(yy, p)
        metrics[name] = m
        print(f"  {name:>15s}  halflife {hl:>7.0f}  "
              f"log loss {m['log_loss']:.5f}  brier {m['brier']:.5f}  "
              f"auc {m['auc']:.5f}  acc {m['accuracy']:.4f}")

    # The reference arm is the static season average, because that is what #20 says
    # the committed features already are. A POSITIVE delta means the other arm is
    # BETTER than the static baseline.
    ref = "season-average"
    print(f"\npaired bootstrap vs {ref} (positive = other arm is better)")
    print(f"  {'arm':>15s}  {'delta':>10s}  {'95% CI':>24s}  {'P(better)':>10s}  {'|z|':>6s}")
    comparisons = {}
    for name, hl in ARMS:
        if name == ref:
            continue
        d = nd.paired_bootstrap_delta(y, preds[ref], preds[name],
                                      n_boot=args.n_boot, seed=SEED)
        se = (d["ci_hi"] - d["ci_lo"]) / (2 * 1.959964)
        z = abs(d["delta"] / se) if se > 0 else float("nan")
        comparisons[name] = {**d, "z": z, "halflife": hl}
        ci = f"[{d['ci_lo']:+.5f}, {d['ci_hi']:+.5f}]"
        print(f"  {name:>15s}  {d['delta']:+.5f}  {ci:>24s}  "
              f"{d['p_positive']:>10.3f}  {z:>6.2f}")

    print("\nVERDICT")
    sig = [n for n, d in comparisons.items()
           if d["ci_lo"] > 0 or d["ci_hi"] < 0]
    if sig:
        print(f"  intervals EXCLUDING zero: {sig}")
        print("  A difference this size survives the interval. Read it before quoting.")
    else:
        print("  Every interval STRADDLES zero. No decay speed is distinguishable")
        print("  from the static season average on this evidence.")
        best = min(metrics, key=lambda k: metrics[k]["log_loss"])
        print(f"  Best point estimate is {best} at "
              f"{metrics[best]['log_loss']:.5f}, but the spread across arms is "
              f"{max(m['log_loss'] for m in metrics.values()) - min(m['log_loss'] for m in metrics.values()):.5f}"
              " nats -- smaller than the width of any single interval above.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_games": int(len(games)),
        "n_splits": len(splits),
        "seed": SEED,
        "reference_arm": ref,
        "metrics": metrics,
        "comparisons": comparisons,
    }, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())