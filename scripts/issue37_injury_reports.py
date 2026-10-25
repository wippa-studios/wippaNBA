"""Issue #37: do timestamped official injury designations add predictive signal?

Pre-registered in ``results/experiments.jsonl`` as ``I37-prereg-injury`` before
this script ran.  The question is deliberately NOT "are injuries important" -
they obviously are - but "does the official pre-tip report tell a model
anything the prior-game roster does not already imply".

DATA
----
``Data/injury.sqlite`` (append-only capture of the NBA's own report PDFs) for
the 2021-22 season, backfilled date by date.  A game is covered when at least
one snapshot is labelled at least ``BUFFER_MINUTES`` before its tip.  The
report lists every game on the date, so a game with no designated player is a
covered game with no discount - not missing data.  Coverage is reported before
any score.

PROTOCOL
--------
One season, one chronological split: ``nba_data.walk_forward_split`` with 5
contiguous test blocks after the first 50% of the season.  Every arm and its
reference are scored on the same rows.  Three pre-registered arms:

``e14_plus_injury``
    The 43-column E14 frame plus the four injury-adjusted availability columns.
    Reference: E14.  This is the direct analogue of the issue-#36 measurement
    that established the unadjusted availability block.
``canonical_replace`` (primary)
    The canonical model with its four prior-game availability columns replaced
    by the injury-adjusted versions.  Reference: canonical.
``canonical_plus``
    The canonical model plus the four injury columns under their own names.
    Reference: canonical.

A positive delta is ``loss(reference) - loss(candidate)``; promotion needs
``|z| > 3`` (Bonferroni over the three arms).  Two diagnostics are printed
whether or not an arm moves: the unadjusted availability block re-measured on
this season (a positive control - if the known +0.0038 effect cannot be seen
here, this subset cannot see a smaller one), and a placebo that shuffles which
designated player receives which weight inside each game (20 permutations).
The placebo delta distribution shows how much of any gain is just "the column
is nonzero sometimes".

SENSITIVITY
-----------
The point-in-time gate is re-run with a 30-minute buffer instead of 15, so a
result cannot be an artefact of using a report whose PDF appeared minutes after
its printed label.

Run:  python scripts/issue37_injury_reports.py
"""
from __future__ import annotations

import argparse
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

from DataProviders import injury_features as inf      # noqa: E402
from DataProviders import nba_data as nd              # noqa: E402
from Models import logreg_rolling as lr               # noqa: E402
from Models import measurement as ms                  # noqa: E402

SEASON = "2021-22"
DB = ROOT / "Data" / "injury.sqlite"
JSONL = ROOT / "results" / "experiments.jsonl"
OUT_JSON = ROOT / "results" / "issue37_injury.json"
OUT_MD = ROOT / "results" / "issue37_injury.md"
N_BOOT = 2000
SEED = 42
BUFFER_MINUTES = 15.0
SENSITIVITY_BUFFER = 30.0
N_PLACEBO = 20
C = 1.0


def season_games(season: str = SEASON) -> pd.DataFrame:
    g = nd.build_games()
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"]).dt.normalize()
    s = g[g["SEASON"] == season].sort_values("GAME_DATE").reset_index(drop=True)
    if s.empty:
        raise SystemExit(f"no games for season {season}")
    return s


def inputs(games: pd.DataFrame, buffer_minutes: float = BUFFER_MINUTES):
    """Resolve the store against this season and return weights + coverage."""
    events, snaps = inf.load_events(DB)
    events = events[events["game_date"].isin(games["GAME_DATE"])].copy()
    if events.empty:
        raise SystemExit(
            "no injury events for the evaluation season; run the backfill first")
    resolved = inf.resolve_players(events)
    designations, coverage = inf.select_pre_tip(
        resolved, snaps, games, buffer_minutes=buffer_minutes)
    weights = inf.availability_weights(designations)
    return resolved, designations, coverage, weights


def oof(X: pd.DataFrame, features: list[str], games: pd.DataFrame) -> np.ndarray:
    y = games["y"].to_numpy(dtype=int)
    p = np.full(len(games), np.nan)
    for tr, te in nd.walk_forward_split(games, n_splits=5, initial_frac=0.5):
        model = lr.RollingLogreg(C=C, features=features).fit(X.iloc[tr], y[tr])
        p[te] = model.predict_proba(X.iloc[te])
    return p


def oof_e14(X: pd.DataFrame, games: pd.DataFrame) -> np.ndarray:
    p, _ = ms.oof_logistic(X.to_numpy(dtype=float),
                           games["y"].to_numpy(dtype=int), games)
    return p


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    m = nd.evaluate(np.asarray(y, dtype=int), np.asarray(p, dtype=float))
    return {k: (int(v) if k == "n" else float(v))
            for k, v in m.items() if k in ("n", "log_loss", "brier",
                                           "accuracy", "auc")}


def delta(y: np.ndarray, p_candidate: np.ndarray,
          p_reference: np.ndarray) -> dict:
    d = ms.delta_stats(y, p_candidate, p_reference,
                       n_boot=N_BOOT, seed=SEED)
    d["bootstrap_seed"] = SEED
    return d


def canonical_blocks(games: pd.DataFrame, weights: pd.DataFrame):
    """(X with the canonical block, X with the injury block, feature list)."""
    features = lr.resolve_features(lr.DEFAULT_MODELS["full"])
    X_ref = lr.load_design(games=games)
    X_inj = lr.load_design(games=games, availability_weights=weights)
    return X_ref, X_inj, features


def build_placebo(designations: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle PLAYER_ID inside each game, keeping the weight multiset.

    The join is by ``(GAME_ID, PLAYER_ID)``, so this asks the same game to
    discount a different set of its designated players.  It preserves how many
    nonzero discounts a game gets and how severe they are; it destroys only
    "this particular player is the one carrying the designation".
    """
    out = []
    for gid, part in designations.groupby("GAME_ID", sort=False):
        p = part.copy()
        p["PLAYER_ID"] = rng.permutation(p["PLAYER_ID"].to_numpy())
        out.append(p)
    return pd.concat(out, ignore_index=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--season", default=SEASON,
                   help="Season to score (default 2021-22, the registered primary)")
    args = p.parse_args(argv)
    season = args.season
    primary = season == SEASON
    out_json = OUT_JSON if primary else ROOT / "results" / (
        f"issue37_injury_{season}.json")
    out_md = OUT_MD if primary else ROOT / "results" / (
        f"issue37_injury_{season}.md")
    t0 = time.time()
    games = season_games(season)
    y = games["y"].to_numpy(dtype=int)
    resolved, designations, coverage, weights = inputs(games)
    diag = inf.diagnostics(resolved, designations, coverage, games)

    X_e14 = ms.build_e14_frame(games)
    X_e14_inj = pd.concat(
        [X_e14, nd.availability_features(games, weights=weights)], axis=1)

    X_ref, X_inj, features = canonical_blocks(games, weights)
    inj_cols = list(nd.availability_features(games).columns)
    X_plus = pd.concat(
        [X_ref, nd.availability_features(games, weights=weights)
         .rename(columns={c: c.replace("expected", "injury") for c in inj_cols})],
        axis=1)
    plus_features = features + [c for c in X_plus.columns
                                if c.startswith(("home_injury", "away_injury"))]

    X_ctrl = lr.load_design(games=games, with_availability=False)
    ctrl_features = [f for f in features if f not in lr.AVAILABILITY_COLUMNS]

    preds = {
        "e14": oof_e14(X_e14, games),
        "e14_plus_injury": oof(X_e14_inj, list(X_e14.columns) + inj_cols, games),
        "canonical": oof(X_ref, features, games),
        "canonical_without_availability": oof(X_ctrl, ctrl_features, games),
        "canonical_replace": oof(X_inj, features, games),
        "canonical_plus": oof(X_plus, plus_features, games),
    }
    mask = np.isfinite(preds["canonical"])
    for name, p in preds.items():
        if not np.isfinite(p[mask]).all():
            raise AssertionError(f"{name} has missing scored predictions")
    covered_lookup = coverage.set_index("GAME_ID")["has_snapshot"]
    covered = mask & games["GAME_ID"].map(covered_lookup).fillna(False).to_numpy(
        dtype=bool)

    arms = {
        "e14_plus_injury": ("e14_plus_injury", "e14"),
        "canonical_replace": ("canonical_replace", "canonical"),
        "canonical_plus": ("canonical_plus", "canonical"),
    }
    arm_rows = {}
    for arm, (cand, ref) in arms.items():
        p_c = np.full(len(games), np.nan)
        p_r = np.full(len(games), np.nan)
        p_c[mask] = preds[cand][mask]
        p_r[mask] = preds[ref][mask]
        arm_rows[arm] = {
            "candidate": cand,
            "reference": ref,
            "metrics_candidate": metrics(y[mask], preds[cand][mask]),
            "metrics_reference": metrics(y[mask], preds[ref][mask]),
            "delta": delta(y[mask], preds[cand][mask], preds[ref][mask]),
            "delta_covered_only": delta(y[covered], preds[cand][covered],
                                        preds[ref][covered]),
            "metrics_covered_only": metrics(y[covered], preds[cand][covered]),
            "per_fold": ms.per_fold_delta(y, p_c, p_r, games),
        }

    # Positive control: the already-recorded availability block, re-measured on
    # this season.  This is a diagnostic, never a promotion.
    control = {
        "candidate": "canonical",
        "reference": "canonical_without_availability",
        "metrics_candidate": metrics(y[mask], preds["canonical"][mask]),
        "metrics_reference": metrics(
            y[mask], preds["canonical_without_availability"][mask]),
        "delta": delta(y[mask], preds["canonical"][mask],
                       preds["canonical_without_availability"][mask]),
        "per_fold": ms.per_fold_delta(
            y, preds["canonical"], preds["canonical_without_availability"], games),
    }

    # Placebo: same games, same weights, different designated players.
    rng = np.random.default_rng(SEED)
    placebo = []
    p_ref = preds["canonical"][mask]
    for k in range(N_PLACEBO):
        shuffled = build_placebo(designations, rng)
        w = inf.availability_weights(shuffled)
        X_pl = lr.load_design(games=games, availability_weights=w)
        p_pl = oof(X_pl, features, games)
        placebo.append(float(
            ms.log_loss(y[mask], p_ref) - ms.log_loss(y[mask], p_pl[mask])))
    placebo = np.asarray(placebo)
    true_delta = arm_rows["canonical_replace"]["delta"][
        "delta_candidate_minus_reference"]
    perm_p = float((placebo >= true_delta).mean())

    # Sensitivity: a stricter point-in-time gate.
    _, des30, cov30, w30 = inputs(games, buffer_minutes=SENSITIVITY_BUFFER)
    X30 = lr.load_design(games=games, availability_weights=w30)
    p30 = oof(X30, features, games)
    sens = {
        "buffer_minutes": SENSITIVITY_BUFFER,
        "games_with_pre_tip_snapshot": int(cov30["has_snapshot"].sum()),
        "designations_applied": int(len(des30)),
        "delta": delta(y[mask], p30[mask], preds["canonical"][mask]),
        "per_fold": ms.per_fold_delta(y, p30, preds["canonical"], games),
    }

    promoted = [a for a, row in arm_rows.items()
                if row["delta"]["delta_candidate_minus_reference"] > 0
                and row["delta"]["abs_z"] > 3]
    best = max(arm_rows, key=lambda a: arm_rows[a]["delta"]
               ["delta_candidate_minus_reference"])
    best_row = arm_rows[best]
    verdict = (
        f"PROMOTED: {', '.join(promoted)} clear |z| > 3 with a positive delta."
        if promoted else
        "NULL: no injury arm clears its reference model at |z| > 3. The "
        f"strongest arm is {best} "
        f"({best_row['delta']['delta_candidate_minus_reference']:+.6f}, "
        f"|z| {best_row['delta']['abs_z']:.2f}; covered-only "
        f"{best_row['delta_covered_only']['delta_candidate_minus_reference']:+.6f}, "
        f"|z| {best_row['delta_covered_only']['abs_z']:.2f}). This is an "
        "absence of evidence, not evidence that the designations are "
        f"uninformative: the placebo spread is {placebo.std(ddof=1):.6f} nats "
        f"on {int(mask.sum())} scored games."
    )

    payload = {
        "issue": 37,
        "protocol_ref": "I37-prereg-injury",
        "generated_utc": ms.utc_now(),
        "runtime_seconds": round(time.time() - t0, 1),
        "season": season,
        "split": "nba_data.walk_forward_split(n_splits=5, initial_frac=0.5)",
        "n_games": int(len(games)),
        "n_scored": int(mask.sum()),
        "buffer_minutes": BUFFER_MINUTES,
        "coverage": diag,
        "status_weights": inf.STATUS_WEIGHTS,
        "arms": arm_rows,
        "availability_positive_control": control,
        "placebo": {
            "n_permutations": N_PLACEBO,
            "deltas": [float(x) for x in placebo],
            "mean": float(placebo.mean()),
            "sd": float(placebo.std(ddof=1)),
            "min": float(placebo.min()),
            "max": float(placebo.max()),
            "true_canonical_replace_delta": float(true_delta),
            "one_sided_permutation_p": perm_p,
            "interpretation": "fraction of within-game player shuffles that "
                              "match or beat the true assignment's delta",
        },
        "sensitivity_buffer_30": sens,
        "verdict": verdict,
    }
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ms.append_jsonl(JSONL, {
        "name": ("I37-run-injury-reports" if primary
                 else f"I37-run-injury-reports-{season}"),
        "protocol_ref": "I37-prereg-injury",
        "declared_utc": ms.utc_now(),
        "experiment_type": "official pre-tip injury designations (issue #37)",
        "season": season,
        "n_scored": int(mask.sum()),
        "coverage": diag,
        "arms": {a: {"delta": r["delta"],
                     "delta_covered_only": r["delta_covered_only"],
                     "per_fold": r["per_fold"]}
                 for a, r in arm_rows.items()},
        "availability_positive_control": control["delta"],
        "placebo_mean": float(placebo.mean()),
        "placebo_sd": float(placebo.std(ddof=1)),
        "placebo_permutation_p": perm_p,
        "sensitivity_buffer_30": sens["delta"],
        "verdict": verdict,
        "status": "executed",
    })

    L: list[str] = []
    a = L.append
    a("=" * 78)
    a(f"ISSUE 37 -- TIMESTAMPED INJURY REPORTS ({season})")
    a("=" * 78)
    a(f"games {len(games)}; scored {int(mask.sum())}; "
      f"pre-tip coverage {diag['games_with_pre_tip_snapshot']}/"
      f"{diag['games']} ({diag['coverage_share']:.1%})")
    a(f"events resolved {diag['events_resolved']}/{diag['events_stored']} "
      f"({diag['resolution_rate']:.1%}); ambiguous {diag['events_ambiguous']}")
    a(f"designations applied {diag['designations_applied']} across "
      f"{diag['games_with_designation']} games; on a prior-game squad "
      f"{diag['designations_on_prior_squad']} across "
      f"{diag['games_with_prior_squad_designation']} games")
    a(f"statuses on prior squad: {diag['status_on_prior_squad']}")
    a("")
    a(f"{'arm':<34} {'cand LL':>9} {'ref LL':>9} {'delta':>10} {'z':>7} "
      f"{'folds':>7} {'covered d':>10}")
    for arm, row in arm_rows.items():
        d = row["delta"]
        a(f"{arm:<34} {row['metrics_candidate']['log_loss']:>9.6f} "
          f"{row['metrics_reference']['log_loss']:>9.6f} "
          f"{d['delta_candidate_minus_reference']:>+10.6f} {d['z']:>7.2f} "
          f"{row['per_fold']['n_folds_candidate_better']}/"
          f"{row['per_fold']['n_folds']:>2} "
          f"{row['delta_covered_only']['delta_candidate_minus_reference']:>+10.6f}")
    a(f"covered-only n = {int(covered.sum())}")
    a("")
    d = control["delta"]
    a(f"positive control (availability block) "
      f"{d['delta_candidate_minus_reference']:>+10.6f} |z| {d['abs_z']:.2f}")
    a(f"placebo delta mean {placebo.mean():+.6f}, sd {placebo.std(ddof=1):.6f}, "
      f"range [{placebo.min():+.6f}, {placebo.max():+.6f}]; "
      f"true assignment beats {perm_p:.0%} of shuffles")
    d = sens["delta"]
    a(f"buffer 30 min: coverage {sens['games_with_pre_tip_snapshot']}, "
      f"delta {d['delta_candidate_minus_reference']:+.6f} |z| {d['abs_z']:.2f}")
    a("")
    a(verdict)
    out_md.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
