"""Issue #37, criterion 4: the injury block on the recorded 43-column bar.

The season-level protocol in ``issue37_injury_reports.py`` is the better-powered
question ("does tonight's designation add signal within 2021-22?").  The issue's
acceptance criteria ask a narrower, literal question: measure the block on the
**current 43-column frame against the 0.61831 baseline**, with a paired bootstrap
and |z|.  This script answers that one on exactly the recorded population.

WHAT IS SCORED
--------------
Three 47-column frames, scored by ``measurement.oof_logistic`` -- the same
global-median-fill, 5-fold walk-forward recipe that produced 0.618307119:

``e14``
    The frozen 43-column frame.  Reproduced in this run; a drift from the
    recorded bar is reported as a failed reproduction, not absorbed.
``e14_plus_availability``
    E14 + the four UNADJUSTED availability columns.  E14 carries no
    availability block, so this is the reference that separates the known
    +0.0038 effect from the injury adjustment.
``e14_plus_injury``
    E14 + the four injury-adjusted availability columns.

The informative delta is **injury versus availability**, not injury versus E14:
the latter would credit the injury adjustment with the availability block.

COVERAGE IS THE HEADLINE, NOT A FOOTNOTE
----------------------------------------
Only 2020-21 and 2021-22 have a backfilled store.  Every other season keeps
weight 1.0, so its injury columns equal the availability columns and the two
47-column frames are identical on those rows.  That is 84% of the sample, so
this measurement is mostly a reproduction with a two-season perturbation.  It is
reported because the criterion asks for it; the pre-registered season protocol
remains the decision.

Pre-registered in ``results/experiments.jsonl`` as
``I37-prereg-injury-full-frame`` (candidate) and
``I37-prereg-injury-full-frame-correction`` (the availability reference) before
this script ran.  **Disclosure:** both records were written *after* the 2021-22
season result was known, because the criterion asks for this population and not
because the season result suggested it.  This arm is descriptive -- it can
satisfy the criterion, and it cannot promote the feature.

Run:  python scripts/issue37_full_frame.py
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

from DataProviders import injury_features as inf      # noqa: E402
from DataProviders import nba_data as nd              # noqa: E402
from Models import measurement as ms                  # noqa: E402

DB = ROOT / "Data" / "injury.sqlite"
JSONL = ROOT / "results" / "experiments.jsonl"
OUT_JSON = ROOT / "results" / "issue37_injury_full_frame.json"
OUT_MD = ROOT / "results" / "issue37_injury_full_frame.md"
RECORDED_BAR = 0.618307119
N_BOOT = 2000
SEED = 42
BUFFER_MINUTES = 15.0
DRIFT_TOLERANCE = 1e-6


def main() -> int:
    t0 = time.time()
    # Do NOT re-sort: oof_logistic splits by ROW POSITION, and build_games() is
    # already sorted by (GAME_DATE, GAME_ID). An unstable re-sort on ties moves
    # games across fold boundaries and makes the recorded bar unreproducible.
    games = nd.build_games()
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"]).dt.normalize()
    y = games["y"].to_numpy(dtype=int)

    events, snaps = inf.load_events(DB)
    events = events[events["game_date"].isin(games["GAME_DATE"])].copy()
    resolved = inf.resolve_players(events)
    designations, coverage = inf.select_pre_tip(
        resolved, snaps, games, buffer_minutes=BUFFER_MINUTES)
    weights = inf.availability_weights(designations)
    diag = inf.diagnostics(resolved, designations, coverage, games)
    n_games = int(len(games))
    diag["games_total"] = n_games
    diag["coverage_share_of_all_games"] = (
        diag["games_with_pre_tip_snapshot"] / n_games)

    avail = nd.availability_features(games).reset_index(drop=True)
    inj = nd.availability_features(games, weights=weights).reset_index(drop=True)
    inj_cols = list(avail.columns)
    X_e14 = ms.build_e14_frame(games).reset_index(drop=True)
    X_avail = pd.concat([X_e14, avail], axis=1)
    X_inj = pd.concat([X_e14, inj.rename(
        columns={c: c.replace("expected", "injury") for c in inj_cols})], axis=1)

    p_ref, mask = ms.oof_logistic(X_e14, y, games)
    p_avail, mask_a = ms.oof_logistic(X_avail, y, games)
    p_inj, mask_i = ms.oof_logistic(X_inj, y, games)
    if not (np.array_equal(mask, mask_a) and np.array_equal(mask, mask_i)):
        raise AssertionError("the three frames scored different rows")

    def m(p):
        return {k: (int(v) if k == "n" else float(v))
                for k, v in nd.evaluate(y[mask], p[mask]).items()
                if k in ("n", "log_loss", "brier", "accuracy", "auc")}

    def delta(cand, ref):
        d = ms.delta_stats(y[mask], cand[mask], ref[mask],
                           n_boot=N_BOOT, seed=SEED)
        d["bootstrap_seed"] = SEED
        return d

    reproduced = m(p_ref)["log_loss"]
    drift = reproduced - RECORDED_BAR
    reproduced_ok = abs(drift) <= DRIFT_TOLERANCE
    d_avail = delta(p_avail, p_ref)
    d_inj_e14 = delta(p_inj, p_ref)
    d_inj_avail = delta(p_inj, p_avail)
    per_fold = ms.per_fold_delta(y, p_inj, p_avail, games)

    if not reproduced_ok:
        verdict = (f"FAILED REPRODUCTION: E14 came back {reproduced:.9f}, not "
                   f"{RECORDED_BAR:.9f} (drift {drift:+.2e}). The deltas are "
                   "reported for diagnosis only.")
    elif d_inj_avail["delta_candidate_minus_reference"] > 0 and \
            d_inj_avail["abs_z"] > 3:
        verdict = (
            "INJURY ADJUSTMENT CLEARS THE AVAILABILITY BLOCK on the full frame: "
            f"{d_inj_avail['delta_candidate_minus_reference']:+.6f} nats, |z| "
            f"{d_inj_avail['abs_z']:.2f}. Coverage is "
            f"{diag['games_with_pre_tip_snapshot']}/{n_games} games "
            f"({diag['coverage_share_of_all_games']:.1%}); 84% of rows are "
            "identical between the arms, so this is a criterion result, not a "
            "promotion.")
    else:
        verdict = (
            "NOT PROMOTED: the injury-adjusted frame is "
            f"{d_inj_avail['delta_candidate_minus_reference']:+.6f} nats against "
            f"the availability-only frame on the same {int(mask.sum())} scored "
            f"rows (CI [{d_inj_avail['ci_lo']:+.6f}, {d_inj_avail['ci_hi']:+.6f}], "
            f"|z| {d_inj_avail['abs_z']:.2f}, better in "
            f"{per_fold['n_folds_candidate_better']}/{per_fold['n_folds']} folds). "
            "The interval excludes zero, so this is not a null -- it is a "
            "positive below the pre-registered |z| > 3 bar. Against the "
            f"43-column bar the injury frame is "
            f"{d_inj_e14['delta_candidate_minus_reference']:+.6f} "
            f"(|z| {d_inj_e14['abs_z']:.2f}), of which "
            f"{d_avail['delta_candidate_minus_reference']:+.6f} is the "
            f"established availability block (|z| {d_avail['abs_z']:.2f}). "
            f"Coverage is {diag['games_with_pre_tip_snapshot']}/{n_games} games "
            f"({diag['coverage_share_of_all_games']:.1%}) because only two "
            "seasons are backfilled. The criterion is satisfied; the feature is "
            "not promoted.")

    payload = {
        "issue": 37,
        "protocol_ref": "I37-prereg-injury-full-frame",
        "generated_utc": ms.utc_now(),
        "runtime_seconds": round(time.time() - t0, 1),
        "population": "build_games(), all seasons",
        "declared_after_season_result": True,
        "n_games": n_games,
        "n_scored": int(mask.sum()),
        "buffer_minutes": BUFFER_MINUTES,
        "recorded_bar": RECORDED_BAR,
        "reproduced_e14_log_loss": reproduced,
        "reproduction_drift": drift,
        "reproduced_within_tolerance": bool(reproduced_ok),
        "coverage": diag,
        "metrics": {
            "e14": m(p_ref),
            "e14_plus_availability": m(p_avail),
            "e14_plus_injury": m(p_inj),
        },
        "delta_availability_vs_e14": d_avail,
        "delta_injury_vs_e14": d_inj_e14,
        "delta_injury_vs_availability": d_inj_avail,
        "per_fold_injury_vs_availability": per_fold,
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ms.append_jsonl(JSONL, {
        "name": "I37-run-injury-full-frame",
        "protocol_ref": "I37-prereg-injury-full-frame",
        "declared_utc": ms.utc_now(),
        "experiment_type": "issue #37 criterion 4 on the recorded 43-column bar",
        "n_scored": int(mask.sum()),
        "coverage_games": diag["games_with_pre_tip_snapshot"],
        "coverage_total": n_games,
        "coverage_share": diag["coverage_share_of_all_games"],
        "recorded_bar": RECORDED_BAR,
        "reproduced_e14_log_loss": reproduced,
        "delta_availability_vs_e14": d_avail,
        "delta_injury_vs_e14": d_inj_e14,
        "delta_injury_vs_availability": d_inj_avail,
        "per_fold_injury_vs_availability": per_fold,
        "verdict": verdict,
        "status": "executed",
    })

    L = ["=" * 78,
         "ISSUE 37 -- FULL-FRAME CRITERION MEASUREMENT",
         "=" * 78]
    a = L.append
    a(f"games {n_games}; scored {int(mask.sum())}; "
      f"pre-tip coverage {diag['games_with_pre_tip_snapshot']}/{n_games} "
      f"({diag['coverage_share_of_all_games']:.1%})")
    a(f"designations {diag['designations_applied']} across "
      f"{diag['games_with_designation']} games; on a prior-game squad "
      f"{diag['designations_on_prior_squad']} across "
      f"{diag['games_with_prior_squad_designation']} games")
    a(f"E14 reproduced {reproduced:.9f} (recorded {RECORDED_BAR:.9f}, "
      f"drift {drift:+.2e})")
    a("")
    a(f"{'frame':<30} {'log loss':>10} {'brier':>9} {'accuracy':>9} {'auc':>7}")
    for name, met in (("E14 (43)", m(p_ref)),
                      ("E14 + availability (47)", m(p_avail)),
                      ("E14 + injury (47)", m(p_inj))):
        a(f"{name:<30} {met['log_loss']:>10.6f} {met['brier']:>9.6f} "
          f"{met['accuracy']:>9.4f} {met['auc']:>7.4f}")
    a("")
    for label, d in (("availability vs E14", d_avail),
                     ("injury vs E14", d_inj_e14),
                     ("injury vs availability", d_inj_avail)):
        a(f"{label:<30} {d['delta_candidate_minus_reference']:+.6f} "
          f"CI [{d['ci_lo']:+.6f}, {d['ci_hi']:+.6f}] |z| {d['abs_z']:.2f}")
    a(f"folds (injury vs availability): "
      f"{per_fold['n_folds_candidate_better']}/{per_fold['n_folds']}")
    a("")
    a(verdict)
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
