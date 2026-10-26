"""Issue #68: close #37's promotion path on a powered season holdout.

#37's full-frame criterion arm clears |z| 4.47 on the completed store but was
declared descriptive: promotion requires a season protocol, and the original
season protocol (2021-22 615-row, 2020-21 540-row) was underpowered.  This arm
runs the IDENTICAL criterion arms -- E14 + availability (baseline) vs
E14 + injury-weighted availability (candidate), 47-column frames, 15-minute
pre-tip gate -- on a FORWARD HOLD-OUT instead of the pooled walk-forward:
train = everything before 2024-10-22, test = the 2024-25 and 2025-26 seasons
(where the store is dense).  Pre-registered I68-prereg-injury-promotion.

DECISION: promote the injury feature iff loss(baseline) - loss(candidate) > 0
with |z| > 3 on 2,000 paired bootstrap resamples (seed 42) over the combined
two-season test.  The per-season cells are declared descriptive.

Run::

    OMP_NUM_THREADS=1 python scripts/issue68_injury_promotion.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from DataProviders import injury_features as inf     # noqa: E402
from DataProviders import nba_data as nd             # noqa: E402
from Models import measurement as ms                 # noqa: E402

OUT_JSON = ROOT / "results" / "issue68_injury_promotion.json"
OUT_MD = ROOT / "results" / "issue68_injury_promotion.md"
JSONL = ROOT / "results" / "experiments.jsonl"
DB = ROOT / "Data" / "injury.sqlite"

OPENER24 = pd.Timestamp("2024-10-22")
OPENER25 = pd.Timestamp("2025-10-21")
N_BOOT = 2000
SEED = 42
BUFFER_MINUTES = 15.0
# The #37 full-frame E14 walk-forward bar and its drift tolerance: the
# reproduction anchor for this script's frame construction.
RECORDED_E14_BAR = 0.618307119
DRIFT_TOLERANCE = 1e-6

_CORE = ("n", "log_loss", "brier", "accuracy", "auc")


def _pipe():
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("scale", StandardScaler()),
                     ("logit", LogisticRegression(C=1.0, max_iter=4000,
                                                  random_state=42))])


def metrics(y, p):
    out = nd.evaluate(y, p)
    return {k: (int(out[k]) if k == "n" else float(out[k])) for k in _CORE
            if k in out}


def delta(y, p_c, p_r):
    d = ms.delta_stats(y, p_c, p_r, n_boot=N_BOOT, seed=SEED)
    d["bootstrap_seed"] = SEED
    return d


def main() -> int:
    t0 = time.time()
    games = nd.build_games().reset_index(drop=True)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"]).dt.normalize()
    y = games["y"].to_numpy(dtype=int)
    dates = games["GAME_DATE"]

    tr = (dates < OPENER24).to_numpy()
    te_two = (dates >= OPENER24).to_numpy()
    te_2425 = ((dates >= OPENER24) & (dates < OPENER25)).to_numpy()
    te_2526 = (dates >= OPENER25).to_numpy()

    events, snaps = inf.load_events(DB)
    events = events[events["game_date"].isin(games["GAME_DATE"])].copy()
    resolved = inf.resolve_players(events)
    designations, coverage = inf.select_pre_tip(
        resolved, snaps, games, buffer_minutes=BUFFER_MINUTES)
    w_final = inf.availability_weights(designations)
    diag = inf.diagnostics(resolved, designations, coverage, games)
    diag["games_total"] = int(len(games))
    diag["coverage_share_of_all_games"] = (
        diag["games_with_pre_tip_snapshot"] / diag["games_total"])
    # The effective sample for the injury arm: test games where at least one
    # prior-squad member carries a weight other than 1.0.
    squad = inf.prior_squad_keys(games)
    changed = w_final.merge(squad, on=["GAME_ID", "PLAYER_ID"], how="inner")
    changed = changed[changed["weight"] != 1.0]
    changed_games = set(changed["GAME_ID"])
    diag["games_with_changed_weight"] = int(len(changed_games))
    diag["test_games_with_changed_weight"] = int(
        games.loc[te_two, "GAME_ID"].isin(changed_games).sum())

    X_e14 = ms.build_e14_frame(games).reset_index(drop=True)
    avail = nd.availability_features(games).reset_index(drop=True)
    inj = nd.availability_features(games, weights=w_final).reset_index(drop=True)
    X_avail = pd.concat([X_e14, avail], axis=1)
    X_inj = pd.concat([X_e14, inj.rename(
        columns={c: c.replace("expected", "injury") for c in avail.columns})],
        axis=1)

    # Reproduction anchor: re-run #37's E14 walk-forward on the identical frame
    # and compare with its recorded bar (tolerance 1e-6, the repo standard).
    p_e14_full, mask_e14 = ms.oof_logistic(X_e14, y, games)
    reproduced = float(ms.log_loss(y[mask_e14], p_e14_full[mask_e14]))
    drift = reproduced - RECORDED_E14_BAR
    reproduced_ok = abs(drift) <= DRIFT_TOLERANCE
    print(f"E14 reproduced {reproduced:.9f} (recorded {RECORDED_E14_BAR:.9f}, "
          f"drift {drift:+.2e}, ok={reproduced_ok})")

    def fit_predict(X, te):
        m = _pipe().fit(X.iloc[tr], y[tr])
        return m.predict_proba(X.iloc[te])[:, 1]

    def arm_row(te, label):
        p_av = fit_predict(X_avail, te)
        p_inj = fit_predict(X_inj, te)
        return {
            "label": label,
            "n": int(te.sum()),
            "availability_frame": metrics(y[te], p_av),
            "injury_frame": metrics(y[te], p_inj),
            "delta_injury_vs_availability": delta(y[te], p_inj, p_av),
        }

    combined = arm_row(te_two, "2024-25 + 2025-26")
    s2425 = arm_row(te_2425, "2024-25 only")
    s2526 = arm_row(te_2526, "2025-26 only")

    # positive control: availability vs E14 (the +0.0038 block) on the holdout
    p_ref = fit_predict(X_e14, te_two)
    p_ctrl = fit_predict(X_avail, te_two)
    d_ctrl = delta(y[te_two], p_ctrl, p_ref)

    # coverage note on the test rows
    designations_total = int(len(designations))

    d = combined["delta_injury_vs_availability"]
    point, az = d["delta_candidate_minus_reference"], d["abs_z"]
    if point > 0 and az > 3:
        verdict = (
            "PROMOTED: the injury multiplier clears |z| > 3 on the "
            f"last-two-seasons holdout ({point:+.6f} nats, |z| {az:.2f}, "
            f"CI [{d['ci_lo']:+.6f}, {d['ci_hi']:+.6f}]) -- the criterion "
            "reading survives a forward season holdout. Promotion-grade "
            "evidence for the feature."
        )
    else:
        verdict = (
            "NOT PROMOTED / NULL: the injury multiplier does not clear |z| > 3 "
            f"on the last-two-seasons holdout ({point:+.6f} nats, "
            f"CI [{d['ci_lo']:+.6f}, {d['ci_hi']:+.6f}], |z| {az:.2f}). "
            "The criterion reading does not transfer to the seasons a bettor "
            "could act on."
        )

    payload = {
        "issue": 68,
        "protocol_ref": "I68-prereg-injury-promotion",
        "generated_utc": ms.utc_now(),
        "runtime_seconds": round(time.time() - t0, 1),
        "n_train": int(tr.sum()),
        "n_test_two_seasons": int(te_two.sum()),
        "reproduction": {
            "e14_walk_forward_log_loss": reproduced,
            "recorded_bar": RECORDED_E14_BAR,
            "drift": drift,
            "within_tolerance": bool(reproduced_ok),
            "n_scored": int(mask_e14.sum()),
        },
        "coverage_note": {
            "designations_in_store": designations_total,
            "test_games": int(te_two.sum()),
            "diagnostics": diag,
        },
        "arms": {"combined": combined, "season_2024_25": s2425,
                 "season_2025_26": s2526},
        "positive_control_availability_vs_e14": d_ctrl,
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ms.append_jsonl(JSONL, {
        "name": "I68-run-injury-promotion",
        "protocol_ref": "I68-prereg-injury-promotion",
        "declared_utc": ms.utc_now(),
        "experiment_type": "issue #68: injury promotion, last-two-seasons holdout",
        "n_train": int(tr.sum()),
        "n_test": int(te_two.sum()),
        "reproduction": payload["reproduction"],
        "arms": payload["arms"],
        "positive_control": d_ctrl,
        "verdict": verdict,
        "status": "executed",
    })

    def fmt(r):
        d = r["delta_injury_vs_availability"]
        return (f"{r['label']:<18} n={r['n']:>5} "
                f"avail {r['availability_frame']['log_loss']:.5f} "
                f"injury {r['injury_frame']['log_loss']:.5f} "
                f"delta {d['delta_candidate_minus_reference']:+.6f} "
                f"(|z| {abs(float(d['abs_z'])):.2f})")

    L = ["=" * 78,
         "ISSUE 68 -- INJURY PROMOTION, LAST-TWO-SEASONS HOLDOUT",
         "=" * 78]
    a = L.append
    a(f"train {int(tr.sum())} rows (< {OPENER24.date()}); test {int(te_two.sum())} "
      f"rows (2024-25 + 2025-26)")
    a(f"E14 reproduction {reproduced:.9f} (recorded {RECORDED_E14_BAR:.9f}, "
      f"drift {drift:+.2e}, ok={reproduced_ok})")
    a("arms (E14+availability baseline vs E14+injury, 47 cols, 15-min gate):")
    a(fmt(combined))
    a(fmt(s2425))
    a(fmt(s2526))
    a(f"positive control (availability vs E14): "
      f"{d_ctrl['delta_candidate_minus_reference']:+.6f} "
      f"|z| {d_ctrl['abs_z']:.2f}")
    a("")
    a(verdict)
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())