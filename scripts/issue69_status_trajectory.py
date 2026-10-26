"""Issue #69: injury-report status trajectory -- does the intraday CHANGE add?

The #37 final-status block clears |z| 4.47 on the completed store, but reads
only the LAST pre-tip slot.  The store's two same-day slots (17:00 -> 20:00)
let this arm ask the question no public repo can: is the intraday CHANGE in a
player's designation informative, over and above the final status?

DECISION (pre-registered I69-prereg-status-trajectory): the 2025-26 holdout
(train = before 2025-10-21; test = the 1,225 games of 2025-26).  Baseline =
canonical 26 with the FINAL-designation weighted availability (the #37 frame,
re-fit on train).  Candidate = baseline + the five trajectory aggregates
(mean_delta, n_downgrade, n_upgrade, sum_abs_delta, max_downgrade) computed
over players with BOTH an early and a final label.  Promote iff
loss(baseline) - loss(candidate) > 0 with |z| > 3 (2,000 paired bootstrap,
seed 42).  Positive control: final-injury vs raw availability on the same
split, reported descriptively.

Run::

    OMP_NUM_THREADS=1 python scripts/issue69_status_trajectory.py
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

from DataProviders import injury_features as inf     # noqa: E402
from DataProviders import nba_data as nd             # noqa: E402
from Models import logreg_rolling as lr              # noqa: E402
from Models import measurement as ms                 # noqa: E402

OUT_JSON = ROOT / "results" / "issue69_status_trajectory.json"
OUT_MD = ROOT / "results" / "issue69_status_trajectory.md"
JSONL = ROOT / "results" / "experiments.jsonl"
DB = ROOT / "Data" / "injury.sqlite"

OPENER25 = pd.Timestamp("2025-10-21")
OPENER24 = pd.Timestamp("2024-10-22")
N_BOOT = 2000
SEED = 42

_CORE = ("n", "log_loss", "brier", "accuracy", "auc")


def metrics(y, p):
    out = nd.evaluate(y, p)
    return {k: (int(out[k]) if k == "n" else float(out[k])) if k in out
            else float("nan") for k in _CORE}


def delta(y, p_c, p_r):
    d = ms.delta_stats(y, p_c, p_r, n_boot=N_BOOT, seed=SEED)
    d["bootstrap_seed"] = SEED
    return d


def trajectory_block(final, early, games):
    """Per-game aggregates over designated players with BOTH labels."""
    m = final.merge(early, on=["GAME_ID", "PLAYER_ID"], how="inner")
    m["delta"] = m["weight"] - m["weight_early"]
    agg = m.groupby("GAME_ID").agg(
        mean_delta=("delta", "mean"),
        n_downgrade=("delta", lambda s: int((s < 0).sum())),
        n_upgrade=("delta", lambda s: int((s > 0).sum())),
        sum_abs_delta=("delta", lambda s: float(np.abs(s).sum())),
        max_downgrade=("delta", lambda s: float(s.min() if (s < 0).any() else 0.0)),
    )
    gid = games["GAME_ID"].astype(str)
    agg.index = agg.index.astype(str)
    block = agg.reindex(gid).fillna(0.0).reset_index(drop=True)
    return block.astype(float)


def main() -> int:
    t0 = time.time()
    games = nd.build_games().reset_index(drop=True)
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])
    y = games["y"].to_numpy(dtype=int)

    events, snaps = inf.load_events(DB)
    resolved = inf.resolve_players(events)
    final, _ = inf.select_pre_tip(resolved, snaps, games, buffer_minutes=15.0)
    early = inf.earliest_pre_tip_designation(resolved, snaps, games, buffer_minutes=15.0)
    w_final = inf.availability_weights(final)

    block = trajectory_block(final, early, games)

    dates = pd.to_datetime(games["GAME_DATE"])
    tr25 = (dates < OPENER25).to_numpy()
    te25 = (dates >= OPENER25).to_numpy()

    X_base = lr.load_design(games=games, availability_weights=w_final).reset_index(drop=True)
    X_cand = pd.concat([X_base, block], axis=1).reset_index(drop=True)
    X_raw = lr.load_design(games=games).reset_index(drop=True)   # weights=None

    Y_tr, Y_te = y[tr25], y[te25]
    feat = lr.resolve_features(lr.DEFAULT_MODELS["full"])
    ref = lr.RollingLogreg(C=1.0, features=feat).fit(X_base.iloc[tr25], Y_tr)
    p_ref = ref.predict_proba(X_base.iloc[te25])
    cand = lr.RollingLogreg(C=1.0, features=list(X_cand.columns)).fit(
        X_cand.iloc[tr25], Y_tr)
    p_cand = cand.predict_proba(X_cand.iloc[te25])

    ref_m = metrics(Y_te, p_ref)
    cand_m = metrics(Y_te, p_cand)
    d = delta(Y_te, p_cand, p_ref)

    # positive control: final-injury vs raw availability on the same split
    raw = lr.RollingLogreg(C=1.0, features=feat).fit(X_raw.iloc[tr25], Y_tr)
    p_raw = raw.predict_proba(X_raw.iloc[te25])
    d_ctrl = delta(Y_te, p_ref, p_raw)   # final-injury vs raw; positive = injury better

    # sensitivity: 2024-25 replication, top1-only, direction-only
    # (2024-25 replication: train < 2024-10-22, test = the 2024-25 season)
    tr24 = (dates < OPENER24).to_numpy()
    te24 = ((dates >= OPENER24) & (dates < OPENER25)).to_numpy()
    Xb24 = lr.load_design(games=games, availability_weights=w_final).reset_index(drop=True)
    ref24 = lr.RollingLogreg(C=1.0, features=feat).fit(Xb24.iloc[tr24], y[tr24])
    p_ref24 = ref24.predict_proba(Xb24.iloc[te24])
    cand24 = lr.RollingLogreg(C=1.0, features=list(X_cand.columns)).fit(
        X_cand.iloc[tr24], y[tr24])
    p_cand24 = cand24.predict_proba(X_cand.iloc[te24])
    d24 = delta(y[te24], p_cand24, p_ref24)

    b_top1 = pd.concat([X_base, block[["max_downgrade"]]], axis=1).reset_index(drop=True)
    m_t1 = lr.RollingLogreg(C=1.0, features=list(b_top1.columns)).fit(
        b_top1.iloc[tr25], Y_tr)
    p_t1 = m_t1.predict_proba(b_top1.iloc[te25])
    d_t1 = delta(Y_te, p_t1, p_ref)

    b_dir = pd.concat([X_base, block[["n_downgrade", "n_upgrade"]]], axis=1).reset_index(drop=True)
    m_do = lr.RollingLogreg(C=1.0, features=list(b_dir.columns)).fit(
        b_dir.iloc[tr25], Y_tr)
    p_do = m_do.predict_proba(b_dir.iloc[te25])
    d_do = delta(Y_te, p_do, p_ref)

    point, az = d["delta_candidate_minus_reference"], d["abs_z"]
    if point > 0 and az > 3:
        verdict = (f"TRAJECTORY CLEARS THE BAR on 2025-26: +{point:.6f} nats, "
                   f"|z| {az:.2f} -- the intraday change adds information "
                   "over the final status.")
    else:
        verdict = (
            f"NULL on 2025-26: trajectory {point:+.6f} nats "
            f"(CI [{d['ci_lo']:+.6f}, {d['ci_hi']:+.6f}], |z| {az:.2f}); "
            "the intraday change is inside the price (or irrelevant). "
            f"Positive control (final-injury vs raw) {d_ctrl['delta_candidate_minus_reference']:+.6f} "
            f"|z| {d_ctrl['abs_z']:.2f}."
        )

    payload = {
        "issue": 69,
        "protocol_ref": "I69-prereg-status-trajectory",
        "generated_utc": ms.utc_now(),
        "runtime_seconds": round(time.time() - t0, 1),
        "n_test_2025_26": int(te25.sum()),
        "trajectory_block_columns": list(block.columns),
        "n_paired_players": int(len(final.merge(
            early, on=["GAME_ID", "PLAYER_ID"], how="inner"))),
        "metrics": {"baseline_final_status": ref_m,
                    "candidate_plus_trajectory": cand_m},
        "delta_trajectory": d,
        "positive_control_final_vs_raw": d_ctrl,
        "sensitivity": {"replication_2024_25": d24,
                        "top1_downgrade_only": d_t1,
                        "direction_only": d_do},
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ms.append_jsonl(JSONL, {
        "name": "I69-run-status-trajectory",
        "protocol_ref": "I69-prereg-status-trajectory",
        "declared_utc": ms.utc_now(),
        "experiment_type": "issue #69: injury status trajectory, 2025-26 holdout",
        "n_test": int(te25.sum()),
        "metrics": payload["metrics"],
        "delta_trajectory": d,
        "positive_control": d_ctrl,
        "sensitivity": payload["sensitivity"],
        "verdict": verdict,
        "status": "executed",
    })

    L = ["=" * 78,
         "ISSUE 69 -- INJURY STATUS TRAJECTORY, 2025-26 HOLDOUT",
         "=" * 78]
    a = L.append
    a(f"test {int(te25.sum())} rows (2025-26); block columns {list(block.columns)}")
    a(f"baseline (final status)    {ref_m['log_loss']:.6f}")
    a(f"candidate (+ trajectory)   {cand_m['log_loss']:.6f}")
    a(f"delta {point:+.6f} CI [{d['ci_lo']:+.6f}, {d['ci_hi']:+.6f}] "
      f"|z| {az:.2f}")
    a(f"positive control (final vs raw) {d_ctrl['delta_candidate_minus_reference']:+.6f} "
      f"|z| {d_ctrl['abs_z']:.2f}")
    a(f"2024-25 replication   {d24['delta_candidate_minus_reference']:+.6f} "
      f"|z| {d24['abs_z']:.2f}")
    a(f"top1 downgrade only    {d_t1['delta_candidate_minus_reference']:+.6f} "
      f"|z| {d_t1['abs_z']:.2f}")
    a(f"direction only         {d_do['delta_candidate_minus_reference']:+.6f} "
      f"|z| {d_do['abs_z']:.2f}")
    a("")
    a(verdict)
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())