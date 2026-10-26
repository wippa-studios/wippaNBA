"""Issue #71: is the availability block at its optimum, or on a plateau?

Availability (+0.00390 nats, |z| 4.78) is the project's foundational non-null,
and its construction inherits choices made when it was first written: a 15-game
player halflife, the previous game's squad, and a sum over every squad member.
This script asks whether that point is a grid interior or a boundary.

GRID (pre-registered I71-prereg-availability-sensitivity): halflife
{10, 15, 20, 30} x squad {prior_game, season_average} x minutes-count
{all, 5, 7} = 24 cells.  Each cell is the canonical 22-column base plus the
four rebuilt availability columns, i.e. the canonical 26 with the block
replaced.  The recorded configuration (15, prior_game, all) is one cell and
must reproduce the recorded canonical bar exactly.

DECISION: ROBUST / INTERIOR iff (a) no cell beats the recorded configuration at
|z| > 3 on the full frame, and (b) every cell's own availability effect is
positive with |z| > 3.  Otherwise FRAGILE / BOUNDARY, with the failing cells
named.  No new value is promoted; the canonical model is unchanged.

Run::

    OMP_NUM_THREADS=1 python scripts/issue71_availability_sensitivity.py
"""
from __future__ import annotations

import importlib.util
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

from DataProviders import nba_data as nd        # noqa: E402
from Models import logreg_rolling as lr         # noqa: E402
from Models import measurement as ms            # noqa: E402

OUT_JSON = ROOT / "results" / "issue71_availability_sensitivity.json"
OUT_MD = ROOT / "results" / "issue71_availability_sensitivity.md"
JSONL = ROOT / "results" / "experiments.jsonl"

N_BOOT = 2000
SEED = 42
RECORDED_CANONICAL = 0.615663649
RECORDED_CLOSE = 0.62034
DRIFT_TOLERANCE = 1e-6
#: The close stack's recorded number predates the current odds snapshot; the
#: refresh drift documented there is ~1e-5, so this is a reporting tolerance,
#: not a decision gate.
CLOSE_DRIFT_TOLERANCE = 1e-4

HALFLIFES = (10.0, 15.0, 20.0, 30.0)
SQUADS = ("prior_game", "season_average")
COUNTS = ("all", 5, 7)

_CORE = ("n", "log_loss", "brier", "accuracy", "auc")


def _closing_line():
    spec = importlib.util.spec_from_file_location(
        "closing_line", ROOT / "scripts" / "closing_line.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def metrics(y, p):
    out = nd.evaluate(y, p)
    return {k: (int(out[k]) if k == "n" else float(out[k])) for k in _CORE
            if k in out}


def delta(y, p_c, p_r):
    d = ms.delta_stats(y, p_c, p_r, n_boot=N_BOOT, seed=SEED)
    d["bootstrap_seed"] = SEED
    return d


def rolling_probs(X, y, games, features):
    p = np.full(len(y), np.nan)
    for tr, te in nd.walk_forward_split(games, n_splits=5):
        m = lr.RollingLogreg(C=1.0, features=features).fit(X.iloc[tr], y[tr])
        p[te] = m.predict_proba(X.iloc[te])
    return p


def close_probs(X, y, games, features):
    """The closing_line.py estimator, on the same folds."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    p = np.full(len(y), np.nan)
    for tr, te in nd.walk_forward_split(games, n_splits=5):
        pipe = Pipeline([("i", SimpleImputer(strategy="median")),
                         ("s", StandardScaler()),
                         ("c", LogisticRegression(C=1.0, max_iter=300,
                                                  random_state=SEED))])
        pipe.fit(X.iloc[tr], y[tr])
        p[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return p


def cell_name(h, squad, count):
    return f"hl{int(h)}_{squad}_top{count}"


def main() -> int:
    t0 = time.time()
    games = nd.build_games().reset_index(drop=True)
    y = games["y"].to_numpy(dtype=int)
    features_full = lr.resolve_features(lr.DEFAULT_MODELS["full"])
    avail_cols = list(lr.resolve_features("availability"))
    if len(avail_cols) != 4:
        raise AssertionError("availability must be four columns")

    # The canonical reproduction anchor.
    _, p_canon, mask = ms.canonical_oof(games=games)
    reproduced = float(ms.log_loss(y[mask], p_canon[mask]))
    drift = reproduced - RECORDED_CANONICAL
    reproduced_ok = abs(drift) <= DRIFT_TOLERANCE
    print(f"canonical reproduced {reproduced:.9f} (recorded "
          f"{RECORDED_CANONICAL:.9f}, drift {drift:+.2e}, "
          f"ok={reproduced_ok})")

    X_design = lr.load_design(with_availability=False,
                              games=games).reset_index(drop=True)
    base_features = [f for f in features_full if f not in avail_cols]
    if len(base_features) != 22:
        raise AssertionError("canonical base must be 22 features")
    X_base = X_design[base_features].reset_index(drop=True)
    X_default = lr.load_design(with_availability=True,
                               games=games)[features_full].reset_index(drop=True)
    if not list(X_default.columns[-4:]) == avail_cols:
        raise AssertionError("canonical availability columns moved")
    if len(X_base.columns) != 22 or len(X_default.columns) != 26:
        raise AssertionError("expected 22 -> 26 columns")

    # The close arm is a different, WIDER frame: closing_line.py scores
    # load_design's full output (30 base columns + the 4 availability
    # columns), not the 26-column canonical subset.  That is what the recorded
    # 0.62034 is, so reproducing it requires the same frame.
    X_close_base = X_design.copy()
    X_close_default = lr.load_design(
        with_availability=True, games=games).reset_index(drop=True)
    close_base_features = list(X_close_base.columns)
    close_features = list(X_close_default.columns)
    if len(close_base_features) != 30 or len(close_features) != 34:
        raise AssertionError("close frame must be 30 -> 34 columns")
    if close_features[:30] != close_base_features:
        raise AssertionError("close availability columns moved")
    if close_features[-4:] != avail_cols:
        raise AssertionError("close availability columns are not the block")

    p_base = rolling_probs(X_base, y, games, base_features)

    # The close comparison, on the recorded market sample: games with BOTH a
    # closing line and an out-of-fold prediction (the walk-forward's first half
    # is training-only, so early odds rows carry no prediction).
    cl = _closing_line()
    odds = cl.load_odds()
    joined = cl.join_odds(games, odds)
    row_of = {gid: i for i, gid in enumerate(games["GAME_ID"].to_numpy())}
    idx_all = joined["GAME_ID"].map(row_of).to_numpy()
    p_base_close_all = close_probs(X_close_base, y, games, close_base_features)
    keep = (np.isfinite(p_base_close_all[idx_all])
            & np.isfinite(joined["p_market"].to_numpy()))
    idx = idx_all[keep]
    yy = y[idx]
    p_market = joined["p_market"].to_numpy()[keep]
    p_canon_close = close_probs(X_close_default, y, games, close_features)[idx]
    p_base_close = p_base_close_all[idx]
    if len(yy) != 5788:
        raise AssertionError(f"expected 5,788 close games, got {len(yy)}")
    default_close_m = metrics(yy, p_canon_close)
    close_drift = default_close_m["log_loss"] - RECORDED_CLOSE
    close_ok = abs(close_drift) <= CLOSE_DRIFT_TOLERANCE
    print(f"close sample {len(yy):,} games; default close log loss "
          f"{default_close_m['log_loss']:.5f} (recorded {RECORDED_CLOSE:.5f}, "
          f"drift {close_drift:+.2e}, ok={close_ok})")

    cells = []
    p_rl_by_name = {}
    p_cl_by_name = {}
    for h in HALFLIFES:
        for squad in SQUADS:
            for count in COUNTS:
                top_n = None if count == "all" else int(count)
                block = nd.availability_features(
                    games, halflife=h, squad_mode=squad,
                    top_n=top_n).reset_index(drop=True)
                if list(block.columns) != avail_cols:
                    raise AssertionError(
                        f"block columns changed at {h}/{squad}/{count}")
                X = pd.concat([X_base, block], axis=1)
                if len(X.columns) != 26:
                    raise AssertionError("cell frame must be 26 columns")
                Xc = pd.concat([X_close_base, block], axis=1)
                if len(Xc.columns) != 34:
                    raise AssertionError("close cell frame must be 34 columns")
                name = cell_name(h, squad, count)
                if (h, squad, count) == (15.0, "prior_game", "all"):
                    pd.testing.assert_frame_equal(
                        X.reset_index(drop=True),
                        X_default.reset_index(drop=True))
                    pd.testing.assert_frame_equal(
                        Xc.reset_index(drop=True),
                        X_close_default.reset_index(drop=True))
                p_rl = rolling_probs(X, y, games, features_full)
                if not np.array_equal(np.isfinite(p_rl), np.isfinite(p_canon)):
                    raise AssertionError(f"{name} scored different rows")
                p_cl = close_probs(Xc, y, games, close_features)
                p_rl_by_name[name] = p_rl
                p_cl_by_name[name] = p_cl

                d_canon = delta(y[mask], p_rl[mask], p_canon[mask])
                d_base = delta(y[mask], p_rl[mask], p_base[mask])
                pcell_close = p_cl[idx]
                cells.append({
                    "name": name,
                    "halflife": h,
                    "squad_mode": squad,
                    "top_n": count,
                    "full_log_loss": float(nd.evaluate(y[mask],
                                                       p_rl[mask])["log_loss"]),
                    "delta_vs_canonical": d_canon,
                    "availability_effect": d_base,
                    "close_log_loss": float(nd.evaluate(
                        yy, pcell_close)["log_loss"]),
                    "close_metrics": metrics(yy, pcell_close),
                    "close_delta_vs_market": delta(yy, pcell_close, p_market),
                    "close_delta_vs_canonical": delta(yy, pcell_close,
                                                      p_canon_close),
                    "close_availability_effect": delta(yy, pcell_close,
                                                       p_base_close),
                })
                print(f"  {name:<26} full {cells[-1]['full_log_loss']:.6f} "
                      f"vs canon {d_canon['delta_candidate_minus_reference']:+.6f} "
                      f"(|z| {d_canon['abs_z']:.2f})  avail "
                      f"{d_base['delta_candidate_minus_reference']:+.6f} "
                      f"(|z| {d_base['abs_z']:.2f})")

    # The recorded cell must be numerically identical to the canonical frame.
    ref = next(c for c in cells if c["name"] == "hl15_prior_game_topall")
    if abs(ref["delta_vs_canonical"]["delta_candidate_minus_reference"]) > 1e-12:
        raise AssertionError("the recorded cell did not reproduce the canonical")
    if not np.array_equal(np.isfinite(p_rl_by_name[ref["name"]]),
                          np.isfinite(p_canon)):
        raise AssertionError("the recorded cell scored different rows")

    # Pre-registered decision.
    beating = [c["name"] for c in cells
               if c["delta_vs_canonical"]["delta_candidate_minus_reference"] > 0
               and c["delta_vs_canonical"]["abs_z"] > 3]
    failing_avail = [c["name"] for c in cells
                     if not (c["availability_effect"]
                             ["delta_candidate_minus_reference"] > 0
                             and c["availability_effect"]["abs_z"] > 3)]
    # Descriptive: cells that are reliably WORSE than the recorded point.  The
    # decision rule only asks whether anything beats it; this names the other
    # direction so a degraded axis is visible rather than silently ignored.
    worse = [c["name"] for c in cells
             if c["delta_vs_canonical"]["delta_candidate_minus_reference"] < 0
             and c["delta_vs_canonical"]["abs_z"] > 3]
    failing_squads = sorted({c["squad_mode"] for c in cells
                             if c["name"] in failing_avail})
    if not beating and not failing_avail:
        verdict = ("INTERIOR / ROBUST: no cell beats the recorded configuration "
                   "at |z| > 3, and the availability effect clears |z| > 3 in "
                   "all 24 cells. The recorded point is a plateau, not an edge.")
    elif beating and not failing_avail:
        verdict = ("BOUNDARY / OPTIMISABLE: the recorded configuration is not "
                   "the best cell -- " + ", ".join(beating) + " beat(s) it at "
                   "|z| > 3. The availability effect itself survives every "
                   "cell. No new value is promoted.")
    elif failing_avail:
        detail = ""
        if set(failing_squads) == {"season_average"}:
            detail = (" Every failing cell is on the season_average roster axis "
                      "and none is on prior_game, so halflife and minutes-count "
                      "read as a plateau while the membership convention is "
                      "load-bearing.")
        verdict = ("FRAGILE / BOUNDARY: the availability effect loses |z| > 3 "
                   "in " + ", ".join(failing_avail) + "." + detail +
                   " The recorded point is an edge on the axis that fails; no "
                   "new value is promoted.")
    else:
        verdict = "UNEXPECTED combination of decision flags"

    # One-axis reads at the recorded settings for the other two knobs.  Both
    # the cell-vs-canonical delta and the block's own effect are reported: a
    # cell can be worse overall while its availability block still works.
    def _axis(c: dict, value) -> dict:
        return {
            "delta_vs_canonical": c["delta_vs_canonical"]
            ["delta_candidate_minus_reference"],
            "delta_vs_canonical_abs_z": c["delta_vs_canonical"]["abs_z"],
            "availability_effect": c["availability_effect"]
            ["delta_candidate_minus_reference"],
            "availability_abs_z": c["availability_effect"]["abs_z"],
            "close_log_loss": c["close_log_loss"],
            "value": value,
        }

    axis: dict[str, dict] = {}
    axis["halflife_at_prior_all"] = {
        str(c["halflife"]): _axis(c, c["halflife"]) for c in cells
        if c["squad_mode"] == "prior_game" and c["top_n"] == "all"}
    axis["squad_at_hl15_all"] = {
        c["squad_mode"]: _axis(c, c["squad_mode"]) for c in cells
        if c["halflife"] == 15.0 and c["top_n"] == "all"}
    axis["top_n_at_hl15_prior"] = {
        str(c["top_n"]): _axis(c, c["top_n"]) for c in cells
        if c["halflife"] == 15.0 and c["squad_mode"] == "prior_game"}

    payload = {
        "issue": 71,
        "protocol_ref": "I71-prereg-availability-sensitivity",
        "generated_utc": ms.utc_now(),
        "runtime_seconds": round(time.time() - t0, 1),
        "grid": {"halflife": list(HALFLIFES), "squad_mode": list(SQUADS),
                 "minutes_count": list(COUNTS)},
        "reproduction": {
            "canonical_log_loss": reproduced,
            "recorded": RECORDED_CANONICAL,
            "drift": drift,
            "within_tolerance": bool(reproduced_ok),
            "close_log_loss": default_close_m["log_loss"],
            "close_recorded": RECORDED_CLOSE,
            "close_drift": close_drift,
            "close_within_tolerance": bool(close_ok),
            "close_tolerance": CLOSE_DRIFT_TOLERANCE,
            "close_games": int(len(yy)),
        },
        "cells": cells,
        "axis_reads": axis,
        "beating_recorded_at_z3": beating,
        "worse_than_recorded_at_z3": worse,
        "availability_effect_fails_z3": failing_avail,
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ms.append_jsonl(JSONL, {
        "name": "I71-run-availability-sensitivity",
        "protocol_ref": "I71-prereg-availability-sensitivity",
        "declared_utc": ms.utc_now(),
        "experiment_type": ("issue #71: availability construction grid, full "
                            "frame and at the close"),
        "grid": payload["grid"],
        "reproduction": payload["reproduction"],
        "cells": [{k: c[k] for k in ("name", "halflife", "squad_mode",
                                     "top_n", "full_log_loss",
                                     "delta_vs_canonical",
                                     "availability_effect", "close_log_loss",
                                     "close_delta_vs_market")}
                  for c in cells],
        "beating_recorded_at_z3": beating,
        "worse_than_recorded_at_z3": worse,
        "availability_effect_fails_z3": failing_avail,
        "verdict": verdict,
        "status": "executed",
    })

    L = ["=" * 78,
         "ISSUE 71 -- AVAILABILITY SENSITIVITY GRID",
         "=" * 78]
    a = L.append
    a(f"canonical reproduction {reproduced:.9f} (recorded "
      f"{RECORDED_CANONICAL:.9f}, drift {drift:+.2e}, ok={reproduced_ok})")
    a(f"close sample {len(yy):,} games; default cell close "
      f"{default_close_m['log_loss']:.5f} (recorded {RECORDED_CLOSE:.5f})")
    a("")
    a(f"{'cell':<26} {'full LL':>9} {'vs canon':>10} {'|z|':>6} "
      f"{'avail eff':>10} {'|z|':>6} {'close LL':>9}")
    for c in cells:
        dc, da = c["delta_vs_canonical"], c["availability_effect"]
        a(f"{c['name']:<26} {c['full_log_loss']:>9.6f} "
          f"{dc['delta_candidate_minus_reference']:>+10.6f} "
          f"{dc['abs_z']:>6.2f} "
          f"{da['delta_candidate_minus_reference']:>+10.6f} "
          f"{da['abs_z']:>6.2f} {c['close_log_loss']:>9.5f}")
    a("")
    for key in ("halflife_at_prior_all", "squad_at_hl15_all",
                "top_n_at_hl15_prior"):
        parts = []
        for value, r in axis[key].items():
            parts.append(
                f"{value}: cell {r['delta_vs_canonical']:+.6f} "
                f"(|z| {r['delta_vs_canonical_abs_z']:.2f}), avail "
                f"{r['availability_effect']:+.6f} "
                f"(|z| {r['availability_abs_z']:.2f})")
        a(f"{key}: " + " | ".join(parts))
    a("")
    a(f"beats recorded at |z| > 3: {beating or 'none'}")
    a(f"worse than recorded at |z| > 3: {worse or 'none'}")
    a("")
    a(verdict)
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
