"""Issue #58: variance-conditioned staking.

Does gating or sizing the selective-betting rule by PREDICTIVE UNCERTAINTY move
the P&L?  The MDPI uncertainty paper motivates the dimension, but its own
honest table loses about -7% under 0.3xKelly, and this repo's breakeven rule
loses -8.71% on the current comparison (recorded -8.23% on the older frame).
The prior is against it.

DESIGN (pre-registered I58-prereg-variance-staking)
---------------------------------------------------
- Population: the same 5,788-game market comparison (build_games joined with
  closing lines); model arm = OOF walk-forward model+availability; the rule
  being conditioned is the breakeven rule (both sides, positive edge over the
  raw price).
- Dispersion: per-game standard deviation of the OOF logistic probability
  across a RESIDUAL BOOTSTRAP of each training block: B=50 fits on bootstrap
  resamples of the block, predicting the test rows (seed 42 for the resampler).
- Rules: GATE (from the breakeven bet set, bet only when dispersion is in the
  lowest 25/50/75%, 50% being the primary) and SIZE (stake weighted by
  1 - var_percentile, and by 0.3*Kelly*(1 - var_percentile)), both reported.
- Decision: promotion requires a rule to beat the reproduced baseline ROI by a
  positive difference with |z| > 3 on a 2,000-resample bootstrap of the
  rule-vs-baseline difference (seed 42).  Anything else is a null.  The
  median-gate rule is the gating decision; no cutpoint shopping.

Run::

    OMP_NUM_THREADS=1 python scripts/issue58_variance_staking.py
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

from DataProviders import nba_data as nd            # noqa: E402
from Models import measurement as ms                # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


cl = _load_script("closing_line")

OUT_JSON = ROOT / "results" / "issue58_variance_staking.json"
OUT_MD = ROOT / "results" / "issue58_variance_staking.md"
JSONL = ROOT / "results" / "experiments.jsonl"

B_VAR = 50
N_BOOT = 2000
SEED = 42
GATES = (0.25, 0.50, 0.75)
RECORDED_BASELINE_DESC = "recorded 4,828 bets / -8.23% (older frame); current reproduction printed"


def _boot_oof_variance(X, y, games):
    """OOF probs (mean of resample fits? NO: base fit once) + per-game sd.

    Base probs come from ONE fit per training block (= the canonical OOF arm);
    dispersion comes from B resample fits per block predicting the same test
    rows.  Returns (base_probs, dispersion) aligned to len(games); NaN on the
    unscored first half.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(SEED)
    n = len(games)
    base = np.full(n, np.nan)
    disp = np.full(n, np.nan)

    def _pipe():
        return Pipeline([("i", SimpleImputer(strategy="median")),
                         ("s", StandardScaler()),
                         ("c", LogisticRegression(C=1.0, max_iter=300,
                                                  random_state=SEED))])

    for tr, te in nd.walk_forward_split(games, n_splits=5):
        p_base = _pipe()
        p_base.fit(X.iloc[tr], y[tr])
        base[te] = p_base.predict_proba(X.iloc[te])[:, 1]
        cols = list(X.columns)
        P = np.empty((len(te), B_VAR))
        for b in range(B_VAR):
            idx = rng.integers(0, len(tr), size=len(tr))
            pib = _pipe()
            pib.fit(X.iloc[tr[idx]], y[tr[idx]])
            P[:, b] = pib.predict_proba(X.iloc[te])[:, 1]
        disp[te] = P.std(axis=1, ddof=1)
    return base, disp


def _pa_settle_row(p, p_h, p_a, y):
    """Return (bet_mask, won, side_q, edge, profit, stake) for the breakeven
    rule, mirroring edge_betting.settle with margin=0."""
    p_away = 1.0 - p
    take_home = (p - p_h) >= (p_away - p_a)
    side_p = np.where(take_home, p, p_away)
    side_q = np.where(take_home, p_h, p_a)
    edge = np.where(take_home, p - p_h, p_away - p_a)
    bet = edge > 0.0
    won = np.where(take_home, y, 1 - y).astype(float)
    profit = np.where(bet, won * (1.0 / side_q - 1.0) - (1.0 - won), 0.0)
    stake = np.where(bet, side_q, 0.0)
    return bet, won, side_q, edge, profit, stake


def roi_of(w, won, side_q, profit, stake, bet):
    """Weighted rule ROI: sum(profit*w)/sum(stake*w) over the bet set."""
    b = bet & (w > 0)
    if b.sum() == 0:
        return {"bets": 0, "roi": float("nan")}
    sp = float((profit[b] * w[b]).sum())
    sk = float((stake[b] * w[b]).sum())
    return {"bets": int(b.sum()), "roi": float(sp / sk) if sk > 0 else float("nan")}


def gate_rule(var_pct, gate):
    return (var_pct <= gate).astype(float)


def kelly_weight(p, q, frac=0.3):
    """Fractional Kelly weight for a bet at model p against price q."""
    b = (1.0 - q) / q          # net odds
    f = b * p - (1.0 - p)      # full-Kelly fraction
    return np.clip(frac * f, 0.0, 1.0)


def main() -> int:
    t0 = time.time()
    games = nd.build_games().reset_index(drop=True)
    odds = cl.load_odds()
    joined = cl.join_odds(games, odds).reset_index(drop=True)

    X = cl.feature_frame(games, with_availability=True)
    y_all = games["y"].to_numpy().astype(int)
    p_av, disp = _boot_oof_variance(X, y_all, games)

    g = games[["GAME_ID", "y"]].copy()
    g["p_model"] = p_av
    g["dispersion"] = disp
    joined = joined.merge(g, on="GAME_ID", how="left")
    joined = joined.dropna(subset=["p_model", "p_market", "dispersion", "y"]
                           ).reset_index(drop=True)

    y = joined["y"].to_numpy().astype(int)
    p = joined["p_model"].to_numpy(float)
    p_h = joined["p_market_raw"].to_numpy(float)
    p_a = joined["p_away_raw"].to_numpy(float)
    var = joined["dispersion"].to_numpy(float)

    bet, won, side_q, edge, profit, stake = _pa_settle_row(p, p_h, p_a, y)
    n_bet = int(bet.sum())
    base_roi = float(profit[bet].sum() / stake[bet].sum())
    # variance percentile WITHIN the bet set (the rules operate on bets).
    var_b = var[bet]
    pct = (var_b.argsort().argsort() + 1) / n_bet

    rng = np.random.default_rng(SEED)
    pb = profit[bet]
    sb = stake[bet]
    gate_rows = {}
    for q in GATES:
        w = gate_rule(pct, q)
        keep = w > 0
        gw = w[keep]
        r = roi_of(np.ones(int(keep.sum())), won[bet][keep], side_q[bet][keep],
                   pb[keep], sb[keep], np.ones(int(keep.sum()), bool))
        diffs = np.empty(N_BOOT)
        for i in range(N_BOOT):
            pick = rng.integers(0, n_bet, size=n_bet)
            base_draws = pb[pick].sum() / sb[pick].sum()
            gp = pb[pick] * w[pick]
            gs = sb[pick] * w[pick]
            gated = (gp.sum() / gs.sum()) if gs.sum() > 0 else float("nan")
            diffs[i] = gated - base_draws
        diff_se = float(diffs.std(ddof=1))
        diff_mean = float(diffs.mean())
        gate_rows[f"gate_{int(q*100)}"] = {
            "n_bets": r["bets"],
            "roi": r["roi"],
            "baseline_roi": base_roi,
            "rule_minus_baseline": diff_mean,
            "ci": [float(np.percentile(diffs, 2.5)),
                   float(np.percentile(diffs, 97.5))],
            "z": (diff_mean / diff_se if diff_se > 0 else float("nan")),
            "p_positive": float((diffs > 0).mean()),
        }

    # Sized rules (descriptive; cannot promote).
    w_var = (1.0 - pct)
    sized_var = roi_of(w_var, won[bet], side_q[bet], profit[bet],
                       stake[bet], np.ones(n_bet, bool))
    w_kelly = kelly_weight(p[bet], side_q[bet], frac=0.3) * w_var
    sized_kelly = roi_of(w_kelly, won[bet], side_q[bet], profit[bet],
                         stake[bet], np.ones(n_bet, bool))

    # Diagnostics: is dispersion informative about correctness?
    acc = won[bet]  # did the bet side win
    wrong = (p[bet] > 0.5) != acc
    corr = float(np.corrcoef(var_b, wrong)[0][1])
    # realized win rate by variance tercile (bet set)
    terc = np.quantile(var_b, [1 / 3, 2 / 3])
    win_by_tercile = {
        "low": float(acc[var_b <= terc[0]].mean()) if (var_b <= terc[0]).sum() else float("nan"),
        "mid": float(acc[(var_b > terc[0]) & (var_b <= terc[1])].mean())
        if ((var_b > terc[0]) & (var_b <= terc[1])).sum() else float("nan"),
        "high": float(acc[var_b > terc[1]].mean()) if (var_b > terc[1]).sum() else float("nan"),
    }

    g50 = gate_rows["gate_50"]
    point = g50["rule_minus_baseline"]
    z = g50["z"]
    if point > 0 and z is not None and abs(z) > 3:
        verdict = ("VARIANCE GATING CLEARS THE BAR: the median-gate rule beats "
                   f"the baseline by {point:+.4f} ROI with |z| {abs(z):.2f} on "
                   "the rule-vs-baseline bootstrap -- uncertainty is priced "
                   "into profitability.")
    else:
        verdict = (
            "NULL: variance conditioning does not rescue the rule. "
            f"The median gate moves ROI by {point:+.4f} vs the "
            f"{base_roi*100:+.2f}% baseline (|z| {abs(z) if z is not None else float('nan'):.2f}), "
            "below the bar; the win-rate-by-variance-tercile diagnostic shows "
            f"whether uncertainty even flags where the model is wrong. No "
            "promotion."
        )

    payload = {
        "issue": 58,
        "protocol_ref": "I58-prereg-variance-staking",
        "generated_utc": ms.utc_now(),
        "runtime_seconds": round(time.time() - t0, 1),
        "n_games_with_line": int(len(joined)),
        "baseline_breakeven": {"n_bets": n_bet, "roi": base_roi,
                               "recorded": RECORDED_BASELINE_DESC},
        "dispersion": {
            "source": "residual bootstrap of the walk-forward logistic (B=50, "
                      "seed 42)",
            "mean": float(var.mean()), "sd": float(var.std(ddof=1)),
            "corr_with_mistake": corr,
            "win_rate_by_variance_tercile": win_by_tercile,
        },
        "gate_rules": gate_rows,
        "sized_rules": {"stake_prop_1_minus_var": sized_var,
                        "stake_prop_0.3kelly_times_1_minus_var": sized_kelly},
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ms.append_jsonl(JSONL, {
        "name": "I58-run-variance-staking",
        "protocol_ref": "I58-prereg-variance-staking",
        "declared_utc": ms.utc_now(),
        "experiment_type": "issue #58: variance-conditioned staking",
        "n_games_with_line": int(len(joined)),
        "baseline_breakeven": payload["baseline_breakeven"],
        "dispersion": payload["dispersion"],
        "gate_rules": gate_rows,
        "sized_rules": payload["sized_rules"],
        "verdict": verdict,
        "status": "executed",
    })

    L = ["=" * 78,
         "ISSUE 58 -- VARIANCE-CONDITIONED STAKING",
         "=" * 78]
    a = L.append
    a(f"games with line {len(joined)}; baseline breakeven: {n_bet:,} bets, "
      f"ROI {base_roi*100:+.2f}% [recorded 4,828 / -8.23%]")
    a(f"dispersion (residual bootstrap, B=50): mean {var.mean():.4f}, "
      f"sd {var.std(ddof=1):.4f}; corr(dispersion, wrong-bet) {corr:+.3f}; "
      f"win rate low/mid/high tercile "
      f"{win_by_tercile['low']:.3f}/{win_by_tercile['mid']:.3f}/"
      f"{win_by_tercile['high']:.3f}")
    a("")
    a("gate rules (bet only when dispersion percentile <= gate):")
    for k, r in gate_rows.items():
        a(f"  {k:<8} n={r['n_bets']:>4} ROI {r['roi']*100:>+8.2f}%  "
          f"delta-vs-base {r['rule_minus_baseline']*100:>+8.2f}pp  "
          f"|z| {abs(float(r['z'])):.2f}")
    a("sized rules (descriptive):")
    a(f"  stake ~ 1-var_pctile : ROI {sized_var['roi']*100:+.2f}% "
      f"(n {sized_var['bets']})")
    a(f"  stake ~ 0.3kelly(1-var): ROI {sized_kelly['roi']*100:+.2f}% "
      f"(n {sized_kelly['bets']})")
    a("")
    a(verdict)
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())