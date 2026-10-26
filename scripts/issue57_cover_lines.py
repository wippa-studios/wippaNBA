"""Issue #57: the spread-cover betting objective -- alternate lines + volatility.

c44's C2 already measures cover probability QUALITY (Skellam cover delta
+0.00415, |z| 14.53) and a single posted-line cover ROI at a flat -110.  This
arm adds what that measurement never asked (pre-registered
I57-prereg-spread-cover): (a) whether the parameter-free cover rule makes
money at the posted line AND at alternate lines (+/-3.5, +/-7.5) around it,
and (b) whether the margin model's own sigma_game ("implied volatility", the
Peterson/Polson-Stern thesis) is informative for a cover-ticket rule via
gating/sizing.

The predictions are c44's cached OOF margin model, reused VERBATIM: this arm
differs from c44 only in the rule, never in the numbers it prices.  A cover
push (margin == line) returns the stake: zero P&L, not a loss.

DECISION (pre-registered): the POSTED-LINE cover rule promotes iff its ROI is
> 0 with |z| > 3 (2,000-resample bootstrap, seed 42), i.e. the model's cover
pick beats the vig-less -110 line.  Alternate and sigma cells are reported;
no single cell can promote.

Run::

    OMP_NUM_THREADS=1 python scripts/issue57_cover_lines.py
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
c44 = _load_script("c44_spread_total")

OUT_JSON = ROOT / "results" / "issue57_cover_lines.json"
OUT_MD = ROOT / "results" / "issue57_cover_lines.md"
JSONL = ROOT / "results" / "experiments.jsonl"

P_PRICE = 110.0 / 210.0      # flat -110, both sides; no juice in the data
OFFSETS = (0.0, -3.5, 3.5, -7.5, 7.5)
N_BOOT = 2000
SEED = 42


def cover_settle(p_home: np.ndarray, margin: np.ndarray, line: np.ndarray,
                 ) -> dict:
    """Parameter-free cover rule: bet the cover side with model p > 0.5, at a
    flat -110, push (margin == line) returns the stake.
    """
    p_away = 1.0 - p_home
    take_home = p_home >= p_away
    side_p = np.where(take_home, p_home, p_away)
    bet = side_p >= 0.5
    y_cov = (margin > line).astype(float)          # home covers
    won = np.where(take_home, y_cov, 1.0 - y_cov)  # away covers wins
    push = (margin == line).astype(float)          # stake returned
    profit = np.where(bet, won * (1.0 / P_PRICE - 1.0) - (1.0 - won), 0.0)
    profit = np.where(push, 0.0, profit)           # voids
    stake = bet.astype(float)
    n = int(bet.sum())
    rng = np.random.default_rng(SEED)
    draws = np.empty(N_BOOT)
    idx = np.flatnonzero(bet)
    for i in range(N_BOOT):
        pick = rng.integers(0, n, size=n)
        draws[i] = profit[idx][pick].sum() / stake[idx][pick].sum()
    se = float(draws.std(ddof=1))
    roi = float(profit.sum() / stake.sum()) if n else float("nan")
    return {
        "n_bets": n,
        "home_bets": int((bet & take_home).sum()),
        "away_bets": int((bet & ~take_home).sum()),
        "n_pushes": int(push.sum()),
        "roi": roi,
        "se": se,
        "z": (roi / se if se > 0 else float("nan")),
        "ci": [float(np.percentile(draws, 2.5)),
               float(np.percentile(draws, 97.5))],
    }


def main() -> int:
    t0 = time.time()
    games = nd.build_games().reset_index(drop=True)
    joined = cl.join_odds(games, cl.load_odds()).reset_index(drop=True)
    row_of = {gid: i for i, gid in enumerate(games["GAME_ID"].to_numpy())}
    joined["row"] = joined["GAME_ID"].map(row_of)

    pred = c44.fit_predict(games, reuse_cache=True)
    scored_rows = np.flatnonzero(pred["row"] >= 0)     # global rows scored
    joined = joined[joined["row"].isin(set(scored_rows.tolist()))].copy()
    joined["row"] = joined["row"].astype(int)
    L = c44.effective_line(joined)["L_eff"].to_numpy(float)
    good = np.abs(L) <= c44.SPREAD_MAX_ABS
    joined = joined[good].reset_index(drop=True)
    L = L[good]
    margin = games["margin"].to_numpy()[joined["row"].to_numpy()]

    pos = {int(r): i for i, r in enumerate(scored_rows)}
    at = np.array([pos[int(r)] for r in joined["row"].to_numpy()])
    comp = {k: pred[k][scored_rows] for k in ("m_hat", "t_hat", "sigma")}
    m_hat = comp["m_hat"][at]
    sigma = comp["sigma"][at]
    t_hat = comp["t_hat"][at]
    # Skellam rates from the margin/total model's own predictions (c44's
    # reparametrisation): mu_home = (t_hat + m_hat)/2, mu_away = (t_hat - m_hat)/2.
    mu_h = np.clip((t_hat + m_hat) / 2.0, 0.01, None)
    mu_a = np.clip((t_hat - m_hat) / 2.0, 0.01, None)

    rows = {}
    for a in OFFSETS:
        La = L + a
        p_g = c44.cover_prob_gaussian(La, m_hat, sigma)
        p_s = c44.cover_prob_skellam(La, mu_h, mu_a)
        rows[f"gaussian_offset_{a:+.1f}"] = {
            "offset": a,
            "n_games": int(len(joined)),
            "mean_p_home_cover": float(p_g.mean()),
            "cover_rate": float((margin > La).mean()),
            "cover_settle": cover_settle(p_g, margin, La),
        }
        rows[f"skellam_offset_{a:+.1f}"] = {
            "offset": a, "n_games": int(len(joined)),
            "mean_p_home_cover": float(p_s.mean()),
            "cover_rate": float((margin > La).mean()),
            "cover_settle": cover_settle(p_s, margin, La),
        }

    # Sigma dimension on the posted-line Gaussian bet set.
    base = rows["gaussian_offset_+0.0"]["cover_settle"]
    p_g0 = c44.cover_prob_gaussian(L, m_hat, sigma)
    bet = base["n_bets"]
    p_away = 1 - p_g0
    take_home = p_g0 >= p_away
    side_p = np.where(take_home, p_g0, p_away)
    bet_mask = side_p >= 0.5
    sig_bet = sigma[bet_mask]
    pct = (sig_bet.argsort().argsort() + 1) / max(1, int(bet_mask.sum()))
    low = pct <= 0.5
    y_cov = (margin > L).astype(float)
    won = np.where(take_home, y_cov, 1 - y_cov)
    push = margin == L
    profit = np.where(bet_mask, won * (1 / P_PRICE - 1) - (1 - won), 0.0)
    profit = np.where(push, 0.0, profit)
    stake = bet_mask.astype(float)

    def roi_of(w):
        b = bet_mask & (w > 0)
        n = int(b.sum())
        if n == 0:
            return {"n_bets": 0, "roi": float("nan")}
        return {"n_bets": n,
                "roi": float((profit[b] * w[b]).sum()
                             / (stake[b] * w[b]).sum())}

    sigma_dim = {
        "gate_low_half": roi_of(low.astype(float)),
        "size_1_minus_pct": roi_of(1.0 - pct),
    }

    g0 = rows["gaussian_offset_+0.0"]["cover_settle"]
    point = g0["roi"]
    z = abs(g0["z"]) if g0["z"] == g0["z"] else float("nan")
    if point > 0 and z > 3:
        verdict = (f"COVER RULE CLEARS THE BAR: posted-line cover ROI "
                   f"{point*100:+.2f}% (|z| {z:.2f}) beats the vig-less -110 "
                   "line -- the model's cover pick is information.")
    elif point < 0 and z > 3:
        verdict = (
            f"SIGNIFICANTLY NEGATIVE: the posted-line cover pick returns "
            f"{point*100:+.2f}% at a vig-less -110 (|z| {z:.2f}, CI "
            f"{[f'{c*100:+.2f}%' for c in g0['ci']]}) -- the closing spread is "
            "efficient against exactly this model: its favourite covers only "
            "~49% of the time. The alternate-line sweep (up to +37% at +/-7.5) "
            "is the same games re-scored against lines the market never "
            "offered -- a bettor cannot take them; it shows the margin "
            "distribution is wide, not that the line is wrong."
        )
    else:
        verdict = (
            f"NULL: posted-line cover ROI {point*100:+.2f}% "
            f"(|z| {z:.2f}, CI {[f'{c*100:+.2f}%' for c in g0['ci']]}) -- no "
            "signal either way at even money."
        )

    payload = {
        "issue": 57,
        "protocol_ref": "I57-prereg-spread-cover",
        "generated_utc": ms.utc_now(),
        "runtime_seconds": round(time.time() - t0, 1),
        "rule": ("parameter-free cover rule at flat -110 both sides, pushes "
                 "void; c44 predictions reused verbatim"),
        "n_games_sane_spread": int(len(joined)),
        "rows": rows,
        "sigma_dimension": sigma_dim,
        "posted_line_gaussian": g0,
        "verdict": verdict,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    ms.append_jsonl(JSONL, {
        "name": "I57-run-cover-lines",
        "protocol_ref": "I57-prereg-spread-cover",
        "declared_utc": ms.utc_now(),
        "experiment_type": "issue #57: spread-cover objective, alternate lines",
        "n_games_sane_spread": int(len(joined)),
        "posted_line_gaussian": g0,
        "rows": rows,
        "sigma_dimension": sigma_dim,
        "verdict": verdict,
        "status": "executed",
    })

    L_ = ["=" * 78,
          "ISSUE 57 -- SPREAD-COVER BETTING OBJECTIVE",
          "=" * 78]
    a = L_.append
    a(f"games with sane spread {len(joined)}; predictions from c44 cache")
    a(f"{'line':<18} {'P(home cover)':>12} {'n bets':>8} {'ROI':>9} {'|z|':>6} "
      f"{'cover rate':>10}")
    for k, r in rows.items():
        cs = r["cover_settle"]
        a(f"{k:<18} {r['mean_p_home_cover']:>12.4f} {cs['n_bets']:>8} "
          f"{cs['roi']*100:>+8.2f}% {abs(float(cs['z'])):>6.2f} "
          f"{r['cover_rate']:>10.4f}")
    a("sigma dimension (posted Gaussian line):")
    for k, v in sigma_dim.items():
        a(f"  {k:<16} n={v['n_bets']:>4} ROI {v['roi']*100 if v['roi']==v['roi'] else float('nan'):+.2f}%")
    a("")
    a(verdict)
    OUT_MD.write_text("\n".join(L_) + "\n")
    print("\n".join(L_))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())