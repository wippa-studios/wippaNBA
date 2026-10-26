"""Issue #74: form-mismatch spots, priced.

The spot the issue thread proposed: a home team that is 5-5 or better over its
last 10 games, hosting a team that is 2-8 or worse, is underpriced at the
close.  The naive version of the belief is about winning; the payable version
is about the price.  A -400 favourite that wins 75% of the time is not a bet.

The family is pre-registered in ``results/experiments.jsonl`` as
``I74-prereg-form-mismatch-spots``.  The ONE promotion gate is the exact cell
on the home spread at a flat -110, pooled over every priced season
(2015-16..2025-26, n=965), at |z| > 3 on 2,000 bootstrap resamples (seed 42).
Everything else -- the moneyline, the mirror, the form-gap ladder, the
per-season rows, the 2025-26-only slice, the within-season-form sensitivity --
is descriptive and cannot promote.  The model diagnostic (canonical + the cell
indicator on the 2025-26 holdout) answers "does the spot carry information at
all"; it cannot promote a betting claim, because a feature the model lacks can
still be fully priced.

Why this is not a repeat of I60/I60b: those tested recent-form win counts as
MODEL FEATURES, both sides' cumulative ladders, with no interaction.  This
tests the JOINT state as a BETTING SPOT at the price.  The prior makes a null
the likely outcome; the point of running it is to price the belief rather than
argue about it.

Run::

    OMP_NUM_THREADS=1 python scripts/issue74_form_mismatch.py
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

OUT_JSON = ROOT / "results" / "issue74_form_mismatch.json"
OUT_MD = ROOT / "results" / "issue74_form_mismatch.md"
JSONL = ROOT / "results" / "experiments.jsonl"

PROTOCOL = "I74-prereg-form-mismatch-spots"
OPENER = pd.Timestamp("2025-10-21")
N_BOOT = 2000
SEED = 42
P_PRICE = 110.0 / 210.0          # flat -110, both sides; no juice in the data
WINDOW = 10
MIN_HISTORY = 10


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cl = _load_script("closing_line")
c44 = _load_script("c44_spread_total")


# --------------------------------------------------------------------------- #
# Causal form + history
# --------------------------------------------------------------------------- #

def _long(games: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game, already sorted chronologically within team."""
    g = games.reset_index(drop=True)
    long = pd.concat([
        g[["GAME_ID", "GAME_DATE", "SEASON", "home", "away",
           "home_pts", "away_pts"]].assign(team=lambda d: d["home"]),
        g[["GAME_ID", "GAME_DATE", "SEASON", "home", "away",
           "home_pts", "away_pts"]].assign(team=lambda d: d["away"]),
    ], ignore_index=True)
    long["won"] = (
        ((long["team"] == long["home"]) & (long["home_pts"] > long["away_pts"]))
        | ((long["team"] != long["home"]) & (long["away_pts"] > long["home_pts"]))
    )
    return long.sort_values(["team", "GAME_DATE"])


def form_wins_entering(games: pd.DataFrame, window: int = WINDOW,
                       reset_seasons: bool = False) -> pd.DataFrame:
    """Wins in the team's most recent up-to-``window`` games before tonight.

    ``reset_seasons=False`` is the definition I60b used and must equal
    ``nba_data.recent_form_features`` exactly (pinned by a test).  With
    ``reset_seasons=True`` the window restarts each season: a team's October
    form is its October form, not last April's.
    """
    long = _long(games)
    keys = ["team", "SEASON"] if reset_seasons else ["team"]
    counts: dict[int, int] = {}
    for _key, grp in long.groupby(keys, sort=False):
        acc = 0
        ring: list[int] = []
        for idx, won in zip(grp.index, grp["won"].astype(int).tolist()):
            counts[idx] = acc
            ring.append(int(won))
            acc += int(won)
            if len(ring) > window:
                acc -= ring.pop(0)
    long["form"] = long.index.map(counts)
    h = long[long["team"] == long["home"]].set_index("GAME_ID")["form"]
    a = long[long["team"] != long["home"]].set_index("GAME_ID")["form"]
    return pd.DataFrame({
        "home_form10_enter": games["GAME_ID"].map(h).to_numpy(dtype=float),
        "away_form10_enter": games["GAME_ID"].map(a).to_numpy(dtype=float),
    }, index=games.index)


def games_played_entering(games: pd.DataFrame,
                          reset_seasons: bool = False) -> pd.DataFrame:
    """Prior games in the frame (or in the season), strictly before tonight."""
    long = _long(games)
    keys = ["team", "SEASON"] if reset_seasons else ["team"]
    long["gp"] = long.groupby(keys, sort=False).cumcount()
    h = long[long["team"] == long["home"]].set_index("GAME_ID")["gp"]
    a = long[long["team"] != long["home"]].set_index("GAME_ID")["gp"]
    return pd.DataFrame({
        "home_gp_enter": games["GAME_ID"].map(h).to_numpy(dtype=int),
        "away_gp_enter": games["GAME_ID"].map(a).to_numpy(dtype=int),
    }, index=games.index)


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #

def spot_masks(form: pd.DataFrame, history: pd.DataFrame) -> dict[str, np.ndarray]:
    """The pre-registered cells, fixed before any outcome was read."""
    h = form["home_form10_enter"].to_numpy(dtype=int)
    a = form["away_form10_enter"].to_numpy(dtype=int)
    known = ((history["home_gp_enter"].to_numpy(dtype=int) >= MIN_HISTORY)
             & (history["away_gp_enter"].to_numpy(dtype=int) >= MIN_HISTORY))
    gap = h - a
    out = {
        "exact": known & (h >= 5) & (a <= 2),
        "mirror": known & (h <= 2) & (a >= 5),
    }
    for k in (2, 4, 6, 8):
        out[f"gap_ge{k}"] = known & (gap >= k)
    return out


# --------------------------------------------------------------------------- #
# Pricing
# --------------------------------------------------------------------------- #

def _roi(profit: np.ndarray, stake: np.ndarray, won: np.ndarray | None = None,
         push: np.ndarray | None = None) -> dict:
    """ROI with a nonparametric bootstrap over bets, exactly as #57 does it."""
    active = stake > 0
    n = int(active.sum())
    if n == 0:
        return {"n_bets": 0, "roi": float("nan"), "se": float("nan"),
                "z": float("nan"), "ci": [float("nan"), float("nan")],
                "win_rate": float("nan"), "n_pushes": 0,
                "total_profit": 0.0, "total_staked": 0.0}
    idx = np.flatnonzero(active)
    roi = float(profit[active].sum() / stake[active].sum())
    rng = np.random.default_rng(SEED)
    draws = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.integers(0, len(idx), size=len(idx))
        draws[i] = profit[idx][pick].sum() / stake[idx][pick].sum()
    se = float(draws.std(ddof=1))
    graded = active if push is None else (active & ~push)
    return {
        "n_bets": n,
        "n_pushes": int((push & active).sum()) if push is not None else 0,
        "roi": roi,
        "se": se,
        "z": (roi / se if se > 0 else float("nan")),
        "ci": [float(np.percentile(draws, 2.5)),
               float(np.percentile(draws, 97.5))],
        "win_rate": (float(won[graded].mean())
                     if won is not None and graded.sum() else float("nan")),
        "total_profit": float(profit[active].sum()),
        "total_staked": float(stake[active].sum()),
    }


def settle_ats(mask: np.ndarray, margin: np.ndarray, line: np.ndarray,
               side: str) -> dict:
    """Flat -110 spread bet; a push returns the stake (0 P&L, stake counted)."""
    bet = np.asarray(mask, dtype=bool)
    won = (margin > line) if side == "home" else (margin < line)
    push = margin == line
    profit = np.where(bet & ~push,
                      np.where(won, 1.0 / P_PRICE - 1.0, -1.0), 0.0)
    stake = bet.astype(float)
    out = _roi(profit, stake, won=won, push=push)
    out["mean_line"] = float(line[bet].mean()) if bet.any() else float("nan")
    out["break_even_win_rate"] = float(P_PRICE)
    return out


def settle_ml(mask: np.ndarray, y: np.ndarray, price_home: np.ndarray,
              price_away: np.ndarray, side: str) -> dict:
    """Moneyline at the posted raw price (vig included), fixed side."""
    bet = np.asarray(mask, dtype=bool)
    price = price_home if side == "home" else price_away
    won = (y == 1) if side == "home" else (y == 0)
    profit = np.where(bet, np.where(won, 1.0 / price - 1.0, -1.0), 0.0)
    stake = bet.astype(float)
    out = _roi(profit, stake, won=won)
    out["mean_price"] = float(price[bet].mean()) if bet.any() else float("nan")
    out["mean_break_even_rate"] = out["mean_price"]
    return out


# --------------------------------------------------------------------------- #
# Model diagnostic
# --------------------------------------------------------------------------- #

def _fit_predict(X: pd.DataFrame, y: np.ndarray, tr: np.ndarray,
                 te: np.ndarray, features: list[str]) -> np.ndarray:
    model = lr.RollingLogreg(C=1.0, features=features).fit(X.iloc[tr], y[tr])
    return model.predict_proba(X.iloc[te])


def model_diagnostic(games: pd.DataFrame, exact: np.ndarray) -> dict:
    dates = pd.to_datetime(games["GAME_DATE"])
    tr = (dates < OPENER).to_numpy()
    te = (dates >= OPENER).to_numpy()
    y = games["y"].to_numpy(dtype=int)
    X = lr.load_design(games=games).reset_index(drop=True)
    features = lr.resolve_features(lr.DEFAULT_MODELS["full"])
    col = "in_form_mismatch"
    Xc = X.assign(**{col: exact.astype(float)})

    p_ref = _fit_predict(X, y, tr, te, features)
    p_cand = _fit_predict(Xc, y, tr, te, features + [col])
    delta = ms.delta_stats(y[te], p_cand, p_ref, n_boot=N_BOOT, seed=SEED)

    fitted = lr.RollingLogreg(C=1.0, features=features + [col]).fit(
        Xc.iloc[tr], y[tr])
    coef = float(fitted.coefficients()["coef"][col])

    avail = lr.resolve_features("availability")
    p_no_avail = _fit_predict(X, y, tr, te,
                              [f for f in features if f not in avail])
    control = ms.delta_stats(y[te], p_ref, p_no_avail,
                             n_boot=N_BOOT, seed=SEED)
    return {
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "n_test_in_cell": int(exact[te].sum()),
        "log_loss_reference": float(nd.evaluate(y[te], p_ref)["log_loss"]),
        "log_loss_candidate": float(nd.evaluate(y[te], p_cand)["log_loss"]),
        "delta": delta,
        "indicator_coef": coef,
        "positive_control_availability_on_minus_off": control,
    }


# --------------------------------------------------------------------------- #
# Artifact
# --------------------------------------------------------------------------- #

def _verdict(primary: dict, diagnostic: dict) -> dict:
    roi, z = primary["roi"], primary["z"]
    if roi > 0 and abs(z) > 3:
        betting = ("The exact cell CLEARS the bar on the home spread at -110: "
                   f"ROI {roi:+.2%}, |z| {abs(z):.2f} over {primary['n_bets']} "
                   "priced games. It is a retrospective multi-season result; "
                   "the next step is a pre-registered forward paper-trading leg, "
                   "not a model change.")
    elif roi > 0:
        betting = ("The exact cell is NOT PROMOTED: ROI "
                   f"{roi:+.2%} (CI {primary['ci'][0]:+.2%} to "
                   f"{primary['ci'][1]:+.2%}), |z| {abs(z):.2f}, over "
                   f"{primary['n_bets']} bets. Positive but below the bar, and "
                   "the stated power arithmetic cannot resolve an edge this "
                   "size; not evidence of a small edge.")
    else:
        betting = ("The exact cell LOSES at the price: ROI "
                   f"{roi:+.2%} (CI {primary['ci'][0]:+.2%} to "
                   f"{primary['ci'][1]:+.2%}), |z| {abs(z):.2f}, over "
                   f"{primary['n_bets']} bets. The market has the spot.")
    d = diagnostic["delta"]
    if d["delta_candidate_minus_reference"] > 0 and d["abs_z"] > 3:
        model = ("The cell indicator CLEARS the model diagnostic: "
                 f"+{d['delta_candidate_minus_reference']:.6f} nats, |z| "
                 f"{d['abs_z']:.2f} on the 2025-26 holdout. The spot carries "
                 "information the canonical features lack; whether the market "
                 "prices it is the betting test's question.")
    elif d["delta_candidate_minus_reference"] > 0:
        model = ("The cell indicator is null in the model diagnostic: "
                 f"{d['delta_candidate_minus_reference']:+.6f} nats "
                 f"(CI {d['ci_lo']:+.6f} to {d['ci_hi']:+.6f}), |z| "
                 f"{d['abs_z']:.2f}. Positive, below the bar, not promoted.")
    else:
        model = ("The cell indicator is null/negative in the model diagnostic: "
                 f"{d['delta_candidate_minus_reference']:+.6f} nats "
                 f"(CI {d['ci_lo']:+.6f} to {d['ci_hi']:+.6f}), |z| "
                 f"{d['abs_z']:.2f}. No evidence the spot adds information "
                 "beyond the canonical features.")
    return {"primary_betting": betting, "model_diagnostic": model}


def _row(label: str, r: dict, price: str) -> dict:
    return {
        "label": label,
        "price": price,
        "n_bets": r["n_bets"],
        "n_pushes": r.get("n_pushes", 0),
        "roi": r["roi"],
        "se": r["se"],
        "z": r["z"],
        "ci": r["ci"],
        "win_rate": r["win_rate"],
        "break_even_win_rate": r.get("break_even_win_rate",
                                     r.get("mean_break_even_rate")),
        "mean_line": r.get("mean_line"),
        "mean_price": r.get("mean_price"),
    }


def main() -> int:
    t0 = time.time()
    games = nd.build_games().reset_index(drop=True)
    form = form_wins_entering(games, WINDOW, reset_seasons=False)
    history = games_played_entering(games, reset_seasons=False)
    masks = spot_masks(form, history)

    joined = cl.join_odds(games, cl.load_odds())
    meta = pd.DataFrame({
        "GAME_ID": games["GAME_ID"].to_numpy(),
        "home_form10_enter": form["home_form10_enter"].to_numpy(),
        "away_form10_enter": form["away_form10_enter"].to_numpy(),
        "home_gp_enter": history["home_gp_enter"].to_numpy(),
        "away_gp_enter": history["away_gp_enter"].to_numpy(),
        "margin": games["margin"].to_numpy(),
        "y": games["y"].to_numpy(dtype=int),
    })
    j = joined.merge(meta, on="GAME_ID", how="left", validate="one_to_one")
    if j[["margin", "home_form10_enter", "away_form10_enter"]].isna().any().any():
        raise AssertionError("form/margin attachment dropped rows")
    j = c44.effective_line(j).reset_index(drop=True)
    dropped_no_spread = int((~np.isfinite(j["L_eff"])).sum())
    if dropped_no_spread:
        # A moneyline-only quote cannot settle the primary ATS test. Drop it
        # from BOTH prices so the two arms describe the same games, and record
        # the count (the no-silent-drops rule).
        j = j.loc[np.isfinite(j["L_eff"])].reset_index(drop=True)

    # Re-derive the masks on the priced frame so the two never drift.
    f = pd.DataFrame({
        "home_form10_enter": j["home_form10_enter"].to_numpy(dtype=float),
        "away_form10_enter": j["away_form10_enter"].to_numpy(dtype=float),
    })
    hist = pd.DataFrame({
        "home_gp_enter": j["home_gp_enter"].to_numpy(dtype=int),
        "away_gp_enter": j["away_gp_enter"].to_numpy(dtype=int),
    })
    m = spot_masks(f, hist)
    margin = j["margin"].to_numpy(dtype=float)
    line = j["L_eff"].to_numpy(dtype=float)
    y = j["y"].to_numpy(dtype=int)
    p_home = j["p_market_raw"].to_numpy(dtype=float)
    p_away = j["p_away_raw"].to_numpy(dtype=float)

    exact = m["exact"]
    pooled = {
        "exact_home_ats": settle_ats(exact, margin, line, "home"),
        "exact_home_ml": settle_ml(exact, y, p_home, p_away, "home"),
        "exact_away_ats": settle_ats(exact, margin, line, "away"),
        "exact_away_ml": settle_ml(exact, y, p_home, p_away, "away"),
    }
    mirror = {
        "mirror_away_ats": settle_ats(m["mirror"], margin, line, "away"),
        "mirror_away_ml": settle_ml(m["mirror"], y, p_home, p_away, "away"),
        "mirror_home_ats": settle_ats(m["mirror"], margin, line, "home"),
        "mirror_home_ml": settle_ml(m["mirror"], y, p_home, p_away, "home"),
    }
    ladder = {}
    for k in (2, 4, 6, 8):
        ladder[f"gap_ge{k}"] = {
            "home_ats": settle_ats(m[f"gap_ge{k}"], margin, line, "home"),
            "home_ml": settle_ml(m[f"gap_ge{k}"], y, p_home, p_away, "home"),
        }

    seasons = {}
    for season, sub in j.groupby("SEASON", sort=True):
        mask = exact[sub.index.to_numpy()]
        if mask.sum() == 0:
            seasons[season] = {"n_bets": 0}
            continue
        seasons[season] = {
            "n_bets": int(mask.sum()),
            "home_ats": settle_ats(mask, sub["margin"].to_numpy(float),
                                   sub["L_eff"].to_numpy(float), "home"),
            "home_ml": settle_ml(mask, sub["y"].to_numpy(int),
                                 sub["p_market_raw"].to_numpy(float),
                                 sub["p_away_raw"].to_numpy(float), "home"),
        }

    season_2526 = j["SEASON"] == "2025-26"
    mask_2526 = exact & season_2526.to_numpy()
    recent = {
        "n_bets": int(mask_2526.sum()),
        "home_ats": settle_ats(mask_2526, margin, line, "home"),
        "home_ml": settle_ml(mask_2526, y, p_home, p_away, "home"),
        "odds_end": str(pd.to_datetime(j.loc[season_2526, "GAME_DATE"]).max().date()),
    }

    # Within-season-form sensitivity (descriptive; cannot promote).
    form_ws = form_wins_entering(games, WINDOW, reset_seasons=True)
    hist_ws = games_played_entering(games, reset_seasons=True)
    masks_ws = spot_masks(form_ws, hist_ws)
    ws = masks_ws["exact"]
    ws_meta = pd.DataFrame({
        "GAME_ID": games["GAME_ID"].to_numpy(),
        "exact_ws": ws,
    })
    jw = j.merge(ws_meta, on="GAME_ID", how="left", validate="one_to_one")
    mask_ws = jw["exact_ws"].to_numpy(dtype=bool)
    within = {
        "n_bets": int(mask_ws.sum()),
        "home_ats": settle_ats(mask_ws, margin, line, "home"),
        "home_ml": settle_ml(mask_ws, y, p_home, p_away, "home"),
    }

    exact_all = masks["exact"]
    diagnostic = model_diagnostic(games, exact_all)

    primary = pooled["exact_home_ats"]
    verdicts = _verdict(primary, diagnostic)

    payload = {
        "issue": 74,
        "title": "form-mismatch spots, priced",
        "protocol_ref": PROTOCOL,
        "generated_utc": ms.utc_now(),
        "runtime_seconds": round(time.time() - t0, 1),
        "config": {
            "form": ("wins in the most recent up-to-10 games strictly before "
                     "tonight, no season reset (nd.recent_form_features); both "
                     "teams need >= 10 prior games"),
            "rules": {
                "exact": "home_form10 >= 5 and away_form10 <= 2 -> home",
                "mirror": "home_form10 <= 2 and away_form10 >= 5 -> away",
                "gap": "gap = home_form10 - away_form10; home for gap >= 2/4/6/8",
            },
            "prices": {
                "ats": "flat -110; push returns the stake (0 P&L, stake counted)",
                "ml": "closing moneyline, raw implied probability, vig included",
            },
            "primary": "exact cell, home ATS, pooled 2015-16..2025-26",
            "bar": "ROI > 0 and |z| > 3, 2,000 bootstrap resamples, seed 42",
            "power": ("n=965, per-bet sd ~0.95 -> SE ~3.1%, so |z|>3 needs "
                      "ROI ~9%+. A null rules out a large edge, not a 2-3% one."),
        },
        "population": {
            "priced_games_pooled": int(len(j)),
            "priced_games_2025_26": int(season_2526.sum()),
            "cell_games_all_seasons": int(exact_all.sum()),
            "cell_games_pooled": int(exact.sum()),
            "odds_end_2025_26": recent["odds_end"],
            "dropped_no_spread": dropped_no_spread,
        },
        "primary_home_ats": _row("exact cell, home ATS", primary, "ATS -110"),
        "pooled": {
            name: _row(name, r, "ATS -110" if name.endswith("ats") else "ML")
            for name, r in pooled.items()
        },
        "mirror": {
            name: _row(name, r, "ATS -110" if name.endswith("ats") else "ML")
            for name, r in mirror.items()
        },
        "gap_ladder": {
            k: {"home_ats": _row(f"{k} home ATS", v["home_ats"], "ATS -110"),
                "home_ml": _row(f"{k} home ML", v["home_ml"], "ML")}
            for k, v in ladder.items()
        },
        "per_season": {
            season: ({"n_bets": 0} if v["n_bets"] == 0 else {
                "n_bets": v["n_bets"],
                "home_ats": _row("home ATS", v["home_ats"], "ATS -110"),
                "home_ml": _row("home ML", v["home_ml"], "ML"),
            })
            for season, v in seasons.items()
        },
        "season_2025_26": {
            "n_bets": recent["n_bets"],
            "home_ats": _row("home ATS", recent["home_ats"], "ATS -110"),
            "home_ml": _row("home ML", recent["home_ml"], "ML"),
            "odds_end": recent["odds_end"],
        },
        "within_season_form": {
            "n_bets": within["n_bets"],
            "home_ats": _row("home ATS", within["home_ats"], "ATS -110"),
            "home_ml": _row("home ML", within["home_ml"], "ML"),
        },
        "model_diagnostic": diagnostic,
        "verdicts": verdicts,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    # ---- Markdown -------------------------------------------------------- #
    L = ["# Issue #74 - form-mismatch spots, priced", "",
         f"Pre-registered as `{PROTOCOL}`. Generated {payload['generated_utc']}.", "",
         "The spot: a home team **5-5 or better over its last 10**, hosting a "
         "team **2-8 or worse**, both with at least 10 prior games. The question "
         "is not whether the home team wins; it is whether it beats the price.", "",
         f"Pooled priced games: **{payload['population']['priced_games_pooled']:,}** "
         f"across 2015-16..2025-26; the cell holds "
         f"**{payload['population']['cell_games_pooled']:,}**. "
         f"2025-26 odds end {recent['odds_end']}.", "",
         "## Primary (only promotion gate)", "",
         "| bet | n | ROI | SE | |z| | 95% CI | win rate | break-even |",
         "|---|---|---|---|---|---|---|---|",
         f"| exact cell, home ATS -110 | {primary['n_bets']:,} | "
         f"{primary['roi']:+.2%} | {primary['se']:.2%} | {abs(primary['z']):.2f} | "
         f"{primary['ci'][0]:+.2%} to {primary['ci'][1]:+.2%} | "
         f"{primary['win_rate']:.1%} | {primary['break_even_win_rate']:.2%} |", "",
         f"**{verdicts['primary_betting']}**", "",
         "## The same cell at the moneyline, and the mirror", "",
         "| bet | n | ROI | SE | |z| | win rate | mean price / line |",
         "|---|---|---|---|---|---|---|",
    ]
    for name in ("exact_home_ml", "exact_away_ats", "exact_away_ml"):
        r = payload["pooled"][name]
        if r.get("mean_price") is not None:
            market = f"price {r['mean_price']:.2%}"
        elif r.get("mean_line") is not None:
            market = f"line {r['mean_line']:+.1f}"
        else:
            market = "-"
        L.append(f"| {name.replace('_', ' ')} | {r['n_bets']:,} | "
                 f"{r['roi']:+.2%} | {r['se']:.2%} | {abs(r['z']):.2f} | "
                 f"{r['win_rate']:.1%} | {market} |")
    L += ["", "## Form-gap ladder (descriptive)", "",
          "| gap | bets | home ATS ROI | |z| | home ML ROI | |z| |",
          "|---|---|---|---|---|---|"]
    for k, v in payload["gap_ladder"].items():
        L.append(f"| >= {k.replace('gap_ge', '')} | {v['home_ats']['n_bets']:,} | "
                 f"{v['home_ats']['roi']:+.2%} | {abs(v['home_ats']['z']):.2f} | "
                 f"{v['home_ml']['roi']:+.2%} | {abs(v['home_ml']['z']):.2f} |")
    L += ["", "## Per season (descriptive)", "",
          "| season | bets | home ATS ROI | home ML ROI |",
          "|---|---|---|---|"]
    for season, v in payload["per_season"].items():
        if v["n_bets"] == 0:
            L.append(f"| {season} | 0 | - | - |")
        else:
            L.append(f"| {season} | {v['n_bets']:,} | "
                     f"{v['home_ats']['roi']:+.2%} | {v['home_ml']['roi']:+.2%} |")
    L += ["", "## 2025-26 slice and within-season-form sensitivity", "",
          f"- 2025-26, {recent['n_bets']} priced cell games: home ATS "
          f"{recent['home_ats']['roi']:+.2%} (|z| {abs(recent['home_ats']['z']):.2f}), "
          f"home ML {recent['home_ml']['roi']:+.2%} "
          f"(|z| {abs(recent['home_ml']['z']):.2f}). This slice cannot decide anything.",
          f"- Within-season form, {within['n_bets']} cell games: home ATS "
          f"{within['home_ats']['roi']:+.2%} "
          f"(|z| {abs(within['home_ats']['z']):.2f}), home ML "
          f"{within['home_ml']['roi']:+.2%} "
          f"(|z| {abs(within['home_ml']['z']):.2f}). Descriptive sensitivity.",
          "", "## Model diagnostic (2025-26 holdout)", "",
          f"Canonical + the cell indicator, trained on "
          f"{diagnostic['n_train']:,} rows and scored on "
          f"{diagnostic['n_test']:,} ({diagnostic['n_test_in_cell']} in the cell). "
          f"Delta {diagnostic['delta']['delta_candidate_minus_reference']:+.6f} nats, "
          f"|z| {diagnostic['delta']['abs_z']:.2f}; indicator coefficient "
          f"{diagnostic['indicator_coef']:+.4f}. Positive control (availability "
          f"on minus off): "
          f"{diagnostic['positive_control_availability_on_minus_off']['delta_candidate_minus_reference']:+.6f} "
          f"nats, |z| "
          f"{diagnostic['positive_control_availability_on_minus_off']['abs_z']:.2f}.", "",
          f"**{verdicts['model_diagnostic']}**", "",
          "## What a null means", "",
          "At 965 bets the bootstrap SE is about 3%, so the |z| > 3 gate needs "
          "an ROI near 9%. A null here rules out the large edge the spot is "
          "usually sold as. It does not rule out a 2-3% edge, and a positive "
          "point estimate below the bar is not evidence of one.", ""]
    OUT_MD.write_text("\n".join(L))

    ms.append_jsonl(JSONL, {
        "name": "I74-run-form-mismatch",
        "protocol_ref": PROTOCOL,
        "declared_utc": ms.utc_now(),
        "experiment_type": "issue #74: form-mismatch spots, pooled meta-replication + 2025-26 diagnostic",
        "population": payload["population"],
        "primary_home_ats": payload["primary_home_ats"],
        "pooled": payload["pooled"],
        "mirror": payload["mirror"],
        "gap_ladder": payload["gap_ladder"],
        "per_season": payload["per_season"],
        "season_2025_26": payload["season_2025_26"],
        "within_season_form": payload["within_season_form"],
        "model_diagnostic": diagnostic,
        "verdicts": verdicts,
        "status": "executed",
    })

    print(f"pooled priced games {len(j):,}; cell {int(exact.sum()):,}")
    print(f"primary home ATS: n={primary['n_bets']}, ROI {primary['roi']:+.2%}, "
          f"|z| {abs(primary['z']):.2f}")
    print(f"model diagnostic: {diagnostic['delta']['delta_candidate_minus_reference']:+.6f} "
          f"nats, |z| {diagnostic['delta']['abs_z']:.2f}")
    print(f"wrote {OUT_JSON.name} and {OUT_MD.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
