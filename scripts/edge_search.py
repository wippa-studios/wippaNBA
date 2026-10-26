"""Pre-registered search loop over the repo's own closing-line archive.

WHY THIS EXISTS
---------------
The project has spent years testing one hypothesis at a time and closing each
with a null. That is the right way to avoid p-hacking, but it is a slow way to
find the one structural soft spot that may exist. This script runs a *family* of
related arms in one pass, with the family registered before the numbers are
computed, every cell reported, and a multiplicity bar fixed in advance.

THE SPLIT
---------
Discovery: 2015-16 .. 2024-25.
Holdout:   2025-26.

Nothing in the holdout chooses an arm, a threshold, or a side. Arms are fixed in
the `FAMILIES` registry below; the script writes a pre-registration record to
`results/experiments.jsonl` before it reads a single price.

THE HOLD OUT SOURCE
-------------------
The shipped `odds_2025-26` table is a single early snapshot that stops on
2026-01-07 (514 priced games), so as a holdout it is both small and not a close.
`odds_2025-26_espn` is a separate, full-season closing-line table backfilled from
ESPN's core odds API (provider preference DraftKings -> ESPN BET -> any non-live
provider with a close). The old table is left untouched so #74/#81 stay
reproducible; only this harness swaps the holdout.

THE BAR
-------
The repo's promotion bar is |z| > 3 on a bootstrap over bets. A family with m
arms raises that bar to the two-sided Bonferroni value
`norm.ppf(1 - 0.05 / (2m))`, never below 3.0. A discovery cell above the bar is
a *candidate*; promotion also needs the holdout to be positive. Every cell is
reported, including the negative ones.

USAGE
-----
    python scripts/edge_search.py --family price_structure --preregister
    python scripts/edge_search.py --family price_structure

The first command appends the pre-registration and exits; the second computes
and writes `results/edge_search/<family>.{json,md}` plus a ledger record.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

import closing_line as cl                      # noqa: E402
from c44_spread_total import effective_line    # noqa: E402
from DataProviders import nba_data as nd       # noqa: E402

OUT_DIR = ROOT / "results" / "edge_search"
LEDGER = ROOT / "results" / "experiments.jsonl"
HOLDOUT = "2025-26"
ESPN_TABLE = "odds_2025-26_espn"
QUOTE_COLS = ["Date", "Home", "Away", "ML_Home", "ML_Away", "OU", "Spread"]
N_BOOT = 2000
SEED = 42
ATS_PAYOUT = 100.0 / 110.0
ATS_BREAK_EVEN = 110.0 / 210.0


# --------------------------------------------------------------------------- #
# Frame
# --------------------------------------------------------------------------- #
def read_espn_table(db_path: Path | str = cl.ODDS_DB) -> pd.DataFrame:
    """The full-season 2025-26 close, in `cl.load_odds`'s quote columns."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return pd.read_sql(f"SELECT {', '.join(QUOTE_COLS)} FROM '{ESPN_TABLE}'", con)
    finally:
        con.close()


def swap_holdout(base: pd.DataFrame, espn: pd.DataFrame,
                 season: str = HOLDOUT) -> pd.DataFrame:
    """Replace `season`'s rows in a load_odds frame with the ESPN close.

    The two sources are never mixed: a season's prices must come from one place,
    or the de-vig and the ROI denominator are computed from different books. The
    old rows are dropped whole rather than de-duplicated, and the ESPN keys must
    already be unique.
    """
    missing = [c for c in QUOTE_COLS if c not in espn.columns]
    if missing:
        raise ValueError(f"{ESPN_TABLE} is missing columns {missing}")
    dup = espn.duplicated(subset=["Date", "Home", "Away"], keep=False)
    if dup.any():
        sample = espn.loc[dup, ["Date", "Home", "Away"]].head(5).to_dict("records")
        raise ValueError(f"{ESPN_TABLE} has duplicate game keys, e.g. {sample}")
    keep = base.loc[base["SEASON"] != season, QUOTE_COLS + ["SEASON"]].copy()
    replacement = espn.loc[:, QUOTE_COLS].copy()
    replacement["SEASON"] = season
    return pd.concat([keep, replacement], ignore_index=True)


def load_search_odds() -> pd.DataFrame:
    """`cl.load_odds()` with the 2025-26 snapshot replaced by the ESPN close."""
    return swap_holdout(cl.load_odds(), read_espn_table())


def priced_frame() -> tuple[pd.DataFrame, dict]:
    """One row per game with a closing line, discovery/holdout labels."""
    g = nd.build_games().reset_index(drop=True)
    odds = load_search_odds()
    j = cl.join_odds(g, odds)
    # `join_odds` already carries SEASON; merging it in again is what produced
    # SEASON_x/SEASON_y and broke `effective_line`. Only outcomes are merged.
    j = j.merge(g[["GAME_ID", "margin", "total", "y"]], on="GAME_ID", how="left")
    j = effective_line(j)
    for c in ("ML_Home", "ML_Away", "OU", "Spread", "margin", "total", "y"):
        j[c] = pd.to_numeric(j[c], errors="coerce")
    # A 0 moneyline is a missing price, not even money; it would divide by zero.
    zero_ml = (j[["ML_Home", "ML_Away"]] == 0).any(axis=1)
    j = j.loc[~zero_ml]
    j = j.dropna(subset=["p_market", "L_eff", "margin", "total", "y",
                         "ML_Home", "ML_Away", "OU"]).reset_index(drop=True)
    j["split"] = np.where(j["SEASON"] < HOLDOUT, "discovery",
                          np.where(j["SEASON"] == HOLDOUT, "holdout", "other"))
    stats = {
        "games_in": int(len(g)),
        "priced_games": int(len(j)),
        "dropped_zero_moneyline": int(zero_ml.sum()),
        "discovery": int((j["split"] == "discovery").sum()),
        "holdout": int((j["split"] == "holdout").sum()),
        "holdout_source": f"Data/OddsData.sqlite::{ESPN_TABLE}",
        "seasons": j.groupby("SEASON").size().to_dict(),
    }
    return j, stats


# --------------------------------------------------------------------------- #
# Settlement
# --------------------------------------------------------------------------- #
def _side_mask(j: pd.DataFrame, side_home) -> np.ndarray:
    if np.ndim(side_home) == 0:
        return np.full(len(j), bool(side_home))
    return np.asarray(side_home, dtype=bool)


def settle_ats(j: pd.DataFrame, side_home) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat -110 ATS: profit, cover, push. A push returns the stake."""
    home = _side_mask(j, side_home)
    margin = j["margin"].to_numpy(dtype=float)
    line = j["L_eff"].to_numpy(dtype=float)
    win = np.where(home, margin > line, margin < line)
    push = margin == line
    profit = np.where(push, 0.0, np.where(win, ATS_PAYOUT, -1.0))
    return profit, win, push


def settle_ml(j: pd.DataFrame, side_home) -> tuple[np.ndarray, np.ndarray]:
    """Posted-price moneyline for the chosen side, vig included."""
    home = _side_mask(j, side_home)
    price = np.where(home, j["ML_Home"].to_numpy(dtype=float),
                     j["ML_Away"].to_numpy(dtype=float))
    if np.any(price == 0):
        raise ValueError("a zero moneyline reached settlement")
    dec = np.where(price < 0, 1.0 + 100.0 / (-price), 1.0 + price / 100.0)
    won = np.where(home, j["y"].to_numpy() == 1, j["y"].to_numpy() == 0)
    return np.where(won, dec - 1.0, -1.0), won


def settle_total(j: pd.DataFrame, side_over) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flat -110 total: profit, win, push. A push returns the stake."""
    over = bool(side_over)
    points = j["total"].to_numpy(dtype=float)
    line = j["OU"].to_numpy(dtype=float)
    win = points > line if over else points < line
    push = points == line
    profit = np.where(push, 0.0, np.where(win, ATS_PAYOUT, -1.0))
    return profit, win, push


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def boot(profit: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    p = np.asarray(profit, dtype=float)
    if not len(p):
        return {"n": 0, "roi": float("nan"), "se": float("nan"),
                "z": float("nan"), "ci": [float("nan"), float("nan")]}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(p), size=(n_boot, len(p)))
    means = p[idx].mean(axis=1)
    roi = float(p.mean())
    se = float(means.std(ddof=1))
    return {"n": int(len(p)), "roi": roi, "se": se,
            "z": float(roi / se) if se else float("nan"),
            "ci": [float(np.percentile(means, 2.5)),
                   float(np.percentile(means, 97.5))]}


def stats(j: pd.DataFrame, mask: np.ndarray, kind: str, side) -> dict:
    """Score one settled side on the rows selected by `mask`."""
    sub = j.loc[mask].reset_index(drop=True)
    side_masked = (np.asarray(side)[mask] if np.ndim(side) else side)
    if kind == "ats":
        p, win, push = settle_ats(sub, side_masked)
        out = boot(p)
        out["n_pushes"] = int(push.sum())
        out["win_rate"] = float(win.sum() / max((~push).sum(), 1))
        out["break_even"] = ATS_BREAK_EVEN
    elif kind == "ml":
        p, won = settle_ml(sub, side_masked)
        out = boot(p)
        out["n_pushes"] = 0
        out["win_rate"] = float(won.mean()) if len(won) else float("nan")
        out["break_even"] = float("nan")
    elif kind == "total":
        p, win, push = settle_total(sub, side_masked)
        out = boot(p)
        out["n_pushes"] = int(push.sum())
        out["win_rate"] = float(win.sum() / max((~push).sum(), 1))
        out["break_even"] = ATS_BREAK_EVEN
    else:
        raise ValueError(kind)
    out["kind"] = kind
    out["n_rows"] = int(len(sub))
    return out


def arm(j: pd.DataFrame, mask: np.ndarray, kind: str, side, label: str) -> dict:
    """Score one arm on discovery, holdout, and pooled."""
    d = stats(j, mask & (j["split"] == "discovery").to_numpy(), kind, side)
    h = stats(j, mask & (j["split"] == "holdout").to_numpy(), kind, side)
    p = stats(j, mask & (j["split"] != "other").to_numpy(), kind, side)
    return {"label": label, "kind": kind, "discovery": d,
            "holdout": h, "pooled": p}


def bar_z(m: int) -> float:
    return float(max(3.0, norm.ppf(1.0 - 0.05 / (2.0 * max(m, 1)))))


def verdict(a: dict, bar: float) -> str:
    d, h = a["discovery"], a["holdout"]
    if not d["n"] or not h["n"]:
        return "no data"
    if d["roi"] > 0 and abs(d["z"]) > bar and h["roi"] > 0:
        return "candidate (discovery clears; holdout positive)"
    if d["roi"] > 0 and abs(d["z"]) > bar:
        return "discovery-only"
    if d["roi"] > 0 and h["roi"] > 0:
        return "same sign, underpowered"
    return "no evidence"


# --------------------------------------------------------------------------- #
# Families
# --------------------------------------------------------------------------- #
def family_price_structure(j: pd.DataFrame) -> tuple[list[dict], dict]:
    """Where is the closing market soft, if anywhere?

    A wide descriptive scan, not a candidate hunt: 42 arms put the bar at about
    3.24, and the holdout is one season. Its job is to establish the sign and
    size of every structural cell so later families do not re-test the same
    thing under a new name.
    """
    arms = []
    n = len(j)
    ones = np.ones(n, dtype=bool)
    zeros = np.zeros(n, dtype=bool)

    # A1/A2: moneyline favourite and underdog by de-vigged favourite probability.
    p_home = j["p_market"].to_numpy(dtype=float)
    fav_prob = np.maximum(p_home, 1.0 - p_home)
    fav_home = p_home > 0.5
    buckets = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70),
               (0.70, 0.75), (0.75, 0.80), (0.80, 1.01)]
    for lo, hi in buckets:
        m = (fav_prob >= lo) & (fav_prob < hi)
        arms.append(arm(j, m, "ml", fav_home, f"ML favourite {lo:.2f}-{hi:.2f}"))
        arms.append(arm(j, m, "ml", ~fav_home, f"ML underdog {lo:.2f}-{hi:.2f}"))

    # A3/A4: home and away ATS by absolute closing line.
    abs_line = j["L_eff"].abs().to_numpy()
    line_buckets = [(0.0, 0.5), (0.5, 2.5), (2.5, 4.5), (4.5, 6.5),
                    (6.5, 8.5), (8.5, 10.5), (10.5, 12.5), (12.5, 99.0)]
    for lo, hi in line_buckets:
        m = (abs_line >= lo) & (abs_line < hi)
        arms.append(arm(j, m, "ats", ones, f"home ATS |line| {lo}-{hi}"))
        arms.append(arm(j, m, "ats", zeros, f"away ATS |line| {lo}-{hi}"))

    # A5/A6: over and under by closing total.
    ou = j["OU"].to_numpy()
    total_buckets = [(0, 200), (200, 210), (210, 220), (220, 230), (230, 999)]
    for lo, hi in total_buckets:
        m = (ou >= lo) & (ou < hi)
        arms.append(arm(j, m, "total", True, f"over total {lo}-{hi}"))
        arms.append(arm(j, m, "total", False, f"under total {lo}-{hi}"))

    # A7/A8: dog vs favourite ATS. A pick'em has no underdog and is excluded.
    line = j["L_eff"].to_numpy()
    home_dog = line < 0
    home_fav = line > 0
    dog_mask = home_dog | home_fav
    dog_side_home = home_dog  # home is the dog exactly when the home line is negative
    arms.append(arm(j, dog_mask, "ats", dog_side_home, "back the underdog ATS"))
    arms.append(arm(j, dog_mask, "ats", ~dog_side_home, "back the favourite ATS"))

    assert len(arms) == 42, len(arms)
    meta = {
        "family": "price_structure",
        "question": "Where does the closing market show a structural bias?",
        "arms": len(arms),
        "bar_z": bar_z(len(arms)),
        "population": int(len(j)),
        "discovery": int((j["split"] == "discovery").sum()),
        "holdout": int((j["split"] == "holdout").sum()),
        "holdout_source": f"{ESPN_TABLE} (ESPN core odds close)",
    }
    return arms, meta


FAMILIES = {
    "price_structure": family_price_structure,
}


# --------------------------------------------------------------------------- #
# Registration and output
# --------------------------------------------------------------------------- #
def prereg_record(family: str) -> dict:
    if family == "price_structure":
        arms = [
            "ML favourite and underdog by de-vigged favourite probability: "
            "0.50-0.55, 0.55-0.60, 0.60-0.65, 0.65-0.70, 0.70-0.75, 0.75-0.80, 0.80+",
            "home and away ATS by |L_eff|: 0-0.5, 0.5-2.5, 2.5-4.5, 4.5-6.5, "
            "6.5-8.5, 8.5-10.5, 10.5-12.5, 12.5+",
            "over and under by closing total: <200, 200-210, 210-220, 220-230, 230+",
            "back the underdog ATS vs back the favourite ATS (home line sign; "
            "pick'em excluded)",
        ]
        m = 42
    else:
        raise SystemExit(f"unknown family {family}")
    return {
        "name": f"E01-prereg-{family}",
        "experiment_type": f"search loop family 1: {family}",
        "declared_utc": datetime.now(timezone.utc).isoformat(),
        "split": {"discovery": "2015-16..2024-25", "holdout": HOLDOUT},
        "holdout_source": f"{ESPN_TABLE} (ESPN core odds close), old "
                          f"odds_{HOLDOUT} table left untouched",
        "arms": arms,
        "n_arms": m,
        "bar_z": bar_z(m),
        "settlement": "ATS and totals at flat -110, side-aware, pushes at 0; "
                      "moneyline at the posted price, vig included",
        "decision": "candidate if discovery ROI > 0 and |z| > bar; promotion "
                    "also requires holdout ROI > 0. Every cell is reported.",
        "status": "registered",
    }


def _append(record: dict) -> None:
    with open(LEDGER, "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


def write_outputs(family: str, arms: list[dict], meta: dict,
                  frame_stats: dict | None = None) -> dict:
    bar = meta["bar_z"]
    for a in arms:
        a["verdict"] = verdict(a, bar)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "family": family,
        "meta": meta,
        "frame": frame_stats or {},
        "arms": arms,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"{family}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (OUT_DIR / f"{family}.md").write_text("\n".join(_markdown(payload)) + "\n")

    rec = {
        "name": f"E01-run-{family}",
        "experiment_type": f"search loop family 1: {family}",
        "declared_utc": payload["generated_utc"],
        "meta": meta,
        "frame": frame_stats or {},
        "arms": arms,
        "status": "executed",
    }
    _append(rec)
    return payload


def _pct(x: float) -> str:
    return f"{x:+.2%}" if np.isfinite(x) else "-"


def _z(x: float) -> str:
    return f"{abs(x):.2f}" if np.isfinite(x) else "-"


def _win(x: float) -> str:
    return f"{x:.1%}" if np.isfinite(x) else "-"


def _markdown(p: dict) -> list[str]:
    m = p["meta"]
    f = p.get("frame") or {}
    L = [f"# Search family: {p['family']}", "",
         f"Generated {p['generated_utc']}. Pre-registered in "
         "`results/experiments.jsonl`.", "",
         f"**Question.** {m['question']}", "",
         f"Discovery **{m['discovery']:,}** priced games (2015-16..2024-25); "
         f"holdout **{m['holdout']:,}** (2025-26, {m.get('holdout_source', '')}). "
         f"{m['arms']} arms; multiplicity bar **|z| > {m['bar_z']:.2f}** "
         "(Bonferroni, floor 3.0).", "",
         "| arm | n disc | ROI disc | \\|z\\| | cover disc | n hold | ROI hold "
         "| \\|z\\| | cover hold | verdict |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for a in p["arms"]:
        d, h = a["discovery"], a["holdout"]
        L.append(f"| {a['label']} | {d['n']:,} | {_pct(d['roi'])} | {_z(d['z'])} "
                 f"| {_win(d['win_rate'])} | {h['n']:,} | {_pct(h['roi'])} "
                 f"| {_z(h['z'])} | {_win(h['win_rate'])} | {a['verdict']} |")
    L += ["", "## Reading the table", "",
          "`cover` is wins divided by settled bets; pushes are excluded from the "
          "denominator but kept in `n` at zero profit. A cell is a *candidate* "
          "only when discovery is positive above the multiplicity bar. A "
          "*promoted* rule needs the holdout positive as well; with a one-season "
          "holdout that is a high bar, and a null holdout does not rescue a "
          "discovery cell. Cells below the bar are noise, including the positive "
          "ones.", ""]
    if f:
        L += ["## Frame", "",
              f"`build_games()` {f.get('games_in', 0):,} regular-season games; "
              f"**{f.get('priced_games', 0):,}** joined to a closing line; "
              f"discovery {f.get('discovery', 0):,}, holdout "
              f"{f.get('holdout', 0):,}. Holdout source: "
              f"`{f.get('holdout_source', '')}`.", ""]
    return L


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--preregister", action="store_true")
    args = ap.parse_args()

    if args.preregister:
        rec = prereg_record(args.family)
        _append(rec)
        print(f"registered {rec['name']} with bar |z| > {rec['bar_z']:.2f}")
        return 0

    j, frame_stats = priced_frame()
    arms, meta = FAMILIES[args.family](j)
    payload = write_outputs(args.family, arms, meta, frame_stats)
    print(f"{args.family}: {meta['arms']} arms, discovery {meta['discovery']:,}, "
          f"holdout {meta['holdout']:,}, bar |z| > {meta['bar_z']:.2f}")
    for a in payload["arms"]:
        if a["verdict"].startswith(("candidate", "discovery-only")):
            d, h = a["discovery"], a["holdout"]
            print(f"  {a['label']}: disc {_pct(d['roi'])} |z| {_z(d['z'])}; "
                  f"hold {_pct(h['roi'])} |z| {_z(h['z'])} -> {a['verdict']}")
    print(f"wrote {OUT_DIR / (args.family + '.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
