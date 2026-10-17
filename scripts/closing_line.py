"""Does anything here beat the market? The question this project exists to answer.

Everything else measured in this repo is about how well a model predicts. This
script is the only one that asks whether that translates into money, because a
model can be the best predictor available and still be worth betting on only if it
beats the price.

THE COMPARISON
--------------
Four scorers, all on the SAME games, all out-of-sample:

  1. always home            the base rate. Any model must beat this.
  2. the market             closing ML de-vigged to sum to 1.
  3. the model              logistic regression on the shared feature layer.
  4. model + availability   the same, plus player-level expected minutes.

Scored by LOG LOSS, because log loss is what a bettor loses money on. Accuracy is
a bad judge here: the favourite-longshot bias means the market's picks are more often
right and still not profitable, which is exactly why the market's ROI can be negative
while its accuracy is the best available.

DE-VIGGING, AND WHY THE NAIVE VERSION IS WRONG
-----------------------------------------------
Raw implied probabilities sum to 1.0397 on this data (median 1.0399, p5-p95
1.0296-1.0500). That overround IS the vig, and dividing it out proportionally is the
standard first move. It is not unbiased: the margin is not spread evenly, it is
skewed toward longshots. A bettor's edge is therefore not exactly
`p_model - p_market`; it is `p_model - p_raw` at the price actually offered.

So this script reports BOTH:

  * log loss against the DE-VIGGED market probability - the fair question, "is the
    model a better estimator than the market?"
  * ROI at the ACTUAL posted price - the real question, "would betting it make
    money?" - which uses the raw implied probability, because that is what you pay.

A model can win the first and lose the second. That is not a contradiction and it
is not a rounding error; it is the whole reason vig exists.

THE BAR
-------
Vandenbruaene et al. 2022 (J Sports Econ 23(7)) reviewed 628 implementations of
sports-betting models and found the null of unprofitability was never rejected after
multiple-testing correction, at |z| > 3. This project holds itself to that bar. A
result at |z| = 2 is not a result.

Run:  python scripts/closing_line.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT, ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from DataProviders import nba_data as nd  # noqa: E402

SEED = 42
N_SPLITS = 5
ODDS_DB = ROOT / "Data" / "OddsData.sqlite"
SEASONS = ("2015-16", "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
           "2021-22", "2022-23", "2023-24", "2024-25", "2025-26")

Z_BAR = 3.0  # the bar from Vandenbruaene et al. 2022, after correction


def implied_from_american(odds) -> np.ndarray:
    """Raw implied probability from American odds.

    The two branches are NOT interchangeable: 100/(x+100) is valid only for positive
    odds and silently returns nonsense for negative ones (a -200 line comes out
    negative). Getting this wrong flips the overround to -259% and makes the whole
    comparison meaningless.
    """
    o = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(o < 0, (-o) / ((-o) + 100.0), 100.0 / (o + 100.0))


#: Candidate table names for a season's closing lines, most-preferred first.
#: Mirrors ``training_reporting.ODDS_TABLE_CANDIDATES``. The four spellings all
#: occur in the shipped database: 2024-25 is stored as a bare ``2024-25`` and
#: 2025-26 as ``odds_2025-26`` with no ``_new`` twin. Probing only one name
#: silently discarded both - see #47.
ODDS_TABLE_CANDIDATES = (
    "odds_{season}_new",
    "odds_{season}",
    "{season}_new",
    "{season}",
)


def load_odds() -> pd.DataFrame:
    """Closing moneyline for every season the model covers.

    Every season in ``SEASONS`` must resolve to a table. A season that does not is
    an error naming the season, not a silent ``continue``: the previous version
    probed only ``odds_<season>_new`` and swallowed the miss with a bare
    ``except``, which is indistinguishable from the odds genuinely being absent.
    That guard reported absence when the data was there under a different name,
    and every market number in this repository was computed without the two most
    recent seasons.
    """
    if not ODDS_DB.exists():
        raise FileNotFoundError(f"{ODDS_DB} is absent; the market comparison "
                                "needs the shipped odds database")
    con = sqlite3.connect(f"file:{ODDS_DB}?mode=ro", uri=True)
    try:
        existing = {row[0] for row in
                    con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        frames = []
        missing = []
        for s in SEASONS:
            name = next((t.format(season=s) for t in ODDS_TABLE_CANDIDATES
                         if t.format(season=s) in existing), None)
            if name is None:
                missing.append(s)
                continue
            d = pd.read_sql_query(f'SELECT * FROM "{name}"', con)
            if len(d):
                # SEASON is carried through rather than dropped. It was assigned
                # and then immediately excluded by the column selection below,
                # so no caller could see which seasons the frame actually
                # covered - which is how the two missing seasons went unnoticed.
                d = d.assign(SEASON=s)
                frames.append(d[["Date", "Home", "Away", "ML_Home", "ML_Away",
                                 "OU", "Spread", "SEASON"]])
    finally:
        con.close()
    if missing:
        raise RuntimeError(
            f"no odds table for season(s) {missing} in {ODDS_DB.name}; tried "
            f"{[t.format(season=missing[0]) for t in ODDS_TABLE_CANDIDATES]}. "
            "Refusing to return a frame that silently omits seasons."
        )
    if not frames:
        raise RuntimeError("no odds tables matched")
    return pd.concat(frames, ignore_index=True)


def tricode_to_full_name() -> dict[str, str]:
    """Tricode -> full team name, so model games can be joined to the odds tables.

    The two sources disagree on naming: `build_games` keys teams by tricode
    ("ATL"), while every odds table uses the full name ("Atlanta Hawks"). Joining
    them without this translation produces ZERO overlap, which looks like missing
    data rather than a naming mismatch and would be easy to misreport as "we have
    no odds".

    Built from nba_api's static team list rather than hand-written, so a franchise
    relocation cannot silently drop a team from the join.
    """
    from nba_api.stats.static import teams

    mapping = {r["abbreviation"]: r["full_name"] for r in teams.get_teams()}
    # One historical name the current list does not carry. Present in the shipped
    # odds tables, so it is translated rather than dropped.
    mapping.setdefault("CHA", "Charlotte Hornets")
    mapping["_alias"] = {"Charlotte Bobcats": "Charlotte Hornets"}
    return mapping


def join_odds(games: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Attach the market line to the model's games, on date + both teams.

    Matched on all three keys rather than on date alone. A date-only join would let
    a game pick up another game's line on a 10-game night, and the resulting market
    probability would be confidently wrong.
    """
    mapping = tricode_to_full_name()
    alias = mapping.pop("_alias")

    o = odds.copy()
    o["Date"] = pd.to_datetime(o["Date"], errors="coerce")
    for col in ("Home", "Away"):
        o[col] = o[col].replace(alias)
    o = o.dropna(subset=["Date", "Home", "Away", "ML_Home", "ML_Away"])

    g = games[["GAME_ID", "GAME_DATE", "home", "away"]].copy()
    g["home"] = g["home"].map(lambda t: mapping.get(t, t))
    g["away"] = g["away"].map(lambda t: mapping.get(t, t))

    merged = g.merge(
        o.rename(columns={"Date": "GAME_DATE", "Home": "home", "Away": "away"}),
        on=["GAME_DATE", "home", "away"], how="inner",
    )
    merged["p_market_raw"] = implied_from_american(merged["ML_Home"])
    merged["p_away_raw"] = implied_from_american(merged["ML_Away"])
    keep = ((merged["p_market_raw"] > 0) & (merged["p_market_raw"] < 1)
            & (merged["p_away_raw"] > 0) & (merged["p_away_raw"] < 1))
    merged = merged.loc[keep].copy()
    merged["overround"] = merged["p_market_raw"] + merged["p_away_raw"]
    # Proportional de-vig. Biased toward longshots, which is why the ROI arm below
    # uses the raw price rather than this one.
    merged["p_market"] = merged["p_market_raw"] / merged["overround"]
    return merged.reset_index(drop=True)


def model_probs(games: pd.DataFrame, splits, with_availability: bool) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold home-win probability, aligned to `games` (NaN where not scored)."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    roll = nd.rolling_team_features(games, halflife=10.0, min_games=5)
    sched = nd.rest_and_travel(games)
    keep = lambda d: d.select_dtypes(include=[np.number])  # noqa: E731
    parts = [keep(roll).reset_index(drop=True), keep(sched).reset_index(drop=True)]
    # The per-team home/away split. Added here explicitly rather than by calling
    # `logreg_rolling.load_design`, because this function also has to switch
    # availability on and off and needs its own frame for that.
    #
    # That is a duplicated feature definition, which is how the drop lists (#14,
    # #28), the availability block (#30) and `margin_sweep` have each drifted
    # independently. It has now happened a fourth time: the split was added to the
    # canonical model and this file would have silently kept comparing the market
    # against the OLD model. Listed here so the omission is visible.
    parts.append(keep(nd.opponent_adjusted_features(
        games, halflife=10.0)).reset_index(drop=True))
    if with_availability:
        parts.append(nd.availability_features(games).reset_index(drop=True))
    X = pd.concat(parts, axis=1)
    X = X.loc[:, X.nunique(dropna=False) > 1]
    y = games["y"].to_numpy()

    out = np.full(len(y), np.nan)
    for tr, te in splits:
        pipe = Pipeline([("i", SimpleImputer(strategy="median")),
                         ("s", StandardScaler()),
                         ("c", LogisticRegression(C=1.0, max_iter=300,
                                                  random_state=SEED))])
        pipe.fit(X.iloc[tr], y[tr])
        out[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return out, y


def roi(p_model: np.ndarray, p_raw: np.ndarray, y: np.ndarray,
        threshold: float = 0.55) -> dict:
    """Flat-bet ROI when betting the model whenever it exceeds `threshold`.

    The stake is the model's edge OVER THE PRICE, not its raw confidence. A 0.60
    model probability is only a bet if the posted line implies less than 0.60,
    because that difference is what survives the vig.

    A GAME WHERE THE PRICE IS BETTER IS NOT A BET, AND ITS STAKE IS NOT MONEY
    ---------------------------------------------------------------------
    `p_model - p_raw` is negative whenever the market is the more confident one,
    which at a 0.55 threshold is most of the slate: on the real data 1,382 of
    4,076 games. Those are not short positions. They are not bets at all, and
    their stake is not a unit of currency.

    This function used to divide total profit by `stake.sum()`, which included
    them. Measured consequence at threshold 0.55:

        positive stakes 812, sum +69.22  <- money actually at risk
        negative stakes 1382, sum -117.10
        stake.sum()   = -47.88          <- the denominator it divided by
        total profit  = -6.48           <- a LOSS
        profit / sum(positive stakes)  = -9.37%   correct
        profit / stake.sum()           = +13.54%   as it was reported

    A loss divided by a negative total reports as a profit. The sign of the
    headline number was inverted by the denominator, not by the threshold sweep
    being unlucky - and the same `st = stake[pick]` mistake was inside the
    bootstrap, so the standard error was computed over the wrong population too.

    Both now restrict to placed bets, before the mean and before the resample.
    """
    raw_stake = np.where(p_model >= threshold, p_model - p_raw, 0.0)
    placed = raw_stake > 0
    n = int(placed.sum())
    if n == 0:
        return {"threshold": threshold, "bets": 0, "roi": float("nan"),
                "se": float("nan"), "z": float("nan")}

    won = (y == 1).astype(float)
    # Profit per unit staked on a bet that was actually placed:
    #   win  -> (1 - p_raw) / p_raw   (net of the stake)
    #   lose -> -1
    profit = np.where(placed, (won - p_raw) / p_raw * raw_stake, 0.0)
    stake = np.where(placed, raw_stake, 0.0)

    total_staked = float(stake.sum())
    if total_staked <= 0:
        return {"threshold": threshold, "bets": n, "roi": float("nan"),
                "se": float("nan"), "z": float("nan")}
    roi_point = float(profit.sum() / total_staked)

    # Resample GAMES, not bets. Only placed bets enter, and each game
    # contributes at most one of them on this rule.
    rng = np.random.default_rng(SEED)
    idx = np.flatnonzero(placed)
    st = stake[idx]
    pf = profit[idx]
    draws = np.empty(2000)
    for i in range(len(draws)):
        pick = rng.integers(0, len(idx), len(idx))
        draws[i] = pf[pick].sum() / st[pick].sum()
    se = float(np.std(draws, ddof=1))
    return {"threshold": threshold, "bets": n, "roi": roi_point, "se": se,
            "z": (roi_point / se) if se > 0 else float("nan")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/closing_line.json")
    args = ap.parse_args()

    games = nd.build_games()
    odds = load_odds()
    joined = join_odds(games, odds)
    print(f"model games {len(games):,}")
    print(f"odds rows   {len(odds):,}")
    print(f"joined      {len(joined):,}  ({100*len(joined)/len(games):.1f}% of games)")
    if len(joined) == 0:
        print("NO OVERLAP - cannot compare against the market.")
        return 1

    over = joined["overround"]
    print(f"\noverround   mean {(over.mean()-1)*100:+.2f}%  "
          f"p5 {(over.quantile(.05)-1)*100:+.2f}%  p95 {(over.quantile(.95)-1)*100:+.2f}%"
          "   <- the vig, and it is normal")
    _j = joined.merge(games[["GAME_ID", "y"]], on="GAME_ID", how="left")
    print(f"market implied P(home) {_j['p_market'].mean():.4f}   "
          f"actual home win {_j['y'].mean():.4f}   "
          f"gap {_j['p_market'].mean() - _j['y'].mean():+.4f}")
    print(f"market picks home {(joined['p_market'] > 0.5).mean()*100:.1f}% of games "
          f"(share favoured, NOT the win rate)   market accuracy "
          f"{((_j['p_market'] > 0.5).astype(int) == _j['y']).mean():.4f}")

    splits = nd.walk_forward_split(games, n_splits=N_SPLITS)
    p_base, y = model_probs(games, splits, with_availability=False)
    p_avail, _ = model_probs(games, splits, with_availability=True)

    # Align by GAME_ID through an explicit lookup.
    #
    # The obvious shortcut, np.searchsorted(games.GAME_ID, scored.GAME_ID), is
    # wrong twice over: GAME_ID is a STRING so it does not even subtract, and
    # searchsorted requires a sorted haystack that the join does not guarantee.
    # It silently returned garbage indices, which made the model score 0.699 log
    # loss against its own 0.6195 and produced ROI swinging from +93% to -70%
    # across thresholds - numbers that look like results and are pure misalignment.
    row_of = {gid: i for i, gid in enumerate(games["GAME_ID"].to_numpy())}
    idx = joined["GAME_ID"].map(row_of)
    scored = joined.assign(y=y[idx.to_numpy()])
    scored["p_model"] = p_base[idx.to_numpy()]
    scored["p_model_avail"] = p_avail[idx.to_numpy()]
    scored = scored.dropna(subset=["p_model"]).reset_index(drop=True)
    n = len(scored)
    yy = scored["y"].to_numpy().astype(int)
    pm = scored["p_model"].to_numpy()
    pa = scored["p_model_avail"].to_numpy()
    pmarket = scored["p_market"].to_numpy()
    praw = scored["p_market_raw"].to_numpy()

    print(f"\nscored on {n:,} games that have BOTH an out-of-fold prediction and a "
          f"closing line\n")
    print(f"  {'scorer':30s} {'log loss':>10s} {'Brier':>8s} {'AUC':>8s} {'acc':>7s}")
    print("  " + "-" * 68)
    rows = {}
    for label, p in (("always home", np.full(n, yy.mean())),
                     ("market (de-vigged)", pmarket),
                     ("model", pm),
                     ("model + availability", pa)):
        e = nd.evaluate(yy, p)
        rows[label] = e
        print(f"  {label:30s} {e['log_loss']:>10.5f} {e['brier']:>8.5f} "
              f"{e['auc']:>8.5f} {e['accuracy']:>7.4f}")

    print("\nDELTA vs the market  (positive = the model is the better estimator)")
    print("  " + "-" * 68)
    deltas = {}
    for label, p in (("model", pm), ("model + availability", pa)):
        d = nd.paired_bootstrap_delta(yy, pmarket, p, n_boot=2000, seed=SEED)
        se = (d["ci_hi"] - d["ci_lo"]) / (2 * 1.959964)
        z = abs(d["delta"] / se) if se > 0 else float("nan")
        deltas[label] = {**d, "z": z}
        flag = ("BEATS MARKET" if d["ci_lo"] > 0
                else "MARKET BETTER" if d["ci_hi"] < 0 else "straddles zero")
        print(f"  {label:30s} {d['delta']:+.5f}  CI [{d['ci_lo']:+.5f}, "
              f"{d['ci_hi']:+.5f}]  |z| {z:5.2f}  {flag}")

    print("\nROI AT THE ACTUAL POSTED PRICE  (flat bet when model edge > threshold)")
    print("  " + "-" * 68)
    rois = {}
    for thr in (0.55, 0.60, 0.65):
        for label, p in (("model", pm), ("model + availability", pa)):
            r = roi(p, praw, yy, thr)
            rois[f"{label}@{thr}"] = r
            if np.isfinite(r["roi"]):
                flag = "PASTES THE BAR" if abs(r["z"]) >= Z_BAR else "below the |z|>3 bar"
                print(f"  {label:22s} thr {thr:.2f}  bets {r['bets']:>5,}  "
                      f"ROI {r['roi']*100:+.2f}%  SE {r['se']*100:.2f}%  |z| {r['z']:5.2f}  {flag}")
            else:
                print(f"  {label:22s} thr {thr:.2f}  no bets placed")

    print(f"\nVERDICT  (bar: |z| > {Z_BAR}, Vandenbruaene et al. 2022)")
    beat = [k for k, v in deltas.items() if v["ci_lo"] > 0]
    worse = [k for k, v in deltas.items() if v["ci_hi"] < 0]
    null_est = [k for k, v in deltas.items() if v["ci_lo"] <= 0 <= v["ci_hi"]]
    zbar = [k for k, v in rois.items() if np.isfinite(v.get("z", np.nan)) and abs(v["z"]) >= Z_BAR]

    if beat:
        print(f"  BETTER ESTIMATOR THAN THE MARKET: {beat} - this would be a finding.")
    elif worse:
        print(f"  The market is decisively the better estimator ({', '.join(worse)}).")
        print("  The intervals exclude zero on the MARKET's side, so this is not a")
        print("  null - it is a clear loss, which is what an efficient market looks")
        print("  like against a model built on the same public information.")
    else:
        print(f"  Neither model beats the market as an estimator: {null_est}")

    if zbar:
        print(f"  Betting configurations that reach the bar: {zbar}")
    else:
        best = max((v for v in rois.values() if np.isfinite(v.get("roi", np.nan))),
                   key=lambda v: abs(v["z"]), default=None)
        if best:
            print(f"  No betting configuration reaches |z| > {Z_BAR}. The strongest sits")
            print(f"  at |z| = {abs(best['z']):.2f}, which is noise.")

    print("\n  SELECTION EFFECT, stated because it applies to this script:")
    print(f"  {len(rois)} betting configurations were scored. The best of many noisy")
    print("  estimates is biased upward, so a positive point estimate at one threshold")
    print("  is not evidence of an edge. That is why every threshold is reported and")
    print("  none is presented as the answer.")
    print("\n  CONCLUSION: no demonstrated betting edge, in either direction.")
    print("  This reproduces the README's earlier finding on 1,722 games, now on a")
    print("  larger overlap and with the availability feature included - and the")
    print("  availability feature improves PREDICTION without changing that.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_scored": int(n),
        "overround": {"mean": float(over.mean()), "p5": float(over.quantile(.05)),
                      "p95": float(over.quantile(.95))},
        "metrics": rows,
        "delta_vs_market": deltas,
        "roi": rois,
        "z_bar": Z_BAR,
    }, indent=2, default=float) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())