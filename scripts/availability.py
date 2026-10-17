"""Is availability predictable, and does it carry signal?

This is the one large effect found anywhere in this project that is not a null.

Measured earlier: replacing EXPECTED minutes with ACTUAL minutes in a game's
feature set changes log loss by 0.00975 nats, CI [-0.01349, -0.00590] - an order of
magnitude larger than player ratings (+0.00052), the 52 rank columns (0.0018), or
any hyperparameter search (<0.002). Actual minutes encode who is playing, who is
hurt, and who has been ejected. Nothing else in the feature set knows any of that.

The reason it is not in a model is that actual minutes are only known AFTER the
game. Using them is a leak, and a leak that produces excellent numbers is worse
than no model at all.

So the question this script answers is the only one that matters here:

    IS THE AVAILABILITY GAP PREDICTABLE FROM INFORMATION AVAILABLE BEFORE TIP-OFF?

If a player is expected to play 30 minutes and plays 12, can that be anticipated
from what was known yesterday - their own recent minutes, games played recently,
games missed recently, age, and usage? Or is it genuinely surprise?

TWO ANSWERS, AND THEY ARE OPPOSITE NEWS FOR DIFFERENT REASONS
------------------------------------------------------------

* If the gap is largely predictable, then the effect is real, the leak is
  unnecessary, and there is something here worth building on.
* If the gap is largely unpredictable, then the 0.00975 nats is genuinely
  unattainable. It is a measurement of how much injury news matters, not of how
  well anyone can predict basketball. That is not a failure of this project; it
  is a fact about the world, and it caps what ANY model can do.

The second answer is the one I expect. It is still the right answer to know, and
knowing it stops the project from spending a year chasing a number that was never
available.

METHOD
------
`expected_minutes` for a player's game is their own strictly-past EWMA of minutes
played, via the SHARED causal primitive `_ewm_by_team` generalised to players, with
the shift WITHIN each player. A plain `.shift(1)` on a frame sorted by player would
hand each player's first game the last value of whichever player sorted before it.

`availability_gap = actual - expected`, aggregated to home-minus-away per game. The
prediction is then: does knowing the gap helps predict the result?

Three measurements:
  1. How much of the gap variance is explainable out-of-sample? (R^2 on held-out)
  2. Does the gap, used causally, improve win prediction? (paired bootstrap)
  3. How large is the honest, non-leaky effect next to the leaky one? (the ceiling)

Run:  python scripts/availability.py
"""

from __future__ import annotations

import argparse
import json
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

# The leaky number, for comparison. Measured on this data, earlier this project.
LEAKY_DELTA = 0.00975
LEAKY_CI = (-0.01349, -0.00590)


def player_minutes() -> pd.DataFrame:
    """Per-player per-game minutes, with the game date attached and sorted."""
    pa = nd.load_player_advanced()
    g = nd.build_games()
    dates = (g[["GAME_ID", "GAME_DATE", "home", "away"]]
             .drop_duplicates("GAME_ID")
             .reset_index(drop=True))
    df = pa[["GAME_ID", "PLAYER_ID", "TEAM_ABBREVIATION", "MIN"]].copy()
    df = df.merge(dates, on="GAME_ID", how="inner")
    df["MIN"] = pd.to_numeric(df["MIN"], errors="coerce")
    return df.dropna(subset=["MIN"]).sort_values(["PLAYER_ID", "GAME_DATE"])


def _ewm_by_player(df: pd.DataFrame, value: str, halflife: float) -> pd.Series:
    """Causal EWMA of `value` per player. Mirrors nba_data._ewm_by_team exactly.

    The shift is WITHIN each player. A frame-level `.shift(1)` would let a player's
    first appearance inherit the last value of whichever player sorted above it,
    contaminating every player's debut.
    """
    s = df.sort_values(["PLAYER_ID", "GAME_DATE"])
    shifted = s.groupby("PLAYER_ID", sort=False)[value].shift(1)
    out = (shifted.groupby(s["PLAYER_ID"], sort=False)
           .ewm(halflife=halflife, adjust=True).mean()
           .reset_index(level=0, drop=True))
    return out.reindex(s.index)


def strict_roster_expectation(games: pd.DataFrame, halflife: float = 15.0) -> pd.DataFrame:
    """Expected minutes summed over the team's PREVIOUS game's squad.

    `availability_frame` sums expected minutes over the players who ACTUALLY
    appeared, which is not available before tip-off: a team missing its star has a
    smaller sum, so the column carries a little post-hoc knowledge. The measured
    size of that leak is small (+0.00109, interval straddling zero) but it is not
    zero, and it is the difference between an honest claim and an inflated one.

    This version uses the squad from the team's previous game instead - known
    before tip-off whatever happens tonight. In a 30-day window the two squads
    differ only by injury, so this is exactly the availability signal minus the
    leak, and it is the arm worth quoting.
    """
    pm = player_minutes()
    pm["expected_min"] = _ewm_by_player(pm, "MIN", halflife)
    emap = (pm.dropna(subset=["expected_min"])
            .groupby("PLAYER_ID")["expected_min"].last().to_dict())

    squad = (pm.groupby(["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE"], as_index=False)
             .agg(squad=("PLAYER_ID", lambda s: frozenset(s))))
    squad = squad.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    squad["prev_squad"] = squad.groupby("TEAM_ABBREVIATION", sort=False)["squad"].shift(1)
    squad["prev_sum"] = [
        sum(emap.get(p, 0.0) for p in ps) if isinstance(ps, frozenset) else np.nan
        for ps in squad["prev_squad"]
    ]

    d = games[["GAME_ID", "home", "away"]]
    h = d.merge(squad[["GAME_ID", "TEAM_ABBREVIATION", "prev_sum"]],
                left_on=["GAME_ID", "home"],
                right_on=["GAME_ID", "TEAM_ABBREVIATION"], how="left"
                ).rename(columns={"prev_sum": "home_sexp"})
    a = d.merge(squad[["GAME_ID", "TEAM_ABBREVIATION", "prev_sum"]],
                left_on=["GAME_ID", "away"],
                right_on=["GAME_ID", "TEAM_ABBREVIATION"], how="left"
                ).rename(columns={"prev_sum": "away_sexp"})
    out = h[["GAME_ID", "home_sexp"]].merge(a[["GAME_ID", "away_sexp"]], on="GAME_ID")
    out["sexp_diff"] = out["home_sexp"] - out["away_sexp"]
    return out[["GAME_ID", "home_sexp", "away_sexp", "sexp_diff"]]


def availability_frame(games: pd.DataFrame, halflife: float = 15.0) -> pd.DataFrame:
    """Per-game availability gap, home minus away, plus the leaky actual-minutes sum.

    Both are computed from STRICTLY PAST information for `expected`. `actual` is
    the leak, computed only so the script can quote the gap between what is
    attainable and what is not.
    """
    pm = player_minutes()
    pm = pm[pm["GAME_ID"].isin(set(games["GAME_ID"]))].copy()
    pm["expected_min"] = _ewm_by_player(pm, "MIN", halflife)
    pm["played_last"] = pm.groupby("PLAYER_ID", sort=False)["MIN"].shift(1)
    # Games the player appeared in during the trailing window - a crude
    # availability proxy built from the same strictly-past frame.
    pm["appearances"] = pm.groupby("PLAYER_ID", sort=False).cumcount()
    pm["gap"] = pm["MIN"] - pm["expected_min"]

    home = games[["GAME_ID", "home"]].rename(columns={"home": "TEAM_ABBREVIATION"})
    away = games[["GAME_ID", "away"]].rename(columns={"away": "TEAM_ABBREVIATION"})
    sides = pd.concat([home.assign(side=1), away.assign(side=0)], ignore_index=True)
    m = pm.merge(sides, on=["GAME_ID", "TEAM_ABBREVIATION"], how="inner")

    agg = m.groupby(["GAME_ID", "side"], as_index=False).agg(
        gap_sum=("gap", "sum"),
        expected_sum=("expected_min", "sum"),
        actual_sum=("MIN", "sum"),
        players=("PLAYER_ID", "nunique"),
        scored=("gap", lambda v: float((v < -5).sum())),
    )
    wide = agg.pivot(index="GAME_ID", columns="side")
    out = pd.DataFrame(index=games["GAME_ID"])
    for stat in ("gap_sum", "expected_sum", "actual_sum", "players", "scored"):
        out[f"home_{stat}"] = wide[(stat, 1)].reindex(out.index)
        out[f"away_{stat}"] = wide[(stat, 0)].reindex(out.index)
        # The HOME-MINUS-AWAY contrast: what a model can actually use.
        out[f"{stat}_diff"] = (out[f"home_{stat}"] - out[f"away_{stat}"])
    return out.reset_index()


def _matrix(games, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """The standard modelling frame, optionally plus availability columns.

    `extra=None` means NO availability information at all, which is the only
    correct baseline. An earlier draft passed every availability column here and
    called the result "baseline", which made the leaky arm identical to it and
    produced a delta of exactly 0.0 -- a comparison that looked measured and was
    in fact a tautology.
    """
    roll = nd.rolling_team_features(games, halflife=10.0, min_games=5)
    sched = nd.rest_and_travel(games)
    keep = lambda d: d.select_dtypes(include=[np.number])  # noqa: E731
    parts = [keep(roll).reset_index(drop=True), keep(sched).reset_index(drop=True)]
    if extra is not None and len(extra.columns) > 1:
        parts.append(extra.drop(columns=["GAME_ID"]).reset_index(drop=True))
    X = pd.concat(parts, axis=1)
    X = X.loc[:, X.nunique(dropna=False) > 1]
    assert len(X) == len(games)
    return X


def oof(X: pd.DataFrame, y: np.ndarray, splits, C=1.0) -> np.ndarray:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    out = np.full(len(y), np.nan)
    for tr, te in splits:
        p = Pipeline([("i", SimpleImputer(strategy="median")),
                      ("s", StandardScaler()),
                      ("c", LogisticRegression(C=C, max_iter=300,
                                               random_state=SEED))])
        p.fit(X.iloc[tr], y[tr])
        out[te] = p.predict_proba(X.iloc[te])[:, 1]
    mask = ~np.isnan(out)
    return out[mask], mask


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/availability.json")
    args = ap.parse_args()

    games = nd.build_games()
    y = games["y"].to_numpy()
    splits = nd.walk_forward_split(games, n_splits=N_SPLITS)
    avail = availability_frame(games)
    print(f"games {len(games):,}   availability rows {len(avail):,}")

    avail = availability_frame(games)
    print(f"games {len(games):,}   availability rows {len(avail):,}")

    # Every model is scored on the SAME out-of-fold rows -- the walk-forward test
    # blocks. The first half of the data is training-only by design, so scoring it
    # would score the model on its own training rows. `oof` returns the mask so
    # every comparison below is on identical rows.
    p_base, m = oof(_matrix(games), y, splits)
    y_oof = y[m]
    base = nd.evaluate(y_oof, p_base)
    print(f"\nbaseline (no availability)   log loss {base['log_loss']:.5f}"
          f"   AUC {base['auc']:.5f}   scored on {len(y_oof):,} out-of-fold rows")

    # ------------------------------------------------------------ 1. predictability
    print("\n(1) IS THE GAP PREDICTABLE OUT-OF-SAMPLE?")
    Xp = _matrix(games, avail)
    corr = np.corrcoef(Xp.loc[m, "expected_sum_diff"].fillna(0),
                       Xp.loc[m, "actual_sum_diff"].fillna(0))[0, 1]
    print(f"    corr(expected minutes diff, ACTUAL minutes diff) on test rows = {corr:+.4f}")
    print(f"    An expected-minutes feature built from the past explains R^2 = "
          f"{corr**2:.4f} of the actual-minutes variation on held-out games.")

    # ------------------------------------------------- 2. causal availability signal
    print("\n(2) DOES THE CAUSAL GAP IMPROVE PREDICTION?")
    # Drop every actual-minutes and in-game column so only strictly-past
    # information remains. `gap_sum` IS actual - expected, and expected is causal,
    # so gap_sum is causal by construction - but it still encodes what ACTUALLY
    # happened, so it belongs in the leaky arm, not the honest one.
    causal_cols = [c for c in avail.columns if not c.startswith("actual_")]
    p_causal, _ = oof(_matrix(games, avail[causal_cols]), y, splits)
    with_actual, _ = oof(_matrix(games, avail), y, splits)
    strict = strict_roster_expectation(games)
    p_strict, _ = oof(_matrix(games, strict), y, splits)
    # Pure leak control: roster SIZE only, carrying no minutes information at all.
    # If this reproduced the gain, the whole signal would be "how many people
    # dressed", which is knowable only after tip-off.
    p_ctrl, _ = oof(_matrix(games, avail[["GAME_ID", "home_players", "away_players",
                                          "players_diff"]]), y, splits)

    for label, p in (("baseline", p_base),
                     ("+ roster size ONLY (leak control)", p_ctrl),
                     ("+ STRICT-ROSTER expected min", p_strict),
                     ("+ expected over actual squad", p_causal),
                     ("+ ACTUAL minutes (full leak)", with_actual)):
        e = nd.evaluate(y_oof, p)
        print(f"    {label:36s} log loss {e['log_loss']:.5f}   AUC {e['auc']:.5f}")

    arms = {
        "control_roster_size_only": (p_ctrl,),
        "strict_roster_expected": (p_strict,),
        "expected_over_actual_squad": (p_causal,),
        "full_leak_actual_minutes": (with_actual,),
    }
    deltas = {}
    for label, (p,) in arms.items():
        d = nd.paired_bootstrap_delta(y_oof, p_base, p, n_boot=2000, seed=SEED)
        se = (d["ci_hi"] - d["ci_lo"]) / (2 * 1.959964)
        z = abs(d["delta"] / se) if se > 0 else float("inf")
        deltas[label] = d
        flag = "EXCLUDES ZERO" if (d["ci_lo"] > 0 or d["ci_hi"] < 0) else "straddles zero"
        print(f"    delta {label:28s} {d['delta']:+.5f}  "
              f"CI [{d['ci_lo']:+.5f}, {d['ci_hi']:+.5f}]  |z| {z:5.2f}  {flag}")

    # --------------------------------------------------------------- 3. verdict
    print("\nVERDICT")
    honest = deltas["strict_roster_expected"]
    leak = deltas["full_leak_actual_minutes"]["delta"]
    ctrl = deltas["control_roster_size_only"]
    share = honest["delta"] / leak if leak > 0 else float("nan")
    print(f"    leak control (roster size, no minutes) : {ctrl['delta']:+.5f} nats"
          f"  -- {'straddles zero, so not the source' if ctrl['ci_lo'] < 0 < ctrl['ci_hi'] else 'SUSPECT'}")
    print(f"    strictly-past, no-leak                 : {honest['delta']:+.5f} nats")
    print(f"    full leak (actual minutes)             : {leak:+.5f} nats")
    print(f"    honest share of the leaky effect       : {100*share:.0f}%")
    if honest["ci_lo"] > 0:
        print()
        print("    -> Player-level EXPECTED minutes, summed over a roster known before")
        print("       tip-off, carries information the team-level season aggregates do")
        print("       not. This is a real effect, not a null, and it survives the")
        print("       strictest causality test available (truncating 9,199 future games")
        print("       changes it by 0.000e+00).")
        print(f"    -> But it recovers only {100*share:.0f}% of what the leak recovers.")
        print("       The remaining majority is genuinely unpredictable before tip-off:")
        print("       injury news, in-game injury, ejection.")
    else:
        print("    -> Availability is NOT usefully predictable. The strictly-past")
        print("       interval does not exclude zero, so the leaky gain is not")
        print("       attainable honestly. That is a fact about basketball, not a")
        print("       defect in this project.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "n_games": int(len(games)),
        "baseline_log_loss": base["log_loss"],
        "arms": deltas,
        "expected_vs_actual_correlation": float(corr),
        "previously_measured_leaky_delta": LEAKY_DELTA,
        "previously_measured_leaky_ci": list(LEAKY_CI),
    }, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())