"""H11 (theory dossier): role-substitution / usage redistribution.

Pre-registered in results/experiments.jsonl as ``H11-prereg-absence-context``
BEFORE this script was run (``python scripts/h11_absence_context.py --prereg``).
That entry governs.

THE HYPOTHESIS
--------------
When a player is out, the team does not lose his value linearly - some teams
redistribute usage efficiently, others collapse. The canonical model's
availability block (expected minutes of the prior-game squad,
``nba_data.availability_features``) says HOW MUCH is missing and is the project's
only confirmed mechanism (+0.0038 nats [0.0023, 0.0055],
results/availability.json :: strict_roster_expected). H11 asks whether WHAT KIND
of missing value is being absorbed adds anything ON TOP of that: an
``absence_context`` block improves held-out log loss by >= 0.002 nats
incrementally on top of the canonical frame + the availability block, with the
95% CI excluding zero and |z| > 3.

THE CANDIDATE BLOCK (fixed here, before measurement; <= 6 columns)
-----------------------------------------------------------------
Three feature groups, each emitted for both sides
(``home_*``/``away_*``, 6 columns, one per group per side):

1. ``{side}_missing_usage_share``  (missing-usage concentration)
   The trailing usage-minutes SHARE of the highest-usage presumed-missing
   player. Large = one star is out (star absence); small with a missing count =
   diffuse absence; exactly 0.0 = nobody presumed out. The pre-registered
   square / "top-1 vs spread" dummy is DROPPED to respect the 6-column cap
   (docs/modeling_exploration.md: wide blocks in this 60-odd-column frame go
   nowhere - the 66-column play-type PPP block).

2. ``{side}_margin_without_missing_top``  (performance without that player)
   Trailing EWMA of the team's margin in past games where the highest-usage
   missing player was absent, shrunk to the team's overall trailing margin EWMA
   when the sample is thin (w = n/(n+5)).

3. ``{side}_margin_without_top2_usage``  (remaining-squad absorbency)
   The same construction over past games played WITHOUT at least one of the
   team's current top-2 usage players - a "next-man-up" capacity estimate.
   ("without any of its top-2" is read as "missing at least one of the two":
   games missing BOTH are rare enough that the estimator would fall back to the
   team's overall form in almost every row.)

PRE-GAME CAUSAL CONVENTION (non-negotiable)
-------------------------------------------
The prior-game squad is the availability proxy - membership is exactly what
``availability_features`` uses: players with a ``player_minutes`` row in the
team's previous game. Nobody knows tonight's lineup before tip-off
(docs/player_elo_leak.md: the |z| = 5.18 player-Elo result was leakage through
ACTUAL-PARTICIPANT membership). A player is presumed OUT for game t iff he was
in the recent rotation (>= 2 appearances in the team's trailing 10 team-games,
strictly before t) and did NOT play in the team's most recent game. Absence in
PAST games is read from the player game logs (a rotation player with no row in
that game): past participation is historical fact and is causal for future
games. Every player-level quantity is carried forward by an as-of lookup with
strictly-before semantics; every team-level trailing quantity uses the repo's
shift-within-team discipline.

ARMS (declared in the pre-registration)
--------------------------------------
B1 = pinned canonical 67-column base (feature_block_ablation's
     build_feature_frame minus the 6 relative_rest columns - identical to H4's
     A1 and to the historical instrument frame, asserted == 67) PLUS the
     existing availability block's three measured columns
     (home_expected_minutes, away_expected_minutes,
     home_expected_minutes_diff from
     nba_data.availability_features(halflife=15) - the strict_roster_expected
     arm of results/availability.json, consumed not redefined).
B2 = B1 + the six absence_context columns.
B2-g = B2 minus one feature group (diagnostic arms, primary setting only).

"Canonical frame" is pinned to the ablation instrument's 67-col base (the
0.61945 reference) rather than load_design's contrast frame (the 0.61957
reference) because this is a feature-block measurement in the H4 instrument and
load_design is frozen for the concurrent market re-run. The leaderboard reports
B1/B2 against BOTH references.

COMPARISON / SIGN CONVENTION
----------------------------
C1 = B2 vs B1: delta = loss(B1) - loss(B2) via
``nba_data.paired_bootstrap_delta(y, p_first, p_second)``; POSITIVE delta means
the SECOND arm (the candidate) is BETTER. Group diagnostics:
delta = loss(B2-g) - loss(B2); POSITIVE = that group helps.

SETTINGS: primary bootstrap seed 42 / 5 folds (n_boot 2000); replications at
seeds 137 and 2718 (5 folds) and at 8 folds / seed 42. Arms are fit once per
fold count and re-bootstrapped per seed (walk_forward_split and the lbfgs fit
are seed-invariant - issue #45 erratum), so the seed dimension probes CI
variability only and the 8-fold setting is the real replication.

VERDICT (pre-registered): H11 succeeds iff the primary C1 delta >= 0.002 nats
AND its 95% CI excludes zero in the helpful direction AND |z| > 3 AND the delta
is positive at every replication setting. Sub-bar lead: CI excludes zero but
|z| <= 3. Null: CI straddles zero. Multiplicity: 7 looks (C1 x 4 settings +
3 group diagnostics) -> Bonferroni 0.05/7 = 0.0071 one-sided (|z| ~ 2.45); the
project's |z| > 3 bar governs on top.

Run:  python scripts/h11_absence_context.py --prereg   # declaration only
      python scripts/h11_absence_context.py            # the measurement
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from DataProviders import nba_data as nd          # noqa: E402
import feature_block_ablation as fba              # noqa: E402
import recency_ablation as ra                     # noqa: E402

JSONL = ROOT / "results" / "experiments.jsonl"
OUT = ROOT / "results" / "h11_absence_context.json"

CACHE_DIR = Path("/tmp/opencode/h11")
N_BOOT = 2000
# (bootstrap seed, fold count). Primary first. H4's settings, for comparability.
SETTINGS = ((42, 5), (137, 5), (2718, 5), (42, 8))
PRIMARY = (42, 5)
TARGET_NATS = 0.002
TRUNCATION_CUT = "2021-10-20"   # the round3_verify probe date

# --------------------------------------------------------------------------- #
# The block, declared as an explicit literal registry (the H4 pattern). A
# pattern-derived block would silently change meaning on a rename; a visible
# list either names a column that exists or raises.
# --------------------------------------------------------------------------- #
ABSENCE_CONTEXT_COLUMNS = (
    "home_missing_usage_share", "away_missing_usage_share",
    "home_margin_without_missing_top", "away_margin_without_missing_top",
    "home_margin_without_top2_usage", "away_margin_without_top2_usage",
)

FEATURE_GROUPS: dict[str, list[str]] = {
    "missing_usage_concentration": ["home_missing_usage_share",
                                    "away_missing_usage_share"],
    "performance_without_missing_top": ["home_margin_without_missing_top",
                                        "away_margin_without_missing_top"],
    "next_man_up_absorbency": ["home_margin_without_top2_usage",
                               "away_margin_without_top2_usage"],
}

# The availability block's three measured columns (strict_roster_expected arm of
# results/availability.json), consumed from the shared builder verbatim.
AVAILABILITY_COLUMNS = ("home_expected_minutes", "away_expected_minutes",
                        "home_expected_minutes_diff")

# Feature construction parameters, fixed ex ante. NO SWEEP over these.
PLAYER_HL = 15.0        # player trailing EWMAs (matches availability_features)
TEAM_HL = 10.0          # margin EWMAs (matches rolling_team_features)
ROTATION_WINDOW = 10    # team-games; >= 2 appearances = "in the recent rotation"
TOP2_WINDOW = 20        # team-games; >= 1 appearance = "currently on the team"
ABSENCE_WINDOW = 40     # team-games looked back for the conditional EWMAs
SHRINK_K = 5.0          # w = n/(n+SHRINK_K) toward the team's overall EWMA
DEWMA = 0.5 ** (1.0 / TEAM_HL)   # pandas ewm(halflife=10) per-step decay


# --------------------------------------------------------------------------- #
# The builder
# --------------------------------------------------------------------------- #
def absence_context_features(games: pd.DataFrame,
                             roll: pd.DataFrame | None = None) -> pd.DataFrame:
    """The 6-column absence_context block, aligned row-wise to `games`.

    Definitions (per side; `i` is the team's own game sequence, `t` = team-game
    `i`, every input strictly from sequence < i):

    * presumed-missing set M(i) = {players with >= 2 appearances in the team's
      trailing ROTATION_WINDOW team-games strictly before i} minus the prior-game
      squad (the players who appeared in game i-1). No row reads game i's own
      participants.
    * u_p(i) = (trailing EWMA of USG_PCT) x (trailing EWMA of MIN) for player p,
      carried forward from p's most recent appearance strictly before i -
      "trailing usage-minutes".
    * ``{side}_missing_usage_share`` = u of the highest-u player in M(i) divided
      by the total u of the rotation; 0.0 when nobody is presumed missing; NaN
      only when the team has no prior game (i == 0), like availability itself.
    * ``{side}_margin_without_missing_top``: exponentially weighted mean of the
      team's margin over the past ABSENCE_WINDOW team-games in which that player
      was absent (and already established on the team), weights
      DEWMA ** (team-games ago), normalised over the qualifying games - the
      repo's documented NaN-masked ewm convention (adjust=True, ignore_na=False)
      with the window pre-registered. Shrunk: w = n/(n+SHRINK_K) toward the
      team's overall trailing margin EWMA (rolling_team_features' ewm_margin,
      consumed, not redefined). With nobody missing the value IS that EWMA.
    * ``{side}_margin_without_top2_usage``: same, over past games missing at
      least one of the team's top-2 u players among those with an appearance in
      the trailing TOP2_WINDOW games ("current top-2 usage players").

    Truncation-invariance is asserted by `causality_checks`.
    """
    g = games.reset_index(drop=True)
    if roll is None:
        roll = nd.rolling_team_features(g, halflife=TEAM_HL, min_games=5)
    roll = roll.reset_index(drop=True)

    # ---- team-game long frame (margin signed from the row's own team) -------
    def _side(team_col, pts_col, opp_col, overall_col):
        return pd.DataFrame({
            "grow": np.arange(len(g)),
            "GAME_ID": g["GAME_ID"].to_numpy(),
            "GAME_DATE": g["GAME_DATE"].to_numpy(),
            "team": g[team_col].to_numpy(),
            "margin": (g[pts_col].to_numpy(dtype=float)
                       - g[opp_col].to_numpy(dtype=float)),
            "overall": roll[overall_col].to_numpy(dtype=float),
        })

    tg = pd.concat([_side("home", "home_pts", "away_pts", "home_ewm_margin"),
                    _side("away", "away_pts", "home_pts", "away_ewm_margin")],
                   ignore_index=True)
    tg = tg.sort_values(["team", "GAME_DATE", "GAME_ID"]).reset_index(drop=True)
    tg["seq"] = tg.groupby("team").cumcount()

    # ---- player appearances (membership = player_minutes = the availability
    # block's own squad definition; MIN-notna rows only) ---------------------
    pm = nd.player_minutes(g)[["GAME_ID", "GAME_DATE", "PLAYER_ID",
                               "TEAM_ABBREVIATION", "MIN"]].copy()
    usg = nd.load_player_usage()
    usg = usg[["GAME_ID", "PLAYER_ID", "USG_PCT"]].drop_duplicates(
        subset=["GAME_ID", "PLAYER_ID"])
    pm = pm.merge(usg, on=["GAME_ID", "PLAYER_ID"], how="left")
    pm["USG_PCT"] = pd.to_numeric(pm["USG_PCT"], errors="coerce")
    pm["exp_min"] = nd._ewm_by_player(pm, "MIN", PLAYER_HL)
    pm["exp_usg"] = nd._ewm_by_player(pm, "USG_PCT", PLAYER_HL)
    pm["u"] = pm["exp_usg"] * pm["exp_min"]
    pm = pm.merge(tg[["GAME_ID", "team", "seq"]],
                  left_on=["GAME_ID", "TEAM_ABBREVIATION"],
                  right_on=["GAME_ID", "team"], how="inner")

    out = np.full((len(g), 6), np.nan)
    col_of = {c: j for j, c in enumerate(ABSENCE_CONTEXT_COLUMNS)}
    vals = _side_features(tg, pm)          # ndarray (len(tg), 3), per team-game
    grow = tg["grow"].to_numpy()
    team = tg["team"].to_numpy()
    for side in ("home", "away"):
        # Each team-game row belongs to exactly one side of its game. Assert
        # rather than assume: a swapped side would silently negate every
        # contrast (nba_data documents exactly that trap), and the two side
        # masks must partition the team-game rows.
        m = team == g[side].to_numpy()[grow]
        assert m.any(), f"no team-game rows for the {side} side"
        out[grow[m], col_of[f"{side}_missing_usage_share"]] = vals[m, 0]
        out[grow[m], col_of[f"{side}_margin_without_missing_top"]] = vals[m, 1]
        out[grow[m], col_of[f"{side}_margin_without_top2_usage"]] = vals[m, 2]
    # Coverage must look like availability's: NaN exactly where a team has no
    # prior game on record (its first game). A wider NaN pattern means the
    # prior-game-squad join dropped rows somewhere. Checked per side as "at most
    # one NaN row per team", which is scale-aware (a prefix of the schedule has
    # one first game per team too, just fewer total rows).
    for j, side in ((0, "home"), (1, "away")):
        n_nan = int((~np.isfinite(out[:, j])).sum())
        n_teams = int(g[side].nunique())
        assert n_nan <= n_teams, (
            f"{side} share column has {n_nan} NaN rows for {n_teams} teams: "
            f"more than one unfillable row per team means the prior-game "
            f"squad join dropped rows")
    return pd.DataFrame(out, columns=list(ABSENCE_CONTEXT_COLUMNS))


def _side_features(tg: pd.DataFrame, pm: pd.DataFrame) -> np.ndarray:
    """(len(tg), 3) array: share, without_missing_top, without_top2_usage.

    Row k of the result belongs to team-game tg.iloc[k]; the team's local game
    index is tg.iloc[k]["seq"].
    """
    out = np.full((len(tg), 3), np.nan)
    for team, tgrp in tg.groupby("team", sort=False):
        tgrp = tgrp.sort_values("seq")
        pos = tgrp.index.to_numpy()                 # positions in tg / out
        n_t = len(tgrp)
        assert np.array_equal(tgrp["seq"].to_numpy(), np.arange(n_t)), \
            "team-game sequence is not contiguous"
        margin = tgrp["margin"].to_numpy(dtype=float)
        overall = tgrp["overall"].to_numpy(dtype=float)

        tpm = pm[pm["team"] == team]
        players: dict[int, dict] = {}
        for pid, pgrp in tpm.groupby("PLAYER_ID", sort=False):
            pgrp = pgrp.sort_values("seq")
            seqs = pgrp["seq"].to_numpy()
            u_at = pgrp["u"].to_numpy(dtype=float)
            appeared = np.zeros(n_t, dtype=bool)
            appeared[seqs] = True
            # before[i] = # appearances strictly before team-game i.
            before = np.searchsorted(seqs, np.arange(n_t), side="left")
            u_of = np.full(n_t, np.nan)
            has = before > 0
            u_of[has] = u_at[before[has] - 1]   # latest appearance with seq < i
            first = int(seqs[0])
            cnt = lambda w: before - np.searchsorted(  # noqa: E731
                seqs, np.clip(np.arange(n_t) - w, 0, None), side="left")
            players[int(pid)] = {
                "u_of": u_of,
                "cnt10": cnt(ROTATION_WINDOW),
                "cnt20": cnt(TOP2_WINDOW),
                "absent_est": (~appeared) & (np.arange(n_t) > first),
                "appeared": appeared,
            }

        at_seq: dict[int, set[int]] = {}
        for pid, pl in players.items():
            for s in np.flatnonzero(pl["appeared"]).tolist():
                at_seq.setdefault(int(s), set()).add(pid)

        for i in range(1, n_t):     # i == 0: no prior game -> NaN
            rot = [p for p, pl in players.items() if pl["cnt10"][i] >= 2]
            u_rot = {p: players[p]["u_of"][i] for p in rot
                     if np.isfinite(players[p]["u_of"][i])}
            prev_squad = at_seq.get(i - 1, set())
            missing = [p for p in u_rot if p not in prev_squad]

            # ---- group 1: missing-usage concentration --------------------
            # Every reduction below is ORDER-INDEPENDENT (sorted inputs; ties
            # broken by player id). Dict iteration order follows the merge's
            # row order, which is not guaranteed stable across differently
            # sized builds - and float addition is not associative. Without
            # this the truncation test sees 1-ulp noise that looks like (but is
            # not) non-causality.
            denom = float(sum(sorted(u_rot.values())))
            if missing:
                p_star = max(sorted(missing),
                             key=lambda p: (u_rot[p], -p))
                share = float(u_rot[p_star] / denom) if denom > 0 else np.nan
            else:
                p_star, share = None, 0.0

            # ---- group 2: performance without the missing top player -----
            if p_star is None:
                w_top = float(overall[i])          # nobody out: team form
            else:
                w_top = _conditional_margin(players[p_star]["absent_est"],
                                            margin, overall[i], i)

            # ---- group 3: next-man-up absorbency -------------------------
            cand = [p for p, pl in players.items() if pl["cnt20"][i] >= 1
                    and np.isfinite(pl["u_of"][i])]
            if not cand:
                w_top2 = float(overall[i])
            else:
                top2 = sorted(cand,
                              key=lambda p: (players[p]["u_of"][i], -p),
                              reverse=True)[:2]
                mask = np.zeros(n_t, dtype=bool)
                for p in top2:
                    mask |= players[p]["absent_est"]
                w_top2 = _conditional_margin(mask, margin, overall[i], i)

            out[pos[i], 0] = share
            out[pos[i], 1] = w_top
            out[pos[i], 2] = w_top2
    return out


def _conditional_margin(mask: np.ndarray, margin: np.ndarray,
                        target: float, i: int) -> float:
    """Trailing EWMA of the team's margin over qualifying past games, shrunk
    toward `target` (the team's overall trailing margin EWMA) by n/(n+SHRINK_K).

    Weights are DEWMA ** (team-games ago), normalised over the qualifying games
    only - the repo's NaN-masked ewm convention with the ABSENCE_WINDOW
    pre-registered. Every qualifying index is strictly before i.
    """
    lo = max(0, i - ABSENCE_WINDOW)
    ks = np.arange(lo, i)
    assert (ks < i).all()
    qual = mask[lo:i]
    n = int(qual.sum())
    if n == 0:
        return float(target)
    w = DEWMA ** (i - ks[qual]).astype(float)
    m = margin[lo:i][qual]
    ewm = float(np.sum(w * m) / np.sum(w))
    return float((n / (n + SHRINK_K)) * ewm
                 + (SHRINK_K / (n + SHRINK_K)) * target)


# --------------------------------------------------------------------------- #
# Leakage checks (mirroring the H3/H4 scripts and scripts/round3_verify.py)
# --------------------------------------------------------------------------- #
def causality_checks(games: pd.DataFrame, cand: pd.DataFrame) -> dict:
    """(1) truncation invariance, (2) membership discipline.

    The truncation test is the strong one: rebuild the block on games cut at
    TRUNCATION_CUT and require every column bit-identical on the overlapping
    rows, NaN pattern included. A membership leak through actual-participant
    data (the |z| = 5.18 player-Elo failure) fails this test immediately.
    """
    k = int((games["GAME_DATE"] >= pd.Timestamp(TRUNCATION_CUT)).argmax())
    assert 0 < k < len(games), "truncation cut is degenerate"
    g_cut = games.iloc[:k].reset_index(drop=True)
    cand_cut = absence_context_features(g_cut)
    a = cand.iloc[:k].to_numpy(dtype=float)
    b = cand_cut.to_numpy(dtype=float)
    same_nan = np.array_equal(np.isnan(a), np.isnan(b))
    max_abs = float(np.nanmax(np.abs(a - b))) if not np.all(np.isnan(a)) else 0.0
    identical = same_nan and np.allclose(np.nan_to_num(a), np.nan_to_num(b),
                                         rtol=0, atol=0)
    if not identical:
        raise AssertionError(
            f"TRUNCATION LEAK: cutting the future at {TRUNCATION_CUT} changed "
            f"the block on the first {k} rows (NaN pattern equal: {same_nan}, "
            f"max |diff| {max_abs:g})")
    return {"truncation_cut": TRUNCATION_CUT, "rows_checked": int(k),
            "bit_identical": bool(identical), "nan_pattern_equal": bool(same_nan),
            "max_abs_diff": max_abs,
            "membership": "presumed-missing set, top-2 usage players and every "
                          "carried-forward u/exp-minutes value are derived from "
                          "team-game sequence numbers strictly below the row's "
                          "own (asserted in _conditional_margin; lookups are "
                          "searchsorted side='left'); game i's own participants "
                          "are never read"}


# --------------------------------------------------------------------------- #
# Scoring machinery (H4's, kept identical by construction)
# --------------------------------------------------------------------------- #
def fit_oof_keep_models(X, y, splits):
    """ra.fit_oof's loop, but also keeping the fitted pipelines.

    Identical by construction to the sanctioned instrument (same
    ra.make_pipeline, same fit/predict calls); kept local only because the
    pre-registered coefficient diagnostic needs the coefficients. `fit_oof`
    itself is still called on one arm and asserted equal below, so drift would
    be caught. Imputation and scaling are fit on TRAIN rows only (inside
    ra.make_pipeline's fit on X.iloc[tr]).
    """
    oof = np.full(len(y), np.nan)
    models = {}
    for k, (tr, te) in enumerate(splits, start=1):
        pipe = ra.make_pipeline()
        pipe.fit(X.iloc[tr], y[tr])
        oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]
        models[k] = pipe
    return oof, ~np.isnan(oof), models


def delta_row(y, p_first, p_second, seed) -> dict:
    """loss(first) - loss(second); POSITIVE = second arm better."""
    d = nd.paired_bootstrap_delta(y, p_first, p_second,
                                  n_boot=N_BOOT, seed=seed)
    se = (d["ci_hi"] - d["ci_lo"]) / (2 * 1.959964)
    return {"seed": seed, "n_boot": N_BOOT,
            "delta": float(d["delta"]), "ci_lo": float(d["ci_lo"]),
            "ci_hi": float(d["ci_hi"]), "p_positive": float(d["p_positive"]),
            "se": float(se),
            "z": float(abs(d["delta"]) / se) if se > 0 else float("nan")}


def _ll(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def per_fold_deltas(y, splits, p_first_full, p_second_full) -> list[dict]:
    """loss(first) - loss(second) per walk-forward test block (in-row-space)."""
    return [{"fold": k, "n": int(len(te)),
             "delta": _ll(y[te], p_first_full[te]) - _ll(y[te], p_second_full[te])}
            for k, (_tr, te) in enumerate(splits, start=1)]


def per_season_deltas(y, seasons, p_first, p_second) -> dict:
    out = {}
    for s in pd.unique(seasons):
        m = seasons == s
        out[str(s)] = {"n": int(m.sum()),
                       "delta": _ll(y[m], p_first[m]) - _ll(y[m], p_second[m])}
    return out


# --------------------------------------------------------------------------- #
# Frame assembly + caching
# --------------------------------------------------------------------------- #
def build_arms(games: pd.DataFrame, use_cache: bool = True):
    """B1 = pinned canonical 67-col base + availability block; B2 = B1 + cand."""
    key = int(pd.util.hash_pandas_object(games["GAME_ID"], index=False).sum())
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / "arms.pkl"
    meta = {"n_games": int(len(games)), "game_id_hash": key,
            "availability_columns": list(AVAILABILITY_COLUMNS),
            "absence_context_columns": list(ABSENCE_CONTEXT_COLUMNS)}
    if use_cache and cache.exists():
        blob = pd.read_pickle(cache)
        if blob.get("meta") == meta:
            print(f"[cache] arms loaded from {cache}", flush=True)
            return blob["a1"], blob["avail"], blob["cand"], meta
        print("[cache] meta mismatch -> rebuilding", flush=True)

    frame = fba.build_feature_frame(games)
    a1 = fba.drop_block(frame, fba.resolve_block("relative_rest", frame))
    assert a1.shape[1] == 67, \
        f"pinned canonical base is {a1.shape[1]} cols, expected 67"
    avail = nd.availability_features(games)[list(AVAILABILITY_COLUMNS)]
    assert len(avail) == len(games)
    cand = absence_context_features(games)
    assert list(cand.columns) == list(ABSENCE_CONTEXT_COLUMNS)
    pd.to_pickle({"meta": meta, "a1": a1, "avail": avail, "cand": cand}, cache)
    print(f"[cache] arms written to {cache}", flush=True)
    return a1, avail, cand, meta


def describe_cand(cand: pd.DataFrame) -> dict:
    """Feature-distribution sanity (no outcome is read here)."""
    desc = {}
    for c in cand.columns:
        v = cand[c].to_numpy(dtype=float)
        desc[c] = {"non_null_share": round(float(np.isfinite(v).mean()), 4),
                   "mean": round(float(np.nanmean(v)), 4),
                   "sd": round(float(np.nanstd(v)), 4),
                   "p50": round(float(np.nanpercentile(v, 50)), 4),
                   "p95": round(float(np.nanpercentile(v, 95)), 4),
                   "max": round(float(np.nanmax(v)), 4)}
    return desc


# --------------------------------------------------------------------------- #
# Pre-registration (appended by --prereg, BEFORE any run)
# --------------------------------------------------------------------------- #
def prereg_entry() -> dict:
    return {
        "name": "H11-prereg-absence-context",
        "status": "declared_before_execution",
        "declared_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_type": "feature-block candidate (theory dossier H11): "
                           "role-substitution / usage redistribution",
        "hypothesis": "An absence-context block improves held-out log loss by "
                      ">= 0.002 nats INCREMENTALLY on top of the canonical "
                      "frame + the existing availability block (expected "
                      "minutes of the prior-game squad; measured +0.0038 nats "
                      "[0.0023, 0.0055]), with the 95% CI excluding zero and "
                      "|z| > 3. The availability block says HOW MUCH is "
                      "missing; H11 asks whether WHAT KIND of missing value is "
                      "being absorbed adds anything on top (star-absence vs "
                      "diffuse absence, how the team has historically played "
                      "without that player, and the remaining squad's "
                      "next-man-up capacity).",
        "candidate_block": {
            "name": "absence_context",
            "columns": list(ABSENCE_CONTEXT_COLUMNS),
            "groups": {k: list(v) for k, v in FEATURE_GROUPS.items()},
            "size_note": "6 columns <= the pre-registered cap; the square / "
                         "'top-1 vs spread' dummy named in the dossier is "
                         "DROPPED to respect it (docs/modeling_exploration.md: "
                         "the 66-col play-type PPP block).",
            "definitions": {
                "presumed_missing_set": "M(i) = {players with >= 2 appearances "
                    "in the team's trailing 10 team-games strictly before game "
                    "i} minus the prior-game squad (players who appeared in the "
                    "team's previous game). 'A player who was in the recent "
                    "rotation and did not play' - never tonight's participants.",
                "usage_minutes": "u_p(i) = (trailing EWMA of USG_PCT, halflife "
                    "15) x (trailing EWMA of MIN, halflife 15), carried "
                    "forward from p's most recent appearance strictly before i. "
                    "Shift-within-player via nba_data._ewm_by_player.",
                "g1_{side}_missing_usage_share": "u of the highest-u "
                    "presumed-missing player / total u of the rotation; 0.0 if "
                    "nobody is presumed missing; NaN only for a team's first "
                    "game on record.",
                "g2_{side}_margin_without_missing_top": "exponentially "
                    "weighted mean (weights 0.5^(team-games ago/10), normalised "
                    "over qualifying games - the repo's NaN-masked ewm "
                    "convention) of the team's margin over the past 40 "
                    "team-games in which that player was absent (and already "
                    "established on the team); shrunk w = n/(n+5) toward "
                    "rolling_team_features' ewm_margin (consumed, not "
                    "redefined); with nobody missing the value IS that EWMA.",
                "g3_{side}_margin_without_top2_usage": "same construction over "
                    "past games missing at least one of the team's current top-2 "
                    "u players (appearance in the trailing 20 team-games). "
                    "'without any of its top-2' is pre-registered as 'missing "
                    "at least one of the two': both-out games are thin enough "
                    "that the estimator would fall back to overall form almost "
                    "everywhere.",
            },
            "parameters_fixed_ex_ante": {
                "PLAYER_HL": PLAYER_HL, "TEAM_HL": TEAM_HL,
                "ROTATION_WINDOW": ROTATION_WINDOW, "TOP2_WINDOW": TOP2_WINDOW,
                "ABSENCE_WINDOW": ABSENCE_WINDOW, "SHRINK_K": SHRINK_K,
                "note": "no sweep over any of these; no variants; no column "
                        "drops mid-search",
            },
        },
        "arms": "B1 = pinned canonical 67-col base (feature_block_ablation's "
                "build_feature_frame minus the 6 relative_rest columns - "
                "identical to H4's A1 / the historical instrument frame, "
                "asserted == 67) + the existing availability block's three "
                "measured columns (home_expected_minutes, away_expected_minutes,"
                " home_expected_minutes_diff from "
                "nba_data.availability_features(halflife=15) - the "
                "strict_roster_expected arm of results/availability.json, "
                "consumed not redefined). B2 = B1 + the 6 absence_context "
                "columns. B2-g = B2 minus one feature group (diagnostic arms, "
                "primary setting only).",
        "frame_choice_note": "'Canonical frame' is pinned to the ablation "
                "instrument's 67-col base (the 0.61945 reference), not "
                "load_design's contrast frame (the 0.61957 reference), because "
                "this is a feature-block measurement in the H4 instrument and "
                "load_design is frozen for the concurrent market re-run. The "
                "leaderboard reports B1/B2 against BOTH references.",
        "comparisons": "C1 = B2 vs B1 (primary). delta = loss(B1) - loss(B2) "
                "via nba_data.paired_bootstrap_delta, POSITIVE = candidate "
                "better. Per-feature-group diagnostics: leave-one-group-out "
                "B2-g vs B2 at the primary setting (delta = loss(B2-g) - "
                "loss(B2), POSITIVE = the group helps) plus standardised "
                "|coef| per candidate column per fold (descriptive, no test, "
                "no look).",
        "settings": "Primary: bootstrap seed 42 / 5 walk-forward folds "
                "(n_boot 2000). Replications: seeds 137 and 2718 at 5 folds, "
                "and 8 folds at seed 42. Arms fit once per fold count and "
                "re-bootstrapped per seed (walk_forward_split and the lbfgs fit "
                "are seed-invariant - issue #45 erratum), so the seed dimension "
                "probes CI variability only and the 8-fold setting is the real "
                "replication. Splits: nba_data.walk_forward_split(n_splits, "
                "initial_frac=0.5, gap=0), identical across all arms; scored "
                "rows are the walk-forward test blocks only (6,600 rows).",
        "metrics": "Primary: log loss. Also Brier, ROC-AUC, accuracy and a "
                "calibration table (nba_data.evaluate) for both arms at every "
                "setting; per-fold deltas; per-season log-loss deltas at the "
                "primary setting (season-drift read). SE = (ci_hi - "
                "ci_lo)/(2*1.959964); |z| = delta/SE.",
        "decision_rule": "H11 succeeds iff at the primary setting (a) delta >= "
                "0.002 nats (the pre-registered minimum effect of interest), "
                "(b) the 95% CI excludes zero in the helpful direction, "
                "(c) |z| > 3, AND (d) delta > 0 at every replication setting. "
                "Otherwise: 'sub-bar lead' if the primary CI excludes zero with "
                "|z| <= 3; 'null' if the primary CI straddles zero. A null is a "
                "complete result.",
        "multiplicity": "7 looks in the H11 family: C1 x 4 settings + 3 "
                "leave-one-group-out diagnostics at the primary setting -> "
                "Bonferroni 0.05/7 = 0.0071 one-sided (|z| ~ 2.45). The "
                "project's |z| > 3 bar (~0.00135 one-sided) is stricter and "
                "governs. The bootstrap seeds do not add looks (same rows "
                "resampled). No threshold sweeps and no selection among column "
                "subsets: the block is fixed here, before it is measured.",
        "sizing": "SE ~ 0.0005-0.0008 nats for a small block on 6,600 OOF "
                "rows; |z| > 3 needs roughly 0.002-0.0024 nats; the "
                "pre-registered minimum effect (0.002 nats) sits at the bar.",
        "causal_convention": "Pre-game information only. The prior-game squad "
                "is the availability proxy (availability_features' own "
                "membership: players with a player_minutes row in the team's "
                "previous game). Absences are inferred from player game logs - "
                "a player who was in the recent rotation and did not play (in "
                "the team's most recent game) - never from who actually played "
                "tonight. Past-game participation is historical fact and is "
                "causal for future games.",
        "leakage_checks_asserted_in_code": "(1) truncation invariance: "
                "rebuilding the block on games cut at 2021-10-20 must "
                "reproduce every column bit-identically on the overlapping "
                "rows, NaN pattern included; (2) membership: rotation, "
                "prior-game squad, missing set, top-2 usage players and every "
                "carried-forward player value come from team-game sequence "
                "numbers strictly below the row's own; (3) identical OOF rows "
                "and labels across arms (asserted) and arm parity with "
                "recency_ablation.fit_oof (asserted); (4) imputation/scaling "
                "fit on train folds only (ra.make_pipeline inside ra.fit_oof).",
        "prior_evidence": "FOR: availability is the project's only confirmed "
                "mechanism (+0.0038 nats [0.0023, 0.0055]) and it is a coarse "
                "sum over the prior-game squad - the dossier's premise is that "
                "WHO is missing is discarded by it. AGAINST: every player-level "
                "block measured so far is a null or a leak - player-Elo over "
                "actual participants was leakage at |z| 5.18 and its honest "
                "version is +0.00065 (null); RAPM player ratings +0.00151 CI "
                "[-0.00059, +0.00358] (null); round-2 A1_lineup_continuity "
                "-0.0117 at 5.7% coverage (a coverage failure, not a hypothesis "
                "failure); every reweighting in this repo (recency windows, "
                "decay speeds, H3 recalibration, H4 relative rest) is a null.",
        "overlap_note": "src/DataProviders/feat_squad_continuity.py and "
                "feat_squad_quality.py are sibling in-flight lanes on the "
                "composition of the prior-game squad; H11's columns are "
                "distinct (missing-player concentration, without-player "
                "history, next-man-up capacity) and none of their columns is "
                "measured in results/experiments.jsonl as of this declaration. "
                "This is not a re-run of any logged null.",
        "runtime_note": "Arms ~20s-5min each; the pinned base frame, the "
                "availability columns and the player-log join are cached under "
                "/tmp/opencode/h11/ and the cache is keyed on the game set.",
    }


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prereg", action="store_true",
                    help="append the pre-registration to "
                         "results/experiments.jsonl and exit")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--stats-only", action="store_true",
                    help="build the block, run the leakage checks, print "
                         "feature stats, fit nothing")
    args = ap.parse_args()

    if args.prereg:
        rec = prereg_entry()
        with JSONL.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(f"appended pre-registration {rec['name']!r} to {JSONL}")
        return 0

    # The pre-registration must exist BEFORE any measurement is produced.
    declared = [json.loads(l)["name"] for l in JSONL.open()
                if "H11-prereg-absence-context" in l]
    if "H11-prereg-absence-context" not in declared:
        raise SystemExit("results/experiments.jsonl has no H11 pre-registration; "
                         "run with --prereg first")

    t0 = time.time()
    games = nd.build_games()
    y = games["y"].to_numpy(dtype=int)
    a1, avail, cand, meta = build_arms(games, use_cache=not args.no_cache)
    b1 = pd.concat([a1.reset_index(drop=True), avail.reset_index(drop=True)],
                   axis=1)
    b2 = pd.concat([b1, cand.reset_index(drop=True)], axis=1)
    assert b1.shape[1] == 70 and b2.shape[1] == 76, \
        f"unexpected arm widths: B1 {b1.shape[1]}, B2 {b2.shape[1]}"
    for nm, fr in (("B1", b1), ("B2", b2)):
        dupes = sorted({c for c in fr.columns
                        if list(fr.columns).count(c) > 1})
        assert not dupes, f"{nm} has duplicate columns {dupes}"
    print(f"[{time.time()-t0:.0f}s] games {len(games):,}  B1 {b1.shape[1]} cols "
          f"  B2 {b2.shape[1]} cols", flush=True)

    checks = causality_checks(games, cand)
    print(f"[{time.time()-t0:.0f}s] leakage checks: "
          f"truncation bit-identical = {checks['bit_identical']} on "
          f"{checks['rows_checked']} rows (max |diff| {checks['max_abs_diff']:g})",
          flush=True)
    stats = describe_cand(cand)
    for c, s in stats.items():
        print(f"  {c:36s} non-null {s['non_null_share']:.3f}  "
              f"mean {s['mean']:+.3f}  sd {s['sd']:.3f}  "
              f"p50 {s['p50']:+.3f}  p95 {s['p95']:+.3f}", flush=True)
    if args.stats_only:
        print(json.dumps({"checks": checks, "stats": stats}, indent=2))
        return 0

    arms = {"B1": b1, "B2": b2}
    for gname, gcols in FEATURE_GROUPS.items():
        arms[f"B2-{gname}"] = b2.drop(columns=gcols)

    results: list[dict] = []
    coef_diag: dict = {}
    season_diag: dict = {}
    for folds in sorted({f for _s, f in SETTINGS}):
        splits = nd.walk_forward_split(games, n_splits=folds)
        preds_full, masks, models = {}, {}, {}
        for name, fr in arms.items():
            p_full, m, mods = fit_oof_keep_models(fr, y, splits)
            preds_full[name], masks[name], models[name] = p_full, m, mods
        # Guard against drift from the sanctioned harness + OOF-row identity.
        p_check, y_check, _ = ra.fit_oof(b2, y, splits)
        assert np.allclose(p_check, preds_full["B2"][masks["B2"]])
        assert np.array_equal(y_check, y[masks["B2"]])
        ref_mask = masks["B1"]
        for name in arms:
            assert np.array_equal(masks[name], ref_mask), \
                f"{name} disagreed on which rows are out-of-fold"
        y_oof = y[ref_mask]
        preds = {name: preds_full[name][ref_mask] for name in arms}
        seasons = games["SEASON"].to_numpy()[ref_mask]
        print(f"[{time.time()-t0:.0f}s] folds={folds}: {len(arms)} arms fit, "
              f"{len(y_oof):,} OOF rows (fit_oof parity + identical rows "
              f"asserted)", flush=True)

        for seed, f in SETTINGS:
            if f != folds:
                continue
            row = delta_row(y_oof, preds["B1"], preds["B2"], seed)
            row.update({
                "name": f"H11-run-C1-s{seed}-f{folds}",
                "comparison": "C1", "arms": "B2 vs B1",
                "sign_convention": "delta = loss(B1) - loss(B2); POSITIVE = "
                                   "the candidate (B2) is better",
                "n_splits": folds, "n_scored": int(len(y_oof)),
                "log_loss_B1": float(nd.evaluate(y_oof, preds["B1"])["log_loss"]),
                "log_loss_B2": float(nd.evaluate(y_oof, preds["B2"])["log_loss"]),
                "metrics_B1": nd.evaluate(y_oof, preds["B1"]),
                "metrics_B2": nd.evaluate(y_oof, preds["B2"]),
                "per_fold_deltas": per_fold_deltas(
                    y, splits, preds_full["B1"], preds_full["B2"]),
            })
            if (seed, f) == PRIMARY:
                season_diag = per_season_deltas(
                    y_oof, seasons, preds["B1"], preds["B2"])
                row["per_season_deltas"] = season_diag
            results.append(row)
            print(f"[{time.time()-t0:6.0f}s] {row['name']:24s} "
                  f"delta {row['delta']:+.5f} "
                  f"CI [{row['ci_lo']:+.5f},{row['ci_hi']:+.5f}] "
                  f"|z| {row['z']:.2f}", flush=True)

            if (seed, f) == PRIMARY:
                for gname in FEATURE_GROUPS:
                    grow = delta_row(y_oof, preds[f"B2-{gname}"], preds["B2"],
                                     seed)
                    grow.update({
                        "name": f"H11-diag-{gname}-s{seed}-f{folds}",
                        "comparison": f"group:{gname}",
                        "arms": f"B2-{gname} vs B2",
                        "sign_convention": "delta = loss(B2-g) - loss(B2); "
                                           "POSITIVE = the group helps",
                        "n_splits": folds, "n_scored": int(len(y_oof)),
                        "log_loss_minus": float(nd.evaluate(
                            y_oof, preds[f"B2-{gname}"])["log_loss"]),
                        "log_loss_B2": float(nd.evaluate(
                            y_oof, preds["B2"])["log_loss"]),
                    })
                    results.append(grow)
                    print(f"[{time.time()-t0:6.0f}s] {grow['name']:32s} "
                          f"delta {grow['delta']:+.5f} "
                          f"CI [{grow['ci_lo']:+.5f},{grow['ci_hi']:+.5f}] "
                          f"|z| {grow['z']:.2f}", flush=True)

        # ---- coefficient diagnostic (primary fold count only) -------------
        if folds == PRIMARY[1]:
            cols = list(b2.columns)
            per = {c: [] for c in ABSENCE_CONTEXT_COLUMNS}
            for k, pipe in models["B2"].items():
                beta = np.asarray(pipe.steps[-1][1].coef_).ravel()
                for c in ABSENCE_CONTEXT_COLUMNS:
                    per[c].append(float(beta[cols.index(c)]))
            coef_diag = {
                "units": "standardised features -> coef in log-odds per SD",
                "per_fold": {c: [round(v, 5) for v in vs]
                             for c, vs in per.items()},
                "mean_abs": {c: round(float(np.mean(np.abs(vs))), 5)
                             for c, vs in per.items()},
                "mean_abs_by_group": {
                    g: round(float(np.mean([np.abs(v) for c in cs
                                            for v in per[c]])), 5)
                    for g, cs in FEATURE_GROUPS.items()},
            }
            print(f"[{time.time()-t0:.0f}s] coef diagnostic (mean |coef|): "
                  f"{coef_diag['mean_abs_by_group']}", flush=True)

    # ---- verdict against the pre-registered rule --------------------------
    def get(name):
        return next(r for r in results if r["name"] == name)

    prim = get(f"H11-run-C1-s{PRIMARY[0]}-f{PRIMARY[1]}")
    reps = [get(f"H11-run-C1-s{s}-f{f}") for s, f in SETTINGS
            if (s, f) != PRIMARY]
    signs_ok = all(r["delta"] > 0 for r in [prim] + reps)
    rule_ok = (prim["delta"] >= TARGET_NATS and prim["ci_lo"] > 0
               and prim["z"] > 3 and signs_ok)
    if rule_ok:
        decision = "H11 candidate EARNS its place"
    elif prim["ci_lo"] > 0:
        decision = ("H11 candidate is a SUB-BAR LEAD: primary CI excludes zero "
                    "but the pre-registered rule is not met (|z| <= 3 and/or "
                    "delta < 0.002 nats)")
    elif prim["ci_hi"] < 0:
        decision = ("H11 candidate FAILS and HURTS: primary CI excludes zero "
                    "in the unhelpful direction")
    else:
        decision = ("H11 candidate is a NULL: the primary CI straddles zero - "
                    "nothing was established")
    verdict = {
        "rule": "H11 succeeds iff primary C1 delta >= 0.002 nats AND its 95% CI "
                "excludes zero in the helpful direction AND |z| > 3 AND delta "
                "is positive at every replication setting "
                "(H11-prereg-absence-context)",
        "primary": {"setting": f"seed {PRIMARY[0]} / {PRIMARY[1]} folds",
                    "delta": prim["delta"], "ci_lo": prim["ci_lo"],
                    "ci_hi": prim["ci_hi"], "z": prim["z"],
                    "log_loss_B1": prim["log_loss_B1"],
                    "log_loss_B2": prim["log_loss_B2"]},
        "replications": [{"setting": f"seed {s} / {f} folds",
                          "delta": r["delta"], "ci_lo": r["ci_lo"],
                          "ci_hi": r["ci_hi"], "z": r["z"]}
                         for r, (s, f) in zip(reps, [p for p in SETTINGS
                                                     if p != PRIMARY])],
        "signs_consistent": bool(signs_ok),
        "target_nats": TARGET_NATS,
        "decision": decision,
        "per_season_deltas_primary": season_diag,
        "group_diagnostics_primary": {
            g: {k: get(f"H11-diag-{g}-s{PRIMARY[0]}-f{PRIMARY[1]}")[k]
                for k in ("delta", "ci_lo", "ci_hi", "z", "log_loss_minus")}
            for g in FEATURE_GROUPS},
        "multiplicity": "7 looks (C1 x 4 settings + 3 group diagnostics) -> "
                        "Bonferroni 0.05/7 = 0.0071 one-sided (|z| ~ 2.45); "
                        "the project bar |z| > 3 governs on top",
    }

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "pre_registration": "results/experiments.jsonl :: "
                            "H11-prereg-absence-context",
        "frames": {"B1_canonical_plus_availability": int(b1.shape[1]),
                   "B2_plus_absence_context": int(b2.shape[1]),
                   "pinned_canonical_base": 67,
                   "availability_columns": list(AVAILABILITY_COLUMNS),
                   "absence_context_columns": list(ABSENCE_CONTEXT_COLUMNS),
                   "reference_pooled_oof": {"ablation_pinned_A1_base": 0.61945,
                                            "canonical_load_design": 0.61957}},
        "cache": str(CACHE_DIR / "arms.pkl"),
        "leakage_checks": checks,
        "feature_stats": stats,
        "coef_diagnostic": coef_diag,
        "results": results,
        "verdict": verdict,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    print(f"\nwrote {OUT}", flush=True)

    ts = datetime.now(timezone.utc).isoformat()
    with JSONL.open("a") as fh:
        for r in results:
            rec = {k: v for k, v in r.items()
                   if k not in ("per_fold_deltas", "per_season_deltas",
                                "metrics_B1", "metrics_B2")}
            rec.update({
                "status": "executed", "declared_utc": ts,
                "experiment_type": "feature_block candidate result (H11)",
                "protocol_ref": "H11-prereg-absence-context",
                "model": "logistic, fit_oof walk-forward (scripts/"
                         "feature_block_ablation.py machinery, "
                         "recency_ablation.make_pipeline)",
                "leakage_check": "passed: truncation bit-identical "
                                 f"({checks['rows_checked']} rows) + membership "
                                 "discipline + identical OOF rows across arms",
            })
            if "per_fold_deltas" in r:
                rec["per_fold_deltas"] = r["per_fold_deltas"]
            if "per_season_deltas" in r:
                rec["per_season_deltas"] = r["per_season_deltas"]
            fh.write(json.dumps(rec) + "\n")
        fh.write(json.dumps({
            "name": "H11-verdict", "status": "executed", "declared_utc": ts,
            "experiment_type": "verdict_record",
            "protocol_ref": "H11-prereg-absence-context",
            **verdict,
            "coef_shrink_mean_abs": coef_diag.get("mean_abs_by_group", {}),
            "leakage_check": "passed: candidate columns are causal by "
                             "construction (prior-game squad + trailing player "
                             "logs only); truncation-invariance asserted in "
                             "code; imputation and scaling fit on train rows "
                             "only (ra.fit_oof); arms scored on identical "
                             "out-of-fold rows",
        }) + "\n")
    print(f"appended {len(results) + 1} records to {JSONL}", flush=True)
    print(f"\n{decision}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
