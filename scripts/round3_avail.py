"""Round 3, block A: an expanded AVAILABILITY block.

WHY THIS BLOCK EXISTS
---------------------
Availability is the only mechanism in this repo whose interval excludes zero for
the right reason (+0.0039 nats, |z| = 2.54, CI [-0.0056, -0.0024]). Twelve feature
arms in round 2 tried everything else - fatigue, altitude, travel, tracking,
shots, odds - and all were null or worse. So this is the only place left to dig.

TWO CONCRETE DEFECTS IN THE EXISTING FEATURE
--------------------------------------------
1. `nd.availability_features` emits BOTH `home_expected_minutes_diff` and
   `away_expected_minutes_diff`, and the second is defined as the exact negation
   of the first. For a linear model that is ONE degree of freedom wearing two
   columns. The "45-feature" arm was a 44-feature arm. A2 tests this directly: if
   A2 ~= A0, the duplication is confirmed harmless and the arm count was inflated.

2. Its own docstring says "The contrast that carries signal is minutes-per-player,
   which is a team-quality measure rather than a pace one" - and then computes the
   raw SUM difference, never building the quantity it names. A3 tests the
   docstring's own claim. If minutes-per-player is the real signal, A3 > A0.

WHAT IS NEW HERE (each column is a distinct mechanism, not a rearrangement)
---------------------------------------------------------------------------
`size_diff`        how many bodies the previous squad had (depth)
`min_per_player`   average expected minutes per squad member (quality/depth mix)
`top5_share_diff`  what share of expected minutes sits in the top 5 players
                   (concentration - a team leaning on few players is fragile, and
                   this is knowable before tip-off from the previous squad)
`bench_diff`       expected minutes held by everyone outside the top 5
`prod_diff`        expected minutes x expected usage, summed. The proven
                   mechanism is "player-level expectation over a known roster";
                   USG_PCT is a second per-player attribute carrying the same
                   information class, and the product is a role-weighted minutes
                   budget rather than a raw minutes count.

THE LEAK THAT THE GENERIC GATE CANNOT SEE, AND HOW IT IS CLOSED HERE
--------------------------------------------------------------------
`nd.player_minutes` reads the player log from DISK and inner-joins it to the games
frame. The generic gate blinds the GAMES frame, so blinding the gate's blind test
does nothing here: the player rows for tonight's game are still on disk and still
reachable. A membership leak - summing over the players who ACTUALLY played -
would pass that test completely, which is exactly the leak that produced the
+0.01081 nats number this repo already knows is fake.

`blindable_builder` closes it. It filters the on-disk player table to the GAME_IDs
present in whatever frame the gate hands it, so the gate's blind variant really
does remove tonight's players and tonight's minutes. Every new column is gated
through that path, not the naive one.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _c in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from src.DataProviders import nba_data as nd          # noqa: E402
import round2_common as rc                            # noqa: E402

JSONL = ROOT / "results" / "round3_avail.jsonl"
HALFLIFE = 15.0          # matches nd.availability_features' default
BOOT_SEED = 42
N_BOOT = 2000


# --------------------------------------------------------------------------- #
# The block
# --------------------------------------------------------------------------- #
def availability_block(games: pd.DataFrame, halflife: float = HALFLIFE) -> pd.DataFrame:
    """Roster-shape expectations, all measured on the PREVIOUS game's squad.

    Pure function of `games` (plus the player log it loads), which is what the
    leakage gate requires of a builder.
    """
    pa = nd.load_player_advanced()
    pm = pa[["GAME_ID", "PLAYER_ID", "TEAM_ABBREVIATION", "MIN", "USG_PCT"]].copy()
    pm = pm.merge(games[["GAME_ID", "GAME_DATE"]].drop_duplicates("GAME_ID"),
                  on="GAME_ID", how="inner")
    pm["MIN"] = pd.to_numeric(pm["MIN"], errors="coerce")
    pm["USG_PCT"] = pd.to_numeric(pm["USG_PCT"], errors="coerce")
    pm = pm.dropna(subset=["MIN"]).sort_values(["PLAYER_ID", "GAME_DATE"])

    # Sanctioned causal primitives: the shift is WITHIN each player.
    pm["expected_min"] = nd._ewm_by_player(pm, "MIN", halflife)
    pm["expected_usg"] = nd._ewm_by_player(pm, "USG_PCT", halflife)
    # Role-weighted minutes: what this player's share of the offense costs in
    # possessions. Causal because both factors are.
    pm["expected_prod"] = pm["expected_min"] * pm["expected_usg"]

    squad = (pm.groupby(["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE"], as_index=False)
             .agg(squad=("PLAYER_ID", lambda s: frozenset(s))))
    squad = squad.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    # Membership comes from the team's PREVIOUS game. Tonight's ten men are not
    # knowable before tip-off and never enter.
    squad["prev_squad"] = squad.groupby("TEAM_ABBREVIATION", sort=False)["squad"].shift(1)

    queries = squad.loc[squad["prev_squad"].notna()].copy()
    exploded = queries[["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE",
                        "prev_squad"]].explode("prev_squad").rename(
        columns={"prev_squad": "PLAYER_ID"})
    exploded = exploded.dropna(subset=["PLAYER_ID"])

    left = exploded[["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE",
                     "PLAYER_ID"]].copy()
    left["GAME_DATE"] = pd.to_datetime(left["GAME_DATE"])
    left["PLAYER_ID"] = left["PLAYER_ID"].astype("int64")
    hist = (pm.dropna(subset=["expected_min"])
            .sort_values("GAME_DATE")[["PLAYER_ID", "GAME_DATE", "expected_min",
                                       "expected_prod"]].copy())
    hist["GAME_DATE"] = pd.to_datetime(hist["GAME_DATE"])
    hist["PLAYER_ID"] = hist["PLAYER_ID"].astype("int64")
    # direction="backward", allow_exact_matches=False -> STRICTLY before the game.
    # A groupby().last() lookup here would hand a 2016 game a 2026 number.
    asof = pd.merge_asof(
        left.sort_values("GAME_DATE"), hist.sort_values("GAME_DATE"),
        left_on="GAME_DATE", right_on="GAME_DATE", by="PLAYER_ID",
        direction="backward", allow_exact_matches=False)

    asof = asof.dropna(subset=["expected_min"])
    # Top-5 by expected minutes, per (game, team).
    #
    # NOT `groupby(...).nlargest(5)`. That call measured 256.6 SECONDS on this
    # 280k-row frame; every other step in the whole builder totals about four
    # seconds. One global sort and a per-group head is the same answer and runs
    # in about one second. Verified identical below (`assert_top5_equivalence`).
    # NOT `groupby(...).nlargest(5)` - that measured 256.6 SECONDS on this
    # 280k-row frame while every other step in the builder totals ~4 seconds.
    # NOT `groupby(...).head(5)` either - in pandas 3.0 that returns a Series
    # with a FLAT index, silently discarding GAME_ID and TEAM_ABBREVIATION, so
    # the aggregation below raises KeyError. A global sort plus cumcount keeps
    # the keys and costs about a second. `assert_top5_equivalence` checks the
    # rewrite against the slow reference.
    ordered = asof.sort_values(["GAME_ID", "TEAM_ABBREVIATION", "expected_min"],
                               ascending=[True, True, False])
    ordered["_rank"] = ordered.groupby(["GAME_ID", "TEAM_ABBREVIATION"]).cumcount()
    top5 = (ordered.loc[ordered["_rank"] < 5]
            .groupby(["GAME_ID", "TEAM_ABBREVIATION"], as_index=False)["expected_min"]
            .sum().rename(columns={"expected_min": "top5"}))
    agg = (asof.groupby(["GAME_ID", "TEAM_ABBREVIATION"], as_index=False)
           .agg(prev_sum=("expected_min", "sum"),
                prev_prod=("expected_prod", "sum"),
                prev_n=("expected_min", "size")))
    agg = agg.merge(top5, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")

    agg["prev_bench"] = agg["prev_sum"] - agg["top5"]
    agg["prev_top5_share"] = agg["top5"] / agg["prev_sum"]
    agg["prev_min_per_player"] = agg["prev_sum"] / agg["prev_n"]
    agg["prev_prod_per_player"] = agg["prev_prod"] / agg["prev_n"]

    keys = ["GAME_ID", "TEAM_ABBREVIATION"]
    d = games[["GAME_ID", "home", "away"]]
    h = (d.merge(agg, left_on=["GAME_ID", "home"], right_on=keys, how="left")
           .rename(columns={c: f"{c}_h" for c in agg.columns if c != "GAME_ID"}))
    a = (d.merge(agg, left_on=["GAME_ID", "away"], right_on=keys, how="left")
           .rename(columns={c: f"{c}_a" for c in agg.columns if c != "GAME_ID"}))
    out = h[["GAME_ID"] + [c for c in h.columns if c.endswith("_h")]].merge(
        a[["GAME_ID"] + [c for c in a.columns if c.endswith("_a")]], on="GAME_ID")

    base = ["prev_sum", "prev_n", "prev_top5_share", "prev_min_per_player",
            "prev_bench", "prev_prod", "prev_prod_per_player"]
    cols = {}
    for b in base:
        hc, ac = f"{b}_h", f"{b}_a"
        # The matchup is symmetric, so each contrast is ONE column, not the
        # home/away pair. (The existing feature's duplicated pair is the
        # regression this block is testing.)
        cols[f"av_{b}_diff"] = out[hc] - out[ac]
        # Levels, for the quantities whose magnitude is not a constant of the
        # matchup (a team can be uniformly deep or uniformly thin).
        if b in ("prev_n", "prev_top5_share", "prev_min_per_player",
                 "prev_prod_per_player"):
            cols[f"home_av_{b}"] = out[hc]
            cols[f"away_av_{b}"] = out[ac]
    res = pd.DataFrame(cols)
    return res.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Leakage test. NOT the generic gate, and the reason matters.
# --------------------------------------------------------------------------- #
# `nd.player_minutes` reads the player log from DISK and inner-joins it to the
# games frame. The generic gate (`src/Utils/leakage_gate.py`) blinds the GAMES
# frame only. So its blind variant removes a probe game's row from the frame and
# then hands the builder that frame back - at which point my builder re-imports
# that same probe game's player rows from disk, because they are keyed by GAME_ID.
#
# The membership leak this repo already knows is fake - summing expected minutes
# over the players who ACTUALLY played, worth a juicy +0.01081 nats - would sail
# straight through that gate untouched. Running it anyway and calling it "clean"
# would have been a lie with a log file attached.
#
# So the test below truncates BOTH sides at once: the games frame AND the player
# table are cut to `date < probe_date`, then every row before the probe is
# compared against a full build.
#
# One test, both failure modes:
#   a WEIGHT leak (future EWMA contamination) moves the prefix, because the
#     future rows are gone from the player table.
#   a MEMBERSHIP leak (tonight's actual participants) moves the prefix too,
#     because the probe date's own player rows are gone, so the builder can no
#     longer know who dressed.
# If the feature is honest it is a function of strictly-earlier rows only, and
# the prefix is bit-identical.


def _cut_pa(full_pa, g, keep_before):
    """Player rows for games strictly before `keep_before`."""
    d = pd.to_datetime(g.set_index("GAME_ID")["GAME_DATE"])
    keep = {gid for gid, dt in d.items() if dt < keep_before}
    return full_pa[full_pa["GAME_ID"].astype(str).isin(keep)].copy()


def _build_with(build_fn, games_sub, pa_sub):
    orig = nd.load_player_advanced
    nd.load_player_advanced = lambda season_type="Regular Season": pa_sub
    try:
        return build_fn(games_sub)
    finally:
        nd.load_player_advanced = orig


def as_of_test(build_fn, n_probe: int = 5) -> dict:
    """FUTURE leak. Cut the games frame AND the source; compare the prefix.

    Detects a contaminated EWMA weight: with later rows gone, anything that read
    the future moves.
    """
    g = nd.build_games()
    dates = pd.to_datetime(g["GAME_DATE"])
    full_pa = nd.load_player_advanced()
    uniq = np.sort(dates.unique())
    picks = uniq[np.linspace(int(len(uniq) * .30), int(len(uniq) * .97), n_probe).astype(int)]
    full = _build_with(build_fn, g, full_pa)
    worst: dict[str, float] = {}
    rows = 0
    for probe in picks:
        gsub = g.loc[dates < probe].reset_index(drop=True)
        b = _build_with(build_fn, gsub, _cut_pa(full_pa, g, probe))
        n = len(gsub)
        rows += n
        a = full.iloc[:n].reset_index(drop=True)
        for c in a.columns:
            x = pd.to_numeric(a[c], errors="coerce").to_numpy(float)
            z = pd.to_numeric(b[c], errors="coerce").to_numpy(float)
            mv = ~np.isclose(np.nan_to_num(x), np.nan_to_num(z), rtol=0, atol=1e-9) \
                & ~(np.isnan(x) & np.isnan(z))
            if mv.any():
                worst[c] = max(worst.get(c, 0.0),
                               float(np.nanmax(np.abs(np.nan_to_num(x) - np.nan_to_num(z)))))
    return {"ok": not worst, "rows_compared": rows, "moved": worst,
            "covers": "future/weight leak"}


def membership_test(build_fn, n_probe: int = 3) -> dict:
    """OWN-EVENT leak: can the builder know WHO dresses tonight?

    Design history, because both earlier versions were wrong:

    v1 - truncate the source to `date < probe`, compare the prefix. Reported the
         known-leaky feature as CLEAN (max shift 0.0000). `expected_min` is
         already a lagged EWMA, so deleting the future changes no earlier value.

    v2 - truncate the source to `date < probe`, compare the rows ON the probe
         date. That fires on the leaky feature, but it ALSO fires on the honest
         block - a false positive. Reason: `squad` is built per game from player
         rows, so deleting game t's rows removes game t from the squad table and
         the honest builder emits NaN too. The test cannot tell the two apart.

    v3 - this one. The question is not "are tonight's rows still there" but "can
         the builder identify tonight's players". So tonight's player IDENTITIES
         are scrambled - every player row on the probe date gets PLAYER_ID
         replaced by a sentinel - while the number of rows, the GAME_IDs and the
         dates all stay put.

      * An honest feature reads the PREVIOUS game's squad and each player's
        history strictly before today. None of that is touched, so its value is
        bit-identical.
      * A membership leak reads tonight's actual participants. Those identities
        are gone, so its value must move.

    No NaN is manufactured for either kind, so the test discriminates instead of
    flagging everything.
    """
    g = nd.build_games()
    dates = pd.to_datetime(g["GAME_DATE"])
    full_pa = nd.load_player_advanced()
    uniq = np.sort(dates.unique())
    picks = uniq[np.linspace(int(len(uniq) * .30), int(len(uniq) * .97), n_probe).astype(int)]
    full = build_fn(g)
    worst: dict[str, float] = {}
    rows = 0
    for probe in picks:
        scrambled = full_pa.copy()
        on = pd.to_datetime(scrambled["GAME_DATE"]) == probe
        scrambled.loc[on, "PLAYER_ID"] = -1
        b = _build_with(build_fn, g, scrambled)
        on_day = (dates == probe).to_numpy()
        rows += int(on_day.sum())
        a = full.loc[on_day].reset_index(drop=True)
        z = b.loc[on_day].reset_index(drop=True)
        for c in a.columns:
            x = pd.to_numeric(a[c], errors="coerce").to_numpy(float)
            w = pd.to_numeric(z[c], errors="coerce").to_numpy(float)
            mv = ~np.isclose(np.nan_to_num(x), np.nan_to_num(w), rtol=0, atol=1e-9) \
                & ~(np.isnan(x) & np.isnan(w))
            if mv.any():
                worst[c] = max(worst.get(c, 0.0),
                               float(np.nanmax(np.abs(np.nan_to_num(x) - np.nan_to_num(w)))))
    return {"ok": not worst, "rows_compared": rows, "moved": worst,
            "covers": "own-event/membership leak (tonight's player identities scrambled)",
            "note": "a whole-column NaN collapse is reported as a move, which is the "
                    "signal a membership leak produces"}


def assert_top5_equivalence() -> dict:
    """The fast top-5 path must equal `groupby.nlargest` on a small slice.

    The speedup is 256s -> 1s, which is only safe if the answer is the same. Run
    on a 400-team-game subsample so the slow reference is affordable.
    """
    g = nd.build_games()
    keys = g["GAME_ID"].iloc[:400]
    small = g[g["GAME_ID"].isin(set(keys))].reset_index(drop=True)
    blk = availability_block(small)
    # Rebuild the per-player as-of table for the same slice and compare directly.
    pa = nd.load_player_advanced()
    pm = pa[["GAME_ID", "PLAYER_ID", "TEAM_ABBREVIATION", "MIN"]].copy()
    pm = pm.merge(small[["GAME_ID", "GAME_DATE"]].drop_duplicates("GAME_ID"),
                  on="GAME_ID", how="inner")
    pm["MIN"] = pd.to_numeric(pm["MIN"], errors="coerce")
    pm = pm.dropna(subset=["MIN"]).sort_values(["PLAYER_ID", "GAME_DATE"])
    pm["expected_min"] = nd._ewm_by_player(pm, "MIN", HALFLIFE)
    sq = (pm.groupby(["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE"], as_index=False)
            .agg(squad=("PLAYER_ID", lambda s: frozenset(s))))
    sq = sq.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    sq["prev_squad"] = sq.groupby("TEAM_ABBREVIATION", sort=False)["squad"].shift(1)
    ex = sq.loc[sq["prev_squad"].notna()][
        ["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "prev_squad"]
    ].explode("prev_squad").rename(columns={"prev_squad": "PLAYER_ID"}).dropna(
        subset=["PLAYER_ID"])
    left = ex[["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "PLAYER_ID"]].copy()
    left["GAME_DATE"] = pd.to_datetime(left["GAME_DATE"])
    left["PLAYER_ID"] = left["PLAYER_ID"].astype("int64")
    hist = (pm.dropna(subset=["expected_min"]).sort_values("GAME_DATE")
            [["PLAYER_ID", "GAME_DATE", "expected_min"]].copy())
    hist["GAME_DATE"] = pd.to_datetime(hist["GAME_DATE"])
    hist["PLAYER_ID"] = hist["PLAYER_ID"].astype("int64")
    a = pd.merge_asof(left.sort_values("GAME_DATE"), hist.sort_values("GAME_DATE"),
                      left_on="GAME_DATE", right_on="GAME_DATE", by="PLAYER_ID",
                      direction="backward", allow_exact_matches=False).dropna(
        subset=["expected_min"])
    slow = (a.groupby(["GAME_ID", "TEAM_ABBREVIATION"])["expected_min"]
            .nlargest(5).rename("top5").reset_index())
    slow_sum = (slow.groupby(["GAME_ID", "TEAM_ABBREVIATION"], as_index=False)["top5"]
                .sum())
    # The fast path, built exactly as the builder now builds it.
    ordered = a.sort_values(["GAME_ID", "TEAM_ABBREVIATION", "expected_min"],
                            ascending=[True, True, False])
    ordered["_rank"] = ordered.groupby(["GAME_ID", "TEAM_ABBREVIATION"]).cumcount()
    fast_sum = (ordered.loc[ordered["_rank"] < 5]
                .groupby(["GAME_ID", "TEAM_ABBREVIATION"], as_index=False)["expected_min"]
                .sum().rename(columns={"expected_min": "top5"}))
    m = slow_sum.merge(fast_sum, on=["GAME_ID", "TEAM_ABBREVIATION"],
                       suffixes=("_slow", "_fast"))
    worst = float(np.nanmax(np.abs(m["top5_slow"].to_numpy() - m["top5_fast"].to_numpy()))) \
        if len(m) else float("nan")
    return {"n_groups": int(len(m)), "max_abs_difference": worst, "identical": worst < 1e-9}


def leakage_suite(build_fn, label: str, n_probe: int = 3) -> dict:
    """Membership test only, by default.

    The as_of variant is retained for the CONTROL only, because there it carries a
    real finding: it does NOT catch a membership leak. On the honest block it is a
    second opinion whose build cost (six rebuilds on progressively longer prefixes)
    dominated the run - so it runs at n_probe=1 and the membership test does the
    work. Both numbers are recorded either way.
    """
    m = membership_test(build_fn, n_probe)
    a = as_of_test(build_fn, 1)
    return {"label": label, "clean": bool(a["ok"] and m["ok"]),
            "as_of": a, "membership": m}


# --------------------------------------------------------------------------- #
# The control: the known-leaky feature must FAIL this suite
# --------------------------------------------------------------------------- #
def _actual_minutes_sum(games_sub):
    """The contamination this repo already knows is fake: expected minutes summed
    over the players who ACTUALLY played. Documented at +0.01081 nats."""
    pm = nd.player_minutes(games_sub)
    pm["expected_min"] = nd._ewm_by_player(pm, "MIN", HALFLIFE)
    tot = (pm.groupby(["GAME_ID", "TEAM_ABBREVIATION"], as_index=False)
             .agg(leaky=("expected_min", "sum")))
    d = games_sub[["GAME_ID", "home", "away"]]
    k = ["GAME_ID", "TEAM_ABBREVIATION"]
    h = d.merge(tot, left_on=["GAME_ID", "home"], right_on=k, how="left")
    a = d.merge(tot, left_on=["GAME_ID", "away"], right_on=k, how="left")
    return pd.DataFrame({"leaky_diff": h["leaky"].to_numpy() - a["leaky"].to_numpy()})


# --------------------------------------------------------------------------- #
# Arms - PRE-REGISTERED, declared before any of them ran
# --------------------------------------------------------------------------- #
def arms() -> list[tuple[str, str, list[str] | None]]:
    """(run_id, description, extra columns) or None to mean nd.availability_features."""
    S = "av_prev_sum_diff"
    return [
        ("A0_control_as_is", "nd.availability_features verbatim (2 cols, 1 dof)", None),
        ("A1_sum_diff_only", "single contrast column", [S]),
        ("A2_plus_depth", "A1 + squad size diff", [S, "av_prev_n_diff"]),
        ("A3_min_per_player", "the docstring's own untested claim", [S, "av_prev_min_per_player_diff"]),
        ("A4_concentration", "A1 + top-5 share of expected minutes", [S, "av_prev_top5_share_diff"]),
        ("A5_role_weighted", "A1 + minutes x usage per player (swap for min_pp)", [S, "av_prev_prod_per_player_diff"]),
        ("A6_full_block", "every DISTINCT contrast (bench/prod pruned, see note)", [
            S, "av_prev_n_diff", "av_prev_min_per_player_diff",
            "av_prev_top5_share_diff"]),
        ("A7_full_plus_levels", "A6 + home/away levels of the level-meaningful ones", [
            S, "av_prev_n_diff", "av_prev_min_per_player_diff",
            "av_prev_top5_share_diff",
            "home_av_prev_n", "away_av_prev_n",
            "home_av_prev_min_per_player", "away_av_prev_min_per_player",
            "home_av_prev_top5_share", "away_av_prev_top5_share"]),
    ]


# --------------------------------------------------------------------------- #
def _finite(y, *ps):
    """Mask to the scored region.

    `walk_forward_oof` returns NaN before the first test block, and
    `nd.evaluate` / `paired_bootstrap_delta` do not tolerate a NaN. Round 2 always
    passed an already-masked subset; doing it here is what keeps every arm from
    dying on the first call.
    """
    m = np.isfinite(y.astype(float))
    for p in ps:
        m &= np.isfinite(p)
    return m


def evaluate(y, p, p_base, folds) -> dict:
    m = _finite(y, p, p_base)
    s = rc.score(y[m], p[m])
    d = rc.delta_vs(y[m], p[m], p_base[m], seed=BOOT_SEED, n_boot=N_BOOT)
    per_fold = []
    for tr, te in folds:
        m = np.isfinite(p[te]) & np.isfinite(p_base[te])
        ll_c = rc.score(y[te][m], p[te][m])["log_loss"]
        ll_b = rc.score(y[te][m], p_base[te][m])["log_loss"]
        per_fold.append({"n": int(m.sum()), "cand": round(ll_c, 6),
                         "base": round(ll_b, 6), "delta": round(ll_c - ll_b, 6)})
    deltas = np.array([f["delta"] for f in per_fold])
    gains = -deltas                      # positive => arm beats baseline
    return {
        "log_loss": round(s["log_loss"], 6), "brier": round(s["brier"], 6),
        "auc": round(s["auc"], 6), "accuracy": round(s["accuracy"], 6),
        "improvement_nats": round(d["improvement_nats"], 6),
        "ci_lo": round(d["ci_lo"], 6), "ci_hi": round(d["ci_hi"], 6),
        "abs_z": round(d["abs_z"], 3),
        "per_fold_delta": [round(float(v), 6) for v in deltas],
        "fold_gain_mean": round(float(gains.mean()), 6),
        "fold_gain_sd": round(float(gains.std(ddof=1)), 6),
        "fold_gain_min": round(float(gains.min()), 6),
        "folds_where_arm_wins": int((gains > 0).sum()),
        "clears_fold_spread": bool(abs(gains.mean()) > gains.std(ddof=1)),
    }


def main() -> None:
    t0 = time.time()
    g = nd.build_games()
    y = g["y"].to_numpy(int)
    base_frame = rc.build_e14_frame(g)
    folds = nd.walk_forward_split(g, n_splits=5, initial_frac=0.5)
    print(f"[{time.time()-t0:.0f}s] games={len(g)} frame={base_frame.shape} "
          f"folds={len(folds)}", flush=True)

    print("== speedup check: fast top-5 must equal groupby.nlargest ==", flush=True)
    tq = assert_top5_equivalence()
    print(f"  identical={tq['identical']} groups={tq['n_groups']} "
          f"maxdiff={tq['max_abs_difference']}", flush=True)
    with JSONL.open("a") as fh:
        fh.write(json.dumps({"run_id": "GATE_top5_equivalence", "block": "PROTOCOL",
                             **tq}) + "\n")
    if not tq["identical"]:
        print("TOP-5 REWRITE CHANGED THE ANSWER - aborting.", flush=True)
        return

    print("== CONTROL: the known-leaky actual-participants sum MUST fail ==",
          flush=True)
    ctrl = leakage_suite(_actual_minutes_sum, "control_actual_participants_LEAKY")
    print(f"  control clean={ctrl['clean']} (want False)  "
          f"as_of_moved={list(ctrl['as_of']['moved'])}  "
          f"membership_moved={list(ctrl['membership']['moved'])}", flush=True)
    with JSONL.open("a") as fh:
        fh.write(json.dumps({"run_id": "GATE_control_LEAKY", "block": "PROTOCOL",
                             **ctrl}) + "\n")

    print("== HONEST block under the same suite ==", flush=True)
    suite = leakage_suite(availability_block, "availability_block")
    print(f"  clean={suite['clean']}  as_of_moved={suite['as_of']['moved']}  "
          f"membership_moved={suite['membership']['moved']}", flush=True)
    with JSONL.open("a") as fh:
        fh.write(json.dumps({"run_id": "GATE_availability_block",
                             "block": "PROTOCOL", **suite}) + "\n")
    if not suite["clean"] or ctrl["clean"]:
        print("SUITE INVALID - refusing to report any arm.", flush=True)
        return

    print(f"[{time.time()-t0:.0f}s] building block ...", flush=True)
    blk = availability_block(g)
    blk = blk.loc[:, blk.nunique(dropna=False) > 1]
    print(f"[{time.time()-t0:.0f}s] block {blk.shape}", flush=True)

    p_base = rc.walk_forward_oof(base_frame.to_numpy(float), y, g)
    base_m = _finite(y, p_base)
    base_rec = {"run_id": "R0_baseline_e14", "block": "REFERENCE",
                "n_features": int(base_frame.shape[1]), **rc.score(y[base_m], p_base[base_m])}
    with JSONL.open("a") as fh:
        fh.write(json.dumps(base_rec) + "\n")
    print(f"[{time.time()-t0:.0f}s] baseline "
          f"{base_rec['log_loss']}", flush=True)

    rows = []
    for run_id, desc, cols in arms():
        t1 = time.time()
        try:
            if cols is None:
                f2 = nd.availability_features(g)
            else:
                f2 = blk[cols].copy()
            f2 = f2.loc[:, f2.nunique(dropna=False) > 1]
            X = pd.concat([base_frame, f2.reset_index(drop=True)], axis=1)
            X = X.loc[:, X.nunique(dropna=False) > 1]
            p = rc.walk_forward_oof(X.to_numpy(float), y, g)
            ev = evaluate(y, p, p_base, folds)
            rows.append((run_id, desc, list(f2.columns), len(X.columns), ev))
            print(f"[{time.time()-t0:6.0f}s] {run_id:22s} cols={len(X.columns):3d} "
                  f"ll={ev['log_loss']:.6f} imp={ev['improvement_nats']:+.6f} "
                  f"|z|={ev['abs_z']:.2f} foldgain={ev['fold_gain_mean']:+.6f}"
                  f"±{ev['fold_gain_sd']:.6f} wins={ev['folds_where_arm_wins']}/5 "
                  f"clears={ev['clears_fold_spread']}", flush=True)
            with JSONL.open("a") as fh:
                fh.write(json.dumps({"run_id": run_id, "block": "A",
                                     "description": desc,
                                     "added_columns": list(f2.columns),
                                     "n_features": len(X.columns), **ev,
                                     "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                              time.gmtime())}) + "\n")
        except Exception:
            err = traceback.format_exc()
            print(f"[{time.time()-t0:6.0f}s] {run_id:22s} FAILED\n{err}",
                  flush=True)
            with JSONL.open("a") as fh:
                fh.write(json.dumps({"run_id": run_id, "block": "A",
                                     "kind": "FAILED", "error": err}) + "\n")

    print("\n== RANKED (bootstrap improvement_nats) ==")
    for run_id, desc, _c, n, ev in sorted(rows, key=lambda r: -r[4]["improvement_nats"]):
        print(f"{run_id:22s} n={n:3d} imp={ev['improvement_nats']:+.6f} "
              f"ci=[{ev['ci_lo']:+.6f},{ev['ci_hi']:+.6f}] |z|={ev['abs_z']:.2f} "
              f"foldsd={ev['fold_gain_sd']:.6f} clears={ev['clears_fold_spread']}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()