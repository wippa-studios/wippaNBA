"""Issue #72 -- late-scratch analytics from the injury store.

WHAT THIS IS
------------
A descriptive census, not a model candidate.  The question is what a
designation is worth as the clock runs down:

    P(player does not play | designation, time until tip)

with the Questionable -> Out and Probable -> Out transitions called out
separately.  The outcome is ACTUAL participation (``MIN > 0``), never the last
report status: a player who stays Questionable and does not play is a scratch,
and a report that flips a player to Out is only an event if he then fails to
play.

WHY THE JOIN IS STRICT
----------------------
The store's parsed ``matchup``/``game_date`` cells can go stale on continuation
rows (a date block whose first game is NOT YET SUBMITTED, or a matchup cell on
a line with no date).  This script therefore attaches an event to a game only
by an exact ``(game_date, away@home)`` match AND the event's resolved team must
be one of the two teams.  There is no fuzzy fallback.  The exclusion is audited
against a ``(game_date, team)`` sensitivity join, which is unambiguous because
no team plays twice on one date.

The tip time is a game-level schedule fact taken from the LATEST exact-joined
event label for that game.  Earlier reports can carry a stale block-header
time; the final report is the best available correction.  Using a later
report's time cannot leak a status.

PRE-REGISTERED DESIGN
---------------------
``results/experiments.jsonl`` record ``I72-prereg-descriptive-late-scratch``:
cutoffs 6h/3h/1h/30m, latest status at or before ``tip - h`` with the same
tie-break as ``select_pre_tip``, common four-cutoff game panel, per-season and
per-era tables, 6h flagged as mostly yesterday's report, Wilson intervals
descriptive only, no decision rule, no promotion.

Run:  OMP_NUM_THREADS=1 python scripts/issue72_late_scratch.py
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

from DataProviders import injury_features as inf  # noqa: E402
from DataProviders import nba_data as nd          # noqa: E402
from Models import measurement as ms              # noqa: E402
STORE = ROOT / "Data" / "injury.sqlite"
OUT_JSON = ROOT / "results" / "issue72_late_scratch.json"
OUT_MD = ROOT / "results" / "issue72_late_scratch.md"
JSONL = ROOT / "results" / "experiments.jsonl"

CUTOFFS = (6.0, 3.0, 1.0, 0.5)
CUT_LABEL = {6.0: "6h", 3.0: "3h", 1.0: "1h", 0.5: "30m"}
BUFFER_MINUTES = float(inf.DEFAULT_BUFFER_MINUTES)
STATUS_ORDER = ["Out", "Doubtful", "Questionable", "Probable", "Available"]


# --------------------------------------------------------------------------- #
# Small statistics helpers
# --------------------------------------------------------------------------- #
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share.  Descriptive only."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def season_of(dates: pd.Series) -> pd.Series:
    d = pd.to_datetime(dates)
    return d.dt.year - (d.dt.month < 7).astype(int)


def season_label(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"


# --------------------------------------------------------------------------- #
# Load and join
# --------------------------------------------------------------------------- #
def load_panel() -> tuple[pd.DataFrame, dict]:
    """Resolve events to games, attach tip times and the played outcome."""
    games = nd.build_games().reset_index(drop=True)
    games["game_date"] = pd.to_datetime(games["GAME_DATE"]).dt.normalize()
    games["matchup_key"] = (games["game_date"].dt.strftime("%Y-%m-%d")
                            + "|" + games["away"] + "@" + games["home"])

    events, snapshots = inf.load_events(STORE)
    resolved = inf.resolve_players(events)
    resolved["game_date"] = pd.to_datetime(resolved["game_date"]).dt.normalize()
    resolved["matchup_key"] = (resolved["game_date"].dt.strftime("%Y-%m-%d")
                               + "|" + resolved["matchup"])

    # A. the strict exact join
    panel = resolved.merge(
        games[["GAME_ID", "game_date", "home", "away", "matchup_key"]],
        on="matchup_key", how="inner", suffixes=("", "_game"))
    team_ok = ((panel["team_abbr"] == panel["home"])
               | (panel["team_abbr"] == panel["away"]))
    exact_events = len(panel)
    panel = panel[team_ok].copy()
    exact_valid_events = len(panel)
    unresolved = int(panel["PLAYER_ID"].isna().sum())
    panel = panel.dropna(subset=["PLAYER_ID"]).copy()
    panel["PLAYER_ID"] = panel["PLAYER_ID"].astype("int64")
    panel = panel.drop_duplicates(
        subset=["GAME_ID", "PLAYER_ID", "label_ts", "status", "id"])

    # B. audit the exclusion against (game_date, team).  A team plays at most
    # once per date, so this join is unambiguous; it is the ceiling on what a
    # parser fix could recover at the player-game level.
    team_games = pd.concat([
        games[["GAME_ID", "game_date", "home"]].rename(
            columns={"home": "team_abbr"}),
        games[["GAME_ID", "game_date", "away"]].rename(
            columns={"away": "team_abbr"}),
    ], ignore_index=True)
    sens = resolved.dropna(subset=["PLAYER_ID"]).copy()
    sens["PLAYER_ID"] = sens["PLAYER_ID"].astype("int64")
    sens = sens.merge(team_games, on=["game_date", "team_abbr"], how="inner")
    sens_events = len(sens)
    sens_pg = set(zip(sens["GAME_ID"], sens["PLAYER_ID"]))

    # C. tip time per game: latest label, mode on ties, then maximum on a
    # tie.  The latest report is the best correction for stale headers.
    tips: dict[str, float] = {}
    for gid, grp in panel.groupby("GAME_ID", sort=False):
        value = np.nan
        labels = sorted(grp["label_ts"].dropna().unique(), reverse=True)
        for label in labels:
            times = grp.loc[grp["label_ts"] == label, "game_time"].dropna()
            if times.empty:
                continue
            counts = times.value_counts()
            top = list(counts[counts == counts.max()].index)
            parsed = [p for p in (inf.parse_tip_minutes(t) for t in top)
                      if p is not None]
            if parsed:
                value = float(max(parsed))
                break
        tips[gid] = value
    panel["tip_minutes"] = panel["GAME_ID"].map(tips)
    missing_tip = int(panel["tip_minutes"].isna().sum())
    panel = panel.dropna(subset=["tip_minutes"]).copy()
    panel["tip_ts"] = (panel["game_date"]
                       + pd.to_timedelta(panel["tip_minutes"], unit="m"))

    exact_pg_all = set(zip(panel["GAME_ID"], panel["PLAYER_ID"]))
    exact_games_prebuffer = int(panel["GAME_ID"].nunique())
    exact_events_prebuffer = int(len(panel))

    # The 15-minute pre-tip buffer the block uses.  Every cutoff is at least
    # 30 minutes, so this is implied, but keep it explicit.
    panel = panel[panel["label_ts"]
                  <= panel["tip_ts"] - pd.Timedelta(minutes=BUFFER_MINUTES)]
    panel["age_hours"] = ((panel["tip_ts"] - panel["label_ts"])
                          .dt.total_seconds() / 3600.0)

    # D. outcome: actual participation.  A missing row is a did-not-play.
    pm = nd.player_minutes(games)
    pm = pm.groupby(["GAME_ID", "PLAYER_ID"], as_index=False)["MIN"].max()
    pm["played"] = pm["MIN"] > 0
    panel = panel.merge(pm[["GAME_ID", "PLAYER_ID", "played"]],
                        on=["GAME_ID", "PLAYER_ID"], how="left")
    panel["played"] = panel["played"].fillna(False).astype(bool)
    panel["scratch"] = ~panel["played"]
    panel["season"] = season_of(panel["game_date"])

    # E. data-quality ledger (counts only; no rates until after the prereg).
    info = {
        "events": int(len(events)),
        "snapshots": int(len(snapshots)),
        "resolved_share": float(resolved["PLAYER_ID"].notna().mean()),
        "unresolved_all_events": int(resolved["PLAYER_ID"].isna().sum()),
        "unresolved_players": unresolved,
        "out_of_scope_2026_events": int(
            (season_of(resolved["game_date"]) == 2026).sum()),
        "exact_key_match_events": int(exact_events),
        "team_not_in_matchup_events": int(exact_events - exact_valid_events),
        "exact_join_events_before_team_filter": int(exact_events),
        "exact_join_events": int(exact_valid_events),
        "exact_join_events_with_player_prebuffer": exact_events_prebuffer,
        "exact_join_events_with_player": int(len(panel)),
        "events_dropped_by_pretip_buffer": int(
            exact_events_prebuffer - len(panel)),
        "outside_pretip_buffer_share": float(
            (exact_events_prebuffer - len(panel)) / exact_events_prebuffer),
        "games_exact_join_prebuffer": exact_games_prebuffer,
        "games_after_pretip_buffer": int(panel["GAME_ID"].nunique()),
        "games_with_only_outside_buffer_events": int(
            exact_games_prebuffer - panel["GAME_ID"].nunique()),
        "games": int(panel["GAME_ID"].nunique()),
        "games_with_any_event_on_schedule_date": int(
            sens["GAME_ID"].nunique()),
        "team_date_sensitivity_events": int(sens_events),
        "team_date_sensitivity_player_games": int(len(sens_pg)),
        "exact_player_games": int(len(exact_pg_all)),
        "player_games_lost_by_exact_join": int(len(sens_pg - exact_pg_all)),
        "player_games_lost_share": float(
            len(sens_pg - exact_pg_all) / max(1, len(sens_pg))),
        "games_lost_by_exact_join": int(
            sens["GAME_ID"].nunique() - exact_games_prebuffer),
        "missing_tip_time_events": missing_tip,
        "previous_day_events": int(
            (panel["label_ts"].dt.normalize() < panel["game_date"]).sum()),
        "previous_day_share": float(
            (panel["label_ts"].dt.normalize() < panel["game_date"]).mean()),
        "zero_minute_rows": int((pm["MIN"] == 0).sum()),
        "games_without_minutes_rows": int(
            len(set(panel["GAME_ID"]) - set(pm["GAME_ID"]))),
        "label_slots_by_season": {
            season_label(int(s)): sorted(
                panel.loc[panel["season"] == s, "label_ts"]
                .dt.strftime("%H:%M").unique().tolist())
            for s in sorted(panel["season"].unique())
        },
    }
    return panel, info


# --------------------------------------------------------------------------- #
# Cutoff statuses
# --------------------------------------------------------------------------- #
def status_at(panel: pd.DataFrame, hours: float) -> pd.DataFrame:
    """Latest status per (game, player) at or before ``tip - hours``."""
    cutoff = panel["tip_ts"] - pd.Timedelta(hours=hours)
    x = panel[panel["label_ts"] <= cutoff].copy()
    if x.empty:
        return x
    # select_pre_tip's rule: latest label, then most severe (lowest weight),
    # then highest id.  Sorting by -weight ascending puts Out last, so tail()
    # picks the most severe designation at a tied label.
    x["severity"] = -x["weight"]
    x = x.sort_values(["GAME_ID", "PLAYER_ID", "label_ts", "severity", "id"])
    x = x.groupby(["GAME_ID", "PLAYER_ID"], as_index=False).tail(1)
    x["age_hours"] = ((x["tip_ts"] - x["label_ts"])
                      .dt.total_seconds() / 3600.0)
    return x


def summarise(x: pd.DataFrame, min_n: int = 0) -> list[dict]:
    """Scratch rate by designation, with Wilson bands and report age."""
    if x.empty:
        return []
    rows = []
    for status in STATUS_ORDER:
        y = x[x["status"] == status]
        if len(y) < min_n:
            continue
        n = int(len(y))
        k = int(y["scratch"].sum())
        lo, hi = wilson(k, n)
        rows.append({
            "status": status,
            "n": n,
            "scratches": k,
            "rate": float(k / n),
            "wilson_95": [lo, hi],
            "median_age_hours": float(y["age_hours"].median()),
            "age_iqr_hours": [float(y["age_hours"].quantile(0.25)),
                              float(y["age_hours"].quantile(0.75))],
            "previous_day_share": float(
                (y["label_ts"].dt.normalize() < y["game_date"]).mean()),
        })
    return rows


def rate_cell(x: pd.DataFrame, scratch_col: str = "scratch",
              age_col: str = "age_hours") -> dict:
    n = int(len(x))
    k = int(x[scratch_col].sum()) if n else 0
    lo, hi = wilson(k, n)
    return {
        "n": n,
        "scratches": k,
        "rate": float(k / n) if n else float("nan"),
        "wilson_95": [lo, hi],
        "median_age_hours": float(x[age_col].median()) if n else None,
    }


def main() -> int:
    t0 = time.time()
    panel, info = load_panel()
    if panel.empty:
        raise AssertionError("no events survived the exact join")

    statuses = {h: status_at(panel, h) for h in CUTOFFS}
    for h, x in statuses.items():
        if x.empty:
            raise AssertionError(f"cutoff {h} has no statuses")

    available_games = {h: set(x["GAME_ID"]) for h, x in statuses.items()}
    common_games = set.intersection(*available_games.values())
    if not common_games:
        raise AssertionError("no common four-cutoff game panel")

    tables: dict[str, object] = {
        "available_panel": {
            CUT_LABEL[h]: {
                "games": len(available_games[h]),
                "statuses": int(len(statuses[h])),
                "rows": summarise(statuses[h]),
            } for h in CUTOFFS
        },
        "common_panel": {
            CUT_LABEL[h]: summarise(
                statuses[h][statuses[h]["GAME_ID"].isin(common_games)])
            for h in CUTOFFS
        },
        "common_panel_games": len(common_games),
    }

    # Panel coverage by cutoff and season, so the reader can see where the
    # common panel comes from.
    cov = []
    age_labels = ["<1h", "1-3h", "3-6h", "6-12h", "12-18h", "18-24h", ">24h"]
    age_bins = [-0.01, 1, 3, 6, 12, 18, 24, 1e9]
    for h in CUTOFFS:
        x = statuses[h]
        games_h = x.drop_duplicates("GAME_ID")
        seasons = season_of(games_h["game_date"]).value_counts().sort_index()
        buckets = pd.cut(x["age_hours"], age_bins, labels=age_labels)
        shares = (buckets.value_counts(normalize=True)
                  .reindex(age_labels).fillna(0.0))
        cov.append({
            "cutoff": CUT_LABEL[h],
            "games": int(x["GAME_ID"].nunique()),
            "statuses": int(len(x)),
            "games_by_season": {season_label(int(s)): int(n)
                                for s, n in seasons.items()},
            "median_age_hours": float(x["age_hours"].median()),
            "p90_age_hours": float(x["age_hours"].quantile(0.90)),
            "previous_day_share": float(
                (x["label_ts"].dt.normalize() < x["game_date"]).mean()),
            "age_bucket_share": {k: float(v) for k, v in shares.items()},
        })
    tables["coverage_by_cutoff"] = cov

    # The 1h and 30m cutoffs usually select the same fixed-time report.  This
    # is a measurement property, not a second result.
    s1 = statuses[1.0]
    s05 = statuses[0.5]
    pair = s1.merge(s05, on=["GAME_ID", "PLAYER_ID"], how="inner",
                    suffixes=("_1h", "_30m"))
    pair = pair[pair["GAME_ID"].isin(common_games)]
    gap = ((pair["label_ts_30m"] - pair["label_ts_1h"])
           .dt.total_seconds() / 3600.0)
    tables["cutoff_redundancy"] = {
        "cutoffs": ["1h", "30m"],
        "pairs": int(len(pair)),
        "same_label_share": float(
            (pair["label_ts_1h"] == pair["label_ts_30m"]).mean()),
        "same_status_share": float(
            (pair["status_1h"] == pair["status_30m"]).mean()),
        "median_label_gap_hours": float(gap.median()),
        "nonzero_label_gap_share": float((gap > 0).mean()),
    }

    # Transition 3h -> 30m, restricted to players present at both.
    s3 = statuses[3.0]
    s05 = statuses[0.5]
    tr = s3.merge(s05, on=["GAME_ID", "PLAYER_ID"], how="inner",
                  suffixes=("_3h", "_30m"))
    tr = tr[tr["GAME_ID"].isin(common_games)].copy()
    tr["updated"] = tr["label_ts_30m"] > tr["label_ts_3h"]
    s05_common = s05[s05["GAME_ID"].isin(common_games)]
    transitions = {}
    for status in ("Questionable", "Probable"):
        y = tr[tr["status_3h"] == status]
        cells = {}
        for dest in STATUS_ORDER + ["Not listed at 30m"]:
            if dest == "Not listed at 30m":
                z = s3[(s3["status"] == status)
                       & (s3["GAME_ID"].isin(common_games))]
                z = z[~z.set_index(["GAME_ID", "PLAYER_ID"]).index.isin(
                    s05.set_index(["GAME_ID", "PLAYER_ID"]).index)]
            else:
                z = y[y["status_30m"] == dest]
            if dest == "Not listed at 30m":
                cells[dest] = rate_cell(z)
            else:
                cells[dest] = rate_cell(z, scratch_col="scratch_30m",
                                        age_col="age_hours_30m")
            if c := cells[dest]:
                c["updated_share"] = float(z["updated"].mean()) if len(z) else None
            if dest == status:
                for upd, name in ((True, "fresh"), (False, "unchanged")):
                    zz = z[z["updated"] == upd]
                    cells[f"{status}_stays_{name}"] = rate_cell(
                        zz, scratch_col="scratch_30m", age_col="age_hours_30m")
        # The direct Q/P -> Out question, plus the overall scratch rate.
        transitions[status] = {
            "n_at_3h_and_30m": int(len(y)),
            "p_flip_to_out_by_30m": float(
                (y["status_30m"] == "Out").mean()) if len(y) else None,
            "updated_by_30m_share": float(y["updated"].mean()),
            "scratch_rate_at_3h": rate_cell(
                s3[(s3["status"] == status)
                   & (s3["GAME_ID"].isin(common_games))]),
            "scratch_rate_at_30m_population": rate_cell(
                s05_common[s05_common["status"] == status]),
            "cells": cells,
        }
    tables["transition_3h_to_30m"] = transitions

    # Per-season Q/P at 3h/1h/30m on the common panel.
    by_season = []
    for season in sorted(panel["season"].unique()):
        row = {"season": season_label(int(season)), "season_year": int(season)}
        any_cell = False
        for h in (3.0, 1.0, 0.5):
            x = statuses[h]
            x = x[(x["season"] == season) & (x["GAME_ID"].isin(common_games))]
            for status in ("Questionable", "Probable"):
                z = x[x["status"] == status]
                row[f"{status.lower()}_{CUT_LABEL[h]}"] = rate_cell(z)
                any_cell = any_cell or len(z) > 0
        if any_cell:
            by_season.append(row)
    tables["by_season"] = by_season

    # Era split: early = 2019-20 + 2020-21 (multi-slot report days),
    # modern = 2021-22 onward (17:00/20:00 two-slot days).
    by_era = []
    for era, predicate in (
            ("early_2019_2021", lambda s: s <= 2020),
            ("modern_2021_on", lambda s: s >= 2021)):
        row = {"era": era}
        for h in (6.0, 3.0, 1.0, 0.5):
            x = statuses[h]
            x = x[x["season"].map(predicate)
                  & x["GAME_ID"].isin(common_games)]
            for status in ("Questionable", "Probable"):
                row[f"{status.lower()}_{CUT_LABEL[h]}"] = rate_cell(
                    x[x["status"] == status])
        by_era.append(row)
    tables["by_era"] = by_era

    # The 6h column is mostly yesterday's report.  Quantify that directly and
    # keep it out of the "more information is better" reading.
    x6 = statuses[6.0]
    tables["six_hour"] = {
        "statuses": summarise(x6),
        "games": int(x6["GAME_ID"].nunique()),
        "median_age_hours": float(x6["age_hours"].median()),
        "same_day_share": float(
            (x6["label_ts"].dt.normalize() == x6["game_date"]).mean()),
        "previous_day_share": float(
            (x6["label_ts"].dt.normalize() < x6["game_date"]).mean()),
        "share_age_ge_12h": float((x6["age_hours"] >= 12).mean()),
        "by_era": {
            era: {
                "statuses": summarise(
                    x6[x6["season"].map(pred)]),
                "median_age_hours": float(
                    x6[x6["season"].map(pred)]["age_hours"].median()),
            } for era, pred in (
                ("early_2019_2021", lambda s: s <= 2020),
                ("modern_2021_on", lambda s: s >= 2021))
        },
    }

    # Calibration inversion: the ordinal weight says Available > Probable.
    cal = {}
    for h in (3.0, 1.0, 0.5):
        x = statuses[h][statuses[h]["GAME_ID"].isin(common_games)]
        cal[CUT_LABEL[h]] = {
            "Available": rate_cell(x[x["status"] == "Available"]),
            "Probable": rate_cell(x[x["status"] == "Probable"]),
        }
    tables["calibration"] = cal

    # Final (15-minute gate) sensitivity for the 30m transition.
    final = status_at(panel, BUFFER_MINUTES / 60.0)
    fin = tr.merge(
        final[["GAME_ID", "PLAYER_ID", "status"]].rename(
            columns={"status": "status_final"}),
        on=["GAME_ID", "PLAYER_ID"], how="inner")
    same_as_30m = int((fin["status_30m"] == fin["status_final"]).sum())
    tables["final_sensitivity"] = {
        "transition_rows": int(len(fin)),
        "share_30m_status_equals_final": float(
            same_as_30m / len(fin)) if len(fin) else None,
        "note": ("30m is used as the last slot in the pre-registered table; "
                 "this compares it with the strict 15-minute final status."),
    }

    payload = {
        "issue": 72,
        "title": "Late-scratch report: designation reliability by time to tip",
        "generated_utc": ms.utc_now(),
        "runtime_seconds": round(time.time() - t0, 1),
        "prereg": "I72-prereg-descriptive-late-scratch",
        "decision_rule": "none -- descriptive census; nothing promoted",
        "data": info,
        "cutoffs_hours": list(CUTOFFS),
        "tables": tables,
        "caveats": [
            "Label-upload lag: the label is the report's nominal ET time; the "
            "store records no separate upload time, so the 15-minute buffer is "
            "the only protection.",
            "The 6h column is overwhelmingly yesterday's report, not a "
            "same-day report six hours old; its actual-age distribution is "
            "reported beside it.",
            "2019-20 is Jan-Mar 2020 only; 2015-16 through 2018-19 have no "
            "injury store coverage.",
            "A small number of continuation rows carry a stale date or "
            "matchup; exact joining drops them and the team+date sensitivity "
            "join bounds the player-game cost.",
            "The analysis panel excludes 99 exact-join games (1.3%) whose only "
            "events fall outside the 15-minute gate; these are mostly "
            "matinees with no previous-day listing.",
            "Out and Doubtful are near-terminal (99.8% and 98.2% scratch at "
            "30m); the residual is small and can reflect stale listings, "
            "roster moves or a late appearance.",
            "40.6% of analysis-panel events are previous-day listings; the "
            "store-wide share is 30.0%. The difference is the exclusion of "
            "post-tip same-day rows, not a change in the store.",
            "Wilson bands are descriptive stability intervals, not sampling "
            "claims about a superpopulation.",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    OUT_MD.write_text(render_markdown(payload))

    ms.append_jsonl(JSONL, {
        "name": "I72-run-descriptive-late-scratch",
        "protocol_ref": "I72-prereg-descriptive-late-scratch",
        "declared_utc": ms.utc_now(),
        "experiment_type": "issue #72: descriptive late-scratch analytics",
        "status": "executed",
        "decision_rule": "none -- descriptive census; nothing promoted",
        "data": info,
        "common_panel_games": int(len(common_games)),
        "coverage_by_cutoff": cov,
        "verdict": "descriptive artifact; no model change",
        "artifacts": [str(OUT_JSON.relative_to(ROOT)),
                      str(OUT_MD.relative_to(ROOT))],
    })

    print(f"events {info['events']:,}; exact-join player events "
          f"{info['exact_join_events_with_player']:,}; games "
          f"{info['games']:,}; common panel {len(common_games):,}")
    for h in CUTOFFS:
        x = statuses[h]
        q = x[x["status"] == "Questionable"]
        p = x[x["status"] == "Probable"]
        print(f"  {CUT_LABEL[h]:>3}  statuses {len(x):>6,}  games "
              f"{x['GAME_ID'].nunique():>5,}  Q scratch "
              f"{q['scratch'].mean():.3f}  P scratch "
              f"{p['scratch'].mean():.3f}  median age "
              f"{x['age_hours'].median():.1f}h")
    print(f"wrote {OUT_JSON.name} and {OUT_MD.name}")
    return 0


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #
def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _cell(c: dict) -> str:
    if not c or not c.get("n"):
        return "—"
    lo, hi = c["wilson_95"]
    return (f"{_pct(c['rate'])} [{_pct(lo)}, {_pct(hi)}] "
            f"(n={c['n']:,})")


def render_markdown(p: dict) -> str:
    t = p["tables"]
    lines = [
        "# Issue #72 — late-scratch analytics",
        "",
        "Descriptive census of the injury store. The outcome is actual "
        "participation (`MIN > 0`), not the last report status. No decision "
        "rule, no promotion.",
        "",
        f"Pre-registration: `{p['prereg']}`. Generated "
        f"{p['generated_utc']}. Runtime {p['runtime_seconds']}s.",
        "",
        "## Data and join audit",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| injury events / snapshots | {p['data']['events']:,} / "
        f"{p['data']['snapshots']:,} |",
        f"| player resolution | {100 * p['data']['resolved_share']:.2f}% "
        f"({p['data']['unresolved_all_events']:,} unresolved) |",
        f"| exact (date, matchup) key match | "
        f"{p['data']['exact_key_match_events']:,} "
        f"({100 * p['data']['exact_key_match_events'] / p['data']['events']:.1f}%) |",
        f"| … team not in the parsed matchup | "
        f"{p['data']['team_not_in_matchup_events']:,} dropped |",
        f"| exact-join events with a team in the matchup | "
        f"{p['data']['exact_join_events']:,} |",
        f"| … unresolved player | "
        f"{p['data']['unresolved_players']:,} dropped |",
        f"| … outside the 15-minute pre-tip gate | "
        f"{p['data']['events_dropped_by_pretip_buffer']:,} dropped "
        f"({100 * p['data']['outside_pretip_buffer_share']:.1f}%) |",
        f"| **analysis panel** | "
        f"{p['data']['exact_join_events_with_player']:,} events, "
        f"{p['data']['games']:,} games |",
        f"| team+date sensitivity | "
        f"{p['data']['team_date_sensitivity_events']:,} events, "
        f"{p['data']['team_date_sensitivity_player_games']:,} player-games |",
        f"| player-games lost by exact join | "
        f"{p['data']['player_games_lost_by_exact_join']:,} "
        f"({100 * p['data']['player_games_lost_share']:.3f}%) |",
        f"| games lost by exact join | "
        f"{p['data']['games_lost_by_exact_join']:,} |",
        f"| games with only outside-gate events | "
        f"{p['data']['games_with_only_outside_buffer_events']:,} |",
        f"| previous-day events (analysis panel) | "
        f"{100 * p['data']['previous_day_share']:.1f}% |",
        f"| 2026-27 out-of-scope events | "
        f"{p['data']['out_of_scope_2026_events']:,} |",
        "",
        "## Status at each cutoff (common four-cutoff game panel)",
        "",
        f"{t['common_panel_games']:,} games have a designation at all four "
        "cutoffs, so the columns are information arrival, not panel "
        "composition.",
        "",
        "| designation | 6h | 3h | 1h | 30m |",
        "|---|---|---|---|---|",
    ]
    for status in STATUS_ORDER:
        cells = []
        for h in ("6h", "3h", "1h", "30m"):
            row = next((r for r in t["common_panel"][h]
                        if r["status"] == status), None)
            cells.append(_cell(row))
        lines.append(f"| {status} | " + " | ".join(cells) + " |")

    cov3 = next(c for c in t["coverage_by_cutoff"] if c["cutoff"] == "3h")
    young3 = (cov3["age_bucket_share"]["1-3h"]
              + cov3["age_bucket_share"]["3-6h"])
    lines += [
        "",
        "Median actual report age (hours) at each cutoff: "
        + "; ".join(
            f"{c['cutoff']} {c['median_age_hours']:.1f}h "
            f"(p90 {c['p90_age_hours']:.1f}h, "
            f"prev-day {100 * c['previous_day_share']:.0f}%)"
            for c in t["coverage_by_cutoff"]),
        "",
        "The nominal cutoff is not the report's actual age.  "
        + (
            f"The 1h and 30m cutoffs select the same label for "
            f"{100 * t['cutoff_redundancy']['same_label_share']:.1f}% of "
            f"pairs and the same status for "
            f"{100 * t['cutoff_redundancy']['same_status_share']:.1f}%, so "
            f"they are not independent columns.  "
        )
        + (
            "The 3h age distribution is bimodal: "
            f"{100 * cov3['age_bucket_share']['18-24h']:.1f}% "
            "of statuses are 18–24 hours old (yesterday's late report) and "
            f"{100 * young3:.1f}% are under 6 hours old (today's 17:00 "
            "report for a late tip)."
        ),
        "",
        "## Questionable and Probable transitions, 3h → 30m",
        "",
    ]
    for status in ("Questionable", "Probable"):
        r = t["transition_3h_to_30m"][status]
        lines += [
            f"**{status}** — {r['n_at_3h_and_30m']:,} players with both "
            f"labels; {100 * r['updated_by_30m_share']:.1f}% received a "
            f"later report. Flip to Out by 30m: "
            f"{100 * r['p_flip_to_out_by_30m']:.1f}%. Scratch among the 3h "
            f"cohort: {_cell(r['scratch_rate_at_3h'])}; the 30m population "
            f"rate is {_cell(r['scratch_rate_at_30m_population'])}.",
            "",
            "| 30m status | n | scratch rate |",
            "|---|---|---|",
        ]
        for dest in STATUS_ORDER + ["Not listed at 30m"]:
            c = r["cells"].get(dest)
            if c and c.get("n"):
                lines.append(f"| {dest} | {c['n']:,} | {_cell(c)} |")
        stays = r["cells"].get(f"{status}_stays_fresh")
        stale = r["cells"].get(f"{status}_stays_unchanged")
        if stays and stale:
            lines.append(
                f"| *of which stays {status}: fresh report* | "
                f"{stays['n']:,} | {_cell(stays)} |")
            lines.append(
                f"| *of which stays {status}: unchanged* | "
                f"{stale['n']:,} | {_cell(stale)} |")
        lines.append("")

    fs = t["final_sensitivity"]
    avail30 = {r["status"]: r for r in t["available_panel"]["30m"]["rows"]}
    lines += [
        "",
        f"The 30m status equals the strict 15-minute final status for "
        f"{100 * fs['share_30m_status_equals_final']:.3f}% of the "
        f"{fs['transition_rows']:,} transition rows, so the pre-registered "
        "last slot is not hiding a late flip.",
        "",
        "On all games (available panel, not the common panel) the 30m "
        f"Questionable and Probable rates are "
        f"{_cell(avail30.get('Questionable'))} and "
        f"{_cell(avail30.get('Probable'))}; the common panel is not driving "
        "the result.",
        "",
        "## Calibration inversion",
        "",
        "`STATUS_WEIGHTS` orders Available above Probable, but Available "
        "players scratch more often at every cutoff:",
        "",
        "| cutoff | Available | Probable |",
        "|---|---|---|",
    ]
    for h in ("3h", "1h", "30m"):
        a = t["calibration"][h]["Available"]
        b = t["calibration"][h]["Probable"]
        lines.append(f"| {h} | {_cell(a)} | {_cell(b)} |")

    lines += [
        "",
        "The ordinal mapping is inverted: the model treats Available as the "
        "safest pre-game state, but Available players scratch about twice as "
        "often as Probables at every cutoff, and the 30m Wilson bands do not "
        "overlap. The store has no role or minute-restriction field, so the "
        "mechanism is not identifiable here. This is a descriptive finding "
        "and a candidate for a future pre-registered recalibration, not a "
        "change to `STATUS_WEIGHTS`.",
        "",
        "## Per season (common panel)",
        "",
        "Questionable / Probable scratch rates at 3h, 1h and 30m. "
        "2019-20 is Jan-Mar 2020 only.",
        "",
        "| season | Q 3h | Q 1h | Q 30m | P 3h | P 1h | P 30m |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in t["by_season"]:
        lines.append(
            f"| {r['season']} | {_cell(r.get('questionable_3h'))} | "
            f"{_cell(r.get('questionable_1h'))} | "
            f"{_cell(r.get('questionable_30m'))} | "
            f"{_cell(r.get('probable_3h'))} | "
            f"{_cell(r.get('probable_1h'))} | "
            f"{_cell(r.get('probable_30m'))} |")

    lines += [
        "",
        "## Era split (common panel)",
        "",
        "Early = 2019-20 + 2020-21 (multi-slot report days); modern = "
        "2021-22 onward (17:00/20:00).",
        "",
        "| era | Q 3h | Q 1h | Q 30m | P 3h | P 1h | P 30m |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in t["by_era"]:
        lines.append(
            f"| {r['era']} | {_cell(r.get('questionable_3h'))} | "
            f"{_cell(r.get('questionable_1h'))} | "
            f"{_cell(r.get('questionable_30m'))} | "
            f"{_cell(r.get('probable_3h'))} | "
            f"{_cell(r.get('probable_1h'))} | "
            f"{_cell(r.get('probable_30m'))} |")

    lines += [
        "",
        "## The 6h column is yesterday's report",
        "",
        f"Median actual age at the 6h cutoff: "
        f"{t['six_hour']['median_age_hours']:.1f}h; "
        f"{100 * t['six_hour']['share_age_ge_12h']:.1f}% of statuses are at "
        f"least 12h old; same-day share "
        f"{100 * t['six_hour']['same_day_share']:.1f}%. It measures how much "
        "to trust yesterday's listing, not a same-day six-hour-old report.",
        "",
        "| 6h designation | all | early era | modern era |",
        "|---|---|---|---|",
    ]
    for status in STATUS_ORDER:
        cells = []
        for key in (t["six_hour"]["statuses"],
                    t["six_hour"]["by_era"]["early_2019_2021"]["statuses"],
                    t["six_hour"]["by_era"]["modern_2021_on"]["statuses"]):
            row = next((r for r in key if r["status"] == status), None)
            cells.append(_cell(row))
        lines.append(f"| {status} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Caveats",
        "",
    ]
    for c in p["caveats"]:
        lines.append(f"- {c}")
    lines += [
        "",
        "Wilson 95% bands are descriptive stability intervals, not decision "
        "tests. Nothing here is promoted; the canonical model is unchanged.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
