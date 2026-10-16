#!/usr/bin/env python3
"""
download_nba_data.py - pull everything needed to build NBA game-prediction models
(Elo, logistic regression, margin/total models, player-based ratings, GBM, stacking,
style features, in-game win probability) using nba_api.

Usage examples
--------------
  # Fast "core" data for 2015-16 .. 2025-26 (resumable; re-run any time)
  python download_nba_data.py

  # Core + season-level style features + league-wide shot data
  python download_nba_data.py --stages core,extras

  # Per-game inactives/officials and rotations (slow: ~1 call per game)
  python download_nba_data.py --stages summary,rotations --start 2022 --end 2025

  # Smoke-test a per-game stage on 20 games before committing to hours of downloading
  python download_nba_data.py --stages rotations --start 2024 --end 2024 --max-games 20

Stages (run in this order; groups in brackets)
----------------------------------------------
  meta        [core]   teams, approximate arena coordinates, all-players list
  schedule    [core]   league schedule per season (rest days, back-to-backs, travel)
  game_logs   [core]   LeagueGameLog, team + player, every game (Elo / logistic / margin models)
  adv_logs    [core]   per-game Advanced / Four Factors / Misc / Scoring (team)
                       and Advanced / Usage / Misc / Scoring (player) game logs
  snapshots   [core]   season-END team/player stats, estimated metrics, lineups, bio
                       (use ONLY as prior-season carryover, see leakage note below)
  style       [extras] tracking, synergy, shot locations, defense tracking, hustle (season-level)
  shots       [extras] league-wide shot chart detail (xFG / shot-quality model)
  summary     [heavy]  BoxScoreSummaryV2 per game: InactivePlayers, Officials, attendance, ...
  rotations   [heavy]  GameRotation per game: player stints with PT_DIFF (RAPM inputs)
  winprob              WinProbabilityPBP per game (benchmark for in-game model)
  pbp                  PlayByPlayV3 per game (very large)
                         NOTE: stats.nba.com's playbyplayv3 returns HTTP 500 with an
                         empty body on every game tried, so this stage falls back to
                         cdn.nba.com/static/json/liveData/playbyplay/, which serves
                         completed games too. That host rate-limits aggressively and
                         needs the backoff below.

  Groups: core, extras, heavy, all.   Default: core

LEAKAGE NOTE
------------
Everything in `snapshots/`, `style/` and `shots/` is a full-season aggregate pulled
today. Using it as a feature for games inside that same season leaks the future.
Use it only as a PRIOR for the FOLLOWING season, or rebuild rolling features
from the per-game tables in `game_logs/` and `*_gamelogs/`.
"""
from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests
from nba_api.stats import endpoints as ep
from nba_api.stats.static import teams as static_teams

log = logging.getLogger("nba")

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

# Approximate city-level arena coordinates (good enough for travel-distance features).
TEAM_COORDS = {
    "ATL": (33.757, -84.396), "BOS": (42.366, -71.062), "BKN": (40.683, -73.975),
    "CHA": (35.225, -80.839), "CHI": (41.881, -87.674), "CLE": (41.496, -81.688),
    "DAL": (32.790, -96.810), "DEN": (39.749, -105.008), "DET": (42.341, -83.055),
    "GSW": (37.768, -122.388), "HOU": (29.751, -95.362), "IND": (39.764, -86.156),
    "LAC": (34.043, -118.267), "LAL": (34.043, -118.267), "MEM": (35.138, -90.051),
    "MIA": (25.781, -80.187), "MIL": (43.045, -87.917), "MIN": (44.979, -93.276),
    "NOP": (29.949, -90.082), "NYK": (40.751, -73.994), "OKC": (35.463, -97.515),
    "ORL": (28.539, -81.384), "PHI": (39.901, -75.172), "PHX": (33.446, -112.071),
    "POR": (45.532, -122.667), "SAC": (38.580, -121.500), "SAS": (29.427, -98.437),
    "TOR": (43.643, -79.379), "UTA": (40.768, -111.901), "WAS": (38.898, -77.021),
}

# (measure type, per mode) combos for season-end snapshots
TEAM_SNAPSHOTS = [
    ("Base", "PerGame"), ("Base", "Per100Possessions"), ("Advanced", "PerGame"),
    ("Four Factors", "PerGame"), ("Misc", "PerGame"), ("Scoring", "PerGame"),
    ("Opponent", "PerGame"), ("Defense", "PerGame"),
]
PLAYER_SNAPSHOTS = [
    ("Base", "PerGame"), ("Base", "Per100Possessions"), ("Advanced", "PerGame"),
    ("Usage", "PerGame"), ("Defense", "PerGame"),
]

# Per-game logs (bulk, one call per season/type; Base is already in LeagueGameLog)
TEAM_LOG_MEASURES = ["Advanced", "Four Factors", "Misc", "Scoring"]
PLAYER_LOG_MEASURES = ["Advanced", "Usage", "Misc", "Scoring"]

PT_MEASURES = ["SpeedDistance", "Rebounding", "Possessions", "CatchShoot", "PullUpShot",
               "Defense", "Drives", "Passing", "ElbowTouch", "PostTouch", "PaintTouch",
               "Efficiency"]
PLAY_TYPES = ["Transition", "Isolation", "PRBallHandler", "PRRollman", "Postup", "Spotup",
              "Handoff", "Cut", "OffScreen", "OffRebound", "Misc"]
DEFENSE_CATEGORIES = ["Overall", "3 Pointers", "2 Pointers", "Less Than 6Ft",
                      "Less Than 10Ft", "Greater Than 15Ft"]

# First season start-year with data for each family (earlier seasons are skipped)
MIN_YEAR = {"tracking": 2013, "synergy": 2015, "hustle": 2015}

STAGE_ORDER = ["meta", "schedule", "game_logs", "adv_logs", "snapshots", "style",
               "shots", "summary", "rotations", "winprob", "pbp"]
GROUPS = {
    "core": ["meta", "schedule", "game_logs", "adv_logs", "snapshots"],
    "extras": ["style", "shots"],
    "heavy": ["summary", "rotations"],
    "all": STAGE_ORDER,
}

README = """NBA data download - folder guide
================================
meta/               teams (+ approx. arena lat/lon), all-players list
schedule/           ScheduleLeagueV2 per season (includes future games)
game_logs/          league_team_<season>.csv, league_player_<season>.csv  (LeagueGameLog)
team_gamelogs/      per-game team Advanced / Four Factors / Misc / Scoring
player_gamelogs/    per-game player Advanced / Usage / Misc / Scoring
snapshots/          SEASON-END aggregates (prior-season features only; leaks within a season)
style/, shots/      season-level style data and league-wide shot charts (same leakage rule)
summary/            per-game inactives, officials, line scores (appended CSVs)
rotations/          per-game player stints with PT_DIFF (RAPM inputs)
winprob/, pbp/      in-game data
_failures.log       every call that failed after retries; re-run the script to retry
Notes
-----
* Read CSVs with dtype={'GAME_ID': str} so leading zeros survive.
* *_RANK columns are dropped on purpose (they are ranks as of download time).
* Season labels look like '2024-25'. SEASON_TYPE is Regular Season / PlayIn / Playoffs.
* Per-game stage files are appended as they download; _done_<season>.txt tracks progress.
"""


@dataclass
class Ctx:
    out: Path
    sleep: float
    timeout: int
    retries: int
    proxy: str | None
    force: bool
    max_games: int | None
    max_consecutive: int


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def season_str(year: int) -> str:
    return f"{year}-{(year + 1) % 100:02d}"


def start_year(season: str) -> int:
    return int(season[:4])


def season_types(season: str) -> list[str]:
    types = ["Regular Season", "Playoffs"]
    if start_year(season) >= 2020:  # play-in tournament began in 2020-21
        types.insert(1, "PlayIn")
    return types


def by_type(season: str, param: str):
    """Variants that loop an endpoint over season types, tagging rows with SEASON_TYPE."""
    return [({param: st}, {"SEASON_TYPE": st}) for st in season_types(season)]


def slug(text: str) -> str:
    return text.lower().replace(" ", "_")


def fail(ctx: Ctx, label: str, err: Exception) -> None:
    msg = str(err).replace("\n", " ")[:300]
    log.error("FAILED %s: %s", label, msg)
    ctx.out.mkdir(parents=True, exist_ok=True)
    with (ctx.out / "_failures.log").open("a") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{label}\t{msg}\n")


def call(ctx: Ctx, cls, label: str, **kw):
    """Instantiate an nba_api endpoint with retries, backoff and polite throttling."""
    last = None
    for attempt in range(1, ctx.retries + 1):
        try:
            obj = cls(**kw, proxy=ctx.proxy, timeout=ctx.timeout)
            time.sleep(ctx.sleep + random.uniform(0, ctx.sleep / 2))
            return obj
        except Exception as e:  # network errors, JSON decode errors, HTTP 4xx/5xx
            last = e
            log.warning("%s failed (attempt %d/%d): %s", label, attempt, ctx.retries,
                        str(e)[:150])
            if attempt < ctx.retries:
                time.sleep(min(90, 3 * 2 ** attempt))
    raise RuntimeError(f"{label}: {last}")


# nba_api has two response shapes. Most endpoints return {"resultSets": [...]} and parse
# fine on their own. Sixteen V3-style endpoints (ScheduleLeagueV2, PlayByPlayV3, the
# BoxScore*V3 family, ...) instead return nested JSON with no "resultSets" key, and are
# only parsed correctly when the endpoint name is supplied to get_data_sets() so nba_api
# dispatches to its custom parser. Supplying the name for an endpoint WITHOUT a custom
# parser is equally fatal (KeyError on the registry), so the two cases must be kept apart.
try:  # private, but the only authoritative source; degrade gracefully if it moves
    from nba_api.stats.endpoints._parsers import _PARSER_REGISTRY as _CUSTOM_PARSER_ENDPOINTS
except ImportError:  # pragma: no cover
    _CUSTOM_PARSER_ENDPOINTS = frozenset()


def frames(obj) -> dict[str, pd.DataFrame]:
    """Map dataset name -> DataFrame, flattening multi-level headers (shot locations)."""
    name = getattr(obj, "endpoint", None)
    if name not in _CUSTOM_PARSER_ENDPOINTS:
        name = None  # plain tabular response; let nba_api use its default path
    data_sets = obj.nba_response.get_data_sets(name)
    out = {}
    # Names and frames come from the same ordered mapping: the endpoint builds its own
    # self.data_sets from this very call, so zipping them stays aligned.
    for (key, _ds), df in zip(data_sets.items(), obj.get_data_frames()):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(str(p) for p in c if str(p)) for c in df.columns]
        out[key] = df
    return out


def save(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info("saved %s (%d rows)", path, len(df))


def pull(ctx: Ctx, relpath: str, cls, label: str, base_kw: dict, *,
         variants=None, season: str | None = None, frame=0) -> pd.DataFrame | None:
    """
    Call `cls` once per variant, concatenate, tag, and save to ctx.out/relpath.
    Skips if the file exists (unless --force). If any call fails, nothing is written
    so a re-run retries the whole file rather than silently keeping a partial one.
    """
    path = ctx.out / relpath
    if path.exists() and not ctx.force:
        log.info("skip (exists) %s", relpath)
        return None
    parts, failed = [], False
    for extra_kw, tags in (variants or [({}, {})]):
        tag_txt = ",".join(str(v) for v in tags.values())
        lbl = f"{label} {season or ''} {tag_txt}".strip()
        try:
            fr = frames(call(ctx, cls, lbl, **{**base_kw, **extra_kw}))
            df = fr.get(frame) if isinstance(frame, str) else list(fr.values())[frame]
        except Exception as e:
            fail(ctx, lbl, e)
            failed = True
            continue
        if df is None or df.empty:
            log.info("empty: %s", lbl)
            continue
        df = df.loc[:, ~df.columns.astype(str).str.endswith("_RANK")].copy()
        if season is not None and "SEASON" not in df.columns:
            df.insert(0, "SEASON", season)
        for k, v in tags.items():
            if k not in df.columns:
                df[k] = v
        parts.append(df)
    if failed:
        log.warning("not saving %s because at least one call failed (re-run to retry)", relpath)
        return None
    if not parts:
        log.warning("no data for %s", relpath)
        return None
    out = pd.concat(parts, ignore_index=True)
    save(path, out)
    return out


def append_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #

def stage_meta(ctx: Ctx, seasons: list[str]) -> None:
    path = ctx.out / "meta" / "teams.csv"
    if not path.exists() or ctx.force:
        df = pd.DataFrame(static_teams.get_teams())
        df["LAT"] = df["abbreviation"].map(lambda a: TEAM_COORDS.get(a, (None, None))[0])
        df["LON"] = df["abbreviation"].map(lambda a: TEAM_COORDS.get(a, (None, None))[1])
        save(path, df)
    pull(ctx, "meta/players_all.csv", ep.CommonAllPlayers, "common_all_players",
         {"is_only_current_season": 0, "season": seasons[-1], "league_id": "00"})


def stage_schedule(ctx: Ctx, seasons: list[str]) -> None:
    for s in seasons:
        pull(ctx, f"schedule/schedule_{s}.csv", ep.ScheduleLeagueV2, "schedule",
             {"season": s}, season=s)


def stage_game_logs(ctx: Ctx, seasons: list[str]) -> None:
    for s in seasons:
        for code, name in (("T", "team"), ("P", "player")):
            pull(ctx, f"game_logs/league_{name}_{s}.csv", ep.LeagueGameLog, f"gamelog_{name}",
                 {"season": s, "player_or_team_abbreviation": code},
                 variants=by_type(s, "season_type_all_star"), season=s)


def stage_adv_logs(ctx: Ctx, seasons: list[str]) -> None:
    for s in seasons:
        for m in TEAM_LOG_MEASURES:
            pull(ctx, f"team_gamelogs/{slug(m)}_{s}.csv", ep.TeamGameLogs, f"team_log_{m}",
                 {"season_nullable": s, "measure_type_player_game_logs_nullable": m},
                 variants=by_type(s, "season_type_nullable"), season=s)
        for m in PLAYER_LOG_MEASURES:
            pull(ctx, f"player_gamelogs/{slug(m)}_{s}.csv", ep.PlayerGameLogs, f"player_log_{m}",
                 {"season_nullable": s, "measure_type_player_game_logs_nullable": m},
                 variants=by_type(s, "season_type_nullable"), season=s)


def stage_snapshots(ctx: Ctx, seasons: list[str]) -> None:
    for s in seasons:
        for m, mode in TEAM_SNAPSHOTS:
            pull(ctx, f"snapshots/team_{slug(m)}_{slug(mode)}_{s}.csv", ep.LeagueDashTeamStats,
                 f"team_snap_{m}_{mode}",
                 {"season": s, "measure_type_detailed_defense": m, "per_mode_detailed": mode},
                 season=s)
        for m, mode in PLAYER_SNAPSHOTS:
            pull(ctx, f"snapshots/player_{slug(m)}_{slug(mode)}_{s}.csv",
                 ep.LeagueDashPlayerStats, f"player_snap_{m}_{mode}",
                 {"season": s, "measure_type_detailed_defense": m, "per_mode_detailed": mode},
                 season=s)
        pull(ctx, f"snapshots/player_estimated_metrics_{s}.csv", ep.PlayerEstimatedMetrics,
             "player_est_metrics", {"season": s}, season=s)
        pull(ctx, f"snapshots/team_estimated_metrics_{s}.csv", ep.TeamEstimatedMetrics,
             "team_est_metrics", {"season": s}, season=s)
        pull(ctx, f"snapshots/lineups5_advanced_{s}.csv", ep.LeagueDashLineups, "lineups5",
             {"season": s, "group_quantity": 5, "measure_type_detailed_defense": "Advanced",
              "per_mode_detailed": "Totals"}, season=s)
        pull(ctx, f"snapshots/player_bio_{s}.csv", ep.LeagueDashPlayerBioStats, "player_bio",
             {"season": s}, season=s)


def stage_style(ctx: Ctx, seasons: list[str]) -> None:
    for s in seasons:
        y = start_year(s)
        if y >= MIN_YEAR["tracking"]:
            for ent in ("Team", "Player"):
                for pt in PT_MEASURES:
                    pull(ctx, f"style/ptstats_{ent.lower()}_{slug(pt)}_{s}.csv",
                         ep.LeagueDashPtStats, f"ptstats_{ent}_{pt}",
                         {"season": s, "pt_measure_type": pt, "player_or_team": ent,
                          "per_mode_simple": "PerGame"}, season=s)
            variants = [({"defense_category": c}, {"DEFENSE_CATEGORY": c})
                        for c in DEFENSE_CATEGORIES]
            pull(ctx, f"style/pt_team_defend_{s}.csv", ep.LeagueDashPtTeamDefend,
                 "pt_team_defend", {"season": s, "per_mode_simple": "PerGame"},
                 variants=variants, season=s)
            pull(ctx, f"style/pt_player_defend_{s}.csv", ep.LeagueDashPtDefend,
                 "pt_player_defend", {"season": s, "per_mode_simple": "PerGame"},
                 variants=variants, season=s)
        # shot locations (multi-level headers get flattened in frames())
        for m in ("Base", "Opponent"):
            pull(ctx, f"style/team_shot_locations_{slug(m)}_{s}.csv",
                 ep.LeagueDashTeamShotLocations, f"team_shotloc_{m}",
                 {"season": s, "distance_range": "By Zone", "measure_type_simple": m,
                  "per_mode_detailed": "PerGame"}, season=s)
        pull(ctx, f"style/player_shot_locations_{s}.csv", ep.LeagueDashPlayerShotLocations,
             "player_shotloc", {"season": s, "distance_range": "By Zone",
                                "per_mode_detailed": "PerGame"}, season=s)
        if y >= MIN_YEAR["synergy"]:
            for code, ent in (("T", "team"), ("P", "player")):
                variants = [({"play_type_nullable": pt, "type_grouping_nullable": g},
                             {"PLAY_TYPE": pt, "TYPE_GROUPING": g})
                            for g in ("offensive", "defensive") for pt in PLAY_TYPES]
                pull(ctx, f"style/synergy_{ent}_{s}.csv", ep.SynergyPlayTypes, f"synergy_{ent}",
                     {"season": s, "player_or_team_abbreviation": code,
                      "per_mode_simple": "Totals"}, variants=variants, season=s)
        if y >= MIN_YEAR["hustle"]:
            pull(ctx, f"style/hustle_team_{s}.csv", ep.LeagueHustleStatsTeam, "hustle_team",
                 {"season": s, "per_mode_time": "PerGame"}, season=s)
            pull(ctx, f"style/hustle_player_{s}.csv", ep.LeagueHustleStatsPlayer, "hustle_player",
                 {"season": s, "per_mode_time": "PerGame"}, season=s)


def stage_shots(ctx: Ctx, seasons: list[str]) -> None:
    """League-wide shot chart (player_id=0, team_id=0). Big responses: raise --timeout if needed."""
    for s in seasons:
        pull(ctx, f"shots/shotchart_{s}.csv", ep.ShotChartDetail, "shotchart",
             {"team_id": 0, "player_id": 0, "season_nullable": s,
              "context_measure_simple": "FGA"},
             variants=by_type(s, "season_type_all_star"), season=s, frame=0)


# ---- per-game stages ------------------------------------------------------- #

def game_ids(ctx: Ctx, season: str) -> list[str]:
    p = ctx.out / "game_logs" / f"league_team_{season}.csv"
    if not p.exists():
        stage_game_logs(ctx, [season])
    if not p.exists():
        log.error("no game log for %s; cannot enumerate games", season)
        return []
    return sorted(pd.read_csv(p, usecols=["GAME_ID"], dtype=str)["GAME_ID"].unique())


def per_game(ctx: Ctx, stage: str, seasons: list[str], fetch) -> None:
    """Resumable one-call-per-game downloader. `fetch(ctx, game_id)` -> {name: DataFrame}."""
    d = ctx.out / stage
    d.mkdir(parents=True, exist_ok=True)
    for s in seasons:
        done_path = d / f"_done_{s}.txt"
        if ctx.force:
            for f in d.glob(f"*_{s}.csv"):
                f.unlink()
            done_path.unlink(missing_ok=True)
        done = set(done_path.read_text().split()) if done_path.exists() else set()
        todo = [g for g in game_ids(ctx, s) if g not in done]
        if ctx.max_games:
            todo = todo[: ctx.max_games]
        log.info("[%s %s] %d games to download (%d already done)", stage, s, len(todo), len(done))
        consecutive = 0
        for i, gid in enumerate(todo, 1):
            try:
                result = fetch(ctx, gid)
                consecutive = 0
            except Exception as e:
                fail(ctx, f"{stage} {s} {gid}", e)
                consecutive += 1
                if consecutive >= ctx.max_consecutive:
                    log.error("[%s] %d consecutive failures - endpoint looks blocked or "
                              "unavailable. Stopping this stage.", stage, consecutive)
                    return
                continue
            for name, df in result.items():
                if df is not None and not df.empty:
                    append_csv(d / f"{name}_{s}.csv", df)
            with done_path.open("a") as fh:
                fh.write(gid + "\n")
            if i % 50 == 0 or i == len(todo):
                log.info("[%s %s] %d/%d games", stage, s, i, len(todo))


def fetch_summary(ctx: Ctx, gid: str) -> dict:
    fr = frames(call(ctx, ep.BoxScoreSummaryV2, f"summary {gid}", game_id=gid))
    keep = ["GameInfo", "GameSummary", "LineScore", "OtherStats", "Officials", "InactivePlayers"]
    return {k: fr[k] for k in keep if k in fr}


def fetch_rotation(ctx: Ctx, gid: str) -> dict:
    fr = frames(call(ctx, ep.GameRotation, f"rotation {gid}", game_id=gid))
    parts = []
    for side, key in (("AWAY", "AwayTeam"), ("HOME", "HomeTeam")):
        df = fr.get(key)
        if df is not None and not df.empty:
            df = df.copy()
            df["TEAM_SIDE"] = side
            parts.append(df)
    return {"rotation": pd.concat(parts, ignore_index=True)} if parts else {}


def fetch_winprob(ctx: Ctx, gid: str) -> dict:
    fr = frames(call(ctx, ep.WinProbabilityPBP, f"winprob {gid}", game_id=gid))
    return {"winprob": next(iter(fr.values()))}


CDN_LIVE_BASE = "https://cdn.nba.com/static/json/liveData"
CDN_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _flatten_cdn_playbyplay(payload: dict) -> pd.DataFrame:
    """Turn a cdn.nba.com play-by-play document into one row per action.

    The live document nests everything under game.actions with period, clock,
    running score and a free-text description. The score is what makes it usable
    for an in-game model: each row carries the game state immediately after the
    action, so win probability can be scored against state rather than against a
    final result.
    """
    actions = ((payload.get("game") or {}).get("actions")) or []
    rows = []
    for a in actions:
        rows.append({
            "GAME_ID": a.get("gameId"),
            "actionNumber": a.get("actionNumber"),
            "period": a.get("period"),
            "clock": a.get("clock"),
            "teamId": a.get("teamId"),
            "teamTricode": a.get("teamTricode"),
            "playerId": a.get("personId"),
            "playerName": a.get("playerName"),
            "actionType": a.get("actionType"),
            "subType": a.get("subType"),
            "description": a.get("description"),
            "shotValue": a.get("shotValue"),
            "isHomeTeam": a.get("isHomeTeam"),
            "scoreHome": a.get("scoreHome"),
            "scoreAway": a.get("scoreAway"),
            "shotClock": a.get("shotClock"),
            "x": a.get("x"),
            "y": a.get("y"),
        })
    return pd.DataFrame(rows)


def fetch_pbp(ctx: Ctx, gid: str) -> dict:
    """Play-by-play for one game.

    stats.nba.com's PlayByPlayV3 currently answers HTTP 500 with an empty body on
    every game id tried (0 of 12 across two seasons), so the per-game route through
    that host is not currently usable. cdn.nba.com serves the same content under
    liveData and does work for completed games, so it is tried second rather than
    first: the canonical endpoint is preferred when it recovers, and the CDN route
    is the fallback that keeps the stage from producing nothing.
    """
    try:
        fr = frames(call(ctx, ep.PlayByPlayV3, f"pbp {gid}", game_id=gid))
        return {"playbyplay": next(iter(fr.values()))}
    except Exception as exc:
        log.warning("PlayByPlayV3 unavailable for %s (%s); trying cdn.nba.com",
                    gid, str(exc)[:120])
    url = f"{CDN_LIVE_BASE}/playbyplay/playbyplay_{gid}.json"
    last = None
    for attempt in range(ctx.retries):
        try:
            resp = requests.get(url, headers=CDN_HEADERS, timeout=ctx.timeout,
                                proxies=({"http": ctx.proxy, "https": ctx.proxy}
                                         if ctx.proxy else None))
            if resp.status_code == 200 and resp.text.strip().startswith("{"):
                df = _flatten_cdn_playbyplay(resp.json())
                if not df.empty:
                    return {"playbyplay": df}
            last = RuntimeError(
                f"cdn playbyplay {resp.status_code} ({len(resp.text)} bytes)")
        except Exception as exc:  # network error, timeout, JSON decode
            last = exc
        time.sleep(min(30, 3 * 2 ** (attempt + 1)))
    raise RuntimeError(f"playbyplay {gid}: {last}")


def stage_summary(ctx, seasons):
    per_game(ctx, "summary", seasons, fetch_summary)


def stage_rotations(ctx, seasons):
    per_game(ctx, "rotations", seasons, fetch_rotation)


def stage_winprob(ctx, seasons):
    per_game(ctx, "winprob", seasons, fetch_winprob)


def stage_pbp(ctx, seasons):
    per_game(ctx, "pbp", seasons, fetch_pbp)


STAGE_FUNCS = {
    "meta": stage_meta, "schedule": stage_schedule, "game_logs": stage_game_logs,
    "adv_logs": stage_adv_logs, "snapshots": stage_snapshots, "style": stage_style,
    "shots": stage_shots, "summary": stage_summary, "rotations": stage_rotations,
    "winprob": stage_winprob, "pbp": stage_pbp,
}


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_stages(text: str) -> list[str]:
    wanted: set[str] = set()
    for tok in text.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok in GROUPS:
            wanted.update(GROUPS[tok])
        elif tok in STAGE_FUNCS:
            wanted.add(tok)
        else:
            sys.exit(f"Unknown stage/group '{tok}'. Stages: {', '.join(STAGE_ORDER)}; "
                     f"groups: {', '.join(GROUPS)}")
    return [s for s in STAGE_ORDER if s in wanted]


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/nba", help="output directory (default data/nba)")
    ap.add_argument("--start", type=int, default=2015,
                    help="first season by START year, e.g. 2015 = 2015-16 (default 2015)")
    ap.add_argument("--end", type=int, default=2025,
                    help="last season by START year, e.g. 2025 = 2025-26 (default 2025). "
                         "Use 2026 to include the 2026-27 season")
    ap.add_argument("--stages", default="core", help="comma-separated stages/groups (default core)")
    ap.add_argument("--sleep", type=float, default=0.8, help="seconds between calls (default 0.8)")
    ap.add_argument("--timeout", type=int, default=60, help="per-request timeout seconds")
    ap.add_argument("--retries", type=int, default=3, help="attempts per call")
    ap.add_argument("--proxy", default=None, help="optional proxy, e.g. http://user:pass@host:port "
                    "(stats.nba.com often blocks cloud/VPN IPs)")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    ap.add_argument("--max-games", type=int, default=None,
                    help="limit games per season for per-game stages (for testing)")
    ap.add_argument("--max-consecutive-failures", type=int, default=15,
                    help="abort a per-game stage after this many failures in a row")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(out / "download.log")])

    if args.end < args.start:
        sys.exit("--end must be >= --start")
    seasons = [season_str(y) for y in range(args.start, args.end + 1)]
    stages = parse_stages(args.stages)
    ctx = Ctx(out, args.sleep, args.timeout, args.retries, args.proxy, args.force,
              args.max_games, args.max_consecutive_failures)

    readme = out / "README.txt"
    if not readme.exists():
        readme.write_text(README)

    log.info("Seasons: %s .. %s | stages: %s | out: %s", seasons[0], seasons[-1],
             ", ".join(stages), out.resolve())
    t0 = time.time()
    try:
        for stage in stages:
            log.info("===== stage: %s =====", stage)
            STAGE_FUNCS[stage](ctx, seasons)
    except KeyboardInterrupt:
        log.warning("Interrupted. Progress is saved; re-run the same command to resume.")
        sys.exit(130)

    files = [p for p in out.rglob("*.csv")]
    size_mb = sum(p.stat().st_size for p in files) / 1e6
    log.info("Done in %.1f min. %d CSV files, %.0f MB in %s", (time.time() - t0) / 60,
             len(files), size_mb, out.resolve())
    if (out / "_failures.log").exists():
        log.warning("Some calls failed - see %s. Re-run to retry only what's missing.",
                    out / "_failures.log")


if __name__ == "__main__":
    main()
