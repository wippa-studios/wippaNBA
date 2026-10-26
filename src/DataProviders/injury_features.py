"""Official pre-tip injury reports as a point-in-time availability feature (#37).

WHAT THIS IS
------------
``Process-Data/Get_Injury_Data.py`` captures and parses the NBA's own injury
report PDFs.  This module turns the stored ``injury_status_events`` into a
same-game, pre-tip discount on each player's expected minutes, in the exact
shape ``nba_data.availability_features`` consumes (``GAME_ID``, ``PLAYER_ID``,
``weight``).

The discount is applied to the squad that played the team's PREVIOUS game -
the canonical availability construction - so a designation can only remove
minutes the model already expected.  It cannot add minutes for a returning
player who was not in the prior squad, and it never reads who actually played.

POINT-IN-TIME RULE
------------------
For each game, the usable snapshots are the ones whose printed label is at
least ``buffer_minutes`` before tip.  The latest such snapshot wins.  The label
is optimistic by the PDF upload lag (see ``status_known_as_of``), which is why
the buffer exists; 15 minutes is the default and a sensitivity arm re-runs the
measurement at 30.

The report lists every game on the date, so "this game has no rows" is a real
answer - nobody was designated - not missing coverage.  Coverage is decided by
whether a usable snapshot exists, not by whether a player appears.

PLAYER IDENTITY
---------------
The parser's ``team`` cell is the trustworthy field; its ``matchup`` cell can
be mispaired when a team block straddles a page or a wrapped row (3.27% of the
stored 2021 rows name a team that is not in their matchup).  Identity is
therefore resolved from the player's NAME against the season roster, preferring
a player who was on the report's team, and the player's team - not the parsed
matchup - decides which game the designation belongs to.
"""
from __future__ import annotations

import re
import sqlite3
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from . import nba_data as nd

__all__ = [
    "DEFAULT_BUFFER_MINUTES",
    "STATUS_WEIGHTS",
    "availability_weights",
    "diagnostics",
    "load_events",
    "resolve_players",
    "select_pre_tip",
]

#: Ordinal discount for each official designation.  ``Out`` removes the
#: expected minutes; the doubtful/questionable/probable ladder is the league's
#: own probability language, mapped linearly.  ``Available`` is a designation
#: (usually a returning player) and keeps the full expectation.
STATUS_WEIGHTS: dict[str, float] = {
    "Out": 0.0,
    "Doubtful": 0.25,
    "Questionable": 0.5,
    "Probable": 0.75,
    "Available": 1.0,
}

DEFAULT_BUFFER_MINUTES = 15.0

_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})
_TIME_RE = re.compile(r"(\d{1,2}):(\d{2})\s*(AM|PM)?", re.IGNORECASE)


def _flat(text) -> str:
    """ASCII, lowercase, punctuation-free, single-spaced."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return ""
    s = unicodedata.normalize("NFKD", str(text))
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace("'", "").replace(".", "").replace(",", " ")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def name_key(name) -> str:
    """Order-insensitive full-name key: ``Carter-Williams, Michael`` ->
    ``carter michael williams``.  Suffixes (Jr/Sr/II...) are dropped.

    Sorting the tokens is what makes the report's ``Last, First`` and the
    roster's ``First Last`` compare equal without a fragile last-name split:
    ``Carter-Williams, Michael``, ``Michael Carter-Williams`` and
    ``Brown Jr., Chaundee`` / ``Chaundee Brown Jr.`` all normalise cleanly.
    """
    toks = [t for t in _flat(name).split() if t not in _SUFFIXES]
    return " ".join(sorted(toks))


def _nickname_key(nickname, player_name) -> str:
    """``Svi`` + ``Sviatoslav Mykhailiuk`` -> ``mykhailiuk svi``.

    The league's reports often print the nickname the player goes by, while
    the roster carries the legal first name.  The nickname column is exactly
    the bridge.  Multi-word surnames lose their first word here, but the
    full-name key still covers those players.
    """
    nick = _flat(nickname)
    full = _flat(player_name).split()
    if not nick or not full:
        return ""
    surname = full[-1] if full[-1] not in _SUFFIXES else (full[-2] if len(full) > 1 else "")
    if not surname:
        return ""
    return " ".join(sorted(f"{nick} {surname}".split()))


def parse_tip_minutes(game_time) -> float | None:
    """``07:30 (ET)`` -> 1170.0 minutes after midnight.  ``None`` for TBD.

    The report prints the Eastern clock WITHOUT an AM/PM suffix and every NBA
    game is an afternoon or evening game, so hours 1-11 are PM and ``12`` is
    noon.  Reading ``07:30`` as 07:30 (as a naive 24-hour parse does) puts
    every evening tip seven hours before the afternoon report and silently
    makes the point-in-time gate reject every snapshot.
    """
    if game_time is None or (isinstance(game_time, float) and np.isnan(game_time)):
        return None
    m = _TIME_RE.search(str(game_time))
    if not m:
        return None
    hour, minute, suffix = int(m.group(1)), int(m.group(2)), m.group(3)
    if suffix:
        suffix = suffix.upper()
        if not 1 <= hour <= 12:
            return None
        hour = hour % 12 + (12 if suffix == "PM" else 0)
    else:
        if not 1 <= hour <= 12:
            return None
        if hour < 12:
            hour += 12
    if hour > 23 or minute > 59:
        return None
    return float(hour * 60 + minute)


def load_events(db_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the two evidence tables READ-ONLY.

    Returns ``(events, snapshots)``.  A missing store is a hard error here: a
    silent empty frame would make the feature look like "nobody was injured",
    which is the opposite of a null result.
    """
    p = Path(db_path)
    if not p.exists() or p.stat().st_size == 0:
        raise FileNotFoundError(
            f"injury store {p} is missing or empty; run "
            "`python -m src.Process-Data.Get_Injury_Data capture ...` first")
    con = sqlite3.connect(f"file:{p}?mode=ro", uri=True)
    try:
        events = pd.read_sql_query(
            "SELECT id, snapshot_id, line_no, snapshot_ts_et, snapshot_ts_utc,"
            " game_date, game_time, matchup, team, player, status, reason"
            " FROM injury_status_events", con)
        snaps = pd.read_sql_query(
            "SELECT id, report_date, report_time_label, snapshot_ts_et,"
            " snapshot_ts_utc, raw_path, parsed_rows"
            " FROM injury_snapshots", con)
    except sqlite3.Error as exc:
        raise RuntimeError(f"injury store {p} is not readable: {exc}") from exc
    finally:
        con.close()
    if events.empty:
        raise ValueError(
            f"injury store {p} has no parsed events; run the parse command")
    events["game_date"] = pd.to_datetime(events["game_date"]).dt.normalize()
    snaps["report_date"] = pd.to_datetime(snaps["report_date"]).dt.normalize()
    snaps["label_ts"] = pd.to_datetime(snaps["snapshot_ts_et"])
    events["label_ts"] = pd.to_datetime(events["snapshot_ts_et"])
    return events, snaps


def _team_abbr_map(pa: pd.DataFrame) -> dict[str, str]:
    """Team full name -> abbreviation, tolerant of the usual aliases."""
    m = (pa.drop_duplicates("TEAM_NAME")
         .set_index("TEAM_NAME")["TEAM_ABBREVIATION"].to_dict())
    for full, abbr in list(m.items()):
        m.setdefault(_flat(full), abbr)
    aliases = {
        "la clippers": "LAC", "los angeles clippers": "LAC",
        "brooklyn nets": "BKN", "new jersey nets": "BKN",
        "new orleans pelicans": "NOP", "new orleans hornets": "NOP",
        "charlotte bobcats": "CHA", "charlotte hornets": "CHA",
        "utah jazz": "UTA", "oklahoma city thunder": "OKC",
        "golden state warriors": "GSW", "washington wizards": "WAS",
    }
    m.update(aliases)
    return m


def resolve_players(events: pd.DataFrame, pa: pd.DataFrame | None = None,
                    seasons: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Attach ``PLAYER_ID`` and the team the player actually played for.

    Matching order per event:

    1. exact normalised full name among players on the report's TEAM, in the
       event's season or the season either side (trades, late call-ups);
    2. exact full name among all players active within that window, when the
       match is unique;
    3. exact nickname+surname key, same team-preference and uniqueness rules.

    Anything still ambiguous stays unresolved and is counted by
    :func:`diagnostics`; it is never guessed.  A player who is not on a
    prior-game squad cannot move the feature anyway, so unresolved rows are
    reported by "would it have mattered", not as a silent loss.
    """
    if pa is None:
        pa = nd.load_player_advanced()
    ev = events.copy()
    if seasons is not None:
        pa = pa[pa["SEASON"].astype(str).isin(seasons)]
    pa = pa[["PLAYER_ID", "PLAYER_NAME", "NICKNAME", "SEASON",
             "TEAM_ABBREVIATION", "TEAM_NAME", "GAME_DATE"]].copy()
    pa["GAME_DATE"] = pd.to_datetime(pa["GAME_DATE"]).dt.normalize()

    team_map = _team_abbr_map(pa)
    ev["team_abbr"] = ev["team"].map(team_map)
    if ev["team_abbr"].isna().any():
        # Second chance: the nickname part of a team name ("Trail Blazers").
        tail = ev["team"].map(lambda s: _flat(s).split()[-1] if _flat(s) else "")
        known = {_flat(k).split()[-1]: v for k, v in team_map.items() if _flat(k)}
        ev["team_abbr"] = ev["team_abbr"].fillna(tail.map(known))

    # One row per player-season-team.  The full-name and nickname keys are
    # built once so resolution is a dict lookup, not a per-event scan.
    pa = pa.drop_duplicates(["PLAYER_ID", "SEASON", "TEAM_ABBREVIATION"])
    pa["key_full"] = pa["PLAYER_NAME"].map(name_key)
    pa["key_nick"] = [
        _nickname_key(n, p) for n, p in zip(pa["NICKNAME"], pa["PLAYER_NAME"])
    ]
    pa["season_start"] = pa["SEASON"].str[:4].astype(int)

    by_full: dict[str, list[tuple]] = {}
    by_nick: dict[str, list[tuple]] = {}
    for row in pa.itertuples(index=False):
        rec = (row.PLAYER_ID, row.SEASON, row.TEAM_ABBREVIATION,
               row.season_start, row.GAME_DATE)
        by_full.setdefault(row.key_full, []).append(rec)
        if row.key_nick:
            by_nick.setdefault(row.key_nick, []).append(rec)

    # An NBA season spans two calendar years: games in Oct-Dec belong to
    # YYYY-(YYYY+1); games in Jan-Sep belong to (YYYY-1)-YYYY.
    ev["season"] = (
        ev["game_date"].dt.year.astype(str) + "-" +
        (ev["game_date"].dt.year + 1).astype(str).str[2:])
    jan_sep = ev["game_date"].dt.month < 10
    ev.loc[jan_sep, "season"] = (
        (ev["game_date"].dt.year - 1).astype(str) + "-" +
        ev["game_date"].dt.year.astype(str).str[2:])
    ev["season_start"] = ev["season"].str[:4].astype(int)

    key_full = ev["player"].map(name_key)
    key_nick = ev["player"].map(lambda s: _nickname_key(*_split_report_name(s)))
    ev["key_full"] = key_full
    ev["key_nick"] = key_nick

    out_id: list = []
    out_team: list = []
    out_method: list = []
    out_ambig: list = []
    for i, row in enumerate(ev.itertuples(index=False)):
        patch = _resolve_one(
            row.team_abbr, row.game_date, row.season_start,
            row.key_full, row.key_nick, by_full, by_nick)
        out_id.append(patch[0])
        out_team.append(patch[1])
        out_method.append(patch[2])
        out_ambig.append(patch[3])
    ev["PLAYER_ID"] = out_id
    ev["player_team"] = out_team
    ev["match_method"] = out_method
    ev["ambiguous"] = out_ambig
    ev["PLAYER_ID"] = ev["PLAYER_ID"].astype("Int64")
    # The report's team is the trusted field, but when it is not one of the
    # player's teams near the date the roster is.  A mispaired matchup leaves
    # the team cell correct, so this only fires on genuinely odd rows.
    ev["team_abbr"] = ev["team_abbr"].fillna(ev["player_team"])
    need_roster = ev["team_abbr"].isna() & ev["player_team"].notna()
    ev.loc[need_roster, "team_abbr"] = ev.loc[need_roster, "player_team"]
    ev["weight"] = ev["status"].map(STATUS_WEIGHTS)
    return ev


def _split_report_name(name) -> tuple[str, str]:
    """Split the report's ``Last, First`` for the nickname key.

    ``name_key`` deliberately discards order; the nickname key needs the
    surname, which in the report is the part before the comma.
    """
    text = str(name)
    if "," in text:
        last, first = text.split(",", 1)
        return first.strip(), last.strip()
    toks = _flat(text).split()
    if len(toks) < 2:
        return "", text
    return " ".join(toks[:-1]), toks[-1]


def _resolve_one(team_abbr, game_date, season_start, key_full, key_nick,
                 by_full, by_nick):
    if not key_full:
        return pd.NA, None, "unresolved", False
    window = {season_start - 1, season_start, season_start + 1}

    def pick(cands, prefer_team):
        cands = [c for c in cands if c[3] in window]
        if not cands:
            return None, None, False
        same_team = [c for c in cands if prefer_team and c[2] == prefer_team]
        pool = same_team or cands
        ids = {c[0] for c in pool}
        if len(ids) > 1:
            # Same name, several players: pick the one whose season row is
            # closest to the game date, but only if it is unambiguous.
            def distance(c):
                return abs((c[4] - game_date).days)
            pool = sorted(pool, key=distance)
            if pool[1][0] != pool[0][0] and distance(pool[1]) == distance(pool[0]):
                return None, None, True
        c = pool[0]
        # The player's team near the game: prefer a record within 120 days.
        near = [x for x in pool if x[0] == c[0] and abs((x[4] - game_date).days) <= 120]
        team = (near or [x for x in pool if x[0] == c[0]])[0][2]
        return c[0], team, False

    for method, table, key in (("exact", by_full, key_full),
                               ("nickname", by_nick, key_nick)):
        cands = table.get(key)
        if not cands:
            continue
        pid, team, ambig = pick(cands, team_abbr)
        if pid is not None:
            return pid, team, method, ambig
        if ambig:
            return pd.NA, None, "ambiguous", True
    return pd.NA, None, "unresolved", False


def _tip_table(events: pd.DataFrame) -> pd.DataFrame:
    """``(game_date, matchup) -> tip minutes`` from the report's own rows.

    A row prints the game time once per block and later rows inherit it, so
    the mode is the value for the game.  The matchup can be mispaired on a
    minority of rows; the mode of a real matchup is therefore still its own
    time as long as the real rows outnumber the stray ones.
    """
    ok = events.dropna(subset=["matchup"]).copy()
    ok["tip_minutes"] = ok["game_time"].map(parse_tip_minutes)
    ok = ok.dropna(subset=["tip_minutes"])
    if ok.empty:
        return pd.DataFrame(columns=["game_date", "matchup", "tip_minutes"])
    return (ok.groupby(["game_date", "matchup"], as_index=False)["tip_minutes"]
            .agg(lambda s: float(s.mode().iloc[0])))


def select_pre_tip(events: pd.DataFrame, snapshots: pd.DataFrame,
                   games: pd.DataFrame,
                   buffer_minutes: float = DEFAULT_BUFFER_MINUTES
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One designation per ``(GAME_ID, PLAYER_ID)``, known before that tip.

    Returns ``(designations, coverage)``.  ``coverage`` has one row per game
    with ``tip_minutes``, ``snapshot_label`` and ``has_snapshot``; it is what
    makes "no designation" distinguishable from "no report".
    """
    g = games.copy()
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"]).dt.normalize()
    g["game_date"] = g["GAME_DATE"]
    g["matchup"] = g["away"] + "@" + g["home"]
    tips = _tip_table(events)
    cov = g[["GAME_ID", "game_date", "matchup", "home", "away"]].merge(
        tips, on=["game_date", "matchup"], how="left")

    snaps = snapshots[["report_date", "label_ts", "report_time_label"]].copy()
    snaps["label_minutes"] = (snaps["label_ts"].dt.hour * 60
                              + snaps["label_ts"].dt.minute)
    usable = []
    for i, row in enumerate(cov.itertuples(index=False)):
        if pd.isna(row.tip_minutes):
            usable.append((i, pd.NaT, None))
            continue
        cutoff = row.tip_minutes - buffer_minutes
        cand = snaps[(snaps["report_date"] == row.game_date)
                     & (snaps["label_minutes"] <= cutoff)]
        if cand.empty:
            usable.append((i, pd.NaT, None))
            continue
        best = cand.sort_values("label_ts").iloc[-1]
        usable.append((i, best["label_ts"], best["report_time_label"]))
    cov["snapshot_label"] = [u[1] for u in usable]
    cov["snapshot_slot"] = [u[2] for u in usable]
    cov["has_snapshot"] = cov["snapshot_label"].notna()

    ev = events.dropna(subset=["PLAYER_ID", "weight"]).copy()
    ev["PLAYER_ID"] = ev["PLAYER_ID"].astype("int64")
    ev = ev.merge(cov[["GAME_ID", "game_date", "home", "away",
                       "snapshot_label"]],
                  on="game_date", how="inner")
    # Keep only games where the player's team is playing, and only the
    # snapshot chosen for that game.
    playing = (ev["team_abbr"] == ev["home"]) | (ev["team_abbr"] == ev["away"])
    ev = ev[playing & ev["snapshot_label"].notna()].copy()
    ev = ev[ev["label_ts"] <= ev["snapshot_label"]]
    if ev.empty:
        return pd.DataFrame(columns=["GAME_ID", "PLAYER_ID", "weight",
                                     "status", "snapshot_label"]), cov
    # Latest label wins; at one label the most severe designation wins and
    # the highest row id breaks the remaining tie deterministically.
    ev["severity"] = -ev["weight"]
    ev = ev.sort_values(["GAME_ID", "PLAYER_ID", "label_ts", "severity", "id"])
    ev = ev.groupby(["GAME_ID", "PLAYER_ID"], as_index=False).tail(1)
    ev = ev.rename(columns={"label_ts": "status_label_ts"})
    return ev[["GAME_ID", "PLAYER_ID", "weight", "status", "team_abbr",
               "status_label_ts", "match_method", "ambiguous", "player"]], cov


def earliest_pre_tip_designation(events: pd.DataFrame, snapshots: pd.DataFrame,
                                 games: pd.DataFrame,
                                 buffer_minutes: float = DEFAULT_BUFFER_MINUTES
                                 ) -> pd.DataFrame:
    """The EARLIEST same-day pre-tip designation per ``(GAME_ID, PLAYER_ID)``.

    Issue #69's trajectory arm needs the status at the FIRST usable slot of the
    day, while ``select_pre_tip`` (the authoritative final-status picker) takes
    the LAST.  This helper mirrors the same slot walk: a player's EARLY label
    is the earliest same-day snapshot that is still at least ``buffer_minutes``
    before tip, and the designation is the one in that snapshot (most severe at
    a label, highest ``id`` on a tie).  Games with no early label (single-slot
    days) are simply absent from the result -- the trajectory aggregates in the
    arm are computed over players with BOTH labels only.

    Returns columns ``GAME_ID, PLAYER_ID, weight_early, status_early, team_abbr``.
    """
    g = games.copy()
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"]).dt.normalize()
    g["game_date"] = g["GAME_DATE"]
    g["matchup"] = g["away"] + "@" + g["home"]
    tips = _tip_table(events)
    cov = g[["GAME_ID", "game_date", "matchup", "home", "away"]].merge(
        tips, on=["game_date", "matchup"], how="left")
    snaps = snapshots[["report_date", "label_ts", "report_time_label"]].copy()
    snaps["label_minutes"] = (snaps["label_ts"].dt.hour * 60
                              + snaps["label_ts"].dt.minute)
    earliest = []
    for row in cov.itertuples(index=False):
        if pd.isna(row.tip_minutes):
            earliest.append(pd.NaT)
            continue
        cutoff = row.tip_minutes - buffer_minutes
        cand = snaps[(snaps["report_date"] == row.game_date)
                     & (snaps["label_minutes"] <= cutoff)]
        earliest.append(cand["label_ts"].min() if len(cand) else pd.NaT)
    cov["earliest_label"] = earliest
    cov = cov[cov["earliest_label"].notna()]

    ev = events.dropna(subset=["PLAYER_ID", "weight"]).copy()
    ev["PLAYER_ID"] = ev["PLAYER_ID"].astype("int64")
    ev = ev.merge(cov[["GAME_ID", "game_date", "home", "away",
                       "earliest_label"]],
                  on="game_date", how="inner")
    playing = (ev["team_abbr"] == ev["home"]) | (ev["team_abbr"] == ev["away"])
    ev = ev[playing & (ev["label_ts"] <= ev["earliest_label"])].copy()
    ev["severity"] = -ev["weight"]
    ev = ev.sort_values(["GAME_ID", "PLAYER_ID", "label_ts", "severity", "id"])
    ev = ev.groupby(["GAME_ID", "PLAYER_ID"], as_index=False).head(1)
    return ev.rename(columns={"weight": "weight_early",
                              "status": "status_early"})[
        ["GAME_ID", "PLAYER_ID", "weight_early", "status_early", "team_abbr"]]


def availability_weights(designations: pd.DataFrame) -> pd.DataFrame:
    """The ``(GAME_ID, PLAYER_ID, weight)`` frame ``availability_features`` takes."""
    w = designations[["GAME_ID", "PLAYER_ID", "weight"]].copy()
    if w.duplicated(["GAME_ID", "PLAYER_ID"]).any():
        raise AssertionError("designations are not unique per game and player")
    return w


def prior_squad_keys(games: pd.DataFrame) -> pd.DataFrame:
    """``(GAME_ID, PLAYER_ID)`` for every player on his team's PREVIOUS game.

    These are the only designations that can move ``availability_features``:
    the block sums expected minutes over the prior-game squad, so a player who
    was not in it has no minutes to discount.  Kept here rather than in a
    script because every measurement should count "would it have mattered" the
    same way.
    """
    g = games.copy()
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"]).dt.normalize()
    pm = nd.player_minutes(g)
    squad = (pm.groupby(["TEAM_ABBREVIATION", "GAME_ID"], as_index=False)
             .agg(squad=("PLAYER_ID", lambda s: frozenset(s))))
    dates = g.set_index("GAME_ID")["GAME_DATE"]
    squad["GAME_DATE"] = squad["GAME_ID"].map(dates)
    squad = squad.sort_values(["TEAM_ABBREVIATION", "GAME_DATE", "GAME_ID"])
    squad["prev_squad"] = squad.groupby(
        "TEAM_ABBREVIATION", sort=False)["squad"].shift(1)
    rows = []
    for row in squad.itertuples(index=False):
        if isinstance(row.prev_squad, frozenset):
            rows.extend((row.GAME_ID, pid) for pid in row.prev_squad)
    return pd.DataFrame(rows, columns=["GAME_ID", "PLAYER_ID"])


def diagnostics(events: pd.DataFrame, designations: pd.DataFrame,
                coverage: pd.DataFrame, games: pd.DataFrame) -> dict:
    """Coverage and identity numbers, reported alongside every measurement."""
    g = games.copy()
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"]).dt.normalize()
    n_games = int(len(g))
    has = coverage[coverage["has_snapshot"]]
    out = {
        "games": n_games,
        "games_with_pre_tip_snapshot": int(len(has)),
        "coverage_share": float(len(has) / n_games) if n_games else 0.0,
        "events_stored": int(len(events)),
        "events_resolved": int(events["PLAYER_ID"].notna().sum()),
        "events_ambiguous": int(events["ambiguous"].sum()),
        "resolution_rate": float(events["PLAYER_ID"].notna().mean()),
        "match_methods": {
            str(k): int(v) for k, v in
            events["match_method"].value_counts().items()
        },
        "status_counts": {
            str(k): int(v) for k, v in
            designations["status"].value_counts().items()
        },
        "designations_applied": int(len(designations)),
        "games_with_designation": int(designations["GAME_ID"].nunique()),
    }
    if not designations.empty:
        keys = prior_squad_keys(g)
        applied = designations.merge(keys, on=["GAME_ID", "PLAYER_ID"],
                                     how="inner")
        out["designations_on_prior_squad"] = int(len(applied))
        out["games_with_prior_squad_designation"] = int(
            applied["GAME_ID"].nunique())
        out["status_on_prior_squad"] = {
            str(k): int(v) for k, v in
            applied["status"].value_counts().items()
        }
    else:
        out["designations_on_prior_squad"] = 0
        out["games_with_prior_squad_designation"] = 0
        out["status_on_prior_squad"] = {}
    return out
