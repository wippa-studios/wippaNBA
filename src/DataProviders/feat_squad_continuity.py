"""Squad continuity and load decay: the decomposition of availability that is missing.

WHY THIS EXISTS, AND WHY IT IS NOT ROUND 2's A1
------------------------------------------------
`availability_features` sums each player's expected minutes over the squad that
appeared in the team's previous game. That is the only confirmed feature win in
this project (+0.0038 nats). The project's own docstring says why it works and,
in the same breath, where it stops:

    "The squad's expected total is ~240 minutes for both teams, so the raw
     difference is mostly schedule noise. The contrast that carries signal is
     minutes-per-player, which is a team-quality measure rather than a pace one."

A basketball team plays 240 minutes of player-time whether its best player is
playing or not. The SUM therefore cannot see absence - it is pinned near 240 by
construction, and what survives is a rotation-depth proxy. Everything that
matters about availability is inside the composition of that 240, which the sum
throws away.

Round 2's `A1_lineup_continuity` asked the composition question - overlap
between consecutive lineups - and scored -0.0117 nats, which looks like a refutation
until you read its own coverage line: "2015-16 covers 26 of 1,230 games (2.1%);
2020-21 covers 722 of 1,080 (66.9%). Combined 748 of 13,199 = 5.7%, spanning 2
seasons", and only 134 of those rows fell in the out-of-fold region. At n=134 a
true effect of availability's size would not have cleared the noise floor. **A1
was a coverage failure, not a hypothesis failure**, and its author said so in the
preregistration ("Whatever |z| comes out is measured on a small sample and is
NOT a league-wide claim").

This module rebuilds the SAME hypothesis from `player_gamelogs/advanced_*.csv`,
which covers all 11 seasons. That is the specific reason this is not a repeat of
a logged null, and it is the whole justification.

THE THREE QUANTITIES
--------------------
  ``{side}_squad_overlap_2``    |prev_squad ∩ squad_{t-2}| / |prev_squad|.
        How much of last night's lineup was the same as the night before.
  ``{side}_minutes_overlap_2``  the same overlap weighted by each player's
        EXPECTED MINUTES rather than counted equally. A replaced star and a
        replaced end-of-bench both count as 1 above and cost wildly different
        amounts below.
  ``{side}_squad_turnover_2``   |prev_squad \ squad_{t-2}| / |prev_squad|. The
        complement, carried because a model that can see both directions is not
        relying on the sign of a one-sided ratio.
  ``{side}_load_decay``         sum over the prev squad of each player's OWN
        expected-minutes decay since the previous game. This is the quantity
        the raw sum masks: 240 minutes either way, but redistributed.

MEMBERSHIP, which is the leak this repo has been burned by twice
-----------------------------------------------------------------
Both `squad_{t-1}` and `squad_{t-2}` are complete before game t starts. Neither
is tonight's squad. There is no `all_prior` roster here precisely because the
question only makes sense against a bounded window; `player_elo_features` had to
rebuild one and is still a documented membership risk.

`Tests/test_leakage_gate.py` proves it mechanically for this builder:
truncation rebuilds on a truncated frame and requires the retained rows
unchanged -- the NaN pattern included -- and membership checks every expanded
query row against the team's own previous game.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _c in (ROOT, ROOT / "src"):
    if str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from src.DataProviders import nba_data as nd              # noqa: E402
from src.DataProviders.feat_squad_quality import (          # noqa: E402
    _player_streams, _prev_squad_roster, _asof_attach)


def _team_squad_sequence(streams: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, game) with the squad and the two previous squads.

    `shift(1)` and `shift(2)` over the team's own date-ordered game sequence.
    A team's game at index j needs nothing from index j or later.
    """
    per_game = (streams[["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "PLAYER_ID"]]
                .drop_duplicates(["TEAM_ABBREVIATION", "GAME_ID", "PLAYER_ID"]))
    per_game = (per_game.groupby(["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE"],
                                 as_index=False)
                .agg(squad=("PLAYER_ID", lambda s: frozenset(s))))
    per_game = per_game.sort_values(["TEAM_ABBREVIATION", "GAME_DATE"]).reset_index(drop=True)
    g = per_game.groupby("TEAM_ABBREVIATION", sort=False)
    per_game["squad_t1"] = g["squad"].shift(1)
    per_game["squad_t2"] = g["squad"].shift(2)
    return per_game


def _minutes_weighted_overlap(attached: pd.DataFrame) -> pd.DataFrame:
    """Overlap counts and minutes-weighted overlap, per (game, team).

    `in_both` is the AND of the two membership flags. Because both flags are
    derived from sets that are complete before the game, no information about
    tonight's participants enters.
    """
    d = attached.copy()
    d["w_exp_min"] = d["exp_min_now"].fillna(0.0)
    d["w_prev_exp_min"] = d["prev_exp_min"].fillna(0.0)
    d["in_both"] = (d["in_squad_t1"] & d["in_squad_t2"]).astype(float)
    d["w_in_both"] = d["in_both"] * d["w_exp_min"]
    d["in_t1"] = d["in_squad_t1"].astype(float)
    g = d.groupby(["GAME_ID", "TEAM_ABBREVIATION"], as_index=False).agg(
        squad_overlap_2=("in_both", "sum"),
        minutes_overlap_2=("w_in_both", "sum"),
        squad_t1_n=("in_t1", "sum"),
        squad_t1_exp_min=("w_exp_min", "sum"),
    )
    g["squad_overlap_2"] = g["squad_overlap_2"] / g["squad_t1_n"].replace(0, np.nan)
    g["minutes_overlap_2"] = g["minutes_overlap_2"] / g["squad_t1_exp_min"].replace(0, np.nan)
    g["squad_turnover_2"] = 1.0 - g["squad_overlap_2"]
    return g


def _explode_two_back(per_game: pd.DataFrame) -> pd.DataFrame:
    q = per_game.loc[per_game["squad_t1"].notna() & per_game["squad_t2"].notna()].copy()
    rows = []
    for _, r in q.iterrows():
        t1, t2 = r["squad_t1"], r["squad_t2"]
        for pid in t1:
            rows.append((r["GAME_ID"], r["TEAM_ABBREVIATION"], r["GAME_DATE"],
                         pid, True, pid in t2))
    return pd.DataFrame(rows, columns=["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE",
                                        "PLAYER_ID", "in_squad_t1", "in_squad_t2"])


def squad_continuity_features(games: pd.DataFrame, halflife: float = 15.0) -> pd.DataFrame:
    """Lineup-continuity and load-decay contrasts over the previous two games.

    Columns, for side in (home, away):
      ``{side}_squad_overlap_2``    count overlap, share of the prev squad
      ``{side}_minutes_overlap_2``  expected-minutes-weighted overlap, share
      ``{side}_squad_turnover_2``   complement of the count overlap
      ``{side}_load_decay``         sum over the prev squad of each player's own
                                    expected-minutes decay since the previous game
      ``{side}_squad_depth``        players in the prev squad who also played
                                    two games ago (a raw count, kept because the
                                    shares above are bounded and a count is not)
      and the mirrored ``*_diff`` for each, minus ``squad_depth``.
    """
    streams = _player_streams(games, halflife)
    per_game = _team_squad_sequence(streams)
    exploded = _explode_two_back(per_game)

    # Each player's expected minutes as of now, and as of their own previous
    # appearance. Both come from strictly-earlier rows: the first through the
    # standard strictly-before asof, the second by asking the same lookup one
    # game earlier.
    attached = _asof_attach(
        exploded[["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "PLAYER_ID"]],
        streams[["PLAYER_ID", "GAME_DATE", "exp_min"]],
        ["exp_min"],
    ).rename(columns={"exp_min": "exp_min_now"})

    # The previous-game value: as-of the team's previous game date, strictly
    # before it. Built by re-running the same asof against the same stream with
    # each team's own t-1 date, so it cannot read the game being predicted.
    prev_dates = per_game[["TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE"]].copy()
    prev_dates["prev_date"] = (prev_dates.groupby("TEAM_ABBREVIATION", sort=False)
                               ["GAME_DATE"].shift(1))
    # GAME_ID alone is NOT unique here: a game has one home-team row and one
    # away-team row, so the lookup key must be the (game, team) pair.
    ex_prev = exploded[["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID"]].copy()
    ex_prev = ex_prev.merge(prev_dates[["GAME_ID", "TEAM_ABBREVIATION", "prev_date"]],
                            on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")
    ex_prev["GAME_DATE"] = ex_prev["prev_date"]
    back = _asof_attach(
        ex_prev[["GAME_ID", "TEAM_ABBREVIATION", "GAME_DATE", "PLAYER_ID"]],
        streams[["PLAYER_ID", "GAME_DATE", "exp_min"]],
        ["exp_min"],
    ).rename(columns={"exp_min": "prev_exp_min"})
    attached = attached.merge(back[["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID",
                                    "prev_exp_min"]], on=["GAME_ID", "TEAM_ABBREVIATION",
                                                         "PLAYER_ID"], how="left")

    attached = attached.merge(
        exploded[["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID",
                  "in_squad_t1", "in_squad_t2"]],
        on=["GAME_ID", "TEAM_ABBREVIATION", "PLAYER_ID"], how="left")

    # Load decay: how much expected minutes each prev-squad player has LOST
    # since the previous game. max(0, ...) because a gain is the availability
    # feature's business; this column is asked only about loss.
    attached["load_decay"] = np.maximum(
        0.0, attached["prev_exp_min"].fillna(0.0) - attached["exp_min_now"].fillna(0.0))

    g = _minutes_weighted_overlap(attached)
    decay = (attached.groupby(["GAME_ID", "TEAM_ABBREVIATION"], as_index=False)
             .agg(load_decay=("load_decay", "sum"),
                  squad_depth=("in_squad_t2", "sum")))
    g = g.merge(decay, on=["GAME_ID", "TEAM_ABBREVIATION"], how="left")

    carry = ["squad_overlap_2", "minutes_overlap_2", "squad_turnover_2",
             "load_decay", "squad_depth"]
    g = g.set_index(["GAME_ID", "TEAM_ABBREVIATION"])
    if not g.index.is_unique:
        raise AssertionError(
            f"squad_continuity produced duplicate keys: "
            f"{g.index[g.index.duplicated()].tolist()[:3]}")

    gi = games[["GAME_ID", "home", "away"]].reset_index(drop=True)
    out = pd.DataFrame(index=gi.index)
    for side in ("home", "away"):
        idx = pd.MultiIndex.from_arrays([gi["GAME_ID"], gi[side]])
        for c in carry:
            out[f"{side}_{c}"] = g[c].reindex(idx).to_numpy()
    for c in carry:
        if c == "squad_depth":          # a count: no contrast
            continue
        out[f"home_{c}_diff"] = out[f"home_{c}"] - out[f"away_{c}"]
        out[f"away_{c}_diff"] = -out[f"home_{c}_diff"]

    if len(out) != len(games):
        raise AssertionError(f"{len(out)} rows for {len(games)} games")
    return out.reset_index(drop=True)