"""Does a feature block earn its place? Score any named block, one at a time.

Issue #19 wanted to know whether the 52 league-wide ``*_RANK`` columns were
redundant. It measured one block, once, by hand. There was no instrument behind
that, so the next block - and the one after that - each got their own ad-hoc
script. This is the instrument: give it a block name, it fits `full` and
`minus <block>` across the same 5 walk-forward splits, and returns a paired
bootstrap interval on the difference.

    python scripts/feature_block_ablation.py --list-blocks
    python scripts/feature_block_ablation.py --blocks rank four_factors_rebound
    python scripts/feature_block_ablation.py          # every registered block

WHAT A RESULT MEANS, AND WHAT IT DOES NOT
-----------------------------------------
The number reported is ``loss(full) - loss(minus <block>)`` in nats of log loss.
Read it with this repo's sign convention, which has been got wrong three times:

    ``paired_bootstrap_delta(y, p_a, p_b)`` returns ``loss(a) - loss(b)``, so a
    POSITIVE delta means **b is better**.

Applied here, ``a = full`` and ``b = minus <block>``. So:

    delta < 0  (interval below zero)  ->  removing the block HURT. It earns its
                                          place. Keep it.
    delta > 0  (interval above zero)  ->  removing the block HELPED. It is not
                                          earning its place.
    interval straddles zero           ->  NULL. Nothing was established.

The interval is the result; the point estimate is decoration. A block whose
interval straddles zero is labelled a null and is not described as a winner or a
loser, because at the sample sizes here a bare point estimate of order 0.001-0.002
nats is noise with a decimal point on it. Every printed line carries its interval
for exactly that reason - there is no code path in this file that prints a
comparison without one beside it.

THE BLOCK REGISTRY IS THE POINT
-------------------------------
Blocks are declared in ``FEATURE_BLOCKS``, an explicit name -> column-list map,
not discovered by matching column-name patterns. A pattern-derived block changes
meaning silently when a column is renamed or when a builder starts emitting a new
``ewm_*`` field, and the resulting ablation is reported under a familiar name
while measuring something else. A visible literal list cannot do that: it either
names a column that exists or it raises.

THE ``*_RANK`` BLOCK IS NOT THE COLUMNS #19 MEASURED
-----------------------------------------------------
This is worth stating at the top rather than in a footnote. Issue #19's block was
the 52 ``*_RANK`` columns of the legacy ``Data/dataset.sqlite`` snapshot table
(26 per side, suffixed ``.1`` for the away club). Those columns DO NOT EXIST in the
leak-free frame this harness models on. ``download_nba_data.py`` deletes every
column ending ``_RANK`` at download time, because a rank computed against the
same day's whole field is a contemporaneous league quantity that carried the
answer; ``Create_Recency_Features.py`` excludes them for the same reason.

So the `rank` block here is a *causal reconstruction*: the league-wide standing a
team would have held entering the game, computed from strictly-past games only. It
answers the question #19 was asking - does a league-relative ordinal view of team
strength add anything over the underlying continuous value - without the leak that
made the original columns unusable. It is NOT a re-measurement of #19's columns
and must not be reported as one.

FAILURE MODES THIS FILE GUARDS AGAINST
---------------------------------------
Each of these has actually cost this repository a result. They are listed here,
guarded below, and pinned in ``Tests/test_feature_block_ablation.py``.

1. IMPUTATION AND SCALING FIT ON THE FULL FRAME.
   The imputer median and the scaler mean/sd are statistics of the rows they are
   fit on. Fitting them over the whole frame lets each test block's own
   distribution inform the representation it is scored in. The leak is small -
   O(1/n_test) - and therefore invisible: it does not move the metric enough to
   notice and not large enough to look like anything. Guarded by ``fit_oof`` in
   ``recency_ablation.py``, which the ablation reuses rather than re-implementing,
   so both scripts fit identically by construction.

2. SCORING ROWS THAT WERE NEVER OUT-OF-FOLD.
   ``walk_forward_split`` holds the first half of the data as training-only, so
   roughly 6,600 of the 13,199 rows have never been predicted by a model that had
   not seen them. Scoring them would score the model on its own training rows and
   would flatter every arm equally - which is worse than useless, because the
   arms stay paired and the comparison still "looks" valid. Guarded by returning
   only the non-NaN out-of-fold rows, and by asserting every arm agrees on which
   rows those are.

3. READING THE BOOTSTRAP SIGN BACKWARDS.
   Guarded by ``paired_bootstrap_delta``'s own convention plus ``classify()``
   below, which is a pure function of the interval and is unit-tested against a
   case where the right answer is forced by construction. The printed table
   restates the convention in its own header, immediately above the numbers.

4. AN ABLATION THAT DOES NOT ACTUALLY ABLATE.
   If a block names columns absent from the frame, dropping them is a no-op and
   `full` and `minus` are the same model - so the delta is exactly 0.0 with a
   degenerate interval, which reads like a clean null rather than a bug. Guarded:
   every registry column must resolve against the built frame, and the two models
   must be fitted on different column sets or the run raises.

Run:  python scripts/feature_block_ablation.py
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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from DataProviders import nba_data as nd  # noqa: E402
import recency_ablation as ra  # noqa: E402

SEED = 42
N_SPLITS = 5
# The committed default, and the arm recency_followup.py showed to be the optimum.
# Held fixed so a block result is comparable with the decay results: if the
# halflife moved between runs, no two numbers here could be compared at all.
HALFLIFE = 10.0
MIN_GAMES = 5

# League-wide standing features, built causally by `league_rank_features` below.
# Kept as module constants because the block registry names them.
RANK_METRICS = ("margin", "net_rating", "off_rating", "def_rating")

# The Four Factors. Both rebound shares are reachable: `rolling_team_features` takes
# EFG_PCT / TM_TOV_PCT / FTA_RATE from the four-factors table and OREB_PCT / DREB_PCT
# from the advanced table, and emits all five as `ewm_*`. Historically it asked the
# four-factors table for ORB_PCT/DRB_PCT, which it has never published, so both
# rebounding features were silently absent from every model here.
FOUR_FACTOR_METRICS = ("efg", "tov", "ftr", "orb", "drb")

# The relative-rest candidate block (theory dossier H4), declared before it was
# measured in results/experiments.jsonl ("H4-prereg-relative-rest"). Kept here,
# above the registry, because FEATURE_BLOCKS names these columns. Built by
# `relative_rest_features` below from rest_and_travel() outputs.
RELATIVE_REST_COLUMNS = (
    "rest_diff_cap4", "b2b_diff",
    "rel_rest_d_le_m2", "rel_rest_d_m1", "rel_rest_d_p1", "rel_rest_d_ge_p2",
)


def _sides(*stems: str) -> list[str]:
    """Every stem as its home and away column, in a stable order."""
    return [f"{side}_{stem}" for stem in stems for side in ("home", "away")]


# --------------------------------------------------------------------------- #
# THE BLOCK REGISTRY
# --------------------------------------------------------------------------- #
#
# An explicit map, deliberately. See the module docstring for why this is a
# literal list and not a prefix pattern: a pattern that silently matches a new
# column keeps the block's name while changing what the block is.
#
# Every name below must resolve against the frame `build_feature_frame` returns.
# `resolve_block` raises on an unresolved name rather than quietly scoring an
# ablation that dropped nothing, so a renamed column is a loud failure.
FEATURE_BLOCKS: dict[str, dict[str, object]] = {
    # The issue's block. Causal reconstruction of the league-relative standing;
    # see the module docstring for why these are not #19's actual columns.
    "rank": {
        "columns": [f"{side}_rank_{m}"
                    for m in RANK_METRICS
                    for side in ("home", "away")],
        "note": "league-wide standing entering the game, rebuilt causally",
    },
    # The Four Factors in full, including both rebound shares.
    "four_factors_rebound": {
        "columns": _sides(*(f"ewm_{m}" for m in FOUR_FACTOR_METRICS)),
        "note": "Dean Oliver four factors, EWM'd: eFG%, TOV%, FT rate, OREB%, DREB%",
    },
    # Schedule state. Levels, not contrasts: "this club is travelling 2,000
    # miles" is a level fact about the game, not a difference between the clubs.
    "schedule": {
        "columns": [
            "home_rest_days", "away_rest_days",
            "home_back_to_back", "away_back_to_back",
            "home_in_3_of_4", "away_in_3_of_4",
            "travel_miles",
        ],
        "note": "rest, back-to-back, 3-in-4, travel",
    },
    # ---------------------------------------------------------------------
    # RELATIVE REST - pre-registered candidate (theory dossier H4), declared in
    # results/experiments.jsonl ("H4-prereg-relative-rest") BEFORE it was
    # measured. Rest as a DIFFERENTIAL between the clubs (Esteves et al. 2021),
    # not as two absolute day counts. New column names by design: the schedule
    # block keeps its absolute rest columns, so test_blocks_do_not_overlap holds
    # and the two blocks can be ablated independently.
    #
    # NOTE ON THE FRAME: these columns are emitted by build_feature_frame (see
    # relative_rest_features), so `full` here carries them. H4's own comparisons
    # pin the canonical 67-column base explicitly
    # (scripts/h4_rest_differential.py, arms A1/A2/A3) rather than relying on
    # whatever the frame currently totals.
    # ---------------------------------------------------------------------
    "relative_rest": {
        "columns": list(RELATIVE_REST_COLUMNS),
        "note": "rest-day DIFFERENTIAL capped at 4+ days, B2B differential, and "
                "Esteves-style relative-rest categories (equal rest is the "
                "reference level)",
    },
    # Games played by each side. Included because it is the variable
    # recency_followup.py stratified on, and a block that cannot be scored under
    # the same stratification cannot be compared with that result.
    "experience": {
        "columns": ["home_games_played", "away_games_played"],
        "note": "how deep into the season each side is",
    },
    # Pace. Its own block because it is the one quantity in this frame that is a
    # property of the GAME rather than of either club, and ablating it asks
    # whether the game-speed term earns its place at all.
    "pace": {
        "columns": ["home_ewm_pace", "away_ewm_pace"],
        "note": "expected tempo, both sides",
    },
    # Net rating, kept as its own block rather than folded into a rating block.
    # `nba_data`'s KNOWN CAVEAT documents it as a redundant restatement of
    # off_rating - def_rating (measured R^2 0.99999735 on the real frame), and says
    # per-coefficient readings of it are not safe. Ablating it is the measurement
    # that caveat is asking for.
    # ---------------------------------------------------------------------
    # PRE-REGISTERED CANDIDATES.
    #
    # Every decay experiment in this project came back null, because all of them
    # reweight the SAME information. These four were written down and registered
    # BEFORE any was measured, so that the four results below cannot be
    # cherry-picked out of a longer search. The bar is Bonferroni across the four
    # looks (p < 0.0125) AND the |z| > 3 threshold this project adopted from
    # Vandenbruaene et al. (2022), whose null of unprofitability was never
    # rejected across 628 implementations.
    #
    # The prior differs across them and is stated rather than hidden:
    #   opponent_adjusted  corr 0.98 with the existing ewm_margin, so mostly a
    #                      rearrangement - expects a null, like every other
    #                      reweighting in this repo.
    #   home_away_split    corr 0.08 with it, i.e. nearly independent - the only
    #                      one carrying information the frame does not have.
    #   strength_of_schedule  a second-difference of the same quantity.
    #   opponent_characteristics  columns present in the source table since the
    #                      first download and never read by any model.
    # ---------------------------------------------------------------------
    "opponent_adjusted": {
        "columns": _sides("ewm_adj_margin"),
        "note": "form minus the opponent's own margin: beating a good club counts "
                "for less. Uses opponent quality, which no existing feature reads.",
    },
    "home_away_split": {
        "columns": _sides("ewm_margin_home", "ewm_margin_away", "home_edge"),
        "note": "each team's OWN home advantage instead of one league constant",
    },
    "strength_of_schedule": {
        "columns": _sides("ewm_sos"),
        "note": "quality of the opposition faced, carried rather than subtracted",
    },
    "opponent_characteristics": {
        "columns": _sides("ewm_opp_ftr", "ewm_opp_tov", "ewm_opp_orb"),
        "note": "OPP_FTA_RATE / OPP_TOV_PCT / OPP_OREB_PCT, present in every "
                "downloaded season and never read by any model here",
    },
    # ---------------------------------------------------------------------
    # PLAYER-LEVEL ELO, TOTALLED OVER THE EXPECTED-AVAILABLE ROSTER.
    #
    # Every other feature in this repo describes an INSTITUTION. This one
    # describes people: each player carries an online Elo rating, and the feature
    # is the sum of `rating x expected minutes` over the squad expected to be
    # available tonight. It exists because the one mechanism in this project that
    # is NOT a null is availability (+0.00382 nats), and that mechanism can only
    # discount a roster by WHO is in it if something knows who that is.
    #
    # `src/Models/player_ratings.py` already found RAPM-style player ratings a
    # null across three families and both estimation methods. This is a different
    # estimator - sequential, online, one parameter - aimed at a different
    # question. It is also the block most likely to be a duplicate of team form:
    # a player's rating is largely a measure of the team he plays for, so this may
    # turn out to be a 1500-scaled restatement of `ewm_margin`. The measured
    # correlation is printed alongside the ablation for exactly that reason.
    #
    # Availability means EXPECTED, from trailing minutes. `AVAILABLE_FLAG` in the
    # source data is written after the game and would be a leak.
    # ---------------------------------------------------------------------
    "player_elo": {
        "columns": _sides("elo_available", "elo_mean", "elo_top5"),
        "note": "per-player Elo summed (or averaged) over the expected-available "
                "squad, weighted by expected minutes",
    },
    "net_rating": {
        "columns": ["home_ewm_net_rating", "away_ewm_net_rating"],
        "note": "documented as redundant with off - def; scored to check",
    },
}

# Verdict vocabulary. Kept as constants so the printing code, the JSON artifact
# and the tests cannot drift into describing the same outcome three ways.
EARNS = "earns its place"
DOES_NOT_EARN = "does not earn its place"
NULL = "null: interval straddles zero"


def classify(ci_lo: float, ci_hi: float) -> str:
    """What a bootstrap interval says about a block.

    Pure, and deliberately not a function of the point estimate: the sign of a
    0.001-nat point estimate on this data is not evidence, and letting it pick the
    verdict is how a null gets promoted to a finding.

    `delta = loss(full) - loss(minus block)`, so a POSITIVE delta means the ablated
    model is BETTER and the block was not earning its place.
    """
    if ci_hi < 0:
        # Removing the block reliably costs log loss. The block was doing work.
        return EARNS
    if ci_lo > 0:
        # Removing the block reliably saves log loss. The block was a net cost.
        return DOES_NOT_EARN
    return NULL


# --------------------------------------------------------------------------- #
# Frame construction - all through the shared layer
# --------------------------------------------------------------------------- #

def league_rank_features(games: pd.DataFrame, roll: pd.DataFrame,
                         metrics: tuple[str, ...] = RANK_METRICS) -> pd.DataFrame:
    """League-wide rank of each side's strength entering the game, built causally.

    Issue #19's block was a league-wide RANK. The original columns cannot be
    rebuilt from leak-free data because they were deleted at download time as a
    leak, so this reconstructs the *quantity* instead: for each game, where each
    club sits in the league's ordering of a metric, using only games that already
    happened.

    CAUSALITY. `rolling_team_features` computes each team's EWM from its own past
    only (see `_ewm_by_team`). Ranking teams by that value within a game date
    therefore uses nothing from that date or later: the value a club carries into
    its game was already fixed before tip-off. The league the rank is taken over
    is "all clubs with a value as of that date", which is a function of the past,
    not of the game being predicted.

    Rank 1 is the strongest club. NaN for a side with no history yet, which the
    pipeline's imputer handles on training rows only.
    """
    dates = roll["GAME_DATE"].to_numpy()
    frames = []
    for side in ("home", "away"):
        block = pd.DataFrame({
            "team": games[side].to_numpy(),
            "GAME_DATE": dates,
            # Mirror each club's appearances onto a single league calendar so a
            # club's most recent value carries forward to the dates it did not
            # play. Without this a club is absent from every date it sat out and
            # cannot be ranked at all on those dates.
            **{m: roll[f"{side}_ewm_{m}"].to_numpy(dtype=float) for m in metrics},
        })
        frames.append(block)
    long = pd.concat(frames, ignore_index=True)

    calendar = pd.Index(sorted(pd.unique(dates)))
    out = {}
    for m in metrics:
        # (date x team) -> that club's most recent value, forward-filled.
        panel = (long.pivot_table(index="GAME_DATE", columns="team", values=m,
                                  aggfunc="last")
                     .reindex(calendar)
                     .ffill())
        # RANK THE PREVIOUS GAME DATE'S SNAPSHOT, not this one's.
        #
        # Without the shift, a club that plays on date d is ranked on the value it
        # carries INTO its own game (everything strictly before d), while a club
        # sitting out that night is ranked on a value carried forward from some
        # earlier date. That is two different league tables inside one row, and
        # which one a club lands in depends on whether it happens to be playing -
        # so the same club holds two different ranks on either side of its own
        # rest day. Shifting by one game date puts every club in a single snapshot
        # taken before ANY game on d, which is the standing that was actually knowable
        # entering the night. It is also strictly causal in the strong sense: a
        # game's row cannot depend on which other clubs happened to play that day.
        ranked = panel.shift(1).rank(axis=1, ascending=False, method="average")

        dpos = {d: i for i, d in enumerate(ranked.index)}
        tpos = {t: j for j, t in enumerate(ranked.columns)}
        vals = ranked.to_numpy(dtype=float)
        rows = pd.Series(dates).map(dpos).to_numpy(dtype=float)
        for side in ("home", "away"):
            cols = games[side].map(tpos).to_numpy(dtype=float)
            out[f"{side}_rank_{m}"] = vals[rows.astype(int), cols.astype(int)]

    return pd.DataFrame(out, index=games.index).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# RELATIVE REST - pre-registered candidate block (theory dossier H4)
# --------------------------------------------------------------------------- #
#
# Declared in results/experiments.jsonl ("H4-prereg-relative-rest") BEFORE it
# was measured. Built from rest_and_travel() outputs - the existing schedule
# data - as DERIVED columns with new names, so the `schedule` block keeps its
# absolute rest columns and test_blocks_do_not_overlap holds. This is the same
# pattern as league_rank_features: new feature-block code lives here, and the
# shared layer is not touched. Column names: RELATIVE_REST_COLUMNS above.


def relative_rest_features(games: pd.DataFrame,
                           sched: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rest as a DIFFERENTIAL between the two clubs, not as two day counts.

    ``rest_diff_cap4``
        clip(rest_days, 1, 4) per side, then home minus away, in [-3, 3]. The
        cap makes "4+ days" one category, which also kills the October
        season-indicator artifact the raw column carries (a 288-day offseason
        rest is not rest information).
    ``b2b_diff``
        home_back_to_back - away_back_to_back, in [-1, 1]: nonzero exactly when
        one side is on a back-to-back and the other is not.
    ``rel_rest_d_*``
        the Esteves-style relative-rest factor, as dummies on
        d = rest_diff_cap4 with equal rest (d == 0) as the reference level.

    Pre-declared caveat (H4-prereg-relative-rest): the continuous diff and the
    factor dummies are collinear by construction. The block is scored as ONE
    unit and no per-column attribution is claimed.

    Causality is inherited from rest_and_travel: every input is knowable from
    the schedule before tip-off. NaN stays NaN (a team's first game on record
    has no prior date) and is left to the pipeline's train-only imputer.
    """
    if sched is None:
        sched = nd.rest_and_travel(games)
    h = sched["home_rest_days"].to_numpy(dtype=float)
    a = sched["away_rest_days"].to_numpy(dtype=float)
    d = np.clip(h, 1.0, 4.0) - np.clip(a, 1.0, 4.0)

    out = pd.DataFrame(index=games.reset_index(drop=True).index)
    out["rest_diff_cap4"] = d
    out["b2b_diff"] = (sched["home_back_to_back"].to_numpy(dtype=float)
                       - sched["away_back_to_back"].to_numpy(dtype=float))
    for col, mask in (("rel_rest_d_le_m2", d <= -2.0),
                      ("rel_rest_d_m1", d == -1.0),
                      ("rel_rest_d_p1", d == 1.0),
                      ("rel_rest_d_ge_p2", d >= 2.0)):
        # NaN in, NaN out: a missing differential must not land on the
        # reference level as if the teams were equally rested.
        out[col] = np.where(np.isnan(d), np.nan, mask.astype(float))
    return out.reset_index(drop=True)


def build_feature_frame(games: pd.DataFrame, halflife: float = HALFLIFE,
                        include_ranks: bool = True) -> pd.DataFrame:
    """The standard modelling frame, plus the causal league-rank block.

    Every feature comes from the shared layer in ``src/DataProviders/nba_data.py``:
    ``rolling_team_features`` for team form, ``rest_and_travel`` for schedule state.
    This file builds no second loader and no second feature computation, so a block
    result here is comparable with any other number produced from that layer.

    The four factors need no separate source. ``rolling_team_features`` reads
    EFG_PCT / TM_TOV_PCT / FTA_RATE from the four-factors table and OREB_PCT /
    DREB_PCT from the advanced table, and emits all five as ``ewm_*``. An earlier
    version of that function asked the four-factors table for ``ORB_PCT`` and
    ``DRB_PCT``, which it has never published, so both rebound features were
    silently absent from every model in this repo; ``resolve_block`` is what makes
    that class of disappearance loud rather than invisible.
    """
    roll = nd.rolling_team_features(games, halflife=halflife, min_games=MIN_GAMES)
    sched = nd.rest_and_travel(games)
    rel = relative_rest_features(games, sched)
    opp = nd.opponent_adjusted_features(games, halflife=halflife)
    elo = nd.player_elo_features(games, halflife=halflife)

    keep = lambda d: d.select_dtypes(include=[np.number])  # noqa: E731
    # roll and sched each carry a GAME_DATE; concatenating datetime columns hands
    # the imputer a dtype it cannot promote to float. Select numerics rather than
    # maintaining a delete-list.
    blocks = [keep(roll).reset_index(drop=True),
              keep(sched).reset_index(drop=True),
              keep(rel).reset_index(drop=True),
              keep(opp).reset_index(drop=True),
              keep(elo).reset_index(drop=True)]
    if include_ranks:
        blocks.append(league_rank_features(games, roll).reset_index(drop=True))

    frame = pd.concat(blocks, axis=1)
    # Two sources producing the same column name means one of them is redundant and
    # which one is winning is arbitrary. Refused rather than silently de-duplicated:
    # pandas would otherwise keep the first and quietly discard the second.
    dupes = sorted({c for c in frame.columns if list(frame.columns).count(c) > 1})
    if dupes:
        raise AssertionError(
            f"the feature builders emitted {len(dupes)} duplicate column(s): "
            f"{dupes}. Two sources are writing the same feature; dropping one by "
            f"hand here would hide which one is correct."
        )
    # A feature that is identical in every row cannot be predicted from, and only
    # makes the design matrix singular. `is_home` is constant 1.0 by construction.
    frame = frame.loc[:, frame.nunique(dropna=False) > 1]
    assert len(frame) == len(games), "feature block lost rows"
    return frame.reset_index(drop=True)


def resolve_block(name: str, frame: pd.DataFrame) -> list[str]:
    """The registry columns for `name` that are actually present in `frame`.

    Raises on an unknown block. Raises on a known block whose columns are missing
    from the frame, rather than returning a short list: an ablation that silently
    drops nothing fits the same model twice, yields a delta of exactly 0.0 with a
    zero-width interval, and is reported as a clean null. That is a bug wearing a
    null's clothes.
    """
    if name not in FEATURE_BLOCKS:
        raise KeyError(f"unknown feature block {name!r}. Known blocks: "
                       f"{sorted(FEATURE_BLOCKS)}")
    columns = list(FEATURE_BLOCKS[name]["columns"])  # type: ignore[index]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(
            f"feature block {name!r} names {len(missing)} column(s) that are not "
            f"in the frame: {missing}. The registry has drifted from the feature "
            f"builders; fix the block rather than scoring an ablation that drops "
            f"nothing."
        )
    return columns


def drop_block(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """`frame` without `columns`, in the original column order.

    Refuses an empty list and asserts the width actually changed. A no-op drop is
    the failure mode above and it is invisible downstream - the two arms are the
    same model, the delta is exactly 0.0, and the interval has zero width, which is
    indistinguishable from a real null in the output. Refused here, at the only
    place that can still see it.
    """
    if not columns:
        raise AssertionError(
            "refusing to ablate zero columns: both arms would be the same model, "
            "so the comparison would report a meaningless delta of exactly 0.0 with "
            "a zero-width interval. A block that resolves to no columns is a "
            "registry problem - see `resolve_block`."
        )
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"cannot drop columns absent from the frame: {missing}")
    out = frame.drop(columns=list(columns))
    if out.shape[1] != frame.shape[1] - len(columns):
        raise AssertionError(
            f"dropping {len(columns)} column(s) changed the width by "
            f"{frame.shape[1] - out.shape[1]}; the ablation did not ablate"
        )
    return out


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

def score_block(frame: pd.DataFrame, y, splits, block: str, *,
                n_boot: int = 2000, seed: int = SEED) -> dict:
    """Paired-bootstrap comparison of `full` against `full` minus one block.

    Both arms use the same splits, the same seed and the same fitter, and are
    scored on exactly the same out-of-fold rows, so the difference is paired and
    the bootstrap of that difference is the right uncertainty on it.
    """
    columns = resolve_block(block, frame)
    y = np.asarray(y)

    p_full, y_oof, mask = ra.fit_oof(frame, y, splits)
    p_minus, y_minus, mask_minus = ra.fit_oof(drop_block(frame, columns), y, splits)

    # Guards failure mode 2 for the paired comparison specifically: the bootstrap
    # resamples rows, so two arms scored on different rows would not be paired at
    # all and the interval would describe nothing.
    if not np.array_equal(mask, mask_minus):
        raise AssertionError("arms disagreed on which rows are out-of-fold")
    if not np.array_equal(y_oof, y_minus):
        raise AssertionError("arms disagreed on the scored labels")

    d = nd.paired_bootstrap_delta(y_oof, p_full, p_minus, n_boot=n_boot, seed=seed)
    se = (d["ci_hi"] - d["ci_lo"]) / (2 * 1.959964)
    z = abs(d["delta"] / se) if se > 0 else float("nan")

    full_metrics = nd.evaluate(y_oof, p_full)
    minus_metrics = nd.evaluate(y_oof, p_minus)
    return {
        "block": block,
        "n_columns": len(columns),
        "columns": columns,
        "delta": d["delta"],
        "ci_lo": d["ci_lo"],
        "ci_hi": d["ci_hi"],
        "p_positive": d["p_positive"],
        "n_boot": n_boot,
        "z": z,
        # Classified from the INTERVAL, never from the sign of the point estimate.
        "verdict": classify(d["ci_lo"], d["ci_hi"]),
        "log_loss_full": full_metrics["log_loss"],
        "log_loss_minus": minus_metrics["log_loss"],
        "auc_full": full_metrics["auc"],
        "auc_minus": minus_metrics["auc"],
        "n_scored": int(len(y_oof)),
    }


def _fmt(result: dict) -> str:
    """One comparison, always with its interval attached."""
    ci = f"[{result['ci_lo']:+.5f}, {result['ci_hi']:+.5f}]"
    return (f"  {result['block']:>22s}  {result['n_columns']:>3d} cols  "
            f"{result['delta']:+.5f}  {ci:>24s}  "
            f"{result['p_positive']:>7.3f}  {result['z']:>5.2f}  "
            f"{result['verdict']}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score named feature blocks by paired bootstrap.")
    ap.add_argument("--blocks", nargs="+", metavar="NAME",
                    help="blocks to score (default: every registered block)")
    ap.add_argument("--list-blocks", action="store_true",
                    help="print the block registry and exit")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="bootstrap seed. Changing it is the REPLICATION: the "
                         "script's own verdict text says an exclusion is only a "
                         "candidate until it survives a second seed.")
    ap.add_argument("--splits", type=int, default=N_SPLITS,
                    help="walk-forward folds. A result that only exists at five "
                         "folds is a property of the fold count.")
    ap.add_argument("--out", default="results/feature_block_ablation.json")
    args = ap.parse_args()

    if args.list_blocks:
        print("registered feature blocks\n")
        width = max(len(n) for n in FEATURE_BLOCKS)
        for name in sorted(FEATURE_BLOCKS):
            spec = FEATURE_BLOCKS[name]
            cols = list(spec["columns"])  # type: ignore[arg-type]
            print(f"  {name:<{width}s}  {len(cols):>3d} cols  {spec['note']}")
            print(f"  {'':<{width}s}        {', '.join(cols)}")
        return 0

    unknown = [b for b in (args.blocks or []) if b not in FEATURE_BLOCKS]
    if unknown:
        raise SystemExit(f"unknown feature block(s) {unknown}. "
                         f"Known blocks: {sorted(FEATURE_BLOCKS)}")
    blocks = args.blocks or sorted(FEATURE_BLOCKS)

    games = nd.build_games()
    splits = nd.walk_forward_split(games, n_splits=args.splits)
    frame = build_feature_frame(games)

    # Every selected block must resolve BEFORE any model is fit, so a registry
    # typo fails in one second rather than after the first multi-second fit.
    resolved = {b: resolve_block(b, frame) for b in blocks}

    print(f"games {len(games):,}  splits {len(splits)}  "
          f"train sizes {[len(tr) for tr, _ in splits]}")
    print(f"test sizes  {[len(te) for _, te in splits]}  "
          f"frame {frame.shape[1]} cols  halflife {HALFLIFE:g}")
    print(f"out-of-fold rows: {sum(len(te) for _, te in splits):,} "
          f"(the first {int(len(games) * 0.5):,} rows are training-only by "
          f"design and are never scored)")

    y = games["y"].to_numpy()
    print("\nreference metrics (full frame)")
    p_full, y_oof, _ = ra.fit_oof(frame, y, splits)
    m = nd.evaluate(y_oof, p_full)
    print(f"  log loss {m['log_loss']:.5f}  brier {m['brier']:.5f}  "
          f"auc {m['auc']:.5f}  acc {m['accuracy']:.4f}")

    print("\nblock ablations, each vs the full frame")
    # The convention, restated immediately above the numbers it governs. This is
    # the header that stops the table below from being read backwards.
    print("  delta = loss(full) - loss(minus <block>); "
          "POSITIVE means the block does NOT earn its place")
    print(f"  {'block':>22s}  {'size':>8s}  {'delta':>8s}  "
          f"{'95% CI':>24s}  {'P(>0)':>7s}  {'|z|':>5s}  verdict")

    results = {}
    for block in blocks:
        res = score_block(frame, y, splits, block,
                          n_boot=args.n_boot, seed=args.seed)
        results[block] = res
        print(_fmt(res))

    # ------------------------------------------------------------------ verdict
    print("\nVERDICT")
    earn = [r["block"] for r in results.values() if r["verdict"] == EARNS]
    lose = [r["block"] for r in results.values() if r["verdict"] == DOES_NOT_EARN]
    null = [r["block"] for r in results.values() if r["verdict"] == NULL]

    for block in earn + lose:
        res = results[block]
        ci = f"[{res['ci_lo']:+.5f}, {res['ci_hi']:+.5f}]"
        print(f"  {block}: interval {ci} EXCLUDES zero -> {res['verdict']}.")
    if null:
        print(f"  NULL result (interval straddles zero): {null}")
        for block in null:
            res = results[block]
            ci = f"[{res['ci_lo']:+.5f}, {res['ci_hi']:+.5f}]"
            print(f"    {block:<22s} delta {res['delta']:+.5f}  {ci:>24s}  "
                  f"-- no evidence either way")
    if not earn and not lose:
        print("  No block's interval excludes zero. Nothing here distinguishes a")
        print("  block that earns its place from one that does not, and the")
        print("  point estimates above are not evidence to the contrary.")

    # A block whose point estimate leans one way but whose interval straddles zero
    # is STILL a null. Naming the lean is useful; calling it a result is not.
    for block in null:
        res = results[block]
        if res["p_positive"] >= 0.90 or res["p_positive"] <= 0.10:
            direction = "removing it HELPED" if res["p_positive"] >= 0.90 \
                else "removing it HURT"
            print(f"    {block}: {res['p_positive']:.3f} of bootstrap draws say "
                  f"{direction}, but the interval straddles zero. That is a null "
                  f"with a lean, not a finding.")

    # MULTIPLICITY. Each interval is a 95% interval, and scoring k blocks at 95%
    # means roughly k*0.05 false positives are expected from noise alone. Say so
    # rather than letting a single exclusion read as a discovery. The blocks here
    # are a registry fixed before the run, not a search over candidates, which is
    # the same reason issue #7's 100-trial sweep was unresolvable: the number of
    # looks taken matters, and it is stated here so a reader can weigh it.
    n_blocks = len(results)
    n_excluded = len(earn) + len(lose)
    if n_excluded:
        print(f"\n  MULTIPLICITY: {n_excluded} of {n_blocks} scored blocks has an "
              f"interval excluding zero at 95%.")
        print(f"  With {n_blocks} looks at a 95% level, about "
              f"{n_blocks * 0.05:.1f} exclusion(s) are expected from noise alone.")
        print("  Treat the exclusion above as a candidate worth a second angle - a")
        print("  replication at another bootstrap seed, or a different fold count -")
        print("  and not as a settled result.")
    print("  Read every line with its interval: a bare point estimate of order")
    print("  0.001-0.002 nats on this sample is noise, not a finding.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated_utc": pd.Timestamp.now("UTC").isoformat(),
        # Run state, not module constants. This writer used to record SEED and
        # N_SPLITS here, so every replication run - the runs whose entire point
        # is a non-default --seed or --splits - was logged under the DEFAULT
        # configuration, turning a replication into a second copy of the
        # original on paper. `halflife` is the constant only because no flag can
        # change it; if one appears, it must come from args too.
        "halflife": HALFLIFE,
        "seed": args.seed,
        "n_splits": args.splits,
        "n_games": int(len(games)),
        "n_out_of_fold": int(len(y_oof)),
        "n_features_full": int(frame.shape[1]),
        # Stated in the artifact as well as on screen, because the artifact is
        # what gets read months later and a delta without its sign rule is a
        # number waiting to be quoted backwards.
        "sign_convention": "delta = loss(full) - loss(minus <block>); a POSITIVE "
                           "delta means the ablated model is BETTER, so the block "
                           "does not earn its place",
        "registry": {name: list(spec["columns"])  # type: ignore[arg-type]
                     for name, spec in FEATURE_BLOCKS.items()},
        "resolved_blocks": {b: resolved[b] for b in blocks},
        "reference_metrics": m,
        "blocks": results,
        "summary": {"earns_its_place": earn,
                    "does_not_earn_its_place": lose,
                    "null": null},
    }, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())