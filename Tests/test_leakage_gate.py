"""The mechanical leakage gate for the three prior-game-squad feature builders.

WHY THIS FILE EXISTS
--------------------
`feat_squad_quality`'s docstring has promised that "`Tests/test_leakage_gate.py`
plus `scripts/round3_verify.py` check both mechanically" -- membership and
truncation. The test file did not exist in repo history. This is it. It pins,
fixture-level and with no network, the leakage properties of the three builders:

  `feat_squad_quality`     squad-quality contrasts over the previous game's squad
  `feat_squad_continuity`  lineup continuity over the previous TWO games
  `feat_team_elo`          pre-game team Elo ratings

THE SIN IT GUARDS
-----------------
`docs/player_elo_leak.md`: a feature block measured |z| = 5.18 -- the largest
effect ever measured in this project -- by summing over the players who
ACTUALLY played. The future was not the leak; the MEMBERSHIP was. Nobody knows
before tip-off which five men take the floor, and a feature that quietly reads
tonight's participants is unattainable at prediction time no matter how causal
its weights look. A leak that produces excellent numbers is worse than no model
at all, and truncation tests alone cannot see it: cutting the future does not
change who was in the room tonight.

So the properties checked here are mechanical, not argued:

  MEMBERSHIP CAUSALITY    features for game t aggregate over squads of games
                          strictly before t (shift(1)/shift(2) semantics). A
                          player debuting tonight contributes NOTHING to
                          tonight's features; a player absent tonight still
                          contributes if he played the prior game (the
                          availability convention: the squad is what played,
                          not who dresses).
  TRUNCATION INVARIANCE   building features on a game set with the tail dropped
                          leaves the retained rows unchanged, NaN pattern
                          included.
  STRICTLY-CASUAL ATTACH  the value merge is
                          `merge_asof(..., allow_exact_matches=False)`, so it
                          never attaches a same-day row: in a same-day
                          doubleheader, game 2 cannot reach game 1's features.
  TEAM-ELO, PRE-UPDATE    `team_elo_features` attaches
                          `rating_*_before` from `baselines.elo_rating_pass`
                          under its documented `same_day="batched"` convention
                          -- the state prior to each game's own update, with no
                          same-day bleed between doubleheader games.

THE TOLERANCE, PER BUILDER
--------------------------
Measured, not assumed. The Elo pass is a sequential recursion and is
BIT-IDENTICAL under truncation and under every perturbation here, so its tests
assert exact equality. The squad builders groupby-sum floats whose summation
order depends on the frame being sorted, and the retained rows move by up to
~8.5e-14 (squad quality) / ~8.9e-16 (continuity) when the tail is dropped --
float-summation noise, not a leak. Those builders assert the round-3 gate's own
pass bar, 1e-12 (same bar as `scripts/round3_verify.py::gate1_truncation` and
`Tests/test_logreg_rolling.py`), and the test names say so. The NaN pattern is
ALWAYS compared exactly: a cell that backfills when future games are added is a
leak with no tolerance that could excuse it.

`scripts/round3_verify.py` runs the same gates on the full cached frame as a
pre-scoring check; this file keeps them under pytest so a regression fails the
suite instead of only the round.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT, ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from src.DataProviders import nba_data as nd                       # noqa: E402
from src.DataProviders import feat_squad_quality as fsq            # noqa: E402
from src.DataProviders import feat_squad_continuity as fsc         # noqa: E402
from src.DataProviders import feat_team_elo as fte                 # noqa: E402
from src.Models import baselines as bl                             # noqa: E402

#: The round-3 P1 gate's pass bar for a builder that sums floats.
ROUND3_GATE = 1e-12


# --------------------------------------------------------------------------- #
# Synthetic fixtures: a four-team universe, fully determined, no download
# --------------------------------------------------------------------------- #

#: (GAME_ID, GAME_DATE, home, away). GAME_IDs sort chronologically, like the
#: real NBA ids, so a same-date pair still orders by tip-off sequence.
GAMES_ROWS = [
    ("2001", "2024-10-22", "AAA", "BBB"),
    ("2002", "2024-10-24", "CCC", "DDD"),
    ("2003", "2024-10-26", "BBB", "AAA"),
    ("2004", "2024-10-28", "AAA", "DDD"),   # "tonight" for the membership tests
    ("2005", "2024-10-30", "AAA", "CCC"),   # doubleheader, game 1
    ("2006", "2024-10-30", "DDD", "AAA"),   # doubleheader, game 2 (AAA away)
    ("2007", "2024-11-01", "AAA", "BBB"),   # truncation tail
]

#: Who appeared in each game. Deliberate irregularities around game 2004:
#:   106 debuts in 2004 -- his ONLY appearance row, ever.
#:   104 played 2001 and 2003 but is absent from 2004 (and returns for 2006).
#:   2006's AAA squad is six deep and includes 107, so a roster shift that
#:   grabbed the wrong same-date game changes `sq_n` from 5 to 6 -- the count
#:   itself discriminates the ordering.
SQUADS = {
    "2001": {"AAA": [101, 102, 103, 104], "BBB": [201, 202, 203, 204]},
    "2002": {"CCC": [301, 302, 303, 304], "DDD": [401, 402, 403, 404]},
    "2003": {"BBB": [201, 202, 203, 205], "AAA": [101, 102, 103, 104, 105]},
    "2004": {"AAA": [101, 102, 103, 105, 106], "DDD": [401, 402, 403, 404]},
    "2005": {"AAA": [101, 102, 103, 105], "CCC": [301, 302, 303, 305]},
    "2006": {"DDD": [401, 402, 403, 405], "AAA": [101, 102, 104, 105, 106, 107]},
    "2007": {"AAA": [101, 102, 103], "BBB": [201, 202, 203, 204]},
}


def synthetic_games() -> pd.DataFrame:
    return pd.DataFrame({
        "GAME_ID": [r[0] for r in GAMES_ROWS],
        "GAME_DATE": pd.to_datetime([r[1] for r in GAMES_ROWS]),
        "SEASON": ["2024-25"] * len(GAMES_ROWS),
        "home": [r[2] for r in GAMES_ROWS],
        "away": [r[3] for r in GAMES_ROWS],
        "y": [1] * len(GAMES_ROWS),
        "margin": [10.0] * len(GAMES_ROWS),
    })


def synthetic_player_advanced() -> pd.DataFrame:
    """One box row per appearance; values are functions of (player, game), so
    any value attached to the wrong game is numerically visible."""
    rows = []
    for gid, _date, home, away in GAMES_ROWS:
        gi = int(gid)
        for team in (home, away):
            for pid in SQUADS[gid][team]:
                rows.append({
                    "PLAYER_ID": pid, "GAME_ID": gid, "TEAM_ABBREVIATION": team,
                    "MIN": float(12 + ((pid * 3 + gi * 5) % 20)),
                    "NET_RATING": float(((pid + gi * 7) % 21) - 10),
                    "PIE": 0.05 + ((pid * 2 + gi) % 15) / 100.0,
                    "TS_PCT": 0.40 + ((pid + gi * 3) % 30) / 100.0,
                    "USG_PCT": 0.15 + ((pid * 5 + gi) % 25) / 100.0,
                    "AST_TO": 0.5 + ((pid + gi * 11) % 20) / 10.0,
                })
    df = pd.DataFrame(rows)
    # The leak sentinel: game 2006's box scores are absurd on purpose. If any of
    # them reaches game 2005's features, no tolerance could hide the movement.
    dh = df["GAME_ID"] == "2006"
    df.loc[dh, "MIN"] = 48.0
    df.loc[dh, "NET_RATING"] = 777.0
    df.loc[dh, "PIE"] = 0.99
    df.loc[dh, "TS_PCT"] = 0.99
    df.loc[dh, "USG_PCT"] = 0.99
    df.loc[dh, "AST_TO"] = 99.0
    return df


def build_squads(games: pd.DataFrame, pa: pd.DataFrame
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both squad builders against a synthetic player stream.

    `load_player_advanced` is stubbed at the module the builders actually bind
    (`src.DataProviders.nba_data`), so this is fixture-level: no download, no
    network, and nothing outside the two builders under test.
    """
    with mock.patch.object(fsq.nd, "load_player_advanced", lambda *a, **k: pa.copy()):
        return (fsq.squad_quality_features(games, 15.0),
                fsc.squad_continuity_features(games, 15.0))


def build_elo(games: pd.DataFrame) -> pd.DataFrame:
    """`team_elo_features` with the opponent-adjusted dependency stubbed.

    `elo_edge` is a residual against ANOTHER feature block; that block has its
    own tests and is not this gate's business. The stub supplies a zero
    `home_home_edge` so the Elo columns under test are the ones exercised.
    """
    def _fake_oa(g, halflife=10.0, **kw):
        return pd.DataFrame({"home_home_edge": np.zeros(len(g))})

    with mock.patch.object(fte.nd, "opponent_adjusted_features", _fake_oa):
        return fte.team_elo_features(games, 10.0)


def row_of(feat: pd.DataFrame, games: pd.DataFrame, gid: str) -> pd.Series:
    return feat.loc[games.index[games["GAME_ID"] == gid][0]]


# --------------------------------------------------------------------------- #
# Assertion helpers
# --------------------------------------------------------------------------- #

class FeatureEqualityMixin:
    """Compare two feature builds row by row.

    A leak moves values by whole points; float-summation noise moves them by
    ~1e-14. `tol` separates those, and `tol=0.0` demands bit identity. The NaN
    pattern is ALWAYS compared exactly -- a cell that backfills when the frame
    changes is a leak with no tolerance that could excuse it.
    """

    def assertSameFeatures(self, a, b, tol, what):
        a = a.reset_index(drop=True)
        b = b.reset_index(drop=True)
        self.assertEqual(list(a.columns), list(b.columns),
                         f"{what}: column set differs")
        self.assertEqual(len(a), len(b), f"{what}: row count differs")
        self.assertTrue((a.isna().to_numpy() == b.isna().to_numpy()).all(),
                        f"{what}: NaN pattern changed, which is a leak outright")
        num = a.select_dtypes(include=[np.number]).columns
        x = a[num].to_numpy(dtype=float)
        y = b[num].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        maxdiff = float(np.abs(x[m] - y[m]).max()) if m.any() else 0.0
        if tol == 0.0:
            self.assertEqual(
                maxdiff, 0.0,
                f"{what}: values are not bit-identical (max |diff| = "
                f"{maxdiff:.3e}); this builder is expected to be exact")
        else:
            self.assertLess(
                maxdiff, tol,
                f"{what}: values moved by {maxdiff:.3e} >= {tol}; a real leak "
                f"moves them by whole points, so anything at this bar is "
                f"contamination, not float noise")
        return maxdiff

    def assertFeaturesDiffer(self, a, b, what):
        """The anti-vacuity guard: a perturbation that was supposed to matter."""
        a = a.reset_index(drop=True)
        b = b.reset_index(drop=True)
        self.assertEqual(len(a), len(b), f"{what}: row count differs")
        same_nan = (a.isna().to_numpy() == b.isna().to_numpy()).all()
        num = a.select_dtypes(include=[np.number]).columns
        x = a[num].to_numpy(dtype=float)
        y = b[num].to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        maxdiff = float(np.abs(x[m] - y[m]).max()) if m.any() else 0.0
        self.assertTrue((not same_nan) or maxdiff > 0.0,
                        f"{what}: nothing moved, so the test proves nothing")


# --------------------------------------------------------------------------- #
# Membership causality: the roster is who played BEFORE, not who plays tonight
# --------------------------------------------------------------------------- #

class TestMembershipCausality(FeatureEqualityMixin, unittest.TestCase):
    """Game t's squad is `shift(1)` of the team's own game sequence.

    This is the property `docs/player_elo_leak.md` says truncation cannot see:
    cutting the future does not change who was in the room tonight. So the
    checks perturb tonight's PARTICIPANTS and demand tonight's features not
    move; and they perturb a prior game's participants and demand they do.
    """

    def setUp(self):
        self.games = synthetic_games()
        self.pa = synthetic_player_advanced()

    def test_a_player_debuting_tonight_contributes_nothing_to_tonights_features(self):
        """106's only appearance row, ever, is game 2004 -- tonight.

        His box score cannot enter 2004's aggregates: he is not in the previous
        squad (membership) and he has no strictly-earlier stream row (value
        merge). So removing his row must leave 2004's row untouched for BOTH
        builders -- squad quality (which reads squad_{t-1}) and continuity
        (which reads squad_{t-1} and squad_{t-2}).
        """
        sq, ct = build_squads(self.games, self.pa)
        pa_without = self.pa[self.pa["PLAYER_ID"] != 106]
        sq2, ct2 = build_squads(self.games, pa_without)
        tonight = self.games.index[self.games["GAME_ID"] == "2004"]
        self.assertSameFeatures(sq.loc[tonight], sq2.loc[tonight], ROUND3_GATE,
                                "squad_quality at the debut game")
        self.assertSameFeatures(ct.loc[tonight], ct2.loc[tonight], ROUND3_GATE,
                                "squad_continuity at the debut game")

    def test_a_player_absent_tonight_still_contributes_from_the_prior_game(self):
        """104 played 2001 and 2003 and is absent from tonight (2004).

        The availability convention aggregates over the squad that PLAYED the
        previous game -- a set finished before tip-off -- so 104 counts tonight
        despite dressing nowhere. `home_sq_n` must be 5 (101..105: 104 in, 106
        out) and `home_squad_depth` 4 (101..104 played both prior games), and
        removing 104's history must move both rows -- the perturbation has to
        matter or the invariance tests above prove nothing.
        """
        sq, ct = build_squads(self.games, self.pa)
        t = row_of(sq, self.games, "2004")
        self.assertEqual(t["home_sq_n"], 5.0,
                         "the previous squad is 5 deep -- 104 must be counted "
                         "even though he is absent tonight, and 106 must not be "
                         "counted even though he plays tonight")
        self.assertEqual(row_of(ct, self.games, "2004")["home_squad_depth"], 4.0,
                         "squad_{t-2} is game 2001's squad {101..104}: 104 "
                         "played both prior games and counts despite being "
                         "absent tonight")

        pa_without = self.pa[~((self.pa["PLAYER_ID"] == 104)
                               & self.pa["GAME_ID"].isin(["2001", "2003"]))]
        sq2, ct2 = build_squads(self.games, pa_without)
        tonight = self.games.index[self.games["GAME_ID"] == "2004"]
        self.assertFeaturesDiffer(sq.loc[tonight], sq2.loc[tonight],
                                  "squad_quality: 104's history must matter")
        self.assertFeaturesDiffer(ct.loc[tonight], ct2.loc[tonight],
                                  "squad_continuity: 104's history must matter")

    def test_the_counts_are_the_shift_semantics(self):
        """Hand-computed shift(1)/shift(2) on this fixture, exact.

        AAA's game sequence is 2001, 2003, 2004, 2005, 2006, 2007. So at 2004
        the previous squad is 2003's {101..105} (5) and the one before is
        2001's {101..104} (4 shared -> depth 4, overlap 4/5 = 0.8, turnover
        0.2). At 2005 the previous squad is 2004's {101,102,103,105,106} (5 --
        NOT 2006's six, which would be same-day membership leakage). At 2006 it
        is 2005's four.
        """
        sq, ct = build_squads(self.games, self.pa)
        self.assertEqual(row_of(sq, self.games, "2004")["home_sq_n"], 5.0)
        self.assertEqual(row_of(sq, self.games, "2005")["home_sq_n"], 5.0)
        self.assertEqual(row_of(sq, self.games, "2006")["away_sq_n"], 4.0)
        c = row_of(ct, self.games, "2004")
        self.assertEqual(c["home_squad_depth"], 4.0)
        self.assertAlmostEqual(c["home_squad_overlap_2"], 0.8, places=12)
        self.assertAlmostEqual(c["home_squad_turnover_2"], 0.2, places=12)


# --------------------------------------------------------------------------- #
# Truncation invariance: adding the future must not move the past
# --------------------------------------------------------------------------- #

class TestTruncationInvariance(FeatureEqualityMixin, unittest.TestCase):
    """Drop the tail of the game set; the retained rows must not move.

    The synthetic doubleheader makes the second cut sharper than a plain tail
    drop: cutting 2006+2007 removes the SECOND half of a same-date pair while
    keeping game 1 of it, so any same-date reach shows up here too.
    """

    def setUp(self):
        self.games = synthetic_games()
        self.pa = synthetic_player_advanced()

    def test_squad_quality_retained_rows_match_within_the_round3_gates_1e_12_bar(self):
        """Float-summation noise, measured: 1.4e-17 over these cuts.

        The builder groupby-sums weighted values, and pandas' summation order
        depends on the frame being sorted, so the retained rows are bit-close
        rather than bit-equal. A leak moves them by whole points; the round-3
        P1 gate's own pass bar separates the two.
        """
        for drop in (1, 2):
            full, _ = build_squads(self.games, self.pa)
            trunc, _ = build_squads(self.games.iloc[:-drop], self.pa)
            self.assertSameFeatures(
                full.iloc[:-drop], trunc, ROUND3_GATE,
                f"squad_quality dropping the last {drop} game(s)")

    def test_squad_continuity_retained_rows_match_within_the_round3_gates_1e_12_bar(self):
        """Same bar, same reason: continuity sums too (8.9e-16 on the
        downloaded frame; exact on this fixture, asserted at the bar so the
        name states the contract rather than today's arithmetic luck)."""
        for drop in (1, 2):
            _, full = build_squads(self.games, self.pa)
            _, trunc = build_squads(self.games.iloc[:-drop], self.pa)
            self.assertSameFeatures(
                full.iloc[:-drop], trunc, ROUND3_GATE,
                f"squad_continuity dropping the last {drop} game(s)")

    def test_team_elo_retained_rows_are_bit_identical(self):
        """The Elo pass is a sequential recursion over rows, with no groupby
        sums anywhere: every perturbation here has measured 0.0, so this one is
        held to exact equality rather than a tolerance."""
        for drop in (1, 2):
            full = build_elo(self.games)
            trunc = build_elo(self.games.iloc[:-drop])
            self.assertSameFeatures(
                full.iloc[:-drop], trunc, 0.0,
                f"team_elo dropping the last {drop} game(s)")


class TestTruncationInvarianceOnTheDownloadedFrame(FeatureEqualityMixin,
                                                   unittest.TestCase):
    """The same property on the real frame, where the summation-order noise
    actually shows (~8.5e-14 for squad quality on this cut).

    Data-gated exactly like the rest of the suite: the loaders raise
    `unittest.SkipTest` when the download is absent, and a fresh clone runs
    these as skips while the synthetic tests above still gate the builders.
    The cut is 2,400 games vs 1,800 rather than the full frame only to keep the
    suite affordable; `scripts/round3_verify.py::gate1_truncation` runs the
    whole frame at round time.
    """

    N_FULL, N_TRUNC = 2400, 1800

    @classmethod
    def setUpClass(cls):
        try:
            cls.games = nd.build_games()
        except Exception as exc:
            raise unittest.SkipTest(f"NBA data unavailable: {exc}")

    def test_squad_quality_retained_rows_match_within_the_round3_gates_1e_12_bar(self):
        full = fsq.squad_quality_features(self.games.head(self.N_FULL), 15.0)
        trunc = fsq.squad_quality_features(self.games.head(self.N_TRUNC), 15.0)
        self.assertSameFeatures(
            full.iloc[:self.N_TRUNC], trunc, ROUND3_GATE,
            "squad_quality on the downloaded frame")

    def test_squad_continuity_retained_rows_match_within_the_round3_gates_1e_12_bar(self):
        full = fsc.squad_continuity_features(self.games.head(self.N_FULL), 15.0)
        trunc = fsc.squad_continuity_features(self.games.head(self.N_TRUNC), 15.0)
        self.assertSameFeatures(
            full.iloc[:self.N_TRUNC], trunc, ROUND3_GATE,
            "squad_continuity on the downloaded frame")

    def test_team_elo_retained_rows_are_bit_identical(self):
        full = fte.team_elo_features(self.games.head(self.N_FULL), 10.0)
        trunc = fte.team_elo_features(self.games.head(self.N_TRUNC), 10.0)
        self.assertSameFeatures(
            full.iloc[:self.N_TRUNC], trunc, 0.0,
            "team_elo on the downloaded frame")


# --------------------------------------------------------------------------- #
# Strictly-casual attachment: allow_exact_matches=False is the enforcement
# --------------------------------------------------------------------------- #

class TestStrictlyCasualAttachment(FeatureEqualityMixin, unittest.TestCase):
    """The value merge may never attach a same-day row.

    NBA teams do not play twice in a day, so real data cannot exercise this and
    a fixture must. In the fixture, game 2006 (the second half of the
    doubleheader) carries absurd box scores -- MIN 48, NET_RATING 777 -- so any
    reach backwards into game 2005 is unmistakable.
    """

    def setUp(self):
        self.games = synthetic_games()
        self.pa = synthetic_player_advanced()

    def test_the_value_merge_never_attaches_a_same_day_row(self):
        """Directly on `_asof_attach`: a same-day query gets the strictly
        earlier value or nothing, and the day AFTER gets the same-day value --
        by then it is legitimately history. Both halves matter: the second
        proves the fixture can attach when allowed, so the first is not passing
        on an empty merge."""
        hist = pd.DataFrame({
            "PLAYER_ID": [101, 101],
            "GAME_DATE": pd.to_datetime(["2024-10-28", "2024-10-30"]),
            "exp_min": [20.0, 48.0],
        })
        queries = pd.DataFrame({
            "GAME_ID": ["2101", "2102"],
            "TEAM_ABBREVIATION": ["AAA", "AAA"],
            "GAME_DATE": pd.to_datetime(["2024-10-30", "2024-10-31"]),
            "PLAYER_ID": [101, 101],
        })
        out = fsq._asof_attach(queries, hist, ["exp_min"])
        self.assertEqual(float(out["exp_min"].iloc[0]), 20.0,
                         "a query on 10-30 attached the 10-30 row; "
                         "allow_exact_matches=False is what forbids exactly this")
        self.assertEqual(float(out["exp_min"].iloc[1]), 48.0,
                         "the 10-31 query must still see the 10-30 row")

    def test_game_two_of_a_doubleheader_does_not_reach_game_one(self):
        """Game 2005's features must be identical whether or not game 2006 --
        same date, same AAA squad, absurd box scores -- exists at all.

        `home_sq_n` = 5 pins the roster side too: 2005's previous squad is
        2004's five, never 2006's six, so the roster shift cannot walk
        same-date-forward either. Removing 2006's rows empties its own output
        row, which is the anti-vacuity guard for the sentinel."""
        sq, ct = build_squads(self.games, self.pa)
        pa_without = self.pa[self.pa["GAME_ID"] != "2006"]
        sq2, ct2 = build_squads(self.games, pa_without)
        g1 = self.games.index[self.games["GAME_ID"] == "2005"]
        self.assertSameFeatures(sq.loc[g1], sq2.loc[g1], ROUND3_GATE,
                                "squad_quality at doubleheader game 1")
        self.assertSameFeatures(ct.loc[g1], ct2.loc[g1], ROUND3_GATE,
                                "squad_continuity at doubleheader game 1")

        self.assertEqual(row_of(sq, self.games, "2005")["home_sq_n"], 5.0,
                         "game 1's roster is game 2004's five; 2006's six would "
                         "mean the shift reached a same-date game")
        self.assertEqual(row_of(ct, self.games, "2005")["home_squad_depth"], 4.0,
                         "t-1 is 2004's squad {101,102,103,105,106} and t-2 is "
                         "2003's {101..105}: four shared")
        self.assertTrue(pd.isna(row_of(sq2, self.games, "2006")["away_sq_n"]),
                        "with game 2's rows removed its own features must be "
                        "empty, or the sentinel rows were never in the fixture")


# --------------------------------------------------------------------------- #
# Team Elo: the before-state, delegated to the rating pass
# --------------------------------------------------------------------------- #

class TestTeamEloAttachesTheBeforeState(FeatureEqualityMixin, unittest.TestCase):
    """`team_elo_features` must attach `rating_*_before` and nothing else.

    The arithmetic is NOT re-derived here: it delegates to
    `baselines.elo_rating_pass(same_day="batched")`, whose own contract
    (`Tests/test_baselines.py`, `Tests/test_ratings.py`) pins "predict before
    you update" and "no game on a date sees another game on the same date".
    These tests pin that THIS builder passes that convention through -- the
    wiring, not the arithmetic.
    """

    def setUp(self):
        self.games = synthetic_games()

    @staticmethod
    def _pass(games, same_day="batched"):
        gg = games.rename(columns={"y": "home_win", "GAME_DATE": "date"})
        return bl.elo_rating_pass(gg, k=10.0, home_advantage=fte.HFA,
                                  same_day=same_day, use_margin=True,
                                  season_reversion=0.75).games

    def test_ratings_are_the_pre_update_state_delegated_to_the_rating_pass(self):
        feat = build_elo(self.games)
        passed = self._pass(self.games)
        self.assertTrue(np.array_equal(feat["home_elo"].to_numpy(),
                                       passed["rating_home_before"].to_numpy()),
                        "home_elo must be rating_home_before from the pass")
        self.assertTrue(np.array_equal(feat["away_elo"].to_numpy(),
                                       passed["rating_away_before"].to_numpy()),
                        "away_elo must be rating_away_before from the pass")
        # And not the after-state. The comparison must have teeth: the two
        # differ on this fixture from the very first update onward.
        self.assertFalse(np.array_equal(feat["home_elo"].to_numpy(),
                                        passed["rating_home_after"].to_numpy()),
                         "home_elo matches rating_home_after -- the game's own "
                         "update bled into its own feature")

    def test_a_games_own_result_cannot_reach_its_own_ratings(self):
        """Flip game 2004's outcome and margin. Row 2004's ratings are the
        state BEFORE its own update and must not move at all; later rows must
        move (the test is worthless if the flip does nothing)."""
        base = build_elo(self.games)
        flipped = self.games.copy()
        flipped.loc[3, "y"] = 0
        flipped.loc[3, "margin"] = -10.0
        after = build_elo(flipped)
        own = self.games.index[self.games["GAME_ID"] == "2004"]
        self.assertSameFeatures(base.loc[own], after.loc[own], 0.0,
                                "team_elo at the flipped game")
        later = self.games.index[self.games["GAME_DATE"] > "2024-10-28"]
        self.assertFeaturesDiffer(base.loc[later], after.loc[later],
                                  "the flip must reach LATER games")

    def test_another_teams_result_cannot_move_a_teams_rating(self):
        """Flip 2002 (CCC-DDD). Games after it that involve neither team must
        not move -- a rating is fed by the team's OWN games, never by who
        happened to play nearby. This is the team-level form of the membership
        sin in docs/player_elo_leak.md: no summing over whoever actually
        played."""
        base = build_elo(self.games)
        flipped = self.games.copy()
        flipped.loc[1, "y"] = 0
        flipped.loc[1, "margin"] = -10.0
        after = build_elo(flipped)
        before_and_uninvolved = self.games.index[self.games["GAME_ID"]
                                                 .isin(["2001", "2002", "2003"])]
        self.assertSameFeatures(base.loc[before_and_uninvolved],
                                after.loc[before_and_uninvolved], 0.0,
                                "ratings up to and including the first "
                                "AAA/BBB game after the flipped game")
        involved = self.games.index[self.games["GAME_ID"].isin(["2004", "2005",
                                                                "2006"])]
        self.assertFeaturesDiffer(base.loc[involved], after.loc[involved],
                                  "DDD plays 2004 onwards, so the flip must "
                                  "reach those rows")

    def test_a_same_day_doubleheader_does_not_bleed_into_the_second_game(self):
        """AAA plays 2005 and 2006 on 2024-10-30. Under `same_day="batched"`
        every game on a date is predicted from the state at the end of the
        previous date, so AAA's pre-game rating is the SAME number in both, and
        neither game's result can touch the other's attached ratings.

        The teeth check is the sequential pass: on this fixture it DOES let
        game 2 see game 1, so if the builder ever stopped passing
        `same_day="batched"` through, the equality below would break first.
        """
        base = build_elo(self.games)
        r_g1 = row_of(base, self.games, "2005")   # AAA home
        r_g2 = row_of(base, self.games, "2006")   # AAA away
        self.assertEqual(r_g1["home_elo"], r_g2["away_elo"],
                         "AAA's rating must be the day-start state in both "
                         "games of the doubleheader")

        f1 = self.games.copy()
        f1.loc[4, "y"] = 0
        f1.loc[4, "margin"] = -10.0
        after1 = build_elo(f1)
        g2 = self.games.index[self.games["GAME_ID"] == "2006"]
        self.assertSameFeatures(base.loc[g2], after1.loc[g2], 0.0,
                                "game 2's ratings after flipping game 1")

        f2 = self.games.copy()
        f2.loc[5, "y"] = 0
        f2.loc[5, "margin"] = -10.0
        after2 = build_elo(f2)
        g1 = self.games.index[self.games["GAME_ID"] == "2005"]
        self.assertSameFeatures(base.loc[g1], after2.loc[g1], 0.0,
                                "game 1's ratings after flipping game 2")
        self.assertSameFeatures(base.loc[g2], after2.loc[g2], 0.0,
                                "game 2's own ratings after flipping itself")

        seq = self._pass(self.games, same_day="sequential")
        self.assertNotEqual(float(seq["rating_away_before"].iloc[5]),
                            float(r_g2["away_elo"]),
                            "under same_day='sequential' game 2 WOULD see game "
                            "1; if this holds the batched convention is doing "
                            "real work in the assertion above")

    def test_a_debuting_teams_first_game_carries_only_the_initial_rating(self):
        """Both teams debut in game 2001: neither has any history, so both
        carry the initial rating (0.0) and nothing of their own game."""
        base = build_elo(self.games)
        r = row_of(base, self.games, "2001")
        self.assertEqual(r["home_elo"], 0.0)
        self.assertEqual(r["away_elo"], 0.0)


if __name__ == "__main__":
    unittest.main()
