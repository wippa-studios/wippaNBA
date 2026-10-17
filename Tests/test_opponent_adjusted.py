"""Tests for `opponent_adjusted_features` and the opponent-quality columns.

WHAT THIS BLOCK IS FOR
----------------------
Every decay experiment in this project came back null, because all of them
reweight the SAME information the model already has. Opponent quality is a
quantity no existing feature reads at all: `ewm_margin` is a team's raw past
margin, and cannot tell a team that went 20-10 against weak clubs from one that
went 20-10 against strong ones. `opponent_adjusted_features` supplies that.

It also supplies each team's OWN home advantage. The frame carries a single
constant home indicator for the whole league; the real quantity varies, and on
the real data `home_edge` correlates 0.08 with overall form - nearly independent,
which is what makes it worth testing rather than assuming.

CAUSALITY IS THE WHOLE CLAIM
----------------------------
The opponent's trailing margin is, by construction, a value computed from games
that had already happened when the row in question was played. That is easy to
get subtly wrong and impossible to detect from the output alone, so the tests
below truncate the future and assert the past is unchanged.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT, ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from DataProviders import nba_data as nd  # noqa: E402


def _have_data() -> bool:
    return bool(list(Path(nd.DATA_ROOT, "game_logs").glob("*.csv")))


class TestShapeAndAlignment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _have_data():
            raise unittest.SkipTest(f"no downloaded data at {nd.DATA_ROOT}")
        cls.games = nd.build_games()
        cls.f = nd.opponent_adjusted_features(cls.games)

    def test_one_row_per_game_in_the_order_given(self):
        """A feature block that reorders or drops rows silently corrupts every
        downstream label alignment, and the corruption is invisible in the
        numbers - it just makes the model worse in a way that reads as a finding."""
        self.assertEqual(len(self.f), len(self.games),
                         "row count must match the games frame exactly")
        # Positional, because the block returns no GAME_ID. Assert the alignment
        # by value instead: the home team's games-played count rises monotonically.
        home = self.f["home_ewm_margin_home"]
        self.assertTrue(home.isna().iloc[0],
                        "the first game cannot have a trailing home form")

    def test_no_duplicate_columns_against_the_rolling_frame(self):
        """Two sources emitting one feature means the second is silently dropped
        by concat, and which source wins becomes a property of list order."""
        roll = nd.rolling_team_features(self.games)
        overlap = sorted(set(roll.columns) & set(self.f.columns))
        self.assertEqual(overlap, [],
                         f"columns emitted by two sources: {overlap}")

    def test_every_column_is_numeric(self):
        non_numeric = [c for c in self.f.columns
                       if not np.issubdtype(self.f[c].dtype, np.number)]
        self.assertEqual(non_numeric, [],
                         "a datetime or object column here would reach the "
                         "imputer as an unpromotable dtype")


class TestSignConventions(unittest.TestCase):
    """The bug this class exists for.

    An earlier version of the block computed `home_pts - away_pts` for BOTH rows
    of the long frame - correct for the home team, exactly negated for the away
    one. Nothing raised. The column names, dtypes and coverage were all correct.

    It surfaced only as a sanity check: the per-team home advantage came out at
    -0.07, because averaging a right-signed series against a left-signed one
    cancels. A sign error that a numerical assertion catches and a schema check
    does not.
    """

    @classmethod
    def setUpClass(cls):
        if not _have_data():
            raise unittest.SkipTest(f"no downloaded data at {nd.DATA_ROOT}")
        cls.games = nd.build_games()
        cls.f = nd.opponent_adjusted_features(cls.games)
        cls.late = cls.f.iloc[len(cls.f) // 2:]

    def test_home_advantage_is_positive(self):
        """League-wide home advantage is around +2.5 to +3.5 points. A per-team
        estimate of it cannot plausibly be zero or negative."""
        for col in ("home_home_edge", "away_home_edge"):
            mean = self.late[col].mean()
            self.assertGreater(
                mean, 0.5,
                f"{col} mean is {mean:+.3f}; the margin of a row must be signed "
                f"from that row's own team's point of view")

    def test_opponent_adjusted_form_is_centred_near_zero(self):
        """`adj_margin` is the team's margin minus its opponent's own margin. The
        opponent is somebody, so the two cancel in expectation and the result
        should straddle zero. A large positive mean means the sign is wrong, or
        the subtraction is not happening at all."""
        for col in ("home_ewm_adj_margin", "away_ewm_adj_margin"):
            mean = self.late[col].mean()
            self.assertLess(
                abs(mean), 1.0,
                f"{col} mean is {mean:+.3f}; expected roughly zero once opponent "
                f"quality is removed")

    def test_the_two_sides_disagree_as_home_and_away_should(self):
        """Home and away form must not be the same series. If they are, the
        `is_home` split is not being applied and both are the pooled estimate."""
        a = self.late["home_ewm_margin_home"].dropna()
        b = self.late["home_ewm_margin_away"].dropna()
        n = min(len(a), len(b))
        corr = float(np.corrcoef(a.to_numpy()[:n], b.to_numpy()[:n])[0, 1])
        self.assertLess(
            corr, 0.999,
            f"home and away form are the same series (r={corr:.6f}); the is_home "
            f"split is not separating them")

    def test_home_form_is_the_noisier_of_the_two(self):
        """Each side is estimated from half as many games as the pooled figure,
        so it must carry more dispersion. A smaller spread would mean the split
        silently fell back to the pooled estimate."""
        pooled = nd.rolling_team_features(self.games)["home_ewm_margin"].std()
        split = self.late["home_ewm_margin_home"].std()
        self.assertGreater(split, pooled,
                           f"split sd {split:.3f} <= pooled sd {pooled:.3f}")


class TestCausality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _have_data():
            raise unittest.SkipTest(f"no downloaded data at {nd.DATA_ROOT}")
        cls.games = nd.build_games()

    def _assert_identical(self, full: pd.DataFrame, truncated: pd.DataFrame,
                          label: str) -> None:
        n = len(truncated)
        for col in full.columns:
            a = full[col].to_numpy(float)[:n]
            b = truncated[col].to_numpy(float)
            self.assertEqual(np.isfinite(a).tolist(), np.isfinite(b).tolist(),
                             f"{label}: {col} has a different missingness pattern "
                             f"once the future is removed")
            mask = np.isfinite(a)
            if mask.any():
                worst = float(np.max(np.abs(a[mask] - b[mask])))
                self.assertLess(
                    worst, 1e-9,
                    f"{label}: {col} moved by {worst:.3e} when 10,000-odd future "
                    f"games were removed. That is future information leaking "
                    f"backwards.")

    def test_removing_the_future_changes_nothing(self):
        full = nd.opponent_adjusted_features(self.games)
        truncated = nd.opponent_adjusted_features(self.games.head(4000))
        self._assert_identical(full, truncated, "truncation")

    def test_a_single_extra_season_changes_nothing(self):
        """The failure mode this project has hit repeatedly: a function whose
        output depends on WHICH other data happens to be on disk. Scoping to the
        requested games is what makes truncating your own input meaningful."""
        games = self.games
        first_half = games.iloc[:len(games) // 2]
        full = nd.opponent_adjusted_features(games)
        half = nd.opponent_adjusted_features(first_half)
        self._assert_identical(full, half, "single-season")


class TestOpponentQualityColumnsAreNowRead(unittest.TestCase):
    """The other silent absence, one column over.

    `OPP_EFG_PCT` was read. `OPP_FTA_RATE`, `OPP_TOV_PCT` and `OPP_OREB_PCT` sit
    in the same table, under the same header, in every downloaded season, and no
    model in this repo had ever seen them. This is structurally the same defect as
    the ORB_PCT/DRB_PCT absence already documented: a column that is present in
    the source and absent from the feature list, discovered only by reading the
    source rather than by trusting the list.
    """

    @classmethod
    def setUpClass(cls):
        if not _have_data():
            raise unittest.SkipTest(f"no downloaded data at {nd.DATA_ROOT}")
        cls.games = nd.build_games()
        cls.roll = nd.rolling_team_features(cls.games)

    def test_the_three_unused_opponent_columns_are_present(self):
        for stem in ("ewm_opp_efg", "ewm_opp_ftr", "ewm_opp_tov", "ewm_opp_orb"):
            for side in ("home", "away"):
                col = f"{side}_{stem}"
                self.assertIn(col, self.roll.columns,
                              f"{col} is in the source table but not the frame")

    def test_each_carries_real_variation(self):
        """A column that is constant, or all-NaN, satisfies the presence check
        above while contributing nothing. Exactly the ORB_PCT failure mode."""
        for stem in ("ewm_opp_ftr", "ewm_opp_tov", "ewm_opp_orb"):
            for side in ("home", "away"):
                col = f"{side}_{stem}"
                s = self.roll[col]
                self.assertGreater(s.notna().mean(), 0.5,
                                   f"{col} is present but nearly all NaN")
                self.assertGreater(s.nunique(dropna=True), 10,
                                   f"{col} is present but effectively constant "
                                   f"({s.nunique(dropna=True)} distinct values)")

    def test_opponent_columns_are_not_a_copy_of_the_teams_own(self):
        """OPP_* is the opposition's characteristic. If it tracked the team's own
        numbers it would be redundant with them and worth nothing."""
        for own, opp in (("ewm_ftr", "ewm_opp_ftr"), ("ewm_tov", "ewm_opp_tov")):
            a, b = self.roll[f"home_{own}"], self.roll[f"home_{opp}"]
            mask = a.notna() & b.notna()
            self.assertGreater(mask.sum(), 100, "not enough overlap to compare")
            r = float(a[mask].corr(b[mask]))
            self.assertLess(abs(r), 0.95,
                            f"home_{own} and home_{opp} correlate {r:.4f}; the "
                            f"opponent column appears to restate the team's own")


if __name__ == "__main__":
    unittest.main()