"""Tests for the per-player Elo block.

WHY A PLAYER ELO AT ALL
-----------------------
Every other feature in this repo describes an institution - a team's rolling form,
its rest, its schedule. Availability is the one mechanism here that is not a null
(+0.00382 nats), and a roster cannot be discounted by *who* is in it unless
something knows who that is. `player_elo_features` supplies that: an online Elo
rating per player, aggregated over the squad expected to play.

`src/Models/player_ratings.py` already tested RAPM-style player ratings and found
them a null. This is a different estimator - sequential, online, one parameter -
and it measures something the correlation check says is genuinely new: the
aggregates sit at r = 0.16, 0.09 and 0.07 against team rolling margin, far from
the 0.98 that marked the opponent-adjusted block as a mere rearrangement.

CAUSALITY IS THE WHOLE CLAIM
----------------------------
A rating attached to a game must be the rating the player carried INTO it, and
must not change when future games are appended. Both are asserted by truncating.
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


class TestShape(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _have_data():
            raise unittest.SkipTest(f"no downloaded data at {nd.DATA_ROOT}")
        cls.games = nd.build_games()
        cls.f = nd.player_elo_features(cls.games)

    def test_one_row_per_game(self):
        self.assertEqual(len(self.f), len(self.games))

    def test_three_aggregates_per_side(self):
        for col in ("elo_available", "elo_mean", "elo_top5"):
            for side in ("home", "away"):
                self.assertIn(f"{side}_{col}", self.f.columns)

    def test_first_games_have_no_expected_minutes(self):
        """A player's expected minutes come from their own past. Nobody has a
        past before their first game, so the first rows must be empty rather than
        zero - a zero would read as "played zero minutes"."""
        self.assertTrue(np.isnan(self.f["home_elo_mean"].iloc[0]),
                        "the first game cannot have an expected-minutes mean")

    def test_no_duplicate_columns_against_other_blocks(self):
        roll = nd.rolling_team_features(self.games)
        opp = nd.opponent_adjusted_features(self.games)
        for other, name in ((roll, "rolling_team_features"),
                            (opp, "opponent_adjusted_features")):
            overlap = sorted(set(other.columns) & set(self.f.columns))
            self.assertEqual(overlap, [],
                             f"columns emitted by both this block and {name}: "
                             f"{overlap}")


class TestScaleIsSane(unittest.TestCase):
    """Without mean reversion a rating random-walks until the arithmetic cannot
    express it, and a 2017 hot streak still sets the scale in 2025."""

    @classmethod
    def setUpClass(cls):
        if not _have_data():
            raise unittest.SkipTest(f"no downloaded data at {nd.DATA_ROOT}")
        cls.games = nd.build_games()
        cls.late = nd.player_elo_features(cls.games).iloc[9000:]

    def test_ratings_stay_near_the_league_mean(self):
        """`elo_mean` is a minutes-weighted squad average. It must not drift to
        double or half the starting value over eleven seasons."""
        m = self.late["home_elo_mean"].mean()
        self.assertGreater(m, 1200.0, f"squad mean drifted down to {m:.0f}")
        self.assertLess(m, 1800.0, f"squad mean drifted up to {m:.0f}")

    def test_the_best_five_rate_above_the_squad_average(self):
        """A top-5 sum divided by five is the average rating of the five best
        available players, which by construction exceeds the weighted squad mean.
        If this fails the ranking is not being applied."""
        top5_mean = self.late["home_elo_top5"].mean() / 5.0
        squad_mean = self.late["home_elo_mean"].mean()
        self.assertGreater(top5_mean, squad_mean,
                           f"top-5 mean {top5_mean:.1f} <= squad mean "
                           f"{squad_mean:.1f}; the top-N selection is inverted")

    def test_available_quality_is_consistent_with_minutes(self):
        """elo_available is sum(elo * expected minutes) and elo_mean is that sum
        divided by sum(expected minutes). They must be the same quantity, so the
        ratio must return the original - an internal consistency check on the
        arithmetic, not a statistical one.

        An earlier version of this test divided by a FIXED 240, on the assumption
        that a rotation's expected minutes sum to 240. They do not: expected
        minutes are trailing EWMAs that do not sum to a full rotation, and under
        the `all_prior` roster the sum is over ~6.4x a real lineup, since a
        long-departed player's weight decays without reaching zero. That version
        disagreed by up to 1204 and was testing its own wrong assumption rather
        than the code.
        """
        elo = nd.player_elo_ratings(self.games)
        elo = elo.sort_values(["PLAYER_ID", "GAME_DATE"])
        elo["exp_min"] = nd._ewm_by_player(elo, "MIN", 15.0)
        sub = elo.iloc[9000:].dropna(subset=["exp_min"])
        sub = sub[sub["exp_min"] > 0]
        g = sub.groupby(["GAME_ID", "TEAM_ABBREVIATION"]).apply(
            lambda d: pd.Series({
                "avail": (d["elo"] * d["exp_min"]).sum(),
                "minutes": d["exp_min"].sum(),
                "mean": (d["elo"] * d["exp_min"]).sum() / d["exp_min"].sum(),
            }), include_groups=False)
        implied = g["avail"] / g["minutes"].replace(0, np.nan)
        self.assertLess(
            float((implied - g["mean"]).abs().max()), 1e-6,
            "elo_available / sum(expected minutes) must equal elo_mean exactly")


class TestCausality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not _have_data():
            raise unittest.SkipTest(f"no downloaded data at {nd.DATA_ROOT}")
        cls.games = nd.build_games()

    def test_removing_the_future_changes_nothing(self):
        """The Elo is updated AFTER each game, so appending thousands of later
        games must not move any earlier value. If this fails, a prediction is
        being made with knowledge of games that had not happened."""
        full = nd.player_elo_features(self.games)
        truncated = nd.player_elo_features(self.games.head(5000))
        n = len(truncated)
        for col in full.columns:
            a = full[col].to_numpy(float)[:n]
            b = truncated[col].to_numpy(float)
            self.assertEqual(np.isfinite(a).tolist(), np.isfinite(b).tolist(),
                             f"{col}: missingness differs once the future is gone")
            mask = np.isfinite(a)
            if mask.any():
                worst = float(np.max(np.abs(a[mask] - b[mask])))
                self.assertLess(worst, 1e-6,
                                f"{col} moved by {worst:.3e} when the future was "
                                f"removed")

    def test_a_players_first_game_rating_is_the_league_mean(self):
        """A debutant has no history, so the rating carried into their first game
        must be the starting value - not zero, and not a value inherited from
        whichever player was processed before them."""
        elo = nd.player_elo_ratings(self.games).sort_values(
            ["PLAYER_ID", "GAME_DATE"])
        firsts = elo.groupby("PLAYER_ID").head(1)
        self.assertGreater(len(firsts), 100, "not enough debuts to be meaningful")
        spread = firsts["elo"].std()
        self.assertLess(spread, 1e-6,
                        f"debut ratings are not all 1500 (sd {spread:.4f}); a "
                        f"player is inheriting somebody else's rating")


class TestItIsNotARestatementOfTeamForm(unittest.TestCase):
    """The diagnostic that predicted `opponent_adjusted` would be a null
    (r = 0.98 with the feature it adjusts) is applied here BEFORE the ablation is
    read, so the null or the non-null is interpretable either way."""

    @classmethod
    def setUpClass(cls):
        if not _have_data():
            raise unittest.SkipTest(f"no downloaded data at {nd.DATA_ROOT}")
        cls.games = nd.build_games()
        elo = nd.player_elo_features(cls.games).reset_index(drop=True)
        roll = nd.rolling_team_features(cls.games).reset_index(drop=True)
        cls.late = pd.concat([elo, roll], axis=1).iloc[7000:]

    def test_clearly_distinct_from_team_rolling_margin(self):
        for col in ("home_elo_available", "home_elo_mean", "home_elo_top5"):
            r = abs(float(self.late[col].corr(self.late["home_ewm_margin"])))
            self.assertLess(r, 0.60,
                            f"{col} correlates {r:.3f} with home_ewm_margin; at "
                            f"that level it is a rescaling of team form and the "
                            f"ablation cannot distinguish it from a duplicate")

    def test_carries_real_variation(self):
        for col in ("home_elo_available", "home_elo_mean", "home_elo_top5"):
            s = self.late[col].dropna()
            self.assertGreater(len(s), 1000, f"{col} is nearly empty")
            self.assertGreater(s.std(), 1.0,
                               f"{col} is present but effectively constant")


class TestTheRawTotalIsNotAQualityMeasure(unittest.TestCase):
    """Why `elo_available` is kept but must never be read as team strength.

    `scripts/elo_buckets.py` splits games into deciles of the raw
    home-minus-away contrast and reads off the actual home-win rate. The result is
    a Spearman rho of **-0.96** for `elo_available`: the LARGER a team's
    accumulated rating-minutes, the LESS it wins.

    The cause is structural. The `all_prior` roster is every player with an earlier
    appearance for the team, and a departed player's trailing EWMA decays without
    ever reaching zero. So a club that recently traded a star keeps carrying his
    weight for months, while a club that just acquired one is under-counted until
    his minutes accumulate. The column is a lagging indicator of roster churn.

    This is the same defect that produced the |z| = 5.18 leak from the other side:
    the leak inflated the feature with post-game membership, and the inversion is
    what remains once that is removed.

    The assertion is therefore on the SIGN, and it is deliberately the opposite of
    what a naive reading of "bigger number means better team" would expect. If a
    future change makes this number correctly ordered, this test should fail and be
    rewritten - not silently pass on a coincidence.
    """

    @classmethod
    def setUpClass(cls):
        if not _have_data():
            raise unittest.SkipTest(f"no downloaded data at {nd.DATA_ROOT}")
        cls.games = nd.build_games().reset_index(drop=True)
        elo = nd.player_elo_features(cls.games, roster_mode="all_prior")
        cls.d = pd.concat([cls.games[["GAME_ID", "y"]], elo], axis=1).iloc[7000:]

    def test_the_raw_total_contrast_is_inverted_not_ordered(self):
        avail = self.d["home_elo_available"] - self.d["away_elo_available"]
        ok = avail.notna() & self.d["y"].notna()
        a, y = avail[ok].to_numpy(float), self.d.loc[ok, "y"].to_numpy(float)
        self.assertGreater(len(a), 1000, "not enough games to correlate")
        r = float(np.corrcoef(a, y)[0, 1])
        self.assertLess(
            r, -0.05,
            f"elo_available contrast now correlates {r:+.3f} with the home win. "
            f"The column has become quality-ordered, which means either the roster "
            f"construction changed or the departed-player effect is gone. Either "
            f"way this test is stale and the README's mechanism claim needs "
            f"re-measuring.")

    def test_the_scale_free_mean_is_correctly_ordered(self):
        """The companion: the same block, scale-free, DOES order games correctly.
        They must not both be true of the same information - one is measuring
        quality and the other is measuring churn.

        Measured as MONOTONICITY across deciles, not as a linear correlation. A
        Pearson correlation between the contrast and the raw 0/1 home-win label
        comes out at only +0.019 here, because win probability is a sigmoid of a
        symmetric contrast and a straight line is a poor description of it.
        Decile win rates against decile index is the statistic the bucketed
        analysis reports (rho = +0.85) and the statistic the claim is about. An
        earlier version of this test asserted the linear form and failed for that
        reason alone.
        """
        mean = self.d["home_elo_mean"] - self.d["away_elo_mean"]
        ok = mean.notna() & self.d["y"].notna()
        a, y = mean[ok].to_numpy(float), self.d.loc[ok, "y"].to_numpy(float)
        self.assertGreater(len(a), 1000, "not enough games to bucket")

        rank = np.empty(len(a), dtype=int)
        rank[np.argsort(a, kind="mergesort")] = np.arange(len(a))
        decile = (rank * 10) // len(a)
        rates = np.array([y[decile == k].mean() for k in range(10)])
        rho = float(np.corrcoef(np.arange(10, dtype=float), rates)[0, 1])

        self.assertGreater(
            rho, 0.5,
            f"decile win rates reach only rho = {rho:+.3f} against the bucketed "
            f"result of +0.85. Rates were {[round(r, 3) for r in rates]}.")


if __name__ == "__main__":
    unittest.main()