"""Tests for the recency-weighted (EWMA) feature layer.

Two things are being pinned down here, and they are not the same thing:

* the ARITHMETIC -- ``ewma``'s weight convention, the strictly-before alignment
  of every statistic, and the possession estimate, each against a value computed
  by hand;
* the LEAKAGE GATE -- that a feature which folds in the game it is supposed to
  predict is refused loudly rather than written out and scored. That is the bug
  behind this project's original 68.9% headline, so the refusal is the test that
  matters most.

Everything runs offline. The snapshot machinery is exercised against synthetic
tables built in memory; ``TeamData.sqlite`` and the odds database are only ever
opened read-only, and are not required to exist for this file to pass.
"""

import importlib.util
import sqlite3
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_PATH = (Path(__file__).resolve().parents[1] / "src" / "Process-Data"
               / "Create_Recency_Features.py")
SPEC = importlib.util.spec_from_file_location("create_recency_features", MODULE_PATH)
recency = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recency)

# The metric set the small tests use, so they stay fast and so the hand-computed
# expectations stay checkable by eye. The production default is GAME_METRICS.
TEST_METRICS = ("PTS", "PLUS_MINUS", "FGA", "FTA", "OREB")


def game_lines(dates, home_values, away_values, season="2024-25",
               home="Lakers", away="Celtics"):
    """Two rows per date: a home team and an away team with known stat lines."""
    rows = []
    for date_value, home_line, away_line in zip(dates, home_values, away_values):
        for is_home, team, opponent, line in (
            (True, home, away, home_line),
            (False, away, home, away_line),
        ):
            row = {
                recency.GAME_ID_COLUMN: recency.game_id(date_value, home, away),
                recency.DATE_COLUMN: pd.Timestamp(date_value),
                recency.TEAM_COLUMN: team,
                recency.OPPONENT_COLUMN: opponent,
                recency.SEASON_COLUMN: season,
                recency.IS_HOME_COLUMN: is_home,
            }
            row.update(line)
            rows.append(row)
    return pd.DataFrame(rows)


def line(points=110.0, plus_minus=2.0, fga=90.0, fta=22.0, orb=11.0, wins=1.0):
    """One plausible box-score line, with every test metric present."""
    return {
        "PTS": points,
        "PLUS_MINUS": plus_minus,
        "FGM": points / 2 - 0.5,
        "FGA": fga,
        "FG3M": 12.0,
        "FG3A": 36.0,
        "FTM": points - 2 * (points / 2 - 0.5) - 12.0,
        "FTA": fta,
        "REB": 45.0,
        "OREB": orb,
        "DREB": 34.0,
        "AST": 25.0,
        "TOV": 14.0,
        "STL": 8.0,
        "BLK": 5.0,
        "PF": 19.0,
        "W": wins,
        "TS_PCT": points / (2.0 * (fga + 0.44 * fta)),
        "EFG_PCT": 0.52,
        "AST_TOV": 25.0 / 14.0,
        "FTR": fta / fga,
        "TPA_RATIO": 0.4,
    }


def snapshot(team_totals, teams):
    """A snapshot table: cumulative TOTALS in, the endpoint's per-game AVERAGES out.

    Mirrors ``load_team_snapshots``: GP plus every stat as an average over the
    season so far, which is the only shape ``leaguedashteamstats`` returns -- and a
    real table carries the whole box score, so this emits every
    ``GAME_COUNT_COLUMNS`` entry, taking whatever the fixture supplied and zero
    for the rest. Zero is safe here precisely because the column is CUMULATIVE: a
    stat that is zero in both bounding snapshots differences back to zero, and no
    test here reads one.

    Every team in ``teams`` gets a row, INCLUDING one that has not played yet --
    ``GP = 0`` with zeroed stats, which ``extract_game_logs`` drops on its
    ``played < 1`` check. That keeps "present but has not played" distinct from
    "the snapshot does not mention this team at all", which is its other
    rejection reason.

    Indexed by team name, ``drop=False``, because that is what
    ``load_team_snapshots`` hands ``extract_game_logs`` and the extractor looks
    rows up by name rather than by position.
    """
    columns = list(dict.fromkeys(
        list(recency.GAME_COUNT_COLUMNS)
        + [column for totals in team_totals.values() for column in totals]
    ))
    rows = []
    for team in teams:
        totals = team_totals[team]
        played = totals.get("GP", 0)
        row = {"TEAM_NAME": team, "GP": played}
        for column in columns:
            if column == "GP":
                continue
            value = totals.get(column, 0.0)
            row[column] = value / played if played else 0.0
        rows.append(row)
    return pd.DataFrame(rows).set_index("TEAM_NAME", drop=False)


class TestEwmaArithmetic(unittest.TestCase):
    """The weight convention, against arithmetic done by hand."""

    def test_ewma_matches_hand_computed_value_exactly(self):
        """Series [5, 15, 40, 60] (oldest first), halflife 1 game.

        Weights count backwards from the END: the newest observation (60) gets
        weight 1, the one before it (40) gets 2**(-1/1) = 1/2, then 15 gets 2**(-2)
        = 1/4, and 5 gets 2**(-3) = 1/8.

            numerator   = 60*1 + 40*(1/2) + 15*(1/4) + 5*(1/8)
                        = 60 + 20 + 3.75 + 0.625            = 84.375
            denominator =  1  + 1/2   + 1/4    + 1/8       =  1.875
            ewma        = 84.375 / 1.875                    = 45

        Every term is a dyadic rational, so both 84.375 and 1.875 are exact in
        binary floating point and 45 is exact too -- this compares equal, not
        merely close.
        """
        value = recency.ewma([5.0, 15.0, 40.0, 60.0], halflife=1)
        self.assertEqual(value, 45.0)

    def test_ewma_weight_convention_is_two_to_the_minus_i(self):
        weights = recency.ewma_weights(4, halflife=10)
        # Compared with a relative tolerance: the exponentiation differs in the
        # last bit between CPython 3.10 and 3.11, and an exact float equality
        # would fail on one interpreter for no substantive reason.
        for got, want in zip(weights, [2.0 ** (-3 / 10), 2.0 ** (-2 / 10),
                                       2.0 ** (-1 / 10), 1.0]):
            self.assertAlmostEqual(got, want, delta=abs(want) * 1e-12)
        self.assertEqual(recency.ewma_weights(1, halflife=10)[0], 1.0)

    def test_ewma_halflife_ten_ignores_the_old_half_of_a_short_series(self):
        """Series [1, 2, 3, 4], halflife 10 games.

            weights = 2**(-3/10)=0.8122523964  2**(-2/10)=0.8705505633
                     2**(-1/10)=0.9330329915  2**(0)    =1
            numerator   = 4*1 + 3*0.9330329915 + 2*0.8705505633 + 1*0.8122523964
                        = 9.3524524975
            denominator = 3.6158359512
            ewma        = 2.5866230...
        """
        value = recency.ewma([1.0, 2.0, 3.0, 4.0], halflife=10)
        self.assertAlmostEqual(value, 9.3524524975 / 3.6158359512, places=10)
        # The point of a ten-game half-life on four games: barely any decay yet.
        self.assertAlmostEqual(value, 3.2, delta=0.7)

    def test_ewma_halflife_one_weights_the_newest_game_by_a_quarter_over_two_back(self):
        # Newest carries 1, one back carries 1/2, two back carries 1/4.
        value = recency.ewma([0.0, 0.0, 1.0], halflife=1)
        self.assertAlmostEqual(value, (1 * 1 + 0 * 0.5 + 0 * 0.25) / 1.75, places=12)

    def test_ewma_empty_and_all_nan_are_nan(self):
        self.assertTrue(np.isnan(recency.ewma([])))
        self.assertTrue(np.isnan(recency.ewma([np.nan, np.nan, np.nan])))
        self.assertTrue(np.isnan(recency.ewma([5.0], min_periods=2)))

    def test_ewma_drops_nan_and_reweights_the_rest(self):
        """A gap costs that game's influence instead of poisoning the average.

        [1, NaN, 4] with halflife 1: the two usable observations are 4 (weight 1)
        and 1 (weight 1/4), giving (4 + 0.25) / 1.25 = 3.4.
        """
        self.assertAlmostEqual(
            recency.ewma([1.0, np.nan, 4.0], halflife=1), 3.4, places=12
        )

    def test_ewma_rejects_a_nonsense_halflife(self):
        for bad in (0, -1, float("nan"), float("inf"), None):
            with self.assertRaises(ValueError):
                recency.ewma([1.0, 2.0], halflife=bad)

    def test_default_halflife_is_ten_games_and_is_a_parameter(self):
        self.assertEqual(recency.DEFAULT_HALFLIFE_GAMES, 10.0)
        self.assertEqual(recency.DEFAULT_WINDOWS, (5, 10, 20))
        self.assertNotEqual(
            recency.ewma([1.0, 5.0, 9.0], halflife=2),
            recency.ewma([1.0, 5.0, 9.0], halflife=10),
        )


class TestEwmaHistoryCausality(unittest.TestCase):
    """The strictly-before alignment, which is the anti-leakage argument."""

    def test_row_i_never_reads_row_i(self):
        values = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        history = recency.ewma_history(values, halflife=2)
        self.assertTrue(np.isnan(history[0, 0]))
        for i in range(1, len(values)):
            self.assertAlmostEqual(
                history[i, 0], recency.ewma(values[:i, 0], halflife=2), places=12
            )

    def test_perturbing_row_i_cannot_move_row_i(self):
        values = np.array([[1.0], [2.0], [3.0], [4.0]])
        history = recency.ewma_history(values, halflife=3)
        bumped = values.copy()
        bumped[2, 0] = 10000.0
        after = recency.ewma_history(bumped, halflife=3)
        self.assertEqual(history[2, 0], after[2, 0])   # row 2 unchanged
        self.assertNotEqual(history[3, 0], after[3, 0])  # row 3 did move

    def test_rollup_of_an_inclusive_endpoint_would_look_different(self):
        """What an inclusive endpoint does to the same series, for contrast.

        Including the game's own value in its features is not a subtle
        difference -- it moves the number, which is why the as-of assertion in
        ``build_recency_dataset`` has something to catch.
        """
        values = np.array([[100.0], [100.0], [140.0]])
        exclusive = recency.ewma_history(values, halflife=10)[2, 0]
        inclusive = recency.ewma(values.ravel(), halflife=10)
        self.assertAlmostEqual(exclusive, 100.0, places=12)
        self.assertGreater(inclusive, 110.0)

    def test_history_before_is_strict(self):
        frame = pd.DataFrame({
            recency.DATE_COLUMN: pd.to_datetime(["2024-01-01", "2024-01-02",
                                                 "2024-01-03"]),
        })
        cut = pd.Timestamp("2024-01-02")
        self.assertEqual(
            list(recency.history_before(frame, cut)[recency.DATE_COLUMN]),
            [pd.Timestamp("2024-01-01")],
        )

    def test_assert_strictly_before_rejects_an_as_of_on_the_game_day(self):
        with self.assertRaises(recency.LeakageError):
            recency.assert_strictly_before(["2024-01-02"], pd.Timestamp("2024-01-02"),
                                           "test")
        self.assertTrue(recency.assert_strictly_before(
            [pd.NaT, "2024-01-01"], pd.Timestamp("2024-01-02"), "test"))


class TestRollingFeatures(unittest.TestCase):

    def build(self, **kwargs):
        dates = [date(2024, 1, d) for d in (5, 7, 9, 12, 15)]
        home_points = [100.0, 110.0, 120.0, 130.0, 140.0]
        away_points = [95.0, 105.0, 115.0, 125.0, 135.0]
        rows = game_lines(
            dates,
            home_values=[line(points=p) for p in home_points],
            away_values=[line(points=p) for p in away_points],
            **kwargs,
        )
        return recency.rolling_features(rows, halflife=2, windows=(2, 3),
                                        metrics=TEST_METRICS)

    def test_features_use_only_prior_games(self):
        built = self.build()
        home = built[built[recency.IS_HOME_COLUMN]].sort_values(recency.DATE_COLUMN)
        points = [100.0, 110.0, 120.0, 130.0, 140.0]
        for position in range(1, len(home)):
            self.assertAlmostEqual(
                home["PTS_ewma"].iloc[position],
                recency.ewma(points[:position], halflife=2),
                places=12,
            )

    def test_as_of_is_the_previous_game_and_games_counts_history(self):
        built = self.build()
        home = built[built[recency.IS_HOME_COLUMN]].sort_values(recency.DATE_COLUMN)
        self.assertTrue(pd.isna(home[recency.AS_OF_COLUMN].iloc[0]))
        self.assertEqual(home[recency.GAMES_COLUMN].iloc[0], 0)
        for position in range(1, len(home)):
            self.assertEqual(
                home[recency.AS_OF_COLUMN].iloc[position],
                home[recency.DATE_COLUMN].iloc[position - 1],
            )
            self.assertEqual(home[recency.GAMES_COLUMN].iloc[position], position)

    def test_rolling_statistics_stay_blank_until_the_window_is_full(self):
        built = self.build()
        home = built[built[recency.IS_HOME_COLUMN]].sort_values(recency.DATE_COLUMN)
        self.assertTrue(np.isnan(home["PTS_mean2"].iloc[0]))
        self.assertTrue(np.isnan(home["PTS_mean2"].iloc[1]))
        self.assertAlmostEqual(home["PTS_mean2"].iloc[2], 105.0, places=12)
        self.assertAlmostEqual(home["PTS_mean3"].iloc[3], 110.0, places=12)
        self.assertTrue(np.isnan(home["PTS_std3"].iloc[2]))

    def test_rolling_slope_matches_a_hand_fit(self):
        rows = game_lines(
            [date(2024, 1, 5), date(2024, 1, 7), date(2024, 1, 9), date(2024, 1, 12)],
            home_values=[line(points=p) for p in [0.0, 2.0, 4.0, 6.0]],
            away_values=[line(points=1.0) for _ in range(4)],
        )
        built = recency.rolling_features(rows, halflife=2, windows=(3,),
                                         metrics=TEST_METRICS)
        home = built[built[recency.IS_HOME_COLUMN]].sort_values(recency.DATE_COLUMN)
        # Slope of [0, 2, 4] against t = 0,1,2 is exactly 2 per game.
        self.assertAlmostEqual(home["PTS_slope3"].iloc[3], 2.0, places=12)

    def test_ewma_resets_at_the_season_boundary(self):
        """The weighted average starts empty in October; carrying it is opt-in.

        Four games in 2023-24 and then one in 2024-25, so the boundary game is the
        one worth looking at. By default history is keyed on (Team, Season), so
        that game is a COLD START: Games 0, EWMA NaN. With
        ``carry_across_seasons=True`` it keeps all four games instead, and the
        value is the module's weight convention, oldest first, halflife 2 games:

            weights   = 2**(-1.5), 2**(-1), 2**(-0.5), 1
                      = 0.35355339, 0.5, 0.70710678, 1      sum    2.56066017
            numerator = 100*0.35355339 + 110*0.5 + 120*0.70710678 + 130*1
                      = 35.35533906 + 55 + 84.85281363 + 130  = 305.20815280
            ewma      = 305.20815280 / 2.56066017              = 119.19119771

        The fixture has to hold TWO seasons to say anything: with one season in
        the frame, ``carry_across_seasons=True`` and ``False`` scope the same
        groups and return identical frames, so the flag is untested.
        """
        dates = [date(2024, 1, d) for d in (5, 7, 9, 12, 15)]
        rows = game_lines(
            dates,
            home_values=[line(points=p) for p in (100.0, 110.0, 120.0, 130.0, 140.0)],
            away_values=[line(points=1.0) for _ in dates],
        )
        # The first four games belong to the season that is closing; the last
        # opens the next one.
        rows.loc[rows[recency.DATE_COLUMN] < pd.Timestamp("2024-01-15"),
                 recency.SEASON_COLUMN] = "2023-24"

        carried = recency.rolling_features(rows, halflife=2, windows=(2,),
                                           metrics=TEST_METRICS,
                                           carry_across_seasons=True)
        reset = recency.rolling_features(rows, halflife=2, windows=(2,),
                                         metrics=TEST_METRICS)
        boundary_carried = carried[carried[recency.IS_HOME_COLUMN]].iloc[-1]
        boundary_reset = reset[reset[recency.IS_HOME_COLUMN]].iloc[-1]

        self.assertEqual(boundary_reset[recency.GAMES_COLUMN], 0)
        self.assertTrue(np.isnan(boundary_reset["PTS_ewma"]))
        self.assertEqual(boundary_carried[recency.GAMES_COLUMN], 4)
        self.assertAlmostEqual(boundary_carried["PTS_ewma"], 119.19119771, places=8)

    def test_min_history_drops_the_cold_start_row(self):
        rows = self.build()
        everything = recency.rolling_features(rows, halflife=2, windows=(2,),
                                              metrics=TEST_METRICS)
        warmed = recency.rolling_features(rows, halflife=2, windows=(2,),
                                          metrics=TEST_METRICS, min_history=3)
        self.assertEqual(len(everything.index), 10)
        # Games runs 0..4 per team, so games >= 3 is the fourth and fifth game of
        # each: 2 home + 2 away. A six-game fixture would give six.
        self.assertEqual(len(warmed.index), 4)
        self.assertTrue((warmed[recency.GAMES_COLUMN] >= 3).all())

    def test_missing_metric_column_is_refused_with_a_useful_message(self):
        rows = self.build()
        with self.assertRaises(ValueError) as caught:
            recency.rolling_features(rows.drop(columns=["FGA"]), metrics=TEST_METRICS)
        self.assertIn("extract_game_logs", str(caught.exception))


class TestOpponentContextAndPace(unittest.TestCase):

    def features(self):
        dates = [date(2024, 1, 5), date(2024, 1, 7)]
        rows = game_lines(
            dates,
            home_values=[line(points=120.0, plus_minus=10.0, fga=95.0, orb=12.0),
                         line(points=120.0, plus_minus=10.0, fga=95.0, orb=12.0)],
            away_values=[line(points=100.0, plus_minus=-8.0, fga=85.0, orb=8.0),
                         line(points=100.0, plus_minus=-8.0, fga=85.0, orb=8.0)],
        )
        return recency.rolling_features(rows, halflife=2, windows=(2,),
                                        metrics=TEST_METRICS)

    def test_sos_uses_the_opponents_pre_game_ewma(self):
        built = recency.attach_opponent_context(self.features(), metrics=TEST_METRICS)
        home = built[built[recency.IS_HOME_COLUMN] &
                      (built[recency.DATE_COLUMN] == pd.Timestamp("2024-01-07"))]
        # League average of PLUS_MINUS_ewma on the first date is (10 + -8)/2 = 1.0;
        # the opponent's is -8.0; the team's raw is 10.0.
        # sos = 10 + (1 - (-8)) * (10 / 1) = 100.
        self.assertAlmostEqual(home["PLUS_MINUS_sos"].iloc[0], 100.0, places=9)
        # And it is the opponent that moved, not this team's own history.
        self.assertAlmostEqual(home["PLUS_MINUS_ewma"].iloc[0], 10.0, places=12)

    def test_opponent_context_needs_the_ewma_columns_first(self):
        with self.assertRaises(ValueError) as caught:
            recency.attach_opponent_context(
                pd.DataFrame({recency.TEAM_COLUMN: ["A"], recency.OPPONENT_COLUMN: ["B"],
                              recency.DATE_COLUMN: pd.to_datetime(["2024-01-01"])})
            )
        self.assertIn("rolling_features", str(caught.exception))

    def test_pace_team_basis_matches_the_documented_formula(self):
        built = recency.pace_adjust(self.features(), metrics=("PTS",))
        row = built[built[recency.IS_HOME_COLUMN]].iloc[0]
        # Row 0 has no history, so the possession estimate is NaN there; use the
        # second game, whose EWMA is the first game's line.
        row = built[built[recency.IS_HOME_COLUMN]].iloc[1]
        expected_possessions = 95.0 + 0.44 * 22.0 - 0.5 * 12.0
        self.assertAlmostEqual(row["Possessions"], expected_possessions, places=9)
        self.assertAlmostEqual(
            row["PTS_per100"], 100.0 * 120.0 / expected_possessions, places=9
        )

    def test_pace_matched_basis_requires_the_opponents_estimates(self):
        built = self.features()
        with self.assertRaises(ValueError) as caught:
            recency.pace_adjust(built, basis="matched")
        self.assertIn("attach_opponent_context", str(caught.exception))

    def test_pace_matched_basis_is_the_two_sided_formula(self):
        built = recency.attach_opponent_context(
            self.features(), metrics=TEST_METRICS, keep_opponent_columns=True
        )
        adjusted = recency.pace_adjust(built, metrics=("PTS",), basis="matched")
        row = adjusted[adjusted[recency.IS_HOME_COLUMN]].iloc[1]
        expected = 0.5 * (
            (95.0 + 0.44 * 22.0 - 8.0) + (85.0 + 0.44 * 22.0 - 12.0)
        )
        self.assertAlmostEqual(row["Possessions"], expected, places=9)


class TestExtractGameLogs(unittest.TestCase):
    """Recovering one game's line out of two cumulative snapshots."""

    TEAMS = ("Lakers", "Celtics", "Bulls", "Suns")

    def snapshots(self, per_game_lines, start=date(2024, 1, 1), gap=2):
        """Build dated snapshots from known per-game lines.

        Snapshot dated ``d`` accumulates every game whose date is strictly before
        ``d``, which is the convention the committed data actually follows.

        Snapshot DATES come from a regular grid -- ``start``, ``start + gap``,
        ``start + 2 * gap``, ... -- run up to the last game and closed by one
        snapshot the day after it so the final game still has a bounding pair. The
        spacing is not decoration: ``extract_game_logs`` bounds a game on ``D``
        with the NEWEST snapshot at or before ``D``, so a snapshot landing exactly
        on a game date would sit inside that game's window instead of behind it,
        and the difference between the two snapshots would no longer be the game.
        """
        frames = {}
        game_dates = sorted({entry["date"] for entry in per_game_lines})
        last_game = game_dates[-1]
        grid = []
        day = start
        while day <= last_game:
            grid.append(day)
            day += timedelta(days=gap)
        grid.append(last_game + timedelta(days=1))

        for snapshot_date in sorted(set(grid)):
            totals = {team: {} for team in self.TEAMS}
            for entry in per_game_lines:
                if entry["date"] >= snapshot_date:
                    continue
                bucket = totals[entry["team"]]
                bucket["GP"] = bucket.get("GP", 0) + 1
                for column, value in entry["values"].items():
                    # GP is already counted, once per game. A fixture that repeats
                    # it inside "values" must not count it a second time, or every
                    # window comes out twice as wide as the game it holds.
                    if column == "GP":
                        continue
                    bucket[column] = bucket.get(column, 0.0) + value
            frames[snapshot_date] = snapshot(totals, self.TEAMS)
        return frames

    def test_averages_are_converted_to_totals_before_differencing(self):
        """The core correctness claim, on numbers where the shortcut is obvious.

        The Lakers play 110 and then 130 points. Cumulative totals are 110 then
        240. The snapshots therefore hold PTS averages of 110.0 (GP 1) and 120.0
        (GP 2). Differencing the AVERAGES gives 10 -- nonsense. Converting to
        totals first (120 * 2 - 110 * 1) gives 130, the real game.
        """
        lines = [
            {"date": date(2024, 1, 1), "team": "Lakers",
             "values": {"GP": 1, "PTS": 110.0, "PLUS_MINUS": 4.0,
                        "FGA": 80.0, "FTA": 20.0}},
            {"date": date(2024, 1, 3), "team": "Lakers",
             "values": {"GP": 1, "PTS": 130.0, "PLUS_MINUS": 9.0,
                        "FGA": 90.0, "FTA": 24.0}},
            # The opponent needs a line in the window too: a game is emitted only
            # when BOTH sides are recovered, so a Lakers-only fixture is a game
            # that is correctly dropped and asserts nothing about differencing.
            {"date": date(2024, 1, 3), "team": "Celtics",
             "values": {"GP": 1, "PTS": 105.0, "PLUS_MINUS": -9.0}},
        ]
        snapshots = self.snapshots(lines)
        self.assertAlmostEqual(
            snapshots[date(2024, 1, 4)].set_index("TEAM_NAME").at["Lakers", "PTS"], 120.0
        )

        games = pd.DataFrame({
            "Date": ["2024-01-03"], "Home": ["Lakers"], "Away": ["Celtics"],
            "Season": ["2024-25"],
        })
        logs, stats = recency.extract_game_logs(
            snapshots, games, season_starts={"2024-25": date(2023, 10, 24)}
        )
        self.assertEqual(stats["kept_games"], 1)
        lakers = logs[logs[recency.TEAM_COLUMN] == "Lakers"].iloc[0]
        self.assertAlmostEqual(lakers["PTS"], 130.0, places=9)
        self.assertAlmostEqual(lakers["PLUS_MINUS"], 9.0, places=9)
        self.assertEqual(lakers["Window_Games"], 1)
        self.assertAlmostEqual(
            lakers["TS_PCT"], 130.0 / (2.0 * (lakers["FGA"] + 0.44 * lakers["FTA"])),
            places=12,
        )

    def test_rates_are_recomputed_never_differenced(self):
        lines = [
            {"date": date(2024, 1, 1), "team": "Lakers",
             "values": {"GP": 1, "FGA": 80.0, "FTA": 20.0, "FGM": 40.0, "FG3M": 10.0}},
            {"date": date(2024, 1, 3), "team": "Lakers",
             "values": {"GP": 1, "FGA": 100.0, "FTA": 30.0, "FGM": 30.0, "FG3M": 5.0}},
            {"date": date(2024, 1, 3), "team": "Celtics",
             "values": {"GP": 1, "FGA": 88.0, "FTA": 18.0, "FGM": 42.0, "FG3M": 8.0}},
        ]
        snapshots = self.snapshots(lines)
        games = pd.DataFrame({"Date": ["2024-01-03"], "Home": ["Lakers"],
                              "Away": ["Celtics"], "Season": ["2024-25"]})
        logs, _ = recency.extract_game_logs(snapshots, games,
                                            season_starts={"2024-25": date(2023, 10, 24)})
        lakers = logs[logs[recency.TEAM_COLUMN] == "Lakers"].iloc[0]
        self.assertAlmostEqual(lakers["FGA"], 100.0, places=9)
        self.assertAlmostEqual(lakers["EFG_PCT"], (30.0 + 0.5 * 5.0) / 100.0, places=12)

    def test_a_window_holding_two_games_is_dropped_not_smeared(self):
        lines = [
            {"date": date(2024, 1, 1), "team": "Lakers",
             "values": {"GP": 1, "PTS": 100.0}},
            {"date": date(2024, 1, 2), "team": "Lakers",
             "values": {"GP": 1, "PTS": 120.0}},
            # Both sides span two games here: the game is dated between the two,
            # so both bounding snapshots sit outside it.
            {"date": date(2024, 1, 1), "team": "Celtics",
             "values": {"GP": 1, "PTS": 108.0}},
            {"date": date(2024, 1, 2), "team": "Celtics",
             "values": {"GP": 1, "PTS": 112.0}},
        ]
        snapshots = self.snapshots(lines)
        games = pd.DataFrame({"Date": ["2024-01-02"], "Home": ["Lakers"],
                              "Away": ["Celtics"], "Season": ["2024-25"]})
        exact, exact_stats = recency.extract_game_logs(
            snapshots, games, season_starts={"2024-25": date(2023, 10, 24)}
        )
        self.assertTrue(exact.empty)
        self.assertEqual(exact_stats["window_spans_many"], 1)

        loose, loose_stats = recency.extract_game_logs(
            snapshots, games, exact_only=False,
            season_starts={"2024-25": date(2023, 10, 24)}
        )
        self.assertEqual(loose_stats["kept_games"], 1)
        lakers = loose[loose[recency.TEAM_COLUMN] == "Lakers"].iloc[0]
        self.assertEqual(lakers["Window_Games"], 2)
        self.assertAlmostEqual(lakers["PTS"], 110.0, places=9)  # window mean

    def test_a_window_may_not_cross_the_season_start(self):
        # The game sits ON the season's first day, which is what makes this the
        # scenario: the newest snapshot at or before it predates the season, so
        # the window it would be recovered from begins in the previous season's
        # accumulation and the totals in it are not this season's prefix.
        lines = [
            {"date": date(2023, 12, 28), "team": "Lakers",
             "values": {"GP": 1, "PTS": 100.0}},
            {"date": date(2024, 10, 22), "team": "Lakers",
             "values": {"GP": 1, "PTS": 120.0}},
        ]
        snapshots = self.snapshots(lines)
        games = pd.DataFrame({"Date": ["2024-10-22"], "Home": ["Lakers"],
                              "Away": ["Celtics"], "Season": ["2024-25"]})
        logs, stats = recency.extract_game_logs(
            snapshots, games, season_starts={"2024-25": date(2024, 10, 22)}
        )
        self.assertTrue(logs.empty)
        self.assertEqual(stats["crosses_season"], 1)

    def test_a_gp_reset_inside_the_window_is_refused(self):
        lines = [
            {"date": date(2024, 1, 1), "team": "Lakers",
             "values": {"GP": 1, "PTS": 100.0}},
        ]
        snapshots = self.snapshots(lines)
        # A snapshot where the season restarted: GP is back to 1, PTS back to 115.
        snapshots[date(2024, 1, 9)] = snapshot(
            {
                "Lakers": {"GP": 1, "PTS": 115.0},
                **{team: {"GP": 1, "PTS": 100.0}
                   for team in self.TEAMS if team != "Lakers"},
            },
            self.TEAMS,
        )
        games = pd.DataFrame({"Date": ["2024-01-08"], "Home": ["Lakers"],
                              "Away": ["Celtics"], "Season": ["2024-25"]})
        logs, stats = recency.extract_game_logs(snapshots, games)
        self.assertTrue(logs.empty)
        self.assertEqual(stats["no_game_in_window"], 1)

    def test_a_game_is_only_emitted_when_both_sides_are_recovered(self):
        lines = [
            {"date": date(2024, 1, 1), "team": "Lakers",
             "values": {"GP": 1, "PTS": 100.0}},
            {"date": date(2024, 1, 3), "team": "Lakers",
             "values": {"GP": 1, "PTS": 120.0}},
            {"date": date(2024, 1, 3), "team": "Celtics",
             "values": {"GP": 1, "PTS": 110.0}},
        ]
        snapshots = self.snapshots(lines)
        games = pd.DataFrame({"Date": ["2024-01-03"], "Home": ["Lakers"],
                              "Away": ["Celtics"], "Season": ["2024-25"]})
        logs, stats = recency.extract_game_logs(snapshots, games)
        self.assertEqual(len(logs.index), 2)  # both sides, or nothing
        self.assertEqual(stats["kept_games"], 1)

    def test_team_aliases_are_resolved_on_both_sides(self):
        self.assertEqual(recency.SNAPSHOT_TEAM_ALIASES["LA Clippers"],
                         "Los Angeles Clippers")
        self.assertIn("Charlotte Hornets", recency.SNAPSHOT_TEAM_ALIASES)


class TestLoadTeamSnapshots(unittest.TestCase):

    def test_loader_reads_dated_tables_and_skips_short_ones(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "teams.sqlite"
            complete = snapshot(
                {team: {"GP": 3, "PTS": 330.0} for team in ("A", "B", "C")},
                ("A", "B", "C"),
            )
            with sqlite3.connect(path) as con:
                complete.to_sql("2024-01-01", con, if_exists="replace", index=False)
                complete.head(2).to_sql("2024-01-02", con, if_exists="replace", index=False)
                pd.DataFrame({"nonsense": [1]}).to_sql("not_a_date", con,
                                                        if_exists="replace", index=False)
            before = path.read_bytes()

            loaded = recency.load_team_snapshots(path, expected_team_rows=3)

            self.assertEqual(list(loaded), [date(2024, 1, 1)])
            self.assertEqual(loaded[date(2024, 1, 1)].at["A", "GP"], 3)
            self.assertEqual(loaded[date(2024, 1, 1)].at["A", "PTS"], 110.0)
            # Read-only: the loader must not have written a thing.
            self.assertEqual(path.read_bytes(), before)


class TestBuildRecencyDatasetAndLeakageGate(unittest.TestCase):

    def build_rows(self):
        dates = [date(2024, 1, d) for d in (5, 7, 9, 12, 15, 18)]
        home_points = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0]
        away_points = [95.0, 105.0, 115.0, 125.0, 135.0, 145.0]
        rows = game_lines(
            dates,
            home_values=[line(points=p, plus_minus=p - 100.0) for p in home_points],
            away_values=[line(points=p, plus_minus=100.0 - p) for p in away_points],
        )
        rows[recency.LABEL_COLUMN] = 1
        # One score per GAME, and a game is two rows, so the six scores are
        # broadcast across the twelve team rows -- game i's score rides on both
        # of its rows, which is what a game-level column on team_games looks like.
        rows[recency.SCORE_COLUMN] = [score for score in home_points for _ in (0, 1)]
        rows["Days-Rest-Home"] = 2
        rows["Days-Rest-Away"] = 3
        return rows

    def dataset(self, **kwargs):
        kwargs.setdefault("metrics", TEST_METRICS)
        kwargs.setdefault("windows", (2,))
        kwargs.setdefault("halflife", 2)
        return recency.build_recency_dataset(self.build_rows(), **kwargs)

    def test_dataset_shape_and_column_convention(self):
        built = self.dataset()
        # Six games, minus the first (cold start on both sides) = five rows.
        self.assertEqual(len(built.index), 5)
        self.assertIn("PTS_ewma", built.columns)
        self.assertIn("PTS_ewma.1", built.columns)
        self.assertIn(recency.HOME_AS_OF, built.columns)
        self.assertIn(recency.AWAY_AS_OF, built.columns)
        self.assertTrue((built[recency.HOME_AS_OF] < built[recency.DATE_COLUMN]).all())
        self.assertTrue((built[recency.AWAY_AS_OF] < built[recency.DATE_COLUMN]).all())

    def test_dataset_carries_no_label_or_provenance_columns_as_features(self):
        features = recency.feature_columns(self.dataset())
        for banned in (recency.LABEL_COLUMN, recency.SCORE_COLUMN,
                       recency.HOME_AS_OF, recency.AWAY_AS_OF,
                       "Is_Win", "Window_Games"):
            self.assertNotIn(banned, features)
            self.assertNotIn(banned + ".1", features)
        self.assertIn("PTS_ewma", features)
        self.assertIn("PTS_ewma.1", features)

    def test_away_side_features_are_the_away_teams_own(self):
        built = self.dataset()
        first = built.iloc[0]
        self.assertGreater(first["PTS_ewma"], first["PTS_ewma.1"])

    def test_leaky_supplied_feature_is_refused(self):
        """THE CHECK THAT MATTERS MOST.

        A feature computed from game D's OWN result is handed back to
        build_recency_dataset under the name of a real one. The strictly-prior
        derivation cannot reproduce it, so the build stops instead of writing a
        contaminated dataset.
        """
        built = self.dataset()
        leaky = pd.DataFrame({
            recency.GAME_ID_COLUMN: built[recency.GAME_ID_COLUMN],
            # PTS_ewma plus the game that is being predicted.
            "PTS_ewma": built["PTS_ewma"] + 40.0,
        })
        with self.assertRaises(recency.LeakageError) as caught:
            recency.build_recency_dataset(
                self.build_rows(), metrics=TEST_METRICS, windows=(2,), halflife=2,
                supplied_features=leaky,
            )
        message = str(caught.exception)
        self.assertIn("PTS_ewma", message)
        self.assertIn("strictly-prior", message)

    def test_leaky_feature_of_the_away_side_is_refused_too(self):
        built = self.dataset()
        leaky = pd.DataFrame({
            recency.GAME_ID_COLUMN: built[recency.GAME_ID_COLUMN],
            "PTS_ewma.1": built["PTS_ewma.1"] - 45.0,
        })
        with self.assertRaises(recency.LeakageError):
            recency.build_recency_dataset(
                self.build_rows(), metrics=TEST_METRICS, windows=(2,), halflife=2,
                supplied_features=leaky,
            )

    def test_supplied_column_named_after_a_result_is_refused_before_arithmetic(self):
        leaky = pd.DataFrame({
            recency.GAME_ID_COLUMN: self.dataset()[recency.GAME_ID_COLUMN],
            "Score": 220.0,
        })
        with self.assertRaises(recency.LeakageError) as caught:
            recency.build_recency_dataset(
                self.build_rows(), metrics=TEST_METRICS, windows=(2,), halflife=2,
                supplied_features=leaky,
            )
        self.assertIn("Score", str(caught.exception))

    def test_unverifiable_supplied_feature_is_refused_unless_verification_is_off(self):
        invented = pd.DataFrame({
            recency.GAME_ID_COLUMN: self.dataset()[recency.GAME_ID_COLUMN],
            "Magic_Number": np.arange(5, dtype=float),
        })
        with self.assertRaises(recency.LeakageError) as caught:
            recency.build_recency_dataset(
                self.build_rows(), metrics=TEST_METRICS, windows=(2,), halflife=2,
                supplied_features=invented,
            )
        self.assertIn("verify=False", str(caught.exception))

        accepted = recency.build_recency_dataset(
            self.build_rows(), metrics=TEST_METRICS, windows=(2,), halflife=2,
            supplied_features=invented, verify=False,
        )
        self.assertIn("Magic_Number", accepted.columns)

    def test_matching_supplied_feature_is_accepted(self):
        built = self.dataset()
        honest = pd.DataFrame({
            recency.GAME_ID_COLUMN: built[recency.GAME_ID_COLUMN],
            "PTS_ewma": built["PTS_ewma"],
        })
        accepted = recency.build_recency_dataset(
            self.build_rows(), metrics=TEST_METRICS, windows=(2,), halflife=2,
            supplied_features=honest,
        )
        self.assertEqual(len(accepted.index), 5)
        self.assertNotIn("PTS_ewma__supplied", accepted.columns)

    def test_assert_no_leakage_fires_on_the_original_bug(self):
        """Recreate the inclusive-endpoint leak directly, and watch it be caught.

        The original 68.9% build fed a game's own snapshot into its own features.
        Here that shows up as an As_Of equal to the game date, and the gate
        refuses the frame instead of scoring it.
        """
        built = self.dataset()
        contaminated = built.copy()
        contaminated[recency.HOME_AS_OF] = contaminated[recency.DATE_COLUMN]
        with self.assertRaises(recency.LeakageError) as caught:
            recency.assert_no_leakage(contaminated)
        message = str(caught.exception)
        self.assertIn("68.9%", message)
        self.assertIn(recency.HOME_AS_OF, message)

    def test_assert_no_leakage_requires_provenance(self):
        built = self.dataset()
        with self.assertRaises(recency.LeakageError) as caught:
            recency.assert_no_leakage(built.drop(columns=[recency.AWAY_AS_OF]))
        self.assertIn("provenance", str(caught.exception))

    def test_assert_no_leakage_catches_history_count_disagreement(self):
        built = self.dataset()
        broken = built.copy()
        broken.loc[broken.index[0], recency.HOME_AS_OF] = pd.NaT
        with self.assertRaises(recency.LeakageError):
            recency.assert_no_leakage(broken)

    def test_assert_no_leakage_refuses_a_result_column_present_as_a_feature(self):
        built = self.dataset()
        built["Final_Score"] = 220.0
        with self.assertRaises(recency.LeakageError):
            recency.assert_no_leakage(built)

    def test_leakage_error_is_a_value_error(self):
        # Existing `except ValueError` handlers around this pipeline keep working.
        self.assertTrue(issubclass(recency.LeakageError, ValueError))

    def test_cold_start_rows_are_dropped_not_filled(self):
        every_game = self.dataset(min_history=0)
        warmed = self.dataset(min_history=4)
        self.assertEqual(len(every_game.index), 6)
        self.assertEqual(len(warmed.index), 2)
        self.assertTrue((warmed[recency.HOME_GAMES] >= 4).all())

    def test_a_one_sided_frame_is_refused(self):
        rows = self.build_rows()
        broken = rows[~((rows[recency.TEAM_COLUMN] == "Celtics")
                        & (rows[recency.DATE_COLUMN] == pd.Timestamp("2024-01-09")))]
        with self.assertRaises(ValueError) as caught:
            recency.build_recency_dataset(broken, metrics=TEST_METRICS, windows=(2,),
                                          halflife=2)
        self.assertIn("extract_game_logs", str(caught.exception))

    def test_matched_pace_basis_runs_end_to_end(self):
        built = recency.build_recency_dataset(
            self.build_rows(), metrics=TEST_METRICS, windows=(2,), halflife=2,
            pace_basis="matched",
        )
        self.assertIn("Possessions", built.columns)
        self.assertTrue(np.isfinite(built["Possessions"]).any())


class TestModuleSurface(unittest.TestCase):

    def test_the_documented_functions_all_exist_and_are_callable(self):
        for name in ("ewma", "ewma_weights", "ewma_history", "rolling_mean",
                     "rolling_std", "rolling_slope", "history_before",
                     "rolling_features", "attach_opponent_context", "pace_adjust",
                     "build_recency_dataset", "assert_no_leakage",
                     "extract_game_logs", "load_team_snapshots",
                     "load_odds_games", "feature_columns"):
            self.assertTrue(callable(getattr(recency, name)), name)

    def test_game_id_is_stable_across_date_spellings(self):
        self.assertEqual(recency.game_id("2024-01-02", "A", "B"),
                         recency.game_id(date(2024, 1, 2), "A", "B"))
        self.assertEqual(recency.game_id(date(2024, 1, 2), "A", "B"),
                         "2024-01-02|A|B")

    def test_select_odds_table_prefers_the_new_table(self):
        with sqlite3.connect(":memory:") as con:
            con.execute('CREATE TABLE "odds_2024-25" (Date TEXT)')
            con.execute('CREATE TABLE "odds_2024-25_new" (Date TEXT)')
            self.assertEqual(recency.select_odds_table(con, "2024-25"),
                             "odds_2024-25_new")
            self.assertIsNone(recency.select_odds_table(con, "1999-00"))

    def test_module_has_no_type_annotations(self):
        source = MODULE_PATH.read_text()
        tree = __import__("ast").parse(source)
        for node in __import__("ast").walk(tree):
            if isinstance(node, (__import__("ast").FunctionDef,
                                 __import__("ast").AsyncFunctionDef)):
                self.assertIsNone(node.returns, f"{node.name} has a return annotation")
                for argument in node.args.args:
                    self.assertIsNone(argument.annotation,
                                      f"{node.name}:{argument.arg} is annotated")


if __name__ == "__main__":
    unittest.main()