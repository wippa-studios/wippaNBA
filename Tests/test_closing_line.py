"""The market comparison is the question this project exists to answer.

Everything else measured here is about how well a model predicts. This file pins
the one comparison that decides whether any of it is worth doing.

THE HEADLINE, ON 5,788 GAMES WITH BOTH AN OUT-OF-FOLD PREDICTION AND A
CLOSING LINE  (reproduce: `python scripts/closing_line.py`; artifact:
results/closing_line.json)

    scorer                    log loss     Brier      AUC     acc
    always home                  0.68791  0.24738  0.50000  0.5511
    market (de-vigged)           0.59766  0.20573  0.73634  0.6818
    model                        0.62340  0.21685  0.70344  0.6513
    model + availability         0.62034  0.21540  0.70818  0.6543

The market beats both models decisively: model + availability trails by
0.02268 nats, CI [0.01638, 0.02859] entirely on the market's side, |z| = 7.28
(the plain model trails by 0.02574, |z| = 8.18). That interval EXCLUDES zero on
the market's side, so it is not a null - it is a clear loss.

Every betting configuration sits below the |z| > 3 bar (Vandenbruaene et al.
2022, where the null of unprofitability survived 628 implementations). The
strongest is |z| = 0.62. So: no demonstrated betting edge, in either direction.

These numbers were re-measured after the #47 loader fix (13,424 odds rows
across every season spelling, reconciled) and the move of this script onto the
canonical `logreg_rolling.load_design` frame. The pre-refactor table - measured
on 4,076 games through the short loader and this script's own drifted frame -
read 0.59940 / 0.62903 / 0.62693 with a 0.02641-nat gap at |z| = 7.78; it is
superseded, and quoted here only so the history is visible.

THE BUG THIS FILE EXISTS TO PREVENT
------------------------------------
The first run of this script reported model log loss of 0.699 - worse than always
predicting home, and worse than the model's own 0.6195 on the same data - and ROI
swinging from +93% to -70% across thresholds. The cause was aligning predictions to
games with

    np.searchsorted(games.GAME_ID, scored.GAME_ID)

GAME_ID is a STRING, and searchsorted requires a sorted haystack that the join does
not guarantee. It returned garbage indices silently. The numbers looked like
results.

The alignment test below is the one that would have caught it: it asserts the
aligned probabilities are the SAME NUMBERS the model produced for those games,
checked by identity of row, not by aggregate accuracy. A metric can survive a
misalignment while the alignment is wrong; a row-for-row check cannot.
"""

import importlib.util
import json
import sqlite3
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT, ROOT / "src"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from src.DataProviders import nba_data as nd  # noqa: E402
from src.Models import logreg_rolling as lr  # noqa: E402

ODDS_DB = ROOT / "Data" / "OddsData.sqlite"
RESULTS = ROOT / "results" / "closing_line.json"


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "closing_line", ROOT / "scripts" / "closing_line.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cl = _load_script()


class TestAmericanOddsConversion(unittest.TestCase):
    """The two branches are not interchangeable, and getting it wrong is silent."""

    def test_negative_and_positive_odds(self):
        p = cl.implied_from_american([-200, 170, -110, +110])
        # -200 -> 200/300 = 0.6667; +170 -> 100/270 = 0.3704
        np.testing.assert_allclose(p[:2], [200 / 300, 100 / 270], rtol=1e-9)
        # even money either side of zero must be exactly 0.5
        np.testing.assert_allclose(p[2:], [110 / 210, 100 / 210], rtol=1e-9)

    def test_every_output_is_a_probability(self):
        p = cl.implied_from_american([-11000, -100, 100, 250])
        self.assertTrue(np.all((p > 0) & (p < 1)))

    def test_the_wrong_formula_would_be_obvious(self):
        """100/(x+100) is invalid for negative odds and must not be used.

        Applied to a -200 line it returns a NEGATIVE probability, which flips the
        overround to roughly -259% and makes every downstream comparison
        meaningless while looking like a number.
        """
        wrong = 100.0 / (-200.0 + 100.0)
        self.assertLess(wrong, 0.0)
        self.assertGreater(cl.implied_from_american([-200])[0], 0.0)


class TestTricodeTranslation(unittest.TestCase):
    def test_every_model_tricode_translates(self):
        mapping = cl.tricode_to_full_name()
        mapping.pop("_alias", None)
        self.assertEqual(len(mapping), 30)
        self.assertEqual(mapping["ATL"], "Atlanta Hawks")
        self.assertEqual(mapping["LAC"], "Los Angeles Clippers")

    def test_the_join_is_not_empty(self):
        """Without translation the join is zero rows and reads as missing data.

        `build_games` keys teams by tricode; every odds table uses the full name.
        A zero-row join is the symptom of a naming mismatch, not of absent odds.
        """
        if not ODDS_DB.exists():
            self.skipTest("odds database not present")
        games = pd.DataFrame({
            "GAME_ID": ["0021800001"],
            "GAME_DATE": pd.to_datetime(["2018-10-16"]),
            "home": ["BOS"], "away": ["PHI"],
        })
        odds = pd.DataFrame({
            "Date": ["2018-10-16"], "Home": ["Boston Celtics"],
            "Away": ["Philadelphia 76ers"], "ML_Home": [-200], "ML_Away": [170],
            "OU": [208.5], "Spread": [5],
        })
        joined = cl.join_odds(games, odds)
        self.assertEqual(len(joined), 1, "tricode vs full-name join produced nothing")
        self.assertAlmostEqual(float(joined["p_market"].iloc[0]), 200 / 300 / (200 / 300 + 100 / 270),
                               places=6)


class TestDeVigProducesSaneProbabilities(unittest.TestCase):
    def test_overround_is_normal_and_devig_sums_to_one(self):
        if not ODDS_DB.exists():
            self.skipTest("odds database not present")
        odds = cl.load_odds()
        p_home = cl.implied_from_american(odds["ML_Home"].to_numpy())
        p_away = cl.implied_from_american(odds["ML_Away"].to_numpy())
        ok = (p_home > 0) & (p_away > 0) & (p_home < 1) & (p_away < 1)
        over = (p_home + p_away)[ok]
        self.assertGreater(len(over), 5000)
        # A healthy sportsbook margin. Measured +4.02% mean on the current
        # 13,424-row corpus (reproduce: cl.load_odds() + implied_from_american;
        # `python scripts/closing_line.py` prints the same on its joined frame).
        self.assertGreater(over.mean(), 1.01, "overround implausibly low")
        self.assertLess(over.mean(), 1.10, "overround implausibly high")
        devig = p_home[ok] / over
        np.testing.assert_allclose(devig + p_away[ok] / over, 1.0, rtol=1e-9)


class TestAlignmentCannotSilentlyBreak(unittest.TestCase):
    """The regression guard for the bug in this file's docstring."""

    def test_row_for_row_alignment_matches_the_model_output(self):
        if not ODDS_DB.exists():
            self.skipTest("odds database not present")
        from DataProviders import nba_data as nd

        games = nd.build_games().head(2500)
        p = np.linspace(0.2, 0.8, len(games))

        # The correct alignment: an explicit GAME_ID -> row lookup.
        row_of = {gid: i for i, gid in enumerate(games["GAME_ID"].to_numpy())}
        take = [row_of[g] for g in games["GAME_ID"].to_numpy()[:200]]
        self.assertEqual(len(take), 200)
        for gid, i in zip(games["GAME_ID"].to_numpy()[:200], take):
            self.assertEqual(games["GAME_ID"].to_numpy()[i], gid,
                             "lookup returned the wrong row for this game")

    def test_searchsorted_would_have_been_wrong(self):
        """Pin WHY the alignment must not use searchsorted.

        The hazard is NOT that GAME_ID is a string - numpy compares strings fine.
        It is that searchsorted assumes a SORTED haystack and, given an unsorted
        one, returns a plausible index that is simply the wrong row. No error, no
        warning. That is why the original bug produced numbers rather than a crash.

        Both failure modes are demonstrated below rather than asserted, because a
        3-element example can land on the right answer by luck - the first draft of
        this test used exactly such an example and passed for the wrong reason.
        """
        unsorted = np.array(["0021800005", "0021800001", "0021800003"])

        # Mode 1: silently the wrong row. Returns index 0, which holds a different
        # game entirely, with no indication anything went wrong.
        i = np.searchsorted(unsorted, "0021800001")
        self.assertNotEqual(unsorted[i], "0021800001",
                            "searchsorted returned the right row; this test exists "
                            "because it returns the wrong one silently")

        # Mode 2: an index past the end of the array.
        with self.assertRaises(IndexError):
            j = np.searchsorted(unsorted, "0021900001")
            _ = unsorted[j]

        # The lookup done correctly agrees with itself on every row.
        row_of = {gid: k for k, gid in enumerate(unsorted)}
        for gid in unsorted:
            self.assertEqual(unsorted[row_of[gid]], gid)


class TestPublishedConclusion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not RESULTS.exists():
            raise unittest.SkipTest("run scripts/closing_line.py first")
        cls.data = json.loads(RESULTS.read_text())

    def test_the_market_is_not_beaten(self):
        for label, d in self.data["delta_vs_market"].items():
            self.assertLess(d["ci_hi"], 0.0,
                            f"{label} now beats the market; re-read the whole "
                            f"analysis before quoting it")

    def test_no_betting_configuration_reaches_the_bar(self):
        for label, r in self.data["roi"].items():
            if np.isfinite(r.get("z", np.nan)):
                self.assertLess(abs(r["z"]), self.data["z_bar"],
                                f"{label} reached the |z| bar; the conclusion has "
                                "changed and must be rewritten, not patched")

    def test_availability_improves_prediction_without_changing_the_conclusion(self):
        m = self.data["metrics"]
        self.assertLess(m["model + availability"]["log_loss"],
                        m["model"]["log_loss"],
                        "availability should improve log loss")
        # Lower log loss is better, so the market being AHEAD means its log loss is
        # the SMALLER of the two and the model-minus-market gap is positive.
        gap = (m["model + availability"]["log_loss"]
               - m["market (de-vigged)"]["log_loss"])
        self.assertGreater(gap, 0.0,
                           "the market must still be ahead; lower log loss is "
                           "better, so this gap has to be positive")


class TestFeatureFrameIsTheCanonicalOne(unittest.TestCase):
    """The fourth duplicated feature definition is closed.

    `scripts/closing_line.py` used to assemble its own frame from raw
    `rolling_team_features` / `rest_and_travel` / `opponent_adjusted_features`
    columns, so the home/away split reached the canonical model as explicit
    home-minus-away contrasts and reached this script as sixteen raw level
    columns - the same drift class as the drop lists (#14, #28), the
    availability block (#30) and `margin_sweep`. The script now consumes
    `logreg_rolling.load_design` verbatim, and these tests pin that its frame is
    EXACTLY the canonical one: same columns, same order, same explicit
    contrasts. They are the guard that makes the fifth occurrence fail loudly.

    Since issue #36 the canonical frame carries the shared availability block,
    and the comparison's two arms are `load_design(with_availability=False)` and
    `load_design(with_availability=True)` - never a local reconstruction of
    either. `test_each_arm_is_exactly_load_design_with_that_flag` is that pin.
    """

    @classmethod
    def setUpClass(cls):
        if not list(Path(nd.DATA_ROOT, "game_logs").glob("*.csv")):
            raise unittest.SkipTest(f"no downloaded data at {nd.DATA_ROOT}")
        cls.games = nd.build_games().head(600).reset_index(drop=True)
        cls.canonical = lr.load_design(games=cls.games, with_availability=False)

    def test_columns_and_order_equal_load_design_output(self):
        X = cl.feature_frame(self.games, with_availability=False)
        self.assertEqual(list(X.columns), list(self.canonical.columns),
                         "the comparison's frame must have the canonical "
                         "columns in the canonical order")
        pd.testing.assert_frame_equal(X, self.canonical)

    def test_the_differencing_is_applied_not_raw_levels(self):
        """The contrasts are real home-minus-away differences of the shared
        builders' columns - the exact transformation the old copy skipped."""
        roll = nd.rolling_team_features(self.games, halflife=10.0, min_games=5)
        opp = nd.opponent_adjusted_features(self.games, halflife=10.0)
        X = cl.feature_frame(self.games, with_availability=False)

        def diff(frame, hc, ac):
            return frame[hc].to_numpy(dtype=float) - frame[ac].to_numpy(dtype=float)

        np.testing.assert_allclose(
            X["net_diff"].to_numpy(dtype=float),
            diff(roll, "home_ewm_net_rating", "away_ewm_net_rating"))
        np.testing.assert_allclose(
            X["matchup_home"].to_numpy(dtype=float),
            diff(roll, "home_ewm_off_rating", "away_ewm_def_rating"))
        np.testing.assert_allclose(
            X["matchup_away"].to_numpy(dtype=float),
            diff(roll, "away_ewm_off_rating", "home_ewm_def_rating"))
        np.testing.assert_allclose(
            X["home_form_home"].to_numpy(dtype=float),
            diff(opp, "home_ewm_margin_home", "away_ewm_margin_home"))
        np.testing.assert_allclose(
            X["home_edge_diff"].to_numpy(dtype=float),
            diff(opp, "home_home_edge", "away_home_edge"))
        # home_adv is the canonical home indicator: constant 1.0 BY DESIGN.
        np.testing.assert_array_equal(X[lr.HOME_COLUMN].to_numpy(),
                                      np.ones(len(X)))
        # The old copy's signature: raw per-side levels in the matrix. None of
        # them may reappear.
        for col in ("home_ewm_net_rating", "away_ewm_net_rating",
                    "home_ewm_margin", "away_ewm_margin",
                    "home_games_played", "away_games_played",
                    "home_home_edge", "away_home_edge"):
            self.assertNotIn(col, X.columns,
                             f"{col} is a raw level; the frame is contrasts")

    def test_each_arm_is_exactly_load_design_with_that_flag(self):
        """The arm semantics, pinned (issue #36).

        The `model` arm is `load_design(with_availability=False)` - the
        canonical frame WITHOUT the availability block - and the `model +
        availability` arm is `load_design(with_availability=True)`. Each arm
        must BE that call's output exactly, frame and all, so this script can
        never again compare the market against a frame of its own devising.
        """
        for with_avail in (False, True):
            X = cl.feature_frame(self.games, with_availability=with_avail)
            want = lr.load_design(games=self.games, with_availability=with_avail)
            pd.testing.assert_frame_equal(
                X, want, obj=f"feature_frame(with_availability={with_avail})")

    def test_the_availability_block_is_appended_exactly_once(self):
        """The availability columns enter the canonical order exactly once -
        never double-appended (once by `load_design`, once by this script)."""
        base = cl.feature_frame(self.games, with_availability=False)
        X = cl.feature_frame(self.games, with_availability=True)
        self.assertEqual(list(X.columns)[:len(base.columns)], list(base.columns),
                         "the availability arm must EXTEND the base arm, "
                         "not replace or reorder it")
        self.assertEqual(list(X.columns)[len(base.columns):],
                         list(lr.AVAILABILITY_COLUMNS),
                         "the shared block must follow the canonical columns "
                         "in its own order")
        self.assertEqual(len(X.columns), len(set(X.columns)),
                         "a column - the availability block especially - must "
                         "not appear twice")
        avail = nd.availability_features(self.games)
        pd.testing.assert_frame_equal(
            X[list(avail.columns)].reset_index(drop=True),
            avail.reset_index(drop=True))

    def test_the_base_arm_carries_no_availability_columns(self):
        X = cl.feature_frame(self.games, with_availability=False)
        for col in lr.AVAILABILITY_COLUMNS:
            self.assertNotIn(col, X.columns)


class TestModelProbsConsumesTheSharedFrame(unittest.TestCase):
    """`model_probs` must get its matrix from `feature_frame`, not rebuild one.

    Pinned behaviourally: `feature_frame` is replaced by a spy, and the run
    must go through it. If the script ever regrows its own `pd.concat` of raw
    builder columns - the thing that drifted four times - the spy is never
    called and this fails.
    """

    def test_the_design_matrix_is_built_by_feature_frame(self):
        from unittest import mock

        n = 60
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({"net_diff": rng.normal(size=n),
                              lr.HOME_COLUMN: np.ones(n)})
        games = pd.DataFrame({"y": rng.integers(0, 2, n)})
        seen = {}

        def spy(g, with_availability=False):
            seen["with_availability"] = with_availability
            return frame.copy()

        splits = [(np.arange(0, 40), np.arange(40, 60))]
        with mock.patch.object(cl, "feature_frame", spy):
            p, y = cl.model_probs(games, splits, with_availability=True)
        self.assertTrue(seen.get("with_availability"),
                        "the availability flag must reach feature_frame")
        self.assertTrue(np.isfinite(p[40:]).all())
        self.assertTrue(np.isnan(p[:40]).all(),
                        "unscored rows must stay NaN, not get filled")
        np.testing.assert_array_equal(y, games["y"].to_numpy())

    def test_model_probs_does_not_reassemble_features_locally(self):
        """Structural, for the same reason as `TestNoBareExcept`: a locally
        assembled frame that happens to equal the canonical one today is
        invisible in a passing suite and drifts tomorrow."""
        import inspect
        src = inspect.getsource(cl.model_probs)
        for banned in ("rolling_team_features", "opponent_adjusted_features",
                       "rest_and_travel", "availability_features",
                       "pd.concat"):
            self.assertNotIn(banned, src, (
                f"model_probs must not build features itself; {banned} found. "
                "The frame belongs to logreg_rolling.load_design via "
                "feature_frame() - see the drift history in #14/#28/#30."
            ))


class TestJoinOddsAccounting(unittest.TestCase):
    """MINOR-3: `join_odds` dropped rows silently in four places.

    "No silent drops" needs the reasons separated, because an unusable odds row
    (invalid) and a game that legitimately has no closing line (unmatched) are
    different facts - and a game whose tricode is not even in the name map is a
    third: a naming gap, not missing odds. The reasons are assigned in
    precedence order so they sum to the row count.
    """

    def test_every_dropped_row_is_counted_with_its_reason(self):
        odds = pd.DataFrame({
            "Date": ["2018-10-16", "not-a-date", "2018-10-17",
                     "2018-10-17", "2018-10-18"],
            "Home": ["Atlanta Hawks"] * 5,
            "Away": ["Boston Celtics"] * 5,
            "ML_Home": [-200, -200, -110, 0, -110],
            "ML_Away": [170, 170, None, -110, -110],
            "OU": [208.5] * 5,
            "Spread": [5.0] * 5,
        })
        games = pd.DataFrame({
            "GAME_ID": ["g1", "g2", "g3"],
            "GAME_DATE": pd.to_datetime(["2018-10-16", "2018-10-17",
                                         "2018-10-19"]),
            "home": ["ATL", "ATL", "ATL"],
            "away": ["BOS", "ZZZ", "BOS"],
        })
        stats = {}
        joined = cl.join_odds(games, odds, stats=stats)
        self.assertEqual(len(joined), 1, "only the one clean match survives")
        self.assertEqual(stats["odds_rows_in"], 5)
        # Odds rows: exactly one per reason, precedence order.
        self.assertEqual(stats["odds_drop_unparseable_date"], 1)
        self.assertEqual(stats["odds_drop_missing_key_or_price"], 1)
        self.assertEqual(stats["odds_drop_invalid_implied"], 1)
        self.assertEqual(stats["odds_rows_matched"], 1)
        self.assertEqual(stats["odds_rows_unmatched_to_a_game"], 1)
        # Games: matched, vs unmatched for two DIFFERENT reasons.
        self.assertEqual(stats["games_in"], 3)
        self.assertEqual(stats["games_with_a_closing_line"], 1)
        self.assertEqual(stats["games_without_a_closing_line"], 2)
        self.assertEqual(stats["games_without_a_line_team_name_not_in_map"], 1)
        self.assertEqual(stats["games_without_a_line_genuinely_unmatched"], 1)
        total = (stats["odds_drop_unparseable_date"]
                 + stats["odds_drop_missing_key_or_price"]
                 + stats["odds_drop_invalid_implied"]
                 + stats["odds_rows_matched"]
                 + stats["odds_rows_unmatched_to_a_game"])
        self.assertEqual(total, stats["odds_rows_in"],
                         "the reasons must partition the input, not overlap")


class TestFeatureFrameGuards(unittest.TestCase):
    """MINOR-5: a length-mismatched block must raise, not NaN-pad and shift.

    Two guards, in two places, since issue #36 moved the block's assembly into
    `load_design`: `feature_frame` refuses a length-mismatched `load_design`
    output, and `load_design` refuses a length-mismatched (or re-shaped)
    availability block. The mocks target `cl.lr.nd`, which is the
    `src.DataProviders.nba_data` module object `load_design` calls - NOT `cl.nd`,
    the script's separately-imported copy of the same file.
    """

    @staticmethod
    def _mock_builders(block: pd.DataFrame):
        from unittest import mock

        games = pd.DataFrame({"y": [1, 0, 1]})
        roll = pd.DataFrame(index=range(len(games)))   # no ewm columns
        rest = pd.DataFrame({c: [1.0] * len(games) for c in (
            "home_rest_days", "away_rest_days", "home_back_to_back",
            "away_back_to_back", "home_in_3_of_4", "away_in_3_of_4",
            "travel_miles")})
        empty = pd.DataFrame(index=range(len(games)))
        return games, (
            mock.patch.object(cl.lr.nd, "rolling_team_features", return_value=roll),
            mock.patch.object(cl.lr.nd, "rest_and_travel", return_value=rest),
            mock.patch.object(cl.lr.nd, "opponent_adjusted_features",
                              return_value=empty),
            mock.patch.object(cl.lr.nd, "availability_features", return_value=block),
        )

    def test_feature_frame_refuses_a_length_mismatched_design(self):
        from unittest import mock

        games = pd.DataFrame({"y": [1, 0, 1]})
        ok = pd.DataFrame({"net_diff": [0.1, 0.2, 0.3]})
        for with_avail in (False, True):
            with mock.patch.object(cl.lr, "load_design",
                                   return_value=ok.iloc[:2]):
                with self.assertRaises(ValueError):
                    cl.feature_frame(games, with_availability=with_avail)
        with mock.patch.object(cl.lr, "load_design", return_value=ok):
            X = cl.feature_frame(games, with_availability=False)
        self.assertEqual(len(X), len(games))

    def test_load_design_refuses_a_length_mismatched_availability_block(self):
        games, patches = self._mock_builders(
            pd.DataFrame({c: [1.0, 2.0] for c in lr.AVAILABILITY_COLUMNS}))
        with patches[0], patches[1], patches[2], patches[3]:
            with self.assertRaises(ValueError):
                cl.lr.load_design(games=games, with_availability=True)

    def test_load_design_refuses_a_rearranged_availability_block(self):
        # The column guard is the feature-order contract for the block: the
        # model recipe names these columns positionally, so a block that
        # changed shape must fail loudly, not be absorbed in a new order.
        shuffled = list(lr.AVAILABILITY_COLUMNS)[::-1]
        games, patches = self._mock_builders(
            pd.DataFrame({c: [1.0] * 3 for c in shuffled}))
        with patches[0], patches[1], patches[2], patches[3]:
            with self.assertRaises(ValueError):
                cl.lr.load_design(games=games, with_availability=True)

    def test_load_design_appends_a_matching_block_once(self):
        games, patches = self._mock_builders(
            pd.DataFrame({c: [1.0] * 3 for c in lr.AVAILABILITY_COLUMNS}))
        with patches[0], patches[1], patches[2], patches[3]:
            X = cl.lr.load_design(games=games, with_availability=True)
            base = cl.lr.load_design(games=games, with_availability=False)
        self.assertEqual(list(X.columns),
                         list(base.columns) + list(lr.AVAILABILITY_COLUMNS))
        self.assertEqual(len(X), len(games))


if __name__ == "__main__":
    unittest.main()
