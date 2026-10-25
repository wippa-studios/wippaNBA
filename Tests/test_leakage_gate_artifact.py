"""The reusable leakage gate, tested against the three leaks it exists to catch.

WHY THIS FILE EXISTS
--------------------
``src/Utils/leakage_gate.py`` is the artifact issue #46 asked for: one importable
function that reports whether a feature builder uses information from after each
row's date. A gate never shown to fire on a known leak is an untested gate, so
this file pins six fixtures with no download and no network:

======================  ============================================  ==============
fixture                 the failure it encodes                         must report
======================  ============================================  ==============
``headline_leak``       the 68.9% headline: a same-day snapshot that   blind
                        already contains the game's result
``headline_clean``      the same feature against the prior snapshot     nothing
``elo_actual``          player-Elo summed over players who ACTUALLY    blind
                        played -- the |z| = 5.18 membership leak
``elo_prior``           the same ratings over the previous game's       nothing
                        squad (the honest construction)
``availability_actual`` expected minutes over tonight's actual squad   blind
``availability_strict`` expected minutes over the previous squad        nothing
======================  ============================================  ==============

Plus the two properties issue #48 added:

* ``rolling_pre`` is strictly causal and reads its own row's pre-game value.
  A trailing mean must **pass the blind probe**; under the old blind design it
  was flagged because its own input was deleted. It is allowed to appear in the
  optional ``self_dependencies`` diagnostic, and must never appear in ``leaks``.
* ``tomorrows_label`` reads a later row. That must land in ``leaks`` (the
  ``as_of`` probe), not in ``membership_dependencies``.

The fixtures are synthetic on purpose: the gate's scope note forbids a dependency
on ``nba_data`` or the 382 MB download, so it can be pointed at another project.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.Utils.leakage_gate import (
    ColumnFinding,
    LeakageReport,
    audit_future_dependence,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# The synthetic universe
# --------------------------------------------------------------------------- #
# Six dates, seven games, one same-date doubleheader (g5/g6). Every squad and
# every value is a function of (player, game), so a value attached to the wrong
# game is numerically visible rather than hidden by a constant.
GAMES = [
    # id,   date,        home,  away,  label, pre-game strength
    ("g1", "2024-11-01", "AAA", "BBB", 1, 0.8),
    ("g2", "2024-11-03", "CCC", "DDD", 0, -0.4),
    ("g3", "2024-11-05", "AAA", "CCC", 1, 0.5),
    ("g4", "2024-11-08", "BBB", "DDD", 1, 0.2),
    ("g5", "2024-11-10", "AAA", "DDD", 1, 0.9),
    ("g6", "2024-11-10", "BBB", "CCC", 0, -0.3),
    ("g7", "2024-11-14", "AAA", "BBB", 0, 0.1),
]

#: Squads change between games. 5 and 6 replace 5 on AAA; 16 and 17 replace 14
#: and 15 on BBB. That is what makes "actual participants" differ from "the
#: previous game's squad" -- without it the leak and the clean arm coincide.
SQUADS = {
    "g1": {"AAA": [1, 2, 3, 4, 5], "BBB": [11, 12, 13, 14, 15]},
    "g2": {"CCC": [21, 22, 23, 24, 25], "DDD": [31, 32, 33, 34, 35]},
    "g3": {"AAA": [1, 2, 3, 4, 6], "CCC": [21, 22, 23, 24, 26]},
    "g4": {"BBB": [11, 12, 13, 16, 17], "DDD": [31, 32, 33, 34, 35]},
    "g5": {"AAA": [1, 2, 6, 7, 8], "DDD": [31, 32, 33, 36, 37]},
    "g6": {"BBB": [11, 12, 16, 17, 18], "CCC": [21, 22, 26, 27, 28]},
    "g7": {"AAA": [1, 3, 6, 7, 8], "BBB": [11, 13, 16, 17, 18]},
}

#: Identity columns the self-probe stub keeps so each builder can recognise the
#: event after its feature inputs have been nulled.
SELF_KEEP = ("record_type", "game_id", "home", "away", "team")


def fixture_frame() -> pd.DataFrame:
    """One heterogeneous record universe: games, snapshots, and box scores.

    ``known_before`` is the caller's declaration of what was available before the
    first event on a date. Every row here is post-event -- the snapshot contains
    the game's result, and the box score contains who actually played -- except
    the game row's own identity, which the gate always keeps for the probe event.
    """
    rows = []
    wins = {team: 0 for team in ("AAA", "BBB", "CCC", "DDD")}

    for gi, (gid, date, home, away, label, pre) in enumerate(GAMES, start=1):
        rows.append({
            "row_key": gid,
            "date": date,
            "record_type": "game",
            "known_before": False,
            "game_id": gid,
            "home": home,
            "away": away,
            "label": label,
            "home_pre": pre,
            "away_pre": -pre,
        })

        wins[home] += label
        wins[away] += 1 - label

        for team in (home, away):
            rows.append({
                "row_key": f"snap:{date}:{team}",
                "date": date,
                "record_type": "snapshot",
                "known_before": False,
                "team": team,
                "wins_to_date": wins[team],
            })
            for pid in SQUADS[gid][team]:
                rows.append({
                    "row_key": f"box:{gid}:{pid}",
                    "date": date,
                    "record_type": "box",
                    "known_before": False,
                    "game_id": gid,
                    "team": team,
                    "player_id": pid,
                    "minutes": float(12 + ((pid * 3 + gi * 5) % 20)),
                    "rating": float(1500 + ((pid * 7 + gi * 11) % 200)),
                    "expected_minutes": float(10 + ((pid * 5 + gi * 3) % 25)),
                })

    return pd.DataFrame(rows)


def _games(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["record_type"] == "game"].copy()


def _snapshots(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["record_type"] == "snapshot"].copy()


def _boxes(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["record_type"] == "box"].copy()


def _frame(values: dict[str, float], name: str) -> pd.DataFrame:
    return pd.DataFrame({name: values})


# --------------------------------------------------------------------------- #
# Fixture 1: the 68.9% headline -- a same-day snapshot carrying the result
# --------------------------------------------------------------------------- #

def headline_leak(frame: pd.DataFrame) -> pd.DataFrame:
    """Join each game to the snapshot dated the SAME day.

    That snapshot was pulled after tip-off, so ``wins_to_date`` already contains
    the game being predicted. This is the shape of the original headline leak.
    """
    games, snaps = _games(frame), _snapshots(frame)
    out = {}
    for _, game in games.iterrows():
        home = snaps[(snaps["date"] == game["date"]) & (snaps["team"] == game["home"])]
        away = snaps[(snaps["date"] == game["date"]) & (snaps["team"] == game["away"])]
        out[game["row_key"]] = (
            float(home.iloc[0]["wins_to_date"] - away.iloc[0]["wins_to_date"])
            if len(home) and len(away) else np.nan
        )
    return _frame(out, "snapshot_wins_diff")


def headline_clean(frame: pd.DataFrame) -> pd.DataFrame:
    """The same feature against each team's most recent PRIOR snapshot."""
    games, snaps = _games(frame), _snapshots(frame)
    out = {}
    for _, game in games.iterrows():
        values = []
        for team in (game["home"], game["away"]):
            prior = snaps[(snaps["team"] == team) & (snaps["date"] < game["date"])]
            prior = prior.sort_values("date")
            values.append(float(prior.iloc[-1]["wins_to_date"]) if len(prior) else np.nan)
        out[game["row_key"]] = values[0] - values[1]
    return _frame(out, "snapshot_wins_diff")


# --------------------------------------------------------------------------- #
# Fixtures 2 and 3: player-Elo and availability, actual squad vs prior squad
# --------------------------------------------------------------------------- #

def _previous_game(games: pd.DataFrame, gid: str, team: str) -> str | None:
    current = games[games["game_id"] == gid].iloc[0]
    prior = games[
        (games["date"] < current["date"])
        & ((games["home"] == team) | (games["away"] == team))
    ]
    return None if prior.empty else str(prior.sort_values("date").iloc[-1]["game_id"])


def _team_boxes(boxes: pd.DataFrame, gid: str, team: str) -> pd.DataFrame:
    return boxes[(boxes["game_id"] == gid) & (boxes["team"] == team)]


def _player_feature(frame: pd.DataFrame, *, weighting: str, actual_membership: bool,
                    name: str) -> pd.DataFrame:
    games, boxes = _games(frame), _boxes(frame)
    out = {}
    for _, game in games.iterrows():
        totals = []
        for team in (game["home"], game["away"]):
            gid = game["game_id"] if actual_membership else _previous_game(
                games, game["game_id"], team
            )
            if gid is None:
                totals.append(np.nan)
                continue
            part = _team_boxes(boxes, gid, team)
            if part.empty:
                totals.append(np.nan)
            elif weighting == "minutes":
                totals.append(float((part["rating"] * part["minutes"]).sum()))
            else:
                totals.append(float(part["expected_minutes"].sum()))
        out[game["row_key"]] = totals[0] - totals[1]
    return _frame(out, name)


def elo_actual(frame: pd.DataFrame) -> pd.DataFrame:
    """The |z| = 5.18 leak: sum rating x minutes over tonight's participants."""
    return _player_feature(frame, weighting="minutes", actual_membership=True,
                           name="player_elo_actual")


def elo_prior(frame: pd.DataFrame) -> pd.DataFrame:
    """The honest arm: the previous game's squad, a set fixed before tip-off."""
    return _player_feature(frame, weighting="minutes", actual_membership=False,
                           name="player_elo_prior")


def availability_actual(frame: pd.DataFrame) -> pd.DataFrame:
    """Expected minutes over the players who actually played -- contaminated."""
    return _player_feature(frame, weighting="expected", actual_membership=True,
                           name="availability_actual")


def availability_strict(frame: pd.DataFrame) -> pd.DataFrame:
    """Expected minutes over the previous game's squad -- the shipped block."""
    return _player_feature(frame, weighting="expected", actual_membership=False,
                           name="availability_strict")


# --------------------------------------------------------------------------- #
# Known-clean controls
# --------------------------------------------------------------------------- #

def rest(frame: pd.DataFrame) -> pd.DataFrame:
    """Rest-day differential, from the published schedule alone."""
    games = _games(frame)
    out = {}
    for _, game in games.iterrows():
        rest_days = {}
        for team in (game["home"], game["away"]):
            prior = games[
                (games["date"] < game["date"])
                & ((games["home"] == team) | (games["away"] == team))
            ]
            rest_days[team] = (
                (pd.Timestamp(game["date"]) - pd.Timestamp(prior["date"].max())).days
                if len(prior) else np.nan
            )
        out[game["row_key"]] = rest_days[game["home"]] - rest_days[game["away"]]
    return _frame(out, "rest_diff")


def rolling_pre(frame: pd.DataFrame) -> pd.DataFrame:
    """A strictly causal trailing mean that reads its own row's pre-game value.

    This is the #48 fixture. The value is known before tip-off, the window is
    trailing rather than centred, and the feature for game t uses game t's own
    pre-game number. It must pass ``blind``; under the old design (delete the
    probe row) it was flagged because its own input disappeared.
    """
    games = _games(frame)
    per_team: dict[tuple[str, str], float] = {}
    for team in sorted(set(games["home"]) | set(games["away"])):
        team_games = games[(games["home"] == team) | (games["away"] == team)]
        team_games = team_games.sort_values(["date", "game_id"])
        values = [
            row["home_pre"] if row["home"] == team else row["away_pre"]
            for _, row in team_games.iterrows()
        ]
        rolled = pd.Series(values, dtype=float).rolling(3, min_periods=1).mean()
        for (_, row), value in zip(team_games.iterrows(), rolled):
            per_team[(row["game_id"], team)] = float(value)

    out = {}
    for _, game in games.iterrows():
        out[game["row_key"]] = (
            per_team[(game["game_id"], game["home"])]
            - per_team[(game["game_id"], game["away"])]
        )
    return _frame(out, "rolling_pre")


def tomorrows_label(frame: pd.DataFrame) -> pd.DataFrame:
    """A deliberately impossible feature: the next game's result.

    The ``as_of`` probe must catch this one. It is the control that proves the
    sound discriminator is actually wired, not merely documented.
    """
    games = _games(frame)
    out = {}
    for _, game in games.iterrows():
        later = games[games["date"] > game["date"]].sort_values("date")
        out[game["row_key"]] = float(later.iloc[0]["label"]) if len(later) else np.nan
    return _frame(out, "tomorrows_label")


# --------------------------------------------------------------------------- #
# The gate call used by most tests
# --------------------------------------------------------------------------- #

def audit(builder, frame: pd.DataFrame | None = None, **kwargs) -> LeakageReport:
    kwargs.setdefault("include_self_probe", True)
    return audit_future_dependence(
        builder,
        fixture_frame() if frame is None else frame,
        "date",
        "row_key",
        known_before_col="known_before",
        self_keep_cols=SELF_KEEP,
        **kwargs,
    )


def names(findings) -> set[str]:
    return {finding.feature for finding in findings}


# --------------------------------------------------------------------------- #
# #46: fires on the three known leaks
# --------------------------------------------------------------------------- #

class TestKnownLeaksFire(unittest.TestCase):
    """A gate never shown to fire on a known leak is an untested gate."""

    def test_the_689_headline_same_day_snapshot_is_flagged(self):
        report = audit(headline_leak)
        self.assertIn("snapshot_wins_diff", names(report.membership_dependencies))
        self.assertEqual(report.leaks, (),
                         "a same-day snapshot is not a later-dated dependency; "
                         "the sound as_of probe must stay quiet")
        self.assertFalse(report.clean)
        self.assertTrue(report.needs_investigation)

    def test_player_elo_over_actual_participants_is_flagged(self):
        report = audit(elo_actual)
        self.assertIn("player_elo_actual", names(report.membership_dependencies))
        self.assertEqual(report.leaks, ())
        self.assertFalse(report.clean)

    def test_availability_over_the_actual_squad_is_flagged(self):
        report = audit(availability_actual)
        self.assertIn("availability_actual", names(report.membership_dependencies))
        self.assertEqual(report.leaks, ())
        self.assertFalse(report.clean)

    def test_the_flags_name_the_mechanism_not_just_a_column(self):
        """A finding carries probes, changed count, and a worked example."""
        finding = audit(elo_actual).membership_dependencies[0]
        self.assertGreater(finding.changed, 0)
        self.assertGreaterEqual(finding.changed, 1)
        self.assertLessEqual(finding.changed, finding.probes)
        self.assertTrue(finding.examples)
        self.assertEqual(finding.variant, "blind")
        self.assertFalse(finding.is_leak)


# --------------------------------------------------------------------------- #
# #46: passes on known-clean features
# --------------------------------------------------------------------------- #

class TestKnownCleanPass(unittest.TestCase):

    def assertClean(self, builder):
        report = audit(builder)
        self.assertEqual(report.leaks, (), report.summary())
        self.assertEqual(report.membership_dependencies, (), report.summary())
        self.assertTrue(report.clean, report.summary())
        self.assertTrue(report.future_dependence_free)
        return report

    def test_the_689_feature_on_a_prior_snapshot_passes(self):
        self.assertClean(headline_clean)

    def test_player_elo_over_the_previous_squad_passes(self):
        self.assertClean(elo_prior)

    def test_the_availability_strict_roster_construction_passes(self):
        self.assertClean(availability_strict)

    def test_rest_and_travel_passes(self):
        self.assertClean(rest)

    def test_a_self_referential_trailing_mean_passes_the_blind_probe(self):
        """#48's fixture: causal, reads its own row, must not be a leak.

        It is expected to move under the optional self probe -- that is what
        "reads its own row" means -- and the point of the split is that the
        self finding is a different collection and a different verdict.
        """
        report = self.assertClean(rolling_pre)
        self.assertIn("rolling_pre", names(report.self_dependencies))
        self.assertNotIn("rolling_pre", names(report.membership_dependencies))
        self.assertNotIn("rolling_pre", names(report.leaks))
        self.assertIn("diagnostics", report.summary())
        self.assertIn("PASS on both probes", report.summary())


# --------------------------------------------------------------------------- #
# #48: the API keeps the two questions apart
# --------------------------------------------------------------------------- #

class TestReportSeparatesTheQuestions(unittest.TestCase):

    def test_a_later_dated_dependency_is_a_leak_not_a_membership_finding(self):
        report = audit(tomorrows_label)
        self.assertIn("tomorrows_label", names(report.leaks))
        self.assertNotIn("tomorrows_label", names(report.membership_dependencies))
        self.assertFalse(report.clean)
        self.assertIn("LEAK CONFIRMED", report.summary())

    def test_leaks_and_membership_dependencies_are_distinct_collections(self):
        report = audit(headline_leak)
        self.assertIsInstance(report.leaks, tuple)
        self.assertIsInstance(report.membership_dependencies, tuple)
        self.assertIsInstance(report.self_dependencies, tuple)
        self.assertFalse(hasattr(report, "findings"),
                         "one findings list with a variant tag is the shape #48 "
                         "asked to remove")

    def test_findings_carry_their_variant_and_is_leak_only_for_as_of(self):
        leak = audit(tomorrows_label).leaks[0]
        blind = audit(headline_leak).membership_dependencies[0]
        self.assertEqual(leak.variant, "as_of")
        self.assertTrue(leak.is_leak)
        self.assertEqual(blind.variant, "blind")
        self.assertFalse(blind.is_leak)

    def test_the_summary_names_the_distinction_in_words(self):
        membership = audit(headline_leak).summary()
        self.assertIn("membership dependencies: 1", membership)
        self.assertIn("not proof of leakage", membership)
        self.assertNotIn("LEAK CONFIRMED", membership)

        leak = audit(tomorrows_label).summary()
        self.assertIn("LEAK CONFIRMED", leak)
        self.assertIn("Do not trust scores", leak)

    def test_the_self_probe_is_opt_in_and_never_promoted(self):
        without = audit(rolling_pre, include_self_probe=False)
        self.assertEqual(without.self_dependencies, ())
        self.assertTrue(without.clean)

        with_self = audit(rolling_pre, include_self_probe=True)
        self.assertIn("rolling_pre", names(with_self.self_dependencies))
        self.assertEqual(with_self.leaks, ())
        self.assertEqual(with_self.membership_dependencies, ())


# --------------------------------------------------------------------------- #
# Contract and failure modes
# --------------------------------------------------------------------------- #

class TestGateContract(unittest.TestCase):

    def test_a_duplicate_row_key_is_refused(self):
        frame = fixture_frame()
        frame.loc[frame.index[-1], "row_key"] = frame.iloc[0]["row_key"]
        with self.assertRaisesRegex(ValueError, "must be unique"):
            audit(headline_clean, frame=frame)

    def test_a_non_numeric_feature_is_refused_with_its_name(self):
        def builder(frame):
            games = _games(frame)
            return pd.DataFrame({"team": games["home"].to_numpy()}, index=games["row_key"])

        with self.assertRaisesRegex(TypeError, "not numeric"):
            audit(builder)

    def test_a_feature_key_that_is_not_in_the_frame_is_refused(self):
        def builder(frame):
            games = _games(frame)
            return pd.DataFrame({"x": np.arange(len(games), dtype=float)},
                                index=[f"ghost:{i}" for i in range(len(games))])

        with self.assertRaisesRegex(ValueError, "cannot be dated"):
            audit(builder)

    def test_an_unparseable_date_is_refused(self):
        frame = fixture_frame()
        frame.loc[frame.index[0], "date"] = "not-a-date"
        with self.assertRaisesRegex(ValueError, "unparseable dates"):
            audit(headline_clean, frame=frame)

    def test_an_empty_frame_is_refused(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            audit(headline_clean, frame=fixture_frame().iloc[0:0])

    def test_a_missing_known_before_column_is_refused(self):
        with self.assertRaisesRegex(KeyError, "known_before_col"):
            audit_future_dependence(
                headline_clean, fixture_frame(), "date", "row_key",
                known_before_col="absent",
            )

    def test_float_noise_below_tolerance_is_not_a_finding(self):
        """Adding rows changes groupby summation order by ~1e-12, not a leak."""
        def noisy(frame):
            games = _games(frame)
            # Every value is written with a last-bit perturbation that depends on
            # how many rows the rebuild contains: exactly the summation-order
            # artefact the tolerance exists for.
            bump = 0.0 if len(frame) > 20 else 1e-13
            return _frame(
                {row["row_key"]: float(len(frame)) * 0.0 + bump
                 for _, row in games.iterrows()},
                "noisy",
            )

        report = audit(noisy, tol=1e-9)
        self.assertEqual(report.leaks, ())
        self.assertEqual(report.membership_dependencies, ())

    def test_a_real_move_above_tolerance_is_a_finding(self):
        def tiny_leak(frame):
            games = _games(frame)
            out = {}
            for _, game in games.iterrows():
                later = games[games["date"] > game["date"]].sort_values("date")
                out[game["row_key"]] = 1e-6 if len(later) else 0.0
            return _frame(out, "tiny")

        report = audit(tiny_leak, tol=1e-9)
        self.assertIn("tiny", names(report.leaks))

    def test_a_probe_that_cannot_be_built_is_skipped_loudly(self):
        def fragile(frame):
            if len(frame) < len(fixture_frame()):
                raise RuntimeError("needs the whole frame")
            return headline_clean(frame)

        report = audit(fragile, on_error="skip")
        self.assertEqual(report.probes, 0)
        self.assertTrue(report.inconclusive)
        self.assertFalse(report.clean)
        self.assertTrue(report.skipped)
        self.assertIn("INCONCLUSIVE", report.summary())
        self.assertIn("RuntimeError", report.summary())

    def test_on_error_raise_propagates(self):
        def fragile(frame):
            if len(frame) < len(fixture_frame()):
                raise RuntimeError("needs the whole frame")
            return headline_clean(frame)

        with self.assertRaisesRegex(RuntimeError, "needs the whole frame"):
            audit(fragile, on_error="raise")

    def test_probe_selection_is_seeded_and_capped(self):
        first = audit(headline_clean, n_probe=3, seed=7)
        second = audit(headline_clean, n_probe=3, seed=7)
        self.assertEqual(first.probes, second.probes)
        self.assertEqual(first.probes, 3)
        self.assertLessEqual(first.probes, len({row[1] for row in GAMES}))

    def test_the_report_exposes_the_probe_count_and_rows(self):
        report = audit(headline_clean)
        self.assertEqual(report.rows_available, len(GAMES))
        self.assertEqual(report.features, 1)
        self.assertEqual(report.probes, len({row[1] for row in GAMES}))


class TestImportIsIndependentOfTheRepoData(unittest.TestCase):
    """The scope note in #46: no ``nba_data``, no download, no repo layout."""

    def test_importing_the_gate_does_not_import_nba_data(self):
        code = (
            "import sys;"
            "import src.Utils.leakage_gate as g;"
            "print('nba_data' in sys.modules);"
            "print(hasattr(g, 'audit_future_dependence'))"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        imported, has_function = result.stdout.strip().splitlines()
        self.assertEqual(imported, "False")
        self.assertEqual(has_function, "True")


if __name__ == "__main__":
    unittest.main()
