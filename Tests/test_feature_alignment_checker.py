"""Issue #14: the alignment checker must validate the contract the trainers use.

``scripts/check_feature_alignment.py`` is the script that exists to catch a
SILENT misalignment -- XGBoost and Keras are fed a positional matrix, so a
frame with the right columns in the wrong order raises nothing and predicts
wrong. ``src/Train-Models/training_common.py`` records that this has already
happened once: the two totals trainers disagreed about whether the closing total
``OU`` is a feature, and the runner fed them an order neither had trained on.

The checker had drifted from the contract it was written to check. Measured
against the live ``dataset_2012-26`` table (117 columns):

    checker's DROP_COLUMNS_ML   10 entries, 8 of them real -> 109 features
    checker's DROP_COLUMNS_UO    9 entries, 7 of them real -> 110 features
    drop_columns_for(MONEYLINE) 11 entries, all real     -> 106 features
    drop_columns_for(TOTALS)    10 entries, all real     -> 107 features

109 and 110 match no committed model -- ``Models/XGBoost_Models/*.json`` were
fitted on 106 and 107 -- so the script was reporting two widths that do not
exist. What it got wrong, against the live table:

  * its TOTALS list did not drop the closing total ``OU``, so it expected the
    totals model to be fed 110 columns when it is fed 107. ``load_live_frames``
    copies the moneyline frame and re-attaches ``OU`` to it for the totals
    comparison, so every live run reported that ``OU`` as an EXTRA column, and
    the 104th input -- ``OU`` itself, in its dataset position between the
    away-team block and ``OU-Cover`` -- was absent from the expectation it was
    checking the frame against. That is the exact historical defect, reversed.
  * neither list dropped ``index``, ``index.1`` or ``GameDate``, which every
    trainer drops: three phantom model inputs.
  * both lists named ``TEAM_ID``/``TEAM_ID.1``, columns ``Create_Games`` strips
    before writing the table, so those entries matched nothing and
    ``117 - 10`` never reconciled.

Six of its ten entries were still correct, which is the worse shape: the checker
agreed with the trainers on most of the contract and disagreed exactly where a
permutation could hide. So the two lists are now the trainer function's own
return values, and what is pinned below is:

  a. both lists equal the trainers' -- LIST AND ORDER, order first, because a
     set comparison would pass on a list that is right and reversed;
  b. ``OU`` is absent from the moneyline list and present in the totals list,
     which is the specific regression and the one most likely to come back;
  c. no literal drop list survives anywhere in the script, so the copy cannot
     quietly start drifting again;
  d. the feature widths the script prints are 106 and 107 on the real table,
     and match the committed boosters.

HOW THE HYPHEN IS HANDLED
=========================
``src/Train-Models`` contains a hyphen and can never be an importable package,
so it goes on ``sys.path`` and ``training_common`` imports under its plain
top-level name -- the convention its own docstring describes. The script itself
is loaded by file path under a synthetic name, as ``pyproject.toml`` states for
everything in ``scripts/``.
"""

import contextlib
import importlib.util
import io
import sqlite3
import subprocess
import sys
import unittest
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_MODELS_DIR = REPO_ROOT / "src" / "Train-Models"
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_feature_alignment.py"
DATASET_DB = REPO_ROOT / "Data" / "dataset.sqlite"
MODEL_DIR = REPO_ROOT / "Models"

for _path in (REPO_ROOT, TRAIN_MODELS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import training_common  # noqa: E402

from src.Utils import features  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("check_feature_alignment", SCRIPT_PATH)
checker = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(checker)

MONEYLINE = training_common.drop_columns_for(training_common.MONEYLINE_TARGET)
TOTALS = training_common.drop_columns_for(training_common.TOTALS_TARGET)

# Every column that must never reach a model, whatever the target. Taken from
# training_common's own constants where it exports them, spelled out for the
# rest so that a rename in training_common fails here loudly rather than making
# this list quietly wrong.
ALWAYS_DROPPED = (
    "index",
    "index.1",
    "Score",
    "Date",
    "Date.1",
    training_common.GAME_DATE_COLUMN,
    training_common.TOTALS_TARGET,
    "TEAM_NAME",
    "TEAM_NAME.1",
)


class DropListEqualityMixin:
    """Order-sensitive list equality that reports the first position to differ."""

    def assertSameList(self, name, expected, actual):
        self.assertEqual(
            len(actual),
            len(expected),
            f"{name} has {len(actual)} entries, the trainers use {len(expected)}",
        )
        for position, (want, got) in enumerate(zip(expected, actual)):
            self.assertEqual(
                got,
                want,
                f"{name} diverges from the trainers at position {position} "
                f"(and possibly at every position after it): the trainers drop "
                f"{want!r}, the checker drops {got!r}. Order is part of the "
                f"contract, so this is not a cosmetic difference.",
            )


# ---------------------------------------------------------------------------
# #14  the checker's drop lists ARE the trainers'
# ---------------------------------------------------------------------------


class TestCheckerUsesTheTrainersDropLists(DropListEqualityMixin, unittest.TestCase):
    """Acceptance criteria 1 and 2: one source of truth, list and order."""

    def test_the_moneyline_list_is_the_trainers_own(self):
        self.assertSameList("DROP_COLUMNS_ML", MONEYLINE, checker.DROP_COLUMNS_ML)

    def test_the_totals_list_is_the_trainers_own(self):
        self.assertSameList("DROP_COLUMNS_UO", TOTALS, checker.DROP_COLUMNS_UO)

    def test_the_closing_total_is_dropped_for_moneyline_and_kept_for_totals(self):
        # THE regression this issue found, and the one most likely to come back.
        # `OU` is the closing total: known before tip-off, so it is a legitimate
        # input for the TOTALS model and never wanted by the moneyline one. The
        # drifted checker had it exactly backwards for the totals comparison --
        # its DROP_COLUMNS_UO left OU out, so the script expected 110 totals
        # inputs where the model is fitted on 107, and reported the OU that
        # load_live_frames deliberately re-attaches as an EXTRA column.
        self.assertIn(
            training_common.CLOSING_TOTAL_COLUMN,
            MONEYLINE,
            "training_common no longer drops OU for the moneyline target; "
            "this test needs re-measuring, not deleting",
        )
        self.assertNotIn(
            training_common.CLOSING_TOTAL_COLUMN,
            TOTALS,
            "training_common now drops OU for the totals target too",
        )
        self.assertIn(
            training_common.CLOSING_TOTAL_COLUMN,
            checker.DROP_COLUMNS_ML,
            "the moneyline list must drop OU, which every moneyline trainer drops",
        )
        self.assertNotIn(
            training_common.CLOSING_TOTAL_COLUMN,
            checker.DROP_COLUMNS_UO,
            "the totals list must NOT drop OU: it is the totals model's input",
        )
        # The two lists differ by exactly that one entry, and by nothing else.
        self.assertEqual(
            [
                column
                for column in checker.DROP_COLUMNS_ML
                if column not in checker.DROP_COLUMNS_UO
            ],
            [training_common.CLOSING_TOTAL_COLUMN],
        )
        self.assertEqual(
            [
                column
                for column in checker.DROP_COLUMNS_UO
                if column not in checker.DROP_COLUMNS_ML
            ],
            [],
            "the totals list drops something the moneyline list keeps",
        )

    def test_the_keys_every_trainer_drops_are_dropped_here_too(self):
        # The columns the drifted copy omitted: index, index.1 and GameDate were
        # absent from both of its lists, so the script counted three phantom
        # model inputs that the trainers never see.
        for name, drop in (
            ("DROP_COLUMNS_ML", checker.DROP_COLUMNS_ML),
            ("DROP_COLUMNS_UO", checker.DROP_COLUMNS_UO),
        ):
            for column in ("index", "index.1", training_common.GAME_DATE_COLUMN):
                self.assertIn(
                    column,
                    drop,
                    f"{name} no longer drops {column!r}, which every trainer drops",
                )

    def test_no_column_the_list_names_is_absent_from_the_shipped_table(self):
        # The reason the old count never reconciled: TEAM_ID/TEAM_ID.1 are not
        # columns of dataset_2012-26, because Create_Games strips them before
        # writing. A drop list that names a column the table does not have is
        # counting against a dataset that is not the one being measured.
        if not DATASET_DB.exists():
            self.skipTest("dataset.sqlite is unavailable")
        columns = checker.get_table_columns(training_common.DEFAULT_DATASET)
        for name, drop in (
            ("DROP_COLUMNS_ML", checker.DROP_COLUMNS_ML),
            ("DROP_COLUMNS_UO", checker.DROP_COLUMNS_UO),
        ):
            for column in drop:
                self.assertIn(
                    column,
                    columns,
                    f"{name} names {column!r}, which the shipped table does not have",
                )

    def test_every_drop_list_in_the_script_is_derived_from_the_trainer(self):
        # Not a value assertion -- a structural one. Both lists are the RETURN
        # VALUE of a `training_common.drop_columns_for(...)` call, so a future
        # edit that hardcodes a literal again (or copies one in from a trainer
        # and hand-edits it) fails here instead of drifting silently, which is
        # what the issue was about.
        import ast

        source = SCRIPT_PATH.read_text()
        tree = ast.parse(source)
        assigned = {
            node.targets[0].id: node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        }

        # Each list must be the call for the right target, with no keywords.
        # `include_team_ids` is deliberately left off: the four XGBoost/NN
        # trainers pass none, and on the shipped table the entry is a no-op
        # either way, so a keyword here would mean the checker had been pointed
        # at the other pair of trainers without saying so.
        expected_target = {
            "DROP_COLUMNS_ML": "MONEYLINE_TARGET",
            "DROP_COLUMNS_UO": "TOTALS_TARGET",
        }
        for name, attribute in expected_target.items():
            self.assertIn(name, assigned, f"{name} is no longer a module constant")
            value = assigned[name]
            self.assertIsInstance(
                value,
                ast.Call,
                f"{name} is a literal again; it must be a call into "
                f"training_common.drop_columns_for",
            )
            self.assertIsInstance(value.func, ast.Attribute, name)
            self.assertEqual(
                getattr(value.func.value, "id", None),
                "training_common",
                f"{name} calls something other than the trainer's function",
            )
            self.assertEqual(value.func.attr, "drop_columns_for", name)
            self.assertEqual(len(value.args), 1, f"{name} passes more than one target")
            argument = value.args[0]
            self.assertIsInstance(argument, ast.Attribute, f"{name}: target is a literal")
            self.assertEqual(getattr(argument.value, "id", None), "training_common", name)
            self.assertEqual(
                argument.attr,
                attribute,
                f"{name} asks drop_columns_for for the wrong target",
            )
            self.assertFalse(
                value.keywords,
                f"{name} passes keywords to drop_columns_for; the checker's lists "
                f"must match the four XGBoost/NN trainers, which pass none",
            )
            # And the constant on the module agrees with what that call returns.
            expected = training_common.drop_columns_for(
                getattr(training_common, attribute)
            )
            self.assertSameList(name, expected, getattr(checker, name))

        # And no list literal named like a drop list survives anywhere in the
        # file, including inside a function body.
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and "DROP_COLUMNS" in target.id:
                    self.assertNotIsInstance(
                        node.value,
                        (ast.List, ast.Tuple, ast.Set),
                        f"{target.id} is a literal collection again",
                    )

    def test_the_default_dataset_is_the_one_the_trainers_read(self):
        # The parser default was "dataset_2012-26_new", a table that does not
        # exist in Data/dataset.sqlite, so a bare `python scripts/
        # check_feature_alignment.py` died inside sqlite3 before printing a
        # single number. Same class of drift this file was fixed for: a constant
        # spelled out here that no trainer reads.
        self.assertEqual(
            training_common.DEFAULT_DATASET,
            "dataset_2012-26",
            "training_common.DEFAULT_DATASET moved; re-measure this checker",
        )

        # Checked structurally rather than by scanning the text, so the comment
        # explaining the old default cannot fail the test that forbids it.
        import ast

        tree = ast.parse(SCRIPT_PATH.read_text())
        defaults = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "--dataset"
        ]
        self.assertEqual(len(defaults), 1, "the --dataset flag is gone or duplicated")
        keywords = {kw.arg: kw.value for kw in defaults[0].keywords}
        self.assertIn("default", keywords, "--dataset no longer has a default")
        default = keywords["default"]
        self.assertIsInstance(
            default,
            ast.Attribute,
            "the --dataset default is a literal again; it must be "
            "training_common.DEFAULT_DATASET",
        )
        self.assertEqual(getattr(default.value, "id", None), "training_common")
        self.assertEqual(default.attr, "DEFAULT_DATASET")

    def test_a_dataset_name_that_does_not_exist_says_so(self):
        # The default above used to reach sqlite3 as `no such table:`, which
        # names neither the file it looked in nor what was available. The
        # trainer's guard reports both, and refuses a leaky table by name.
        if not DATASET_DB.exists():
            self.skipTest("dataset.sqlite is unavailable")
        with self.assertRaises(LookupError) as caught:
            checker.get_table_columns("dataset_2012-26_new")
        message = str(caught.exception)
        self.assertIn("dataset.sqlite", message)
        self.assertIn(training_common.DEFAULT_DATASET, message)
        for table in training_common.LEAKED_DATASET_TABLES:
            with self.assertRaises(LookupError) as leaked:
                checker.get_table_columns(table)
            self.assertIn("same-day-join", str(leaked.exception), table)


# ---------------------------------------------------------------------------
# #14  the widths the checker reports are the widths the models were fitted on
# ---------------------------------------------------------------------------


class TestCheckerCountsAgainstTheRealTable(DropListEqualityMixin, unittest.TestCase):
    """The checker's output has to agree with reality or it checks nothing."""

    def setUp(self):
        if not DATASET_DB.exists():
            self.skipTest("dataset.sqlite is unavailable")
        self.columns = checker.get_table_columns(training_common.DEFAULT_DATASET)

    def test_the_table_has_the_shape_the_checker_claims_to_measure(self):
        # The issue measured 117 columns. If the build changes, the widths below
        # must be re-measured -- not quietly re-baselined by the checker.
        self.assertEqual(len(self.columns), 117)
        self.assertIn(training_common.GAME_DATE_COLUMN, self.columns)
        self.assertIn(training_common.CLOSING_TOTAL_COLUMN, self.columns)
        for column in ALWAYS_DROPPED:
            self.assertIn(column, self.columns, f"{column} is no longer in the table")

    def test_the_feature_widths_are_106_and_107(self):
        moneyline = checker.get_training_columns(
            training_common.DEFAULT_DATASET, checker.DROP_COLUMNS_ML
        )
        totals = checker.get_training_columns(
            training_common.DEFAULT_DATASET, checker.DROP_COLUMNS_UO
        )
        # The widths the committed boosters in Models/ were fitted on, and the
        # widths Tests/test_game_date.py pins for the trainers.
        self.assertEqual(len(moneyline), 106)
        self.assertEqual(len(totals), 107)

    def test_the_feature_list_is_the_dataset_order_minus_the_drop_list(self):
        # `expected_feature_order` is what src/Utils/features.py gives the
        # inference path. The checker must derive the same ordered names, or it
        # compares an inference frame against a contract of its own devising.
        for label, drop in (
            ("ML", checker.DROP_COLUMNS_ML),
            ("UO", checker.DROP_COLUMNS_UO),
        ):
            self.assertSameList(
                f"{label} feature order",
                features.expected_feature_order(self.columns, drop),
                checker.get_training_columns(training_common.DEFAULT_DATASET, drop),
            )

    def test_nothing_that_must_not_reach_a_model_survives_the_drop(self):
        for label, drop in (
            ("ML", checker.DROP_COLUMNS_ML),
            ("UO", checker.DROP_COLUMNS_UO),
        ):
            surviving = checker.get_training_columns(training_common.DEFAULT_DATASET, drop)
            for column in ALWAYS_DROPPED:
                self.assertNotIn(
                    column, surviving, f"{label}: {column!r} is now a model input"
                )

    def test_the_closing_total_is_a_moneyline_non_feature_and_a_totals_feature(self):
        # The end-to-end statement of the regression, on the real table: OU does
        # not reach the moneyline model and does reach the totals model, at the
        # dataset position Create_Games writes it to.
        moneyline = checker.get_training_columns(
            training_common.DEFAULT_DATASET, checker.DROP_COLUMNS_ML
        )
        totals = checker.get_training_columns(
            training_common.DEFAULT_DATASET, checker.DROP_COLUMNS_UO
        )
        self.assertNotIn(
            training_common.CLOSING_TOTAL_COLUMN,
            moneyline,
            "the closing total reached the moneyline features",
        )
        self.assertIn(
            training_common.CLOSING_TOTAL_COLUMN,
            totals,
            "the closing total is the totals model's input and must be present",
        )
        # Index 104 of 107, sitting between the away-team block and the two
        # rest columns. This is the input the historical permutation moved:
        # src/Utils/features.py records the totals model being fed OU at 106
        # having trained with it at 104, with nothing raising because the width
        # was the same either way. Pinned by index AND by neighbour name, so a
        # dataset rebuild that shifts it has to be re-measured out loud.
        position = totals.index(training_common.CLOSING_TOTAL_COLUMN)
        self.assertEqual(
            position,
            104,
            "the closing total moved index; it is part of the contract",
        )
        self.assertEqual(
            totals[position - 1],
            "PLUS_MINUS_RANK.1",
            "the last away-team column before the closing total changed",
        )
        self.assertEqual(
            totals[position + 1:],
            ["Days-Rest-Home", "Days-Rest-Away"],
            "the columns after the closing total moved",
        )
        # The dataset position Create_Games writes it to, which is what makes
        # the feature index above follow from the drop list alone.
        self.assertEqual(
            self.columns.index(training_common.CLOSING_TOTAL_COLUMN) - 9,
            position,
            "the closing total's feature index no longer follows from its "
            "dataset position; re-measure the contract",
        )

    def test_the_checker_agrees_with_the_inference_side_contract(self):
        # src/Utils/features.py holds the mirror-image lists, recorded as
        # "verbatim from XGBoost_Model_ML/UO.py". If those and the trainers ever
        # disagree, the checker has picked a side silently; this says which.
        self.assertEqual(features.ML_DROP_COLUMNS, checker.DROP_COLUMNS_ML)
        self.assertEqual(features.UO_LABEL_DROP_COLUMNS, checker.DROP_COLUMNS_UO)

    def test_the_reported_widths_match_the_committed_boosters(self):
        # End to end: what the script PRINTS against what the saved models
        # were fitted on. This is the check that would have failed before.
        if checker.xgb is None:
            self.skipTest("xgboost not installed")
        model = checker.find_latest_model("XGBoost_*_ML*.json")
        uo_model = checker.find_latest_model("XGBoost_*_UO*.json")
        if model is None or uo_model is None:
            self.skipTest("no committed XGBoost model to compare against")
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            ml_features = checker.describe_xgb_input(model, "XGB ML")
            uo_features = checker.describe_xgb_input(uo_model, "XGB OU")
        self.assertEqual(ml_features, len(self.columns) - len(MONEYLINE))
        self.assertEqual(uo_features, len(self.columns) - len(TOTALS))

    def test_the_checker_runs_as_a_standalone_script(self):
        # Loaded by path, executed by path, from a working directory that is not
        # the repository root -- the invocation the README documents.
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            cwd=str(Path(REPO_ROOT).parent),
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(
            completed.returncode, 0, completed.stderr[-2000:] or completed.stdout[-2000:]
        )
        self.assertIn("Training columns ML: 106", completed.stdout)
        self.assertIn("Training columns OU: 107", completed.stdout)


# ---------------------------------------------------------------------------
# #14  a checker that only compares sets would not catch a permutation
# ---------------------------------------------------------------------------


class TestCheckerStillCatchesAPermutation(DropListEqualityMixin, unittest.TestCase):
    """The comparison has to be able to see the failure the script exists for."""

    def setUp(self):
        if not DATASET_DB.exists():
            self.skipTest("dataset.sqlite is unavailable")
        self.columns = checker.get_table_columns(training_common.DEFAULT_DATASET)

    def _expected(self, drop):
        # The ordered feature names the trainers build, which is also what
        # src/Utils/features.py hands the inference path.
        return checker.get_training_columns(
            training_common.DEFAULT_DATASET, drop
        )

    def test_a_correctly_ordered_frame_compares_clean(self):
        for label, drop in (
            ("ML", checker.DROP_COLUMNS_ML),
            ("UO", checker.DROP_COLUMNS_UO),
        ):
            expected = self._expected(drop)
            frame = pd.DataFrame({name: [1.0] for name in expected})
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                checker.compare_columns(label, expected, list(frame.columns))
            printed = buffer.getvalue()
            self.assertIn(f"{label} missing: 0; extra: 0", printed)
            self.assertNotIn("missing sample", printed)

    def test_swapping_two_columns_changes_the_list_but_not_the_set(self):
        # This is the failure mode. A frame with the right columns in the wrong
        # order trains and predicts happily -- nothing raises -- so the checker
        # has to compare ORDERED lists. `compare_columns` only diffs sets, so
        # this also records that the set view cannot see the swap: the ordered
        # expectation, plus features.validate_feature_frame at the inference
        # boundary, is what covers it.
        train_cols = self._expected(checker.DROP_COLUMNS_ML)
        permuted = list(train_cols)
        permuted[0], permuted[1] = permuted[1], permuted[0]

        self.assertNotEqual(
            train_cols,
            permuted,
            "swapping two adjacent columns produced an identical list",
        )
        self.assertEqual(
            sorted(train_cols),
            sorted(permuted),
            "the swap changed the SET, not only the order, so this test no longer "
            "demonstrates a silent misalignment",
        )
        self.assertEqual(
            sorted(set(train_cols) - set(permuted)),
            [],
            "a set comparison of the two lists cannot see this failure",
        )

        mismatches = features.describe_order_mismatch(train_cols, permuted, limit=5)
        self.assertEqual(
            mismatches[0],
            f"position 0: expected {train_cols[0]!r}, got {permuted[0]!r}",
        )

        # And the frame that carries the swapped order is rejected by the
        # contract validator, which is where the order actually gets enforced.
        frame = pd.DataFrame({name: [1.0] for name in permuted})
        with self.assertRaises(ValueError) as caught:
            features.validate_feature_frame(frame, train_cols)
        self.assertIn("WRONG order", str(caught.exception))

    def test_the_closing_total_swap_between_moneyline_and_totals_is_visible(self):
        # The specific historical defect: OU at position 106 for a model trained
        # with it at 104. The two drop lists differ by exactly one entry, and the
        # checker's ML feature list must not contain OU while the totals one must.
        ml = self._expected(checker.DROP_COLUMNS_ML)
        uo = self._expected(checker.DROP_COLUMNS_UO)
        self.assertEqual(len(uo), len(ml) + 1)
        self.assertEqual(sorted(uo), sorted(ml + [training_common.CLOSING_TOTAL_COLUMN]))


if __name__ == "__main__":
    unittest.main()
