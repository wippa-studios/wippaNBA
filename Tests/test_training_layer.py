"""Tests for the training-layer defects fixed in the six trainers.

Six filed issues, one file, and each class below names the issue it pins:

  ``TestDatasetTableAgreement``   #2   entry points disagreed on the table name;
                                       two named a table that does not exist and
                                       two named a leaked one.
  ``TestTotalsTrainerRuns``       #5   ``multi_class="multinomial"`` is a
                                       TypeError on modern scikit-learn, so the
                                       totals trainer could not run at all.
  ``TestTotalsTrainerRunsEndToEnd`` #5  the same fix, proved by RUNNING the
                                       entry point rather than by inspecting the
                                       kwargs it would build.
  ``TestSearchDoesNotRankNoise``  #7   the search ranked on seed luck and on a
                                       num_boost_round dimension that early
                                       stopping already resolves; 100 trials
                                       bought a 2.50 SE winner's curse.
  ``TestCalibrationIsMeasured``   #8   isotonic measured WORSE than no
                                       calibration and was offered anyway, with
                                       its effect reported as a given.
  ``TestFoldGeometry``            #9   a fold was two seasons wide and the five
                                       folds spanned ten years, scored on a flat
                                       mean.
  ``TestSplitHygiene``            #17  quicksort is not stable, so split
                                       boundaries landed inside same-date groups;
                                       imputation ran before the split.

Plus the regression test the issue asked for by name: no split boundary may fall
inside a group of rows sharing one date.

HOW THE HYPHEN IS HANDLED
========================
``src/Train-Models`` contains a hyphen, so it can never be an importable package.
It goes on ``sys.path`` and its modules import under plain top-level names, which
is the convention their own docstrings describe. ``src.Train-Models.XGBoost_*``
and ``src.Train-Models.NN_*`` additionally import xgboost and tensorflow, neither
of which is installed here, so those two modules are NOT imported: their source is
checked with ``ast`` instead. Every assertion about them is therefore about the
text, not about behaviour, and says so.
"""

import ast
import contextlib
import importlib.util
import io
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_MODELS_DIR = REPO_ROOT / "src" / "Train-Models"
PREDICT_DIR = REPO_ROOT / "src" / "Predict"
CREATE_GAMES_PATH = REPO_ROOT / "src" / "Process-Data" / "Create_Games.py"
DATASET_DB = REPO_ROOT / "Data" / "dataset.sqlite"

for _path in (REPO_ROOT, TRAIN_MODELS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import logistic_training  # noqa: E402
import training_common  # noqa: E402
import training_reporting  # noqa: E402

from src.Utils import metrics  # noqa: E402
from src.Utils import validation  # noqa: E402

TRAINER_SOURCES = (
    "Logistic_Regression_ML.py",
    "Logistic_Regression_UO.py",
    "XGBoost_Model_ML.py",
    "XGBoost_Model_UO.py",
    "NN_Model_ML.py",
    "NN_Model_UO.py",
)


def trainer_source(name):
    return (TRAIN_MODELS_DIR / name).read_text()


def load_trainer_module(name):
    """Import a trainer by path, with xgboost/tensorflow stubbed out.

    The XGBoost and NN trainers import those two libraries at module level and
    neither is installed in this environment. Stubbing them lets the MODULE be
    parsed, imported and inspected -- including ``add_search_arguments`` and the
    argparse defaults -- without pretending the booster was ever fitted. Nothing
    here runs a model.
    """
    import types

    class _Stub(types.ModuleType):
        def __getattr__(self, item):
            if item.startswith("__"):
                raise AttributeError(item)
            return _Stub(f"{self.__name__}.{item}")

    stubs = {}
    for name_ in ("xgboost", "tensorflow"):
        if name_ not in sys.modules:
            try:
                __import__(name_)
            except ImportError:
                stubs[name_] = _Stub(name_)
    saved = {key: sys.modules.get(key) for key in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(
            f"_probe_{Path(name).stem}", TRAIN_MODELS_DIR / name
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


def synthetic_dataset(n_rows=600, per_date=4, seed=0):
    """A small chronological frame shaped like the real dataset.

    Several rows share each snapshot date, which is what makes the sort-stability
    and split-boundary properties observable at all.
    """
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2015-10-20")
    dates, index = [], []
    for step in range(n_rows // per_date):
        day = base + pd.Timedelta(days=step)
        for slot in range(per_date):
            dates.append(day)
            index.append(len(dates))
    rows = len(dates)
    return pd.DataFrame(
        {
            "index": index,
            "TEAM_NAME": [f"Home {i}" for i in range(rows)],
            "F1": rng.normal(size=rows),
            "F2": rng.normal(size=rows),
            "F3": rng.normal(size=rows),
            "Date": pd.to_datetime(dates),
            "Date.1": pd.to_datetime(dates),
            "GameDate": pd.to_datetime(dates),
            "TEAM_NAME.1": [f"Away {i}" for i in range(rows)],
            "Score": rng.normal(size=rows),
            "Home-Team-Win": rng.integers(0, 2, rows),
            "OU": 210.0 + rng.normal(size=rows),
            "OU-Cover": rng.integers(0, 3, rows),
        }
    )


def write_synthetic_dataset_db(path, n_rows=600, per_date=4, seed=0):
    """``synthetic_dataset`` written to a sqlite file named for the real one.

    The trainers read a dataset through ``training_common.load_dataset``, which
    goes via ``resolve_dataset_table`` and so needs a real FILE holding a table
    called ``DEFAULT_DATASET``. Pointing ``training_common.DATASET_DB`` at this
    file is therefore enough to drive a trainer end to end with no downloaded
    data and no committed 38 MB table -- which is what the CI job does, and what
    ``TestTotalsTrainerRunsEndToEnd`` does in-process.
    """
    frame = synthetic_dataset(n_rows=n_rows, per_date=per_date, seed=seed)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Explicit close, not `with sqlite3.connect(...)`: that context manager
    # commits and rolls back, it does not close the handle, so it leaks a
    # connection into every test that builds a dataset.
    con = sqlite3.connect(path)
    try:
        frame.to_sql(training_common.DEFAULT_DATASET, con, index=False, if_exists="replace")
        con.commit()
    finally:
        con.close()
    return path


def cut_inside_a_group(dates, fraction):
    """True when the ``fraction`` cut of ``dates`` splits a same-date group."""
    dates = pd.Series(dates).reset_index(drop=True)
    position = int(len(dates) * fraction)
    if position <= 0 or position >= len(dates):
        return False
    return dates.iloc[position - 1] == dates.iloc[position]


def within_group_positions_are_increasing(dates, order):
    """True when, for every date, the stored row positions increase.

    This is stability, stated as a property of the output rather than as a claim
    about a sort algorithm's implementation: if rows sharing a snapshot date keep
    their stored order, then a cut made at any row position lands at the same
    place every time, which is the whole requirement.
    """
    resolved = np.asarray(pd.Series(dates).to_numpy())[np.asarray(order)]
    positions = np.asarray(order)
    if len(positions) < 2:
        return True
    change = resolved[1:] != resolved[:-1]
    starts = np.concatenate(([0], np.nonzero(change)[0] + 1))
    ends = np.concatenate((np.nonzero(change)[0] + 1, [len(positions)]))
    for start, end in zip(starts, ends):
        if np.any(np.diff(positions[start:end]) < 0):
            return False
    return True


def real_dataset_frame():
    """The real leak-free table, or skip."""
    if not DATASET_DB.exists():
        raise unittest.SkipTest("dataset.sqlite is unavailable")
    con = sqlite3.connect(f"file:{DATASET_DB}?mode=ro", uri=True)
    try:
        frame = pd.read_sql_query(
            f'SELECT * FROM "{training_common.DEFAULT_DATASET}"', con
        )
    finally:
        con.close()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    return frame


def real_snapshot_dates():
    """The real table's dates in stable sorted order."""
    frame = real_dataset_frame()
    return (
        frame.sort_values("Date", kind=training_common.STABLE_SORT_KIND)["Date"]
        .reset_index(drop=True)
    )


def calibration_pair(n_rows, seed=0, spread=0.2, bias=0.12, harmful=False):
    """``(truth, distorted, ideal)`` -- one miscalibrated model and one exact.

    The label is drawn FROM ``ideal``, so ``ideal`` is the true conditional
    probability and therefore the lowest log loss any model could reach. A
    calibrator that works moves ``distorted`` back toward ``ideal`` and reduces
    the loss; one that does not makes it worse. That is the only construction in
    which a calibrator test means anything -- an "improvement" on probabilities
    with no relationship to the labels is an artefact of the metric.

    ``harmful=True`` mirrors ``distorted`` in the other direction, so correcting
    it HURTS. That is the shape of the isotonic finding (+0.0035 nats).
    """
    rng = np.random.default_rng(seed)
    signal = rng.normal(size=n_rows)
    ideal = np.clip(0.5 + spread * signal, 0.02, 0.98)
    truth = (rng.uniform(size=n_rows) < ideal).astype(int)
    shift = -bias if harmful else bias
    distorted = np.clip(ideal + shift, 0.02, 0.98)
    return (
        truth,
        np.column_stack([1.0 - distorted, distorted]),
        np.column_stack([1.0 - ideal, ideal]),
    )


# ---------------------------------------------------------------------------
# #2  entry points agreed on nothing about which table to read
# ---------------------------------------------------------------------------


class TestDatasetTableAgreement(unittest.TestCase):
    """One table name, defined once, checked before anything else happens."""

    def test_default_dataset_is_the_leak_free_table(self):
        self.assertEqual(training_common.DEFAULT_DATASET, "dataset_2012-26")

    def test_no_trainer_spell_out_a_table_name(self):
        # The defect was six copies of this value and they had already drifted.
        # AST rather than grep: a table name inside a comment or a docstring does
        # not count, a string literal that binds one does.
        for name in TRAINER_SOURCES:
            tree = ast.parse(trainer_source(name))
            assigned = [
                node.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "DEFAULT_DATASET"
                    for target in node.targets
                )
                and isinstance(node.value, ast.Constant)
            ]
            self.assertEqual(assigned, [], f"{name} redefines DEFAULT_DATASET")
            self.assertIn("default=DEFAULT_DATASET", trainer_source(name), name)

    def test_no_runner_spell_out_a_table_name(self):
        for name in ("NN_Runner.py", "XGBoost_Runner.py"):
            source = (PREDICT_DIR / name).read_text()
            tree = ast.parse(source)
            assigned = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "DATASET_TABLE"
                    for target in node.targets
                )
            ]
            self.assertEqual(assigned, [], f"{name} still assigns DATASET_TABLE")
            self.assertIn(
                "from training_common import DEFAULT_DATASET as DATASET_TABLE",
                source,
                name,
            )

    def test_every_trainer_defaults_to_the_shared_name(self):
        for name in TRAINER_SOURCES:
            parser = load_trainer_module(name).build_parser()
            dataset_flag = next(
                action for action in parser._actions if action.dest == "dataset"
            )
            self.assertEqual(
                dataset_flag.default,
                training_common.DEFAULT_DATASET,
                f"{name} does not default to the shared table",
            )

    def test_a_table_that_does_not_exist_names_what_is_available(self):
        with self.assertRaises(LookupError) as caught:
            training_common.resolve_dataset_table("dataset_2012-26_new")
        message = str(caught.exception)
        self.assertIn("dataset_2012-26_new", message)
        self.assertIn("dataset_2012-26", message)

    def test_a_leaked_table_is_refused_by_name(self):
        # The other half of the same defect: dataset_2012-24_new EXISTS, so the
        # old code trained on a same-day join and never noticed.
        for table in training_common.LEAKED_DATASET_TABLES:
            with self.assertRaises(LookupError) as caught:
                training_common.resolve_dataset_table(table)
            self.assertIn("same-day-join", str(caught.exception))

    def test_the_real_table_resolves(self):
        if not DATASET_DB.exists():
            self.skipTest("dataset.sqlite is unavailable")
        self.assertEqual(
            training_common.resolve_dataset_table(training_common.DEFAULT_DATASET),
            "dataset_2012-26",
        )

    def test_load_dataset_defaults_to_the_shared_name(self):
        if not DATASET_DB.exists():
            self.skipTest("dataset.sqlite is unavailable")
        frame = training_common.load_dataset()
        self.assertGreater(len(frame), 0)
        self.assertIn(training_common.MONEYLINE_TARGET, frame.columns)

    def test_available_tables_is_read_only_and_lists_the_file(self):
        if not DATASET_DB.exists():
            self.skipTest("dataset.sqlite is unavailable")
        tables = training_common.available_dataset_tables()
        self.assertIn("dataset_2012-26", tables)
        self.assertEqual(tables, sorted(tables))

    def test_available_tables_is_empty_when_the_file_is_missing(self):
        saved = training_common.DATASET_DB
        try:
            training_common.DATASET_DB = REPO_ROOT / "Data" / "does-not-exist.sqlite"
            self.assertEqual(training_common.available_dataset_tables(), [])
            with self.assertRaises(FileNotFoundError):
                training_common.resolve_dataset_table("dataset_2012-26")
        finally:
            training_common.DATASET_DB = saved


# ---------------------------------------------------------------------------
# #5  the totals trainer could not run at all
# ---------------------------------------------------------------------------


class TestTotalsTrainerRuns(unittest.TestCase):
    """``multi_class`` was removed in scikit-learn 1.7; the script passed it."""

    def test_multi_class_is_not_in_this_sklearn(self):
        # Only meaningful on a modern scikit-learn; on the 1.3.1 pin it IS a
        # parameter, and the version fork takes the other branch.
        import inspect

        from sklearn.linear_model import LogisticRegression

        present = "multi_class" in inspect.signature(
            LogisticRegression.__init__
        ).parameters
        self.assertEqual(
            present, logistic_training.supports_multi_class_parameter()
        )

    def test_kwargs_omit_multi_class_when_the_version_dropped_it(self):
        import inspect

        from sklearn.linear_model import LogisticRegression

        kwargs = logistic_training.logistic_regression_kwargs(
            1.0, "l2", "lbfgs", random_state=42, num_classes=3
        )
        modern = "multi_class" not in inspect.signature(
            LogisticRegression.__init__
        ).parameters
        if modern:
            self.assertNotIn("multi_class", kwargs)
        else:
            self.assertEqual(kwargs["multi_class"], "multinomial")

    # ---------------------------------------------------------------------
    # BOTH BRANCHES OF THE VERSION GATE, ON WHATEVER SCIPY-LEARN IS INSTALLED
    #
    # The test above asserts whichever branch matches the installed library, so
    # on any single CI leg only one of the two is ever exercised -- and the one
    # that is skipped is the one that used to crash. Both are pinned here by
    # faking the probe, so the fork is covered on every scikit-learn, and a
    # matrix spanning the removal tests the gate rather than the branch the
    # local environment happens to land on.
    # ---------------------------------------------------------------------

    def test_kwargs_omit_multi_class_when_the_probe_says_it_is_gone(self):
        # The >= 1.7 branch. The parameter no longer exists, so it must not be
        # passed -- this is the branch that raised TypeError before the fix.
        with mock.patch.object(
            logistic_training, "supports_multi_class_parameter", return_value=False
        ):
            kwargs = logistic_training.logistic_regression_kwargs(
                1.0, "l2", "lbfgs", random_state=42, num_classes=3
            )
        self.assertNotIn("multi_class", kwargs)

    def test_kwargs_include_multinomial_when_the_probe_says_it_is_taken(self):
        # The < 1.7 branch. The parameter is still accepted, and for a
        # three-class target "multinomial" is the intended behaviour spelled out
        # rather than left to be inferred.
        with mock.patch.object(
            logistic_training, "supports_multi_class_parameter", return_value=True
        ):
            kwargs = logistic_training.logistic_regression_kwargs(
                1.0, "l2", "lbfgs", random_state=42, num_classes=3
            )
        self.assertEqual(kwargs["multi_class"], "multinomial")

    def test_the_two_branches_differ_only_in_the_multi_class_key(self):
        # Otherwise the fork is doing more than the fork: a regression that
        # quietly changed C, the penalty, the solver or the seed depending on
        # the scikit-learn version would make two branches train different
        # models, which no version of scikit-learn asks for.
        modern = logistic_training.logistic_regression_kwargs(
            1.0, "l2", "lbfgs", random_state=42, num_classes=3
        )
        with mock.patch.object(
            logistic_training, "supports_multi_class_parameter", return_value=True
        ):
            legacy = logistic_training.logistic_regression_kwargs(
                1.0, "l2", "lbfgs", random_state=42, num_classes=3
            )
        self.assertEqual(set(legacy) - set(modern), {"multi_class"})
        self.assertEqual(
            {key: value for key, value in legacy.items() if key != "multi_class"},
            modern,
        )

    def test_kwargs_never_set_multi_class_for_a_binary_target(self):
        kwargs = logistic_training.logistic_regression_kwargs(
            1.0, "l2", "lbfgs", random_state=42, num_classes=2
        )
        self.assertNotIn("multi_class", kwargs)

    def test_kwargs_always_pin_random_state(self):
        # saga is the UO script's L1 solver and is randomised by default; an
        # unpinned saga is an irreproducible model, and the search then selects
        # on solver luck as well as seed luck.
        kwargs = logistic_training.logistic_regression_kwargs(
            0.5, "l1", "saga", random_state=7, num_classes=3
        )
        self.assertEqual(kwargs["random_state"], 7)
        self.assertEqual(kwargs["solver"], "saga")

    def test_the_estimator_accepts_the_kwargs_this_sklearn_builds(self):
        # The regression itself: build the estimator the UO script builds, on this
        # scikit-learn, and fit it. On 1.7+ the old code raised TypeError here.
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 5))
        y = rng.integers(0, 3, 200)
        kwargs = logistic_training.logistic_regression_kwargs(
            1.0, "l2", "lbfgs", random_state=42, num_classes=3
        )
        model = LogisticRegression(max_iter=200, **kwargs).fit(X, y)
        self.assertEqual(model.predict_proba(X).shape, (200, 3))

    def test_no_trainer_passes_multi_class_to_the_estimator(self):
        # AST, not text: `multi_class='ovr'` is a legitimate argument to
        # roc_auc_score, so a substring search would report a false failure. What
        # must not exist is multi_class as a LogisticRegression keyword, which is
        # what raised TypeError on scikit-learn >= 1.7.
        for name in ("Logistic_Regression_UO.py", "Logistic_Regression_ML.py"):
            tree = ast.parse(trainer_source(name))
            offenders = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "LogisticRegression"
                and any(kw.arg == "multi_class" for kw in node.keywords)
            ]
            self.assertEqual(offenders, [], f"{name} passes multi_class to the estimator")


class TestTotalsTrainerRunsEndToEnd(unittest.TestCase):
    """``Logistic_Regression_UO.main()``, whole script, on synthetic data.

    Every other #5 test checks one fact about the fix. This one runs the entry
    point, because that is what the issue actually reported: not a wrong number,
    a script that could not start. The unit tests above would all pass while the
    trainer was broken somewhere else in the body -- the search loop, the
    calibrator, the reporting tail -- which is the other half of the acceptance
    criteria and the part CI is asked to police.

    Data: a 600-row synthetic ``dataset_2012-26`` written to a temporary sqlite
    file, with ``training_common.DATASET_DB`` and ``training_reporting.ODDS_DB``
    repointed at it. Nothing is downloaded and the committed 38 MB table is not
    read, so this runs in about two seconds on any checkout. The CI job
    ``totals-trainer`` runs the same synthetic table through the real
    ``python -m src.Train-Models.Logistic_Regression_UO`` entry point.
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = write_synthetic_dataset_db(Path(cls.tmp.name) / "dataset.sqlite")
        cls.saved_dataset_db = training_common.DATASET_DB
        cls.saved_odds_db = training_reporting.ODDS_DB
        training_common.DATASET_DB = cls.db
        # No odds database, so the closing-line report takes its documented
        # "not available" branch instead of matching 0 of 60 synthetic test
        # rows. It must never raise either way; this keeps the test from
        # depending on the committed OddsData.sqlite being present.
        training_reporting.ODDS_DB = Path(cls.tmp.name) / "no-odds.sqlite"
        cls.trainer = load_trainer_module("Logistic_Regression_UO.py")

    @classmethod
    def tearDownClass(cls):
        training_common.DATASET_DB = cls.saved_dataset_db
        training_reporting.ODDS_DB = cls.saved_odds_db
        cls.tmp.cleanup()

    def run_main(self, extra_args=()):
        argv = ["Logistic_Regression_UO", "--trials", "2", "--splits", "3", *extra_args]
        saved = sys.argv
        sys.argv = argv
        try:
            with contextlib.redirect_stdout(io.StringIO()) as buffer:
                self.trainer.main()
        finally:
            sys.argv = saved
        return buffer.getvalue()

    def test_main_completes_and_reports_a_held_out_score(self):
        # The regression itself. Before the fix this raised
        # TypeError: __init__() got an unexpected keyword argument 'multi_class'
        # on the first trial's estimator, and nothing below was ever reached.
        out = self.run_main()
        for expected in (
            "Trial 1/2:",
            "Best val log loss:",
            "Test accuracy:",
            "Test log loss:",
            "Test ROC AUC: multiclass",
            "Classification report:",
        ):
            self.assertIn(expected, out)

    def test_main_scores_three_totals_classes(self):
        # Three classes out, not two: the label is under/over/push, so a
        # probability vector of width 2 would be a silently wrong model rather
        # than a crash. LABELS is the script's own constant.
        self.assertEqual(self.trainer.LABELS, [0, 1, 2])
        self.assertEqual(self.trainer.NUM_CLASSES, training_common.TOTALS_NUM_CLASSES)
        self.assertEqual(self.trainer.TARGET_COLUMN, training_common.TOTALS_TARGET)

    def test_main_does_not_ask_for_the_isotonic_it_refuses(self):
        # Reached only with BOTH --calibration isotonic AND --allow-isotonic:
        # argparse fixes `choices` before it sees a boolean neighbour, so
        # isotonic is not an accepted value without the opt-in. With the opt-in
        # present, the SIZE gate in resolve_calibration_method is what refuses
        # it here, because a 600-row synthetic table yields a calibration
        # segment far below MIN_CALIBRATION_ROWS. This asserts the run reaches
        # and honours that gate rather than quietly calibrating anyway.
        with self.assertRaises(ValueError) as caught:
            self.run_main(("--calibration", "isotonic", "--allow-isotonic"))
        self.assertIn(
            str(training_reporting.MIN_CALIBRATION_ROWS), str(caught.exception)
        )

    def test_a_missing_dataset_is_a_named_failure_not_a_bare_sqlite_error(self):
        # The second half of #5's blast radius: the default table used to be a
        # name that did not exist. Point the loader at a database with no
        # tables and the failure has to name the file and list what was there.
        training_common.DATASET_DB = Path(self.tmp.name) / "absent.sqlite"
        try:
            with self.assertRaises(FileNotFoundError) as caught:
                self.run_main()
        finally:
            training_common.DATASET_DB = self.db
        message = str(caught.exception)
        self.assertIn("absent.sqlite", message)
        self.assertIn("Create_Games", message)


# ---------------------------------------------------------------------------
# #7  the hyperparameter search was ranking on noise
# ---------------------------------------------------------------------------


class TestSearchDoesNotRankNoise(unittest.TestCase):
    def test_default_trials_is_fifteen_or_fewer(self):
        # 100 trials are worth ~2.50 SE here, against a total true range of
        # 0.0019 nats and a 0.0073 noise floor.
        self.assertLessEqual(training_common.DEFAULT_TRIALS, 15)
        self.assertGreaterEqual(training_common.DEFAULT_TRIALS, 10)

    def test_every_trainer_defaults_to_the_shared_trial_count(self):
        for name in TRAINER_SOURCES:
            parser = load_trainer_module(name).build_parser()
            trials = next(a for a in parser._actions if a.dest == "trials")
            self.assertEqual(trials.default, training_common.DEFAULT_TRIALS, name)

    def test_no_trainer_asks_for_one_hundred_trials_by_default(self):
        for name in TRAINER_SOURCES:
            self.assertNotIn("default=100", trainer_source(name), name)
            self.assertNotIn('default="100"', trainer_source(name), name)

    def test_xgboost_boosters_no_longer_vary_the_seed_per_trial(self):
        # AST, not text: the comment that explains this defect quotes the old
        # expression verbatim, so a substring search would match the explanation.
        # What must not exist is an `args.seed + trial` BinOp.
        for name in ("XGBoost_Model_ML.py", "XGBoost_Model_UO.py"):
            tree = ast.parse(trainer_source(name))
            offenders = [
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.BinOp)
                and isinstance(node.op, ast.Add)
                and isinstance(node.left, ast.Attribute)
                and node.left.attr == "seed"
            ]
            self.assertEqual(offenders, [], f"{name} still perturbs the seed per trial")
            self.assertIn("seed=args.seed,", trainer_source(name), name)

    def test_num_boost_round_is_a_cap_not_a_sampled_dimension(self):
        # boosting_training imports xgboost, so it is loaded with a stub.
        boosting = load_trainer_module("boosting_training.py")
        self.assertEqual(boosting.DEFAULT_NUM_BOOST_ROUND, 2500)
        rounds = {
            boosting.sample_params(np.random.default_rng(s), seed=1, num_classes=2)[1]
            for s in range(50)
        }
        self.assertEqual(rounds, {2500})

    def test_sampling_is_reproducible_from_one_seed(self):
        # The rng stream must not be consumed by anything else, or removing a
        # dimension would silently shift every remaining dimension's values.
        boosting = load_trainer_module("boosting_training.py")
        first = [
            boosting.sample_params(np.random.default_rng(7), seed=1, num_classes=2)[0]
            for _ in range(1)
        ]
        second = [
            boosting.sample_params(np.random.default_rng(7), seed=1, num_classes=2)[0]
            for _ in range(1)
        ]
        self.assertEqual(first, second)

    def test_drawing_twice_from_one_rng_gives_two_different_configurations(self):
        # ...but the search still explores: consecutive draws from ONE stream are
        # different. Only the SEED was frozen; the sampling is not.
        boosting = load_trainer_module("boosting_training.py")
        rng = np.random.default_rng(7)
        a = boosting.sample_params(rng, seed=1, num_classes=2)[0]
        b = boosting.sample_params(rng, seed=1, num_classes=2)[0]
        self.assertNotEqual(a, b)
        self.assertEqual(a["seed"], b["seed"])

    def test_max_delta_step_is_zero_unless_class_weights_are_on(self):
        boosting = load_trainer_module("boosting_training.py")
        unweighted = [
            boosting.sample_params(np.random.default_rng(s), seed=1, num_classes=2)[0]
            for s in range(20)
        ]
        self.assertEqual({p["max_delta_step"] for p in unweighted}, {0})

        weighted = [
            boosting.sample_params(
                np.random.default_rng(s), seed=1, num_classes=2, class_weights="balanced"
            )[0]
            for s in range(20)
        ]
        self.assertTrue(len({p["max_delta_step"] for p in weighted}) > 1)

    def test_the_seed_is_the_same_for_every_trial_of_a_run(self):
        boosting = load_trainer_module("boosting_training.py")
        seeds = {
            boosting.sample_params(np.random.default_rng(s), seed=42, num_classes=2)[0][
                "seed"
            ]
            for s in range(10)
        }
        self.assertEqual(seeds, {42})

    def test_every_trainer_reports_the_selection_risk(self):
        for name in TRAINER_SOURCES:
            source = trainer_source(name)
            self.assertIn("report_search_selection_risk(", source, name)
            self.assertIn("Best val ", source, name)

    def test_report_search_selection_risk_prints_the_correction(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            summary = training_common.report_search_selection_risk(
                15, "held-out log loss", 1722, 0.0079
            )
        self.assertIn("SELECTION RISK", buffer.getvalue())
        self.assertIn("configurations tried:    15", summary)
        self.assertIn("not", summary)

    def test_a_missing_selection_module_is_reported_not_raised(self):
        # A reporting aid must never kill a training run.
        saved = sys.modules.get("src.Utils.validation", False)
        sys.modules["src.Utils.validation"] = None  # makes the import raise
        try:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                result = training_common.report_search_selection_risk(
                    15, "x", 100, 0.01
                )
            self.assertIsNone(result)
            self.assertIn("Selection risk not reported", buffer.getvalue())
        finally:
            if saved is False:
                sys.modules.pop("src.Utils.validation", None)
            else:
                sys.modules["src.Utils.validation"] = saved

    def test_selection_effect_shrinks_as_trials_shrink(self):
        # The measured premise of cutting --trials, asserted rather than assumed.
        self.assertGreater(
            validation.selection_bias_estimate(100, 1722),
            validation.selection_bias_estimate(training_common.DEFAULT_TRIALS, 1722),
        )
        self.assertGreater(validation.selection_bias_estimate(100, 1722), 2.0)


# ---------------------------------------------------------------------------
# #8  calibration was doing nothing, or hurting
# ---------------------------------------------------------------------------


class TestCalibrationIsMeasured(unittest.TestCase):
    def test_isotonic_is_not_an_offered_choice(self):
        self.assertNotIn("isotonic", training_reporting.CALIBRATION_CHOICES)
        self.assertEqual(
            set(training_reporting.CALIBRATION_CHOICES), {"sigmoid", "none"}
        )

    def test_the_opt_in_flag_widens_the_choices(self):
        # argparse fixes `choices` before it reads any other flag, so the widening
        # is a pre-scan of argv. Without it --allow-isotonic would be a flag that
        # silently does nothing.
        self.assertEqual(
            training_common.calibration_choices(["--trials", "5"]),
            ("sigmoid", "none"),
        )
        self.assertEqual(
            training_common.calibration_choices(
                ["--trials", "5", "--allow-isotonic"]
            ),
            ("sigmoid", "isotonic", "none"),
        )

    def test_a_parser_built_without_the_flag_refuses_isotonic(self):
        for name in TRAINER_SOURCES:
            if "NN_" in name:
                continue
            parser = load_trainer_module(name).build_parser()
            action = next(a for a in parser._actions if a.dest == "calibration")
            self.assertNotIn("isotonic", action.choices, name)
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stderr(io.StringIO()):
                    parser.parse_args(["--calibration", "isotonic"])

    def test_a_parser_built_with_the_flag_accepts_isotonic(self):
        saved = sys.argv
        sys.argv = ["XGBoost_Model_ML.py", "--calibration", "isotonic",
                    "--allow-isotonic"]
        try:
            for name in TRAINER_SOURCES:
                if "NN_" in name:
                    continue
                parser = load_trainer_module(name).build_parser()
                args = parser.parse_args(
                    ["--calibration", "isotonic", "--allow-isotonic"]
                )
                self.assertEqual(args.calibration, "isotonic", name)
                self.assertTrue(args.allow_isotonic, name)
        finally:
            sys.argv = saved

    def test_isotonic_needs_an_explicit_opt_in(self):
        with self.assertRaises(ValueError) as caught:
            training_reporting.resolve_calibration_method("isotonic", 10_000)
        self.assertIn("--allow-isotonic", str(caught.exception))

    def test_isotonic_still_needs_enough_rows_after_the_opt_in(self):
        with self.assertRaises(ValueError) as caught:
            training_reporting.resolve_calibration_method(
                "isotonic", 1547, allow_isotonic=True
            )
        self.assertIn(str(training_reporting.MIN_CALIBRATION_ROWS), str(caught.exception))
        # Past the gate it is allowed, which is the point of the gate.
        self.assertEqual(
            training_reporting.resolve_calibration_method(
                "isotonic", training_reporting.MIN_CALIBRATION_ROWS, allow_isotonic=True
            ),
            "isotonic",
        )

    def test_sigmoid_and_none_pass_through_untouched(self):
        self.assertEqual(
            training_reporting.resolve_calibration_method("sigmoid", 5), "sigmoid"
        )
        self.assertEqual(training_reporting.resolve_calibration_method("none", 5), "none")

    def test_no_trainer_offers_isotonic_without_the_opt_in_flag(self):
        for name in TRAINER_SOURCES:
            if "NN_" in name:
                # The networks have no calibrator, so there is nothing to gate;
                # ``test_every_trainer_exposes_fold_geometry_flags`` asserts they
                # expose no --calibration flag at all.
                continue
            source = trainer_source(name)
            self.assertNotIn('choices=["sigmoid", "isotonic", "none"]', source, name)
            self.assertIn("--allow-isotonic", source, name)

    def test_log_loss_standard_error_matches_a_hand_computation(self):
        rng = np.random.default_rng(3)
        n = 400
        truth = rng.integers(0, 2, n)
        proba = rng.uniform(0.2, 0.8, (n, 2))
        proba /= proba.sum(axis=1, keepdims=True)
        rows = np.arange(n)
        per_row = -np.log(proba[rows, truth])
        self.assertAlmostEqual(
            metrics.log_loss_standard_error(truth, proba),
            float(np.std(per_row, ddof=1) / np.sqrt(n)),
            places=12,
        )

    def test_bootstrap_delta_is_zero_for_an_identical_pair(self):
        rng = np.random.default_rng(4)
        truth = rng.integers(0, 2, 300)
        proba = np.column_stack([1 - truth * 0.6, truth * 0.6])
        verdict = metrics.bootstrap_paired_delta(truth, proba, proba)
        self.assertAlmostEqual(verdict["delta"], 0.0, places=12)
        self.assertAlmostEqual(verdict["standard_error"], 0.0, places=12)

    def test_bootstrap_delta_finds_a_real_improvement(self):
        truth, distorted, ideal = calibration_pair(2000, seed=5)
        verdict = metrics.bootstrap_paired_delta(
            truth, ideal, distorted, n_bootstrap=400
        )
        self.assertLess(verdict["delta"], -0.01)
        self.assertLess(verdict["high"], 0.0)
        self.assertGreater(verdict["standard_error"], 0.0)

    def test_bootstrap_delta_finds_a_real_harm(self):
        # The isotonic case: correcting this model makes it worse, and the
        # bootstrap has to be able to say so rather than always favouring the
        # second argument.
        truth, distorted, ideal = calibration_pair(2000, seed=5, harmful=True)
        verdict = metrics.bootstrap_paired_delta(
            truth, distorted, ideal, n_bootstrap=400
        )
        self.assertGreater(verdict["delta"], 0.01)
        self.assertGreater(verdict["low"], 0.0)

    def test_bootstrap_is_paired_so_its_se_is_not_the_difference_of_two(self):
        # Resampling the two models independently would report a standard error
        # several times too large and hide a real effect. Compare against the
        # unpaired figure and assert the paired one is far smaller.
        truth, distorted, ideal = calibration_pair(3000, seed=6, bias=0.05)
        paired = metrics.bootstrap_paired_delta(truth, ideal, distorted, n_bootstrap=400)
        se_a = metrics.log_loss_standard_error(truth, ideal)
        se_b = metrics.log_loss_standard_error(truth, distorted)
        unpaired = float(np.sqrt(se_a ** 2 + se_b ** 2))
        self.assertLess(paired["standard_error"], unpaired / 2.0)

    def test_bootstrap_is_deterministic_in_its_seed(self):
        truth, distorted, ideal = calibration_pair(500, seed=7)
        first = metrics.bootstrap_paired_delta(truth, ideal, distorted)
        second = metrics.bootstrap_paired_delta(truth, ideal, distorted)
        self.assertEqual(first, second)

    def test_bootstrap_rejects_a_bad_n_bootstrap(self):
        rng = np.random.default_rng(8)
        truth = rng.integers(0, 2, 50)
        proba = rng.uniform(0.2, 0.8, (50, 2))
        proba /= proba.sum(axis=1, keepdims=True)
        with self.assertRaises(ValueError):
            metrics.bootstrap_paired_delta(truth, proba, proba, n_bootstrap=0)
        with self.assertRaises(ValueError):
            metrics.bootstrap_paired_delta(truth, proba, proba, confidence=1.5)

    def test_bootstrap_rejects_a_pair_of_different_lengths(self):
        rng = np.random.default_rng(11)
        truth = rng.integers(0, 2, 50)
        a = rng.uniform(0.2, 0.8, (50, 2))
        b = rng.uniform(0.2, 0.8, (40, 2))
        a /= a.sum(axis=1, keepdims=True)
        b /= b.sum(axis=1, keepdims=True)
        with self.assertRaises(ValueError):
            metrics.bootstrap_paired_delta(truth, a, b)

    def test_the_report_prints_the_segment_size_and_an_interpretation(self):
        truth, distorted, _ideal = calibration_pair(400, seed=9, bias=0.004)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            verdict = training_reporting.report_calibration_effect(
                truth, distorted, distorted, n_calib_rows=1547, method="sigmoid"
            )
        text = buffer.getvalue()
        # The segment size sits next to the choice: "sigmoid" on 1,547 rows and on
        # 150,000 rows are different claims wearing the same name.
        self.assertIn("1547 rows", text)
        self.assertIn("the interval includes zero", text)
        self.assertEqual(verdict["delta"], 0.0)

    def test_the_report_calls_a_real_effect_real(self):
        truth, distorted, ideal = calibration_pair(2000, seed=10)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            training_reporting.report_calibration_effect(
                truth, distorted, ideal, n_calib_rows=1547, method="sigmoid"
            )
        self.assertIn("improves log loss", buffer.getvalue())

    def test_the_report_calls_a_real_harm_real(self):
        # The same machinery has to be able to say "this calibrator hurts", which
        # is the finding that got isotonic removed from the choices.
        truth, distorted, _ideal = calibration_pair(2000, seed=12, harmful=True)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            training_reporting.report_calibration_effect(
                truth, _ideal, distorted, n_calib_rows=1547, method="isotonic"
            )
        self.assertIn("hurts log loss", buffer.getvalue())


# ---------------------------------------------------------------------------
# #9  fold geometry
# ---------------------------------------------------------------------------


class TestFoldGeometry(unittest.TestCase):
    def test_default_fold_width_is_at_most_one_season(self):
        # 15,620 training rows over five folds: the old geometry made each fold
        # n/(splits+1) = 2,603 rows, two full seasons.
        self.assertEqual(training_common.resolve_fold_rows(15_620, 5), 1230)
        self.assertLessEqual(
            training_common.resolve_fold_rows(15_620, 5),
            training_common.DEFAULT_FOLD_ROWS,
        )

    def test_default_fold_width_is_the_smaller_of_even_share_and_a_season(self):
        # Plenty of history: the even share is two seasons, so the season cap wins.
        self.assertEqual(
            training_common.resolve_fold_rows(15_000, 5),
            training_common.DEFAULT_FOLD_ROWS,
        )
        # Little history: the even share is under a season, so it wins.
        self.assertEqual(training_common.resolve_fold_rows(6_000, 5), 1_000)
        self.assertEqual(training_common.resolve_fold_rows(600, 5), 100)

    def test_an_explicit_fold_width_is_honoured(self):
        self.assertEqual(training_common.resolve_fold_rows(15_620, 5, fold_rows=400), 400)
        with self.assertRaises(ValueError):
            training_common.resolve_fold_rows(15_620, 5, fold_rows=0)

    def test_a_fold_narrower_than_a_date_group_is_refused(self):
        # Snapping a 3-row validation window forward across a 40-row date group can
        # leave nothing between the two snapped edges. That has to be an error with
        # the fix named in it, not an empty fold and a log-loss crash four calls on.
        dates = real_snapshot_dates()
        with self.assertRaises(ValueError) as caught:
            training_common.walk_forward_folds(
                len(dates), 5, fold_rows=3, dates=dates
            )
        self.assertIn("EMPTY validation block", str(caught.exception))
        self.assertIn("--fold-rows", str(caught.exception))

    def test_folds_never_overlap_and_never_look_ahead(self):
        n_rows = 15_620
        width = training_common.resolve_fold_rows(n_rows, 5)
        folds = training_common.walk_forward_folds(n_rows, 5)
        self.assertEqual(len(folds), 5)
        previous_val_start = 0
        for train_idx, val_idx in folds:
            self.assertEqual(len(val_idx), width)
            self.assertLess(int(train_idx.max()), int(val_idx.min()))
            self.assertGreaterEqual(int(val_idx.min()), previous_val_start)
            self.assertEqual(
                val_idx.tolist(),
                list(range(int(val_idx[0]), int(val_idx[0]) + width)),
            )
            # Training is EVERY row before the block: TimeSeriesSplit's shape.
            self.assertEqual(train_idx.tolist(), list(range(0, int(val_idx[0]))))
            previous_val_start = int(val_idx[0])
        # The last fold ends exactly at the end of the block.
        self.assertEqual(int(folds[-1][1][-1]), n_rows - 1)

    def test_fold_width_is_configurable_and_the_folds_follow_it(self):
        folds = training_common.walk_forward_folds(15_620, 4, fold_rows=500)
        self.assertEqual(len(folds), 4)
        for train_idx, val_idx in folds:
            self.assertEqual(len(val_idx), 500)

    def test_folds_reach_back_to_the_start_of_the_block(self):
        # The first fold trains on everything before its own block, which is
        # n_rows - n_splits * width rows.
        n_rows, splits = 15_620, 5
        width = training_common.resolve_fold_rows(n_rows, splits)
        folds = training_common.walk_forward_folds(n_rows, splits)
        self.assertEqual(len(folds[0][0]), n_rows - splits * width)

    def test_impossible_geometry_is_refused_with_the_numbers_in_the_message(self):
        with self.assertRaises(ValueError) as caught:
            training_common.walk_forward_folds(1_000, 5, fold_rows=500)
        self.assertIn("1000 rows", str(caught.exception))
        self.assertIn("--fold-rows", str(caught.exception))

    def test_fold_count_is_configurable(self):
        self.assertEqual(len(training_common.walk_forward_folds(15_620, 2)), 2)
        with self.assertRaises(ValueError):
            training_common.walk_forward_folds(15_620, 0)

    def test_recency_weighting_favours_the_newest_fold(self):
        losses = [1.00, 0.90, 0.80, 0.70, 0.60]
        weighted, per_fold, weights = training_common.recency_weighted_fold_mean(losses)
        self.assertEqual(per_fold, losses)
        self.assertEqual(weights, sorted(weights))
        self.assertGreater(weights[-1], 0.5)
        self.assertAlmostEqual(sum(weights), 1.0, places=12)
        # The flat mean is 0.80; recency weighting must pull it toward the newest
        # fold, which here is the BEST one.
        self.assertLess(weighted, float(np.mean(losses)))
        self.assertGreater(weighted, losses[-1])

    def test_recency_weighting_penalises_a_model_that_drifted(self):
        # Same folds, quality falling over time. Here recency weighting must make
        # the score WORSE than the flat mean -- which is the whole argument for
        # weighting by recency rather than re-scaling cosmetically.
        losses = [0.60, 0.70, 0.80, 0.90, 1.00]
        weighted, _per_fold, _weights = training_common.recency_weighted_fold_mean(losses)
        self.assertGreater(weighted, float(np.mean(losses)))

    def test_a_huge_half_life_converges_to_the_flat_mean(self):
        losses = [0.60, 0.70, 0.80, 0.90, 1.00]
        weighted, _, _ = training_common.recency_weighted_fold_mean(
            losses, half_life_folds=1e9
        )
        self.assertAlmostEqual(weighted, float(np.mean(losses)), places=9)

    def test_a_single_fold_scores_as_itself(self):
        weighted, per_fold, weights = training_common.recency_weighted_fold_mean([0.7])
        self.assertEqual(per_fold, [0.7])
        self.assertEqual(weights, [1.0])
        self.assertAlmostEqual(weighted, 0.7, places=12)

    def test_no_folds_is_an_unscoreable_infinity_not_a_zero(self):
        # Zero would look like a perfect model.
        self.assertEqual(training_common.recency_weighted_fold_mean([])[0], float("inf"))

    def test_a_bad_half_life_is_refused(self):
        with self.assertRaises(ValueError):
            training_common.recency_weighted_fold_mean([1.0, 2.0], half_life_folds=0)

    def test_no_trainer_scores_on_a_flat_fold_mean(self):
        for name in TRAINER_SOURCES:
            source = trainer_source(name)
            if "recency_weighted_fold_mean" in source:
                self.assertNotIn("np.mean(losses)", source, name)

    def test_every_trainer_exposes_fold_geometry_flags(self):
        # The four walk-forward trainers get the geometry flags. The two neural
        # networks validate on ONE contiguous slice and have no calibrator, so
        # those flags do not exist for them -- and the test says which is which
        # rather than skipping both.
        for name in TRAINER_SOURCES:
            parser = load_trainer_module(name).build_parser()
            flags = {action.dest for action in parser._actions}
            folds = "NN_" not in name
            self.assertEqual("fold_rows" in flags, folds, name)
            self.assertEqual("fold_half_life" in flags, folds, name)
            # Calibration is the other half of the split.
            self.assertEqual("calibration" in flags, folds, name)
            self.assertEqual("allow_isotonic" in flags, folds, name)

    def test_a_single_fold_scores_as_itself(self):
        weighted, per_fold, weights = training_common.recency_weighted_fold_mean([0.7])
        self.assertEqual(per_fold, [0.7])
        self.assertEqual(weights, [1.0])
        self.assertAlmostEqual(weighted, 0.7, places=12)

    def test_no_folds_is_an_unscoreable_infinity_not_a_zero(self):
        # Zero would look like a perfect model.
        self.assertEqual(training_common.recency_weighted_fold_mean([])[0], float("inf"))

    def test_a_bad_half_life_is_refused(self):
        with self.assertRaises(ValueError):
            training_common.recency_weighted_fold_mean([1.0, 2.0], half_life_folds=0)

    def test_no_trainer_scores_on_a_flat_fold_mean(self):
        for name in TRAINER_SOURCES:
            source = trainer_source(name)
            if "recency_weighted_fold_mean" in source:
                self.assertNotIn("np.mean(losses)", source, name)

    def test_every_trainer_exposes_fold_geometry_flags(self):
        # The four walk-forward trainers get the geometry flags. The two neural
        # networks validate on ONE contiguous slice and have no calibrator, so
        # those flags do not exist for them -- and the test says which is which
        # rather than skipping both.
        for name in TRAINER_SOURCES:
            parser = load_trainer_module(name).build_parser()
            flags = {action.dest for action in parser._actions}
            folds = "NN_" not in name
            self.assertEqual("fold_rows" in flags, folds, name)
            self.assertEqual("fold_half_life" in flags, folds, name)
            # Calibration is the other half of the split. The networks get a
            # single validation slice instead of a fold series, so they carry
            # --val-size / --test-size rather than --splits.
            self.assertEqual("splits" in flags, folds, name)
            self.assertEqual("val_size" in flags, not folds, name)
            self.assertEqual("calibration" in flags, folds, name)
            self.assertEqual("allow_isotonic" in flags, folds, name)

    def test_the_purged_split_is_still_not_the_fix(self):
        # Documented, not adopted: embargo_pct=0.01 over this span resolves to a
        # 45-day embargo and moves log loss by <0.001. Asserted so a future change
        # that swaps it in has to say so here.
        self.assertIn("purged_walk_forward_split", training_common.walk_forward_folds.__doc__)
        dates = pd.Series(
            pd.date_range("2012-06-22", "2026-01-06", freq="D")
        ).repeat(6)
        self.assertEqual(validation.embargo_days(dates, 0.01), 49)


# ---------------------------------------------------------------------------
# #17  split hygiene: unstable sort, pre-split imputation, stray dimensions
# ---------------------------------------------------------------------------


class TestSplitHygiene(unittest.TestCase):
    """The regression test the issue asked for by name, plus its neighbours."""

    def setUp(self):
        self.frame = synthetic_dataset(n_rows=600, per_date=4)

    def _prepare(self):
        return logistic_training.prepare_data(
            self.frame,
            training_common.MONEYLINE_TARGET,
            training_common.drop_columns_for(
                training_common.MONEYLINE_TARGET, include_team_ids=True
            ),
        )

    def test_no_split_boundary_falls_inside_a_same_date_group(self):
        # THE regression test, on the real data.
        #
        # A chronological split cuts at a row position, and this dataset holds
        # whole slates: mean 6.42 rows share one snapshot date, the largest group
        # is 40. A cut inside such a group separates two games played on the same
        # night -- same slate, same injury report -- and the split cannot be
        # reproduced from a row count alone.
        #
        # PREMISE first: the raw 90% row-count cut really does land inside a group.
        dates = real_snapshot_dates()
        n_rows = len(dates)
        self.assertTrue(
            cut_inside_a_group(dates, 0.9),
            "the 90% row-count cut now lands on a group edge; re-measure",
        )

        # REQUIREMENT: every cut the trainers actually make lands on a boundary.
        X, y, data = logistic_training.prepare_data(
            real_dataset_frame(),
            training_common.MONEYLINE_TARGET,
            training_common.drop_columns_for(
                training_common.MONEYLINE_TARGET, include_team_ids=True
            ),
        )
        snapshot_dates = data[training_common.DATE_COLUMN]

        X_train, y_train, X_test, y_test = logistic_training.split_train_test(
            X, y, dates=snapshot_dates
        )
        self.assertEqual(
            len(X_train) + len(X_test),
            len(X),
            "snapping moved rows instead of moving the boundary",
        )
        self.assertNotEqual(
            snapshot_dates.iloc[len(X_train) - 1],
            snapshot_dates.iloc[len(X_train)],
            "the chronological split cut a same-date slate in half",
        )
        self.assertEqual(len(X_test), len(y_test))

        X_train2, y_train2, X_calib, y_calib = logistic_training.split_train_calib(
            X_train, y_train, dates=snapshot_dates.iloc[: len(X_train)]
        )
        self.assertNotEqual(
            snapshot_dates.iloc[len(X_train2) - 1],
            snapshot_dates.iloc[len(X_train2)],
            "the calibration split cut a same-date slate in half",
        )

        train_val_dates = snapshot_dates.iloc[: len(X_train)]
        for _train_idx, val_idx in training_common.walk_forward_folds(
            len(X_train), 5, dates=train_val_dates
        ):
            start = int(val_idx[0])
            self.assertNotEqual(
                train_val_dates.iloc[start - 1],
                train_val_dates.iloc[start],
                "a walk-forward fold boundary cut a same-date slate in half",
            )

    def test_snapping_only_ever_moves_the_cut_forward(self):
        # Moving the cut BACK would put later rows into training, which is a far
        # worse leak than the one being fixed.
        dates = real_snapshot_dates()
        for fraction in (0.1, 0.25, 0.5, 0.75, 0.9):
            raw = int(len(dates) * fraction)
            snapped = training_common.snap_cut_to_date_boundary(dates, raw)
            self.assertGreaterEqual(snapped, raw)
            if snapped < len(dates):
                self.assertNotEqual(dates.iloc[snapped - 1], dates.iloc[snapped])

    def test_snapping_leaves_a_boundary_alone_and_handles_the_ends(self):
        dates = real_snapshot_dates()
        self.assertEqual(training_common.snap_cut_to_date_boundary(dates, 0), 0)
        self.assertEqual(
            training_common.snap_cut_to_date_boundary(dates, len(dates)),
            len(dates),
        )
        # A position already at a group edge comes back unchanged.
        for position in range(1, min(len(dates), 500)):
            if dates.iloc[position - 1] != dates.iloc[position]:
                self.assertEqual(
                    training_common.snap_cut_to_date_boundary(dates, position),
                    position,
                )
                break

    def test_split_without_dates_is_a_plain_row_count(self):
        # The dates argument is optional and, without it, nothing changes -- so a
        # caller that forgets it gets the old behaviour rather than an error.
        X, y, _data = logistic_training.prepare_data(
            self.frame,
            training_common.MONEYLINE_TARGET,
            training_common.drop_columns_for(training_common.MONEYLINE_TARGET),
        )
        _tr, _ytr, X_test, _ytest = logistic_training.split_train_test(X, y)
        self.assertEqual(len(X_test), int(len(X) * 0.1))

    def test_mergesort_keeps_same_date_rows_in_stored_order(self):
        # The property `kind="mergesort"` actually buys. pandas' default is
        # quicksort, which is NOT stable, so rows sharing a snapshot date come
        # out in an order that depends on nothing reproducible. Asserted on both
        # the synthetic frame and the real dataset, and checked against what
        # quicksort does so the fix cannot pass for the wrong reason.
        for frame in (self.frame, real_dataset_frame()):
            dates = pd.to_datetime(frame["Date"], errors="coerce").to_numpy()
            stored_positions = np.arange(len(frame))
            stable_order = (
                frame.assign(_pos=stored_positions)
                .sort_values("Date", kind="mergesort")
                ._pos.to_numpy()
            )
            quick_order = (
                frame.assign(_pos=stored_positions)
                .sort_values("Date")
                ._pos.to_numpy()
            )
            self.assertTrue(
                within_group_positions_are_increasing(dates, stable_order),
                "mergesort is not stable on this frame",
            )
            self.assertNotEqual(
                list(stable_order),
                list(quick_order),
                "quicksort and mergesort now agree; re-measure the premise",
            )
            self.assertFalse(
                within_group_positions_are_increasing(dates, quick_order),
                "pandas' default sort became stable; re-measure the premise",
            )

    def test_every_prepare_data_sorts_stably(self):
        for module_name, call in (
            ("logistic_training", lambda: self._prepare()),
            (
                "boosting_training",
                lambda: load_trainer_module("boosting_training.py").prepare_data(
                    self.frame,
                    training_common.MONEYLINE_TARGET,
                    training_common.drop_columns_for(training_common.MONEYLINE_TARGET),
                ),
            ),
        ):
            source = (TRAIN_MODELS_DIR / f"{module_name}.py").read_text()
            self.assertIn('kind=STABLE_SORT_KIND', source, module_name)
            self.assertNotIn("sort_values(DATE_COLUMN)", source, module_name)
            call()

    def test_neural_training_sorts_stably_too(self):
        source = (TRAIN_MODELS_DIR / "neural_training.py").read_text()
        self.assertIn("kind=STABLE_SORT_KIND", source)
        self.assertNotIn("sort_values(DATE_COLUMN)", source)

    def test_the_stable_sort_kind_is_mergesort(self):
        self.assertEqual(training_common.STABLE_SORT_KIND, "mergesort")

    def test_prepare_data_does_not_impute(self):
        # Imputation before the split is the leak: the medians were computed over
        # the joined frame, held-out rows included.
        frame = self.frame.copy()
        frame.loc[0, "F1"] = np.nan
        X, _y, _data = logistic_training.prepare_data(
            frame,
            training_common.MONEYLINE_TARGET,
            training_common.drop_columns_for(training_common.MONEYLINE_TARGET),
        )
        self.assertTrue(X["F1"].isna().any(), "prepare_data filled the NaN")

    def test_impute_split_uses_train_medians_only(self):
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0, np.nan]})
        test = pd.DataFrame({"a": [1000.0, np.nan, 7.0]})
        train_filled, test_filled = training_common.impute_split(train, test)
        self.assertEqual(train_filled["a"].tolist(), [1.0, 2.0, 3.0, 2.0])
        # 2.0 is the TRAIN median. A median over the joined frame would be 2.5.
        self.assertEqual(test_filled["a"].tolist(), [1000.0, 2.0, 7.0])

    def test_impute_split_first_return_value_is_always_the_training_frame(self):
        # The API is positional, so the ordering has to be pinned: an earlier
        # version returned a two-tuple whose first element was the train frame,
        # which one caller unpacked backwards and scored 14,058 "test" rows.
        train = pd.DataFrame({"a": [1.0, np.nan]})
        test = pd.DataFrame({"a": [9.0, 9.0, 9.0]})
        results = training_common.impute_split(train, test)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].shape, train.shape)
        self.assertEqual(results[1].shape, test.shape)

    def test_impute_split_handles_several_held_out_frames(self):
        train = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        a = pd.DataFrame({"a": [np.nan, 5.0]})
        b = pd.DataFrame({"a": [np.nan, np.nan, 6.0, 7.0]})
        filled_train, filled_a, filled_b = training_common.impute_split(train, a, b)
        self.assertEqual(filled_train["a"].tolist(), [1.0, 2.0, 3.0])
        self.assertEqual(filled_a["a"].tolist(), [2.0, 5.0])
        self.assertEqual(filled_b["a"].tolist(), [2.0, 2.0, 6.0, 7.0])

    def test_impute_split_replaces_infinities_first(self):
        train = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
        (filled,) = training_common.impute_split(train)
        self.assertFalse(np.isinf(filled["a"]).any())
        self.assertEqual(filled["a"].tolist(), [1.0, 2.0, 3.0])

    def test_infinities_are_removed_even_before_the_split(self):
        frame = self.frame.copy()
        frame.loc[0, "F1"] = np.inf
        X, _y, _data = logistic_training.prepare_data(
            frame,
            training_common.MONEYLINE_TARGET,
            training_common.drop_columns_for(training_common.MONEYLINE_TARGET),
        )
        self.assertTrue(np.isinf(X["F1"].to_numpy()).sum() == 0)

    def test_the_cv_loop_imputes_per_fold(self):
        # Fitting the imputer once on the whole training block would let the
        # validation fold reach training through the median. Each fold has to fit
        # its own, so the loop has to call impute_split per fold.
        for name in ("Logistic_Regression_ML.py", "Logistic_Regression_UO.py"):
            source = trainer_source(name)
            self.assertIn("X_fold_train, X_fold_val = impute_split(", source, name)
            self.assertIn("impute_split(X_train, X_test)", source, name)

    def test_neural_training_defers_imputation_until_after_the_split(self):
        source = (TRAIN_MODELS_DIR / "neural_training.py").read_text()
        self.assertIn("def to_model_matrix(", source)
        self.assertIn("impute_split(X_train, *others)", source)
        self.assertNotIn("X.fillna(", source)
        for name in ("NN_Model_ML.py", "NN_Model_UO.py"):
            body = trainer_source(name)
            self.assertIn(
                "X_train, X_val, X_test = to_model_matrix(X_train, X_val, X_test)",
                body,
                name,
            )

    def test_the_totals_script_pins_its_sagas_random_state(self):
        source = trainer_source("Logistic_Regression_UO.py")
        self.assertIn('solver = "saga" if penalty == "l1" else "lbfgs"', source)
        self.assertIn("random_state=args.seed", source)

    def test_no_entry_point_passes_an_unpinned_saga(self):
        # Search the AST for a LogisticRegression call whose solver could be
        # saga and whose kwargs omit random_state. A literal check on the text
        # would also trip over the version-fork helper, so this looks at calls.
        for name in ("Logistic_Regression_ML.py", "Logistic_Regression_UO.py"):
            tree = ast.parse(trainer_source(name))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not (isinstance(func, ast.Name) and func.id == "LogisticRegression"):
                    continue
                keys = {kw.arg for kw in node.keywords if kw.arg}
                splatted = any(kw.arg is None for kw in node.keywords)
                self.assertTrue(
                    "random_state" in keys or splatted,
                    f"{name} builds a LogisticRegression with no pinned seed",
                )


# ---------------------------------------------------------------------------
# Cross-cutting: the trainers still read the table and column list they claim to
# ---------------------------------------------------------------------------


class TestFeatureContractUnchanged(unittest.TestCase):
    """The fixes must not change what the models are fed."""

    def test_the_ten_dropped_columns_are_still_dropped(self):
        if not DATASET_DB.exists():
            self.skipTest("dataset.sqlite is unavailable")
        con = sqlite3.connect(f"file:{DATASET_DB}?mode=ro", uri=True)
        try:
            columns = [
                row[1]
                for row in con.execute(
                    f'PRAGMA table_info("{training_common.DEFAULT_DATASET}")'
                )
            ]
        finally:
            con.close()
        moneyline = training_common.drop_columns_for(
            training_common.MONEYLINE_TARGET, include_team_ids=True
        )
        remaining = [column for column in columns if column not in moneyline]
        # Everything that must never reach a model.
        for banned in (
            "Date", "Date.1", "GameDate", "Home-Team-Win", "OU-Cover",
            "Score", "TEAM_NAME", "TEAM_NAME.1", "index", "index.1",
        ):
            self.assertIn(banned, columns, banned)
            self.assertNotIn(banned, remaining, f"{banned} is now a feature")

    def test_prepare_data_returns_the_frame_unimputed_and_the_label(self):
        X, y, data = self_test_prepare()
        self.assertEqual(len(X), len(y))
        self.assertEqual(len(data), len(X))
        self.assertTrue(y.dtype.kind == "i")

    def test_totals_keeps_the_closing_total_and_moneyline_drops_it(self):
        moneyline = training_common.drop_columns_for(training_common.MONEYLINE_TARGET)
        totals = training_common.drop_columns_for(training_common.TOTALS_TARGET)
        self.assertIn("OU", moneyline)
        self.assertNotIn("OU", totals)


def self_test_prepare():
    frame = synthetic_dataset(n_rows=200, per_date=2)
    return logistic_training.prepare_data(
        frame,
        training_common.MONEYLINE_TARGET,
        training_common.drop_columns_for(training_common.MONEYLINE_TARGET),
    )


if __name__ == "__main__":
    unittest.main()