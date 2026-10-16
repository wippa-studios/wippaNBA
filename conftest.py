"""Make the repository root importable for the test suite.

Two of the five existing test modules do ``from src.Utils import ...``, which
only resolves if the repo root is on ``sys.path``. ``python -m pytest`` masks
that need because running a module puts the current working directory on
``sys.path`` implicitly; invoking the ``pytest`` console script directly does
not, and fails with ``ModuleNotFoundError: No module named 'src'``.

Inserting the repo root explicitly fixes every invocation style -- bare
``pytest``, ``python -m pytest``, and plain ``unittest`` -- in one place. The
same setting is mirrored in ``pyproject.toml`` under
``[tool.pytest.ini_options] pythonpath``; this file is the fallback for anyone
running pytest < 7.0, where that ini option does not exist.

The hyphenated directories ``src/Process-Data`` and ``src/Train-Models`` are
NOT renamed here -- they are not valid Python identifiers, so the tests that
need those modules load them by file path with
``importlib.util.spec_from_file_location`` instead.
"""

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------- #
# Skip the model suites when the NBA data has not been downloaded
# --------------------------------------------------------------------------- #
# src/Models/*.py read real game data from src/DataProviders/nba_data.py, which
# resolves DATA_ROOT to a local download directory (382 MB, 310 CSVs). That
# directory is deliberately not in the repository, so on a fresh clone or a CI
# runner the loaders find nothing and raise `ValueError: No objects to
# concatenate`. That is a missing-input condition, not a defect in the models,
# and it used to fail the whole suite at collection time.
#
# Downloading is a deliberate step (see the Quick start section of the README),
# so when the data is absent these tests skip with an actionable message rather
# than reporting a false failure. When the data IS present they run in full.

DATA_ROOT = Path(os.environ.get(
    "NBA_DATA_ROOT", "/home/Joel/Desktop/Legit_NBA_Predictions/data/nba"))
NBA_DATA_PRESENT = (DATA_ROOT / "game_logs").is_dir()

DATA_DEPENDENT_TESTS = (
    "test_baselines",
    "test_gbm",
    "test_logreg_rolling",
    "test_margin_total",
    "test_player_ratings",
    "test_ratings",
)


def pytest_collection_modifyitems(config, items):
    """Skip data-dependent model tests when the NBA data is not present."""
    if NBA_DATA_PRESENT:
        return
    missing = [i for i in items
               if any(k in Path(str(i.fspath)).name for k in DATA_DEPENDENT_TESTS)]
    if not missing:
        return
    reason = (
        f"NBA data not found at {DATA_ROOT}. These model tests need the download "
        f"from download_nba_data.py (see README 'What this does'). Set "
        f"NBA_DATA_ROOT to point at it."
    )
    skipper = pytest.mark.skip(reason=reason)
    for item in missing:
        item.add_marker(skipper)