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

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))