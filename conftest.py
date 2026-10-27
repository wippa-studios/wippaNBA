"""Make the repository's ``src`` root importable for the test suite.

Every application module now lives in the ``wippa_nba`` package under ``src``,
so the tests need ``src`` on ``sys.path``. ``python -m pytest`` masks that need
because running a module puts the current working directory on ``sys.path``
implicitly; invoking the ``pytest`` console script directly does not, and fails
with ``ModuleNotFoundError: No module named 'wippa_nba'``.

Inserting the source root explicitly fixes every invocation style in one place.
The same setting is mirrored in ``pyproject.toml`` under
``[tool.pytest.ini_options] pythonpath``; this file is the fallback for anyone
running pytest < 7.0, where that ini option does not exist.

The repository root stays on the path as well because ``scripts/`` modules and
``main.py`` are executed from there, and some tests import ``main`` directly.
"""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
SRC_ROOT = REPO_ROOT / "src"

for _path in (REPO_ROOT, SRC_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
