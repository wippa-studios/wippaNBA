"""Test package marker.

Its presence is what allows ``python -m unittest discover -s tests -t .`` to
work: with ``-t .`` set as the top-level directory, unittest requires ``tests``
to be an importable package and otherwise aborts with
``ImportError: Start directory is not importable: '.../tests'``.

The three modules that exercise ``src/Process-Data`` and ``src/Train-Models``
load their targets by file path via
``importlib.util.spec_from_file_location`` under synthetic names
(``create_games``, ``get_data``, ``get_odds_data``). Those names belong to no
package, so importing the tests as ``tests.*`` cannot collide with them and
there is no double-import. ``tests/integration/test_create_games.py`` is the concrete
check: it runs ``main()`` against throwaway sqlite files in a
``tempfile.TemporaryDirectory`` and asserts on a single generated row, which
would fail loudly if the module were executed twice against stale state.
"""

import sys
from pathlib import Path

# ``python -m unittest discover`` does not read conftest.py, so the package
# marker is the only place that runs before any test module is imported. Both
# the repository root and ``src`` go on the path: the tests import ``main`` and
# ``scripts.*`` from the root, and ``wippa_nba`` from ``src``. The insertion is
# idempotent and has no other side effect, so importing this package twice is
# harmless.
_REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPO_ROOT, _REPO_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))