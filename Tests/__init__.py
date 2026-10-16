"""Test package marker.

Its presence is what allows ``python -m unittest discover -s Tests -t .`` to
work: with ``-t .`` set as the top-level directory, unittest requires ``Tests``
to be an importable package and otherwise aborts with
``ImportError: Start directory is not importable: '.../Tests'``.

The three modules that exercise ``src/Process-Data`` and ``src/Train-Models``
load their targets by file path via
``importlib.util.spec_from_file_location`` under synthetic names
(``create_games``, ``get_data``, ``get_odds_data``). Those names belong to no
package, so importing the tests as ``Tests.*`` cannot collide with them and
there is no double-import. ``Tests/test_create_games.py`` is the concrete
check: it runs ``main()`` against throwaway sqlite files in a
``tempfile.TemporaryDirectory`` and asserts on a single generated row, which
would fail loudly if the module were executed twice against stale state.
"""