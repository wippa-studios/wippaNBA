"""Hard stop for maintained scripts when a derived data artefact is absent.

The repository ships no data: ``Data/`` holds only ``manifest.json`` and
``data/`` does not exist until ``download_nba_data.py`` builds it. Every sqlite
database and the ``data/nba`` CSV tree are build products. Before this guard, a
maintained script run on a fresh clone either died with a raw traceback deep
inside sqlite3 or pandas, or quietly produced a zero-row result that looked
like a measurement.

One behaviour, deliberately narrow: print a single clear message to stderr and
exit non-zero. No fallbacks, no empty databases, no dummy values. If a script
cannot run without an artefact, it says so in one line and stops.

    from _data_guard import require, require_data_dir

    def main():
        require(BASE_DIR / "Data" / "dataset.sqlite")
        require_data_dir()

The messages deliberately mirror ``tests/_data_guard.py``, which skips the
tests that need the same artefacts; the build commands below are that file's
BUILD_COMMANDS table, which is itself sourced from scripts/README_DATA.md.

A present-but-empty sqlite file is treated as absent. A zero-byte sqlite file
is what a failed ``sqlite3.connect`` leaves behind; it has no schema, and
reading it raises ``sqlite3.DatabaseError`` rather than producing a meaningful
result. Treating it as absent keeps one message for one condition.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Artefact -> the repository command that builds it. Same table, same entry
# points, as tests/_data_guard.py; none of them is invented here.
BUILD_COMMANDS = {
    "Data/TeamData.sqlite": (
        "python -m wippa_nba.legacy.process_data.Get_Data"
    ),
    "Data/OddsData.sqlite": (
        "python -m wippa_nba.legacy.process_data.Get_Odds_Data"
    ),
    "Data/dataset.sqlite": (
        "python -m wippa_nba.legacy.process_data.Create_Games"
    ),
    "data/nba": "python scripts/download_nba_data.py",
}

# The minimum a usable CSV tree must contain. Mirrors
# wippa_nba.data.nba_data._REQUIRED_DATA_DIRS; checked for actual CSVs, not
# merely for the directory existing, because an interrupted download leaves
# exactly the half-built state that crashes a loader.
_REQUIRED_DATA_DIRS = ("game_logs", "team_gamelogs")


def _fail(artefact: str, hint: str) -> None:
    print(
        f"{artefact} is not in this repository.\n"
        f"It is a derived artefact - rebuild it with `{hint}`; see "
        "scripts/README_DATA.md for how to rebuild it.",
        file=sys.stderr,
    )
    raise SystemExit(1)


def require(path, what: str | None = None, hint: str | None = None) -> Path:
    """Return ``path`` when the artefact is usable; exit with a clear message when not.

    ``path`` may be a Path or an artefact name relative to the repository root
    ("Data/dataset.sqlite"). ``what`` names the artefact in the message when the
    path alone is unclear. ``hint`` overrides the default rebuild pointer.
    """
    if isinstance(path, str):
        path = REPO_ROOT / path
    path = Path(path)
    try:
        rel = str(path.relative_to(REPO_ROOT))
    except ValueError:
        rel = str(path)

    hint = hint or BUILD_COMMANDS.get(
        rel, "the repository's documented build steps"
    )

    absent = False
    if not path.exists():
        absent = True
    elif path.is_file() and path.stat().st_size == 0:
        # Same rule, same reason, as tests/_data_guard.py.
        absent = True

    if absent:
        _fail(rel, hint)
    return path


def require_data_dir(what: str = "the downloaded NBA CSVs") -> Path:
    """Return the resolved ``data/nba`` root when a usable tree is present.

    Honours the same resolution order as wippa_nba.data.nba_data: the
    NBA_DATA_ROOT environment variable overrides everything, then the
    repository's own data/nba. Checks that the two required sub-directories
    each contain at least one CSV, because an empty directory produces the
    silent zero-row result this guard exists to prevent.
    """
    override = os.environ.get("NBA_DATA_ROOT")
    if override:
        root = Path(override)
        if not root.is_dir():
            _fail(
                f"data/nba (NBA_DATA_ROOT={override})",
                "python scripts/download_nba_data.py",
            )
    else:
        root = REPO_ROOT / "data" / "nba"
        if not root.is_dir():
            _fail("data/nba", "python scripts/download_nba_data.py")

    for sub in _REQUIRED_DATA_DIRS:
        d = root / sub
        has_csv = d.is_dir() and any(d.glob("*.csv"))
        if not has_csv:
            print(
                f"data/nba is present but has no CSVs under {sub}/.\n"
                "An interrupted download leaves exactly this state; re-run "
                "`python scripts/download_nba_data.py` "
                "(see scripts/README_DATA.md).",
                file=sys.stderr,
            )
            raise SystemExit(1)
    return root
