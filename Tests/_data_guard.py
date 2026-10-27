"""Explicit skips for tests that need a dataset the repository does not ship.

Only ``Data/manifest.json`` is committed. The sqlite databases and the
downloaded NBA CSVs are build products, so a fresh clone can run most of the
suite but not the tests that read them. Those tests must skip, visibly, with a
message that names the missing artefact and the command that builds it.

``unittest.SkipTest`` is raised rather than ``pytest.skip`` so the same guard
works under both entry points:

    python -m pytest Tests/ -q
    python -m unittest discover -s Tests -t .

pytest reports ``unittest.SkipTest`` as a skip for both ``unittest.TestCase``
classes and plain test classes, so there is one mechanism, not two.

A present-but-empty file is treated as absent. A zero-byte sqlite file is what a
failed ``sqlite3.connect`` leaves behind; it has no schema, and reading it
raises ``sqlite3.DatabaseError`` rather than producing a meaningful skip. The
message says "empty" instead of "absent" so the two states stay distinguishable.
"""

from __future__ import annotations

import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Artefact -> the repository command that builds it. These are the entry points
# already documented in scripts/README_DATA.md and the module docstrings; none
# of them is invented here.
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


def require_dataset(name: str, *also: str) -> Path:
    """Return the path to ``name``, or skip the calling test.

    ``also`` names any further artefacts the same test needs. The first absent
    one is the one reported, so the skip message stays specific rather than
    listing every input the test would eventually touch.
    """
    for artefact in (name,) + also:
        path = REPO_ROOT / artefact
        if not path.exists():
            reason = f"{artefact} is absent"
        elif path.is_file() and path.stat().st_size == 0:
            reason = f"{artefact} is empty (0 bytes)"
        else:
            continue
        build = BUILD_COMMANDS[artefact]
        raise unittest.SkipTest(
            f"{reason}; this test needs the shipped data. Build it with "
            f"`{build}` (see scripts/README_DATA.md)."
        )
    return REPO_ROOT / name
