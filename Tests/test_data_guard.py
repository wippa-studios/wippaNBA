"""The data guard must skip, and it must skip for the right reason.

`src/DataProviders/nba_data.py` raises `unittest.SkipTest` from the loaders rather
than from the test files, because two runners execute this suite - `pytest` and
`python -m unittest discover` - and they honour different skip mechanisms. Putting
the check in the shared module makes one change satisfy both.

THAT GUARD WAS NOT ENOUGH, AND CI PROVED IT
-------------------------------------------
CI on `master` was red for two consecutive commits. Every error was the same:

    ValueError: No objects to concatenate
      nba_data.load_team_games -> pd.concat([])

The guard checked `(DATA_ROOT / "game_logs").is_dir()`. A directory existing is
not the same as data being in it: an empty `game_logs/` passes that check, every
subsequent glob returns nothing, and `pd.concat` is handed an empty list. So the
exact failure the guard was written to prevent still got through, by the one route
the guard did not cover.

Two defects, both fixed and both pinned here:

1. The guard tested for a DIRECTORY, not for FILES.
2. `DATA_ROOT` defaulted to one developer's home directory,
   `/home/Joel/Desktop/Legit_NBA_Predictions/data/nba`. In a shared repository
   that is wrong for everyone else: someone who ran the downloader exactly as the
   README says, into `./data/nba`, was told the data was missing.

Every scenario runs in a SUBPROCESS. `DATA_ROOT` is resolved at import time, so
reloading the module in-process to change it is fragile and order-dependent - an
earlier draft of this file did exactly that and failed three ways. A subprocess is
also simply how CI sees it: a fresh interpreter, an environment variable, and a
directory that may or may not exist.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

PROBE = r"""
import sys, unittest
sys.path.insert(0, %r)
from DataProviders import nba_data as nd
try:
    g = nd.build_games()
    print("RESULT|ROWS|" + str(len(g)))
except unittest.SkipTest as e:
    print("RESULT|SKIP|" + str(e))
except Exception as e:
    print("RESULT|ERROR|" + type(e).__name__ + ": " + str(e)[:90])
""" % str(ROOT / "src")


def _probe(data_root=None, repo_default=False):
    """Run one fresh interpreter and return (kind, detail)."""
    env = dict(os.environ)
    env.pop("NBA_DATA_ROOT", None)
    if data_root is not None:
        env["NBA_DATA_ROOT"] = data_root
    out = subprocess.run([PY, "-c", PROBE], capture_output=True, text=True,
                         env=env, cwd=str(ROOT), timeout=600)
    for line in out.stdout.splitlines():
        if line.startswith("RESULT|"):
            _, kind, detail = line.split("|", 2)
            return kind, detail
    return "CRASH", (out.stderr or out.stdout)[-300:]


class TestMissingDataSkipsRatherThanRaises(unittest.TestCase):
    def test_a_missing_directory_skips(self):
        kind, detail = _probe("/nonexistent/path/for/tests")
        self.assertEqual(kind, "SKIP", f"expected a skip, got {kind}: {detail}")

    def test_an_empty_game_logs_directory_skips(self):
        """THE CI FAILURE. A directory existing is not data being in it.

        `game_logs/` present but holding no CSVs used to sail past an `is_dir()`
        check and reach `pd.concat([])`.
        """
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "game_logs").mkdir()
            (Path(tmp) / "team_gamelogs").mkdir()
            kind, detail = _probe(tmp)
            self.assertEqual(
                kind, "SKIP",
                f"an empty game_logs/ must skip, not raise. Got {kind}: {detail}")

    def test_a_partly_downloaded_directory_skips(self):
        """game_logs populated, team_gamelogs still empty.

        The downloader writes tables one stage at a time, so an interrupted run
        leaves exactly this state. It must be treated as no data rather than
        reaching `pd.concat([])` on the second glob.

        Note what is NOT required: `player_gamelogs` and `schedule`. The real
        download has them, but the synthetic fixtures in this repo deliberately do
        not, and a guard that demands a full download to run a unit test is an
        obstacle rather than a protection.
        """
        with tempfile.TemporaryDirectory() as tmp:
            logs = Path(tmp) / "game_logs"
            logs.mkdir()
            (logs / "regular_season_2023-24.csv").write_text(
                "GAME_ID,TEAM_ABBREVIATION,GAME_DATE,PTS\n0022300001,BOS,2023-10-24,110\n")
            (Path(tmp) / "team_gamelogs").mkdir()
            kind, detail = _probe(tmp)
            self.assertEqual(kind, "SKIP",
                             f"a partial download must skip. Got {kind}: {detail}")

    def test_the_message_names_the_path_and_the_fix(self):
        with tempfile.TemporaryDirectory() as tmp:
            kind, detail = _probe(tmp)
            self.assertEqual(kind, "SKIP")
            self.assertIn(tmp, detail, "must name the path it looked at")
            self.assertIn("download_nba_data.py", detail, "must say how to fix it")


class TestTheDefaultPathWorksForAnyone(unittest.TestCase):
    def test_default_resolves_inside_the_repository(self):
        """The default must work for anyone who followed the README.

        It was one developer's absolute home path, so on every other machine
        `DATA_ROOT` pointed somewhere that could never exist.
        """
        probe = (
            "import sys; sys.path.insert(0, %r)\n"
            "from DataProviders.nba_data import DATA_ROOT, _REPO_ROOT\n"
            "print('RESULT|ROOT|' + DATA_ROOT + '|' + str(_REPO_ROOT))\n"
        ) % str(ROOT / "src")
        out = subprocess.run([PY, "-c", probe], capture_output=True, text=True,
                             cwd=str(ROOT), timeout=600,
                             env={k: v for k, v in os.environ.items()
                                  if k != "NBA_DATA_ROOT"})
        line = next(l for l in out.stdout.splitlines() if l.startswith("RESULT|"))
        _, _, data_root, repo_root = line.split("|", 3)
        self.assertNotIn("/Desktop/Legit_NBA_Predictions",
                         data_root,
                         "DATA_ROOT still falls back to a personal Desktop path")
        self.assertTrue(
            data_root.startswith(repo_root) or
            data_root.startswith(str(Path(repo_root).parent)),
            f"DATA_ROOT {data_root} is neither in-repo nor the legacy sibling "
            f"of {repo_root}")

    def test_the_env_var_still_wins(self):
        """The override is how CI and a downloaded dataset are pointed at."""
        kind, detail = _probe("/tmp/opencode/definitely-not-here")
        self.assertEqual(kind, "SKIP")
        self.assertIn("/tmp/opencode/definitely-not-here", detail,
                      "the override was ignored; DATA_ROOT did not follow it")


class TestRealDataStillWorks(unittest.TestCase):
    def test_with_data_present_it_builds(self):
        """The guard must not fire when the data is actually there."""
        sys.path.insert(0, str(ROOT / "src"))
        from DataProviders.nba_data import DATA_ROOT

        if not list(Path(DATA_ROOT, "game_logs").glob("*.csv")):
            self.skipTest(f"no downloaded data at {DATA_ROOT}")
        kind, detail = _probe(DATA_ROOT)
        self.assertEqual(kind, "ROWS", f"expected rows, got {kind}: {detail}")
        self.assertGreater(int(detail), 1000,
                           f"suspiciously few rows: {detail}")


if __name__ == "__main__":
    unittest.main()