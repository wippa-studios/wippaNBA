"""The shipped odds database names its season tables four different ways.

`scripts/closing_line.py::load_odds` used to probe only `odds_<season>_new`
and swallow every miss with a bare `except: continue`. Two seasons are stored
under different names, so 1,880 closing lines were discarded and every market
number in this repository was computed without the two most recent seasons
(#47).

A guard that cannot distinguish "the data is absent" from "I failed to find
it" will always report absence. These tests pin the distinction.
"""

import sqlite3

import pytest

import closing_line as cl


class TestOddsTableResolution:
    def test_every_season_in_the_list_resolves_to_a_table(self):
        """Each of SEASONS must resolve. A miss is an error, not a skip."""
        con = sqlite3.connect(f"file:{cl.ODDS_DB}?mode=ro", uri=True)
        try:
            existing = {r[0] for r in
                        con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        finally:
            con.close()

        unresolved = [
            s for s in cl.SEASONS
            if not any(t.format(season=s) in existing for t in cl.ODDS_TABLE_CANDIDATES)
        ]
        assert unresolved == [], (
            f"seasons in cl.SEASONS with no odds table: {unresolved}. Either the "
            "database is missing them or the candidate list is incomplete."
        )

    def test_the_last_two_seasons_are_actually_present(self):
        """2024-25 and 2025-26 are the two that were dropped.

        This is the regression test: it fails on the old single-name probe,
        because neither `odds_2024-25_new` nor `odds_2025-26_new` exists.
        """
        odds = cl.load_odds()
        seasons_in_frame = set(odds["SEASON"].unique())
        for s in ("2024-25", "2025-26"):
            assert s in seasons_in_frame, (
                f"{s} is in cl.SEASONS but contributed no rows to load_odds()"
            )

    def test_row_count_covers_the_two_recovered_seasons(self):
        """Not just present - non-trivial. Guards against an empty-table match."""
        odds = cl.load_odds()
        counts = odds.groupby("SEASON").size()
        assert counts["2024-25"] > 1000, counts.to_dict()
        assert counts["2025-26"] > 400, counts.to_dict()

    def test_all_seasons_contribute(self):
        odds = cl.load_odds()
        assert set(odds["SEASON"].unique()) == set(cl.SEASONS)


class TestMissingSeasonIsLoud:
    def test_an_unresolvable_season_raises_rather_than_skipping(self, monkeypatch):
        """The whole point: absence of a table must not look like absence of data.

        If load_odds ever goes back to swallowing the miss, a silently shorter
        frame is indistinguishable from a smaller dataset.
        """
        monkeypatch.setattr(cl, "SEASONS", ("1999-00",))
        with pytest.raises(RuntimeError, match="1999-00"):
            cl.load_odds()

    def test_the_error_names_the_season_and_the_names_tried(self, monkeypatch):
        monkeypatch.setattr(cl, "SEASONS", ("1999-00",))
        with pytest.raises(RuntimeError) as exc:
            cl.load_odds()
        msg = str(exc.value)
        assert "1999-00" in msg
        assert "odds_1999-00_new" in msg


class TestNoBareExcept:
    def test_load_odds_does_not_swallow_exceptions(self):
        """A bare `except: continue` around the table read is the bug.

        Guarded structurally rather than by behaviour, because a bare except is
        invisible in a passing test and only shows up as missing data later.
        """
        import inspect
        src = inspect.getsource(cl.load_odds)
        assert "except Exception" not in src, (
            "load_odds must not swallow table-read failures; see #47"
        )
        assert "except:" not in src.replace("except sqlite3", "")