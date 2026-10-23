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
import sys
from pathlib import Path

import pandas as pd
import pytest

# `closing_line` lives in scripts/, which is not on sys.path by default. This
# module must be importable on its own - a bare
# `pytest Tests/test_odds_table_resolution.py` used to fail with
# ModuleNotFoundError and only worked in a full run because an earlier test
# module had inserted scripts/ first. Order-dependent imports are the same
# defect class as the one this file exists to pin.
ROOT = Path(__file__).resolve().parents[1]
for _candidate in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

import closing_line as cl  # noqa: E402


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


# --------------------------------------------------------------------------- #
# The same bug, reproduced on a fixture rather than the shipped database.
# --------------------------------------------------------------------------- #

def _make_odds_db(path, tables):
    """Create `path` holding one odds table per entry.

    An entry maps a table name to either an int (that many generated rows, each
    with a unique (Date, Home, Away) key) or an explicit list of row dicts. The
    schema mirrors the columns `load_odds` selects from the shipped tables.
    Counts differ per table so a dropped table or a dropped slice of rows is
    visible in the returned frame rather than hidden by symmetry.
    """
    con = sqlite3.connect(path)
    try:
        for name, spec in tables.items():
            if isinstance(spec, int):
                rows = [{
                    # Unique (Date, Home, Away) key per row: 1 + i // 28 walks
                    # the month so no two generated rows collide.
                    "Date": f"2025-{1 + i // 28:02d}-{(i % 28) + 1:02d}",
                    "Home": "Atlanta Hawks",
                    "Away": "Boston Celtics",
                    "ML_Home": -110 - i,
                    "ML_Away": -110 + i,
                    "OU": 210.5,
                    "Spread": -1.5,
                } for i in range(spec)]
            else:
                rows = spec
            # An empty spec must still write the SCHEMA: pandas' to_sql emits
            # `CREATE TABLE ... ()` for a columnless frame, which sqlite rejects.
            frame = (pd.DataFrame(rows) if rows else pd.DataFrame(
                columns=["Date", "Home", "Away", "ML_Home", "ML_Away",
                         "OU", "Spread"]))
            frame.to_sql(name, con, index=False)
    finally:
        con.close()
    return path


def _old_single_name_probe(db_path, seasons):
    """The pre-#47 loader, behaviour for behaviour: probe `odds_<season>_new`
    only and swallow every miss with a bare `except: continue`.

    Kept in the test rather than the loader so the row-drop #47 caused can be
    REPRODUCED on demand - a bug pinned only by "the new code works" cannot
    show a reader what the old code did.
    """
    con = sqlite3.connect(db_path)
    frames = []
    try:
        for s in seasons:
            try:
                d = pd.read_sql_query(f'SELECT * FROM "odds_{s}_new"', con)
            except Exception:
                continue
            if len(d):
                frames.append(d.assign(SEASON=s))
    finally:
        con.close()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


class TestIssue47OnAFixture:
    """The 1,880-line drop, reproduced on a small fixture.

    The shipped database names its season tables four different ways: 2024-25
    sits under a bare `2024-25`, 2025-26 under `odds_2025-26` with no `_new`
    twin. The fixture reproduces exactly that shape, and the old single-name
    probe demonstrably drops everything that is not `odds_<season>_new` (#47).
    """

    # The shipped database's shape, in miniature: one season under each of the
    # three spellings #47 turned on.
    TABLES = {"odds_2023-24_new": 30, "2024-25": 20, "odds_2025-26": 10}
    SEASONS = ("2023-24", "2024-25", "2025-26")

    def _load(self, tmp_path, monkeypatch, tables=None):
        db = _make_odds_db(tmp_path / "odds.sqlite", tables or self.TABLES)
        monkeypatch.setattr(cl, "ODDS_DB", db)
        monkeypatch.setattr(cl, "SEASONS", self.SEASONS)
        return cl.load_odds()

    def test_every_row_under_every_spelling_is_loaded(self, tmp_path, monkeypatch):
        """The regression test for the drop itself: 20 of the fixture's 60 rows
        live under a bare season name and 10 under a non-`_new` name. The old
        probe returned 30 rows and no error; the loader must return all 60."""
        odds = self._load(tmp_path, monkeypatch)
        counts = odds.groupby("SEASON").size().to_dict()
        assert counts == {"2023-24": 30, "2024-25": 20, "2025-26": 10}, counts
        assert len(odds) == sum(self.TABLES.values())

    def test_the_old_probe_reproduces_the_drop_on_the_same_fixture(
            self, tmp_path, monkeypatch):
        """The #47 failure mode, demonstrated rather than asserted.

        On the same fixture the old single-name probe returns 30 of 60 rows -
        the rows under the two unusual spellings are gone and nothing says so.
        That silent 50% loss is the bug in miniature; on the shipped database
        it was 1,880 closing lines and two whole seasons.
        """
        db = _make_odds_db(tmp_path / "odds.sqlite", self.TABLES)
        old = _old_single_name_probe(db, self.SEASONS)
        assert set(old["SEASON"].unique()) == {"2023-24"}, (
            "the old probe is supposed to see only odds_<season>_new tables"
        )
        assert len(old) == 30, "the old probe must reproduce the silent drop"
        monkeypatch.setattr(cl, "ODDS_DB", db)
        monkeypatch.setattr(cl, "SEASONS", self.SEASONS)
        assert len(cl.load_odds()) == 60, (
            "the fixed loader must not drop what the old probe dropped"
        )

    def test_an_empty_season_table_raises_instead_of_a_shorter_frame(
            self, tmp_path, monkeypatch):
        """A table that exists but holds no rows must not look like a season
        with no games. The old `if len(d)` skipped it silently."""
        tables = {"odds_2023-24_new": 5, "2024-25": 0, "odds_2025-26": 5}
        db = _make_odds_db(tmp_path / "odds.sqlite", tables)
        monkeypatch.setattr(cl, "ODDS_DB", db)
        monkeypatch.setattr(cl, "SEASONS", self.SEASONS)
        with pytest.raises(RuntimeError, match="2024-25"):
            cl.load_odds()

    def test_a_season_with_no_table_raises_naming_the_season_and_names_tried(
            self, tmp_path, monkeypatch):
        tables = {"odds_2023-24_new": 5, "2024-25": 5}
        db = _make_odds_db(tmp_path / "odds.sqlite", tables)
        monkeypatch.setattr(cl, "ODDS_DB", db)
        monkeypatch.setattr(cl, "SEASONS", self.SEASONS)
        with pytest.raises(RuntimeError) as exc:
            cl.load_odds()
        msg = str(exc.value)
        assert "2025-26" in msg
        assert "odds_2025-26_new" in msg

    def test_the_season_column_is_carried_so_coverage_is_visible(
            self, tmp_path, monkeypatch):
        """#47 hid for as long as it did partly because no caller could see
        which seasons a frame covered."""
        odds = self._load(tmp_path, monkeypatch)
        assert set(odds["SEASON"]) == set(self.SEASONS)


class TestTwinSpellingsReconcileOrFail:
    """MAJOR-2 of the #47 review: two spellings of one season must not resolve
    to "first match wins", which silently drops the twin's rows.

    The shipped database has twin spellings for 9 of its 11 seasons, so the
    fix here is SUM-AND-RECONCILE, not raise-on-any-twin: identical quotes
    under two spellings collapse with a count; a quote whose rows differ only
    in the date string (the legacy `odds_<s>` spelling stores `2015-16-1027`
    for `2015-10-27`) collapses to the row whose date parses. The one thing
    refused is AMBIGUITY: the same game quoting different moneylines under two
    spellings raises, because choosing a closing price is a measurement
    decision, not a loader's.
    """

    QUOTE = {"Home": "Atlanta Hawks", "Away": "Boston Celtics",
             "ML_Home": -150, "ML_Away": 130, "OU": 210.5, "Spread": -3.5}

    def _load(self, tmp_path, monkeypatch, tables, season="2023-24"):
        db = _make_odds_db(tmp_path / "odds.sqlite", tables)
        monkeypatch.setattr(cl, "ODDS_DB", db)
        monkeypatch.setattr(cl, "SEASONS", (season,))
        stats = {}
        odds = cl.load_odds(stats=stats)
        return odds, stats

    def test_identical_quotes_under_twin_spellings_reconcile_not_drop(
            self, tmp_path, monkeypatch):
        """Five rows across two spellings, three distinct closing lines: the
        loader must return three rows and account for the two collapses."""
        quote = self.QUOTE
        tables = {
            "odds_2023-24_new": [
                {"Date": "2023-10-24", **quote},
                {"Date": "2023-10-25", **quote,
                 "ML_Home": -120, "ML_Away": 100},
            ],
            "2023-24": [
                # The same first game under the legacy mangled date: one quote,
                # two date spellings.
                {"Date": "2023-24-1024", **quote},
                # The same second game byte for byte: an exact duplicate.
                {"Date": "2023-10-25", **quote,
                 "ML_Home": -120, "ML_Away": 100},
                # A game only this spelling carries - it must SURVIVE (the
                # 2023-24 pair is genuinely complementary).
                {"Date": "2023-10-26", **quote,
                 "ML_Home": 140, "ML_Away": -160},
            ],
        }
        odds, stats = self._load(tmp_path, monkeypatch, tables)
        assert sorted(odds["Date"]) == ["2023-10-24", "2023-10-25", "2023-10-26"], (
            "the parseable twin must win the date-format collapse and the "
            "complementary game must survive"
        )
        assert stats["rows_expected"] == 5
        assert stats["rows_collapsed"] == 2, stats
        assert stats["rows_returned"] == 3
        s = stats["seasons"]["2023-24"]
        assert s["collapsed_exact"] == 1 and s["collapsed_date_variant"] == 1, s

    def test_two_games_sharing_a_quote_but_with_two_parseable_dates_both_stay(
            self, tmp_path, monkeypatch):
        """The date-variant collapse must not eat two real games that happen
        to close at the same price - both dates parse, so both rows stay."""
        quote = self.QUOTE
        tables = {"odds_2023-24_new": [
            {"Date": "2023-10-24", **quote},
            {"Date": "2023-10-25", **quote},
        ]}
        odds, stats = self._load(tmp_path, monkeypatch, tables)
        assert len(odds) == 2
        assert stats["rows_collapsed"] == 0

    def test_conflicting_closing_moneylines_raise_rather_than_choose(
            self, tmp_path, monkeypatch):
        """The reconcile option is for duplicates. A genuine price conflict is
        AMBIGUITY and raises - the old first-match loader would have returned
        one of the two prices silently, and which one depended on table order."""
        quote = self.QUOTE
        tables = {
            "2024-25_new": [{"Date": "2024-10-22", **quote}],
            "2024-25": [{"Date": "2024-10-22", **quote,
                         "ML_Home": -155, "ML_Away": 135}],
        }
        db = _make_odds_db(tmp_path / "odds.sqlite", tables)
        monkeypatch.setattr(cl, "ODDS_DB", db)
        monkeypatch.setattr(cl, "SEASONS", ("2024-25",))
        with pytest.raises(RuntimeError, match="2024-25"):
            cl.load_odds()

    def test_same_price_with_a_differing_totals_line_reconciles(
            self, tmp_path, monkeypatch):
        """This script measures the MONEYLINE. One 2023-24 game disagrees only
        in OU across spellings; the preferred spelling's row is kept and the
        collapse is counted rather than silent."""
        quote = self.QUOTE
        tables = {
            "odds_2023-24_new": [{"Date": "2023-10-24", **quote}],
            "2023-24": [{"Date": "2023-10-24", **quote, "OU": 211.5}],
        }
        odds, stats = self._load(tmp_path, monkeypatch, tables)
        assert len(odds) == 1
        assert float(odds["OU"].iloc[0]) == 210.5, (
            "the most-preferred spelling supplies the ancillary line"
        )
        assert stats["seasons"]["2023-24"]["collapsed_ancillary"] == 1
        assert stats["rows_expected"] == 2 and stats["rows_collapsed"] == 1


class TestTheLoaderCountsWhatItReads:
    """The reconciliation tripwire, pinned structurally.

    The counts must come from `SELECT COUNT(*)` per table - independent of the
    read - so a future WHERE clause shrinks the read but not the count and the
    two disagree loudly (the review's MINOR-4). A check that never fires is
    invisible in a passing suite, hence source-level like `TestNoBareExcept`.
    """

    def test_counts_come_from_count_star_not_from_the_read(self):
        import inspect
        src = inspect.getsource(cl.load_odds)
        assert "SELECT COUNT(*)" in src, (
            "load_odds must count rows independently of the read; see MINOR-4"
        )
        assert "rows_expected" in src and "rows_collapsed" in src, (
            "load_odds must reconcile COUNT(*) against rows returned + rows "
            "collapsed; see #47 and MAJOR-2"
        )
        assert "len(d) != n_expected" in src.replace(" ", "") or \
            "len(d)!=n_expected" in src.replace(" ", ""), (
            "the read must be checked against COUNT(*); see MINOR-4"
        )
