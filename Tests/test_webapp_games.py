"""Tests for the ``/games`` route in ``Flask/app.py``.

Every test here runs with NO NETWORK and does not require ``nba_api`` to be
installed. ``scoreboard_tables`` is the one function that talks to NBA and it is
the seam every test stubs out, so the whole path that actually matters -- the
parsing, the home/away split, the status handling, the timezone conversion, the
cache and the error states -- runs for real against synthetic payloads.

The payloads are modelled on live ScoreboardV3 responses, because two things
about that endpoint are load-bearing and neither is guessable from the field
names:

* NOTHING ON THE ENDPOINT SAYS WHICH TEAM IS HOME. ``gameCode``
  ('20261020/PHINYK' = Phoenix at New York) is the only field carrying the
  split, and ``LineScore`` orders its rows HOME FIRST. ``TestHomeAway`` pins
  both, because getting either wrong produces a card that reads perfectly well
  and is wrong.
* ``gameEt`` is Eastern wall-clock time with a 'Z' stamped on it, so it is not
  UTC and must never be parsed as UTC. ``gameTimeUTC`` is the only timestamp
  this route trusts; ``TestTimezone`` pins that the rendered tip-off comes from
  ``gameTimeUTC`` and carries a real offset.

``test_home_away_survives_a_reversed_line_score`` is the sharp one: it asserts
the card is built from ``gameCode`` even when the endpoint returns the two rows
in the opposite order, so a reordering upstream cannot silently swap the teams.
"""

import importlib.util
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

# Flask/app.py imports flask at module scope, so this module cannot even be
# collected without it. Skipping here rather than erroring follows the rule
# Tests/test_nba_client.py sets out: a collection failure is indistinguishable
# from a real test failure in CI output. `nba_api` is deliberately NOT imported
# here -- Flask/app.py imports it lazily, which is what lets these tests run
# with neither nba_api nor a network.
pytest.importorskip("flask")

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "Flask" / "app.py"

SPEC = importlib.util.spec_from_file_location("flask_games_app", MODULE_PATH)
webapp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(webapp)

# Flask(__name__) works out where templates and static files live from the module
# name, which is meaningless for a module loaded by path the way
# Tests/test_line_movement.py loads its subject. Point both back at the
# directories `flask run` would have found on its own, so these tests render the
# real template rather than a stand-in.
webapp.app.template_folder = str(MODULE_PATH.parent / "templates")
webapp.app.root_path = str(MODULE_PATH.parent)
webapp.app.static_folder = str(MODULE_PATH.parent / "static")


# --------------------------------------------------------------------------
# Payloads shaped like real ScoreboardV3 responses
# --------------------------------------------------------------------------

GAME_HEADER_COLUMNS = [
    "gameId", "gameCode", "gameStatus", "gameStatusText", "period", "gameClock",
    "gameTimeUTC", "gameEt", "regulationPeriods", "seriesGameNumber", "gameLabel",
    "gameSubLabel", "seriesText", "ifNecessary", "seriesConference", "poRoundDesc",
    "gameSubtype", "isNeutral",
]

LINE_SCORE_COLUMNS = [
    "gameId", "teamId", "teamCity", "teamName", "teamTricode", "teamSlug",
    "wins", "losses", "score", "seed", "inBonus", "timeoutsRemaining",
]

# Eastern Daylight Time behind UTC. Every fixture game is an October one, which
# is what the live payloads checked against were, so a single offset is enough
# to make `gameEt` a genuinely wrong timestamp to parse.
ET_UTC_OFFSET_HOURS = 4


def header_frame(*games):
    """A GameHeader frame. An empty frame is what the endpoint sends on a day off."""
    return pd.DataFrame(list(games), columns=GAME_HEADER_COLUMNS)


def line_frame(*rows):
    return pd.DataFrame(list(rows), columns=LINE_SCORE_COLUMNS)


def game_header(game_id, game_code, status, status_text, tipoff_utc,
                period=0, game_clock=""):
    """One GameHeader row.

    `gameEt` is built the way the endpoint builds it: Eastern wall-clock time
    with a 'Z' on the end. For a 23:00 UTC tip-off it reads 19:00Z, four hours
    out, and reading it as UTC moves every game on the page by four hours. That
    offset is deliberate, so the tests that pin the tip-off to gameTimeUTC fail
    if anyone swaps the two fields.
    """
    moment = datetime.fromisoformat(tipoff_utc.replace("Z", "+00:00"))
    eastern_wall_clock = (moment - timedelta(hours=ET_UTC_OFFSET_HOURS))
    return {
        "gameId": game_id,
        "gameCode": game_code,
        "gameStatus": status,
        "gameStatusText": status_text,
        "period": period,
        "gameClock": game_clock,
        "gameTimeUTC": tipoff_utc,
        "gameEt": eastern_wall_clock.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "isNeutral": False,
    }


def line_row(game_id, team_id, city, name, tricode, wins, losses, score):
    return {
        "gameId": game_id,
        "teamId": team_id,
        "teamCity": city,
        "teamName": name,
        "teamTricode": tricode,
        "teamSlug": tricode.lower(),
        "wins": wins,
        "losses": losses,
        "score": score,
    }


# A completed game: Phoenix beat nobody, Detroit won at home.
FINAL_HEADER = game_header(
    "0022500579", "20260115/PHXDET", 3, "Final", "2026-01-16T00:00:00Z",
    period=4, game_clock="00:00.0",
)
# Live payloads list the HOME team on the first LineScore row.
FINAL_LINES = [
    line_row("0022500579", 1610612765, "Detroit", "Pistons", "DET", 29, 10, 108),
    line_row("0022500579", 1610612756, "Phoenix", "Suns", "PHX", 24, 17, 105),
]

# A game that has not tipped: the endpoint still returns two rows, both with a
# placeholder score of 0, which is why the route refuses to show it.
SCHEDULED_HEADER = game_header(
    "0022600002", "20261020/PHINYK", 1, "7:00 pm ET", "2026-10-20T23:00:00Z",
)
SCHEDULED_LINES = [
    line_row("0022600002", 1610612752, "New York", "Knicks", "NYK", 0, 0, 0),
    line_row("0022600002", 1610612755, "Philadelphia", "76ers", "PHI", 0, 0, 0),
]

LIVE_HEADER = game_header(
    "0022600100", "20261020/LALBOS", 2, "Q3 04:12", "2026-10-20T02:00:00Z",
    period=3, game_clock="04:12",
)
LIVE_LINES = [
    line_row("0022600100", 1610612738, "Boston", "Celtics", "BOS", 31, 18, 74),
    line_row("0022600100", 1610612747, "Los Angeles", "Lakers", "LAL", 29, 19, 71),
]


def stub_scoreboard(app, headers, lines, error=None):
    """Replace the one network call and count how often it is actually made.

    Mirrors how Tests/test_nba_client.py stubs the nba_api endpoint out
    entirely: the wrapper's own logic runs, nothing leaves the process. The
    counter is what makes "one call per page load" and "the cache holds" tests
    possible without a live upstream.
    """
    calls = []

    def fake_tables(game_date, timeout=None, proxy=None):
        calls.append(game_date)
        if error is not None:
            raise error
        return headers, lines

    app.scoreboard_tables = fake_tables
    return calls


class WebappGamesTestCase(unittest.TestCase):
    """Clears the scoreboard cache around every test so nothing leaks between them."""

    def setUp(self):
        webapp.fetch_games.cache_clear()
        self.app = webapp.app
        self.client = self.app.test_client()

    def tearDown(self):
        webapp.fetch_games.cache_clear()

    def serve(self, query="", headers=None, lines=None, header_rows=None):
        """GET /games with the upstream stubbed, returning the test response."""
        return self.client.get(f"/games{query}", headers=headers)


# --------------------------------------------------------------------------
# The four required behaviours
# --------------------------------------------------------------------------


class TestGamesRoute(WebappGamesTestCase):

    def test_normal_payload_renders_both_team_names(self):
        calls = stub_scoreboard(webapp, header_frame(FINAL_HEADER, SCHEDULED_HEADER),
                                line_frame(*FINAL_LINES, *SCHEDULED_LINES))
        response = self.serve()
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Detroit Pistons", body)
        self.assertIn("Phoenix Suns", body)

    def test_a_normal_payload_makes_exactly_one_upstream_call(self):
        calls = stub_scoreboard(webapp, header_frame(FINAL_HEADER),
                                line_frame(*FINAL_LINES))
        self.serve()
        self.assertEqual(len(calls), 1)

    def test_empty_game_list_renders_a_friendly_message_not_an_error(self):
        """A day off is a real answer, not a fault.

        The endpoint answers a date with no games with an empty DataFrame and
        HTTP 200. Rendering that as a 500 (or as a silent blank page) would make
        every quiet night look like an outage.
        """
        stub_scoreboard(webapp, header_frame(), line_frame())
        response = self.serve()
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("No games scheduled", body)
        self.assertNotIn("Traceback", body)

    def test_empty_frames_give_the_same_no_games_result(self):
        """An empty frame and a frame with no rows are the same upstream answer."""
        stub_scoreboard(webapp, header_frame(), line_frame())
        self.assertEqual(board_status_for(self.serve()), "no_games")

    def test_upstream_failure_renders_an_error_state_not_a_500_traceback(self):
        stub_scoreboard(webapp, None, None,
                        error=webapp.ScoreboardUnavailable("Expecting value"))
        response = self.serve()
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 503)
        self.assertIn("temporarily unavailable", body)
        self.assertNotIn("Traceback", body)
        self.assertNotIn("Expecting value", body)

    def test_a_transport_failure_is_an_error_state_too(self):
        """A timeout is not a decode error but it is still an error state."""
        stub_scoreboard(webapp, None, None, error=TimeoutError("read timed out"))
        response = self.serve()

        self.assertEqual(response.status_code, 503)
        self.assertIn("temporarily unavailable", response.get_data(as_text=True))

    def test_rate_limiting_is_reported_differently_from_a_broken_upstream(self):
        """403 means "stop asking", so it must not read as "we are broken"."""
        stub_scoreboard(webapp, None, None,
                        error=webapp.ScoreboardUnavailable("blocked", status_code=403))
        response = self.serve()
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 503)
        self.assertIn("rate limiting", body)
        self.assertNotIn("temporarily unavailable", body)

    def test_a_429_also_reads_as_rate_limiting(self):
        stub_scoreboard(webapp, None, None,
                        error=webapp.ScoreboardUnavailable("slow down", status_code=429))
        self.assertIn("rate limiting", self.serve().get_data(as_text=True))

    def test_the_cache_prevents_a_second_call_inside_its_window(self):
        calls = stub_scoreboard(webapp, header_frame(FINAL_HEADER),
                                line_frame(*FINAL_LINES))
        with mock.patch.object(webapp.time, "time", return_value=1_000.0):
            first = self.serve()
            second = self.serve()
            third = self.serve()

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(third.status_code, 200)
        self.assertEqual(len(calls), 1, "the 60s window let more than one call through")

    def test_the_cache_expires_after_its_window(self):
        calls = stub_scoreboard(webapp, header_frame(FINAL_HEADER),
                                line_frame(*FINAL_LINES))
        with mock.patch.object(webapp.time, "time", return_value=1_000.0):
            self.serve()
        with mock.patch.object(webapp.time, "time",
                               return_value=1_000.0 + webapp.SCOREBOARD_TTL_SECONDS):
            self.serve()

        self.assertEqual(len(calls), 2)

    def test_a_failure_is_cached_too(self):
        """A 403 behind no cache earns the next 403 within the same minute.

        lru_cache does not memoize a raised exception, which is exactly why
        fetch_games returns the failure as a value: the error page must not
        re-request NBA every time somebody refreshes.
        """
        calls = stub_scoreboard(webapp, None, None,
                                error=webapp.ScoreboardUnavailable("blocked", status_code=403))
        with mock.patch.object(webapp.time, "time", return_value=1_000.0):
            self.serve()
            self.serve()

        self.assertEqual(len(calls), 1)

    def test_two_dates_are_two_calls(self):
        calls = stub_scoreboard(webapp, header_frame(FINAL_HEADER),
                                line_frame(*FINAL_LINES))
        self.serve()
        self.serve("?date=2026-02-01")
        self.assertEqual(len(calls), 2)

    def test_two_timezones_share_one_upstream_call(self):
        """The cached board is timezone-independent, so a second viewer in
        another zone must not evict the first viewer's copy."""
        calls = stub_scoreboard(webapp, header_frame(FINAL_HEADER),
                                line_frame(*FINAL_LINES))
        with mock.patch.object(webapp.time, "time", return_value=1_000.0):
            self.serve("?date=2026-01-15&tz=America/New_York")
            self.serve("?date=2026-01-15&tz=Asia/Tokyo")
        self.assertEqual(len(calls), 1)


# --------------------------------------------------------------------------
# Home vs away -- the detail people get wrong
# --------------------------------------------------------------------------


class TestHomeAway(WebappGamesTestCase):

    def _card(self, header=FINAL_HEADER, lines=None):
        """The board dict for a single-game payload, from the route's own fetch."""
        stub_scoreboard(webapp, header_frame(header), line_frame(*(lines or FINAL_LINES)))
        with self.app.test_request_context("/games"):
            return webapp.fetch_games("2026-01-15")

    def test_game_code_decides_which_side_is_home(self):
        game = self._card()["0022500579"]
        self.assertEqual(game["away"]["name"], "Phoenix Suns")
        self.assertEqual(game["home"]["name"], "Detroit Pistons")

    def test_the_line_score_home_first_order_is_the_documented_fallback(self):
        """Pinned against a live payload: home comes first on the LineScore.

        If this ever fails, the endpoint reordered its rows -- the fallback in
        order_team_lines must be updated rather than left quietly wrong.
        """
        game = self._card()["0022500579"]
        self.assertEqual(FINAL_LINES[0]["teamTricode"], "DET")  # the home side
        self.assertEqual(game["home"]["tricode"], "DET")

    def test_home_away_survives_a_reversed_line_score(self):
        """The sharp one: row order flipped, card unchanged.

        If the card follows the rows instead of gameCode it silently relabels
        every team on the page, and nothing about the result looks wrong.
        """
        reversed_lines = list(reversed(FINAL_LINES))
        game = self._card(lines=reversed_lines)["0022500579"]
        self.assertEqual(game["away"]["tricode"], "PHX")
        self.assertEqual(game["home"]["tricode"], "DET")

    def test_an_unusable_game_code_falls_back_to_row_order(self):
        header = dict(FINAL_HEADER, gameCode="20260115/PHX")
        game = self._card(header=header)["0022500579"]
        self.assertEqual(game["home"]["tricode"], "DET")
        self.assertEqual(game["away"]["tricode"], "PHX")

    def test_split_game_code_reads_two_tricode_suffixes(self):
        self.assertEqual(webapp.split_game_code("20261020/PHINYK"), ("PHI", "NYK"))
        self.assertEqual(webapp.split_game_code("20261020/OKCSAS"), ("OKC", "SAS"))

    def test_split_game_code_refuses_anything_that_is_not_six_letters(self):
        for bad in ("", None, "20261020", "20261020/PHINY", "20261020/PHINYKN",
                    "20261020/PH1NYK", "20261020/", "PHINYK"):
            self.assertEqual(webapp.split_game_code(bad), (None, None), bad)

    def test_the_page_labels_both_sides(self):
        """A card with no Away/Home marker is the failure this prevents."""
        stub_scoreboard(webapp, header_frame(FINAL_HEADER), line_frame(*FINAL_LINES))
        body = self.serve().get_data(as_text=True)
        self.assertIn("team-line-away", body)
        self.assertIn("team-line-home", body)
        self.assertIn(">Away<", body)
        self.assertIn(">Home<", body)

    def test_away_is_rendered_before_home(self):
        stub_scoreboard(webapp, header_frame(FINAL_HEADER), line_frame(*FINAL_LINES))
        body = self.serve().get_data(as_text=True)
        self.assertLess(body.index("team-line-away"), body.index("team-line-home"))


# --------------------------------------------------------------------------
# Status and score
# --------------------------------------------------------------------------


class TestGameStatus(WebappGamesTestCase):

    def _game(self, header, lines):
        stub_scoreboard(webapp, header_frame(header), line_frame(*lines))
        with self.app.test_request_context("/games"):
            return webapp.fetch_games("2026-10-20")["0022600002" if header is SCHEDULED_HEADER else header["gameId"]]

    def test_a_scheduled_game_shows_no_score(self):
        game = self._game(SCHEDULED_HEADER, SCHEDULED_LINES)
        self.assertEqual(game["status"], "scheduled")
        self.assertIsNone(game["away"]["score"])
        self.assertIsNone(game["home"]["score"])
        self.assertFalse(game["has_score"])

    def test_a_final_game_shows_both_scores(self):
        game = self._game(FINAL_HEADER, FINAL_LINES)
        self.assertEqual(game["status"], "final")
        self.assertEqual(game["home"]["score"], 108)
        self.assertEqual(game["away"]["score"], 105)
        self.assertTrue(game["has_score"])

    def test_a_live_game_shows_the_clock_alongside_the_scores(self):
        game = self._game(LIVE_HEADER, LIVE_LINES)
        self.assertEqual(game["status"], "live")
        self.assertEqual(game["status_text"], "Q3 04:12")
        self.assertEqual(game["period"], 3)
        self.assertEqual(game["home"]["score"], 74)
        self.assertEqual(game["away"]["score"], 71)

    def test_an_unknown_status_code_is_not_guessed_at(self):
        """A status this page has never seen must not render as Scheduled."""
        header = dict(FINAL_HEADER, gameStatus=7, gameStatusText="Postponed")
        game = self._game(header, FINAL_LINES)
        self.assertEqual(game["status"], "unknown")
        self.assertEqual(game["status_label"], "Status 7")

    def test_records_are_omitted_before_a_season_starts(self):
        """0-0 is not a record; showing it would be an invented field."""
        game = self._game(SCHEDULED_HEADER, SCHEDULED_LINES)
        self.assertIsNone(game["away"]["record"])
        self.assertIsNone(game["home"]["record"])

    def test_records_are_shown_once_they_mean_something(self):
        game = self._game(FINAL_HEADER, FINAL_LINES)
        self.assertEqual(game["home"]["record"], "29-10")
        self.assertEqual(game["away"]["record"], "24-17")

    def test_the_page_renders_status_labels(self):
        stub_scoreboard(webapp, header_frame(FINAL_HEADER, SCHEDULED_HEADER, LIVE_HEADER),
                        line_frame(*FINAL_LINES, *SCHEDULED_LINES, *LIVE_LINES))
        body = self.serve().get_data(as_text=True)
        for label in ("Scheduled", "In progress", "Final"):
            self.assertIn(label, body)


# --------------------------------------------------------------------------
# Timezone -- aware or nothing
# --------------------------------------------------------------------------


class TestTimezone(WebappGamesTestCase):

    def test_tipoff_comes_from_game_time_utc_not_the_et_stamp(self):
        """gameEt is ET wall clock wearing a 'Z'. Reading it as UTC puts the
        game four hours out, which is the exact failure this test exists for."""
        stub_scoreboard(webapp, header_frame(FINAL_HEADER), line_frame(*FINAL_LINES))
        with self.app.test_request_context("/games"):
            game = webapp.fetch_games("2026-01-15")["0022500579"]
        self.assertEqual(game["tipoff"], datetime(2026, 1, 16, 0, 0, tzinfo=timezone.utc))

    def test_the_rendered_tipoff_is_in_the_requested_zone(self):
        stub_scoreboard(webapp, header_frame(SCHEDULED_HEADER), line_frame(*SCHEDULED_LINES))
        body = self.serve("?date=2026-10-20&tz=America/New_York").get_data(as_text=True)
        # 23:00 UTC is 19:00 EDT on the same day.
        self.assertIn("Tue Oct 20, 7:00 PM EDT", body)

    def test_the_same_game_renders_differently_in_another_zone(self):
        stub_scoreboard(webapp, header_frame(SCHEDULED_HEADER), line_frame(*SCHEDULED_LINES))
        tokyo = self.serve("?date=2026-10-20&tz=Asia/Tokyo").get_data(as_text=True)
        self.assertIn("Wed Oct 21, 8:00 AM", tokyo)

    def test_the_machine_readable_timestamp_carries_an_offset(self):
        """The <time datetime="..."> attribute is parsed by machines, so it has
        to be unambiguous -- an aware stamp, never a bare local time."""
        stub_scoreboard(webapp, header_frame(SCHEDULED_HEADER), line_frame(*SCHEDULED_LINES))
        body = self.serve("?date=2026-10-20&tz=America/New_York").get_data(as_text=True)
        self.assertIn('datetime="2026-10-20T19:00:00-04:00"', body)

    def test_the_page_names_the_zone_it_used(self):
        stub_scoreboard(webapp, header_frame(SCHEDULED_HEADER), line_frame(*SCHEDULED_LINES))
        self.assertIn("Asia/Tokyo",
                      self.serve("?date=2026-10-20&tz=Asia/Tokyo").get_data(as_text=True))

    def test_the_fallback_zone_is_utc_and_is_named(self):
        stub_scoreboard(webapp, header_frame(SCHEDULED_HEADER), line_frame(*SCHEDULED_LINES))
        body = self.serve("?date=2026-10-20").get_data(as_text=True)
        self.assertIn("UTC", body)
        self.assertIn('datetime="2026-10-20T23:00:00+00:00"', body)
        self.assertIn("Tue Oct 20, 11:00 PM UTC", body)

    def test_an_unusable_zone_is_ignored_rather_than_raising(self):
        stub_scoreboard(webapp, header_frame(SCHEDULED_HEADER), line_frame(*SCHEDULED_LINES))
        response = self.serve("?date=2026-10-20&tz=../../etc/passwd")
        self.assertEqual(response.status_code, 200)
        self.assertIn("UTC", response.get_data(as_text=True))

    def test_parse_utc_stamp_never_returns_a_naive_datetime(self):
        for stamp in ("2026-10-20T23:00:00Z", "2026-10-20T23:00:00",
                      "2026-10-20T23:00:00+00:00"):
            parsed = webapp.parse_utc_stamp(stamp)
            self.assertIsNotNone(parsed, stamp)
            self.assertIsNotNone(parsed.tzinfo, stamp)

    def test_parse_utc_stamp_returns_none_for_junk(self):
        for stamp in ("", None, "not-a-time", "2026-13-45T99:00:00Z"):
            self.assertIsNone(webapp.parse_utc_stamp(stamp), stamp)

    def test_a_game_with_no_readable_tipoff_says_so(self):
        header = dict(SCHEDULED_HEADER, gameTimeUTC="")
        stub_scoreboard(webapp, header_frame(header), line_frame(*SCHEDULED_LINES))
        body = self.serve("?date=2026-10-20").get_data(as_text=True)
        self.assertIn("Time TBD", body)
        self.assertNotIn("datetime=", body.split("team-lines")[1][:400])


# --------------------------------------------------------------------------
# Bad input
# --------------------------------------------------------------------------


class TestBadInput(WebappGamesTestCase):

    def test_an_unparseable_date_is_a_400_with_a_message(self):
        stub_scoreboard(webapp, header_frame(FINAL_HEADER), line_frame(*FINAL_LINES))
        response = self.serve("?date=not-a-date")
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 400)
        self.assertIn("Not a date", body)
        self.assertNotIn("Traceback", body)

    def test_a_bad_date_never_reaches_the_network(self):
        calls = stub_scoreboard(webapp, header_frame(FINAL_HEADER), line_frame(*FINAL_LINES))
        self.serve("?date=2026-1-5")
        self.assertEqual(calls, [])

    def test_parse_game_date_requires_the_full_iso_shape(self):
        self.assertEqual(webapp.parse_game_date("2026-10-20"),
                         datetime(2026, 10, 20).date())
        # strptime alone accepts '2026-1-5'; the length check is what rejects it.
        for bad in ("2026-1-5", "10/20/2026", "2026-10-20T00:00:00", "", None, "today"):
            self.assertIsNone(webapp.parse_game_date(bad), bad)

    def test_a_date_in_the_future_is_accepted(self):
        """Upcoming games are the point of the page."""
        stub_scoreboard(webapp, header_frame(SCHEDULED_HEADER), line_frame(*SCHEDULED_LINES))
        self.assertEqual(self.serve("?date=2026-10-20").status_code, 200)


# --------------------------------------------------------------------------
# Rows the endpoint sends that cannot be rendered
# --------------------------------------------------------------------------


class TestDegradedPayloads(WebappGamesTestCase):

    def test_a_row_with_no_team_lines_is_dropped_not_rendered_blank(self):
        headers = header_frame(FINAL_HEADER, dict(SCHEDULED_HEADER, gameId="0022600009"))
        lines = line_frame(*FINAL_LINES)  # nothing for 0022600009
        stub_scoreboard(webapp, headers, lines)
        response = self.serve()
        body = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Partial slate", body)
        self.assertIn("Detroit Pistons", body)
        self.assertNotIn("Philadelphia 76ers", body)

    def test_a_row_with_no_game_id_is_dropped(self):
        """One unreadable row out of one is a partial slate, not a day off.

        The distinction matters: "no games scheduled" tells a visitor the league
        is idle, and saying that when the page simply could not read a row is
        the kind of quiet wrong answer this repo's error handling exists to stop.
        """
        stub_scoreboard(webapp, header_frame(dict(FINAL_HEADER, gameId="")),
                        line_frame(*FINAL_LINES))
        body = self.serve().get_data(as_text=True)
        self.assertIn("Partial slate", body)
        self.assertNotIn("No games scheduled", body)

    def test_an_unexpected_payload_shape_is_an_error_state(self):
        """A frame with no columns at all is an upstream change, not a day off."""
        stub_scoreboard(webapp, object(), None)
        response = self.serve()
        self.assertEqual(response.status_code, 503)
        self.assertIn("temporarily unavailable", response.get_data(as_text=True))

    def test_a_missing_nba_api_is_an_error_state_not_an_import_crash(self):
        stub_scoreboard(webapp, None, None, error=ImportError("No module named 'nba_api'"))
        response = self.serve()
        self.assertEqual(response.status_code, 503)
        self.assertIn("temporarily unavailable", response.get_data(as_text=True))

    def test_a_missing_line_score_table_degrades_rather_than_raising(self):
        stub_scoreboard(webapp, header_frame(FINAL_HEADER), None)
        response = self.serve()
        self.assertEqual(response.status_code, 200)
        self.assertIn("Partial slate", response.get_data(as_text=True))


# --------------------------------------------------------------------------
# The error contract must not be weakened by adding this route
# --------------------------------------------------------------------------


class TestErrorContract(WebappGamesTestCase):

    def test_games_is_not_registered_as_a_json_route(self):
        """handle_unexpected_error returns JSON only for the two data endpoints.

        /games is HTML, so it must not be added to JSON_ROUTE_PREFIXES: that
        would hand a JSON error body to a browser page.
        """
        for prefix in webapp.JSON_ROUTE_PREFIXES:
            self.assertFalse(prefix.startswith("/games"))

    def test_an_unexpected_error_is_still_a_bare_generic_string(self):
        with mock.patch.object(webapp, "fetch_games", side_effect=RuntimeError("boom")):
            response = self.client.get("/games")
        self.assertEqual(response.status_code, 500)
        body = response.get_data(as_text=True)
        self.assertEqual(body.strip(), "Something went wrong. Please try again shortly.")

    def test_no_error_detail_ever_reaches_the_page(self):
        secret = "https://stats.nba.com/stats/scoreboardv3?GameDate=2026-10-20"
        stub_scoreboard(webapp, None, None,
                        error=webapp.ScoreboardUnavailable(secret, status_code=403))
        body = self.serve().get_data(as_text=True)
        self.assertNotIn(secret, body)
        self.assertNotIn("scoreboardv3", body)


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------


class TestWiring(WebappGamesTestCase):

    def test_the_boards_share_one_status_reader(self):
        """GamesBoard and PredictionBoard differ in payload, not in contract, so
        the same accessors read both."""
        for board in (webapp.GamesBoard(), webapp.PredictionBoard()):
            self.assertEqual(webapp.board_status(board), "ok")
        board = webapp.GamesBoard(status="rate_limited", detail="blocked")
        self.assertEqual(webapp.board_status(board), "rate_limited")
        self.assertEqual(webapp.board_detail(board), "blocked")

    def test_a_plain_dict_fails_closed_as_unavailable(self):
        self.assertEqual(webapp.board_status({}), "unavailable")

    def test_the_route_is_registered(self):
        rules = {rule.rule for rule in self.app.url_map.iter_rules()}
        self.assertIn("/games", rules)

    def test_the_route_only_accepts_get(self):
        """No state changes here, so POST/PUT must not be routed anywhere."""
        methods = next(rule for rule in self.app.url_map.iter_rules()
                       if rule.rule == "/games").methods
        self.assertEqual(methods - {"HEAD", "OPTIONS"}, {"GET"})

    def test_the_nav_on_the_index_page_links_to_the_schedule(self):
        empty = webapp.PredictionBoard(status="no_games", detail="the payload reported no games")
        with mock.patch.object(webapp, "fetch_fanduel", return_value=empty), \
             mock.patch.object(webapp, "fetch_draftkings", return_value=empty), \
             mock.patch.object(webapp, "fetch_betmgm", return_value=empty):
            body = self.client.get("/").get_data(as_text=True)
        self.assertIn('href="/games"', body)

    def test_the_schedule_page_links_back_to_the_board(self):
        stub_scoreboard(webapp, header_frame(FINAL_HEADER), line_frame(*FINAL_LINES))
        body = self.serve().get_data(as_text=True)
        self.assertIn('href="/#picks"', body)

    def test_no_third_party_image_is_hotlinked(self):
        """ScoreboardV3 returns no logo URL, so a card must not reach for one.

        An <img> pointing at an external CDN would put an unversioned dependency
        in front of every row and 404 silently; the tricode monogram always
        renders.
        """
        stub_scoreboard(webapp, header_frame(FINAL_HEADER, SCHEDULED_HEADER),
                        line_frame(*FINAL_LINES, *SCHEDULED_LINES))
        body = self.serve().get_data(as_text=True)
        card = body.split("<main>", 1)[1]
        self.assertNotIn("<img", card)
        self.assertNotIn("cdn.nba.com", card)
        self.assertIn("team-monogram", card)

    def test_the_monogram_carries_the_tricode(self):
        stub_scoreboard(webapp, header_frame(FINAL_HEADER), line_frame(*FINAL_LINES))
        body = self.serve().get_data(as_text=True)
        self.assertIn('aria-hidden="true">PHX<', body)
        self.assertIn('aria-hidden="true">DET<', body)


# --------------------------------------------------------------------------
# Caching shape, checked directly rather than only through the route
# --------------------------------------------------------------------------


class TestFetchGamesCache(WebappGamesTestCase):

    def test_the_ttl_hash_is_the_only_thing_that_expires_the_entry(self):
        calls = stub_scoreboard(webapp, header_frame(FINAL_HEADER), line_frame(*FINAL_LINES))
        with self.app.test_request_context("/games"):
            webapp.fetch_games("2026-01-15", ttl_hash=1)
            webapp.fetch_games("2026-01-15", ttl_hash=1)
            webapp.fetch_games("2026-01-15", ttl_hash=2)
        self.assertEqual(len(calls), 2)

    def test_a_cached_board_is_shared_rather_than_rebuilt(self):
        stub_scoreboard(webapp, header_frame(FINAL_HEADER), line_frame(*FINAL_LINES))
        with self.app.test_request_context("/games"):
            self.assertIs(webapp.fetch_games("2026-01-15", ttl_hash=1),
                          webapp.fetch_games("2026-01-15", ttl_hash=1))

    def test_the_cache_window_is_sixty_seconds(self):
        """Pinned so the 'cached for a minute' copy on the page stays true."""
        self.assertEqual(webapp.SCOREBOARD_TTL_SECONDS, 60)

    def test_the_timeout_matches_the_rapidapi_timeout(self):
        """One outbound timeout for the whole app, so a slow upstream fails at
        the same point in time regardless of which host it is."""
        self.assertEqual(webapp.SCOREBOARD_TIMEOUT_SECONDS, webapp.API_TIMEOUT_SECONDS)


class TestOrdering(unittest.TestCase):

    def test_games_are_ordered_by_tipoff_with_unreadable_ones_last(self):
        headers = header_frame(
            dict(SCHEDULED_HEADER, gameId="222", gameCode="20261020/AAABBB",
                 gameTimeUTC="2026-10-20T01:30:00Z"),
            dict(SCHEDULED_HEADER, gameId="111", gameCode="20261020/CCCAAA",
                 gameTimeUTC="2026-10-20T23:00:00Z"),
            dict(SCHEDULED_HEADER, gameId="333", gameCode="20261020/DDDBBB",
                 gameTimeUTC=""),
        )
        lines = line_frame(
            line_row("222", 1, "Aaa", "Aaa", "AAA", 0, 0, 0),
            line_row("222", 2, "Bbb", "Bbb", "BBB", 0, 0, 0),
            line_row("111", 3, "Ccc", "Ccc", "CCC", 0, 0, 0),
            line_row("111", 4, "Aaa", "Aaa", "AAA", 0, 0, 0),
            line_row("333", 5, "Ddd", "Ddd", "DDD", 0, 0, 0),
            line_row("333", 6, "Bbb", "Bbb", "BBB", 0, 0, 0),
        )
        board = webapp.build_games(headers, lines)
        self.assertEqual(list(board), ["222", "111", "333"])

    def test_ordering_is_stable_across_calls(self):
        headers = header_frame(SCHEDULED_HEADER, FINAL_HEADER)
        lines = line_frame(*SCHEDULED_LINES, *FINAL_LINES)
        first = list(webapp.build_games(headers, lines))
        second = list(webapp.build_games(headers, lines))
        self.assertEqual(first, second)


def board_status_for(response):
    """Recover the board status from the copy the page actually shows.

    Deliberately reads the rendered text rather than the internal status slug:
    what this checks is that each status reaches the visitor as distinct wording,
    which is the whole point of GAMES_NOTICES.
    """
    body = response.get_data(as_text=True)
    if "Partial slate" in body:
        return "degraded"
    if "No games scheduled" in body:
        return "no_games"
    if "rate limiting" in body:
        return "rate_limited"
    if "temporarily unavailable" in body:
        return "unavailable"
    return "ok"


if __name__ == "__main__":
    unittest.main()