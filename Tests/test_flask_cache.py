"""Tests for the shared per-key TTL cache behind every route in ``Flask/app.py``.

This is the issue #35 fix, and the bug it removes is worth restating because the
arithmetic looks harmless until you place two requests against it.

The cache used to be keyed on a BUCKET of time::

    def get_ttl_hash(seconds=600):
        return round(time.time() / seconds)

``round`` puts a bucket boundary halfway through each window, so the window a
caller actually got was "whatever was left of the current minute": a request
landing a millisecond before a boundary refreshed at once, and the unlucky
client could be handed two upstream calls 0.2 seconds apart. ``floor`` -- the
obvious repair -- removes that but guarantees the other half of the same problem,
a nearly-full window for whoever drew the short straw. Neither is right, because
the error is in bucketing a clock rather than in rounding it.

So the key is no longer a bucket. ``TTLCache`` stores each value with the moment
it was stored and compares the window against THAT, per key. The property under
test is therefore exactly one: **for any key, two upstream calls can never land
closer together than the window the caller asked for** -- and the first call
after a window is always a new one.

Every test here runs with NO NETWORK and never sleeps. The clock is a
``FakeClock`` swapped into app.py's globals for the duration of a block, so a
60-second window is crossed in microseconds and deterministically; each route's
real fetcher is replaced with a counting stub, so what is under test is the
caching above it and nothing else. The routes themselves are exercised for real
at the bottom of this file -- all four, with their templates -- so a change to
the cache that quietly broke one would fail here rather than in production.
"""

import contextlib
import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

# Flask/app.py imports flask at module scope, so this module cannot even be
# collected without it. Same rule, same reason as Tests/test_webapp_games.py: a
# collection failure is indistinguishable from a real test failure in CI output.
pytest.importorskip("flask")

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "Flask" / "app.py"

SPEC = importlib.util.spec_from_file_location("flask_cache_app", MODULE_PATH)
webapp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(webapp)

# `flask run` finds these directories from a meaningful module name; a module
# loaded by path has none, so they are pointed back at Flask/ explicitly.
webapp.app.template_folder = str(MODULE_PATH.parent / "templates")
webapp.app.root_path = str(MODULE_PATH.parent)
webapp.app.static_folder = str(MODULE_PATH.parent / "static")


# --------------------------------------------------------------------------
# A clock the test moves by hand
# --------------------------------------------------------------------------


class FakeClock:
    """The only thing `TTLCache` reads from `time`.

    `advance` is how a window is crossed in a test. Nothing here sleeps, so the
    whole file runs in well under a second and every assertion is exact rather
    than timing-dependent.
    """

    def __init__(self, now=1_000.0):
        self.now = float(now)

    def time(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds
        return self.now


@contextlib.contextmanager
def frozen_clock(at=1_000.0):
    """Swap app.py's `time` for a `FakeClock`, and hand it to the block.

    The NAME is patched, not `time.time`, so nothing outside app.py -- logging,
    the test runner, the clock the assertions themselves use -- is affected.
    """
    clock = FakeClock(at)
    with mock.patch.object(webapp, "time", clock):
        yield clock


# Pairs of instants 0.2 seconds apart that sit either side of a step in the clock
# arithmetic the old cache used. BOTH are here on purpose: `round(t / 60)` steps
# at t = 30 (mod 60) and `floor(t / 60)` steps at t = 0 (mod 60), so a test that
# straddles only the first is a test the second passes.
STEPS_UNDER_TEST = ((1_469.9, 1_470.1), (1_499.9, 1_500.1))

# The first of those pairs, which is the one the offsets test works from.
BEFORE_BOUNDARY, AFTER_BOUNDARY = STEPS_UNDER_TEST[0]


# --------------------------------------------------------------------------
# Payloads, and the stubs that count what would have gone to the network
# --------------------------------------------------------------------------


GAME_HEADER = {
    "gameId": "0022500579",
    "gameCode": "20260115/PHXDET",
    "gameStatus": webapp.GAME_STATUS_SCHEDULED,
    "gameStatusText": "7:00 pm ET",
    "period": 0,
    "gameClock": "",
    "gameTimeUTC": "2026-01-15T23:00:00Z",
    "gameEt": "2026-01-15T18:00:00Z",
}

# LineScore rows come back HOME FIRST, which is what the gameCode suffix above
# decodes around; the card is built from the code, not from this order.
LINE_SCORES = [
    {"gameId": "0022500579", "teamId": 1, "teamCity": "Detroit", "teamName": "Pistons",
     "teamTricode": "DET", "wins": 30, "losses": 12, "score": 0},
    {"gameId": "0022500579", "teamId": 2, "teamCity": "Phoenix", "teamName": "Suns",
     "teamTricode": "PHX", "wins": 28, "losses": 14, "score": 0},
]

PREDICTIONS_ENTRY = {
    "home_team": "Detroit Pistons",
    "away_team": "Phoenix Suns",
    "away_confidence": 61.2,
    "home_confidence": 58.8,
    "ou_pick": "UNDER",
    "ou_value": 227.5,
    "ou_confidence": 55.0,
    "away_team_ev": 3.5,
    "home_team_ev": -1.5,
    "away_team_odds": -110,
    "home_team_odds": -110,
}


def scoreboard_stub(headers=None, lines=None):
    """A `scoreboard_tables` replacement that records the dates it was asked for."""
    calls = []

    def fake_tables(game_date, timeout=None, proxy=None):
        calls.append(game_date)
        return pd.DataFrame(headers if headers is not None else [GAME_HEADER]), \
            pd.DataFrame(lines if lines is not None else LINE_SCORES)

    return calls, fake_tables


def game_data_stub():
    """A `fetch_game_data` replacement that records each sportsbook it ran for."""
    calls = []

    def fake_game_data(sportsbook="fanduel"):
        calls.append(sportsbook)
        game = webapp.build_game(PREDICTIONS_ENTRY)
        return webapp.PredictionBoard({f"{game['away_team']}:{game['home_team']}": game})

    return calls, fake_game_data


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class FakeRequests:
    """Stands in for the `requests` module inside app.py's globals.

    Answered from a table keyed on the URL, so a test cannot make a real request
    even if a URL is wrong: an unknown one is a KeyError here rather than a
    connection to RapidAPI. The NAME is patched rather than `requests.get`, so
    the stub cannot leak into any other module in the process.
    """

    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    def get(self, url, headers=None, params=None, timeout=None):
        self.calls.append(url)
        return FakeResponse(self.payloads[url])


ROSTER_PAYLOAD = {
    "statusCode": 200,
    "body": {
        "roster": [
            {"longName": "Cade Cunningham", "shortName": "C. Cunningham",
             "pos": "G", "height": "6'6\"", "weight": 211, "playerID": "2544"},
        ]
    },
}

PLAYER_INFO_PAYLOAD = {
    "statusCode": 200,
    "body": {"longName": "Cade Cunningham", "position": "Point Guard",
             "college": "Oklahoma State", "injury": None},
}

PLAYER_GAMES_PAYLOAD = {
    "statusCode": 200,
    "body": {"20260115/DETPHX": {"gameID": "20260115/DETPHX", "points": 22}},
}


# --------------------------------------------------------------------------
# The window itself
# --------------------------------------------------------------------------


class TestCacheWindowIsPerFetch(unittest.TestCase):
    """One upstream call per window, measured from that window's own fetch."""

    def setUp(self):
        self.client = webapp.app.test_client()
        # The store is module-level and shared, exactly as it is in production, so
        # each case starts from an empty one rather than inheriting the previous
        # case's warm cache (which is what `Tests/test_webapp_games.py` does too).
        webapp.SCOREBOARD_CACHE.clear()
        self.addCleanup(webapp.SCOREBOARD_CACHE.clear)
        self.calls, fake_tables = scoreboard_stub()
        patcher = mock.patch.object(webapp, "scoreboard_tables", fake_tables)
        patcher.start()
        self.addCleanup(patcher.stop)

    def serve(self, query=""):
        return self.client.get(f"/games{query}")

    def test_two_calls_inside_one_window_are_one_upstream_call(self):
        """The headline: repeats inside the window cost nothing.

        Three page loads, one instant, one upstream call -- and then a fourth
        after the window has passed, which is the second. The instant is chosen
        to be unremarkable (1000.0 is not one of the old key's steps), so this is
        the plain reading of the window and the tests below are the sharp ones.
        """
        with frozen_clock() as clock:
            for _ in range(3):
                self.assertEqual(self.serve().status_code, 200)
            self.assertEqual(len(self.calls), 1)

            # Cross the window the way a client would: wait out the rest of it.
            clock.advance(webapp.SCOREBOARD_TTL_SECONDS + 1)
            self.assertEqual(self.serve().status_code, 200)
            self.assertEqual(len(self.calls), 2, "the window never expired")

    def test_a_call_after_the_window_is_a_second_upstream_call(self):
        with frozen_clock() as clock:
            self.serve()
            clock.advance(webapp.SCOREBOARD_TTL_SECONDS + 1)
            self.serve()

        self.assertEqual(len(self.calls), 2)

    def test_the_window_is_still_open_at_exactly_its_length(self):
        """60s after a fetch is inside the window it opened, not past it.

        Measured from the fetch, so the whole of `[stored, stored + seconds]` is a
        hit. Under the old bucket key this instant could fall in the next bucket
        and refresh instantly, which is the same defect as the test below seen
        from the other side.
        """
        with frozen_clock() as clock:
            self.serve()
            clock.advance(webapp.SCOREBOARD_TTL_SECONDS)
            self.serve()

        self.assertEqual(len(self.calls), 1)

    def test_two_calls_a_tenth_of_a_second_apart_are_one_upstream_call(self):
        """The regression this whole change exists for, checked against both
        arithmetic repairs of the bucketed key.

        Two GETs 0.2 seconds apart, straddling a step in the old key, are ONE
        upstream call -- whether that step belongs to `round` (t = 30 mod 60) or
        to `floor` (t = 0 mod 60). Under the code as it stood they were two cache
        entries, so one visitor reloading the page could be the second of two NBA
        calls, and ScoreboardV3 is the endpoint this repo documents as 403-ing
        under burst (issue #27).

        Neither repair is right on its own: `round` also hands the unlucky
        requester a nearly-empty window and `floor` a nearly-full one. Only
        storing the fetch time gives both callers the window they asked for.
        """
        for before, after in STEPS_UNDER_TEST:
            with self.subTest(before=before, after=after):
                webapp.SCOREBOARD_CACHE.clear()
                self.calls.clear()
                with frozen_clock(before):
                    self.serve()
                with frozen_clock(after):
                    self.serve()

                self.assertEqual(
                    len(self.calls), 1,
                    "two requests 0.2s apart were served by two upstream calls")

    def test_no_offset_inside_the_window_can_earn_a_second_call(self):
        """Every instant of the window is a hit, not just the ones near its start.

        The offsets are deliberately out of order, and the window deliberately
        spans one of the old key's steps, because the bug was never "the window is
        too short": it was that the window had edges in places it should not have
        had them.
        """
        offsets = (0.0, 0.2, 59.9, 12.5, 30.0, webapp.SCOREBOARD_TTL_SECONDS)
        with frozen_clock(BEFORE_BOUNDARY) as clock:
            self.serve()
            for offset in offsets:
                clock.now = BEFORE_BOUNDARY + offset
                self.serve()
            self.assertEqual(len(self.calls), 1, "a request inside the window refetched")

            clock.now = BEFORE_BOUNDARY + webapp.SCOREBOARD_TTL_SECONDS + 0.001
            self.serve()
        self.assertEqual(len(self.calls), 2)

    def test_the_guarantee_holds_from_every_starting_instant(self):
        """The property, swept: calls closer together than the window are impossible.

        For a spread of first-call instants -- either side of both places the old
        key changed its mind -- a second call is only ever let through once the
        window has genuinely elapsed. This is the statement the other tests
        illustrate.
        """
        seconds = webapp.SCOREBOARD_TTL_SECONDS
        # First-call instants either side of the steps the old key took, at both
        # of its step positions: 1470.0 for `round` and 1500.0 for `floor`.
        starts = [1_000.0] + [step + offset
                              for pair in STEPS_UNDER_TEST
                              for step in pair for offset in (-0.2, 0.2)]
        deltas = (0.0, 0.2, 5.0, 59.9, 60.0, 60.1, 90.0)

        for start in starts:
            for delta in deltas:
                with self.subTest(start=start, delta=delta):
                    calls = []
                    cache = webapp.TTLCache()
                    with frozen_clock(start):
                        cache.get("date", lambda: calls.append(start) or "board", seconds)
                    with frozen_clock(start + delta):
                        cache.get("date", lambda: calls.append(start + delta) or "board", seconds)
                    if len(calls) > 1:
                        self.assertGreater(
                            calls[1] - calls[0], seconds,
                            f"two fetches {calls[1] - calls[0]}s apart inside a {seconds}s window")

    def test_a_slow_fetch_does_not_eat_into_its_own_window(self):
        """The stamp is taken when the fetch RETURNS.

        A fetch that takes 10 of the window's 60 seconds must still be followed by
        a full 60, or a slow upstream could hand out two calls closer together
        than the caller asked for -- which is the guarantee this fix is making.
        """
        calls = []
        cache = webapp.TTLCache()
        clock = FakeClock(1_000.0)

        def slow_fetch():
            calls.append(clock.now)
            clock.advance(10.0)          # the upstream call, spent inside the window
            return "board"

        with mock.patch.object(webapp, "time", clock):
            cache.get("date", slow_fetch, 60)
            cache.get("date", lambda: calls.append(clock.now) or "board", 60)
            self.assertEqual(len(calls), 1, "the window was shortened by the fetch itself")

            clock.advance(60.0)           # 70s after the fetch started, 60s after it finished
            cache.get("date", lambda: calls.append(clock.now) or "board", 60)
            self.assertEqual(
                len(calls), 1,
                "the fetch's own 10 seconds were taken out of the window it opened")

            clock.advance(1.0)            # 61s after the fetch finished: the window has passed
            cache.get("date", lambda: calls.append(clock.now) or "board", 60)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1] - calls[0], 71.0)

    def test_a_fetch_that_raises_caches_nothing(self):
        """Failures are the caller's to handle, exactly as under `lru_cache`.

        This is why `fetch_games` returns a failed board as a value rather than
        raising: a raise would store nothing, and the 403 it just earned would be
        requested again on the next page load.
        """
        calls = []

        def raiser():
            calls.append(1)
            raise RuntimeError("blocked")

        cache = webapp.TTLCache()
        with frozen_clock():
            with self.assertRaises(RuntimeError):
                cache.get("date", raiser, 60)
            with self.assertRaises(RuntimeError):
                cache.get("date", raiser, 60)
        self.assertEqual(len(calls), 2)


# --------------------------------------------------------------------------
# Keys, and the fact that all three routes share one primitive
# --------------------------------------------------------------------------


class TestKeysAreIndependent(unittest.TestCase):
    def setUp(self):
        webapp.PREDICTION_CACHE.clear()
        webapp.SCOREBOARD_CACHE.clear()
        webapp.PLAYER_STATS_CACHE.clear()
        self.addCleanup(webapp.PREDICTION_CACHE.clear)
        self.addCleanup(webapp.SCOREBOARD_CACHE.clear)
        self.addCleanup(webapp.PLAYER_STATS_CACHE.clear)

    def test_every_route_uses_the_same_primitive_rather_than_its_own(self):
        """`/games` shares the cache with the other routes; nothing is forked.

        Three stores, one implementation, one `TTLCache` class. A route that grew
        a private cache -- or a second expiry rule -- is a failing test here, not
        a discrepancy nobody notices until one endpoint earns a 403.
        """
        for store in (webapp.PREDICTION_CACHE, webapp.SCOREBOARD_CACHE,
                      webapp.PLAYER_STATS_CACHE):
            self.assertIsInstance(store, webapp.TTLCache)

        stores = {id(webapp.PREDICTION_CACHE), id(webapp.SCOREBOARD_CACHE),
                  id(webapp.PLAYER_STATS_CACHE)}
        self.assertEqual(len(stores), 3, "two routes are sharing one store")

    def test_clearing_one_route_cache_leaves_the_others_alone(self):
        webapp.PREDICTION_CACHE.get("fanduel", lambda: "board", 600)
        webapp.SCOREBOARD_CACHE.get("2026-01-15", lambda: "board", 60)

        webapp.SCOREBOARD_CACHE.clear()

        self.assertEqual(len(webapp.SCOREBOARD_CACHE.entries), 0)
        self.assertEqual(len(webapp.PREDICTION_CACHE.entries), 1)

    def test_two_dates_are_two_windows(self):
        calls, fake_tables = scoreboard_stub()
        with mock.patch.object(webapp, "scoreboard_tables", fake_tables):
            webapp.fetch_games("2026-01-15")
            webapp.fetch_games("2026-02-01")
            webapp.fetch_games("2026-01-15")
            webapp.fetch_games("2026-02-01")
        self.assertEqual(sorted(calls), ["2026-01-15", "2026-02-01"])

    def test_three_sportsbooks_are_three_windows(self):
        calls, fake_game_data = game_data_stub()
        with mock.patch.object(webapp, "fetch_game_data", fake_game_data):
            for book in ("fanduel", "draftkings", "betmgm"):
                webapp.fetch_prediction(book)
            webapp.fetch_prediction("fanduel")
        self.assertEqual(sorted(calls), ["betmgm", "draftkings", "fanduel"])

    def test_two_players_are_two_windows(self):
        calls = []

        def fake_player_stats(player_id, season):
            calls.append(player_id)
            return {"success": True, "player": {"name": player_id}}

        with mock.patch.object(webapp, "fetch_player_stats", fake_player_stats):
            webapp.cached_player_stats("2544", "2024")
            webapp.cached_player_stats("2545", "2024")
            webapp.cached_player_stats("2544", "2024")
        self.assertEqual(sorted(calls), ["2544", "2545"])

    def test_the_store_is_still_bounded(self):
        """The eviction the bare `@lru_cache()` used to do is not lost with it.

        The bound is the whole of the protection against `/games?date=` walking a
        year of history (issue #35, item 2), so it is pinned here even though that
        item's auth and metering questions are not decided yet.
        """
        cache = webapp.TTLCache(maxsize=2)
        for key in ("a", "b", "c"):
            cache.get(key, lambda key=key: key, 60)
        self.assertEqual(list(cache.entries), ["b", "c"])


# --------------------------------------------------------------------------
# Every route the app serves, exercised for real with its own template
# --------------------------------------------------------------------------


class TestEveryRouteStillResolves(unittest.TestCase):
    """`/`, `/games`, `/team-data` and `/player-stats` all still serve.

    Nothing here is about the cache except the last test. The point is that the
    cache rewrite touched all three cached entry points, and a route that stopped
    resolving would be a silent failure until somebody loaded the page.
    """

    def setUp(self):
        webapp.PREDICTION_CACHE.clear()
        webapp.SCOREBOARD_CACHE.clear()
        webapp.PLAYER_STATS_CACHE.clear()
        self.addCleanup(webapp.PREDICTION_CACHE.clear)
        self.addCleanup(webapp.SCOREBOARD_CACHE.clear)
        self.addCleanup(webapp.PLAYER_STATS_CACHE.clear)
        self.client = webapp.app.test_client()

    def test_the_four_routes_are_still_registered(self):
        """Named here rather than discovered, so a lost route is a failing test."""
        rules = {rule.rule for rule in webapp.app.url_map.iter_rules()}
        self.assertLessEqual({"/", "/games", "/team-data/<team_name>",
                              "/player-stats/<player_id>"}, rules)

    def test_the_index_renders_from_the_prediction_cache(self):
        calls, fake_game_data = game_data_stub()
        with mock.patch.object(webapp, "fetch_game_data", fake_game_data):
            response = self.client.get("/")

        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Phoenix Suns", body)
        self.assertEqual(sorted(calls), ["betmgm", "draftkings", "fanduel"])

    def test_games_renders_from_the_scoreboard_cache(self):
        calls, fake_tables = scoreboard_stub()
        with mock.patch.object(webapp, "scoreboard_tables", fake_tables):
            response = self.client.get("/games?date=2026-01-15")

        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Detroit Pistons", body)
        self.assertIn("Phoenix Suns", body)
        self.assertEqual(calls, ["2026-01-15"])

    def test_games_still_rejects_an_unusable_date_without_fetching(self):
        calls, fake_tables = scoreboard_stub()
        with mock.patch.object(webapp, "scoreboard_tables", fake_tables):
            response = self.client.get("/games?date=2026-1-5")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(calls, [], "a rejected date still cost an upstream call")

    def test_team_data_still_answers_and_still_rejects_an_unknown_team(self):
        requests_stub = FakeRequests({
            "https://tank01-fantasy-stats.p.rapidapi.com/getNBATeamRoster": ROSTER_PAYLOAD,
        })
        with mock.patch.object(webapp, "RAPIDAPI_KEY", "test-key"), \
                mock.patch.object(webapp, "requests", requests_stub):
            response = self.client.get("/team-data/Detroit Pistons")
            unknown = self.client.get("/team-data/Charlotte Bobcats")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Cade Cunningham", response.get_data(as_text=True))
        self.assertEqual(unknown.status_code, 400)
        self.assertEqual(len(requests_stub.calls), 1)

    def test_team_data_still_reports_an_unconfigured_service(self):
        with mock.patch.object(webapp, "RAPIDAPI_KEY", ""):
            response = self.client.get("/team-data/Detroit Pistons")

        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.get_json()["success"])

    def test_player_stats_still_answers_and_still_rejects_a_bad_id(self):
        requests_stub = FakeRequests({
            "https://tank01-fantasy-stats.p.rapidapi.com/getNBAPlayerInfo": PLAYER_INFO_PAYLOAD,
            "https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForPlayer": PLAYER_GAMES_PAYLOAD,
        })
        with mock.patch.object(webapp, "RAPIDAPI_KEY", "test-key"), \
                mock.patch.object(webapp, "requests", requests_stub):
            response = self.client.get("/player-stats/2544")
            bad = self.client.get("/player-stats/2544;drop")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Cade Cunningham", response.get_data(as_text=True))
        self.assertEqual(bad.status_code, 400)

    def test_every_route_costs_one_upstream_call_per_window(self):
        """The same property, checked at each route rather than in the abstract.

        One upstream call per window for the scoreboard and for each sportsbook's
        prediction run; a player looked up twice inside the window is billed
        once. A route that had quietly been forked off the shared cache would show
        up here as a count of 2.
        """
        scoreboard_calls, fake_tables = scoreboard_stub()
        game_data_calls, fake_game_data = game_data_stub()
        player_calls = []

        def fake_player_stats(player_id, season):
            player_calls.append(player_id)
            return {"success": True, "player": {"name": "Cade Cunningham"}}

        requests_stub = FakeRequests({
            "https://tank01-fantasy-stats.p.rapidapi.com/getNBAPlayerInfo": PLAYER_INFO_PAYLOAD,
            "https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForPlayer": PLAYER_GAMES_PAYLOAD,
        })

        with mock.patch.object(webapp, "scoreboard_tables", fake_tables), \
                mock.patch.object(webapp, "fetch_game_data", fake_game_data), \
                mock.patch.object(webapp, "fetch_player_stats", fake_player_stats), \
                mock.patch.object(webapp, "requests", requests_stub), \
                mock.patch.object(webapp, "RAPIDAPI_KEY", "test-key"):
            with frozen_clock() as clock:
                self.client.get("/")
                self.client.get("/")
                self.client.get("/games?date=2026-01-15")
                self.client.get("/games?date=2026-01-15")
                self.client.get("/player-stats/2544")
                self.client.get("/player-stats/2544")

                self.assertEqual(len(scoreboard_calls), 1)
                self.assertEqual(sorted(game_data_calls),
                                 ["betmgm", "draftkings", "fanduel"])
                self.assertEqual(player_calls, ["2544"])
                # self.client.get("/team-data/...") is deliberately absent: that
                # route has no cache at all today (issue #35, item 2), so it is
                # not evidence of anything here.

                clock.advance(max(webapp.SCOREBOARD_TTL_SECONDS,
                                  webapp.PLAYER_STATS_TTL_SECONDS) + 1)
                self.client.get("/")
                self.client.get("/games?date=2026-01-15")
                self.client.get("/player-stats/2544")

        self.assertEqual(len(scoreboard_calls), 2)
        self.assertEqual(len(game_data_calls), 6, "a sportsbook refetched inside its window")
        self.assertEqual(player_calls, ["2544", "2544"])


if __name__ == "__main__":
    unittest.main()