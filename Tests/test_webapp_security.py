"""Tests for the metering and optional authentication on the public routes.

Issue #50. Three routes spend upstream quota and used to be reachable without
any accounting:

* ``/games?date=...`` -- one ScoreboardV3 call per distinct date;
* ``/team-data/<team>`` -- one RapidAPI roster call;
* ``/player-stats/<id>`` -- one RapidAPI player call.

The fix is a ``before_request`` hook with two independent checks, because the
two attacks are different shapes. ``SlidingWindowLimiter`` caps requests per
client address per minute, which bounds the JSON routes. ``DistinctDateBudget``
caps DISTINCT days per client address per hour, because a caller under the
per-minute ceiling can still walk ``?date=`` one day at a time and cost one
upstream call per day -- the pattern that earns the 403 documented in #27.

Every test here is about the same property: **the limit is a function of the
caller, not of the path.** Varying ``date``, ``team_name``, ``player_id``, a
query parameter, or ``X-Forwarded-For`` must not manufacture a fresh bucket.
The tests therefore drive the real Flask app and stub only the two network
seams (``fetch_games`` and the RapidAPI lookup), so the hook, the error
envelope, the template and the limiter all run for real.

No test sleeps and no test needs a network. The clock is swapped for a
``FakeClock`` in the one test that crosses a window.
"""

import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import pytest

# Flask/app.py imports flask at module scope; skip rather than fail collection
# when the web dependencies are absent, the same way test_webapp_games.py does.
pytest.importorskip("flask")

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "Flask" / "app.py"

SPEC = importlib.util.spec_from_file_location("flask_security_app", MODULE_PATH)
webapp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(webapp)

# A module loaded by path has no package for Flask to infer template/static
# locations from, so point them at the directories `flask run` would find.
webapp.app.template_folder = str(MODULE_PATH.parent / "templates")
webapp.app.root_path = str(MODULE_PATH.parent)
webapp.app.static_folder = str(MODULE_PATH.parent / "static")


class FakeClock:
    """Only the two things the limiters read from `time`."""

    def __init__(self, now=1_000.0):
        self.now = float(now)

    def time(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds
        return self.now


class PublicRouteSecurityTestCase(unittest.TestCase):
    """Shared harness: real routes, stubbed upstreams, deterministic limits."""

    def setUp(self):
        self.client = webapp.app.test_client()
        webapp.reset_rate_limits()

        # Save every module global this suite edits, so one test's limit cannot
        # leak into the next (the app module is shared by the whole file).
        for name in ("API_TOKEN", "RATE_LIMIT_REQUESTS",
                     "GAMES_RATE_LIMIT_REQUESTS", "GAMES_DATE_BUDGET",
                     "GAMES_DATE_WINDOW_SECONDS", "RAPIDAPI_KEY"):
            self.addCleanup(setattr, webapp, name, getattr(webapp, name))

        webapp.API_TOKEN = ""
        webapp.RAPIDAPI_KEY = "test-key"
        webapp.RATE_LIMIT_REQUESTS = 1_000
        webapp.GAMES_RATE_LIMIT_REQUESTS = 1_000
        webapp.GAMES_DATE_BUDGET = 1_000
        webapp.GAMES_DATE_WINDOW_SECONDS = 3_600

        self.games_calls = []
        self.player_calls = []
        self.team_calls = []

        def fake_fetch_games(game_date, seconds=None):
            self.games_calls.append(game_date)
            return webapp.GamesBoard(status="no_games")

        def fake_cached_player_stats(player_id, season, seconds=None):
            self.player_calls.append((player_id, season))
            return {"success": True, "playerID": player_id}

        def fake_get_player_data(team_abv):
            self.team_calls.append(team_abv)
            return {"success": True, "players": []}

        for attribute, function in (
            ("fetch_games", fake_fetch_games),
            ("cached_player_stats", fake_cached_player_stats),
            ("get_player_data", fake_get_player_data),
        ):
            patcher = mock.patch.object(webapp, attribute, function)
            patcher.start()
            self.addCleanup(patcher.stop)

    def request(self, path, address="127.0.0.1", **kwargs):
        """GET `path` from a chosen source address."""
        return self.client.get(path, environ_base={"REMOTE_ADDR": address}, **kwargs)

    # -- JSON data routes ---------------------------------------------------

    def test_player_stats_limit_survives_a_changed_player_id(self):
        webapp.RATE_LIMIT_REQUESTS = 3

        statuses = [
            self.request(f"/player-stats/{player_id}").status_code
            for player_id in (1, 2, 3, 4)
        ]

        self.assertEqual(statuses, [200, 200, 200, 429],
                         "varying player_id opened a new rate-limit bucket")
        blocked = self.request("/player-stats/5")
        self.assertEqual(blocked.status_code, 429)
        self.assertEqual(blocked.get_json(),
                         {"success": False, "error": "Too many requests"})
        self.assertGreaterEqual(int(blocked.headers["Retry-After"]), 1)

    def test_team_data_limit_survives_a_changed_team_name(self):
        webapp.RATE_LIMIT_REQUESTS = 2

        first = self.request("/team-data/Boston%20Celtics")
        second = self.request("/team-data/Miami%20Heat")
        third = self.request("/team-data/Denver%20Nuggets")

        self.assertEqual([first.status_code, second.status_code, third.status_code],
                         [200, 200, 429])
        self.assertEqual(third.get_json()["error"], "Too many requests")
        # The third request never reached the upstream seam.
        self.assertEqual(self.team_calls, ["BOS", "MIA"])

    def test_limit_is_per_client_address(self):
        webapp.RATE_LIMIT_REQUESTS = 1

        self.assertEqual(self.request("/player-stats/1", address="10.0.0.1").status_code, 200)
        self.assertEqual(self.request("/player-stats/2", address="10.0.0.1").status_code, 429)
        self.assertEqual(self.request("/player-stats/3", address="10.0.0.2").status_code, 200)

    def test_forwarded_for_header_cannot_mint_a_new_bucket(self):
        webapp.RATE_LIMIT_REQUESTS = 1

        self.assertEqual(self.request(
            "/player-stats/1", headers={"X-Forwarded-For": "203.0.113.1"}).status_code, 200)
        spoofed = self.request(
            "/player-stats/2", headers={"X-Forwarded-For": "203.0.113.2"})

        self.assertEqual(spoofed.status_code, 429,
                         "X-Forwarded-For was trusted as the limiter key")

    def test_rate_limit_applies_before_a_token_guess(self):
        """Unauthenticated callers are metered too, so guesses are not free."""
        webapp.API_TOKEN = "s3cret"
        webapp.RATE_LIMIT_REQUESTS = 2

        statuses = [self.request("/team-data/Boston%20Celtics").status_code
                    for _ in range(3)]

        self.assertEqual(statuses, [401, 401, 429])

    # -- /games -------------------------------------------------------------

    def test_games_request_limit_survives_a_changed_date(self):
        webapp.GAMES_RATE_LIMIT_REQUESTS = 2
        webapp.GAMES_DATE_BUDGET = 1_000

        first = self.request("/games?date=2026-01-01")
        second = self.request("/games?date=2026-01-02")
        third = self.request("/games?date=2026-01-03")

        self.assertEqual([first.status_code, second.status_code, third.status_code],
                         [200, 200, 429])
        self.assertIn(b"Too many requests from this address", third.data)
        self.assertGreaterEqual(int(third.headers["Retry-After"]), 1)
        self.assertEqual(len(self.games_calls), 2)

    def test_games_distinct_date_budget_survives_a_changed_date(self):
        webapp.GAMES_RATE_LIMIT_REQUESTS = 1_000
        webapp.GAMES_DATE_BUDGET = 2

        self.assertEqual(self.request("/games?date=2026-01-01").status_code, 200)
        self.assertEqual(self.request("/games?date=2026-01-02").status_code, 200)
        blocked = self.request("/games?date=2026-01-03")

        self.assertEqual(blocked.status_code, 429)
        self.assertIn(b"Too many requests from this address", blocked.data)
        self.assertEqual(self.games_calls, ["2026-01-01", "2026-01-02"])

        # A day already paid for is free: the previous/next links keep working.
        # (The stub replaces `fetch_games`, cache included, so the repeated day
        # does reach the stub again; what matters here is that the budget let it
        # through instead of answering 429.)
        self.assertEqual(self.request("/games?date=2026-01-01").status_code, 200)

    def test_invalid_dates_do_not_spend_the_distinct_date_budget(self):
        webapp.GAMES_DATE_BUDGET = 1

        invalid = self.request("/games?date=not-a-date")
        valid = self.request("/games?date=2026-01-01")

        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(valid.status_code, 200,
                         "a rejected date consumed the one-day budget")
        self.assertEqual(self.games_calls, ["2026-01-01"])

    def test_distinct_date_budget_expires_after_its_window(self):
        webapp.GAMES_DATE_BUDGET = 1
        webapp.GAMES_DATE_WINDOW_SECONDS = 60
        clock = FakeClock()

        with mock.patch.object(webapp, "time", clock):
            self.assertEqual(self.request("/games?date=2026-01-01").status_code, 200)
            self.assertEqual(self.request("/games?date=2026-01-02").status_code, 429)

            clock.advance(61)

            self.assertEqual(self.request("/games?date=2026-01-02").status_code, 200)

    def test_blocked_requests_do_not_extend_the_request_window(self):
        webapp.RATE_LIMIT_REQUESTS = 1
        clock = FakeClock()

        with mock.patch.object(webapp, "time", clock):
            self.assertEqual(self.request("/player-stats/1").status_code, 200)
            for _ in range(5):
                self.assertEqual(self.request("/player-stats/1").status_code, 429)

            clock.advance(61)

            self.assertEqual(self.request("/player-stats/1").status_code, 200)

    def test_reset_rate_limits_clears_both_stores(self):
        webapp.RATE_LIMIT_REQUESTS = 1
        webapp.GAMES_RATE_LIMIT_REQUESTS = 1
        self.assertEqual(self.request("/player-stats/1").status_code, 200)
        self.assertEqual(self.request("/games").status_code, 200)

        webapp.reset_rate_limits()

        self.assertEqual(self.request("/player-stats/2").status_code, 200)
        self.assertEqual(self.request("/games").status_code, 200)

    # -- optional bearer authentication -------------------------------------

    def test_data_routes_stay_public_when_no_token_is_configured(self):
        self.assertEqual(self.request("/team-data/Boston%20Celtics").status_code, 200)
        self.assertEqual(self.request("/player-stats/201939").status_code, 200)

    def test_configured_token_rejects_missing_and_wrong_credentials(self):
        webapp.API_TOKEN = "s3cret"

        missing = self.request("/team-data/Boston%20Celtics")
        wrong = self.request("/player-stats/201939",
                             headers={"Authorization": "Bearer wrong"})

        for response in (missing, wrong):
            self.assertEqual(response.status_code, 401)
            self.assertEqual(response.get_json(),
                             {"success": False, "error": "Unauthorized"})
            self.assertEqual(response.headers["WWW-Authenticate"], "Bearer")
        self.assertEqual(self.team_calls, [])
        self.assertEqual(self.player_calls, [])

    def test_configured_token_accepts_the_bearer_header(self):
        webapp.API_TOKEN = "s3cret"

        response = self.request(
            "/team-data/Boston%20Celtics",
            headers={"Authorization": "Bearer s3cret"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.team_calls, ["BOS"])

    def test_token_in_the_query_string_is_not_authentication(self):
        """A token in a URL leaks into logs and history; only the header counts."""
        webapp.API_TOKEN = "s3cret"

        response = self.request("/player-stats/201939?api_token=s3cret")

        self.assertEqual(response.status_code, 401)
        self.assertEqual(self.player_calls, [])

    def test_games_stays_public_html_when_a_token_is_configured(self):
        webapp.API_TOKEN = "s3cret"

        self.assertEqual(self.request("/games").status_code, 200)


if __name__ == "__main__":
    unittest.main()
