from datetime import date, datetime, timedelta, timezone
import json
import os
import sys
import threading
from zoneinfo import ZoneInfo
from flask import Flask, render_template,jsonify, request
import subprocess, requests, re, time
from collections import OrderedDict
from werkzeug.exceptions import HTTPException


# Third-party credentials come from the environment, never from source.
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = "tank01-fantasy-stats.p.rapidapi.com"
PLAYER_STATS_SEASON = "2024"

# `main.py` scrapes odds and then runs TensorFlow/XGBoost, so it is slow but finite.
PREDICTION_TIMEOUT_SECONDS = 300
# How long a request waits for a free prediction slot before giving up.
PREDICTION_SLOT_TIMEOUT_SECONDS = 30
# Bound on every outbound RapidAPI call.
API_TIMEOUT_SECONDS = 15
# Cap on prediction children running at the same time (one per sportsbook).
MAX_CONCURRENT_PREDICTIONS = 3
# How long one sportsbook's board is served from cache before the pipeline runs
# again. Every window in this app is measured from the moment its own fetch
# finished, not from a wall-clock boundary -- see TTLCache.
PREDICTIONS_TTL_SECONDS = 600
# How long a cached player-stats lookup is served before it is fetched again.
PLAYER_STATS_TTL_SECONDS = 600

# The /games page asks stats.nba.com for one day's scoreboard and reuses that
# answer for this long. ONE upstream call per page load, and none from the
# browser at all: NBA blocks bursts from server IPs (issue #27 records hard HTTP
# 403s), so a page that polled every few seconds would earn a 403 within the
# first minute and then show nothing for hours afterwards. A minute of staleness
# on a live score is a fair price for staying reachable at all.
#
# "This long" means from the moment that day was fetched (see TTLCache), so no
# two page loads one second apart can both reach NBA: issue #35 is the bucketed
# clock that used to make that possible.
SCOREBOARD_TTL_SECONDS = 60
# Bound on the outbound scoreboard call. The same value as API_TIMEOUT_SECONDS so
# a slow NBA and a slow RapidAPI fail at the same point in time rather than one
# hanging twice as long as the other.
SCOREBOARD_TIMEOUT_SECONDS = API_TIMEOUT_SECONDS

# The prediction runners end their run with exactly one machine-readable line:
#
#     __NBA_PREDICTIONS_JSON__ {"schema":1,"games":[{...}]}
#
# The contract, including every per-game key, is documented in
# src/Predict/XGBoost_Runner.py (and its twin in NN_Runner.py). Everything else
# main.py prints is human-readable and free to be reworded: this app no longer
# scrapes it, so a changed `print` cannot silently empty the page.
PREDICTIONS_SENTINEL = "__NBA_PREDICTIONS_JSON__ "
PREDICTIONS_SCHEMA_VERSION = 1

# Derived from this file instead of the launch directory: `flask run` is started
# from Flask/ in the README but from the repo root by anything else, and the child
# process has to find main.py either way.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN_SCRIPT = os.path.join(REPO_ROOT, "main.py")

# An expected value is Pwin * payout - (1 - Pwin) * 100, so it can never fall below
# -100 nor rise above the payout its moneyline offers (at most 100 on a short-priced
# side, and the quoted odds on a long shot). Values outside that are a changed scale
# rather than a changed number, and are logged instead of being shown as a green edge.
EV_FLOOR = -100.0
EV_CEILING_WITHOUT_ODDS = 100.0
EV_TOLERANCE = 0.01

PLAYER_ID_RE = re.compile(r"^[0-9]+$")

# One lock per sportsbook so concurrent requests share one child process, plus a
# global semaphore so a burst of requests cannot spawn unbounded children.
sportsbook_locks = {}
sportsbook_locks_guard = threading.Lock()
prediction_slots = threading.BoundedSemaphore(MAX_CONCURRENT_PREDICTIONS)


def sportsbook_lock(sportsbook):
    """Return the single-flight lock guarding one sportsbook's prediction run."""
    with sportsbook_locks_guard:
        return sportsbook_locks.setdefault(sportsbook, threading.Lock())


class TTLCache:
    """One bounded store of per-key values, each with its own real expiry window.

    This is the cache every route in this app shares. It replaces the
    `lru_cache` + `get_ttl_hash` pair they used to cache with, which keyed each
    entry on `round(time.time() / seconds)`: the window a caller actually got was
    whatever was left of a wall-clock bucket, so a request landing a millisecond
    before a boundary refreshed immediately and a client could be handed two
    upstream calls back to back. That is mild for the RapidAPI routes and
    materially worse for `/games`, which sits on ScoreboardV3 -- the endpoint
    this repo documents as 403-ing under burst (issue #27; issue #35 for the
    bucketing itself).

    The window is not fixable from the arithmetic: `floor` removes the double
    call and hands the unlucky requester a nearly-full window instead of a
    nearly-empty one. So nothing buckets the clock here. Each value is stored
    next to the moment it was stored and the window is measured against THAT, and
    two calls for one key therefore cannot land closer together than the window
    the caller asked for.

    A key is whatever the caller wants to cache on -- a date, a sportsbook, a
    (player, season) pair -- and nothing is shared between keys. A fetch that
    raises stores nothing and propagates, exactly as an `lru_cache` entry would;
    that is why `fetch_games` returns its failures as values.
    """

    def __init__(self, maxsize=128):
        # 128 is the maxsize the bare `@lru_cache()` these functions used to carry,
        # kept deliberately: the bound is what stops `/games?date=` walking a year
        # of history from growing this without limit (issue #35, item 2).
        self.maxsize = maxsize
        self.entries = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key, fetch, seconds):
        """The cached value for `key`, calling `fetch()` at most once per window.

        `fetch` is a callable rather than a value so a miss can be decided and
        then served without holding the lock: an upstream call takes seconds, and
        holding a shared lock through one would serialise every key behind it.
        Two threads can therefore both miss the same empty key and both fetch it,
        which is exactly what the `lru_cache` it replaces allowed as well.
        Closing that would mean single-flighting every key, which is a larger
        change than this fix calls for.
        """
        with self.lock:
            entry = self.entries.get(key)
            if entry is not None:
                stored, value = entry
                if time.time() - stored <= seconds:
                    self.entries.move_to_end(key)
                    return value

        value = fetch()

        with self.lock:
            # Stamped when the fetch RETURNS, not when it was asked for: a slow
            # upstream then eats into the NEXT window rather than the one before
            # it, so the gap between two upstream calls is always at least
            # `seconds`. Least-recently-used eviction, as `lru_cache` did.
            self.entries[key] = (time.time(), value)
            self.entries.move_to_end(key)
            while len(self.entries) > self.maxsize:
                self.entries.popitem(last=False)
        return value

    def clear(self):
        """Forget every key. The test suite, and nothing else."""
        with self.lock:
            self.entries.clear()


def rapidapi_headers():
    """Build the RapidAPI request headers from the RAPIDAPI_KEY env var."""
    return {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }


def rapidapi_unavailable_response():
    """503 returned by the data endpoints when RAPIDAPI_KEY is not configured."""
    app.logger.error("RAPIDAPI_KEY is not set; refusing to call the RapidAPI endpoints")
    return jsonify({
        'success': False,
        'error': 'Player data service is not configured'
    }), 503


PREDICTION_CACHE = TTLCache()


def fetch_prediction(sportsbook, seconds=PREDICTIONS_TTL_SECONDS):
    """Run the prediction pipeline for one sportsbook at most once per TTL window.

    Keyed on the sportsbook and served until `seconds` have passed since THAT
    book's own run finished. The per-sportsbook lock in the wrappers below makes
    concurrent requests wait for the in-flight run instead of each spawning their
    own child process.
    """
    return PREDICTION_CACHE.get(
        sportsbook, lambda: fetch_game_data(sportsbook=sportsbook), seconds)


def fetch_fanduel(seconds=PREDICTIONS_TTL_SECONDS):
    with sportsbook_lock("fanduel"):
        return fetch_prediction("fanduel", seconds)


def fetch_draftkings(seconds=PREDICTIONS_TTL_SECONDS):
    with sportsbook_lock("draftkings"):
        return fetch_prediction("draftkings", seconds)


def fetch_betmgm(seconds=PREDICTIONS_TTL_SECONDS):
    with sportsbook_lock("betmgm"):
        return fetch_prediction("betmgm", seconds)


# `cache_clear` is the seam the `@lru_cache`d versions of these three functions
# had and that the test suite still calls on fetch_games. Each alias is bound to
# its own store, so clearing one route's cache cannot quietly drop another's.
fetch_prediction.cache_clear = PREDICTION_CACHE.clear


class PredictionBoard(dict):
    """One sportsbook's games, keyed "away:home", plus the status of the run.

    Subclasses dict because Flask/templates/index.html iterates the result with
    `.items()` and that template cannot be edited: the status lives on the side so
    the caller can tell a genuinely empty slate ("no_games" -- a day off) from a run
    that produced nothing machine-readable ("unavailable"), which used to render as
    an indistinguishable blank page.
    """

    def __init__(self, games=None, status="ok", detail=""):
        super().__init__(games or {})
        self.status = status
        self.detail = detail


def board_status(board):
    """Status of a prediction board, failing closed on anything unexpected."""
    return getattr(board, "status", "unavailable")


def board_detail(board):
    """Short reason a prediction board came back the way it did."""
    return getattr(board, "detail", "")


def strict_predictions_decoder():
    """A JSON decoder that refuses the non-standard NaN/Infinity literals.

    The runners map non-finite floats to null, so meeting a bare NaN here means the
    contract is broken on the producer side; raising makes that visible instead of
    quietly accepting a value no other JSON parser would. raw_decode stops at the
    end of the JSON value, so trailing noise on the line (a colour reset written
    after the payload, say) is ignored rather than failing the whole run.
    """
    def reject(name):
        raise ValueError(f"payload contains the non-finite literal {name}")

    return json.JSONDecoder(parse_constant=reject)


def find_predictions_payload(stdout):
    """Return the JSON text that follows the sentinel line, or None.

    Every line is scanned and the LAST match wins: main.py prints a closing
    separator after the runner returns, so the sentinel is the runner's last line
    rather than the last line of stdout. rfind also survives anything else having
    written to the same line first.
    """
    found = None
    for line in (stdout or "").splitlines():
        marker = line.rfind(PREDICTIONS_SENTINEL)
        if marker != -1:
            found = line[marker + len(PREDICTIONS_SENTINEL):]
    return found


def entry_text(value):
    """A payload string, or None when it is missing or blank."""
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return text or None


def entry_number(value, integer=False):
    """A payload number, kept exactly as sent, or None when it is not one.

    The template filters most of these through `|float` and `|int`, so anything
    unparseable is dropped to None rather than handed to Jinja as a string. Odds
    are the one field that must be whole: American odds are integers.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if integer else value


def check_expected_value(team, value, odds):
    """Log a warning when an EV falls outside the range its moneyline allows.

    This only warns. The number the model produced is the number the page shows;
    quietly rewriting it would hide the very scale change this is looking for.
    """
    if value is None:
        return
    ceiling = EV_CEILING_WITHOUT_ODDS if odds is None else max(EV_CEILING_WITHOUT_ODDS, odds)
    if value < EV_FLOOR - EV_TOLERANCE or value > ceiling + EV_TOLERANCE:
        app.logger.warning(
            "Expected value %s for %s is outside the %s..%s its %s moneyline allows; "
            "the scale may have changed",
            value, team, EV_FLOOR, ceiling, odds,
        )


def build_game(entry):
    """Map one payload entry onto the field names Flask/templates/index.html reads.

    The key order matches the dict this replaced so the rendered card keeps the same
    shape, and the names are the template's: it reads away_team, home_team,
    away_confidence, home_confidence, ou_pick, ou_value, ou_confidence,
    away_team_ev, home_team_ev, away_team_odds and home_team_odds. Returns None for
    an entry without both team names, which the caller drops rather than render as
    an empty card.
    """
    home_team = entry_text(entry.get("home_team"))
    away_team = entry_text(entry.get("away_team"))
    if home_team is None or away_team is None:
        return None

    home_odds = entry_number(entry.get("home_team_odds"), integer=True)
    away_odds = entry_number(entry.get("away_team_odds"), integer=True)
    home_ev = entry_number(entry.get("home_team_ev"))
    away_ev = entry_number(entry.get("away_team_ev"))
    check_expected_value(home_team, home_ev, home_odds)
    check_expected_value(away_team, away_ev, away_odds)

    return {
        "away_team": away_team,
        "home_team": home_team,
        "away_confidence": entry_number(entry.get("away_confidence")),
        "home_confidence": entry_number(entry.get("home_confidence")),
        "ou_pick": entry_text(entry.get("ou_pick")),
        "ou_value": entry_number(entry.get("ou_value")),
        "ou_confidence": entry_number(entry.get("ou_confidence")),
        "away_team_ev": away_ev,
        "home_team_ev": home_ev,
        "away_team_odds": away_odds,
        "home_team_odds": home_odds,
        "away_kelly_fraction": entry_number(entry.get("away_kelly_fraction")),
        "home_kelly_fraction": entry_number(entry.get("home_kelly_fraction")),
    }


def parse_predictions(stdout, sportsbook="unknown"):
    """Build one sportsbook's board from the machine-readable line in main.py's output.

    Returns a PredictionBoard whose `status` is one of:
      "ok"           games parsed and ready to render
      "no_games"     the run succeeded and reported an empty slate, which is a real
                     answer for a day off and not a fault
      "degraded"     the payload parsed but some entries were unusable
      "unavailable"  no machine-readable line, or one that did not decode
    """
    raw = find_predictions_payload(stdout)
    if raw is None:
        app.logger.error(
            "The %s prediction run printed no %s line, so there is nothing to render",
            sportsbook, PREDICTIONS_SENTINEL.strip(),
        )
        return PredictionBoard(status="unavailable", detail="no machine-readable output")

    try:
        payload, _end = strict_predictions_decoder().raw_decode(raw)
    except ValueError as error:
        app.logger.error("Could not decode the %s predictions payload: %s", sportsbook, error)
        return PredictionBoard(status="unavailable", detail=f"undecodable payload: {error}")

    if not isinstance(payload, dict) or payload.get("schema") != PREDICTIONS_SCHEMA_VERSION:
        app.logger.error(
            "Unsupported predictions payload from %s: schema %r, expected %s",
            sportsbook,
            payload.get("schema") if isinstance(payload, dict) else payload,
            PREDICTIONS_SCHEMA_VERSION,
        )
        return PredictionBoard(status="unavailable", detail="unsupported schema")

    games = payload.get("games")
    if not isinstance(games, list):
        app.logger.error("The %s predictions payload has no games list", sportsbook)
        return PredictionBoard(status="unavailable", detail="no games list")

    board = PredictionBoard()
    unusable = 0
    for entry in games:
        game = build_game(entry) if isinstance(entry, dict) else None
        if game is None:
            unusable += 1
            continue
        app.logger.debug("%s prediction: %s", sportsbook, json.dumps(game, sort_keys=True))
        board[f"{game['away_team']}:{game['home_team']}"] = game

    if unusable:
        app.logger.error(
            "Dropped %s of %s prediction entries from %s as unusable",
            unusable, len(games), sportsbook,
        )
        board.status = "degraded"
        board.detail = f"{unusable} unusable entries"
    elif not board:
        board.status = "no_games"
        board.detail = "the payload reported no games"

    app.logger.debug("Parsed %s games from the %s predictions line", len(board), sportsbook)
    return board


def run_main_prediction(sportsbook):
    """Run `main.py -xgb -odds=<sportsbook>` once and return (stdout, failure).

    `failure` is None on success and a short reason otherwise. CalledProcessError
    is turned into a reason rather than left to escape as an unhandled 500, and
    sys.executable plus a __file__-derived working directory keep the child
    findable from any launch directory and inside a virtualenv, where a bare
    "python" does not exist.
    """
    cmd = [sys.executable, MAIN_SCRIPT, "-xgb", f"-odds={sportsbook}"]
    if not prediction_slots.acquire(timeout=PREDICTION_SLOT_TIMEOUT_SECONDS):
        app.logger.error("No prediction slot available for %s", sportsbook)
        raise RuntimeError("prediction capacity exhausted")
    try:
        # stderr is folded into stdout so a TensorFlow/XGBoost complaint reaches the
        # server log instead of vanishing, and so a failure can quote it. The parser
        # looks only for its sentinel line, so the extra noise is harmless.
        stdout = subprocess.check_output(
            cmd,
            cwd=REPO_ROOT,
            timeout=PREDICTION_TIMEOUT_SECONDS,
            stderr=subprocess.STDOUT,
        ).decode("utf-8", "replace")
        return stdout, None
    except subprocess.TimeoutExpired:
        app.logger.error("Prediction run for %s timed out after %s seconds",
                         sportsbook, PREDICTION_TIMEOUT_SECONDS)
        raise
    except subprocess.CalledProcessError as error:
        tail = (error.output or b"").decode("utf-8", "replace").strip()[-500:]
        app.logger.error("Prediction run for %s exited with status %s: %s",
                         sportsbook, error.returncode, tail)
        return None, f"main.py exited with status {error.returncode}"
    finally:
        prediction_slots.release()


def fetch_game_data(sportsbook="fanduel"):
    """Today's board for one sportsbook, in the shape Flask/templates/index.html reads.

    The board comes from the single machine-readable line the runner prints (see the
    contract in src/Predict/XGBoost_Runner.py) instead of from three regexes over the
    human-readable output, so rewording a `print` can no longer quietly empty the
    page or drop a field. `.status` says whether an empty board means no games or a
    pipeline that produced nothing readable.
    """
    stdout, failure = run_main_prediction(sportsbook)
    if failure is not None:
        return PredictionBoard(status="unavailable", detail=failure)
    return parse_predictions(stdout, sportsbook)


app = Flask(__name__)
app.jinja_env.add_extension('jinja2.ext.loopcontrols')
# Handle every exception here instead of letting the WSGI server render a
# traceback (and, under the debugger, the application source) to the client.
app.config["PROPAGATE_EXCEPTIONS"] = False

JSON_ROUTE_PREFIXES = ("/team-data", "/player-stats")


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    """Log server-side and return a generic response, never source or details."""
    if isinstance(error, HTTPException):
        return error
    app.logger.exception("Unhandled error while serving %s", request.path)
    if any(request.path.startswith(prefix) for prefix in JSON_ROUTE_PREFIXES):
        return jsonify({'success': False, 'error': 'Internal server error'}), 500
    return "Something went wrong. Please try again shortly.", 500


@app.route("/")
def index():
    # One window for all three books, as when a single shared ttl_hash drove them
    # all. Each is still cached separately from the moment its own run finished,
    # so a slow book cannot shorten the other two's window.
    seconds = PREDICTIONS_TTL_SECONDS
    try:
        fanduel = fetch_fanduel(seconds)
        draftkings = fetch_draftkings(seconds)
        betmgm = fetch_betmgm(seconds)
    except Exception:
        app.logger.exception("Failed to build today's predictions")
        return "Predictions are temporarily unavailable. Please try again shortly.", 503

    books = {"fanduel": fanduel, "draftkings": draftkings, "betmgm": betmgm}
    # An empty slate and a scraper that returned nothing readable render identically,
    # which is how a broken pipeline used to look like a quiet night. Refuse to render
    # the second one instead of serving a page that claims there are no games.
    broken = [name for name, board in books.items() if board_status(board) == "unavailable"]
    if broken:
        app.logger.error("No machine-readable predictions from %s; not rendering an empty slate",
                         ", ".join(broken))
        return "Predictions are temporarily unavailable. Please try again shortly.", 503
    for name, board in books.items():
        if board_status(board) != "ok":
            app.logger.warning("Predictions for %s came back %s: %s",
                               name, board_status(board), board_detail(board))

    return render_template('index.html', today=date.today(), data=books)


# ---------------------------------------------------------------------------
# Upcoming games -- nba_api ScoreboardV3
# ---------------------------------------------------------------------------
# ONE upstream call per page load, cached for SCOREBOARD_TTL_SECONDS (see the
# constant above for why a browser never polls this page).
#
# `nba_api` is imported INSIDE scoreboard_tables rather than at module scope, for
# the reason src/Utils/nba_client.py documents at length: a module-scope import
# makes every `flask` command fail on a machine that does not have nba_api
# installed, and makes this module unimportable for a test suite that must run
# with no network. Only the one function that talks to NBA pays the import.
#
# TWO UPSTREAM TRAPS, BOTH CONFIRMED AGAINST LIVE PAYLOADS
# --------------------------------------------------------
# 1. NOTHING ON THIS ENDPOINT SAYS WHICH TEAM IS HOME. GameHeader has no
#    home/away flag and LineScore does not either; the only field that carries
#    the split is `gameCode`, which is 'YYYYMMDD/AWAYHOME' ('20261020/PHINYK'
#    is Phoenix at New York). Building a card from row order alone gets it wrong
#    often enough to matter, so gameCode decides and row order is the documented
#    fallback -- see order_team_lines.
# 2. `gameEt` IS EASTERN WALL-CLOCK TIME WITH A 'Z' STAMPED ON IT. For a game
#    tipping at 23:00 UTC it reads 'T19:00:00Z'. It is not UTC and it is not
#    parseable as UTC. `gameTimeUTC` is the only timestamp here this app trusts;
#    `gameEt` and `gameStatusText` are shown as the endpoint's own wording and
#    never converted.

# ScoreboardV3 GameHeader.gameStatus values, read off live responses: 1 before
# tip-off, 2 while the game is being played, 3 once it is over. Anything else is
# rendered as the raw number rather than guessed at. Each value maps to (slug
# used in the template's class name and logic, wording shown on the page).
GAME_STATUS_SCHEDULED = 1
GAME_STATUS_IN_PROGRESS = 2
GAME_STATUS_FINAL = 3
GAME_STATUS_LABELS = {
    GAME_STATUS_SCHEDULED: ("scheduled", "Scheduled"),
    GAME_STATUS_IN_PROGRESS: ("live", "In progress"),
    GAME_STATUS_FINAL: ("final", "Final"),
}

# LeagueID for the NBA. nba_api's own default is also '00'; it is named here so
# the value this page depends on is visible rather than inherited.
SCOREBOARD_LEAGUE_ID = "00"

# The two tricodes in a gameCode suffix after the slash, and nothing else.
GAME_CODE_TEAM_LENGTH = 6

# A query-string date has to be this shape. strptime alone is not enough: it
# happily accepts '2026-1-5', which is a different string from the '2026-01-05'
# every other link on the page emits.
GAME_DATE_FORMAT = "%Y-%m-%d"
GAME_DATE_LENGTH = 10

# Cookie holding the zone the visitor's browser is in. Server-side rendering
# means the server cannot otherwise know, and guessing a zone would silently put
# every tip-off an hour out for half the country. See resolve_viewer_timezone.
VIEWER_TZ_COOKIE = "viewer_tz"

# HTTP statuses that mean "stop asking" rather than "that went wrong". NBA sends
# 403 when it blocks a burst and 429 when it wants the same thing politely.
RATE_LIMITED_STATUS_CODES = (403, 429)

# Copy for every board status that is not "ok", next to the code that sets it.
# The template is left dumb on purpose: these strings are the whole difference
# between "no games tonight" and "we could not reach NBA", and a quiet day off
# must never be rendered as an outage. Keys match parse_predictions' statuses
# so the two failure paths read the same way.
GAMES_NOTICES = {
    "no_games": (
        "No games scheduled",
        "NBA reports no games on this date. Try the day before or after.",
    ),
    "rate_limited": (
        "NBA is rate limiting this host",
        "stats.nba.com refused the scoreboard request, which is what happens "
        "when too many requests arrive from one IP address in a short window. "
        "This page asks NBA at most once a minute, so please try again shortly.",
    ),
    "unavailable": (
        "Game data is temporarily unavailable",
        "The NBA scoreboard could not be reached. Please try again shortly.",
    ),
}


class ScoreboardUnavailable(Exception):
    """A scoreboard fetch that failed, carrying the HTTP status if there was one.

    nba_api turns a blocked request into a JSON decode error and throws the
    status code away. 403 is not "the network is down": it is this host being
    rate limited, and the page has to say so differently -- otherwise every
    blocked request is reported to whoever is debugging it as a decode bug. The
    status is dug back out of the response object and carried here.
    """

    def __init__(self, reason, status_code=None):
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code


def scoreboard_status_code(endpoint):
    """HTTP status nba_api saw, or None when no response ever arrived.

    `get_response()` is None when nba_api replayed a debug fixture from disk, so
    this returns None rather than raising in that case; the caller treats an
    unknown status as an ordinary failure, which it is.
    """
    response = getattr(
        getattr(endpoint, "nba_response", None), "get_response", lambda: None
    )()
    return getattr(response, "status_code", None)


def scoreboard_tables(game_date, timeout=SCOREBOARD_TIMEOUT_SECONDS, proxy=None):
    """The two ScoreboardV3 tables this page reads, as pandas DataFrames.

    Raises ScoreboardUnavailable for anything that went wrong, carrying the HTTP
    status when a response ever arrived. A timeout, a reset connection and a 403
    all arrive here the same way but are told apart by `status_code`: None means
    the request never got an answer, 403 means this host is being blocked, and
    the page words those differently.

    get_request is switched off in the constructor and issued by hand so the
    failure can still be attributed. Constructing with get_request=True raises
    from inside the constructor and takes the response object -- the only place
    the status code is kept -- with it.
    """
    from nba_api.stats.endpoints.scoreboardv3 import ScoreboardV3

    endpoint = ScoreboardV3(
        game_date=game_date,
        league_id=SCOREBOARD_LEAGUE_ID,
        timeout=timeout,
        proxy=proxy,
        get_request=False,
    )
    try:
        endpoint.get_request()
    except Exception as error:
        raise ScoreboardUnavailable(str(error), scoreboard_status_code(endpoint)) from error

    return endpoint.game_header.get_data_frame(), endpoint.line_score.get_data_frame()


def board_records(frame):
    """A DataFrame as a list of plain dicts, or [] for a missing or empty one.

    pandas boxes numpy scalars to native Python types on the way out of
    `to_dict("records")`, which is what lets `entry_number` and `entry_text`
    below be reused unchanged for these columns.
    """
    if frame is None or len(frame.index) == 0:
        return []
    return [row for row in frame.to_dict("records") if isinstance(row, dict)]


class GamesBoard(dict):
    """One date's games keyed by gameId, plus the status of the fetch.

    Mirrors PredictionBoard for the same reason: the games live in the dict so a
    template can iterate them with `.items()`, and the status lives on the side
    so a caller can tell a genuinely empty slate ("no_games" -- a day off) from a
    fetch that failed ("unavailable" / "rate_limited"). Both render as an empty
    page otherwise, and a broken endpoint is exactly what looks like a quiet
    night. `board_status` and `board_detail` read this class unchanged.
    """

    def __init__(self, games=None, status="ok", detail=""):
        super().__init__(games or {})
        self.status = status
        self.detail = detail


def split_game_code(game_code):
    """(away_tricode, home_tricode) from a gameCode, or (None, None).

    '20261020/PHINYK' is Phoenix at New York: the suffix is two three-letter
    tricodes, away first. Returns None for anything that is not exactly that
    shape so the caller falls back to row order rather than inventing a split --
    a wrong home/away is the kind of mistake nobody catches by reading the card.
    """
    text = entry_text(game_code)
    if text is None or "/" not in text:
        return None, None
    suffix = text.rsplit("/", 1)[1].upper()
    if len(suffix) != GAME_CODE_TEAM_LENGTH or not suffix.isalpha():
        return None, None
    return suffix[:GAME_CODE_TEAM_LENGTH // 2], suffix[GAME_CODE_TEAM_LENGTH // 2:]


def line_tricode(line):
    """The three-letter code on a LineScore row, upper-cased, or None."""
    return (entry_text(line.get("teamTricode")) or "").upper() or None


def order_team_lines(header, lines):
    """(away_line, home_line) for one game, or (None, None) when it has no lines.

    gameCode decides, but only when BOTH of its tricodes actually appear among
    this game's LineScore rows -- which is what keeps a six-letter suffix from
    silently labelling the two teams backwards on an unfamiliar matchup.

    The fallback is the endpoint's own row order: HOME FIRST, away second. That
    is not a guess; it is what live payloads do (game `20261020/PHINYK` returns
    New York on the first LineScore row and Philadelphia on the second), and it
    is pinned by a test so a reordering upstream is a failing test rather than a
    page quietly listing the visitors first.
    """
    if not lines:
        return None, None
    away_code, home_code = split_game_code(header.get("gameCode"))
    if away_code and home_code:
        by_code = {}
        for line in lines:
            code = line_tricode(line)
            if code:
                by_code.setdefault(code, line)
        away = by_code.get(away_code)
        home = by_code.get(home_code)
        if away is not None and home is not None:
            return away, home
    return lines[1], lines[0] if len(lines) > 1 else None


def build_team_side(line, scored):
    """One team on a card, in the fields Flask/templates/games.html reads.

    `scored` is False before tip-off, and then the score is None rather than the
    endpoint's placeholder 0: a card reading "0 - 0" before a game starts
    claims a result nobody has played.
    """
    city = entry_text(line.get("teamCity"))
    name = entry_text(line.get("teamName"))
    full_name = " ".join(part for part in (city, name) if part) or None
    if full_name is None:
        return None

    wins = entry_number(line.get("wins"), integer=True)
    losses = entry_number(line.get("losses"), integer=True)
    # Records read 0-0 before a season starts, which is true but useless; leave
    # it out rather than showing a record that is not a record.
    record = f"{wins}-{losses}" if wins and losses else None

    return {
        "name": full_name,
        "city": city,
        "tricode": line_tricode(line) or "",
        "record": record,
        "score": entry_number(line.get("score"), integer=True) if scored else None,
    }


def parse_utc_stamp(text):
    """An aware UTC datetime from a ScoreboardV3 timestamp, or None.

    Aware or nothing: `datetime.fromisoformat` on '2026-10-20T19:00:00' with no
    offset returns a NAIVE datetime, and a naive tip-off interpreted in the
    server's local zone is an hour out for most of the country. A stamp with no
    offset is therefore read as UTC, which is what this endpoint sends, rather
    than being allowed through as naive.
    """
    stamp = entry_text(text)
    if stamp is None:
        return None
    if stamp.endswith(("Z", "z")):
        stamp = stamp[:-1] + "+00:00"
    try:
        moment = datetime.fromisoformat(stamp)
    except ValueError:
        app.logger.warning("Could not read the scoreboard timestamp %r", text)
        return None
    return moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment


def game_status(header):
    """(status_slug, status_label) for one GameHeader row.

    The slug is one of GAME_STATUS_LABELS' first values or "unknown". An unknown
    code is labelled with the raw number rather than being folded into one of
    the known buckets: a postponed or abandoned game must not be rendered as
    "Scheduled" with a score against it.
    """
    code = entry_number(header.get("gameStatus"), integer=True)
    return GAME_STATUS_LABELS.get(code, ("unknown", f"Status {code}"))


def build_scoreboard_game(header, lines):
    """One GameHeader row plus its LineScore rows, in the template's vocabulary.

    Returns None when the row carries no game id or no usable team rows, and the
    caller drops it rather than render a card with a blank side. This only maps
    what the endpoint actually returned: there is no seed, no network, and no
    inferred field anywhere below.
    """
    game_id = entry_text(header.get("gameId"))
    if game_id is None:
        return None

    away_line, home_line = order_team_lines(header, lines)
    if away_line is None or home_line is None:
        return None

    status, label = game_status(header)
    scored = status in ("live", "final")

    away = build_team_side(away_line, scored)
    home = build_team_side(home_line, scored)
    if away is None or home is None:
        return None

    tipoff = parse_utc_stamp(header.get("gameTimeUTC"))
    return {
        "game_id": game_id,
        # The endpoint's own wording for the state ("Final", "Final/OT",
        # "Q3 04:12", "7:00 pm ET"). Shown verbatim next to our own label
        # because "Final/OT" and a running clock are both worth saying and
        # neither is something to reconstruct.
        "status_text": entry_text(header.get("gameStatusText")),
        "status": status,
        "status_label": label,
        # The raw timing fields the endpoint returns for a game in progress. Kept
        # even though the card shows status_text instead: gameStatusText is a
        # pre-rendered string and these are the numbers behind it.
        "period": entry_number(header.get("period"), integer=True),
        "game_clock": entry_text(header.get("gameClock")),
        "tipoff": tipoff,
        "away": away,
        "home": home,
        "has_score": scored,
    }


def build_games(header_frame, line_frame):
    """Every game on the returned scoreboard, keyed by gameId, as a GamesBoard.

    Status is one of:
      "ok"           games parsed and ready to render
      "no_games"     the endpoint answered and the slate was empty, which is a
                     real answer for a day off and not a fault
      "degraded"     the slate parsed but some rows were unusable
    """
    headers = board_records(header_frame)
    lines = board_records(line_frame)

    if not headers:
        # An empty DataFrame is how this endpoint says "nothing scheduled", not
        # an exception and not an empty page: see the empty-payload handling in
        # fetch_games.
        return GamesBoard(status="no_games", detail="the scoreboard reported no games")

    lines_by_game = {}
    for line in lines:
        lines_by_game.setdefault(entry_text(line.get("gameId")) or "", []).append(line)

    board = GamesBoard()
    unusable = 0
    for header in headers:
        game = build_scoreboard_game(header, lines_by_game.get(entry_text(header.get("gameId")) or "", []))
        if game is None:
            unusable += 1
            continue
        app.logger.debug("%s game %s: %s @ %s (%s)",
                         game["game_id"], game["away"]["name"], game["home"]["name"],
                         game["status_label"], game["status_text"])
        board[game["game_id"]] = game

    if unusable:
        app.logger.error("Dropped %s of %s scoreboard rows as unusable", unusable, len(headers))
        board.status = "degraded"
        board.detail = f"{unusable} of {len(headers)} games could not be read"

    # Earliest tip-off first. Games with no readable tip-off sort last, on the
    # game id, so the order is stable across reloads instead of reshuffling.
    ordered = sorted(board.items(),
                     key=lambda item: (item[1]["tipoff"] is None,
                                       item[1]["tipoff"] or datetime.max.replace(tzinfo=timezone.utc),
                                       item[0]))
    return GamesBoard(ordered, status=board.status, detail=board.detail)


def failed_games_board(game_date, reason, status_code=None):
    """The board for a fetch that failed: a status, never a raised error.

    The reason goes to the log and the page shows GAMES_NOTICES copy, which is
    the same contract the rest of this app follows: detail server-side, a
    generic message to the client, and never source or a traceback.
    """
    status = "rate_limited" if status_code in RATE_LIMITED_STATUS_CODES else "unavailable"
    app.logger.error("NBA scoreboard for %s came back %s (HTTP %s): %s",
                     game_date, status, status_code if status_code else "no response", reason)
    return GamesBoard(status=status, detail=str(reason))


SCOREBOARD_CACHE = TTLCache()


def fetch_games(game_date, seconds=SCOREBOARD_TTL_SECONDS):
    """One date's GamesBoard, with at most one upstream call per TTL window.

    Cached per date and served until `seconds` have passed since THAT date's own
    fetch, so two page loads a second apart cannot both reach NBA however they
    line up against a clock boundary (issue #35).

    FAILURES ARE RETURNED, NOT RAISED, and that is deliberate for the cache as
    much as for the route. This cache stores nothing for a fetch that raised, so
    a raising fetch would be requested again on the very next page load -- which
    is exactly the behaviour that earns a 403 in the first place, and would turn
    one blocked request into a permanent ban. Returning the failure puts it
    behind the same 60-second window as a success.

    The result is shared by every caller for the window, so it is tz-independent
    on purpose: two visitors in different zones must not each evict the other's
    copy. Rendering into a timezone happens per request, in format_tipoff.
    """
    return SCOREBOARD_CACHE.get(game_date, lambda: load_games(game_date), seconds)


# The seam the test suite clears between cases, kept from the `@lru_cache`d
# version of this function. `clear` empties only this store, so the scoreboard
# cache and the prediction cache can never evict each other by accident.
fetch_games.cache_clear = SCOREBOARD_CACHE.clear


def load_games(game_date):
    """The upstream call behind fetch_games: one date's board, or the reason there
    is not one.

    Everything that can go wrong here arrives as a GamesBoard status rather than
    as an exception, for the reason fetch_games gives. An empty header frame is
    NOT one of those failures: it is the endpoint's answer for a day off, and
    build_games turns it into a "no_games" board.
    """
    try:
        header_frame, line_frame = scoreboard_tables(game_date)
    except ScoreboardUnavailable as error:
        return failed_games_board(game_date, error.reason, error.status_code)
    except ImportError as error:
        return failed_games_board(game_date, f"nba_api is not installed: {error}")
    except Exception as error:
        app.logger.exception("NBA scoreboard request for %s failed", game_date)
        return failed_games_board(game_date, error)

    try:
        return build_games(header_frame, line_frame)
    except Exception as error:
        app.logger.exception("NBA scoreboard payload for %s could not be read", game_date)
        return failed_games_board(game_date, error)


def parse_game_date(raw):
    """A `date` from a `YYYY-MM-DD` query value, or None when it is not one."""
    text = entry_text(raw)
    if text is None or len(text) != GAME_DATE_LENGTH:
        return None
    try:
        return datetime.strptime(text, GAME_DATE_FORMAT).date()
    except ValueError:
        return None


def resolve_viewer_timezone():
    """(ZoneInfo, label) for the tip-offs on this page.

    `?tz=` wins, then the cookie the page set from the browser's own zone, then
    UTC. The page renders server-side, so those are the only ways to know where
    the viewer is; guessing a zone would silently move every tip-off by an hour
    for part of the country, and the label is shown on the page so a wrong guess
    is visible rather than believed. The fallback is `timezone.utc`, never a
    naive datetime.

    An unusable name is logged and skipped rather than raised, so one bad query
    value cannot turn into a 500.
    """
    candidates = [request.args.get("tz"), request.cookies.get(VIEWER_TZ_COOKIE)]
    for candidate in candidates:
        name = entry_text(candidate)
        if name is None:
            continue
        try:
            return ZoneInfo(name), name
        except (ValueError, KeyError, OSError):
            app.logger.warning("Ignoring unusable timezone %r", name)
    return timezone.utc, "UTC"


def format_tipoff(tipoff, zone):
    """(ISO string, human string) for one tip-off in `zone`, or (None, "TBD").

    The clock is built from the datetime parts rather than with strftime's `%-I`
    / `%-m` pads: those are glibc extensions that silently raise on Windows, and
    src/Utils/nba_client.py already documents that trap for this repo.
    """
    if tipoff is None:
        return None, "Time TBD"
    local = tipoff.astimezone(zone)
    clock = "%d:%02d %s" % (local.hour % 12 or 12, local.minute,
                            "AM" if local.hour < 12 else "PM")
    label = f"{local.strftime('%a %b')} {local.day}, {clock}"
    # %Z is empty on a plain UTC ZoneInfo, where the offset is the useful part.
    return local.isoformat(), f"{label} {local.tzname() or local.strftime('%Z%z')}"


@app.route("/games")
def games():
    """The NBA slate for one date, rendered server-side from the cached scoreboard.

    `?date=YYYY-MM-DD` picks the day and `?tz=IANA` the display zone; both are
    validated, and an unusable one renders a message rather than raising. Every
    failure mode of this endpoint -- a day with no games, a 403 from a blocked
    burst, a timeout, a payload that does not parse -- is a GamesBoard status
    the template can say something specific about. Nothing here raises, so
    handle_unexpected_error stays a backstop rather than the normal path.
    """
    zone, zone_label = resolve_viewer_timezone()
    today = datetime.now(zone).date()
    # Carried through so the day links preserve the zone the visitor asked for.
    tz_query = request.args.get("tz")

    requested = request.args.get("date")
    day = parse_game_date(requested)
    if requested is not None and day is None:
        app.logger.warning("Rejected an unusable /games date %r", requested)
        return render_template(
            "games.html",
            game_date=None,
            today=today,
            timezone_name=zone_label,
            tz_query=tz_query,
            games=[],
            status="bad_date",
            notice=("Not a date", f"{requested!r} is not a YYYY-MM-DD date."),
            games_count=0,
            yesterday=today - timedelta(days=1),
            tomorrow=today + timedelta(days=1),
        ), 400

    if day is None:
        day = today

    board = fetch_games(day.isoformat(), seconds=SCOREBOARD_TTL_SECONDS)
    status = board_status(board)
    if status != "ok":
        app.logger.warning("Games for %s came back %s: %s", day, status, board_detail(board))

    cards = []
    for game in board.values():
        tipoff_iso, tipoff_label = format_tipoff(game["tipoff"], zone)
        cards.append(dict(game, tipoff_iso=tipoff_iso, tipoff_label=tipoff_label))

    # A failed fetch answers 503, the same way index() does for an unreadable
    # prediction run: the page renders its message, but a crawler or a monitor
    # must not read "no games tonight" as a 200.
    response_status = 200 if status in ("ok", "no_games", "degraded") else 503
    return render_template(
        "games.html",
        game_date=day,
        today=today,
        timezone_name=zone_label,
        tz_query=tz_query,
        games=cards,
        games_count=len(cards),
        status=status,
        notice=GAMES_NOTICES.get(status),
        # Only a degraded count reaches the page. An error `detail` is a
        # traceback-shaped string about an upstream we do not control, and this
        # app's error contract keeps that server-side.
        degraded_note=board_detail(board) if status == "degraded" else None,
        yesterday=day - timedelta(days=1),
        tomorrow=day + timedelta(days=1),
    ), response_status


def get_player_data(team_abv):
    """Fetch player data for a given team abbreviation"""
    url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBATeamRoster"
    headers = rapidapi_headers()
    querystring = {"teamAbv": team_abv}
    
    try:
        response = requests.get(url, headers=headers, params=querystring, timeout=API_TIMEOUT_SECONDS)
        data = response.json()
        
        if data.get('statusCode') == 200:
            formatted_players = []
            roster = data.get('body', {}).get('roster', [])
            
            for player in roster:
                # Format injury status
                injury_status = "Healthy"
                if player.get('injury'):
                    injury_info = player['injury']
                    if injury_info.get('designation'):
                        injury_status = injury_info['designation']
                        if injury_info.get('description'):
                            injury_status += f" - {injury_info['description']}"
                
                formatted_player = {
                    'name': player.get('longName'),
                    'shortName': player.get('shortName'),
                    'headshot': player.get('nbaComHeadshot'),
                    'injury': injury_status,
                    'position': player.get('pos'),
                    'height': player.get('height'),
                    'weight': player.get('weight'),
                    'college': player.get('college'),
                    'experience': player.get('exp'),
                    'jerseyNum': player.get('jerseyNum'),
                    'playerId': player.get('playerID'),
                    'birthDate': player.get('bDay')
                }
                formatted_players.append(formatted_player)
            
            return {
                'success': True,
                'players': formatted_players
            }
        
        return {
            'success': False,
            'error': 'Failed to fetch team data'
        }
        
    except Exception:
        app.logger.exception("Error in get_player_data for %s", team_abv)
        return {
            'success': False,
            'error': 'Failed to fetch team data'
        }


# TODO(security): /team-data and /player-stats are unauthenticated and unmetered.
# They need authentication and rate limiting before this app is exposed publicly.
@app.route("/team-data/<team_name>")
def team_data(team_name):
    # Convert full team name to abbreviation using the existing dictionary
    team_abv = team_abbreviations.get(team_name)

    if not team_abv:
        return jsonify({
            'success': False,
            'error': f'Team abbreviation not found for {team_name}'
        }), 400

    if not RAPIDAPI_KEY:
        return rapidapi_unavailable_response()

    # Fetch and return the player data
    result = get_player_data(team_abv)
    return jsonify(result)


    
@app.route("/player-stats/<player_id>")
def player_stats(player_id):
    # NBA player ids are numeric; reject anything else before spending a call.
    if not player_id or not PLAYER_ID_RE.match(player_id):
        app.logger.warning("Rejected non-numeric player id: %r", player_id)
        return jsonify({
            'success': False,
            'error': 'Invalid player id'
        }), 400

    if not RAPIDAPI_KEY:
        return rapidapi_unavailable_response()

    try:
        return jsonify(cached_player_stats(player_id, PLAYER_STATS_SEASON,
                                           seconds=PLAYER_STATS_TTL_SECONDS))
    except Exception:
        app.logger.exception("Error in player_stats for player %s", player_id)
        return jsonify({
            'success': False,
            'error': 'Failed to fetch player data'
        }), 503


PLAYER_STATS_CACHE = TTLCache()


def cached_player_stats(player_id, season, seconds=PLAYER_STATS_TTL_SECONDS):
    """Cache the (player_id, season) lookups so repeats are not billed twice.

    Each pair is cached for `seconds` from its own fetch, so a repeat inside the
    window costs nothing and the first request after it is the one that pays.
    """
    return PLAYER_STATS_CACHE.get(
        (player_id, season), lambda: fetch_player_stats(player_id, season), seconds)


# The third `cache_clear` seam, for the reason the first one gave.
cached_player_stats.cache_clear = PLAYER_STATS_CACHE.clear


def fetch_player_stats(player_id, season):
    headers = rapidapi_headers()

    # First get player info
    info_url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBAPlayerInfo"
    info_querystring = {"playerID": player_id}
    
    # Then get game stats
    games_url = "https://tank01-fantasy-stats.p.rapidapi.com/getNBAGamesForPlayer"
    games_querystring = {
        "playerID": player_id,
        "season": season,
    }
    
    try:
        # Get both responses
        info_response = requests.get(info_url, headers=headers, params=info_querystring, timeout=API_TIMEOUT_SECONDS)
        games_response = requests.get(games_url, headers=headers, params=games_querystring, timeout=API_TIMEOUT_SECONDS)
        
        info_data = info_response.json()
        games_data = games_response.json()
        
        if info_data.get('statusCode') == 200 and games_data.get('statusCode') == 200:
            # Process games data
            games = list(games_data['body'].values())
            games.sort(key=lambda x: x['gameID'], reverse=True)
            recent_games = games[:10]
            
            # Get player info
            player_info = info_data['body']
            
            # Format injury info
            injury_status = "Healthy"
            if player_info.get('injury'):
                injury_info = player_info['injury']
                injury_status = injury_info

            # Combine and return all data
            return {
                'success': True,
                'games': recent_games,
                'player': {
                    'name': player_info.get('longName'),
                    'position': player_info.get('pos'),
                    'number': player_info.get('jerseyNum'),
                    'height': player_info.get('height'),
                    'weight': player_info.get('weight'),
                    'team': player_info.get('team'),
                    'college': player_info.get('college'),
                    'experience': player_info.get('exp'),
                    'headshot': player_info.get('nbaComHeadshot'),
                    'injury': injury_status
                }
            }

        return {
            'success': False,
            'error': 'Failed to fetch player data'
        }
    except Exception:
        app.logger.exception("Failed to fetch player data for player %s", player_id)
        return {
            'success': False,
            'error': 'Failed to fetch player data'
        }

        
team_abbreviations = {
    'Orlando Magic': 'ORL',
    'Minnesota Timberwolves': 'MIN',
    'Miami Heat': 'MIA',
    'Boston Celtics': 'BOS',
    'LA Clippers': 'LAC',
    'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET',
    'Atlanta Hawks': 'ATL',
    'Cleveland Cavaliers': 'CLE',
    'Toronto Raptors': 'TOR',
    'Washington Wizards': 'WAS',
    'Phoenix Suns': 'PHO',
    'San Antonio Spurs': 'SA',
    'Chicago Bulls': 'CHI',
    'Charlotte Hornets': 'CHA',
    'Philadelphia 76ers': 'PHI',
    'New Orleans Pelicans': 'NO',
    'Sacramento Kings': 'SAC',
    'Dallas Mavericks': 'DAL',
    'Houston Rockets': 'HOU',
    'Brooklyn Nets': 'BKN',
    'New York Knicks': 'NY',
    'Utah Jazz': 'UTA',
    'Oklahoma City Thunder': 'OKC',
    'Portland Trail Blazers': 'POR',
    'Indiana Pacers': 'IND',
    'Milwaukee Bucks': 'MIL',
    'Golden State Warriors': 'GS',
    'Memphis Grizzlies': 'MEM',
    'Los Angeles Lakers': 'LAL'
}