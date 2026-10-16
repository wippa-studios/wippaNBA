from datetime import date
import json
import os
import sys
import threading
from flask import Flask, render_template,jsonify, request
from functools import lru_cache
import subprocess, requests, re, time
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
# Cached lookups are keyed on a bucket of this many seconds.
PLAYER_STATS_TTL_SECONDS = 600

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
# -100 nor rise above the payout its moneyline offers (at most 100 on a lay, and the
# quoted odds on a long shot). Values outside that are a changed scale rather than a
# changed number, and are logged instead of being shown as a green edge.
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


@lru_cache()
def fetch_prediction(sportsbook, ttl_hash=None):
    """Run the prediction pipeline for one sportsbook at most once per TTL window.

    `ttl_hash` only takes part in the cache key so the result expires. The
    per-sportsbook lock in the wrappers below makes concurrent requests wait for
    the in-flight run instead of each spawning their own child process.
    """
    del ttl_hash
    return fetch_game_data(sportsbook=sportsbook)


def fetch_fanduel(ttl_hash=None):
    with sportsbook_lock("fanduel"):
        return fetch_prediction("fanduel", ttl_hash)


def fetch_draftkings(ttl_hash=None):
    with sportsbook_lock("draftkings"):
        return fetch_prediction("draftkings", ttl_hash)


def fetch_betmgm(ttl_hash=None):
    with sportsbook_lock("betmgm"):
        return fetch_prediction("betmgm", ttl_hash)


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


def get_ttl_hash(seconds=600):
    """Return the same value withing `seconds` time period"""
    return round(time.time() / seconds)


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
    ttl_hash = get_ttl_hash()
    try:
        fanduel = fetch_fanduel(ttl_hash=ttl_hash)
        draftkings = fetch_draftkings(ttl_hash=ttl_hash)
        betmgm = fetch_betmgm(ttl_hash=ttl_hash)
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
                                            ttl_hash=get_ttl_hash(seconds=PLAYER_STATS_TTL_SECONDS)))
    except Exception:
        app.logger.exception("Error in player_stats for player %s", player_id)
        return jsonify({
            'success': False,
            'error': 'Failed to fetch player data'
        }), 503


@lru_cache()
def cached_player_stats(player_id, season, ttl_hash=None):
    """Cache the (player_id, season) lookups so repeats are not billed twice.

    `ttl_hash` only takes part in the cache key so the cached payload expires.
    """
    del ttl_hash
    return fetch_player_stats(player_id, season)


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