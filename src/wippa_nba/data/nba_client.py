"""A thin, lazily-imported wrapper around ``nba_api`` for team-stat snapshots.

WHY THIS EXISTS
===============
Until now the pipeline hand-assembled a ~900-character query string for
``https://stats.nba.com/stats/leaguedashteamstats`` (see ``data_url`` in
``config.toml``, formatted by ``Get_Data.fetch_data``). Thirty-odd parameters,
URL-encoded by hand, silently rot: rename a single query parameter upstream and
the request still returns HTTP 200 with a differently-shaped payload, or with no
rows, and nothing in the pipeline says so.

``nba_api`` is a maintained wrapper around the SAME host and the SAME endpoint.
Adopting it removes the hand-built string. This module is deliberately THIN: it
owns no behaviour beyond date arithmetic and column normalisation, so swapping
the existing scraper over to it later is a mechanical change, not a rewrite.

Nothing in this module is wired into the pipeline yet. ``Get_Data.py`` and
``main.py`` still hand-build their URLs; rewiring them is a separate task and is
NOT done here.

``nba_api`` IS IMPORTED LAZILY, INSIDE THE FUNCTIONS
====================================================
There is no ``import nba_api`` at the top of this file, and that is deliberate
rather than a style preference. The test suite and the GitHub Actions job are
built to run without heavy or networked dependencies being importable at
collection time; a module-scope import would make ``pytest Tests/`` fail to
*collect* on any environment where ``nba_api`` is absent, and collection
failures are indistinguishable from real test failures in CI output. Importing
inside the functions means the module is always importable, and only the two
functions that actually talk to the network pay the import cost. ``pytest
Tests/`` on a machine with no ``nba_api`` installed is exercised by
``Tests/test_nba_client.py::test_module_imports_without_nba_api``.

OPERATIONAL NOTES -- THE HOST IS THE RISK, NOT THE CODE
=======================================================
Everything below is a property of ``stats.nba.com``, not of this wrapper.
Read it before blaming a fetch for returning nothing.

* **The host is hardcoded.** ``nba_api.stats.library.http.NBAStatsHTTP`` ships
  ``headers = {'Host': 'stats.nba.com', ...}`` and
  ``base_url = 'https://stats.nba.com/stats/{endpoint}'``. There is no setting
  that points it somewhere else, so this wrapper has no "which host" knob to
  offer. It is the same host the repo already scrapes, which is the point.

* **The host times out from datacenter IPs.** It is fronted by Akamai and is
  routinely unreachable from cloud ranges -- connect timeouts, read timeouts,
  TLS resets, with no HTTP status to inspect because no response ever arrives.
  Residential/proxy egress works. **``proxy=`` is the mitigation**: every
  function here forwards ``proxy`` straight to the endpoint.

* **The host blocks many server IPs outright.** A blocked request fails just as
  hard as an unreachable one, and again without a status code. This is why the
  upstream contract matters: **a transport failure is RETRYABLE and means
  nothing about the data.** An empty frame from a *successful* response is a
  different thing entirely -- early in a season the endpoint legitimately omits
  teams that have not played yet. Never collapse the two. ``normalise_snapshot``
  exists to make that distinction cheap: it returns ``None`` for "this snapshot
  is unusable", never for "the network is down".

* **``headers=`` replaces, it does not merge.** The endpoint does
  ``self.headers = headers`` only when ``headers is not None``, so passing a
  dict here DROPS the library defaults -- including ``Host: stats.nba.com`` --
  and the request fails in a way that looks like a network problem. Pass a
  complete header dict, or leave it ``None``.

AVAILABILITY SUMMARY -- ``describe()``
======================================
``describe()`` returns the same three facts as machine-readable strings, in one
place, so a caller (or a log line) can surface them without re-reading this
docstring.
"""

from datetime import date, datetime, timedelta

import pandas as pd

# Bound on every outbound stats.nba.com call. Deliberately the same value as
# REQUEST_TIMEOUT_SECONDS in src/Utils/tools.py, so the scraper and this wrapper
# fail at the same point in time rather than one hanging three times as long.
# nba_api's own endpoint default is 30s, which is tighter than the 60s the rest
# of this repo already relies on; the wrapper default therefore wins.
DEFAULT_TIMEOUT_SECONDS = 60

# A complete leaguedashteamstats snapshot has exactly one row per team.
#
# MIRRORS `EXPECTED_TEAM_ROWS` in src/Process-Data/Create_Games.py, which is the
# enforcing copy -- `build_game_features` returns None for any snapshot whose row
# count is not exactly this, silently dropping the game. That number is
# load-bearing, so it is duplicated rather than imported: src/Process-Data is not
# an importable module name (the hyphen), and Utils must not reach up into
# Process-Data. **If one copy ever changes, change both.**
DEFAULT_EXPECTED_TEAMS = 30

# Columns pinned to the front of every normalised snapshot, in this order, when
# present. TEAM_ID is identity, TEAM_NAME is the join key the trainers drop but
# `Create_Games` keeps. Everything else keeps the order the endpoint returned it
# in, because that order IS the dataset column order the trained models were fit
# against -- re-sorting it here would move every feature under a working model.
SNAPSHOT_LEADING_COLUMNS = ["TEAM_ID", "TEAM_NAME"]


def format_date_to(value):
    """Render a date as the ``M/D/YYYY`` string the endpoint's DateTo wants.

    Matches the shape ``config.toml`` builds by hand
    (``DateTo={month}%2F{day}%2F{year}``), i.e. unpadded month and day. Built
    with an f-string rather than ``strftime('%-m/%-d/%Y')`` because ``%-m`` is a
    glibc extension and silently raises on Windows.
    """
    if value is None:
        return None
    if isinstance(value, str):
        # Already formatted by the caller; pass it through untouched rather than
        # trying to parse a string whose format we cannot guarantee.
        return value
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return f"{value.month}/{value.day}/{value.year}"
    raise ValueError(
        "date_to must be a date, a datetime or a 'M/D/YYYY' string, "
        f"got {value!r}"
    )


def describe():
    """The operational facts about this endpoint, as printable strings.

    Single source of truth for the three things that decide whether a fetch can
    work at all: which host, why it is often unreachable, and that a failure is
    retryable rather than evidence of missing data.
    """
    return {
        "host": "stats.nba.com",
        "host_is_hardcoded": (
            "nba_api.stats.library.http.NBAStatsHTTP hardcodes "
            "base_url='https://stats.nba.com/stats/{endpoint}' and "
            "headers['Host']='stats.nba.com'. There is no host override."
        ),
        "unreachable_from_datacenters": (
            "The host is fronted by Akamai and commonly times out, resets TLS or "
            "drops the connection from datacenter IP ranges, with no HTTP status "
            "to inspect. Pass proxy= to route through residential/allowed egress."
        ),
        "blocks_server_ips": (
            "stats.nba.com blocks many server IPs outright. A blocked or "
            "unreachable request is RETRYABLE and says nothing about the data; an "
            "empty frame from a successful response is a different condition and "
            "often legitimate (teams that have not played yet are omitted)."
        ),
        "headers_replace_do_not_merge": (
            "Passing headers= REPLACES nba_api's defaults, including "
            "'Host': 'stats.nba.com'. Pass a complete dict or leave it None."
        ),
        "date_to_is_inclusive": (
            "DateTo is INCLUSIVE: a snapshot dated D contains games played on D. "
            "Use fetch_pregame_team_stats for anything that predicts a game."
        ),
    }


def _endpoint_data_frame(endpoint):
    """Pull the first result set off an endpoint response as a DataFrame.

    NOTE ON THE DOCUMENTED API: the nba_api README shows
    ``LeagueDashTeamStats(...).league_dash_team_stats.get_data_frame()``, but on
    the installed library the endpoint class exposes ``get_data_frames()`` (a
    LIST) and has no ``get_data_frame`` of its own -- ``get_data_frame`` lives
    on the individual ``DataSet``. Both shapes are handled here so the wrapper
    survives whichever the library settles on.
    """
    if hasattr(endpoint, "get_data_frames"):
        frames = endpoint.get_data_frames()
        if not frames:
            return pd.DataFrame(data={})
        return frames[0]

    data_sets = getattr(endpoint, "data_sets", None)
    if not data_sets:
        return pd.DataFrame(data={})
    return data_sets[0].get_data_frame()


def fetch_team_stats(season, date_to=None, measure_type="Base", per_mode="PerGame",
                     season_type="Regular Season", timeout=DEFAULT_TIMEOUT_SECONDS,
                     proxy=None, headers=None):
    """Fetch one league-wide team-stats snapshot as a pandas DataFrame.

    Thin pass-through to ``nba_api``'s ``LeagueDashTeamStats``. This performs no
    reshaping, no column renaming and no retrying -- it exists so the endpoint is
    named once, in one place, instead of being spelled as 900 characters of
    query string in ``config.toml``.

    *** ``date_to`` IS INCLUSIVE -- THIS IS THE LEAKAGE TRAP ***
    ===========================================================
    The NBA endpoint treats ``DateTo`` as an inclusive bound. A snapshot
    requested with ``date_to=D`` contains the results of games played ON ``D``.

    So for a game being played on ``D``, ``date_to=D`` puts that game's own
    result -- and every ``*_RANK`` column derived from it -- into the feature
    vector. That is target leakage, it is measurable, and it inflated this
    project's headline accuracy: **a rule comparing ``W[D]`` to ``W[D-1]`` for
    both teams recovered the label 96.11% of the time.** The 68.9% figure this
    project reported came from a build that did exactly that.

    For anything that predicts a game, do NOT call this function with the game's
    own date. Call ``fetch_pregame_team_stats``, which cannot be misused this
    way. ``normalise_snapshot`` cannot detect the leak: the frame it sees is
    perfectly well-formed, it simply contains the answer.

    Arguments mirror the endpoint's own constructor kwargs, renamed to the
    vocabulary this repo already uses in ``config.toml``:

    season      Season key in 'YYYY-YY' form, e.g. '2024-25'.
    date_to     Inclusive upper bound on the snapshot. Accepts a ``date``, a
                ``datetime`` or a pre-formatted 'M/D/YYYY' string; rendered as
                M/D/YYYY to match what the hand-built URL produced. ``None``
                leaves it empty, i.e. the whole season.
    measure_type 'Base' matches the ``MeasureType=Base`` the repo has always
                requested. 'Advanced'/'Defense' are different column sets and are
                not interchangeable with the trained models.
    per_mode    'PerGame' matches ``PerMode=PerGame``. NOTE the library's own
                default is 'Totals'; passing nothing here would quietly switch
                every value's scale.
    season_type 'Regular Season' matches ``SeasonType=Regular+Season``.

    timeout/proxy/headers are forwarded to nba_api, which accepts all three as
    constructor kwargs. See the module docstring before setting ``headers``.

    KNOWN DIFFERENCE FROM THE HAND-BUILT URL, RECORDED NOT PAPERED OVER
    ------------------------------------------------------------------
    ``config.toml`` also pins ``DateFrom=10/01/{start_year}``. This function does
    not expose ``date_from``, because the endpoint returns cumulative
    season-to-date rows and the floor is not what selects a game's data -- the
    inclusive ``DateTo`` is. That equivalence has NOT been verified against the
    live host from here (it times out; see the module docstring), so whoever
    rewires ``Get_Data.py`` must confirm it against a real response before
    switching the pipeline over, rather than assuming it.

    Returns a DataFrame with the endpoint's columns, including ``TEAM_ID``,
    ``TEAM_NAME``, the stats and the ``*_RANK`` columns. Feed it to
    ``normalise_snapshot`` before writing it anywhere.
    """
    # Lazy on purpose -- see "IMPORTED LAZILY" in the module docstring.
    from nba_api.stats.endpoints.leaguedashteamstats import LeagueDashTeamStats

    endpoint = LeagueDashTeamStats(
        measure_type_detailed_defense=measure_type,
        per_mode_detailed=per_mode,
        season=season,
        season_type_all_star=season_type,
        date_to_nullable=format_date_to(date_to),
        timeout=timeout,
        proxy=proxy,
        headers=headers,
    )
    return _endpoint_data_frame(endpoint)


def fetch_pregame_team_stats(season, game_date, measure_type="Base",
                             per_mode="PerGame", season_type="Regular Season",
                             timeout=DEFAULT_TIMEOUT_SECONDS, proxy=None,
                             headers=None):
    """Fetch the snapshot a game on ``game_date`` must be predicted FROM.

    Calls ``fetch_team_stats`` with ``date_to = game_date - 1 day``, so the
    frame contains every game played strictly BEFORE ``game_date`` and none of
    the games played on it.

    READ THIS BEFORE "SIMPLIFYING" THE FUNCTION AWAY
    =================================================
    Do not replace a call to this with ``fetch_team_stats(season, game_date)``.
    They differ by exactly one day and that one day is the whole difference
    between a model and a lookup table.

    The endpoint's ``DateTo`` is INCLUSIVE. A snapshot dated ``D`` contains the
    results of games played on ``D``. So building a feature row for a game on
    ``D`` from a snapshot dated ``D`` feeds the game's own result into the
    vector that is supposed to predict it.

    That is not a theoretical concern. This project's reported 68.9% accuracy
    was produced by a build with this bug, and the leak was then measured
    directly: **a rule comparing ``W[D]`` against ``W[D-1]`` for both teams
    recovered the label 96.11% of the time.** Any feature vector carrying ``W``,
    ``L``, ``W_PCT``, ``PTS``, ``PLUS_MINUS`` or their ``*_RANK`` siblings from
    a same-day snapshot carries the label with it.

    The one-day offset is the whole defence. It is applied here, in one place,
    so no caller has to remember it -- which is why this function exists instead
    of a comment asking people to be careful. ``Create_Games.find_prior_snapshot``
    enforces the same rule on the database side ("STRICTLY before game_date, no
    same-day fallback"); this is its network-side twin, and the two must agree.

    ``game_date`` may be a ``date``, a ``datetime`` or a pre-formatted
    'M/D/YYYY' string; it is shifted back one calendar day, which crosses month,
    year and leap-day boundaries correctly because the arithmetic is done on
    ``date`` objects, not on strings.
    """
    if isinstance(game_date, str):
        # Round-trip a pre-formatted string through date so the subtraction below
        # is real calendar arithmetic rather than string surgery. A caller who
        # passes "1/1/2025" gets date(2024, 12, 31), not "12/31/2024" by luck.
        game_date = datetime.strptime(game_date, "%m/%d/%Y").date()
    elif isinstance(game_date, datetime):
        game_date = game_date.date()
    elif not isinstance(game_date, date):
        raise ValueError(
            "game_date must be a date, a datetime or a 'M/D/YYYY' string, "
            f"got {game_date!r}"
        )

    return fetch_team_stats(
        season=season,
        date_to=game_date - timedelta(days=1),
        measure_type=measure_type,
        per_mode=per_mode,
        season_type=season_type,
        timeout=timeout,
        proxy=proxy,
        headers=headers,
    )


def normalise_snapshot(df, expected_teams=DEFAULT_EXPECTED_TEAMS, table_name=None):
    """Make a raw endpoint frame match what the repo's writers expect, or give up.

    CONTRACT: THIS FUNCTION NEVER RAISES FOR A BAD SNAPSHOT. It returns ``None``
    -------------------------------------------------------------------
    ``df`` is ``None``, empty, missing ``TEAM_NAME``, holding duplicate team
    names, or not holding exactly ``expected_teams`` rows -> return ``None``.

    This is a deliberate choice, and the caller-visible consequence is stated
    plainly because it is easy to get backwards:

    * **A ``None`` return means "this snapshot is unusable -- fall back to an
      earlier one."** It is the same signal ``Create_Games.scan_team_tables``
      already acts on when a dated table does not hold exactly
      ``EXPECTED_TEAM_ROWS`` rows, and it exists for the same reason: early in a
      season ``leaguedashteamstats`` omits teams that have not played yet, so a
      short frame is expected rather than exceptional.
    * **A malformed frame is NOT a programming error.** Raising here would abort
      a backfill of hundreds of dates on the first short snapshot. The snapshot
      is discarded, the reason is printed, and the caller moves to the previous
      date.
    * **Only genuine API misuse raises** -- a ``table_name`` that is not a
      date-like value. That is a bug in the calling code, not bad upstream data,
      and it should be loud.
    * The one thing ``None`` must NOT be read as: the network being down. See
      ``describe()``. Transport failures propagate out of ``fetch_team_stats``
      as exceptions; they never reach here.

    What a good snapshot gets
    -------------------------
    * ``TEAM_NAME`` present, non-null, no duplicates. The trainers join on it and
      ``Create_Games`` keeps it in the dataset.
    * Exactly ``expected_teams`` rows (30). ``build_game_features`` rejects
      anything else with ``return None``, dropping the game without a word --
      this check is what turns a silent game-drop into a visible message.
    * **Row order: sorted by ``TEAM_NAME``, index reset to 0..n-1.** This is not
      cosmetic. ``Dictionaries.team_index_current`` is POSITIONAL -- it maps each
      team name to its row index in the snapshot ('Atlanta Hawks': 0,
      'Boston Celtics': 1, ...) and ``build_game_features`` reaches into rows with
      ``team_df.iloc[home_index]``. A snapshot whose rows arrive in a different
      order therefore produces features that are silently attributed to the wrong
      team: no exception, plausible numbers, wrong answers. Sorting by team name
      reproduces exactly the order ``team_index_current`` encodes (it is
      alphabetical by team name, 0..29), so this is a no-op on correctly
      ordered upstream data and a repair when it is not. Caveat: the index map
      also carries ``'LA Clippers'`` as an alias for ``'Los Angeles Clippers'``
      (both 12); a snapshot using the short name sorts elsewhere. The endpoint
      returns the long form.
    * A ``Date`` column holding ``table_name``, matching what ``Get_Data.py``
      writes into every dated table (``df["Date"] = table_name``). Pass
      ``table_name=None`` to leave it out -- but then the column the trainers
      drop by name (``'Date'`` in every ``DROP_COLUMNS`` list) will be absent,
      which those lists tolerate (``errors="ignore"``) and this does not invent.
    * Column order: ``TEAM_ID``, ``TEAM_NAME`` pinned to the front when present,
      every other column left exactly where the endpoint returned it, ``Date``
      appended last. Upstream order is preserved rather than sorted because it
      IS the dataset column order the committed models were trained against;
      reordering it here would move every feature under a working model.
    """
    if df is None:
        print("nba_client: no snapshot to normalise (None).")
        return None

    if df.empty:
        print("nba_client: empty snapshot, skipping.")
        return None

    if "TEAM_NAME" not in df.columns:
        # Documented contract item, stated as code: a frame without the join key
        # is discarded, not repaired. Inventing the column would fabricate team
        # identities, which is worse than losing a date.
        print(
            "nba_client: snapshot has no TEAM_NAME column (got "
            f"{list(df.columns)}), skipping."
        )
        return None

    if df["TEAM_NAME"].isnull().any():
        print("nba_client: snapshot has a null TEAM_NAME, skipping.")
        return None

    if df["TEAM_NAME"].duplicated().any():
        duplicated = sorted(set(df.loc[df["TEAM_NAME"].duplicated(), "TEAM_NAME"]))
        print(f"nba_client: snapshot repeats team names {duplicated}, skipping.")
        return None

    row_count = len(df.index)
    if row_count != expected_teams:
        # The common early-season case: leaguedashteamstats omits teams that have
        # not played yet, so the first couple of weeks of every season come back
        # short. Print, return None, let the caller step back to an earlier date.
        print(
            f"nba_client: snapshot has {row_count} teams, expected "
            f"{expected_teams}, skipping."
        )
        return None

    # mergesort is stable, so two frames with the same names sort identically.
    # drop the old index so positional lookups in build_game_features line up
    # with team_index_current, which is 0-based and contiguous.
    snapshot = df.sort_values("TEAM_NAME", kind="mergesort").reset_index(drop=True)

    if table_name is not None:
        if isinstance(table_name, (date, datetime)):
            table_name = table_name.strftime("%Y-%m-%d")
        elif not isinstance(table_name, str):
            raise ValueError(
                "table_name must be a date, a datetime or a '%Y-%m-%d' string, "
                f"got {table_name!r}"
            )
        snapshot["Date"] = table_name

    # Pin identity first, Date last, leave everything else in upstream order.
    leading = [c for c in SNAPSHOT_LEADING_COLUMNS if c in snapshot.columns]
    trailing = [c for c in ("Date",) if c in snapshot.columns]
    middle = [c for c in snapshot.columns if c not in leading and c not in trailing]
    return snapshot[leading + middle + trailing]
