# `/games` — the NBA slate route

**Repo:** `/home/Joel/.cache/nba-fix-worktree` (branch `improve-model-v2`)
**Endpoint:** `GET /games?date=YYYY-MM-DD&tz=<IANA zone>`
**Tests:** `Tests/test_games_route.py` and its neighbours — 65 tests cover the route
without touching the network.

Run it:

```bash
export RAPIDAPI_KEY="<your rapidapi key>"   # optional; team/player endpoints 503 without it
cd Flask && flask run

curl 'http://127.0.0.1:5000/games'
curl 'http://127.0.0.1:5000/games?date=2026-10-21&tz=America/New_York'
```

## How it fetches

Server-rendered from `ScoreboardV3`, **one upstream call per 60-second window**
shared across all viewers. The endpoint is documented as returning 403 under
burst, so the page never polls from the browser — a browser-polled page would both
trip the rate limit and turn every visitor into a request.

`?date=` defaults to today. `?tz=` takes any IANA zone and falls back to a
`viewer_tz` cookie, then to UTC. **The zone is always named on the page**, because a
tip-off time without its zone is not a tip-off time.

## Three upstream traps this route handles

All three confirmed against live payloads, not against the schema.

**Nothing on `ScoreboardV3` says which team is home.** Neither `GameHeader` nor
`LineScore` carries a home/away flag. The only carrier is `gameCode`, formatted
`YYYYMMDD/AWAYHOME` — and `LineScore` orders rows **home first**, which was checked
against real payloads rather than assumed from the field order.

**`gameEt` is Eastern wall-clock with a `Z` stamped on it.** A 23:00 UTC tip-off
reads `19:00:00Z`. Only `gameTimeUTC` is used anywhere in this route; `gameEt` is
the field that produces a page that is six hours wrong and looks right.

**`no_games` is a true answer, not a failure**, and is worded as such. So is a 403,
a dead upstream, and a partial payload — each gets its own status and its own
sentence. Reporting one as another is the failure mode the existing `PredictionBoard`
side-channel exists to prevent: a slate that silently degrades into "nothing is on
tonight" is worse than one that says the upstream is down.

## Logos

Team logos are CSS tricode monograms. `ScoreboardV3` exposes no logo URL, and
hotlinking `cdn.nba.com` would put a second origin in the render path for something
a CSS monogram does better. Nothing can 404.

## Known gaps, stated rather than glossed

The **in-progress branch** (`gameStatus == 2`, rendering a quarter clock) is
modelled from the endpoint's schema and was **not observed live** — no game was in
progress during the build window. The scheduled and final paths were both confirmed
against real payloads. This is the one part of the route that has not been seen
working, and it is labelled as such here rather than left for a user to discover.

## Security, inherited from the audit

The original app shipped a hardcoded RapidAPI key in `Flask/app.py`, reachable
through two unauthenticated routes, and its README told users to run
`flask --debug run` — whose Werkzeug debugger serves application source, key
included, on any unhandled exception. Both are fixed: the key is read from
`RAPIDAPI_KEY` and endpoints fail closed with a 503, and debug mode is gone.
Unhandled exceptions return a generic response logged server-side. See
[What this fork changes](../README.md#security-and-operations).