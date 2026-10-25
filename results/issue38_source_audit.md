==============================================================================
ISSUE 38 -- SECOND PREDICTION POINT: SOURCE AUDIT
==============================================================================
odds store: /home/Joel/Desktop/Drive/Legit-NBA-Predictions-gate/Data/OddsData.sqlite  tables=36  odds_snapshots=no
season tables: 36; any timestamp column: False
tables with >1 row per (Date, Home, Away): 0
local lineup/rotation/starter files: 0
roster starter columns: []
roster='confirmed' guard: {'raises_not_implemented': True, 'message': "roster='confirmed' needs a timestamped pre-tip starting-lineup archive; none is checked in or downloadable historically. Capturing one forward is issue #38 - do not substitute roster='top_minutes' and call it confirmed."}

[ ] publication timing verified per season
      No archived pre-tip lineup document exists locally or in a free historical source. The league's 30-minute filing rule is a rule, not per-season evidence of what was published.
[ ] second prediction point written down
      It cannot be defined against a data source that does not exist. The point is T-45min in the issue text, but no artifact records a lineup at that time.
[ ] market price at that timestamp
      Data/OddsData.sqlite stores one day-level row per game (Date is a season/date key, not a timestamp) and has no odds_snapshots table. The closing line is the only stored price, and it is set later than T-45min.
[x] 0.61831 comparison labelled not like-for-like
      This audit does not compare any starter model to 0.61831; the issue text is the only place the number is mentioned.
[x] canonical model unchanged
      No feature or model file was changed by this audit. The roster='confirmed' path was already unreachable from the canonical model; it now raises instead of silently returning the top-minutes proxy.

BLOCKED, NOT NULL: three of the five acceptance criteria cannot be met with the repository's data. There is no timestamped pre-tip starter archive and no intraday odds history. The honest disposition is to keep #38 open and capture both forward.
