# The injury-report pipeline: point-in-time availability, as evidence

`src/Process-Data/Get_Injury_Data.py` — the data layer that makes "what did the
market know at 6 p.m. ET?" an answerable question. The NBA publishes an
official injury-report PDF at a deterministic, immutable CDN URL
(`https://ak-static.cms.nba.com/referee/injury/Injury-Report_{YYYY-MM-DD}_{time}.pdf`);
the URL is the timestamp. Fetch the snapshot whose label predates a cutoff and
read it. This page is the ingestion system behind that: how the PDFs are
captured, parsed, stored and queried, the two production bugs caught before
they could damage anything, and what the security audit concluded.

**Reproduce / operate:**

```bash
python -m src.Process-Data.Get_Injury_Data capture --date 2026-10-23 --max-requests 3
python -m src.Process-Data.Get_Injury_Data capture --from 2026-10-01 --to 2026-10-23
python -m src.Process-Data.Get_Injury_Data parse
python -m src.Process-Data.Get_Injury_Data query --player "Doe, John" \
    --game 2026-10-23 --cutoff "2026-10-23T18:00:00-04:00"
python -m src.Process-Data.Get_Injury_Data slots --date 2026-10-23
```

No top-level call fetches anything; `main()` fetches only what the command
line asked for, under the request budget it was given. Tests:
`Tests/test_injury_data.py`.

## Three filename eras

The filename format and the snapshot cadence both changed over the years. One
URL constructor speaks all three, selected by the report date:

| era | dates | cadence | label | example |
|---|---|---|---|---|
| 1 | Dec 2018 – 2020-21 | 2–3 snapshots/day | hour only | `Injury-Report_2019-01-15_05PM.pdf` |
| 2 | 2021-22 – 2025-12-21 | hourly | hour only | `Injury-Report_2023-02-14_12AM.pdf` |
| 3 | 2025-12-22 – now | every 15 minutes | hour + minute | `Injury-Report_2026-10-23_12_00AM.pdf` |

Eras 1 and 2 share a format and differ only in cadence, so the format boundary
that matters to the URL is **2025-12-22**, **not** 2025-26 opening night: the
league switched to minute-bearing filenames *mid-season*.  The 2025-12-22
reading is empirical — during the 2025-26 backfill, every minute-era URL before
that date (e.g. `Injury-Report_2025-12-21_05_00PM.pdf`) returned the CDN's
403, while the matching hour-only URL returned 200 and minute-era URLs are 200
from 2025-12-22 on.  A report before the boundary constructed with the old
constant was therefore a 403 that burned the day's tiny request budget and was
**silently dropped** (see [the boundary incident](#the-2025-26-boundary-incident));
the constant was corrected and the gap re-captured.  The **2021-10-19**
boundary only changes slot density.  Both are constants in the module; every
consumer reads them from there, and a recovered PDF that proves a boundary
wrong changes the constant once.  Era 1 published only 2–3 snapshots a day and
which slots those were is not documented, so that era probes the full 24-hour
grid and **records the absences**: an absent slot is a measurement ("no report
was published at this hour"), not a gap in the download.

Filename labels are **Eastern wall-clock**. Both the label
(`snapshot_ts_et`) and the instant it denotes (`snapshot_ts_utc`) are stored,
converted through `zoneinfo`, and the two DST edge cases (fall-back 1 a.m.
hour → first occurrence; spring-forward 2 a.m. hour → pre-transition offset,
landing just after the gap) are resolved deterministically and pinned by
tests. Naive datetimes are rejected on input — this project has already paid
for a naive-local-time bug once ([measurement bugs](measurement_bugs.md)).

## Point-in-time semantics

`status_known_as_of(player, game, cutoff)` returns the latest designation for
a player in a game that was known **at or before** a cutoff instant. The
example the tests pin: a report that says **Questionable at 5:44 p.m.** and
**Out at 5:45 p.m.** answers "Questionable" to a 5:44 p.m. cutoff and "Out" to
a 5:45 p.m. cutoff. If the answer differs from the final status, that is not a
bug — that is the feature: late scratches happen after T−30m and only show up
in the last snapshots before the game.

One honest caveat is documented in the module: **label time is optimistic by
the upload lag.** The cutoff compares against the report's label, but the PDF
lands on the CDN a few minutes later ("known at T" = "published under a label
at or before T"). A consumer that needs strict visibility semantics should cut
a few minutes earlier than the label boundary.

## Append-only evidence, sha256 provenance

The **evidence** layer — `injury_snapshots` and the raw PDFs under
`Data/injury_raw/` — is append-only. Raw bytes are stored keyed by
`(source_url, sha256)`; a duplicate fetch is a no-op, and `UNIQUE
(source_url, sha256)` is structural so a resumed backfill cannot double-record
an attempt even across processes. A re-fetch of *different* bytes at the same
URL (the CDN is documented as immutable, so this is an anomaly) is recorded as
an **additional** row with its own raw file and reported loudly as a conflict —
never silently overwritten. Nothing issues a DELETE, a DROP or a rewrite of a
stored PDF.

The **derived** layer — `injury_status_events` and `injury_unparsed_rows` — is
a regenerable view of the PDFs. `parse` skips already-parsed snapshots;
`parse --reparse` replaces a snapshot's derived rows (and only theirs) in one
transaction, so a parser improvement can be re-run over stored evidence. Both
timestamps (label and UTC instant) are stored per event. Anything that does
not parse as a row is **recorded with its raw text and a reason, never
dropped** — this repository has a documented allergy to silent drops
([measurement bugs](measurement_bugs.md)), and an injury row that vanishes is
exactly that class of bug.

## Two production bugs, caught before damage

Both were found while building and testing the pipeline, before a single
stored row depended on them. Both are pinned by tests.

**1. The dead upload-lag retry.** The gate for re-probing a recorded 404 keyed
on the **probe clock** — invisible to a 3-hourly batch, whose absences are
always older than the window. Every 404 recorded during the CDN's 5–20-minute
upload lag would have become a **permanent wrong absence**, and a point-in-time
query would then serve a stale designation for exactly the late scratches the
module exists for. The gate now keys on the **slot's** age measured from the
report's own label (`ABSENT_RETRY_SECONDS` = 2 h): a 404 five minutes after a
label is a PDF that has not landed yet and is retried at any capture cadence,
while old slots become final immediately so a historical backfill runs at full
throughput. Pinned by `TestAbsenceResumePolicy`.

**2. Wrap attribution stealing reason text — the Ben Simmons ×16 case.**
Line-at-a-time nearest-row attribution gives the *second* sub-line of a wrapped
reason to the **next** row (14 pt from its own row vs ~8 pt from the next in
the real geometry) — silently truncating one player's reason and inflating
another's. The same nearest-row merge also glued the team-level
"NOT YET SUBMITTED" slate lines that end each report onto the last player's
reason — which is what happened to **Ben Simmons on every 2026-10-23 snapshot
(×16)**, his `Out` designation picking up the next slate's furniture. Wraps
are now attributed as **runs** (prepends keep reading order) and team-level
lines are recorded as furniture, not glued. Pinned by `TestWrapRuns` and the
slate test.

## Security audit: safe unattended, with availability caveats

The verdict: safe to leave running unattended. The caveats are availability
caveats, not safety ones — the upstream can throttle or refuse (it does; NBA
CDN 403s and refusals are routine), budget exhaustion is an explicit stop
rather than a failure, and a resumed run is always safe. The hostile-response
caps (F-1/F-2/F-3) were added on top; everything about a response is untrusted
except the URL space of the CDN host we control:

- a body is read **as a stream** under a hard byte budget (25 MiB); a
  `Content-Length` over the cap is refused before the first byte;
- redirects are followed **manually** (max 3 hops) and every hop is
  re-validated against the injury-report CDN host — no redirect may leave the
  CDN (SSRF), and no request is sent to a host that fails the check;
- every hop, including redirect hops, spends one unit of the request budget,
  so `--max-requests N` means at most N outbound requests, period;
- one wall-clock deadline (60 s) covers the whole fetch, so a slow-drip body
  cannot stall a batch;
- parse-side caps (20 MiB / 200 pages / 500k glyphs per page) so a hostile PDF
  cannot OOM an unattended batch — over-size documents are refused and
  recorded, an over-glyph page is skipped with its reason recorded and the
  other pages still processed.

The supply chain is clear of **CVE-2025-64512**. `pdfplumber` is imported
lazily inside `parse_injury_pdf`, never at module scope, so the URL, time and
store layers stay testable on a machine without the PDF stack.

## Operations

Rate limits, by design: pacing 5–10 s (random) between requests; exponential
backoff on 429/5xx/transport errors (3 attempts, 10 s then 20 s); a request
budget of **20 per invocation** by default (retries count against it) so a
smoke test cannot become a bulk download by forgetting a flag — the historical
backfill passes `--max-requests` explicitly. Future slots are never probed (no
PDF yet, and recording it as absent would poison resumability).

Current state (2026-10-26): the store holds **2,046 snapshots across 1,037
report dates**, 237,436 status events and 40,103 non-row lines (titles, footers
and anything that did not parse as a row — kept, never dropped), with ~32 MB of
raw PDFs under `Data/injury_raw/`. The **live capture loop is running**
(3-hourly slots) and the **historical backfill is complete**: every game date
from 2020-01-01 through the end of 2025-26 has stored coverage (see the table
below; 2019-20's October–December dates are not captured yet — the parser
handles them since #73, but that backfill leg has not been run). Historical
recovery beyond the live page goes through the
Wayback CDX API (PDFs back to Dec 2018) — a single public service, not a CDN:
use CDX only for gap discovery, politely, and enumerate slots deterministically
via this module's own URLs. Nothing in the module calls CDX today.

The backfill now covers every game date from January 2020 onward.  From
`2020-01-01` (the earliest stored report) through the end of 2025-26, every
season in `build_games()` has a stored snapshot for every game date — 901 of
the 941 game dates across 2019-20 → 2025-26, with 2019-20's January-onward
portion and its bubble covered and its October–December dates not captured yet
(the parser handles them since #73; see below).  The capture walks **game dates**
rather than calendar days (a season has ~150 game dates and ~215 calendar days,
so a range capture spends a third of its budget recording 404s on days when no
report can exist):

| season | game dates | snapshots | requests | 404s | parse failures | status events |
|---|---|---|---|---|---|---|
| 2019-20 (Jan–Mar + bubble) | 81 | 240 | 240 | 0 | 0 | 22,007 |
| 2020-21 | 140 | 140 | 140 | 0 | 0 | 12,071 |
| 2021-22 | 165 | 330 | 206 | 0 | 0 | 39,524 |
| 2022-23 | 164 | 328 | 328 | 0 | 0 | 36,523 |
| 2023-24 | 160 | 320 | 320 | 0 | 0 | 40,306 |
| 2024-25 | 162 | 324 | 324 | 0 | 0 | 44,080 |
| 2025-26 | 163 | 326 | 446† | 0 | 0 | 42,695 |

† 2025-26 needed a second pass: the first leg hit the (then-wrong) minute-era
boundary and mounted 326 requests to record 207 snapshots with the rest of the
season silently aborted; the corrected re-run (see
[the boundary incident](#the-2025-26-boundary-incident)) added the gap with a
further 120 requests.  0 absences and 0 errors across every leg, and every
aborted slot was either stored or flagged.

The CDN serves all seasons directly — no Wayback round-trip was needed.  Slot
density differs by era: 2021-22 onward published a 17:00 and a 20:00 report
each game day; 2020-21 published only the 17:00 slot (12:00 returns 403 for
that era, so afternoon games are uncovered there by construction); 2019-20
regular-season dates used 13:00/17:00/20:00 and the bubble used 11:00/14:00/
17:00, matching each era's published cadence.

### The pre-2020 layouts (#73)

The stored window still begins in January 2020, but that is now a capture
choice, not a parser limit.  The 2018-19 and 2019-Q4 reports printed three
column grammars; all three are anchored by label and parse:

| window (observed) | columns | status position |
|---|---|---|
| 2018-12-17 → 2019-11-01 | Date, Time, Matchup, Team, Player, **Category**, Reason, **Current Status**, **Previous Status** | after Reason |
| 2019-11-15 | Date, Time, Matchup, Team, Player, Reason, **Current Status**, **Previous Status** | after Reason |
| 2019-12-01 | Date, Time, Matchup, Team, Player, **Current Status**, Reason, **Previous Status** | before Reason |
| 2019-12-20 onward | the canonical seven | before Reason |

The parser now anchors `Category` and `Previous Status` as delimiter columns.
Their text can no longer leak into Player, Reason or Current Status; neither is
stored, because the evidence schema has no field for them.  `Previous Status`
also exposed a header-matching bug: the one-word `status` alias re-anchored
`COL_STATUS` to the *last* column, so every row lost its Current Status and
was rejected as `missing required cell(s)`.  A column is now anchored once, by
its longest matching label, and never re-anchored.

The same era printed a Team cell on a line of its own (`Minnesota` then
`Timberwolves`) or on the line carrying the first player's wrapped Reason
(`Portland Trail Blazers Injury/Illness - Left shoulder;`).  Those lines are
block **context**: they update the inherited team/time/matchup and complete the
row above (for a wrapped team name) instead of being filed as a
NOT-YET-SUBMITTED slate line.  The wrapped reason still merges into the player
row above or below by the normal run rule.

**Validation.**  Eleven real PDFs spanning 2018-12-17 → 2020-01-05
(2018-12-17, 2019-01-15, 2019-02-01, 2019-03-01, 2019-04-01, 2019-10-25,
2019-11-01, 2019-11-15, 2019-12-01, 2019-12-20, 2020-01-05) parse with
**0 unparsed lines** and only the five official designations.  The
2019-01-15 report that produced 0 rows / 94 rejections now produces **91 rows /
0 rejections**.  A 120-PDF sample of the modern store parses cell-for-cell
identically before and after the change, so no stored row needs a re-parse.
The audit is reproducible and its artifact is committed:
`python scripts/issue73_pre2020_parser_audit.py` →
`results/issue73_pre2020_parser_audit.{json,md}` (12 samples including the
2018-12-14 403 control; **1,140 rows, 0 unparsed**).

**Backfill policy for the newly usable window:**

- The CDN archive starts at **2018-12-17** (2018-12-01 … 12-14 all return
  403).  The 2018-19 season before that date, and anything earlier, would
  need Wayback recovery; it is out of scope here.
- Capture by **game date**, as the existing backfill does: 107 dates from
  2018-12-17 → 2019-04-10 and 69 dates from 2019-10-22 → 2019-12-31.  Era 1
  has no documented slot list, so probe the full 24-hour grid and record
  absences; at the module's 5–10 s pacing this is ≈4,200 probes, 6–12 hours.
  Run it as its own explicit, resumable leg with `--max-requests`; do not fold
  it into the live loop.
- Before treating a date's 403s as "no report", confirm a known-good slot on
  that date (the 2025-26 boundary incident was a wrong URL, not an absence).
  Store every probe outcome; an absence is evidence.
- Parse only the newly captured snapshots.  Do not `parse --reparse` the
  2020+ store for this change: their derived rows are unchanged.
- Report the leg's snapshots, status events, unparsed rows and parse failures
  before using the window in any model test.  This change unlocks the window;
  it does not silently expand the evidence.

## The 2025-26 boundary incident

The 2025-26 backfill leg finished with `0 absences, 0 errors` and a full
TOTAL line — and yet the season had **no evidence at all for 2025-10-21 →
2025-12-21** (56 game dates, the first nine weeks). What happened is the kind
of genuinely silent drop this page exists to stop, so it is recorded in full.

**The cause.** The minute-era filename constant was 2025-10-21 (assumed
"2025-26 opening night").  Before the true switch the league used minute-era
filenames only from **2025-12-22**; for earlier dates those URLs return the
CDN's 403.  A 403 is classified as retryable, and the backfill gave each date a
request budget of exactly its slot count.  So on every early-season date, the
17:00 minute-era URL 403'd, the retry spent the date's second budget unit, a
third attempt could not be funded, and `capture_days` broke with
`budget_exhausted` — **before storing anything**: no fetch, no absence.  The
wrapper (at the time) printed neither `budget_exhausted` nor the skipped slots,
so the date read as a clean no-op.  56 dates × that pattern = "0 absences, 0
errors, everything fetched".  The same mechanism quietly cost the second slot
on 2022-12-18, 2024-02-10, 2025-03-30 and 2026-03-21: those `08PM` reports
exist (probing them returns 200) but were never stored.

**The evidence for the true boundary.** Probing the CDN directly:
`Injury-Report_2025-11-01_05_00PM.pdf`, `2025-12-15`, `2025-12-20` and
`2025-12-21` all return 403 while the matching hour-only `05PM` filenames
return 200; `2025-12-22_05_00PM.pdf` (the first stored minute-era row) and
every later date return 200.  So the league's filename format changed
**mid-season on 2025-12-22**, not at opening night.

**The fixes.**
- `MINUTE_ERA_START` corrected to 2025-12-22; the era test now pins the two
  dates either side of the true boundary.
- The backfill wrapper now flags `budget_exhausted` per date, counts it in the
  TOTAL line, and returns a nonzero exit code when any date aborted, so a
  mid-date budget abort can never pass as a clean no-op again.  It also grants
  each date `slots × MAX_ATTEMPTS` of headroom (still inside the run-wide
  `--max-requests` cap) so one retry can no longer starve the rest of a date.
- The 2025-26 gap (2025-10-21 → 2025-12-21) was re-captured as hour-only, and
  the four lost `08PM` slots were re-captured individually.

After the re-capture, 2025-26 is the one season in the store whose snapshots
carry **two label families** (hour-only `05PM`/`08PM` before 2025-12-22,
minute `05_00PM`/`08_00PM` after).  Both denote the same 17:00/20:00 clock
slots and convert to UTC identically for the point-in-time gate; only the
filename style differs.  The era-1 caveat applies equally here: an absent or
403-dropped slot on an old date is now recorded or flagged, never silent.

## What it measured (#37): season protocol null, criterion signal

`scripts/issue37_injury_reports.py` turns the stored designations into a
same-game discount on each player's expected minutes — the
`(GAME_ID, PLAYER_ID, weight)` multiplier that `availability_features` accepts
— and scores three pre-registered arms (`results/experiments.jsonl`:
`I37-prereg-injury`). It is **not** a re-implementation of the availability
block: the squad, the halflife and the causality stay the block's own, and only
the multiplier changes.

Design: 2021-22, within-season 5-fold walk-forward, the latest report labelled
at least 15 minutes before that game's tip, positive delta =
`loss(reference) − loss(candidate)`, promotion at `|z| > 3` over three arms.
Coverage is reported before any score: **1,144 of 1,230 games (93.0%)** have a
usable pre-tip snapshot, 98.9% of the 39,524 events resolve to a player, and
2,097 of the 11,327 designations land on a player who was on that team's
prior-game squad (the only ones that can move the feature).

| arm | delta | 95% CI | z | covered-only delta |
|---|---|---|---|---|
| `e14_plus_injury` | +0.004619 | [−0.008980, +0.018293] | 0.66 | +0.002302 |
| `canonical_replace` | +0.003868 | [−0.004013, +0.011707] | 0.96 | +0.004068 |
| `canonical_plus` | +0.001235 | [−0.008067, +0.010219] | 0.26 | +0.001033 |
| positive control (availability block) | +0.004801 | [−0.005338, +0.015063] | 0.92 | — |

**Verdict for the season protocol: null, not promoted.** But the reason is
power, not evidence against the feature, and the positive control is what
proves it: the *known* +0.0038
availability effect measures +0.0048 at |z| 0.92 on this same 615-row subset.
The delta's standard error is 0.0040, so detecting a 0.004-nat effect at |z| > 3
would need roughly 6,000 scored games — the availability effect itself only
reached |z| 4.78 on the full 13,199-game frame. A 615-row season subset cannot
answer this question at this bar by construction.

The within-game placebo makes the same point from the other side: shuffling
which designated player receives which weight yields a delta distribution with
mean +0.004379 and sd 0.001313, and the true assignment (+0.003868) beats only
**65% of the 20 shuffles**. Merely making the column vary within a game buys as
much as knowing *which* player is designated, which is what an underpowered
column looks like. The 30-minute buffer changes nothing (1,144 covered, same
delta): in that era the only usable slots are 17:00 and 20:00, and the games
that lose the 17:00 report at a 30-minute buffer had already lost it at 15.

The 2020-21 replication (`I37-prereg-injury-replication`, 964/1,080 covered,
540 scored) is also null and weaker: `canonical_replace` +0.000760 (|z| 0.24),
`e14_plus_injury` −0.000459, `canonical_plus` −0.001736, and its positive
control is −0.001195 (|z| 0.22). A season where the known effect is invisible
cannot contradict or confirm a smaller one.  That is the honest reading of the
**season protocol**, and it is unchanged.

The full 2019-20 → 2025-26 span the two seasons were too small to answer is now
backfilled and scored in the criterion measurement below: on the completed
store the same injury multiplier reads **+0.002632 nats, |z| 4.47** against the
availability block — a criterion-level signal (see next section).  The original
2020-21/2021-22 season protocol above remains a null; the powered replacement
protocol pre-registered as issue #68 (**Promoted**, below) is what carries the
feature over the bar.

### The recorded 43-column bar (criterion 4)

The season protocol answers a better-powered question, but the issue's fourth
criterion asks for the literal one: the **43-column E14 frame against the
0.61831 baseline**. `scripts/issue37_full_frame.py` runs exactly the recorded
recipe on the full 13,199-game population and reproduces the bar to
**0.618307859** (recorded 0.618307119; drift +7.4e-07). Because E14 contains no
availability columns, the honest reference is E14 **plus the unadjusted
availability block**, not E14 itself.

The measurement was run **twice**: first on the two-season store (2026-10-25,
as recorded in `results/issue37_injury_full_frame.json`), then — pre-registered
in `results/experiments.jsonl` as `I37-prereg-injury-full-frame-refresh`,
written before it ran — on the completed store after the 2019-20 → 2025-26
backfill.  Both runs share the identical frame, folds, bootstrap and seed; only
the covered sample changed, exactly as the refresh declaration required.  The
original result is kept and reported alongside the refresh, never replaced:

| frame | log loss | delta vs E14 | 95% CI | \|z\| |
|---|---|---|---|---|
| E14 (43) | 0.618308 | — | — | — |
| E14 + availability (47) | 0.614508 | +0.003799 | [+0.002296, +0.005486] | 4.67 |
| E14 + injury (47) — two-season store | 0.613690 | +0.004618 | [+0.002925, +0.006433] | 5.16 |
| E14 + injury (47) — completed store | 0.611876 | +0.006432 | [+0.004433, +0.008541] | 6.14 |
| injury vs availability — two-season store | — | +0.000818 | [+0.000222, +0.001422] | 2.67 |
| **injury vs availability — completed store** | — | **+0.002632** | **[+0.001471, +0.003779]** | **4.47** |

The availability control reproduces the known +0.0038 at |z| 4.67 on **both**
stores, and the E14 reproduction is identical in both runs — the frame and the
estimator are on their recorded footing throughout.  The injury adjustment over
the availability-only frame reads:

- **Two-season store:** +0.000818, CI entirely above zero, 5 of 5 folds, |z|
  2.67 — below the `|z| > 3` bar.  Coverage was 2,108/13,199 games (16.0%) and
  84% of scored rows were byte-identical between the arms.
- **Completed store:** +0.002632, CI [+0.001471, +0.003779] (p_positive 1.0),
  5 of 5 folds, **|z| 4.47 — clears the `|z| > 3` bar**.  Coverage is
  7,206/13,199 games (54.6%) and 73.6% of scored rows now carry a changed
  weight, so the dilution that bounded the two-season reading is largely gone.

The refresh declaration's hypothesis was that the two-season reading was a real
per-game effect: if so, the identical arm on the larger sample should keep the
same sign.  It did — the point estimate grew 3.2× and its interval stayed clear
of zero.  Two honest caveats on the literal prediction: the standard error
*grew* (0.00031 → 0.00059) rather than shrinking, because the effect itself
grew faster than its noise — the |z| climb to 4.47 is the effect outrunning
its error, not an SE fall.  And this is the **criterion** reading: the arm was
declared after the season protocol and can only answer criterion 4.  Per the
refresh declaration, a `|z| > 3` criterion result is reported as a signal but
**cannot promote** — promotion requires a pre-registered season protocol, and
the original one (2020-21/2021-22) remains a null (2021-22 canonical_replace
+0.003868 |z| 0.96; 2020-21 +0.000760 |z| 0.24).  That criterion reading is
what motivated the powered forward protocol in #68, which promotes the feature
below.  The `availability_weights=` parameter exists, its default is
bit-identical to the pre-#37 block (pinned by a test), and no experiment record
claims otherwise.

### What it measured (#68): promoted on a powered forward holdout

The #37 record ended with a criterion signal that could not promote because the
original season protocol was underpowered.  Issue #68 closed that path with a
new pre-registration (`I68-prereg-injury-promotion`): the exact #37 criterion
arms — E14 + unadjusted availability vs E14 + injury-weighted availability, 47
columns, 15-minute gate — fitted once on everything before 2024-10-22 (10,749
rows) and scored **forward** on the 2024-25 and 2025-26 seasons (2,450 games).
The decision rule was the repository bar, declared before execution: promote
iff the paired delta is positive with |z| > 3 on 2,000 bootstrap resamples
(seed 42) over the combined two-season holdout; the per-season cells were
declared descriptive.

| arm | n | availability | injury | delta | 95% CI | \|z\| |
|---|---|---|---|---|---|---|
| **2024-25 + 2025-26 (decision)** | 2,450 | 0.59742 | 0.59384 | **+0.003579** | **[+0.001729, +0.005376]** | **3.85** |
| 2024-25 only (descriptive) | 1,225 | 0.60041 | 0.59791 | +0.002494 | [−0.000186, +0.005230] | 1.80 |
| 2025-26 only (descriptive) | 1,225 | 0.59443 | 0.58977 | +0.004663 | [+0.002006, +0.007290] | 3.46 |

Positive control on the same forward holdout: availability vs E14 **+0.006576,
|z| 5.49** — the frame resolves the known effect, so the injury delta is not an
underpowered reading.  Coverage: 1,861 of the 2,450 test games (76.0%) carry at
least one injury-adjusted weight.

**Verdict: PROMOTED.** The injury multiplier is the first candidate to clear
`|z| > 3` on a forward season holdout rather than a pooled walk-forward, and it
does so with the positive control resolving in the same frame.  Three caveats
stay attached:

- **Season stability is only descriptive.**  The combined cell is the
  pre-registered decision; 2025-26 alone clears (|z| 3.46), 2024-25 alone does
  not (|z| 1.80).  One season carries most of the evidence, and the interval for
  2024-25 crosses zero.
- **The reproduction anchor drifted, and is kept as failed.**  The E14
  walk-forward reproduces at 0.618303736 against #37's recorded 0.618307119 —
  drift −3.4e-6, tolerance 1e-6, with the scored-row count identical at 6,600.
  The data snapshot has been refreshed since #37 (whose own refresh already sat
  +7.4e-7 from its constant).  Both arms are fitted and scored on the same
  refreshed rows, so the paired delta cancels common drift; the drift is 0.09%
  of the measured effect.  The payload keeps `within_tolerance: false` and the
  ledger carries `I68-annotation-reproduction-drift` rather than a rewritten
  number.
- **Promotion is an evidence verdict, not a model edit.**  The canonical
  `full` model does not consume `availability_weights` by default and is
  unchanged; nothing in the betting stack silently switched to the
  injury-adjusted block.

### What they are worth late (#72): descriptive, no promotion

Issue #72 asked the consumer's question — how much to distrust a designation as
the clock runs down — and answered it as a census, not a model.  On the common
four-cutoff panel a Questionable scratches **50.2% / 51.0% / 50.0% / 50.1%** at
6h/3h/1h/30m; a Probable improves from **12.8% to 9.9%**; and `Available`
scratches **19.8%** at 30m against **9.9%** for Probable, the reverse of the
ordinal weights.  Later reports mostly reclassify players (15.5% of 3h
Questionables flip to Out by 30m) rather than steadily raising the hazard on
the ones who remain Questionable.  The 6h column is 91% yesterday's report.
Nothing is promoted, `STATUS_WEIGHTS` is unchanged, and a recalibration would
need its own pre-registered forward test.  See
[Late scratches](late_scratch.md).

## What this is for

Point-in-time designations are the input the availability feature reads (see
[the availability block](availability_block.md) for what is measured today and
where its boundary is).  The first use is measured above: null on the original
underpowered season protocol, promoted on the powered forward holdout in #68.
The pipeline keeps banking evidence append-only, so any question that outgrows
its first sample can be asked again on a sample that can actually answer it.
