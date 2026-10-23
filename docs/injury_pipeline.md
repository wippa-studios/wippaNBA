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
| 2 | 2021-22 – 2024-25 | hourly | hour only | `Injury-Report_2023-02-14_12AM.pdf` |
| 3 | 2025-26 – now | every 15 minutes | hour + minute | `Injury-Report_2026-10-23_12_00AM.pdf` |

Eras 1 and 2 share a format and differ only in cadence, so the format boundary
that matters to the URL is **2025-10-21** (2025-26 opening night); the
**2021-10-19** boundary only changes slot density. Both are constants in the
module; every consumer reads them from there, and a recovered PDF that proves a
boundary wrong changes the constant once. Era 1 published only 2–3 snapshots a
day and which slots those were is not documented, so that era probes the full
24-hour grid and **records the absences**: an absent slot is a measurement
("no report was published at this hour"), not a gap in the download.

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

Current state (2026-10-23): the store holds **119 snapshots / 7,996 status
events**, with 1,677 non-row lines recorded (titles, footers and anything that
did not parse as a row — kept, never dropped) and raw PDFs under
`Data/injury_raw/` across 44 date directories — and growing. The **live capture
loop is running** (3-hourly slots) and the **historical backfill is launched** and in
flight (2021-10-19 → 2025-10-20, era-aware slots, budget-capped and resumable).
Historical recovery beyond the live page goes through the Wayback
CDX API (PDFs back to Dec 2018) — a single public service, not a CDN: use CDX
only for gap discovery, politely, and enumerate slots deterministically via
this module's own URLs. Nothing in the module calls CDX today.

## What this is for

Point-in-time designations are the input a future availability feature would
read (see [the availability block](availability_block.md) for what is measured
today and where its boundary is). This pipeline banks the evidence now,
append-only, so that the question can be asked later against data that did not
exist when the question was first asked.
