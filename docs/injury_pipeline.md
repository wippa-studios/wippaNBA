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

Current state (2026-10-25): the store holds **503 snapshots across 306 report
dates**, 54,400 status events and 9,142 non-row lines (titles, footers and
anything that did not parse as a row — kept, never dropped), with ~19 MB of raw
PDFs under `Data/injury_raw/`. The **live capture loop is running** (3-hourly
slots) and the **historical backfill is in flight** (2021-10-19 → 2025-10-20,
era-aware slots, budget-capped and resumable). Historical recovery beyond the
live page goes through the Wayback CDX API (PDFs back to Dec 2018) — a single
public service, not a CDN: use CDX only for gap discovery, politely, and
enumerate slots deterministically via this module's own URLs. Nothing in the
module calls CDX today.

Two seasons were backfilled completely with
`scripts/backfill_injury_reports.py`, which walks **game dates** rather than
calendar days (a season has ~150 game dates and ~215 calendar days, so a range
capture spends a third of its budget recording 404s on days when no report can
exist):

| season | game dates | snapshots | requests | 404s | parse failures | status events |
|---|---|---|---|---|---|---|
| 2021-22 | 165 | 330 | 206 | 0 | 0 | 39,524 |
| 2020-21 | 140 | 140 | 140 | 0 | 0 | 12,071 |

The CDN serves both seasons directly — no Wayback round-trip was needed for
either. The 2021-22 era published a 17:00 and a 20:00 report each game day;
2020-21 published only the 17:00 slot (12:00 returns 403 for that era, so
afternoon games are uncovered there by construction).

**A limitation, recorded rather than papered over:** the 2018-19 layout does
not parse under the current cell grammar. A 2019-01-15 report yields 0 rows and
94 line-shaped rejections, all `missing required cell(s): Current Status` — the
era printed a different column set, and the header guard reports it instead of
mis-attributing statuses. 2020-02 onward parses cleanly (127 rows, 0 unparsed
for 2020-02-01), so the usable backfill window starts with the 2019-20 season
and the parser refuses, loudly, before that. Extending it to 2018-19 is a
parser job, not a capture job.

## What it measured (#37): below the bar

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
cannot contradict or confirm a smaller one; the honest reading is that both
seasons are too small, and that the question needs the full 2019-20 → 2024-25
span that the pipeline is still backfilling.

### The recorded 43-column bar (criterion 4)

The season protocol answers a better-powered question, but the issue's fourth
criterion asks for the literal one: the **43-column E14 frame against the
0.61831 baseline**. `scripts/issue37_full_frame.py` runs exactly the recorded
recipe on the full 13,199-game population and reproduces the bar to
**0.618307859** (recorded 0.618307119; drift +7.4e-07). Because E14 contains no
availability columns, the honest reference is E14 **plus the unadjusted
availability block**, not E14 itself:

| frame | log loss | delta vs E14 | 95% CI | \|z\| |
|---|---|---|---|---|
| E14 (43) | 0.618308 | — | — | — |
| E14 + availability (47) | 0.614508 | +0.003799 | [+0.002296, +0.005486] | 4.67 |
| E14 + injury (47) | 0.613690 | +0.004618 | [+0.002925, +0.006433] | 5.16 |
| **injury vs availability** | — | **+0.000818** | **[+0.000222, +0.001422]** | **2.67** |

The availability control reproduces the known +0.0038 at |z| 4.67, so the frame
and the estimator are on their recorded footing. The injury adjustment adds
+0.0008 nats on top of it, with a CI that excludes zero and **5 of 5 folds
better** — and |z| 2.67, below the pre-registered `|z| > 3` bar. That is the
closest this data has come to a signal, and it is still not a promotion. Two
caveats keep it honest: coverage is 2,108/13,199 games (16.0%), and 84% of rows
are byte-identical between the two 47-column frames, so the measured delta is a
diluted full-population average and the 16% of games with a report carry all of
the difference. This arm was declared after the season result was known, in
order to answer the criterion; it is descriptive and cannot promote.

Nothing was promoted, so the canonical model is unchanged: the
`availability_weights=` parameter exists, its default is bit-identical to the
pre-#37 block (pinned by a test), and no experiment record claims otherwise.

## What this is for

Point-in-time designations are the input the availability feature reads (see
[the availability block](availability_block.md) for what is measured today and
where its boundary is). The first use is measured above and comes back null
with an uninformative-interval caveat; the pipeline keeps banking evidence
append-only, so the question can be asked again on a sample that can actually
answer it.
