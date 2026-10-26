# Late scratches: what a designation is worth as the clock runs down

The injury store records a status for every listed player at fixed report
times.  This page answers the consumer's question — **how much should you
distrust a designation the further it is from tip?** — as a descriptive
census.  The outcome is actual participation (`MIN > 0`), never the last
report status: a player who stays Questionable and does not play is a scratch,
and a report that flips him to Out is only an event if he then fails to play.

Nothing here is promoted.  There is no decision rule, no z-test gate, and no
change to `STATUS_WEIGHTS` or the canonical model.  The pre-registration is
`I72-prereg-descriptive-late-scratch` in `results/experiments.jsonl`; the full
artifact is `results/issue72_late_scratch.json` / `.md`.

**The short version:** a Questionable is a coin flip at every cutoff,
including yesterday's report.  A Probable improves modestly as fresher
reports arrive.  Later reports mostly *reclassify* players — they do not
steadily raise the hazard on the players who remain Questionable.  And
`Available` is empirically riskier than `Probable`, the reverse of the
ordinal weights.

## How the panel is built

240,241 events from 2,046 snapshots are resolved to 98.70% of players and
joined to `build_games()` by an exact `(game_date, away@home)` key with the
event's resolved team required to be one of the two teams.  There is no fuzzy
fallback.  A game's tip time is the parsed `game_time` on its latest
exact-joined event label, because earlier reports can carry a stale
block-header time; the latest report is a schedule fact and cannot leak a
status.  Events after `tip − 15 minutes` are dropped, matching the availability
block's gate.

| step | events |
|---|---|
| injury events / snapshots | 240,241 / 2,046 |
| player resolution | 98.70% (3,131 unresolved) |
| exact `(date, matchup)` key match | 225,093 (93.7%) |
| … team not in the parsed matchup | 8,162 dropped |
| exact-join events with a team in the matchup | 216,931 |
| … unresolved player within the exact join | 2,312 dropped |
| … outside the 15-minute pre-tip gate | 55,375 dropped (25.8%) |
| **analysis panel** | **159,244 events, 7,624 games** |
| team+date sensitivity | 220,865 events, 77,573 player-games |
| player-games lost by exact join | 22 (0.028%) |

The exact join drops 3 games and 22 player-games against a
`(game_date, team)` sensitivity join, which is unambiguous because no team
plays twice on one date.  The 8,162 team-not-in-matchup events are a parser
artifact: on a continuation row the matchup cell can carry the previous game's
teams.  They are excluded rather than repaired, and the sensitivity join shows
the cost is negligible.  A further 99 games have only outside-gate events
(mostly matinees with no previous-day listing) and are not in the panel.

Of the analysis panel, 40.6% of events are previous-day listings.  The
store-wide share is 30.0%; the difference is the removal of post-tip same-day
rows, not a change in the store.

## Scratch rate by designation and nominal time to tip

5,115 games have a designation at all four cutoffs, so the columns compare
information arrival on the same games rather than different panels.  Cells are
the scratch rate and the number of player-designations; Wilson 95% bands are in
the artifact.

| designation | 6h | 3h | 1h | 30m |
|---|---|---|---|---|
| Out | 99.0% (n=32,481) | 99.4% (n=34,705) | 99.8% (n=37,888) | 99.8% (n=38,139) |
| Doubtful | 95.0% (n=1,138) | 96.4% (n=1,139) | 98.2% (n=926) | 98.2% (n=868) |
| **Questionable** | **50.2%** (n=7,679) | **51.0%** (n=7,756) | **50.0%** (n=6,110) | **50.1%** (n=5,780) |
| **Probable** | **12.8%** (n=2,745) | **11.2%** (n=2,883) | **9.6%** (n=2,417) | **9.9%** (n=2,267) |
| Available | 17.7% (n=808) | 22.2% (n=1,491) | 20.8% (n=3,610) | 19.8% (n=3,956) |

Questionable is flat at ~50% across every cutoff.  The 6h value is not a noisy
same-day reading: 91% of those statuses are previous-day listings and the
median actual report age is 23.5 hours.  Yesterday's Questionable is exactly as
likely to scratch as the last report's Questionable.

Probable is the designation that responds to fresher information: 12.8% at 6h,
11.2% at 3h, 9.6% at 1h, 9.9% at 30m.  The movement is small but in the
expected direction, and the 6h and 30m Wilson bands do not overlap.

Out and Doubtful are near-terminal at the last report (99.8% and 98.2%).  The
residual is small and can reflect a stale listing, a roster move, or a late
appearance; it is not treated as report error.

On all games with a 30m designation (the available panel, not the 5,115-game
common panel) the 30m rates are 50.0% [49.0%, 51.1%] for Questionable (n=8,706)
and 9.5% [8.5%, 10.5%] for Probable (n=3,363), so the common panel is not
driving the result.

### The nominal cutoff is not the report's age

Reports arrive at fixed clock times, not at fixed distances from tip, so the
actual age of the status used at each cutoff is not the cutoff itself.  The
median and p90 actual ages, with the share of statuses that are previous-day
listings (all statuses available at that cutoff; the cutoff table above is the
common four-cutoff panel):

| cutoff | median age | p90 age | previous-day share | same-day share |
|---|---|---|---|---|
| 6h | 23.5h | 26.0h | 91.4% | 8.6% |
| 3h | 5.0h | 23.5h | 35.8% | 64.2% |
| 1h | 2.5h | 3.5h | 5.5% | 94.5% |
| 30m | 2.5h | 3.0h | 5.4% | 94.6% |

Two consequences matter for reading the table:

- **The 1h and 30m columns are nearly the same report.**  With report slots at
  17:00 and 20:00 ET, a 19:30 tip has no report between 17:00 and tip, so both
  cutoffs select the 17:00 report, which is 2.5 hours old.  Across the common
  panel, 95.6% of 1h/30m pairs select the same label and 98.9% carry the same
  status.  The last slot is a *last-available-report* column, not a
  30-minute-old report, and the two columns are not independent evidence.
- **The 3h cutoff is bimodal.**  For an East Coast 19:30 tip, `tip − 3h` is
  16:30 and the latest report is yesterday's 20:00 (23.5 hours old); for a West
  Coast 22:00 tip, `tip − 3h` is 19:00 and today's 17:00 report is 5 hours old.
  Across all statuses at the 3h cutoff, 31.9% are 18–24 hours old and 63.0% are
  under 6 hours old.  The 3h rate is therefore a mixture of yesterday's and
  today's information, not a clean three-hour dose.

This is why every cutoff is reported with its actual-age distribution rather
than being read as a controlled dose-response.  The store has no role or
minute-restriction field, so it cannot say why a player is listed.

## Transitions: reclassification, not a rising hazard

Restricted to the common panel, 7,756 players were Questionable at 3h, all with
a 30m status; 2,883 were Probable at 3h, all with a 30m status.  The 30m status
is the latest report at `tip − 30m` (usually the 17:00 or 20:00 report, as
above).

**Questionable at 3h → 30m.**  70.8% received a later report.  By 30m, 15.5%
had flipped to Out, 15.6% to Available, 66.3% were still Questionable, and
2.5% moved to Probable or Doubtful.

| 30m status | n | scratch rate |
|---|---|---|
| Out | 1,202 | 100.0% |
| Doubtful | 65 | 100.0% |
| Questionable | 5,145 | 48.9% |
| Probable | 132 | 6.8% |
| Available | 1,212 | 13.4% |

The Q→Out flip is real but resolves only one in six Questionables.  The rest
either stay Questionable (and still scratch 48.9%) or are upgraded.  A Q that
*stays* Q on a later report scratches 48.0% (n=2,879); a Q with no newer
report scratches 50.0% (n=2,266).  The flatness is not an artifact of stale
reports — a fresh Questionable is still a coin flip.  (The fresh/unchanged
split is a descriptive decomposition of the same transition cells, added after
the pre-registration; it is not a separate test.)

**Probable at 3h → 30m.**  68.9% received a later report.  Only 1.1% flipped
to Out, while 27.3% were upgraded to Available; 69.8% stayed Probable and
scratched 9.9%.

| 30m status | n | scratch rate |
|---|---|---|
| Out | 33 | 100.0% |
| Doubtful | 7 | 100.0% |
| Questionable | 44 | 75.0% |
| Probable | 2,012 | 9.9% |
| Available | 787 | 6.4% |

A Probable that stays Probable on a fresh report scratches 8.5% (n=1,114); one
with no newer report scratches 11.6% (n=898).  That is the opposite pattern
from Questionable: for P, a fresh report is mildly reassuring, while for Q it
carries no information about the remaining 50%.

The pre-registered last slot is not hiding a late flip: the 30m status equals
the strict 15-minute final status for 99.996% of the 47,974 transition rows.

## `Available` is not the safest state

`STATUS_WEIGHTS` orders `Available` above `Probable`, so the model treats an
Available designation as more certain than a Probable.  The store says the
opposite at every cutoff:

| cutoff | Available | Probable |
|---|---|---|
| 3h | 22.2% [20.2%, 24.4%] (n=1,491) | 11.2% [10.1%, 12.4%] (n=2,883) |
| 1h | 20.8% [19.5%, 22.2%] (n=3,610) | 9.6% [8.5%, 10.8%] (n=2,417) |
| 30m | **19.8% [18.6%, 21.1%]** (n=3,956) | **9.9% [8.7%, 11.2%]** (n=2,267) |

Available players scratch about twice as often as Probables, and the 30m bands
do not overlap.  One plausible reading is that `Available` is a
*return-from-injury* or *minutes-restriction* flag as often as it is a clean
bill of health — the store has no field that would confirm it.  Either way the
ordinal mapping is empirically inverted on this sample.

This is a descriptive finding, not a licence to edit `STATUS_WEIGHTS`.  The
weights are part of the canonical model, the inversion was found after seeing
the data, and changing a model on a post-hoc descriptive cut is exactly the
kind of move the repository's pre-registration rule exists to prevent.  A
future issue can pre-register a recalibration arm and measure it on a forward
holdout; until then the canonical model is unchanged.

## Stability by season and era

Questionable is a coin flip in every season, not an artifact of one roster
cycle.  At 30m on the common panel:

| season | Q 30m | n | P 30m | n |
|---|---|---|---|---|
| 2019-20 (Jan–Mar) | 46.3% | 270 | 10.4% | 221 |
| 2020-21 | 51.4% | 477 | 12.1% | 239 |
| 2021-22 | 46.9% | 995 | 7.0% | 299 |
| 2022-23 | 51.8% | 1,028 | 11.1% | 404 |
| 2023-24 | 51.8% | 975 | 9.5% | 317 |
| 2024-25 | 52.8% | 1,079 | 11.1% | 422 |
| 2025-26 | 47.1% | 956 | 7.9% | 365 |

The Q range is 46.3–52.8%; the P range is 7.0–12.1%.  Neither has a monotone
trend, and the season-to-season spread is wider than the movement across
cutoffs within a season.  Splitting at the report-slot change:

| era | Q 6h | Q 3h | Q 1h | Q 30m | P 6h | P 3h | P 1h | P 30m |
|---|---|---|---|---|---|---|---|---|
| early (2019-20, 2020-21) | 49.6% | 50.5% | 49.7% | 49.5% | 14.5% | 13.0% | 11.5% | 11.3% |
| modern (2021-22 onward) | 50.3% | 51.1% | 50.1% | 50.2% | 12.4% | 10.7% | 9.1% | 9.5% |

The early era used more report slots (13:00/17:00/20:00 in 2019-20, then
11:00/14:00/17:00 in 2020-21); the modern era uses 17:00/20:00.  Q is flat in
both.  P is slightly lower in the modern era, but the era also changes which
report slot feeds each cutoff, so the difference is not a clean
same-day-versus-yesterday comparison.  The full per-season 3h/1h/30m table is
in `results/issue72_late_scratch.md`.

## What this changes

Nothing in the model.  The canonical log loss, `STATUS_WEIGHTS`, and the
availability block are unchanged, and no new feature is proposed.  The
descriptive facts that matter for future work are:

1. **A Questionable is not a soft Out.**  It is a genuine 50/50 even when the
   last report is fresh.  Any lineup-timing or starter model (#38) should treat
   a Questionable as a coin flip, not as a designation that resolves late.
2. **A late report mostly reclassifies.**  The Q→Out hazard is 15.5% by 30m,
   not a steady climb; the flat aggregate rate hides offsetting moves to Out
   and Available.
3. **`Available` deserves a pre-registered recalibration arm.**  Its scratch
   rate is roughly double a Probable's at every cutoff.  The honest next step
   is a forward-tested alternative mapping, not an edit on this sample.

## Caveats

- **Label-upload lag.**  The label is the report's nominal ET timestamp; the
  store records no separate upload time.  Availability is assumed at the label
  and the 15-minute pre-tip buffer is the only protection.
- **Era-1 slot sparsity.**  2019-20 reports at 13:00/17:00/20:00, 2020-21 at
  11:00/14:00/17:00, modern seasons at 17:00/20:00.  The nominal cutoff is not
  the actual report age, and the 1h/30m columns are nearly the same label.
- **Coverage.**  2019-20 is January–March 2020 only.  2015-16 through 2018-19
  have no injury store.  The 2026-27 capture (2,805 events) is out of scope
  because it is not in `build_games()`.
- **Previous-day listings.**  40.6% of analysis-panel events are previous-day
  listings; the 6h column is 91% such listings and is a test of yesterday's
  report, not a same-day six-hour-old report.
- **Parser artifacts.**  A small number of continuation rows carry a stale
  date or matchup.  The exact join drops them; the team+date sensitivity join
  bounds the cost at 22 player-games (0.028%).
- **Near-terminal residuals.**  Out and Doubtful still scratch 99.8% and
  98.2% at 30m.  The residual can be a stale listing, a roster move, or a late
  appearance; the store cannot distinguish them.
- **Wilson bands are descriptive.**  They are stability intervals for these
  counts, not sampling-error claims about a superpopulation, and they are not
  used to accept or reject any hypothesis.

## Reproduce

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/issue72_late_scratch.py
```

Writes `results/issue72_late_scratch.json` and `.md`.  The ledger records the
pre-registration, the executed runs (superseded development runs are annotated,
not deleted), and the summary in `results/experiments.jsonl`.
