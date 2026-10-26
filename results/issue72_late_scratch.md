# Issue #72 — late-scratch analytics

Descriptive census of the injury store. The outcome is actual participation (`MIN > 0`), not the last report status. No decision rule, no promotion.

Pre-registration: `I72-prereg-descriptive-late-scratch`. Generated 2026-10-26T04:57:05.905197+00:00. Runtime 64.0s.

## Data and join audit

| quantity | value |
|---|---|
| injury events / snapshots | 240,241 / 2,046 |
| player resolution | 98.70% (3,131 unresolved) |
| exact (date, matchup) key match | 225,093 (93.7%) |
| … team not in the parsed matchup | 8,162 dropped |
| exact-join events with a team in the matchup | 216,931 |
| … unresolved player | 2,312 dropped |
| … outside the 15-minute pre-tip gate | 55,375 dropped (25.8%) |
| **analysis panel** | 159,244 events, 7,624 games |
| team+date sensitivity | 220,865 events, 77,573 player-games |
| player-games lost by exact join | 22 (0.028%) |
| games lost by exact join | 3 |
| games with only outside-gate events | 99 |
| previous-day events (analysis panel) | 40.6% |
| 2026-27 out-of-scope events | 2,805 |

## Status at each cutoff (common four-cutoff game panel)

5,115 games have a designation at all four cutoffs, so the columns are information arrival, not panel composition.

| designation | 6h | 3h | 1h | 30m |
|---|---|---|---|---|
| Out | 99.0% [98.8%, 99.1%] (n=32,481) | 99.4% [99.3%, 99.5%] (n=34,705) | 99.8% [99.8%, 99.9%] (n=37,888) | 99.8% [99.8%, 99.9%] (n=38,139) |
| Doubtful | 95.0% [93.6%, 96.1%] (n=1,138) | 96.4% [95.2%, 97.3%] (n=1,139) | 98.2% [97.1%, 98.9%] (n=926) | 98.2% [97.0%, 98.9%] (n=868) |
| Questionable | 50.2% [49.1%, 51.3%] (n=7,679) | 51.0% [49.9%, 52.1%] (n=7,756) | 50.0% [48.8%, 51.3%] (n=6,110) | 50.1% [48.8%, 51.4%] (n=5,780) |
| Probable | 12.8% [11.6%, 14.1%] (n=2,745) | 11.2% [10.1%, 12.4%] (n=2,883) | 9.6% [8.5%, 10.8%] (n=2,417) | 9.9% [8.7%, 11.2%] (n=2,267) |
| Available | 17.7% [15.2%, 20.5%] (n=808) | 22.2% [20.2%, 24.4%] (n=1,491) | 20.8% [19.5%, 22.2%] (n=3,610) | 19.8% [18.6%, 21.1%] (n=3,956) |

Median actual report age (hours) at each cutoff: 6h 23.5h (p90 26.0h, prev-day 91%); 3h 5.0h (p90 23.5h, prev-day 36%); 1h 2.5h (p90 3.5h, prev-day 6%); 30m 2.5h (p90 3.0h, prev-day 5%)

The nominal cutoff is not the report's actual age.  The 1h and 30m cutoffs select the same label for 95.6% of pairs and the same status for 98.9%, so they are not independent columns.  The 3h age distribution is bimodal: 31.9% of statuses are 18–24 hours old (yesterday's late report) and 63.0% are under 6 hours old (today's 17:00 report for a late tip).

## Questionable and Probable transitions, 3h → 30m

**Questionable** — 7,756 players with both labels; 70.8% received a later report. Flip to Out by 30m: 15.5%. Scratch among the 3h cohort: 51.0% [49.9%, 52.1%] (n=7,756); the 30m population rate is 50.1% [48.8%, 51.4%] (n=5,780).

| 30m status | n | scratch rate |
|---|---|---|
| Out | 1,202 | 100.0% [99.7%, 100.0%] (n=1,202) |
| Doubtful | 65 | 100.0% [94.4%, 100.0%] (n=65) |
| Questionable | 5,145 | 48.9% [47.5%, 50.3%] (n=5,145) |
| Probable | 132 | 6.8% [3.6%, 12.5%] (n=132) |
| Available | 1,212 | 13.4% [11.6%, 15.5%] (n=1,212) |
| *of which stays Questionable: fresh report* | 2,879 | 48.0% [46.2%, 49.9%] (n=2,879) |
| *of which stays Questionable: unchanged* | 2,266 | 50.0% [47.9%, 52.1%] (n=2,266) |

**Probable** — 2,883 players with both labels; 68.9% received a later report. Flip to Out by 30m: 1.1%. Scratch among the 3h cohort: 11.2% [10.1%, 12.4%] (n=2,883); the 30m population rate is 9.9% [8.7%, 11.2%] (n=2,267).

| 30m status | n | scratch rate |
|---|---|---|
| Out | 33 | 100.0% [89.6%, 100.0%] (n=33) |
| Doubtful | 7 | 100.0% [64.6%, 100.0%] (n=7) |
| Questionable | 44 | 75.0% [60.6%, 85.4%] (n=44) |
| Probable | 2,012 | 9.9% [8.7%, 11.3%] (n=2,012) |
| Available | 787 | 6.4% [4.9%, 8.3%] (n=787) |
| *of which stays Probable: fresh report* | 1,114 | 8.5% [7.0%, 10.3%] (n=1,114) |
| *of which stays Probable: unchanged* | 898 | 11.6% [9.7%, 13.8%] (n=898) |


The 30m status equals the strict 15-minute final status for 99.996% of the 47,974 transition rows, so the pre-registered last slot is not hiding a late flip.

On all games (available panel, not the common panel) the 30m Questionable and Probable rates are 50.0% [49.0%, 51.1%] (n=8,706) and 9.5% [8.5%, 10.5%] (n=3,363); the common panel is not driving the result.

## Calibration inversion

`STATUS_WEIGHTS` orders Available above Probable, but Available players scratch more often at every cutoff:

| cutoff | Available | Probable |
|---|---|---|
| 3h | 22.2% [20.2%, 24.4%] (n=1,491) | 11.2% [10.1%, 12.4%] (n=2,883) |
| 1h | 20.8% [19.5%, 22.2%] (n=3,610) | 9.6% [8.5%, 10.8%] (n=2,417) |
| 30m | 19.8% [18.6%, 21.1%] (n=3,956) | 9.9% [8.7%, 11.2%] (n=2,267) |

The ordinal mapping is inverted: the model treats Available as the safest pre-game state, but Available players scratch about twice as often as Probables at every cutoff, and the 30m Wilson bands do not overlap. The store has no role or minute-restriction field, so the mechanism is not identifiable here. This is a descriptive finding and a candidate for a future pre-registered recalibration, not a change to `STATUS_WEIGHTS`.

## Per season (common panel)

Questionable / Probable scratch rates at 3h, 1h and 30m. 2019-20 is Jan-Mar 2020 only.

| season | Q 3h | Q 1h | Q 30m | P 3h | P 1h | P 30m |
|---|---|---|---|---|---|---|
| 2019-20 | 50.6% [45.9%, 55.3%] (n=427) | 46.9% [41.2%, 52.6%] (n=290) | 46.3% [40.4%, 52.3%] (n=270) | 11.3% [8.2%, 15.5%] (n=291) | 10.9% [7.5%, 15.5%] (n=239) | 10.4% [7.0%, 15.1%] (n=221) |
| 2020-21 | 50.5% [46.2%, 54.7%] (n=527) | 51.4% [46.9%, 55.8%] (n=479) | 51.4% [46.9%, 55.8%] (n=477) | 14.9% [11.1%, 19.6%] (n=269) | 12.1% [8.5%, 16.8%] (n=240) | 12.1% [8.6%, 16.9%] (n=239) |
| 2021-22 | 48.9% [46.1%, 51.6%] (n=1,254) | 47.2% [44.2%, 50.2%] (n=1,058) | 46.9% [43.9%, 50.0%] (n=995) | 7.9% [5.5%, 11.2%] (n=354) | 6.9% [4.6%, 10.2%] (n=319) | 7.0% [4.6%, 10.5%] (n=299) |
| 2022-23 | 52.5% [49.8%, 55.2%] (n=1,341) | 51.7% [48.7%, 54.6%] (n=1,082) | 51.8% [48.8%, 54.9%] (n=1,028) | 11.4% [8.9%, 14.4%] (n=519) | 11.1% [8.4%, 14.4%] (n=434) | 11.1% [8.4%, 14.6%] (n=404) |
| 2023-24 | 51.8% [49.2%, 54.4%] (n=1,413) | 51.5% [48.5%, 54.5%] (n=1,069) | 51.8% [48.7%, 54.9%] (n=975) | 10.8% [8.3%, 13.9%] (n=465) | 8.6% [6.1%, 12.1%] (n=347) | 9.5% [6.7%, 13.2%] (n=317) |
| 2024-25 | 52.5% [50.0%, 55.1%] (n=1,435) | 52.1% [49.2%, 55.1%] (n=1,118) | 52.8% [49.8%, 55.8%] (n=1,079) | 11.7% [9.4%, 14.6%] (n=580) | 10.5% [8.0%, 13.7%] (n=455) | 11.1% [8.5%, 14.5%] (n=422) |
| 2025-26 | 49.3% [46.6%, 52.0%] (n=1,359) | 47.7% [44.7%, 50.8%] (n=1,014) | 47.1% [43.9%, 50.2%] (n=956) | 10.9% [8.2%, 14.3%] (n=405) | 7.6% [5.3%, 10.7%] (n=383) | 7.9% [5.6%, 11.2%] (n=365) |

## Era split (common panel)

Early = 2019-20 + 2020-21 (multi-slot report days); modern = 2021-22 onward (17:00/20:00).

| era | Q 3h | Q 1h | Q 30m | P 3h | P 1h | P 30m |
|---|---|---|---|---|---|---|
| early_2019_2021 | 50.5% [47.4%, 53.7%] (n=954) | 49.7% [46.2%, 53.2%] (n=769) | 49.5% [46.0%, 53.1%] (n=747) | 13.0% [10.5%, 16.1%] (n=560) | 11.5% [8.9%, 14.6%] (n=479) | 11.3% [8.7%, 14.5%] (n=460) |
| modern_2021_on | 51.1% [49.9%, 52.2%] (n=6,802) | 50.1% [48.8%, 51.4%] (n=5,341) | 50.2% [48.8%, 51.5%] (n=5,033) | 10.7% [9.5%, 12.0%] (n=2,323) | 9.1% [7.9%, 10.5%] (n=1,938) | 9.5% [8.2%, 11.0%] (n=1,807) |

## The 6h column is yesterday's report

Median actual age at the 6h cutoff: 23.5h; 91.4% of statuses are at least 12h old; same-day share 8.6%. It measures how much to trust yesterday's listing, not a same-day six-hour-old report.

| 6h designation | all | early era | modern era |
|---|---|---|---|
| Out | 99.0% [98.8%, 99.1%] (n=32,481) | 99.2% [98.9%, 99.4%] (n=5,035) | 98.9% [98.8%, 99.0%] (n=27,446) |
| Doubtful | 95.0% [93.6%, 96.1%] (n=1,138) | 96.0% [91.5%, 98.1%] (n=149) | 94.8% [93.3%, 96.1%] (n=989) |
| Questionable | 50.2% [49.1%, 51.3%] (n=7,679) | 49.6% [46.4%, 52.8%] (n=926) | 50.3% [49.1%, 51.5%] (n=6,753) |
| Probable | 12.8% [11.6%, 14.1%] (n=2,745) | 14.5% [11.8%, 17.7%] (n=538) | 12.4% [11.1%, 13.8%] (n=2,207) |
| Available | 17.7% [15.2%, 20.5%] (n=808) | 16.8% [11.5%, 23.8%] (n=143) | 17.9% [15.2%, 21.0%] (n=665) |

## Caveats

- Label-upload lag: the label is the report's nominal ET time; the store records no separate upload time, so the 15-minute buffer is the only protection.
- The 6h column is overwhelmingly yesterday's report, not a same-day report six hours old; its actual-age distribution is reported beside it.
- 2019-20 is Jan-Mar 2020 only; 2015-16 through 2018-19 have no injury store coverage.
- A small number of continuation rows carry a stale date or matchup; exact joining drops them and the team+date sensitivity join bounds the player-game cost.
- The analysis panel excludes 99 exact-join games (1.3%) whose only events fall outside the 15-minute gate; these are mostly matinees with no previous-day listing.
- Out and Doubtful are near-terminal (99.8% and 98.2% scratch at 30m); the residual is small and can reflect stale listings, roster moves or a late appearance.
- 40.6% of analysis-panel events are previous-day listings; the store-wide share is 30.0%. The difference is the exclusion of post-tip same-day rows, not a change in the store.
- Wilson bands are descriptive stability intervals, not sampling claims about a superpopulation.

Wilson 95% bands are descriptive stability intervals, not decision tests. Nothing here is promoted; the canonical model is unchanged.
