# The search loop: a pre-registered family scan

The project has tested one hypothesis at a time and closed each with a null.
That is the right way to avoid p-hacking, but it is a slow way to cover the
space. This is the alternative: register a **family** of related arms before
computing any of them, report every cell, and set the significance bar from the
family size rather than from the best result.

The loop is deliberately conservative. A family is allowed to find nothing, and
most should. Its job is to make "we looked here and there is no price" a
measured statement instead of an absence of effort.

## Protocol

| element | rule |
|---|---|
| discovery | **2015-16 .. 2024-25** |
| holdout | **2025-26** |
| bar | \|z\| > max(3.0, Bonferroni for the family's arm count), paired bootstrap over bets, 2,000 resamples, seed 42 |
| candidate | discovery ROI > 0 **and** \|z\| above the bar |
| promotion | candidate **and** holdout ROI > 0 |
| reporting | every arm, including the negative ones |
| model change | none. A promoted cell becomes a forward paper-trading candidate, not a model change |

The bar is two-sided Bonferroni: a family with `m` arms needs
`norm.ppf(1 - 0.05 / (2m))`. For 42 arms that is **3.24**, not 1.96. A
discovery cell that clears it is still only a candidate; the holdout has to
agree in sign.

## The holdout was a snapshot, so it was replaced

The shipped `odds_2025-26` table stops on **2026-01-07** and prices 514 games.
It is one early scrape, not a close: on the 553 games both sources share, the
median absolute spread difference is 0.5 points, the 90th percentile is 3.0, and
the maximum is 9.0 (LAC–UTA on 2026-01-01 is stored as +22.5 / −7143 and closed
at +13.5 / −850). On **23 of 553** games the two sources disagree about which
side is the favourite. Using it as a holdout would understate the sample and
price bets at a number that is not the closing number.

`odds_2025-26_espn` is a separate full-season table backfilled from ESPN's core
odds API (`close` object; provider preference DraftKings → ESPN BET → any
non-live provider). The old table is **not modified**, so #74/#81 stay
reproducible; only `scripts/edge_search.py` swaps the holdout, through
`swap_holdout()`, which refuses duplicate keys and never blends the two sources.

Validation against the project's own logs:

| check | result |
|---|---|
| rows fetched | 1,321 (2025-10-21 .. 2026-06-13) |
| regular-season games priced | **1,224 / 1,225** |
| playoff games priced | **85 / 85** |
| duplicate game keys | 0 |
| missing regular-season game | MIL–WAS, 2025-10-22 |
| provider mix | DraftKings 1,060, ESPN BET 261 |
| agreement with ESPN's `pickcenter` on 1,062 shared rows | exact on spread and total |

The one missing game is a season-opener with no close on the core endpoint. It
is left missing rather than filled from the old snapshot: one holdout game is
immaterial, and mixing books inside a season is not.

## Family 1: `price_structure`

Registered in [issue #83](https://github.com/wippa-studios/Legit-NBA-Predictions/issues/83)
before the run.

**Question.** Where does the closing market show a structural bias, if anywhere?

42 arms, bar **\|z\| > 3.24**:

1. moneyline favourite and underdog by de-vigged favourite probability, seven
   buckets from 0.50 to 0.80+ (14 arms);
2. home and away ATS by `|L_eff|`, eight buckets from pick'em to 12.5+ (16 arms;
   **unconditional on which side is the favourite**);
3. over and under by closing total, five buckets (10 arms);
4. back the underdog ATS vs back the favourite ATS, side read from the sign of
   the home line, pick'em excluded (2 arms).

Settlement: ATS and totals at a flat −110, side-aware, a push returns the stake
and stays in `n`; moneyline at the posted price, vig included.

### Result: no arm clears the bar

Population **13,095** priced games: discovery **11,871**, holdout **1,224**.

| arm | n disc | ROI disc | \|z\| | n hold | ROI hold | \|z\| |
|---|---|---|---|---|---|---|
| away ATS, \|line\| 10.5-12.5 | 919 | **+3.52%** | 1.12 | 111 | **+3.19%** | 0.35 |
| home ATS, \|line\| 12.5+ | 859 | +1.42% | 0.44 | 207 | +1.45% | 0.22 |
| ML underdog, fav prob 0.80+ | 2,240 | +1.92% | 0.36 | 267 | −19.66% | 1.60 |
| home ATS, \|line\| 2.5-4.5 | 2,544 | +1.49% | 0.78 | 243 | −2.58% | 0.42 |
| **back the underdog ATS** | 11,640 | **−5.63%** | **6.45** | 1,224 | −2.99% | 1.09 |
| **ML favourite, fav prob 0.80+** | 2,240 | **−5.26%** | **6.21** | 267 | −4.20% | 1.80 |
| **away ATS, \|line\| 2.5-4.5** | 2,544 | **−10.45%** | **5.50** | 243 | −6.51% | 1.06 |
| **under, total 220-230** | 3,797 | **−7.27%** | **4.79** | 458 | −3.71% | 0.85 |
| **home ATS, \|line\| 10.5-12.5** | 919 | **−12.47%** | **3.96** | 111 | −12.29% | 1.36 |

Six of 42 arms are positive on discovery. The best is **+3.52%** at \|z\| 1.12 —
nowhere near 3.24 — and its holdout is +3.19% with a 95% interval of
**−14.0% to +20.4%** over 111 games. That is a point estimate, not evidence.

Fourteen arms are negative with \|z\| > 3, and none is positive. That asymmetry is
the market working: both sides of a spread lose the vig, so the family mean is
**−4.10%** and the median **−4.09%**. The large negative cells are bucket-level
deviations from that mean, not edges, and the holdout does not reproduce their
size (home 10.5-12.5 goes from −12.47% to −12.29% but with a 9.0% standard
error; away 2.5-4.5 from −10.45% to −6.51%).

### The two sides of the spread pay the vig

Over the 11,640 discovery games with a line:

| side | n | ROI | \|z\| | cover |
|---|---|---|---|---|
| underdog ATS | 11,640 | −5.63% | 6.45 | 49.38% |
| favourite ATS | 11,640 | −3.30% | 3.78 | 50.62% |
| both sides | 23,280 stakes | −8.94% | — | — |

The dog−favourite gap is −2.33pp with a **paired** standard error of 1.75pp
(\|z\| 1.33), so the apparent "fade the favourite" asymmetry is inside noise. The
one number that is not noise is the sum: **−8.94%** on the two sides, which is
the −110 hold. There is no side of the spread that avoids it.

### What the holdout could see

With 1,224 games the full-holdout standard error is about **2.7%**, so the
holdout can only confirm an edge larger than roughly 8% at \|z\| 3. A 9% bucket
(111 games) has a 9.0% standard error. The discovery side is better powered —
11,871 games, 0.9% standard error overall and 1.5–3.2% per bucket — and even
there no positive cell reaches 1.2σ. This family can rule out a structural edge
of 5%+ in these cells; it cannot rule out a 1–2% one.

## Decision

**No candidate. The family is closed.** The close does not show a positive
structural cell that survives its own multiplicity bar, and the strong negative
cells are the vig plus bucket noise. Nothing changes in the model.

One pattern was noticed but **not** tested here: the ATS line buckets are
unconditional on which side is favoured, so a "big underdog" interaction would
need its own registration. Testing it on 2025-26 now would be a post-hoc use of
the holdout this family already read. It is recorded as a hypothesis for a
future season, not as a result.

## Adding a family

1. Write `family_<name>(j)` in `scripts/edge_search.py`; return `(arms, meta)`
   with `meta["arms"] == len(arms)`.
2. Add it to `FAMILIES`, and add its arm list plus `m` to `prereg_record()`.
3. Append the pre-registration **before** the run:
   `python scripts/edge_search.py --family <name> --preregister`.
4. Run it, then write the family section here with every cell.
5. If a cell clears the bar, report it as a candidate; the holdout decides
   promotion, and the result is a paper-trading rule, never a model change.

## Files

- `scripts/edge_search.py` — harness, settlement, statistics, families
- `scripts/backfill_espn_close.py` — rebuilds `odds_2025-26_espn`
- `results/edge_search/price_structure.{json,md}` — the authoritative artifact
- `Tests/test_edge_search.py` — settlement, swap, bar, registry, holdout table
- Ledger: `E01-prereg-price_structure`, `E01-run-price_structure`
