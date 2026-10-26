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
deviations from that mean, not edges; the holdout keeps most of their signs but
cannot resolve their size (home 10.5-12.5 goes from −12.47% to −12.29% with a
9.0% standard error; away 2.5-4.5 from −10.45% to −6.51%).

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
of about 5% in its largest cells and 10% in its smallest; it cannot rule out a
1–2% one.

## Decision

**No candidate. The family is closed.** The close does not show a positive
structural cell that survives its own multiplicity bar, and the strong negative
cells are the vig plus bucket noise. Nothing changes in the model.

One pattern was noticed but **not** tested here: the ATS line buckets are
unconditional on which side is favoured, so a "big underdog" interaction would
need its own registration. Testing it on 2025-26 now would be a post-hoc use of
the holdout this family already read. It is recorded as a hypothesis for a
future season, not as a result.

## Family 3: `bestprice_ml` — the profit objective at the best bookable price

Families 1 and 2 measure the market at **one book's** closing line, where the
two sides of a spread sum to −8.94% and every structural bet pays the vig. The
objective changed with #86: not "beat the market" but **"make a profit"** —
and those settle at different prices. Profit is earned at the best price a
bettor with accounts everywhere could actually get, which is not one book's
close.

### The data

OddsPortal's season archives publish, for every NBA game, the **highest
pre-game decimal moneyline odds across the bookmakers it tracks** — one number
per side, each possibly from a different book. `scripts/backfill_oddsportal.py`
crawls the 2015/16–2025/26 archives into `odds_ml_bestbooks`:

| check | result |
|---|---|
| rows promoted | **14,305** (regular season + play-in + playoffs) |
| regular-season games priced | **~12,600 / 12,704 (98.6%)** |
| score agreement with the game logs | **0 disagreements** in ~14,600 outcomes, except one: the documented NBA scoring correction (Bucks–Hornets, 2022-01-08) |
| moneyline overround 1/O1+1/O2 | **mean 1.006–1.020 by season** (~0.3–1.0% per side) vs ~4.5% single-book |
| duplicate listings | 78 (same game under two event URLs, agreeing odds) — deduped |
| date convention | headers render in the browser timezone after hydration; crawled with `America/New_York` so dates match the game logs |

The crawl survived three site behaviours worth recording: hash-only pagination
is silently ignored by the SPA, a reload boots back to page 1, and under load
the archive URL can serve the *current* season (detected via upcoming games
with no scores). The working cycle — click Next, then reload — and every
recovery are in `scripts/backfill_oddsportal.py`.

### Result: the price improvement is real; the edge is not

14 arms (favourite/underdog by de-vigged probability, 7 buckets), bar
**\|z\| > 3.0**. Population **13,106** priced regular-season games: discovery
**11,902**, holdout **1,204**.

| bet | n disc | ROI disc | \|z\| | n hold | ROI hold |
|---|---|---|---|---|---|
| ML underdog 0.80+ (best price) | 2,313 | **+20.41%** | **3.29** | 289 | −3.94% |
| back the underdog, no bucket (reference) | 11,902 | **+4.10%** | 2.39 | 1,204 | −1.62% |
| back the favourite, no bucket (reference) | 11,902 | −1.62% | 2.48 | 1,204 | −1.76% |
| ML favourite 0.80+ (best price) | 2,313 | −3.03% | 3.42 | 289 | −2.33% |

One discovery cell cleared the bar — big underdogs at **+20.41%** — and the
aggregate "back the underdog" reference carries a 95% interval of
**[+0.45%, +6.73%]**, entirely positive. Neither survives scrutiny:

1. **The holdout is negative.** The 0.80+ cell returned −3.94% on 289
   holdout bets. Per the registered rule: discovery-only, not a candidate.
2. **The edge is a snapshot artifact.** The best-of-both-sides price lets the
   two sides' quotes come from different books at different moments. Rows where
   1/O1 + 1/O2 < 1 (13.7% of games) are exactly those snapshots. On
   **coherent rows only**, the discovery underdog aggregate collapses from
   **+4.10% to −0.14%**, and the dog win rate (32.28%) lands on the implied
   break-even (32.10%). The market at coherent best-of-books prices is fair:
   dogs priced to the cent, favourites paying the ~1% vig.
3. **The both-sides "arbitrage" is the same artifact.** 1,791 rows admit a
   riskless both-sides profit averaging 1.45% per unit staked — real as a
   snapshot, unbookable at volume, and not a betting strategy.

**Family closed: no candidate.** The closing moneyline is efficient at
coherent best-of-books prices. What this family *did* establish is the honest
size of the price improvement: settling family 1's structural bets at best
available prices moves ROI by roughly +2 to +4 points — which is why
single-book cells sit at −3% to −5% while the same bets at best prices sit at
−1% to +4%. The remaining distance to profit is information, not price.

## Family 4: `model_bestprice` — the repo's model at the best bookable price

Family 3 closed the price route. The one route left with existing data is
information: the repo's own model, settled at best-of-books prices. The model
carries the project's one measured prediction edge (availability,
+0.00390 nats at \|z\| 4.78) and availability is the kind of news soft books
update late. Against that: the model sits **0.022 nats behind the market** on
log loss, and the same bet rule at single-book close returned **−8.23%**.

**Pre-registered rule (#88), no free parameters:** one bet per game, on the
side whose model probability exceeds the best-price implied probability;
settlement at the posted decimal odds. Population 6,530 games scored by the
walk-forward model AND priced (discovery 5,326, holdout 1,204). Four arms,
bar \|z\| > 3.0.

| arm | n disc | ROI disc | \|z\| | n hold | ROI hold |
|---|---|---|---|---|---|
| base model, all rows | 5,326 | +6.02% | 2.29 | 1,204 | −3.45% |
| base model, coherent rows | 4,481 | −2.33% | 1.08 | 923 | −4.00% |
| model + availability, all rows | 5,326 | +6.77% | 2.61 | 1,204 | −0.21% |
| model + availability, coherent rows | 4,481 | −1.46% | 0.68 | 923 | −2.01% |

**Family closed: no candidate.** No arm cleared the bar (max \|z\| 2.61), the
holdout is negative everywhere, and the discovery surplus shows the exact
incoherent-row signature of family 3: it lives in the snapshot rows and
vanishes on coherent prices, where the model's bets simply pay the vig. The
mechanism is visible in one number — the model backs the **underdog 68.2%**
of the time (its probabilities are flatter than the market's), so the
disagreement bets are mostly longshot bets, and longshots at coherent best
prices are exactly fair (family 3).

With families 1, 3 and 4 the map is closed for the data this repo holds:
the close is structurally null, best-of-books prices are fair once snapshot
artifacts are removed, and the model adds no information the price lacks.
The routes that remain open need data that does not exist here — player
props, live/in-play, alternate and second-half lines, or a multi-season
opening-line archive.

## Exploratory follow-ups (user-directed, not pre-registered)

Two questions asked outside the loop, recorded here with the same honesty
rule — every number measured, nothing promoted without a bar.

### What does a team that wins its 4th straight look like? (`EXP1`)

1,596 team-games enter on exactly a 3-game win streak. **58.3% win the 4th**
(home 66.5%, away 50.0%), and the market prices the spot almost exactly:
implied 57.7% vs actual 58.4% — betting every 3-streak team at best prices
returns +0.9% (z 0.30), −2.7% on coherent rows.

What separates the teams that extend from the ones that stop at three is the
**matchup, not the streak**: opponent Elo (d −0.59), opponent last-10 form
(d −0.42), own Elo (+0.41), home (+0.35). How the streak was built carries
essentially nothing — the quality of the three victims (d −0.005) and the
margin of the wins (d +0.25) are near-noise, and rest/travel/B2B are all
below 0.12. Subgroups: streaks die in hot-vs-hot games (opponent on a 2+
streak: 51.1% win) and on B2Bs (52.6%). One bright cell — 3-streak *mild
underdogs* (de-vigged 0.40-0.50) win 49.7% vs 44.9% implied, ROI +9.5% — is
one bucket of five with n=179; multiplicity applies, it is not evidence.

### Does betting the underdog make money when the Elo gap is under 100? (`EXP2`)

**No — and at the spread it is the worst measured spot in this repo.** With
the adjusted Elo gap under 100 (issue-#81 Elo pass), betting the ATS
underdog at −110 returns **−9.84% on discovery (z −7.48, n 5,249)** and
−3.18% on the holdout; |gap| < 50 is −9.9%/−9.4%. The ML at best-of-books
prices looks like +2.20% discovery / +0.43% holdout — the same snapshot
artifact family 3 exposed; coherent rows are −2.23%/+3.08%.

Why: when Elo says "even" but the market prices a favourite, the market is
usually right — it knows about injuries, rest, and matchups a single Elo
number cannot carry. In the 28% of parity games where Elo and the market
disagree on the favourite, the market's favourite covers at **+2.75%**
(n 1,622) while the dog loses 11.8%; in the 72% where they agree, both sides
pay the hold. Fading raw Elo is the profitable side of the disagreement —
consistent with the repo's oldest result, the market beats every model.

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
