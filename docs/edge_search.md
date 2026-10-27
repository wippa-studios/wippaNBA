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

## Family 5: `user_rules` — 13 fan-style rules, pre-registered before the run

The user supplied 13 rules built only from odds, schedule and form (#91) —
delivered *before* any number existed, which makes them a genuine
pre-registration. All 13 settle at the best-of-books moneyline (O−1/−1, no
pushes); coherent-row ROI is reported beside each as the family-3 artifact
diagnostic. Discovery 11,902 priced games, holdout 1,204; bar \|z\| > 3.0.

| rule | n disc | ROI disc | \|z\| | n hold | ROI hold | verdict |
|---|---|---|---|---|---|---|
| R1 home fav 7-3+ vs 3-7- away, away on B2B | 147 | −0.29% | 0.08 | 11 | +16.3% | no evidence |
| R2 road fav 8-2+ vs home 3-7- | 349 | −3.24% | 0.99 | 37 | +4.4% | no evidence |
| R3 home dog 6-4+ vs fav 4-6- | 120 | +0.78% | 0.07 | 8 | −41.8% | no evidence |
| R4 home 4+ straight home wins vs opp 4+ road losses | 215 | −2.37% | 0.61 | 24 | −2.2% | no evidence |
| R5 home fav ≤1.40, 7-3+, opp road form ≤2-8 | 362 | −2.75% | 1.19 | 50 | −7.9% | no evidence |
| R6 rested (2+d) 5-5+ team vs B2B opponent | 1,873 | +1.37% | 0.60 | 184 | −1.2% | no evidence |
| R7 home off 15+ pt loss vs opp road B2B | 204 | −1.82% | 0.23 | 27 | −6.8% | no evidence |
| R8 home opponent of a road favourite on its 4th game in 6 nights | 1,291 | **+9.36%** | **2.21** | 146 | **−4.62%** | no evidence |
| R9 home returning from a 4+ game road trip, 6-4+ | 341 | −4.54% | 1.06 | 41 | −10.1% | no evidence |
| R10 home opponent of a road-B2B team off a 2OT+ game | 91 | +16.97% | 1.57 | 9 | −43.9% | no evidence |
| R11 home on 3-game losing streak, season win% > .550 | 118 | −1.92% | 0.26 | 7 | −16.7% | no evidence |
| R12 home off a 20+ pt loss, 6-4+ | 197 | +1.85% | 0.35 | 27 | +3.9% | same sign, underpowered |
| R13 home dog that won 3 of the last 4 meetings | 607 | +2.50% | 0.46 | 94 | +10.8% | same sign, underpowered |

**Family closed: no candidate.** No arm reached \|z\| 3.0. The two largest
discovery surpluses are exactly the shape the loop has learned to distrust:
fatigue-flavoured (R8, R10) and holdout-negative. The holdout samples for the
rare-event rules (R3, R10, R11) are single digits — those cells can never
clear a bar, which is a finding in itself.

Two arms are *same sign, underpowered* and go on the watchlist with the
family-1 pair: **R12** (home team bounced back from a 20-point loss, 6-4+,
+1.9%/+3.9%) and **R13** (home dog with 3 of the last 4 head-to-head wins,
+2.5%/+10.8%, n 607/94). Neither is evidence; both are the kind of spot a
20-bet season would spend its budget on only after a forward season confirms.

The fatigue rules collectively confirm the parked family-2 probe: schedule
spots that look compelling (2OT road B2Bs, 4-in-6 favourites) show fat
discovery numbers and negative or single-digit holdouts — the market's
schedule pricing is not beatable from this data.

## Family 6: `user_rules_50` — the grid expanded to 50 rules

Family 5's 13 rules were expanded to a **systematic 50-rule grid** (#93): the
originals (R1–R13, unchanged and reproduced exactly) plus a schedule × form ×
venue/price expansion — opponent B2B/road-B2B/4-in-6 fades at plain, 5-5+ and
6-4+ form gates, road trips, post-2OT legs, letdown spots, price roles
(chalk ≤1.40/≤1.30, dogs ≥2.50), near-equal-form dogs, and rest edges.
**Disclosure:** family 5's numbers were known when the grid was drawn; the
expansion is systematic rather than hand-picked, and the holdout remains the
protection. Bar **\|z\| > 3.0** (Bonferroni floor for 50).

| rule | n disc | ROI disc | \|z\| | n hold | ROI hold | verdict |
|---|---|---|---|---|---|---|
| R1 home fav 7-3+ vs 3-7- away on B2B | 147 | -0.29% | 0.08 | 11 | +16.27% | no evidence |
| R2 road fav 8-2+ vs home 3-7- | 349 | -3.24% | 0.99 | 37 | +4.38% | no evidence |
| R3 home dog 6-4+ vs fav 4-6- | 120 | +0.78% | 0.07 | 8 | -41.75% | no evidence |
| R4 home 4+ straight home wins vs opp 4+ straight road losses | 215 | -2.37% | 0.61 | 24 | -2.21% | no evidence |
| R5 home fav <= 1.40, 7-3+, opp road form <= 2-8 | 362 | -2.75% | 1.19 | 50 | -7.94% | no evidence |
| R6 rested (2+d) 5-5+ team vs B2B opponent | 1,873 | +1.37% | 0.60 | 184 | -1.16% | no evidence |
| R7 home off 15+ pt loss vs opp road B2B | 204 | -1.82% | 0.23 | 27 | -6.78% | no evidence |
| R8 home opp of road fav on 4th game in 6 nights | 1,291 | +9.36% | 2.21 | 146 | -4.62% | no evidence |
| R9 home returning from 4+ game road trip, 6-4+ | 341 | -4.54% | 1.06 | 41 | -10.07% | no evidence |
| R10 home opp of road-B2B team off a 2OT+ game | 91 | +16.97% | 1.57 | 9 | -43.89% | no evidence |
| R11 home on 3-game losing streak, season win% > .550 | 118 | -1.92% | 0.26 | 7 | -16.71% | no evidence |
| R12 home off 20+ pt loss, 6-4+ | 197 | +1.85% | 0.35 | 27 | +3.89% | same sign, underpowered |
| R13 home dog that won 3 of last 4 meetings | 607 | +2.50% | 0.46 | 94 | +10.77% | same sign, underpowered |
| R14 opp B2B, backed rest>=2 (either venue) | 3,152 | +0.60% | 0.30 | 299 | -0.61% | no evidence |
| R15 opp B2B, backed rest>=2, backed 6-4+ | 813 | -1.22% | 0.50 | 63 | -11.78% | no evidence |
| R16 opp B2B, backed rest>=3 | 508 | -0.32% | 0.08 | 36 | +6.56% | no evidence |
| R17 opp away on road B2B (plain) | 1,580 | -0.93% | 0.38 | 148 | -12.65% | no evidence |
| R18 opp away on road B2B, home 5-5+ | 941 | -3.47% | 1.46 | 89 | -12.84% | no evidence |
| R19 opp away on road B2B, home 6-4+ | 653 | -3.56% | 1.26 | 60 | -16.57% | no evidence |
| R20 opp away on road B2B, home rest>=2 | 1,249 | -3.62% | 1.33 | 100 | -18.33% | no evidence |
| R21 opp on 4-in-6, backed rest>=2 (either venue) | 5,098 | -0.42% | 0.26 | 519 | -2.80% | no evidence |
| R22 opp on 4-in-6, backed 5-5+ (either venue) | 3,808 | -0.38% | 0.23 | 396 | -7.07% | no evidence |
| R23 opp on 4-in-6 playing away (plain) | 3,698 | +1.84% | 1.03 | 371 | -2.28% | no evidence |
| R24 opp away on road B2B and 4-in-6 | 945 | +2.14% | 0.65 | 89 | -26.17% | no evidence |
| R25 opp away on B2B that won last night; bet home | 727 | +3.26% | 0.84 | 57 | -8.33% | no evidence |
| R26 opp away on B2B that lost last night; bet home | 853 | -4.50% | 1.51 | 91 | -15.35% | no evidence |
| R27 home returning from 4+ road trip, opp rest>=2 | 731 | -4.97% | 1.32 | 73 | -3.33% | no evidence |
| R28 home returning from 4+ road trip (plain) | 933 | -3.71% | 1.07 | 92 | -7.20% | no evidence |
| R29 home win streak>=3 vs opp loss streak>=3 | 288 | -4.67% | 1.25 | 37 | +15.95% | no evidence |
| R30 any team win streak>=4 | 1,998 | -3.08% | 1.40 | 213 | -3.32% | no evidence |
| R31 home loss streak==3, season win%>.500 | 189 | +1.83% | 0.32 | 9 | +0.44% | same sign, underpowered |
| R32 both teams on 3+ game losing streaks; bet home | 226 | -4.91% | 0.75 | 26 | -43.65% | no evidence |
| R33 home off a 10+ pt loss (plain) | 2,929 | +0.03% | 0.01 | 331 | -4.37% | no evidence |
| R34 home off a 10+ pt loss, 5-5+ | 1,118 | +4.03% | 1.45 | 123 | -1.05% | no evidence |
| R35 opp off a 15+ pt win (letdown, either venue) | 3,309 | +2.86% | 1.01 | 403 | -12.93% | no evidence |
| R36 away fav on B2B; bet home (plain) | 752 | +1.90% | 0.36 | 81 | -12.65% | no evidence |
| R37 away fav on B2B, home 5-5+ | 221 | -9.98% | 1.15 | 30 | -21.00% | no evidence |
| R38 home dog vs opp B2B | 752 | +1.90% | 0.36 | 81 | -12.65% | no evidence |
| R39 home dog vs opp on 4-in-6 | 1,291 | +9.36% | 2.21 | 146 | -4.62% | no evidence |
| R40 dog >=2.50 vs opp B2B | 1,104 | -0.14% | 0.03 | 135 | +9.04% | no evidence |
| R41 favourite <=1.40 (plain) | 3,437 | -2.37% | 2.73 | 356 | -1.58% | no evidence |
| R42 favourite <=1.30 | 2,428 | -2.88% | 3.13 | 256 | -3.72% | no evidence |
| R43 favourite <=1.40 vs opp B2B | 939 | -1.10% | 0.71 | 80 | -7.59% | no evidence |
| R44 road underdogs (plain) | 7,509 | +6.56% | 2.82 | 705 | -1.51% | no evidence |
| R45 road underdogs 5-5+ | 3,622 | +6.99% | 2.09 | 325 | +0.12% | same sign, underpowered |
| R46 home underdogs 5-5+ | 1,593 | -3.57% | 1.09 | 189 | +5.67% | no evidence |
| R47 underdog when form10 differs by <=1 (either venue) | 1,463 | -2.91% | 0.86 | 174 | +9.39% | no evidence |
| R48 rest edge >=1 day, both 5-5+ | 1,209 | -0.01% | 0.00 | 106 | -6.60% | no evidence |
| R49 opp away on 3rd+ consecutive road game | 1,605 | -3.76% | 1.45 | 157 | -3.95% | no evidence |
| R50 opp away on 4+ consecutive road game | 665 | -3.69% | 0.96 | 64 | -6.42% | no evidence |

**Family closed: no candidate.** Fifty arms produced exactly one \|z\| > 3
result — and it is a **negative**: R42, the favourite priced ≤1.30 at best
prices, returns **−2.88% (z −3.13)** on discovery, −2.64% on coherent rows
(z −2.80) and −3.72% on the holdout. Heavy chalk is significantly overbet at
the best-of-books maximum; its dog-side mirror is +3.64% but noise-level
(z 0.84) because longshot variance dwarfs it.

The positive leads all die the established deaths:

- **R44 road dogs (plain): +6.56% (z 2.82)** → coherent rows **+0.24%**;
  holdout −1.51%. The family-3 artifact, again.
- **R45 road dogs 5-5+: +6.99% (z 2.09)** → coherent −2.93%; holdout +0.12%.
- **R8/R39 (home opponent of a road favourite on its 4th game in 6 nights):**
  +9.36% (z 2.21), **coherent +9.59% (z 2.07)** — the one cell that survives
  the artifact filter — and the holdout kills it (−4.62%, coherent −2.51%).
- **R10 (fade the road-B2B team off a 2OT game): +17.0%** on 91 discovery
  bets, **−43.9%** on 9 holdout bets.

Same-sign-both-splits and underpowered, joining the watchlist: **R45**
(+6.99%/+0.12%) and, weakly, **R31** (+1.83%/+0.44% on 9 holdout bets).

The structural lesson of the grid: fading tired teams — the most intuitive
schedule angle in basketball — is *negative* across every variant measured
(R17–R20, R26, R49, R50 all negative on discovery), because the market
already prices the fatigue. The only significant signal in 50 arms is that
**heavy favourites at the best-of-books maximum lose more than the vig** —
a pricing asymmetry worth remembering, but its mirror is too noisy to bet.

## Family 7: `lay_rules` — laying the favourite at its own price

A different bet structure (#95): **lay bets**, using the moneyline as lay
odds. Laying team X at decimal odds O wins **+1** if X loses and loses
**−(O−1)** if X wins. Pre-registered with the identity that frames it:
**lay ROI = −back ROI at the same odds**, so this family re-expresses the
measured favourite-overbet result (family 3) rather than discovering one.
The genuine questions: coherence, commission, and the sportsbook fallback.

12 arms; settlement +1 on the laid side losing, −(O−1) on it winning;
discovery 11,902 priced games, holdout 1,204; bar \|z\| > 3.0.

| arm | n disc | ROI disc | \|z\| | coh disc | n hold | ROI hold | verdict |
|---|---|---|---|---|---|---|---|
| L1: lay the favourite 0.50-0.55 | 1,668 | -1.49% | 0.64 | -0.30% | 143 | +14.82% | no evidence |
| L2: lay the favourite 0.55-0.60 | 1,820 | +1.99% | 1.00 | +1.22% | 183 | +1.32% | same sign, underpowered |
| L3: lay the favourite 0.60-0.65 | 1,868 | +3.11% | 1.69 | +4.44% | 157 | -0.82% | no evidence |
| L4: lay the favourite 0.65-0.70 | 1,648 | +1.26% | 0.76 | +0.83% | 164 | -1.35% | no evidence |
| L5: lay the favourite 0.70-0.75 | 1,417 | +1.31% | 0.83 | +1.09% | 146 | -4.88% | no evidence |
| L6: lay the favourite 0.75-0.80 | 1,168 | +1.17% | 0.70 | +1.79% | 122 | +1.21% | same sign, underpowered |
| L7: lay the favourite 0.80-1.01 | 2,313 | +3.03% | 3.42 | +2.81% | 289 | +2.33% | candidate (see mirror-identity caveat) |
| L8: lay the favourite priced <= 1.40 | 4,788 | +2.04% | 2.80 | +2.31% | 540 | +1.28% | same sign, underpowered |
| L9: lay the favourite priced <= 1.30 | 3,268 | +2.22% | 2.78 | +2.12% | 382 | +2.85% | same sign, underpowered |
| L10: lay the underdog (any price) | 4,393 | +0.10% | 0.04 | +0.76% | 499 | +1.77% | same sign, underpowered |
| L11: lay the home favourite priced <= 1.30 | 2,428 | +2.88% | 3.13 | +2.64% | 256 | +3.72% | candidate (see mirror-identity caveat) |
| L12: lay the road favourite priced <= 1.30 | 840 | +0.31% | 0.20 | +0.67% | 126 | +1.09% | same sign, underpowered |

Two arms meet the formal candidate rule:

- **L7 lay the favourite 0.80+: +3.03% (z 3.42)** discovery, +2.33% holdout —
  the exact mirror of family 3's back-favourite 0.80+ (−3.03%, z −3.42).
- **L11 lay the home favourite ≤1.30: +2.88% (z 3.13)** discovery, +3.72%
  holdout — the exact mirror of family 6's R42 (−2.88%, z −3.13).

Per the pre-registered caveat, these are **re-expressions of measured back
results, not new information** — the family's value is the three quantities
it adds:

1. **Coherence:** the favourite-overbet survives the coherent filter from the
   lay side — L7 coherent discovery is **+2.81% (z 3.09)**. Unlike the
   road-dog artifact (families 3/6), this signal is not a stale-snapshot
   mirage.
2. **Commission:** at a 5% exchange commission on winnings, the lays remain
   +1.4% to +2.2% per backer stake (L7: +2.23%; L9: +1.35%; L11: +2.01%).
3. **The sportsbook fallback:** on the same 3,268 games, *betting the dog at
   its own price* returns **+3.64%** vs the lay's +2.22% per backer stake —
   the stale-quote value sits on the dog side of the book. Per unit risked,
   though, the lay is far more capital-efficient: mean liability ≈ 0.17 vs
   1.00 staked.

**Family closed.** Candidates L7/L11 are recorded as re-expressions, not
promoted as new information. The actionable content: the measured
favourite-overbet is executable **only on an exchange** (commission
adjusted ~+2%), while at a regular sportsbook the same view is the dog bet,
which measures +3.6% at z 0.84 — suggestive, not significant.

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
