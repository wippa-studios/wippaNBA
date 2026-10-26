# Layered Elo and recent form: does stacking two filters find a price?

**Question.** #74 measured the folk form cell alone: a home team 5-5 or better
over its last 10, hosting a team 2-8 or worse, returned **−5.28%** ATS over 965
games at |z| 1.71. If that cell is already priced, perhaps the *joint* state is
not: require a large pre-game Elo gap first, then require the higher-rated team
to be hot and the lower-rated team to be cold. The proposal was a team **6-5 or
better against 3-7 or worse**; the registered primary fixes both at the 10-game
window (6-4 or better / 3-7 or worse) so it is comparable with #74.

**Why this is a closing test, not a candidate hunt.** Two facts were measured
before registration and reported in [issue #81](https://github.com/wippa-studios/Legit-NBA-Predictions/issues/81):

1. The higher-Elo side is the market favourite in **95.2%** of games with a
   100-point gap. The Elo layer is close to "back the favourite", which returns
   **−3.66%** ATS over all 12,385 priced games at |z| 4.31. (The issue text
   first quoted −4.41% over 10,396 games; that is the subset where the Elo side
   and the favourite *agree*, not the favourite baseline. The correction does
   not change the conclusion.)
2. The Elo gap correlates **0.736** with the 10-game form gap. Elo updates on
   every result, so the form filter re-reads much of what Elo already absorbed.

The layers are therefore not independent information. This closes the form
family with a number rather than opening a new branch of it.

## Design

| element | definition |
|---|---|
| population | `build_games()` Regular Season joined to a closing line, 2015-16..2025-26, **12,385** games |
| Elo | `baselines.elo_rating_pass(k=10, home_advantage=70, season_reversion=0.75, same_day="batched", use_margin=True)`; pre-game ratings |
| Elo layer | gap = `home_elo + 70 − away_elo`; qualify at **\|gap\| ≥ 100**; back the higher-rated side |
| form layer | `recent_form_features` window 10, wins strictly before tip, no season reset; high side **≥ 6**, low side **≤ 3**; both sides **≥ 10 prior games** |
| prices | ATS at a flat **−110**, side-aware, push returns the stake; moneyline at the **posted** price, vig included |
| primary gate | ATS pooled, ROI > 0 **and** \|z\| > 3, 2,000 bootstrap resamples over bets, seed 42 |

## Result

| bet | n | ROI | SE | \|z\| | cover |
|---|---|---|---|---|---|
| **primary: high side ATS −110** | **1,886** | **−4.42%** | 2.19% | 2.01 | 50.03% |
| high side moneyline (descriptive) | 1,886 | **−4.06%** | 1.23% | 3.32 | 78.53% |
| high side at home, ATS | 1,147 | −5.81% | 2.78% | 2.09 | 50.13% |
| high side away, ATS | 739 | −2.25% | 3.48% | 0.65 | 51.18% |
| Elo layer alone (no form filter), ATS | 6,735 | −5.11% | — | 4.32 | 48.82% |
| back the favourite, no filter, ATS | 12,385 | −3.66% | 0.85% | 4.31 | 50.43% |

The form filter moves the Elo layer from **−5.11% to −4.42%** — about +0.7
points — and the cell remains negative. The cell-minus-complement contrast is
**+0.96pp** against a bootstrap SE of 2.56pp (|z| 0.38): the improvement is
inside noise. The moneyline is negative at |z| 3.32, so this is not merely an
underpowered ATS null.

### The Elo ladder

| gap | n | ATS ROI | \|z\| | ML ROI | \|z\| |
|---|---|---|---|---|---|
| ≥ 50 | 2,178 | −4.34% | 2.14 | −4.40% | 3.66 |
| ≥ 75 | 2,040 | −4.28% | 2.02 | −4.26% | 3.54 |
| **≥ 100** | **1,886** | **−4.42%** | **2.01** | **−4.06%** | **3.32** |
| ≥ 125 | 1,731 | −4.19% | 1.87 | −3.88% | 3.12 |
| ≥ 150 | 1,547 | −4.95% | 2.07 | −3.84% | 3.08 |

Every threshold returns the same sign and magnitude. The 100-point cut is one
standard deviation of the adjusted gap (sd 97), not an outcome-selected number.

### The 11-game reading

The literal "6-5 against 3-7" is an 11-game record. Run as a declared
sensitivity (high side ≥ 6 of 11, low side ≤ 3 of 11), it returns **−5.28%**
ATS (n 1,885, |z| 2.41) — worse than the 10-game primary. The 10-game version
is the better of the two and still loses.

### The four form sub-cells

Inside the Elo filter, without the history gate (n 1,893 for the hot/cold cell
vs 1,886 for the primary):

| high side | low side | n | ATS | \|z\| |
|---|---|---|---|---|
| hot (≥ 6) | cold (≤ 3) | 1,893 | −4.47% | 2.05 |
| hot (≥ 6) | not cold | 2,515 | **−3.78%** | 2.00 |
| not hot | cold (≤ 3) | 1,295 | −5.51% | 2.09 |
| not hot | not cold | 1,032 | **−9.01%** | 3.06 |

The ordering weakly supports the *high-side* condition — hot teams do better
than not-hot teams by +2.99pp (|z| 1.24) — and gives nothing to the cold-side
condition (+0.41pp, |z| 0.18). Requiring the opponent to be 3-7 or worse makes
the cell slightly worse than dropping that condition. All four are negative.

### Per season

Eight of eleven seasons are negative. The three that are not (2019-20 −0.15%,
2021-22 −0.26%, 2023-24 −0.17%) are effectively zero; 2024-25 is +0.42%. No
season carries a positive cell large enough to suggest a regime.

## Verdict

**Not promoted.** The layer loses at the price. The Elo layer selects the
favourite, and the form layer re-reads what Elo already absorbed; the stack does
not create a price. The moneyline is significantly negative, so the result is
not rescued by "underpowered ATS". The form family is closed: the best cell in
it is still −3.78% to −4.47%, and the vig is about −2.4%.

## Power

At n = 1,886 the bootstrap SE is **2.19%**, so the |z| > 3 gate needs an ROI
near **6.57%**. The test can rule out a 6.6% edge; it cannot rule out a 2% one.
A positive point estimate below the bar would not be evidence of a small edge.

## Files

- `scripts/issue81_layered_elo_form.py` — measurement and Markdown writer
- `results/issue81_layered_elo_form.json` — authoritative artifact
- `results/issue81_layered_elo_form.md` — generated summary
- `Tests/test_issue81_layered_elo_form.py` — 13 tests: side-aware settlement,
  push rule, causality, Elo direction, constant drift, artifact schema
- Ledger: `I81-prereg-layered-elo-form`, `I81-run-layered-elo-form`
