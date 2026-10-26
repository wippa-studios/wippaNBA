# The research links, assessed: what the literature says and what it does not

A standing file of `research_links.md` pages (~38 URLs found while researching
NBA prediction and betting) was triaged against this repository's measured
results.  This page quotes the findings that matter, links each to the paper,
and separates **what independently corroborates this project's conclusions**,
**what is genuinely untested here**, and **what repeats leakage classes this
repo already found and fixed**.

Every external number on this page is quoted from its source.  None of them was
verified by rerunning it here; quality caveats (sample, leakage risk, backtest
realism) are stated next to each.  The only numbers this project treats as
measured are the ones in `results/experiments.jsonl`.

---

## What the literature independently confirms

### The model loses to the market, and that is normal
- [**Exploiting sports-betting market using machine learning** — Hubáček,
  Šourek & Železný (2019)](https://doi.org/10.1016/j.ijforecast.2019.01.001),
  *International Journal of Forecasting* 35(2)
  ([author PDF](http://ida.felk.cvut.cz/zelezny/pubs/ijf.2019.pdf)).  On 9,093
  NBA games their neural models reach ~67% accuracy while the bookmaker's
  implied model reaches **69 ± 2.5%** — and a model that merely agrees with the
  bookmaker loses money by the margin.  This is the strongest single external
  confirmation that "prediction accuracy alone does not beat the book" — the
  same conclusion this repo reaches with a log-loss gap of 0.02268 nats
  (|z| 7.28) on 5,788 games.  Their response to that fact — training against
  correlation with the bookmaker's odds (a **decorrelation** loss term) plus
  portfolio-optimal bet sizing — is the one idea in the whole file this repo has
  not explicitly measured (see [what we might have missed](#what-we-might-have-missed)).
- [**Optimal sports betting strategies in practice** — Uhrín, Šourek, Hubáček
  & Železný (2021)](https://arxiv.org/abs/2107.08827), IMA J. Management Math.
  A deliberately strategy-level review: bet *sizing* (Kelly/portfolio), not
  prediction, is where the reproducible choices live.  Repo's selective-betting
  result (−8.23% over 4,828 bets, no threshold cell at |z| > 3) is consistent
  with its conclusion that no framing rescues an average model's moneyline bets.
- [**NBA Gambling Inefficiencies: A Second Look** — Compton & Sigler
  (2012)](https://thesportjournal.org/article/nba-gambling-inefficiencies-a-second-look/),
  *The Sport Journal*.  10,325 games (2000–08), closing lines: totals and
  spreads are **efficient overall**, and the famous "home underdog" edge that
  showed +69% in 2000–04 **fades to nothing by 2004–08** (log-LR 0.08).  The
  repo's finding that no margin/threshold cell reaches |z| > 3 is the same
  pattern on a later sample.
- [**Sentiment Bias and Asset Prices: Evidence from NBA Betting**
  — Feddersen, Humphreys & Soebbing (2018)](https://metatoc.com/papers/79725-sentiment-bias-in-national-basketball-association-betting),
  *Journal of Sports Economics* (32,000+ games, 1981–2012).  Bookmakers shade
  prices against popular home teams, but the shading **"does not translate into
  higher returns for bettors"**.  Popularity is in the price.

### External, leak-aware pipelines land at exactly this project's ceiling
- [**Uncertainty-aware machine learning for NBA forecasting in digital betting
  markets** (2024/25)](https://www.mdpi.com/2078-2489/17/1/56), MDPI
  *Information* 17(1):56.  An independent pre-game pipeline with explicit
  chronological splits and no-leakage discipline reports **log-loss
  0.597 ± 0.018, Brier 0.205 ± 0.008, accuracy 0.679 ± 0.014, AUC 0.744** — and
  an honest betting backtest of **ROI −7.0% ± 18.3%** under 0.3×Kelly.  Two
  numbers are worth pausing on: their *model's* log loss (~0.597) is essentially
  this repo's *market* log loss (0.59766), and their betting loss matches this
  repo's −8.23%.  (Their equity-curve figure text is inconsistent with their own
  table — cite the table.)
- [**ML for sports betting: model selection by accuracy or calibration?**
  — Walsh & Joshi (2024)](https://arxiv.org/html/2303.06021v4).  Argues that
  **calibration, not raw accuracy, is what a bettor needs** — the repo's
  log-loss (strictly proper) selection is already the calibrated choice.
- [**Wages of wins** (2017)](https://arxiv.org/abs/1702.05982) and
  [**Luck is Hard to Beat** (2017)](https://arxiv.org/abs/1706.02447): "high
  accuracy does not automatically equal high payout" and a predictability
  ceiling across 198 leagues — useful upper-bound framing for why 65–69% is a
  ceiling, not a floor of opportunity.

---

## What we might have missed (cheaply testable)

These are the only items in the file that are both **new** and **testable in
the existing harness** (the canonical 26-feature walk-forward + the
market-anchored stack used for the availability block).

1. **Time-zone / travel-quantity features.**
   [Langborg (2023), *The Impact that Travel Across Multiple Time Zones has on
   Win Rate in the NBA and the Relationship to Point Spreads in Sports
   Betting*](https://dataspace.princeton.edu/handle/88435/dsp01qz20sw77d)
   (Princeton ECON senior thesis):
   teams crossing **2+ time zones win ~41%** and oddsmakers fail to centre
   those games at 50%.  The 26 canonical features include rest days and
   back-to-back flags but no distance or time-zone-crossing quantity.  Smallest
   test: add (a) |timezone offset| home−away and (b) away-team zones crossed to
   the canonical logistic; same walk-forward and market-stack protocol as the
   availability measurement.  **Status (issue #56): measured null.**  The two
   columns (`tz_offset_diff`, `away_tz_crossed`, built in `rest_and_travel`)
   take the canonical 26 → 28 and read **+0.000035 nats, CI [−0.000228,
   +0.000272], |z| 0.28, better in 2/5 folds** — interval includes zero, so
   the Langborg effect is not visible on this frame's pre-game stylisation,
   and stage 2 (the market stack) was correctly not run.  [results/issue56_tz_travel.json](../results/issue56_tz_travel.json).
2. **Spread-cover as the betting objective.**
   [**Peterson (2020)**, CMU capstone](https://www.stat.cmu.edu/capstoneresearch/spring2020/495-Peterson-Poster.pdf),
   building on [Polson & Stern (2015), *The implied volatility of a sports
   game*](https://doi.org/10.1515/jqas-2014-0095): independently reproduces
   this repo's two
   nulls (margin-threshold betting only lifts win rate to 52–56%; totals cannot
   beat the books), then tries the one objective the repo has not: betting
   **margin > spread** under margin = Normal(spread, σ_game) with a per-game
   "implied volatility".  Smallest test: calibrated P(cover) from the repo's
   OOF margin predictions against posted and alternate (±7.5) lines.
3. **Variance-conditioned staking.**
   From the MDPI paper above: gate/size bets by **predictive uncertainty**, not
   only by edge over the posted price.  Smallest test: take the existing
   4,828-bet P&L and add a variance dimension (5-seed MLP dispersion or a
   logistic bootstrap); compare ROI vs the −8.23% baseline.  The prior is
   against it — the MDPI paper's own table shows uncertainty conditioning does
   not rescue profitability.  **Status (issue #58): measured NULL.**  Per-game
   dispersion from a 50-resample residual bootstrap of the walk-forward
   logistic is **uninformative** about where the model is wrong (corr with
   wrong-bets **−0.05**; realized win rates flat across variance terciles
   0.391 / 0.384 / 0.402), so no gate can help: restricting to low-dispersion
   bets makes the rule *worse* (gate-25% −15.3% vs the −8.71% breakeven
   baseline), the 75% gate wobbles +1.6pp, and all gate deltas sit at
   |z| ≤ 0.6.  Sized rules lose too (1−var −10.4%; 0.3·Kelly(1−var) −19.8% —
   Kelly amplifies a losing rule).  [results/issue58_variance_staking.json](../results/issue58_variance_staking.json).
4. **Early-season market drift (weakest claim).**
   Only an affiliate blog
   ([SportsBoom](https://www.sportsboom.us/betting/nba-betting-strategies))
   claims lines are systematically mispriced early each season (citing
   [Girdner, Davis, Fodor & Kirch (2013), *Early Season NBA Over/Under Bias*,
   J. Prediction Markets](https://ircommons.uwf.edu/esploro/outputs/journalArticle/Early-season-NBA-overunder-bias/99380090347306600) —
   a *totals*-market result, week-1 unders at 58.2%, n≈60–65).  Smallest test:
   a `games_into_season` split of the existing market comparison; ~10 lines.
   **Status (issue #59): measured FLAT — claim closed.**  Splitting the
   5,788-game market comparison by min(home_n, away_n) ≤ 15, the market's log
   loss is directionally worse early (+0.0185 nats early minus late: 0.60813
   vs 0.58984) but **|z| 1.77, CI [−0.0029, +0.0384]** — below the bar and
   crossing zero; the closest cutoff (≤ 20 games) is |z| 2.99, still under.
   The bet-rule split runs the *opposite* way (early −6.1% vs late −10.9% on
   the exact recorded breakeven rule), so there is no early-season edge to
   target.  [results/issue59_early_season.json](../results/issue59_early_season.json).

Runner-up (not promoted): **momentum streaks** —
[Byrnes & Farinella (2016)](https://thesportjournal.org/article/the-effect-of-momentum-on-the-nba-point-spread-market/)
claim betting *with* ≥4-game streaks wins 56.5%, directly contradicting
Paul & Weinbach (2005; bet *against* streaks).  The repo's recency weighting
is already null; two contradictory claims do not justify re-opening it without
a streak-count feature test.  **Status (issue #60, revised parity): measured
null on the 2025-26 season only.**  Pre-game entering-streak brackets
(>=2/3/4/6, home and away; season-reset) appended to the canonical 26 read
**+0.000260 nats, CI [−0.001431, +0.002066], |z| 0.29** on the 1,225-game
2025-26 holdout (train = everything before 2025-10-21), with sensitivity arms
(home-only, away-only, loss-streaks) all below |z| 1.0.  Since the identical
holdout resolves the known availability effect at **+0.009247 nats, |z| 4.93**,
this null is informative, not underpowered: if pre-game streaks carried
availability-sized signal, the season that would be bet would have shown it.
[results/issue60_momentum.json](../results/issue60_momentum.json).  **Follow-on
(issue #60 thread, 2026-10-26): the same question re-asked as recent-FORM win
counts is also null.**  Full cumulative ladders for wins in the last 5 and last
10 games (both sides; every bracket introduced in the thread) read **form5
−0.000620 nats (|z| 0.64)** and **form10 −0.000764 nats (|z| 0.63)** on the
same 1,225-game 2025-26 holdout, with the availability positive control
reproducing at **+0.009247, |z| 4.93** and the fitted bracket ladders flopping
sign level-to-level (no coherent recent-form effect).  "A team that is 4-from-5
beats a team that is 1-from-5": not measurably, at any bracket of either
window, on the season that would be bet.
[results/issue60b_recent_form.json](../results/issue60b_recent_form.json).

---

## Sources that repeat leakage classes this repo already found

These report impressive numbers **because they predict after the fact**.  Their
methodology is the exact class of bug this repo diagnosed in the 68.9%-leak
headline and the player-Elo leak:

| source | reported number | leak class |
|---|---|---|
| [**He & Choi (2025)**, Scientific Reports](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12357926/) | 83.3% accuracy / AUC 0.921 on 3,690 games | features are the **same-game box score** (FG, 3P, rebounds) with a random 80/20 split — retroactive prediction |
| [**Houde (2022)**, Bryant honors thesis](https://digitalcommons.bryant.edu/honors_data_science/1/) | 65.1% via Gaussian NB | season-aggregate team-stat matrix — the season-to-date leak class |
| [**Ouyang et al. (2024)**, PLOS ONE](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11265715/) | 72–93% in-game | quarter-level box stats + time-leaky 10-fold CV; the high numbers are *in-game*, never pre-game |
| [**Rios et al. (2025)**, arXiv LSTM](https://arxiv.org/abs/2512.08591v1) | 72.35% on 9,840-game sequences | long-lag outcome-derived sequences; verify their temporal split before citing; also contradicts the repo's "complex models lose" result without verifiable baselines |
| [**Shaviro (2025)**, CS224R deep-RL](https://cs224r.stanford.edu/spring_2025/projects/pdfs/CS_224r_Final_Report%20(3)1.pdf) | >30% ROI claimed | 2-season holdout, single run, no vig/execution detail — motivation, not evidence (features do include rest/travel/availability, which corroborates the repo's availability mechanism) |

---

## The rest of the file, in one table

| link | record | verdict |
|---|---|---|
| [Sun (2024), Princeton thesis](https://theses-dissertations.princeton.edu/items/a00f44d5-9fab-481d-a0c8-140624777bb4) | ridge regression ~63% win accuracy + Kelly-criterion betting framework | Already covered; same genre |
| [Ji (2024), Princeton ORFE thesis](https://dataspace.princeton.edu/handle/88435/dsp01s7526g767) | neural nets on historical + player + odds data; betting-strategy analysis | Already covered |
| [uudmae (2017), CS229](https://cs229.stanford.edu/proj2017/final-reports/5231214.pdf) | ~63% accuracy, no market/backtest | Course project; nothing above canonical |
| [Bucquet & Sarukkai (2018), CS229](https://cs229.stanford.edu/proj2018/poster/3.pdf) | regression on combined points total; totals betting | Totals already lose in-repo |
| [Tsagris, Adam, Pantatosakis (2024), Springer Discover AI](https://doaj.org/article/cc41e46aaf474ab486da273e52c5597e) | NBA outcome from half-time stats; same-/next-season generalization | In-game; not a pre-game edge |
| [Rodrigues & Pires (2025), IJCSS](https://reference-global.com/article/10.2478/ijcss-2025-0006) | 20 seasons; best 64.1% accuracy using rankings + fatigue features | Corroborates the ceiling and the fatigue feature family |
| [Uhrín (2018), CTU master's thesis](https://dspace.cvut.cz/handle/10467/76604) | Kelly wealth allocation + betting stress-testing | Predecessor of the 2021 review; sizing covered |
| [Beggy et al. (2023), Wharton](https://wsb.wharton.upenn.edu/wp-content/uploads/2023/05/Beggy_2023__Betting_Kelly.pdf) | full Kelly bankrupts in 100% of simulations; partial-Kelly recommendation | Confirmatory; the ~80%/yr claim is a red-flag example, not a target |
| [Paul & Weinbach (2008), Int. J. Sport Finance](https://ideas.repec.org/a/jsf/intjsf/v3y2008i3p137-145.html) | favorites overbet; totals overs dominate; no profit betting against sentiment (unlike NFL) | Efficient close; consistent with repo |
| [Sciberras (2023), Univ. of Malta](https://www.um.edu.mt/library/oar/handle/123456789/115275) | player-props RNN/LSTM/transformer; cannot beat the book | Different bet type; consistent |
| [Luo & Krishnamurthy (2023), graph NNs](https://arxiv.org/pdf/2303.16741) | GATv2-GCN player-performance prediction | Player-level signal already null in-repo |
| [Galekwa et al. (2024), systematic review](https://arxiv.org/pdf/2410.21484) | survey of ML in sports betting, no new experiments | Background |
| [Park (2014), RMIT PhD](https://research-repository.rmit.edu.au/articles/thesis/The_prediction_of_outcomes_in_the_National_Basketball_Association/27590031) | pre-betting-era NBA outcomes statistics | Historical context |
| [Hou et al. (2025)](https://www.researchgate.net/publication/398545691_Research_on_Predicting_the_Probability_of_Winning_Basketball_Games_Based_on_Machine_Learning_Models) | ML win-probability models; full text unverifiable (ResearchGate) | Low value unverified |

**Dead or blocked links** (no content recovered): the ECML/PKDD MLSA'17
workshop paper [18], the fitpublishing node [26], and the ictprojects Malta
abstract [36] are dead (404); the Inderscience article [10] and eScholarship
item [14] could not be fetched.  Duplicates in the source file: the Princeton
Sun thesis appears twice, the Hubáček 2019 paper three times (ResearchGate
page + CVUT copies), and the graph-NN arXiv paper twice.

---

## How this was assessed

The file was split in two and each half assessed against this repository's
measured results (canonical 0.6157 log loss; market 0.59766 vs 0.62034,
|z| 7.28; bet rule −8.23%; availability +0.00390, |z| 4.78, priced out at the
close; injury multiplier +0.000818, |z| 2.67, not promoted; the two
documented leak classes).  Buckets: **A** already known/covered, **B** new but
not actionable, **C** genuinely missed / worth acting on.  The working notes
live at `/tmp/opencode/research_links_A.md` and `research_links_B.md`.