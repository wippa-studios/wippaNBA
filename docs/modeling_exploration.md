# What else could be modelled on this NBA dataset

**Repo:** `/home/Joel/.cache/nba-fix-worktree` (branch `improve-model-v2`)
**Data:** `NBA_DATA_ROOT=/home/Joel/Desktop/Legit_NBA_Predictions/data/nba`
**Built:** 2026-10-18

This is a proposal document, not a result document. It exists because the framing question
is **"what information exists that no feature currently reads"**, not "what algorithm scores
better" — the closing market beats the model by 0.026 nats at |z| = 7.36, roughly seven times
the largest improvement ever found here, so algorithmic headroom is not the binding
constraint.

I did not modify any project code. Most claims below are backed by measurements I ran against
the real data; where I did not measure, I say so and give the command instead.

---

## 0. Headline

I built nine candidate blocks out of the tables nobody has read, measured all of them against
the current 59-column frame on the same five walk-forward splits (6,600 out-of-fold rows),
and **every one is a null or worse.**

| block | what it uses | cols added | OOF log loss | gain vs base | z |
|---|---|---|---|---|---|
| *(base frame, 59 cols)* | — | — | 0.61915 | — | — |
| actual travel | origin → destination | 3 | 0.61890 | **+0.00026** | +1.02 |
| margin volatility | trailing sd of margin | 2 | 0.61889 | **+0.00026** | +1.19 |
| matchup deviation | this pairing, not team strength | 3 | 0.61911 | +0.00004 | +0.39 |
| momentum (2 vs 10) | second difference of margin | 4 | 0.61919 | −0.00004 | −0.06 |
| road-trip length | consecutive away games | 2 | 0.61924 | −0.00009 | −0.68 |
| never-read team columns | transition / 3PA mix / self-creation | 28 | 0.61932 | −0.00017 | −0.22 |
| shot-chart shot mix | 1.1M shots, 49% game coverage | 14 | 0.61952 | −0.00037 | −0.59 |
| league era / season index | league 3PA rate, season counter | 4 | 0.62043 | **−0.00128** | −2.25 |
| prior-season play-type PPP | Synergy play types, crossed | 22 | 0.62084 | **−0.00169** | −2.32 |
| prior-season play-type PPP | + possession mix, full block | 66 | 0.62763 | **−0.00847** | −4.96 |

This is not a failure of imagination — several of these blocks pass the duplicate test
cleanly (correlation 0.006–0.17 with the features that already carry the signal) and are
still null. That is the finding. See §5.

Two blocks are *actively harmful* and one of them is the most interesting object in the whole
dataset, because its information is provably **not** redundant and it still loses. That
combination is the most useful thing I found, and it is the subject of §3.4.

### Three documentation/measurement errors found along the way

These are in files I was asked to read, and they change what should be tried next.

1. **`results/feature_block_ablation.json` reports a leaked result as a win.** Its
   `player_elo` row reads `delta −0.006451, z 5.18, "earns its place"` and its
   `reference_metrics.log_loss` is 0.61229. Both are the number the `nba_data` module
   docstring explicitly identifies as *the leak*: "An earlier version aggregated over
   `elo_long` directly … it measured −0.00645 nats at |z| 5.18." The fix (`_asof_roster`,
   `roster_mode`) is uncommitted in the working tree and was last written at 00:22; the JSON
   was generated at 00:07. **The 0.61229 reference figure and the `player_elo` verdict in that
   file are stale and must be regenerated before anyone reads them.**
   By contrast the honest base I measured on the current code is **0.61915**, and 0.61835 on
   the frame without the Elo columns.

2. **The "logistic + PCA(20) beats plain logistic" result does not reproduce on a matched
   feature set.** `results/experiments.jsonl` logs E00 at 0.61835 (53 columns) and E41 at
   0.61605 (59 columns), implying PCA(20) bought 0.0023 nats. Re-run on the current code:

   | frame | plain logistic | PCA(20) | gain |
   |---|---|---|---|
   | 59 cols (current, honest Elo) | 0.61915 | 0.61920 | −0.00005 |
   | 53 cols (no player-Elo) | **0.61835** | 0.61919 | −0.00083 |

   The 53-column plain number reproduces E00 to five decimals, so the frame is right. On
   that frame PCA(20) **loses**. The 6 extra columns in E41's frame are exactly the
   `player_elo` block, which at that time was the leaking version. So the PCA result is
   best read as PCA recovering part of a leak by projecting it onto a component, not as a
   real effect of dimensionality reduction. This matters because it is the one place a
   "regularisation beats the baseline" claim is currently being carried forward.
   (Related: the honest Elo columns are worth about +0.0008 in the full frame, not −0.0065.)

3. **`summary/` and `pbp/` do not exist.** The repository's own `data/nba/README.txt`
   describes `summary/` as "per-game inactives, officials, line scores" and `pbp/` as
   in-game data, and `scripts/download_nba_data.py` lists both stages. Neither directory was
   ever written and neither string appears in `download.log`. **Referee crews and confirmed
   inactive lists are absent from this dataset**, not merely unread.

---

## 1. What the data actually contains

Everything below was read off the CSVs, not inferred. `13,199` games,
`2015-10-27 .. 2026-04-12`, 11 seasons, regular season only, home win rate `0.5645`.

### 1.1 What the feature layer already reads

`src/DataProviders/nba_data.py` produces a **59-column** frame:

- `rolling_team_features` — per side, EWM(halflife 10) of `margin`, `NET_RATING`,
  `OFF_RATING`, `DEF_RATING`, `PACE`, `TS_PCT`, `EFG_PCT`, `TM_TOV_PCT`, `FTA_RATE`,
  `OREB_PCT`, `DREB_PCT`, `OPP_EFG_PCT`, `OPP_FTA_RATE`, `OPP_TOV_PCT`, `OPP_OREB_PCT`,
  plus `games_played`.
- `rest_and_travel` — `rest_days`, `back_to_back`, `in_3_of_4`, `travel_miles`.
- `opponent_adjusted_features` — `ewm_adj_margin`, `ewm_sos`, home/away split, `home_edge`.
- `player_elo_features` — `elo_available`, `elo_mean`, `elo_top5`, per side.

Two structural facts about this frame that shape everything below:

- **There is no calendar feature in it.** Nothing encodes the date or the season. The closest
  proxy is `games_played`, and my measurement shows `corr(season_index, games_played) = 0.996`.
  So any "era" idea is, mechanically, an era-idea wearing a `games_played` costume.
- **Every feature describes a team against nobody.** The only opponent information is the
  four `OPP_*` league-average columns and `ewm_sos`. `ewm_adj_margin` differences the same
  margins against the opponent's own trailing margin, which is why it correlates 0.98 with
  `ewm_margin` and adds nothing.

### 1.2 Columns that exist and nobody reads

| table | unread columns | coverage |
|---|---|---|
| `team_gamelogs/scoring_*.csv` | `PCT_FGA_2PT`, `PCT_FGA_3PT`, `PCT_PTS_2PT`, `PCT_PTS_3PT`, `PCT_PTS_FB`, `PCT_PTS_PAINT`, `PCT_PTS_OFF_TOV`, `PCT_AST_2PM/MR/3PM`, **`PCT_UAST_2PM/3PM/FGM`** (self-creation), `PCT_AST_FGM` | 26,398 team-games, **100%**, zero NaN |
| `team_gamelogs/misc_*.csv` | `PTS_OFF_TOV`, `PTS_FB`, `PTS_2ND_CHANCE`, `PTS_PAINT` and their `OPP_` twins | 100% |
| `player_gamelogs/usage_*.csv` | `PCT_FGA_2PT/3PT`, `PCT_FTB`, `PCT_PF/PFD`, full `PCT_*` block | 100% |
| `player_gamelogs/scoring_*.csv` | `PCT_PTS_3PT`, `PCT_PTS_PAINT`, `PCT_AST_3PM`, `PCT_UAST_3PM` | 100% |
| `shots/shotchart_*.csv` | 1,101,513 shots; `ACTION_TYPE` (53 values), `SHOT_ZONE_BASIC/AREA/RANGE`, `SHOT_DISTANCE`, `LOC_X/Y` | **49.2% of games** (per-season 46.5–54.4%) |
| `style/pt_player_defend_*.csv` | `DEFENSE_CATEGORY` ∈ {Overall, 2 Pointers, Less Than 6Ft, Less Than 10Ft, Greater Than 15Ft, 3 Pointers} × `D_FG_PCT`, `NORMAL_FG_PCT`, `PCT_PLUSMINUS`, `FREQ` | 35,332 rows, 1,539 players, 11 seasons — but `D_FG_PCT` is populated **only** on the 5,927 `Overall` rows; see §4.1 |
| `style/synergy_team_*.csv` | `PLAY_TYPE` ∈ {Transition, Isolation, PRBallHandler, PRRollMan, Postup, Spotup, Handoff, Cut, OffScreen, OffRebound, Misc} × `PPP` × `TYPE_GROUPING` ∈ {Offensive, Defensive}, plus `POSS_PCT`, `PERCENTILE` | 6,600 rows; **missing 2023-24**; 10 seasons |
| `style/ptstats_team_*.csv` (12 families) | Drive / catch-and-shoot / pull-up / paint-touch / elbow-touch / passing / speed-distance PPP and percentages | 30 teams × 10–11 seasons |
| `style/hustle_player_*.csv`, `hustle_team_*.csv` | `CONTESTED_SHOTS`, `DEFLECTIONS`, `CHARGES_DRAWN`, `SCREEN_ASSISTS`, loose-ball and box-out percentages | 30 teams / ~550 players per season |
| `style/team_shot_locations_base_*.csv` | FGA/FG% by zone: Restricted Area, Non-RA Paint, Mid-Range, L/R Corner 3, Above the Break 3, Backcourt | 30 × 11 |
| `meta/teams.csv` | `LAT`/`LON` — already hard-coded as `TEAM_COORDS` | 30 rows |
| `meta/players_all.csv` | `FROM_YEAR`, `TO_YEAR`, `OTHERLEAGUE_EXPERIENCE_CH`, `ROSTERSTATUS` | 5,236 rows |
| `schedule/*.csv` | `arenaCity`, `arenaName`, `isNeutral`, `gameSubtype`, `weekNumber`, `gameSequence` | — |

Two tables that look available and are not:

- **`rotations/`** — `rotation_2015-16.csv` only, 1,652 rows. `download.log` line 1527:
  `[rotations] 25 consecutive failures - endpoint looks blocked or unavailable.`
  **`winprob/`** is an empty directory; `download.log` line 1633 records the same failure.
- `isNeutral` is `False` for every game that survives the cleaning filters, so neutral-site
  games are not a data issue in this frame.

### 1.3 The walk-forward split, because it decides what "new" means

| | |
|---|---|
| training | `2015-10-27 .. 2021-03-21` (halves of 2015-16 .. 2020-21) |
| test (6,600 rows) | `2021-03-21 .. 2026-04-12` (2020-21 .. 2025-26) |
| base-rate log loss on those rows | 0.68758 |
| base-rate log loss on all 13,199 | 0.68484 |

The training half is the low-three-point league and the test half is the high-three-point
league. League 3PA share runs 0.285 (2015-16) → 0.415 (2025-26), +13.0 points, a 46% relative
rise; eFG% runs 0.276 → 0.310. **Any feature estimated on the early era and applied to the
late era faces a real distribution shift**, and §3.4 shows that where that shift bites
hardest, the feature does the most damage.

---

## 2. The full idea table

`r` is `corr(candidate, home_ewm_margin)` — the correlation that decides whether an idea is
a new quantity or a rearrangement. Anything above ~0.8 is a rearrangement and was not worth
proposing. Measured gain is from adding the block to the current 59-column frame, same
splits, paired bootstrap, 2,000 resamples.

`r` is `corr(candidate, home_ewm_margin)` — the correlation that decides whether an idea is
a new quantity or a rearrangement. Anything above ~0.8 is a rearrangement and was not worth
proposing. Measured gain is from adding the block to the current 59-column frame, same
splits, paired bootstrap, 2,000 resamples. Items 1–5 are the ones written up in §3.

| # | idea | the new information | why not a duplicate | `r` | measured | prior |
|---|---|---|---|---|---|---|
| **1** | **Heteroscedastic scale from trailing margin volatility** | the *spread* of a team's results, not the centre | `home/away_sdm10` vs `ewm_margin`: **+0.044 / −0.003** | 0.04 | **+0.00026, z 1.19** | **weak** |
| **2** | **Actual travel: where the visitors came from** | distance actually travelled, vs the current home-city-to-home-city proxy | vs `travel_miles`: **+0.269**; vs `ewm_margin`: **+0.006** | 0.01 | **+0.00026, z 1.02** | **weak** |
| **3** | **The 28 never-read per-game team columns** | transition points, 3PA share, unassisted-FG share, paint points | `home_e_uast_share` vs `ewm_margin` **−0.060**; `home_e_pts_2nd` **+0.058**; `home_e_pts_paint` **+0.060**; `home_e_fga3_share` **+0.135** | 0.06 | **−0.00017, z −0.22** | **null — the sharpest negative result in the table** |
| **4** | **Prior-season play-type PPP profile (off + def), crossed** | offensive *and defensive* identity decomposed into 11 play types | `home_x_Postup` vs `ewm_off_rating` **+0.079**, vs `ewm_def_rating` **−0.056**; crossed vs simple 0.638 | 0.08 | **−0.00169 to −0.00847** | **null as formulated** |
| **5** | **Pairing-specific margin deviation** | how team A does against *team B* beyond both clubs' strength | clean version (`pair mean − team mean`, same shrinkage): vs `ewm_margin` **+0.166**, sd 5.6 | 0.17 | **+0.00004, z 0.39** | **null** |
| 6 | Momentum as a second difference | last-2 mean margin minus last-10 mean margin | `home_mom2v10` vs `ewm_margin` **+0.062**; `home_momE3v20` **+0.343** | 0.06 | −0.00004, z −0.06 | **null** |
| 7 | Road-trip length / homestand length | consecutive away (or home) games; a *directional* schedule fact | `away_road_len` vs `ewm_margin` **+0.008**; vs `travel_miles` **+0.321** | 0.01 | −0.00009, z −0.68 | **null** |
| 8 | Shot-chart shot mix (14 cols) | 3PA/corner-3/ATB/rim shares, mean shot distance | `home_e_shot_dist` vs `ewm_margin` **+0.156**; all well under 0.9 | 0.16 | −0.00037, z −0.59 | **null, and 51% of games have no shots** |
| 9 | League era / season index | the league's own 3PA rate, total and home-win rate before the game | passes the correlation test on its face, but `lg3_share` vs `games_played` is **0.945** and `season_idx` is **0.996** — it is a calendar proxy | 0.00 | **−0.00128, z −2.25** | **harmful** |
| 10 | Interaction: rest × travel, rest × rating | does fatigue matter more on the road | `away_rest_x_travel` vs `away_rest_days` **0.720** — the interaction is 72% its own parent | 0.72 | inside block 2, null | **null** |
| 11 | Interaction: volatility × margin (heteroscedastic probit in linear form) | lets the margin coefficient vary with predicted noise | `home_sdm10` is orthogonal (0.044), so the interaction is not degenerate | 0.04 | §3.1 | **weak** |
| 12 | Prior-season team-rolled zone defence profile | measured defensive FG% by zone | **+0.537** vs `ewm_def_rating`, −0.358 vs `ewm_margin` (prior-season key, 90.7% coverage) — a rearrangement | 0.54 | not run | **reject as duplicate** |
| 13 | Player-level zone defence × opponent shot mix | per-player defensive FG% by zone, crossed with who the opponent shoots at | **not buildable**: `D_FG_PCT` is populated only for the `Overall` category — see §4.1 | — | not run | **reject on data grounds** |
| 14 | Synergy play-type *possession mix* alone | how a team distributes its possessions | `home_mix_Postup` vs `ewm_pace` −0.286, vs `ewm_off_rating` −0.270; orthogonal, but the parts sum to 1 and it is 22 collinear columns | 0.01 | −0.00184, z −1.97 | **null** |
| 15 | Two-point / three-point era splits by team | per-team regime change | the trailing 3PA share already sits inside `ewm_efg` and `ewm_ts_pct`; the residual is era-dependent, and era hurts | — | — | **reject** |

---

## 3. Top five, in full

### 3.1 Heteroscedastic scale — trailing margin volatility

**The idea.** Every prediction in this project is a plug-in `Φ(m̂/σ)` with one σ for the
league. If a game can be known in advance to be noisier than average — two teams that
normally play tight, or a high-variance offensive profile in the 3-point era — then the
log-odds that minimises log loss is `m/σ`, not `m`. A linear model cannot express that,
because it has a single coefficient on the margin term. The fix is either a second scale term
in the probit, or — inside the existing linear harness — one interaction column:
`margin_gap × volatility_gap`.

**Why it is not a transformation of what is present.** `ewm_margin` is a *location*
estimator. The trailing standard deviation of a team's margins is a *scale* estimator of the
same series, and the two are close to independent by construction:

```
corr(home_sdm10, home_ewm_margin) = +0.044
corr(away_sdm10, home_ewm_margin) = −0.003
corr(home_sdm10, home_ewm_off_rating) = +0.130
```

**How it is computed.** Build the long team-per-game frame (`GAME_ID`, `team`, `margin`
signed from that team's own point of view, sorted by `GAME_DATE`). Then
`shift(1)` **within team** — the same discipline `_ewm_by_team` enforces — and
`rolling(10, min_periods=5).std()`. NaN for a team's first five games, which the existing
`SimpleImputer(median)` handles. Emit `home_sdm10`, `away_sdm10` and, if the interaction is
tested, `sdm_gap × margin_gap` where `margin_gap = home_ewm_margin − away_ewm_margin`.

**Rationale (pre-registered style).** *Under a single-σ probit the margin coefficient is
forced to be constant across games; if a team's trailing margin dispersion predicts the
dispersion of its next margin, the log-odds should scale inversely with it, and a
heteroscedastic term is worth log loss even though the scale carries no mean information.*

**Honest prior: weak.** Two reasons. First, the measured gain is +0.00026 at z 1.19 — a null
by this project's own bar, though it is the largest of the nine blocks I ran. Second, and more
importantly: the expected gain is small *in principle*. Under a correctly specified model,
σ affects the optimal probability but the total variation it can produce on a game with
m̂ ≈ 0 and σ ≈ 12 is a few thousandths of a nat. It is a real effect of the right sign and it
is not going to close a 0.026-nat gap. The reason it is worth testing anyway is that it is
the only mechanism I found that improves the *shape* of the prediction rather than adding a
term, and the one existing result — logistic+PCA appearing to help — is itself a
shape/variance phenomenon that turned out to be an artifact.

**Cheapest falsification.** Run the block, look at `z`. One `fit_oof` pair, about 40 seconds,
no new code beyond the 6 lines above. If `|z| < 1`, stop. Do not extend it to a full
heteroscedastic probit unless the linear interaction already shows `|z| > 2`.

---

### 3.2 Actual travel — where the visitors came from

**The idea.** `rest_and_travel` computes `travel_miles` as the great-circle distance between
**the two clubs' home arenas**. That is not the distance anyone travels. It is a symmetric
function of the two teams, so it says a Los Angeles–Boston game is "2,600 miles of travel"
while assigning it to the home club, which did not travel at all. The real quantity is
asymmetric and conditional: *how far did the away team travel since its last game*, and
*how far did the home team travel to get back home.*

**Why it is not a transformation of what is present.** It is a **different definition**, not
a reweighting, and the correlation confirms it is a genuinely different number:

```
corr(away_travel_actual, travel_miles) = +0.269
corr(away_travel_actual, home_ewm_margin) = +0.006
corr(away_travel_actual, home_rest_days)  = +0.015
```

A correlation of 0.27 means the two agree about a quarter of the time — which is what you
would expect if both are driven by geography but one conditions on *where you came from*. The
existing feature has no idea a team was already on the road.

**How it is computed.** `schedule/*.csv` carries `arenaCity` per `gameId`. Join
`gameId == GAME_ID` (read `dtype={"gameId": str}`; see `DATA_NOTES.md` trap 2). Build a long
`(GAME_ID, team, side, arena_city)` frame, sort by `(team, GAME_DATE)`, `shift(1)` within team
to get `prev_city`, and take the great-circle distance using `nd.TEAM_COORDS` and
`nd._EARTH_RADIUS_MI` — the same haversine `rest_and_travel` already has. Emit
`home_travel_actual`, `away_travel_actual` and the gap.

Caveat I measured: **coverage is 74–80%, not 100%.** `arenaCity` is a real venue name, not a
team city, and it contains `Abu Dhabi`, `Anaheim`, `Austin`, `Beijing, China`, `Bossier
City` and others that match no franchise. The residual imputes to the median, which is
roughly "no travel", so it will under-state the effect on the missing quarter.

**Rationale (pre-registered style).** *The existing travel column measures the distance
between two fixed points and is symmetric in the two clubs; the distance the visitors
actually covered since their last game is an asymmetric, pre-game-knowable quantity that the
frame does not contain, and the two agree only 27% of the time.*

**Honest prior: weak.** Measured +0.00026 at z 1.02. The schedule block as a whole is already
the third-strongest block in the ablation (delta −0.0014, z 1.94) and its existing members are
almost pure levels — rest days, back-to-back, 3-in-4 — none of which is directional. The
mechanism is real and documented in the sports-science literature; the magnitude in *this*
frame is not. The honest read is that a correct fix to a mis-specified feature bought
something indistinguishable from zero.

**Cheapest falsification.** Already measured: `+0.00026, z 1.02`, null. The next cheap step,
only worth taking if someone wants it, is to fill the 26% of missing `arenaCity` rows with the
schedule's `homeTeam_teamTricode` (the nominal home club) rather than imputing, which is one
`fillna` and one more `fit_oof`.

---

### 3.3 The 28 never-read per-game team columns — the sharpest negative result

**The idea.** `team_gamelogs/scoring_*.csv` and `team_gamelogs/misc_*.csv` carry 14 columns
per team-game that no feature in this project has ever read, at **100% coverage with zero
NaN** across all 26,398 team-games: three-point and paint shot shares, points off
turnovers, fast-break points, second-chance points, and the assisted/unassisted split of
field goals. These are offensive *style* quantities, not the efficiency rates the frame
already carries.

**Why it is not a transformation of what is present.** This block passes the duplicate test
better than anything else I tried. The two most interesting members are close to orthogonal
to everything in the frame:

```
corr(home_e_uast_share,   home_ewm_margin) = −0.060   # unassisted FG share
corr(home_e_pts_2nd,      home_ewm_margin) = +0.058   # second-chance points
corr(home_e_pts_paint,    home_ewm_margin) = +0.060   # paint points
corr(home_e_pts3_share,   home_ewm_margin) = +0.168   # points from three
corr(home_e_fga3_share,   home_ewm_margin) = +0.135   # attempts from three
```

`ewm_efg` and `ewm_ts_pct` blend two- and three-point shooting into one number.
`unassisted FG share` — how much of a team's offence is self-generated rather than created —
is a different question entirely and nothing in the frame asks it. And it is *available
everywhere*, which the shot chart is not.

**How it is computed.** `_read_measures("team_gamelogs/scoring_*.csv", "Regular Season")` and
`("team_gamelogs/misc_*.csv", ...)`, keyed on `GAME_ID` + `TEAM_ABBREVIATION`. Rename the
columns, join `GAME_DATE` from `build_games`, sort by `(team, GAME_DATE)`, and pass each
through `_ewm_by_team(..., halflife=10.0)` — the existing causal helper. Per-100-possession
versions divide by `POSS` from `team_gamelogs/advanced_*.csv` (note: `load_team_advanced`
does **not** expose `POSS`; read the advanced table with `_read_measures` or the merge fails).
Emit home and away for each, 28 columns.

**Rationale (pre-registered style).** *The frame's shooting features are efficiency rates
that blend two- and three-point attempts and cannot express offensive style; a trailing
profile of transition points, self-generated field goals and shot-location shares is
available for every team-game at zero cost and is close to orthogonal to team form, so if
offensive composition predicts anything beyond offensive rating it will show up here.*

**Honest prior: null, and this is the block I would point at to justify that verdict.** The
measured gain is **−0.00017 at z −0.22**. This matters more than a null, because the block has
the three properties that distinguish a real information source from a rearrangement:

- it is **not correlated** with the existing features (0.06 to 0.17, not 0.9),
- it is **fully covered** — every row, no imputation, no 49% subsample,
- and it is **still worth nothing**.

The `opponent_adjusted` null had a different cause: 0.98 correlation meant rearrangement.
Here there is nothing to rearrange. `PCT_UAST_FGM` measures how a team creates shots, and how
it creates shots is already inside `OFF_RATING`. That is a genuinely useful thing to know,
because it bounds how much more can be extracted from team-level box-score style.

**Cheapest falsification.** Already run. If anyone wants to salvage it, the only version worth
trying is a **single scalar** — the first principal component of the 14 home columns,
appended once — because 28 near-collinear columns added to a 59-column frame is the same
mistake as §3.4 and §3.4 is a known −0.0085.

---

### 3.4 Prior-season play-type PPP, crossed — the most informative failure

**The idea.** `synergy_team` decomposes every team-season into eleven play types, with
offensive *and defensive* points-per-possession and a possession-share column. It is an
offensive and defensive **identity**, not a rating. The team-season-level correlations with
mean `OFF_RATING` are `Spotup` **+0.488**, `Postup` **+0.422**, `PRBallHandler` **+0.409**,
`Transition` **+0.362**, `Isolation` **+0.341** — and the *crossed* column (my offensive PPP
for a type minus your defensive PPP for it) drops to **+0.079** against `ewm_off_rating` and
**−0.056** against `ewm_def_rating`.

**Why it is not a transformation of what is present.** Established above with numbers. This
is the cleanest non-redundancy result in the whole exploration, and it is the reason this
idea is on the list despite losing.

**How it is computed.** Read `style/synergy_team_<season>.csv`, filter
`TYPE_GROUPING` to `Offensive` and `Defensive`, pivot on
`(SEASON, TEAM_ABBREVIATION) × PLAY_TYPE` over `PPP`. Map `SEASON` to the **previous** season
(season-end aggregate — the `DATA_NOTES.md` rule), join on
`TEAM_ABBREVIATION == home/away`. Emit 11 offensive + 11 defensive per side, plus the 22
crossed columns. 81.4% coverage as built, because `synergy_team` has no 2023-24 file.

**Rationale (pre-registered style).** *A team's play-type PPP profile is a decomposition of
offensive and defensive identity into eleven parts, and the crossed profile — my play types
against theirs — correlates below 0.08 with either club's rating, so it is information the
frame does not contain.*

**Honest prior: null as formulated, with a specific and diagnosable reason.** Measured:

| variant | cols | log loss | gain | z |
|---|---|---|---|---|
| offensive PPP only | 22 | 0.62190 | −0.00275 | −3.33 |
| defensive PPP only | 22 | 0.62064 | −0.00149 | −1.90 |
| possession mix only | 22 | 0.62099 | −0.00184 | −1.97 |
| crossed only | 22 | 0.62084 | −0.00169 | −2.32 |
| all of it | 66 | 0.62763 | −0.00847 | −4.96 |

Six variants, all negative, monotonically worse with more columns. This is the signature of
**overfitting, not of absent signal** — and the project already has a second, independent
piece of evidence for that reading: logistic + PCA(20) looked like a win and turned out to be
the leak. The 66 columns are 11 play types that sum to ~100% of possessions, so they are
rank-deficient by construction, they are all estimated from a *single prior season*, and the
test window is six seasons further into a league that has changed its play-type distribution
more than it has changed its 3PA share. The frame has 59 columns and no regularisation worth
the name; adding 66 correlated prior-season estimates to it is asking 13,199 rows to carry
125 degrees of freedom.

I also tested whether dimension reduction rescues it. It does not: crossed → 3 PCs gives
−0.00344, → 5 PCs −0.00370, → 8 PCs −0.00328; quality+mix → 5 PCs gives −0.00589.

**What the failure is worth.** The block proves the frame can receive information it did not
have and still lose. That is a statement about the *harness*, not the data: the project's
remaining binding constraint is not feature discovery and not model family, it is the
signal-to-architecture ratio of a 59-column unregularised frame.

**Cheapest falsification.** Already run and it is a kill. If anyone wants to revive it, the
only version worth trying is a **single scalar** — one principal component of the crossed
block, fitted inside the fold, appended to the frame — because the raw block is known to lose
by 0.0085 and the project's |z| > 3 bar says that is not noise.

---

### 3.5 Pairing-specific margin deviation

**The idea.** Teams have genuine stylistic matchup effects. Ask, causally, "how has team A
done against team B, compared with how team A has done against everybody?" and use the
residual.

**Why it is not a transformation of what is present.** My first construction was a
disaster and finding out why is the useful part. `mm_shrunk − ewm_adj_margin` correlated
**−0.578** with the margin gap — a large spurious negative correlation, because
`mm_shrunk` is shrunk toward the team's own historical mean while `ewm_adj_margin` is an
exponential average of the same margins. Differencing two estimators of the same quantity
manufactures a relationship.

The correct construction differences the *same estimator applied to two populations*:

```
pair_shrunk  = (sum of prior margins vs this opponent + K·team_mean) / (n_prior + K)
team_shrunk  = (sum of prior margins vs all opponents + K·team_mean) / (n + K)
mm_dev       = pair_shrunk − team_shrunk          # K = 6
```

That gives `corr(mm_dev_edge, margin_gap) = +0.166`, sd 5.6, coverage 100%. Low enough to be
a distinct quantity.

**How it is computed.** From `build_games`, emit both directions of every game into a
`(GAME_ID, team, opp, margin)` frame sorted by `(team, opp, GAME_DATE, GAME_ID)`.
`n_prior = cumcount()`, `sum_prior = cumsum() − margin`, `team_mean_prior` from
`groupby("team").margin.shift(1).expanding().mean()`. Then as above. Sample size is real:
**mean 16.0 prior meetings per ordered pair over the 11-season window, median 15, p90 33** —
about 1.5 per season, which is why K = 6 shrinkage is not optional.

**Rationale (pre-registered style).** *Matchup effects are real but the frame holds no
opponent-specific quantity; differencing the same shrinkage estimator applied to one
opponent against all opponents isolates the pairing without reintroducing the team-strength
term that killed `opponent_adjusted`.*

**Honest prior: null.** Measured +0.00004 at z 0.39. The mechanism is sound and the sample is
too thin: 16 prior meetings with a per-game margin sd near 12 gives a standard error of about
3 points on the pair estimate, and after shrinking with K = 6 toward a prior of sd 8, what
survives has an effective sample size of roughly one game. That is the honest reason it is a
null, and it is not fixable — the dataset does not contain enough meetings.

**Cheapest falsification.** Already run. If anyone wants to push it, the only version with a
chance is **cross-season style matchup** (does A historically give up more to ball-handling
teams?) rather than head-to-head history, because that pools 30 opponents instead of 1. That
is a different idea and needs the play-type profile of §3.4, which is itself a null.

---

## 4. Considered and rejected

### 4.1 Player-level zone defence × opponent shot mix — rejected on data grounds

This was the idea I expected to be the strongest, and checking it properly killed it. The
reason is worth recording precisely, because the table is genuinely attractive on paper:
`style/pt_player_defend_*.csv` has 35,332 rows, 1,539 players, 11 seasons, a `G` column with
a median of 51 games per player-season, and a `DEFENSE_CATEGORY` that splits into
{Overall, 2 Pointers, Less Than 6Ft, Less Than 10Ft, Greater Than 15Ft, 3 Pointers}. A
zone-resolved defensive rate, rolled over the expected-available roster via the existing
`nd._asof_roster(..., roster="prev_game")` machinery and crossed with the opponent's shot
mix, would be the one opponent-specific quantity the frame structurally cannot hold — and
availability (+0.00382) is the precedent that player-level composition can pay.

**It is not there.** Measured:

| `DEFENSE_CATEGORY` | rows | `D_FG_PCT` populated |
|---|---|---|
| Overall | 5,927 | **5,927 (100%)** |
| 2 Pointers | 5,900 | 0 |
| 3 Pointers | 5,870 | 0 |
| Greater Than 15Ft | 5,896 | 0 |
| Less Than 10Ft | 5,879 | 0 |
| Less Than 6Ft | 5,860 | 0 |

`D_FG_PCT`, `D_FGM`, `D_FGA`, `NORMAL_FG_PCT` and `PCT_PLUSMINUS` are populated **only** on
the `Overall` rows — 16.78% of the table. The five zone rows carry the raw
`FG2M/FG2A`, `FG3M/FG3A`, `FGM_LT_06/FGA_LT_06`, `FGM_LT_10/FGA_LT_10`, `FGM_GT_15/FGA_GT_15`
counters, but only in ~20% of them (≈1,150 rows per category across all eleven seasons). So a
zone-resolved defensive rate can only be *reconstructed* as FGM/FGA from a per-player sample
an order of magnitude thinner than the `Overall` median of 51 games. There is no published
rate to smooth it with.

The remaining `Overall`-only version is a duplicate anyway: with the **correct prior-season
key** (my first attempt keyed on the current season, which is the leak `DATA_NOTES.md`
warns about, and is why I initially measured +0.691), prior-season team-weighted
`D_FG_PCT(Overall)` correlates **+0.537** with `ewm_def_rating` and **−0.358** with
`ewm_margin`, over 11,969 games (90.7%, correctly zero in 2015-16).

Rejected. The player-level direction remains the only structurally different space available
— but the specific table that would fund it does not contain the quantity it is famous for,
and the aggregate-only version is 54% redundant.

### 4.2 Team-rolled prior-season zone defence profile

Covered by §4.1: **+0.537** vs `ewm_def_rating`. Below the 0.98 that killed
`opponent_adjusted` but not a new quantity — it is `DEF_RATING` measured by a different
route.

### 4.3 League era / season index

This was the brief's suggestion and it is measurably harmful: **−0.00128, z −2.25**. Two reasons. It is a calendar proxy — `corr(season_idx,
games_played) = 0.996`, `corr(lg3_share, games_played) = 0.945` — so it mostly re-expresses a
column the experience ablation already showed to be a null. And the drift it is meant to
capture is already absorbed: `ewm_off_rating` is per-100-possessions and era-normalised, and
the walk-forward fit retrains through the whole shift. The interesting detail is that
`lg3_share` correlates **+0.598** with `ewm_off_rating`, so the league's three-point rate
has partly leaked into the *team* ratings already — the frame is not era-blind, it just does
not want to be told what era it is in.

**Two-point vs three-point era splits.** Same failure as above, plus the shot mix is already
inside `ewm_efg` and `ewm_ts_pct`. The measurable residual is the 28-column block in the
table (`PCT_FGA_3PT` etc.), which is a null at z −0.22 with 100% coverage. The shot-chart
version is worse still (z −0.59) *and* only 49% of games have shots, so half the rows would
impute. The league's three-point revolution is real — +13.0 points of shot share across the
window — and it has no exploitable marginal value in this frame.

**Interactions with the schedule block.** `away_rest_days × travel_actual` correlates 0.720
with `away_rest_days` itself, so the interaction is 72% its own parent: the thing it is
supposed to add is mostly already there. `travel_miles × away_back_to_back` correlates −0.044
with anything. The brief's hypothesis ("does rest matter more for a team that travels?") is
mechanically plausible and empirically invisible in a linear frame.

**Predicted total as a variance proxy.** This is the "pace predicts variance" idea and it is
worth recording *why* it is not available. In the existing frame the predicted total is
already an almost exact linear function of `ewm_off_rating` and `ewm_def_rating`, so adding
it to a linear model adds nothing — it can only matter to a non-linear model, and
non-linear models have lost by a wide margin (XGBoost 0.6233, HistGB 0.6236). An OLS of
`|margin|` on trailing 3PA rate gives a coefficient of −3.50 with a standard error of 8.00
(null) while trailing pace gives +10.52 ± 3.25 — so pace does predict margin dispersion, and
it is already a column. §3.1 is the version that survives.

**Shot-chart catch-and-shoot rate.** I tried to classify `ACTION_TYPE` into catch-and-shoot
vs drive/pullup/fadeaway. The resulting feature had mean 0.885 and sd 0.091 — the NBA labels
the overwhelming majority of three-pointers simply `Jump Shot`, so the classifier has almost
no dynamic range. Not worth pursuing; the shot chart's value is in zone mix, and zone mix is
a null.

**Confirming the current `travel_miles` is broken.** It is mis-specified, not wrong: it
measures inter-city distance symmetrically. Fixing the definition is §3.2 and it bought
+z 1.02. Worth fixing for correctness even though it did not pay.

**Anything derived from `meta/players_all.csv`.** `FROM_YEAR`/`TO_YEAR` and
`OTHERLEAGUE_EXPERIENCE_CH` encode roster tenure, which is a reweighting of the expected
minutes that availability already sums. `meta/teams.csv` LAT/LON is already hard-coded as
`TEAM_COORDS`.

**Anything from `schedule` broadcast columns.** ~150 `nationalTvBroadcasters_*`,
`homeTvBroadcasters_*`, `awayRadioBroadcasters_*` columns exist and are mostly empty
strings. Broadcast reach and market size are a real thing, but there is no clean per-game
value, and a national game is broadcast to everyone.

---

## 5. What this implies about the dataset

The honest summary is uncomfortable and I think it is the right conclusion:

**The dataset is close to exhausted for new *team-level aggregate* information, and my
measurements are the evidence rather than the argument.**

The correlation screen is what makes this a finding. Several of these candidates are as
orthogonal to the existing features as it is possible for a team statistic to be — trailing
margin volatility at r = 0.04, the last-2-vs-last-10 margin contrast at r = 0.06, unassisted
field-goal share at r = −0.06 — and they are all null. If a quantity carries no correlation
with team strength, schedule state, or opponent quality, and it still does not predict the
winner, then it is not a weaker version of team strength. It is nothing. The opponent-adjusted
null had a different cause: it was 0.98 correlated, so it was a rearrangement. These are not
rearrangements, and they are still worth zero.

Two things this does *not* mean:

1. **It does not mean the feature layer is complete.** §3.4 found a block whose information is
   provably non-redundant (r ≈ 0.08) and which costs 0.0085 nats. The information arrived and
   the frame could not carry it. That is an architecture problem — 59 unregularised columns,
   no shrinkage, 13,199 rows — and the one project result that looked like it might already be
   an architecture win (logistic + PCA) turned out to be a leak. Those two findings point at
   the same place.
2. **It does not rule out the player level**, because the player level is a different kind of
   object, and §3.3 is the evidence for that: the frame absorbed a 100%-coverage,
   near-orthogonal block of offensive *style* and got nothing. The one player-level mechanism
   that paid is **availability** (+0.00382), and its reason is specific — a team-level
   average cannot express *who is on the floor*. That is a statement about absence, not about
   how good a roster is. Whether *quality* is player-predictable here is untested: RAPM is
   null across three families and two methods, and the honest player-Elo number does not exist
   yet. §4.1 removes the one table that looked like it could answer the question, so this
   stays open rather than closed.

And the framing constraint holds throughout: even a genuine +0.005 is a fifth of what the
market already extracts from the same games. **No feature in this dataset closes a 0.026-nat
gap, and I do not believe one exists.** What the feature work is for is calibration quality
and honest uncertainty, not closing the gap.

---

## 6. What I did NOT check — so the gaps stay visible

- **`summary/` and `pbp/` do not exist** (§0, finding 3). Referee crews, per-game inactive
  lists, and play-by-play are absent, so any idea depending on them is untested and untestable
  from here. A referee effect is the classic small-but-real NBA signal and it is the most
  notable missing table.
- **Rotation-level stints.** The backfill is still incomplete: `rotations/` holds 2020-21
  (728 downloaded games, 46,330 stint rows, 724 usable) plus 26 stalled 2015-16 games, against
  1,080 games in the 2020-21 schedule. True RAPM **has now been run** on the usable 2020-21
  games (`scripts/issue39_rapm.py`, `src/Models/rapm.py`), walk-forward over 362 test games.
  It is a null/inconclusive result at this sample size: 0.6594 log loss vs 0.6682 for the
  same-season 43-column frame (|z| 0.47) and 0.6516 for the same-season canonical model
  (|z| 0.54). That replaces the old "null for a proxy" statement with a measured RAPM null on
  one season; completing the backfill (#34) is still the largest change to what is knowable
  from player-level data here.
- **`winprob/`** — empty, same endpoint failure.
- **Confirmed starting lineups.** `player_elo_features` names `roster="confirmed"` as a mode
  and then documents that it is not computable from this dataset. Whether a lineup-confirmed
  run is a better product or a differently-timed one is a product decision, not a data gap.
- **The honest player-Elo result.** Still in progress. The number that would replace the
  stale ablation JSON is the only measurement in this project that could distinguish
  "player information is worthless" from "this particular player estimator is worthless", and
  it does not exist yet.
- **Any pre-2015-16 data.** Eleven seasons is the whole window; there is no cross-era
  validation of any feature outside it.
- **Non-regular-season games.** Zero playoff games by construction. All findings here are
  regular-season only and should not be read as covering the playoffs, where the rotation and
  availability mechanisms plausibly matter far more.
- **Zone-resolved defensive rates.** `style/pt_player_defend_*.csv` populates `D_FG_PCT`,
  `D_FGM`, `D_FGA`, `NORMAL_FG_PCT` and `PCT_PLUSMINUS` on the `Overall` category only; all
  five zone categories are 100% NaN on those columns (§4.1). Every idea that depends on
  "which defenders give up threes" is therefore untestable from this download, and the
  `shots/shotchart` table that could substitute for it covers only 49% of games.
- **The 369-game gap between `schedule/` (13,568 clean finished games) and
  `build_games` (13,199).** I did not chase which games those are. `build_games` also emits
  "dropping 0 duplicate home / 10 duplicate away rows" — a real data-quality warning that is
  currently swallowed by a `warnings.warn` and costs 5 games.
- **Injury reports, news, and market prices.** Outside the dataset by construction, and
  correctly so: `README.md` is explicit that odds are never a feature.
- **Hyperparameter tuning of the current frame.** Out of scope — that is algorithm search,
  which the brief rules out — but note that `C=0.1` was measured at 0.61822 against `C=1.0`
  at 0.61835, i.e. regularisation strength is not where anything is hiding either.

---

## Appendix: reproducing this

Project code is untouched. Scratch scripts are in `/tmp/opencode/nba/`:

| script | what it does |
|---|---|
| `cand.py` → `stageA.pkl` | rebuilds the 59-column frame and the sequence features |
| `candC2.py` → `stageC.pkl` | actual travel, never-read team columns, league era, matchup |
| `candB.py` → `stageB.pkl` | the 1.1M-shot per-team-game aggregation (49% coverage) |
| `playtype.py` → `playtype.pkl` | Synergy play-type PPP, prior-season, crossed |
| `stageF.py`, `ptpca.py`, `pcactrl.py` | the block ablations, the PCA controls |
| `dupcheck.py`, `shotcov.py`, `prof.py` | correlation and coverage audits |

Every log-loss number was produced by `scripts/recency_ablation.py:fit_oof` — the same
fold-fitting routine `scripts/feature_block_ablation.py` reuses, so the arms are directly
comparable with the existing block table. Splits are `nd.walk_forward_split(games, 5,
initial_frac=0.5, gap=0)`; all arms are scored on the identical 6,600 out-of-fold rows;
intervals are 2,000-resample paired bootstraps.
