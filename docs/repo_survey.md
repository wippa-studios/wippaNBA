# NBA Game Prediction Models — A Survey of Open-Source Repos

**Compiled:** 17 October 2026
**Scope:** Public GitHub repositories that model NBA game outcomes, spreads, totals, or player props.
**Method:** Repository search across 8 query angles (NBA prediction, betting model, win probability, Elo, spread prediction, etc.) yielded 123 candidates. Non-modelling repos were filtered out (API clients, score apps, fantasy lineup optimisers, salary/attendance/position predictors). README files were opened on ~45 of the remainder and scanned for algorithm, feature and reported-metric content.

> **Reading note:** every "reported results" figure below is transcribed from the repository's own README. Nothing here was re-run or independently validated. Repos marked **none** publish no metrics in their README — in many cases the numbers live in a notebook and were not surfaced.

---

## 1. Game-outcome and betting models

| Repo | ★ | Model type | Key features | Reported results |
|---|---|---|---|---|
| [kyleskom/NBA-Machine-Learning-Sports-Betting](https://github.com/kyleskom/NBA-Machine-Learning-Sports-Betting) | 1746 | XGBoost + neural net (dual) | team stats, SBR odds, days-rest; 2012–26 dataset | **none in README** |
| [cmunch1/nba-prediction](https://github.com/cmunch1/nba-prediction) | 312 | Classifier + Hopsworks feature store | engineered matchup features | **0.615** acc (2022-23); home-always baseline **0.580**; best public model **0.656** |
| [NBA-Betting/NBA_Betting](https://github.com/NBA-Betting/NBA_Betting) | 208 | AutoGluon AutoML + custom sequence architecture | multi-source aggregate, play-by-play | accuracy shown only as images; states it **does not beat Vegas** |
| [saccofrancesco/deepshot](https://github.com/saccofrancesco/deepshot) | 172 | XGBoost | EWMA rolling stats, basketball-reference | **none** |
| [nealmick/Sports-Betting-ML-Tools-NBA](https://github.com/nealmick/Sports-Betting-ML-Tools-NBA) | 88 | Black-box trainer | **500+ features/game**, 20k+ games; FG/3P/FT%, pts/ast/blk/stl/reb/min/foul/to | spread accuracy + P/L surfaced in-app, not in README |
| [whisdev/NBA-prediction-sports-betting](https://github.com/whisdev/NBA-prediction-sports-betting) | 90 | XGBoost (ONNX) | form, rest, live FanDuel/DK lines, EV + Kelly sizing | **~69% validation accuracy** (moneyline); totals model |
| [luke-lite/NBA-Prediction-Modeling](https://github.com/luke-lite/NBA-Prediction-Modeling) | 53 | LogReg, Random Forest, NN, **Gaussian NB + PCA** | 10 seasons, 11,979 games, 46 columns; 10/20/30-game aggregates | best = **Gaussian Naive Bayes + PCA**; large false-positive reduction vs other models |
| [agentscope-ai/DojoZero](https://github.com/agentscope-ai/DojoZero) | 56 | LLM agent platform | realtime events + odds, reproducible trial workflows | n/a — benchmark harness, not a model |
| [JoshWeiner/NBA_Game_Prediction](https://github.com/JoshWeiner/NBA_Game_Prediction) | 29 | Course project (CIS 545) | — | **no README at all** |
| [jensenbenjamin/NBA_Betting_Model](https://github.com/jensenbenjamin/NBA_Betting_Model) | 26 | Gradient Boosting **Regression** | score totals → over/under | **none** |
| [bendominguez0111/nba-models](https://github.com/bendominguez0111/nba-models) | 20 | 3-point prop model | player prop lines | **none** |
| [Spencer-Weston/NBApredict](https://github.com/Spencer-Weston/NBApredict) | 18 | **Four-Factor Regression** | scraped team stats, game results, betting lines in SQLite | **none** |
| [garfjohnson/Nba-Sports-Betting-Model](https://github.com/garfjohnson/Nba-Sports-Betting-Model) | 18 | Bets every game in a season | — | **none** |
| [VinceDiR/Prop_Betting_Regression_Project](https://github.com/VinceDiR/Prop_Betting_Regression_Project) | 14 | Regression | NBA prop betting strategy | **none** |
| [digsallday/Beating-the-House](https://github.com/digsallday/Beating-the-House-Identifying-Inefficiencies-in-Sports-Betting-Markets) | 12 | +EV probability model | moneyline, vig removal | claims **above-market returns** on NFL/NBA/NCAAF/NCAAB |
| [DevanshDaxini/Sports-EV-Bot](https://github.com/DevanshDaxini/Sports-EV-Bot) | 12 | **21-model XGBoost + LightGBM ensemble** (60/40) | **451 features**, 4 seasons, exponential recency weights, Optuna Bayesian HPO | full table below |
| [Pirkn/NBA-Game-Outcome-Prediction](https://github.com/Pirkn/NBA-Game-Outcome-Prediction) | 12 | Ridge, XGBoost, Neural Network | **Elo + rolling averages**, team- and matchup-specific features | **NN 66.9%**, Ridge 65.05%, XGBoost 64.8% |
| [Exidekat/math178project](https://github.com/Exidekat/math178project) | 4 | Linear + Logistic + HMM | ±1 roster indicator; home−away stat differences | **51.6% (roster) → 54.9% (stats) → 57.3% (both)** |
| [rileypsmith/nba-game-prediction](https://github.com/rileypsmith/nba-game-prediction) | 4 | Random Forest (PCA) vs Logistic Regression | open APIs, correlation-pruned features | RF best; **80 PCs marginally > 10 PCs**; LogReg clearly worse |
| [swapnil-ahlawat/NBA_Game_Predictor](https://github.com/swapnil-ahlawat/NBA_Game_Predictor) | 4 | Comparative study | team-only / player-only / combined | three separate scripts; **numbers only in program output** |
| [josedv82/NBA_Schedule_XGBoost_Classifier](https://github.com/josedv82/NBA_Schedule_XGBoost_Classifier) | 4 | XGBoost | **schedule only**, SHAP attribution | **RMSE 0.4247, LogLoss 0.5333, AUCPR 0.8073** |
| [muyangye/NBA_Prediction](https://github.com/muyangye/NBA_Prediction) | 4 | Logistic Regression | **Elo, 3PT%, offensive/defensive rating, opponent stats** | **none** |
| [developermarko/crystal_basketball](https://github.com/developermarko/crystal_basketball) | 2 | **Random Forest + SVM** | three markets: game line, over/under, point spread | **none** |
| [julesguasp/NBA_Spread_Model](https://github.com/julesguasp/NBA_Spread_Model) | 2 | **Lasso** (glm, binary classification) | point spread winners | **none** |
| [nivekgnay/nba-prediction-models](https://github.com/nivekgnay/nba-prediction-models) | 6 | Multiple | — | results live in a notebook, not the README |
| [7antoniosegovia/ML_NBA_Predictions](https://github.com/7antoniosegovia/ML_NBA_Predictions) | 6 | Multiple (R) | — | **none** |

---

## 2. Elo and rating systems

| Repo | ★ | Model type | Key features | Reported results |
|---|---|---|---|---|
| [tonyelhabr/nba_wp](https://github.com/tonyelhabr/nba_wp) | 7 | **443 discrete-time hazard regressions** + LOESS | repeats all records home/away to guarantee symmetry | reports log loss / mse / rate; **explicitly states his log loss is worse than Benz's 538 model** |
| [colekev/nba_win_prob_calc](https://github.com/colekev/nba_win_prob_calc) | 9 | Win probability calculator (R) | — | qualitative only |
| [nicidob/nba_elo](https://github.com/nicidob/nba_elo) | 12 | 538 Elo reimplementation + logistic-regression-as-Elo | PID control experiments | exploratory notebooks |
| [bbwieland/nba-elo-2](https://github.com/bbwieland/nba-elo-2) | 3 | Elo, updated daily | — | publishes daily standings + probability table |
| [bbwieland/NBAEloRatings](https://github.com/bbwieland/NBAEloRatings) | 2 | R package for Elo ratings | — | n/a |
| [tmont/nba-elo](https://github.com/tmont/nba-elo) | 2 | Elo (JavaScript) | — | n/a |
| [lucas-bishop/nba_wl_model](https://github.com/lucas-bishop/nba_wl_model) | 2 | 538 Elo + 4 other models | — | analysis, no single figure |
| [DanielTomaro13/Basketball-Modelling](https://github.com/DanielTomaro13/Basketball-Modelling) | 2 | **Sample-shrunk opponent-adjusted profiles + margin-aware Elo + Monte Carlo** | pace, offense/defense; markets read off simulation | **walk-forward backtest (log-loss / Brier / accuracy) vs home-court baseline** |
| [xpoes123/nba-modeling](https://github.com/xpoes123/nba-modeling) | 2 | **Hybrid RAPM (ridge) + Elo incremental updates** | stint-level data → Off/Def/Pace ratings → spread model | no score |
| [SidTheKid007/CarmElo-2.0](https://github.com/SidTheKid007/CarmElo-2.0) | 2 | CarmElo-inspired | — | no score |
| [Kush-Meta/OddsForge](https://github.com/Kush-Meta/OddsForge) | 2 | Elo + head-to-head + form ensemble (Rust) | edge detection API | no score |

---

## 3. Player prop models

| Repo | ★ | Model type | Key features | Reported results |
|---|---|---|---|---|
| [kpundhir/Prop-Model](https://github.com/kpundhir/Prop-Model) | 5 | RF + Logistic Regression, 5-fold stratified CV, SHAP | rolling averages, usage rates, home/away splits | **~60%+ test accuracy**; log loss console-only |
| [ejjlittle/ev-betting-model](https://github.com/ejjlittle/ev-betting-model) | 6 | +EV player props (TypeScript) | tracked over time, profit view | no score |
| [arnavsinghvi11/NBA_Betting_Model](https://github.com/arnavsinghvi11/NBA_Betting_Model) | 9 | Betting model | — | no score |
| [PerryGraham/Betting-Model-NBA](https://github.com/PerryGraham/Betting-Model-NBA) | 7 | Data cleaning + feature engineering + regression | — | no score |

---

## 4. Market structure and non-ML approaches

| Repo | ★ | Model type | Key features | Reported results |
|---|---|---|---|---|
| [yangyuan-zhen/polysniper](https://github.com/yangyuan-zhen/polysniper) | 14 | No model — arbitrage monitor | ESPN odds vs Polymarket prediction markets, WebSocket heartbeat, exit strategy | n/a |
| [neeilan/DeepPlayByPlay](https://github.com/neeilan/DeepPlayByPlay) | 183 | Deep learning, play-by-play labelling | raw SportVU tracking data | n/a — labelling, not outcome prediction |
| [danchyy/Basketball_Analytics](https://github.com/danchyy/Basketball_Analytics) | 172 | Notebook collection | defensive specialists, clutch splits, MVP | n/a — analysis, not a deployed model |
| [fsmosca/Basketball-Winning-Probability](https://github.com/fsmosca/Basketball-Winning-Probability) | 2 | Multiple linear regression | 2PT%, steals, fouls, EFF; international tournaments | regression weights published per tournament phase |

---

## 5. The most fully documented result set

**DevanshDaxini/Sports-EV-Bot** is the only repo publishing a complete per-target metric table. 21-model XGBoost/LightGBM ensemble trained on 4 seasons with exponential recency sample weights and Optuna Bayesian hyperparameter optimisation:

| Target | MAE | R² | Directional accuracy |
|---|---|---|---|
| PTS | 4.77 | 0.461 | 60.2% |
| REB | 1.94 | 0.411 | 57.9% |
| AST | 1.43 | 0.465 | 55.5% |
| FG3M | 0.94 | 0.240 | 54.6% |
| FG3A | 1.56 | 0.514 | 54.0% |
| FGA | 2.79 | **0.570** | 60.5% |
| FGM | 1.84 | 0.413 | 58.3% |
| FT% (FTM) | 1.48 | 0.348 | 50.5% |
| PRA | 6.09 | 0.516 | 61.1% |
| FPTS | 7.77 | 0.459 | 61.4% |
| TOV | 0.96 | 0.203 | 62.0% |
| BLK | 0.54 | **0.164** | 39.7% |

The spread between the best-modelled target (FGA, R² 0.570) and the worst (BLK, R² 0.164) is the useful finding: **volume is predictable; rare events are not.** Directional accuracy for blocks is 39.7% — below a coin flip.

---

## 6. Cross-cutting findings

### 6.1 The accuracy ceiling is ~65–69%, and almost nobody benchmarks against it

Reported moneyline accuracies across the set:

| Repo | Accuracy |
|---|---|
| whisdev (moneyline) | ~0.69 |
| Pirkn (neural network) | 0.669 |
| Pirkn (ridge) | 0.6505 |
| Pirkn (XGBoost) | 0.648 |
| cmunch1 | 0.615 |
| Exidekat (combined features) | 0.573 |
| Exidekat (player stats only) | 0.549 |
| Exidekat (roster indicator only) | 0.516 |

**cmunch1/nba-prediction is the only repository that reports a baseline and a public comparator alongside its own number** — 0.615 against 0.580 for "home team always wins", and 0.656 for the best public model on the same period. Every other figure in this document is unfalsifiable without rerunning the code.

### 6.2 Elo is used as a feature far more often than as a model

muyangye/NBA_Prediction and Pirkn/NBA-Game-Outcome-Prediction both feed Elo into a classifier as one input among many. The repos where Elo *is* the model (tonyelhabr, bbwieland, lucas-bishop) publish standings and probability tables but no validation numbers. DanielTomaro13 combines it with Monte Carlo and is the only one backtesting it properly.

### 6.3 Two repositories are honest about losing

- **tonyelhabr/nba_wp** states outright that its log loss is worse than Benz's 538 model, and offers the explanation (less training data, higher parity in the NBA than college basketball).
- **NBA-Betting/NBA_Betting** states that "consistently outperforming Vegas lines remains a challenge", noting the additional hurdle of the vig.

This is the **only** mention of the vig across ~45 README files.

### 6.4 Feature volume does not track performance

nealmick uses **500+ features per game** across 20,000+ games and reports no accuracy. DevanshDaxini's 451-feature, 21-model ensemble yields R² 0.570 on the best target and 0.164 on the worst. The repos with the strongest numbers (Exidekat, Pirkn) use compact hand-engineered feature sets.

### 6.5 Reporting is the main weakness

- **13 of 28** game-outcome repos report no metrics at all in their README.
- One 29★ repo has **no README whatsoever**.
- Several publish results only as images (NBA-Betting) or only as notebook output (swapnil-ahlawat, nivekgnay).
- Only one repo (DevanshDaxini) publishes a machine-readable metric table.

### 6.6 Two genuinely distinct architectures stand out

Both report numbers, and both are among the smallest repos in the set:

1. **DevanshDaxini/Sports-EV-Bot** — 21-model gradient-boosting ensemble with recency weighting, per-target metrics published.
2. **DanielTomaro13/Basketball-Modelling** — Monte-Carlo simulation over sample-shrunk opponent-adjusted profiles blended with a margin-aware Elo, evaluated with a walk-forward backtest against a home-court baseline.

---

## 7. Caveats

- All figures are transcribed from repository READMEs. **None were independently verified or re-run.** Where a repo claims a result, the claim is reported, not endorsed.
- Accuracy figures are **not comparable across repos**: different evaluation windows, different splits, different label definitions (home-win vs moneyline vs spread), and mostly no held-out test discipline described.
- "Directional accuracy" in the DevanshDaxini table is a different quantity from classification accuracy elsewhere and should not be compared to the 60–69% figures.
- The vig — typically 4–8% on a two-way market — is not accounted for in any reported accuracy figure found. A 65% accuracy model is roughly break-even only after vig is deducted, which is consistent with the two honest repos in §6.3.

---

## 8. Suggested next steps

1. **Recover the unpublished metrics** — parse notebooks from the 13 repos reporting nothing, to fill in the table.
2. **Re-verify the top claims** — rerun cmunch1, Pirkn and Exidekat against a common held-out window and a common vig-adjusted metric. These three are the only ones with enough detail to reproduce.
3. **Look at DanielTomaro13** — the only repo in the set that backtests properly (walk-forward, against a stated baseline). Its architecture is also the least common.