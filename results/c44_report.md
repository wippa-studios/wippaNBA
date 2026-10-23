# Batch C / issue #44 - spread and total markets (results/c44_spread_total.json)

Pre-registration: results/experiments.jsonl :: C44-prereg-spread-total (written before any arm ran). 6 looks, Bonferroni 0.0083 one-sided; the |z| > 3 bar governs.
> Correction on record: Secondary-metric bug in the FIRST execution: skellam_bucket_prob_wide truncated half-integer bounds with astype(int) (toward zero) and was off by one on the lower bound, so 5-point bucket masses summed to ~1.20 instead of 1. The wide-bucket log score and the bucket-calibration Skellam column of that run are WRONG; the helper now uses cdf(5j+2)-cdf(5j-3) and _assert_bucket_partition enforces the partition. Re-run from the same cached predictions. PRIMARY endpoints (1-point margin log score via skellam.logpmf, cover/over probabilities via sf, C1 RMSEs) were computed from exact pmf/cdf values and are UNAFFECTED; the primary rows stand as logged.

## Inventory
- Odds DB: 13,424 reconciled rows (24,740 counted across every table spelling, 11,316 collapsed as duplicate spellings) over 2015-16..2025-26, every row carrying Spread and OU. **Spread and total lines exist -> C2 runs at full scope.**
- Per-season rows / spread / OU coverage: 2015-16 1,316/1,316/1,316 (frac(spread>0) 0.98); 2016-17 1,309/1,309/1,309 (frac(spread>0) 0.97); 2017-18 1,312/1,312/1,312 (frac(spread>0) 0.97); 2018-19 1,312/1,311/1,312 (frac(spread>0) 0.97); 2019-20 1,143/1,143/1,143 (frac(spread>0) 0.97); 2020-21 1,171/1,171/1,171 (frac(spread>0) 0.98); 2021-22 1,323/1,323/1,323 (frac(spread>0) 0.97); 2022-23 1,321/1,321/1,321 (frac(spread>0) 0.65); 2023-24 1,337/1,337/1,337 (frac(spread>0) 0.63); 2024-25 1,326/1,326/1,326 (frac(spread>0) 0.60); 2025-26 554/554/554 (frac(spread>0) 0.58)
- Spread sign convention: 2022-23..2025-26 signed expected home margin; 2015-16..2021-22 unsigned favourite handicap resolved by the de-vigged moneyline favourite (L_eff; inventory cover rates 47-52% after resolution vs 39-43% naive).
- Quarter-level scores: NOT BANKED: data/nba/summary/ (BoxScoreSummaryV2 -> LineScore) was never downloaded; scripts/download_nba_data.py defines the stage but no summary/ directory or LineScore CSV exists. No per-quarter margins are available; the margin distribution is game-level only.
- Juice: no price columns beyond ML_Home/ML_Away exist; spread and total bets are settled at a flat -110 (p_raw = 110/210)
- Market rows scored: {'joined_scored': 5788, 'signed_era_rows': 4098, 'unsigned_era_rows': 1690, 'spread_rows_sane': 5788, 'spread_rows_excluded_out_of_bounds': 0, 'total_rows_sane': 5782, 'total_rows_excluded_out_of_bounds': 6, 'cover_pushes_excluded': 80, 'over_pushes_excluded': 44}

## C1 - pace-first decomposition vs the monolithic margin/total model
delta = RMSE(baseline) - RMSE(candidate); POSITIVE = pace-first better.
### margin RMSE
| seed | look | baseline | pace-first | delta | 95% CI | SE | \|z\| |
|---|---|---|---|---|---|---|---|
| 42 | primary | 14.0183 | 14.0125 | +0.0058 | [-0.0153, +0.0260] | 0.0105 | 0.55 |
| 137 | replication | 14.0183 | 14.0125 | +0.0058 | [-0.0154, +0.0268] | 0.0108 | 0.53 |
| 2718 | replication | 14.0183 | 14.0125 | +0.0058 | [-0.0152, +0.0261] | 0.0105 | 0.55 |
MAE at the primary look: baseline 10.9878 vs pace-first 10.9895.
### total RMSE
| seed | look | baseline | pace-first | delta | 95% CI | SE | \|z\| |
|---|---|---|---|---|---|---|---|
| 42 | primary | 18.7766 | 18.7557 | +0.0209 | [-0.0063, +0.0480] | 0.0139 | 1.51 |
| 137 | replication | 18.7766 | 18.7557 | +0.0209 | [-0.0063, +0.0469] | 0.0136 | 1.54 |
| 2718 | replication | 18.7766 | 18.7557 | +0.0209 | [-0.0055, +0.0473] | 0.0135 | 1.55 |
MAE at the primary look: baseline 14.8635 vs pace-first 14.8430.

Per-split parity with the module's own report (README: margin 14.01, total 18.78):
- baseline split-mean margin RMSE 14.013, total RMSE 18.775
| split | n | margin base | margin pf | total base | total pf |
|---|---|---|---|---|---|
| 1 | 1,320 | 14.025 | 13.953 | 19.124 | 19.108 |
| 2 | 1,320 | 13.415 | 13.421 | 18.775 | 18.789 |
| 3 | 1,320 | 13.896 | 13.917 | 18.503 | 18.514 |
| 4 | 1,320 | 14.115 | 14.108 | 18.491 | 18.431 |
| 5 | 1,320 | 14.614 | 14.636 | 18.980 | 18.928 |

Decomposition diagnostics (descriptive): {'poss_hat_mean': 101.12457708987387, 'poss_true_mean': 101.88351818181818, 'ppp_home_hat_mean': 1.125760137806185, 'ppp_away_hat_mean': 1.1082540822092386, 'share_mu_away_negative': 0.0}

## C2 - Skellam margin distribution vs Phi(mhat/sigma)
delta = logloss(gaussian) - logloss(skellam); POSITIVE = Skellam better.
### (a) margin-bucket log score (1-point buckets, nats)
| seed | look | gaussian | skellam | delta | 95% CI | SE | \|z\| |
|---|---|---|---|---|---|---|---|
| 42 | primary | 4.05981 | 4.06382 | -0.00401 | [-0.00713, -0.00081] | 0.00161 | 2.49 |
| 137 | replication | 4.05981 | 4.06382 | -0.00401 | [-0.00733, -0.00054] | 0.00173 | 2.32 |
| 2718 | replication | 4.05981 | 4.06382 | -0.00401 | [-0.00739, -0.00060] | 0.00173 | 2.32 |
5-point buckets (robustness, no decision weight): gaussian 2.45766 vs skellam 2.46128, delta -0.00362 [-0.00667, -0.00043], |z| 2.27.

### (b) cover probabilities vs the closing spread (5,708 rows, pushes excluded)
| seed | look | logloss gaussian | logloss skellam | delta | 95% CI | SE | \|z\| |
|---|---|---|---|---|---|---|---|
| 42 | primary | 0.71908 | 0.71493 | +0.00415 | [+0.00358, +0.00470] | 0.00029 | 14.53 |
| 137 | replication | 0.71908 | 0.71493 | +0.00415 | [+0.00359, +0.00469] | 0.00028 | 14.75 |
| 2718 | replication | 0.71908 | 0.71493 | +0.00415 | [+0.00359, +0.00472] | 0.00029 | 14.38 |
Calibration, gaussian: ECE 0.0928, calibration-in-the-large +0.0048
  [0.0,0.1) n=1 pred 0.091 actual 1.000
  [0.1,0.2) n=16 pred 0.172 actual 0.438
  [0.2,0.3) n=126 pred 0.265 actual 0.476
  [0.3,0.4) n=837 pred 0.362 actual 0.496
  [0.4,0.5) n=1,937 pred 0.453 actual 0.508
  [0.5,0.6) n=1,801 pred 0.545 actual 0.481
  [0.6,0.7) n=776 pred 0.638 actual 0.485
  [0.7,0.8) n=191 pred 0.733 actual 0.545
  [0.8,0.9) n=18 pred 0.827 actual 0.389
  [0.9,1.0) n=5 pred 0.938 actual 0.800
Calibration, skellam: ECE 0.0858, calibration-in-the-large +0.0048
  [0.1,0.2) n=9 pred 0.165 actual 0.444
  [0.2,0.3) n=94 pred 0.266 actual 0.489
  [0.3,0.4) n=746 pred 0.364 actual 0.499
  [0.4,0.5) n=2,069 pred 0.453 actual 0.505
  [0.5,0.6) n=1,916 pred 0.545 actual 0.486
  [0.6,0.7) n=725 pred 0.637 actual 0.488
  [0.7,0.8) n=136 pred 0.732 actual 0.493
  [0.8,0.9) n=9 pred 0.829 actual 0.222
  [0.9,1.0) n=4 pred 0.926 actual 1.000
Sensitivity (continuous Phi, descriptive): {'n': 5708, 'log_loss': 0.7192055627928036, 'brier': 0.2619776566670629}

### margin-bucket distribution calibration (mean predicted mass vs observed)
| bucket | n | observed | gauss mass | skellam mass |
|---|---|---|---|---|
| [-57, -53] | 6 | 0.0009 | 0.0001 | 0.0003 |
| [-52, -48] | 4 | 0.0006 | 0.0003 | 0.0007 |
| [-47, -43] | 7 | 0.0011 | 0.0010 | 0.0018 |
| [-42, -38] | 25 | 0.0038 | 0.0026 | 0.0042 |
| [-37, -33] | 54 | 0.0082 | 0.0062 | 0.0088 |
| [-32, -28] | 114 | 0.0173 | 0.0133 | 0.0171 |
| [-27, -23] | 167 | 0.0253 | 0.0257 | 0.0300 |
| [-22, -18] | 276 | 0.0418 | 0.0444 | 0.0480 |
| [-17, -13] | 442 | 0.0670 | 0.0688 | 0.0699 |
| [-12, -8] | 720 | 0.1091 | 0.0954 | 0.0926 |
| [-7, -3] | 841 | 0.1274 | 0.1184 | 0.1115 |
| [-2, 2] | 602 | 0.0912 | 0.1315 | 0.1220 |
| [3, 7] | 943 | 0.1429 | 0.1307 | 0.1214 |
| [8, 12] | 857 | 0.1298 | 0.1162 | 0.1097 |
| [13, 17] | 606 | 0.0918 | 0.0925 | 0.0901 |
| [18, 22] | 384 | 0.0582 | 0.0659 | 0.0673 |
| [23, 27] | 238 | 0.0361 | 0.0420 | 0.0458 |
| [28, 32] | 150 | 0.0227 | 0.0239 | 0.0283 |
| [33, 37] | 83 | 0.0126 | 0.0122 | 0.0159 |
| [38, 42] | 38 | 0.0058 | 0.0056 | 0.0081 |
| [43, 47] | 21 | 0.0032 | 0.0023 | 0.0038 |
| [48, 52] | 12 | 0.0018 | 0.0008 | 0.0016 |
| [53, 57] | 6 | 0.0009 | 0.0003 | 0.0006 |
| [58, 62] | 3 | 0.0005 | 0.0001 | 0.0002 |
| [73, 77] | 1 | 0.0002 | 0.0000 | 0.0000 |

### total-over vs the closing total (secondary, no decision weight)
over log loss: gaussian total 0.71450 (Brier 0.26004); Poisson(t_hat) 0.72788 (Brier 0.26579); delta(poisson - gaussian loss) +0.01338 [+0.01176, +0.01490] (POSITIVE = gaussian better).
Gaussian-total over-calibration: ECE 0.0866, in-the-large -0.0344
  [0.0,0.1) n=2 pred 0.063 actual 0.500
  [0.1,0.2) n=10 pred 0.159 actual 0.500
  [0.2,0.3) n=160 pred 0.271 actual 0.463
  [0.3,0.4) n=1,195 pred 0.361 actual 0.497
  [0.4,0.5) n=2,294 pred 0.451 actual 0.516
  [0.5,0.6) n=1,494 pred 0.543 actual 0.502
  [0.6,0.7) n=523 pred 0.637 actual 0.501
  [0.7,0.8) n=56 pred 0.726 actual 0.446
  [0.8,0.9) n=4 pred 0.834 actual 0.500

## Issue #44 market tests (the closing line as a forecaster, and ROI)
MAE / RMSE against ACTUAL margins and totals, model vs closing line, same rows (delta = MAE(line) - MAE(model); POSITIVE = model better):
- spread/margin [margin_total]: model MAE 10.967 RMSE 13.924 vs line MAE 10.542 RMSE 13.478; delta -0.4252 [-0.5149, -0.3312], |z| 9.07
- spread/margin [pace_first]: model MAE 10.963 RMSE 13.911 vs line MAE 10.542 RMSE 13.478; delta -0.4213 [-0.5149, -0.3232], |z| 8.61
- totals [margin_total]: model MAE 14.953 RMSE 18.846 vs line MAE 14.420 RMSE 18.107; delta -0.5333 [-0.6487, -0.4170], |z| 9.02
- totals [pace_first]: model MAE 14.931 RMSE 18.827

ROI, parameter-free rule at a flat -110 (p_raw=0.52381), both sides considered, closing_line.roi() reused (threshold 0.5):
- spread_gaussian: 4,685 bets, ROI -4.86% SE 1.73%, |z| 2.80  (below the |z|>3 bar)
- spread_skellam_secondary: 4,599 bets, ROI -4.86% SE 1.79%, |z| 2.71  (below the |z|>3 bar)
- total_gaussian: 4,730 bets, ROI -3.99% SE 1.77%, |z| 2.26  (below the |z|>3 bar)

## Per-season (primary endpoints)
| season | n | margin base | margin pf | total base | total pf | logloss G | logloss Sk | cover n | cover G | cover Sk |
|---|---|---|---|---|---|---|---|---|---|---|
| 2020-21 | 460 | 14.348 | 14.244 | 18.863 | 18.912 | 4.0875 | 4.0841 | 456 | 0.7331 | 0.7266 |
| 2021-22 | 1,230 | 14.177 | 14.110 | 19.141 | 19.129 | 4.0723 | 4.0719 | 1,212 | 0.7109 | 0.7080 |
| 2022-23 | 1,230 | 12.984 | 13.025 | 18.859 | 18.866 | 3.9858 | 4.0025 | 1,212 | 0.7228 | 0.7179 |
| 2023-24 | 1,230 | 14.135 | 14.164 | 18.390 | 18.387 | 4.0681 | 4.0716 | 1,198 | 0.7220 | 0.7177 |
| 2024-25 | 1,225 | 14.021 | 14.006 | 18.377 | 18.299 | 4.0597 | 4.0639 | 1,120 | 0.7182 | 0.7143 |
| 2025-26 | 1,225 | 14.600 | 14.625 | 19.068 | 19.022 | 4.1029 | 4.1018 | 510 | 0.7125 | 0.7086 |

## Verdicts
- C1: **pace-first decomposition does NOT beat the monolithic model** (margin: null (CI straddles zero); total: null (CI straddles zero))
- C2: **a proper margin distribution EARNs its place** (margin log score: decisive loss; cover: clears the bar)

Reproduce:
```
.venv/bin/python scripts/c44_spread_total.py --prereg   # declaration (already logged)
.venv/bin/python scripts/c44_spread_total.py            # full measurement
```
Nothing in src/Models/margin_total.py, scripts/closing_line.py or the shared data layer was modified; every fitted quantity is train-only; no threshold sweep, no variants (selection cost: zero).
