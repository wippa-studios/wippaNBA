# NBA moneyline baselines -- pooled walk-forward results

Data root: `/home/Joel/Desktop/Legit_NBA_Predictions/data/nba`  
Splits: 5 chronological walk-forward blocks (initial_frac=0.5, gap=0)  
Elo K selected on validation: **10** (test rows never used for the choice)

**The bar: `log-linear on elo` at log loss 0.6234, AUC 0.7026.**

## RESULTS (log loss first)

                              model    n  log_loss  log_loss_se  brier  brier_se    auc  accuracy  accuracy_se  mean_pred                                                         note
                  log-linear on elo 6600    0.6234       0.0044 0.2169    0.0020 0.7026    0.6542       0.0059     0.5713        logistic on (R_home, R_away, 1), fitted on train only
                      elo (k tuned) 6600    0.6260       0.0050 0.2180    0.0022 0.7020    0.6512       0.0059     0.5853                          k=10, hfa=70, mov, same-day batched
            elo, sequential (alpha) 6600    0.6260       0.0050 0.2180    0.0022 0.7020    0.6512       0.0059     0.5853                      same-date games processed one at a time
          elo, sequential (GAME_ID) 6600    0.6260       0.0050 0.2180    0.0022 0.7020    0.6512       0.0059     0.5853                  same-date games processed in schedule order
                    elo + rest term 6600    0.6271       0.0051 0.2182    0.0022 0.7017    0.6529       0.0059     0.5856                   3 rating points per extra rest day, causal
elo (k=20, FiveThirtyEight default) 6600    0.6300       0.0061 0.2187    0.0025 0.7055    0.6594       0.0058     0.5791                              no tuning at all, for reference
                        elo, no mov 6600    0.6343       0.0040 0.2220    0.0018 0.6912    0.6394       0.0059     0.5920                             margin-of-victory multiplier off
                          base rate 6600    0.6883       0.0018 0.2476    0.0009 0.5010    0.5527       0.0061     0.5712                      train home-win rate, constant per split
     srs + intercept (walk-forward) 6600    0.6890       0.0015 0.2479    0.0007 0.5020    0.5500       0.0061     0.5529 refit every 250 games, calibration on a time-ordered holdout
           srs + intercept (frozen) 6600    0.6923       0.0020 0.2495    0.0010 0.4867    0.5503       0.0061     0.5711   logistic with intercept on the SRS margin, fitted on train
       bradley-terry (walk-forward) 6600    0.6929       0.0033 0.2494    0.0015 0.5514    0.5568       0.0061     0.5705                                        refit every 250 games
           srs (module calibration) 6600    0.6933       0.0002 0.2501    0.0001 0.4875    0.4855       0.0062     0.5000     home edge subtracted twice; expect ~0.50 mean prediction
             bradley-terry (frozen) 6600    0.7025       0.0034 0.2537    0.0016 0.5325    0.5464       0.0061     0.5718                              batch MLE on the training block

## Paired bootstrap vs base rate

`paired_bootstrap_delta(y, p_base, p_model)` = loss(base) - loss(model), so a POSITIVE delta means the model is BETTER than the base rate.

                              model reference  delta_vs_ref   ci_lo   ci_hi  p_better  better_than_ref
             bradley-terry (frozen) base rate       -0.0142 -0.0198 -0.0086    0.0000            False
           srs (module calibration) base rate       -0.0051 -0.0085 -0.0017    0.0015            False
       bradley-terry (walk-forward) base rate       -0.0046 -0.0100  0.0007    0.0395            False
           srs + intercept (frozen) base rate       -0.0040 -0.0057 -0.0024    0.0000            False
     srs + intercept (walk-forward) base rate       -0.0008 -0.0024  0.0009    0.1825            False
                        elo, no mov base rate        0.0539  0.0473  0.0603    1.0000             True
elo (k=20, FiveThirtyEight default) base rate        0.0582  0.0465  0.0690    1.0000             True
                    elo + rest term base rate        0.0612  0.0519  0.0700    1.0000             True
                      elo (k tuned) base rate        0.0622  0.0531  0.0708    1.0000             True
          elo, sequential (GAME_ID) base rate        0.0622  0.0531  0.0708    1.0000             True
            elo, sequential (alpha) base rate        0.0622  0.0531  0.0708    1.0000             True
                  log-linear on elo base rate        0.0649  0.0567  0.0727    1.0000             True

## Paired bootstrap vs `log-linear on elo`

Same convention: POSITIVE delta means that model is BETTER than `log-linear on elo`.

                              model         reference  delta_vs_ref   ci_lo   ci_hi  p_better  better_than_ref
             bradley-terry (frozen) log-linear on elo       -0.0791 -0.0877 -0.0703    0.0000            False
           srs (module calibration) log-linear on elo       -0.0700 -0.0785 -0.0613    0.0000            False
       bradley-terry (walk-forward) log-linear on elo       -0.0695 -0.0776 -0.0614    0.0000            False
           srs + intercept (frozen) log-linear on elo       -0.0689 -0.0771 -0.0605    0.0000            False
     srs + intercept (walk-forward) log-linear on elo       -0.0657 -0.0732 -0.0578    0.0000            False
                          base rate log-linear on elo       -0.0649 -0.0727 -0.0567    0.0000            False
                        elo, no mov log-linear on elo       -0.0110 -0.0134 -0.0085    0.0000            False
elo (k=20, FiveThirtyEight default) log-linear on elo       -0.0067 -0.0110 -0.0027    0.0000            False
                    elo + rest term log-linear on elo       -0.0037 -0.0060 -0.0016    0.0000            False
            elo, sequential (alpha) log-linear on elo       -0.0027 -0.0043 -0.0012    0.0010            False
                      elo (k tuned) log-linear on elo       -0.0027 -0.0043 -0.0012    0.0010            False
          elo, sequential (GAME_ID) log-linear on elo       -0.0027 -0.0043 -0.0012    0.0010            False

## Elo K validation sweep

Selected on the last 20% of each TRAINING block. Test rows never entered this choice.

      k  val_log_loss_mean  val_split_0  val_split_1  val_split_2  val_split_3  val_split_4  val_rank
10.0000             0.6401       0.6642       0.6427       0.6423       0.6311       0.6199    1.0000
12.0000             0.6403       0.6654       0.6413       0.6437       0.6318       0.6194    2.0000
 8.0000             0.6407       0.6636       0.6455       0.6413       0.6312       0.6217    3.0000
15.0000             0.6419       0.6682       0.6408       0.6466       0.6338       0.6201    4.0000
 6.0000             0.6429       0.6639       0.6506       0.6415       0.6329       0.6255    5.0000
20.0000             0.6463       0.6740       0.6430       0.6525       0.6391       0.6231    6.0000
 4.0000             0.6476       0.6656       0.6585       0.6439       0.6371       0.6328    7.0000
25.0000             0.6521       0.6806       0.6476       0.6593       0.6454       0.6276    8.0000
30.0000             0.6587       0.6881       0.6533       0.6666       0.6526       0.6329    9.0000
40.0000             0.6730       0.7044       0.6661       0.6815       0.6679       0.6450   10.0000

## Same-date cost

POSITIVE `vs_batched_delta` = the sequential variant scored better, i.e. it gained from seeing same-date results. All three are identical because no team plays twice in a day, which makes the equality a proof rather than a measurement.

                 variant  pooled_log_loss  split_0_log_loss  split_1_log_loss  split_2_log_loss  split_3_log_loss  split_4_log_loss  vs_batched_delta  vs_batched_ci_lo  vs_batched_ci_hi
       batched (default)           0.6260            0.6370            0.6433            0.6203            0.6248            0.6048               NaN               NaN               NaN
sequential, alphabetical           0.6260            0.6370            0.6433            0.6203            0.6248            0.6048            0.0000            0.0000            0.0000
     sequential, GAME_ID           0.6260            0.6370            0.6433            0.6203            0.6248            0.6048            0.0000            0.0000            0.0000

## Per split

                              model split    n  log_loss  log_loss_se  brier    auc  accuracy  mean_pred
                          base rate     0 1320    0.6893       0.0042 0.2481 0.5000    0.5508     0.5763
                          base rate     1 1320    0.6823       0.0040 0.2446 0.5000    0.5735     0.5720
                          base rate     2 1320    0.6900       0.0040 0.2484 0.5000    0.5470     0.5722
                          base rate     3 1320    0.6910       0.0038 0.2489 0.5000    0.5424     0.5691
                          base rate     4 1320    0.6887       0.0036 0.2478 0.5000    0.5500     0.5661
                      elo (k tuned)     0 1320    0.6370       0.0110 0.2229 0.6892    0.6447     0.5896
                      elo (k tuned)     1 1320    0.6433       0.0112 0.2250 0.6700    0.6402     0.5873
                      elo (k tuned)     2 1320    0.6203       0.0106 0.2164 0.7060    0.6386     0.5811
                      elo (k tuned)     3 1320    0.6248       0.0117 0.2177 0.7089    0.6508     0.5842
                      elo (k tuned)     4 1320    0.6048       0.0119 0.2079 0.7323    0.6818     0.5844
elo (k=20, FiveThirtyEight default)     0 1320    0.6388       0.0135 0.2222 0.6986    0.6614     0.5830
elo (k=20, FiveThirtyEight default)     1 1320    0.6548       0.0133 0.2286 0.6673    0.6348     0.5825
elo (k=20, FiveThirtyEight default)     2 1320    0.6265       0.0131 0.2185 0.7053    0.6485     0.5734
elo (k=20, FiveThirtyEight default)     3 1320    0.6264       0.0140 0.2176 0.7147    0.6583     0.5784
elo (k=20, FiveThirtyEight default)     4 1320    0.6038       0.0142 0.2066 0.7382    0.6939     0.5784
                        elo, no mov     0 1320    0.6469       0.0085 0.2278 0.6724    0.6197     0.5952
                        elo, no mov     1 1320    0.6428       0.0086 0.2256 0.6648    0.6326     0.5927
                        elo, no mov     2 1320    0.6301       0.0086 0.2207 0.6976    0.6386     0.5891
                        elo, no mov     3 1320    0.6325       0.0093 0.2213 0.6996    0.6394     0.5906
                        elo, no mov     4 1320    0.6193       0.0093 0.2148 0.7184    0.6667     0.5925
            elo, sequential (alpha)     0 1320    0.6370       0.0110 0.2229 0.6892    0.6447     0.5896
            elo, sequential (alpha)     1 1320    0.6433       0.0112 0.2250 0.6700    0.6402     0.5873
            elo, sequential (alpha)     2 1320    0.6203       0.0106 0.2164 0.7060    0.6386     0.5811
            elo, sequential (alpha)     3 1320    0.6248       0.0117 0.2177 0.7089    0.6508     0.5842
            elo, sequential (alpha)     4 1320    0.6048       0.0119 0.2079 0.7323    0.6818     0.5844
          elo, sequential (GAME_ID)     0 1320    0.6370       0.0110 0.2229 0.6892    0.6447     0.5896
          elo, sequential (GAME_ID)     1 1320    0.6433       0.0112 0.2250 0.6700    0.6402     0.5873
          elo, sequential (GAME_ID)     2 1320    0.6203       0.0106 0.2164 0.7060    0.6386     0.5811
          elo, sequential (GAME_ID)     3 1320    0.6248       0.0117 0.2177 0.7089    0.6508     0.5842
          elo, sequential (GAME_ID)     4 1320    0.6048       0.0119 0.2079 0.7323    0.6818     0.5844
                    elo + rest term     0 1320    0.6381       0.0111 0.2232 0.6894    0.6455     0.5899
                    elo + rest term     1 1320    0.6418       0.0112 0.2244 0.6714    0.6455     0.5872
                    elo + rest term     2 1320    0.6196       0.0106 0.2161 0.7070    0.6409     0.5814
                    elo + rest term     3 1320    0.6275       0.0120 0.2185 0.7068    0.6500     0.5851
                    elo + rest term     4 1320    0.6084       0.0122 0.2090 0.7299    0.6826     0.5846
                  log-linear on elo     0 1320    0.6340       0.0098 0.2217 0.6902    0.6409     0.5803
                  log-linear on elo     1 1320    0.6388       0.0098 0.2234 0.6733    0.6462     0.5734
                  log-linear on elo     2 1320    0.6189       0.0093 0.2156 0.7067    0.6379     0.5682
                  log-linear on elo     3 1320    0.6211       0.0102 0.2163 0.7091    0.6583     0.5683
                  log-linear on elo     4 1320    0.6038       0.0104 0.2078 0.7310    0.6879     0.5662
             bradley-terry (frozen)     0 1320    0.7391       0.0094 0.2702 0.4834    0.5061     0.5808
             bradley-terry (frozen)     1 1320    0.7024       0.0079 0.2536 0.5249    0.5470     0.5716
             bradley-terry (frozen)     2 1320    0.6941       0.0070 0.2502 0.5455    0.5447     0.5709
             bradley-terry (frozen)     3 1320    0.6901       0.0068 0.2481 0.5556    0.5576     0.5691
             bradley-terry (frozen)     4 1320    0.6866       0.0066 0.2466 0.5635    0.5765     0.5666
       bradley-terry (walk-forward)     0 1320    0.7223       0.0087 0.2628 0.5033    0.5197     0.5782
       bradley-terry (walk-forward)     1 1320    0.6948       0.0075 0.2501 0.5380    0.5485     0.5712
       bradley-terry (walk-forward)     2 1320    0.6846       0.0068 0.2458 0.5675    0.5614     0.5699
       bradley-terry (walk-forward)     3 1320    0.6830       0.0066 0.2448 0.5746    0.5712     0.5675
       bradley-terry (walk-forward)     4 1320    0.6797       0.0065 0.2433 0.5828    0.5833     0.5658
           srs (module calibration)     0 1320    0.6931       0.0003 0.2500 0.4995    0.4932     0.5001
           srs (module calibration)     1 1320    0.6931       0.0000 0.2500 0.4933    0.4879     0.5000
           srs (module calibration)     2 1320    0.6956       0.0005 0.2512 0.4219    0.4500     0.5000
           srs (module calibration)     3 1320    0.6941       0.0006 0.2505 0.4853    0.4902     0.4999
           srs (module calibration)     4 1320    0.6908       0.0008 0.2488 0.5438    0.5061     0.4999
           srs + intercept (frozen)     0 1320    0.6912       0.0046 0.2489 0.4995    0.5538     0.5766
           srs + intercept (frozen)     1 1320    0.6858       0.0044 0.2463 0.4933    0.5735     0.5717
           srs + intercept (frozen)     2 1320    0.7046       0.0047 0.2555 0.4219    0.5318     0.5724
           srs + intercept (frozen)     3 1320    0.6946       0.0042 0.2507 0.4853    0.5417     0.5688
           srs + intercept (frozen)     4 1320    0.6851       0.0039 0.2460 0.5438    0.5508     0.5659
     srs + intercept (walk-forward)     0 1320    0.6888       0.0033 0.2478 0.5195    0.5576     0.5435
     srs + intercept (walk-forward)     1 1320    0.6864       0.0035 0.2466 0.4933    0.5621     0.5545
     srs + intercept (walk-forward)     2 1320    0.6892       0.0036 0.2480 0.5058    0.5462     0.5637
     srs + intercept (walk-forward)     3 1320    0.6920       0.0036 0.2494 0.5011    0.5341     0.5579
     srs + intercept (walk-forward)     4 1320    0.6889       0.0025 0.2479 0.4698    0.5500     0.5448

## Calibration

                              model  bin     lo     hi    n  mean_pred  actual
                          base rate    5 0.5000 0.6000 6600     0.5712  0.5527
                      elo (k tuned)    0 0.0000 0.1000    5     0.0857  0.0000
                      elo (k tuned)    1 0.1000 0.2000  127     0.1645  0.1811
                      elo (k tuned)    2 0.2000 0.3000  421     0.2570  0.2423
                      elo (k tuned)    3 0.3000 0.4000  676     0.3535  0.3580
                      elo (k tuned)    4 0.4000 0.5000  923     0.4518  0.4160
                      elo (k tuned)    5 0.5000 0.6000 1140     0.5514  0.5184
                      elo (k tuned)    6 0.6000 0.7000 1263     0.6492  0.6097
                      elo (k tuned)    7 0.7000 0.8000 1113     0.7481  0.7008
                      elo (k tuned)    8 0.8000 0.9000  796     0.8434  0.8028
                      elo (k tuned)    9 0.9000 1.0000  136     0.9200  0.8603
elo (k=20, FiveThirtyEight default)    0 0.0000 0.1000   57     0.0800  0.1053
elo (k=20, FiveThirtyEight default)    1 0.1000 0.2000  329     0.1576  0.2128
elo (k=20, FiveThirtyEight default)    2 0.2000 0.3000  527     0.2510  0.3017
elo (k=20, FiveThirtyEight default)    3 0.3000 0.4000  662     0.3495  0.3882
elo (k=20, FiveThirtyEight default)    4 0.4000 0.5000  823     0.4513  0.4313
elo (k=20, FiveThirtyEight default)    5 0.5000 0.6000  899     0.5510  0.5428
elo (k=20, FiveThirtyEight default)    6 0.6000 0.7000  995     0.6495  0.6121
elo (k=20, FiveThirtyEight default)    7 0.7000 0.8000 1014     0.7495  0.6667
elo (k=20, FiveThirtyEight default)    8 0.8000 0.9000  903     0.8484  0.7730
elo (k=20, FiveThirtyEight default)    9 0.9000 1.0000  391     0.9284  0.8440

## Notes

1. Elo reaches log loss 0.6260 against a base rate of 0.6883, and AUC 0.7020 against 0.5010. Against the base rate that is +0.0622 nats, 95% CI [+0.0531, +0.0708] -- excludes zero, Elo is the better model. The log-linear variant is +0.0649, CI [+0.0567, +0.0727] -- excludes zero, the log-linear variant is the better model.

2. The log-linear recalibration of the Elo ratings is 0.0012 to 0.0043 nats better than raw Elo (0.6234 vs 0.6260): `paired_bootstrap_delta(y, p_elo, p_loglinear)` is -0.0027, 95% CI [-0.0043, -0.0012] in loss(elo) - loss(log-linear) terms -- excludes zero, raw Elo is the worse model. Note what that improvement is: a three-parameter recalibration of the same information, not a new source of information.

3. Ranking and pricing come apart. FiveThirtyEight's K=20 has a HIGHER AUC than the tuned K=10 (0.7055 vs 0.7020) and a WORSE log loss (0.6300 vs 0.6260): the larger learning rate orders games better and prices them worse. This suite judges on the pricing question, so K=10 wins; a suite that judged on AUC would have taken the other one, and said so.

4. Bradley-Terry and SRS come out at 0.6929 and 0.6890 against a base rate of 0.6883 -- -0.0046, 95% CI [-0.0100, +0.0007] and -0.0008, 95% CI [-0.0024, +0.0009] respectively, where POSITIVE means better. Their AUCs (0.5514, 0.5020) say how much ordering they contribute over the 0.5 a coin flip gets. The cause is structural rather than a bug: a batch fit over a multi-season training block reports where a team has BEEN, and the next game depends on where it IS. Refitting every 25 games instead of 250 moves the pooled log loss by 0.002, which is noise here. Memory of the last few weeks is the one thing the walk-forward Elo has and these do not.

5. SRS as shipped cannot express a home edge: its mean prediction is 0.5000 on a league whose home teams win 56% of the time, because the ratings are centred on `margin - mean(margin)` and the calibration then centres the predicted margin again. The consequence is that it is blind to the home WIN RATE: hand it a league where home teams win 80% and it still prices the average home game at a coin flip, because its probabilities are anchored at 0.5 for any average margin and can only see a home edge as a change from the training average, never as a level. That is pinned in `Tests/test_baselines.py`. The `srs + intercept` rows put the intercept back, and they are the fair SRS.

6. The same-date question is free. Sequential and same-day-batched Elo are identical here because no team plays twice in a date, which makes the equality a proof rather than a measurement. Batching stays the default because its correctness does not depend on that schedule property continuing to hold.

7. Nothing here is a betting strategy. A prediction-quality improvement of this size is not an edge. The literature bar for claiming one is |z| > 3 after multiple-testing correction (Vandenbruaene et al. 2022, J Sports Econ 23(7), 628 implementations, null of unprofitability never rejected), and nothing in this file makes that claim.
* Log-loss SE is the standard error of the mean per-game log loss; it bounds uncertainty in the average, not in a difference between models. Differences use the paired bootstrap.
* Vandenbruaene et al. (2022, J Sports Econ 23(7)) surveyed 628 strategy implementations and never rejected the null of unprofitability after multiple-testing correction; their bar is |z| > 3. Nothing here is a betting system and no number here is evidence of profit.
