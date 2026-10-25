==============================================================================
ISSUE 41 -- HIERARCHICAL PLAYER-TO-TEAM RATINGS
==============================================================================
candidate panel rows 403305; scored games 6600
pooling strength selected inside each outer training fold; expected roster only

PER FOLD
 k train  test   alpha  theta sd  cal a   cal b  logloss
 1  6599  1320    3.00    4.0754  0.110  -0.073   0.6624
   inner grid 0.3:0.6721, 1.0:0.6661, 3.0:0.6657, 10.0:0.6714, 30.0:0.6792, 100.0:0.6856, 300.0:0.6885
 2  7919  1320    1.00    6.4521  0.088   0.029   0.6499
   inner grid 0.3:0.6746, 1.0:0.6680, 3.0:0.6707, 10.0:0.6763, 30.0:0.6808, 100.0:0.6844, 300.0:0.6861
 3  9239  1320    1.00    6.6721  0.106   0.061   0.6476
   inner grid 0.3:0.6599, 1.0:0.6578, 3.0:0.6607, 10.0:0.6678, 30.0:0.6751, 100.0:0.6808, 300.0:0.6833
 4 10559  1320    1.00    6.9055  0.113   0.039   0.6448
   inner grid 0.3:0.6570, 1.0:0.6544, 3.0:0.6591, 10.0:0.6663, 30.0:0.6737, 100.0:0.6803, 300.0:0.6834
 5 11879  1320    1.00    7.2118  0.118  -0.059   0.6548
   inner grid 0.3:0.6687, 1.0:0.6605, 3.0:0.6622, 10.0:0.6692, 30.0:0.6770, 100.0:0.6838, 300.0:0.6870

POOLED, SAME ROWS
model                        n   logloss    brier     auc     acc
hierarchical              6600  0.651898 0.229994  0.6522  0.6168
e14                       6600  0.618308 0.214597  0.7093  0.6550
canonical                 6600  0.615664 0.213333  0.7141  0.6573

delta vs E14 (positive = hierarchical better)       -0.033590 [-0.040719,-0.026468] |z| 9.24
delta vs canonical (positive = hierarchical better) -0.036234 [-0.043901,-0.029100] |z| 9.60

verdict: NULL: hierarchical partial pooling does not beat the same-row canonical model at |z| > 3, nor the 43-column frame. At this sample size the team-level aggregates already carry the recoverable signal.
