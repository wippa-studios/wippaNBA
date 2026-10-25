==============================================================================
ISSUE 39 -- TRUE RAPM, 2020-21, WALK-FORWARD
==============================================================================
rotation games in frame       1080
games usable (5v5, PT_DIFF)   724
games partial / missing       4
season games never downloaded 352
stint rows / intervals         46108 / 21492
players                        521
outer test rows scored         362

PER FOLD (alpha chosen inside the training block)
 k train  test     alpha  theta sd  hca/48  cal a  logloss
 1   362    72    100000    1.8896   1.524  0.330   0.6913
   inner grid 30:0.7100, 100:0.7046, 300:0.7008, 1000:0.6971, 3000:0.6939, 10000:0.6915, 30000:0.6903, 100000:0.6897, 300000:0.6901, 1000000:0.6917
 2   434    72    100000    2.0041   1.530  0.426   0.6808
   inner grid 30:0.6895, 100:0.6873, 300:0.6857, 1000:0.6836, 3000:0.6809, 10000:0.6774, 30000:0.6744, 100000:0.6727, 300000:0.6733, 1000000:0.6764
 3   506    73    300000    1.0476   1.575  0.344   0.6417
   inner grid 30:0.6891, 100:0.6896, 300:0.6897, 1000:0.6893, 3000:0.6879, 10000:0.6857, 30000:0.6837, 100000:0.6820, 300000:0.6815, 1000000:0.6827
 4   579    72    100000    2.2161   1.707  0.473   0.6653
   inner grid 30:0.6817, 100:0.6819, 300:0.6825, 1000:0.6829, 3000:0.6819, 10000:0.6798, 30000:0.6787, 100000:0.6784, 300000:0.6790, 1000000:0.6814
 5   651    73    100000    2.2285   1.480  0.831   0.6186
   inner grid 30:0.6907, 100:0.6893, 300:0.6880, 1000:0.6862, 3000:0.6836, 10000:0.6796, 30000:0.6769, 100000:0.6763, 300000:0.6776, 1000000:0.6811

POOLED, SAME ROWS
model                            n   logloss    brier     auc     acc
rapm                           362  0.659381 0.233313  0.6444  0.5994
e14 same season                362  0.668171 0.235238  0.6500  0.6188
canonical same season          362  0.651609 0.230161  0.6542  0.6188

delta vs E14 (positive = RAPM better)      +0.008790 [-0.026419, +0.046987] |z| 0.47
delta vs canonical (positive = RAPM better) -0.007773 [-0.036365, +0.020416] |z| 0.54
folds RAPM better than E14: 2/5
folds RAPM better than canonical: 2/5

verdict: NULL / INCONCLUSIVE: true RAPM does not beat the same-season canonical model at |z| > 3, and its interval spans zero against the same-season E14 frame.
