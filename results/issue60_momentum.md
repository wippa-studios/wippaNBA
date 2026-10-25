==============================================================================
ISSUE 60 -- MOMENTUM STREAK BRACKETS, 2025-26 HOLDOUT
==============================================================================
train 11974 rows (< 2025-10-21); test 1225 rows (2025-26)
brackets [2, 3, 4, 6]; columns canonical 26 + 8 momentum

model                          log loss     brier  accuracy     auc
canonical 26 (holdout)         0.593740  0.203843    0.6669  0.7416
+ momentum 34 (holdout)        0.593480  0.203649    0.6718  0.7423
availability off 22 (control)   0.602987  0.207735    0.6694  0.7297

delta (momentum - canonical)  +0.000260 CI [-0.001431, +0.002066] |z| 0.29
positive control (avail on vs off): +0.009247 |z| 4.93

per-bracket coefficients (train fit, descriptive):
  home_streak_ge2        -0.0357
  home_streak_ge3        +0.0658
  home_streak_ge4        -0.0516
  home_streak_ge6        -0.0225
  away_streak_ge2        +0.0444
  away_streak_ge3        -0.0135
  away_streak_ge4        +0.0222
  away_streak_ge6        -0.0025

sensitivity (descriptive, cannot promote):
  home_only    log loss 0.593589  delta +0.000152  |z| 0.22
  away_only    log loss 0.593596  delta +0.000144  |z| 0.27
  loss_streaks log loss 0.594659  delta -0.000919  |z| 0.83

NULL on 2025-26: the streak brackets' point estimate is +0.000260 nats against the canonical model (CI [-0.001431, +0.002066], |z| 0.29). The interval includes zero, so there is no evidence the brackets help on the season that would actually be bet; positive is not promoted.
