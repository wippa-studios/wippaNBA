==============================================================================
ISSUE 60 FOLLOW-ON -- RECENT-FORM WINDOWS (5 / 10), 2025-26 HOLDOUT
==============================================================================
train 11974 rows (< 2025-10-21); test 1225 rows (2025-26)

model                        log loss     brier  accuracy     auc
canonical 26 (holdout)       0.593740  0.203843    0.6669  0.7416
+ form5 ladders (36)         0.594360  0.204022    0.6735  0.7409
+ form10 ladders (46)        0.594504  0.204210    0.6678  0.7405

form5: delta -0.000620 CI [-0.002502, +0.001308] |z| 0.64
form10: delta -0.000764 CI [-0.003176, +0.001590] |z| 0.63
positive control (avail on vs off): +0.009247 |z| 4.93

per-side bracket coefficients (train fit, descriptive):
  form5:
    home_form5_ge1       -0.0169
    home_form5_ge2       -0.0048
    home_form5_ge3       -0.0012
    home_form5_ge4       -0.0301
    home_form5_ge5       -0.0276
    away_form5_ge1       +0.0066
    away_form5_ge2       -0.0155
    away_form5_ge3       +0.0501
    away_form5_ge4       +0.0288
    away_form5_ge5       +0.0162
  form10:
    home_form10_ge1      -0.0209
    home_form10_ge2      +0.0425
    home_form10_ge3      -0.0308
    home_form10_ge4      +0.0301
    home_form10_ge5      -0.0394
    home_form10_ge6      +0.0315
    home_form10_ge7      -0.0236
    home_form10_ge8      -0.0001
    home_form10_ge9      -0.0192
    home_form10_ge10     +0.0139
    away_form10_ge1      +0.0353
    away_form10_ge2      -0.0490
    away_form10_ge3      +0.0310
    away_form10_ge4      -0.0028
    away_form10_ge5      -0.0371
    away_form10_ge6      +0.0437
    away_form10_ge7      +0.0183
    away_form10_ge8      +0.0482
    away_form10_ge9      +0.0006
    away_form10_ge10     -0.0238

sensitivity (descriptive, cannot promote):
  form5_and_form10   log loss 0.595467  delta -0.001726  |z| 1.16
  form5_home_only    log loss 0.593707  delta +0.000033  |z| 0.05
  form5_away_only    log loss 0.594395  delta -0.000655  |z| 0.91

form5 NULL/NEGATIVE: point -0.000620 nats, CI [-0.002502, +0.001308], |z| 0.64.
form10 NULL/NEGATIVE: point -0.000764 nats, CI [-0.003176, +0.001590], |z| 0.63.
