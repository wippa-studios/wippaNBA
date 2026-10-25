==============================================================================
M01 -- FEED-FORWARD NETWORK ON THE CANONICAL 26-FEATURE FRAME
==============================================================================
games 13199; scored 6600; canonical columns 26
architecture (64, 32); seeds [42, 43, 44, 45, 46]; alpha 0.0001
canonical reproduced 0.615663649 (recorded 0.615663649, drift +3.43e-10)

model                          log loss     brier  accuracy     auc
canonical logistic             0.615664  0.213333    0.6573  0.7141
MLP, 5-seed average            0.616655  0.213815    0.6617  0.7122

delta (MLP - canonical)      -0.000991 CI [-0.002946, +0.000837] |z| 1.03
folds better: 2/5; seed delta sd 0.001537

per seed (descriptive):
  42  log loss 0.620653  delta -0.004990  |z| 3.02
  43  log loss 0.622965  delta -0.007301  |z| 4.96
  44  log loss 0.624697  delta -0.009034  |z| 5.05
  45  log loss 0.621798  delta -0.006135  |z| 3.68
  46  log loss 0.621736  delta -0.006073  |z| 3.47

sensitivity (descriptive, cannot promote):
  one_hidden_32        log loss 0.620735  delta -0.005072  |z| 3.19
  wide_128_64          log loss 0.620869  delta -0.005205  |z| 3.59
  strong_l2_1e-2       log loss 0.622400  delta -0.006737  |z| 3.99
  no_early_stopping    log loss 1.181548  delta -0.565884  |z| 27.72

NULL/NEGATIVE: the five-seed MLP does not beat the canonical model. Its point estimate is -0.000991 nats (CI [-0.002946, +0.000837], |z| 1.03), better in 2/5 folds. The interval includes zero, so this is not evidence of a difference either way; it is certainly not an improvement. A non-linear model class adds no measurable signal on this frame.
