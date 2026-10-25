==============================================================================
ISSUE 56 -- TIME-ZONE / TRAVEL FEATURES ON THE CANONICAL FRAME
==============================================================================
games 13199; scored 6600; canonical columns 26; tz columns ['tz_offset_diff', 'away_tz_crossed']
canonical reproduced 0.615663649 (recorded 0.615663649, drift +3.43e-10)

model                      log loss     brier  accuracy     auc
canonical (26)             0.615664  0.213333    0.6573  0.7141
+ tz block (28)            0.615628  0.213312    0.6561  0.7142

delta (tz - canonical)      +0.000035 CI [-0.000228, +0.000272] |z| 0.28
folds better: 2/5

sensitivity (descriptive, cannot promote):
  tz_offset_only     n=27 log loss 0.615691 delta -0.000027 |z| 0.65
  crossed_only       n=27 log loss 0.615648 delta +0.000015 |z| 0.14

NULL: the tz block's point estimate is +0.000035 nats against the canonical model (CI [-0.000228, +0.000272], |z| 0.28, better in 2/5 folds). The interval includes zero, so this is not evidence of a difference either way; positive is not promoted, and stage 2 is not run.
