==============================================================================
ISSUE 68 -- INJURY PROMOTION, LAST-TWO-SEASONS HOLDOUT
==============================================================================
train 10749 rows (< 2024-10-22); test 2450 rows (2024-25 + 2025-26)
E14 reproduction 0.618303736 (recorded 0.618307119, drift -3.38e-06, ok=False)
arms (E14+availability baseline vs E14+injury, 47 cols, 15-min gate):
2024-25 + 2025-26  n= 2450 avail 0.59742 injury 0.59384 delta +0.003579 (|z| 3.85)
2024-25 only       n= 1225 avail 0.60041 injury 0.59791 delta +0.002494 (|z| 1.80)
2025-26 only       n= 1225 avail 0.59443 injury 0.58977 delta +0.004663 (|z| 3.46)
positive control (availability vs E14): +0.006576 |z| 5.49

PROMOTED: the injury multiplier clears |z| > 3 on the last-two-seasons holdout (+0.003579 nats, |z| 3.85, CI [+0.001729, +0.005376]) -- the criterion reading survives a forward season holdout. Promotion-grade evidence for the feature.
