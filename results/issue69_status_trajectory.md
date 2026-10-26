==============================================================================
ISSUE 69 -- INJURY STATUS TRAJECTORY, 2025-26 HOLDOUT
==============================================================================
test 1225 rows (2025-26); block columns ['mean_delta', 'n_downgrade', 'n_upgrade', 'sum_abs_delta', 'max_downgrade']
baseline (final status)    0.588977
candidate (+ trajectory)   0.588292
delta +0.000685 CI [-0.001419, +0.002768] |z| 0.64
positive control (final vs raw) +0.004763 |z| 3.33
2024-25 replication   -0.000191 |z| 0.15
top1 downgrade only    +0.000898 |z| 0.93
direction only         +0.000629 |z| 0.59

NULL on 2025-26: trajectory +0.000685 nats (CI [-0.001419, +0.002768], |z| 0.64); the intraday change is inside the price (or irrelevant). Positive control (final-injury vs raw) +0.004763 |z| 3.33.
