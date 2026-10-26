==============================================================================
ISSUE 70 -- NULL META-REPLICATION, 2024-25 HOLDOUT
==============================================================================
train 10749 rows (< 2024-10-22); test 1225 rows (2024-25)
canonical reproduction 0.615663649 (recorded 0.615663649, drift +3.43e-10, ok=True)
reference (canonical 26) log loss 0.601833

momentum  + 8 cols  log loss 0.601799  delta +0.000033 CI [-0.001691, +0.001983] |z| 0.04
form5     +10 cols  log loss 0.602060  delta -0.000227 CI [-0.002271, +0.001800] |z| 0.22
form10    +20 cols  log loss 0.602950  delta -0.001117 CI [-0.003649, +0.001390] |z| 0.87
tz        + 2 cols  log loss 0.602058  delta -0.000225 CI [-0.000934, +0.000445] |z| 0.64

availability control (on - off) +0.004787 |z| 2.68  clears=False

NULL (underpowered): momentum reads +0.000033 nats, |z| 0.04 on 2024-25, consistent with the recorded +0.000260 (|z| 0.29).
NULL (underpowered): form5 reads -0.000227 nats, |z| 0.22 on 2024-25, consistent with the recorded -0.000620 (|z| 0.64).
NULL (underpowered): form10 reads -0.001117 nats, |z| 0.87 on 2024-25, consistent with the recorded -0.000764 (|z| 0.63).
NULL (underpowered): tz reads -0.000225 nats, |z| 0.64 on 2024-25, consistent with the recorded +0.000035 (|z| 0.28).

INCONCLUSIVE: the availability positive control did not clear (|z| > 3) on 2024-25, so this season cannot resolve an availability-sized effect. The nulls above are underpowered, not evidence of stability.
