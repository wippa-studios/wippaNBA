==============================================================================
ISSUE 59 -- EARLY-SEASON MARKET DRIFT
==============================================================================
games with line 5788; reproduction: market 0.59766 (recorded 0.59766), model+avail 0.62033 (recorded 0.62034)
combined breakeven rule: 4,822 bets, ROI -8.71% (|z| 1.65) [recorded 4,828 bets / -8.23%]
primary ('season_depth <= 15'):
  n early 2473 late  3315 | market ll early 0.60813 late 0.58984 | gap +0.01852 (|z| 1.77) | bet early -6.05% (n 2022) late -10.90% (n 2800)
sensitivity:
  n early 1718 late  4070 | market ll early 0.60267 late 0.59554 | gap +0.00730 (|z| 0.65) | bet early -8.95% (n 1396) late -8.61% (n 3426)
  n early 3156 late  2632 | market ll early 0.61219 late 0.58023 | gap +0.03202 (|z| 2.99) | bet early -2.09% (n 2593) late -17.43% (n 2229)
  n early  773 late  5015 | market ll early 0.61637 late 0.59477 | gap +0.02138 (|z| 1.46) | bet early +2.87% (n 639) late -10.79% (n 4183)

FLAT: the market log loss is +0.01852 nats worse early (|z| 1.77, CI [-0.0028522823683491626, 0.03838632496130026]), below the |z| > 3 bar. The claim is not supported; any early-season effect is noise on this sample.
