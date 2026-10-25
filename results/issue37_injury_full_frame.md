==============================================================================
ISSUE 37 -- FULL-FRAME CRITERION MEASUREMENT
==============================================================================
games 13199; scored 6600; pre-tip coverage 2108/13199 (16.0%)
designations 18856 across 2108 games; on a prior-game squad 3642 across 1646 games
E14 reproduced 0.618307859 (recorded 0.618307119, drift +7.40e-07)

frame                            log loss     brier  accuracy     auc
E14 (43)                         0.618308  0.214597    0.6550  0.7093
E14 + availability (47)          0.614508  0.212862    0.6598  0.7148
E14 + injury (47)                0.613690  0.212536    0.6609  0.7157

availability vs E14            +0.003799 CI [+0.002296, +0.005486] |z| 4.67
injury vs E14                  +0.004618 CI [+0.002925, +0.006433] |z| 5.16
injury vs availability         +0.000818 CI [+0.000222, +0.001422] |z| 2.67
folds (injury vs availability): 5/5

NOT PROMOTED: the injury-adjusted frame is +0.000818 nats against the availability-only frame on the same 6600 scored rows (CI [+0.000222, +0.001422], |z| 2.67, better in 5/5 folds). The interval excludes zero, so this is not a null -- it is a positive below the pre-registered |z| > 3 bar. Against the 43-column bar the injury frame is +0.004618 (|z| 5.16), of which +0.003799 is the established availability block (|z| 4.67). Coverage is 2108/13199 games (16.0%) because only two seasons are backfilled. The criterion is satisfied; the feature is not promoted.
