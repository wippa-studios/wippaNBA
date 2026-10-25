==============================================================================
ISSUE 37 -- FULL-FRAME CRITERION MEASUREMENT
==============================================================================
games 13199; scored 6600; pre-tip coverage 7206/13199 (54.6%)
designations 71393 across 7206 games; on a prior-game squad 15114 across 6007 games
E14 reproduced 0.618307859 (recorded 0.618307119, drift +7.40e-07)

frame                            log loss     brier  accuracy     auc
E14 (43)                         0.618308  0.214597    0.6550  0.7093
E14 + availability (47)          0.614508  0.212862    0.6598  0.7148
E14 + injury (47)                0.611876  0.211770    0.6624  0.7182

availability vs E14            +0.003799 CI [+0.002296, +0.005486] |z| 4.67
injury vs E14                  +0.006432 CI [+0.004433, +0.008541] |z| 6.14
injury vs availability         +0.002632 CI [+0.001471, +0.003779] |z| 4.47
folds (injury vs availability): 5/5

INJURY ADJUSTMENT CLEARS THE AVAILABILITY BLOCK on the full frame: +0.002632 nats, |z| 4.47. Coverage is 7206/13199 games (54.6%); 26.4% of scored rows are identical between the arms, so the delta is diluted by construction. This is a criterion result, not a promotion.
