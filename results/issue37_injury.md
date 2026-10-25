==============================================================================
ISSUE 37 -- TIMESTAMPED INJURY REPORTS (2021-22)
==============================================================================
games 1230; scored 615; pre-tip coverage 1144/1230 (93.0%)
events resolved 39077/39524 (98.9%); ambiguous 0
designations applied 11327 across 1144 games; on a prior-game squad 2097 across 916 games
statuses on prior squad: {'Out': 1076, 'Questionable': 557, 'Probable': 212, 'Available': 201, 'Doubtful': 51}

arm                                  cand LL    ref LL      delta       z   folds  covered d
e14_plus_injury                     0.642582  0.647201  +0.004619    0.66 3/ 5  +0.002302
canonical_replace                   0.634950  0.638818  +0.003868    0.96 3/ 5  +0.004068
canonical_plus                      0.637584  0.638818  +0.001235    0.26 3/ 5  +0.001033
covered-only n = 567

positive control (availability block)  +0.004801 |z| 0.92
placebo delta mean +0.004379, sd 0.001313, range [+0.002061, +0.006552]; true assignment beats 65% of shuffles
buffer 30 min: coverage 1144, delta +0.003868 |z| 0.96

NULL: no injury arm clears its reference model at |z| > 3. The strongest arm is e14_plus_injury (+0.004619, |z| 0.66; covered-only +0.002302, |z| 0.32). This is an absence of evidence, not evidence that the designations are uninformative: the placebo spread is 0.001313 nats on 615 scored games.
