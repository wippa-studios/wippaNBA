==============================================================================
ISSUE 37 -- TIMESTAMPED INJURY REPORTS (2020-21)
==============================================================================
games 1080; scored 540; pre-tip coverage 964/1080 (89.3%)
events resolved 11955/12071 (99.0%); ambiguous 0
designations applied 7529 across 964 games; on a prior-game squad 1499 across 704 games
statuses on prior squad: {'Out': 612, 'Questionable': 421, 'Probable': 302, 'Available': 93, 'Doubtful': 71}

arm                                  cand LL    ref LL      delta       z   folds  covered d
e14_plus_injury                     0.642338  0.641879  -0.000459   -0.09 3/ 5  -0.001340
canonical_replace                   0.630708  0.631468  +0.000760    0.24 3/ 5  +0.000386
canonical_plus                      0.633205  0.631468  -0.001736   -0.93 1/ 5  -0.002111
covered-only n = 501

positive control (availability block)  -0.001195 |z| 0.22
placebo delta mean +0.001228, sd 0.001095, range [-0.000136, +0.003886]; true assignment beats 55% of shuffles
buffer 30 min: coverage 964, delta +0.000760 |z| 0.24

NULL: no injury arm clears its reference model at |z| > 3. The strongest arm is canonical_replace (+0.000760, |z| 0.24; covered-only +0.000386, |z| 0.11). This is an absence of evidence, not evidence that the designations are uninformative: the placebo spread is 0.001095 nats on 540 scored games.
