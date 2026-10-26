==============================================================================
ISSUE 71 -- AVAILABILITY SENSITIVITY GRID
==============================================================================
canonical reproduction 0.615663649 (recorded 0.615663649, drift +3.43e-10, ok=True)
close sample 5,788 games; default cell close 0.62033 (recorded 0.62034)

cell                         full LL   vs canon    |z|  avail eff    |z|  close LL
hl10_prior_game_topall      0.615887  -0.000223   2.68  +0.003680   4.65   0.62049
hl10_prior_game_top5        0.615232  +0.000431   0.50  +0.004335   3.95   0.61969
hl10_prior_game_top7        0.615467  +0.000197   0.28  +0.004100   3.95   0.61989
hl10_season_average_topall  0.618256  -0.002592   2.85  +0.001311   3.18   0.62253
hl10_season_average_top5    0.619021  -0.003357   3.64  +0.000546   1.20   0.62292
hl10_season_average_top7    0.619030  -0.003366   3.99  +0.000537   1.69   0.62283
hl15_prior_game_topall      0.615664  +0.000000   0.00  +0.003903   4.78   0.62033
hl15_prior_game_top5        0.614884  +0.000779   0.86  +0.004683   3.86   0.61947
hl15_prior_game_top7        0.615136  +0.000527   0.75  +0.004431   4.02   0.61967
hl15_season_average_topall  0.618425  -0.002761   3.02  +0.001142   2.98   0.62263
hl15_season_average_top5    0.619018  -0.003354   3.53  +0.000549   1.05   0.62297
hl15_season_average_top7    0.618977  -0.003313   3.88  +0.000590   1.56   0.62282
hl20_prior_game_topall      0.615598  +0.000066   1.30  +0.003969   4.77   0.62029
hl20_prior_game_top5        0.614700  +0.000964   1.04  +0.004867   3.86   0.61932
hl20_prior_game_top7        0.614960  +0.000703   0.99  +0.004607   4.09   0.61955
hl20_season_average_topall  0.618521  -0.002857   3.13  +0.001046   2.85   0.62268
hl20_season_average_top5    0.619048  -0.003384   3.49  +0.000519   0.91   0.62303
hl20_season_average_top7    0.618987  -0.003323   3.90  +0.000581   1.41   0.62285
hl30_prior_game_topall      0.615596  +0.000068   0.58  +0.003971   4.78   0.62029
hl30_prior_game_top5        0.614556  +0.001108   1.19  +0.005011   3.88   0.61921
hl30_prior_game_top7        0.614720  +0.000944   1.31  +0.004847   4.17   0.61934
hl30_season_average_topall  0.618610  -0.002947   3.25  +0.000957   2.72   0.62274
hl30_season_average_top5    0.619066  -0.003402   3.48  +0.000501   0.83   0.62309
hl30_season_average_top7    0.619085  -0.003421   4.05  +0.000482   1.15   0.62296

halflife_at_prior_all: 10.0: cell -0.000223 (|z| 2.68), avail +0.003680 (|z| 4.65) | 15.0: cell +0.000000 (|z| 0.00), avail +0.003903 (|z| 4.78) | 20.0: cell +0.000066 (|z| 1.30), avail +0.003969 (|z| 4.77) | 30.0: cell +0.000068 (|z| 0.58), avail +0.003971 (|z| 4.78)
squad_at_hl15_all: prior_game: cell +0.000000 (|z| 0.00), avail +0.003903 (|z| 4.78) | season_average: cell -0.002761 (|z| 3.02), avail +0.001142 (|z| 2.98)
top_n_at_hl15_prior: all: cell +0.000000 (|z| 0.00), avail +0.003903 (|z| 4.78) | 5: cell +0.000779 (|z| 0.86), avail +0.004683 (|z| 3.86) | 7: cell +0.000527 (|z| 0.75), avail +0.004431 (|z| 4.02)

beats recorded at |z| > 3: none
worse than recorded at |z| > 3: ['hl10_season_average_top5', 'hl10_season_average_top7', 'hl15_season_average_topall', 'hl15_season_average_top5', 'hl15_season_average_top7', 'hl20_season_average_topall', 'hl20_season_average_top5', 'hl20_season_average_top7', 'hl30_season_average_topall', 'hl30_season_average_top5', 'hl30_season_average_top7']

FRAGILE / BOUNDARY: the availability effect loses |z| > 3 in hl10_season_average_top5, hl10_season_average_top7, hl15_season_average_topall, hl15_season_average_top5, hl15_season_average_top7, hl20_season_average_topall, hl20_season_average_top5, hl20_season_average_top7, hl30_season_average_topall, hl30_season_average_top5, hl30_season_average_top7. Every failing cell is on the season_average roster axis and none is on prior_game, so halflife and minutes-count read as a plateau while the membership convention is load-bearing. The recorded point is an edge on the axis that fails; no new value is promoted.
