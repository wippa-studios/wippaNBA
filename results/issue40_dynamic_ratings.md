==============================================================================
ISSUE 40 -- DYNAMIC RATINGS, WALK-FORWARD
==============================================================================
n scored 6600; home edge fixed at 70 Elo points, never learned into team strength

model              logloss    brier     auc     acc  mean p
canonical         0.615664 0.213333  0.7141  0.6573  0.5641
e14               0.618308 0.214597  0.7093  0.6550  0.5505
glicko            0.630953 0.220022  0.6979  0.6442  0.5772
state_space_bt    0.628757 0.219150  0.6977  0.6485  0.5641

DELTAS (positive = dynamic model better)
glicko           vs canonical -0.015289 [-0.020766,-0.009833] |z| 5.48; vs E14 -0.012645 [-0.017804,-0.007365] |z| 4.75
                 folds better than canonical: 0/5
state_space_bt   vs canonical -0.013093 [-0.018542,-0.007764] |z| 4.76; vs E14 -0.010449 [-0.015575,-0.005317] |z| 3.99
                 folds better than canonical: 0/5

UNCERTAINTY TERM
glicko           {'rd_home_mean': 91.18566469356821, 'rd_away_mean': 91.13932658244515, 'rd_home_median': 82.03785173921057, 'rd_min': 71.21264001692127, 'rd_max': 350.0}
state_space_bt   {'var_home_mean': 0.35896427388525526, 'var_away_mean': 0.3571129768117071, 'var_home_median': 0.2537703729018857, 'var_min': 0.20401670834544303, 'var_max': 5.971176992570541}

verdict glicko: NULL: glicko does not beat the same-row canonical model at |z| > 3 (delta -0.015289, |z| 5.48).
verdict state_space_bt: NULL: state_space_bt does not beat the same-row canonical model at |z| > 3 (delta -0.013093, |z| 4.76).
