==============================================================================
ISSUE 57 -- SPREAD-COVER BETTING OBJECTIVE
==============================================================================
games with sane spread 5788; predictions from c44 cache
line               P(home cover)   n bets       ROI    |z| cover rate
gaussian_offset_+0.0       0.4996     5788    -6.00%   4.79     0.4881
skellam_offset_+0.0       0.4997     5788    -5.97%   4.76     0.4881
gaussian_offset_-3.5       0.5945     5788   +10.82%   8.67     0.6035
skellam_offset_-3.5       0.5870     5788   +10.86%   8.71     0.6035
gaussian_offset_+3.5       0.4001     5788   +12.42%  10.19     0.3659
skellam_offset_+3.5       0.4081     5788   +12.45%  10.22     0.3659
gaussian_offset_-7.5       0.6987     5788   +35.51%  30.92     0.7134
skellam_offset_-7.5       0.6838     5788   +35.51%  30.92     0.7134
gaussian_offset_+7.5       0.2966     5788   +37.52%  33.34     0.2607
skellam_offset_+7.5       0.3118     5788   +37.52%  33.34     0.2607
sigma dimension (posted Gaussian line):
  gate_low_half    n=2894 ROI -6.76%
  size_1_minus_pct n=5787 ROI -6.68%

SIGNIFICANTLY NEGATIVE: the posted-line cover pick returns -6.00% at a vig-less -110 (|z| 4.79, CI ['-8.47%', '-3.59%']) -- the closing spread is efficient against exactly this model: its favourite covers only ~49% of the time. The alternate-line sweep (up to +37% at +/-7.5) is the same games re-scored against lines the market never offered -- a bettor cannot take them; it shows the margin distribution is wide, not that the line is wrong.
