==============================================================================
ISSUE 58 -- VARIANCE-CONDITIONED STAKING
==============================================================================
games with line 5788; baseline breakeven: 4,822 bets, ROI -8.71% [recorded 4,828 / -8.23%]
dispersion (residual bootstrap, B=50): mean 0.0237, sd 0.0067; corr(dispersion, wrong-bet) -0.050; win rate low/mid/high tercile 0.391/0.384/0.402

gate rules (bet only when dispersion percentile <= gate):
  gate_25  n=1205 ROI   -15.27%  delta-vs-base    -6.19pp  |z| 0.58
  gate_50  n=2411 ROI   -11.94%  delta-vs-base    -3.25pp  |z| 0.60
  gate_75  n=3616 ROI    -7.05%  delta-vs-base    +1.61pp  |z| 0.58
sized rules (descriptive):
  stake ~ 1-var_pctile : ROI -10.40% (n 4821)
  stake ~ 0.3kelly(1-var): ROI -19.84% (n 4243)

NULL: variance conditioning does not rescue the rule. The median gate moves ROI by -0.0325 vs the -8.71% baseline (|z| 0.60), below the bar; the win-rate-by-variance-tercile diagnostic shows whether uncertainty even flags where the model is wrong. No promotion.
