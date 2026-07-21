# Balance-based threshold selection

The validation policy is selected without using the outcome. For each dataset,
we run a prespecified grid of matching thresholds, calculate absolute
standardized mean differences (SMDs) for every matching covariate, and rank
eligible configurations by the smallest RMS SMD. A configuration is
eligible only when at least 95% of the original treated units occur in a
matched set. ATT, bias, potential outcomes, and published effect estimates are
reserved for the subsequent evaluation and do not enter tuning or ranking.

## Common grid

The propensity-score caliper is fixed a priori at 0.20 (standardized logit scale).
The Gower/RDMatcher grid is 0.15 through 0.45 in 0.05 increments. The
Mahalanobis/RDMatcher grid is 2.50 through 4.50 in 0.05 increments. RDMatcher uses 250 control candidates,
no replacement, global completion, and control-reference SD-based Gower
weights (multiplier 1.96) for its Gower methods.

The four RDMatcher policies are searched as follows: RDM searches Gower;
PSM+RDM searches each Gower value at the fixed 0.20 caliper; Maha (RDM) searches
Mahalanobis; and PSM+Maha (RDM) searches each Mahalanobis value at the fixed
0.20 caliper. The MatchIt comparators, plus PSM (RDM) and scaled-Euclidean
MatchIt, are included in the same validation
grid: PSM (MatchIt) and PSM+Maha (MatchIt) use the fixed 0.20 caliper, while
Maha (MatchIt) is evaluated once without a caliper.
MatchIt does not provide an unnamed overall Mahalanobis-distance cutoff;
calipers must be attached to named observed variables. Their matching
definitions remain MatchIt nearest-neighbor matching without replacement.

## Reproducible scripts

`tune_lalonde_balance.py` applies the grid to the CPS LaLonde dataset and
writes `plots/lalonde_balance_threshold_tuning.csv`. `tune_ihdp_balance.py`
applies the identical grid to IHDP replications and writes chunk files that
can be combined with `--aggregate` into
`plots/ihdp_balance_threshold_tuning.csv`. Use `--start`/`--stop` (and, for
IHDP, `--start-rep`/`--n-reps`) to divide a run into independent chunks.

The final LaLonde ATT plot uses the policies documented in
`LALONDE_POLICY_SELECTION.md`; its outcome metrics are an evaluation of the
balance-selected policies, not the selection criterion.
