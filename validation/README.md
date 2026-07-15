# Balance-based threshold selection

The validation policy is selected without using the outcome. For each dataset,
we run a prespecified grid of matching thresholds, calculate absolute
standardized mean differences (SMDs) for every matching covariate, and rank
eligible configurations by the smallest mean absolute SMD. A configuration is
eligible only when at least 90% of the original treated units occur in a
matched set. ATT, bias, potential outcomes, and published effect estimates are
reserved for the subsequent evaluation and do not enter tuning or ranking.

## Common grid

The propensity-score caliper is fixed a priori at 0.20 (standardized logit scale).
The Gower/RDMatcher grid is 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25,
0.275, and 0.30. The Mahalanobis/RDMatcher grid is 1.0, 1.25, 1.5, 1.75,
2.0, 2.25, 2.5, 2.75, and 3.0. RDMatcher uses 250 control candidates,
no replacement, global completion, and control-reference SD-based Gower
weights (multiplier 1.96) for its Gower methods.

The four RDMatcher policies are searched as follows: RDM searches Gower;
PSM+RDM searches each Gower value at both calipers; Maha (RDM) searches
Mahalanobis; and PSM+Maha (RDM) searches each Mahalanobis value at both
calipers. The three MatchIt comparators are included in the same validation
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
