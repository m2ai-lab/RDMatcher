# LaLonde policy selection

RDMatcher policies were selected on covariate balance only. The LaLonde outcome and ATT estimate were not used in the threshold search or in policy ranking.

The propensity-score caliper was fixed a priori at 0.20 on the standardized
logit scale for propensity-restricted methods. For each RDMatcher method, the
balance-only search retained configurations with treated retention of at least
0.90 and maximum absolute SMD no greater than 0.20, then selected the smallest
RMS SMD across covariates.

| Method | Selected policy | Mean absolute SMD | Retention |
| --- | --- | ---: | ---: |
| RDM | Gower threshold 0.20; 250 candidates | 0.0321 | 100.0% |
| PSM+RDM | Gower threshold 0.275; standardized-logit caliper 0.20; 250 candidates | 0.0311 | 100.0% |
| Maha (RDM) | Mahalanobis threshold 2.75; 250 candidates | 0.0467 | 100.0% |
| PSM+Maha (RDM) | Mahalanobis threshold 2.25; standardized-logit caliper 0.20; 250 candidates | 0.0374 | 99.5% |

The final LaLonde plot and CSV use these policies for ATT estimation. The
complete expanded balance-only search is saved in
`plots/lalonde_balance_threshold_tuning_expanded.csv` and can be reproduced
with `tune_lalonde_balance.py`.
