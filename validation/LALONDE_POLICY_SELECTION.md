# LaLonde policy selection

RDMatcher policies were selected on covariate balance only. The LaLonde outcome and ATT estimate were not used in the threshold search or in policy ranking.

The propensity-score caliper was fixed at 0.20 on the standardized logit scale
for every propensity-restricted method. For each RDMatcher method, the
balance-only search retained configurations with treated retention of at least
0.95 and maximum absolute SMD no greater than 0.20, then selected the smallest
RMS SMD across covariates.

| Method | Selected policy | Mean absolute SMD | Retention |
| --- | --- | ---: | ---: |
| RDM | Gower threshold 0.30; 250 candidates | 0.0321 | 100.0% |
| PSM+RDM | Gower threshold 0.35; standardized-logit caliper 0.20; 250 candidates | 0.0405 | 100.0% |
| Maha (RDM) | Mahalanobis threshold 3.15; 250 candidates | 0.0423 | 100.0% |
| PSM+Maha (RDM) | Mahalanobis threshold 3.60; standardized-logit caliper 0.20; 250 candidates | 0.0297 | 100.0% |

The final LaLonde plot and CSV use these policies for ATT estimation. The
complete expanded balance-only search is saved in
`plots/lalonde_balance_threshold_tuning.csv` and can be reproduced
with `tune_lalonde_balance.py`.
