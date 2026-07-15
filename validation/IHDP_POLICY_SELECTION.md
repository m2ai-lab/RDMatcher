# IHDP policy selection

Policies were selected using the prespecified balance-only grid in
`README.md`: the propensity-score caliper was fixed a priori at 0.20;
configurations were required to have mean treated retention of at least 0.80
and mean maximum absolute SMD no greater than 0.20, and the smallest RMS SMD
was selected. No outcome, ATT, or bias metric was used.

The selected policies are RDM (Gower 0.30), PSM+RDM (Gower 0.15, propensity
caliper 0.20), Maha (RDM) (Mahalanobis 4.50), PSM+Maha (RDM) (Mahalanobis
5.00, caliper 0.20), PSM (MatchIt) (caliper 0.20), Maha (MatchIt), and
PSM+Maha (MatchIt) (caliper 0.20). The refined Mahalanobis grid was 4.00
through 6.00 in 0.25 increments.
