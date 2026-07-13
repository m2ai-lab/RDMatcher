# RDMatcher

RDMatcher is a Python package for population matching and causal inference analysis in observational data. It emphasizes robust, practical matching for rare‑exposure (or rare‑outcome) scenarios using mixed‑data Gower distance and a staged matching strategy that combines fast greedy allocation with globally optimal assignment on a reduced problem.

## Table of Contents
- [Overview](#overview)
- [Tested implementations](#tested-implementations)
- [Quickstart (Gower default)](#quickstart-gower-default)
- [RDMatcher class (API)](#rdmatcher-class-api)
  - [Constructor](#constructor)
  - [Key methods and attributes](#key-methods-and-attributes)
  - [Outputs](#outputs)
- [rare_matching with Gower (usage & options)](#rare_matching-with-gower-usage--options)
- [Math: Gower distance (implementation details)](#math-gower-distance-implementation-details)
  - [Computational complexity: linear vs superlinear phases](#computational-complexity-linear-vs-superlinear-phases)
- [Addendum: Propensity score (optional)](#addendum-propensity-score-optional)
- [Mahalanobis distance (numeric-only)](#mahalanobis-distance-numeric-only)
- [PSM+RDM: Propensity‑Score‑Calibrated Gower Matching](#psmrdm-propensityscorecalibrated-gower-matching)
- [Advanced topics & troubleshooting](#advanced-topics--troubleshooting)
- [Installation & dependencies](#installation--dependencies)
- [Authors & License](#authors--license)

## Overview
RDMatcher provides tools to:
- Preprocess numeric / categorical / datetime features (optional)
- Compute propensity logits (optional) via several strategies (bagging, downsampling, hard‑negative mining)
- Match exposed subjects to control subjects using a staged algorithm:
  - Candidate prefiltering via k‑nearest neighbors (Gower or numeric metrics)
  - Identification of "safe" vs "competitive" controls
  - Competitive greedy allocation for limited exposed subjects
  - Global optimal assignment (Hungarian) on the reduced bipartite problem

This design reduces the effective search space for the expensive global solver and makes the approach practical for many real datasets where cases are relatively few.

## Tested implementations
- Distances: Gower (mixed‑type; implemented as GowerKNN), Euclidean, Cosine, Mahalanobis
- Mahalanobis candidate search: original full-distance backend and optional whitened sklearn `NearestNeighbors` backend
- Matching pipeline: batched neighbor prefiltering, safe/competitive categorization, greedy competitive allocation, global optimal assignment (Hungarian) on the reduced problem
- Sparse global solver: optional min‑cost‑flow (mcf) implementation using OR‑Tools for very large sparse instances
- Propensity modeling: `fit_propensity_model` wrapper (includes propensity utilities supporting downsampling, bagging, hard‑negative mining)
- Diagnostics: SMD summary table and feature balance plotting utilities

## Quickstart (Gower default)

Note: For Gower, keep categorical columns as pandas object/category dtype and do not one‑hot encode them. If you plan to use Euclidean or Cosine distances instead, convert categorical variables to numeric (e.g., one‑hot) via the preprocessing pipeline or externally.

```python
import pandas as pd
from rdmatcher import RDMatcher

# Load the full population DataFrame (must include the exposure column coded 0/1)
pop_df = pd.read_csv("population.csv")

# Initialize RDMatcher (Gower‑focused defaults)
matcher = RDMatcher(
    pop_df=pop_df,
    patient_id_col='patient_id',
    exposure_status='exposure_status',
    features_numeric=['age', 'bmi', 'lab_value'],
    features_categorical=['sex', 'race_ethnicity'],
    features_datetime=['first_visit_date'],
    process_features=False,   # keep raw types for Gower
    onehot=False,             # IMPORTANT: For Gower, avoid one‑hot encoding
    debug=False
)


# Run rare matching using Gower distance
matcher.rare_matching(
    threshold=0.25,
    n_neighbors=1,
    k_candidates=500,
    distance_metric='gower',     # default: Gower for mixed data
    gower_sd_weights=True,       # optional: SD-calibrated numeric weights
    gower_sd_reference='controls',  # default donor-only scaling
    gower_sd_weights_mult=1.96,  # default: make ~2 SD shifts comparable
    global_optimal=True,
    replacement=False,
    competitive_match=True,
    diagnostics=True
)

# Results
matched_df = matcher.pop_matched
summary = matcher.summary_table
```

## RDMatcher class (API)
### Constructor
- `pop_df`: pandas DataFrame containing both exposed and control samples (exposure indicated by exposure_status)
- `patient_id_col`: unique identifier column name (must be unique; duplicates raise an error)
- `exposure_status`: column name containing 0/1 exposure indicator (0=control, 1=exposed)
- `features_numeric`, `features_categorical`, `features_datetime`, `features_log`, `features_bin`
- `process_features` (bool): run internal preprocessing pipeline
- `onehot` (bool): whether to one-hot encode categorical features (default False; do not use with Gower)

### Key methods and attributes
- `process_features()`: RDMatcher provides an internal preprocessing pipeline, but for Gower distance the recommended workflow is to keep categorical columns as object/category dtype and not one‑hot encode them.
  - The package exposes `build_preprocessing_pipeline` and `apply_preprocessing_pipeline` for explicit transformations (scaling, log transforms, binning, and optional one‑hot encoding).
  - If you use Euclidean or Cosine metrics, categorical features must be converted to numeric (e.g., one‑hot). You can either preprocess externally or call `process_features()` with `onehot=True`.
- `fit_propensity_model(formula: Optional[str]=None, random_state=404, \*\*kwargs)`: compute `propensity_score` and `propensity_logit` and merge scalar columns back into the population view. This is an optional component to the matching function. It is implemented to provide another feature to match on, if desired. The `propensity_logit` column can be individually weighted with the `gower_weights` dictionary.
- `rare_matching(...)`: main matching routine (see next section)

After execution the object exposes useful attributes:
- `pop`: combined raw population DataFrame
- `pop_processed`: preprocessed DataFrame (if process_features=False this will be a copy of pop, after datetime conversion if applicable)


### Outputs
- `pop_matched` (DataFrame): merged matched table; includes at minimum the patient id, exposure status, `match_group`, `n_matches`, and `match_distance`. If propensity logits were computed, `propensity_logit` will also be included.
- `matched_exposed` / `matched_control` (DataFrames): subsets of `pop_matched` by exposure status
- `unmatched_exposed` (DataFrame): exposed subjects that were not matched
- `summary_table`: diagnostics table of SMDs after matching

## `rare_matching` with Gower (usage & options)

- Core idea: find k nearest candidate controls for each exposed subject using a fast neighbor search (GowerKNN for mixed data). Candidates within `threshold` are classified into:
  - safe: candidate assigned only to one exposed subject (fast, deterministic)
  - competitive: candidate shared across multiple exposed subjects (requires more careful allocation)

- Phases
  1. Prefilter: kneighbors to get top `k_candidates` and distances
  2. Categorize safe vs competitive via vectorized usage counts
  3. Competitive allocation: greedy, deterministic assignment for limited subjects
  4. Global optimal: Hungarian algorithm on the reduced bipartite graph (only competitive leftover subjects)

- Important parameters (selected)
  - `threshold`: maximum allowable distance for a match
  - `n_neighbors`: number of matches per exposed subject
  - `k_candidates`: how many nearest controls to consider per exposed subject
  - `global_optimal`: whether to run the Hungarian solver in the final phase
  - `competitive_match`: whether to run the competitive allocation phase
  - `distance_metric`: `'gower'` (mixed data), `'euclidean'`, or `'cosine'`
  - Categorical semantics: columns listed in `features_categorical` when constructing `RDMatcher` are treated as nominal categorical features in Gower matching, regardless of pandas dtype. Integer-coded categorical values are not treated as ordinal numeric distances unless you intentionally exclude them from `features_categorical` or pass an explicit `gower_cat_features` override.
  - `mcf`: use min‑cost‑flow sparse solver (requires `ortools`)
  - `gower_weights`: per‑feature weights passed to GowerKNN
    - Preferred: dict keyed by *original* feature names.
      - Numeric feature: provide a numeric value (applies to the transformed column used for matching).
      - Categorical feature: provide either a numeric value (applies to all children) or a dict of child weights. Child keys may be full processed names (e.g., raceeth_White) or suffixes (e.g., White). Only two levels are supported. If you omit some child weights, RDMatcher will warn and default missing children to 1.0.
    - Legacy: list/tuple/ndarray is accepted but order‑dependent; length must match the exact feature column order used for matching or it will raise. Make sure to check the log files to ensure the weights are applied correctly.
  - `gower_sd_weights`: bool (default False). When True, RDMatcher builds Gower weights automatically so SD-scale numeric differences are not diluted by large observed ranges. This is useful for heavy-tailed numeric covariates or very large control pools where rare extremes can expand min-max ranges.
  - `gower_sd_reference`: `'controls'` or `'pooled'` (default `'controls'`). Controls-only is the recommended ATT-style donor reference. `'pooled'` is available as a sensitivity-analysis option that includes treated units in the scale calculation for the current matching view.
  - `gower_sd_weights_mult`: float (default 1.96). SD multiplier used by `gower_sd_weights`. A numeric difference of about `gower_sd_weights_mult` standard deviations is made comparable to a categorical mismatch before block normalization. Use either `gower_sd_weights=True` or explicit `gower_weights`, not both.
  - `batch_size`: batch size for neighbor/distance computations
  - `safe_matches`: number of guaranteed "safe" matches to secure before further allocation (defaults to n_neighbors)
  - `n_jobs`: concurrency control for neighbor/distance computations. Default is 1 (single-threaded). Set to -1 to use all CPUs, or >1 to specify a fixed number of threads. Note: for shared clusters, prefer explicit values (e.g., n_jobs=4) rather than -1.
  - `streaming`: 'auto'|'on'|'off' (default 'auto'). When 'on' streaming top‑k (block-wise) is forced; 'off' disables streaming; 'auto' enables streaming when the estimated full query×control distance matrix exceeds stream_threshold_gb.
  - `stream_block_size`: integer (default 50000). Number of control records processed per streaming block when streaming is enabled. Smaller blocks reduce peak memory but increase overhead.
  - `stream_threshold_gb`: float (default 1.0). When streaming='auto' this threshold (in GB) determines whether to use the streaming path for a given query batch.
  - `parallel_chunk_size`: integer (default None). Controls the per-worker query chunk size used by the internal thread pool — i.e., how many queries each worker handles at once. Smaller values give finer parallelism but higher scheduling overhead.
  - `memory_limit_gb`: float (default None). Preferred kwarg to set the memory-warning threshold (in GB) for vectorized block merges. If unspecified the code will fall back to the RD_MATCHER_MEMORY_LIMIT_GB environment variable and then to 4.0 GB. Use this kwarg to avoid mixing env vars and API args.

Additional concurrency notes for Gower distance
- The Gower prefilter (kneighbors) can run in parallel across query chunks when `n_jobs > 1`. This is implemented with a thread pool and avoids copying large arrays between processes. Because each worker allocates temporary buffers of size roughly (chunk_size × n_controls), parallel runs increase peak memory usage. The library will emit a warning if estimated temporary memory exceeds the default 4 GB.
- The cdist() method (used as a fallback in some assignment code paths) will only parallelize when n_jobs > 1 and the pairwise problem size n_queries × n_references is larger than 1e5 (heuristic). Otherwise cdist remains single-threaded to avoid overhead on small calls.
- When streaming is enabled (or forced), RDMatcher processes the control pool in blocks of size `stream_block_size` and maintains per-query top‑k candidates incrementally. This avoids allocating a full n_queries×n_controls distance matrix and can drastically reduce peak memory use.
- The `memory_limit_gb` kwarg controls when the block-wise per-row merge falls back from a fast vectorized merge to a memory-safer per-row merge. Set it explicitly when calling rare_matching to ensure reproducible behavior across runs and avoid reliance on environment variables.


Note: `replacement=True` is not implemented (will raise NotImplementedError). Use `replacement=False`.

Method behavior note:
- `method='propensity'` requires that propensity scores/logits are present in the preprocessed data (column `propensity_logit`). Run `matcher.fit_propensity_model()` before calling `rare_matching` with `method='propensity'`. It works buy setting the weights for all features besides the `propensity_logit` to 0.
- `method='multi'` is the default and will use all features specified on class construction. 

## Math: Gower distance (implementation details)

Gower distance (as implemented)

Gower handles mixed numeric and categorical features. RDMatcher uses `features_categorical` as the source of truth for nominal categorical columns. For numeric features we compute a range‑normalized absolute difference; for categorical features the contribution is 0 when equal and 1 when different. Missing values are handled by excluding that feature's weight from the denominator on a per‑pair basis.

More precisely, let features be indexed by $j = 1, \dots, p$. For a pair of observations $x$ and $y$ define the per‑feature contribution

$$
\delta_j(x_j, y_j) =
\begin{cases}
\dfrac{|x_j - y_j|}{r_j}, & \text{if feature } j \text{ is numeric}, \\
\mathbf{1}\{x_j \ne y_j\}, & \text{if feature } j \text{ is categorical},
\end{cases}
$$

where $r_j$ is the feature range computed on the fitted reference pool. The implementation applies safeguards for zero ranges (zero‑length ranges are set to 1.0 and a numerical floor is enforced for stability).

Missingness is handled by a per‑pair mask

$$
m_j(x_j,y_j) = \begin{cases}
1, & \text{if both } x_j \text{ and } y_j \text{ are observed}, \\
0, & \text{otherwise}.
\end{cases}
$$

Given non‑negative feature weights $w_j$, the implemented pairwise Gower distance is the weighted, mask‑aware average

$$
d(x,y) = \frac{\sum_{j=1}^p w_j\, m_j(x_j,y_j)\, \delta_j(x_j,y_j)}{\sum_{j=1}^p w_j\, m_j(x_j,y_j)}.
$$

Implementation notes and corner cases:
- If the denominator $\sum_j w_j m_j(x_j,y_j)$ is zero (no comparable features for the pair), the implementation explicitly sets $d(x,y)=1.0$ (maximum distance).
- Categorical missing values are encoded internally and excluded via the mask $m_j$.
- Feature weights default to equal weights when none are supplied; internally numeric and categorical weights are stored separately.
- Computations use float32 for memory efficiency; small numerical tolerances are present when comparing to matching thresholds.
- True nominal categorical Gower changes threshold interpretation. With six equally weighted complete features, one categorical mismatch contributes about `1/6 = 0.167` to distance before numeric differences are added. Thresholds that were tuned under integer-coded dtype inference are not comparable to true categorical Gower thresholds.

#### SD-calibrated Gower weights

Standard Gower normalizes numeric features by the fitted reference range. With very large control pools or heavy-tailed numeric variables, rare extremes can increase the range and make clinically meaningful numeric differences contribute too little to the final distance.

Set `gower_sd_weights=True` in `rare_matching()` to automatically build weights from the control/reference pool:

```python
matcher.rare_matching(
    threshold=0.25,
    n_neighbors=1,
    k_candidates=500,
    distance_metric="gower",
    gower_sd_weights=True,
    gower_sd_reference="controls",
    gower_sd_weights_mult=1.96,
)
```

For each numeric feature, RDMatcher computes a raw weight proportional to:

$$
\frac{\text{range}_j}{\text{gower\_sd\_weights\_mult} \times \text{SD}_j}
$$

By default the reference pool is the control cohort, which preserves ATT-style donor scaling. Set `gower_sd_reference="pooled"` if you want a sensitivity-analysis variant that uses treated + control records from the same matching view when computing the SD and range. The Gower distance formula is unchanged; only the feature weights change. The resolved weights are logged and stored on the matcher as `matcher.gower_weights_`.

### Computational complexity: linear vs superlinear phases
- Prefiltering and candidate categorization are essentially linear in the number of kneighbors results produced — if you compute `k_candidates` for each exposed subject, the work to produce and scan those distances is $O(N)$ in the total kneighbors output size.
- However, the global assignment phase is an assignment problem solved with the Hungarian algorithm (or an alternative min‑cost‑flow solver). The runtime and memory use can grow superlinearly (often cubic) in the smaller matrix dimension. RDMatcher reduces the practical burden by limiting `k_candidates`, assigning "safe" controls greedily, and only invoking the global solver on a heavily pruned instance. An optional sparse MCF path (`mcf=True`) leverages OR‑Tools to operate on a sparse network representation and may reduce memory overhead.

## Addendum: Propensity score (optional)
Propensity-score modeling is supported as an optional component of the workflow. It is intentionally presented as an addendum because the primary matching approach in this library is Gower-based multi-covariate matching. Use propensity scores when you prefer to match on a single scalar summary of covariates or when you want to include the propensity logit as an additional numeric feature in the Gower distance.
What the API provides:
- RDMatcher.fit_propensity_model(formula=None, random_state=404, \*\*kwargs) builds a one-hot encoded modeling view (categoricals -> dummies), fits a logistic model (by default), and returns/merges scalar columns `propensity_score` (probability) and `propensity_logit` (log-odds) into `self.pop` and `self.pop_processed`. The method also stores metadata about the processed model matrix and parsed formula in `get_propensity_feature_map()``.
- You can then:
  - match only on the propensity scalar (method='propensity' in rare_matching), or
  - include `propensity_logit` among your matching features and weight it via `gower_weights` when using Gower (e.g., `{'propensity_logit': 2.0}`). In that case the logit is treated as a numeric feature in the Gower computation.
When to use propensity as an extra feature:
- Reasonable when you want a single summary score to guide selectivity or to prioritize balance on a modeled risk. Treat it as an additional covariate rather than a replacement for careful multi-covariate matching.

### Caveats and recommendations:
- Propensity scores are derived from the same covariates you may already be using for matching. Adding propensity as a feature effectively re-uses those covariates in a compressed form and can overweight their contribution if you give the propensity feature a large weight. Be deliberate about the weight you assign and consider sensitivity checks without the propensity feature.
- `propensity_logit` is an unbounded log-odds value by design. When including it in Gower, the code treats it as a numeric column; you may choose to scale it (e.g., via your preprocessing pipeline) or control its influence via `gower_weights`.
### Formula representation:
- Use original feature names when writing formulas. Supported operators: `+`, `:`, `*`, and parentheses for grouping (the parser lives in src/rdmatcher/formula.py).
- If `formula=None`, the propensity routine uses all original features as main effects.
- Advanced options in `fit_propensity_model` include `n_bags` (bagging ensemble), `downsample_ratio` (train on sampled controls), and `n_hard_mining` (two-step mining). You can also pass a custom sklearn estimator or estimator kwargs.

## Mahalanobis distance (numeric-only)

Mahalanobis matching is supported as a numeric-only distance option for cases where all matching covariates are continuous (or have already been encoded numerically). It is useful when you want a scale-aware distance that accounts for covariate covariance structure rather than simple Euclidean scaling.

Note: RDMatcher uses the standard (square-root) Mahalanobis distance
d(x) = sqrt((x - mu)^T Sigma^{-1} (x - mu)). The distances returned by
`MahalanobisKNN.cdist` / `kneighbors` are the square-rooted Mahalanobis
distance (not the squared form). Any `threshold` passed to `rare_matching`
is compared directly against these Mahalanobis distance values.

Key points:
- Input must be numeric-only. DataFrame inputs with any non-numeric column will raise a ValueError; prefer passing a numpy array or a DataFrame containing only numeric covariates.
- Missing values are handled with a MatchIt-style complete-case policy: rows with any NaN are dropped from the reference set at `fit()` time. Query rows (used in `cdist()` or `kneighbors()`) that contain NaN will return NaN distances and an index of `-1` for that row. `cdist(..., Y_ref=...)` will raise a ValueError if `Y_ref` contains any NaN — callers must preprocess or impute reference data first.
- Covariance options:
  - `cov_source='reference'` (default): covariance is computed from the reference set (the controls) used in `fit()`.
  - `cov_source='pooled'`: covariance is computed from a pooled dataset (typically treated+controls). This mirrors MatchIt behavior when MatchIt computes a pooled covariance for Mahalanobis matching. Use the `pooled_X` argument to pass the pooled matrix.
  - `VI`: callers may optionally supply a precomputed precision (inverse covariance) matrix to control the distance metric precisely.
- Neighbor-search backend options:
  - `mahalanobis_neighbor_backend='cdist'` (default): preserves the original exact candidate-search path. It computes full batch-by-control Mahalanobis distances and then selects the top `k_candidates`.
  - `mahalanobis_neighbor_backend='sklearn'`: whitens controls once and uses `sklearn.neighbors.NearestNeighbors(metric='euclidean')` to find the top `k_candidates` in whitened space. This is intended as a faster candidate-search path for large cohorts.
  - `mahalanobis_algorithm='auto'` (default): passed to sklearn `NearestNeighbors(algorithm=...)` when `mahalanobis_neighbor_backend='sklearn'`. Accepted values are sklearn's values, including `'auto'`, `'ball_tree'`, `'kd_tree'`, and `'brute'`.
  - `n_jobs`: passed to sklearn `NearestNeighbors` for the `'sklearn'` backend. Use `n_jobs=-1` for all CPUs or a positive integer for a fixed worker count.
- Numerical and API details:
  - Internally the estimator computes a whitening transform and evaluates Mahalanobis via whitened Euclidean distances for speed. The implementation adapts regularization automatically when covariance matrices are near-singular.
  - Distances are returned as `float32` (memory-conscious choice); indices are `int32` and map back to the original input row indices when rows were dropped during fit.
  - Tie-breaking for equal distances is deterministic: ties are resolved by `(distance, original_index)` ordering to ensure reproducible neighbor lists.

Usage (low-level)

```python
from rdmatcher.mahalanobis import MahalanobisKNN

# X_control: numeric controls (DataFrame or ndarray)
# X_pooled: numeric pooled (treated + controls) if you want pooled covariance
model = MahalanobisKNN(
    neighbor_backend="sklearn",
    algorithm="auto",
    n_jobs=-1,
)
model.fit(X_control, cov_source='pooled', pooled_X=X_pooled)

# Query (treated) kneighbors
distances, indices = model.kneighbors(X_treated, n_neighbors=1)
```

Usage through `RDMatcher.rare_matching()`

```python
matched = matcher.rare_matching(
    threshold=2.0,
    n_neighbors=1,
    k_candidates=200,
    method="multi",
    distance_metric="mahalanobis",
    global_optimal=True,
    competitive_match=True,
    return_matched_data=True,
    mahalanobis_neighbor_backend="sklearn",  # "cdist" or "sklearn"
    mahalanobis_algorithm="auto",            # sklearn NearestNeighbors algorithm
    n_jobs=-1,
)
```

The Mahalanobis backend only changes the candidate-search step. It does not change the downstream competitive allocation or global optimal matching behavior.

Notes on parity with MatchIt
- The estimator can be configured to compute a pooled covariance (the `cov_source='pooled'` path) to align with MatchIt's Mahalanobis distance computation. However, exact pairwise results can still differ from MatchIt because MatchIt performs a global nearest-without-replacement allocation and may use different internal numerical regularization and tie-breaking rules. We provide optional parity tests (which use `rpy2` and R's MatchIt) that run only when R tooling is available.

Integration
- The RDMatcher `Matcher` path wires Mahalanobis to use pooled covariance by default when `distance_metric='mahalanobis'`.
- The high-level `RDMatcher.rare_matching()` API accepts `mahalanobis_neighbor_backend` and `mahalanobis_algorithm` as keyword arguments.


## PSM+RDM: Propensity‑Score‑Calibrated Gower Matching

The propensity score section above describes using propensity as an optional feature within Gower distance or as a standalone matching dimension. Both of those approaches treat propensity as a feature that contributes to the distance calculation itself.

PSM+RDM takes a different approach: the propensity score is used as a **pre‑restriction filter** on the candidate control pool. For each treated subject, only controls within a propensity‑score caliper are eligible for matching. Gower distance then operates exclusively within these eligible sets to select the best match.

This is conceptually similar to the propensity‑score + Mahalanobis hybrid available in R's MatchIt (`distance='glm', mahvars=~covariates, caliper=0.2`), but uses Gower distance instead of Mahalanobis for the within‑caliper selection. The propensity score defines which controls are candidates; Gower distance decides which candidate wins.

Because the caliper creates separate pools of eligible controls for each treated subject, the matching problem is naturally decomposed into independent subproblems. A control that falls within the caliper of multiple treated subjects appears in multiple eligible sets, and the competitive allocation phase resolves these conflicts. Controls outside a treated subject's caliper are never considered as matches for that subject.

### Usage

```python
import pandas as pd
from rdmatcher import RDMatcher

df = pd.read_csv("population.csv")

matcher = RDMatcher(
    pop_df=df,
    patient_id_col='patient_id',
    exposure_status='exposure_status',
    features_numeric=['age', 'bmi', 'lab_value'],
    features_categorical=['sex', 'race_ethnicity'],
    process_features=False,
    onehot=False,
)

# Fit propensity model (required for PSM+RDM)
matcher.fit_propensity_model(
    formula='age + bmi + lab_value + sex + race_ethnicity',
    random_state=404,
)

# PSM+RDM: propensity caliper restricts the candidate pool,
# then Gower distance selects the best match within eligible controls
matched = matcher.rare_matching(
    threshold=0.3,
    n_neighbors=1,
    k_candidates=500,
    method='multi',
    distance_metric='gower',
    global_optimal=True,
    competitive_match=True,
    ps_hybrid=True,
    ps_caliper=0.2,
    gower_sd_weights=True,
    return_matched_data=True,
)
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `ps_hybrid` | `False` | Enable propensity‑score caliper pre‑filtering. When `True`, only controls within the PS caliper are eligible matches for each treated subject. |
| `ps_caliper` | `0.2` | Caliper width as a multiple of the standard deviation of the propensity logit. A value of 0.2 means only controls whose propensity logit is within 0.2 SD of the treated subject's logit are eligible. |
| `gower_sd_weights` | `False` | Optional SD-calibrated Gower feature weighting for the within-caliper RDM step. |
| `gower_sd_reference` | `'controls'` | Reference pool for SD-calibrated Gower weights. `'controls'` is the default ATT-style donor reference; `'pooled'` includes treated units for sensitivity analysis. |
| `gower_sd_weights_mult` | `1.96` | SD multiplier used when `gower_sd_weights=True`. |

When `ps_hybrid=True`, the propensity logit is excluded from the Gower distance computation to avoid double‑counting. The propensity score acts only as a filter; Gower matching operates on the original covariates within the eligible pool.

## Advanced topics & troubleshooting
- OR‑Tools (mcf): if you plan to use `mcf=True`, install `ortools` (`pip install ortools`). The MCF implementation requires precomputed kneighbors distances to construct sparse arcs.
- One‑hot + Gower: Do not one‑hot encode when using Gower; it treats categorical features natively and one‑hot will distort distances.
- Numerical stability: Gower uses float32 internal computation in many places to conserve memory; small numerical tolerances are present when comparing to the threshold.

## Installation & dependencies
- Core: Python 3.8+ recommended
- Required (runtime): numpy, pandas, scikit‑learn, scipy, matplotlib, seaborn
- Optional: ortools (for `mcf` / min‑cost‑flow)

Repository install (recommended for RDMatcher)

```bash
git clone https://github.com/noahrbaker/RDMatcher.git
cd RDMatcher
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install ortools   # optional, only required for mcf=True
```

## Authors & License
- Noah Baker, MPH - PhD Candidate, Biomedical Informatics, UCSF - noah.baker@ucsf.edu

License: GPL-3.0 (see LICENSE file)
