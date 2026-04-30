# RDMatcher

RDMatcher is a Python package for population matching and causal inference analysis in observational data. It emphasizes robust, scalable matching for rare-exposure (or rare-outcome) scenarios using mixed-data Gower distance and a staged matching strategy that combines fast greedy allocation with globally optimal assignment.

## Table of Contents
- [Overview](#overview)
- [Tested implementations](#tested-implementations)
- [Quickstart (Gower default)](#quickstart-gower-default)
- [Preprocessing](#preprocessing)
- [RDMatcher class (API)](#rdmatcher-class-api)
  - [Constructor](#constructor)
  - [Key methods and attributes](#key-methods-and-attributes)
  - [Outputs](#outputs)
- [rare_matching with Gower (usage & options)](#rare_matching-with-gower-usage--options)
- [Math: Gower distance (implementation details)](#math-gower-distance-implementation-details)
  - [Computational complexity: linear vs superlinear phases](#computational-complexity-linear-vs-superlinear-phases)
- [Advanced topics & troubleshooting](#advanced-topics--troubleshooting)
- [Installation & dependencies](#installation--dependencies)

## Overview
RDMatcher provides tools to:
- Preprocess numeric/categorical/datetime features
- Compute propensity logits (optional) using several strategies (full, bagging, hard negative mining)
- Match exposed subjects to control subjects using a flexible staged algorithm:
  - Candidate prefiltering via k-nearest neighbors (Gower or numeric metrics)
  - Identification of "safe" vs "competitive" controls
  - Competitive greedy allocation for limited exposed subjects
  - Global optimal assignment (Hungarian) on the reduced problem

This design reduces the effective search space for the expensive global solver and scales to large datasets.

## Tested implementations
- Distances: Gower (mixed-type; implemented as GowerKNN), Euclidean, Cosine
- Matching pipeline: batched neighbor prefiltering, safe/competitive categorization, greedy competitive allocation, global optimal assignment (Hungarian) on the reduced problem
- Sparse global solver: optional min-cost-flow (mcf) implementation using OR-Tools for very large sparse instances
- Propensity modeling: `propensity_logits_simple` (supports downsampling, bagging, hard-negative mining) and `propensity_logits_full` (interaction terms)
- Diagnostics: SMD summary table and feature balance plotting utilities

## Quickstart (Gower default)

Note: For Gower, keep categorical columns as pandas object/category dtype and do not one-hot encode them. If you plan to use Euclidean or Cosine distances instead, you should convert categorical variables to numeric (e.g., one-hot) via the preprocessing pipeline or externally.

```python
import pandas as pd
from rdmatcher import RDMatcher

# Load the full population DataFrame (must include the exposure column coded 0/1)
pop_df = pd.read_csv("population.csv")

# Initialize RDMatcher (Gower-focused defaults)
matcher = RDMatcher(
    pop_df=pop_df,
    patient_id_col='patient_id',
    exposure_status='exposure_status',
    features_numeric=['age', 'bmi', 'lab_value'],
    features_categorical=['sex', 'race_ethnicity'],
    features_datetime=['first_visit_date'],
    process_features=False,   # keep raw types for Gower
    onehot=False,             # IMPORTANT: For Gower, avoid one-hot encoding
    debug=False
)

# (Optional) Compute propensity logits (simple or full)
matcher.calculate_propensity_logits(method='simple')

# Run rare matching using Gower distance
matcher.rare_matching(
    threshold=0.25,
    n_neighbors=1,
    k_candidates=500,
    method='multi',              # multi-covariate matching; includes propensity_logit if present
    distance_metric='gower',     # default: Gower for mixed data
    global_optimal=True,
    replacement=False,
    competitive_match=True,
    diagnostics=True
)

# Results
matched_df = matcher.pop_matched
summary = matcher.summary_table

```

Preprocessing
RDMatcher provides an internal preprocessing pipeline, but for Gower distance the recommended workflow is to keep categorical columns as object/category dtype and not one-hot encode them.

- The package provides build_preprocessing_pipeline and apply_preprocessing_pipeline if you prefer to transform data prior to matching (scaling, log transforms, binning, and optional one-hot encoding).
- If you use Euclidean or Cosine metrics, categorical features must be converted to numeric (e.g., one-hot). You can either preprocess externally or set process_features=True and onehot=True (or onehot_scalar=True to apply scaling after one-hot encoding).

## RDMatcher class (API)
### Constructor
- pop_df: pandas DataFrame containing both exposed and control samples (exposure indicated by exposure_status)
- patient_id_col: unique identifier column name (must be unique; duplicates raise an error)
- exposure_status: column name containing 0/1 exposure indicator (0=control, 1=exposed)
- features_numeric, features_categorical, features_datetime, features_log, features_bin
- process_features (bool): run internal preprocessing pipeline
- onehot (bool): whether to one-hot encode categorical features (default False; do not use with Gower)

### Key methods and attributes
- process_features(): run preprocessing pipeline (public wrapper)
- calculate_propensity_logits(method='simple'|'full', ...): compute propensity_score and propensity_logit
- rare_matching(...): main matching routine (see next section)
- Attributes after execution:
  - pop: combined raw population DataFrame
  - pop_processed: preprocessed DataFrame (if process_features=False this will be a copy of pop, after datetime conversion if applicable)
  - matched_data / pop_matched: matched table with match_group and n_matches
  - matched_exposed / matched_control / unmatched_exposed
  - summary_table: diagnostics table of SMDs after matching

### Outputs
- pop_matched (DataFrame): merged matched table; includes at minimum the patient id, exposure status, `match_group`, `n_matches`, and `match_distance`. If propensity logits were computed, `propensity_logit` will also be included.
- matched_exposed / matched_control (DataFrames): subsets of pop_matched by exposure status
- unmatched_exposed (DataFrame): exposed subjects that were not matched

rare_matching with Gower (usage & options)
- Core idea: find k nearest candidate controls for each exposed subject using a fast neighbor search (GowerKNN for mixed data). Candidates within threshold are classified into:
  - safe: candidate assigned only to one exposed subject (fast, deterministic)
  - competitive: candidate shared across multiple exposed subjects (requires more careful allocation)

- Phases
  1. Prefilter: kneighbors to get top k_candidates and distances
  2. Categorize safe vs competitive via vectorized usage counts
  3. Competitive allocation: greedy, deterministic assignment for limited subjects
  4. Global optimal: Hungarian algorithm on the reduced bipartite graph (only competitive leftover subjects)

- Important parameters
  - threshold: maximum allowable distance for a match
  - n_neighbors: number of matches per exposed subject
  - k_candidates: how many nearest controls to consider per exposed subject
  - global_optimal: whether to run the Hungarian solver in the final phase
  - competitive_match: whether to run the competitive allocation phase
  - distance_metric: 'gower' (mixed data), 'euclidean', or 'cosine'
  - mcf: use min-cost-flow sparse solver (requires ortools)
  - gower_weights: per-feature weights passed to GowerKNN
    - Preferred: dict keyed by *original* feature names.
      - Numeric feature: provide a number (applies to the transformed column used for matching).
      - Categorical feature: provide either a number (applies to all children) or a dict of child weights.
        Child keys may be full processed names (e.g., raceeth_White) or suffixes (e.g., White). Only two levels are supported.
        If you omit some child weights, RDMatcher will warn and default missing children to 1.0.
    - Legacy: list/tuple/ndarray is accepted but order-dependent; length must match the exact feature column order used for matching or it will raise.
  - gower_cat_features: explicit list or mask for categorical features (auto-detected if not provided)
  - batch_size: batch size for neighbor/distance computations
  - safe_matches: number of guaranteed "safe" matches to secure before further allocation (defaults to n_neighbors)
  - fuzzy_threshold / fuzzy_threshold_limit: enable limited fuzzy matching beyond the main threshold
  - n_jobs: concurrency control for neighbor/distance computations. Default is 1 (single-threaded). Set to -1 to use all CPUs, or >1 to specify a fixed number of threads. Note: for shared clusters, prefer explicit values (e.g., n_jobs=4) rather than -1.
  - streaming: 'auto'|'on'|'off' (default 'auto'). When 'on' streaming top‑k (block-wise) is forced; 'off' disables streaming; 'auto' enables streaming when the estimated full query×control distance matrix exceeds stream_threshold_gb.
  - stream_block_size: integer (default 50000). Number of control records processed per streaming block when streaming is enabled. Smaller blocks reduce peak memory but increase overhead.
  - stream_threshold_gb: float (default 1.0). When streaming='auto' this threshold (in GB) determines whether to use the streaming path for a given query batch.
  - parallel_chunk_size: integer (default None). Controls the per-worker query chunk size used by the internal thread pool — i.e., how many queries each worker handles at once. Smaller values give finer parallelism but higher scheduling overhead.
  - memory_limit_gb: float (default None). Preferred kwarg to set the memory-warning threshold (in GB) for vectorized block merges. If unspecified the code will fall back to the RD_MATCHER_MEMORY_LIMIT_GB environment variable and then to 4.0 GB. Use this kwarg to avoid mixing env vars and API args.

Additional concurrency notes for Gower distance
- The Gower prefilter (kneighbors) can run in parallel across query chunks when n_jobs > 1. This is implemented with a thread pool and avoids copying large arrays between processes. Because each worker allocates temporary buffers of size roughly (chunk_size × n_controls), parallel runs increase peak memory usage. The library will emit a warning if estimated temporary memory exceeds RD_MATCHER_MEMORY_LIMIT_GB (default 4 GB).
- The cdist() method (used as a fallback in some assignment code paths) will only parallelize when n_jobs > 1 and the pairwise problem size n_queries × n_references is larger than 1e5 (heuristic). Otherwise cdist remains single-threaded to avoid overhead on small calls.
 - When streaming is enabled (or forced), RDMatcher processes the control pool in blocks of size `stream_block_size` and maintains per-query top‑k candidates incrementally. This avoids allocating a full n_queries×n_controls distance matrix and can drastically reduce peak memory use.
 - The `memory_limit_gb` kwarg controls when the block-wise per-row merge falls back from a fast vectorized merge to a memory-safer per-row merge. Set it explicitly when calling rare_matching to ensure reproducible behavior across runs and avoid reliance on environment variables.

Best practices on shared clusters
- Default is conservative: n_jobs=1. Avoid using n_jobs=-1 on shared systems. Instead specify an explicit small number of threads (e.g., n_jobs=4) that you requested from your scheduler.
- To avoid over-subscription when BLAS is multi-threaded (MKL/OpenBLAS), set environment variables to limit BLAS threads when using n_jobs > 1, for example:

  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1

Note: `replacement=True` is not implemented (will raise NotImplementedError). Use replacement=False.

Method behavior note:
- `method='propensity'` requires that propensity scores/logits are present in the preprocessed data (column `propensity_logit`). Run `matcher.calculate_propensity_logits()` before calling rare_matching with `method='propensity'`.
- `method='multi'` will use propensity if present but does not require it.

## Math: Gower distance (implementation details)

Gower distance (as implemented)

Gower handles mixed numeric and categorical features. For numeric features we compute a range-normalized absolute difference; for categorical features the contribution is 0 when equal and 1 when different. Missing values are handled by excluding that feature's weight from the denominator on a per-pair basis.

More precisely, let features be indexed by $j = 1, \dots, p$. For a pair of observations $x$ and $y$ define the per-feature contribution

$$
\delta_j(x_j, y_j) =
\begin{cases}
\dfrac{|x_j - y_j|}{r_j}, & \text{if feature } j \text{ is numeric}, \\
\mathbf{1}\{x_j \ne y_j\}, & \text{if feature } j \text{ is categorical},
\end{cases}
$$

where $r_j$ is the feature range computed on the fitted reference pool. The implementation applies safeguards for zero ranges (zero-length ranges are set to 1.0 and a numerical floor of $1\times10^{-8}$ is enforced for stability).

Missingness is handled by a per-pair mask

$$
m_j(x_j,y_j) = \begin{cases}
1, & \text{if both } x_j \text{ and } y_j \text{ are observed}, \\
0, & \text{otherwise}.
\end{cases}
$$

Given non-negative feature weights $w_j$, the implemented pairwise Gower distance is the weighted, mask-aware average

$$
d(x,y) = \frac{\sum_{j=1}^p w_j\, m_j(x_j,y_j)\, \delta_j(x_j,y_j)}{\sum_{j=1}^p w_j\, m_j(x_j,y_j)}.
$$

Implementation notes and corner cases:
- If the denominator $\sum_j w_j m_j(x_j,y_j)$ is zero (no comparable features for the pair), the implementation explicitly sets $d(x,y)=1.0$ (maximum distance).
- Categorical missing values are encoded internally as $-9$ and excluded via the mask $m_j$.
- Feature weights default to equal weights when none are supplied; internally numeric and categorical weights are stored separately as `w_num_` and `w_cat_`.
- Computations use float32 for memory efficiency; small numerical tolerances are present when comparing to matching thresholds.

Computational complexity: linear vs superlinear phases
- Prefiltering and candidate categorization are essentially linear in the number of kneighbors results produced — if you compute k_candidates for each exposed subject, the work to produce and scan those distances is O(N) in the total kneighbors output size.

- However, the global assignment phase is an assignment problem solved with the Hungarian algorithm (or an alternative min-cost-flow solver). The implementation constructs an augmented cost matrix whose dimensions scale with the number of exposed subjects remaining and the number of candidate controls; the Hungarian solver's runtime and memory use can grow superlinearly (often cubic) in the smaller matrix dimension. Thus while the prefiltering is O(N) (linear), the global phase can dominate asymptotically and cause practical intractability for large reduced-problem sizes.

- RDMatcher reduces the practical burden by limiting k_candidates, assigning "safe" controls greedily, and only invoking the global solver on a heavily pruned instance. An optional sparse MCF path (mcf=True) leverages OR-Tools to operate on a sparse network representation to reduce memory overhead.

Advanced topics & troubleshooting
- OR-Tools (mcf): if you plan to use mcf=True, install ortools (pip install ortools). The MCF implementation requires precomputed kneighbors distances to construct sparse arcs.
- Replacement=True: Matching with replacement is not implemented (explicit NotImplementedError). Use replacement=False.
- One-hot + Gower: Do not one-hot encode when using Gower; it treats categorical features natively and one-hot will distort distances.
- Numerical stability: Gower uses float32 internal computation in many places to conserve memory; small numerical tolerances are present when comparing to the threshold.

Installation & dependencies
- Core: Python 3.8+ recommended
- Required (runtime): numpy, pandas, scikit-learn, scipy, matplotlib, seaborn
- Optional: ortools (for mcf/min-cost-flow)

Note: SciPy is required for the Hungarian assignment and sparse matrix utilities. OR-Tools is optional and only necessary for the sparse min-cost-flow (mcf=True) path.

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

Contact & License
- Author: Noah Baker <noah.baker@ucsf.edu>
- License: MIT (see LICENSE file)
