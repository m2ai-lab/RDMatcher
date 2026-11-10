# RDMatcher

RDMatcher kernelizes weighted bipartite graphs to enable optimal matching on a reduced
problem. Cases and controls form the two vertex sets, eligible pairs form the edges, and
distances define edge costs. Distance-ordered candidate retrieval constructs the reduced
graph, and preservation certificates validate it. Optimal allocation solves this graph
without replacement. Hybrid allocation commits competitive matches first and solves the
residual problem. The package supports mixed-data and numeric distances, feature
weighting, propensity-score eligibility and candidate-graph export. Supported distances
are Gower, Euclidean, Manhattan, cosine and Mahalanobis.

This document describes the **0.2.0.dev0** public API, including installation, matching
workflows, parameter definitions, distance calculations and troubleshooting.

## Table of contents

- [Overview and terminology](#overview-and-terminology)
- [Installation and dependencies](#installation-and-dependencies)
- [Quickstart: mixed-data Gower matching](#quickstart-mixed-data-gower-matching)
- [RDM Optimal and RDM Hybrid](#rdm-optimal-and-rdm-hybrid)
- [Candidate retrieval and preservation](#candidate-retrieval-and-preservation)
- [RDMatcher class and input contract](#rdmatcher-class-and-input-contract)
- [Complete matching and execution reference](#complete-matching-and-execution-reference)
- [Outputs, balance and inspection](#outputs-balance-and-inspection)
- [Math: Gower distance (implementation details)](#math-gower-distance-implementation-details)
- [Numeric distances and Mahalanobis matching](#numeric-distances-and-mahalanobis-matching)
- [Propensity-score modeling](#propensity-score-modeling)
- [Propensity-only matching](#propensity-only-matching)
- [PS-restricted Gower and Mahalanobis matching](#ps-restricted-gower-and-mahalanobis-matching)
- [Exporting the candidate graph](#exporting-the-candidate-graph)
- [Global solvers and numerical precision](#global-solvers-and-numerical-precision)
- [Complexity, memory and concurrency](#complexity-memory-and-concurrency)
- [Reproducibility and troubleshooting](#reproducibility-and-troubleshooting)
- [Tests and source map](#tests-and-source-map)
- [Supplying a precomputed matrix to RDM](#supplying-a-precomputed-matrix-to-rdm)
- [Authors and license](#authors-and-license)

## Overview and terminology

A population DataFrame contains both **cases** (`exposure_status == 1`) and **controls**
(`exposure_status == 0`). Treated and untreated subjects use the same binary roles in
observational applications.

The requested matching ratio is **controls per case**. `ratio=1` requests
one control per case; `ratio=4` requests up to four. Each control is used at
most once per matching call. Eligibility and competition determine whether a case
receives the requested number, a partial set or no controls. Each call starts a new
allocation with the supplied population.

RDMatcher separates four decisions:

1. **Representation:** which original covariates enter the distance, and how
   numeric/categorical/datetime features are represented.
2. **Eligibility:** the distance threshold and optional propensity caliper.
3. **Kernelization:** which distance-ordered eligible edges are retained to
   preserve an optimal solution to the matching problem.
4. **Allocation:** whether to solve the retained graph directly or commit
   competitive matches first, and which coverage/distance priorities to use.

Distance, caliper, weighting and preprocessing settings define the matching problem. To
compare computational backends, hold the feature representation, eligible edges and
allocation objective constant.

## Installation and dependencies

The package declares **Python 3.10 or newer**. Runtime dependencies are NumPy, pandas,
scikit-learn, SciPy, Matplotlib and seaborn. OR-Tools is optional for the min-cost-flow
solver; pytest is optional for the package tests. Dependency versions are not pinned by
the package. Record the tested dependency versions with matching outputs to identify the
environment used for a run.

From a checkout containing this README and `pyproject.toml`:

```sh
python -m pip install .
```

For development and optional features:

```sh
python -m pip install -e '.[test]'
python -m pip install '.[mcf]'
```

For a separate environment, create and activate a virtual environment first:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

The build backend is Flit. RDMatcher installs as a standalone Python package. Core
matching uses the Python dependencies listed above. R, MatchIt and other external
comparators have separate dependency requirements. If several checkouts or editable
installs are available, check which package Python imports:

```python
import rdmatcher

print(rdmatcher.__version__)
print(rdmatcher.__file__)
```

## Quickstart: mixed-data Gower matching

This self-contained example generates a small synthetic population. Replace it with your
DataFrame or `pd.read_csv(...)` when working with prepared inputs. Later examples
reuse `population`, `matcher` and the imports from this section.
Integer-coded nominal variables still belong in `features_categorical` if their codes do
not represent ordered numeric differences.

```python
import numpy as np
import pandas as pd
from rdmatcher import RDMatcher, MatchingOptions, ExecutionOptions

rng = np.random.default_rng(404)
population = pd.DataFrame({
    "patient_id": [f"person_{i}" for i in range(120)],
    "treated": [1] * 30 + [0] * 90,
    "age": rng.normal(55, 12, 120),
    "bmi": rng.normal(27, 4, 120),
    "lab_value": rng.lognormal(0, 0.4, 120),
    "sex": rng.choice(["F", "M"], 120),
    "site": rng.choice(["A", "B", "C"], 120),
})

matcher = RDMatcher(
    pop_df=population,
    patient_id_col="patient_id",
    exposure_status="treated",
    features_numeric=["age", "bmi", "lab_value"],
    features_categorical=["sex", "site"],
    process_features=False,
    onehot=False,
)

matched = matcher.match(
    threshold=0.3,
    ratio=1,
    distance_metric="gower",
    allocation="optimal",
    matching=MatchingOptions(prioritize_coverage=True),
    execution=ExecutionOptions(n_jobs=1),
)

print(matched.head())
print(matcher.summary_table)
print(matcher.assignment_diagnostics_["preservation_condition_satisfied"])
```

The repository also includes a complete [notebook example](example.ipynb) using
the same current API. It covers mixed-data Gower matching, propensity-only
matching, propensity-restricted Gower matching, and numeric Mahalanobis matching.

`threshold` is expressed in the selected metric's units. Use
`float("inf")` to remove the distance cutoff. The example uses
`0.3`; select a threshold for your distance representation and matching
requirements. Inspect coverage and balance after matching.

The returned table includes original population columns and matching metadata. Unmatched
cases are included by default; unmatched controls are not. Inspect case
`n_matches` when counting retained cases rather than counting every row with
`treated == 1`. See [Outputs](#outputs-balance-and-inspection) for the current filtering and
diagnostic behavior.

## RDM Optimal and RDM Hybrid

Both allocation modes construct the same candidate graph for the same inputs, metric,
eligibility and requested ratio. Safe controls appear in only one active retained case
list; competitive controls appear in multiple lists.

| Allocation | Setting | Behavior |
| --- | --- | --- |
| RDM Optimal | `allocation="optimal"` (default) | Sends the retained graph to global assignment without competitive commitments. |
| RDM Hybrid | `allocation="hybrid"` | Performs competitive allocation first, then solves remaining demands using remaining controls. |

```python
hybrid = matcher.match(
    threshold=0.3,
    ratio=2,
    allocation="hybrid",
    matching=MatchingOptions(prioritize_coverage=True),
    execution=ExecutionOptions(solver="scipy_sparse"),
)
```

Hybrid keeps its competitive assignments fixed during the residual solve. Its global
solution is conditional on those commitments. The selected cohort and total distance
therefore depend on the competitive phase as well as the reduced graph. Without
competitive commitments, Hybrid follows the Optimal allocation path. Solver choice is
independent of allocation mode.

### Coverage versus unmatched-slot penalties

`MatchingOptions(prioritize_coverage=False)` is the software default. It solves an assignment objective
combining distances and unmatched-slot penalties; the base unmatched penalty is
1,000,000. Filling a slot reduces this penalty and adds the assigned edge's distance.
The relative scale of the penalty and edge costs determines the resulting priority.

`prioritize_coverage=True` constructs penalties for the following ordered objectives:

1. maximize cases receiving at least one control;
2. maximize the total number of filled control slots;
3. minimize distance among assignments satisfying those priorities.

In Hybrid, already covered cases and used controls are taken as fixed when the residual
objective is constructed. At ratios greater than one, giving a first control to an
uncovered case takes priority over giving an additional control to an already covered
case. Coverage priority changes allocation, not the metric or eligible-control sets. It
cannot create an eligible edge for a case with no suitable control.

The matched sample consists of cases that receive controls. When coverage is incomplete,
an effect estimated in that sample concerns the covered treated subset; extending it to
all treated subjects requires additional justification. Outcome modeling and
specification of the causal estimand are separate from the matching routines provided
here.

## Candidate retrieval and preservation

Kernelization retains distance-ordered eligible edges using the preservation
certificates listed below. These certificates preserve an optimal solution for the
selected assignment objective in the reduced graph. Optimal allocation solves that
reduced problem; Hybrid fixes competitive assignments before its residual solve. Gower
and numeric metrics share the demand-loaded prefix interface. The internal cap is
`min(number_of_controls, number_of_cases * ratio)`. It is computed for each call and
is not user-configurable.

Retrieval starts with a prefix of up to 16 entries and grows as the restriction
procedure needs more neighbors. Prefix buffers grow at least geometrically when
necessary. An extra neighbor is inspected at a retrieval boundary; a boundary tie
triggers retrieval to the cap for that case. These queries store the full capped prefix
instead of stopping at the tied boundary.

Indexed backends query prefixes as they expand. The prefix adapter stores capped
candidate pools for brute-force numeric search, the Mahalanobis `cdist`
backend and propensity-restricted neighborhoods. Later requests reuse these pools. Gower
without propensity restriction queries its neighbor backend for each requested prefix;
that backend selects its indexed or full-scan distance path. Stored pools include
candidates beyond the retained horizon when reduction stops before the cap. They contain
capped neighbor lists rather than a full case-by-control distance matrix.

The implementation checks one of three certificates for each case:

| Certificate | Meaning |
| --- | --- |
| `complete_eligible` | The eligible list is retained completely. |
| `singleton_safe` | At least the requested number of exclusive Safe controls are retained. |
| `Cr_prefix` | The retained eligible prefix contains at least case-count × ratio entries. |

The default cap and stopping rules support these certificates. Candidate construction
raises an error if a case remains uncertified. The certificates describe preservation of
the assignment problem during reduction. Solver precision and Hybrid's fixed commitments
remain separate considerations.

Prefix consistency is checked during expansion. If a previously loaded prefix changes,
the implementation raises `CandidatePrefixChanged`; there is no automatic restart in eager
mode. Index selection and ties affect resource use. The retrieval diagnostics identify
the backend and work performed for a call.

`assignment_diagnostics_["candidate_retrieval"]` records backend calls, requested/fetched/stored entries, repeated
prefix queries and tie-related full-cap rows for backends that report these counters.
The underlying implementation is in [_candidate_retrieval.py](src/rdmatcher/_candidate_retrieval.py), [_general_retrieval.py](src/rdmatcher/_general_retrieval.py) and
[_neighbor_backends.py](src/rdmatcher/_neighbor_backends.py).

## RDMatcher class and input contract

### Constructor

```text
RDMatcher(pop_df, patient_id_col, exposure_status,
          features_numeric, features_categorical,
          features_datetime=None, features_log=None, features_bin=None,
          process_features=False, bin_method=None, bin_width=10,
          onehot=False, onehot_scalar=False, onehot_drop='first',
          debug=False, log_file=None, log_to_console=False)
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `pop_df` | required | DataFrame containing both cases and controls. |
| `patient_id_col` | required | Nonmissing, unique patient-identifier column; IDs are preserved in outputs. |
| `exposure_status` | required | Binary role column: 1 = case, 0 = control. |
| `features_numeric` | required | List of numeric matching covariates; use `[]` if none. |
| `features_categorical` | required | List of nominal matching covariates, independent of their pandas numeric/string dtype; use `[]` if none. |
| `features_datetime` | `None` | Columns converted to elapsed days from each column's earliest observed timestamp; corresponding `_days` columns enter numeric features. |
| `features_log` | `None` | Numeric subset receiving `log1p` in the enabled numeric preprocessing path. |
| `features_bin` | `None` | Numeric subset subject to the binning/scaling policy below. |
| `process_features` | `False` | Enable optional initial preprocessing; also governs optional log/bin transformations in numeric matching. |
| `bin_method` | `None` | Numeric transformation policy: passthrough, `"scaler"`, `"binned"` or `"binned_scaler"`. |
| `bin_width` | `10` | Width used by the binning transformer. |
| `onehot` | `False` | Enable categorical encoding for numeric-metric matching; Gower uses original categories. |
| `onehot_scalar` | `False` | Additional categorical scaling option used by preprocessing; it is not a replacement for declaring `onehot=True` on the public numeric matching path. |
| `onehot_drop` | `"first"` | Encoder drop policy: `"first"`, `"if_binary"` or `None`. |
| `debug` | `False` | Request debug logging. |
| `log_file` | `None` | Optional log-file path. |
| `log_to_console` | `False` | Enable console logging. |

Use role values 0 and 1 with both groups present. Population construction selects those
rows and excludes other role labels before validation. Missing/duplicate IDs are
rejected. Supply identifiers in `patient_id_col`. Missing feature names and log/bin
features absent from `features_numeric` also raise errors.

Datetime conversion coerces unparseable values to missing and uses fractional days, not
just integer calendar-day differences. Match on the resulting numeric feature with the
same care as any other time-scale variable. Supply fresh feature-list objects when
reusing constructor arguments: datetime expansion appends names to the numeric feature
list.

```python
dated_population = population.assign(
    first_visit=pd.Timestamp("2020-01-01")
    + pd.to_timedelta(np.arange(len(population)), unit="D")
)
dated_matcher = RDMatcher(
    dated_population, "patient_id", "treated",
    features_numeric=["age", "bmi"],
    features_categorical=["sex"],
    features_datetime=["first_visit"],
)
dated_graph = dated_matcher.return_graph(threshold=float("inf"))
assert "first_visit_days" in dated_matcher.matching_data_
```

### Preprocessing and matching views

Gower matching uses the original declared numeric and categorical features. One-hot
encoding and the numeric-metric log/bin pipeline do not replace these matching columns.
For a transformed numeric feature in Gower, create the column in the input and declare
it as a matching feature.

For a non-Gower metric, categorical covariates require `onehot=True`. Numeric
columns without requested log/bin transformations pass through when
`bin_method=None`; enabling preprocessing with a scaling/binning policy changes the
numeric representation. The pipeline uses `log1p` for requested log
features; values outside its mathematical domain need handling before matching.
`matching_data_` records the representation used for matching; `pop_processed` records
the initial preprocessing view.

The preprocessing helpers are available in [processing.py](src/rdmatcher/processing.py): `build_preprocessing_pipeline`,
`apply_preprocessing_pipeline`, and `build_propensity_preprocessor`. They are useful when an explicit
fitted transformation and feature map are needed outside the high-level interface.

### Public methods

| Method | Purpose and return |
| --- | --- |
| `match(...)` | Construct candidates, allocate without replacement, return the population-record table. |
| `return_graph(...)` | Construct and return a `CandidateGraph` before allocation. |
| `process_features()` | Populate the preprocessing view; repeated calls skip already completed processing. |
| `fit_propensity_model(formula=None, random_state=404, **kwargs)` | Fit scores/logits and store aligned columns and model metadata; inspect attributes rather than expecting a returned table. |
| `get_propensity_feature_map()` | Return the last propensity formula, design columns, categorical references and fit metadata. |
| `propensity_eligible_sets(...)` | Return per-case arrays of control positions for native propensity-caliper eligibility. |
| `plot_propensity_coverage(compare_matching=False, figsize=(10, 6), save_path=None)` | Plot fitted logit distributions before/after matching; returns figure and axes. |

## Complete matching and execution reference

### Matching call

```text
match(threshold, ratio=1, *, distance_metric='gower',
      allocation='optimal', matching=None, execution=None)
match(request=MatchRequest(...))
return_graph(threshold, ratio=1, *, distance_metric='gower',
             matching=None, execution=None)
```

`threshold` is required, nonnegative and not NaN; positive infinity is
allowed. `ratio` must be a positive integer (not a boolean) and is controls
per case.
`allocation` accepts `"optimal"` or `"hybrid"`. Passing
`None` for the settings constructs the default immutable dataclasses.
Loose matching/execution keyword arguments are not accepted by these public methods.

For reusable configurations, collect the complete request in `MatchRequest`:

```python
from rdmatcher import ExecutionOptions, MatchRequest, MatchingOptions

request = MatchRequest(
    threshold=float("inf"),
    ratio=4,
    matching=MatchingOptions(prioritize_coverage=True),
    execution=ExecutionOptions(n_jobs=1),
)
matched = model.match(request)
graph = model.return_graph(request)
```

### MatchingOptions

The following table lists every field and its default value.

| Field | Default | Meaning |
| --- | --- | --- |
| `method` | `"multi"` | Match on covariates; `"propensity"` instead uses raw logit differences. |
| `prioritize_coverage` | `False` | Enable covered-case, filled-slot, distance priorities. |
| `gower_weights` | `None` | Mapping from original matching-feature names to weights; omitted features use 1. |
| `propensity_col` | `"propensity_logit"` | Logit column used by propensity-only or PS-restricted matching. |
| `propensity_caliper` | `None` | In propensity-only mode, replace `threshold` by this multiplier × full-cohort sample SD of logits. |
| `ps_hybrid` | `False` | Restrict multi-covariate eligibility to propensity neighborhoods; supported with Gower or Mahalanobis. |
| `ps_caliper` | `0.2` | PS-hybrid logit caliper in multiples of the full-cohort sample SD. |
| `ps_caliper_strict` | `True` | Keep the specified PS caliper; when false, an empty neighborhood is retried at twice its width. |
| `gower_sd_weights` | `False` | Build SD-calibrated Gower weights. |
| `gower_sd_reference` | `"controls"` | Reference for automatic weights; `"pooled"` includes cases. |
| `gower_sd_weights_mult` | `1.96` | Positive SD multiplier in automatic numeric weights. |
| `weight_numeric` | `1.0` | Scalar multiplier of numeric-metric covariate coordinates; not a Gower feature-weight dictionary. |
| `weight_propensity` | `1.0` | Scalar for a designated propensity coordinate when one is passed through the lower-level numeric construction; see limitations below. |
| `return_unmatched_cases` | `True` | Request inclusion of unmatched cases; the current false-setting limitation is described under Outputs. |
| `pca_filter` | `False` | Apply the lower-level numeric PCA reduction; this changes the representation and distance problem. |

In current public `method="multi"` matching, covariates are selected from the
declared numeric/categorical lists. `weight_propensity` does not automatically add a
fitted propensity column to those lists. In propensity-only mode the implementation
fixes numeric/propensity multipliers to 0/1. For Gower, use `gower_weights`;
numeric-coordinate multipliers do not substitute for those weights. Scalar
Euclidean-coordinate multipliers act before the distance, so their squared contribution
appears in squared Euclidean distance.

Caliper and scalar-weight options must be finite and nonnegative. The SD multiplier must
be positive. A propensity column name must be nonempty. Explicit and SD-calibrated Gower
weights are mutually exclusive. SD-calibrated Gower is supported only for
`method="multi"`, `distance_metric="gower"`. See [options.py](src/rdmatcher/options.py) for the
dataclass validators.

### ExecutionOptions

| Field | Default | Meaning |
| --- | --- | --- |
| `n_jobs` | `1` | Distance/retrieval parallelism; positive integer or `-1` for all CPUs as determined by the backend. |
| `batch_size` | `10` | Positive neighbor-query batch size used by candidate construction. |
| `solver` | `"scipy_sparse"` | Global solver: `"scipy_sparse"`, `"hungarian"` or optional `"mcf"`. |
| `streaming` | `"auto"` | Gower streaming policy: `"auto"`, `"on"`, `"off"`. |
| `stream_block_size` | `None` | Positive explicit control-block size; `None` leaves backend sizing in effect. |
| `stream_threshold_gb` | `1.0` | Positive estimated matrix-size threshold for automatic streaming, using the implementation's binary-size calculation. |
| `memory_limit_gb` | `None` | Positive warning/merge-strategy memory setting; not a process-wide RAM cap. |
| `parallel_chunk_size` | `None` | Positive explicit query chunk size for supported internal parallel paths. |

Streaming and memory options apply to the supporting distance backends; dense global
assignment still requires its own matrix. `n_jobs` controls internal
parallel work. Independent matching processes, CPU allocation and numerical- library
thread limits must be configured separately.

## Outputs, balance and inspection

### Returned table and object attributes

| Output | Contents |
| --- | --- |
| Returned DataFrame / `pop_matched` | Original covariates and match metadata, including unmatched cases by default. |
| `matched_data` | Lower-level matching table used to link back to original records. |
| `matched_exposed`, `matched_control` | Role subsets created during linking; see current caveats below before counting them. |
| `unmatched_exposed` | Linking-stage unmatched-case attribute; inspect explicit row metadata for unmatched-case accounting. |
| `summary_table` | Numeric/categorical SMD diagnostics from the table passed to the diagnostic routine. |
| `assignment_diagnostics_` | Certificates, retrieval details, graph hash, solver, timing and allocation counters. |
| `matching_candidate_diagnostics` | PS-hybrid candidate diagnostics when available. |
| `matching_data_` | The actual feature/role/ID view passed into matching. |
| `gower_weights_`, `gower_sd_reference_` | Resolved weight specification and automatic-weight reference, when set. |
| `candidate_graph_` | Most recently exported graph. |
| `pop`, `pop_processed` | Combined original population view and preprocessing view. |

`match_group` links a case and its assigned controls. On case rows,
`n_matches` is the number assigned; on control rows it is NaN. A matched
case's `match_distance` is 0 and each assigned control carries its case-to-control
distance. Unmatched cases have `n_matches == 0` and missing distance. Interpret
metadata by row role: NaN in a control's `n_matches` column is expected.
Population construction combines controls first, then cases. Join outputs to inputs
using the supplied unique ID.

### Filtering matched and unmatched records

Role attributes are constructed from the lower-level table before final filtering. That
table includes unmatched cases. Count positive case `n_matches` to measure
coverage instead of using the sizes of `matched_exposed` or `unmatched_exposed`.
Automatic diagnostics use the same table before filtering and include unmatched cases
when present.

**Known issue in 0.2.0.dev0:** `return_unmatched_cases=False` filters all rows on
`n_matches > 0`. Control rows carry NaN in this column and are omitted by that
filter. For this version, keep the default and construct the retained table explicitly
as shown below.

```python
case_rows = matched.loc[matched.treated.eq(1)]
covered_cases = case_rows.loc[case_rows.n_matches.gt(0)]
unmatched_cases = case_rows.loc[case_rows.n_matches.eq(0)]
assigned_controls = matched.loc[matched.treated.eq(0)]
retained = matched.loc[matched.treated.eq(0) | matched.n_matches.gt(0)].copy()

assert assigned_controls.patient_id.is_unique
assert covered_cases.n_matches.le(1).all()  # quickstart requested 1:1
assert len(covered_cases) + len(unmatched_cases) == 30

from rdmatcher.matching import summary_stats_table

retained_balance = summary_stats_table(
    retained,
    features_numeric=["age", "bmi", "lab_value"],
    features_categorical=["sex", "site"],
    exposure_col="treated",
)
```

Inspect coverage, per-case ratios, control uniqueness and balance together. SMD
describes covariate balance; a small SMD or assignment objective alone does not
establish exchangeability or adequate overlap. Analysis weights for variable-ratio sets,
outcome modeling and treatment-effect uncertainty are handled outside these matching
routines.

### Assignment diagnostics

Useful keys include `preservation_condition_satisfied`, `preservation_condition_by_case`, `preservation_certificate_types`,
`cases_failing_preservation_condition`, `eligible_set_retained`, `safe_control_counts`,
`competitive_enabled`, `solver`, `initial_candidate_graph_sha256`,
`preallocated_control_count`, `residual_requested_slots`, `competitive_allocation_sec`,
`global_assignment_sec` and the nested `candidate_retrieval` dictionary. Graph export
returns after candidate construction and before allocation, so allocation-specific
counters and timings are not yet populated. Check which keys are available for the type
of call being inspected.

## Math: Gower distance (implementation details)

Gower handles mixed numeric and categorical features. RDMatcher uses
`features_categorical` as the source of truth for nominal categorical columns. For
numeric features we compute a range‑normalized absolute difference; for categorical
features the contribution is 0 when equal and 1 when different. Missing values are
handled by excluding that feature's weight from the denominator on a per‑pair basis.

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

- If the denominator $\sum_j w_j m_j(x_j,y_j)$ is zero (no comparable features for the pair), the implementation explicitly sets $d(x,y)=1.0$ as its fallback distance.
- Categorical missing values are encoded internally and excluded via the mask $m_j$.
- Feature weights default to equal weights when none are supplied; internally numeric and categorical weights are stored separately.
- Computations use float32 for memory efficiency; small numerical tolerances are present when comparing to matching thresholds.
- True nominal categorical Gower changes threshold interpretation. With six equally weighted complete features, one categorical mismatch contributes about `1/6 = 0.167` to distance before numeric differences are added. Thresholds that were tuned under integer-coded dtype inference are not comparable to true categorical Gower thresholds.

### SD-calibrated Gower weights

Standard Gower normalizes numeric features by the fitted reference range. Extreme
reference values increase that range and reduce the contribution of a fixed numeric
difference. SD-calibrated weights adjust the numeric contribution using the reference
standard deviation.

Set `gower_sd_weights=True` in `MatchingOptions` to automatically build weights from
the control/reference pool:

```python
matcher.match(
    threshold=0.25,
    ratio=1,
    distance_metric="gower",
    matching=MatchingOptions(
        gower_sd_weights=True,
        gower_sd_reference="controls",
        gower_sd_weights_mult=1.96,
    ),
)
```

For each numeric feature, RDMatcher computes a raw weight proportional to:

$$
\frac{\text{range}_j}{\text{gower\_sd\_weights\_mult} \times \text{SD}_j}
$$

With `gower_sd_reference="controls"`, calibration uses the control cohort's SD and range. With
`gower_sd_reference="pooled"`, calibration uses cases and controls from the same matching view.
The Gower distance formula is unchanged; only the feature weights change. Resolved
weights are logged and stored as `matcher.gower_weights_`.

Gower is fitted on the controls. Numeric differences are divided by the control range
without clipping to [0, 1]. Differences larger than the control range therefore
contribute more than one. A threshold such as 0.2 is a distance cutoff. Entirely missing numeric columns require handling before fitting because their reference ranges are undefined.

For controls-based SD weights, the numeric numerator simplifies to approximately
`abs(x - y) / (multiplier * SD_controls)` before division by the sum of comparable weights. The denominator
still contains the weights, so changing weights changes the scale of the final distance
as well as relative feature importance. With `gower_sd_reference="pooled"`, weights use pooled
ranges and SDs but the fitted Gower normalizer still uses control ranges; the
cancellation applies only when the calibration and normalization ranges coincide.
Compare thresholds and balance anew when changing this setting.

Degenerate SD/range ratios fall back to weight 1, and resolved numeric weights have a
positive floor. Automatic calibration uses the declared numeric and categorical feature
lists. Inspect `gower_weights_` and `matching_data_` to confirm the weights
and columns used for matching.

### Explicit feature weights

Use original feature names and nonnegative, finite values. Unspecified matching features
default to weight 1. Zero excludes a feature's contribution and its denominator weight.
Unknown top-level feature names raise an error. Although the underlying resolver maps
two-level dictionaries to processed child columns, the public Gower path uses raw
nominal categories; use one scalar per original categorical feature rather than
supplying level-specific dummy weights. Pass a Python dict; ordered lists are not
accepted by `MatchingOptions`. Do not combine explicit weights and SD-calibrated
weights.

```python
weighted = matcher.match(
    threshold=0.3,
    distance_metric="gower",
    matching=MatchingOptions(
        gower_weights={"age": 2.0, "bmi": 1.0, "lab_value": 1.5,
                       "sex": 1.0, "site": 1.0},
        prioritize_coverage=True,
    ),
)
resolved_weights = matcher.gower_weights_
```

Multiplying all weights by the same positive constant leaves the mathematical weighted
average unchanged; changing their relative values changes the distance. The option
validator checks the mapping type but leaves mapped-value validation to the caller. Use
finite, nonnegative weights. An all-zero comparable denominator returns the fallback
distance defined above.

### Inspecting distances directly

`GowerKNN` computes and retrieves distances independently of assignment,
score fitting and case/control-role validation. For DataFrame input,
`cat_features` accepts a list of categorical column names. Reference fitting uses
a fixed default shuffle seed; returned neighbor indices map to the original reference
rows, not the internal shuffled storage.

```python
from rdmatcher.distance import GowerKNN

features = ["age", "bmi", "lab_value", "sex", "site"]
reference = population.loc[population.treated.eq(0), features]
queries = population.loc[population.treated.eq(1), features]
gower = GowerKNN(cat_features=["sex", "site"], n_jobs=1).fit(reference)
distances, positions = gower.kneighbors(queries.iloc[:3], n_neighbors=2)
pairwise_distances = gower.cdist(queries.iloc[:3])
assert distances.shape == (3, 2)
assert pairwise_distances.shape == (3, len(reference))
```

`cdist` returns the full query-by-reference distance matrix for the
supplied batch. Use a small query batch when inspecting distances to limit the size of
that matrix. See [distance.py](src/rdmatcher/distance.py) for the implementation.

## Numeric distances and Mahalanobis matching

### Euclidean, Manhattan and cosine

For numeric representations, Euclidean distance is the square root of summed squared
coordinate differences; Manhattan distance sums absolute differences. Cosine measures
angular dissimilarity between normalized vectors, comparing direction rather than
magnitude. Its range is 0 to 2. Zero vectors require an explicit interpretation in the
input representation. The backend uses its normalization/dot-product convention without
a separate missing-data policy.

Numeric scales determine contributions to Euclidean and Manhattan distance. The raw
representation uses coordinate differences in the original feature units. Supply
standardized numeric features or configure preprocessing when scaling is needed. Raw
mixed columns with `process_features=False` still require a numeric representation.

```python
numeric_matcher = RDMatcher(
    pop_df=population,
    patient_id_col="patient_id",
    exposure_status="treated",
    features_numeric=["age", "bmi", "lab_value"],
    features_categorical=["sex", "site"],
    process_features=True,
    bin_method="scaler",
    onehot=True,
    onehot_drop=None,
)

euclidean_matches = numeric_matcher.match(
    threshold=2.0,
    distance_metric="euclidean",
    matching=MatchingOptions(prioritize_coverage=True),
)
manhattan_graph = numeric_matcher.return_graph(
    threshold=float("inf"), distance_metric="manhattan"
)
cosine_graph = numeric_matcher.return_graph(
    threshold=0.3, distance_metric="cosine"
)
```

The numeric backend selects Euclidean/Manhattan indexing automatically. Cosine
normalizes the fitted controls once and reuses capped candidate pools for later prefix
requests. Indexing and pool reuse reduce repeated work in the supported paths; the
worst-case distance-search complexity remains unchanged.

### Mahalanobis distance and covariance choice

Mahalanobis distance accounts for covariance in a numeric representation:

$$
d(x,y)=\sqrt{(x-y)^\top\Sigma^{-1}(x-y)}.
$$

The returned distance is the **square-rooted** distance, not its square. Thresholds are
compared directly with that value. For example, a threshold of 2 is in
Mahalanobis-distance units.

The high-level path estimates a regularized covariance matrix from the pooled
case/control numeric representation and searches for exact Euclidean neighbors after
whitening. Categorical covariates require an explicit numeric encoding. Use
`onehot=True` for the public encoding path; different category encodings define
different distance representations.

```python
mahalanobis_matches = numeric_matcher.match(
    threshold=3.0,
    ratio=1,
    distance_metric="mahalanobis",
    matching=MatchingOptions(prioritize_coverage=True),
    execution=ExecutionOptions(n_jobs=1),
)
```

The lower-level estimator offers explicit covariance and backend control:

```python
from rdmatcher.mahalanobis import MahalanobisKNN

numeric_columns = ["age", "bmi", "lab_value"]
X_control = population.loc[population.treated.eq(0), numeric_columns]
X_treated = population.loc[population.treated.eq(1), numeric_columns]
X_pooled = population[numeric_columns]

maha = MahalanobisKNN(neighbor_backend="sklearn", algorithm="auto", n_jobs=1)
maha.fit(X_control, cov_source="pooled", pooled_X=X_pooled)
maha_distances, maha_positions = maha.kneighbors(X_treated.iloc[:3], n_neighbors=2)
assert maha_distances.shape == (3, 2)
```

| Low-level argument | Default | Meaning |
| --- | --- | --- |
| `regularization` | `1e-6` | Initial covariance regularization; the estimator adapts it if necessary. |
| `neighbor_backend` | `"cdist"` | Full-distance backend; `"sklearn"` searches whitened numeric vectors. The high-level default is the latter. |
| `algorithm` | `"auto"` | Neighbor-search policy for the sklearn backend. |
| `n_jobs` | `1` | Neighbor parallelism; `-1` requests all backend-visible CPUs. |
| `fit(..., cov_source=...)` | `"reference"` | Use the fitted reference/control covariance, or `"pooled"` with `pooled_X`. |
| `fit(..., VI=...)` | `None` | Supply a precision matrix with shape `(features, features)`. |

`MahalanobisKNN` requires numeric two-dimensional inputs. During reference fitting
it removes rows containing NaN and maintains mappings to original reference positions. A
missing query row returns invalid/missing neighbor information; do not treat index
`-1` as a valid reference. An explicit `Y_ref` with
missing values in `cdist` is rejected. Pooled covariance data also need a
defined finite covariance; the reference-row policy is not an imputation method for
arbitrary missing pooled inputs. Prepare complete numeric data for the high-level
workflow rather than relying on backend-specific missing behavior.

Precision and whitening matrices are available as `precision_` and
`whitener_`, with `covariance_` recording the fitted covariance.
Distances and candidate storage use float32 in the candidate path, while covariance
estimation uses float64. Compare distances and objectives numerically when checking
nearly tied results across backends. See [mahalanobis.py](src/rdmatcher/mahalanobis.py).

## Propensity-score modeling

Propensity-score estimation is optional. Covariate matching uses the declared features
without a fitted score. Score-based options provide scalar logit matching,
propensity-caliper eligibility for multivariable matching, and an explicitly declared
score feature in the distance.

### Fitting, alignment and model metadata

`fit_propensity_model` builds a separate modeling view, with numeric processing and
categorical encoding, fits a model, and joins score/logit columns back by unique patient
ID. The modeling view does not replace raw nominal Gower features. Re-fitting replaces
prior score columns rather than duplicating them.

```python
matcher.fit_propensity_model(
    formula="age + bmi + lab_value + sex + site",
    random_state=404,
)
propensity_metadata = matcher.get_propensity_feature_map()
assert "propensity_score" in matcher.pop
assert "propensity_logit" in matcher.pop
assert matcher.propensity_logits.index.is_unique
```

`propensity_score` is a probability and `propensity_logit` is the corresponding
log-odds. The latter is the distance/caliper scale used by the native propensity
matching paths. Score alignment is explicit and checked: missing joins or duplicate
identifiers are errors. `propensity_logits` is stored as an ID-indexed Series. The fit
method populates attributes; it does not return a matched cohort.

`get_propensity_feature_map()` exposes the original formula, parsed terms, design-matrix column
names, original-to-processed feature mappings, omitted categorical reference levels, and
optimizer metadata. Preserve those details along with the input IDs/order and fitted
score columns for reproducibility. This retains the resolved model specification as well
as the formula string.

### Formula representation

Use original feature names. The parser supports `+`,
`:`, `*` and parentheses: `age + sex`,
`age:sex`, or `(age + bmi) * sex`. An interaction with a categorical
feature expands through its encoded modeling columns. `formula=None` uses all
declared original numeric/categorical features as main effects.

The formula parser supports the operators listed above rather than the full R or Patsy
syntax. Response prefixes such as `treated ~ ...`, function-call transformations,
subtraction and arbitrary Python expressions are outside that syntax. Create transformed
columns before fitting and refer to them by name. Inspect the resolved design columns
and categorical reference levels afterward.

### Estimator and fitting strategies

The default estimator is logistic regression with `max_iter=10000`,
`class_weight=None`, `solver="newton-cholesky"`, `C=inf` and
`tol=1e-10`. The helper removes an unsupported `penalty` keyword before
fitting. These defaults require a scikit-learn version supporting the selected solver
and parameter values. The package metadata does not pin scikit-learn to a particular
version.

| Option to `fit_propensity_model` | Purpose |
| --- | --- |
| `random_state` | Default 404; controls supported randomized fitting operations. |
| `downsample_ratio` | Train with a sampled control-to-case ratio, then score the population. |
| `n_bags` | Fit a bagging ensemble and combine predictions. |
| `n_hard_mining` | Use the two-stage scout/filter/refit strategy with the specified hard-control count. |
| `estimator` | Custom compatible sklearn-style estimator factory/class. |
| `estimator_kwargs` | Dictionary of estimator settings, overriding duplicate loose estimator kwargs. |
| Additional keyword arguments | Forwarded as model settings, such as `solver`, `tol`, `max_iter` and `class_weight`. |

Only one of downsampling, bagging and hard-negative mining is supported at a time; the
current helper rejects simultaneous positive strategy settings. The sampling/model
strategy affects fitted probabilities and the resulting caliper neighborhoods as well as
fitting cost. Check convergence, separation, score support and model metadata before
using the fitted scores for matching.

### Including a score as a Gower feature

Fitting scores adds score columns to the population view, but multi-covariate matching
still selects declared feature lists. To include a supplied logit in Gower distance,
explicitly declare it as a numeric feature in a new matcher:

```python
score_augmented = RDMatcher(
    pop_df=matcher.pop.copy(),
    patient_id_col="patient_id",
    exposure_status="treated",
    features_numeric=["age", "bmi", "lab_value", "propensity_logit"],
    features_categorical=["sex", "site"],
)
score_augmented_graph = score_augmented.return_graph(
    threshold=0.3,
    matching=MatchingOptions(gower_weights={"propensity_logit": 0.5}),
)
```

This uses the score as another range-normalized numeric covariate. It differs from
propensity-only matching and from a propensity caliper. Because a score is derived from
covariates already represented in the distance, adding it introduces another
contribution from those covariates. Examine balance and sensitivity to its relative
weight. `weight_propensity` alone does not add this coordinate in public multi
matching.

## Propensity-only matching

Fit scores first. Use `method="propensity"` with `distance_metric="euclidean"`; the
implementation then uses raw absolute **logit** differences without covariate
standardization. Other distance-metric strings are rejected in this mode. Setting
`propensity_caliper` replaces the supplied threshold by a multiplier times the sample
SD of the full population's logits (`ddof=1`).

```python
propensity_matches = matcher.match(
    threshold=float("inf"),
    ratio=1,
    distance_metric="euclidean",
    matching=MatchingOptions(
        method="propensity",
        propensity_caliper=0.2,
        prioritize_coverage=True,
    ),
)
propensity_distance_metadata = matcher.propensity_matching_metadata_
```

Without `propensity_caliper`, `threshold` is already in raw-logit distance
units. A multiplier of 0.2 specifies a logit-SD caliper. Record the caliper convention, reference SD and score representation. This path
requires the fitted propensity state in addition to the score column. When integrating
externally prepared scores, preserve those scores and their eligibility definition
explicitly; fitting again changes the prepared inputs.

## PS-restricted Gower and Mahalanobis matching

Here the propensity logit defines eligible neighborhoods, and the multivariable distance
ranks controls **within** each neighborhood. Enable `ps_hybrid=True` with
`method="multi"` and either Gower or Mahalanobis. This setting concerns
eligibility; it is independent of `allocation="hybrid"`. PS-restricted matching supports
Optimal allocation. Hybrid allocation is also available without a propensity caliper.

```python
ps_restricted = matcher.match(
    threshold=0.3,
    ratio=1,
    distance_metric="gower",
    allocation="optimal",
    matching=MatchingOptions(
        method="multi",
        ps_hybrid=True,
        ps_caliper=0.2,
        ps_caliper_strict=True,
        gower_sd_weights=True,
        gower_sd_reference="controls",
        prioritize_coverage=True,
    ),
)
neighborhoods = matcher.propensity_eligible_sets(
    ps_caliper=0.2,
    ps_caliper_strict=True,
)
```

The native caliper is `ps_caliper * SD(full-cohort propensity_logit, ddof=1)` and uses an inclusive `<=`
comparison. With strict eligibility, empty neighborhoods remain empty. With
`ps_caliper_strict=False`, only an initially empty neighborhood is retried at twice the
width; it is not an unrestricted nearest-control fallback. That relaxation changes the
eligibility policy and must be recorded if used.

Returned neighborhood arrays contain **control positions**, not patient IDs. They follow
the current population view's control order. A control shared by multiple neighborhoods
connects those cases in the same assignment problem. The final allocation uses each
control at most once.

The ordinary multivariable feature lists do not automatically include the fitted logit,
so fitting scores and enabling PS eligibility does not itself double-count the logit in
Gower. Adding a score to the numeric feature list includes it in the distance.
For caliper-only use, exclude the logit from the distance features. The distance
threshold still applies inside eligible neighborhoods.

## Exporting the candidate graph

`return_graph` applies the same representation, eligibility, retrieval and
preservation checks as matching, but stops before allocation. Each call builds its own
graph; requesting a graph and then calling `match` does not reuse the
exported object as an allocation cache.

```python
graph = matcher.return_graph(
    threshold=0.3,
    ratio=1,
    distance_metric="gower",
    matching=MatchingOptions(prioritize_coverage=True),
    execution=ExecutionOptions(n_jobs=1),
)

distance_matrix = graph.matrix
case_ids = graph.case_ids
control_ids = graph.control_ids
assert graph.shape == (len(case_ids), len(control_ids))
assert graph.nnz == distance_matrix.nnz
row = 0
begin, end = distance_matrix.indptr[row:row + 2]
candidate_ids = control_ids[distance_matrix.indices[begin:end]]
candidate_distances = distance_matrix.data[begin:end]
```

| Attribute | Meaning |
| --- | --- |
| `matrix` | SciPy CSR matrix of stored candidate distances: rows are cases, columns are controls. |
| `case_ids`, `control_ids` | Original patient-ID mappings for the two matrix axes. |
| `shape`, `nnz` | Matrix dimensions and number of stored edges. |
| `distance_metric`, `threshold` | Metric name and applied distance cutoff. |
| `candidate_horizons` | Final per-case candidate horizons from construction. |

An absent sparse entry means no retained edge, whereas a **stored zero** is a valid
zero-distance edge. Inspect CSR structure/data to distinguish them;
`matrix[i, j] == 0` alone cannot. Do not call `eliminate_zeros()` on a graph whose
zero-distance edges are retained.

`graph.for_scipy_sparse_matching()` returns a copied graph with 1 added to stored edge costs so
zero-distance edges survive SciPy's sparse matching convention. The uniform shift
preserves the mathematical objective for fixed-cardinality assignments, subject to
floating-point precision. It does not add unmatched dummy slots, encode coverage
priority, enforce a ratio greater than one, or make an infeasible full matching
feasible. Use the package allocation path if you need those semantics. See
[candidate_graph.py](src/rdmatcher/candidate_graph.py).

## Global solvers and numerical precision

Choose a global backend in `ExecutionOptions`, independently of Optimal/Hybrid:

| `solver` | Representation | Dependency / trade-off |
| --- | --- | --- |
| `"scipy_sparse"` | Sparse minimum-weight bipartite assignment with dummy slots. | Default, supplied by SciPy; retains sparse structure. |
| `"hungarian"` | Dense assignment matrix with ineligible/dummy costs. | Supplied by SciPy; memory scales with the dense residual matrix. |
| `"mcf"` | Min-cost-flow network with integerized costs. | Optional OR-Tools; inspect numerical scaling and rounding. |

```python
dense_solver_matches = matcher.match(
    threshold=0.3,
    matching=MatchingOptions(prioritize_coverage=True),
    execution=ExecutionOptions(solver="hungarian"),
)
```

All three global backends use already computed candidate distances in the normal public
path. Solvers handle residual case demand, control uniqueness and unmatched slots;
choosing one does not change the earlier competitive allocation policy. Request the
optional MCF solver after installing the `mcf` extra; it is not needed
to import or run the default solver.

The sparse path uses a positive cost shift so true zero-cost edges remain represented.
The solver compares shifted floating-point costs. Original costs that become equal after
shifting are indistinguishable to the solver. A candidate preservation certificate
cannot repair numerical loss in a solver. Compare objectives on the original distances
when investigating small differences. Different pair sets are equivalent when they
satisfy the same assignment priorities and have equal original-distance cost.

The MCF backend uses integer cost scaling (recorded as `mcf_cost_scale` in
diagnostics, with a default of 1,000,000). Costs that round to the same integer are
indistinguishable to that solver. The current high-level execution options do not expose
a cost-scale parameter. Dense assignment avoids that integerization, but also has finite
precision and stores a dense residual matrix. Check cross-solver objectives using the
original distances; eligibility thresholds and candidate certificates describe the
graph, while numerical precision is determined by the solver.

## Complexity, memory and concurrency

Let `T` be the number of cases, `C` the controls,
`F` the feature count and `E` the retained edges.
The internal per-case cap is `k = min(C, T * ratio)`; the final graph has at most
`T * k` candidate edges, often fewer after threshold and Safe stopping.
This bound depends on the case count, control count and requested ratio. When the cap
reaches `C`, the bound is `T * C`, the number of edges
in the full case-control graph.

### Distance and retrieval work

An exact fallback that compares all cases to all controls requires `O(T * C * F)`
arithmetic. Gower has an exact grouped L1-tree acceleration for compatible data and
query settings. Numeric metrics use exact neighbor indices or full scans according to
the selected backend. Those accelerations depend on dimensionality, category patterns,
missingness, requested prefix size and overlap. Their favorable behavior does not
eliminate the all-pairs worst case.

Candidate categorization scans the produced edge lists and is linear in the number of
entries scanned. Distance retrieval has its own cost. Indexed expansion repeats queries
as prefixes grow; scan-backed retrieval stores a capped pool and reuses its prefixes.
Boundary ties trigger full-cap retrieval. The final graph contains the retained edges,
while retrieval and storage counters also include candidates excluded during reduction.

### Assignment work

Global assignment has superlinear worst-case cost in the residual graph or matrix
dimensions. Dense rectangular assignment also expands case demand into requested slots
and adds unmatched alternatives. Sparse/MCF formulations represent the retained network
without materializing every absent edge; their memory and runtime depend on the residual
network as well as `E`. Hybrid removes committed assignments from the
residual demands. The residual solution remains conditional on those commitments.

### Streaming and memory controls

The Gower fallback processes control blocks and maintains top-k candidates without
holding the full query-by-control distance matrix in streaming mode.
`streaming="auto"` considers an estimated matrix size, `"on"` requests
the supported streaming path and `"off"` disables it.
`stream_block_size=None` leaves backend sizing; set an explicit positive size when you need
reproducible blocking. A smaller block trades overhead for temporary-memory reduction.

`memory_limit_gb` influences warnings and vectorized-versus-row-wise merge strategy
in relevant paths. Where no explicit value is supplied, those paths consult
`RD_MATCHER_MEMORY_LIMIT_GB` and then a 4 GiB fallback. These calculations use powers of 1024
even though parameter names say `gb`. The setting controls
distance-kernel warnings and merge strategy, not OS memory allocation or dense
assignment storage. Stored candidates, fitted indices, data copies and solver buffers
all contribute to resident memory.

### Parallel execution

Gower and precomputed-matrix retrieval parallelize independent query rows with
threads when `n_jobs > 1`; numeric backends apply their own `n_jobs` policies.
Precomputed matrix retrieval bounds its row-selection pool at 16 workers
because each worker may allocate a control-row-sized selection buffer.
Concurrent query chunks allocate separate temporary buffers. For small
workloads, thread scheduling adds overhead relative to a single-threaded call.
A positive explicit `n_jobs` is easier to interpret than `-1` when running
multiple independent jobs.

For externally parallel experiments, set BLAS/OpenMP thread limits before starting
Python and record both the number of matching processes and threads inside each process.
`ExecutionOptions(n_jobs=1)` alone does not set every numerical library's thread count. CPU
affinity, concurrency, resource contention, sampling interval and what is included in
the timed region affect reported performance. CPU affinity and hardware allocation are
managed outside the package.

## Reproducibility and troubleshooting

### What to save for a matching run

- The prepared population or its saved location, its hash, unique IDs and
  declared feature lists/types, including any externally created transformations.
- Package/source version and numerical dependency versions; the selected distance,
  threshold, ratio, allocation mode, and every matching/execution option.
- Fitted score columns, design metadata and eligibility policy if propensity
  modeling or external prepared scores are part of the analysis.
- Resolved Gower weights or the Mahalanobis covariance/precision policy, the
  candidate diagnostics/graph hash, and original-distance allocation checks.
- Covered and uncovered cases, assigned controls and ratio distribution, balance
  summaries calculated on the analysis sample, and any analysis weights.
- Hardware, process/worker counts, thread limits and timing/RAM definitions when
  interpreting resource measurements.

Use the same input snapshot, fitted scores, weights and feature scales when repeating a
matching problem. Changing a seed, fitted score or feature scale changes the matching
inputs. Retain failures and empty neighborhoods when summarizing coverage and runtime.

### Common issues

| Symptom | Checks and explanation |
| --- | --- |
| Unexpected keyword argument | Match the current public signatures; move settings into the two option objects and remove retired keywords. |
| Unknown Gower weight key | Use an original declared matching-feature name. A fitted score is not automatically a declared distance feature. |
| Non-Gower categorical error | Set `onehot=True` or prepare a numeric-only representation explicitly. |
| Scores exist as columns but propensity mode refuses to run | The native score paths require fitted propensity state, not just a similarly named column. Preserve supplied scores through an explicit prepared-input integration. |
| PS-hybrid rejects a metric | Its supported multivariable metrics are Gower and Mahalanobis. |
| No/too few matches | Check feature representation, distance units, threshold, score-caliper neighborhoods and competition for controls. An infinite distance cutoff does not disable a separate PS caliper. |
| Cases unexpectedly counted as matched | Count positive case `n_matches`; role attributes and default output include unmatched cases. |
| Missing controls after suppressing unmatched cases | Keep the default output flag and construct the retained subset explicitly; see the documented filtering limitation. |
| Balance appears inconsistent with retained cases | Recompute diagnostics on the explicitly retained analysis table; automatic diagnostics run before final optional filtering. |
| Very large memory usage | Check stored versus retained edges, cap growth with cases × ratio, numeric dimensions, query chunks, concurrency and dense-solver buffers. |
| Slow retrieval with many ties | Boundary ties trigger full-cap loading; inspect retrieval diagnostics and the number of stored entries. |
| Prefix consistency error | A backend changed an already loaded prefix. Preserve the failing input/settings for investigation; the current path does not silently fall back to eager retrieval. |
| Small objective differences between solvers | Compare original-distance objectives and inspect shifts, float32 storage, integer rounding and true ties. Candidate certificates concern graph reduction. |
| Propensity fit fails or warns | Check missing/nonfinite inputs, formula expansion, separation, solver support and convergence metadata. Use only one sampling strategy at a time. |
| Missing optional MCF dependency | Install `.[mcf]` in the interpreter actually running the job, or use a SciPy solver. |
| Import appears to use old behavior | Inspect `rdmatcher.__file__`, the active environment and editable-install/source-path configuration. |

### Missingness and categorical semantics

Gower implements per-pair comparable-feature masks; that is not a universal missing-data
policy for every metric or every model fit. Numeric neighbor backends require finite
encoded arrays, and propensity fitting has its own modeling constraints. Decide on
missingness handling before analysis and inspect resulting coverage rather than
expecting the library to impute a cohort.

Declare nominal categories explicitly, including integer-coded nominal variables. One
categorical mismatch among six equally weighted, fully observed features contributes
approximately one sixth of the total Gower distance. A threshold chosen when such a
column was incorrectly treated as a continuous code does not keep the same meaning after
correcting the feature declaration. Likewise, one-hot Euclidean geometry differs from
nominal categorical Gower geometry.

## Tests and source map

From this package checkout with the `test` extra installed:

```sh
python -m pytest tests -q
```

Tests cover public option validation, raw categorical semantics, prefix loading, numeric
metrics, solver/certificate behavior and propensity integrations. Optional dependencies
and backend-specific tests have their own requirements. Check the input contract and
numerical scales for each application in addition to running the package tests.

| Source | Responsibility |
| --- | --- |
| [RDMatcher.py](src/rdmatcher/RDMatcher.py) | Public orchestration, views, scores, eligibility, outputs. |
| [options.py](src/rdmatcher/options.py) | Matching and execution dataclasses and validators. |
| [matcher.py](src/rdmatcher/matcher.py) | Candidate restriction, certificates, competitive and global allocation. |
| [distance.py](src/rdmatcher/distance.py) | Gower distances, masks, weights, indexing and streaming. |
| [mahalanobis.py](src/rdmatcher/mahalanobis.py) | Covariance, whitening and lower-level Mahalanobis search. |
| [_cosine_neighbors.py](src/rdmatcher/_cosine_neighbors.py) | Normalization reuse and cosine search. |
| [matching.py](src/rdmatcher/matching.py) | Assignment solvers, penalties and balance summaries. |
| [candidate_graph.py](src/rdmatcher/candidate_graph.py) | CSR export and ID mappings. |
| [processing.py](src/rdmatcher/processing.py), [feature_map.py](src/rdmatcher/feature_map.py) | Transformations and feature-name mapping. |
| [train.py](src/rdmatcher/train.py), [formula.py](src/rdmatcher/formula.py) | Score fitting, strategies, formula expansion and metadata. |
| [plot.py](src/rdmatcher/plot.py) | Balance/support plotting helpers. |

For specialized plotting, inspect the plotting function's current arguments and pass the
explicitly retained analysis sample. Public propensity plotting is available as
`plot_propensity_coverage`; matching itself does not automatically render a complete analysis
report or export outcome estimates.

## Supplying a precomputed matrix to RDM

Supply authoritative case-by-control costs without matching features:

```python
model = RDMatcher.from_distance_matrix(
    population, "patient_id", "treated", distances,
    case_ids=case_ids, control_ids=control_ids,
)
matched = model.match(threshold=float("inf"), ratio=4)
graph = model.return_graph(threshold=0.5, ratio=4)
```

`distance_matrix` accepts a NumPy array (including a read-only memory map), a
labeled pandas DataFrame, or a SciPy CSR matrix. Arrays and CSR matrices require
explicit `case_ids` and `control_ids`. DataFrame index/columns supply these IDs;
additional IDs must agree with those labels. IDs must be unique, nonmissing,
and match the population's treated and control sets exactly. Matrix axes may
have any order; alignment uses IDs. Exposure status must be binary (0/1).
Full-population square matrices are not converted.

Dense zero and explicitly stored CSR zero are valid edges. Dense positive
infinity, stored positive infinity, and absent CSR entries exclude edges.
Negative costs, NaN, negative infinity, incorrect shapes, duplicate IDs, and
duplicate CSR coordinates are rejected. Validation reads bounded value blocks,
using storage-sharing views for contiguous dense inputs.
Dense arrays and memory maps are borrowed without an unconditional full copy;
a homogeneous DataFrame may also share storage. **Do not modify supplied values
or CSR structure while the model is in use.** Heterogeneous DataFrames may need
an array conversion. Candidate selection still reads an unsorted dense row to
find its nearest controls, but precomputed retrieval starts with an exact
one-control prefix and widens only rows whose certificate needs more edges.
It retains only the loaded prefix per case, up to the configured cap. Supplying
costs avoids feature-distance computation and index fitting, but does not
eliminate row scanning, classification, assignment work, or the memory occupied
by the input.

The default `distance_metric=None` chooses precomputed costs when supplied and
Gower otherwise. Explicit `'precomputed'` requires a matrix. Other metrics
conflict with a supplied matrix. Matching-time distance transforms (PCA, Gower
weights, and nondefault feature weights) and constructor feature transforms
conflict with authoritative matrix costs. Use `method='multi'` (the default).
Population metadata remains available in matched outputs.

An additional propensity eligibility mask can use scores from an explicitly
fitted propensity model or an existing population column:

```python
options = MatchingOptions(ps_hybrid=True, propensity_col="my_logit", ps_caliper=0.2)
matched = model.match(float("inf"), matching=options)
```

The native caliper uses pooled sample SD; `ps_caliper_strict=False` retries empty
neighborhoods at twice the width. In precomputed mode, `propensity_caliper` also
requests this mask (and takes precedence over `ps_caliper` if both are set).
Missing/nonfinite scores raise an error; matching does not fit a propensity
model automatically. Calipers and the distance threshold restrict eligibility
before ranking; costs remain the supplied distances. Ties use the population's
control positions. Retrieval starts with exact minimum selection, widens only
when a case needs more candidates, resolves cutoff ties by control position,
and sorts only the retained prefix. `ExecutionOptions(n_jobs=...)` parallelizes
independent matrix rows with a bounded worker pool. Each call builds fresh
eligibility and retrieval state. Matrix validation and population-ID alignment
run during construction; reuse the initialized model to avoid repeating that
setup when trying different ratios or thresholds.

### Filter a supplied matrix into a sparse candidate graph

Use `return_graph()` when the goal is candidate filtering rather than final
matching. It applies the same ID alignment, matrix validation, eligibility
rules, threshold, tie handling, and candidate-preservation checks as
`match()`, then stops before allocation:

```python
model = RDMatcher.from_distance_matrix(
    population, "patient_id", "treated", distances,
    case_ids=case_ids,
    control_ids=control_ids,
)

graph = model.return_graph(
    threshold=float("inf"),
    ratio=1,
    execution=ExecutionOptions(n_jobs=1),
)

sparse_distances = graph.matrix
case_ids_out = graph.case_ids
control_ids_out = graph.control_ids
```

`graph.matrix` is a case-by-control SciPy CSR matrix with the original supplied
distances on retained edges. Its row and column mappings are `graph.case_ids`
and `graph.control_ids`. An absent CSR coordinate means the edge was not
retained or was ineligible; a stored zero remains a valid zero-distance edge.
Do not call `eliminate_zeros()` if zero-distance edges matter.

The exported matrix is a reduced candidate graph. It is the graph RDM uses for its
candidate-preserving Optimal workflow. If the full eligible edge set is needed
for another application, construct that CSR matrix outside RDM rather than
interpreting absent exported coordinates as all forbidden edges.

For a globally optimal assignment, use `allocation="optimal"` (the default).
The reduced graph must pass the complete-list, private-control, or case-count ×
ratio prefix certificate; uncertified caps raise an error. The guarantee is
preservation of an optimum on the eligible graph, including any threshold and
propensity exclusions. Different optimal pairs can exist when costs tie.
`allocation="hybrid"` retains greedy preallocation and does not carry this global
optimality guarantee.

Precomputed costs use exact binary integer scaling, without decimal quantization.
Complete Optimal assignments can use compiled SciPy with exactly represented
integer input weights; an independent exact check validates its result, including
any effects of floating-point solver arithmetic. Unverified internal calls retain
a conservative arithmetic-range guard. Incomplete assignments and costs outside
the supported input-weight range use an arbitrary-integer
min-cost-flow fallback, so tiny cost differences are not swallowed by large dummy
penalties. Precomputed MCF also uses exact binary scaling and falls back if its
integer range is insufficient; the fixed MCF scale applies to feature-computed
matching. Fallbacks are recorded in `assignment_diagnostics_`, including exact
objective numerator/denominator where applicable. The fallback has a 128 MiB
estimated graph-memory budget and raises instead of returning an approximation
when that budget is exceeded.

Every returned precomputed Optimal assignment also passes an independent exact
residual-network optimality check reconstructed from the certified candidate
graph. A failed check triggers exact recovery and rechecking; an unverified
assignment is not returned. Inspect `assignment_diagnostics_['optimality_verified']`
and `assignment_diagnostics_['optimality_certificate']`. The checker has its own
128 MiB estimated graph-memory budget. These certificates concern the represented
supplied costs and the selected eligibility and coverage/slot priorities.

Retrieval and reported edge distances preserve float64 input values. Inputs with
more than float64 precision, or integer values that cannot be represented exactly
in float64, are rejected instead of silently narrowed. For extremely large costs,
the displayed floating-point total can overflow; exact fallback diagnostics retain
the integer objective and binary denominator. Graph export keeps raw distances;
`graph.for_scipy_sparse_matching()` rejects precomputed weights if its documented
shift would round them. Feature-computed matching retains its existing numerical
behavior.

For a quick self-contained precomputed matching sanity check, run
`python validation/precomputed_smoke.py`. It reports phase timings and exact
optimality-certificate status; it is not a general performance benchmark.

## Authors and license

Noah Baker, MPH — Biomedical Informatics, UCSF — noah.baker@ucsf.edu.

RDMatcher is distributed under GPL-3.0; see [LICENSE](LICENSE). Project m