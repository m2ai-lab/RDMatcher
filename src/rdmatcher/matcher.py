import os
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import List, Literal, Optional, Any, Dict, Union
import logging
from collections import defaultdict

from .distance import GowerKNN
from ._general_retrieval import DemandNeighbors
from ._cosine_neighbors import CosineNeighbors
from .mahalanobis import MahalanobisKNN
from .matching import (
    recover_exact_precomputed_assignment,
    solve_optimal_assignment,
    solve_optimal_assignment_mcf,
    solve_optimal_assignment_scipy_sparse,
    extract_exposure_indices,
    compute_weighted_features,
    verbose_matching_results,
    hide_columns
)
from .candidate_graph import CandidateGraph, candidate_graph_from_prefilter
from .plot import plot_pca_threshold


class Matcher:
    def _resolve_gower_weights_dict(
        self,
        weights_spec: Dict[str, Union[float, int, Dict[str, Union[float, int]]]],
        processed_cols: List[str],
        processed_to_original: Optional[Dict[str, str]] = None,
        original_to_processed: Optional[Dict[str, List[str]]] = None,
        original_categorical_features: Optional[set] = None,
        default_weight: float = 1.0,
    ) -> List[float]:
        """Resolve user gower_weights dict onto the processed column order."""
        processed_to_original = processed_to_original or {c: c for c in processed_cols}
        original_to_processed = original_to_processed or {}
        original_categorical_features = original_categorical_features or set()

        valid_original = set(original_to_processed.keys()) if original_to_processed else set(processed_to_original.values())
        unknown_top = [k for k in weights_spec.keys() if k not in valid_original]
        if unknown_top:
            raise ValueError(f"Unknown gower_weights keys: {unknown_top}. Valid original features: {sorted(valid_original)}")

        weights_by_processed: Dict[str, float] = {c: float(default_weight) for c in processed_cols}

        for orig_key, val in weights_spec.items():
            if isinstance(val, dict):
                continue
            w = float(val)
            children = original_to_processed.get(orig_key, [])
            if not children:
                children = [c for c in processed_cols if processed_to_original.get(c, c) == orig_key]
            for c in children:
                weights_by_processed[c] = w

        for orig_key, val in weights_spec.items():
            if not isinstance(val, dict):
                continue
            if orig_key not in original_categorical_features:
                raise ValueError(f"gower_weights['{orig_key}'] is a dict, but '{orig_key}' is not a categorical feature")

            children = original_to_processed.get(orig_key, [])
            if not children:
                children = [c for c in processed_cols if processed_to_original.get(c, c) == orig_key]
            if not children:
                raise ValueError(f"No processed columns found for categorical feature '{orig_key}'")

            resolved_child_weights: Dict[str, float] = {}
            for child_key, child_w in val.items():
                if isinstance(child_w, dict):
                    raise ValueError(f"gower_weights['{orig_key}']['{child_key}'] is a dict; only two levels supported")
                w = float(child_w)

                if child_key in children:
                    resolved_child_weights[child_key] = w
                    continue

                candidate = f"{orig_key}_{child_key}"
                if candidate in children:
                    resolved_child_weights[candidate] = w
                    continue

                valid_suffixes = [c.split(f"{orig_key}_",1)[1] if c.startswith(f"{orig_key}_") else c for c in children]
                raise ValueError(
                    f"Unknown child key '{child_key}' for categorical '{orig_key}'.\n"
                    f"Valid processed child column names: {children}\n"
                    f"Valid child suffixes: {valid_suffixes}"
                )

            missing = [c for c in children if c not in resolved_child_weights]
            if missing:
                self.logger.warning(
                    f"gower_weights['{orig_key}'] did not include weights for {len(missing)} children. "
                    f"Missing will use default={default_weight}. Valid children: {children}"
                )

            for c, w in resolved_child_weights.items():
                weights_by_processed[c] = w

        try:
            mapped = {c: (processed_to_original.get(c, c), weights_by_processed[c]) for c in processed_cols}
            self.logger.info(f"Resolved gower_weights (processed_col -> (original, weight)): {mapped}")
        except Exception:
            pass

        return [weights_by_processed[c] for c in processed_cols]

    def __init__(self,
                 df: pd.DataFrame,
                 exposure_status: str,
                 patient_id: str,
                 distance_metric: Literal['gower', 'euclidean', 'cosine'] = "gower",
                 threshold: float = 0.2,
                 n_neighbors: int = 1,
                 gower_weights: Optional[Any] = None,
                 gower_cat_features: Optional[List[str]] = None,
                 weight_numeric: float = 1.0,
                 weight_propensity: float = 0.0,
                 propensity_col: Optional[str] = None,
                 ps_eligible_sets: Optional[List[np.ndarray]] = None,
                 pca_filter: bool = False,
                 logger=None,
                   **kwargs):

        self.df = df
        self.exposure_status = exposure_status
        self.patient_id = patient_id
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.n_neighbors = n_neighbors
        self.mahalanobis_neighbor_backend = kwargs.get('mahalanobis_neighbor_backend', 'sklearn')
        self.mahalanobis_algorithm = kwargs.get('mahalanobis_algorithm',kwargs.get('algorithm','auto'))

        requested_n_jobs = kwargs.get('n_jobs', 1)
        if requested_n_jobs is None:
            self.n_jobs = 1
        elif requested_n_jobs == -1:
            self.n_jobs = os.cpu_count() or 1
        elif isinstance(requested_n_jobs, int) and requested_n_jobs >= 1:
            self.n_jobs = int(requested_n_jobs)
        else:
            raise ValueError("n_jobs must be None, -1, or an integer >= 1")

        self.parallel_chunk_size = kwargs.get('parallel_chunk_size', None)

        self.streaming = kwargs.get('streaming', 'auto')
        self.stream_block_size = kwargs.get('stream_block_size', None)
        self.stream_threshold_gb = float(kwargs.get('stream_threshold_gb', kwargs.get('stream_threshold_gb', 1.0)))
        self.memory_limit_gb = kwargs.get('memory_limit_gb', None)
        if self.memory_limit_gb is not None:
            try:
                self.memory_limit_gb = float(self.memory_limit_gb)
            except Exception:
                raise ValueError("memory_limit_gb must be a number (GB)")

        if self.streaming not in ('auto', 'on', 'off'):
            raise ValueError("streaming must be one of 'auto', 'on', or 'off'")

        if self.stream_block_size is not None:
            try:
                sbs_val = int(self.stream_block_size)
            except Exception:
                raise ValueError("stream_block_size must be an integer >= 1")
            if sbs_val < 1:
                raise ValueError("stream_block_size must be an integer >= 1")
            self.stream_block_size = sbs_val

        if self.parallel_chunk_size is not None:
            try:
                pcs_val = int(self.parallel_chunk_size)
            except Exception:
                raise ValueError("parallel_chunk_size must be an integer >= 1")
            if pcs_val < 1:
                raise ValueError("parallel_chunk_size must be an integer >= 1")
            self.parallel_chunk_size = pcs_val

        base = logger if logger is not None else logging.getLogger('rdmatcher')
        self.logger = base.getChild('matcher')
        self.logger.setLevel(logging.NOTSET)

        self.ps_eligible_sets = ps_eligible_sets
        if self.ps_eligible_sets is not None:
            self.logger.info(f"PS-Hybrid mode: {len(self.ps_eligible_sets)} eligible sets provided.")

        self.df_match = hide_columns(self.df.copy(), [self.patient_id])

        self.exposed_indices, self.control_indices = extract_exposure_indices(self.df_match, self.exposure_status)

        if self.distance_metric == "precomputed":
            self.precomputed_distances = kwargs['precomputed_distances']
            self.distance_dtype = self.precomputed_distances.dtype
            self.X_exposed = np.arange(len(self.exposed_indices))[:, None]
            self.X_control = np.arange(len(self.control_indices))[:, None]
            self.nbrs_model = self.precomputed_distances
            return

        if self.distance_metric == "gower":
            self.logger.info("Gower Metric: Keeping data as DataFrame to preserve Categorical types.")
            features_only = self.df_match.drop(columns=[self.exposure_status])
            self.X_exposed = features_only.iloc[self.exposed_indices]
            self.X_control = features_only.iloc[self.control_indices]

            weights_vector = None
            if gower_weights is None:
                weights_vector = None
            elif isinstance(gower_weights, (list, tuple, np.ndarray)):
                if len(gower_weights) != features_only.shape[1]:
                    raise ValueError(
                        f"gower_weights list length {len(gower_weights)} does not match number of matching features "
                        f"{features_only.shape[1]}. Feature order: {list(features_only.columns)}"
                    )
                weights_vector = list(map(float, gower_weights))
                self.logger.warning(
                    "gower_weights provided as a list is deprecated and order-dependent. "
                    "Prefer a dict keyed by original feature names."
                )
            elif isinstance(gower_weights, dict):
                weights_vector = self._resolve_gower_weights_dict(
                    gower_weights,
                    processed_cols=list(features_only.columns),
                    processed_to_original=kwargs.get('feature_name_map_processed_to_original'),
                    original_to_processed=kwargs.get('feature_name_map_original_to_processed'),
                    original_categorical_features=set(kwargs.get('original_categorical_features', []) or []),
                )
            else:
                raise ValueError("gower_weights must be a dict[str, float|dict] or a list/tuple/ndarray")

            if weights_vector is not None:
                for col, w in zip(features_only.columns, weights_vector):
                    if 'propensity' in col.lower() and w != 0.0:
                        self.logger.warning(
                            f"Feature '{col}' has Gower weight {w}. "
                            "Propensity scores used in distance matching may distort covariate balance."
                        )

            self.logger.info("Fitting GowerKNN model...")
            self.nbrs_model = GowerKNN(
                weights=weights_vector,
                cat_features=gower_cat_features,
                n_jobs=self.n_jobs,
                parallel_chunk_size=self.parallel_chunk_size,
                streaming=self.streaming,
                stream_block_size=self.stream_block_size,
                stream_threshold_gb=self.stream_threshold_gb,
                memory_limit_gb=self.memory_limit_gb,
                logger=self.logger
            )
            self.nbrs_model.fit(self.X_control)
            cached_components = kwargs.get('gower_feature_components', None)
            if cached_components is not None:
                self.nbrs_model.set_feature_component_cache(cached_components)

        else:
            self.logger.info(f"{self.distance_metric} Metric: Converting to numeric Numpy array.")
            _, self.X_combined, cov_cols, feature_names = compute_weighted_features(
                df=self.df,
                exposure_status=self.exposure_status,
                weight_numeric=weight_numeric,
                patient_id=self.patient_id,
                propensity_col=propensity_col,
                weight_propensity=weight_propensity
            )
            self.X_exposed = self.X_combined[self.exposed_indices]
            self.X_control = self.X_combined[self.control_indices]

            if pca_filter:
                self._apply_pca()

            self.logger.info("Checking feature weights before fitting the model...")
            self.logger.info(f"Features used for matching ({len(feature_names)}): {feature_names}")
            eff_weights = []
            if propensity_col and weight_propensity > 0:
                eff_weights.append((propensity_col, float(weight_propensity)))
            eff_weights.extend([(c, float(weight_numeric)) for c in cov_cols])
            nonzero = [f for f, w in eff_weights if w != 0.0]
            self.logger.info(f"Non-zero weighted features: {nonzero}")
            self.logger.info(f"Effective weights: {eff_weights}")


            if self.distance_metric == 'mahalanobis':
                self.logger.info("Fitting MahalanobisKNN model...")
                self.nbrs_model = MahalanobisKNN(
                    logger=self.logger,
                    neighbor_backend=self.mahalanobis_neighbor_backend,
                    algorithm=self.mahalanobis_algorithm,
                    n_jobs=self.n_jobs,
                )
                self.nbrs_model.fit(self.X_control, cov_source='pooled', pooled_X=self.X_combined)
            elif self.distance_metric=='cosine' and kwargs.get('algorithm','auto') in ('auto','brute'):
                self.nbrs_model=CosineNeighbors(n_jobs=self.n_jobs).fit(self.X_control)
            else:
                self.logger.info(f"Fitting NearestNeighbors ({self.distance_metric}) model...")
                self.nbrs_model = NearestNeighbors(
                    metric=self.distance_metric,
                    n_jobs=self.n_jobs,
                    algorithm=kwargs.get('algorithm', 'auto')
                )
                self.nbrs_model.fit(self.X_control)

    def _apply_pca(self):
        self.logger.info("Applying PCA filter...")
        pca_res = plot_pca_threshold(self.X_exposed, plot=False, return_pca=True, variance_threshold=0.95)
        if pca_res:
            pca_model, cutoff = pca_res
            self.X_exposed = pca_model.transform(self.X_exposed)[:, :cutoff]
            self.X_control = pca_model.transform(self.X_control)[:, :cutoff]
            self.logger.info(f"PCA reduced features to {cutoff} dimensions.")

    def _prefilter_candidates(self, k_candidates, safe_matches, fuzzy_threshold, fuzzy_limit, batch_size):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        self.logger.info(f"Pre-filtering candidates (k={k_candidates})...")

        k_candidates = min(k_candidates, self.X_control.shape[0])
        if k_candidates <= 0:
            raise ValueError("No control subjects available for matching.")

        n_exposed = self.X_exposed.shape[0]

        if self.ps_eligible_sets is not None and self.distance_metric not in {"gower", "mahalanobis", "precomputed"}:
            raise ValueError("PS-Hybrid mode (ps_eligible_sets) is only supported with Gower or Mahalanobis distance metric.")

        backend = getattr(self.nbrs_model, '_nn_model', None)
        if backend is None:
            backend = self.nbrs_model
        tree = getattr(backend, '_tree', None)
        tree_before = tree.get_n_calls() if tree is not None else None
        lazy = DemandNeighbors(self, k_candidates, batch_size)
        distances, indices = lazy.distances, lazy.indices
        self.candidate_retrieval_stats_ = lazy.stats

        horizons = np.ones(n_exposed, dtype=np.int32)
        iteration_count = 0

        n_controls = len(self.control_indices)
        first_controls = indices[:, 0]
        first_valid = (first_controls >= 0) & (distances[:, 0] <= self.threshold)
        usage_counts = np.bincount(
            first_controls[first_valid], minlength=n_controls
        ).astype(np.int32, copy=False)
        owners = np.full(n_controls, -1, dtype=np.int32)
        first_rows = np.flatnonzero(first_valid).astype(np.int32, copy=False)
        if first_rows.size:
            singleton_mask = usage_counts[first_controls[first_valid]] == 1
            owners[first_controls[first_valid][singleton_mask]] = first_rows[singleton_mask]

        safe_counts = np.zeros(n_exposed, dtype=np.int32)
        if first_rows.size:
            safe_rows = first_rows[usage_counts[first_controls[first_valid]] == 1]
            safe_counts[safe_rows] += 1

        while True:
            expandable = (safe_counts < safe_matches) & (horizons < k_candidates)
            if not np.any(expandable):
                break

            rows = np.flatnonzero(expandable).astype(np.int32, copy=False)
            if lazy is not None:
                lazy.ensure(rows, horizons[rows] + 1)
            activated_controls = indices[rows, horizons[rows]]
            activated_distances = distances[rows, horizons[rows]]
            valid = (activated_controls >= 0) & (activated_distances <= self.threshold)
            if np.any(valid):
                edge_rows = rows[valid]
                edge_controls = activated_controls[valid]
                unique_controls, inverse, arrivals = np.unique(
                    edge_controls, return_inverse=True, return_counts=True
                )
                old_counts = usage_counts[unique_controls].copy()

                becoming_competitive = unique_controls[old_counts == 1]
                if becoming_competitive.size:
                    prior_owners = owners[becoming_competitive]
                    prior_owners = prior_owners[prior_owners >= 0]
                    np.add.at(safe_counts, prior_owners, -1)

                becoming_safe = (old_counts == 0) & (arrivals == 1)
                if np.any(becoming_safe):
                    first_arrival = np.full(unique_controls.size, -1, dtype=np.int32)
                    _, first_indices = np.unique(inverse, return_index=True)
                    first_arrival[inverse[first_indices]] = edge_rows[first_indices]
                    new_controls = unique_controls[becoming_safe]
                    new_owners = first_arrival[becoming_safe]
                    owners[new_controls] = new_owners
                    safe_counts[new_owners] += 1

                usage_counts[unique_controls] += arrivals.astype(np.int32, copy=False)
                owners[unique_controls[usage_counts[unique_controls] != 1]] = -1

            horizons[rows] += 1
            iteration_count += 1

        self.candidate_horizons_ = horizons.copy()
        self.candidate_horizon_iterations_ = int(iteration_count + 1)
        self.candidate_horizon_locked_ = horizons >= k_candidates

        candidate_list = []

        for i in range(n_exposed):
            dist_row = distances[i]
            idx_row  = indices[i]
            horizon = int(horizons[i])

            safe_positions, safe_distances = [], []
            competitive_positions, competitive_distances = [], []
            fuzzy_positions, fuzzy_distances = [], []

            for j, ctrl_idx in enumerate(idx_row[:horizon]):
                d = float(dist_row[j])

                if int(ctrl_idx) >= 0 and np.isfinite(d) and d <= self.threshold:
                    if usage_counts[int(ctrl_idx)] == 1:
                        safe_positions.append(int(ctrl_idx))
                        safe_distances.append(d)
                        if len(safe_positions) >= safe_matches:
                            break
                    else:
                        competitive_positions.append(int(ctrl_idx))
                        competitive_distances.append(d)

                elif fuzzy_threshold and (fuzzy_limit is not None) and d <= fuzzy_limit:
                    fuzzy_positions.append(int(ctrl_idx))
                    fuzzy_distances.append(d)
                else:
                    break

            candidate_list.append({
                'safe_positions': np.asarray(safe_positions, dtype=np.int32),
                'safe_distances': np.asarray(safe_distances, dtype=getattr(self, "distance_dtype", np.float32)),
                'competitive_positions': np.asarray(competitive_positions, dtype=np.int32),
                'competitive_distances': np.asarray(competitive_distances, dtype=getattr(self, "distance_dtype", np.float32)),
                'fuzzy_positions': np.asarray(fuzzy_positions, dtype=np.int32),
                'fuzzy_distances': np.asarray(fuzzy_distances, dtype=getattr(self, "distance_dtype", np.float32)),
                'neighbor_indices': idx_row,
                'neighbor_distances': dist_row,
                'candidate_horizon': horizon,
            })

        if lazy is not None and self.ps_eligible_sets is not None:
            self.ps_hybrid_diagnostics_=[]
            for i,eligible in enumerate(self.ps_eligible_sets):
                valid=indices[i]>=0
                values=distances[i][valid]
                finite=values[np.isfinite(values)]
                within=int(np.count_nonzero(values<=self.threshold))
                self.ps_hybrid_diagnostics_.append(dict(treated_position=i,
                    ps_eligible=len(eligible),retained=len(values),within_threshold=within,
                    retained_within_threshold=within,
                    min_distance=float(finite.min()) if len(finite) else np.nan,
                    max_retained_distance=float(finite.max()) if len(finite) else np.nan,
                    truncated_by_k=len(eligible)>len(values),
                    within_threshold_truncated_by_k=bool(len(eligible)>len(values) and within>0),
                    retrieval_scope='loaded_prefix'))
        warning_limit_count = sum(1 for c in candidate_list if len(c['safe_positions']) < safe_matches)
        if warning_limit_count > 0:
            self.logger.debug(
                f"{warning_limit_count} exposed subjects have fewer than {safe_matches} safe matches "
                f"within the top {k_candidates} candidates."
            )

        self.candidate_retrieval_stats_['neighbor_backend'] = type(self.nbrs_model).__name__
        self.candidate_retrieval_stats_['neighbor_algorithm'] = getattr(backend, '_fit_method', None)
        if hasattr(self.nbrs_model,'normalization_stats_'):
            self.candidate_retrieval_stats_['normalization']=dict(self.nbrs_model.normalization_stats_)
        if tree is not None:
            self.candidate_retrieval_stats_['tree_distance_calls'] = int(tree.get_n_calls()-tree_before)
        return candidate_list

    def _require_exact_kernel(self, candidates, global_optimal):
        """Validate Safe, complete-list or case-count-times-ratio certificates."""
        bound=len(candidates)*self.n_neighbors
        complete=self.assignment_diagnostics_['eligible_set_retained']
        kinds=[]
        for i,c in enumerate(candidates):
            if complete[i]:kind='complete_eligible'
            elif len(c['safe_positions'])>=self.n_neighbors:kind='singleton_safe'
            elif len(c['safe_positions'])+len(c['competitive_positions'])>=bound:kind='Cr_prefix'
            else:kind='uncertified'
            kinds.append(kind)
        certified=[k!='uncertified' for k in kinds]
        self.assignment_diagnostics_.update(preservation_certificate_types=kinds,
            preservation_condition_by_case=certified,
            preservation_condition_satisfied=all(certified),
            cases_failing_preservation_condition=sum(not x for x in certified))
        if global_optimal and not all(certified):
            raise ValueError('The requested candidate cap does not certify an exact Optimal kernel. Use the default cap or a larger certified prefix.')

    def match(self, k_candidates=None, global_optimal=True, competitive_match=False,
              replacement=False, safe_matches=None, fuzzy_threshold=False, fuzzy_threshold_limit=None,
              competitive_safe_mode="virtual",
              batch_size=10, enable_incremental_counts=True,
              log_matching_summary=False,
              solver=None, return_candidate_graph=False,
              prioritize_coverage=False,
              **kwargs) -> Union[pd.DataFrame, CandidateGraph]:
        """
        Perform matching of exposed subjects to control subjects.
        Parameters
        ----------
        k_candidates : int, optional
            Number of candidate controls to consider per exposed subject. If
            omitted, defaults to ``n_exposed * n_neighbors`` for this match,
            where ``n_exposed`` is the number of treated/exposed rows. This
            makes the candidate-pool default depend on the requested matching
            ratio and the size of the treated sample.
        global_optimal : bool, default=True
            If True, perform global optimal matching phase.
        competitive_match : bool, default=False
            If True, perform competitive allocation phase before global optimal.
        prioritize_coverage : bool, default=False
            If True, globally maximize the number of previously unmatched cases
            receiving at least one eligible control before filling extra slots
            and minimizing distance. False uses the
            unmatched-slot objective.
            This can change retained treated membership and, for ratios above
            one, trade additional slots for already-covered cases against first
            matches for other cases. Unless all treated cases are covered, an
            effect estimate targets the covered treated subset.
        replacement : bool, default=False
            Only False is supported; controls are assigned without replacement.
        safe_matches : int, optional
            Number of "safe" matches to require per exposed subject. If None, defaults to n_neighbors.
        competitive_safe_mode : {"virtual", "assign"}, default="virtual"
            In the competitive phase, ``"virtual"`` uses safe controls as
            unconstrained capacity for prioritization but leaves them available
            to the global phase. ``"assign"`` performs immediate safe
            control assignment.
        fuzzy_threshold : bool, default=False
            If True, allow fuzzy matching beyond the main threshold.
        fuzzy_threshold_limit : float, optional
            Maximum distance for fuzzy matches. Required if fuzzy_threshold is True.
        solver : {None, "hungarian", "mcf", "scipy_sparse"}, optional
            Global assignment backend. When omitted, SciPy's sparse
            minimum-weight full bipartite matching (LAPJVsp) is used.
        return_candidate_graph : bool, default=False
            Return the prefiltered case-control distance graph without running
            competitive or global allocation.
        Returns
        -------
        pd.DataFrame or CandidateGraph
            Matched exposed/control rows, or the sparse candidate graph when
            ``return_candidate_graph=True``.
        """
        self.log_matching_summary = bool(log_matching_summary)

        if replacement:
             raise NotImplementedError("Matching with replacement is not implemented yet.")

        if k_candidates is None:
            k_candidates = int(len(self.exposed_indices) * self.n_neighbors)
        else:
            k_candidates = int(k_candidates)
        if k_candidates < 1:
            raise ValueError("k_candidates must be >= 1")
        if competitive_safe_mode not in {"virtual", "assign"}:
            raise ValueError("competitive_safe_mode must be either 'virtual' or 'assign'")
        valid_solvers = {"hungarian", "mcf", "scipy_sparse"}
        if solver is not None and solver not in valid_solvers:
            raise ValueError(f"solver must be one of {sorted(valid_solvers)}")
        selected_solver = solver if solver is not None else "scipy_sparse"

        if safe_matches is None:
            safe_matches = self.n_neighbors

        if fuzzy_threshold and fuzzy_threshold_limit is None:
            fuzzy_threshold_limit = self.threshold

        matching_started = perf_counter()
        candidate_list = self._prefilter_candidates(k_candidates, safe_matches, fuzzy_threshold, fuzzy_threshold_limit, batch_size)

        candidate_finished = perf_counter()
        complete_eligible_rows = []
        for i, candidates in enumerate(candidate_list):
            retrieved_positions = candidates["neighbor_indices"]
            retrieved_distances = candidates["neighbor_distances"]
            eligible_size = (len(self.ps_eligible_sets[i]) if self.ps_eligible_sets is not None
                             else len(self.control_indices))
            all_admissible_retrieved = (eligible_size <= len(retrieved_positions)
                                       or retrieved_positions[-1] < 0
                                       or retrieved_distances[-1] > self.threshold)
            admissible_retrieved = int(np.count_nonzero(
                (retrieved_positions >= 0) & np.isfinite(retrieved_distances)
                & (retrieved_distances <= self.threshold)))
            retained = len(candidates["safe_positions"]) + len(candidates["competitive_positions"])
            complete_eligible_rows.append(bool(all_admissible_retrieved and retained == admissible_retrieved))
        preservation_condition = [complete_eligible_rows[i] or len(c["safe_positions"]) >= self.n_neighbors
                                  for i, c in enumerate(candidate_list)]
        self.assignment_diagnostics_ = {
            "candidate_preparation_sec": candidate_finished - matching_started,
            "candidate_edge_count": int(sum(len(c["safe_positions"]) + len(c["competitive_positions"]) for c in candidate_list)),
            "case_count": int(len(self.exposed_indices)),
            "control_count": int(len(self.control_indices)),
            "retrieval_cap": int(min(k_candidates, len(self.control_indices))),
            "candidate_retrieval": dict(self.candidate_retrieval_stats_),
            "candidate_horizons": self.candidate_horizons_.tolist(),
            "safe_control_counts": [int(len(c["safe_positions"])) for c in candidate_list],
            "cases_below_requested_safe_count": int(sum(len(c["safe_positions"]) < self.n_neighbors for c in candidate_list)),
            "eligible_set_retained": complete_eligible_rows,
            "preservation_condition_by_case": preservation_condition,
            "preservation_condition_satisfied": bool(all(preservation_condition)),
            "cases_failing_preservation_condition": int(sum(not value for value in preservation_condition)),
            "competitive_enabled": bool(competitive_match),
            "solver": selected_solver,
            "unmatched_slot_penalty": 1_000_000.0,
            "mcf_cost_scale": float(kwargs.get("mcf_cost_scale", 1_000_000.0)) if selected_solver == "mcf" else None,
        }
        import hashlib
        digest = hashlib.sha256()
        digest.update(np.asarray([len(candidate_list),len(self.control_indices)],dtype=np.int64).tobytes())
        for candidates in candidate_list:
            for key in ['safe_positions','safe_distances','competitive_positions','competitive_distances']:
                values = np.asarray(candidates[key])
                digest.update(np.asarray([len(values)],dtype=np.int64).tobytes())
                digest.update(memoryview(values).cast('B'))
        self.assignment_diagnostics_['initial_candidate_graph_sha256'] = digest.hexdigest()
        self._require_exact_kernel(candidate_list, global_optimal)

        if return_candidate_graph:
            graph = candidate_graph_from_prefilter(
                candidate_list=candidate_list,
                n_controls=len(self.control_indices),
                case_ids=self.df.iloc[self.exposed_indices][self.patient_id].to_numpy(),
                control_ids=self.df.iloc[self.control_indices][self.patient_id].to_numpy(),
                distance_metric=self.distance_metric,
                threshold=self.threshold,
            )
            self.candidate_graph_ = graph
            return graph

        match_dict = {}
        used_controls = set()

        if competitive_match:
            self.logger.info("Running Competitive Allocation Phase...")
            match_dict, used_controls = self._run_competitive_allocation(
                candidate_list, match_dict, used_controls,
                safe_matches, fuzzy_threshold, fuzzy_threshold_limit,
                enable_incremental_counts=enable_incremental_counts,
                safe_mode=competitive_safe_mode,
            )

        competitive_finished = perf_counter()
        greedy_count = sum(len(values) for values in match_dict.values())
        self.assignment_diagnostics_["competitive_allocation_sec"] = competitive_finished - candidate_finished
        self.assignment_diagnostics_["preallocated_control_count"] = int(greedy_count)
        self.assignment_diagnostics_["residual_requested_slots"] = int(len(self.exposed_indices) * self.n_neighbors - greedy_count)

        if global_optimal:
            exposed_to_solve_indices = []
            residual_demands = []
            coverage_priority = []
            cleaned_candidate_list = []
            precomputed_subset = []

            for i in range(len(self.exposed_indices)):
                exposed_orig_id = self.exposed_indices[i]
                current_matches = match_dict.get(exposed_orig_id, [])
                if len(current_matches) < self.n_neighbors:
                    exposed_to_solve_indices.append(i)
                    residual_demands.append(self.n_neighbors - len(current_matches))
                    coverage_priority.append(bool(prioritize_coverage and len(current_matches) == 0))
                    cands = candidate_list[i]
                    positions = np.concatenate((
                        cands['safe_positions'], cands['competitive_positions']
                    ))
                    candidate_distances = np.concatenate((
                        cands['safe_distances'], cands['competitive_distances']
                    ))
                    control_ids = self.control_indices[positions]
                    available = ~np.isin(control_ids, list(used_controls)) if used_controls else np.ones(len(positions), dtype=bool)
                    cleaned_candidate_list.append(positions[available])

                    precomputed_subset.append({
                        'candidate_positions': positions[available],
                        'candidate_distances': candidate_distances[available],
                    })

            if exposed_to_solve_indices:
                self.logger.info(f"Running Global Optimal Phase for {len(exposed_to_solve_indices)} subjects...")

                if isinstance(self.X_exposed, pd.DataFrame):
                    subset_X_exposed = self.X_exposed.iloc[exposed_to_solve_indices]
                else:
                    subset_X_exposed = self.X_exposed[exposed_to_solve_indices]

                if selected_solver == "mcf":
                    new_matches = solve_optimal_assignment_mcf(
                        X_exposed_subset=subset_X_exposed,
                        X_control=self.X_control,
                        candidate_lists=cleaned_candidate_list,
                        threshold=self.threshold,
                        metric=self.distance_metric,
                        n_neighbors=self.n_neighbors,
                        demands=residual_demands,
                        coverage_priority=coverage_priority,
                        cost_scale=kwargs.get("mcf_cost_scale", 1_000_000.0),
                        all_control_indices=self.control_indices,
                        gower_model=self.nbrs_model if self.distance_metric == "gower" else None,
                        precomputed=precomputed_subset,
                        solver_diagnostics=self.assignment_diagnostics_,
                    )
                elif selected_solver == "scipy_sparse":
                    new_matches = solve_optimal_assignment_scipy_sparse(
                        X_exposed_subset=subset_X_exposed,
                        X_control=self.X_control,
                        candidate_lists=cleaned_candidate_list,
                        threshold=self.threshold,
                        metric=self.distance_metric,
                        n_neighbors=self.n_neighbors,
                        demands=residual_demands,
                        coverage_priority=coverage_priority,
                        all_control_indices=self.control_indices,
                        gower_model=self.nbrs_model if self.distance_metric == "gower" else None,
                        precomputed=precomputed_subset,
                        solver_diagnostics=self.assignment_diagnostics_,
                        _certified_proposal=self.distance_metric == 'precomputed' and not competitive_match,
                    )
                else:
                    new_matches = solve_optimal_assignment(
                        X_exposed_subset=subset_X_exposed,
                        X_control=self.X_control,
                        candidate_lists=cleaned_candidate_list,
                        threshold=self.threshold,
                        metric=self.distance_metric,
                        n_neighbors=self.n_neighbors,
                        demands=residual_demands,
                        coverage_priority=coverage_priority,
                        all_control_indices=self.control_indices,
                        gower_model=self.nbrs_model if self.distance_metric == "gower" else None,
                        precomputed=precomputed_subset,
                        solver_diagnostics=self.assignment_diagnostics_,
                        _certified_proposal=self.distance_metric == 'precomputed' and not competitive_match,
                    )

                del cleaned_candidate_list

                if self.distance_metric == 'precomputed' and not competitive_match:
                    from ._optimality_certificate import verify_assignment
                    certificate_started = perf_counter()
                    # Reconstruct from the certified reduction, independently of
                    # the solver's working arrays and its cost transformation.
                    certified_edges = []
                    for row in exposed_to_solve_indices:
                        candidates = candidate_list[row]
                        certified_edges.append(dict(
                            candidate_positions=np.concatenate((candidates['safe_positions'], candidates['competitive_positions'])),
                            candidate_distances=np.concatenate((candidates['safe_distances'], candidates['competitive_distances']))))
                    certificate = verify_assignment(certified_edges, residual_demands,
                        coverage_priority, self.control_indices, new_matches, self.threshold)
                    if not certificate['verified']:
                        new_matches = recover_exact_precomputed_assignment(certified_edges,
                            residual_demands, coverage_priority, self.control_indices,
                            self.assignment_diagnostics_)
                        certificate = verify_assignment(certified_edges, residual_demands,
                            coverage_priority, self.control_indices, new_matches, self.threshold)
                    if not certificate['verified']:
                        raise RuntimeError('Exact optimality verification failed; no assignment was returned')
                    self.assignment_diagnostics_['optimality_verified'] = True
                    self.assignment_diagnostics_['optimality_certificate'] = certificate
                    self.assignment_diagnostics_['optimality_verification_sec'] = perf_counter() - certificate_started

                for i_subset, matched_ctrl_ids in new_matches.items():
                    orig_idx = exposed_to_solve_indices[i_subset]
                    exposed_id = self.exposed_indices[orig_idx]
                    current = match_dict.get(exposed_id, [])
                    needed = self.n_neighbors - len(current)
                    if needed > 0:
                        if len(matched_ctrl_ids) > needed:
                            raise RuntimeError("Assignment backend exceeded the case residual demand")
                        match_dict[exposed_id] = current + matched_ctrl_ids
                        used_controls.update(matched_ctrl_ids)

        self.assignment_diagnostics_["global_assignment_sec"] = perf_counter() - competitive_finished
        self.match_distances_ = {}
        for exp_id, control_ids in match_dict.items():
            rel_idx = int(np.searchsorted(self.exposed_indices, exp_id))
            cands = candidate_list[rel_idx]
            for control_id in control_ids:
                ctrl_pos = int(np.searchsorted(self.control_indices, control_id))
                hit = np.flatnonzero(cands['neighbor_indices'] == ctrl_pos)
                if hit.size:
                    self.match_distances_[(int(exp_id), int(control_id))] = float(
                        cands['neighbor_distances'][hit[0]]
                    )

        self.matched_data = self._build_dataframe(match_dict, candidate_list)

        selected_count = sum(len(values) for values in match_dict.values())
        distances = list(self.match_distances_.values())
        self.assignment_diagnostics_.update({
            "matched_control_count": int(selected_count),
            "unfilled_slots": int(len(self.exposed_indices) * self.n_neighbors - selected_count),
            "selected_distance_count": int(len(distances)),
            "total_matching_distance": float(sum(distances)) if len(distances) == selected_count else None,
            "preallocated_fraction": float(greedy_count / selected_count) if selected_count else 0.0,
            "matching_total_sec": perf_counter() - matching_started,
        })
        return self.matched_data

    def _run_competitive_allocation(self, candidate_list, match_dict, used_controls,
                                    safe_matches, fuzzy_threshold, fuzzy_threshold_limit,
                                    enable_incremental_counts=True, safe_mode="virtual",
                                    defer_horizon_locked=False, candidate_cap=None):
        limited_indices = []
        for i, cand in enumerate(candidate_list):
            horizon_locked = (
                defer_horizon_locked
                and candidate_cap is not None
                and int(cand.get('candidate_horizon', 0)) >= int(candidate_cap)
            )
            if len(cand['safe_positions']) < safe_matches and not horizon_locked:
                limited_indices.append(i)

        if not limited_indices:
            self.logger.debug("No limited subjects identified for competitive allocation.")
            return match_dict, used_controls

        safe_capacity = {
            i: min(len(candidate_list[i]['safe_positions']), self.n_neighbors)
            for i in limited_indices
        }
        if safe_mode == "virtual":
            matches_needed = {
                self.exposed_indices[i]: max(0, self.n_neighbors - safe_capacity[i])
                for i in limited_indices
            }
        else:
            matches_needed = {self.exposed_indices[i]: self.n_neighbors for i in limited_indices}

        if safe_mode == "assign":
            self.logger.info("Phase 1: Assigning safe controls for limited group.")
            for i in limited_indices:
                exposed_id = self.exposed_indices[i]
                assigned_count = 0

                for ctrl_idx in candidate_list[i]['safe_positions']:
                    ctrl_id = int(self.control_indices[ctrl_idx])
                    if ctrl_id not in used_controls:
                        match_dict.setdefault(exposed_id, []).append(ctrl_id)
                        used_controls.add(ctrl_id)
                        matches_needed[exposed_id] -= 1
                        assigned_count += 1
                        self.logger.debug(f"Assigned safe match for {exposed_id}: {ctrl_id} (needs: {matches_needed[exposed_id]})")
                    if assigned_count >= self.n_neighbors:
                        break
        else:
            self.logger.info("Phase 1: Retaining safe controls as virtual capacity for limited group.")

        self.logger.info("Phase 2: Starting iterative greedy for competitive/fuzzy controls.")
        active_indices = [i for i in limited_indices if matches_needed[self.exposed_indices[i]] > 0]

        if active_indices:
            if enable_incremental_counts:
                competitive_sorted_map = {}
                fuzzy_sorted_map = {}
                reverse_index = defaultdict(list)

                top_k = None

                for i in active_indices:
                    cands = candidate_list[i]
                    comp_pairs = [
                        (int(self.control_indices[pos]), float(d))
                        for pos, d in zip(cands['competitive_positions'], cands['competitive_distances'])
                    ]

                    comp_pairs.sort(key=lambda x: (x[1], x[0]))
                    if top_k is not None:
                        comp_pairs = comp_pairs[:top_k]
                    competitive_sorted_map[i] = comp_pairs
                    for ctrl_id, _ in comp_pairs:
                        reverse_index[ctrl_id].append((i, 'competitive'))

                    fuzzy_pairs = [
                        (int(self.control_indices[pos]), float(d))
                        for pos, d in zip(cands['fuzzy_positions'], cands['fuzzy_distances'])
                    ]

                    fuzzy_pairs.sort(key=lambda x: (x[1], x[0]))
                    if top_k is not None:
                        fuzzy_pairs = fuzzy_pairs[:top_k]
                    fuzzy_sorted_map[i] = fuzzy_pairs
                    for ctrl_id, _ in fuzzy_pairs:
                        reverse_index[ctrl_id].append((i, 'fuzzy'))

                comp_avail_count = {i: sum(1 for cid, _ in competitive_sorted_map.get(i, []) if cid not in used_controls)
                                    for i in active_indices}
                fuzzy_avail_count = {i: sum(1 for cid, _ in fuzzy_sorted_map.get(i, []) if cid not in used_controls)
                                     for i in active_indices}

                max_iterations = len(active_indices) * self.n_neighbors * 2
                iteration_count = 0

                while active_indices and iteration_count < max_iterations:
                    iteration_count += 1

                    def sort_key(idx):
                        exp_id = self.exposed_indices[idx]
                        effective_secured = len(match_dict.get(exp_id, [])) + safe_capacity[idx]
                        return (effective_secured,
                                comp_avail_count.get(idx, 0),
                                fuzzy_avail_count.get(idx, 0),
                                idx)

                    active_indices.sort(key=sort_key)
                    made_progress = False

                    for i in active_indices:
                        exposed_id = self.exposed_indices[i]
                        if matches_needed[exposed_id] <= 0:
                            continue

                        best_ctrl_id = None
                        best_type = None
                        best_distance = None

                        for ctrl_id, d in competitive_sorted_map.get(i, []):
                            if ctrl_id in used_controls:
                                continue
                            if d is not None and d <= self.threshold:
                                best_ctrl_id = ctrl_id
                                best_type = 0
                                best_distance = d
                                break

                        if best_ctrl_id is None and fuzzy_threshold and (fuzzy_threshold_limit is not None):
                            for ctrl_id, d in fuzzy_sorted_map.get(i, []):
                                if ctrl_id in used_controls:
                                    continue
                                if d is not None and d <= fuzzy_threshold_limit:
                                    best_ctrl_id = ctrl_id
                                    best_type = 1
                                    best_distance = d + self.threshold
                                    break

                        if best_ctrl_id is not None:
                            match_dict.setdefault(exposed_id, []).append(best_ctrl_id)
                            used_controls.add(best_ctrl_id)
                            matches_needed[exposed_id] -= 1
                            made_progress = True
                            for (exp_j, list_type) in reverse_index.get(best_ctrl_id, []):
                                if list_type == 'competitive':
                                    if comp_avail_count.get(exp_j, 0) > 0:
                                        comp_avail_count[exp_j] -= 1
                                else:
                                    if fuzzy_avail_count.get(exp_j, 0) > 0:
                                        fuzzy_avail_count[exp_j] -= 1

                            self.logger.debug(
                                f"Greedy matched {exposed_id} with {best_ctrl_id} "
                                f"(type: {best_type}, score: {best_distance:.3f}, "
                                f"needs: {matches_needed[exposed_id]})"
                            )
                            break

                    if not made_progress:
                        self.logger.debug("No new matches in this iteration. Stopping greedy allocation.")
                        break

                    active_indices = [i for i in active_indices if matches_needed[self.exposed_indices[i]] > 0]
            else:
                max_iterations = len(active_indices) * self.n_neighbors * 2
                iteration_count = 0

                while active_indices and iteration_count < max_iterations:
                    iteration_count += 1

                    def sort_key(idx):
                        exp_id = self.exposed_indices[idx]
                        comp_avail = sum(
                            1 for c in candidate_list[idx]['competitive_positions']
                                if int(self.control_indices[c]) not in used_controls
                        )
                        fuzzy_avail = sum(
                            1 for c in candidate_list[idx]['fuzzy_positions']
                                if int(self.control_indices[c]) not in used_controls
                        )
                        effective_secured = len(match_dict.get(exp_id, [])) + safe_capacity[idx]
                        return (effective_secured, comp_avail, fuzzy_avail, idx)


                    active_indices.sort(key=sort_key)
                    made_progress = False

                    for i in active_indices:
                        exposed_id = self.exposed_indices[i]
                        if matches_needed[exposed_id] <= 0:
                            continue

                        cands = candidate_list[i]
                        potential = []

                        for ctrl_idx, d in zip(cands['competitive_positions'], cands['competitive_distances']):
                            ctrl_id = int(self.control_indices[ctrl_idx])
                            if ctrl_id in used_controls:
                                continue
                            if d <= self.threshold:
                                potential.append((d, 0, ctrl_id))

                        if fuzzy_threshold and (fuzzy_threshold_limit is not None):
                            for ctrl_idx, d in zip(cands['fuzzy_positions'], cands['fuzzy_distances']):
                                ctrl_id = int(self.control_indices[ctrl_idx])
                                if ctrl_id in used_controls:
                                    continue
                                if d <= fuzzy_threshold_limit:
                                    penalized = d + self.threshold
                                    potential.append((penalized, 1, ctrl_id))

                        potential.sort(key=lambda x: (x[0], x[1], x[2]))
                        if potential:
                            best_distance, best_type, best_ctrl_id = potential[0]
                            match_dict.setdefault(exposed_id, []).append(best_ctrl_id)
                            used_controls.add(best_ctrl_id)
                            matches_needed[exposed_id] -= 1
                            made_progress = True
                            self.logger.debug(
                                f"Greedy matched {exposed_id} with {best_ctrl_id} "
                                f"(type: {best_type}, score: {best_distance:.3f}, "
                                f"needs: {matches_needed[exposed_id]})"
                            )
                            break

                    if not made_progress:
                        self.logger.debug("No new matches in this iteration. Stopping greedy allocation.")
                        break

                    active_indices = [i for i in active_indices if matches_needed[self.exposed_indices[i]] > 0]
        return match_dict, used_controls

    def _build_dataframe(self, match_dict, candidate_list):
        rows = []
        matched_exposed_set = set(sorted(match_dict.keys()))
        all_exposed = sorted(list(self.exposed_indices))

        exp_orig_to_rel = {orig: i for i, orig in enumerate(self.exposed_indices)}

        for grp_id, exp_idx in enumerate(sorted(match_dict.keys())):
            ctrl_idxs = sorted(match_dict[exp_idx])
            real_exp_id = self.df.iloc[exp_idx][self.patient_id]

            rows.append({
                'patient_id': real_exp_id,
                'match_group': grp_id,
                'is_exposed': 1,
                'n_matches': len(ctrl_idxs),
                'match_distance': 0.0
            })

            for c_idx in ctrl_idxs:
                real_c_id = self.df.iloc[c_idx][self.patient_id]

                dist = self.match_distances_.get((int(exp_idx), int(c_idx)), np.nan)

                rows.append({
                    'patient_id': real_c_id,
                    'match_group': grp_id,
                    'is_exposed': 0,
                    'n_matches': np.nan,
                    'match_distance': dist
                })

        unmatched_exposed = [e for e in all_exposed if e not in matched_exposed_set]
        if len(unmatched_exposed) > 0:
            self.logger.debug(f"{len(unmatched_exposed)} exposed subjects were unmatched. Included with n_matches=0.")

        for exp_idx in unmatched_exposed:
            real_exp_id = self.df.iloc[exp_idx][self.patient_id]
            rows.append({
                'patient_id': real_exp_id,
                'match_group': np.nan,
                'is_exposed': 1,
                'n_matches': 0,
                'match_distance': np.nan
            })

        if not rows:
            self.logger.warning("No matches formed. Returning empty DataFrame.")
            return pd.DataFrame()

        meta_df = pd.DataFrame(rows)
        final_df = self.df.merge(meta_df, left_on=self.patient_id, right_on='patient_id', how='inner')

        if getattr(self, "log_matching_summary", False):
            verbose_matching_results(final_df, self.df, self.exposure_status, self.patient_id)
        return final_df
