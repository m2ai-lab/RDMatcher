import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import List, Literal, Optional, Any, Dict, Union
import logging

# Import your custom modules
from .distance import GowerKNN
from .matching import (
    solve_optimal_assignment, 
    solve_optimal_assignment_mcf,
    extract_exposure_indices,
    compute_weighted_features,
    verbose_matching_results,
    hide_columns
)
from .plot import plot_pca_threshold
# from .logger import rdlogger
# logger = rdlogger(__name__, level="INFO")


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
        """Resolve user gower_weights dict onto the processed column order.

        Rules:
        - Top-level keys must be original feature names.
        - Numeric value applies to all processed columns derived from that original feature.
        - Dict value is only for categorical parents: maps to child columns (either full child column names
          or suffix keys). Only two levels are supported.
        - Missing child weights: warn and default missing children to default_weight.
        """
        processed_to_original = processed_to_original or {c: c for c in processed_cols}
        original_to_processed = original_to_processed or {}
        original_categorical_features = original_categorical_features or set()

        # Validate top-level keys
        valid_original = set(original_to_processed.keys()) if original_to_processed else set(processed_to_original.values())
        unknown_top = [k for k in weights_spec.keys() if k not in valid_original]
        if unknown_top:
            raise ValueError(f"Unknown gower_weights keys: {unknown_top}. Valid original features: {sorted(valid_original)}")

        # Start with defaults
        weights_by_processed: Dict[str, float] = {c: float(default_weight) for c in processed_cols}

        # Apply top-level numeric weights
        for orig_key, val in weights_spec.items():
            if isinstance(val, dict):
                continue
            w = float(val)
            children = original_to_processed.get(orig_key, [])
            if not children:
                # If mapping missing, try direct match
                children = [c for c in processed_cols if processed_to_original.get(c, c) == orig_key]
            for c in children:
                weights_by_processed[c] = w

        # Apply categorical child dicts
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

            # Map provided child keys to actual processed children
            resolved_child_weights: Dict[str, float] = {}
            for child_key, child_w in val.items():
                if isinstance(child_w, dict):
                    raise ValueError(f"gower_weights['{orig_key}']['{child_key}'] is a dict; only two levels supported")
                w = float(child_w)

                # Accept full processed child name
                if child_key in children:
                    resolved_child_weights[child_key] = w
                    continue

                # Accept suffix key: map to f"{orig_key}_{suffix}" if present
                candidate = f"{orig_key}_{child_key}"
                if candidate in children:
                    resolved_child_weights[candidate] = w
                    continue

                raise ValueError(
                    f"Unknown child key '{child_key}' for categorical '{orig_key}'. Valid children: {children}"
                )

            # Warn if not all children provided
            missing = [c for c in children if c not in resolved_child_weights]
            if missing:
                self.logger.warning(
                    f"gower_weights['{orig_key}'] did not include weights for {len(missing)} children. "
                    f"Missing will use default={default_weight}. Valid children: {children}"
                )

            # Apply resolved child overrides
            for c, w in resolved_child_weights.items():
                weights_by_processed[c] = w

        # Log final mapping (original->processed)
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
                 pca_filter: bool = False,
                 logger=None,
                   **kwargs):
        
        self.df = df
        self.exposure_status = exposure_status
        self.patient_id = patient_id
        self.distance_metric = distance_metric
        self.threshold = threshold
        self.n_neighbors = n_neighbors

        # Normalize n_jobs semantics: default to single-threaded unless specified
        requested_n_jobs = kwargs.get('n_jobs', 1)
        if requested_n_jobs is None:
            self.n_jobs = 1
        elif requested_n_jobs == -1:
            self.n_jobs = os.cpu_count() or 1
        elif isinstance(requested_n_jobs, int) and requested_n_jobs >= 1:
            self.n_jobs = int(requested_n_jobs)
        else:
            raise ValueError("n_jobs must be None, -1, or an integer >= 1")

        # Optional: control internal parallel chunking size for Gower
        # If None, GowerKNN will compute chunk_size based on number of workers
        self.parallel_chunk_size = kwargs.get('parallel_chunk_size', None)

        # Streaming/stream block options for Gower kneighbors
        self.streaming = kwargs.get('streaming', 'auto')
        self.stream_block_size = kwargs.get('stream_block_size', None)
        # threshold (GB) for auto-stream decision
        self.stream_threshold_gb = float(kwargs.get('stream_threshold_gb', kwargs.get('stream_threshold_gb', 1.0)))
        # Memory limit (GB) for guarding vectorized concat operations. If None, default to 4.0
        self.memory_limit_gb = kwargs.get('memory_limit_gb', None)
        if self.memory_limit_gb is not None:
            try:
                self.memory_limit_gb = float(self.memory_limit_gb)
            except Exception:
                raise ValueError("memory_limit_gb must be a number (GB)")

        # Validate streaming option
        if self.streaming not in ('auto', 'on', 'off'):
            raise ValueError("streaming must be one of 'auto', 'on', or 'off'")

        # Validate stream_block_size if provided
        if self.stream_block_size is not None:
            try:
                sbs_val = int(self.stream_block_size)
            except Exception:
                raise ValueError("stream_block_size must be an integer >= 1")
            if sbs_val < 1:
                raise ValueError("stream_block_size must be an integer >= 1")
            self.stream_block_size = sbs_val

        # Validate parallel_chunk_size if provided
        if self.parallel_chunk_size is not None:
            try:
                pcs_val = int(self.parallel_chunk_size)
            except Exception:
                raise ValueError("parallel_chunk_size must be an integer >= 1")
            if pcs_val < 1:
                raise ValueError("parallel_chunk_size must be an integer >= 1")
            self.parallel_chunk_size = pcs_val

        # logger
        base = logger if logger is not None else logging.getLogger('rdmatcher')
        self.logger = base.getChild('matcher')
        self.logger.setLevel(logging.NOTSET)       

        # 1. Prepare Base Data (Hide ID)
        self.df_match = hide_columns(self.df.copy(), [self.patient_id])
        
        # Get integer indices for splitting (0..N)
        self.exposed_indices, self.control_indices = extract_exposure_indices(self.df_match, self.exposure_status)

        # 2. Fork Logic based on Metric
        if self.distance_metric == "gower":
            self.logger.info("Gower Metric: Keeping data as DataFrame to preserve Categorical types.")
            features_only = self.df_match.drop(columns=[self.exposure_status])
            self.X_exposed = features_only.iloc[self.exposed_indices]
            self.X_control = features_only.iloc[self.control_indices]

            # Resolve gower_weights provided by user (dict preferred; list supported)
            weights_vector = None
            if gower_weights is None:
                weights_vector = None
            elif isinstance(gower_weights, (list, tuple, np.ndarray)):
                # Legacy positional weights; require exact length match
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
            
            self.logger.info("Fitting GowerKNN model...")
            # Pass n_jobs and parallel_chunk_size into GowerKNN so cdist/kneighbors can default to the requested concurrency
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

            # I need to check the weights have been applied correctly before fitting the model, otherwise the distances will be wrong.
            # I need to see which features still remain that do not have 0 weight
            self.logger.info("Checking feature weights before fitting the model...")
            self.logger.info(f"Features used for matching ({len(feature_names)}): {feature_names}")
            # sanity: show effective weights that were actually applied
            eff_weights = []
            if propensity_col and weight_propensity > 0:
                eff_weights.append((propensity_col, float(weight_propensity)))
            eff_weights.extend([(c, float(weight_numeric)) for c in cov_cols])
            nonzero = [f for f, w in eff_weights if w != 0.0]
            self.logger.info(f"Non-zero weighted features: {nonzero}")
            self.logger.info(f"Effective weights: {eff_weights}")


            self.logger.info(f"Fitting NearestNeighbors ({self.distance_metric}) model...")
            self.nbrs_model = NearestNeighbors(
                metric=self.distance_metric, 
                n_jobs=self.n_jobs,
                algorithm=kwargs.get('algorithm', 'brute')  # <--- Add this for maximum stability
            )
            self.nbrs_model.fit(self.X_control)
            # self.nbrs_model = NearestNeighbors(metric=self.distance_metric, n_jobs=kwargs.get('n_jobs', 1))
            # self.nbrs_model.fit(self.X_control)

    def _apply_pca(self):
        self.logger.info("Applying PCA filter...")
        pca_res = plot_pca_threshold(self.X_exposed, plot=False, return_pca=True, variance_threshold=0.95)
        if pca_res:
            pca_model, cutoff = pca_res
            self.X_exposed = pca_model.transform(self.X_exposed)[:, :cutoff]
            self.X_control = pca_model.transform(self.X_control)[:, :cutoff]
            self.logger.info(f"PCA reduced features to {cutoff} dimensions.")

    def _prefilter_candidates(self, k_candidates, safe_matches, fuzzy_threshold, fuzzy_limit, batch_size):
        self.logger.info(f"Pre-filtering candidates (k={k_candidates})...")
        
        k_candidates = min(k_candidates, self.X_control.shape[0])
        if k_candidates <= 0:
            raise ValueError("No control subjects available for matching.")

        n_exposed = self.X_exposed.shape[0]
        
        # 1. Initialize result arrays
        # We pre-allocate the full result matrix.
        # using float32/int32 saves 50% RAM compared to default float64
        all_distances = np.zeros((n_exposed, k_candidates), dtype=np.float32)
        all_indices = np.zeros((n_exposed, k_candidates), dtype=np.int32)

        # 2. Run Neighbor Search in Batches
        # This loop prevents the "Euclidean Kernel Blow-up" by ensuring we never 
        # query the entire dataset against the entire control set at once.
        
        for start in range(0, n_exposed, batch_size):
            end = min(start + batch_size, n_exposed)
            
            # A. Safe Slicing (Handles DataFrame vs Numpy)
            if isinstance(self.X_exposed, pd.DataFrame):
                batch_X = self.X_exposed.iloc[start:end]
            else:
                batch_X = self.X_exposed[start:end]
            
            # B. Query Neighbors
            if self.distance_metric == "gower":
                # GowerKNN handles its own memory safety for features, 
                # but we batch here to match the outer loop structure.
                dists, idxs = self.nbrs_model.kneighbors(
                    batch_X, n_neighbors=k_candidates, batch_size=batch_size, n_jobs=self.n_jobs # type: ignore
                )
            else:
                # Standard Scikit-Learn (Euclidean/Cosine)
                # We DO NOT pass batch_size here; we rely on the outer loop 
                # to feed it bite-sized chunks so it doesn't crash.
                dists, idxs = self.nbrs_model.kneighbors(batch_X, n_neighbors=k_candidates)
            
            # C. Store Results
            all_distances[start:end] = dists.astype(np.float32)
            all_indices[start:end] = idxs.astype(np.int32)

            # Optional: Log progress for very large datasets
            if (start // batch_size) % 10 == 0 and start > 0:
                self.logger.debug(f"Prefilter progress: {end}/{n_exposed} subjects processed")

        # 3. Vectorized Sort (Deterministic Tie-Breaking)
        # Even though kneighbors returns sorted results, we enforce a strict 
        # (Distance, Index) sort order to ensure run-to-run reproducibility.
        structured = np.empty(all_distances.shape, dtype=[('dist', all_distances.dtype), ('idx', all_indices.dtype)])
        structured['dist'] = all_distances
        structured['idx'] = all_indices
        
        # Fast sorting across the rows
        order = np.argsort(structured, axis=1)
        
        distances = np.take_along_axis(all_distances, order, axis=1)
        indices = np.take_along_axis(all_indices, order, axis=1)

        # Free memory of the temporary arrays
        del all_distances, all_indices, structured, order

        # 4. Vectorized Pass 1: Usage Counts
        # Identify how many times each control is used within the threshold.
        # This replaces the slow Python loop.
        within_mask = distances <= self.threshold
        valid_indices_flat = indices[within_mask]
        
        # bincount is the fastest way to count integer occurrences
        usage_counts = np.bincount(valid_indices_flat, minlength=len(self.control_indices))

        candidate_list = []
        
        # 5. Pass 2: Categorize Candidates
        # This loop constructs the final dictionaries. It is fast enough in Python.
        for i in range(n_exposed):
            dist_row = distances[i]
            idx_row  = indices[i]

            safe = []
            competitive = []
            fuzzy = []

            # Precompute maps for reuse (Pass these to the solver later)
            # idx_to_id = {ctrl_idx: self.control_indices[ctrl_idx] for ctrl_idx in idx_row.tolist()}
            idx_to_id = {int(ctrl_idx): int(self.control_indices[int(ctrl_idx)]) for ctrl_idx in idx_row.tolist()}

            dist_map = {}
            for pos_idx, d_val in zip(idx_row.tolist(), dist_row.tolist()):
                orig_id = int(self.control_indices[int(pos_idx)])
                dist_map[orig_id] = float(d_val)

            # Sanity checks AFTER dist_map exists
            if idx_row.size:
                sample_pos = int(idx_row[0])
                sample_orig = int(self.control_indices[sample_pos])
                if sample_orig not in dist_map:
                    raise RuntimeError("dist_map keying is wrong (expected original control ids).")
                if idx_to_id.get(sample_pos) != sample_orig:
                    raise RuntimeError("idx_to_id mapping is wrong (expected positional->original).")

            # 2. Map ORIGINAL ID -> Distance (instead of Positional -> Distance)
            for j, ctrl_idx in enumerate(idx_row):
                d = float(dist_row[j])
                
                if d <= self.threshold:
                    if usage_counts[int(ctrl_idx)] == 1:
                        # Safe: This control is ONLY within threshold for THIS exposed subject
                        safe.append(int(ctrl_idx))
                        # If we have secured enough safe matches, we are done.
                        if len(safe) >= safe_matches:
                            break
                    else:
                        # Competitive: This control is within threshold for multiple subjects
                        competitive.append(int(ctrl_idx))
                        
                elif fuzzy_threshold and (fuzzy_limit is not None) and d <= fuzzy_limit:
                    fuzzy.append(int(ctrl_idx))
                else:
                    break

            candidate_list.append({
                'safe': safe,
                'competitive': competitive,
                'fuzzy': fuzzy,
                'neighbor_indices': idx_row,
                'neighbor_distances': dist_row,
                'dist_map': dist_map,
                'idx_to_id': idx_to_id
            })

        # 6. Diagnostics Log
        warning_limit_count = sum(1 for c in candidate_list if len(c['safe']) < safe_matches)
        if warning_limit_count > 0:
            self.logger.warning(
                f"{warning_limit_count} exposed subjects have fewer than {safe_matches} safe matches "
                f"within the top {k_candidates} candidates."
            )

        return candidate_list

    def match(self, k_candidates=500, global_optimal=True, competitive_match=False, 
              replacement=False, safe_matches=None, fuzzy_threshold=False, fuzzy_threshold_limit=None,
              batch_size=1024, mcf=False,
              **kwargs) -> pd.DataFrame:
        """
        Perform matching of exposed subjects to control subjects. 
        Parameters
        ----------
        k_candidates : int, default=500
            Number of candidate controls to consider per exposed subject.
        global_optimal : bool, default=True
            If True, perform global optimal matching phase.
        competitive_match : bool, default=False
            If True, perform competitive allocation phase before global optimal.
        replacement : bool, default=False
            If True, allow controls to be matched multiple times.
        safe_matches : int, optional
            Number of "safe" matches to require per exposed subject. If None, defaults to n_neighbors.
        fuzzy_threshold : bool, default=False
            If True, allow fuzzy matching beyond the main threshold.
        fuzzy_threshold_limit : float, optional
            Maximum distance for fuzzy matches. Required if fuzzy_threshold is True.
        Returns
        -------
        pd.DataFrame
            DataFrame containing matched exposed and control subjects.
        """
        
        if replacement:
             raise NotImplementedError("Matching with replacement is not implemented yet.")

        # Dynamic default for safe_matches
        if safe_matches is None:
            safe_matches = self.n_neighbors

        if fuzzy_threshold and fuzzy_threshold_limit is None:
            fuzzy_threshold_limit = self.threshold
        
        # 1. Get Candidates
        candidate_list = self._prefilter_candidates(k_candidates, safe_matches, fuzzy_threshold, fuzzy_threshold_limit, batch_size)
        
        match_dict = {}
        used_controls = set()  # Stores original DF indices

        # 2. Competitive Allocation
        if competitive_match:
            self.logger.info("Running Competitive Allocation Phase...")
            match_dict, used_controls = self._run_competitive_allocation(
                candidate_list, match_dict, used_controls, 
                safe_matches, fuzzy_threshold, fuzzy_threshold_limit
            )

        # 3. Global Optimal Phase
        if global_optimal:
            exposed_to_solve_indices = []
            cleaned_candidate_list = []
            precomputed_subset = []

            for i in range(len(self.exposed_indices)):
                exposed_orig_id = self.exposed_indices[i]
                current_matches = match_dict.get(exposed_orig_id, [])
                if len(current_matches) < self.n_neighbors:
                    exposed_to_solve_indices.append(i)
                    cands = candidate_list[i]
                    all_cands = cands['safe'] + cands['competitive']
                    valid_cands = [c for c in all_cands if cands['idx_to_id'][c] not in used_controls]
                    cleaned_candidate_list.append(valid_cands)

                    # pass neighbor arrays for reuse (no cdist)
                    precomputed_subset.append({
                        'neighbor_indices': cands['neighbor_indices'],
                        'neighbor_distances': cands['neighbor_distances']
                    })

            if exposed_to_solve_indices:
                self.logger.info(f"Running Global Optimal Phase for {len(exposed_to_solve_indices)} subjects...")

                if isinstance(self.X_exposed, pd.DataFrame):
                    subset_X_exposed = self.X_exposed.iloc[exposed_to_solve_indices]
                else:
                    subset_X_exposed = self.X_exposed[exposed_to_solve_indices]

                if mcf: 
                    new_matches = solve_optimal_assignment_mcf(
                        X_exposed_subset=subset_X_exposed,
                        X_control=self.X_control,
                        candidate_lists=cleaned_candidate_list,
                        threshold=self.threshold,
                        metric=self.distance_metric,
                        n_neighbors=self.n_neighbors,
                        all_control_indices=self.control_indices,
                        gower_model=self.nbrs_model if self.distance_metric == "gower" else None,
                        precomputed=precomputed_subset  # reuse kneighbors distances
                    )
                else:
                    new_matches = solve_optimal_assignment(
                        X_exposed_subset=subset_X_exposed,
                        X_control=self.X_control,
                        candidate_lists=cleaned_candidate_list,
                        threshold=self.threshold,
                        metric=self.distance_metric,
                        n_neighbors=self.n_neighbors,
                        all_control_indices=self.control_indices,
                        gower_model=self.nbrs_model if self.distance_metric == "gower" else None,
                        precomputed=precomputed_subset  # reuse kneighbors distances
                    )

                del cleaned_candidate_list #, precomputed_subset

                for i_subset, matched_ctrl_ids in new_matches.items():
                    orig_idx = exposed_to_solve_indices[i_subset]
                    exposed_id = self.exposed_indices[orig_idx]
                    current = match_dict.get(exposed_id, [])
                    needed = self.n_neighbors - len(current)
                    if needed > 0:
                        match_dict[exposed_id] = current + matched_ctrl_ids[:needed]
                        used_controls.update(matched_ctrl_ids[:needed])

        # 4. Construct Result
        self.matched_data = self._build_dataframe(match_dict, candidate_list)

        return self.matched_data

    def _run_competitive_allocation(self, candidate_list, match_dict, used_controls, safe_matches, fuzzy_threshold, fuzzy_threshold_limit):
        # Identify limited subjects
        limited_indices = []
        for i, cand in enumerate(candidate_list):
            if len(cand['safe']) < safe_matches:
                limited_indices.append(i)

        if not limited_indices:
            self.logger.debug("No limited subjects identified for competitive allocation.")
            return match_dict, used_controls

        matches_needed = {self.exposed_indices[i]: self.n_neighbors for i in limited_indices}

        # Phase 1: Assign Safe Controls
        self.logger.info("Phase 1: Assigning safe controls for limited group.")
        for i in limited_indices:
            exposed_id = self.exposed_indices[i]
            assigned_count = 0

            for ctrl_idx in candidate_list[i]['safe']:
                ctrl_id = candidate_list[i]['idx_to_id'][ctrl_idx]
                if ctrl_id not in used_controls:
                    match_dict.setdefault(exposed_id, []).append(ctrl_id)
                    used_controls.add(ctrl_id)
                    matches_needed[exposed_id] -= 1
                    assigned_count += 1
                    self.logger.debug(f"Assigned safe match for {exposed_id}: {ctrl_id} (needs: {matches_needed[exposed_id]})")
                    if assigned_count >= self.n_neighbors:
                        break

        # Phase 2: Iterative Greedy for competitive/fuzzy
        self.logger.info("Phase 2: Starting iterative greedy for competitive/fuzzy controls.")
        active_indices = [i for i in limited_indices if matches_needed[self.exposed_indices[i]] > 0]

        if active_indices:
            max_iterations = len(active_indices) * self.n_neighbors * 2
            iteration_count = 0

            while active_indices and iteration_count < max_iterations:
                iteration_count += 1

                def sort_key(idx):
                    exp_id = self.exposed_indices[idx]
                    comp_avail = sum(
                        1 for c in candidate_list[idx]['competitive']
                            if candidate_list[idx]['idx_to_id'][c] not in used_controls
                    )
                    fuzzy_avail = sum(
                        1 for c in candidate_list[idx]['fuzzy']
                        if candidate_list[idx]['idx_to_id'][c] not in used_controls
                    )
                    # Deterministic ordering
                    return (len(match_dict.get(exp_id, [])), comp_avail, fuzzy_avail, idx)


                active_indices.sort(key=sort_key)
                made_progress = False

                for i in active_indices:
                    exposed_id = self.exposed_indices[i]
                    if matches_needed[exposed_id] <= 0:
                        continue

                    cands = candidate_list[i]
                    dist_map = cands['dist_map']
                    idx_to_id = cands['idx_to_id']


                    # Sanity check: Ensure candidate indices map correctly to original IDs and distances
                    for ctrl_idx in (cands['competitive'][:3] + cands['fuzzy'][:3]):
                        ctrl_id = idx_to_id.get(ctrl_idx)
                        if ctrl_id is None:
                            raise RuntimeError("Candidate ctrl_idx not found in idx_to_id.")
                        if ctrl_id not in dist_map:
                            raise RuntimeError("Candidate ctrl_id not found in dist_map (index-space mismatch).")

                    # Build potential with deterministic tie-breakers
                    potential = []

                    # Competitive (prefer over fuzzy): type_priority = 0
                    for ctrl_idx in cands['competitive']:
                        ctrl_id = idx_to_id[ctrl_idx]
                        if ctrl_id in used_controls:
                            continue
                        d = dist_map.get(ctrl_id)
                        if d is not None and d <= self.threshold:
                            # (distance, type_priority, control_id)
                            potential.append((d, 0, ctrl_id))

                    # Fuzzy (penalized): type_priority = 1
                    if fuzzy_threshold and (fuzzy_threshold_limit is not None):
                        for ctrl_idx in cands['fuzzy']:
                            ctrl_id = idx_to_id[ctrl_idx]
                            if ctrl_id in used_controls:
                                continue
                            d = dist_map.get(ctrl_id)
                            if d is not None and d <= fuzzy_threshold_limit:
                                penalized = d + self.threshold
                                potential.append((penalized, 1, ctrl_id))

                    # Deterministic sort and pick
                    potential.sort(key=lambda x: (x[0], x[1], x[2]))  # distance, then type, then control id
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
                        break  # re-sort next iteration

                if not made_progress:
                    self.logger.debug("No new matches in this iteration. Stopping greedy allocation.")
                    break

                active_indices = [i for i in active_indices if matches_needed[self.exposed_indices[i]] > 0]

        return match_dict, used_controls

    def _build_dataframe(self, match_dict, candidate_list):
        rows = []
        matched_exposed_set = set(sorted(match_dict.keys()))
        all_exposed = sorted(list(self.exposed_indices))

        # 1. Map Original DataFrame Index -> Relative Index (0 to N_exposed)
        # This allows us to find the correct entry in candidate_list
        exp_orig_to_rel = {orig: i for i, orig in enumerate(self.exposed_indices)}

        # Matched exposed
        for grp_id, exp_idx in enumerate(sorted(match_dict.keys())):
            ctrl_idxs = sorted(match_dict[exp_idx])
            real_exp_id = self.df.iloc[exp_idx][self.patient_id]
            
            # --- PREPARE DISTANCE LOOKUP ---
            # Get the relative index for this exposed subject to look up their specific distance map
            rel_idx = exp_orig_to_rel[exp_idx]
            # dist_map keys are Control Original Indices, values are Float Distances
            current_dist_map = candidate_list[rel_idx]['dist_map']

            # Append Exposed Row
            rows.append({
                'patient_id': real_exp_id,
                'match_group': grp_id,
                'is_exposed': 1,
                'n_matches': len(ctrl_idxs),
                'match_distance': 0.0 # Distance to self is 0 (or use np.nan)
            })
            
            # Append Control Rows
            for c_idx in ctrl_idxs:
                real_c_id = self.df.iloc[c_idx][self.patient_id]
                
                # Retrieve the specific distance for this pair
                dist = current_dist_map.get(c_idx, np.nan)
                
                rows.append({
                    'patient_id': real_c_id,
                    'match_group': grp_id,
                    'is_exposed': 0,
                    'n_matches': np.nan,
                    'match_distance': dist
                })

        # Unmatched exposed
        unmatched_exposed = [e for e in all_exposed if e not in matched_exposed_set]
        if len(unmatched_exposed) > 0:
            self.logger.warning(f"{len(unmatched_exposed)} exposed subjects were unmatched. Included with n_matches=0.")    
        
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

        verbose_matching_results(final_df, self.df, self.exposure_status, self.patient_id)
        return final_df
