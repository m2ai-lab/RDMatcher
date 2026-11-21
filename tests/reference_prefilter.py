# Test-only scalar rule from working commit 775ee13.
import numpy as np
from rdmatcher._general_retrieval import DemandNeighbors

def _prefilter_candidates(self, k_candidates, safe_matches, fuzzy_threshold, fuzzy_limit, batch_size, *, competition=False):
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    self.logger.info(f'Pre-filtering candidates (k={k_candidates})...')
    k_candidates = min(k_candidates, self.X_control.shape[0])
    if k_candidates <= 0:
        raise ValueError('No control subjects available for matching.')
    n_exposed = self.X_exposed.shape[0]
    if self.ps_eligible_sets is not None and self.distance_metric not in {'gower', 'mahalanobis', 'precomputed'}:
        raise ValueError('PS-Hybrid mode (ps_eligible_sets) is only supported with Gower or Mahalanobis distance metric.')
    backend = getattr(self.nbrs_model, '_nn_model', None)
    if backend is None:
        backend = self.nbrs_model
    tree = getattr(backend, '_tree', None)
    tree_before = tree.get_n_calls() if tree is not None else None
    lazy = DemandNeighbors(self, k_candidates, batch_size)
    distances, indices = (lazy.distances, lazy.indices)
    self.candidate_retrieval_stats_ = lazy.stats
    horizons = np.ones(n_exposed, dtype=np.int32)
    iteration_count = 0
    n_controls = len(self.control_indices)
    first_controls = indices[:, 0]
    first_valid = (first_controls >= 0) & (distances[:, 0] <= self.threshold)
    usage_counts = np.bincount(first_controls[first_valid], minlength=n_controls).astype(np.int32, copy=False)
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
            unique_controls, inverse, arrivals = np.unique(edge_controls, return_inverse=True, return_counts=True)
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
        idx_row = indices[i]
        horizon = int(horizons[i])
        safe_positions, safe_distances = ([], [])
        competitive_positions, competitive_distances = ([], [])
        fuzzy_positions, fuzzy_distances = ([], [])
        for j, ctrl_idx in enumerate(idx_row[:horizon]):
            d = float(dist_row[j])
            if int(ctrl_idx) >= 0 and np.isfinite(d) and (d <= self.threshold):
                if usage_counts[int(ctrl_idx)] == 1:
                    safe_positions.append(int(ctrl_idx))
                    safe_distances.append(d)
                    if len(safe_positions) >= safe_matches:
                        break
                else:
                    competitive_positions.append(int(ctrl_idx))
                    competitive_distances.append(d)
            elif fuzzy_threshold and fuzzy_limit is not None and (d <= fuzzy_limit):
                fuzzy_positions.append(int(ctrl_idx))
                fuzzy_distances.append(d)
            else:
                break
        candidate_list.append({'safe_positions': np.asarray(safe_positions, dtype=np.int32), 'safe_distances': np.asarray(safe_distances, dtype=getattr(self, 'distance_dtype', np.float32)), 'competitive_positions': np.asarray(competitive_positions, dtype=np.int32), 'competitive_distances': np.asarray(competitive_distances, dtype=getattr(self, 'distance_dtype', np.float32)), 'fuzzy_positions': np.asarray(fuzzy_positions, dtype=np.int32), 'fuzzy_distances': np.asarray(fuzzy_distances, dtype=getattr(self, 'distance_dtype', np.float32)), 'neighbor_indices': idx_row, 'neighbor_distances': dist_row, 'candidate_horizon': horizon})
    if lazy is not None and self.ps_eligible_sets is not None:
        self.ps_hybrid_diagnostics_ = []
        for i, eligible in enumerate(self.ps_eligible_sets):
            valid = indices[i] >= 0
            values = distances[i][valid]
            finite = values[np.isfinite(values)]
            within = int(np.count_nonzero(values <= self.threshold))
            self.ps_hybrid_diagnostics_.append(dict(treated_position=i, ps_eligible=len(eligible), retained=len(values), within_threshold=within, retained_within_threshold=within, min_distance=float(finite.min()) if len(finite) else np.nan, max_retained_distance=float(finite.max()) if len(finite) else np.nan, truncated_by_k=len(eligible) > len(values), within_threshold_truncated_by_k=bool(len(eligible) > len(values) and within > 0), retrieval_scope='loaded_prefix'))
    warning_limit_count = sum((1 for c in candidate_list if len(c['safe_positions']) < safe_matches))
    if warning_limit_count > 0:
        self.logger.debug(f'{warning_limit_count} exposed subjects have fewer than {safe_matches} safe matches within the top {k_candidates} candidates.')
    self.candidate_retrieval_stats_['neighbor_backend'] = type(self.nbrs_model).__name__
    self.candidate_retrieval_stats_['neighbor_algorithm'] = getattr(backend, '_fit_method', None)
    if hasattr(self.nbrs_model, 'normalization_stats_'):
        self.candidate_retrieval_stats_['normalization'] = dict(self.nbrs_model.normalization_stats_)
    if tree is not None:
        self.candidate_retrieval_stats_['tree_distance_calls'] = int(tree.get_n_calls() - tree_before)
    return candidate_list
