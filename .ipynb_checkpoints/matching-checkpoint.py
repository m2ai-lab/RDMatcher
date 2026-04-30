import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix, csr_matrix
from collections import Counter
import logging
import seaborn as sns
import matplotlib.pyplot as plt

from utils import hide_columns
from plot import plot_numeric_feature_balance, plot_pca_threshold
from logger import epilogger




logger = epilogger(__name__, level="INFO")


# ---------------------------
# Validate Input
# ---------------------------
def validate_input_columns(df, required_cols):
    """Ensure that all required columns are present in the DataFrame."""
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in the input dataframe.")


# ---------------------------
# Compute Weighted Features
# ---------------------------
def compute_weighted_features(df, exposure_status, propensity_col, weight_propensity, weight_numeric, patient_id):
    """
    Drop the sensitive patient_id from the features and compute the weighted features.
    
    Returns:
        df_match: DataFrame for matching (without subject id).
        X_combined: Combined weighted features (propensity and numeric covariates).
        covariate_cols: List of covariate columns used.
    """
    df_match = df.copy()
    df_match = hide_columns(df_match, [patient_id])
    
    # Weighted propensity scores.
    propensity_scores = df_match[propensity_col].values.reshape(-1, 1)
    propensity_weighted = weight_propensity * propensity_scores

    # Prepare the numeric/covariate matrix.
    covariate_cols = [col for col in df_match.columns if col not in [exposure_status, propensity_col]]
    X_covariates = df_match[covariate_cols].values
    X_weighted = weight_numeric * X_covariates

    # Combine the weighted propensity with the covariates.
    X_combined = np.hstack([propensity_weighted, X_weighted])
    return df_match, X_combined, covariate_cols

# ---------------------------
# Extract Groups
# ---------------------------
def extract_exposure_indices(df, exposure_status):
    """
    Returns the indices for exposed and control subjects.
    """
    exposed_indices = df.index[df[exposure_status] == 1].tolist()
    control_indices = df.index[df[exposure_status] == 0].tolist()

    if len(exposed_indices) == 0 or len(control_indices) == 0:
        raise ValueError("There must be at least one exposed and one control subject in the data.")
    return exposed_indices, control_indices

# ---------------------------
# Compute Distance Matrix
# ---------------------------
def compute_distance_matrix(X_exposed, X_control, distance_metric="euclidean", custom_distance_fn=None):
    """
    Compute the distance matrix between exposed and control subjects.
    """
    if custom_distance_fn is not None:
        return custom_distance_fn(X_exposed, X_control)
    elif isinstance(distance_metric, str) or callable(distance_metric):
        return cdist(X_exposed, X_control, metric=distance_metric) # type: ignore
    else:
        raise ValueError("distance_metric must be a valid string or a callable function.")


# -------------------------------------------------------------------
# Pre-filter Candidate Controls per Exposed
# -------------------------------------------------------------------
def prefilter_candidates(X_exposed, X_control, threshold, k_candidates=500, distance_metric="euclidean", 
                         safe_matches=10, fuzzy_threshold=False, fuzzy_threshold_limit=None):
    """
    For each exposed subject, retrieve up to k_candidates controls, then identify safe and competitive controls.
    
    A control is considered "safe" if it's only within threshold distance of one exposed subject.
    A control is "competitive" if it's within threshold distance of multiple exposed subjects.
    A control is "fuzzy" if it is just beyond the threshold but still within the threshold limit. These are ordered from closest to furthest.
    
    Returns:
    candidate_indices_list: A list (length == len(X_exposed)) where each element contains
                          a dict with 'safe' and 'competitive' control indices.
    """
    nbrs = NearestNeighbors(n_neighbors=k_candidates, metric=distance_metric).fit(X_control)
    distances, indices = nbrs.kneighbors(X_exposed, n_neighbors=k_candidates)
    
    # First pass: build reverse index of controls to exposed subjects
    control_to_exposed = {}
    for exp_idx in range(X_exposed.shape[0]):
        valid_controls = indices[exp_idx][distances[exp_idx] <= threshold]
        for ctrl_idx in valid_controls:
            if ctrl_idx not in control_to_exposed:
                control_to_exposed[ctrl_idx] = []
            control_to_exposed[ctrl_idx].append(exp_idx)
    
    # Second pass: categorize controls as safe, competitive, or fuzzy for each exposed subject
    candidate_indices_list = []
    for exp_idx in range(X_exposed.shape[0]):
        valid_controls = indices[exp_idx][distances[exp_idx] <= threshold]
        safe_controls = []
        competitive_controls = []
        fuzzy_controls = []
        
        # Sort controls by distance for this exposed subject
        ctrl_distances = distances[exp_idx][distances[exp_idx] <= threshold]
        sorted_pairs = sorted(zip(valid_controls, ctrl_distances), key=lambda x: x[1])

        # Categorize each control
        for ctrl_idx, dist in sorted_pairs:
            if len(control_to_exposed[ctrl_idx]) == 1:  # Safe control
                safe_controls.append(ctrl_idx)
                if len(safe_controls) >= safe_matches:
                    break
            else:  # Competitive control
                competitive_controls.append(ctrl_idx)

        if fuzzy_threshold and fuzzy_threshold_limit is not None:
            # Include fuzzy controls that are just beyond the threshold but within the fuzzy limit.
            # fuzzy_controls = indices[exp_idx][(distances[exp_idx] > threshold) & (distances[exp_idx] <= fuzzy_threshold_limit)]
            fuzzy_distances = distances[exp_idx][(distances[exp_idx] > threshold) & (distances[exp_idx] <= fuzzy_threshold_limit)]
            fuzzy_sorted_pairs = sorted(zip(fuzzy_controls, fuzzy_distances), key=lambda x: x[1])
            for ctrl_idx, dist in fuzzy_sorted_pairs:
                fuzzy_controls.append(ctrl_idx)
        
        candidate_indices_list.append({
            'safe': safe_controls,
            'competitive': competitive_controls,
            'fuzzy': fuzzy_controls
        })
    
    return candidate_indices_list

def check_candidate_availability(X_exposed, X_control, threshold, k_candidates, distance_metric, safe_matches, fuzzy_threshold, fuzzy_threshold_limit):
    """
    Check candidate availability using the prefilter stage.
    Now checks for both safe and competitive controls.
    """
    candidate_indices_list = prefilter_candidates(X_exposed, X_control, threshold, k_candidates, distance_metric, safe_matches, fuzzy_threshold, fuzzy_threshold_limit)
    
    no_candidates = [i for i, candidates in enumerate(candidate_indices_list) 
                    if len(candidates['safe']) == 0 and len(candidates['competitive']) == 0]
    
    if no_candidates:
        raise ValueError(f"The threshold of {threshold} is too restrictive. "
                       f"Exposed subjects at indices {no_candidates} have no candidate controls. "
                       "Consider relaxing the threshold or reviewing your weighting/distance metric. "
                       "Note that 'fuzzy' controls are not considered in this check.")
    
    # Log information about safe vs competitive controls
    for i, candidates in enumerate(candidate_indices_list):
        logger.debug(f"Exposed subject {i}: {len(candidates['safe'])} safe controls, "
                    f"{len(candidates['competitive'])} competitive controls, "
                    f"{len(candidates['fuzzy'])} fuzzy controls (Only considered if competitive_match is emabled for these subjects).")
    
    return candidate_indices_list



# -------------------------------------------------------------------
# Build Sparse Cost Matrix for Candidate Pairs
# -------------------------------------------------------------------
# def build_sparse_cost_matrix(X_exposed, X_control, candidate_indices_list, threshold, distance_metric="euclidean"):
#     """
#     Build a sparse cost matrix where rows correspond to exposed subjects and columns to control subjects.
#     Only candidate pairs (within the provided candidate_indices_list) within the threshold are populated.
    
#     Returns:
#         cost_matrix_sparse: A csr_matrix of shape (n_exposed, n_controls)
#     """
#     rows = []
#     cols = []
#     data = []
    
#     for i, candidates in enumerate(candidate_indices_list):
#         if not candidates:
#             continue
#         # Combine safe and competitive candidates (or choose one as required)
#         candidate_idxs = candidates.get("safe", []) + candidates.get("competitive", [])
#         if not candidate_idxs:  # Skip if there are no candidates after combining.
#             continue
#         candidate_idxs = np.array(candidate_idxs, dtype=int)
#         # Compute distances for candidate controls only.
#         if isinstance(distance_metric, str) or callable(distance_metric):
#             distances = cdist([X_exposed[i]], X_control[candidate_idxs], metric=distance_metric).flatten() # type: ignore
#         else:
#             raise ValueError("distance_metric must be a valid string or a callable function.")
#         for j, d in enumerate(distances):
#             if d <= threshold:
#                 rows.append(i)
#                 cols.append(candidate_idxs[j])
#                 data.append(d)
    
#     cost_matrix_sparse = coo_matrix((data, (rows, cols)), shape=(X_exposed.shape[0], X_control.shape[0])).tocsr()
#     return cost_matrix_sparse



# -------------------------------------------------------------------
# Build Reduced Sparse Cost Matrix
# -------------------------------------------------------------------
def build_reduced_sparse_cost_matrix(X_exposed, X_control, candidate_indices_list, threshold, distance_metric="euclidean"):
    """
    Build a sparse cost matrix only using the candidate controls (union across all exposed subjects).
    
    Returns:
        cost_matrix_sparse: A csr_matrix of shape (n_exposed, n_candidates)
        union_candidate_list: A sorted list of the unique candidate control indices (original indices).
        mapping: A dict mapping original control indices to the reduced column index.
    """
    # 1. Compute the union of candidate controls.
    union_candidate_set = set()
    for candidates in candidate_indices_list:
        candidate_idxs = candidates.get("safe", []) + candidates.get("competitive", [])
        union_candidate_set.update(candidate_idxs)
    union_candidate_list = sorted(list(union_candidate_set))
    
    # 2. Build a mapping from original control index to new (reduced) column index.
    mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(union_candidate_list)}
    
    # 3. Loop over each exposed subject and compute distances only for these candidate controls.
    rows = []
    cols = []
    data = []
    n_exposed = X_exposed.shape[0]
    
    for i, candidates in enumerate(candidate_indices_list):
        candidate_idxs = candidates.get("safe", []) + candidates.get("competitive", [])
        if not candidate_idxs:
            continue
        # Instead of computing distances for all candidate controls at once (which assumes a shared ordering),
        # we compute the distance for each candidate control individually and then place the result in the reduced column.
        for orig_idx in candidate_idxs:
            # Check that this candidate is in the union (it should be)
            if orig_idx not in mapping:
                continue
            reduced_col = mapping[orig_idx]
            # Compute the distance between the exposed subject i and this candidate control.
            d = cdist([X_exposed[i]], X_control[[orig_idx]], metric=distance_metric).item() # type: ignore
            if d <= threshold:
                rows.append(i)
                cols.append(reduced_col)
                data.append(d)
                
    shape = (n_exposed, len(union_candidate_list))
    cost_matrix_sparse = coo_matrix((data, (rows, cols)), shape=shape).tocsr()
    
    return cost_matrix_sparse, union_candidate_list, mapping


# ---------------------------
# Greedy / Local Matching
# ---------------------------
def match_greedy(X_exposed, X_control, exposed_indices, control_indices, threshold, n_neighbors, **kwargs):
    """
    Perform the greedy (local optimal) matching approach using NearestNeighbors.
    
    Returns:
        match_dict: A dictionary mapping exposed subject id to a list of control ids.
    """

    debug = kwargs.get('debug', False)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info("Starting greedy (local optimal) matching.")


    nbrs = NearestNeighbors(radius=threshold, n_neighbors=n_neighbors, metric='euclidean')
    nbrs.fit(X_control)
    distances_all, indices_all = nbrs.radius_neighbors(X_exposed)

    match_dict = {}
    used_control_indices = set()

    for i, (dists, neighbor_idxs) in enumerate(zip(distances_all, indices_all)):
        exposed_id = exposed_indices[i]
        if len(neighbor_idxs) == 0:
            logger.debug(f"Exposed subject {exposed_id} has no neighbors within threshold.")
            continue

        sorted_order = np.argsort(dists)
        sorted_neighbor_idxs = neighbor_idxs[sorted_order]

        available_controls = [
            control_indices[j]
            for j in sorted_neighbor_idxs
            if control_indices[j] not in used_control_indices
        ]
        if available_controls:
            selected_controls = available_controls[:n_neighbors]
            match_dict[exposed_id] = selected_controls
            used_control_indices.update(selected_controls)
            logger.debug(f"Exposed subject {exposed_id} matched with controls: {selected_controls}")
        else:
            logger.debug(f"Exposed subject {exposed_id} has neighbors but none are available (already used).")

    if len(match_dict) == 0:
        raise ValueError("No matches found. Consider relaxing the threshold or reviewing the feature weighting.")
    
    logger.info("Greedy matching completed successfully.")

    return match_dict

# ---------------------------
# Build Matched DataFrame
# ---------------------------
def build_matched_dataframe(df, match_dict):
    """
    Given the original DataFrame and a dictionary of matches, build the final matched DataFrame,
    adding a 'match_group' and 'n_matches' column.
    """
    exposed_list = []
    controls_list = []
    for group_id, (exposed_id, control_ids) in enumerate(match_dict.items()):
        # Get exposed subject row self-contained.
        exposed_df = df.loc[[exposed_id]].copy()
        exposed_df["match_group"] = group_id
        exposed_df["n_matches"] = len(control_ids) if control_ids else 0
        exposed_list.append(exposed_df)

        # For each control, carry along all columns.
        if control_ids is not None:
            controls_df = df.loc[control_ids].copy()
            controls_df["match_group"] = group_id
            controls_df["n_matches"] = None
            controls_list.append(controls_df)

    matched_exposed = pd.concat(exposed_list)
    matched_controls = pd.concat(controls_list)
    matched_data = pd.concat([matched_exposed, matched_controls]).reset_index(drop=True)
    return matched_data



# ----------------------------
# Optimal matching with Hungarian Algorithm
# ----------------------------
def match_optimal(X_exposed, X_control, candidate_list, threshold, distance_metric, 
                  n_neighbors, match_dict, exposed_indices):
    """
    Perform optimal matching using the Hungarian algorithm (linear_sum_assignment).
    This function builds a reduced sparse cost matrix over only the union of candidate controls,
    expands it to account for n_neighbors, and then solves the assignment problem.
    """
    reduced_sparse, union_candidate_list, mapping = build_reduced_sparse_cost_matrix(
        X_exposed, X_control, candidate_list, threshold, distance_metric
    )
    logger.debug(f"Reduced cost matrix shape: {reduced_sparse.shape}")

    # Convert the reduced sparse matrix to a dense matrix.
    dense_cost_reduced = np.full((X_exposed.shape[0], len(union_candidate_list)), np.inf)
    coo = reduced_sparse.tocoo()
    dense_cost_reduced[coo.row, coo.col] = coo.data

    # Expand each row n_neighbors times.
    expanded_cost = np.repeat(dense_cost_reduced, n_neighbors, axis=0)
    logger.debug(f"Expanded cost matrix shape: {expanded_cost.shape}")

    # Add placeholder controls to avoid infeasibility.
    def add_placeholder_controls(expanded_cost, placeholder_cost=1e6):
        num_rows, num_cols = expanded_cost.shape
        placeholder_matrix = np.full((num_rows, num_rows), placeholder_cost)
        return np.hstack([expanded_cost, placeholder_matrix])
    augmented_cost = add_placeholder_controls(expanded_cost)
    logger.debug(f"Augmented cost matrix shape: {augmented_cost.shape}")

    # Solve the assignment problem.
    row_ind, col_ind = linear_sum_assignment(augmented_cost)

    # Regroup assignments. For each exposed subject (its cost info repeated n_neighbors times)
    temp_match = {i: [] for i in range(X_exposed.shape[0])}
    for exp_rep, col in zip(row_ind, col_ind):
        if col < expanded_cost.shape[1]:
            original_exp_index = exp_rep // n_neighbors
            temp_match[original_exp_index].append(col)

    # Map the reduced column indices back to the original control indices and update used_control_indices.
    for j, reduced_cols in temp_match.items():
        if reduced_cols:
            matched_controls = [union_candidate_list[r] for r in reduced_cols]
            match_dict[exposed_indices[j]] = matched_controls
        else:
            logger.warning(f"Exposed subject {exposed_indices[j]} has no valid matches under the threshold.")
    if len(match_dict) == 0:
        raise ValueError("No matches found under the threshold constraint. Consider relaxing the threshold.")

    return match_dict




# ----------------------------
# Helper Function for Global Optimal Matching with Multiple Neighbors
# ----------------------------
# def match_global_optimal(
#         X_exposed, 
#         X_control, 
#         exposed_indices, 
#         control_indices,
#         threshold, 
#         n_neighbors, 
#         k_candidates,
#         safe_matches,
#         distance_metric, 
#         custom_distance_fn,
#         replacement,
#         debug=False,
#         competitive_match=False,
#         fuzzy_threshold=False,
#         fuzzy_threshold_limit=None,
#         **kwargs
#         ):
#     """
#     Perform global optimal matching using the Hungarian algorithm (linear_sum_assignment).
#     If 'competitive_match' is True, then subjects with limited safe controls 
#     (i.e. fewer than safe_matches) are handled via greedy matching first (in order of lowest
#     number of safe controls and, in case of ties, lowest number of competitive controls). The remaining 
#     subjects are then matched using the standard global optimal routine.
#     """
#     if debug:
#         logger.setLevel(logging.DEBUG)
#     else:
#         logger.setLevel(logging.INFO)
#     logger.info(f"Starting global optimal matching with {n_neighbors} neighbor(s).")
#     if competitive_match:
#         logger.info("Competitive matching is enabled. Subjects with limited safe controls will be handled separately and first.")

#     # fix input parameters if fuzzy_threshold is True and I screwed up the defaults.
#     if fuzzy_threshold_limit is None:
#         fuzzy_threshold_limit = threshold
    
#     if replacement:
#         match_dict = {}
#         # Use the full cost matrix (can be optimized similarly if needed)
#         cost_matrix = compute_distance_matrix(X_exposed, X_control, distance_metric, custom_distance_fn)
#         cost_matrix_masked = cost_matrix.copy()
#         cost_matrix_masked[cost_matrix_masked > threshold] = np.inf
#         for i, row in enumerate(cost_matrix_masked):
#             exposed_id = exposed_indices[i]
#             candidate_control_idxs = np.argsort(row)
#             valid_candidates = [j for j in candidate_control_idxs if row[j] != np.inf]
#             if len(valid_candidates) == 0:
#                 logger.warning(f"Exposed subject {exposed_id} has no valid matches under the threshold.")
#                 continue
#             selected = valid_candidates[:n_neighbors]
#             matched_controls = [control_indices[j] for j in selected]
#             match_dict[exposed_id] = matched_controls
#             logger.debug(f"Exposed subject {exposed_id} matched with controls (with replacement): {matched_controls}")
    
#     else:
#         # Without replacement:
#         # Pre-filter candidate controls.
#         candidate_list = check_candidate_availability(
#             X_exposed=X_exposed, 
#             X_control=X_control, 
#             threshold=threshold, 
#             k_candidates=k_candidates,
#             distance_metric=distance_metric, 
#             safe_matches=safe_matches,
#             fuzzy_threshold=fuzzy_threshold,
#             fuzzy_threshold_limit=fuzzy_threshold_limit
#         )
        
#         # If competitive_match is not requested, proceed with standard global optimal matching.
#         if not competitive_match:

#             match_dict = {}
#             match_dict = match_optimal(X_exposed, X_control, candidate_list, threshold, distance_metric,
#                                        n_neighbors, match_dict, exposed_indices)
                
#         else:
#             # Competitive matching is requested:
#             # Partition subjects into two groups: limited subjects get greedy matching first.
#             limited_group = []
#             regular_group = []
#             for i, cand in enumerate(candidate_list):
#                 # Define "limited" as having fewer safe candidates than safe_matches.
#                 if len(cand.get("safe", [])) < safe_matches:
#                     limited_group.append(i)
#                     # see if they also have a limited number of competitive candidates.
#                     if len(cand.get("competitive", [])) < n_neighbors:
#                         logger.warning(f"Exposed subject {exposed_indices[i]} is in the limited group with {len(cand.get('safe', []))} safe and {len(cand.get('competitive', []))} competitive controls.")
#                         logger.warning(f"  Consider relaxing the threshold or increasing the number of safe matches.")
#                 else:
#                     regular_group.append(i)
            
#             match_dict = {}
#             used_control_indices = set()
            
#             ### Uncomment >>>
#             # # First, handle the limited group using greedy matching.
#             # if limited_group:
#             #     # Reorder the limited group: sort by number of safe controls, then by number of competitive controls, then by number of fuzzy controls (all ascending).
#             #     greedy_sorted = sorted(
#             #         limited_group,
#             #         key=lambda i: (len(candidate_list[i].get("safe", [])),
#             #                        len(candidate_list[i].get("competitive", [])),
#             #                        len(candidate_list[i].get("fuzzy", [])))
#             #     )
#             #     for i in greedy_sorted:
#             #         exposed_id = exposed_indices[i]
#             #         # For greedy matching, consider only available controls not yet used.
#             #         available_control_idxs = [j for j in range(X_control.shape[0])
#             #                                   if control_indices[j] not in used_control_indices]
#             #         if len(available_control_idxs) == 0:
#             #             logger.debug(f"No available controls left for exposed subject {exposed_id}.")
#             #             continue
#             #         X_control_avail = X_control[available_control_idxs]
#             #         # Perform a radius search on the available controls.
#             #         if fuzzy_threshold and fuzzy_threshold_limit is not None:
#             #             # If fuzzy matching is enabled, we need to consider controls that are just beyond the threshold.
#             #             # This is done by adjusting the radius_neighbors call.
#             #             nbrs = NearestNeighbors(
#             #                 radius=fuzzy_threshold_limit, n_neighbors=n_neighbors, metric=distance_metric
#             #             )
#             #         else:
#             #             nbrs = NearestNeighbors(
#             #                 radius=threshold, n_neighbors=n_neighbors, metric=distance_metric
#             #             )
#             #         X_exposed_single = X_exposed[i].reshape(1, -1)
#             #         nbrs.fit(X_control_avail)
#             #         distances, neighbor_idxs = nbrs.radius_neighbors(X_exposed_single)
#             #         if len(neighbor_idxs[0]) == 0:
#             #             logger.debug(f"Exposed subject {exposed_id} has no neighbors within threshold in greedy matching.")
#             #             continue
#             #         sorted_order = np.argsort(distances[0])
#             #         sorted_neighbor_idxs = neighbor_idxs[0][sorted_order]
#             #         # Map available indices to original control IDs.
#             #         available_controls = [control_indices[available_control_idxs[j]] for j in sorted_neighbor_idxs]
#             #         if available_controls:
#             #             selected_controls = available_controls[:n_neighbors]
#             #             match_dict[exposed_id] = selected_controls
#             #             used_control_indices.update(selected_controls)
#             #             logger.debug(f"Greedy (limited) matched exposed subject {exposed_id} with controls: {selected_controls}")
#             #         else:
#             #             logger.debug(f"Exposed subject {exposed_id} has neighbors but none available (already used).")
#             # ### Uncomment <<<
            
#             # First, handle the limited group using greedy matching.
#             if limited_group:
#                 # Keep track of how many matches each limited individual still needs
#                 # This dict will contain {exposed_id: matches_remaining}
#                 matches_needed = {exposed_indices[i]: n_neighbors for i in limited_group}

#                 # --- PHASE 1: Assign Safe Controls for the Limited Group ---
#                 # These are non-competitive and should be processed first for limited individuals.
#                 logger.debug("Phase 1: Assigning safe controls for limited group.")
#                 for i in limited_group:
#                     exposed_id = exposed_indices[i]
#                     # `candidate_list[i].get("safe", [])` is assumed to contain original control IDs.
#                     current_safe_controls_for_subject = candidate_list[i].get("safe", []) 
                    
#                     assigned_from_safe_count = 0
#                     for safe_control_id_orig in current_safe_controls_for_subject:
#                         # Even though 'safe' means non-competitive, we check `not in used_control_indices`
#                         # for robustness, e.g., if a previous (hypothetical) step already used it.
#                         if safe_control_id_orig not in used_control_indices:
#                             match_dict.setdefault(exposed_id, []).append(safe_control_id_orig)
#                             used_control_indices.add(safe_control_id_orig)
#                             matches_needed[exposed_id] -= 1
#                             assigned_from_safe_count += 1
#                             logger.debug(f"Assigned safe match for {exposed_id}: {safe_control_id_orig}. Needs: {matches_needed[exposed_id]}")
#                             # Stop assigning safe if target n_neighbors reached for this individual
#                             if assigned_from_safe_count >= n_neighbors:
#                                 break
#                     if matches_needed[exposed_id] <= 0:
#                         logger.debug(f"Exposed subject {exposed_id} fully matched by safe controls ({len(match_dict.get(exposed_id, []))}/{n_neighbors}).")

#                 # Filter to only include individuals who still need more matches after safe assignments
#                 active_limited_group_indices = [
#                     i for i in limited_group
#                     if matches_needed[exposed_indices[i]] > 0
#                 ]
                
#                 # If all limited subjects were satisfied by safe matches, skip competitive/fuzzy phase
#                 if not active_limited_group_indices:
#                     logger.debug("All limited subjects satisfied by safe matches. Skipping competitive/fuzzy phase.")
#                 else:
#                     # --- PHASE 2: Iterative Greedy for Competitive/Fuzzy Controls ---
#                     logger.debug("Phase 2: Starting iterative greedy for competitive/fuzzy controls.")
#                     iteration_count = 0
#                     # Safety break: roughly twice the total matches still needed
#                     max_iterations = sum(matches_needed[exposed_indices[idx]] for idx in active_limited_group_indices) * 2 

#                     # Fit one NearestNeighbors tree on the full control set.
#                     # This is the most significant efficiency gain by avoiding repeated `nbrs.fit` inside the loop.
#                     # Use the maximum radius needed (fuzzy_threshold_limit) to get all potential candidates.
#                     global_nbrs_tree = NearestNeighbors(
#                         radius=max(threshold, fuzzy_threshold_limit),
#                         metric=distance_metric
#                     )
#                     global_nbrs_tree.fit(X_control)

#                     while active_limited_group_indices and iteration_count < max_iterations:
#                         iteration_count += 1
                        
#                         # Re-sort active limited group by current "neediness" (fewest *available* options)
#                         # This key filters dynamically against `used_control_indices`.
#                         greedy_sorted_current = sorted(
#                             active_limited_group_indices,
#                             key=lambda i: (
#                                 # Primary sort: least number of matches already obtained by this individual (most "needed")
#                                 len(match_dict.get(exposed_indices[i], [])),
#                                 # Secondary sort: fewest available safe controls (should mostly be 0 for active_limited_group_indices here)
#                                 #len([c for c in candidate_list[i].get("safe", []) if c not in used_control_indices]),
#                                 # Tertiary sort: fewest available competitive controls
#                                 len([c for c in candidate_list[i].get("competitive", []) if c not in used_control_indices]),
#                                 # Quaternary sort: fewest available fuzzy controls
#                                 len([c for c in candidate_list[i].get("fuzzy", []) if c not in used_control_indices])
#                             )
#                         )

#                         made_progress_in_iteration = False
#                         for i in greedy_sorted_current:
#                             exposed_id = exposed_indices[i]
#                             # Skip if this individual is already fully matched
#                             if matches_needed[exposed_id] <= 0:
#                                 continue

#                             X_exposed_single = X_exposed[i].reshape(1, -1)
                            
#                             # Query the pre-fitted global tree to find all potential candidates for this individual
#                             distances_raw, indices_raw = global_nbrs_tree.radius_neighbors(
#                                 X_exposed_single,
#                                 radius=fuzzy_threshold_limit, # Search up to the widest allowed radius
#                                 return_distance=True
#                             )
                            
#                             distances_for_subject = distances_raw[0]
#                             indices_in_X_control_for_subject = indices_raw[0]

#                             potential_matches_for_subject = []
#                             # Iterate through all discovered neighbors, apply thresholds, and penalize fuzzy
#                             for k_idx, internal_c_idx in enumerate(indices_in_X_control_for_subject):
#                                 original_control_id = control_indices[internal_c_idx]
#                                 current_distance = distances_for_subject[k_idx]

#                                 # Skip if this control has already been used by anyone
#                                 if original_control_id in used_control_indices:
#                                     continue

#                                 # Prioritize 'competitive' (within 'threshold') over 'fuzzy' (within 'fuzzy_threshold_limit')
#                                 if current_distance <= threshold:
#                                     potential_matches_for_subject.append((current_distance, original_control_id, "competitive"))
#                                 elif fuzzy_threshold and current_distance <= fuzzy_threshold_limit:
#                                     # Apply a large penalty to fuzzy matches to ensure they are chosen only if
#                                     # competitive matches are exhausted. The value `threshold * 1000` makes
#                                     # any penalized fuzzy distance higher than any non-penalized competitive distance.
#                                     penalized_dist = current_distance + (threshold * 1000.0) 
#                                     potential_matches_for_subject.append((penalized_dist, original_control_id, "fuzzy"))
                            
#                             potential_matches_for_subject.sort(key=lambda x: x[0]) # Sort by effective distance

#                             # Attempt to assign one best available match
#                             for penalized_dist, control_id_orig, match_type in potential_matches_for_subject:
#                                 # Final check if control is still available before assignment
#                                 if control_id_orig not in used_control_indices:
#                                     match_dict.setdefault(exposed_id, []).append(control_id_orig)
#                                     used_control_indices.add(control_id_orig)
#                                     matches_needed[exposed_id] -= 1
#                                     made_progress_in_iteration = True
#                                     logger.debug(
#                                         f"Greedy (limited) matched {exposed_id} with {control_id_orig} "
#                                         f"(type: {match_type}, effective_dist: {penalized_dist:.2f}, "
#                                         f"needs: {matches_needed[exposed_id]})"
#                                     )
#                                     break # Assigned one match, move to the next individual in this round

#                         # If no individual made progress in this entire iteration, stop trying for the limited group
#                         if not made_progress_in_iteration:
#                             logger.debug("No new matches found in this iteration for the limited group. Stopping greedy allocation.")
#                             break
                        
#                         # Update the list of active individuals who still need matches
#                         active_limited_group_indices = [
#                             i for i in active_limited_group_indices
#                             if matches_needed[exposed_indices[i]] > 0
#                         ]
#             ### Uncomment <<<
            
            
            
#             # Next, process the regular group using the global optimal matching routine.
#             if regular_group:
#                 logger.debug(f"Processing regular group with {len(regular_group)} subjects.")
                
#                 cand_list_regular = []
#                 for i in regular_group:
#                     cand = candidate_list[i]
#                     # Filter safe and competitive candidates based on used_control_indices.
#                     updated_safe = [c for c in cand.get("safe", []) if control_indices[c] not in used_control_indices]
#                     updated_competitive = [c for c in cand.get("competitive", []) if control_indices[c] not in used_control_indices]
#                     updated_cand = {"safe": updated_safe, "competitive": updated_competitive}
#                     cand_list_regular.append(updated_cand)
                
#                 # Now subset X_exposed and exposed_indices for the regular group.
#                 X_exposed_regular = X_exposed[regular_group]
#                 exposed_indices_regular = [exposed_indices[i] for i in regular_group]
                
#                 match_dict = match_optimal(X_exposed_regular, X_control, cand_list_regular, threshold, distance_metric,
#                                            n_neighbors, match_dict, exposed_indices_regular)
    
#     logger.info(f"Completed global optimal matching with {n_neighbors} neighbor(s) successfully.")
#     return match_dict


# >>>
def match_global_optimal(
        X_exposed, 
        X_control, 
        exposed_indices, 
        control_indices,
        threshold, 
        n_neighbors, 
        k_candidates,
        safe_matches,
        distance_metric, 
        custom_distance_fn,
        replacement,
        debug=False,
        competitive_match=False,
        fuzzy_threshold=False,
        fuzzy_threshold_limit=None,
        **kwargs
        ):
    """
    Perform global optimal matching using the Hungarian algorithm (linear_sum_assignment).
    If 'competitive_match' is True, then subjects with limited safe controls 
    (i.e. fewer than safe_matches) are handled via greedy matching first (in order of lowest
    number of safe controls and, in case of ties, lowest number of competitive controls). The remaining 
    subjects are then matched using the standard global optimal routine.
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info(f"Starting global optimal matching with {n_neighbors} neighbor(s).")
    if competitive_match:
        logger.info("Competitive matching is enabled. Subjects with limited safe controls will be handled separately and first.")

    # fix input parameters if fuzzy_threshold is True and I screwed up the defaults.
    if fuzzy_threshold_limit is None:
        fuzzy_threshold_limit = threshold
    
    if replacement:
        match_dict = {}
        # Use the full cost matrix (can be optimized similarly if needed)
        cost_matrix = compute_distance_matrix(X_exposed, X_control, distance_metric, custom_distance_fn)
        cost_matrix_masked = cost_matrix.copy()
        cost_matrix_masked[cost_matrix_masked > threshold] = np.inf
        for i, row in enumerate(cost_matrix_masked):
            exposed_id = exposed_indices[i]
            candidate_control_idxs = np.argsort(row)
            valid_candidates = [j for j in candidate_control_idxs if row[j] != np.inf]
            if len(valid_candidates) == 0:
                logger.warning(f"Exposed subject {exposed_id} has no valid matches under the threshold.")
                continue
            selected = valid_candidates[:n_neighbors]
            matched_controls = [control_indices[j] for j in selected]
            match_dict[exposed_id] = matched_controls
            logger.debug(f"Exposed subject {exposed_id} matched with controls (with replacement): {matched_controls}")
    
    else:
        # Without replacement:
        # Pre-filter candidate controls.
        candidate_list = check_candidate_availability(
            X_exposed=X_exposed, 
            X_control=X_control, 
            threshold=threshold, 
            k_candidates=k_candidates,
            distance_metric=distance_metric, 
            safe_matches=safe_matches,
            fuzzy_threshold=fuzzy_threshold,
            fuzzy_threshold_limit=fuzzy_threshold_limit
        )
        
        # If competitive_match is not requested, proceed with standard global optimal matching.
        if not competitive_match:
            match_dict = {}
            match_dict = match_optimal(X_exposed, X_control, candidate_list, threshold, distance_metric,
                                       n_neighbors, match_dict, exposed_indices)
                
        else:
            # Competitive matching is requested:
            # Partition subjects into two groups: limited subjects get greedy matching first.
            limited_group = []
            regular_group = []
            for i, cand in enumerate(candidate_list):
                # Define "limited" as having fewer safe candidates than safe_matches.
                if len(cand.get("safe", [])) < safe_matches:
                    limited_group.append(i)
                    # see if they also have a limited number of competitive candidates.
                    if len(cand.get("competitive", [])) < n_neighbors:
                        logger.warning(f"Exposed subject {exposed_indices[i]} is in the limited group with {len(cand.get('safe', []))} safe and {len(cand.get('competitive', []))} competitive controls.")
                        logger.warning(f"  Consider relaxing the threshold or increasing the number of safe matches.")
                else:
                    regular_group.append(i)
            
            match_dict = {}
            used_control_indices = set()
            
            # First, handle the limited group using greedy matching.
            if limited_group:
                # Keep track of how many matches each limited individual still needs
                matches_needed = {exposed_indices[i]: n_neighbors for i in limited_group}

                # --- PHASE 1: Assign Safe Controls for the Limited Group ---
                logger.debug("Phase 1: Assigning safe controls for limited group.")
                for i in limited_group:
                    exposed_id = exposed_indices[i]
                    # Get safe control indices from candidate list
                    safe_control_indices = candidate_list[i].get("safe", [])
                    
                    assigned_from_safe_count = 0
                    for safe_idx in safe_control_indices:
                        # Convert index to original control ID
                        safe_control_id = control_indices[safe_idx]
                        
                        if safe_control_id not in used_control_indices:
                            match_dict.setdefault(exposed_id, []).append(safe_control_id)
                            used_control_indices.add(safe_control_id)
                            matches_needed[exposed_id] -= 1
                            assigned_from_safe_count += 1
                            logger.debug(f"Assigned safe match for {exposed_id}: {safe_control_id}. Needs: {matches_needed[exposed_id]}")
                            
                            if assigned_from_safe_count >= n_neighbors:
                                break
                                
                    if matches_needed[exposed_id] <= 0:
                        logger.debug(f"Exposed subject {exposed_id} fully matched by safe controls ({len(match_dict.get(exposed_id, []))}/{n_neighbors}).")

                # Filter to only include individuals who still need more matches after safe assignments
                active_limited_group_indices = [
                    i for i in limited_group
                    if matches_needed[exposed_indices[i]] > 0
                ]
                
                # --- PHASE 2: Iterative Greedy for Competitive/Fuzzy Controls ---
                if active_limited_group_indices:
                    logger.debug("Phase 2: Starting iterative greedy for competitive/fuzzy controls.")
                    iteration_count = 0
                    max_iterations = sum(matches_needed[exposed_indices[idx]] for idx in active_limited_group_indices) * 2 

                    # Pre-fit NearestNeighbors for efficiency
                    search_radius = fuzzy_threshold_limit if fuzzy_threshold else threshold
                    global_nbrs_tree = NearestNeighbors(
                        radius=search_radius,
                        metric=distance_metric
                    )
                    global_nbrs_tree.fit(X_control)

                    while active_limited_group_indices and iteration_count < max_iterations:
                        iteration_count += 1
                        
                        # Sort by neediness - prioritize those with fewer current matches
                        greedy_sorted_current = sorted(
                            active_limited_group_indices,
                            key=lambda i: (
                                # Primary: fewest current matches (most needy)
                                len(match_dict.get(exposed_indices[i], [])),
                                # Secondary: fewest available competitive controls
                                len([idx for idx in candidate_list[i].get("competitive", []) 
                                     if control_indices[idx] not in used_control_indices]),
                                # Tertiary: fewest available fuzzy controls
                                len([idx for idx in candidate_list[i].get("fuzzy", []) 
                                     if control_indices[idx] not in used_control_indices])
                            )
                        )

                        made_progress_in_iteration = False
                        for i in greedy_sorted_current:
                            exposed_id = exposed_indices[i]
                            if matches_needed[exposed_id] <= 0:
                                continue

                            X_exposed_single = X_exposed[i].reshape(1, -1)
                            
                            # Find all potential matches within search radius
                            distances_raw, indices_raw = global_nbrs_tree.radius_neighbors(
                                X_exposed_single,
                                radius=search_radius,
                                return_distance=True
                            )
                            
                            distances_for_subject = distances_raw[0]
                            control_indices_for_subject = indices_raw[0]

                            potential_matches = []
                            
                            # Process each potential match and categorize
                            for k_idx, control_array_idx in enumerate(control_indices_for_subject):
                                original_control_id = control_indices[control_array_idx]
                                current_distance = distances_for_subject[k_idx]

                                # Skip if already used
                                if original_control_id in used_control_indices:
                                    continue

                                # Categorize and potentially penalize fuzzy matches
                                if current_distance <= threshold:
                                    potential_matches.append((current_distance, original_control_id, "competitive"))
                                elif fuzzy_threshold and current_distance <= fuzzy_threshold_limit:
                                    # Apply penalty to fuzzy matches to prefer competitive ones
                                    # penalized_distance = current_distance + (threshold * 1.0)  # Reduced penalty
                                    potential_matches.append((current_distance, original_control_id, "fuzzy"))
                            
                            # Sort by effective distance and try to assign the best available match
                            potential_matches.sort(key=lambda x: x[0])

                            for penalized_dist, control_id, match_type in potential_matches:
                                if control_id not in used_control_indices:
                                    match_dict.setdefault(exposed_id, []).append(control_id)
                                    used_control_indices.add(control_id)
                                    matches_needed[exposed_id] -= 1
                                    made_progress_in_iteration = True
                                    logger.debug(
                                        f"Greedy matched {exposed_id} with {control_id} "
                                        f"(type: {match_type}, dist: {penalized_dist:.3f}, "
                                        f"needs: {matches_needed[exposed_id]})"
                                    )
                                    break

                        # Stop if no progress made
                        if not made_progress_in_iteration:
                            logger.debug("No new matches found in this iteration. Stopping greedy allocation.")
                            break
                        
                        # Update active list
                        active_limited_group_indices = [
                            i for i in active_limited_group_indices
                            if matches_needed[exposed_indices[i]] > 0
                        ]
            
            # Process the regular group using global optimal matching
            if regular_group:
                logger.debug(f"Processing regular group with {len(regular_group)} subjects.")
                
                # Create filtered candidate list for regular group
                cand_list_regular = []
                for i in regular_group:
                    cand = candidate_list[i]
                    # Filter candidates based on what's still available
                    updated_safe = [idx for idx in cand.get("safe", []) 
                                   if control_indices[idx] not in used_control_indices]
                    updated_competitive = [idx for idx in cand.get("competitive", []) 
                                          if control_indices[idx] not in used_control_indices]
                    updated_fuzzy = [idx for idx in cand.get("fuzzy", []) 
                                    if control_indices[idx] not in used_control_indices]
                    
                    updated_cand = {
                        "safe": updated_safe, 
                        "competitive": updated_competitive,
                        "fuzzy": updated_fuzzy
                    }
                    cand_list_regular.append(updated_cand)
                
                # Prepare data for regular group
                X_exposed_regular = X_exposed[regular_group]
                exposed_indices_regular = [exposed_indices[i] for i in regular_group]
                
                # Run optimal matching on regular group
                match_dict = match_optimal(
                    X_exposed_regular, 
                    X_control, 
                    cand_list_regular, 
                    threshold, 
                    distance_metric,
                    n_neighbors, 
                    match_dict, 
                    exposed_indices_regular
                )
    
    logger.info(f"Completed global optimal matching with {n_neighbors} neighbor(s) successfully.")
    return match_dict


# <<<


# -------------------------------------------------------------------------
# Simple Matching Function
# -------------------------------------------------------------------------
def simple_matching(
    df,
    exposure_status="is_hht",
    propensity_col="propensity_score",
    threshold=0.2,
    n_neighbors=1,
    # verbose=True,
    patient_id="patientdurablekey",
    **kwargs
):
    """
    Performs simplified propensity score matching using only the propensity score.

    The function splits the data into exposed and control groups, then uses a
    radius-based nearest neighbor search (using only the propensity score) to find
    controls within a specified threshold. Each exposed subject is assigned a matching group
    along with the identifiers of matched controls (controls are used only once).

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame which must include the binary exposure indicator and propensity score.
    exposure_status : str, default "is_hht"
        Name of the binary exposure indicator column.
    propensity_col : str, default "propensity_score"
        Name of the column containing propensity scores.
    threshold : float, default 0.05
        Maximum allowed Euclidean distance (on the propensity score scale) for a match.
    n_neighbors : int, default 1
        Number of controls to match to each exposed subject.
    verbose : bool, default True
        If True, logs summary information.
    patient_id : str, default "patientdurablekey"
        Column representing the unique subject identifier. This column is dropped from the matching 
        features to avoid leakage.

    Returns
    -------
    matched_data : pandas.DataFrame
        DataFrame containing the exposed subjects (each appears only once) and their matched controls,
        with an additional column 'match_group' indicating the matching group. An 'n_matches' column is added
        for the exposed subjects (set to None for controls).
    """

    debug = kwargs.get('debug', False)

    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    logger.info("Starting simple (propensity score only) matching.")
    
    # Create working copy and drop the patient_id from the matching features.
    df_match = df.copy()

    # Check for required columns.
    for col in [exposure_status, propensity_col]:
        if col not in df_match.columns:
            raise ValueError(f"Missing required column '{col}' in the input dataframe.")

    df_match = df_match.drop(columns=[patient_id], errors='ignore')
    
    # Extract the propensity scores.
    propensity_scores = df_match[propensity_col].values.reshape(-1, 1)
    
    print(f"Exposure Status: {exposure_status}")
    
    # Split indices for exposed and control subjects.
    exposed_indices = df.index[df[exposure_status] == 1].tolist()
    control_indices = df.index[df[exposure_status] == 0].tolist()
    
    if len(exposed_indices) == 0 or len(control_indices) == 0:
        raise ValueError("There must be at least one exposed and one control subject in the data.")
    
    X_exposed = propensity_scores[exposed_indices]
    X_control = propensity_scores[control_indices]
    
    # Fit a NearestNeighbors model on the control propensity scores.
    nbrs = NearestNeighbors(radius=threshold, n_neighbors=n_neighbors, metric='euclidean')
    nbrs.fit(X_control)
    distances_all, indices_all = nbrs.radius_neighbors(X_exposed)
    
    # A dictionary to map each exposed subject index to its matched control indices.
    match_dict = {}
    used_control_indices = set()
    
    for i, (dists, neighbor_idxs) in enumerate(zip(distances_all, indices_all)):
        exposed_id = exposed_indices[i]
        if len(neighbor_idxs) == 0:
            logger.debug(f"Exposed subject {exposed_id} has no neighbors within threshold.")
            continue

        # Sort neighbors by distance.
        sorted_order = np.argsort(dists)
        sorted_neighbor_idxs = neighbor_idxs[sorted_order]
        
        # Filter neighbors for control subjects that have not yet been used.
        available_controls = [
            control_indices[j]
            for j in sorted_neighbor_idxs
            if control_indices[j] not in used_control_indices
        ]
        if available_controls:
            selected_controls = available_controls[:n_neighbors]
            match_dict[exposed_id] = selected_controls
            used_control_indices.update(selected_controls)
            logger.debug(f"Exposed subject {exposed_id} matched with controls: {selected_controls}")
        else:
            logger.debug(f"Exposed subject {exposed_id} has neighbors but none available (already used).")
    
    if len(match_dict) == 0:
        raise ValueError("No matches found. Consider relaxing the threshold.")
    
    # Build the final matched dataframe using the original dataframe (retaining all columns).
    exposed_list = []
    controls_list = []
    for group_id, (exposed_id, control_ids) in enumerate(match_dict.items()):
        # Exposed row with match group information.
        exposed_df = df.loc[[exposed_id]].copy()
        exposed_df["match_group"] = group_id
        exposed_df["n_matches"] = len(control_ids)
        exposed_list.append(exposed_df)
        
        # Matched control rows with the same match group.
        controls_df = df.loc[control_ids].copy()
        controls_df["match_group"] = group_id
        controls_df["n_matches"] = None
        controls_list.append(controls_df)
    
    matched_exposed = pd.concat(exposed_list)
    matched_controls = pd.concat(controls_list)
    
    matched_data = pd.concat([matched_exposed, matched_controls]).reset_index(drop=True)
    
    verbose_matching_results(matched_data, df, exposure_status, patient_id)

    logger.info("Simple matching completed successfully.")
    
    return matched_data



# ---------------------------
# Multicovariate Matching Function
# ---------------------------
def multi_covariate_adjusted_matching(
    df,
    exposure_status,
    propensity_col,
    patient_id,
    threshold,
    n_neighbors,
    k_candidates,
    global_optimal,
    replacement,
    features_numeric,
    features_categorical,
    **kwargs,
):
    """
    Improved multi-covariate adjusted matching integrating the propensity score with additional
    covariates and allowing for global optimal matching (with or without replacement) and custom distance metrics.

    The returned matched dataframe contains the match_group and n_matches columns.
    """
    # check if the required columns are present in the DataFrame.
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input df must be a pandas DataFrame.")
    if exposure_status not in df.columns:
        raise ValueError(f"Exposure status column '{exposure_status}' not found in the DataFrame.")
    if propensity_col not in df.columns:
        raise ValueError(f"Propensity score column '{propensity_col}' not found in the DataFrame.")
    if patient_id not in df.columns:
        raise ValueError(f"Subject ID column '{patient_id}' not found in the DataFrame.")
    if not isinstance(threshold, (int, float)):
        raise ValueError("Threshold must be a numeric value (int or float).")
    if not isinstance(n_neighbors, int) or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer.")
    if not isinstance(k_candidates, int) or k_candidates < 1:
        raise ValueError("k_candidates must be a positive integer.")
    if not isinstance(global_optimal, bool):
        raise ValueError("global_optimal must be a boolean value.")
    if not isinstance(replacement, bool):
        raise ValueError("replacement must be a boolean value.")
    if not isinstance(features_numeric, (list, type(None))):
        raise ValueError("features_numeric must be a list of numeric feature names or None.")
    if not isinstance(features_categorical, (list, type(None))):
        raise ValueError("features_categorical must be a list of categorical feature names or None.")
    
    # Check if kwargs is a dictionary.
    if not isinstance(kwargs, dict):
        raise ValueError("kwargs must be a dictionary containing optional parameters.")

    # assign the kwargs to local variables with defaults.
    weight_propensity = kwargs.get('weight_propensity', 1.0)
    weight_numeric = kwargs.get('weight_numeric', 1.0)
    debug = kwargs.get('debug', False)
    distance_metric = kwargs.get('distance_metric', 'euclidean')
    custom_distance_fn = kwargs.get('custom_distance_fn', None)
    pca_filter = kwargs.get('pca_filter', False)
    safe_matches = kwargs.get('safe_matches', n_neighbors)
    competitive_match = kwargs.get('competitive_match', False)

    fuzzy_threshold = kwargs.get('fuzzy_threshold', False)
    fuzzy_threshold_limit = kwargs.get('fuzzy_threshold_limit', None)
    if fuzzy_threshold:
        if 'fuzzy_threshold_limit' in kwargs:
            if not isinstance(kwargs['fuzzy_threshold_limit'], (int, float)):
                raise ValueError("fuzzy_threshold_limit must be a numeric value (int or float).")
            fuzzy_threshold_limit = kwargs.get('fuzzy_threshold_limit', threshold)
        else:
            logger.warning("fuzzy_threshold is True but no fuzzy_threshold_limit provided. Using default threshold.")
            fuzzy_threshold_limit = threshold
    if not fuzzy_threshold and 'fuzzy_threshold_limit' in kwargs:
        raise ValueError("fuzzy_threshold_limit provided but fuzzy_threshold is False.")


    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Validate required columns.
    validate_input_columns(df, [exposure_status, propensity_col])

    # Compute weighted features.
    df_match, X_combined, _ = compute_weighted_features(
        df,
        exposure_status,
        propensity_col,
        weight_propensity,
        weight_numeric,
        patient_id
    )

    # Extract indices for exposed and control subjects.
    exposed_indices, control_indices = extract_exposure_indices(df_match, exposure_status)

    # Split the weighted feature matrix.
    X_exposed = X_combined[exposed_indices]
    X_control = X_combined[control_indices]


    if pca_filter:
        logger.info("Applying PCA filter to reduce dimensionality.")
        logger.debug(f"Initial dimensions: Exposed {X_exposed.shape}, Control {X_control.shape}")
        try:
            pca = plot_pca_threshold(
                X_exposed,
                max_components=50, 
                sample_fraction=1.0, 
                incremental=False,
                plot=False,
                return_pca=True,
                batch_size=None,
                variance_threshold=0.95,
                random_state=404
            )
        except Exception as e:
            logger.error("PCA fitting failed: " + str(e))
            raise
        if pca is None:
            raise ValueError("plot_pca_threshold returned None. Check the function implementation or input data.")
        else: 
            pca_exposed, cutoff_idx = pca
            X_exposed = pca_exposed.transform(X_exposed)[:, :cutoff_idx]
            X_control = pca_exposed.transform(X_control)[:, :cutoff_idx]
        logger.info(f"PCA filter applied successfully.")
        logger.debug(f"Reduced dimensions: Exposed {X_exposed.shape}, Control {X_control.shape}")


    # Perform matching according to the selected method.
    if global_optimal:
        match_dict = match_global_optimal(
                X_exposed=X_exposed,
                X_control=X_control,
                exposed_indices=exposed_indices,
                control_indices=control_indices,
                threshold=threshold,
                n_neighbors=n_neighbors,
                k_candidates=k_candidates,
                safe_matches=safe_matches,
                distance_metric=distance_metric,
                custom_distance_fn=custom_distance_fn,
                replacement=replacement,
                debug=debug,
                competitive_match=competitive_match,
                fuzzy_threshold=fuzzy_threshold,
                fuzzy_threshold_limit=fuzzy_threshold_limit
            )
    else:
        match_dict = match_greedy(
            X_exposed,
            X_control,
            exposed_indices,
            control_indices,
            threshold,
            n_neighbors,
            debug=debug
        )

    # Build the matched dataframe with match groups and added columns.
    matched_data = build_matched_dataframe(df, match_dict)

    try:
        verbose_matching_results(matched_data, df, exposure_status, patient_id=patient_id)
    except Exception as e:
        logger.info("Verbose matching results function is not available or failed: " + str(e))
    # if debug:
    #     try:
    #         run_diagnostics(matched_data, exposure_status, features_numeric, features_categorical)
    #     except Exception as e:
    #         logger.info("Diagnostics functions are not available or failed: " + str(e))

    return matched_data










# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def verbose_matching_results(matched_data, cohort, exposure_status, patient_id='patientdurablekey'):
    logger.info("Matching Summary:")
    logger.info("=" * 80)

    exposed = matched_data[matched_data[exposure_status] == 1]
    control = matched_data[matched_data[exposure_status] == 0]
    cohort_pat_set = set(cohort[cohort[exposure_status] == 1][patient_id])

    # Count unique subjects in each group.
    n_exposed = exposed[patient_id].nunique()
    n_control = control[patient_id].nunique()

    logger.info(f" Unique exposed subjects: {n_exposed}")
    logger.info(f" Unique control subjects: {n_control}")
    logger.info(f" Total subjects in cohort: {n_exposed + n_control}")
    logger.info("-" * 80)

    logger.info(f" Matching ratio (control:exposed): {n_control/n_exposed:.2f}:1")
    logger.info("-" * 80)

    # Check for duplicates.
    exposed_dups = exposed[patient_id].duplicated().sum()
    control_dups = control[patient_id].duplicated().sum()

    logger.info(f" Duplicate exposed subjects: {exposed_dups}")
    logger.info(f" Duplicate control subjects: {control_dups}")
    logger.info("-" * 80)

    # Check for exposed subjects that were not matched.
    not_matched = cohort_pat_set.difference(set(matched_data[matched_data[exposure_status] == 1][patient_id]))
    logger.info(f" Number of exposed subjects not matched: {len(not_matched)}")
    logger.info("-" * 80)

    counts = Counter(exposed['n_matches'])
    counts_sorted = dict(sorted(counts.items(), reverse=True))

    if len(counts_sorted) <= 10:
        logger.info(f" Distribution of matching: {counts_sorted}")
    else:
        first_10 = dict(list(counts_sorted.items())[:10])
        logger.info(f" Distribution of matching: {first_10} ...")



def _calculate_smd_numeric(group1, group2, var_list):
    """
    Calculate the standardized mean differences (SMD) for numeric variables.
    """
    smds = {}
    for var in var_list:
        mean1 = group1[var].mean()
        mean2 = group2[var].mean()
        std1 = group1[var].std()
        std2 = group2[var].std()

        pooled_std = np.sqrt((std1**2 + std2**2) / 2)
        smd = abs(mean1 - mean2) / pooled_std if pooled_std != 0 else 0.0
        smds[var] = smd
    return smds


def _calculate_smd_categorical(prop_exposed, prop_control, covariate):
    """
    Calculate the SMD for categorical variables based on group proportions.
    """
    smds = {}
    for category in prop_exposed.index:
        prop_exposed_val = prop_exposed.get(category, 0)
        prop_control_val = prop_control.get(category, 0)
        pooled_var = (prop_exposed_val * (1 - prop_exposed_val) + prop_control_val * (1 - prop_control_val)) / 2
        smd = abs(prop_exposed_val - prop_control_val) / np.sqrt(pooled_var) if pooled_var > 0 else 0.0
        smds[category] = smd
    return smds


def summary_stats_table(data, features_numeric, features_categorical, exposure_col, smd_threshold=0.1):
    """
    Build a summary table of matching diagnostics.
    
    For numeric covariates, includes means, standard deviations, and SMDs.
    For categorical covariates, includes proportions (as percentages) and SMDs.
    """
    rows = []

    # Numeric covariates.
    for feature in features_numeric:
        mean_exposed = data[data[exposure_col] == 1][feature].mean()
        std_exposed = data[data[exposure_col] == 1][feature].std()
        mean_control = data[data[exposure_col] == 0][feature].mean()
        std_control = data[data[exposure_col] == 0][feature].std()

        smd = _calculate_smd_numeric(data[data[exposure_col] == 1],
                            data[data[exposure_col] == 0],
                            [feature])[feature]

        rows.append({
            'Feature': feature,
            'Mean_exposed': mean_exposed,
            'Std_exposed': std_exposed,
            'Mean_Control': mean_control,
            'Std_Control': std_control,
            'SMD': smd
        })

    # Categorical covariates.
    for feature in features_categorical:
        prop_exposed = data[data[exposure_col] == 1][feature].value_counts(normalize=True)
        prop_control = data[data[exposure_col] == 0][feature].value_counts(normalize=True)

        smds = _calculate_smd_categorical(prop_exposed, prop_control, feature)

        for category in prop_exposed.index:
            rows.append({
                'Feature': f"{feature}_{category[:10]}", # Shorten category name for display.
                'Mean_exposed': prop_exposed[category] * 100,
                'Std_exposed': np.nan,  # Not applicable for proportions.
                'Mean_Control': prop_control.get(category, 0) * 100,
                'Std_Control': np.nan,  # Not applicable for proportions.
                'SMD': smds.get(category, np.nan)
            })

    summary_df = pd.DataFrame(rows)
    summary_df['SMD Result'] = summary_df['SMD'].apply(lambda x: "OK" if x < smd_threshold else "BAD")
    return np.round(summary_df, 3)


def matching_diagnostics(matched_data, exposure_status, features_numeric, features_categorical):
    """
    Run balance diagnostics as a summary table inline with logger.
    """

    if features_numeric is None or features_categorical is None:
        logger.warning("Diagnostics enabled but features_numeric and/or features_categorical not provided. Skipping detailed balance diagnostics.")
    else:
        summary_table = summary_stats_table(
            data=matched_data,
            features_numeric=features_numeric,
            features_categorical=features_categorical,
            exposure_col=exposure_status,
            smd_threshold=0.1
        )
        if isinstance(summary_table, pd.DataFrame):
            logger.info("Summary Stats Table After Matching:\n" + summary_table.to_string(index=False))
        else:
            logger.error("Summary Stats Table is not a DataFrame. Check the output of summary_stats_table.")
        plot_numeric_feature_balance(matched_data, features_numeric, exposure_status)
