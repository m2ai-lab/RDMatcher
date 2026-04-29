import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix
from collections import Counter
from typing import List, Union, Optional, Dict
import logging

# from .logger import rdlogger
from .utils import hide_columns
from .plot import plot_feature_balance

# logger = rdlogger(__name__, level="INFO")
logger = logging.getLogger('rdmatcher.matching')

# =============================================================================
# 1. Feature Preparation (Stateless)
# =============================================================================

def compute_weighted_features(df, exposure_status, weight_numeric, patient_id, propensity_col=None, weight_propensity=0.0):
    """
    Prepares the feature matrix by handling propensity scores and numeric weights.
    Returns:
        df_match: Copy of df with patient_id hidden.
        X_combined: The weighted numpy array used for distance calculations.
        covariate_cols: List of columns used (excluding propensity/exposure).
    """
    df_match = df.copy() 
    df_match = hide_columns(df_match, [patient_id])


    
    # 1. Handle Propensity
    if propensity_col and weight_propensity > 0:
        if propensity_col not in df_match.columns:
            raise ValueError(f"Propensity column '{propensity_col}' missing.")
        categorical_cols = df_match.select_dtypes(include=['object', 'category']).columns.tolist()
        if any(col in categorical_cols for col in df_match.columns if col not in [exposure_status, propensity_col]):
            logger.warning("Categorical columns detected in the feature set. Ensure proper preprocessing before matching.")
            logger.warning(f"Proceeding with only numeric columns: {df_match.select_dtypes(include=['number']).columns.tolist()}")
            df_match = df_match.select_dtypes(include=['number', 'float', 'int'])
        prop_scores = df_match[propensity_col].values.reshape(-1, 1) * weight_propensity
        cov_cols = [c for c in df_match.columns if c not in [exposure_status, propensity_col]]
    else:
        prop_scores = np.empty((len(df_match), 0))
        cov_cols = [c for c in df_match.columns if c != exposure_status]

    # 2. Handle Numeric Covariates
    if weight_numeric != 1.0:
        logger.info(f"Applying weight {weight_numeric} to numeric covariates.")
        categorical_cols = df_match.select_dtypes(include=['object', 'category']).columns.tolist()
        if any(col in categorical_cols for col in df_match.columns if col not in [exposure_status, propensity_col]):
            logger.warning("Categorical columns detected in the feature set. Ensure proper preprocessing before matching.")
            logger.warning(f"Proceeding with only numeric columns: {df_match.select_dtypes(include=['number']).columns.tolist()}")
            df_match = df_match.select_dtypes(include=['number'])
        X_cov = df_match[cov_cols].to_numpy(dtype=float) * weight_numeric
    else:
        X_cov = df_match[cov_cols].to_numpy(dtype=float)

    # ensure robustness
    X_cov = np.nan_to_num(X_cov, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 3. Combine
    # Check if we have data in both arrays to avoid shape mismatch errors
    feature_names = []
    if propensity_col and weight_propensity > 0:
        feature_names.append(propensity_col)
    feature_names.extend(cov_cols)

    if prop_scores.size > 0 and X_cov.size > 0:
        X_combined = np.hstack([prop_scores, X_cov])
    elif prop_scores.size > 0:
        X_combined = prop_scores
    else:
        X_combined = X_cov

    return df_match, X_combined, cov_cols, feature_names
    

def extract_exposure_indices(df, exposure_col):
    """
    Returns numpy arrays of integer indices (0 to N) for exposed and control groups.
    """
    if exposure_col not in df.columns:
         raise ValueError(f"Exposure column '{exposure_col}' not found.")
         
    # Boolean mask is faster than .index lookup
    mask = (df[exposure_col] == 1).values
    exp_indices = np.where(mask)[0]
    ctrl_indices = np.where(~mask)[0]
    
    if len(exp_indices) == 0 or len(ctrl_indices) == 0:
        raise ValueError("Data must contain at least one exposed and one control subject.")
        
    return np.array(exp_indices), np.array(ctrl_indices)


# =============================================================================
# 2. Optimization Logic (The Math)
# =============================================================================

def solve_optimal_assignment(
    X_exposed_subset: Union[np.ndarray, pd.DataFrame],
    X_control: Union[np.ndarray, pd.DataFrame],
    candidate_lists: List[List[int]],
    threshold: float,
    metric: str,
    n_neighbors: int,
    all_control_indices: np.ndarray,
    gower_model=None,  # <--- MUST BE PASSED if metric='gower'
    precomputed: Optional[List[dict]] = None
):
    """
    Global optimal matching via Hungarian algorithm with Single-Allocation Memory Optimization.
    Compatible with custom GowerKNN.
    """

    # --- 1. Filter Invalid Rows (The "Desert" Problem) ---
    # Identify subjects with NO valid candidates
    valid_exposed_mask = [len(c) > 0 for c in candidate_lists]
    n_total = len(candidate_lists)
    n_valid = sum(valid_exposed_mask)
    n_dropped = n_total - n_valid
    
    # WARNING LOGIC
    if n_dropped > 0:
        logger.warning(
            f"Dropping {n_dropped} subjects from optimal matching because they have 0 valid candidates "
            f"within the threshold ({threshold}). Consider increasing threshold or relaxing candidate selection."
        )

    if n_valid == 0:
        return {}

    valid_indices = np.where(valid_exposed_mask)[0]
    
    # Subset inputs (Preserve DataFrame type for Gower)
    if isinstance(X_exposed_subset, pd.DataFrame):
        X_exposed_active = X_exposed_subset.iloc[valid_indices]
    else:
        X_exposed_active = X_exposed_subset[valid_indices]
        
    active_candidate_lists = [candidate_lists[i] for i in valid_indices]
    
    if precomputed:
        active_precomputed = [precomputed[i] for i in valid_indices]
    else:
        active_precomputed = None

    # --- 2. Prepare Candidates ---
    unique_candidates = sorted(list(set().union(*active_candidate_lists)))
    if not unique_candidates:
        return {}

    col_map = {orig: new for new, orig in enumerate(unique_candidates)}
    n_exp = len(X_exposed_active)
    n_cand = len(unique_candidates)

    rows, cols, data = [], [], []

    # --- 3. Compute Costs (Fast Path vs Fallback) ---
    
    # A) FAST PATH: Use Precomputed Distances (Standard)
    # This avoids recalculating Gower distances (which are expensive).
    if active_precomputed is not None:
        for i in range(n_exp):
            cands = active_candidate_lists[i]
            if not cands: continue
            
            # Map: Candidate Control Index -> Distance
            # neighbor_indices/distances come from your fast GowerKNN.kneighbors
            p = active_precomputed[i]
            local_map = dict(zip(p['neighbor_indices'].tolist(), p['neighbor_distances'].tolist()))
            
            for ctrl_idx in cands:
                d = local_map.get(ctrl_idx, None)
                # Note: Floating point precision issues can sometimes make 0.2000001 > 0.2
                # GowerKNN returns float32, so we trust it.
                if d is not None and d <= threshold:
                    rows.append(i)
                    cols.append(col_map[ctrl_idx])
                    data.append(d)

    # B) SLOW PATH: Recalculate Distances (Fallback)
    else:
        # Prepare Subset of Control Data
        if isinstance(X_control, pd.DataFrame):
            X_control_sub = X_control.iloc[unique_candidates]
        else:
            X_control_sub = X_control[unique_candidates]

        # Call the correct cdist function
        if metric == "gower":
            if gower_model is None:
                raise ValueError("metric='gower' requires passing 'gower_model' to solve_optimal_assignment")
            # Your custom GowerKNN.cdist handles DataFrames vs Arrays automatically
            dblock = gower_model.cdist(X_exposed_active, X_control_sub)
        else:
            # Standard Euclidean (expects numeric arrays)
            dblock = cdist(X_exposed_active, X_control_sub, metric=metric) # type: ignore

        # Fill sparse data
        for i in range(n_exp):
            cands = active_candidate_lists[i]
            if not cands: continue
            reduced_cols = [col_map[c] for c in cands if c in col_map]
            if not reduced_cols: continue
            
            d_row = dblock[i]
            for rc in reduced_cols:
                d = d_row[rc]
                if d <= threshold:
                    rows.append(i)
                    cols.append(rc)
                    data.append(d)

    if not rows:
        return {}

    # --- 4. Single-Allocation Matrix Construction (RAM Optimization) ---
    
    # Final dimensions:
    final_h = n_exp * n_neighbors
    final_w = n_cand + final_h 
    
    # MEMORY GUARDRAIL
    total_elements = final_h * final_w
    safety_limit = 500_000_000 # 2GB input -> ~8GB Peak RAM
    
    if total_elements > safety_limit:
        logger.warning(
            f"Dense matrix allocation too large: {final_h}x{final_w} ({total_elements/1e9:.2f} billion elements). "
            f"This requires ~{total_elements * 4 / 1e9:.2f} GB of RAM for the matrix, "
            f"and ~{(total_elements * 4 * 4) / 1e9:.2f} GB during processing.\n"
            "If you encounter MemoryError or a crash, consider reducing n_neighbors or using sparse mcf=True matching."
        )

    # 1. Allocate with Dummy Cost (1e6)
    logger.debug(f"Allocating cost matrix: {final_h}x{final_w} (Float32)")    
    augmented_cost = np.full((final_h, final_w), 1e6, dtype=np.float32)
    
    # 2. Build Small Dense Matrix
    cost_coo = coo_matrix((data, (rows, cols)), shape=(n_exp, n_cand))
    dense_small = np.full((n_exp, n_cand), np.inf, dtype=np.float32)
    dense_small[cost_coo.row, cost_coo.col] = cost_coo.data
    
    del cost_coo, rows, cols, data

    # 3. Fill Left Side
    if n_neighbors == 1:
        augmented_cost[:, :n_cand] = dense_small
    else:
        augmented_cost[:, :n_cand] = np.repeat(dense_small, n_neighbors, axis=0)
    
    del dense_small

    # --- 5. Solve ---
    r_ind, c_ind = linear_sum_assignment(augmented_cost)
    del augmented_cost

    # # --- 6. Map Back ---
    # matches = {}
    # for r, c in zip(r_ind, c_ind):
    #     if c < n_cand:
    #         active_row_idx = r // n_neighbors
    #         original_exp_idx = valid_indices[active_row_idx] # Map to original index
    #         control_id = all_control_indices[unique_candidates[c]]
    #         matches.setdefault(original_exp_idx, []).append(control_id)
    # --- 6. Map Back (fixed) ---
    matches = {}
    for r, c in zip(r_ind, c_ind):
        if c < n_cand:
            # compute row within the *active* (pruned) exposed block
            active_row_idx = r // n_neighbors   # 0..(n_exp-1)
            # map that active row back to index *within the input candidate_lists*
            original_subset_idx = valid_indices[active_row_idx]  # index into passed candidate_lists (0..len(candidate_lists)-1)
            # map candidate column back to original control index in full dataset
            control_id = all_control_indices[unique_candidates[c]]
            matches.setdefault(int(original_subset_idx), []).append(int(control_id))

    return matches

# =============================================================================
# Solve Sparse Version with pywrapgraph 
# ============================================================================
def solve_optimal_assignment_mcf(
    X_exposed_subset: Union[np.ndarray, pd.DataFrame],
    X_control: Union[np.ndarray, pd.DataFrame],
    candidate_lists: List[List[int]],
    threshold: float,
    metric: str,
    n_neighbors: int,
    all_control_indices: np.ndarray,
    gower_model=None,
    precomputed: Optional[List[dict]] = None,
    fuzzy_threshold: bool = False,
    fuzzy_threshold_limit: Optional[float] = None,
    fuzzy_penalty: float = 0.0,
) -> Dict[int, List[int]]:

    # Typed OR-Tools min-cost-flow (new API)
    logger.warning("ENTERED solve_optimal_assignment_mcf()")
    try:
        from ortools.graph.python import min_cost_flow
    except ImportError:
        raise ImportError("OR-Tools typed wrappers not available. `pip install ortools>=9.7`")

    logger.warning("IMPORTED ortools min_cost_flow OK")
    smcf = min_cost_flow.SimpleMinCostFlow()

    # Union of candidates
    unique_candidates = sorted(list(set().union(*candidate_lists)))
    if not unique_candidates:
        return {}

    n_exp = len(X_exposed_subset)
    n_cand = len(unique_candidates)
    ctrl_local_map = {orig: j for j, orig in enumerate(unique_candidates)}

    logger.warning(f"Checkpoint A")
    # Build sparse edges (exposed -> control)
    edges = []  # (i_exp_row, j_ctrl_local, eff_cost_float)
    max_cost = 0.0

    logger.warning(f"Checkpoint B")

    for i in range(n_exp):
        cands = candidate_lists[i]
        if not cands:
            continue

        if precomputed is not None:
            idx_row = precomputed[i]['neighbor_indices']
            dist_row = precomputed[i]['neighbor_distances']
            local_map = dict(zip(idx_row.tolist(), dist_row.tolist()))
            for ctrl_idx in cands:
                d = local_map.get(ctrl_idx)
                if d is None:
                    continue
                if d <= threshold:
                    eff_cost = d
                elif fuzzy_threshold and (fuzzy_threshold_limit is not None) and d <= fuzzy_threshold_limit:
                    eff_cost = d + fuzzy_penalty
                else:
                    continue
                j_local = ctrl_local_map.get(ctrl_idx)
                if j_local is not None:
                    edges.append((i, j_local, eff_cost))
                    if eff_cost > max_cost:
                        max_cost = eff_cost
        else:
            raise ValueError("Sparse MCF requires precomputed kneighbors distances.")

    if not edges:
        return {}
    
    logger.warning(f"Checkpoint C")

    # Integerize costs
    scale = 1_000_000.0 / (max_cost if max_cost > 0 else 1.0)
    edge_arr = np.array(edges)
    edge_i = edge_arr[:, 0].astype(np.int32)
    edge_j = edge_arr[:, 1].astype(np.int32)
    edge_costs = (edge_arr[:, 2] * scale).round().astype(np.int32)

    logger.warning(f"Checkpoint D")

    scaled_max_cost_int = int(edge_costs.max()) if edge_costs.size else 0
    penalty_int = min(scaled_max_cost_int + 10000, 2_000_000_000)  # safe int32

    # Node indices
    source = 0
    sink = n_exp + n_cand + 1

    # Build arc arrays (int32)
    # A) source -> exposed
    t1 = np.zeros(n_exp, dtype=np.int32)
    h1 = np.arange(1, n_exp + 1, dtype=np.int32)
    cap1 = np.full(n_exp, n_neighbors, dtype=np.int32)
    cost1 = np.zeros(n_exp, dtype=np.int32)

    logger.warning(f"Checkpoint E")

    # B) exposed -> control
    t2 = 1 + edge_i
    h2 = 1 + n_exp + edge_j
    cap2 = np.ones(len(edge_i), dtype=np.int32)
    cost2 = edge_costs

    # C) control -> sink
    t3 = np.arange(n_exp + 1, n_exp + n_cand + 1, dtype=np.int32)
    h3 = np.full(n_cand, sink, dtype=np.int32)
    cap3 = np.ones(n_cand, dtype=np.int32)
    cost3 = np.zeros(n_cand, dtype=np.int32)

    logger.warning(f"Checkpoint F")

    # D) dummy exposed -> sink (feasibility)
    t4 = h1.copy()
    h4 = np.full(n_exp, sink, dtype=np.int32)
    cap4 = np.full(n_exp, n_neighbors, dtype=np.int32)
    cost4 = np.full(n_exp, penalty_int, dtype=np.int32)

    # Concatenate
    all_tails = np.concatenate([t1, t2, t3, t4]).astype(np.int32)
    all_heads = np.concatenate([h1, h2, h3, h4]).astype(np.int32)
    all_caps  = np.concatenate([cap1, cap2, cap3, cap4]).astype(np.int32)
    all_costs = np.concatenate([cost1, cost2, cost3, cost4]).astype(np.int32)

    logger.warning(f"Checkpoint G")

    # Add arcs (batched if available; else per-arc)
    try:
        smcf.add_arcs_with_capacity_and_unit_cost(all_tails, all_heads, all_caps, all_costs) # type: ignore
    except AttributeError:
        # Fallback for slightly older versions of the new API
        for u, v, c, w in zip(all_tails.tolist(), all_heads.tolist(), all_caps.tolist(), all_costs.tolist()): # type: ignore
            smcf.add_arc_with_capacity_and_unit_cost(int(u), int(v), int(c), int(w))

    logger.warning(f"Checkpoint H")

    # Supplies
    total_supply = int(n_exp * n_neighbors)
    smcf.set_node_supply(source, total_supply)
    smcf.set_node_supply(sink, -total_supply)

    # Solve
    logger.warning(
        f"MCF size: n_exp={n_exp}, n_cand={n_cand}, edges={len(edges)}, "
        f"arcs_total~{len(edges) + (2*n_exp + n_cand)}"
    )
    status = smcf.solve()
    
    # --- FIX IS HERE ---
    # Status constants are on the instance (smcf.OPTIMAL) or class (SimpleMinCostFlow.OPTIMAL)
    if status != smcf.OPTIMAL:
        raise RuntimeError(f"Min-cost flow did not find optimal solution. Status: {status}")

    # Extract matches for exposed->control arcs (Block B)
    offset_start = len(t1)              # start of block B
    offset_end   = offset_start + len(t2)

    matches: Dict[int, List[int]] = {}

    # Try batched flows() if present
    try:
        arc_indices = np.arange(offset_start, offset_end, dtype=np.int32)
        flows_b = smcf.flows(arc_indices)  # type: ignore
        flows_b = np.array(flows_b, dtype=np.int32)  # ensure numpy array
        mask = (flows_b == 1)
        selected_i = edge_i[mask]
        selected_j = edge_j[mask]
        for i_idx, j_local in zip(selected_i.tolist(), selected_j.tolist()): # type: ignore
            ctrl_idx_orig = unique_candidates[int(j_local)]
            ctrl_id = int(all_control_indices[ctrl_idx_orig])
            matches.setdefault(int(i_idx), []).append(ctrl_id)
    except AttributeError:
        # Fallback: iterate arcs
        for a in range(smcf.num_arcs()):
            flow = smcf.flow(a)
            if flow != 1:
                continue
            # Only exposed->control arcs (in block B range)
            if offset_start <= a < offset_end:
                # Map arc index back to edge arrays: position in block B
                pos = a - offset_start
                i_idx = int(edge_i[pos])
                j_local = int(edge_j[pos])
                ctrl_idx_orig = unique_candidates[j_local]
                ctrl_id = int(all_control_indices[ctrl_idx_orig])
                matches.setdefault(i_idx, []).append(ctrl_id)

    return matches

# =============================================================================
# 3. Reporting & Logging
# =============================================================================

def verbose_matching_results(matched_data, cohort, exposure_status, patient_id):
    """
    Logs detailed statistics about the matching process.
    """
    logger.info("Matching Summary:")
    logger.info("=" * 80)

    # Calculate sets
    cohort_exposed_ids = set(cohort[cohort[exposure_status] == 1][patient_id])
    
    if matched_data.empty:
        logger.warning("Matched DataFrame is empty. No matches found.")
        return

    exposed_matched = matched_data[matched_data[exposure_status] == 1]
    control_matched = matched_data[matched_data[exposure_status] == 0]

    n_exposed = exposed_matched[patient_id].nunique()
    n_control = control_matched[patient_id].nunique()

    logger.info(f" Unique exposed subjects matched: {n_exposed}")
    logger.info(f" Unique control subjects matched: {n_control}")
    logger.info(f" Total subjects in matched set: {n_exposed + n_control}")
    
    if n_exposed > 0:
        logger.info(f" Matching ratio (control:exposed): {n_control/n_exposed:.2f}:1")
    
    logger.info("-" * 80)

    # Check for duplicates (Controls should usually be unique if without replacement)
    exposed_dups = exposed_matched[patient_id].duplicated().sum()
    control_dups = control_matched[patient_id].duplicated().sum()

    logger.info(f" Duplicate exposed rows (expected due to joins): {exposed_dups}")
    logger.info(f" Duplicate control rows (should be 0 if no repl.): {control_dups}")
    logger.info("-" * 80)

    # Unmatched
    matched_exposed_ids = set(exposed_matched[patient_id])
    not_matched_count = len(cohort_exposed_ids - matched_exposed_ids)
    logger.info(f" Number of exposed subjects NOT matched: {not_matched_count}")
    logger.info("-" * 80)

    # Match Distribution
    if 'n_matches' in exposed_matched.columns:
        counts = Counter(exposed_matched['n_matches'])
        counts_sorted = dict(sorted(counts.items(), reverse=True))
        
        display_counts = dict(list(counts_sorted.items())[:10])
        logger.info(f" Distribution of matches per exposed: {display_counts}")
        if len(counts_sorted) > 10:
            logger.info(" ... (truncated)")


# =============================================================================
# 4. Diagnostics (SMD & Tables)
# =============================================================================

def _calculate_smd_numeric(group1, group2, var_list):
    """Calculate the standardized mean differences (SMD) for numeric variables."""
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
    """Calculate the SMD for categorical variables based on group proportions."""
    smds = {}
    # Iterate over union of categories
    all_cats = set(prop_exposed.index).union(set(prop_control.index))
    
    for category in all_cats:
        p1 = prop_exposed.get(category, 0)
        p2 = prop_control.get(category, 0)
        pooled_var = (p1 * (1 - p1) + p2 * (1 - p2)) / 2
        
        if pooled_var > 0:
            smd = abs(p1 - p2) / np.sqrt(pooled_var)
        else:
            smd = 0.0
        smds[category] = smd
    return smds


def summary_stats_table(data, features_numeric, features_categorical, exposure_col, smd_threshold=0.1):
    """
    Builds a summary table of matching diagnostics (Means, Stds, SMDs).
    """
    rows = []
    
    exposed_data = data[data[exposure_col] == 1]
    control_data = data[data[exposure_col] == 0]

    if exposed_data.empty or control_data.empty:
        logger.warning("Cannot calculate diagnostics: One group is empty.")
        return None

    # 1. Numeric Diagnostics
    if features_numeric:
        smds_num = _calculate_smd_numeric(exposed_data, control_data, features_numeric)
        
        for feature in features_numeric:
            rows.append({
                'Feature': feature,
                'Mean_Exposed': exposed_data[feature].mean(),
                'Std_Exposed': exposed_data[feature].std(),
                'Mean_Control': control_data[feature].mean(),
                'Std_Control': control_data[feature].std(),
                'SMD': smds_num.get(feature, np.nan)
            })

    # 2. Categorical Diagnostics
    if features_categorical:
        for feature in features_categorical:
            prop_exposed = exposed_data[feature].value_counts(normalize=True)
            prop_control = control_data[feature].value_counts(normalize=True)

            smds_cat = _calculate_smd_categorical(prop_exposed, prop_control, feature)

            for category, smd_val in smds_cat.items():
                rows.append({
                    'Feature': f"{feature}_{str(category)[:15]}", # Shorten name
                    'Mean_Exposed': prop_exposed.get(category, 0) * 100,
                    'Std_Exposed': np.nan,
                    'Mean_Control': prop_control.get(category, 0) * 100,
                    'Std_Control': np.nan,
                    'SMD': smd_val
                })

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df['SMD Result'] = summary_df['SMD'].apply(lambda x: "OK" if x < smd_threshold else "BAD")
        return np.round(summary_df, 3)
    return pd.DataFrame()


# def matching_diagnostics(matched_data, exposure_status, features_numeric, features_categorical):
#     """
#     Main entry point for running and logging diagnostics.
#     """
#     if features_numeric is None and features_categorical is None:
#         logger.warning("No features provided for diagnostics.")
#         return None

#     logger.info("Running Matching Diagnostics...")
    
#     summary_table = summary_stats_table(
#         data=matched_data,
#         features_numeric=features_numeric or [],
#         features_categorical=features_categorical or [],
#         exposure_col=exposure_status,
#         smd_threshold=0.1
#     )
    
#     if summary_table is not None and not np.array(summary_table).size == 0:
#         logger.info("\n" + np.array2string(np.array(summary_table)))
    
#     # Plotting (Optional Visuals)
#     features_all = (features_numeric or []) + (features_categorical or [])
#     for feature in features_all:
#         try:
#             plot_feature_balance(cohort_before=matched_data, feature=feature, exposure_status_col=exposure_status)
#         except Exception as e:
#             logger.warning(f"Failed to plot balance for {feature}: {e}")

#     return summary_table


def matching_diagnostics(matched_data, exposure_status, features_numeric, features_categorical, plot_features: bool = False):
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
        features_all = features_numeric + features_categorical
        if plot_features:
            for feature in features_all:
                plot_feature_balance(cohort_before=matched_data, feature=feature, exposure_status_col=exposure_status)
    return summary_table