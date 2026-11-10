import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching, maximum_bipartite_matching
from collections import Counter
from typing import List, Union, Optional, Dict
import logging

from .utils import hide_columns
from .plot import plot_feature_balance

logger = logging.getLogger('rdmatcher.matching')
_EXACT_COST_FLOW_BUDGET_BYTES = 128 * 2**20


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

    X_cov = np.nan_to_num(X_cov, nan=0.0, posinf=0.0, neginf=0.0)

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

    mask = (df[exposure_col] == 1).values
    exp_indices = np.where(mask)[0]
    ctrl_indices = np.where(~mask)[0]

    if len(exp_indices) == 0 or len(ctrl_indices) == 0:
        raise ValueError("Data must contain at least one exposed and one control subject.")

    return np.array(exp_indices), np.array(ctrl_indices)



def _assignment_demands(n_rows, n_neighbors, demands=None):
    """Validate per-case residual demands; omitted demands preserve fixed ratios."""
    if demands is None:
        return np.full(n_rows, int(n_neighbors), dtype=np.int64)
    values = np.asarray(demands)
    if values.shape != (n_rows,):
        raise ValueError("demands must have one entry per candidate row")
    if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values != np.floor(values)):
        raise ValueError("demands must be finite non-negative integers")
    if np.any(values > int(n_neighbors)):
        raise ValueError("residual demands cannot exceed n_neighbors")
    return values.astype(np.int64)


def _coverage_priority_mask(n_rows, coverage_priority=None):
    """Return which rows still need their first assigned control prioritized."""
    if coverage_priority is None:
        return np.zeros(n_rows, dtype=bool)
    values = np.asarray(coverage_priority, dtype=bool)
    if values.shape != (n_rows,):
        raise ValueError("coverage_priority must have one entry per candidate row")
    return values


def _assignment_penalty_levels(
    n_slots,
    max_abs_edge_cost,
    base_unmatched_penalty=1_000_000.0,
    prioritize_coverage=False,
):
    """Build penalties for opt-in coverage, slot-fill, and distance priorities."""
    if n_slots <= 0 or not prioritize_coverage:
        return float(base_unmatched_penalty), float(base_unmatched_penalty)
    distance_span = float(n_slots) * float(max_abs_edge_cost)
    extra_slot_penalty = max(float(base_unmatched_penalty), distance_span + 1.0)
    coverage_penalty = float(n_slots) * extra_slot_penalty + distance_span + 1.0
    return extra_slot_penalty, coverage_penalty


def _exact_sparse_weight_shift(costs):
    """Return nonzero weights only when adding one is exact for every edge.

    TwoSum's residual detects rounding that a subtract-and-compare round trip
    misses for large values. A constant exact shift preserves the objective
    because the augmented graph assigns every slot to a real or dummy edge.
    """
    values = np.asarray(costs, dtype=np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        shifted = values + 1.0
        virtual = shifted - values
        residual = (values - (shifted - virtual)) + (1.0 - virtual)
    if not np.isfinite(shifted).all() or np.any(residual != 0):
        return None
    return shifted


def _has_complete_assignment(n_rows, n_controls, rows, cols, demands):
    if sum(int(value) for value in demands) > n_controls:
        return False
    structure = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_rows, n_controls))
    degrees = np.diff(structure.indptr)
    if sum(int(d) * int(q) for d, q in zip(degrees, demands)) >= np.iinfo(np.int32).max:
        raise ValueError('Expanded assignment graph exceeds the SciPy int32 edge-index range')
    slots = structure[np.repeat(np.arange(n_rows), demands)]
    return bool(np.all(maximum_bipartite_matching(slots, perm_type='column') >= 0))


def _binary_integer_costs(costs):
    ratios = [float(cost).as_integer_ratio() for cost in costs]
    denominator = max((d for _, d in ratios), default=1)
    return [n * (denominator // d) for n, d in ratios], denominator


def _scipy_integer_weights(costs, slots, shift, *, certified_proposal=False):
    """Use exact binary units with an integer-arithmetic margin for SciPy.

    A caller committing to independent exact verification may use representable
    input weights and certify the answer afterward. Other callers retain the
    conservative bound on intermediate arithmetic. No unverified relaxed solve
    is exposed through RDMatcher Optimal.
    """
    integers, denominator = _binary_integer_costs(costs)
    largest = max(integers, default=0) + shift
    unsafe = (largest >= 2**52 if certified_proposal
              else 8 * (int(slots) + 1)**2 * largest >= 2**53)
    if unsafe:
        return None, denominator
    return np.asarray([v + shift for v in integers], dtype=np.float64), denominator


def _solve_exact_cost_flow(rows, cols, costs, demands, coverage, valid_indices,
                           controls, all_control_indices, diagnostics, reason):
    """Optimize represented binary costs exactly on a bounded residual graph.

    Every finite float is an integer divided by a power of two. A common binary
    denominator therefore preserves *all* supplied costs without quantization.
    Successive shortest augmenting paths use Python integer costs/potentials.
    This handles incomplete assignments without floating-point dummy penalties.
    """
    from heapq import heappop, heappush
    integer_costs, denominator = _binary_integer_costs(costs)
    slots = int(sum(demands))
    span = slots * max(integer_costs, default=0)
    extra_penalty = span + 1
    coverage_penalty = slots * extra_penalty + span + 1
    n_rows, n_controls = len(demands), len(controls)
    nodes = n_rows + n_controls + 2
    estimated_bytes = ((len(costs) + 3*n_rows + n_controls) * 2
                       * (160 + (coverage_penalty.bit_length() + 7)//8) + nodes*128)
    if estimated_bytes > _EXACT_COST_FLOW_BUDGET_BYTES:
        raise ValueError("Exact-cost fallback graph exceeds its memory budget; no approximate assignment was returned")
    source, sink = 0, nodes - 1
    graph = [[] for _ in range(nodes)]

    def add_edge(tail, head, capacity, cost):
        forward = [head, len(graph[head]), int(capacity), int(cost)]
        reverse = [tail, len(graph[tail]), 0, -int(cost)]
        graph[tail].append(forward)
        graph[head].append(reverse)
        return forward

    for row, demand in enumerate(demands):
        add_edge(source, 1 + row, demand, 0)
        remaining = int(demand)
        if coverage[row] and remaining:
            add_edge(1 + row, sink, 1, coverage_penalty)
            remaining -= 1
        if remaining:
            add_edge(1 + row, sink, remaining, extra_penalty)
    for col in range(n_controls):
        add_edge(1 + n_rows + col, sink, 1, 0)
    real_edges = []
    for row, col, cost in zip(rows, cols, integer_costs):
        edge = add_edge(1 + int(row), 1 + n_rows + int(col), 1, cost)
        real_edges.append((int(row), int(col), cost, edge))

    potentials, flow = [0] * nodes, 0
    while flow < slots:
        distances, previous = [None] * nodes, [None] * nodes
        distances[source] = 0
        queue = [(0, source)]
        while queue:
            distance, tail = heappop(queue)
            if distance != distances[tail]:
                continue
            for index, edge in enumerate(graph[tail]):
                head, _, capacity, cost = edge
                if not capacity:
                    continue
                reduced = cost + potentials[tail] - potentials[head]
                if reduced < 0:
                    raise RuntimeError("Exact-cost flow lost its feasible-potential invariant")
                candidate = distance + reduced
                if distances[head] is None or candidate < distances[head]:
                    distances[head] = candidate
                    previous[head] = (tail, index)
                    heappush(queue, (candidate, head))
        if distances[sink] is None:
            raise RuntimeError("Exact-cost flow could not fill its dummy-augmented demand")
        for node, distance in enumerate(distances):
            if distance is not None:
                potentials[node] += distance
        amount, node = slots - flow, sink
        while node != source:
            tail, index = previous[node]
            amount = min(amount, graph[tail][index][2])
            node = tail
        node = sink
        while node != source:
            tail, index = previous[node]
            edge = graph[tail][index]
            edge[2] -= amount
            graph[node][edge[1]][2] += amount
            node = tail
        flow += amount

    matches, objective = {}, 0
    for row, col, cost, edge in real_edges:
        if edge[2] == 0:
            matches.setdefault(int(valid_indices[row]), []).append(int(all_control_indices[controls[col]]))
            objective += cost
    if diagnostics is not None:
        diagnostics.update(requested_solver=diagnostics.get('solver'), solver='exact_integer',
                           solver_fallback_reason=reason, cost_arithmetic='exact_binary_integer',
                           exact_distance_numerator=str(objective), exact_distance_denominator=str(denominator),
                           unmatched_slot_penalty=None, coverage_penalty=None,
                           exact_unmatched_penalty=str(extra_penalty),
                           exact_coverage_penalty=str(coverage_penalty), mcf_cost_scale=None)
    return matches


def recover_exact_precomputed_assignment(precomputed, demands, coverage_priority,
                                         all_control_indices, diagnostics):
    """Recover a result rejected by the independent post-solve certificate."""
    controls = sorted({int(position) for row in precomputed for position in row['candidate_positions']})
    column = {position: i for i, position in enumerate(controls)}
    rows, cols, costs = [], [], []
    for i, row in enumerate(precomputed):
        for position, cost in zip(row['candidate_positions'], row['candidate_distances']):
            rows.append(i)
            cols.append(column[int(position)])
            costs.append(float(cost))
    return _solve_exact_cost_flow(rows, cols, costs, demands, coverage_priority,
        np.arange(len(precomputed)), controls, all_control_indices, diagnostics,
        'exact_optimality_certificate_failed')


def solve_optimal_assignment(
    X_exposed_subset: Union[np.ndarray, pd.DataFrame],
    X_control: Union[np.ndarray, pd.DataFrame],
    candidate_lists: List[List[int]],
    threshold: float,
    metric: str,
    n_neighbors: int,
    all_control_indices: np.ndarray,
    gower_model=None,
    precomputed: Optional[List[dict]] = None,
    demands=None,
    coverage_priority=None,
    solver_diagnostics=None,
    _certified_proposal=False,
):
    """Assign eligible controls using dense Hungarian assignment."""

    row_demands = _assignment_demands(len(candidate_lists), n_neighbors, demands)
    row_coverage_priority = _coverage_priority_mask(len(candidate_lists), coverage_priority)
    valid_exposed_mask = [len(c) > 0 and row_demands[i] > 0 for i, c in enumerate(candidate_lists)]
    n_total = len(candidate_lists)
    n_valid = sum(valid_exposed_mask)
    n_dropped = n_total - n_valid

    if n_dropped > 0:
        logger.debug(
            f"Dropping {n_dropped} subjects from optimal matching because they have 0 valid candidates "
            f"within the threshold ({threshold}). Consider increasing threshold or relaxing candidate selection."
        )

    if n_valid == 0:
        return {}

    valid_indices = np.where(valid_exposed_mask)[0]

    if isinstance(X_exposed_subset, pd.DataFrame):
        X_exposed_active = X_exposed_subset.iloc[valid_indices]
    else:
        X_exposed_active = X_exposed_subset[valid_indices]

    active_candidate_lists = [candidate_lists[i] for i in valid_indices]
    active_coverage_priority = row_coverage_priority[valid_indices]

    if precomputed:
        active_precomputed = [precomputed[i] for i in valid_indices]
    else:
        active_precomputed = None

    unique_candidates = sorted(list(set().union(*active_candidate_lists)))
    if not unique_candidates:
        return {}

    col_map = {orig: new for new, orig in enumerate(unique_candidates)}
    n_exp = len(X_exposed_active)
    n_cand = len(unique_candidates)

    rows, cols, data = [], [], []


    if active_precomputed is not None:
        for i in range(n_exp):
            cands = active_candidate_lists[i]
            if len(cands) == 0: continue
            p = active_precomputed[i]
            if 'candidate_positions' in p:
                for ctrl_idx, d in zip(p['candidate_positions'], p['candidate_distances']):
                    if d <= threshold:
                        rows.append(i)
                        cols.append(col_map[int(ctrl_idx)])
                        data.append(float(d))
            else:
                local_map = dict(zip(p['neighbor_indices'].tolist(), p['neighbor_distances'].tolist()))
                for ctrl_idx in cands:
                    d = local_map.get(ctrl_idx, None)
                    if d is not None and d <= threshold:
                        rows.append(i)
                        cols.append(col_map[ctrl_idx])
                        data.append(d)

    else:
        if isinstance(X_control, pd.DataFrame):
            X_control_sub = X_control.iloc[unique_candidates]
        else:
            X_control_sub = X_control[unique_candidates]

        if metric == "gower":
            if gower_model is None:
                raise ValueError("metric='gower' requires passing 'gower_model' to solve_optimal_assignment")
            dblock = gower_model.cdist(X_exposed_active, X_control_sub)
        else:
            dblock = cdist(X_exposed_active, X_control_sub, metric=metric) # type: ignore

        for i in range(n_exp):
            cands = active_candidate_lists[i]
            if len(cands) == 0: continue
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


    active_demands = row_demands[valid_indices]
    final_h = int(active_demands.sum())
    complete_precomputed = (metric == "precomputed" and _has_complete_assignment(
        n_exp, n_cand, rows, cols, active_demands))
    if metric == "precomputed" and not complete_precomputed:
        return _solve_exact_cost_flow(rows, cols, data, active_demands,
            active_coverage_priority, valid_indices, unique_candidates,
            all_control_indices, solver_diagnostics, 'incomplete_assignment_priorities')
    slot_to_case = np.repeat(np.arange(n_exp), active_demands)
    if metric == "precomputed":
        exact_weights, denominator = _scipy_integer_weights(data, final_h, shift=0,
                                                            certified_proposal=_certified_proposal)
        if exact_weights is None:
            return _solve_exact_cost_flow(rows, cols, data, active_demands,
                active_coverage_priority, valid_indices, unique_candidates,
                all_control_indices, solver_diagnostics, 'costs_exceed_exact_float64_integer_range')
        data = exact_weights
        if solver_diagnostics is not None:
            solver_diagnostics.update(cost_arithmetic='exact_binary_scaled_integers',
                                      cost_unit_denominator=str(denominator))
    final_w = n_cand if complete_precomputed else n_cand + final_h
    matrix_dtype = np.float64 if active_coverage_priority.any() or metric == "precomputed" else np.float32
    extra_slot_penalty, coverage_penalty = ((0., 0.) if complete_precomputed else
        _assignment_penalty_levels(final_h,
            max((abs(float(value)) for value in data), default=0.0),
            prioritize_coverage=bool(active_coverage_priority.any())))
    if solver_diagnostics is not None:
        solver_diagnostics.update(unmatched_slot_penalty=extra_slot_penalty,
                                  coverage_penalty=coverage_penalty)

    total_elements = final_h * final_w
    bytes_per_cost = np.dtype(matrix_dtype).itemsize
    safety_limit = 500_000_000

    if total_elements > safety_limit:
        logger.warning(
            f"Dense matrix allocation too large: {final_h}x{final_w} ({total_elements/1e9:.2f} billion elements). "
            f"This requires ~{total_elements * bytes_per_cost / 1e9:.2f} GB of RAM for the matrix, "
            f"and ~{(total_elements * bytes_per_cost * 4) / 1e9:.2f} GB during processing.\n"
            "Use a lower ratio or solver='scipy_sparse' or solver='mcf' if dense allocation exceeds available memory."
        )

    logger.debug(f"Allocating cost matrix: {final_h}x{final_w} ({matrix_dtype.__name__})")
    augmented_cost = np.full((final_h, final_w), extra_slot_penalty, dtype=matrix_dtype)
    slot_offsets = np.concatenate(([0], np.cumsum(active_demands)))
    for active_row, needs_coverage in enumerate(active_coverage_priority):
        if needs_coverage:
            augmented_cost[int(slot_offsets[active_row]), n_cand:] = coverage_penalty

    cost_coo = coo_matrix((data, (rows, cols)), shape=(n_exp, n_cand))
    dense_small = np.full((n_exp, n_cand), np.inf, dtype=matrix_dtype)
    dense_small[cost_coo.row, cost_coo.col] = cost_coo.data

    del cost_coo, rows, cols, data

    augmented_cost[:, :n_cand] = dense_small[slot_to_case]

    del dense_small

    r_ind, c_ind = linear_sum_assignment(augmented_cost)
    del augmented_cost

    matches = {}
    for r, c in zip(r_ind, c_ind):
        if c < n_cand:
            active_row_idx = int(slot_to_case[r])
            original_subset_idx = valid_indices[active_row_idx]
            control_id = all_control_indices[unique_candidates[c]]
            matches.setdefault(int(original_subset_idx), []).append(int(control_id))

    return matches


def solve_optimal_assignment_scipy_sparse(
    X_exposed_subset: Union[np.ndarray, pd.DataFrame],
    X_control: Union[np.ndarray, pd.DataFrame],
    candidate_lists: List[List[int]],
    threshold: float,
    metric: str,
    n_neighbors: int,
    all_control_indices: np.ndarray,
    gower_model=None,
    precomputed: Optional[List[dict]] = None,
    demands=None,
    coverage_priority=None,
    solver_diagnostics=None,
    _certified_proposal=False,
) -> Dict[int, List[int]]:
    """Assign controls from precomputed candidate distances using sparse assignment.

    Unmatched slots use private dummy columns with slot penalties."""
    if precomputed is None:
        raise ValueError("SciPy sparse assignment requires precomputed candidate distances.")

    row_demands = _assignment_demands(len(candidate_lists), n_neighbors, demands)
    row_coverage_priority = _coverage_priority_mask(len(candidate_lists), coverage_priority)
    valid_indices = [i for i, candidates in enumerate(candidate_lists) if len(candidates) and row_demands[i] > 0]
    if not valid_indices:
        return {}

    unique_candidates = sorted(set().union(*(candidate_lists[i] for i in valid_indices)))
    if not unique_candidates:
        return {}
    col_map = {int(candidate): j for j, candidate in enumerate(unique_candidates)}

    edge_rows, edge_cols, edge_costs = [], [], []
    for active_row, input_row in enumerate(valid_indices):
        pre = precomputed[input_row]
        if "candidate_positions" in pre:
            pairs = zip(pre["candidate_positions"], pre["candidate_distances"])
        else:
            distance_map = dict(zip(pre["neighbor_indices"].tolist(), pre["neighbor_distances"].tolist()))
            pairs = ((candidate, distance_map.get(candidate)) for candidate in candidate_lists[input_row])

        for candidate, distance in pairs:
            if distance is None or not np.isfinite(distance) or distance > threshold:
                continue
            local_col = col_map.get(int(candidate))
            if local_col is not None:
                edge_rows.append(active_row)
                edge_cols.append(local_col)
                edge_costs.append(float(distance))

    if not edge_rows:
        return {}

    n_exp = len(valid_indices)
    active_demands = row_demands[valid_indices]
    n_slots = int(active_demands.sum())
    n_controls = len(unique_candidates)
    complete_precomputed = (metric == "precomputed" and _has_complete_assignment(
        n_exp, n_controls, edge_rows, edge_cols, active_demands))
    if metric == "precomputed" and not complete_precomputed:
        return _solve_exact_cost_flow(edge_rows, edge_cols, edge_costs, active_demands,
            row_coverage_priority[valid_indices], valid_indices, unique_candidates,
            all_control_indices, solver_diagnostics, 'incomplete_assignment_priorities')
    slot_offsets = np.concatenate(([0], np.cumsum(active_demands)))
    slot_to_case = np.repeat(np.arange(n_exp), active_demands)
    extra_slot_penalty, coverage_penalty = ((0., 0.) if complete_precomputed else
        _assignment_penalty_levels(n_slots,
            max((abs(value) for value in edge_costs), default=0.0),
            prioritize_coverage=bool(row_coverage_priority[valid_indices].any())))
    if solver_diagnostics is not None:
        solver_diagnostics.update(unmatched_slot_penalty=extra_slot_penalty,
                                  coverage_penalty=coverage_penalty)
    rows, cols, costs = [], [], []
    for row, col, cost in zip(edge_rows, edge_cols, edge_costs):
        for slot in range(int(slot_offsets[row]), int(slot_offsets[row + 1])):
            rows.append(slot)
            cols.append(col)
            costs.append(cost)

    if not complete_precomputed:
        rows.extend(range(n_slots))
        cols.extend(n_controls + slot for slot in range(n_slots))
    slot_penalties = np.full(n_slots, extra_slot_penalty, dtype=np.float64)
    for active_row, needs_coverage in enumerate(row_coverage_priority[valid_indices]):
        if needs_coverage:
            slot_penalties[int(slot_offsets[active_row])] = coverage_penalty
    if not complete_precomputed:
        costs.extend(slot_penalties.tolist())

    integer_scaled = metric == "precomputed"
    if metric == "precomputed":
        weights, denominator = _scipy_integer_weights(costs, n_slots, shift=1,
                                                      certified_proposal=_certified_proposal)
    else:
        # LAPJVsp treats zero-valued sparse entries as absent. A floating +1
        # can erase small but real cost differences, so use it only when the
        # exact-sum check proves every represented edge survives unchanged.
        weights = _exact_sparse_weight_shift(costs)
        if weights is None:
            weights, denominator = _scipy_integer_weights(costs, n_slots, shift=1)
            integer_scaled = True
    if weights is None:
        return _solve_exact_cost_flow(edge_rows, edge_cols, edge_costs, active_demands,
            row_coverage_priority[valid_indices], valid_indices, unique_candidates,
            all_control_indices, solver_diagnostics, 'costs_exceed_exact_float64_integer_range')
    if solver_diagnostics is not None:
        if integer_scaled:
            solver_diagnostics.update(cost_arithmetic='exact_binary_scaled_integers',
                                      cost_unit_denominator=str(denominator))
        else:
            solver_diagnostics.update(cost_arithmetic='exact_sparse_weight_shift')
    graph = csr_matrix(
        (weights, (rows, cols)),
        shape=(n_slots, n_controls if complete_precomputed else n_controls + n_slots),
    )
    slot_rows, selected_cols = min_weight_full_bipartite_matching(graph)

    matches: Dict[int, List[int]] = {}
    for slot_row, selected_col in zip(slot_rows.tolist(), selected_cols.tolist()):
        if selected_col >= n_controls:
            continue
        active_row = int(slot_to_case[slot_row])
        input_row = valid_indices[active_row]
        control_position = unique_candidates[selected_col]
        control_id = all_control_indices[control_position]
        matches.setdefault(int(input_row), []).append(int(control_id))
    return matches

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
    demands=None,
    cost_scale: float = 1_000_000.0,
    unmatched_penalty: float = 1_000_000.0,
    coverage_priority=None,
    solver_diagnostics=None,
) -> Dict[int, List[int]]:

    row_demands = _assignment_demands(len(candidate_lists), n_neighbors, demands)
    row_coverage_priority = _coverage_priority_mask(len(candidate_lists), coverage_priority)

    try:
        from ortools.graph.python import min_cost_flow
    except ImportError:
        raise ImportError("OR-Tools typed wrappers not available. `pip install ortools>=9.7`")

    smcf = min_cost_flow.SimpleMinCostFlow()

    unique_candidates = sorted(list(set().union(*candidate_lists)))
    if not unique_candidates:
        return {}

    n_exp = len(X_exposed_subset)
    n_cand = len(unique_candidates)
    ctrl_local_map = {orig: j for j, orig in enumerate(unique_candidates)}

    edges = []
    max_cost = 0.0


    for i in range(n_exp):
        cands = candidate_lists[i]
        if len(cands) == 0:
            continue

        if precomputed is not None:
            p = precomputed[i]
            if 'candidate_positions' in p:
                candidate_pairs = zip(p['candidate_positions'], p['candidate_distances'])
            else:
                local_map = dict(zip(p['neighbor_indices'].tolist(), p['neighbor_distances'].tolist()))
                candidate_pairs = ((ctrl_idx, local_map.get(ctrl_idx)) for ctrl_idx in cands)
            for ctrl_idx, d in candidate_pairs:
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


    if not np.isfinite(cost_scale) or cost_scale <= 0:
        raise ValueError("cost_scale must be finite and positive")
    if not np.isfinite(unmatched_penalty) or unmatched_penalty <= 0:
        raise ValueError("unmatched_penalty must be finite and positive")
    scale = float(cost_scale)
    n_slots = int(row_demands.sum())
    edge_arr = np.array(edges)
    edge_i = edge_arr[:, 0].astype(np.int32)
    edge_j = edge_arr[:, 1].astype(np.int32)
    if metric == "precomputed":
        integers, denominator = _binary_integer_costs(edge_arr[:, 2])
        span = n_slots * max(integers, default=0)
        extra_penalty_int = span + 1
        coverage_penalty_int = n_slots * extra_penalty_int + span + 1
        bound = (n_exp + n_cand + 2) * (n_slots + 1) * max(extra_penalty_int, coverage_penalty_int)
        if bound >= 2**60:
            return _solve_exact_cost_flow(edge_i, edge_j, edge_arr[:, 2], row_demands,
                row_coverage_priority, np.arange(n_exp), unique_candidates,
                all_control_indices, solver_diagnostics, 'costs_exceed_safe_int64_flow_range')
        edge_costs = np.asarray(integers, dtype=np.int64)
        if solver_diagnostics is not None:
            solver_diagnostics.update(cost_arithmetic='exact_binary_scaled_int64',
                cost_unit_denominator=str(denominator), mcf_cost_scale=None,
                unmatched_slot_penalty=None, coverage_penalty=None,
                exact_unmatched_penalty=str(extra_penalty_int),
                exact_coverage_penalty=str(coverage_penalty_int))
    else:
        extra_slot_penalty, coverage_penalty = _assignment_penalty_levels(
            n_slots, max((abs(edge[2]) for edge in edges), default=0.0),
            unmatched_penalty, prioritize_coverage=bool(row_coverage_priority.any()))
        if solver_diagnostics is not None:
            solver_diagnostics.update(unmatched_slot_penalty=extra_slot_penalty,
                                      coverage_penalty=coverage_penalty)
        max_penalty = max(extra_slot_penalty, coverage_penalty)
        if max(max_cost, max_penalty) * scale >= np.iinfo(np.int64).max:
            raise ValueError("scaled assignment costs exceed int64 capacity")
        edge_costs = (edge_arr[:, 2] * scale).round().astype(np.int64)
        extra_penalty_int = int(round(extra_slot_penalty * scale))
        coverage_penalty_int = int(round(coverage_penalty * scale))

    source = 0
    sink = n_exp + n_cand + 1

    t1 = np.zeros(n_exp, dtype=np.int32)
    h1 = np.arange(1, n_exp + 1, dtype=np.int32)
    cap1 = row_demands.astype(np.int32)
    cost1 = np.zeros(n_exp, dtype=np.int64)


    t2 = 1 + edge_i
    h2 = 1 + n_exp + edge_j
    cap2 = np.ones(len(edge_i), dtype=np.int32)
    cost2 = edge_costs

    t3 = np.arange(n_exp + 1, n_exp + n_cand + 1, dtype=np.int32)
    h3 = np.full(n_cand, sink, dtype=np.int32)
    cap3 = np.ones(n_cand, dtype=np.int32)
    cost3 = np.zeros(n_cand, dtype=np.int64)


    dummy_tails, dummy_caps, dummy_costs = [], [], []
    for row, demand in enumerate(row_demands.tolist()):
        if demand <= 0:
            continue
        if row_coverage_priority[row]:
            dummy_tails.append(row + 1)
            dummy_caps.append(1)
            dummy_costs.append(coverage_penalty_int)
            remaining = demand - 1
        else:
            remaining = demand
        if remaining > 0:
            dummy_tails.append(row + 1)
            dummy_caps.append(remaining)
            dummy_costs.append(extra_penalty_int)
    t4 = np.asarray(dummy_tails, dtype=np.int32)
    h4 = np.full(len(t4), sink, dtype=np.int32)
    cap4 = np.asarray(dummy_caps, dtype=np.int32)
    cost4 = np.asarray(dummy_costs, dtype=np.int64)

    all_tails = np.concatenate([t1, t2, t3, t4]).astype(np.int32)
    all_heads = np.concatenate([h1, h2, h3, h4]).astype(np.int32)
    all_caps  = np.concatenate([cap1, cap2, cap3, cap4]).astype(np.int32)
    all_costs = np.concatenate([cost1, cost2, cost3, cost4]).astype(np.int64)


    try:
        smcf.add_arcs_with_capacity_and_unit_cost(all_tails, all_heads, all_caps, all_costs) # type: ignore
    except AttributeError:
        for u, v, c, w in zip(all_tails.tolist(), all_heads.tolist(), all_caps.tolist(), all_costs.tolist()): # type: ignore
            smcf.add_arc_with_capacity_and_unit_cost(int(u), int(v), int(c), int(w))


    total_supply = int(row_demands.sum())
    smcf.set_node_supply(source, total_supply)
    smcf.set_node_supply(sink, -total_supply)

    logger.debug(
        "MCF size: n_exp=%s, n_cand=%s, edges=%s, arcs_total~%s",
        n_exp, n_cand, len(edges), len(edges) + (2 * n_exp + n_cand),
    )
    status = smcf.solve()

    if status != smcf.OPTIMAL:
        raise RuntimeError(f"Min-cost flow did not find optimal solution. Status: {status}")

    offset_start = len(t1)
    offset_end   = offset_start + len(t2)

    matches: Dict[int, List[int]] = {}

    try:
        arc_indices = np.arange(offset_start, offset_end, dtype=np.int32)
        flows_b = smcf.flows(arc_indices)  # type: ignore
        flows_b = np.array(flows_b, dtype=np.int32)
        mask = (flows_b == 1)
        selected_i = edge_i[mask]
        selected_j = edge_j[mask]
        for i_idx, j_local in zip(selected_i.tolist(), selected_j.tolist()): # type: ignore
            ctrl_idx_orig = unique_candidates[int(j_local)]
            ctrl_id = int(all_control_indices[ctrl_idx_orig])
            matches.setdefault(int(i_idx), []).append(ctrl_id)
    except AttributeError:
        for a in range(smcf.num_arcs()):
            flow = smcf.flow(a)
            if flow != 1:
                continue
            if offset_start <= a < offset_end:
                pos = a - offset_start
                i_idx = int(edge_i[pos])
                j_local = int(edge_j[pos])
                ctrl_idx_orig = unique_candidates[j_local]
                ctrl_id = int(all_control_indices[ctrl_idx_orig])
                matches.setdefault(i_idx, []).append(ctrl_id)

    return matches


def verbose_matching_results(matched_data, cohort, exposure_status, patient_id):
    """
    Logs detailed statistics about the matching process.
    """
    logger.info("Matching Summary:")
    logger.info("=" * 80)

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

    exposed_dups = exposed_matched[patient_id].duplicated().sum()
    control_dups = control_matched[patient_id].duplicated().sum()

    logger.info(f" Duplicate exposed rows (expected due to joins): {exposed_dups}")
    logger.info(f" Duplicate control rows (should be 0 if no repl.): {control_dups}")
    logger.info("-" * 80)

    matched_exposed_ids = set(exposed_matched[patient_id])
    not_matched_count = len(cohort_exposed_ids - matched_exposed_ids)
    logger.info(f" Number of exposed subjects NOT matched: {not_matched_count}")
    logger.info("-" * 80)

    if 'n_matches' in exposed_matched.columns:
        counts = Counter(exposed_matched['n_matches'])
        counts_sorted = dict(sorted(counts.items(), reverse=True))

        display_counts = dict(list(counts_sorted.items())[:10])
        logger.info(f" Distribution of matches per exposed: {display_counts}")
        if len(counts_sorted) > 10:
            logger.info(" ... (truncated)")



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

    if features_categorical:
        for feature in features_categorical:
            prop_exposed = exposed_data[feature].value_counts(normalize=True)
            prop_control = control_data[feature].value_counts(normalize=True)

            smds_cat = _calculate_smd_categorical(prop_exposed, prop_control, feature)

            for category, smd_val in smds_cat.items():
                rows.append({
                    'Feature': f"{feature}_{str(category)[:15]}",
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
