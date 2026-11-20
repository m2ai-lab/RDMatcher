from rdmatcher import MatchingOptions, ExecutionOptions
import numpy as np
from rdmatcher.candidate_graph import candidate_graph_from_prefilter
from rdmatcher.matching import solve_optimal_assignment, solve_optimal_assignment_scipy_sparse

def _solve_both(candidate_lists, costs_by_row, n_neighbors=1, threshold=np.inf):
    precomputed = [{'candidate_positions': np.asarray(candidates, dtype=np.int32), 'candidate_distances': np.asarray(costs, dtype=np.float32)} for candidates, costs in zip(candidate_lists, costs_by_row)]
    args = dict(X_exposed_subset=np.zeros((len(candidate_lists), 1)), X_control=np.zeros((max((max(row, default=-1) for row in candidate_lists)) + 1, 1)), candidate_lists=candidate_lists, threshold=threshold, metric='euclidean', n_neighbors=n_neighbors, all_control_indices=np.arange(100, 100 + max((max(row, default=-1) for row in candidate_lists)) + 1), precomputed=precomputed)
    sparse = solve_optimal_assignment_scipy_sparse(**args)
    dense = solve_optimal_assignment(**args)
    return (sparse, dense)

def test_sparse_solver_matches_hungarian_with_zero_costs_and_ratio_two():
    candidates = [[0, 1, 2], [0, 1, 2]]
    costs = [[0.0, 0.2, 0.7], [0.1, 0.0, 0.4]]
    sparse, dense = _solve_both(candidates, costs, n_neighbors=2)
    assert sparse == dense
    assert all((len(matches) <= 2 for matches in sparse.values()))
    selected = [control for matches in sparse.values() for control in matches]
    assert len(selected) == len(set(selected))

def test_sparse_solver_preserves_tiny_feature_cost_differences():
    candidates = [[0, 1], [0, 1]]
    costs = [[0.0, 1e-20], [0.0, 2e-20]]
    sparse, dense = _solve_both(candidates, costs)
    assert sparse == dense == {0: [101], 1: [100]}

def test_sparse_solver_preserves_strictly_positive_cosine_costs():
    candidates = [np.array([0, 1]), np.array([0, 1])]
    costs = [np.array([1e-20, 2e-20]), np.array([1e-20, 3e-20])]
    diagnostics = {}
    matches = solve_optimal_assignment_scipy_sparse(
        X_exposed_subset=None, X_control=None, candidate_lists=candidates,
        threshold=np.inf, metric='cosine', n_neighbors=1,
        all_control_indices=np.array([100, 101]), coverage_priority=[True, True],
        precomputed=[{'candidate_positions': row, 'candidate_distances': values}
                     for row, values in zip(candidates, costs)],
        solver_diagnostics=diagnostics,
    )
    assert matches == {0: [101], 1: [100]}
    assert diagnostics['cost_arithmetic'] == 'unshifted_positive_costs'

def test_sparse_solver_preserves_unmatched_slots_when_threshold_blocks_edges():
    sparse, dense = _solve_both(candidate_lists=[[0], [0]], costs_by_row=[[0.1], [2.0]], threshold=0.5)
    assert sparse == dense
    assert sum(map(len, sparse.values())) == 1

def test_coverage_priority_is_opt_in_for_optimal_solvers():
    candidates = [[0, 1], [0]]
    costs = [[0.0, 0.0], [0.5]]
    precomputed = [{'candidate_positions': np.asarray(row, dtype=np.int32), 'candidate_distances': np.asarray(row_costs, dtype=np.float32)} for row, row_costs in zip(candidates, costs)]
    args = dict(X_exposed_subset=np.zeros((2, 1)), X_control=np.zeros((2, 1)), candidate_lists=candidates, threshold=1.0, metric='euclidean', n_neighbors=2, demands=[2, 2], all_control_indices=np.array([10, 11]), precomputed=precomputed)
    for solver in (solve_optimal_assignment, solve_optimal_assignment_scipy_sparse):
        penalized_matches = solver(**args)
        assert penalized_matches == {0: [10, 11]}
        matches = solver(**args, coverage_priority=[True, True])
        assert set(matches) == {0, 1}
        assert matches[1] == [10]
        assert matches[0] == [11]

def test_candidate_graph_contains_prefiltered_threshold_edges_and_original_ids():
    candidates = [{'safe_positions': np.array([0, 2]), 'safe_distances': np.array([0.0, 0.2]), 'competitive_positions': np.array([1]), 'competitive_distances': np.array([0.9]), 'candidate_horizon': 3}, {'safe_positions': np.array([1]), 'safe_distances': np.array([0.3]), 'competitive_positions': np.array([], dtype=np.int32), 'competitive_distances': np.array([], dtype=np.float32), 'candidate_horizon': 2}]
    graph = candidate_graph_from_prefilter(candidate_list=candidates, n_controls=3, case_ids=np.array(['case-A', 'case-B']), control_ids=np.array(['ctrl-X', 'ctrl-Y', 'ctrl-Z']), distance_metric='gower', threshold=0.5)
    assert graph.matrix.shape == (2, 3)
    assert graph.nnz == 3
    assert graph.matrix[0, 0] == 0.0
    assert graph.matrix[0, 2] == 0.2
    assert graph.matrix[1, 1] == 0.3
    np.testing.assert_array_equal(graph.case_ids, ['case-A', 'case-B'])
    np.testing.assert_array_equal(graph.control_ids, ['ctrl-X', 'ctrl-Y', 'ctrl-Z'])
    np.testing.assert_array_equal(graph.candidate_horizons, [3, 2])
    scipy_weights = graph.for_scipy_sparse_matching()
    assert scipy_weights[0, 0] == 1.0
    assert scipy_weights.nnz == graph.nnz

def test_public_api_exports_graph_and_accepts_sparse_solver():
    import pandas as pd
    from rdmatcher import CandidateGraph, RDMatcher
    population = pd.DataFrame({'patient_id': ['case-1', 'case-2', 'control-1', 'control-2', 'control-3'], 'exposure': [1, 1, 0, 0, 0], 'x': [0.0, 1.0, 0.1, 1.1, 4.0]})
    graph_matcher = RDMatcher(population, 'patient_id', 'exposure', ['x'], [])
    graph = graph_matcher.return_graph(threshold=1.0, n_neighbors=1)
    assert isinstance(graph, CandidateGraph)
    assert graph_matcher.candidate_graph_ is graph
    np.testing.assert_array_equal(graph.case_ids, ['case-1', 'case-2'])
    assert graph_matcher.pop_matched is None
    assert graph_matcher.matched_data is None
    solver_matcher = RDMatcher(population, 'patient_id', 'exposure', ['x'], [])
    matched = solver_matcher.rare_matching(threshold=1.0, n_neighbors=1)
    assert matched['n_matches'].sum() == 2

def test_rare_matching_exposes_coverage_priority_as_opt_in():
    import pandas as pd
    from rdmatcher import RDMatcher
    population = pd.DataFrame({'patient_id': ['case-a', 'case-b', 'control-a', 'control-b'], 'exposure': [1, 1, 0, 0], 'x': [0.0, 1.0, 0.0, 0.1]})
    options = dict(threshold=0.95, n_neighbors=2, distance_metric='euclidean')
    penalized = RDMatcher(population, 'patient_id', 'exposure', ['x'], []).rare_matching(**options)
    prioritized = RDMatcher(population, 'patient_id', 'exposure', ['x'], []).rare_matching(**options, matching=MatchingOptions(prioritize_coverage=True))
    assert int((penalized.loc[penalized['exposure'] == 1, 'n_matches'] > 0).sum()) == 1
    assert int((prioritized.loc[prioritized['exposure'] == 1, 'n_matches'] > 0).sum()) == 2
