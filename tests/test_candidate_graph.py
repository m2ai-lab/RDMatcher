import numpy as np

from rdmatcher.candidate_graph import candidate_graph_from_prefilter


def test_export_filters_edges_sums_duplicates_and_keeps_explicit_zero():
    candidates = [dict(
        safe_positions=np.array([2, -1, 1, 8, 0]),
        safe_distances=np.array([0., 4., 2., 9., np.nan]),
        competitive_positions=np.array([2, 3, 4, 0]),
        competitive_distances=np.array([.5, np.inf, 5., 0.]),
        candidate_horizon=8,
    ), dict(
        safe_positions=np.array([], dtype=int), safe_distances=np.array([]),
        competitive_positions=np.array([], dtype=int), competitive_distances=np.array([]),
        candidate_horizon=0,
    )]
    graph = candidate_graph_from_prefilter(
        candidates, 5, np.array([10, 11]), np.arange(5), 'euclidean', 2.)
    np.testing.assert_array_equal(graph.matrix.indptr, [0, 3, 3])
    np.testing.assert_array_equal(graph.matrix.indices, [0, 1, 2])
    np.testing.assert_array_equal(graph.matrix.data, [0., 2., .5])
    np.testing.assert_array_equal(graph.candidate_horizons, [8, 0])
    assert graph.matrix.has_canonical_format
    assert graph.matrix.dtype == np.float64


def test_export_empty_case_list():
    graph = candidate_graph_from_prefilter([], 5, np.array([]), np.arange(5), 'euclidean', np.inf)
    assert graph.shape == (0, 5)
    assert graph.nnz == 0
