import sys
from pathlib import Path
import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
from rdmatcher._cosine_neighbors import CosineNeighbors
from rdmatcher._general_retrieval import DemandNeighbors
from rdmatcher import RDMatcher
from kernel_fixtures import frame, make

@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('dims', [1, 4, 12, 40])
@pytest.mark.parametrize('scale', [1e-25, 1.0, 1e+25])
def test_cached_normalization_is_bitwise_cosine(dtype, dims, scale):
    rng = np.random.default_rng(7)
    controls = (rng.normal(size=(83, dims)) * scale).astype(dtype)
    queries = (rng.normal(size=(13, dims)) * scale).astype(dtype)
    controls[:2] = 0
    queries[0] = 0
    controls[2:5] = queries[2]
    model = CosineNeighbors().fit(controls)
    with np.errstate(over='ignore', under='ignore'):
        expected = cosine_distances(queries, controls)
        actual = model.distances(queries)
    assert np.array_equal(actual, expected)
    original = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=1).fit(controls)
    for k in (1, 16, 83):
        with np.errstate(over='ignore', under='ignore'):
            d, i = original.kneighbors(queries, n_neighbors=k)
            new_d, new_i = model.kneighbors(queries, n_neighbors=k)
        assert np.array_equal(i, new_i)
        assert np.array_equal(d, new_d)
    assert model.normalization_stats_['reference_normalizations'] == 1

@pytest.mark.parametrize('threshold', [0.0, 0.1, 1.0, np.inf])
@pytest.mark.parametrize('ratio', [1, 2, 3])
def test_cosine_kernel_matches_sklearn_at_thresholds(threshold, ratio):
    df = frame(cases=25, controls=100, tied=True)
    df.loc[25:44, list('abcd')] = df.loc[:19, list('abcd')].to_numpy()
    df.loc[[0, 25], list('abcd')] = 0
    models = []
    for reuse in (False, True):
        m = make(df, 'cosine', 'demand', n_neighbors=ratio, algorithm='brute')
        if not reuse:
            m.nbrs_model = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=1).fit(m.X_control)
        m.threshold = threshold
        m.match(global_optimal=True, competitive_match=False, prioritize_coverage=True, batch_size=5)
        models.append(m)
    a, b = models
    assert a.assignment_diagnostics_['initial_candidate_graph_sha256'] == b.assignment_diagnostics_['initial_candidate_graph_sha256']
    assert a.assignment_diagnostics_['matched_control_count'] == b.assignment_diagnostics_['matched_control_count']
    assert a.assignment_diagnostics_['total_matching_distance'] == b.assignment_diagnostics_['total_matching_distance']

def test_cosine_controls_normalized_once_across_prefix_requests():
    m = make(frame(cases=40, controls=170), 'cosine', 'demand', algorithm='brute')
    lazy = DemandNeighbors(m, 40, 7)
    for k in (17, 33, 40):
        lazy.ensure(np.arange(40), np.full(40, k))
    assert m.nbrs_model.normalization_stats_ == dict(reference_normalizations=1, query_rows=40)

def test_parallel_cosine_queries_match_sklearn():
    rng = np.random.default_rng(11)
    ref = rng.normal(size=(97, 7))
    query = rng.normal(size=(13, 7))
    a = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=2).fit(ref)
    b = CosineNeighbors(n_jobs=2).fit(ref)
    expected = a.kneighbors(query, n_neighbors=17)
    actual = b.kneighbors(query, n_neighbors=17)
    assert all((np.array_equal(x, y) for x, y in zip(expected, actual)))

def test_actual_normalization_calls_are_not_repeated(monkeypatch):
    import rdmatcher._cosine_neighbors as module
    original = module.normalize
    calls = []

    def spy(values, *args, **kwargs):
        calls.append(len(values))
        return original(values, *args, **kwargs)
    monkeypatch.setattr(module, 'normalize', spy)
    m = make(frame(cases=40, controls=170), 'cosine', 'demand', algorithm='brute')
    lazy = DemandNeighbors(m, 40, 7)
    for k in (17, 33, 40):
        lazy.ensure(np.arange(40), np.full(40, k))
    assert calls.count(170) == 1
    assert sum((c for c in calls if c != 170)) == 40


@pytest.mark.parametrize('metric', ['euclidean', 'cosine'])
def test_shared_index_exact_order_through_tied_cutoffs(metric):
    from rdmatcher._neighbor_backends import IndexedNeighbors

    rng = np.random.default_rng(75)
    controls = rng.normal(size=(159, 5))
    controls[1:17] = controls[0]
    controls[-5:] = 0
    cases = np.vstack((controls[0], np.zeros(5), rng.normal(size=(5, 5))))
    index = IndexedNeighbors(controls, metric)
    positions = np.broadcast_to(np.arange(len(controls), dtype=np.int32),
                                (len(cases), len(controls)))
    if metric == 'cosine':
        expected_costs = cosine_distances(cases, controls).astype(np.float32)
    else:
        expected_costs = np.linalg.norm(cases[:, None] - controls, axis=2).astype(np.float32)
    order = np.lexsort((positions, expected_costs), axis=1)
    costs = np.take_along_axis(expected_costs, order, axis=1)
    for width in (1, 7, 16, 27, 100, len(controls)):
        actual_costs, actual_ids = index.kneighbors(cases, width)
        np.testing.assert_array_equal(actual_ids, order[:, :width])
        np.testing.assert_allclose(actual_costs, costs[:, :width], rtol=0, atol=2e-15)


@pytest.mark.parametrize('threshold', [0.0, 0.1, 1.0, np.inf])
@pytest.mark.parametrize('ratio', [1, 2, 3])
def test_indexed_cosine_preserves_exhaustive_matching(threshold, ratio):
    df = frame(cases=25, controls=100, tied=True)
    df.loc[25:44, list('abcd')] = df.loc[:19, list('abcd')].to_numpy()
    df.loc[[0, 25], list('abcd')] = 0
    results = []
    for algorithm in ('auto', 'brute'):
        matcher = make(df, 'cosine', n_neighbors=ratio, algorithm=algorithm)
        matcher.threshold = threshold
        matcher.match(global_optimal=True, competitive_match=False,
                      prioritize_coverage=True, batch_size=5)
        diagnostics = matcher.assignment_diagnostics_
        results.append((diagnostics['matched_control_count'],
                        diagnostics['total_matching_distance']))
    assert results[0][0] == results[1][0]
    assert results[0][1] == pytest.approx(results[1][1], abs=2e-15)
