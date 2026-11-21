"""Storage contracts independent of candidate reduction or distance kernels."""
import numpy as np
import pytest
from rdmatcher._candidate_retrieval import NeighborRows

@pytest.mark.parametrize('dtype', [np.float32, np.float64, np.int32])
def test_random_prefix_migration_and_batched_gather(dtype):
    rng = np.random.default_rng(804)
    store = NeighborRows(257, dtype)
    reference = [np.empty(0, dtype=dtype) for _ in range(257)]
    for _ in range(250):
        rows = rng.choice(257, size=int(rng.integers(1, 80)), replace=False)
        width = int(rng.choice([1, 3, 16, 32, 64]))
        values = rng.integers(0, 1000, size=(len(rows), width)).astype(dtype)
        store[rows] = values
        for row, value in zip(rows, values):
            reference[row] = value.copy()
        widths = np.array([len(row) for row in store.rows])
        active = np.flatnonzero(widths)
        columns = rng.integers(0, widths[active])
        expected = np.asarray([reference[r][c] for r, c in zip(active, columns)])
        np.testing.assert_array_equal(store[active, columns], expected)
        for row in rows:
            np.testing.assert_array_equal(store[row], reference[row])
    store[np.arange(257)] = np.zeros((257, 7), dtype=dtype)
    assert all(len(row) == 7 for row in store.rows)
    assert store.matches_prefix(np.arange(257), np.zeros((257, 9), dtype=dtype))
    changed = np.zeros((257, 9), dtype=dtype)
    changed[41, 6] = 1
    assert not store.matches_prefix(np.arange(257), changed)
    store[np.arange(257)] = np.empty((257, 0), dtype=dtype)
    assert not any(len(row) for row in store.rows)


@pytest.mark.parametrize('metric', ['euclidean', 'gower', 'mahalanobis', 'cosine'])
@pytest.mark.parametrize('ratio', [1, 2, 3])
@pytest.mark.parametrize('threshold', [0.0, 0.4, np.inf])
def test_batched_expansion_matches_single_rounds(monkeypatch, metric, ratio, threshold):
    from rdmatcher._candidate_retrieval import LazyNeighbors
    from kernel_fixtures import frame, make
    df = frame(seed=82, cases=35, controls=160, tied=True)
    reference = make(df, metric, 'first', n_neighbors=ratio)
    optimized = make(df, metric, n_neighbors=ratio)
    reference.threshold = optimized.threshold = threshold
    args = dict(k_candidates=35*ratio, safe_matches=ratio, fuzzy_threshold=False,
                fuzzy_limit=None, batch_size=11)
    expected = reference._prefilter_candidates(**args)
    actual = optimized._prefilter_candidates(**args)
    assert np.all(optimized.candidate_horizons_ <= reference.candidate_horizons_)
    for a, b in zip(actual, expected):
        for key in ['safe_positions', 'safe_distances', 'competitive_positions',
                    'competitive_distances', 'fuzzy_positions', 'fuzzy_distances']:
            np.testing.assert_array_equal(a[key], b[key])


def test_shared_rounds_actually_batch_and_preserve_horizons(monkeypatch):
    from rdmatcher._candidate_retrieval import LazyNeighbors
    from kernel_fixtures import frame, make
    from test_exact_kernel import OrderedOracle
    cases, controls = 40, 100
    df = frame(cases=cases, controls=controls)
    models = [make(df, 'euclidean', mode) for mode in ['first','repeat']]
    for model in models:
        model.X_exposed = np.arange(cases)[:, None]
        model.X_control = np.arange(controls)[:, None]
        costs = np.tile(np.arange(controls, dtype=np.float32), (cases, 1))
        for row in range(cases):
            order = np.roll(np.arange(20), row % 20)
            costs[row, order] = np.arange(20)
        model.nbrs_model = OrderedOracle(costs)
    args = dict(k_candidates=cases, safe_matches=1, fuzzy_threshold=False,
                fuzzy_limit=None, batch_size=10)
    expected = models[0]._prefilter_candidates(**args)
    actual = models[1]._prefilter_candidates(**args)
    assert models[1].candidate_retrieval_stats_['expansion_backend'] == 'compiled_case_queue'
    np.testing.assert_array_equal(models[0].candidate_horizons_, models[1].candidate_horizons_)
    for a, b in zip(actual, expected):
        np.testing.assert_array_equal(a['competitive_positions'], b['competitive_positions'])


def test_native_initial_prefix_is_small_and_boundary_ties_are_local():
    from rdmatcher._general_retrieval import DemandNeighbors
    from kernel_fixtures import frame, make
    from test_exact_kernel import OrderedOracle
    class TrackingOracle(OrderedOracle):
        def __init__(self, costs):
            super().__init__(costs)
            self.requests = []
        def kneighbors(self, query, n_neighbors):
            self.requests.append(n_neighbors)
            return super().kneighbors(query, n_neighbors)
    matcher = make(frame(cases=4, controls=1000), 'euclidean')
    matcher.X_exposed = np.arange(4)[:, None]
    matcher.X_control = np.arange(1000)[:, None]
    costs = np.tile(np.arange(1000, dtype=np.float32)+1, (4,1))
    costs[:, [8, 13]] = 0
    matcher.nbrs_model = TrackingOracle(costs)
    cache = DemandNeighbors(matcher, 128, 4)
    assert matcher.nbrs_model.requests == [2, 4]
    np.testing.assert_array_equal(cache.loaded, np.ones(4))
    np.testing.assert_array_equal(cache.indices[:,0], np.full(4,8))
    cache.ensure(np.arange(4), np.full(4,2))
    np.testing.assert_array_equal(cache.indices[:,1], np.full(4,13))
    calls = matcher.nbrs_model.calls
    cache.ensure(np.arange(4), np.full(4,2))
    assert matcher.nbrs_model.calls == calls
    assert cache.stats['boundary_tie_full_cap_rows'] == 0


@pytest.mark.parametrize('algorithm', ['kd_tree', 'ball_tree'])
@pytest.mark.parametrize('metric', ['euclidean', 'manhattan', 'chebyshev', 'minkowski'])
def test_direct_compiled_query_matches_neighbor_model(monkeypatch, algorithm, metric):
    from kernel_fixtures import frame, make
    from rdmatcher._neighbor_backends import PrefixQuery
    matcher = make(frame(cases=9,controls=400,tied=True),metric,algorithm=algorithm)
    expected = matcher.nbrs_model.kneighbors(matcher.X_exposed,n_neighbors=37)
    provider = PrefixQuery(matcher,3,64)
    assert provider.tree is matcher.nbrs_model._tree
    def redundant_wrapper(*args, **kwargs):
        raise AssertionError('Numeric queries should reuse the fitted compiled tree')
    monkeypatch.setattr(matcher.nbrs_model,'kneighbors',redundant_wrapper)
    actual = provider.query(np.arange(9),37)
    np.testing.assert_array_equal(actual[0],expected[0])
    np.testing.assert_array_equal(actual[1],expected[1])


@pytest.mark.parametrize('dtype', [np.float32,np.float64])
@pytest.mark.parametrize('ratio',[0,1,2,4])
@pytest.mark.parametrize('fuzzy_limit',[None,0.8,np.inf])
def test_compiled_classification_matches_scalar_reference(dtype,ratio,fuzzy_limit):
    from rdmatcher._candidate_retrieval import candidate_masks
    rng=np.random.default_rng(409)
    for _ in range(60):
        ids=rng.integers(-1,30,size=45,dtype=np.int32)
        costs=rng.choice(np.array([0.0,0.1,0.4,0.9,np.inf,np.nan],dtype=dtype),size=45)
        usage=rng.integers(0,5,size=30)
        threshold=float(np.nextafter(np.float64(np.float32(0.4)),-np.inf))
        expected=[[],[],[]]
        for position,idx in enumerate(ids):
            value=float(costs[position])
            if idx>=0 and np.isfinite(value) and value<=threshold:
                if usage[idx]==1:
                    expected[0].append(position)
                    if len(expected[0])>=ratio:
                        break
                else:
                    expected[1].append(position)
            elif fuzzy_limit is not None and value<=fuzzy_limit:
                expected[2].append(position)
            else:
                break
        masks=candidate_masks(ids,costs,usage,threshold,ratio,fuzzy_limit)
        for mask,positions in zip(masks,expected):
            np.testing.assert_array_equal(np.flatnonzero(mask),positions)


def test_candidate_values_borrow_complete_prefix_and_keep_precision():
    from rdmatcher._candidate_retrieval import candidate_values
    ids=np.arange(7,dtype=np.int32)
    costs=np.linspace(0.0,1.0,7,dtype=np.float64)
    positions,distances=candidate_values(ids,costs,np.ones(7,dtype=bool))
    assert np.shares_memory(positions,ids)
    assert np.shares_memory(distances,costs)
    assert distances.dtype==np.float64
    mask=ids%2==0
    positions,distances=candidate_values(ids,costs,mask)
    np.testing.assert_array_equal(positions,ids[mask])
    np.testing.assert_array_equal(distances,costs[mask])
    assert not np.shares_memory(positions,ids)
    assert not np.shares_memory(distances,costs)




@pytest.mark.parametrize('metric',['euclidean','cosine','mahalanobis'])
def test_full_scan_prefix_is_stored_once_in_shared_cache(monkeypatch,metric):
    from kernel_fixtures import frame, make
    from rdmatcher._general_retrieval import DemandNeighbors
    options={'algorithm':'brute'} if metric in ('euclidean', 'cosine') else {}
    if metric=='mahalanobis':
        options['mahalanobis_neighbor_backend']='cdist'
    matcher=make(frame(cases=12,controls=500),metric,**options)
    original=matcher.nbrs_model.kneighbors
    expected_d,expected_i=original(matcher.X_exposed,n_neighbors=100)
    expected_d=expected_d.astype(np.float32);expected_i=expected_i.astype(np.int32)
    order=np.lexsort((expected_i,expected_d),axis=1)
    expected_i=np.take_along_axis(expected_i,order,axis=1)
    expected_d=np.take_along_axis(expected_d,order,axis=1)
    calls=[]
    def observed(query,n_neighbors=1,**kwargs):
        calls.append(n_neighbors)
        return original(query,n_neighbors=n_neighbors,**kwargs)
    monkeypatch.setattr(matcher.nbrs_model,'kneighbors',observed)
    cache=DemandNeighbors(matcher,100,12)
    assert calls==[100]
    np.testing.assert_array_equal(cache.loaded,np.full(12,100))
    for k in [1,2,17,99,100]:
        cache.ensure(np.arange(12),np.full(12,k))
    assert calls==[100]
    assert not hasattr(cache.provider,'pool_distances')
    for row in range(12):
        np.testing.assert_array_equal(cache.indices[row],expected_i[row])
        np.testing.assert_array_equal(cache.distances[row],expected_d[row])


def test_shared_neighbor_order_matches_lexsort_for_float32_bit_patterns():
    from rdmatcher._neighbor_backends import _neighbor_order, order_prefix
    rng = np.random.default_rng(506)
    bits = rng.integers(0, 2**32, size=(40, 4096), dtype=np.uint32)
    # Include zeros, infinities, signed NaNs, subnormals and repeated costs/IDs.
    special = np.array([0, 0x80000000, 0x7f800000, 0xff800000,
                        0x7fc00000, 0xffc00001, 1, 0x80000001], dtype=np.uint32)
    bits[:, :len(special)] = special
    costs = bits.view(np.float32)
    controls = rng.integers(-2**31, 2**31, size=costs.shape, dtype=np.int32)
    costs[:, 100:120] = costs[:, 100:101]
    controls[:, 100:110] = -1
    expected = np.lexsort((controls, costs), axis=1)
    actual = _neighbor_order(costs, controls)
    np.testing.assert_array_equal(actual, expected)
    # Compare original bits too, so signed zero and NaN payload order is pinned.
    np.testing.assert_array_equal(np.take_along_axis(bits, actual, axis=1),
                                  np.take_along_axis(bits, expected, axis=1))
    sorted_costs, sorted_controls = order_prefix(costs.copy(), controls.copy())
    np.testing.assert_array_equal(sorted_costs.view(np.uint32),
                                  np.take_along_axis(bits, expected, axis=1))
    np.testing.assert_array_equal(sorted_controls, np.take_along_axis(controls, expected, axis=1))
    for width in [0, 1]:
        np.testing.assert_array_equal(_neighbor_order(costs[:, :width], controls[:, :width]),
                                      np.lexsort((controls[:, :width], costs[:, :width]), axis=1))

@pytest.mark.parametrize('metric', ['euclidean', 'precomputed'])
def test_one_pool_is_reused_and_closed_per_retrieval_session(monkeypatch, metric):
    import rdmatcher._general_retrieval as retrieval
    from kernel_fixtures import frame, make
    pools = []
    original = retrieval.ThreadPoolExecutor
    class ObservedPool(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.map_calls = 0
            pools.append(self)
        def map(self, *args, **kwargs):
            self.map_calls += 1
            return super().map(*args, **kwargs)
    monkeypatch.setattr(retrieval, 'ThreadPoolExecutor', ObservedPool)
    if metric == 'precomputed':
        from test_precomputed import model
        from rdmatcher import ExecutionOptions
        rdm = model(np.tile(np.arange(80, dtype=float), (24, 1)))
        rdm.rare_matching(np.inf, execution=ExecutionOptions(n_jobs=2, batch_size=5))
    else:
        rdm = make(frame(cases=24, controls=80), metric)
        rdm.n_jobs = 2
        rdm._prefilter_candidates(24, 1, False, None, 5)
    assert len(pools) == 1
    assert pools[0].map_calls > 1
    assert pools[0]._shutdown
    assert not any(thread.is_alive() for thread in pools[0]._threads)


@pytest.mark.parametrize('failure', ['initial_query', 'expansion'])
def test_retrieval_pool_closes_on_failure(monkeypatch, failure):
    import rdmatcher._general_retrieval as retrieval
    from rdmatcher._candidate_retrieval import LazyNeighbors
    from kernel_fixtures import frame, make
    pools = []
    original = retrieval.ThreadPoolExecutor
    def observed(*args, **kwargs):
        pool = original(*args, **kwargs)
        pools.append(pool)
        return pool
    monkeypatch.setattr(retrieval, 'ThreadPoolExecutor', observed)
    model = make(frame(cases=24, controls=80), 'euclidean')
    model.n_jobs = 2
    def fail(*args, **kwargs):
        raise RuntimeError('injected failure')
    if failure == 'initial_query':
        monkeypatch.setattr(retrieval.DemandNeighbors, '_retrieve', fail)
    else:
        monkeypatch.setattr(LazyNeighbors, 'expand', fail)
    with pytest.raises(RuntimeError, match='injected failure'):
        model._prefilter_candidates(24, 1, False, None, 5)
    assert len(pools) == 1
    assert pools[0]._shutdown
    assert not any(thread.is_alive() for thread in pools[0]._threads)
