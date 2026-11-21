import numpy as np
import pandas as pd
import pytest
import subprocess
import sys
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from rdmatcher import RDMatcher, MatchingOptions, ExecutionOptions


def population(n=3, m=5):
    return pd.DataFrame(dict(id=[f"t{i}" for i in range(n)] + [f"c{i}" for i in range(m)],
                             treated=[1]*n + [0]*m))


def model(costs, pop=None, **kwargs):
    n, m = costs.shape
    pop = population(n, m) if pop is None else pop
    if not isinstance(costs, pd.DataFrame):
        kwargs.setdefault('case_ids', [f"t{i}" for i in range(n)])
        kwargs.setdefault('control_ids', [f"c{i}" for i in range(m)])
    return RDMatcher(pop, 'id', 'treated', [], [], distance_matrix=costs, **kwargs)


@pytest.mark.parametrize('solver', ['hungarian', 'scipy_sparse', 'mcf'])
@pytest.mark.parametrize('allocation', ['optimal', 'hybrid'])
@pytest.mark.parametrize('sparse', [False, True])
def test_solvers_forbidden_zero_ratio(solver, allocation, sparse):
    if solver == 'mcf':
        pytest.importorskip('ortools')
        # Isolate the native backend and load it before pandas/Arrow: some
        # installed protobuf builds abort when imported in the opposite order.
        root = Path(__file__).resolve().parents[1]
        script = (
            "from ortools.graph.python import min_cost_flow; import sys; "
            f"sys.path[:0] = {[str(root / 'src'), str(root / 'tests')]!r}; "
            "from test_precomputed import check_solver_fixture; "
            f"check_solver_fixture({solver!r}, {allocation!r}, {sparse!r})"
        )
        result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
        return
    check_solver_fixture(solver, allocation, sparse)


def check_solver_fixture(solver, allocation, sparse):
    costs = np.array([[0., .3, np.inf, .9, .8], [.2, .1, .4, np.inf, .7],
                      [np.inf]*5])
    if sparse:
        rows, cols = np.where(np.isfinite(costs))
        supplied = csr_matrix((costs[rows, cols], (rows, cols)), shape=costs.shape)
    else:
        supplied = costs
    rdm = model(supplied)
    result = rdm.rare_matching(np.inf, 2, allocation=allocation,
                              execution=ExecutionOptions(solver=solver))
    selected = result[(result.treated == 0) & result.match_group.notna()]
    assert len(selected) == 4
    assert selected.id.is_unique
    assert rdm.assignment_diagnostics_['total_matching_distance'] == pytest.approx(1.3)
    assert rdm.assignment_diagnostics_['unfilled_slots'] == 2


@pytest.mark.parametrize('ratio', [1, 2, 3])
@pytest.mark.parametrize('solver', ['hungarian', 'scipy_sparse'])
def test_full_assignment_oracle(ratio, solver):
    for seed in range(8):
        costs = np.random.default_rng(seed).uniform(0, 1, (4, 14))
        expanded = np.repeat(costs, ratio, axis=0)
        rows, cols = linear_sum_assignment(expanded)
        rdm = model(costs)
        rdm.rare_matching(np.inf, ratio, execution=ExecutionOptions(solver=solver))
        assert rdm.assignment_diagnostics_['total_matching_distance'] == pytest.approx(expanded[rows, cols].sum(), abs=1e-12)


@pytest.mark.parametrize('solver', ['hungarian', 'scipy_sparse'])
def test_precision_alignment_and_no_computation(monkeypatch, solver):
    from rdmatcher.distance import GowerKNN
    from sklearn.neighbors import NearestNeighbors
    import rdmatcher.matching as assignment
    def fail(*args, **kwargs):
        raise AssertionError("computed features or distances")
    monkeypatch.setattr(GowerKNN, 'fit', fail)
    monkeypatch.setattr(NearestNeighbors, 'fit', fail)
    monkeypatch.setattr(assignment, 'cdist', fail)
    monkeypatch.setattr(RDMatcher, '_build_multi_matching_view', fail)
    costs = np.array([[1., 1.+1e-8, 8.], [1., 1.+2e-8, 9.]])
    labels = pd.DataFrame(costs, index=['t0','t1'], columns=['c0','c1','c2'])
    labels = labels.iloc[::-1, ::-1]
    rdm = model(labels, population(2,3).sample(frac=1, random_state=5))
    result = rdm.rare_matching(np.inf, execution=ExecutionOptions(solver=solver))
    assert rdm.assignment_diagnostics_['total_matching_distance'] == 2.+1e-8
    for _, group in result.groupby('match_group'):
        ids = set(group.id)
        assert ids in ({'t0','c1'}, {'t1','c0'})
    graph = rdm.return_graph(np.inf, 2)
    assert graph.distance_metric == 'precomputed'
    assert 1.+2e-8 in graph.matrix.data


def test_repeated_calls_caliper_and_ties():
    pop = population(2,3)
    pop['score'] = [0., 1., 1., 0., 0.]
    costs = np.array([[0., .2, .2], [0., .5, .5]])
    rdm = model(costs, pop)
    rdm.rare_matching(np.inf)
    assert rdm.assignment_diagnostics_['matched_control_count'] == 2
    options = MatchingOptions(ps_hybrid=True, ps_caliper=0, propensity_col='score')
    graph = rdm.return_graph(.3, matching=options)
    assert graph.nnz == 2
    assert graph.matrix[0,1] == .2  # population control position resolves ties
    rdm.rare_matching(0., matching=options)
    assert rdm.assignment_diagnostics_['matched_control_count'] == 1
    graph2 = rdm.return_graph(.3, matching=MatchingOptions(propensity_caliper=0, propensity_col='score'))
    np.testing.assert_array_equal(graph.matrix.toarray(), graph2.matrix.toarray())
    with pytest.raises(ValueError, match='column'):
        model(costs).rare_matching(1., matching=options)


def test_borrow_readonly_memmap(tmp_path):
    values = np.arange(24, dtype=np.float64).reshape(4,6)[:, ::2]
    values.setflags(write=False)
    rdm = model(values)
    assert np.shares_memory(rdm._precomputed_distances.matrix, values)
    rdm.rare_matching(np.inf)
    path = tmp_path / 'costs.dat'
    writable = np.memmap(path, mode='w+', dtype='float64', shape=(2,3))
    writable[:] = [[0., 1., 2.], [2., 0., 1.]]
    writable.flush()
    borrowed = np.memmap(path, mode='r', dtype='float64', shape=(2,3))
    rdm = model(borrowed)
    assert rdm._precomputed_distances.matrix is borrowed
    rdm.rare_matching(np.inf)
    np.testing.assert_array_equal(borrowed, writable)


@pytest.mark.parametrize('bad', [np.nan, -np.inf, -1.])
def test_bad_values(bad):
    for values in (np.array([[bad]]), csr_matrix(np.array([[bad]]))):
        with pytest.raises(ValueError, match='Distances'):
            model(values)


def test_bad_ids_shapes_and_options():
    costs = np.zeros((2,3))
    with pytest.raises(ValueError, match='required'):
        RDMatcher(population(2,3), 'id', 'treated', [], [], distance_matrix=costs)
    for kwargs in (dict(case_ids=['t0','t0']), dict(control_ids=['c0','c1','wrong']),
                   dict(case_ids=['t0'])):
        with pytest.raises(ValueError):
            model(costs, **kwargs)
    with pytest.raises(ValueError, match='shape'):
        model(costs, population(2,2), case_ids=['t0','t1'], control_ids=['c0','c1'])
    duplicate = csr_matrix(([0., 1.], [0, 0], [0, 2]), shape=(1,1))
    with pytest.raises(ValueError, match='Duplicate CSR'):
        model(duplicate)
    labels = pd.DataFrame(costs, index=['t0','t1'], columns=['c0','c1','c2'])
    with pytest.raises(ValueError, match='conflict'):
        model(labels, case_ids=['t1','t0'])
    with pytest.raises(ValueError, match='conflict'):
        model(costs).rare_matching(1., distance_metric='gower')
    for options in (MatchingOptions(pca_filter=True), MatchingOptions(gower_weights={'x':1}),
                    MatchingOptions(weight_numeric=2)):
        with pytest.raises(ValueError, match='conflict'):
            model(costs).rare_matching(1., matching=options)
    with pytest.raises(ValueError, match='requires distance_matrix'):
        RDMatcher(population(), 'id', 'treated', [], []).rare_matching(1., distance_metric='precomputed')


def test_computed_equivalence():
    pop = population(3,7)
    pop['x'] = np.random.default_rng(4).normal(size=10)
    costs = np.abs(pop.x.to_numpy()[:3,None] - pop.x.to_numpy()[None,3:])
    precomputed = model(costs, pop)
    computed = RDMatcher(pop, 'id', 'treated', ['x'], [])
    for rdm, metric in ((precomputed, None), (computed, 'euclidean')):
        rdm.rare_matching(np.inf, 2, distance_metric=metric)
    assert precomputed.assignment_diagnostics_['matched_control_count'] == computed.assignment_diagnostics_['matched_control_count']
    assert precomputed.assignment_diagnostics_['total_matching_distance'] == pytest.approx(computed.assignment_diagnostics_['total_matching_distance'])


@pytest.mark.parametrize('sparse', [False, True])
def test_unlabeled_axis_permutations_and_input_unchanged(sparse):
    costs = np.array([[np.inf, 0., .4], [.1, .3, np.inf]])
    row_order, col_order = [1, 0], [2, 0, 1]
    supplied = costs[row_order][:, col_order]
    if sparse:
        rows, cols = np.where(np.isfinite(supplied))
        supplied = csr_matrix((supplied[rows, cols], (rows, cols)), shape=supplied.shape)
        original = (supplied.data.copy(), supplied.indices.copy(), supplied.indptr.copy())
        for array in (supplied.data, supplied.indices, supplied.indptr):
            array.setflags(write=False)
    else:
        original = supplied.copy()
    rdm = model(supplied, population(2,3).sample(frac=1, random_state=8),
                case_ids=['t1','t0'], control_ids=['c2','c0','c1'])
    rdm.rare_matching(np.inf)
    assert rdm.assignment_diagnostics_['total_matching_distance'] == .1
    if sparse:
        for before, after in zip(original, (supplied.data, supplied.indices, supplied.indptr)):
            np.testing.assert_array_equal(before, after)
    else:
        np.testing.assert_array_equal(original, supplied)


@pytest.mark.parametrize('ratio', [1, 2, 3])
def test_full_eligible_assignment_with_forbidden_edges(ratio):
    for seed in range(8):
        costs = np.random.default_rng(seed).uniform(0, 1, (4,7))
        costs[costs > .35] = np.inf
        slots = len(costs)*ratio
        expanded = np.repeat(costs, ratio, axis=0)
        oracle = np.full((slots, costs.shape[1]+slots), np.inf)
        oracle[:, :costs.shape[1]] = expanded
        oracle[np.arange(slots), costs.shape[1]+np.arange(slots)] = 1e6
        rows, cols = linear_sum_assignment(oracle)
        selected = cols < costs.shape[1]
        rdm = model(costs)
        rdm.rare_matching(np.inf, ratio)
        diag = rdm.assignment_diagnostics_
        assert diag['matched_control_count'] == selected.sum()
        assert diag['total_matching_distance'] == pytest.approx(oracle[rows[selected], cols[selected]].sum(), abs=1e-12)


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('pattern', ['continuous', 'discrete', 'all_equal', 'boundary_tie'])
def test_partial_prefix_matches_full_order_with_permuted_controls(dtype, pattern):
    from rdmatcher._precomputed import _ordered_prefix
    rng = np.random.default_rng(404)
    controls = rng.permutation(5000)
    if pattern == 'continuous':
        costs = rng.random(len(controls)).astype(dtype)
    elif pattern == 'discrete':
        costs = rng.integers(0, 5, size=len(controls)).astype(dtype)
    elif pattern == 'all_equal':
        costs = np.zeros(len(controls), dtype=dtype)
    else:
        costs = np.ones(len(controls), dtype=dtype)
        costs[:8] = 0
        costs[-8:] = np.nextafter(dtype(1), dtype(2))
    before = costs.copy()
    expected = np.lexsort((controls, costs))
    costs.setflags(write=False)
    controls.setflags(write=False)
    for count in (0, 1, 8, 17, 512, len(costs), len(costs) + 1):
        selected = _ordered_prefix(costs, controls, count)
        np.testing.assert_array_equal(selected, expected[:count])
    np.testing.assert_array_equal(costs, before)


def test_large_boundary_tie_sorts_only_retained_edges(monkeypatch):
    from rdmatcher._precomputed import _ordered_prefix
    costs = np.zeros(50000)
    controls = np.arange(len(costs))[::-1]
    original = np.lexsort
    def limited_sort(keys, *args, **kwargs):
        assert len(keys[0]) <= 17
        return original(keys, *args, **kwargs)
    monkeypatch.setattr(np, 'lexsort', limited_sort)
    selected = _ordered_prefix(costs, controls, 17)
    np.testing.assert_array_equal(controls[selected], np.arange(17))


@pytest.mark.parametrize('sparse', [False, True])
def test_partial_selection_preserves_filtered_prefixes_and_padding(sparse):
    from types import SimpleNamespace
    from rdmatcher._precomputed import MatrixPrefixQuery
    costs = np.tile(np.array([0., 0., .1, .2, .2, .2, .3, .4, .5, np.inf]), (2, 1))
    supplied = pd.DataFrame(costs, index=['t0', 't1'], columns=[f'c{i}' for i in range(10)])
    supplied = supplied.iloc[::-1, ::-1]
    if sparse:
        rows, columns = np.where(np.isfinite(supplied.to_numpy()))
        supplied = csr_matrix((supplied.to_numpy()[rows, columns], (rows, columns)), shape=(2,10))
        rdm = model(supplied, case_ids=['t1','t0'], control_ids=[f'c{i}' for i in range(9,-1,-1)])
    else:
        rdm = model(supplied)
    eligible = [np.array([1, 3, 4, 5, 7, 8]), np.array([], dtype=int)]
    owner = SimpleNamespace(precomputed_distances=rdm._precomputed_distances,
                            threshold=.2, ps_eligible_sets=eligible)
    query = MatrixPrefixQuery(owner, cap=3)
    for count in (1, 2, 3):
        distances, indices = query.query(np.array([0, 1]), count)
        np.testing.assert_array_equal(indices[0], np.array([1,3,4])[:count])
        np.testing.assert_array_equal(distances[0], np.array([0., .2, .2])[:count])
        np.testing.assert_array_equal(indices[1], np.full(count, -1))
        assert np.isinf(distances[1]).all()


def test_precomputed_retrieval_starts_with_exact_singleton_prefix(monkeypatch):
    from rdmatcher._precomputed import MatrixPrefixQuery
    observed_counts = []
    original_query = MatrixPrefixQuery.query

    def observed_query(self, rows, count):
        observed_counts.append(int(count))
        return original_query(self, rows, count)

    monkeypatch.setattr(MatrixPrefixQuery, 'query', observed_query)
    # The shared nearest control forces expansion after the initial singleton
    # query.  Exact ordering and the eventual assignment remain unchanged.
    costs = np.array([[0., .2, .3, .4],
                      [0., .1, .5, .6],
                      [.7, .8, .9, 1.]])
    rdm = model(costs)
    rdm.rare_matching(np.inf)
    assert observed_counts[0] == 1
    assert any(count > 1 for count in observed_counts)
    retrieval = rdm.assignment_diagnostics_['candidate_retrieval']
    assert retrieval['prefix_expansion_row_queries'] > 0
    assert rdm.assignment_diagnostics_['total_matching_distance'] == pytest.approx(1.)


def test_precomputed_retrieval_honors_n_jobs():
    costs = np.random.default_rng(123).uniform(size=(12, 128))
    rdm = model(costs, population(12, 128))
    rdm.rare_matching(np.inf, execution=ExecutionOptions(n_jobs=2, batch_size=12))
    retrieval = rdm.assignment_diagnostics_['candidate_retrieval']
    assert retrieval['parallel_workers'] == 2


@pytest.mark.parametrize('scale', [1e-300, 1e-20, 1., 1e9, 1e100])
@pytest.mark.parametrize('solver', ['scipy_sparse', 'hungarian'])
@pytest.mark.parametrize('coverage', [False, True])
def test_cost_scale_does_not_change_complete_optimum(scale, solver, coverage):
    costs = np.array([[0., scale], [0., 2*scale]])
    rdm = model(costs)
    rdm.rare_matching(np.inf, execution=ExecutionOptions(solver=solver),
                     matching=MatchingOptions(prioritize_coverage=coverage))
    assert rdm.assignment_diagnostics_['matched_control_count'] == 2
    assert rdm.assignment_diagnostics_['total_matching_distance'] == scale
    if solver == 'scipy_sparse' and scale < 1e-16:
        assert rdm.assignment_diagnostics_['solver'] == 'exact_integer'
        assert rdm.assignment_diagnostics_['solver_fallback_reason'] == 'costs_exceed_exact_float64_integer_range'


@pytest.mark.parametrize('solver', ['scipy_sparse', 'hungarian'])
def test_large_finite_cost_is_matched_instead_of_fixed_penalty(solver):
    rdm = model(np.array([[2e6]]))
    rdm.rare_matching(np.inf, execution=ExecutionOptions(solver=solver))
    assert rdm.assignment_diagnostics_['matched_control_count'] == 1
    assert rdm.assignment_diagnostics_['total_matching_distance'] == 2e6


def test_exact_weight_shift_detects_rounding_at_both_magnitudes():
    from rdmatcher.matching import _exact_sparse_weight_shift
    assert _exact_sparse_weight_shift([0., .25, 1., 1e6]) is not None
    assert _exact_sparse_weight_shift([1e-20]) is None
    assert _exact_sparse_weight_shift([float(2**54)]) is None


def test_extreme_costs_use_exact_arithmetic_without_priority_overflow():
    rdm = model(np.full((2,2), np.finfo(float).max))
    rdm.rare_matching(np.inf)
    assert rdm.assignment_diagnostics_['matched_control_count'] == 2
    assert rdm.assignment_diagnostics_['solver'] == 'exact_integer'
    assert int(rdm.assignment_diagnostics_['exact_distance_numerator']) == 2*int(np.finfo(float).max)


def test_costs_are_not_silently_narrowed():
    with pytest.raises(ValueError, match='exactly representable'):
        model(np.array([[2**53+1]], dtype=np.int64))
    with pytest.raises(ValueError, match='exactly representable'):
        model(np.array([[2**64-1]], dtype=np.uint64))
    assert model(np.array([[2**54]], dtype=np.int64)) is not None
    if np.dtype(np.longdouble).itemsize > 8:
        with pytest.raises(ValueError, match='up to float64'):
            model(np.ones((1,1), dtype=np.longdouble))


def check_mcf_precision_guard():
    for costs in (np.array([[1e-9]]), np.array([[0.,1e-20],[0.,2e-20]])):
        rdm = model(costs)
        rdm.rare_matching(np.inf, execution=ExecutionOptions(solver='mcf'))
        assert rdm.assignment_diagnostics_['matched_control_count'] == len(costs)
        assert rdm.assignment_diagnostics_['total_matching_distance'] == (1e-9 if len(costs)==1 else 1e-20)


def test_mcf_preserves_supplied_costs_without_fixed_scale_quantization():
    pytest.importorskip('ortools')
    root = Path(__file__).resolve().parents[1]
    script = ('from ortools.graph.python import min_cost_flow; import sys; '
              f'sys.path[:0] = {[str(root / "src"), str(root / "tests")]!r}; '
              'from test_precomputed import check_mcf_precision_guard; check_mcf_precision_guard()')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize('solver', ['hungarian', 'scipy_sparse'])
@pytest.mark.parametrize('costs', [np.array([[2e-20], [1e-20]]),
                                  np.array([[1.+np.spacing(1.)], [1.]])])
def test_incomplete_assignment_does_not_lose_costs_to_dummy_penalties(solver, costs):
    rdm = model(costs)
    result = rdm.rare_matching(np.inf, execution=ExecutionOptions(solver=solver))
    selected = result.loc[(result.treated == 1) & (result.n_matches > 0), 'id'].tolist()
    assert selected == ['t1']
    assert rdm.assignment_diagnostics_['solver'] == 'exact_integer'
    assert rdm.assignment_diagnostics_['total_matching_distance'] == costs[1, 0]


def test_precomputed_graph_export_rejects_lossy_shift():
    graph = model(np.array([[0.,1e-20],[0.,2e-20]])).return_graph(np.inf)
    with pytest.raises(ValueError, match='round supplied distances'):
        graph.for_scipy_sparse_matching()


@pytest.mark.parametrize('solver', ['scipy_sparse', 'hungarian'])
@pytest.mark.parametrize('ratio', [1, 2])
@pytest.mark.parametrize('coverage', [False, True])
def test_against_enumerated_full_eligible_objective(solver, ratio, coverage):
    from itertools import product
    for seed in range(12):
        rng = np.random.default_rng(seed)
        coefficients = rng.integers(0, 10, (3,4))
        eligible = rng.random((3,4)) > .35
        expected = None
        for assignment in product(range(-1,3), repeat=4):
            counts, total = [0,0,0], 0
            for control, case in enumerate(assignment):
                if case < 0:
                    continue
                if not eligible[case, control]:
                    break
                counts[case] += 1
                total += int(coefficients[case, control])
            else:
                if max(counts) > ratio:
                    continue
                objective = (-sum(c > 0 for c in counts) if coverage else 0,
                             -sum(counts), total)
                expected = objective if expected is None else min(expected, objective)
        for scale in (2.**-80, 1., 2.**40):
            costs = np.where(eligible, coefficients*scale, np.inf)
            rdm = model(costs)
            output = rdm.rare_matching(np.inf, ratio,
                matching=MatchingOptions(prioritize_coverage=coverage),
                execution=ExecutionOptions(solver=solver))
            diag = rdm.assignment_diagnostics_
            covered = int(((output.treated == 1) & (output.n_matches > 0)).sum())
            actual = (-covered if coverage else 0, -diag['matched_control_count'],
                      int(round(diag['total_matching_distance']/scale)))
            assert actual == expected


def test_exact_fallback_fails_closed_when_memory_budget_is_exceeded(monkeypatch):
    import rdmatcher.matching as matching
    monkeypatch.setattr(matching, '_EXACT_COST_FLOW_BUDGET_BYTES', 1)
    rdm = model(np.array([[2e-20],[1e-20]]))
    with pytest.raises(ValueError, match='no approximate assignment'):
        rdm.rare_matching(np.inf)
    assert rdm.matched_data is None


@pytest.mark.parametrize('solver', ['scipy_sparse', 'hungarian'])
@pytest.mark.parametrize('coverage', [False, True])
def test_mixed_binary_magnitudes_against_exact_fraction_oracle(solver, coverage):
    from fractions import Fraction
    from itertools import product
    for seed in range(8):
        rng = np.random.default_rng(seed)
        exponents = rng.choice([-1074,-200,-52,0,20,200,1000], size=(3,4))
        with np.errstate(under='ignore'):
            costs = np.ldexp(rng.integers(0,8,size=(3,4)).astype(float), exponents)
        eligible = rng.random((3,4)) > .35
        rational = [[Fraction.from_float(costs[i,j]) for j in range(4)] for i in range(3)]
        expected = None
        for assignment in product(range(-1,3), repeat=4):
            counts, total = [0]*3, Fraction(0)
            for control, case in enumerate(assignment):
                if case < 0:
                    continue
                if not eligible[case,control]:
                    break
                counts[case] += 1
                total += rational[case][control]
            else:
                if max(counts) <= 2:
                    objective = (-sum(c > 0 for c in counts) if coverage else 0, -sum(counts), total)
                    expected = objective if expected is None else min(expected, objective)
        rdm = model(np.where(eligible, costs, np.inf))
        output = rdm.rare_matching(np.inf, 2,
            matching=MatchingOptions(prioritize_coverage=coverage),
            execution=ExecutionOptions(solver=solver))
        total = Fraction(0)
        for _, group in output.dropna(subset=['match_group']).groupby('match_group'):
            case = int(group.loc[group.treated == 1, 'id'].iloc[0][1:])
            for control in group.loc[group.treated == 0, 'id']:
                total += rational[case][int(control[1:])]
        diag = rdm.assignment_diagnostics_
        covered = int(((output.treated == 1) & (output.n_matches > 0)).sum())
        assert (-covered if coverage else 0, -diag['matched_control_count'], total) == expected


def test_mcf_against_independent_enumerated_objectives():
    pytest.importorskip('ortools')
    root = Path(__file__).resolve().parents[1]
    script = ('from ortools.graph.python import min_cost_flow\nimport sys\n'
              f'sys.path[:0] = {[str(root / "src"), str(root / "tests")]!r}\n'
              'from test_precomputed import test_against_enumerated_full_eligible_objective as regular\n'
              'from test_precomputed import test_mixed_binary_magnitudes_against_exact_fraction_oracle as mixed\n'
              'for coverage in (False,True):\n'
              '    for ratio in (1,2): regular("mcf",ratio,coverage)\n'
              '    mixed("mcf",coverage)\n')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_unsupported_index_ranges_fail_before_large_allocations():
    matrix = csr_matrix((1, int(np.iinfo(np.int32).max)+1))
    with pytest.raises(ValueError, match='neighbor-index range'):
        RDMatcher(population(1,1), 'id', 'treated', [], [], distance_matrix=matrix,
                  case_ids=['t0'], control_ids=['c0'])
    with pytest.raises(ValueError, match='assignment-index range'):
        model(np.zeros((2,2))).rare_matching(np.inf, int(np.iinfo(np.int32).max))


def test_residual_certificate_rejects_a_suboptimal_feasible_assignment():
    from rdmatcher._optimality_certificate import verify_assignment
    edges = [dict(candidate_positions=np.array([0,1]), candidate_distances=np.array(costs))
             for costs in ([0., 1e-20], [0., 2e-20])]
    args = (edges, [1,1], [False,False], np.array([10,11]))
    assert not verify_assignment(*args, {0:[10],1:[11]}, np.inf)['verified']
    result = verify_assignment(*args, {0:[11],1:[10]}, np.inf)
    assert result['verified']
    assert result['method'] == 'exact_rational_residual_potentials'
    assert not verify_assignment(*args, {0:[10],1:[10]}, np.inf)['verified']


def test_bad_solver_result_is_recovered_and_certified(monkeypatch):
    import rdmatcher.matcher as allocation
    def bad_solver(**kwargs):
        ids = kwargs['all_control_indices']
        return {0:[int(ids[0])], 1:[int(ids[1])]}
    monkeypatch.setattr(allocation, 'solve_optimal_assignment_scipy_sparse', bad_solver)
    rdm = model(np.array([[0., .25], [0., .5]]))
    rdm.rare_matching(np.inf)
    assert rdm.assignment_diagnostics_['total_matching_distance'] == .25
    assert rdm.assignment_diagnostics_['optimality_verified']
    assert rdm.assignment_diagnostics_['solver_fallback_reason'] == 'exact_optimality_certificate_failed'


def test_certificate_does_not_trust_solver_working_costs(monkeypatch):
    import rdmatcher.matcher as allocation
    def bad_solver(**kwargs):
        # Make a wrong assignment appear cheap in the solver's private arrays.
        for row in kwargs['precomputed']:
            row['candidate_distances'][:] = 0
        ids=kwargs['all_control_indices']
        return {0:[int(ids[0])],1:[int(ids[1])]}
    monkeypatch.setattr(allocation,'solve_optimal_assignment_scipy_sparse',bad_solver)
    rdm=model(np.array([[0.,.25],[0.,.5]]))
    rdm.rare_matching(np.inf)
    assert rdm.assignment_diagnostics_['total_matching_distance'] == .25
    assert rdm.assignment_diagnostics_['optimality_verified']


def test_large_logical_demand_does_not_expand_impossible_slots():
    rdm = model(np.zeros((2,2)))
    result = rdm.rare_matching(np.inf, 10**9,
                               matching=MatchingOptions(prioritize_coverage=True))
    assert rdm.assignment_diagnostics_['matched_control_count'] == 2
    assert int(((result.treated == 1) & (result.n_matches > 0)).sum()) == 2
    assert rdm.assignment_diagnostics_['optimality_verified']


def test_certificate_memory_failure_returns_no_approximation(monkeypatch):
    import rdmatcher._optimality_certificate as certificate
    monkeypatch.setattr(certificate, '_CERTIFICATE_MEMORY_BUDGET_BYTES', 1)
    rdm = model(np.zeros((2,2)))
    with pytest.raises(ValueError, match='no assignment was returned'):
        rdm.rare_matching(np.inf)
    assert rdm.matched_data is None


@pytest.mark.parametrize('solver', ['scipy_sparse', 'hungarian'])
def test_against_independent_networkx_integer_solver(solver):
    nx = pytest.importorskip('networkx')
    from fractions import Fraction
    from math import lcm
    for seed in range(24):
        rng = np.random.default_rng(seed)
        n, m = int(rng.integers(4,15)), int(rng.integers(8,35))
        ratio, coverage = int(rng.integers(1,4)), bool(seed%2)
        values = rng.random((n,m)).astype(np.float32).astype(float)
        if seed%3 == 0:
            values = np.ldexp(rng.integers(0,8,(n,m)).astype(float),
                              rng.choice([-300,-52,-5,0,20,200],size=(n,m)))
        eligible = rng.random((n,m)) > .3
        costs = np.where(eligible, values, np.inf)
        fractions = {(i,j):Fraction.from_float(values[i,j]) for i,j in zip(*np.where(eligible))}
        denominator = lcm(*(value.denominator for value in fractions.values()))
        units = {edge:int(value*denominator) for edge,value in fractions.items()}
        slots = n*ratio
        extra = (slots+1)*(max(units.values(),default=0)+1)
        high = (slots+1)*(extra+max(units.values(),default=0)+1)
        graph = nx.DiGraph()
        graph.add_node('source',demand=-slots)
        graph.add_node('sink',demand=slots)
        for i in range(n):
            graph.add_edge('source',('case',i),capacity=ratio,weight=0)
            low_capacity = ratio-int(coverage)
            if low_capacity:
                graph.add_edge(('case',i),('low',i),capacity=low_capacity,weight=extra)
                graph.add_edge(('low',i),'sink',capacity=low_capacity,weight=0)
            if coverage:
                graph.add_edge(('case',i),('high',i),capacity=1,weight=high)
                graph.add_edge(('high',i),'sink',capacity=1,weight=0)
        for j in range(m):
            graph.add_edge(('control',j),'sink',capacity=1,weight=0)
        for (i,j),cost in units.items():
            graph.add_edge(('case',i),('control',j),capacity=1,weight=cost)
        _, flow = nx.network_simplex(graph)
        expected_pairs = [(i,j) for i,j in units if flow[('case',i)].get(('control',j),0)]
        expected = (-len({i for i,j in expected_pairs}) if coverage else 0,
                    -len(expected_pairs), sum((fractions[p] for p in expected_pairs),Fraction(0)))
        rdm = model(costs)
        output = rdm.rare_matching(np.inf,ratio,
             matching=MatchingOptions(prioritize_coverage=coverage),
             execution=ExecutionOptions(solver=solver))
        selected=[]
        for _,group in output.dropna(subset=['match_group']).groupby('match_group'):
            i=int(group.loc[group.treated == 1,'id'].iloc[0][1:])
            selected.extend((i,int(control[1:])) for control in group.loc[group.treated == 0,'id'])
        actual=(-len({i for i,j in selected}) if coverage else 0,-len(selected),
                sum((fractions[p] for p in selected),Fraction(0)))
        assert actual == expected
        assert rdm.assignment_diagnostics_['optimality_verified']


@pytest.mark.parametrize('order', ['C','F'])
def test_contiguous_validation_views_check_every_value(order):
    values=np.array([[0.,.5,.75],[.25,.125,1.]],order=order)
    values.setflags(write=False)
    rdm=model(values)
    assert rdm._precomputed_distances.matrix is values
    rdm.rare_matching(np.inf)
    assert rdm.assignment_diagnostics_['optimality_verified']
    for bad in (np.nan,-1.,-np.inf):
        invalid=np.array(values,order=order)
        invalid[-1,-1]=bad
        invalid.setflags(write=False)
        with pytest.raises(ValueError,match='Distances'):
            model(invalid)


@pytest.mark.parametrize('solver', ['scipy_sparse','hungarian'])
def test_compiled_solver_transfers_when_result_is_independently_certified(monkeypatch,solver):
    import rdmatcher.matching as assignment
    def unexpected_fallback(*args,**kwargs):
        raise AssertionError('Conservative arithmetic bound unnecessarily forced Python flow')
    monkeypatch.setattr(assignment,'_solve_exact_cost_flow',unexpected_fallback)
    costs=np.array([[0.,2.**50],[0.,2.**50+1]])
    rdm=model(costs)
    rdm.rare_matching(np.inf,execution=ExecutionOptions(solver=solver))
    assert rdm.assignment_diagnostics_['solver']==solver
    assert rdm.assignment_diagnostics_['optimality_verified']
    assert rdm.assignment_diagnostics_['total_matching_distance']==2.**50


def test_shared_cache_stops_at_precomputed_padding(monkeypatch):
    from types import SimpleNamespace
    from rdmatcher._general_retrieval import DemandNeighbors
    from rdmatcher._precomputed import MatrixPrefixQuery
    supplied=np.full((2,10),np.inf,dtype=np.float64)
    supplied[0,0]=0.1
    rdm=model(supplied)
    owner=SimpleNamespace(precomputed_distances=rdm._precomputed_distances,
        threshold=0.2,ps_eligible_sets=None,n_jobs=1,distance_metric='precomputed',
        distance_dtype=np.float64,X_exposed=np.empty((2,0)),X_control=np.empty((10,0)))
    calls=[]
    original=MatrixPrefixQuery._compute_prefix
    def observed(self,row,count):
        calls.append((int(row),int(count)))
        return original(self,row,count)
    monkeypatch.setattr(MatrixPrefixQuery,'_compute_prefix',observed)
    cache=DemandNeighbors(owner,8,2)
    cache.expand(2, owner.threshold)
    assert calls == [(0, 1), (1, 1), (0, 2)]
    np.testing.assert_array_equal(cache.loaded, [2, 1])
    assert cache.stats['exhausted_eligibility_rows'] == 2
    assert not hasattr(cache.provider,'rows')
    assert cache.provider.stats['stored_pool_entries']==0
    assert cache.distances[0][0]==supplied[0,0]
    assert cache.distances[0].dtype==np.float64
    np.testing.assert_array_equal(cache.indices[0],np.array([0, -1]))
    np.testing.assert_array_equal(cache.indices[1],np.array([-1]))
