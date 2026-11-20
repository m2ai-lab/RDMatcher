import ast
from itertools import product
import math
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from rdmatcher.matcher import Matcher
from kernel_fixtures import frame, make

def function(path, name):
    tree = ast.parse(Path(path).read_text())
    result = next((n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name))
    if result.body and isinstance(result.body[0], ast.Expr) and isinstance(result.body[0].value, ast.Constant) and isinstance(result.body[0].value.value, str):
        result.body = result.body[1:]
    return result

def test_scalar_reference_safe_rule_is_pinned():
    import hashlib
    import json
    import rdmatcher.matcher
    expected = json.loads(Path(__file__).with_name('kernel_structure.json').read_text())
    after = function(Path(__file__).with_name('reference_prefilter.py'), '_prefilter_candidates')
    loop = next(n for n in ast.walk(after) if isinstance(n, ast.While))
    # The shared-cache fast-forward is independently checked against ordinary
    # single-round expansion. Pin the remaining owner/safe-count logic here.
    loop.body = [statement for statement in loop.body
                 if not (isinstance(statement, ast.Assign)
                         and any(isinstance(target, ast.Name) and target.id == 'advanced'
                                 for target in statement.targets))
                 and not (isinstance(statement, ast.If)
                          and ast.unparse(statement.test) == 'advanced')]
    for node in ast.walk(loop):
        if isinstance(node, ast.If) and ast.unparse(node.test) == 'lazy is not None':
            node.body = [statement for statement in node.body
                         if not (isinstance(statement, ast.Assign)
                                 and any(isinstance(target, ast.Name) and target.id == 'advanced'
                                         for target in statement.targets))
                         and not (isinstance(statement, ast.If)
                                  and ast.unparse(statement.test) == 'advanced')]
    def stable(node):
        if isinstance(node, ast.AST):
            return [type(node).__name__, {key: stable(value) for key, value in ast.iter_fields(node) if value is not None and value != []}]
        if isinstance(node, list):
            return [stable(value) for value in node]
        return node
    for name, node in [('loop', loop)]:
        assert hashlib.sha256(json.dumps(stable(node), sort_keys=True).encode()).hexdigest() == expected[name]


class OrderedOracle:

    def __init__(self, costs):
        self.costs = costs
        self.calls = 0

    def kneighbors(self, query, n_neighbors):
        self.calls += 1
        rows = np.asarray(query)[:, 0].astype(int)
        values = self.costs[rows]
        positions = np.broadcast_to(np.arange(values.shape[1]), values.shape)
        order = np.lexsort((positions, values), axis=1)[:, :n_neighbors]
        return (np.take_along_axis(values, order, axis=1), order)

def models(cases, controls, ratio, threshold):
    pairs = []
    for mode in ('first', 'repeat'):
        m = make(frame(cases=cases, controls=controls), 'euclidean', mode, n_neighbors=ratio)
        m.threshold = threshold
        m.X_exposed = np.arange(cases)[:, None]
        m.X_control = np.arange(controls)[:, None]
        m.nbrs_model = OrderedOracle(np.zeros((cases, controls), dtype=np.float32))
        pairs.append(m)
    return pairs

def objective(model, coverage):
    selected = model.match_distances_
    return (-len({i for i, j in selected}) if coverage else 0, -len(selected), math.fsum(selected.values()))

def full_reference(costs, ratio, threshold, coverage):
    cases, controls = costs.shape
    slots = cases * ratio
    expanded = np.repeat(costs, ratio, axis=0).astype(np.float64)
    expanded[expanded > threshold] = np.inf
    matrix = np.full((slots, controls + slots), np.inf)
    matrix[:, :controls] = expanded
    penalties = np.full(slots, 1000000.0)
    if coverage:
        penalties[::ratio] = 1000000.0 * (slots + 1)
    matrix[np.arange(slots), controls + np.arange(slots)] = penalties
    ii, jj = linear_sum_assignment(matrix)
    real = jj < controls
    rows = ii[real] // ratio
    cols = jj[real]
    return (-len(set(rows)) if coverage else 0, -int(real.sum()), math.fsum((float(costs[i, j]) for i, j in zip(rows, cols))))

def certify(model):
    c = len(model.X_exposed)
    r = model.n_neighbors
    cap = min(c * r, len(model.X_control))
    d, ids = model.nbrs_model.kneighbors(model.X_exposed, n_neighbors=cap)
    kept = []
    for i, h in enumerate(model.candidate_horizons_):
        active = ids[i, :h]
        valid = d[i, :h] <= model.threshold
        kept.append(active[valid])
    degree = np.bincount(np.concatenate(kept), minlength=len(model.X_control))
    for i, active in enumerate(kept):
        singleton = np.count_nonzero(degree[active] == 1) >= r
        complete = len(active) == np.count_nonzero(model.nbrs_model.costs[i] <= model.threshold)
        cap_bound = len(active) >= c * r
        assert singleton or complete or cap_bound

def compare(costs, ratio, threshold, coverage, pair=None):
    if pair is None:
        pair = models(*costs.shape, ratio, threshold)
    for model in pair:
        model.nbrs_model.costs = costs
        model.match(global_optimal=True, competitive_match=False, safe_matches=ratio, prioritize_coverage=coverage, solver='scipy_sparse', batch_size=5)
    left, right = pair
    assert left.assignment_diagnostics_['initial_candidate_graph_sha256'] == right.assignment_diagnostics_['initial_candidate_graph_sha256']
    assert np.array_equal(left.candidate_horizons_, right.candidate_horizons_)
    expected = full_reference(costs, ratio, threshold, coverage)
    assert objective(left, coverage) == objective(right, coverage) == expected
    certify(right)

def test_exhaustive_small_ordered_graphs():
    count = 0
    for ratio in (1, 2, 3):
        for threshold in (1.0, np.inf):
            for coverage in (False, True):
                pair = models(2, 3, ratio, threshold)
                for values in product((0.0, 1.0, 2.0), repeat=6):
                    compare(np.asarray(values, dtype=np.float32).reshape(2, 3), ratio, threshold, coverage, pair)
                    count += 1
    assert count == 8748

@pytest.mark.parametrize('ratio', [1, 2, 3])
def test_growing_prefixes_shared_reservoir_and_full_optimum(ratio):
    c, n = (20, 90)
    values = (np.arange(n)[None, :] * (c + 1) + np.arange(c)[:, None]).astype(np.float32)
    pair = models(c, n, ratio, np.inf)
    compare(values, ratio, np.inf, True, pair)
    assert pair[1].nbrs_model.calls > 1

@pytest.mark.parametrize('metric', ['euclidean', 'cosine', 'gower', 'mahalanobis'])
@pytest.mark.parametrize('threshold', [0.0, 0.2, np.inf])
@pytest.mark.parametrize('fuzzy', [False, True])
def test_real_backends_thresholds_fuzzy_and_ties(metric, threshold, fuzzy):
    df = frame(cases=25, controls=90, tied=True)
    df.loc[25:34, list('abcd')] = df.loc[:9, list('abcd')].to_numpy()
    result = []
    for mode in ('first', 'repeat'):
        m = make(df, metric, mode)
        m.threshold = threshold
        m.match(global_optimal=True, competitive_match=False, fuzzy_threshold=fuzzy, fuzzy_threshold_limit=threshold + 0.1, prioritize_coverage=True, batch_size=5)
        result.append(m)
    a, b = result
    assert a.assignment_diagnostics_['initial_candidate_graph_sha256'] == b.assignment_diagnostics_['initial_candidate_graph_sha256']
    assert np.array_equal(a.candidate_horizons_, b.candidate_horizons_)
    assert objective(a, True) == objective(b, True)

@pytest.mark.parametrize('metric', ['gower', 'mahalanobis'])
def test_eligibility_subsets_include_empty_and_incomplete_lists(metric):
    df = frame(cases=25, controls=90)
    rng = np.random.default_rng(7)
    eligible = [np.sort(rng.choice(90, size=i * 7 % 60, replace=False)) for i in range(25)]
    result = []
    for mode in ('first', 'repeat'):
        m = make(df, metric, mode, ps_eligible_sets=eligible, n_neighbors=2)
        m.match(global_optimal=True, competitive_match=False, prioritize_coverage=True, batch_size=5)
        result.append(m)
    assert result[0].assignment_diagnostics_['initial_candidate_graph_sha256'] == result[1].assignment_diagnostics_['initial_candidate_graph_sha256']
    assert objective(result[0], True) == objective(result[1], True)

def test_optimal_rejects_an_uncertified_small_cap():
    costs = np.asarray([[1.0, 2.0], [1.0, 100.0]], dtype=np.float32)
    m = models(2, 2, 1, np.inf)[1]
    m.nbrs_model.costs = costs
    with pytest.raises(ValueError, match='does not certify'):
        m.match(k_candidates=1, global_optimal=True, competitive_match=False)
    m.match(global_optimal=True, competitive_match=False)
    assert objective(m, False) == full_reference(costs, 1, np.inf, False)
    assert m.assignment_diagnostics_['preservation_condition_satisfied']
