import inspect

import numpy as np
import pandas as pd
import pytest

from rdmatcher import RDMatcher, ExecutionOptions, MatchRequest, MatchingOptions


def model():
    population = pd.DataFrame({
        'record': ['a', 'b', 'c', 'd', 'e'],
        'treated': [1, 1, 0, 0, 0],
        'x': [0., 3., .1, 2.9, 8.],
    })
    return RDMatcher(population, 'record', 'treated', ['x'], [])


def test_public_matching_contract():
    signature = inspect.signature(RDMatcher.match)
    assert list(signature.parameters) == [
        'self', 'threshold', 'ratio', 'n_neighbors', 'distance_metric',
        'allocation', 'matching', 'execution', 'request',
    ]
    assert signature.parameters['allocation'].default == 'optimal'
    assert all(p.kind != inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
    result = model().match(np.inf, ratio=1, distance_metric='euclidean')
    assert isinstance(result, pd.DataFrame)
    assert result.loc[result.treated == 1, 'n_matches'].tolist() == [1, 1]
    legacy = model().rare_matching(np.inf, n_neighbors=1, distance_metric='euclidean')
    assert legacy.equals(result)


def test_request_and_precomputed_constructor():
    request = MatchRequest(
        threshold=np.inf, ratio=1, distance_metric='euclidean',
        matching=MatchingOptions(), execution=ExecutionOptions(),
    )
    positional = model().match(request)
    keyword = model().match(request=request)
    assert positional.equals(keyword)
    graph = model().return_graph(request)
    assert graph.nnz == 2

    population = pd.DataFrame({'record': ['t', 'c'], 'treated': [1, 0]})
    precomputed = RDMatcher.from_distance_matrix(
        population, 'record', 'treated', np.array([[0.25]]),
        case_ids=['t'], control_ids=['c'],
    )
    matched = precomputed.match(np.inf)
    assert matched.loc[matched.treated.eq(0), 'match_distance'].iloc[0] == 0.25


@pytest.mark.parametrize('argument', [
    'candidate_retrieval', 'candidate_cache_max_gb', 'k_candidates',
    'global_optimal', 'competitive_match', 'return_matched_data', 'diagnostics',
])
def test_removed_arguments_raise(argument):
    with pytest.raises(TypeError, match='unexpected keyword'):
        model().rare_matching(np.inf, **{argument: True})


@pytest.mark.parametrize('factory,settings', [
    (ExecutionOptions, {'batch_size': 0}),
    (ExecutionOptions, {'n_jobs': 0}),
    (ExecutionOptions, {'solver': 'unknown'}),
    (ExecutionOptions, {'streaming': 'unknown'}),
    (ExecutionOptions, {'memory_limit_gb': np.nan}),
    (MatchingOptions, {'method': 'unknown'}),
    (MatchingOptions, {'ps_caliper': -1.}),
    (MatchingOptions, {'gower_sd_weights': True, 'gower_weights': {'x': 1.}}),
])
def test_invalid_settings_raise(factory, settings):
    with pytest.raises(ValueError):
        factory(**settings)


@pytest.mark.parametrize('settings', [
    {'allocation': 'unknown'}, {'ratio': 0},
    {'ratio': 1.5}, {'threshold': -1.}, {'threshold': np.nan},
])
def test_invalid_matching_arguments_raise(settings):
    with pytest.raises(ValueError):
        model().match(**{'threshold': np.inf, **settings})


def test_ratio_and_legacy_n_neighbors_cannot_be_combined():
    with pytest.raises(TypeError, match='either ratio or legacy n_neighbors'):
        model().match(np.inf, ratio=2, n_neighbors=2)


def test_graph_and_assignment_share_restriction():
    graph_model = model()
    graph_model.return_graph(np.inf, distance_metric='euclidean')
    graph_diagnostics = graph_model.assignment_diagnostics_
    assignment_model = model()
    assignment_model.rare_matching(np.inf, distance_metric='euclidean')
    assignment_diagnostics = assignment_model.assignment_diagnostics_
    assert graph_diagnostics['initial_candidate_graph_sha256'] == assignment_diagnostics['initial_candidate_graph_sha256']
