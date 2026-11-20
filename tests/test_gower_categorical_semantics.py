from rdmatcher import MatchingOptions, ExecutionOptions
import numpy as np
import pandas as pd
from rdmatcher import RDMatcher
from rdmatcher.distance import GowerKNN

def test_declared_integer_categories_are_nominal_not_ordinal():
    controls = pd.DataFrame({'cat': [1, 4]})
    query = pd.DataFrame({'cat': [2]})
    nominal = GowerKNN(cat_features=['cat']).fit(controls)
    nominal_dist = nominal.cdist(query, controls)
    inferred_numeric = GowerKNN().fit(controls)
    numeric_dist = inferred_numeric.cdist(query, controls)
    assert np.allclose(nominal_dist, [[1.0, 1.0]])
    assert np.allclose(numeric_dist, [[1.0 / 3.0, 2.0 / 3.0]])

def test_mixed_gower_distance_uses_weighted_numeric_and_categorical_terms():
    controls = pd.DataFrame({'num': [0.0, 10.0], 'cat': [1, 2]})
    query = pd.DataFrame({'num': [5.0], 'cat': [1]})
    model = GowerKNN(weights=[1.0, 1.0], cat_features=['cat']).fit(controls)
    dist = model.cdist(query, controls)
    assert np.allclose(dist, [[0.25, 0.75]])

def test_missing_values_are_excluded_from_pairwise_denominator():
    controls = pd.DataFrame({'num': [0.0, 10.0], 'cat': [1, np.nan]})
    query = pd.DataFrame({'num': [5.0], 'cat': [1.0]})
    model = GowerKNN(weights=[1.0, 1.0], cat_features=['cat']).fit(controls)
    dist = model.cdist(query, controls)
    assert np.allclose(dist, [[0.25, 0.5]])

def test_kneighbors_and_cdist_agree_with_declared_categories():
    controls = pd.DataFrame({'num': [0.0, 10.0, 5.0, 5.0], 'cat': [1, 1, 2, 4]})
    query = pd.DataFrame({'num': [5.0, 0.0], 'cat': [2, 4]})
    model = GowerKNN(weights=[1.0, 1.0], cat_features=['cat']).fit(controls)
    cdist = model.cdist(query, controls)
    knn_dist, knn_idx = model.kneighbors(query, n_neighbors=2)
    returned_dist = np.take_along_axis(cdist, knn_idx, axis=1)
    kth_dist = np.sort(cdist, axis=1)[:, 1:2]
    assert np.allclose(knn_dist, returned_dist)
    assert np.all(knn_dist <= kth_dist + 1e-07)
    for row in knn_idx:
        assert len(set(row.tolist())) == len(row)

def test_rdmatcher_defaults_to_declared_categorical_semantics_for_gower():
    df = pd.DataFrame({'patient_id': [1, 2, 3], 'exposure_status': [1, 0, 0], 'cat': [2, 1, 4]})
    matcher = RDMatcher(pop_df=df, patient_id_col='patient_id', exposure_status='exposure_status', features_numeric=[], features_categorical=['cat'], process_features=False, onehot=False, debug=False, log_to_console=False)
    matched = matcher.rare_matching(threshold=0.5, n_neighbors=1, distance_metric='gower')
    assert matched['match_group'].notna().sum() == 0
