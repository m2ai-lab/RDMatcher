from rdmatcher import MatchingOptions, ExecutionOptions
import pandas as pd
import numpy as np
from rdmatcher.RDMatcher import RDMatcher

def make_cohort():
    rows = []
    rows.append({'patient_id': 1, 'exposure_status': 1, 'age': 0.0})
    rows.append({'patient_id': 2, 'exposure_status': 1, 'age': 1.0})
    rows.append({'patient_id': 3, 'exposure_status': 1, 'age': 2.0})
    rows.append({'patient_id': 4, 'exposure_status': 0, 'age': 0.1})
    rows.append({'patient_id': 5, 'exposure_status': 0, 'age': 10.0})
    return pd.DataFrame(rows)

def test_return_unmatched_included_and_excluded():
    cohort = make_cohort()
    matcher = RDMatcher(pop_df=cohort, patient_id_col='patient_id', exposure_status='exposure_status', features_numeric=['age'], features_categorical=[], process_features=False, onehot=False, debug=False)
    res_incl = matcher.rare_matching(threshold=0.5, n_neighbors=1, distance_metric='euclidean', matching=MatchingOptions(method='multi', return_unmatched_cases=True))
    assert 3 in res_incl['patient_id'].values
    matcher2 = RDMatcher(pop_df=cohort, patient_id_col='patient_id', exposure_status='exposure_status', features_numeric=['age'], features_categorical=[], process_features=False, onehot=False, debug=False)
    res_excl = matcher2.rare_matching(threshold=0.5, n_neighbors=1, distance_metric='euclidean', matching=MatchingOptions(method='multi', return_unmatched_cases=False))
    assert 3 not in res_excl['patient_id'].values
