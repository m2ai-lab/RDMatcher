from rdmatcher import MatchingOptions, ExecutionOptions
import logging
import pandas as pd
from rdmatcher import RDMatcher

def _cohort():
    return pd.DataFrame({'patient_id': [1, 2, 3, 4, 5, 6], 'exposure_status': [1, 1, 1, 0, 0, 0], 'age_at_first_visit': [30.0, 45.0, 60.0, 31.0, 46.0, 61.0], 'person_time': [10.0, 20.0, 30.0, 11.0, 21.0, 31.0], 'encounters': [1.0, 2.0, 3.0, 1.0, 2.0, 3.0], 'sex': ['Female', 'Male', 'Female', 'Female', 'Male', 'Female'], 'race_ethnicity': ['A', 'B', 'A', 'A', 'B', 'A']})

def _build_onehot_matcher(onehot_drop='first'):
    return RDMatcher(pop_df=_cohort(), patient_id_col='patient_id', exposure_status='exposure_status', features_numeric=['age_at_first_visit', 'person_time', 'encounters'], features_categorical=['sex', 'race_ethnicity'], process_features=False, onehot=True, onehot_drop=onehot_drop, debug=False, log_to_console=False)

def test_onehot_applies_to_mahalanobis_without_scaling():
    matcher = _build_onehot_matcher()
    matched = matcher.rare_matching(threshold=2.0, n_neighbors=1, distance_metric='mahalanobis', allocation='hybrid', matching=MatchingOptions(method='multi'))
    assert not matched.empty
    assert matched['match_group'].notna().any()
    assert 'sex_Male' in matcher.matching_data_.columns or 'sex_Female' in matcher.matching_data_.columns
    assert 'propensity_logit' not in matcher.matching_data_.columns

def test_onehot_applies_to_euclidean_and_cosine():
    for metric in ('euclidean', 'cosine'):
        matcher = _build_onehot_matcher()
        matched = matcher.rare_matching(threshold=2.0, n_neighbors=1, distance_metric=metric, allocation='hybrid', matching=MatchingOptions(method='multi'))
        assert not matched.empty
        assert matched['match_group'].notna().any()
        assert 'propensity_logit' not in matcher.matching_data_.columns

def test_gower_warns_when_onehot_is_enabled(caplog):
    matcher = _build_onehot_matcher()
    matcher.logger.propagate = True
    with caplog.at_level(logging.WARNING, logger='rdmatcher'):
        matched = matcher.rare_matching(threshold=2.0, n_neighbors=1, distance_metric='gower', allocation='hybrid', matching=MatchingOptions(method='multi'))
    assert not matched.empty
    assert any(('one-hot encoding is enabled' in record.message.lower() for record in caplog.records))

def test_propensity_scores_do_not_leak_into_non_propensity_matching():
    matcher = _build_onehot_matcher()
    matcher.fit_propensity_model()
    matched = matcher.rare_matching(threshold=2.0, n_neighbors=1, distance_metric='mahalanobis', allocation='hybrid', matching=MatchingOptions(method='multi'))
    assert not matched.empty
    assert 'propensity_logit' not in matcher.matching_data_.columns

def test_ps_hybrid_uses_fitted_propensity_scores_after_propensity_fit():
    matcher = _build_onehot_matcher()
    matcher.fit_propensity_model(formula='age_at_first_visit + person_time + encounters + sex + race_ethnicity', random_state=404)
    matched = matcher.rare_matching(threshold=0.1, n_neighbors=1, distance_metric='gower', allocation='hybrid', matching=MatchingOptions(method='multi', ps_hybrid=True, ps_caliper=0.2, ps_caliper_strict=True))
    assert not matched.empty
    assert 'propensity_logit' not in matcher.matching_data_.columns
    assert getattr(matcher, 'matching_candidate_diagnostics', None) is not None

def test_ps_hybrid_works_for_mahalanobis_without_leakage():
    matcher = _build_onehot_matcher()
    matcher.fit_propensity_model(formula='age_at_first_visit + person_time + encounters + sex + race_ethnicity', random_state=404)
    matched = matcher.rare_matching(threshold=0.1, n_neighbors=1, distance_metric='mahalanobis', allocation='hybrid', matching=MatchingOptions(method='multi', ps_hybrid=True, ps_caliper=0.2, ps_caliper_strict=True))
    assert not matched.empty
    assert 'propensity_logit' not in matcher.matching_data_.columns
    assert getattr(matcher, 'matching_candidate_diagnostics', None) is not None
