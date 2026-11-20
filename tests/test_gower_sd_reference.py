from rdmatcher import MatchingOptions, ExecutionOptions
import pandas as pd
import pytest
from rdmatcher import RDMatcher

def _cohort():
    return pd.DataFrame({'patient_id': [1, 2, 3, 4, 5, 6], 'exposure_status': [1, 1, 0, 0, 0, 0], 'age': [80.0, 82.0, 20.0, 22.0, 24.0, 26.0], 'bmi': [34.0, 35.0, 25.0, 26.0, 27.0, 28.0], 'sex': ['F', 'M', 'F', 'M', 'F', 'M']})

def _matcher():
    return RDMatcher(pop_df=_cohort(), patient_id_col='patient_id', exposure_status='exposure_status', features_numeric=['age', 'bmi'], features_categorical=['sex'], process_features=False, onehot=False, debug=False, log_to_console=False)

def test_build_gower_sd_weights_default_is_per_feature_calibrated():
    matcher = _matcher()
    matching_data, match_features, _, _ = matcher._build_multi_matching_view('gower')
    weights = matcher._build_gower_sd_weights(matching_data=matching_data, match_features=match_features, gower_sd_reference='controls', gower_sd_weights_mult=1.96)
    controls = matching_data.loc[matching_data.exposure_status.eq(0)]
    expected_age = (controls.age.max() - controls.age.min()) / (1.96 * controls.age.std(ddof=1))
    assert weights['age'] == pytest.approx(expected_age)
    assert weights['sex'] == pytest.approx(1.0)

def test_build_gower_sd_weights_pooled_returns_same_feature_keys():
    matcher = _matcher()
    matching_data, match_features, _, _ = matcher._build_multi_matching_view('gower')
    weights = matcher._build_gower_sd_weights(matching_data=matching_data, match_features=match_features, gower_sd_reference='pooled', gower_sd_weights_mult=1.96)
    assert weights
    assert set(weights) == set(match_features)
    assert weights['sex'] == pytest.approx(1.0)

def test_rare_matching_rejects_invalid_gower_sd_reference():
    matcher = _matcher()
    with pytest.raises(ValueError, match='gower_sd_reference'):
        matcher.rare_matching(threshold=0.5, n_neighbors=1, distance_metric='gower', allocation='hybrid', matching=MatchingOptions(method='multi', gower_sd_weights=True, gower_sd_reference='invalid'))

def test_rare_matching_ignores_gower_sd_reference_when_sd_weights_disabled():
    matcher = _matcher()
    matched = matcher.rare_matching(threshold=0.5, n_neighbors=1, distance_metric='gower', allocation='hybrid', matching=MatchingOptions(method='multi', gower_sd_weights=False, gower_sd_reference='pooled'))
    assert not matched.empty
    assert getattr(matcher, 'gower_sd_reference_', None) is None

def test_rare_matching_pooled_gower_sd_weights_runs_and_sets_state():
    matcher = _matcher()
    matched = matcher.rare_matching(threshold=0.5, n_neighbors=1, distance_metric='gower', allocation='hybrid', matching=MatchingOptions(method='multi', gower_sd_weights=True, gower_sd_reference='pooled'))
    assert not matched.empty
    assert getattr(matcher, 'gower_weights_', None)
    assert matcher.gower_sd_reference_ == 'pooled'
