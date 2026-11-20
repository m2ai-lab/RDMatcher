import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from rdmatcher.RDMatcher import RDMatcher


def test_plot_propensity_coverage_uses_seaborn_without_option_error(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)

    df = pd.DataFrame({
        "patient_id": [1, 2, 3, 4, 5, 6],
        "exposure_status": [1, 1, 0, 0, 1, 0],
        "age_at_first_visit": [20, 25, 30, 35, 40, 45],
        "person_time": [1, 2, 3, 4, 5, 6],
        "encounters": [0, 1, 2, 3, 4, 5],
        "sex": ["F", "F", "M", "M", "F", "M"],
        "race_ethnicity": ["A", "A", "B", "B", "A", "B"],
    })

    matcher = RDMatcher(
        pop_df=df,
        patient_id_col="patient_id",
        exposure_status="exposure_status",
        features_numeric=["age_at_first_visit", "person_time", "encounters"],
        features_categorical=["sex", "race_ethnicity"],
        process_features=False,
        onehot=False,
        debug=False,
        log_to_console=False,
        log_file=None,
    )

    matcher.fit_propensity_model(formula="age_at_first_visit + sex + race_ethnicity", random_state=404)
    fig, axes = matcher.plot_propensity_coverage(compare_matching=False)
    assert fig is not None
    assert len(axes) == 1
