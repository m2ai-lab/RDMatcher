import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from rdmatcher.matching import matching_diagnostics


def test_matching_diagnostics_numeric_plot_does_not_crash(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)

    df = pd.DataFrame({
        "exposure_status": [1, 1, 0, 0, 1, 0],
        "age_at_first_visit": [10, 12, 14, 16, 11, 15],
        "sex": ["F", "F", "M", "M", "F", "M"],
    })

    summary = matching_diagnostics(
        matched_data=df,
        exposure_status="exposure_status",
        features_numeric=["age_at_first_visit"],
        features_categorical=["sex"],
        plot_features=True,
    )

    assert summary is not None
