import pandas as pd
import numpy as np

from rdmatcher.matcher import Matcher
from rdmatcher.mahalanobis import MahalanobisKNN


def test_matcher_constructs_mahalanobis_model():
    # simple numeric dataset: 3 controls, 1 treated
    df = pd.DataFrame({
        "patient_id": [1, 2, 3, 4],
        "exposure_status": [1, 0, 0, 0],
        "x1": [0.0, 0.1, 1.0, -0.5],
        "x2": [1.0, 0.9, 0.1, -1.0],
    })

    matcher = Matcher(df=df, exposure_status="exposure_status", patient_id="patient_id", distance_metric="mahalanobis", threshold=10.0, n_neighbors=1, n_jobs=1)

    assert hasattr(matcher, 'nbrs_model')
    assert isinstance(matcher.nbrs_model, MahalanobisKNN)

    # sanity kneighbors call
    q = matcher.X_exposed
    d, idx = matcher.nbrs_model.kneighbors(q, n_neighbors=1)
    assert d.shape[0] == q.shape[0]
    assert idx.shape[1] == 1
