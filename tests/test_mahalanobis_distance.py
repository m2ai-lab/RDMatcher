import numpy as np
import pandas as pd
from rdmatcher.mahalanobis import MahalanobisKNN
from scipy.spatial.distance import cdist


def test_mahalanobis_matches_scipy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 5))
    Q = rng.normal(size=(10, 5))

    model = MahalanobisKNN()
    model.fit(X)

    d_my = model.cdist(Q)
    d_scipy = cdist(Q, X, metric="mahalanobis", VI=model.precision_)

    assert np.allclose(d_my, d_scipy, rtol=1e-5, atol=1e-6)


def test_kneighbors_agrees_with_pairwise():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 3))
    Q = rng.normal(size=(6, 3))

    model = MahalanobisKNN()
    model.fit(X)

    dists = model.cdist(Q)
    d_knn, idx = model.kneighbors(Q, n_neighbors=3)

    # brute force check: sort each row
    for i in range(Q.shape[0]):
        order = np.argsort(dists[i])[:3]
        assert list(order) == list(idx[i])
