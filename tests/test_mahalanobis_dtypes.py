import numpy as np

from rdmatcher.mahalanobis import MahalanobisKNN


def test_output_dtypes():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(20, 5))
    Q = rng.normal(size=(4, 5))

    model = MahalanobisKNN()
    model.fit(X)
    d = model.cdist(Q)
    assert d.dtype == "float32"

    d_knn, idx = model.kneighbors(Q, n_neighbors=3)
    assert d_knn.dtype == "float32"
    assert idx.dtype == "int32"
