import numpy as np
import pytest

from rdmatcher.mahalanobis import MahalanobisKNN


def test_cdist_raises_on_yref_with_nans():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(10, 3))
    Q = rng.normal(size=(2, 3))
    Y = X.copy()
    Y[0, 0] = np.nan

    model = MahalanobisKNN()
    model.fit(X)

    with pytest.raises(ValueError, match="Y_ref contains missing values"):
        _ = model.cdist(Q, Y_ref=Y)


def test_cdist_accepts_clean_yref():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(12, 4))
    Q = rng.normal(size=(3, 4))
    Y = rng.normal(size=(8, 4))

    model = MahalanobisKNN()
    model.fit(X)
    d = model.cdist(Q, Y_ref=Y)
    assert d.shape == (Q.shape[0], Y.shape[0])
