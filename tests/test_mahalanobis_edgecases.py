import numpy as np
import pandas as pd
import pytest

from rdmatcher.mahalanobis import MahalanobisKNN


def test_dataframe_vs_ndarray_equivalence():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(40, 4))
    Q = rng.normal(size=(7, 4))

    model_arr = MahalanobisKNN()
    model_arr.fit(X)
    d_arr = model_arr.cdist(Q)

    X_df = pd.DataFrame(X, columns=[f"c{i}" for i in range(X.shape[1])])
    Q_df = pd.DataFrame(Q, columns=[f"c{i}" for i in range(Q.shape[1])])
    model_df = MahalanobisKNN()
    model_df.fit(X_df)
    d_df = model_df.cdist(Q_df)

    assert np.allclose(d_arr, d_df, rtol=1e-6, atol=1e-8)


def test_drop_rows_and_index_mapping():
    X = np.array([
        [0.0, 1.0],
        [np.nan, 2.0],
        [1.0, 0.0],
        [3.0, np.nan],
    ])
    model = MahalanobisKNN()
    model.fit(X)

    # only rows 0 and 2 are complete
    assert model.n_samples_fit_ == 2
    assert list(model._ref_index_map) == [0, 2]

    queries = np.array([[0.0, 1.0], [np.nan, 0.0]])
    d, idx = model.kneighbors(queries, n_neighbors=1)
    # first query valid, second query contains NaN -> NaN distance and -1 index
    assert np.isfinite(d[0, 0])
    assert np.isnan(d[1, 0])
    assert idx[0, 0] >= 0
    assert idx[1, 0] == -1


def test_singular_covariance_regularization():
    # second feature is collinear with the first
    X = np.vstack([np.linspace(0, 1, 10), np.linspace(0, 2, 10)]).T
    model = MahalanobisKNN()
    # should not raise
    model.fit(X)
    Q = X[:3]
    d = model.cdist(Q)
    assert np.all(np.isfinite(d))


def test_tie_breaking_deterministic():
    # two identical rows (indices 0 and 1). Query equals that row so distances equal.
    X = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 1.0]])
    model = MahalanobisKNN()
    model.fit(X)
    q = np.array([[0.0, 0.0]])
    d, idx = model.kneighbors(q, n_neighbors=2)
    # indices should be in ascending original index order (stable tie-break)
    assert idx[0, 0] < idx[0, 1]


def test_n_neighbors_greater_than_n_samples():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(3, 3))
    Q = rng.normal(size=(2, 3))
    model = MahalanobisKNN()
    model.fit(X)
    d, idx = model.kneighbors(Q, n_neighbors=5)
    # should return min(n_neighbors, n_ref) columns
    assert d.shape[1] == 3
    assert idx.shape[1] == 3
