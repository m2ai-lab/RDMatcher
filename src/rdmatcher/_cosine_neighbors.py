"""Cosine neighbor search with fitted-reference normalization reuse."""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils import gen_even_slices
from joblib import effective_n_jobs, Parallel, delayed

class CosineNeighbors(BaseEstimator):
    """Normalize controls once and select cosine neighbors by partial sorting."""

    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        self._fit_X = check_array(X, dtype=[np.float64, np.float32], order='C')
        self.n_samples_fit_, self.n_features_in_ = self._fit_X.shape
        workers = effective_n_jobs(self.n_jobs)
        if workers == 1:
            self._reference = normalize(self._fit_X, copy=True)
            self._parts = None
        else:
            self._reference = None
            self._parts = [
                (part, normalize(self._fit_X[part], copy=True))
                for part in gen_even_slices(self.n_samples_fit_, workers)
            ]
        self._fit_method = 'brute'
        self.effective_metric_ = 'cosine'
        self.normalization_stats_ = dict(reference_normalizations=1, query_rows=0)
        return self

    def distances(self, queries):
        check_is_fitted(self, ['_reference'])
        query = check_array(queries, dtype=[np.float64, np.float32], order='C')
        if query.shape[1] != self.n_features_in_:
            raise ValueError('Feature count changed')
        unit = normalize(query, copy=True)

        def block(reference):
            values = pairwise_kernels(unit, reference, metric='linear', n_jobs=1)
            values *= -1
            values += 1
            np.clip(values, 0.0, 2.0, out=values)
            return values
        if self._parts is None:
            distances = block(self._reference)
        else:
            distances = np.empty((len(query), self.n_samples_fit_), dtype=self._fit_X.dtype, order='F')

            def fill(part, reference):
                distances[:, part] = block(reference)
            Parallel(n_jobs=self.n_jobs, backend='threading')(
                delayed(fill)(part, reference) for part, reference in self._parts
            )
        if query is self._fit_X:
            np.fill_diagonal(distances, 0.0)
        self.normalization_stats_['query_rows'] += len(query)
        return distances

    def kneighbors(self, queries, n_neighbors=1, return_distance=True):
        if not isinstance(n_neighbors, (int, np.integer)) or not 1 <= n_neighbors <= self.n_samples_fit_:
            raise ValueError('n_neighbors must be between one and the control count')
        distances = self.distances(queries)
        rows = np.arange(len(distances))[:, None]
        indices = np.argpartition(distances, n_neighbors - 1, axis=1)[:, :n_neighbors]
        indices = indices[rows, np.argsort(distances[rows, indices])]
        return (distances[rows, indices], indices) if return_distance else indices
