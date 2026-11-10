"""Neighbor backend adapters for shared prefix loading."""
import inspect

import numpy as np


class PrefixQuery:
    """Retrieve ordered prefixes from an index or stored candidate pool."""

    def __init__(self, matcher, batch_size, cap=None):
        self.matcher = matcher
        self.model = matcher.nbrs_model
        backend = getattr(self.model, '_nn_model', None)
        if backend is None:
            backend = self.model
        self.scan = (
            getattr(backend, '_fit_method', None) == 'brute'
            or getattr(self.model, 'neighbor_backend', None) == 'cdist'
            or getattr(matcher, 'ps_eligible_sets', None) is not None
        )
        self.cap = cap
        count = len(matcher.X_exposed)
        self.pool_distances = [None] * count
        self.pool_indices = [None] * count
        self.stats = dict(
            backend_calls=0, backend_row_queries=0, backend_fetched_entries=0,
            reused_prefix_rows=0, stored_pool_entries=0,
            prefix_source='stored_cap' if self.scan else 'native_prefix',
        )
        parameters = inspect.signature(self.model.kneighbors).parameters
        self.options = {}
        if 'batch_size' in parameters:
            self.options['batch_size'] = batch_size
        if 'n_jobs' in parameters:
            self.options['n_jobs'] = matcher.n_jobs

    def query(self, rows, count):
        if not self.scan:
            return self._native_query(rows, count)
        if self.cap is None:
            raise ValueError('A candidate cap is required for stored prefixes')
        missing = np.asarray([r for r in rows if self.pool_indices[r] is None], dtype=np.int32)
        self.stats['reused_prefix_rows'] += len(rows) - len(missing)
        if len(missing):
            distances, indices = self._native_query(missing, self.cap)
            distances, indices = (distances.astype(np.float32), indices.astype(np.int32))
            order = np.lexsort((indices, distances), axis=1)
            distances = np.take_along_axis(distances, order, axis=1)
            indices = np.take_along_axis(indices, order, axis=1)
            for i, row in enumerate(missing):
                self.pool_distances[row] = distances[i]
                self.pool_indices[row] = indices[i]
                self.stats['stored_pool_entries'] += len(indices[i])
        return (
            np.stack([self.pool_distances[r][:count] for r in rows]),
            np.stack([self.pool_indices[r][:count] for r in rows]),
        )

    def _native_query(self, rows, count):
        owner = self.matcher
        if getattr(owner, 'ps_eligible_sets', None) is not None:
            distances = np.full((len(rows), count), np.inf, dtype=np.float32)
            indices = np.full((len(rows), count), -1, dtype=np.int32)
            for local, row in enumerate(rows):
                eligible = owner.ps_eligible_sets[row]
                k = min(count, len(eligible))
                if not k:
                    continue
                self.stats['backend_calls'] += 1
                self.stats['backend_row_queries'] += 1
                self.stats['backend_fetched_entries'] += k
                query = (
                    owner.X_exposed.iloc[row:row + 1]
                    if hasattr(owner.X_exposed, 'iloc') else owner.X_exposed[row:row + 1]
                )
                compact = getattr(self.model, 'kneighbors_subset_positions', None)
                result = (
                    compact(query, eligible, k=k, query_position=int(row))
                    if compact is not None else None
                )
                if result is None:
                    reference = (
                        owner.X_control.iloc[eligible]
                        if hasattr(owner.X_control, 'iloc') else owner.X_control[eligible]
                    )
                    result = self.model.kneighbors_subset(
                        query, reference, k=k, streaming=owner.streaming,
                        stream_block_size=owner.stream_block_size, **self.options,
                    )
                d, ids = result
                distances[local, :k] = d[0]
                indices[local, :k] = eligible[ids[0]]
            return (distances, indices)
        self.stats['backend_calls'] += 1
        self.stats['backend_row_queries'] += len(rows)
        self.stats['backend_fetched_entries'] += len(rows) * count
        query = owner.X_exposed.iloc[rows] if hasattr(owner.X_exposed, 'iloc') else owner.X_exposed[rows]
        return self.model.kneighbors(query, n_neighbors=count, **self.options)
