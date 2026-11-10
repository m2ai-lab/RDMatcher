"""Metric-independent retrieval of ordered candidate prefixes."""
from time import perf_counter
import numpy as np
from ._candidate_retrieval import LazyNeighbors
from ._neighbor_backends import PrefixQuery
from ._precomputed import MatrixPrefixQuery

class DemandNeighbors(LazyNeighbors):
    """Ordered backend requests with prefix consistency checks."""

    def __init__(self, matcher, cap, batch_size):
        self.query_counts = np.zeros(len(matcher.X_exposed), dtype=np.int32)
        self.provider = (MatrixPrefixQuery(matcher, cap) if matcher.distance_metric == "precomputed"
                         else PrefixQuery(matcher, batch_size, cap))
        super().__init__(matcher, cap, batch_size)
        self.stats['mode'] = 'demand'

    def _retrieve(self, rows, count):
        self.stats['repeated_prefix_rows'] = (
            self.stats.get('repeated_prefix_rows', 0)
            + int(np.count_nonzero(self.query_counts[rows]))
        )
        self.query_counts[rows] += 1
        begin = perf_counter()
        distances, indices = self.provider.query(rows, count)
        self.stats.update(self.provider.stats)
        self.stats['neighbor_query_sec'] = self.stats.get('neighbor_query_sec', 0.0) + perf_counter() - begin
        self.stats['retrieval_calls'] += 1
        self.stats['fetched_neighbor_entries'] += int(len(rows) * count)
        if self.provider.scan:
            return (distances, indices)
        distances, indices = (distances.astype(np.float32), indices.astype(np.int32))
        order = np.lexsort((indices, distances), axis=1)
        return (np.take_along_axis(distances, order, axis=1), np.take_along_axis(indices, order, axis=1))
