"""Metric-independent retrieval of ordered candidate prefixes."""
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor
from joblib import effective_n_jobs
import numpy as np
from ._candidate_retrieval import LazyNeighbors
from ._neighbor_backends import PrefixQuery, order_prefix
from ._precomputed import MatrixPrefixQuery

class DemandNeighbors(LazyNeighbors):
    """Ordered backend requests with prefix consistency checks."""

    def __init__(self, matcher, cap, batch_size):
        self.query_counts = np.zeros(len(matcher.X_exposed), dtype=np.int32)
        workers = effective_n_jobs(matcher.n_jobs)
        self.executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        map_tasks = self.executor.map if self.executor is not None else map
        try:
            self.provider = (MatrixPrefixQuery(matcher, cap, map_tasks, workers)
                             if matcher.distance_metric == "precomputed" else
                             PrefixQuery(matcher, batch_size, cap, map_tasks, workers))
            super().__init__(matcher, cap, batch_size)
        except BaseException:
            self.close()
            raise
        self.stats['mode'] = 'demand'

    def close(self):
        """Release the one worker pool owned by this retrieval session."""
        if self.executor is not None:
            self.executor.shutdown(wait=True)

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
        self.stats['fetched_neighbor_entries'] += int(distances.size)
        if getattr(self.provider, 'exact_prefix', False):
            return (distances, indices)
        return order_prefix(distances.astype(np.float32), indices.astype(np.int32))
