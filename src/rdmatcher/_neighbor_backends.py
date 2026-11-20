"""Neighbor backend adapters for shared prefix loading."""
import inspect

import numpy as np
from numba import njit
from scipy.spatial import cKDTree
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from ._precomputed import _ordered_prefix


def _neighbor_order(distances, indices):
    """Stable (float32 cost, int32 control) order using one NumPy integer sort.

    Map IEEE float bits into numeric order, with equal keys for signed zeros
    and for all NaNs, matching lexsort. The low word orders signed control IDs.
    Precomputed float64 costs keep their separate exact-prefix contract.
    """
    bits = distances.view(np.uint32)
    costs = np.where(bits >> 31, ~bits, bits ^ np.uint32(0x80000000))
    costs[distances == 0] = np.uint32(0x80000000)
    costs[np.isnan(distances)] = np.uint32(0xffffffff)
    keys = ((costs.astype(np.uint64) << 32)
            | (indices.view(np.uint32) ^ np.uint32(0x80000000)))
    return np.argsort(keys, axis=1, kind='stable')


@njit(cache=True)
def _order_ties(costs, indices):
    """Canonicalize equal-cost runs and identify rows that need reordering."""
    unordered = np.zeros(len(costs), dtype=np.bool_)
    for row in range(len(costs)):
        start = 0
        while start < costs.shape[1]:
            cost = costs[row, start]
            if np.isnan(cost) or (start and cost < costs[row, start - 1]):
                unordered[row] = True
                break
            end = start + 1
            while end < costs.shape[1] and costs[row, end] == cost:
                if cost == 0 and np.signbit(cost) != np.signbit(costs[row, end]):
                    unordered[row] = True
                    break
                end += 1
            if unordered[row]:
                break
            if end - start > 1:
                indices[row, start:end].sort()
            start = end
    return unordered


def order_prefix(costs, indices):
    """Sort owned float32/int32 arrays in place by cost and control position."""
    if costs.shape[1] > 1:
        unordered = _order_ties(costs, indices)
        if unordered.any():
            rows = costs[unordered]
            controls = indices[unordered]
            order = _neighbor_order(rows, controls)
            costs[unordered] = np.take_along_axis(rows, order, axis=1)
            indices[unordered] = np.take_along_axis(controls, order, axis=1)
    return costs, indices


@njit(cache=True)
def _cosine_costs(queries, controls, indices):
    costs = np.empty(indices.shape, dtype=np.float32)
    for row in range(len(queries)):
        for position in range(indices.shape[1]):
            similarity = 0.0
            for feature in range(queries.shape[1]):
                similarity += queries[row, feature] * controls[indices[row, position], feature]
            costs[row, position] = min(2.0, max(0.0, 1.0 - similarity))
    return costs


class IndexedNeighbors:
    """One Euclidean index for numeric metrics with a Euclidean ordering."""

    _fit_method = 'kd_tree'
    prefix_growth = 8
    exact_prefix = True

    def __init__(self, controls, metric):
        self.metric = metric
        self._dimensions = controls.shape[1]
        self._cosine = metric == 'cosine'
        reference = self._normalize(controls) if self._cosine else controls
        self._padding = False
        if self._cosine:
            # Dot-product order is Euclidean order when controls have equal norm.
            # Lift controls that normalize to a short vector onto the unit sphere,
            # keeping query coordinates zero along that added axis.
            norms = np.einsum('ij,ij->i', reference, reference)
            self._error = 16 * (self._dimensions + 2) * np.finfo(np.float64).eps
            self._padding = bool(np.any(norms < 1 - self._error / 4))
            if self._padding:
                reference = np.column_stack((reference, np.sqrt(np.maximum(0, 1 - norms))))
        self._index = cKDTree(reference, leafsize=48)

    @staticmethod
    def _normalize(values):
        return normalize(np.asarray(values, dtype=np.float64), copy=True)

    def prepare_queries(self, queries):
        if not self._cosine:
            return np.asarray(queries)
        unit = self._normalize(queries)
        if self._padding:
            unit = np.column_stack((unit, np.zeros(len(unit))))
        return unit

    def query(self, queries, k, return_distance=True, *, workers=1):
        n_controls = self._index.n
        fetch = min(k + 1, n_controls)
        distances, ids = self._index.query(queries, k=fetch, workers=workers)
        distances = distances.reshape(len(queries), fetch)
        ids = ids.reshape(len(queries), fetch)
        candidates = ids[:, :k]
        costs = self._costs(queries, distances[:, :k], candidates)

        if fetch > k:
            cutoff = costs.max(axis=1)
            if self._cosine:
                norm = np.einsum('ij,ij->i', queries, queries)
                offset = (1.0 - norm) / 2.0
                lower = np.maximum(0, distances[:, k] ** 2 / 2.0 + offset - self._error)
                upper = np.nextafter(cutoff, np.float32(np.inf), dtype=np.float32)
                radius = np.sqrt(np.maximum(0, 2.0 * (upper.astype(float) - offset + self._error)))
            else:
                lower = distances[:, k]
                radius = np.nextafter(cutoff, np.float32(np.inf), dtype=np.float32)

            # The metric can round neighbors on either side of the tree's kth
            # rank to the same cost. Include the whole boundary before selecting
            # the lowest control positions, so every returned prefix is nested.
            for row in np.flatnonzero(lower.astype(np.float32) <= cutoff):
                point = queries[row:row + 1]
                if self._cosine and not np.any(point):
                    candidates[row] = np.arange(k)
                    costs[row] = 1.0
                    continue
                tied = np.asarray(self._index.query_ball_point(point[0], float(radius[row])),
                                  dtype=np.intp)
                values = self._boundary_costs(point, tied)
                order = _ordered_prefix(values, tied, k)
                candidates[row] = tied[order]
                costs[row] = values[order]

        return order_prefix(costs, candidates.astype(np.int32))

    def _boundary_costs(self, point, indices):
        if self._cosine:
            return self._costs(point, None, indices[None, :])[0]
        delta = self._index.data[indices] - point
        return np.sqrt(np.einsum('ij,ij->i', delta, delta)).astype(np.float32)

    def _costs(self, queries, distances, indices):
        if self._cosine:
            return _cosine_costs(queries[:, :self._dimensions],
                                 self._index.data[:, :self._dimensions], indices)
        return distances.astype(np.float32)

    def kneighbors(self, queries, n_neighbors, *, workers=1):
        return self.query(self.prepare_queries(queries), n_neighbors, workers=workers)


class PrefixQuery:
    """Query indexed prefixes or a complete scan result for the shared cache."""

    def __init__(self, matcher, batch_size, cap=None, map_tasks=map, workers=1):
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
        # Reuse the fitted tree and the session's worker pool. Transforming
        # wrappers retain their own query contract and compiled metric kernels.
        self.tree = (getattr(self.model, '_tree', None)
                     if type(self.model) is NearestNeighbors else None)
        self.queries = matcher.X_exposed
        if isinstance(self.model, IndexedNeighbors):
            self.tree = self.model
            self.queries = self.model.prepare_queries(self.queries)
        self.exact_prefix = not self.scan and getattr(backend, 'exact_prefix', False)
        self.prefix_growth = getattr(self.model, 'prefix_growth', 2)
        self.map_tasks, self.workers = map_tasks, workers
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
        # A scan already evaluates every reference row. Return its complete
        # capped result once; LazyNeighbors owns it without a second pool.
        return self._native_query(rows, self.cap)

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
        query = self.queries.iloc[rows] if hasattr(self.queries, 'iloc') else self.queries[rows]
        if self.tree is not None:
            chunks = np.array_split(np.asarray(query), min(self.workers, len(rows)))
            result = list(self.map_tasks(
                lambda block: self.tree.query(block, k=count, return_distance=True), chunks))
            if len(result) == 1:
                return result[0]
            return tuple(np.concatenate(values, axis=0) for values in zip(*result))
        return self.model.kneighbors(query, n_neighbors=count, **self.options)
