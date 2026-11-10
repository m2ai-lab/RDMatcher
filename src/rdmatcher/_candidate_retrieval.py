"""Demand-loaded, distance-ordered neighbors for candidate reduction."""
import numpy as np

class CandidatePrefixChanged(RuntimeError):
    """A larger neighbor query changed a previously retrieved prefix."""

class NeighborRows:
    """Variable-length rows with the indexing used by horizon expansion."""

    def __init__(self, count, dtype):
        self.dtype = np.dtype(dtype)
        self.rows = [np.empty(0, dtype=dtype) for _ in range(count)]

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            return self.rows[int(key)]
        rows, columns = key
        rows = np.arange(len(self.rows))[rows] if isinstance(rows, slice) else np.asarray(rows)
        rows = np.atleast_1d(rows)
        columns = np.broadcast_to(columns, rows.shape)
        return np.asarray([self.rows[int(r)][int(c)] for r, c in zip(rows, columns)], dtype=self.dtype)

    def __setitem__(self, key, values):
        if isinstance(key, tuple):
            rows, prefix = key
            if not isinstance(prefix, slice) or prefix.start is not None or prefix.step is not None:
                raise IndexError('Neighbor rows support prefix assignment only')
        else:
            rows = key
        for index, row in enumerate(np.atleast_1d(rows)):
            self.rows[int(row)] = np.asarray(values[index], dtype=self.dtype).copy()

class LazyNeighbors:
    """Fetch ordered neighbor prefixes within the configured cap."""

    def __init__(self, matcher, cap, batch_size):
        self.matcher = matcher
        self.cap = cap
        self.batch_size = batch_size
        count = len(matcher.X_exposed)
        self.distances = NeighborRows(count, getattr(matcher, "distance_dtype", np.float32))
        self.indices = NeighborRows(count, np.int32)
        self.loaded = np.zeros(count, dtype=np.int32)
        self.stats = dict(
            mode='lazy', buffer_scheme='ragged', retrieval_calls=0,
            fetched_neighbor_entries=0, stored_neighbor_entries=0,
            max_loaded_neighbors=0, boundary_tie_full_cap_rows=0,
        )
        self.ensure(np.arange(count), np.ones(count, dtype=np.int32))

    def _retrieve(self, rows, count):
        raise NotImplementedError

    def _store(self, rows, distances, indices):
        for position, row in enumerate(rows):
            loaded = int(self.loaded[row])
            if loaded and (
                not np.array_equal(self.indices[row], indices[position, :loaded])
                or not np.array_equal(self.distances[row], distances[position, :loaded])
            ):
                raise CandidatePrefixChanged('Neighbor ordering changed during expansion')
        self.distances[rows] = distances
        self.indices[rows] = indices
        self.loaded[rows] = distances.shape[1]

    def ensure(self, rows, required):
        required = np.asarray(required)
        missing = self.loaded[rows] < required
        rows, required = (rows[missing], required[missing])
        if not len(rows):
            return
        exact_prefix = bool(getattr(self, "provider", None)
                            and getattr(self.provider, "exact_prefix", False))
        if exact_prefix:
    # Start matrix-backed rows with exact top-1 prefixes. Expand only rows
    # whose preservation checks require additional neighbors.
            targets = np.where(
                self.loaded[rows] == 0,
                required,
                np.maximum(required, np.maximum(16, self.loaded[rows] * 2)),
            )
            targets = np.minimum(self.cap, targets)
        else:
            targets = np.minimum(self.cap, np.maximum(required, np.maximum(16, self.loaded[rows] * 2)))
        for target in np.unique(targets):
            group = rows[targets == target]
            for start in range(0, len(group), self.batch_size):
                batch = group[start:start + self.batch_size]
                fetch = int(target) if exact_prefix else min(self.cap, int(target) + 1)
                distances, indices = self._retrieve(batch, fetch)
                if exact_prefix or fetch == self.cap:
                    self._store(batch, distances, indices)
                else:
                    boundary_tie = distances[:, int(target) - 1] == distances[:, int(target)]
                    self._store(
                        batch[~boundary_tie], distances[~boundary_tie, :target],
                        indices[~boundary_tie, :target],
                    )
                    tied = batch[boundary_tie]
                    if len(tied):
                        full_distances, full_indices = self._retrieve(tied, self.cap)
                        self._store(tied, full_distances, full_indices)
                        self.stats['boundary_tie_full_cap_rows'] += len(tied)
        self.stats['max_loaded_neighbors'] = int(self.loaded.max(initial=0))
        self.stats['stored_neighbor_entries'] = int(self.loaded.sum())
