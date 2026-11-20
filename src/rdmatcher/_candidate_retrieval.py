"""Demand-loaded, distance-ordered neighbors for candidate reduction."""
import numpy as np
from numba import njit
from numba.typed import List


@njit(cache=True)
def _empty_rows(count, empty):
    rows = List()
    for _ in range(count):
        rows.append(empty)
    return rows


@njit(cache=True)
def _replace_rows(buffer, rows, values):
    delta = 0
    for offset, row in enumerate(rows):
        value = values[offset].copy()
        delta += value.nbytes - buffer[row].nbytes
        buffer[row] = value
    return delta


@njit(cache=True)
def _advance(indices, distances, loaded, horizons, usage, owners, safe,
             queue, pending, state, affected, ratio, cap, threshold):
    """Process cached windows in queue order, pausing at the next cache miss."""
    head, tail, size, visits, reads = state
    count = len(queue)
    while size:
        row = queue[head]
        begin = horizons[row]
        needs_edges = safe[row] < ratio and begin < cap
        if needs_edges and begin == loaded[row]:
            state[:] = head, tail, size, visits, reads
            return row

        head = (head + 1) % count
        size -= 1
        pending[row] = False
        visits += 1
        if not needs_edges:
            continue

        reads += 1
        ids = indices[row]
        costs = distances[row]
        added = 0
        n_affected = 0
        needed = ratio - safe[row]
        end = len(ids)
        for position in range(begin, end):
            control = ids[position]
            if control < 0 or not costs[position] <= threshold:
                continue
            degree = usage[control]
            if degree == 0:
                added += 1
                owners[control] = row
                usage[control] = 1
            elif degree == 1:
                prior = owners[control]
                if prior >= 0:
                    safe[prior] -= 1
                    affected[n_affected] = prior
                    n_affected += 1
                owners[control] = -1
                usage[control] = 2
            # Once shared, the control stays competitive as prefixes grow.
            if added >= needed:
                end = position + 1
                break

        safe[row] += added
        horizons[row] = end
        affected[n_affected] = row
        n_affected += 1
        # Preserve case order when multiple owners lost a safe control.
        if n_affected > 1:
            affected[:n_affected].sort()
        for case in affected[:n_affected]:
            if safe[case] < ratio and horizons[case] < cap and not pending[case]:
                queue[tail] = case
                pending[case] = True
                tail = (tail + 1) % count
                size += 1
    state[:] = head, tail, size, visits, reads
    return -1

class CandidatePrefixChanged(RuntimeError):
    """A larger neighbor query changed a previously retrieved prefix."""


def candidate_masks(indices, distances, usage, threshold, safe_matches, fuzzy_limit=None):
    """Classify an active prefix, stopping at its rth safe or first rejected edge."""
    # The former scalar loop converted each cost to Python float before its
    # threshold comparison. Keep that float64 comparison for float32 inputs.
    costs = distances.astype(np.float64, copy=False)
    eligible = (indices >= 0) & np.isfinite(costs) & (costs <= threshold)
    fuzzy = np.zeros(len(indices), dtype=bool)
    if fuzzy_limit is not None:
        fuzzy = ~eligible & (costs <= fuzzy_limit)
    rejected = np.flatnonzero(~(eligible | fuzzy))
    stop = int(rejected[0]) if len(rejected) else len(indices)
    safe = np.zeros(len(indices), dtype=bool)
    positions = np.flatnonzero(eligible[:stop])
    safe[positions] = usage[indices[positions]] == 1
    safe_positions = np.flatnonzero(safe)
    if len(safe_positions) and len(safe_positions) >= safe_matches:
        target = max(1, int(np.ceil(safe_matches)))
        stop = min(stop, int(safe_positions[target-1]) + 1)
    safe[stop:] = False
    eligible[stop:] = False
    fuzzy[stop:] = False
    return safe, eligible & ~safe, fuzzy


def candidate_values(indices, distances, mask):
    """Borrow a complete selected prefix; allocate only for a filtered subset."""
    if np.all(mask):
        return indices, distances
    return indices[mask], distances[mask]


class NeighborRows:
    """Variable-length rows with the indexing used by horizon expansion."""

    def __init__(self, count, dtype):
        self.dtype = np.dtype(dtype)
        self.rows = _empty_rows(count, np.empty(0, dtype=dtype))
        self.nbytes = 0

    def matches_prefix(self, rows, values):
        return all(np.array_equal(self.rows[int(row)], values[local, :len(self.rows[int(row)])])
                   for local, row in enumerate(rows))

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
        values = np.asarray(values, dtype=self.dtype)
        self.nbytes += _replace_rows(self.rows, np.atleast_1d(rows), values)

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
        if (not self.indices.matches_prefix(rows, indices)
                or not self.distances.matches_prefix(rows, distances)):
            raise CandidatePrefixChanged('Neighbor ordering changed during expansion')
        self.distances[rows] = distances
        self.indices[rows] = indices
        self.loaded[rows] = distances.shape[1]

    def expand(self, ratio, threshold):
        """Reach the r-safe prefix fixed point using the shared compiled queue."""
        count = len(self.loaded)
        cap = self.cap
        controls = len(self.matcher.X_control)
        first_ids = self.indices[:, 0]
        first_valid = (first_ids >= 0) & (self.distances[:, 0] <= threshold)
        horizons = np.ones(count, dtype=np.int32)
        usage = np.minimum(2, np.bincount(first_ids[first_valid],
                                         minlength=controls)).astype(np.uint8)
        owners = np.full(controls, -1, dtype=np.int32)
        first = np.flatnonzero(first_valid & (usage[np.maximum(first_ids, 0)] == 1))
        owners[first_ids[first]] = first
        safe = np.zeros(count, dtype=np.int32)
        safe[first] = 1

        pending = (safe < ratio) & (horizons < cap)
        initial = np.flatnonzero(pending)
        queue = np.empty(count, dtype=np.int32)
        queue[:len(initial)] = initial
        state = np.array([0, len(initial) % max(count, 1), len(initial), 0, 0], dtype=np.int64)
        affected = np.empty(cap + 1, dtype=np.int32)
        threshold_type = np.result_type(self.distances.dtype, threshold).type
        threshold = threshold_type(threshold)

        while state[2]:
            row = _advance(self.indices.rows, self.distances.rows, self.loaded,
                           horizons, usage, owners, safe, queue, pending, state,
                           affected, ratio, cap, threshold)
            if row >= 0:
                # Fetch exhausted pending rows together without activating them early.
                exhausted = pending & (horizons >= self.loaded)
                exhausted[row] = False
                rows = np.append(np.flatnonzero(exhausted), row)
                self.ensure(rows, horizons[rows] + 1)

        assert np.all((safe >= ratio) | (horizons == cap))
        visits = int(state[3])
        self.stats.update(expansion_backend='compiled_case_queue',
                          case_visits=visits, window_reads=int(state[4]))
        return horizons, usage, visits

    def ensure(self, rows, required):
        required = np.asarray(required)
        missing = self.loaded[rows] < required
        rows, required = (rows[missing], required[missing])
        if not len(rows):
            return
        exact_prefix = bool(getattr(self, "provider", None)
                            and getattr(self.provider, "exact_prefix", False))
        # Every distance provider starts with the actual requested prefix.
        # Grow only cases whose expansion consumes their cached neighbors.
        growth = getattr(getattr(self, 'provider', None), 'prefix_growth', 2)
        targets = np.where(self.loaded[rows] == 0, required,
                           np.maximum(required, self.loaded[rows] * growth))
        targets = np.minimum(self.cap, targets)
        for target in np.unique(targets):
            group = rows[targets == target]
            for start in range(0, len(group), self.batch_size):
                batch = group[start:start + self.batch_size]
                if exact_prefix and getattr(self.provider, 'padded_prefix', False):
                    # Exact matrix prefixes mark exhausted eligibility with -1.
                    # Extend that known padding from the shared cache instead
                    # of repeating selection or keeping a second adapter pool.
                    completed = np.zeros(len(batch), dtype=bool)
                    loaded = self.loaded[batch] > 0
                    completed[loaded] = self.indices[
                        batch[loaded], self.loaded[batch[loaded]] - 1] < 0
                    if np.any(completed):
                        terminal = batch[completed]
                        padding_d = np.full((len(terminal), int(target)), np.inf,
                                            dtype=self.distances.dtype)
                        padding_i = np.full((len(terminal), int(target)), -1, dtype=np.int32)
                        for local, row in enumerate(terminal):
                            size = int(self.loaded[row])
                            padding_d[local, :size] = self.distances[row]
                            padding_i[local, :size] = self.indices[row]
                        self._store(terminal, padding_d, padding_i)
                        batch = batch[~completed]
                    if not len(batch):
                        continue
                fetch = int(target) if exact_prefix else min(self.cap, int(target) + 1)
                distances, indices = self._retrieve(batch, fetch)
                if exact_prefix or distances.shape[1] == self.cap:
                    self._store(batch, distances, indices)
                else:
                    boundary_tie = distances[:, int(target) - 1] == distances[:, int(target)]
                    self._store(
                        batch[~boundary_tie], distances[~boundary_tie, :target],
                        indices[~boundary_tie, :target],
                    )
                    tied = batch[boundary_tie]
                    if len(tied):
                        cutoff = distances[boundary_tie, int(target) - 1]
                        tie_fetch = fetch
                        while len(tied):
                            tie_fetch = min(self.cap, max(tie_fetch + 1, tie_fetch * 2))
                            tie_distances, tie_indices = self._retrieve(tied, tie_fetch)
                            if tie_fetch == self.cap:
                                self._store(tied, tie_distances, tie_indices)
                                self.stats['boundary_tie_full_cap_rows'] += len(tied)
                                break
                            complete = tie_distances[:, -1] > cutoff
                            self._store(tied[complete], tie_distances[complete, :target],
                                        tie_indices[complete, :target])
                            tied, cutoff = tied[~complete], cutoff[~complete]
        self.stats['max_loaded_neighbors'] = int(self.loaded.max(initial=0))
        self.stats['stored_neighbor_entries'] = int(self.loaded.sum())
        self.stats['prefix_buffer_bytes'] = self.distances.nbytes + self.indices.nbytes
