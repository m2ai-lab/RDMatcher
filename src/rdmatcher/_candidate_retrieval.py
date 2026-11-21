"""Demand-loaded, distance-ordered neighbors for candidate reduction."""
from time import perf_counter

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
def _matches_prefix(buffer, rows, values):
    for offset, row in enumerate(rows):
        old = buffer[row]
        if len(old) > values.shape[1]:
            return False
        for rank in range(len(old)):
            if old[rank] != values[offset, rank]:
                return False
    return True


@njit(cache=True)
def _unchanged_certificates(indices, horizons, changed_at, witnessed_at, candidates):
    """Reuse a whole-prefix certificate until one of its controls gains an edge."""
    reusable = np.zeros(len(horizons), dtype=np.bool_)
    for row in candidates:
        stamp = witnessed_at[row]
        if stamp < 0:
            continue
        reusable[row] = True
        for control in indices[row][:horizons[row]]:
            if changed_at[control] > stamp:
                reusable[row] = False
                witnessed_at[row] = -1
                break
    return reusable


@njit(cache=True)
def _bounded_competition(indices, horizons, degree, candidates, ratio,
                         lower_bounds, visit_budget, work):
    """Certify unresolved rows on one snapshot of the activated prefixes."""
    certified = np.zeros(len(horizons), dtype=np.bool_)
    # One control alone can expose too many distinct competitors.
    requested = np.zeros(len(degree), dtype=np.bool_)
    plausible = []
    for row in candidates:
        width = horizons[row]
        limit = width - ratio
        if limit < lower_bounds[row]:
            work[1] += 1
            continue
        ids = indices[row][:width]
        possible = True
        for control in ids:
            if degree[control] > limit + 1:
                lower_bounds[row] = max(lower_bounds[row], degree[control] - 1)
                possible = False
                break
        if possible:
            plausible.append(row)
            for control in ids:
                if degree[control] > 1:
                    requested[control] = True
    if not plausible:
        return certified

    offsets = np.empty(len(degree) + 1, dtype=np.int64)
    offsets[0] = 0
    for control in range(len(degree)):
        offsets[control + 1] = offsets[control] + (degree[control] if requested[control] else 0)
    incident = np.empty(offsets[-1], dtype=np.int32)
    cursor = offsets[:-1].copy()
    for row in range(len(horizons)):
        for control in indices[row][:horizons[row]]:
            if requested[control]:
                incident[cursor[control]] = row
                cursor[control] += 1

    counts = np.zeros(len(horizons), dtype=np.int32)
    touched = np.empty(len(horizons), dtype=np.int32)
    visited = 0
    for row in plausible:
        if visited >= visit_budget:
            break
        limit = horizons[row] - ratio
        total = length = 0
        for control in indices[row][:horizons[row]]:
            start = offsets[control]
            end = start + min(offsets[control + 1] - start, visit_budget - visited)
            for pos in range(start, end):
                competitor = incident[pos]
                if competitor == row or counts[competitor] == ratio:
                    continue
                if counts[competitor] == 0:
                    touched[length] = competitor
                    length += 1
                counts[competitor] += 1
                total += 1
                if total > limit:
                    end = pos + 1
                    break
            visited += end - start
            if total > limit or visited >= visit_budget:
                break
        lower_bounds[row] = max(lower_bounds[row], total)
        # An interrupted traversal supplies a lower bound, never a certificate.
        certified[row] = visited < visit_budget and total <= limit
        for competitor in touched[:length]:
            counts[competitor] = 0
    work[0] = visited
    return certified


@njit(cache=True)
def _advance(indices, distances, loaded, horizons, usage, owners, safe,
             queue, pending, state, affected, ratio, cap, threshold,
             retrieval_limit, terminal, paused, checkpoint, track_degree,
             changed_at, epoch):
    """Process cached windows in queue order, pausing at the next cache miss."""
    head, tail, size, visits, reads, activated = state
    count = len(queue)
    while size:
        if activated >= checkpoint:
            state[:] = head, tail, size, visits, reads, activated
            return -2
        row = queue[head]
        begin = horizons[row]
        needs_edges = safe[row] < ratio and begin < cap and not terminal[row] and not paused[row]
        if needs_edges and begin == loaded[row]:
            state[:] = head, tail, size, visits, reads, activated
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
            cost = costs[position]
            if control < 0 or not np.isfinite(cost) or cost > retrieval_limit:
                terminal[row] = True
                end = position
                break
            if cost > threshold:
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
            elif track_degree:
                # Safe accounting saturates at two. The competition check uses exact degrees.
                usage[control] += 1
            if track_degree:
                changed_at[control] = epoch
            if added >= needed:
                end = position + 1
                break

        safe[row] += added
        horizons[row] = end
        activated += end - begin
        affected[n_affected] = row
        n_affected += 1
        # Preserve case order when multiple owners lost a safe control.
        if n_affected > 1:
            affected[:n_affected].sort()
        for case in affected[:n_affected]:
            if safe[case] < ratio and horizons[case] < cap and not terminal[case] and not paused[case] and not pending[case]:
                queue[tail] = case
                pending[case] = True
                tail = (tail + 1) % count
                size += 1
    state[:] = head, tail, size, visits, reads, activated
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
        return _matches_prefix(self.rows, np.asarray(rows), values)

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
        start = perf_counter()
        if (not self.indices.matches_prefix(rows, indices)
                or not self.distances.matches_prefix(rows, distances)):
            raise CandidatePrefixChanged('Neighbor ordering changed during expansion')
        checked = perf_counter()
        self.stats['prefix_comparison_sec'] = self.stats.get('prefix_comparison_sec', 0.) + checked - start
        self.distances[rows] = distances
        self.indices[rows] = indices
        self.loaded[rows] = distances.shape[1]
        self.stats['prefix_copy_sec'] = self.stats.get('prefix_copy_sec', 0.) + perf_counter() - checked
        self.stats['prefix_copied_bytes'] = self.stats.get('prefix_copied_bytes', 0) + distances.nbytes + indices.nbytes

    def expand(self, ratio, threshold, *, fuzzy_limit=None, competition=False):
        """Activate ordered prefixes until every case has a stopping condition."""
        count = len(self.loaded)
        cap = self.cap
        controls = len(self.matcher.X_control)
        first_ids = self.indices[:, 0]
        first_costs = self.distances[:, 0].astype(np.float64, copy=False)
        retrieval_limit = threshold if fuzzy_limit is None else max(threshold, fuzzy_limit)
        terminal = (first_ids < 0) | ~np.isfinite(first_costs) | (first_costs > retrieval_limit)
        first_valid = ~terminal & (first_costs <= threshold)
        horizons = (~terminal).astype(np.int32)
        usage = np.bincount(first_ids[first_valid], minlength=controls)
        usage = usage.astype(np.int32) if competition else np.minimum(2, usage).astype(np.uint8)
        owners = np.full(controls, -1, dtype=np.int32)
        first = np.flatnonzero(first_valid & (usage[np.maximum(first_ids, 0)] == 1))
        owners[first_ids[first]] = first
        safe = np.zeros(count, dtype=np.int32)
        safe[first] = 1

        pending = (safe < ratio) & (horizons < cap) & ~terminal
        initial = np.flatnonzero(pending)
        queue = np.empty(count, dtype=np.int32)
        queue[:len(initial)] = initial
        state = np.array([0, len(initial) % max(count, 1), len(initial), 0, 0,
                          horizons.sum()], dtype=np.int64)
        affected = np.empty(cap + 1, dtype=np.int32)
        paused = np.zeros(count, dtype=bool)
        # Check dependencies at geometric intervals and when the queue drains.
        initial_edges = int(state[5])
        checkpoint = initial_edges + max(1, len(initial)) if competition else np.iinfo(np.int64).max
        checks = 0
        competition_sec = 0.
        lower_bounds = np.zeros(count, dtype=np.int32)
        work = np.zeros(2, dtype=np.int64)
        last_check_edges = initial_edges
        # Short prefixes can resolve cascades before expansion becomes costly.
        budget_start = count * min(1024, max(1, cap // 16))
        competitor_visits = skipped_checks = budget_hits = 0
        changed_at = np.zeros(controls if competition else 0, dtype=np.int32)
        witnessed_at = np.full(count, -1, dtype=np.int32)
        reused_checks = 0
        deep_refill = False
        while state[2]:
            row = _advance(self.indices.rows, self.distances.rows, self.loaded,
                           horizons, usage, owners, safe, queue, pending, state,
                           affected, ratio, cap, threshold, retrieval_limit, terminal,
                           paused, checkpoint, competition, changed_at, checks + 1)
            if row >= 0:
                exhausted = pending & ~terminal & ~paused & (horizons >= self.loaded)
                exhausted[row] = False
                rows = np.append(np.flatnonzero(exhausted), row)
                self.ensure(rows, horizons[rows] + 1, deep=deep_refill)
            elif competition and (row == -2 or paused.any()):
                start = perf_counter()
                unresolved = (safe < ratio) & (horizons < cap) & ~terminal
                current_edges = int(state[5])
                visit_budget = np.iinfo(np.int64).max
                if current_edges - initial_edges >= budget_start:
                    visit_budget = min(8_000_000, max(64, count,
                                                      2 * (current_edges - last_check_edges)))
                work[:] = 0
                # Revalidate changed provisional stops first. Unchecked rows resume expansion.
                candidates = np.concatenate((np.flatnonzero(unresolved & paused),
                                             np.flatnonzero(unresolved & ~paused)))
                witnessed = _unchanged_certificates(
                    self.indices.rows, horizons, changed_at, witnessed_at, candidates)
                checked = _bounded_competition(
                    self.indices.rows, horizons, usage, candidates[~witnessed[candidates]],
                    ratio, lower_bounds, visit_budget, work)
                witnessed_at[checked] = checks + 1
                paused = witnessed | checked
                reused_checks += int(witnessed.sum())
                deep_refill = work[0] >= visit_budget
                competitor_visits += int(work[0])
                skipped_checks += int(work[1])
                budget_hits += int(deep_refill)
                last_check_edges = current_edges
                # Keep waiting rows in queue order. Requeue invalidated certificates.
                waiting = queue[(state[0] + np.arange(state[2])) % count]
                waiting = waiting[unresolved[waiting] & ~paused[waiting]]
                pending[:] = False
                pending[waiting] = True
                resumed = np.flatnonzero(unresolved & ~paused & ~pending)
                active = np.concatenate((waiting, resumed))
                pending[resumed] = True
                queue[:len(active)] = active
                state[:3] = 0, len(active) % count, len(active)
                # Space out costly checks when their traversal budget is exhausted.
                growth = 4 if deep_refill else 2
                checkpoint = initial_edges + max(1, (current_edges - initial_edges) * growth)
                checks += 1
                competition_sec += perf_counter() - start

        assert np.all((safe >= ratio) | (horizons == cap) | terminal | paused)
        self.competition_certified = paused
        self.exhausted_eligibility = terminal
        visits = int(state[3])
        self.stats.update(expansion_backend='compiled_case_queue',
                          case_visits=visits, window_reads=int(state[4]),
                          exhausted_eligibility_rows=int(terminal.sum()),
                          competition_checks=checks, competition_sec=competition_sec,
                          competition_visited_entries=competitor_visits,
                          competition_lower_bound_skips=skipped_checks,
                          competition_budget_hits=budget_hits,
                          competition_witness_reuses=reused_checks,
                          competition_certified_rows=int(paused.sum()))
        return horizons, usage, visits

    def ensure(self, rows, required, *, deep=False):
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
            batch_size = self.batch_size
            if deep and exact_prefix and target == self.cap:
                # Bound transient query results while batching deep requests.
                batch_size = max(batch_size, min(8 * batch_size, 1_000_000 // int(target)))
                if batch_size > self.batch_size:
                    self.stats['deep_refill_rows'] = self.stats.get('deep_refill_rows', 0) + len(group)
            for start in range(0, len(group), batch_size):
                batch = group[start:start + batch_size]
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
