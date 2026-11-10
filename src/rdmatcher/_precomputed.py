"""Validated, borrowed case-by-control costs and per-call ordered prefixes."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from scipy.sparse import isspmatrix_csr


def _ordered_prefix(costs, controls, count):
    """Select the exact (distance, control-position) prefix without a full sort.

    Argpartition alone can choose arbitrary members of a boundary tie. Keep all
    strictly cheaper edges and fill the remaining slots with the lowest control
    positions at the cutoff, then sort only that selected prefix. Inputs are
    borrowed and never partitioned in place.
    """
    count = min(count, len(costs))
    if count == 0:
        return np.empty(0, dtype=np.intp)
    if count == 1:
        cutoff = costs.min()
        tied = np.flatnonzero(costs == cutoff)
        if len(tied) == 1:
            return tied
        return tied[np.argmin(controls[tied])].reshape(1)
    if count == len(costs):
        return np.lexsort((controls, costs))

    partition = np.argpartition(costs, count - 1)[:count]
    cutoff = costs[partition].max()
    cheaper = partition[costs[partition] < cutoff]
    tied = np.flatnonzero(costs == cutoff)
    remaining = count - len(cheaper)
    if len(tied) > remaining:
        tied = tied[np.argpartition(controls[tied], remaining - 1)[:remaining]]
    selected = np.concatenate((cheaper, tied))
    return selected[np.lexsort((controls[selected], costs[selected]))]


class PrecomputedDistances:
    def __init__(self, matrix, case_ids, control_ids, population_cases, population_controls):
        if isinstance(matrix, pd.DataFrame):
            for supplied, labels in ((case_ids, matrix.index), (control_ids, matrix.columns)):
                if supplied is not None and not pd.Index(supplied).equals(labels):
                    raise ValueError("Additional IDs conflict with distance_matrix labels")
            case_ids, control_ids = matrix.index, matrix.columns
            matrix = matrix.to_numpy(copy=False)
        self.sparse = isspmatrix_csr(matrix)
        if not self.sparse and not isinstance(matrix, np.ndarray):
            raise TypeError("distance_matrix must be a NumPy array, DataFrame, or SciPy CSR matrix")
        if any(size > np.iinfo(np.int32).max for size in matrix.shape):
            raise ValueError("distance_matrix axes exceed the supported int32 neighbor-index range")
        if case_ids is None or control_ids is None:
            raise ValueError("case_ids and control_ids are required for array and CSR inputs")
        axes = []
        for name, supplied, population in (("case_ids", case_ids, population_cases),
                                            ("control_ids", control_ids, population_controls)):
            labels = pd.Index(supplied)
            if not labels.is_unique or labels.hasnans:
                raise ValueError(f"{name} must contain unique, nonmissing IDs")
            positions = labels.get_indexer(population)
            if len(labels) != len(population) or np.any(positions < 0):
                raise ValueError(f"{name} must match the population IDs exactly")
            axes.append(positions)
        if matrix.ndim != 2 or matrix.shape != (len(case_ids), len(control_ids)):
            raise ValueError("distance_matrix shape must be (number of cases, number of controls)")
        if matrix.dtype.kind not in "fiu":
            raise ValueError("distance_matrix must contain real numeric distances")
        if matrix.dtype.kind == "f" and matrix.dtype.itemsize > 8:
            raise ValueError("Distance solvers support up to float64 precision; convert higher-precision inputs explicitly")
        if self.sparse:
            if (len(matrix.indptr) != matrix.shape[0] + 1 or matrix.indptr[0] != 0
                    or np.any(np.diff(matrix.indptr) < 0)
                    or matrix.indptr[-1] != len(matrix.data)
                    or len(matrix.indices) != len(matrix.data)
                    or np.any(matrix.indices < 0) or np.any(matrix.indices >= matrix.shape[1])):
                raise ValueError("Invalid CSR structure")
        # Validate values in bounded blocks, including non-contiguous dense inputs.
        if self.sparse:
            values = matrix.data
        elif matrix.flags.c_contiguous:
            values = matrix.reshape(-1)
        elif matrix.flags.f_contiguous:
            values = matrix.reshape(-1, order='F')
        else:
            # A flatiter slice copies, but keeps arbitrary strides bounded.
            values = matrix.flat
        for start in range(0, matrix.nnz if self.sparse else matrix.size, 65536):
            block = values[start:start + 65536]
            if np.any(np.isnan(block)) or np.any(block < 0):
                raise ValueError("Distances must be nonnegative and cannot contain NaN or negative infinity")
            if matrix.dtype.kind in "iu":
                converted = block.astype(np.float64)
                upper = float(2 ** (8 * matrix.dtype.itemsize - (matrix.dtype.kind == 'i')))
                if np.any(converted >= upper) or np.any(converted.astype(matrix.dtype) != block):
                    raise ValueError("Integer distances must be exactly representable in float64")
        if self.sparse:
            for row in range(matrix.shape[0]):
                indices = matrix.indices[matrix.indptr[row]:matrix.indptr[row + 1]]
                if len(np.unique(indices)) != len(indices):
                    raise ValueError("Duplicate CSR coordinates are not allowed")
        self.matrix = matrix
        self.case_positions, self.control_positions = axes
        self.matrix_to_control = np.empty(len(control_ids), dtype=np.intp)
        self.matrix_to_control[self.control_positions] = np.arange(len(control_ids))
        self.dtype = np.result_type(matrix.dtype, np.float32)


class MatrixPrefixQuery:
    """Demand-load exact matrix prefixes; never reorder the caller's matrix."""
    scan = True
    exact_prefix = True
    _MAX_PARALLEL_WORKERS = 16

    def __init__(self, matcher, cap):
        self.matcher, self.cap = matcher, cap
        self.source = matcher.precomputed_distances
        self.rows = {}
        self.stats = dict(prefix_source="precomputed", backend_calls=0,
                          backend_row_queries=0, prefix_expansion_row_queries=0,
                          stored_pool_entries=0, parallel_workers=1)

    def _eligible_costs(self, row):
        """Return borrowed or eligibility-gathered values for one case row."""
        source = self.source
        matrix_row = source.case_positions[row]
        if source.sparse:
            start, end = source.matrix.indptr[matrix_row:matrix_row + 2]
            costs = source.matrix.data[start:end]
            controls = source.matrix_to_control[source.matrix.indices[start:end]]
            if self.matcher.ps_eligible_sets is not None:
                # CSR coordinates and population control positions are unique.
                eligible = np.asarray(self.matcher.ps_eligible_sets[row], dtype=np.intp)
                selected = np.isin(controls, eligible, assume_unique=True)
                costs, controls = costs[selected], controls[selected]
            return costs, controls

        costs = source.matrix[matrix_row]
        controls = source.matrix_to_control
        if self.matcher.ps_eligible_sets is None:
            return costs, controls

        # Eligibility sets are population control positions.  Gather directly
        # from the matrix axis rather than constructing a million-element mask.
        eligible = np.asarray(self.matcher.ps_eligible_sets[row], dtype=np.intp)
        eligible = eligible[(eligible >= 0) & (eligible < len(controls))]
        if eligible.size == 0:
            return costs[:0], eligible
        if eligible.size > 1:
            eligible = np.unique(eligible)
        return costs[source.control_positions[eligible]], eligible

    def _compute_prefix(self, row, count):
        costs, controls = self._eligible_costs(row)
        order = _ordered_prefix(costs, controls, count)
        costs, controls = costs[order], controls[order]
        # Selection before filtering is exact: if fewer than ``count`` values
        # survive, every later value is at least as large and cannot be eligible.
        valid = np.isfinite(costs)
        if self.matcher.threshold != np.inf:
            valid &= costs <= self.matcher.threshold
        return costs[valid], controls[valid]

    def _cache_prefix(self, row, count, costs, controls):
        old = self.rows.get(int(row))
        if old is not None:
            self.stats["stored_pool_entries"] -= len(old[0])
            self.stats["prefix_expansion_row_queries"] += 1
        self.rows[int(row)] = (costs, controls, len(costs) < count)
        self.stats["stored_pool_entries"] += len(costs)
        self.stats["backend_row_queries"] += 1

    def query(self, rows, count):
        distances = np.full((len(rows), count), np.inf, dtype=self.source.dtype)
        indices = np.full((len(rows), count), -1, dtype=np.int32)
        rows = np.asarray(rows, dtype=np.intp)
        missing = []
        for row in rows:
            cached = self.rows.get(int(row))
            if cached is None or (len(cached[0]) < count and not cached[2]):
                missing.append(int(row))

        workers = min(self._MAX_PARALLEL_WORKERS,
                      max(1, int(getattr(self.matcher, "n_jobs", 1))),
                      len(missing) or 1)
        self.stats["parallel_workers"] = max(self.stats["parallel_workers"], workers)
        if workers == 1:
            results = [(row, *self._compute_prefix(row, count)) for row in missing]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = [
                    (row, *result)
                    for row, result in zip(
                        missing,
                        executor.map(lambda item: self._compute_prefix(item, count), missing),
                    )
                ]
        for row, costs, controls in results:
            self._cache_prefix(row, count, costs, controls)

        for local, row in enumerate(rows):
            costs, controls, _ = self.rows[int(row)]
            size = min(count, len(costs))
            distances[local, :size], indices[local, :size] = costs[:size], controls[:size]
        return distances, indices
