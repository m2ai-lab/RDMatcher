"""Sparse candidate-graph representation used by RDMatcher integrations."""

from dataclasses import dataclass
import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class CandidateGraph:
    """A case-by-control candidate network with distances and ID mappings.

    ``matrix[i, j]`` is the precomputed distance from ``case_ids[i]`` to
    ``control_ids[j]``. Missing entries are ineligible edges. The matrix is a
    SciPy CSR matrix and can be passed to downstream sparse-graph workflows.
    """

    matrix: csr_matrix
    case_ids: np.ndarray
    control_ids: np.ndarray
    distance_metric: str
    threshold: float
    candidate_horizons: np.ndarray

    @property
    def shape(self):
        return self.matrix.shape

    @property
    def nnz(self):
        return self.matrix.nnz

    def for_scipy_sparse_matching(self) -> csr_matrix:
        """Return positive edge weights suitable for SciPy's sparse matcher.

        SciPy interprets stored zero weights as absent edges. Adding one to
        each stored distance preserves the optimum for fixed-cardinality
        assignments and keeps genuine zero-distance edges present.
        """
        result = self.matrix.copy()
        result.data = result.data + 1.0
        return result


def candidate_graph_from_prefilter(
    candidate_list,
    n_controls: int,
    case_ids,
    control_ids,
    distance_metric: str,
    threshold: float,
) -> CandidateGraph:
    """Create a CSR graph from the candidate arrays already produced by RDM."""
    rows, cols, distances = [], [], []
    horizons = np.empty(len(candidate_list), dtype=np.int32)

    for row, candidates in enumerate(candidate_list):
        horizons[row] = int(candidates.get("candidate_horizon", 0))
        positions = np.concatenate(
            (candidates["safe_positions"], candidates["competitive_positions"])
        )
        costs = np.concatenate(
            (candidates["safe_distances"], candidates["competitive_distances"])
        )
        if positions.size:
            valid = (
                (positions >= 0)
                & (positions < n_controls)
                & np.isfinite(costs)
                & (costs <= threshold)
            )
            positions = positions[valid]
            costs = costs[valid]
            rows.extend([row] * len(positions))
            cols.extend(positions.tolist())
            distances.extend(costs.astype(np.float64, copy=False).tolist())

    matrix = csr_matrix(
        (np.asarray(distances, dtype=np.float64), (rows, cols)),
        shape=(len(candidate_list), n_controls),
        dtype=np.float64,
    )
    return CandidateGraph(
        matrix=matrix,
        case_ids=np.asarray(case_ids).copy(),
        control_ids=np.asarray(control_ids).copy(),
        distance_metric=str(distance_metric),
        threshold=float(threshold),
        candidate_horizons=horizons,
    )
