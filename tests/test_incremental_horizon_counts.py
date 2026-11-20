import numpy as np
import pandas as pd

from rdmatcher.matcher import Matcher


class _FixedNeighbors:
    """Return a deterministic candidate matrix for one prefilter batch."""

    def __init__(self, distances, indices):
        self.distances = distances
        self.indices = indices

    def kneighbors(self, queries, n_neighbors, **kwargs):
        rows = np.asarray(queries)[:, 0].astype(int)
        assert 1 <= n_neighbors <= self.distances.shape[1]
        return self.distances[rows, :n_neighbors].copy(), self.indices[rows, :n_neighbors].copy()


def test_incremental_horizon_counts_every_safe_loss_for_the_same_row():
    """A row that loses two safe edges in one round must expand twice more.

    At rank 2, rows B and C respectively activate controls 0 and 3, both
    previously owned solely by row A.  A then gains control 5 at rank 3 and
    control 8 at rank 4.  If repeated owners are decremented only once, A
    incorrectly appears complete at rank 3 and its final candidate list has
    only one truly safe control.
    """
    frame = pd.DataFrame(
        {
            "patient_id": np.arange(12),
            "exposure_status": [1, 1, 1] + [0] * 9,
            "x": np.arange(12, dtype=float),
        }
    )
    matcher = Matcher(
        frame,
        exposure_status="exposure_status",
        patient_id="patient_id",
        distance_metric="euclidean",
        threshold=np.inf,
        n_neighbors=2,
    )
    matcher.nbrs_model = _FixedNeighbors(
        distances=np.tile(np.arange(5, dtype=np.float32), (3, 1)),
        indices=np.array(
            [
                [0, 3, 5, 8, 1],  # A: owns 0 then 3; later needs 5 and 8
                [1, 4, 0, 6, 2],  # B steals 0 at rank 2
                [2, 4, 3, 7, 1],  # C steals 3 at rank 2
            ],
            dtype=np.int32,
        ),
    )

    candidates = matcher._prefilter_candidates(
        k_candidates=5,
        safe_matches=2,
        fuzzy_threshold=False,
        fuzzy_limit=None,
        batch_size=3,
    )

    np.testing.assert_array_equal(candidates[0]["safe_positions"], [5, 8])
    assert candidates[0]["candidate_horizon"] == 4
