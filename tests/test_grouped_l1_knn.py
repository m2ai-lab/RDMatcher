import numpy as np
import pandas as pd

from rdmatcher.distance import GowerKNN


def test_grouped_l1_tree_matches_full_gower_and_reuses_cache(monkeypatch):
    """Grouped top-k includes tied boundaries and retains baseline ordering."""
    rng = np.random.default_rng(7)
    controls = pd.DataFrame(
        {
            "x1": rng.integers(0, 7, size=120),
            "x2": rng.integers(0, 5, size=120),
            "x3": rng.integers(0, 4, size=120),
            "cat1": rng.choice(["a", "b", "c"], size=120),
            "cat2": rng.choice(["u", "v"], size=120),
        }
    )
    queries = pd.DataFrame(
        {
            "x1": rng.integers(0, 7, size=17),
            "x2": rng.integers(0, 5, size=17),
            "x3": rng.integers(0, 4, size=17),
            "cat1": rng.choice(["a", "b", "c", "unseen"], size=17),
            "cat2": rng.choice(["u", "v"], size=17),
        }
    )
    model = GowerKNN(
        weights=[0.2, 0.3, 0.1, 0.25, 0.15], cat_features=["cat1", "cat2"]
    ).fit(controls, seed=11)

    monkeypatch.setenv("RD_MATCHER_GROUPED_L1_KNN", "0")
    expected_dists, expected_indices = model.kneighbors(queries, k=12, n_jobs=1, streaming="off")
    monkeypatch.setenv("RD_MATCHER_GROUPED_L1_KNN", "1")
    actual_dists, actual_indices = model.kneighbors(queries, k=12, n_jobs=1, streaming="off")
    np.testing.assert_array_equal(actual_indices, expected_indices)
    np.testing.assert_array_equal(actual_dists, expected_dists)

    positions_id, trees_id = id(model._grouped_l1_positions_), id(model._grouped_l1_trees_)
    cached_dists, cached_indices = model.kneighbors(queries, k=12, n_jobs=1, streaming="off")
    assert id(model._grouped_l1_positions_) == positions_id
    assert id(model._grouped_l1_trees_) == trees_id
    np.testing.assert_array_equal(cached_indices, expected_indices)
    np.testing.assert_array_equal(cached_dists, expected_dists)
