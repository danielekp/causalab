"""Shape, route, and behavior tests for centroid_graph_geodesic.

Lives at ${SESSION_DIR}/code/methods/centroid_graph_geodesic/tests/.

Test design note: in Euclidean space the straight skip edge between two points is
never longer than going through a colinear intermediate (triangle inequality, with
equality when colinear). So routing *through* intermediates is forced only by graph
sparsity, not geometry. The fixtures below use strictly-increasing gaps with k=1 so
each node's unique nearest neighbor is its predecessor — yielding a deterministic
pure chain with no skip edges.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from methods.centroid_graph_geodesic import centroid_graph_geodesic


def _chain_centroids(n: int = 6, d: int = 4) -> tuple[torch.Tensor, list[str]]:
    """n centroids on a 1-D axis with strictly increasing gaps (1, 2, 3, ...).

    With k=1 the unique nearest neighbor of node i (i>0) is node i-1, so the
    symmetrized graph is a pure chain and the shortest route between the endpoints
    passes through every intermediate node in order.
    """
    gaps = torch.arange(n, dtype=torch.float32)  # 0,1,2,...; cumsum -> 0,1,3,6,...
    positions = torch.cumsum(gaps, dim=0)
    coords = torch.zeros(n, d, dtype=torch.float32)
    coords[:, 0] = positions
    labels = [f"c{i}" for i in range(n)]
    return coords, labels


def test_shape_and_dtype():
    coords, labels = _chain_centroids(n=6, d=4)
    out = centroid_graph_geodesic(
        coords, labels, "c0", "c5", k=1, steps_per_segment=5
    )
    assert isinstance(out["path_points"], torch.Tensor)
    assert out["path_points"].shape[1] == 4
    assert out["path_points"].dtype == coords.dtype
    # 5 segments: first contributes 5 points, the other 4 contribute 4 each.
    assert out["path_points"].shape[0] == 5 + 4 * 4


def test_route_passes_through_intermediates():
    coords, labels = _chain_centroids(n=6, d=4)
    out = centroid_graph_geodesic(
        coords, labels, "c0", "c5", k=1, steps_per_segment=3
    )
    assert out["route_labels"] == ["c0", "c1", "c2", "c3", "c4", "c5"]
    assert out["route_indices"] == [0, 1, 2, 3, 4, 5]
    assert torch.allclose(out["path_points"][0], coords[0])
    assert torch.allclose(out["path_points"][-1], coords[5])


def test_connectivity_bump_increases_k():
    """Two tight clusters joined only by a long bridge: k=1 disconnects them."""
    coords = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
            [5.0, 0.0],
            [5.1, 0.0],
            [5.2, 0.0],
        ],
        dtype=torch.float32,
    )
    labels = [f"c{i}" for i in range(6)]
    out = centroid_graph_geodesic(coords, labels, "c0", "c5", k=1, steps_per_segment=2)
    assert out["k_used"] >= 2  # had to grow k to bridge the clusters
    assert out["route_labels"][0] == "c0"
    assert out["route_labels"][-1] == "c5"


def test_segment_lengths_match_route():
    coords, labels = _chain_centroids(n=4, d=3)  # positions 0,1,3,6 -> gaps 1,2,3
    out = centroid_graph_geodesic(
        coords, labels, "c0", "c3", k=1, steps_per_segment=2
    )
    assert len(out["segment_lengths"]) == len(out["route_indices"]) - 1
    assert np.allclose(out["segment_lengths"], [1.0, 2.0, 3.0])


def test_validation_errors():
    coords, labels = _chain_centroids(n=4, d=3)
    with pytest.raises(ValueError):
        centroid_graph_geodesic(coords, labels, "c0", "c0", k=2, steps_per_segment=2)
    with pytest.raises(ValueError):
        centroid_graph_geodesic(coords, labels, "c0", "zzz", k=2, steps_per_segment=2)
    with pytest.raises(ValueError):
        centroid_graph_geodesic(coords, labels, "c0", "c3", k=2, steps_per_segment=1)
