"""centroid_graph_geodesic: data-aware shortest-path route over a centroid cloud.

Builds a k-nearest-neighbor graph over a set of class centroids (edge weight =
Euclidean distance between centroids), runs Dijkstra between a start and end
centroid, and materializes the piecewise-linear path along the resulting route.
It is the path-construction primitive a steering analysis uses to walk between two
distant centroids *through* intermediate centroids, instead of cutting straight
across the space.

This is a *method* (interpretability primitive) — see `ARCHITECTURE.md` §3.
Layering rules respected by this module:
  - imports only from third-party libs (no causalab.* needed)
  - no imports from causalab/runner/ or causalab/analyses/
  - no hyperparameter defaults (the consuming analysis's Hydra config supplies them)
  - no disk I/O (the consuming analysis decides where results land)
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.sparse.csgraph import shortest_path
from sklearn.neighbors import kneighbors_graph


def centroid_graph_geodesic(
    centroids: torch.Tensor,
    labels: list[str],
    start: str,
    end: str,
    *,
    k: int,
    steps_per_segment: int,
) -> dict:
    """Build a k-NN + Dijkstra route over ``centroids`` and the path along it.

    Parameters
    ----------
    centroids : torch.Tensor
        ``(n, d)`` float tensor; one centroid per class, in the working space
        (e.g. PCA-k). Row ``i`` corresponds to ``labels[i]``.
    labels : list[str]
        Length ``n``; label for each centroid row.
    start, end : str
        Endpoint labels; both must appear in ``labels``.
    k : int
        Number of neighbors in the k-NN graph (hyperparameter, no default).
        Auto-incremented if ``start`` and ``end`` land in different connected
        components, up to ``n - 1``.
    steps_per_segment : int
        Number of interpolation points per consecutive-waypoint segment
        (hyperparameter, no default). Must be ``>= 2``.

    Returns
    -------
    dict
        ``{
            "route_labels":   list[str],     # ordered waypoint labels, start..end
            "route_indices":  list[int],     # indices into centroids/labels
            "path_points":    torch.Tensor,  # (S, d) concatenated linear segments
            "segment_lengths": list[float],  # Euclidean length per route edge
            "k_used":         int,           # final k after connectivity bump
        }``

    Notes
    -----
    The graph metric is fixed to Euclidean in the working space by design (a
    locked decision: the graph is built in PCA space, not in any external
    coordinate system, to avoid presupposing the geography under test).
    """
    if steps_per_segment < 2:
        raise ValueError(f"steps_per_segment must be >= 2, got {steps_per_segment}")
    if start not in labels:
        raise ValueError(f"start label {start!r} not in labels")
    if end not in labels:
        raise ValueError(f"end label {end!r} not in labels")

    n = len(labels)
    if centroids.shape[0] != n:
        raise ValueError(
            f"centroids has {centroids.shape[0]} rows but {n} labels were given"
        )
    if start == end:
        raise ValueError("start and end must be different labels")

    src = labels.index(start)
    dst = labels.index(end)

    # Work in float64 numpy for the graph; keep the original dtype/device for output.
    feats = centroids.detach().cpu().numpy().astype(np.float64)

    # Connectivity guard: grow k until src and dst are in the same component.
    k_used = max(1, min(int(k), n - 1))
    predecessors = None
    while True:
        # Symmetric, distance-weighted k-NN adjacency.
        adj = kneighbors_graph(
            feats, n_neighbors=k_used, mode="distance", include_self=False
        )
        adj = adj.maximum(adj.T)  # mutual-or-either -> undirected

        dist_matrix, preds = shortest_path(
            adj, method="D", directed=False, indices=src, return_predecessors=True
        )
        if np.isfinite(dist_matrix[dst]):
            predecessors = preds
            break
        if k_used >= n - 1:
            raise ValueError(
                f"{start!r} and {end!r} remain disconnected even at k={k_used} "
                f"(n={n}); centroid graph cannot be connected."
            )
        k_used += 1

    # Reconstruct the route src -> dst by walking predecessors backward from dst.
    route_indices: list[int] = []
    node = dst
    while node != src and node >= 0:
        route_indices.append(int(node))
        node = int(predecessors[node])
    if node != src:
        raise ValueError(f"failed to reconstruct route from {start!r} to {end!r}")
    route_indices.append(src)
    route_indices.reverse()

    route_labels = [labels[i] for i in route_indices]

    # Per-edge Euclidean lengths in the working space.
    segment_lengths = [
        float(np.linalg.norm(feats[route_indices[i + 1]] - feats[route_indices[i]]))
        for i in range(len(route_indices) - 1)
    ]

    # Assemble the piecewise-linear path; drop duplicated segment boundaries.
    pts = centroids.detach()
    segments: list[torch.Tensor] = []
    for i in range(len(route_indices) - 1):
        a = pts[route_indices[i]]
        b = pts[route_indices[i + 1]]
        alphas = torch.linspace(
            0.0, 1.0, steps_per_segment, device=a.device, dtype=a.dtype
        )
        seg = a.unsqueeze(0) + alphas.unsqueeze(1) * (b - a).unsqueeze(0)
        # Drop the first point of every segment after the first to avoid
        # duplicating the shared waypoint.
        segments.append(seg if i == 0 else seg[1:])
    path_points = torch.cat(segments, dim=0)

    return {
        "route_labels": route_labels,
        "route_indices": route_indices,
        "path_points": path_points,
        "segment_lengths": segment_lengths,
        "k_used": int(k_used),
    }
