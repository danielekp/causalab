---
name: centroid_graph_geodesic
---

# Method spec: `centroid_graph_geodesic`

5-section spec consumed by `/setup-methods`. Lands at `${SESSION_DIR}/code/methods/centroid_graph_geodesic/set_up_method.md`.

---

## §1. Identity

**Purpose:** A pure geometric primitive that computes a *data-aware shortest-path route* over a cloud of
class centroids and materializes the corresponding piecewise-linear path. It builds a k-nearest-neighbor
graph over the centroids (edge weight = Euclidean distance between centroids), runs Dijkstra between a
start and end centroid, and returns the ordered waypoint route plus a concatenation of straight-line
segments between consecutive waypoints. It is the path-construction primitive a steering analysis uses to
walk between two distant centroids *through* intermediate centroids, instead of cutting straight across
the space. No model, no I/O — operates purely on the centroid tensor passed in.

---

## §2. Surface

```python
def centroid_graph_geodesic(
    centroids: torch.Tensor,   # (n, d) float; one centroid per class, in the working space (PCA-k)
    labels: list[str],         # length n; label for each centroid row (row i ↔ labels[i])
    start: str,                # start label (must be in labels)
    end: str,                  # end label (must be in labels)
    *,
    k: int,                    # k for the k-NN graph (hyperparameter — no default)
    steps_per_segment: int,    # interpolation points per consecutive-waypoint segment (no default)
) -> dict:
    """Build a k-NN + Dijkstra route over `centroids` and the piecewise-linear path along it.

    Returns:
        {
          "route_labels":  list[str],     # ordered waypoint labels, start..end inclusive
          "route_indices": list[int],     # indices into `centroids`/`labels` for each waypoint
          "path_points":   torch.Tensor,  # (S, d) concatenated linear segments in centroid space;
                                          #   S = (len(route)-1) * steps_per_segment - overlaps,
                                          #   duplicate segment-boundary points removed
          "segment_lengths": list[float], # Euclidean length of each consecutive-waypoint edge
          "k_used":        int,           # final k after any connectivity bump (>= k)
        }
    """
```

**Behavioral notes (implementation requirements):**
- **Connectivity guard:** if the start and end are not connected in the k-NN graph, increase `k` by 1
  and rebuild until they connect (cap at `n-1`); report the final value as `k_used`. If still
  disconnected at `k = n-1`, raise `ValueError`.
- **Graph is symmetric:** symmetrize the k-NN adjacency (mutual-or-either) before shortest-path so edges
  are undirected.
- **Path assembly:** for each consecutive waypoint pair `(a, b)`, emit `steps_per_segment` points via
  `linspace(centroids[a], centroids[b])`; drop the duplicated boundary point shared with the next segment
  so `path_points` is a clean ordered polyline from `start` to `end`.

**Rule compliance:** `k` and `steps_per_segment` are hyperparameters → keyword-only, **no defaults**
(ARCHITECTURE §3 Inv 5). The graph metric is fixed to Euclidean in PCA space by design (locked decision —
not exposed as a knob to avoid implying the lat/lon geography we are testing for).

---

## §3. Dependencies

| Symbol | Source | Why |
|---|---|---|
| `torch` | third-party | input/output tensors, `linspace` interpolation |
| `numpy` | third-party | array glue for the graph libs |
| `kneighbors_graph` | `sklearn.neighbors` | build the sparse k-NN distance graph |
| `shortest_path` | `scipy.sparse.csgraph` | Dijkstra over the sparse graph (`method="D"`, `return_predecessors=True`) |

No imports from `causalab/` (pure primitive). **Forbidden** (`causalab/runner/`, `causalab/analyses/`): none used.

---

## §4. Hyperparameters

| Name | Type | Range / values | Description |
|---|---|---|---|
| `k` | `int` | `1 .. n-1` | neighbors in the k-NN graph; auto-bumped if start/end disconnected |
| `steps_per_segment` | `int` | `>= 2` | interpolation points per consecutive-waypoint segment |

Defaults for these live in the consuming analysis's `analysis.yaml`, not here.

---

## §5. Side effects

`None`. Returns an in-memory dict; the consuming analysis decides where (and whether) to persist
`route_labels` / `path_points`.

---

## Notes (optional)

- `path_points` is intended to feed the same steering/decoding machinery `path_steering` uses for its
  `geometric`/`linear` paths (i.e. featurize → patch → forward → read output distribution).
- Reference for shape conventions: `causalab/analyses/path_steering/path_mode.py` `_build_linear_path_kd`
  (single straight segment in feature space); this method is the multi-segment generalization driven by a
  graph route.
- Runtime: negligible (< 1 s) for n ≈ 30 centroids.
