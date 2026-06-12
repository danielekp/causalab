# Graph Path Steering

Graph Path Steering answers: *Does steering the model's answer along a graph-geodesic route between two
distant countries hand prediction mass off through geographically-intermediate countries, more than
straight-line steering?* It is a session-local, stripped-down sibling of shipped `path_steering`: for a
configured endpoint pair it builds up to three steering paths and records the per-step output distribution
over the answer-country tokens, then compares how each path's argmax country sweeps along the path.

It sits at the end of the country_borders geometry chain (`baseline → locate → subspace →
activation_manifold → graph_path_steering`). It reuses the fitted PCA subspace and spline manifold rather
than re-fitting them, and reuses shipped `path_steering` machinery (`resolve_path_modes`,
`collect_grid_distributions`, `plot_saved_pair_distributions`) plus the session-local method
`centroid_graph_geodesic`.

The three path modes:
- **`graph_geodesic`** (new) — k-NN graph over the PCA country centroids → Dijkstra route; steered in **raw
  activation space** (identity featurizer) along the route countries' raw centroids — the same mechanism as
  `linear`, differing only by the route (through intermediates vs. straight). The graph (route selection)
  lives in PCA; the steering lives in raw space. (Steering in the PCA subspace via an inverse-PCA lift was a
  no-op — the reconstruction dropped the off-subspace activation mass, so the patch had no effect.)
- **`geometric`** (baseline) — geodesic in the manifold's intrinsic (lat, lon) coordinates.
- **`linear`** (baseline) — straight line in raw activation space.

---

## Configuration

**Root config** — shared params used by this analysis:
- `experiment_root` — output root (session-local: `agent_logs/<session>/artifacts/${task.name}/${model.id}`)
- `seed` — eval-prompt generation seed (eval prompts use `seed + 100`)
- `model.*` — model name/device/dtype for the steered forward passes
- `task.target_variable`, `task.max_new_tokens`, `task.colormap`, `task.color_by_dim`

**Module config** (`${SESSION_DIR}/code/configs/analysis/graph_path_steering.yaml`):

```yaml
graph_path_steering:
  _name_: graph_path_steering
  _subdir: ${.subspace}/${.activation_manifold}
  _output_dir: ${experiment_root}/graph_path_steering/${._subdir}
  subspace: pca_k32                    # subspace dir to read centroids/PCA from
  activation_manifold: L33_last_token/spline_s0.0  # manifold cell for the featurizer
  selected_pairs: [["Spain", "Russia"]]            # endpoint pairs to steer
  path_modes: [graph_geodesic, geometric, linear]  # paths to build & compare
  graph_k: 4                           # k-NN neighbors (auto-bumped if endpoints disconnect)
  steps_per_segment: 6                 # interpolation steps per graph-route segment
  num_steps_along_path: 25             # steps for the geometric/linear straight paths
  n_eval_samples: 64                   # counterfactual eval prompts generated
  n_prompts: 16                        # base prompts steered per step
  batch_size: 16                       # inference batch size
  visualization:
    figure_format: png
```

---

## Outputs

### Interpretation

- **`route.json`** — the Dijkstra country chain (Spain → … → Russia) for `graph_geodesic`, with per-edge
  lengths and the final `k_used`. A sensible chain of bordering countries supports H1; a single long jump
  or non-adjacent hops falsifies it. `<pair>_oracle` entries record the hand-specified geographic route
  steered by the `oracle` mode (same raw-space piecewise mechanism as `graph_geodesic` — isolates route
  *quality* from route *construction*; run-2 found the k-NN route geographically wrong, e.g. Italy–Russia).
- **`graph_quality.json`** — the PCA-centroid k-NN graph's undirected edges vs the task's true border
  graph (symmetrized NEIGHBOR_OF): `edge_precision`/`edge_recall`/`edge_f1` plus the explicit
  false-positive (activation-adjacent, not bordering) and false-negative (bordering, not
  activation-adjacent) edge lists. Low precision falsifies the H3 premise that activation proximity
  encodes geographic adjacency.
- **`intermediate_coverage.json`** — per path mode (corrected readout, patch_parity verdict 2026-06-12:
  the steered variable is the question's *subject*; the model answers with its *neighbor*, so route
  coverage must be decoded, not read off the answer tokens):
  - `subject_decode_sequence` — per step, the subject S whose neighbor signature
    `NEIGHBOR_OF[(S, prompt_direction)]` best matches the answer distribution (prompt-averaged). This is
    the route as the model "understood" it; `n_intermediate_subjects` / `intermediate_subjects` count
    decoded subjects other than the endpoints. `graph_geodesic` having more intermediate subjects than
    `geometric`/`linear` supports H2/H3.
  - `answer_argmax_sequence` — the raw top-1 answer token per step (the model's actual answers; under a
    working steer these are *neighbors* of the route, not the route).
  - `subject_signature_matrix` — (num_steps × 30) prompt-averaged signature mass per candidate subject,
    for plotting the hand-off directly.
- **`vis/paths/<mode>/pair_Spain_Russia.png`** — output-probability landscape along each path (answer
  tokens — read it as the neighbor shadow of the route).

### Saved artifacts

| File | Shape / Format | Used by |
|---|---|---|
| `route.json` | `{ "<start>_<end>": {pair, route_labels, segment_lengths, k_used} }` | `/interpret-experiment` |
| `intermediate_coverage.json` | `{ "<start>_<end>": { "<mode>": {subject_decode_sequence, n_intermediate_subjects, intermediate_subjects, answer_argmax_sequence, subject_signature_matrix} } }` | `/interpret-experiment` |
| `vis/paths/<mode>/pair_<start>_<end>.png` | matplotlib landscape | human reference |
| `metadata.json` | run-config snapshot | provenance |
