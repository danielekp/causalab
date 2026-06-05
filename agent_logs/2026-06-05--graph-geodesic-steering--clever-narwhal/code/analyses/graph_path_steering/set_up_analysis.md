---
name: graph_path_steering
---

# Analysis spec: `graph_path_steering`

5-section spec consumed by `/setup-analyses`. Lands at `${SESSION_DIR}/code/analyses/graph_path_steering/set_up_analysis.md`.

A session-local prototype that extends the shipped `path_steering` analysis with a third, data-aware path
mode. It does **not** shadow `path_steering` (forbidden — dispatcher prefers shipped); it has a distinct
name and *imports* shipped `path_steering` machinery. Promotion (folding the graph-geodesic mode back into
shipped `path_steering`) is a later manual step.

---

## §1. Identity

**Research question:**

> *Does steering the model's answer along a graph-geodesic route between two distant countries hand
> prediction mass off through geographically-intermediate countries, more than straight-line steering?*

**What it does mechanically.** Forward passes only (no training). A stripped-down sibling of shipped
`path_steering` (no criteria/isometry/belief-space/dual-manifold machinery — just the steering + landscape
plot). Loads the country-centroid features and PCA from `subspace` and the fitted-manifold featurizer from
`activation_manifold` exactly as `path_steering` does (build interchange target → `load_featurizer`;
compute `pca_centroids`/`raw_centroids`/`spline_centroids` from row-aligned saved features). For the
configured pair (Spain → Russia) it builds three paths and steers along each via
`collect_grid_distributions`, recording the per-step output distribution over country tokens:
1. **graph_geodesic** (new) — operates in PCA space using the **same featurizer override as the shipped
   `linear_subspace` mode** (`featurizer.stages[0]`), but the path is the piecewise-linear route from
   `centroid_graph_geodesic(pca_centroids, ...)` instead of a straight `pm.build_path` line.
2. **geometric** (baseline) — shipped `path_steering` `geometric` PathMode (geodesic in intrinsic lat/lon).
3. **linear** (baseline) — shipped `path_steering` `linear` PathMode (straight line in raw activation space).
It then writes the route, a per-mode intermediate-country coverage summary (argmax country per step), and
one output-landscape figure per mode via `plot_saved_pair_distributions`. Terminal node — interpreted in
`/interpret-experiment`.

**Key reuse insight:** `graph_geodesic` ≡ `linear_subspace` with a custom path. Get the PCA-stage override
from `resolve_path_modes(["linear_subspace"], composed_featurizer=featurizer)[0].featurizer_override`, set
it on the interchange target, feed `centroid_graph_geodesic(...)["path_points"]` (PCA space) to
`collect_grid_distributions`; `collect_grid_distributions` inverts PCA→activation for patching.

---

## §2. Position in the DAG

### Upstream artifacts read

| Artifact | Produced by | Path |
|---|---|---|
| PCA-32 centroids + raw features | `subspace` | `${experiment_root}/subspace/pca_k32/country/features/raw_features.safetensors` (+ `train_dataset.json`); PCA via `causalab.io.sklearn_pca.load_pca` |
| fitted-manifold featurizer | `activation_manifold` | `${experiment_root}/activation_manifold/.../L33_last_token/spline_s0.0/...` (loaded via `causalab.analyses.activation_manifold.loading.load_featurizer`) |

Pin `subspace: pca_k32` and `activation_manifold: L33_last_token/spline_s0.0` explicitly (do not rely on
`null` auto-discovery) so the run reuses the exact fit from the country_borders geometry work.

### Downstream consumers

`null` — terminal; outputs are read by `/interpret-experiment`.

---

## §3. Methods used

| Symbol | Source | Notes |
|---|---|---|
| `centroid_graph_geodesic` | `methods.centroid_graph_geodesic` (session) | scaffold first via `/setup-methods` |
| `resolve_path_modes` | `causalab.analyses.path_steering.path_mode` | resolve `geometric` / `linear` baselines AND the `linear_subspace` override reused for `graph_geodesic` |
| `collect_grid_distributions` | `causalab.methods.steer.collect` | THE steering executor: `(grid_points, interchange_target, samples, var_indices) → (steps, n_prompts, W)` probs |
| `plot_saved_pair_distributions` | `causalab.analyses.path_steering.path_visualization` | per-mode output-landscape PNG |
| `resolve_task`, `generate_datasets`, `build_targets_for_grid` | `causalab.runner.helpers` | task + interchange-target setup (as in `path_steering.main`) |
| `load_pipeline` | `causalab.io.pipelines` | model load (needs GPU) |
| `load_subspace_metadata`, `load_activation_manifold_metadata` | `causalab.io.pipelines` | layer / token-position / k_features metadata |
| `load_featurizer` | `causalab.analyses.activation_manifold.loading` | load spline featurizer + manifold |
| `load_subspace_onto_target` | `causalab.analyses.subspace` | PCA-only featurizer for the `linear` raw-space baseline path, if needed |
| `tokenize_variable_values` | `causalab.methods.metric` | `var_indices` for reading the output distribution over country tokens |
| `load_counterfactual_examples` | `causalab.io.counterfactuals` | row-aligned `train_dataset.json` → centroid grouping |
| `load_file` | `safetensors.torch` | read `raw_features.safetensors` / `training_features.safetensors` |

Analysis→analysis imports are already used in this codebase (`path_steering/main.py` imports from
`activation_manifold.loading`), so reusing shipped `path_steering` helpers here is consistent with the
existing layering.

### Centroid construction (in `main.py`)

Group features by **answer country** exactly as the country_borders notebook does
(`NEIGHBOR_OF[(country, direction)][0]`), assert `len(features) == len(train_dataset)` rows before
grouping (row-alignment guard), and mean-pool per answer country to get the `(n_countries, 32)` centroid
matrix + `labels`. Pass these to `centroid_graph_geodesic`.

---

## §4. Config schema

| Knob | Type | Default | Description |
|---|---|---|---|
| `subspace` | `str \| null` | `"pca_k32"` | subspace dir to read centroids/PCA from |
| `activation_manifold` | `str \| null` | `"L33_last_token/spline_s0.0"` | manifold cell to load the featurizer from |
| `selected_pairs` | `list[list[str]]` | `[["Spain", "Russia"]]` | endpoint pairs to steer |
| `path_modes` | `list[str]` | `["graph_geodesic", "geometric", "linear"]` | paths to build & compare |
| `graph_k` | `int` | `4` | k for the k-NN centroid graph |
| `steps_per_segment` | `int` | `6` | interpolation steps per route segment |
| `num_steps_along_path` | `int` | `25` | steps for the baseline (geometric/linear) paths |
| `visualization.figure_format` | `str` | `"png"` | figure format (invariant 6) |

```yaml
_subdir: ${.subspace}/${.activation_manifold}
```

Rationale: mirrors shipped `path_steering`'s nesting (subspace / manifold) so outputs are addressable and
do not collide across subspace/manifold choices. `target_variable` (`country`) comes from `cfg.task.*`.

---

## §5. Outputs

Under `${experiment_root}/graph_path_steering/${subspace}/${activation_manifold}/${task.target_variable}/`:

| File | Format | What it shows | Used by |
|---|---|---|---|
| `route.json` | `{"pair": ["Spain","Russia"], "route_labels": [...], "segment_lengths": [...], "k_used": int}` | the Dijkstra country chain (H1 evidence) | `/interpret-experiment` |
| `intermediate_coverage.json` | `{"<mode>": {"n_intermediate_argmax": int, "argmax_sequence": ["Spain", "France", ...]}}` | distinct route-countries that become argmax along each path (H2/H3 metric) | `/interpret-experiment` |
| `vis/paths/graph_geodesic/pair_Spain_Russia.png` | matplotlib | per-step output-probability landscape, graph-geodesic | human reference |
| `vis/paths/geometric/pair_Spain_Russia.png` | matplotlib | per-step landscape, geometric baseline | human reference |
| `vis/paths/linear/pair_Spain_Russia.png` | matplotlib | per-step landscape, linear baseline | human reference |
| `metadata.json` | run-config snapshot | provenance | `/run-experiment` Step 6 |

`intermediate_coverage.json` schema detail: for each mode, `argmax_sequence` is the ordered list of the
top-1 country token at each path step; `n_intermediate_argmax` counts distinct countries in that sequence
other than the two endpoints. This is the quantitative basis for H2 (graph_geodesic > 0 intermediates) and
H3 (graph_geodesic ≥ geometric, linear).

---

## Notes (optional)

- Model closely on `causalab/analyses/path_steering/main.py` (artifact loading, featurizer/manifold load,
  per-step steering loop) and `path_visualization.py` (landscape plotting). The new mode is an additional
  `path_modes` branch whose path comes from `centroid_graph_geodesic` rather than `PathMode.build_path`.
- The graph operates in PCA space; the resulting `path_points` are already in the subspace the featurizer
  expects, so they feed the same decode/patch path as the `linear` baseline (identity beyond PCA).
- Compute: ~3–10 min on 1 GPU (3 paths × ~15–25 steps of Gemma 3 4B-PT forward passes).
