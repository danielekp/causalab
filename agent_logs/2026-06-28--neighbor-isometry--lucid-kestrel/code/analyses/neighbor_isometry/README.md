# neighbor_isometry

neighbor_isometry answers: *Within each compass direction, is Gemma3-4B-PT's neighbor-output
distribution (grouped by subject country) isometric to the subject's L12 activation geography — and
stronger than the direction-marginalized baseline?* It builds, per direction, a 2-D subject-geography
activation manifold (cached L12 `pca_k32` features, standardized) and a 2-D Hellinger behavior
manifold (cached `per_example_output_dists`), both with control points = subject capital
`[lat, lon]`, then reuses `compute_isometry_from_manifolds` to score geometric + linear isometry.

It is the single executed node of the `neighbor-isometry--lucid-kestrel` session: a **model-free**
post-processing analysis that consumes artifacts cached by `subspace` and `output_manifold` (it never
loads model weights). It realizes the user-selected *per-direction subject walk* as 8 stratified 2-D
manifolds, because the 3-D joint `(lat, lon, direction)` manifold is not config-achievable
(`train_spline_manifold` excludes non-target variables).

---

## Configuration

**Root config** — shared params:
- `experiment_root` — output + cache root (here the global pod tree
  `/workspace/causalab/artifacts/country_borders/gemma3_4b_pt`).
- `seed` — dataset-reconstruction seed (must match the cached run).
- `task.*` — `name`, `target_variable`, `n_train`, `n_test`, `enumerate_all`, `resample_variable`
  (must reproduce the dataset order the cache was built from; verified by the alignment gate).

**Module config** (`configs/analysis/neighbor_isometry.yaml`):

```yaml
neighbor_isometry:
  _name_: neighbor_isometry
  _subdir: default
  _output_dir: ${experiment_root}/neighbor_isometry/${._subdir}
  subspace: pca_k32            # subspace dir holding the cached activation features
  activation_features: null    # null -> auto-discover training_features.safetensors
  output_manifold_sub: null    # null -> auto-discover per_example_output_dists.safetensors
  directions: null             # null -> all 8 compass directions
  n_arc_steps: 150             # geodesic arc resolution (matches prior run)
  n_interior_per_pair: 1       # interior points per pair geodesic (denser scatter; per-direction r)
  smoothness: 0.0              # TPS smoothness for both manifolds (exact interpolation)
  path_modes: [geometric, linear]
  baseline_geometric_r: 0.204  # prior marginalized-subject isometry, for the summary comparison
  baseline_linear_r: 0.033
  visualization:
    figure_format: pdf
```

---

## Outputs

### Interpretation

- **`summary.json`** — the headline. `per_direction.<DIR>.{geometric_r, linear_r}` = isometry per
  compass direction; `aggregate.<mode>.{mean_r, min_r, max_r, pooled_centroid_r}` = across-direction
  summary; `baseline` = the marginalized-subject reference (0.204 / 0.033). **H1 supported** if the
  per-direction / mean geometric r sits well above 0.204; **falsified** if it does not.
- **`alignment_match_rate`** (in summary) — sanity number; a correctly aligned cache yields a rate
  ≈ in-set accuracy. The run **raises** below 0.20 (misaligned cache → scores would be meaningless).
- **`dir_<D>/criteria/isometry/<mode>/{metrics.json,tensors.safetensors}`** — per-direction isometry
  D-matrices + Pearson r, in the same layout as `path_steering` so the figure script reads them
  unchanged.
- **`dir_<D>/manifolds.safetensors`** — fitted control points / centroids / preprocess mean-std, for
  the figure script to rebuild the manifolds model-free.

### Saved artifacts

| File | Shape / Format | Used by |
|---|---|---|
| `summary.json` / `results.json` | nested dict (per-direction + aggregate r, baseline) | interpret-experiment, human |
| `dir_<D>/criteria/isometry/<mode>/metrics.json` | `{pearson_r, n_pairs, …}` | figure script, human |
| `dir_<D>/criteria/isometry/<mode>/tensors.safetensors` | `D_manifold, D_output, grid_points_valid(+_belief)` | figure script (Fig-3 composite) |
| `dir_<D>/manifolds.safetensors` | `act/bel_control_points, act_centroids, act_mean, act_std, bel_centroids_prob` | figure script |
| `metadata.json` | resolved-config snapshot | provenance |
