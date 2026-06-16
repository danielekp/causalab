# directional_isometry

*Does a **direction-conditioned** behavior manifold sharpen the activation↔behavior
isometry on country_borders?* — option (b) of the manifold-steering Fig 2/3 reproduction.

The shipped `output_manifold` → `path_steering` isometry path builds each country's
belief centroid by **averaging** its answer distribution over all prompts (hence over all
8 directions). Because country_borders is relational (the answer is a different *neighbor*
per direction), that average is direction-blurred and the isometry is weak (geometric
r≈0.20, linear r≈0.03). This analysis instead represents each country by its
**per-direction answer distributions, concatenated and √-transformed (Hellinger)**, fits a
belief manifold on those fingerprints — sharing the activation manifold's (lat, lon)
parameterization so the two align by class — and recomputes the same
`compute_isometry_from_manifolds` metric. If relational blurring caused the weak isometry,
the geometric r should rise.

It is **model-free**: the belief side reuses `output_manifold`'s cached
`per_example_output_dists.safetensors` (regrouped by `(subject, direction)`); the
activation side reuses the cached activation-manifold cell; the isometry metric needs no
forward passes. A lite (weights-free) pipeline is loaded only to rebuild the
`InterchangeTarget` for `load_featurizer`, so it never OOMs.

## Configuration

Root config:
- `experiment_root` — single source of truth for output paths.
- `seed`, `task.*` (`name`, `target_variable`, `n_train`, `n_test`, `enumerate_all`,
  `balanced`, `resample_variable`, `max_new_tokens`, `distance_function`) — used to
  regenerate the row-aligned dataset and label the scatter.
- `model.name` — model id for the lite pipeline.

Module config (`cfg.directional_isometry`):
```yaml
subspace: pca_k32_ctok                # subspace dir to read the activation manifold from
activation_manifold: L12_country/spline_s0.0   # activation-manifold cell (must be intrinsic_dim=2)
smoothness: 0.0                       # exact TPS interpolation through the fingerprints
direction_order: [N, NE, E, SE, S, SW, W, NW]  # fixed concat order for the fingerprint
n_arc_steps: 150                      # arc-length resolution along each geodesic
n_interior_per_pair: 0                # 0 = centroid pairs only; >0 densifies with interior points
path_modes: [geometric, linear]       # on-manifold arc vs straight chord
visualization: {figure_format: pdf}
```

**Caveat (zero-fill):** country_borders enumerates only the valid `(country, direction)`
cells (~166 of 30×8). Missing cells are zero-filled, so a country's fingerprint norm
grows with its number of valid directions. This is documented, not corrected — a full
30×8 model query would remove it but needs the model.

**Prerequisites:** `subspace/pca_k32_ctok`, `activation_manifold/.../L12_country/spline_s0.0`
(intrinsic_dim=2), and `output_manifold/per_example_output_dists.safetensors` must already
exist under `experiment_root`.

## Outputs

### Interpretation
- `directional_isometry_summary.json` — `pearson_r` for `geometric` vs `linear`. Compare
  against the direction-averaged option-(a) numbers (geometric 0.20 / linear 0.03). A
  higher geometric r supports the relational-blurring explanation; geometric ≫ linear is
  the qualitative "manifold is behaviorally faithful" signal.
- `criteria/isometry/<mode>/metrics.json` — full metric (`pearson_r`, `n_pairs`, …).
- `vis/isometry/<mode>/isometry_scatter.*` — D_belief vs D_activation scatter (the
  "Scaled Isometry" panel).

### Saved artifacts
| Path | Contents |
|---|---|
| `directional_isometry_summary.json` | headline r per path mode + coverage stats |
| `criteria/isometry/<mode>/metrics.json` | isometry metric dict |
| `criteria/isometry/<mode>/tensors.safetensors` | D_manifold, D_output, grid points |
| `vis/isometry/<mode>/isometry_scatter.*` | scatter figure |
| `directional_fingerprints.safetensors` | (n_countries, n_dir·(W+1)) fingerprints + control points |
| `metadata.json` | run config snapshot |
