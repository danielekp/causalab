# setup-analyses spec — `neighbor_isometry` (session-local, custom)

Consumed by `/setup-analyses`. Scaffolds `${SESSION_DIR}/code/analyses/neighbor_isometry/` +
`${SESSION_DIR}/code/configs/analysis/neighbor_isometry.yaml`. **Model-free** — must never call
`load_pipeline`; use `load_lite_pipeline` only if a tokenizer/causal-model handle is needed.

## Purpose

Per-direction stratified activation↔behavior isometry for `country_borders` / Gemma3-4B-PT. Tests
whether conditioning the output readout on direction (one canonical neighbor per (subject,direction)
cell) recovers a stronger isometry than the direction-marginalized baseline (geometric r=0.204,
linear r=0.033). Realizes the user-selected *per-direction subject walk* as 8 independent 2-D
subject-geography manifolds (the 3-D joint manifold is not config-achievable; see PLAN §E).

## Inputs (all under `cfg.experiment_root`, model-free)

- `subspace/pca_k32/**/features/training_features.safetensors` — key `features`, per-example L12
  `country`-token activations in the `pca_k32` subspace. Auto-discover the cell dir under
  `subspace/pca_k32/`; surface the resolved path in the log.
- `output_manifold/**/per_example_output_dists.safetensors` — key `dists`, shape `(664, 31)` =
  softmax mass on the 30 country score-tokens + "other".
- `output_manifold/**/hellinger_pca.safetensors` (+ `.meta.json`) — for the figure's belief PCA views
  (not required for the r computation).
- `country_borders` causal model + `config.py` (`COUNTRIES`, `DIRECTIONS`, `NEIGHBOR_OF`,
  `LAT_LON_OF`, `VALID_CELLS`) — for the deterministic example→(country,direction,template) map and
  the subject lat/lon control coordinates.

## Example-order reconstruction (CRITICAL — gate before grouping)

The cached tensors are ordered by the dataset's enumeration. Reconstruct that order from the task
(same loader the baseline used) to label each row with (country, direction, template). **Pre-flight
gate:** for a sample of rows, `COUNTRY_FIRST_TOKEN`-decode of `argmax(dists[row])` must be a valid
neighbor of the reconstructed (country,direction) cell at a rate ≈ baseline accuracy. If alignment
fails, raise and stop — do not produce scores. (Prefer reading a baseline-saved manifest of example
metadata if one exists; fall back to deterministic reconstruction only if it passes the gate.)

## Algorithm (per direction d in DIRECTIONS)

1. `idx_d` = rows whose direction == d. Subjects in stratum = countries with a valid d-neighbor.
2. **Activation manifold:** group `features[idx_d]` by subject → centroids `(W_d, k)`; control
   coords `U = [[lat, lon] for subject]` `(W_d, 2)`. Fit a 2-D parameter-mode TPS
   (`smoothness=0.0`) by constructing control points manually and calling the low-level spline fit
   (e.g. `methods/spline/builders.py` + the TPS fit in `methods/spline/`), **bypassing**
   `train_spline_manifold`'s `intervention_variable` exclusion (`train.py:361-364`).
3. **Behavior manifold:** group `dists[idx_d]` by subject → mean dist `(W_d, 31)`; `sqrt` → Hellinger;
   control coords = same `U`. Fit a 2-D TPS in sqrt-p space (mirror `output_manifold` belief fitting).
4. Both manifolds now share identical W_d control points aligned by subject index (satisfies the
   `isometry.py:264-274` contract). Call
   `compute_isometry_from_manifolds(act_mfd, bel_mfd, n_arc_steps=150, n_interior_per_pair=1)` for
   `path_mode ∈ {geometric, linear}`.
5. Save per (d, path_mode): `criteria/isometry/<mode>/tensors.safetensors` (D_X, D_Y,
   grid_points_valid) + `metrics.json` (Pearson r), matching the `path_steering` isometry layout so
   the figure script reads them unchanged. Save the fitted `activation_manifold.safetensors` /
   `belief_manifold.safetensors` (control_points + centroids + preprocess mean/std) for the figure.

## Aggregate

Write `neighbor_isometry/summary.json`:
- per-direction `{geometric_r, linear_r, W_d, n_pairs_retained}`
- pooled `{geometric_r, linear_r}` — Pearson over the concatenation of *retained* (non-same-geodesic)
  off-diagonal pairs across all 8 strata
- `baseline` = `{geometric_r: 0.204, linear_r: 0.033}` for reference

## Reuse (do not reimplement)

- `causalab/methods/scores/isometry.py::compute_isometry_from_manifolds` — the metric.
- `causalab/methods/spline/builders.py` (`compute_centroids`, parameter extraction) + the TPS fit —
  for manifold fitting with manually supplied control points.
- Hellinger = elementwise `sqrt` of the probability simplex (as `output_manifold` does).

## Config (`configs/analysis/neighbor_isometry.yaml`)

```yaml
# @package neighbor_isometry
subspace: pca_k32
activation_features: null        # auto-discover under subspace/pca_k32/**/features/
output_manifold_sub: null        # auto-discover under output_manifold/**/
n_arc_steps: 150
n_interior_per_pair: 1
smoothness: 0.0
path_modes: [geometric, linear]
directions: null                 # null → all 8 from config.DIRECTIONS
```

## Out of scope for the analysis

- No steering, no pullback, no coherence/conformal. No figure rendering (that's the separate
  session-local `make_neighbor_isometry_fig.py`, adapted from `make_fig3_composite.py`).
