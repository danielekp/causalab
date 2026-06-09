# patch_parity

*Is the steering executor (`run_interpolation_interventions` + `replace_fn`) causally
equivalent to locate's centroid mechanism (interchange intervention +
`source_representations`) when patching the same raw-space country centroid at the same
site?* This diagnostic exists because on country_borders/Gemma3-4B-PT the locate
centroid scan scores 0.41 at L33/last_token while every steering landscape through
`collect_grid_distributions` sits at the 1/30 chance floor — same patch value, two
executors, contradictory results (see the session's `result/steering_review.md`,
experiment 1). It patches each configured country's raw centroid into one fixed set of
prompts through four arms — `interchange_source` (locate's mechanism),
`interpolation_replica` (verbatim copy of the steering executor's `replace_fn`,
instrumented with shape logging), `interpolation_shapefix` (same executor, replacement
expanded to `f_base`'s exact shape), and `no_patch` (floor) — and records
concept-normalized and full-vocab readouts per (country, arm). It reads only the
`subspace` analysis's saved artifacts and is terminal: its verdict decides whether
`path_steering`/`graph_path_steering` results are trustworthy.

## Configuration

Root-config params read: `experiment_root` (artifact root; also where the upstream
`subspace/` artifacts are found), `seed` (eval prompts use `seed + 100`, matching
`graph_path_steering`), `task.*` (task name, `target_variable`, `max_new_tokens`),
`model.*` (name, device, dtype, `model_class`).

```yaml
patch_parity:
  subspace: pca_k32            # subspace dir providing raw features + (layer, token) site
  countries: ["Spain", "Russia"]  # one full arm set per country
  arms: [interchange_source, interpolation_replica, interpolation_shapefix, no_patch]
  n_eval_samples: 64           # counterfactual pool size (generate_dataset, seed+100)
  n_prompts: 16                # prompts actually patched, shared across all arms
  batch_size: 16               # inference batch size
  interpolation_self_pairs: false  # true = self-pairs in arms B/B2 (confound check)
```

## Outputs

### Interpretation

- `results.json → per_country.<C>.<arm>`: compare `p_target_norm` and
  `argmax_match_rate` across arms. **`interpolation_replica ≈ no_patch` while
  `interchange_source` is well above** → the steering executor silently drops the
  patch; if `interpolation_shapefix` recovers `interchange_source`'s numbers, the
  `(B,1,H)` vs `(B,H)` rank mismatch in `collect.py`'s `replace_fn` is confirmed and
  the fix is to expand the replacement to `f_base`'s shape.
  **`interpolation_replica ≈ interchange_source`** → the executor is fine and the flat
  landscapes were a full-vocab readout artifact: re-read existing steering results
  concept-normalized (review experiment 2).
- `results.json → shape_probe`: gathered `f_base` rank inside the interpolation
  intervention. `f_base_shape == [B, 1, H]` with `replica_out_shape == [B, H]` is the
  smoking gun for the rank hypothesis.
- `no_patch` also calibrates the floor: per-country `p_target_norm` under no
  intervention (≈ the model's marginal preference for that country).

### Saved artifacts

| File | Contents |
|---|---|
| `results.json` | `{site, value_labels, shape_probe, per_country.<C>.<arm>.{p_target_norm, p_target_fullvocab, argmax_match_rate, mean_norm_dist, top5}}` |
| `metadata.json` | resolved knobs + site + model/task provenance |
