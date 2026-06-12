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

**Run-1 verdict (2026-06-12):** all three patch arms came back bit-identical and
`shape_probe` showed `f_base_shape == [16, 2560]` — the executor is exonerated and the
rank-mismatch hypothesis is dead. The patched-country top-5s (Spain → Portugal/France;
Russia → Poland/Ukraine/Belarus) revealed the real bug: the intervened variable is the
question's *subject* country and the causal model answers with its *neighbor*
(`raw_output = NEIGHBOR_OF[(country, direction)]`), so `p_target_norm` /
`argmax_match_rate` score a token that is never the correct answer. The readout now
also reports the correct metrics:

- `p_expected_norm` / `argmax_expected_rate` / `p_canonical_norm`: per-prompt mass /
  argmax-rate / canonical-neighbor mass on `NEIGHBOR_OF[(patched_country,
  prompt_direction)]` — what locate's string_match scoring implicitly measured
  (hence its 0.41 while every p(target)-based landscape sat at chance). Prompts whose
  (patched country, direction) cell has no in-set neighbor are excluded
  (`n_prompts_with_expected` counts the rest). Compare each patch arm against
  `no_patch` on these.
- `p_target_norm` / `argmax_match_rate` (kept for continuity): mass on the patched
  country itself — the *wrong* readout for this task; expect ≈ chance even when the
  intervention works.
- `results.json → shape_probe`: gathered `f_base` rank inside the interpolation
  intervention (run 1: already `(B, H)`, no mismatch).
- `no_patch` calibrates the floor for both readouts.

### Saved artifacts

| File | Contents |
|---|---|
| `results.json` | `{site, value_labels, shape_probe, per_country.<C>.<arm>.{p_expected_norm, p_canonical_norm, argmax_expected_rate, n_prompts_with_expected, p_target_norm, p_target_fullvocab, argmax_match_rate, mean_norm_dist, top5}}` |
| `metadata.json` | resolved knobs + site + model/task provenance |
