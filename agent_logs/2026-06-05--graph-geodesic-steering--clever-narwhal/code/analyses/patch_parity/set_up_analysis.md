---
name: patch_parity
---

# Analysis spec: `patch_parity`

5-section spec consumed by `/setup-analyses`. Implements experiment 1 of
`result/steering_review.md`: the patch-parity A/B that decides whether the flat
country_borders steering landscapes are an executor bug or a readout artifact.

---

## §1. Identity

**Research question:**

> *Is the steering executor (`run_interpolation_interventions` + `replace_fn`) causally
> equivalent to locate's centroid mechanism (interchange intervention +
> `source_representations`) when patching the same raw-space country centroid at the
> same (layer, token_position) site?*

**What the analysis does mechanically.** Loads the subspace step's saved
`raw_features.safetensors` + row-aligned `train_dataset.json`, computes per-country
raw-activation centroids (same construction as the session's `graph_path_steering`),
builds one fixed set of eval prompts, and for each configured target country patches
that country's raw centroid into the prompts' L{layer}/{token_position} site through
four arms:

- **`interchange_source`** (arm A): interchange intervention + `source_representations`
  — the exact mechanism `run_centroid_layer_scan` (locate centroid mode) uses, which
  scored 0.41 at L33/last_token.
- **`interpolation_replica`** (arm B): exact replica of
  `collect_grid_distributions`'s internal `replace_fn` (including the
  `target.unsqueeze(0).expand(B, -1)` rank behavior) run through
  `run_interpolation_interventions` — the steering executor under suspicion. Logs
  `f_base` / `f_src` / output shapes into the results (the (B,1,H)-vs-(B,H) probe).
- **`interpolation_shapefix`** (arm B2): same executor, but `replace_fn` returns the
  centroid expanded to `f_base`'s *exact* shape (`expand_as`) — isolates the
  rank-mismatch hypothesis.
- **`no_patch`** (arm C): plain generation on the same prompts — the no-effect floor.

Per (country, arm) it records concept-normalized and full-vocab P(target), the
concept-argmax match rate, and the mean concept-normalized distribution; everything
lands in one `results.json`. Forward passes only, no training, no figures.

---

## §2. Position in the DAG

### Upstream artifacts read

| Artifact | Produced by | Path |
|---|---|---|
| raw activations (N, 2560) | `subspace` | `${experiment_root}/subspace/${.subspace}/<target_variable>/features/raw_features.safetensors` |
| row-aligned dataset | `subspace` | `${experiment_root}/subspace/${.subspace}/<target_variable>/train_dataset.json` |
| site metadata (layer, token_position) | `subspace` | via `load_subspace_metadata(root, subspace, target_variable)` |

No baseline / manifold artifacts needed — the manifold featurizer is deliberately
**not** loaded; all arms patch in raw activation space through identity featurizers.

### Downstream consumers

None — terminal diagnostic. Its verdict gates how
`graph_path_steering` / `path_steering` results are read (executor bug vs readout
artifact), per `result/steering_review.md`.

---

## §3. Methods used

All shipped; no session-local methods required.

| Symbol | Source | Notes |
|---|---|---|
| `resolve_task`, `build_targets_for_grid` | `causalab.runner.helpers` | task + interchange target at the configured site |
| `load_pipeline`, `load_subspace_metadata` | `causalab.io.pipelines` | model + site discovery |
| `load_counterfactual_examples` | `causalab.io.counterfactuals` | row-aligned dataset for centroids |
| `load_task_counterfactuals` | `causalab.tasks.loader` | eval-prompt generation (same as graph_path_steering) |
| `prepare_intervenable_model`, `delete_intervenable_model`, `device_for_layer` | `causalab.neural.activations.intervenable_model` | arm A executor |
| `prepare_intervenable_inputs` | `causalab.neural.activations.interchange_mode` | locations for arm A |
| `run_interpolation_interventions` | `causalab.neural.activations.interpolate` | arms B / B2 executor |
| `tokenize_variable_values`, `scores_to_joint_probs` | `causalab.methods.metric` | country-token readout, both normalizations |
| `resolve_device` | `causalab.neural.pipeline` | centroid placement |

---

## §4. Config schema

| Knob | Type | Default | Description |
|---|---|---|---|
| `subspace` | `str` | `"pca_k32"` | subspace dir whose raw features + metadata define centroids and site |
| `countries` | `list[str]` | `["Spain", "Russia"]` | target countries whose centroids get patched |
| `arms` | `list[str]` | all four | which arms to run (`interchange_source`, `interpolation_replica`, `interpolation_shapefix`, `no_patch`) |
| `n_eval_samples` | `int` | `64` | counterfactual samples generated for the prompt pool (seed = `cfg.seed + 100`, matching graph_path_steering) |
| `n_prompts` | `int` | `16` | prompts actually patched (shared across all arms) |
| `batch_size` | `int` | `16` | inference batch size |
| `interpolation_self_pairs` | `bool` | `false` | `false` = production behavior (samples keep their real `counterfactual_inputs`); `true` = self-pairs like arm A, to rule out the source-prompt confound on a re-run |

`_subdir: ${.subspace}`. No `visualization:` block — JSON-only outputs.

---

## §5. Outputs

| File | Format | What it shows | Used by |
|---|---|---|---|
| `results.json` | see schema below | per-(country, arm) readouts + shape probe + site | human / `/interpret-experiment` |
| `metadata.json` | resolved-config snapshot | provenance | `/run-experiment` verification |

`results.json` schema:

```json
{
  "site": {"layer": 33, "token_position": "last_token"},
  "value_labels": ["France", "..."],
  "shape_probe": {"f_base_shape": [...], "f_src_shape": [...],
                   "replica_out_shape": [...], "target_shape": [...]},
  "per_country": {
    "Spain": {
      "interchange_source": {
        "p_target_norm": 0.0, "p_target_fullvocab": 0.0,
        "argmax_match_rate": 0.0,
        "mean_norm_dist": {"France": 0.0, "...": 0.0},
        "top5": [["Spain", 0.0]]
      },
      "interpolation_replica": {"...": "same keys"},
      "interpolation_shapefix": {"...": "same keys"},
      "no_patch": {"...": "same keys"}
    }
  }
}
```

Reading the verdict: `interpolation_replica ≈ no_patch` while `interchange_source`
is well above it → executor bug (and `interpolation_shapefix` succeeding pinpoints
the rank mismatch). `interpolation_replica ≈ interchange_source` → the flat
landscapes were a full-vocab readout artifact; re-read existing results
concept-normalized.

---

## Notes

- Arm A mirrors `causalab/methods/interchange/layer_scan.py:434-485` exactly
  (self-pairs, `(B, 1, H)` source tensor, `sources=None`).
- Arm B mirrors `causalab/methods/steer/collect.py:241-273` exactly, including the
  post-run featurizer-device restore loop; the only addition is shape logging.
- Centroid computation mirrors the session's `graph_path_steering/main.py::_compute_centroids`
  (duplicated, ~25 lines, to keep this diagnostic standalone).
- `experiment_root` stays at the global default on RunPod (same as the session's
  graph-steering runner) so the existing `subspace/pca_k32` artifacts are reused.
- Compute footprint: 4 arms × 2 countries × 16 prompts × 1 forward each ≈ a couple of
  minutes on the RunPod GPU, plus model load.
