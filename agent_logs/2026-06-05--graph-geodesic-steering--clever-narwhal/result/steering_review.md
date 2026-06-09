# Review: why manifold steering looks broken on country_borders / Gemma3-4B-PT

Reviewed 2026-06-09 against `demos/country_borders_geometry.ipynb`, the session-local
`graph_path_steering` analysis + `centroid_graph_geodesic` method, and the shipped
`path_steering` / `activation_manifold` / `locate` pipelines. Compared against
`demos/weekdays_geometry.ipynb` (Llama-3.1-8B), where the identical machinery works.

## Diagnosis

### 1. The failure was measurable at fit time, before any path steering

Cell 34's `activation_manifold` log line:

```
Reconstruction test: mean score = 0.0333 across 30 control points
```

`country_borders` sets `intervention_metric: string_match`, which resolves to
**argmax accuracy (higher = better)** via `resolve_intervention_metric`
(`causalab/runner/helpers.py:433`). 0.0333 = 1/30 = **chance**. Steering to a
country's own control point does not make the model output that country.

Footgun: the weekdays demo logged `mean score = -0.0515` from the *same* log line —
but weekdays uses `intervention_metric: kl` (negated KL, closer to 0 = better), so
that number meant *success*. Same log message, opposite semantics. The notebook never
surfaced or asserted on the reconstruction score, so a chance-level result silently
flowed into the path-steering section.

### 2. The steering executor is not globally broken

The weekdays demo runs the exact same chain
(`collect_grid_distributions` → `run_interpolation_interventions` →
`FeatureInterpolateIntervention`) and produces textbook landscapes: geodesic sweeps
Monday→Tuesday→Wednesday→Thursday with P(token) peaks ≈ 0.7. So the failure is
specific to the country_borders/Gemma3-4B configuration, not the steering code path
in general.

### 3. The unresolved contradiction (the thing to nail down first)

At the same site (L33 / last_token), with essentially the same patch value
(per-country raw-activation centroid):

| Path | Mechanism | Result |
|---|---|---|
| `locate` centroid mode (cell 19) | interchange intervention + `source_representations` | **0.41** argmax accuracy (floor 0.03) |
| `path_steering` linear mode (cell 41) | interpolation intervention + `replace_fn`, identity featurizer | flat landscape, ≈ floor |

These should be equivalent operations. Two things differ simultaneously:

- **Executor**: `run_centroid_layer_scan` passes the centroid as a `(B, 1, H)`
  `source_representations` tensor into a `FeatureInterchangeIntervention`;
  `collect_grid_distributions`'s `replace_fn` returns `target.unsqueeze(0).expand(B, -1)`
  — a `(B, H)` tensor — inside a `FeatureInterpolateIntervention`. If pyvene gathers
  base as `(B, 1, H)`, the rank mismatch (and `f_base[:, k_t:]` slicing the *position*
  dim instead of the feature dim) is a plausible silent-corruption point.
- **Readout basis**: locate's 0.41 is computed on **concept-normalized** distributions
  (mass renormalized over the 30 country first-tokens); the landscape plots use
  `full_vocab_softmax=True` on a model that puts ~40% of next-token mass on
  non-country tokens ("other" ≈ 0.4 in every plot). Relative structure among country
  tokens can be invisible in the full-vocab plot.

Experiment 1 below discriminates between these in one short run.

### 4. Calibration: this task/model pair has a low ceiling

- Baseline: strict accuracy 0.68, **prob_accuracy 0.37**; confusion mass concentrated
  on a handful of frequently-correct neighbors.
- Even the *working* centroid patch tops out at 0.41 ≈ prob_accuracy. The per-class
  reference distributions are mushy; weekday-style 0.75-peak landscapes are not an
  attainable target here. Geometry diagnostics (PC0/PC1 R² = 0.89/0.83 vs lat/lon,
  centroid-NN geographic hit rate 0.77 vs 0.12 chance) are genuinely good — the
  *representational* geography is real; the *behavioral* readout is noisy.

### 5. L33 is the final decoder layer — its locate win is partly trivial

Gemma3-4B-PT (text branch) has layers 0–33; L33's output feeds only the final norm +
unembedding. Patching there is close to writing the logits directly, so "centroid
intervention works at L33" carries little localization information, and 0.41 ≈
prob_accuracy is what you'd expect from replaying average class behavior into the
readout. The scientifically interesting sites are mid-stack (locate showed L24=0.15,
L30=0.28 at last_token; the 24–32 range was never scanned densely).

### 6. A misleading code comment worth fixing

`graph_path_steering/main.py` (~line 277) justifies raw-space steering with:
"Steering in the PCA subspace via an inverse-PCA lift was a no-op: the reconstruction
dropped the off-subspace activation mass." That mechanism is wrong:
`SubspaceInverseFeaturizerModule.forward` (`causalab/methods/trained_subspace/subspace.py:39`)
**adds the base error (off-subspace component) back**. If subspace-restricted patching
truly had no behavioral effect, the correct inference is that the pca_k32 span at
L33/last_token does not *causally* carry the answer-country variable — a testable and
important claim (experiment 3), not a serialization artifact.

## Code review notes (session-local code)

`centroid_graph_geodesic.py` — correct and clean: symmetrized distance-weighted kNN,
connectivity-driven k bump, Dijkstra with predecessor walk, duplicate-waypoint
dedup. No issues.

`graph_path_steering/main.py` — good: row-alignment guard, featurizer restore in
`finally`, route/coverage JSON, `_argmax_country_sequence` restricted to country
columns (the right readout for this task). Issues:

1. Per-mode path resolution differs: graph_geodesic gets `steps_per_segment=6` per
   route edge (~31 points for a 6-hop route) vs 25 for geometric/linear. Argmax
   coverage counts aren't directly comparable; resample to equal arc-length steps.
2. The `(num_steps, n_prompts, W)` probs tensor is never persisted — only plots and
   argmax JSON. Save it (like shipped path_steering's pair_distributions) so replots
   and re-analyses don't need a GPU pass.
3. The "no-op patch" comment (see §6).

Notebook: add a cell after the manifold fit that reads `reconstruction_score` from the
artifact metadata and refuses to proceed to path steering below a threshold; log the
metric name next to the score.

## Suggested experiments (dependency order)

1. **Patch-parity A/B** (small script, minutes on RunPod; the critical unblocker).
   Same 16 prompts, raw Spain centroid, L33/last_token, three arms:
   (a) interchange + `source_representations` (locate's mechanism),
   (b) `collect_grid_distributions` with a single grid point + identity featurizer
   (steering's mechanism), (c) no patch. Compare **concept-normalized** mean
   distributions. → b≈c: executor bug — instrument `f_base.shape` inside `replace_fn`
   (the `(B,1,H)` vs `(B,H)` rank question) and fix `collect.py`. → b≈a: the
   landscapes were a readout artifact; re-read everything concept-normalized.
2. **Replot existing landscapes concept-normalized** + plot the argmax sequence per
   step. Near-free; may already reveal Spain→…→Russia structure hidden under "other".
3. **Causal subspace validation**: interchange restricted to the pca_k32 components
   (featurizer-mediated interchange, counterfactual pairs) at L33 and L30; compare
   flip rate vs full-rank interchange. Decides whether *any* manifold steering inside
   pca_k32 can work, independent of path shape.
4. **Per-class steering battery**: steer to all 30 raw centroids through the steering
   executor; per-class concept-normalized accuracy table vs locate's. This is the
   proper "is steering working" gate — much sharper than one pair's landscape.
5. **Move off the final layer**: dense locate scan L24–L33, pick the best
   non-final-layer site, refit subspace + manifold there, rerun 3–4.
6. **Graph-geodesic comparison** (only after 1–5 pass): add a geographic oracle-route
   arm (Spain→France→Germany→Poland→Belarus→Russia), equal arc-length steps across
   modes, and report peak intermediate mass per country in addition to argmax coverage.
7. Optional probes: scale sweep α·centroid, α ∈ {0.5, 1, 2, 4} (centroid norm
   shrinkage); single-example counterfactual activations instead of centroids as a
   patchability upper bound.
