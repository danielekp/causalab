# patch_parity run 1 — verdict (2026-06-12)

Run: `country_borders_patch_parity` on a fresh RunPod (subspace artifacts regenerated
via `country_borders_subspace_demo`; PCA fit is deterministic, so the pod swap does not
affect the verdict). Site: L33 / last_token. Countries: Spain, Russia. 16 prompts.

## Verdict 1 — the steering executor is exactly correct

`interchange_source`, `interpolation_replica`, and `interpolation_shapefix` produced
**bit-identical** distributions for both countries (every value matches to the last
decimal). `shape_probe`:

```
f_base_shape: [16, 2560]   target_shape: [2560]
replica_out_shape: [16, 2560]   shapefix_out_shape: [16, 2560]
```

pyvene gathers `(B, H)` at this site — there never was a `(B,1,H)`-vs-`(B,H)` rank
mismatch, so `collect_grid_distributions`'s `replace_fn` needs no shape fix. The
executor-bug hypothesis (steering_review.md experiment 1, branch (a)) is **dead**, and
the `interpolation_shapefix` arm is moot. Hypotheses retired with it:

- commit cfb529c's rationale ("inverse-PCA lift was a no-op") — already disproved
  mechanistically, now disproved empirically too;
- the full-vocab-softmax readout artifact as the *primary* explanation (it compresses
  the scale ~2.5×, but the flat landscapes have a deeper cause, below).

## Verdict 2 — the patch works; every steering readout scored the wrong token

The patched-country distributions are geographically structured:

| patch | top of distribution | reading |
|---|---|---|
| Spain | Portugal 0.220, France 0.162 | Spain's only in-set neighbors |
| Russia | Poland 0.129, Ukraine 0.111, Russia 0.110, Belarus 0.079 | Russia's border set (+ self-echo) |
| no_patch | Poland 0.103, France 0.101, Russia 0.076 | mixed marginal |

This matches the task's causal model exactly (`causal_models.py`): `country` is the
**subject** of the question ("Which country lies to the {direction} of {country}?")
and `raw_output = NEIGHBOR_OF[(country, direction)]`. Patching the centroid sets the
subject variable, and the model **correctly recomputes the neighbor answer
downstream** — which is precisely what a successful interchange on `country` should
do. Aggregate neighbor-of-target mass (direction-agnostic, from `mean_norm_dist`):

- Spain: 0.128 (no_patch) → **0.382** (patched)
- Russia: 0.338 (no_patch) → **0.511** (patched)

Meanwhile `p(patched-country token)` — the readout used by every steering landscape,
the manifold reconstruction test, and `graph_path_steering`'s scoring — is a token
that is *never* the correct answer under the causal model. It sits at ≈ chance by
construction (Spain 0.032 ≈ 1/30), which is exactly the "flat landscape" symptom.

This also resolves the central contradiction of the review: locate's
`run_centroid_layer_scan` scores string_match against the counterfactual's
`raw_output` (the neighbor) — the **correct** readout — hence 0.41; the steering
stack scores p(target country itself) — the **wrong** readout — hence floor. Same
mechanism, same effect, two readouts.

## Consequences for the experiment queue (steering_review.md)

- **Experiment 2 is redefined**: not "concept-normalized replots" but **rescore with
  the correct readout** — per prompt, expected-answer mass
  `Σ p(NEIGHBOR_OF[(steered_country, prompt_direction)])`. `patch_parity` now emits
  this (`p_expected_norm`, `argmax_expected_rate`, `p_canonical_norm`); a rerun
  quantifies the per-direction effect cleanly. The same mapping must be applied to
  `graph_path_steering` / landscape scoring before any geometry conclusion is drawn.
- The manifold reconstruction test's 0.0333 needs re-reading under the same lens
  before concluding the spline fit failed.
- Experiments 3–7 (causal subspace validation, per-class battery, layer scan,
  geodesic comparison) remain, but all scoring must go through the expected-answer
  mapping.
- Caveat to carry: L33 is the final layer, so part of the effect may still be
  "writing the answer's precursor directly"; the layer scan (experiment 5) still
  matters.

## Rerun

```bash
export CAUSALAB_SESSION_CODE=/workspace/causalab/agent_logs/2026-06-05--graph-geodesic-steering--clever-narwhal
bash scripts/run_exp.sh country_borders_patch_parity \
  'patch_parity.arms=[interchange_source,no_patch]' \
  patch_parity.n_prompts=64 patch_parity.n_eval_samples=64
```

(arms reduced — parity is settled; more prompts for tighter per-direction stats.)

---

## Addendum: L12/country site validation (2026-06-12, run 3)

Dense locate scan at the `country` token position: L0 peaks (0.29, ≈ token-embedding
substitution — trivial), decays upward, with a mid-depth bump at L12 (0.22). Subspace
refit at L12/country → `subspace/pca_k32_ctok`; patch_parity rerun there:

| metric | L33/last_token | **L12/country** | no_patch |
|---|---|---|---|
| Spain p_expected_norm | 0.182 | **0.214** | 0.088 |
| Spain argmax_expected | 0.219 | **0.281** | 0.156 |
| Russia p_expected_norm | 0.129 | **0.301** | 0.035 |
| Russia argmax_expected | 0.083 | **0.417** | 0.042 |

Russia is decisive: its 8 neighbors spread across directions, so the L33 frozen
marginal (final layer ⇒ patch fully determines logits, direction unrecoverable) capped
argmax_expected at 0.083; L12/country reaches 0.417 — the model re-integrates the
prompt's direction with the patched subject downstream. p_target_norm also rises to
~0.13 (subject-echo). **L12/country (`pca_k32_ctok`) is the site for the geometry
pipeline.** All downstream scoring must use the subject/neighbor-signature readout
(now implemented in graph_path_steering's `subject_decode_sequence`).

---

## Addendum: first corrected-readout steering run at L12/country (2026-06-12, run 4)

`graph_path_steering` at `pca_k32_ctok/L12_country/spline_s0.0`, Spain→Russia:

| mode | decoded subject route | intermediates |
|---|---|---|
| **geometric** (manifold geodesic) | Spain → Italy → Slovakia → Ukraine → Belarus → Russia | **4** |
| graph_geodesic (k-NN route) | Spain → Italy → Russia | 1 |
| linear | Spain → Italy → Russia | 1 |

- **H2 supported**: the manifold geodesic hands the decoded subject off through a
  geographically ordered chain at a site where the model still computes (the old
  readout would have scored every step ≈ chance).
- Decode fidelity confirmed: graph_geodesic's decoded subjects exactly reproduce its
  route control points (route.json: Spain→Italy→Russia, k_used=4).
- **H3 premise in doubt**: the PCA k-NN graph contains a non-border Italy–Russia edge;
  the "route" is activation-proximity, not geography. Italy decoding at the midpoint of
  *both* raw-space modes suggests Italy's centroid sits near the global center /
  Spain–Russia midpoint in activation space.
- Follow-ups implemented (run 5): `oracle` path mode (hand-specified true-border route
  Spain→France→Germany→Poland→Belarus→Russia, same raw-space mechanism) and
  `graph_quality.json` (k-NN edges vs true border graph, precision/recall + explicit
  false-positive/false-negative edge lists).

---

## Addendum: oracle route + graph quality (2026-06-12, run 5)

| mode | decoded subjects | intermediates |
|---|---|---|
| geometric (manifold geodesic) | Spain → Italy → Slovakia → Ukraine → Belarus → Russia | 4 |
| oracle (true-border route, raw space) | Spain → Czech Republic → Ukraine → Belarus → Russia | 3 |
| graph_geodesic (k-NN route) | Spain → Italy → Russia | 1 |
| linear | Spain → Italy → Russia | 1 |

`graph_quality.json`: k=4 PCA-graph edge precision **0.27**, recall **0.38**, f1 0.32.

- **H3 split verdict**: the *mechanism* works (oracle route walks a contiguous
  west-to-east chain through raw centroids) but the *data-driven route* fails — the
  k-NN graph over PCA centroids does not encode borders (~3 of 4 edges are
  non-borders, e.g. Italy–Russia). Geographic adjacency is NOT recoverable from raw
  activation distances at this site, even though geographic *traversal* in activation
  space works when the route is supplied externally (oracle list, or the manifold's
  lat/lon parameterization).
- **Decoder bias identified**: oracle's control points France/Germany/Poland decoded
  as Czech Republic/Ukraine — the subject-echo mass (~0.13 on the subject's own
  token) is harvested by *neighbors* of the true subject under the neighbor-only
  signature. Fixed by an echo-augmented decode (signature additionally claims the
  subject's own token), emitted side-by-side as `subject_decode_sequence_echo`
  (run 6).
