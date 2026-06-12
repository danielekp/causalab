# Session report — graph-geodesic steering on country_borders (Gemma3-4B-PT)

Session: `2026-06-05--graph-geodesic-steering--clever-narwhal` · concluded 2026-06-12
Detailed run-by-run evidence: `patch_parity_verdict.md`, `steering_review.md` (addendum).

## Objective

Test whether the model's country representation supports *geometric* steering: does
walking a path in activation space between two distant countries' representations pass
through intermediate countries (H2), and can that path be found from activation data
alone via a centroid k-NN graph (H3)? The session opened with a blocker: all steering
landscapes sat at the 1/30 chance floor ("manifold steering not working").

## What was wrong (and what wasn't)

1. **The steering executor was never broken.** A four-arm A/B (`patch_parity`) showed
   `collect_grid_distributions`'s replace_fn is bit-identical to locate's interchange
   mechanism; the suspected (B,1,H)/(B,H) rank mismatch does not exist.
2. **Every steering readout scored the wrong token.** In this task `country` is the
   question's *subject* ("Which country lies to the {direction} of {country}?"); the
   correct answer is `NEIGHBOR_OF[(country, direction)]`. p(steered-country token) is
   never the right answer → chance floor by construction. Locate scored 0.41 because
   string_match checks the neighbor — the correct readout all along.
3. **L33/last_token (the original site) is trivial for steering.** Final layer ⇒ a
   full-residual patch freezes the output to the centroid's direction-marginal
   (Russia argmax_expected capped at 0.083).

## The validated setup

- **Site: L12 / `country` token position** (`subspace/pca_k32_ctok`), found via a
  dense locate scan at the country position (L0 = 0.29 is token-embedding-trivial;
  L12 = 0.22 is the mid-depth bump). patch_parity there: Russia argmax_expected
  **0.417** vs 0.042 floor — the model re-integrates the prompt's direction with the
  patched subject (~20 layers of downstream computation).
- **Readout: subject signature decode.** Each step's answer distribution is decoded to
  the subject S maximizing mass on `NEIGHBOR_OF[(S, prompt_direction)]` ∪ {S}
  (echo-augmented; the ~0.13 subject-echo mass otherwise leaks to the true subject's
  neighbors and shifts the decode one country over).

## Findings (runs 6–7, four endpoint pairs × four path modes)

**F1 — Manifold geodesics traverse geography (H2 supported).** The `geometric` mode
(spline geodesic in intrinsic lat/lon coordinates, *no waypoints supplied*) decodes
contiguous, geographically ordered subject chains on every pair:

| pair | geometric echo-decode |
|---|---|
| Spain→Russia | Spain France Italy Austria Poland Ukraine Russia |
| Portugal→Finland | Spain France Italy/France Germany Poland Latvia Russia Finland |
| Greece→Norway | Greece Bulgaria Serbia Hungary Poland Germany Denmark Sweden |
| Portugal→Greece | Spain France Italy Greece (the *sea* geodesic, not the land detour) |

Intermediate points on the lat/lon-parameterized manifold decode to **working
representations** of geographically intermediate countries — deep enough that the
model computes direction-conditioned answers from them.

**F2 — Raw-space traversal works when the route is supplied (oracle).** Piecewise
paths through true-border raw centroids recover essentially every control point in
order, including the 8-edge Portugal→Greece route. Steering + decode fidelity is high.

**F3 — The straight line is a clean negative control.** `linear` always jumps the
middle in one hop (Spain→Ukraine; Spain→Russia; Bulgaria→Denmark; Spain→Greece): the
chord leaves the manifold and crosses empty activation space.

**F4 — Geographic adjacency is NOT recoverable from activation distances (H3
falsified at the premise).** The k=4 PCA-centroid k-NN graph has border-edge
precision 0.27 / recall 0.38; its Dijkstra routes are non-geographic (Greece→Norway
via *Iberia*). So: activation space *supports* geographic traversal, but the geography
must be supplied externally (lat/lon chart or oracle route) — it is not encoded in
raw centroid proximity at this site/representation.

**F5 — Decode quirks (localized).** Norway never decodes as itself (Sweden/Finland win
even at Norway's own centroid — weak centroid or signature domination); mutual-neighbor
endpoints flicker (Portugal↔Spain). Neither affects F1–F4.

**F6 — Shuffle control passes (run 8): F1 is not a spline artifact.** Refitting the
manifold with a permuted country↔lat/lon assignment (`embedding_shuffle_seed=0`,
recon_mse 0.54 vs the canonical fit) and re-steering the geometric mode yields decoded
chains with similar intermediate *counts* but destroyed geographic *order*:
border-contiguity of consecutive decoded subjects drops from **0.78 (18/23 transitions
canonical)** to **0.38 (8/21 shuffled)**, with absurd jumps (Greece–Denmark,
Spain–Poland). The canonical breaks are benign (Italy–Greece = the Mediterranean
crossing). Residual shuffled contiguity ≈ 0.38 matches the k-NN border recall — i.e.,
what remains is the representation's own neighbor structure, not the chart. The
geographic ordering in F1 therefore comes from the alignment between the lat/lon chart
and the activation geometry, not from TPS interpolation mechanics.

## Caveats

- The geometric mode's chart is lat/lon-parameterized (`intrinsic_mode: parameter`):
  geography is an *input* to the spline fit. F1 says the activation manifold is
  compatible with that chart (smooth, decodable between countries) — not that the
  geometry was discovered unsupervised (F4 says the opposite for raw distances).
- Single model (Gemma3-4B-PT), single site, 16 prompts/step, single seed.
- TPS at ambient_dim=32 warns about numerical stability.
- The fit-time "reconstruction test" metric still uses the old readout; ignore it.

## Highest-value next steps

1. ~~Shuffle control~~ — **done (run 8, F6): passed.**
2. Per-class battery: patch all 30 centroids through patch_parity to map weak classes
   (Norway) and quantify mean argmax_expected across the full set.
3. Quantify linear geography: Procrustes / correlation between 2-D PCA of L12
   centroids and capital lat/lon (how much geography is linearly present vs only
   spline-recoverable).
4. Sweep graph_k / try graph over PCA-2 or spline-intrinsic coordinates to see if any
   data-derived graph recovers borders.
5. More shuffle seeds (the contiguity gap 0.78 vs 0.38 is from one permutation; 3–5
   seeds would give a null distribution for the contiguity statistic).
