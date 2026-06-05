# Research Objective

## Session

**Session:** `agent_logs/2026-06-05--graph-geodesic-steering--clever-narwhal/`

---

## Objective

Determine whether the country representations in Gemma 3 4B-PT's residual stream form a *navigable* geographic map — i.e. whether steering the model's answer between two distant countries by stepping through a route of intermediate country representations drives the prediction continuously across geographically-intermediate countries, more so than a straight-line interpolation does.

---

## Motivation

The country_borders geometry work found that the answer-country centroids at the late-layer last-token site lie on an approximately 2-D, map-like surface (PCA structure regresses onto real lat/lon). A natural follow-up question is whether that map is not just *shaped* like Europe but *navigable* like Europe: can the model's prediction be walked from one country to a far one by passing through the in-between countries? The current straight-line steering between two endpoints under-tests this, because for far pairs (Spain → Russia) a straight segment in coordinate space skips most intermediate-country representations. Routing the steering path through the actual intermediate country representations is a stronger, more direct test of map-likeness, and bears on whether the residual-stream geometry supports compositional/relational navigation rather than merely encoding position.

---

## Scope boundaries

- **Out-of-scope model:** only Gemma 3 4B-PT this session; no cross-model comparison.
- **Out-of-scope site:** the late-layer last-token site already localized in the country_borders work is taken as given; no re-localization sweep.
- **Out-of-scope geometry:** no differential-geometry surface geodesic (geodesic ODE on the fitted spline). "Geodesic" here means a shortest path over a graph of country representations.
- **Out-of-scope pairs (primary):** evaluation centers on Spain → Russia; other pairs are a stretch goal only.
- **Out-of-scope:** changing the upstream subspace/manifold fit; we reuse the existing PCA-32 subspace and fitted manifold.

---

## Success criteria *(recommended)*

- The new graph-geodesic path mode runs end-to-end on country_borders (Gemma 3 4B-PT) and produces an output landscape for Spain → Russia alongside `geometric` and `linear`.
- The graph route from Spain to Russia is geographically sensible (an ordered chain of bordering/near-bordering countries), not a single jump.
- Along the graph-geodesic path, prediction mass passes through ≥ 2 distinct intermediate countries that lie on the route (peak output probability shifts to an intermediate country at some interior step), whereas the `linear` path does not.
- Qualitatively, the graph-geodesic landscape shows a more monotone/contiguous hand-off of probability between adjacent route countries than `geometric` or `linear`.

---

## Hypotheses *(recommended)*

- **H1.** A k-NN + Dijkstra route over PCA centroids from Spain to Russia recovers a chain of geographically-adjacent countries. *Falsified if* the shortest path jumps between non-adjacent countries or is dominated by a single long edge.
- **H2.** Steering along that route hands prediction mass off through intermediate countries (the argmax output sweeps across ≥ 2 route countries at interior steps). *Falsified if* the output stays pinned to Spain then jumps to Russia with no intermediate country ever becoming the argmax.
- **H3.** The graph-geodesic sweeps through more intermediate countries than `geometric` (straight line in lat/lon) and `linear` (straight line in PCA). *Falsified if* `geometric`/`linear` match or exceed the graph-geodesic in intermediate-country coverage.
