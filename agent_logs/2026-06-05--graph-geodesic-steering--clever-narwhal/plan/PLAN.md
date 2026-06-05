# PLAN — graph-geodesic-steering

Companion to `RESEARCH_OBJECTIVE.md`. Read by `/run-experiment` and `/interpret-experiment`.

This plan is **implementation-heavy**: all upstream analyses already exist and were run in the
`country_borders` geometry work (baseline → locate → subspace → activation_manifold). The only new
work is a **graph-geodesic path mode** for steering. The two new nodes are `custom`.

---

## §B. Causal model & dataset

**Task:** `country_borders` — package at `causalab/tasks/country_borders/`
- **Status:** `exists`

### Causal variables

| Name | Type | Cardinality | Sketch of value space |
|---|---|---|---|
| `country` | categorical | 30 | European country names (`Spain`, `France`, …); also the target/answer variable |
| `direction` | categorical | 8 | cardinal/intercardinal (`N, NE, E, …`) |
| (template) | categorical | 4 | paraphrase templates |

### Mechanism summary

- `answer = NEIGHBOR_OF[(country, direction)][0]` — the primary neighboring country in the given direction.

### Expected behavior

```
Input:  "Which country borders Spain to the east?"   Output: "France"
Input:  "... France to the north?"                    Output: "Belgium"
```

(The country_borders baseline already confirmed the model solves this; not re-run unless artifacts are missing.)

### Counterfactual generator

| Variable | `task.resample_variable` | Why |
|---|---|---|
| all | `"all"` | We operate on per-answer-country **centroids**, not pairwise CFs — matches the centroid approach used to fit the manifold. |

### Dataset sizing

```yaml
task:
  enumerate_all: true   # 30 countries × 8 directions × 4 templates; test = train
  target_variable: country
```

Rationale: reuse the exact enumeration the subspace/manifold fit was built on so centroids are row-comparable.

---

## §C. Neural surface

### Model(s)

| Model | Config | Why |
|---|---|---|
| Gemma 3 4B-PT | `model: gemma3_4b_pt` | the model the country_borders geometry result was established on; pre-trained (non-instruct), completes plain `Q:/A:` templates |

### Site

- Late-layer last-token site already localized in the country_borders work. Subspace = **PCA-32 at layer 33, `last_token`**; manifold = thin-plate spline `L33_last_token/spline_s0.0`, `intrinsic_dim=2`. Reused as-is (no re-localization).

### Tokenization-check predictions

- Country names already validated as steerable answer tokens in the prior baseline/locate runs; no new check needed.

### Compute budget

| Phase | Where | Expected wall time | GPUs |
|---|---|---|---|
| reuse upstream (baseline/locate/subspace/activation_manifold) | cached artifacts | 0 (already produced) | — |
| graph route computation (CPU, 30 centroids) | inline | < 1 s | 0 |
| graph-geodesic steering (Spain→Russia) + geometric/linear baselines | runner | ~3–10 min | 1 |

### Hardware constraints

- Inherits whatever environment produced the country_borders runs (Gemma 3 4B-PT requires a GPU for the forward passes; `scripts/run_exp.sh` is the launcher). The new method itself is CPU-only.

---

## §D. Analysis-chain DAG

### DAG diagram

```
[cached] baseline ─► locate ─► subspace(pca_k32) ─► activation_manifold(spline,2D)
                                                          │
                                                          ▼
                                       graph_path_steering  (NEW analysis, custom)
                                                          ▲
                                                          │ uses
                                       centroid_graph_geodesic (NEW method, custom)
```

### Per-node detail

#### Node 1: `centroid_graph_geodesic` — `custom (method, to be implemented)`

- **Research question (scoped):** Given the country centroids, what ordered route of intermediate countries connects two endpoints along the data manifold?
- **What it is:** a pure primitive. Input: centroid matrix `(n, d)` in PCA space, `labels: list[str]`, `start`, `end`, `k` (neighbors), optional metric. Output: `(route_labels: list[str], path_points: Tensor)` where `path_points` concatenates straight-line segments between consecutive route centroids (with a per-segment step budget).
- **Algorithm:** k-NN graph (edge weight = Euclidean distance in PCA space) → Dijkstra `start`→`end` → ordered centroid waypoints → linear interpolation per segment. Connectivity guard: if the k-NN graph is disconnected between endpoints, raise / fall back to increasing `k`.
- **Upstream artifacts consumed:** none directly (operates on in-memory centroids passed by the analysis). Pure, no I/O (ARCHITECTURE §3 Inv 4).
- **Downstream artifacts produced:** none (returns in-memory result).
- **Non-default knobs:** `k` (k-NN), `steps_per_segment` (or total step budget), `metric`.
- **Pre-flight check:** on the real Spain→Russia centroids the route must be a connected chain of > 2 distinct countries with no single edge dominating total path length (sanity for H1).
- **Spec path:** `${SESSION_DIR}/plan/setup_method_centroid_graph_geodesic.md`
- **Estimated runtime + GPU footprint:** < 1 s on CPU.

#### Node 2: `graph_path_steering` — `custom (analysis, to be implemented)`

- **Research question (scoped):** Does steering the model along the graph-geodesic route hand prediction mass off through intermediate countries, more than straight-line steering?
- **Why a NEW analysis (not an edit to `path_steering`):** session-local code **must not shadow** a shipped analysis (`setup-analyses` SKILL.md:89 — dispatcher prefers shipped over session). So during the research session we prototype under a distinct name `graph_path_steering`. It **imports and reuses** `causalab.analyses.path_steering` machinery (artifact loading, featurizer/manifold load, steering execution, `path_visualization`) and adds the graph-geodesic path. Promotion (folding the mode back into the shipped `path_steering` as the `build_path` signature/call-site change the user described) is a manual post-stabilization step — see §E.
- **Method:** reuses `centroid_graph_geodesic` (Node 1) for the route; reuses `path_steering`'s `geometric`/`linear` `PathMode`s for the two baselines.
- **Upstream artifacts consumed:** `subspace/pca_k32/country/...` (centroids + PCA), `activation_manifold/.../L33_last_token/spline_s0.0/...` (featurizer/manifold). Auto-discovery: `subspace: null` / `activation_manifold: null` resolve most-recent dirs (ANALYSIS_GUIDE Auto-Discovery), but we pin `pca_k32` and `L33_last_token/spline_s0.0` explicitly to match the prior fit.
- **Downstream artifacts produced (under `${experiment_root}/graph_path_steering/...`):** `vis/paths/graph_geodesic/pair_Spain_Russia.png`, `vis/paths/geometric/pair_Spain_Russia.png`, `vis/paths/linear/pair_Spain_Russia.png`, a `route.json` (ordered country chain for Spain→Russia), and an `intermediate_coverage.json` summarizing how many distinct route countries become argmax along each path (the H2/H3 metric).
- **Non-default knobs:** `selected_pairs: [["Spain","Russia"]]`; `path_modes: [graph_geodesic, geometric, linear]`; `subspace: pca_k32`; `activation_manifold: L33_last_token/spline_s0.0`; graph `k`.
- **Pre-flight check:** the analysis loads the same 30 centroids the manifold was fit on (`len(centroids) == len(names)` from the country_borders subspace) before steering. If the featurizer has no manifold stage, stop.
- **Spec path:** `${SESSION_DIR}/plan/setup_analysis_graph_path_steering.md`
- **Estimated runtime + GPU footprint:** ~3–10 min on 1 GPU (model forward passes for ~15–25 steps × 3 paths).

### Cross-analysis post-steps

- None. (Single steering pair; comparison rendered within `graph_path_steering`.)

---

## §E. Risk register & contingency

### Pitfalls active for this plan

- **No-shadow rule** — cannot override shipped `path_steering` from the session. Mitigated by the distinct-name `graph_path_steering` prototype (§D Node 2).
- **Centroid row-alignment** — `raw_features.safetensors` must align with `train_dataset.json` rows (same hazard flagged in the country_borders notebook review). The analysis asserts `len(features) == len(rows)` before grouping.
- **k-NN disconnection** — too-small `k` can disconnect Spain from Russia. Contingency: auto-increase `k` until connected, log the final `k`.
- **Graph space choice** — graph + weights are in **PCA space** (not intrinsic lat/lon) by design, to avoid circularly assuming the geography we're testing for (locked decision).
- **Reused-fit drift** — if the session's `experiment_root` doesn't already contain the subspace/manifold artifacts, they must be regenerated or symlinked/copied from the country_borders run. See contingency below.

### Per-step contingency

| Node | If pre-flight fails, then |
|---|---|
| upstream artifacts | subspace/manifold not present under session `experiment_root` → re-run the country_borders subspace + activation_manifold steps into this session (cheap, deterministic) before steering. |
| `centroid_graph_geodesic` | route is a single jump / disconnected → increase `k`; if still degenerate, switch to ε-ball graph. Report and stop (H1 falsified). |
| `graph_path_steering` | featurizer lacks manifold stage / centroid count mismatch → stop; revisit subspace/manifold reuse. |

---

## §F. Outputs of the plan itself

### Runner config(s)

Single config (no sweep):

- `${SESSION_DIR}/code/configs/runners/demos/country_borders_graph_steering_demo.yaml`

### Sweep & cache strategy

Not applicable — single runner config, single steering pair, single site.

### Expected artifact tree

```
${SESSION_DIR}/artifacts/country_borders/gemma3_4b_pt/
├── subspace/pca_k32/country/...                 (reused / regenerated)
├── activation_manifold/.../L33_last_token/spline_s0.0/...  (reused / regenerated)
└── graph_path_steering/
    ├── route.json                               # Spain→Russia ordered country chain
    ├── intermediate_coverage.json               # distinct argmax route-countries per path mode
    └── vis/paths/
        ├── graph_geodesic/pair_Spain_Russia.png
        ├── geometric/pair_Spain_Russia.png
        └── linear/pair_Spain_Russia.png
```

### Hand-off

1. After plan approval: `/setup-methods` (Node 1) then `/setup-analyses` (Node 2) scaffold the two custom pieces under `${SESSION_DIR}/code/`.
2. `/run-experiment` materializes `country_borders_graph_steering_demo.yaml` and executes.
3. `/interpret-experiment` reads artifacts + this plan.

---

## Review checkpoint

Surfaced to user:

1. **Hypotheses + success criteria** — agreed (objective approved).
2. **Implementation strategy fork** — RESOLVED: session-local prototype (`graph_path_steering` analysis + `centroid_graph_geodesic` method); promote into shipped `path_steering` later.
3. **Compute** — ~3–10 min on 1 GPU; trivial.

(Approval logged to `approval.log`.)
