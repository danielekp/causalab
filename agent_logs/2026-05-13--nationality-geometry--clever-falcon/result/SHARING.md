# Sharing package — country_borders geometry

Draft artifacts for publicizing the result and getting Goodfire's attention.

**Status: gate CLEARED.** Phase-1 defensibility done (#3/#5); #2 retrieval-vs-computation ran (REPORT §6.4) *and* the layer-sweep control resolved its confound (REPORT §6.5): the relational-answer map is strong at **every** layer (R² 1.13–1.67) while the direct entity-token readout is weak at **every** layer (R² 0.36–0.59, at the null ceiling). The differentiating claim is now supported with a precise, defensible wording (below). Remaining work is packaging + dissemination, not more defensibility.

**Wording discipline (carry into the post):** the claim is *mechanistic and task-scoped* — "for relational geographic queries, Llama-3.1-8B's coherent Europe map is a property of the computed-answer representation, not a linear readout of the entity token at any layer." It is **not** "Llama has no stored world-map" — Gurnee & Tegmark show it does under their setup. We add the complementary mechanism, we don't overturn them.

---

## A. LessWrong / Alignment Forum post — draft

**Working title (now earned; pick one):**
- *"Llama-3.1-8B builds its Europe map at the answer, not the entity: a relational-vs-stored contrast"*
- *"The map is in the computation: Llama-3.1-8B's Europe geometry lives at the relational answer, not the country token"*

> Safe to use "computed / constructed by the relational computation". Still avoid "traverses" unless you mean it literally — we showed *where* the map is (computed-answer representation), not a step-by-step traversal path.

### TL;DR
From a directional-border task ("Which country lies to the east of Spain?"), one centroid per *answer* country: the first two PCs of Llama-3.1-8B's answer-position representation are an affine image of European lat/lon (PC0≈lat R²0.77, PC1≈lon R²0.83), permutation p=0.0005, bootstrap CI [1.24,1.80]; East/West and language collapse to chance once geography is partialled out. The new part: a controlled within-task contrast + an 8-layer sweep shows this clean map is **absent from the country's own entity-token representation at every layer** (R² 0.36–0.59) and present in the relational-answer representation at every layer (R² 1.13–1.67). The map the model uses to answer relational geographic queries is produced by the computation, not read off the entity.

### 1. Background & relation to prior work
Gurnee & Tegmark, *Language Models Represent Space and Time* (arXiv 2310.02207, ICLR 2024): linear probes on **entity-token activations** recover a world map in Llama-2 — a *stored* entity→coordinate readout. Causal follow-up: *More than Correlation* (arXiv 2312.16257). "LLMs store a geographic map" is established. **Our contribution:** a *mechanistic* result — in a relational task, the clean map is a property of the **computed-answer** representation, and the entity-token readout (same prompts, same model, every layer 4–31) does *not* linearly carry it. Complementary to G&T, not contradictory; methodology is Goodfire's open-source causal-abstraction framework `causalab`.

### 2. Setup
- Task `country_borders`: 30 European countries × 8 directions × 4 templates; 584 valid prompts. Llama-3.1-8B (base); baseline strict accuracy 65% on valid prompts.
- Relational centroid: mean answer-position activation over all (entity, direction) prompts whose primary neighbor is X → entity-disentangled. Direct centroid: mean country-token activation over prompts that name X. PCA-32, identical featurisation both conditions.

### 3. Result
- **Geometry:** PC0 R²=0.77, PC1 R²=0.83, PC2 0.07; rotation −11.7°; combined R²=1.604/2.0. (fig `pca_vs_geography.png`)
- **Significant:** permutation p=0.0005 (null max 0.55), bootstrap CI [1.24,1.80]. (fig `geometry_null_hist.png`)
- **Specific:** geography-residual probe — East/West +0.30→+0.00, language +0.14→−0.04; EU not decodable (−0.10).
- **Computed, not stored (the new bit):** relational vs direct, swept over layers {4,8,12,16,20,24,28,31}. Relational R² 1.13–1.67 at all layers; direct 0.36–0.59 at all layers (Procrustes disparity 0.89 at L28 — different geometries). No layer rescues the entity-token readout. (figs `retrieval_vs_computation.png`, `layer_sweep.png`)

### 4. Limitations (state plainly)
Single model, single task; capital coords proxy country position; n=30 (mitigated by permutation null + bootstrap CI). The `direct` condition is the entity token *within border prompts*, not a G&T-style diverse-prompt probe — so this bounds the mechanism *for this task*, it does not claim the model lacks a stored world-map in general. Sweep used n_perm=500 (p floor 0.002); canonical L28 cell confirmed at 2000 (p=0.0005).

### 5. Reproduction
`result/country_borders_reproduce.ipynb` (self-contained); `code/analyses/{geometry_robustness,retrieval_vs_computation,layer_sweep}.py`; full report `result/REPORT.md`. All in `causalab`.

---

## B. causalab PR scope

Target: `goodfire-ai/causalab`. Tight and reviewable — this PR *is* the high-signal contact with the maintainers (Atticus Geiger / Goodfire).

**Include:**
1. `causalab/tasks/country_borders/` + `causalab/configs/task/country_borders.yaml` (incl. the `colormap` fix).
2. A shipped analysis `causalab/analyses/geometry_isomorphism/` promoting the three `code/analyses/*.py` (robustness, retrieval-vs-computation, layer-sweep) into one module with a `README.md` (Saved-artifacts table, Research-question section) per repo conventions.
3. `demos/country_borders_geometry.ipynb` — the reproduction notebook, repathed off the session tree.

**Exclude:** session-local `agent_logs/` content, the messy working notebook, raw artifacts.

**PR description:** lead with the computed-vs-stored mechanistic result and the causal-abstraction angle; cite Gurnee & Tegmark for positioning; link the LW post. One paragraph, no overclaiming, keep the §-level wording discipline.

---

## C. Outreach checklist

1. [x] #2 retrieval-vs-computation + layer-sweep control — **done** (REPORT §6.4/§6.5).
2. [x] REPORT.md finalized with the resolved verdict.
3. [ ] Polish the LW/AF post from §A; embed the four figures; keep the wording discipline.
4. [ ] Open the causalab PR (§B) — clean, small, well-described.
5. [ ] Publish the LW/AF post; link the PR and the reproduction notebook.
6. [ ] Short X/Twitter thread; tag Goodfire, Atticus Geiger, Wes Gurnee.
7. [ ] Warm outreach: reference the post + PR in a Goodfire Fellowship application (causal-analysis track) or a short, specific email — "built this in your framework, here's the post + PR."

**Optional strengthener (not a gate):** causal intervention along the longitude PC (steer the model's directional answer). Converts the correlational geometry into a causal claim — squarely the causalab/Geiger wheelhouse. Adds punch to the post but the result stands and is publishable without it.
