# Sharing package — country_borders geometry

Draft artifacts for publicizing the result and getting Goodfire's attention.
Status: defensibility done (Phase-1 #3/#5). #2 retrieval-vs-computation **has run** (REPORT §6.4) with a *surprising* result: at layer 28 the map is strong at the relational answer position but weak/different at the direct entity-token position (combined R² 1.60 vs 0.55; Procrustes disparity 0.89). This is *suggestive* of "computed not stored" but is confounded by the fixed layer — the direct/stored condition was not given its best layer. **The publishable headline is now gated on the layer-sweep control (REPORT §10), not merely on #2 having run.** Do not publish the "traversed vs. stored" framing until the layer sweep resolves the confound.

---

## A. LessWrong / Alignment Forum post — draft

**Working title (pick one, honest to what's proven):**
- *"A statistically-defended map of Europe inside Llama-3.1-8B, recovered from a relational border task"*
- *"Llama-3.1-8B answers 'which country is east of X?' from a coherent internal lat/lon map"*

> Avoid "the model *traverses* (vs. stores) a map" in the title — that's the retrieval-vs-computation claim, not yet measured (#2).

### TL;DR
Building one centroid per *answer* country from a directional-border task ("Which country lies to the east of Spain?"), the first two PCs of Llama-3.1-8B's answer-position representation are an affine image of European latitude/longitude (PC0≈lat R²0.77, PC1≈lon R²0.83), recovered with essentially no rotation. A permutation null and bootstrap CI show this is far from chance, and a geography-residual probe shows the apparent Cold-War and linguistic structure is *entirely* a geographic shadow. The map is *specifically* geographic and is recovered from a purely *relational* task.

### 1. Background & relation to prior work
Gurnee & Tegmark, *Language Models Represent Space and Time* (arXiv 2310.02207, ICLR 2024), showed linear probes on **entity-token activations** recover a world map in Llama-2 — direct entity→coordinate readout. The causal follow-up *More than Correlation* (arXiv 2312.16257) asks whether that representation is causal. **State the contribution honestly:** "LLMs store a geographic map" is established. This post adds (a) the map is recovered at the *answer* position of a **relational** task (the country name never appears in the prompt for its own centroid), and (b) a causal-abstraction methodology (Goodfire's open-source `causalab`). The decisive relational-vs-retrieval contrast (#2) is in progress and explicitly marked open.

### 2. Setup
- Task: `country_borders` — 30 European countries × 8 directions × 4 paraphrase templates; 584 geographically valid prompts. Model: Llama-3.1-8B (base). Baseline strict accuracy 65% on valid prompts.
- Per-answer-country centroid: mean answer-position activation over all (entity, direction) prompts whose primary neighbor is country X → entity-disentangled. 30 centroids in PCA-32 space, layer 28.

### 3. Result
- PC0 R²=0.77 (β_lat −0.30, β_lon ≈0.05); PC1 R²=0.83 (β_lon −0.18, β_lat ≈0.02); PC2 R²=0.07. Best in-plane rotation −11.7°; combined R²=1.604/2.0. Clean, near-orthogonal lat/lon factorization — stronger than the R² magnitude alone. *(figure: pca_vs_geography.png — note it's mirror-flipped; PCA orientation is arbitrary, R² sign-invariant.)*
- **Significance:** 2000-permutation null over country↔coordinate assignment never reaches observed (p=0.0005; null max 0.55). Bootstrap 95% CI [1.24, 1.80] — lower bound ~2× the null max. *(figure: geometry_null_hist.png)*
- **Specificity:** regress (lat,lon) out of the centroids and re-probe — East/West lift +0.30→**+0.00** (exactly chance); linguistic family +0.14→**−0.04**. EU membership not decodable at all (lift −0.10). Every non-geographic axis is a geographic shadow or absent.

### 4. Limitations (state plainly)
Single model; layer 28 is a carry-forward, not located (`locate` incomplete); 30 points (mitigated by null+CI but bootstrap CI is wide ~0.56); capital coords proxy country position; **retrieval-vs-computation not yet separated** — cannot yet claim the map is *traversed* rather than *stored*.

### 5. Reproduction
Self-contained notebook `result/country_borders_reproduce.ipynb`; robustness module `code/analyses/geometry_robustness.py`; full report `result/REPORT.md`. All in `causalab` (Goodfire's open-source causal-abstraction framework).

---

## B. causalab PR scope

Target: `goodfire-ai/causalab`. Keep it tight and reviewable — this PR *is* the high-signal contact with the maintainers (Atticus Geiger / Goodfire).

**Include:**
1. `causalab/tasks/country_borders/` — the task package (already clean) + `causalab/configs/task/country_borders.yaml` (incl. the `colormap` fix).
2. A shipped analysis `causalab/analyses/geometry_isomorphism/` promoting `geometry_robustness.py` + the centroid/regression pipeline into a proper module with `README.md` (Saved-artifacts table, Research-question section) per repo conventions.
3. `demos/country_borders_geometry.ipynb` — the reproduction notebook, repathed off the session tree.

**Exclude:** session-local `agent_logs/` content, the messy working notebook, raw artifacts.

**PR description:** lead with the relational-task framing and the causal-abstraction angle; cite Gurnee & Tegmark for positioning; link the LW post. One paragraph, no overclaiming.

---

## C. Outreach checklist (sequence — do not reorder)

1. [ ] Run #2 (retrieval-vs-computation) + #4 (locate) — the narrative upgrades. **#2 before any public post**: it's what makes this distinct from known work.
2. [ ] Finalize REPORT.md with #2/#4 results.
3. [ ] Polish the LW/AF post from §A; embed both figures; explicit prior-work positioning.
4. [ ] Open the causalab PR (§B) — clean, small, well-described.
5. [ ] Publish the LW/AF post; link the PR and the reproduction notebook.
6. [ ] Short X/Twitter thread; tag Goodfire, Atticus Geiger, Wes Gurnee.
7. [ ] *Then* warm outreach: reference the post + PR in a Goodfire Fellowship application (causal-analysis track) or a short, specific email — "built X in your framework, here's the post + PR," not a cold result dump.

**Honest gate:** steps 3–7 are materially weaker without step 1's #2 result. If bandwidth is limited, do #2 first; everything else compounds off it.
