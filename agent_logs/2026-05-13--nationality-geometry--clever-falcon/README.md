# 2026-05-13--nationality-geometry--clever-falcon

Probe how Llama-3.1-8B represents European country geometry. Pivoted from capital-retrieval (v1, v2) to **directional border queries** on 2026-05-14 after diagnosing that capital-retrieval is methodologically mismatched with the manifold paradigm (it is retrieval, not computation; the entity token appears in every prompt for its own centroid).

**Current task (`country_borders`, 2026-05-14, active):** 30 European countries × 8 directions (N/NE/E/SE/S/SW/W/NW) × 4 paraphrase templates = 616 valid prompts. Question form: *"Which country lies to the east of Spain?"*. The model must traverse its internal country geometry to answer, and centroids are built per **answer country** (averaging across many entity → primary-neighbor prompts) so each centroid is entity-disentangled. Strong prior: if the model has internalized a Europe map, PC1/PC2 should be an affine function of (lat, lon). Notebook: `result/country_borders_geometry.ipynb`.

**Archived (capital-retrieval lineage):**
- **v1 (2026-05-13):** 40 countries × 6 continents, probing continent/Western-vs-Global-South/colonial-pair/linguistic hypotheses. Layer-24 last-token PC0 was dominated by Europe-vs-Latin-America; continent clustering survived BPE confound (ratio 1.42→1.24) but colonial-pair hypotheses largely failed (Portugal↔Brazil far; UK↔India far; only Spain↔Mexico close, likely a Mexico-in-European-orbit artifact). Artifacts at `artifacts/_archived_nationality_capitals/llama31_8b_v1_40country/`.
- **v2 (2026-05-14):** Pruned to 33 European countries (capital-retrieval). Designed and scaffolded but never executed — diagnosed the methodological mismatch before running and pivoted to `country_borders`. Code in git history.

Deliverable: a `result/REPORT.md` interpreting the European centroid geometry as evidence for (or against) the model encoding country relations as a coherent map.

## Layout

- `plan/` — research objective, task-spec drafts, approval-checkpoint logs
- `run/` — resolved-config snapshot (`--cfg job` output), `run.log`, slurm logs
- `result/` — `REPORT.md` (single consolidated interpretation written by `/interpret-experiment`), `figures/` for embedded plots/tables
- `code/` — session-local Python + Hydra (via `/setup-methods`, `/setup-analyses`, `/run-experiment`)
- `artifacts/` — raw experiment outputs at `{task}/{model}/{analysis}/...`
- `issues.md` — top-level issue log spanning all phases (managed by `/document-issues`)
