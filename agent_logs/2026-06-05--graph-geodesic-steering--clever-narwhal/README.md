# 2026-06-05--graph-geodesic-steering--clever-narwhal

This session implements a **data-aware "true geodesic" path mode** for the `path_steering` analysis on
the `country_borders` task (Gemma 3 4B-PT). Today's `geometric` mode interpolates linearly in the
fitted 2-D (lat, lon) intrinsic space, which for far pairs (Spain → Russia) only sweeps countries that
happen to lie on the straight parameter-space segment. The deliverable is a new path mode that builds a
k-NN graph over the answer-country centroids in PCA space, finds the Dijkstra shortest path between the
two endpoints, and steers along the resulting waypoint sequence with straight-line interpolation per
segment — so the prediction is forced to sweep through geographically-intermediate countries. Scope: a
reusable geodesic primitive (method), the path-mode wiring + centroid-cloud threading through
`build_path` (analysis), and a comparison of graph-geodesic vs `geometric` vs `linear` output landscapes.

## Layout

- `plan/` — research objective, task-spec drafts, approval-checkpoint logs
- `run/` — resolved-config snapshot (`--cfg job` output), `run.log`, slurm logs
- `result/` — `REPORT.md` (single consolidated interpretation written by `/interpret-experiment`), `figures/` for embedded plots/tables
- `code/` — session-local Python + Hydra (via `/setup-methods`, `/setup-analyses`, `/run-experiment`)
- `artifacts/` — raw experiment outputs at `{task}/{model}/{analysis}/...`
- `issues.md` — top-level issue log spanning all phases (managed by `/document-issues`)
