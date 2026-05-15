#!/usr/bin/env python
"""#2 — retrieval vs. computation: is the Europe map *traversed* or just *stored*?

Gurnee & Tegmark (arXiv 2310.02207) showed LLMs *store* a linear world map,
read directly off an entity's own activation. Our REPORT.md result recovers a
map at the *answer* position of a *relational* border task — the country is
reached by computation ("what is east of X"), its name never appears in the
prompt for its own centroid. This script runs the controlled within-task
contrast that separates the two:

  relational  : country_borders, last_token features, centroids aggregated by
                ANSWER country (NEIGHBOR_OF) — the computed representation.
                Source: the existing subspace run (`--relational-root`).

  direct      : the SAME country_borders prompts, same model, same layer 28,
                same PCA-32, but features read at the COUNTRY (entity) token
                and grouped by the NAMED country — the stored representation.
                Source: the `country_borders_subspace_entitypos` run
                (`--direct-root`; produced into a sibling experiment-root).

For each condition we run the identical geometry pipeline (PCA -> (lat,lon)
regression -> permutation null + bootstrap CI, reusing geometry_robustness),
then a Procrustes alignment between the two per-country geometries.

Interpretation:
  * direct map-like (replicates G&T) AND relational map-like, with high
    Procrustes agreement  ->  the model uses ONE coherent map both to store
    positions and to answer relational queries: the map is *traversed*. This
    is the differentiating claim for the writeup.
  * relational NOT map-like, or the two geometries unrelated  ->  the border
    task is solved some other way; do not make the traversal claim.

Run on the box with both subspace runs (RunPod), after producing the
entity-position run:

    bash scripts/run_exp.sh \\
      --experiment-root agent_logs/2026-05-13--nationality-geometry--clever-falcon/artifacts/country_borders/llama31_8b/_entitypos \\
      country_borders_subspace_entitypos

    uv run python agent_logs/2026-05-13--nationality-geometry--clever-falcon/\
code/analyses/retrieval_vs_computation.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.torch import load_file
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from causalab.tasks.country_borders import COUNTRIES, LAT_LON_OF

SESSION_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SESSION_DIR / "code" / "analyses"))
import geometry_robustness as gr  # noqa: E402  (build_centroids, best_rotation, null)


def _load(subspace_root: Path):
    blob = load_file(subspace_root / "features" / "training_features.safetensors")
    feats = blob["features"].cpu().float().numpy()
    with open(subspace_root / "train_dataset.json") as f:
        ds = json.load(f)
    return feats, ds


def direct_centroids(subspace_root: Path):
    """Per-NAMED-country centroids (group features by the input country).

    This is the direct entity readout — the country is named in the prompt and
    we average its own-token features across every prompt that mentions it.
    """
    feats, ds = _load(subspace_root)
    by = {}
    for j, ex in enumerate(ds):
        by.setdefault(ex["input"]["country"], []).append(j)
    countries, C = [], []
    for c in COUNTRIES:
        if c in by:
            countries.append(c)
            C.append(feats[by[c]].mean(axis=0))
    return countries, np.asarray(C), sum(len(v) for v in by.values())


def geometry(countries, centroids, n_perm, n_boot, seed):
    pca = PCA(n_components=3).fit(centroids)
    P = pca.transform(centroids)
    geo = np.column_stack([[LAT_LON_OF[c][0] for c in countries],
                           [LAT_LON_OF[c][1] for c in countries]])
    per_pc = []
    for k in range(3):
        r = LinearRegression().fit(geo, P[:, k])
        per_pc.append({"pc": k, "r2": round(r.score(geo, P[:, k]), 3),
                       "beta_lat": round(float(r.coef_[0]), 4),
                       "beta_lon": round(float(r.coef_[1]), 4),
                       "var": round(float(pca.explained_variance_ratio_[k]), 3)})
    null = gr.geographic_isomorphism_null(countries, centroids,
                                          n_perm=n_perm, n_boot=n_boot, seed=seed)
    null.pop("_null_samples", None)
    return {"n": len(countries), "per_pc": per_pc, "isomorphism_null": null,
            "_P2": P[:, :2], "_countries": countries}


def main() -> None:
    ar = SESSION_DIR / "artifacts" / "country_borders" / "llama31_8b"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--relational-root", type=Path,
                    default=ar / "subspace" / "pca_k32" / "country")
    ap.add_argument("--direct-root", type=Path,
                    default=ar / "_entitypos" / "subspace" / "pca_k32" / "country")
    ap.add_argument("--out", type=Path, default=SESSION_DIR / "result")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for label, root in (("relational", args.relational_root),
                        ("direct", args.direct_root)):
        if not (root / "features" / "training_features.safetensors").exists():
            raise SystemExit(
                f"[{label}] features missing under {root}. The 'direct' set "
                "comes from the country_borders_subspace_entitypos run — see "
                "this file's docstring for the exact command."
            )

    # relational: reuse the published answer-country centroid construction.
    rel_countries, rel_C, rel_n = gr.build_centroids(args.relational_root)
    dir_countries, dir_C, dir_n = direct_centroids(args.direct_root)
    print(f"relational centroids: {len(rel_countries)}  ({rel_n} examples)")
    print(f"direct centroids    : {len(dir_countries)}  ({dir_n} examples)\n")

    rel = geometry(rel_countries, rel_C, args.n_perm, args.n_boot, args.seed)
    dr = geometry(dir_countries, dir_C, args.n_perm, args.n_boot, args.seed)

    def show(tag, g):
        iso = g["isomorphism_null"]
        print(f"=== {tag} (n={g['n']}) ===")
        for p in g["per_pc"]:
            print(f"  PC{p['pc']}: R^2={p['r2']:.3f} "
                  f"beta_lat={p['beta_lat']:+.3f} beta_lon={p['beta_lon']:+.3f}")
        print(f"  combined R^2 {iso['observed_combined_r2']}/2.0  "
              f"p={iso['permutation']['p_value']}  "
              f"CI95={iso['bootstrap']['ci95']}\n")

    show("RELATIONAL (computed)", rel)
    show("DIRECT (stored)", dr)

    # Same map? Procrustes over the countries present in both conditions.
    shared = [c for c in rel["_countries"] if c in set(dr["_countries"])]
    ri = {c: i for i, c in enumerate(rel["_countries"])}
    di = {c: i for i, c in enumerate(dr["_countries"])}
    A = rel["_P2"][[ri[c] for c in shared]]
    B = dr["_P2"][[di[c] for c in shared]]
    m1, m2, disparity = procrustes(A, B)
    r = float(np.corrcoef(m1.ravel(), m2.ravel())[0, 1])
    print(f"=== same map? Procrustes over {len(shared)} shared countries ===")
    print(f"  disparity = {disparity:.4f}  (0 = identical shape)")
    print(f"  aligned-coord correlation r = {r:.3f}\n")

    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        "relational": {k: v for k, v in rel.items() if not k.startswith("_")},
        "direct": {k: v for k, v in dr.items() if not k.startswith("_")},
        "same_map": {"n_shared": len(shared),
                     "procrustes_disparity": round(disparity, 4),
                     "aligned_correlation": round(r, 3)},
    }
    (args.out / "retrieval_vs_computation.json").write_text(
        json.dumps(payload, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        for a, (tag, g) in zip(ax, (("RELATIONAL (computed)", rel),
                                    ("DIRECT (stored)", dr))):
            geo = np.column_stack([[LAT_LON_OF[c][0] for c in g["_countries"]],
                                   [LAT_LON_OF[c][1] for c in g["_countries"]]])
            th, _ = gr.best_rotation(g["_P2"], geo)
            R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            Q = g["_P2"] @ R
            a.scatter(-Q[:, 1], -Q[:, 0], s=70, c="steelblue",
                      edgecolors="black", linewidths=0.5)
            for c, q in zip(g["_countries"], Q):
                a.annotate(c, (-q[1], -q[0]), fontsize=7, alpha=0.7)
            a.set_title(f"{tag}\ncombined R^2="
                        f"{g['isomorphism_null']['observed_combined_r2']}")
            a.grid(alpha=0.3)
        fig.suptitle(f"Procrustes disparity {disparity:.3f}, "
                     f"aligned r {r:.2f} (lower / higher = same map)")
        fig.tight_layout()
        (args.out / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out / "figures" / "retrieval_vs_computation.png",
                    dpi=150, bbox_inches="tight")
        print(f"wrote {args.out/'retrieval_vs_computation.json'} and "
              f"figures/retrieval_vs_computation.png")
    except Exception as e:
        print(f"(figure skipped: {e}); wrote "
              f"{args.out/'retrieval_vs_computation.json'}")


if __name__ == "__main__":
    main()
