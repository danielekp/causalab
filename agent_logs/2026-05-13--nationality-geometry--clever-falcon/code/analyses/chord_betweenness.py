#!/usr/bin/env python
"""Stage A of the steering experiment — geometric betweenness (CPU, no new run).

Hypothesis (the steering prediction, tested *geometrically* first): if the
Europe manifold is locally metric, then for a country B that is geographically
*between* far-apart A and C, B's centroid c_B should lie near the chord
c_A -> c_C in activation space, at roughly the same fractional position B holds
geographically. If this holds, the Stage-B causal steering (interpolate
c_A -> c_C, expect B to appear at the midpoint) is well-motivated. If it
fails, the manifold is curved — itself a finding, learned before spending GPU.

For every triple (A, B, C) over the relational answer-country centroids
(§6.3 featurisation; PCA-32 space, layer 28):
  * geographic: project B onto the A-C segment in (lat, lon) ->
    t_geo in [0,1] and perpendicular distance; classify "between" vs "off".
  * activation: project c_B onto the chord c_A->c_C, in the recovered
    geographic PC plane AND in full PCA-32 space -> t_act and relative
    residual (perp distance / chord length).

Predictions if the map is metric:
  (1) corr(t_geo, t_act) high & positive for "between" triples;
  (2) relative residual small for "between", large for "off"/random;
  (3) (1) survives a permutation null over country<->centroid labels.

Reuses geometry_robustness.build_centroids so featurisation is identical to
§6.3-§6.5. Run on the box with the L28 relational subspace features:

    uv run python agent_logs/2026-05-13--nationality-geometry--clever-falcon/\
code/analyses/chord_betweenness.py
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

SESSION_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SESSION_DIR / "code" / "analyses"))
import geometry_robustness as gr  # noqa: E402

from causalab.tasks.country_borders import LAT_LON_OF  # noqa: E402

# A,C must be genuinely far (>= this percentile of pairwise geo distance);
# "between" = B's projection well inside the segment and close to the line;
# "off" = B clearly to the side. Tunable; defaults are conservative.
FAR_PCTILE = 60.0
BETWEEN_T = (0.20, 0.80)
BETWEEN_PERP = 0.25      # perp_geo / chord_geo
OFF_PERP = 0.60


def _proj(p, a, c):
    """Return (t, rel_resid): projection param of p onto seg a->c and the
    perpendicular residual normalised by the chord length."""
    ac = c - a
    L2 = float(ac @ ac)
    if L2 <= 1e-12:
        return np.nan, np.nan
    t = float((p - a) @ ac / L2)
    resid = np.linalg.norm((p - a) - t * ac)
    return t, resid / (np.sqrt(L2) + 1e-12)


def main() -> None:
    ar = SESSION_DIR / "artifacts" / "country_borders" / "llama31_8b"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subspace-root", type=Path,
                    default=ar / "subspace" / "pca_k32" / "country")
    ap.add_argument("--out", type=Path, default=SESSION_DIR / "result")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not (args.subspace_root / "features"
            / "training_features.safetensors").exists():
        raise SystemExit(
            f"L28 relational features missing under {args.subspace_root}. "
            "Run on the box with the subspace artifacts (RunPod)."
        )

    countries, C_full, n_ex = gr.build_centroids(args.subspace_root)
    idx = {c: i for i, c in enumerate(countries)}
    geo = {c: np.array(LAT_LON_OF[c], float) for c in countries}
    P2 = PCA(n_components=2).fit_transform(C_full)        # the map plane
    cp = {c: P2[idx[c]] for c in countries}
    cf = {c: C_full[idx[c]] for c in countries}

    pair_d = [np.linalg.norm(geo[a] - geo[b])
              for a, b in combinations(countries, 2)]
    far_min = float(np.percentile(pair_d, FAR_PCTILE))

    between, off = [], []
    for a, c in combinations(countries, 2):
        ga, gc = geo[a], geo[c]
        chord_g = np.linalg.norm(gc - ga)
        if chord_g < far_min:
            continue
        for b in countries:
            if b in (a, c):
                continue
            tg, rg = _proj(geo[b], ga, gc)
            if np.isnan(tg):
                continue
            tp, rp = _proj(cp[b], cp[a], cp[c])     # plane
            tf, rf = _proj(cf[b], cf[a], cf[c])     # full 32-d
            rec = {"A": a, "B": b, "C": c, "t_geo": tg, "perp_geo": rg,
                   "t_plane": tp, "resid_plane": rp,
                   "t_full": tf, "resid_full": rf,
                   "collinearity": round(1.0 - rg, 3)}
            if BETWEEN_T[0] <= tg <= BETWEEN_T[1] and rg <= BETWEEN_PERP:
                between.append(rec)
            elif rg >= OFF_PERP:
                off.append(rec)

    if not between:
        raise SystemExit("no 'between' triples found — loosen thresholds.")

    def col(rows, k):
        return np.array([r[k] for r in rows], float)

    tg_b = col(between, "t_geo")
    rng = np.random.default_rng(args.seed)

    def corr(x, y):
        return float(np.corrcoef(x, y)[0, 1])

    res = {"n_centroids": len(countries), "n_examples": n_ex,
           "far_min_geo": round(far_min, 3),
           "n_between": len(between), "n_off": len(off)}
    for space in ("plane", "full"):
        tb = col(between, f"t_{space}")
        rb = col(between, f"resid_{space}")
        ro = col(off, f"resid_{space}") if off else np.array([np.nan])
        r_obs = corr(tg_b, tb)
        # permutation null: shuffle which centroid each country owns, rebuild
        # the per-triple t in this space, recompute corr(t_geo, t_act).
        order = list(range(len(countries)))
        null = np.empty(args.n_perm)
        src = cp if space == "plane" else cf
        vecs = np.array([src[c] for c in countries])
        for i in range(args.n_perm):
            rng.shuffle(order)
            perm = {countries[j]: vecs[order[j]] for j in range(len(countries))}
            tt = [_proj(perm[r["B"]], perm[r["A"]], perm[r["C"]])[0]
                  for r in between]
            null[i] = corr(tg_b, np.array(tt))
        p = (1 + int(np.sum(null >= r_obs))) / (1 + args.n_perm)
        res[space] = {
            "corr_tgeo_tact": round(r_obs, 3),
            "perm_p": round(p, 5),
            "null_mean": round(float(np.nanmean(null)), 3),
            "resid_between_median": round(float(np.median(rb)), 3),
            "resid_off_median": round(float(np.nanmedian(ro)), 3),
        }
        print(f"[{space:5s}] corr(t_geo,t_act)={r_obs:+.3f} (p={p:.4f})  "
              f"resid between/off median = "
              f"{np.median(rb):.3f} / {np.nanmedian(ro):.3f}")

    # Illustrative near-collinear, far triples for the writeup / Stage B.
    top = sorted(between, key=lambda r: (-r["collinearity"],
                                         abs(r["t_geo"] - 0.5)))[:12]
    res["illustrative_triples"] = [
        {k: (round(v, 3) if isinstance(v, float) else v)
         for k, v in r.items()} for r in top
    ]
    print("\ntop near-collinear far triples (A -> B -> C):")
    for r in top[:8]:
        print(f"  {r['A']:>14s} -> {r['B']:>14s} -> {r['C']:<14s}  "
              f"t_geo={r['t_geo']:.2f} t_plane={r['t_plane']:+.2f} "
              f"resid_plane={r['resid_plane']:.2f}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "chord_betweenness.json").write_text(json.dumps(res, indent=2))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        ax[0].scatter(tg_b, col(between, "t_plane"), s=18, alpha=0.5,
                      c="steelblue")
        ax[0].plot([0, 1], [0, 1], "k--", lw=1)
        ax[0].set(xlabel="t_geo (geographic position on A-C)",
                  ylabel="t_act (plane)",
                  title=f"between triples: r={res['plane']['corr_tgeo_tact']}")
        ax[0].grid(alpha=0.3)
        ax[1].hist(col(between, "resid_plane"), bins=30, alpha=0.7,
                   label="between", color="steelblue", density=True)
        if off:
            ax[1].hist(col(off, "resid_plane"), bins=30, alpha=0.5,
                       label="off-line", color="grey", density=True)
        ax[1].set(xlabel="relative residual to chord (plane)",
                  ylabel="density",
                  title="B near the A-C chord?")
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        fig.tight_layout()
        (args.out / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out / "figures" / "chord_betweenness.png", dpi=150)
        print(f"\nwrote {args.out/'chord_betweenness.json'} and "
              f"figures/chord_betweenness.png")
    except Exception as e:
        print(f"\n(figure skipped: {e}); wrote "
              f"{args.out/'chord_betweenness.json'}")


if __name__ == "__main__":
    main()
