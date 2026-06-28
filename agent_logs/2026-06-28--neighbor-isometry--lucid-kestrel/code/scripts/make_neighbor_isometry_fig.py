#!/usr/bin/env python
"""Fig-3-style composite for ONE direction of the neighbor-output isometry.

Renders the manifold-steering paper's three-column layout (arXiv 2605.05115) for
a single compass direction of the per-direction subject walk
(country_borders / Gemma3-4B-PT), reading the artifacts written by the
session-local `neighbor_isometry` analysis:

    rows                cols
    ----                ----
    Behavior space      [ PCA top | PCA side ]  [ MDS ]  [ Scaled isometry scatter ]
    Activation space    [ PCA top | PCA side ]  [ MDS ]  [ Scaled isometry scatter ]

Inputs (all under ${experiment_root}/neighbor_isometry/default/dir_<D>/), MODEL-FREE:
  * manifolds.safetensors                 -> rebuild the activation + belief TPS splines
  * criteria/isometry/{geometric,linear}/ -> D_manifold / D_output + Pearson r (metrics.json)
plus output_manifold/**/hellinger_pca.safetensors for the belief PCA projection.

Both manifolds are 2-D (subject lat/lon) TPS splines, rebuilt from the saved
control points + centroids — no model load. This is a STRATIFIED, single-direction
view (direction held constant), so the activation sheet varies only by subject
geography and the belief sheet by the neighbor-output distribution.

Usage (on the pod, from the repo root):
    python agent_logs/2026-06-28--neighbor-isometry--lucid-kestrel/code/scripts/make_neighbor_isometry_fig.py \
        --experiment-root /workspace/causalab/artifacts/country_borders/gemma3_4b_pt \
        --direction E \
        --out /workspace/causalab/artifacts/country_borders/gemma3_4b_pt/neighbor_isometry/fig_neighbor_isometry_E.png
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("neighfig")

CMAP = "rainbow"  # matches the task colormap; colors the sequential lat axis


# ---------------------------------------------------------------------------
# Artifact discovery + loading
# ---------------------------------------------------------------------------


def _first(pattern: str) -> str | None:
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None


def discover_paths(experiment_root: str, direction: str) -> dict:
    """Resolve every artifact path we need for one direction."""
    root = os.path.abspath(experiment_root)
    dir_out = os.path.join(root, "neighbor_isometry", "default", f"dir_{direction}")
    if not os.path.isdir(dir_out):
        raise FileNotFoundError(
            f"No neighbor_isometry output for direction {direction!r} at {dir_out}. "
            "Run the neighbor_isometry analysis first, or check --direction."
        )

    iso = {}
    for mode in ("geometric", "linear"):
        d = os.path.join(dir_out, "criteria", "isometry", mode)
        t = os.path.join(d, "tensors.safetensors")
        m = os.path.join(d, "metrics.json")
        if os.path.exists(t):
            iso[mode] = {"tensors": t, "metrics": m if os.path.exists(m) else None}

    manifolds = os.path.join(dir_out, "manifolds.safetensors")
    if not os.path.exists(manifolds):
        raise FileNotFoundError(f"manifolds.safetensors not found at {manifolds}")

    om_base = os.path.join(root, "output_manifold")
    hpca = _first(os.path.join(om_base, "**", "hellinger_pca.safetensors"))

    return {
        "root": root,
        "dir_out": dir_out,
        "iso": iso,
        "manifolds": manifolds,
        "hellinger_pca_path": hpca,
    }


def rebuild_manifolds(manifolds_path: str):
    """Rebuild (act_manifold, mean, std, bel_manifold) from saved control pts + centroids."""
    import torch
    from safetensors.torch import load_file
    from causalab.methods.spline.builders import build_spline_manifold
    from causalab.methods.spline.belief_fit import _prob_to_hellinger

    t = load_file(manifolds_path)
    act_cp = t["act_control_points"].float()
    act_cent = t["act_centroids"].float()
    mean = t["act_mean"].float()
    std = t["act_std"].float()
    bel_cp = t["bel_control_points"].float()
    bel_prob = t["bel_centroids_prob"].float()
    bel_hell = _prob_to_hellinger(bel_prob).float()

    act_manifold = build_spline_manifold(
        control_points=act_cp, centroids=act_cent,
        intrinsic_dim=2, ambient_dim=act_cent.shape[1], smoothness=0.0,
        sphere_project=False,
    ).cpu().eval()
    bel_manifold = build_spline_manifold(
        control_points=bel_cp, centroids=bel_hell,
        intrinsic_dim=2, ambient_dim=bel_hell.shape[1], smoothness=0.0,
        sphere_project=True,
    ).cpu().eval()
    return act_manifold, mean, std, bel_manifold


# ---------------------------------------------------------------------------
# Manifold helpers (decode sheets / geodesics)
# ---------------------------------------------------------------------------


def _u_grid(control_points, res: int):
    import torch

    cp = control_points
    los = cp.min(dim=0).values
    his = cp.max(dim=0).values
    axes = [torch.linspace(float(los[d]), float(his[d]), res) for d in range(cp.shape[1])]
    gx, gy = torch.meshgrid(axes[0], axes[1], indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


def _geodesic(u_a, u_b, n: int):
    import torch

    t = torch.linspace(0, 1, n).unsqueeze(1)
    return u_a.unsqueeze(0) + t * (u_b - u_a).unsqueeze(0)


def activation_3d_bundle(manifold, mean, std, sheet_res: int, n_path: int):
    """3D-PCA coords of the activation manifold sheet, centroids, and geodesic paths."""
    import torch
    from sklearn.decomposition import PCA

    cp = manifold.control_points.cpu()

    def decode_ambient(u):
        with torch.no_grad():
            return (manifold.decode(u) * (std + 1e-6) + mean).cpu().numpy()

    sheet_amb = decode_ambient(_u_grid(cp, sheet_res))
    cent_amb = decode_ambient(cp)
    pca = PCA(n_components=3).fit(sheet_amb)
    sheet_3d = pca.transform(sheet_amb)
    cent_3d = pca.transform(cent_amb)

    paths_3d = []
    W = cp.shape[0]
    d2 = ((cp.unsqueeze(0) - cp.unsqueeze(1)) ** 2).sum(-1)
    d2.fill_diagonal_(float("inf"))
    nn = d2.argmin(dim=1)
    for i in range(W):
        paths_3d.append(pca.transform(decode_ambient(_geodesic(cp[i], cp[nn[i]], n_path))))

    return {"sheet": sheet_3d, "centroids": cent_3d, "paths": paths_3d,
            "color": cp[:, 0].numpy(), "sheet_res": sheet_res}


def belief_3d_bundle(manifold, hellinger_pca, sheet_res: int, n_path: int):
    """3D coords of the belief manifold via the saved hellinger_pca projection."""
    import torch

    cp = manifold.control_points.cpu()
    pca_dim = int(hellinger_pca.n_features_in_)

    def _fit_dim(arr: np.ndarray) -> np.ndarray:
        if arr.shape[-1] == pca_dim:
            return arr
        if arr.shape[-1] == pca_dim - 1:
            other = np.clip(1.0 - arr.sum(-1, keepdims=True), 0, None)
            return np.concatenate([arr, other], axis=-1)
        return arr[:, :pca_dim]

    def decode_hellinger(u):
        with torch.no_grad():
            h = manifold.decode(u).cpu().numpy()  # already Hellinger (sqrt-p)
        return hellinger_pca.transform(_fit_dim(h).astype(np.float32))

    sheet_3d = decode_hellinger(_u_grid(cp, sheet_res))
    cent_3d = decode_hellinger(cp)

    paths_3d = []
    W = cp.shape[0]
    d2 = ((cp.unsqueeze(0) - cp.unsqueeze(1)) ** 2).sum(-1)
    d2.fill_diagonal_(float("inf"))
    nn = d2.argmin(dim=1)
    for i in range(W):
        paths_3d.append(decode_hellinger(_geodesic(cp[i], cp[nn[i]], n_path)))

    return {"sheet": sheet_3d, "centroids": cent_3d, "paths": paths_3d,
            "cloud": None, "color": cp[:, 0].numpy(), "sheet_res": sheet_res}


# ---------------------------------------------------------------------------
# Isometry artifacts (MDS + scatter)
# ---------------------------------------------------------------------------


def load_iso(tensors_path: str, metrics_path: str | None):
    from safetensors.numpy import load_file

    t = load_file(tensors_path)
    out = {"D_manifold": t["D_manifold"], "D_output": t["D_output"]}
    if "grid_points_valid" in t:
        out["grid"] = t["grid_points_valid"]
    r = float("nan")
    if metrics_path and os.path.exists(metrics_path):
        with open(metrics_path) as f:
            r = json.load(f).get("pearson_r", float("nan"))
    out["r"] = r
    return out


def mds3(D: np.ndarray) -> np.ndarray:
    from sklearn.manifold import MDS

    m = MDS(n_components=3, dissimilarity="precomputed", normalized_stress="auto",
            random_state=42, n_init=8)
    return m.fit_transform(D)


def _unit(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(0)
    s = np.abs(x).max()
    return x / s if s > 0 else x


def upper(D: np.ndarray) -> np.ndarray:
    iu = np.triu_indices_from(D, k=1)
    return D[iu]


# ---------------------------------------------------------------------------
# Rendering  (reused from make_fig3_composite.py)
# ---------------------------------------------------------------------------


def render(act, bel, iso_geo, iso_lin, out_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = matplotlib.colormaps[CMAP]

    def cnorm(c):
        lo, hi = float(np.min(c)), float(np.max(c))
        return (c - lo) / (hi - lo + 1e-9)

    fig = plt.figure(figsize=(19, 9))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.5, 1.15, 1.0, 1.0],
                          hspace=0.28, wspace=0.30)

    def pca_cell(col, bundle, row_title, cloud_key=None):
        sub = gs[col, 0].subgridspec(1, 2, wspace=0.05)
        views = [("Top view", 0, 1), ("Side view", 0, 2)]
        for k, (vname, ax_i, ax_j) in enumerate(views):
            ax = fig.add_subplot(sub[0, k])
            sh = bundle["sheet"]
            ax.scatter(sh[:, ax_i], sh[:, ax_j], s=2, c="0.82", alpha=0.5,
                       linewidths=0, zorder=1)
            if cloud_key and bundle.get(cloud_key) is not None:
                cl = bundle[cloud_key]
                ax.scatter(cl[:, ax_i], cl[:, ax_j], s=4, c="0.7", alpha=0.15,
                           linewidths=0, zorder=0)
            cn = cnorm(bundle["color"])
            for i, p in enumerate(bundle["paths"]):
                ax.plot(p[:, ax_i], p[:, ax_j], color=cmap(cn[i]), lw=1.0,
                        alpha=0.6, zorder=2)
            ce = bundle["centroids"]
            ax.scatter(ce[:, ax_i], ce[:, ax_j], s=42, c=cmap(cn), marker="D",
                       edgecolors="k", linewidths=0.4, zorder=3)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(vname, fontsize=9, color="#555")
            if k == 0:
                ax.set_ylabel(row_title, fontsize=12, fontweight="bold")

    def mds_panel(col, embeds_with_labels, color, panel_title):
        ax = fig.add_subplot(gs[col, 2], projection="3d")
        cn = cnorm(color)
        xoff = 0.0
        for emb, lbl, marker in embeds_with_labels:
            e = _unit(emb)
            ax.scatter(e[:, 0] + xoff, e[:, 1], e[:, 2], c=cmap(cn),
                       s=28, marker=marker, edgecolors="k", linewidths=0.3,
                       depthshade=True)
            if lbl:
                ax.text(xoff, e[:, 1].max(), e[:, 2].max(), lbl, fontsize=8, color="#333")
            xoff += 2.4
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(panel_title, fontsize=10, color="#333")
        ax.view_init(elev=18, azim=-60)

    def scatter_panel(col, iso, label):
        ax = fig.add_subplot(gs[col, 3])
        dx, dy = upper(iso["D_manifold"]), upper(iso["D_output"])
        ax.scatter(dx, dy, s=9, c="#2b6cb0", alpha=0.55, linewidths=0)
        if len(dx) > 1:
            a, b = np.polyfit(dx, dy, 1)
            xs = np.array([dx.min(), dx.max()])
            ax.plot(xs, a * xs + b, color="#c0392b", lw=1.3)
        ax.text(0.05, 0.92, f"{label}\nr = {iso['r']:.3f}",
                transform=ax.transAxes, fontsize=11, va="top",
                color="#c0392b", fontweight="bold")
        ax.set_xlabel("Activation manifold path length", fontsize=9)
        ax.set_ylabel("Behavior manifold path length", fontsize=9)
        ax.tick_params(labelsize=7)

    V = iso_geo["D_output"].shape[0]
    iso_color = (iso_geo["grid"][:, 0] if iso_geo.get("grid") is not None
                 and iso_geo["grid"].shape[0] == V else np.arange(V))

    pca_cell(0, bel, "Behavior space", cloud_key="cloud")
    mds_panel(0, [(mds3(iso_geo["D_output"]), None, "o")], iso_color, "Behavior Space MDS")
    scatter_panel(0, iso_geo, "Manifold paths")

    pca_cell(1, act, "Activation space")
    act_embeds = [(mds3(iso_geo["D_manifold"]), "Manifold", "o")]
    if iso_lin is not None:
        act_embeds.append((mds3(iso_lin["D_manifold"]), "Linear", "^"))
    mds_panel(1, act_embeds, iso_color, "Activation Space MDS")
    scatter_panel(1, iso_lin if iso_lin is not None else iso_geo,
                  "Linear paths" if iso_lin is not None else "Manifold paths")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.99)
    fig.text(0.5, 0.005,
             "Structural correspondence (PCA)            "
             "MDS embedding            Scaled isometry",
             ha="center", fontsize=10, color="#777")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    pdf = os.path.splitext(out_path)[0] + ".pdf"
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s and %s", out_path, pdf)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiment-root", required=True)
    ap.add_argument("--direction", default="E",
                    help="compass direction to render (N, NE, E, SE, S, SW, W, NW)")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--sheet-res", type=int, default=36)
    ap.add_argument("--n-path", type=int, default=24)
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    paths = discover_paths(args.experiment_root, args.direction)
    log.info("dir_out         = %s", paths["dir_out"])
    log.info("hellinger_pca   = %s", paths["hellinger_pca_path"])
    log.info("isometry modes  = %s", list(paths["iso"].keys()))

    if "geometric" not in paths["iso"]:
        log.error("No geometric isometry tensors under %s/criteria/isometry/", paths["dir_out"])
        return 2
    if not paths["hellinger_pca_path"]:
        log.error("Could not locate hellinger_pca.safetensors under output_manifold/.")
        return 2

    iso_geo = load_iso(paths["iso"]["geometric"]["tensors"], paths["iso"]["geometric"]["metrics"])
    iso_lin = (load_iso(paths["iso"]["linear"]["tensors"], paths["iso"]["linear"]["metrics"])
               if "linear" in paths["iso"] else None)

    from causalab.io.sklearn_pca import load_pca

    hpca = load_pca(os.path.dirname(paths["hellinger_pca_path"]), "hellinger_pca")

    log.info("Rebuilding manifolds ...")
    act_mfd, mean, std, bel_mfd = rebuild_manifolds(paths["manifolds"])
    log.info("Building activation manifold bundle ...")
    act = activation_3d_bundle(act_mfd, mean, std, args.sheet_res, args.n_path)
    log.info("Building belief manifold bundle ...")
    bel = belief_3d_bundle(bel_mfd, hpca, args.sheet_res, args.n_path)

    title = args.title or (
        f"Neighbor-output isometry  |  country_borders / Gemma3-4B-PT  |  direction={args.direction} "
        f"(L12/country, per-direction subject walk)"
    )
    out = args.out or os.path.join(paths["root"], "neighbor_isometry",
                                   f"fig_neighbor_isometry_{args.direction}.png")
    render(act, bel, iso_geo, iso_lin, out, title)

    print("\n=== neighbor-output isometry figure ===")
    print(f"  direction                   = {args.direction}")
    print(f"  geometric (manifold paths)  r = {iso_geo['r']:.4f}")
    if iso_lin is not None:
        print(f"  linear    (chord paths)     r = {iso_lin['r']:.4f}")
    print(f"  figure: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
