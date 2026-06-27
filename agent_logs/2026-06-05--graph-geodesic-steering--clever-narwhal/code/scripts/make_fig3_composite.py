#!/usr/bin/env python
"""Static reproduction of the manifold-steering paper's Fig 2/3 (arXiv 2605.05115).

Lays out a single composite figure with the paper's three columns, for the
country_borders / Gemma3-4B-PT activation<->behavior isometry:

    rows                cols
    ----                ----
    Behavior space      [ PCA top | PCA side ]  [ MDS ]  [ Scaled isometry scatter ]
    Activation space    [ PCA top | PCA side ]  [ MDS ]  [ Scaled isometry scatter ]

Everything is read from artifacts already on disk after the isometry run; no
model is loaded (both manifolds are small TPS splines, CPU-only):

  * activation manifold (+ mean/std)  -> .../manifold_spline/ckpt_final.safetensors
  * belief manifold                   -> output_manifold/<sub>/manifold_spline/
  * hellinger_pca / per-example dists -> output_manifold/
  * isometry D-matrices (both modes)  -> <steering_dir>/criteria/isometry/{geometric,linear}/

Column 1 (structural correspondence) is built MODEL-FREE: the manifold sheet,
its centroids, and the geodesic "manifold paths" come straight from decoding
the spline, so it needs none of the steered-output tensors.

Columns 2 (MDS) and 3 (scatter) come from the saved D_manifold / D_output
distance matrices: D_manifold(geometric)=arc length, D_manifold(linear)=chord,
D_output=belief (Hellinger) arc length. The Pearson r printed on each scatter
is read from the run's metrics.json (geometric ~0.20, linear ~0.03).

Usage (on the pod, from the repo root):
    python agent_logs/<session>/code/scripts/make_fig3_composite.py \
        --experiment-root /workspace/causalab/artifacts/country_borders/gemma3_4b_pt \
        --steering-dir   /workspace/causalab/artifacts/country_borders/gemma3_4b_pt/path_steering/pca_k32_ctok/L12_country/spline_s0.0/country \
        --out            /workspace/causalab/artifacts/country_borders/gemma3_4b_pt/fig3_composite.png

Both directory args also accept globs; sensible auto-discovery is attempted when
--steering-dir is omitted (first path_steering/**/country leaf under the root).
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
log = logging.getLogger("fig3")

CMAP = "rainbow"  # matches the task colormap; colors the sequential lat axis


# ---------------------------------------------------------------------------
# Artifact discovery + loading
# ---------------------------------------------------------------------------


def _first(pattern: str) -> str | None:
    hits = sorted(glob.glob(pattern, recursive=True))
    return hits[0] if hits else None


def discover_paths(experiment_root: str, steering_dir: str | None) -> dict:
    """Resolve every artifact path we need, by globbing under the root."""
    root = os.path.abspath(experiment_root)

    if steering_dir is None:
        steering_dir = _first(os.path.join(root, "path_steering", "**", "country"))
        if steering_dir is None:
            # fall back: any dir holding criteria/isometry
            iso_any = _first(os.path.join(root, "**", "criteria", "isometry"))
            steering_dir = os.path.dirname(os.path.dirname(iso_any)) if iso_any else None
    if steering_dir is None or not os.path.isdir(steering_dir):
        raise FileNotFoundError(
            "Could not locate a path_steering output dir (the 'country' leaf "
            "that holds criteria/isometry/). Pass --steering-dir explicitly."
        )
    steering_dir = os.path.abspath(steering_dir)

    iso = {}
    for mode in ("geometric", "linear"):
        d = os.path.join(steering_dir, "criteria", "isometry", mode)
        t = os.path.join(d, "tensors.safetensors")
        m = os.path.join(d, "metrics.json")
        if os.path.exists(t):
            iso[mode] = {"tensors": t, "metrics": m if os.path.exists(m) else None}

    # Activation manifold checkpoint: a manifold_spline NOT under output_manifold.
    act_ckpt = None
    for p in sorted(
        glob.glob(os.path.join(root, "**", "manifold_spline", "ckpt_final.safetensors"),
                  recursive=True)
    ):
        if os.sep + "output_manifold" + os.sep not in p:
            act_ckpt = p
            break
    act_spline_dir = os.path.dirname(act_ckpt) if act_ckpt else None

    # Belief manifold: a manifold_spline UNDER output_manifold.
    bel_ckpt = _first(
        os.path.join(root, "output_manifold", "**", "manifold_spline",
                     "ckpt_final.safetensors")
    )
    bel_spline_dir = os.path.dirname(bel_ckpt) if bel_ckpt else None

    # hellinger_pca + per-example natural dists live directly under output_manifold/
    om_base = os.path.join(root, "output_manifold")
    hpca = _first(os.path.join(om_base, "**", "hellinger_pca.safetensors"))
    nat = _first(os.path.join(om_base, "**", "per_example_output_dists.safetensors"))

    return {
        "root": root,
        "steering_dir": steering_dir,
        "iso": iso,
        "act_spline_dir": act_spline_dir,
        "bel_spline_dir": bel_spline_dir,
        "hellinger_pca_path": hpca,
        "natural_dists_path": nat,
    }


# ---------------------------------------------------------------------------
# Manifold helpers (decode sheets / geodesics)  -- needs torch + causalab
# ---------------------------------------------------------------------------


def _u_grid(control_points, res: int):
    """Dense intrinsic grid spanning the control-point bounding box."""
    import torch

    cp = control_points
    los = cp.min(dim=0).values
    his = cp.max(dim=0).values
    axes = [torch.linspace(float(los[d]), float(his[d]), res) for d in range(cp.shape[1])]
    if cp.shape[1] == 1:
        return axes[0].unsqueeze(1)
    gx, gy = torch.meshgrid(axes[0], axes[1], indexing="ij")
    return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=1)


def _geodesic(u_a, u_b, n: int):
    import torch

    t = torch.linspace(0, 1, n).unsqueeze(1)
    return u_a.unsqueeze(0) + t * (u_b - u_a).unsqueeze(0)


def load_activation_manifold(spline_dir: str):
    """Return (manifold, mean, std) for the activation TPS."""
    from causalab.methods.spline.belief_fit import _load_spline_checkpoint

    manifold, _meta, pre = _load_spline_checkpoint(spline_dir)
    return manifold.cpu().eval(), pre["mean"].cpu(), pre["std"].cpu()


def load_belief_manifold(spline_dir: str):
    from causalab.methods.spline.belief_fit import _load_spline_checkpoint

    manifold, _meta, _pre = _load_spline_checkpoint(spline_dir)
    return manifold.cpu().eval()


def activation_3d_bundle(spline_dir: str, sheet_res: int, n_path: int):
    """3D-PCA coords of the activation manifold sheet, centroids, and geodesic paths.

    PCA(3) is fit on the decoded (un-standardised) sheet so the projection
    reflects the manifold's own spread, independent of the steered outputs.
    """
    import torch
    from sklearn.decomposition import PCA

    manifold, mean, std = load_activation_manifold(spline_dir)
    cp = manifold.control_points.cpu()  # (W, d) intrinsic (lat, lon)

    def decode_ambient(u):
        with torch.no_grad():
            return (manifold.decode(u) * (std + 1e-6) + mean).cpu().numpy()

    sheet_u = _u_grid(cp, sheet_res)
    sheet_amb = decode_ambient(sheet_u)
    cent_amb = decode_ambient(cp)

    pca = PCA(n_components=3).fit(sheet_amb)
    sheet_3d = pca.transform(sheet_amb)
    cent_3d = pca.transform(cent_amb)

    # Geodesic "manifold paths": connect each centroid to its nearest neighbour
    # in intrinsic (lat, lon) space -> a believable border-walk overlay.
    paths_3d = []
    W = cp.shape[0]
    d2 = ((cp.unsqueeze(0) - cp.unsqueeze(1)) ** 2).sum(-1)
    d2.fill_diagonal_(float("inf"))
    nn = d2.argmin(dim=1)
    for i in range(W):
        u_path = _geodesic(cp[i], cp[nn[i]], n_path)
        paths_3d.append(pca.transform(decode_ambient(u_path)))

    color = cp[:, 0].numpy()  # latitude -> sequential color
    return {
        "sheet": sheet_3d, "centroids": cent_3d, "paths": paths_3d,
        "color": color, "sheet_res": sheet_res,
    }


def belief_3d_bundle(spline_dir: str, hellinger_pca, natural_dists,
                     sheet_res: int, n_path: int):
    """3D coords of the belief manifold via the saved hellinger_pca projection."""
    import torch

    manifold = load_belief_manifold(spline_dir)
    cp = manifold.control_points.cpu()
    pca_dim = int(hellinger_pca.n_features_in_)

    def _fit_dim(arr: np.ndarray) -> np.ndarray:
        if arr.shape[-1] == pca_dim:
            return arr
        if arr.shape[-1] == pca_dim - 1:  # append the "other" mass column
            other = np.clip(1.0 - arr.sum(-1, keepdims=True), 0, None)
            return np.concatenate([arr, other], axis=-1)
        return arr[:, :pca_dim]

    def decode_hellinger(u):
        with torch.no_grad():
            h = manifold.decode(u).cpu().numpy()  # already in Hellinger (sqrt-p) space
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

    cloud_3d = None
    if natural_dists is not None:
        nd = np.clip(np.asarray(natural_dists, dtype=np.float32), 0, None)
        nd = _fit_dim(nd)
        cloud_3d = hellinger_pca.transform(np.sqrt(nd).astype(np.float32))

    return {
        "sheet": sheet_3d, "centroids": cent_3d, "paths": paths_3d,
        "cloud": cloud_3d, "color": cp[:, 0].numpy(), "sheet_res": sheet_res,
    }


# ---------------------------------------------------------------------------
# Isometry artifacts (MDS + scatter)  -- pure numpy / sklearn
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
# Rendering
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

    # ---- helper: a PCA cell = top (PC0/PC1) + side (PC0/PC2) stacked ----
    def pca_cell(col, bundle, row_title, cloud_key=None):
        sub = gs[col, 0].subgridspec(1, 2, wspace=0.05)
        views = [("Top view", 0, 1), ("Side view", 0, 2)]
        for k, (vname, ax_i, ax_j) in enumerate(views):
            ax = fig.add_subplot(sub[0, k])
            res = bundle["sheet_res"]
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

    # ---- helper: MDS 3D panel ----
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
                ax.text(xoff, e[:, 1].max(), e[:, 2].max(), lbl, fontsize=8,
                        color="#333")
            xoff += 2.4
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(panel_title, fontsize=10, color="#333")
        ax.view_init(elev=18, azim=-60)

    # ---- helper: scaled-isometry scatter ----
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

    # MDS panels embed the FULL isometry vertex set (V = W + interior points),
    # so color by the per-vertex latitude the isometry run saved, not the W
    # manifold centroids. Same vertex order across geometric/linear tensors.
    V = iso_geo["D_output"].shape[0]
    iso_color = (iso_geo["grid"][:, 0] if iso_geo.get("grid") is not None
                 and iso_geo["grid"].shape[0] == V else np.arange(V))

    # Row 0: Behavior space
    pca_cell(0, bel, "Behavior space", cloud_key="cloud")
    mds_panel(0, [(mds3(iso_geo["D_output"]), None, "o")], iso_color,
              "Behavior Space MDS")
    scatter_panel(0, iso_geo, "Manifold paths")

    # Row 1: Activation space
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
    ap.add_argument("--steering-dir", default=None,
                    help="path_steering '.../country' leaf (auto-discovered if omitted)")
    ap.add_argument("--out", default=None, help="output PNG path")
    ap.add_argument("--sheet-res", type=int, default=36)
    ap.add_argument("--n-path", type=int, default=24)
    ap.add_argument("--title",
                    default="Activation<->Behavior isometry  |  country_borders / Gemma3-4B-PT (L12/country)")
    args = ap.parse_args()

    paths = discover_paths(args.experiment_root, args.steering_dir)
    log.info("steering_dir       = %s", paths["steering_dir"])
    log.info("activation spline  = %s", paths["act_spline_dir"])
    log.info("belief spline      = %s", paths["bel_spline_dir"])
    log.info("hellinger_pca      = %s", paths["hellinger_pca_path"])
    log.info("isometry modes     = %s", list(paths["iso"].keys()))

    if "geometric" not in paths["iso"]:
        log.error("No geometric isometry tensors found under %s/criteria/isometry/",
                  paths["steering_dir"])
        return 2
    if not paths["act_spline_dir"] or not paths["bel_spline_dir"]:
        log.error("Could not locate both manifold_spline checkpoints.")
        return 2

    iso_geo = load_iso(paths["iso"]["geometric"]["tensors"],
                       paths["iso"]["geometric"]["metrics"])
    iso_lin = (load_iso(paths["iso"]["linear"]["tensors"],
                        paths["iso"]["linear"]["metrics"])
               if "linear" in paths["iso"] else None)

    from causalab.io.sklearn_pca import load_pca
    from safetensors.numpy import load_file as _np_load

    hpca = load_pca(os.path.dirname(paths["hellinger_pca_path"]), "hellinger_pca")
    natural = None
    if paths["natural_dists_path"]:
        try:
            natural = _np_load(paths["natural_dists_path"])["dists"]
        except Exception as e:  # noqa: BLE001
            log.warning("Could not load natural dists cloud: %s", e)

    log.info("Building activation manifold bundle ...")
    act = activation_3d_bundle(paths["act_spline_dir"], args.sheet_res, args.n_path)
    log.info("Building belief manifold bundle ...")
    bel = belief_3d_bundle(paths["bel_spline_dir"], hpca, natural,
                           args.sheet_res, args.n_path)

    out = args.out or os.path.join(paths["root"], "fig3_composite.png")
    render(act, bel, iso_geo, iso_lin, out, args.title)

    print("\n=== Fig-3 reproduction summary ===")
    print(f"  geometric (manifold paths)  r = {iso_geo['r']:.4f}")
    if iso_lin is not None:
        print(f"  linear    (chord paths)     r = {iso_lin['r']:.4f}")
    print(f"  figure: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
