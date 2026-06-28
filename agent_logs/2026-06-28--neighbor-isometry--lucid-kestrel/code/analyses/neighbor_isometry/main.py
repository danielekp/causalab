"""neighbor_isometry: per-direction stratified activation<->behavior isometry.

Tests whether Gemma3-4B-PT's neighbor-output distribution (grouped by subject
country, CONDITIONED on the query direction) varies isometrically with the
subject's L12 activation geography — and whether that isometry is stronger than
the direction-marginalized subject baseline (geometric r=0.204, linear r=0.033)
reproduced last session.

The user-selected *per-direction subject walk* is realized as 8 independent 2-D
subject-geography manifolds (one per compass direction). The 3-D joint
(lat, lon, direction) manifold is NOT config-achievable because
`train_spline_manifold` hard-excludes every variable except the single
intervention_variable (methods/spline/train.py); this analysis sidesteps that by
building control points manually and calling `build_spline_manifold` directly.

MODEL-FREE: consumes cached activation features (subspace step) and cached
per-example output distributions (output_manifold step). Never calls
load_pipeline.

This is a session-local *analysis* (research-question wrapper) — see ARCHITECTURE.md §3.
Layering rules respected:
  - depends on causalab/{methods,io,runner.helpers,tasks}, never on causalab/analyses/ peers
  - all disk I/O routes through causalab.io.* primitives (+ isometry's own saver)
  - no hyperparameter defaults inline — every knob comes from cfg.neighbor_isometry.<knob>
  - cfg.experiment_root is the single source of truth for output paths
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

from causalab.io.artifacts import (
    save_experiment_metadata,
    save_json_results,
    save_tensor_results,
)
from causalab.runner.helpers import generate_datasets, resolve_task

from causalab.methods.scores.isometry import (
    _save_isometry_artifacts,
    compute_isometry_from_manifolds,
)
from causalab.methods.spline.belief_fit import _prob_to_hellinger
from causalab.methods.spline.builders import build_spline_manifold

# Session-local analysis specific to country_borders — importing the shipped,
# read-only task config for the geographic embedding and adjacency table is fine
# (analyses may depend on causalab/tasks).
from causalab.tasks.country_borders.config import (
    COUNTRIES,
    DIRECTIONS,
    LAT_LON_OF,
    all_neighbors,
)

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "neighbor_isometry"

_ALIGN_GATE_MIN = 0.20  # min in-set argmax-vs-neighbor match rate; below => misaligned cache


# ---------------------------------------------------------------------------
# Cache discovery
# ---------------------------------------------------------------------------


def _discover_activation_features(cfg: DictConfig, analysis: DictConfig) -> str:
    """Resolve the cached per-example activation features safetensors."""
    explicit = analysis.get("activation_features")
    if explicit:
        if not os.path.exists(explicit):
            raise FileNotFoundError(f"activation_features not found: {explicit}")
        return explicit
    sub = analysis.subspace
    root = cfg.experiment_root
    patterns = [
        os.path.join(root, "subspace", sub, "**", "features", "training_features.safetensors"),
        os.path.join(root, "subspace", sub, "**", "training_features.safetensors"),
        # Fallback: features are also referenced from the activation_manifold tree.
        os.path.join(root, "activation_manifold", "**", "features", "training_features.safetensors"),
        os.path.join(root, "activation_manifold", "**", "training_features.safetensors"),
    ]
    hits: list[str] = []
    for pat in patterns:
        new = sorted(glob.glob(pat, recursive=True))
        if new:
            hits = new
            break
    if not hits:
        raise FileNotFoundError(
            f"No training_features.safetensors under {os.path.join(root, 'subspace', sub)} "
            f"or {os.path.join(root, 'activation_manifold')}. Run the subspace analysis first "
            "(or set neighbor_isometry.activation_features to the explicit path)."
        )
    if len(hits) > 1:
        logger.warning("Multiple activation feature files found; using first:\n  %s", "\n  ".join(hits))
    logger.info("Activation features: %s", hits[0])
    return hits[0]


def _discover_output_dists(cfg: DictConfig, analysis: DictConfig) -> str:
    """Resolve the cached per-example output distributions safetensors."""
    root = cfg.experiment_root
    sub = analysis.get("output_manifold_sub")
    if sub:
        cand = os.path.join(root, "output_manifold", sub, "per_example_output_dists.safetensors")
        if not os.path.exists(cand):
            raise FileNotFoundError(f"per_example_output_dists not found: {cand}")
        return cand
    hits = sorted(
        set(glob.glob(os.path.join(root, "output_manifold", "**", "per_example_output_dists.safetensors"), recursive=True))
    )
    if not hits:
        raise FileNotFoundError(
            f"No per_example_output_dists.safetensors under {os.path.join(root, 'output_manifold')}. "
            "Run the output_manifold analysis first."
        )
    if len(hits) > 1:
        logger.warning("Multiple per_example_output_dists found; using first:\n  %s", "\n  ".join(hits))
    logger.info("Output dists: %s", hits[0])
    return hits[0]


# ---------------------------------------------------------------------------
# Example-order reconstruction + alignment gate
# ---------------------------------------------------------------------------


def _example_labels(train_dataset) -> tuple[list[str], list[str]]:
    """Per-example (country, direction) from the input traces, aligned to row order."""
    countries: list[str] = []
    directions: list[str] = []
    for ex in train_dataset:
        tr = ex["input"]
        countries.append(str(tr["country"]))
        directions.append(str(tr["direction"]))
    return countries, directions


def _alignment_gate(
    dists: torch.Tensor,
    countries: list[str],
    directions: list[str],
    intervention_values: list[str],
) -> float:
    """Fraction of rows whose argmax country is a valid in-set neighbor of its cell.

    A correctly aligned cache yields a rate ≈ in-set accuracy; a shuffled/mismatched
    cache yields ≈ chance. Raises if below ``_ALIGN_GATE_MIN``.
    """
    n_country_cols = len(intervention_values)
    pred_idx = dists[:, :n_country_cols].argmax(dim=-1).tolist()
    matches = 0
    checked = 0
    for i, (c, d) in enumerate(zip(countries, directions)):
        try:
            valid = set(all_neighbors(c, d))
        except KeyError:
            continue  # cell absent from table (shouldn't happen post input_filter)
        checked += 1
        pred_country = intervention_values[pred_idx[i]]
        if pred_country in valid:
            matches += 1
    rate = matches / checked if checked else 0.0
    logger.info(
        "Alignment gate: in-set argmax-vs-neighbor match rate = %.3f over %d rows", rate, checked
    )
    if rate < _ALIGN_GATE_MIN:
        raise RuntimeError(
            f"Alignment gate FAILED: match rate {rate:.3f} < {_ALIGN_GATE_MIN}. The cached "
            "tensors do not align with the reconstructed (country,direction) order. Refusing to "
            "produce isometry scores from a misaligned cache. Check task.enumerate_all / seed match "
            "the cached run, or recover the true example ordering from a baseline manifest."
        )
    return rate


# ---------------------------------------------------------------------------
# Per-stratum manifold construction
# ---------------------------------------------------------------------------


def _stratum_subjects(countries: list[str], directions: list[str], d: str) -> list[str]:
    """Ordered (COUNTRIES order) list of subjects with at least one example in direction d."""
    present = {countries[i] for i in range(len(countries)) if directions[i] == d}
    return [c for c in COUNTRIES if c in present]


def _group_mean(rows: torch.Tensor, idx: list[int]) -> torch.Tensor:
    return rows[idx].mean(dim=0)


def _build_manifolds_for_direction(
    d: str,
    feats_std: torch.Tensor,
    dists: torch.Tensor,
    countries: list[str],
    directions: list[str],
    act_mean: torch.Tensor,
    act_std: torch.Tensor,
    smoothness: float,
):
    """Return (act_manifold, bel_manifold, subjects, control_points) for one direction."""
    subjects = _stratum_subjects(countries, directions, d)
    if len(subjects) < 4:
        return None
    control = torch.tensor([[float(LAT_LON_OF[c][0]), float(LAT_LON_OF[c][1])] for c in subjects], dtype=torch.float32)

    act_centroids = []
    bel_centroids_prob = []
    for c in subjects:
        idx = [i for i in range(len(countries)) if directions[i] == d and countries[i] == c]
        act_centroids.append(_group_mean(feats_std, idx))
        bel_centroids_prob.append(_group_mean(dists, idx))
    act_centroids = torch.stack(act_centroids, dim=0).float()  # (W_d, k) standardized
    bel_centroids_prob = torch.stack(bel_centroids_prob, dim=0).float()  # (W_d, 31) prob
    bel_centroids_hell = _prob_to_hellinger(bel_centroids_prob).float()  # (W_d, 31) sqrt-p

    act_manifold = build_spline_manifold(
        control_points=control,
        centroids=act_centroids,
        intrinsic_dim=2,
        ambient_dim=act_centroids.shape[1],
        smoothness=smoothness,
        sphere_project=False,
    )
    bel_manifold = build_spline_manifold(
        control_points=control,
        centroids=bel_centroids_hell,
        intrinsic_dim=2,
        ambient_dim=bel_centroids_hell.shape[1],
        smoothness=smoothness,
        sphere_project=True,  # belief manifold lives on the Hellinger sphere
    )
    return act_manifold, bel_manifold, subjects, control, act_centroids, bel_centroids_prob


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the neighbor_isometry analysis (model-free)."""
    analysis = cfg[ANALYSIS_NAME]
    out_dir = analysis._output_dir
    os.makedirs(out_dir, exist_ok=True)

    n_arc_steps = int(analysis.n_arc_steps)
    n_interior = int(analysis.n_interior_per_pair)
    smoothness = float(analysis.smoothness)
    path_modes = list(analysis.path_modes)
    directions = list(analysis.directions) if analysis.get("directions") else list(DIRECTIONS)

    # --- Task + deterministic dataset reconstruction (model-free) ---
    task, _ = resolve_task(
        task_name=cfg.task.name,
        task_config=OmegaConf.to_container(cfg.task, resolve=True),
        target_variable=cfg.task.get("target_variable"),
        seed=cfg.seed,
    )
    train_dataset, _ = generate_datasets(
        task,
        n_train=cfg.task.n_train,
        n_test=cfg.task.n_test,
        seed=cfg.seed,
        balanced=cfg.task.get("balanced", False),
        enumerate_all=cfg.task.enumerate_all,
        resample_variable=cfg.task.get("resample_variable", "all"),
    )
    intervention_values = [str(v) for v in task.intervention_values]
    countries, ex_directions = _example_labels(train_dataset)

    # --- Load cached tensors ---
    feat_path = _discover_activation_features(cfg, analysis)
    dist_path = _discover_output_dists(cfg, analysis)
    features = load_file(feat_path)["features"].float()  # (N, k)
    dists = load_file(dist_path)["dists"].float()  # (N, 31)

    n = len(train_dataset)
    if not (features.shape[0] == dists.shape[0] == n):
        raise RuntimeError(
            f"Row-count mismatch: train_dataset={n}, features={features.shape[0]}, "
            f"dists={dists.shape[0]}. The cache was produced from a different dataset configuration."
        )
    if dists.shape[1] != len(intervention_values) + 1:
        raise RuntimeError(
            f"Output dists have {dists.shape[1]} columns but expected "
            f"{len(intervention_values) + 1} (={len(intervention_values)} countries + other)."
        )

    # --- Alignment gate (STOP if cache order != reconstruction) ---
    align_rate = _alignment_gate(dists, countries, ex_directions, intervention_values)

    # --- Global standardization of activation features (matches baseline space) ---
    act_mean = features.mean(dim=0)
    act_std = features.std(dim=0, unbiased=False)
    feats_std = (features - act_mean) / (act_std + 1e-6)

    # --- Per-direction isometry ---
    per_direction: dict[str, dict[str, Any]] = {}
    pooled: dict[str, dict[str, list[float]]] = {pm: {"dx": [], "dy": []} for pm in path_modes}

    for d in directions:
        built = _build_manifolds_for_direction(
            d, feats_std, dists, countries, ex_directions, act_mean, act_std, smoothness
        )
        if built is None:
            logger.warning("Direction %s has <4 subjects; skipping.", d)
            continue
        act_mfd, bel_mfd, subjects, control, act_centroids, bel_prob = built
        d_out = os.path.join(out_dir, f"dir_{d}")
        os.makedirs(d_out, exist_ok=True)

        # Save the fitted manifolds for the figure script (model-free rebuild).
        save_tensor_results(
            {
                "act_control_points": control,
                "act_centroids": act_centroids,
                "act_mean": act_mean,
                "act_std": act_std,
                "bel_control_points": control,
                "bel_centroids_prob": bel_prob,
            },
            d_out,
            "manifolds.safetensors",
        )

        per_direction[d] = {"n_subjects": len(subjects), "subjects": subjects}
        for pm in path_modes:
            # Reporting call: with interior points (matches the prior run / figure scatter).
            metrics, D_X, D_Y, vco, vco_bel = compute_isometry_from_manifolds(
                act_mfd, act_mean, act_std, bel_mfd,
                n_arc_steps=n_arc_steps, path_mode=pm, n_interior_per_pair=n_interior,
            )
            iso_dir = os.path.join(d_out, "criteria", "isometry", pm)
            _save_isometry_artifacts(
                metrics, D_X, D_Y, vco, iso_dir,
                metadata={"direction": d, "path_mode": pm, "subjects": subjects},
                grid_points_belief=vco_bel,
            )
            per_direction[d][f"{pm}_r"] = metrics["pearson_r"]
            per_direction[d][f"{pm}_n_pairs"] = metrics["n_pairs"]

            # Pooling call: centroid-only (K=0) so every retained pair is a distinct
            # country pair (no same-geodesic exclusions); accumulate triu distances.
            _m0, DX0, DY0, _v, _vb = compute_isometry_from_manifolds(
                act_mfd, act_mean, act_std, bel_mfd,
                n_arc_steps=n_arc_steps, path_mode=pm, n_interior_per_pair=0,
            )
            iu = np.triu_indices_from(DX0, k=1)
            pooled[pm]["dx"].extend(DX0[iu].tolist())
            pooled[pm]["dy"].extend(DY0[iu].tolist())

        logger.info(
            "dir %s: %s",
            d,
            ", ".join(f"{pm} r={per_direction[d].get(f'{pm}_r', float('nan')):.3f}" for pm in path_modes),
        )

    # --- Aggregate ---
    summary: dict[str, Any] = {
        "alignment_match_rate": align_rate,
        "baseline": {
            "geometric_r": float(analysis.get("baseline_geometric_r", float("nan"))),
            "linear_r": float(analysis.get("baseline_linear_r", float("nan"))),
        },
        "config": {
            "n_arc_steps": n_arc_steps,
            "n_interior_per_pair": n_interior,
            "smoothness": smoothness,
            "path_modes": path_modes,
        },
        "per_direction": per_direction,
        "aggregate": {},
    }
    for pm in path_modes:
        rs = [per_direction[d][f"{pm}_r"] for d in per_direction if f"{pm}_r" in per_direction[d]]
        rs = [r for r in rs if r == r]  # drop NaN
        dx = np.asarray(pooled[pm]["dx"], dtype=np.float64)
        dy = np.asarray(pooled[pm]["dy"], dtype=np.float64)
        pooled_r = (
            float(np.corrcoef(dx, dy)[0, 1])
            if dx.size > 1 and np.std(dx) > 0 and np.std(dy) > 0
            else float("nan")
        )
        summary["aggregate"][pm] = {
            "mean_r": float(np.mean(rs)) if rs else float("nan"),
            "median_r": float(np.median(rs)) if rs else float("nan"),
            "min_r": float(np.min(rs)) if rs else float("nan"),
            "max_r": float(np.max(rs)) if rs else float("nan"),
            "n_directions": len(rs),
            "pooled_centroid_r": pooled_r,
            "pooled_n_pairs": int(dx.size),
        }

    save_json_results(summary, out_dir, "summary.json")
    save_json_results(summary, out_dir, "results.json")
    save_experiment_metadata(OmegaConf.to_container(cfg, resolve=True), out_dir)

    for pm in path_modes:
        agg = summary["aggregate"].get(pm, {})
        logger.info(
            "AGGREGATE %s: mean per-dir r=%.3f (min %.3f, max %.3f, n=%d) | pooled-centroid r=%.3f | baseline=%.3f",
            pm,
            agg.get("mean_r", float("nan")),
            agg.get("min_r", float("nan")),
            agg.get("max_r", float("nan")),
            agg.get("n_directions", 0),
            agg.get("pooled_centroid_r", float("nan")),
            summary["baseline"]["geometric_r"] if pm == "geometric" else summary["baseline"]["linear_r"],
        )
    return summary
