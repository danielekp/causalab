"""directional_isometry: does a DIRECTION-CONDITIONED behavior manifold sharpen the
activation<->behavior isometry on country_borders?

Option (b) of the manifold-steering Fig-2/3 reproduction. The shipped pipeline
(`output_manifold` -> `path_steering` isometry) builds each country's belief centroid
by AVERAGING its answer distribution over all prompts — i.e. over all 8 directions.
country_borders is RELATIONAL (the answer is a different NEIGHBOR per direction), so
that average is a direction-blurred profile, and the resulting isometry is weak
(geometric r=0.20, linear r=0.03; result/REPORT.md F7-pre).

This analysis instead gives each country a DIRECTION-CONDITIONED behavioral fingerprint:
the per-direction answer distributions, concatenated (in a fixed direction order) and
sqrt-transformed (Hellinger). It fits a belief manifold on those fingerprints — sharing
the activation manifold's (lat, lon) parameterization so the two align by class — and
recomputes the same `compute_isometry_from_manifolds` metric. If the relational blurring
was the cause of the weak isometry, the geometric r should rise.

MODEL-FREE by construction: the belief side reuses the per-example output distributions
already collected by `output_manifold` (regrouped by (subject, direction)); the
activation side reuses the cached activation manifold; the isometry metric needs no
forward passes. A lite (weights-free) pipeline is loaded only to rebuild the
InterchangeTarget needed by `load_featurizer`. This sidesteps the OOM hit when
re-running `output_manifold` on a memory-constrained pod.

CAVEAT (zero-fill): country_borders enumerates only the valid (country, direction) cells
(~166 of 30x8). Cells with no example are zero-filled in the fingerprint. A country's
fingerprint norm therefore grows with its number of valid directions; this is documented,
not corrected — it is the simplest faithful first cut. A full 30x8 model query (incl.
no-neighbor cells) would remove it but needs the model.

Session-local *analysis* (ARCHITECTURE.md §3). Layering respected:
  - depends only on causalab/{neural,methods,io,causal,tasks,runner.helpers,analyses}
  - all disk I/O via causalab.io.* primitives or the shipped isometry saver
  - no hyperparameter defaults inline — every knob from cfg.<...>
  - cfg.experiment_root is the single source of truth for output paths
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from causalab.runner.helpers import (
    resolve_task,
    generate_datasets,
    build_targets_for_grid,
)
from causalab.io.artifacts import (
    load_tensor_results,
    save_json_results,
    save_tensor_results,
    save_experiment_metadata,
)
from causalab.io.pipelines import (
    load_lite_pipeline,
    load_subspace_metadata,
    load_activation_manifold_metadata,
)
from causalab.methods.spline.manifold import SplineManifold
from causalab.methods.scores.isometry import (
    compute_isometry_from_manifolds,
    _save_isometry_artifacts,
    _plot_isometry_scatter,
)
from causalab.analyses.activation_manifold.loading import load_featurizer

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "directional_isometry"


def _directional_fingerprints(
    per_example: torch.Tensor,
    train_ds: list,
    task,
    n_values: int,
    direction_order: list[str],
) -> tuple[torch.Tensor, int, int]:
    """Per-country direction-conditioned Hellinger fingerprint.

    Groups the cached per-example output distributions by (subject country, direction),
    means within each (country, direction) cell, concatenates the cells in
    ``direction_order``, and sqrt-transforms (Hellinger). Missing cells are zero-filled.

    Returns (fingerprints (n_values, n_dir*(W+1)), n_missing_cells, n_filled_cells).
    Row order matches ``task.intervention_values`` (class-index order), so the rows
    align with the activation manifold's control points.
    """
    n_dir = len(direction_order)
    w1 = per_example.shape[1]  # W + 1 ("other" bin included)
    dir_idx = {d: i for i, d in enumerate(direction_order)}

    cell_sum = torch.zeros(n_values, n_dir, w1)
    cell_cnt = torch.zeros(n_values, n_dir)
    for i, ex in enumerate(train_ds[: per_example.shape[0]]):
        ci = task.intervention_value_index(ex)  # subject country index
        direction = ex["input"]["direction"]
        di = dir_idx.get(direction)
        if di is None:
            raise ValueError(
                f"Example direction {direction!r} not in configured direction_order "
                f"{direction_order}"
            )
        cell_sum[ci, di] += per_example[i]
        cell_cnt[ci, di] += 1

    cell_mean = cell_sum / cell_cnt.clamp(min=1).unsqueeze(-1)
    cell_mean[cell_cnt == 0] = 0.0  # zero-fill missing (country, direction) cells
    fingerprints = torch.sqrt(cell_mean.clamp(min=0)).reshape(n_values, n_dir * w1)

    n_missing = int((cell_cnt == 0).sum())
    n_filled = int((cell_cnt > 0).sum())
    return fingerprints, n_missing, n_filled


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the directional_isometry analysis.

    Artifacts land under
    ``cfg.experiment_root/directional_isometry/${._subdir}/${task.target_variable}/``.
    """
    analysis = cfg[ANALYSIS_NAME]
    root = cfg.experiment_root
    tv = cfg.task.get("target_variable")

    ss_sub = analysis.subspace
    m_sub = analysis.activation_manifold
    if ss_sub is None or m_sub is None:
        raise ValueError(
            "directional_isometry requires explicit `subspace` and "
            "`activation_manifold` (pin the country_borders fit)."
        )

    n_arc_steps = analysis.n_arc_steps
    n_interior = analysis.n_interior_per_pair
    smoothness = analysis.smoothness
    path_modes = list(OmegaConf.to_container(analysis.path_modes, resolve=True))
    direction_order = list(OmegaConf.to_container(analysis.direction_order, resolve=True))
    dist_fn = cfg.task.get("distance_function", "hellinger")
    figure_fmt = analysis.get("visualization", {}).get("figure_format", "pdf")

    out_dir = os.path.join(analysis._output_dir, tv) if tv else analysis._output_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- Task ---
    task, _ = resolve_task(
        task_name=cfg.task.name,
        task_config=OmegaConf.to_container(cfg.task, resolve=True),
        target_variable=tv,
        seed=cfg.seed,
    )
    values = task.intervention_values
    n_values = len(values)

    # --- 1. Cached per-example belief distributions (collected by output_manifold) ---
    bm_root = os.path.join(root, "output_manifold")
    per_example = load_tensor_results(bm_root, "per_example_output_dists.safetensors")[
        "dists"
    ]
    # Regenerate the row-aligned dataset exactly as output_manifold did (deterministic
    # via seed) so each distribution maps back to its (subject country, direction).
    train_ds, _ = generate_datasets(
        task,
        n_train=cfg.task.n_train,
        n_test=cfg.task.n_test,
        seed=cfg.seed,
        balanced=cfg.task.get("balanced", False),
        enumerate_all=cfg.task.enumerate_all,
        resample_variable=cfg.task.get("resample_variable", "all"),
    )
    if len(train_ds) < per_example.shape[0]:
        raise ValueError(
            f"Regenerated dataset rows ({len(train_ds)}) < cached distribution rows "
            f"({per_example.shape[0]}); cannot align belief rows. Re-run output_manifold "
            f"with the same seed/task config."
        )

    # --- 2. Direction-conditioned fingerprints (Hellinger, zero-filled) ---
    fingerprints, n_missing, n_filled = _directional_fingerprints(
        per_example, train_ds, task, n_values, direction_order
    )
    logger.info(
        "Directional fingerprints: %d countries x %d dirs, %d filled / %d missing cells, "
        "fingerprint dim=%d",
        n_values,
        len(direction_order),
        n_filled,
        n_missing,
        fingerprints.shape[1],
    )

    # --- 4. Activation manifold (lite pipeline => no model weights, no OOM) ---
    ss_meta = load_subspace_metadata(root, ss_sub, target_variable=tv)
    m_meta = load_activation_manifold_metadata(root, ss_sub, m_sub, target_variable=tv)
    layer = ss_meta.get("layer") or m_meta.get("layer")
    if layer is None:
        raise ValueError(f"No layer in metadata for {ss_sub}/{m_sub}")
    tp_name = ss_meta.get("token_position") or m_meta.get("token_position")

    pipeline = load_lite_pipeline(
        model_name=cfg.model.name,
        max_new_tokens=cfg.task.max_new_tokens,
    )
    targets, tp_list = build_targets_for_grid(
        pipeline, task, [layer], [tp_name] if tp_name else None
    )
    token_pos = tp_list[0]
    interchange_target = next(iter(targets.values()))

    manifold_out = os.path.join(root, "activation_manifold", ss_sub, m_sub)
    if tv:
        manifold_out = os.path.join(manifold_out, tv)
    featurizer = load_featurizer(manifold_out, interchange_target, layer, token_pos.id)

    has_manifold = len(featurizer.stages) >= 2 and hasattr(
        featurizer.stages[-1].featurizer, "manifold"
    )
    if not has_manifold:
        raise ValueError(
            "Loaded featurizer has no manifold stage; directional_isometry needs the "
            "activation_manifold fit (PCA >> standardize >> spline)."
        )
    act_manifold = featurizer.stages[-1].featurizer.manifold
    std_stage = featurizer.stages[-2].featurizer
    act_mean = std_stage._mean
    act_std = std_stage._std

    act_cps = act_manifold.control_points.detach().cpu()
    if act_cps.shape[0] != n_values:
        raise ValueError(
            f"Activation manifold has {act_cps.shape[0]} control points but the task "
            f"has {n_values} countries; row alignment for isometry is not guaranteed. "
            f"(Did the activation manifold drop empty-centroid classes?)"
        )

    # --- 3. Belief manifold over the directional fingerprints ---
    # Reuse the activation manifold's (lat, lon) control points verbatim so both
    # manifolds share the SAME intrinsic u-space (exact correspondence for interior
    # points; row alignment for centroid pairs). sphere_project is off: a concatenation
    # of 8 sqrt-p blocks is not a single unit-sphere point.
    belief_manifold = SplineManifold(
        control_points=act_cps,
        target_points=fingerprints.to(act_cps.dtype),  # match dtype for the TPS solve
        intrinsic_dim=act_cps.shape[1],
        ambient_dim=fingerprints.shape[1],
        smoothness=smoothness,
        spline_method="tps",
        sphere_project=False,
    )

    # --- 5. Isometry per path mode (shipped primitive; model-free) ---
    iso_results: dict[str, Any] = {}
    for pm in path_modes:
        metrics, D_X, D_Y, grid, grid_belief = compute_isometry_from_manifolds(
            activation_manifold=act_manifold,
            activation_mean=act_mean,
            activation_std=act_std,
            belief_manifold=belief_manifold,
            n_arc_steps=n_arc_steps,
            path_mode=pm,
            n_interior_per_pair=n_interior,
        )
        iso_results[pm] = metrics
        crit_dir = os.path.join(out_dir, "criteria", "isometry", pm)
        _save_isometry_artifacts(
            metrics,
            D_X,
            D_Y,
            grid,
            crit_dir,
            grid_points_belief=grid_belief,
            metadata={
                "n_arc_steps": n_arc_steps,
                "n_centroids": metrics.get("n_centroids", int(act_cps.shape[0])),
                "n_interior_per_pair": n_interior,
                "path_mode": pm,
                "behavior_space": "direction_conditioned",
            },
        )
        try:
            _plot_isometry_scatter(
                D_X,
                D_Y,
                metrics,
                os.path.join(out_dir, "vis", "isometry", pm),
                distance_function=dist_fn,
                figure_format=figure_fmt,
            )
        except Exception as e:  # viz must never sink the metric
            logger.warning("isometry scatter [%s] failed: %s", pm, e, exc_info=True)
        logger.info(
            "isometry/%s (direction-conditioned): r=%.4f over %d pairs",
            pm,
            metrics["pearson_r"],
            metrics["n_pairs"],
        )

    # --- Persist fingerprints + summary + metadata ---
    save_tensor_results(
        {"fingerprints": fingerprints, "control_points": act_cps},
        out_dir,
        "directional_fingerprints.safetensors",
    )
    summary = {
        "behavior_space": "direction_conditioned",
        "pearson_r": {pm: iso_results[pm]["pearson_r"] for pm in path_modes},
        "n_pairs": {pm: iso_results[pm]["n_pairs"] for pm in path_modes},
        "n_countries": n_values,
        "n_directions": len(direction_order),
        "direction_order": direction_order,
        "n_filled_cells": n_filled,
        "n_missing_cells": n_missing,
        "fingerprint_dim": int(fingerprints.shape[1]),
        "n_interior_per_pair": n_interior,
        "subspace": ss_sub,
        "activation_manifold": m_sub,
    }
    save_json_results(summary, out_dir, "directional_isometry_summary.json")
    save_experiment_metadata(
        {
            "analysis": ANALYSIS_NAME,
            "model": cfg.model.name,
            "task": cfg.task.name,
            "subspace": ss_sub,
            "activation_manifold": m_sub,
            "smoothness": smoothness,
            "n_arc_steps": n_arc_steps,
            "n_interior_per_pair": n_interior,
            "path_modes": path_modes,
            "direction_order": direction_order,
            "seed": cfg.seed,
        },
        out_dir,
    )

    logger.info("directional_isometry complete: %s", out_dir)
    return {"isometry": iso_results, "summary": summary, "output_dir": out_dir}
