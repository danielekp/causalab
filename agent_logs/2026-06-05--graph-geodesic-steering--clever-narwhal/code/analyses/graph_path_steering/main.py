"""graph_path_steering: does a graph-geodesic route hand prediction mass off through intermediate countries?

A stripped-down sibling of shipped `path_steering` (no criteria/isometry/belief-space
machinery — just the steering + landscape plot). For a configured endpoint pair it
builds and steers along up to three paths and records the per-step output distribution
over the answer-country tokens.

READOUT SEMANTICS (patch_parity verdict, 2026-06-12): the steered variable is the
question's SUBJECT country; the model answers with its NEIGHBOR
(raw_output = NEIGHBOR_OF[(country, direction)]). So the per-step answer-token argmax
shows the route's neighbors, not the route. The analysis therefore decodes each step's
distribution back to its most-consistent *subject*: score every candidate subject S by
the answer mass on NEIGHBOR_OF[(S, prompt_direction)], average over prompts, argmax
over S ("subject_decode_sequence"). Intermediate-coverage is computed on the decoded
subjects; the raw answer argmax is kept as "answer_argmax_sequence".

  - graph_geodesic (new): operates in PCA space using the same featurizer override as
    the shipped `linear_subspace` mode, but the path is the piecewise-linear route from
    `centroid_graph_geodesic(pca_centroids, ...)` instead of a straight line.
  - geometric (baseline): shipped `path_steering` `geometric` PathMode (geodesic in the
    intrinsic lat/lon coordinates of the fitted manifold).
  - linear (baseline): shipped `path_steering` `linear` PathMode (straight line in raw
    activation space, identity featurizer).

It reuses the shipped `collect_grid_distributions` steering executor and
`plot_saved_pair_distributions` plotter. Analysis→analysis imports of `path_steering`
mirror the existing `path_steering -> activation_manifold.loading` dependency.

This is a session-local *analysis* (research-question wrapper) — see ARCHITECTURE.md §3.
Layering rules respected by this module:
  - depends on causalab/{neural,methods,io,causal,tasks,runner.helpers,analyses}, never
    on causalab/runner/run_exp internals
  - all disk I/O routes through causalab.io.* primitives or json (invariant 3)
  - no hyperparameter defaults inline — every knob comes from `cfg.graph_path_steering.<knob>`
    or `cfg.task.<knob>` per invariants 5 and 11
  - `cfg.experiment_root` is the single source of truth for output paths (invariant 7)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

from causalab.runner.helpers import (
    resolve_task,
    build_targets_for_grid,
)
from causalab.io.pipelines import (
    load_pipeline,
    load_subspace_metadata,
    load_activation_manifold_metadata,
)
from causalab.io.counterfactuals import load_counterfactual_examples
from causalab.neural.pipeline import resolve_device
from causalab.neural.featurizer import Featurizer
from causalab.methods.metric import tokenize_variable_values
from causalab.methods.steer.collect import collect_grid_distributions
from causalab.analyses.activation_manifold.loading import load_featurizer
from causalab.analyses.path_steering.path_mode import (
    _build_geodesic_path,
    _build_linear_path_kd,
)
from causalab.analyses.path_steering.path_visualization import (
    plot_saved_pair_distributions,
)
from causalab.tasks.country_borders.config import NEIGHBOR_OF
from causalab.tasks.loader import load_task_counterfactuals

from methods.centroid_graph_geodesic import centroid_graph_geodesic

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "graph_path_steering"


def _compute_centroids(
    features: torch.Tensor,
    train_ds: list,
    task,
    n_values: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool features per intervention-value index. Returns (centroids, mask).

    Row-aligned with ``train_ds`` (the dataset saved alongside the features), so
    the i-th feature row corresponds to ``train_ds[i]`` (invariant: asserted by
    the caller before this is reached).
    """
    d = features.shape[1]
    centroids = torch.zeros(n_values, d, device=device)
    counts = torch.zeros(n_values)
    for i, ex in enumerate(train_ds[: features.shape[0]]):
        ci = task.intervention_value_index(ex)
        centroids[ci] += features[i].to(device)
        counts[ci] += 1
    mask = counts > 0
    for ci in range(n_values):
        if counts[ci] > 0:
            centroids[ci] /= counts[ci]
    return centroids, mask


def _argmax_country_sequence(
    probs: torch.Tensor,
    n_values: int,
    value_strs: list[str],
) -> list[str]:
    """Per-step top-1 ANSWER-token label, averaged over prompts.

    ``probs`` is (num_steps, n_prompts, W); the first ``n_values`` columns are the
    per-country mass and any trailing column is "other". We argmax over the country
    columns only. NOTE: this is the answer the model gives, which under the task's
    causal model is a NEIGHBOR of the steered subject — use the subject decode below
    for route coverage.
    """
    mean_probs = probs.float().mean(dim=1)  # (num_steps, W)
    country_probs = mean_probs[:, :n_values]  # drop "other"
    idx = country_probs.argmax(dim=1).tolist()
    return [value_strs[i] for i in idx]


def _build_signature_mask(
    eval_samples: list,
    value_strs: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-prompt subject→answer signature mask from the task's neighbor table.

    Returns (M, valid): M is (n_prompts, n_subjects, n_answers) with
    M[j, S, a] = 1 iff answer-country ``a`` is an in-set neighbor of subject S in
    prompt j's direction; valid is (n_prompts, n_subjects), False where the
    (S, direction) cell has no in-set neighbor (excluded from the prompt average).
    """
    n_values = len(value_strs)
    idx_of = {v: i for i, v in enumerate(value_strs)}
    M = torch.zeros(len(eval_samples), n_values, n_values)
    for j, s in enumerate(eval_samples):
        direction = s["input"]["direction"]
        for S, name in enumerate(value_strs):
            for nb in NEIGHBOR_OF.get((name, direction), []):
                if nb in idx_of:
                    M[j, S, idx_of[nb]] = 1.0
    return M, M.sum(dim=-1) > 0


def _subject_signature_decode(
    probs: torch.Tensor,
    sig_mask: torch.Tensor,
    sig_valid: torch.Tensor,
    n_values: int,
    value_strs: list[str],
) -> tuple[list[str], torch.Tensor]:
    """Decode each step's answer distribution to its most-consistent subject.

    signature[t, S] = mean over prompts j (with a non-empty (S, direction_j) cell) of
    the answer mass on NEIGHBOR_OF[(S, direction_j)]. Returns (per-step argmax subject
    labels, the full (num_steps, n_subjects) signature matrix).
    """
    ans = probs.float()[:, :, :n_values]  # (T, P, A) — drop "other"
    sig = torch.einsum("tpa,psa->tps", ans, sig_mask)  # (T, P, S)
    counts = sig_valid.float().sum(dim=0).clamp(min=1)  # (S,)
    mean_sig = (sig * sig_valid.float().unsqueeze(0)).sum(dim=1) / counts  # (T, S)
    seq = [value_strs[i] for i in mean_sig.argmax(dim=1).tolist()]
    return seq, mean_sig


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the graph_path_steering analysis.

    Artifacts land under
    ``cfg.experiment_root/graph_path_steering/${._subdir}/${task.target_variable}/``.
    """
    analysis = cfg[ANALYSIS_NAME]
    figure_fmt = analysis.get("visualization", {}).get("figure_format", "png")
    root = cfg.experiment_root
    tv = cfg.task.get("target_variable")

    ss_sub = analysis.subspace
    m_sub = analysis.activation_manifold
    if ss_sub is None or m_sub is None:
        raise ValueError(
            "graph_path_steering requires explicit `subspace` and "
            "`activation_manifold` (pin the country_borders fit)."
        )

    out_dir = os.path.join(analysis._output_dir, tv) if tv else analysis._output_dir
    os.makedirs(out_dir, exist_ok=True)

    # --- Task + model ---
    task, _ = resolve_task(
        task_name=cfg.task.name,
        task_config=OmegaConf.to_container(cfg.task, resolve=True),
        target_variable=tv,
        seed=cfg.seed,
    )
    pipeline = load_pipeline(
        model_name=cfg.model.name,
        task=task,
        max_new_tokens=cfg.task.max_new_tokens,
        device=cfg.model.device,
        dtype=cfg.model.get("dtype"),
        eager_attn=cfg.model.get("eager_attn"),
        use_chat_template=cfg.model.get("use_chat_template", False),
        model_class=cfg.model.get("model_class"),
    )

    device = torch.device(resolve_device(cfg.model.device))

    # --- Interchange target at the localized (layer, token_position) ---
    ss_meta = load_subspace_metadata(root, ss_sub, target_variable=tv)
    layer = ss_meta.get("layer")
    m_meta = load_activation_manifold_metadata(root, ss_sub, m_sub, target_variable=tv)
    layer = layer or m_meta.get("layer")
    if layer is None:
        raise ValueError(f"No layer in metadata for {ss_sub}/{m_sub}")
    tp_name = ss_meta.get("token_position") or m_meta.get("token_position")
    targets, tp_list = build_targets_for_grid(
        pipeline, task, [layer], [tp_name] if tp_name else None
    )
    token_pos = tp_list[0]
    interchange_target = next(iter(targets.values()))

    # --- Manifold featurizer (full PCA->standardize->spline pipeline) ---
    manifold_out = os.path.join(root, "activation_manifold", ss_sub, m_sub)
    if tv:
        manifold_out = os.path.join(manifold_out, tv)
    featurizer = load_featurizer(manifold_out, interchange_target, layer, token_pos.id)

    # --- Load saved features + row-aligned dataset for centroids ---
    subspace_out_dir = os.path.join(root, "subspace", ss_sub)
    if tv:
        subspace_out_dir = os.path.join(subspace_out_dir, tv)
    feat_dir = os.path.join(subspace_out_dir, "features")
    pca_features = load_file(os.path.join(feat_dir, "training_features.safetensors"))[
        "features"
    ]
    raw_path = os.path.join(feat_dir, "raw_features.safetensors")
    have_raw = os.path.exists(raw_path)
    raw_features = load_file(raw_path)["features"] if have_raw else None

    saved_ds_path = os.path.join(subspace_out_dir, "train_dataset.json")
    train_ds = load_counterfactual_examples(saved_ds_path, task.causal_model)
    # Row-alignment guard (the centroid grouping is meaningless otherwise).
    if len(train_ds) < pca_features.shape[0]:
        raise ValueError(
            f"train_dataset rows ({len(train_ds)}) < feature rows "
            f"({pca_features.shape[0]}); cannot row-align centroids."
        )

    values = task.intervention_values
    n_values = len(values)
    value_strs = [str(v) for v in values]

    pca_centroids, mask = _compute_centroids(
        pca_features, train_ds, task, n_values, device
    )
    raw_centroids = None
    if have_raw:
        raw_centroids, _ = _compute_centroids(
            raw_features, train_ds, task, n_values, device
        )

    # --- Intrinsic (spline) centroids + manifold object ---
    has_manifold = len(featurizer.stages) >= 2 and hasattr(
        featurizer.stages[-1].featurizer, "manifold"
    )
    if not has_manifold:
        raise ValueError(
            "Loaded featurizer has no manifold stage; graph_path_steering needs the "
            "activation_manifold fit (geometric mode + intrinsic centroids)."
        )
    manifold_obj = featurizer.stages[-1].featurizer.manifold.to(device)
    std_stage = featurizer.stages[-2].featurizer
    feat_mean = std_stage._mean.to(device)
    feat_std = std_stage._std.to(device)
    standardized = (pca_centroids - feat_mean) / (feat_std + 1e-6)
    spline_centroids, _ = manifold_obj.encode(standardized)

    # --- Output-token columns for reading the per-step distribution ---
    var_indices = tokenize_variable_values(
        pipeline.tokenizer, values, task.result_token_pattern
    )

    # --- Steering knobs ---
    num_steps = analysis.num_steps_along_path
    n_eval = analysis.n_eval_samples
    n_prompts = analysis.n_prompts
    batch_size = analysis.batch_size
    graph_k = analysis.graph_k
    steps_per_segment = analysis.steps_per_segment

    # --- Eval prompts (counterfactual samples, as in path_steering) ---
    cf_mod = load_task_counterfactuals(task.name)
    filtered_samples = cf_mod.generate_dataset(
        task.causal_model, n_eval, cfg.seed + 100
    )
    eval_samples = filtered_samples[: min(n_prompts, len(filtered_samples))]

    modes_cfg = list(OmegaConf.to_container(analysis.path_modes, resolve=True))

    # --- Subject-signature decode mask (corrected readout; see module docstring) ---
    sig_mask, sig_valid = _build_signature_mask(eval_samples, value_strs)

    # --- Endpoint pair(s) ---
    selected_pairs = [list(p) for p in analysis.selected_pairs]

    coverage: dict[str, dict] = {}
    route_record: dict[str, Any] = {}

    for pair in selected_pairs:
        start_label, end_label = pair[0], pair[1]
        if start_label not in value_strs or end_label not in value_strs:
            raise ValueError(f"pair {pair} not in task values")
        si = value_strs.index(start_label)
        ei = value_strs.index(end_label)
        if not mask[si] or not mask[ei]:
            raise ValueError(f"missing centroid for pair {pair}")

        for mode in modes_cfg:
            # Build the path + pick the featurizer for this mode.
            if mode == "graph_geodesic":
                if raw_centroids is None:
                    logger.warning(
                        "Skipping graph_geodesic: no raw_features.safetensors"
                    )
                    continue
                # The ROUTE is selected by the PCA-space graph (denoised neighbor
                # structure), but STEERING happens in raw activation space — the same
                # mechanism as the `linear` baseline. So graph_geodesic and linear
                # differ only by the route (straight vs. through intermediates), which
                # is exactly the variable we want to isolate. (patch_parity confirmed
                # the raw-space executor is exact; the earlier "inverse-PCA lift was a
                # no-op" rationale was wrong — the inverse featurizer re-adds the
                # off-subspace error term.)
                valid_idx = [i for i in range(n_values) if mask[i]]
                valid_labels = [value_strs[i] for i in valid_idx]
                valid_centroids = pca_centroids[valid_idx]
                route = centroid_graph_geodesic(
                    valid_centroids,
                    valid_labels,
                    start_label,
                    end_label,
                    k=graph_k,
                    steps_per_segment=steps_per_segment,
                )
                # Build the piecewise-linear path between the route countries' RAW
                # centroids (drop duplicate segment boundaries).
                route_global = [value_strs.index(lbl) for lbl in route["route_labels"]]
                segs: list[torch.Tensor] = []
                for a, b in zip(route_global[:-1], route_global[1:]):
                    alphas = torch.linspace(
                        0.0, 1.0, steps_per_segment,
                        device=device, dtype=raw_centroids.dtype,
                    )
                    seg = raw_centroids[a].unsqueeze(0) + alphas.unsqueeze(1) * (
                        raw_centroids[b] - raw_centroids[a]
                    ).unsqueeze(0)
                    segs.append(seg if not segs else seg[1:])
                grid_points = torch.cat(segs, dim=0)
                override = Featurizer(id="identity")  # steer in raw space, like `linear`
                route_record[f"{start_label}_{end_label}"] = {
                    "pair": [start_label, end_label],
                    "route_labels": route["route_labels"],
                    "segment_lengths": route["segment_lengths"],
                    "k_used": route["k_used"],
                }
            elif mode == "geometric":
                grid_points = _build_geodesic_path(
                    spline_centroids[si], spline_centroids[ei], num_steps, manifold_obj
                )
                override = None  # full manifold featurizer
            elif mode == "linear":
                if raw_centroids is None:
                    logger.warning("Skipping linear mode: no raw_features.safetensors")
                    continue
                grid_points = _build_linear_path_kd(
                    raw_centroids[si], raw_centroids[ei], num_steps
                )
                override = Featurizer(id="identity")
            else:
                raise ValueError(f"Unknown path mode: {mode!r}")

            # Apply the mode's featurizer to the interchange target.
            for u in interchange_target.flatten():
                u.set_featurizer(override if override is not None else featurizer)
            try:
                probs = collect_grid_distributions(
                    pipeline=pipeline,
                    grid_points=grid_points,
                    interchange_target=interchange_target,
                    filtered_samples=eval_samples,
                    var_indices=var_indices,
                    batch_size=batch_size,
                    n_base_samples=len(eval_samples),
                    average=False,
                    full_vocab_softmax=True,
                )  # (num_steps, n_prompts, W)
            finally:
                for u in interchange_target.flatten():
                    u.set_featurizer(featurizer)

            # Per-mode coverage. Route coverage is computed on the DECODED SUBJECT
            # sequence (which subject is most consistent with the answers), not on
            # the raw answer argmax — see module docstring.
            ans_seq = _argmax_country_sequence(probs, n_values, value_strs)
            subj_seq, sig_matrix = _subject_signature_decode(
                probs, sig_mask, sig_valid, n_values, value_strs
            )
            intermediates = [
                c for c in dict.fromkeys(subj_seq) if c not in (start_label, end_label)
            ]
            coverage.setdefault(f"{start_label}_{end_label}", {})[mode] = {
                "subject_decode_sequence": subj_seq,
                "n_intermediate_subjects": len(intermediates),
                "intermediate_subjects": intermediates,
                "answer_argmax_sequence": ans_seq,
                "subject_signature_matrix": [
                    [round(x, 6) for x in row] for row in sig_matrix.tolist()
                ],
            }

            # Landscape plot for this mode/pair.
            pair_distributions = probs.unsqueeze(0)  # (1, num_steps, n_prompts, W)
            plot_saved_pair_distributions(
                pair_distributions=pair_distributions,
                pairs=[(si, ei)],
                value_labels=value_strs,
                output_dir=os.path.join(out_dir, "vis", "paths", mode),
                path_mode_label=mode,
                score_labels=value_strs,
                colormap=cfg.task.get("colormap", "rainbow"),
                output_token_values=task.output_token_values or task.intervention_values,
                color_by_dim=cfg.task.get("color_by_dim", 0),
                figure_format=figure_fmt,
            )
            logger.info(
                "%s [%s->%s]: subjects %s | %d intermediates",
                mode,
                start_label,
                end_label,
                " > ".join(dict.fromkeys(subj_seq)),
                coverage[f"{start_label}_{end_label}"][mode][
                    "n_intermediate_subjects"
                ],
            )

    # --- Persist route + coverage + metadata (json; no causalab.io tensor outputs) ---
    with open(os.path.join(out_dir, "route.json"), "w") as f:
        json.dump(route_record, f, indent=2)
    with open(os.path.join(out_dir, "intermediate_coverage.json"), "w") as f:
        json.dump(coverage, f, indent=2)
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(
            {
                "analysis": ANALYSIS_NAME,
                "subspace": ss_sub,
                "activation_manifold": m_sub,
                "model": cfg.model.name,
                "task": cfg.task.name,
                "path_modes": modes_cfg,
                "selected_pairs": selected_pairs,
                "graph_k": graph_k,
                "steps_per_segment": steps_per_segment,
                "num_steps_along_path": num_steps,
                "n_eval_samples": n_eval,
                "n_prompts": n_prompts,
                "seed": cfg.seed,
            },
            f,
            indent=2,
        )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("graph_path_steering complete: %s", out_dir)
    return {"route": route_record, "coverage": coverage, "output_dir": out_dir}
