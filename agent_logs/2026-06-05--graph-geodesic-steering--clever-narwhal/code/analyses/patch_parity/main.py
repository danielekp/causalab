"""patch_parity: is the steering executor causally equivalent to locate's centroid patch?

Diagnostic A/B for the flat country_borders steering landscapes (see the session's
result/steering_review.md, experiment 1). At one (layer, token_position) site, the same
raw-space country centroid is patched into the same prompts through four arms:

  - interchange_source:      interchange intervention + ``source_representations`` —
                             the mechanism behind locate's centroid mode
                             (``run_centroid_layer_scan``), which scored 0.41 here.
  - interpolation_replica:   exact replica of ``collect_grid_distributions``'s internal
                             ``replace_fn`` run through ``run_interpolation_interventions``
                             — the steering executor under suspicion. Logs f_base/f_src/
                             output shapes (the (B,1,H)-vs-(B,H) rank probe).
  - interpolation_shapefix:  same executor, but the replacement is expanded to f_base's
                             exact shape — isolates the rank-mismatch hypothesis.
  - no_patch:                plain generation — the no-effect floor.

Verdict key: replica ≈ no_patch while interchange_source is well above it → executor
bug (shapefix succeeding pinpoints the rank mismatch). replica ≈ interchange_source →
the flat landscapes were a full-vocab readout artifact.

Run-1 verdict (2026-06-12): all three patch arms bit-identical, f_base already (B,H) —
executor exonerated. The patch visibly sets the question's SUBJECT country (Spain patch
→ Portugal/France answers), so p(target-country token) is the wrong readout for this
task: the causal model maps country → NEIGHBOR_OF[(country, direction)]. The readout
below therefore also scores expected-answer (neighbor) mass per prompt — the metric
that locate's string_match scoring implicitly used, which is why locate saw 0.41 while
every p(target)-based landscape sat at chance.

Layering (ARCHITECTURE.md §3): depends on causalab/{neural,methods,io,tasks,runner.helpers}
only; disk I/O through causalab.io.artifacts; every knob from cfg.patch_parity.* /
cfg.task.* / cfg.seed; cfg.experiment_root is the single output-path root.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable

import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

from causalab.io.artifacts import save_experiment_metadata, save_json_results
from causalab.io.counterfactuals import load_counterfactual_examples
from causalab.io.pipelines import load_pipeline, load_subspace_metadata
from causalab.methods.metric import scores_to_joint_probs, tokenize_variable_values
from causalab.neural.activations.interchange_mode import prepare_intervenable_inputs
from causalab.neural.activations.intervenable_model import (
    delete_intervenable_model,
    device_for_layer,
    prepare_intervenable_model,
)
from causalab.neural.activations.interpolate import run_interpolation_interventions
from causalab.runner.helpers import build_targets_for_grid, resolve_task
from causalab.tasks.country_borders.config import NEIGHBOR_OF
from causalab.tasks.loader import load_task_counterfactuals

logger = logging.getLogger(__name__)

ANALYSIS_NAME = "patch_parity"

ARM_INTERCHANGE = "interchange_source"
ARM_REPLICA = "interpolation_replica"
ARM_SHAPEFIX = "interpolation_shapefix"
ARM_NO_PATCH = "no_patch"
KNOWN_ARMS = (ARM_INTERCHANGE, ARM_REPLICA, ARM_SHAPEFIX, ARM_NO_PATCH)


def _compute_centroids(
    features: torch.Tensor,
    train_ds: list,
    task,
    n_values: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean-pool features per intervention-value index. Returns (centroids, mask).

    Same construction as the session's graph_path_steering (duplicated to keep this
    diagnostic standalone): row i of ``features`` corresponds to ``train_ds[i]``.
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


def _make_replica_fn(
    centroid: torch.Tensor, shape_probe: dict[str, Any]
) -> Callable[..., torch.Tensor]:
    """Exact replica of collect_grid_distributions' replace_fn (collect.py:241-254),
    plus one-shot shape logging. No dtype cast, ``unsqueeze(0).expand(B, -1)`` rank
    behavior preserved verbatim — that behavior is the thing under test."""

    def replace_fn(
        f_base: torch.Tensor,
        f_src: torch.Tensor,
        target: torch.Tensor = centroid,
        **_kwargs: Any,
    ) -> torch.Tensor:
        target = target.to(f_base.device)
        if "f_base_shape" not in shape_probe:
            shape_probe["f_base_shape"] = list(f_base.shape)
            shape_probe["f_src_shape"] = list(f_src.shape)
            shape_probe["target_shape"] = list(target.shape)
        B, k_full = f_base.shape[0], f_base.shape[-1]
        k_t = target.shape[-1]
        if k_t < k_full:
            opt = target.unsqueeze(0).expand(B, -1)
            out = torch.cat([opt, f_base[:, k_t:]], dim=-1)
        else:
            out = target.unsqueeze(0).expand(B, -1)
        if "replica_out_shape" not in shape_probe:
            shape_probe["replica_out_shape"] = list(out.shape)
        return out

    return replace_fn


def _make_shapefix_fn(
    centroid: torch.Tensor, shape_probe: dict[str, Any]
) -> Callable[..., torch.Tensor]:
    """Replacement expanded to f_base's exact shape (and dtype) — the candidate fix."""

    def replace_fn(
        f_base: torch.Tensor,
        f_src: torch.Tensor,
        target: torch.Tensor = centroid,
        **_kwargs: Any,
    ) -> torch.Tensor:
        target = target.to(f_base.device, dtype=f_base.dtype)
        if target.shape[-1] != f_base.shape[-1]:
            raise ValueError(
                f"shapefix arm expects a full-rank target: target k={target.shape[-1]} "
                f"vs f_base k={f_base.shape[-1]}"
            )
        out = target.view(*([1] * (f_base.dim() - 1)), -1).expand_as(f_base)
        if "shapefix_out_shape" not in shape_probe:
            shape_probe["shapefix_out_shape"] = list(out.shape)
        return out

    return replace_fn


def main(cfg: DictConfig) -> dict[str, Any]:
    """Run the patch_parity diagnostic.

    Artifacts land under
    ``cfg.experiment_root/patch_parity/${._subdir}/${task.target_variable}/``.
    """
    analysis = cfg[ANALYSIS_NAME]
    root = cfg.experiment_root
    tv = cfg.task.get("target_variable")

    arms = list(OmegaConf.to_container(analysis.arms, resolve=True))
    unknown = [a for a in arms if a not in KNOWN_ARMS]
    if unknown:
        raise ValueError(f"Unknown arm(s) {unknown}; known: {list(KNOWN_ARMS)}")

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

    # --- Site from subspace metadata; identity featurizers throughout ---
    ss_sub = analysis.subspace
    ss_meta = load_subspace_metadata(root, ss_sub, target_variable=tv)
    layer = ss_meta.get("layer")
    if layer is None:
        raise ValueError(f"No layer in subspace metadata for {ss_sub}")
    tp_name = ss_meta.get("token_position")
    targets, tp_list = build_targets_for_grid(
        pipeline, task, [layer], [tp_name] if tp_name else None
    )
    interchange_target = next(iter(targets.values()))
    device = device_for_layer(pipeline, layer)
    site = {"layer": int(layer), "token_position": str(tp_list[0].id)}
    logger.info("patch_parity site: %s", site)

    # --- Raw centroids from the subspace step's saved features ---
    subspace_out_dir = os.path.join(root, "subspace", ss_sub)
    if tv:
        subspace_out_dir = os.path.join(subspace_out_dir, tv)
    raw_features = load_file(
        os.path.join(subspace_out_dir, "features", "raw_features.safetensors")
    )["features"]
    train_ds = load_counterfactual_examples(
        os.path.join(subspace_out_dir, "train_dataset.json"), task.causal_model
    )
    if len(train_ds) < raw_features.shape[0]:
        raise ValueError(
            f"train_dataset rows ({len(train_ds)}) < feature rows "
            f"({raw_features.shape[0]}); cannot row-align centroids."
        )

    values = task.intervention_values
    n_values = len(values)
    value_strs = [str(v) for v in values]
    raw_centroids, mask = _compute_centroids(
        raw_features, train_ds, task, n_values, device
    )
    var_indices = tokenize_variable_values(
        pipeline.tokenizer, values, task.result_token_pattern
    )

    # --- Shared eval prompts (same construction/seed offset as graph_path_steering) ---
    cf_mod = load_task_counterfactuals(task.name)
    filtered_samples = cf_mod.generate_dataset(
        task.causal_model, analysis.n_eval_samples, cfg.seed + 100
    )
    eval_samples = filtered_samples[: min(analysis.n_prompts, len(filtered_samples))]
    batch_size = analysis.batch_size

    # Arm A uses self-pairs (as run_centroid_layer_scan does). Arms B/B2 default to
    # production behavior (samples keep their real counterfactual_inputs, exactly as
    # collect_grid_distributions receives them); interpolation_self_pairs=true switches
    # them to self-pairs to rule out the source-prompt confound.
    self_pairs = [
        {"input": s["input"], "counterfactual_inputs": [s["input"]]}
        for s in eval_samples
    ]
    if analysis.interpolation_self_pairs:
        interp_examples = self_pairs
    else:
        interp_examples = [
            {"input": s["input"], "counterfactual_inputs": [s["input"]]}
            if "counterfactual_inputs" not in s
            else s
            for s in eval_samples
        ]

    # --- Arm executors (each returns scores as a list of per-batch step lists) ---

    def run_interchange_source(centroid: torch.Tensor) -> list:
        # Mirror of run_centroid_layer_scan (layer_scan.py:434-485).
        im = prepare_intervenable_model(pipeline, interchange_target)
        all_scores = []
        try:
            for start in range(0, len(self_pairs), batch_size):
                batch = self_pairs[start : start + batch_size]
                batched_base, _, inv_locations, feature_indices = (
                    prepare_intervenable_inputs(pipeline, batch, interchange_target)
                )
                src = centroid.view(1, 1, -1).expand(len(batch), 1, -1).to(device)
                out = pipeline.intervenable_generate(
                    im,
                    batched_base,
                    None,
                    inv_locations,
                    feature_indices,
                    source_representations=[src],
                    output_scores=True,
                )
                all_scores.append(out["scores"])
        finally:
            delete_intervenable_model(im)
        return all_scores

    def run_interpolation(fn: Callable[..., torch.Tensor]) -> list:
        # Mirror of collect_grid_distributions (collect.py:256-273), single grid point.
        results = run_interpolation_interventions(
            pipeline=pipeline,
            counterfactual_dataset=interp_examples,
            interchange_target=interchange_target,
            fn=fn,
            params={},
            batch_size=batch_size,
            output_scores=True,
        )
        # Same featurizer-device restore as collect_grid_distributions.
        for group in interchange_target:
            for unit in group:
                unit_device = device_for_layer(pipeline, unit.layer)
                unit.featurizer.featurizer.to(unit_device)
                unit.featurizer.inverse_featurizer.to(unit_device)
        return results["scores"]

    def run_no_patch() -> list:
        all_scores = []
        for start in range(0, len(eval_samples), batch_size):
            batch = [s["input"] for s in eval_samples[start : start + batch_size]]
            out = pipeline.generate(batch, output_scores=True)
            all_scores.append(out["scores"])
        return all_scores

    def expected_index_sets(country: str) -> list[list[int]]:
        """Per-prompt CORRECT readout targets when `tv` is patched to `country`.

        The causal model is raw_output = NEIGHBOR_OF[(country, direction)], so the
        expected answer set for prompt j is the in-set neighbors of the *patched*
        country in prompt j's direction — not the patched country itself. Empty list
        = unanswerable cell (no in-set neighbor); excluded from expected-* metrics.
        """
        sets: list[list[int]] = []
        for s in eval_samples:
            direction = s["input"]["direction"]
            names = NEIGHBOR_OF.get((country, direction), [])
            sets.append([value_strs.index(n) for n in names if n in value_strs])
        return sets

    def readout(
        raw_scores: list, target_idx: int, expected_sets: list[list[int]]
    ) -> dict[str, Any]:
        joint_norm = scores_to_joint_probs(raw_scores, var_indices)
        joint_fvs = scores_to_joint_probs(raw_scores, var_indices, full_vocab_softmax=True)
        if joint_norm is None or joint_fvs is None:
            raise ValueError("Arm returned no scores; cannot compute readout.")
        joint_norm = joint_norm.float()
        joint_norm = joint_norm / joint_norm.sum(-1, keepdim=True).clamp(min=1e-10)
        if joint_norm.shape[0] != len(expected_sets):
            raise ValueError(
                f"score rows ({joint_norm.shape[0]}) != eval prompts "
                f"({len(expected_sets)}); prompt alignment broken."
            )
        mean_norm = joint_norm.mean(dim=0)
        mean_fvs = joint_fvs.float().mean(dim=0)
        top = torch.topk(mean_norm, k=min(5, mean_norm.shape[0]))
        argmaxes = joint_norm.argmax(dim=-1)
        p_exp, argmax_exp, p_canon = [], [], []
        for j, idxs in enumerate(expected_sets):
            if not idxs:
                continue
            p_exp.append(joint_norm[j, idxs].sum().item())
            argmax_exp.append(float(argmaxes[j].item() in idxs))
            p_canon.append(joint_norm[j, idxs[0]].item())
        n_valid = len(p_exp)
        return {
            "p_target_norm": mean_norm[target_idx].item(),
            "p_target_fullvocab": mean_fvs[target_idx].item(),
            "argmax_match_rate": (argmaxes == target_idx).float().mean().item(),
            # Correct readout: mass/argmax on the *expected answer* (neighbor) set
            # for the patched country, per prompt direction.
            "p_expected_norm": sum(p_exp) / n_valid if n_valid else None,
            "p_canonical_norm": sum(p_canon) / n_valid if n_valid else None,
            "argmax_expected_rate": sum(argmax_exp) / n_valid if n_valid else None,
            "n_prompts_with_expected": n_valid,
            "mean_norm_dist": {
                value_strs[i]: round(mean_norm[i].item(), 6) for i in range(n_values)
            },
            "top5": [
                [value_strs[i], round(mean_norm[i].item(), 6)]
                for i in top.indices.tolist()
            ],
        }

    # --- Run all (country, arm) cells ---
    shape_probe: dict[str, Any] = {}
    per_country: dict[str, dict[str, Any]] = {}
    no_patch_scores = run_no_patch() if ARM_NO_PATCH in arms else None

    for country in list(analysis.countries):
        if country not in value_strs:
            raise ValueError(f"country {country!r} not in task values")
        ci = value_strs.index(country)
        if not mask[ci]:
            raise ValueError(f"no centroid for {country!r} (no training rows)")
        centroid = raw_centroids[ci]
        exp_sets = expected_index_sets(country)

        per_country[country] = {}
        for arm in arms:
            if arm == ARM_INTERCHANGE:
                scores = run_interchange_source(centroid)
            elif arm == ARM_REPLICA:
                scores = run_interpolation(_make_replica_fn(centroid, shape_probe))
            elif arm == ARM_SHAPEFIX:
                scores = run_interpolation(_make_shapefix_fn(centroid, shape_probe))
            else:  # ARM_NO_PATCH — shared across countries
                scores = no_patch_scores
            per_country[country][arm] = readout(scores, ci, exp_sets)
            logger.info(
                "%s / %s: p_expected_norm=%s argmax_expected=%s p_target_norm=%.4f",
                country,
                arm,
                per_country[country][arm]["p_expected_norm"],
                per_country[country][arm]["argmax_expected_rate"],
                per_country[country][arm]["p_target_norm"],
            )

    results: dict[str, Any] = {
        "site": site,
        "value_labels": value_strs,
        "shape_probe": shape_probe,
        "per_country": per_country,
    }
    save_json_results(results, out_dir, "results.json")
    save_experiment_metadata(
        {
            "analysis": ANALYSIS_NAME,
            "subspace": ss_sub,
            "model": cfg.model.name,
            "task": cfg.task.name,
            "target_variable": tv,
            "site": site,
            "arms": arms,
            "countries": list(analysis.countries),
            "n_eval_samples": analysis.n_eval_samples,
            "n_prompts": len(eval_samples),
            "batch_size": batch_size,
            "interpolation_self_pairs": bool(analysis.interpolation_self_pairs),
            "seed": cfg.seed,
        },
        out_dir,
    )

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("patch_parity complete: %s", out_dir)
    return {**results, "output_dir": out_dir}
