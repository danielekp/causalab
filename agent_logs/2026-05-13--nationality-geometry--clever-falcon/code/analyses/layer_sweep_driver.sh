#!/usr/bin/env bash
# Layer-sweep control for #2 (the decisive experiment, REPORT §10).
#
# Runs one single-cell subspace per (layer, token_position) into its own
# experiment-root so nothing clobbers anything and the sweep is RESUMABLE
# (a cell whose features already exist is skipped). Run from the repo root
# on the GPU box (RunPod):
#
#   bash agent_logs/2026-05-13--nationality-geometry--clever-falcon/code/analyses/layer_sweep_driver.sh
#
# Then the post-hoc curve (CPU):
#
#   uv run python agent_logs/2026-05-13--nationality-geometry--clever-falcon/code/analyses/layer_sweep.py
#
# Trim LAYERS to spend less GPU; 8 layers x 2 positions = 16 cells, each a
# model forward over the dataset (~minutes on an H100).
set -euo pipefail

SESSION="agent_logs/2026-05-13--nationality-geometry--clever-falcon"
SWEEP_ROOT="${SESSION}/artifacts/country_borders/llama31_8b/_sweep"

LAYERS=(4 8 12 16 20 24 28 31)
POSITIONS=(last_token country)   # relational (answer) vs direct (entity)

total=$(( ${#LAYERS[@]} * ${#POSITIONS[@]} ))
i=0
for pos in "${POSITIONS[@]}"; do
  for L in "${LAYERS[@]}"; do
    i=$((i + 1))
    cell_root="${SWEEP_ROOT}/L${L}_${pos}"
    feat="${cell_root}/subspace/pca_k32/country/features/training_features.safetensors"
    if [ -f "${feat}" ]; then
      echo "[${i}/${total}] skip  L${L} ${pos} (already done)"
      continue
    fi
    echo "[${i}/${total}] run   L${L} ${pos} -> ${cell_root}"
    bash scripts/run_exp.sh \
      --experiment-root "${cell_root}" \
      country_borders_subspace_cell \
      "subspace.layers=[${L}]" \
      "subspace.token_positions=[${pos}]"
  done
done
echo "sweep complete: ${total} cells under ${SWEEP_ROOT}"
