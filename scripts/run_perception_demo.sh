#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${EFFICIENT_ROBOTICS_VLM_PYTHON:-python3}"
model_path="${EFFICIENT_ROBOTICS_QWEN_MODEL:-${project_root}/models/Qwen3-VL-8B-Instruct}"

if [[ $# -ne 1 ]]; then
    echo "Usage: PHYSICAL_GPU=N $0 INPUT_JSON" >&2
    exit 2
fi
if [[ -z "${PHYSICAL_GPU:-}" || ! "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]]; then
    echo "Set PHYSICAL_GPU to one physical GPU index." >&2
    exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export NVIDIA_VISIBLE_DEVICES="${PHYSICAL_GPU}"

exec "${python_bin}" \
    "${project_root}/scripts/qwen3_vl_logits.py" \
    "$1" \
    --model-path "${model_path}"
