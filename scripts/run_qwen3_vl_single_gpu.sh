#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vlm_venv="${EFFICIENT_ROBOTICS_VLM_VENV:-${project_root}/.venv-vlm}"
model_path="${EFFICIENT_ROBOTICS_QWEN_MODEL:-${project_root}/models/Qwen3-VL-8B-Instruct}"
vlm_python="${vlm_venv}/bin/python"

if [[ ! -x "${vlm_python}" ]]; then
    echo "VLM Python is not executable: ${vlm_python}" >&2
    exit 1
fi
if [[ $# -lt 1 ]]; then
    echo "Usage: $0 INPUT_JSON [adapter arguments...]" >&2
    exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
physical_gpu="${PHYSICAL_GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
export CUDA_VISIBLE_DEVICES="${physical_gpu}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${project_root}/.cache/huggingface}"
export HF_HUB_DISABLE_TELEMETRY=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "PHYSICAL_GPU=${physical_gpu}"
echo "TORCH_DEVICE=cuda:0"
echo "MULTI_GPU=false"
echo "MODEL_PATH=${model_path}"

exec "${vlm_python}" \
    "${project_root}/scripts/qwen3_vl_logits.py" \
    "$1" \
    --model-path "${model_path}" \
    "${@:2}"
