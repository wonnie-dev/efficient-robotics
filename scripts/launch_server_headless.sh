#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
isaacsim_venv="${EFFICIENT_ROBOTICS_ISAACSIM_VENV:-/data/wonheekoh/isaacsim_venv}"
isaac_python="${isaacsim_venv}/bin/python"

if [[ ! -x "${isaac_python}" ]]; then
    echo "Isaac Sim Python is not executable: ${isaac_python}" >&2
    exit 1
fi

if [[ -z "${PHYSICAL_GPU:-}" ]]; then
    echo "Set PHYSICAL_GPU to one physical GPU index." >&2
    exit 2
fi
if [[ ! "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]]; then
    echo "PHYSICAL_GPU must be a non-negative integer." >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NVIDIA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
render_quality="${EFFICIENT_ROBOTICS_RENDER_QUALITY:-preview}"

if [[ "${render_quality}" != "preview" && "${render_quality}" != "paper" ]]; then
    echo "Unsupported EFFICIENT_ROBOTICS_RENDER_QUALITY: ${render_quality}" >&2
    exit 1
fi

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES}"
echo "ISAACSIM_VENV=${isaacsim_venv}"
echo "RENDERER_PHYSICAL_GPU=${PHYSICAL_GPU}"
echo "PHYSICS_CUDA_DEVICE=0"
echo "MULTI_GPU=false"
echo "MAX_GPU_COUNT=1"
echo "RENDER_QUALITY=${render_quality}"

exec "${isaac_python}" \
    "${project_root}/scripts/open_minimal_scene.py" \
    --scene-profile minimal \
    --headless \
    --renderer-gpu "${PHYSICAL_GPU}" \
    --physics-gpu 0 \
    --render-quality "${render_quality}" \
    --capture-video
