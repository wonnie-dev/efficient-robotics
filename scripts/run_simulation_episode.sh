#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
isaacsim_venv="${EFFICIENT_ROBOTICS_ISAACSIM_VENV:-}"
seed="${1:-0}"

if [[ -z "${PHYSICAL_GPU:-}" || ! "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]]; then
    echo "Set PHYSICAL_GPU to one physical GPU index." >&2
    exit 2
fi
if [[ -z "${isaacsim_venv}" || ! -x "${isaacsim_venv}/bin/python" ]]; then
    echo "Set EFFICIENT_ROBOTICS_ISAACSIM_VENV to an Isaac Sim environment." >&2
    exit 2
fi
if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Seed must be a non-negative integer." >&2
    exit 2
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${PHYSICAL_GPU}"
export NVIDIA_VISIBLE_DEVICES="${PHYSICAL_GPU}"

exec "${isaacsim_venv}/bin/python" \
    "${project_root}/scripts/open_minimal_scene.py" \
    --scene-profile benchmark \
    --headless \
    --renderer-gpu "${PHYSICAL_GPU}" \
    --physics-gpu 0 \
    --seed "${seed}" \
    --household-perception-pilot \
    --scanned-basket-perception-pilot \
    --calibration-scene-variant auto
