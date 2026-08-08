#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -ne 1 ]]; then
    echo "Usage: PHYSICAL_GPU=N $0 INPUT_JSON" >&2
    exit 2
fi
if [[ -z "${PHYSICAL_GPU:-}" || ! "${PHYSICAL_GPU}" =~ ^[0-9]+$ ]]; then
    echo "Set PHYSICAL_GPU to one physical GPU index." >&2
    exit 2
fi

exec "${project_root}/scripts/run_qwen3_vl_single_gpu.sh" "$1"
