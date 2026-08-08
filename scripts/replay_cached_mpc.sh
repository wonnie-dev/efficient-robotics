#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${EFFICIENT_ROBOTICS_CPU_PYTHON:-python3}"

exec "${python_bin}" \
    "${project_root}/scripts/run_cover_search_scene_graph_mpc_integration.py" \
    --config "${project_root}/configs/research/cover_search_scene_graph_mpc_integration.json"
