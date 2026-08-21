#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${EFFICIENT_ROBOTICS_CPU_PYTHON:-python3}"
config="${EFFICIENT_ROBOTICS_HARDWARE_CONFIG:-${project_root}/configs/hardware/rg6_lid_transfer_calibration.json}"

echo "Configuration check only. No robot command will be sent."
exec "${python_bin}" \
    "${project_root}/scripts/rg6_lid_calibration.py" \
    --config "${config}" \
    --require-transfer-ready
