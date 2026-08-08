"""Launch the bounded GPU-5 calibration pilot independently of the SSH shell."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    ROOT
    / "outputs"
    / "calibration_pilot"
    / "scanned_basket_seed100_109"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--seed-stop", type=int, default=110)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--existing-capture-manifest", type=Path, nargs="+"
    )
    parser.add_argument(
        "--minimum-candidate-proposals", type=int, default=1
    )
    parser.add_argument("--perception-only-basket", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    job_path = output_root / "background_job.json"
    if job_path.is_file():
        previous = json.loads(job_path.read_text(encoding="utf-8"))
        previous_pid = int(previous.get("pid", -1))
        if previous_pid > 0:
            try:
                os.kill(previous_pid, 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(
                    f"Calibration job is already running with PID {previous_pid}"
                )
    log_path = output_root / "background.log"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_grounded_qwen_calibration_pilot.py"),
        "--seed-start",
        str(args.seed_start),
        "--seed-stop",
        str(args.seed_stop),
        "--output-root",
        str(output_root),
        "--minimum-candidate-proposals",
        str(args.minimum_candidate_proposals),
    ]
    if args.existing_capture_manifest is not None:
        command.append("--existing-capture-manifest")
        command.extend(
            str(path.resolve())
            for path in args.existing_capture_manifest
        )
    if args.perception_only_basket:
        command.append("--perception-only-basket")
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "5",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NVIDIA_VISIBLE_DEVICES": "5",
            "PHYSICAL_GPU": "5",
        }
    )
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    payload = {
        "schema_version": "background-calibration-job-v1",
        "pid": process.pid,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "log": str(log_path),
        "status_file": str(output_root / "status.json"),
        "completed_marker": str(output_root / "COMPLETED.json"),
        "failed_marker": str(output_root / "FAILED.json"),
        "physical_gpu": 5,
        "multi_gpu": False,
    }
    job_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
