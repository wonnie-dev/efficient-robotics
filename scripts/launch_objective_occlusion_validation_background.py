"""Launch a bounded GPU-5 objective-occlusion capture job."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        required=True,
        choices=(
            "inside_clear",
            "outside",
            "rim_occluded",
            "covered_unknown",
            "behind_ambiguous",
        ),
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    allowed_root = (ROOT / "outputs" / "live_pipeline").resolve()
    if not output_root.is_relative_to(allowed_root):
        raise ValueError(f"Output root must be under {allowed_root}")
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
                    f"Capture job is already running with PID {previous_pid}"
                )
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_objective_occlusion_validation.py"),
        "--seed-start",
        str(args.seed_start),
        "--output-root",
        str(output_root),
        "--variants",
        *args.variants,
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "5",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NVIDIA_VISIBLE_DEVICES": "5",
            "PHYSICAL_GPU": "5",
        }
    )
    for name in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
    ):
        environment.pop(name, None)
    log_path = output_root / "background.log"
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
        "schema_version": "background-objective-occlusion-job-v1",
        "pid": process.pid,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "log": str(log_path),
        "progress": str(output_root / "progress.json"),
        "summary": str(output_root / "summary.json"),
        "physical_gpu": 5,
        "multi_gpu": False,
    }
    job_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
