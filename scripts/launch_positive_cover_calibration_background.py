#!/usr/bin/env python3
"""Launch the bounded positive-cover calibration batch under a GPU watchdog."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "research" / "positive_cover_observation_calibration_v1.json"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    physical_gpu = int(config["physical_gpu"])
    forbidden = [int(index) for index in config["forbidden_gpus"]]
    output_root = resolve_path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    watchdog_path = output_root / "watchdog.json"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_with_forbidden_gpu_watchdog.py"),
        "--physical-gpu", str(physical_gpu),
    ]
    for gpu in forbidden:
        command.extend(["--forbidden-gpu", str(gpu)])
    command.extend(
        [
            "--status-json", str(watchdog_path), "--",
            sys.executable,
            str(ROOT / "scripts" / "run_positive_cover_observation_calibration_batch.py"),
            "--config", str(config_path),
        ]
    )
    environment = dict(os.environ)
    gpu_text = str(physical_gpu)
    environment.update(
        {
            "PHYSICAL_GPU": gpu_text,
            "CUDA_VISIBLE_DEVICES": gpu_text,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NVIDIA_VISIBLE_DEVICES": gpu_text,
        }
    )
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
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
        "schema_version": "positive-cover-background-job-v1",
        "pid": process.pid,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "batch_config": str(config_path),
        "log": str(log_path),
        "status": str(output_root / "status.json"),
        "watchdog": str(watchdog_path),
        "completed_marker": str(output_root / "COMPLETED.json"),
        "failed_marker": str(output_root / "FAILED.json"),
        "physical_gpu": physical_gpu,
        "forbidden_gpus": forbidden,
        "multi_gpu": False,
    }
    job_path = output_root / "background_job.json"
    temporary = job_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(job_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
