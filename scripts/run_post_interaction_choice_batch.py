#!/usr/bin/env python3
"""Run post-remove joint-choice inference as independent single-GPU shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/research/post_interaction_candidate_choice.json"
RUNNER = ROOT / "scripts/score_post_interaction_candidates.py"
EVALUATOR = ROOT / "scripts/evaluate_post_interaction_choice.py"
PYTHON = Path(
    os.environ.get(
        "EFFICIENT_ROBOTICS_PERCEPTION_PYTHON",
        ROOT / ".venv-perception/bin/python",
    )
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu-ids", nargs="+", type=int)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--limit-per-shard", type=int)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    gpu_ids = args.gpu_ids or [int(value) for value in config["gpu_ids"]]
    allowed = {0, 2, 4, 5}
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)) or not set(gpu_ids) <= allowed:
        raise ValueError("Only unique physical GPUs 0, 2, 4, and 5 are allowed")
    output_root = resolve_path(config["output_root"])
    logs = output_root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    state = {
        "schema_version": "post-interaction-choice-batch-status-v1",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_ids": gpu_ids,
        "one_model_instance_per_gpu": True,
        "batch_size_per_gpu": 1,
        "distributed": False,
        "ddp_used": False,
        "nccl_used": False,
        "training_performed": False,
        "reserved_test_seeds_used": False,
        "shards": [],
    }
    write_json(output_root / "status.json", state)
    jobs = []
    for shard_index, gpu in enumerate(gpu_ids):
        stdout_path = logs / f"gpu{gpu}_stdout.log"
        stderr_path = logs / f"gpu{gpu}_stderr.log"
        stdout = stdout_path.open("w", encoding="utf-8")
        stderr = stderr_path.open("w", encoding="utf-8")
        environment = dict(os.environ)
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "NVIDIA_VISIBLE_DEVICES": str(gpu),
                "PHYSICAL_GPU": str(gpu),
            }
        )
        for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
            environment.pop(name, None)
        command = [
            str(PYTHON),
            str(RUNNER),
            "--config",
            str(config_path),
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(len(gpu_ids)),
        ]
        if args.limit_per_shard is not None:
            command.extend(["--limit", str(args.limit_per_shard)])
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, text=True)
        jobs.append((shard_index, gpu, process, stdout, stderr, stdout_path, stderr_path))

    failures = []
    for shard_index, gpu, process, stdout, stderr, stdout_path, stderr_path in jobs:
        try:
            returncode = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            returncode = process.wait(timeout=30.0)
        finally:
            stdout.close()
            stderr.close()
        record = {
            "shard_index": shard_index,
            "physical_gpu": gpu,
            "status": "completed" if returncode == 0 else "failed",
            "returncode": returncode,
            "stdout": str(stdout_path.resolve()),
            "stderr": str(stderr_path.resolve()),
        }
        state["shards"].append(record)
        if returncode != 0:
            failures.append(record)
        write_json(output_root / "status.json", state)
    if failures:
        state.update({"status": "failed", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "failures": failures})
        write_json(output_root / "status.json", state)
        raise SystemExit(1)
    evaluation = subprocess.run(
        [str(PYTHON), str(EVALUATOR), "--config", str(config_path)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    (logs / "evaluation_stdout.log").write_text(evaluation.stdout, encoding="utf-8")
    (logs / "evaluation_stderr.log").write_text(evaluation.stderr, encoding="utf-8")
    if evaluation.returncode != 0:
        state.update({"status": "failed", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_returncode": evaluation.returncode})
        write_json(output_root / "status.json", state)
        raise SystemExit(evaluation.returncode)
    state.update({"status": "completed", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation": str((output_root / "evaluation.json").resolve())})
    write_json(output_root / "status.json", state)
    print(json.dumps(state, indent=2))


if __name__ == "__main__":
    main()
