#!/usr/bin/env python3
"""Run V16 calibration inference as four independent single-GPU shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_CONFIG = ROOT / "configs/research/icra_v16_calibration_36episode.json"
CAPTURE_ROOT = ROOT / "outputs/live_pipeline/icra_v16_calibration_36episode"
OUTPUT_ROOT = ROOT / "outputs/calibration/icra_v16_calibration_perception"
RUNNER = ROOT / "scripts/run_icra_v15_calibration_perception.py"
PERCEPTION_PYTHON = Path(
    os.environ.get(
        "EFFICIENT_ROBOTICS_PERCEPTION_PYTHON",
        str(ROOT / ".venv-perception/bin/python"),
    )
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def completed_shard(path: Path) -> bool:
    marker = path / "COMPLETED.json"
    if not marker.is_file():
        return False
    result = load_json(marker)
    expected = int(result["sample_count"])
    actual = len(
        list((path / "grounded_sam2_qwen_rankings").glob("*/result.json"))
    )
    return result.get("status") == "completed" and actual == expected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-ids", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--timeout-seconds", type=float, default=7200.0)
    parser.add_argument("--capture-config", type=Path, default=CAPTURE_CONFIG)
    parser.add_argument("--capture-root", type=Path, default=CAPTURE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--reserved-test", action="store_true")
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--confirm-open-reserved-test", action="store_true")
    args = parser.parse_args()
    capture_config = args.capture_config.resolve()
    capture_root = args.capture_root.resolve()
    output_root = args.output_root.resolve()
    if args.reserved_test:
        if not args.confirm_open_reserved_test:
            raise RuntimeError("Reserved-test perception requires explicit confirmation")
        if args.freeze_manifest is None:
            raise RuntimeError("Reserved-test perception requires a freeze manifest")
        freeze = load_json(args.freeze_manifest.resolve())
        if freeze.get("status") != "frozen_ready_for_reserved_test":
            raise RuntimeError("The V16 calibration package is not frozen")
        opened = capture_root / "RESERVED_TEST_OPENED.json"
        capture_status = capture_root / "batch_status.json"
        if not opened.is_file() or not capture_status.is_file():
            raise RuntimeError("Reserved-test capture has not been explicitly opened")
        captured = load_json(capture_status)
        expected = {int(value) for value in freeze["reserved_test_seeds"]}
        if (
            captured.get("status") != "completed"
            or {int(value) for value in captured["completed_seeds"]} != expected
        ):
            raise RuntimeError("Reserved-test RGB-D capture is incomplete")
    gpu_ids = list(args.gpu_ids)
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError("gpu-ids must be a non-empty unique list")
    if any(gpu < 0 or gpu > 5 for gpu in gpu_ids):
        raise ValueError("gpu-ids must be host indices in [0, 5]")
    if not PERCEPTION_PYTHON.is_file():
        raise FileNotFoundError(PERCEPTION_PYTHON)

    shard_root = output_root / "shards"
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "schema_version": (
            "icra-v16-reserved-test-perception-sharded-status-v1"
            if args.reserved_test
            else "icra-v16-calibration-perception-sharded-status-v1"
        ),
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_ids": gpu_ids,
        "shard_count": len(gpu_ids),
        "one_model_instance_per_gpu": True,
        "batch_size_per_gpu": 1,
        "distributed": False,
        "ddp_used": False,
        "nccl_used": False,
        "training_performed": False,
        "data_split": "test" if args.reserved_test else "calibration",
        "calibration_data_processed": not args.reserved_test,
        "calibration_performed": False,
        "testing_performed": args.reserved_test,
        "reserved_test_seeds_used": args.reserved_test,
        "shards": [],
    }
    write_json(output_root / "status.json", state)

    jobs = []
    for shard_index, gpu in enumerate(gpu_ids):
        destination = shard_root / f"gpu{gpu}"
        if completed_shard(destination):
            state["shards"].append(
                {
                    "shard_index": shard_index,
                    "physical_gpu": gpu,
                    "status": "completed",
                    "cache_hit": True,
                    "output_root": str(destination.resolve()),
                }
            )
            continue
        stdout_path = log_root / f"gpu{gpu}_stdout.log"
        stderr_path = log_root / f"gpu{gpu}_stderr.log"
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
            str(PERCEPTION_PYTHON),
            str(RUNNER),
            "--physical-gpu",
            str(gpu),
            "--capture-config",
            str(capture_config),
            "--capture-root",
            str(capture_root),
            "--output-root",
            str(destination),
            "--shard-index",
            str(shard_index),
            "--shard-count",
            str(len(gpu_ids)),
        ]
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        jobs.append(
            (
                shard_index,
                gpu,
                destination,
                process,
                stdout,
                stderr,
                stdout_path,
                stderr_path,
            )
        )

    failures = []
    for (
        shard_index,
        gpu,
        destination,
        process,
        stdout,
        stderr,
        stdout_path,
        stderr_path,
    ) in jobs:
        try:
            returncode = process.wait(timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=30.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=30.0)
            returncode = -9
        finally:
            stdout.close()
            stderr.close()
        passed = returncode == 0 and completed_shard(destination)
        record = {
            "shard_index": shard_index,
            "physical_gpu": gpu,
            "status": "completed" if passed else "failed",
            "cache_hit": False,
            "returncode": returncode,
            "output_root": str(destination.resolve()),
            "stdout": str(stdout_path.resolve()),
            "stderr": str(stderr_path.resolve()),
        }
        state["shards"].append(record)
        if not passed:
            failures.append(record)
        write_json(output_root / "status.json", state)

    if failures:
        state.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "failures": failures,
            }
        )
        write_json(output_root / "status.json", state)
        raise SystemExit(1)

    merged = output_root / "grounded_sam2_qwen_rankings"
    merged.mkdir(parents=True, exist_ok=True)
    sample_ids = set()
    for gpu in gpu_ids:
        source = shard_root / f"gpu{gpu}" / "grounded_sam2_qwen_rankings"
        for result in sorted(source.glob("*/result.json")):
            sample_id = result.parent.name
            if sample_id in sample_ids:
                raise RuntimeError(f"Duplicate perception sample: {sample_id}")
            sample_ids.add(sample_id)
            destination = merged / sample_id
            target = result.parent.resolve()
            if destination.exists() or destination.is_symlink():
                if destination.resolve() != target:
                    raise RuntimeError(
                        f"Conflicting merged sample: {destination}"
                    )
            else:
                destination.symlink_to(target, target_is_directory=True)

    expected = sum(
        int(load_json(shard_root / f"gpu{gpu}" / "COMPLETED.json")["sample_count"])
        for gpu in gpu_ids
    )
    if len(sample_ids) != expected:
        raise RuntimeError(
            f"Expected {expected} merged samples, found {len(sample_ids)}"
        )
    from run_icra_v15_calibration_perception import build_config

    full_config, full_manifest = build_config(
        output_root,
        capture_config_path=capture_config,
        capture_root=capture_root,
    )
    write_json(output_root / "inference_config.json", full_config)
    write_json(output_root / "observation_manifest.json", full_manifest)
    state.update(
        {
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "merged_sample_count": len(sample_ids),
            "expected_sample_count": expected,
        }
    )
    write_json(output_root / "status.json", state)
    write_json(output_root / "COMPLETED.json", state)
    print(f"ICRA_V16_CALIBRATION_PERCEPTION={output_root / 'COMPLETED.json'}")


if __name__ == "__main__":
    main()
