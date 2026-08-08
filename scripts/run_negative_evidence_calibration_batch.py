#!/usr/bin/env python3
"""Run independent negative-evidence calibration episodes sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from audit_cover_calibration_readiness import audit, load_json


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "negative_evidence_calibration_remaining_v1.json"
)
PROTOCOL = (
    ROOT / "configs" / "research" / "icra_simulation_evaluation_protocol_v1.json"
)
LIVE_OUTPUT_ROOT = ROOT / "outputs" / "live_pipeline"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def successful_negative_seeds(protocol: dict[str, Any]) -> set[int]:
    readiness = audit(protocol, LIVE_OUTPUT_ROOT)
    return {
        int(row["seed"])
        for row in readiness["episodes"][
            "empty_container_negative_evidence"
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    batch_path = args.config.resolve()
    batch = load_json(batch_path)
    protocol = load_json(PROTOCOL)
    seeds = [int(seed) for seed in batch["seeds"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Batch seeds must be unique")
    if set(seeds) & set(range(200, 210)):
        raise ValueError("Reserved final-test seeds are forbidden")
    episode_config = resolve_path(batch["episode_config"])
    episode = load_json(episode_config)
    relation_path = resolve_path(episode["rgbd_relation_config"])
    relation = load_json(relation_path)
    source_path = resolve_path(relation["calibration_source"])
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if relation.get("frozen") is not True:
        raise ValueError("Runtime relation configuration is not frozen")
    if source_hash != relation.get("calibration_source_sha256"):
        raise ValueError("Runtime relation calibration hash mismatch")
    if args.validate_only:
        print(
            json.dumps(
                {
                    "status": "valid",
                    "seeds": seeds,
                    "episode_config": str(episode_config),
                    "relation_config": str(relation_path),
                    "physical_gpu": int(batch["physical_gpu"]),
                    "forbidden_gpus": batch["forbidden_gpus"],
                },
                indent=2,
            )
        )
        return
    physical_gpu = int(batch["physical_gpu"])
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(physical_gpu):
        raise RuntimeError(
            f"Expected CUDA_VISIBLE_DEVICES={physical_gpu}, got {visible!r}"
        )
    if os.environ.get("PHYSICAL_GPU") != str(physical_gpu):
        raise RuntimeError("PHYSICAL_GPU does not match the batch manifest")

    output_root = resolve_path(batch["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"
    completed_path = output_root / "COMPLETED.json"
    failed_path = output_root / "FAILED.json"
    completed_path.unlink(missing_ok=True)
    failed_path.unlink(missing_ok=True)
    state: dict[str, Any] = {
        "schema_version": "negative-evidence-calibration-batch-status-v1",
        "status": "running",
        "batch_config": str(batch_path),
        "physical_gpu": physical_gpu,
        "forbidden_gpus": [int(value) for value in batch["forbidden_gpus"]],
        "planned_seeds": seeds,
        "episode_results": [],
        "training_performed": False,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
    }
    write_json_atomic(status_path, state)
    started = time.perf_counter()
    try:
        for seed in seeds:
            accepted_before = successful_negative_seeds(protocol)
            if seed in accepted_before:
                result = {
                    "seed": seed,
                    "status": "skipped_existing_strict_success",
                    "runtime_seconds": 0.0,
                }
                state["episode_results"].append(result)
                write_json_atomic(status_path, state)
                print(f"NEGATIVE_EVIDENCE_BATCH_SKIP=seed{seed:03d}", flush=True)
                continue

            stdout_path = output_root / f"seed{seed:03d}.stdout.log"
            stderr_path = output_root / f"seed{seed:03d}.stderr.log"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_negative_evidence_live_smoke.py"),
                "--config",
                str(episode_config),
                "--seed",
                str(seed),
                "--timeout-seconds",
                "3600",
            ]
            episode_started = time.perf_counter()
            print(f"NEGATIVE_EVIDENCE_BATCH_START=seed{seed:03d}", flush=True)
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
                "w", encoding="utf-8"
            ) as stderr:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=dict(os.environ),
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    check=False,
                )
            accepted_after = successful_negative_seeds(protocol)
            success = completed.returncode == 0 and seed in accepted_after
            result = {
                "seed": seed,
                "status": "completed" if success else "failed",
                "returncode": completed.returncode,
                "runtime_seconds": time.perf_counter() - episode_started,
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
                "strict_physics_readiness_accepted": seed in accepted_after,
            }
            state["episode_results"].append(result)
            state["last_completed_seed"] = seed if success else state.get(
                "last_completed_seed"
            )
            write_json_atomic(status_path, state)
            print(
                f"NEGATIVE_EVIDENCE_BATCH_{'COMPLETE' if success else 'FAILED'}="
                f"seed{seed:03d}",
                flush=True,
            )
            if not success and batch.get("stop_on_first_failure", True):
                raise RuntimeError(f"Negative-evidence calibration failed at seed {seed}")

        readiness = audit(protocol, LIVE_OUTPUT_ROOT)
        state.update(
            {
                "status": "completed",
                "runtime_seconds": time.perf_counter() - started,
                "readiness": readiness,
            }
        )
        write_json_atomic(status_path, state)
        write_json_atomic(completed_path, state)
        write_json_atomic(
            ROOT
            / "outputs"
            / "final_evaluation"
            / "icra_protocol_v1"
            / "cover_calibration_readiness.json",
            readiness,
        )
        print("NEGATIVE_EVIDENCE_BATCH_COMPLETED", flush=True)
    except Exception as error:
        state.update(
            {
                "status": "failed",
                "runtime_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        write_json_atomic(status_path, state)
        write_json_atomic(failed_path, state)
        raise


if __name__ == "__main__":
    main()
