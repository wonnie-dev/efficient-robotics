#!/usr/bin/env python3
"""Run independent positive cover-observation calibration episodes."""

from __future__ import annotations

import argparse
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
    / "positive_cover_observation_calibration_v1.json"
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


def successful_positive_seeds(protocol: dict[str, Any]) -> set[int]:
    readiness = audit(protocol, LIVE_OUTPUT_ROOT)
    return {
        int(row["seed"])
        for row in readiness["episodes"]["target_inside_after_cover_removal"]
    }


def validate_batch(batch: dict[str, Any]) -> list[int]:
    for field in ("training_performed", "testing_performed", "reserved_test_seeds_used"):
        if batch.get(field) is not False:
            raise ValueError(f"{field} must be false")
    seeds = [int(seed) for seed in batch["seeds"]]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Batch seeds must be unique")
    leaked = sorted(set(seeds) & set(range(200, 210)))
    if leaked:
        raise ValueError(f"Reserved final-test seeds are forbidden: {leaked}")
    physical_gpu = int(batch["physical_gpu"])
    forbidden = {int(index) for index in batch["forbidden_gpus"]}
    if physical_gpu in forbidden:
        raise ValueError("Selected GPU is also forbidden")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    batch = load_json(config_path)
    seeds = validate_batch(batch)
    physical_gpu = int(batch["physical_gpu"])
    if args.validate_only:
        print(json.dumps({"status": "valid", "seeds": seeds, "physical_gpu": physical_gpu}, indent=2))
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match physical_gpu")
    if os.environ.get("PHYSICAL_GPU") != str(physical_gpu):
        raise RuntimeError("PHYSICAL_GPU does not match physical_gpu")

    protocol = load_json(PROTOCOL)
    output_root = resolve_path(batch["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "status.json"
    completed_path = output_root / "COMPLETED.json"
    failed_path = output_root / "FAILED.json"
    completed_path.unlink(missing_ok=True)
    failed_path.unlink(missing_ok=True)
    state: dict[str, Any] = {
        "schema_version": "positive-cover-observation-calibration-status-v1",
        "status": "running",
        "batch_config": str(config_path),
        "planned_seeds": seeds,
        "physical_gpu": physical_gpu,
        "forbidden_gpus": batch["forbidden_gpus"],
        "episode_results": [],
        "training_performed": False,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
    }
    write_json_atomic(status_path, state)
    started = time.perf_counter()
    try:
        for seed in seeds:
            if seed in successful_positive_seeds(protocol):
                state["episode_results"].append(
                    {"seed": seed, "status": "skipped_existing_strict_success", "runtime_seconds": 0.0}
                )
                write_json_atomic(status_path, state)
                continue
            stdout_path = output_root / f"seed{seed:03d}.stdout.log"
            stderr_path = output_root / f"seed{seed:03d}.stderr.log"
            command = [
                sys.executable,
                str(ROOT / "scripts" / "run_remove_cover_live_smoke.py"),
                "--seed", str(seed),
                "--timeout-seconds", "3600",
                "--rg6-lid-calibration-config", str(resolve_path(batch["rg6_lid_calibration_config"])),
                "--allow-provisional-rg6-lid-physics",
                "--rg6-coupling-mode", "coordinated_drives",
                "--coordinated-rg6-total-drive-effort-limit-nm",
                str(batch["coordinated_rg6_total_drive_effort_limit_nm"]),
                "--replan-after-remove-cover",
                "--execute-post-remove-grasp",
                "--force-remove-cover-for-observation-calibration",
            ]
            episode_started = time.perf_counter()
            print(f"POSITIVE_COVER_BATCH_START=seed{seed:03d}", flush=True)
            with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=dict(os.environ),
                    stdout=stdout,
                    stderr=stderr,
                    text=True,
                    check=False,
                )
            accepted = successful_positive_seeds(protocol)
            success = completed.returncode == 0 and seed in accepted
            state["episode_results"].append(
                {
                    "seed": seed,
                    "status": "completed" if success else "failed",
                    "returncode": completed.returncode,
                    "runtime_seconds": time.perf_counter() - episode_started,
                    "stdout": str(stdout_path),
                    "stderr": str(stderr_path),
                    "strict_physics_readiness_accepted": seed in accepted,
                }
            )
            write_json_atomic(status_path, state)
            print(f"POSITIVE_COVER_BATCH_{'COMPLETE' if success else 'FAILED'}=seed{seed:03d}", flush=True)
            if not success and batch.get("stop_on_first_failure", True):
                raise RuntimeError(f"Positive cover calibration failed at seed {seed}")

        readiness = audit(protocol, LIVE_OUTPUT_ROOT)
        readiness_path = ROOT / "outputs" / "final_evaluation" / "icra_protocol_v1" / "cover_calibration_readiness.json"
        write_json_atomic(readiness_path, readiness)
        calibration = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_full_action_conditioned_observation_calibration.py")],
            cwd=ROOT,
            env=dict(os.environ),
            capture_output=True,
            text=True,
            check=False,
        )
        state.update(
            {
                "status": "completed",
                "runtime_seconds": time.perf_counter() - started,
                "readiness": readiness,
                "observation_calibration_returncode": calibration.returncode,
                "observation_calibration_stdout": calibration.stdout,
                "observation_calibration_stderr": calibration.stderr,
            }
        )
        write_json_atomic(status_path, state)
        write_json_atomic(completed_path, state)
        print("POSITIVE_COVER_BATCH_COMPLETED", flush=True)
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
