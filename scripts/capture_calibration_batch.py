#!/usr/bin/env python3
"""Collect the predeclared calibration observations across GPU shards."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/research/calibration_episodes.json"
REMOVE_RUNNER = ROOT / "scripts/capture_scenario_episode.py"
STATIC_RUNNER = ROOT / "scripts/capture_reachable_views.py"
CALIBRATION_OBSERVATION_FILES = (
    "rgb.png",
    "depth_m.npy",
    "camera_calibration.json",
)
sys.path.insert(0, str(ROOT / "scripts"))

from run_closed_loop_batch import (  # noqa: E402
    gpu_device_minors,
    isolated_command,
    require_gpu_devices,
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def assignments(config: dict[str, Any]) -> list[dict[str, Any]]:
    families = list(config["family_cycle"])
    gpu_ids = [int(value) for value in config["gpu_ids"]]
    count = int(config["episode_count"])
    start = int(config["seed_start"])
    schedule = config.get("family_schedule")
    if schedule is None:
        family_indices = [offset % len(families) for offset in range(count)]
        schedule_source = "legacy_balanced_cycle"
    else:
        family_indices = [int(value) for value in schedule]
        if len(family_indices) != count:
            raise ValueError("family_schedule must contain one entry per episode")
        if any(index < 0 or index >= len(families) for index in family_indices):
            raise ValueError("family_schedule contains an invalid family index")
        support = {
            index: family_indices.count(index) for index in range(len(families))
        }
        expected_counts = config.get("expected_family_counts")
        if expected_counts is not None:
            declared = {
                index: int(expected_counts[families[index]["family"]])
                for index in range(len(families))
            }
            if support != declared:
                raise ValueError(
                    "family_schedule must preserve expected_family_counts"
                )
        else:
            expected = int(config["episodes_per_family"])
            if any(value != expected for value in support.values()):
                raise ValueError(
                    "family_schedule must preserve the declared balanced support"
                )
        schedule_source = "predeclared_counterbalanced_schedule"
    rows = []
    for offset in range(count):
        rows.append(
            {
                **families[family_indices[offset]],
                "seed": start + offset,
                "physical_gpu": gpu_ids[offset % len(gpu_ids)],
                "family_schedule_index": family_indices[offset],
                "family_schedule_source": schedule_source,
            }
        )
    return rows


def completed(output_root: Path, row: dict[str, Any]) -> Path | None:
    seed_root = output_root / row["family"] / f"seed{int(row['seed']):03d}"
    name = (
        "cover_removal_result.json"
        if row["collector"] == "remove_cover_counterfactual"
        else "result.json"
    )
    for path in sorted(seed_root.glob(f"run*/{name}"), reverse=True):
        result = load_json(path)
        if row["collector"] == "remove_cover_counterfactual":
            removal = result.get("cover_removal_execution") or {}
            run_dir = path.parent
            artifacts = all(
                (run_dir / "observations" / view / name).is_file()
                for view in ("center", "post_remove", "close_high", "right")
                for name in CALIBRATION_OBSERVATION_FILES
            )
            passed = bool(
                result.get("visibility_outcome_passed") is True
                and removal.get("removal_verified") is True
                and removal.get("status") == "completed"
                and removal.get("bilateral_contact_before_lift") is True
                and not removal.get("unexpected_environment_pairs")
                and removal.get("contact_force_within_limit") is True
                and removal.get("contact_penetration_within_limit") is True
                and artifacts
            )
        else:
            trajectories = list(result.get("trajectories", []))
            passed = bool(
                len(trajectories) == 2
                and all(item.get("status") == "completed" for item in trajectories)
                and all(
                    item.get("actual_robot_motion_executed") is True
                    for item in trajectories
                )
                and all(item.get("collision_checked") is True for item in trajectories)
                and result.get("calibration_visibility_validation", {}).get("passed")
            )
        if passed:
            return path
    return None


def next_run_dir(seed_root: Path) -> Path:
    indices = []
    for path in seed_root.glob("run*"):
        try:
            indices.append(int(path.name.removeprefix("run")))
        except ValueError:
            continue
    return seed_root / f"run{max(indices, default=0) + 1:03d}"


def logical_environment(host_gpu: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NVIDIA_VISIBLE_DEVICES": "0",
            "PHYSICAL_GPU": "0",
            "EFFICIENT_ROBOTICS_HOST_PHYSICAL_GPU": str(host_gpu),
            "EFFICIENT_ROBOTICS_ISAAC_PYTHON": environment.get(
                "EFFICIENT_ROBOTICS_ISAAC_PYTHON",
                str(ROOT / ".venv-isaac/bin/python"),
            ),
            "EFFICIENT_ROBOTICS_LIBERO_ROOT": environment.get(
                "EFFICIENT_ROBOTICS_LIBERO_ROOT",
                str(ROOT / "third_party/LIBERO"),
            ),
        }
    )
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)
    return environment


def authorize_reserved_test(
    config: dict[str, Any], *, confirm_open: bool
) -> dict[str, Any]:
    """Fail closed until a frozen package explicitly authorizes test opening."""
    if not confirm_open:
        raise RuntimeError(
            "Opening a reserved split requires --confirm-open-reserved-test"
        )
    if config.get("reserved_test_opened") is not False:
        raise RuntimeError("The source test config must still declare opened=false")
    if config.get("launch_authorized") is not True:
        raise RuntimeError("Use the capture config emitted by the freeze step")
    protocol = load_json(resolve_path(str(config["final_evaluation_protocol"])))
    if (
        protocol.get("status") != "frozen_before_untouched_test"
        or protocol.get("reserved_test_launch_authorized") is not True
    ):
        raise RuntimeError("The final protocol is not frozen and authorized")
    manifest = load_json(resolve_path(str(config["freeze_manifest"])))
    if manifest.get("status") != "frozen_ready_for_reserved_test":
        raise RuntimeError("The calibration manifest is not frozen")
    expected = set(range(int(config["seed_start"]), int(config["seed_start"]) + int(config["episode_count"])))
    frozen = {int(value) for value in manifest["reserved_test_seeds"]}
    if frozen != expected or manifest.get("reserved_test_opened") is not False:
        raise RuntimeError("The frozen test split changed or was already opened")
    return {"protocol": protocol, "manifest": manifest}


# Kept for older callers while all reserved splits use the generic guard above.
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-waves", type=int)
    parser.add_argument("--confirm-open-reserved-test", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    data_split = str(config.get("data_split", "calibration"))
    if data_split not in {"calibration", "test"}:
        raise ValueError(f"Unsupported data split: {data_split}")
    is_test = data_split == "test"
    authorization = None
    if is_test:
        authorization = authorize_reserved_test(
            config, confirm_open=bool(args.confirm_open_reserved_test)
        )
    rows = assignments(config)
    gpu_ids = [int(value) for value in config["gpu_ids"]]
    require_gpu_devices(gpu_ids)
    minors = gpu_device_minors(gpu_ids)
    output_root = resolve_path(config["output_root"])
    status_path = output_root / "batch_status.json"
    if authorization is not None:
        marker = output_root / "RESERVED_TEST_OPENED.json"
        if not marker.exists():
            write_json(
                marker,
                {
                    "schema_version": "reserved-test-open-marker-v1",
                    "opened_at_utc": datetime.now(timezone.utc).isoformat(),
                    "seed_start": int(config["seed_start"]),
                    "episode_count": int(config["episode_count"]),
                    "freeze_manifest": str(
                        resolve_path(str(config["freeze_manifest"])).resolve()
                    ),
                    "test_data_used_for_tuning_forbidden": True,
                },
            )
    completed_seeds = {
        int(row["seed"]) for row in rows if completed(output_root, row) is not None
    }
    queues = {
        gpu: [row for row in rows if int(row["physical_gpu"]) == gpu and int(row["seed"]) not in completed_seeds]
        for gpu in gpu_ids
    }
    state: dict[str, Any] = {
        "schema_version": "observation-capture-status-v1",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_ids": gpu_ids,
        "gpu_device_minors": {str(k): v for k, v in minors.items()},
        "device_isolation": "bubblewrap_single_nvidia_device",
        "one_process_per_gpu": True,
        "multi_gpu_inside_process": False,
        "completed_seeds": sorted(completed_seeds),
        "failed_episodes": [],
        "data_split": data_split,
        "training_performed": False,
        "calibration_data_collection": not is_test,
        "calibration_performed": bool(
            config.get("calibration_performed", False)
        ),
        "testing_performed": is_test,
        "reserved_test_seeds_used": is_test,
        "valid_for_final_evaluation": False,
    }
    write_json(status_path, state)
    total_waves = max((len(queue) for queue in queues.values()), default=0)
    if args.max_waves is not None:
        total_waves = min(total_waves, args.max_waves)
    for wave in range(total_waves):
        jobs = []
        for host_gpu, queue in queues.items():
            if wave >= len(queue):
                continue
            row = queue[wave]
            seed = int(row["seed"])
            seed_root = output_root / row["family"] / f"seed{seed:03d}"
            logs = output_root / "logs"
            logs.mkdir(parents=True, exist_ok=True)
            stdout_path = logs / f"seed{seed:03d}_stdout.log"
            stderr_path = logs / f"seed{seed:03d}_stderr.log"
            if row["collector"] == "remove_cover_counterfactual":
                episode_config = {
                    **config,
                    "output_root": str(output_root.resolve()),
                    "assignments": [{**row, "physical_gpu": 0}],
                }
                episode_config_path = output_root / "episode_configs" / f"seed{seed:03d}.json"
                write_json(episode_config_path, episode_config)
                command = [
                    sys.executable,
                    str(REMOVE_RUNNER),
                    "--config",
                    str(episode_config_path),
                    "--physical-gpu",
                    "0",
                    "--seed",
                    str(seed),
                ]
            else:
                run_dir = next_run_dir(seed_root)
                command = [
                    sys.executable,
                    str(STATIC_RUNNER),
                    "--seed",
                    str(seed),
                    "--scanned-basket-scene",
                    "--calibration-scene-variant",
                    row["scene_variant"],
                    "--basket-contact-physics",
                    "--requested-views",
                    "close_high",
                    "right",
                    "--startup-timeout-seconds",
                    str(float(config.get("startup_timeout_seconds", 1200.0))),
                    "--output-dir",
                    str(run_dir),
                ]
                protocol_value = config.get("final_evaluation_protocol")
                if protocol_value:
                    protocol = resolve_path(str(protocol_value))
                    command.extend(
                        [
                            "--final-evaluation-authorized",
                            "--protocol",
                            str(protocol),
                        ]
                    )
            stdout_stream = stdout_path.open("w", encoding="utf-8")
            stderr_stream = stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                isolated_command(command, minors[host_gpu]),
                cwd=ROOT,
                env=logical_environment(host_gpu),
                stdout=stdout_stream,
                stderr=stderr_stream,
                text=True,
            )
            jobs.append((row, process, stdout_stream, stderr_stream, stdout_path, stderr_path))
        wave_failed = False
        for row, process, stdout_stream, stderr_stream, stdout_path, stderr_path in jobs:
            returncode = process.wait()
            stdout_stream.close()
            stderr_stream.close()
            result_path = completed(output_root, row)
            if result_path is not None:
                state["completed_seeds"].append(int(row["seed"]))
            else:
                wave_failed = True
                state["failed_episodes"].append(
                    {
                        **row,
                        "returncode": returncode,
                        "stdout": str(stdout_path.resolve()),
                        "stderr": str(stderr_path.resolve()),
                    }
                )
            state["completed_seeds"] = sorted(set(state["completed_seeds"]))
            write_json(status_path, state)
        if wave_failed:
            break
    all_complete = len(state["completed_seeds"]) == int(config["episode_count"])
    state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    state["status"] = (
        "completed"
        if all_complete
        else ("failed" if state["failed_episodes"] else "partial")
    )
    state["valid_for_final_evaluation"] = bool(is_test and all_complete)
    write_json(status_path, state)
    print(json.dumps(state, indent=2))
    if state["failed_episodes"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
