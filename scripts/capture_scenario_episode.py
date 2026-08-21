#!/usr/bin/env python3
"""Capture one development scene for action-differentiating validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/scenario_capture.json"
OBSERVATION_FILES = ("rgb.png", "depth_m.npy", "camera_calibration.json")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assignment_for_seed(config: dict[str, Any], seed: int) -> dict[str, Any]:
    matches = [row for row in config["assignments"] if int(row["seed"]) == seed]
    if len(matches) != 1:
        raise ValueError(f"Expected one scenario assignment for seed {seed}, found {len(matches)}")
    return matches[0]


def output_root(config: dict[str, Any]) -> Path:
    path = Path(config["output_root"])
    return path if path.is_absolute() else ROOT / path


def episode_status_path(root: Path, seed: int) -> Path:
    """Keep concurrent isolated episodes from sharing one temporary file."""
    return root / "episode_status" / f"seed{seed:03d}.json"


def completed_run(config: dict[str, Any], assignment: dict[str, Any]) -> tuple[Path, dict[str, Any]] | None:
    seed = int(assignment["seed"])
    seed_root = output_root(config) / assignment["family"] / f"seed{seed:03d}"
    counterfactual_views = list(
        config.get("counterfactual_post_remove_views", ["close_high", "right"])
    )
    for result_path in sorted(seed_root.glob("run*/cover_removal_result.json"), reverse=True):
        result = load_json(result_path)
        run_dir = result_path.parent
        removal = result.get("cover_removal_execution") or {}
        artifacts = all(
            (run_dir / "observations" / view / name).is_file()
            for view in config["views"]
            for name in OBSERVATION_FILES
        )
        if (
            int(result.get("seed", -1)) == seed
            and result.get("counterfactual_cache_only") is True
            and result.get("counterfactual_views") == counterfactual_views
            and removal.get("status") == "completed"
            and removal.get("removal_verified") is True
            and removal.get("bilateral_contact_before_lift") is True
            and removal.get("contact_force_within_limit") is True
            and removal.get("contact_penetration_within_limit") is True
            and not removal.get("unexpected_environment_pairs")
            and artifacts
        ):
            return run_dir, result
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--physical-gpu", type=int, required=True, choices=range(6))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Render a new run even when a valid development cache exists.",
    )
    args = parser.parse_args()

    gpu = str(args.physical_gpu)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must match --physical-gpu")
    if os.environ.get("PHYSICAL_GPU") != gpu:
        raise RuntimeError("PHYSICAL_GPU must match --physical-gpu")

    config = load_json(args.config.resolve())
    assignment = assignment_for_seed(config, args.seed)
    if int(assignment["physical_gpu"]) != args.physical_gpu:
        raise ValueError(
            f"Seed {args.seed} is assigned to GPU {assignment['physical_gpu']}, not GPU {args.physical_gpu}"
        )
    existing = None if args.force else completed_run(config, assignment)
    if existing is not None:
        run_dir, _ = existing
        print(f"V6_STRESS_CACHE_HIT={run_dir}")
        return

    root = output_root(config)
    family_root = root / assignment["family"]
    physics = Path(config["physics_config"])
    physics = physics if physics.is_absolute() else ROOT / physics
    log_root = root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"seed{args.seed:03d}_gpu{args.physical_gpu}.log"
    status_path = episode_status_path(root, args.seed)
    state = {
        "schema_version": "scenario-capture-status-v1",
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "physical_gpu": args.physical_gpu,
        "renderer_active_gpu": args.physical_gpu,
        "physics_cuda_device": 0,
        "multi_gpu": False,
        "assignment": assignment,
        "training_performed": False,
        "calibration_performed": bool(config.get("calibration_performed", False)),
        "testing_performed": bool(config.get("testing_performed", False)),
    }
    write_json_atomic(status_path, state)

    command = [
        sys.executable,
        str(ROOT / "scripts/execute_cover_removal.py"),
        "--seed", str(args.seed),
        "--scene-variant", assignment["scene_variant"],
        "--timeout-seconds", "3600",
        "--rg6-lid-calibration-config", str(physics),
        "--allow-provisional-rg6-lid-physics",
        "--output-root", str(family_root),
        "--rg6-coupling-mode", "coordinated_drives",
        "--coordinated-rg6-total-drive-effort-limit-nm", "18.0",
        "--replan-after-remove-cover",
        "--counterfactual-post-remove-views",
        *list(config.get("counterfactual_post_remove_views", ["close_high", "right"])),
    ]
    protocol_value = config.get("final_evaluation_protocol")
    if protocol_value:
        protocol = Path(protocol_value)
        protocol = protocol if protocol.is_absolute() else ROOT / protocol
        command.extend(
            [
                "--final-evaluation-authorized",
                "--protocol",
                str(protocol),
            ]
        )
    else:
        command.append("--force-remove-cover-for-observation-calibration")
    if config.get("disable_manipulation_video", False):
        command.append("--disable-manipulation-video")
    if config.get("physics_only_manipulation_steps", False):
        command.append("--physics-only-manipulation-steps")
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": gpu,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NVIDIA_VISIBLE_DEVICES": gpu,
            "PHYSICAL_GPU": gpu,
            "EFFICIENT_ROBOTICS_ISAACSIM_VENV": environment.get(
                "EFFICIENT_ROBOTICS_ISAACSIM_VENV", str(ROOT / ".venv-isaac")
            ),
            "EFFICIENT_ROBOTICS_LIBERO_ROOT": environment.get(
                "EFFICIENT_ROBOTICS_LIBERO_ROOT", str(ROOT / "third_party/LIBERO")
            ),
        }
    )
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)

    started = time.perf_counter()
    with log_path.open("a", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=4000.0,
            check=False,
        )
    elapsed = time.perf_counter() - started
    found = completed_run(config, assignment)
    if found is None:
        state.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "returncode": completed.returncode,
                "runtime_seconds": elapsed,
                "log": str(log_path.resolve()),
            }
        )
        write_json_atomic(status_path, state)
        raise RuntimeError(f"Scenario validation seed {args.seed} failed; see {log_path}")

    run_dir, result = found
    state.update(
        {
            "status": "completed" if completed.returncode == 0 else "completed_with_teardown_issue",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": completed.returncode,
            "runtime_seconds": elapsed,
            "reported_runtime_seconds": result.get("runtime_seconds"),
            "run_dir": str(run_dir.resolve()),
            "result_sha256": sha256(run_dir / "cover_removal_result.json"),
            "valid_cache_artifacts": True,
            "reserved_test_seeds_used": bool(
                config.get("reserved_test_seeds_used", False)
            ),
            "valid_for_final_counterfactual_evaluation": bool(
                config.get("testing_performed", False)
            ),
            "log": str(log_path.resolve()),
        }
    )
    write_json_atomic(status_path, state)
    print(f"SCENARIO_CAPTURE={args.seed}:{state['status']}:{run_dir}")


if __name__ == "__main__":
    main()
