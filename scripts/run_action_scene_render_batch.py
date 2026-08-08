"""Render a bounded development batch of action-differentiating scenes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from scanned_basket_scene import ACTION_DIFFERENTIATING_SCENE_VARIANTS


ROOT = Path(__file__).resolve().parents[1]
ISAAC_PYTHON = Path("/data/wonheekoh/isaacsim_venv/bin/python")
DEFAULT_OUTPUT_ROOT = (
    ROOT / "outputs" / "live_pipeline" / "action_differentiating_scene_pilot"
)
RESERVED_TEST_SEEDS = set(range(200, 210))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=187)
    parser.add_argument("--seed-stop-exclusive", type=int, default=197)
    parser.add_argument("--cycle-anchor-seed", type=int, default=185)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    seeds = list(range(args.seed_start, args.seed_stop_exclusive))
    if not seeds or args.seed_start < 0:
        raise ValueError("Expected a nonempty nonnegative seed range")
    overlap = sorted(set(seeds) & RESERVED_TEST_SEEDS)
    if overlap:
        raise ValueError(f"Reserved test seeds are forbidden: {overlap}")
    physical_gpu = os.environ.get("PHYSICAL_GPU")
    if physical_gpu is None or os.environ.get("CUDA_VISIBLE_DEVICES") != physical_gpu:
        raise RuntimeError("PHYSICAL_GPU and CUDA_VISIBLE_DEVICES must match")

    output_root = args.output_root.resolve()
    allowed_root = (ROOT / "outputs" / "live_pipeline").resolve()
    if not output_root.is_relative_to(allowed_root):
        raise ValueError(f"Output root must stay below {allowed_root}")
    batch_root = output_root / f"batch_seed{seeds[0]:03d}_{seeds[-1]:03d}"
    batch_root.mkdir(parents=True, exist_ok=True)
    status_path = batch_root / "status.json"
    status: dict[str, Any] = {
        "schema_version": "action-scene-render-batch-v1",
        "status": "running",
        "seed_start": seeds[0],
        "seed_stop_inclusive": seeds[-1],
        "cycle_anchor_seed": args.cycle_anchor_seed,
        "variant_cycle": list(ACTION_DIFFERENTIATING_SCENE_VARIANTS),
        "physical_gpu": int(physical_gpu),
        "renderer_active_gpu": int(physical_gpu),
        "physics_cuda_device": 0,
        "multi_gpu": False,
        "scenes": [],
        "training_performed": False,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
    }
    write_json(status_path, status)
    batch_started = time.perf_counter()

    for seed in seeds:
        variant = ACTION_DIFFERENTIATING_SCENE_VARIANTS[
            (seed - args.cycle_anchor_seed)
            % len(ACTION_DIFFERENTIATING_SCENE_VARIANTS)
        ]
        seed_root = output_root / variant / f"seed{seed:03d}"
        run_index = 1
        while (seed_root / f"run{run_index:03d}").exists():
            run_index += 1
        run_dir = seed_root / f"run{run_index:03d}"
        command = [
            str(ISAAC_PYTHON),
            str(ROOT / "scripts" / "run_actual_view_motion_smoke.py"),
            "--seed",
            str(seed),
            "--scanned-basket-perception-pilot",
            "--calibration-scene-variant",
            variant,
            "--basket-collision-physics-pilot",
            "--requested-views",
            "close_high",
            "right",
            "--output-dir",
            str(run_dir),
        ]
        started = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=dict(os.environ),
            capture_output=True,
            text=True,
            check=False,
        )
        stdout_path = batch_root / f"seed{seed:03d}_stdout.log"
        stderr_path = batch_root / f"seed{seed:03d}_stderr.log"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        smoke_path = run_dir / "smoke_result.json"
        smoke = (
            json.loads(smoke_path.read_text(encoding="utf-8"))
            if smoke_path.is_file()
            else None
        )
        visibility = (
            smoke.get("calibration_visibility_validation")
            if smoke is not None
            else None
        )
        scene_status = {
            "seed": seed,
            "variant": variant,
            "run_dir": str(run_dir),
            "command_returncode": completed.returncode,
            "runtime_seconds": time.perf_counter() - started,
            "smoke_result_present": smoke is not None,
            "robot_motion_completed": (
                smoke is not None and smoke.get("status") == "completed"
            ),
            "visibility_validation_passed": (
                bool(visibility.get("passed"))
                if visibility is not None
                else False
            ),
            "objective_visible_fraction_of_amodal": (
                visibility.get("objective_visible_fraction_of_amodal")
                if visibility is not None
                else None
            ),
            "failure_reasons": (
                visibility.get("failure_reasons", [])
                if visibility is not None
                else ["missing_smoke_result"]
            ),
            "eligible_for_calibration": (
                completed.returncode == 0
                and visibility is not None
                and bool(visibility.get("passed"))
            ),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        }
        status["scenes"].append(scene_status)
        write_json(status_path, status)

    status["status"] = "completed"
    status["runtime_seconds"] = time.perf_counter() - batch_started
    status["scene_count"] = len(status["scenes"])
    status["eligible_scene_count"] = sum(
        item["eligible_for_calibration"] for item in status["scenes"]
    )
    status["failed_scene_count"] = (
        status["scene_count"] - status["eligible_scene_count"]
    )
    write_json(status_path, status)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
