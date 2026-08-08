"""Run a seed-specific RGB-D localization to automatic contact-grasp pilot."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from rgbd_target_localization import localize_observation
from run_single_gpu_pilot import require_single_gpu_policy


ROOT = Path(__file__).resolve().parents[1]
ISAAC_PYTHON = Path("/data/wonheekoh/isaacsim_venv/bin/python")
OUTPUT_ROOT = ROOT / "outputs" / "live_pipeline" / "rgbd_localized_grasp"


def next_output_dir(seed: int) -> Path:
    seed_root = OUTPUT_ROOT / f"seed{seed:03d}"
    seed_root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in seed_root.glob("run*"):
        try:
            indices.append(int(path.name.removeprefix("run")))
        except ValueError:
            continue
    return seed_root / f"run{max(indices, default=0) + 1:03d}"


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def run_logged(command: list[str], stdout_path: Path, stderr_path: Path) -> None:
    with stdout_path.open("w", encoding="utf-8") as stdout_stream, (
        stderr_path.open("w", encoding="utf-8")
    ) as stderr_stream:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "5",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "NVIDIA_VISIBLE_DEVICES": "5",
            },
            stdout=stdout_stream,
            stderr=stderr_stream,
            text=True,
            timeout=600.0,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with return code {completed.returncode}: "
            f"{command}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--maximum-localization-error-m", type=float, default=0.02
    )
    args = parser.parse_args()
    require_single_gpu_policy()
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.maximum_localization_error_m <= 0:
        raise ValueError("maximum localization error must be positive")

    output_dir = next_output_dir(args.seed)
    output_dir.mkdir(parents=True, exist_ok=False)
    capture_dir = output_dir / "capture"
    localization_path = output_dir / "rgbd_localization.json"
    grasp_dir = output_dir / "grasp"
    started = time.perf_counter()
    stages = []

    try:
        capture_started = time.perf_counter()
        run_logged(
            [
                "python",
                str(ROOT / "scripts" / "run_actual_view_motion_smoke.py"),
                "--seed",
                str(args.seed),
                "--output-dir",
                str(capture_dir),
            ],
            output_dir / "capture_stdout.log",
            output_dir / "capture_stderr.log",
        )
        stages.append(
            {
                "stage": "actual_ur10e_rgbd_capture",
                "status": "completed",
                "runtime_seconds": time.perf_counter() - capture_started,
            }
        )

        localization = localize_observation(
            capture_dir / "observations" / "close_high",
            ("target_red", "occluder_orange"),
        )
        scene_layout = json.loads(
            (capture_dir / "scene_layout.json").read_text(encoding="utf-8")
        )
        ground_truth_target = np.asarray(
            scene_layout["physical_positions_world_m"][
                "target_red_settled"
            ],
            dtype=np.float64,
        )
        estimated_target = np.asarray(
            localization["estimates"]["target_red"]["center_world_m"],
            dtype=np.float64,
        )
        error = float(np.linalg.norm(estimated_target - ground_truth_target))
        xy_error = float(
            np.linalg.norm(estimated_target[:2] - ground_truth_target[:2])
        )
        localization["debug_evaluation_after_estimation"] = {
            "simulator_ground_truth_target_world_m": (
                ground_truth_target.tolist()
            ),
            "target_position_error_m": error,
            "target_xy_error_m": xy_error,
            "maximum_allowed_error_m": args.maximum_localization_error_m,
            "passed": error <= args.maximum_localization_error_m,
            "ground_truth_used_by_estimator": False,
        }
        write_json(localization_path, localization)
        stages.append(
            {
                "stage": "masked_rgbd_world_localization",
                "status": (
                    "completed"
                    if error <= args.maximum_localization_error_m
                    else "failed"
                ),
                "target_position_error_m": error,
                "target_xy_error_m": xy_error,
            }
        )
        if error > args.maximum_localization_error_m:
            raise RuntimeError(
                f"RGB-D target error {error:.6f} m exceeds "
                f"{args.maximum_localization_error_m:.6f} m"
            )

        grasp_started = time.perf_counter()
        run_logged(
            [
                str(ISAAC_PYTHON),
                str(
                    ROOT
                    / "scripts"
                    / "run_ur10e_rg6_composite_grasp.py"
                ),
                "--headless",
                "--seed",
                str(args.seed),
                "--same-scene-benchmark",
                "--automatic-ik-smoke",
                "--enable-arm-collisions",
                "--rgbd-localization",
                str(localization_path),
                "--output-root",
                str(grasp_dir),
            ],
            output_dir / "grasp_stdout.log",
            output_dir / "grasp_stderr.log",
        )
        grasp_result = json.loads(
            (grasp_dir / "result.json").read_text(encoding="utf-8")
        )
        grasp_success = (
            grasp_result.get("status") == "completed"
            and grasp_result.get("verified_lift_delta_m", 0.0) >= 0.10
        )
        stages.append(
            {
                "stage": "rgbd_localized_automatic_contact_grasp",
                "status": "completed" if grasp_success else "failed",
                "runtime_seconds": time.perf_counter() - grasp_started,
            }
        )
        if not grasp_success:
            raise RuntimeError(
                f"RGB-D localized grasp failed: {grasp_result.get('status')}"
            )

        result = {
            "schema_version": "rgbd-localized-grasp-pilot-v1",
            "status": "completed",
            "seed": args.seed,
            "runtime_seconds": time.perf_counter() - started,
            "stages": stages,
            "localization": localization,
            "grasp_result": str((grasp_dir / "result.json").resolve()),
            "video": grasp_result.get("video"),
            "qwen_loaded": False,
            "training_performed": False,
            "simulator_ground_truth_used_for_grasp_planning": False,
            "simulator_ground_truth_used_after_estimation_for_debug_metric": True,
            "gpu_policy": {
                "physical_gpu": 5,
                "renderer_active_gpu": 5,
                "physics_cuda_device": 0,
                "multi_gpu": False,
            },
            "valid_for_final_evaluation": False,
        }
        write_json(output_dir / "result.json", result)
        print(f"RGBD_LOCALIZED_GRASP_RESULT={output_dir / 'result.json'}")
    except Exception as error:
        write_json(
            output_dir / "error.json",
            {
                "status": "failed",
                "seed": args.seed,
                "error_type": type(error).__name__,
                "error": str(error),
                "runtime_seconds": time.perf_counter() - started,
                "stages": stages,
            },
        )
        raise


if __name__ == "__main__":
    main()
