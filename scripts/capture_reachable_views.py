"""Validate live UR10e right and close-high reobservation motions without Qwen."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from run_closed_loop_pipeline import (
    ISAAC_PYTHON,
    ROOT,
    wait_for_path,
    write_json_atomic,
)
from single_gpu_runtime import require_single_gpu_policy
from single_gpu_runtime import configured_physical_gpu
from final_evaluation_authorization import (
    DEFAULT_PROTOCOL,
    validate_output_authorization,
)
from scanned_basket_scene import ALL_CALIBRATION_SCENE_VARIANTS


OUTPUT_ROOT = ROOT / "outputs" / "live_pipeline" / "reachable_view_capture"


def transport_or_saved_result_succeeded(
    returncode: int | None, server_result: dict
) -> bool:
    """Accept a complete server result when Kit stalls only during shutdown."""
    return bool(
        returncode == 0 or server_result.get("status") == "completed"
    )


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


def target_visibility_measurements(
    output_dir: Path, sequence: list[str]
) -> dict:
    """Measure target pixels from simulator-generated instance masks."""
    measurements = {}
    for view_id in sequence:
        view_dir = output_dir / "observations" / view_id
        labels = json.loads(
            (view_dir / "instance_labels.json").read_text(encoding="utf-8")
        )
        target_ids = [
            int(instance_id)
            for instance_id, label in labels.items()
            if label.get("class") == "target_red"
        ]
        instance_ids = np.load(view_dir / "instance_ids.npy")
        measurement = {
            "target_instance_ids": target_ids,
            "target_visible_pixel_count": int(
                np.isin(instance_ids, target_ids).sum()
            ),
            "image_pixel_count": int(instance_ids.size),
        }
        objective_occlusion_path = (
            view_dir / "objective_occlusion.json"
        )
        if objective_occlusion_path.is_file():
            measurement["objective_occlusion"] = json.loads(
                objective_occlusion_path.read_text(encoding="utf-8")
            )
        objective_reference_occlusion_path = (
            view_dir / "objective_reference_occlusion.json"
        )
        if objective_reference_occlusion_path.is_file():
            measurement["objective_reference_occlusion"] = json.loads(
                objective_reference_occlusion_path.read_text(
                    encoding="utf-8"
                )
            )
        objective_behind_path = (
            view_dir / "objective_camera_relative_behind.json"
        )
        if objective_behind_path.is_file():
            measurement["objective_camera_relative_behind"] = json.loads(
                objective_behind_path.read_text(encoding="utf-8")
            )
        measurements[view_id] = measurement
    return measurements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute-grasp",
        action="store_true",
        help=(
            "After the two actual re-observation motions, reuse the same "
            "composite articulation for the validated seed-0 contact grasp."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--scanned-basket-scene",
        action="store_true",
        help=(
            "Use the procedural household mugs with the locally available "
            "textured LIBERO scanned basket."
        ),
    )
    parser.add_argument(
        "--active-occlusion-scene",
        action="store_true",
        help="Use the controlled center-occlusion scanned-basket layout.",
    )
    parser.add_argument(
        "--calibration-scene-variant",
        choices=("auto", *ALL_CALIBRATION_SCENE_VARIANTS),
        help="Use one deterministic factorized calibration scene variant.",
    )
    parser.add_argument(
        "--basket-contact-physics",
        action="store_true",
        help=(
            "Add the static five-box scanned-basket collision approximation."
        ),
    )
    parser.add_argument(
        "--requested-views",
        nargs="+",
        choices=("right", "close_high"),
        help=(
            "View actions to execute after center. Defaults to the original "
            "right then close_high validation sequence."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional explicit session directory under outputs/live_pipeline.",
    )
    parser.add_argument(
        "--startup-timeout-seconds",
        type=float,
        default=1200.0,
        help=(
            "Maximum wait for the first observation. Concurrent Isaac Sim "
            "startup on the shared host can take several minutes."
        ),
    )
    parser.add_argument("--final-evaluation-authorized", action="store_true")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.startup_timeout_seconds <= 0:
        raise ValueError("--startup-timeout-seconds must be positive")
    if (
        args.active_occlusion_scene
        and not args.scanned_basket_scene
    ):
        raise ValueError(
            "--active-occlusion-scene requires "
            "--scanned-basket-scene"
        )
    if (
        args.basket_contact_physics
        and not args.scanned_basket_scene
    ):
        raise ValueError(
            "--basket-contact-physics requires "
            "--scanned-basket-scene"
        )
    if (
        args.calibration_scene_variant is not None
        and not args.scanned_basket_scene
    ):
        raise ValueError(
            "--calibration-scene-variant requires "
            "--scanned-basket-scene"
        )
    if (
        args.active_occlusion_scene
        and args.calibration_scene_variant is not None
    ):
        raise ValueError(
            "--active-occlusion-scene and --calibration-scene-variant "
            "are mutually exclusive"
        )
    if args.execute_grasp and args.seed != 0:
        raise ValueError(
            "--execute-grasp currently supports only the validated seed 0 "
            "terminal path; use the automatic-IK physics runner for new seeds"
        )
    requested_views = args.requested_views or ["right", "close_high"]
    if args.execute_grasp and requested_views != ["right", "close_high"]:
        raise ValueError(
            "--execute-grasp requires the validated right then close_high sequence"
        )
    require_single_gpu_policy()
    physical_gpu = configured_physical_gpu()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else next_output_dir(args.seed)
    )
    validate_output_authorization(
        seed=args.seed,
        output_dir=output_dir,
        final_evaluation_authorized=args.final_evaluation_authorized,
        protocol_path=args.protocol,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    base_config = json.loads(
        (
            ROOT
            / "configs"
            / "sim"
            / "closed_loop_execution.json"
        ).read_text(encoding="utf-8")
    )
    config = copy.deepcopy(base_config)
    config["viewpoint_execution"]["mode"] = "interpolated_joint_physics"
    config["viewpoint_execution"]["debug_ee_positions_world_m"] = {}
    config_path = output_dir / "effective_method_config.json"
    write_json_atomic(config_path, config)

    stdout_path = output_dir / "isaac_stdout.log"
    stderr_path = output_dir / "isaac_stderr.log"
    command = [
        str(ISAAC_PYTHON),
        str(ROOT / "scripts" / "isaac_sim_server.py"),
        "--scene-profile",
        "benchmark",
        "--headless",
        "--renderer-gpu",
        str(physical_gpu),
        "--physics-gpu",
        "0",
        "--live-pipeline-server",
        "--actual-view-motion",
        "--live-session-dir",
        str(output_dir),
        "--method-config",
        str(config_path),
        "--seed",
        str(args.seed),
    ]
    if args.scanned_basket_scene:
        command.extend(
            [
                "--tabletop-scene",
                "--scanned-basket-scene",
            ]
        )
    if args.active_occlusion_scene:
        command.append("--active-occlusion-scene")
    if args.calibration_scene_variant is not None:
        command.extend(
            [
                "--calibration-scene-variant",
                args.calibration_scene_variant,
            ]
        )
    if args.basket_contact_physics:
        command.append("--basket-contact-physics")
    if args.final_evaluation_authorized:
        command.extend(
            [
                "--final-evaluation-authorized",
                "--final-evaluation-protocol",
                str(args.protocol.resolve()),
            ]
        )
    if args.execute_grasp:
        command.append("--execute-persistent-composite-grasp")
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_stream, (
        stderr_path.open("w", encoding="utf-8")
    ) as stderr_stream:
        server = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stdout_stream,
            stderr=stderr_stream,
            text=True,
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": str(physical_gpu),
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "NVIDIA_VISIBLE_DEVICES": str(physical_gpu),
                "PHYSICAL_GPU": str(physical_gpu),
            },
        )
        try:
            requests = tuple(
                f"viewpoint_{view}" for view in requested_views
            ) + (("grasp" if args.execute_grasp else "stop"),)
            for index, action_type in enumerate(requests):
                wait_for_path(
                    output_dir / f"observation_ready_{index:03d}.json",
                    server,
                    timeout=(
                        args.startup_timeout_seconds
                        if index == 0
                        else 180.0
                    ),
                )
                write_json_atomic(
                    output_dir / f"action_request_{index:03d}.json",
                    {
                        "schema_version": "reachable-view-request-v1",
                        "index": index,
                        "type": action_type,
                        "selected_candidate": "target_red",
                        "source_decision": "validation_without_vlm",
                    },
                )
            wait_for_path(
                output_dir / "server_result.json",
                server,
                timeout=600.0 if args.execute_grasp else 300.0,
            )
            server.wait(timeout=60.0)
        finally:
            if server.poll() is None:
                server.terminate()
                server.wait(timeout=30.0)

    server_result = json.loads(
        (output_dir / "server_result.json").read_text(encoding="utf-8")
    )
    trajectories = [
        event["trajectory"] for event in server_result["events"][1:]
    ]
    success = (
        transport_or_saved_result_succeeded(server.returncode, server_result)
        and len(trajectories) == len(requested_views)
        and all(item["status"] == "completed" for item in trajectories)
        and all(item["actual_robot_motion_executed"] for item in trajectories)
        and all(item["collision_checked"] for item in trajectories)
        and (
            not args.execute_grasp
            or (
                server_result.get("grasp_executed")
                and server_result.get("grasp_execution", {}).get(
                    "lift_verified"
                )
            )
        )
    )
    result = {
        "schema_version": "reachable-view-capture-v1",
        "status": "completed" if success else "failed",
        "runtime_seconds": time.perf_counter() - started,
        "transport_returncode": server.returncode,
        "transport_exit_warning": bool(
            server.returncode not in (0, None)
            and server_result.get("status") == "completed"
        ),
        "sequence": ["center", *requested_views],
        "seed": args.seed,
        "trajectories": trajectories,
        "gpu_policy": {
            "physical_gpu": physical_gpu,
            "renderer_active_gpu": physical_gpu,
            "physics_cuda_device": 0,
            "multi_gpu": False,
        },
        "qwen_loaded": False,
        "grasp_requested": args.execute_grasp,
        "grasp_execution": server_result.get("grasp_execution"),
        "training_performed": False,
        "testing_performed": bool(args.final_evaluation_authorized),
        "reserved_test_seeds_used": bool(args.final_evaluation_authorized),
        "valid_for_final_evaluation": bool(args.final_evaluation_authorized),
    }
    if args.calibration_scene_variant is not None:
        from scanned_basket_scene import validate_calibration_visibility

        household_scene = json.loads(
            (output_dir / "household_scene.json").read_text(
                encoding="utf-8"
            )
        )
        calibration_ground_truth = household_scene[
            "calibration_ground_truth"
        ]
        visibility_measurements = target_visibility_measurements(
            output_dir, result["sequence"]
        )
        calibration_ground_truth["observed_target_visibility"] = (
            visibility_measurements
        )
        calibration_ground_truth[
            "objective_occlusion_ground_truth"
        ] = {
            view_id: measurement.get("objective_occlusion")
            for view_id, measurement in visibility_measurements.items()
        }
        calibration_ground_truth[
            "objective_reference_occlusion_ground_truth"
        ] = {
            view_id: measurement.get(
                "objective_reference_occlusion"
            )
            for view_id, measurement in visibility_measurements.items()
        }
        calibration_ground_truth[
            "objective_camera_relative_behind_ground_truth"
        ] = {
            view_id: measurement.get(
                "objective_camera_relative_behind"
            )
            for view_id, measurement in visibility_measurements.items()
        }
        visibility_validation = validate_calibration_visibility(
            calibration_ground_truth["variant"],
            visibility_measurements,
        )
        calibration_ground_truth["visibility_validation"] = (
            visibility_validation
        )
        write_json_atomic(
            output_dir / "calibration_ground_truth.json",
            calibration_ground_truth,
        )
        result["calibration_scene_variant"] = (
            calibration_ground_truth["variant"]
        )
        result["calibration_ground_truth_file"] = (
            "calibration_ground_truth.json"
        )
        result["calibration_visibility_validation"] = (
            visibility_validation
        )
        success = success and visibility_validation["passed"]
        result["status"] = "completed" if success else "failed"
    write_json_atomic(output_dir / "result.json", result)
    if not success:
        raise RuntimeError(f"Reachable-view capture failed: {result}")
    print(f"REACHABLE_VIEW_CAPTURE={output_dir / 'result.json'}")


if __name__ == "__main__":
    main()
