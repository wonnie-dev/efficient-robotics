"""Run external Qwen replanning against one persistent Isaac process.

Atomic JSON files form the handoff boundary, while Isaac retains ownership of
the stage, articulation, cameras, and rigid-body state for the full episode.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path

from export_vlm_dataset import ANONYMOUS_IDS, export_view
from rgbd_target_localization import localize_mask_files
from single_gpu_runtime import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_MODEL,
    cached_inference,
    configured_physical_gpu,
    evaluate_debug_only,
    fuse_beliefs,
    output_belief,
    require_single_gpu_policy,
    select_action,
)
from run_non_oracle_hybrid_planner import plan as make_belief_mpc_plan
from qwen_belief_adapter import (
    normalize as normalize_debug_belief,
    prepare_replan_config,
    qwen_to_planner_belief,
    weighted_log_belief_update,
)


ROOT = Path(__file__).resolve().parents[1]
LIVE_ROOT = ROOT / "outputs" / "live_pipeline"
ISAAC_PYTHON = Path(
    os.environ.get(
        "EFFICIENT_ROBOTICS_ISAAC_PYTHON",
        str(
            Path(
                os.environ.get(
                    "EFFICIENT_ROBOTICS_ISAACSIM_VENV",
                    ROOT / ".venv-isaac",
                )
            )
            / "bin"
            / "python"
        ),
    )
)
AVAILABLE_VIEWS = {"center", "close_high", "right", "left", "overhead"}


def next_session_dir(seed: int) -> Path:
    """Allocate the next run directory for a deterministic seed."""
    LIVE_ROOT.mkdir(parents=True, exist_ok=True)
    prefix = f"benchmark_seed{seed:03d}_run"
    indices = []
    for path in LIVE_ROOT.glob(prefix + "*"):
        try:
            indices.append(int(path.name.removeprefix(prefix)))
        except ValueError:
            continue
    return LIVE_ROOT / f"{prefix}{max(indices, default=0) + 1:03d}"


def write_json_atomic(path: Path, payload: dict) -> None:
    """Publish one complete IPC message without exposing a partial JSON file."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


def wait_for_path(path: Path, process: subprocess.Popen, timeout: float) -> None:
    """Wait for an IPC message while also watching the Isaac server lifecycle."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            return
        returncode = process.poll()
        if returncode is not None:
            raise RuntimeError(
                f"Isaac server exited with code {returncode} before {path}"
            )
        time.sleep(0.1)
    raise TimeoutError(f"Timed out waiting for {path}")


def main() -> None:
    """Run one cached-or-live single-GPU perception and replanning episode."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--allow-cache-miss-inference", action="store_true")
    parser.add_argument("--maximum-observations", type=int, default=4)
    parser.add_argument(
        "--belief-mpc-config",
        type=Path,
        help=(
            "Use action-conditioned future-belief planning and the tempered "
            "Qwen belief adapter instead of the legacy threshold controller."
        ),
    )
    parser.add_argument(
        "--execute-contact-grasp",
        action="store_true",
        help=(
            "After a live terminal grasp decision, run the validated actual-RG6 "
            "PhysX bilateral-contact grasp and lift in a fresh single-GPU process."
        ),
    )
    parser.add_argument(
        "--execute-persistent-composite-grasp",
        action="store_true",
        help=(
            "After a live terminal grasp decision, execute the validated "
            "UR10e+RG6 PhysX grasp inside the same persistent Isaac process "
            "and stage used for live observations."
        ),
    )
    parser.add_argument(
        "--actual-view-motion",
        action="store_true",
        help=(
            "Execute right and close_high reobservations with the official "
            "UR10e articulation instead of fixed debug camera coordinates. "
            "The unvalidated overhead action is disabled."
        ),
    )
    args = parser.parse_args()
    require_single_gpu_policy()
    physical_gpu = configured_physical_gpu()
    if physical_gpu < 0:
        raise ValueError("PHYSICAL_GPU must be non-negative")
    if args.execute_contact_grasp and args.execute_persistent_composite_grasp:
        raise ValueError(
            "Select either the fresh-process or persistent-process grasp, not both"
        )
    if args.seed < 0:
        raise ValueError("Seed must be non-negative")
    if args.maximum_observations < 1:
        raise ValueError("maximum-observations must be positive")
    if not ISAAC_PYTHON.is_file():
        raise FileNotFoundError(ISAAC_PYTHON)
    belief_mpc_config = (
        json.loads(args.belief_mpc_config.read_text(encoding="utf-8"))
        if args.belief_mpc_config is not None
        else None
    )
    if args.actual_view_motion and belief_mpc_config is None:
        raise ValueError("--actual-view-motion requires --belief-mpc-config")
    if args.actual_view_motion:
        belief_mpc_config = copy.deepcopy(belief_mpc_config)
        viewpoint_execution = belief_mpc_config.setdefault(
            "viewpoint_execution", {}
        )
        viewpoint_execution["mode"] = "interpolated_joint_physics"
        viewpoint_execution["debug_ee_positions_world_m"] = {}
        viewpoint_execution["status"] = (
            "simulation_actual_ur10e_joint_motion_with_conservative_aabb_abort"
        )
        overhead_action = belief_mpc_config["actions"]["viewpoint_overhead"]
        overhead_action["enabled"] = False
        overhead_action.pop(
            "enable_after_completed_reobservations", None
        )
        overhead_action["disabled_until"] = (
            "actual_collision_checked_overhead_ik_is_validated"
        )
        close_high_action = belief_mpc_config["actions"][
            "viewpoint_close_high"
        ]
        close_high_action["enabled"] = False
        close_high_action["enable_after_completed_reobservations"] = 1
        close_high_action["disabled_until"] = (
            "one_actual_right_reobservation_is_completed"
        )
    if belief_mpc_config is not None:
        adapter_config = belief_mpc_config["qwen_belief_adapter"]
        raw_logit_temperature = float(
            adapter_config["raw_logit_temperature"]
        )
        observation_log_weight = float(
            adapter_config["observation_log_weight"]
        )

    session_dir = next_session_dir(args.seed)
    session_dir.mkdir(parents=True, exist_ok=False)
    effective_method_config_path = (
        args.belief_mpc_config
        if args.belief_mpc_config is not None
        else ROOT / "configs" / "research" / "initial_method_design.json"
    )
    if args.actual_view_motion:
        effective_method_config_path = (
            session_dir / "effective_method_config.json"
        )
        write_json_atomic(
            effective_method_config_path, belief_mpc_config
        )
    stdout_path = session_dir / "isaac_stdout.log"
    stderr_path = session_dir / "isaac_stderr.log"
    stdout_stream = stdout_path.open("w", encoding="utf-8")
    stderr_stream = stderr_path.open("w", encoding="utf-8")
    # Vulkan uses the host's physical renderer index. Once that same device is
    # the only CUDA-visible GPU, PhysX addresses it with the local ordinal 0.
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
        "--live-session-dir",
        str(session_dir),
        "--method-config",
        str(effective_method_config_path.resolve()),
        "--seed",
        str(args.seed),
    ]
    if args.execute_persistent_composite_grasp:
        command.append("--execute-persistent-composite-grasp")
    if args.actual_view_motion:
        command.append("--actual-view-motion")
    started = time.perf_counter()
    # Keep this process alive across every observation and terminal action so
    # no scene or articulation state has to be reconstructed from files.
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
        },
    )

    steps = []
    belief = (
        {
            "target": normalize_debug_belief(
                belief_mpc_config["initial_belief"]["target"]
            ),
            "relation": normalize_debug_belief(
                belief_mpc_config["initial_belief"]["relation"]
            ),
        }
        if belief_mpc_config is not None
        else None
    )
    visited_views: set[str] = set()
    terminal_action = None
    try:
        for observation_index in range(args.maximum_observations):
            event_path = (
                session_dir / f"observation_ready_{observation_index:03d}.json"
            )
            wait_for_path(event_path, server, timeout=180.0)
            event = json.loads(event_path.read_text(encoding="utf-8"))
            view = event["view"]
            visited_views.add(view)
            input_path, ground_truth_path = export_view(
                view,
                session_dir / "vlm_dataset",
                observation_root=session_dir / "observations",
                episode_id=session_dir.name,
            )
            inference = cached_inference(
                input_path,
                args.cache_root.resolve(),
                args.model_path.resolve(),
                args.max_pixels,
                args.allow_cache_miss_inference,
            )
            planner_result = None
            if belief_mpc_config is not None:
                observation_belief = qwen_to_planner_belief(
                    inference["output"], raw_logit_temperature
                )
                belief_before_update = json.loads(json.dumps(belief))
                belief = weighted_log_belief_update(
                    belief,
                    observation_belief,
                    observation_log_weight,
                )
                planner_config = prepare_replan_config(
                    belief_mpc_config,
                    belief,
                    completed_reobservations=max(0, observation_index),
                    visited_views=visited_views,
                )
                planner_result = make_belief_mpc_plan(planner_config)
                action_type = planner_result["action_request"]["type"]
                action = {
                    "type": action_type,
                    "reason": planner_result["action_request"]["reason"],
                    "selected_candidate": max(
                        belief["target"], key=belief["target"].get
                    ),
                    "commitment_gate": planner_result["commitment_gate"],
                    "valid_for_final_evaluation": False,
                }
            else:
                observation_belief = output_belief(inference["output"])
                belief_before_update = None
                belief = (
                    observation_belief
                    if belief is None
                    else fuse_beliefs(belief, observation_belief)
                )
                action = select_action(
                    belief,
                    current_view=view,
                    available_views=AVAILABLE_VIEWS,
                    visited_views=visited_views,
                )
            if (
                observation_index == args.maximum_observations - 1
                and action["type"].startswith("viewpoint_")
            ):
                action = {
                    **action,
                    "type": "stop",
                    "reason": "maximum_observations_reached",
                }
            terminal_localization_path = None
            if (
                action["type"] == "grasp"
                and args.execute_persistent_composite_grasp
                and args.actual_view_motion
            ):
                selected_semantic = action.get("selected_candidate")
                if selected_semantic != "target_red":
                    raise RuntimeError(
                        "The current physical target prim supports only "
                        f"target_red, but Qwen selected {selected_semantic!r}"
                    )
                selected_anonymous = ANONYMOUS_IDS[selected_semantic]
                model_input = json.loads(
                    input_path.read_text(encoding="utf-8")
                )
                candidates_by_id = {
                    candidate["candidate_id"]: candidate
                    for candidate in model_input["candidates"]
                }
                required_ids = {selected_anonymous, "object_002"}
                if not required_ids.issubset(candidates_by_id):
                    raise RuntimeError(
                        "Selected target or occluder candidate mask is "
                        f"unavailable: {required_ids}"
                    )

                def candidate_mask(candidate_id: str) -> Path:
                    path = (
                        ROOT
                        / candidates_by_id[candidate_id]["mask_path"]
                    ).resolve()
                    if not path.is_relative_to(ROOT):
                        raise ValueError(
                            f"Candidate mask escapes project root: {path}"
                        )
                    return path

                # The selected mask and metric depth come from the same saved
                # wrist observation, preserving their pixel and camera-frame
                # alignment for backprojection.
                terminal_localization = localize_mask_files(
                    session_dir / "observations" / view,
                    {
                        "selected_target": candidate_mask(
                            selected_anonymous
                        ),
                        "occluder_orange": candidate_mask("object_002"),
                    },
                )
                terminal_localization["selection"] = {
                    "planner_candidate": selected_semantic,
                    "anonymous_candidate_id": selected_anonymous,
                    "source_qwen_output": inference["cache_dir"],
                    "source_view": view,
                }
                terminal_localization_path = (
                    session_dir / "terminal_rgbd_localization.json"
                )
                write_json_atomic(
                    terminal_localization_path, terminal_localization
                )
                action["rgbd_localization_path"] = str(
                    terminal_localization_path
                )
            step = {
                "index": observation_index,
                "view": view,
                "event": event,
                "input_path": str(input_path),
                "ground_truth_path": str(ground_truth_path),
                "cache_key": inference["cache_key"],
                "cache_dir": inference["cache_dir"],
                "cache_hit": inference["cache_hit"],
                "cache_source": inference["cache_source"],
                "metrics": inference["metrics"],
                "vlm_output": inference["output"],
                "qwen_observation_belief": observation_belief,
                "belief_before_update": belief_before_update,
                "belief_after_update": belief,
                "belief_update": (
                    {
                        "method": (
                            "weighted_log_space_generalized_bayes_debug_update"
                        ),
                        "raw_logit_temperature": raw_logit_temperature,
                        "observation_log_weight": observation_log_weight,
                        "calibrated": False,
                    }
                    if belief_mpc_config is not None
                    else {"method": "legacy_uncalibrated_product_fusion"}
                ),
                "future_belief_plan": planner_result,
                "selected_action": action,
            }
            steps.append(step)
            write_json_atomic(
                session_dir / f"decision_{observation_index:03d}.json",
                step,
            )
            write_json_atomic(
                session_dir / f"action_request_{observation_index:03d}.json",
                {
                    "schema_version": "live-action-request-v1",
                    "index": observation_index,
                    "type": action["type"],
                    "selected_candidate": action.get("selected_candidate"),
                    "rgbd_localization_path": (
                        str(terminal_localization_path)
                        if terminal_localization_path is not None
                        else None
                    ),
                    "source_decision": str(
                        session_dir / f"decision_{observation_index:03d}.json"
                    ),
                },
            )
            print(
                f"LIVE_STEP={observation_index} VIEW={view} "
                f"ACTION={action['type']}",
                flush=True,
            )
            if not action["type"].startswith("viewpoint_"):
                terminal_action = action["type"]
                break
        else:
            terminal_action = "stop"

        # server_result is written only after the persistent process has
        # applied the final request and finished any requested manipulation.
        server_result_path = session_dir / "server_result.json"
        wait_for_path(
            server_result_path,
            server,
            timeout=(
                600.0
                if args.execute_persistent_composite_grasp
                else 120.0
            ),
        )
        server.wait(timeout=60.0)
        if server.returncode != 0:
            raise RuntimeError(f"Isaac server failed with code {server.returncode}")

        server_result = json.loads(
            server_result_path.read_text(encoding="utf-8")
        )
        grasp_execution = (
            server_result.get("grasp_execution")
            if args.execute_persistent_composite_grasp
            else None
        )
        if args.execute_persistent_composite_grasp and terminal_action == "grasp":
            if (
                not grasp_execution
                or grasp_execution.get("status") != "completed"
                or not grasp_execution.get("lift_verified")
            ):
                raise RuntimeError(
                    "Persistent terminal contact grasp did not verify a lift"
                )
        if terminal_action == "grasp" and args.execute_contact_grasp:
            grasp_root = session_dir / "grasp_execution"
            grasp_stdout_path = session_dir / "grasp_stdout.log"
            grasp_stderr_path = session_dir / "grasp_stderr.log"
            grasp_command = [
                str(ISAAC_PYTHON),
                str(ROOT / "scripts" / "execute_contact_grasp.py"),
                "--headless",
                "--renderer-gpu",
                str(physical_gpu),
                "--physics-gpu",
                "0",
                "--seed",
                str(args.seed),
                "--output-root",
                str(grasp_root),
            ]
            grasp_started = time.perf_counter()
            with grasp_stdout_path.open(
                "w", encoding="utf-8"
            ) as grasp_stdout, grasp_stderr_path.open(
                "w", encoding="utf-8"
            ) as grasp_stderr:
                completed_grasp = subprocess.run(
                    grasp_command,
                    cwd=ROOT,
                    stdout=grasp_stdout,
                    stderr=grasp_stderr,
                    text=True,
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": str(physical_gpu),
                        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                        "NVIDIA_VISIBLE_DEVICES": str(physical_gpu),
                        "PHYSICAL_GPU": str(physical_gpu),
                    },
                    timeout=600.0,
                    check=False,
                )
            grasp_result_path = grasp_root / "result.json"
            if not grasp_result_path.is_file():
                raise RuntimeError(
                    "Contact grasp process did not produce result.json; "
                    f"returncode={completed_grasp.returncode}"
                )
            grasp_execution = json.loads(
                grasp_result_path.read_text(encoding="utf-8")
            )
            grasp_execution["orchestration_runtime_seconds"] = (
                time.perf_counter() - grasp_started
            )
            grasp_execution["trigger"] = {
                "terminal_action": terminal_action,
                "selected_candidate": steps[-1]["selected_action"].get(
                    "selected_candidate"
                ),
            }
            if (
                completed_grasp.returncode != 0
                or grasp_execution.get("status") != "completed"
                or not grasp_execution.get("lift_verified")
            ):
                raise RuntimeError(
                    "Terminal contact grasp failed: "
                    f"returncode={completed_grasp.returncode}, "
                    f"result={grasp_execution.get('status')}"
                )

        debug_results = [
            evaluate_debug_only(
                step["vlm_output"],
                Path(step["input_path"]),
            )
            for step in steps
        ]
        result = {
            "schema_version": "live-single-gpu-pipeline-v1",
            "status": "completed",
            "purpose": "live_pipeline_validation_only_not_final_evaluation",
            "session_dir": str(session_dir),
            "seed": args.seed,
            "steps": steps,
            "terminal_action": terminal_action,
            "server_result": server_result,
            "debug_ground_truth_after_planning": debug_results,
            "runtime_seconds": time.perf_counter() - started,
            "gpu_policy": {
                "physical_gpu": physical_gpu,
                "visible_cuda_devices": 1,
                "batch_size": 1,
                "parallel_vlm_jobs": False,
                "distributed": False,
            },
            "training_performed": False,
            "calibration_performed": False,
            "grasp_executed": bool(
                grasp_execution
                and grasp_execution.get("lift_verified")
            ),
            "grasp_execution": grasp_execution,
            "grasp_execution_process": (
                "same_persistent_isaac_process_and_stage_as_live_observation_server"
                if args.execute_persistent_composite_grasp and grasp_execution
                else (
                    "fresh_single_gpu_physics_process_after_live_terminal_action"
                    if grasp_execution
                    else None
                )
            ),
            "grasp_trajectory_source": (
                grasp_execution.get("trajectory_source")
                if grasp_execution
                else None
            ),
            "pre_captured_candidate_replay": False,
            "observation_motion_mode": (
                "actual_ur10e_interpolated_joint_physics_with_aabb_abort"
                if args.actual_view_motion
                else "fixed_debug_wrist_coordinates"
            ),
            "controller_mode": (
                "action_conditioned_future_belief_planner"
                if belief_mpc_config is not None
                else "legacy_confidence_threshold_controller"
            ),
            "valid_for_final_evaluation": False,
        }
        write_json_atomic(session_dir / "pipeline_result.json", result)
        print(f"LIVE_PIPELINE_RESULT={session_dir / 'pipeline_result.json'}")
    except Exception as error:
        write_json_atomic(
            session_dir / "pipeline_error.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
                "steps": steps,
            },
        )
        if server.poll() is None:
            request_index = len(steps)
            write_json_atomic(
                session_dir / f"action_request_{request_index:03d}.json",
                {
                    "schema_version": "live-action-request-v1",
                    "index": request_index,
                    "type": "stop",
                    "reason": "external_pipeline_error",
                },
            )
            try:
                server.wait(timeout=15.0)
            except subprocess.TimeoutExpired:
                server.terminate()
                server.wait(timeout=15.0)
        raise
    finally:
        stdout_stream.close()
        stderr_stream.close()


if __name__ == "__main__":
    main()
