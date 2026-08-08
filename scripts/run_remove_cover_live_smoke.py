"""Run one contact-gated UR10e+RG6 cover-removal development smoke."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from run_live_single_gpu_pipeline import (
    ISAAC_PYTHON,
    ROOT,
    wait_for_path,
    write_json_atomic,
)
from run_single_gpu_pilot import configured_physical_gpu
from run_single_gpu_pilot import require_single_gpu_policy
from run_cover_search_belief_mpc import (
    execute_observation_action,
    normalize,
    plan,
    validate_config,
)


OUTPUT_ROOT = ROOT / "outputs" / "live_pipeline" / "remove_cover_physics_smoke"
CALIBRATION_RESULT = (
    ROOT
    / "outputs"
    / "offline_mpc"
    / "scene_conditioned_future_belief_seed185_196"
    / "result.json"
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


def target_pixel_count(observation_dir: Path) -> int:
    labels = json.loads(
        (observation_dir / "instance_labels.json").read_text(encoding="utf-8")
    )
    target_ids = [
        int(instance_id)
        for instance_id, value in labels.items()
        if value.get("class") == "target_red"
    ]
    ids = np.load(observation_dir / "instance_ids.npy")
    return int(np.isin(ids, target_ids).sum())


def learned_post_remove_localization(
    output_dir: Path,
    observation_dir: Path,
    *,
    observation_index: int,
    view: str = "post_remove",
    task_overrides: dict | None = None,
) -> dict:
    """Run sequential learned perception and localize Qwen's selected mask."""
    from rgbd_target_localization import localize_mask_files
    from run_live_learned_scanned_basket_pipeline import (
        resolve_input_asset,
        run_current_observation_perception,
    )

    ranking, ranking_path, perception_config_path, stages = (
        run_current_observation_perception(
            session_dir=output_dir,
            observation_dir=observation_dir,
            view=view,
            step_index=observation_index,
            task_overrides=task_overrides,
        )
    )
    selected_candidate = ranking.get("selected_candidate_id")
    if not selected_candidate:
        raise RuntimeError(
            "Learned post-remove perception did not select a target candidate"
        )
    model_input = json.loads(
        Path(ranking["input_path"]).read_text(encoding="utf-8")
    )
    candidates = {
        candidate["candidate_id"]: candidate
        for candidate in model_input["candidates"]
    }
    if selected_candidate not in candidates:
        raise ValueError(
            "Qwen selected a candidate that is absent from its input: "
            f"{selected_candidate}"
        )
    target_mask = resolve_input_asset(
        model_input,
        candidates[selected_candidate]["mask_path"],
    )
    localization = localize_mask_files(
        observation_dir,
        {"selected_target": target_mask},
    )
    localization.update(
        {
            "selection": {
                "selected_candidate_id": selected_candidate,
                "source_ranking": str(ranking_path.resolve()),
                "source_perception_config": str(
                    perception_config_path.resolve()
                ),
                "source_view": view,
                "mask_path": str(target_mask.resolve()),
                "simulator_ground_truth_used": False,
            },
            "perception_stages": stages,
            "learned_perception_executed": True,
            "simulator_ground_truth_used_for_estimate": False,
        }
    )
    localization_name = (
        "post_remove_rgbd_localization.json"
        if view == "post_remove"
        else f"post_remove_{view}_rgbd_localization.json"
    )
    localization_path = output_dir / localization_name
    write_json_atomic(localization_path, localization)
    return {
        "ranking": ranking,
        "ranking_path": str(ranking_path.resolve()),
        "perception_config_path": str(perception_config_path.resolve()),
        "selected_candidate_id": selected_candidate,
        "target_mask_path": str(target_mask.resolve()),
        "localization": localization,
        "localization_path": str(localization_path.resolve()),
        "stages": stages,
    }


def removal_contact_success(removal: dict) -> bool:
    """Use the correct contact contract for hold-only or released covers."""
    if removal.get("cover_placed_and_released"):
        placement = removal.get("supported_placement") or {}
        return bool(
            removal.get("bilateral_contact_before_release")
            and removal.get("contact_maintained_before_release")
            and placement.get("release_executed")
            and placement.get("retreat_executed")
            and placement.get("finger_contact_cleared_after_retreat")
            and placement.get("stable_after_release")
        )
    return bool(removal.get("bilateral_contact_after_lift"))


def contact_grasp_success(server_result: dict) -> bool:
    """Require physical lift and all contact/collision safety gates."""
    grasp = server_result.get("grasp_execution") or {}
    return bool(
        server_result.get("grasp_executed")
        and grasp.get("lift_verified")
        and grasp.get("bilateral_contact_before_lift")
        and not grasp.get("unexpected_environment_pairs")
        and grasp.get("contact_force_within_limit")
        and grasp.get("contact_penetration_within_limit")
    )


def calibration_authorizes_remove_cover(
    seed: int, *, forced_observation_calibration: bool = False
) -> dict:
    if forced_observation_calibration:
        if 200 <= seed <= 209:
            raise ValueError(
                f"Seed {seed} is reserved for final testing and cannot be "
                "used for forced calibration"
            )
        return {
            "authorization_mode": "forced_observation_calibration_intervention",
            "seed": seed,
            "scene_variant": "cover_removal_required",
            "future_observation_used_for_selection": False,
            "testing_performed": False,
            "valid_for_final_evaluation": False,
        }
    result = json.loads(CALIBRATION_RESULT.read_text(encoding="utf-8"))
    matches = [
        fold for fold in result["folds"] if int(fold["held_out_seed"]) == seed
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one calibration fold for seed {seed}")
    fold = matches[0]
    if fold["selected_action"] != "remove_cover":
        raise RuntimeError(
            f"Calibration did not authorize remove_cover for seed {seed}: {fold}"
        )
    return fold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=188)
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    parser.add_argument(
        "--rg6-lid-calibration-config",
        type=Path,
        help="Completed lab-measured RG6/lid calibration file.",
    )
    parser.add_argument(
        "--require-transfer-ready-physics",
        action="store_true",
        help="Refuse provisional RG6/lid physics before launching Isaac Sim.",
    )
    parser.add_argument(
        "--allow-provisional-rg6-lid-physics",
        action="store_true",
        help=(
            "Development only: use an explicit public-spec/simulation proxy "
            "without claiming transfer readiness."
        ),
    )
    parser.add_argument(
        "--rg6-coupling-mode",
        choices=("passive_mimic", "coordinated_drives"),
        default="passive_mimic",
    )
    parser.add_argument(
        "--coordinated-rg6-total-drive-effort-limit-nm",
        type=float,
    )
    parser.add_argument(
        "--replan-after-remove-cover",
        action="store_true",
        help=(
            "Consume the physical post-remove observation, update the "
            "cover-search belief, and send the replanned second action while "
            "the same Isaac process remains alive."
        ),
    )
    parser.add_argument(
        "--execute-post-remove-grasp",
        action="store_true",
        help=(
            "After cover removal, run GroundingDINO, SAM2.1, and Qwen on "
            "the new RGB-D observation, replan, and physically execute the "
            "selected target grasp in the same Isaac process."
        ),
    )
    parser.add_argument(
        "--force-remove-cover-for-observation-calibration",
        action="store_true",
        help=(
            "Calibration only: execute the remove-cover intervention without "
            "claiming that MPC selected it. Reserved test seeds are rejected."
        ),
    )
    args = parser.parse_args()
    if args.timeout_seconds <= 0.0:
        parser.error("--timeout-seconds must be positive")
    if args.execute_post_remove_grasp and not args.replan_after_remove_cover:
        parser.error(
            "--execute-post-remove-grasp requires "
            "--replan-after-remove-cover"
        )
    if args.require_transfer_ready_physics and not args.rg6_lid_calibration_config:
        parser.error(
            "--require-transfer-ready-physics requires "
            "--rg6-lid-calibration-config"
        )
    if (
        args.allow_provisional_rg6_lid_physics
        and not args.rg6_lid_calibration_config
    ):
        parser.error(
            "--allow-provisional-rg6-lid-physics requires "
            "--rg6-lid-calibration-config"
        )
    if (
        args.allow_provisional_rg6_lid_physics
        and args.require_transfer_ready_physics
    ):
        parser.error(
            "Provisional and transfer-ready physics modes are mutually exclusive"
        )
    if args.rg6_coupling_mode == "coordinated_drives" and not (
        args.allow_provisional_rg6_lid_physics
        and args.coordinated_rg6_total_drive_effort_limit_nm is not None
    ):
        parser.error(
            "coordinated_drives requires provisional physics and an explicit "
            "aggregate drive-effort limit"
        )
    if (
        args.rg6_coupling_mode == "passive_mimic"
        and args.coordinated_rg6_total_drive_effort_limit_nm is not None
    ):
        parser.error(
            "coordinated drive-effort limit requires coordinated_drives"
        )
    calibration_path = None
    calibration_report = None
    if args.rg6_lid_calibration_config is not None:
        from rg6_lid_calibration import load_json, validate_calibration

        calibration_path = args.rg6_lid_calibration_config
        if not calibration_path.is_absolute():
            calibration_path = ROOT / calibration_path
        calibration_report = validate_calibration(load_json(calibration_path))
        calibration_authorized = bool(
            calibration_report["transfer_ready"]
            or (
                args.allow_provisional_rg6_lid_physics
                and calibration_report["development_proxy_usable"]
            )
        )
        if not calibration_authorized:
            parser.error(
                "RG6/lid calibration is not transfer-ready: "
                f"{calibration_report}"
            )
    require_single_gpu_policy()
    physical_gpu = configured_physical_gpu()
    fold = calibration_authorizes_remove_cover(
        args.seed,
        forced_observation_calibration=(
            args.force_remove_cover_for_observation_calibration
        ),
    )
    output_dir = next_output_dir(args.seed)
    output_dir.mkdir(parents=True, exist_ok=False)

    base_config = json.loads(
        (
            ROOT / "configs" / "research" / "first_belief_mpc_integration.json"
        ).read_text(encoding="utf-8")
    )
    base_config["viewpoint_execution"]["mode"] = "interpolated_joint_physics"
    base_config["viewpoint_execution"]["debug_ee_positions_world_m"] = {}
    method_path = output_dir / "effective_method_config.json"
    write_json_atomic(method_path, base_config)

    command = [
        str(ISAAC_PYTHON),
        str(ROOT / "scripts" / "open_minimal_scene.py"),
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
        str(method_path),
        "--seed",
        str(args.seed),
        "--household-perception-pilot",
        "--scanned-basket-perception-pilot",
        "--calibration-scene-variant",
        "cover_removal_required",
        "--basket-collision-physics-pilot",
        "--execute-persistent-remove-cover",
    ]
    if calibration_path is not None:
        command.extend(
            ["--rg6-lid-calibration-config", str(calibration_path)]
        )
    if args.require_transfer_ready_physics:
        command.append("--require-transfer-ready-physics")
    if args.allow_provisional_rg6_lid_physics:
        command.append("--allow-provisional-rg6-lid-physics")
    if args.rg6_coupling_mode == "coordinated_drives":
        command.extend(
            [
                "--rg6-coupling-mode",
                "coordinated_drives",
                "--coordinated-rg6-total-drive-effort-limit-nm",
                str(args.coordinated_rg6_total_drive_effort_limit_nm),
            ]
        )
    if args.replan_after_remove_cover:
        command.append("--continue-after-remove-cover")
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(physical_gpu),
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NVIDIA_VISIBLE_DEVICES": str(physical_gpu),
            "PHYSICAL_GPU": str(physical_gpu),
        }
    )
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)

    stdout_path = output_dir / "isaac_stdout.log"
    stderr_path = output_dir / "isaac_stderr.log"
    stdout = stdout_path.open("w", encoding="utf-8")
    stderr = stderr_path.open("w", encoding="utf-8")
    started = time.perf_counter()
    server = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=stdout,
        stderr=stderr,
        text=True,
    )
    try:
        initial_event_path = output_dir / "observation_ready_000.json"
        # Isaac's first RTX/Replicator capture can exceed four minutes on the
        # shared server even when the selected GPU is otherwise idle.  Keep
        # this bounded but separate from the manipulation timeout.
        wait_for_path(
            initial_event_path,
            server,
            timeout=min(args.timeout_seconds, 600.0),
        )
        initial_event = json.loads(initial_event_path.read_text(encoding="utf-8"))
        write_json_atomic(
            output_dir / "action_request_000.json",
            {
                "schema_version": "live-remove-cover-request-v1",
                "index": 0,
                "type": "remove_cover",
                "interface_target": "cover_01",
                "source_calibration_result": (
                    None
                    if args.force_remove_cover_for_observation_calibration
                    else str(CALIBRATION_RESULT)
                ),
                "source_calibration_fold": fold,
                "forced_observation_calibration_intervention": (
                    args.force_remove_cover_for_observation_calibration
                ),
                "future_observation_used_for_selection": False,
            },
        )
        replanning = None
        learned_perception = None
        if args.replan_after_remove_cover:
            post_event_path = output_dir / "observation_ready_001.json"
            wait_for_path(
                post_event_path,
                server,
                timeout=args.timeout_seconds,
            )
            post_event_live = json.loads(
                post_event_path.read_text(encoding="utf-8")
            )
            post_observation_dir = Path(post_event_live["observation_dir"])
            post_pixels_live = target_pixel_count(post_observation_dir)
            if args.execute_post_remove_grasp:
                learned_perception = learned_post_remove_localization(
                    output_dir,
                    post_observation_dir,
                    observation_index=1,
                )
                outcome = "target_detected"
            else:
                outcome = (
                    "target_detected"
                    if post_pixels_live >= 100
                    else "empty_container"
                )
            planner_config = json.loads(
                (
                    ROOT
                    / "configs"
                    / "research"
                    / "cover_search_belief_mpc_cpu_pilot.json"
                ).read_text(encoding="utf-8")
            )
            validate_config(planner_config)
            belief_before = normalize(planner_config["initial_belief"])
            belief_update = execute_observation_action(
                belief_before,
                "remove_cover",
                outcome,
                planner_config,
            )
            replanned_policy = plan(
                belief_update["posterior"], planner_config
            )
            replanning = {
                "schema_version": "live-post-remove-replan-v1",
                "observation_index": 1,
                "post_action_observation": outcome,
                "post_action_target_visible_pixels": post_pixels_live,
                "belief_before": belief_before,
                "belief_after": belief_update["posterior"],
                "planner": replanned_policy,
                "selected_action": replanned_policy["selected_action"],
                "simulator_instance_mask_used_for_pilot_control": (
                    not args.execute_post_remove_grasp
                ),
                "learned_perception_executed": bool(learned_perception),
                "learned_perception": learned_perception,
                "valid_for_final_evaluation": False,
            }
            write_json_atomic(
                output_dir / "post_remove_replan.json", replanning
            )
            write_json_atomic(
                output_dir / "action_request_001.json",
                {
                    "schema_version": "live-post-remove-action-request-v1",
                    "index": 1,
                    "type": replanned_policy["selected_action"],
                    "source_replan": str(
                        output_dir / "post_remove_replan.json"
                    ),
                    "rgbd_localization_path": (
                        learned_perception["localization_path"]
                        if learned_perception is not None
                        else None
                    ),
                    "physical_execution_requested": bool(
                        args.execute_post_remove_grasp
                    ),
                },
            )
        wait_for_path(
            output_dir / "server_result.json",
            server,
            timeout=args.timeout_seconds,
        )
        server.wait(timeout=90.0)
    finally:
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=30.0)
        stdout.close()
        stderr.close()

    server_result = json.loads(
        (output_dir / "server_result.json").read_text(encoding="utf-8")
    )
    post_event_path = output_dir / "observation_ready_001.json"
    post_event = (
        json.loads(post_event_path.read_text(encoding="utf-8"))
        if post_event_path.is_file()
        else None
    )
    initial_dir = Path(initial_event["observation_dir"])
    post_dir = Path(post_event["observation_dir"]) if post_event else None
    initial_pixels = target_pixel_count(initial_dir)
    post_pixels = target_pixel_count(post_dir) if post_dir else 0
    removal = server_result.get("cover_removal_execution") or {}
    final_grasp = server_result.get("grasp_execution") or {}
    final_grasp_success = contact_grasp_success(server_result)
    success = bool(
        server.returncode == 0
        and server_result.get("status") == "completed"
        and server_result.get("cover_removal_executed")
        and server_result.get("post_removal_observation_generated")
        and removal.get("removal_verified")
        and removal.get("bilateral_contact_before_lift")
        and removal_contact_success(removal)
        and not removal.get("unexpected_environment_pairs")
        and removal.get("contact_force_within_limit")
        and removal.get("contact_penetration_within_limit")
        and post_pixels > initial_pixels
        and (
            not args.execute_post_remove_grasp
            or final_grasp_success
        )
    )
    result = {
        "schema_version": "live-remove-cover-physics-smoke-v1",
        "status": "completed" if success else "failed",
        "seed": args.seed,
        "runtime_seconds": time.perf_counter() - started,
        "root_action": "remove_cover",
        "root_action_source": (
            "forced_observation_calibration_intervention"
            if args.force_remove_cover_for_observation_calibration
            else "scene_conditioned_calibration_model"
        ),
        "future_observation_used_for_root_selection": False,
        "initial_target_visible_pixel_count": initial_pixels,
        "post_removal_target_visible_pixel_count": post_pixels,
        "target_visibility_gain_pixels": post_pixels - initial_pixels,
        "cover_removal_execution": removal,
        "post_removal_observation": str(post_dir) if post_dir else None,
        "gpu_policy": {
            "physical_gpu": physical_gpu,
            "renderer_active_gpu": physical_gpu,
            "physics_cuda_device": 0,
            "multi_gpu": False,
        },
        "rg6_lid_physics": {
            "mode": (
                "lab_measured_calibration"
                if (
                    calibration_report is not None
                    and calibration_report["transfer_ready"]
                )
                else (
                    "explicit_provisional_public_spec"
                    if calibration_report is not None
                    else "implicit_provisional_development"
                )
            ),
            "calibration_config": (
                str(calibration_path) if calibration_path is not None else None
            ),
            "calibration_report": calibration_report,
            "transfer_ready_required": args.require_transfer_ready_physics,
            "provisional_explicitly_allowed": (
                args.allow_provisional_rg6_lid_physics
            ),
        },
        "training_performed": False,
        "calibration_performed": True,
        "testing_performed": False,
        "negative_evidence_update_performed": False,
        "replanning_performed": replanning is not None,
        "post_remove_replan": replanning,
        "post_remove_replanned_action": (
            replanning["selected_action"]
            if replanning is not None
            else None
        ),
        "post_remove_learned_perception": learned_perception,
        "post_remove_replanned_action_physical_execution": bool(
            args.execute_post_remove_grasp and final_grasp_success
        ),
        "final_grasp_executed": final_grasp_success,
        "final_grasp_execution": final_grasp,
        "valid_for_final_evaluation": False,
    }
    write_json_atomic(output_dir / "remove_cover_smoke_result.json", result)
    print(f"REMOVE_COVER_SMOKE_RESULT={output_dir / 'remove_cover_smoke_result.json'}")
    if not success:
        raise RuntimeError(f"Remove-cover smoke failed: {result}")


if __name__ == "__main__":
    main()
