"""Run one physical two-update negative-evidence development episode."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path

from audit_saved_learned_relation import audit_relation
from run_cover_search_belief_mpc import (
    execute_observation_action,
    normalize,
    plan,
    validate_config,
)
from run_live_single_gpu_pipeline import (
    ISAAC_PYTHON,
    ROOT,
    wait_for_path,
    write_json_atomic,
)
from run_remove_cover_live_smoke import (
    contact_grasp_success,
    learned_post_remove_localization,
    removal_contact_success,
)
from run_single_gpu_pilot import (
    configured_physical_gpu,
    require_single_gpu_policy,
)
from final_evaluation_authorization import (
    DEFAULT_PROTOCOL,
    validate_output_authorization,
)
from core_method_runtime import (
    calibrated_commitment_gate,
    calibrated_target_identity,
    relation_observation_from_audit,
    scene_graph_snapshot,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "negative_evidence_live_development.json"
)
OUTPUT_ROOT = (
    ROOT
    / "outputs"
    / "live_pipeline"
    / "negative_evidence_live_development"
)


def load_json(path: Path) -> dict:
    """Load a JSON configuration or result artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    """Resolve a repository-relative path."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def next_output_dir(seed: int, output_root: Path = OUTPUT_ROOT) -> Path:
    """Allocate a new run directory without overwriting earlier results."""
    seed_root = output_root / f"seed{seed:03d}"
    seed_root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in seed_root.glob("run*"):
        try:
            indices.append(int(path.name.removeprefix("run")))
        except ValueError:
            continue
    return seed_root / f"run{max(indices, default=0) + 1:03d}"


def effective_planner(config: dict) -> dict:
    """Build the planner configuration used for this episode."""
    if config.get("frozen_planner_config"):
        planner = load_json(resolve_path(config["frozen_planner_config"]))
        validate_config(planner)
        return planner
    planner = copy.deepcopy(load_json(resolve_path(config["base_planner_config"])))
    overrides = config["planner_overrides"]
    planner["objective"]["minimum_grasp_success_probability"] = float(
        overrides["minimum_grasp_success_probability"]
    )
    for action, key in (
        ("viewpoint_close_high", "viewpoint_close_high_open_likelihood"),
        ("viewpoint_right", "viewpoint_right_open_likelihood"),
    ):
        planner["observation_model"][action]["likelihood"].update(
            overrides[key]
        )
    validate_config(planner)
    return planner


def automatic_post_remove_outcome(
    household_scene: dict,
    *,
    target_visible_pixels: int,
    minimum_target_pixels: int,
) -> str:
    """Map automatic simulator membership to an inspected-container symbol.

    Target pixels anywhere in the image do not imply that the target was
    found inside the inspected container.  This development-only adapter uses
    generator 3D membership after the physical action; final evaluation must
    replace it with frozen learned/RGB-D relation evidence.
    """
    ground_truth = household_scene.get("calibration_ground_truth") or {}
    membership = (
        ground_truth.get("world_ground_truth") or {}
    ).get("membership")
    if membership == "outside":
        return "empty_container"
    if membership == "inside":
        return (
            "target_detected"
            if target_visible_pixels >= minimum_target_pixels
            else "empty_container"
        )
    raise ValueError(
        f"Automatic simulator membership is unavailable: {membership!r}"
    )


def learned_post_remove_outcome(relation_audit: dict) -> str:
    """Return a conservative symbol only when Qwen and RGB-D agree."""
    qwen_label = relation_audit.get("qwen_relation_top_label")
    rgbd_label = (
        relation_audit.get("rgbd_relation", {})
        .get("membership_world_evidence", {})
        .get("label")
    )
    if qwen_label == rgbd_label == "outside":
        return "empty_container"
    if qwen_label == rgbd_label == "inside":
        return "target_detected"
    raise RuntimeError(
        "Learned post-remove relation evidence is absent or disagrees: "
        f"Qwen={qwen_label!r}, RGB-D={rgbd_label!r}"
    )


def main() -> None:
    """Run the target-absent closed loop and verify a safe defer decision."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the calibration seed without changing the base config.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument(
        "--final-evaluation-authorized",
        action="store_true",
        help="Write a reserved test episode under the frozen final protocol.",
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    seed = int(config["seed"] if args.seed is None else args.seed)
    if seed < 0:
        raise ValueError("Seed must be non-negative")
    if seed in set(range(200, 210)) and not args.final_evaluation_authorized:
        raise ValueError("Reserved test seeds are forbidden in development")
    require_single_gpu_policy()
    physical_gpu = configured_physical_gpu()
    planner_config = effective_planner(config)
    frozen_parameters = load_json(
        resolve_path(config["frozen_method_parameters"])
    )
    target_temperature = float(
        frozen_parameters["target_identity_calibration"]["temperature"]
    )
    commitment_threshold = float(
        frozen_parameters["task_cost_and_commitment_gate"][
            "minimum_grasp_success_probability"
        ]
    )
    expected = list(config["expected_action_sequence"])
    initial_belief = normalize(planner_config["initial_belief"])
    root_policy = plan(initial_belief, planner_config)
    if root_policy["selected_action"] != expected[0]:
        raise RuntimeError(f"Unexpected root action: {root_policy}")

    output_root = args.output_root.resolve()
    output_dir = next_output_dir(seed, output_root)
    validate_output_authorization(
        seed=seed,
        output_dir=output_dir.resolve(),
        final_evaluation_authorized=args.final_evaluation_authorized,
        protocol_path=args.protocol.resolve(),
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output_dir / "effective_planner_config.json", planner_config)
    physics_path = resolve_path(config["rg6_lid_calibration_config"])
    base_method = load_json(
        ROOT / "configs" / "research" / "first_belief_mpc_integration.json"
    )
    base_method["viewpoint_execution"]["mode"] = "interpolated_joint_physics"
    base_method["viewpoint_execution"]["debug_ee_positions_world_m"] = {}
    method_path = output_dir / "effective_method_config.json"
    write_json_atomic(method_path, base_method)

    command = [
        str(ISAAC_PYTHON),
        str(ROOT / "scripts" / "open_minimal_scene.py"),
        "--scene-profile", "benchmark",
        "--headless",
        "--renderer-gpu", str(physical_gpu),
        "--physics-gpu", "0",
        "--live-pipeline-server",
        "--actual-view-motion",
        "--live-session-dir", str(output_dir),
        "--method-config", str(method_path),
        "--seed", str(seed),
        "--household-perception-pilot",
        "--scanned-basket-perception-pilot",
        "--calibration-scene-variant", config["scene_variant"],
        "--basket-collision-physics-pilot",
        "--execute-persistent-remove-cover",
        "--continue-after-remove-cover",
        "--rg6-lid-calibration-config", str(physics_path),
        "--allow-provisional-rg6-lid-physics",
        "--rg6-coupling-mode", config["rg6_coupling_mode"],
        "--coordinated-rg6-total-drive-effort-limit-nm",
        str(config["coordinated_rg6_total_drive_effort_limit_nm"]),
    ]
    if args.final_evaluation_authorized:
        command.extend(
            [
                "--final-evaluation-authorized",
                "--final-evaluation-protocol",
                str(args.protocol.resolve()),
            ]
        )
    environment = dict(os.environ)
    gpu_text = str(physical_gpu)
    environment.update(
        {
            "PHYSICAL_GPU": gpu_text,
            "CUDA_VISIBLE_DEVICES": gpu_text,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NVIDIA_VISIBLE_DEVICES": gpu_text,
        }
    )
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)

    started = time.perf_counter()
    stdout = (output_dir / "isaac_stdout.log").open("w", encoding="utf-8")
    stderr = (output_dir / "isaac_stderr.log").open("w", encoding="utf-8")
    server = subprocess.Popen(
        command,
        cwd=ROOT,
        env=environment,
        stdout=stdout,
        stderr=stderr,
        text=True,
    )
    updates = []
    planner_policies = [root_policy]
    scene_graph = []
    post_remove_perception = None
    post_remove_relation = None
    perception = None
    right_relation = None
    target_identity = None
    commitment_gate = None
    try:
        wait_for_path(output_dir / "observation_ready_000.json", server, 600.0)
        write_json_atomic(
            output_dir / "action_request_000.json",
            {
                "schema_version": "negative-evidence-live-request-v1",
                "index": 0,
                "type": expected[0],
                "source_policy": root_policy,
                "future_observation_used_for_selection": False,
            },
        )
        wait_for_path(output_dir / "observation_ready_001.json", server, args.timeout_seconds)
        post_dir = output_dir / "observations" / "post_remove"
        post_remove_perception = learned_post_remove_localization(
            output_dir,
            post_dir,
            observation_index=1,
            view="post_remove",
            task_overrides=config["perception_task_overrides"],
        )
        post_remove_relation = audit_relation(
            post_dir,
            Path(post_remove_perception["ranking"]["input_path"]),
            Path(post_remove_perception["ranking_path"]),
            resolve_path(config["rgbd_relation_config"]),
        )
        post_remove_relation_path = output_dir / "post_remove_learned_relation.json"
        write_json_atomic(post_remove_relation_path, post_remove_relation)
        outcome = learned_post_remove_outcome(post_remove_relation)
        if outcome != "empty_container":
            raise RuntimeError(
                f"Expected learned empty-container evidence, got {outcome}"
            )
        first_update = execute_observation_action(
            initial_belief, "remove_cover", outcome, planner_config
        )
        updates.append(first_update)
        post_remove_identity = calibrated_target_identity(
            post_remove_perception, target_temperature
        )
        post_remove_evidence = relation_observation_from_audit(
            post_remove_relation
        )
        scene_graph.append(
            scene_graph_snapshot(
                step=1,
                view="post_remove",
                identity=post_remove_identity,
                belief=first_update["posterior"],
                relation_evidence=post_remove_evidence,
            )
        )
        write_json_atomic(
            output_dir / "probabilistic_scene_graph_step001.json",
            scene_graph[-1],
        )
        first_replan = plan(first_update["posterior"], planner_config)
        planner_policies.append(first_replan)
        if first_replan["selected_action"] != expected[1]:
            raise RuntimeError(f"Unexpected post-empty action: {first_replan}")
        write_json_atomic(
            output_dir / "action_request_001.json",
            {
                "schema_version": "negative-evidence-live-request-v1",
                "index": 1,
                "type": expected[1],
                "source_policy": first_replan,
                "physical_execution_requested": True,
            },
        )
        wait_for_path(output_dir / "observation_ready_002.json", server, 900.0)
        right_dir = output_dir / "observations" / "right"
        perception = learned_post_remove_localization(
            output_dir,
            right_dir,
            observation_index=2,
            view="right",
            task_overrides=config["perception_task_overrides"],
        )
        right_relation = audit_relation(
            right_dir,
            Path(perception["ranking"]["input_path"]),
            Path(perception["ranking_path"]),
            resolve_path(config["rgbd_relation_config"]),
        )
        write_json_atomic(
            output_dir / "right_learned_relation.json", right_relation
        )
        right_evidence = relation_observation_from_audit(right_relation)
        second_update = execute_observation_action(
            first_update["posterior"],
            "viewpoint_right",
            right_evidence["planner_observation"],
            planner_config,
        )
        updates.append(second_update)
        second_replan = plan(second_update["posterior"], planner_config)
        planner_policies.append(second_replan)
        if second_replan["selected_action"] != expected[2]:
            raise RuntimeError(f"Unexpected terminal action: {second_replan}")
        target_identity = calibrated_target_identity(
            perception,
            target_temperature,
        )
        commitment_gate = calibrated_commitment_gate(
            terminal_action=second_replan["selected_action"],
            posterior=second_update["posterior"],
            identity=target_identity,
            minimum_probability=commitment_threshold,
        )
        scene_graph.append(
            scene_graph_snapshot(
                step=2,
                view="right",
                identity=target_identity,
                belief=second_update["posterior"],
                relation_evidence=right_evidence,
            )
        )
        write_json_atomic(
            output_dir / "probabilistic_scene_graph_step002.json",
            scene_graph[-1],
        )
        if not commitment_gate["authorized"]:
            raise RuntimeError(
                "Calibrated commitment gate rejected the terminal grasp: "
                f"{commitment_gate}"
            )
        write_json_atomic(
            output_dir / "action_request_002.json",
            {
                "schema_version": "negative-evidence-live-request-v1",
                "index": 2,
                "type": expected[2],
                "source_policy": second_replan,
                "calibrated_commitment_gate": commitment_gate,
                "rgbd_localization_path": perception["localization_path"],
                "physical_execution_requested": True,
            },
        )
        wait_for_path(output_dir / "server_result.json", server, args.timeout_seconds)
        server.wait(timeout=120.0)
    finally:
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=30.0)
        stdout.close()
        stderr.close()

    server_result = load_json(output_dir / "server_result.json")
    removal = server_result.get("cover_removal_execution") or {}
    scientific_success = bool(
        server_result.get("status") == "completed"
        and removal.get("removal_verified")
        and removal_contact_success(removal)
        and contact_grasp_success(server_result)
        and bool(commitment_gate and commitment_gate["authorized"])
        and len(updates) >= int(config["minimum_belief_updates"])
        and server_result.get("post_remove_replanned_action") == expected[2]
    )
    clean_server_shutdown = server.returncode == 0
    success = scientific_success
    result = {
        "schema_version": "negative-evidence-live-development-result-v1",
        "status": "completed" if success else "failed",
        "experiment_id": config["experiment_id"],
        "episode_config": str(config_path),
        "seed": seed,
        "runtime_seconds": time.perf_counter() - started,
        "scientific_episode_success": scientific_success,
        "server_exit_code": server.returncode,
        "clean_server_shutdown": clean_server_shutdown,
        "post_result_teardown_issue": bool(
            scientific_success and not clean_server_shutdown
        ),
        "action_sequence": expected,
        "post_remove_observation": "empty_container",
        "negative_evidence_update_performed": True,
        "belief_updates": updates,
        "belief_update_count": len(updates),
        "action_conditioned_mpc_policies": planner_policies,
        "planner_horizon": int(planner_config["horizon"]),
        "learned_post_remove_perception": post_remove_perception,
        "learned_post_remove_relation": post_remove_relation,
        "learned_post_reobservation_perception": perception,
        "learned_post_reobservation_relation": right_relation,
        "target_identity_calibration": target_identity,
        "probabilistic_scene_graph": scene_graph,
        "calibrated_commitment_gate": commitment_gate,
        "frozen_method_parameters": str(
            resolve_path(config["frozen_method_parameters"])
        ),
        "cover_removal_execution": removal,
        "final_grasp_execution": server_result.get("grasp_execution"),
        "gpu_policy": {
            "physical_gpu": physical_gpu,
            "renderer_active_gpu": physical_gpu,
            "physics_cuda_device": 0,
            "multi_gpu": False,
        },
        "negative_evidence_source": config["negative_evidence_source"],
        "relation_config": str(resolve_path(config["rgbd_relation_config"])),
        "training_performed": False,
        "calibration_performed": False,
        "frozen_calibration_applied": True,
        "testing_performed": args.final_evaluation_authorized,
        "reserved_test_seeds_used": args.final_evaluation_authorized,
        "valid_for_final_evaluation": bool(
            args.final_evaluation_authorized and success
        ),
        "limitations": [
            "The joint grasp gate uses a factorized identity-location product assumption.",
            *(
                []
                if args.final_evaluation_authorized
                else [
                    "This is a development integration episode, not a reserved final-test result."
                ]
            ),
            "RG6 and cover physics remain provisional and transfer_ready=false."
        ]
    }
    write_json_atomic(output_dir / "negative_evidence_live_result.json", result)
    print(f"NEGATIVE_EVIDENCE_LIVE_RESULT={output_dir / 'negative_evidence_live_result.json'}")
    if not success:
        raise RuntimeError(f"Negative-evidence live smoke failed: {result}")


if __name__ == "__main__":
    main()
