"""Compile a graph-bound remove-cover request into a guarded CPU plan.

This module does not run Isaac Sim, solve IK, command the UR10e/RG6, or claim
continuous motion MPC.  It validates the high-level action binding and
generates Cartesian phase targets plus explicit live-execution safety gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "remove_cover_primitive_cpu_contract.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def vec3(name: str, value: Any) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain three values")
    result = tuple(float(component) for component in value)
    if not all(math.isfinite(component) for component in result):
        raise ValueError(f"{name} must be finite")
    return result


def add(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(a + b for a, b in zip(left, right))


def subtract(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right))


def horizontal_aabb_gap(
    center_a: tuple[float, float, float],
    extents_a: tuple[float, float, float],
    center_b: tuple[float, float, float],
    extents_b: tuple[float, float, float],
) -> float:
    gaps = [
        abs(center_a[index] - center_b[index])
        - 0.5 * (extents_a[index] + extents_b[index])
        for index in (0, 1)
    ]
    if max(gaps) > 0.0:
        return max(gaps)
    return max(gaps)


def validate_action_binding(
    request: dict[str, Any],
    graph: dict[str, Any],
) -> None:
    if request.get("schema_version") != "cover-search-action-request-v1":
        raise ValueError("Unsupported action-request schema")
    if request.get("type") != "remove_cover":
        raise ValueError("Only remove_cover can enter this primitive")
    if request.get("interface_target") != "cover_01":
        raise ValueError("remove_cover must target cover_01")
    if not request.get("execute_only_first_action", False):
        raise ValueError("First-action-only execution is required")
    if request.get("future_observation_used_for_selection", True):
        raise ValueError("Future observation leakage is forbidden")
    expected_hash = canonical_hash(graph)
    if request.get("source_scene_graph_sha256") != expected_hash:
        raise ValueError("Action request does not match the source graph")
    if request.get("episode_id") != graph.get("episode_id"):
        raise ValueError("Request and graph episode IDs do not match")


def validate_geometry(config: dict[str, Any]) -> dict[str, Any]:
    robot = config["robot"]
    basket = config["basket"]
    cover = config["cover"]
    primitive = config["primitive"]
    safety = config["safety"]
    basket_center = vec3("basket.center_local_m", basket["center_local_m"])
    basket_extents = vec3(
        "basket.outer_full_extents_m",
        basket["outer_full_extents_m"],
    )
    cover_center = vec3("cover.center_local_m", cover["center_local_m"])
    cover_extents = vec3(
        "cover.full_extents_m", cover["full_extents_m"]
    )
    handle = cover.get("handle")
    if not cover.get("rigid_body", False):
        raise ValueError("Cover must be a dynamic rigid body")
    if not isinstance(handle, dict):
        raise ValueError("Cover has no RG6 grasp handle")
    handle_center = vec3(
        "cover.handle.center_local_m", handle["center_local_m"]
    )
    handle_extents = vec3(
        "cover.handle.full_extents_m", handle["full_extents_m"]
    )
    opening = float(handle["required_opening_m"])
    if opening <= 0.0 or opening > float(robot["maximum_rg6_opening_m"]):
        raise ValueError("Cover handle is outside the RG6 opening range")
    cover_top = cover_center[2] + cover_extents[2] * 0.5
    handle_bottom = handle_center[2] - handle_extents[2] * 0.5
    if handle_bottom < cover_top - 1e-9:
        raise ValueError("Cover handle intersects the cover plate")
    staging_center = vec3(
        "primitive.staging_cover_center_local_m",
        primitive["staging_cover_center_local_m"],
    )
    staging_gap = horizontal_aabb_gap(
        basket_center,
        basket_extents,
        staging_center,
        cover_extents,
    )
    minimum_clearance = float(safety["minimum_static_clearance_m"])
    if staging_gap < minimum_clearance:
        raise ValueError(
            "Staging cover position does not clear the basket"
        )
    basket_top = basket_center[2] + basket_extents[2] * 0.5
    lifted_cover_center_z = (
        cover_center[2] + float(primitive["vertical_lift_m"])
    )
    lifted_cover_bottom_z = lifted_cover_center_z - cover_extents[2] * 0.5
    required_transfer_bottom_z = basket_top + float(
        primitive["transfer_clearance_above_basket_m"]
    )
    if lifted_cover_bottom_z < required_transfer_bottom_z:
        raise ValueError("Vertical lift is too low for horizontal transfer")
    support_plane = float(basket["support_plane_local_z_m"])
    staged_cover_bottom = staging_center[2] - cover_extents[2] * 0.5
    if not math.isclose(staged_cover_bottom, support_plane, abs_tol=1e-6):
        raise ValueError("Staged cover is not placed on the support plane")
    return {
        "handle_fits_rg6": True,
        "handle_plate_separation_m": handle_bottom - cover_top,
        "staging_horizontal_gap_m": staging_gap,
        "lifted_cover_bottom_clearance_over_basket_m": (
            lifted_cover_bottom_z - basket_top
        ),
        "staged_cover_support_error_m": abs(
            staged_cover_bottom - support_plane
        ),
        "static_geometry_gate_passed": True,
    }


def build_cartesian_plan(config: dict[str, Any]) -> list[dict[str, Any]]:
    cover = config["cover"]
    primitive = config["primitive"]
    cover_center = vec3("cover.center_local_m", cover["center_local_m"])
    handle_center = vec3(
        "cover.handle.center_local_m",
        cover["handle"]["center_local_m"],
    )
    staging_cover_center = vec3(
        "primitive.staging_cover_center_local_m",
        primitive["staging_cover_center_local_m"],
    )
    handle_offset = subtract(handle_center, cover_center)
    lift_delta = (0.0, 0.0, float(primitive["vertical_lift_m"]))
    lifted_handle = add(handle_center, lift_delta)
    staged_handle = add(staging_cover_center, handle_offset)
    staged_above = (
        staged_handle[0],
        staged_handle[1],
        lifted_handle[2],
    )
    pregrasp = add(
        handle_center,
        (0.0, 0.0, float(primitive["pregrasp_clearance_m"])),
    )
    retreat = add(
        staged_handle,
        (0.0, 0.0, float(primitive["retreat_clearance_m"])),
    )
    return [
        {
            "phase": "move_to_handle_pregrasp",
            "tool_center_local_m": list(pregrasp),
            "gripper": "open",
            "required_gates": [
                "source_graph_hash_match",
                "ik_success",
                "continuous_collision_check",
                "finite_joint_state",
            ],
        },
        {
            "phase": "descend_to_handle",
            "tool_center_local_m": list(handle_center),
            "gripper": "open",
            "required_gates": [
                "continuous_collision_check",
                "arm_tracking_error",
            ],
        },
        {
            "phase": "close_on_handle",
            "tool_center_local_m": list(handle_center),
            "gripper": "close_until_contact",
            "required_gates": [
                "bilateral_handle_contact",
                "minimum_grip_force",
                "maximum_force",
                "maximum_penetration",
            ],
        },
        {
            "phase": "vertical_cover_lift",
            "tool_center_local_m": list(lifted_handle),
            "gripper": "hold",
            "required_gates": [
                "bilateral_handle_contact",
                "continuous_collision_check",
                "maximum_horizontal_slip",
            ],
        },
        {
            "phase": "transfer_cover_to_staging",
            "tool_center_local_m": list(staged_above),
            "gripper": "hold",
            "required_gates": [
                "bilateral_handle_contact",
                "continuous_collision_check",
                "cover_clears_basket",
            ],
        },
        {
            "phase": "lower_cover_to_support",
            "tool_center_local_m": list(staged_handle),
            "gripper": "hold",
            "required_gates": [
                "continuous_collision_check",
                "staging_support_contact",
            ],
        },
        {
            "phase": "release_cover",
            "tool_center_local_m": list(staged_handle),
            "gripper": "open",
            "required_gates": [
                "cover_stable_at_staging",
                "cover_outside_basket_opening",
            ],
        },
        {
            "phase": "retreat_and_request_reobservation",
            "tool_center_local_m": list(retreat),
            "gripper": "open",
            "required_gates": [
                "continuous_collision_check",
                "new_rgbd_required_before_belief_update",
            ],
        },
    ]


def compile_plan(
    config: dict[str, Any],
    request: dict[str, Any],
    graph: dict[str, Any],
) -> dict[str, Any]:
    validate_action_binding(request, graph)
    geometry = validate_geometry(config)
    phases = build_cartesian_plan(config)
    return {
        "schema_version": "remove-cover-primitive-plan-v1",
        "experiment_id": config["experiment_id"],
        "status": "ready_for_live_ik_and_physics_validation",
        "request_id": request["request_id"],
        "episode_id": request["episode_id"],
        "source_scene_graph_sha256": request[
            "source_scene_graph_sha256"
        ],
        "action": "remove_cover",
        "interface_target": "cover_01",
        "frame_id": config["frame_id"],
        "robot": config["robot"],
        "geometry_validation": geometry,
        "phases": phases,
        "safety_limits": config["safety"],
        "failure_to_observation_mapping": {
            "ik_failure": "action_failed",
            "collision": "action_failed",
            "lost_bilateral_contact": "action_failed",
            "force_or_penetration_limit": "action_failed",
            "tracking_error_or_nonfinite_state": "action_failed",
            "successful_cover_removal": (
                "capture_new_rgbd_then_classify_target_detected_or_"
                "empty_container"
            ),
        },
        "execute_only_first_high_level_action": True,
        "physical_execution_authorized": False,
        "ik_computed": False,
        "narrow_phase_collision_checked": False,
        "cover_manipulation_executed": False,
        "new_observation_generated": False,
        "gpu_used": False,
        "gpu_memory_gib": 0.0,
        "training_performed": False,
        "calibration_performed": False,
        "testing_performed": False,
        "valid_for_final_evaluation": False,
        "limitations": [
            "Cartesian targets are expressed in the basket frame only.",
            "UR10e IK and time parameterization were not computed.",
            "No swept-volume or narrow-phase collision query was run.",
            "The new dynamic cover assembly has not been rendered or physically validated.",
            "No RGB-D, learned perception, belief update, or replanning was executed.",
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def run(config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_json(config_path)
    request = load_json(resolve_path(config["source_action_request"]))
    graph = load_json(resolve_path(config["source_scene_graph"]))
    output_root = resolve_path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    plan = compile_plan(config, request, graph)
    plan["runtime_seconds"] = time.perf_counter() - started
    write_json(output_root / "remove_cover_plan.json", plan)
    summary = {
        key: value
        for key, value in plan.items()
        if key not in ("phases", "safety_limits", "robot")
    }
    summary["phase_count"] = len(plan["phases"])
    write_json(output_root / "summary.json", summary)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = run(args.config.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "phase_count": len(result["phases"]),
                "gpu_used": result["gpu_used"],
                "physical_execution_authorized": result[
                    "physical_execution_authorized"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
