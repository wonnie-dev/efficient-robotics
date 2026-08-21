"""Execute the validated seed-0 UR10e+RG6 grasp in an existing Isaac stage.

This module does not create or close ``SimulationApp``.  It is used by the live
Isaac observation server after a terminal grasp request so perception,
replanning, and manipulation share one persistent stage and process.

Fresh standalone composites re-express the authored world in the robot-base
frame. A reused live composite stays in its existing world frame. In both
cases, the target remains dynamic: transport must come from simulated contact,
never an attachment or copied pose.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np


ROBOT_BASE_IN_AUTHORED_WORLD = np.asarray(
    [-0.20, 0.32, 0.76], dtype=np.float64
)
WORLD_TO_ROBOT_BASE = -ROBOT_BASE_IN_AUTHORED_WORLD
PHYSICS_DT_SECONDS = 1.0 / 60.0
HOUSEHOLD_MUG_OUTER_RADIUS_M = 0.041
HOUSEHOLD_MUG_HEIGHT_M = 0.102
HOUSEHOLD_MUG_MASS_KG = 0.30
PROVISIONAL_STATIC_FRICTION = 0.80
PROVISIONAL_DYNAMIC_FRICTION = 0.60
PROVISIONAL_EPDM_LID_STATIC_FRICTION = 1.00
PROVISIONAL_EPDM_LID_DYNAMIC_FRICTION = 0.80
MAX_ALLOWED_CONTACT_FORCE_N = 60.0
MAX_ALLOWED_PENETRATION_M = 0.003
MINIMUM_GRIP_FORCE_PER_FINGER_N = 3.0
PROVISIONAL_LID_GRIP_FORCE_PER_FINGER_N = 8.0
PROVISIONAL_LID_COMBINED_GRIP_FORCE_N = 25.0
CONTACT_CONTROLLED_CLOSURE_STEPS = 1500
GRIP_FORCE_TARGET_INCREMENT_RAD = 0.00035
PROVISIONAL_RG6_MASTER_MAX_TORQUE_NM = 0.60
PROVISIONAL_TARGET_COORDINATED_DRIVE_EFFORT_LIMIT_NM = 6.0
PROVISIONAL_TARGET_MINIMUM_GRIP_FORCE_PER_FINGER_N = 5.0
PROVISIONAL_TARGET_PASSIVE_FORCE_CONTROLLER_MAX_TORQUE_NM = 1.2
RG6_HANDLE_CLOSE_MASTER_RAD = 0.60
MAX_RELATIVE_TARGET_TRANSLATION_M = 0.015
MAX_RELATIVE_TARGET_ROTATION_RAD = math.radians(10.0)
MAX_RELATIVE_TARGET_ANGULAR_SPEED_RAD_S = 2.0
MAX_FORCE_SAMPLE_AGE_STEPS = 12
TERMINAL_FORCE_WINDOW_STEPS = 120
MINIMUM_TERMINAL_FORCE_QUALIFYING_STEPS = 3
GRIP_CONTROLLER_FORCE_WINDOW_STEPS = 30
GRIP_CONTROLLER_STEP_SLIP_M = 0.00025
MAX_TRANSFER_HORIZONTAL_ERROR_M = 0.020
MAX_CONSECUTIVE_CONTACT_GAP_STEPS = 3
MICRO_LIFT_HEIGHT_M = 0.010
MAX_MICRO_LIFT_RELATIVE_TRANSLATION_M = 0.005
MINIMUM_MICRO_LIFT_DELTA_M = 0.007
LIFT_IK_POSITION_TOLERANCE_M = 0.0002
LIFT_IK_ORIENTATION_TOLERANCE_RAD = 0.002
MAX_LIFT_IK_POSITION_ERROR_M = 0.001
MAX_LIFT_IK_ORIENTATION_ERROR_RAD = 0.005
MICRO_LIFT_CARTESIAN_WAYPOINTS = 5
FULL_LIFT_CARTESIAN_WAYPOINTS = 18
MAX_LIFT_IK_JOINT_STEP_RAD = 0.15
# Execute the 0.18 m contact lift over 15 s at 60 Hz.  The former 180/300
# total-step schedules advanced each 1 cm waypoint faster than the loaded
# UR10e drive could track, causing a monotonic 0.013 -> 0.054 rad lag while
# bilateral contact remained valid.  Slowing the command preserves the same
# fail-closed 0.05 rad tracking gate instead of relaxing it.
FULL_LIFT_TRAJECTORY_STEPS = 900
# Total motion steps across the dense transfer path (12 s at 60 Hz).
TRANSFER_TRAJECTORY_STEPS = 720
TRANSFER_CARTESIAN_WAYPOINTS = 9
PLACEMENT_CARTESIAN_WAYPOINTS = 33
PLACEMENT_TRAJECTORY_STEPS = 60
PLACEMENT_SUPPORT_PENETRATION_M = 0.0005
PLACEMENT_MAX_SUPPORT_PENETRATION_M = 0.002
MINIMUM_SUPPORT_OVERLAP_FRACTION = 0.75
MINIMUM_SUPPORT_CENTER_MARGIN_M = 0.03
RELEASE_TRAJECTORY_STEPS = 360
RETREAT_CARTESIAN_WAYPOINTS = 15
RETREAT_DISTANCE_M = 0.15
COORDINATED_FOLLOWER_REQUEST_BLEND = 0.75
PROVISIONAL_TARGET_FOLLOWER_REQUEST_BLEND = 1.0
PREGRASP_MAX_TRAJECTORY_SPEED_RAD_S = 0.35
DESCENT_MAX_TRAJECTORY_SPEED_RAD_S = 0.20
MAX_DESCENT_IK_JOINT_STEP_RAD = 0.45
DESCENT_OFFSETS_M = tuple(
    round(0.18 - 0.01 * index, 2) for index in range(19)
)
# Canonical table-aligned yaw from the collision-checked seed-0 RG6 grasp.
# This is an approach orientation only: the target position still comes from
# the current episode's Qwen-selected RGB-D mask.  Keeping this orientation
# also avoids an unnecessary near-pi wrist_3 roll from the live observation
# posture before the vertical approach begins.
DEFAULT_TABLE_ALIGNED_GRASP_YAW_RAD = 2.0579804469193013
RG6_OPEN_FINGER_CENTER_HALF_SPAN_M = 0.085
ENVIRONMENT_ROOTS = (
    "/World/Ground",
    "/World/LabBackWall",
    "/World/LabSideWall",
    "/World/WorkBench",
    "/World/WorkMat",
    "/World/OpenContainer",
    "/World/TargetRed",
    "/World/OccluderOrange",
    "/World/DistractorYellow",
    "/World/DistractorBlue",
    "/World/DistractorGreen",
    "/World/BoundaryPurple",
    "/World/RearRedCandidate",
    "/World/LabProps",
)


def quintic_time_scaling(value: float) -> float:
    """Smooth zero-velocity/acceleration interpolation on [0, 1]."""
    if not 0.0 <= value <= 1.0:
        raise ValueError("quintic time value must be in [0, 1]")
    return value**3 * (10.0 + value * (-15.0 + 6.0 * value))


def rotation_angle_rad(rotation: np.ndarray) -> float:
    """Return the unsigned angle of a finite 3x3 rotation matrix."""
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.acos(cosine))


def rotation_vector_from_matrix(rotation: np.ndarray) -> np.ndarray:
    """Return a small-to-moderate SO(3) rotation vector."""
    angle = rotation_angle_rad(rotation)
    if angle < 1.0e-10:
        return np.zeros(3, dtype=np.float64)
    sine = math.sin(angle)
    if abs(sine) < 1.0e-8:
        raise ValueError("rotation vector is undefined near pi")
    axis = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * sine)
    return axis * angle


def grasp_yaw_from_pinch_axis_world(
    pinch_axis_world: np.ndarray,
) -> float:
    """Map a horizontal handle pinch axis to this RG6 asset's wrist yaw.

    For the imported RG6, the line between the fingertip collision centers is
    ``grasp_yaw - pi/2`` in the world XY plane.  Aligning that line with the
    handle pinch axis therefore requires a +pi/2 wrist-yaw offset.
    """
    axis = np.asarray(pinch_axis_world, dtype=np.float64)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)):
        raise ValueError("pinch_axis_world must be a finite 3-vector")
    if float(np.linalg.norm(axis[:2])) < 1.0e-9:
        raise ValueError("pinch_axis_world must have a horizontal component")
    pinch_yaw = math.atan2(float(axis[1]), float(axis[0]))
    yaw = pinch_yaw + math.pi * 0.5
    return float(math.atan2(math.sin(yaw), math.cos(yaw)))


def parallel_gripper_yaw_candidates(grasp_yaw_rad: float) -> tuple[float, float]:
    """Return both wrist yaws representing the same unoriented pinch line."""
    if not math.isfinite(grasp_yaw_rad):
        raise ValueError("grasp_yaw_rad must be finite")
    primary = math.atan2(math.sin(grasp_yaw_rad), math.cos(grasp_yaw_rad))
    opposite = math.atan2(
        math.sin(primary + math.pi),
        math.cos(primary + math.pi),
    )
    return float(primary), float(opposite)


def rank_outside_container_grasp_yaws(
    grasp_yaw_rad: float,
    target_xy_world_m: np.ndarray,
    wall_bounds_world_m: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[tuple[float, ...], list[dict]]:
    """Rank parallel and orthogonal pinch axes by basket-wall clearance.

    An outside target can sit beside a basket corner.  A collision-free
    vertical wrist path may then require rotating the RG6 pinch line by 90
    degrees so that both open fingers remain clear of the nearest wall end.
    This ranking uses only the selected RGB-D position and current scene
    collision geometry; it does not use the target identity ground truth.
    """
    target_xy = np.asarray(target_xy_world_m, dtype=np.float64)
    if target_xy.shape != (2,) or not np.all(np.isfinite(target_xy)):
        raise ValueError("target_xy_world_m must be a finite 2-vector")
    if not wall_bounds_world_m:
        return parallel_gripper_yaw_candidates(grasp_yaw_rad), []

    candidates = (
        *parallel_gripper_yaw_candidates(grasp_yaw_rad),
        *parallel_gripper_yaw_candidates(grasp_yaw_rad + math.pi * 0.5),
    )
    diagnostics = []
    for order, yaw in enumerate(candidates):
        pinch_yaw = yaw - math.pi * 0.5
        pinch_axis = np.asarray(
            [math.cos(pinch_yaw), math.sin(pinch_yaw)], dtype=np.float64
        )
        finger_centers = (
            target_xy
            + RG6_OPEN_FINGER_CENTER_HALF_SPAN_M * pinch_axis,
            target_xy
            - RG6_OPEN_FINGER_CENTER_HALF_SPAN_M * pinch_axis,
        )
        clearances = []
        for center in finger_centers:
            for lower, upper in wall_bounds_world_m:
                lower_xy = np.asarray(lower, dtype=np.float64)[:2]
                upper_xy = np.asarray(upper, dtype=np.float64)[:2]
                outside_delta = np.maximum(
                    np.maximum(lower_xy - center, center - upper_xy),
                    0.0,
                )
                clearances.append(float(np.linalg.norm(outside_delta)))
        diagnostics.append(
            {
                "grasp_yaw_rad": float(yaw),
                "pinch_axis_xy": pinch_axis.tolist(),
                "predicted_open_finger_centers_xy_world_m": [
                    center.tolist() for center in finger_centers
                ],
                "minimum_wall_aabb_clearance_m": min(clearances),
                "input_order": order,
            }
        )
    diagnostics.sort(
        key=lambda row: (
            -float(row["minimum_wall_aabb_clearance_m"]),
            int(row["input_order"]),
        )
    )
    return (
        tuple(float(row["grasp_yaw_rad"]) for row in diagnostics),
        diagnostics,
    )


def closest_equivalent_joint_configuration(
    joints_rad: np.ndarray,
    reference_rad: np.ndarray,
) -> np.ndarray:
    """Return 2-pi-equivalent revolute angles nearest to the reference."""
    joints = np.asarray(joints_rad, dtype=np.float64)
    reference = np.asarray(reference_rad, dtype=np.float64)
    if joints.shape != reference.shape or not (
        np.all(np.isfinite(joints)) and np.all(np.isfinite(reference))
    ):
        raise ValueError("joint configuration and reference must be finite peers")
    delta = np.arctan2(
        np.sin(joints - reference),
        np.cos(joints - reference),
    )
    return reference + delta


def supported_root_height_m(
    *,
    support_top_z_m: float,
    target_bottom_offset_from_root_m: float,
    commanded_support_penetration_m: float = PLACEMENT_SUPPORT_PENETRATION_M,
) -> float:
    """Return the target-root height for a shallow, contact-seeking placement."""
    values = (
        support_top_z_m,
        target_bottom_offset_from_root_m,
        commanded_support_penetration_m,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("placement geometry must be finite")
    if not 0.0 <= commanded_support_penetration_m <= 0.001:
        raise ValueError(
            "commanded_support_penetration_m must be between 0 and 1 mm"
        )
    return (
        support_top_z_m
        - target_bottom_offset_from_root_m
        - commanded_support_penetration_m
    )


def planar_support_metrics(
    *,
    support_min_xy: np.ndarray,
    support_max_xy: np.ndarray,
    target_min_xy: np.ndarray,
    target_max_xy: np.ndarray,
) -> dict[str, float]:
    """Measure stable planar support while permitting bounded overhang."""
    support_min = np.asarray(support_min_xy, dtype=np.float64)
    support_max = np.asarray(support_max_xy, dtype=np.float64)
    target_min = np.asarray(target_min_xy, dtype=np.float64)
    target_max = np.asarray(target_max_xy, dtype=np.float64)
    arrays = (support_min, support_max, target_min, target_max)
    if any(array.shape != (2,) for array in arrays) or not all(
        np.all(np.isfinite(array)) for array in arrays
    ):
        raise ValueError("support and target bounds must be finite XY pairs")
    if np.any(support_max <= support_min) or np.any(target_max <= target_min):
        raise ValueError("support and target bounds must have positive area")
    overlap_extents = np.maximum(
        np.minimum(support_max, target_max)
        - np.maximum(support_min, target_min),
        0.0,
    )
    target_extents = target_max - target_min
    target_center = 0.5 * (target_min + target_max)
    return {
        "overlap_fraction": float(
            np.prod(overlap_extents) / np.prod(target_extents)
        ),
        "center_margin_m": float(
            min(
                *(target_center - support_min),
                *(support_max - target_center),
            )
        ),
        "full_footprint_margin_m": float(
            min(
                *(target_min - support_min),
                *(support_max - target_max),
            )
        ),
    }


def should_rebuild_persistent_physics(
    *, reuse_existing_composite: bool, executor_already_prepared: bool
) -> bool:
    """Rebuild PhysX only for the first persistent manipulation.

    A stop/play cycle restores authored rigid-body poses. It is required when
    the first persistent executor call prepares the articulation, reporters,
    and RG6 drives, but a second cycle in the same episode would move an
    already placed cover back to its authored pose on top of the basket.
    """
    return not (reuse_existing_composite and executor_already_prepared)


def summarize_terminal_force_window(
    samples_by_side: dict[str, list[dict]],
    *,
    current_step: int,
    minimum_force_per_finger_n: float,
    window_steps: int = TERMINAL_FORCE_WINDOW_STEPS,
    minimum_qualifying_steps: int = (
        MINIMUM_TERMINAL_FORCE_QUALIFYING_STEPS
    ),
) -> dict:
    """Summarize distinct recent force timesteps for a terminal grasp gate."""
    if current_step < 0 or window_steps <= 0 or minimum_qualifying_steps <= 0:
        raise ValueError("terminal force window parameters must be positive")
    if minimum_force_per_finger_n <= 0.0:
        raise ValueError("minimum terminal force must be positive")
    first_step = max(0, current_step - window_steps + 1)
    by_side = {}
    for side in ("left", "right"):
        maximum_by_step: dict[int, float] = {}
        for sample in samples_by_side.get(side, []):
            step = int(sample["step"])
            force_n = float(sample["force_n"])
            if first_step <= step <= current_step and math.isfinite(force_n):
                maximum_by_step[step] = max(
                    maximum_by_step.get(step, 0.0), force_n
                )
        values = list(maximum_by_step.values())
        qualifying_steps = sum(
            force_n >= minimum_force_per_finger_n for force_n in values
        )
        by_side[side] = {
            "sampled_step_count": len(values),
            "qualifying_step_count": int(qualifying_steps),
            "maximum_force_n": max(values, default=0.0),
            "mean_force_n": (
                float(sum(values) / len(values)) if values else 0.0
            ),
            "sufficient": bool(
                qualifying_steps >= minimum_qualifying_steps
            ),
        }
    return {
        "first_step": first_step,
        "last_step": current_step,
        "window_steps": window_steps,
        "minimum_force_per_finger_n": minimum_force_per_finger_n,
        "minimum_qualifying_steps_per_side": minimum_qualifying_steps,
        "by_side": by_side,
        "samples_available_both_sides": all(
            by_side[side]["sampled_step_count"] > 0
            for side in ("left", "right")
        ),
        "sufficient_both_sides": all(
            by_side[side]["sufficient"] for side in ("left", "right")
        ),
    }


def execute_persistent_composite_grasp(
    *,
    project_root: Path,
    stage,
    simulation_app,
    rep,
    overview_rgb_annotator,
    overview_camera_path: str,
    output_root: Path,
    seed: int,
    reuse_existing_composite: bool = False,
    initial_arm_positions_rad: list[float] | None = None,
    rgbd_localization: dict | None = None,
    grasp_height_offset_m: float = 0.0,
    manipulation_target_path: str = "/World/TargetRed",
    contact_target_path: str | None = None,
    manipulation_label: str = "target",
    transfer_offset_world_m: list[float] | None = None,
    placement_support_path: str | None = None,
    release_after_placement: bool = False,
    minimum_verified_lift_m: float = 0.10,
    planning_target_world_m: list[float] | None = None,
    planning_grasp_yaw_rad: float | None = None,
    planning_membership: str | None = None,
    pregrasp_settle_steps: int = 60,
    rg6_master_max_torque_nm: float = PROVISIONAL_RG6_MASTER_MAX_TORQUE_NM,
    minimum_grip_force_per_finger_n: float = MINIMUM_GRIP_FORCE_PER_FINGER_N,
    minimum_combined_grip_force_n: float | None = None,
    grip_static_friction: float = PROVISIONAL_STATIC_FRICTION,
    grip_dynamic_friction: float = PROVISIONAL_DYNAMIC_FRICTION,
    enable_micro_lift_force_validation: bool = False,
    force_controller_max_torque_nm: float | None = None,
    rg6_coupling_mode: str = "passive_mimic",
    coordinated_total_drive_effort_limit_nm: float | None = None,
    coordinated_follower_request_blend: float = (
        COORDINATED_FOLLOWER_REQUEST_BLEND
    ),
    grip_compliant_contact_stiffness_n_m: float = 0.0,
    grip_compliant_contact_damping_n_s_m: float = 0.0,
    record_debug_video: bool = True,
    physics_only_steps: bool = False,
) -> dict:
    """Run contact-gated manipulation in the caller's live Isaac stage.

    Success requires measured bilateral contact, bounded force and penetration,
    continuous target motion, and a verified lift. No fixed joint, attachment,
    or target-pose copying is introduced by this executor.
    """
    execution_started = time.perf_counter()
    if grasp_height_offset_m < 0.0 or grasp_height_offset_m > 0.08:
        raise ValueError(
            "grasp_height_offset_m must be between 0.0 and 0.08"
        )
    if seed != 0 and rgbd_localization is None and planning_target_world_m is None:
        raise ValueError(
            "A nonzero seed requires RGB-D localization for dynamic IK"
        )
    if planning_target_world_m is not None:
        if len(planning_target_world_m) != 3:
            raise ValueError("planning_target_world_m must have three values")
        planning_target_world_m = [
            float(value) for value in planning_target_world_m
        ]
        if not np.all(np.isfinite(planning_target_world_m)):
            raise ValueError("planning_target_world_m must be finite")
    if planning_grasp_yaw_rad is not None and not math.isfinite(
        planning_grasp_yaw_rad
    ):
        raise ValueError("planning_grasp_yaw_rad must be finite")
    if planning_membership not in (None, "inside", "outside"):
        raise ValueError("planning_membership must be inside, outside, or None")
    if minimum_verified_lift_m <= 0.0:
        raise ValueError("minimum_verified_lift_m must be positive")
    if pregrasp_settle_steps < 60:
        raise ValueError("pregrasp_settle_steps must be at least 60")
    if not 0.0 < rg6_master_max_torque_nm <= 2.0:
        raise ValueError("rg6_master_max_torque_nm must be in (0, 2]")
    if not 0.0 < minimum_grip_force_per_finger_n <= MAX_ALLOWED_CONTACT_FORCE_N:
        raise ValueError(
            "minimum_grip_force_per_finger_n must be in "
            f"(0, {MAX_ALLOWED_CONTACT_FORCE_N}]"
        )
    if (
        minimum_combined_grip_force_n is not None
        and not 0.0
        < minimum_combined_grip_force_n
        <= 2.0 * MAX_ALLOWED_CONTACT_FORCE_N
    ):
        raise ValueError(
            "minimum_combined_grip_force_n must be in "
            f"(0, {2.0 * MAX_ALLOWED_CONTACT_FORCE_N}] or None"
        )
    if not 0.0 < grip_dynamic_friction <= grip_static_friction <= 2.0:
        raise ValueError(
            "grip friction must satisfy 0 < dynamic <= static <= 2"
        )
    if not math.isfinite(grip_compliant_contact_stiffness_n_m) or (
        grip_compliant_contact_stiffness_n_m < 0.0
    ):
        raise ValueError(
            "grip_compliant_contact_stiffness_n_m must be non-negative"
        )
    if not math.isfinite(grip_compliant_contact_damping_n_s_m) or (
        grip_compliant_contact_damping_n_s_m < 0.0
    ):
        raise ValueError(
            "grip_compliant_contact_damping_n_s_m must be non-negative"
        )
    if (
        grip_compliant_contact_stiffness_n_m == 0.0
        and grip_compliant_contact_damping_n_s_m != 0.0
    ):
        raise ValueError(
            "compliant damping requires positive compliant stiffness"
        )
    if force_controller_max_torque_nm is None:
        force_controller_max_torque_nm = rg6_master_max_torque_nm
    if not (
        rg6_master_max_torque_nm
        <= force_controller_max_torque_nm
        <= 12.0
    ):
        raise ValueError(
            "force_controller_max_torque_nm must be between the initial "
            "RG6 torque and 12 Nm"
        )
    if rg6_coupling_mode not in ("passive_mimic", "coordinated_drives"):
        raise ValueError(
            "rg6_coupling_mode must be passive_mimic or coordinated_drives"
        )
    if rg6_coupling_mode == "coordinated_drives":
        if coordinated_total_drive_effort_limit_nm is None:
            raise ValueError(
                "coordinated_drives requires an aggregate drive-effort limit"
            )
        if not (
            force_controller_max_torque_nm
            < coordinated_total_drive_effort_limit_nm
            <= 18.0
        ):
            raise ValueError(
                "coordinated aggregate drive effort must exceed the passive "
                "limit and be no greater than 18 Nm"
            )
        if not 0.5 <= coordinated_follower_request_blend <= 1.0:
            raise ValueError(
                "coordinated follower request blend must be in [0.5, 1.0]"
            )
    elif coordinated_total_drive_effort_limit_nm is not None:
        raise ValueError(
            "coordinated drive-effort limit requires coordinated_drives"
        )
    if transfer_offset_world_m is not None:
        if len(transfer_offset_world_m) != 3:
            raise ValueError("transfer_offset_world_m must have three values")
        transfer_offset_world_m = [
            float(value) for value in transfer_offset_world_m
        ]
        if not np.all(np.isfinite(transfer_offset_world_m)):
            raise ValueError("transfer_offset_world_m must be finite")
    if placement_support_path is not None:
        if not placement_support_path.startswith("/World/"):
            raise ValueError("placement_support_path must be an absolute USD path")
        if transfer_offset_world_m is None:
            raise ValueError("placement requires a transfer offset")
        if planning_target_world_m is None:
            raise ValueError("placement requires dynamic same-stage IK planning")
    if release_after_placement and placement_support_path is None:
        raise ValueError("release_after_placement requires placement_support_path")

    import isaacsim.core.experimental.utils.app as app_utils
    import omni.usd
    from isaacsim.core.experimental.prims import Articulation
    from isaacsim.core.utils.extensions import get_extension_path_from_name
    from isaacsim.core.utils.rotations import rot_matrix_to_quat
    from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import ContactEventType
    from omni.physx.scripts.physicsUtils import add_physics_material_to_prim
    from pxr import (
        Gf,
        PhysicsSchemaTools,
        PhysxSchema,
        Sdf,
        Usd,
        UsdGeom,
        UsdPhysics,
        UsdShade,
    )

    from build_observation_video import build_frame_sequence_video

    output_root.mkdir(parents=True, exist_ok=True)
    frame_root = output_root / "frames"
    frame_root.mkdir(parents=True, exist_ok=True)

    validated_result_path = None
    if rgbd_localization is None and planning_target_world_m is None:
        validated_result_path = (
            project_root
            / "outputs"
            / "ur10e_rg6_physics"
            / "same_scene_seed000_run019_whole_arm_collision"
            / "result.json"
        )
        validated = json.loads(
            validated_result_path.read_text(encoding="utf-8")
        )
        if (
            validated.get("status") != "completed"
            or not validated.get("ur10e_link_collision_enabled")
        ):
            raise RuntimeError(
                f"Validated collision-enabled seed-0 plan is unavailable: "
                f"{validated_result_path}"
            )
        plan = validated["ik"]
        trajectory_source = (
            "prevalidated_seed0_simulator_ground_truth_debug_ik"
        )
    else:
        if (
            rgbd_localization is not None
            and rgbd_localization.get(
                "simulator_ground_truth_used_for_estimate"
            )
        ):
            raise ValueError(
                "Terminal RGB-D localization reports ground-truth leakage"
            )
        estimates = (
            rgbd_localization.get("estimates", {})
            if rgbd_localization is not None
            else {}
        )
        if planning_target_world_m is None and "selected_target" not in estimates:
            raise ValueError("Terminal localization requires selected_target")
        planning_target = np.asarray(
            planning_target_world_m
            if planning_target_world_m is not None
            else estimates["selected_target"]["center_world_m"],
            dtype=np.float64,
        )
        occluder_estimate = estimates.get("occluder_orange")
        planning_occluder = (
            np.asarray(
                occluder_estimate["center_world_m"],
                dtype=np.float64,
            )
            if occluder_estimate is not None
            else None
        )
        if not np.all(np.isfinite(planning_target)):
            raise ValueError("Terminal localization is non-finite")
        if planning_grasp_yaw_rad is not None:
            grasp_yaw = float(planning_grasp_yaw_rad)
            grasp_yaw_source = "explicit_contact_target_pinch_axis"
        elif planning_occluder is not None:
            if not np.all(np.isfinite(planning_occluder)):
                raise ValueError("Terminal occluder localization is non-finite")
            target_from_occluder_xy = (
                planning_target[:2] - planning_occluder[:2]
            )
            grasp_yaw = math.atan2(
                target_from_occluder_xy[0],
                -target_from_occluder_xy[1],
            )
            grasp_yaw_source = "target_to_explicit_occluder"
        else:
            # The corrected basket-rim scene has no explicit occluder.  Use
            # the collision-checked table-aligned seed-0 orientation, while
            # retaining the current episode's RGB-D target position.  The
            # contact and collision gates still reject an unsafe approach.
            grasp_yaw = DEFAULT_TABLE_ALIGNED_GRASP_YAW_RAD
            grasp_yaw_source = (
                "fixed_collision_checked_table_aligned_orientation"
            )
        motion_generation_path = Path(
            get_extension_path_from_name(
                "isaacsim.robot_motion.motion_generation"
            )
        )
        ur10e_config = (
            motion_generation_path
            / "motion_policy_configs"
            / "universal_robots"
            / "ur10e"
        )
        kinematics = LulaKinematicsSolver(
            robot_description_path=str(
                ur10e_config
                / "rmpflow"
                / "ur10e_robot_description.yaml"
            ),
            urdf_path=str(ur10e_config / "ur10e.urdf"),
        )
        kinematics.set_robot_base_pose(
            np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
        kinematics_frame = next(
            frame
            for frame in ("flange", "ee_link", "tool0", "wrist_3_link")
            if frame in kinematics.get_all_frame_names()
        )
        initial_ik_warm_start = np.asarray(
            (
                initial_arm_positions_rad
                if initial_arm_positions_rad is not None
                else [
                    -1.5708,
                    -1.5708,
                    1.5708,
                    -1.5708,
                    -1.5708,
                    -3.141247102,
                ]
            ),
            dtype=np.float64,
        )
        grasp_position = planning_target + np.asarray(
            [0.0, 0.0, 0.287], dtype=np.float64
        )

        def solve_local_pose_ik(
            label: str,
            requested_position: np.ndarray,
            requested_rotation: np.ndarray,
            initial_joints: np.ndarray,
        ) -> tuple[np.ndarray, dict]:
            """Use damped local differential IK without global branch flips."""
            joints = np.asarray(initial_joints, dtype=np.float64).copy()
            damping = 1.0e-4
            maximum_update_rad = 0.03
            for iteration in range(1, 81):
                position, rotation = kinematics.compute_forward_kinematics(
                    kinematics_frame,
                    joint_positions=joints,
                )
                position = np.asarray(position, dtype=np.float64)
                rotation = np.asarray(rotation, dtype=np.float64)
                position_error = requested_position - position
                orientation_error = rotation_vector_from_matrix(
                    np.asarray(requested_rotation, dtype=np.float64)
                    @ rotation.T
                )
                position_error_norm = float(np.linalg.norm(position_error))
                orientation_error_norm = float(
                    np.linalg.norm(orientation_error)
                )
                if (
                    position_error_norm <= LIFT_IK_POSITION_TOLERANCE_M
                    and orientation_error_norm
                    <= LIFT_IK_ORIENTATION_TOLERANCE_RAD
                ):
                    return joints, {
                        "solver": "lula_analytic_jacobian_damped_local_ik",
                        "iterations": iteration - 1,
                        "position_error_m": position_error_norm,
                        "orientation_error_rad": orientation_error_norm,
                    }
                jacobian = np.asarray(
                    kinematics._kinematics.jacobian(
                        joints, kinematics_frame
                    ),
                    dtype=np.float64,
                )
                error = np.concatenate(
                    [position_error, orientation_error]
                )
                regularized = (
                    jacobian @ jacobian.T
                    + (damping**2) * np.eye(6, dtype=np.float64)
                )
                update = jacobian.T @ np.linalg.solve(regularized, error)
                update_scale = max(
                    1.0,
                    float(np.max(np.abs(update))) / maximum_update_rad,
                )
                joints = joints + update / update_scale
            raise RuntimeError(
                f"{label} local IK did not converge in 80 iterations"
            )

        # A parallel gripper's pinch line is unoriented: yaw and yaw + pi are
        # physically equivalent for both the cover handle and the target.
        # Lula can nevertheless map the two wrist poses to different UR10e
        # elbow branches.  Evaluate both from the *current* same-stage arm
        # state and accept only a dense Cartesian descent whose consecutive
        # joint solutions remain local.  This is especially important after
        # cover placement, where the arm no longer starts from the original
        # center-view posture.
        wall_clearance_ranking = []
        if planning_membership == "outside":
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]
            )
            wall_bounds = []
            for wall_name in (
                "WallFront",
                "WallBack",
                "WallLeft",
                "WallRight",
            ):
                collision_prim = stage.GetPrimAtPath(
                    f"/World/OpenContainer/CollisionApproximation/{wall_name}"
                )
                wall_prim = (
                    collision_prim
                    if collision_prim.IsValid()
                    else stage.GetPrimAtPath(f"/World/OpenContainer/{wall_name}")
                )
                if not wall_prim.IsValid():
                    continue
                aligned = bbox_cache.ComputeWorldBound(
                    wall_prim
                ).ComputeAlignedRange()
                wall_bounds.append(
                    (
                        np.asarray(aligned.GetMin(), dtype=np.float64),
                        np.asarray(aligned.GetMax(), dtype=np.float64),
                    )
                )
            grasp_yaw_candidates, wall_clearance_ranking = (
                rank_outside_container_grasp_yaws(
                    grasp_yaw,
                    planning_target[:2],
                    wall_bounds,
                )
            )
            grasp_yaw_source = (
                "outside_container_wall_clearance_ranked_from_"
                f"{grasp_yaw_source}"
            )
        else:
            grasp_yaw_candidates = parallel_gripper_yaw_candidates(grasp_yaw)
        descent_planning_attempts = []
        selected_descent = None
        for candidate_yaw in grasp_yaw_candidates:
            candidate_orientation = np.asarray(
                [
                    0.0,
                    math.cos(candidate_yaw * 0.5),
                    math.sin(candidate_yaw * 0.5),
                    0.0,
                ],
                dtype=np.float64,
            )
            candidate_warm_start = initial_ik_warm_start.copy()
            candidate_plan = []
            candidate_rotation = None
            failure_reason = None
            maximum_local_step = 0.0
            for offset_m in DESCENT_OFFSETS_M:
                waypoint_position = grasp_position + np.asarray(
                    [0.0, 0.0, offset_m], dtype=np.float64
                )
                local_solver = None
                if not candidate_plan:
                    joints, ik_success = kinematics.compute_inverse_kinematics(
                        kinematics_frame,
                        waypoint_position,
                        candidate_orientation,
                        warm_start=candidate_warm_start,
                        position_tolerance=0.003,
                        orientation_tolerance=0.03,
                    )
                else:
                    try:
                        joints, local_solver = solve_local_pose_ik(
                            "Dynamic terminal descent",
                            waypoint_position,
                            candidate_rotation,
                            candidate_warm_start,
                        )
                        ik_success = True
                    except RuntimeError as error:
                        failure_reason = (
                            f"local_ik_failed_at_offset_{offset_m:.2f}_m: "
                            f"{error}"
                        )
                        break
                joints = closest_equivalent_joint_configuration(
                    np.asarray(joints, dtype=np.float64),
                    candidate_warm_start,
                )
                maximum_joint_step = float(
                    np.max(np.abs(joints - candidate_warm_start))
                )
                if not ik_success:
                    failure_reason = (
                        f"ik_failed_at_offset_{offset_m:.2f}_m"
                    )
                    break
                if (
                    candidate_plan
                    and maximum_joint_step > MAX_DESCENT_IK_JOINT_STEP_RAD
                ):
                    failure_reason = (
                        "ik_branch_change_at_offset_"
                        f"{offset_m:.2f}_m_step_{maximum_joint_step:.6f}_rad"
                    )
                    break
                maximum_local_step = max(
                    maximum_local_step,
                    maximum_joint_step if candidate_plan else 0.0,
                )
                if candidate_rotation is None:
                    _first_position, candidate_rotation = (
                        kinematics.compute_forward_kinematics(
                            kinematics_frame,
                            joint_positions=joints,
                        )
                    )
                    candidate_rotation = np.asarray(
                        candidate_rotation, dtype=np.float64
                    )
                candidate_plan.append(
                    {
                        "offset_m": offset_m,
                        "position_world_m": waypoint_position.tolist(),
                        "joints_rad": joints.tolist(),
                        "ik_success": True,
                        "solver": (
                            local_solver["solver"]
                            if local_solver is not None
                            else "lula_global_warm_start"
                        ),
                        "local_ik": local_solver,
                        "maximum_wrapped_joint_step_rad": (
                            maximum_joint_step if candidate_plan else None
                        ),
                    }
                )
                candidate_warm_start = joints
            descent_planning_attempts.append(
                {
                    "grasp_yaw_rad": candidate_yaw,
                    "status": (
                        "accepted" if failure_reason is None else "rejected"
                    ),
                    "failure_reason": failure_reason,
                    "waypoints_solved": len(candidate_plan),
                    "maximum_local_joint_step_rad": maximum_local_step,
                }
            )
            if failure_reason is None:
                selected_descent = (
                    candidate_yaw,
                    candidate_orientation,
                    candidate_plan,
                    candidate_warm_start,
                )
                break
        if selected_descent is None:
            raise RuntimeError(
                "No continuous dynamic terminal descent IK path: "
                f"{descent_planning_attempts}"
            )
        grasp_yaw, downward_orientation, descent_plan, warm_start = (
            selected_descent
        )
        exact_grasp_position, exact_grasp_rotation = (
            kinematics.compute_forward_kinematics(
                kinematics_frame,
                joint_positions=np.asarray(warm_start, dtype=np.float64),
            )
        )
        exact_grasp_position = np.asarray(
            exact_grasp_position, dtype=np.float64
        )
        exact_grasp_orientation = rot_matrix_to_quat(
            np.asarray(exact_grasp_rotation, dtype=np.float64)
        )

        def validate_lift_ik_pose(
            label: str,
            joints: np.ndarray,
            requested_position: np.ndarray,
        ) -> dict:
            """Reject lift IK that rotates the long RG6 tool sideways."""
            solved_position, solved_rotation = (
                kinematics.compute_forward_kinematics(
                    kinematics_frame,
                    joint_positions=np.asarray(joints, dtype=np.float64),
                )
            )
            solved_position = np.asarray(solved_position, dtype=np.float64)
            solved_rotation = np.asarray(solved_rotation, dtype=np.float64)
            position_error_m = float(
                np.linalg.norm(solved_position - requested_position)
            )
            orientation_error_rad = rotation_angle_rad(
                solved_rotation @ np.asarray(exact_grasp_rotation).T
            )
            if (
                position_error_m > MAX_LIFT_IK_POSITION_ERROR_M
                or orientation_error_rad > MAX_LIFT_IK_ORIENTATION_ERROR_RAD
            ):
                raise RuntimeError(
                    f"{label} IK Cartesian validation failed: "
                    f"position_error={position_error_m:.6f} m, "
                    f"orientation_error={orientation_error_rad:.6f} rad"
                )
            return {
                "requested_position_world_m": requested_position.tolist(),
                "solved_fk_position_world_m": solved_position.tolist(),
                "position_error_m": position_error_m,
                "orientation_error_rad": orientation_error_rad,
            }

        def solve_local_cartesian_ik(
            label: str,
            requested_position: np.ndarray,
            initial_joints: np.ndarray,
        ) -> tuple[np.ndarray, dict]:
            """Use damped local differential IK without global branch flips."""
            return solve_local_pose_ik(
                label,
                requested_position,
                np.asarray(exact_grasp_rotation, dtype=np.float64),
                initial_joints,
            )

        def solve_vertical_lift_path(
            label: str,
            final_height_m: float,
            waypoint_count: int,
        ) -> list[dict]:
            """Solve a dense local Cartesian-Z path from the grasp pose."""
            path = []
            previous_joints = np.asarray(warm_start, dtype=np.float64)
            for index in range(1, waypoint_count + 1):
                fraction = index / waypoint_count
                requested_position = exact_grasp_position + np.asarray(
                    [0.0, 0.0, final_height_m * fraction],
                    dtype=np.float64,
                )
                joints, local_solver = solve_local_cartesian_ik(
                    label,
                    requested_position,
                    previous_joints,
                )
                joints = closest_equivalent_joint_configuration(
                    np.asarray(joints, dtype=np.float64),
                    previous_joints,
                )
                maximum_joint_step_rad = float(
                    np.max(np.abs(joints - previous_joints))
                )
                if maximum_joint_step_rad > MAX_LIFT_IK_JOINT_STEP_RAD:
                    raise RuntimeError(
                        f"{label} IK changed branch at waypoint "
                        f"{index}/{waypoint_count}: "
                        f"maximum_joint_step={maximum_joint_step_rad:.6f} rad"
                    )
                validation = validate_lift_ik_pose(
                    label,
                    joints,
                    requested_position,
                )
                path.append(
                    {
                        "index": index,
                        "fraction": fraction,
                        "height_m": final_height_m * fraction,
                        "joints_rad": joints.tolist(),
                        "maximum_joint_step_rad": maximum_joint_step_rad,
                        "local_ik": local_solver,
                        "ik_cartesian_validation": validation,
                    }
                )
                previous_joints = joints
            return path

        micro_lift_path = solve_vertical_lift_path(
            "Dynamic terminal micro-lift",
            MICRO_LIFT_HEIGHT_M,
            MICRO_LIFT_CARTESIAN_WAYPOINTS,
        )
        full_lift_path = solve_vertical_lift_path(
            "Dynamic terminal lift",
            0.18,
            FULL_LIFT_CARTESIAN_WAYPOINTS,
        )
        micro_lift_position = np.asarray(
            micro_lift_path[-1]["ik_cartesian_validation"][
                "requested_position_world_m"
            ],
            dtype=np.float64,
        )
        micro_lift_joints = np.asarray(
            micro_lift_path[-1]["joints_rad"], dtype=np.float64
        )
        lift_position = np.asarray(
            full_lift_path[-1]["ik_cartesian_validation"][
                "requested_position_world_m"
            ],
            dtype=np.float64,
        )
        lift_joints = np.asarray(
            full_lift_path[-1]["joints_rad"], dtype=np.float64
        )
        transfer_position = None
        transfer_joints = None
        transfer_waypoints = []
        placement_position = None
        placement_joints = None
        placement_waypoints = []
        retreat_position = None
        retreat_joints = None
        retreat_waypoints = []
        placement_geometry = None
        if transfer_offset_world_m is not None:
            transfer_position = exact_grasp_position + np.asarray(
                transfer_offset_world_m, dtype=np.float64
            )
            transfer_warm_start = np.asarray(
                lift_joints, dtype=np.float64
            )
            for index in range(1, TRANSFER_CARTESIAN_WAYPOINTS + 1):
                fraction = index / TRANSFER_CARTESIAN_WAYPOINTS
                waypoint_position = (
                    lift_position
                    + fraction * (transfer_position - lift_position)
                )
                waypoint_joints, local_solver = solve_local_cartesian_ik(
                    "Dynamic terminal transfer",
                    waypoint_position,
                    transfer_warm_start,
                )
                waypoint_joints = np.asarray(
                    waypoint_joints, dtype=np.float64
                )
                waypoint_joints = closest_equivalent_joint_configuration(
                    waypoint_joints, transfer_warm_start
                )
                wrapped_delta = waypoint_joints - transfer_warm_start
                maximum_joint_step = float(
                    np.max(np.abs(wrapped_delta))
                )
                if maximum_joint_step > MAX_LIFT_IK_JOINT_STEP_RAD:
                    raise RuntimeError(
                        "Dynamic terminal transfer IK changed branch at "
                        f"fraction {fraction:.3f}: "
                        f"maximum_joint_step={maximum_joint_step:.6f} rad"
                    )
                waypoint_validation = validate_lift_ik_pose(
                    "Dynamic terminal transfer",
                    waypoint_joints,
                    waypoint_position,
                )
                transfer_waypoints.append(
                    {
                        "fraction": fraction,
                        "position_world_m": waypoint_position.tolist(),
                        "joints_rad": waypoint_joints.tolist(),
                        "maximum_wrapped_joint_step_rad": maximum_joint_step,
                        "local_ik": local_solver,
                        "ik_cartesian_validation": waypoint_validation,
                    }
                )
                transfer_warm_start = waypoint_joints
            transfer_joints = transfer_warm_start

        if placement_support_path is not None:
            support_prim = stage.GetPrimAtPath(placement_support_path)
            target_root_prim = stage.GetPrimAtPath(manipulation_target_path)
            target_plate_prim = stage.GetPrimAtPath(
                f"{manipulation_target_path}/Plate"
            )
            basket_bottom_prim = stage.GetPrimAtPath(
                "/World/OpenContainer/Bottom"
            )
            if not support_prim.IsValid():
                raise RuntimeError(
                    f"Placement support is missing: {placement_support_path}"
                )
            if not target_root_prim.IsValid():
                raise RuntimeError(
                    f"Placement target is missing: {manipulation_target_path}"
                )
            if not target_plate_prim.IsValid():
                target_plate_prim = target_root_prim

            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(),
                [
                    UsdGeom.Tokens.default_,
                    UsdGeom.Tokens.proxy,
                    UsdGeom.Tokens.render,
                ],
                useExtentsHint=True,
            )

            def aligned_world_bounds(prim) -> tuple[np.ndarray, np.ndarray]:
                aligned_range = bbox_cache.ComputeWorldBound(
                    prim
                ).ComputeAlignedRange()
                return (
                    np.asarray(aligned_range.GetMin(), dtype=np.float64),
                    np.asarray(aligned_range.GetMax(), dtype=np.float64),
                )

            support_min, support_max = aligned_world_bounds(support_prim)
            target_min, target_max = aligned_world_bounds(target_plate_prim)
            target_root_position = np.asarray(
                omni.usd.get_world_transform_matrix(
                    target_root_prim
                ).ExtractTranslation(),
                dtype=np.float64,
            )
            target_bottom_offset = float(
                target_min[2] - target_root_position[2]
            )
            desired_target_root_z = supported_root_height_m(
                support_top_z_m=float(support_max[2]),
                target_bottom_offset_from_root_m=target_bottom_offset,
            )
            expected_transferred_target_root = (
                target_root_position
                + np.asarray(transfer_offset_world_m, dtype=np.float64)
            )
            placement_delta_z = float(
                desired_target_root_z
                - expected_transferred_target_root[2]
            )
            if not -0.50 <= placement_delta_z < -0.02:
                raise RuntimeError(
                    "Placement lowering distance is outside the conservative "
                    f"range: {placement_delta_z:.6f} m"
                )

            horizontal_shift = np.asarray(
                transfer_offset_world_m[:2], dtype=np.float64
            )
            planned_target_min_xy = target_min[:2] + horizontal_shift
            planned_target_max_xy = target_max[:2] + horizontal_shift
            support_metrics = planar_support_metrics(
                support_min_xy=support_min[:2],
                support_max_xy=support_max[:2],
                target_min_xy=planned_target_min_xy,
                target_max_xy=planned_target_max_xy,
            )
            support_edge_margin_m = support_metrics[
                "full_footprint_margin_m"
            ]
            if (
                support_metrics["center_margin_m"]
                < MINIMUM_SUPPORT_CENTER_MARGIN_M
                or support_metrics["overlap_fraction"]
                < MINIMUM_SUPPORT_OVERLAP_FRACTION
            ):
                raise RuntimeError(
                    "Planned cover placement lacks stable planar support: "
                    f"center_margin={support_metrics['center_margin_m']:.6f} m, "
                    "required_center_margin="
                    f"{MINIMUM_SUPPORT_CENTER_MARGIN_M:.6f} m, "
                    f"overlap_fraction={support_metrics['overlap_fraction']:.6f}, "
                    "required_overlap_fraction="
                    f"{MINIMUM_SUPPORT_OVERLAP_FRACTION:.6f}; "
                    f"support_min={support_min.tolist()}, "
                    f"support_max={support_max.tolist()}, "
                    f"planned_target_min_xy={planned_target_min_xy.tolist()}, "
                    f"planned_target_max_xy={planned_target_max_xy.tolist()}"
                )

            basket_clearance_m = None
            if basket_bottom_prim.IsValid():
                basket_min, basket_max = aligned_world_bounds(
                    basket_bottom_prim
                )
                axis_gaps = np.maximum(
                    np.maximum(
                        basket_min[:2] - planned_target_max_xy,
                        planned_target_min_xy - basket_max[:2],
                    ),
                    0.0,
                )
                basket_clearance_m = float(np.linalg.norm(axis_gaps))
                if basket_clearance_m < 0.03:
                    raise RuntimeError(
                        "Planned cover placement is too close to the basket: "
                        f"{basket_clearance_m:.6f} m"
                    )

            def solve_cartesian_segment_path(
                label: str,
                start_position: np.ndarray,
                end_position: np.ndarray,
                start_joints: np.ndarray,
                waypoint_count: int,
            ) -> list[dict]:
                path = []
                previous_joints = np.asarray(start_joints, dtype=np.float64)
                for index in range(1, waypoint_count + 1):
                    fraction = index / waypoint_count
                    requested_position = (
                        start_position
                        + fraction * (end_position - start_position)
                    )
                    joints, local_solver = solve_local_cartesian_ik(
                        label,
                        requested_position,
                        previous_joints,
                    )
                    joints = closest_equivalent_joint_configuration(
                        np.asarray(joints, dtype=np.float64),
                        previous_joints,
                    )
                    maximum_joint_step_rad = float(
                        np.max(np.abs(joints - previous_joints))
                    )
                    if maximum_joint_step_rad > MAX_LIFT_IK_JOINT_STEP_RAD:
                        raise RuntimeError(
                            f"{label} IK changed branch at waypoint "
                            f"{index}/{waypoint_count}: maximum_joint_step="
                            f"{maximum_joint_step_rad:.6f} rad"
                        )
                    validation = validate_lift_ik_pose(
                        label, joints, requested_position
                    )
                    path.append(
                        {
                            "index": index,
                            "fraction": fraction,
                            "position_world_m": requested_position.tolist(),
                            "joints_rad": joints.tolist(),
                            "maximum_joint_step_rad": maximum_joint_step_rad,
                            "local_ik": local_solver,
                            "ik_cartesian_validation": validation,
                        }
                    )
                    previous_joints = joints
                return path

            placement_position = transfer_position + np.asarray(
                [0.0, 0.0, placement_delta_z], dtype=np.float64
            )
            placement_waypoints = solve_cartesian_segment_path(
                "Dynamic supported placement",
                transfer_position,
                placement_position,
                transfer_joints,
                PLACEMENT_CARTESIAN_WAYPOINTS,
            )
            placement_joints = np.asarray(
                placement_waypoints[-1]["joints_rad"], dtype=np.float64
            )
            retreat_position = placement_position + np.asarray(
                [0.0, 0.0, RETREAT_DISTANCE_M], dtype=np.float64
            )
            retreat_waypoints = solve_cartesian_segment_path(
                "Dynamic post-release retreat",
                placement_position,
                retreat_position,
                placement_joints,
                RETREAT_CARTESIAN_WAYPOINTS,
            )
            retreat_joints = np.asarray(
                retreat_waypoints[-1]["joints_rad"], dtype=np.float64
            )
            placement_geometry = {
                "support_path": placement_support_path,
                "support_bounds_world_m": {
                    "minimum": support_min.tolist(),
                    "maximum": support_max.tolist(),
                },
                "target_plate_bounds_before_transfer_world_m": {
                    "minimum": target_min.tolist(),
                    "maximum": target_max.tolist(),
                },
                "target_root_before_transfer_world_m": (
                    target_root_position.tolist()
                ),
                "target_bottom_offset_from_root_m": target_bottom_offset,
                "desired_target_root_z_m": desired_target_root_z,
                "placement_delta_z_m": placement_delta_z,
                "commanded_support_penetration_m": (
                    PLACEMENT_SUPPORT_PENETRATION_M
                ),
                "support_edge_margin_m": support_edge_margin_m,
                "support_center_margin_m": support_metrics[
                    "center_margin_m"
                ],
                "support_overlap_fraction": support_metrics[
                    "overlap_fraction"
                ],
                "basket_horizontal_clearance_m": basket_clearance_m,
            }
        plan = {
            "frame": kinematics_frame,
            "grasp_planning_target_position_world_m": (
                planning_target.tolist()
            ),
            "grasp_planning_occluder_position_world_m": (
                planning_occluder.tolist()
                if planning_occluder is not None
                else None
            ),
            "grasp_planning_position_source": (
                "simulator_cover_handle_pose_physics_pilot"
                if planning_target_world_m is not None
                else "qwen_selected_candidate_masked_rgbd"
            ),
            "orientation_wxyz": downward_orientation.tolist(),
            "exact_grasp_fk_position_world_m": exact_grasp_position.tolist(),
            "exact_grasp_fk_orientation_wxyz": np.asarray(
                exact_grasp_orientation, dtype=np.float64
            ).tolist(),
            "collision_avoidance_grasp_yaw_rad": grasp_yaw,
            "grasp_yaw_source": grasp_yaw_source,
            "planning_membership": planning_membership,
            "outside_container_wall_clearance_ranking": (
                wall_clearance_ranking
            ),
            "descent_planning_attempts": descent_planning_attempts,
            "descent_waypoints": descent_plan,
            "lift_joints_rad": np.asarray(
                lift_joints, dtype=np.float64
            ).tolist(),
            "micro_lift_position_world_m": micro_lift_position.tolist(),
            "micro_lift_joints_rad": np.asarray(
                micro_lift_joints, dtype=np.float64
            ).tolist(),
            "micro_lift_cartesian_waypoints": micro_lift_path,
            "full_lift_cartesian_waypoints": full_lift_path,
            "micro_lift_ik_cartesian_validation": micro_lift_path[-1][
                "ik_cartesian_validation"
            ],
            "lift_ik_cartesian_validation": full_lift_path[-1][
                "ik_cartesian_validation"
            ],
            "transfer_position_world_m": (
                transfer_position.tolist()
                if transfer_position is not None
                else None
            ),
            "transfer_joints_rad": (
                np.asarray(transfer_joints, dtype=np.float64).tolist()
                if transfer_joints is not None
                else None
            ),
            "transfer_waypoints": transfer_waypoints,
            "placement_position_world_m": (
                placement_position.tolist()
                if placement_position is not None
                else None
            ),
            "placement_joints_rad": (
                placement_joints.tolist()
                if placement_joints is not None
                else None
            ),
            "placement_waypoints": placement_waypoints,
            "retreat_position_world_m": (
                retreat_position.tolist()
                if retreat_position is not None
                else None
            ),
            "retreat_joints_rad": (
                retreat_joints.tolist()
                if retreat_joints is not None
                else None
            ),
            "retreat_waypoints": retreat_waypoints,
            "placement_geometry": placement_geometry,
        }
        trajectory_source = (
            "same_episode_qwen_selected_mask_rgbd_dynamic_debug_ik"
        )
    descent_plan = plan["descent_waypoints"]
    if grasp_height_offset_m:
        descent_plan = [
            waypoint
            for waypoint in descent_plan
            if float(waypoint["offset_m"])
            >= grasp_height_offset_m - 1e-9
        ]
        if not descent_plan:
            raise RuntimeError(
                "No validated descent waypoint remains for the requested "
                f"grasp height offset {grasp_height_offset_m:.3f} m"
            )
    lift_joints = np.asarray(plan["lift_joints_rad"], dtype=np.float64)
    micro_lift_joints = (
        np.asarray(plan["micro_lift_joints_rad"], dtype=np.float64)
        if "micro_lift_joints_rad" in plan
        else None
    )
    micro_lift_waypoints = [
        np.asarray(waypoint["joints_rad"], dtype=np.float64)
        for waypoint in plan.get("micro_lift_cartesian_waypoints", [])
    ]
    if not micro_lift_waypoints and micro_lift_joints is not None:
        micro_lift_waypoints = [micro_lift_joints]
    full_lift_waypoints = [
        (
            float(waypoint["height_m"]),
            np.asarray(waypoint["joints_rad"], dtype=np.float64),
        )
        for waypoint in plan.get("full_lift_cartesian_waypoints", [])
    ]
    if not full_lift_waypoints:
        full_lift_waypoints = [(0.18, lift_joints)]
    if enable_micro_lift_force_validation and micro_lift_joints is None:
        raise RuntimeError(
            "Micro-lift force validation requires a dynamic micro-lift IK pose"
        )
    transfer_joints = (
        np.asarray(plan["transfer_joints_rad"], dtype=np.float64)
        if plan.get("transfer_joints_rad") is not None
        else None
    )
    transfer_waypoints = plan.get("transfer_waypoints", [])
    placement_joints = (
        np.asarray(plan["placement_joints_rad"], dtype=np.float64)
        if plan.get("placement_joints_rad") is not None
        else None
    )
    placement_waypoints = plan.get("placement_waypoints", [])
    retreat_joints = (
        np.asarray(plan["retreat_joints_rad"], dtype=np.float64)
        if plan.get("retreat_joints_rad") is not None
        else None
    )
    retreat_waypoints = plan.get("retreat_waypoints", [])

    import_result = json.loads(
        (
            project_root
            / "assets"
            / "robots"
            / "ur10e_rg6"
            / "isaac6_import"
            / "import_result.json"
        ).read_text(encoding="utf-8")
    )
    asset = Path(import_result["output_usd"]).resolve()

    def world_position(prim) -> np.ndarray:
        return np.asarray(
            omni.usd.get_world_transform_matrix(prim).ExtractTranslation(),
            dtype=np.float64,
        )

    def world_rotation(prim) -> np.ndarray:
        return np.asarray(
            omni.usd.get_world_transform_matrix(prim).ExtractRotationMatrix(),
            dtype=np.float64,
        )

    def fresh_aligned_world_bounds(path: str) -> tuple[np.ndarray, np.ndarray]:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Bounding-box prim is missing: {path}")
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.proxy,
                UsdGeom.Tokens.render,
            ],
            useExtentsHint=True,
        )
        aligned_range = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        return (
            np.asarray(aligned_range.GetMin(), dtype=np.float64),
            np.asarray(aligned_range.GetMax(), dtype=np.float64),
        )

    def set_cube(cube, position, full_extents) -> None:
        cube.CreateSizeAttr().Set(1.0)
        xform = UsdGeom.Xformable(cube)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddScaleOp().Set(Gf.Vec3f(*full_extents))

    def translate_root(path: str) -> None:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return
        xformable = UsdGeom.Xformable(prim)
        translate_ops = [
            op
            for op in xformable.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
        ]
        if translate_ops:
            current = np.asarray(translate_ops[0].Get(), dtype=np.float64)
            translate_ops[0].Set(Gf.Vec3d(*(current + WORLD_TO_ROBOT_BASE)))
        else:
            xformable.AddTranslateOp().Set(Gf.Vec3d(*WORLD_TO_ROBOT_BASE))

    if reuse_existing_composite:
        if initial_arm_positions_rad is None:
            raise ValueError(
                "initial_arm_positions_rad is required when reusing the "
                "live observation composite"
            )
        if len(initial_arm_positions_rad) != 6:
            raise ValueError("Expected six live UR10e arm positions")

    prepared_robot = stage.GetPrimAtPath("/World/UR10eRG6")
    executor_already_prepared = bool(
        prepared_robot.IsValid()
        and prepared_robot.GetCustomDataByKey(
            "persistent_composite_executor_prepared"
        )
    )
    rebuild_persistent_physics = should_rebuild_persistent_physics(
        reuse_existing_composite=reuse_existing_composite,
        executor_already_prepared=executor_already_prepared,
    )
    if rebuild_persistent_physics:
        app_utils.stop()
        for _ in range(3):
            simulation_app.update()

    if not reuse_existing_composite:
        # The validated standalone plan uses robot-base coordinates. Shift the
        # authored environment once; a live composite is already in the frame
        # used by its RGB-D localization and must not be shifted again.
        for path in ENVIRONMENT_ROOTS:
            translate_root(path)

    overview_camera = stage.GetPrimAtPath(overview_camera_path)
    if overview_camera.IsValid() and not reuse_existing_composite:
        matrix = omni.usd.get_world_transform_matrix(overview_camera)
        matrix.SetTranslateOnly(
            Gf.Vec3d(*(np.asarray(matrix.ExtractTranslation()) + WORLD_TO_ROBOT_BASE))
        )
        xformable = UsdGeom.Xformable(overview_camera)
        xformable.ClearXformOpOrder()
        xformable.MakeMatrixXform().Set(matrix)

    legacy_system = stage.GetPrimAtPath("/World/RobotSystem")
    if legacy_system.IsValid():
        legacy_system.SetActive(False)

    robot_path = "/World/UR10eRG6"
    existing_robot = stage.GetPrimAtPath(robot_path)
    if reuse_existing_composite:
        if not existing_robot.IsValid() or not existing_robot.IsActive():
            raise RuntimeError(
                "Live observation composite is unavailable for terminal grasp"
            )
    else:
        if existing_robot.IsValid():
            existing_robot.SetActive(False)
        robot_prim = stage.DefinePrim(robot_path, "Xform")
        robot_prim.GetReferences().AddReference(str(asset))
        variant = robot_prim.GetVariantSets().GetVariantSet("Physics")
        if variant.IsValid():
            variant.SetVariantSelection("physx")
        for _ in range(30):
            simulation_app.update()

    arm_names = [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    rg6_names = [
        "rg6_finger_joint",
        "rg6_left_inner_knuckle_joint",
        "rg6_left_inner_finger_joint",
        "rg6_right_outer_knuckle_joint",
        "rg6_right_inner_knuckle_joint",
        "rg6_right_inner_finger_joint",
    ]
    rg6_master_name = "rg6_finger_joint"
    rg6_follower_ratios = {
        "rg6_left_inner_knuckle_joint": -1.0,
        "rg6_left_inner_finger_joint": 1.0,
        "rg6_right_outer_knuckle_joint": -1.0,
        "rg6_right_inner_knuckle_joint": -1.0,
        "rg6_right_inner_finger_joint": 1.0,
    }
    for name in arm_names:
        joint = stage.GetPrimAtPath(f"{robot_path}/Physics/{name}")
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if drive:
            drive.CreateStiffnessAttr().Set(1000.0)
            drive.CreateDampingAttr().Set(50.0)
            drive.CreateMaxForceAttr().Set(400.0)

    follower_drive_removal: dict[str, bool] = {}
    mimic_api_removal: dict[str, bool] = {}
    rg6_drives: dict[str, object] = {}
    restored_mimic_requires_physics_rebuild = False
    active_robot_prim = stage.GetPrimAtPath(robot_path)
    if not active_robot_prim.IsValid():
        raise RuntimeError("Persistent composite robot prim is unavailable")
    if rg6_coupling_mode == "passive_mimic":
        # Preserve the original importer coupling for non-cover tasks.  A
        # preceding cover manipulation may have temporarily replaced the
        # mimic constraints with coordinated follower drives; restore the
        # authored Newton schema in memory before grasping the target.
        coordinated_prepared = bool(
            active_robot_prim.GetCustomDataByKey(
                "rg6_coordinated_drives_prepared"
            )
        )
        for name in rg6_follower_ratios:
            joint = stage.GetPrimAtPath(f"{robot_path}/Physics/{name}")
            if "NewtonMimicAPI" not in joint.GetAppliedSchemas():
                if not coordinated_prepared:
                    raise RuntimeError(
                        f"RG6 follower lacks NewtonMimicAPI: {name}"
                    )
                if not joint.ApplyAPI("NewtonMimicAPI"):
                    raise RuntimeError(
                        f"Could not restore NewtonMimicAPI: {name}"
                    )
                mimic_joint = joint.GetRelationship("newton:mimicJoint")
                if not mimic_joint:
                    mimic_joint = joint.CreateRelationship(
                        "newton:mimicJoint"
                    )
                mimic_joint.SetTargets(
                    [Sdf.Path(f"{robot_path}/Physics/{rg6_master_name}")]
                )
                mimic_coefficient = joint.GetAttribute("newton:mimicCoef1")
                if not mimic_coefficient:
                    mimic_coefficient = joint.CreateAttribute(
                        "newton:mimicCoef1", Sdf.ValueTypeNames.Float
                    )
                mimic_coefficient.Set(float(rg6_follower_ratios[name]))
                if not mimic_joint.HasAuthoredTargets() or not (
                    mimic_coefficient.HasAuthoredValueOpinion()
                ):
                    raise RuntimeError(
                        "Restored RG6 mimic schema lacks authored relation "
                        f"properties: {name}"
                    )
            had_drive = bool(UsdPhysics.DriveAPI.Get(joint, "angular"))
            if had_drive:
                joint.RemoveAPI(UsdPhysics.DriveAPI, "angular")
            follower_drive_removal[name] = (
                had_drive and not bool(
                    UsdPhysics.DriveAPI.Get(joint, "angular")
                )
            )
        if not all(follower_drive_removal.values()):
            raise RuntimeError(
                f"Could not remove all RG6 follower drives: "
                f"{follower_drive_removal}"
            )
        active_robot_prim.SetCustomDataByKey(
            "rg6_coordinated_drives_prepared", False
        )
        restored_mimic_requires_physics_rebuild = coordinated_prepared
        drive_names = (rg6_master_name,)
    else:
        # The loaded passive mimic stalled one jaw.  The fixture-validated
        # development proxy removes mimic APIs before enabling coordinated
        # six-joint drives.  Never retain mimic and follower drives together.
        coordinated_prepared = bool(
            active_robot_prim.GetCustomDataByKey(
                "rg6_coordinated_drives_prepared"
            )
        )
        for name in rg6_follower_ratios:
            joint = stage.GetPrimAtPath(f"{robot_path}/Physics/{name}")
            has_mimic = "NewtonMimicAPI" in joint.GetAppliedSchemas()
            if has_mimic:
                joint.RemoveAPI("NewtonMimicAPI")
            elif not coordinated_prepared:
                raise RuntimeError(
                    f"RG6 follower lacks NewtonMimicAPI before coordinated "
                    f"drive preparation: {name}"
                )
            elif not UsdPhysics.DriveAPI.Get(joint, "angular"):
                raise RuntimeError(
                    "Previously prepared RG6 follower has no coordinated "
                    f"drive: {name}"
                )
            mimic_api_removal[name] = (
                "NewtonMimicAPI" not in joint.GetAppliedSchemas()
            )
        if not all(mimic_api_removal.values()):
            raise RuntimeError(
                f"Could not remove all RG6 mimic APIs: {mimic_api_removal}"
            )
        active_robot_prim.SetCustomDataByKey(
            "rg6_coordinated_drives_prepared", True
        )
        drive_names = tuple(rg6_names)

    for name in drive_names:
        joint = stage.GetPrimAtPath(f"{robot_path}/Physics/{name}")
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        if rg6_coupling_mode == "coordinated_drives":
            drive.CreateStiffnessAttr().Set(20.0)
            drive.CreateDampingAttr().Set(1.0)
        rg6_drives[name] = drive

    def set_grip_drive_effort(total_effort_nm: float) -> None:
        per_drive_effort_nm = total_effort_nm / len(rg6_drives)
        for drive in rg6_drives.values():
            drive.CreateMaxForceAttr().Set(per_drive_effort_nm)

    set_grip_drive_effort(rg6_master_max_torque_nm)

    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(f"{robot_path}/Geometry/"):
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(True)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxContactReportAPI.Apply(
                prim
            ).CreateThresholdAttr().Set(0.0)

    for path in (
        "/World/WorkBench/Top",
        "/World/WorkMat",
        "/World/OpenContainer/Bottom",
        "/World/OpenContainer/WallFront",
        "/World/OpenContainer/WallBack",
        "/World/OpenContainer/WallLeft",
        "/World/OpenContainer/WallRight",
    ):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            UsdPhysics.CollisionAPI.Apply(prim)

    target_prim = stage.GetPrimAtPath(manipulation_target_path)
    if not target_prim.IsValid():
        raise RuntimeError(
            f"Manipulation target is missing: {manipulation_target_path}"
        )
    existing_dynamic_assembly = manipulation_target_path != "/World/TargetRed"
    target_position = (
        world_position(target_prim)
        if existing_dynamic_assembly
        else (
        world_position(target_prim)
        if rgbd_localization is not None and reuse_existing_composite
        else np.asarray(
            plan["settled_target_position_world_m"], dtype=np.float64
        )
        )
    )
    household_mug_physics_proxy = (
        not existing_dynamic_assembly and target_prim.GetTypeName() == "Xform"
    )
    if existing_dynamic_assembly:
        if not target_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(
                "Existing manipulation assembly must be a rigid body"
            )
        target_collision = stage.GetPrimAtPath(
            contact_target_path or manipulation_target_path
        )
        if not target_collision.IsValid() or not target_collision.HasAPI(
            UsdPhysics.CollisionAPI
        ):
            raise RuntimeError(
                "Existing manipulation contact target must have collision: "
                f"{contact_target_path}"
            )
        target_mass_kg = float(
            UsdPhysics.MassAPI(target_prim).GetMassAttr().Get() or 0.0
        )
        target_position_semantics = "existing_dynamic_assembly_root"
    elif household_mug_physics_proxy:
        # The household pilot keeps the visible mug mesh as children of this
        # Xform.  Author a collision child and a rigid body on the root so the
        # rendered mug and physical body move together.  The prior code wrapped
        # this Xform in UsdGeom.Cube without changing its type, leaving no real
        # collision geometry.
        mug_height = HOUSEHOLD_MUG_HEIGHT_M
        mug_radius = HOUSEHOLD_MUG_OUTER_RADIUS_M
        if rgbd_localization is not None and reuse_existing_composite:
            # The live household mug Xform is already authored at its bottom
            # support contact.  Preserve that origin when physics is enabled.
            target_root_position = target_position
            target_position_semantics = (
                "existing_household_mug_bottom_contact_origin"
            )
        else:
            target_root_position = target_position - np.asarray(
                [0.0, 0.0, mug_height * 0.5], dtype=np.float64
            )
            target_position_semantics = "center_to_bottom_origin_conversion"
        target_xform = UsdGeom.Xformable(target_prim)
        target_xform.ClearXformOpOrder()
        target_xform.AddTranslateOp().Set(Gf.Vec3d(*target_root_position))
        target_collision_prim = UsdGeom.Cylinder.Define(
            stage, f"{target_prim.GetPath()}/PhysicsCollision"
        )
        target_collision_prim.CreateAxisAttr("Z")
        target_collision_prim.CreateRadiusAttr(mug_radius)
        target_collision_prim.CreateHeightAttr(mug_height)
        target_collision_prim.CreateDisplayOpacityAttr([0.0])
        collision_xform = UsdGeom.Xformable(
            target_collision_prim.GetPrim()
        )
        collision_xform.AddTranslateOp().Set(
            Gf.Vec3d(0.0, 0.0, mug_height * 0.5)
        )
        target_collision = target_collision_prim.GetPrim()
        UsdPhysics.CollisionAPI.Apply(target_collision)
    else:
        target_extents = np.asarray(
            [0.045, 0.045, 0.12], dtype=np.float64
        )
        set_cube(UsdGeom.Cube(target_prim), target_position, target_extents)
        target_collision = target_prim
        UsdPhysics.CollisionAPI.Apply(target_collision)
    if not existing_dynamic_assembly:
        # Keep the target as a gravity-enabled dynamic body. The fingers must
        # carry it through frictional contact; there is no grasp attachment.
        target_body = UsdPhysics.RigidBodyAPI.Apply(target_prim)
        target_body.CreateRigidBodyEnabledAttr().Set(True)
        target_body.CreateKinematicEnabledAttr().Set(False)
        target_mass_kg = (
            HOUSEHOLD_MUG_MASS_KG if household_mug_physics_proxy else 0.04
        )
        UsdPhysics.MassAPI.Apply(target_prim).CreateMassAttr().Set(
            target_mass_kg
        )
        PhysxSchema.PhysxRigidBodyAPI.Apply(
            target_prim
        ).CreateDisableGravityAttr().Set(False)
    PhysxSchema.PhysxContactReportAPI.Apply(
        target_prim
    ).CreateThresholdAttr().Set(0.0)

    rg6_base_prim = next(
        (
            prim
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(robot_path)
            and prim.GetName() == "rg6_onrobot_rg6_base_link"
        ),
        None,
    )
    if rg6_base_prim is None:
        raise RuntimeError("Persistent composite RG6 base frame is missing")
    chain = str(rg6_base_prim.GetPath())
    left_link = f"{chain}/rg6_left_outer_knuckle/rg6_left_inner_finger"
    right_link = f"{chain}/rg6_right_outer_knuckle/rg6_right_inner_finger"
    for path in (left_link, right_link):
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Persistent RG6 contact link is missing: {path}")
        PhysxSchema.PhysxContactReportAPI.Apply(
            prim
        ).CreateThresholdAttr().Set(0.0)

    # The imported RG6 fingertip collision is nested inside an instanceable
    # reference. Stage.Traverse therefore sees only the instance root, while
    # PhysX reports contact on ``.../inner_finger_1/inner_finger_1``. Make only
    # these two in-memory fingertip instances editable so the physics material
    # can be bound to the actual collision meshes. Source USD files remain
    # untouched.
    fingertip_instance_roots = {
        "left": f"{left_link}/inner_finger_1",
        "right": f"{right_link}/inner_finger_1",
    }
    for side, path in fingertip_instance_roots.items():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(
                f"RG6 {side} fingertip instance root is missing: {path}"
            )
        prim.SetInstanceable(False)

    material = UsdShade.Material.Define(
        stage, "/World/PhysicsMaterials/PersistentCompositeGrip"
    )
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(
        grip_static_friction
    )
    material_api.CreateDynamicFrictionAttr().Set(
        grip_dynamic_friction
    )
    material_api.CreateRestitutionAttr().Set(0.0)
    physx_material_api = PhysxSchema.PhysxMaterialAPI.Apply(
        material.GetPrim()
    )
    if grip_compliant_contact_stiffness_n_m > 0.0:
        physx_material_api.CreateCompliantContactAccelerationSpringAttr().Set(
            False
        )
        physx_material_api.CreateCompliantContactStiffnessAttr().Set(
            grip_compliant_contact_stiffness_n_m
        )
        physx_material_api.CreateCompliantContactDampingAttr().Set(
            grip_compliant_contact_damping_n_s_m
        )
    target_collision_path = str(target_collision.GetPath())
    target_collision_paths = {
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(manipulation_target_path)
        and prim.HasAPI(UsdPhysics.CollisionAPI)
    }
    if target_collision_path not in target_collision_paths:
        target_collision_paths.add(target_collision_path)
    left_collision_paths = {
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(f"{left_link}/")
        and prim.HasAPI(UsdPhysics.CollisionAPI)
    }
    right_collision_paths = {
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(f"{right_link}/")
        and prim.HasAPI(UsdPhysics.CollisionAPI)
    }
    if not left_collision_paths or not right_collision_paths:
        raise RuntimeError(
            "RG6 fingertip collision descendants are missing: "
            f"left={sorted(left_collision_paths)}, "
            f"right={sorted(right_collision_paths)}"
        )
    grip_material_collision_paths = {
        *target_collision_paths,
        *left_collision_paths,
        *right_collision_paths,
    }
    for path in grip_material_collision_paths:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            add_physics_material_to_prim(stage, prim, material.GetPath())

    target_initial_world_position = world_position(target_prim)
    secondary_placement_support_paths = (
        ("/World/WorkBench/Top",)
        if placement_support_path == "/World/WorkMat"
        else ()
    )

    contacts = {
        "left": 0,
        "right": 0,
        "latest_force_n": {"left": 0.0, "right": 0.0},
        "last_reported_force_n": {"left": 0.0, "right": 0.0},
        "maximum_force_n": {"left": 0.0, "right": 0.0},
        "maximum_penetration_m": {"left": 0.0, "right": 0.0},
        "last_impulse_vector_ns": {
            "left": [0.0, 0.0, 0.0],
            "right": [0.0, 0.0, 0.0],
        },
        "last_contact_position_world_m": {
            "left": None,
            "right": None,
        },
        "last_contact_normal_world": {
            "left": None,
            "right": None,
        },
        "force_sample_count": {"left": 0, "right": 0},
        "last_force_step": {"left": None, "right": None},
        "active_contact_pair_count": {"left": 0, "right": 0},
        "contact_lost_count": {"left": 0, "right": 0},
        "raw_pairs": [],
        "unexpected_environment_pairs": [],
        "unexpected_target_environment_pairs": [],
        "placement_support_contact_events": 0,
        "placement_support_contact_pairs": [],
        "secondary_placement_support_contact_events": 0,
        "secondary_placement_support_contact_pairs": [],
        "placement_support_maximum_penetration_m": 0.0,
        "events_by_phase": {},
    }
    active_contact_pairs: dict[str, set[tuple[str, str]]] = {
        "left": set(),
        "right": set(),
    }
    force_sample_history: dict[str, list[dict]] = {
        "left": [],
        "right": [],
    }
    active_placement_support_pairs: set[tuple[str, str]] = set()
    target_path = target_collision_path
    contact_context = {"phase": "initialization", "step": 0}

    def on_contact(headers, contact_data) -> None:
        for header in headers:
            pair = {
                str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
            }
            sample = sorted(pair)
            pair_key = tuple(sample)
            target_paths_in_pair = pair.intersection(target_collision_paths)
            placement_support_in_pair = (
                placement_support_path is not None
                and any(
                    path == placement_support_path
                    or path.startswith(f"{placement_support_path}/")
                    for path in pair
                )
            )
            secondary_placement_support_in_pair = any(
                path == support_path
                or path.startswith(f"{support_path}/")
                for support_path in secondary_placement_support_paths
                for path in pair
            )
            target_side = None
            if target_path in pair:
                other = next(
                    (path for path in pair if path != target_path), ""
                )
                if "/rg6_left_" in other:
                    target_side = "left"
                elif "/rg6_right_" in other:
                    target_side = "right"
            if header.type == ContactEventType.CONTACT_LOST:
                if target_side is not None:
                    active_contact_pairs[target_side].discard(pair_key)
                    contacts["active_contact_pair_count"][target_side] = len(
                        active_contact_pairs[target_side]
                    )
                    contacts["contact_lost_count"][target_side] += 1
                if target_paths_in_pair and placement_support_in_pair:
                    active_placement_support_pairs.discard(pair_key)
                continue
            if header.type not in (
                ContactEventType.CONTACT_FOUND,
                ContactEventType.CONTACT_PERSIST,
            ):
                continue
            if len(contacts["raw_pairs"]) < 20 and sample not in contacts["raw_pairs"]:
                contacts["raw_pairs"].append(sample)
            robot_paths = [path for path in pair if path.startswith(robot_path)]
            if target_paths_in_pair and placement_support_in_pair:
                active_placement_support_pairs.add(pair_key)
                contacts["placement_support_contact_events"] += 1
                if (
                    len(contacts["placement_support_contact_pairs"]) < 8
                    and sample
                    not in contacts["placement_support_contact_pairs"]
                ):
                    contacts["placement_support_contact_pairs"].append(sample)
                for index in range(
                    header.contact_data_offset,
                    header.contact_data_offset + header.num_contact_data,
                ):
                    contacts[
                        "placement_support_maximum_penetration_m"
                    ] = max(
                        contacts[
                            "placement_support_maximum_penetration_m"
                        ],
                        max(0.0, -float(contact_data[index].separation)),
                    )
            if target_paths_in_pair and secondary_placement_support_in_pair:
                contacts["secondary_placement_support_contact_events"] += 1
                if (
                    len(contacts["secondary_placement_support_contact_pairs"])
                    < 8
                    and sample
                    not in contacts[
                        "secondary_placement_support_contact_pairs"
                    ]
                ):
                    contacts[
                        "secondary_placement_support_contact_pairs"
                    ].append(sample)
                for index in range(
                    header.contact_data_offset,
                    header.contact_data_offset + header.num_contact_data,
                ):
                    contacts[
                        "placement_support_maximum_penetration_m"
                    ] = max(
                        contacts[
                            "placement_support_maximum_penetration_m"
                        ],
                        max(0.0, -float(contact_data[index].separation)),
                    )
            if target_paths_in_pair and not robot_paths:
                other_paths = pair.difference(target_collision_paths)
                target_clear_of_support = (
                    world_position(target_prim)[2]
                    > target_initial_world_position[2] + 0.020
                )
                monitored_motion_phase = contact_context["phase"].startswith(
                    (
                        "persistent_contact_lift",
                        "persistent_lift_verification",
                        "persistent_contact_transfer",
                        "persistent_transfer_verification",
                        "persistent_supported_placement",
                        "persistent_supported_release",
                        "persistent_post_release_retreat",
                    )
                )
                supported_placement_phase = contact_context["phase"].startswith(
                    (
                        "persistent_supported_placement",
                        "persistent_supported_release",
                        "persistent_post_release_retreat",
                    )
                )
                intentional_support_contact = (
                    (
                        placement_support_in_pair
                        or secondary_placement_support_in_pair
                    )
                    and supported_placement_phase
                )
                if (
                    other_paths
                    and (
                        target_clear_of_support
                        or supported_placement_phase
                    )
                    and monitored_motion_phase
                    and not intentional_support_contact
                    and sample
                    not in contacts["unexpected_target_environment_pairs"]
                ):
                    contacts["unexpected_target_environment_pairs"].append(
                        sample
                    )
            if not robot_paths:
                continue
            phase = contact_context["phase"]
            phase_record = contacts["events_by_phase"].setdefault(
                phase, {"event_count": 0, "pairs": [], "first_step": None}
            )
            phase_record["event_count"] += 1
            if phase_record["first_step"] is None:
                phase_record["first_step"] = contact_context["step"]
            if len(phase_record["pairs"]) < 12 and sample not in phase_record["pairs"]:
                phase_record["pairs"].append(sample)

            if target_path in pair:
                other = next((path for path in pair if path != target_path), "")
                side = target_side
                if "/rg6_left_" in other:
                    contacts["left"] += 1
                elif "/rg6_right_" in other:
                    contacts["right"] += 1
                elif other.startswith(robot_path):
                    if sample not in contacts["unexpected_environment_pairs"]:
                        contacts["unexpected_environment_pairs"].append(sample)
                if side is not None:
                    active_contact_pairs[side].add(pair_key)
                    contacts["active_contact_pair_count"][side] = len(
                        active_contact_pairs[side]
                    )
                    total_impulse = np.zeros(3, dtype=np.float64)
                    maximum_penetration = 0.0
                    strongest_impulse_norm = -1.0
                    strongest_position = None
                    strongest_normal = None
                    start = header.contact_data_offset
                    end = start + header.num_contact_data
                    for index in range(start, end):
                        impulse = np.asarray(
                            contact_data[index].impulse,
                            dtype=np.float64,
                        )
                        total_impulse += impulse
                        impulse_norm = float(np.linalg.norm(impulse))
                        if impulse_norm > strongest_impulse_norm:
                            strongest_impulse_norm = impulse_norm
                            strongest_position = np.asarray(
                                contact_data[index].position,
                                dtype=np.float64,
                            )
                            strongest_normal = np.asarray(
                                contact_data[index].normal,
                                dtype=np.float64,
                            )
                        maximum_penetration = max(
                            maximum_penetration,
                            max(
                                0.0,
                                -float(contact_data[index].separation),
                            ),
                        )
                    force_n = float(
                        np.linalg.norm(total_impulse)
                        / PHYSICS_DT_SECONDS
                    )
                    contacts["latest_force_n"][side] = force_n
                    contacts["last_impulse_vector_ns"][side] = (
                        total_impulse.tolist()
                    )
                    contacts["last_contact_position_world_m"][side] = (
                        strongest_position.tolist()
                        if strongest_position is not None
                        else None
                    )
                    contacts["last_contact_normal_world"][side] = (
                        strongest_normal.tolist()
                        if strongest_normal is not None
                        else None
                    )
                    contacts["last_reported_force_n"][side] = force_n
                    contacts["maximum_force_n"][side] = max(
                        contacts["maximum_force_n"][side], force_n
                    )
                    contacts["maximum_penetration_m"][side] = max(
                        contacts["maximum_penetration_m"][side],
                        maximum_penetration,
                    )
                    contacts["force_sample_count"][side] += 1
                    contacts["last_force_step"][side] = contact_context[
                        "step"
                    ]
                    force_sample_history[side].append(
                        {
                            "step": int(contact_context["step"]),
                            "force_n": force_n,
                        }
                    )
                    if len(force_sample_history[side]) > 4096:
                        del force_sample_history[side][:-4096]
                continue
            if not all(path.startswith(robot_path) for path in pair):
                if sample not in contacts["unexpected_environment_pairs"]:
                    contacts["unexpected_environment_pairs"].append(sample)

    # Contact reports are the manipulation contract, not just diagnostics.
    # Active pairs establish bilateral grasping, while impulse and separation
    # samples bound force and interpenetration before motion may continue.
    contact_subscription = (
        get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
    )

    home = np.asarray(
        [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
        dtype=np.float64,
    )
    pregrasp = np.asarray(descent_plan[0]["joints_rad"], dtype=np.float64)
    # In a stage that has already run the observation articulation, Isaac Sim
    # 6.0 can stall a newly inserted wrist_3 drive during the near-pi move from
    # zero. Wrist_3 only rolls the airborne tool at this safe configuration, so
    # author it at the already validated pregrasp roll before the first physics
    # frame. All Cartesian approach joints still execute under collision and
    # tracking-error monitoring.
    if reuse_existing_composite:
        home = np.asarray(initial_arm_positions_rad, dtype=np.float64)
    else:
        home[5] = pregrasp[5]
    open_master = -0.20
    # The thin cover handle is encountered close to the RG6 linkage's closed
    # configuration.  A 0.45 rad target stops inside the compliant material's
    # contact envelope without compressing it (run024/run025: bilateral
    # contact events but zero impulse). Command a near-closed, still
    # joint-limit-safe target for explicit handle contacts; the independent
    # force and penetration gates remain authoritative.
    close_master = (
        RG6_HANDLE_CLOSE_MASTER_RAD
        if contact_target_path is not None
        else 0.45
    )
    initial_by_name = {
        **dict(zip(arm_names, home)),
        "rg6_finger_joint": open_master,
        "rg6_left_inner_knuckle_joint": -open_master,
        "rg6_left_inner_finger_joint": open_master,
        "rg6_right_outer_knuckle_joint": -open_master,
        "rg6_right_inner_knuckle_joint": -open_master,
        "rg6_right_inner_finger_joint": open_master,
    }
    for name, position_rad in initial_by_name.items():
        joint = stage.GetPrimAtPath(f"{robot_path}/Physics/{name}")
        state = PhysxSchema.JointStateAPI.Get(joint, "angular")
        if not state:
            state = PhysxSchema.JointStateAPI.Apply(joint, "angular")
        state.CreatePositionAttr().Set(math.degrees(float(position_rad)))
        state.CreateVelocityAttr().Set(0.0)
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if drive:
            drive.CreateTargetPositionAttr().Set(
                math.degrees(float(position_rad))
            )
            drive.CreateTargetVelocityAttr().Set(0.0)

    # The live observation server has already created the SimulationManager
    # context. Re-running setup_simulation here leaves newly inserted
    # articulation drives in an inconsistent state in Isaac Sim 6.0. Reuse the
    # existing single-stage context. Restoring Newton mimic after a cover task
    # requires one more PhysX rebuild. Preserve the released cover's current
    # local pose in the session layer before stop/play and verify it afterward.
    preserved_cover_pose = None
    if restored_mimic_requires_physics_rebuild:
        preserved_cover = stage.GetPrimAtPath(
            "/World/OpenContainer/CalibrationCover"
        )
        if not preserved_cover.IsValid():
            raise RuntimeError(
                "Cannot rebuild restored RG6 mimic without the released cover"
            )
        preserved_cover_position = world_position(preserved_cover)
        preserved_cover_rotation = world_rotation(preserved_cover)
        preserved_cover_xform = UsdGeom.Xformable(preserved_cover)
        preserved_cover_local = preserved_cover_xform.GetLocalTransformation()
        preserved_cover_pose = {
            "position": preserved_cover_position,
            "rotation": preserved_cover_rotation,
        }
        app_utils.stop()
        for _ in range(3):
            simulation_app.update()
        # stop() restores the rigid body to its previously authored pose.  Set
        # the captured released pose only after that restore, before play()
        # creates the new PhysX objects for the Newton mimic constraints.
        preserved_cover_xform = UsdGeom.Xformable(preserved_cover)
        preserved_cover_xform.ClearXformOpOrder()
        preserved_cover_xform.MakeMatrixXform().Set(preserved_cover_local)
    robot = Articulation(robot_path)
    simulation_app.update()
    if rebuild_persistent_physics or restored_mimic_requires_physics_rebuild:
        app_utils.play()
    simulation_app.update()
    if preserved_cover_pose is not None:
        restored_cover = stage.GetPrimAtPath(
            "/World/OpenContainer/CalibrationCover"
        )
        restored_cover_position_error = float(
            np.linalg.norm(
                world_position(restored_cover)
                - preserved_cover_pose["position"]
            )
        )
        restored_cover_rotation_error = rotation_angle_rad(
            world_rotation(restored_cover)
            @ preserved_cover_pose["rotation"].T
        )
        if (
            restored_cover_position_error > 0.002
            or restored_cover_rotation_error > math.radians(2.0)
        ):
            raise RuntimeError(
                "Released cover pose changed during RG6 mimic rebuild: "
                f"position_error_m={restored_cover_position_error:.6f}, "
                f"rotation_error_rad={restored_cover_rotation_error:.6f}"
            )
    prepared_robot_after_rebuild = stage.GetPrimAtPath(robot_path)
    if not prepared_robot_after_rebuild.IsValid():
        raise RuntimeError(
            "Persistent composite robot disappeared during PhysX rebuild"
        )
    prepared_robot_after_rebuild.SetCustomDataByKey(
        "persistent_composite_executor_prepared", True
    )
    dof_names = list(robot.dof_names)
    if len(dof_names) != 12 or set(dof_names) != set(arm_names + rg6_names):
        raise RuntimeError(f"Unexpected persistent composite DOFs: {dof_names}")
    arm_indices = [dof_names.index(name) for name in arm_names]
    master_index = dof_names.index(rg6_master_name)
    rg6_indices = [dof_names.index(name) for name in rg6_names]

    def full_configuration(arm: np.ndarray, master: float) -> np.ndarray:
        by_name = {
            **dict(zip(arm_names, arm)),
            rg6_master_name: master,
            **{
                name: ratio * master
                for name, ratio in rg6_follower_ratios.items()
            },
        }
        return np.asarray([by_name[name] for name in dof_names], dtype=np.float32)

    def set_targets(arm: np.ndarray, master: float) -> None:
        if rg6_coupling_mode == "coordinated_drives":
            measured_master = measured_by_name()[rg6_master_name]
            follower_master_target = measured_master + (
                coordinated_follower_request_blend
                * (master - measured_master)
            )
            rg6_targets = {
                rg6_master_name: master,
                **{
                    # Blend actual and requested master states. Pure request
                    # tracking raced ahead under load; pure measured tracking
                    # lost bilateral squeeze. This remains a development-only
                    # coupling proxy pending the real RG6 linkage model.
                    name: ratio * follower_master_target
                    for name, ratio in rg6_follower_ratios.items()
                },
            }
            robot.set_dof_position_targets(
                np.asarray(
                    [*arm, *[rg6_targets[name] for name in rg6_names]],
                    dtype=np.float32,
                ),
                dof_indices=[*arm_indices, *rg6_indices],
            )
        else:
            robot.set_dof_position_targets(
                np.asarray([*arm, master], dtype=np.float32),
                dof_indices=[*arm_indices, master_index],
            )

    def measured() -> np.ndarray:
        values = robot.get_dof_positions().numpy()
        return values[0] if values.ndim > 1 else values

    def measured_by_name() -> dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(dof_names, measured())
        }

    def arm_error(target: np.ndarray) -> float:
        by_name = measured_by_name()
        return float(
            max(
                abs(by_name[name] - desired)
                for name, desired in zip(arm_names, target)
            )
        )

    def mimic_error() -> float:
        by_name = measured_by_name()
        master = by_name[rg6_master_name]
        return float(
            max(
                abs(by_name[name] - ratio * master)
                for name, ratio in rg6_follower_ratios.items()
            )
        )

    def mimic_error_by_joint() -> dict[str, float]:
        by_name = measured_by_name()
        master = by_name[rg6_master_name]
        return {
            name: float(by_name[name] - ratio * master)
            for name, ratio in rg6_follower_ratios.items()
        }

    def normalized_world_matrix(prim) -> np.ndarray:
        matrix = omni.usd.get_world_transform_matrix(prim)
        value = np.asarray(
            [[float(matrix[row][column]) for column in range(4)] for row in range(4)],
            dtype=np.float64,
        )
        rotation = value[:3, :3]
        left, _, right = np.linalg.svd(rotation)
        value[:3, :3] = left @ right
        return value

    def target_relative_to_gripper() -> np.ndarray:
        target_world = normalized_world_matrix(target_prim)
        gripper_world = normalized_world_matrix(rg6_base_prim)
        return target_world @ np.linalg.inv(gripper_world)

    def gripper_world_position() -> np.ndarray:
        return world_position(rg6_base_prim)

    stability = {
        "reference": None,
        "previous": None,
        "maximum_relative_translation_m": 0.0,
        "maximum_relative_rotation_rad": 0.0,
        "maximum_relative_angular_speed_rad_s": 0.0,
        "latest_step_relative_translation_m": 0.0,
        "latest_step_relative_rotation_rad": 0.0,
        "sample_count": 0,
    }
    contact_continuity = {
        "current_gap_steps": 0,
        "maximum_gap_steps": 0,
    }
    grip_force_controller = {
        "enabled": False,
        "target_frozen_after_pre_lift_gate": False,
        "target_master_rad": None,
        "initial_torque_limit_nm": float(rg6_master_max_torque_nm),
        "current_torque_limit_nm": float(rg6_master_max_torque_nm),
        "maximum_drive_effort_limit_nm": float(
            coordinated_total_drive_effort_limit_nm
            if rg6_coupling_mode == "coordinated_drives"
            else force_controller_max_torque_nm
        ),
        "adjustment_count": 0,
        "contact_recovery_adjustment_count": 0,
        "slip_response_adjustment_count": 0,
        "history": [],
    }

    def start_target_stability_tracking() -> None:
        relative = target_relative_to_gripper()
        stability["reference"] = relative
        stability["previous"] = relative
        stability["sample_count"] = 1
        stability["target_start_world_m"] = world_position(
            target_prim
        ).tolist()
        stability["gripper_start_world_m"] = gripper_world_position().tolist()

    def sample_target_stability() -> None:
        reference = stability["reference"]
        if reference is None:
            return
        relative = target_relative_to_gripper()
        translation = float(
            np.linalg.norm(relative[3, :3] - reference[3, :3])
        )
        rotation_delta = relative[:3, :3] @ reference[:3, :3].T
        rotation = rotation_angle_rad(rotation_delta)
        previous = stability["previous"]
        step_translation = float(
            np.linalg.norm(relative[3, :3] - previous[3, :3])
        )
        step_rotation = rotation_angle_rad(
            relative[:3, :3] @ previous[:3, :3].T
        )
        angular_speed = step_rotation / PHYSICS_DT_SECONDS
        stability["maximum_relative_translation_m"] = max(
            stability["maximum_relative_translation_m"], translation
        )
        stability["maximum_relative_rotation_rad"] = max(
            stability["maximum_relative_rotation_rad"], rotation
        )
        stability["maximum_relative_angular_speed_rad_s"] = max(
            stability["maximum_relative_angular_speed_rad_s"], angular_speed
        )
        stability["latest_step_relative_translation_m"] = step_translation
        stability["latest_step_relative_rotation_rad"] = step_rotation
        stability["previous"] = relative
        stability["sample_count"] += 1

    def start_grip_force_controller(master_target: float) -> None:
        grip_force_controller["enabled"] = True
        grip_force_controller["target_frozen_after_pre_lift_gate"] = False
        grip_force_controller["target_master_rad"] = float(master_target)
        set_grip_drive_effort(
            grip_force_controller["current_torque_limit_nm"]
        )

    def controlled_master_target(requested_master: float) -> float:
        if not grip_force_controller["enabled"]:
            return float(requested_master)
        return float(grip_force_controller["target_master_rad"])

    def stop_grip_force_controller() -> None:
        grip_force_controller["enabled"] = False
        grip_force_controller["target_master_rad"] = None

    def update_grip_force_controller() -> None:
        if not grip_force_controller["enabled"]:
            return
        bilateral_now = all(
            contacts["active_contact_pair_count"][side] > 0
            for side in ("left", "right")
        )
        controller_force_window = summarize_terminal_force_window(
            force_sample_history,
            current_step=int(contact_context["step"]),
            minimum_force_per_finger_n=minimum_grip_force_per_finger_n,
            window_steps=GRIP_CONTROLLER_FORCE_WINDOW_STEPS,
            minimum_qualifying_steps=1,
        )
        recent_force = bool(
            controller_force_window["samples_available_both_sides"]
        )
        minimum_force_ready = bool(
            controller_force_window["sufficient_both_sides"]
        )
        force_values = {
            side: controller_force_window["by_side"][side][
                "maximum_force_n"
            ]
            for side in ("left", "right")
        }
        combined_force_ready = (
            minimum_combined_grip_force_n is None
            or sum(force_values.values()) >= minimum_combined_grip_force_n
        )
        relative_slip_m = float(
            stability["latest_step_relative_translation_m"]
        )
        slip_response = relative_slip_m > GRIP_CONTROLLER_STEP_SLIP_M
        force_response = not (
            bilateral_now
            and recent_force
            and minimum_force_ready
            and combined_force_ready
        )
        if not force_response and not slip_response:
            return
        torque_increment = 0.0
        target_increment = 0.0
        if force_response:
            torque_increment = 0.05 if bilateral_now else 0.15
            target_increment = (
                GRIP_FORCE_TARGET_INCREMENT_RAD
                if bilateral_now
                else 2.0 * GRIP_FORCE_TARGET_INCREMENT_RAD
            )
        if slip_response:
            torque_increment = max(torque_increment, 0.10)
            target_increment = max(
                target_increment,
                1.5 * GRIP_FORCE_TARGET_INCREMENT_RAD,
            )
            grip_force_controller["slip_response_adjustment_count"] += 1
        if not bilateral_now:
            grip_force_controller[
                "contact_recovery_adjustment_count"
            ] += 1
        old_torque = float(grip_force_controller["current_torque_limit_nm"])
        old_target = float(grip_force_controller["target_master_rad"])
        new_torque = min(
            float(grip_force_controller["maximum_drive_effort_limit_nm"]),
            old_torque + torque_increment,
        )
        if grip_force_controller["target_frozen_after_pre_lift_gate"]:
            # Once bilateral force and penetration pass the pre-lift safety
            # gate, changing the RG6 linkage geometry can walk a tall object
            # through the fingertips.  Keep the verified closure pose fixed
            # during transport; contact loss still fails closed below.
            new_target = old_target
        else:
            new_target = min(close_master, old_target + target_increment)
        grip_force_controller["current_torque_limit_nm"] = new_torque
        grip_force_controller["target_master_rad"] = new_target
        grip_force_controller["adjustment_count"] += 1
        set_grip_drive_effort(new_torque)
        history = grip_force_controller["history"]
        if len(history) < 240:
            history.append(
                {
                    "step": int(contact_context["step"]),
                    "phase": contact_context["phase"],
                    "bilateral": bilateral_now,
                    "recent_force": recent_force,
                    "force_n": dict(force_values),
                    "relative_slip_m": relative_slip_m,
                    "master_target_rad": new_target,
                    "torque_limit_nm": new_torque,
                }
            )

    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22
        )
    except OSError:
        font = ImageFont.load_default()
    frames: list[Path] = []
    debug_capture_seconds = 0.0
    video_encoding_seconds = 0.0

    if physics_only_steps:
        from isaacsim.core.simulation_manager import SimulationManager

        def advance_simulation() -> None:
            SimulationManager.step(
                steps=1,
                update_fabric=SimulationManager.is_fabric_enabled(),
            )
    else:
        def advance_simulation() -> None:
            simulation_app.update()
    trajectory_records: list[dict] = []

    def save_failure_diagnostics(label: str, reason: str) -> None:
        left_collision_positions = np.asarray(
            [
                world_position(stage.GetPrimAtPath(path))
                for path in sorted(left_collision_paths)
            ],
            dtype=np.float64,
        )
        right_collision_positions = np.asarray(
            [
                world_position(stage.GetPrimAtPath(path))
                for path in sorted(right_collision_paths)
            ],
            dtype=np.float64,
        )
        left_collision_position = left_collision_positions.mean(axis=0)
        right_collision_position = right_collision_positions.mean(axis=0)
        diagnostics = {
            "schema_version": "persistent-composite-grasp-failure-v1",
            "status": "failed",
            "phase": label,
            "reason": reason,
            "contact_events": contacts,
            "contact_continuity": contact_continuity,
            "target_gripper_relative_stability": {
                key: value
                for key, value in stability.items()
                if key not in {"reference", "previous"}
            },
            "micro_lift_world_displacement_m": {
                "target_xyz": (
                    world_position(target_prim)
                    - np.asarray(
                        stability.get(
                            "target_start_world_m",
                            world_position(target_prim).tolist(),
                        ),
                        dtype=np.float64,
                    )
                ).tolist(),
                "gripper_xyz": (
                    gripper_world_position()
                    - np.asarray(
                        stability.get(
                            "gripper_start_world_m",
                            gripper_world_position().tolist(),
                        ),
                        dtype=np.float64,
                    )
                ).tolist(),
            },
            "lift_cartesian_plan": {
                "micro_waypoints": plan.get(
                    "micro_lift_cartesian_waypoints", []
                ),
                "full_lift_waypoints": plan.get(
                    "full_lift_cartesian_waypoints", []
                ),
            },
            "grip_force_controller": grip_force_controller,
            "grip_compliant_contact": {
                "enabled": grip_compliant_contact_stiffness_n_m > 0.0,
                "stiffness_n_m": grip_compliant_contact_stiffness_n_m,
                "damping_n_s_m": grip_compliant_contact_damping_n_s_m,
                "acceleration_spring": False,
            },
            "measured_dofs_rad": measured().tolist(),
            "measured_dofs_by_name_rad": measured_by_name(),
            "rg6_coupling_error_by_joint_rad": mimic_error_by_joint(),
            "trajectory_records": trajectory_records,
            "target_position_world_m": world_position(target_prim).tolist(),
            "grasp_geometry_world_m": {
                "contact_target": world_position(target_collision).tolist(),
                "left_finger_collision": left_collision_position.tolist(),
                "right_finger_collision": right_collision_position.tolist(),
                "finger_midpoint": (
                    0.5
                    * (left_collision_position + right_collision_position)
                ).tolist(),
                "finger_midpoint_to_contact_target": (
                    0.5
                    * (left_collision_position + right_collision_position)
                    - world_position(target_collision)
                ).tolist(),
                "left_collision_paths": sorted(left_collision_paths),
                "right_collision_paths": sorted(right_collision_paths),
            },
            "grasp_height_offset_m": grasp_height_offset_m,
            "rg6_coupling_mode": rg6_coupling_mode,
            "rg6_follower_target_basis": (
                "quarter_measured_three_quarter_requested_master_each_physics_step"
                if rg6_coupling_mode == "coordinated_drives"
                else "newton_mimic_api"
            ),
            "rg6_follower_drive_removal_in_memory": follower_drive_removal,
            "rg6_mimic_api_removal_in_memory": mimic_api_removal,
            "rg6_coupling_error_rad": mimic_error(),
            "qwen_loaded": False,
            "valid_for_final_evaluation": False,
        }
        (output_root / "failure_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2) + "\n",
            encoding="utf-8",
        )

    def enforce_bilateral_contact(label: str) -> None:
        bilateral_now = all(
            contacts["active_contact_pair_count"][side] > 0
            for side in ("left", "right")
        )
        if bilateral_now:
            contact_continuity["current_gap_steps"] = 0
            return
        contact_continuity["current_gap_steps"] += 1
        contact_continuity["maximum_gap_steps"] = max(
            contact_continuity["maximum_gap_steps"],
            contact_continuity["current_gap_steps"],
        )
        if (
            contact_continuity["current_gap_steps"]
            > MAX_CONSECUTIVE_CONTACT_GAP_STEPS
        ):
            save_failure_diagnostics(
                label, "bilateral_contact_gap_exceeded"
            )
            raise RuntimeError(
                "Bilateral contact gap exceeded "
                f"{MAX_CONSECUTIVE_CONTACT_GAP_STEPS} steps during {label}"
            )

    def capture(label: str) -> None:
        nonlocal debug_capture_seconds
        if not record_debug_video:
            return
        capture_started = time.perf_counter()
        rep.orchestrator.step(rt_subframes=2)
        rgba = np.asarray(overview_rgb_annotator.get_data()).copy()
        image = Image.fromarray(rgba[:, :, :3].astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((15, 15, 900, 92), radius=7, fill=(0, 0, 0))
        draw.text((25, 22), label, fill=(255, 255, 255), font=font)
        draw.text(
            (25, 56),
            (
                f"contact L/R={contacts['left']}/{contacts['right']}  "
                f"force={contacts['last_reported_force_n']['left']:.1f}/"
                f"{contacts['last_reported_force_n']['right']:.1f} N  "
                f"target z={world_position(target_prim)[2]:.3f} m"
            ),
            fill=(200, 225, 240),
            font=font,
        )
        path = frame_root / f"frame_{len(frames):04d}.png"
        image.save(path)
        frames.append(path)
        debug_capture_seconds += time.perf_counter() - capture_started

    def check_state(label: str) -> None:
        if not np.all(np.isfinite(measured())):
            raise RuntimeError(f"Non-finite persistent articulation during {label}")
        current_mimic_error = mimic_error()
        maximum_coupling_error_rad = (
            0.15 if rg6_coupling_mode == "coordinated_drives" else 0.05
        )
        if current_mimic_error > maximum_coupling_error_rad:
            save_failure_diagnostics(
                label,
                f"rg6_coupling_error_rad:{current_mimic_error:.9f}",
            )
            raise RuntimeError(
                f"RG6 coupling error {current_mimic_error:.6f} rad during "
                f"{label}"
            )

    def reset_latest_contact_forces() -> None:
        contacts["latest_force_n"]["left"] = 0.0
        contacts["latest_force_n"]["right"] = 0.0

    def grip_contact_limits_exceeded() -> bool:
        return (
            any(
                force > MAX_ALLOWED_CONTACT_FORCE_N
                for force in contacts["maximum_force_n"].values()
            )
            or any(
                depth > MAX_ALLOWED_PENETRATION_M
                for depth in contacts["maximum_penetration_m"].values()
            )
        )

    def hold(
        arm,
        master,
        steps,
        label,
        *,
        require_bilateral_contact=False,
    ) -> None:
        contact_context["phase"] = label
        for step in range(steps):
            set_targets(arm, controlled_master_target(master))
            reset_latest_contact_forces()
            advance_simulation()
            contact_context["step"] += 1
            sample_target_stability()
            if require_bilateral_contact:
                update_grip_force_controller()
                enforce_bilateral_contact(label)
            if contacts["unexpected_target_environment_pairs"]:
                save_failure_diagnostics(
                    label, "target_environment_collision_after_lift"
                )
                raise RuntimeError(
                    "Manipulation target contacted the environment after "
                    f"lift: {contacts['unexpected_target_environment_pairs']}"
                )
            if step % 10 == 0:
                check_state(label)
                if grip_contact_limits_exceeded():
                    save_failure_diagnostics(
                        label, "grip_contact_safety_limit_exceeded"
                    )
                    raise RuntimeError(
                        f"Grip contact safety limit exceeded during {label}"
                    )
            if step % 12 == 0:
                capture(label)
        trajectory_records.append(
            {
                "phase": label,
                "kind": "hold",
                "steps": steps,
                "final_arm_error_rad": arm_error(arm),
                "unexpected_contact_count": len(
                    contacts["unexpected_environment_pairs"]
                ),
            }
        )

    def transition(
        start_arm,
        end_arm,
        start_master,
        end_master,
        steps,
        label,
        *,
        settle_steps=60,
        require_bilateral_contact=False,
        max_relative_translation_m=None,
    ) -> None:
        contact_context["phase"] = label
        unexpected_before = len(contacts["unexpected_environment_pairs"])
        maximum_error = 0.0
        for step in range(1, steps + 1):
            alpha = quintic_time_scaling(step / steps)
            arm = start_arm + alpha * (end_arm - start_arm)
            master = start_master + alpha * (end_master - start_master)
            set_targets(arm, controlled_master_target(master))
            reset_latest_contact_forces()
            advance_simulation()
            contact_context["step"] += 1
            sample_target_stability()
            if require_bilateral_contact:
                update_grip_force_controller()
                enforce_bilateral_contact(label)
            if (
                max_relative_translation_m is not None
                and stability["maximum_relative_translation_m"]
                > max_relative_translation_m
            ):
                save_failure_diagnostics(
                    label, "relative_translation_gate_exceeded"
                )
                raise RuntimeError(
                    f"Relative translation exceeded "
                    f"{max_relative_translation_m:.6f} m during {label}"
                )
            if contacts["unexpected_target_environment_pairs"]:
                save_failure_diagnostics(
                    label, "target_environment_collision_after_lift"
                )
                raise RuntimeError(
                    "Manipulation target contacted the environment after "
                    f"lift: {contacts['unexpected_target_environment_pairs']}"
                )
            if step % 10 == 0:
                check_state(label)
                maximum_error = max(maximum_error, arm_error(arm))
                if grip_contact_limits_exceeded():
                    save_failure_diagnostics(
                        label, "grip_contact_safety_limit_exceeded"
                    )
                    raise RuntimeError(
                        f"Grip contact safety limit exceeded during {label}"
                    )
                if len(contacts["unexpected_environment_pairs"]) > unexpected_before:
                    new_pairs = contacts["unexpected_environment_pairs"][
                        unexpected_before:
                    ]
                    save_failure_diagnostics(
                        label,
                        f"unexpected_collision_during_transition:{new_pairs}",
                    )
                    raise RuntimeError(
                        f"Unexpected persistent collision during {label}: "
                        f"{new_pairs}"
                    )
            if step % 12 == 0:
                capture(label)
        for step in range(settle_steps):
            set_targets(
                end_arm, controlled_master_target(end_master)
            )
            reset_latest_contact_forces()
            advance_simulation()
            contact_context["step"] += 1
            sample_target_stability()
            if require_bilateral_contact:
                update_grip_force_controller()
                enforce_bilateral_contact(label)
            if (
                max_relative_translation_m is not None
                and stability["maximum_relative_translation_m"]
                > max_relative_translation_m
            ):
                save_failure_diagnostics(
                    label, "relative_translation_gate_exceeded"
                )
                raise RuntimeError(
                    f"Relative translation exceeded "
                    f"{max_relative_translation_m:.6f} m while settling "
                    f"{label}"
                )
            if contacts["unexpected_target_environment_pairs"]:
                save_failure_diagnostics(
                    label, "target_environment_collision_after_lift"
                )
                raise RuntimeError(
                    "Manipulation target contacted the environment after "
                    f"lift: {contacts['unexpected_target_environment_pairs']}"
                )
            if step % 10 == 0:
                check_state(label)
                maximum_error = max(maximum_error, arm_error(end_arm))
                if grip_contact_limits_exceeded():
                    save_failure_diagnostics(
                        label, "grip_contact_safety_limit_exceeded"
                    )
                    raise RuntimeError(
                        f"Grip contact safety limit exceeded while settling "
                        f"{label}"
                    )
                if len(contacts["unexpected_environment_pairs"]) > unexpected_before:
                    new_pairs = contacts["unexpected_environment_pairs"][
                        unexpected_before:
                    ]
                    save_failure_diagnostics(
                        label,
                        f"unexpected_collision_while_settling:{new_pairs}",
                    )
                    raise RuntimeError(
                        f"Unexpected persistent collision while settling "
                        f"{label}: {new_pairs}"
                    )
            if step % 12 == 0:
                capture(label)
        final_error = arm_error(end_arm)
        trajectory_records.append(
            {
                "phase": label,
                "kind": "transition",
                "steps": steps,
                "settle_steps": settle_steps,
                "maximum_arm_error_rad": maximum_error,
                "final_arm_error_rad": final_error,
                "new_unexpected_contact_count": (
                    len(contacts["unexpected_environment_pairs"])
                    - unexpected_before
                ),
            }
        )
        if final_error > 0.05:
            save_failure_diagnostics(
                label,
                f"final_arm_error_rad:{final_error:.9f}",
            )
            raise RuntimeError(
                f"Persistent arm error {final_error:.6f} rad after {label}; "
                f"measured_dofs={measured().tolist()}, "
                f"target_arm={end_arm.tolist()}"
            )

    def close_until_bilateral_contact(arm: np.ndarray) -> float:
        label = "persistent_rg6_contact_controlled_closure"
        contact_context["phase"] = label
        unexpected_before = len(contacts["unexpected_environment_pairs"])
        maximum_error = 0.0
        actual_master = open_master
        bilateral_step = None
        bilateral_master = None
        force_ready_step = None
        force_ready_master = None
        maximum_combined_force_n = 0.0
        for step in range(1, CONTACT_CONTROLLED_CLOSURE_STEPS + 1):
            alpha = step / CONTACT_CONTROLLED_CLOSURE_STEPS
            scheduled_master = (
                open_master + alpha * (close_master - open_master)
            )
            # Before contact, follow the geometric closing schedule. Once
            # bilateral contact enables the force controller, freeze that
            # schedule and advance only by its small bounded increments. This
            # prevents a sparse contact-report interval from causing a large
            # unobserved squeeze before the next force sample.
            actual_master = (
                controlled_master_target(scheduled_master)
                if grip_force_controller["enabled"]
                else scheduled_master
            )
            set_targets(arm, actual_master)
            reset_latest_contact_forces()
            advance_simulation()
            contact_context["step"] += 1
            maximum_error = max(maximum_error, arm_error(arm))
            if grip_contact_limits_exceeded():
                save_failure_diagnostics(
                    label, "grip_contact_safety_limit_exceeded"
                )
                raise RuntimeError(
                    "Grip contact safety limit exceeded during controlled "
                    "closure"
                )
            if (
                len(contacts["unexpected_environment_pairs"])
                > unexpected_before
            ):
                new_pairs = contacts["unexpected_environment_pairs"][
                    unexpected_before:
                ]
                save_failure_diagnostics(
                    label,
                    f"unexpected_collision_during_closure:{new_pairs}",
                )
                raise RuntimeError(
                    f"Unexpected collision during contact closure: {new_pairs}"
                )
            if step % 48 == 0:
                capture(
                    label
                    if bilateral_step is None
                    else "persistent_rg6_force_ramp"
                )
            bilateral = all(
                contacts["active_contact_pair_count"][side] > 0
                for side in ("left", "right")
            )
            if bilateral and bilateral_step is None:
                bilateral_step = step
                bilateral_master = actual_master
                # A thin handle is contacted close to the RG6 linkage's
                # closed configuration. The provisional initial 2 Nm drive
                # can establish geometric contact without producing a useful
                # normal impulse. Only after verified bilateral contact,
                # enable the bounded force controller so it may gradually
                # increase torque up to the calibration-config ceiling.
                start_grip_force_controller(actual_master)
            if bilateral:
                update_grip_force_controller()
            combined_force_n = sum(contacts["latest_force_n"].values())
            maximum_combined_force_n = max(
                maximum_combined_force_n, combined_force_n
            )
            individual_force_ready = bilateral and all(
                contacts["maximum_force_n"][side]
                >= minimum_grip_force_per_finger_n
                for side in ("left", "right")
            )
            combined_force_ready = (
                minimum_combined_grip_force_n is None
                or maximum_combined_force_n
                >= minimum_combined_grip_force_n
            )
            force_ready = individual_force_ready and combined_force_ready
            if force_ready:
                force_ready_step = step
                force_ready_master = controlled_master_target(actual_master)
                break
        if bilateral_master is None:
            save_failure_diagnostics(
                label, "bilateral_contact_not_reached"
            )
            raise RuntimeError(
                "RG6 reached its closure limit without bilateral contact"
            )
        if force_ready_master is None:
            save_failure_diagnostics(
                label,
                (
                    "minimum_bilateral_force_not_reached:"
                    f"last_reported="
                    f"{contacts['last_reported_force_n']},"
                    f"maximum={contacts['maximum_force_n']},"
                    f"maximum_combined={maximum_combined_force_n}"
                ),
            )
            raise RuntimeError(
                "RG6 reached its closure limit without sufficient bilateral "
                f"force: last_reported="
                f"{contacts['last_reported_force_n']}, "
                f"maximum_combined={maximum_combined_force_n}"
            )
        grasp_master = force_ready_master
        for step in range(60):
            set_targets(arm, grasp_master)
            reset_latest_contact_forces()
            advance_simulation()
            contact_context["step"] += 1
            maximum_error = max(maximum_error, arm_error(arm))
            if grip_contact_limits_exceeded():
                save_failure_diagnostics(
                    label, "grip_contact_safety_limit_exceeded"
                )
                raise RuntimeError(
                    "Grip contact safety limit exceeded while settling "
                    "controlled closure"
                )
            if step % 12 == 0:
                capture("persistent_rg6_contact_settle")

        trajectory_records.append(
            {
                "phase": label,
                "kind": "contact_controlled_closure",
                "maximum_steps": CONTACT_CONTROLLED_CLOSURE_STEPS,
                "bilateral_contact_step": bilateral_step,
                "bilateral_contact_master_rad": bilateral_master,
                "minimum_force_per_finger_n": (
                    minimum_grip_force_per_finger_n
                ),
                "minimum_combined_force_n": minimum_combined_grip_force_n,
                "maximum_combined_force_n": maximum_combined_force_n,
                "force_ready_step": force_ready_step,
                "force_ready_master_rad": force_ready_master,
                "final_master_rad": grasp_master,
                "maximum_arm_error_rad": maximum_error,
                "maximum_force_n": dict(contacts["maximum_force_n"]),
                "maximum_penetration_m": dict(
                    contacts["maximum_penetration_m"]
                ),
            }
        )
        return grasp_master

    robot.set_dof_positions(full_configuration(home, open_master))
    robot.set_dof_velocities(np.zeros(len(dof_names), dtype=np.float32))
    set_targets(home, open_master)
    hold(
        home,
        open_master,
        120,
        (
            "persistent_live_view_hold"
            if reuse_existing_composite
            else "persistent_safe_home"
        ),
    )
    transition(
        home,
        pregrasp,
        open_master,
        open_master,
        max(
            360,
            int(
                math.ceil(
                    1.875
                    * float(np.max(np.abs(pregrasp - home)))
                    / (
                        PREGRASP_MAX_TRAJECTORY_SPEED_RAD_S
                        * PHYSICS_DT_SECONDS
                    )
                )
            ),
        ),
        "persistent_move_to_pregrasp",
        settle_steps=pregrasp_settle_steps,
    )
    hold(pregrasp, open_master, 60, "persistent_verify_pregrasp")
    previous = pregrasp
    for index, waypoint in enumerate(descent_plan[1:], start=1):
        following = np.asarray(waypoint["joints_rad"], dtype=np.float64)
        maximum_joint_delta = float(np.max(np.abs(following - previous)))
        transition(
            previous,
            following,
            open_master,
            open_master,
            max(
                60,
                int(
                    math.ceil(
                        1.875
                        * maximum_joint_delta
                        / (
                            DESCENT_MAX_TRAJECTORY_SPEED_RAD_S
                            * PHYSICS_DT_SECONDS
                        )
                    )
                ),
            ),
            f"persistent_descent_{index:02d}",
        )
        previous = following
    grasp_joints = previous
    hold(grasp_joints, open_master, 60, "persistent_grasp_alignment")
    initial_target = world_position(target_prim)
    grasp_master = close_until_bilateral_contact(grasp_joints)
    start_grip_force_controller(grasp_master)
    hold(
        grasp_joints,
        grasp_master,
        120,
        "persistent_bilateral_hold",
        require_bilateral_contact=True,
    )

    pre_lift_target = world_position(target_prim)
    pre_lift_displacement = float(
        np.linalg.norm(pre_lift_target - initial_target)
    )
    bilateral = all(
        contacts["active_contact_pair_count"][side] > 0
        for side in ("left", "right")
    )
    force_measurements_available = all(
        count > 0 and contacts["maximum_force_n"][side] > 0.0
        for side, count in contacts["force_sample_count"].items()
    )
    pre_lift_grip_force_sufficient = all(
        contacts["maximum_force_n"][side]
        >= minimum_grip_force_per_finger_n
        for side in ("left", "right")
    )
    force_within_limit = all(
        force <= MAX_ALLOWED_CONTACT_FORCE_N
        for force in contacts["maximum_force_n"].values()
    )
    penetration_within_limit = all(
        depth <= MAX_ALLOWED_PENETRATION_M
        for depth in contacts["maximum_penetration_m"].values()
    )
    pre_lift_error = arm_error(grasp_joints)
    # Fail closed before any upward motion. A plausible gripper pose is not
    # enough if either finger lacks load, the target moved during closure, or
    # an unintended collision was reported.
    safety_gate = (
        np.all(np.isfinite(measured()))
        and bilateral
        and force_measurements_available
        and pre_lift_grip_force_sufficient
        and force_within_limit
        and penetration_within_limit
        and pre_lift_error <= 0.05
        and pre_lift_displacement <= 0.05
        and not contacts["unexpected_environment_pairs"]
    )
    if not safety_gate:
        save_failure_diagnostics(
            "persistent_pre_lift_gate",
            (
                f"bilateral={bilateral},error={pre_lift_error:.9f},"
                f"displacement={pre_lift_displacement:.9f},"
                f"force_measurements_available="
                f"{force_measurements_available},"
                f"pre_lift_grip_force_sufficient="
                f"{pre_lift_grip_force_sufficient},"
                f"force_within_limit={force_within_limit},"
                f"penetration_within_limit={penetration_within_limit},"
                f"unexpected={contacts['unexpected_environment_pairs']}"
            ),
        )
        raise RuntimeError(
            "Persistent pre-lift gate failed: "
            f"bilateral={bilateral}, error={pre_lift_error:.6f}, "
            f"displacement={pre_lift_displacement:.6f}, "
            f"force_measurements_available="
            f"{force_measurements_available}, "
            f"pre_lift_grip_force_sufficient="
            f"{pre_lift_grip_force_sufficient}, "
            f"maximum_force_n={contacts['maximum_force_n']}, "
            f"maximum_penetration_m="
            f"{contacts['maximum_penetration_m']}, "
            f"unexpected={contacts['unexpected_environment_pairs']}"
        )
    contact_lost_at_lift_start = dict(contacts["contact_lost_count"])
    contact_continuity["current_gap_steps"] = 0
    contact_continuity["maximum_gap_steps"] = 0
    grip_force_controller["target_frozen_after_pre_lift_gate"] = True
    start_target_stability_tracking()
    full_lift_start_joints = grasp_joints
    micro_lift_result = None
    if enable_micro_lift_force_validation:
        previous_micro_joints = grasp_joints
        micro_steps = max(
            12, math.ceil(120 / len(micro_lift_waypoints))
        )
        for index, following_micro_joints in enumerate(
            micro_lift_waypoints, start=1
        ):
            transition(
                previous_micro_joints,
                following_micro_joints,
                grasp_master,
                grasp_master,
                micro_steps,
                f"persistent_contact_micro_lift_{index:02d}",
                settle_steps=(
                    60 if index == len(micro_lift_waypoints) else 0
                ),
                require_bilateral_contact=True,
                max_relative_translation_m=(
                    MAX_MICRO_LIFT_RELATIVE_TRANSLATION_M
                ),
            )
            previous_micro_joints = following_micro_joints
        micro_target = world_position(target_prim)
        micro_lift_delta = float(
            micro_target[2] - pre_lift_target[2]
        )
        micro_lift_result = {
            "requested_height_m": MICRO_LIFT_HEIGHT_M,
            "measured_target_lift_m": micro_lift_delta,
            "minimum_required_target_lift_m": MINIMUM_MICRO_LIFT_DELTA_M,
            "maximum_relative_translation_m": stability[
                "maximum_relative_translation_m"
            ],
            "maximum_allowed_relative_translation_m": (
                MAX_MICRO_LIFT_RELATIVE_TRANSLATION_M
            ),
            "bilateral_contact": all(
                contacts["active_contact_pair_count"][side] > 0
                for side in ("left", "right")
            ),
        }
        if (
            micro_lift_delta < MINIMUM_MICRO_LIFT_DELTA_M
            or not micro_lift_result["bilateral_contact"]
        ):
            save_failure_diagnostics(
                "persistent_contact_micro_lift_gate",
                f"micro_lift_gate_failed:{micro_lift_result}",
            )
            raise RuntimeError(
                f"Persistent micro-lift gate failed: {micro_lift_result}"
            )
        full_lift_start_joints = micro_lift_joints
    remaining_lift_waypoints = [
        joints
        for height_m, joints in full_lift_waypoints
        if (
            not enable_micro_lift_force_validation
            or height_m > MICRO_LIFT_HEIGHT_M + 1.0e-9
        )
    ]
    previous_lift_joints = full_lift_start_joints
    lift_steps = max(
        10,
        math.ceil(
            FULL_LIFT_TRAJECTORY_STEPS / len(remaining_lift_waypoints)
        ),
    )
    for index, following_lift_joints in enumerate(
        remaining_lift_waypoints, start=1
    ):
        transition(
            previous_lift_joints,
            following_lift_joints,
            grasp_master,
            grasp_master,
            lift_steps,
            f"persistent_contact_lift_{index:02d}",
            settle_steps=0,
            require_bilateral_contact=True,
        )
        previous_lift_joints = following_lift_joints
    hold(
        lift_joints,
        grasp_master,
        120,
        "persistent_lift_verification",
        require_bilateral_contact=True,
    )
    final_arm_joints = lift_joints
    if transfer_joints is not None:
        previous_transfer_joints = lift_joints
        transfer_steps_per_waypoint = max(
            20,
            math.ceil(
                TRANSFER_TRAJECTORY_STEPS / len(transfer_waypoints)
            ),
        )
        for index, waypoint in enumerate(transfer_waypoints, start=1):
            following_transfer_joints = np.asarray(
                waypoint["joints_rad"], dtype=np.float64
            )
            transition(
                previous_transfer_joints,
                following_transfer_joints,
                grasp_master,
                grasp_master,
                transfer_steps_per_waypoint,
                f"persistent_contact_transfer_{index:02d}",
                settle_steps=0,
                require_bilateral_contact=True,
            )
            previous_transfer_joints = following_transfer_joints
        hold(
            transfer_joints,
            grasp_master,
            120,
            "persistent_transfer_verification",
            require_bilateral_contact=True,
        )
        final_arm_joints = transfer_joints

    transported_target = world_position(target_prim)
    transported_lift_delta = float(
        transported_target[2] - initial_target[2]
    )
    placement_result = {
        "planned": placement_joints is not None,
        "support_path": placement_support_path,
        "secondary_support_paths": list(
            secondary_placement_support_paths
        ),
        "support_contact_before_release": False,
        "support_contact_after_release": False,
        "support_contact_after_retreat": False,
        "release_executed": False,
        "retreat_executed": False,
        "finger_contact_cleared_after_retreat": False,
        "target_translation_after_release_m": None,
        "target_rotation_after_release_rad": None,
        "final_support_gap_m": None,
        "final_support_penetration_m": None,
        "stable_after_release": False,
    }
    placement_pose_target = None
    placement_pose_rotation = None
    if placement_joints is not None:
        previous_placement_joints = transfer_joints
        for index, waypoint in enumerate(placement_waypoints, start=1):
            following_placement_joints = np.asarray(
                waypoint["joints_rad"], dtype=np.float64
            )
            transition(
                previous_placement_joints,
                following_placement_joints,
                grasp_master,
                grasp_master,
                PLACEMENT_TRAJECTORY_STEPS,
                f"persistent_supported_placement_{index:02d}",
                settle_steps=0,
                require_bilateral_contact=True,
            )
            previous_placement_joints = following_placement_joints
        hold(
            placement_joints,
            grasp_master,
            180,
            "persistent_supported_placement_verification",
            require_bilateral_contact=True,
        )
        placement_result["support_contact_before_release"] = bool(
            active_placement_support_pairs
        )
        if not placement_result["support_contact_before_release"]:
            save_failure_diagnostics(
                "persistent_supported_placement_gate",
                "declared_support_contact_not_reached",
            )
            raise RuntimeError(
                "Cover placement did not reach the declared support surface"
            )
        if (
            contacts["placement_support_maximum_penetration_m"]
            > PLACEMENT_MAX_SUPPORT_PENETRATION_M
        ):
            save_failure_diagnostics(
                "persistent_supported_placement_gate",
                "support_penetration_limit_exceeded",
            )
            raise RuntimeError(
                "Cover placement exceeded the 2 mm support-penetration gate"
            )
        placement_pose_target = world_position(target_prim)
        placement_pose_rotation = world_rotation(target_prim)
        final_arm_joints = placement_joints

    pre_release_bilateral_contact = all(
        contacts["active_contact_pair_count"][side] > 0
        for side in ("left", "right")
    )
    pre_release_force_window = summarize_terminal_force_window(
        force_sample_history,
        current_step=int(contact_context["step"]),
        minimum_force_per_finger_n=minimum_grip_force_per_finger_n,
    )
    pre_release_force_sample_recent = bool(
        pre_release_force_window["samples_available_both_sides"]
    )
    pre_release_grip_force_sufficient = bool(
        pre_release_force_window["sufficient_both_sides"]
    )
    contact_maintained_before_release = (
        pre_release_bilateral_contact
        and contact_continuity["maximum_gap_steps"]
        <= MAX_CONSECUTIVE_CONTACT_GAP_STEPS
    )
    grasp_transport_stability = {
        "maximum_relative_translation_m": stability[
            "maximum_relative_translation_m"
        ],
        "maximum_relative_rotation_rad": stability[
            "maximum_relative_rotation_rad"
        ],
        "maximum_relative_angular_speed_rad_s": stability[
            "maximum_relative_angular_speed_rad_s"
        ],
        "sample_count": stability["sample_count"],
    }
    grasp_transport_stability_within_limit = (
        grasp_transport_stability["maximum_relative_translation_m"]
        <= MAX_RELATIVE_TARGET_TRANSLATION_M
        and grasp_transport_stability["maximum_relative_rotation_rad"]
        <= MAX_RELATIVE_TARGET_ROTATION_RAD
        and grasp_transport_stability["maximum_relative_angular_speed_rad_s"]
        <= MAX_RELATIVE_TARGET_ANGULAR_SPEED_RAD_S
    )

    if release_after_placement:
        stop_grip_force_controller()
        transition(
            placement_joints,
            placement_joints,
            grasp_master,
            open_master,
            RELEASE_TRAJECTORY_STEPS,
            "persistent_supported_release",
            settle_steps=120,
        )
        placement_result["release_executed"] = True
        placement_result["support_contact_after_release"] = bool(
            active_placement_support_pairs
        )
        previous_retreat_joints = placement_joints
        for index, waypoint in enumerate(retreat_waypoints, start=1):
            following_retreat_joints = np.asarray(
                waypoint["joints_rad"], dtype=np.float64
            )
            transition(
                previous_retreat_joints,
                following_retreat_joints,
                open_master,
                open_master,
                PLACEMENT_TRAJECTORY_STEPS,
                f"persistent_post_release_retreat_{index:02d}",
                settle_steps=0,
            )
            previous_retreat_joints = following_retreat_joints
        hold(
            retreat_joints,
            open_master,
            120,
            "persistent_post_release_retreat_verification",
        )
        placement_result["retreat_executed"] = True
        placement_result["support_contact_after_retreat"] = bool(
            active_placement_support_pairs
        )
        placement_result["finger_contact_cleared_after_retreat"] = all(
            contacts["active_contact_pair_count"][side] == 0
            for side in ("left", "right")
        )
        final_arm_joints = retreat_joints

        final_placed_target = world_position(target_prim)
        final_placed_rotation = world_rotation(target_prim)
        placement_result["target_translation_after_release_m"] = float(
            np.linalg.norm(final_placed_target - placement_pose_target)
        )
        placement_result["target_rotation_after_release_rad"] = (
            rotation_angle_rad(
                final_placed_rotation @ placement_pose_rotation.T
            )
        )
        support_min, support_max = fresh_aligned_world_bounds(
            placement_support_path
        )
        plate_path = f"{manipulation_target_path}/Plate"
        if not stage.GetPrimAtPath(plate_path).IsValid():
            plate_path = manipulation_target_path
        plate_min, plate_max = fresh_aligned_world_bounds(plate_path)
        placement_result["final_support_gap_m"] = max(
            0.0, float(plate_min[2] - support_max[2])
        )
        placement_result["final_support_penetration_m"] = max(
            0.0, float(support_max[2] - plate_min[2])
        )
        placement_result["final_plate_bounds_world_m"] = {
            "minimum": plate_min.tolist(),
            "maximum": plate_max.tolist(),
        }
        placement_result["stable_after_release"] = bool(
            placement_result["support_contact_after_release"]
            and placement_result["support_contact_after_retreat"]
            and placement_result["finger_contact_cleared_after_retreat"]
            and placement_result["target_translation_after_release_m"]
            <= 0.020
            and placement_result["target_rotation_after_release_rad"]
            <= math.radians(5.0)
            and placement_result["final_support_gap_m"] <= 0.002
            and placement_result["final_support_penetration_m"]
            <= PLACEMENT_MAX_SUPPORT_PENETRATION_M
            and contacts["placement_support_maximum_penetration_m"]
            <= PLACEMENT_MAX_SUPPORT_PENETRATION_M
        )

    final_target = world_position(target_prim)
    lift_delta = transported_lift_delta
    horizontal_slip = float(
        np.linalg.norm(final_target[:2] - pre_lift_target[:2])
    )
    expected_horizontal_motion = (
        float(np.linalg.norm(np.asarray(transfer_offset_world_m)[:2]))
        if transfer_offset_world_m is not None
        else 0.0
    )
    horizontal_motion_error = abs(
        horizontal_slip - expected_horizontal_motion
    )
    final_force_within_limit = all(
        force <= MAX_ALLOWED_CONTACT_FORCE_N
        for force in contacts["maximum_force_n"].values()
    )
    final_penetration_within_limit = all(
        depth <= MAX_ALLOWED_PENETRATION_M
        for depth in contacts["maximum_penetration_m"].values()
    )
    final_bilateral_contact = all(
        contacts["active_contact_pair_count"][side] > 0
        for side in ("left", "right")
    )
    final_force_window = summarize_terminal_force_window(
        force_sample_history,
        current_step=int(contact_context["step"]),
        minimum_force_per_finger_n=minimum_grip_force_per_finger_n,
    )
    final_force_sample_recent = bool(
        final_force_window["samples_available_both_sides"]
    )
    final_grip_force_sufficient = bool(
        final_force_window["sufficient_both_sides"]
    )
    contact_maintained_after_lift = (
        final_bilateral_contact
        and contact_continuity["maximum_gap_steps"]
        <= MAX_CONSECUTIVE_CONTACT_GAP_STEPS
    )
    target_stability_within_limit = grasp_transport_stability_within_limit
    terminal_contact_gate = (
        placement_result["stable_after_release"]
        if release_after_placement
        else (
            final_bilateral_contact
            and final_grip_force_sufficient
            and contact_maintained_after_lift
        )
    )
    success = (
        np.all(np.isfinite(measured()))
        and lift_delta >= minimum_verified_lift_m
        and force_measurements_available
        and pre_release_bilateral_contact
        and pre_release_grip_force_sufficient
        and contact_maintained_before_release
        and terminal_contact_gate
        and target_stability_within_limit
        and (
            horizontal_slip <= 0.03
            if transfer_offset_world_m is None
            else horizontal_motion_error <= MAX_TRANSFER_HORIZONTAL_ERROR_M
        )
        and final_force_within_limit
        and final_penetration_within_limit
        and not contacts["unexpected_environment_pairs"]
        and not contacts["unexpected_target_environment_pairs"]
    )
    if record_debug_video:
        video_started = time.perf_counter()
        video = build_frame_sequence_video(
            frames,
            output_root / "persistent_composite_grasp.mp4",
            fps=10,
            crf=17,
            preset="slow",
            purpose="same_process_composite_grasp_physics_pilot",
        )
        video_encoding_seconds = time.perf_counter() - video_started
    else:
        video = {
            "status": "disabled",
            "output_path": None,
            "frame_count": 0,
            "purpose": "evaluation_without_debug_recording",
        }
    execution_seconds = time.perf_counter() - execution_started
    result = {
        "schema_version": "persistent-composite-grasp-pilot-v3",
        "status": "completed" if success else "failed",
        "seed": seed,
        "timing": {
            "execution_wall_seconds": execution_seconds,
            "debug_frame_capture_seconds": debug_capture_seconds,
            "video_encoding_seconds": video_encoding_seconds,
            "core_execution_excluding_debug_recording_seconds": max(
                0.0,
                execution_seconds
                - debug_capture_seconds
                - video_encoding_seconds,
            ),
            "debug_frame_count": len(frames),
            "debug_recording_enabled": record_debug_video,
            "physics_only_steps_enabled": physics_only_steps,
        },
        "manipulation_label": manipulation_label,
        "manipulation_target_path": manipulation_target_path,
        "contact_target_path": contact_target_path,
        "same_isaac_process_as_observation_server": True,
        "same_stage_prims_preserved": True,
        "coordinate_reexpression_applied": (
            False
            if reuse_existing_composite
            else WORLD_TO_ROBOT_BASE.tolist()
        ),
        "validated_seed0_plan_source": (
            str(validated_result_path)
            if validated_result_path is not None
            else None
        ),
        "trajectory_source": trajectory_source,
        "grasp_height_offset_m": grasp_height_offset_m,
        "ik": plan,
        "rgbd_localization_selection": (
            rgbd_localization.get("selection")
            if rgbd_localization is not None
            else None
        ),
        "ur10e_link_collision_enabled": True,
        "target_physics_proxy": (
            "existing_dynamic_assembly"
            if existing_dynamic_assembly
            else (
                "household_mug_cylinder_child"
                if household_mug_physics_proxy
                else "prevalidated_cube"
            )
        ),
        "target_physics": {
            "outer_radius_m": (
                HOUSEHOLD_MUG_OUTER_RADIUS_M
                if household_mug_physics_proxy
                else None
            ),
            "height_m": (
                HOUSEHOLD_MUG_HEIGHT_M
                if household_mug_physics_proxy
                else None
            ),
            "mass_kg": target_mass_kg,
            "static_friction": grip_static_friction,
            "dynamic_friction": grip_dynamic_friction,
            "values_are_provisional_not_real_robot_calibrated": True,
            "initial_position_semantics": (
                target_position_semantics
                if household_mug_physics_proxy
                else "cube_center"
            ),
        },
        "dof_count": len(dof_names),
        "rg6_actuation": {
            "mode": rg6_coupling_mode,
            "master_joint": rg6_master_name,
            "follower_ratios": rg6_follower_ratios,
            "follower_drives_removed_in_memory": follower_drive_removal,
            "mimic_apis_removed_in_memory": mimic_api_removal,
            "provisional_initial_drive_effort_nm": (
                rg6_master_max_torque_nm
            ),
            "provisional_aggregate_drive_effort_limit_nm": (
                coordinated_total_drive_effort_limit_nm
                if rg6_coupling_mode == "coordinated_drives"
                else force_controller_max_torque_nm
            ),
            "drive_effort_interpretation": (
                "development_joint_drive_effort_not_rg6_motor_torque"
                if rg6_coupling_mode == "coordinated_drives"
                else "single_master_drive_torque_proxy"
            ),
            "follower_target_basis": (
                f"measured_plus_{coordinated_follower_request_blend:.3f}_"
                "times_requested_delta_each_physics_step"
                if rg6_coupling_mode == "coordinated_drives"
                else "newton_mimic_api"
            ),
            "final_coupling_error_rad": mimic_error(),
            "transfer_ready": False,
            "source_asset_files_modified": False,
        },
        "contact_control": {
            "mode": "bilateral_force_controlled_closure",
            "force_gate": (
                "active_contact_pairs_closure_peak_and_terminal_2s_window"
            ),
            "closure_limit_rad": close_master,
            "closure_steps": CONTACT_CONTROLLED_CLOSURE_STEPS,
            "minimum_force_per_finger_n": (
                minimum_grip_force_per_finger_n
            ),
            "minimum_combined_force_n": minimum_combined_grip_force_n,
            "final_master_rad": grasp_master,
            "final_commanded_master_rad": (
                open_master if release_after_placement else grasp_master
            ),
            "maximum_allowed_force_n": MAX_ALLOWED_CONTACT_FORCE_N,
            "maximum_allowed_penetration_m": (
                MAX_ALLOWED_PENETRATION_M
            ),
            "grip_material_collision_paths": sorted(
                grip_material_collision_paths
            ),
            "continuous_force_controller": grip_force_controller,
            "compliant_contact": {
                "enabled": grip_compliant_contact_stiffness_n_m > 0.0,
                "stiffness_n_m": grip_compliant_contact_stiffness_n_m,
                "damping_n_s_m": grip_compliant_contact_damping_n_s_m,
                "acceleration_spring": False,
                "values_are_provisional_not_real_epdm_measurements": True,
            },
            "micro_lift_validation": micro_lift_result,
        },
        "bilateral_contact_before_lift": bilateral,
        "bilateral_contact_after_lift": final_bilateral_contact,
        "bilateral_contact_before_release": pre_release_bilateral_contact,
        "left_contact_events": contacts["left"],
        "right_contact_events": contacts["right"],
        "contact_force_n": {
            "latest": contacts["latest_force_n"],
            "last_reported": contacts["last_reported_force_n"],
            "maximum": contacts["maximum_force_n"],
            "sample_count": contacts["force_sample_count"],
        },
        "active_contact_pair_count": contacts[
            "active_contact_pair_count"
        ],
        "contact_lost_count": contacts["contact_lost_count"],
        "contact_lost_at_lift_start": contact_lost_at_lift_start,
        "contact_maintained_after_lift": (
            contact_maintained_before_release
            if release_after_placement
            else contact_maintained_after_lift
        ),
        "contact_maintained_before_release": (
            contact_maintained_before_release
        ),
        "contact_continuity": {
            **contact_continuity,
            "maximum_allowed_gap_steps": (
                MAX_CONSECUTIVE_CONTACT_GAP_STEPS
            ),
            "physics_dt_seconds": PHYSICS_DT_SECONDS,
        },
        "final_force_sample_recent": final_force_sample_recent,
        "final_force_window": final_force_window,
        "pre_release_force_sample_recent": (
            pre_release_force_sample_recent
        ),
        "pre_release_force_window": pre_release_force_window,
        "contact_force_measurements_available": (
            force_measurements_available
        ),
        "pre_lift_grip_force_sufficient": (
            pre_lift_grip_force_sufficient
        ),
        "final_grip_force_sufficient": final_grip_force_sufficient,
        "pre_release_grip_force_sufficient": (
            pre_release_grip_force_sufficient
        ),
        "contact_events_by_phase": contacts["events_by_phase"],
        "raw_contact_pairs": contacts["raw_pairs"],
        "maximum_contact_penetration_m": contacts[
            "maximum_penetration_m"
        ],
        "unexpected_environment_pairs": contacts[
            "unexpected_environment_pairs"
        ],
        "unexpected_target_environment_pairs": contacts[
            "unexpected_target_environment_pairs"
        ],
        "pre_lift_arm_error_rad": pre_lift_error,
        "pre_lift_target_displacement_m": pre_lift_displacement,
        "verified_lift_delta_m": lift_delta,
        "horizontal_slip_during_lift_m": horizontal_slip,
        "expected_horizontal_motion_m": expected_horizontal_motion,
        "horizontal_motion_error_m": horizontal_motion_error,
        "maximum_transfer_horizontal_error_m": (
            MAX_TRANSFER_HORIZONTAL_ERROR_M
        ),
        "target_gripper_relative_stability": {
            "measurement_window": "grasp_through_supported_pre_release_hold",
            "maximum_translation_m": grasp_transport_stability[
                "maximum_relative_translation_m"
            ],
            "maximum_rotation_rad": grasp_transport_stability[
                "maximum_relative_rotation_rad"
            ],
            "maximum_angular_speed_rad_s": grasp_transport_stability[
                "maximum_relative_angular_speed_rad_s"
            ],
            "sample_count": grasp_transport_stability["sample_count"],
            "limits": {
                "maximum_translation_m": (
                    MAX_RELATIVE_TARGET_TRANSLATION_M
                ),
                "maximum_rotation_rad": MAX_RELATIVE_TARGET_ROTATION_RAD,
                "maximum_angular_speed_rad_s": (
                    MAX_RELATIVE_TARGET_ANGULAR_SPEED_RAD_S
                ),
            },
            "within_limit": target_stability_within_limit,
        },
        "transfer_executed": transfer_joints is not None,
        "supported_placement": {
            **placement_result,
            "planned_geometry": plan.get("placement_geometry"),
            "contact_event_count": contacts[
                "placement_support_contact_events"
            ],
            "contact_pairs": contacts[
                "placement_support_contact_pairs"
            ],
            "maximum_reported_support_penetration_m": contacts[
                "placement_support_maximum_penetration_m"
            ],
            "maximum_allowed_support_penetration_m": (
                PLACEMENT_MAX_SUPPORT_PENETRATION_M
            ),
        },
        "minimum_verified_lift_m": minimum_verified_lift_m,
        "contact_force_within_limit": final_force_within_limit,
        "contact_penetration_within_limit": (
            final_penetration_within_limit
        ),
        "lift_verified": success,
        "removal_verified": bool(
            success
            and manipulation_label == "cover"
            and transfer_joints is not None
        ),
        "cover_placed_and_released": bool(
            success
            and manipulation_label == "cover"
            and release_after_placement
            and placement_result["stable_after_release"]
        ),
        "final_arm_joints_rad": final_arm_joints.tolist(),
        "finite_final_joint_state": bool(np.all(np.isfinite(measured()))),
        "trajectory_records": trajectory_records,
        "video": video,
        # Keep the no-attachment rule explicit in every saved result so lift
        # evidence cannot be confused with a fixed-joint or pose-copy shortcut.
        "target_attachment_used": False,
        "target_pose_copying_used": False,
        "observation_viewpoint_motion_mode": (
            "actual_ur10e_continuous_joint_physics"
            if reuse_existing_composite
            else "fixed_debug_wrist_coordinates_before_terminal_manipulation"
        ),
        "valid_for_final_evaluation": False,
    }
    result_path = output_root / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    # Release this callback handle without touching the caller-owned app or
    # stage; the persistent server may continue with another observation.
    contact_subscription = None
    return result
