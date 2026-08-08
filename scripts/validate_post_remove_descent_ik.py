#!/usr/bin/env python3
"""Validate both parallel-gripper descent IK branches from a saved arm state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp

from persistent_composite_grasp import (
    DEFAULT_TABLE_ALIGNED_GRASP_YAW_RAD,
    DESCENT_OFFSETS_M,
    MAX_DESCENT_IK_JOINT_STEP_RAD,
    LIFT_IK_ORIENTATION_TOLERANCE_RAD,
    LIFT_IK_POSITION_TOLERANCE_M,
    closest_equivalent_joint_configuration,
    parallel_gripper_yaw_candidates,
    rotation_vector_from_matrix,
)
from run_single_gpu_pilot import configured_physical_gpu, require_single_gpu_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("removal_result", type=Path)
    parser.add_argument("localization", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require_single_gpu_policy()
    physical_gpu = configured_physical_gpu()
    removal = json.loads(args.removal_result.read_text(encoding="utf-8"))
    localization = json.loads(args.localization.read_text(encoding="utf-8"))
    warm_start = np.asarray(removal["final_arm_joints_rad"], dtype=np.float64)
    target = np.asarray(
        localization["estimates"]["selected_target"]["center_world_m"],
        dtype=np.float64,
    )

    app = SimulationApp(
        {
            "headless": True,
            "active_gpu": physical_gpu,
            "physics_gpu": 0,
            "multi_gpu": False,
            "max_gpu_count": 1,
            "extra_args": ["--/renderer/multiGpu/autoEnable=false"],
            "fast_shutdown": True,
        }
    )
    try:
        from isaacsim.core.utils.extensions import get_extension_path_from_name
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver

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
                ur10e_config / "rmpflow" / "ur10e_robot_description.yaml"
            ),
            urdf_path=str(ur10e_config / "ur10e.urdf"),
        )
        kinematics.set_robot_base_pose(
            np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        )
        frame = next(
            name
            for name in ("flange", "ee_link", "tool0", "wrist_3_link")
            if name in kinematics.get_all_frame_names()
        )
        grasp_position = target + np.asarray([0.0, 0.0, 0.287])

        def solve_local_pose(
            position: np.ndarray,
            rotation: np.ndarray,
            initial: np.ndarray,
        ) -> tuple[np.ndarray, dict]:
            joints = np.asarray(initial, dtype=np.float64).copy()
            for iteration in range(1, 81):
                actual_position, actual_rotation = (
                    kinematics.compute_forward_kinematics(
                        frame, joint_positions=joints
                    )
                )
                position_error = position - np.asarray(actual_position)
                orientation_error = rotation_vector_from_matrix(
                    rotation @ np.asarray(actual_rotation).T
                )
                if (
                    np.linalg.norm(position_error)
                    <= LIFT_IK_POSITION_TOLERANCE_M
                    and np.linalg.norm(orientation_error)
                    <= LIFT_IK_ORIENTATION_TOLERANCE_RAD
                ):
                    return joints, {"iterations": iteration - 1}
                jacobian = np.asarray(
                    kinematics._kinematics.jacobian(joints, frame),
                    dtype=np.float64,
                )
                error = np.concatenate([position_error, orientation_error])
                regularized = (
                    jacobian @ jacobian.T
                    + 1.0e-8 * np.eye(6, dtype=np.float64)
                )
                update = jacobian.T @ np.linalg.solve(regularized, error)
                update /= max(1.0, float(np.max(np.abs(update))) / 0.03)
                joints += update
            raise RuntimeError("local IK did not converge in 80 iterations")

        attempts = []
        for yaw in parallel_gripper_yaw_candidates(
            DEFAULT_TABLE_ALIGNED_GRASP_YAW_RAD
        ):
            orientation = np.asarray(
                [0.0, np.cos(yaw * 0.5), np.sin(yaw * 0.5), 0.0],
                dtype=np.float64,
            )
            current = warm_start.copy()
            fixed_rotation = None
            waypoints = []
            failure = None
            for offset_m in DESCENT_OFFSETS_M:
                position = grasp_position + np.asarray([0.0, 0.0, offset_m])
                if not waypoints:
                    joints, success = kinematics.compute_inverse_kinematics(
                        frame,
                        position,
                        orientation,
                        warm_start=current,
                        position_tolerance=0.003,
                        orientation_tolerance=0.03,
                    )
                    local_solver = None
                else:
                    try:
                        joints, local_solver = solve_local_pose(
                            position, fixed_rotation, current
                        )
                        success = True
                    except RuntimeError as error:
                        failure = (
                            f"local_ik_failed_at_offset_{offset_m:.2f}_m: "
                            f"{error}"
                        )
                        break
                joints = closest_equivalent_joint_configuration(
                    np.asarray(joints, dtype=np.float64), current
                )
                step = float(np.max(np.abs(joints - current)))
                if not success:
                    failure = f"ik_failed_at_offset_{offset_m:.2f}_m"
                    break
                if waypoints and step > MAX_DESCENT_IK_JOINT_STEP_RAD:
                    failure = (
                        f"ik_branch_change_at_offset_{offset_m:.2f}_m_"
                        f"step_{step:.6f}_rad"
                    )
                    break
                if fixed_rotation is None:
                    _position, fixed_rotation = (
                        kinematics.compute_forward_kinematics(
                            frame, joint_positions=joints
                        )
                    )
                    fixed_rotation = np.asarray(
                        fixed_rotation, dtype=np.float64
                    )
                waypoints.append(
                    {
                        "offset_m": offset_m,
                        "maximum_joint_step_rad": step,
                        "joints_rad": joints.tolist(),
                        "solver": (
                            "lula_global_warm_start"
                            if local_solver is None
                            else "lula_analytic_jacobian_damped_local_ik"
                        ),
                    }
                )
                current = joints
            attempts.append(
                {
                    "grasp_yaw_rad": yaw,
                    "status": "accepted" if failure is None else "rejected",
                    "failure_reason": failure,
                    "waypoints_solved": len(waypoints),
                    "maximum_local_joint_step_rad": max(
                        (item["maximum_joint_step_rad"] for item in waypoints[1:]),
                        default=0.0,
                    ),
                    "waypoints": waypoints,
                }
            )
        result = {
            "schema_version": "post-remove-descent-ik-validation-v1",
            "status": (
                "completed"
                if any(item["status"] == "accepted" for item in attempts)
                else "failed"
            ),
            "source_removal_result": str(args.removal_result.resolve()),
            "source_localization": str(args.localization.resolve()),
            "initial_arm_joints_rad": warm_start.tolist(),
            "target_center_world_m": target.tolist(),
            "attempts": attempts,
            "physical_execution_performed": False,
            "valid_for_final_evaluation": False,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        temporary.replace(args.output)
        print(json.dumps(result, indent=2))
        if result["status"] != "completed":
            raise RuntimeError("No continuous post-remove descent IK branch")
    finally:
        app.close()


if __name__ == "__main__":
    main()
