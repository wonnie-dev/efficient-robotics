"""CPU tests for the removable-cover physical stability contract."""

import math
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from persistent_composite_grasp import (  # noqa: E402
    MAX_ALLOWED_CONTACT_FORCE_N,
    MAX_ALLOWED_PENETRATION_M,
    MAX_MICRO_LIFT_RELATIVE_TRANSLATION_M,
    LIFT_IK_ORIENTATION_TOLERANCE_RAD,
    LIFT_IK_POSITION_TOLERANCE_M,
    MAX_LIFT_IK_ORIENTATION_ERROR_RAD,
    MAX_LIFT_IK_POSITION_ERROR_M,
    MAX_LIFT_IK_JOINT_STEP_RAD,
    MICRO_LIFT_CARTESIAN_WAYPOINTS,
    FULL_LIFT_CARTESIAN_WAYPOINTS,
    FULL_LIFT_TRAJECTORY_STEPS,
    TRANSFER_TRAJECTORY_STEPS,
    MAX_DESCENT_IK_JOINT_STEP_RAD,
    MICRO_LIFT_HEIGHT_M,
    MINIMUM_MICRO_LIFT_DELTA_M,
    PREGRASP_MAX_TRAJECTORY_SPEED_RAD_S,
    DESCENT_MAX_TRAJECTORY_SPEED_RAD_S,
    DESCENT_OFFSETS_M,
    GRIP_FORCE_TARGET_INCREMENT_RAD,
    GRIP_CONTROLLER_FORCE_WINDOW_STEPS,
    GRIP_CONTROLLER_STEP_SLIP_M,
    PROVISIONAL_EPDM_LID_DYNAMIC_FRICTION,
    PROVISIONAL_EPDM_LID_STATIC_FRICTION,
    PROVISIONAL_LID_COMBINED_GRIP_FORCE_N,
    PROVISIONAL_LID_GRIP_FORCE_PER_FINGER_N,
    RG6_HANDLE_CLOSE_MASTER_RAD,
    PROVISIONAL_DYNAMIC_FRICTION,
    PROVISIONAL_TARGET_COORDINATED_DRIVE_EFFORT_LIMIT_NM,
    PROVISIONAL_TARGET_FOLLOWER_REQUEST_BLEND,
    PROVISIONAL_TARGET_MINIMUM_GRIP_FORCE_PER_FINGER_N,
    PROVISIONAL_TARGET_PASSIVE_FORCE_CONTROLLER_MAX_TORQUE_NM,
    PLACEMENT_CARTESIAN_WAYPOINTS,
    PLACEMENT_MAX_SUPPORT_PENETRATION_M,
    PLACEMENT_SUPPORT_PENETRATION_M,
    PLACEMENT_TRAJECTORY_STEPS,
    MINIMUM_SUPPORT_CENTER_MARGIN_M,
    MINIMUM_SUPPORT_OVERLAP_FRACTION,
    RETREAT_CARTESIAN_WAYPOINTS,
    RETREAT_DISTANCE_M,
    TERMINAL_FORCE_WINDOW_STEPS,
    MINIMUM_TERMINAL_FORCE_QUALIFYING_STEPS,
    grasp_yaw_from_pinch_axis_world,
    parallel_gripper_yaw_candidates,
    rank_outside_container_grasp_yaws,
    closest_equivalent_joint_configuration,
    quintic_time_scaling,
    rotation_angle_rad,
    rotation_vector_from_matrix,
    should_rebuild_persistent_physics,
    summarize_terminal_force_window,
    supported_root_height_m,
    planar_support_metrics,
)
from scanned_basket_scene import (  # noqa: E402
    BASKET_COLLISION_BOXES_LOCAL_M,
    CALIBRATION_COVER_FULL_EXTENTS_M,
    MANIPULABLE_COVER_CENTER_OF_MASS_LOCAL_M,
    MANIPULABLE_COVER_HANDLE_FULL_EXTENTS_M,
    MANIPULABLE_COVER_INERTIA_KG_M2,
    MANIPULABLE_COVER_MASS_KG,
    _composite_cover_mass_properties,
)
from execute_cover_removal import (  # noqa: E402
    calibration_authorizes_remove_cover,
    contact_grasp_success,
    removal_contact_success,
)


class LidPhysicsStabilityTests(unittest.TestCase):
    def test_forced_calibration_rejects_reserved_seed(self) -> None:
        with self.assertRaises(ValueError):
            calibration_authorizes_remove_cover(
                1100, forced_observation_calibration=True
            )

    def test_forced_calibration_records_intervention(self) -> None:
        result = calibration_authorizes_remove_cover(
            1099, forced_observation_calibration=True
        )
        self.assertEqual(
            result["authorization_mode"],
            "forced_observation_calibration_intervention",
        )
        self.assertFalse(result["testing_performed"])

    def test_terminal_force_window_uses_recent_distinct_timesteps(self) -> None:
        samples = {
            "left": [
                {"step": 50, "force_n": 20.0},
                {"step": 190, "force_n": 3.2},
                {"step": 191, "force_n": 0.0},
                {"step": 191, "force_n": 3.4},
                {"step": 192, "force_n": 3.1},
            ],
            "right": [
                {"step": 190, "force_n": 3.3},
                {"step": 191, "force_n": 3.2},
                {"step": 192, "force_n": 3.1},
            ],
        }
        summary = summarize_terminal_force_window(
            samples,
            current_step=200,
            minimum_force_per_finger_n=3.0,
            window_steps=20,
            minimum_qualifying_steps=3,
        )
        self.assertTrue(summary["sufficient_both_sides"])
        self.assertEqual(
            summary["by_side"]["left"]["qualifying_step_count"], 3
        )
        self.assertEqual(summary["by_side"]["left"]["maximum_force_n"], 3.4)

    def test_terminal_force_window_rejects_one_sided_or_old_force(self) -> None:
        samples = {
            "left": [
                {"step": 1, "force_n": 30.0},
                {"step": 99, "force_n": 4.0},
                {"step": 100, "force_n": 4.0},
                {"step": 101, "force_n": 4.0},
            ],
            "right": [
                {"step": 99, "force_n": 4.0},
                {"step": 100, "force_n": 0.0},
                {"step": 101, "force_n": 0.0},
            ],
        }
        summary = summarize_terminal_force_window(
            samples,
            current_step=101,
            minimum_force_per_finger_n=3.0,
            window_steps=10,
            minimum_qualifying_steps=3,
        )
        self.assertFalse(summary["sufficient_both_sides"])
        self.assertEqual(
            summary["by_side"]["right"]["qualifying_step_count"], 1
        )
        self.assertEqual(TERMINAL_FORCE_WINDOW_STEPS, 120)
        self.assertEqual(MINIMUM_TERMINAL_FORCE_QUALIFYING_STEPS, 3)
        self.assertEqual(GRIP_CONTROLLER_FORCE_WINDOW_STEPS, 30)
        self.assertGreater(GRIP_CONTROLLER_STEP_SLIP_M, 0.0)
        self.assertLess(GRIP_CONTROLLER_STEP_SLIP_M, 0.001)
        self.assertEqual(
            PROVISIONAL_TARGET_COORDINATED_DRIVE_EFFORT_LIMIT_NM, 6.0
        )
        self.assertEqual(PROVISIONAL_TARGET_FOLLOWER_REQUEST_BLEND, 1.0)
        self.assertEqual(
            PROVISIONAL_TARGET_MINIMUM_GRIP_FORCE_PER_FINGER_N, 5.0
        )
        self.assertEqual(
            PROVISIONAL_TARGET_PASSIVE_FORCE_CONTROLLER_MAX_TORQUE_NM, 1.2
        )

    def test_persistent_physics_rebuild_occurs_only_on_first_call(self) -> None:
        self.assertTrue(
            should_rebuild_persistent_physics(
                reuse_existing_composite=True,
                executor_already_prepared=False,
            )
        )
        self.assertFalse(
            should_rebuild_persistent_physics(
                reuse_existing_composite=True,
                executor_already_prepared=True,
            )
        )
        self.assertTrue(
            should_rebuild_persistent_physics(
                reuse_existing_composite=False,
                executor_already_prepared=True,
            )
        )

    def test_contact_lift_uses_transfer_conservative_duration(self) -> None:
        self.assertEqual(FULL_LIFT_TRAJECTORY_STEPS, 900)
        self.assertGreaterEqual(
            math.ceil(
                FULL_LIFT_TRAJECTORY_STEPS
                / (FULL_LIFT_CARTESIAN_WAYPOINTS - 1)
            ),
            50,
        )

    def test_released_cover_success_uses_pre_release_not_final_contact(self) -> None:
        removal = {
            "cover_placed_and_released": True,
            "bilateral_contact_after_lift": False,
            "bilateral_contact_before_release": True,
            "contact_maintained_before_release": True,
            "supported_placement": {
                "release_executed": True,
                "retreat_executed": True,
                "finger_contact_cleared_after_retreat": True,
                "stable_after_release": True,
            },
        }
        self.assertTrue(removal_contact_success(removal))
        removal["supported_placement"]["stable_after_release"] = False
        self.assertFalse(removal_contact_success(removal))

    def test_post_remove_target_grasp_requires_every_physical_gate(self) -> None:
        grasp = {
            "grasp_executed": True,
            "grasp_execution": {
                "lift_verified": True,
                "bilateral_contact_before_lift": True,
                "unexpected_environment_pairs": [],
                "contact_force_within_limit": True,
                "contact_penetration_within_limit": True,
            },
        }
        self.assertTrue(contact_grasp_success(grasp))
        grasp["grasp_execution"]["bilateral_contact_before_lift"] = False
        self.assertFalse(contact_grasp_success(grasp))

    def test_full_episode_has_explicit_provisional_coordinated_coupling(self) -> None:
        source = (ROOT / "scripts" / "persistent_composite_grasp.py").read_text(
            encoding="utf-8"
        )
        launcher = (ROOT / "scripts" / "isaac_sim_server.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('rg6_coupling_mode: str = "passive_mimic"', source)
        self.assertIn('joint.RemoveAPI("NewtonMimicAPI")', source)
        self.assertIn("Never retain mimic and follower drives together", source)
        self.assertIn("total_effort_nm / len(rg6_drives)", source)
        self.assertIn("ratio * follower_master_target", source)
        self.assertIn("rg6_coordinated_drives_prepared", source)
        self.assertIn("restored_mimic_requires_physics_rebuild", source)
        self.assertIn("Released cover pose changed during RG6 mimic rebuild", source)
        self.assertIn(
            "Previously prepared RG6 follower has no coordinated", source
        )
        self.assertIn(
            "quarter_measured_three_quarter_requested_master_each_physics_step",
            source,
        )
        self.assertIn('"development_joint_drive_effort_not_rg6_motor_torque"', source)
        self.assertIn('"transfer_ready": False', source)
        self.assertIn('choices=("passive_mimic", "coordinated_drives")', launcher)
        self.assertIn(
            "rg6_coupling_mode=args.rg6_coupling_mode", launcher
        )
        self.assertIn(
            "args.coordinated_rg6_total_drive_effort_limit_nm", launcher
        )

    def test_ik_joint_solution_is_unwrapped_near_current_state(self) -> None:
        result = closest_equivalent_joint_configuration(
            np.asarray([2.0 * math.pi + 0.1, -2.0 * math.pi - 0.2]),
            np.asarray([0.0, 0.0]),
        )
        np.testing.assert_allclose(result, [0.1, -0.2], atol=1.0e-12)

    def test_pregrasp_speed_limit_is_conservative(self) -> None:
        self.assertGreater(PREGRASP_MAX_TRAJECTORY_SPEED_RAD_S, 0.0)
        self.assertLessEqual(PREGRASP_MAX_TRAJECTORY_SPEED_RAD_S, 0.35)

    def test_descent_uses_dense_local_branch_limited_waypoints(self) -> None:
        self.assertLessEqual(DESCENT_MAX_TRAJECTORY_SPEED_RAD_S, 0.20)
        self.assertLessEqual(MAX_DESCENT_IK_JOINT_STEP_RAD, 0.45)
        self.assertEqual(DESCENT_OFFSETS_M[0], 0.18)
        self.assertEqual(DESCENT_OFFSETS_M[-1], 0.0)
        self.assertTrue(
            all(
                math.isclose(left - right, 0.01, abs_tol=1.0e-12)
                for left, right in zip(
                    DESCENT_OFFSETS_M, DESCENT_OFFSETS_M[1:]
                )
            )
        )

    def test_handle_local_y_pinch_axis_maps_to_table_aligned_rg6(self) -> None:
        yaw = grasp_yaw_from_pinch_axis_world(
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        )
        self.assertAlmostEqual(abs(yaw), math.pi)
        closure_axis_yaw = yaw - math.pi * 0.5
        self.assertAlmostEqual(closure_axis_yaw, math.pi * 0.5)

    def test_parallel_gripper_yaw_candidates_share_the_same_pinch_line(self) -> None:
        first, second = parallel_gripper_yaw_candidates(math.pi)
        self.assertAlmostEqual(abs(first), math.pi)
        self.assertAlmostEqual(second, 0.0, places=12)
        self.assertAlmostEqual(
            abs(math.atan2(math.sin(second - first), math.cos(second - first))),
            math.pi,
        )

    def test_outside_corner_grasp_prefers_axis_clear_of_wall_end(self) -> None:
        target_xy = np.asarray([0.913, 0.001], dtype=np.float64)
        wall_bounds = [
            (
                np.asarray([0.505, 0.013, 0.0], dtype=np.float64),
                np.asarray([0.855, 0.031, 0.2], dtype=np.float64),
            ),
            (
                np.asarray([1.003, -0.445, 0.0], dtype=np.float64),
                np.asarray([1.021, 0.005, 0.2], dtype=np.float64),
            ),
        ]
        candidates, ranking = rank_outside_container_grasp_yaws(
            2.0579804469193013,
            target_xy,
            wall_bounds,
        )
        best_pinch_axis = np.asarray(ranking[0]["pinch_axis_xy"])
        self.assertGreater(
            ranking[0]["minimum_wall_aabb_clearance_m"],
            ranking[-1]["minimum_wall_aabb_clearance_m"],
        )
        self.assertGreater(abs(best_pinch_axis[1]), abs(best_pinch_axis[0]))
        self.assertEqual(candidates[0], ranking[0]["grasp_yaw_rad"])

    def test_vertical_pinch_axis_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            grasp_yaw_from_pinch_axis_world(
                np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
            )

    def test_development_compliant_contact_derivation(self) -> None:
        per_finger_force_n = 12.5
        intended_deflection_m = 0.0005
        stiffness_n_m = per_finger_force_n / intended_deflection_m
        self.assertEqual(stiffness_n_m, 25000.0)
        self.assertLess(intended_deflection_m, MAX_ALLOWED_PENETRATION_M)

    def test_quintic_time_scaling_has_smooth_endpoints(self) -> None:
        self.assertEqual(quintic_time_scaling(0.0), 0.0)
        self.assertEqual(quintic_time_scaling(1.0), 1.0)
        values = [quintic_time_scaling(index / 100) for index in range(101)]
        self.assertTrue(all(left <= right for left, right in zip(values, values[1:])))
        epsilon = 1.0e-4
        self.assertLess(quintic_time_scaling(epsilon) / epsilon, 1.0e-5)
        self.assertLess(
            (1.0 - quintic_time_scaling(1.0 - epsilon)) / epsilon,
            1.0e-5,
        )

    def test_cover_mass_properties_are_finite_and_physically_valid(self) -> None:
        self.assertGreater(MANIPULABLE_COVER_MASS_KG, 0.20)
        self.assertTrue(
            all(math.isfinite(value) for value in MANIPULABLE_COVER_INERTIA_KG_M2)
        )
        self.assertTrue(all(value > 0.0 for value in MANIPULABLE_COVER_INERTIA_KG_M2))
        ixx, iyy, izz = MANIPULABLE_COVER_INERTIA_KG_M2
        self.assertLess(ixx, iyy + izz)
        self.assertLess(iyy, ixx + izz)
        self.assertLess(izz, ixx + iyy)
        self.assertGreater(MANIPULABLE_COVER_CENTER_OF_MASS_LOCAL_M[2], 0.16)
        total_volume = math.prod(CALIBRATION_COVER_FULL_EXTENTS_M) + math.prod(
            MANIPULABLE_COVER_HANDLE_FULL_EXTENTS_M
        )
        effective_density = MANIPULABLE_COVER_MASS_KG / total_volume
        self.assertGreater(effective_density, 250.0)

    def test_full_micro_lift_uses_exact_grasp_fk(self) -> None:
        source = (ROOT / "scripts" / "persistent_composite_grasp.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("compute_forward_kinematics", source)
        self.assertIn("rot_matrix_to_quat", source)
        self.assertIn('"exact_grasp_fk_position_world_m"', source)
        self.assertIn('"micro_lift_world_displacement_m"', source)
        self.assertIn("validate_lift_ik_pose", source)
        self.assertIn('"micro_lift_ik_cartesian_validation"', source)

    def test_lift_ik_tolerances_limit_long_tool_lateral_swing(self) -> None:
        self.assertLessEqual(LIFT_IK_POSITION_TOLERANCE_M, 0.0002)
        self.assertLessEqual(LIFT_IK_ORIENTATION_TOLERANCE_RAD, 0.002)
        self.assertLessEqual(MAX_LIFT_IK_POSITION_ERROR_M, 0.001)
        self.assertLessEqual(MAX_LIFT_IK_ORIENTATION_ERROR_RAD, 0.005)
        self.assertEqual(MICRO_LIFT_CARTESIAN_WAYPOINTS, 5)
        self.assertEqual(FULL_LIFT_CARTESIAN_WAYPOINTS, 18)
        self.assertLessEqual(MAX_LIFT_IK_JOINT_STEP_RAD, 0.15)
        self.assertGreaterEqual(TRANSFER_TRAJECTORY_STEPS, 240)
        source = (ROOT / "scripts" / "persistent_composite_grasp.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("solve_vertical_lift_path", source)
        self.assertIn("kinematics._kinematics.jacobian", source)
        self.assertIn("lula_analytic_jacobian_damped_local_ik", source)
        self.assertIn('"micro_lift_cartesian_waypoints"', source)
        self.assertIn('"full_lift_cartesian_waypoints"', source)
        self.assertIn("settle_steps=0", source)

    def test_lab_cover_mass_properties_are_recomputed_not_reused(self) -> None:
        center, inertia = _composite_cover_mass_properties(
            mass_kg=0.90,
            plate_full_extents_m=(0.40, 0.30, 0.02),
            plate_center_local_m=(0.0, 0.0, 0.17),
            handle_full_extents_m=(0.10, 0.025, 0.04),
            handle_center_local_m=(0.0, 0.0, 0.205),
        )
        self.assertGreater(center[2], 0.17)
        self.assertLess(center[2], 0.205)
        self.assertTrue(all(value > 0.0 for value in inertia))
        self.assertNotEqual(tuple(inertia), MANIPULABLE_COVER_INERTIA_KG_M2)

    def test_rotation_angle(self) -> None:
        self.assertAlmostEqual(rotation_angle_rad(np.eye(3)), 0.0)
        angle = math.radians(7.5)
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self.assertAlmostEqual(rotation_angle_rad(rotation), angle)
        vector = rotation_vector_from_matrix(rotation)
        np.testing.assert_allclose(vector, [0.0, 0.0, angle], atol=1.0e-12)

    def test_provisional_lid_grip_has_force_margin(self) -> None:
        # 25 N is the RG6 datasheet minimum adjustable force. Until the real
        # tool is calibrated, use it as a combined proxy plus an independent
        # bilateral floor with frictional holding margin above 1.7x.
        self.assertEqual(PROVISIONAL_LID_COMBINED_GRIP_FORCE_N, 25.0)
        self.assertGreaterEqual(
            PROVISIONAL_LID_COMBINED_GRIP_FORCE_N,
            2.0 * PROVISIONAL_LID_GRIP_FORCE_PER_FINGER_N,
        )
        self.assertLess(
            PROVISIONAL_LID_GRIP_FORCE_PER_FINGER_N,
            MAX_ALLOWED_CONTACT_FORCE_N,
        )
        self.assertGreater(
            PROVISIONAL_EPDM_LID_STATIC_FRICTION,
            PROVISIONAL_EPDM_LID_DYNAMIC_FRICTION,
        )
        holding_force_n = (
            2.0
            * PROVISIONAL_EPDM_LID_DYNAMIC_FRICTION
            * PROVISIONAL_LID_GRIP_FORCE_PER_FINGER_N
        )
        cover_weight_n = MANIPULABLE_COVER_MASS_KG * 9.81
        self.assertGreater(holding_force_n / cover_weight_n, 1.7)

    def test_micro_lift_gate_is_stricter_than_requested_motion(self) -> None:
        self.assertEqual(MICRO_LIFT_HEIGHT_M, 0.010)
        self.assertLess(
            MAX_MICRO_LIFT_RELATIVE_TRANSLATION_M,
            MINIMUM_MICRO_LIFT_DELTA_M,
        )
        self.assertLess(MINIMUM_MICRO_LIFT_DELTA_M, MICRO_LIFT_HEIGHT_M)

    def test_contact_force_limits_remain_hard_safety_gates(self) -> None:
        self.assertLessEqual(MAX_ALLOWED_CONTACT_FORCE_N, 60.0)
        self.assertLessEqual(MAX_ALLOWED_PENETRATION_M, 0.003)
        self.assertGreater(RG6_HANDLE_CLOSE_MASTER_RAD, 0.45)
        self.assertLess(RG6_HANDLE_CLOSE_MASTER_RAD, math.radians(36.0))
        self.assertLessEqual(GRIP_FORCE_TARGET_INCREMENT_RAD, 0.00035)

    def test_supported_root_height_commands_only_shallow_contact(self) -> None:
        root_z = supported_root_height_m(
            support_top_z_m=0.74,
            target_bottom_offset_from_root_m=0.159,
        )
        self.assertAlmostEqual(
            root_z,
            0.74 - 0.159 - PLACEMENT_SUPPORT_PENETRATION_M,
        )
        self.assertLessEqual(PLACEMENT_SUPPORT_PENETRATION_M, 0.001)
        self.assertLessEqual(PLACEMENT_MAX_SUPPORT_PENETRATION_M, 0.002)
        with self.assertRaises(ValueError):
            supported_root_height_m(
                support_top_z_m=0.74,
                target_bottom_offset_from_root_m=0.159,
                commanded_support_penetration_m=0.002,
            )

    def test_cover_release_has_dense_place_and_retreat_contract(self) -> None:
        self.assertGreaterEqual(PLACEMENT_CARTESIAN_WAYPOINTS, 30)
        self.assertGreaterEqual(PLACEMENT_TRAJECTORY_STEPS, 60)
        self.assertGreaterEqual(RETREAT_CARTESIAN_WAYPOINTS, 12)
        self.assertGreaterEqual(RETREAT_DISTANCE_M, 0.12)
        executor = (
            ROOT / "scripts" / "persistent_composite_grasp.py"
        ).read_text(encoding="utf-8")
        launcher = (ROOT / "scripts" / "isaac_sim_server.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("persistent_supported_placement_verification", executor)
        self.assertIn("declared_support_contact_not_reached", executor)
        self.assertIn("persistent_supported_release", executor)
        self.assertIn("persistent_post_release_retreat", executor)
        self.assertIn('"cover_placed_and_released"', executor)
        self.assertIn('placement_support_path="/World/WorkMat"', launcher)
        self.assertIn('if placement_support_path == "/World/WorkMat"', executor)
        self.assertIn('("/World/WorkBench/Top",)', executor)
        self.assertIn("secondary_placement_support_contact_pairs", executor)
        self.assertIn("release_after_placement=True", launcher)

    def test_planar_support_allows_bounded_overhang_with_center_support(self) -> None:
        metrics = planar_support_metrics(
            support_min_xy=np.asarray([0.145, -0.63]),
            support_max_xy=np.asarray([1.195, 0.03]),
            target_min_xy=np.asarray([0.083, -0.585]),
            target_max_xy=np.asarray([0.437, -0.255]),
        )
        self.assertGreaterEqual(
            metrics["center_margin_m"], MINIMUM_SUPPORT_CENTER_MARGIN_M
        )
        self.assertGreaterEqual(
            metrics["overlap_fraction"], MINIMUM_SUPPORT_OVERLAP_FRACTION
        )
        self.assertLess(metrics["full_footprint_margin_m"], 0.0)


if __name__ == "__main__":
    unittest.main()
