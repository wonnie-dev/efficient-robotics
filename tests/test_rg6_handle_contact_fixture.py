"""CPU contracts for the isolated RG6-handle contact fixture."""

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_rg6_handle_contact_fixture.py"


class RG6HandleContactFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.text)

    def test_fixture_uses_actual_imported_rg6_not_proxy_pads(self) -> None:
        self.assertIn("actual_imported_rg6_collision_used", self.text)
        self.assertNotIn("RG6-sized collision pad", self.text)
        self.assertIn("inner_finger_1", self.text)

    def test_fixture_has_no_attachment_or_pose_copying(self) -> None:
        self.assertIn('"object_attachment_used": False', self.text)
        self.assertIn('"target_pose_copying_used": False', self.text)

    def test_fixture_retains_strict_micro_lift_gates(self) -> None:
        self.assertIn('acceptance["requested_micro_lift_m"]', self.text)
        self.assertIn('acceptance["maximum_relative_translation_m"]', self.text)
        self.assertIn('acceptance["maximum_penetration_m"]', self.text)
        self.assertIn('acceptance["maximum_contact_force_per_finger_n"]', self.text)

    def test_micro_lift_uses_exact_fixture_fk_pose(self) -> None:
        self.assertIn("compute_forward_kinematics", self.text)
        self.assertIn("rot_matrix_to_quat", self.text)
        self.assertIn('acceptance["requested_micro_lift_m"]', self.text)

    def test_fixture_locks_only_lateral_lid_motion(self) -> None:
        self.assertIn("UsdPhysics.PrismaticJoint.Define", self.text)
        self.assertIn('CreateAxisAttr().Set("Z")', self.text)
        self.assertIn('"vertical_guide_free_axis": "world_z"', self.text)

    def test_force_ramp_ignores_speculative_contact_pairs(self) -> None:
        self.assertIn("geometric_handle_contact_master_rad", self.text)
        self.assertIn(
            '"force_control_trigger": "measured_geometric_contact_envelope"',
            self.text,
        )

    def test_fixture_uses_measured_jaw_center_alignment(self) -> None:
        self.assertIn("FIXTURE_HANDLE_ALIGNMENT_OFFSET_WORLD_M", self.text)
        self.assertIn("-0.000615", self.text)
        self.assertIn("handle_alignment_source", self.text)

    def test_fixture_enforces_one_visible_gpu_and_no_multi_gpu(self) -> None:
        self.assertIn('os.environ.get("CUDA_VISIBLE_DEVICES")', self.text)
        self.assertIn('"multi_gpu": False', self.text)
        self.assertIn('"max_gpu_count": 1', self.text)

    def test_coordinated_mode_removes_mimic_before_adding_drives(self) -> None:
        self.assertIn('choices=("passive_mimic", "coordinated_drives")', self.text)
        self.assertIn('joint.RemoveAPI("NewtonMimicAPI")', self.text)
        self.assertIn("UsdPhysics.DriveAPI.Apply", self.text)
        self.assertIn('"mimic_api_removal_in_memory"', self.text)

    def test_coordinated_mode_commands_all_six_rg6_joints(self) -> None:
        self.assertIn("RG6_NAMES = (MASTER_NAME, *FOLLOWER_RATIOS)", self.text)
        self.assertIn("def set_grip_targets", self.text)
        self.assertIn("dof_indices=rg6_indices", self.text)
        self.assertIn('"maximum_coupling_error_rad"', self.text)

    def test_coordinated_torque_is_a_shared_total_budget(self) -> None:
        self.assertIn("total_torque_nm / len(rg6_drives)", self.text)
        self.assertIn('"provisional_aggregate_drive_effort_nm"', self.text)
        self.assertIn(
            '"development_joint_drive_effort_not_rg6_motor_torque"',
            self.text,
        )
        self.assertIn("no greater than 18 Nm", self.text)
        self.assertIn('"transfer_ready": False', self.text)

    def test_force_gate_is_rechecked_after_contact_settling(self) -> None:
        self.assertIn("def bilateral_force_gate_ready", self.text)
        self.assertIn(
            "if bilateral_force_gate_ready():", self.text
        )
        self.assertIn('"force_settle_steps": force_settle_steps', self.text)
        self.assertIn(
            '"force_gate_uses_latest_active_contact": True', self.text
        )

    def test_loaded_lift_waits_for_actual_gripper_convergence(self) -> None:
        self.assertIn("def gripper_world_position", self.text)
        self.assertIn("for step in range(1, 301)", self.text)
        self.assertIn('"measured_gripper_lift_m"', self.text)
        self.assertIn('"loaded_target_convergence_steps"', self.text)
        self.assertIn('"stop_reason"', self.text)

    def test_fixture_source_parses(self) -> None:
        self.assertIsInstance(self.tree, ast.Module)


if __name__ == "__main__":
    unittest.main()
