"""CPU contracts for the provisional symmetric RG6 drive smoke."""

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "smoke_rg6_coordinated_drives.py"


class RG6CoordinatedDriveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.text)

    def test_source_parses(self) -> None:
        self.assertIsInstance(self.tree, ast.Module)

    def test_mimic_is_removed_before_coordinated_drives(self) -> None:
        self.assertIn('joint.RemoveAPI("NewtonMimicAPI")', self.text)
        self.assertIn(
            '"coordinated_six_joint_drives_no_mimic"', self.text
        )

    def test_total_torque_budget_is_divided(self) -> None:
        self.assertIn(
            "PROVISIONAL_TOTAL_DRIVE_TORQUE_BUDGET_NM / len(RG6_NAMES)",
            self.text,
        )

    def test_open_close_reopen_and_stability_gates_exist(self) -> None:
        self.assertIn('(\"close\", 0.45)', self.text)
        self.assertIn('(\"reopen\", -0.20)', self.text)
        self.assertIn("maximum_tracking_error <= 0.05", self.text)
        self.assertIn("maximum_coupling_error <= 0.05", self.text)

    def test_single_gpu_and_non_final_contract(self) -> None:
        self.assertIn('os.environ.get("CUDA_VISIBLE_DEVICES")', self.text)
        self.assertIn('"multi_gpu": False', self.text)
        self.assertIn('"transfer_ready": False', self.text)
        self.assertIn('"valid_for_final_evaluation": False', self.text)


if __name__ == "__main__":
    unittest.main()
