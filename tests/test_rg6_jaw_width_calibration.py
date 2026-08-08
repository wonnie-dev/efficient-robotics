"""CPU contracts for the imported RG6 jaw-width calibration."""

import ast
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_rg6_jaw_width_calibration.py"


class RG6JawWidthCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.text = SCRIPT.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.text)

    def test_source_parses(self) -> None:
        self.assertIsInstance(self.tree, ast.Module)

    def test_actual_collision_and_mimic_are_required(self) -> None:
        self.assertIn("actual_imported_rg6_collision_used", self.text)
        self.assertIn("NewtonMimicAPI", self.text)
        self.assertIn("inner_finger_1", self.text)

    def test_no_attachment_or_pose_copying(self) -> None:
        self.assertIn('"object_attachment_used": False', self.text)
        self.assertIn('"target_pose_copying_used": False', self.text)

    def test_single_gpu_contract(self) -> None:
        self.assertIn('os.environ.get("CUDA_VISIBLE_DEVICES")', self.text)
        self.assertIn('"multi_gpu": False', self.text)
        self.assertIn('"max_gpu_count": 1', self.text)

    def test_projection_gap_for_two_boxes(self) -> None:
        namespace = {"np": np}
        function = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "projected_surface_gap"
        )
        module = ast.Module(body=[function], type_ignores=[])
        exec(compile(module, str(SCRIPT), "exec"), namespace)
        left = np.asarray(
            [[x, y, z] for x in (-0.01, 0.01) for y in (-0.04, -0.03) for z in (0.0, 0.02)]
        )
        right = np.asarray(
            [[x, y, z] for x in (-0.01, 0.01) for y in (0.03, 0.04) for z in (0.0, 0.02)]
        )
        gap, axis, _, _ = namespace["projected_surface_gap"](left, right)
        self.assertAlmostEqual(gap, 0.06)
        self.assertGreater(axis[1], 0.99)


if __name__ == "__main__":
    unittest.main()
