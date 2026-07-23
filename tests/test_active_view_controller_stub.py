"""Tests for the provisional active-view controller utilities."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_active_view_controller_stub import gate_passed, mean_joint_motion  # noqa: E402


class ActiveViewControllerStubTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {
            "required_relation": "inside",
            "temporary_execution_gate": {
                "target_probability_minimum": 0.9,
                "relation_probability_minimum": 0.9,
            },
        }

    def test_gate_requires_both_beliefs(self) -> None:
        self.assertFalse(
            gate_passed(
                {"target": 0.95, "other": 0.05},
                {"inside": 0.8, "outside": 0.2},
                self.config,
            )
        )

    def test_gate_passes_when_both_are_sufficient(self) -> None:
        self.assertTrue(
            gate_passed(
                {"target": 0.95, "other": 0.05},
                {"inside": 0.92, "outside": 0.08},
                self.config,
            )
        )

    def test_joint_motion_is_mean_absolute_difference(self) -> None:
        poses = {"poses_rad": {"a": [0.0, 1.0], "b": [0.2, 1.4]}}
        self.assertAlmostEqual(mean_joint_motion("a", "b", poses), 0.3)


if __name__ == "__main__":
    unittest.main()
