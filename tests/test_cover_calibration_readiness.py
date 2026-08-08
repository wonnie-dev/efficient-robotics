"""Focused tests for independent covered-container calibration counting."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_cover_calibration_readiness import audit  # noqa: E402


class CoverCalibrationReadinessTests(unittest.TestCase):
    def test_repeated_successful_seed_counts_once(self) -> None:
        protocol = {
            "calibration_freeze_requirements": {
                "minimum_cover_negative_evidence_calibration_episodes": 2
            },
            "physical_success_gates": {
                "minimum_verified_lift_m": 0.15,
                "maximum_force_per_finger_n": 60.0,
                "maximum_penetration_m": 0.003,
                "maximum_target_relative_translation_m": 0.015,
                "maximum_target_relative_rotation_rad": 0.17453292519943295,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run in ("run001", "run002"):
                path = root / "negative" / "seed197" / run
                path.mkdir(parents=True)
                (path / "negative_evidence_live_result.json").write_text(
                    json.dumps(
                        {
                            "status": "completed",
                            "seed": 197,
                            "post_remove_observation": "empty_container",
                            "belief_update_count": 2,
                            "action_sequence": [
                                "remove_cover",
                                "viewpoint_right",
                                "grasp_outside",
                            ],
                            "cover_removal_execution": self.execution(
                                "/World/OpenContainer/CalibrationCover"
                            ),
                            "final_grasp_execution": self.execution(),
                        }
                    ),
                    encoding="utf-8",
                )
            result = audit(protocol, root)
        self.assertEqual(
            result["independent_negative_evidence_episode_count"], 1
        )
        self.assertEqual(
            result["status"],
            "blocked_insufficient_independent_negative_evidence",
        )

    @staticmethod
    def execution(target_path: str = "/World/TargetRed") -> dict:
        return {
            "status": "completed",
            "manipulation_target_path": target_path,
            "verified_lift_delta_m": 0.18,
            "bilateral_contact_before_lift": True,
            "contact_maintained_after_lift": True,
            "contact_maintained_before_release": True,
            "contact_force_n": {"maximum": {"left": 10.0, "right": 10.0}},
            "maximum_contact_penetration_m": {"left": 0.001, "right": 0.001},
            "target_gripper_relative_stability": {
                "maximum_translation_m": 0.001,
                "maximum_rotation_rad": 0.01,
            },
            "unexpected_environment_pairs": [],
            "unexpected_target_environment_pairs": [],
            "target_attachment_used": False,
            "target_pose_copying_used": False,
            "finite_final_joint_state": True,
        }


if __name__ == "__main__":
    unittest.main()
