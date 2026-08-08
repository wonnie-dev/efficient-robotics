"""CPU-only tests for the controlled-occlusion belief pilot."""

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_non_oracle_hybrid_planner import plan  # noqa: E402
from run_scanned_basket_occlusion_belief_pilot import (  # noqa: E402
    softmax,
    track_mapping,
)
from run_scanned_basket_two_step_belief_pilot import (  # noqa: E402
    planner_config,
)


class ScannedBasketOcclusionBeliefPilotTests(unittest.TestCase):
    def test_softmax_is_normalized(self) -> None:
        values = softmax([-2.0, 1.0], 4.0)
        self.assertAlmostEqual(sum(values), 1.0)
        self.assertGreater(values[1], values[0])

    def test_rgbd_tracking_handles_candidate_order_swap(self) -> None:
        mapping, metadata = track_mapping(
            {
                "candidate_001": [0.0, 0.0, 0.0],
                "candidate_002": [1.0, 0.0, 0.0],
            },
            {
                "candidate_001": [1.01, 0.0, 0.0],
                "candidate_002": [0.01, 0.0, 0.0],
            },
        )
        self.assertEqual(mapping["candidate_001"], "track_002")
        self.assertEqual(mapping["candidate_002"], "track_001")
        self.assertFalse(metadata["simulator_ground_truth_used"])

    def test_geometry_likelihood_selects_close_high(self) -> None:
        config = json.loads(
            (
                ROOT
                / "configs"
                / "research"
                / "scanned_basket_occlusion_belief_mpc_pilot.json"
            ).read_text(encoding="utf-8")
        )
        config["initial_belief"]["target"] = {
            "track_001": 0.3923368301671084,
            "track_002": 0.6076631698328917,
        }
        config["initial_belief"]["relation"] = {
            "inside": 0.45943658261592435,
            "outside": 0.3868843914808439,
            "unknown": 0.1536790259032317,
        }
        result = plan(config)
        self.assertEqual(
            result["action_request"]["type"], "viewpoint_close_high"
        )
        self.assertEqual(result["provenance"]["future_capture_files_read"], [])
        self.assertFalse(result["provenance"]["oracle"])

    def test_two_reobservations_reach_debug_grasp_request(self) -> None:
        path = (
            ROOT
            / "configs"
            / "research"
            / "scanned_basket_occlusion_belief_mpc_pilot.json"
        )
        base = json.loads(path.read_text(encoding="utf-8"))
        config = planner_config(
            base,
            {
                "target": {
                    "track_001": 0.1052105367030863,
                    "track_002": 0.8947894632969136,
                },
                "relation": {
                    "inside": 0.9509243220053256,
                    "outside": 0.0365844022704271,
                    "unknown": 0.012491275724247268,
                },
            },
            completed_reobservations=2,
            executed_actions=[
                "viewpoint_close_high",
                "viewpoint_right",
            ],
            perception_config_path=(
                ROOT
                / "configs"
                / "perception"
                / "scanned_basket_occlusion_two_step_seed000.json"
            ),
        )
        result = plan(config)
        self.assertEqual(result["action_request"]["type"], "grasp")
        self.assertTrue(result["commitment_gate"]["grasp_allowed"])


if __name__ == "__main__":
    unittest.main()
