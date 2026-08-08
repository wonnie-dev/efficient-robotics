"""CPU-only tests for pre-action future-belief planning."""

import json
import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_non_oracle_hybrid_planner import plan  # noqa: E402


class NonOracleHybridPlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = (
            ROOT
            / "configs"
            / "research"
            / "first_belief_mpc_integration.json"
        )
        cls.result = plan(json.loads(path.read_text(encoding="utf-8")))

    def test_planning_does_not_read_future_capture_files(self) -> None:
        provenance = self.result["provenance"]
        self.assertEqual(provenance["future_capture_files_read"], [])
        self.assertFalse(provenance["oracle"])

    def test_view_actions_predict_normalized_observation_branches(self) -> None:
        for action in ("viewpoint_left", "viewpoint_right"):
            branches = self.result["pre_action_forecasts"][action][
                "observation_branches"
            ]
            self.assertEqual(len(branches), 6)
            self.assertAlmostEqual(
                sum(branch["probability"] for branch in branches), 1.0
            )
            for branch in branches:
                self.assertAlmostEqual(
                    sum(branch["posterior"]["target"].values()), 1.0
                )
                self.assertAlmostEqual(
                    sum(branch["posterior"]["relation"].values()), 1.0
                )

    def test_candidate_actions_have_different_future_belief_metrics(self) -> None:
        forecasts = self.result["pre_action_forecasts"]
        self.assertNotEqual(
            forecasts["viewpoint_left"]["expected_task_failure_risk"],
            forecasts["viewpoint_right"]["expected_task_failure_risk"],
        )
        self.assertGreater(
            forecasts["viewpoint_right"]["expected_information_gain_nats"],
            0.0,
        )
        self.assertEqual(
            forecasts["grasp"]["observation_branches"],
            [],
        )

    def test_horizon_two_compares_view_then_grasp_with_immediate_grasp(self) -> None:
        sequences = {
            tuple(candidate["sequence"])
            for candidate in self.result["candidate_sequences"]
        }
        self.assertNotIn(("grasp",), sequences)
        self.assertNotIn(("viewpoint_left", "grasp"), sequences)
        self.assertIn(("viewpoint_right", "grasp"), sequences)
        self.assertEqual(len(self.result["selected_sequence"]), 2)

    def test_predicted_empty_view_is_infeasible(self) -> None:
        feasibility = self.result["action_feasibility"]["viewpoint_left"]
        self.assertFalse(feasibility["feasible"])
        self.assertEqual(
            feasibility["blocking_reasons"],
            ["predicted_observation_not_usable"],
        )

    def test_unsafe_grasp_is_blocked_by_commitment_gate(self) -> None:
        gate = self.result["commitment_gate"]
        self.assertFalse(gate["grasp_allowed"])
        self.assertIn(
            "task_failure_risk_above_threshold", gate["blocking_reasons"]
        )
        self.assertIn(
            "insufficient_completed_reobservations", gate["blocking_reasons"]
        )

    def test_no_safe_sensing_action_returns_defer(self) -> None:
        path = (
            ROOT
            / "configs"
            / "research"
            / "first_belief_mpc_integration.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        config = copy.deepcopy(config)
        config["completed_reobservations"] = 2
        config["initial_belief"]["target"] = {
            "target_red": 0.41,
            "rear_red_candidate": 0.59,
        }
        config["initial_belief"]["relation"] = {
            "inside": 0.89,
            "behind": 0.04,
            "unknown": 0.07,
        }
        for name, action in config["actions"].items():
            if name.startswith("viewpoint_"):
                action["enabled"] = False
        result = plan(config)
        self.assertEqual(result["action_request"]["type"], "defer")
        self.assertEqual(result["candidate_sequences"], [])
        self.assertFalse(result["commitment_gate"]["grasp_allowed"])

    def test_belief_tree_mpc_has_no_forced_reobservation_count(self) -> None:
        path = (
            ROOT
            / "configs"
            / "research"
            / "scanned_basket_belief_tree_mpc_pilot.json"
        )
        config = json.loads(path.read_text(encoding="utf-8"))
        config["initial_belief"]["target"] = {
            "track_001": 0.9497,
            "track_002": 0.0503,
        }
        config["initial_belief"]["relation"] = {
            "inside": 0.9770,
            "outside": 0.0120,
            "unknown": 0.0110,
        }
        result = plan(config)
        self.assertTrue(result["provenance"]["actual_mpc_solver"])
        self.assertEqual(
            result["planner"],
            "discrete_belief_tree_receding_horizon_mpc",
        )
        self.assertEqual(
            result["commitment_gate"][
                "minimum_completed_reobservations"
            ],
            0,
        )
        self.assertNotIn(
            "insufficient_completed_reobservations",
            result["commitment_gate"]["blocking_reasons"],
        )
        self.assertIn(
            result["action_request"]["type"],
            {"viewpoint_right", "viewpoint_close_high"},
        )
        continuations = {
            branch["continuation_action"]
            for branch in result["belief_tree_policy"]["selected"][
                "observation_branches"
            ]
        }
        self.assertEqual(continuations, {"grasp", "defer"})


if __name__ == "__main__":
    unittest.main()
