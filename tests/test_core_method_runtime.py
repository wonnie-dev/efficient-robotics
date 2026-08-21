"""CPU tests for the calibrated runtime adapters."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from core_method_runtime import (  # noqa: E402
    calibrated_commitment_gate,
    calibrated_target_identity,
    expected_cost_commitment_decision,
    joint_hypothesis_probability,
    joint_scene_graph_snapshot,
    update_joint_hypothesis_belief,
    relation_observation_from_audit,
    next_observation_after_rejected_commitment,
    scene_graph_snapshot,
)


class CoreMethodRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.perception = {
            "ranking": {
                "selected_candidate_id": "candidate_001",
                "candidate_ids": ["candidate_001", "candidate_002"],
                "raw_match_logits": [18.125, -11.125],
            }
        }
        self.posterior = {
            "inside|covered": 0.0,
            "inside|open": 0.023383451711096764,
            "outside_near|covered": 0.0,
            "outside_near|open": 0.9766165482889032,
        }

    def test_target_identity_uses_frozen_temperature(self) -> None:
        identity = calibrated_target_identity(self.perception, 5.825)
        self.assertTrue(identity["calibrated"])
        self.assertGreater(identity["probability"], 0.95)
        self.assertLess(identity["probability"], 0.97)

    def test_rgbd_relation_drives_planner_observation(self) -> None:
        audit = {
            "qwen_relation_top_label": "outside",
            "rgbd_relation": {
                "membership_world_evidence": {"label": "outside"}
            },
        }
        evidence = relation_observation_from_audit(audit)
        self.assertEqual(evidence["planner_observation"], "outside_evidence")
        self.assertTrue(evidence["qwen_rgbd_agree"])

    def test_disagreement_is_recorded_without_changing_rgbd_symbol(self) -> None:
        audit = {
            "qwen_relation_top_label": "inside",
            "rgbd_relation": {
                "membership_world_evidence": {"label": "outside"}
            },
        }
        evidence = relation_observation_from_audit(audit)
        self.assertEqual(evidence["planner_observation"], "outside_evidence")
        self.assertFalse(evidence["qwen_rgbd_agree"])

    def test_joint_commitment_gate_passes_calibrated_example(self) -> None:
        identity = calibrated_target_identity(self.perception, 5.825)
        gate = calibrated_commitment_gate(
            terminal_action="grasp_outside",
            posterior=self.posterior,
            identity=identity,
            minimum_probability=0.9,
        )
        self.assertTrue(gate["authorized"])
        self.assertGreater(gate["joint_commitment_probability"], 0.9)
        self.assertFalse(gate["joint_probability_calibrated"])
        self.assertEqual(gate["protocol_status"], "legacy_v11_reproduction_only")

    def test_direct_joint_belief_rejects_unnormalized_scores(self) -> None:
        with self.assertRaises(ValueError):
            joint_hypothesis_probability(
                {"candidate_001|inside": 0.8, "candidate_002|outside": 0.4},
                candidate_id="candidate_001",
                membership="inside",
            )

    def test_joint_bayes_update_does_not_multiply_marginal_scores(self) -> None:
        posterior = update_joint_hypothesis_belief(
            {
                "candidate_001|inside": 0.25,
                "candidate_001|outside": 0.25,
                "candidate_002|inside": 0.25,
                "candidate_002|outside": 0.25,
            },
            {
                "candidate_001|inside": 0.80,
                "candidate_001|outside": 0.10,
                "candidate_002|inside": 0.05,
                "candidate_002|outside": 0.05,
            },
        )
        self.assertAlmostEqual(sum(posterior.values()), 1.0)
        self.assertEqual(
            max(posterior, key=posterior.get),
            "candidate_001|inside",
        )
        self.assertAlmostEqual(posterior["candidate_001|inside"], 0.8)

    def test_expected_cost_reobserves_when_commitment_risk_is_high(self) -> None:
        decision = expected_cost_commitment_decision(
            terminal_action="grasp_inside",
            candidate_id="candidate_001",
            membership="inside",
            joint_belief={
                "candidate_001|inside": 0.70,
                "candidate_001|outside": 0.10,
                "candidate_002|inside": 0.10,
                "candidate_002|outside": 0.10,
            },
            conditional_execution_success_probability=0.95,
            costs={
                "grasp": 0.12,
                "wrong_commitment": 1.0,
                "execution_failure": 0.5,
            },
            alternative_action_values=[
                {"action": "viewpoint_right", "expected_cost": 0.20},
                {"action": "defer", "expected_cost": 1.0},
            ],
        )
        self.assertFalse(decision["authorized"])
        self.assertEqual(decision["selected_action"], "viewpoint_right")
        self.assertFalse(decision["marginal_product_used"])

    def test_joint_scene_graph_requires_persistent_tracks(self) -> None:
        belief = {
            "track_001|inside": 0.7,
            "track_001|outside": 0.1,
            "track_002|inside": 0.1,
            "track_002|outside": 0.1,
        }
        graph = joint_scene_graph_snapshot(
            step=1,
            view="right",
            joint_belief=belief,
            candidate_track_ids=["track_001", "track_002"],
            observation={"source": "learned_rgbd"},
        )
        self.assertAlmostEqual(
            graph["candidate_target_marginals"]["track_001"], 0.8
        )
        self.assertFalse(graph["marginal_confidence_product_used"])
        with self.assertRaises(ValueError):
            joint_scene_graph_snapshot(
                step=1,
                view="right",
                joint_belief={
                    "candidate_001|inside": 0.5,
                    "candidate_001|outside": 0.5,
                },
                candidate_track_ids=["candidate_001"],
                observation={},
            )

    def test_expected_cost_grasps_when_it_is_the_best_action(self) -> None:
        decision = expected_cost_commitment_decision(
            terminal_action="grasp_inside",
            candidate_id="candidate_001",
            membership="inside",
            joint_belief={
                "candidate_001|inside": 0.96,
                "candidate_001|outside": 0.01,
                "candidate_002|inside": 0.02,
                "candidate_002|outside": 0.01,
            },
            conditional_execution_success_probability=0.98,
            costs={
                "grasp": 0.12,
                "wrong_commitment": 1.0,
                "execution_failure": 0.5,
            },
            alternative_action_values=[
                {"action": "viewpoint_right", "expected_cost": 0.25},
                {"action": "defer", "expected_cost": 1.0},
            ],
        )
        self.assertTrue(decision["authorized"])
        self.assertEqual(decision["selected_action"], "grasp_inside")

    def test_optional_risk_cap_is_explicit_not_a_confidence_constant(self) -> None:
        decision = expected_cost_commitment_decision(
            terminal_action="grasp_inside",
            candidate_id="candidate_001",
            membership="inside",
            joint_belief={
                "candidate_001|inside": 0.94,
                "candidate_001|outside": 0.02,
                "candidate_002|inside": 0.02,
                "candidate_002|outside": 0.02,
            },
            conditional_execution_success_probability=1.0,
            costs={
                "grasp": 0.01,
                "wrong_commitment": 1.0,
                "execution_failure": 1.0,
            },
            alternative_action_values=[
                {"action": "defer", "expected_cost": 0.5},
            ],
            maximum_wrong_commitment_risk=0.05,
        )
        self.assertFalse(decision["authorized"])
        self.assertEqual(decision["reason"], "wrong_commitment_risk_cap_exceeded")

    def test_scene_graph_contains_probabilistic_relation_edge(self) -> None:
        identity = calibrated_target_identity(self.perception, 5.825)
        graph = scene_graph_snapshot(
            step=2,
            view="right",
            identity=identity,
            belief=self.posterior,
            relation_evidence={"planner_observation": "outside_evidence"},
        )
        probabilities = graph["relation_edges"][0]["probabilities"]
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)
        self.assertGreater(probabilities["outside"], 0.97)

    def test_rejected_grasp_selects_lowest_cost_observation(self) -> None:
        policy = {
            "planner": "test_planner",
            "horizon": 3,
            "action_values": [
                {"action": "defer", "kind": "terminal", "cost": 0.8},
                {"action": "grasp_inside", "kind": "terminal", "cost": 0.17},
                {"action": "viewpoint_right", "kind": "observation", "cost": 0.28},
            ],
        }
        decision = next_observation_after_rejected_commitment(policy)
        self.assertEqual(decision["selected_action"], "viewpoint_right")
        self.assertFalse(decision["future_observation_used_for_selection"])


if __name__ == "__main__":
    unittest.main()
