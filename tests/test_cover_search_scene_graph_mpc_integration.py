"""CPU-only tests for Scene Graph and cover-search MPC integration."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cover_search_scene_graph_mpc_integration import (  # noqa: E402
    action_request,
    build_scene_graph,
    execute_contract_stub,
    graph_to_planner_belief,
    run_episode,
)
from run_cover_search_belief_mpc import plan  # noqa: E402
from validate_uncertainty_scene_graph import validate  # noqa: E402


class CoverSearchSceneGraphMpcIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.integration_config = json.loads(
            (
                ROOT
                / "configs"
                / "research"
                / "cover_search_scene_graph_mpc_integration.json"
            ).read_text(encoding="utf-8")
        )
        cls.planner_config = json.loads(
            (
                ROOT
                / cls.integration_config["planner_config"]
            ).read_text(encoding="utf-8")
        )

    def initial_graph(self) -> dict:
        return build_scene_graph(
            self.integration_config,
            self.planner_config,
            episode_id="unit_test",
            belief=self.planner_config["initial_belief"],
            sequence_index=0,
            observation_symbol="initial_center_symbolic",
        )

    def test_scene_graph_joint_belief_roundtrip(self) -> None:
        graph = self.initial_graph()
        validate(graph)
        restored = graph_to_planner_belief(graph)
        self.assertEqual(
            restored,
            self.planner_config["initial_belief"],
        )
        relation = graph["edges"][0]["belief"][
            "relation_distribution"
        ]
        self.assertAlmostEqual(relation["inside"], 0.65)
        cover = graph["nodes"][2]["belief"]["class_distribution"]
        self.assertAlmostEqual(cover["covered"], 1.0)

    def test_request_is_bound_to_graph_revision(self) -> None:
        graph = self.initial_graph()
        policy = plan(
            graph_to_planner_belief(graph),
            self.planner_config,
        )
        request = action_request(graph, policy, step_index=0)
        self.assertEqual(request["type"], "remove_cover")
        self.assertEqual(
            len(request["source_scene_graph_sha256"]), 64
        )
        self.assertFalse(
            request["future_observation_used_for_selection"]
        )

    def test_contract_result_requires_matching_action(self) -> None:
        graph = self.initial_graph()
        policy = plan(
            graph_to_planner_belief(graph),
            self.planner_config,
        )
        request = action_request(graph, policy, step_index=0)
        result = execute_contract_stub(
            request,
            expected_action="remove_cover",
            observation="empty_container",
        )
        self.assertEqual(result["request_id"], request["request_id"])
        self.assertTrue(result["result_arrived_after_request"])
        self.assertFalse(result["physical_execution"])
        with self.assertRaises(ValueError):
            execute_contract_stub(
                request,
                expected_action="viewpoint_right",
                observation="empty_container",
            )

    def test_all_diagnostic_episodes_complete_interface_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for episode in self.planner_config[
                "scripted_diagnostic_episodes"
            ]:
                result = run_episode(
                    episode,
                    self.integration_config,
                    self.planner_config,
                    Path(directory) / episode["episode_id"],
                )
                self.assertEqual(result["status"], "completed")
                self.assertTrue(result["terminal_matches_expected"])
                self.assertFalse(
                    result["future_observation_used_during_planning"]
                )
                self.assertFalse(result["true_state_used_for_control"])
                self.assertFalse(result["physical_execution"])
                for trace in result["trace"]:
                    self.assertFalse(trace["physical_execution"])


if __name__ == "__main__":
    unittest.main()
