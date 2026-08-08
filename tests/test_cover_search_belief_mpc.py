"""CPU-only tests for removable-cover belief-tree MPC."""

import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cover_search_belief_mpc import (  # noqa: E402
    execute_observation_action,
    plan,
    predict_belief,
    run_scripted_episode,
    validate_config,
)


class CoverSearchBeliefMpcTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (
                ROOT
                / "configs"
                / "research"
                / "cover_search_belief_mpc_cpu_pilot.json"
            ).read_text(encoding="utf-8")
        )
        validate_config(cls.config)

    def test_remove_cover_transition_is_normalized(self) -> None:
        predicted = predict_belief(
            self.config["initial_belief"],
            "remove_cover",
            self.config,
        )
        self.assertTrue(math.isclose(sum(predicted.values()), 1.0))
        self.assertAlmostEqual(
            predicted["inside|open"]
            + predicted["outside_near|open"],
            0.97,
        )

    def test_empty_container_is_negative_evidence_for_inside(self) -> None:
        update = execute_observation_action(
            self.config["initial_belief"],
            "remove_cover",
            "empty_container",
            self.config,
        )
        self.assertTrue(update["negative_evidence_applied"])
        self.assertLess(
            update["location_after"]["inside"],
            update["location_before"]["inside"],
        )
        self.assertGreater(
            update["location_after"]["outside_near"],
            0.85,
        )

    def test_negative_evidence_ablation_preserves_location_prior(self) -> None:
        update = execute_observation_action(
            self.config["initial_belief"],
            "remove_cover",
            "empty_container",
            self.config,
            negative_evidence_enabled=False,
        )
        self.assertFalse(update["negative_evidence_applied"])
        self.assertAlmostEqual(
            update["location_after"]["inside"],
            update["location_before"]["inside"],
        )

    def test_root_policy_selects_remove_cover_without_future_oracle(self) -> None:
        result = plan(
            self.config["initial_belief"],
            self.config,
        )
        self.assertEqual(result["selected_action"], "remove_cover")
        self.assertFalse(
            result["future_observation_used_for_action_selection"]
        )
        selected = next(
            item
            for item in result["action_values"]
            if item["action"] == "remove_cover"
        )
        self.assertAlmostEqual(
            selected["branch_probability_sum"], 1.0
        )
        self.assertEqual(
            {
                branch["continuation_action"]
                for branch in selected["observation_branches"]
            },
            {"grasp_inside", "grasp_outside", "remove_cover"},
        )

    def test_scripted_positive_and_negative_episodes_replan(self) -> None:
        for episode in self.config["scripted_diagnostic_episodes"][:2]:
            result = run_scripted_episode(episode, self.config)
            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["terminal_matches_expected"])
            self.assertFalse(
                result[
                    "future_scripted_observations_used_during_planning"
                ]
            )
            self.assertFalse(result["true_state_used_during_planning"])

    def test_action_failure_causes_remove_cover_retry(self) -> None:
        episode = self.config["scripted_diagnostic_episodes"][2]
        result = run_scripted_episode(episode, self.config)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            [step["selected_action"] for step in result["steps"]],
            ["remove_cover", "remove_cover", "grasp_outside"],
        )

    def test_without_negative_evidence_needs_an_extra_view(self) -> None:
        episode = self.config["scripted_diagnostic_episodes"][1]
        result = run_scripted_episode(
            episode,
            self.config,
            negative_evidence_enabled=False,
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            [step["selected_action"] for step in result["steps"]],
            [
                "remove_cover",
                "viewpoint_close_high",
                "grasp_outside",
            ],
        )


if __name__ == "__main__":
    unittest.main()
