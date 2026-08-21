"""Tests for the unified semantic-belief task planner."""

import json
import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from unified_task_belief_planner import (  # noqa: E402
    plan,
    update_belief_and_task_state,
    update_semantic_belief,
    validate_model,
    validate_unified_method_contract,
)


CONFIG = ROOT / "configs/research/task_belief_validation.json"


class UnifiedTaskBeliefPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = json.loads(CONFIG.read_text(encoding="utf-8"))
        validate_model(self.model)

    def peaked(self, hypothesis: str) -> dict[str, float]:
        residual = 0.02 / (len(self.model["semantic_hypotheses"]) - 1)
        return {
            value: 0.98 if value == hypothesis else residual
            for value in self.model["semantic_hypotheses"]
        }

    def test_outside_target_can_be_grasped_without_forced_cover_removal(self) -> None:
        policy = plan(
            self.peaked("track_center_selected|outside"),
            "covered",
            self.model,
            horizon=2,
        )
        self.assertEqual(
            policy["selected_action"],
            "grasp:track_center_selected:outside",
        )
        self.assertFalse(policy["fixed_confidence_threshold_used"])

    def test_inside_target_requires_interaction_or_observation_when_covered(self) -> None:
        policy = plan(
            self.peaked("track_center_selected|inside"),
            "covered",
            self.model,
            horizon=3,
        )
        self.assertIn(
            policy["selected_action"],
            {"remove_cover", "viewpoint_close_high", "viewpoint_right"},
        )
        self.assertNotEqual(
            policy["selected_action"], "grasp:track_center_selected:inside"
        )

    def test_empty_container_is_negative_evidence_for_inside(self) -> None:
        action = self.model["information_actions"]["remove_cover"]
        posterior = update_semantic_belief(
            self.model["initial_semantic_belief"],
            "covered",
            action,
            "opened_empty",
        )
        prior_inside = sum(
            value
            for key, value in self.model["initial_semantic_belief"].items()
            if key.endswith("|inside")
        )
        posterior_inside = sum(
            value for key, value in posterior.items() if key.endswith("|inside")
        )
        self.assertLess(posterior_inside, prior_inside)
        self.assertGreater(
            posterior["target_absent|not_applicable"],
            self.model["initial_semantic_belief"]["target_absent|not_applicable"],
        )

    def test_information_and_terminal_actions_share_one_value_table(self) -> None:
        policy = plan(
            self.model["initial_semantic_belief"],
            self.model["initial_task_state"],
            self.model,
        )
        kinds = {value["kind"] for value in policy["action_values"]}
        self.assertIn("interaction_observation", kinds)
        self.assertIn("observation", kinds)
        self.assertIn("terminal_grasp", kinds)
        self.assertIn("terminal_defer", kinds)

    def test_target_absent_has_no_grasp_action(self) -> None:
        policy = plan(
            self.peaked("target_absent|not_applicable"),
            "open",
            self.model,
            horizon=0,
        )
        self.assertEqual(policy["selected_action"], "defer")

    def test_observable_cover_state_cannot_be_hidden_in_semantic_belief(self) -> None:
        invalid = copy.deepcopy(self.model)
        invalid["semantic_hypotheses"][0] = "track_center_selected|inside|covered"
        old = "track_center_selected|inside"
        new = "track_center_selected|inside|covered"
        invalid["initial_semantic_belief"][new] = invalid[
            "initial_semantic_belief"
        ].pop(old)
        for action in invalid["information_actions"].values():
            for likelihood in action["observation_likelihood"].values():
                likelihood[new] = likelihood.pop(old)
        invalid["terminal_grasp_actions"][
            "grasp:track_center_selected:inside"
        ]["semantic_hypothesis"] = new
        with self.assertRaisesRegex(ValueError, "must not be encoded"):
            validate_model(invalid)

    def test_fixed_grasp_threshold_is_rejected(self) -> None:
        invalid = copy.deepcopy(self.model)
        invalid["minimum_grasp_success_probability"] = 0.9
        with self.assertRaisesRegex(ValueError, "Fixed confidence gates"):
            validate_model(invalid)

    def test_remove_cover_failure_keeps_task_state_covered(self) -> None:
        posterior, state = update_belief_and_task_state(
            self.model["initial_semantic_belief"],
            "covered",
            self.model["information_actions"]["remove_cover"],
            "removal_failed",
        )
        self.assertEqual(state, "covered")
        self.assertAlmostEqual(sum(posterior.values()), 1.0)

    def test_paper_contract_has_one_unified_action_set(self) -> None:
        validate_unified_method_contract(self.model)
        policy = plan(
            self.model["initial_semantic_belief"],
            "covered",
            self.model,
        )
        actions = {row["action"] for row in policy["action_values"]}
        self.assertTrue(
            {"remove_cover", "viewpoint_close_high", "viewpoint_right", "defer"}
            <= actions
        )
        self.assertTrue(any(action.startswith("grasp:") for action in actions))


if __name__ == "__main__":
    unittest.main()
