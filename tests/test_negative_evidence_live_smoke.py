"""CPU contract tests for the live negative-evidence episode."""

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cover_search_belief_mpc import (  # noqa: E402
    execute_observation_action,
    normalize,
    plan,
)
from run_negative_evidence_live_smoke import (  # noqa: E402
    automatic_post_remove_outcome,
    effective_planner,
    learned_post_remove_outcome,
)


class NegativeEvidenceLiveSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (
                ROOT
                / "configs"
                / "research"
                / "negative_evidence_live_development.json"
            ).read_text(encoding="utf-8")
        )
        cls.planner = effective_planner(cls.config)

    def test_two_updates_produce_required_action_sequence(self) -> None:
        belief = normalize(self.planner["initial_belief"])
        root = plan(belief, self.planner)
        self.assertEqual(root["selected_action"], "remove_cover")
        after_empty = execute_observation_action(
            belief,
            "remove_cover",
            "empty_container",
            self.planner,
        )
        first_replan = plan(after_empty["posterior"], self.planner)
        self.assertEqual(
            first_replan["selected_action"], "viewpoint_right"
        )
        after_right = execute_observation_action(
            after_empty["posterior"],
            "viewpoint_right",
            "outside_evidence",
            self.planner,
        )
        second_replan = plan(after_right["posterior"], self.planner)
        self.assertEqual(
            second_replan["selected_action"], "grasp_outside"
        )
        self.assertGreater(
            after_right["posterior"]["outside_near|open"], 0.99
        )

    def test_reserved_test_seed_is_not_used(self) -> None:
        self.assertNotIn(int(self.config["seed"]), set(range(200, 210)))
        self.assertFalse(self.config["reserved_test_seeds_used"])
        self.assertFalse(self.config["valid_for_final_evaluation"])

    def test_instruction_does_not_leak_membership_relation(self) -> None:
        task = self.config["perception_task_overrides"]
        instruction = task["instruction"].lower()
        direct_prompt = task["qwen_direct_prompt"].lower()
        for leaked_relation in ("inside", "outside", "behind", "near"):
            self.assertNotIn(leaked_relation, instruction)
            self.assertNotIn(leaked_relation, direct_prompt)

    def test_visible_target_outside_container_is_negative_evidence(self) -> None:
        scene = {
            "calibration_ground_truth": {
                "world_ground_truth": {"membership": "outside"}
            }
        }
        self.assertEqual(
            automatic_post_remove_outcome(
                scene,
                target_visible_pixels=7769,
                minimum_target_pixels=100,
            ),
            "empty_container",
        )

    def test_inside_target_still_requires_post_action_visibility(self) -> None:
        scene = {
            "calibration_ground_truth": {
                "world_ground_truth": {"membership": "inside"}
            }
        }
        self.assertEqual(
            automatic_post_remove_outcome(
                scene,
                target_visible_pixels=0,
                minimum_target_pixels=100,
            ),
            "empty_container",
        )
        self.assertEqual(
            automatic_post_remove_outcome(
                scene,
                target_visible_pixels=101,
                minimum_target_pixels=100,
            ),
            "target_detected",
        )

    def test_learned_outside_agreement_is_empty_container(self) -> None:
        audit = {
            "qwen_relation_top_label": "outside",
            "rgbd_relation": {
                "membership_world_evidence": {"label": "outside"}
            },
        }
        self.assertEqual(
            learned_post_remove_outcome(audit), "empty_container"
        )

    def test_learned_relation_disagreement_abstains(self) -> None:
        audit = {
            "qwen_relation_top_label": "outside",
            "rgbd_relation": {
                "membership_world_evidence": {"label": "unknown"}
            },
        }
        with self.assertRaises(RuntimeError):
            learned_post_remove_outcome(audit)


if __name__ == "__main__":
    unittest.main()
