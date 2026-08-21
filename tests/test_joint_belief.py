"""Tests for the direct joint-belief development evaluator."""

import sys
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_joint_belief import (  # noqa: E402
    HYPOTHESES,
    fit_joint_model,
    observation_symbol,
    state_for_episode,
    terminal_values,
)
from run_closed_loop_episode import (  # noqa: E402
    localization_is_plausible,
    posthoc_semantic_audit,
)
from joint_belief_runtime import fuse_static_track_localizations  # noqa: E402
from task_belief_runtime import (  # noqa: E402
    condition_joint_model_on_view_resolvability,
    predict_view_mode,
    select_scene_conditioned_view,
)


def episode(correct: bool, membership: str, future_agreement: str) -> dict:
    center = {
        "action": "initial_observation",
        "selected_target_correct_posthoc": correct,
        "world_membership_state_posthoc": membership,
        "identity_bin": "high" if correct else "low",
        "membership_observation": membership,
    }
    future = {
        "action": "viewpoint_right",
        "membership_observation": membership,
        "track_agreement_observation": future_agreement,
        "center_track_confidence_bin": "high" if correct else "low",
    }
    return {"initial_observation": center, "viewpoint_right": future}


class JointBeliefTests(unittest.TestCase):
    def test_static_track_fusion_rejects_one_view_center_outlier(self) -> None:
        estimates = [
            {"center_world_m": [0.9390, 0.0093, 0.062]},
            {"center_world_m": [0.9395, 0.0093, 0.046]},
            {"center_world_m": [0.9075, 0.0001, 0.032]},
        ]
        fused = fuse_static_track_localizations(estimates)
        self.assertEqual(fused["center_world_m"], [0.9390, 0.0093, 0.032])
        self.assertEqual(
            fused["multi_view_center_fusion"]["method"],
            "horizontal_coordinate_median_latest_height",
        )

    def test_static_track_fusion_keeps_latest_with_two_views(self) -> None:
        estimates = [
            {"center_world_m": [0.90, 0.00, 0.04]},
            {"center_world_m": [0.91, 0.01, 0.05]},
        ]
        fused = fuse_static_track_localizations(estimates)
        self.assertEqual(fused["center_world_m"], [0.91, 0.01, 0.05])

    def test_large_merged_mask_localization_is_rejected(self) -> None:
        self.assertTrue(
            localization_is_plausible(
                {"robust_extent_m": [0.11, 0.07, 0.10]}, 0.18
            )
        )
        self.assertFalse(
            localization_is_plausible(
                {"robust_extent_m": [0.33, 0.32, 0.14]}, 0.18
            )
        )

    def test_scene_conditioned_view_uses_only_current_features(self) -> None:
        query = {"values": [-5.0, 2.0, 0.1, 400.0, 300.0]}
        model = {
            "neighbor_count": 1,
            "episodes": [
                {"seed": 1, "view_mode": "close_high", "features": query},
                {
                    "seed": 2,
                    "view_mode": "right",
                    "features": {"values": [-8.0, 1.0, -0.1, 780.0, 230.0]},
                },
            ],
        }
        prediction = predict_view_mode(query, model)
        policy = select_scene_conditioned_view(
            prediction,
            {"viewpoint_close_high": 0.08, "viewpoint_right": 0.06},
            1.0,
        )
        self.assertEqual(policy["selected_action"], "viewpoint_close_high")
        self.assertFalse(prediction["future_view_observation_used"])

    def test_scene_conditioned_prediction_can_modify_sensor_model_without_override(self) -> None:
        hypotheses = [
            "track_center_selected|inside",
            "track_center_selected|outside",
        ]
        base = {
            "observation_vocabulary": {
                "viewpoint_close_high": ["resolved", "other|unknown"],
                "viewpoint_right": ["resolved", "other|unknown"],
            },
            "joint_observation_likelihood": {
                action: {
                    hypotheses[0]: {"resolved": 0.9, "other|unknown": 0.1},
                    hypotheses[1]: {"resolved": 0.2, "other|unknown": 0.8},
                }
                for action in ("viewpoint_close_high", "viewpoint_right")
            },
        }
        prediction = {
            "view_mode_probabilities": {
                "close_high": 0.75,
                "right": 0.0,
                "none": 0.25,
            }
        }
        conditioned = condition_joint_model_on_view_resolvability(
            base, prediction
        )
        for action in ("viewpoint_close_high", "viewpoint_right"):
            for row in conditioned["joint_observation_likelihood"][action].values():
                self.assertAlmostEqual(sum(row.values()), 1.0)
        right_rows = conditioned["joint_observation_likelihood"][
            "viewpoint_right"
        ]
        self.assertEqual(
            right_rows[hypotheses[0]], right_rows[hypotheses[1]]
        )
        self.assertFalse(
            conditioned["scene_conditioned_sensor_model"]["policy_override_used"]
        )

    def test_posthoc_audit_rejects_wrong_membership_and_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "household_scene.json").write_text(
                json.dumps(
                    {
                        "calibration_ground_truth": {
                            "world_ground_truth": {"membership": "outside"},
                            "action_outcome_design": {
                                "resolving_view_actions": ["viewpoint_right"],
                                "required_action_sequence": [
                                    "remove_cover",
                                    "viewpoint_right",
                                    "grasp_outside",
                                ],
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            audit = posthoc_semantic_audit(
                output_dir,
                "grasp:track_other_target:inside",
                ["remove_cover", "viewpoint_close_high", "grasp:track_other_target:inside"],
                {"manipulation_target_path": "/World/TargetRed"},
            )
        self.assertFalse(audit["passed"])
        self.assertFalse(audit["membership_correct"])
        self.assertFalse(audit["view_choice_correct"])

    def test_posthoc_audit_accepts_null_action_outcome_design(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "household_scene.json").write_text(
                json.dumps(
                    {
                        "calibration_ground_truth": {
                            "world_ground_truth": {"membership": "outside"},
                            "action_outcome_design": None,
                        }
                    }
                ),
                encoding="utf-8",
            )
            audit = posthoc_semantic_audit(
                output_dir,
                "grasp:track_center_selected:outside",
                ["grasp:track_center_selected:outside"],
                {"manipulation_target_path": "/World/TargetRed"},
            )
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["resolving_view_actions"], [])

    def test_state_crosses_track_identity_and_membership(self) -> None:
        self.assertEqual(
            state_for_episode(episode(True, "inside", "same")),
            "track_center_selected|inside",
        )
        self.assertEqual(
            state_for_episode(episode(False, "outside", "different")),
            "track_other_target|outside",
        )

    def test_observation_uses_persistent_track_evidence(self) -> None:
        row = episode(True, "inside", "same")["viewpoint_right"]
        self.assertEqual(observation_symbol(row), "same_high|inside")

    def test_joint_likelihood_is_normalized_per_state(self) -> None:
        episodes = {
            1: episode(True, "inside", "same"),
            2: episode(False, "outside", "different"),
            3: episode(True, "outside", "same"),
            4: episode(False, "inside", "missing"),
        }
        model = fit_joint_model(episodes, ["viewpoint_right"], 0.5)
        self.assertAlmostEqual(sum(model["prior"].values()), 1.0)
        self.assertEqual(set(model["prior"]), set(HYPOTHESES))
        for state in HYPOTHESES:
            self.assertAlmostEqual(
                sum(model["joint_observation_likelihood"]["viewpoint_right"][state].values()),
                1.0,
            )
        self.assertFalse(model["marginal_confidence_product_used"])

    def test_newly_discovered_track_can_be_grasped(self) -> None:
        belief = {
            "track_center_selected|inside": 0.02,
            "track_center_selected|outside": 0.01,
            "track_other_target|inside": 0.96,
            "track_other_target|outside": 0.01,
        }
        config = {
            "conditional_execution_success_probability": 0.95,
            "costs": {
                "grasp": 0.12,
                "wrong_commitment": 1.0,
                "execution_failure": 0.5,
                "defer": 0.8,
            },
        }
        selected = min(
            terminal_values(belief, config),
            key=lambda item: item["expected_cost"],
        )
        self.assertEqual(
            selected["action"], "grasp:track_other_target:inside"
        )


if __name__ == "__main__":
    unittest.main()
