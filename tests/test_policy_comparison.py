import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_policy_comparison import (  # noqa: E402
    DEFAULT_FREEZE_ROOT,
    DEFAULT_DEFINITIONS,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PERCEPTION_ROOT,
    EXPECTED_TEST_SEEDS,
    factorize_joint_belief,
    replay_native_greedy,
    run,
    update_with_native_scores,
)


class PolicyComparisonTests(unittest.TestCase):
    def test_factorization_preserves_absence_and_marginals(self) -> None:
        belief = {
            "track_center_selected|inside": 0.36,
            "track_center_selected|outside": 0.04,
            "track_other_target|inside": 0.09,
            "track_other_target|outside": 0.31,
            "target_absent|not_applicable": 0.20,
        }
        result = factorize_joint_belief(belief)
        self.assertAlmostEqual(result["target_absent|not_applicable"], 0.20)
        self.assertAlmostEqual(
            result["track_center_selected|inside"]
            + result["track_center_selected|outside"],
            0.40,
        )
        self.assertAlmostEqual(
            result["track_center_selected|inside"]
            + result["track_other_target|inside"],
            0.45,
        )
        self.assertAlmostEqual(sum(result.values()), 1.0)

    def test_declared_method_set_is_complete(self) -> None:
        definitions = json.loads(
            (ROOT / "configs/research/method_definitions.json").read_text(
                encoding="utf-8"
            )
        )
        expected_methods = {
            "proposed_task_risk_aware_joint_belief_mpc",
            "immediate_grasp",
            "fixed_right_view",
            "fixed_close_high_view",
            "confidence_greedy",
            "myopic_information_gain",
            "open_loop_belief_planner",
            "random_feasible_action",
            "oracle_simulator_ground_truth",
        }
        expected_ablations = {
            "no_joint_belief",
            "no_calibration",
            "no_negative_evidence",
            "no_action_conditioned_future_belief",
            "no_task_risk_cost",
            "no_persistent_tracking",
            "no_scene_conditioned_view_model",
        }
        self.assertEqual(set(definitions["methods"]), expected_methods)
        self.assertEqual(set(definitions["ablations"]), expected_ablations)
        self.assertFalse(
            definitions["shared_policy_rules"][
                "fixed_grasp_confidence_threshold_used"
            ]
        )

    def test_native_score_update_is_normalized_and_uses_raw_evidence(self) -> None:
        prior = {
            "track_center_selected|inside": 0.2,
            "track_center_selected|outside": 0.2,
            "track_other_target|inside": 0.2,
            "track_other_target|outside": 0.2,
            "target_absent|not_applicable": 0.2,
        }
        row = {
            "center_track_candidate_id": "candidate_a",
            "candidate_evidence": {
                "candidate_a": {"raw_match_logit": 4.0, "membership": "inside"},
                "candidate_b": {"raw_match_logit": -4.0, "membership": "outside"},
            },
        }
        posterior = update_with_native_scores(prior, row)
        self.assertAlmostEqual(sum(posterior.values()), 1.0)
        self.assertGreater(
            posterior["track_center_selected|inside"],
            posterior["track_other_target|outside"],
        )

    def test_native_greedy_records_view_before_using_its_observation(self) -> None:
        outcomes = ["no_target_evidence", "center_target|outside", "unseen"]
        view_action = {
            "allowed_task_states": ["open"],
            "stage_cost": 0.05,
            "outcomes": outcomes,
            "next_task_state_by_outcome": {
                "open": {outcome: "open" for outcome in outcomes}
            },
        }
        model = {
            "information_actions": {
                "viewpoint_right": view_action,
                "viewpoint_close_high": view_action,
            },
            "terminal_grasp_actions": {
                "grasp:track_center_selected:outside": {
                    "allowed_task_states": ["open"],
                    "semantic_hypothesis": "track_center_selected|outside",
                    "stage_cost": 0.1,
                    "conditional_execution_success_probability": 1.0,
                }
            },
            "costs": {
                "wrong_commitment": 1.0,
                "execution_failure": 1.0,
                "defer": 0.5,
            },
        }
        episode = {
            "seed": 0,
            "family": "unit_test",
            "initial_task_state": "open",
            "true_joint_hypothesis": "track_center_selected|outside",
            "rows": {
                "initial_observation": {"observation_symbol": "no_target_evidence"},
                "viewpoint_right": {"observation_symbol": "no_target_evidence"},
                "viewpoint_close_high": {
                    "observation_symbol": "center_target|outside"
                },
            },
        }

        result = replay_native_greedy(episode, model)

        self.assertEqual(
            result["action_sequence"],
            [
                "viewpoint_right",
                "viewpoint_close_high",
                "grasp:track_center_selected:outside",
            ],
        )
        self.assertEqual(result["information_action_count"], 2)

    def test_final_evaluator_uses_reserved_output_paths_and_seed_split(self) -> None:
        for path in (DEFAULT_PERCEPTION_ROOT, DEFAULT_FREEZE_ROOT, DEFAULT_OUTPUT_ROOT):
            self.assertNotIn("development", str(path))
        self.assertEqual(EXPECTED_TEST_SEEDS, set(range(1100, 1160)))

    def test_unfrozen_candidate_cannot_write_final_evaluation(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "development diagnostics"):
            run(
                DEFAULT_DEFINITIONS,
                DEFAULT_PERCEPTION_ROOT,
                DEFAULT_FREEZE_ROOT,
                DEFAULT_OUTPUT_ROOT,
                ROOT / "outputs/calibration/unfrozen_candidate",
            )


if __name__ == "__main__":
    unittest.main()
