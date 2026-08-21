import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_joint_observation_model import (  # noqa: E402
    fit_model,
    likelihood_for_with_source,
    match_label,
    normalize_counts,
    observation_symbol,
    persistent_observation_symbol,
)


PREFLIGHT = ROOT / "configs/research/unified_task_belief.json"


class JointObservationCalibrationTests(unittest.TestCase):
    def test_exact_semantic_symbol_backs_off_across_calibrated_actions(self):
        hypotheses = (
            "track_center_selected|inside",
            "track_center_selected|outside",
            "track_other_target|inside",
            "track_other_target|outside",
            "target_absent|not_applicable",
        )

        def rows(outcomes, positive_hypothesis):
            return {
                hypothesis: {
                    outcome: (
                        0.9
                        if outcome == "other_target|outside"
                        and hypothesis == positive_hypothesis
                        else 0.025
                        if outcome == "other_target|outside"
                        else 0.1
                    )
                    for outcome in outcomes
                }
                for hypothesis in hypotheses
            }

        close_outcomes = ["center_target|inside", "unseen"]
        right_outcomes = ["other_target|outside", "unseen"]
        model = {
            "observation_model": {
                "viewpoint_close_high": {
                    "outcomes": close_outcomes,
                    "likelihood": rows(
                        close_outcomes, "track_center_selected|inside"
                    ),
                },
                "viewpoint_right": {
                    "outcomes": right_outcomes,
                    "likelihood": rows(
                        right_outcomes, "track_other_target|outside"
                    ),
                },
            }
        }
        likelihood, provenance = likelihood_for_with_source(
            model,
            "viewpoint_close_high",
            "other_target|outside",
            semantic_backoff_model=model,
        )
        self.assertEqual(
            provenance["source"],
            "cross_action_exact_semantic_symbol_backoff",
        )
        self.assertEqual(provenance["backoff_actions"], ["viewpoint_right"])
        self.assertGreater(
            likelihood["track_other_target|outside"],
            likelihood["target_absent|not_applicable"],
        )

    def test_persistent_symbol_uses_strongest_match_and_explicit_negative(self):
        row = {
            "center_track_candidate_id": "candidate_a",
            "candidate_evidence": {
                "candidate_a": {
                    "raw_match_logit": 0.5,
                    "membership": "outside",
                },
                "candidate_b": {
                    "raw_match_logit": 3.0,
                    "membership": "inside",
                },
            },
        }
        self.assertEqual(
            persistent_observation_symbol(row), "other_target|inside"
        )
        for value in row["candidate_evidence"].values():
            value["raw_match_logit"] = -1.0
        self.assertEqual(
            persistent_observation_symbol(row), "no_target_evidence"
        )

    def test_native_match_boundary_is_not_a_probability_gate(self):
        self.assertEqual(match_label(0.0), "match")
        self.assertEqual(match_label(-0.001), "nonmatch")

    def test_observation_symbol_keeps_joint_identity_relation_evidence(self):
        self.assertEqual(
            observation_symbol(
                {
                    "action": "viewpoint_right",
                    "center_track_candidate_id": "candidate_001",
                    "candidate_evidence": {
                        "candidate_001": {
                            "raw_match_logit": -3.0,
                            "membership": "outside",
                        },
                        "candidate_002": {
                            "raw_match_logit": 3.0,
                            "membership": "inside",
                        },
                    },
                }
            ),
            "other_target|inside",
        )

    def test_non_target_relation_does_not_update_target_membership(self):
        self.assertEqual(
            observation_symbol(
                {
                    "action": "remove_cover",
                    "center_track_candidate_id": "candidate_001",
                    "candidate_evidence": {
                        "candidate_001": {
                            "raw_match_logit": -2.0,
                            "membership": "inside",
                        }
                    },
                }
            ),
            "target_unresolved|unknown",
        )

    def test_dirichlet_counts_are_normalized(self):
        values = normalize_counts(Counter({"a": 2}), ("a", "b"), 0.5)
        self.assertAlmostEqual(sum(values.values()), 1.0)
        self.assertGreater(values["a"], values["b"])

    def test_remove_cover_failure_outcome_preserves_covered_state(self):
        import json

        preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
        episodes = {
            1: {
                "true_joint_hypothesis": "track_center_selected|inside",
                "rows": {
                    "initial_observation": {
                        "observation_symbol": "no_target_evidence",
                        "true_joint_hypothesis": "track_center_selected|inside",
                    },
                    "remove_cover": {
                        "observation_symbol": "center_target|inside",
                        "true_joint_hypothesis": "track_center_selected|inside",
                    },
                },
            }
        }
        model = fit_model(episodes, 0.5, preflight)
        action = model["information_actions"]["remove_cover"]
        self.assertIn("removal_failed", action["outcomes"])
        self.assertEqual(
            action["next_task_state_by_outcome"]["covered"]["removal_failed"],
            "covered",
        )
        self.assertEqual(
            action["next_task_state_by_outcome"]["covered"][
                "center_target|inside"
            ],
            "open",
        )


if __name__ == "__main__":
    unittest.main()
