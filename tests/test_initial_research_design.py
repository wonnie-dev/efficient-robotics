"""Tests for calibration, negative evidence, and non-oracle planning."""

import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from calibrated_belief import (  # noqa: E402
    bayesian_update,
    binary_detection_likelihood,
    entropy,
    fit_temperature_grid,
    softmax_temperature,
)
from run_non_oracle_hybrid_planner import observation_branches, plan  # noqa: E402
from run_non_oracle_hybrid_planner import task_failure_risk  # noqa: E402


class InitialResearchDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (ROOT / "configs/research/initial_method_design.json").read_text(
                encoding="utf-8"
            )
        )

    def test_temperature_softmax_is_categorical(self) -> None:
        probabilities = softmax_temperature([2.0, 1.0, -1.0], 1.5)
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertGreater(probabilities[0], probabilities[1])

    def test_temperature_fit_returns_positive_value(self) -> None:
        fitted = fit_temperature_grid([[3.0, 1.0], [0.5, 2.0]], [0, 1])
        self.assertGreater(fitted["temperature"], 0.0)
        self.assertTrue(fitted["calibrated"])

    def test_negative_evidence_reduces_detectable_hypothesis(self) -> None:
        prior = {"target_red": 0.6, "rear_red_candidate": 0.4}
        detection = {"target_red": 0.9, "rear_red_candidate": 0.2}
        posterior = bayesian_update(
            prior, binary_detection_likelihood(detection, detected=False)
        )
        self.assertLess(posterior["target_red"], prior["target_red"])

    def test_observation_branches_form_probability_distribution(self) -> None:
        belief = self.config["initial_belief"]
        branches = observation_branches(
            {"target": belief["target"], "relation": belief["relation"]},
            "viewpoint_right",
            self.config["observation_model"],
        )
        self.assertAlmostEqual(sum(branch["probability"] for branch in branches), 1.0)

    def test_planner_does_not_report_oracle(self) -> None:
        result = plan(copy.deepcopy(self.config))
        self.assertFalse(result["provenance"]["oracle"])
        self.assertEqual(result["provenance"]["future_capture_files_read"], [])
        self.assertIn(result["action_request"]["type"], self.config["actions"])
        self.assertEqual(result["selected_sequence"][-1], "grasp")

    def test_entropy_is_only_uncertainty_summary(self) -> None:
        self.assertGreater(entropy({"a": 0.5, "b": 0.5}), entropy({"a": 0.99, "b": 0.01}))

    def test_risk_uses_most_likely_target_hypothesis(self) -> None:
        belief = {
            "target": {"candidate_a": 0.8, "candidate_b": 0.2},
            "relation": {"inside": 0.9, "unknown": 0.1},
        }
        self.assertAlmostEqual(task_failure_risk(belief, self.config["objective"]), 0.28)


if __name__ == "__main__":
    unittest.main()
