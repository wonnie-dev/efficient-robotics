import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_grounded_qwen_calibration_pilot import (  # noqa: E402
    binary_metrics,
    entity_view_ground_truth,
    factor_label,
    fit_factor,
    objective_occlusion_label,
)


class GroundedQwenCalibrationPilotTest(unittest.TestCase):
    def test_binary_metrics_are_finite(self):
        result = binary_metrics(
            [[3.0, 0.0], [-2.0, 0.0]],
            [0, 1],
            1.0,
        )
        self.assertGreaterEqual(result["accuracy"], 0.0)
        self.assertLessEqual(result["accuracy"], 1.0)
        self.assertGreaterEqual(result["brier_score"], 0.0)
        self.assertGreaterEqual(result["negative_log_likelihood"], 0.0)

    def test_metrics_support_relation_label_space(self):
        result = binary_metrics(
            [[4.0, 1.0, 0.0, -1.0], [0.0, 3.0, 1.0, -2.0]],
            [0, 1],
            1.0,
        )
        self.assertEqual(result["accuracy"], 1.0)

    def test_factorized_ground_truth_lookup(self):
        ground_truth = {
            "view_observable_intent": {
                "center": {
                    "entities": {
                        "target_red": {
                            "membership_observable": "unknown",
                            "behind": "yes",
                            "occluded_by": {"label": "yes"},
                        }
                    }
                }
            }
        }
        entity = entity_view_ground_truth(
            ground_truth, "center", "target_red"
        )
        self.assertEqual(factor_label(entity, "membership"), "unknown")
        self.assertEqual(factor_label(entity, "behind"), "yes")
        self.assertEqual(factor_label(entity, "occluded_by"), "yes")

    def test_factor_fit_requires_all_factor_labels(self):
        result = fit_factor(
            [[4.0, 0.0, -2.0], [0.0, 4.0, -2.0]],
            [0, 1],
            {"inside", "outside"},
            factor="membership",
            calibration_seed_count=20,
        )
        self.assertIn(
            "missing_labels:unknown", result["blocking_reasons"]
        )

    def test_objective_occlusion_uses_rendered_severity(self):
        ground_truth = {
            "objective_reference_occlusion_ground_truth": {
                "center": {"valid": True, "severity": "partial"},
                "right": {"valid": True, "severity": "no"},
            }
        }
        self.assertEqual(
            objective_occlusion_label(ground_truth, "center"), "yes"
        )
        self.assertEqual(
            objective_occlusion_label(ground_truth, "right"), "no"
        )

    def test_factor_fit_rejects_zero_recall_supported_class(self):
        result = fit_factor(
            [
                [5.0, 0.0, -1.0],
                [0.0, 5.0, -1.0],
                [4.0, 0.0, 1.0],
            ],
            [0, 1, 2],
            {"inside", "outside", "unknown"},
            factor="membership",
            calibration_seed_count=20,
        )
        self.assertIn(
            "zero_recall_labels:unknown",
            result["blocking_reasons"],
        )
        self.assertEqual(
            result["uncalibrated_class_diagnostics"]["unknown"]["recall"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
