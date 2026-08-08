import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cached_perception_calibration_cv import (  # noqa: E402
    build_examples,
    evaluate,
    expected_calibration_error,
    validate_folds,
)


class CachedPerceptionCalibrationCvTest(unittest.TestCase):
    def test_build_examples_preserves_target_and_objective_occlusion(self):
        records = [
            {
                "seed": 1,
                "sample_id": "seed001_center",
                "candidates": [
                    {
                        "raw_match_logit": 2.0,
                        "target_label": True,
                        "matched_simulator_entity": "target_red",
                        "relation_ground_truth": {
                            "membership": "inside",
                            "behind": "no",
                            "occluded_by": "yes",
                        },
                        "relation_ground_truth_sources": {
                            "occluded_by": (
                                "rendered_reference_removed_amodal_fraction"
                            )
                        },
                        "factorized_relation_scores": {
                            "membership": {
                                "labels": ["inside", "outside", "unknown"],
                                "raw_logits": [2.0, 0.0, -1.0],
                            },
                            "behind": {
                                "labels": ["yes", "no", "unknown"],
                                "raw_logits": [0.0, 2.0, -1.0],
                            },
                            "occluded_by": {
                                "labels": ["yes", "no"],
                                "raw_logits": [2.0, 0.0],
                            },
                        },
                    }
                ],
            }
        ]
        examples = build_examples(records)
        self.assertEqual(len(examples["target_identity"]), 1)
        self.assertEqual(len(examples["membership"]), 1)
        self.assertEqual(len(examples["occluded_by"]), 1)
        self.assertEqual(examples["target_identity"][0]["label"], 0)

    def test_evaluate_reports_per_class_recall(self):
        examples = [
            {"seed": 1, "logits": [4.0, 0.0], "label": 0},
            {"seed": 2, "logits": [0.0, 4.0], "label": 1},
        ]
        result = evaluate(examples, {1: 1.0, 2: 1.0}, bins=5)
        self.assertEqual(result["accuracy"], 1.0)
        self.assertEqual(result["per_class"]["0"]["recall"], 1.0)
        self.assertEqual(result["per_class"]["1"]["recall"], 1.0)

    def test_ece_is_zero_for_perfect_unit_confidence(self):
        self.assertEqual(
            expected_calibration_error([1.0, 1.0], [True, True], 10),
            0.0,
        )

    def test_validate_folds_rejects_missing_seed(self):
        with self.assertRaises(ValueError):
            validate_folds(
                [{"fold_id": 0, "held_out_seeds": [1]}],
                {1, 2},
            )


if __name__ == "__main__":
    unittest.main()
