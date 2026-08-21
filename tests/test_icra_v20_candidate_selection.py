import unittest

from scripts.evaluate_icra_v20_candidate_selection import (
    candidate_probabilities,
    decision,
)


class CandidateSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sample = {
            "candidates": [
                {
                    "detection_id": "a",
                    "qwen_logit": 3.0,
                    "detector_score": 0.1,
                    "correct_at_mask_iou_0_5": True,
                },
                {
                    "detection_id": "b",
                    "qwen_logit": 1.0,
                    "detector_score": 0.9,
                    "correct_at_mask_iou_0_5": False,
                },
            ]
        }

    def test_qwen_and_detector_endpoints(self) -> None:
        self.assertEqual(decision(self.sample, 1.0)["selected_detection_id"], "a")
        self.assertEqual(decision(self.sample, 0.0)["selected_detection_id"], "b")

    def test_probabilities_are_normalized(self) -> None:
        values = candidate_probabilities(self.sample, 0.5)
        self.assertAlmostEqual(sum(values), 1.0)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in values))


if __name__ == "__main__":
    unittest.main()
