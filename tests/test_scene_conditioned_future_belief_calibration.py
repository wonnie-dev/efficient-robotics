"""Focused CPU tests for the scene-conditioned calibration smoke."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_scene_conditioned_future_belief_calibration import (  # noqa: E402
    VARIANTS,
    predict_variant_distribution,
    select_action,
)


class SceneConditionedFutureBeliefTests(unittest.TestCase):
    def test_pure_variant_beliefs_choose_their_resolving_actions(self) -> None:
        expected = {
            "close_high_only": "viewpoint_close_high",
            "right_only": "viewpoint_right",
            "either_view": "viewpoint_right",
            "cover_removal_required": "remove_cover",
        }
        for variant, action in expected.items():
            probabilities = {
                current: float(current == variant) for current in VARIANTS
            }
            self.assertEqual(
                select_action(probabilities)["selected_action"], action
            )

    def test_nonparametric_prediction_is_normalized(self) -> None:
        rows = [
            {
                "seed": 1,
                "variant": "close_high_only",
                "features": {"values": [0.0, 1.0, 1.0, 0.2, 0.1]},
            },
            {
                "seed": 2,
                "variant": "right_only",
                "features": {"values": [1.0, 2.0, 3.0, 0.3, 0.2]},
            },
            {
                "seed": 3,
                "variant": "cover_removal_required",
                "features": {"values": [-1.0, 1.0, 0.8, 0.6, 0.4]},
            },
        ]
        prediction = predict_variant_distribution(
            {"values": [0.1, 1.0, 1.1, 0.2, 0.1]}, rows, k=3
        )
        self.assertAlmostEqual(
            sum(prediction["variant_probabilities"].values()), 1.0
        )
        self.assertEqual({item["seed"] for item in prediction["neighbors"]}, {1, 2, 3})


if __name__ == "__main__":
    unittest.main()
