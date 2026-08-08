"""CPU-only tests for the causal view-action scene design."""

import json
import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_action_differentiating_scene_manifest import (  # noqa: E402
    build_scene_manifest,
)
from scanned_basket_scene import (  # noqa: E402
    ACTION_DIFFERENTIATING_SCENE_VARIANTS,
    ACTION_OCCLUDER_TARGET_DISTANCE_M,
    NEGATIVE_EVIDENCE_TARGET_X_OFFSET_M,
    NEGATIVE_EVIDENCE_TARGET_Y_OFFSET_M,
    ACTION_PARTIAL_COVER_FULL_EXTENTS_M,
    ACTION_VERTICAL_OCCLUDER_HEIGHT_M,
    action_variant_for_seed,
    compute_action_differentiating_layout,
    factorized_calibration_ground_truth,
    validate_action_differentiating_visibility,
)


def measurement(value: float) -> dict:
    return {
        "target_visible_pixel_count": int(value * 1000),
        "objective_occlusion": {
            "valid": True,
            "visible_fraction_of_amodal": value,
        },
    }


class ActionDifferentiatingSceneDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (
                ROOT
                / "configs"
                / "research"
                / "action_differentiating_scene_pilot.json"
            ).read_text(encoding="utf-8")
        )

    def test_action_variant_cycle_is_balanced(self) -> None:
        variants = [
            action_variant_for_seed(seed) for seed in range(8)
        ]
        self.assertEqual(
            variants,
            list(ACTION_DIFFERENTIATING_SCENE_VARIANTS) * 2,
        )
        with self.assertRaises(ValueError):
            action_variant_for_seed(-1)

    def test_layout_blocks_only_declared_view_by_design(self) -> None:
        basket = (0.48, 0.10, 0.755)
        expected = {
            "close_high_only": "right",
            "right_only": "close_high",
            "either_view": None,
            "cover_removal_required": None,
        }
        for variant, blocked in expected.items():
            layout = compute_action_differentiating_layout(
                basket, variant, 185
            )
            self.assertEqual(layout["blocked_view"], blocked)
            self.assertEqual(
                layout["cover_required"],
                variant == "cover_removal_required",
            )
            if blocked is None:
                self.assertIsNone(
                    layout["action_occluder_position_world_m"]
                )
            else:
                geometry = layout["action_occluder_geometry"]
                if blocked == "right":
                    self.assertEqual(
                        geometry["type"], "supported_upright_cylinder"
                    )
                    self.assertEqual(
                        geometry["height_m"], ACTION_VERTICAL_OCCLUDER_HEIGHT_M
                    )
                    target = layout["target_position_world_m"]
                    occluder = layout["action_occluder_position_world_m"]
                    planar_distance = math.hypot(
                        target[0] - occluder[0], target[1] - occluder[1]
                    )
                    self.assertAlmostEqual(
                        planar_distance, ACTION_OCCLUDER_TARGET_DISTANCE_M
                    )
                else:
                    self.assertEqual(
                        geometry["type"],
                        "rim_supported_partial_cover_bar",
                    )
                    self.assertEqual(
                        geometry["full_extents_m"],
                        list(ACTION_PARTIAL_COVER_FULL_EXTENTS_M),
                    )

    def test_ground_truth_declares_causal_action_outcomes(self) -> None:
        expected = {
            "close_high_only": ["viewpoint_close_high"],
            "right_only": ["viewpoint_right"],
            "either_view": [
                "viewpoint_close_high",
                "viewpoint_right",
            ],
            "cover_removal_required": [],
        }
        for variant, actions in expected.items():
            truth = factorized_calibration_ground_truth(variant)
            design = truth["action_outcome_design"]
            self.assertEqual(design["resolving_view_actions"], actions)
            self.assertEqual(
                design["required_interaction_action"],
                (
                    "remove_cover"
                    if variant == "cover_removal_required"
                    else None
                ),
            )

    def test_empty_cover_variant_requires_negative_evidence_sequence(self) -> None:
        truth = factorized_calibration_ground_truth(
            "empty_cover_then_right"
        )
        self.assertEqual(
            truth["world_ground_truth"]["membership"], "outside"
        )
        design = truth["action_outcome_design"]
        self.assertEqual(design["post_interaction_observation"], "empty_container")
        self.assertEqual(design["minimum_belief_updates"], 2)
        self.assertEqual(
            design["required_action_sequence"],
            ["remove_cover", "viewpoint_right", "grasp_outside"],
        )
        self.assertGreater(NEGATIVE_EVIDENCE_TARGET_X_OFFSET_M, 0.0)
        self.assertGreater(NEGATIVE_EVIDENCE_TARGET_Y_OFFSET_M, 0.0)

    def test_empty_cover_visibility_requires_right_view_resolution(self) -> None:
        passed = {
            "center": measurement(0.20),
            "close_high": measurement(0.35),
            "right": measurement(0.85),
        }
        failed = {
            "center": measurement(0.20),
            "close_high": measurement(0.72),
            "right": measurement(0.80),
        }
        from scanned_basket_scene import validate_calibration_visibility

        self.assertTrue(
            validate_calibration_visibility(
                "empty_cover_then_right", passed
            )["passed"]
        )
        self.assertFalse(
            validate_calibration_visibility(
                "empty_cover_then_right", failed
            )["passed"]
        )

    def test_rendered_visibility_gates_are_action_specific(self) -> None:
        close_only = {
            "center": measurement(0.35),
            "close_high": measurement(0.85),
            "right": measurement(0.50),
        }
        right_only = {
            "center": measurement(0.35),
            "close_high": measurement(0.45),
            "right": measurement(0.85),
        }
        either = {
            "center": measurement(0.35),
            "close_high": measurement(0.80),
            "right": measurement(0.75),
        }
        covered = {
            view: measurement(0.0)
            for view in ("center", "close_high", "right")
        }
        self.assertTrue(
            validate_action_differentiating_visibility(
                "close_high_only", close_only
            )["passed"]
        )
        self.assertTrue(
            validate_action_differentiating_visibility(
                "right_only", right_only
            )["passed"]
        )
        self.assertTrue(
            validate_action_differentiating_visibility(
                "either_view", either
            )["passed"]
        )
        self.assertTrue(
            validate_action_differentiating_visibility(
                "cover_removal_required", covered
            )["passed"]
        )
        self.assertFalse(
            validate_action_differentiating_visibility(
                "right_only", close_only
            )["passed"]
        )
        self.assertFalse(
            validate_action_differentiating_visibility(
                "close_high_only",
                {
                    "center": measurement(0.35),
                    "close_high": measurement(0.95),
                    "right": measurement(0.70),
                },
            )["passed"]
        )
        self.assertFalse(
            validate_action_differentiating_visibility(
                "right_only",
                {
                    "center": measurement(0.35),
                    "close_high": measurement(0.70),
                    "right": measurement(0.95),
                },
            )["passed"]
        )

    def test_manifest_is_balanced_and_avoids_reserved_test_seeds(self) -> None:
        manifest = build_scene_manifest(self.config)
        self.assertEqual(manifest["scene_count"], 12)
        self.assertEqual(
            set(manifest["variant_counts"].values()),
            {3},
        )
        self.assertTrue(
            set(manifest["seeds"]).isdisjoint(
                self.config["reserved_test_seeds"]
            )
        )
        self.assertFalse(manifest["gpu_used"])
        self.assertTrue(
            all(
                scene["render_validation_passed"] is None
                for scene in manifest["scenes"]
            )
        )


if __name__ == "__main__":
    unittest.main()
