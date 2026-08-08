import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from scanned_basket_scene import (  # noqa: E402
    CALIBRATION_SCENE_VARIANTS,
    calibration_variant_for_seed,
    compute_behind_ambiguous_target_layout,
    compute_behind_boundary_unknown_target_layout,
    compute_rim_occluded_target_layout,
    factorized_calibration_ground_truth,
    validate_calibration_visibility,
)
from seeded_benchmark import generate_layout  # noqa: E402


class ScannedBasketSceneTest(unittest.TestCase):
    def test_behind_ambiguous_layout_aligns_ray_and_clears_basket(self):
        positions = set()
        for seed in range(1000):
            result = compute_behind_ambiguous_target_layout(
                (0.48, 0.10, 0.755), seed
            )
            self.assertTrue(result["center_camera_ray_aligned"])
            self.assertTrue(result["geometry_validation_passed"])
            self.assertGreaterEqual(
                result["basket_planar_clearance_m"], 0.012 - 1e-9
            )
            positions.add(
                tuple(round(value, 5) for value in result[
                    "target_position_world_m"
                ])
            )
        self.assertGreater(len(positions), 900)

    def test_calibration_variant_cycle_is_balanced_and_deterministic(self):
        self.assertEqual(
            [calibration_variant_for_seed(seed) for seed in range(8)],
            list(CALIBRATION_SCENE_VARIANTS) * 2,
        )
        with self.assertRaises(ValueError):
            calibration_variant_for_seed(-1)

    def test_behind_boundary_unknown_layout_is_clear_and_in_band(self):
        positions = set()
        for seed in range(1000):
            result = compute_behind_boundary_unknown_target_layout(
                (0.48, 0.10, 0.755), seed
            )
            self.assertTrue(result["geometry_validation_passed"])
            self.assertGreaterEqual(
                result["basket_planar_clearance_m"], 0.012 - 1e-9
            )
            self.assertLessEqual(
                abs(result["analytic_transform_origin_far_edge_offset_m"]),
                0.003 + 1e-9,
            )
            positions.add(
                tuple(
                    round(value, 5)
                    for value in result["target_position_world_m"]
                )
            )
        self.assertGreater(len(positions), 900)

    def test_factorized_ground_truth_separates_world_and_observability(self):
        outside = factorized_calibration_ground_truth("outside")
        self.assertEqual(
            outside["world_ground_truth"]["membership"], "outside"
        )
        self.assertEqual(
            outside["view_observable_intent"]["center"][
                "membership_observable"
            ],
            "outside",
        )
        rim = factorized_calibration_ground_truth("rim_occluded")
        self.assertEqual(rim["world_ground_truth"]["membership"], "inside")
        self.assertEqual(
            rim["view_observable_intent"]["center"][
                "membership_observable"
            ],
            "unknown",
        )
        self.assertEqual(
            rim["view_observable_intent"]["center"]["behind"], "yes"
        )
        self.assertEqual(
            rim["view_observable_intent"]["center"]["occluded_by"][
                "occluder_id"
            ],
            "basket_01",
        )
        self.assertEqual(
            rim["view_observable_intent"]["center"]["entities"][
                "rear_red_candidate"
            ]["membership_observable"],
            "outside",
        )

    def test_covered_unknown_is_inside_in_world_but_unknown_in_views(self):
        result = factorized_calibration_ground_truth("covered_unknown")
        self.assertEqual(
            result["world_ground_truth"]["membership"], "inside"
        )
        for view in result["view_observable_intent"].values():
            self.assertEqual(view["membership_observable"], "unknown")
            self.assertEqual(view["behind"], "unknown")
            self.assertEqual(view["occluded_by"]["label"], "yes")
            self.assertEqual(
                view["occluded_by"]["occluder_id"], "cover_01"
            )
        self.assertFalse(result["manual_annotation"])

    def test_behind_ambiguous_resolves_after_reobservation(self):
        result = factorized_calibration_ground_truth(
            "behind_ambiguous"
        )
        self.assertEqual(
            result["world_ground_truth"]["membership"], "outside"
        )
        center = result["view_observable_intent"]["center"]["entities"][
            "target_red"
        ]
        self.assertEqual(center["membership_observable"], "unknown")
        self.assertEqual(center["behind"], "unknown")
        self.assertEqual(center["occluded_by"]["label"], "yes")
        for view_id in ("close_high", "right"):
            resolved = result["view_observable_intent"][view_id][
                "entities"
            ]["target_red"]
            self.assertEqual(
                resolved["membership_observable"], "outside"
            )
            self.assertEqual(resolved["behind"], "yes")
            self.assertEqual(resolved["occluded_by"]["label"], "no")

    def test_rendered_visibility_gate_matches_variant_intent(self):
        visible = {
            "center": {"target_visible_pixel_count": 100},
            "close_high": {"target_visible_pixel_count": 300},
            "right": {"target_visible_pixel_count": 200},
        }
        self.assertTrue(
            validate_calibration_visibility(
                "inside_clear", visible
            )["passed"]
        )
        self.assertTrue(
            validate_calibration_visibility(
                "rim_occluded", visible
            )["passed"]
        )
        hidden = {
            view: {"target_visible_pixel_count": 0}
            for view in ("center", "close_high", "right")
        }
        self.assertTrue(
            validate_calibration_visibility(
                "covered_unknown", hidden
            )["passed"]
        )
        self.assertFalse(
            validate_calibration_visibility(
                "covered_unknown", visible
            )["passed"]
        )
        self.assertTrue(
            validate_calibration_visibility(
                "behind_ambiguous", visible
            )["passed"]
        )

    def test_rim_occluded_target_clears_basket_across_seed_range(self):
        basket_center = (0.48, 0.10, 0.755)
        for seed in range(1000):
            target = tuple(
                generate_layout(seed)["positions_world_m"]["target_red"]
            )
            result = compute_rim_occluded_target_layout(target, basket_center)
            self.assertTrue(result["geometry_validation_passed"])
            self.assertFalse(result["explicit_occluder_primitive_visible"])
            self.assertGreaterEqual(result["wall_clearance_m"], 0.020 - 1e-9)
            self.assertEqual(
                result["target_transform_origin"], "mug_bottom_contact"
            )
            self.assertAlmostEqual(
                result["target_position_world_m"][2],
                basket_center[2] + 0.020,
            )
            self.assertAlmostEqual(
                result["target_position_world_m"][1],
                basket_center[1] + 0.020,
            )


if __name__ == "__main__":
    unittest.main()
