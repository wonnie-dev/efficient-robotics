import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_hybrid_rgbd_relation_pilot import (  # noqa: E402
    binary_dilate,
    classify_behind,
    classify_membership,
    classify_occlusion,
    reference_geometry,
)


def reference(valid=True):
    return {
        "valid": valid,
        "bounds_world_m": {
            "lower": [0.0, 0.0, 0.0],
            "upper": [1.0, 1.0, 0.5],
        },
    }


class HybridRgbdRelationPilotTest(unittest.TestCase):
    def test_reference_geometry_can_use_measured_full_extents(self):
        points = np.asarray(
            [
                [0.1, 0.2, 0.0],
                [0.9, 0.8, 0.1],
            ]
        )
        result = reference_geometry(
            points,
            {
                "robust_percentiles": [0.0, 100.0],
                "expected_full_extents_xy_m": [1.0, 1.0],
                "minimum_extent_ratio": 0.5,
                "maximum_extent_ratio": 1.5,
                "frame_assumption": "axis_aligned",
                "use_expected_xy_extents_for_relation": True,
            },
        )
        np.testing.assert_allclose(
            result["bounds_world_m"]["lower"][:2], [0.0, 0.0]
        )
        np.testing.assert_allclose(
            result["bounds_world_m"]["upper"][:2], [1.0, 1.0]
        )
        self.assertEqual(
            result["relation_xy_extent_source"],
            "measured_reference_dimensions_centered_on_observed_geometry",
        )

    def test_binary_dilate_expands_without_wraparound(self):
        mask = np.zeros((5, 5), dtype=bool)
        mask[0, 0] = True
        result = binary_dilate(mask, 1)
        self.assertEqual(int(result.sum()), 4)
        self.assertFalse(result[-1, -1])

    def test_membership_uses_boundary_abstention(self):
        inside = classify_membership(
            np.asarray([0.5, 0.5, 0.1]), reference(), 0.02
        )
        near = classify_membership(
            np.asarray([1.01, 0.5, 0.1]), reference(), 0.02
        )
        outside = classify_membership(
            np.asarray([1.10, 0.5, 0.1]), reference(), 0.02
        )
        self.assertEqual(inside["label"], "inside")
        self.assertEqual(near["label"], "unknown")
        self.assertEqual(outside["label"], "outside")

    def test_invalid_reference_forces_membership_abstention(self):
        result = classify_membership(
            np.asarray([0.5, 0.5, 0.1]), reference(False), 0.02
        )
        self.assertEqual(result["label"], "unknown")

    def test_occlusion_requires_incomplete_extent_and_adjacency(self):
        settings = {
            "yes_max_visible_height_ratio": 0.72,
            "yes_min_reference_adjacency": 0.10,
            "no_min_visible_height_ratio": 0.85,
            "no_max_reference_adjacency": 0.03,
        }
        self.assertEqual(
            classify_occlusion(0.60, 0.20, settings)["label"], "yes"
        )
        self.assertEqual(
            classify_occlusion(0.95, 0.20, settings)["label"], "no"
        )
        self.assertEqual(
            classify_occlusion(0.78, 0.08, settings)["label"],
            "unknown",
        )

    def test_behind_detects_far_edge_and_inside_rim_cases(self):
        settings = {
            "minimum_candidate_bbox_overlap": 0.1,
            "far_edge_abstention_m": 0.02,
        }
        camera = np.asarray([-1.0, 0.5, 1.0])
        outside = classify_behind(
            np.asarray([1.2, 0.5, 0.1]),
            camera,
            reference(),
            {"label": "outside"},
            {"label": "no"},
            0.5,
            settings,
        )
        inside_rim = classify_behind(
            np.asarray([0.5, 0.5, 0.1]),
            camera,
            reference(),
            {"label": "inside"},
            {"label": "yes"},
            0.5,
            settings,
        )
        self.assertEqual(outside["label"], "yes")
        self.assertEqual(inside_rim["label"], "yes")


if __name__ == "__main__":
    unittest.main()
