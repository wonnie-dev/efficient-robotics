"""Tests for benchmark color-ID component disambiguation."""

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from observation_capture import (  # noqa: E402
    _split_id_pair_by_component_area,
    _split_id_pair_by_horizontal_components,
    objective_camera_relative_behind_measurement,
    objective_occlusion_measurement,
    objective_reference_occlusion_measurement,
)


class BenchmarkInstanceLabelTests(unittest.TestCase):
    def test_disconnected_pair_is_assigned_left_and_right(self) -> None:
        ids = np.zeros((20, 40), dtype=np.uint32)
        ids[2:12, 3:10] = 2
        ids[4:16, 25:34] = 3
        _split_id_pair_by_horizontal_components(
            ids, pair=(2, 3), left_id=3, right_id=2
        )
        self.assertTrue(np.all(ids[2:12, 3:10] == 3))
        self.assertTrue(np.all(ids[4:16, 25:34] == 2))

    def test_area_split_assigns_large_component_to_rear_candidate(self) -> None:
        ids = np.zeros((20, 40), dtype=np.uint32)
        ids[2:5, 3:7] = 6
        ids[4:16, 25:34] = 6
        _split_id_pair_by_component_area(
            ids,
            pair=(6, 7),
            small_id=6,
            large_id=7,
            minimum_large_pixels=20,
        )
        self.assertTrue(np.all(ids[2:5, 3:7] == 6))
        self.assertTrue(np.all(ids[4:16, 25:34] == 7))

    def test_objective_occlusion_severity_thresholds(self) -> None:
        amodal = np.ones((10, 10), dtype=bool)
        clear = amodal.copy()
        partial = amodal.copy()
        partial[:3] = False
        severe = amodal.copy()
        severe[:7] = False

        self.assertEqual(
            objective_occlusion_measurement(clear, amodal)["severity"],
            "no",
        )
        self.assertEqual(
            objective_occlusion_measurement(partial, amodal)["severity"],
            "partial",
        )
        severe_result = objective_occlusion_measurement(severe, amodal)
        self.assertEqual(severe_result["severity"], "severe")
        self.assertAlmostEqual(severe_result["occlusion_fraction"], 0.7)

    def test_objective_occlusion_invalid_and_shape_checks(self) -> None:
        empty = np.zeros((3, 4), dtype=bool)
        result = objective_occlusion_measurement(empty, empty)
        self.assertFalse(result["valid"])
        self.assertEqual(result["severity"], "unknown")
        with self.assertRaises(ValueError):
            objective_occlusion_measurement(
                np.zeros((2, 2), dtype=bool),
                np.zeros((3, 3), dtype=bool),
            )

    def test_out_of_amodal_color_spill_is_not_counted_as_target(self) -> None:
        amodal = np.zeros((5, 5), dtype=bool)
        amodal[1:4, 1:4] = True
        visible = amodal.copy()
        visible[0, :] = True
        result = objective_occlusion_measurement(visible, amodal)
        self.assertEqual(result["raw_visible_target_id_pixels"], 14)
        self.assertEqual(result["visible_target_pixels"], 9)
        self.assertEqual(result["out_of_amodal_id_spill_pixels"], 5)
        self.assertEqual(result["occlusion_fraction"], 0.0)

    def test_reference_occlusion_counts_only_newly_revealed_pixels(self) -> None:
        amodal = np.ones((10, 10), dtype=bool)
        visible = amodal.copy()
        visible[:4] = False
        reference_removed = amodal.copy()
        reference_removed[:1] = False
        result = objective_reference_occlusion_measurement(
            visible, reference_removed, amodal
        )
        self.assertEqual(result["reference_revealed_target_pixels"], 30)
        self.assertAlmostEqual(
            result["reference_occlusion_fraction"], 0.3
        )
        self.assertEqual(result["severity"], "partial")

    def test_nonreference_occlusion_is_not_attributed_to_reference(self) -> None:
        amodal = np.ones((10, 10), dtype=bool)
        visible = amodal.copy()
        visible[:3] = False
        reference_removed = visible.copy()
        result = objective_reference_occlusion_measurement(
            visible, reference_removed, amodal
        )
        self.assertEqual(result["reference_occlusion_fraction"], 0.0)
        self.assertEqual(result["severity"], "no")

    def test_objective_camera_relative_behind_uses_geometry_and_overlap(
        self,
    ) -> None:
        target_mask = np.zeros((20, 30), dtype=bool)
        target_mask[6:14, 15:21] = True
        reference_mask = np.zeros_like(target_mask)
        reference_mask[4:17, 8:18] = True
        camera_to_world = np.eye(4)
        camera_to_world[3, :3] = [-1.0, 0.5, 1.0]
        result = objective_camera_relative_behind_measurement(
            target_center_world_m=[1.2, 0.5, 0.1],
            reference_bounds_world_m={
                "lower": [0.0, 0.0, 0.0],
                "upper": [1.0, 1.0, 0.5],
            },
            camera_to_world_row_vector_matrix=camera_to_world.tolist(),
            target_amodal_mask=target_mask,
            reference_visible_mask=reference_mask,
            membership="outside",
            reference_occlusion_fraction=0.0,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["label"], "yes")
        self.assertGreater(result["far_edge_offset_m"], 0.02)
        self.assertGreaterEqual(
            result["target_bbox_overlap_with_reference"], 0.1
        )

    def test_objective_inside_reference_occlusion_counts_as_behind(self) -> None:
        target_mask = np.ones((6, 6), dtype=bool)
        result = objective_camera_relative_behind_measurement(
            target_center_world_m=[0.5, 0.5, 0.1],
            reference_bounds_world_m={
                "lower": [0.0, 0.0, 0.0],
                "upper": [1.0, 1.0, 0.5],
            },
            camera_to_world_row_vector_matrix=np.eye(4).tolist(),
            target_amodal_mask=target_mask,
            reference_visible_mask=target_mask,
            membership="inside",
            reference_occlusion_fraction=0.3,
        )
        self.assertEqual(result["label"], "yes")
        self.assertEqual(
            result["reason"],
            "inside_target_hidden_by_reference_facing_surface",
        )

    def test_objective_behind_abstains_when_target_is_absent(self) -> None:
        empty = np.zeros((6, 6), dtype=bool)
        result = objective_camera_relative_behind_measurement(
            target_center_world_m=[0.5, 0.5, 0.1],
            reference_bounds_world_m={
                "lower": [0.0, 0.0, 0.0],
                "upper": [1.0, 1.0, 0.5],
            },
            camera_to_world_row_vector_matrix=np.eye(4).tolist(),
            target_amodal_mask=empty,
            reference_visible_mask=empty,
            membership="not_applicable",
            reference_occlusion_fraction=None,
        )
        self.assertTrue(result["valid"])
        self.assertEqual(result["label"], "unknown")
        self.assertEqual(
            result["reason"], "target_absent_relation_not_applicable"
        )


if __name__ == "__main__":
    unittest.main()
