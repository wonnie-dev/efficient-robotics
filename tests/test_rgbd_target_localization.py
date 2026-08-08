import unittest

import numpy as np

from scripts.rgbd_target_localization import (
    backproject_distance_pixels,
    estimate_instance_center,
    estimate_mask_center,
)


class RgbdTargetLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.calibration = {
            "fx_pixels": 100.0,
            "fy_pixels": 100.0,
            "cx_pixels": 1.5,
            "cy_pixels": 1.5,
            "camera_to_world_row_vector_matrix": np.eye(4).tolist(),
        }

    def test_center_pixel_projects_along_negative_camera_z(self):
        points = backproject_distance_pixels(
            np.asarray([1.5]),
            np.asarray([1.5]),
            np.asarray([2.0]),
            self.calibration,
        )
        np.testing.assert_allclose(points[0], [0.0, 0.0, -2.0])

    def test_row_vector_camera_translation_is_applied(self):
        calibration = dict(self.calibration)
        matrix = np.eye(4)
        matrix[3, :3] = [1.0, 2.0, 3.0]
        calibration["camera_to_world_row_vector_matrix"] = matrix.tolist()
        points = backproject_distance_pixels(
            np.asarray([1.5]),
            np.asarray([1.5]),
            np.asarray([2.0]),
            calibration,
        )
        np.testing.assert_allclose(points[0], [1.0, 2.0, 1.0])

    def test_instance_center_rejects_too_few_pixels(self):
        depth = np.ones((4, 4), dtype=np.float32)
        ids = np.zeros((4, 4), dtype=np.uint32)
        ids[0, 0] = 1
        with self.assertRaises(ValueError):
            estimate_instance_center(depth, ids, 1, self.calibration)

    def test_external_mask_center_uses_no_semantic_instance_id(self):
        depth = np.ones((6, 6), dtype=np.float32)
        mask = np.ones((6, 6), dtype=bool)
        result = estimate_mask_center(
            depth, mask, self.calibration, label="selected_target"
        )
        self.assertEqual(result["mask_label"], "selected_target")
        self.assertNotIn("instance_id", result)
        self.assertFalse(result["simulator_ground_truth_used_for_estimate"])


if __name__ == "__main__":
    unittest.main()
