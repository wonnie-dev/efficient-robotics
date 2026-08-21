import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_icra_v15b_scene_conditioned_view_calibration import (  # noqa: E402
    bbox_features,
    view_mode,
)


class V15bSceneConditionedViewCalibrationTests(unittest.TestCase):
    def test_v20_headline_families_have_post_remove_labels(self):
        self.assertEqual(view_mode("covered_close_high_resolving"), "close_high")
        self.assertEqual(view_mode("covered_right_resolving"), "right")
        self.assertEqual(view_mode("target_absent_covered"), "none")

    def test_bbox_features_are_normalized(self):
        self.assertEqual(bbox_features([0, 0, 50, 50], 100, 100), (0.25, 0.25, 0.25))

    def test_new_hard_outside_requires_right_view(self):
        self.assertEqual(view_mode("outside_other_target_right_resolving"), "right")


if __name__ == "__main__":
    unittest.main()
