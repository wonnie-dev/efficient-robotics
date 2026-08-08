import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_saved_learned_relation import audit_relation  # noqa: E402


class SavedLearnedRelationAuditTests(unittest.TestCase):
    def test_selected_candidate_outside(self):
        import tempfile

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        observation = tmp_path / "observation"
        observation.mkdir()
        depth = np.ones((20, 20), dtype=np.float32)
        np.save(observation / "depth_m.npy", depth)
        calibration = {
        "camera_to_world_row_vector_matrix": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
            "fx_pixels": 10.0,
            "fy_pixels": 10.0,
            "cx_pixels": 10.0,
            "cy_pixels": 10.0,
        }
        (observation / "camera_calibration.json").write_text(
            json.dumps(calibration)
        )
        reference = np.zeros((20, 20), dtype=np.uint8)
        reference[5:15, 5:15] = 255
        candidate = np.zeros((20, 20), dtype=np.uint8)
        candidate[8:12, 17:20] = 255
        Image.fromarray(reference).save(tmp_path / "reference.png")
        Image.fromarray(candidate).save(tmp_path / "candidate.png")
        model_input = {
        "reference_entities": [{
            "reference_id": "container_001",
            "mask_path": str(tmp_path / "reference.png"),
            "bbox_xyxy": [5, 5, 14, 14],
        }],
        "candidates": [{
            "candidate_id": "candidate_001",
            "mask_path": str(tmp_path / "candidate.png"),
            "bbox_xyxy": [17, 8, 19, 11],
        }],
        }
        ranking = {
        "selected_candidate_id": "candidate_001",
        "selected_candidate_relation": {"top_label": "outside"},
        }
        config = {
        "reference_geometry": {
            "frame_assumption": "world_axis_aligned_simulation_basket",
            "expected_full_extents_xy_m": [1.0, 1.0],
            "robust_percentiles": [0.0, 100.0],
            "minimum_extent_ratio": 0.1,
            "maximum_extent_ratio": 2.0,
            "boundary_abstention_m": 0.01,
            "minimum_valid_depth_pixels": 20,
        },
        "candidate_geometry": {
            "robust_percentiles": [0.0, 100.0],
            "minimum_valid_depth_pixels": 4,
            "known_mug_height_m": 0.1,
        },
        "occlusion_evidence": {
            "mask_dilation_pixels": 1,
            "yes_max_visible_height_ratio": 0.72,
            "yes_min_reference_adjacency": 0.1,
            "no_min_visible_height_ratio": 0.85,
            "no_max_reference_adjacency": 0.03,
        },
        "behind_evidence": {
            "minimum_candidate_bbox_overlap": 0.1,
            "far_edge_abstention_m": 0.02,
        },
        }
        input_path = tmp_path / "input.json"
        ranking_path = tmp_path / "ranking.json"
        config_path = tmp_path / "config.json"
        input_path.write_text(json.dumps(model_input))
        ranking_path.write_text(json.dumps(ranking))
        config_path.write_text(json.dumps(config))

        result = audit_relation(
            observation, input_path, ranking_path, config_path
        )

        self.assertEqual(
            result["rgbd_relation"]["membership_world_evidence"]["label"],
            "outside",
        )
        self.assertFalse(result["simulator_ground_truth_used_for_prediction"])


if __name__ == "__main__":
    unittest.main()
