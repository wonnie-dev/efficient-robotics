"""Tests for benchmark Scene Graph construction."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_benchmark_scene_graphs import build_graph  # noqa: E402


class BenchmarkSceneGraphTests(unittest.TestCase):
    def test_occluder_relation_targets_red_object(self) -> None:
        config = {
            "scene_id": "test",
            "scenario": {},
            "seed": 0,
            "objects": [
                {
                    "id": "occluder_orange",
                    "role": "occluder",
                    "shape": "cylinder",
                    "relation": "in_front_of_target",
                    "prim": "/World/OccluderOrange",
                    "position_m": [0, 0, 0],
                }
            ],
        }
        observation = {
            "visible": True,
            "pixel_count": 10,
            "visible_fraction": 0.1,
            "bbox_xyxy": [0, 0, 1, 1],
            "depth_mean_m": 1.0,
            "depth_min_m": 0.9,
            "depth_max_m": 1.1,
            "instance_ids": [1],
            "depth_valid_pixels": 10,
        }
        graph = build_graph(
            "center",
            config,
            {"container": observation, "occluder_orange": observation},
        )
        self.assertIn(
            {
                "source": "occluder_orange",
                "relation": "occludes",
                "target": "target_red",
                "value": True,
                "source_type": "ground_truth_config",
            },
            graph["edges"],
        )


if __name__ == "__main__":
    unittest.main()
