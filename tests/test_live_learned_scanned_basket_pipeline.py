import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_live_learned_scanned_basket_pipeline import (  # noqa: E402
    grounded_qwen_cache_key,
    make_perception_config,
    mask_iou,
    partial_ranking_belief,
    partial_track_mapping,
    selected_target_mask,
)
from scanned_basket_scene import (  # noqa: E402
    BASKET_COLLISION_BOXES_LOCAL_M,
    PHYSICS_CLEARANCE_BASKET_SCALE_XYZ,
    PHYSICS_CLEARANCE_OUTSIDE_MUG_POSITION_WORLD_M,
    PERCEPTION_BASKET_SCALE_XYZ,
)


class LiveLearnedScannedBasketPipelineTest(unittest.TestCase):
    def test_incremental_config_uses_no_removed_occluder_concept(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as temporary:
            session = Path(temporary)
            observation = session / "observations" / "center"
            observation.mkdir(parents=True)
            path = make_perception_config(
                session_dir=session,
                observation_dir=observation,
                sample_id="seed000_center",
                step_index=0,
            )
            config = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(config["samples"]), 1)
            self.assertNotIn(
                "orange cylinder",
                config["task"]["open_vocabulary_concepts"],
            )
            self.assertEqual(
                config["task"]["minimum_candidate_proposals"],
                1,
            )
            self.assertFalse(config["training_performed"])
            self.assertFalse(config["calibration_performed"])

    def test_selected_target_mask_uses_track_mapping(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as temporary:
            root = Path(temporary)
            mask = root / "candidate_002_mask.png"
            mask.touch()
            input_path = root / "input.json"
            input_path.write_text(
                json.dumps(
                    {
                        "candidates": [
                            {
                                "candidate_id": "candidate_001",
                                "mask_path": str(root / "candidate_001_mask.png"),
                            },
                            {
                                "candidate_id": "candidate_002",
                                "mask_path": str(mask),
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            selected = selected_target_mask(
                {"input_path": str(input_path)},
                {
                    "candidate_001": "track_001",
                    "candidate_002": "track_002",
                },
                "track_002",
            )
            self.assertEqual(selected, mask.resolve())

    def test_physics_basket_matches_perception_and_outside_mug_clears_wall(self):
        self.assertEqual(
            PHYSICS_CLEARANCE_BASKET_SCALE_XYZ,
            PERCEPTION_BASKET_SCALE_XYZ,
        )
        right_wall = next(
            box
            for box in BASKET_COLLISION_BOXES_LOCAL_M
            if box["name"] == "WallRight"
        )
        scale = PHYSICS_CLEARANCE_BASKET_SCALE_XYZ[0] / 2.10
        wall_outer_x = (
            0.48
            + right_wall["center"][0] * scale
            + 0.5 * right_wall["full_extents"][0] * scale
        )
        mug_inner_x = PHYSICS_CLEARANCE_OUTSIDE_MUG_POSITION_WORLD_M[0] - 0.041
        self.assertGreater(mug_inner_x - wall_outer_x, 0.01)

    def test_partial_tracking_retains_unobserved_track(self):
        mapping, details = partial_track_mapping(
            {
                "candidate_001": [0.0, 0.0, 0.0],
                "candidate_002": [1.0, 0.0, 0.0],
            },
            {"candidate_001": [0.01, 0.0, 0.0]},
        )
        self.assertEqual(mapping, {"candidate_001": "track_001"})
        self.assertEqual(details["unobserved_tracks"], ["track_002"])

    def test_partial_tracking_rejects_distant_duplicate(self):
        mapping, details = partial_track_mapping(
            {
                "candidate_001": [0.0, 0.0, 0.0],
                "candidate_002": [1.0, 0.0, 0.0],
            },
            {
                "candidate_001": [0.40, 0.0, 0.0],
                "candidate_002": [0.01, 0.0, 0.0],
            },
            maximum_track_distance_m=0.08,
        )
        self.assertEqual(mapping, {"candidate_002": "track_001"})
        self.assertEqual(
            details["unmatched_current_candidates"],
            ["candidate_001"],
        )
        self.assertEqual(details["unobserved_tracks"], ["track_002"])

    def test_partial_belief_assigns_zero_log_evidence_to_unseen_track(self):
        ranking = {
            "candidate_ids": ["candidate_001"],
            "raw_match_logits": [4.0],
            "selected_candidate_id": "candidate_001",
            "selected_candidate_relation": {
                "labels": ["inside", "outside", "behind", "unknown"],
                "raw_logits": [4.0, 0.0, 0.0, 0.0],
            },
        }
        belief = partial_ranking_belief(
            ranking,
            {"candidate_001": "track_001"},
            ["track_001", "track_002"],
            4.0,
        )
        expected = math.exp(1.0) / (math.exp(1.0) + 1.0)
        self.assertAlmostEqual(belief["target"]["track_001"], expected)
        self.assertAlmostEqual(
            belief["target"]["track_001"]
            + belief["target"]["track_002"],
            1.0,
        )

    def test_mask_iou_rejects_duplicate_semantic_proposal(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as temporary:
            root = Path(temporary)
            target = np.zeros((8, 8), dtype=np.uint8)
            target[1:4, 1:4] = 255
            duplicate = target.copy()
            separate = np.zeros_like(target)
            separate[5:8, 5:8] = 255
            target_path = root / "target.png"
            duplicate_path = root / "duplicate.png"
            separate_path = root / "separate.png"
            Image.fromarray(target).save(target_path)
            Image.fromarray(duplicate).save(duplicate_path)
            Image.fromarray(separate).save(separate_path)
            self.assertEqual(mask_iou(target_path, duplicate_path), 1.0)
            self.assertEqual(mask_iou(target_path, separate_path), 0.0)

    def test_grounded_qwen_cache_ignores_episode_identifiers(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as temporary:
            root = Path(temporary)
            assets = {}
            for name in ("rgb", "crop", "mask", "context", "ref", "overlay"):
                path = root / f"{name}.png"
                path.write_bytes(name.encode("utf-8"))
                assets[name] = str(path)
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "models": {
                            "qwen": {
                                "repository": "Qwen/Qwen3-VL-8B-Instruct",
                                "revision": "test",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            def write_input(sample_id: str) -> Path:
                path = root / f"{sample_id}.json"
                path.write_text(
                    json.dumps(
                        {
                            "schema_version": "vlm-input-v1",
                            "sample_id": sample_id,
                            "episode_id": sample_id,
                            "view_id": "center",
                            "instruction": "find target",
                            "target_description": "target",
                            "image": {"rgb_path": assets["rgb"]},
                            "candidates": [
                                {
                                    "candidate_id": "candidate_001",
                                    "bbox_xyxy": [0, 0, 1, 1],
                                    "crop_path": assets["crop"],
                                    "mask_path": assets["mask"],
                                    "context_path": assets["context"],
                                }
                            ],
                            "reference_entities": [
                                {
                                    "reference_id": "container_001",
                                    "bbox_xyxy": [0, 0, 1, 1],
                                    "mask_path": assets["ref"],
                                    "overlay_path": assets["overlay"],
                                }
                            ],
                            "relation_queries": [],
                        }
                    ),
                    encoding="utf-8",
                )
                return path

            first = grounded_qwen_cache_key(
                write_input("episode_a"), config_path
            )
            second = grounded_qwen_cache_key(
                write_input("episode_b"), config_path
            )
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
