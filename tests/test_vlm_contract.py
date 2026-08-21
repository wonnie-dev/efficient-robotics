"""Tests for anonymous VLM interchange files."""

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_vlm_dataset import relation_queries  # noqa: E402
from synthetic_vlm_output import build_synthetic_output  # noqa: E402
from validate_vlm_contract import validate_contract  # noqa: E402


class VlmContractTests(unittest.TestCase):
    def sample_input(self) -> dict:
        return {
            "schema_version": "vlm-input-v1",
            "sample_id": "sample",
            "episode_id": "episode",
            "view_id": "center",
            "instruction": "Retrieve the red object inside the container.",
            "image": {"rgb_path": "rgb.png", "width": 640, "height": 480},
            "candidates": [
                {
                    "candidate_id": "object_001",
                    "bbox_xyxy": [0, 0, 10, 10],
                    "crop_path": "crop.png",
                    "mask_path": "mask.png",
                },
                {
                    "candidate_id": "object_002",
                    "bbox_xyxy": [20, 20, 30, 30],
                    "crop_path": "crop2.png",
                    "mask_path": "mask2.png",
                },
            ],
            "relation_queries": [
                {
                    "query_id": "q1",
                    "source_id": "object_001",
                    "target_id": "container_001",
                    "label_space": ["inside", "outside", "unknown"],
                }
            ],
        }

    def test_synthetic_output_matches_contract(self) -> None:
        model_input = self.sample_input()
        validate_contract(model_input, build_synthetic_output(model_input))

    def test_semantic_id_leakage_is_rejected(self) -> None:
        model_input = self.sample_input()
        model_input["candidates"][0]["candidate_id"] = "target_red"
        with self.assertRaises(ValueError):
            validate_contract(model_input, build_synthetic_output(model_input))

    def test_relation_queries_use_anonymous_ids(self) -> None:
        serialized = json.dumps(relation_queries())
        self.assertNotIn("target_red", serialized)
        self.assertNotIn("occluder_orange", serialized)


if __name__ == "__main__":
    unittest.main()
