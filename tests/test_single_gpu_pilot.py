"""CPU-only tests for the single-GPU pilot belief adapter."""

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_single_gpu_pilot import (  # noqa: E402
    cache_request,
    fuse_distributions,
    output_belief,
    select_action,
    softmax,
)


class SingleGpuPilotTests(unittest.TestCase):
    def test_cache_identity_ignores_session_ids_and_asset_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            keys = []
            for index in range(2):
                sample_root = root / f"run_{index}"
                sample_root.mkdir()
                for name in ("rgb.png", "crop.png", "mask.png"):
                    (sample_root / name).write_bytes(b"same-content-" + name.encode())
                payload = {
                    "sample_id": f"session_{index}_center",
                    "episode_id": f"session_{index}",
                    "image": {"rgb_path": "rgb.png"},
                    "candidates": [
                        {
                            "candidate_id": "object_001",
                            "crop_path": "crop.png",
                            "mask_path": "mask.png",
                        }
                    ],
                    "reference_entities": [],
                    "instruction": "find the target",
                }
                input_path = sample_root / "vlm_input.json"
                input_path.write_text(json.dumps(payload), encoding="utf-8")
                key, request = cache_request(
                    input_path, root / "model", 1234
                )
                keys.append((key, request))
            self.assertEqual(keys[0][0], keys[1][0])
            self.assertNotEqual(
                keys[0][1]["input_sha256"], keys[1][1]["input_sha256"]
            )

    def test_softmax_is_normalized(self) -> None:
        probabilities = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertEqual(probabilities.index(max(probabilities)), 2)

    def test_product_fusion_strengthens_agreement(self) -> None:
        fused = fuse_distributions(
            {"a": 0.7, "b": 0.3},
            {"a": 0.8, "b": 0.2},
        )
        self.assertGreater(fused["a"], 0.8)
        self.assertAlmostEqual(sum(fused.values()), 1.0)

    def test_low_inside_confidence_requests_right_view(self) -> None:
        belief = {
            "target": {"object_001": 0.9, "object_002": 0.1},
            "relations": {
                "object_001_to_container": {
                    "inside": 0.2,
                    "outside": 0.8,
                }
            },
        }
        action = select_action(
            belief,
            current_view="center",
            available_views={"left", "center", "right"},
            visited_views={"center"},
        )
        self.assertEqual(action["type"], "viewpoint_right")

    def test_high_joint_confidence_requests_grasp(self) -> None:
        belief = {
            "target": {"object_001": 0.9, "object_002": 0.1},
            "relations": {
                "object_001_to_container": {
                    "inside": 0.8,
                    "outside": 0.2,
                }
            },
        }
        action = select_action(
            belief,
            current_view="right",
            available_views={"left", "center", "right"},
            visited_views={"center", "right"},
        )
        self.assertEqual(action["type"], "grasp")

    def test_output_adapter_does_not_require_ground_truth(self) -> None:
        output = {
            "target": {
                "candidate_ids": ["object_001", "object_002"],
                "raw_logits": [2.0, 1.0],
            },
            "relations": [
                {
                    "query_id": "object_001_to_container",
                    "labels": ["inside", "outside"],
                    "raw_logits": [3.0, 1.0],
                }
            ],
        }
        belief = output_belief(output)
        self.assertFalse(belief["calibrated"])
        self.assertGreater(belief["target"]["object_001"], 0.5)


if __name__ == "__main__":
    unittest.main()
