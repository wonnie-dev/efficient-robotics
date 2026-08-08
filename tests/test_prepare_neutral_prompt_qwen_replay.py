import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from prepare_neutral_prompt_qwen_replay import prepare  # noqa: E402


class NeutralPromptReplayTests(unittest.TestCase):
    def test_preparation_rewrites_text_without_copying_assets(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "outputs") as temporary:
            root = Path(temporary)
            source = root / "source"
            sample = source / "grounded_sam2_qwen_inputs" / "sample_001"
            sample.mkdir(parents=True)
            input_path = sample / "input.json"
            input_path.write_text(
                json.dumps(
                    {
                        "instruction": "Find the mug inside the basket.",
                        "target_description": "mug inside basket",
                        "image": {"rgb_path": "unchanged.png"},
                    }
                )
            )
            (source / "grounded_sam2_qwen_inputs" / "manifest.json").write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "sample_id": "sample_001",
                                "input_path": str(input_path),
                                "candidate_count": 1,
                            }
                        ]
                    }
                )
            )
            (source / "perception_config.json").write_text(
                json.dumps(
                    {
                        "experiment_id": "source",
                        "output_root": str(source),
                        "task": {
                            "instruction": "old",
                            "target_description": "old",
                            "qwen_direct_prompt": "old",
                        },
                    }
                )
            )
            destination = root / "destination"
            result = prepare(source, destination)
            rewritten = json.loads(
                (
                    destination
                    / "grounded_sam2_qwen_inputs"
                    / "sample_001"
                    / "input.json"
                ).read_text()
            )
            self.assertEqual(result["sample_count"], 1)
            self.assertEqual(rewritten["image"]["rgb_path"], "unchanged.png")
            self.assertNotIn("inside", rewritten["instruction"].lower())


if __name__ == "__main__":
    unittest.main()
