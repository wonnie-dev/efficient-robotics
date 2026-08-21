import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_grounded_proposal_qwen_ranking import (  # noqa: E402
    RANKING_PROMPT_VERSION,
    cached_result_matches_input,
    sha256 as input_sha256,
)
from run_perception_grounding_pilot import (  # noqa: E402
    cached_result_matches_image,
    sha256,
)


class PerceptionCacheProvenanceTests(unittest.TestCase):
    def test_image_cache_is_invalidated_when_rgb_bytes_change(self) -> None:
        with self.subTest("same path, changed RGB bytes"):
            import tempfile

            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                image = root / "rgb.png"
                result = root / "result.json"
                image.write_bytes(b"first")
                result.write_text(
                    json.dumps(
                        {
                            "image_path": str(image),
                            "image_sha256": sha256(image),
                        }
                    ),
                    encoding="utf-8",
                )
                self.assertTrue(cached_result_matches_image(result, image))

                image.write_bytes(b"second")
                self.assertFalse(cached_result_matches_image(result, image))

    def test_qwen_cache_is_invalidated_when_exported_input_changes(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_input = root / "input.json"
            result = root / "result.json"
            model_input.write_text(
                '{"source_rgb_sha256":"first"}', encoding="utf-8"
            )
            result.write_text(
                json.dumps(
                    {
                        "input_path": str(model_input),
                        "input_sha256": input_sha256(model_input),
                        "prompt_version": RANKING_PROMPT_VERSION,
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(cached_result_matches_input(result, model_input))

            model_input.write_text(
                '{"source_rgb_sha256":"second"}', encoding="utf-8"
            )
            self.assertFalse(cached_result_matches_input(result, model_input))


if __name__ == "__main__":
    unittest.main()
