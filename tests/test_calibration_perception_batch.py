"""CPU-only checks for the sharded perception launcher."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_calibration_perception_batch import (  # noqa: E402
    completed_shard,
)


class CalibrationPerceptionBatchTests(unittest.TestCase):
    def test_completed_shard_requires_every_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "COMPLETED.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "sample_count": 2,
                    }
                ),
                encoding="utf-8",
            )
            first = root / "grounded_sam2_qwen_rankings" / "sample_a"
            first.mkdir(parents=True)
            (first / "result.json").write_text("{}", encoding="utf-8")
            self.assertFalse(completed_shard(root))
            second = root / "grounded_sam2_qwen_rankings" / "sample_b"
            second.mkdir(parents=True)
            (second / "result.json").write_text("{}", encoding="utf-8")
            self.assertTrue(completed_shard(root))


if __name__ == "__main__":
    unittest.main()
