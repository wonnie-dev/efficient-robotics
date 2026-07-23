"""Tests for the provisional benchmark uncertainty and view-selection pipeline."""

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_benchmark_uncertainty_graphs import (  # noqa: E402
    confidence_from_pixels,
    normalize,
)
from run_benchmark_active_view_stub import task_risk  # noqa: E402


class BenchmarkUncertaintyPipelineTests(unittest.TestCase):
    def test_normalize_sums_to_one(self) -> None:
        distribution = normalize({"a": 2.0, "b": 1.0}, 1e-6)
        self.assertAlmostEqual(sum(distribution.values()), 1.0)
        self.assertGreater(distribution["a"], distribution["b"])

    def test_more_pixels_increase_confidence(self) -> None:
        self.assertGreater(
            confidence_from_pixels(600, 300), confidence_from_pixels(45, 300)
        )

    def test_task_risk_falls_with_better_target_and_relation(self) -> None:
        low = task_risk({"target": 0.5}, {"inside": 0.6}, "target")
        high = task_risk({"target": 0.9}, {"inside": 0.9}, "target")
        self.assertLess(high, low)


if __name__ == "__main__":
    unittest.main()
