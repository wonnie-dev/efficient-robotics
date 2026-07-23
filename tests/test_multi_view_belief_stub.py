"""Unit tests for the provisional multi-view belief fusion functions."""

import math
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from update_multi_view_belief_stub import fuse_distributions  # noqa: E402


class MultiViewBeliefStubTests(unittest.TestCase):
    def test_first_observation_is_normalized(self) -> None:
        result = fuse_distributions(None, {"target": 0.7, "other": 0.3}, 1e-6)
        self.assertTrue(math.isclose(sum(result.values()), 1.0))
        self.assertAlmostEqual(result["target"], 0.7)

    def test_consistent_observation_increases_dominant_belief(self) -> None:
        prior = {"target": 0.7, "other": 0.3}
        result = fuse_distributions(prior, {"target": 0.8, "other": 0.2}, 1e-6)
        self.assertGreater(result["target"], prior["target"])
        self.assertTrue(math.isclose(sum(result.values()), 1.0))

    def test_label_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fuse_distributions(
                {"inside": 0.6, "outside": 0.4},
                {"inside": 0.6, "near": 0.4},
                1e-6,
            )


if __name__ == "__main__":
    unittest.main()
