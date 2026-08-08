"""CPU-only checks for relation-preserving benchmark randomization."""

import math
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from seeded_benchmark import generate_layout  # noqa: E402


class SeededBenchmarkTests(unittest.TestCase):
    def test_layout_is_deterministic(self) -> None:
        self.assertEqual(generate_layout(7), generate_layout(7))
        self.assertNotEqual(generate_layout(7), generate_layout(8))

    def test_target_and_inside_distractor_remain_inside(self) -> None:
        for seed in range(20):
            positions = generate_layout(seed)["positions_world_m"]
            for name in ("target_red", "distractor_yellow"):
                x, y, _z = positions[name]
                self.assertGreater(x, 0.14)
                self.assertLess(x, 0.82)
                self.assertGreater(y, -0.15)
                self.assertLess(y, 0.35)

    def test_occluder_stays_in_front_without_center_overlap(self) -> None:
        for seed in range(20):
            positions = generate_layout(seed)["positions_world_m"]
            target = positions["target_red"]
            occluder = positions["occluder_orange"]
            distance = math.dist(target[:2], occluder[:2])
            self.assertGreaterEqual(distance, 0.095)
            self.assertLessEqual(distance, 0.115)
            self.assertLess(occluder[1], target[1])

    def test_relation_specific_regions_are_preserved(self) -> None:
        for seed in range(20):
            positions = generate_layout(seed)["positions_world_m"]
            self.assertLess(positions["distractor_blue"][1], -0.25)
            self.assertGreater(positions["distractor_green"][0], 0.74)
            self.assertGreater(positions["boundary_purple"][0], 0.78)
            self.assertGreater(positions["rear_red_candidate"][1], 0.28)

    def test_negative_seed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generate_layout(-1)


if __name__ == "__main__":
    unittest.main()
