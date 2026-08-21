"""Split and scheduling contracts for development calibration."""

from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CalibrationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = json.loads(
            (ROOT / "configs/research/unified_calibration_episodes.json")
            .read_text(encoding="utf-8")
        )

    def test_split_stops_before_the_untouched_test_range(self) -> None:
        first = self.config["seed_start"]
        last = first + self.config["episode_count"] - 1
        self.assertEqual((first, last), (1064, 1099))
        self.assertLess(last, 1100)

    def test_six_balanced_families_are_declared(self) -> None:
        families = self.config["family_cycle"]
        self.assertEqual(len(families), 6)
        self.assertEqual(self.config["episode_count"], 36)
        self.assertEqual(self.config["episodes_per_family"], 6)
        self.assertEqual(len({row["family"] for row in families}), 6)

    def test_outside_ambiguity_is_close_high_resolving(self) -> None:
        row = next(
            item
            for item in self.config["family_cycle"]
            if item["family"] == "outside_reobservation_required"
        )
        self.assertEqual(row["resolving_view"], "close_high")
        self.assertEqual(row["blocked_view"], "center")

    def test_no_training_or_test_data_use(self) -> None:
        self.assertFalse(self.config["training_performed"])
        self.assertFalse(self.config["calibration_performed"])
        self.assertFalse(self.config["testing_performed"])
        self.assertFalse(self.config["valid_for_final_evaluation"])
        self.assertTrue(self.config["calibration_data_collection"])


if __name__ == "__main__":
    unittest.main()
