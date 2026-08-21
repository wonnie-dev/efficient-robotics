import json
import sys
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_icra_v15_calibration_capture_batch import (
    assignments,
    authorize_reserved_test,
)


class FiveScenarioProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(
            (ROOT / "configs/research/icra_v20_five_scenario_final_evaluation_protocol.json").read_text()
        )
        cls.capture = json.loads(
            (ROOT / "configs/research/icra_v20_reserved_test_60episode.json").read_text()
        )
        cls.rows = assignments(cls.capture)

    def test_reserved_split_is_still_closed(self) -> None:
        self.assertEqual(self.capture["status"], "blocked_until_method_and_calibration_freeze")
        self.assertFalse(self.capture["reserved_test_opened"])
        self.assertFalse(self.capture["launch_authorized"])
        self.assertFalse(self.capture["reserved_test_seeds_used"])
        self.assertFalse(self.protocol["reserved_test_launch_authorized"])

    def test_reserved_split_cannot_bypass_freeze_guard(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "freeze step"):
            authorize_reserved_test(self.capture, confirm_open=True)
        with self.assertRaisesRegex(RuntimeError, "confirm-open"):
            authorize_reserved_test(self.capture, confirm_open=False)

    def test_seed_and_gpu_policy(self) -> None:
        self.assertEqual([row["seed"] for row in self.rows], list(range(1100, 1160)))
        self.assertEqual(self.capture["gpu_ids"], [0, 2, 4, 5])
        self.assertEqual({row["physical_gpu"] for row in self.rows}, {0, 2, 4, 5})

    def test_headline_scenarios_are_balanced(self) -> None:
        counts = Counter(row["headline_scenario"] for row in self.rows)
        self.assertEqual(
            counts,
            Counter(
                {
                    "visible_open": 12,
                    "partially_occluded": 12,
                    "covered_container": 12,
                    "ambiguous_inside_outside": 12,
                    "target_absent": 12,
                }
            ),
        )

    def test_family_counts_match_predeclaration(self) -> None:
        counts = Counter(row["family"] for row in self.rows)
        self.assertEqual(counts, Counter(self.capture["expected_family_counts"]))

    def test_policy_does_not_receive_the_answer(self) -> None:
        instruction = self.protocol["instruction_protocol"]
        self.assertTrue(instruction["scenario_label_hidden_from_policy"])
        self.assertTrue(instruction["ground_truth_hidden_from_policy"])
        self.assertTrue(instruction["required_action_hidden_from_policy"])
        self.assertEqual(
            set(self.protocol["action_set"]),
            {"viewpoint_right", "viewpoint_close_high", "remove_cover", "grasp", "defer"},
        )

    def test_development_choices_are_recorded_before_test(self) -> None:
        record = json.loads(
            (ROOT / self.protocol["development_selection_record"]).read_text()
        )
        self.assertEqual(record["selection_data_split"], "development_only")
        self.assertFalse(record["reserved_test_seed_range_opened"])
        self.assertTrue(
            record["decisions"]["direct_vlm_baseline"][
                "retained_for_final_comparison"
            ]
        )

    def test_nonreserved_preflight_is_balanced(self) -> None:
        development = json.loads(
            (ROOT / "configs/research/icra_v20_five_scenario_development_20episode.json").read_text()
        )
        rows = assignments(development)
        self.assertEqual([row["seed"] for row in rows], list(range(1334, 1354)))
        self.assertTrue(set(range(1100, 1160)).isdisjoint(row["seed"] for row in rows))
        self.assertEqual(
            Counter(row["headline_scenario"] for row in rows),
            Counter(
                {
                    "visible_open": 4,
                    "partially_occluded": 4,
                    "covered_container": 4,
                    "ambiguous_inside_outside": 4,
                    "target_absent": 4,
                }
            ),
        )


if __name__ == "__main__":
    unittest.main()
