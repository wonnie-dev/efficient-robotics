import copy
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_evaluation_protocol import audit  # noqa: E402


class EvaluationProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(
            (ROOT / "configs/research/final_evaluation_protocol.json").read_text()
        )

    def test_candidate_protocol_is_complete_but_not_open(self) -> None:
        result = audit(self.protocol)
        self.assertEqual(result["structural_failures"], [])
        self.assertEqual(result["reserved_test_episode_count"], 60)
        self.assertEqual(result["scenario_family_count"], 5)
        self.assertEqual(result["status"], "blocked_before_reserved_test")

    def test_frozen_protocol_passes(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["status"] = "frozen_before_untouched_test"
        protocol["reserved_test_launch_authorized"] = True
        result = audit(protocol)
        self.assertEqual(result["status"], "passed")

    def test_information_leak_is_rejected(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["instruction_protocol"]["ground_truth_hidden_from_policy"] = False
        self.assertIn(
            "information_leak_guard_missing:ground_truth_hidden_from_policy",
            audit(protocol)["structural_failures"],
        )


if __name__ == "__main__":
    unittest.main()
