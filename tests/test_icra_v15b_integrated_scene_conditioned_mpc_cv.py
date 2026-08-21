import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_icra_v15b_integrated_scene_conditioned_mpc_cv import (  # noqa: E402
    NO_TARGET_EVIDENCE,
    semantic_observation,
)


class IntegratedSceneConditionedMpcCvTests(unittest.TestCase):
    def test_all_nonmatches_are_negative_evidence(self):
        row = {
            "center_track_candidate_id": "candidate_001",
            "candidate_evidence": {
                "candidate_001": {"raw_match_logit": -2.0, "membership": "outside"},
                "candidate_002": {"raw_match_logit": -1.0, "membership": "inside"},
            },
        }
        self.assertEqual(semantic_observation(row), NO_TARGET_EVIDENCE)

    def test_match_keeps_persistent_identity_and_membership(self):
        row = {
            "center_track_candidate_id": "candidate_001",
            "candidate_evidence": {
                "candidate_001": {"raw_match_logit": -2.0, "membership": "outside"},
                "candidate_002": {"raw_match_logit": 3.0, "membership": "inside"},
            },
        }
        self.assertEqual(semantic_observation(row), "other_target|inside")


if __name__ == "__main__":
    unittest.main()
