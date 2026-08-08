"""CPU-only tests for the Qwen-to-planner belief adapter."""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_qwen_belief_mpc_replay import (  # noqa: E402
    fuse_planner_beliefs,
    qwen_to_planner_belief,
    weighted_log_belief_update,
)


class QwenBeliefMpcReplayTests(unittest.TestCase):
    def test_anonymous_qwen_output_maps_to_debug_hypotheses(self) -> None:
        output = {
            "target": {
                "candidate_ids": ["object_001", "object_007"],
                "raw_logits": [2.0, 0.0],
            },
            "relations": [
                {
                    "query_id": "object_001_to_container",
                    "labels": [
                        "inside",
                        "outside",
                        "behind",
                        "near_boundary",
                        "unknown",
                    ],
                    "raw_logits": [3.0, 0.0, -1.0, -2.0, -3.0],
                }
            ],
        }
        belief = qwen_to_planner_belief(output)
        self.assertGreater(
            belief["target"]["target_red"],
            belief["target"]["rear_red_candidate"],
        )
        self.assertGreater(
            belief["relation"]["inside"], belief["relation"]["behind"]
        )
        self.assertAlmostEqual(sum(belief["target"].values()), 1.0)
        self.assertAlmostEqual(sum(belief["relation"].values()), 1.0)

    def test_product_fusion_is_normalized(self) -> None:
        fused = fuse_planner_beliefs(
            {
                "target": {"target_red": 0.6, "rear_red_candidate": 0.4},
                "relation": {"inside": 0.5, "behind": 0.4, "unknown": 0.1},
            },
            {
                "target": {"target_red": 0.8, "rear_red_candidate": 0.2},
                "relation": {"inside": 0.9, "behind": 0.05, "unknown": 0.05},
            },
        )
        self.assertAlmostEqual(sum(fused["target"].values()), 1.0)
        self.assertAlmostEqual(sum(fused["relation"].values()), 1.0)
        self.assertGreater(fused["target"]["target_red"], 0.8)
        self.assertGreater(fused["relation"]["inside"], 0.9)

    def test_downweighted_update_is_less_confident_than_product(self) -> None:
        prior = {
            "target": {"target_red": 0.6, "rear_red_candidate": 0.4},
            "relation": {"inside": 0.5, "behind": 0.4, "unknown": 0.1},
        }
        observation = {
            "target": {"target_red": 0.99, "rear_red_candidate": 0.01},
            "relation": {"inside": 0.98, "behind": 0.01, "unknown": 0.01},
        }
        full = weighted_log_belief_update(prior, observation, 1.0)
        tempered = weighted_log_belief_update(prior, observation, 0.5)
        self.assertLess(
            tempered["target"]["target_red"], full["target"]["target_red"]
        )
        self.assertLess(
            tempered["relation"]["inside"], full["relation"]["inside"]
        )


if __name__ == "__main__":
    unittest.main()
