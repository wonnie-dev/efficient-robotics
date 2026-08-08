from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_qwen_multiview_postaction_ablation import (  # noqa: E402
    configured_pairs,
    evaluate_results,
    prefixed_model_input,
)


class MultiViewPostactionAblationTest(unittest.TestCase):
    def test_prefixed_model_input_keeps_observations_unambiguous(self) -> None:
        original = {
            "candidates": [{"candidate_id": "candidate_001"}],
            "reference_entities": [{"reference_id": "container_001"}],
            "relation_queries": [
                {
                    "query_id": "candidate_001_membership",
                    "source_id": "candidate_001",
                    "target_id": "container_001",
                }
            ],
        }
        result = prefixed_model_input(original, "current")
        self.assertEqual(original["candidates"][0]["candidate_id"], "candidate_001")
        self.assertEqual(
            result["candidates"][0]["candidate_id"], "current_candidate_001"
        )
        self.assertEqual(
            result["reference_entities"][0]["reference_id"],
            "current_container_001",
        )
        self.assertEqual(
            result["relation_queries"][0]["source_id"], "current_candidate_001"
        )
        self.assertEqual(
            result["relation_queries"][0]["target_id"], "current_container_001"
        )

    def test_configured_pairs_do_not_add_unlisted_seeds(self) -> None:
        config = {
            "development_seeds": [165, 168],
            "historical_view": "center",
            "post_action_views": ["close_high", "right"],
        }
        self.assertEqual(
            configured_pairs(config),
            [
                {
                    "seed": 165,
                    "historical_view": "center",
                    "current_view": "close_high",
                },
                {
                    "seed": 165,
                    "historical_view": "center",
                    "current_view": "right",
                },
                {
                    "seed": 168,
                    "historical_view": "center",
                    "current_view": "close_high",
                },
                {
                    "seed": 168,
                    "historical_view": "center",
                    "current_view": "right",
                },
            ],
        )

    def test_evaluation_excludes_target_proposal_missing_from_accuracy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            tmp_path = Path(temporary_directory)
            records = {
                "records": [
                    {
                        "seed": 165,
                        "view": "right",
                        "calibration_scene_variant": "inside_clear",
                        "candidates": [
                            {
                                "candidate_id": "candidate_001",
                                "target_label": True,
                                "relation_ground_truth": {
                                    "membership": "inside",
                                    "occluded_by": "no",
                                },
                            }
                        ],
                    },
                    {
                        "seed": 168,
                        "view": "right",
                        "calibration_scene_variant": "covered_unknown",
                        "candidates": [
                            {
                                "candidate_id": "candidate_001",
                                "target_label": False,
                                "relation_ground_truth": {
                                    "membership": "outside",
                                    "occluded_by": "no",
                                },
                            }
                        ],
                    },
                ]
            }
            records_path = tmp_path / "records.json"
            records_path.write_text(json.dumps(records), encoding="utf-8")
            single_root = tmp_path / "single"
            for seed in (165, 168):
                destination = single_root / f"seed{seed}_right"
                destination.mkdir(parents=True)
                (destination / "result.json").write_text(
                    json.dumps(
                        {
                            "selected_candidate_id": "candidate_001",
                            "relations": [
                                {
                                    "source_id": "candidate_001",
                                    "relation_type": "membership",
                                    "top_label": "inside",
                                },
                                {
                                    "source_id": "candidate_001",
                                    "relation_type": "occluded_by",
                                    "top_label": "no",
                                },
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
            results = [
                {
                    "seed": seed,
                    "historical_view": "center",
                    "current_view": "right",
                    "selected_candidate_id": "candidate_001",
                    "relations": [
                        {
                            "candidate_id": "candidate_001",
                            "relation_type": "membership",
                            "labels": ["inside", "outside", "unknown"],
                            "raw_logits": [3.0, 1.0, 0.0],
                        },
                        {
                            "candidate_id": "candidate_001",
                            "relation_type": "occluded_by",
                            "labels": ["yes", "no"],
                            "raw_logits": [0.0, 2.0],
                        },
                    ],
                }
                for seed in (165, 168)
            ]
            evaluation = evaluate_results(results, records_path, single_root)
            self.assertEqual(evaluation["pair_count"], 2)
            self.assertEqual(evaluation["target_visible_pair_count"], 1)
            self.assertEqual(evaluation["target_proposal_missing_pair_count"], 1)
            self.assertEqual(
                evaluation["metrics"]["multi_target_selection"],
                {"correct": 1, "evaluated": 1, "accuracy": 1.0},
            )


if __name__ == "__main__":
    unittest.main()
