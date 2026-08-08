"""Summarize the seed-0 household active re-observation perception pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "perception" / "household_mug_basket_seed000.json"
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_root = resolve(config["output_root"])
    evaluation = json.loads(
        (output_root / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    relation_rows = {
        row["sample_id"]: row
        for row in evaluation["selected_relation_evaluation"]["rows"]
    }

    views = []
    for sample in config["samples"]:
        sample_id = sample["sample_id"]
        ranking = json.loads(
            (
                output_root
                / "grounded_sam2_qwen_rankings"
                / sample_id
                / "result.json"
            ).read_text(encoding="utf-8")
        )
        selected_index = ranking["candidate_ids"].index(
            ranking["selected_candidate_id"]
        )
        relation = relation_rows[sample_id]
        views.append(
            {
                "sample_id": sample_id,
                "view_id": sample_id.split("_", 1)[1],
                "selected_candidate_id": ranking["selected_candidate_id"],
                "selected_identity_logit_difference": ranking[
                    "raw_match_logits"
                ][selected_index],
                "selected_relation": relation["top_label"],
                "target_selection_correct_at_mask_iou_0_5": relation[
                    "selected_target_correct_at_0_5"
                ],
                "selected_target_mask_iou": relation[
                    "selected_target_mask_iou"
                ],
                "relation_correct": relation["relation_correct"],
            }
        )

    initial = views[0]
    final = views[-1]
    all_target_selections_correct = all(
        view["target_selection_correct_at_mask_iou_0_5"]
        for view in views
    )
    all_relations_correct = all(view["relation_correct"] for view in views)
    summary = {
        "schema_version": "household-reobservation-pilot-summary-v2",
        "experiment_id": config["experiment_id"],
        "scenario": (
            "red mug with white logo inside an open basket, with a red mug "
            "distractor outside"
        ),
        "observation_sequence": [view["view_id"] for view in views],
        "initial_observation": initial,
        "reobservations": views[1:],
        "final_observation": final,
        "viewpoint_execution": {
            "selection_method": (
                "fixed seed-0 validation sequence, not belief-space MPC"
            ),
            "motion_result": config.get("motion_result"),
        },
        "perception_outcome": (
            "correct_target_mask_and_inside_relation_in_all_views"
            if all_target_selections_correct and all_relations_correct
            else "one_or_more_views_failed_target_or_relation_validation"
        ),
        "grasp_commitment_allowed": False,
        "grasp_commitment_blocker": (
            "Qwen logits are uncalibrated and the scanned basket currently "
            "has visual geometry only, so this perception pilot cannot "
            "authorize a physical grasp."
        ),
        "excluded_views": config["excluded_observations"],
        "training_performed": False,
        "calibration_performed": False,
        "simulator_ground_truth_used_for_inference": False,
        "valid_for_final_evaluation": False,
    }
    destination = output_root / "reobservation_summary.json"
    destination.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
