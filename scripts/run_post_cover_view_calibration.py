#!/usr/bin/env python3
"""Cross-validate the post-cover re-observation selector on cached episodes."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from run_scene_conditioned_future_belief_calibration import (
    ACTION_COST,
    DEFER_COST,
    UNRESOLVED_TASK_COST,
    extract_center_features,
    predict_variant_distribution,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/offline_mpc/post_cover_view_calibration_seed224_239/result.json"
)
EPISODES = (
    (224, "close_high_only", "covered_action_center_ambiguous_seed224_225_gpu0"),
    (225, "right_only", "covered_action_center_ambiguous_seed224_225_gpu0"),
    (226, "close_high_only", "covered_action_center_ambiguous_seed226_229_gpu0"),
    (227, "right_only", "covered_action_center_ambiguous_seed226_229_gpu0"),
    (228, "close_high_only", "covered_action_center_ambiguous_seed226_229_gpu0"),
    (229, "right_only", "covered_action_center_ambiguous_seed226_229_gpu0"),
    (230, "close_high_only", "covered_action_center_ambiguous_semantic_seed230_234_gpu0"),
    (231, "right_only", "covered_action_center_ambiguous_right_seed231_235_gpu0"),
    (232, "close_high_only", "covered_action_center_ambiguous_semantic_seed230_234_gpu0"),
    (233, "right_only", "covered_action_center_ambiguous_right_seed231_235_gpu0"),
    (234, "close_high_only", "covered_action_center_ambiguous_semantic_seed230_234_gpu0"),
    (235, "right_only", "covered_action_center_ambiguous_right_seed231_235_gpu0"),
    (236, "close_high_only", "covered_action_center_ambiguous_balanced_seed236_237_gpu0"),
    (237, "right_only", "covered_action_center_ambiguous_balanced_seed236_237_gpu0"),
    (238, "close_high_only", "covered_action_center_ambiguous_balanced_seed238_239_gpu0"),
    (239, "right_only", "covered_action_center_ambiguous_balanced_seed238_239_gpu0"),
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def extract_post_remove_features(perception_root: Path, seed: int) -> dict[str, Any]:
    """Reuse the frozen feature extractor with a post-remove sample alias."""
    source_id = f"seed{seed}_post_remove"
    alias_id = f"seed{seed}_center"
    detection_source = perception_root / "grounded_sam2" / source_id
    ranking_source = (
        perception_root / "grounded_sam2_qwen_rankings" / source_id
    )
    alias_root = perception_root / ".post_remove_feature_alias"
    detection_alias = alias_root / "grounded_sam2" / alias_id
    ranking_alias = alias_root / "grounded_sam2_qwen_rankings" / alias_id
    detection_alias.mkdir(parents=True, exist_ok=True)
    ranking_alias.mkdir(parents=True, exist_ok=True)
    for source, target in (
        (detection_source / "detections.json", detection_alias / "detections.json"),
        (ranking_source / "result.json", ranking_alias / "result.json"),
    ):
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source)
    features = extract_center_features(alias_root, seed)
    features["sample_id"] = source_id
    features["inference_inputs"] = [
        str(detection_source / "detections.json"),
        str(ranking_source / "result.json"),
    ]
    return features


def select_post_cover_action(probabilities: dict[str, float]) -> dict[str, Any]:
    values = [{"action": "defer", "objective_cost": DEFER_COST}]
    for action, variant in (
        ("viewpoint_close_high", "close_high_only"),
        ("viewpoint_right", "right_only"),
    ):
        resolution_probability = float(probabilities.get(variant, 0.0))
        values.append(
            {
                "action": action,
                "predicted_resolution_probability": resolution_probability,
                "objective_cost": ACTION_COST[action]
                + (1.0 - resolution_probability) * UNRESOLVED_TASK_COST,
            }
        )
    selected = min(values, key=lambda item: (item["objective_cost"], item["action"]))
    return {"selected_action": selected["action"], "action_values": values}


def expected_action(variant: str) -> str:
    return {
        "close_high_only": "viewpoint_close_high",
        "right_only": "viewpoint_right",
    }[variant]


def run(output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    rows = []
    for seed, variant, root_name in EPISODES:
        perception_root = ROOT / "outputs/calibration" / root_name
        analysis = load_json(perception_root / "calibration_perception_analysis.json")
        episode = next(item for item in analysis["episode_checks"] if int(item["seed"]) == seed)
        rows.append(
            {
                "seed": seed,
                "variant": variant,
                "features": extract_post_remove_features(perception_root, seed),
                "winning_view_semantically_resolved": episode[
                    "winning_view_semantically_resolved"
                ],
                "losing_view_semantically_resolved": episode[
                    "losing_view_semantically_resolved"
                ],
            }
        )

    folds = []
    for held_out in rows:
        training = [row for row in rows if row["seed"] != held_out["seed"]]
        prediction = predict_variant_distribution(
            held_out["features"], training, k=3
        )
        decision = select_post_cover_action(prediction["variant_probabilities"])
        expected = expected_action(held_out["variant"])
        folds.append(
            {
                "held_out_seed": held_out["seed"],
                "held_out_variant": held_out["variant"],
                "fit_seeds": [row["seed"] for row in training],
                "held_out_seed_used_for_fit": False,
                "prediction": prediction,
                "decision": decision,
                "expected_action": expected,
                "correct": decision["selected_action"] == expected,
            }
        )

    correct = sum(fold["correct"] for fold in folds)
    fixed = {
        action: sum(expected_action(row["variant"]) == action for row in rows)
        / len(rows)
        for action in ("viewpoint_close_high", "viewpoint_right")
    }
    best_fixed = max(fixed.values())
    result = {
        "schema_version": "post-cover-view-calibration-v1",
        "status": "completed",
        "protocol": "leave_one_seed_out_cached_calibration",
        "episode_count": len(rows),
        "calibration_seeds": [row["seed"] for row in rows],
        "class_balance": {
            variant: sum(row["variant"] == variant for row in rows)
            for variant in ("close_high_only", "right_only")
        },
        "folds": folds,
        "correct_action_count": correct,
        "action_accuracy": correct / len(folds),
        "fixed_action_baselines": fixed,
        "best_fixed_action_accuracy": best_fixed,
        "improvement_over_best_fixed": correct / len(folds) - best_fixed,
        "winning_view_resolved_count": sum(
            row["winning_view_semantically_resolved"] for row in rows
        ),
        "strict_action_differentiation_count": sum(
            row["winning_view_semantically_resolved"]
            and not row["losing_view_semantically_resolved"]
            for row in rows
        ),
        "frozen_full_calibration_model": {"episodes": rows, "neighbor_count": 3},
        "gpu_used": False,
        "training_performed": False,
        "calibration_performed": True,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json_atomic(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output.resolve())
    print(
        json.dumps(
            {
                "action_accuracy": result["action_accuracy"],
                "best_fixed_action_accuracy": result[
                    "best_fixed_action_accuracy"
                ],
                "strict_action_differentiation_count": result[
                    "strict_action_differentiation_count"
                ],
                "episode_count": result["episode_count"],
                "output": str(args.output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
