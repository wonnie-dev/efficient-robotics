"""Calibrate a center-scene-conditioned future-visibility MPC smoke.

The model is a small distance-weighted nonparametric calibrator.  It uses
only learned center-observation features.  Every reported decision is made
with the entire held-out episode excluded, and the held-out future view is
read only afterward for an audit.  This is development calibration, not
foundation-model training or reserved testing.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "outputs"
    / "perception_grounding_pilot"
    / "action_differentiating_seed185_196"
    / "perception_config.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "offline_mpc"
    / "scene_conditioned_future_belief_seed185_196"
    / "result.json"
)
VARIANTS = (
    "close_high_only",
    "right_only",
    "either_view",
    "cover_removal_required",
)
ACTIONS = (
    "viewpoint_close_high",
    "viewpoint_right",
    "remove_cover",
)
RESOLVING_ACTIONS = {
    "close_high_only": {"viewpoint_close_high"},
    "right_only": {"viewpoint_right"},
    "either_view": {"viewpoint_close_high", "viewpoint_right"},
    "cover_removal_required": {"remove_cover"},
}
ACTION_TO_VIEW = {
    "viewpoint_close_high": "close_high",
    "viewpoint_right": "right",
}
ACTION_COST = {
    "viewpoint_close_high": 0.08,
    "viewpoint_right": 0.06,
    "remove_cover": 0.12,
}
DEFER_COST = 0.45
UNRESOLVED_TASK_COST = 1.0
FEATURE_NAMES = (
    "qwen_selected_raw_match_logit",
    "red_candidate_count",
    "orange_max_aspect_ratio",
    "lid_cover_max_score",
    "lid_cover_max_area_fraction",
)


def load_json(path: Path) -> dict[str, Any]:
    """Load one calibration input artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def bbox_aspect(annotation: dict[str, Any]) -> float:
    """Return the width-to-height ratio of a detected box."""
    x0, y0, x1, y1 = annotation["bbox_xyxy_pixels"]
    return float(x1 - x0) / max(1e-9, float(y1 - y0))


def bbox_area_fraction(annotation: dict[str, Any], image_area: float) -> float:
    """Return the fraction of image area covered by a detected box."""
    x0, y0, x1, y1 = annotation["bbox_xyxy_pixels"]
    return float(x1 - x0) * float(y1 - y0) / image_area


def extract_center_features(
    perception_root: Path,
    seed: int,
    *,
    detection_root: Path | None = None,
) -> dict[str, Any]:
    """Read only the initial view features available before action selection."""
    if detection_root is None:
        detection_root = perception_root
    sample_id = f"seed{seed:03d}_center"
    detections = load_json(
        detection_root / "grounded_sam2" / sample_id / "detections.json"
    )
    ranking = load_json(
        perception_root
        / "grounded_sam2_qwen_rankings"
        / sample_id
        / "result.json"
    )
    selected_index = ranking["candidate_ids"].index(
        ranking["selected_candidate_id"]
    )
    image_path = Path(detections["image_path"])
    from PIL import Image

    with Image.open(image_path) as image:
        image_area = float(image.width * image.height)
    orange = [
        item for item in detections["annotations"]
        if item["label"] == "orange object"
    ]
    covers = [
        item for item in detections["annotations"]
        if item["label"] == "lid or cover"
    ]
    values = [
        float(ranking["raw_match_logits"][selected_index]),
        float(len(ranking["candidate_ids"])),
        max((bbox_aspect(item) for item in orange), default=0.0),
        max((float(item["score"]) for item in covers), default=0.0),
        max(
            (bbox_area_fraction(item, image_area) for item in covers),
            default=0.0,
        ),
    ]
    return {
        "sample_id": sample_id,
        "values": values,
        "named_values": dict(zip(FEATURE_NAMES, values)),
        "inference_inputs": [
            str(
                detection_root
                / "grounded_sam2"
                / sample_id
                / "detections.json"
            ),
            str(
                perception_root
                / "grounded_sam2_qwen_rankings"
                / sample_id
                / "result.json"
            ),
        ],
    }


def standardized_distance(
    left: list[float], right: list[float], mean: list[float], std: list[float]
) -> float:
    """Measure feature distance after per-dimension standardization."""
    return math.sqrt(
        sum(
            ((left[index] - right[index]) / std[index]) ** 2
            for index in range(len(left))
        )
    )


def fit_statistics(rows: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    """Estimate feature means and standard deviations from calibration rows."""
    dimension = len(FEATURE_NAMES)
    mean = [
        sum(row["features"]["values"][index] for row in rows) / len(rows)
        for index in range(dimension)
    ]
    std = []
    for index in range(dimension):
        variance = sum(
            (row["features"]["values"][index] - mean[index]) ** 2
            for row in rows
        ) / len(rows)
        std.append(math.sqrt(variance) or 1.0)
    return mean, std


def predict_variant_distribution(
    query: dict[str, Any], training_rows: list[dict[str, Any]], *, k: int = 3
) -> dict[str, Any]:
    """Estimate scene-type belief from neighboring calibration episodes."""
    if not training_rows:
        raise ValueError("At least one calibration episode is required")
    mean, std = fit_statistics(training_rows)
    neighbors = sorted(
        (
            {
                "seed": row["seed"],
                "variant": row["variant"],
                "distance": standardized_distance(
                    query["values"], row["features"]["values"], mean, std
                ),
            }
            for row in training_rows
        ),
        key=lambda item: item["distance"],
    )[: min(k, len(training_rows))]
    weights = {variant: 0.0 for variant in VARIANTS}
    for neighbor in neighbors:
        weight = 1.0 / (float(neighbor["distance"]) + 0.1)
        neighbor["weight"] = weight
        weights[str(neighbor["variant"])] += weight
    total = sum(weights.values())
    probabilities = {
        variant: weights[variant] / total for variant in VARIANTS
    }
    return {
        "variant_probabilities": probabilities,
        "neighbors": neighbors,
        "training_feature_mean": dict(zip(FEATURE_NAMES, mean)),
        "training_feature_std": dict(zip(FEATURE_NAMES, std)),
    }


def select_action(variant_probabilities: dict[str, float]) -> dict[str, Any]:
    """Trade action cost against the predicted chance of resolving the scene."""
    values = [
        {
            "action": "defer",
            "predicted_resolution_probability": 0.0,
            "objective_cost": DEFER_COST,
        }
    ]
    for action in ACTIONS:
        resolve_probability = sum(
            probability
            for variant, probability in variant_probabilities.items()
            if action in RESOLVING_ACTIONS[variant]
        )
        values.append(
            {
                "action": action,
                "predicted_resolution_probability": resolve_probability,
                "action_cost": ACTION_COST[action],
                "unresolved_task_cost": UNRESOLVED_TASK_COST,
                "objective_cost": (
                    ACTION_COST[action]
                    + (1.0 - resolve_probability) * UNRESOLVED_TASK_COST
                ),
            }
        )
    selected = min(values, key=lambda item: item["objective_cost"])
    return {
        "planner": "scene_conditioned_future_visibility_belief_mpc",
        "horizon": 1,
        "action_values": values,
        "selected_action": selected["action"],
        "selected_cost": selected["objective_cost"],
        "future_observation_used_for_action_selection": False,
    }


def expected_action(variant: str) -> str:
    """Return the view action associated with a scene variant."""
    if variant == "either_view":
        return min(
            RESOLVING_ACTIONS[variant], key=lambda action: ACTION_COST[action]
        )
    return next(iter(RESOLVING_ACTIONS[variant]))


def main() -> None:
    """Fit and evaluate the scene-conditioned future-observation model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    started = time.perf_counter()
    config = load_json(args.config.resolve())
    perception_root = ROOT / config["output_root"]
    replay = config.get("neutral_prompt_replay", {})
    detection_root = ROOT / replay.get(
        "source_root", config["output_root"]
    )
    labels = config["evaluation"]["scene_labels_not_used_for_inference"]
    rows = []
    for seed_text, variant in sorted(labels.items()):
        seed = int(seed_text.removeprefix("seed"))
        rows.append(
            {
                "seed": seed,
                "variant": str(variant),
                "features": extract_center_features(
                    perception_root,
                    seed,
                    detection_root=detection_root,
                ),
            }
        )
    evaluation = load_json(perception_root / "evaluation_summary.json")
    evaluation_by_sample = {
        item["sample_id"]: item
        for item in evaluation["selected_relation_evaluation"]["rows"]
    }

    folds = []
    for held_out in rows:
        training = [row for row in rows if row["seed"] != held_out["seed"]]
        prediction = predict_variant_distribution(
            held_out["features"], training, k=3
        )
        if held_out["seed"] in {
            neighbor["seed"] for neighbor in prediction["neighbors"]
        }:
            raise AssertionError("Held-out seed leaked into its neighbors")
        policy = select_action(prediction["variant_probabilities"])
        selected = str(policy["selected_action"])
        expected = expected_action(held_out["variant"])
        post_action = None
        if selected in ACTION_TO_VIEW:
            view = ACTION_TO_VIEW[selected]
            sample_id = f"seed{held_out['seed']:03d}_{view}"
            row = evaluation_by_sample[sample_id]
            post_action = {
                "sample_id": sample_id,
                "read_only_after_root_action_selection": True,
                "selected_target_correct_posthoc": row[
                    "selected_target_correct_at_0_5"
                ],
                "selected_membership_correct_posthoc": row[
                    "relation_correct"
                ],
                "selected_membership_label": row["top_label"],
            }
        elif selected == "remove_cover":
            post_action = {
                "status": "interaction_selected_but_not_executed",
                "negative_evidence_update_pending": True,
            }
        folds.append(
            {
                "held_out_seed": held_out["seed"],
                "held_out_variant_posthoc": held_out["variant"],
                "training_seeds": [row["seed"] for row in training],
                "center_features": held_out["features"],
                "future_belief_prediction": prediction,
                "root_policy": policy,
                "selected_action": selected,
                "expected_action_posthoc": expected,
                "correct_action": selected == expected,
                "post_action_audit": post_action,
                "held_out_scene_label_used_for_action_selection": False,
                "held_out_future_view_used_for_action_selection": False,
            }
        )

    full_mean, full_std = fit_statistics(rows)
    fixed_action_baselines = {}
    for baseline_action in (*ACTIONS, "defer"):
        correct = sum(
            baseline_action == expected_action(row["variant"])
            for row in rows
        )
        fixed_action_baselines[baseline_action] = {
            "correct_action_count": correct,
            "episode_count": len(rows),
            "action_accuracy": correct / len(rows),
        }
    result = {
        "schema_version": "scene-conditioned-future-belief-calibration-v1",
        "status": "completed",
        "protocol": "leave_one_episode_out_calibration",
        "feature_names": list(FEATURE_NAMES),
        "neighbor_count": 3,
        "distance_weight_offset": 0.1,
        "action_cost": ACTION_COST,
        "defer_cost": DEFER_COST,
        "unresolved_task_cost": UNRESOLVED_TASK_COST,
        "folds": folds,
        "episode_count": len(folds),
        "correct_action_count": sum(fold["correct_action"] for fold in folds),
        "action_accuracy": sum(fold["correct_action"] for fold in folds)
        / len(folds),
        "selected_view_target_recovery": {
            "correct": sum(
                fold["post_action_audit"] is not None
                and fold["post_action_audit"].get(
                    "selected_target_correct_posthoc"
                ) is True
                for fold in folds
            ),
            "view_action_count": sum(
                fold["selected_action"] in ACTION_TO_VIEW for fold in folds
            ),
        },
        "remove_cover_selected_count": sum(
            fold["selected_action"] == "remove_cover" for fold in folds
        ),
        "remove_cover_executed": False,
        "fixed_action_baselines": fixed_action_baselines,
        "frozen_full_calibration_model": {
            "episodes": rows,
            "feature_mean": dict(zip(FEATURE_NAMES, full_mean)),
            "feature_std": dict(zip(FEATURE_NAMES, full_std)),
        },
        "runtime_seconds": time.perf_counter() - started,
        "gpu_used": False,
        "training_performed": False,
        "foundation_model_weights_changed": False,
        "calibration_model_fitting_performed": True,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
        "limitations": [
            "Only eleven development episodes are available.",
            "Scene geometry is intentionally structured and easy to separate.",
            "The calibrated predictor is a pilot, not a final learned observation model.",
            "Cover removal and post-removal negative evidence were not executed.",
        ],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "result": str(output),
                "episode_count": result["episode_count"],
                "correct_action_count": result["correct_action_count"],
                "action_accuracy": result["action_accuracy"],
                "selected_view_target_recovery": result[
                    "selected_view_target_recovery"
                ],
                "remove_cover_selected_count": result[
                    "remove_cover_selected_count"
                ],
                "runtime_seconds": result["runtime_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
