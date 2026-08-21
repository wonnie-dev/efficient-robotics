#!/usr/bin/env python3
"""Runtime helpers for the V14 scene-conditioned development controller."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any


FEATURE_NAMES = (
    "qwen_selected_raw_match_logit",
    "candidate_count",
    "rgbd_inside_margin_m",
    "selected_bbox_center_x_px",
    "selected_bbox_center_y_px",
)
VIEW_MODES = ("close_high", "right", "none")


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON artifact from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def extract_live_post_remove_features(
    perception: dict[str, Any], relation_audit: dict[str, Any]
) -> dict[str, Any]:
    """Extract planner-visible features from the current post-remove sample."""
    ranking_path = Path(perception["ranking_path"])
    ranking = load_json(ranking_path)
    model_input = load_json(Path(ranking["input_path"]))
    selected = str(ranking["selected_candidate_id"])
    selected_index = ranking["candidate_ids"].index(selected)
    candidate_by_id = {
        str(item["candidate_id"]): item for item in model_input["candidates"]
    }
    bbox = candidate_by_id[selected]["bbox_xyxy"]
    membership = relation_audit["rgbd_relation"]["membership_world_evidence"]
    values = [
        float(ranking["raw_match_logits"][selected_index]),
        float(len(ranking["candidate_ids"])),
        float(membership["inside_margin_m"]),
        0.5 * (float(bbox[0]) + float(bbox[2])),
        0.5 * (float(bbox[1]) + float(bbox[3])),
    ]
    return {
        "sample_id": str(model_input["sample_id"]),
        "values": values,
        "named_values": dict(zip(FEATURE_NAMES, values)),
        "inference_inputs": [
            str(ranking_path.resolve()),
            str(Path(ranking["input_path"]).resolve()),
            str(Path(relation_audit["config_path"]).resolve()),
        ],
        "simulator_ground_truth_used": False,
    }


def _statistics(rows: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    dimension = len(FEATURE_NAMES)
    mean = [
        sum(float(row["features"]["values"][index]) for row in rows) / len(rows)
        for index in range(dimension)
    ]
    std = []
    for index, center in enumerate(mean):
        variance = sum(
            (float(row["features"]["values"][index]) - center) ** 2
            for row in rows
        ) / len(rows)
        std.append(math.sqrt(variance) or 1.0)
    return mean, std


def predict_view_mode(
    features: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    """Return a distance-weighted view-mode distribution."""
    rows = list(model["episodes"])
    if not rows:
        raise ValueError("The view model has no development episodes")
    mean, std = _statistics(rows)
    neighbors = []
    for row in rows:
        distance = math.sqrt(
            sum(
                (
                    (float(features["values"][index])
                     - float(row["features"]["values"][index]))
                    / std[index]
                ) ** 2
                for index in range(len(FEATURE_NAMES))
            )
        )
        neighbors.append(
            {
                "seed": int(row["seed"]),
                "view_mode": str(row["view_mode"]),
                "distance": distance,
            }
        )
    neighbors.sort(key=lambda item: (item["distance"], item["seed"]))
    neighbors = neighbors[: min(int(model.get("neighbor_count", 3)), len(rows))]
    weights = {mode: 0.0 for mode in VIEW_MODES}
    for neighbor in neighbors:
        weight = 1.0 / (float(neighbor["distance"]) + 0.1)
        neighbor["weight"] = weight
        weights[str(neighbor["view_mode"])] += weight
    total = sum(weights.values())
    probabilities = {mode: weights[mode] / total for mode in VIEW_MODES}
    return {
        "view_mode_probabilities": probabilities,
        "neighbors": neighbors,
        "feature_mean": dict(zip(FEATURE_NAMES, mean)),
        "feature_std": dict(zip(FEATURE_NAMES, std)),
        "future_view_observation_used": False,
        "simulator_ground_truth_used": False,
    }


def condition_joint_model_on_view_resolvability(
    joint_model: dict[str, Any], prediction: dict[str, Any]
) -> dict[str, Any]:
    """Mix each view likelihood with an uninformative observation branch.

    The scene-conditioned predictor is used as an action-conditioned sensor
    model, not as a policy override.  A view predicted to resolve the current
    scene keeps more of its fitted observation likelihood.  The remaining mass
    is assigned to ``other|unknown``, which has equal likelihood across latent
    states and therefore leaves the posterior unchanged.

    Development KNN weights are not calibrated probabilities.  This adapter is
    consequently restricted to integration preflights until a disjoint
    calibration split supplies validated resolution probabilities.
    """
    conditioned = deepcopy(joint_model)
    probabilities = prediction["view_mode_probabilities"]
    action_modes = {
        "viewpoint_close_high": "close_high",
        "viewpoint_right": "right",
    }
    metadata = {}
    for action, mode in action_modes.items():
        if action not in conditioned["joint_observation_likelihood"]:
            continue
        resolution = float(probabilities[mode])
        if not 0.0 <= resolution <= 1.0:
            raise ValueError(f"Invalid resolution weight for {action}: {resolution}")
        vocabulary = list(conditioned["observation_vocabulary"][action])
        unresolved = "other|unknown"
        if unresolved not in vocabulary:
            vocabulary.append(unresolved)
            vocabulary.sort()
            conditioned["observation_vocabulary"][action] = vocabulary
        for state, row in conditioned["joint_observation_likelihood"][action].items():
            base = {symbol: float(row.get(symbol, 0.0)) for symbol in vocabulary}
            mixed = {symbol: resolution * value for symbol, value in base.items()}
            mixed[unresolved] += 1.0 - resolution
            total = sum(mixed.values())
            conditioned["joint_observation_likelihood"][action][state] = {
                symbol: value / total for symbol, value in mixed.items()
            }
        metadata[action] = {
            "view_mode": mode,
            "resolution_weight": resolution,
            "unresolved_symbol": unresolved,
        }
    conditioned["scene_conditioned_sensor_model"] = {
        "method": "resolution_weighted_likelihood_with_uninformative_branch",
        "actions": metadata,
        "policy_override_used": False,
        "calibrated_resolution_probability": False,
        "valid_for_final_evaluation": False,
    }
    return conditioned


def select_scene_conditioned_view(
    prediction: dict[str, Any], costs: dict[str, float], unresolved_cost: float
) -> dict[str, Any]:
    """Choose the view with the smallest predicted unresolved-task cost."""
    probabilities = prediction["view_mode_probabilities"]
    values = []
    for action, mode in (
        ("viewpoint_close_high", "close_high"),
        ("viewpoint_right", "right"),
    ):
        resolution = float(probabilities[mode])
        values.append(
            {
                "action": action,
                "kind": "scene_conditioned_future_visibility",
                "predicted_resolution_probability": resolution,
                "expected_cost": float(costs[action])
                + (1.0 - resolution) * float(unresolved_cost),
            }
        )
    selected = min(values, key=lambda item: (item["expected_cost"], item["action"]))
    return {
        "planner": "scene_conditioned_future_visibility_development_policy",
        "action_values": values,
        "selected_action": selected["action"],
        "selected_expected_cost": selected["expected_cost"],
        "view_mode_belief": probabilities,
        "future_held_out_observation_used_for_action_selection": False,
    }
