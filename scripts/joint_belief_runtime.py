"""Live RGB-D tracking adapters for the joint-belief planner."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from rgbd_target_localization import localize_mask_files
from run_scanned_basket_pipeline import resolve_input_asset


def sigmoid(value: float) -> float:
    """Compute a numerically stable logistic transform."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def identity_bin(raw_logit: float, temperature: float) -> str:
    """Map a temperature-scaled identity logit to a discrete evidence bin."""
    if temperature <= 0.0:
        raise ValueError("Target temperature must be positive")
    probability = sigmoid(float(raw_logit) / float(temperature))
    if probability < 0.2:
        return "very_low"
    if probability < 0.5:
        return "low"
    if probability < 0.8:
        return "medium"
    return "high"


def selected_membership(relation_audit: dict[str, Any]) -> str:
    """Read the selected candidate's planner-visible membership label."""
    label = str(
        relation_audit["rgbd_relation"]["membership_world_evidence"]["label"]
    )
    return label if label in {"inside", "outside"} else "unknown"


def initial_joint_observation_row(
    perception: dict[str, Any],
    relation_audit: dict[str, Any],
    *,
    target_temperature: float,
) -> dict[str, Any]:
    """Build the initial joint identity and membership observation."""
    ranking = perception["ranking"]
    selected = str(ranking["selected_candidate_id"])
    selected_index = ranking["candidate_ids"].index(selected)
    return {
        "action": "initial_observation",
        "identity_bin": identity_bin(
            float(ranking["raw_match_logits"][selected_index]),
            target_temperature,
        ),
        "membership_observation": selected_membership(relation_audit),
        "planner_visible_only": True,
    }


def candidate_localizations(
    perception: dict[str, Any], observation_dir: Path
) -> dict[str, dict[str, Any]]:
    """Backproject every anonymous candidate mask into world coordinates."""
    ranking = perception["ranking"]
    model_input_path = Path(ranking["input_path"])
    model_input = json.loads(model_input_path.read_text(encoding="utf-8"))
    masks = {
        str(candidate["candidate_id"]): resolve_input_asset(
            model_input, candidate["mask_path"]
        )
        for candidate in model_input["candidates"]
    }
    localized = localize_mask_files(observation_dir, masks)
    return dict(localized["estimates"])


def distance(first: list[float], second: list[float]) -> float:
    """Return Euclidean distance between two 3D centers."""
    return math.sqrt(
        sum((float(left) - float(right)) ** 2 for left, right in zip(first, second))
    )


def tracked_joint_observation_row(
    reference_perception: dict[str, Any],
    current_perception: dict[str, Any],
    current_relation_audit: dict[str, Any],
    observation_dir: Path,
    *,
    action: str,
    target_temperature: float,
    maximum_center_distance_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Associate the initial selected candidate with a new view in 3D."""
    reference_center = reference_perception["localization"]["estimates"][
        "selected_target"
    ]["center_world_m"]
    estimates = candidate_localizations(current_perception, observation_dir)
    ranking = current_perception["ranking"]
    selected = str(ranking["selected_candidate_id"])
    nearest_id = None
    nearest_distance = None
    if estimates:
        nearest_id, nearest = min(
            estimates.items(),
            key=lambda item: distance(
                reference_center, item[1]["center_world_m"]
            ),
        )
        nearest_distance = distance(reference_center, nearest["center_world_m"])
        if nearest_distance > float(maximum_center_distance_m):
            nearest_id = None
    if nearest_id is None:
        agreement = "missing"
        confidence = "missing"
    else:
        agreement = "same" if selected == nearest_id else "different"
        index = ranking["candidate_ids"].index(nearest_id)
        confidence = identity_bin(
            float(ranking["raw_match_logits"][index]), target_temperature
        )
    row = {
        "action": action,
        "membership_observation": selected_membership(current_relation_audit),
        "track_agreement_observation": agreement,
        "center_track_confidence_bin": confidence,
        "planner_visible_only": True,
    }
    selected_estimate = estimates.get(selected)
    track_localizations = {
        "track_center_selected": estimates.get(nearest_id) if nearest_id else None,
        "track_other_target": (
            selected_estimate
            if selected_estimate is not None and selected != nearest_id
            else None
        ),
    }
    audit = {
        "schema_version": "rgbd-persistent-track-observation-v1",
        "action": action,
        "reference_center_world_m": reference_center,
        "candidate_estimates": estimates,
        "matched_center_track_candidate_id": nearest_id,
        "matched_center_distance_m": nearest_distance,
        "current_selected_candidate_id": selected,
        "track_agreement_observation": agreement,
        "center_track_confidence_bin": confidence,
        "maximum_center_distance_m": float(maximum_center_distance_m),
        "track_localizations": track_localizations,
        "simulator_ids_used": False,
    }
    return row, audit


def localization_payload(
    estimate: dict[str, Any], observation_dir: Path, track_id: str
) -> dict[str, Any]:
    """Serialize a selected track location for the grasp executor."""
    return {
        "schema_version": "rgbd-selected-mask-localization-v1",
        "observation_dir": str(observation_dir.resolve()),
        "estimates": {"selected_target": estimate},
        "selection": {
            "persistent_track_id": track_id,
            "source": "rgbd_nearest_center_tracking",
            "simulator_ground_truth_used": False,
        },
        "training_performed": False,
        "simulator_ground_truth_used_for_estimate": False,
        "valid_for_final_evaluation": False,
    }


def fuse_static_track_localizations(
    estimates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fuse repeated 3D centers for an object that is stationary between views."""
    if not estimates:
        raise ValueError("At least one localization estimate is required")
    fused = dict(estimates[-1])
    valid_centers = [
        [float(value) for value in estimate["center_world_m"]]
        for estimate in estimates
        if len(estimate.get("center_world_m", [])) == 3
    ]
    if not valid_centers:
        raise ValueError("Localization estimates must contain 3D centers")
    if len(valid_centers) >= 3:
        fused["center_world_m"] = [
            statistics.median(center[0] for center in valid_centers),
            statistics.median(center[1] for center in valid_centers),
            valid_centers[-1][2],
        ]
        method = "horizontal_coordinate_median_latest_height"
    else:
        method = "latest_valid_estimate"
    fused["multi_view_center_fusion"] = {
        "method": method,
        "observation_count": len(valid_centers),
    }
    return fused
