"""Evaluate a non-oracle RGB-D relation adapter on saved calibration views.

The prediction pass consumes learned SAM masks, metric depth, and saved camera
calibration. Simulator IDs and relation labels are read only afterward by the
audit pass. The output intentionally separates:

* geometric world evidence (membership and camera-relative behind);
* observation quality (partial visibility, adjacency, invalid geometry);
* legacy RGB-only view-observable labels used by the Qwen pilot.

This is calibration-only method development. It does not fit Qwen, alter MPC,
or evaluate reserved test episodes.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rgbd_target_localization import backproject_distance_pixels


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs/perception/hybrid_rgbd_relation_pilot.json"
)


def resolve_path(value: str | Path) -> Path:
    """Resolve a repository-relative artifact path."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def load_mask(path: str | Path) -> np.ndarray:
    """Load any nonzero mask pixel as foreground."""
    return (
        np.asarray(Image.open(resolve_path(path)).convert("L"), dtype=np.uint8)
        > 0
    )


def binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square (Chebyshev-radius) dilation used only for mask adjacency."""
    source = np.asarray(mask, dtype=bool)
    if radius < 0:
        raise ValueError("Dilation radius must be nonnegative")
    if radius == 0:
        return source.copy()
    height, width = source.shape
    padded = np.pad(source, radius, mode="constant", constant_values=False)
    result = np.zeros_like(source)
    size = 2 * radius + 1
    for offset_y in range(size):
        for offset_x in range(size):
            result |= padded[
                offset_y : offset_y + height,
                offset_x : offset_x + width,
            ]
    return result


def masked_world_points(
    depth_m: np.ndarray,
    mask: np.ndarray,
    calibration: dict[str, Any],
    *,
    minimum_valid_pixels: int,
) -> np.ndarray:
    """Backproject finite positive depth samples under a learned mask."""
    depth = np.asarray(depth_m, dtype=np.float64)
    binary = np.asarray(mask, dtype=bool)
    if depth.shape != binary.shape:
        raise ValueError(
            f"Depth/mask shape mismatch: {depth.shape} vs {binary.shape}"
        )
    valid = binary & np.isfinite(depth) & (depth > 0.0)
    pixel_v, pixel_u = np.nonzero(valid)
    if pixel_u.size < minimum_valid_pixels:
        raise ValueError(
            f"Only {pixel_u.size} valid masked depth pixels; "
            f"need {minimum_valid_pixels}"
        )
    return backproject_distance_pixels(
        pixel_u,
        pixel_v,
        depth[valid],
        calibration,
    )


def robust_bounds(
    points: np.ndarray,
    percentiles: list[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute percentile-trimmed, world-axis-aligned bounds."""
    if len(percentiles) != 2 or percentiles[0] >= percentiles[1]:
        raise ValueError("Expected increasing [lower, upper] percentiles")
    lower, upper = np.percentile(
        np.asarray(points, dtype=np.float64),
        percentiles,
        axis=0,
    )
    return lower, upper


def bbox_overlap_fraction(
    source: list[float],
    reference: list[float],
) -> float:
    """Measure inclusive-pixel intersection as a fraction of the source box."""
    sx0, sy0, sx1, sy1 = source
    rx0, ry0, rx1, ry1 = reference
    intersection_width = max(0.0, min(sx1, rx1) - max(sx0, rx0) + 1.0)
    intersection_height = max(0.0, min(sy1, ry1) - max(sy0, ry0) + 1.0)
    source_area = max(0.0, sx1 - sx0 + 1.0) * max(
        0.0, sy1 - sy0 + 1.0
    )
    return (
        intersection_width * intersection_height / source_area
        if source_area > 0.0
        else 0.0
    )


def reference_geometry(
    points: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Validate the learned reference mask and form its relation footprint."""
    observed_lower, observed_upper = robust_bounds(
        points, settings["robust_percentiles"]
    )
    extents = observed_upper - observed_lower
    expected_xy = np.asarray(
        settings["expected_full_extents_xy_m"], dtype=np.float64
    )
    ratios = extents[:2] / expected_xy
    plausible = bool(
        np.all(ratios >= float(settings["minimum_extent_ratio"]))
        and np.all(ratios <= float(settings["maximum_extent_ratio"]))
    )
    lower = observed_lower.copy()
    upper = observed_upper.copy()
    if settings.get("use_expected_xy_extents_for_relation", False):
        # Known dimensions replace noisy visible extents, but not the observed center.
        center_xy = 0.5 * (observed_lower[:2] + observed_upper[:2])
        lower[:2] = center_xy - 0.5 * expected_xy
        upper[:2] = center_xy + 0.5 * expected_xy
    return {
        "valid": plausible,
        "valid_depth_pixel_count": int(points.shape[0]),
        "bounds_world_m": {
            "lower": lower.tolist(),
            "upper": upper.tolist(),
        },
        "observed_robust_bounds_world_m": {
            "lower": observed_lower.tolist(),
            "upper": observed_upper.tolist(),
        },
        "robust_extents_world_m": extents.tolist(),
        "expected_extents_xy_m": expected_xy.tolist(),
        "extent_ratios_xy": ratios.tolist(),
        "invalid_reason": (
            None if plausible else "reference_mask_extent_out_of_range"
        ),
        "frame_assumption": settings["frame_assumption"],
        "relation_xy_extent_source": (
            "measured_reference_dimensions_centered_on_observed_geometry"
            if settings.get("use_expected_xy_extents_for_relation", False)
            else "observed_robust_bounds"
        ),
    }


def classify_membership(
    candidate_center: np.ndarray,
    reference: dict[str, Any],
    boundary_abstention_m: float,
) -> dict[str, Any]:
    """Classify the candidate center against the reference XY footprint."""
    if not reference["valid"]:
        return {
            "label": "unknown",
            "reason": "invalid_reference_geometry",
            "outside_distance_m": None,
            "inside_margin_m": None,
        }
    lower = np.asarray(
        reference["bounds_world_m"]["lower"], dtype=np.float64
    )
    upper = np.asarray(
        reference["bounds_world_m"]["upper"], dtype=np.float64
    )
    center_xy = np.asarray(candidate_center, dtype=np.float64)[:2]
    outside_components = np.maximum(
        np.maximum(lower[:2] - center_xy, center_xy - upper[:2]),
        0.0,
    )
    # Distance is the largest axis violation, matching the axis-aligned footprint.
    outside_distance = float(np.max(outside_components))
    inside_margin = float(
        min(
            center_xy[0] - lower[0],
            upper[0] - center_xy[0],
            center_xy[1] - lower[1],
            upper[1] - center_xy[1],
        )
    )
    if inside_margin >= 0.0:
        label = "inside"
        reason = "candidate_center_inside_learned_reference_footprint"
    elif outside_distance <= boundary_abstention_m:
        label = "unknown"
        reason = "candidate_center_within_boundary_abstention_band"
    else:
        label = "outside"
        reason = "candidate_center_outside_learned_reference_footprint"
    return {
        "label": label,
        "reason": reason,
        "outside_distance_m": outside_distance,
        "inside_margin_m": inside_margin,
        "calibrated": False,
    }


def classify_occlusion(
    visible_height_ratio: float,
    reference_adjacency: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Apply an abstaining heuristic from visible height and mask adjacency."""
    if (
        visible_height_ratio
        <= float(settings["yes_max_visible_height_ratio"])
        and reference_adjacency
        >= float(settings["yes_min_reference_adjacency"])
    ):
        label = "yes"
        reason = "short_visible_extent_and_reference_mask_adjacency"
    elif (
        visible_height_ratio
        >= float(settings["no_min_visible_height_ratio"])
        or reference_adjacency
        <= float(settings["no_max_reference_adjacency"])
    ):
        label = "no"
        reason = "near_complete_extent_or_no_reference_adjacency"
    else:
        label = "unknown"
        reason = "intermediate_visibility_evidence"
    return {
        "label": label,
        "reason": reason,
        "visible_height_ratio": float(visible_height_ratio),
        "reference_adjacency": float(reference_adjacency),
        "calibrated": False,
    }


def classify_behind(
    candidate_center: np.ndarray,
    camera_center: np.ndarray,
    reference: dict[str, Any],
    membership: dict[str, Any],
    occlusion: dict[str, Any],
    candidate_bbox_overlap: float,
    settings: dict[str, Any],
) -> dict[str, Any]:
    """Classify camera-relative behind evidence in the horizontal world plane."""
    if not reference["valid"]:
        return {
            "label": "unknown",
            "reason": "invalid_reference_geometry",
            "far_edge_offset_m": None,
        }
    lower = np.asarray(
        reference["bounds_world_m"]["lower"], dtype=np.float64
    )
    upper = np.asarray(
        reference["bounds_world_m"]["upper"], dtype=np.float64
    )
    reference_center = 0.5 * (lower + upper)
    camera_to_reference_xy = (
        reference_center[:2]
        - np.asarray(camera_center, dtype=np.float64)[:2]
    )
    norm = float(np.linalg.norm(camera_to_reference_xy))
    if not math.isfinite(norm) or norm <= 1e-9:
        return {
            "label": "unknown",
            "reason": "degenerate_camera_reference_direction",
            "far_edge_offset_m": None,
        }
    direction = camera_to_reference_xy / norm
    # Project the axis-aligned footprint onto the camera-to-reference direction.
    half_extent_along_ray = 0.5 * (
        abs(direction[0]) * (upper[0] - lower[0])
        + abs(direction[1]) * (upper[1] - lower[1])
    )
    center_offset = float(
        np.dot(
            np.asarray(candidate_center, dtype=np.float64)[:2]
            - reference_center[:2],
            direction,
        )
    )
    # Positive values lie beyond the reference's far edge, away from the camera.
    far_edge_offset = center_offset - half_extent_along_ray
    minimum_overlap = float(settings["minimum_candidate_bbox_overlap"])
    abstention = float(settings["far_edge_abstention_m"])
    if (
        membership["label"] == "inside"
        and occlusion["label"] == "yes"
        and candidate_bbox_overlap >= minimum_overlap
    ):
        label = "yes"
        reason = "inside_candidate_hidden_by_view_facing_reference_surface"
    elif (
        far_edge_offset > abstention
        and candidate_bbox_overlap >= minimum_overlap
    ):
        label = "yes"
        reason = "candidate_beyond_reference_far_edge_along_camera_ray"
    elif (
        abs(far_edge_offset) <= abstention
        and candidate_bbox_overlap >= minimum_overlap
    ):
        label = "unknown"
        reason = "candidate_near_reference_far_edge_abstention_band"
    else:
        label = "no"
        reason = "no_camera_relative_behind_evidence"
    return {
        "label": label,
        "reason": reason,
        "camera_ray_center_offset_m": center_offset,
        "reference_half_extent_along_ray_m": float(
            half_extent_along_ray
        ),
        "far_edge_offset_m": float(far_edge_offset),
        "candidate_bbox_overlap": float(candidate_bbox_overlap),
        "calibrated": False,
    }


def predict_candidate(
    candidate: dict[str, Any],
    reference_entity: dict[str, Any],
    depth_m: np.ndarray,
    calibration: dict[str, Any],
    reference_mask: np.ndarray,
    reference: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Derive relation evidence from learned masks, depth, and calibration."""
    candidate_mask = load_mask(candidate["mask_path"])
    candidate_points = masked_world_points(
        depth_m,
        candidate_mask,
        calibration,
        minimum_valid_pixels=int(
            config["candidate_geometry"]["minimum_valid_depth_pixels"]
        ),
    )
    lower, upper = robust_bounds(
        candidate_points,
        config["candidate_geometry"]["robust_percentiles"],
    )
    center_lower, center_upper = robust_bounds(
        candidate_points,
        config["candidate_geometry"].get(
            "center_robust_percentiles",
            config["candidate_geometry"]["robust_percentiles"],
        ),
    )
    # Center trimming may be stricter than the bounds used for visible extent.
    center = 0.5 * (center_lower + center_upper)
    candidate_extent = upper - lower
    ring = binary_dilate(
        candidate_mask,
        int(config["occlusion_evidence"]["mask_dilation_pixels"]),
    ) & ~candidate_mask
    # Adjacency is the share of the candidate's outer ring covered by the reference.
    reference_adjacency = float(
        np.logical_and(ring, reference_mask).sum() / max(1, ring.sum())
    )
    visible_height_ratio = float(
        # World Z is vertical under the axis-aligned calibration-scene assumption.
        candidate_extent[2]
        / float(config["candidate_geometry"]["known_mug_height_m"])
    )
    bbox_overlap = bbox_overlap_fraction(
        candidate["bbox_xyxy"], reference_entity["bbox_xyxy"]
    )
    membership = classify_membership(
        center,
        reference,
        float(
            config["reference_geometry"]["boundary_abstention_m"]
        ),
    )
    occlusion = classify_occlusion(
        visible_height_ratio,
        reference_adjacency,
        config["occlusion_evidence"],
    )
    camera_center = np.asarray(
        calibration["camera_to_world_row_vector_matrix"],
        dtype=np.float64,
    )[3, :3]
    behind = classify_behind(
        center,
        camera_center,
        reference,
        membership,
        occlusion,
        bbox_overlap,
        config["behind_evidence"],
    )
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_geometry": {
            "valid_depth_pixel_count": int(candidate_points.shape[0]),
            "center_world_m": center.tolist(),
            "bounds_world_m": {
                "lower": lower.tolist(),
                "upper": upper.tolist(),
            },
            "center_robust_bounds_world_m": {
                "lower": center_lower.tolist(),
                "upper": center_upper.tolist(),
            },
            "robust_extents_world_m": candidate_extent.tolist(),
            "bbox_overlap_with_reference": bbox_overlap,
        },
        "membership_world_evidence": membership,
        "behind_camera_relative_evidence": behind,
        "occluded_by_reference_evidence": occlusion,
        "simulator_ground_truth_used_for_prediction": False,
    }


def predict_sample(
    sample: dict[str, Any],
    calibration_root: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Predict every anonymous proposal before any audit labels are loaded."""
    sample_id = sample["sample_id"]
    input_path = (
        calibration_root
        / "perception"
        / "grounded_sam2_qwen_inputs"
        / sample_id
        / "input.json"
    )
    ranking_path = (
        calibration_root
        / "perception"
        / "grounded_sam2_qwen_rankings"
        / sample_id
        / "result.json"
    )
    model_input = load_json(input_path)
    ranking = load_json(ranking_path)
    observation_dir = resolve_path(sample["observation_dir"])
    depth_m = np.load(observation_dir / "depth_m.npy")
    calibration = load_json(
        observation_dir / "camera_calibration.json"
    )
    reference_entity = model_input["reference_entities"][0]
    reference_mask = load_mask(reference_entity["mask_path"])
    try:
        reference_points = masked_world_points(
            depth_m,
            reference_mask,
            calibration,
            minimum_valid_pixels=int(
                config["reference_geometry"][
                    "minimum_valid_depth_pixels"
                ]
            ),
        )
        reference = reference_geometry(
            reference_points, config["reference_geometry"]
        )
    except ValueError as error:
        reference = {
            "valid": False,
            "invalid_reason": str(error),
            "frame_assumption": config["reference_geometry"][
                "frame_assumption"
            ],
        }
    predictions = []
    failures = []
    for candidate in model_input["candidates"]:
        try:
            predictions.append(
                predict_candidate(
                    candidate,
                    reference_entity,
                    depth_m,
                    calibration,
                    reference_mask,
                    reference,
                    config,
                )
            )
        except ValueError as error:
            failures.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "reason": str(error),
                }
            )
    selected_candidate_id = ranking.get("selected_candidate_id")
    selected = next(
        (
            item
            for item in predictions
            if item["candidate_id"] == selected_candidate_id
        ),
        None,
    )
    return {
        "sample_id": sample_id,
        "episode_id": model_input["episode_id"],
        "view_id": model_input["view_id"],
        "calibration_scene_variant": sample[
            "calibration_scene_variant"
        ],
        "selected_candidate_id_from_qwen_identity": selected_candidate_id,
        "reference_geometry": reference,
        "candidate_predictions": predictions,
        "candidate_failures": failures,
        "selected_candidate_relation_evidence": selected,
        "prediction_inputs": [
            "learned_candidate_masks",
            "learned_reference_mask",
            "metric_depth",
            "camera_calibration",
            "known_reference_and_mug_dimensions",
        ],
        "simulator_ground_truth_used_for_prediction": False,
    }


def confusion_summary(
    rows: list[dict[str, Any]],
    *,
    prediction_key: str,
    truth_key: str,
    labels: list[str],
    abstention_label: str = "unknown",
) -> dict[str, Any]:
    """Report accuracy plus accuracy conditional on a non-unknown decision."""
    matrix = {
        truth: {prediction: 0 for prediction in labels}
        for truth in labels
    }
    correct = 0
    covered = 0
    selective_correct = 0
    per_class = {}
    for row in rows:
        truth = row[truth_key]
        prediction = row[prediction_key]
        if truth not in matrix:
            matrix[truth] = {item: 0 for item in labels}
        if prediction not in matrix[truth]:
            matrix[truth][prediction] = 0
        matrix[truth][prediction] += 1
        correct += int(prediction == truth)
        if prediction != abstention_label:
            covered += 1
            selective_correct += int(prediction == truth)
    for truth in labels:
        support = sum(
            1 for row in rows if row[truth_key] == truth
        )
        class_correct = sum(
            1
            for row in rows
            if row[truth_key] == truth
            and row[prediction_key] == truth
        )
        per_class[truth] = {
            "support": support,
            "correct": class_correct,
            "recall": class_correct / support if support else None,
        }
    count = len(rows)
    return {
        "record_count": count,
        "accuracy": correct / count if count else None,
        "coverage": covered / count if count else None,
        "selective_accuracy": (
            selective_correct / covered if covered else None
        ),
        "confusion": matrix,
        "per_class": per_class,
    }


def audit_predictions(
    predictions: list[dict[str, Any]],
    calibration_root: Path,
    perception_config: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Join simulator identities and truth only for post-hoc calibration audit."""
    calibration_records = {
        item["sample_id"]: item
        for item in load_json(
            calibration_root / "calibration_records.json"
        )["records"]
    }
    sample_config = {
        item["sample_id"]: item
        for item in perception_config["samples"]
    }
    rows = []
    target_proposal_missing = 0
    for sample_prediction in predictions:
        sample_id = sample_prediction["sample_id"]
        record = calibration_records[sample_id]
        record_by_candidate = {
            item["candidate_id"]: item
            for item in record["candidates"]
        }
        sample = sample_config[sample_id]
        ground_truth = load_json(
            resolve_path(sample["calibration_ground_truth_file"])
        )
        if not record["target_proposal_present"]:
            target_proposal_missing += 1
        for candidate_prediction in sample_prediction[
            "candidate_predictions"
        ]:
            candidate_id = candidate_prediction["candidate_id"]
            candidate_record = record_by_candidate[candidate_id]
            entity_id = candidate_record["matched_simulator_entity"]
            world_membership = ground_truth["world_ground_truth"][
                "entities"
            ][entity_id]["membership"]
            view_truth = candidate_record["relation_ground_truth"]
            qwen = candidate_record["factorized_relation_scores"]
            ground_truth_sources = candidate_record.get(
                "relation_ground_truth_sources", {}
            )
            objective_occlusion_gt = (
                view_truth["occluded_by"]
                if ground_truth_sources.get("occluded_by")
                == "rendered_reference_removed_amodal_fraction"
                else None
            )
            objective_behind_measurement = (
                ground_truth.get(
                    "objective_camera_relative_behind_ground_truth", {}
                ).get(record["view"])
            )
            objective_behind_applies = (
                entity_id
                == ground_truth["world_ground_truth"]["target_id"]
            )
            objective_behind_gt = (
                objective_behind_measurement.get("label")
                if isinstance(objective_behind_measurement, dict)
                and objective_behind_measurement.get("valid")
                and objective_behind_applies
                else None
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "seed": record["seed"],
                    "view_id": record["view"],
                    "variant": record["calibration_scene_variant"],
                    "candidate_id": candidate_id,
                    "matched_entity_posthoc": entity_id,
                    "target_label_posthoc": candidate_record[
                        "target_label"
                    ],
                    "reference_geometry_valid": sample_prediction[
                        "reference_geometry"
                    ]["valid"],
                    "hybrid_membership": candidate_prediction[
                        "membership_world_evidence"
                    ]["label"],
                    "world_membership_gt": world_membership,
                    "legacy_view_membership_gt": view_truth[
                        "membership"
                    ],
                    "qwen_membership": qwen["membership"]["top_label"],
                    "hybrid_behind": candidate_prediction[
                        "behind_camera_relative_evidence"
                    ]["label"],
                    "legacy_view_behind_gt": view_truth["behind"],
                    "qwen_behind": qwen["behind"]["top_label"],
                    "objective_behind_gt": objective_behind_gt,
                    "objective_behind_ground_truth_source": (
                        "simulator_geometry_and_rendered_projection"
                        if objective_behind_gt is not None
                        else None
                    ),
                    "hybrid_occluded_by": candidate_prediction[
                        "occluded_by_reference_evidence"
                    ]["label"],
                    "legacy_view_occluded_by_gt": view_truth[
                        "occluded_by"
                    ],
                    "qwen_occluded_by": qwen["occluded_by"][
                        "top_label"
                    ],
                    "objective_occluded_by_gt": (
                        objective_occlusion_gt
                    ),
                    "occluded_by_ground_truth_source": (
                        ground_truth_sources.get("occluded_by")
                    ),
                    "visible_height_ratio": candidate_prediction[
                        "occluded_by_reference_evidence"
                    ]["visible_height_ratio"],
                    "reference_adjacency": candidate_prediction[
                        "occluded_by_reference_evidence"
                    ]["reference_adjacency"],
                    "membership_outside_distance_m": (
                        candidate_prediction[
                            "membership_world_evidence"
                        ]["outside_distance_m"]
                    ),
                    "behind_far_edge_offset_m": candidate_prediction[
                        "behind_camera_relative_evidence"
                    ]["far_edge_offset_m"],
                }
            )
    world_membership_labels = ["inside", "outside", "unknown"]
    view_factor_labels = ["yes", "no", "unknown"]
    objective_occlusion_rows = [
        row
        for row in rows
        if row["objective_occluded_by_gt"] is not None
    ]
    objective_behind_rows = [
        row for row in rows if row["objective_behind_gt"] is not None
    ]
    metrics = {
        "hybrid_world_membership": confusion_summary(
            rows,
            prediction_key="hybrid_membership",
            truth_key="world_membership_gt",
            labels=world_membership_labels,
        ),
        "qwen_vs_legacy_view_membership": confusion_summary(
            rows,
            prediction_key="qwen_membership",
            truth_key="legacy_view_membership_gt",
            labels=world_membership_labels,
        ),
        "hybrid_vs_legacy_view_membership": confusion_summary(
            rows,
            prediction_key="hybrid_membership",
            truth_key="legacy_view_membership_gt",
            labels=world_membership_labels,
        ),
        "qwen_vs_legacy_view_behind": confusion_summary(
            rows,
            prediction_key="qwen_behind",
            truth_key="legacy_view_behind_gt",
            labels=view_factor_labels,
        ),
        "hybrid_vs_legacy_view_behind": confusion_summary(
            rows,
            prediction_key="hybrid_behind",
            truth_key="legacy_view_behind_gt",
            labels=view_factor_labels,
        ),
        "qwen_vs_objective_camera_relative_behind": confusion_summary(
            objective_behind_rows,
            prediction_key="qwen_behind",
            truth_key="objective_behind_gt",
            labels=view_factor_labels,
        ),
        "hybrid_vs_objective_camera_relative_behind": confusion_summary(
            objective_behind_rows,
            prediction_key="hybrid_behind",
            truth_key="objective_behind_gt",
            labels=view_factor_labels,
        ),
        "objective_camera_relative_behind_record_count": len(
            objective_behind_rows
        ),
        "qwen_vs_legacy_view_occluded_by": confusion_summary(
            rows,
            prediction_key="qwen_occluded_by",
            truth_key="legacy_view_occluded_by_gt",
            labels=view_factor_labels,
        ),
        "hybrid_vs_legacy_view_occluded_by": confusion_summary(
            rows,
            prediction_key="hybrid_occluded_by",
            truth_key="legacy_view_occluded_by_gt",
            labels=view_factor_labels,
        ),
        "qwen_vs_objective_occluded_by": confusion_summary(
            objective_occlusion_rows,
            prediction_key="qwen_occluded_by",
            truth_key="objective_occluded_by_gt",
            labels=view_factor_labels,
        ),
        "hybrid_vs_objective_occluded_by": confusion_summary(
            objective_occlusion_rows,
            prediction_key="hybrid_occluded_by",
            truth_key="objective_occluded_by_gt",
            labels=view_factor_labels,
        ),
        "objective_occlusion_target_record_count": len(
            objective_occlusion_rows
        ),
        "reference_geometry_invalid_samples": sum(
            not item["reference_geometry"]["valid"]
            for item in predictions
        ),
        "target_proposal_missing_observations": target_proposal_missing,
        "rgbd_resolves_legacy_membership_unknown_count": sum(
            row["legacy_view_membership_gt"] == "unknown"
            and row["hybrid_membership"] in {"inside", "outside"}
            and row["hybrid_membership"] == row["world_membership_gt"]
            for row in rows
        ),
        "legacy_occlusion_label_conflict_count": sum(
            row["hybrid_occluded_by"]
            != row["legacy_view_occluded_by_gt"]
            for row in rows
        ),
    }
    return metrics, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write flattened relation predictions for cross-view analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path) -> dict[str, Any]:
    """Run the label-free prediction pass, then audit its frozen outputs."""
    config = load_json(config_path)
    calibration_root = resolve_path(config["calibration_root"])
    output_root = resolve_path(config["output_root"])
    perception_config = load_json(
        calibration_root / "perception_config.json"
    )
    predictions = [
        predict_sample(sample, calibration_root, config)
        for sample in perception_config["samples"]
    ]
    # Ground-truth artifacts are opened only inside this separate audit call.
    metrics, audit_rows = audit_predictions(
        predictions, calibration_root, perception_config
    )
    objective_occlusion_audit = config.get(
        "objective_occlusion_audit"
    )
    occlusion_blocking_reason = (
        "objective_occlusion_yes_recall_requires_method_revision"
        if objective_occlusion_audit is not None
        else "occlusion_label_definition_requires_objective_revision"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / "predictions.json"
    audit_path = output_root / "audit_rows.csv"
    result_path = output_root / "summary.json"
    predictions_path.write_text(
        json.dumps(
            {
                "schema_version": "hybrid-rgbd-relation-predictions-v1",
                "samples": predictions,
                "simulator_ground_truth_used_for_prediction": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_csv(audit_path, audit_rows)
    result = {
        "schema_version": "hybrid-rgbd-relation-pilot-summary-v1",
        "experiment_id": config["experiment_id"],
        "split": "calibration_only",
        "sample_count": len(predictions),
        "candidate_record_count": len(audit_rows),
        "metrics": metrics,
        "artifacts": {
            "predictions": str(predictions_path.resolve()),
            "audit_rows_csv": str(audit_path.resolve()),
        },
        "interpretation": {
            "world_membership_primary_for_rgbd_geometry_audit": True,
            "legacy_view_labels_were_authored_for_rgb_only_qwen": True,
            "legacy_view_labels_must_not_be_silently_reused_as_rgbd_truth": True,
            "objective_occlusion_ground_truth_used_for_target_audit": (
                objective_occlusion_audit is not None
            ),
            "scores_are_calibrated_probabilities": False,
        },
        "deployment_decision": {
            "apply_to_mpc": False,
            "blocking_reasons": [
                "calibration_only_not_tested",
                "reference_frame_is_simulation_axis_aligned",
                occlusion_blocking_reason,
                "occlusion_thresholds_not_calibrated",
                "action_conditioned_observation_model_not_fitted",
                "task_risk_gate_not_calibrated",
            ],
        },
        "training_performed": False,
        "calibration_performed": False,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
    }
    result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"HYBRID_RELATION_SUMMARY={result_path.resolve()}")
    return result


def main() -> None:
    """Estimate candidate relations from learned masks and metric depth."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config.resolve())


if __name__ == "__main__":
    main()
