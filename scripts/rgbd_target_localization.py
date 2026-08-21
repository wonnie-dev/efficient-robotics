"""Estimate an object center in world coordinates from RGB-D and a mask.

Depth values are Euclidean distances along camera rays, not optical-axis
depth. Learned pipelines enter through ``localize_mask_files`` after selecting
an anonymous RGB mask; controlled pilots can instead select a semantic instance
with ``localize_observation``. Neither estimator reads saved world positions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def backproject_distance_pixels(
    pixel_u: np.ndarray,
    pixel_v: np.ndarray,
    distance_m: np.ndarray,
    calibration: dict,
) -> np.ndarray:
    """Backproject Euclidean ray distances using the saved USD camera pose.

    Image coordinates use ``u`` right and ``v`` down. The USD camera looks down
    camera ``-Z`` with ``+Y`` up, and the saved transform multiplies row vectors.
    """
    u = np.asarray(pixel_u, dtype=np.float64)
    v = np.asarray(pixel_v, dtype=np.float64)
    distance = np.asarray(distance_m, dtype=np.float64)
    x = (u - float(calibration["cx_pixels"])) / float(
        calibration["fx_pixels"]
    )
    # Negating image v converts the image-down convention to camera-up +Y.
    y = -(v - float(calibration["cy_pixels"])) / float(
        calibration["fy_pixels"]
    )
    rays = np.stack([x, y, -np.ones_like(x)], axis=1)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    camera_points = rays * distance[:, None]
    homogeneous = np.concatenate(
        [camera_points, np.ones((camera_points.shape[0], 1))],
        axis=1,
    )
    camera_to_world = np.asarray(
        calibration["camera_to_world_row_vector_matrix"],
        dtype=np.float64,
    )
    # Translation lives in the last row under the saved row-vector convention.
    return (homogeneous @ camera_to_world)[:, :3]


def estimate_instance_center(
    depth_m: np.ndarray,
    instance_ids: np.ndarray,
    instance_id: int,
    calibration: dict,
    *,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
) -> dict:
    """Estimate the midpoint of a percentile-trimmed world-axis bounding box."""
    depth = np.asarray(depth_m, dtype=np.float64)
    ids = np.asarray(instance_ids)
    if depth.shape != ids.shape:
        raise ValueError(
            f"Depth and instance shapes differ: {depth.shape} vs {ids.shape}"
        )
    mask = (ids == int(instance_id)) & np.isfinite(depth) & (depth > 0.0)
    pixel_v, pixel_u = np.nonzero(mask)
    if pixel_u.size < 20:
        raise ValueError(
            f"Instance {instance_id} has only {pixel_u.size} valid depth pixels"
        )
    points = backproject_distance_pixels(
        pixel_u,
        pixel_v,
        depth[mask],
        calibration,
    )
    lower = np.percentile(points, lower_percentile, axis=0)
    upper = np.percentile(points, upper_percentile, axis=0)
    # A box midpoint is less sensitive to uneven visible surface area than a centroid.
    center = 0.5 * (lower + upper)
    return {
        "instance_id": int(instance_id),
        "valid_pixel_count": int(pixel_u.size),
        "center_world_m": center.tolist(),
        "robust_bounds_world_m": {
            "lower": lower.tolist(),
            "upper": upper.tolist(),
        },
        "robust_extent_m": (upper - lower).tolist(),
        "percentile_interval": [
            float(lower_percentile),
            float(upper_percentile),
        ],
        "estimator": "masked_depth_robust_world_aabb_center",
        "simulator_ground_truth_used_for_estimate": False,
    }


def estimate_mask_center(
    depth_m: np.ndarray,
    mask: np.ndarray,
    calibration: dict,
    *,
    label: str,
) -> dict:
    """Estimate a center from an externally selected anonymous candidate mask."""
    binary_mask = np.asarray(mask, dtype=bool)
    synthetic_ids = binary_mask.astype(np.uint8)
    estimate = estimate_instance_center(
        depth_m,
        synthetic_ids,
        1,
        calibration,
    )
    estimate.pop("instance_id")
    estimate["mask_label"] = label
    estimate["mask_source"] = "selected_anonymous_candidate_mask"
    return estimate


def localize_mask_files(
    observation_dir: Path,
    masks: dict[str, Path],
) -> dict:
    """Apply depth only after upstream perception has selected the masks."""
    from PIL import Image

    depth = np.load(observation_dir / "depth_m.npy")
    calibration = json.loads(
        (observation_dir / "camera_calibration.json").read_text(
            encoding="utf-8"
        )
    )
    estimates = {}
    for label, mask_path in masks.items():
        mask = np.asarray(
            Image.open(mask_path).convert("L"), dtype=np.uint8
        ) > 0
        estimates[label] = estimate_mask_center(
            depth, mask, calibration, label=label
        )
    return {
        "schema_version": "rgbd-selected-mask-localization-v1",
        "observation_dir": str(observation_dir.resolve()),
        "estimates": estimates,
        "training_performed": False,
        "simulator_ground_truth_used_for_estimate": False,
        "valid_for_final_evaluation": False,
    }


def instance_id_for_class(
    labels: dict,
    semantic_class: str,
) -> int:
    """Require one unambiguous simulator instance for a controlled pilot."""
    matches = [
        int(instance_id)
        for instance_id, value in labels.items()
        if isinstance(value, dict)
        and value.get("class") == semantic_class
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {semantic_class!r} instance, found {matches}"
        )
    return matches[0]


def localize_observation(
    observation_dir: Path,
    semantic_classes: tuple[str, ...],
) -> dict:
    """Localize named instance masks without reading saved object positions."""
    depth = np.load(observation_dir / "depth_m.npy")
    instance_ids = np.load(observation_dir / "instance_ids.npy")
    labels = json.loads(
        (observation_dir / "instance_labels.json").read_text(
            encoding="utf-8"
        )
    )
    calibration = json.loads(
        (observation_dir / "camera_calibration.json").read_text(
            encoding="utf-8"
        )
    )
    estimates = {}
    for semantic_class in semantic_classes:
        instance_id = instance_id_for_class(labels, semantic_class)
        estimates[semantic_class] = estimate_instance_center(
            depth,
            instance_ids,
            instance_id,
            calibration,
        )
    return {
        "schema_version": "rgbd-instance-localization-v1",
        "observation_dir": str(observation_dir.resolve()),
        "estimates": estimates,
        "training_performed": False,
        "simulator_ground_truth_used_for_estimate": False,
        "valid_for_final_evaluation": False,
    }


def main() -> None:
    """Run the command-line RGB-D localization entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("observation_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--semantic-class",
        action="append",
        dest="semantic_classes",
        default=[],
    )
    args = parser.parse_args()
    semantic_classes = tuple(args.semantic_classes or ["target_red"])
    result = localize_observation(
        args.observation_dir.resolve(), semantic_classes
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"RGBD_LOCALIZATION_RESULT={args.output.resolve()}")


if __name__ == "__main__":
    main()
