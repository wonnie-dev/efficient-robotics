"""Author the locally available LIBERO scanned basket as a USD mesh.

The source asset remains in the existing LIBERO checkout.  This module does
not download, copy, or modify it; it records the source and license in the
experiment metadata so pilot outputs remain attributable and reproducible.

Names ending in ``_WORLD_M`` use the stage world frame. Basket collision boxes
are basket-local, and target mug positions use the mug's bottom contact as the
transform origin.
"""

from __future__ import annotations

import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LIBERO_ROOT = Path(
    os.environ.get(
        "EFFICIENT_ROBOTICS_LIBERO_ROOT",
        ROOT / "external" / "LIBERO",
    )
)
BASKET_ASSET_DIR = (
    LIBERO_ROOT
    / "libero"
    / "libero"
    / "assets"
    / "stable_scanned_objects"
    / "basket"
)
BASKET_OBJ = BASKET_ASSET_DIR / "basket.obj"
BASKET_TEXTURE = BASKET_ASSET_DIR / "texture.png"
VISIBLE_OUTSIDE_MUG_POSITION_WORLD_M = (0.74, -0.08, 0.739)
CALIBRATION_OUTSIDE_TARGET_POSITION_WORLD_M = (0.62, -0.17, 0.739)
# Physics and perception must use the same scene geometry.  Keep the legacy
# names as aliases so older runners/configs do not silently change meaning.
PHYSICS_CLEARANCE_OUTSIDE_MUG_POSITION_WORLD_M = (
    VISIBLE_OUTSIDE_MUG_POSITION_WORLD_M
)
TARGET_MUG_OUTER_RADIUS_M = 0.041
ACTIVE_RIM_TARGET_Y_OFFSET_M = 0.020
ACTIVE_RIM_TARGET_WALL_MARGIN_M = 0.020
# The household mug Xform is authored at the mug's bottom, not its center.
# Vertical ray checks against the scanned basket mesh over the complete mug
# footprint give a highest interior support surface of 20 mm at this layout.
SCANNED_BASKET_INNER_SUPPORT_Z_OFFSET_M = 0.020
PERCEPTION_BASKET_SCALE_XYZ = (2.10, 2.10, 1.05)
PHYSICS_CLEARANCE_BASKET_SCALE_XYZ = PERCEPTION_BASKET_SCALE_XYZ
BASKET_COLLISION_BOXES_LOCAL_M = (
    {
        "name": "Bottom",
        "center": (0.0, 0.0, 0.010),
        "full_extents": (0.336, 0.312, 0.020),
    },
    {
        "name": "WallLeft",
        "center": (-0.175, 0.0, 0.0775),
        "full_extents": (0.014, 0.312, 0.155),
    },
    {
        "name": "WallRight",
        "center": (0.175, 0.0, 0.0775),
        "full_extents": (0.014, 0.312, 0.155),
    },
    {
        "name": "WallFront",
        "center": (0.0, -0.163, 0.0775),
        "full_extents": (0.336, 0.014, 0.155),
    },
    {
        "name": "WallBack",
        "center": (0.0, 0.163, 0.0775),
        "full_extents": (0.336, 0.014, 0.155),
    },
)
CALIBRATION_SCENE_VARIANTS = (
    "inside_clear",
    "outside",
    "rim_occluded",
    "covered_unknown",
)
ADDITIONAL_CALIBRATION_SCENE_VARIANTS = (
    "behind_ambiguous",
    "behind_boundary_unknown",
)
ACTION_DIFFERENTIATING_SCENE_VARIANTS = (
    "close_high_only",
    "right_only",
    "either_view",
    "cover_removal_required",
)
NEGATIVE_EVIDENCE_SCENE_VARIANTS = (
    "empty_cover_then_right",
)
ALL_CALIBRATION_SCENE_VARIANTS = (
    CALIBRATION_SCENE_VARIANTS
    + ADDITIONAL_CALIBRATION_SCENE_VARIANTS
    + ACTION_DIFFERENTIATING_SCENE_VARIANTS
    + NEGATIVE_EVIDENCE_SCENE_VARIANTS
)
CALIBRATION_COVER_LOCAL_CENTER_M = (0.0, 0.0, 0.166)
CALIBRATION_COVER_FULL_EXTENTS_M = (0.362, 0.338, 0.014)
MANIPULABLE_COVER_HANDLE_LOCAL_CENTER_M = (0.0, 0.0, 0.1905)
MANIPULABLE_COVER_HANDLE_FULL_EXTENTS_M = (0.080, 0.030, 0.035)
# Development value for a removable rigid cover.  The old 0.20 kg value made
# the visibly solid 36 cm plate behave like low-density foam.  This remains a
# provisional value until the lab cover is weighed.
MANIPULABLE_COVER_MASS_KG = 0.55
MANIPULABLE_COVER_LINEAR_DAMPING = 0.15
MANIPULABLE_COVER_ANGULAR_DAMPING = 0.80


def _composite_cover_mass_properties(
    *,
    mass_kg: float = MANIPULABLE_COVER_MASS_KG,
    plate_full_extents_m: tuple[float, float, float] = (
        CALIBRATION_COVER_FULL_EXTENTS_M
    ),
    plate_center_local_m: tuple[float, float, float] = (
        CALIBRATION_COVER_LOCAL_CENTER_M
    ),
    handle_full_extents_m: tuple[float, float, float] = (
        MANIPULABLE_COVER_HANDLE_FULL_EXTENTS_M
    ),
    handle_center_local_m: tuple[float, float, float] = (
        MANIPULABLE_COVER_HANDLE_LOCAL_CENTER_M
    ),
) -> tuple[
    tuple[float, float, float], tuple[float, float, float]
]:
    """Return COM and diagonal inertia for the plate-plus-handle boxes."""
    boxes = (
        (
            plate_full_extents_m,
            plate_center_local_m,
        ),
        (
            handle_full_extents_m,
            handle_center_local_m,
        ),
    )
    volumes = [math.prod(extents) for extents, _ in boxes]
    total_volume = sum(volumes)
    masses = [
        mass_kg * volume / total_volume
        for volume in volumes
    ]
    center = tuple(
        sum(
            mass * box_center[axis]
            for mass, (_, box_center) in zip(masses, boxes)
        )
        / mass_kg
        for axis in range(3)
    )
    inertia = [0.0, 0.0, 0.0]
    for mass, (extents, box_center) in zip(masses, boxes):
        x, y, z = extents
        local = (
            mass * (y * y + z * z) / 12.0,
            mass * (x * x + z * z) / 12.0,
            mass * (x * x + y * y) / 12.0,
        )
        dx = box_center[0] - center[0]
        dy = box_center[1] - center[1]
        dz = box_center[2] - center[2]
        parallel_axis = (
            mass * (dy * dy + dz * dz),
            mass * (dx * dx + dz * dz),
            mass * (dx * dx + dy * dy),
        )
        for axis in range(3):
            inertia[axis] += local[axis] + parallel_axis[axis]
    return center, tuple(inertia)


MANIPULABLE_COVER_CENTER_OF_MASS_LOCAL_M, MANIPULABLE_COVER_INERTIA_KG_M2 = (
    _composite_cover_mass_properties()
)
AMBIGUOUS_TRAY_SCALE_XYZ = (2.10, 2.10, 0.40)
AMBIGUOUS_TARGET_BACK_CLEARANCE_M = 0.012
AMBIGUOUS_CENTER_CAMERA_XY_M = (-0.03, -0.42)
AMBIGUOUS_TARGET_RAY_OFFSET_M = 0.325
# Actual composite-wrist center pose expressed in the unshifted benchmark
# scene frame.  This is used only to author a calibration scene around the
# predeclared 2 cm camera-relative far-edge abstention band.
OBJECTIVE_CENTER_CAMERA_XY_M = (
    0.11203225542107548,
    -0.2766633078724949,
)
BOUNDARY_UNKNOWN_LATERAL_OFFSET_M = 0.075
BOUNDARY_UNKNOWN_RAY_JITTER_M = 0.003
BOUNDARY_UNKNOWN_LATERAL_JITTER_M = 0.006
NEGATIVE_EVIDENCE_TARGET_X_OFFSET_M = 0.470
NEGATIVE_EVIDENCE_TARGET_Y_OFFSET_M = 0.300
ACTION_TARGET_X_JITTER_M = 0.035
ACTION_TARGET_Y_OFFSET_M = 0.035
ACTION_OCCLUDER_TARGET_DISTANCE_M = 0.100
ACTION_VERTICAL_OCCLUDER_RADIUS_M = 0.050
ACTION_VERTICAL_OCCLUDER_HEIGHT_M = 0.140
ACTION_PARTIAL_COVER_FULL_EXTENTS_M = (0.362, 0.110, 0.014)
ACTION_PARTIAL_COVER_LOCAL_CENTER_Z_M = 0.166
# These environment-local camera centers are derived from the saved,
# successful seed-169 wrist calibrations after undoing the benchmark
# environment shift [0.20, -0.32, -0.76]. They are design references only;
# every authored scene must still pass rendered objective-visibility gates.
ACTION_REFERENCE_CAMERA_POSITION_SCENE_M = {
    "close_high": (
        0.370206029184295,
        0.00731642585917285,
        1.2159143281330713,
    ),
    "right": (
        0.2514759943513152,
        -0.23436567885398208,
        1.1778739419843442,
    ),
}
ACTION_VISIBILITY_RESOLVED_MINIMUM = 0.65
ACTION_VISIBILITY_GAIN_MINIMUM = 0.15
ACTION_VISIBILITY_DOMINANCE_MINIMUM = 0.15
ACTION_VISIBILITY_HIDDEN_MAXIMUM = 0.02
# Analytic layouts are only proposals. A scene is accepted when the rendered
# masks satisfy these view-specific visibility gates.


def calibration_variant_for_seed(seed: int) -> str:
    """Map non-negative seeds to a balanced deterministic variant cycle."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return CALIBRATION_SCENE_VARIANTS[seed % len(CALIBRATION_SCENE_VARIANTS)]


def action_variant_for_seed(seed: int) -> str:
    """Map non-negative seeds to the balanced action-difference cycle."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return ACTION_DIFFERENTIATING_SCENE_VARIANTS[
        seed % len(ACTION_DIFFERENTIATING_SCENE_VARIANTS)
    ]


def factorized_calibration_ground_truth(variant: str) -> dict:
    """Return generator-derived world and view-observable scene labels.

    Membership and view-dependent relations are deliberately separated.
    In particular, ``unknown`` means insufficient evidence in a given image;
    it is not a third physical world state.
    """
    if variant not in ALL_CALIBRATION_SCENE_VARIANTS:
        raise ValueError(f"Unknown calibration scene variant: {variant}")

    world_membership = (
        "outside"
        if variant
        in (
            "outside",
            "behind_ambiguous",
            "behind_boundary_unknown",
            "empty_cover_then_right",
        )
        else "inside"
    )
    view_specs = {}
    for view_id in ("center", "close_high", "right"):
        target_view = {
            "membership_observable": world_membership,
            "behind": "no",
            "occluded_by": {
                "label": "no",
                "occluder_id": None,
            },
            "target_visibility_intent": "clear",
        }
        distractor_view = {
            "membership_observable": "outside",
            "behind": "no",
            "occluded_by": {
                "label": "no",
                "occluder_id": None,
            },
            "target_visibility_intent": "clear",
        }
        view_specs[view_id] = {
            **target_view,
            "entities": {
                "target_red": target_view,
                "rear_red_candidate": distractor_view,
            },
        }
    if variant == "rim_occluded":
        target_view = {
            "membership_observable": "unknown",
            "behind": "yes",
            "occluded_by": {
                "label": "yes",
                "occluder_id": "basket_01",
            },
            "target_visibility_intent": "partial",
        }
        view_specs["center"].update(target_view)
        view_specs["center"]["entities"]["target_red"] = target_view
    elif variant == "covered_unknown":
        for view_id in view_specs:
            target_view = {
                "membership_observable": "unknown",
                "behind": "unknown",
                "occluded_by": {
                    "label": "yes",
                    "occluder_id": "cover_01",
                },
                "target_visibility_intent": "none",
            }
            view_specs[view_id].update(target_view)
            view_specs[view_id]["entities"]["target_red"] = target_view
    elif variant == "behind_ambiguous":
        center_target = {
            "membership_observable": "unknown",
            "behind": "unknown",
            "occluded_by": {
                "label": "yes",
                "occluder_id": "basket_01",
            },
            "target_visibility_intent": "partial",
        }
        view_specs["center"].update(center_target)
        view_specs["center"]["entities"]["target_red"] = center_target
        for view_id in ("close_high", "right"):
            resolved_target = {
                "membership_observable": "outside",
                "behind": "yes",
                "occluded_by": {
                    "label": "no",
                    "occluder_id": None,
                },
                "target_visibility_intent": "clear",
            }
            view_specs[view_id].update(resolved_target)
            view_specs[view_id]["entities"]["target_red"] = (
                resolved_target
            )
    elif variant == "behind_boundary_unknown":
        for view_id in view_specs:
            target_view = {
                "membership_observable": "unknown",
                "behind": "unknown",
                "occluded_by": {
                    "label": "unknown",
                    "occluder_id": "basket_01",
                },
                "target_visibility_intent": "partial",
            }
            view_specs[view_id].update(target_view)
            view_specs[view_id]["entities"]["target_red"] = target_view
    elif variant in ACTION_DIFFERENTIATING_SCENE_VARIANTS:
        center_target = {
            "membership_observable": "unknown",
            "behind": (
                "unknown"
                if variant == "cover_removal_required"
                else "yes"
            ),
            "occluded_by": {
                "label": "yes",
                "occluder_id": (
                    "cover_01"
                    if variant == "cover_removal_required"
                    else "basket_01"
                ),
            },
            "target_visibility_intent": (
                "none"
                if variant == "cover_removal_required"
                else "partial"
            ),
        }
        view_specs["center"].update(center_target)
        view_specs["center"]["entities"]["target_red"] = center_target
        resolved_views = {
            "close_high_only": {"close_high"},
            "right_only": {"right"},
            "either_view": {"close_high", "right"},
            "cover_removal_required": set(),
        }[variant]
        blocked_view = {
            "close_high_only": "right",
            "right_only": "close_high",
        }.get(variant)
        for view_id in ("close_high", "right"):
            if view_id in resolved_views:
                target_view = {
                    "membership_observable": "inside",
                    "behind": "no",
                    "occluded_by": {
                        "label": "no",
                        "occluder_id": None,
                    },
                    "target_visibility_intent": "clear",
                }
            else:
                occluder_id = (
                    "cover_01"
                    if variant == "cover_removal_required"
                    else (
                        "occluder_orange"
                        if view_id == blocked_view
                        else "basket_01"
                    )
                )
                target_view = {
                    "membership_observable": "unknown",
                    "behind": "unknown",
                    "occluded_by": {
                        "label": "yes",
                        "occluder_id": occluder_id,
                    },
                    "target_visibility_intent": (
                        "none"
                        if variant == "cover_removal_required"
                        else "partial"
                    ),
                }
            view_specs[view_id].update(target_view)
            view_specs[view_id]["entities"]["target_red"] = target_view
    elif variant == "empty_cover_then_right":
        center_target = {
            "membership_observable": "unknown",
            "behind": "yes",
            "occluded_by": {
                "label": "yes",
                "occluder_id": "basket_01",
            },
            "target_visibility_intent": "partial",
        }
        close_high_target = {
            "membership_observable": "unknown",
            "behind": "yes",
            "occluded_by": {
                "label": "yes",
                "occluder_id": "basket_01",
            },
            "target_visibility_intent": "partial",
        }
        right_target = {
            "membership_observable": "outside",
            "behind": "yes",
            "occluded_by": {
                "label": "no",
                "occluder_id": None,
            },
            "target_visibility_intent": "clear",
        }
        for view_id, target_view in (
            ("center", center_target),
            ("close_high", close_high_target),
            ("right", right_target),
        ):
            view_specs[view_id].update(target_view)
            view_specs[view_id]["entities"]["target_red"] = target_view

    return {
        "schema_version": "factorized-calibration-ground-truth-v1",
        "variant": variant,
        "labels_source": "deterministic_scene_generator",
        "world_ground_truth": {
            "target_id": "target_red",
            "reference_id": "basket_01",
            "membership": world_membership,
            "entities": {
                "target_red": {
                    "membership": world_membership,
                },
                "rear_red_candidate": {
                    "membership": "outside",
                },
            },
        },
        "view_observable_intent": view_specs,
        "label_factorization": {
            "membership": ["inside", "outside"],
            "membership_observable": ["inside", "outside", "unknown"],
            "behind": ["yes", "no", "unknown"],
            "occluded_by": ["yes", "no", "unknown"],
        },
        "manual_annotation": False,
        "valid_for_final_evaluation": False,
        "action_outcome_design": (
            {
                "class": variant,
                "resolving_view_actions": {
                    "close_high_only": ["viewpoint_close_high"],
                    "right_only": ["viewpoint_right"],
                    "either_view": [
                        "viewpoint_close_high",
                        "viewpoint_right",
                    ],
                    "cover_removal_required": [],
                }[variant],
                "required_interaction_action": (
                    "remove_cover"
                    if variant == "cover_removal_required"
                    else None
                ),
                "render_validation_required": True,
            }
            if variant in ACTION_DIFFERENTIATING_SCENE_VARIANTS
            else (
                {
                    "class": variant,
                    "resolving_view_actions": ["viewpoint_right"],
                    "required_interaction_action": "remove_cover",
                    "post_interaction_observation": "empty_container",
                    "required_action_sequence": [
                        "remove_cover",
                        "viewpoint_right",
                        "grasp_outside",
                    ],
                    "minimum_belief_updates": 2,
                    "render_validation_required": True,
                }
                if variant == "empty_cover_then_right"
                else None
            )
        ),
    }


def _objective_visibility_fraction(measurement: dict) -> float | None:
    objective = measurement.get("objective_occlusion")
    if not isinstance(objective, dict) or not objective.get("valid", False):
        return None
    value = objective.get("visible_fraction_of_amodal")
    return float(value) if value is not None else None


def validate_action_differentiating_visibility(
    variant: str,
    measurements: dict,
) -> dict:
    """Accept an action variant only when rendered masks show its intended gain."""
    if variant not in ACTION_DIFFERENTIATING_SCENE_VARIANTS:
        raise ValueError(f"Not an action-differentiating variant: {variant}")
    required_views = ("center", "close_high", "right")
    missing = [view for view in required_views if view not in measurements]
    if missing:
        return {
            "passed": False,
            "checks": [],
            "failure_reasons": [f"missing_views:{','.join(missing)}"],
        }
    fractions = {
        view: _objective_visibility_fraction(measurements[view])
        for view in required_views
    }
    if any(value is None for value in fractions.values()):
        return {
            "passed": False,
            "checks": [],
            "failure_reasons": [
                "objective_visibility_fraction_required_for_action_variant"
            ],
            "values": fractions,
        }
    center = float(fractions["center"])
    close_high = float(fractions["close_high"])
    right = float(fractions["right"])
    checks = []
    if variant == "cover_removal_required":
        checks.append(
            {
                "name": "viewpoint_change_cannot_reveal_covered_target",
                "passed": all(
                    float(value) <= ACTION_VISIBILITY_HIDDEN_MAXIMUM
                    for value in fractions.values()
                ),
                "values": fractions,
            }
        )
    else:
        checks.append(
            {
                "name": "center_is_not_already_resolved",
                "passed": center < ACTION_VISIBILITY_RESOLVED_MINIMUM,
                "values": fractions,
            }
        )
        if variant == "close_high_only":
            checks.extend(
                [
                    {
                        "name": "close_high_resolves",
                        "passed": (
                            close_high >= ACTION_VISIBILITY_RESOLVED_MINIMUM
                            and close_high - center
                            >= ACTION_VISIBILITY_GAIN_MINIMUM
                        ),
                        "values": fractions,
                    },
                    {
                        "name": "close_high_dominates_right",
                        "passed": (
                            close_high - right
                            >= ACTION_VISIBILITY_DOMINANCE_MINIMUM
                        ),
                        "values": fractions,
                    },
                    {
                        "name": "right_remains_unresolved",
                        "passed": (
                            right < ACTION_VISIBILITY_RESOLVED_MINIMUM
                        ),
                        "values": fractions,
                    },
                ]
            )
        elif variant == "right_only":
            checks.extend(
                [
                    {
                        "name": "right_resolves",
                        "passed": (
                            right >= ACTION_VISIBILITY_RESOLVED_MINIMUM
                            and right - center
                            >= ACTION_VISIBILITY_GAIN_MINIMUM
                        ),
                        "values": fractions,
                    },
                    {
                        "name": "right_dominates_close_high",
                        "passed": (
                            right - close_high
                            >= ACTION_VISIBILITY_DOMINANCE_MINIMUM
                        ),
                        "values": fractions,
                    },
                    {
                        "name": "close_high_remains_unresolved",
                        "passed": (
                            close_high < ACTION_VISIBILITY_RESOLVED_MINIMUM
                        ),
                        "values": fractions,
                    },
                ]
            )
        else:
            checks.extend(
                [
                    {
                        "name": "close_high_resolves",
                        "passed": (
                            close_high >= ACTION_VISIBILITY_RESOLVED_MINIMUM
                            and close_high - center
                            >= ACTION_VISIBILITY_GAIN_MINIMUM
                        ),
                        "values": fractions,
                    },
                    {
                        "name": "right_resolves",
                        "passed": (
                            right >= ACTION_VISIBILITY_RESOLVED_MINIMUM
                            and right - center
                            >= ACTION_VISIBILITY_GAIN_MINIMUM
                        ),
                        "values": fractions,
                    },
                ]
            )
    failure_reasons = [
        check["name"] for check in checks if not check["passed"]
    ]
    return {
        "passed": not failure_reasons,
        "checks": checks,
        "failure_reasons": failure_reasons,
        "objective_visible_fraction_of_amodal": fractions,
        "thresholds": {
            "resolved_minimum": ACTION_VISIBILITY_RESOLVED_MINIMUM,
            "gain_minimum": ACTION_VISIBILITY_GAIN_MINIMUM,
            "dominance_minimum": ACTION_VISIBILITY_DOMINANCE_MINIMUM,
            "hidden_maximum": ACTION_VISIBILITY_HIDDEN_MAXIMUM,
        },
    }


def validate_calibration_visibility(
    variant: str, measurements: dict
) -> dict:
    """Apply the rendered-visibility acceptance gate for a calibration scene."""
    if variant not in ALL_CALIBRATION_SCENE_VARIANTS:
        raise ValueError(f"Unknown calibration scene variant: {variant}")
    required_views = ("center", "close_high", "right")
    missing = [view for view in required_views if view not in measurements]
    if missing:
        return {
            "passed": False,
            "checks": [],
            "failure_reasons": [f"missing_views:{','.join(missing)}"],
        }
    pixels = {
        view: int(measurements[view]["target_visible_pixel_count"])
        for view in required_views
    }
    if variant in ACTION_DIFFERENTIATING_SCENE_VARIANTS:
        return validate_action_differentiating_visibility(
            variant, measurements
        )
    if variant == "empty_cover_then_right":
        fractions = {
            view: _objective_visibility_fraction(measurements[view])
            for view in required_views
        }
        if fractions["center"] is None or fractions["right"] is None:
            return {
                "passed": False,
                "checks": [],
                "failure_reasons": [
                    "objective_visibility_fraction_required_for_negative_evidence_variant"
                ],
                "values": fractions,
            }
        center = float(fractions["center"])
        close_high = float(fractions["close_high"] or 0.0)
        fractions["close_high"] = close_high
        right = float(fractions["right"])
        checks = [
            {
                "name": "center_is_not_already_resolved",
                "passed": center < ACTION_VISIBILITY_RESOLVED_MINIMUM,
                "values": fractions,
            },
            {
                "name": "right_resolves_outside_target",
                "passed": (
                    right >= ACTION_VISIBILITY_RESOLVED_MINIMUM
                    and right - center >= ACTION_VISIBILITY_GAIN_MINIMUM
                ),
                "values": fractions,
            },
            {
                "name": "right_dominates_close_high",
                "passed": (
                    right - close_high
                    >= ACTION_VISIBILITY_DOMINANCE_MINIMUM
                ),
                "values": fractions,
            },
        ]
        failure_reasons = [
            check["name"] for check in checks if not check["passed"]
        ]
        return {
            "passed": not failure_reasons,
            "checks": checks,
            "failure_reasons": failure_reasons,
            "objective_visible_fraction_of_amodal": fractions,
        }
    checks = []
    if variant in ("inside_clear", "outside"):
        checks.append(
            {
                "name": "target_visible_in_every_view",
                "passed": all(value > 0 for value in pixels.values()),
                "values": pixels,
            }
        )
    elif variant in ("rim_occluded", "behind_ambiguous"):
        checks.extend(
            [
                {
                    "name": "target_partially_visible_in_every_view",
                    "passed": all(value > 0 for value in pixels.values()),
                    "values": pixels,
                },
                {
                    "name": "reobservation_reveals_more_than_center",
                    "passed": (
                        max(pixels["close_high"], pixels["right"])
                        > pixels["center"]
                    ),
                    "values": pixels,
                },
            ]
        )
    elif variant == "behind_boundary_unknown":
        objective = measurements["center"].get(
            "objective_camera_relative_behind"
        )
        checks.extend(
            [
                {
                    "name": "target_partially_visible_in_every_view",
                    "passed": all(value > 0 for value in pixels.values()),
                    "values": pixels,
                },
                {
                    "name": "center_objective_behind_is_unknown",
                    "passed": (
                        isinstance(objective, dict)
                        and objective.get("valid") is True
                        and objective.get("label") == "unknown"
                    ),
                    "values": objective,
                },
            ]
        )
    else:
        checks.append(
            {
                "name": "covered_target_hidden_in_every_view",
                "passed": all(value == 0 for value in pixels.values()),
                "values": pixels,
            }
        )
    failure_reasons = [
        check["name"] for check in checks if not check["passed"]
    ]
    return {
        "passed": not failure_reasons,
        "checks": checks,
        "failure_reasons": failure_reasons,
    }


def compute_rim_occluded_target_layout(
    target_position_world_m: tuple[float, float, float],
    basket_center_world_m: tuple[float, float, float],
) -> dict:
    """Move the target behind the scanned basket rim with no extra primitive."""
    target_x, _target_y, _target_z = target_position_world_m
    basket_x, basket_y, basket_z = basket_center_world_m
    half_x = 0.173412 * PERCEPTION_BASKET_SCALE_XYZ[0] * 0.5
    half_y = 0.162194 * PERCEPTION_BASKET_SCALE_XYZ[1] * 0.5
    inset = TARGET_MUG_OUTER_RADIUS_M + ACTIVE_RIM_TARGET_WALL_MARGIN_M
    target_x = min(
        max(target_x, basket_x - half_x + inset),
        basket_x + half_x - inset,
    )
    target_y = min(
        max(
            basket_y + ACTIVE_RIM_TARGET_Y_OFFSET_M,
            basket_y - half_y + inset,
        ),
        basket_y + half_y - inset,
    )
    target_z = basket_z + SCANNED_BASKET_INNER_SUPPORT_Z_OFFSET_M
    wall_clearance = min(
        target_x - TARGET_MUG_OUTER_RADIUS_M - (basket_x - half_x),
        (basket_x + half_x) - (target_x + TARGET_MUG_OUTER_RADIUS_M),
        target_y - TARGET_MUG_OUTER_RADIUS_M - (basket_y - half_y),
        (basket_y + half_y) - (target_y + TARGET_MUG_OUTER_RADIUS_M),
    )
    if wall_clearance + 1e-9 < ACTIVE_RIM_TARGET_WALL_MARGIN_M:
        raise ValueError(
            f"Rim-occluded target wall clearance is insufficient: {wall_clearance}"
        )
    return {
        "target_position_world_m": [target_x, target_y, target_z],
        "target_transform_origin": "mug_bottom_contact",
        "support_surface_z_offset_m": (
            SCANNED_BASKET_INNER_SUPPORT_Z_OFFSET_M
        ),
        "support_validation": (
            "vertical_rays_over_mug_footprint_against_scanned_mesh"
        ),
        "wall_clearance_m": wall_clearance,
        "geometry_validation_passed": True,
        "occlusion_source": "scanned_basket_front_rim_and_weave",
        "explicit_occluder_primitive_visible": False,
    }


def compute_behind_ambiguous_target_layout(
    basket_center_world_m: tuple[float, float, float],
    seed: int = 0,
) -> dict:
    """Place a table-supported mug behind the basket along the center ray."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    basket_x, basket_y, _basket_z = basket_center_world_m
    ray_x = basket_x - AMBIGUOUS_CENTER_CAMERA_XY_M[0]
    ray_y = basket_y - AMBIGUOUS_CENTER_CAMERA_XY_M[1]
    ray_norm = math.hypot(ray_x, ray_y)
    unit_x = ray_x / ray_norm
    unit_y = ray_y / ray_norm
    ray_offset = (
        AMBIGUOUS_TARGET_RAY_OFFSET_M
        + 0.018 * math.sin(seed * 1.731)
    )
    lateral_offset = 0.024 * math.sin(seed * 2.399 + 0.7)
    perpendicular_x = -unit_y
    perpendicular_y = unit_x
    target_position = (
        basket_x
        + unit_x * ray_offset
        + perpendicular_x * lateral_offset,
        basket_y
        + unit_y * ray_offset
        + perpendicular_y * lateral_offset,
        CALIBRATION_OUTSIDE_TARGET_POSITION_WORLD_M[2],
    )
    half_x = 0.173412 * PERCEPTION_BASKET_SCALE_XYZ[0] * 0.5
    half_y = 0.162194 * PERCEPTION_BASKET_SCALE_XYZ[1] * 0.5
    relative_x = abs(target_position[0] - basket_x)
    relative_y = abs(target_position[1] - basket_y)
    closest_dx = max(0.0, relative_x - half_x)
    closest_dy = max(0.0, relative_y - half_y)
    planar_clearance = (
        math.hypot(closest_dx, closest_dy) - TARGET_MUG_OUTER_RADIUS_M
    )
    if planar_clearance + 1e-9 < AMBIGUOUS_TARGET_BACK_CLEARANCE_M:
        raise ValueError(
            "Behind-ambiguous target clearance is insufficient: "
            f"{planar_clearance}"
        )
    return {
        "target_position_world_m": list(target_position),
        "center_camera_ray_aligned": True,
        "seed": seed,
        "ray_offset_m": ray_offset,
        "lateral_offset_m": lateral_offset,
        "basket_planar_clearance_m": planar_clearance,
        "geometry_validation_passed": True,
    }


def compute_behind_boundary_unknown_target_layout(
    basket_center_world_m: tuple[float, float, float],
    seed: int = 0,
) -> dict:
    """Place an outside mug in the camera-relative far-edge abstention band."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    basket_x, basket_y, _basket_z = basket_center_world_m
    ray_x = basket_x - OBJECTIVE_CENTER_CAMERA_XY_M[0]
    ray_y = basket_y - OBJECTIVE_CENTER_CAMERA_XY_M[1]
    ray_norm = math.hypot(ray_x, ray_y)
    unit_x = ray_x / ray_norm
    unit_y = ray_y / ray_norm
    perpendicular_x = -unit_y
    perpendicular_y = unit_x
    half_x = 0.173412 * PERCEPTION_BASKET_SCALE_XYZ[0] * 0.5
    half_y = 0.162194 * PERCEPTION_BASKET_SCALE_XYZ[1] * 0.5
    far_edge_ray_offset = abs(unit_x) * half_x + abs(unit_y) * half_y
    ray_offset = (
        far_edge_ray_offset
        + BOUNDARY_UNKNOWN_RAY_JITTER_M * math.sin(seed * 1.17)
    )
    lateral_offset = (
        BOUNDARY_UNKNOWN_LATERAL_OFFSET_M
        + BOUNDARY_UNKNOWN_LATERAL_JITTER_M
        * math.sin(seed * 2.11 + 0.4)
    )
    target_position = (
        basket_x
        + unit_x * ray_offset
        + perpendicular_x * lateral_offset,
        basket_y
        + unit_y * ray_offset
        + perpendicular_y * lateral_offset,
        CALIBRATION_OUTSIDE_TARGET_POSITION_WORLD_M[2],
    )
    relative_x = abs(target_position[0] - basket_x)
    relative_y = abs(target_position[1] - basket_y)
    closest_dx = max(0.0, relative_x - half_x)
    closest_dy = max(0.0, relative_y - half_y)
    planar_clearance = (
        math.hypot(closest_dx, closest_dy) - TARGET_MUG_OUTER_RADIUS_M
    )
    if planar_clearance + 1e-9 < AMBIGUOUS_TARGET_BACK_CLEARANCE_M:
        raise ValueError(
            "Behind-boundary target clearance is insufficient: "
            f"{planar_clearance}"
        )
    return {
        "target_position_world_m": list(target_position),
        "seed": seed,
        "ray_offset_m": ray_offset,
        "far_edge_ray_offset_m": far_edge_ray_offset,
        "analytic_transform_origin_far_edge_offset_m": (
            ray_offset - far_edge_ray_offset
        ),
        "lateral_offset_m": lateral_offset,
        "basket_planar_clearance_m": planar_clearance,
        "geometry_validation_passed": True,
        "rendered_objective_label_required": "unknown",
    }


def compute_action_differentiating_layout(
    basket_center_world_m: tuple[float, float, float],
    variant: str,
    seed: int = 0,
) -> dict:
    """Compute a stable target and optional view-specific occluder layout."""
    if variant not in ACTION_DIFFERENTIATING_SCENE_VARIANTS:
        raise ValueError(f"Not an action-differentiating variant: {variant}")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    basket_x, basket_y, basket_z = basket_center_world_m
    target_position = (
        basket_x
        + ACTION_TARGET_X_JITTER_M * math.sin(seed * 1.913 + 0.4),
        basket_y + ACTION_TARGET_Y_OFFSET_M,
        basket_z + SCANNED_BASKET_INNER_SUPPORT_Z_OFFSET_M,
    )
    blocked_view = {
        "close_high_only": "right",
        "right_only": "close_high",
    }.get(variant)
    occluder_position = None
    occluder_geometry = None
    if blocked_view == "right":
        camera = ACTION_REFERENCE_CAMERA_POSITION_SCENE_M[blocked_view]
        direction_x = camera[0] - target_position[0]
        direction_y = camera[1] - target_position[1]
        direction_norm = math.hypot(direction_x, direction_y)
        if direction_norm <= 1e-9:
            raise ValueError("Reference camera and target share the same XY")
        unit_x = direction_x / direction_norm
        unit_y = direction_y / direction_norm
        occluder_position = (
            target_position[0]
            + unit_x * ACTION_OCCLUDER_TARGET_DISTANCE_M,
            target_position[1]
            + unit_y * ACTION_OCCLUDER_TARGET_DISTANCE_M,
            basket_z
            + SCANNED_BASKET_INNER_SUPPORT_Z_OFFSET_M
            + ACTION_VERTICAL_OCCLUDER_HEIGHT_M * 0.5,
        )
        occluder_geometry = {
            "type": "supported_upright_cylinder",
            "radius_m": ACTION_VERTICAL_OCCLUDER_RADIUS_M,
            "height_m": ACTION_VERTICAL_OCCLUDER_HEIGHT_M,
            "support_surface": "scanned_basket_interior",
            "base_z_world_m": (
                basket_z + SCANNED_BASKET_INNER_SUPPORT_Z_OFFSET_M
            ),
        }
    elif blocked_view == "close_high":
        # A floor-standing object cannot realistically hide a mug from the
        # steep close-high camera without being implausibly tall.  Use a
        # narrow partial-cover bar whose ends rest on the basket rims.  The
        # right oblique view can still see the mug body below the bar.
        occluder_position = (
            basket_x,
            target_position[1],
            basket_z + ACTION_PARTIAL_COVER_LOCAL_CENTER_Z_M,
        )
        occluder_geometry = {
            "type": "rim_supported_partial_cover_bar",
            "full_extents_m": list(ACTION_PARTIAL_COVER_FULL_EXTENTS_M),
            "support_surface": "scanned_basket_left_and_right_rims",
            "center_z_world_m": occluder_position[2],
        }
    return {
        "variant": variant,
        "seed": seed,
        "target_position_world_m": list(target_position),
        "target_support": {
            "surface": "scanned_basket_interior",
            "base_z_offset_from_reference_m": (
                SCANNED_BASKET_INNER_SUPPORT_Z_OFFSET_M
            ),
        },
        "blocked_view": blocked_view,
        "resolving_views": {
            "close_high_only": ["close_high"],
            "right_only": ["right"],
            "either_view": ["close_high", "right"],
            "cover_removal_required": [],
        }[variant],
        "action_occluder_position_world_m": (
            list(occluder_position)
            if occluder_position is not None
            else None
        ),
        "action_occluder_target_center_distance_m": (
            math.dist(target_position, occluder_position)
            if occluder_position is not None
            else None
        ),
        "action_occluder_geometry": occluder_geometry,
        "cover_required": variant == "cover_removal_required",
        "reference_camera_positions_scene_m": {
            key: list(value)
            for key, value in ACTION_REFERENCE_CAMERA_POSITION_SCENE_M.items()
        },
        "geometry_design_status": (
            "analytic_initialization_pending_rendered_objective_mask_gate"
        ),
        "manual_annotation": False,
        "valid_for_final_evaluation": False,
    }


def _set_target_position(stage, position_world_m: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    target_prim = stage.GetPrimAtPath("/World/TargetRed")
    if not target_prim.IsValid():
        raise RuntimeError("Target red mug prim is missing")
    target_xform = UsdGeom.Xformable(target_prim)
    target_xform.ClearXformOpOrder()
    target_xform.AddTranslateOp().Set(Gf.Vec3d(*position_world_m))


def _configure_action_occluder(
    stage,
    action_layout: dict,
) -> None:
    from pxr import Gf, UsdGeom

    occluder = stage.GetPrimAtPath("/World/OccluderOrange")
    if not occluder.IsValid():
        raise RuntimeError("Orange action occluder prim is missing")
    position_world_m = tuple(
        action_layout["action_occluder_position_world_m"]
    )
    geometry = action_layout["action_occluder_geometry"]
    geometry_type = geometry["type"]
    xform = UsdGeom.Xformable(occluder)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position_world_m))
    if geometry_type == "supported_upright_cylinder":
        occluder.SetTypeName("Cylinder")
        cylinder = UsdGeom.Cylinder(occluder)
        cylinder.CreateRadiusAttr().Set(geometry["radius_m"])
        cylinder.CreateHeightAttr().Set(geometry["height_m"])
    elif geometry_type == "rim_supported_partial_cover_bar":
        occluder.SetTypeName("Cube")
        cube = UsdGeom.Cube(occluder)
        cube.CreateSizeAttr().Set(1.0)
        xform.AddScaleOp().Set(
            Gf.Vec3f(*geometry["full_extents_m"])
        )
    else:
        raise ValueError(
            f"Unsupported action occluder geometry: {geometry_type}"
        )
    UsdGeom.Gprim(occluder).CreateDisplayColorAttr(
        [Gf.Vec3f(0.92, 0.30, 0.055)]
    )
    UsdGeom.Imageable(occluder).MakeVisible()


def _raise_target_logo_for_partial_visibility(stage) -> None:
    """Keep the target attribute visible above a shallow tray rim."""
    from pxr import Gf, UsdGeom

    for name in ("WhiteLogoFront", "WhiteLogoSide", "WhiteLogoLeft"):
        prim = stage.GetPrimAtPath(f"/World/TargetRed/{name}")
        if not prim.IsValid():
            raise RuntimeError(f"Target logo prim is missing: {name}")
        xform = UsdGeom.Xformable(prim)
        translate_ops = [
            op
            for op in xform.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
        ]
        if len(translate_ops) != 1:
            raise RuntimeError(
                f"Expected one target-logo translate op: {name}"
            )
        current = translate_ops[0].Get()
        translate_ops[0].Set(
            Gf.Vec3d(float(current[0]), float(current[1]), 0.085)
        )


def _author_calibration_cover(
    stage,
    root_path: str,
    *,
    manipulable: bool = False,
    physics_calibration: dict | None = None,
) -> dict:
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    cover_path = f"{root_path}/CalibrationCover"
    if not manipulable:
        cover = UsdGeom.Cube.Define(stage, cover_path)
        cover.CreateSizeAttr().Set(1.0)
        cover.CreateDisplayColorAttr([Gf.Vec3f(0.72, 0.55, 0.31)])
        xform = UsdGeom.Xformable(cover.GetPrim())
        xform.AddTranslateOp().Set(
            Gf.Vec3d(*CALIBRATION_COVER_LOCAL_CENTER_M)
        )
        xform.AddScaleOp().Set(
            Gf.Vec3f(*CALIBRATION_COVER_FULL_EXTENTS_M)
        )
        collision = UsdPhysics.CollisionAPI.Apply(cover.GetPrim())
        collision.CreateCollisionEnabledAttr().Set(True)
        return {
            "enabled": True,
            "semantic_id": "cover_01",
            "prim_path": cover_path,
            "center_local_m": list(CALIBRATION_COVER_LOCAL_CENTER_M),
            "full_extents_m": list(CALIBRATION_COVER_FULL_EXTENTS_M),
            "manipulable": False,
            "rigid_body": False,
            "grasp_affordance": None,
            "geometry_validation": (
                "cover_bottom_above_target_top_and_full_basket_opening_"
                "coverage"
            ),
        }

    physics = {
        "cover_mass_kg": MANIPULABLE_COVER_MASS_KG,
        "cover_plate_center_local_m": list(
            CALIBRATION_COVER_LOCAL_CENTER_M
        ),
        "cover_plate_full_extents_m": list(
            CALIBRATION_COVER_FULL_EXTENTS_M
        ),
        "cover_handle_center_local_m": list(
            MANIPULABLE_COVER_HANDLE_LOCAL_CENTER_M
        ),
        "cover_handle_full_extents_m": list(
            MANIPULABLE_COVER_HANDLE_FULL_EXTENTS_M
        ),
        "physical_parameters_status": (
            "provisional_development_values_pending_lab_measurement"
        ),
    }
    if physics_calibration is not None:
        physics.update(physics_calibration)
    mass_kg = float(physics["cover_mass_kg"])
    plate_center = tuple(
        float(value) for value in physics["cover_plate_center_local_m"]
    )
    plate_extents = tuple(
        float(value) for value in physics["cover_plate_full_extents_m"]
    )
    handle_center = tuple(
        float(value) for value in physics["cover_handle_center_local_m"]
    )
    handle_extents = tuple(
        float(value) for value in physics["cover_handle_full_extents_m"]
    )
    center_of_mass, diagonal_inertia = _composite_cover_mass_properties(
        mass_kg=mass_kg,
        plate_full_extents_m=plate_extents,
        plate_center_local_m=plate_center,
        handle_full_extents_m=handle_extents,
        handle_center_local_m=handle_center,
    )

    assembly = UsdGeom.Xform.Define(stage, cover_path)
    rigid_body = UsdPhysics.RigidBodyAPI.Apply(assembly.GetPrim())
    rigid_body.CreateRigidBodyEnabledAttr().Set(True)
    mass = UsdPhysics.MassAPI.Apply(assembly.GetPrim())
    mass.CreateMassAttr().Set(mass_kg)
    mass.CreateCenterOfMassAttr().Set(
        Gf.Vec3f(*center_of_mass)
    )
    mass.CreateDiagonalInertiaAttr().Set(
        Gf.Vec3f(*diagonal_inertia)
    )
    mass.CreatePrincipalAxesAttr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    cover_body = PhysxSchema.PhysxRigidBodyAPI.Apply(assembly.GetPrim())
    cover_body.CreateLinearDampingAttr().Set(
        MANIPULABLE_COVER_LINEAR_DAMPING
    )
    cover_body.CreateAngularDampingAttr().Set(
        MANIPULABLE_COVER_ANGULAR_DAMPING
    )

    plate_path = f"{cover_path}/Plate"
    plate = UsdGeom.Cube.Define(stage, plate_path)
    plate.CreateSizeAttr().Set(1.0)
    plate.CreateDisplayColorAttr([Gf.Vec3f(0.72, 0.55, 0.31)])
    plate_xform = UsdGeom.Xformable(plate.GetPrim())
    plate_xform.AddTranslateOp().Set(
        Gf.Vec3d(*plate_center)
    )
    plate_xform.AddScaleOp().Set(
        Gf.Vec3f(*plate_extents)
    )
    plate_collision = UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
    plate_collision.CreateCollisionEnabledAttr().Set(True)

    handle_path = f"{cover_path}/Handle"
    handle = UsdGeom.Cube.Define(stage, handle_path)
    handle.CreateSizeAttr().Set(1.0)
    handle.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.18, 0.18)])
    handle_xform = UsdGeom.Xformable(handle.GetPrim())
    handle_xform.AddTranslateOp().Set(
        Gf.Vec3d(*handle_center)
    )
    handle_xform.AddScaleOp().Set(
        Gf.Vec3f(*handle_extents)
    )
    handle_collision = UsdPhysics.CollisionAPI.Apply(handle.GetPrim())
    handle_collision.CreateCollisionEnabledAttr().Set(True)

    return {
        "enabled": True,
        "semantic_id": "cover_01",
        "prim_path": cover_path,
        "plate_prim_path": plate_path,
        "center_local_m": list(plate_center),
        "full_extents_m": list(plate_extents),
        "manipulable": True,
        "rigid_body": True,
        "mass_kg": mass_kg,
        "center_of_mass_local_m": list(center_of_mass),
        "diagonal_inertia_kg_m2": list(diagonal_inertia),
        "linear_damping": MANIPULABLE_COVER_LINEAR_DAMPING,
        "angular_damping": MANIPULABLE_COVER_ANGULAR_DAMPING,
        "physical_parameters_status": physics[
            "physical_parameters_status"
        ],
        "grasp_affordance": {
            "type": "rg6_parallel_pinch_handle",
            "prim_path": handle_path,
            "center_local_m": list(handle_center),
            "full_extents_m": list(handle_extents),
            "pinch_axis": "local_y",
            "required_opening_m": (
                handle_extents[1]
            ),
        },
        "geometry_validation": (
            "dynamic_cover_plate_with_rg6_handle_pending_render_and_"
            "physics_validation"
        ),
    }


def _parse_obj(
    path: Path,
) -> tuple[
    list[tuple[float, float, float]],
    list[tuple[float, float]],
    list[int],
    list[int],
]:
    points: list[tuple[float, float, float]] = []
    texcoords: list[tuple[float, float]] = []
    vertex_indices: list[int] = []
    texcoord_indices: list[int] = []
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            if line.startswith("v "):
                fields = line.split()
                points.append(
                    (float(fields[1]), float(fields[2]), float(fields[3]))
                )
            elif line.startswith("vt "):
                fields = line.split()
                texcoords.append((float(fields[1]), float(fields[2])))
            elif line.startswith("f "):
                fields = line.split()[1:]
                if len(fields) != 3:
                    raise ValueError(
                        f"Expected triangulated OBJ, got {len(fields)} vertices"
                    )
                for field in fields:
                    components = field.split("/")
                    vertex_indices.append(int(components[0]) - 1)
                    texcoord_indices.append(int(components[1]) - 1)
    if not points or not vertex_indices:
        raise ValueError(f"OBJ has no geometry: {path}")
    if len(vertex_indices) != len(texcoord_indices):
        raise ValueError(f"OBJ UV indexing is incomplete: {path}")
    return points, texcoords, vertex_indices, texcoord_indices


def _bind_texture(stage, mesh_prim, material_path: str) -> None:
    from pxr import Sdf, UsdShade

    material = UsdShade.Material.Define(stage, material_path)
    surface = UsdShade.Shader.Define(stage, f"{material_path}/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput(
        "roughness", Sdf.ValueTypeNames.Float
    ).Set(0.55)
    surface.CreateInput(
        "metallic", Sdf.ValueTypeNames.Float
    ).Set(0.0)

    texture = UsdShade.Shader.Define(stage, f"{material_path}/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput(
        "file", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath(str(BASKET_TEXTURE)))
    texture.CreateInput(
        "sourceColorSpace", Sdf.ValueTypeNames.Token
    ).Set("sRGB")

    primvar = UsdShade.Shader.Define(stage, f"{material_path}/Primvar")
    primvar.CreateIdAttr("UsdPrimvarReader_float2")
    primvar.CreateInput(
        "varname", Sdf.ValueTypeNames.Token
    ).Set("st")
    texture.CreateInput(
        "st", Sdf.ValueTypeNames.Float2
    ).ConnectToSource(primvar.ConnectableAPI(), "result")
    surface.CreateInput(
        "diffuseColor", Sdf.ValueTypeNames.Color3f
    ).ConnectToSource(texture.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(
        surface.ConnectableAPI(), "surface"
    )
    UsdShade.MaterialBindingAPI.Apply(mesh_prim).Bind(material)


def _add_static_basket_collision(
    stage, root_path: str, *, xy_scale_factor: float
) -> dict:
    """Add a conservative five-box collision approximation.

    The textured scan remains visual-only.  Convex decomposition of the raw
    triangle mesh would close the basket opening, so a documented bottom plus
    four-wall approximation preserves the usable interior for the physics
    pilot.
    """
    from pxr import Gf, UsdGeom, UsdPhysics

    collision_root = UsdGeom.Xform.Define(
        stage, f"{root_path}/CollisionApproximation"
    )
    authored = []
    for box in BASKET_COLLISION_BOXES_LOCAL_M:
        center = (
            box["center"][0] * xy_scale_factor,
            box["center"][1] * xy_scale_factor,
            box["center"][2],
        )
        full_extents = (
            box["full_extents"][0] * xy_scale_factor,
            box["full_extents"][1] * xy_scale_factor,
            box["full_extents"][2],
        )
        path = f"{collision_root.GetPath()}/{box['name']}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr().Set(1.0)
        cube.CreateDisplayOpacityAttr([0.0])
        UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*center))
        xform.AddScaleOp().Set(Gf.Vec3f(*full_extents))
        collision = UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        collision.CreateCollisionEnabledAttr().Set(True)
        authored.append(
            {
                "prim_path": path,
                "center_local_m": list(center),
                "full_extents_m": list(full_extents),
            }
        )
    return {
        "type": "five_box_static_approximation",
        "exact_mesh_collision": False,
        "purpose": "single_seed_physics_safety_smoke",
        "xy_scale_factor_from_perception_scene": xy_scale_factor,
        "boxes": authored,
    }


def replace_procedural_basket_with_scan(
    stage,
    *,
    active_occlusion_pilot: bool = False,
    collision_physics_pilot: bool = False,
    calibration_scene_variant: str | None = None,
    calibration_seed: int = 0,
    cover_physics_calibration: dict | None = None,
) -> dict:
    """Replace the container visual while preserving its authored world frame."""
    from pxr import Gf, Sdf, Usd, UsdGeom

    if not BASKET_OBJ.is_file() or not BASKET_TEXTURE.is_file():
        raise FileNotFoundError(
            f"Local LIBERO basket asset is incomplete: {BASKET_ASSET_DIR}"
        )
    root_path = "/World/OpenContainer"
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError("Open-container benchmark prim is missing")

    for child in root.GetChildren():
        child.SetActive(False)
    if (
        calibration_scene_variant is not None
        and calibration_scene_variant
        not in ALL_CALIBRATION_SCENE_VARIANTS
    ):
        raise ValueError(
            f"Unknown calibration scene variant: {calibration_scene_variant}"
        )
    if active_occlusion_pilot and calibration_scene_variant is not None:
        raise ValueError(
            "active_occlusion_pilot and calibration_scene_variant are mutually exclusive"
        )

    # The original "rear" distractor is almost completely hidden by this
    # thicker scanned rim in all three reachable wrist views.  Put it beside
    # the basket so the deterministic pilot genuinely contains two visible
    # red candidates: the logo mug inside and a no-logo mug outside.
    outside_mug = stage.GetPrimAtPath("/World/RearRedCandidate")
    if not outside_mug.IsValid():
        raise RuntimeError("Rear red candidate benchmark prim is missing")
    outside_xform = UsdGeom.Xformable(outside_mug)
    outside_xform.ClearXformOpOrder()
    outside_position = (
        PHYSICS_CLEARANCE_OUTSIDE_MUG_POSITION_WORLD_M
        if collision_physics_pilot
        else VISIBLE_OUTSIDE_MUG_POSITION_WORLD_M
    )
    outside_xform.AddTranslateOp().Set(Gf.Vec3d(*outside_position))

    occlusion_metadata = {
        "enabled": False,
        "purpose": "normal_two_candidate_visibility_pilot",
    }
    basket_position = tuple(
        float(value)
        for value in UsdGeom.Xformable(root)
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        .ExtractTranslation()
    )
    target_prim = stage.GetPrimAtPath("/World/TargetRed")
    if not target_prim.IsValid():
        raise RuntimeError("Target red mug prim is missing")
    target_position = tuple(
        float(value)
        for value in UsdGeom.Xformable(target_prim)
        .ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        .ExtractTranslation()
    )
    cover_metadata = {"enabled": False}
    action_occluder_enabled = False
    calibration_ground_truth = None
    target_support = {
        "surface": "scanned_basket_interior",
        "base_z_offset_from_reference_m": (
            SCANNED_BASKET_INNER_SUPPORT_Z_OFFSET_M
        ),
        "validation": (
            "vertical_rays_over_mug_footprint_against_scanned_mesh"
        ),
    }
    if calibration_scene_variant == "outside":
        _set_target_position(
            stage, CALIBRATION_OUTSIDE_TARGET_POSITION_WORLD_M
        )
        target_position = CALIBRATION_OUTSIDE_TARGET_POSITION_WORLD_M
        target_support = {
            "surface": "table",
            "base_z_world_m": CALIBRATION_OUTSIDE_TARGET_POSITION_WORLD_M[2],
            "validation": "target_mug_bottom_origin_on_table_top",
        }
    elif calibration_scene_variant == "behind_ambiguous":
        ambiguous_layout = compute_behind_ambiguous_target_layout(
            basket_position,
            calibration_seed,
        )
        target_position = tuple(
            ambiguous_layout["target_position_world_m"]
        )
        _set_target_position(stage, target_position)
        _raise_target_logo_for_partial_visibility(stage)
        target_support = {
            "surface": "table",
            "base_z_world_m": target_position[2],
            "validation": (
                "target_mug_bottom_on_table_behind_shallow_basket_with_"
                "analytic_back_wall_clearance"
            ),
        }
        occlusion_metadata = {
            "enabled": True,
            "semantic_id": "container",
            "target_position_world_m": list(target_position),
            "basket_planar_clearance_m": ambiguous_layout[
                "basket_planar_clearance_m"
            ],
            "occlusion_source": "shallow_scanned_basket_back_alignment",
            "explicit_occluder_primitive_visible": False,
            "purpose": (
                "center_view_cannot_resolve_inside_vs_behind_while_"
                "reachable_reobservation_views_resolve_outside_behind"
            ),
            "manual_annotation": False,
            "generator_revision": "behind-ambiguous-ray-jitter-v2",
        }
    elif calibration_scene_variant == "behind_boundary_unknown":
        boundary_layout = compute_behind_boundary_unknown_target_layout(
            basket_position,
            calibration_seed,
        )
        target_position = tuple(
            boundary_layout["target_position_world_m"]
        )
        _set_target_position(stage, target_position)
        _raise_target_logo_for_partial_visibility(stage)
        target_support = {
            "surface": "table",
            "base_z_world_m": target_position[2],
            "validation": (
                "target_mug_bottom_on_table_outside_basket_with_"
                "analytic_planar_clearance"
            ),
        }
        occlusion_metadata = {
            "enabled": True,
            "semantic_id": "container",
            "target_position_world_m": list(target_position),
            "basket_planar_clearance_m": boundary_layout[
                "basket_planar_clearance_m"
            ],
            "occlusion_source": "camera_relative_far_edge_boundary",
            "explicit_occluder_primitive_visible": False,
            "purpose": (
                "calibrate_objective_camera_relative_behind_unknown"
            ),
            "manual_annotation": False,
            "generator_revision": "behind-boundary-unknown-v1",
            "analytic_layout": boundary_layout,
        }
    elif calibration_scene_variant == "empty_cover_then_right":
        negative_layout = compute_behind_ambiguous_target_layout(
            basket_position,
            calibration_seed,
        )
        target_position = (
            basket_position[0] + NEGATIVE_EVIDENCE_TARGET_X_OFFSET_M,
            basket_position[1] + NEGATIVE_EVIDENCE_TARGET_Y_OFFSET_M,
            negative_layout["target_position_world_m"][2],
        )
        negative_layout["target_position_world_m"] = list(target_position)
        negative_layout["negative_evidence_target_offsets_m"] = [
            NEGATIVE_EVIDENCE_TARGET_X_OFFSET_M,
            NEGATIVE_EVIDENCE_TARGET_Y_OFFSET_M,
        ]
        _set_target_position(stage, target_position)
        _raise_target_logo_for_partial_visibility(stage)
        target_support = {
            "surface": "table",
            "base_z_world_m": target_position[2],
            "validation": (
                "target_mug_bottom_on_table_behind_covered_empty_basket"
            ),
        }
        cover_metadata = _author_calibration_cover(
            stage,
            root_path,
            manipulable=True,
            physics_calibration=cover_physics_calibration,
        )
        occlusion_metadata = {
            "enabled": True,
            "semantic_id": "basket_01",
            "target_position_world_m": list(target_position),
            "basket_planar_clearance_m": negative_layout[
                "basket_planar_clearance_m"
            ],
            "occlusion_source": "covered_empty_basket_front_of_target",
            "purpose": (
                "physical_empty_cover_negative_evidence_then_right_reobservation"
            ),
            "manual_annotation": False,
            "generator_revision": "empty-cover-negative-evidence-v1",
            "render_validation_required": True,
        }
    elif (
        calibration_scene_variant
        in ACTION_DIFFERENTIATING_SCENE_VARIANTS
    ):
        action_layout = compute_action_differentiating_layout(
            basket_position,
            calibration_scene_variant,
            calibration_seed,
        )
        target_position = tuple(
            action_layout["target_position_world_m"]
        )
        _set_target_position(stage, target_position)
        target_support = {
            "surface": "scanned_basket_interior",
            "base_z_offset_from_reference_m": (
                SCANNED_BASKET_INNER_SUPPORT_Z_OFFSET_M
            ),
            "validation": (
                "target_mug_bottom_on_scanned_basket_support_pending_"
                "rendered_action_difference_gate"
            ),
        }
        occluder_position = action_layout[
            "action_occluder_position_world_m"
        ]
        if occluder_position is not None:
            _configure_action_occluder(stage, action_layout)
            action_occluder_enabled = True
        if calibration_scene_variant == "cover_removal_required":
            cover_metadata = _author_calibration_cover(
                stage,
                root_path,
                manipulable=True,
                physics_calibration=cover_physics_calibration,
            )
        occlusion_metadata = {
            "enabled": True,
            "semantic_id": (
                "cover_01"
                if calibration_scene_variant
                == "cover_removal_required"
                else (
                    "occluder_orange"
                    if action_occluder_enabled
                    else "container"
                )
            ),
            "target_position_world_m": list(target_position),
            "action_scene_layout": action_layout,
            "purpose": (
                "causal_view_action_differentiation_for_belief_mpc"
            ),
            "manual_annotation": False,
            "generator_revision": "action-differentiating-layout-v1",
            "render_validation_required": True,
        }
    elif calibration_scene_variant in (
        "inside_clear",
        "rim_occluded",
        "covered_unknown",
    ):
        target_layout = compute_rim_occluded_target_layout(
            target_position,
            basket_position,
        )
        if calibration_scene_variant == "inside_clear":
            # Put the target near the basket center.  The absence of an added
            # occluder/cover makes this the easy visible-inside case.
            target_layout["target_position_world_m"][1] = basket_position[1]
            target_layout["occlusion_source"] = "none_intended"
        _set_target_position(
            stage, tuple(target_layout["target_position_world_m"])
        )
        target_position = tuple(target_layout["target_position_world_m"])
        if calibration_scene_variant == "rim_occluded":
            occlusion_metadata = {
                "enabled": True,
                "semantic_id": "container",
                **target_layout,
                "disabled_primitive": "/World/OccluderOrange",
                "purpose": (
                    "scanned_basket_rim_partial_occlusion_for_active_reobservation"
                ),
                "manual_annotation": False,
            }
        elif calibration_scene_variant == "covered_unknown":
            cover_metadata = _author_calibration_cover(stage, root_path)
    elif active_occlusion_pilot:
        occluder_prim = stage.GetPrimAtPath("/World/OccluderOrange")
        if not occluder_prim.IsValid():
            raise RuntimeError("Orange occluder benchmark prim is missing")
        target_layout = compute_rim_occluded_target_layout(
            target_position,
            basket_position,
        )
        _set_target_position(
            stage, tuple(target_layout["target_position_world_m"])
        )
        target_position = tuple(target_layout["target_position_world_m"])
        occlusion_metadata = {
            "enabled": True,
            "semantic_id": "container",
            **target_layout,
            "disabled_primitive": "/World/OccluderOrange",
            "purpose": (
                "use_the_scanned_basket_rim_to_reduce_center_visibility_while_preserving_"
                "lateral_and_close_high_reobservation"
            ),
            "manual_annotation": False,
        }
    occluder_prim = stage.GetPrimAtPath("/World/OccluderOrange")
    if occluder_prim.IsValid() and not action_occluder_enabled:
        UsdGeom.Imageable(occluder_prim).MakeInvisible()
    if calibration_scene_variant is not None:
        calibration_ground_truth = factorized_calibration_ground_truth(
            calibration_scene_variant
        )

    points, texcoords, vertex_indices, texcoord_indices = _parse_obj(
        BASKET_OBJ
    )
    mesh = UsdGeom.Mesh.Define(stage, f"{root_path}/ScannedBasketMesh")
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr(
        [3] * (len(vertex_indices) // 3)
    )
    mesh.CreateFaceVertexIndicesAttr(vertex_indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    mesh_xform = UsdGeom.Xformable(mesh.GetPrim())
    basket_scale = (
        AMBIGUOUS_TRAY_SCALE_XYZ
        if calibration_scene_variant
        in (
            "behind_ambiguous",
            "behind_boundary_unknown",
            "empty_cover_then_right",
        )
        else PERCEPTION_BASKET_SCALE_XYZ
    )
    # OBJ vertices and collision boxes remain basket-local; the existing
    # container root supplies the shared world pose for rendering and physics.
    mesh_xform.AddScaleOp().Set(Gf.Vec3f(*basket_scale))

    st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.faceVarying,
    )
    st.Set([Gf.Vec2f(*uv) for uv in texcoords])
    st.SetIndices(texcoord_indices)
    _bind_texture(stage, mesh.GetPrim(), f"{root_path}/ScannedMaterial")
    # The physics pilot uses the open five-box proxy instead of the raw scan;
    # a convex hull would seal the basket and create unsafe false contacts.
    collision_metadata = (
        _add_static_basket_collision(
            stage,
            root_path,
            xy_scale_factor=(
                PHYSICS_CLEARANCE_BASKET_SCALE_XYZ[0]
                / PERCEPTION_BASKET_SCALE_XYZ[0]
            ),
        )
        if collision_physics_pilot
        else {
            "type": "not_added_perception_only",
            "exact_mesh_collision": False,
        }
    )

    return {
        "visual_class": "open_scanned_basket",
        "procedural_geometry": False,
        "source_asset": str(BASKET_OBJ),
        "texture_asset": str(BASKET_TEXTURE),
        "source_project": "LIBERO",
        "source_project_url": "https://github.com/Lifelong-Robot-Learning/LIBERO",
        "license": "CC BY 4.0 (LIBERO datasets/assets attribution)",
        "scale_xyz": list(basket_scale),
        "approximate_size_world_m": [
            0.173412 * basket_scale[0],
            0.162194 * basket_scale[1],
            0.147613 * basket_scale[2],
        ],
        "collision_geometry": collision_metadata,
        "layout_override": {
            "rear_red_candidate_position_world_m": list(
                outside_position
            ),
            "rear_red_candidate_relation": "outside",
            "reason": (
                "keep_both_red_candidates_visible_for_identity_and_relation_pilot"
            ),
        },
        "active_occlusion": occlusion_metadata,
        "calibration_scene_variant": calibration_scene_variant,
        "calibration_ground_truth": calibration_ground_truth,
        "calibration_generator_revision": (
            (
                (
                    (
                        "behind-ambiguous-ray-jitter-v2"
                        if calibration_scene_variant == "behind_ambiguous"
                        else "behind-boundary-unknown-v1"
                    )
                    if calibration_scene_variant
                    in ("behind_ambiguous", "behind_boundary_unknown")
                    else "empty-cover-negative-evidence-v1"
                )
                if calibration_scene_variant
                in (
                    "behind_ambiguous",
                    "behind_boundary_unknown",
                    "empty_cover_then_right",
                )
                else "factorized-scene-generator-v1"
            )
            if calibration_scene_variant
            not in ACTION_DIFFERENTIATING_SCENE_VARIANTS
            else "action-differentiating-layout-v1"
        ),
        "target_position_world_m": list(target_position),
        "target_support": target_support,
        "cover": cover_metadata,
    }
