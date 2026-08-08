"""Deterministic, relation-preserving layouts for the benchmark pilot."""

from __future__ import annotations

import math

import numpy as np


PRIM_PATHS = {
    "target_red": "/World/TargetRed",
    "occluder_orange": "/World/OccluderOrange",
    "distractor_yellow": "/World/DistractorYellow",
    "distractor_blue": "/World/DistractorBlue",
    "distractor_green": "/World/DistractorGreen",
    "boundary_purple": "/World/BoundaryPurple",
    "rear_red_candidate": "/World/RearRedCandidate",
}


def generate_layout(seed: int) -> dict:
    """Generate bounded positions while preserving every relation label."""
    if seed < 0:
        raise ValueError("Seed must be non-negative")
    rng = np.random.default_rng(seed)
    target = np.asarray(
        [
            rng.uniform(0.43, 0.55),
            rng.uniform(-0.005, 0.095),
            # A tall target remains inside the 12 cm container wall while its
            # top strip is visible from the initial wrist view.  This creates
            # partial rather than complete occlusion for the first
            # integration experiment.
            0.84,
        ]
    )
    camera_xy = np.asarray([-0.03, -0.42])
    toward_camera = math.atan2(
        camera_xy[1] - target[1],
        camera_xy[0] - target[0],
    )
    occluder_angle = toward_camera + rng.uniform(-0.22, 0.22)
    occluder_distance = rng.uniform(0.095, 0.115)
    occluder = np.asarray(
        [
            target[0] + occluder_distance * math.cos(occluder_angle),
            target[1] + occluder_distance * math.sin(occluder_angle),
            0.805,
        ]
    )
    positions = {
        "target_red": target,
        "occluder_orange": occluder,
        "distractor_yellow": np.asarray(
            [rng.uniform(0.25, 0.31), rng.uniform(0.16, 0.20), 0.80]
        ),
        "distractor_blue": np.asarray(
            [rng.uniform(0.43, 0.58), rng.uniform(-0.35, -0.29), 0.79]
        ),
        "distractor_green": np.asarray(
            [rng.uniform(0.76, 0.84), rng.uniform(-0.21, -0.13), 0.80]
        ),
        "boundary_purple": np.asarray(
            [rng.uniform(0.79, 0.81), rng.uniform(0.04, 0.16), 0.805]
        ),
        "rear_red_candidate": np.asarray(
            [rng.uniform(0.61, 0.71), rng.uniform(0.30, 0.34), 0.80]
        ),
    }
    return {
        "schema_version": "seeded-benchmark-layout-v1",
        "seed": seed,
        "generator": "relation_preserving_uniform_v1",
        "positions_world_m": {
            name: [float(value) for value in position]
            for name, position in positions.items()
        },
        "relations_preserved": {
            "target_red": "inside",
            "occluder_orange": "in_front_of_target",
            "distractor_yellow": "inside",
            "distractor_blue": "outside_near",
            "distractor_green": "outside_near",
            "boundary_purple": "near_boundary",
            "rear_red_candidate": "behind_container",
        },
        "geometry_overrides_world_m": {
            "target_red_scale": [0.045, 0.045, 0.12]
        },
        "manual_annotation": False,
        "valid_for_final_evaluation": False,
    }


def apply_layout(stage, layout: dict) -> None:
    """Apply a generated layout to an already-open USD stage."""
    from pxr import Gf, UsdGeom

    for name, position in layout["positions_world_m"].items():
        prim = stage.GetPrimAtPath(PRIM_PATHS[name])
        if not prim.IsValid():
            raise RuntimeError(f"Seeded benchmark prim is missing: {PRIM_PATHS[name]}")
        translate_ops = [
            op
            for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
        ]
        if len(translate_ops) != 1:
            raise RuntimeError(
                f"Expected one translate op for {PRIM_PATHS[name]}, "
                f"found {len(translate_ops)}"
            )
        translate_ops[0].Set(Gf.Vec3d(*position))
    target_scale = layout.get("geometry_overrides_world_m", {}).get(
        "target_red_scale"
    )
    if target_scale is not None:
        target = stage.GetPrimAtPath(PRIM_PATHS["target_red"])
        scale_ops = [
            op
            for op in UsdGeom.Xformable(target).GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeScale
        ]
        if len(scale_ops) != 1:
            raise RuntimeError(
                "Expected one scale op for the seeded target geometry"
            )
        scale_ops[0].Set(Gf.Vec3f(*target_scale))
