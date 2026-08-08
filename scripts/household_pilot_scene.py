"""Procedural household-shaped assets for the seed-0 perception pilot.

This module deliberately avoids downloading or copying third-party meshes.
It replaces only the visual geometry of the existing benchmark target,
rear distractor, and open container. The benchmark prim paths and simulator
semantic labels remain unchanged so hidden evaluation masks still work.
"""

from __future__ import annotations

import math


def hollow_mug_mesh(
    radial_segments: int = 48,
    handle_segments: int = 24,
    handle_tube_segments: int = 10,
) -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    """Return points, face counts, and face indices for an open mug with handle."""
    if radial_segments < 12 or handle_segments < 6 or handle_tube_segments < 6:
        raise ValueError("Mug mesh resolution is too low")

    outer_radius = 0.041
    inner_radius = 0.034
    bottom_z = 0.0
    inner_bottom_z = 0.007
    top_z = 0.102
    points: list[tuple[float, float, float]] = []
    counts: list[int] = []
    indices: list[int] = []

    rings = {}
    for name, radius, z in (
        ("outer_bottom", outer_radius, bottom_z),
        ("outer_top", outer_radius, top_z),
        ("inner_top", inner_radius, top_z),
        ("inner_bottom", inner_radius, inner_bottom_z),
    ):
        rings[name] = []
        for index in range(radial_segments):
            angle = 2.0 * math.pi * index / radial_segments
            rings[name].append(len(points))
            points.append((radius * math.cos(angle), radius * math.sin(angle), z))

    def quad(a: int, b: int, c: int, d: int) -> None:
        counts.append(4)
        indices.extend((a, b, c, d))

    for index in range(radial_segments):
        nxt = (index + 1) % radial_segments
        quad(
            rings["outer_bottom"][index],
            rings["outer_bottom"][nxt],
            rings["outer_top"][nxt],
            rings["outer_top"][index],
        )
        quad(
            rings["inner_top"][index],
            rings["inner_top"][nxt],
            rings["inner_bottom"][nxt],
            rings["inner_bottom"][index],
        )
        quad(
            rings["outer_top"][index],
            rings["outer_top"][nxt],
            rings["inner_top"][nxt],
            rings["inner_top"][index],
        )
        quad(
            rings["inner_bottom"][index],
            rings["inner_bottom"][nxt],
            rings["outer_bottom"][nxt],
            rings["outer_bottom"][index],
        )

    # A half-torus handle in the X-Z plane, attached to the mug's right side.
    handle_grid: list[list[int]] = []
    major_radius = 0.034
    tube_radius = 0.007
    center_x = outer_radius
    center_z = 0.055
    for segment in range(handle_segments + 1):
        theta = -math.pi / 2.0 + math.pi * segment / handle_segments
        row = []
        for tube_index in range(handle_tube_segments):
            phi = 2.0 * math.pi * tube_index / handle_tube_segments
            radial = major_radius + tube_radius * math.cos(phi)
            row.append(len(points))
            points.append(
                (
                    center_x + radial * math.cos(theta),
                    tube_radius * math.sin(phi),
                    center_z + radial * math.sin(theta),
                )
            )
        handle_grid.append(row)
    for segment in range(handle_segments):
        for tube_index in range(handle_tube_segments):
            nxt = (tube_index + 1) % handle_tube_segments
            quad(
                handle_grid[segment][tube_index],
                handle_grid[segment][nxt],
                handle_grid[segment + 1][nxt],
                handle_grid[segment + 1][tube_index],
            )
    return points, counts, indices


def _set_color(prim, color: tuple[float, float, float]) -> None:
    from pxr import Gf, UsdGeom

    UsdGeom.Gprim(prim).CreateDisplayColorAttr(
        [Gf.Vec3f(*color)]
    )


def _replace_with_mug(
    stage,
    prim_path: str,
    *,
    position: tuple[float, float, float],
    color: tuple[float, float, float],
    white_logo: bool,
) -> None:
    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Mug benchmark prim is missing: {prim_path}")
    prim.SetTypeName("Xform")
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))

    points, counts, indices = hollow_mug_mesh()
    mesh = UsdGeom.Mesh.Define(stage, f"{prim_path}/MugMesh")
    mesh.CreatePointsAttr([Gf.Vec3f(*point) for point in points])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    _set_color(mesh.GetPrim(), color)

    if white_logo:
        logo_front = UsdGeom.Cube.Define(
            stage, f"{prim_path}/WhiteLogoFront"
        )
        logo_front.CreateSizeAttr(1.0)
        front_xform = UsdGeom.Xformable(logo_front.GetPrim())
        front_xform.AddTranslateOp().Set(
            Gf.Vec3d(0.0, -0.0415, 0.061)
        )
        front_xform.AddScaleOp().Set(Gf.Vec3f(0.028, 0.002, 0.020))
        _set_color(logo_front.GetPrim(), (0.96, 0.96, 0.92))

        # The benchmark target is viewed from both the front and the right.
        # Model the same printed logo wrapping slightly around the side so the
        # re-observation reveals semantic evidence rather than only geometry.
        logo_side = UsdGeom.Cube.Define(
            stage, f"{prim_path}/WhiteLogoSide"
        )
        logo_side.CreateSizeAttr(1.0)
        side_xform = UsdGeom.Xformable(logo_side.GetPrim())
        side_xform.AddTranslateOp().Set(
            Gf.Vec3d(0.0415, 0.0, 0.061)
        )
        side_xform.AddScaleOp().Set(Gf.Vec3f(0.002, 0.028, 0.020))
        _set_color(logo_side.GetPrim(), (0.96, 0.96, 0.92))
        logo_left = UsdGeom.Cube.Define(
            stage, f"{prim_path}/WhiteLogoLeft"
        )
        logo_left.CreateSizeAttr(1.0)
        left_xform = UsdGeom.Xformable(logo_left.GetPrim())
        left_xform.AddTranslateOp().Set(
            Gf.Vec3d(-0.0415, 0.0, 0.061)
        )
        left_xform.AddScaleOp().Set(Gf.Vec3f(0.002, 0.028, 0.020))
        _set_color(logo_left.GetPrim(), (0.96, 0.96, 0.92))


def _replace_container_with_basket(stage) -> None:
    from pxr import Gf, UsdGeom

    root_path = "/World/OpenContainer"
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError("Open-container benchmark prim is missing")

    # Preserve the bottom, but replace the solid walls and rims with basket slats.
    for name in (
        "WallFront",
        "WallBack",
        "WallLeft",
        "WallRight",
        "FrontRim",
        "BackRim",
    ):
        child = stage.GetPrimAtPath(f"{root_path}/{name}")
        if child.IsValid():
            child.SetActive(False)
    bottom = stage.GetPrimAtPath(f"{root_path}/Bottom")
    if bottom.IsValid():
        _set_color(bottom, (0.34, 0.17, 0.055))

    basket_color = (0.56, 0.30, 0.09)
    rim_color = (0.30, 0.12, 0.035)

    def add_cube(
        name: str,
        position: tuple[float, float, float],
        scale: tuple[float, float, float],
        color: tuple[float, float, float] = basket_color,
    ) -> None:
        cube = UsdGeom.Cube.Define(stage, f"{root_path}/BasketVisual/{name}")
        cube.CreateSizeAttr(1.0)
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*position))
        xform.AddScaleOp().Set(Gf.Vec3f(*scale))
        _set_color(cube.GetPrim(), color)

    # Horizontal open-weave rails.
    for side_name, y in (("Front", -0.242), ("Back", 0.242)):
        for rail_index, z in enumerate((0.025, 0.060, 0.095)):
            add_cube(
                f"{side_name}Rail{rail_index}",
                (0.0, y, z),
                (0.35, 0.012, 0.010),
            )
        for post_index, x in enumerate((-0.32, -0.16, 0.0, 0.16, 0.32)):
            add_cube(
                f"{side_name}Post{post_index}",
                (x, y, 0.062),
                (0.010, 0.013, 0.125),
            )
        add_cube(
            f"{side_name}Rim",
            (0.0, y, 0.126),
            (0.37, 0.020, 0.018),
            rim_color,
        )

    for side_name, x in (("Left", -0.332), ("Right", 0.332)):
        for rail_index, z in enumerate((0.025, 0.060, 0.095)):
            add_cube(
                f"{side_name}Rail{rail_index}",
                (x, 0.0, z),
                (0.012, 0.225, 0.010),
            )
        for post_index, y in enumerate((-0.21, -0.105, 0.0, 0.105, 0.21)):
            add_cube(
                f"{side_name}Post{post_index}",
                (x, y, 0.062),
                (0.013, 0.010, 0.125),
            )
        add_cube(
            f"{side_name}Rim",
            (x, 0.0, 0.126),
            (0.020, 0.245, 0.018),
            rim_color,
        )


def apply_household_pilot_visuals(stage, seeded_layout: dict) -> dict:
    """Replace debug target/container visuals without changing semantic paths."""
    positions = seeded_layout["positions_world_m"]
    container_bottom_top_z = 0.755 + 0.018 * 0.5
    table_top_z = 0.735 + 0.008 * 0.5

    target_xy = positions["target_red"][:2]
    rear_xy = positions["rear_red_candidate"][:2]
    _replace_with_mug(
        stage,
        "/World/TargetRed",
        position=(float(target_xy[0]), float(target_xy[1]), container_bottom_top_z),
        color=(0.72, 0.018, 0.012),
        white_logo=True,
    )
    _replace_with_mug(
        stage,
        "/World/RearRedCandidate",
        position=(float(rear_xy[0]), float(rear_xy[1]), table_top_z),
        color=(0.62, 0.025, 0.018),
        white_logo=False,
    )
    _replace_container_with_basket(stage)
    return {
        "schema_version": "household-perception-scene-v1",
        "seed": int(seeded_layout["seed"]),
        "target": {
            "semantic_id": "target_red",
            "visual_class": "red_mug_with_white_logo",
            "relation": "inside",
            "procedural_geometry": True,
        },
        "target_distractor": {
            "semantic_id": "rear_red_candidate",
            "visual_class": "red_mug_without_logo",
            "relation": "behind",
            "procedural_geometry": True,
        },
        "reference": {
            "semantic_id": "container",
            "visual_class": "open_slatted_basket",
            "procedural_geometry": True,
        },
        "manual_annotation": False,
        "training_performed": False,
        "valid_for_final_evaluation": False,
    }
