"""UR10e observation poses and wrist-camera RGB-D capture helpers.

Positions are expressed in the authored USD world frame in meters. Camera
calibration follows USD's row-vector transform convention, with the optical
axis along camera -Z and image rows increasing downward.
"""

import json
import math
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image


JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

SEMANTIC_OBJECTS = {
    "/World/TargetRed": "target_red",
    "/World/DistractorBlue": "distractor_blue",
    "/World/OpenContainer": "container",
}

BENCHMARK_SEMANTIC_OBJECTS = {
    "/World/TargetRed": "target_red",
    "/World/OccluderOrange": "occluder_orange",
    "/World/DistractorYellow": "distractor_yellow",
    "/World/DistractorBlue": "distractor_blue",
    "/World/DistractorGreen": "distractor_green",
    "/World/BoundaryPurple": "boundary_purple",
    "/World/RearRedCandidate": "rear_red_candidate",
    "/World/OpenContainer": "container",
}

BENCHMARK_ID_COLORS = {
    "target_red": (255, 0, 0),
    "occluder_orange": (255, 128, 0),
    "distractor_yellow": (255, 255, 0),
    "distractor_blue": (0, 0, 255),
    "distractor_green": (0, 255, 0),
    "boundary_purple": (128, 0, 255),
    "rear_red_candidate": (255, 0, 255),
    "container": (0, 255, 255),
}

# RTX tone mapping shifts emissive inputs. These prototypes were measured from
# the deterministic benchmark ID pass; ambiguous hue pairs are split spatially.
BENCHMARK_RENDERED_ID_PROTOTYPES = {
    "target_red": (248, 47, 43),
    "occluder_orange": (248, 240, 54),
    "distractor_yellow": (248, 240, 54),
    "distractor_blue": (25, 45, 240),
    "distractor_green": (3, 240, 12),
    "boundary_purple": (240, 30, 245),
    "rear_red_candidate": (240, 30, 245),
    "container": (10, 235, 240),
}
BENCHMARK_ID_CHROMATICITY_DISTANCE_THRESHOLD = 0.30
OBJECTIVE_OCCLUSION_THRESHOLDS = {
    "no_max_fraction_exclusive": 0.10,
    "partial_max_fraction_exclusive": 0.60,
}


def _split_id_pair_by_horizontal_components(
    instance_ids: np.ndarray,
    pair: tuple[int, int],
    left_id: int,
    right_id: int,
) -> None:
    """Resolve known color-managed ID pairs using their disconnected components."""
    mask = np.isin(instance_ids, pair)
    visited = np.zeros(mask.shape, dtype=bool)
    components = []
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        visited[y, x] = True
        pixels = []
        while queue:
            cy, cx = queue.popleft()
            pixels.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and mask[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        if len(pixels) >= 20:
            components.append(pixels)
    if len(components) < 2:
        return
    largest = sorted(components, key=len, reverse=True)[:2]
    largest.sort(key=lambda pixels: sum(x for _, x in pixels) / len(pixels))
    for object_id, pixels in zip((left_id, right_id), largest):
        ys, xs = zip(*pixels)
        instance_ids[np.asarray(ys), np.asarray(xs)] = object_id


def _split_id_pair_by_component_area(
    instance_ids: np.ndarray,
    pair: tuple[int, int],
    *,
    small_id: int,
    large_id: int,
    minimum_large_pixels: int = 500,
) -> None:
    """Resolve a same-color pair whose semantic objects differ strongly in area.

    The moved rear red mug can appear to either side of the small purple
    boundary marker, so a left/right rule is no longer stable. The mug is the
    largest connected component in every reachable benchmark view.
    """
    mask = np.isin(instance_ids, pair)
    visited = np.zeros(mask.shape, dtype=bool)
    components = []
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if visited[y, x]:
            continue
        queue = deque([(int(y), int(x))])
        visited[y, x] = True
        pixels = []
        while queue:
            cy, cx = queue.popleft()
            pixels.append((cy, cx))
            for ny, nx in (
                (cy - 1, cx),
                (cy + 1, cx),
                (cy, cx - 1),
                (cy, cx + 1),
            ):
                if (
                    0 <= ny < height
                    and 0 <= nx < width
                    and mask[ny, nx]
                    and not visited[ny, nx]
                ):
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        if len(pixels) >= 20:
            components.append(pixels)
    if not components:
        return
    largest = max(components, key=len)
    for pixels in components:
        object_id = (
            large_id
            if pixels is largest and len(pixels) >= minimum_large_pixels
            else small_id
        )
        ys, xs = zip(*pixels)
        instance_ids[np.asarray(ys), np.asarray(xs)] = object_id


def add_scene_labels(stage, semantic_objects=None) -> None:
    """Attach class labels to the scene prims used by capture annotators."""
    import omni.replicator.core as rep

    semantic_objects = semantic_objects or SEMANTIC_OBJECTS
    for prim_path, label in semantic_objects.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Semantic object prim is missing: {prim_path}")
        rep.functional.modify.semantics(prim, {"class": label}, mode="add")


def load_observation_config(project_root: Path) -> dict:
    path = project_root / "configs" / "sim" / "observation_poses.json"
    with path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    if tuple(config["joint_order"]) != JOINT_NAMES:
        raise ValueError("Observation pose joint order does not match UR10e")
    return config


def set_pose(robot, config: dict, pose_name: str, update, warmup_frames: int = 8) -> None:
    values = config["poses_rad"][pose_name]
    indices = [robot.dof_names.index(name) for name in JOINT_NAMES]
    robot.set_dof_positions(values, dof_indices=indices)
    # Direct state placement must also clear the pre-existing tensor velocity.
    # Otherwise the next timeline resume can integrate a large stale velocity
    # even though the position target itself is valid.
    robot.set_dof_velocities(
        [0.0] * len(indices),
        dof_indices=indices,
    )
    robot.set_dof_position_targets(values, dof_indices=indices)
    for _ in range(warmup_frames):
        update()


def move_pose_interpolated(
    robot,
    config: dict,
    pose_name: str,
    update,
    contact_force_reader=None,
    collision_checker=None,
) -> dict:
    """Move to a configured pose through joint-space position-target waypoints."""
    settings = config["trajectory"]
    indices = [robot.dof_names.index(name) for name in JOINT_NAMES]
    current_array = robot.get_dof_positions().numpy()
    current = current_array[0] if current_array.ndim > 1 else current_array
    start = np.asarray(current[indices], dtype=np.float64)
    target = np.asarray(config["poses_rad"][pose_name], dtype=np.float64)
    lower_raw, upper_raw = robot.get_dof_limits(dof_indices=indices)
    lower_array, upper_array = lower_raw.numpy(), upper_raw.numpy()
    lower = lower_array[0] if lower_array.ndim > 1 else lower_array
    upper = upper_array[0] if upper_array.ndim > 1 else upper_array
    if not np.all(np.isfinite(start)):
        raise RuntimeError(
            f"Cannot start {pose_name} motion from non-finite UR10e state: "
            f"{start.tolist()}"
        )
    if np.any(start < lower) or np.any(start > upper):
        raise RuntimeError(
            f"Cannot start {pose_name} motion outside UR10e joint limits: "
            f"start={start.tolist()} lower={lower.tolist()} "
            f"upper={upper.tolist()}"
        )
    if np.any(target < lower) or np.any(target > upper):
        raise ValueError(f"Target pose {pose_name} violates UR10e joint limits")

    maximum_delta = float(np.max(np.abs(target - start)))
    steps = max(1, math.ceil(maximum_delta / settings["maximum_joint_step_rad"]))
    maximum_contact_force = 0.0
    collision_detected = False
    state_invalid = False
    failure_reason = None
    collision_check_method = (
        "physx_contact_force"
        if contact_force_reader is not None
        else "world_aabb_overlap"
        if collision_checker is not None
        else "none"
    )
    waypoint_records = []

    for step in range(1, steps + 1):
        alpha = step / steps
        waypoint = start + alpha * (target - start)
        if np.any(waypoint < lower) or np.any(waypoint > upper):
            raise ValueError(
                f"Interpolated waypoint {step} violates UR10e joint limits: "
                f"start={start.tolist()} target={target.tolist()} "
                f"waypoint={waypoint.tolist()} lower={lower.tolist()} "
                f"upper={upper.tolist()}"
            )
        robot.set_dof_position_targets(waypoint.tolist(), dof_indices=indices)
        for _ in range(settings["control_frames_per_waypoint"]):
            update()
            if contact_force_reader is not None:
                force = float(contact_force_reader())
                maximum_contact_force = max(maximum_contact_force, force)
                if force > settings["contact_force_abort_threshold_n"]:
                    collision_detected = True
                    break
            if collision_checker is not None and collision_checker():
                collision_detected = True
                break
        measured_array = robot.get_dof_positions().numpy()
        measured_vector = measured_array[0] if measured_array.ndim > 1 else measured_array
        measured = np.asarray(measured_vector[indices], dtype=np.float64)
        if not np.all(np.isfinite(measured)):
            state_invalid = True
            failure_reason = "non_finite_joint_state"
        elif np.any(measured < lower) or np.any(measured > upper):
            state_invalid = True
            failure_reason = "measured_joint_limit_violation"
        waypoint_records.append(
            {
                "step": step,
                "alpha": alpha,
                "maximum_waypoint_error_rad": (
                    float(np.max(np.abs(measured - waypoint)))
                    if not state_invalid
                    else None
                ),
                "maximum_contact_force_n": maximum_contact_force,
                "finite_joint_state": bool(np.all(np.isfinite(measured))),
            }
        )
        if state_invalid:
            break
        if collision_detected:
            robot.set_dof_position_targets(measured.tolist(), dof_indices=indices)
            break

    settle_frames = 0
    if not collision_detected and not state_invalid:
        for settle_frames in range(1, settings["maximum_final_settle_frames"] + 1):
            robot.set_dof_position_targets(target.tolist(), dof_indices=indices)
            update()
            if contact_force_reader is not None:
                force = float(contact_force_reader())
                maximum_contact_force = max(maximum_contact_force, force)
                if force > settings["contact_force_abort_threshold_n"]:
                    collision_detected = True
                    break
            if collision_checker is not None and collision_checker():
                collision_detected = True
                break
            measured_array = robot.get_dof_positions().numpy()
            measured_vector = measured_array[0] if measured_array.ndim > 1 else measured_array
            measured = np.asarray(measured_vector[indices], dtype=np.float64)
            if not np.all(np.isfinite(measured)):
                state_invalid = True
                failure_reason = "non_finite_joint_state"
                break
            if np.any(measured < lower) or np.any(measured > upper):
                state_invalid = True
                failure_reason = "measured_joint_limit_violation"
                break
            if float(np.max(np.abs(measured - target))) <= settings["final_tolerance_rad"]:
                break

    measured_array = robot.get_dof_positions().numpy()
    measured_vector = measured_array[0] if measured_array.ndim > 1 else measured_array
    measured = np.asarray(measured_vector[indices], dtype=np.float64)
    finite_final_state = bool(np.all(np.isfinite(measured)))
    maximum_final_error = (
        float(np.max(np.abs(measured - target)))
        if finite_final_state
        else None
    )
    return {
        "status": (
            "state_abort"
            if state_invalid
            else "collision_abort"
            if collision_detected
            else "completed"
            if maximum_final_error is not None
            and maximum_final_error <= settings["final_tolerance_rad"]
            else "final_tolerance_failed"
        ),
        "failure_reason": failure_reason,
        "pose_name": pose_name,
        "motion_mode": "interpolated_joint_position_targets",
        "waypoint_count": steps,
        "control_frames_per_waypoint": settings["control_frames_per_waypoint"],
        "settle_frames": settle_frames,
        "joint_limits_checked": True,
        "collision_monitoring_enabled": (
            contact_force_reader is not None or collision_checker is not None
        ),
        "collision_check_method": collision_check_method,
        "contact_force_monitoring_enabled": contact_force_reader is not None,
        "collision_detected": collision_detected,
        "finite_final_joint_state": finite_final_state,
        "maximum_contact_force_n": maximum_contact_force,
        "contact_force_abort_threshold_n": settings["contact_force_abort_threshold_n"],
        "requested_joint_positions_rad": target.tolist(),
        "measured_joint_positions_rad": measured.tolist(),
        "maximum_joint_error_rad": maximum_final_error,
        "verification_tolerance_rad": settings["final_tolerance_rad"],
        "waypoints": waypoint_records,
    }


def align_tool_and_camera(stage, ee_prim, rg6_prim, camera_prim, config: dict, ee_pose=None) -> None:
    """Express the tool and wrist-camera world poses in their parent frames."""
    import omni.usd
    from pxr import Gf, UsdGeom

    robot_system = stage.GetPrimAtPath("/World/RobotSystem")
    if ee_pose is None:
        ee_world = omni.usd.get_world_transform_matrix(ee_prim)
    else:
        position, quaternion_wxyz = ee_pose
        quat = Gf.Quatd(float(quaternion_wxyz[0]), Gf.Vec3d(*map(float, quaternion_wxyz[1:4])))
        ee_world = Gf.Matrix4d(1.0)
        ee_world.SetRotate(quat)
        ee_world.SetTranslateOnly(Gf.Vec3d(*map(float, position)))
    parent_world = omni.usd.get_world_transform_matrix(robot_system)

    # Gf composes row vectors, so a child local transform is world * parent^-1.
    rg6_xform = UsdGeom.Xformable(rg6_prim)
    rg6_xform.ClearXformOpOrder()
    rg6_xform.MakeMatrixXform().Set(ee_world * parent_world.GetInverse())

    ee_position = ee_world.ExtractTranslation()
    offset = config["camera"]["mount_offset_ee_m"]
    target = Gf.Vec3d(*config["camera"]["look_at_world_m"])
    camera_position = Gf.Vec3d(
        ee_position[0] + offset[0], ee_position[1] + offset[1], ee_position[2] + offset[2]
    )
    view_direction = target - camera_position
    view_direction.Normalize()
    camera_position += view_direction * float(config["camera"]["optical_clearance_toward_target_m"])
    camera_world = Gf.Matrix4d(1.0)
    camera_world.SetLookAt(camera_position, target, Gf.Vec3d(0, 0, 1))
    camera_world = camera_world.GetInverse()
    rg6_world = omni.usd.get_world_transform_matrix(rg6_prim)
    camera_xform = UsdGeom.Xformable(camera_prim)
    camera_xform.ClearXformOpOrder()
    camera_xform.MakeMatrixXform().Set(camera_world * rg6_world.GetInverse())


def align_world_camera_to_ee(
    camera_prim,
    config: dict,
    ee_pose,
) -> None:
    """Place a world-root wrist camera from a physical end-effector pose."""
    from pxr import Gf, UsdGeom

    position, _quaternion_wxyz = ee_pose
    ee_position = Gf.Vec3d(*map(float, position))
    offset = config["camera"]["mount_offset_ee_m"]
    target = Gf.Vec3d(*config["camera"]["look_at_world_m"])
    camera_position = Gf.Vec3d(
        ee_position[0] + offset[0],
        ee_position[1] + offset[1],
        ee_position[2] + offset[2],
    )
    view_direction = target - camera_position
    view_direction.Normalize()
    camera_position += view_direction * float(
        config["camera"]["optical_clearance_toward_target_m"]
    )
    camera_world = Gf.Matrix4d(1.0)
    camera_world.SetLookAt(
        camera_position, target, Gf.Vec3d(0, 0, 1)
    )
    camera_world = camera_world.GetInverse()
    camera_xform = UsdGeom.Xformable(camera_prim)
    camera_xform.ClearXformOpOrder()
    camera_xform.MakeMatrixXform().Set(camera_world)


def make_gripper_kinematic(rg6_prim) -> None:
    """Keep the provisional visual gripper rigidly attachable until assembly is calibrated."""
    from pxr import UsdGeom, UsdPhysics

    for prim in UsdGeom.Imageable(rg6_prim).GetPrim().GetStage().Traverse():
        if not prim.GetPath().HasPrefix(rg6_prim.GetPath()):
            continue
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Set(True)


def configure_camera(camera_prim, config: dict) -> None:
    from pxr import Gf, UsdGeom

    camera = UsdGeom.Camera(camera_prim)
    settings = config["camera"]
    camera.GetFocalLengthAttr().Set(float(settings["focal_length_mm"]))
    camera.GetHorizontalApertureAttr().Set(float(settings["horizontal_aperture_mm"]))
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(*settings["clipping_range_m"]))


def camera_calibration(camera_prim, resolution: tuple[int, int]) -> dict:
    """Return pinhole parameters for metric distance-to-camera backprojection."""
    import omni.usd
    from pxr import UsdGeom

    width, height = map(int, resolution)
    camera = UsdGeom.Camera(camera_prim)
    focal_length_mm = float(camera.GetFocalLengthAttr().Get())
    horizontal_aperture_mm = float(
        camera.GetHorizontalApertureAttr().Get()
    )
    focal_pixels = width * focal_length_mm / horizontal_aperture_mm
    camera_to_world = np.asarray(
        omni.usd.get_world_transform_matrix(camera_prim),
        dtype=np.float64,
    )
    return {
        "schema_version": "pinhole-distance-camera-calibration-v1",
        "resolution": [width, height],
        "fx_pixels": focal_pixels,
        "fy_pixels": focal_pixels,
        "cx_pixels": (width - 1) * 0.5,
        "cy_pixels": (height - 1) * 0.5,
        "focal_length_mm": focal_length_mm,
        "horizontal_aperture_mm": horizontal_aperture_mm,
        "camera_to_world_row_vector_matrix": camera_to_world.tolist(),
        "camera_axes": {
            "right": "+X",
            "up": "+Y",
            "forward": "-Z",
        },
        "pixel_axis": {"u": "right", "v": "down"},
        "depth_definition": "euclidean_distance_to_camera_center_m",
        "homogeneous_transform_convention": (
            "[x_camera,y_camera,z_camera,1] @ camera_to_world"
        ),
    }


def create_fixed_overview_camera(stage, config: dict):
    from pxr import Gf, UsdGeom

    settings = config["overview_camera"]
    camera = UsdGeom.Camera.Define(stage, settings["path"])
    camera.GetFocalLengthAttr().Set(float(settings["focal_length_mm"]))
    camera.GetHorizontalApertureAttr().Set(
        float(settings["horizontal_aperture_mm"])
    )
    camera.GetClippingRangeAttr().Set(
        Gf.Vec2f(*settings["clipping_range_m"])
    )
    position = Gf.Vec3d(*settings["position_world_m"])
    target = Gf.Vec3d(*settings["look_at_world_m"])
    up_axis = Gf.Vec3d(*settings["up_axis"])
    camera_world = Gf.Matrix4d(1.0)
    camera_world.SetLookAt(position, target, up_axis)
    camera_world = camera_world.GetInverse()
    camera_xform = UsdGeom.Xformable(camera.GetPrim())
    camera_xform.ClearXformOpOrder()
    camera_xform.MakeMatrixXform().Set(camera_world)
    camera.GetPrim().SetCustomDataByKey("purpose", settings["purpose"])
    camera.GetPrim().SetCustomDataByKey(
        "used_by_vlm_or_planner",
        settings["used_by_vlm_or_planner"],
    )
    return camera.GetPrim()


def create_capture_pipeline(camera_path: str, resolution: tuple[int, int]):
    """Create pixel-aligned RGB and metric-depth annotators for one camera."""
    import omni.replicator.core as rep

    # Sharing one render product keeps RGB and distance samples on the same
    # camera pose, resolution, and pixel grid.
    render_product = rep.create.render_product(camera_path, resolution)
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    depth = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb.attach(render_product)
    depth.attach(render_product)
    return rep, render_product, rgb, depth


def create_rgb_capture_pipeline(camera_path: str, resolution: tuple[int, int]):
    """Create the RGB-only stream used by the external overview camera."""
    import omni.replicator.core as rep

    render_product = rep.create.render_product(camera_path, resolution)
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(render_product)
    return render_product, rgb


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _colorize_instance_ids(instance_ids: np.ndarray) -> np.ndarray:
    preview = np.zeros((*instance_ids.shape, 3), dtype=np.uint8)
    for instance_id in np.unique(instance_ids):
        if int(instance_id) == 0:
            continue
        value = int(instance_id)
        preview[instance_ids == instance_id] = (
            (value * 53) % 255,
            (value * 97) % 255,
            (value * 193) % 255,
        )
    return preview


def _object_statistics(
    instance_ids: np.ndarray,
    id_to_labels: dict,
    depth: np.ndarray,
    expected_objects=None,
) -> dict:
    statistics = {}
    expected_objects = expected_objects or SEMANTIC_OBJECTS.values()
    for expected_label in expected_objects:
        matching_ids = []
        for instance_id, labels in id_to_labels.items():
            if isinstance(labels, dict) and str(labels.get("class")) == expected_label:
                matching_ids.append(int(instance_id))
        mask = np.isin(instance_ids, matching_ids)
        ys, xs = np.nonzero(mask)
        valid_depth = depth[mask & np.isfinite(depth)]
        statistics[expected_label] = {
            "instance_ids": matching_ids,
            "visible": bool(xs.size),
            "pixel_count": int(xs.size),
            "visible_fraction": float(xs.size / mask.size),
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if xs.size else None,
            "depth_valid_pixels": int(valid_depth.size),
            "depth_min_m": float(valid_depth.min()) if valid_depth.size else None,
            "depth_mean_m": float(valid_depth.mean()) if valid_depth.size else None,
            "depth_max_m": float(valid_depth.max()) if valid_depth.size else None,
        }
    return statistics


def _instance_ids_from_scene_colors(rgb: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, dict]:
    """Deterministic fallback for this fixed-color scene when RTX SDG is unstable."""
    rgb_float = rgb.astype(np.float32)
    red, green, blue = rgb_float[:, :, 0], rgb_float[:, :, 1], rgb_float[:, :, 2]
    finite = np.isfinite(depth)
    target = finite & (red > 180) & ((red - green) > 55) & ((red - blue) > 45)
    distractor = finite & (blue > 145) & ((blue - red) > 35) & ((blue - green) > 20)
    channel_spread = np.maximum.reduce([red, green, blue]) - np.minimum.reduce([red, green, blue])
    container = finite & (red > 210) & (green > 205) & (blue > 190) & (channel_spread < 42)
    container &= ~target & ~distractor

    instance_ids = np.zeros(depth.shape, dtype=np.uint32)
    instance_ids[container] = 3
    instance_ids[distractor] = 2
    instance_ids[target] = 1
    labels = {
        "1": {"class": "target_red", "primPath": "/World/TargetRed"},
        "2": {"class": "distractor_blue", "primPath": "/World/DistractorBlue"},
        "3": {"class": "container", "primPath": "/World/OpenContainer"},
    }
    return instance_ids, labels


def render_benchmark_id_pass(stage, rep, rgb_annotator) -> tuple[np.ndarray, dict]:
    """Render temporary unique object colors, then restore visible scene colors."""
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    material_root = "/World/__BenchmarkIdMaterials"
    restored_bindings = {}
    materials = {}
    background_material = UsdShade.Material.Define(stage, f"{material_root}/background")
    background_shader = UsdShade.Shader.Define(
        stage, f"{material_root}/background/Shader"
    )
    background_shader.CreateIdAttr("UsdPreviewSurface")
    background_shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.0, 0.0, 0.0)
    )
    background_shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(0.0, 0.0, 0.0)
    )
    background_material.CreateSurfaceOutput().ConnectToSource(
        background_shader.ConnectableAPI(), "surface"
    )
    for label, color in BENCHMARK_ID_COLORS.items():
        material = UsdShade.Material.Define(stage, f"{material_root}/{label}")
        shader = UsdShade.Shader.Define(stage, f"{material_root}/{label}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        normalized_color = Gf.Vec3f(*(channel / 255.0 for channel in color))
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(normalized_color)
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(normalized_color)
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        materials[label] = material

    for prim in stage.Traverse():
        if prim.GetPath().HasPrefix(material_root):
            continue
        gprim = UsdGeom.Gprim(prim)
        if not gprim:
            continue
        binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
        relationship = binding_api.GetDirectBindingRel()
        restored_bindings[str(prim.GetPath())] = (
            relationship,
            relationship.GetTargets(),
        )
        binding_api.Bind(background_material)

    for prim_path, label in BENCHMARK_SEMANTIC_OBJECTS.items():
        root_path = stage.GetPrimAtPath(prim_path).GetPath()
        for prim in stage.Traverse():
            if not prim.GetPath().HasPrefix(root_path):
                continue
            gprim = UsdGeom.Gprim(prim)
            if not gprim:
                continue
            binding_api = UsdShade.MaterialBindingAPI.Apply(prim)
            binding_api.Bind(materials[label])

    for _ in range(2):
        rep.orchestrator.step(rt_subframes=4)
    color_pass = np.asarray(rgb_annotator.get_data())[:, :, :3].astype(np.float32).copy()

    for relationship, previous_targets in restored_bindings.values():
        if previous_targets:
            relationship.SetTargets(previous_targets)
        else:
            relationship.ClearTargets(True)
    stage.RemovePrim(material_root)
    for _ in range(2):
        rep.orchestrator.step(rt_subframes=4)

    instance_ids, labels = classify_benchmark_color_pass(color_pass)
    return instance_ids, labels, color_pass.astype(np.uint8)


def render_target_amodal_id_pass(
    stage,
    rep,
    rgb_annotator,
    target_prim_path: str = "/World/TargetRed",
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Render the target silhouette with every non-target Gprim hidden.

    This counterfactual pass keeps the camera and target pose fixed.  It is
    simulator-only evaluation ground truth and must never be exposed to the
    learned perception stack, belief update, or planner.
    """
    from pxr import UsdGeom

    target_prim = stage.GetPrimAtPath(target_prim_path)
    if not target_prim.IsValid():
        raise RuntimeError(f"Amodal target prim is missing: {target_prim_path}")
    target_path = target_prim.GetPath()
    restored_visibility = []
    for prim in stage.Traverse():
        if prim.GetPath().HasPrefix(target_path):
            continue
        gprim = UsdGeom.Gprim(prim)
        if not gprim:
            continue
        imageable = UsdGeom.Imageable(prim)
        visibility = imageable.GetVisibilityAttr()
        restored_visibility.append(
            (
                visibility,
                visibility.HasAuthoredValueOpinion(),
                visibility.Get(),
            )
        )
        imageable.MakeInvisible()

    try:
        for _ in range(2):
            rep.orchestrator.step(rt_subframes=4)
        return render_benchmark_id_pass(stage, rep, rgb_annotator)
    finally:
        for visibility, had_authored_value, previous_value in restored_visibility:
            if had_authored_value:
                visibility.Set(previous_value)
            else:
                visibility.Clear()
        for _ in range(2):
            rep.orchestrator.step(rt_subframes=4)


def render_reference_removed_target_id_pass(
    stage,
    rep,
    rgb_annotator,
    reference_prim_path: str = "/World/OpenContainer",
) -> tuple[np.ndarray, dict, np.ndarray]:
    """Render the scene after hiding only the reference container subtree."""
    from pxr import UsdGeom

    reference_prim = stage.GetPrimAtPath(reference_prim_path)
    if not reference_prim.IsValid():
        raise RuntimeError(
            f"Reference occluder prim is missing: {reference_prim_path}"
        )
    reference_path = reference_prim.GetPath()
    restored_visibility = []
    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(reference_path):
            continue
        gprim = UsdGeom.Gprim(prim)
        if not gprim:
            continue
        imageable = UsdGeom.Imageable(prim)
        visibility = imageable.GetVisibilityAttr()
        restored_visibility.append(
            (
                visibility,
                visibility.HasAuthoredValueOpinion(),
                visibility.Get(),
            )
        )
        imageable.MakeInvisible()

    try:
        for _ in range(2):
            rep.orchestrator.step(rt_subframes=4)
        return render_benchmark_id_pass(stage, rep, rgb_annotator)
    finally:
        for visibility, had_authored_value, previous_value in restored_visibility:
            if had_authored_value:
                visibility.Set(previous_value)
            else:
                visibility.Clear()
        for _ in range(2):
            rep.orchestrator.step(rt_subframes=4)


def objective_occlusion_measurement(
    visible_target_mask: np.ndarray,
    amodal_target_mask: np.ndarray,
    thresholds: dict | None = None,
) -> dict:
    """Measure scene occlusion from aligned visible and amodal target masks."""
    thresholds = thresholds or OBJECTIVE_OCCLUSION_THRESHOLDS
    visible_target_mask = np.asarray(visible_target_mask, dtype=bool)
    amodal_target_mask = np.asarray(amodal_target_mask, dtype=bool)
    if visible_target_mask.shape != amodal_target_mask.shape:
        raise ValueError(
            "Visible and amodal target masks must have the same shape: "
            f"{visible_target_mask.shape} != {amodal_target_mask.shape}"
        )
    raw_visible_pixels = int(visible_target_mask.sum())
    amodal_pixels = int(amodal_target_mask.sum())
    supported_visible_target_mask = (
        visible_target_mask & amodal_target_mask
    )
    visible_pixels = int(supported_visible_target_mask.sum())
    spill_pixels = raw_visible_pixels - visible_pixels
    result = {
        "schema_version": "objective-amodal-occlusion-v1",
        "definition": (
            "1 - pixels(actual_target_id_mask intersect target_only_amodal_"
            "support) / pixels(target_only_amodal_support), at the same "
            "camera and target pose"
        ),
        "valid": amodal_pixels > 0,
        "raw_visible_target_id_pixels": raw_visible_pixels,
        "visible_target_pixels": visible_pixels,
        "amodal_target_pixels": amodal_pixels,
        "out_of_amodal_id_spill_pixels": spill_pixels,
        "visible_mask_clipped_to_amodal_support": True,
        "hidden_target_pixels": (
            max(amodal_pixels - visible_pixels, 0)
            if amodal_pixels > 0
            else None
        ),
        "visible_fraction_of_amodal": None,
        "occlusion_fraction": None,
        "severity": "unknown",
        "fully_hidden": None,
        "thresholds": dict(thresholds),
        "simulator_ground_truth_only": True,
        "exposed_to_model_or_planner": False,
    }
    if amodal_pixels == 0:
        return result

    visible_fraction = min(max(visible_pixels / amodal_pixels, 0.0), 1.0)
    occlusion_fraction = 1.0 - visible_fraction
    no_threshold = float(thresholds["no_max_fraction_exclusive"])
    partial_threshold = float(
        thresholds["partial_max_fraction_exclusive"]
    )
    if not 0.0 <= no_threshold < partial_threshold <= 1.0:
        raise ValueError(f"Invalid objective occlusion thresholds: {thresholds}")
    if occlusion_fraction < no_threshold:
        severity = "no"
    elif occlusion_fraction < partial_threshold:
        severity = "partial"
    else:
        severity = "severe"
    result.update(
        {
            "visible_fraction_of_amodal": visible_fraction,
            "occlusion_fraction": occlusion_fraction,
            "severity": severity,
            "fully_hidden": visible_pixels == 0,
        }
    )
    return result


def objective_reference_occlusion_measurement(
    visible_target_mask: np.ndarray,
    reference_removed_target_mask: np.ndarray,
    amodal_target_mask: np.ndarray,
    thresholds: dict | None = None,
) -> dict:
    """Measure target pixels revealed specifically by removing the reference."""
    thresholds = thresholds or OBJECTIVE_OCCLUSION_THRESHOLDS
    visible = np.asarray(visible_target_mask, dtype=bool)
    reference_removed = np.asarray(
        reference_removed_target_mask, dtype=bool
    )
    amodal = np.asarray(amodal_target_mask, dtype=bool)
    if visible.shape != amodal.shape or reference_removed.shape != amodal.shape:
        raise ValueError(
            "Visible, reference-removed, and amodal masks must share shape: "
            f"{visible.shape}, {reference_removed.shape}, {amodal.shape}"
        )
    supported_visible = visible & amodal
    supported_reference_removed = reference_removed & amodal
    revealed_by_reference_removal = (
        supported_reference_removed & ~supported_visible
    )
    amodal_pixels = int(amodal.sum())
    revealed_pixels = int(revealed_by_reference_removal.sum())
    result = {
        "schema_version": "objective-reference-occlusion-v1",
        "definition": (
            "pixels newly revealed after hiding only the reference container "
            "subtree, divided by target-only amodal pixels, with camera and "
            "target pose fixed"
        ),
        "valid": amodal_pixels > 0,
        "raw_visible_target_id_pixels": int(visible.sum()),
        "raw_reference_removed_target_id_pixels": int(
            reference_removed.sum()
        ),
        "visible_target_pixels_in_amodal_support": int(
            supported_visible.sum()
        ),
        "reference_removed_target_pixels_in_amodal_support": int(
            supported_reference_removed.sum()
        ),
        "reference_revealed_target_pixels": revealed_pixels,
        "amodal_target_pixels": amodal_pixels,
        "reference_occlusion_fraction": None,
        "severity": "unknown",
        "thresholds": dict(thresholds),
        "reference_prim_path": "/World/OpenContainer",
        "simulator_ground_truth_only": True,
        "exposed_to_model_or_planner": False,
    }
    if amodal_pixels == 0:
        return result
    fraction = min(max(revealed_pixels / amodal_pixels, 0.0), 1.0)
    no_threshold = float(thresholds["no_max_fraction_exclusive"])
    partial_threshold = float(
        thresholds["partial_max_fraction_exclusive"]
    )
    if not 0.0 <= no_threshold < partial_threshold <= 1.0:
        raise ValueError(
            f"Invalid objective reference occlusion thresholds: {thresholds}"
        )
    if fraction < no_threshold:
        severity = "no"
    elif fraction < partial_threshold:
        severity = "partial"
    else:
        severity = "severe"
    result.update(
        {
            "reference_occlusion_fraction": fraction,
            "severity": severity,
        }
    )
    return result


OBJECTIVE_BEHIND_THRESHOLDS = {
    "minimum_target_bbox_overlap": 0.10,
    "far_edge_abstention_m": 0.02,
    "minimum_reference_occlusion_fraction": 0.10,
}


def _mask_bbox_xyxy(mask: np.ndarray) -> list[int] | None:
    pixel_y, pixel_x = np.nonzero(np.asarray(mask, dtype=bool))
    if pixel_x.size == 0:
        return None
    return [
        int(pixel_x.min()),
        int(pixel_y.min()),
        int(pixel_x.max()),
        int(pixel_y.max()),
    ]


def _bbox_overlap_fraction(
    source: list[int], reference: list[int]
) -> float:
    sx0, sy0, sx1, sy1 = source
    rx0, ry0, rx1, ry1 = reference
    width = max(0.0, min(sx1, rx1) - max(sx0, rx0) + 1.0)
    height = max(0.0, min(sy1, ry1) - max(sy0, ry0) + 1.0)
    source_area = max(0.0, sx1 - sx0 + 1.0) * max(
        0.0, sy1 - sy0 + 1.0
    )
    return width * height / source_area if source_area > 0.0 else 0.0


def objective_camera_relative_behind_measurement(
    *,
    target_center_world_m: list[float],
    reference_bounds_world_m: dict,
    camera_to_world_row_vector_matrix: list[list[float]],
    target_amodal_mask: np.ndarray,
    reference_visible_mask: np.ndarray,
    membership: str,
    reference_occlusion_fraction: float | None,
    thresholds: dict | None = None,
) -> dict:
    """Compute simulator-only camera-relative behind ground truth.

    The metric combines exact simulator geometry with rendered instance-ID
    support.  It is never exposed to perception or planning.
    """
    thresholds = thresholds or OBJECTIVE_BEHIND_THRESHOLDS
    target = np.asarray(target_center_world_m, dtype=np.float64)
    lower = np.asarray(
        reference_bounds_world_m["lower"], dtype=np.float64
    )
    upper = np.asarray(
        reference_bounds_world_m["upper"], dtype=np.float64
    )
    camera_matrix = np.asarray(
        camera_to_world_row_vector_matrix, dtype=np.float64
    )
    if target.shape != (3,) or lower.shape != (3,) or upper.shape != (3,):
        raise ValueError("Target center and reference bounds must be 3D")
    if camera_matrix.shape != (4, 4):
        raise ValueError("Camera-to-world matrix must be 4x4")
    if membership not in {"inside", "outside"}:
        raise ValueError(f"Unsupported world membership: {membership}")
    target_mask = np.asarray(target_amodal_mask, dtype=bool)
    reference_mask = np.asarray(reference_visible_mask, dtype=bool)
    if target_mask.shape != reference_mask.shape:
        raise ValueError(
            "Target-amodal and reference-visible masks must share shape"
        )
    target_bbox = _mask_bbox_xyxy(target_mask)
    reference_bbox = _mask_bbox_xyxy(reference_mask)
    result = {
        "schema_version": "objective-camera-relative-behind-v1",
        "definition": (
            "target center beyond the reference far edge along the current "
            "camera-to-reference XY ray with projected target/reference "
            "overlap; an inside target occluded by the reference-facing "
            "surface also counts as behind"
        ),
        "valid": False,
        "label": "unknown",
        "reason": "missing_projected_support",
        "target_center_world_m": target.tolist(),
        "reference_bounds_world_m": {
            "lower": lower.tolist(),
            "upper": upper.tolist(),
        },
        "target_amodal_bbox_xyxy": target_bbox,
        "reference_visible_bbox_xyxy": reference_bbox,
        "target_bbox_overlap_with_reference": None,
        "camera_ray_center_offset_m": None,
        "reference_half_extent_along_ray_m": None,
        "far_edge_offset_m": None,
        "membership_world_ground_truth": membership,
        "reference_occlusion_fraction": reference_occlusion_fraction,
        "thresholds": dict(thresholds),
        "simulator_ground_truth_only": True,
        "exposed_to_model_or_planner": False,
    }
    if target_bbox is None or reference_bbox is None:
        return result

    reference_center = 0.5 * (lower + upper)
    camera_center = camera_matrix[3, :3]
    camera_to_reference_xy = reference_center[:2] - camera_center[:2]
    norm = float(np.linalg.norm(camera_to_reference_xy))
    if not math.isfinite(norm) or norm <= 1e-9:
        result["reason"] = "degenerate_camera_reference_direction"
        return result
    direction = camera_to_reference_xy / norm
    half_extent = 0.5 * (
        abs(direction[0]) * (upper[0] - lower[0])
        + abs(direction[1]) * (upper[1] - lower[1])
    )
    center_offset = float(
        np.dot(target[:2] - reference_center[:2], direction)
    )
    far_edge_offset = center_offset - half_extent
    overlap = _bbox_overlap_fraction(target_bbox, reference_bbox)
    minimum_overlap = float(thresholds["minimum_target_bbox_overlap"])
    abstention = float(thresholds["far_edge_abstention_m"])
    minimum_reference_occlusion = float(
        thresholds["minimum_reference_occlusion_fraction"]
    )
    inside_reference_occlusion = (
        membership == "inside"
        and reference_occlusion_fraction is not None
        and reference_occlusion_fraction >= minimum_reference_occlusion
        and overlap >= minimum_overlap
    )
    if inside_reference_occlusion:
        label = "yes"
        reason = "inside_target_hidden_by_reference_facing_surface"
    elif far_edge_offset > abstention and overlap >= minimum_overlap:
        label = "yes"
        reason = "target_beyond_reference_far_edge_along_camera_ray"
    elif abs(far_edge_offset) <= abstention and overlap >= minimum_overlap:
        label = "unknown"
        reason = "target_near_reference_far_edge_abstention_band"
    else:
        label = "no"
        reason = "no_camera_relative_behind_evidence"
    result.update(
        {
            "valid": True,
            "label": label,
            "reason": reason,
            "target_bbox_overlap_with_reference": overlap,
            "camera_ray_center_offset_m": center_offset,
            "reference_half_extent_along_ray_m": float(half_extent),
            "far_edge_offset_m": float(far_edge_offset),
        }
    )
    return result


def classify_benchmark_color_pass(color_pass: np.ndarray) -> tuple[np.ndarray, dict]:
    """Convert the measured RTX color-ID pass into benchmark instance IDs."""
    color_pass = color_pass.astype(np.float32)
    brightness = color_pass.max(axis=2)
    chroma = color_pass.max(axis=2) - color_pass.min(axis=2)
    normalized = color_pass / np.maximum(color_pass.sum(axis=2, keepdims=True), 1.0)
    prototypes = []
    for color in BENCHMARK_RENDERED_ID_PROTOTYPES.values():
        prototype = np.asarray(color, dtype=np.float32)
        prototypes.append(prototype / prototype.sum())
    distances = np.stack(
        [np.linalg.norm(normalized - prototype, axis=2) for prototype in prototypes],
        axis=2,
    )
    closest = np.argmin(distances, axis=2)
    minimum_distance = np.min(distances, axis=2)
    # Procedural household materials render the emissive target ID as a more
    # saturated red than the original debug cube. A 0.20 chromaticity radius
    # therefore dropped most valid mug pixels even though the closest class
    # was unambiguous. The nearest competing prototype remains more than 0.6
    # away for the measured mug pixels, so 0.30 recovers the complete ID mask
    # without merging the red and orange classes.
    valid = (
        (brightness > 35)
        & (chroma > 35)
        & (
            minimum_distance
            < BENCHMARK_ID_CHROMATICITY_DISTANCE_THRESHOLD
        )
    )
    instance_ids = np.where(valid, closest + 1, 0).astype(np.uint32)
    _split_id_pair_by_horizontal_components(
        instance_ids, pair=(2, 3), left_id=3, right_id=2
    )
    _split_id_pair_by_component_area(
        instance_ids,
        pair=(6, 7),
        small_id=6,
        large_id=7,
    )
    labels = {}
    for instance_id, (label, color) in enumerate(BENCHMARK_ID_COLORS.items(), start=1):
        prim_path = next(
            path for path, value in BENCHMARK_SEMANTIC_OBJECTS.items() if value == label
        )
        labels[str(instance_id)] = {"class": label, "primPath": prim_path}
    return instance_ids, labels


def save_capture(
    output_root: Path,
    pose_name: str,
    rgb_data,
    depth_data,
    instance_override=None,
    target_amodal_override=None,
    target_reference_removed_override=None,
    overview_rgb_data=None,
    camera_provenance=None,
    camera_calibration_data=None,
    objective_behind_geometry=None,
) -> None:
    """Write one aligned RGB-D sample and its simulator-only diagnostics."""
    pose_dir = output_root / pose_name
    pose_dir.mkdir(parents=True, exist_ok=True)
    rgba = np.asarray(rgb_data)
    Image.fromarray(rgba[:, :, :3].astype(np.uint8), "RGB").save(pose_dir / "rgb.png")
    overview_shape = None
    if overview_rgb_data is not None:
        overview_rgba = np.asarray(overview_rgb_data)
        overview_rgb = overview_rgba[:, :, :3].astype(np.uint8)
        Image.fromarray(overview_rgb, "RGB").save(
            pose_dir / "overview_rgb.png"
        )
        overview_shape = list(overview_rgb.shape)
    elif (pose_dir / "overview_rgb.png").is_file():
        with Image.open(pose_dir / "overview_rgb.png") as existing_overview:
            overview_shape = [
                existing_overview.height,
                existing_overview.width,
                3,
            ]
    depth = np.asarray(depth_data, dtype=np.float32)
    np.save(pose_dir / "depth_m.npy", depth)
    finite = np.isfinite(depth)
    # Keep metric depth untouched on disk; the 8-bit image is only a quick
    # visual check and must not be used for geometry or localization.
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if finite.any():
        near, far = np.percentile(depth[finite], [2, 98])
        if far > near:
            preview[finite] = np.clip((depth[finite] - near) / (far - near) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(preview, "L").save(pose_dir / "depth_preview.png")
    if instance_override is None:
        instance_ids, id_to_labels = _instance_ids_from_scene_colors(rgba[:, :, :3], depth)
        expected_objects = SEMANTIC_OBJECTS.values()
        segmentation_source = "rgb_color_key_fallback"
        segmentation_limitation = (
            "Replace with RTX instance annotator after Isaac Sim synthetic-data crash is resolved."
        )
    else:
        instance_ids, id_to_labels, raw_color_pass = instance_override
        Image.fromarray(raw_color_pass, "RGB").save(pose_dir / "instance_color_pass.png")
        expected_objects = BENCHMARK_SEMANTIC_OBJECTS.values()
        segmentation_source = "temporary_unique_color_id_render_pass"
        segmentation_limitation = (
            "Simulator-only fallback; visually validate masks and replace with native "
            "instance IDs when the RTX annotator is stable."
        )
    np.save(pose_dir / "instance_ids.npy", instance_ids)
    Image.fromarray(_colorize_instance_ids(instance_ids), "RGB").save(
        pose_dir / "instance_segmentation.png"
    )
    (pose_dir / "instance_labels.json").write_text(
        json.dumps(_json_safe(id_to_labels), indent=2), encoding="utf-8"
    )
    object_statistics = _object_statistics(
        instance_ids, id_to_labels, depth, expected_objects=expected_objects
    )
    (pose_dir / "objects.json").write_text(
        json.dumps(object_statistics, indent=2), encoding="utf-8"
    )
    objective_occlusion_file = None
    objective_reference_occlusion_file = None
    objective_camera_relative_behind_file = None
    if target_amodal_override is not None:
        (
            amodal_instance_ids,
            amodal_id_to_labels,
            amodal_color_pass,
        ) = target_amodal_override
        visible_target_ids = [
            int(instance_id)
            for instance_id, labels in id_to_labels.items()
            if isinstance(labels, dict)
            and labels.get("class") == "target_red"
        ]
        amodal_target_ids = [
            int(instance_id)
            for instance_id, labels in amodal_id_to_labels.items()
            if isinstance(labels, dict)
            and labels.get("class") == "target_red"
        ]
        raw_visible_target_mask = np.isin(
            instance_ids, visible_target_ids
        )
        amodal_target_mask = np.isin(
            amodal_instance_ids, amodal_target_ids
        )
        visible_target_mask = (
            raw_visible_target_mask & amodal_target_mask
        )
        Image.fromarray(
            raw_visible_target_mask.astype(np.uint8) * 255, "L"
        ).save(pose_dir / "target_visible_mask_raw.png")
        Image.fromarray(
            visible_target_mask.astype(np.uint8) * 255, "L"
        ).save(pose_dir / "target_visible_mask.png")
        Image.fromarray(
            amodal_target_mask.astype(np.uint8) * 255, "L"
        ).save(pose_dir / "target_amodal_mask.png")
        Image.fromarray(
            np.asarray(amodal_color_pass, dtype=np.uint8), "RGB"
        ).save(pose_dir / "target_amodal_color_pass.png")
        objective_occlusion = objective_occlusion_measurement(
            raw_visible_target_mask,
            amodal_target_mask,
        )
        objective_occlusion_file = "objective_occlusion.json"
        (pose_dir / objective_occlusion_file).write_text(
            json.dumps(_json_safe(objective_occlusion), indent=2) + "\n",
            encoding="utf-8",
        )
        if target_reference_removed_override is not None:
            (
                reference_removed_instance_ids,
                reference_removed_id_to_labels,
                reference_removed_color_pass,
            ) = target_reference_removed_override
            reference_removed_target_ids = [
                int(instance_id)
                for instance_id, labels in (
                    reference_removed_id_to_labels.items()
                )
                if isinstance(labels, dict)
                and labels.get("class") == "target_red"
            ]
            raw_reference_removed_target_mask = np.isin(
                reference_removed_instance_ids,
                reference_removed_target_ids,
            )
            reference_removed_target_mask = (
                raw_reference_removed_target_mask & amodal_target_mask
            )
            Image.fromarray(
                raw_reference_removed_target_mask.astype(np.uint8) * 255,
                "L",
            ).save(pose_dir / "target_reference_removed_mask_raw.png")
            Image.fromarray(
                reference_removed_target_mask.astype(np.uint8) * 255,
                "L",
            ).save(pose_dir / "target_reference_removed_mask.png")
            Image.fromarray(
                np.asarray(
                    reference_removed_color_pass, dtype=np.uint8
                ),
                "RGB",
            ).save(
                pose_dir / "target_reference_removed_color_pass.png"
            )
            objective_reference_occlusion = (
                objective_reference_occlusion_measurement(
                    raw_visible_target_mask,
                    raw_reference_removed_target_mask,
                    amodal_target_mask,
                )
            )
            objective_reference_occlusion_file = (
                "objective_reference_occlusion.json"
            )
            (
                pose_dir / objective_reference_occlusion_file
            ).write_text(
                json.dumps(
                    _json_safe(objective_reference_occlusion), indent=2
                )
                + "\n",
                encoding="utf-8",
            )
            if (
                objective_behind_geometry is not None
                and camera_calibration_data is not None
            ):
                reference_ids = [
                    int(instance_id)
                    for instance_id, labels in id_to_labels.items()
                    if isinstance(labels, dict)
                    and labels.get("class") == "container"
                ]
                reference_visible_mask = np.isin(
                    instance_ids, reference_ids
                )
                objective_behind = (
                    objective_camera_relative_behind_measurement(
                        target_center_world_m=(
                            objective_behind_geometry[
                                "target_center_world_m"
                            ]
                        ),
                        reference_bounds_world_m=(
                            objective_behind_geometry[
                                "reference_bounds_world_m"
                            ]
                        ),
                        camera_to_world_row_vector_matrix=(
                            camera_calibration_data[
                                "camera_to_world_row_vector_matrix"
                            ]
                        ),
                        target_amodal_mask=amodal_target_mask,
                        reference_visible_mask=reference_visible_mask,
                        membership=objective_behind_geometry["membership"],
                        reference_occlusion_fraction=(
                            objective_reference_occlusion[
                                "reference_occlusion_fraction"
                            ]
                        ),
                    )
                )
                objective_camera_relative_behind_file = (
                    "objective_camera_relative_behind.json"
                )
                (
                    pose_dir / objective_camera_relative_behind_file
                ).write_text(
                    json.dumps(_json_safe(objective_behind), indent=2)
                    + "\n",
                    encoding="utf-8",
                )
    if camera_calibration_data is not None:
        (pose_dir / "camera_calibration.json").write_text(
            json.dumps(
                _json_safe(camera_calibration_data), indent=2
            )
            + "\n",
            encoding="utf-8",
        )
    previous_metadata_path = pose_dir / "metadata.json"
    if camera_provenance is None and previous_metadata_path.is_file():
        previous_metadata = json.loads(
            previous_metadata_path.read_text(encoding="utf-8")
        )
        camera_provenance = previous_metadata.get("camera_provenance")
    metadata = {
        "pose": pose_name,
        "rgb_shape": list(rgba.shape),
        "depth_shape": list(depth.shape),
        "valid_depth_pixels": int(finite.sum()),
        "depth_min_m": float(depth[finite].min()) if finite.any() else None,
        "depth_max_m": float(depth[finite].max()) if finite.any() else None,
        "semantic_classes": sorted(
            {
                str(labels["class"])
                for labels in id_to_labels.values()
                if isinstance(labels, dict) and "class" in labels
            }
        ),
        "object_statistics_file": "objects.json",
        "segmentation_source": segmentation_source,
        "segmentation_limitation": segmentation_limitation,
        "camera_provenance": camera_provenance,
        "camera_calibration_file": (
            "camera_calibration.json"
            if camera_calibration_data is not None
            else None
        ),
        "overview_rgb_file": (
            "overview_rgb.png" if overview_shape is not None else None
        ),
        "overview_rgb_shape": overview_shape,
        "objective_occlusion_file": objective_occlusion_file,
        "objective_reference_occlusion_file": (
            objective_reference_occlusion_file
        ),
        "objective_camera_relative_behind_file": (
            objective_camera_relative_behind_file
        ),
    }
    (pose_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
