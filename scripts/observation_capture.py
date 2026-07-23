"""UR10e provisional observation poses and wrist-camera RGB/depth capture."""

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


def add_scene_labels(stage, semantic_objects=None) -> None:
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
    if np.any(target < lower) or np.any(target > upper):
        raise ValueError(f"Target pose {pose_name} violates UR10e joint limits")

    maximum_delta = float(np.max(np.abs(target - start)))
    steps = max(1, math.ceil(maximum_delta / settings["maximum_joint_step_rad"]))
    maximum_contact_force = 0.0
    collision_detected = False
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
            raise ValueError(f"Interpolated waypoint {step} violates UR10e joint limits")
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
        waypoint_records.append(
            {
                "step": step,
                "alpha": alpha,
                "maximum_waypoint_error_rad": float(np.max(np.abs(measured - waypoint))),
                "maximum_contact_force_n": maximum_contact_force,
            }
        )
        if collision_detected:
            robot.set_dof_position_targets(measured.tolist(), dof_indices=indices)
            break

    settle_frames = 0
    if not collision_detected:
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
            if float(np.max(np.abs(measured - target))) <= settings["final_tolerance_rad"]:
                break

    measured_array = robot.get_dof_positions().numpy()
    measured_vector = measured_array[0] if measured_array.ndim > 1 else measured_array
    measured = np.asarray(measured_vector[indices], dtype=np.float64)
    maximum_final_error = float(np.max(np.abs(measured - target)))
    return {
        "status": (
            "collision_abort"
            if collision_detected
            else "completed"
            if maximum_final_error <= settings["final_tolerance_rad"]
            else "final_tolerance_failed"
        ),
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
        "maximum_contact_force_n": maximum_contact_force,
        "contact_force_abort_threshold_n": settings["contact_force_abort_threshold_n"],
        "requested_joint_positions_rad": target.tolist(),
        "measured_joint_positions_rad": measured.tolist(),
        "maximum_joint_error_rad": maximum_final_error,
        "verification_tolerance_rad": settings["final_tolerance_rad"],
        "waypoints": waypoint_records,
    }


def align_tool_and_camera(stage, ee_prim, rg6_prim, camera_prim, config: dict, ee_pose=None) -> None:
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


def create_capture_pipeline(camera_path: str, resolution: tuple[int, int]):
    import omni.replicator.core as rep

    render_product = rep.create.render_product(camera_path, resolution)
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    depth = rep.AnnotatorRegistry.get_annotator("distance_to_camera")
    rgb.attach(render_product)
    depth.attach(render_product)
    return rep, render_product, rgb, depth


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
    valid = (brightness > 35) & (chroma > 35) & (minimum_distance < 0.20)
    instance_ids = np.where(valid, closest + 1, 0).astype(np.uint32)
    _split_id_pair_by_horizontal_components(
        instance_ids, pair=(2, 3), left_id=3, right_id=2
    )
    _split_id_pair_by_horizontal_components(
        instance_ids, pair=(6, 7), left_id=7, right_id=6
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
) -> None:
    pose_dir = output_root / pose_name
    pose_dir.mkdir(parents=True, exist_ok=True)
    rgba = np.asarray(rgb_data)
    Image.fromarray(rgba[:, :, :3].astype(np.uint8), "RGB").save(pose_dir / "rgb.png")
    depth = np.asarray(depth_data, dtype=np.float32)
    np.save(pose_dir / "depth_m.npy", depth)
    finite = np.isfinite(depth)
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
    }
    (pose_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
