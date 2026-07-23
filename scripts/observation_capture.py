"""UR10e provisional observation poses and wrist-camera RGB/depth capture."""

import json
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


def add_scene_labels(stage) -> None:
    import omni.replicator.core as rep

    for prim_path, label in SEMANTIC_OBJECTS.items():
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


def _object_statistics(instance_ids: np.ndarray, id_to_labels: dict, depth: np.ndarray) -> dict:
    statistics = {}
    for expected_label in SEMANTIC_OBJECTS.values():
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


def save_capture(output_root: Path, pose_name: str, rgb_data, depth_data) -> None:
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
    instance_ids, id_to_labels = _instance_ids_from_scene_colors(rgba[:, :, :3], depth)
    np.save(pose_dir / "instance_ids.npy", instance_ids)
    Image.fromarray(_colorize_instance_ids(instance_ids), "RGB").save(
        pose_dir / "instance_segmentation.png"
    )
    (pose_dir / "instance_labels.json").write_text(
        json.dumps(_json_safe(id_to_labels), indent=2), encoding="utf-8"
    )
    object_statistics = _object_statistics(instance_ids, id_to_labels, depth)
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
        "segmentation_source": "rgb_color_key_fallback",
        "segmentation_limitation": "Replace with RTX instance annotator after Isaac Sim synthetic-data crash is resolved.",
    }
    (pose_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
