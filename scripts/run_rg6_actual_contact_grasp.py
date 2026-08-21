"""Grasp and lift a dynamic target with the actual articulated RG6 asset.

The RG6 is imported by Isaac Sim 6 with its visual and collision meshes.  A
kinematic mount drives the fixed base while the six RG6 joints and their
collision shapes remain physical.  The target is never attached or pose-copied:
bilateral PhysX contacts and friction must support it during the lift.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_BASE = ROOT / "outputs" / "rg6_physics"
IMPORT_RESULT = (
    ROOT
    / "assets"
    / "robots"
    / "onrobot_rg6"
    / "isaac6_import"
    / "import_result.json"
)


def configured_gpu() -> int:
    value = (
        os.environ.get("PHYSICAL_GPU")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
        or "0"
    )
    if not value.isdigit():
        raise RuntimeError("Configure exactly one integer GPU index")
    return int(value)


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--renderer-gpu", type=int, default=configured_gpu())
parser.add_argument("--physics-gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output-root", type=Path)
parser.add_argument("--no-video", action="store_true")
args = parser.parse_args()
if args.renderer_gpu != configured_gpu() or args.physics_gpu != 0:
    raise ValueError("Renderer must use the configured GPU and physics must use cuda:0")
if args.seed < 0:
    raise ValueError("seed must be non-negative")

OUTPUT_ROOT = (
    args.output_root.resolve()
    if args.output_root is not None
    else OUTPUT_BASE / f"actual_contact_grasp_seed{args.seed:03d}"
)
RG6_ASSET = Path(
    json.loads(IMPORT_RESULT.read_text(encoding="utf-8"))["output_usd"]
).resolve()
if not RG6_ASSET.is_file():
    raise FileNotFoundError(f"Reimported RG6 asset is missing: {RG6_ASSET}")

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "active_gpu": args.renderer_gpu,
        "physics_gpu": 0,
        "multi_gpu": False,
        "max_gpu_count": 1,
        "extra_args": ["--/renderer/multiGpu/autoEnable=false"],
        "renderer": "RaytracedLighting",
        "anti_aliasing": 4,
        "fast_shutdown": True,
    }
)

import omni.replicator.core as rep
import omni.usd
import warp as wp
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
import isaacsim.core.experimental.utils.app as app_utils
from omni.physx import get_physx_simulation_interface
from omni.physx.bindings._physx import ContactEventType
from omni.physx.scripts.physicsUtils import add_physics_material_to_prim
from pxr import (
    Gf,
    PhysicsSchemaTools,
    PhysxSchema,
    Sdf,
    UsdGeom,
    UsdPhysics,
    UsdShade,
)

from build_observation_video import build_frame_sequence_video
from observation_capture import create_fixed_overview_camera, load_observation_config
from seeded_benchmark import apply_layout, generate_layout


def set_cube_transform(
    cube: UsdGeom.Cube,
    position: tuple[float, float, float],
    full_extents: tuple[float, float, float],
) -> None:
    cube.CreateSizeAttr().Set(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddScaleOp().Set(Gf.Vec3f(*full_extents))


def set_xform_pose(
    prim,
    position: tuple[float, float, float],
    rotation_x_degrees: float = 180.0,
) -> None:
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(
        Gf.Rotation(Gf.Vec3d(1, 0, 0), rotation_x_degrees).GetQuat()
    )
    matrix.SetTranslateOnly(Gf.Vec3d(*position))
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.MakeMatrixXform().Set(matrix)


context = omni.usd.get_context()
scene_path = ROOT / "assets" / "scenes" / "open_container_benchmark.usda"
if not context.open_stage(str(scene_path)):
    raise RuntimeError(f"Could not open {scene_path}")
for _ in range(30):
    simulation_app.update()
stage = context.get_stage()

layout = generate_layout(args.seed)
apply_layout(stage, layout)
target_x, target_y, _ = layout["positions_world_m"]["target_red"]

# Hide and disable the legacy RG6 reference; the Isaac-6 reimport below is the
# only gripper participating in rendering and physics.
legacy_rg6 = stage.GetPrimAtPath("/World/RobotSystem/RG6")
legacy_physics = legacy_rg6.GetVariantSets().GetVariantSet("Physics")
if legacy_physics.IsValid():
    legacy_physics.SetVariantSelection("none")
UsdGeom.Imageable(legacy_rg6).MakeInvisible()

target_size = 0.045
target_start_z = 0.764 + target_size * 0.5 + 0.002
target_prim = stage.GetPrimAtPath("/World/TargetRed")
target_cube = UsdGeom.Cube(target_prim)
set_cube_transform(
    target_cube,
    (target_x, target_y, target_start_z),
    (target_size, target_size, target_size),
)
UsdPhysics.CollisionAPI.Apply(target_prim)
target_rigid = UsdPhysics.RigidBodyAPI.Apply(target_prim)
target_rigid.CreateRigidBodyEnabledAttr().Set(True)
target_rigid.CreateKinematicEnabledAttr().Set(False)
UsdPhysics.MassAPI.Apply(target_prim).CreateMassAttr().Set(0.08)
PhysxSchema.PhysxRigidBodyAPI.Apply(target_prim).CreateDisableGravityAttr().Set(
    False
)
PhysxSchema.PhysxContactReportAPI.Apply(target_prim).CreateThresholdAttr().Set(
    0.0
)

for path in (
    "/World/WorkBench/Top",
    "/World/WorkMat",
    "/World/OpenContainer/Bottom",
    "/World/OpenContainer/WallFront",
    "/World/OpenContainer/WallBack",
    "/World/OpenContainer/WallLeft",
    "/World/OpenContainer/WallRight",
):
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        UsdPhysics.CollisionAPI.Apply(prim)

# The imported RG6 is fixed to this kinematic body instead of to the world.
# Moving the mount therefore moves the physical articulation base through a
# PhysX fixed joint, while the finger links remain articulated rigid bodies.
# At this height the lower finger-pad faces overlap the top centimetre of the
# 4.5 cm cube without the knuckles/collision hulls scraping the container base.
base_position = (target_x, target_y, target_start_z + 0.285)
mount = UsdGeom.Cube.Define(stage, "/World/RG6ActualMount")
mount.CreateSizeAttr().Set(0.01)
set_xform_pose(mount.GetPrim(), base_position)
mount.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
mount_rigid = UsdPhysics.RigidBodyAPI.Apply(mount.GetPrim())
mount_rigid.CreateRigidBodyEnabledAttr().Set(True)
mount_rigid.CreateKinematicEnabledAttr().Set(True)
UsdPhysics.MassAPI.Apply(mount.GetPrim()).CreateMassAttr().Set(1.0)

rg6_prim = stage.DefinePrim("/World/RG6Actual", "Xform")
rg6_prim.GetReferences().AddReference(str(RG6_ASSET))
set_xform_pose(rg6_prim, base_position)
rg6_variant = rg6_prim.GetVariantSets().GetVariantSet("Physics")
if rg6_variant.IsValid():
    rg6_variant.SetVariantSelection("physx")
for _ in range(20):
    simulation_app.update()

rg6_root_joint_path = "/World/RG6Actual/Physics/root_joint"
rg6_root_joint = UsdPhysics.FixedJoint(
    stage.GetPrimAtPath(rg6_root_joint_path)
)
if not rg6_root_joint:
    raise RuntimeError(f"RG6 root joint is missing: {rg6_root_joint_path}")
rg6_root_joint.CreateBody0Rel().SetTargets([Sdf.Path("/World/RG6ActualMount")])
rg6_root_joint.CreateBody1Rel().SetTargets(
    [
        Sdf.Path(
            "/World/RG6Actual/Geometry/onrobot_rg6_base_link"
        )
    ]
)
# The external mount constraint is not one of the gripper's internal joints.
# Excluding it keeps the kinematic mount outside the RG6 articulation, which is
# required by PhysX while still constraining the articulation base to the mount.
rg6_root_joint.CreateExcludeFromArticulationAttr().Set(True)

left_link_path = (
    "/World/RG6Actual/Geometry/onrobot_rg6_base_link/"
    "left_outer_knuckle/left_inner_finger"
)
right_link_path = (
    "/World/RG6Actual/Geometry/onrobot_rg6_base_link/"
    "right_outer_knuckle/right_inner_finger"
)
left_collision_path = f"{left_link_path}/inner_finger_1"
right_collision_path = f"{right_link_path}/inner_finger_1"
for path in (left_link_path, right_link_path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"RG6 physical finger link is missing: {path}")
    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)

material = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/RG6ActualGrip")
material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
material_api.CreateStaticFrictionAttr().Set(4.0)
material_api.CreateDynamicFrictionAttr().Set(3.0)
material_api.CreateRestitutionAttr().Set(0.0)
for path in (
    str(target_prim.GetPath()),
    left_collision_path,
    right_collision_path,
):
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        add_physics_material_to_prim(stage, prim, material.GetPath())

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
frame_root = OUTPUT_ROOT / "frames"
frame_root.mkdir(parents=True, exist_ok=True)
for old_frame in frame_root.glob("frame_*.png"):
    old_frame.unlink()

config = load_observation_config(ROOT)
config["overview_camera"]["resolution"] = [960, 540]
config["overview_camera"]["position_world_m"] = [1.18, -1.18, 1.62]
config["overview_camera"]["look_at_world_m"] = [target_x, target_y, 0.88]
config["overview_camera"]["focal_length_mm"] = 58.0
overview = create_fixed_overview_camera(stage, config)
render_product = rep.create.render_product(str(overview.GetPath()), (960, 540))
rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annotator.attach(render_product)

contacts = {
    "left": 0,
    "right": 0,
    "events": [],
    "raw_pairs": [],
}
target_path = str(target_prim.GetPath())


def on_contact(headers, _data) -> None:
    for header in headers:
        if header.type not in (
            ContactEventType.CONTACT_FOUND,
            ContactEventType.CONTACT_PERSIST,
        ):
            continue
        pair = {
            str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
            str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
        }
        if target_path not in pair:
            continue
        sorted_pair = sorted(pair)
        if (
            len(contacts["raw_pairs"]) < 20
            and sorted_pair not in contacts["raw_pairs"]
        ):
            contacts["raw_pairs"].append(sorted_pair)
        gripper_path = next((path for path in pair if path != target_path), "")
        side = None
        if "/left_" in gripper_path:
            contacts["left"] += 1
            side = "left"
        elif "/right_" in gripper_path:
            contacts["right"] += 1
            side = "right"
        if side and len(contacts["events"]) < 150:
            contacts["events"].append(side)


contact_subscription = (
    get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
)
SimulationManager.setup_simulation(dt=1.0 / 120.0)
gripper = Articulation("/World/RG6Actual")
app_utils.play()
for _ in range(5):
    simulation_app.update()

dof_names = list(gripper.dof_names)
expected_dofs = {
    "finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_inner_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
}
if set(dof_names) != expected_dofs:
    raise RuntimeError(f"Unexpected RG6 DOFs: {dof_names}")


def targets_for(master: float) -> list[float]:
    target_by_name = {
        "finger_joint": master,
        "left_inner_knuckle_joint": -master,
        "right_outer_knuckle_joint": -master,
        "right_inner_knuckle_joint": -master,
        "left_inner_finger_joint": master,
        "right_inner_finger_joint": master,
    }
    return [target_by_name[name] for name in dof_names]


open_master = -0.45
close_master = 0.45
gripper.set_dof_positions(targets_for(open_master))
gripper.set_dof_position_targets(targets_for(open_master))

mount_body = RigidPrim(paths=["/World/RG6ActualMount"])
mount_physics_view = mount_body._physics_rigid_body_view
mount_device = mount_physics_view.get_transforms().device
mount_indices = wp.array([0], dtype=wp.int32, device=mount_device)

from PIL import Image, ImageDraw, ImageFont

try:
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 25
    )
except OSError:
    font = ImageFont.load_default()

frames: list[Path] = []


def world_position(prim) -> np.ndarray:
    return np.asarray(
        omni.usd.get_world_transform_matrix(prim).ExtractTranslation(),
        dtype=np.float64,
    )


def dof_positions() -> dict[str, float]:
    values = gripper.get_dof_positions().numpy()
    values = values[0] if values.ndim > 1 else values
    return {name: float(value) for name, value in zip(dof_names, values)}


def capture(label: str) -> None:
    if args.no_video:
        return
    rep.orchestrator.step(rt_subframes=2)
    rgba = np.asarray(rgb_annotator.get_data()).copy()
    image = Image.fromarray(rgba[:, :, :3].astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    target_z = world_position(target_prim)[2]
    master = dof_positions()["finger_joint"]
    metrics = (
        f"joint={master:+.3f} rad  "
        f"contact L/R={contacts['left']}/{contacts['right']}  "
        f"target z={target_z:.3f} m"
    )
    label_bounds = draw.textbbox((0, 0), label, font=font)
    metrics_bounds = draw.textbbox((0, 0), metrics, font=font)
    box_width = max(label_bounds[2], metrics_bounds[2]) + 50
    draw.rounded_rectangle(
        (15, 15, min(945, box_width), 102),
        radius=7,
        fill=(0, 0, 0),
    )
    draw.text((25, 22), label, fill=(255, 255, 255), font=font)
    draw.text((25, 60), metrics, fill=(201, 224, 238), font=font)
    path = frame_root / f"frame_{len(frames):04d}.png"
    image.save(path)
    frames.append(path)


def move_mount(z: float) -> None:
    targets = mount_physics_view.get_transforms().numpy()
    targets[0, :3] = np.asarray(
        [base_position[0], base_position[1], z],
        dtype=np.float32,
    )
    targets[0, 3:7] = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    mount_physics_view.set_kinematic_targets(
        wp.array(targets, dtype=wp.float32, device=mount_device),
        mount_indices,
    )


def mount_physics_position() -> np.ndarray:
    positions, _ = mount_body.get_world_poses()
    values = positions.numpy()
    return np.asarray(values[0] if values.ndim > 1 else values, dtype=np.float64)


for step in range(90):
    gripper.set_dof_position_targets(targets_for(open_master))
    simulation_app.update()
    if step % 3 == 0:
        capture("Actual RG6 open / dynamic target settled")

initial_target_position = world_position(target_prim)
for step in range(1, 151):
    alpha = step / 150.0
    master_target = open_master + alpha * (close_master - open_master)
    gripper.set_dof_position_targets(targets_for(master_target))
    simulation_app.update()
    if step % 3 == 0:
        capture("Closing actual articulated RG6 fingers")

for step in range(90):
    gripper.set_dof_position_targets(targets_for(close_master))
    simulation_app.update()
    if step % 3 == 0:
        capture("Holding with bilateral PhysX contact")

bilateral_before_lift = contacts["left"] > 0 and contacts["right"] > 0
mount_position_before_lift = mount_physics_position()
finger_positions_before_lift = {
    "left": world_position(stage.GetPrimAtPath(left_link_path)).tolist(),
    "right": world_position(stage.GetPrimAtPath(right_link_path)).tolist(),
}
if bilateral_before_lift:
    for step in range(1, 181):
        lift = 0.18 * step / 180.0
        gripper.set_dof_position_targets(targets_for(close_master))
        move_mount(base_position[2] + lift)
        simulation_app.update()
        if step % 3 == 0:
            capture("Lifting via actual RG6 collision and friction")

for step in range(90):
    gripper.set_dof_position_targets(targets_for(close_master))
    simulation_app.update()
    if step % 3 == 0:
        capture("Actual RG6 lift verification")

final_target_position = world_position(target_prim)
final_mount_position = mount_physics_position()
lift_delta = float(final_target_position[2] - initial_target_position[2])
joint_state = dof_positions()
finite_joints = all(math.isfinite(value) for value in joint_state.values())
success = bilateral_before_lift and finite_joints and lift_delta >= 0.10

video = None
if frames:
    video = build_frame_sequence_video(
        frames,
        OUTPUT_ROOT / "rg6_actual_contact_grasp.mp4",
        fps=8,
        crf=17,
        preset="slow",
        purpose="actual_rg6_contact_physics_pilot_not_final_evaluation",
    )

result = {
    "schema_version": "rg6-actual-contact-grasp-pilot-v1",
    "status": "completed" if success else "failed",
    "seed": args.seed,
    "physics_implementation": (
        "isaac_sim_6_reimported_actual_rg6_articulation_and_collision_meshes"
    ),
    "rg6_asset": str(RG6_ASSET),
    "rg6_dof_names": dof_names,
    "final_joint_positions_rad": joint_state,
    "finite_joint_state": finite_joints,
    "target_dynamics": {
        "rigid_body": True,
        "kinematic": False,
        "gravity": True,
        "mass_kg": 0.08,
        "explicit_target_attachment_used": False,
        "target_pose_copying_used": False,
    },
    "mount_motion": {
        "kinematic": True,
        "fixed_joint_to_actual_rg6_base": True,
        "position_before_lift_world_m": mount_position_before_lift.tolist(),
        "final_position_world_m": final_mount_position.tolist(),
    },
    "contacts": {
        "left_event_count": contacts["left"],
        "right_event_count": contacts["right"],
        "bilateral_before_lift": bilateral_before_lift,
        "raw_collider_pairs_sample": contacts["raw_pairs"],
    },
    "finger_link_positions_before_lift_world_m": finger_positions_before_lift,
    "initial_target_position_world_m": initial_target_position.tolist(),
    "final_target_position_world_m": final_target_position.tolist(),
    "verified_lift_delta_m": lift_delta,
    "success_threshold_m": 0.10,
    "lift_verified": success,
    "video": video,
    "gpu_policy": {
        "physical_gpu": 5,
        "physics_cuda_device": 0,
        "multi_gpu": False,
    },
    "valid_for_final_evaluation": False,
    "evaluation_scope": (
        "Single deterministic pilot for contact-pipeline validation only."
    ),
}
(OUTPUT_ROOT / "result.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8"
)
print(f"RG6_ACTUAL_GRASP_RESULT={OUTPUT_ROOT / 'result.json'}", flush=True)
if video is not None:
    print(
        f"RG6_ACTUAL_GRASP_VIDEO="
        f"{OUTPUT_ROOT / 'rg6_actual_contact_grasp.mp4'}",
        flush=True,
    )
app_utils.stop()
contact_subscription = None
simulation_app.close()
raise SystemExit(0 if success else 2)
