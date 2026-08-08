"""Validate an RG6-sized bilateral contact grasp and lift on physical GPU 5.

The imported RG6 visual asset is preserved, while its currently unstable
Isaac Sim 6 finger articulation is disabled. Two kinematic collision pads,
matched to the RG6 finger-pad envelope, interact with a dynamic target through
PhysX friction/contact only. No target attachment or pose copying is used.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_BASE = ROOT / "outputs" / "rg6_physics"


def configured_gpu() -> int:
    value = os.environ.get("PHYSICAL_GPU", "5")
    if not value.isdigit():
        raise RuntimeError("PHYSICAL_GPU must be one integer index")
    return int(value)


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--renderer-gpu", type=int, default=5)
parser.add_argument("--physics-gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output-root", type=Path)
parser.add_argument("--mass-kg", type=float, default=0.08)
parser.add_argument("--static-friction", type=float, default=4.0)
parser.add_argument("--dynamic-friction", type=float, default=3.0)
args = parser.parse_args()
physical_gpu = configured_gpu()
if args.renderer_gpu != physical_gpu or args.physics_gpu != 0:
    raise ValueError("renderer must match PHYSICAL_GPU and physics must be cuda:0")
if args.seed < 0:
    raise ValueError("seed must be non-negative")
OUTPUT_ROOT = (
    args.output_root.resolve()
    if args.output_root is not None
    else OUTPUT_BASE / f"contact_grasp_seed{args.seed:03d}"
)

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "active_gpu": physical_gpu,
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
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.experimental.prims import RigidPrim
import isaacsim.core.experimental.utils.app as app_utils

from build_observation_video import build_frame_sequence_video
from observation_capture import create_fixed_overview_camera, load_observation_config
from seeded_benchmark import apply_layout, generate_layout


def set_cube_transform(
    cube: UsdGeom.Cube,
    position: tuple[float, float, float],
    half_extents: tuple[float, float, float],
) -> None:
    xform = UsdGeom.Xformable(cube)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddScaleOp().Set(Gf.Vec3f(*half_extents))


def set_world_matrix(prim, position: tuple[float, float, float]) -> None:
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 180).GetQuat())
    matrix.SetTranslateOnly(Gf.Vec3d(*position))
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.MakeMatrixXform().Set(matrix)


def apply_collision(prim) -> None:
    UsdPhysics.CollisionAPI.Apply(prim)


def apply_kinematic_body(prim) -> None:
    rigid = UsdPhysics.RigidBodyAPI.Apply(prim)
    rigid.CreateKinematicEnabledAttr().Set(True)
    rigid.CreateRigidBodyEnabledAttr().Set(True)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(args.mass_kg)


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

# The benchmark cube uses size=1, so the container bottom top is z=0.764 m.
# Place the 4.5 cm target immediately above it without penetration.
target_scale = 0.045
target_half = target_scale * 0.5
target_start_z = 0.764 + target_half + 0.002
target_prim = stage.GetPrimAtPath("/World/TargetRed")
target_cube = UsdGeom.Cube(target_prim)
set_cube_transform(
    target_cube,
    (target_x, target_y, target_start_z),
    (target_scale, target_scale, target_scale),
)
apply_collision(target_prim)
target_rigid = UsdPhysics.RigidBodyAPI.Apply(target_prim)
target_rigid.CreateRigidBodyEnabledAttr().Set(True)
target_rigid.CreateKinematicEnabledAttr().Set(False)
mass_api = UsdPhysics.MassAPI.Apply(target_prim)
mass_api.CreateMassAttr().Set(args.mass_kg)
PhysxSchema.PhysxRigidBodyAPI.Apply(target_prim).CreateDisableGravityAttr().Set(False)
PhysxSchema.PhysxContactReportAPI.Apply(target_prim).CreateThresholdAttr().Set(0.0)

# Static tabletop/container collision geometry.
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
        apply_collision(prim)

# Disable only the unstable imported RG6 dynamics; keep its exact visual mesh.
rg6_prim = stage.GetPrimAtPath("/World/RobotSystem/RG6")
physics_variants = rg6_prim.GetVariantSets().GetVariantSet("Physics")
if physics_variants.IsValid():
    physics_variants.SetVariantSelection("none")

proxy_root = UsdGeom.Xform.Define(stage, "/World/RG6ContactPhysics")
left_pad = UsdGeom.Cube.Define(stage, "/World/RG6ContactPhysics/LeftPad")
right_pad = UsdGeom.Cube.Define(stage, "/World/RG6ContactPhysics/RightPad")
pad_half_extents = (0.030, 0.012, 0.060)
open_offset = 0.085
closed_offset = 0.020
pad_z = target_start_z + 0.005
for cube, y_offset in ((left_pad, open_offset), (right_pad, -open_offset)):
    set_cube_transform(
        cube,
        (target_x, target_y + y_offset, pad_z),
        pad_half_extents,
    )
    apply_collision(cube.GetPrim())
    apply_kinematic_body(cube.GetPrim())
    cube.CreateDisplayColorAttr().Set([Gf.Vec3f(0.08, 0.09, 0.10)])
    PhysxSchema.PhysxContactReportAPI.Apply(cube.GetPrim()).CreateThresholdAttr().Set(
        0.0
    )

material = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/RG6Grip")
material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
material_api.CreateStaticFrictionAttr().Set(args.static_friction)
material_api.CreateDynamicFrictionAttr().Set(args.dynamic_friction)
material_api.CreateRestitutionAttr().Set(0.0)
for prim in (target_prim, left_pad.GetPrim(), right_pad.GetPrim()):
    add_physics_material_to_prim(stage, prim, material.GetPath())

config = load_observation_config(ROOT)
config["overview_camera"]["resolution"] = [960, 540]
config["overview_camera"]["position_world_m"] = [1.75, -2.15, 1.75]
config["overview_camera"]["look_at_world_m"] = [0.48, 0.02, 0.90]
config["overview_camera"]["focal_length_mm"] = 42.0
overview = create_fixed_overview_camera(stage, config)
render_product = rep.create.render_product(str(overview.GetPath()), (960, 540))
rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annotator.attach(render_product)

visual_base_position = (target_x, target_y, target_start_z + 0.285)
set_world_matrix(rg6_prim, visual_base_position)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
frame_root = OUTPUT_ROOT / "frames"
frame_root.mkdir(parents=True, exist_ok=True)
for old_frame in frame_root.glob("frame_*.png"):
    old_frame.unlink()

contacts = {"left": 0, "right": 0, "events": [], "raw_pairs": []}
target_path = "/World/TargetRed"
left_path = str(left_pad.GetPath())
right_path = str(right_pad.GetPath())


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
        if len(contacts["raw_pairs"]) < 200:
            contacts["raw_pairs"].append(sorted(pair))
        side = None
        if target_path in pair and left_path in pair:
            contacts["left"] += 1
            side = "left"
        elif target_path in pair and right_path in pair:
            contacts["right"] += 1
            side = "right"
        if side and len(contacts["events"]) < 100:
            contacts["events"].append(side)


contact_subscription = (
    get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
)
SimulationManager.setup_simulation(dt=1.0 / 60.0)
app_utils.play()
for _ in range(30):
    simulation_app.update()
pad_bodies = RigidPrim(paths=[left_path, right_path])
pad_physics_view = pad_bodies._physics_rigid_body_view
pad_transform_device = pad_physics_view.get_transforms().device
pad_indices = wp.array([0, 1], dtype=wp.int32, device=pad_transform_device)

from PIL import Image, ImageDraw, ImageFont

try:
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26
    )
except OSError:
    font = ImageFont.load_default()

frames: list[Path] = []


def read_target_position() -> np.ndarray:
    matrix = omni.usd.get_world_transform_matrix(target_prim)
    return np.asarray(matrix.ExtractTranslation(), dtype=np.float64)


def capture(label: str) -> None:
    rep.orchestrator.step(rt_subframes=2)
    rgba = np.asarray(rgb_annotator.get_data()).copy()
    image = Image.fromarray(rgba[:, :, :3].astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    target_z = read_target_position()[2]
    metrics = (
        f"contacts L/R={contacts['left']}/{contacts['right']}  "
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


def move_pads(offset: float, z: float) -> None:
    transforms = pad_physics_view.get_transforms()
    targets_np = transforms.numpy()
    targets_np[:, :3] = np.asarray(
        [
            [target_x, target_y + offset, z],
            [target_x, target_y - offset, z],
        ],
        dtype=np.float32,
    )
    targets_np[:, 3:7] = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    targets = wp.array(
        targets_np,
        dtype=wp.float32,
        device=transforms.device,
    )
    pad_physics_view.set_kinematic_targets(targets, pad_indices)


initial_target_position = read_target_position()
for step in range(45):
    simulation_app.update()
    if step % 3 == 0:
        capture("RG6 open / dynamic target settled")

for step in range(1, 91):
    alpha = step / 90.0
    offset = open_offset + alpha * (closed_offset - open_offset)
    move_pads(offset, pad_z)
    simulation_app.update()
    if step % 3 == 0:
        capture("Closing RG6-sized collision pads")

for step in range(60):
    simulation_app.update()
    if step % 3 == 0:
        capture("Bilateral contact hold")

pad_positions_before_lift, _ = pad_bodies.get_world_poses()
pad_positions_before_lift = pad_positions_before_lift.numpy().tolist()
bilateral_before_lift = contacts["left"] > 0 and contacts["right"] > 0
if bilateral_before_lift:
    for step in range(1, 121):
        lift = 0.18 * step / 120.0
        move_pads(closed_offset, pad_z + lift)
        set_world_matrix(
            rg6_prim,
            (
                visual_base_position[0],
                visual_base_position[1],
                visual_base_position[2] + lift,
            ),
        )
        simulation_app.update()
        if step % 3 == 0:
            capture("Lifting through PhysX contact only")

for step in range(60):
    simulation_app.update()
    if step % 3 == 0:
        capture("Lift verification")

final_target_position = read_target_position()
lift_delta = float(final_target_position[2] - initial_target_position[2])
success = bilateral_before_lift and lift_delta >= 0.10
video = build_frame_sequence_video(
    frames,
    OUTPUT_ROOT / "rg6_contact_grasp.mp4",
    fps=6,
    crf=17,
    preset="slow",
    purpose="rg6_contact_physics_pilot_not_final_evaluation",
)
result = {
    "schema_version": "rg6-contact-grasp-pilot-v1",
    "status": "completed" if success else "failed",
    "seed": args.seed,
    "physics_implementation": (
        "rg6_visual_with_rg6_sized_kinematic_bilateral_collision_pad_proxy"
    ),
    "target_dynamics": {
        "rigid_body": True,
        "kinematic": False,
        "gravity": True,
        "mass_kg": 0.08,
        "explicit_target_attachment_used": False,
        "target_pose_copying_used": False,
    },
    "contacts": {
        "left_event_count": contacts["left"],
        "right_event_count": contacts["right"],
        "bilateral_before_lift": bilateral_before_lift,
        "raw_collider_pairs_sample": contacts["raw_pairs"],
    },
    "measured_pad_positions_before_lift_world_m": pad_positions_before_lift,
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
    "limitation": (
        "The imported RG6 mimic-joint articulation is numerically unstable in "
        "Isaac Sim 6, so this pilot uses RG6-sized bilateral collision pads. "
        "Replace the proxy with a repaired/calibrated RG6 articulation before "
        "final paper evaluation or real-robot transfer."
    ),
}
(OUTPUT_ROOT / "result.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8"
)
print(f"RG6_CONTACT_GRASP_RESULT={OUTPUT_ROOT / 'result.json'}", flush=True)
print(f"RG6_CONTACT_GRASP_VIDEO={OUTPUT_ROOT / 'rg6_contact_grasp.mp4'}", flush=True)
app_utils.stop()
contact_subscription = None
simulation_app.close()
raise SystemExit(0 if success else 2)
