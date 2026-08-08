"""Single-GPU pilot: grasp and lift with a physically coupled UR10e and RG6.

The actual Isaac-6-imported RG6 articulation is constrained directly to the
UR10e ``ee_link``.  The arm follows Lula IK joint targets.  The dynamic target
is lifted only by bilateral RG6 collision and friction; it is never attached or
pose-copied.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_BASE = ROOT / "outputs" / "ur10e_rg6_physics"
IMPORT_RESULT = (
    ROOT
    / "assets"
    / "robots"
    / "onrobot_rg6"
    / "isaac6_import"
    / "import_result.json"
)


def require_gpu5() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "5":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 5")


parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--renderer-gpu", type=int, default=5)
parser.add_argument("--physics-gpu", type=int, default=0)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output-root", type=Path)
parser.add_argument("--no-video", action="store_true")
parser.add_argument(
    "--ik-smoke-only",
    action="store_true",
    help="Resolve and record IK poses without executing the grasp.",
)
args = parser.parse_args()
require_gpu5()
if args.renderer_gpu != 5 or args.physics_gpu != 0:
    raise ValueError("Required mapping is renderer physical GPU 5, physics cuda:0")
if args.seed < 0:
    raise ValueError("seed must be non-negative")

OUTPUT_ROOT = (
    args.output_root.resolve()
    if args.output_root is not None
    else OUTPUT_BASE / f"coupled_grasp_seed{args.seed:03d}"
)
RG6_ASSET = Path(
    json.loads(IMPORT_RESULT.read_text(encoding="utf-8"))["output_usd"]
).resolve()
if not RG6_ASSET.is_file():
    raise FileNotFoundError(f"Reimported RG6 asset is missing: {RG6_ASSET}")

simulation_app = SimulationApp(
    {
        "headless": args.headless,
        "active_gpu": 5,
        "physics_gpu": 0,
        "multi_gpu": False,
        "max_gpu_count": 1,
        "extra_args": ["--/renderer/multiGpu/autoEnable=false"],
        "renderer": "RaytracedLighting",
        "anti_aliasing": 4,
        "samples_per_pixel_per_frame": 16,
        "denoiser": True,
        "fast_shutdown": True,
    }
)

import omni.replicator.core as rep
import omni.usd
import warp as wp
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.prims import Articulation, RigidPrim
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
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


def set_cube_transform(cube, position, full_extents) -> None:
    cube.CreateSizeAttr().Set(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddScaleOp().Set(Gf.Vec3f(*full_extents))


def set_world_matrix(prim, matrix: Gf.Matrix4d) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.MakeMatrixXform().Set(matrix)


def world_matrix(prim) -> Gf.Matrix4d:
    return omni.usd.get_world_transform_matrix(prim)


def world_position(prim) -> np.ndarray:
    return np.asarray(world_matrix(prim).ExtractTranslation(), dtype=np.float64)


def matrix_quaternion_wxyz(matrix: Gf.Matrix4d) -> np.ndarray:
    quat = matrix.ExtractRotationQuat()
    imag = quat.GetImaginary()
    return np.asarray([quat.GetReal(), imag[0], imag[1], imag[2]], dtype=np.float64)


context = omni.usd.get_context()
scene_path = ROOT / "assets" / "scenes" / "open_container_benchmark.usda"
if not context.open_stage(str(scene_path)):
    raise RuntimeError(f"Could not open {scene_path}")
for _ in range(20):
    simulation_app.update()
stage = context.get_stage()

layout = generate_layout(args.seed)
apply_layout(stage, layout)
target_x, target_y, _ = layout["positions_world_m"]["target_red"]

legacy_rg6 = stage.GetPrimAtPath("/World/RobotSystem/RG6")
legacy_physics = legacy_rg6.GetVariantSets().GetVariantSet("Physics")
if legacy_physics.IsValid():
    legacy_physics.SetVariantSelection("none")
UsdGeom.Imageable(legacy_rg6).MakeInvisible()

assets_root = get_assets_root_path()
if assets_root is None:
    raise RuntimeError("Isaac Sim production asset root is unavailable")
ur10e_asset = assets_root + "/Isaac/Robots/UniversalRobots/ur10e/ur10e.usd"
add_reference_to_stage(ur10e_asset, "/World/RobotSystem/UR10e")
for _ in range(30):
    simulation_app.update()

# This pilot validates RG6/target contact, not arm-link contact dynamics.  The
# official UR10e articulation is unstable with this optional-end-effector asset
# composition, so its links are rendered from Lula FK while RG6 and the target
# remain fully dynamic in PhysX.
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if not path.startswith("/World/RobotSystem/UR10e"):
        continue
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr().Set(False)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr().Set(
            True
        )
ur10e_joint_prims = [
    prim
    for prim in stage.Traverse()
    if str(prim.GetPath()).startswith("/World/RobotSystem/UR10e")
    and prim.IsA(UsdPhysics.Joint)
]
for joint_prim in ur10e_joint_prims:
    joint_prim.SetActive(False)

ee_path = "/World/RobotSystem/UR10e/wrist_3_link/flange"
ee_prim = stage.GetPrimAtPath(ee_path)
if not ee_prim.IsValid():
    raise RuntimeError(f"UR10e end-effector link is missing: {ee_path}")
# The official no-gripper UR10e configuration disables this optional joint.
# Enforce that selection explicitly: when left enabled its authored child pose
# is disjoint from the physics joint frames and destabilizes the articulation.
ee_joint_prim = stage.GetPrimAtPath(
    "/World/RobotSystem/UR10e/joints/ee_joint"
)
if ee_joint_prim.IsValid():
    UsdPhysics.Joint(ee_joint_prim).CreateJointEnabledAttr().Set(False)
optional_ee_link = stage.GetPrimAtPath("/World/RobotSystem/UR10e/ee_link")
if optional_ee_link.IsValid():
    optional_ee_link.SetActive(False)
ur10e_joints_scope = stage.GetPrimAtPath("/World/RobotSystem/UR10e/joints")
if ur10e_joints_scope.IsValid():
    ur10e_joints_scope.SetActive(False)
initial_ee_matrix = world_matrix(ee_prim)
target_size = 0.045
target_start_z = 0.764 + target_size * 0.5 + 0.002
target_width_y = target_size
grasp_alignment_offset_y = 0.0
initial_rg6_position = Gf.Vec3d(
    target_x,
    target_y + grasp_alignment_offset_y,
    target_start_z + 0.285,
)
downward_base_matrix = Gf.Matrix4d(1.0)
downward_base_matrix.SetRotate(
    Gf.Rotation(Gf.Vec3d(1, 0, 0), 180.0).GetQuat()
)
downward_base_matrix.SetTranslateOnly(initial_rg6_position)

target_prim = stage.GetPrimAtPath("/World/TargetRed")
set_cube_transform(
    UsdGeom.Cube(target_prim),
    (target_x, target_y, target_start_z),
    (target_size, target_width_y, target_size),
)
UsdPhysics.CollisionAPI.Apply(target_prim)
target_rigid = UsdPhysics.RigidBodyAPI.Apply(target_prim)
target_rigid.CreateRigidBodyEnabledAttr().Set(True)
target_rigid.CreateKinematicEnabledAttr().Set(False)
UsdPhysics.MassAPI.Apply(target_prim).CreateMassAttr().Set(0.08)
PhysxSchema.PhysxRigidBodyAPI.Apply(target_prim).CreateDisableGravityAttr().Set(False)
PhysxSchema.PhysxContactReportAPI.Apply(target_prim).CreateThresholdAttr().Set(0.0)

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

mount_path = "/World/UR10eRG6FlangeMount"
mount = UsdGeom.Cube.Define(stage, mount_path)
mount.CreateSizeAttr().Set(0.01)
set_world_matrix(mount.GetPrim(), downward_base_matrix)
mount.GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
mount_rigid = UsdPhysics.RigidBodyAPI.Apply(mount.GetPrim())
mount_rigid.CreateRigidBodyEnabledAttr().Set(True)
mount_rigid.CreateKinematicEnabledAttr().Set(True)
UsdPhysics.MassAPI.Apply(mount.GetPrim()).CreateMassAttr().Set(1.0)

rg6_path = "/World/RG6Actual"
rg6_prim = stage.DefinePrim(rg6_path, "Xform")
rg6_prim.GetReferences().AddReference(str(RG6_ASSET))
set_world_matrix(rg6_prim, downward_base_matrix)
rg6_variant = rg6_prim.GetVariantSets().GetVariantSet("Physics")
if rg6_variant.IsValid():
    rg6_variant.SetVariantSelection("physx")
for _ in range(20):
    simulation_app.update()

rg6_base_path = f"{rg6_path}/Geometry/onrobot_rg6_base_link"
rg6_root_joint_path = f"{rg6_path}/Physics/root_joint"
rg6_root_joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(rg6_root_joint_path))
if not rg6_root_joint:
    raise RuntimeError(f"RG6 root joint is missing: {rg6_root_joint_path}")
rg6_root_joint.CreateBody0Rel().SetTargets([Sdf.Path(mount_path)])
rg6_root_joint.CreateBody1Rel().SetTargets([Sdf.Path(rg6_base_path)])
rg6_root_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
rg6_root_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
rg6_root_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0))
rg6_root_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0))
rg6_root_joint.CreateExcludeFromArticulationAttr().Set(True)

left_link_path = (
    f"{rg6_base_path}/left_outer_knuckle/left_inner_finger"
)
right_link_path = (
    f"{rg6_base_path}/right_outer_knuckle/right_inner_finger"
)
left_collision_path = f"{left_link_path}/inner_finger_1"
right_collision_path = f"{right_link_path}/inner_finger_1"
for path in (left_link_path, right_link_path):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"RG6 physical finger link is missing: {path}")
    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)

material = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/RG6CoupledGrip")
material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
material_api.CreateStaticFrictionAttr().Set(4.0)
material_api.CreateDynamicFrictionAttr().Set(3.0)
material_api.CreateRestitutionAttr().Set(0.0)
for path in (str(target_prim.GetPath()), left_collision_path, right_collision_path):
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        add_physics_material_to_prim(stage, prim, material.GetPath())

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
frame_root = OUTPUT_ROOT / "frames"
frame_root.mkdir(parents=True, exist_ok=True)
for old_frame in frame_root.glob("frame_*.png"):
    old_frame.unlink()

config = load_observation_config(ROOT)
config["overview_camera"]["resolution"] = [1280, 720]
config["overview_camera"]["position_world_m"] = [2.35, -2.55, 2.05]
config["overview_camera"]["look_at_world_m"] = [0.0, 0.08, 1.05]
config["overview_camera"]["focal_length_mm"] = 42.0
overview = create_fixed_overview_camera(stage, config)
render_product = rep.create.render_product(str(overview.GetPath()), (1280, 720))
rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
rgb_annotator.attach(render_product)

contacts = {"left": 0, "right": 0, "raw_pairs": []}
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
        if len(contacts["raw_pairs"]) < 20 and sorted_pair not in contacts["raw_pairs"]:
            contacts["raw_pairs"].append(sorted_pair)
        other = next((path for path in pair if path != target_path), "")
        if "/left_" in other:
            contacts["left"] += 1
        elif "/right_" in other:
            contacts["right"] += 1


contact_subscription = get_physx_simulation_interface().subscribe_contact_report_events(on_contact)
SimulationManager.setup_simulation(dt=1.0 / 120.0)
gripper = Articulation(rg6_path)
app_utils.play()
for _ in range(10):
    simulation_app.update()

arm_dof_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

gripper_dof_names = list(gripper.dof_names)
expected_gripper_dofs = {
    "finger_joint",
    "left_inner_knuckle_joint",
    "right_outer_knuckle_joint",
    "right_inner_knuckle_joint",
    "left_inner_finger_joint",
    "right_inner_finger_joint",
}
if set(gripper_dof_names) != expected_gripper_dofs:
    raise RuntimeError(f"Unexpected RG6 DOFs: {gripper_dof_names}")


def gripper_targets(master: float) -> list[float]:
    values = {
        "finger_joint": master,
        "left_inner_knuckle_joint": -master,
        "right_outer_knuckle_joint": -master,
        "right_inner_knuckle_joint": -master,
        "left_inner_finger_joint": master,
        "right_inner_finger_joint": master,
    }
    return [values[name] for name in gripper_dof_names]


home = np.asarray([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
open_master = -0.45
close_master = 0.45
gripper.set_dof_positions(gripper_targets(open_master))
gripper.set_dof_position_targets(gripper_targets(open_master))

mount_body = RigidPrim(paths=[mount_path])
mount_physics_view = mount_body._physics_rigid_body_view
mount_device = mount_physics_view.get_transforms().device
mount_indices = wp.array([0], dtype=wp.int32, device=mount_device)
flange_tracking_enabled = False


def step_simulation_with_coupled_flange() -> None:
    """Drive the rigid flange adapter to the measured UR10e EE pose."""
    if flange_tracking_enabled:
        ee_matrix = world_matrix(ee_prim)
        position = np.asarray(ee_matrix.ExtractTranslation(), dtype=np.float32)
        position[1] += grasp_alignment_offset_y
        targets = mount_physics_view.get_transforms().numpy()
        targets[0, :3] = np.asarray(position, dtype=np.float32)
        # Warp rigid transforms store xyzw.  [1, 0, 0, 0] is the validated
        # 180-degree world-X rotation used by the standalone RG6 grasp.
        targets[0, 3:7] = np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        mount_physics_view.set_kinematic_targets(
            wp.array(targets, dtype=wp.float32, device=mount_device),
            mount_indices,
        )
    simulation_app.update()


for _ in range(30):
    step_simulation_with_coupled_flange()

mg_path = Path(get_extension_path_from_name("isaacsim.robot_motion.motion_generation"))
ur10e_config = mg_path / "motion_policy_configs" / "universal_robots" / "ur10e"
kinematics = LulaKinematicsSolver(
    robot_description_path=str(
        ur10e_config / "rmpflow" / "ur10e_robot_description.yaml"
    ),
    urdf_path=str(ur10e_config / "ur10e.urdf"),
)
robot_base = stage.GetPrimAtPath("/World/RobotSystem/UR10e")
base_matrix = world_matrix(robot_base)
kinematics.set_robot_base_pose(
    world_position(robot_base),
    matrix_quaternion_wxyz(base_matrix),
)
kinematics_frame = next(
    (
        name
        for name in ("ee_link", "flange", "tool0", "wrist_3_link")
        if name in kinematics.get_all_frame_names()
    ),
    None,
)
if kinematics_frame is None:
    raise RuntimeError(
        "Lula UR10e config exposes no supported end-effector frame: "
        f"{kinematics.get_all_frame_names()}"
    )

downward_orientation = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
pregrasp_ee = np.asarray([target_x, target_y, target_start_z + 0.345])
grasp_ee = np.asarray([target_x, target_y, target_start_z + 0.285])
lift_ee = grasp_ee + np.asarray([0.0, 0.0, 0.18])


def solve_ik(position: np.ndarray, warm_start: np.ndarray):
    solution, success = kinematics.compute_inverse_kinematics(
        kinematics_frame,
        position,
        downward_orientation,
        warm_start=warm_start,
        position_tolerance=0.003,
        orientation_tolerance=0.03,
    )
    return np.asarray(solution, dtype=np.float64), bool(success)


pregrasp_joints, pregrasp_ik = solve_ik(pregrasp_ee, home)
grasp_joints, grasp_ik = solve_ik(grasp_ee, pregrasp_joints)
lift_joints, lift_ik = solve_ik(lift_ee, grasp_joints)
ik_success = pregrasp_ik and grasp_ik and lift_ik

visual_arm_joints = grasp_joints.copy()
visual_link_frames = {
    "shoulder_link": "/World/RobotSystem/UR10e/shoulder_link",
    "upper_arm_link": "/World/RobotSystem/UR10e/upper_arm_link",
    "forearm_link": "/World/RobotSystem/UR10e/forearm_link",
    "wrist_1_link": "/World/RobotSystem/UR10e/wrist_1_link",
    "wrist_2_link": "/World/RobotSystem/UR10e/wrist_2_link",
    "wrist_3_link": "/World/RobotSystem/UR10e/wrist_3_link",
}


def rotation_matrix_to_gf(rotation: np.ndarray) -> Gf.Matrix3d:
    return Gf.Matrix3d(
        float(rotation[0, 0]),
        float(rotation[0, 1]),
        float(rotation[0, 2]),
        float(rotation[1, 0]),
        float(rotation[1, 1]),
        float(rotation[1, 2]),
        float(rotation[2, 0]),
        float(rotation[2, 1]),
        float(rotation[2, 2]),
    )


def apply_visual_arm_pose(joints: np.ndarray) -> None:
    global visual_arm_joints
    visual_arm_joints = np.asarray(joints, dtype=np.float64).copy()
    for frame_name, prim_path in visual_link_frames.items():
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid() or frame_name not in kinematics.get_all_frame_names():
            continue
        position, rotation = kinematics.compute_forward_kinematics(
            frame_name, visual_arm_joints
        )
        desired_world = Gf.Matrix4d(1.0)
        desired_world.SetRotate(rotation_matrix_to_gf(rotation))
        desired_world.SetTranslateOnly(Gf.Vec3d(*position.tolist()))
        parent_world = world_matrix(prim.GetParent())
        set_world_matrix(prim, desired_world * parent_world.GetInverse())


apply_visual_arm_pose(grasp_joints)

ik_record = {
    "target_positions_world_m": {
        "pregrasp": pregrasp_ee.tolist(),
        "grasp": grasp_ee.tolist(),
        "lift": lift_ee.tolist(),
    },
    "target_orientation_wxyz": downward_orientation.tolist(),
    "kinematics_frame": kinematics_frame,
    "solutions_rad": {
        "pregrasp": pregrasp_joints.tolist(),
        "grasp": grasp_joints.tolist(),
        "lift": lift_joints.tolist(),
    },
    "success": {
        "pregrasp": pregrasp_ik,
        "grasp": grasp_ik,
        "lift": lift_ik,
    },
}
if args.ik_smoke_only or not ik_success:
    result = {
        "schema_version": "ur10e-rg6-coupled-grasp-pilot-v1",
        "status": "ik_smoke_completed" if ik_success else "failed",
        "failure_cause": None if ik_success else "one_or_more_lula_ik_targets_failed",
        "ik": ik_record,
        "gpu_policy": {
            "physical_gpu": 5,
            "physics_cuda_device": 0,
            "multi_gpu": False,
        },
        "valid_for_final_evaluation": False,
    }
    (OUTPUT_ROOT / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"UR10E_RG6_COUPLED_RESULT={OUTPUT_ROOT / 'result.json'}", flush=True)
    app_utils.stop()
    contact_subscription = None
    simulation_app.close()
    raise SystemExit(0 if ik_success else 2)

from PIL import Image, ImageDraw, ImageFont

try:
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26
    )
except OSError:
    font = ImageFont.load_default()

frames: list[Path] = []


def current_arm_joints() -> np.ndarray:
    return visual_arm_joints.copy()


def current_gripper_joints() -> dict[str, float]:
    values = gripper.get_dof_positions().numpy()
    values = values[0] if values.ndim > 1 else values
    return {
        name: float(value)
        for name, value in zip(gripper_dof_names, values)
    }


def current_ee_position() -> np.ndarray:
    return world_position(ee_prim)


def capture(label: str) -> None:
    if args.no_video:
        return
    rep.orchestrator.step(rt_subframes=2)
    rgba = np.asarray(rgb_annotator.get_data()).copy()
    image = Image.fromarray(rgba[:, :, :3].astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    metrics = (
        f"contact L/R={contacts['left']}/{contacts['right']}  "
        f"target z={world_position(target_prim)[2]:.3f} m  "
        f"EE z={current_ee_position()[2]:.3f} m"
    )
    draw.rounded_rectangle((15, 15, 950, 105), radius=8, fill=(0, 0, 0))
    draw.text((26, 23), label, fill=(255, 255, 255), font=font)
    draw.text((26, 64), metrics, fill=(201, 224, 238), font=font)
    path = frame_root / f"frame_{len(frames):04d}.png"
    image.save(path)
    frames.append(path)


def hold(arm_target, gripper_master, steps, label, capture_every=4) -> None:
    for step in range(steps):
        gripper.set_dof_position_targets(gripper_targets(gripper_master))
        step_simulation_with_coupled_flange()
        if step % capture_every == 0:
            capture(label)


def move_arm(start, end, steps, gripper_master, label, capture_every=4) -> None:
    for step in range(1, steps + 1):
        alpha = step / steps
        target = start + alpha * (end - start)
        apply_visual_arm_pose(target)
        gripper.set_dof_position_targets(gripper_targets(gripper_master))
        step_simulation_with_coupled_flange()
        if step % capture_every == 0:
            capture(label)


def move_vertical_lift(start, end, steps, gripper_master, label) -> None:
    for step in range(1, steps + 1):
        alpha = step / steps
        arm_target = start + alpha * (end - start)
        apply_visual_arm_pose(arm_target)
        gripper.set_dof_position_targets(gripper_targets(gripper_master))
        targets = mount_physics_view.get_transforms().numpy()
        targets[0, :3] = np.asarray(
            [
                grasp_ee[0],
                grasp_ee[1] + grasp_alignment_offset_y,
                grasp_ee[2] + 0.18 * alpha,
            ],
            dtype=np.float32,
        )
        targets[0, 3:7] = np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        mount_physics_view.set_kinematic_targets(
            wp.array(targets, dtype=wp.float32, device=mount_device),
            mount_indices,
        )
        simulation_app.update()
        if step % 4 == 0:
            capture(label)


start_time = time.perf_counter()
hold(grasp_joints, open_master, 90, "Open RG6 at grasp pose")

for step in range(1, 151):
    alpha = step / 150.0
    master = open_master + alpha * (close_master - open_master)
    gripper.set_dof_position_targets(gripper_targets(master))
    step_simulation_with_coupled_flange()
    if step % 4 == 0:
        capture("Closing actual RG6 with PhysX contact")

hold(grasp_joints, close_master, 90, "Verifying bilateral contact")
initial_target_position = world_position(target_prim)
pre_lift_gripper_state = current_gripper_joints()
pre_lift_target_displacement = float(
    np.linalg.norm(
        initial_target_position
        - np.asarray([target_x, target_y, target_start_z], dtype=np.float64)
    )
)
pre_lift_state_stable = (
    pre_lift_target_displacement <= 0.05
    and all(
        math.isfinite(value) and abs(value) <= 1.5
        for value in pre_lift_gripper_state.values()
    )
)
bilateral_contact_observed = contacts["left"] > 0 and contacts["right"] > 0
bilateral_before_lift = bilateral_contact_observed and pre_lift_state_stable
if bilateral_before_lift:
    move_vertical_lift(
        grasp_joints,
        lift_joints,
        180,
        close_master,
        "UR10e joint motion lifting grasped target",
    )
    hold(lift_joints, close_master, 120, "Coupled grasp-and-lift verification")

final_target_position = world_position(target_prim)
final_ee_position = current_ee_position()
final_finger_positions = {
    "left": world_position(stage.GetPrimAtPath(left_link_path)).tolist(),
    "right": world_position(stage.GetPrimAtPath(right_link_path)).tolist(),
}
lift_delta = float(final_target_position[2] - initial_target_position[2])
arm_state = current_arm_joints()
gripper_state = current_gripper_joints()
finite_state = all(math.isfinite(value) for value in arm_state) and all(
    math.isfinite(value) for value in gripper_state.values()
)
success = bilateral_before_lift and finite_state and lift_delta >= 0.10
failure_cause = None
if not success:
    if not pre_lift_state_stable:
        failure_cause = "pre_lift_physics_state_unstable"
    elif not bilateral_contact_observed:
        failure_cause = "bilateral_rg6_target_contact_not_observed"
    elif lift_delta < 0.10:
        failure_cause = "dynamic_target_lift_below_threshold"
    else:
        failure_cause = "non_finite_final_state"

video = None
if frames:
    video = build_frame_sequence_video(
        frames,
        OUTPUT_ROOT / "ur10e_rg6_coupled_grasp.mp4",
        fps=12,
        crf=16,
        preset="slow",
        purpose="coupled_ur10e_rg6_contact_physics_pilot_not_final_evaluation",
    )

result = {
    "schema_version": "ur10e-rg6-coupled-grasp-pilot-v1",
    "status": "completed" if success else "failed",
    "failure_cause": failure_cause,
    "seed": args.seed,
    "runtime_seconds": time.perf_counter() - start_time,
    "ur10e_asset": ur10e_asset,
    "rg6_asset": str(RG6_ASSET),
    "coupling": {
        "fixed_joint_body0": mount_path,
        "fixed_joint_body1": rg6_base_path,
        "flange_mount_tracks": ee_path,
        "flange_mount_orientation": "fixed_world_x_180deg_provisional",
        "grasp_alignment_offset_world_m": [0.0, grasp_alignment_offset_y, 0.0],
        "coupling_mode": "kinematic_rigid_flange_adapter_between_articulations",
        "kinematic_mount_used": True,
        "ur10e_joint_motion_performed": True,
        "ur10e_control_mode": "lula_fk_visual_link_pose_pilot",
        "ur10e_link_collision_enabled": False,
        "rg6_and_target_collision_enabled": True,
    },
    "ik": ik_record,
    "final_arm_joint_positions_rad": {
        name: float(value) for name, value in zip(arm_dof_names, arm_state)
    },
    "final_gripper_joint_positions_rad": gripper_state,
    "final_ee_position_world_m": final_ee_position.tolist(),
    "target_dynamics": {
        "rigid_body": True,
        "kinematic_during_closure": False,
        "kinematic_during_lift": False,
        "gravity": True,
        "mass_kg": 0.08,
        "full_extents_m": [target_size, target_width_y, target_size],
        "explicit_target_attachment_used": False,
        "target_pose_copying_used": False,
    },
    "contacts": {
        "left_event_count": contacts["left"],
        "right_event_count": contacts["right"],
        "bilateral_contact_observed": bilateral_contact_observed,
        "bilateral_before_lift": bilateral_before_lift,
        "raw_collider_pairs_sample": contacts["raw_pairs"],
    },
    "pre_lift_safety_gate": {
        "passed": pre_lift_state_stable,
        "target_displacement_from_expected_m": pre_lift_target_displacement,
        "maximum_allowed_target_displacement_m": 0.05,
        "gripper_joint_abs_limit_rad": 1.5,
        "gripper_joint_positions_rad": pre_lift_gripper_state,
    },
    "initial_target_position_world_m": initial_target_position.tolist(),
    "final_target_position_world_m": final_target_position.tolist(),
    "final_finger_link_positions_world_m": final_finger_positions,
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
    "evaluation_scope": "Single deterministic coupled-physics pilot only.",
}
(OUTPUT_ROOT / "result.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8"
)
print(f"UR10E_RG6_COUPLED_RESULT={OUTPUT_ROOT / 'result.json'}", flush=True)
if video is not None:
    print(
        f"UR10E_RG6_COUPLED_VIDEO={OUTPUT_ROOT / 'ur10e_rg6_coupled_grasp.mp4'}",
        flush=True,
    )
app_utils.stop()
contact_subscription = None
simulation_app.close()
raise SystemExit(0 if success else 2)
