"""Pilot grasp using one physical UR10e+RG6 articulation on one GPU.

The target remains a dynamic body. It is lifted only by RG6 contact and
friction while the six UR10e joints execute the lift; no target attachment or
target-pose copying is used.
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
IMPORT_RESULT = (
    ROOT
    / "assets"
    / "robots"
    / "ur10e_rg6"
    / "isaac6_import"
    / "import_result.json"
)
FLOATING_IMPORT_RESULT = (
    ROOT
    / "assets"
    / "robots"
    / "ur10e_rg6"
    / "isaac6_import_floating"
    / "import_result.json"
)
SCENE_MOUNTED_IMPORT_RESULT = (
    ROOT
    / "assets"
    / "robots"
    / "ur10e_rg6"
    / "isaac6_import_scene_mounted"
    / "import_result.json"
)

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--output-root", type=Path)
parser.add_argument("--no-video", action="store_true")
parser.add_argument(
    "--stability-smoke-only",
    action="store_true",
    help="Hold the seed-0 home pose and exit before pregrasp execution.",
)
parser.add_argument(
    "--enable-arm-collisions",
    action="store_true",
    help="Enable imported UR10e collision meshes and monitor whole-arm contacts.",
)
parser.add_argument(
    "--same-scene-benchmark",
    action="store_true",
    help="Keep the seeded open-container scene and solve grasp/lift IK for its target.",
)
parser.add_argument(
    "--trigger-result",
    type=Path,
    help="Completed live pipeline result whose terminal grasp authorizes this pilot.",
)
parser.add_argument(
    "--automatic-ik-smoke",
    action="store_true",
    help=(
        "Run seed-specific grasp/lift IK as a physics-only debug experiment "
        "without claiming that Qwen authorized the grasp. Simulator ground "
        "truth is used only to validate automatic trajectory generation."
    ),
)
parser.add_argument(
    "--rgbd-localization",
    type=Path,
    help=(
        "Use target_red and occluder_orange world centers estimated from a "
        "saved RGB-D observation for grasp-pose planning. Simulator layout "
        "remains the physical scene and is read only for post-run evaluation."
    ),
)
args = parser.parse_args()

physical_gpu = int(os.environ.get("PHYSICAL_GPU", "5"))
if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
    raise RuntimeError(
        f"CUDA_VISIBLE_DEVICES must be exactly {physical_gpu}"
    )
if args.seed < 0:
    raise ValueError("seed must be non-negative")
if args.automatic_ik_smoke and not args.same_scene_benchmark:
    raise ValueError("--automatic-ik-smoke requires --same-scene-benchmark")
if args.automatic_ik_smoke and args.trigger_result is not None:
    raise ValueError(
        "--automatic-ik-smoke and --trigger-result are mutually exclusive"
    )
if args.rgbd_localization is not None and not args.same_scene_benchmark:
    raise ValueError("--rgbd-localization requires --same-scene-benchmark")

OUTPUT_ROOT = (
    args.output_root.resolve()
    if args.output_root
    else ROOT
    / "outputs"
    / "ur10e_rg6_physics"
    / f"composite_grasp_seed{args.seed:03d}"
)
ASSET = Path(
    json.loads(
        (
            IMPORT_RESULT
        ).read_text(encoding="utf-8")
    )["output_usd"]
).resolve()

app = SimulationApp(
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
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from omni.physx import get_physx_simulation_interface
from omni.physx.bindings._physx import ContactEventType
from omni.physx.scripts.physicsUtils import add_physics_material_to_prim
from pxr import (
    Gf,
    PhysicsSchemaTools,
    PhysxSchema,
    UsdGeom,
    UsdPhysics,
    UsdShade,
)

from build_observation_video import build_frame_sequence_video
from observation_capture import create_fixed_overview_camera, load_observation_config
from seeded_benchmark import apply_layout, generate_layout
def world_position(prim) -> np.ndarray:
    return np.asarray(
        omni.usd.get_world_transform_matrix(prim).ExtractTranslation(),
        dtype=np.float64,
    )


def world_quaternion_wxyz(prim) -> np.ndarray:
    quat = omni.usd.get_world_transform_matrix(prim).ExtractRotation().GetQuat()
    imag = quat.GetImaginary()
    return np.asarray([quat.GetReal(), imag[0], imag[1], imag[2]])


def set_cube(
    cube: UsdGeom.Cube,
    position: tuple[float, float, float],
    extents: tuple[float, float, float],
) -> None:
    cube.CreateSizeAttr().Set(1.0)
    xform = UsdGeom.Xformable(cube)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*position))
    xform.AddScaleOp().Set(Gf.Vec3f(*extents))


context = omni.usd.get_context()
scene_path = ROOT / "assets" / "scenes" / "open_container_benchmark.usda"
if not context.open_stage(str(scene_path)):
    raise RuntimeError(f"Could not open {scene_path}")
for _ in range(25):
    app.update()
stage = context.get_stage()
layout = None
benchmark_robot_base_in_authored_world = np.asarray(
    [-0.20, 0.32, 0.76], dtype=np.float64
)
benchmark_world_to_robot_base = -benchmark_robot_base_in_authored_world
if args.same_scene_benchmark:
    layout = generate_layout(args.seed)
    apply_layout(stage, layout)
    if args.automatic_ik_smoke:
        trigger = None
    else:
        if args.trigger_result is None:
            raise ValueError(
                "--same-scene-benchmark requires either --trigger-result or "
                "--automatic-ik-smoke"
            )
        trigger = json.loads(args.trigger_result.read_text(encoding="utf-8"))
        if (
            trigger.get("status") != "completed"
            or trigger.get("terminal_action") != "grasp"
            or trigger.get("seed") != args.seed
        ):
            raise RuntimeError(
                "Same-scene grasp requires a completed terminal grasp "
                "decision for the same seed"
            )

    # Keep the imported fixed-base articulation at its stable authored origin.
    # Express the complete benchmark relative to the UR10e base frame instead
    # of translating the fixed articulation. This is the same rigid global
    # transform used when a real robot reports table/object poses in base
    # coordinates, and it preserves every robot-to-scene relative pose.
    benchmark_environment_roots = (
        "/World/Ground",
        "/World/LabBackWall",
        "/World/LabSideWall",
        "/World/WorkBench",
        "/World/WorkMat",
        "/World/OpenContainer",
        "/World/TargetRed",
        "/World/OccluderOrange",
        "/World/DistractorYellow",
        "/World/DistractorBlue",
        "/World/DistractorGreen",
        "/World/BoundaryPurple",
        "/World/RearRedCandidate",
        "/World/LabProps",
    )
    for path in benchmark_environment_roots:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            continue
        xformable = UsdGeom.Xformable(prim)
        translate_ops = [
            op
            for op in xformable.GetOrderedXformOps()
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
        ]
        if translate_ops:
            current = np.asarray(translate_ops[0].Get(), dtype=np.float64)
            translate_ops[0].Set(
                Gf.Vec3d(*(current + benchmark_world_to_robot_base))
            )
        else:
            xformable.AddTranslateOp().Set(
                Gf.Vec3d(*benchmark_world_to_robot_base)
            )
else:
    trigger = None

legacy_rg6 = stage.GetPrimAtPath("/World/RobotSystem/RG6")
if legacy_rg6.IsValid():
    physics = legacy_rg6.GetVariantSets().GetVariantSet("Physics")
    if physics.IsValid():
        physics.SetVariantSelection("none")
    UsdGeom.Imageable(legacy_rg6).MakeInvisible()
legacy_ur = stage.GetPrimAtPath("/World/RobotSystem/UR10e")
if legacy_ur.IsValid():
    legacy_ur.SetActive(False)
if not args.same_scene_benchmark:
    for path in (
        "/World/WorkBench",
        "/World/WorkMat",
        "/World/OpenContainer",
        "/World/DistractorBlue",
    ):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            prim.SetActive(False)

robot_path = "/World/UR10eRG6"
robot_prim = stage.DefinePrim(robot_path, "Xform")
robot_prim.GetReferences().AddReference(str(ASSET))
variant = robot_prim.GetVariantSets().GetVariantSet("Physics")
if variant.IsValid():
    variant.SetVariantSelection("physx")
for _ in range(30):
    app.update()

arm_names = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
rg6_names = [
    "rg6_finger_joint",
    "rg6_left_inner_knuckle_joint",
    "rg6_left_inner_finger_joint",
    "rg6_right_outer_knuckle_joint",
    "rg6_right_inner_knuckle_joint",
    "rg6_right_inner_finger_joint",
]
for name in arm_names:
    joint = stage.GetPrimAtPath(f"{robot_path}/Physics/{name}")
    drive = UsdPhysics.DriveAPI.Get(joint, "angular")
    if drive:
        drive.CreateStiffnessAttr().Set(1000.0)
        drive.CreateDampingAttr().Set(50.0)
        drive.CreateMaxForceAttr().Set(400.0)

# Keep legacy runs reproducible, but permit the next safety stage to enable
# every imported UR10e collision mesh. Self-collision remains disabled in the
# imported articulation; contacts against the scene and target are monitored.
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if not path.startswith(f"{robot_path}/Geometry/"):
        continue
    if "/rg6_" in path:
        continue
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(
            args.enable_arm_collisions
        )

# The imported URDF and the Lula configuration use different frame
# conventions.  Use the measured, stable home configuration to place this
# integration target on the real RG6 centerline instead of pretending that an
# invalid IK result is a grasp.
ik_record = None
localization_metrics = None
if args.same_scene_benchmark:
    target_start = np.asarray(
        layout["positions_world_m"]["target_red"], dtype=np.float64
    )
    target_extents = np.asarray(
        layout["geometry_overrides_world_m"]["target_red_scale"],
        dtype=np.float64,
    )
    target_size = float(target_extents[0])
    centerline = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)

    mg_path = Path(
        get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
    )
    ur10e_config = (
        mg_path / "motion_policy_configs" / "universal_robots" / "ur10e"
    )
    kinematics = LulaKinematicsSolver(
        robot_description_path=str(
            ur10e_config / "rmpflow" / "ur10e_robot_description.yaml"
        ),
        urdf_path=str(ur10e_config / "ur10e.urdf"),
    )
    # The same-scene composite is parented below the authored RobotSystem
    # mount in the original scene. The physical articulation stays at its
    # stable origin; the benchmark is now represented in robot-base
    # coordinates by the inverse global transform above.
    kinematics.set_robot_base_pose(
        np.asarray([0.0, 0.0, 0.0], dtype=np.float64),
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    kinematics_frame = next(
        name
        for name in ("flange", "ee_link", "tool0", "wrist_3_link")
        if name in kinematics.get_all_frame_names()
    )
    target_start += benchmark_world_to_robot_base
    scene_occluder_position = (
        np.asarray(
            layout["positions_world_m"]["occluder_orange"],
            dtype=np.float64,
        )
        + benchmark_world_to_robot_base
    )
    occluder_xy = scene_occluder_position[:2]
    rgbd_localization = None
    if args.rgbd_localization is not None:
        rgbd_localization = json.loads(
            args.rgbd_localization.resolve().read_text(encoding="utf-8")
        )
        estimates = rgbd_localization.get("estimates", {})
        required = {"target_red", "occluder_orange"}
        if not required.issubset(estimates):
            raise ValueError(
                "RGB-D localization must contain target_red and "
                "occluder_orange estimates"
            )
        if rgbd_localization.get(
            "simulator_ground_truth_used_for_estimate"
        ):
            raise ValueError(
                "RGB-D localization reports simulator ground-truth leakage"
            )
        perceived_target = np.asarray(
            estimates["target_red"]["center_world_m"],
            dtype=np.float64,
        )
        perceived_occluder = np.asarray(
            estimates["occluder_orange"]["center_world_m"],
            dtype=np.float64,
        )
        if not (
            np.all(np.isfinite(perceived_target))
            and np.all(np.isfinite(perceived_occluder))
        ):
            raise ValueError("RGB-D localization contains non-finite centers")
        occluder_xy = perceived_occluder[:2]
    else:
        perceived_target = None
        perceived_occluder = None
    # Downward-facing flange with its finger-spread axis perpendicular to the
    # target/occluder line. This deterministic pilot rule prevents the open
    # fingers from striking the known debug occluder before closing.
    home_seed = np.asarray(
        [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
        dtype=np.float64,
    )
    # The seeded target is authored 16 mm above its settled pose. Account for
    # the known container-bottom height and target half-height so the final
    # RG6 centerline is based on the resting target rather than its spawn pose.
    settled_target = target_start.copy()
    container_bottom = stage.GetPrimAtPath("/World/OpenContainer/Bottom")
    settled_target[2] = (
        world_position(container_bottom)[2]
        + 0.018 * 0.5
        + target_extents[2] * 0.5
    )
    # Start the dynamic manipulation object on the container bottom. The
    # perception scene authored it 16 mm high for visibility; dropping that
    # tall cuboid at physics start can tip it before the robot arrives.
    authored_target_in_robot_base = target_start.copy()
    target_start = settled_target.copy()
    planning_target = (
        perceived_target.copy()
        if perceived_target is not None
        else settled_target.copy()
    )
    if perceived_target is not None:
        localization_metrics = {
            "evaluation_only_ground_truth_read_after_estimate": True,
            "target_position_error_m": float(
                np.linalg.norm(perceived_target - settled_target)
            ),
            "target_xy_error_m": float(
                np.linalg.norm(
                    perceived_target[:2] - settled_target[:2]
                )
            ),
            "occluder_position_error_m": float(
                np.linalg.norm(
                    perceived_occluder - scene_occluder_position
                )
            ),
            "maximum_target_position_error_m": 0.02,
            "target_error_gate_passed": bool(
                np.linalg.norm(perceived_target - settled_target) <= 0.02
            ),
        }
    target_from_occluder_xy = planning_target[:2] - occluder_xy
    grasp_yaw = math.atan2(
        target_from_occluder_xy[0],
        -target_from_occluder_xy[1],
    )
    downward_orientation = np.asarray(
        [0.0, math.cos(grasp_yaw * 0.5), math.sin(grasp_yaw * 0.5), 0.0],
        dtype=np.float64,
    )
    target_x, target_y, target_start_z = map(float, target_start)
    grasp_ee = planning_target + np.asarray([0.0, 0.0, 0.287])
    descent_offsets_m = [0.18, 0.12, 0.08, 0.04, 0.0]
    descent_plan = []
    warm_start = home_seed
    for offset_m in descent_offsets_m:
        waypoint_ee = grasp_ee + np.asarray([0.0, 0.0, offset_m])
        waypoint_joints, waypoint_ok = kinematics.compute_inverse_kinematics(
            kinematics_frame,
            waypoint_ee,
            downward_orientation,
            warm_start=warm_start,
            position_tolerance=0.003,
            orientation_tolerance=0.03,
        )
        waypoint_joints = np.asarray(waypoint_joints, dtype=np.float64)
        descent_plan.append(
            {
                "offset_m": offset_m,
                "position_world_m": waypoint_ee.tolist(),
                "joints_rad": waypoint_joints,
                "ik_success": bool(waypoint_ok),
            }
        )
        if not waypoint_ok:
            raise RuntimeError(f"Same-scene descent IK failed at {offset_m} m")
        warm_start = waypoint_joints
    grasp_joints = descent_plan[-1]["joints_rad"].copy()
    grasp_ik = descent_plan[-1]["ik_success"]
    lift_ee = grasp_ee + np.asarray([0.0, 0.0, 0.18])
    lift_joints, lift_ik = kinematics.compute_inverse_kinematics(
        kinematics_frame,
        lift_ee,
        downward_orientation,
        warm_start=np.asarray(grasp_joints),
        position_tolerance=0.003,
        orientation_tolerance=0.03,
    )
    grasp_joints = np.asarray(grasp_joints, dtype=np.float64)
    lift_joints = np.asarray(lift_joints, dtype=np.float64)
    if not (grasp_ik and lift_ik):
        raise RuntimeError(
            f"Same-scene IK failed: grasp={grasp_ik}, lift={lift_ik}"
        )
    ik_record = {
        "frame": kinematics_frame,
        "authored_perception_target_position_robot_base_m": (
            authored_target_in_robot_base.tolist()
        ),
        "settled_target_position_world_m": settled_target.tolist(),
        "grasp_planning_target_position_world_m": planning_target.tolist(),
        "grasp_planning_occluder_position_world_m": (
            perceived_occluder.tolist()
            if perceived_occluder is not None
            else scene_occluder_position.tolist()
        ),
        "grasp_planning_position_source": (
            "masked_rgbd_world_point_estimate"
            if perceived_target is not None
            else "simulator_ground_truth_debug"
        ),
        "rgbd_localization_file": (
            str(args.rgbd_localization.resolve())
            if args.rgbd_localization is not None
            else None
        ),
        "pregrasp_position_world_m": descent_plan[0][
            "position_world_m"
        ],
        "grasp_position_world_m": grasp_ee.tolist(),
        "lift_position_world_m": lift_ee.tolist(),
        "orientation_wxyz": downward_orientation.tolist(),
        "collision_avoidance_grasp_yaw_rad": grasp_yaw,
        "collision_avoidance_source": "simulator_ground_truth_debug_only",
        "grasp_success": bool(grasp_ik),
        "lift_success": bool(lift_ik),
        "grasp_joints_rad": grasp_joints.tolist(),
        "lift_joints_rad": lift_joints.tolist(),
        "descent_waypoints": [
            {
                **waypoint,
                "joints_rad": waypoint["joints_rad"].tolist(),
            }
            for waypoint in descent_plan
        ],
    }
else:
    smoke = json.loads(
        (
            ROOT
            / "outputs"
            / "ur10e_rg6_physics"
            / "composite_smoke.json"
        ).read_text(encoding="utf-8")
    )
    poses = smoke["home_rg6_frame_poses"]
    measured_base = np.asarray(
        poses["base"]["position_world_m"], dtype=np.float64
    )
    centerline = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    target_start = measured_base + centerline * 0.287
    target_size = 0.045
    target_extents = np.asarray(
        [target_size, target_size, target_size], dtype=np.float64
    )
    target_x, target_y, target_start_z = map(float, target_start)

# Use a thin tabletop with legs outside the grasp corridor.  A solid pedestal
# directly below the target visually and physically blocks the horizontal RG6
# approach, making the robot appear attached to a tall cuboid.
if not args.same_scene_benchmark:
    table_top_z = target_start_z - target_size * 0.5 - 0.002
    table_thickness = 0.04
    table = UsdGeom.Cube.Define(stage, "/World/CompositePilotTableTop")
    set_cube(
        table,
        (target_x, target_y, table_top_z - table_thickness * 0.5),
        (0.72, 0.52, table_thickness),
    )
    UsdPhysics.CollisionAPI.Apply(table.GetPrim())
    leg_height = table_top_z - table_thickness
    for index, (dx, dy) in enumerate(
        ((-0.29, -0.19), (-0.29, 0.19), (0.29, -0.19), (0.29, 0.19))
    ):
        leg = UsdGeom.Cube.Define(stage, f"/World/CompositePilotTableLeg{index}")
        set_cube(
            leg,
            (target_x + dx, target_y + dy, leg_height * 0.5),
            (0.07, 0.07, leg_height),
        )
        UsdPhysics.CollisionAPI.Apply(leg.GetPrim())
else:
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

target_prim = stage.GetPrimAtPath("/World/TargetRed")
set_cube(
    UsdGeom.Cube(target_prim),
    (target_x, target_y, target_start_z),
    tuple(float(value) for value in target_extents),
)
UsdPhysics.CollisionAPI.Apply(target_prim)
target_body = UsdPhysics.RigidBodyAPI.Apply(target_prim)
target_body.CreateRigidBodyEnabledAttr().Set(True)
target_body.CreateKinematicEnabledAttr().Set(False)
UsdPhysics.MassAPI.Apply(target_prim).CreateMassAttr().Set(0.04)
PhysxSchema.PhysxRigidBodyAPI.Apply(target_prim).CreateDisableGravityAttr().Set(False)
PhysxSchema.PhysxContactReportAPI.Apply(target_prim).CreateThresholdAttr().Set(0.0)

rg6_base_prim = next(
    (
        prim
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(robot_path)
        and prim.GetName() == "rg6_onrobot_rg6_base_link"
    ),
    None,
)
if rg6_base_prim is None:
    raise RuntimeError("Could not find preserved RG6 base frame")
chain = str(rg6_base_prim.GetPath())
left_link = f"{chain}/rg6_left_outer_knuckle/rg6_left_inner_finger"
right_link = f"{chain}/rg6_right_outer_knuckle/rg6_right_inner_finger"
left_collision = f"{left_link}/inner_finger_1"
right_collision = f"{right_link}/inner_finger_1"
for path in (left_link, right_link):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing RG6 contact link: {path}")
    PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(0.0)
for prim in stage.Traverse():
    path = str(prim.GetPath())
    if (
        path.startswith(chain)
        and "/rg6_" in path
        and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ):
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(
            0.0
        )
    if (
        args.enable_arm_collisions
        and path.startswith(robot_path)
        and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ):
        PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr().Set(
            0.0
        )

material = UsdShade.Material.Define(stage, "/World/PhysicsMaterials/CompositeGrip")
material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
material_api.CreateStaticFrictionAttr().Set(4.0)
material_api.CreateDynamicFrictionAttr().Set(3.0)
material_api.CreateRestitutionAttr().Set(0.0)
for path in (str(target_prim.GetPath()), left_collision, right_collision):
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        add_physics_material_to_prim(stage, prim, material.GetPath())

contacts = {
    "left": 0,
    "right": 0,
    "left_finger_pad": 0,
    "right_finger_pad": 0,
    "raw_pairs": [],
    "events_by_phase": {},
    "unexpected_environment_pairs": [],
}
target_path = str(target_prim.GetPath())
contact_context = {"phase": "initialization", "simulation_step": 0}


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
        sample = sorted(pair)
        if len(contacts["raw_pairs"]) < 20 and sample not in contacts["raw_pairs"]:
            contacts["raw_pairs"].append(sample)
        robot_paths = [path for path in pair if path.startswith(robot_path)]
        if not robot_paths:
            continue
        phase = contact_context["phase"]
        phase_record = contacts["events_by_phase"].setdefault(
            phase,
            {"event_count": 0, "pairs": [], "first_simulation_step": None},
        )
        phase_record["event_count"] += 1
        if phase_record["first_simulation_step"] is None:
            phase_record["first_simulation_step"] = contact_context[
                "simulation_step"
            ]
        if len(phase_record["pairs"]) < 12 and sample not in phase_record["pairs"]:
            phase_record["pairs"].append(sample)

        if target_path in pair:
            other = next((path for path in pair if path != target_path), "")
            if "/rg6_left_" in other:
                contacts["left"] += 1
                if other == left_collision:
                    contacts["left_finger_pad"] += 1
            elif "/rg6_right_" in other:
                contacts["right"] += 1
                if other == right_collision:
                    contacts["right_finger_pad"] += 1
            elif other.startswith(robot_path):
                if sample not in contacts["unexpected_environment_pairs"]:
                    contacts["unexpected_environment_pairs"].append(sample)
            continue

        if not all(path.startswith(robot_path) for path in pair):
            if sample not in contacts["unexpected_environment_pairs"]:
                contacts["unexpected_environment_pairs"].append(sample)


contact_subscription = get_physx_simulation_interface().subscribe_contact_report_events(
    on_contact
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
frame_root = OUTPUT_ROOT / "frames"
frame_root.mkdir(parents=True, exist_ok=True)
for old in frame_root.glob("frame_*.png"):
    old.unlink()

config = load_observation_config(ROOT)
config["overview_camera"]["resolution"] = [1920, 1080]
if args.same_scene_benchmark:
    # Preserve a close, high-detail view while lifting the framing enough to
    # include the complete manipulator. This is only about 18 cm farther back
    # and 25 cm higher than the previous camera.
    config["overview_camera"]["position_world_m"] = (
        np.asarray([1.94, -2.50, 2.10])
        + benchmark_world_to_robot_base
    ).tolist()
    config["overview_camera"]["look_at_world_m"] = (
        np.asarray([0.46, 0.02, 1.20])
        + benchmark_world_to_robot_base
    ).tolist()
else:
    config["overview_camera"]["position_world_m"] = [2.15, -2.75, 1.65]
    config["overview_camera"]["look_at_world_m"] = [0.05, -0.45, 0.58]
config["overview_camera"]["focal_length_mm"] = 38.0
camera = create_fixed_overview_camera(stage, config)
render_product = rep.create.render_product(
    str(camera.GetPath()),
    tuple(config["overview_camera"]["resolution"]),
)
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach(render_product)

# Author the safe initial joint state before the first physics frame. If the
# same-scene articulation starts from the imported all-zero state even for one
# frame, the RG6 intersects the bench, mat, and container wall before the
# tensor API is available to teleport it to home.
initial_arm = (
    home_seed
    if args.same_scene_benchmark
    else np.asarray(
        [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0],
        dtype=np.float64,
    )
)
initial_master = -0.20 if args.same_scene_benchmark else -0.45
initial_by_name = {
    **dict(zip(arm_names, initial_arm)),
    "rg6_finger_joint": initial_master,
    "rg6_left_inner_knuckle_joint": -initial_master,
    "rg6_left_inner_finger_joint": initial_master,
    "rg6_right_outer_knuckle_joint": -initial_master,
    "rg6_right_inner_knuckle_joint": -initial_master,
    "rg6_right_inner_finger_joint": initial_master,
}
for name, position_rad in initial_by_name.items():
    joint = stage.GetPrimAtPath(f"{robot_path}/Physics/{name}")
    joint_state = PhysxSchema.JointStateAPI.Get(joint, "angular")
    if not joint_state:
        joint_state = PhysxSchema.JointStateAPI.Apply(joint, "angular")
    joint_state.CreatePositionAttr().Set(math.degrees(float(position_rad)))
    joint_state.CreateVelocityAttr().Set(0.0)
    drive = UsdPhysics.DriveAPI.Get(joint, "angular")
    if drive:
        drive.CreateTargetPositionAttr().Set(
            math.degrees(float(position_rad))
        )
        drive.CreateTargetVelocityAttr().Set(0.0)

SimulationManager.setup_simulation(dt=1.0 / 120.0)
robot = Articulation(robot_path)
app.update()
app_utils.play()
app.update()
dof_names = list(robot.dof_names)
if len(dof_names) != 12 or set(dof_names) != set(arm_names + rg6_names):
    raise RuntimeError(f"Unexpected composite DOFs: {dof_names}")

if not args.same_scene_benchmark:
    home = np.asarray(
        [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
    )
    grasp_joints = home.copy()
    lift_joints = home.copy()
    lift_joints[1] -= 0.35
open_master = -0.20 if args.same_scene_benchmark else -0.45
close_master = 0.45


def command(arm: np.ndarray, master: float) -> np.ndarray:
    by_name = {
        **dict(zip(arm_names, arm)),
        "rg6_finger_joint": master,
        "rg6_left_inner_knuckle_joint": -master,
        "rg6_left_inner_finger_joint": master,
        "rg6_right_outer_knuckle_joint": -master,
        "rg6_right_inner_knuckle_joint": -master,
        "rg6_right_inner_finger_joint": master,
    }
    return np.asarray([by_name[name] for name in dof_names], dtype=np.float32)


def measured() -> np.ndarray:
    values = robot.get_dof_positions().numpy()
    return values[0] if values.ndim > 1 else values


from PIL import Image, ImageDraw, ImageFont

try:
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22
    )
except OSError:
    font = ImageFont.load_default()
frames: list[Path] = []
trajectory_records: list[dict] = []


def write_abort_record(cause: str, phase: str) -> None:
    state = measured()
    record = {
        "schema_version": "ur10e-rg6-trajectory-abort-v1",
        "status": "failed",
        "failure_cause": cause,
        "phase": phase,
        "seed": args.seed,
        "same_scene_benchmark": args.same_scene_benchmark,
        "ur10e_link_collision_enabled": args.enable_arm_collisions,
        "finite_joint_state": bool(np.all(np.isfinite(state))),
        "measured_dofs_rad": state.tolist(),
        "trajectory_records": trajectory_records,
        "contacts": contacts,
        "gpu_policy": {
            "physical_gpu": physical_gpu,
            "renderer_active_gpu": physical_gpu,
            "physics_cuda_device": 0,
            "multi_gpu": False,
        },
        "valid_for_final_evaluation": False,
    }
    (OUTPUT_ROOT / "abort_result.json").write_text(
        json.dumps(record, indent=2) + "\n",
        encoding="utf-8",
    )


def capture(label: str) -> None:
    if args.no_video:
        return
    rep.orchestrator.step(rt_subframes=2)
    rgba = np.asarray(rgb.get_data()).copy()
    image = Image.fromarray(rgba[:, :, :3].astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(image)
    metrics = (
        f"contact L/R={contacts['left']}/{contacts['right']}  "
        f"target z={world_position(target_prim)[2]:.3f} m"
    )
    draw.rounded_rectangle((15, 15, 760, 92), radius=7, fill=(0, 0, 0))
    draw.text((25, 22), label, fill=(255, 255, 255), font=font)
    draw.text((25, 56), metrics, fill=(200, 225, 240), font=font)
    path = frame_root / f"frame_{len(frames):04d}.png"
    image.save(path)
    frames.append(path)


def arm_tracking_error(arm: np.ndarray) -> float:
    current = measured()
    by_name = {name: float(value) for name, value in zip(dof_names, current)}
    return float(
        max(abs(by_name[name] - desired) for name, desired in zip(arm_names, arm))
    )


def hold(arm: np.ndarray, master: float, steps: int, label: str) -> None:
    contact_context["phase"] = label
    desired = command(arm, master)
    for step in range(steps):
        robot.set_dof_position_targets(desired)
        app.update()
        contact_context["simulation_step"] += 1
        if step % 10 == 0 and not np.all(np.isfinite(measured())):
            write_abort_record("non_finite_articulation_state", label)
            raise RuntimeError(f"Non-finite articulation state during {label}")
        if step % 4 == 0:
            capture(label)
    trajectory_records.append(
        {
            "phase": label,
            "kind": "hold",
            "steps": steps,
            "maximum_arm_error_at_end_rad": arm_tracking_error(arm),
            "rg6_base_position_world_m": world_position(
                stage.GetPrimAtPath(chain)
            ).tolist(),
            "unexpected_environment_pair_count": len(
                contacts["unexpected_environment_pairs"]
            ),
        }
    )


def transition(
    start_arm: np.ndarray,
    end_arm: np.ndarray,
    start_master: float,
    end_master: float,
    steps: int,
    label: str,
    collision_check: bool = False,
    settle_steps: int = 60,
) -> None:
    contact_context["phase"] = label
    unexpected_before = len(contacts["unexpected_environment_pairs"])
    maximum_tracking_error = 0.0
    for step in range(1, steps + 1):
        alpha = step / steps
        arm = start_arm + alpha * (end_arm - start_arm)
        master = start_master + alpha * (end_master - start_master)
        robot.set_dof_position_targets(command(arm, master))
        app.update()
        contact_context["simulation_step"] += 1
        if step % 10 == 0 and not np.all(np.isfinite(measured())):
            write_abort_record("non_finite_articulation_state", label)
            raise RuntimeError(f"Non-finite articulation state during {label}")
        if step % 10 == 0:
            maximum_tracking_error = max(
                maximum_tracking_error,
                arm_tracking_error(arm),
            )
            if collision_check and len(
                contacts["unexpected_environment_pairs"]
            ) > unexpected_before:
                write_abort_record(
                    "unexpected_rg6_environment_collision",
                    label,
                )
                raise RuntimeError(
                    f"Unexpected RG6-environment collision during {label}"
                )
        if step % 4 == 0:
            capture(label)
    # A finite-duration position trajectory ends with nonzero servo lag.
    # Hold the final command before applying the 0.05 rad acceptance gate.
    for settle_step in range(1, settle_steps + 1):
        robot.set_dof_position_targets(command(end_arm, end_master))
        app.update()
        contact_context["simulation_step"] += 1
        if settle_step % 10 == 0:
            if not np.all(np.isfinite(measured())):
                write_abort_record("non_finite_articulation_state", label)
                raise RuntimeError(
                    f"Non-finite articulation state while settling {label}"
                )
            maximum_tracking_error = max(
                maximum_tracking_error,
                arm_tracking_error(end_arm),
            )
            if collision_check and len(
                contacts["unexpected_environment_pairs"]
            ) > unexpected_before:
                write_abort_record(
                    "unexpected_rg6_environment_collision",
                    label,
                )
                raise RuntimeError(
                    f"Unexpected RG6-environment collision while settling {label}"
                )
        if settle_step % 4 == 0:
            capture(label)
    final_error = arm_tracking_error(end_arm)
    trajectory_records.append(
        {
            "phase": label,
            "kind": "transition",
            "steps": steps,
            "settle_steps": settle_steps,
            "collision_check_enabled": collision_check,
            "maximum_arm_tracking_error_rad": maximum_tracking_error,
            "final_arm_tracking_error_rad": final_error,
            "rg6_base_position_world_m": world_position(
                stage.GetPrimAtPath(chain)
            ).tolist(),
            "new_unexpected_environment_pair_count": len(
                contacts["unexpected_environment_pairs"]
            )
            - unexpected_before,
        }
    )
    if final_error > 0.05:
        write_abort_record("arm_tracking_error_exceeded", label)
        raise RuntimeError(
            f"Arm tracking error {final_error:.6f} rad after {label}"
        )


def close_until_bilateral_contact(
    arm: np.ndarray,
    start_master: float,
    end_master: float,
    steps: int,
    label: str,
) -> float:
    """Close RG6 slowly and stop at the first bilateral target contact.

    Driving the fingers all the way to the nominal fully-closed command after
    contact stores excessive constraint energy in PhysX.  On release that can
    launch the target and produce secondary container collisions.  The real
    RG6 controller must likewise stop or switch to force control at contact.
    """

    contact_context["phase"] = label
    left_before = contacts["left_finger_pad"]
    right_before = contacts["right_finger_pad"]
    unexpected_before = len(contacts["unexpected_environment_pairs"])
    initial_target_position = world_position(target_prim)
    contact_master = start_master
    bilateral_step = None

    for step in range(1, steps + 1):
        alpha = step / steps
        contact_master = start_master + alpha * (end_master - start_master)
        robot.set_dof_position_targets(command(arm, contact_master))
        app.update()
        contact_context["simulation_step"] += 1

        if not np.all(np.isfinite(measured())):
            write_abort_record("non_finite_articulation_state", label)
            raise RuntimeError(f"Non-finite articulation state during {label}")
        if len(contacts["unexpected_environment_pairs"]) > unexpected_before:
            write_abort_record("unexpected_rg6_environment_collision", label)
            raise RuntimeError(
                f"Unexpected RG6-environment collision during {label}"
            )

        target_displacement = float(
            np.linalg.norm(world_position(target_prim) - initial_target_position)
        )
        if target_displacement > 0.05:
            write_abort_record("target_displaced_during_closure", label)
            raise RuntimeError(
                f"Target displaced {target_displacement:.6f} m during {label}"
            )

        if step % 4 == 0:
            capture(label)

        if (
            contacts["left_finger_pad"] > left_before
            and contacts["right_finger_pad"] > right_before
        ):
            bilateral_step = step
            break

    bilateral_detected = bilateral_step is not None
    # Add only a very small position preload after the two finger pads make
    # contact.  This maintains normal force during lift without driving the
    # RG6 toward its fully closed limit and recreating the earlier explosive
    # over-constraint.
    first_contact_master = float(contact_master)
    preload_delta_rad = 0.01
    holding_master = min(first_contact_master + preload_delta_rad, end_master)
    if bilateral_detected:
        preload_steps = 12
        for preload_step in range(1, preload_steps + 1):
            alpha = preload_step / preload_steps
            contact_master = (
                first_contact_master
                + alpha * (holding_master - first_contact_master)
            )
            robot.set_dof_position_targets(command(arm, contact_master))
            app.update()
            contact_context["simulation_step"] += 1
            if len(contacts["unexpected_environment_pairs"]) > unexpected_before:
                write_abort_record(
                    "unexpected_rg6_environment_collision_during_preload",
                    label,
                )
                raise RuntimeError(
                    f"Unexpected RG6-environment collision during {label} preload"
                )
            target_displacement = float(
                np.linalg.norm(
                    world_position(target_prim) - initial_target_position
                )
            )
            if target_displacement > 0.05:
                write_abort_record("target_displaced_during_preload", label)
                raise RuntimeError(
                    f"Target displaced {target_displacement:.6f} m during preload"
                )
            if preload_step % 4 == 0:
                capture(f"{label} with conservative preload")

    trajectory_records.append(
        {
            "phase": label,
            "kind": "contact_terminated_close",
            "maximum_steps": steps,
            "executed_steps": bilateral_step if bilateral_detected else steps,
            "bilateral_contact_detected": bilateral_detected,
            "first_finger_pad_contact_master_command_rad": first_contact_master,
            "preload_delta_rad": preload_delta_rad,
            "holding_master_command_rad": float(contact_master),
            "left_finger_pad_events_added": (
                contacts["left_finger_pad"] - left_before
            ),
            "right_finger_pad_events_added": (
                contacts["right_finger_pad"] - right_before
            ),
            "target_displacement_m": float(
                np.linalg.norm(
                    world_position(target_prim) - initial_target_position
                )
            ),
        }
    )
    if not bilateral_detected:
        write_abort_record("bilateral_target_contact_not_detected", label)
        raise RuntimeError(f"Bilateral target contact not detected during {label}")

    return float(contact_master)


if args.same_scene_benchmark:
    home = home_seed.copy()
    robot.set_dof_positions(command(home, open_master))
    robot.set_dof_position_targets(command(home, open_master))
    hold(home, open_master, 120, "settle_scene_at_safe_home")
    if args.stability_smoke_only:
        smoke_state = measured()
        smoke_error = arm_tracking_error(home)
        smoke_success = (
            bool(np.all(np.isfinite(smoke_state)))
            and smoke_error <= 0.05
            and not contacts["unexpected_environment_pairs"]
        )
        smoke_video = None
        if frames:
            smoke_video = build_frame_sequence_video(
                frames,
                OUTPUT_ROOT / "ur10e_rg6_stability_smoke.mp4",
                fps=10,
                crf=17,
                preset="slow",
                purpose="ur10e_rg6_table_mount_stability_smoke",
            )
        smoke_result = {
            "schema_version": "ur10e-rg6-table-mount-stability-v1",
            "status": "completed" if smoke_success else "failed",
            "seed": args.seed,
            "coordinate_frame": "ur10e_base",
            "authored_robot_base_world_m": (
                benchmark_robot_base_in_authored_world.tolist()
            ),
            "benchmark_world_to_robot_base_translation_m": (
                benchmark_world_to_robot_base.tolist()
            ),
            "finite_joint_state": bool(np.all(np.isfinite(smoke_state))),
            "maximum_arm_error_rad": smoke_error,
            "maximum_allowed_arm_error_rad": 0.05,
            "unexpected_environment_pairs": contacts[
                "unexpected_environment_pairs"
            ],
            "ur10e_link_collision_enabled": args.enable_arm_collisions,
            "trajectory_records": trajectory_records,
            "video": smoke_video,
            "gpu_policy": {
                "physical_gpu": physical_gpu,
                "renderer_active_gpu": physical_gpu,
                "physics_cuda_device": 0,
                "multi_gpu": False,
            },
            "valid_for_final_evaluation": False,
        }
        smoke_path = OUTPUT_ROOT / "stability_result.json"
        smoke_path.write_text(
            json.dumps(smoke_result, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"UR10E_RG6_STABILITY_RESULT={smoke_path}", flush=True)
        app_utils.stop()
        contact_subscription = None
        app.close()
        raise SystemExit(0 if smoke_success else 2)
    initial_target = world_position(target_prim)
    pregrasp_joints = descent_plan[0]["joints_rad"]
    transition(
        home,
        pregrasp_joints,
        open_master,
        open_master,
        360,
        "move_to_above_container_pregrasp",
        collision_check=True,
    )
    hold(
        pregrasp_joints,
        open_master,
        60,
        "verify_above_container_pregrasp",
    )
    previous_joints = pregrasp_joints
    for waypoint_index, waypoint in enumerate(descent_plan[1:], start=1):
        next_joints = waypoint["joints_rad"]
        transition(
            previous_joints,
            next_joints,
            open_master,
            open_master,
            120,
            (
                f"collision_checked_descent_{waypoint_index:02d}_"
                f"offset_{waypoint['offset_m']:.2f}m"
            ),
            collision_check=True,
        )
        previous_joints = next_joints
else:
    robot.set_dof_positions(command(grasp_joints, open_master))
    robot.set_dof_position_targets(command(grasp_joints, open_master))
    hold(grasp_joints, open_master, 120, "UR10e + actual RG6 at tabletop")
    initial_target = world_position(target_prim)

measured_pregrasp_dofs = measured().copy()
measured_pregrasp_by_name = {
    name: float(value) for name, value in zip(dof_names, measured_pregrasp_dofs)
}
pregrasp_arm_error = float(
    max(
        abs(measured_pregrasp_by_name[name] - desired)
        for name, desired in zip(arm_names, grasp_joints)
    )
)
(OUTPUT_ROOT / "pregrasp_pose_debug.json").write_text(
    json.dumps(
        {
            "rg6_base_position_world_m": world_position(
                stage.GetPrimAtPath(chain)
            ).tolist(),
            "rg6_base_orientation_wxyz": world_quaternion_wxyz(
                stage.GetPrimAtPath(chain)
            ).tolist(),
            "target_position_from_measured_rg6_centerline_world_m":
                target_start.tolist(),
            "measured_rg6_centerline_world": centerline.tolist(),
            "grasp_joints_rad": grasp_joints.tolist(),
            "lift_joints_rad": lift_joints.tolist(),
            "measured_pregrasp_dofs_rad": measured_pregrasp_by_name,
            "maximum_pregrasp_arm_error_rad": pregrasp_arm_error,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
hold(grasp_joints, open_master, 60, "RG6 aligned around tabletop target")

contact_master = close_until_bilateral_contact(
    grasp_joints,
    open_master,
    close_master,
    360,
    "Actual RG6 contact-terminated closing",
)
hold(
    grasp_joints,
    contact_master,
    120,
    "Holding target at first bilateral contact width",
)

state_before_lift = measured()
finite_before_lift = bool(np.all(np.isfinite(state_before_lift)))
pre_lift_target = world_position(target_prim)
pre_lift_displacement = float(np.linalg.norm(pre_lift_target - initial_target))
rg6_base_prim = stage.GetPrimAtPath(chain)
rg6_base_position_before_lift = world_position(rg6_base_prim)
rg6_base_orientation_before_lift = world_quaternion_wxyz(rg6_base_prim)
bilateral = contacts["left"] > 0 and contacts["right"] > 0
safety_gate = (
    finite_before_lift
    and pregrasp_arm_error <= 0.05
    and bilateral
    and pre_lift_displacement <= 0.05
    and not contacts["unexpected_environment_pairs"]
)
if safety_gate:
    transition(
        grasp_joints,
        lift_joints,
        contact_master,
        contact_master,
        180,
        "UR10e lifting target through actual RG6 contact",
        collision_check=args.enable_arm_collisions,
    )
    hold(
        lift_joints,
        contact_master,
        120,
        "Full manipulator lift verification",
    )

final_target = world_position(target_prim)
final_state = measured()
lift_delta = float(final_target[2] - initial_target[2])
finite_final = bool(np.all(np.isfinite(final_state)))
success = (
    safety_gate
    and finite_final
    and lift_delta >= 0.10
    and not contacts["unexpected_environment_pairs"]
)

video = None
if frames:
    video = build_frame_sequence_video(
        frames,
        OUTPUT_ROOT / "ur10e_rg6_composite_grasp.mp4",
        fps=10,
        crf=17,
        preset="slow",
        purpose="single_articulation_ur10e_rg6_contact_grasp_pilot",
    )

result = {
    "schema_version": "ur10e-rg6-composite-grasp-pilot-v1",
    "status": "completed" if success else "failed",
    "failure_cause": None
    if success
    else (
        "pre_lift_safety_gate_failed"
        if not safety_gate
        else "target_lift_below_threshold_or_nonfinite_state"
    ),
    "seed": args.seed,
    "same_scene_benchmark": args.same_scene_benchmark,
    "automatic_ik_smoke": args.automatic_ik_smoke,
    "grasp_authorization": (
        "physics_debug_without_qwen"
        if args.automatic_ik_smoke
        else (
            "completed_same_seed_live_pipeline_terminal_grasp"
            if trigger is not None
            else "standalone_clean_scene"
        )
    ),
    "trigger_result": str(args.trigger_result) if args.trigger_result else None,
    "trigger_terminal_action": (
        trigger.get("terminal_action") if trigger is not None else None
    ),
    "ik": ik_record,
    "rgbd_localization_evaluation": localization_metrics,
    "trajectory_source": (
        "seed_specific_simulator_ground_truth_debug_ik_generated_this_run"
        if args.same_scene_benchmark
        else "standalone_fixed_contact_physics_pose"
    ),
    "asset": str(ASSET),
    "single_articulation": True,
    "dof_count": len(dof_names),
    "dof_names": dof_names,
    "ur10e_joint_motion_performed": True,
    "ur10e_link_collision_enabled": args.enable_arm_collisions,
    "actual_rg6_joint_motion_performed": True,
    "rg6_contact_terminated_close": {
        "enabled": True,
        "contact_master_command_rad": float(contact_master),
    },
    "target_dynamics": {
        "rigid_body": True,
        "kinematic": False,
        "gravity": True,
        "mass_kg": 0.04,
        "explicit_target_attachment_used": False,
        "target_pose_copying_used": False,
    },
    "contacts": {
        "left_event_count": contacts["left"],
        "right_event_count": contacts["right"],
        "left_finger_pad_event_count": contacts["left_finger_pad"],
        "right_finger_pad_event_count": contacts["right_finger_pad"],
        "bilateral_before_lift": bilateral,
        "raw_collider_pairs_sample": contacts["raw_pairs"],
        "events_by_phase": contacts["events_by_phase"],
        "unexpected_environment_pairs": contacts[
            "unexpected_environment_pairs"
        ],
    },
    "trajectory_records": trajectory_records,
    "safety_gate": {
        "passed": safety_gate,
        "finite_joint_state": finite_before_lift,
        "maximum_pregrasp_arm_error_rad": pregrasp_arm_error,
        "maximum_allowed_arm_error_rad": 0.05,
        "pre_lift_target_displacement_m": pre_lift_displacement,
        "maximum_allowed_displacement_m": 0.05,
        "no_unexpected_environment_contact": not contacts[
            "unexpected_environment_pairs"
        ],
    },
    "rg6_base_pose_before_lift": {
        "position_world_m": rg6_base_position_before_lift.tolist(),
        "orientation_wxyz": rg6_base_orientation_before_lift.tolist(),
        "measured_target_position_world_m": target_start.tolist(),
        "measured_centerline_world": centerline.tolist(),
    },
    "initial_target_position_world_m": initial_target.tolist(),
    "final_target_position_world_m": final_target.tolist(),
    "verified_lift_delta_m": lift_delta,
    "success_threshold_m": 0.10,
    "finite_final_joint_state": finite_final,
    "video": video,
    "gpu_policy": {
        "physical_gpu": physical_gpu,
        "renderer_active_gpu": physical_gpu,
        "physics_cuda_device": 0,
        "multi_gpu": False,
    },
    "valid_for_final_evaluation": False,
    "evaluation_scope": "Single deterministic integration pilot only.",
}
(OUTPUT_ROOT / "result.json").write_text(
    json.dumps(result, indent=2) + "\n",
    encoding="utf-8",
)
print(f"UR10E_RG6_COMPOSITE_RESULT={OUTPUT_ROOT / 'result.json'}", flush=True)
if video:
    print(
        f"UR10E_RG6_COMPOSITE_VIDEO="
        f"{OUTPUT_ROOT / 'ur10e_rg6_composite_grasp.mp4'}",
        flush=True,
    )
app_utils.stop()
contact_subscription = None
app.close()
