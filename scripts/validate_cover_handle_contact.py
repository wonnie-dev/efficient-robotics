#!/usr/bin/env python3
"""Isolate real RG6-fingertip contact with the removable-cover handle.

This development fixture keeps the imported UR10e+RG6 articulation and the
same dynamic plate/handle mass model used by the covered-container episode,
but removes the basket walls.  It diagnoses fingertip/handle contact without
VLM, Scene Graph, MPC, object attachment, or target-pose copying.
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
IMPORT_RESULT = (
    ROOT
    / "assets"
    / "robots"
    / "ur10e_rg6"
    / "isaac6_import"
    / "import_result.json"
)


def resolve_asset_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()
OUTPUT_BASE = ROOT / "outputs" / "rg6_handle_contact_fixture"
ARM_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
MASTER_NAME = "rg6_finger_joint"
FOLLOWER_RATIOS = {
    "rg6_left_inner_knuckle_joint": -1.0,
    "rg6_left_inner_finger_joint": 1.0,
    "rg6_right_outer_knuckle_joint": -1.0,
    "rg6_right_inner_knuckle_joint": -1.0,
    "rg6_right_inner_finger_joint": 1.0,
}
RG6_NAMES = (MASTER_NAME, *FOLLOWER_RATIOS)
# Closure-only run027 measured pose, before any micro-lift motion.
FIXTURE_GRASP_ARM_RAD = np.asarray(
    [
        -0.4817589521408081,
        -0.9684213399887085,
        1.6018526554107666,
        -3.772508382797241,
        -4.232039451599121,
        -6.2789530754089355,
    ],
    dtype=np.float64,
)
FIXTURE_COVER_ROOT_WORLD_M = np.asarray(
    # Jaw-width run001 measured the unloaded collision-envelope midpoint at
    # y=-0.220615 m around true 30 mm contact.  Align the diagnostic handle to
    # that measured jaw center instead of the nominal -0.220 m scene target.
    [0.680, -0.220615, -0.009], dtype=np.float64
)
FIXTURE_HANDLE_ALIGNMENT_OFFSET_WORLD_M = np.asarray(
    [0.0, -0.000615, 0.0], dtype=np.float64
)
PHYSICS_DT_SECONDS = 1.0 / 60.0
COORDINATED_FOLLOWER_REQUEST_BLEND = 0.75


def next_output_dir() -> Path:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in OUTPUT_BASE.glob("run*"):
        try:
            indices.append(int(path.name.removeprefix("run")))
        except ValueError:
            continue
    output = OUTPUT_BASE / f"run{max(indices, default=0) + 1:03d}"
    output.mkdir(parents=False, exist_ok=False)
    return output


def normalized_world_matrix(prim, omni_usd) -> np.ndarray:
    matrix = np.asarray(
        omni_usd.get_world_transform_matrix(prim), dtype=np.float64
    )
    rotation = matrix[:3, :3]
    scales = np.linalg.norm(rotation, axis=1)
    matrix[:3, :3] = rotation / scales[:, None]
    return matrix


def rotation_angle_rad(rotation: np.ndarray) -> float:
    cosine = float(
        np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    )
    return float(math.acos(cosine))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--renderer-gpu", type=int, required=True)
    parser.add_argument("--physics-gpu", type=int, default=0)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--coupling-mode",
        choices=("passive_mimic", "coordinated_drives"),
        default="passive_mimic",
    )
    parser.add_argument(
        "--coordinated-total-drive-effort-limit-nm",
        type=float,
        help=(
            "Development-only aggregate joint-drive effort limit for "
            "coordinated_drives; this is not an RG6 motor-torque claim"
        ),
    )
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=Path("configs/hardware/rg6_lid_simulation.json"),
    )
    args = parser.parse_args()

    physical_gpu_text = os.environ.get("PHYSICAL_GPU")
    if physical_gpu_text is None or not physical_gpu_text.isdigit():
        raise RuntimeError("PHYSICAL_GPU must contain one physical GPU index")
    physical_gpu = int(physical_gpu_text)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu):
        raise RuntimeError("CUDA_VISIBLE_DEVICES must equal PHYSICAL_GPU")
    if args.renderer_gpu != physical_gpu or args.physics_gpu != 0:
        raise ValueError(
            "renderer must match PHYSICAL_GPU and physics must be cuda:0"
        )
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else next_output_dir()
    )
    if args.output_root is not None:
        output_root.mkdir(parents=True, exist_ok=False)

    calibration_path = args.calibration_config
    if not calibration_path.is_absolute():
        calibration_path = ROOT / calibration_path
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    lid = calibration["lid"]
    mapping = calibration["simulation_mapping"]
    acceptance = calibration["acceptance"]
    geometric_contact_master_rad = float(
        mapping["geometric_handle_contact_master_rad"]
    )
    if not 0.0 < geometric_contact_master_rad < 0.60:
        raise ValueError(
            "geometric_handle_contact_master_rad must be inside the closure range"
        )

    started = time.perf_counter()
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
    from isaacsim.core.utils.rotations import rot_matrix_to_quat
    from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
    from omni.physx import get_physx_simulation_interface
    from omni.physx.bindings._physx import ContactEventType
    from omni.physx.scripts.physicsUtils import add_physics_material_to_prim
    from pxr import (
        Gf,
        PhysicsSchemaTools,
        PhysxSchema,
        UsdGeom,
        UsdLux,
        UsdPhysics,
        UsdShade,
    )

    from build_observation_video import build_frame_sequence_video
    from observation_capture import (
        create_fixed_overview_camera,
        load_observation_config,
    )
    from scanned_basket_scene import _composite_cover_mass_properties

    asset = resolve_asset_path(
        json.loads(IMPORT_RESULT.read_text(encoding="utf-8"))["output_usd"]
    )
    context = omni.usd.get_context()
    if not context.open_stage(str(asset)):
        raise RuntimeError(f"Could not open composite asset: {asset}")
    for _ in range(30):
        app.update()
    stage = context.get_stage()
    robot_root = stage.GetDefaultPrim()
    robot_path = str(robot_root.GetPath())
    physics_variant = robot_root.GetVariantSets().GetVariantSet("Physics")
    if physics_variant.IsValid():
        physics_variant.SetVariantSelection("physx")
    for _ in range(15):
        app.update()

    physics_root = f"{robot_path}/Physics"
    follower_drive_removal: dict[str, bool] = {}
    mimic_api_removal: dict[str, bool] = {}
    rg6_drives: dict[str, object] = {}
    if args.coupling_mode == "passive_mimic":
        for name in FOLLOWER_RATIOS:
            joint = stage.GetPrimAtPath(f"{physics_root}/{name}")
            if "NewtonMimicAPI" not in joint.GetAppliedSchemas():
                raise RuntimeError(f"RG6 follower lacks NewtonMimicAPI: {name}")
            had_drive = bool(UsdPhysics.DriveAPI.Get(joint, "angular"))
            if had_drive:
                joint.RemoveAPI(UsdPhysics.DriveAPI, "angular")
            follower_drive_removal[name] = had_drive and not bool(
                UsdPhysics.DriveAPI.Get(joint, "angular")
            )
        if not all(follower_drive_removal.values()):
            raise RuntimeError(
                f"Could not remove follower drives: {follower_drive_removal}"
            )
    else:
        for name in FOLLOWER_RATIOS:
            joint = stage.GetPrimAtPath(f"{physics_root}/{name}")
            if "NewtonMimicAPI" not in joint.GetAppliedSchemas():
                raise RuntimeError(f"RG6 follower lacks NewtonMimicAPI: {name}")
            joint.RemoveAPI("NewtonMimicAPI")
            mimic_api_removal[name] = (
                "NewtonMimicAPI" not in joint.GetAppliedSchemas()
            )
        if not all(mimic_api_removal.values()):
            raise RuntimeError(
                f"Could not remove mimic APIs: {mimic_api_removal}"
            )

    for name in ARM_NAMES:
        joint = stage.GetPrimAtPath(f"{physics_root}/{name}")
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if drive:
            drive.CreateStiffnessAttr().Set(1000.0)
            drive.CreateDampingAttr().Set(50.0)
            drive.CreateMaxForceAttr().Set(400.0)
    current_torque_nm = float(mapping["initial_drive_torque_nm"])
    configured_maximum_torque_nm = float(mapping["maximum_drive_torque_nm"])
    maximum_torque_nm = configured_maximum_torque_nm
    if args.coordinated_total_drive_effort_limit_nm is not None:
        if args.coupling_mode != "coordinated_drives":
            raise ValueError(
                "coordinated drive effort override requires coordinated_drives"
            )
        maximum_torque_nm = float(
            args.coordinated_total_drive_effort_limit_nm
        )
        if not configured_maximum_torque_nm < maximum_torque_nm <= 18.0:
            raise ValueError(
                "development aggregate drive effort must be above the "
                "configured limit and no greater than 18 Nm"
            )
    for name in (
        RG6_NAMES
        if args.coupling_mode == "coordinated_drives"
        else (MASTER_NAME,)
    ):
        joint = stage.GetPrimAtPath(f"{physics_root}/{name}")
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        if args.coupling_mode == "coordinated_drives":
            drive.CreateStiffnessAttr().Set(20.0)
            drive.CreateDampingAttr().Set(1.0)
        rg6_drives[name] = drive

    def apply_total_grip_torque_budget(total_torque_nm: float) -> None:
        per_drive_torque_nm = total_torque_nm / len(rg6_drives)
        for drive in rg6_drives.values():
            drive.CreateMaxForceAttr().Set(per_drive_torque_nm)

    apply_total_grip_torque_budget(current_torque_nm)

    open_master = -0.20
    initial_by_name = {
        **dict(zip(ARM_NAMES, FIXTURE_GRASP_ARM_RAD, strict=True)),
        MASTER_NAME: open_master,
        **{
            name: ratio * open_master
            for name, ratio in FOLLOWER_RATIOS.items()
        },
    }
    for name, position_rad in initial_by_name.items():
        joint = stage.GetPrimAtPath(f"{physics_root}/{name}")
        state = PhysxSchema.JointStateAPI.Get(joint, "angular")
        if not state:
            state = PhysxSchema.JointStateAPI.Apply(joint, "angular")
        state.CreatePositionAttr().Set(math.degrees(float(position_rad)))
        state.CreateVelocityAttr().Set(0.0)
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if drive:
            drive.CreateTargetPositionAttr().Set(
                math.degrees(float(position_rad))
            )
            drive.CreateTargetVelocityAttr().Set(0.0)

    cover_root = UsdGeom.Xform.Define(stage, "/FixtureCover")
    cover_xform = UsdGeom.Xformable(cover_root.GetPrim())
    cover_xform.AddTranslateOp().Set(Gf.Vec3d(*FIXTURE_COVER_ROOT_WORLD_M))
    plate = UsdGeom.Cube.Define(stage, "/FixtureCover/Plate")
    plate.CreateSizeAttr().Set(1.0)
    plate.CreateDisplayColorAttr([Gf.Vec3f(0.72, 0.55, 0.31)])
    plate_xform = UsdGeom.Xformable(plate.GetPrim())
    plate_xform.AddTranslateOp().Set(Gf.Vec3d(*lid["plate_center_local_m"]))
    plate_xform.AddScaleOp().Set(Gf.Vec3f(*lid["plate_full_extents_m"]))
    UsdPhysics.CollisionAPI.Apply(plate.GetPrim())
    handle = UsdGeom.Cube.Define(stage, "/FixtureCover/Handle")
    handle.CreateSizeAttr().Set(1.0)
    handle.CreateDisplayColorAttr([Gf.Vec3f(0.30, 0.24, 0.16)])
    handle_xform = UsdGeom.Xformable(handle.GetPrim())
    handle_xform.AddTranslateOp().Set(Gf.Vec3d(*lid["handle_center_local_m"]))
    handle_xform.AddScaleOp().Set(Gf.Vec3f(*lid["handle_full_extents_m"]))
    UsdPhysics.CollisionAPI.Apply(handle.GetPrim())
    body = UsdPhysics.RigidBodyAPI.Apply(cover_root.GetPrim())
    body.CreateRigidBodyEnabledAttr().Set(True)
    body.CreateKinematicEnabledAttr().Set(False)
    mass_kg = float(lid["mass_kg"])
    center_of_mass, inertia = _composite_cover_mass_properties(
        mass_kg=mass_kg,
        plate_full_extents_m=tuple(lid["plate_full_extents_m"]),
        plate_center_local_m=tuple(lid["plate_center_local_m"]),
        handle_full_extents_m=tuple(lid["handle_full_extents_m"]),
        handle_center_local_m=tuple(lid["handle_center_local_m"]),
    )
    mass_api = UsdPhysics.MassAPI.Apply(cover_root.GetPrim())
    mass_api.CreateMassAttr().Set(mass_kg)
    mass_api.CreateCenterOfMassAttr().Set(Gf.Vec3f(*center_of_mass))
    mass_api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*inertia))
    physx_body = PhysxSchema.PhysxRigidBodyAPI.Apply(cover_root.GetPrim())
    physx_body.CreateDisableGravityAttr().Set(False)
    physx_body.CreateLinearDampingAttr().Set(0.15)
    physx_body.CreateAngularDampingAttr().Set(0.80)
    PhysxSchema.PhysxContactReportAPI.Apply(
        cover_root.GetPrim()
    ).CreateThresholdAttr().Set(0.0)

    # Isolate fingertip/handle behavior from the lid sliding away on the flat
    # support.  This passive prismatic guide locks XY and rotation while
    # leaving physical Z translation free for the measured micro-lift.  It is
    # fixture instrumentation, not part of the final covered-basket scene.
    vertical_guide = UsdPhysics.PrismaticJoint.Define(
        stage, "/FixtureVerticalGuide"
    )
    vertical_guide.CreateBody1Rel().SetTargets([cover_root.GetPath()])
    vertical_guide.CreateAxisAttr().Set("Z")
    vertical_guide.CreateLocalPos0Attr().Set(
        Gf.Vec3f(*FIXTURE_COVER_ROOT_WORLD_M)
    )
    vertical_guide.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    vertical_guide.CreateLocalRot0Attr().Set(
        Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    )
    vertical_guide.CreateLocalRot1Attr().Set(
        Gf.Quatf(1.0, 0.0, 0.0, 0.0)
    )
    vertical_guide.CreateLowerLimitAttr().Set(-0.005)
    vertical_guide.CreateUpperLimitAttr().Set(0.050)

    plate_center_z = float(
        FIXTURE_COVER_ROOT_WORLD_M[2] + lid["plate_center_local_m"][2]
    )
    plate_bottom_z = plate_center_z - 0.5 * float(
        lid["plate_full_extents_m"][2]
    )
    support_height = 0.020
    support = UsdGeom.Cube.Define(stage, "/FixtureSupport")
    support.CreateSizeAttr().Set(1.0)
    support.CreateDisplayColorAttr([Gf.Vec3f(0.20, 0.30, 0.38)])
    support_xform = UsdGeom.Xformable(support.GetPrim())
    support_xform.AddTranslateOp().Set(
        Gf.Vec3d(
            float(FIXTURE_COVER_ROOT_WORLD_M[0]),
            float(FIXTURE_COVER_ROOT_WORLD_M[1]),
            plate_bottom_z - 0.5 * support_height - 0.0005,
        )
    )
    support_xform.AddScaleOp().Set(
        Gf.Vec3f(
            0.40,
            0.38,
            support_height,
        )
    )
    UsdPhysics.CollisionAPI.Apply(support.GetPrim())

    key_light = UsdLux.DistantLight.Define(stage, "/FixtureKeyLight")
    key_light.CreateIntensityAttr().Set(3000.0)
    key_light.CreateAngleAttr().Set(1.0)
    key_light_xform = UsdGeom.Xformable(key_light.GetPrim())
    key_light_xform.AddRotateXYZOp().Set(Gf.Vec3f(35.0, -25.0, -35.0))

    rg6_base = next(
        (
            prim
            for prim in stage.Traverse()
            if prim.GetName() == "rg6_onrobot_rg6_base_link"
        ),
        None,
    )
    if rg6_base is None:
        raise RuntimeError("RG6 base link is missing")
    chain = str(rg6_base.GetPath())
    left_link = f"{chain}/rg6_left_outer_knuckle/rg6_left_inner_finger"
    right_link = f"{chain}/rg6_right_outer_knuckle/rg6_right_inner_finger"
    fingertip_roots = {
        "left": f"{left_link}/inner_finger_1",
        "right": f"{right_link}/inner_finger_1",
    }
    for path in fingertip_roots.values():
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Fingertip instance is missing: {path}")
        prim.SetInstanceable(False)
    left_collisions = {
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(f"{left_link}/")
        and prim.HasAPI(UsdPhysics.CollisionAPI)
    }
    right_collisions = {
        str(prim.GetPath())
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(f"{right_link}/")
        and prim.HasAPI(UsdPhysics.CollisionAPI)
    }
    if not left_collisions or not right_collisions:
        raise RuntimeError("Actual fingertip collision descendants are missing")

    material = UsdShade.Material.Define(
        stage, "/PhysicsMaterials/FixtureGrip"
    )
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(float(mapping["static_friction"]))
    material_api.CreateDynamicFrictionAttr().Set(
        float(mapping["dynamic_friction"])
    )
    material_api.CreateRestitutionAttr().Set(0.0)
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    physx_material.CreateCompliantContactAccelerationSpringAttr().Set(False)
    physx_material.CreateCompliantContactStiffnessAttr().Set(
        float(mapping["compliant_contact_stiffness_n_m"])
    )
    physx_material.CreateCompliantContactDampingAttr().Set(
        float(mapping["compliant_contact_damping_n_s_m"])
    )
    for path in {
        str(handle.GetPath()),
        str(plate.GetPath()),
        *left_collisions,
        *right_collisions,
    }:
        add_physics_material_to_prim(
            stage, stage.GetPrimAtPath(path), material.GetPath()
        )

    config = load_observation_config(ROOT)
    config["overview_camera"]["resolution"] = [1280, 720]
    config["overview_camera"]["position_world_m"] = [1.35, -1.35, 0.95]
    config["overview_camera"]["look_at_world_m"] = [0.68, -0.22, 0.24]
    config["overview_camera"]["focal_length_mm"] = 48.0
    camera = create_fixed_overview_camera(stage, config)
    render_product = rep.create.render_product(str(camera.GetPath()), (1280, 720))
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach(render_product)

    contacts = {
        "events": {"left": 0, "right": 0},
        "active_pairs": {"left": set(), "right": set()},
        "latest_force_n": {"left": 0.0, "right": 0.0},
        "maximum_force_n": {"left": 0.0, "right": 0.0},
        "maximum_penetration_m": {"left": 0.0, "right": 0.0},
        "contact_position_world_m": {"left": None, "right": None},
        "contact_normal_world": {"left": None, "right": None},
        "unexpected_robot_pairs": [],
    }
    handle_path = str(handle.GetPath())

    def contact_side(pair: set[str]) -> str | None:
        if handle_path not in pair:
            return None
        if pair.intersection(left_collisions):
            return "left"
        if pair.intersection(right_collisions):
            return "right"
        return None

    def on_contact(headers, contact_data) -> None:
        for header in headers:
            pair = {
                str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
            }
            side = contact_side(pair)
            if side is None:
                robot_paths = [path for path in pair if path.startswith(robot_path)]
                if robot_paths and pair.intersection(
                    {str(handle.GetPath()), str(plate.GetPath())}
                ):
                    sample = sorted(pair)
                    if sample not in contacts["unexpected_robot_pairs"]:
                        contacts["unexpected_robot_pairs"].append(sample)
                continue
            key = tuple(sorted(pair))
            if header.type == ContactEventType.CONTACT_LOST:
                contacts["active_pairs"][side].discard(key)
                continue
            if header.type not in (
                ContactEventType.CONTACT_FOUND,
                ContactEventType.CONTACT_PERSIST,
            ):
                continue
            contacts["active_pairs"][side].add(key)
            contacts["events"][side] += 1
            total_impulse = np.zeros(3, dtype=np.float64)
            maximum_penetration = 0.0
            strongest = -1.0
            position = None
            normal = None
            for index in range(
                header.contact_data_offset,
                header.contact_data_offset + header.num_contact_data,
            ):
                impulse = np.asarray(
                    contact_data[index].impulse, dtype=np.float64
                )
                total_impulse += impulse
                norm = float(np.linalg.norm(impulse))
                if norm > strongest:
                    strongest = norm
                    position = np.asarray(
                        contact_data[index].position, dtype=np.float64
                    )
                    normal = np.asarray(
                        contact_data[index].normal, dtype=np.float64
                    )
                maximum_penetration = max(
                    maximum_penetration,
                    max(0.0, -float(contact_data[index].separation)),
                )
            force_n = float(np.linalg.norm(total_impulse) / PHYSICS_DT_SECONDS)
            contacts["latest_force_n"][side] = force_n
            contacts["maximum_force_n"][side] = max(
                contacts["maximum_force_n"][side], force_n
            )
            contacts["maximum_penetration_m"][side] = max(
                contacts["maximum_penetration_m"][side], maximum_penetration
            )
            contacts["contact_position_world_m"][side] = (
                position.tolist() if position is not None else None
            )
            contacts["contact_normal_world"][side] = (
                normal.tolist() if normal is not None else None
            )

    contact_subscription = (
        get_physx_simulation_interface().subscribe_contact_report_events(
            on_contact
        )
    )

    motion_generation_path = Path(
        get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
    )
    ur10e_config = (
        motion_generation_path
        / "motion_policy_configs"
        / "universal_robots"
        / "ur10e"
    )
    kinematics = LulaKinematicsSolver(
        robot_description_path=str(
            ur10e_config / "rmpflow" / "ur10e_robot_description.yaml"
        ),
        urdf_path=str(ur10e_config / "ur10e.urdf"),
    )
    kinematics.set_robot_base_pose(
        np.zeros(3, dtype=np.float64),
        np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
    )
    frame = next(
        name
        for name in ("flange", "ee_link", "tool0", "wrist_3_link")
        if name in kinematics.get_all_frame_names()
    )
    # The closure pose came from the already executed full episode.  Derive
    # its exact Lula frame pose rather than reconstructing the wrist
    # orientation from a nominal yaw.  The fixture must change only world Z by
    # 1 cm; otherwise an orientation mismatch can make Lula choose a distant
    # but mathematically valid elbow/wrist branch before contact is tested.
    fixture_frame_position, fixture_frame_rotation = (
        kinematics.compute_forward_kinematics(
            frame, joint_positions=FIXTURE_GRASP_ARM_RAD
        )
    )
    micro_position = np.asarray(fixture_frame_position, dtype=np.float64) + np.asarray(
        [0.0, 0.0, float(acceptance["requested_micro_lift_m"])],
        dtype=np.float64,
    )
    fixture_orientation = rot_matrix_to_quat(
        np.asarray(fixture_frame_rotation, dtype=np.float64)
    )
    micro_joints, micro_success = kinematics.compute_inverse_kinematics(
        frame,
        micro_position,
        fixture_orientation,
        warm_start=FIXTURE_GRASP_ARM_RAD,
        position_tolerance=0.003,
        orientation_tolerance=0.03,
    )
    micro_joints = np.asarray(micro_joints, dtype=np.float64)
    micro_joints = FIXTURE_GRASP_ARM_RAD + np.arctan2(
        np.sin(micro_joints - FIXTURE_GRASP_ARM_RAD),
        np.cos(micro_joints - FIXTURE_GRASP_ARM_RAD),
    )
    maximum_micro_joint_step = float(
        np.max(np.abs(micro_joints - FIXTURE_GRASP_ARM_RAD))
    )
    if not micro_success or maximum_micro_joint_step > 0.45:
        raise RuntimeError(
            "Fixture micro-lift IK is not continuous: "
            f"success={micro_success}, step={maximum_micro_joint_step}"
        )

    SimulationManager.setup_simulation(dt=PHYSICS_DT_SECONDS)
    robot = Articulation(robot_path)
    app.update()
    app_utils.play()
    app.update()
    dof_names = list(robot.dof_names)
    arm_indices = [dof_names.index(name) for name in ARM_NAMES]
    master_index = dof_names.index(MASTER_NAME)
    rg6_indices = [dof_names.index(name) for name in RG6_NAMES]
    full_initial = np.asarray(
        [initial_by_name[name] for name in dof_names], dtype=np.float32
    )
    robot.set_dof_positions(full_initial)
    robot.set_dof_velocities(np.zeros(len(dof_names), dtype=np.float32))
    robot.set_dof_position_targets(
        FIXTURE_GRASP_ARM_RAD.tolist(), dof_indices=arm_indices
    )

    def set_grip_targets(master_rad: float) -> None:
        if args.coupling_mode == "coordinated_drives":
            measured_values = robot.get_dof_positions().numpy()
            if measured_values.ndim > 1:
                measured_values = measured_values[0]
            measured_master = float(measured_values[master_index])
            follower_master_target = measured_master + (
                COORDINATED_FOLLOWER_REQUEST_BLEND
                * (master_rad - measured_master)
            )
            targets = {
                MASTER_NAME: master_rad,
                **{
                    name: ratio * follower_master_target
                    for name, ratio in FOLLOWER_RATIOS.items()
                },
            }
            robot.set_dof_position_targets(
                [targets[name] for name in RG6_NAMES],
                dof_indices=rg6_indices,
            )
        else:
            robot.set_dof_position_targets(
                [master_rad], dof_indices=[master_index]
            )

    set_grip_targets(open_master)

    from PIL import Image, ImageDraw, ImageFont

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22
        )
    except OSError:
        font = ImageFont.load_default()
    frame_root = output_root / "frames"
    frame_root.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []

    def cover_relative_to_gripper() -> np.ndarray:
        cover_world = normalized_world_matrix(cover_root.GetPrim(), omni.usd)
        gripper_world = normalized_world_matrix(rg6_base, omni.usd)
        return cover_world @ np.linalg.inv(gripper_world)

    def cover_world_position() -> np.ndarray:
        return np.asarray(
            omni.usd.get_world_transform_matrix(
                cover_root.GetPrim()
            ).ExtractTranslation(),
            dtype=np.float64,
        )

    def gripper_world_position() -> np.ndarray:
        return np.asarray(
            omni.usd.get_world_transform_matrix(rg6_base).ExtractTranslation(),
            dtype=np.float64,
        )

    def capture(label: str) -> None:
        rep.orchestrator.step(rt_subframes=2)
        rgba = np.asarray(rgb_annotator.get_data()).copy()
        image = Image.fromarray(rgba[:, :, :3].astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((15, 15, 930, 92), radius=7, fill=(0, 0, 0))
        draw.text((25, 22), label, fill=(255, 255, 255), font=font)
        draw.text(
            (25, 56),
            (
                f"force L/R={contacts['latest_force_n']['left']:.1f}/"
                f"{contacts['latest_force_n']['right']:.1f} N  "
                f"cover z={cover_world_position()[2]:.4f} m"
            ),
            fill=(200, 225, 240),
            font=font,
        )
        path = frame_root / f"frame_{len(frames):04d}.png"
        image.save(path)
        frames.append(path)

    def bilateral_force_gate_ready() -> bool:
        bilateral = all(
            contacts["active_pairs"][side] for side in ("left", "right")
        )
        individual_ready = all(
            contacts["latest_force_n"][side]
            >= float(mapping["minimum_force_per_finger_n"])
            for side in ("left", "right")
        )
        combined_ready = sum(contacts["latest_force_n"].values()) >= float(
            mapping["minimum_combined_force_n"]
        )
        return bilateral and individual_ready and combined_ready

    for step in range(180):
        app.update()
        if step in (0, 60, 120, 179):
            capture("fixture settle / RG6 open")

    close_limit = 0.60
    force_target_increment = 0.00035
    controlled_target = open_master
    contact_enabled = False
    force_ready = False
    closure_steps = 0
    for step in range(1, 1801):
        closure_steps = step
        scheduled = open_master + (close_limit - open_master) * min(
            step / 1500.0, 1.0
        )
        # PhysX speculative pairs appeared around master=0.398 rad even though
        # the measured collision surfaces still had a 5.53 mm gap.  Begin the
        # bounded force ramp only at the independently measured 30 mm handle
        # contact envelope, not on collision-pair presence alone.
        if scheduled >= geometric_contact_master_rad and not contact_enabled:
            contact_enabled = True
            controlled_target = scheduled
        if contact_enabled:
            controlled_target = min(
                close_limit, controlled_target + force_target_increment
            )
            current_torque_nm = min(
                maximum_torque_nm, current_torque_nm + 0.05
            )
            apply_total_grip_torque_budget(current_torque_nm)
            requested = controlled_target
        else:
            requested = scheduled
        set_grip_targets(requested)
        app.update()
        if step % 90 == 0:
            capture("fixture bilateral force ramp")
        if any(
            contacts["maximum_force_n"][side]
            > float(acceptance["maximum_contact_force_per_finger_n"])
            for side in ("left", "right")
        ) or any(
            contacts["maximum_penetration_m"][side]
            > float(acceptance["maximum_penetration_m"])
            for side in ("left", "right")
        ):
            break
        if bilateral_force_gate_ready():
            force_ready = True
            break

    capture("fixture pre-lift force gate")
    force_settle_steps = 0
    for step in range(1, 601):
        force_settle_steps = step
        set_grip_targets(controlled_target)
        app.update()
        # In headless mode the contact-report stream can lag the physics
        # update until the render/replicator pipeline is synchronized.  The
        # final verification capture used to reveal valid bilateral forces
        # only after this loop had already declared the gate failed.  Keep
        # observation synchronization inside the bounded settling window so
        # the decision uses the same current contact state that is saved in
        # the result artifact.
        if step % 60 == 0:
            capture("fixture force-gate synchronization")
        # Contact force can converge after the scheduled closure reaches its
        # limit. Re-evaluate every physics step rather than once immediately
        # before a render update.
        if bilateral_force_gate_ready():
            force_ready = True
            break

    pre_lift_position = cover_world_position()
    pre_lift_gripper_position = gripper_world_position()
    reference_relative = cover_relative_to_gripper()
    maximum_relative_translation = 0.0
    maximum_relative_rotation = 0.0
    maximum_arm_error = 0.0
    contact_gap_steps = 0
    maximum_contact_gap_steps = 0
    micro_lift_executed = False
    lift_convergence_steps = 0
    lift_stop_reason = "force_gate_not_ready"
    if force_ready:
        micro_lift_executed = True
        lift_stop_reason = None
        for step in range(1, 181):
            alpha = step / 180.0
            smooth = alpha**3 * (10.0 + alpha * (-15.0 + 6.0 * alpha))
            target_arm = FIXTURE_GRASP_ARM_RAD + smooth * (
                micro_joints - FIXTURE_GRASP_ARM_RAD
            )
            robot.set_dof_position_targets(
                target_arm.tolist(), dof_indices=arm_indices
            )
            set_grip_targets(controlled_target)
            app.update()
            measured = robot.get_dof_positions().numpy()
            measured = measured[0] if measured.ndim > 1 else measured
            maximum_arm_error = max(
                maximum_arm_error,
                float(
                    np.max(
                        np.abs(
                            measured[np.asarray(arm_indices)] - target_arm
                        )
                    )
                ),
            )
            relative = cover_relative_to_gripper()
            maximum_relative_translation = max(
                maximum_relative_translation,
                float(np.linalg.norm(relative[3, :3] - reference_relative[3, :3])),
            )
            maximum_relative_rotation = max(
                maximum_relative_rotation,
                rotation_angle_rad(
                    relative[:3, :3] @ reference_relative[:3, :3].T
                ),
            )
            bilateral = all(
                contacts["active_pairs"][side] for side in ("left", "right")
            )
            contact_gap_steps = 0 if bilateral else contact_gap_steps + 1
            maximum_contact_gap_steps = max(
                maximum_contact_gap_steps, contact_gap_steps
            )
            if step % 30 == 0:
                capture("fixture actual UR10e 1 cm micro-lift")
            if (
                maximum_relative_translation
                > float(acceptance["maximum_relative_translation_m"])
                or maximum_contact_gap_steps > 3
            ):
                lift_stop_reason = "slip_or_contact_gap_safety_stop"
                break
            if any(
                contacts["maximum_force_n"][side]
                > float(acceptance["maximum_contact_force_per_finger_n"])
                for side in ("left", "right")
            ) or any(
                contacts["maximum_penetration_m"][side]
                > float(acceptance["maximum_penetration_m"])
                for side in ("left", "right")
            ):
                lift_stop_reason = "force_or_penetration_safety_stop"
                break

        # Position drives can lag the smooth command under load.  Hold the
        # exact IK target while continuing every safety measurement, and stop
        # as soon as both the gripper and the cover reach their tolerances.
        if lift_stop_reason is None:
            for step in range(1, 301):
                lift_convergence_steps = step
                robot.set_dof_position_targets(
                    micro_joints.tolist(), dof_indices=arm_indices
                )
                set_grip_targets(controlled_target)
                app.update()
                measured = robot.get_dof_positions().numpy()
                measured = measured[0] if measured.ndim > 1 else measured
                maximum_arm_error = max(
                    maximum_arm_error,
                    float(
                        np.max(
                            np.abs(
                                measured[np.asarray(arm_indices)]
                                - micro_joints
                            )
                        )
                    ),
                )
                relative = cover_relative_to_gripper()
                maximum_relative_translation = max(
                    maximum_relative_translation,
                    float(
                        np.linalg.norm(
                            relative[3, :3] - reference_relative[3, :3]
                        )
                    ),
                )
                maximum_relative_rotation = max(
                    maximum_relative_rotation,
                    rotation_angle_rad(
                        relative[:3, :3] @ reference_relative[:3, :3].T
                    ),
                )
                bilateral = all(
                    contacts["active_pairs"][side]
                    for side in ("left", "right")
                )
                contact_gap_steps = 0 if bilateral else contact_gap_steps + 1
                maximum_contact_gap_steps = max(
                    maximum_contact_gap_steps, contact_gap_steps
                )
                if step % 60 == 0:
                    capture("fixture loaded target convergence")
                if (
                    maximum_relative_translation
                    > float(acceptance["maximum_relative_translation_m"])
                    or maximum_contact_gap_steps > 3
                ):
                    lift_stop_reason = "slip_or_contact_gap_safety_stop"
                    break
                if any(
                    contacts["maximum_force_n"][side]
                    > float(acceptance["maximum_contact_force_per_finger_n"])
                    for side in ("left", "right")
                ) or any(
                    contacts["maximum_penetration_m"][side]
                    > float(acceptance["maximum_penetration_m"])
                    for side in ("left", "right")
                ):
                    lift_stop_reason = "force_or_penetration_safety_stop"
                    break
                gripper_lift_now = float(
                    gripper_world_position()[2]
                    - pre_lift_gripper_position[2]
                )
                cover_lift_now = float(
                    cover_world_position()[2] - pre_lift_position[2]
                )
                if (
                    gripper_lift_now >= 0.0095
                    and cover_lift_now
                    >= float(acceptance["minimum_measured_lift_m"])
                ):
                    lift_stop_reason = "target_tolerance_reached"
                    break
            if lift_stop_reason is None:
                lift_stop_reason = "loaded_target_convergence_timeout"

    capture("fixture final verification")
    final_position = cover_world_position()
    final_gripper_position = gripper_world_position()
    final_dofs = robot.get_dof_positions().numpy()
    final_dofs = final_dofs[0] if final_dofs.ndim > 1 else final_dofs
    measured_master_rad = float(final_dofs[master_index])
    measured_rg6_rad = {
        name: float(final_dofs[dof_names.index(name)]) for name in RG6_NAMES
    }
    requested_rg6_rad = {
        MASTER_NAME: controlled_target,
        **{
            name: ratio * controlled_target
            for name, ratio in FOLLOWER_RATIOS.items()
        },
    }
    rg6_tracking_error_rad = {
        name: abs(measured_rg6_rad[name] - requested_rg6_rad[name])
        for name in RG6_NAMES
    }
    rg6_coupling_error_rad = {
        name: abs(
            measured_rg6_rad[name]
            - FOLLOWER_RATIOS[name] * measured_master_rad
        )
        for name in FOLLOWER_RATIOS
    }
    measured_lift = float(final_position[2] - pre_lift_position[2])
    measured_gripper_lift = float(
        final_gripper_position[2] - pre_lift_gripper_position[2]
    )
    force_within_limit = all(
        contacts["maximum_force_n"][side]
        <= float(acceptance["maximum_contact_force_per_finger_n"])
        for side in ("left", "right")
    )
    penetration_within_limit = all(
        contacts["maximum_penetration_m"][side]
        <= float(acceptance["maximum_penetration_m"])
        for side in ("left", "right")
    )
    success = (
        force_ready
        and micro_lift_executed
        and measured_lift >= float(acceptance["minimum_measured_lift_m"])
        and maximum_relative_translation
        <= float(acceptance["maximum_relative_translation_m"])
        and maximum_contact_gap_steps <= 3
        and force_within_limit
        and penetration_within_limit
        and not contacts["unexpected_robot_pairs"]
    )
    video = build_frame_sequence_video(
        frames,
        output_root / "rg6_handle_contact_fixture.mp4",
        fps=6,
        crf=17,
        preset="slow",
        purpose="isolated_rg6_handle_contact_development_fixture",
    )
    serializable_contacts = {
        **{
            key: value
            for key, value in contacts.items()
            if key != "active_pairs"
        },
        "active_pair_count": {
            side: len(values)
            for side, values in contacts["active_pairs"].items()
        },
    }
    result = {
        "schema_version": "rg6-handle-contact-fixture-v1",
        "status": "completed" if success else "failed",
        "purpose": "isolate_actual_rg6_fingertip_to_handle_contact_geometry",
        "asset": str(asset),
        "fixture": {
            "basket_walls_present": False,
            "support_present": True,
            "passive_vertical_guide_present": True,
            "vertical_guide_free_axis": "world_z",
            "vertical_guide_limit_m": [-0.005, 0.050],
            "measured_handle_alignment_offset_world_m": (
                FIXTURE_HANDLE_ALIGNMENT_OFFSET_WORLD_M.tolist()
            ),
            "handle_alignment_source": (
                "outputs/rg6_jaw_width_calibration/run001/"
                "jaw_width_calibration.json"
            ),
            "cover_mass_kg": mass_kg,
            "plate_full_extents_m": lid["plate_full_extents_m"],
            "handle_full_extents_m": lid["handle_full_extents_m"],
            "actual_imported_rg6_collision_used": True,
            "object_attachment_used": False,
            "target_pose_copying_used": False,
        },
        "closure": {
            "steps": closure_steps,
            "force_ready": force_ready,
            "force_settle_steps": force_settle_steps,
            "force_gate_uses_latest_active_contact": True,
            "controlled_target_rad": controlled_target,
            "torque_limit_nm": current_torque_nm,
            "force_control_trigger": "measured_geometric_contact_envelope",
            "geometric_contact_master_rad": geometric_contact_master_rad,
            "geometric_contact_source": mapping[
                "geometric_handle_contact_source"
            ],
            "measured_master_joint_rad": measured_master_rad,
            "measured_follower_joints_rad": {
                name: measured_rg6_rad[name]
                for name in FOLLOWER_RATIOS
            },
            "requested_rg6_joints_rad": requested_rg6_rad,
            "tracking_error_rad": rg6_tracking_error_rad,
            "maximum_tracking_error_rad": max(
                rg6_tracking_error_rad.values()
            ),
            "coupling_error_rad": rg6_coupling_error_rad,
            "maximum_coupling_error_rad": max(
                rg6_coupling_error_rad.values()
            ),
        },
        "coupling": {
            "mode": args.coupling_mode,
            "mimic_api_removal_in_memory": mimic_api_removal,
            "follower_drive_removal_in_memory": follower_drive_removal,
            "configured_maximum_drive_torque_nm": (
                configured_maximum_torque_nm
            ),
            "provisional_aggregate_drive_effort_nm": current_torque_nm,
            "provisional_aggregate_drive_effort_limit_nm": (
                maximum_torque_nm
            ),
            "provisional_per_drive_effort_nm": (
                current_torque_nm / len(rg6_drives)
            ),
            "drive_effort_interpretation": (
                "development_joint_drive_effort_not_rg6_motor_torque"
            ),
            "follower_target_basis": (
                "quarter_measured_three_quarter_requested_master_each_physics_step"
            ),
            "asset_files_modified": False,
            "transfer_ready": False,
        },
        "micro_lift": {
            "executed": micro_lift_executed,
            "requested_m": float(acceptance["requested_micro_lift_m"]),
            "measured_lift_m": measured_lift,
            "measured_gripper_lift_m": measured_gripper_lift,
            "loaded_target_convergence_steps": lift_convergence_steps,
            "stop_reason": lift_stop_reason,
            "maximum_relative_translation_m": maximum_relative_translation,
            "maximum_relative_rotation_rad": maximum_relative_rotation,
            "maximum_contact_gap_steps": maximum_contact_gap_steps,
            "maximum_arm_error_rad": maximum_arm_error,
            "continuous_ik_maximum_joint_step_rad": maximum_micro_joint_step,
            "fixture_frame_start_position_world_m": np.asarray(
                fixture_frame_position, dtype=np.float64
            ).tolist(),
            "fixture_frame_orientation_wxyz": np.asarray(
                fixture_orientation, dtype=np.float64
            ).tolist(),
        },
        "contacts": serializable_contacts,
        "gates": {
            "force_within_limit": force_within_limit,
            "penetration_within_limit": penetration_within_limit,
            "unexpected_robot_contact_free": not contacts["unexpected_robot_pairs"],
            "passed": success,
        },
        "runtime_seconds": time.perf_counter() - started,
        "gpu_policy": {
            "physical_gpu": physical_gpu,
            "renderer_active_gpu": physical_gpu,
            "physics_cuda_device": 0,
            "multi_gpu": False,
        },
        "calibration_status": calibration["calibration_status"],
        "training_performed": False,
        "valid_for_final_evaluation": False,
        "video": video,
    }
    result_path = output_root / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"RG6_HANDLE_FIXTURE_RESULT={result_path}", flush=True)
    print(f"RG6_HANDLE_FIXTURE_STATUS={result['status']}", flush=True)
    contact_subscription = None
    app_utils.stop()
    app.close()
    raise SystemExit(0 if success else 2)


if __name__ == "__main__":
    main()
