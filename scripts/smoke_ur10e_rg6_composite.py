"""Smoke-test the single-articulation UR10e+RG6 Isaac Sim asset."""

from __future__ import annotations

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
OUTPUT = ROOT / "outputs" / "ur10e_rg6_physics" / "composite_smoke.json"

if os.environ.get("CUDA_VISIBLE_DEVICES") != "5":
    raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 5")

asset = Path(
    json.loads(IMPORT_RESULT.read_text(encoding="utf-8"))["output_usd"]
).resolve()
app = SimulationApp(
    {
        "headless": True,
        "active_gpu": 5,
        "physics_gpu": 0,
        "multi_gpu": False,
        "max_gpu_count": 1,
        "extra_args": ["--/renderer/multiGpu/autoEnable=false"],
        "fast_shutdown": True,
    }
)

import omni.usd
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Usd, UsdGeom, UsdPhysics


context = omni.usd.get_context()
if not context.open_stage(str(asset)):
    raise RuntimeError(f"Could not open {asset}")
for _ in range(30):
    app.update()
stage = context.get_stage()
root = stage.GetDefaultPrim()
variant = root.GetVariantSets().GetVariantSet("Physics")
if variant.IsValid():
    variant.SetVariantSelection("physx")
for _ in range(10):
    app.update()

articulation_roots = [
    str(prim.GetPath())
    for prim in stage.Traverse()
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
]

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

# The importer applies conservative RG6 gains to all joints. Increase only the
# six arm drives so the heavy UR10e links hold their commanded configuration.
for prim in stage.Traverse():
    if prim.GetName() not in arm_names:
        continue
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if drive:
        drive.CreateStiffnessAttr().Set(1000.0)
        drive.CreateDampingAttr().Set(50.0)
        drive.CreateMaxForceAttr().Set(400.0)

SimulationManager.setup_simulation(dt=1.0 / 120.0)
robot = Articulation(str(root.GetPath()))
app.update()
app_utils.play()
app.update()

dof_names = list(robot.dof_names)
expected = set(arm_names + rg6_names)
if len(dof_names) != 12 or set(dof_names) != expected:
    raise RuntimeError(f"Unexpected composite DOFs: {dof_names}")

home_by_name = {
    "shoulder_pan_joint": -1.5708,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": 0.0,
}


def targets(master: float) -> np.ndarray:
    by_name = {
        **home_by_name,
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


sequence = []
stable = True
previous_master = -0.45
for label, master in (("open", -0.45), ("close", 0.45), ("reopen", -0.45)):
    requested = targets(master)
    if not sequence:
        robot.set_dof_positions(requested)
    # Avoid an unrealistic 0.9-rad target discontinuity across the RG6
    # four-bar linkage. Physical close/open commands are ramped as well.
    ramp = np.linspace(previous_master, master, 181)
    for intermediate_master in ramp[1:]:
        robot.set_dof_position_targets(targets(float(intermediate_master)))
        app.update()
    robot.set_dof_position_targets(requested)
    for _ in range(120):
        app.update()
    actual = measured()
    errors = np.abs(actual - requested)
    finite = bool(np.all(np.isfinite(actual)))
    sane = bool(np.all(np.abs(actual) <= 2.0 * math.pi))
    arm_error = max(float(errors[dof_names.index(name)]) for name in arm_names)
    rg6_error = max(float(errors[dof_names.index(name)]) for name in rg6_names)
    step_stable = finite and sane and arm_error <= 0.08 and rg6_error <= 0.12
    stable = stable and step_stable
    sequence.append(
        {
            "label": label,
            "requested_master_rad": master,
            "measured_rad": {
                name: float(value)
                for name, value in zip(dof_names, actual)
            },
            "max_arm_error_rad": arm_error,
            "max_rg6_error_rad": rg6_error,
            "finite": finite,
            "within_sanity_bound_2pi_rad": sane,
            "stable": step_stable,
        }
    )
    previous_master = master

rg6_base = next(
    (
        prim
        for prim in stage.Traverse()
        if prim.GetName() == "rg6_onrobot_rg6_base_link"
    ),
    None,
)
if rg6_base is None or not rg6_base.IsValid():
    raise RuntimeError("Merged RG6 base frame is missing")


def prim_position(prim) -> list[float]:
    return list(
        map(
            float,
            omni.usd.get_world_transform_matrix(prim).ExtractTranslation(),
        )
    )


def prim_pose(prim) -> dict:
    matrix = omni.usd.get_world_transform_matrix(prim)
    quat = matrix.ExtractRotation().GetQuat()
    imag = quat.GetImaginary()
    return {
        "position_world_m": list(map(float, matrix.ExtractTranslation())),
        "orientation_wxyz": [
            float(quat.GetReal()),
            float(imag[0]),
            float(imag[1]),
            float(imag[2]),
        ],
    }


def prim_world_aabb(prim) -> dict:
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    return {
        "minimum_world_m": list(map(float, box.GetMin())),
        "maximum_world_m": list(map(float, box.GetMax())),
    }


arm_pose_probes = []
home_open = targets(-0.45)
previous_arm_request = home_open
for joint_name in arm_names:
    index = dof_names.index(joint_name)
    for delta in (-0.20, 0.20):
        requested = home_open.copy()
        requested[index] += delta
        for alpha in np.linspace(0.0, 1.0, 121)[1:]:
            intermediate = (
                previous_arm_request
                + alpha * (requested - previous_arm_request)
            )
            robot.set_dof_position_targets(intermediate)
            app.update()
        for _ in range(60):
            app.update()
        arm_pose_probes.append(
            {
                "joint": joint_name,
                "delta_rad": delta,
                "rg6_base_position_world_m": prim_position(rg6_base),
                "finite": bool(np.all(np.isfinite(measured()))),
            }
        )
        previous_arm_request = requested
for alpha in np.linspace(0.0, 1.0, 121)[1:]:
    intermediate = previous_arm_request + alpha * (
        home_open - previous_arm_request
    )
    robot.set_dof_position_targets(intermediate)
    app.update()
for _ in range(60):
    app.update()
home_rg6_base_position = prim_position(rg6_base)
left_finger = stage.GetPrimAtPath(
    f"{rg6_base.GetPath()}/"
    "rg6_left_outer_knuckle/rg6_left_inner_finger"
)
right_finger = stage.GetPrimAtPath(
    f"{rg6_base.GetPath()}/"
    "rg6_right_outer_knuckle/rg6_right_inner_finger"
)
if not left_finger.IsValid() or not right_finger.IsValid():
    raise RuntimeError("Merged RG6 fingertip frames are missing")
home_rg6_frame_poses = {
    "base": prim_pose(rg6_base),
    "left_inner_finger": {
        **prim_pose(left_finger),
        "aabb": prim_world_aabb(left_finger),
    },
    "right_inner_finger": {
        **prim_pose(right_finger),
        "aabb": prim_world_aabb(right_finger),
    },
}
for intermediate_master in np.linspace(-0.45, 0.45, 181)[1:]:
    robot.set_dof_position_targets(targets(float(intermediate_master)))
    app.update()
for _ in range(120):
    app.update()
closed_rg6_frame_poses = {
    "base": prim_pose(rg6_base),
    "left_inner_finger": {
        **prim_pose(left_finger),
        "aabb": prim_world_aabb(left_finger),
    },
    "right_inner_finger": {
        **prim_pose(right_finger),
        "aabb": prim_world_aabb(right_finger),
    },
}

result = {
    "schema_version": "ur10e-rg6-composite-smoke-v1",
    "status": "completed" if stable else "failed",
    "asset": str(asset),
    "articulation_root": str(root.GetPath()),
    "articulation_root_prims": articulation_roots,
    "single_articulation_root": len(articulation_roots) == 1,
    "dof_count": len(dof_names),
    "dof_names": dof_names,
    "sequence": sequence,
    "home_rg6_base_position_world_m": home_rg6_base_position,
    "home_rg6_frame_poses": home_rg6_frame_poses,
    "closed_rg6_frame_poses": closed_rg6_frame_poses,
    "arm_pose_probes": arm_pose_probes,
    "stable": stable,
    "gpu_policy": {
        "physical_gpu": 5,
        "renderer_active_gpu": 5,
        "physics_cuda_device": 0,
        "multi_gpu": False,
    },
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(f"UR10E_RG6_COMPOSITE_SMOKE={OUTPUT}", flush=True)
app_utils.stop()
app.close()
raise SystemExit(0 if stable and len(articulation_roots) == 1 else 2)
