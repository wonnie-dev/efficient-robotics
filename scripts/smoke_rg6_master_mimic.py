"""Validate RG6 master-only actuation with follower mimic constraints.

This smoke test intentionally contains no cup or environment contacts.  It
opens the imported UR10e+RG6 asset in memory, removes angular drives from the
five RG6 follower joints for this stage only, commands the master finger joint,
and checks that every follower remains finite and obeys its mimic ratio.
"""

from __future__ import annotations

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
OUTPUT_ROOT = ROOT / "outputs" / "ur10e_rg6_physics" / "master_mimic_smoke"
PROVISIONAL_MASTER_MAX_TORQUE_NM = 0.60

if os.environ.get("CUDA_VISIBLE_DEVICES") != "5":
    raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 5")


def next_output_dir() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    indices: list[int] = []
    for path in OUTPUT_ROOT.glob("run*"):
        try:
            indices.append(int(path.name.removeprefix("run")))
        except ValueError:
            continue
    output_dir = OUTPUT_ROOT / f"run{max(indices, default=0) + 1:03d}"
    output_dir.mkdir(parents=False, exist_ok=False)
    return output_dir


output_dir = next_output_dir()
asset = Path(
    json.loads(IMPORT_RESULT.read_text(encoding="utf-8"))["output_usd"]
).resolve()
started = time.perf_counter()
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
from pxr import PhysxSchema, UsdPhysics


ARM_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
MASTER_NAME = "rg6_finger_joint"
FOLLOWER_RATIOS = {
    "rg6_left_inner_knuckle_joint": -1.0,
    "rg6_left_inner_finger_joint": 1.0,
    "rg6_right_outer_knuckle_joint": -1.0,
    "rg6_right_inner_knuckle_joint": -1.0,
    "rg6_right_inner_finger_joint": 1.0,
}
HOME_BY_NAME = {
    "shoulder_pan_joint": -1.5708,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": 0.0,
}

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

physics_root = f"{root.GetPath()}/Physics"
follower_drive_removal: dict[str, bool] = {}
for name in FOLLOWER_RATIOS:
    prim = stage.GetPrimAtPath(f"{physics_root}/{name}")
    if not prim.IsValid():
        raise RuntimeError(f"Missing RG6 follower joint: {name}")
    if "NewtonMimicAPI" not in prim.GetAppliedSchemas():
        raise RuntimeError(f"RG6 follower lacks NewtonMimicAPI: {name}")
    had_drive = bool(UsdPhysics.DriveAPI.Get(prim, "angular"))
    if had_drive:
        prim.RemoveAPI(UsdPhysics.DriveAPI, "angular")
    follower_drive_removal[name] = (
        had_drive and not bool(UsdPhysics.DriveAPI.Get(prim, "angular"))
    )

master_prim = stage.GetPrimAtPath(f"{physics_root}/{MASTER_NAME}")
master_drive = UsdPhysics.DriveAPI.Get(master_prim, "angular")
if not master_drive:
    raise RuntimeError("RG6 master angular drive is missing")
master_drive.CreateMaxForceAttr().Set(PROVISIONAL_MASTER_MAX_TORQUE_NM)

# Limit only the master actuator torque; the five follower joints remain
# passive mimic constraints.  Arm drives are strengthened so the heavy UR10e
# links hold a stationary test configuration.
for name in ARM_NAMES:
    prim = stage.GetPrimAtPath(f"{physics_root}/{name}")
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if drive:
        drive.CreateStiffnessAttr().Set(1000.0)
        drive.CreateDampingAttr().Set(50.0)
        drive.CreateMaxForceAttr().Set(400.0)

initial_master = -0.20
initial_by_name = {
    **HOME_BY_NAME,
    MASTER_NAME: initial_master,
    **{
        name: ratio * initial_master
        for name, ratio in FOLLOWER_RATIOS.items()
    },
}
for name, position_rad in initial_by_name.items():
    prim = stage.GetPrimAtPath(f"{physics_root}/{name}")
    state = PhysxSchema.JointStateAPI.Get(prim, "angular")
    if not state:
        state = PhysxSchema.JointStateAPI.Apply(prim, "angular")
    state.CreatePositionAttr().Set(math.degrees(position_rad))
    state.CreateVelocityAttr().Set(0.0)
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if drive:
        drive.CreateTargetPositionAttr().Set(math.degrees(position_rad))
        drive.CreateTargetVelocityAttr().Set(0.0)

SimulationManager.setup_simulation(dt=1.0 / 120.0)
robot = Articulation(str(root.GetPath()))
app.update()
app_utils.play()
app.update()

dof_names = list(robot.dof_names)
expected_names = set(ARM_NAMES + [MASTER_NAME] + list(FOLLOWER_RATIOS))
if len(dof_names) != 12 or set(dof_names) != expected_names:
    raise RuntimeError(f"Unexpected composite DOFs: {dof_names}")
arm_indices = [dof_names.index(name) for name in ARM_NAMES]
master_index = dof_names.index(MASTER_NAME)
robot.set_dof_position_targets(
    [HOME_BY_NAME[name] for name in ARM_NAMES],
    dof_indices=arm_indices,
)
robot.set_dof_position_targets([initial_master], dof_indices=[master_index])
for _ in range(120):
    app.update()


def measured_by_name() -> dict[str, float]:
    values = robot.get_dof_positions().numpy()
    if values.ndim > 1:
        values = values[0]
    return {
        name: float(value)
        for name, value in zip(dof_names, values, strict=True)
    }


sequence: list[dict] = []
stable = all(follower_drive_removal.values())
previous_master = initial_master
for label, requested_master in (
    ("open", -0.20),
    ("close", 0.45),
    ("reopen", -0.20),
):
    for intermediate in np.linspace(
        previous_master, requested_master, 241
    )[1:]:
        robot.set_dof_position_targets(
            [float(intermediate)],
            dof_indices=[master_index],
        )
        app.update()
    for _ in range(120):
        app.update()

    measured = measured_by_name()
    master = measured[MASTER_NAME]
    mimic_errors = {
        name: abs(measured[name] - ratio * master)
        for name, ratio in FOLLOWER_RATIOS.items()
    }
    values = np.asarray(list(measured.values()), dtype=np.float64)
    arm_error = max(
        abs(measured[name] - HOME_BY_NAME[name]) for name in ARM_NAMES
    )
    master_error = abs(master - requested_master)
    max_mimic_error = max(mimic_errors.values())
    finite = bool(np.all(np.isfinite(values)))
    within_sanity_bound = bool(np.all(np.abs(values) <= 2.0 * math.pi))
    step_stable = (
        finite
        and within_sanity_bound
        and arm_error <= 0.08
        and master_error <= 0.12
        and max_mimic_error <= 0.05
    )
    stable = stable and step_stable
    sequence.append(
        {
            "label": label,
            "requested_master_rad": requested_master,
            "measured_rad": measured,
            "master_error_rad": master_error,
            "mimic_error_rad": mimic_errors,
            "max_mimic_error_rad": max_mimic_error,
            "max_arm_error_rad": arm_error,
            "finite": finite,
            "within_sanity_bound_2pi_rad": within_sanity_bound,
            "stable": step_stable,
        }
    )
    previous_master = requested_master

result = {
    "schema_version": "rg6-master-mimic-smoke-v1",
    "status": "completed" if stable else "failed",
    "purpose": "isolated_master_only_rg6_actuation_without_contact",
    "asset": str(asset),
    "asset_files_modified": False,
    "follower_drive_removal_in_memory": follower_drive_removal,
    "master_drive_present": True,
    "provisional_master_max_torque_nm": (
        PROVISIONAL_MASTER_MAX_TORQUE_NM
    ),
    "dof_names": dof_names,
    "sequence": sequence,
    "stable": stable,
    "runtime_seconds": time.perf_counter() - started,
    "gpu_policy": {
        "physical_gpu": 5,
        "renderer_active_gpu": 5,
        "physics_cuda_device": 0,
        "multi_gpu": False,
    },
}
result_path = output_dir / "result.json"
result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(f"RG6_MASTER_MIMIC_SMOKE={result_path}", flush=True)
print(f"RG6_MASTER_MIMIC_STATUS={result['status']}", flush=True)
app_utils.stop()
app.close()
raise SystemExit(0 if stable else 2)
