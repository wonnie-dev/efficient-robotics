#!/usr/bin/env python3
"""Validate a provisional symmetric-drive RG6 coupling without contact.

The imported passive Newton mimic representation is accurate without load but
stalls asymmetrically under unilateral contact.  This isolated development
smoke removes the mimic API in memory and commands all six real RG6 joints with
their URDF ratios.  It does not modify the asset on disk and is not a
transfer-ready real-gripper controller.
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
OUTPUT_BASE = ROOT / "outputs" / "ur10e_rg6_physics" / "coordinated_drive_smoke"
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
HOME_BY_NAME = {
    "shoulder_pan_joint": -1.5708,
    "shoulder_lift_joint": -1.5708,
    "elbow_joint": 1.5708,
    "wrist_1_joint": -1.5708,
    "wrist_2_joint": -1.5708,
    "wrist_3_joint": 0.0,
}
PROVISIONAL_TOTAL_DRIVE_TORQUE_BUDGET_NM = 6.0


def next_output_dir() -> Path:
    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    indices: list[int] = []
    for path in OUTPUT_BASE.glob("run*"):
        try:
            indices.append(int(path.name.removeprefix("run")))
        except ValueError:
            continue
    output = OUTPUT_BASE / f"run{max(indices, default=0) + 1:03d}"
    output.mkdir(parents=False, exist_ok=False)
    return output


def coordinated_targets(master_rad: float) -> dict[str, float]:
    return {
        MASTER_NAME: master_rad,
        **{
            name: ratio * master_rad
            for name, ratio in FOLLOWER_RATIOS.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--renderer-gpu", type=int, required=True)
    parser.add_argument("--physics-gpu", type=int, default=0)
    parser.add_argument("--output-root", type=Path)
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

    started = time.perf_counter()
    app = SimulationApp(
        {
            "headless": args.headless,
            "active_gpu": physical_gpu,
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

    asset = Path(
        json.loads(IMPORT_RESULT.read_text(encoding="utf-8"))["output_usd"]
    ).resolve()
    context = omni.usd.get_context()
    if not context.open_stage(str(asset)):
        raise RuntimeError(f"Could not open composite asset: {asset}")
    for _ in range(30):
        app.update()
    stage = context.get_stage()
    robot_root = stage.GetDefaultPrim()
    robot_path = str(robot_root.GetPath())
    variant = robot_root.GetVariantSets().GetVariantSet("Physics")
    if variant.IsValid():
        variant.SetVariantSelection("physx")
    for _ in range(15):
        app.update()

    physics_root = f"{robot_path}/Physics"
    mimic_removal: dict[str, bool] = {}
    follower_drive_present: dict[str, bool] = {}
    per_joint_torque_nm = (
        PROVISIONAL_TOTAL_DRIVE_TORQUE_BUDGET_NM / len(RG6_NAMES)
    )
    for name in RG6_NAMES:
        joint = stage.GetPrimAtPath(f"{physics_root}/{name}")
        if not joint.IsValid():
            raise RuntimeError(f"Missing RG6 joint: {name}")
        if name != MASTER_NAME:
            if "NewtonMimicAPI" not in joint.GetAppliedSchemas():
                raise RuntimeError(f"Follower lacks NewtonMimicAPI: {name}")
            joint.RemoveAPI("NewtonMimicAPI")
            mimic_removal[name] = (
                "NewtonMimicAPI" not in joint.GetAppliedSchemas()
            )
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if not drive:
            drive = UsdPhysics.DriveAPI.Apply(joint, "angular")
        drive.CreateStiffnessAttr().Set(20.0)
        drive.CreateDampingAttr().Set(1.0)
        drive.CreateMaxForceAttr().Set(per_joint_torque_nm)
        follower_drive_present[name] = bool(drive)
    if not all(mimic_removal.values()):
        raise RuntimeError(f"Could not remove mimic APIs: {mimic_removal}")
    if not all(follower_drive_present.values()):
        raise RuntimeError(
            f"Could not configure coordinated drives: {follower_drive_present}"
        )

    for name in ARM_NAMES:
        joint = stage.GetPrimAtPath(f"{physics_root}/{name}")
        drive = UsdPhysics.DriveAPI.Get(joint, "angular")
        if drive:
            drive.CreateStiffnessAttr().Set(1000.0)
            drive.CreateDampingAttr().Set(50.0)
            drive.CreateMaxForceAttr().Set(400.0)

    initial_master = -0.20
    initial_by_name = {**HOME_BY_NAME, **coordinated_targets(initial_master)}
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

    SimulationManager.setup_simulation(dt=1.0 / 120.0)
    robot = Articulation(robot_path)
    app.update()
    app_utils.play()
    app.update()
    dof_names = list(robot.dof_names)
    arm_indices = [dof_names.index(name) for name in ARM_NAMES]
    rg6_indices = [dof_names.index(name) for name in RG6_NAMES]
    full_initial = np.asarray(
        [initial_by_name[name] for name in dof_names], dtype=np.float32
    )
    robot.set_dof_positions(full_initial)
    robot.set_dof_velocities(np.zeros(len(dof_names), dtype=np.float32))
    robot.set_dof_position_targets(
        [HOME_BY_NAME[name] for name in ARM_NAMES], dof_indices=arm_indices
    )

    def set_rg6_targets(master_rad: float) -> None:
        targets = coordinated_targets(master_rad)
        robot.set_dof_position_targets(
            [targets[name] for name in RG6_NAMES], dof_indices=rg6_indices
        )

    def measured_by_name() -> dict[str, float]:
        values = robot.get_dof_positions().numpy()
        values = values[0] if values.ndim > 1 else values
        return {
            name: float(value)
            for name, value in zip(dof_names, values, strict=True)
        }

    set_rg6_targets(initial_master)
    for _ in range(120):
        app.update()

    sequence: list[dict] = []
    stable = True
    previous_master = initial_master
    for label, requested_master in (
        ("open", -0.20),
        ("close", 0.45),
        ("reopen", -0.20),
    ):
        for intermediate in np.linspace(
            previous_master, requested_master, 241
        )[1:]:
            set_rg6_targets(float(intermediate))
            app.update()
        for _ in range(120):
            app.update()
        measured = measured_by_name()
        requested = coordinated_targets(requested_master)
        tracking_errors = {
            name: abs(measured[name] - requested[name]) for name in RG6_NAMES
        }
        coupling_errors = {
            name: abs(
                measured[name]
                - FOLLOWER_RATIOS[name] * measured[MASTER_NAME]
            )
            for name in FOLLOWER_RATIOS
        }
        values = np.asarray(list(measured.values()), dtype=np.float64)
        arm_error = max(
            abs(measured[name] - HOME_BY_NAME[name]) for name in ARM_NAMES
        )
        maximum_tracking_error = max(tracking_errors.values())
        maximum_coupling_error = max(coupling_errors.values())
        finite = bool(np.all(np.isfinite(values)))
        within_sanity_bound = bool(np.all(np.abs(values) <= 2.0 * math.pi))
        step_stable = (
            finite
            and within_sanity_bound
            and arm_error <= 0.08
            and maximum_tracking_error <= 0.05
            and maximum_coupling_error <= 0.05
        )
        stable = stable and step_stable
        sequence.append(
            {
                "label": label,
                "requested_master_rad": requested_master,
                "measured_rad": measured,
                "tracking_error_rad": tracking_errors,
                "coupling_error_rad": coupling_errors,
                "maximum_tracking_error_rad": maximum_tracking_error,
                "maximum_coupling_error_rad": maximum_coupling_error,
                "maximum_arm_error_rad": arm_error,
                "finite": finite,
                "within_sanity_bound_2pi_rad": within_sanity_bound,
                "stable": step_stable,
            }
        )
        previous_master = requested_master

    result = {
        "schema_version": "rg6-coordinated-drive-smoke-v1",
        "status": "completed" if stable else "failed",
        "purpose": "unloaded_symmetric_rg6_development_coupling",
        "asset": str(asset),
        "asset_files_modified": False,
        "coupling_mode": "coordinated_six_joint_drives_no_mimic",
        "mimic_api_removal_in_memory": mimic_removal,
        "coordinated_drive_present": follower_drive_present,
        "provisional_total_drive_torque_budget_nm": (
            PROVISIONAL_TOTAL_DRIVE_TORQUE_BUDGET_NM
        ),
        "provisional_per_joint_max_torque_nm": per_joint_torque_nm,
        "sequence": sequence,
        "stable": stable,
        "runtime_seconds": time.perf_counter() - started,
        "gpu_policy": {
            "physical_gpu": physical_gpu,
            "renderer_active_gpu": physical_gpu,
            "physics_cuda_device": 0,
            "multi_gpu": False,
        },
        "training_performed": False,
        "transfer_ready": False,
        "valid_for_final_evaluation": False,
    }
    result_path = output_root / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"RG6_COORDINATED_DRIVE_RESULT={result_path}", flush=True)
    print(f"RG6_COORDINATED_DRIVE_STATUS={result['status']}", flush=True)
    app_utils.stop()
    app.close()


if __name__ == "__main__":
    main()
