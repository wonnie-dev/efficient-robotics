#!/usr/bin/env python3
"""Measure the imported RG6 joint-to-fingertip collision envelope.

This is a development geometry calibration, not a manipulation success test.
It drives only the actual RG6 master joint, retains the imported Newton mimic
followers, and measures the world-space collision-mesh gap at each measured
joint value.  No object attachment, pose copying, VLM, Scene Graph, or MPC is
used.
"""

from __future__ import annotations

import argparse
import csv
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
OUTPUT_BASE = ROOT / "outputs" / "rg6_jaw_width_calibration"
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
FIXTURE_COVER_ROOT_WORLD_Z_M = -0.009
PHYSICS_DT_SECONDS = 1.0 / 120.0
PRIOR_FIXTURE_MEASURED_MASTER_RAD = 0.3981180787086487


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


def projected_surface_gap(
    left_vertices: np.ndarray,
    right_vertices: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return the inner surface gap along the measured fingertip-center axis."""
    left_center = np.mean(left_vertices, axis=0)
    right_center = np.mean(right_vertices, axis=0)
    axis = right_center - left_center
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1.0e-9:
        raise ValueError("Fingertip collision centers are coincident")
    axis /= axis_norm
    left_projection = left_vertices @ axis
    right_projection = right_vertices @ axis
    if float(np.mean(left_projection)) <= float(np.mean(right_projection)):
        gap = float(np.min(right_projection) - np.max(left_projection))
    else:
        gap = float(np.min(left_projection) - np.max(right_projection))
        axis = -axis
    return gap, axis, left_center, right_center


def contact_band_vertices(
    vertices: np.ndarray,
    lower_z_m: float,
    upper_z_m: float,
    expansion_m: float = 0.005,
) -> np.ndarray:
    mask = np.logical_and(
        vertices[:, 2] >= lower_z_m - expansion_m,
        vertices[:, 2] <= upper_z_m + expansion_m,
    )
    return vertices[mask]


def interpolate_master_for_gap(
    samples: list[dict], target_gap_m: float, gap_key: str
) -> float | None:
    usable = [
        sample
        for sample in samples
        if sample.get(gap_key) is not None
        and math.isfinite(float(sample[gap_key]))
    ]
    for first, second in zip(usable, usable[1:]):
        gap0 = float(first[gap_key])
        gap1 = float(second[gap_key])
        if (gap0 - target_gap_m) * (gap1 - target_gap_m) > 0.0:
            continue
        master0 = float(first["measured_master_rad"])
        master1 = float(second["measured_master_rad"])
        if abs(gap1 - gap0) < 1.0e-12:
            return 0.5 * (master0 + master1)
        fraction = (target_gap_m - gap0) / (gap1 - gap0)
        return master0 + fraction * (master1 - master0)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--renderer-gpu", type=int, required=True)
    parser.add_argument("--physics-gpu", type=int, default=0)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--calibration-config",
        type=Path,
        default=Path("configs/hardware/rg6_lid_development_proxy.json"),
    )
    parser.add_argument("--minimum-master-rad", type=float, default=-0.20)
    parser.add_argument("--maximum-master-rad", type=float, default=0.60)
    parser.add_argument("--sample-count", type=int, default=17)
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
    if args.sample_count < 3:
        raise ValueError("sample-count must be at least 3")
    if not args.minimum_master_rad < args.maximum_master_rad:
        raise ValueError("minimum-master-rad must be below maximum-master-rad")

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
    handle_width_m = float(lid["handle_full_extents_m"][1])
    handle_center_z_m = (
        FIXTURE_COVER_ROOT_WORLD_Z_M
        + float(lid["handle_center_local_m"][2])
    )
    handle_half_height_m = 0.5 * float(lid["handle_full_extents_m"][2])
    handle_lower_z_m = handle_center_z_m - handle_half_height_m
    handle_upper_z_m = handle_center_z_m + handle_half_height_m

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
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

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
    physics_variant = robot_root.GetVariantSets().GetVariantSet("Physics")
    if physics_variant.IsValid():
        physics_variant.SetVariantSelection("physx")
    for _ in range(15):
        app.update()

    physics_root = f"{robot_path}/Physics"
    follower_drive_removal: dict[str, bool] = {}
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

    for name in ARM_NAMES:
        drive = UsdPhysics.DriveAPI.Get(
            stage.GetPrimAtPath(f"{physics_root}/{name}"), "angular"
        )
        if drive:
            drive.CreateStiffnessAttr().Set(1000.0)
            drive.CreateDampingAttr().Set(50.0)
            drive.CreateMaxForceAttr().Set(400.0)
    master_joint = stage.GetPrimAtPath(f"{physics_root}/{MASTER_NAME}")
    master_drive = UsdPhysics.DriveAPI.Get(master_joint, "angular")
    if not master_drive:
        raise RuntimeError("RG6 master drive is missing")
    master_drive.CreateMaxForceAttr().Set(6.0)

    initial_master = float(args.minimum_master_rad)
    initial_by_name = {
        **dict(zip(ARM_NAMES, FIXTURE_GRASP_ARM_RAD, strict=True)),
        MASTER_NAME: initial_master,
        **{
            name: ratio * initial_master
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
    finger_links = {
        "left": (
            f"{chain}/rg6_left_outer_knuckle/rg6_left_inner_finger"
        ),
        "right": (
            f"{chain}/rg6_right_outer_knuckle/rg6_right_inner_finger"
        ),
    }
    collision_prims: dict[str, list] = {}
    for side, link in finger_links.items():
        instance_root = stage.GetPrimAtPath(f"{link}/inner_finger_1")
        if not instance_root.IsValid():
            raise RuntimeError(f"Missing fingertip instance: {side}")
        instance_root.SetInstanceable(False)
        collision_prims[side] = [
            prim
            for prim in stage.Traverse()
            if str(prim.GetPath()).startswith(f"{link}/")
            and prim.HasAPI(UsdPhysics.CollisionAPI)
            and prim.IsA(UsdGeom.Mesh)
        ]
        if not collision_prims[side]:
            raise RuntimeError(f"Missing fingertip collision mesh: {side}")

    SimulationManager.setup_simulation(dt=PHYSICS_DT_SECONDS)
    robot = Articulation(robot_path)
    app.update()
    app_utils.play()
    app.update()
    dof_names = list(robot.dof_names)
    arm_indices = [dof_names.index(name) for name in ARM_NAMES]
    master_index = dof_names.index(MASTER_NAME)
    full_initial = np.asarray(
        [initial_by_name[name] for name in dof_names], dtype=np.float32
    )
    robot.set_dof_positions(full_initial)
    robot.set_dof_velocities(np.zeros(len(dof_names), dtype=np.float32))
    robot.set_dof_position_targets(
        FIXTURE_GRASP_ARM_RAD.tolist(), dof_indices=arm_indices
    )
    robot.set_dof_position_targets(
        [initial_master], dof_indices=[master_index]
    )
    for _ in range(120):
        app.update()

    def measured_dofs() -> np.ndarray:
        values = robot.get_dof_positions().numpy()
        return values[0] if values.ndim > 1 else values

    def world_vertices(side: str) -> np.ndarray:
        points_world: list[np.ndarray] = []
        for prim in collision_prims[side]:
            points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
            matrix = omni.usd.get_world_transform_matrix(prim)
            points_world.extend(
                np.asarray(matrix.Transform(Gf.Vec3d(*point)), dtype=np.float64)
                for point in points
            )
        return np.asarray(points_world, dtype=np.float64)

    samples: list[dict] = []
    previous_requested = initial_master
    requested_values = np.linspace(
        args.minimum_master_rad,
        args.maximum_master_rad,
        args.sample_count,
    )
    for requested_master in requested_values:
        for target in np.linspace(previous_requested, requested_master, 61)[1:]:
            robot.set_dof_position_targets(
                [float(target)], dof_indices=[master_index]
            )
            app.update()
        for _ in range(90):
            app.update()
        measured = measured_dofs()
        measured_master = float(measured[master_index])
        follower_errors = {
            name: abs(
                float(measured[dof_names.index(name)])
                - ratio * measured_master
            )
            for name, ratio in FOLLOWER_RATIOS.items()
        }
        left = world_vertices("left")
        right = world_vertices("right")
        full_gap, axis, left_center, right_center = projected_surface_gap(
            left, right
        )
        left_band = contact_band_vertices(
            left, handle_lower_z_m, handle_upper_z_m
        )
        right_band = contact_band_vertices(
            right, handle_lower_z_m, handle_upper_z_m
        )
        band_gap = None
        if len(left_band) >= 3 and len(right_band) >= 3:
            band_gap, _, _, _ = projected_surface_gap(left_band, right_band)
        samples.append(
            {
                "sample_index": len(samples),
                "requested_master_rad": float(requested_master),
                "measured_master_rad": measured_master,
                "maximum_mimic_error_rad": max(follower_errors.values()),
                "mimic_errors_rad": follower_errors,
                "left_collision_center_world_m": left_center.tolist(),
                "right_collision_center_world_m": right_center.tolist(),
                "pinch_axis_world": axis.tolist(),
                "center_distance_m": float(
                    np.linalg.norm(right_center - left_center)
                ),
                "full_collision_surface_gap_m": full_gap,
                "handle_height_band_surface_gap_m": band_gap,
                "handle_width_m": handle_width_m,
                "full_gap_compression_margin_m": handle_width_m - full_gap,
                "band_gap_compression_margin_m": (
                    handle_width_m - band_gap
                    if band_gap is not None
                    else None
                ),
                "left_band_vertex_count": int(len(left_band)),
                "right_band_vertex_count": int(len(right_band)),
            }
        )
        previous_requested = float(requested_master)

    full_contact_master = interpolate_master_for_gap(
        samples, handle_width_m, "full_collision_surface_gap_m"
    )
    band_contact_master = interpolate_master_for_gap(
        samples, handle_width_m, "handle_height_band_surface_gap_m"
    )
    measured_sequence = np.asarray(
        [sample["measured_master_rad"] for sample in samples], dtype=np.float64
    )
    gap_sequence = np.asarray(
        [sample["full_collision_surface_gap_m"] for sample in samples],
        dtype=np.float64,
    )
    gap_differences = np.diff(gap_sequence)
    monotonically_closing = bool(np.all(gap_differences <= 0.0005))
    nearest_prior = min(
        samples,
        key=lambda sample: abs(
            float(sample["measured_master_rad"])
            - PRIOR_FIXTURE_MEASURED_MASTER_RAD
        ),
    )
    result = {
        "schema_version": "rg6-jaw-width-calibration-v1",
        "status": "completed",
        "purpose": "measure_imported_rg6_joint_to_collision_envelope",
        "asset": str(asset),
        "actual_imported_rg6_collision_used": True,
        "object_attachment_used": False,
        "target_pose_copying_used": False,
        "sweep": {
            "requested_minimum_master_rad": float(args.minimum_master_rad),
            "requested_maximum_master_rad": float(args.maximum_master_rad),
            "sample_count": int(args.sample_count),
            "measured_minimum_master_rad": float(np.min(measured_sequence)),
            "measured_maximum_master_rad": float(np.max(measured_sequence)),
            "maximum_mimic_error_rad": max(
                sample["maximum_mimic_error_rad"] for sample in samples
            ),
            "surface_gap_monotonically_closing": monotonically_closing,
            "maximum_positive_gap_step_m": float(
                max(0.0, np.max(gap_differences))
            ),
        },
        "procedural_handle": {
            "width_m": handle_width_m,
            "lower_z_m": handle_lower_z_m,
            "upper_z_m": handle_upper_z_m,
            "estimated_full_mesh_contact_master_rad": full_contact_master,
            "estimated_height_band_contact_master_rad": band_contact_master,
        },
        "prior_fixture_comparison": {
            "artifact": (
                "outputs/rg6_handle_contact_fixture/run005/result.json"
            ),
            "measured_master_rad": PRIOR_FIXTURE_MEASURED_MASTER_RAD,
            "nearest_sweep_sample": nearest_prior,
        },
        "samples": samples,
        "runtime_seconds": time.perf_counter() - started,
        "gpu_policy": {
            "physical_gpu": physical_gpu,
            "renderer_active_gpu": physical_gpu,
            "physics_cuda_device": 0,
            "multi_gpu": False,
        },
        "training_performed": False,
        "calibration_type": "simulation_geometry_development_calibration",
        "calibration_performed": True,
        "testing_performed": False,
        "transfer_ready": False,
        "valid_for_final_evaluation": False,
    }
    result_path = output_root / "jaw_width_calibration.json"
    result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    csv_path = output_root / "jaw_width_calibration.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "sample_index",
            "requested_master_rad",
            "measured_master_rad",
            "maximum_mimic_error_rad",
            "center_distance_m",
            "full_collision_surface_gap_m",
            "handle_height_band_surface_gap_m",
            "handle_width_m",
            "full_gap_compression_margin_m",
            "band_gap_compression_margin_m",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            writer.writerow({name: sample.get(name) for name in fieldnames})

    print(f"RG6_JAW_CALIBRATION_RESULT={result_path}", flush=True)
    print(f"RG6_JAW_CALIBRATION_CSV={csv_path}", flush=True)
    app_utils.stop()
    app.close()


if __name__ == "__main__":
    main()
