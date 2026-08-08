"""Import the UR10e+RG6 composite URDF as one Isaac Sim articulation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT / "assets" / "robots" / "ur10e_rg6" / "ur10e_rg6.urdf"
parser = argparse.ArgumentParser()
parser.add_argument(
    "--floating-base",
    action="store_true",
    help="Create a separately relocatable asset without the importer fix-base joint.",
)
parser.add_argument(
    "--scene-mounted",
    action="store_true",
    help="Bake the benchmark RobotSystem mount into a fixed-base URDF.",
)
args = parser.parse_args()
if args.floating_base and args.scene_mounted:
    raise ValueError("Choose only one of --floating-base or --scene-mounted")
import_urdf_path = URDF_PATH
if args.scene_mounted:
    import_urdf_path = (
        ROOT
        / "assets"
        / "robots"
        / "ur10e_rg6"
        / "ur10e_rg6_scene_mounted.urdf"
    )
    tree = ET.parse(URDF_PATH)
    robot = tree.getroot()
    scene_mount = ET.SubElement(robot, "link", {"name": "scene_mount"})
    inertial = ET.SubElement(scene_mount, "inertial")
    ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
    ET.SubElement(inertial, "mass", {"value": "1.0"})
    ET.SubElement(
        inertial,
        "inertia",
        {
            "ixx": "0.001",
            "ixy": "0",
            "ixz": "0",
            "iyy": "0.001",
            "iyz": "0",
            "izz": "0.001",
        },
    )
    mount = ET.SubElement(
        robot,
        "joint",
        {"name": "scene_mount_to_base_link", "type": "fixed"},
    )
    ET.SubElement(mount, "parent", {"link": "scene_mount"})
    ET.SubElement(mount, "child", {"link": "base_link"})
    ET.SubElement(
        mount,
        "origin",
        {"xyz": "-0.20 0.32 0.76", "rpy": "0 0 0"},
    )
    ET.indent(robot, space="  ")
    tree.write(import_urdf_path, encoding="utf-8", xml_declaration=True)
OUTPUT_ROOT = (
    ROOT
    / "assets"
    / "robots"
    / "ur10e_rg6"
    / (
        "isaac6_import_scene_mounted"
        if args.scene_mounted
        else ("isaac6_import_floating" if args.floating_base else "isaac6_import")
    )
)

if os.environ.get("CUDA_VISIBLE_DEVICES") != "5":
    raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 5")
if not URDF_PATH.is_file():
    raise FileNotFoundError(
        f"Build the composite first: {URDF_PATH}"
    )

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

from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig


OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
config = URDFImporterConfig(
    urdf_path=str(import_urdf_path.resolve()),
    usd_path=str(OUTPUT_ROOT.resolve()),
    # Preserve the flange and RG6 base frames.  Merging them into wrist_3
    # changes the imported four-bar frame convention and makes the two distal
    # RG6 pads move asymmetrically.
    merge_fixed_joints=False,
    merge_mesh=False,
    collision_from_visuals=False,
    collision_type="Convex Hull",
    allow_self_collision=False,
    fix_base=not args.floating_base,
    joint_drive_type="force",
    joint_target_type="position",
    override_joint_stiffness=50.0,
    override_joint_damping=2.0,
    run_asset_transformer=True,
    run_multi_physics_conversion=True,
)
importer = URDFImporter(config)
output_path = Path(importer.import_urdf()).resolve()
result = {
    "schema_version": "ur10e-rg6-composite-import-v1",
    "status": "completed",
    "source_urdf": str(import_urdf_path.resolve()),
    "output_usd": str(output_path),
    "output_root": str(OUTPUT_ROOT.resolve()),
    "single_articulation_expected_dofs": 12,
    "merge_fixed_joints": False,
    "fix_base": not args.floating_base,
    "relocatable_scene_asset": args.floating_base,
    "scene_mount_baked": args.scene_mounted,
    "scene_mount_position_world_m": (
        [-0.20, 0.32, 0.76] if args.scene_mounted else None
    ),
    "mount_parent": "flange",
    "mount_child": "rg6_onrobot_rg6_base_link",
    "mount_xyz_m": [0.0, 0.0, 0.0],
    "mount_rpy_rad": [0.0, 0.0, 0.0],
    "gpu_policy": {
        "physical_gpu": 5,
        "renderer_active_gpu": 5,
        "physics_cuda_device": 0,
        "multi_gpu": False,
    },
}
(OUTPUT_ROOT / "import_result.json").write_text(
    json.dumps(result, indent=2) + "\n",
    encoding="utf-8",
)
print(f"UR10E_RG6_COMPOSITE_IMPORT={output_path}", flush=True)
app.close()
