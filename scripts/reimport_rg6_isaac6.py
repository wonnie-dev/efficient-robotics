"""Reimport the project RG6 URDF with the installed Isaac Sim 6 importer."""

from __future__ import annotations

import json
import os
from pathlib import Path

from isaacsim import SimulationApp


ROOT = Path(__file__).resolve().parents[1]
URDF_PATH = ROOT / "assets" / "robots" / "onrobot_rg6" / "onrobot_rg6.urdf"
OUTPUT_ROOT = (
    ROOT / "assets" / "robots" / "onrobot_rg6" / "isaac6_import"
)

if os.environ.get("CUDA_VISIBLE_DEVICES") != "5":
    raise RuntimeError("CUDA_VISIBLE_DEVICES must be exactly 5")

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
    urdf_path=str(URDF_PATH.resolve()),
    usd_path=str(OUTPUT_ROOT.resolve()),
    merge_fixed_joints=False,
    merge_mesh=False,
    collision_from_visuals=False,
    collision_type="Convex Hull",
    allow_self_collision=False,
    fix_base=True,
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
    "schema_version": "rg6-isaac6-reimport-v1",
    "status": "completed",
    "source_urdf": str(URDF_PATH.resolve()),
    "output_usd": str(output_path),
    "output_root": str(OUTPUT_ROOT.resolve()),
    "fix_base": True,
    "joint_drive_type": "force",
    "joint_target_type": "position",
    "stiffness": 50.0,
    "damping": 2.0,
    "gpu_policy": {
        "physical_gpu": 5,
        "physics_cuda_device": 0,
        "multi_gpu": False,
    },
}
(OUTPUT_ROOT / "import_result.json").write_text(
    json.dumps(result, indent=2) + "\n", encoding="utf-8"
)
print(f"RG6_ISAAC6_IMPORT={output_path}", flush=True)
app.close()
