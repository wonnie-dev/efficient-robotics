"""Smoke-test the Isaac-6-reimported RG6 physical finger articulation."""

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
    / "onrobot_rg6"
    / "isaac6_import"
    / "import_result.json"
)
ASSET = Path(
    json.loads(IMPORT_RESULT.read_text(encoding="utf-8"))["output_usd"]
).resolve()
OUTPUT = ROOT / "outputs" / "rg6_physics" / "reimported_articulation_smoke.json"

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

import omni.usd
import isaacsim.core.experimental.utils.app as app_utils
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.simulation_manager import SimulationManager


context = omni.usd.get_context()
if not context.open_stage(str(ASSET)):
    raise RuntimeError(f"Could not open {ASSET}")
for _ in range(30):
    app.update()
stage = context.get_stage()
root = stage.GetDefaultPrim()
variant = root.GetVariantSets().GetVariantSet("Physics")
if variant.IsValid():
    variant.SetVariantSelection("physx")
for _ in range(10):
    app.update()

SimulationManager.setup_simulation(dt=1.0 / 120.0)
gripper = Articulation(str(root.GetPath()))
app.update()
app_utils.play()
app.update()

dof_names = list(gripper.dof_names)
master_index = dof_names.index("finger_joint")
gripper.set_dof_positions([0.0] * len(dof_names))
gripper.set_dof_position_targets([0.0] * len(dof_names))
for _ in range(60):
    app.update()

sequence = []
stable = True
finger_links = {
    "left": stage.GetPrimAtPath(
        f"{root.GetPath()}/Geometry/onrobot_rg6_base_link/"
        "left_outer_knuckle/left_inner_finger"
    ),
    "right": stage.GetPrimAtPath(
        f"{root.GetPath()}/Geometry/onrobot_rg6_base_link/"
        "right_outer_knuckle/right_inner_finger"
    ),
}
for label, target in (
    ("open", -0.45),
    ("close", 0.45),
    ("reopen", -0.45),
):
    # The importer authors drives on every mimic joint.  Give those drives
    # targets consistent with the master so they do not oppose the mimic
    # constraint and artificially hold the hand near zero.
    target_by_name = {
        "finger_joint": target,
        "left_inner_knuckle_joint": -target,
        "right_outer_knuckle_joint": -target,
        "right_inner_knuckle_joint": -target,
        "left_inner_finger_joint": target,
        "right_inner_finger_joint": target,
    }
    gripper.set_dof_position_targets(
        [target_by_name[name] for name in dof_names]
    )
    for _ in range(240):
        app.update()
    values = gripper.get_dof_positions().numpy()
    values = values[0] if values.ndim > 1 else values
    finite = bool(np.all(np.isfinite(values)))
    within_sanity_bound = bool(np.all(np.abs(values) <= math.pi))
    stable = stable and finite and within_sanity_bound
    sequence.append(
        {
            "label": label,
            "requested_master_rad": target,
            "measured_rad": {
                name: float(value)
                for name, value in zip(dof_names, values)
            },
            "finite": finite,
            "within_sanity_bound_pi_rad": within_sanity_bound,
            "finger_link_positions_m": {
                side: list(
                    map(
                        float,
                        omni.usd.get_world_transform_matrix(prim).ExtractTranslation(),
                    )
                )
                for side, prim in finger_links.items()
            },
        }
    )

result = {
    "schema_version": "rg6-reimported-articulation-smoke-v1",
    "status": "completed" if stable else "failed",
    "asset": str(ASSET.resolve()),
    "articulation_root": str(root.GetPath()),
    "dof_names": dof_names,
    "sequence": sequence,
    "stable": stable,
    "gpu_policy": {
        "physical_gpu": 5,
        "physics_cuda_device": 0,
        "multi_gpu": False,
    },
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
print(f"RG6_REIMPORTED_SMOKE={OUTPUT}", flush=True)
app_utils.stop()
app.close()
raise SystemExit(0 if stable else 2)
