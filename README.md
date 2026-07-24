# efficient-robotics

ICRA 2027 project for task-conditioned target and spatial-relation uncertainty in a dynamic Scene Graph, with information-seeking MPC for language-guided object retrieval.

## Current minimal scene

The first deterministic scene contains a table, an open container, one target, one distractor, and the fixed project embodiment: Universal Robots UR10e, OnRobot RG6, and a wrist-mounted Zivid 2 3D/RGB-D camera.

Temporary proxy assets are allowed only for isolated debugging. Every proxy must be documented and replaced by the fixed embodiment before final evaluation.

To launch after Isaac Sim is installed:

```powershell
.\scripts\launch_isaac_sim.ps1 -IsaacSimRoot "C:\path\to\isaac-sim"
```

The loader applies provisional home/left/center/right joint poses and captures synchronized RGB and metric depth automatically. Captures are written to `outputs/observations/<pose>/` as `rgb.png`, `depth_m.npy`, `depth_preview.png`, and `metadata.json`.

Each pose also contains `instance_ids.npy`, `instance_segmentation.png`, `instance_labels.json`, and `objects.json`. The current Isaac Sim 6.0 installation crashes in the native RTX instance-segmentation plugin, so these masks use a documented fixed-scene RGB color-key fallback. Rebuild them without relaunching Isaac Sim with:

```powershell
D:\isaac-sim\python.bat .\scripts\reprocess_observations.py
```

- Scene: `assets/scenes/open_container_minimal.usda`
- Observation configuration: `configs/sim/observation_poses.json`
- Capture implementation: `scripts/observation_capture.py`

The pose values and virtual Zivid optical offset are engineering placeholders, not calibrated lab values. Continuous collision-free transitions are intentionally deferred to the cuRobo/MPC stage.
