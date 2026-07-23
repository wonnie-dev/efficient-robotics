# Benchmark Environment

## Purpose

`open_container_benchmark.usda` is the paper-facing visual and scenario
prototype. The original minimal scene remains the deterministic debugging
environment.

The benchmark scene is designed around the research question rather than
decorative realism: a center observation should leave the red target partially
occluded, while a right-side active observation should reveal more of it.

## Scene inventory

- UR10e, provisional OnRobot RG6, and wrist-mounted Zivid 2 camera
- laboratory floor and walls
- workbench, apron, legs, and dark work mat
- blue-green open container with a defined rim
- red cube target inside the container
- orange cylindrical occluder in front of the target
- yellow cylindrical distractor inside the container
- blue cube and green sphere outside the container
- purple object near the container boundary
- red cylindrical target distractor behind the container
- background storage box and spare can

## Profiles and commands

Minimal debugging scene:

```powershell
.\scripts\launch_isaac_sim.ps1 -IsaacSimRoot D:\isaac-sim -SceneProfile minimal
```

Benchmark visual prototype:

```powershell
.\scripts\launch_isaac_sim.ps1 -IsaacSimRoot D:\isaac-sim -SceneProfile benchmark
```

Benchmark captures are written separately under
`outputs/benchmark_observations/{left,center,right}`.

## Verified visual behavior

- Center: the orange cylinder heavily occludes the red target.
- Right: more of the red target becomes visible.
- Left: the target remains small and partially hidden.
- Inside, outside-near, near-boundary, behind-container, and occluding objects
  are visually represented in one controlled scene.

## Current limitations

- This is a visual/scenario prototype, not a finalized evaluation environment.
- Native Isaac Sim instance segmentation still crashes in the current runtime.
- The RGB color-key fallback cannot distinguish the red target from the similar
  red candidate and can confuse the orange occluder with the target.
- Benchmark Active View execution is therefore disabled until a correct
  multi-object mask/instance pipeline and expanded Scene Graph are available.
- Additional objects are not yet included in the uncertainty-aware graph.
- Object geometry uses simulator primitives rather than finalized lab props.
- Controlled randomization, physics properties, grasps, and real-lab dimension
  matching remain pending.

## Acceptance criteria before final experiments

1. Every task object has a correct, visually verified instance mask.
2. The graph contains all target candidates, distractors, occluders, and their
   task-relevant relations.
3. Center-to-selected-view execution uses only causally available observations.
4. Object positions, lighting, clutter, and occlusion are generated from saved
   seeded scenario configurations.
5. Collision geometry includes the confirmed gripper and camera assembly.
6. The scene dimensions and robot stack are reconciled with Professor Park's
   physical setup.
