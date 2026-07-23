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

## Provisional uncertainty and active-view interface

Each left/center/right capture now produces `uncertainty_scene_graph_stub.json`.
All eight task entities receive existence and task-conditioned target beliefs,
and configured spatial edges receive relation-versus-unknown beliefs. The
one-step controller record is written to
`outputs/benchmark_active_view_controller/decision.json`; its selected motion
request is written separately as `action_request.json`.

These values exercise the full Scene Graph-to-controller interface only. They
are deterministic, uncalibrated, ground-truth-derived stubs and must later be
replaced by VLM/grounding outputs, calibration, an online observation model,
and the approved task-risk-aware MPC formulation.

## Current limitations

- This is a visual/scenario prototype, not a finalized evaluation environment.
- Native Isaac Sim instance segmentation still crashes in the current runtime.
- A benchmark-specific simulator fallback now renders a temporary emissive
  unique-color ID pass with all non-task geometry black, restores the visible
  materials, and converts the pass to eight instance IDs.
- The ID pass and masks were visually verified, but this remains a custom
  simulator-only fallback rather than native RTX instance ground truth.
- A deterministic benchmark Scene Graph includes all eight task entities and
  configured relations. The uncertainty-aware graph still needs to be expanded.
- Benchmark Active View execution remains disabled until the expanded
  uncertainty graph consumes these instances.
- Object geometry uses simulator primitives rather than finalized lab props.
- Controlled randomization, physics properties, grasps, and real-lab dimension
  matching remain pending.

## Acceptance criteria before final experiments

1. Replace or cross-check the custom ID pass with stable native instance IDs.
2. Expand the uncertainty-aware graph to contain all target candidates,
   distractors, occluders, and their task-relevant relation distributions.
3. Center-to-selected-view execution uses only causally available observations.
4. Object positions, lighting, clutter, and occlusion are generated from saved
   seeded scenario configurations.
5. Collision geometry includes the confirmed gripper and camera assembly.
6. The scene dimensions and robot stack are reconciled with Professor Park's
   physical setup.
