# Status

## 2026-07-22

- Research context and robot-platform decision updated.
- Final platform is now the manipulator used in Professor Shinkyu Park's lab; exact model is pending confirmation.
- Local GPU detected: NVIDIA GeForce RTX 4070-class GPU, driver 591.86, CUDA 13.1 reported by `nvidia-smi`.
- Git executable is not available on PATH, so the required checkpoint could not be created.
- Isaac Sim installation was subsequently located at `D:\isaac-sim` (the earlier audit checked non-hyphenated candidate paths).
- Added deterministic minimal scene: `assets/scenes/open_container_minimal.usda`.
- Added scene configuration: `configs/sim/open_container_minimal.json`.
- Added Isaac Sim launcher: `scripts/launch_isaac_sim.ps1`.
- PowerShell syntax and JSON parsing passed; the USDA file has balanced structure and required scene prim definitions.
- Isaac Sim GUI was launched successfully with `assets/scenes/open_container_minimal.usda`; the `kit` process was running and responsive after launch.
- Fixed an initial black viewport: the batch launcher had opened an empty stage, and the first custom-camera framing was invalid. The current loader opens the USD explicitly and frames the scene with the default Perspective camera.
- Explicit GPU configuration is active: renderer GPU 0, PhysX CUDA GPU 0, single-GPU mode, RTX `RaytracedLighting`, and a 60 FPS rate limit.
- Runtime verification after the fix: NVIDIA GeForce RTX 4070 SUPER at 32% utilization with approximately 3.7 GB VRAM in use; the Isaac `kit` process was responsive.
- Final visual verification showed the open-container scene rendered in the viewport rather than a black screen.
- The paper-based provisional robot stack is now integrated: the official Isaac Sim UR10e asset, an imported OnRobot RG6 URDF/USD asset, and a wrist-mounted Zivid 2 camera prim.
- The RG6 is aligned at runtime to the UR10e `ee_link`; robot, gripper, container, target, and distractor bounds were checked and are physically plausible.
- Corrected the container bottom transform, which had previously appeared as an oversized overlapping cube, and removed all placeholder sphere/cylinder geometry.
- Final viewport inspection confirms a clean tabletop scene with separated UR10e, RG6, open container, red target cube, and blue distractor cube.
- Isaac Sim remains running and responsive on the RTX 4070 SUPER. Exact Zivid intrinsics, hand-eye calibration, collision tuning, and the lab's current hardware allocation remain pending physical-lab confirmation.
- Added provisional collision-clear home, left, center, and right UR10e observation configurations in `configs/sim/observation_poses.json`.
- Added synchronized wrist-camera RGB and metric-depth capture in `scripts/observation_capture.py`; each pose writes RGB, raw `.npy` depth, a depth preview, and metadata under `outputs/observations/`.
- Verified PhysX wrist positions for left/center/right at approximately 1.263 m height, above the 0.73 m table and 0.915 m container bounds.
- Visually verified all three final RGB views and the center depth preview: target, container walls, and distractor are rendered without camera housing or table occlusion.
- Observation poses are teleported and rendered with physics paused. Collision-free continuous transitions remain a separate cuRobo/MPC implementation task.
- RG6 is temporarily treated as a kinematic visual mount; finger actuation and the real Zivid hand-eye transform remain pending lab calibration.

## 2026-07-23

- Added semantic class labels for `/World/TargetRed`, `/World/DistractorBlue`, and `/World/OpenContainer`.
- Implemented per-view instance-ID images, label mappings, 2D boxes, pixel counts, visible fractions, and object depth statistics.
- Isaac Sim 6.0 repeatedly crashed inside `rtx.syntheticdata.plugin` when either legacy or fast instance-segmentation annotators were attached; the failure was reproduced without the robot scene.
- To keep the pipeline executable, the current fixed-color scene uses a documented `rgb_color_key_fallback`; it is not claimed as native RTX ground truth.
- Tightened the color-key masks after visual inspection. Red and blue object masks align with their cubes in left/center/right views; the container mask can still include similarly colored robot highlights.
- Added offline reprocessing through `scripts/reprocess_observations.py`, so mask thresholds and object statistics can be regenerated without restarting Isaac Sim.
- Isaac Sim completed all three captures and remains responsive on GPU. Native RTX instance segmentation remains an unresolved simulator/runtime issue.
- Added `scripts/build_scene_graphs.py` and generated one `scene_graph.json` for each left/center/right observation.
- Each graph contains camera-view, target, distractor, and container nodes; per-view visibility, 2D box, visible fraction, and depth observations; and `inside`, `outside_near`, and `visible_from` edges.
- Ground-truth spatial relations are explicitly marked `source_type: ground_truth_config` and are not misrepresented as image-derived predictions.
- Added configurable provisional view scoring and selection in `configs/policy/viewpoint_selection.json` and `scripts/evaluate_viewpoints.py`.
- Current scores: left 0.5356, center 0.7024, right 1.0000. Because center is below the provisional 0.8 threshold, the policy selects the right view with an expected score gain of 0.2976.
- Saved the decision record, text log, CSV data, and SVG score chart under `outputs/viewpoint_selection/`.
- Added the provisional uncertainty-aware Scene Graph `0.2.0-draft` interface, documentation, example, and invariant validator.
- The draft defines task-conditioned target belief on nodes and spatial-relation belief distributions on edges while leaving uncertainty, calibration, task-risk, and MPC equations explicitly pending the Overleaf method definition.
- Git is installed and repository checkpoints are now available via its explicit executable path.
- Added a deterministic rule-based probability stub for testing the uncertainty-aware Scene Graph interface.
- The stub converts per-view target visible fraction into provisional target and target-to-container relation probabilities, then records `1 - max probability` as an uncalibrated uncertainty score.
- Generated and validated one `uncertainty_scene_graph_stub.json` for each left/center/right observation.
- The stub explicitly uses configured object identities and the configured `inside` relation; it is not learned perception, a calibrated probability, a final metric, or valid final-evaluation evidence.

## 2026-07-24

- Added a provisional multi-view belief-update stub that replays the left, center, and right Scene Graph observations.
- The update multiplies matching categorical probabilities and normalizes them under a temporary conditional-independence assumption.
- The fused target probability progresses from `0.654785` to `0.847653` to `0.982772`; the fused `inside` probability progresses from `0.635742` to `0.962242` to `0.999529`.
- A temporary execution gate requiring both target and `inside` probabilities to reach `0.9` is passed only after the right-view update.
- Saved the complete update trace as JSON and CSV under `outputs/belief_update/`, including input beliefs, fused beliefs, entropy changes, and gate decisions.
- Three unit tests passed for normalization, consistent-evidence accumulation, and relation-label mismatch rejection.
- This update is an interface stub using ground-truth-derived inputs. The independence assumption can make beliefs overconfident and is not approved for final evaluation.
