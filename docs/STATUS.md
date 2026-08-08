# Status

## Current milestone — 2026-07-24

- Final embodiment: Universal Robots UR10e + OnRobot RG6 + wrist-mounted Zivid 2 3D/RGB-D camera.
- Authoritative specification: `efficient_robotics_final_project_plan_KO.docx`;
  `docs/FINAL_RESEARCH_SPEC.md` is the secondary implementation summary.
- Preserve the existing deterministic simulation and belief-planning prototype.
- Next major research step: calibrated perception/VLM output and validated action-conditioned posterior prediction.
- Older platform statements below that describe the robot stack as provisional or pending confirmation are preserved as implementation history and are superseded by this current milestone.

## 2026-08-05 — Contact-safe cover staging/release integration

- `outputs/rg6_handle_contact_fixture/run015/result.json` passed the isolated
  handle fixture after synchronizing contact reports during the force-settle
  loop. Bilateral forces were `21.66/22.77 N`, maximum penetrations were
  `0.436/0.459 mm`, and the requested 1 cm fixture lift retained the cover
  without attachment or pose copying.
- `outputs/live_pipeline/remove_cover_physics_smoke/seed188/run038/` completed
  the same-process UR10e+RG6 cover grasp, `0.157888 m` lift, `0.30 m` transfer,
  post-removal observation, belief update, and horizon-3 replanning to
  `grasp_inside`. That run still ended while holding the cover and did not
  physically execute the replanned grasp, so it is not a complete episode or
  final-evaluation evidence.
- Added optional supported placement to
  `scripts/persistent_composite_grasp.py`: dense same-branch Cartesian
  transfer/lowering, declared-support contact verification, shallow
  `0.5 mm` contact-seeking placement, bounded `2 mm` support penetration,
  controlled RG6 release, `0.15 m` open-gripper retreat, and post-release
  stability/contact gates. `open_minimal_scene.py` now requires
  `cover_placed_and_released` before capturing the post-removal observation.
- The workbench bounds are checked at runtime. The development staging offset
  is `[-0.45, 0.0, 0.16] m`, which keeps the cover on
  `/World/WorkBench/Top` and at least 3 cm from support edges and the basket.
  It is a simulation development pose, not a lab-calibrated real-robot pose.
- GPU-0-only revalidation attempts were fail-closed and never touched physical
  GPUs 1–5. `run039` rejected a sparse 45 cm transfer IK branch change;
  `run040` rejected an out-of-bounds positive-Y staging pose; `run042` exposed
  an over-broad contact classifier at the first full-lift waypoint; `run043`
  passed lift and transfer but stopped at supported-placement waypoint 4 when
  arm tracking error reached `0.050435 rad` against a `0.05 rad` gate.
  `run043` had zero unexpected collision pairs.
- The initial cover/basket rim contact is now allowed only while the cover root
  is within 2 cm of its initial supported pose. Once clear, all target/environment
  contacts remain fail-closed; during staging only the declared workbench
  support contact is allowed.
- Supported placement was slowed from 20 to 60 physics steps per Cartesian
  waypoint. The GPU rerun is pending because the current escalated-execution
  approval service returned `approval request failed`; a sandbox-only probe
  correctly self-terminated when its NVIDIA monitor was unavailable.
- The subsequent GPU-0-only `run044` passed bilateral handle grasp, lift, and
  all nine transfer waypoints. The slower placement also reduced waypoint-4
  final arm error from `0.050435 rad` to `0.011777 rad`, but waypoint 5 was
  stopped by the collision gate when the cover plate contacted the UR10e
  shoulder. Run045 then rejected a `+0.40 m` Y candidate before motion because
  it exceeded the transformed workbench bound. The development staging offset
  was next changed from robot-side `[-0.45, 0.0, 0.16] m` to the calculated
  open-workbench offset `[0.0, -0.40, 0.16] m`.
- `run047` passed grasp, lift, transfer, and 30 of 33 slow placement waypoints;
  final arm error remained near `0.0053 rad`. At waypoint 31, the cover touched
  the raised `/World/WorkMat` before the lower declared WorkBench support, so
  the fail-closed target/environment gate stopped the run. Placement now uses
  the actual WorkMat height, offset `[-0.42, -0.20, 0.16] m`, a `0.03 m`
  projected-center margin, at least `75%` footprint overlap, and `0.03 m`
  basket clearance. Thirty-six targeted CPU tests pass. GPU revalidation is
  pending and no final-evaluation claim is made.
- `run048` reached all 33 placement waypoints and recorded primary WorkMat
  contact with final arm error below `0.01 rad`. The overhanging plate then
  also touched the lower WorkBench top, which the single-support classifier
  rejected. The runtime now declares that exact WorkBench top as a secondary
  support only for placement/release/retreat, while retaining mandatory
  WorkMat contact and shared penetration/stability gates. All other contacts
  remain forbidden; GPU revalidation is pending.
- `run049` completed the physical primitive: bilateral grasp, `0.158707 m`
  verified lift, transfer, primary WorkMat plus secondary WorkBench support,
  controlled release, cleared finger contact, `0.15 m` retreat, and stable
  post-release pose. Maximum reported support penetration was `0.5967 mm`,
  with zero unexpected robot/environment or target/environment pairs. The
  post-removal observation gained `16,231` target pixels and horizon-3 belief
  MPC replanned to `grasp_inside` after the inside-open belief reached
  `0.982245`.
- The original run049 top-level smoke status was incorrectly written `failed`
  because its legacy gate required bilateral finger contact at the end of the
  episode, even after intentional release. The gate now checks bilateral
  contact immediately before release plus release, retreat, cleared fingers,
  and post-release stability. The raw physical result and server result were
  completed; the final target grasp was selected but not physically executed.
  This remains a provisional pilot, not final-evaluation evidence.
- The GPU watchdog now accepts `--physical-gpu 0` and sets
  `PHYSICAL_GPU=0`, `CUDA_VISIBLE_DEVICES=0`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  and `NVIDIA_VISIBLE_DEVICES=0` for its child while removing distributed
  environment variables. Thirty-six targeted CPU tests pass.
- Training remains not performed. These runs use provisional simulation
  physics and simulator-derived pilot perception; calibration/testing and
  final paper-scale evaluation remain not performed.

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
- Added a one-step rule-based Active View Controller that starts from the center observation and compares left/right candidate views.
- Candidate utility combines expected target-entropy reduction, expected relation-entropy reduction, and mean joint-motion cost using provisional configurable weights.
- The controller selected `move_to_observation_pose(right)`: right utility `0.607101` exceeded left utility `0.394346`.
- Center alone failed the temporary execution gate; the replayed center-plus-right belief passed with target probability `0.967820` and `inside` probability `0.995905`.
- Saved a complete decision record and a separate robot-action request under `outputs/active_view_controller/`.
- Six stub unit tests passed in total, including execution-gate and joint-motion-cost tests.
- No robot motion was executed. Candidate outcomes currently use offline, ground-truth-derived replay and therefore act as an oracle-style predictor that is invalid for final evaluation.
- Connected the generated `move_to_observation_pose(right)` request to an Isaac Sim execution mode.
- The execution captured a fresh center observation, rebuilt its graph/stub, ran the controller, applied the selected right joint pose, captured a fresh right RGB-D observation, and regenerated the selected-view graph/stub.
- Verified the requested and measured UR10e joint positions: maximum absolute joint error was `0.000148504 rad`, below the provisional `0.02 rad` tolerance.
- Visually checked the new right RGB capture; the red target and open-container interior are clearly rendered.
- Isaac Sim remains running and responsive on GPU at the selected right observation pose.
- Motion boundary: the current pose action directly sets joint state and position targets. It is not a time-parameterized, collision-checked continuous trajectory or MPC execution.
- The first runtime attempt completed center/right captures but stopped during postprocessing because the Isaac-only CLI argument leaked into a reused parser. Argument isolation was added, and the second run completed with a verified execution record.
- Replaced the direct Center-to-Right joint-state jump with deterministic interpolated joint-position targets.
- The verified transition used 15 waypoints at a maximum configured joint increment of `0.02 rad`, with three physics frames per waypoint.
- All interpolated waypoints and the final target passed the UR10e joint-limit checks.
- Final trajectory error was `0.000017715 rad`; the post-capture verification error was `0.000117016 rad`, both below the `0.02 rad` tolerance.
- Isaac Sim's experimental PhysX contact view failed to initialize in this GPU runtime (`AttributeError: 'NoneType' object has no attribute 'check'`).
- A conservative world-AABB overlap fallback checked moving UR10e links against the table, open container, target, and distractor on every physics frame; no overlap was detected.
- Safety boundary: AABB overlap is conservative but not equivalent to narrow-phase collision/contact checking, and the kinematic RG6 visual mount is not included in swept-volume checks.
- Isaac Sim remains running and responsive at the right observation pose after the successful interpolated transition.
- Added a separate paper-facing environment profile, `open_container_benchmark_v1`, while preserving the minimal debugging scene.
- The benchmark scene adds laboratory walls, a workbench and mat, a blue-green rimmed container, task-relevant clutter, an orange occluder, a boundary object, and a similar red candidate behind the container.
- Verified fresh left/center/right RGB-D captures under `outputs/benchmark_observations/`.
- Visual behavior is suitable for the intended scenario prototype: the center view heavily occludes the red target, while the right view reveals substantially more of it.
- Isaac Sim is currently running the benchmark scene and remains responsive on GPU.
- Benchmark limitation: the RGB color-key fallback merges/confuses similar red and orange objects. Benchmark Active View execution is disabled until correct multi-instance segmentation and the expanded graph are implemented.
- Added `docs/BENCHMARK_ENVIRONMENT.md` with profiles, commands, inventory, limitations, and final-experiment acceptance criteria.
- Implemented a benchmark-only temporary emissive color-ID render pass: all non-task geometry is black during the pass, eight task entities receive unique simulator materials, and visible materials are restored before the GUI continues.
- Added RTX tone-mapping-aware ID prototypes and connected-component separation for the orange/yellow and purple/magenta object pairs.
- Visually verified center and right instance masks against their RGB images. The red target, orange occluder, yellow distractor, rear red candidate, boundary object, blue/green distractors, and container are independently represented.
- Verified active-view evidence in the benchmark masks: target pixels increase from `45` at center to `636` at right (about `14.1x`).
- Added deterministic benchmark Scene Graph generation with 9 nodes (camera plus container and seven configured objects) and 15 relation/visibility edges per view.
- Added offline ID-pass reprocessing so classification changes do not require relaunching Isaac Sim.
- Eight unit tests pass, including benchmark relation construction and paired-instance component separation.
- Boundary: the custom color-ID pass is simulator-only and not native RTX ground truth. Benchmark Active View remains disabled until the uncertainty-aware graph is expanded to consume all benchmark entities.
- Expanded the benchmark Scene Graph with provisional beliefs for all eight task entities: existence probability/uncertainty and task-conditioned target probability/uncertainty on nodes, plus relation distributions/uncertainty on configured relation edges.
- Added a graph-level target distribution, required `inside` relation belief, and temporary task-failure risk `1 - P(target_red) * P(inside)`.
- Added a one-step information-seeking controller stub that evaluates left/right replay observations using task-risk reduction, target/relation entropy reduction, and joint-motion cost.
- From the center belief, the stub selected `right`: predicted task-risk reduction was `0.522695`, compared with `0.092036` for left; utilities were `0.749290` and `0.128904`, respectively.
- The target probability changes from `0.574113` at center to `0.854702` in the independently generated right observation graph; the target `inside` probability changes from `0.568253` to `0.931184`.
- Eleven unit tests pass. The benchmark scene loader now regenerates deterministic graphs, uncertainty graphs, and the view-selection request after capturing all three views.
- Research boundary: these are uncalibrated rule-based probabilities and offline ground-truth-derived candidate replay. The controller is an interface prototype, not the final VLM, predictor, risk metric, MPC solver, or valid final-evaluation method.
- Approved and documented the initial research design: temperature-scaled categorical target/relation beliefs, entropy-only uncertainty summaries, Bayesian filtering with negative evidence, pre-action observation prediction, and hybrid receding-horizon planning.
- Added standalone temperature scaling, categorical entropy, and Bayesian-update utilities. Temperature fitting is implemented but remains pending real model logits and a held-out calibration dataset.
- Added a non-oracle horizon-two planner that reads no future RGB, depth, masks, captured objects, or future Scene Graph files. It predicts outcome branches from a pre-action likelihood model.
- The active action subset is `viewpoint + grasp`; `occluder_move` and `uncover` are present but disabled until their planned scenario stages.
- The current plan selects `viewpoint_right -> grasp`; direct grasp, left-view-then-grasp, center-view-then-grasp, and right-view-then-grasp are all recorded with their costs.
- Eighteen unit tests pass, including temperature scaling, negative evidence, branch normalization, entropy, risk, and the explicit non-oracle provenance guard.
- Boundary: the initial belief is uncalibrated and the likelihood model is hand specified. This is a non-oracle receding-horizon engineering prototype, not yet the final learned observation predictor or MPC solver and not valid final-evaluation evidence.
- Connected the non-oracle planner to a benchmark Isaac Sim execution mode: center capture, pre-action plan, interpolated viewpoint motion, actual post-action capture, Bayesian belief update, and replanning.
- Verified the pre-action planner selected `viewpoint_right` without reading any future capture. The UR10e completed 15 interpolated waypoints with no world-AABB collision and a post-capture maximum joint error of approximately `0.000108 rad`.
- The actual right observation was consumed only after motion. It contained 624 target pixels and produced `target_red: 0.574113 -> 0.763479` and `inside: 0.568253 -> 0.863000`.
- Target entropy decreased from `0.713172` to `0.551787` nats; relation entropy decreased from `1.034558` to `0.474410` nats.
- Replanning from the updated belief and the actual right robot pose selected `viewpoint_center -> grasp`. Viewpoint motion costs are recomputed relative to the executed pose.
- Isaac Sim remains running and responsive on the RTX 4070 SUPER; observed GPU utilization was 49% with about 3996 MiB of 12282 MiB in use.
- Nineteen unit tests pass. Execution provenance explicitly records no future capture in pre-action planning and forbids an MPC claim for the current interpolated-controller prototype.
- Defined common VLM input, raw-logit output, and separate ground-truth JSON Schemas for independent model implementations.
- Added an Isaac benchmark exporter that creates full RGB inputs, anonymous object candidates, masked crops, binary masks, bounding boxes, and categorical relation queries.
- Prevented semantic label leakage: inference inputs use `object_001` style IDs and never contain simulator names such as `target_red`; ground truth is written to a separate file.
- Generated and contract-validated three development samples for left/center/right under `outputs/vlm_dataset/`.
- Added a deterministic mock-logit adapter for interface testing only and a cross-file validator that checks sample IDs, candidate order, logit dimensions, relation label order, query coverage, and forbidden semantic tokens.
- Twenty-three unit tests pass. The three current samples are not a training, calibration, or test set because they share one fixed scene and seed.
- Adopted a data strategy based on pretrained models and existing public training data, without manually constructing a large labeled VLM dataset.
- Project-specific calibration and evaluation samples will instead be generated and labeled automatically in Isaac Sim, with episode/seed-level separation to prevent view leakage.
- Added and ran a visual-only slow movement demo: `center -> left -> center -> right -> center`, with 90 paused interpolation steps and a two-second hold at each pose.
- The visual demo is explicitly not physics evidence or MPC. The first attempted slow physics demo exposed UR10e instability under an excessively fine 57-waypoint/long-settle setting; it was discarded, and the previously verified 15-waypoint physics execution remains unchanged.
- The successful visual demo is recorded at `outputs/movement_demo.json`; Isaac Sim remains responsive on GPU after completion.
- Created the separate server VLM environment at `/data/wonheekoh/venvs/efficient-robotics-vlm` with Python 3.10.12. It was initially bootstrapped without pretrained-model packages or checkpoints, and `/data/wonheekoh/isaacsim_venv` was not changed.
- Installed an isolated Qwen3-VL inference stack in the VLM environment:
  PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, Transformers 4.57.3,
  Accelerate 1.12.0, and the supporting pinned packages. The Isaac Sim
  environment was not modified.
- Downloaded `Qwen/Qwen3-VL-8B-Instruct` revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` to
  `/data/wonheekoh/models/Qwen3-VL-8B-Instruct` (17 GB).
- Added a GPU-5-only BF16/SDPA adapter that consumes the anonymous VLM input
  contract and emits target and relation raw logits. It forbids distributed
  environment variables, requires exactly one visible CUDA device, and does
  not use `device_map="auto"`, DataParallel, DDP, or NCCL execution.
- The raw score is the cyclic-letter permutation mean of single-token
  pre-softmax LM logits. This removes forced-choice letter-position bias but
  is still uncalibrated.
- Regenerated the single-seed benchmark observations headlessly with renderer
  physical GPU 5, PhysX `cuda:0`, and multi-GPU disabled, then exported three
  anonymous VLM development samples.
- The Qwen output passed `VLM_CONTRACT_VALID` for center and right. Peak
  allocated memory was 16.700 GiB; final scoring took 12.8--13.1 seconds after
  an approximately 8.3-second model load in a fresh process.
- Development-only result: center target was incorrect and relations were
  2/7; right target remained incorrect by a 1.0 raw-logit margin and relations
  were 3/7, including a correct `object_001 inside container` decision. This
  single fixed seed is an integration smoke test, not a model benchmark or
  calibration result.
- Next VLM milestone: generate a 50--100 episode pilot with seed/episode-level
  split isolation, audit class and viewpoint coverage, then finalize
  calibration/validation/IID-test/viewpoint-shift-test generation.
- Added `scripts/launch_server_headless.sh`, which fixes `CUDA_VISIBLE_DEVICES=5`, physical renderer GPU 5, PhysX CUDA device 0, and single-GPU execution.
- Added headless exit and configurable renderer/physics GPU arguments to `scripts/open_minimal_scene.py`.
- Added `scripts/build_observation_video.py`, which uses CPU `ffmpeg`/libx264 to summarize the left, center, and right RGB captures without consuming another GPU.
- Verified the minimal scene headlessly on physical GPU 5. All three RGB images are 640x480 and all depth arrays are 480x640 with finite metric depth.
- Verified output: `outputs/observations/minimal_observations.mp4` is H.264, 640x480, 10 FPS, 30 frames, and 3.0 seconds long.
- Visually inspected all three RGB captures. The open container, red target, and blue distractor are rendered, and the wrist viewpoints differ as configured.
- The run exited successfully. Physical GPU 5 returned to 1 MiB and 0% utilization with no remaining compute process.
- Twenty-five unit tests pass, including two new headless-video manifest/error-path tests.
- Known server limitation: the provisional kinematic RG6 mount produces static-joint/mimic-joint warnings, and the experimental PhysX contact monitor remains unavailable. These warnings did not prevent observation capture but remain invalid as final gripper-physics evidence.
- Added a simulation-provisional `close_high` UR10e wrist viewpoint without changing the configured camera mount, intrinsics, or hand-eye assumptions.
- Verified `center -> close_high` on physical GPU 5 with one CUDA-visible device and multi-GPU disabled. The 36-waypoint transition completed within the provisional `0.02 rad` joint tolerance; contact/collision monitoring was unavailable, so this is not real-robot safety evidence.
- The close/high observation increased visible target pixels from `44` to `2427` (about `55.2x`) and produced valid RGB-D. A 1920x1080, 12.8-second overview video is stored at `outputs/candidate_view_demo/benchmark_seed000/close_high_candidate_demo.mp4`.
- One batch-size-one pretrained Qwen3-VL inference selected the correct `object_001` target with an uncalibrated softmax value of `0.9149`, versus `0.0851` for `object_007`. Inference took `25.73 s` after a `7.66 s` model load and peaked at `16.961 GiB` on physical GPU 5.
- The same output did not satisfy the relation gate: `P(inside)=0.2836` while `P(near_boundary)=0.6005`. Grasp therefore remains blocked; target recognition improved, but relation scoring is still unresolved.
- The result and all raw outputs are cached under `outputs/pilot_cache/qwen3_vl/8b392957435e51f8aab53ca2c0b617fc8df5b1c90a561774d73ca26fb6ad8396/`. This single deterministic scene is pipeline-validation evidence only, not calibration or final evaluation.
- Added one targeted relation-input revision: an anonymous cyan overlay marks visible `container_001` material, while the prompt defines the interior relative to the visible inner walls and distinguishes an object on the floor from one touching the rim.
- The revised VLM input passes the JSON Schema and all 42 CPU tests pass. Candidate/reference overlay assets are now included in the content-addressed cache key.
- On the same close/high observation, Qwen again selected `object_001`; the uncalibrated softmax values were `P(target)=0.999972` and `P(inside)=0.999952`. The pilot threshold policy therefore selected `grasp`.
- The revised inference took `29.06 s` after an `8.07 s` model load and peaked at `17.080 GiB` on physical GPU 5. The process exited and GPU 5 returned to 1 MiB and 0% utilization.
- Grasp execution remains explicitly unauthorized because RG6 contact physics, collision monitoring, hand-eye calibration, and real-robot geometry are not validated. This one-scene improvement is not a calibrated confidence or accuracy result.
- Revised cached output: `outputs/pilot_cache/qwen3_vl/722c613e24615c7b4f5d5550ef8710e8a005933508f627e6c9ba7453e606cf9a/`.
- Added a deterministic relation-preserving benchmark generator and captured seeds 0--9 at left, center, right, and close/high wrist views. All 10 episodes produced valid RGB-D and anonymous instance assets; capture failures were 0/10.
- The seeded capture batch took `330.63 s`. Across seeds, close/high target visibility remained approximately 2378--2462 pixels, while center visibility ranged from 117--490 pixels.
- Exported and JSON-Schema-validated 40 VLM input samples under `outputs/seeded_pilot/vlm_dataset/`. No manual annotation was created.
- Ran the pretrained Qwen single-GPU pilot sequentially on physical GPU 5. Ten episodes completed, zero failed, 17 observations were consumed, and seven episodes requested close/high reobservation before replanning; three selected grasp directly from center.
- All 10 episodes ended with a `grasp` selection and the final selected target matched simulator ground truth in the post-planning debug check. This is pipeline-debug behavior, not execution success, calibrated accuracy, or a statistical guarantee.
- Mean Qwen inference time was `29.84 s/observation` (27.09--37.34 s), mean model load time was `8.21 s/observation`, maximum allocated GPU memory was `17.152 GiB`, and the VLM batch took `633.02 s`.
- Cache entries use content-addressed directories under `outputs/pilot_cache/qwen3_vl/` with request, output, metrics, stdout, and stderr records. The completed report is `outputs/seeded_pilot/pilot_report.json`.
- Training, LoRA, calibration, and final testing were not performed. Seeded planning replays pre-captured observations; actual grasp/contact physics and a live seeded action-observation loop remain pending.

## 2026-07-24 — Live loop, presentation video, and contact-grasp pilot

- Added a persistent Isaac Sim JSON-IPC server and an external single-GPU Qwen
  orchestrator. One Isaac process now performs a fresh center observation,
  executes requested UR10e viewpoint motion, captures the new RGB-D only after
  motion, and replans from the returned VLM logits.
- Verified seed 0 without pre-captured candidate replay:
  `center -> close_high -> right -> grasp`. Both viewpoint trajectories
  completed in the same Isaac process. The end-to-end runtime was `290.62 s`.
- The three Qwen calls took approximately `59.9`, `57.0`, and `64.4 s` while
  Isaac remained resident on the same physical GPU 5. Peak Qwen allocation was
  approximately `17.16 GiB`; combined use stayed within the 48 GiB A6000.
- Added content-addressed cache reuse to the live loop and retained every
  observation, VLM input, decision, action request, trajectory, and metric
  under `outputs/live_pipeline/benchmark_seed000_run001/`.
- Built the final meeting video
  `outputs/presentation_demo/full_pipeline_with_contact_grasp.mp4`: H.264,
  1920x1080, 6 FPS, 298 frames, and `49.67 s`. It combines the stored
  observation/replanning evidence with the later contact-grasp pilot.
- The imported RG6 USD exposes six physical finger DOFs, but its mimic-joint
  articulation is numerically unstable in Isaac Sim 6 and diverges beyond its
  joint limits. Failed articulation smoke diagnostics are retained at
  `outputs/rg6_physics/articulation_smoke.json`; they are not counted as a
  successful grasp.
- Added an explicitly labeled RG6-sized bilateral collision-pad physics proxy.
  The target is a non-kinematic `0.08 kg` rigid body with gravity and friction.
  No target attachment and no target-pose copying are used.
- The contact pilot completed with 71 left and 71 right PhysX contact events,
  then lifted the target by `0.180315 m`. The verified result and 20.83-second
  MP4 are under `outputs/rg6_physics/contact_grasp_seed000/`.
- `scripts/run_live_single_gpu_pipeline.py --execute-contact-grasp` now
  optionally launches that verified physics stage after a live terminal
  `grasp` decision, sequentially on GPU 5, and records the trigger/result in
  the same pipeline result. It does not run Isaac and grasp jobs in parallel.
- Forty-eight CPU unit tests pass. Python compilation and `git diff --check`
  pass. GPU 5 has no remaining project process after validation.
- Boundary: the observation/replanning loop is live, but the contact-grasp
  stage currently runs in a fresh Isaac process with an RG6-sized physics
  proxy. It is pilot evidence only, not the repaired RG6 articulation, final
  paper evaluation, calibrated success rate, or real-robot execution.
- Verified the optional combined orchestrator in
  `outputs/live_pipeline/benchmark_seed000_run002/`: three fresh live
  observations, terminal grasp, and sequential contact lift all completed in
  `358.18 s`. Qwen inference averaged `60.24 s/observation` and peaked at
  `17.16 GiB`; the grasp subprocess took `58.70 s`.
- The combined run recorded `grasp_executed: true`, 71/71 bilateral contacts,
  and the same `0.180315 m` lift. Physical GPU 5 returned to `1 MiB`, 0%
  utilization, with no remaining project process.
- Observation mode now disables the unused imported RG6 physics variant before
  using it as a visual/camera mount, and skips the unsupported tensor contact
  view. A fresh headless benchmark capture completed with no RG6 static-joint,
  mimic-joint, or invalid-contact-filter errors.
- Upgraded the Qwen cache identity to exclude session/sample IDs and filesystem
  paths. It now keys on the normalized inference payload plus ordered
  image/crop/mask SHA-256 values, model revision, prompt version, resolution
  limit, and inference policy, so identical visual inputs can be reused safely
  across new live-session directories.

## 2026-07-24 — Actual articulated RG6 contact grasp

- Located the RG6 visual and collision STL files included with the installed
  Isaac Sim 6 URDF importer, copied them into
  `assets/robots/onrobot_rg6/meshes/`, changed the URDF to repository-relative
  paths, and reimported without modifying `isaacsim_venv`.
- The new imported asset contains the actual RG6 render and convex-hull
  collision meshes. Its six finger DOFs are stable and track consistent mimic
  targets across open `-0.45 rad`, close `+0.45 rad`, and reopen.
- Added `scripts/run_rg6_actual_contact_grasp.py`. The legacy scene RG6 and the
  earlier collision-pad proxy are disabled in this run; only the reimported
  actual RG6 articulation and finger collision meshes participate.
- Verified seed 0 on physical GPU 5 only: 252 left and 252 right actual
  finger-target contact events, `0.179396 m` measured target lift, finite joint
  state, and a `0.180 m` physical mount lift.
- The target is a non-kinematic `0.08 kg` rigid body with gravity. Explicit
  target attachment and target pose copying are both false.
- The improved close/high overview video is H.264, 960x540, 8 FPS, 200 frames,
  and 25 seconds:
  `outputs/rg6_physics/actual_contact_grasp_seed000/rg6_actual_contact_grasp.mp4`.
- Rebuilt
  `outputs/presentation_demo/full_pipeline_with_contact_grasp.mp4` using the
  actual RG6 clip: H.264, 1920x1080, 6 FPS, 323 frames, and 53.834 seconds.
- `scripts/run_live_single_gpu_pipeline.py --execute-contact-grasp` now invokes
  this actual-RG6 stage sequentially after the live terminal decision. No
  distributed, parallel, or multi-GPU execution is introduced.
- This supersedes the earlier proxy as the current default, but remains
  pipeline-validation evidence only. It is not same-process UR10e/RG6 coupled
  manipulation, real-robot validation, calibration, or final paper evaluation.

## 2026-07-25 — Two-reobservation commitment-gated integration

- Added a temporary irreversible-action gate to the non-oracle receding-horizon
  prototype. Grasp is blocked while task failure risk exceeds `0.15` or fewer
  than two reobservations have completed.
- Viewpoint selection still compares action-conditioned future observation and
  posterior branches. The gate is an engineering safety constraint, not the
  belief-space MPC novelty or a confidence-only replacement for it.
- Rechecked seed 0 on physical GPU 5 only. The trace completed as
  `center -> viewpoint_right -> viewpoint_overhead -> grasp allowed`.
- Task failure risk changed from `0.7140` initially to `0.08234` after the two
  belief updates. Final target belief was `0.92295` for `target_red`; final
  `inside` belief was `0.99427`.
- The original `close_high` candidate was not reused: a fresh capture saw zero
  target pixels. The second observation instead uses an explicitly labeled
  synthetic debug overhead wrist pose and saw 2,277 target pixels.
- Twenty-five relevant CPU tests pass. The Isaac run completed in about 21
  seconds and released GPU 5 after exit.
- Boundary: post-action perception still uses simulator instance labels, the
  overhead pose is not a UR10e collision-checked trajectory, Qwen was not
  loaded in this run, and grasp was selected but not executed in the same
  scene. This is pipeline-debug evidence only.

## 2026-07-25 — Pretrained Qwen belief-planner replay

- Added `scripts/run_qwen_belief_mpc_replay.py` to replace the simulator-label
  belief adapter with pretrained Qwen3-VL-8B raw-logit outputs for the captured
  center, right, and synthetic-overhead observations.
- Added a sensing-action feasibility constraint: a viewpoint whose configured
  expected target detection probability is below `0.10` is not executable.
  This prevented the predicted-empty left view from being selected merely
  because it had lower motion cost.
- The single-GPU replay completed as
  `center -> viewpoint_right -> viewpoint_overhead -> grasp`. Simulator ground
  truth was not consumed during planning.
- Center, right, and overhead Qwen inference took `29.85`, `40.21`, and
  `42.38 s`, respectively. Peak allocated memory was `17.016 GiB`; every
  inference reported one visible CUDA device and `CUDA_VISIBLE_DEVICES=5`.
- All three content-addressed outputs are cached under
  `outputs/pilot_cache/qwen3_vl/`. Re-running the completed replay took
  `52.84 s` because center and right were cache hits and only overhead required
  fresh inference.
- Qwen selected `target_red` and `inside` in all three observations. The final
  product-fused values are numerically near one, but they are explicitly
  uncalibrated and must not be interpreted as success probabilities.
- Training, LoRA, calibration, continuous UR10e motion, and same-scene RG6
  grasp were not performed. This remains pre-captured perception-interface
  validation, not live end-to-end or final evaluation.

## 2026-07-25 — Tempered live Isaac–Qwen future-belief loop

- Replaced temperature-one product fusion in the new integration path with a
  fixed debug temperature of `4.0` and a weighted log-space observation update
  of `0.5`. These values are not fitted calibration parameters.
- Added a safe `defer` terminal action when grasp is blocked and no usable
  sensing action remains. The planner no longer raises an exception in that
  state.
- Connected the tempered Qwen belief adapter and action-conditioned
  future-belief planner to the persistent Isaac JSON-IPC loop.
- The first continuous right-view attempt failed safely because the official
  UR10e `ee_joint` state diverged to values on the order of `1e8 rad`. This
  failure is retained in `outputs/live_pipeline/benchmark_seed000_run003/`.
- The completed debug-pose run is
  `outputs/live_pipeline/benchmark_seed000_run005/pipeline_result.json`.
  It generated each observation only after the corresponding request in one
  persistent Isaac process and completed
  `center -> viewpoint_right -> viewpoint_overhead -> grasp`.
- Center initially favored the rear red candidate (`0.6194` posterior target
  belief) and had `0.6040` task-failure risk. Right reobservation changed the
  selected target to `target_red` but risk remained `0.4696`. Overhead raised
  `target_red` belief to `0.8960` and `inside` to `0.9642`, reducing debug risk
  to `0.1360`; the `0.15` commitment gate then allowed grasp.
- The live run took `260.00 s`. Qwen inference took `59.76`, `65.76`, and
  `62.83 s`; peak Qwen allocation was `17.022 GiB`. Combined observed GPU use
  was about `22.1 GiB`. Every model call saw physical GPU 5 as the only CUDA
  device, and GPU 5 returned to 7 MiB and 0% utilization after exit.
- Training, fitted calibration, and physical grasp were not performed.
  Center/right/overhead use fixed synthetic debug wrist coordinates, so this
  validates live observation/Qwen/belief/replanning timing but not continuous
  UR10e motion, collision safety, or full end-to-end manipulation.

## 2026-07-25 — Same-layout composite UR10e+RG6 grasp attempt

- Extended `scripts/run_ur10e_rg6_composite_grasp.py` so a completed live
  terminal `grasp` decision can trigger a fresh-process grasp attempt using
  the same seed-0 open-container layout and the stable 12-DOF composite
  UR10e+actual-RG6 articulation.
- The first transformed-base attempt was rejected after non-finite PhysX
  transforms. Keeping the imported fixed-base articulation at its authored
  world origin removed that failure.
- The next finite diagnostic is retained at
  `outputs/ur10e_rg6_physics/same_scene_seed000_run004/`. Lula
  returned valid grasp/lift IK, but the open RG6 was displaced by clutter
  contact: maximum arm tracking error was `0.6364 rad`, bilateral target
  contact was absent, and the pre-lift safety gate correctly prevented lift.
- A deterministic wrist-yaw rule based on simulator ground truth was tested
  only as a collision-debug aid. Directly teleporting the grasp pose still
  produced invalid PhysX transforms, so that strategy is discontinued.
- Current result is a failed manipulation integration attempt, not a completed
  end-to-end episode. The next implementation must use an above-container
  pregrasp, collision-checked descent, contact closure, and lift trajectory.
  It must also stop if measured arm error exceeds the new `0.05 rad` gate.
- The execution reconstructed the same scene configuration in a new Isaac
  process; it was not the same persistent process as the Qwen run. Arm-link
  collision geometry remained disabled, and the debug grasp yaw used simulator
  ground truth. None of these runs is valid final evaluation evidence.
- After the changes, all 58 repository CPU unit tests pass.

## 2026-07-26 — Collision-logged pregrasp execution attempt

- Extended `scripts/run_ur10e_rg6_composite_grasp.py` with phase-specific
  contact logging, finite-state and `0.05 rad` arm-error aborts, an
  above-container pregrasp, five descending IK waypoints, bilateral
  actual-finger contact gating, and result/abort JSON output.
- The pregrasp and descent IK solutions are generated successfully in the
  seed-0 layout, but physical trajectory execution is not yet validated.
- Run 006 identified the first exact failure: the fixed-base composite was
  authored at the world origin, placing the RG6 in contact with the workbench
  and work mat before the grasp.
- Runs 007--013 tested parent transforms, explicit fixed joints, a floating-base
  import, and a scene-mounted import. These attempts were retained as failure
  evidence; none produced a stable offset articulation in the benchmark scene.
- The latest attempt is
  `outputs/ur10e_rg6_physics/same_scene_seed000_run014/abort_result.json`.
  During the initial safe-home hold, unexpected RG6/workbench contact first
  appeared at simulation step 17. Maximum arm error was
  `3.1230499846376856e22 rad`; all 12 measured DOFs then became non-finite, so
  execution stopped before reaching the pregrasp.
- Therefore the ordered manipulation result is: collision localization and
  trajectory instrumentation implemented; pregrasp/descent planning
  implemented; monitored physical execution failed; RG6 closure, bilateral
  contact verification, and lift were not executed. No successful MP4 or
  end-to-end result JSON was produced for this attempt.
- The earlier clean-scene composite lift remains valid only as a standalone
  contact-physics pilot. It does not resolve the fixed-base mounting failure in
  the benchmark tabletop layout and is not substituted for this failed run.

## 2026-07-26 — Stable robot-base-frame seed-0 contact lift

- Replaced the unstable attempt to translate the imported fixed-base
  articulation. The UR10e base now remains at its stable authored origin, and
  the complete benchmark environment is expressed in UR10e base coordinates
  with the inverse rigid translation `[0.20, -0.32, -0.76] m`. This preserves
  every robot-to-scene relative pose.
- The home-pose gate passed in
  `outputs/ur10e_rg6_physics/same_scene_seed000_run015_stability/`:
  all joints remained finite, maximum arm error was `0.000857 rad` against the
  `0.05 rad` limit, and there were no unexpected robot/environment contacts.
- Run 016 then stopped safely at the pregrasp because its end-of-trajectory arm
  error was `0.088464 rad`. The target coordinate was also found to be
  overwritten by its pre-transform world coordinate. This run remains failure
  evidence.
- Corrected the dynamic target's robot-base coordinate, initialized it on the
  container bottom rather than dropping the tall cuboid, and added a
  60-physics-step final-command settle before each trajectory acceptance check.
- Seed-0 run 017 completed the complete manipulation sequence:
  safe home, above-container pregrasp, four collision-monitored descent
  segments, actual RG6 closure, bilateral-contact gate, UR10e lift, and final
  hold.
- Run 017 used one 12-DOF UR10e+RG6 articulation. It recorded 421 left and 600
  right target-contact events, no unexpected RG6/environment pair, finite
  final joints, `0.001548 rad` pregrasp arm error, `0.000663 m` pre-lift target
  displacement, and `0.176786 m` target lift.
- Result:
  `outputs/ur10e_rg6_physics/same_scene_seed000_run017/result.json`.
  Video: 960x540 H.264, 10 FPS, 526 frames, 52.6 seconds at
  `outputs/ur10e_rg6_physics/same_scene_seed000_run017/ur10e_rg6_composite_grasp.mp4`.
- Boundary: run 017 reconstructs the Qwen-selected seed-0 layout in a fresh
  Isaac process. Qwen was not rerun, the grasp yaw uses simulator ground truth,
  and non-RG6 arm-link collision geometry remains disabled. It is successful
  same-layout sequential integration evidence, not same-process end-to-end,
  full collision-safe manipulation, calibration, or final paper evaluation.

## 2026-07-26 — Whole-arm collision-enabled seed-0 lift

- Added an explicit `--enable-arm-collisions` mode that enables every imported
  UR10e collision mesh, installs contact reporting on all robot rigid links,
  treats non-finger robot/target contact as unexpected, and includes unexpected
  contact in the close/lift success gates.
- The collision-enabled home-pose smoke test passed in
  `outputs/ur10e_rg6_physics/same_scene_seed000_run018_whole_arm_stability/`:
  finite joints, `0.000857 rad` maximum arm error, and zero unexpected
  robot/environment contacts.
- The full collision-enabled run passed in
  `outputs/ur10e_rg6_physics/same_scene_seed000_run019_whole_arm_collision/`.
  It retained finite joints, passed the bilateral-contact and pre-lift gates,
  recorded 421 left and 600 right target-contact events, had zero unexpected
  arm/RG6/environment contacts, and lifted the target `0.176786 m`.
- Run 019 produced a 960x540, 10 FPS, 52.6-second H.264 video at
  `outputs/ur10e_rg6_physics/same_scene_seed000_run019_whole_arm_collision/ur10e_rg6_composite_grasp.mp4`.
- Boundary: imported collision meshes and runtime contact aborts are now
  enabled, but the joint path is still deterministic interpolation through
  ground-truth-derived IK waypoints rather than a general obstacle-aware
  cuRobo/MPC trajectory. Qwen remains a saved trigger from run 005 and
  manipulation still executes in a fresh same-layout process.

## 2026-07-26 — Persistent Qwen-to-contact-lift seed-0 integration

- Added a terminal composite executor that keeps the live observation server's
  Isaac process and USD stage alive after Qwen selects `grasp`. It re-expresses
  the existing benchmark prims in the validated UR10e-base coordinates, inserts
  the collision-enabled 12-DOF UR10e+RG6 articulation, and executes the
  prevalidated seed-0 waypoint path under the existing finite-state,
  `0.05 rad` arm-error, unexpected-contact, target-displacement, and bilateral
  actual-finger-contact gates.
- Run 006 completed `center -> right -> overhead -> defer`. Its final
  uncalibrated target and inside beliefs were `0.8308` and `0.9662`, producing
  `0.1973` debug task-failure risk, so the unchanged `0.15` irreversible-action
  gate correctly blocked grasp.
- Added the previously captured `close_high` view as a fallback sensing action
  that is enabled only after right and overhead reobservations. With the run
  006 belief, the pre-action future-belief prototype selects
  `viewpoint_close_high` rather than forcing grasp or lowering the safety gate.
- Run 007 reached the Qwen terminal grasp decision but stopped during
  `home -> pregrasp`: the newly inserted articulation's `wrist_3` stalled at
  `-1.8076 rad` against the `-3.1412 rad` target, producing `1.3337 rad`
  tracking error. The other five arm joints tracked normally. The safe airborne
  home configuration now authors `wrist_3` at the validated pregrasp roll
  before the first physics frame, avoiding that large same-stage roll while
  retaining all approach safety checks.
- The no-Qwen same-stage smoke validation passed at
  `outputs/live_pipeline/persistent_composite_grasp_smoke/run002/`.
  It recorded 103 left and 96 right target-contact events, zero unexpected
  environment contacts, finite final joints, and `0.176858 m` measured lift.
- The complete run passed at
  `outputs/live_pipeline/benchmark_seed000_run008/pipeline_result.json`:
  `center -> viewpoint_right -> viewpoint_overhead ->
  viewpoint_close_high -> grasp`, followed by the actual UR10e+RG6 contact
  lift in the same persistent Isaac process and stage.
- Run 008's final uncalibrated beliefs were `0.9702` for `target_red` and
  `0.9891` for `inside`; debug task-failure risk fell to `0.04043`. The grasp
  then recorded 103/96 bilateral contact events, zero unexpected contacts,
  `0.02681 rad` pre-lift arm error, `0.00846 m` pre-lift target displacement,
  finite final joints, and `0.176858 m` lift. No attachment or target pose
  copying was used.
- Run 008 took `550.31 s`. Fresh sequential Qwen load-plus-inference times were
  `47.33`, `53.70`, `53.81`, and `64.15 s`; peak Qwen allocation was
  `17.115 GiB`. Observed simultaneous Isaac+Qwen usage remained about `22 GiB`
  on physical GPU 5 only, and GPU 5 returned to `1 MiB`, 0% utilization after
  completion.
- Content-addressed Qwen artifacts are stored under
  `outputs/pilot_cache/qwen3_vl/<sha256>/` as `request.json`, `output.json`,
  `metrics.json`, and stdout/stderr logs. Run 008 had 0/4 cache hits because
  the newly rendered inputs produced new content hashes.
- Video:
  `outputs/live_pipeline/benchmark_seed000_run008/persistent_grasp/persistent_composite_grasp.mp4`
  (1920x1080 H.264, 10 FPS, 176 encoded frames, 17.6 seconds by
  `ffprobe`; the frame-sequence manifest records 175 source frames).
- Boundary: this is one deterministic seed-0 integration pilot, not final
  paper evidence. Qwen is pretrained inference-only and its scores are not
  calibrated. Observation viewpoints remain fixed debug wrist coordinates,
  the grasp path uses simulator-ground-truth-derived seed-0 IK, and the current
  action-conditioned future-belief planner explicitly reports that it is not
  yet a general MPC solver. Training and calibration were not performed.

## 2026-07-26 — Actual UR10e re-observation in the live seed-0 loop

- Replaced the fixed synthetic observation poses with the already validated
  stable 12-DOF UR10e+actual-RG6 composite from the beginning of the live
  episode. The environment remains expressed in UR10e-base coordinates, so no
  second robot insertion or second coordinate translation is needed at the
  terminal action.
- The official standalone UR10e asset was rejected for this path after five
  retained smoke attempts produced invalid resumed joint states. The imported
  asset reports a disjoint `ee_joint` transform under the existing mount, and
  pause/play caused states from approximately `9.7e5` to `2.8e11 rad`.
- Composite smoke run 006 initially sagged by approximately `0.85 rad` because
  its arm-drive strengths had not been applied. Run 007 then stopped on a false
  collision because the parent-link AABB included the complete descendant
  chain. The final checker enumerates 13 moving collision shapes, rejects an
  empty collision set, and compares those leaf bounds against scene obstacles.
- Actual-motion smoke run 009 passed
  `center -> right -> close_high` in `40.87 s`. Right and close-high final
  maximum joint errors were `0.007684` and `0.008090 rad`, both below the
  `0.02 rad` tolerance. All states remained finite and no configured
  obstacle overlap was detected. Qwen was not loaded in this smoke.
- Same-process physical integration smoke run 010 passed
  `center -> right -> close_high -> grasp`. The terminal executor reused the
  already moved composite articulation, obtained 103/96 left/right target
  contact events, no unexpected contact, and lifted the target `0.176882 m`.
  Runtime was `325.63 s`.
- The complete pretrained-Qwen run passed at
  `outputs/live_pipeline/benchmark_seed000_run009/pipeline_result.json`.
  The action sequence was
  `center -> viewpoint_right -> viewpoint_close_high -> grasp`. Both
  re-observations were continuous physics-driven UR10e motions in the same
  process and stage as the final actual-RG6 contact lift.
- Run 009's uncalibrated beliefs changed from target/inside
  `0.8330/0.7498`, to `0.9552/0.9091`, to `0.9920/0.9793`. These are tempered
  pilot scores, not empirical success probabilities. The final grasp passed
  the temporary commitment gate, recorded 103/96 bilateral contacts, no
  unexpected contact, finite final joints, and a `0.176882 m` lift.
- Run 009 took `530.32 s`. Three fresh sequential Qwen inferences took
  `59.11`, `59.35`, and `57.97 s`; peak allocation was `17.153 GiB`.
  GPU 5 was the only visible CUDA device and returned to `1 MiB` after exit.
- All three Qwen results were cache misses and are now content-addressed under
  `outputs/pilot_cache/qwen3_vl/<sha256>/` as `request.json`, `output.json`,
  `metrics.json`, and stdout/stderr logs.
- Video:
  `outputs/live_pipeline/benchmark_seed000_run009/persistent_grasp/persistent_composite_grasp.mp4`
  (1920x1080 H.264, 10 FPS, 175 frames, 17.5 seconds).
- Boundary: this is one successful deterministic pipeline-validation episode,
  not a final paper experiment or statistical guarantee. Training and fitted
  calibration were not performed. The view controller uses interpolated joint
  targets plus leaf-shape AABB aborts, not a general collision-aware MPC
  trajectory optimizer. The provisional wrist camera position follows the
  physical RG6 base, but its optical axis is re-aimed at a fixed world target
  rather than using a calibrated rigid hand-eye transform. The seed-0 grasp
  waypoints still come from simulator-ground-truth debug IK.

## 2026-07-26 — Seed-1 automatic grasp-trajectory physics smoke

- Added an explicit `--automatic-ik-smoke` mode to
  `scripts/run_ur10e_rg6_composite_grasp.py`. It permits a physics-only
  seed-specific trajectory test without a Qwen terminal action and records
  that the grasp was not VLM-authorized. Normal same-scene execution now also
  rejects a terminal trigger whose seed does not match the physics scene.
- Run
  `outputs/ur10e_rg6_physics/same_scene_seed001_auto_ik_run001/result.json`
  generated a new grasp yaw, five descending IK waypoints, and one lift IK
  solution from the seed-1 target position. All IK calls succeeded; no seed-0
  joint trajectory was reused.
- The seed-1 physical execution passed the finite-state, `0.05 rad` arm-error,
  target-displacement, bilateral-contact, and unexpected-contact gates. Final
  arm error was `0.001452 rad`, pre-lift target displacement was
  `0.001188 m`, left/right contact-event counts were `601/602`, unexpected
  contacts were zero, and the measured lift was `0.177958 m`.
- The run completed in approximately `183.7 s`, used about `3.0 GiB` observed
  GPU memory on physical GPU 5 only, and returned GPU 5 to `1 MiB`.
- Video:
  `outputs/ur10e_rg6_physics/same_scene_seed001_auto_ik_run001/ur10e_rg6_composite_grasp.mp4`
  (960x540 H.264, 10 FPS, 525 frames, 52.5 seconds).
- Boundary: this validates automatic per-seed IK and contact physics only.
  Qwen, belief update, and action selection were intentionally not run.
  Simulator ground-truth target and occluder positions supplied the debug
  grasp pose and yaw; RGB-D target localization is still pending. The result
  is not a final paper experiment.

## 2026-07-26 — Seed-1 masked RGB-D localization to automatic grasp

- Added saved pinhole intrinsics and the wrist-camera world transform to every
  observation's `camera_calibration.json`. Added
  `scripts/rgbd_target_localization.py` to backproject metric
  distance-to-camera pixels and estimate a robust world-frame 3D center from
  an instance mask.
- The actual-motion observation path now places the target at the same settled
  container-bottom pose used by physical manipulation, removing the previous
  16 mm observation/manipulation height mismatch.
- Added `scripts/run_rgbd_localized_grasp_pilot.py`. It runs an actual UR10e
  `center -> right -> close_high` capture, estimates target and occluder
  positions without reading coordinate ground truth, evaluates target error
  afterward, and only launches automatic IK/contact grasp when the target
  error is at most `0.02 m`.
- Seed-1 run
  `outputs/live_pipeline/rgbd_localized_grasp/seed001/run001/result.json`
  succeeded. The RGB-D target estimate was
  `[0.69050, -0.23070, 0.06964] m`; post-estimation simulator evaluation
  measured `0.005764 m` full 3D error and `0.001188 m` XY error.
- The estimated RGB-D positions, not simulator target coordinates, generated
  the grasp yaw, five descent IK waypoints, and lift IK. Physical execution
  recorded 574/597 left/right target-contact events, no unexpected contact,
  `0.001448 rad` final pregrasp arm error, `0.000806 m` pre-lift target
  displacement, finite final joints, and a `0.178239 m` lift.
- Total runtime was `573.27 s`: `51.04 s` for actual-UR10e RGB-D capture and
  `522.21 s` for high-resolution physical grasp rendering and execution.
  Observed GPU-5 use was about `4.0 GiB`; GPU 5 returned to `1 MiB`.
- The grasp overview camera was moved only approximately 18 cm farther and
  25 cm higher, its look-at center was raised, and output was increased from
  960x540 to 1920x1080. The first and final frames visually contain the full
  manipulator, RG6, and tabletop without cropping the robot top.
- Video:
  `outputs/live_pipeline/rgbd_localized_grasp/seed001/run001/grasp/ur10e_rg6_composite_grasp.mp4`
  (1920x1080 H.264, 10 FPS, 525 frames, 52.5 seconds).
- Boundary: Qwen was intentionally not loaded, and the target mask still comes
  from the simulator-only color-ID instance fallback. This validates metric
  RGB-D geometry and physical execution, not learned grounding, calibration,
  general multi-seed robustness, or final evaluation.

## 2026-07-26 — Seed-1 Qwen-selected RGB-D dynamic-IK closed loop

- Connected the Qwen/planner-selected anonymous candidate mask to the saved
  wrist RGB-D calibration and terminal localization. The live action request
  now carries a session-local localization artifact; the Isaac server rejects
  paths outside the session.
- Generalized the same-stage persistent composite executor. For a nonzero seed
  it requires finite, non-ground-truth RGB-D target and occluder estimates,
  computes a new grasp yaw, five descent IK waypoints, and lift IK in the
  current UR10e base frame, and preserves the physical target at its existing
  scene pose.
- The complete run passed at
  `outputs/live_pipeline/benchmark_seed001_run001/pipeline_result.json`.
  Its sequence was
  `center -> viewpoint_right -> viewpoint_close_high -> grasp`.
  Both viewpoint actions were actual continuous UR10e physics motions.
- Qwen-selected `target_red`, mapped to anonymous `object_001`; its 4,773-pixel
  candidate mask produced target position
  `[0.690499, -0.230687, 0.069419] m`. The same episode's masked RGB-D result,
  not simulator target coordinates or a seed-0 path, generated terminal IK.
- Uncalibrated target/inside beliefs changed from `0.7397/0.7708`, to
  `0.9171/0.9351`, to `0.9940/0.9932`. These remain tempered pilot scores, not
  empirical success probabilities.
- Terminal manipulation recorded 101/96 left/right target-contact events, no
  unexpected contact, finite final joints, and a `0.170878 m` lift. The
  trajectory source is recorded as
  `same_episode_qwen_selected_mask_rgbd_dynamic_debug_ik`.
- Total runtime was `528.45 s`. Three fresh sequential Qwen inferences took
  `58.97`, `55.73`, and `52.33 s`; peak allocation was `17.146 GiB`.
  All were cache misses and are now stored in the content-addressed cache.
  GPU 5 was the only visible CUDA device and returned to `1 MiB`.
- Video:
  `outputs/live_pipeline/benchmark_seed001_run001/persistent_grasp/persistent_composite_grasp.mp4`
  (1920x1080 H.264, 10 FPS, 175 frames, 17.5 seconds). The complete robot,
  RG6, tabletop, and lifted target are visible without top cropping.
- Boundary: this is the second successful deterministic integration pilot and
  the first successful nonzero-seed Qwen-to-dynamic-IK episode. It is not
  calibrated, statistically representative, or final paper evidence. Candidate
  masks are still generated by the simulator debug color-ID pass rather than a
  learned grounding model, and the action-conditioned future-belief controller
  is still a research prototype rather than a general MPC solver.

## 2026-07-26 — Seed-2 Qwen-selected RGB-D dynamic-IK closed loop

- Completed the next single-GPU pilot at
  `outputs/live_pipeline/benchmark_seed002_run001/pipeline_result.json`.
  Physical GPU 5 was the only visible CUDA device; batch size was one and no
  distributed or parallel VLM execution was used.
- The live action sequence was
  `center -> viewpoint_right -> viewpoint_close_high -> grasp`. Both
  reobservations used actual continuous UR10e physics motion in the same Isaac
  process and stage as the terminal RG6 grasp.
- The uncalibrated target/inside beliefs changed from `0.8755/0.8667`, to
  `0.9668/0.9628`, to `0.9962/0.9928`. These are tempered pilot scores and are
  not empirical success probabilities.
- Qwen selected `target_red`, mapped to anonymous `object_001`. Its 4,065 valid
  masked depth pixels produced the RGB-D world-position estimate
  `[0.660537, -0.295150, 0.072051] m`. Simulator target coordinates were not
  used for this estimate.
- Same-episode dynamic IK produced five successful descent waypoints and the
  lift solution. Physical execution recorded 101/97 left/right target-contact
  events, zero unexpected environment-contact pairs, finite final joints, and
  a verified `0.172363 m` target lift.
- Total runtime was `535.72 s`. Three fresh Qwen calls took `58.94`, `58.64`,
  and `53.16 s`; peak allocated GPU memory was `17.165 GiB`. All three calls
  were cache misses and are stored in the content-addressed Qwen cache.
- Video:
  `outputs/live_pipeline/benchmark_seed002_run001/persistent_grasp/persistent_composite_grasp.mp4`
  (1920x1080 H.264, 10 FPS, 176 frames, 17.6 seconds). Visual inspection
  confirms that the complete robot remains framed and the target is lifted.
- GPU 5 returned to `1 MiB` and 0% utilization after completion.
- Boundary: this is the third successful deterministic integration pilot, not
  calibration, final testing, or statistical evidence. The selected candidate
  mask still comes from the simulator debug color-ID pass, and the terminal
  path is dynamic debug IK rather than a general obstacle-aware MPC solver.

## 2026-07-27 — Learned grounding pilot on saved seed-0 through seed-2 RGB-D

- Created the dedicated Python 3.12 environment
  `/data/wonheekoh/venvs/efficient-robotics-perception`. The existing
  `isaacsim_venv` and Qwen VLM environment were not modified. The environment
  uses PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, numpy 1.26.4,
  SciPy 1.15.3, transformers 4.57.3, official SAM3 source commit
  `46957e47805eaa273f4aa7bbbd25a88bca9108ce`, and official Grounded-SAM2
  source commit `b7a9c29f196edff0eb54dbe14588d7ae5e3dde28`.
- Downloaded and revision-pinned the local
  `IDEA-Research/grounding-dino-base` and
  `facebook/sam2.1-hiera-large` checkpoints. The official
  `facebook/sam3` checkpoint remains unavailable: the authenticated Hugging
  Face account can read repository metadata but receives a gated-model 403
  for `sam3.pt`. No unofficial mirror was used.
- Added the fixed nine-observation configuration
  `configs/perception/grounding_pilot_seed0_2.json` and one-model-at-a-time
  inference/evaluation tools:
  `scripts/run_perception_grounding_pilot.py`,
  `scripts/evaluate_perception_grounding_pilot.py`,
  `scripts/export_grounded_sam2_qwen_inputs.py`,
  `scripts/run_grounded_proposal_qwen_ranking.py`, and
  `scripts/render_perception_grounding_pilot.py`.
- The inference stages read only the saved 640x480 RGB images. Simulator
  instance IDs, masks, depth, and semantic ground truth are read only by the
  separate post-inference evaluator. Training, LoRA, manual annotation,
  calibration, distributed execution, and multi-GPU execution were not used.
- Qwen3-VL direct relation-conditioned target boxes:
  - valid boxes were emitted for four of nine observations;
  - three of nine observations passed bbox IoU 0.5;
  - mean bbox IoU over all nine observations was `0.2749`;
  - the model returned a valid empty JSON list on five difficult views rather
    than inventing a box;
  - inference averaged `0.729 s` per observation after loading, peak allocated
    memory was `16.472 GiB`, and model loading took `6.62 s`.
- GroundingDINO-Base followed by SAM2.1-Large:
  - all seven object/reference concept prompts were evaluated independently;
  - across 72 visible ground-truth object instances, bbox recall at IoU 0.5
    was `0.7222` and mask recall at IoU 0.5 was `0.7639`;
  - mean assigned mask IoU was `0.7612`;
  - the low uncalibrated proposal threshold produced 329 proposals, of which
    only 55 were matched at mask IoU 0.5, for provisional proposal precision
    `0.1672`; threshold calibration, NMS, and proposal deduplication remain
    necessary;
  - GroundingDINO averaged `1.402 s` per observation at `2.309 GiB` peak;
    SAM2 averaged `0.394 s` at `2.001 GiB` peak.
- Exported only GroundingDINO's red-object masks as anonymous candidates and
  ran one cached Qwen model instance to select the instruction target. The
  semantic proposal labels and simulator IDs were not exposed to Qwen.
  The selected mask passed mask IoU 0.5 on all 9/9 observations, with mean
  mask IoU `0.9422`, mean bbox IoU `0.9084`, and mean RGB-D centroid error
  `0.00227 m` on the six observations with saved camera calibration.
  Qwen ranking averaged `2.852 s` per observation and peaked at `16.899 GiB`.
- A sequential learned-mask observation therefore currently costs about
  `4.65 s` per observation, or about `13.94 s` for a three-view episode,
  excluding amortized model-loading and simulator motion/rendering time. This
  ranking stage evaluates only target selection, not the full set of relation
  scores used by the live planner.
- Results, raw outputs, masks, runtime/VRAM metrics, CSV evaluation, and
  prediction-only visualizations are under
  `outputs/perception_grounding_pilot/seed0_2/`. Representative panels verify
  that direct Qwen grounding can abstain on the occluded center view while
  Qwen can still select the correct small target from anonymous learned masks.
- GPU policy: every model process saw only physical GPU 5 as `cuda:0`, batch
  size was one, and only one model instance was loaded at a time. GPU 5
  returned to `1 MiB` and 0% utilization after completion.
- Boundary: the nine observations contain colored geometric debug objects,
  not final mug/basket household assets. Scores and thresholds are
  uncalibrated, the same three seeds were used for pilot debugging and
  evaluation, SAM3 is not yet evaluated, and these results are not final paper
  evidence or statistical guarantees.

## 2026-07-27 — Seed-0 procedural mug/open-basket re-observation pilot

- Added a separate household-shaped perception scene without downloading or
  copying third-party meshes. `scripts/household_pilot_scene.py` constructs an
  open red mug with a visible white logo, a red mug distractor, and an open
  slatted basket while preserving the benchmark semantic prim paths.
- Captured 960x720 RGB-D on physical GPU 5 only. The provisional `left` joint
  pose looked into the lab wall and `close_high` looked below the table, so
  neither was silently treated as a valid observation. The pilot evaluates the
  valid `center` and `right` captures under
  `outputs/household_perception_pilot/benchmark_seed000/`.
- The temporary color-ID evaluator initially dropped saturated red mug pixels.
  Increased its measured chromaticity radius from 0.20 to 0.30 and added a
  saved-ID reclassification tool. No inference output was changed by this
  evaluator repair.
- Qwen direct target-box generation abstained on both views. GroundingDINO and
  SAM2 generated two anonymous red-mug candidates per view. The low proposal
  threshold produced 23 total mug/basket proposals; mask recall at IoU 0.5 was
  `0.8333`, but provisional proposal precision was only `0.2174`.
- Corrected the Qwen input contract so semantic RGB attributes inside a
  detector box are preserved even when a class-conditioned SAM mask omits a
  contrasting logo. Target identity and spatial relation are now scored as
  separate raw-logit outputs rather than one conflated match question.
- On the occluded center view, Qwen selected the prominent outside mug and
  classified it as `outside`; target mask IoU was `0.0`. After the fixed
  validation re-observation to the right, it selected the correct logo mug,
  classified it as `inside`, achieved bbox IoU `0.9545`, mask IoU `0.7818`,
  and RGB-D centroid error `0.00171 m`.
- The right-view identity match-minus-nonmatch logit remained negative
  (`-3.625`) and uncalibrated even though its relative candidate selection was
  correct. Grasp commitment therefore remains blocked; selecting the maximum
  candidate is not itself a safe commitment gate.
- GroundingDINO averaged about `0.54 s` per valid view in the final run and
  peaked at `2.309 GiB`; SAM2 averaged `0.41 s` and peaked at `1.368 GiB`;
  factorized Qwen target/relation scoring averaged `7.09 s` and peaked at
  `16.988 GiB`. Models were loaded sequentially with batch size one.
- Final artifacts are under
  `outputs/perception_grounding_pilot/household_seed000/`, including
  `evaluation_summary.json`, `reobservation_summary.json`, raw logits,
  predicted masks, RGB-D evaluation, and prediction-only visualizations.
- Boundary: the right-view recovery validates the intended perception effect
  of re-observation, but the view action was a fixed seed-0 transition, not
  belief-space MPC. This is not an end-to-end grasp, calibration set, final
  test, paper result, or statistical guarantee.

## 2026-07-27 — Scanned-basket, actual-wrist, two-candidate perception pilot

- Replaced the temporary slatted reference visual with the locally available
  textured LIBERO scanned-basket mesh. The source asset remains in the LIBERO
  checkout, is not copied into this repository, and its source, path, scale,
  and CC BY 4.0 attribution are recorded in `household_scene.json`.
- Scaled the basket to approximately `0.364 x 0.341 x 0.155 m`. The first
  scanned-basket capture kept the old rear distractor location, which hid that
  mug in all reachable views. GroundingDINO therefore returned only one red
  candidate and candidate export stopped rather than pretending that a
  two-candidate ranking had been tested.
- Moved the no-logo red distractor to a recorded outside-basket pilot
  location and preserved the failed artifacts. In the corrected scene,
  GroundingDINO returned exactly two red-mug candidates in each of `center`,
  `right`, and `close_high`.
- The imported 12-DOF UR10e+RG6 articulation executed
  `center -> right -> close_high` on physical GPU 5 only. Both commanded
  transitions completed with finite joint states, no monitored AABB collision,
  and maximum final arm-joint errors of `0.00768` and `0.00809 rad`. The
  simulator run took `55.21 s` and saved 960x720 RGB-D, calibration, ID-pass,
  and overview images for all three views.
- The learned inference path used RGB only:
  `GroundingDINO-Base -> SAM2.1-Large -> anonymous proposals -> Qwen3-VL-8B`.
  Qwen selected the correct target and classified it as `inside` on all 3/3
  views. Selected-mask bbox IoU averaged `0.9511`, mask IoU averaged `0.9072`,
  and RGB-D centroid error averaged `0.00105 m`.
- Qwen direct bounding boxes were emitted on all three views but passed bbox
  IoU 0.5 on only 1/3. The modular proposal-plus-Qwen path therefore remains
  the current integration path.
- GroundingDINO averaged `0.475 s/view` at `2.309 GiB` peak, SAM2 averaged
  `0.323 s/view` at `1.368 GiB` peak, and factorized Qwen target/relation
  scoring averaged `7.287 s/view` at `17.065 GiB` peak. One model instance,
  batch size one, and no training or calibration were used.
- Corrected outputs are under
  `outputs/perception_grounding_pilot/scanned_basket_two_candidate_seed000/`;
  the actual-motion RGB-D is under
  `outputs/live_pipeline/actual_view_motion_smoke/seed000/run003/`.
- Boundary: the basket currently has visual geometry only, so this scene is
  not approved for contact manipulation. All three views are already easy
  enough for the uncalibrated pilot model, and their order was fixed rather
  than selected by action-conditioned belief-space MPC. This is not a
  calibration set, final test, final paper result, or statistical guarantee.

## 2026-07-27 — Controlled-occlusion action-conditioned belief pilot

- Added an explicit controlled-occlusion variant of the scanned-basket scene.
  The orange cylinder is raised and enlarged so the center view hides most of
  the target logo while preserving lateral and close-high observations. This
  is a procedural simulation intervention, not manual image annotation.
- GroundingDINO returned two red-mug proposals in every tested view. In the
  initial three-view diagnostic, Qwen selected the target mask in all views
  but classified the selected relation as `outside` at center and right and
  as `inside` at close-high. This confirmed that the scene creates relation
  ambiguity rather than a missing-target failure.
- Added a non-oracle engineering planner configuration with `center_repeat`,
  `right`, and `close_high` actions. Before choosing an action, the planner
  reads only center belief and a geometry-informed, hand-specified
  action-conditioned observation-likelihood model. It does not read future
  RGB-D, masks, or Qwen outputs.
- The pre-action horizon-two costs were `0.74247` for center repeat,
  `0.70083` for right, and `0.59345` for close-high. The planner selected
  `close_high`, whose predicted information gain was `0.55419 nats`.
- Re-ran the selected action as a direct `center -> close_high` UR10e+RG6
  trajectory in run 005 rather than reusing the earlier right-to-close
  capture. The 36-waypoint trajectory completed in `48.50 s`, had no monitored
  AABB collision, and ended with maximum arm-joint error `0.00816 rad`.
- Re-ran GroundingDINO, SAM2, and Qwen on run 005 RGB-D only. Anonymous
  candidates were associated across views by learned-mask RGB-D world centers,
  with no simulator IDs or semantic ground truth used for inference,
  tracking, or planning.
- The target-track belief increased from `0.6077` to `0.7186`; the `inside`
  relation belief increased from `0.4594` to `0.6945`; and debug task-failure
  risk decreased from `0.7208` to `0.5009`.
- The replanner correctly refused grasp because the uncalibrated debug risk
  remained above the `0.15` safety threshold, and requested `right` as the next
  observation. This is a useful safe-defer outcome, not a completed retrieval.
- On the two executed observations, the Qwen-selected learned masks had mean
  mask IoU `0.8769` and mean RGB-D centroid error `0.00224 m`. Qwen scoring
  averaged `7.14 s/view` at `17.065 GiB` peak. No training, calibration,
  simulator-mask inference, distributed execution, or multi-GPU execution was
  used.
- One Qwen process received external `SIGTERM` code 143 after completing and
  saving center and right. GPU 5 was clean afterward; the runner resumed and
  computed only the missing close-high result. No completed output was
  discarded.
- Primary artifacts:
  `outputs/scanned_basket_occlusion_belief_mpc_seed000/pilot_result.json`,
  `outputs/scanned_basket_occlusion_belief_mpc_seed000/pre_action_plan.json`,
  `outputs/live_pipeline/actual_view_motion_smoke/seed000/run005/`, and
  `outputs/perception_grounding_pilot/scanned_basket_occlusion_direct_action_seed000/`.
- Boundary: this validates one action-conditioned future-belief integration
  step and replanning decision, but `actual_mpc_solver` remains false. The
  observation likelihood is hand specified and uncalibrated, only seed 0 was
  used, no grasp was executed, and the result is not final paper evidence.

## 2026-07-27 — Two-reobservation belief update through grasp request

- Executed `center -> close_high -> right` in one persistent Isaac process
  using the actual imported UR10e+RG6 articulation. Both viewpoint
  trajectories passed joint-limit, finite-state, and AABB collision checks.
  The full simulation run took `62.43 s`; final arm-joint errors were
  `0.00816 rad` for close-high and `0.00803 rad` for right.
- Re-ran the complete learned perception path on run 006 RGB-D only.
  GroundingDINO produced two red-mug candidates per view, SAM2 produced masks,
  and Qwen scored target identity and relation with one model instance on
  physical GPU 5. Qwen averaged `7.184 s/view` and peaked at `17.069 GiB`.
- The two-step sequential runner writes each pre-action plan before reading
  the selected next-view Qwen output. It selected close-high first and right
  second, matching the executed run 006 sequence. Unselected future outputs
  were not read during either action choice.
- Belief progression:
  - center: target `0.6151`, inside `0.4165`;
  - after close-high: target `0.7186`, inside `0.6491`;
  - after right: target `0.8948`, inside `0.9509`.
- The debug task-failure risk fell to `0.14912`, barely below the unchanged
  temporary `0.15` commitment threshold. The engineering planner therefore
  requested `grasp`.
- The grasp request was deliberately not executed. The scanned basket still
  lacks validated collision geometry, and Qwen scores and the observation
  likelihood remain uncalibrated. The result records both blockers rather
  than treating the planner request as physical authorization.
- Post-hoc simulator evaluation, performed only after inference/planning,
  found correct target selection on 3/3 views and correct `inside` relation on
  close-high and right. Center intentionally produced the wrong top relation
  `outside`. Qwen-selected masks had mean mask IoU `0.8505` and mean RGB-D
  centroid error `0.00199 m`.
- Artifacts:
  `outputs/scanned_basket_occlusion_two_step_belief_mpc_seed000/pilot_result.json`,
  `outputs/scanned_basket_occlusion_two_step_belief_mpc_seed000/pre_action_plan_000.json`,
  `outputs/scanned_basket_occlusion_two_step_belief_mpc_seed000/pre_action_plan_001.json`,
  `outputs/scanned_basket_occlusion_two_step_belief_mpc_seed000/final_replan.json`,
  `outputs/live_pipeline/actual_view_motion_smoke/seed000/run006/`, and
  `outputs/perception_grounding_pilot/scanned_basket_occlusion_two_step_seed000/`.
- Boundary: run 006 validates the planned two-view motion sequence and a
  sequential offline perception/planning replay. Qwen inference did not run
  concurrently inside the persistent Isaac process, the likelihood remains
  hand specified, `actual_mpc_solver` is false, no calibration or grasp was
  performed, and this is not final paper evidence.

## 2026-07-27 — Scanned-basket collision and RG6 physics smoke

- Added an explicitly physics-only scanned-basket variant with a documented
  static bottom-plus-four-walls collision approximation. The perception scene
  remains at scale `[2.10, 2.10, 1.05]`; the manipulation-clearance variant is
  separately recorded at `[3.20, 3.20, 1.05]`, approximately
  `0.555 x 0.519 x 0.155 m`.
- Two narrow-basket attempts correctly stopped at the safety gate. At the
  original scale the RG6 inner finger contacted the right wall; a 4 cm higher
  grasp then made the inner fingers and outer knuckles contact the front wall.
  These are retained as failed debug runs, not successes.
- A later expanded-basket attempt exposed a separate target-physics bug:
  the household target root had been changed from `Cube` to `Xform`, so
  wrapping it as `UsdGeom.Cube` did not create collision geometry. The visible
  mug therefore settled below its intended pose and the empty RG6 closure
  destabilized the arm.
- Fixed the household target by keeping its visible mug children on the Xform,
  applying the rigid body to that root, and adding a hidden cylinder collision
  child matching the mug body. Contact tracking now uses that collider path.
- Run 006 originally passed the then-current physics smoke on physical GPU 5
  in `325.87 s`:
  bilateral RG6 contact was confirmed (`102` left, `99` right), unexpected
  environment contacts were `0`, pre-lift arm error was `0.01682 rad`, and
  verified target lift was `0.17472 m`. Final joints remained finite.
- Artifacts:
  `outputs/live_pipeline/scanned_basket_collision_grasp_smoke/run006/smoke_result.json`
  and
  `outputs/live_pipeline/scanned_basket_collision_grasp_smoke/run006/persistent_grasp/persistent_composite_grasp.mp4`.
- Correction: run 006 is no longer accepted as realistic grasp validation.
  Its hidden mug cylinder used the inner radius (`0.034 m`) rather than the
  visible outer radius (`0.041 m`), the mug mass was only `0.04 kg`, friction
  was `4.0/3.0`, and all six RG6 joints were driven independently despite five
  follower joints carrying mimic constraints. It remains only an early
  contact/lift code-path smoke.
- Boundary: Qwen and the planner were not loaded in this smoke, the grasp used
  the prevalidated seed-0 simulator-ground-truth debug IK trajectory, the
  collision basket is a five-box approximation, and this is not final
  evaluation evidence. It does not validate the realistic physical grasp path;
  run 015 below supersedes it for that narrow purpose.

## 2026-07-27 — Realistic RG6 proxy and master–mimic grasp revalidation

- Replaced the hidden mug proxy with the visible outer radius `0.041 m`,
  height `0.102 m`, provisional mass `0.30 kg`, and provisional friction
  `0.80/0.60`. Added contact-impulse force measurement, penetration tracking,
  a `60 N` per-side force ceiling, a `3 mm` penetration ceiling, and a
  `3 N` minimum measured force per side before lift.
- Runs 007–009 correctly failed rather than manufacturing a lift. Run 007
  stopped at touch with no measurable grip force. Runs 008 and 009 exceeded
  the `60 N` force ceiling. Run 010's attempt to limit all six RG6 drives
  destabilized the articulation; that change was reverted.
- The imported RG6 has one master joint and five `NewtonMimicAPI` followers,
  but the old controller also drove every follower independently. Added
  `scripts/smoke_rg6_master_mimic.py`, removed follower drives only in the
  in-memory experiment stage, and commanded the master alone. Open, close,
  and reopen passed with maximum no-contact mimic error about
  `1.2e-6 rad`.
- Run 011 used master-only control with the original `1000 N·m` imported
  max-force value and correctly stopped at `67.54 N`. Run 012 limited the
  master to `0.50 N·m` and was too weak (`2.69 N` left). Runs 013–014 used
  `0.60 N·m`; both sides exceeded `3 N`, but exposed an incorrect
  same-callback force gate. The gate now tracks active contact pairs and
  contact loss separately from intermittent PhysX force reports.
- Physics-only run 015 completed on physical GPU 5 in `352.94 s`. It retained
  bilateral contact after lift, lifted the mug `0.17970 m`, had only
  `0.000335 m` horizontal slip, and recorded maximum forces of `8.00 N` left
  and `6.82 N` right. Maximum penetration was `0.0000893 m` left and
  `0.00000924 m` right. There were no unexpected environment contacts,
  final mimic error was `0.00145 rad`, and no attachment or target-pose
  copying was used.
- The verified MP4 is `1920x1080`, `174` frames at `10 fps`, and `17.4 s`:
  `outputs/live_pipeline/scanned_basket_collision_grasp_smoke/run015/persistent_grasp/persistent_composite_grasp.mp4`.
  Structured results are in the adjacent `result.json` and run-level
  `smoke_result.json`.
- Boundary: run 015 validates only the seed-0 physics grasp with a provisional
  actuator limit and a simplified cylinder mug collider. It uses a
  prevalidated debug IK trajectory and does not load Qwen or the planner.
  It is not calibration evidence, MPC evidence, a multi-seed result, a
  real-robot result, or final paper evidence.

## 2026-07-27 — Live learned-perception active re-observation pilot

- Added `scripts/run_live_learned_scanned_basket_pipeline.py`, which keeps one
  Isaac Sim process alive while loading GroundingDINO-Base, SAM2.1-Large, and
  Qwen3-VL-8B sequentially on physical GPU 5. Each pre-action plan is written
  before the selected future RGB-D observation is requested.
- The live inference path now supports a partially observed candidate set.
  A persistent candidate that is absent from the current detector output is
  retained with zero log evidence rather than deleted or hallucinated.
- Added an `0.08 m` RGB-D track-distance gate. In the right view it rejected a
  duplicate target proposal that was `0.4407 m` from the outside-cup track,
  while accepting the target proposal at `0.00447 m`.
- Run 001 stopped before Qwen because only one red-mug proposal was available
  in the original expanded-basket center framing. Run 002 completed the first
  learned plan and physical `center -> close_high` motion, then stopped because
  the exporter incorrectly required two candidates in every view. Both failed
  attempts remain recorded and no grasp was attempted.
- Corrected run 004 completed in `304.56 s`:
  `center -> viewpoint_close_high -> viewpoint_right -> defer`. Both physical
  UR10e transitions were finite, collision-checked, and collision-free, with
  maximum final joint errors `0.00816` and `0.00803 rad` under the `0.02 rad`
  limit.
- Belief after the final right observation was `0.56218` for target track 001
  and `0.99361` for `inside`. The resulting uncalibrated debug task-failure
  risk was `0.44142`, above the unchanged `0.15` commitment gate, so the
  controller correctly deferred instead of executing RG6 grasp.
- GroundingDINO inference used at most `2.309 GiB`, SAM2 at most `1.284 GiB`,
  and Qwen allocated at most `17.071 GiB`. Observed combined use with resident
  Isaac reached about `23 GiB`. GPU 5 returned to `1 MiB`; no other physical
  GPU was exposed.
- Artifacts:
  `outputs/live_pipeline/learned_scanned_basket_e2e/seed000/run004/pipeline_result.json`,
  its three RGB-D observation directories, per-stage logs, Qwen outputs,
  belief plans, and Isaac motion records.
- Added a content-addressed cache for subsequent factorized Qwen ranking calls
  at `outputs/pilot_cache/grounded_qwen_factorized/<sha256>/`. The key hashes
  normalized inference content plus RGB, crop, context, candidate-mask,
  container-mask, and overlay bytes, model configuration, and prompt version;
  session and episode identifiers do not affect reuse.
- Boundary: one live pipeline-validation episode completed, but retrieval did
  not complete because grasp was safely deferred. Scores and the observation
  model remain uncalibrated, proposal thresholds are not frozen, the planner
  reports `actual_mpc_solver=false`, and this is not final paper evidence.

## 2026-07-27 — GPU-5 background calibration pilot launched

- Added a bounded, resumable calibration-only pipeline for simulator seeds
  `100` through `109`. Each episode captures `center`, `close_high`, and
  `right` RGB-D observations, then runs GroundingDINO-Base, SAM2.1-Large, and
  Qwen3-VL-8B-Instruct sequentially with one model instance at a time.
- The job performs no training, fine-tuning, LoRA, distributed execution, or
  final testing. Seeds `200` through `209` are reserved and are not read by
  calibration.
- Simulator instance masks are hidden during inference and are used only
  afterward to generate calibration labels and diagnostics. All views from
  one episode remain in the same calibration split.
- Added
  `scripts/run_grounded_qwen_calibration_pilot.py`,
  `scripts/launch_grounded_qwen_calibration_background.py`, and
  `tests/test_grounded_qwen_calibration_pilot.py`. The focused CPU checks
  passed (`9` tests).
- Background PID `2995831` was launched at
  `2026-07-27T18:29:06.825563+00:00`. It is restricted to physical GPU 5,
  which appears as `cuda:0`; the live Isaac process UUID was verified against
  the physical GPU-5 UUID.
- Job root:
  `outputs/calibration_pilot/scanned_basket_seed100_109/`. Progress is written
  atomically to `status.json`, combined output to `background.log`, and the
  terminal result to either `COMPLETED.json` or `FAILED.json`.
- Estimated runtime is approximately `25–45 min`; expected peak combined GPU
  use is approximately `23 GiB` during Qwen plus resident Isaac stages.
- Boundary: the current scenes cover the `inside` relation only. Target
  identity temperature fitting is a calibration pilot, while any relation
  temperature produced by this run is diagnostic and not a deployable final
  relation calibration.

## 2026-07-27 — Background calibration pilot completed

- The GPU-5-only background job completed without a failed episode or failed
  inference stage. It produced `10` episode-disjoint calibration scenes and
  `30` RGB-D observations (`center`, `close_high`, and `right`) in
  `1315.15 s` (`21.92 min` total).
- GroundingDINO, SAM2, Qwen candidate/relation inference, simulator-only
  post-hoc evaluation, and temperature fitting all returned code zero.
  Qwen inference averaged `5.865 s` per observation and peaked at
  `17.353 GiB` allocated GPU memory. The sequential learned stack averaged
  approximately `8.449 s` per observation, excluding capture and model-load
  overhead.
- The selected Qwen mask overlapped the simulator target above IoU `0.5` in
  all `30/30` observations, with mean mask IoU `0.9127` and mean RGB-D
  centroid error `0.00111 m`. The selected relation was `inside` in `30/30`.
- Important interpretation: `28/30` observations contained only one exported
  red-mug candidate; only one observation had two candidates and one had four.
  Thus the `30/30` selection result mostly verifies target proposal retention,
  not robust ambiguous-candidate discrimination.
- GroundingDINO plus SAM2 had mask recall `0.6667` across the three evaluated
  semantic instances per observation and proposal precision `0.6122`; it
  regularly missed the second red candidate and produced duplicates. Proposal
  generation remains the immediate perception bottleneck.
- Individual Qwen match-versus-nonmatch labels were correct for `23/34`
  exported proposals (`67.65%`): all three non-target proposals were rejected,
  but `11/31` target-overlapping proposals received a negative raw match
  margin. Temperature `5.65` reduced calibration-set NLL from `1.0804` to
  `0.4220` and Brier score from `0.5396` to `0.2978`, without changing
  classification accuracy.
- These calibration values are not frozen for deployment: the calibration set
  is highly imbalanced (`31` target-overlapping versus `3` non-target
  proposals), the same calibration split was used to fit and report the
  diagnostic metrics, and relation labels contain only `inside`.
- Artifacts:
  `outputs/calibration_pilot/scanned_basket_seed100_109/COMPLETED.json`,
  `calibration_fit.json`, `calibration_records.json`, the perception
  `evaluation_summary.json`, all `30` Qwen result JSON files, and per-stage
  logs. Training and final testing were not performed.

## 2026-07-28 — Two-candidate inside/outside pilot on physical GPU 4

- Added an opt-in `PHYSICAL_GPU` policy while preserving physical GPU 5 as the
  default. The user explicitly selected physical GPU 4 for this foreground
  run. GPU 4 was empty before launch, appeared alone as `cuda:0`, and returned
  to `1 MiB` after completion. Distributed and multi-GPU execution remained
  disabled.
- Captured seeds `110–113` with the perception-scale scanned basket rather
  than the larger grasp-clearance basket. All four UR10e episodes completed
  `center -> close_high -> right`, producing `12` RGB-D observations.
- The first export attempt deliberately stopped before Qwen because
  `seed111_center` had only one red-mug proposal under a temporary
  two-candidates-per-view gate. Visual inspection showed that the outside mug
  was in frame; the target was nearly hidden behind the intended occluder.
  The gate was corrected to preserve one-proposal views as target-missing
  negative evidence rather than deleting the observation.
- The initial post-hoc evaluation also exposed a simulator-ID bug: after
  moving the rear mug, the old horizontal same-color split swapped the large
  magenta mug and the small purple boundary marker. Replaced that rule with a
  connected-component-area split. Original GT arrays/labels/statistics were
  preserved with `.pre_area_fix` names, and all 12 stored color-ID passes were
  reclassified without modifying RGB, depth, detector, SAM2, or Qwen outputs.
- Corrected GroundingDINO+SAM2 performance across target mug, outside mug,
  and basket was mask recall `0.8889`, proposal precision `0.8421`, mean mask
  IoU `0.8130`, and mean RGB-D centroid error `0.00772 m`. The four misses
  were the intentionally strongly occluded target in center/right for seeds
  111 and 113.
- Eight observations exposed both red candidates and four exposed only the
  outside candidate. Conditional on both candidates being available, Qwen
  ranked the target correctly in `8/8`; end-to-end per-view target selection
  was therefore `8/12`, with all four failures caused by missing target
  proposals.
- Across all 20 exported proposals, Qwen target match/nonmatch classification
  was `17/20` (`85%`): it rejected all `12/12` outside candidates, accepted
  `5/8` detected targets, and gave three detected but occluded targets a
  negative absolute match margin. Target temperature `2.85` reduced
  calibration-set NLL from `0.3823` to `0.2217` and Brier score from `0.2539`
  to `0.1557`; accuracy was unchanged.
- Qwen relation classification was also `17/20` (`85%`): all `12/12`
  distractors were `outside`, while `5/8` targets were `inside` and three
  occluded targets were called `outside`. Relation temperature `2.90` reduced
  calibration-set NLL from `0.7472` to `0.3776`.
- Qwen averaged `6.754 s` per observation and peaked at `17.075 GiB` allocated
  GPU memory. One model instance and batch size one were used. Training,
  fine-tuning, LoRA, and final testing were not performed.
- Artifacts:
  `outputs/calibration_pilot/two_candidate_inside_outside_seed110_113_gpu4/`.
  The corrected `calibration_fit.json`, perception
  `evaluation_summary.json`, 12 saved Qwen result JSON files, raw masks, and
  per-stage logs are retained.
- Boundary: this is a four-episode calibration/debug pilot. Temperatures were
  fit and diagnosed on the same small calibration split, and `behind` and
  `unknown` labels remain uncovered. It is not a final calibration, unbiased
  test, MPC result, grasp result, or paper-scale performance claim.

## 2026-07-28 — Remove active-occluder basket penetration

- Visual review found that the old faceted orange/yellow cylinder visibly
  intersected the front of the perception-scale scanned basket. Its radius was
  `0.055 m`, while its center was only about `0.045 m` behind the basket front
  boundary; it also floated above the basket bottom.
- Replaced the fixed occluder pose with a target-conditioned layout. The
  occluder is placed along the target-to-center-camera direction, clipped to
  the scanned basket interior, placed on the basket bottom, and analytically
  checked against both the basket bounds and the target mug.
- A small non-intersecting cylinder in run 002 removed penetration but also
  made the center view too easy. A runtime attempt to replace the Cylinder prim
  schema with a Cube at the same path stalled Isaac initialization; failed
  run 003 is preserved as a diagnostic and contains no observation output.
- The stable final representation preserves the authored Cylinder schema and
  nonuniformly scales it into an oriented thin elliptical plate. Final full
  extents are `0.085 x 0.020 x 0.160 m`; the plate rests on the basket bottom.
- Seed-111 run 005 completed on physical GPU 4. The recorded minimum basket
  wall clearance is `0.008 m`, target surface clearance is `0.01717 m`, and
  `geometry_validation_passed=true`. The same analytic constraints passed for
  generated seeds `0–999`.
- In the corrected center RGB, the plate hides the target's white logo and
  lower body without penetrating the basket. In `close_high` and `right`, the
  target and logo become more observable. Thin orange strips visible through
  the basket side are seen through the scanned weave openings, not geometry
  penetration.
- Both actual UR10e re-observation trajectories completed collision-free with
  maximum final joint errors `0.00816 rad` and `0.00803 rad`. GPU 4 returned
  to `1 MiB`.
- Artifacts:
  `outputs/live_pipeline/calibration_capture/seed111/run005/`, including the
  three RGB-D observations, `household_scene.json`, and `smoke_result.json`.

## 2026-07-28 — Supersede primitive occluder with basket-rim occlusion

- Correction: the run-005 plate still looked like a faceted yellow primitive
  penetrating the irregular scanned weave, despite clearing the approximate
  analytic basket bounds. The previous claim that this visual was acceptable
  is withdrawn. Do not use runs 001–005 as scene-quality evidence.
- Removed the explicit `/World/OccluderOrange` visual from the active scene.
  The target is now placed deeper inside the basket and the scanned basket rim
  is recorded as the occlusion source. No yellow/orange primitive remains
  inside the basket.
- Seed-111 run 006 completed with `explicit_occluder_primitive_visible=false`,
  target-to-approximate-wall clearance `0.0493 m`, and collision-free actual
  UR10e `close_high` and `right` motions. GPU 4 returned to `1 MiB`.
- Artifact:
  `outputs/live_pipeline/calibration_capture/seed111/run006/`.
- Boundary: run 006 fixes the visual penetration, but the current center wrist
  view looks down far enough that the target and white logo remain too easy to
  see. Run 006 is accepted only as a scene-geometry correction smoke, not as
  the final active-reobservation perception scene. The next scene revision
  must create uncertainty with a physically reachable lower/shallower initial
  wrist view or a validated realistic household occluder.

## 2026-07-28 — Correct scanned-basket mug support contact

- Visual review found that the red target mug in run 006 was floating. The
  household mug is an Xform whose origin is at the mug bottom, but the
  actual-view placement code treated that origin as the object center and
  added half the nominal target height. This raised the visual mug by about
  `0.051 m`.
- Moved the target from the raised rear liner to the flatter central interior
  and aligned its bottom origin with the scanned-mesh support surface. Vertical
  ray checks over the mug footprint set the support offset to `0.020 m` above
  the basket root.
- Seed-111 run 007 completed on physical GPU 4 in `90.34 s`. The recorded
  bottom-contact world position is `[0.64844, -0.20000, 0.01500] m`; the
  separate mug-center position is `[0.64844, -0.20000, 0.06600] m`.
  Center, close-high, and right RGB-D captures completed, and the actual UR10e
  re-observation motions remained collision-free. GPU 4 returned to `1 MiB`.
- Artifact:
  `outputs/live_pipeline/calibration_capture/seed111/run007/`.
- Boundary: run 007 supersedes run 006 for scene-geometry inspection. The mug
  support now looks physically plausible, but the white logo is too visible
  in the center wrist view. Do not use run 007 as the final
  active-reobservation perception scene or as calibration/test evidence.

## 2026-07-28 — Validate lower center view and live learned re-observation

- The first lower-center candidate (run 008) moved the camera from
  `z=0.4383 m` to `z=0.3750 m` but left the white logo fully visible. It is
  retained as a failed view-design attempt.
- A second kinematically generated candidate lowered the camera to
  `z=0.2792 m`. Seed-111 run 009 completed center, close-high, and right
  capture with collision-free UR10e motion. The basket rim hides most of the
  target body/logo in center while the higher views restore the evidence.
- Offline GroundingDINO-Base, SAM2.1-Large, and Qwen3-VL-8B inference on
  run 009 selected the correct target and `inside` relation in all three
  views. Qwen's target match logit changed from `0.375` in center to `18.75`
  in close-high and `19.75` in right. The selected masks averaged
  `0.9412` IoU and `0.00134 m` RGB-D centroid error. Qwen peak allocated
  memory was `17.0644 GiB`.
- The first live attempt, seed-111 run 001, completed both observations and
  the planner decision but exited after writing `server_result.json` because
  post-run debug bbox logging dereferenced the now-inactive legacy robot prim.
  This cleanup failure was fixed by skipping invalid/inactive prims.
- Seed-111 live run 002 then completed successfully in `264.63 s` in one
  persistent Isaac process:
  `center RGB-D -> GroundingDINO -> SAM2 -> Qwen -> belief/plan ->
  collision-checked close-high UR10e motion -> new RGB-D -> learned
  perception -> belief/replan`.
- The initial live belief was target `0.9497`, inside `0.9770`; after
  close-high it became target `0.9975`, inside `0.9955`. The planner requested
  grasp after the re-observation. Grasp execution was intentionally disabled,
  so the external terminal action was `stop`.
- Post-hoc simulator evaluation, not available to inference or planning, gave
  selected-target mask IoU `0.9273` in center and `0.9641` in close-high; both
  selected relations were `inside`.
- Physical GPU 4 was shared with unrelated jobs using about `9 GiB`. The
  pipeline used one visible device, one model instance at a time, and batch
  size one. Runtime is therefore debugging evidence, not an uncontended
  performance benchmark.
- Artifacts:
  `outputs/live_pipeline/learned_scanned_basket_e2e/seed111/run002/` and
  `outputs/perception_grounding_pilot/scanned_basket_seed111_run009/`.
- Boundary: the adapter temperature and observation likelihoods remain
  uncalibrated and hand specified. This is a successful live integration
  pilot, not final belief-space MPC, final grasp, calibration, or paper-scale
  evaluation.

## 2026-07-28 — Complete one live learned-perception contact-grasp episode

- Removed the obsolete orange-occluder proposal from the live perception
  configuration and terminal localization. Terminal IK now uses only the
  Qwen-selected anonymous target mask and its current RGB-D points.
- Matched the scanned-basket collision approximation to the perception-scale
  visual basket. The approximation remains active in PhysX but is explicitly
  hidden from rendering. GPU-4 smoke run 002 completed
  `center -> close_high` with no collision and maximum final joint error
  `0.00803 rad`; visual inspection confirmed that no gray collision box or
  faceted occluder is visible.
- Live seed-0 run 005 reached `center -> close_high -> grasp` but failed before
  pre-grasp because the temporary zero-yaw default requested an unnecessary
  large wrist roll. The final wrist error was `1.33291 rad`; no collision or
  contact caused this failure.
- Replaced that temporary default with the collision-checked table-aligned
  seed-0 approach yaw. Target position is still computed from the same
  episode's Qwen-selected RGB-D mask. Run 006 then completed bilateral contact
  and a verified lift, but its top-level wrapper timed out at `600 s` while
  the server was finishing the 174-frame video. The physics result and
  `server_result.json` are successful; the wrapper timeout is retained as
  diagnostic evidence.
- Increased the configurable terminal-grasp wait default to `1800 s` for a
  contended GPU. Clean run 007 completed in `882.68 s` on physical GPU 4:
  `center RGB-D -> GroundingDINO -> SAM2 -> Qwen -> belief/plan ->
  collision-checked close-high UR10e motion -> new RGB-D -> learned
  perception -> replan -> RGB-D dynamic IK -> bilateral RG6 contact -> lift`.
- Run-007 target localization came only from the selected anonymous mask and
  RGB-D, at `[0.71984, -0.20265, 0.06846] m`. No simulator ground truth was
  used for inference or target localization.
- The contact-gated grasp lifted the mug `0.17972 m`. Bilateral contact was
  present before and after lift; maximum measured forces were `7.30 N` left
  and `6.29 N` right, maximum penetration was `0.0555 mm`, horizontal slip
  was `0.0889 mm`, and unexpected environment collisions were zero. Object
  attachment and pose copying were not used.
- Qwen peak allocated memory was `17.0626 GiB`. Perception took `76.06 s` for
  center and `117.28 s` for close-high under shared-GPU load; Qwen model
  compute inside those stages took `15.76 s` and `34.34 s`. Exact-input Qwen
  outputs are cached as `result.json` plus `metadata.json` under
  `outputs/pilot_cache/grounded_qwen_factorized/<sha256>/`.
- After completion, GPU 4 showed `9050 MiB`, all belonging to the pre-existing
  shared jobs; this pipeline had exited and released its GPU allocation.
- Artifacts:
  `outputs/live_pipeline/learned_scanned_basket_e2e/seed000/run007/pipeline_result.json`,
  `outputs/live_pipeline/learned_scanned_basket_e2e/seed000/run007/persistent_grasp/result.json`,
  and
  `outputs/live_pipeline/learned_scanned_basket_e2e/seed000/run007/persistent_grasp/persistent_composite_grasp.mp4`.
- Boundary: this is one successful single-seed integration/physics pilot.
  Training was not performed. Calibration was not performed. The current
  planner explicitly reports `actual_mpc_solver=false`, and the seed-0 center
  Qwen score was already overconfident before re-observation. Therefore run
  007 does not validate uncertainty-driven action selection, calibrated
  belief, final belief-space MPC, multi-seed robustness, or paper performance.

## 2026-07-28 — Validate the first no-forced-reobservation belief-tree MPC run

- Added an exact discrete action-observation belief-tree solver. Unlike the
  prior fixed-sequence prototype, each predicted observation branch chooses
  its own next action. Unsafe predicted leaves defer instead of pretending
  that a grasp will execute, and the real posterior is replanned after the
  first action.
- Added
  `configs/research/scanned_basket_belief_tree_mpc_pilot.json` with horizon 2,
  `minimum_completed_reobservations=0`, and a provisional uncalibrated
  task-failure-risk limit of `0.06`. The solver reads no future captures.
- CPU tests verified normalized future observation branches, adaptive
  `grasp`/`defer` continuations, absence of the forced re-observation count,
  and non-oracle provenance. The focused suite passed `26/26`.
- Physical GPU 0 was verified empty immediately before execution. Seed-111
  live run 003 completed in `235.90 s` with one visible GPU, one model at a
  time, batch size one, and no distributed execution.
- At center, the current learned belief was target `0.94660` and inside
  `0.98006`, giving provisional task-failure risk `0.07228`. Immediate grasp
  was blocked only by task risk; there was no completed-reobservation count
  requirement.
- The MPC compared adaptive policy-tree values for right (`0.11352`) and
  close-high (`0.11719`) and selected `viewpoint_right`. This differs from the
  earlier forced close-high demonstration and follows the complete expected
  future cost, not confidence alone.
- The UR10e executed the right-view trajectory with no collision and maximum
  final joint error `0.00791 rad`. After the new RGB-D/GroundingDINO/SAM2/Qwen
  observation, the belief became target `0.99486`, inside `0.99713`, and task
  risk `0.007995`. The replanned action was then `grasp`.
- Contact grasp was intentionally disabled for this solver-validation run, so
  the external terminal action was `stop`. The separate seed-0 run 007 remains
  the validated contact-lift evidence.
- Perception runtime was `79.38 s` at center and `74.73 s` at right. Qwen
  model compute took `14.11 s` and `8.42 s`, peaking at `17.0309 GiB`. Both
  exact-input outputs were saved in the factorized Qwen cache. GPU 0 returned
  to `1 MiB`.
- Artifact:
  `outputs/live_pipeline/learned_scanned_basket_e2e/seed111/run003/pipeline_result.json`.
- Boundary: `actual_mpc_solver=true` now describes the discrete feedback
  solver implementation, not final scientific validation. Its observation
  likelihoods, Qwen tempering, and `0.06` risk limit are still hand specified
  and uncalibrated. This single deterministic run is not a final calibration,
  test-set result, baseline comparison, multi-seed statistic, or paper claim.

## 2026-07-28 — Run and reject the first Phase-1 calibration candidate

- Automatically generated four episode-disjoint calibration scenes, seeds
  `120–123`, with center, close-high, and right RGB-D observations. No manual
  annotations were created. Simulator instance ground truth was read only
  after GroundingDINO, SAM2, and Qwen inference.
- The bounded GPU-0 job completed all four captures and 12 learned-perception
  observations in `663.06 s`. Qwen used one instance and batch size one,
  averaged `8.70 s` model compute per view, and peaked at `17.0641 GiB`.
  Training, fine-tuning, LoRA, and testing were not performed. GPU 0 returned
  to `1 MiB`.
- GroundingDINO+SAM2 mask recall was `0.8889`, proposal precision was
  `0.8649`, mean mask IoU was `0.7903`, and mean RGB-D centroid error was
  `0.01298 m`. The right view consistently omitted one non-target instance.
- Qwen selected the correct target and `inside` relation in `12/12` views.
  Its selected target masks had mean IoU `0.9420` and mean centroid error
  `0.000655 m`.
- The calibration records contained 20 candidate proposals: 12 target
  positives and 8 non-target negatives. Target identity and relation
  classification were both perfect on this small easy batch.
- Temperature fitting collapsed to the lower grid boundary `0.25` for both
  target and relation. This would make the already overconfident scores more
  extreme rather than provide reliable uncertainty.
- Added an automatic deployment gate. It rejected the target fit because
  there are only four episodes, the solution is on the temperature-grid
  boundary, and there are no hard identity errors. It rejected the relation
  fit because only `inside/outside` are covered and `behind/unknown` are
  missing. Task-risk-gate and action-conditioned observation-model
  calibration are also explicitly disabled.
- The belief-tree MPC config records this rejected pilot for provenance but
  retains its previous uncalibrated debug temperature and `0.06` risk limit.
  No fitted value from this batch was applied.
- Artifacts:
  `outputs/calibration_pilot/scanned_basket_phase1_seed120_123_gpu0/calibration_fit.json`,
  `calibration_records.json`, `split_manifest.json`, and the `perception/`
  subdirectory.
- Boundary: this run validates automatic capture, inference, ground-truth
  separation, and calibration plumbing only. It is not an accepted
  calibration set or final result. The next generator must add harder
  identity cases and factorized membership/behind/occlusion/unknown coverage
  before fitting the final observation model or task-risk gate.

## 2026-07-28 — Validate four factorized calibration-scene variants

- Added a deterministic four-way calibration scene cycle:
  `inside_clear`, `outside`, `rim_occluded`, and `covered_unknown`.
  Seeds map to variants by `seed % 4`, so later episode generation remains
  reproducible and balanced without manual annotation.
- Separated physical world membership (`inside` or `outside`) from
  view-observable membership (`inside`, `outside`, or `unknown`) and the
  independent `behind` and `occluded_by` labels. A physically inside target
  can therefore correctly be `unknown` in a covered image.
- Added a simple collision-enabled cover that rests above the target and spans
  the basket opening. The existing arbitrary orange occluder remains hidden.
  Added explicit target support metadata so inside mugs use the scanned
  basket support surface and outside mugs use the table top.
- Added simulator-mask visibility measurements and an automatic smoke gate:
  easy variants require nonzero target pixels in every view; the rim variant
  requires a re-observation view to expose more pixels than center; the
  covered variant requires zero target pixels in all tested views.
- Physical GPU 0 was empty before execution. Four single-GPU RGB-D scene
  smokes completed with actual UR10e center, close-high, and right wrist
  views:
  - seed 124 `inside_clear`: `104.57 s`; target pixels
    `5981 / 18324 / 13165`.
  - seed 125 `outside`: `65.79 s`; target pixels
    `21241 / 13951 / 17545`.
  - seed 126 `rim_occluded`: `61.73 s`; target pixels
    `6339 / 17770 / 15343`; re-observation visibility gate passed.
  - seed 127 `covered_unknown`: `60.42 s`; target pixels
    `0 / 0 / 0`; hidden-target gate passed.
- Visual inspection confirmed that the target rests on the intended support,
  the outside target is clear of the basket, the rim creates partial rather
  than complete occlusion, and the cover hides the target without intersecting
  it. All four retrospective visibility-gate checks passed.
- Focused CPU regression tests passed `15/15`. GPU 0 returned to
  `1 MiB / 49140 MiB`.
- Artifacts:
  `outputs/live_pipeline/calibration_variant_smoke/inside_clear/seed124/run001/`,
  `outputs/live_pipeline/calibration_variant_smoke/outside/seed125/run001/`,
  `outputs/live_pipeline/calibration_variant_smoke/rim_occluded/seed126/run001/`,
  and
  `outputs/live_pipeline/calibration_variant_smoke/covered_unknown/seed127/run001/`.
- Boundary: these runs validate scene generation, RGB-D capture, camera
  reachability, and automatic labels only. Qwen, GroundingDINO, SAM2,
  calibration fitting, belief update, MPC, and grasp were not run here. They
  are not calibration results or final paper experiments. The next step is to
  update the calibration runner to consume this factorized schema, then run a
  bounded episode-separated 20-scene pretrained-inference calibration batch.

## 2026-07-28 — Complete the 20-scene factorized calibration pilot

- Updated the calibration pipeline so membership, `behind`, and
  `occluded_by` are separate forced-choice Qwen questions. The inference
  instruction now describes only the target appearance and does not leak the
  target's true relation. Ground-truth files remain outside Qwen input and are
  read only during post-hoc calibration/evaluation.
- Added candidate-level simulator GT for both the white-logo target and the
  no-logo red distractor. A covered target with no proposal is recorded as an
  observation-model miss; the pipeline does not invent a candidate or Qwen
  logit for it.
- A one-scene seed-128 preflight completed in `140.37 s`, verified the full
  factorized output schema, and measured `17.029 GiB` Qwen peak allocation.
  The focused CPU suite then passed `25/25`.
- The main GPU-0 job used seeds `128–147`, five seeds for each of
  `inside_clear`, `outside`, `rim_occluded`, and `covered_unknown`. All 20
  episodes and all 60 center/close-high/right RGB-D captures completed;
  failed captures and failed inference views were zero.
- Total wall time was `2505.06 s` (`41.75 min`), or an amortized
  `125.25 s/episode`. GroundingDINO took `40.71 s`, SAM2 `34.98 s`, Qwen input
  export `26.83 s`, Qwen ranking `970.09 s`, and post-hoc evaluation
  `10.93 s`.
- Qwen processed all 60 observations with one model instance and batch size
  one. Model compute totaled `942.00 s`, averaged `15.70 s/observation`, and
  peaked at `17.0862 GiB`. Training, fine-tuning, LoRA, distributed execution,
  and testing were not performed.
- GroundingDINO+SAM2 produced 219 proposals for 180 evaluated simulator
  instances. Mask recall at IoU 0.5 was `0.8389`, mean mask IoU was `0.7245`,
  and proposal precision at mask IoU 0.5 was `0.6895`. These numbers include
  covered scenes and all configured semantic instances, not only the target.
- Target proposal behavior matched the intended visibility split: all 45
  visible-target observations contained a target proposal, while all 15
  zero-pixel covered observations correctly had no target proposal. No
  artificial score was assigned to the latter.
- Among visible-target observations, Qwen selected the correct target in
  `41/45` views (`91.1%`): `inside_clear 15/15`, `rim_occluded 15/15`, and
  `outside 11/15`. All four selection failures occurred in the outside
  variant, where both red mugs were outside and the white-logo appearance
  cue had to distinguish them.
- Candidate-level identity classification used 93 records and achieved
  `0.8387` accuracy. All 45 target candidates received a positive identity
  logit, but 15 of 48 non-target candidates were also scored positive. The
  fitted temperature candidate was `5.85`, reducing NLL from `1.1948` to
  `0.3793` without changing accuracy.
- Factorized membership used 93 records and achieved `0.9462` accuracy. The
  fitted temperature candidate was `4.525`, reducing NLL from `1.1059` to
  `0.3794`. However, all five visible rim-center `unknown` membership labels
  were predicted as `inside`; temperature scaling softens confidence but
  cannot fix this argmax error.
- `behind` accuracy was `0.9140` with a `0.625` temperature candidate, and
  `occluded_by` accuracy was `0.9570` with a `0.85` candidate. Both fits were
  rejected because their scored candidate records contained only `yes/no`,
  not `unknown`. Covered targets were proposal-missing and therefore could
  not provide a legitimate Qwen factor logit.
- Corrected the post-hoc evaluator so outside and covered variants no longer
  inherit the obsolete fixed `inside` expected label. Corrected the deployment
  gate so accepted component-fit candidates are retained but no temperature
  is automatically applied to MPC while factor coverage, task-risk
  calibration, and the action-conditioned observation model remain
  incomplete.
- GPU 0 was shared with a separate approximately `0.36 GiB` process that was
  neither modified nor stopped. This job exited and released its approximately
  `17 GiB` allocation; only the unrelated small process remained.
- Artifacts:
  `outputs/calibration_pilot/factorized_seed128_147_gpu0/calibration_fit.json`,
  `calibration_records.json`, `perception/evaluation_summary.json`,
  `perception/grounded_sam2_qwen_ranking_summary.json`, and
  `outputs/live_pipeline/factorized_calibration_capture/`.
- Boundary: this is a successful calibration pilot, not final test-set or
  paper performance. The immediate next step is to add visible-but-ambiguous
  examples that legitimately produce `behind/occluded_by=unknown`, then
  complete the action-conditioned observation and task-risk calibration
  before changing the MPC configuration.

## 2026-07-28 — Complete the GPU-5 visible-ambiguity calibration extension

- Added a fifth calibration-only scene family, `behind_ambiguous`. The target
  remains detectable in the center image, but the basket hides its base and
  rear boundary so center-view membership and `behind` are not reliably
  observable. Close-high and right wrist observations expose the target as
  outside and behind the basket.
- The first visual prototype, seed 148 run 001, passed the pixel-count gate
  but was rejected by direct image inspection because the target was too
  obviously outside-left. It is retained only as a scene-design diagnostic.
  Seed 148 run 002 corrected the camera-ray alignment and basket clearance and
  became the accepted scene smoke.
- Added deterministic per-seed geometric jitter and a pure geometry gate.
  Across 1,000 checked seeds, every target kept at least `0.012 m` planar
  clearance from the basket while producing more than 900 distinct poses.
- Made the Qwen forced-choice prompts more conservative: projected image
  overlap is not evidence of containment, and a hidden base or rear boundary
  should permit `unknown`. The candidate-visible occlusion factor remains
  binary `yes/no`; a completely hidden target with no detector proposal is
  represented as observation missingness rather than an invented Qwen score.
- A single-seed GPU-5 Qwen preflight completed in `194.57 s`, averaged
  `12.98 s/view`, and peaked at `17.0367 GiB`. The new center image remained
  a hard case: Qwen chose `outside` rather than membership `unknown`, chose
  `behind=yes` rather than `unknown`, and chose `occluded_by=no` rather than
  `yes`. The unknown score gap nevertheless narrowed enough to justify using
  the scene as calibration evidence rather than discarding the failure.
- Ran the combined episode-separated calibration batch on physical GPU 5
  only: seeds `128–147` from the four existing families plus seeds `148–155`
  from `behind_ambiguous`. All `28/28` episodes and `84/84`
  center/close-high/right observations completed with zero capture or
  inference runtime failures.
- Total wall time was `2009.75 s` (`33.50 min`), or an amortized
  `71.78 s/episode`. GroundingDINO took `198.56 s`, SAM2 `99.08 s`, Qwen input
  export `33.54 s`, Qwen ranking `921.31 s`, and post-hoc evaluation
  `10.23 s`.
- Qwen used one pretrained `Qwen3-VL-8B-Instruct` instance, batch size one,
  no distributed execution, and no training/fine-tuning/LoRA. Model compute
  totaled `855.60 s`, averaged `10.19 s/observation`, and peaked at
  `17.0926 GiB`.
- The additional ambiguity family supplied 24 visible observations. It
  produced a target proposal in 22 and selected the correct target in 20.
  Across the combined batch, the proposal model recorded 84 observations:
  69 had visible target pixels, 67 had a target proposal, two were visible
  detector misses, and 15 were intentionally covered zero-pixel observations.
- The target-identity component used 139 candidate records, reached `0.8561`
  uncalibrated accuracy, and produced a temperature candidate of `5.125`
  (`NLL 0.8851 -> 0.3202`). It passed its component-level calibration gate but
  was not applied to MPC.
- Aggregate relation accuracies concealed critical minority-class failures.
  Membership had `unknown` recall `0/13`; `behind` had `unknown` recall `0/8`;
  and `occluded_by` had `yes` recall `0/13`. Temperature scaling can soften
  confidence but cannot repair these wrong argmax predictions.
- Added a per-class deployment gate. It now rejects any supported required
  class with zero recall. Consequently membership is blocked by
  `zero_recall_labels:unknown`, `behind` by
  `zero_recall_labels:unknown`, and `occluded_by` by
  `zero_recall_labels:yes`. No relation temperature, task-risk threshold, or
  action-conditioned observation model was written into the MPC
  configuration.
- The relevant CPU regression suite passed `28/28`. After completion,
  physical GPU 5 showed `1 MiB / 49140 MiB`, `0%` utilization, and no compute
  process. No other physical GPU was queried or used for this experiment.
- Artifacts:
  `outputs/calibration_pilot/factorized_plus_ambiguous_seed128_155_gpu5/calibration_fit.json`,
  `calibration_records.json`,
  `perception/evaluation_summary.json`, and
  `perception/grounded_sam2_qwen_ranking_summary.json`. Scene captures are
  under `outputs/live_pipeline/factorized_calibration_capture/`.
- Boundary: execution and calibration plumbing succeeded, but the relation
  result is a scientifically useful failure, not a deployable calibration or
  final-paper result. Training was not performed, calibration used only these
  calibration episodes, and final testing was not performed. The next method
  step is a hybrid observation model: retain Qwen for target identity while
  using RGB-D/mask/basket geometry and visibility evidence for containment,
  occlusion, and abstention. That model must then calibrate
  action-conditioned observations and task risk before final MPC evaluation.

## 2026-07-29 — Complete the first hybrid RGB-D relation audit

- Added a calibration-only hybrid adapter,
  `scripts/run_hybrid_rgbd_relation_pilot.py`, with a frozen engineering
  configuration at
  `configs/perception/hybrid_rgbd_relation_pilot.json`.
- The prediction pass consumes only learned candidate/reference SAM masks,
  metric depth, camera calibration, and declared basket/mug dimensions.
  Simulator instance IDs and relation labels are read only by a separate
  post-hoc audit pass.
- The output separates three quantities that the earlier Qwen-only relation
  output conflated:
  - geometric world evidence for `inside/outside`;
  - camera-relative `behind` evidence;
  - observation quality from visible object height, basket-mask adjacency,
    invalid reference geometry, and proposal missingness.
- Reused the 28 calibration episodes and 84 observations from seeds
  `128–155`; no new model inference or simulator capture was required. The
  audit evaluated 139 learned candidate masks.
- RGB-D world-membership prediction was correct on every non-abstained
  candidate: `137/137` selective accuracy with `137/139` coverage. All 30
  inside candidates and 107 outside candidates were classified correctly.
  Two outside candidates were safely returned as `unknown` because a
  cover-contaminated basket mask produced an implausible reference extent.
- The 13 candidates labeled `membership=unknown` for the RGB-only Qwen audit
  were all resolved by metric RGB-D geometry to the correct world relation:
  five were inside and eight were outside. This is not an unknown-class
  failure of the hybrid adapter; it shows that the old label described
  RGB-only observability rather than the evidence available to the final
  RGB-D pipeline.
- Camera-relative geometry recovered all 19 legacy `behind=yes` examples,
  compared with Qwen's 4/19. It also resolved all eight legacy
  `behind=unknown` examples as geometrically behind. These disagreements are
  retained rather than counted as deployable accuracy because the two systems
  consume different modalities.
- The visibility/adjacency rule recovered all 13 legacy
  `occluded_by=yes` examples, whereas Qwen recovered 0/13. However, it also
  disagreed with 35 legacy occlusion labels in total: 12 old `no` examples
  were geometrically flagged `yes`, and 23 were returned `unknown`. Direct
  image review confirmed that some `inside_clear` mugs have their lower body
  hidden by the basket despite being authored as `occluded_by=no`.
- Consequently the hybrid adapter is not applied to MPC. Blocking reasons
  include calibration-only reuse, an axis-aligned simulation reference frame,
  unresolved objective occlusion-label semantics, no fitted
  action-conditioned observation model, and no calibrated task-risk gate.
- Reserved seeds `200–209` remain untouched. Training, fine-tuning, LoRA,
  probability calibration, and final testing were not performed.
- The relevant CPU regression suite passed `33/33`. This experiment was
  CPU-only; no GPU process was launched and no physical GPU other than the
  permitted GPU-5 status check was needed.
- Artifacts:
  `outputs/calibration_pilot/factorized_plus_ambiguous_seed128_155_gpu5/hybrid_rgbd_relation/summary.json`,
  `predictions.json`, and `audit_rows.csv`.
- Immediate next requirement: define an objective, modality-aware occlusion
  label from measurable visible fraction/line-of-sight geometry and separate
  it from RGB-only semantic abstention. Only after that definition is frozen
  should a new episode-separated validation batch be captured and the hybrid
  observation likelihood connected to belief-tree MPC.

## 2026-07-29 — Freeze and validate objective amodal occlusion ground truth

- Added a simulator-only counterfactual target render in
  `scripts/observation_capture.py`. For each benchmark observation it now
  preserves the camera and target pose, hides non-target rendered geometry,
  and saves `target_amodal_mask.png` beside the normal visible target mask.
- The pilot metric is
  `1 - pixels(actual target-ID mask ∩ target-only amodal support) /
  pixels(target-only amodal support)`. The intersection is necessary because
  the temporary emissive color-ID pass can classify red table reflections as
  target pixels. Raw target-ID pixels are preserved in
  `target_visible_mask_raw.png`, and out-of-amodal spill is recorded rather
  than silently counted as object area.
- Froze method-development severity thresholds before the validation audit:
  `no < 0.10`, `partial ∈ [0.10, 0.60)`, and `severe ≥ 0.60`.
  A fully hidden target is recorded separately. These are pilot engineering
  thresholds, not calibrated probabilities or final-paper thresholds.
- Captured validation seeds `156–164` on physical GPU 5 only: two
  `inside_clear`, two `outside`, two `rim_occluded`, two
  `covered_unknown`, and one `behind_ambiguous` episode. All `9/9` episodes
  and `27/27` center/close-high/right observations completed; failed episodes
  were zero. Capture runtime was `892.84 s` (`14.88 min`).
- The objective distribution was `no=11`, `partial=10`, `severe=6`,
  `unknown=0`. Both covered episodes were fully hidden in all six views with
  occlusion fraction `1.0`. Rim-center occlusion was approximately
  `0.397–0.407`, while its reachable re-observation views fell to
  approximately `0.009–0.065`.
- The audit exposed the old authored-label error directly. Nominal
  `inside_clear` center observations were actually about `0.504–0.560`
  occluded by the basket, while close-high/right were about
  `0.010–0.016`. The legacy `occluded_by=no` label is therefore retained only
  for historical comparison and is not used as the new target occlusion GT.
- Added `scripts/run_objective_occlusion_validation.py` and
  `scripts/recompute_objective_occlusion_outputs.py`. The former writes an
  episode manifest so the saved RGB-D observations can be reused without
  rerendering; the latter applies metric-only revisions while preserving raw
  masks. Simulator masks and objective labels remain post-hoc evaluation data
  and are never supplied to GroundingDINO, SAM2, Qwen, belief update, or MPC.
- Artifacts:
  `outputs/live_pipeline/objective_occlusion_validation_seed156_164_gpu5/summary.json`,
  `objective_occlusion_rows.csv`, `capture_manifest.json`, and per-view
  `objective_occlusion.json` files.

## 2026-07-29 — Validate learned perception against objective occlusion GT

- Reused the nine saved episodes; Isaac Sim was not launched again. Ran
  GroundingDINO, SAM2, and one pretrained Qwen3-VL-8B instance sequentially
  on physical GPU 5. No training, fine-tuning, LoRA, distributed execution,
  or final testing was performed.
- All inference stages completed in `575.17 s`: GroundingDINO `96.21 s`,
  SAM2 `50.62 s`, Qwen input export `11.19 s`, Qwen ranking `413.24 s`, and
  post-hoc evaluation `3.72 s`. Qwen processed 27 observations in
  `369.31 s`, averaged `13.68 s/observation`, and peaked at `17.0919 GiB`.
- A separate user process already occupied approximately `7.5 GiB` on GPU 5.
  It was identified as another user's `train.py`, was not stopped or
  modified, and no other physical GPU was queried or used. The Codex Qwen
  process exited and released its approximately `18.5 GiB`; only the
  unrelated process remained.
- GroundingDINO+SAM2 evaluated 81 simulator instances. Mask recall at IoU 0.5
  was `0.8642`, mean mask IoU was `0.7492`, and proposal precision at mask
  IoU 0.5 was `0.6863`. All 21 observations with visible target pixels
  produced a target proposal; all six fully covered observations correctly
  produced no target proposal.
- Qwen selected the correct target in `19/21` visible-target observations
  (`90.5%`). The six fully covered views are observation-missingness outcomes,
  not target-selection errors with invented candidate scores.
- The candidate-level target calibration audit used 44 records, reached
  `0.8409` accuracy, and produced a temperature candidate of `5.45`
  (`NLL 1.0573 -> 0.3729`). It is rejected for deployment because only nine
  episode-disjoint calibration scenes were used.
- Objective target occlusion supplied 21 legitimate candidate records:
  `10 yes` and `11 no`. Qwen predicted every record as `no`, giving
  `11/21 = 52.4%` accuracy and `0/10` yes recall. The fitted temperature hit
  the upper grid boundary `8.0`; it is rejected.
- The RGB-D hybrid rule improved objective occlusion to `16/21 = 76.2%`
  accuracy, `20/21` coverage, and `80.0%` selective accuracy, but recovered
  only `5/10` objective yes cases. RGB-D world membership reached
  `43/44 = 97.7%`; its single abstention gave `43/43` selective accuracy.
- No fitted score or rule was applied to MPC. Blocking issues are insufficient
  episode count, only `50%` objective-occlusion yes recall, uncalibrated
  occlusion thresholds, an axis-aligned simulation reference assumption, no
  action-conditioned observation likelihood, and no calibrated task-risk
  gate.
- Reserved test seeds `200–209` remain untouched. These results are a
  calibration/method-validation pilot and must not be reported as final
  paper performance or a statistical guarantee.
- Artifacts:
  `outputs/calibration_pilot/objective_occlusion_seed156_164_gpu5/calibration_fit.json`,
  `perception/evaluation_summary.json`,
  `perception/grounded_sam2_qwen_ranking_summary.json`, and
  `hybrid_rgbd_relation/summary.json`.

## 2026-07-29 — Correct occlusion attribution and fit a 20-episode action model

- Corrected a semantic mismatch in the first objective audit. The target-only
  amodal pass measures occlusion by the entire scene, while the method factor
  is specifically `occluded_by_reference`. The capture pipeline now saves a
  second counterfactual pass that hides only `/World/OpenContainer` and counts
  target pixels newly revealed by removing that reference.
- Both measurements are retained. Total-scene occlusion remains a visibility
  diagnostic, while reference-attributed occlusion is the calibration target
  for `occluded_by_reference`. The same frozen pilot thresholds are used:
  `no < 0.10`, `partial ∈ [0.10, 0.60)`, and `severe ≥ 0.60`.
- Captured and validated seeds `165–184` as 20 episode-disjoint calibration
  scenes: four `inside_clear`, five `outside`, four `rim_occluded`, five
  `covered_unknown`, and two `behind_ambiguous`. All `20/20` episodes and
  `60/60` center/close-high/right observations completed with zero scene
  failures. Reserved test seeds `200–209` remain untouched.
- Reference-attributed severity over the 60 observations was `no=32`,
  `partial=13`, `severe=15`, and `unknown=0`. Outside close-high views showed
  substantial total-scene occlusion from unrelated foreground geometry but
  nearly zero reference-attributed occlusion, confirming that the new metric
  does not incorrectly blame the basket.
- Reused the saved observations for GroundingDINO, SAM2, and one pretrained
  Qwen3-VL-8B instance on physical GPU 5. No simulator was launched for this
  inference pass. The complete learned pipeline finished in `997.51 s`;
  Qwen processed 60 observations in `606.31 s`, averaged
  `10.11 s/observation`, and peaked at `17.0919 GiB`. Batch size was one,
  model instances were not duplicated, and no training, LoRA, distributed
  execution, or final testing was performed.
- The target had visible pixels and a learned proposal in `45/45`
  observations; all 15 fully covered observations produced proposal
  missingness rather than a fabricated candidate. Qwen selected a mask with
  target IoU at least 0.5 in `40/45` visible-target observations.
- Qwen-only reference occlusion again failed the minority class: it predicted
  `no` for all 45 visible-target records, giving `0/13` yes recall and
  `32/45 = 71.1%` accuracy. Temperature scaling cannot correct this argmax
  failure.
- The frozen RGB-D hybrid rule reached `41/45 = 91.1%` overall accuracy and
  `41/45 = 91.1%` coverage against reference-attributed objective GT. It
  recovered `10/13` yes cases and `31/32` no cases; all 41 non-abstained
  predictions were correct. RGB-D world membership reached `94/97 = 96.9%`
  coverage with `94/94` selective accuracy.
- Fitted a first action-conditioned observation and state-transition table
  from the 20 calibration episodes: 60 observations and 40 center-to-new-view
  transitions. Every declared condition cell now has at least five episodes,
  so sparse-cell count is zero. Among 15 initially reference-occluded scenes,
  close-high removed the reference occlusion in nine and right removed it in
  eight. All five fully covered targets remained hidden under either
  viewpoint action, correctly distinguishing viewpoint change from future
  cover removal.
- The action-conditioned table is not deployed to MPC. It is supported only
  by the current discrete scene families, task-risk calibration is not
  complete, and unbiased final testing has not been performed.
- A GPU-policy issue was also identified from the user's process monitor:
  seed 180 Isaac Sim PID `3823469` used physical GPU 5 for RTX/physics but
  also created a `4 MiB` graphics context on physical GPU 0 despite
  `CUDA_VISIBLE_DEVICES=5`, `renderer/activeGpu=5`, `physics cuda:0`,
  multi-GPU disabled, and `--no-window`. The process has ended, and GPU 5
  subsequently showed `1 MiB`, `0%`, with no compute process. Because any
  access to a non-5 GPU violates the current policy, further Isaac Sim
  launches are suspended until Vulkan/graphics isolation can be verified.
  Saved-data CPU analysis and GPU-5-only CUDA inference may continue.
- The relevant CPU regression suite passed `21/21`.
- Primary artifacts:
  `outputs/live_pipeline/objective_reference_occlusion_validation_seed165_173_gpu5/summary.json`,
  `outputs/live_pipeline/objective_reference_occlusion_calibration_extension_seed174_184_gpu5/summary.json`,
  `outputs/calibration_pilot/reference_occlusion_seed165_184_gpu5/calibration_fit.json`,
  `outputs/calibration_pilot/reference_occlusion_seed165_184_gpu5/hybrid_rgbd_relation/summary.json`,
  and
  `outputs/calibration_pilot/reference_occlusion_seed165_184_gpu5/action_conditioned_observation_likelihood/fit.json`.

## 2026-07-29 — Run the first CPU-only action-conditioned MPC replay

- Added `scripts/run_offline_action_conditioned_mpc_replay.py` and
  `configs/research/offline_action_conditioned_mpc_replay_seed165_184.json`.
  The runner launches neither Isaac Sim nor any learned model. It reuses the
  cached GroundingDINO, SAM2, Qwen, RGB-D, and hybrid-relation outputs from
  seeds `165–184`.
- Used leave-one-episode-out calibration replay. When seed `s` is replayed,
  its action-conditioned observation and transition model is fitted from the
  other 19 seeds. The held-out close-high/right observation is not an input
  to root action selection; it is applied only after the policy fixes that
  action. Simulator labels are used only for the 19-seed fold fit and the
  held-out post-hoc audit.
- Replaced an invalid factorization found in the first run. Target confidence
  and reference occlusion were initially updated independently, causing a
  high-confidence unoccluded outside view to retain approximately 62%
  occlusion belief. Visibility and reference occlusion are now represented
  as one joint latent perception state.
- Added cross-view candidate tracking from nearest learned RGB-D candidate
  centers with a `0.12 m` gate. Simulator IDs are not used for association.
  A new view updates the existing target track belief using track agreement
  and the tracked candidate's Qwen score rather than replacing the strong
  center evidence with the newly selected candidate.
- Preserved the initial invalid-factorization outcome at
  `outputs/offline_mpc/action_conditioned_replay_seed165_184/result_factorized_state_failure.json`.
  It produced 0 correct grasps, 3 wrong grasps, and 17 defers, and is retained
  as a diagnostic rather than relabeled or deleted.
- Final primary replay at the frozen provisional noncompletion cost `0.45`
  produced:
  - immediate grasp: 15 correct, 5 wrong, 0 defer;
  - fixed close-high then grasp: 12 correct, 8 wrong;
  - fixed right then grasp: 13 correct, 7 wrong;
  - confidence-only fixed re-observation: 13 correct, 2 wrong, 5 defer;
  - action-agnostic belief MPC: 15 correct, 0 wrong, 5 defer;
  - action-conditioned belief MPC: 11 correct, 0 wrong, 9 defer.
- The action-conditioned method's four additional defers were visible
  inside/rim targets whose initial target-track probability was `0.8`. The
  policy estimated view-plus-terminal cost at approximately `0.465–0.473`,
  slightly above the provisional `0.45` noncompletion cost. This is
  task-cost sensitivity, not a perception or runtime crash.
- Ran a declared noncompletion-cost sensitivity grid rather than silently
  tuning the primary result. At `0.35` the policy deferred all 20 episodes;
  at `0.45` it returned the primary 11-correct/9-defer result; and at
  `0.55`, `0.65`, and `0.8` it produced 15 correct grasps, zero wrong grasps,
  and five safe defers for fully hidden targets, averaging 1.5 observations.
  No value from this calibration collection is selected as the final
  task-risk cost.
- The action-agnostic ablation matched the best sensitivity outcome under the
  primary `0.45` setting. Therefore this pilot does not show an advantage for
  action conditioning. The current scene families are too simple: both
  reachable views usually preserve the correct tracked target, and no
  cover-removal action is available.
- Runtime was under one second per replay invocation on CPU. GPU use and GPU
  memory were exactly zero; no physical GPU was queried or used. Training,
  fine-tuning, LoRA, new VLM inference, final testing, and paper-scale
  evaluation were not performed.
- The expanded relevant CPU regression suite passed `37/37`.
- Artifacts:
  `outputs/offline_mpc/action_conditioned_replay_seed165_184/summary.json`,
  `result.json`, and `result_factorized_state_failure.json`.
- Next scientific requirement: calibrate task noncompletion and wrong
  commitment on a separate calibration/validation protocol, add scenes in
  which close-high and right have meaningfully different future observations,
  and add a cover-removal action before using frozen test seeds.

## 2026-07-29 — Run nested task-cost calibration and diagnose view-action support

- Added
  `scripts/run_nested_action_conditioned_mpc_calibration.py`,
  `configs/research/nested_action_conditioned_mpc_calibration_seed165_184.json`,
  and focused CPU tests. The experiment reused only the cached seed
  `165–184` observations and launched no simulator, VLM, grounding model, or
  GPU process.
- Split the 20 calibration episodes into four disjoint outer folds of five
  episodes. For each outer fold and each MPC variant, the noncompletion cost
  was selected from `0.35, 0.45, 0.55, 0.65, 0.8` by leave-one-episode-out
  replay inside the remaining 15 episodes. The five outer held-out episodes
  were excluded from both cost selection and action-model fitting.
- The action-agnostic model selected `0.55` in three folds and `0.65` in one.
  Its outer-fold aggregate was 15 correct grasps, zero wrong grasps, and five
  safe defers, all for fully hidden targets.
- The action-conditioned model selected `0.65` in two folds and `0.8` in two.
  Its outer-fold aggregate was 12 correct grasps, zero wrong grasps, and eight
  defers. Five were safe fully-hidden defers, while seeds `165`, `167`, and
  `178` were visible targets that the policy unnecessarily deferred.
- Audited whether the cached views actually support learning different action
  effects. Close-high and right produced different learned observation
  signatures in `10/20` episodes, but their post-hoc physical latent state
  differed in only `1/20`. The maintained target track had identical post-hoc
  correctness under both views in all `20/20` episodes. Much of the apparent
  action difference is therefore perception variability rather than a
  controlled physical consequence of the selected viewpoint.
- This nested result strengthens the negative diagnostic: the current scenes
  do not demonstrate an action-conditioning advantage. They are unsuitable
  for claiming the proposed belief-space MPC benefit, even though they remain
  useful for pipeline debugging.
- Runtime was `9.51 s` on CPU and GPU memory was `0 GiB`. No continuous UR10e
  motion MPC, Isaac Sim execution, new VLM inference, or grasp occurred.
  Training and final testing remain unperformed, and reserved test seeds
  `200–209` remain untouched.
- The expanded relevant regression suite passed `41/41`, and
  `git diff --check` passed.
- Artifacts:
  `outputs/offline_mpc/nested_action_conditioned_mpc_calibration_seed165_184/result.json`
  and `summary.json`.
- Next experimental requirement: author balanced action-differentiating
  calibration scenes in which close-high uniquely resolves some cases, right
  uniquely resolves others, both resolve some, and neither viewpoint can
  resolve covered cases. Add cover removal as a distinct interaction action
  before freezing costs or starting the reserved test split.

## 2026-07-29 — Prepare balanced action-differentiating scenes without GPU use

- Extended `scripts/scanned_basket_scene.py` with four causal
  action-outcome variants:
  `close_high_only`, `right_only`, `either_view`, and
  `cover_removal_required`.
- The first three variants place the target inside the scanned basket behind
  the rim. The one-view variants additionally place the existing orange
  occluder along the reference ray of the view that should remain
  uninformative. The covered variant retains zero resolving viewpoint actions
  and declares `remove_cover` as the required future interaction.
- The reference camera centers are derived from the saved successful seed-169
  camera calibrations after undoing the benchmark environment shift. They are
  used only to initialize geometry. The intended generator labels are not
  accepted as rendered truth.
- Added objective-mask acceptance gates. A resolving view must expose at
  least 65% of the target amodal support and improve at least 15 percentage
  points over center. A one-view winner must exceed the other view by at least
  15 points. The covered case must remain at or below 2% visibility in all
  three viewpoint observations. Scenes that fail are rejected rather than
  relabeled.
- Added
  `configs/research/action_differentiating_scene_pilot.json` and
  `scripts/prepare_action_differentiating_scene_manifest.py`. The GPU-free
  manifest assigns development seeds `185–196` evenly: three scenes per
  variant. Reserved test seeds `200–209` remain untouched.
- Added the new variants to the existing Isaac scene and live-view smoke
  command-line interfaces, but did not launch either command. The manifest
  records the future seed-specific capture commands and marks every scene
  `render_status=not_run_gpu_unavailable`,
  `render_validation_passed=null`, and
  `eligible_for_calibration=false`.
- No Isaac Sim, GPU, VLM, grounding, calibration, training, or test execution
  occurred. This is scene-authoring preparation, not an experimental success
  result. The expanded CPU regression suite passed `53/53`, and
  `git diff --check` passed.
- Artifact:
  `outputs/scene_design/action_differentiating_pilot_seed185_196/manifest.json`.
- When compliant GPU isolation is restored, run only seed `185`
  (`close_high_only`) first. Inspect physical placement and apply the
  objective-mask gate before running the other 11 development scenes. Do not
  start calibration or reserved testing until all four causal classes pass.

## 2026-07-29 — Add CPU-only remove-cover belief MPC and negative evidence

- Added `scripts/run_cover_search_belief_mpc.py`,
  `configs/research/cover_search_belief_mpc_cpu_pilot.json`, and focused
  tests. This is a separate high-level exact discrete belief-tree controller;
  the existing viewpoint/grasp planner remains backward compatible.
- The joint discrete state represents target location
  (`inside` or `outside_near`) and cover state (`covered` or `open`).
  Candidate actions are `viewpoint_close_high`, `viewpoint_right`,
  `remove_cover`, `grasp_inside`, `grasp_outside`, and `defer`.
- `remove_cover` predicts a covered-to-open transition and branches over
  `target_detected`, `empty_container`, and `action_failed`. The planner
  compares complete feedback continuations over horizon three, executes only
  the first scripted diagnostic action, applies the observation afterward,
  and replans.
- Three non-oracle scripted control-flow diagnostics completed:
  - positive evidence: `remove_cover -> grasp_inside`, with final inside
    belief `0.9826`;
  - negative evidence: `remove_cover -> grasp_outside`, with inside belief
    reduced from `0.65` to `0.1210`;
  - interaction failure:
    `remove_cover -> remove_cover -> grasp_outside`, showing retry after the
    failure observation.
- The negative-evidence ablation intentionally made `empty_container`
  nondiscriminative for target location. Its inside belief remained `0.65`,
  so it required
  `remove_cover -> viewpoint_close_high -> grasp_outside`, one more
  observation than the proposed update. Both paths eventually selected the
  same terminal region after the ablation received its extra outside
  evidence.
- The first ablation run stopped at the requested extra viewpoint because its
  scripted observation had not been provided. That diagnostic was preserved
  conceptually, the missing post-action observation was added, and the fair
  rerun completed. No future scripted observation or post-hoc true state was
  available to action selection.
- Runtime was about `0.007 s` on CPU. GPU use and GPU memory were zero. No
  Isaac Sim, VLM, continuous UR10e motion MPC, RG6 contact, or cover
  manipulation was executed. Transition probabilities, observation
  likelihoods, commitment threshold, and task costs are hand-specified debug
  values and are not calibrated or final-paper parameters.
- The expanded relevant CPU regression suite passed `69/69`, and
  `git diff --check` passed.
- Artifacts:
  `outputs/offline_mpc/cover_search_belief_mpc_cpu_pilot/result.json` and
  `summary.json`.
- Next connection point after GPU access returns: execute the authored
  removable-cover scene, replace scripted observations with rendered
  GroundingDINO/SAM/RGB-D/Qwen evidence, and connect the selected
  `remove_cover` request to a collision-checked UR10e/RG6 manipulation
  primitive. The current module validates belief-space logic only.

## 2026-07-29 — Integrate the Scene Graph and remove-cover MPC contract on CPU

- Added
  `configs/research/cover_search_scene_graph_mpc_integration.json`,
  `scripts/run_cover_search_scene_graph_mpc_integration.py`, and focused
  integration tests.
- Extended the provisional Scene Graph with an optional
  `graph_belief.joint_task_state_distribution`. This preserves the exact
  correlation between target location and cover state; reconstructing the
  planner state from independent marginals would lose information.
- Connected the complete high-level interface:
  Scene Graph belief -> graph adapter -> exact discrete belief MPC ->
  SHA-256-bound first-action request -> post-action result -> Scene Graph
  update -> replanning.
- All three scripted diagnostic episodes completed through this interface:
  - positive evidence:
    `remove_cover -> grasp_inside`;
  - empty-container negative evidence:
    `remove_cover -> grasp_outside`;
  - failed cover action:
    `remove_cover -> remove_cover -> grasp_outside`.
- Each action request records the source Scene Graph hash. The CPU executor
  rejects an unexpected action, returns evidence only after the request, and
  never exposes the scripted true state or a future observation to the
  planner.
- The run completed in about `0.012 s` with GPU use and GPU memory both zero.
  It did not launch Isaac Sim, Qwen, GroundingDINO, or SAM. RGB-D paths are
  explicitly marked unavailable, and no UR10e motion, RG6 contact, or
  physical cover manipulation occurred.
- This is a software-interface and belief-control-flow success only. Its
  scripted executor, hand-specified likelihoods, and uncalibrated costs are
  not final experimental evidence, calibration, or testing.
- JSON Schema validation passed for the existing example and generated
  integration graphs. The complete CPU unit-test discovery passed `129/129`,
  and `git diff --check` passed.
- Artifacts:
  `outputs/offline_mpc/cover_search_scene_graph_mpc_cpu_integration/result.json`,
  `summary.json`, and per-episode Scene Graph, action-request, and
  action-result JSON files.
- Next physical connection point: retain this contract but replace the CPU
  result stub with evidence generated after a live rendered action. The
  selected `remove_cover` request must then be mapped to a collision-checked
  UR10e/RG6 primitive before any claim of physical or end-to-end success.

## 2026-07-29 — Compile the first guarded remove-cover manipulation primitive

- Audited the existing calibration cover before connecting it to robot
  execution. It was a flat static collider with no RG6 grasp affordance, so it
  could not support a physically meaningful cover-removal attempt.
- Preserved that static cover for the existing `covered_unknown` calibration
  scenes. Only the `cover_removal_required` action-development scene now
  authors a separate dynamic rigid-body assembly with a plate and an
  `80 x 30 x 35 mm` RG6 pinch handle. Its provisional mass is `0.20 kg`.
- Added
  `configs/research/remove_cover_primitive_cpu_contract.json`,
  `scripts/plan_remove_cover_primitive.py`, and focused CPU tests.
- The compiler accepts only a `remove_cover` first-action request targeting
  `cover_01`, verifies the exact source Scene Graph SHA-256 and episode, and
  rejects future-observation leakage or a mismatched action.
- The valid plan contains eight guarded Cartesian phases in the basket frame:
  pregrasp, handle descent, contact-controlled closure, vertical lift,
  transfer to staging, supported lowering, release, and retreat followed by a
  new RGB-D request.
- Static CPU geometry checks passed. The side-staging cover AABB has
  `0.1105 m` horizontal clearance from the basket, and the lifted cover bottom
  has `0.164 m` vertical clearance over the basket before horizontal
  transfer.
- Live-execution gates are declared but not fabricated: finite joint state,
  IK success, continuous collision checking, maximum `0.05 rad` arm error,
  bilateral handle contact, at least `3 N` grip force per side, at most
  `60 N` per side, at most `3 mm` penetration, at most `30 mm` horizontal
  slip, no unexpected environment contact, and stable staging support.
- Every monitored failure maps to the high-level `action_failed` observation.
  A successful cover move still requires a new RGB-D observation before the
  executor may classify `target_detected` or `empty_container`; the plan does
  not invent that evidence.
- The CPU compilation completed in about `0.00047 s`; GPU use and memory were
  zero. No Isaac Sim, IK solver, joint trajectory, narrow-phase collision
  query, RG6 contact, cover motion, VLM, or belief update ran.
- The complete CPU unit-test discovery passed `135/135`, and
  `git diff --check` passed.
- Artifacts:
  `outputs/offline_mpc/remove_cover_primitive_cpu_contract/remove_cover_plan.json`
  and `summary.json`.
- Current result status is
  `ready_for_live_ik_and_physics_validation`, not physical success. When
  compliant Vulkan/GPU isolation is restored, the dynamic assembly must first
  pass visual/support inspection and a seed-185 home/pregrasp IK and collision
  smoke before contact or lift is attempted.

## 2026-07-30 — Cross-validate cached perception calibration without rerunning models

- Audited the complete seed `165–184` perception cache before using the newly
  available GPUs. All `60` observations already had GroundingDINO detections,
  SAM2 masks, and Qwen target/relation outputs, so no image was processed
  again. Isaac Sim, Omniverse, Vulkan, CUDA, and every physical GPU remained
  unused.
- Added an episode-disjoint four-fold temperature-scaling diagnostic using
  exactly the outer seed folds already declared for the nested MPC
  calibration. Each held-out episode was excluded from the temperature fit
  applied to it.
- Qwen target-identity calibration generalized within the calibration
  collection: held-out aggregate NLL improved from `1.1884` to `0.4023`,
  Brier score from `0.3883` to `0.2668`, and ECE from `0.1957` to `0.0946`.
  Fold temperatures were `5.025`, `5.875`, `6.25`, and `6.125`.
- Relation temperature scaling was not deployable. Membership NLL improved,
  but Brier and ECE worsened and `unknown` recall remained zero. Behind NLL
  worsened from `0.7240` to `0.7396`, with zero `unknown` recall.
  `occluded_by` NLL improved from `1.8317` to `0.6460`, but the model still
  recovered `0/13` positive reference-occlusion cases. Temperature scaling
  softened confidence but could not repair the missing class discrimination.
- This result supports retaining Qwen as the target-identity evidence source
  while keeping RGB-D geometry/hybrid evidence as the relation path. It does
  not freeze a final temperature or deploy any calibration into MPC because
  this was cross-validation inside the calibration collection, not final
  testing. Task-risk cost, action-conditioned observation behavior, and the
  reserved test split remain unresolved.
- Added
  `configs/research/cached_perception_calibration_cv_seed165_184.json`,
  `scripts/run_cached_perception_calibration_cv.py`, and focused tests.
  Artifacts are
  `outputs/calibration_pilot/reference_occlusion_seed165_184_gpu5/cross_validation/result.json`
  and `summary.json`.
- Runtime was about `0.57 s` on CPU with `0 GiB` GPU memory. The complete
  CPU/perception unit-test discovery passed `139/139`, and
  `git diff --check` passed.

## 2026-07-30 — Cross-validate hybrid RGB-D relation likelihoods on cached episodes

- Reused the saved seed `165–184` hybrid-relation audit table. No image was
  reprocessed and Isaac Sim, Vulkan, CUDA, and all physical GPUs remained
  unused.
- Added four-fold episode-disjoint calibration of the categorical observation
  likelihood `P(hybrid evidence | true relation)`. Each fold fit on 15 seeds
  and evaluated on five unseen calibration seeds, using the same fold
  partition as the target-identity and MPC calibration diagnostics.
- The existing hard hybrid membership rule remained correct on `94/97`
  records (`96.91%`) and abstained on three outside cases. The cross-validated
  likelihood posterior obtained NLL `0.0611`, Brier `0.0178`, ECE `0.0548`,
  and classified `97/97` records correctly under the diagnostic uniform
  prior. This is a provisional Scene Graph observation-model candidate.
- For objective target `occluded_by`, the hard hybrid evidence was correct on
  `41/45` records (`91.11%`), with `10/13` positive recall and four
  abstentions. The cross-validated likelihood posterior obtained NLL
  `0.1330`, Brier `0.0517`, ECE `0.0960`, and classified `44/45` records,
  including `13/13` positives. This is also a provisional candidate, but the
  positive support is still only 13.
- Behind is blocked: although its cross-validated NLL was `0.5534`, it retained
  zero recall for the two `unknown` records, and its labels are legacy
  authored view intent rather than objective geometry.
- The comparison does not invent a probability for the original hard rules.
  Their accuracy, abstention coverage, and selective accuracy are reported
  separately. Probability quality is compared with a uniform no-evidence
  posterior, not an arbitrary fixed-confidence rule.
- Added
  `configs/research/hybrid_relation_likelihood_cv_seed165_184.json`,
  `scripts/run_hybrid_relation_likelihood_cv.py`, and focused tests.
  Artifacts are
  `outputs/calibration_pilot/reference_occlusion_seed165_184_gpu5/hybrid_rgbd_relation/cross_validation/result.json`
  and `summary.json`.
- Runtime was about `0.0043 s` on CPU with `0 GiB` GPU memory. This remains
  calibration-collection cross-validation, not final testing. It has not yet
  validated action-conditioned observations, task-risk gates, or reserved
  test seeds.

## 2026-07-30 — Re-run nested MPC with fold-specific calibrated perception

- Connected the cached Qwen target calibration and hybrid RGB-D relation
  evidence to the nested offline belief-MPC replay without rerunning any
  perception model.
- Removed the previous single-temperature weakness. Each outer fold now uses
  the Qwen temperature fitted on its other 15 seeds (`5.025`, `5.875`,
  `6.25`, or `6.125`). The five outer held-out seeds contribute neither to
  that temperature nor to the hybrid relation likelihood fitted for their
  evaluation.
- The leak-controlled results matched the earlier diagnostic. The
  action-agnostic belief MPC produced `15/20` correct grasps, zero wrong
  grasps, five safe fully-hidden deferrals, and a `1.00` safe-outcome rate.
  The action-conditioned belief MPC produced `12/20` correct grasps, zero
  wrong grasps, five safe fully-hidden deferrals, three missed visible
  targets, and a `0.85` safe-outcome rate.
- This does not support a superiority claim for the proposed planner. Only
  `1/20` cached episodes had a different post-hoc latent state between
  close-high and right, and target-track correctness tied in all `20/20`.
  Half of the learned outputs differed, but most differences were perception
  variability rather than a physical action-dependent scene change.
- The result therefore identifies the next data requirement: new calibration
  scenes must deliberately make different reachable actions produce
  different visibility/occlusion outcomes. Existing caches are sufficient
  for software validation but not for learning or evaluating meaningful
  action-conditioned future belief.
- Added
  `configs/research/nested_calibrated_perception_mpc_seed165_184.json` and
  updated the cached replay/nested calibration runners and tests. Artifacts
  are
  `outputs/offline_mpc/nested_calibrated_perception_mpc_seed165_184/result.json`
  and `summary.json`.
- Runtime was about `9.68 s` on CPU with `0 GiB` GPU memory. Isaac Sim,
  Vulkan, CUDA, VLM inference, robot motion, and grasp execution were not
  launched. This remains calibration development, not final testing.

## 2026-07-30 — Abort GPU-1 Vulkan smoke after verified GPU-0 context

- Attempted only seed `185`, variant `close_high_only`, with physical GPU 1
  selected for CUDA and the Isaac renderer, CUDA-visible PhysX device 0, and
  multi-GPU disabled.
- Added a fail-closed process-group watchdog that polls both compute and
  graphics processes through `nvidia-smi pmon`. It terminates the complete
  experiment if one of its PIDs appears on forbidden physical GPU 0, or if
  monitoring itself fails.
- Isaac PID `2593099` appeared on physical GPU 0 as a graphics process using
  `4 MiB`. The watchdog terminated the experiment after about `10.9 s`, before
  scene capture, RGB-D generation, GPU-2 perception, or MPC evaluation.
- Post-termination verification found neither the launcher nor Isaac PID on
  any GPU or in the process table. The pre-existing GPU-0 PID `2066833`
  belongs to another user and was not modified.
- The safety artifact is
  `outputs/live_pipeline/action_differentiating_scene_pilot/close_high_only/seed185/watchdog.json`.
  The partial `run001` directory contains startup logs only and is not a
  completed episode.
- This confirms that the current bare-metal `active_gpu=1` plus
  `CUDA_VISIBLE_DEVICES=1` configuration does not satisfy the zero-access
  GPU-0 policy. No further Isaac/Vulkan run should start without device-level
  container isolation or an administrator-verified host Vulkan/Xorg setup.

## 2026-07-30 — Run leak-controlled full baseline and ablation replay

- Ran nine methods on the cached seed `165–184` calibration collection using
  four episode-disjoint outer folds. Fold-specific Qwen temperature,
  hybrid-relation likelihood fitting, and inner task-cost selection excluded
  every outer held-out seed.
- Fixed policies remained unsafe: immediate grasp was correct on `15/20` with
  five wrong commitments; fixed close-high was `12/20` with eight wrong
  commitments; fixed right was `13/20` with seven wrong commitments.
- The confidence-only baseline produced `12/20` correct grasps, one wrong
  grasp, five safe hidden-target deferrals, and two missed visible targets
  (`0.85` safe-outcome rate).
- Action-agnostic belief MPC was strongest in this development cache:
  `15/20` correct grasps, zero wrong grasps, five correct hidden-target
  deferrals, and a `1.00` safe-outcome rate.
- Action-conditioned belief MPC produced `12/20` correct grasps, zero wrong
  grasps, five correct hidden-target deferrals, and three missed visible
  targets (`0.85` safe-outcome rate). It tied action-agnostic MPC on 17
  episodes and had higher development loss on three.
- Removing Qwen target temperature calibration reduced retrieval from
  `12/20` to `9/20` and the safe-outcome rate from `0.85` to `0.70`.
- Removing hybrid relation evidence did not hurt on this cache: it produced
  `13/20` correct grasps and a `0.90` safe-outcome rate. This means the
  current relation evidence/cost interaction is not yet supported as a
  beneficial component, even though its standalone calibration metrics were
  promising.
- Removing task-risk weights and commitment gates collapsed to immediate
  grasp and made all five `covered_unknown` episodes wrong commitments. This
  supports retaining a risk gate, but not a final numerical threshold.
- A negative-evidence ablation was deliberately not fabricated because the
  open-container cache has no cover-removal or empty-container post-action
  observation.
- The claim gate remains closed: these are reused calibration episodes, only
  `1/20` has a physically action-dependent close-high/right latent outcome,
  the proposed method does not beat every baseline, and no live motion or
  grasp ran.
- Added
  `configs/research/offline_full_baseline_ablation_seed165_184.json`,
  `scripts/run_offline_full_baseline_ablation.py`, and focused tests.
  Artifacts are
  `outputs/offline_mpc/full_baseline_ablation_seed165_184/result.json` and
  `summary.json`.
- Runtime was about `18.16 s` on CPU with `0 GiB` GPU memory. No Isaac Sim,
  Vulkan, CUDA, VLM inference, model-weight training, or reserved test seed
  was used.

## 2026-07-30 — Isolate membership versus occlusion downstream effects

- Ran a four-condition, four-fold episode-disjoint ablation with the same
  action-conditioned MPC and inner task-cost selection:
  target only, target plus membership, target plus occlusion, and target plus
  both relation factors.
- Target only produced `13/20` correct grasps, zero wrong grasps, five safe
  hidden-target deferrals, and two missed visible targets (`0.90` safe-outcome
  rate).
- Adding only hybrid RGB-D membership produced `15/20` correct grasps, zero
  wrong grasps, five safe hidden-target deferrals, and no missed visible
  targets (`1.00` safe-outcome rate).
- Adding only objective-reference occlusion produced `12/20` correct grasps,
  zero wrong grasps, five safe hidden-target deferrals, and three missed
  visible targets (`0.85` safe-outcome rate).
- Using both relations was policy-identical to occlusion-only on all `20`
  episodes. Membership could not change the selected actions once the current
  occlusion belief and gates were active.
- The full model missed visible targets on seeds `165`, `167`, and `178`.
  Seeds 165 and 167 assigned about `0.834` reference-occlusion probability
  at center and deferred instead of selecting a useful view. Seed 178
  re-observed right, then estimated fully-hidden probability `0.2533`, just
  above the provisional `0.25` gate, and deferred.
- The downstream problem is therefore localized to the current
  `occluded_by` observation/transition model and its commitment gates, not to
  membership. Strong standalone occlusion classification does not by itself
  establish useful sequential decision calibration.
- Added
  `configs/research/relation_factor_mpc_ablation_seed165_184.json`,
  `scripts/run_relation_factor_mpc_ablation.py`, and focused tests.
  Artifacts are
  `outputs/offline_mpc/relation_factor_ablation_seed165_184/result.json` and
  `summary.json`.
- Runtime was about `18.67 s` on CPU with `0 GiB` GPU memory. This remains
  calibration replay, not final testing or live simulation.

## 2026-08-01 — Run a GPU-5-only post-action multi-view Qwen diagnostic

- Added a development-only comparison of cached current-view Qwen evidence
  against Qwen inference that receives the historical center observation and
  one legitimately acquired post-action view together. Post-action images are
  never exposed to root action selection, and candidate IDs remain local to
  each image so no simulator identity correspondence is leaked.
- Selected a failure-enriched but scene-diverse calibration subset, seeds
  `165`, `166`, `167`, `168`, `169`, `178`, `179`, and `183`. Both
  `center + close_high` and `center + right` were evaluated, producing 16
  observation pairs. Reserved test seeds `200–209` were not used.
- Used the existing local pretrained `Qwen3-VL-8B-Instruct` checkpoint with
  one model instance, batch size one, and no training, fine-tuning, LoRA,
  download, Isaac Sim, or Vulkan. Physical GPU 5 was the only CUDA-visible
  device. A fail-closed watchdog monitored every project process against
  physical GPUs `0–4`; both the preflight and full run completed with no
  violation.
- Two covered-target pairs had no target proposal and were excluded from
  candidate-ranking and relation accuracy rather than assigning an invented
  candidate score. On the 14 visible-target pairs, cached single-view target
  selection was `9/14` (`64.29%`), while post-action multi-view selection was
  `13/14` (`92.86%`). Multi-view evidence corrected four single-view errors
  without introducing a target-selection regression; seed 166 close-high
  remained incorrect.
- Target-candidate membership was `14/14` for both single-view and multi-view
  inference. Current-view `occluded_by` accuracy decreased from `11/14`
  (`78.57%`) to `10/14` (`71.43%`): one previously correct rim case regressed,
  and three behind-ambiguous errors remained. Historical visual context helps
  identity on this subset but does not repair the blocked occlusion factor.
- Across all 16 cached results, model inference totaled `350.39 s`, averaged
  `21.90 s/pair`, and peaked at `17.7668 GiB`. The full invocation processed
  15 fresh pairs plus one preflight cache hit in `356.81 s`; the separate
  preflight took `71.91 s`. GPU 5 returned to `1 MiB`, 0% utilization, with no
  process after completion.
- Added
  `configs/research/qwen_multiview_postaction_ablation_seed165_184.json`,
  `scripts/run_qwen_multiview_postaction_ablation.py`, and focused tests.
  Extended the forbidden-GPU watchdog to accept multiple forbidden physical
  GPUs. The four directly relevant tests passed. Full discovery passed 150
  tests and encountered one pre-existing environment import error: system
  SciPy is incompatible with the installed NumPy, while the VLM environment
  has no SciPy. No package was changed for this diagnostic.
- Artifacts:
  `outputs/perception_aux/qwen_multiview_postaction_seed165_184_gpu5/summary.json`,
  `evaluation.json`, per-pair `results/*/result.json`,
  `watchdog_preflight.json`, and `watchdog_full.json`.
- Boundary: this is a deliberately small, failure-enriched calibration
  diagnostic, not an unbiased model comparison, final calibration, reserved
  test result, action-conditioned planning result, or paper claim. Direct
  multi-view identity is only a candidate post-action perception ablation.
  The next primary experiment is still the rendered action-differentiating
  scene pilot after compliant Vulkan isolation is available.

## 2026-08-01 — Render the first action-differentiating scene prototypes

- Physical GPUs 0 and 1 were verified empty before launch. The primary Isaac
  renderer and PhysX run used physical GPU 1 (`CUDA_VISIBLE_DEVICES=1`,
  renderer `active_gpu=1`, PhysX logical device 0, multi-GPU disabled). The
  user explicitly allowed the unavoidable small GPU-0 Vulkan graphics context;
  physical GPUs `2–5` remained fail-closed forbidden through the watchdog.
- A first GPU-1-only retry of seed 185, run 002, was correctly terminated in
  `13.92 s` when Isaac opened a `4 MiB` graphics context on physical GPU 0.
  It produced no observation and remains a safety diagnostic. After GPU 0 was
  explicitly allowed, run 003 completed in `180.90 s` with no watchdog
  violation.
- Seed 185 (`close_high_only`) executed actual continuous UR10e motion from
  center to close-high and then right. Both trajectories completed without
  detected collision; maximum final joint errors were `0.00803 rad` for
  close-high and `0.00803 rad` for right, below the `0.02 rad` gate.
- The seed-185 objective visible fractions were center `0.4653`, close-high
  `0.9791`, and right `0.6875`. The implemented gate passed because center was
  unresolved, close-high exceeded the resolution and gain thresholds, and
  close-high exceeded right by `0.2916`. Manual RGB/overview inspection found
  no obvious object penetration or unsupported floating object.
- Seed 185 is not yet accepted as a strict `close_high_only` calibration
  scene. Right visibility itself was slightly above the provisional `0.65`
  resolved threshold, so this rendered prototype demonstrates strong
  close-high dominance rather than literal close-high exclusivity.
- Seed 186 (`right_only`) completed both robot motions and all three RGB-D
  captures in `172.38 s`, with no collision or runtime failure, but correctly
  failed the causal scene gate. Visible fractions were center `0.4464`,
  close-high `0.9902`, and right `0.7186`; the intended blocked close-high view
  was instead the best view. Seed 186 is ineligible for calibration and later
  seeds were not launched.
- Image inspection and geometry audit localized the failure. The orange
  action occluder is aligned with the blocked camera only in the XY plane and
  uses a fixed low vertical center. That is inadequate for the elevated
  close-high wrist camera, so the occluder appears below the target instead of
  blocking it. A physically supported taller occluder or a revised 3D-aware
  placement must be designed and rerendered before continuing the 12-scene
  batch.
- No Qwen, GroundingDINO, SAM2, calibration fitting, MPC replay, grasp,
  training, or reserved test seed was used. After both runs, GPUs 0 and 1
  returned to `1 MiB`, 0% utilization, with no remaining process.
- Artifacts:
  `outputs/live_pipeline/action_differentiating_scene_pilot/close_high_only/seed185/run003/smoke_result.json`,
  `calibration_ground_truth.json`, RGB-D/overview images,
  `watchdog_run003.json`, and
  `outputs/live_pipeline/action_differentiating_scene_pilot/right_only/seed186/run001/smoke_result.json`.

## 2026-08-01 — Validate one close-high-only and one right-only rendered scene

- Replaced the original unsupported action-occluder design with two
  physically supported geometries: a basket-floor-supported upright cylinder
  for `close_high_only`, and a thin partial-cover bar supported by the two
  basket rims for `right_only`. Added an explicit exclusivity gate requiring
  the declared non-resolving view to remain below `0.65` visibility.
- The first revised seed-185 attempt (`run004`) completed both robot motions
  without collision but failed the scene gate: close-high visibility was only
  `0.5640`. The run was retained as a development failure. Lowering the
  supported cylinder from `0.18 m` to `0.14 m` fixed the view separation
  without moving it off its support surface.
- Seed 185 `run005` passed all gates: center `0.0005`, close-high `0.7104`,
  and right `0.1828` objective visible fraction. Close-high was the only
  resolving re-observation. Continuous UR10e motion completed for both view
  actions with no detected collision; maximum final joint error was
  `0.00803 rad`.
- Seed 186 `run002` also passed all gates: center `0.4309`, close-high
  `0.0923`, and right `0.7392`. Right was the only resolving
  re-observation. Both UR10e view motions completed without detected
  collision and with maximum final joint error `0.00803 rad`.
- Manual RGB inspection found no target/occluder penetration or unsupported
  floating object. The seed-186 bar acts as a narrow partial cover resting
  across the basket rims; it is not a free-floating distractor.
- Both runs used physical GPU 1 for the renderer and CUDA-visible PhysX
  device 0, with multi-GPU disabled. The user-approved small GPU-0 Vulkan
  context was allowed; a fail-closed watchdog forbade GPUs 2–5. After the
  runs, every physical GPU reported `1 MiB`, `0%`, with no compute process.
- No Qwen, GroundingDINO, SAM2, belief update, MPC, grasp, training, or
  reserved test seed was used. These are development scene/robot-motion
  validations, not final evaluation episodes.
- Artifacts are
  `outputs/live_pipeline/action_differentiating_scene_pilot/validated_scene_pair_summary.json`,
  `close_high_only/seed185/run005/`, and `right_only/seed186/run002/`.

## 2026-08-01 — Connect learned perception and expose the frozen MPC failure

- Ran GroundingDINO-Base, SAM2.1-Large, and pretrained
  Qwen3-VL-8B-Instruct sequentially on the six saved RGB-D observations from
  validated seeds 185 and 186. One model instance and batch size one were
  used; no training, fine-tuning, LoRA, Isaac Sim, or Vulkan was launched.
- Physical GPU 1 was the only visible CUDA device. A fail-closed watchdog
  forbade physical GPUs 0 and 2–5 for every CUDA stage. Qwen peaked at
  `17.1547 GiB`, averaged `21.86 s` per observation, and all GPUs returned to
  `1 MiB`, `0%` after completion.
- Learned target selection was correct in `4/6` views. The two errors occurred
  in deliberately blocked views: seed-185 center and seed-186 close-high both
  selected the visible outside red mug. The intended resolving observations
  succeeded: seed-185 close-high selected `target_red` at mask IoU `0.9087`,
  and seed-186 right selected it at IoU `0.8990`; both predicted membership
  `inside`.
- Applied the frozen seed-165–184 action-conditioned observation model to each
  new center observation. Future view files and scene labels were not read
  before the root action was selected. Both episodes selected `defer`, so root
  action accuracy against the rendered resolving action was `0/2`.
- For seed 185, costs were defer `0.45`, close-high `0.53`, and right `0.51`.
  For seed 186, they were defer `0.45`, close-high `0.4867`, and right
  `0.4657`. The failure is therefore localized to the old observation model
  and provisional noncompletion cost, which were calibrated on scenes with
  insufficient physical action differentiation. No threshold or cost was
  changed after seeing these outcomes.
- This is a useful failed planner transfer experiment: perception and causal
  scene construction work on the intended views, but the current future-belief
  model does not yet justify executing either view. It is not an end-to-end
  success or final result.
- Artifacts are
  `outputs/perception_grounding_pilot/action_differentiating_seed185_186/pilot_summary.json`,
  `evaluation_summary.json`, cached `grounded_sam2_qwen_rankings/*/result.json`,
  and `outputs/offline_mpc/action_differentiating_seed185_186/result.json`.

## 2026-08-01 — Complete the 11-episode scene-conditioned calibration pilot

- Rendered seeds 187–196 sequentially with physical GPU 1, one Isaac Sim
  process at a time, and multi-GPU disabled. Nine of ten new scenes passed the
  causal visibility gates. Seed 191 was retained as a failed development
  scene because its center visible fraction (`0.6572`) was slightly above the
  strict unresolved threshold (`0.65`). The batch took `1742.86 s`.
- Combined the nine eligible new scenes with validated seeds 185–186, giving
  11 development episodes and 33 independent center/close-high/right views.
  GroundingDINO-Base, SAM2.1-Large, and pretrained Qwen3-VL-8B-Instruct were
  run sequentially with physical GPU 1 as the only CUDA-visible device, batch
  size one, and one model instance at a time.
- Qwen completed all 33 observations in `608.76 s` of inference, averaging
  `18.45 s` per observation and peaking at `17.1607 GiB`. All view-level
  outputs were cached under
  `outputs/perception_grounding_pilot/action_differentiating_seed185_196/grounded_sam2_qwen_rankings/`.
- Post-hoc simulator evaluation found correct selected-target masks in `18/33`
  views and correct selected-candidate membership labels in `25/33` views.
  These figures include deliberately blocked or covered observations and are
  diagnostic perception results, not final metrics.
- Fitted a small scene-conditioned future-belief calibration model using only
  center-observation features and leave-one-episode-out folds. It selected the
  expected first action in `11/11` held-out folds; a single fixed action scored
  at most `5/11`. For the eight episodes that selected a camera move, the
  selected new view recovered the target in `8/8` post-hoc checks.
- This resolves the earlier `0/2` frozen-MPC transfer failure at the
  development calibration level, not at the final-testing level. The three
  covered episodes selected `remove_cover`, but that action and its subsequent
  negative-evidence update were not executed. No final grasp was run.
- Foundation-model training, fine-tuning, and LoRA were not performed.
  Calibration was performed; unbiased testing was not. Reserved test seeds
  were not opened, and no result here is valid as a final paper claim.
- Summary artifact:
  `outputs/offline_mpc/scene_conditioned_future_belief_seed185_196/pilot_summary.json`.

## 2026-08-03 — Close the physical remove-cover-to-replanning integration gap

- Added a causal post-action adapter and live-server continuation path. The
  server now can keep the same Isaac Sim process and stage alive after an
  actual UR10e+RG6 cover removal, publish the new `post_remove` RGB-D
  observation, receive one replanned action, and record that second decision.
- Seed 188 run 010 completed on physical GPU 0 only in `957.16 s` under a
  fail-closed watchdog forbidding GPUs 1–5. The Isaac process peaked near
  `5.47 GiB`; the watchdog recorded no violation and GPU 0 returned to
  `1 MiB` after exit. Multi-GPU was disabled and no VLM was loaded.
- The RG6 grasped the cover handle with bilateral contact, lifted/transferred
  the cover by a verified `0.162487 m`, and generated a new RGB-D observation
  in the same process. Peak measured contact forces were `7.31 N` left and
  `8.27 N` right; maximum penetration was `4.49e-05 m` left and
  `3.67e-06 m` right. There were no unexpected environment collisions,
  object attachment, or target pose copying.
- The target changed from `0` visible pixels before removal to `8,518` pixels
  afterward. The automatic simulator-mask pilot observation
  `target_detected` updated the discrete joint belief from
  `P(inside, covered)=0.65` to `P(inside, open)=0.982245`. Receding-horizon
  belief-MPC then selected `grasp_inside` as the second action while the same
  Isaac server was alive.
- Scene Graph snapshots before and after the physical action were generated
  and schema validated. Twenty-two focused CPU tests covering cover belief,
  Scene Graph/action contracts, remove-cover planning, and the physical-output
  adapter passed.
- This closes the development integration path only through second-action
  selection. The second `grasp_inside` action was not physically executed,
  because RG6 still held the transferred cover. Post-action evidence came
  from simulator instance masks rather than GroundingDINO/SAM2/Qwen, and this
  positive episode did not exercise empty-container negative evidence.
  Therefore it is not final evaluation evidence.
- Primary artifacts:
  `outputs/live_pipeline/remove_cover_physics_smoke/seed188/run010/`,
  `outputs/live_pipeline/remove_cover_replan/seed188/run002/`, and
  `outputs/live_pipeline/remove_cover_closed_loop_watchdog_gpu0.json`.

## 2026-08-04 — Downgrade the cover lift after strict contact-stability revalidation

- Re-audited seed-188 run 010 after the rendered lid motion looked weightless.
  The old acceptance gate used cumulative contact events and peak forces but
  did not bound lid-to-gripper relative translation, rotation, contact gaps,
  or post-lift lid/environment contact. Run 010 remains useful causal
  integration evidence, but is no longer accepted as realistic cover-lift
  physics evidence.
- Replaced the provisional `0.20 kg` cover with a `0.55 kg` composite
  plate/handle mass model, explicit center of mass and diagonal inertia, and
  linear/angular damping. Added quintic arm interpolation, per-step bilateral
  contact monitoring, a maximum three-step (`0.05 s`) contact gap, a `0.015 m`
  lid-to-gripper translation gate, rotation/angular-speed gates, and
  lid/environment collision rejection after lift.
- Official OnRobot RG6 documentation gives an adjustable gripping-force range
  of `25–120 N` and identifies the standard fingertip contact area as EPDM
  rubber. The simulator now records a `25 N` combined contact-force proxy,
  at least `8 N` per side, an EPDM-like provisional friction pair, and the
  exact fingertip collision prims receiving the physics material. These values
  remain uncalibrated simulation proxies rather than real-tool parameters.
- Runs 014 and 015 failed before lift because the new bilateral/combined force
  gates were not met. Run 016 met the force gate (`27.948 N` maximum combined)
  but the lid slipped `0.03048 m` relative to the gripper and lost contact four
  steps into the strict gap gate. Run 017 reached about `22.1/22.7 N` peak
  left/right forces but again slipped `0.03118 m` and lost both contacts.
- Run 018 was an infrastructure-only failure: the imported fingertip collision
  meshes are inside instanceable USD references and were hidden from ordinary
  traversal. Run 019 de-instanced only the two in-memory fingertip references,
  verified the actual collision paths ending in
  `inner_finger_1/inner_finger_1`, and bound the grip material directly. It
  nevertheless reproduced the run-017 lift failure, showing that missing
  material binding was not the sole cause.
- The remaining failure is structural: the simulation substitutes one
  position drive with a torque limit for the RG6's real continuous grip/force
  behavior. It freezes the master position after the first force event and
  cannot yet maintain the handle under lift. Do not increase friction, reduce
  lid mass, attach the lid, or loosen the contact/slip gates to manufacture a
  pass. Implement and calibrate a continuous RG6 grip-force controller or a
  validated fingertip contact proxy before rerunning the full covered episode.
- All revalidation commands used physical GPU 0 only under a fail-closed
  watchdog forbidding GPUs 1–5; no forbidden-GPU violation occurred. Qwen was
  not loaded, and none of these runs is final-evaluation evidence.
- Primary failure artifacts:
  `outputs/live_pipeline/remove_cover_physics_smoke/seed188/run016/`,
  `run017/`, `run019/`, and
  `outputs/live_pipeline/remove_cover_physics_revalidation_watchdog_gpu0_run019.json`.

## 2026-08-04 — Continuous RG6 control and 1 cm micro-lift revalidation

- Added a continuous development grip controller that can increase the RG6
  master closure target and drive torque while monitoring bilateral force and
  target-to-gripper slip. Added a mandatory `0.010 m` micro-lift before the
  full transfer; the micro-lift allows at most `0.005 m` relative translation
  and requires at least `0.007 m` actual lid rise.
- Seed-188 run 020 reached bilateral grasp, passed the `25 N` combined-force
  closure gate, and entered the micro-lift. During the micro-lift the lid moved
  `0.006081 m` relative to the gripper, exceeding the predeclared `0.005 m`
  gate, so execution stopped before the full lift and post-action replanning.
- The controller reached its development ceiling of `6 N m` after 97
  adjustments. Recorded peak forces were `21.40 N` left and `40.34 N` right,
  maximum penetration remained below `0.000435 m`, bilateral contact gaps were
  zero, and no unexpected environment collision was recorded. At the failure
  sample the instantaneous forces had fallen to `0.0 N` left and `1.74 N`
  right despite collision-pair overlap, confirming that pair presence alone
  is not a stable grasp criterion.
- Run 020 is a useful negative physical-validation result, not a successful
  cover removal and not final-paper evidence. Do not proceed to the full lift
  by raising safety limits or relaxing the slip gate. The next physical-model
  step needs a calibrated RG6 fingertip/contact-force proxy and lab-measured
  grip/lift settings; software work can meanwhile retain the micro-lift as a
  fail-fast safety gate.
- Run 020 used physical GPU 0 only under the fail-closed watchdog forbidding
  GPUs 1–5. The watchdog recorded no forbidden-GPU violation and Qwen was not
  loaded. Primary artifact:
  `outputs/live_pipeline/remove_cover_physics_smoke/seed188/run020/`.

## 2026-08-04 — Prepare the lab-calibrated RG6/lid transfer interface

- Added a CPU-only RG6/lid calibration validator, a deliberately incomplete
  lab worksheet, and a Korean measurement procedure. The worksheet records
  the actual RG6/fingertip identity, lid mass and geometry, commanded grip
  force, simulator mapping, and at least five real 1 cm micro-lift trials.
- The fixed safety contract remains a requested `0.010 m` micro-lift, at least
  `0.007 m` measured rise, no more than `0.005 m` relative translation,
  bilateral contact, and no unexpected collision. Simulator revalidation also
  retains the `60 N` per-finger force and `0.003 m` penetration limits.
- Connected validated calibration parameters to the scanned-basket cover
  authoring path and persistent RG6 controller. Measured cover geometry now
  recomputes composite center of mass and inertia; measured mapping values can
  replace the provisional torque, friction, and force thresholds. The runner
  can require transfer-ready physics and fail before Isaac Sim starts if the
  calibration is absent or incomplete.
- The repository template correctly reports `structure_valid=true` and
  `transfer_ready=false`, with zero trials and an explicit list of missing lab
  fields. No real values were guessed and no GPU or Isaac Sim run was started.
  Fifteen focused CPU tests passed.
- Primary files:
  `configs/hardware/rg6_lid_transfer_calibration.json`,
  `scripts/rg6_lid_calibration.py`,
  `docs/RG6_LID_TRANSFER_CALIBRATION_KO.md`, and
  `outputs/hardware_calibration/rg6_lid_transfer/readiness_report.json`.

## 2026-08-04 — Add a public-spec development proxy while lab data is unavailable

- Added `configs/hardware/rg6_lid_development_proxy.json` rather than filling
  unknown lab measurements with fabricated values. Manufacturer-backed fields
  include the RG6 `25–120 N` adjustable force range and the standard fingertip
  PN 100670 EPDM contact-area assumption. The UR10e public model specification
  is recorded as 6 DoF, `1.3 m` reach, and `12.5 kg` maximum payload.
- Kept the current `0.55 kg` lid, procedural handle geometry, `1.0/0.8`
  friction, and `2–6 N m` simulation drive mapping, but explicitly labeled
  them simulation assumptions. The config records run020's `6.081 mm`
  micro-lift slip failure and does not convert that failure into calibration.
- The validator reports `development_proxy_usable=true` and
  `transfer_ready=false`. Isaac and the remove-cover runner require an explicit
  provisional-mode flag to consume this file; transfer-ready mode still
  rejects it before simulation startup. Sixteen focused CPU tests passed and
  no GPU or Isaac Sim process was started.
- The current OnRobot product page states a `150 mm` maximum stroke, while the
  linked RG6 v1.9.1 datasheet states `160 mm`. Both are recorded as a revision-
  dependent discrepancy that must be checked on the lab unit rather than
  silently selecting one for real execution.

## 2026-08-04 — Resolve UR10e descent IK, retain the RG6-handle failure

- Runs 021–028 continued the seed-188 covered-container development
  revalidation on physical GPU 0 only. Every run used the fail-closed watchdog
  forbidding GPUs 1–5; no forbidden-GPU violation occurred. Qwen was not
  loaded, and these are physics-development runs rather than final results.
- Run 021 added compliant EPDM-like contact and reduced micro-lift slip from
  run020's `6.081 mm` to `5.600 mm`, but still failed the fixed `5 mm` gate.
  Contact impulse directions then exposed that the RG6 pinch line was not
  aligned with the handle axis.
- Runs 022 and 023 corrected the handle-axis yaw but exposed two independent
  arm-path problems: a redundant wrist revolution and then an elbow-branch
  switch with `2.314579 rad` final tracking error. Joint-angle unwrapping,
  dual equivalent parallel-gripper yaw candidates, dense `1 cm` Cartesian
  descent waypoints, and a `0.45 rad` local IK branch gate resolved both.
- Run 024 executed all 18 descent transitions with approximately `0.0011 rad`
  final error per segment and no unexpected collision. It reached correctly
  oriented bilateral handle contacts, but the imported RG6's `0.45 rad`
  closure target produced contact events with zero normal impulse. Run 025
  confirmed that increasing only the drive-force ceiling from `2` to `6 N m`
  did not fix the insufficient closure target.
- Run 026 used a joint-limit-safe `0.60 rad` handle closure target and reached
  meaningful bilateral force (`38.88/55.75 N`) with penetration below
  `1.59 mm`, but failed the `1 cm` micro-lift at `17.15 mm` relative slip.
- The development cover was then given explicit removable clearance: a
  `354 x 330 mm` plate leaves about `5 mm` outside clearance and `9 mm` support
  ledge per side against the current basket collision boxes. Contact-driven
  closure now freezes the geometric schedule at bilateral contact and advances
  with bounded force-controller increments. Run 027 stopped conservatively at
  `19.62 N` combined force, below the `25 N` gate.
- Run 028 used a slightly larger `0.00035 rad` contact increment and passed the
  force gate at `19.45/29.02 N`, with maximum penetration below `0.92 mm` at
  closure. It nevertheless failed the micro-lift with `21.96 mm` relative
  translation. The repeat failure after clearance and controller corrections
  isolates the remaining problem to the current RG6 fingertip-collision and
  handle-contact geometry/model, not the UR10e descent path.
- Stop parameter-only tuning here. The current covered episode is not a
  realistic physical success and post-removal replanning was not reached.
  Next validate the fingertip/handle contact geometry in a small isolated
  fixture, then replace the provisional values with lab measurements before
  rerunning the full episode. Twenty-two focused CPU tests pass.
- Primary artifacts:
  `outputs/live_pipeline/remove_cover_physics_smoke/seed188/run021/` through
  `run028/` and
  `outputs/live_pipeline/remove_cover_physics_revalidation_watchdog_gpu0_run028.json`.

## 2026-08-05 — Isolate the actual RG6 fingertip/handle contact failure

- Added `scripts/run_rg6_handle_contact_fixture.py` and focused CPU contracts.
  The fixture opens the imported UR10e+RG6 articulation, removes the RG6
  follower drives in memory so the Newton mimic constraints remain the only
  follower actuation, uses the actual fingertip collision descendants, and
  retains the provisional `0.55 kg` composite lid and `80 x 30 x 35 mm`
  handle. It never attaches the lid or copies its pose.
- The first diagnostic runs exposed and corrected a fixture-only IK error: a
  nominal downward quaternion selected a distant wrist branch. The 1 cm
  micro-lift is now computed from the exact forward-kinematic frame pose and
  has a local maximum joint change of `0.013565 rad`.
- Unguided run004 completed but could not establish force-controlled bilateral
  contact. A passive prismatic measurement guide was then added to lock only
  lid XY motion and rotation while leaving world-Z translation free. This
  guide is diagnostic instrumentation and is not part of the final basket
  scene. Lighting was added so the diagnostic MP4 is inspectable.
- Guided run005 still failed before the micro-lift. The actual master and all
  five mimic followers moved consistently; the final measured master was
  `0.398118 rad`. Both fingertip/handle collision pairs were observed, but
  reported peak normal force and penetration were `0 N` and `0 m` on both
  sides. Therefore the `25 N` combined-force gate did not pass and the 1 cm
  lift was deliberately not executed.
- This rules out free lateral lid motion as the sole cause. The current
  imported RG6 collision/linkage envelope reaches the procedural 30 mm handle
  without producing a compressive grasp under the provisional mapping. The
  next physical-model task is a jaw-width/contact-envelope calibration against
  the real fingertip geometry and RG6 command, not more fitting of friction,
  mass, or safety thresholds to seed 188.
- Run005 took `142.85 s` including Isaac startup and video encoding. It used
  physical GPU 0 only; the watchdog forbade GPUs 1–5 and recorded no
  violation. Qwen, Scene Graph, and MPC were not loaded. The result is a
  development failure, not final-paper evidence.
- Primary artifacts:
  `outputs/rg6_handle_contact_fixture/run005/result.json`,
  `outputs/rg6_handle_contact_fixture/run005/rg6_handle_contact_fixture.mp4`,
  and
  `outputs/rg6_handle_contact_fixture/watchdog_gpu0_run006.json`.
- Twenty-nine focused CPU tests pass.

## 2026-08-05 — Measure the jaw envelope and retain the loaded-mimic failure

- Added `scripts/run_rg6_jaw_width_calibration.py`. In one actual-articulation
  process it swept 17 master commands from `-0.20` to `0.60 rad`, measured the
  five passive mimic followers, and projected the real fingertip collision
  meshes onto their pinch axis. No object, attachment, Qwen, Scene Graph, MPC,
  or training was used.
- The unloaded sweep was numerically clean: measured master range
  `-0.198856` to `0.600050 rad`, maximum mimic error `5.10e-6 rad`, and a
  strictly decreasing collision-surface gap. For the procedural 30 mm handle,
  the height-band surface gap reaches 30 mm at an interpolated master value of
  `0.435487 rad`.
- At run005's measured `0.398118 rad`, the nearest sweep sample still had a
  `35.532 mm` handle-band gap. The earlier collision pairs were therefore
  speculative proximity pairs, not compressive contact. The provisional
  controller now starts its bounded force ramp at the measured geometry
  threshold rather than treating pair presence as force.
- Run006 reached real right-finger contact (`53.21 N`, `1.169 mm`
  penetration) but zero left-finger force. The measured unloaded jaw center
  was offset from the nominal handle center by about `0.615 mm`; run007
  corrected exactly that measured offset, but still produced only right-side
  force (`47.06 N`, `1.029 mm` penetration) and zero left-side contact.
- Both runs remained below the fixed `60 N` and `3 mm` safety ceilings, and
  both correctly refused the 1 cm micro-lift because bilateral force was not
  established. The center correction therefore rules out simple handle
  placement as the cause.
- The imported articulation is symmetric and accurate without load, but its
  current master-plus-passive-mimic representation stalls asymmetrically when
  one fingertip is loaded. Stop fixture pose and friction tuning. The next
  implementation task is to repair or replace the loaded RG6 coupling/contact
  model using the real lab command/fingertip measurements, then rerun this
  same fixed fixture before any full covered episode.
- Jaw sweep runtime was `144.73 s`; centered fixture run007 runtime was
  `71.97 s`. Both used physical GPU 0 only under watchdogs forbidding GPUs
  1–5, with no violation. These are development calibration/failure artifacts,
  not final evaluation.
- Primary artifacts:
  `outputs/rg6_jaw_width_calibration/run001/`,
  `outputs/rg6_handle_contact_fixture/run006/`, and
  `outputs/rg6_handle_contact_fixture/run007/`.
- Thirty-six focused CPU tests pass.

## 2026-08-05 — Pass the isolated RG6 contact gate with coordinated drives

- Added a development-only `coordinated_drives` mode to the unchanged actual
  RG6 contact fixture. It removes the five passive Newton mimic APIs in memory
  and commands all six imported RG6 joints with their URDF ratios. The asset
  on disk is not changed. This is a simulator coupling diagnostic, not the
  real RG6 command interface and not a transfer-ready controller.
- The unloaded open/close/reopen smoke passed with maximum tracking error
  `3.05e-6 rad` and maximum coupling error `1.48e-6 rad` at a provisional
  aggregate `6 N m` drive-effort budget.
- Contact run008 established symmetric real collision contact but the
  conservative aggregate `6 N m` budget produced only `5.53/5.15 N`, below
  the fixed `25 N` combined gate. Run009 used a development-only aggregate
  limit of `18 N m` and reached `17.76/16.72 N`; a software timing bug failed
  to recheck the force gate after the contact-settling interval, so no lift
  was authorized. The gate now uses current active bilateral force and is
  rechecked after settling.
- Run010 executed the physical micro-lift with bilateral contact and no
  unexpected collision, but reached only `5.87 mm`, below the fixed `7 mm`
  acceptance floor. The object-relative slip was `1.29 mm`, so this was arm
  target lag rather than the prior `>20 mm` contact slip. The fixture now
  holds the exact IK target while continuously checking contact, force,
  penetration, and relative motion and records actual gripper displacement.
- Run011 passed the fixed isolated-fixture gates without attachment or pose
  copying: gripper lift `9.37 mm`, cover lift `7.62 mm`, maximum relative
  translation `1.80 mm`, continuous bilateral contact, peak finger forces
  `24.87/21.85 N`, maximum penetration `0.512 mm`, and zero unexpected robot
  contacts. The requested 10 mm target was not reached exactly before the
  300-step convergence limit, so this is a bounded fixture success rather
  than a claim of exact trajectory tracking.
- All runs used physical GPU 0 only under watchdogs forbidding GPUs 1–5; no
  violation occurred. Run011 took `85.75 s` including watchdog overhead.
  Qwen, Scene Graph, MPC, training, and final test seeds were not used.
- The next step is to port this explicitly provisional coupling mode into the
  full covered-container controller, remove the diagnostic vertical guide,
  and rerun cover lift/release/post-action observation/replanning. Lab RG6,
  fingertip, lid, and commanded-force measurements are still required before
  transfer-ready or final-evaluation execution.
- Primary artifacts:
  `outputs/ur10e_rg6_physics/coordinated_drive_smoke/run001/result.json`,
  `outputs/rg6_handle_contact_fixture/run011/result.json`,
  `outputs/rg6_handle_contact_fixture/run011/rg6_handle_contact_fixture.mp4`,
  and `outputs/rg6_handle_contact_fixture/watchdog_gpu0_run012.json`.
- Fifty-five focused CPU tests pass.

## 2026-08-05 — Connect learned post-remove RGB-D localization to the pending grasp

- Extended the persistent covered-container server so a replanned
  `grasp_inside` request can execute a second contact-gated RG6 manipulation
  in the same Isaac Sim process after the cover has been placed, released,
  and the new RGB-D observation has arrived. The request must contain a
  localization file inside the live session and is rejected if it reports
  simulator-ground-truth leakage.
- Extended the remove-cover runner with `--execute-post-remove-grasp`. In this
  mode it runs GroundingDINO-Base, SAM2.1-Large, and Qwen3-VL-8B-Instruct
  sequentially on the newly acquired post-remove observation, backprojects
  the selected anonymous mask with depth, sends the dynamic localization to
  the live Isaac process, and requires bilateral contact, verified lift,
  force, penetration, and collision gates before accepting the final grasp.
- Thirty-one focused CPU tests passed. A saved-observation validation on
  run049 used physical GPU 5 only under a fail-closed watchdog forbidding
  GPUs 0–4. It completed in `39.41 s`, peaked at `16.9484 GiB`, selected
  `candidate_001`, predicted `inside`, and estimated the target center as
  `[0.708807, -0.188551, 0.067998] m` without exposing simulator labels to
  inference.
- A post-hoc simulator-mask diagnostic estimated
  `[0.708405, -0.189317, 0.068014] m`, so the learned-mask RGB-D center differed
  by about `0.87 mm`. The diagnostic was computed only after learned inference
  and was not used for candidate selection or control.
- The complete same-process physical execution has not started because GPU 0
  is currently occupied by another researcher's training process. Bare-metal
  Isaac still opens a Vulkan graphics context on GPU 0 even when another
  renderer GPU is selected, so launching on GPU 1–5 would violate the existing
  non-interference rule. No other researcher's process was modified.
- Primary artifacts:
  `outputs/perception_aux/remove_cover_post_remove_run049_gpu5/` and its
  `watchdog.json`. This is learned-perception integration validation, not a
  reserved-seed final evaluation result; scores remain uncalibrated and the
  provisional RG6/lid simulation proxy remains non-transfer-ready.

## 2026-08-06 — Complete one same-process covered-container closed loop

- Target-only physics run017 replaced the unstable six-drive target proxy
  with the imported RG6 Newton mimic linkage and a provisional `1.2 N m`
  single-master effort ceiling. It lifted the target `179.95 mm` with
  `0.301 mm` maximum gripper-relative translation, `0.50 deg` maximum
  rotation, terminal bilateral forces of about `8.39/8.39 N`, maximum
  penetration `0.512 mm`, and zero unexpected collisions.
- Full seed-188 run058 completed in one Isaac Sim process: physical cover
  grasp/lift/placement/release, new RGB-D capture, GroundingDINO-Base,
  SAM2.1-Large, Qwen3-VL-8B-Instruct, belief/Scene Graph update, exact
  discrete belief-tree MPC replanning, learned-mask RGB-D dynamic IK, and a
  physical target lift. No simulator instance mask or ground-truth target
  pose was used for candidate selection, localization, belief update, or
  action choice.
- The cover was lifted `159.89 mm`, placed and released with continuous
  pre-release bilateral contact, maximum relative translation `0.710 mm`,
  maximum penetration `0.734 mm`, and no unexpected collision. After the new
  observation, the posterior assigned `0.982245` to `inside|open`; the MPC
  selected `grasp_inside` over additional viewpoints and defer.
- The dynamically planned target grasp lifted the target `180.03 mm` with
  terminal mean forces `9.85/9.83 N`, maximum relative translation
  `0.263 mm`, maximum rotation `0.160 deg`, maximum penetration `0.400 mm`,
  continuous bilateral contact, and zero unexpected environment collision.
- Restoring the Newton mimic linkage after the coordinated cover task requires
  a PhysX rebuild. The released cover local pose is now authored after
  `stop()` and verified within `2 mm/2 deg` after `play()`, preventing a
  hidden reset to the basket.
- Run058 took `2503.79 s` under a fail-closed GPU watchdog. Only physical GPU
  0 was exposed; GPUs 1–5 were forbidden and no violation occurred. The
  target-only run017 took `514.63 s` under the same policy.
- The run058 revalidation passed every saved removal gate. All 216 CPU tests
  pass when the grounding/calibration tests use the dedicated perception
  environment; the system Python has an unrelated SciPy/NumPy ABI mismatch.
- This is one deterministic development integration success, not a final
  paper result. Qwen scores are not calibrated, the RG6/lid values are
  provisional and `transfer_ready=false`, and reserved multi-seed testing,
  baselines, ablations, calibration/test separation, and real-robot
  validation remain outstanding.
- Primary artifacts:
  `outputs/live_pipeline/persistent_composite_grasp_smoke/run017/` and
  `outputs/live_pipeline/remove_cover_physics_smoke/seed188/run058/`.

## 2026-08-07 — Complete a two-update physical negative-evidence loop

- Added the development-only `empty_cover_then_right` scene. The target is
  outside and behind an empty covered basket. Its rendered objective target
  visibility is `0.0` in center, `0.0` in close-high, and `0.720665` in the
  physically reachable right view, so the unchanged `0.65` resolution gate
  passes only after the intended re-observation action.
- Scene-development runs001--003 were retained as rejected attempts: the
  right-view visibility was respectively `0.0`, `0.343815`, and `0.567326`.
  Run004 passed after changing target geometry; the acceptance threshold was
  not lowered.
- Extended the persistent Isaac server so a post-cover-removal viewpoint
  action is physically executed in the same process, creates a new RGB-D
  observation, and accepts a second replanned request.
- The first full negative-evidence run001 physically removed the cover but
  correctly failed because its adapter equated target pixels anywhere in the
  image with finding the target inside the inspected container. The global
  pixel rule was removed; development membership now comes from automatic
  simulator 3D ground truth and is explicitly excluded from final evaluation.
- Full development run002 completed in `2635.84 s` with the action sequence
  `remove_cover -> viewpoint_right -> grasp_outside`. The posterior
  `outside_near|open` belief changed from `0.877887` after the empty-container
  update to `0.995309` after right-view outside evidence.
- GroundingDINO-Base, SAM2.1-Large, and Qwen3-VL-8B-Instruct ran only after
  the physical right-view observation. Qwen selected `candidate_001`; its
  anonymous learned mask and RGB-D estimated center
  `[1.163280, 0.075493, 0.031196] m`, and the same-process dynamic IK grasp
  lifted the target `179.68 mm`.
- The final grasp maintained bilateral contact with terminal mean forces
  about `9.07/9.13 N`, maximum gripper-relative translation `0.204 mm`,
  maximum rotation `0.116 deg`, maximum penetration `0.420 mm`, and no
  unexpected collision. No object attachment or pose copying was used.
- Cover removal again passed bilateral-contact, supported-placement, release,
  force, penetration, stability, and unexpected-collision gates. It lifted
  the cover `159.89 mm` and left it stably supported after release.
- The fail-closed watchdog exposed physical GPU 0 only, forbade GPUs 1--5,
  and reported no violation. Seed 197 is a development seed; no reserved test
  seed was used.
- This closes the physical two-update integration gap but is not final paper
  evidence. The empty-container relation still uses simulator generator 3D
  membership, action-conditioned likelihoods are development values, Qwen
  calibration is not frozen, and RG6/cover parameters remain provisional.
- Primary artifact:
  `outputs/live_pipeline/negative_evidence_live_development/seed197/run002/negative_evidence_live_result.json`.

## 2026-08-07 — Remove the relation oracle from the live negative-evidence loop

- Added `scripts/audit_saved_learned_relation.py`, which combines learned
  GroundingDINO/SAM2 candidate and basket masks with metric RGB-D and camera
  calibration. It does not read simulator instance IDs or relation labels.
- Replaced the live development controller's simulator-membership lookup with
  conservative agreement between Qwen's relation choice and the RGB-D
  footprint classifier. Disagreement or `unknown` now aborts instead of
  fabricating an empty-container observation.
- `seed197/run003` completed the physical sequence
  `remove_cover -> viewpoint_right -> grasp_outside` with two belief updates.
  Qwen and RGB-D both classified the selected post-remove target as `outside`;
  the RGB-D center was `0.3173 m` outside the learned basket footprint.
- The run used GPU 0 only for `2615.75 s`. The forbidden-GPU watchdog reported
  no access to GPUs 1--5.
- Physical gates passed without attachment or pose copying: the cover was
  lifted `0.1599 m` and released on support, and the target was lifted
  `0.1796 m` with bilateral contact, no unexpected environment collision,
  maximum penetration `0.000519 m`, and horizontal slip `0.001456 m`.
- This remains development evidence. Qwen/relation scores and
  action-conditioned observation likelihoods are not frozen/calibrated, the
  instruction template still requires final protocol review, and RG6/cover
  parameters remain provisional (`transfer_ready=false`). Reserved test seeds
  were not opened.
- Primary artifact:
  `outputs/live_pipeline/negative_evidence_live_development/seed197/run003/negative_evidence_live_result.json`.

## 2026-08-07 — Predeclare the ICRA simulation protocol and remove prompt leakage

- Added task overrides to the live perception adapter and removed membership
  words from the instruction. The development/final template is now
  `Find and pick up the red mug with the white logo.`; inside/outside must be
  inferred from the current image rather than supplied by the instruction.
- Reprocessed the saved run003 post-remove and right RGB-D observations with
  the neutral prompt. Both Qwen and learned-mask RGB-D geometry still returned
  `outside`; the post-remove evidence mapped to `empty_container`. The audit
  took `62.15 s`, used physical GPU 0 only, and the watchdog reported no
  access to GPUs 1--5.
- Added `configs/research/icra_simulation_evaluation_protocol_v1.json`. It
  predeclares two mandatory scenario families, reserved seeds 200--209, seven
  required methods including the proposed method, six essential ablations,
  three primary metrics, physical success gates, and paired statistical
  reporting. The planned scale is 260 method--scenario evaluations.
- Added a fail-closed preflight audit. Structural checks pass, but reserved
  test launch remains blocked until Qwen temperature, relation likelihoods,
  action-conditioned observation models, task costs, commitment gates, and
  prompt/model hashes are frozen in a separate parameter file.
- Eleven focused tests pass. No reserved seed was rendered or evaluated.
- Primary artifacts:
  `outputs/perception_aux/neutral_prompt_negative_evidence_run003/result.json`
  and `outputs/final_evaluation/icra_protocol_v1/preflight.json`.

## 2026-08-07 — Re-run action-differentiating perception with the neutral prompt

- Recomputed all 33 Qwen rankings for the accepted seed-185--196
  action-differentiating observations with the identity-only instruction
  `Find and pick up the red mug with the white logo.`. GroundingDINO and SAM2
  masks were reused by path, while every Qwen output was regenerated in a new
  output root; no prior Qwen answer was copied.
- The single-model, batch-one run took `590.20 s` of inference time
  (`17.8848 s` per observation), peaked at `17.1607 GiB`, and completed under
  a GPU-0 watchdog with no access to GPUs 1--5.
- Target selection remained `18/33` (`54.55%`) and selected-candidate
  membership remained `25/33` (`75.76%`). Removing the leaked relation phrase
  did not change these counts, but the target-selection rate is still too low
  to claim final perception performance.
- Leave-one-episode-out scene-conditioned action calibration selected the
  designed resolving root action in `11/11` development episodes. The eight
  selected view actions recovered the target in `8/8`; three episodes selected
  cover removal. This is a horizon-one calibration result, not the proposed
  final multi-step belief-space MPC or a reserved test.
- Froze only the evidence-backed identity components: Qwen repository/revision,
  neutral prompt hash, and target temperature `5.825` fitted on 20 calibration
  seeds. Relation likelihoods, the full action-conditioned observation model,
  task costs, and the commitment gate remain unfrozen, so final-test preflight
  stays blocked.
- Primary artifacts:
  `outputs/perception_grounding_pilot/action_differentiating_neutral_seed185_196/`,
  `outputs/offline_mpc/scene_conditioned_future_belief_neutral_seed185_196/result.json`,
  and `configs/research/icra_frozen_parameters_v1.json`.

## 2026-08-07 — Add a second independent empty-container closed-loop calibration episode

- Ran seed 198 through the complete physical sequence
  `remove_cover -> viewpoint_right -> grasp_outside` in one Isaac Sim process.
  The identity-only prompt was used. Qwen and learned-mask RGB-D both returned
  `outside` after cover removal; simulator membership was not used for the
  control decision.
- The first update assigned `0.877887` belief to `outside_near|open`; after the
  physically executed right-view observation, the second update raised it to
  `0.995309`. These are uncalibrated development beliefs, not empirical success
  probabilities.
- The cover was lifted `0.159887 m`, placed and released on the WorkMat, and
  passed the contact, support, force, penetration, stability, and collision
  gates. Maximum forces were `23.30/24.57 N`, maximum penetration was
  `0.734 mm`, and maximum gripper-relative translation was `0.710 mm`.
- The dynamically localized outside target was lifted `0.179681 m` with
  continuous bilateral contact. Maximum forces were `13.51/12.59 N`, maximum
  penetration was `0.462 mm`, horizontal slip was `1.370 mm`, maximum
  gripper-relative translation was `0.700 mm`, and no unexpected environment
  collision occurred. No attachment or pose copying was used.
- Total episode time was `2670.53 s`; the watchdog ran for `2672.11 s`, exposed
  physical GPU 0 only, forbade GPUs 1--5, and recorded no violation. All 237
  CPU tests pass afterward.
- A new strict readiness audit counts repeated runs of one seed only once and
  reapplies the predeclared paper physics gates. It now finds two independent
  empty-container successes (seeds 197 and 198) out of the required ten, and
  one independent target-inside success (seed 188). Reserved test seeds remain
  unopened and this result is not final paper evidence.
- Primary artifacts:
  `outputs/live_pipeline/negative_evidence_live_development/seed198/run001/negative_evidence_live_result.json`,
  `outputs/live_pipeline/negative_evidence_live_development/seed198/watchdog_gpu0.json`,
  and `outputs/final_evaluation/icra_protocol_v1/cover_calibration_readiness.json`.
## 2026-08-07 — Add objective camera-relative behind ground truth and run seed 199

- Captured calibration seed `199`, variant `behind_ambiguous`, with actual
  UR10e center -> close-high -> right wrist-camera motion on physical GPU 0.
  The accepted run completed in `90.98 s`; the rendered visibility gate passed
  with target pixels `2728 -> 6516 -> 4931` in the first smoke and all three
  observations remained valid in the instrumented rerun.
- Added simulator-only `objective_camera_relative_behind.json` generation.
  The label uses the actual target center, reference world bounds, current
  camera ray, and rendered target/reference projection overlap. It is written
  after capture and is explicitly not exposed to the model or planner.
- The objective audit corrected a legacy intent-label disagreement: seed 199
  center was previously labeled `behind=unknown`, but the measured target was
  `0.0727 m` beyond the basket far edge with `0.505` projected overlap, so the
  objective label is `behind=yes`. Close-high and right were also `yes`.
- Reprocessed the instrumented RGB-D with GroundingDINO-Base, SAM2.1-Large,
  and one pretrained Qwen3-VL-8B instance on physical GPU 0. The job completed
  in `64.09 s`; Qwen averaged `11.66 s/observation` and peaked at `17.0407 GiB`.
  The forbidden-GPU watchdog observed no access to physical GPUs 1--5.
- Qwen selected the correct instruction target in all `3/3` views. Against
  target-only objective relation ground truth, Qwen behind was correct in
  `1/3` views (one additional view abstained as unknown), while the RGB-D
  hybrid was correct in `3/3`. For reference occlusion Qwen was `0/3` and the
  hybrid was `2/3`. Hybrid world membership was `6/6` across target and
  distractor candidate rows.
- The single seed cannot freeze relation calibration: objective behind has
  only the `yes` class and lacks `no/unknown` support. Training, fine-tuning,
  LoRA, reserved-seed testing, and final evaluation were not performed.
- Artifacts:
  `outputs/live_pipeline/objective_behind_calibration_seed199_gpu0/run002/`,
  `outputs/calibration_pilot/objective_behind_seed199_gpu0_v2/`, and
  `configs/perception/hybrid_rgbd_relation_objective_behind_seed199.json`.

## 2026-08-07 — Freeze objective hybrid relation likelihoods on 25 calibration seeds

- Completed a 20-scene objective relation batch on seeds `165--184` with
  60 actual-motion RGB-D observations. The five scene families were balanced
  across inside, outside, rim occlusion, full cover, and camera-relative
  behind cases. The job took `2500.39 s`; Qwen averaged `9.697 s` per
  observation and peaked at `17.0919 GiB` on physical GPU 0. The watchdog
  detected no access to GPUs 1--5.
- The first objective audit supported `behind=yes/no` but had no objective
  `unknown` examples. Added the `behind_boundary_unknown` scene and captured
  seeds `185--189`, giving ten objective unknown views and five yes views.
  The five-seed job completed in `683.32 s`; Qwen selected the target in
  `15/15` observations, averaged `12.06 s`, and peaked at `17.0894 GiB`.
- Qwen alone recovered none of the new `behind=unknown` cases. The initial
  RGB-D rule also overcommitted because visible-mask outliers shortened the
  basket footprint and biased the mug center. On calibration data only, the
  method was revised to use measured basket XY dimensions, a 5--95 percentile
  mug center, and a conservative `0.03 m` behind abstention band. Real-world
  transfer therefore requires replacing the simulated basket dimensions with
  the measured lab basket dimensions.
- Five-fold episode-disjoint calibration over 25 seeds accepted all three
  hybrid relation likelihood components. Membership was `127/128`
  (`99.22%`, NLL `0.0457`); objective behind was `63/63`, including
  yes/no/unknown support `20/33/10` (NLL `0.1327`); objective reference
  occlusion classified `56/63` after likelihood calibration (`88.89%`, NLL
  `0.3324`, yes recall `23/28`). All three improved over the no-evidence
  posterior.
- Froze the full-data categorical `P(hybrid evidence | true relation)` tables
  and immutable source hash in `configs/research/icra_frozen_parameters_v1.json`.
  The final preflight now clears relation likelihoods but remains blocked on
  the action-conditioned observation model, task-cost weights, and commitment
  gate. Reserved seeds `200--209` remain unopened; this is calibration, not
  final testing, and no model training or fine-tuning was performed.
- Primary artifacts:
  `outputs/calibration_pilot/objective_relations_seed165_184_gpu0/`,
  `outputs/calibration_pilot/objective_behind_unknown_seed185_189_gpu0/`, and
  `outputs/calibration_pilot/objective_relations_seed165_189_gpu0/hybrid_rgbd_relation_measured_extent/cross_validation/result.json`.
- The complete project regression suite passes `244/244` in the perception
  environment, and `git diff --check` passes.

## 2026-08-07 — Audit the full action-conditioned observation calibration

- Added a CPU-only, calibration-only audit that combines the episode-disjoint
  scene-conditioned view model with physically executed cover-removal
  outcomes. Reserved final-test seeds `200--209` were rejected by construction
  and remain unopened.
- The current-scene view policy selected the designed resolving action in
  `11/11` held-out development episodes. The strongest fixed-action baseline,
  always choosing right, reached `5/11` (`45.45%`). Variant support was
  close-high-only `3`, right-only `3`, either-view `2`, and cover-required `3`.
- Physically executed cover calibration contains ten independent
  outside/empty-container episodes and one independent inside/target-detected
  episode. A smoothed leave-one-episode-out diagnostic classified `11/11`,
  but the inside outcome has only one episode and therefore does not meet the
  predeclared minimum-five support rule. The score does not override the
  sparse-class gate.
- After an empty-container observation, the saved right-view Qwen relation
  output was `outside` in `10/10` independent episodes. This is calibration
  evidence, not a calibrated probability or final-test result.
- The full action-conditioned observation model was deliberately **not**
  frozen or applied to MPC. Four additional independent inside-cover physical
  calibration episodes are required. Training, fine-tuning, final testing,
  and reserved-seed access were not performed.
- Primary artifact:
  `outputs/offline_mpc/full_action_conditioned_observation_calibration_v1/result.json`.
- The complete repository regression suite passes `248/248`, including two
  focused tests for likelihood normalization and episode-disjoint replay;
  `git diff --check` also passes.

## 2026-08-07 — Add the GitHub handoff documentation

- Replaced the development-only README with a concise English entry point for
  model setup, simulation, cached planning, and hardware-readiness checks.
- Added project overview, action-space, model, simulation, real-robot transfer,
  experiment-protocol, and current-limitations documents. The action document
  separates semantic decisions from deterministic motion and safety checks.
- Added four minimal shell entry points for perception, headless simulation,
  cached planning, and a no-motion real-robot configuration check. GPU and
  local environment paths are supplied through environment variables.
- Model weights, caches, experiment outputs, and generated Isaac import
  payloads remain excluded from Git. The complete test suite passes `250/250`.

## 2026-08-08 — Curate the repository landing page for later sharing

- Confirmed that the current GitHub repository is a review staging area for a
  separate lab-facing repository.
- Removed the server-specific GPU index and local-path examples from the
  README and share-facing setup documents.
- Removed transient development status, calibration-only results, and
  incomplete quick commands from the README. No implementation or experiment
  behavior changed in this documentation-only update.
