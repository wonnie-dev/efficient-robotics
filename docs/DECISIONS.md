# Decisions

## Active decisions — 2026-07-24

### Final simulation embodiment

- Universal Robots UR10e.
- OnRobot RG6.
- Wrist-mounted Zivid 2 3D/RGB-D camera, or a documented faithful Isaac Sim approximation.
- Do not substitute the final embodiment without explicit user approval.

### Final research claim

- Maintain calibrated task-conditioned object and relation beliefs.
- Predict action-conditioned future posteriors before executing an action.
- Use receding-horizon optimization of task loss and wrong commitment, together with execution risk and motion cost.
- Update beliefs with positive and negative evidence, then replan.

Statements in older entries that call the robot stack provisional, pending confirmation, or freely replaceable are superseded. Those entries remain below as historical implementation records. Existing decisions about non-oracle planning, Bayesian updates, VLM contracts, calibration utilities, Scene Graph interfaces, and Isaac Sim behavior remain active unless explicitly superseded.

## 2026-08-05 — Cover removal must include supported release and retreat

- Decision: a physical `remove_cover` action is incomplete after lift and
  horizontal transfer. Before post-action RGB-D and replanning, the cover must
  be lowered onto a declared support, support contact must be verified, RG6
  must open, the open gripper must retreat, and the released cover must remain
  stable.
- Runtime contract: use dense same-branch local IK; retain bilateral contact
  through lift, transfer, and lowering; allow only the declared support contact
  during placement; require cleared finger contact and persistent support after
  retreat. Attachment and target-pose copying remain forbidden.
- Development staging pose: offset the cover by `[-0.42, -0.20, 0.16] m` and
  place it on `/World/WorkMat`. Runtime checks require the projected cover
  center to remain at least `0.03 m` inside the support, at least `75%` planar
  footprint overlap, and at least `0.03 m` basket clearance. This pose must be
  replaced or revalidated from measured lab geometry before real-robot execution.
- The earlier `[-0.45, 0.0, 0.16] m` pose was retired after run044: slowing
  placement fixed its tracking-error failure but exposed a fail-closed contact
  between the wide cover plate and the UR10e shoulder during lowering.
- A `+0.40 m` Y candidate was rejected before motion in run045 because the
  benchmark workbench's transformed positive-Y edge is only `0.18 m`. The
  subsequent `-0.40 m` candidate reached placement waypoint 31 in run047 but
  contacted the raised WorkMat before the lower WorkBench support. Placement
  now explicitly targets the WorkMat and permits bounded overhang only when
  the center-margin and overlap-fraction stability gates both pass.
- For that bounded overhang, `/World/WorkBench/Top` is an explicitly declared
  secondary support only during placement, release, and retreat. Primary
  WorkMat contact remains mandatory; penetration and post-release stability
  gates cover both supports, and every other target/environment pair remains
  fail-closed.
- Released-cover success is evaluated with pre-release bilateral contact and
  post-release stability, not final bilateral contact. A correctly opened and
  retreated gripper must have cleared finger contact; retaining contact after
  release is a failure, not a success condition.
- Contact interpretation: existing basket-rim contacts may persist while the
  cover root is within `0.02 m` of its initial pose. After that clearance, any
  cover/environment contact is an abort except the explicitly declared staging
  support during placement/release/retreat.
- Tracking policy: supported placement uses at least 60 physics steps per
  Cartesian waypoint and retains the `0.05 rad` fail-closed arm-error limit.
- Research boundary: completing this primitive closes an execution gap in the
  covered-container pilot; it does not make the run transfer-ready or a final
  paper result. Learned post-removal perception, physical `grasp_inside`, lab
  calibration, held-out testing, baseline/ablation evaluation, and real-robot
  validation are still required.

## 2026-08-05 — GPU watchdog owns the child single-GPU environment

- Decision: the watchdog accepts an explicit physical GPU index and authors
  the child process environment itself, rather than relying on shell-prefixed
  environment assignments.
- For GPU 0 runs it sets only GPU 0 visible, removes rank/world-size variables,
  and monitors compute and graphics contexts on physical GPUs 1–5.
- Any unavailable NVIDIA monitor or forbidden child GPU context remains a
  fail-closed termination, not a reason to run without monitoring.

## 2026-07-22 — Final robot platform follows Professor Park's lab

- Decision: The final simulated and physical robot will be the manipulator used for confirmed real experiments in Professor Shinkyu Park's lab.
- Supersedes: The previous requirement that SO-ARM 101 must be the final robot.
- Reason: Real-robot experiments in Professor Park's lab are confirmed, so matching simulation and physical platforms reduces transfer risk.
- Current unknowns: exact robot model, gripper, camera configuration, control interface, and available Isaac Sim/URDF/USD assets.
- Implementation consequence: build the scene and research loop behind a replaceable robot interface; do not select a substitute robot until the exact lab platform is confirmed.

## 2026-07-22 — Paper-based provisional simulation stack

- Decision: use UR10e + OnRobot RG6 + wrist-mounted Zivid 2 RGB-D camera as the provisional Isaac Sim stack.
- Evidence: *Toward Accurate Long-Horizon Robotic Manipulation: Language-to-Action with Foundation Models via Scene Graphs* explicitly reports this hardware setup.
- Confidence: greater than 99% that the paper used this stack; availability for the present project remains unconfirmed.
- Boundary: this is a provisional engineering assumption, not a final claim about lab allocation. Robot, gripper, and camera remain replaceable through configuration.
- No substitution: do not silently replace the RG6 with a Robotiq gripper merely because Isaac Sim provides a ready-made Robotiq accessory.

## 2026-07-23 — Provisional viewpoint-selection heuristic

- Decision: use a configurable engineering heuristic to exercise the view-scoring and selection pipeline before the research risk metric is finalized.
- Formula: 75% target image-area score plus 25% valid-depth ratio, gated by target visibility.
- Current threshold: 0.8; if the center view is below it, select the higher-scoring view from left and right.
- Boundary: this is not a paper metric, baseline definition, calibrated uncertainty, or MPC objective. Changing it into any of those requires an explicit research decision.

## 2026-07-23 — Draft uncertainty-aware Scene Graph interface

- Decision: define a versioned provisional data interface before selecting the final VLM or uncertainty equation.
- Node belief stores semantic-class distribution, task-conditioned target probability, uncertainty record, and provenance.
- Relation-edge belief stores a full relation distribution, uncertainty record, and provenance instead of one hard label.
- Boundary: entropy, calibration, task-failure risk, belief update, and MPC equations remain unresolved and must not be inferred from the illustrative example values.

## 2026-07-23 — Rule-based probability stub

- Decision: use a deterministic visible-fraction rule to populate the draft belief interface before VLM inference is available.
- Target rule: clamp `0.5 + 50 * target_visible_fraction` to `[0.5, 0.95]`.
- Relation rule: clamp `0.45 + 60 * target_visible_fraction` to `[0.45, 0.95]`; distribute the remaining mass equally over the other allowed relation labels.
- Temporary uncertainty score: `1 - max_probability`, marked uncalibrated.
- Ground-truth boundary: configured object identities and the configured nominal `inside` relation are used, and provenance sets `ground_truth_used_for_control` to true.
- Research boundary: this exists only to test data flow into later belief update and MPC components. It is not a paper metric, baseline, VLM output, or final evaluation method.

## 2026-07-24 — Multi-view belief-update stub

- Decision: replay left, center, and right rule-based graph beliefs and combine matching distributions by normalized multiplication.
- Temporary assumption: observations from different views are conditionally independent.
- Temporary execution gate: target probability and required `inside` relation probability must both be at least `0.9`.
- Purpose: verify temporal graph updates, uncertainty logging, and a future controller's stop-or-reobserve interface.
- Boundary: repeated or correlated observations can make normalized multiplication overconfident. The fusion rule, entropy reporting, and threshold are not final paper methods or metrics.

## 2026-07-24 — One-step Active View Controller stub

- Decision: select between the fixed left/right observation poses from the center view using a configurable one-step utility.
- Temporary utility: weighted expected target-entropy reduction plus expected relation-entropy reduction minus mean joint-motion cost.
- Output interface: emit one `move_to_observation_pose` action request, then evaluate the temporary execution gate on the replayed post-action belief.
- Current result: select the right observation pose.
- Boundary: candidate outcomes are read from previously captured, ground-truth-derived stub graphs. This makes the current predictor oracle-style; it is not MPC, online active perception, or valid evaluation evidence.
- Next integration boundary: execute the action request in Isaac Sim, capture only the selected new observation, and replace outcome replay with a causal predictor before final experiments.

## 2026-07-24 — Isaac Sim Active View action execution

- Decision: add an opt-in `--execute-action-request` runtime mode to the existing deterministic scene loader.
- Runtime sequence: capture center, rebuild its graph/stub, run the one-step controller, apply the selected observation pose, capture the selected view, and rebuild its graph/stub.
- Verification: compare all six requested and measured UR10e joint values and require maximum absolute error no greater than `0.02 rad`.
- Current verified result: the right pose was reached with maximum error `0.000148504 rad`, and fresh RGB-D/graph outputs were produced.
- Motion boundary: `set_pose` directly writes joint positions and position targets. This validates the action interface and observation loop, not a continuous or collision-checked trajectory.
- Research boundary: the candidate predictor remains offline/oracle-style and the motion is not MPC.

## 2026-07-24 — Interpolated observation-pose transition

- Decision: replace the direct active-view pose jump with joint-space interpolation using position targets and physics updates.
- Provisional limits: maximum configured joint step `0.02 rad`, three control frames per waypoint, final tolerance `0.02 rad`.
- Every waypoint is checked against the UR10e articulation's reported lower and upper joint limits.
- Collision policy: prefer PhysX contact-force monitoring with a `5 N` abort threshold; if the contact view is unavailable, use per-frame world-AABB overlap against the table, container, target, and distractor.
- Current result: 15-waypoint Center-to-Right motion completed with no AABB collision and final trajectory error `0.000017715 rad`.
- Boundary: this is a deterministic reference trajectory, not MPC. AABB checks do not replace narrow-phase collision checking, and the provisional kinematic RG6 mount is excluded from swept-volume evaluation.

## 2026-07-24 — Separate paper-facing benchmark environment

- Decision: preserve `open_container_minimal.usda` for debugging and add `open_container_benchmark.usda` as a separate paper-facing scenario profile.
- Benchmark design: laboratory context plus task-relevant clutter, partial target occlusion, inside/outside/boundary/behind relations, and a similar target candidate.
- Current verified behavior: the center view is ambiguous because the orange cylinder occludes the red target; the right view reveals more target evidence.
- Output isolation: benchmark RGB-D captures are stored under `outputs/benchmark_observations/` and do not overwrite minimal-scene observations.
- Boundary: the environment is a visual/scenario prototype, not yet an approved final evaluation scene.
- Safety guard: benchmark Active View execution remains disabled because the current color-key segmentation cannot produce correct instances for similar red/orange objects.
- Finalization requires correct instance masks, expanded uncertainty graph nodes/edges, seeded scenario generation, physics validation, and alignment with the confirmed real-lab setup.

## 2026-07-24 — Benchmark instance-label fallback

- Decision: use a second temporary emissive-material render pass for benchmark instance IDs while the native RTX annotator is unstable.
- During the ID pass, all non-task geometry is black and eight task entities receive distinct ID materials; original bindings are restored immediately afterward.
- RTX tone-mapping shifts the rendered colors, so classification uses measured rendered prototypes. Two close hue pairs are resolved by disconnected horizontal components in the fixed benchmark views.
- Current verification: all eight task entities have independent IDs; target visibility increases from 45 center pixels to 636 right pixels.
- Added a deterministic expanded benchmark graph containing target, similar candidate, occluder, distractors, boundary object, container, and view-dependent visibility.
- Boundary: this is a controlled simulator fallback, not native instance ground truth and not suitable for real-camera masks. The probabilistic graph and VLM replacement remain pending.

## 2026-07-24 — Expanded benchmark uncertainty-flow stub

- Decision: add replaceable engineering probabilities to all eight benchmark entities without fixing the paper's final uncertainty or MPC equations.
- Node interface: existence belief and task-conditioned target belief; relation interface: configured relation versus `unknown`; graph interface: target distribution, required `inside` belief, and temporary task-failure risk.
- Temporary risk expression: `1 - P(target_red) * P(target_red inside container)`.
- The one-step controller combines predicted risk reduction, target/relation entropy reduction, and mean joint-motion cost, and currently selects the right observation pose.
- Boundary: all probability mappings and objective weights are explicitly labeled stubs. Candidate outcomes use captured simulator ground truth, so this is not online prediction, calibrated VLM uncertainty, final MPC, a paper metric, or evaluation evidence.

## 2026-07-24 — Initial uncertainty and planning research design

- Decision: represent target and relation hypotheses as temperature-calibrated categorical beliefs and use categorical entropy in nats as the only scalar uncertainty summary.
- Decision: update beliefs using Bayesian filtering with action-conditioned positive and negative evidence.
- Decision: planning must use a pre-action future-observation model. Reading already captured candidate-view outcomes during planning is classified as offline oracle replay and must never be called MPC.
- Decision: use a hybrid receding-horizon action interface over viewpoint, uncover, occluder movement, and grasp, introduced according to the approved scenario stages.
- Initial objective: expected task-failure risk minus weighted task-conditioned expected information gain, plus motion and collision costs.
- Initial extensions boundary: temperature scaling first; conformal prediction, CVaR, and full chance-constrained MPC remain optional future work.
- Current implementation boundary: viewpoint and grasp are enabled; likelihoods and initial beliefs are engineering stubs pending VLM logits, a calibration split, and a learned or validated observation model.

## 2026-07-24 — Non-oracle Isaac execution boundary

- Decision: the pre-action planner may consume only the current belief and the action-conditioned observation model.
- The selected-view RGB-D and simulator instance labels become available only after the interpolated robot motion completes.
- Post-action simulator labels are converted to detected/not-detected and relation-evidence symbols, then applied through the same Bayesian likelihood model.
- After every executed viewpoint, movement costs are recomputed from the robot's new joint pose and the planner is called again.
- Verified loop: center plan selected right, actual right evidence updated the belief, and replanning selected center followed by grasp.
- Boundary: the post-action adapter still uses simulator instance labels, and the executed joint interpolation is not the final MPC trajectory solver.

## 2026-07-24 — Shared VLM interchange contract

- Decision: the user's VLM and Hansol's VLM may be implemented independently, but both must use the same versioned input/output contract and evaluation split.
- Inputs contain full RGB, anonymous candidate IDs, bounding boxes, masked crops, binary masks, instruction, and explicit relation queries.
- Outputs contain pre-softmax raw target and relation logits in the exact requested order, plus reproducibility provenance.
- Ground truth is stored separately and is forbidden from model inference and planning inputs.
- Semantic simulator IDs are anonymized to prevent names such as `target_red` from revealing the correct answer.
- Boundary: the current three exported views are contract examples only. Seeded scenario generation and leakage-safe train/calibration/test splits are required before model training or temperature fitting.

## 2026-07-24 — VLM data-source strategy

- Decision: do not manually construct a large VLM training dataset; use pretrained models and suitable existing public datasets for training or adaptation.
- Project-specific Isaac Sim samples are still mandatory for temperature calibration, validation, and final simulation evaluation, but they will be generated and labeled automatically.
- Split isolation is by episode and scene seed, not by image. Multiple views of one episode must stay in the same split.
- Boundary: the current single-seed three-view export is interface-development data only and cannot support training, calibration, or evaluation claims.

## 2026-07-24 — Separate VLM and Isaac Sim environments

- Decision: preserve `/data/wonheekoh/isaacsim_venv` unchanged and use `/data/wonheekoh/venvs/efficient-robotics-vlm` as the separate pretrained-VLM environment.
- Initial VLM environment state: Python 3.10.12 with the standard `venv`
  bootstrap only. This initial state is superseded by the later
  Qwen3-VL-specific installation decision below.
- Interface boundary: Isaac Sim produces RGB-D and the versioned VLM input contract; the VLM environment produces raw target and relation logits through the existing output contract.
- Reason: Isaac Sim currently uses Python 3.12 and its own PyTorch/CUDA stack, while the selected VLM may require a different package set.

## 2026-07-24 — GPU-5-only server headless capture

- Decision: server launches set `CUDA_VISIBLE_DEVICES=5`, select physical Vulkan renderer GPU 5, select PhysX CUDA-visible device 0, and keep multi-GPU disabled.
- Decision: server visual verification uses deterministic left/center/right RGB-D captures plus a short CPU-encoded MP4 summary instead of requiring a GUI.
- Video boundary: the current MP4 holds each observation view for one second. It verifies rendered observations but is not continuous-motion or physics-timing evidence.
- Verified result: Isaac Sim 6.0.1 completed the minimal-scene capture, created valid 640x480 RGB-D for all three views, and wrote a three-second H.264 MP4.
- Known limitation: the provisional kinematic RG6 mount continues to emit joint/contact-monitor warnings and must not be treated as validated gripper physics.

## 2026-07-24 — First pretrained VLM and raw-score definition

- Decision: use `Qwen/Qwen3-VL-8B-Instruct` revision
  `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` as the first pretrained VLM.
- Decision: run BF16 inference with PyTorch SDPA in the separate VLM
  environment and expose only physical GPU 5 as `cuda:0`.
- Decision: define each categorical raw score as the mean of pre-softmax
  next-token logits after cyclically assigning every candidate or relation
  label to every single-token choice letter. This removes prompt-position
  letter bias without applying softmax or using ground truth.
- Input use: the adapter consumes the full RGB, input bboxes, anonymous
  candidate IDs, masked crops, instruction, and ordered relation label spaces.
  A bbox/ID overlay is derived from the input only; ground truth remains
  inaccessible to inference.
- Boundary: the fixed-seed center/right results validate implementation and
  interchange compatibility only. They do not establish accuracy,
  calibration, viewpoint generalization, or a paper baseline.
- Next decision boundary: prompt changes and any LoRA/adaptation choice require
  a multi-episode pilot audit. Calibration must be refit for the exact model,
  prompt version, precision, and score definition used at deployment.

## 2026-07-24 — Provisional close/high wrist viewpoint

- Decision: add `close_high` as a simulation-only candidate and prioritize it
  before left/right when it is present in a pilot manifest.
- The physical camera mount, intrinsics, and hand-eye transform are not changed.
  Only the provisional UR10e joint observation pose changes.
- Verified benefit on the deterministic benchmark: target visibility increased
  about 55 times and pretrained Qwen selected the correct anonymous target.
- Grasp remains gated because Qwen's uncalibrated target-to-container relation
  score favored `near_boundary` over `inside`.
- Real-robot use is forbidden until the lab mount, hand-eye calibration, joint
  workspace, self-collision limits, and collision-free trajectory are verified.
- The external overview camera remains presentation/debugging-only and is not a
  VLM or planner input.

## 2026-07-24 — Anonymous container-reference overlay

- Decision: provide Qwen with a cyan visualization of the anonymous container
  surface derived from the same perception mask interface as other candidates.
- The overlay exposes no target identity or relation label. Ground-truth
  relation annotations remain separate and unavailable during inference.
- This single prompt/input revision replaced further prompt sweeping. It passed
  the target and `inside` pilot thresholds on the deterministic close/high view.
- Boundary: the scores are uncalibrated and must be checked across seeded
  episodes before retaining this exact prompt/input representation.

## 2026-07-24 — Ten-seed single-GPU pipeline pilot

- Decision: use a relation-preserving deterministic generator for seeds 0--9
  and capture left, center, right, and close/high observations per episode.
- The generator varies task-object positions within conservative regions while
  preserving inside, outside, boundary, behind, and occluder roles.
- Qwen inference is sequential, batch size one, pretrained-only, and cached;
  training, LoRA, distributed execution, and parallel VLM jobs remain disabled.
- The 10/10 final-target debug result validates the current pipeline wiring
  only. It is not a final metric because scores are uncalibrated, episodes are
  one scenario family, and candidate observations are replayed after capture.
- Before final evaluation: define episode-level calibration/test splits,
  validate real/sim perception inputs without simulator masks at inference,
  complete live action-observation execution, and validate grasp/contact safety.

## 2026-07-24 — Persistent live Isaac–Qwen action/observation loop

- Decision: use file-based atomic JSON IPC so Isaac Sim remains alive while the
  separate VLM environment performs one batch-size-one Qwen inference.
- A candidate observation becomes available only after the requested UR10e
  viewpoint trajectory finishes and fresh RGB-D is captured.
- Every VLM output remains content-addressed and cached; no distributed,
  parallel, multi-GPU, training, LoRA, or calibration job is introduced.
- Cache identity is independent of session/sample names and absolute paths. It
  uses normalized inference content and ordered asset hashes, then adapts only
  the output sample ID when reusing identical content in a new session.
- Verified seed-0 trace: center, close/high, right, then terminal grasp.
- Boundary: the planner and scores remain uncalibrated pilot components and the
  motion controller remains interpolated joint control rather than final MPC.

## 2026-07-24 — RG6 contact-physics fallback boundary

- Decision: do not present the imported RG6 articulation as functional after
  its Isaac Sim 6 mimic joints diverged numerically outside the declared
  limits.
- For pipeline validation only, use the RG6 visual together with two
  RG6-sized kinematic collision pads, a dynamic target with mass/gravity, and
  high-friction PhysX materials.
- A lift may begin only after both left-target and right-target contact events
  are observed. Success requires at least `0.10 m` measured target lift.
- Verified pilot: 71/71 bilateral contact events and `0.180315 m` lift, with no
  explicit target attachment and no target pose copying.
- The optional live orchestrator executes this grasp stage sequentially in a
  fresh GPU-5-only Isaac process after Qwen selects `grasp`.
- Boundary: this is a contact-validating physics proxy, not repaired RG6 joint
  dynamics, not same-process manipulation, not a final success metric, and not
  approved for real-robot transfer.

## 2026-07-24 — Meeting-video evidence boundary

- Decision: combine stored live observation/replanning footage, a disclosure
  transition, and the verified contact-grasp clip into one CPU-encoded
  1920x1080 H.264 presentation video.
- The transition explicitly labels the grasp as a pilot physics proxy and
  states that it is not final paper evidence.
- Output: `outputs/presentation_demo/full_pipeline_with_contact_grasp.mp4`.

## 2026-07-24 — Actual RG6 articulation supersedes the contact proxy

- The earlier proxy remains a recorded fallback and is not deleted, but it is
  no longer the default terminal grasp implementation.
- The original URDF package references had no local STL payloads. The RG6
  visual/collision STL files shipped with the installed Isaac Sim 6 importer
  were copied into the project asset tree, and the URDF now uses
  repository-relative mesh paths.
- Reimport the RG6 with Isaac Sim 6 and resolve the active USD path through
  `assets/robots/onrobot_rg6/isaac6_import/import_result.json`.
- Drive all six imported mimic-joint targets consistently with the master
  finger target. This removes the opposing zero-target drives that had held the
  hand near zero and permits stable `-0.45` to `+0.45 rad` motion.
- The terminal pilot uses the actual RG6 visual meshes, six-joint articulation,
  and finger collision meshes. An external kinematic mount joint is explicitly
  excluded from the articulation, while all internal RG6 joints remain in it.
- A lift begins only after left and right actual finger-collider contacts are
  reported. The target remains dynamic with gravity and is never attached or
  pose-copied.
- The default live orchestrator and presentation builder now select
  `run_rg6_actual_contact_grasp.py` and its verified video.
- Boundary: this is still one deterministic simulator pilot in a fresh
  post-planning Isaac process. The RG6 mount is kinematic; UR10e coupled
  manipulation dynamics, lab hand-eye calibration, real collision geometry,
  and final unbiased evaluation remain pending.

## 2026-07-25 — Temporary grasp commitment gate and debug overhead view

- Decision: an irreversible grasp request is executable only after at least
  two completed reobservations and when the current debug task-failure risk is
  at most `0.15`.
- This gate is a temporary safety constraint. It must not be described as the
  action-selection policy or as the paper's belief-space MPC contribution.
- Candidate sensing actions continue to be selected by the configured
  action-conditioned future-observation and future-posterior objective.
- Decision: retain the ineffective `close_high` capture as failure evidence
  and add a synthetic overhead wrist pose solely to validate the second
  observation/update/replan interface.
- The overhead pose is invalid for real-robot execution and final simulation
  evaluation until it is replaced by a feasible UR10e pose and a
  collision-checked continuous trajectory.

## 2026-07-25 — Sensing feasibility and Qwen replay boundary

- Decision: remove a viewpoint from executable candidates when its configured
  expected target detection probability is below `0.10`. This is a temporary
  observation-validity constraint; remaining sensing actions are still ranked
  by action-conditioned future belief and task cost.
- Decision: use the anonymous tracker mapping only as a debug interface between
  Qwen candidate IDs and the two-hypothesis planner state. Ground-truth
  relation labels are not read during planning.
- Raw Qwen logits use temperature-one softmax and repeated observations use
  pilot-only product fusion. Their near-one values are not calibrated
  probabilities and cannot support a paper accuracy or safety claim.
- The completed replay is not live action-observation execution because its
  center, right, and overhead images were captured before Qwen planning.

## 2026-07-25 — Fixed tempering and live debug-pose execution boundary

- Decision: use raw-logit temperature `4.0` and observation log-weight `0.5`
  only as fixed numerical tempering for integration debugging. Do not call it
  calibration; later calibration must fit parameters on held-out episodes.
- Decision: when no grasp passes the commitment gate and no usable sensing
  action remains, return `defer` rather than fail or force an irreversible
  action.
- Decision: after the official UR10e USD articulation again diverged during
  continuous motion, use fixed center/right/overhead synthetic wrist
  coordinates for the persistent pipeline validation. Every trajectory record
  must state `actual_robot_motion_executed: false`.
- The successful run proves fresh action-triggered observations and Qwen
  replanning in one persistent simulator process. It does not prove a
  collision-checked UR10e trajectory or same-scene RG6 grasp.

## 2026-07-25 — Reject direct grasp-pose teleport in clutter

- Decision: do not use a direct articulation teleport into the seed-0
  open-container grasp pose as manipulation evidence. The actual RG6 can
  collide with the occluder or container before closure, displacing the arm or
  making the PhysX state non-finite.
- A same-layout lift may proceed only after finite joint state, maximum arm
  tracking error at most `0.05 rad`, bilateral actual-finger contact, and at
  most `0.05 m` target displacement are all verified.
- The next manipulation path must explicitly include an above-container
  pregrasp and collision-checked descent before closure and lift. A
  ground-truth-derived wrist yaw is permitted for debugging but is not a
  planner output and cannot be used as final method evidence.
- Reconstructing the same seed/layout in a fresh process must be described as
  same-layout sequential integration, not same-process end-to-end execution.

## 2026-07-26 — Stop on composite fixed-base instability

- Decision: do not continue from planning to RG6 closure or lift when the
  composite articulation becomes non-finite, exceeds `0.05 rad` arm tracking
  error, or reports an unexpected environment collision.
- The scene-mounted, floating-base-plus-fixed-joint, and transformed-parent
  workarounds are rejected as manipulation evidence because all became
  unstable when the imported composite was offset into the benchmark scene.
- The latest run is a failed integration result, not a partial grasp success.
  Steps after monitored trajectory execution remain blocked and must not be
  skipped or replaced by the earlier standalone lift.
- Before another grasp attempt, repair the URDF/USD root mounting structure and
  pass a home-pose stability smoke test in the benchmark scene with finite
  joints, no unexpected RG6/workbench contact, and arm error at most
  `0.05 rad`. Only then rerun pregrasp, descent, closure, bilateral-contact
  gating, and lift.
- The simulator-ground-truth grasp yaw remains debug-only. It is not a learned
  output, a belief-space MPC action, or valid final-evaluation input.

## 2026-07-26 — Use UR10e base coordinates for the fixed-base asset

- Decision: keep the verified fixed-base composite at its authored origin and
  express the benchmark table, container, objects, and debug cameras in UR10e
  base coordinates. A rigid global coordinate change is preferred to moving an
  imported fixed articulation whose world-anchor relationship is not
  relocatable.
- A manipulation trajectory is accepted only after a final-command settle
  period and the existing finite-state, `0.05 rad` arm-error, unexpected
  contact, bilateral-finger-contact, and target-displacement gates pass.
- The first successful seed-0 contact lift under this convention is run 017.
  This supersedes the mounting-instability blocker for the current pilot, but
  does not authorize final-evaluation claims.
- Before calling the path fully collision-safe, enable validated UR10e
  whole-arm collision geometry and use a motion planner that checks the table,
  container, static scene, and manipulatable objects. Before calling it live
  end-to-end, execute the Qwen observation/replanning and terminal manipulation
  in one persistent scene.

## 2026-07-26 — Require whole-arm contact safety for terminal manipulation

- Decision: whole-arm collision geometry and contact reporting are required
  for the terminal seed-0 manipulation path. Any UR10e/RG6 contact with the
  table, container, or non-target object, or any non-finger robot/target
  contact, blocks closure or lift.
- The bilateral actual-finger contact is the only expected robot/target contact
  used to authorize lift. Unexpected-contact absence is part of both the
  pre-lift gate and final success condition.
- Run 019 validates the current deterministic seed-0 waypoint path under these
  contact rules. It does not establish general collision-free planning across
  other seeds and does not replace a motion planner with a scene collision
  model.

## 2026-07-26 — Preserve the grasp gate and add a fallback observation

- Decision: do not lower the temporary `0.15` irreversible-action risk gate
  when a fresh Qwen render is less confident than an earlier seed-0 render.
  A safe `defer` is a valid pipeline outcome, not an error.
- Decision: after right and overhead have both been observed, enable the
  already validated simulation-only `close_high` wrist view as the final
  fallback sensing action. It must be chosen through the configured
  action-conditioned future-observation/future-belief objective.
- The `close_high` likelihood table and fixed camera coordinate are debug
  models, not fitted calibration and not real-robot-approved geometry. They
  must be replaced or validated before final evaluation or lab execution.

## 2026-07-26 — Same-stage terminal manipulation policy

- Decision: when the live terminal action is `grasp`, preserve the current
  Isaac process and USD stage and run the collision-enabled composite
  UR10e+RG6 executor there. Do not describe a later fresh-process grasp as the
  same end-to-end episode.
- The rigid translation `[0.20, -0.32, -0.76] m` is a coordinate
  re-expression of the existing benchmark prims, not a new random scene.
  The validated fixed-base composite remains at the origin because moving its
  imported world anchor previously made PhysX unstable.
- Decision: for this persistent insertion only, author the safe airborne
  `wrist_3` joint at the validated pregrasp roll before the first physics
  frame. This avoids the measured Isaac Sim 6.0 same-stage roll-drive stall.
  All position-changing arm joints, descent waypoints, collisions, tracking
  errors, bilateral contacts, and lift remain physically executed and gated.
- Run 008 is accepted as a successful deterministic same-process integration
  pilot. It is not accepted as calibrated accuracy, general belief-space MPC,
  multi-seed robustness, final simulation evaluation, or real-robot evidence.

## 2026-07-26 — Use the stable composite for physical re-observation

- Decision: use one stable 12-DOF UR10e+actual-RG6 composite articulation for
  both physical re-observation and terminal manipulation. Do not resume using
  the standalone official UR10e asset in this mounted benchmark scene unless
  its disjoint end-effector joint and pause/play instability are repaired and
  revalidated.
- Decision: observation motion must abort on non-finite or out-of-limit joints,
  excessive final tracking error, or overlap between one of the discovered
  moving collision shapes and a configured scene obstacle. An empty discovered
  collision set is an error, not a successful collision check.
- Decision: the terminal grasp executor reuses the live composite and its
  measured final observation pose. It must not translate the environment a
  second time, insert a replacement robot, or describe a fresh process as the
  same episode.
- The current observation controller is a pipeline-validation executor, not
  the claimed general belief-space MPC motion optimizer. Its camera position
  follows the physical RG6 base, but the look-at orientation remains a
  provisional simulation aid until lab Zivid 2 hand-eye calibration provides a
  rigid mount transform.
- Run 009 is accepted as one successful deterministic actual-motion
  end-to-end integration pilot. It is explicitly excluded from calibration,
  multi-seed statistics, final unbiased testing, and final paper claims.

## 2026-07-26 — Separate automatic-IK physics smoke from Qwen authorization

- Decision: a new-seed grasp trajectory may be tested without Qwen only under
  the explicit `automatic_ik_smoke` label. Such a run uses simulator ground
  truth for physics debugging and must not be described as VLM-selected or
  full end-to-end execution.
- Decision: a normal live-to-grasp execution must provide a completed
  `grasp` trigger for the same seed as the physics scene. A seed-0 decision may
  not authorize a seed-1 manipulation.
- Seed-1 run 001 validates that the current grasp/lift IK generator is not
  limited to the stored seed-0 waypoint sequence. It does not validate RGB-D
  localization, calibrated belief, or the complete Qwen loop on seed 1.

## 2026-07-26 — Gate automatic grasp on masked RGB-D localization

- Decision: save camera calibration with each RGB-D capture and compute target
  world position by backprojecting the selected instance's metric depth. The
  estimator must not read simulator coordinate ground truth.
- Simulator ground truth may be read only after estimation to produce a debug
  error metric. For the current pilot, an error above `0.02 m` blocks physical
  grasp execution rather than silently substituting the ground-truth pose.
- Decision: pass the RGB-D target and occluder estimates into grasp-pose and IK
  generation while leaving the simulator layout as the independently created
  physical scene. Do not move the target to the estimate.
- Decision: retain a close overview by increasing resolution to 1920x1080 and
  changing the viewpoint only modestly. Frame the complete manipulator by
  raising the camera and look-at center instead of moving the camera far away.
- Seed-1 RGB-D run 001 is accepted as a successful geometry-to-contact-grasp
  pilot. It is not a Qwen-selected episode because its semantic instance mask
  is still supplied by the simulator debug ID pass.

## 2026-07-26 — Require selected-mask RGB-D for nonzero-seed terminal grasp

- Decision: a nonzero-seed same-stage terminal grasp must receive the
  Qwen/planner-selected anonymous candidate mask localization. It may not fall
  back to the stored seed-0 trajectory or silently read the scene target
  coordinate.
- Decision: use the selected target mask and occluder mask to generate dynamic
  grasp orientation, descent IK, and lift IK. Preserve the actual target prim
  at its independent scene pose; never move it to the perception estimate.
- Decision: reject terminal execution if Qwen selects a candidate for which
  the current physical scene has no supported manipulable prim, if required
  masks are missing, if estimates are non-finite, or if a localization path
  escapes the live-session directory.
- Seed-1 run 001 is accepted as successful same-process, same-stage
  Qwen-selected-mask RGB-D to dynamic-IK contact-grasp integration. It remains
  a pilot because the anonymous masks are simulator-generated and confidence
  calibration and multi-seed testing have not been performed.

## 2026-07-27 — Use modular learned proposals plus Qwen ranking provisionally

- Decision: do not use Qwen3-VL direct bounding-box generation as the sole
  learned perception path. In the fixed nine-view debug pilot it produced a
  correct-IoU box on only three observations and abstained on five difficult
  views, although its close-view boxes were useful when available.
- Decision: retain `GroundingDINO-Base -> SAM2.1-Large -> anonymous candidate
  masks -> Qwen3-VL target/relation ranking` as the current reproducible local
  fallback and integration path. It removed simulator instance masks from
  inference and selected the target mask on all nine debug observations.
- This is not a final model freeze. GroundingDINO's uncalibrated low threshold
  produced many duplicate and false-positive proposals, so class-agnostic
  deduplication, held-out threshold calibration, and final household-object
  scenes are required before calibration or testing.
- Decision: keep the already implemented SAM3 path pending official checkpoint
  access. Do not use an unofficial mirror to bypass Meta's gated-model 403.
  Compare official SAM3 against the modular fallback on the same frozen images
  before selecting the final perception model.
- Decision: simulator masks, IDs, depth, and semantic labels may be used only
  after inference for pilot scoring and RGB-D error measurement. Learned
  inference must consume RGB plus model-produced proposals only.
- The seed-0 through seed-2 grounding results are pipeline-validation evidence
  only. They do not justify a paper claim, confidence calibration, final
  threshold selection, or statistical performance guarantee.

## 2026-07-27 — Factor target identity and relation evidence

- Decision: preserve a detector-box RGB crop for VLM semantic reasoning and a
  separate SAM mask for geometry. Do not black out a contrasting logo merely
  because a class-conditioned segmentation mask excludes it.
- Decision: represent target identity evidence and `inside/outside/behind`
  relation evidence as separate Scene Graph beliefs. Do not conflate both into
  one candidate-match question before calibration or planning.
- Decision: a relative argmax over candidates does not authorize grasp when
  every absolute identity match-minus-nonmatch logit is negative or otherwise
  below an uncalibrated commitment gate. Re-observation, defer, or calibration
  must remain available.
- Decision: exclude the current provisional `left` and `close_high` joint
  configurations from household perception experiments until their wrist
  camera framing and collision-safe UR10e reachability are repaired. A
  synthetic diagnostic camera may not be reported as a reachable planner
  action.
- The seed-0 center-to-right recovery is accepted as perception-pipeline
  debugging evidence only. The right view recovered the correct target,
  `inside` relation, and metric position, but the fixed transition is not the
  proposed task-risk-aware belief-space MPC.

## 2026-07-27 — Use a scanned basket and three validated wrist views

- Decision: use the existing textured LIBERO basket scan for the next
  household perception pilots instead of presenting the procedural slatted
  container as a realistic basket. Keep the source asset in its existing
  checkout and record attribution and exact provenance in every scene result.
- Decision: make both red candidates observably present in the deterministic
  identity/relation pilot. A candidate hidden from all available views tests
  detector recall but cannot validate two-candidate VLM ranking.
- Decision: accept `center`, `right`, and `close_high` as simulated
  UR10e-wrist actions for this pilot because the composite articulation now
  executes both transitions with collision monitoring and bounded joint error.
  This does not approve those joint configurations for the real lab robot.
- Decision: do not authorize grasp or contact using the scanned basket until
  its collision geometry is added and validated. Visual-mesh perception
  success is not manipulation-physics success.
- The 3/3 target/relation result validates perception on this one corrected
  deterministic scene only. Because the initial view is already solved and
  the view sequence is fixed, it is not evidence for active re-observation or
  task-risk-aware belief-space MPC. The next method experiment must introduce
  controlled initial occlusion and compare predicted action-conditioned belief
  outcomes before selecting the first re-observation action.

## 2026-07-27 — Treat controlled-occlusion planning as an engineering prototype

- Decision: use controlled occlusion to create a small relation-ambiguity
  scenario before adding cover or lid manipulation. The initial belief must
  come from learned proposals and Qwen scores, not a manually assigned
  confidence.
- Decision: pre-action selection may use the currently documented
  geometry-informed likelihood model for the deterministic pilot because the
  approved research plan permits simulator likelihoods in the first
  implementation. It must remain labeled hand specified and uncalibrated.
- Decision: planning code must write the pre-action plan before opening the
  selected future perception output. Unselected future images or Qwen results
  may be generated for post-hoc evaluation but may not be read during action
  selection.
- Decision: associate anonymous candidates across views using learned masks,
  metric depth, camera calibration, and nearest 3D track distance. Do not use
  simulator instance IDs to preserve cross-view identity during inference or
  planning.
- The selected `center -> close_high` action, actual collision-checked motion,
  posterior update, and safe replan are accepted as one first integration
  step. They are not accepted as the final proposed method: the observation
  model is not calibrated or learned, the planner is not yet an actual MPC
  solver, and the posterior risk still blocks grasp.
- Decision: do not lower the grasp threshold merely to make this pilot end in
  a grasp. Execute the requested second observation or improve/calibrate the
  belief model before irreversible commitment.

## 2026-07-27 — Separate planner grasp request from physical authorization

- Decision: preserve the planner's final `grasp` request when the debug risk
  reaches `0.14912`, but do not equate it with permission to execute. A
  physical-action gate independently requires validated container collision
  geometry and an approved confidence/calibration status.
- Decision: treat a risk value only `0.00088` below an uncalibrated threshold
  as a sensitivity warning, not robust evidence of safe commitment. Do not
  tune or round the value to manufacture a successful grasp.
- Decision: accept run 006 as a two-reobservation sequential integration
  replay because each plan was saved before its selected observation output
  was opened and the same planned camera sequence was executed collision-free
  in one Isaac process.
- Run 006 is not the final online controller. The current inference/planner
  replay occurs after the simulator has captured the sequence, the observation
  model is hand specified, and the planner reports `actual_mpc_solver=false`.
- Before executing the resulting grasp in the scanned-basket scene, add and
  validate basket collision geometry in an explicitly physics-only smoke test.
  Before using the risk gate as a research result, fit calibration on separate
  held-out episodes and freeze the threshold before final testing.

## 2026-07-27 — Keep perception and manipulation-clearance basket variants separate

- Decision: retain the `2.10`-scale scanned basket for perception experiments,
  but do not claim it is RG6-accessible. Physics runs showed that the RG6
  fingers/knuckles contact its walls even when the grasp is raised by 4 cm.
- Decision: use a separately labeled `3.20`-scale scanned-basket variant for
  the first collision/contact manipulation smoke. Its visible mesh and
  five-box collision approximation must be scaled together, and later
  real-world experiments must select and measure a container with comparable
  gripper clearance.
- Decision: never disable or shrink container collision merely to make a grasp
  pass. Preserve the narrow-scene failures and exact collider pairs as design
  evidence.
- Decision: household visual replacements must receive an explicit physical
  proxy. For the mug pilot, use a rigid Xform root plus hidden cylinder
  collider so the visible mesh and physical body move together.
- Decision: accept scanned-basket collision grasp run 006 only as a
  superseded contact/lift code smoke. The realistic-proxy audit showed that it
  does not authorize a claim of physically credible grasp.

## 2026-07-27 — Drive RG6 through one master and preserve failed safety runs

- Decision: command only `rg6_finger_joint`. Remove angular drives from the
  five `NewtonMimicAPI` followers in the in-memory experiment stage and let
  the mimic constraints propagate motion. Do not modify the imported source
  USD merely to pass a debug run.
- Decision: use the visible mug body's outer radius, a provisional `0.30 kg`
  mass, and moderate provisional friction. Do not shrink the collision proxy,
  reduce mass, raise friction, attach the target, or copy its pose to
  manufacture lift.
- Decision: keep the `60 N` force and `3 mm` penetration ceilings. Preserve
  runs 007–014 as failed diagnostic evidence; do not delete or relabel them.
- Decision: accept run 015 as the new seed-0 physics-grasp smoke because
  bilateral contact remained active through a `0.17970 m` lift, measured
  force and penetration stayed within the fixed limits, horizontal slip was
  `0.000335 m`, and attachment/pose-copying were absent.
- Decision: the `0.60 N·m` master limit, mug material parameters, cylinder
  collider, and force gate remain provisional simulation engineering values.
  Before a paper claim or real-robot transfer, measure/validate RG6 control,
  fingertip material, object mass/geometry, and contact sensing. Do not treat
  run 015 as calibration, belief-space MPC, multi-seed, or final evaluation.

## 2026-07-27 — Preserve unseen hypotheses in live learned perception

- Decision: do not require every persistent target hypothesis to be detected
  in every re-observation. When a tracked candidate is outside the current
  field of view, retain it with zero log evidence instead of deleting it or
  forcing a detector proposal.
- Decision: reject learned candidate-to-track assignments farther than
  `0.08 m` in the current controlled pilot. An unmatched detector proposal is
  excluded from the belief update, while the unmatched persistent hypothesis
  remains in the belief state.
- Decision: exclude orange-cylinder proposals whose predicted mask overlaps
  the selected target mask by IoU `0.10` or more before using the occluder for
  RGB-D grasp-orientation estimation. This prevents a cross-class duplicate
  target proposal from becoming the grasp occluder.
- Run 004's final `defer` is accepted as the correct safe pipeline outcome.
  Do not lower the `0.15` temporary commitment threshold to manufacture a
  grasp. The next research step is episode-separated score/threshold
  calibration and proposal deduplication before another learned-perception
  grasp authorization attempt.
- Boundary: zero log evidence, the `0.08 m` association gate, and the `0.10`
  overlap gate are deterministic pilot engineering choices, not calibrated
  final-method parameters or paper claims.

## 2026-07-27 — Freeze an episode-separated calibration pilot

- Decision: use simulator seeds `100–109` only for the first automatic
  calibration pilot and reserve seeds `200–209` for a later unbiased test.
  Never mix views from one episode across calibration and testing.
- Decision: run `center`, `close_high`, and `right` observation capture first,
  then GroundingDINO, SAM2, and Qwen sequentially. Load one model instance at
  a time on physical GPU 5 with batch size one; do not use DDP, DataParallel,
  NCCL, fine-tuning, LoRA, or a hyperparameter sweep.
- Decision: keep simulator ground truth inaccessible to inference and use it
  only post hoc to label predicted candidate masks for calibration and
  diagnostics. Preserve all raw model outputs and failed episodes.
- Decision: fit target match-versus-nonmatch temperature from the factorized
  Qwen choice logits. Do not describe normalized or temperature-scaled values
  as success probabilities until held-out calibration quality has been
  validated.
- Decision: because this batch has only `inside` scene labels, do not accept
  its relation-temperature fit as final. A later calibration set must cover
  `outside`, `behind`, occlusion, and insufficient-evidence/unknown cases
  before freezing the relation observation model or commitment threshold.

## 2026-07-27 — Do not freeze calibration from the first 10-seed batch

- Decision: accept the seed `100–109` batch as a successful execution and
  instrumentation pilot, but not as adequate calibration evidence. Its
  `31:3` target/non-target proposal imbalance and single `inside` relation
  label do not cover the observation model required by the project.
- Decision: do not report Qwen's `30/30` selected-target result as ambiguous
  target accuracy. `28/30` observations exposed only one candidate, so the
  result is dominated by proposal availability rather than target ranking.
- Decision: the next calibration-scene generator must enforce measurable
  coverage before inference: at least two visible same-class candidates,
  balanced target/non-target proposal labels, and episode-level coverage of
  `inside`, `outside`, `behind`/occluded, and insufficient-evidence cases.
- Decision: preserve temperature `5.65` and its metrics only as a diagnostic
  showing overconfident raw margins. Do not use it to authorize grasp or freeze
  the MPC commitment threshold until an episode-separated, label-diverse
  calibration set and an untouched test split have been evaluated.

## 2026-07-28 — Treat proposal absence as an observation, not a batch error

- Decision: use physical GPU 4 only for the explicitly requested foreground
  pilot and keep physical GPU 5 as the code's default. Require
  `PHYSICAL_GPU`, `CUDA_VISIBLE_DEVICES`, renderer active GPU, and result
  metadata to agree; physics remains logical `cuda:0`.
- Decision: use the perception-scale basket for perception calibration.
  Do not reuse the larger grasp-clearance/collision basket when its geometry
  moves a distractor outside the wrist-camera framing.
- Decision: do not require two detector proposals in every active-perception
  view. A strongly occluded target can be absent from the detector output;
  retain that view as negative/missing evidence and rely on re-observation to
  expose the target. Report both conditional ranking accuracy and full
  per-view selection accuracy.
- Decision: for the moved rear red mug, resolve its shared color-ID prototype
  with the small boundary marker by connected-component area rather than
  horizontal ordering. Preserve the original generated GT before offline
  reclassification and never alter inference outputs during GT correction.
- Decision: accept seeds `110–113` only as an inside/outside calibration
  instrumentation pilot. Do not freeze target temperature `2.85`, relation
  temperature `2.90`, or a grasp threshold until `behind` and `unknown`
  coverage, more episode diversity, and a separate validation/test split are
  available.

## 2026-07-28 — Validate occluder clearance before rendering

- Decision: do not use a fixed large cylinder pose for active occlusion in the
  scanned basket. Compute the pose from the current target and center-camera
  direction and require explicit basket-wall and target-surface clearances.
- Decision: require at least `8 mm` analytic wall clearance and `3 mm` target
  surface clearance over the seeded generator range. Abort scene generation
  instead of rendering an intersecting layout.
- Decision: preserve the existing USD Cylinder schema and use nonuniform
  scaling for the thin oriented plate. Do not replace a live prim with another
  schema at the same path; run 003 showed that this can stall Isaac stage
  initialization.
- Decision: accept seed-111 run 005 as the corrected active-occlusion scene
  smoke. It validates visual separation and reachable re-observation views,
  not learned inference, belief update, MPC, grasp, or final evaluation.

## 2026-07-28 — Reject approximate-bound clearance as visual validation

- Correction: withdraw acceptance of run 005. Clearance against an approximate
  basket bounding box does not prove separation from an irregular scanned
  weave mesh, and the remaining faceted visual is not acceptable.
- Decision: disable the explicit primitive occluder and use run 006 only as a
  penetration-removal smoke. Do not feed runs 001–005 into later calibration,
  testing, figures, or demonstrations.
- Decision: create active uncertainty using either a lower/shallower reachable
  wrist view with the basket rim as the occluder or a separately validated
  realistic household object with mesh/collision separation. Visually inspect
  every final scenario view before launching VLM inference.

## 2026-07-28 — Treat household-mug Xform origin as bottom contact

- Decision: the procedural household mug Xform origin is the mug bottom.
  Never add half the mug height when placing that Xform on a support surface.
- Decision: record the bottom-contact point and object-center point as separate
  fields. RGB-D centroid evaluation may use `target_red_settled`, while scene
  placement must use `target_red_bottom_contact`.
- Decision: for the current scanned basket, use the mesh-validated `0.020 m`
  interior support offset and the flatter `+0.020 m` target Y offset. Run 007
  supersedes run 006 for geometry inspection only.
- Decision: do not launch learned perception on run 007. First obtain a
  physically reachable center wrist pose that creates partial occlusion
  without changing the now-validated support contact.

## 2026-07-28 — Freeze the second lower-center pose for the next pilot

- Decision: retain the original center pose as `center_high_legacy`, retain the
  insufficient run-008 pose as `center_low_candidate1`, and use
  `[-1.5416379, -1.3170297, 1.8950480, -1.8708, -1.5708, 0.0] rad` as the
  current simulation center pose.
- Decision: disable `viewpoint_center_repeat` while the scene is static.
  Repeating the identical camera pose has no new evidence and must not receive
  a fictitious information-gain forecast.
- Decision: accept seed-111 live run 002 as the first successful causal
  learned-perception re-observation integration on the corrected scene. It
  authorizes work on the next grasp integration step but does not validate a
  grasp or the final MPC claim.
- Decision: when GPU 4 is shared, retain single-device sequential execution
  but label runtime and utilization as contended debug measurements. Never
  terminate or modify unrelated GPU processes.
- Decision: before enabling live grasp, remove the obsolete explicit-orange-
  occluder dependency from terminal RGB-D localization and validate collision
  geometry at the same perception-scale basket used by the live observations.

## 2026-07-28 — Accept the first clean live contact-grasp integration pilot

- Decision: use the perception-scale scanned basket for both rendering and
  collision in the live grasp pilot. Keep its box collision approximation
  active in physics but invisible in rendering; never show the approximation
  as scene geometry.
- Decision: terminal target localization consumes only the Qwen-selected
  anonymous candidate mask and current RGB-D. Do not require or infer an
  explicit orange-occluder proposal in the corrected basket-rim scene.
- Decision: use the collision-checked table-aligned seed-0 RG6 approach yaw as
  a fixed pilot orientation. This is a provisional grasp-engineering choice,
  not a learned grasp orientation or final planner output. Target translation
  must still be recomputed from the current episode's selected RGB-D mask.
- Decision: retain run 005 as a wrist-roll tracking failure and run 006 as a
  successful physics result with a top-level timeout. Do not relabel or delete
  either diagnostic.
- Decision: use a configurable `1800 s` grasp-result wait on a shared GPU so
  slow high-resolution frame capture and CPU video encoding are not mistaken
  for physics failure.
- Decision: accept seed-0 run 007 as the first clean, same-process
  learned-perception/re-observation/dynamic-IK/contact-lift integration pilot.
  It authorizes multi-seed debugging and calibrated planner work, but it is
  not a final experiment.
- Decision: do not claim that run 007 demonstrates uncertainty-driven
  belief-space MPC. The current planner metadata says
  `actual_mpc_solver=false`, calibration is absent, and its minimum-one-
  reobservation gate can request close-high even when the raw seed-0 Qwen
  belief is already nearly one. Final evaluation requires calibrated held-out
  beliefs and action-conditioned future-observation comparison without that
  debug-only forced-reobservation rule.

## 2026-07-28 — Replace forced re-observation with a belief-tree MPC pilot

- Decision: use `belief_tree_mpc` for the next method-development runs. Each
  future observation branch must choose an independent continuation action;
  fixed open-loop action-sequence enumeration is retained only for historical
  diagnostics.
- Decision: set `minimum_completed_reobservations=0`. Re-observation must be
  selected by predicted future belief and task cost, not by a mandatory
  observation count.
- Decision: keep an irreversible grasp risk gate as a separate safety
  constraint. The current `0.06` limit is a provisional deterministic-pilot
  value only and must be replaced or justified after episode-separated
  calibration.
- Decision: accept seed-111 run 003 as the first successful live execution of
  the discrete belief-tree MPC solver: it selected right, executed the actual
  right-view motion, updated belief from the new observation, and then selected
  grasp.
- Decision: do not combine run 003 and seed-0 run 007 into a false claim of a
  fully validated final method. Run 003 validates MPC selection without
  contact execution; run 007 validates contact execution with the older debug
  planner. The next integration milestone is a calibrated configuration that
  joins the new solver and contact grasp in one episode.
- Decision: label `actual_mpc_solver=true` as an implementation fact only.
  Final method/evaluation validity remains false until the action-conditioned
  observation model and risk gate are calibrated on held-out episodes and the
  frozen test split, baselines, and ablations are run.

## 2026-07-28 — Reject the easy four-seed calibration fit

- Decision: preserve seeds `120–123` as a Phase-1 calibration-plumbing pilot,
  not an accepted final calibration set. Keep all views from each seed in the
  same split and keep reserved test seeds `200–209` untouched.
- Decision: never apply a temperature fit automatically when it lands on a
  search-grid boundary, contains fewer than 20 episode-disjoint scenes, or
  has no hard identity errors. The `0.25` target temperature from this batch
  is rejected.
- Decision: do not fit or deploy a final relation temperature unless
  `inside`, `outside`, `behind`, and `unknown` are represented under the
  corrected factorized relation definitions. The `0.25` relation temperature
  from this inside/outside-only batch is rejected.
- Decision: do not modify the MPC observation likelihoods or provisional
  `0.06` task-risk gate from this batch. Record the rejected artifact path in
  the MPC config so later work cannot mistake it for an applied calibration.
- Decision: the next automatic generator must deliberately create hard
  same-class identity cases, proposal-missing negative evidence, inside and
  outside membership, view-dependent behind/occlusion, and insufficient-
  evidence unknown labels. Membership and independent relations must be
  calibrated separately rather than as one mutually exclusive label.

## 2026-07-28 — Accept the factorized scene generator for calibration capture

- Decision: accept the deterministic `inside_clear`, `outside`,
  `rim_occluded`, and `covered_unknown` variants as the scene-generation basis
  for the next bounded calibration batch. The four GPU-0 RGB-D smokes passed
  both visual inspection and simulator-mask visibility gates.
- Decision: define physical membership only as `inside` or `outside`.
  `unknown` is a view-observability output, not a third physical state.
  Calibrate membership, `behind`, and `occluded_by` as separate factors.
- Decision: generator-derived world labels and view-observable intent must be
  saved before inference. Measured instance-mask visibility may validate the
  generated scene only after capture; it must never be supplied to Qwen or
  the planner.
- Decision: use the basket rim for partial occlusion and the validated simple
  cover for no-evidence scenes. Keep the arbitrary orange occluder disabled.
  Abort a future capture if the rendered visibility gate fails.
- Decision: do not run the 20-scene VLM batch with the old mutually exclusive
  relation-calibration code. First update its records, metrics, and fitting to
  consume the factorized schema. Keep all views from one seed in one split and
  keep reserved test seeds `200–209` untouched.
- Decision: the four scene smokes are generator validation only. They do not
  improve, calibrate, train, or test Qwen and must not be reported as method
  performance.

## 2026-07-28 — Retain partial calibration fits but do not deploy them to MPC

- Decision: accept the seeds `128–147` run as a completed 20-scene
  episode-separated calibration pilot. Preserve seeds `200–209` as untouched
  final-test candidates. Do not call the calibration batch a test result or a
  paper-scale evaluation.
- Decision: retain target temperature `5.85` and membership temperature
  `4.525` as accepted component-fit candidates. They had all required
  candidate-level classes, non-boundary fits, and hard errors. Do not yet
  write either value into the MPC configuration.
- Decision: reject `behind=0.625` and `occluded_by=0.85` for deployment because
  their scored records lack legitimate `unknown` examples. A fully hidden
  target with no detector proposal is an observation-model miss, not a Qwen
  `unknown` factor score.
- Decision: create a small additional calibration-only scene family in which
  the target remains detectable but the current view does not support a
  reliable `behind` or `occluded_by` decision. Do not manufacture unknown
  labels or expose simulator geometry to Qwen.
- Decision: treat the four outside-view target-selection errors and the five
  membership-unknown errors as real method-development evidence. Temperature
  scaling may adjust confidence but must not be described as correcting the
  wrong argmax predictions.
- Decision: keep proposal missingness separate from candidate-level Qwen
  confidence. The observed split was 45/45 target proposals when target pixels
  were visible and 0/15 when the target was fully covered. Estimate and
  validate this observation likelihood separately before using it in
  action-conditioned future-belief prediction.
- Decision: no calibration component may be applied automatically to MPC until
  all required factors, the task-risk gate, and the action-conditioned
  observation model pass their own gates. The current `calibration_fit.json`
  records component acceptance and system-level deployment blocking reasons
  separately.

## 2026-07-28 — Reject Qwen-only relation calibration after the GPU-5 extension

- Decision: retain `behind_ambiguous` as a hard calibration-only scene family.
  Preserve seed 148 run 001 as a rejected visual diagnostic and run 002 as the
  accepted geometry/visibility smoke. Pixel-count gates do not replace direct
  inspection of a newly designed scene family.
- Decision: accept seeds `128–155` as a completed 28-episode calibration
  batch, but not as testing or final-paper evaluation. Preserve reserved test
  seeds and do not use them to redesign prompts or fit calibration.
- Decision: require per-class recall diagnostics in addition to aggregate
  accuracy and NLL before a component fit can be deployed. A supported
  required label with zero recall is a blocking failure even if class
  imbalance makes aggregate accuracy appear high.
- Decision: reject membership temperature `3.875` because `unknown` recall is
  `0/13`; reject `behind` temperature `1.725` because `unknown` recall is
  `0/8`; and reject `occluded_by` temperature `3.95` because `yes` recall is
  `0/13`. These values remain reproducibility artifacts only.
- Decision: retain target-identity temperature `5.125` as a component-fit
  candidate, but do not apply it to MPC until the complete observation model,
  task-risk gate, and action-conditioned belief prediction are calibrated.
- Decision: do not continue treating Qwen choice logits alone as the spatial
  relation observation model. Use Qwen for instruction-conditioned target
  identity, and develop a hybrid relation/abstention model from selected
  SAM masks, RGB-D geometry, basket geometry, visibility, and proposal
  missingness. Qwen relation outputs may remain auxiliary evidence.
- Decision: calibration must not be described as training. This batch used
  simulator ground truth only after inference to adjust/evaluate confidence;
  it did not update Qwen weights. Final unbiased testing remains unperformed.
- Decision: keep the MPC configuration unchanged. Do not run final multi-seed
  comparison, baseline, or ablation experiments until the hybrid observation
  likelihood, action-conditioned future-belief model, and wrong-commitment
  risk gate have passed calibration on episode-separated data.

## 2026-07-29 — Separate RGB-D world relation from RGB-only abstention

- Decision: the final RGB-D pipeline must not be evaluated against an
  `unknown` label authored only from what a human or Qwen can infer from RGB.
  If a learned candidate mask and metric depth reliably locate the object
  relative to a valid basket footprint, the hybrid observation may resolve
  `inside/outside` even when the RGB-only semantic answer is `unknown`.
- Decision: store modality-specific evidence separately:
  `membership_world_evidence`, `behind_camera_relative_evidence`, and
  `occluded_by_reference_evidence`. Do not collapse these into one relation
  label or describe their deterministic engineering scores as calibrated
  probabilities.
- Decision: use Qwen provisionally for instruction-conditioned target identity
  and use learned masks plus RGB-D for relation geometry. Proposal missingness
  remains a separate observation outcome; never invent a relation score for a
  fully hidden, undetected target.
- Decision: allow the geometry adapter to abstain when the learned reference
  mask produces an implausible basket extent or when the candidate is inside a
  declared boundary uncertainty band. Do not silently replace an invalid mask
  with simulator target or basket coordinates.
- Decision: retain the first 139-candidate result as a calibration-only
  engineering audit. Its `137/137` non-abstained world-membership result is not
  a final test metric because the thresholds were inspected on the existing
  calibration batch and the reference frame is axis-aligned simulation.
- Decision: do not calibrate or deploy the current occlusion rule. The legacy
  labels conflict with measurable partial visibility in 35 cases, including
  nominal `inside_clear` images whose basket hides the mug's lower body.
  First freeze an objective occlusion definition based on visible fraction or
  line-of-sight geometry and document any severity threshold.
- Decision: do not consume reserved seeds `200–209` for this method-design
  correction. After the label protocol is frozen, generate a separate
  episode-level validation range, validate the hybrid observation likelihood,
  and only then connect it to the action-conditioned belief tree and
  wrong-commitment gate.

## 2026-07-29 — Adopt objective target-amodal occlusion GT for method development

- Decision: define target occlusion from an aligned simulator-only
  counterfactual: preserve camera and target pose, render the target without
  other scene geometry, and compare the actual target-ID mask with that
  target-only amodal support.
- Decision: intersect the temporary actual color-ID mask with amodal target
  support before counting visible pixels. Preserve raw mask and spill count
  because emissive RTX reflections can otherwise produce physically
  impossible visible areas larger than the target silhouette.
- Decision: use `no < 0.10`, `partial ∈ [0.10, 0.60)`, and `severe ≥ 0.60`
  only as frozen pilot thresholds. For the current binary target
  `occluded_by` calibration audit, map `no -> no` and
  `partial/severe -> yes`. Do not call these values probabilities or final
  thresholds.
- Decision: treat amodal masks and objective occlusion labels as hidden
  simulator GT. They may be read only after inference for validation or
  calibration; they must never be exposed to learned perception, belief
  update, action selection, or MPC.
- Decision: retire the old generator-authored target `occluded_by` intent as
  calibration truth. Preserve it only for historical audit because nominal
  clear center views were measurably 50–56% occluded.
- Decision: keep proposal missingness separate. A fully covered target with
  no learned proposal is an observation outcome, not a fabricated target or
  relation logit.

## 2026-07-29 — Reject objective-occlusion calibration for MPC deployment

- Decision: accept seeds `156–164` as a successful nine-episode
  method-validation/calibration pilot, not as final testing. Reserved seeds
  `200–209` remain untouched.
- Decision: reject Qwen-only target occlusion as the deployed relation model.
  Against objective GT it predicted `no` for all 21 visible target records,
  yielding `0/10` yes recall; its temperature fit also hit the upper grid
  boundary.
- Decision: retain the RGB-D hybrid result as evidence that geometry is the
  correct direction, but do not deploy it. It improved accuracy from `52.4%`
  to `76.2%`, yet recovered only `5/10` objective occlusions and still uses
  uncalibrated thresholds and an axis-aligned simulation reference.
- Decision: retain Qwen for instruction-conditioned target identity and keep
  objective occlusion as a geometry/visibility observation factor. Qwen
  relation logits may be auxiliary evidence but cannot override measurable
  RGB-D geometry.
- Decision: do not apply target temperature `5.45`, membership temperature
  `4.0`, behind temperature `1.225`, or occlusion temperature `8.0` to MPC.
  The batch has fewer than 20 episode-disjoint scenes, minority-class
  failures remain, and the occlusion fit is boundary-clipped.
- Decision: the next method task is to improve the learned-mask/RGB-D
  occlusion observation model on non-test validation scenes, then fit an
  action-conditioned observation likelihood for each reachable view. Only
  after that model and the wrong-commitment task-risk gate pass calibration
  may they be connected to the belief-tree MPC for final baseline and
  ablation evaluation.
- Decision: calibration is not training. Qwen, GroundingDINO, and SAM2 weights
  remained frozen; simulator GT was used only after inference. Final unbiased
  testing and paper-scale evaluation remain unperformed.

## 2026-07-29 — Use reference-attributed occlusion and retain the 20-episode action model

- Decision: do not use total-scene amodal occlusion as the truth for the
  factor named `occluded_by_reference`. Preserve total-scene occlusion as an
  observation-quality diagnostic, but calibrate the reference factor from
  target pixels newly revealed by hiding only `/World/OpenContainer`.
- Decision: retain the 10% reference-attributed threshold and the existing
  partial/severe split only as frozen method-development settings. They are
  not calibrated probabilities or final real-robot thresholds.
- Decision: accept seeds `165–184` as a completed 20-episode calibration and
  action-model pilot. Do not call the results testing, final evaluation, or
  statistical guarantees, and keep reserved seeds `200–209` untouched.
- Decision: reject Qwen-only `occluded_by_reference` deployment because its
  yes recall is `0/13`. Retain Qwen for instruction-conditioned target
  identity and use learned masks plus RGB-D geometry for membership,
  reference occlusion, and safe abstention.
- Decision: retain the hybrid result (`10/13` yes recall, `31/32` no recall,
  and `100%` selective accuracy) as the current observation-model candidate.
  Do not apply it to MPC because it has not been tested on a frozen split and
  its basket frame and thresholds remain simulation-specific.
- Decision: retain the fitted action-conditioned tables as the first
  data-backed future-observation model. They satisfy the minimum five
  episodes per declared cell but cover only the current discrete scene
  families and viewpoint actions. Do not represent them as a general learned
  dynamics model.
- Decision: do not change the belief-tree MPC configuration until the
  wrong-commitment/task-risk gate is calibrated and a frozen test protocol,
  baselines, and ablations are ready. Training remains unperformed;
  calibration and final testing remain distinct.
- Decision: treat any process allocation on physical GPUs other than 5,
  including a small graphics-only Vulkan context, as a GPU-policy violation.
  Suspend new Isaac Sim launches after PID `3823469` created a `4 MiB`
  graphics context on physical GPU 0 despite the documented GPU-5 settings.
  Resume simulation only after host-level Vulkan isolation is verified to
  expose physical GPU 5 exclusively. Existing saved observations may still be
  analyzed, and pure CUDA inference may run only with physical GPU 5 exposed.

## 2026-07-29 — Retain the CPU-only belief-MPC replay as a calibration diagnostic

- Decision: use leave-one-episode-out replay for the first cached MPC
  integration. Exclude the held-out seed from likelihood fitting, prohibit its
  future view from root action selection, and use its simulator labels only
  for post-hoc audit.
- Decision: reject independent visibility and occlusion belief factors for
  this scene. Preserve the failed factorized result, but use a joint
  visibility-reference-occlusion state because low target confidence and
  occlusion evidence are strongly correlated in covered scenes.
- Decision: maintain target identity across views with nearest learned RGB-D
  candidate-center tracks. New Qwen evidence updates the existing track;
  it must not discard a strong center belief merely because a later view
  ranks a different candidate. Simulator IDs may not be used for tracking.
- Decision: do not select `task_noncompletion_cost=0.55` as a final parameter
  even though values `0.55–0.8` produced 15 correct grasps, zero wrong grasps,
  and five safe hidden-target defers in this sensitivity audit. The same
  calibration collection motivated and measured that grid.
- Decision: retain `0.45` as the primary frozen diagnostic setting and report
  its four unnecessary visible-target defers. Report the complete sensitivity
  curve alongside it rather than publishing only the favorable setting.
- Decision: do not claim that action conditioning outperforms the ablation.
  The action-agnostic MPC achieved 15 correct, zero wrong, and five safe
  defers at the primary setting, while the action-conditioned policy achieved
  11 correct, zero wrong, and nine defers. New action-differentiating scenes
  and interaction actions are required to test the claimed benefit.
- Decision: this is an implementation and calibration replay milestone, not
  final experimental evidence. Task-cost calibration, a frozen unbiased test,
  baselines/ablations on randomized scenes, cover removal/lid opening, and
  live robot execution remain outstanding.
- Decision: continue GPU-free work only from saved artifacts while the server
  GPU restriction is active. Do not launch Isaac Sim, Qwen, GroundingDINO, or
  SAM during this CPU replay stage.

## 2026-07-29 — Do not freeze MPC costs from the nested calibration diagnostic

- Decision: retain the four-fold outer/inner leave-one-episode-out protocol as
  a leakage-resistant development diagnostic. It is still cross-validation
  over an already used calibration collection and must not be labeled an
  unbiased test or final-paper evidence.
- Decision: do not freeze `0.55`, `0.65`, or `0.8` as the task noncompletion
  cost. Different outer folds and action-model variants selected different
  values, showing that the 20-scene collection does not determine a stable
  task-risk parameter.
- Decision: reject any claim that the current action-conditioned model is
  better than the action-agnostic ablation. Nested outer-fold replay gave the
  action-conditioned model 12 correct grasps and three unnecessary visible
  defers, while the action-agnostic model gave 15 correct grasps and no
  unnecessary visible defers; neither made a wrong grasp.
- Decision: treat the current view-action support as insufficient. Although
  learned output signatures differed for 10 of 20 episodes, simulator
  post-hoc latent states differed for only one episode and maintained target
  correctness differed for none. Do not interpret perception variability as
  evidence of action-conditioned scene dynamics.
- Decision: the next scene design must balance four causal action outcomes:
  close-high-only resolution, right-only resolution, resolution by either
  view, and no resolution by viewpoint change. The last group must require a
  future cover-removal or lid-opening action.
- Decision: keep reserved seeds `200–209` untouched until the scene families,
  observation model, task costs, commitment gate, baselines, and ablations are
  frozen. No continuous robot motion MPC was executed in this calibration
  experiment.

## 2026-07-29 — Require rendered causal gates for new view-action scenes

- Decision: use four balanced development classes before fitting another
  action-conditioned model: close-high-only resolution, right-only
  resolution, resolution by either view, and failure of both views requiring
  cover removal.
- Decision: treat the camera-ray layouts as geometry initializations only.
  Generator-authored view intent is not evidence that an action works. Accept
  a scene only from rendered target amodal/visible masks after executing the
  corresponding reachable wrist views.
- Decision: freeze the initial smoke thresholds at 65% resolved visibility,
  15 percentage points gain over center, 15 points winner-versus-loser
  separation, and at most 2% visibility for a covered target. Reject a failed
  scene instead of changing its label to match the rendered outcome.
- Decision: use seeds `185–196` only for geometry smoke and method
  development. They are not calibration or test data. Preserve reserved test
  seeds `200–209`.
- Decision: `remove_cover` is currently an intended high-level interaction,
  not an implemented manipulation primitive. A covered scene passing the
  three-view hidden gate does not validate cover opening, belief update after
  removal, or continuous robot MPC.
- Decision: while no compliant GPU is available, stop after generating the
  CPU manifest and tests. When GPU use resumes, validate seed `185` alone
  before launching a batch; any graphics context on a forbidden physical GPU
  remains a policy violation.

## 2026-07-29 — Retain abstract remove-cover MPC as control-flow validation

- Decision: represent the first removable-cover pilot with a joint belief over
  target location and cover state. Keep positive detection, empty-container
  negative evidence, and interaction failure as distinct post-action
  observations.
- Decision: require first-action execution and replanning. A future scripted
  observation may be applied only after the planner fixes and executes its
  action; post-hoc true state may be used only for diagnostic audit.
- Decision: retain the negative-evidence ablation. In the debug episode,
  empty-container evidence reduced inside belief from `0.65` to `0.1210` and
  avoided one additional viewpoint observation. Without negative evidence,
  the location belief remained unchanged until close-high outside evidence
  arrived.
- Decision: do not describe this result as physical cover opening or
  continuous robot MPC. `remove_cover` is an abstract transition and
  observation action. No UR10e trajectory, RG6 contact, collision physics, or
  learned perception ran.
- Decision: do not freeze the hand-authored `0.97` interaction success model,
  observation likelihoods, `0.85` grasp gate, or task costs. They exist to
  validate the feedback and negative-evidence implementation and must later
  be estimated or validated from episode-disjoint calibration data.
- Decision: the next physical integration must replace scripted symbols with
  post-action learned/RGB-D evidence and connect only the selected first
  `remove_cover` request to a validated manipulation primitive. Preserve the
  same high-level planner interface so physical execution does not leak
  future observations into planning.

## 2026-07-29 — Preserve joint belief and bind every MPC action to a graph revision

- Decision: store the exact target-location by cover-state joint distribution
  as an optional Scene Graph belief field for the covered-container planner.
  Do not reconstruct it from independent marginals because that discards
  action-relevant correlation.
- Decision: bind every selected action request to the canonical SHA-256 hash
  of its source Scene Graph. An executor result must carry the same request
  identifier and graph hash before it may update the graph.
- Decision: make first-action execution explicit. The executor may return a
  new observation only after the action request is fixed; future scripted
  observations and post-hoc true state remain unavailable to control.
- Decision: describe the current CPU executor only as a contract stub. Its
  successful graph-update and replanning traces do not constitute RGB-D
  perception, cover manipulation, continuous motion MPC, or physical
  end-to-end execution.
- Decision: preserve this interface when GPU access returns. Replace only the
  executor/evidence implementation with live Isaac Sim perception and a
  collision-checked UR10e/RG6 primitive, rather than creating a separate
  planner path that could bypass the Scene Graph revision checks.

## 2026-07-29 — Separate static calibration covers from manipulable covers

- Decision: do not attempt RG6 removal of the existing flat static
  calibration cover. It has neither a grasp handle nor a dynamic rigid body
  and therefore cannot provide valid interaction evidence.
- Decision: preserve the static cover for historical `covered_unknown`
  calibration scenes so their geometry is not silently changed. Author a
  separate dynamic plate-and-handle assembly only for the
  `cover_removal_required` action-development class.
- Decision: use a hierarchical control boundary. The high-level belief MPC
  selects `remove_cover`; a separate low-level UR10e/RG6 primitive compiles
  pregrasp, descent, contact closure, lift, transfer, placement, release, and
  retreat. Do not call the deterministic low-level primitive continuous
  belief-space MPC.
- Decision: accept a remove-cover request only when its action, target,
  episode, and canonical source Scene Graph SHA-256 all match. Do not allow a
  general motion executor to bypass that binding.
- Decision: require bilateral handle contact, force and penetration limits,
  finite joints, bounded tracking error, continuous collision checks,
  retained contact/slip limits, stable staging, and a fresh post-action RGB-D
  observation. Any monitored manipulation failure becomes `action_failed`;
  the executor must not guess `target_detected` or `empty_container`.
- Decision: treat the CPU Cartesian plan only as readiness for live IK and
  physics validation. Its basket-frame AABB clearances do not establish
  reachability, swept-volume safety, contact stability, or physical success.
- Decision: after compliant GPU isolation returns, validate only seed `185`
  first. Inspect the new cover support and handle, solve home/pregrasp IK,
  enable whole-arm/RG6/basket collisions, and stop on the first failed gate
  before attempting lift or launching a multi-seed batch.

## 2026-07-30 — Do not rerun complete caches; separate Qwen identity from relation evidence

- Decision: inspect exact-input caches before every pretrained perception
  run. The seed `165–184` collection already contains complete
  GroundingDINO, SAM2, and Qwen outputs for all `60` observations, so using
  GPU 1 or GPU 2 to recompute them would violate the cache-first pilot
  protocol without adding scientific evidence.
- Decision: retain episode-disjoint cross-validation as a calibration
  development diagnostic. A temperature for a held-out episode must be fit
  without any record from that episode.
- Decision: retain Qwen target-identity temperature scaling as a candidate
  because held-out calibration-fold NLL, Brier score, and ECE all improved.
  Do not freeze or deploy the temperature until the calibration protocol,
  action model, task-risk costs, and untouched final test split are complete.
- Decision: do not use temperature scaling to claim that Qwen relation
  failures are solved. It left membership and behind `unknown` recall at zero,
  worsened held-out behind NLL, and left objective `occluded_by=yes` recall at
  zero. Confidence softening cannot create class discrimination absent from
  the raw logits.
- Decision: keep Qwen as semantic target-identity evidence and use calibrated
  RGB-D geometry or a validated hybrid observation model for action-relevant
  relation beliefs. This preserves the declared calibrated relation-edge
  requirement while avoiding unsupported Qwen-only relation probabilities.

## 2026-07-30 — Retain membership and objective occlusion likelihood candidates

- Decision: retain the episode-disjoint categorical likelihoods for hybrid
  RGB-D `membership` and objective target `occluded_by` as provisional Scene
  Graph observation-model candidates. They generalized across the four
  calibration folds and were materially better than a uniform no-evidence
  posterior on held-out NLL and Brier score.
- Decision: do not manufacture a `0.99` probability for a deterministic hard
  relation label. Report hard-rule accuracy, abstention coverage, and
  selective accuracy separately from probability metrics. Evaluate fitted
  likelihoods against a stated probabilistic baseline.
- Decision: exclude `behind` from deployment until objective
  camera-conditioned geometry labels and enough `unknown` examples exist.
  Legacy authored view intent is inadequate calibration truth.
- Decision: preserve `unknown` as an observation symbol rather than silently
  converting an abstention into a hard state. The fitted observation
  likelihood may still update the current Scene Graph prior through Bayes'
  rule.
- Decision: do not apply these candidates to final MPC or report test
  performance yet. First freeze the calibration protocol, validate
  action-conditioned post-action observations and task-risk gates, then use
  untouched reserved test episodes exactly once.

## 2026-07-30 — Do not claim action-conditioned MPC benefit from the current cache

- Decision: use fold-specific Qwen temperatures and refit hybrid relation
  likelihoods only on the outer-training seeds for every nested calibration
  replay. Do not return to a single temperature fitted on all 20 calibration
  seeds when reporting held-out development performance.
- Decision: retain the current action-conditioned MPC code as an integrated
  implementation, but do not claim that it outperforms action-agnostic
  planning. Under leak-controlled replay it achieved a `0.85` safe-outcome
  rate versus `1.00` for action-agnostic belief MPC.
- Decision: treat the result as a scene-design failure, not evidence against
  the research hypothesis. Close-high and right changed the post-hoc latent
  state in only `1/20` episodes, so the cache cannot identify useful
  action-conditioned observation differences.
- Decision: when rendering becomes available, create balanced development
  scenes where close-high and right have deliberately different
  visibility/reference-occlusion outcomes. Verify those differences from
  rendered objective masks before model inference or MPC evaluation.
- Decision: do not spend GPU 1 or GPU 2 recomputing the unchanged 60 cached
  observations. Their GroundingDINO, SAM2, and Qwen outputs are already
  complete; new GPU inference becomes useful only after genuinely new
  action-differentiating RGB-D observations exist.

## 2026-07-30 — Require device-level isolation before another Vulkan launch

- Decision: treat any project PID on physical GPU 0, including a `4 MiB`
  graphics-only Vulkan context, as an immediate-stop violation.
- Decision: retain the forbidden-GPU watchdog for every future server
  graphics smoke. Monitoring failure is also a fail-closed termination.
- Decision: do not interpret `CUDA_VISIBLE_DEVICES`, `NVIDIA_VISIBLE_DEVICES`
  on bare metal, or `SimulationApp(active_gpu=1)` as complete Vulkan device
  isolation. The seed-185 smoke empirically opened GPU 0 despite those
  settings.
- Decision: do not launch GPU-2 perception after a failed scene capture.
  There is no valid new RGB-D observation to process.
- Decision: resume Isaac only in a container/device-cgroup configuration that
  exposes the intended physical GPU and Vulkan graphics capability while
  hiding GPU 0, or after an administrator verifies an equivalent host-level
  Vulkan/Xorg configuration.

## 2026-07-30 — Keep the full comparison harness but close the superiority gate

- Decision: retain immediate grasp, fixed close-high, fixed right,
  confidence-only, action-agnostic belief MPC, and action-conditioned belief
  MPC as the minimum final comparison set. Apply the same episode-disjoint
  calibration rules to every learned/calibrated method.
- Decision: retain Qwen target temperature calibration as a supported
  development component. Removing it reduced the safe-outcome rate from
  `0.85` to `0.70` under the same action-conditioned planner.
- Decision: retain task-risk gating as structurally necessary. Removing its
  costs and commitment constraints caused five wrong grasps in five
  `covered_unknown` episodes.
- Decision: do not claim that the current hybrid relation evidence improves
  task performance. Its removal slightly improved the cached replay, so its
  standalone calibration success must be reconciled with downstream planning
  on genuinely action-differentiating scenes.
- Decision: do not claim action-conditioned MPC superiority. The
  action-agnostic model achieved a `1.00` safe-outcome rate versus `0.85` for
  the proposed model on this development cache.
- Decision: do not create a negative-evidence ablation result from episodes
  that contain no cover-removal or empty-container observation. Run it only
  after valid covered-container interaction episodes exist.
- Decision: reuse this harness for the later frozen comparison, but do not
  open the reserved test split until scene design, perception calibration,
  action-conditioned observation coverage, and risk thresholds are frozen.

## 2026-07-30 — Retain membership and block occlusion from downstream MPC

- Decision: retain calibrated hybrid RGB-D membership as the relation
  candidate for the next development pilot. Target plus membership achieved
  a `1.00` safe-outcome rate with no missed visible targets in the current
  episode-disjoint calibration replay.
- Decision: do not feed the current `occluded_by` likelihood into a deployed
  MPC commitment gate. Occlusion-only reduced the safe-outcome rate to
  `0.85`, and the full policy was identical to occlusion-only on every
  episode.
- Decision: preserve occlusion outputs and objective labels for diagnostics;
  do not delete the factor. Revise its action-conditioned transition model,
  probability semantics, and gate jointly on new balanced view-action
  scenes.
- Decision: do not lower the `0.25` hidden or `0.60` reference-occlusion gate
  post hoc to rescue seeds 165, 167, or 178. Thresholds must be selected
  inside calibration folds and frozen before reserved testing.
- Decision: distinguish standalone perception calibration from downstream
  decision utility. A relation classifier can have good held-out NLL and
  recall yet still harm planning when its state definition, transition model,
  or task gate is mismatched.

## 2026-08-01 — Keep direct multi-view Qwen as a post-action identity ablation only

- Decision: retain the new direct multi-view Qwen path as an optional
  post-action target-identity ablation. On a deliberately failure-enriched
  calibration subset it improved target selection from `9/14` to `13/14`
  without a target-selection regression, so it is useful enough to test later
  under a frozen, balanced protocol.
- Decision: do not substitute this diagnostic for the probabilistic Scene
  Graph or Bayesian belief update. It consumes only observations already
  available after an executed action and cannot predict which unseen view an
  action will produce. It therefore does not by itself provide
  action-conditioned future belief or belief-space MPC.
- Decision: do not use the multi-view result to re-enable `occluded_by` in the
  planner. Current-view occlusion accuracy decreased from `11/14` to `10/14`,
  confirming that historical context can help identity while confusing or
  failing to repair view-specific relation evidence.
- Decision: keep candidate-proposal missingness explicit. Two covered pairs
  had no target proposal and are excluded from ranking accuracy; multi-view
  Qwen must not invent a target candidate that GroundingDINO/SAM did not
  propose.
- Decision: do not tune prompts or thresholds against the eight selected
  seeds and do not open reserved seeds. Revisit direct multi-view identity
  only after balanced action-differentiating scenes exist, then compare it
  with independent-view calibrated fusion using identical episode-disjoint
  folds.
- Decision: for CUDA-only server experiments, expose physical GPU 5 only and
  monitor physical GPUs `0–4` as forbidden. This does not relax the separate
  requirement for device-level Vulkan isolation before another Isaac launch.

## 2026-08-01 — Reject the first right-only render and tighten scene semantics

- Decision: retain seed-185 run 003 as a rendered geometry diagnostic, but do
  not yet mark it eligible for calibration. Passing the current dominance gate
  does not establish literal close-high-only resolution because the right view
  also exceeded the provisional `0.65` visibility threshold.
- Decision: reject seed-186 run 001. Although capture and robot motion
  succeeded, close-high visibility was `0.9902` versus right `0.7186`, the
  opposite of the authored `right_only` action outcome.
- Decision: stop before seeds 187–196. Do not collect VLM outputs or fit an
  action-conditioned model from scene intent labels when rendered objective
  masks contradict those labels.
- Decision: revise the action occluder using physically supported geometry and
  3D camera-ray reasoning, then rerender seeds 185 and 186. Do not float an
  occluder or allow target/occluder intersection merely to satisfy an image
  metric.
- Decision: before resuming the development batch, add an exclusivity check
  for one-view variants so the declared non-resolving view remains below the
  resolution threshold in addition to losing by the dominance margin. This is
  a development acceptance-gate correction and must be validated by focused
  CPU tests before another GPU launch.

## 2026-08-01 — Accept the revised two-scene geometry pilot

- Decision: accept seed-185 run 005 as the first validated
  `close_high_only` development scene and seed-186 run 002 as the first
  validated `right_only` development scene. Both satisfy resolution, gain,
  dominance, and non-winning-view exclusivity gates after rendered objective
  mask measurement.
- Decision: retain seed-185 run 004 as a failed iteration rather than deleting
  it. Robot execution succeeded, but the scene label did not: the original
  `0.18 m` upright occluder also suppressed the intended resolving view.
- Decision: use only physically supported action-conditioning geometry. An
  upright occluder must stand on the basket interior; a steep overhead-like
  view may instead be blocked by a narrow partial-cover bar resting on both
  rims. Do not satisfy image gates using floating or intersecting geometry.
- Decision: do not treat these two passes as final evidence for the proposed
  MPC. They validate causal scene construction and reachable re-observation
  motion only. Qwen/perception inference, calibrated belief updates,
  action-conditioned prediction, online MPC selection, and grasp remain to be
  connected on these scenes before expanding the development batch.

## 2026-08-01 — Do not tune away the first causal-scene MPC failure

- Decision: reject the frozen planner's `defer` actions on seeds 185 and 186
  as task completions. The renderer provided a useful view and learned
  perception recovered the target in each intended resolving view, but the
  planner valued noncompletion below every view action.
- Decision: do not lower view costs, raise the defer cost, or loosen commitment
  gates using these same two outcomes. Such a post-hoc edit would convert a
  diagnostic failure into an invalid success.
- Decision: retain the complete cached perception outputs. They show that the
  intended views provide useful target evidence, while blocked views can
  produce confident-looking but semantically wrong or non-observable relation
  claims.
- Decision: the next calibration collection must contain balanced, physically
  action-differentiating scenes and must fit a future-observation model that
  conditions on current scene/occluder evidence as well as the candidate
  action. Select task costs and gates inside episode-disjoint calibration,
  then evaluate on untouched seeds.

## 2026-08-01 — Treat the 11/11 planner result as calibration only

- Decision: accept nine of seeds 187–196 for development calibration and
  reject seed 191. Do not relax the center-view unresolved threshold or
  relabel seed 191 after observing its rendered visibility.
- Decision: retain the scene-conditioned leave-one-episode-out result
  (`11/11`) as evidence that current-scene cues can repair the old planner's
  `defer` failure. Do not report it as final MPC accuracy, statistical
  evidence, or an unbiased test result.
- Decision: keep training, calibration, and testing separate. No foundation
  model training was performed; the 11 episodes were used for calibration;
  reserved final-test seeds remain unopened.
- Decision: before paper-scale evaluation, execute `remove_cover`, capture the
  post-removal observation, apply negative-evidence belief update, and validate
  the resulting replanning path. Then freeze features, costs, and gates before
  running any reserved test episode.
- Decision: do not add an external dataset or fine-tune Qwen merely because a
  calibrated planner is being built. Revisit model training only if frozen
  pretrained perception fails a predeclared requirement on a separate
  calibration set.

## 2026-08-03 — Accept physical remove-cover replanning as the final integration gate

- Decision: accept seed-188 run 010 as evidence that actual UR10e+RG6 cover
  interaction, same-process post-action RGB-D, Scene Graph belief update, and
  receding-horizon replanning are causally connectable. The post-action image
  did not exist when the root `remove_cover` action was selected.
- Decision: do not call this a complete end-to-end manipulation success. The
  replanned `grasp_inside` request was recorded but not executed because the
  gripper still held the transferred cover. Add a contact-safe cover release
  and retreat before executing the target grasp in the same episode.
- Decision: label simulator-instance-mask evidence as an automatic pilot
  oracle. It may validate the interaction and belief-update contract, but it
  cannot replace GroundingDINO/SAM2/Qwen in the proposed learned-perception
  method or support a final paper claim.
- Decision: the next covered-container development runs must include both a
  target-present positive branch and an empty-container negative-evidence
  branch. Freeze perception calibration, observation likelihoods, costs, and
  risk gates only after both branches and the final physical grasp pass.
- Decision: do not open reserved final-test seeds yet. The remaining gate is
  one same-episode sequence with learned post-action perception and physical
  second-action execution, followed by a separate negative-evidence sequence.

## 2026-08-04 — Reject the current lid lift as transfer-ready physics

- Decision: preserve seed-188 run 010 as causal software-integration evidence
  only. Its old contact gate is insufficient for a realistic lid-lift claim,
  so it must not be cited as successful contact physics or final evaluation.
- Decision: accept runs 016, 017, and 019 as informative development failures.
  All reached bilateral contact and sufficient transient force, but the lid
  slipped about `3.0–3.1 cm` and exceeded the predeclared contact-gap limit
  during lift.
- Decision: keep the strict `15 mm` relative-translation, three-step contact-
  gap, rotation/angular-speed, force, penetration, and post-lift environment-
  collision gates. Do not loosen these gates after observing the failures.
- Decision: do not create a pass by lowering the `0.55 kg` lid mass, attaching
  or copying the lid pose, or assigning an arbitrarily extreme friction
  coefficient. The next accepted run requires a continuous force-maintaining
  RG6 control proxy and fingertip/contact parameters that can be mapped to the
  real lab gripper.
- Decision: the current EPDM-like friction and torque settings are provisional
  development values. Freeze final values only after receiving the lab RG6
  force setting, fingertip type/geometry, lid mass/material, and a simple real
  grip/lift calibration measurement.

## 2026-08-04 — Keep the micro-lift gate after continuous-control run 020

- Decision: reject seed-188 run 020 as a physical cover-removal success. It
  exceeded the `5 mm` lid-to-gripper relative-translation limit during the
  mandatory `1 cm` micro-lift, even though the controller increased its drive
  torque to the provisional `6 N m` ceiling.
- Decision: retain the continuous controller and micro-lift as development
  infrastructure, but do not interpret the controller's drive torque as a
  calibrated real RG6 grip-force command. Collision overlap without meaningful
  instantaneous normal force is not accepted as maintained contact.
- Decision: do not raise torque/force ceilings, reduce lid mass, inflate
  friction, or relax the slip threshold using run 020 as feedback. Obtain the
  real fingertip, handle, commanded grip-force, and lift calibration before
  freezing transfer-ready physics parameters.

## 2026-08-04 — Gate transfer-ready execution on a lab calibration record

- Decision: keep provisional development physics available for debugging, but
  never label it transfer-ready. A transfer-ready run must supply a completed
  `rg6-lid-transfer-calibration-v1` record that passes the validator before
  Isaac Sim initializes.
- Decision: require real RG6/fingertip identification, lid mass and geometry,
  commanded grip force, an explicitly lab-fitted simulation mapping, and at
  least five retained 1 cm micro-lift trials with at least an 80% pass rate.
  This is an engineering calibration criterion, not a statistical paper claim.
- Decision: preserve the existing slip, force, penetration, bilateral-contact,
  and collision limits. The calibration file may provide measured parameters
  but may not weaken those acceptance limits.
- Decision: distinguish hardware calibration from training and final testing.
  No model training is performed here, and passing the calibration gate does
  not make an episode valid final-evaluation evidence.

## 2026-08-04 — Permit an explicit provisional proxy, never a fabricated lab record

- Decision: while lab information is unavailable, use a separate
  `provisional_public_spec` config combining cited manufacturer facts with
  clearly labeled simulation assumptions. Do not write assumed values into the
  `lab_measured` worksheet.
- Decision: provisional physics must require an explicit command-line opt-in
  and must remain `transfer_ready=false`. A later real-robot handoff replaces
  the config and reruns the micro-lift gate; changing labels alone is invalid.
- Decision: preserve manufacturer revision discrepancies, including the RG6
  150 mm product-page versus 160 mm datasheet stroke, until the physical lab
  unit is identified. Do not use public model-level specifications as proof of
  the lab unit's fingertip, controller, payload, or calibration state.

## 2026-08-04 — Stop parameter-only tuning after run 028

- Decision: accept the dense branch-limited UR10e descent from run024 onward
  as the corrected development path. Do not restore the sparse waypoint path,
  the redundant wrist revolution, or the discontinuous elbow solution.
- Decision: retain the `5 mm` relative-slip, `60 N` per-finger force, `3 mm`
  penetration, bilateral-contact, and unexpected-collision gates. Runs026 and
  028 passed force/penetration checks but failed the micro-lift slip gate, so
  neither is a physical cover-removal success.
- Decision: do not continue fitting closure increments, friction, mass, or
  acceptance thresholds to seed 188. The next development unit is an isolated
  RG6 fingertip/handle contact-geometry fixture that measures pad approach,
  contact normals, force balance, and micro-lift slip before another full
  covered-container episode.
- Decision: the `354 x 330 mm` removable cover and current compliant-contact
  values are explicit development proxies only. Replace them with measured lab
  cover, handle, fingertip, and commanded-force parameters before any
  transfer-ready or final-evaluation run.

## 2026-08-05 — Reject contact-pair presence as an RG6 grasp

- Decision: reject isolated fixture run005 as a grasp success. Both imported
  fingertip collision pairs reached the procedural handle, but both peak
  normal-force and penetration measurements remained zero; therefore no
  micro-lift was authorized.
- Decision: accept the exact-FK 1 cm lift IK and the passive world-Z guide only
  as diagnostic infrastructure. The guide may isolate contact geometry but
  must not appear in a final covered-container evaluation scene.
- Decision: do not convert collision-pair presence, a visually enclosed
  handle, or a moving RG6 master joint into a successful-contact label. An
  accepted grasp still requires bilateral nonzero force, the fixed force and
  penetration limits, a measured micro-lift of at least `7 mm`, no more than
  `5 mm` relative slip, and no unexpected collision.
- Decision: stop full covered-episode retries until the imported jaw-width
  curve and collision contact envelope are checked against the lab fingertip
  geometry and commanded RG6 opening/force. Do not compensate for missing
  compression by lowering the lid mass, inflating friction, or weakening the
  gates.

## 2026-08-05 — Freeze geometry tuning after the loaded-mimic diagnosis

- Decision: accept jaw-width run001 as a simulation development calibration
  of the imported collision geometry. Its `0.435487 rad` contact threshold may
  prevent speculative-pair false positives in provisional simulation, but it
  is not a real RG6 command calibration and remains `transfer_ready=false`.
- Decision: reject fixture runs006 and 007 as bilateral grasps. A measured
  `0.615 mm` centering correction did not change the one-sided loaded result,
  so do not continue shifting the handle or fitting friction to force a pass.
- Decision: the next accepted physics change must address the loaded RG6
  coupling itself and be traceable to the real lab fingertip and command/force
  measurements. Validate it first in the unchanged 1 cm fixture, then in the
  full covered-container episode.
- Decision: retain all fixed gates and keep the full paper test set closed.
  Jaw geometry calibration, controller repair, and fixture validation are
  calibration/development, not testing and not final-paper performance.

## 2026-08-05 — Accept coordinated drives only as a provisional coupling fix

- Decision: accept isolated fixture run011 as evidence that the actual
  imported RG6 collision geometry can maintain bilateral, force-bearing
  contact through the fixed micro-lift gate when all six joints are explicitly
  coordinated. It replaces the failed passive loaded-mimic behavior for the
  next simulator development run.
- Decision: do not describe the `18 N m` aggregate joint-drive effort as real
  RG6 motor torque or commanded grip force. The six independent simulator
  drives are an effective development proxy and remain
  `transfer_ready=false` until fitted to measurements from the lab unit.
- Decision: retain run008, run009, and run010 as failures. Run008 lacked the
  force floor, run009 exposed a force-gate timing bug, and run010 missed the
  minimum measured lift. Run011 passes because it meets the pre-existing
  `7 mm` lift, `5 mm` relative-motion, bilateral-contact, `60 N` force,
  `3 mm` penetration, and unexpected-collision gates—not because thresholds
  were weakened.
- Decision: do not use the fixture's passive world-Z guide in the full scene.
  The next accepted result must reproduce contact-safe cover lift in the
  covered basket, then release/retreat, capture a genuinely new observation,
  update the Scene Graph belief, and execute the replanned action.
- Decision: this fixture result is physics development evidence only. It does
  not validate Qwen, belief-space MPC, sim-to-real transfer, or final paper
  performance, and reserved final-test seeds remain unopened.

## 2026-08-05 — Require learned post-remove localization for the second grasp

- Decision: the next complete seed-188 integration must not use the simulator
  instance mask to choose or localize the post-remove target. It must run the
  frozen GroundingDINO-Base, SAM2.1-Large, and Qwen3-VL-8B-Instruct path on
  the newly acquired RGB-D observation and pass the selected anonymous mask
  through depth backprojection.
- Decision: physically execute the replanned `grasp_inside` only when the
  localization file is inside the active live-session directory and reports
  no simulator-ground-truth leakage. Acceptance additionally requires the
  existing bilateral-contact, lift, collision, force, and penetration gates.
- Decision: simulator masks may be read after inference only for diagnostic
  localization error and task scoring; they may not affect candidate
  selection, localization, belief update, or action choice in this learned
  integration run.
- Decision: do not launch bare-metal Isaac while another researcher occupies
  GPU 0. Selecting a different renderer GPU does not prevent the known small
  Vulkan context on GPU 0, so the full execution waits for GPU 0 availability
  or verified device-level isolation. Perception-only work may continue on a
  separately authorized GPU under the forbidden-GPU watchdog.

## 2026-08-06 — Accept run058 as integration evidence, not final evaluation

- Decision: accept seed-188 run058 as the first successful same-process
  covered-container integration of RGB-D, learned grounding/segmentation,
  Qwen, probabilistic belief/Scene Graph update, belief-tree MPC replanning,
  dynamic IK, physical cover manipulation, and physical target lift.
- Decision: use coordinated follower drives only for the provisional cover
  manipulation, then restore the asset-authored Newton mimic linkage for the
  target grasp. The six-drive target proxy is rejected because it caused
  centimeter-scale slip or contact loss; the passive linkage with a
  provisional `1.2 N m` master ceiling passed the unchanged force,
  penetration, slip, rotation, contact-continuity, and collision gates.
- Decision: a Newton mimic restoration must rebuild PhysX. Before rebuilding,
  capture the released cover pose; after `stop()`, author that pose and reject
  the episode if `play()` changes it by more than `2 mm` or `2 deg`. Never
  accept a post-remove episode in which the cover silently resets.
- Decision: retain the `60 N` per-finger force ceiling, `3 mm` penetration
  ceiling, `15 mm/10 deg` target-relative stability limits, bilateral-contact
  continuity, micro-lift, arm-tracking, and unexpected-collision gates. No
  threshold was weakened to obtain run058.
- Decision: do not report run058 as final ICRA performance or a statistical
  guarantee. It is one deterministic development seed; Qwen scores are still
  uncalibrated and all RG6/lid effort, friction, mass, and geometry values
  remain provisional and non-transfer-ready.
- Decision: the next paper-facing work is a frozen calibration/test split,
  randomized held-out multi-seed evaluation, required baselines and ablations,
  and replacement of provisional hardware parameters with measured lab values
  before real-robot validation.

## 2026-08-07 — Require relation-aware negative evidence before final-test freeze

- Decision: accept seed-197 run002 as development integration evidence that
  one persistent Isaac process can execute physical cover removal, consume an
  empty-inspection result, update belief, physically move to the selected
  right viewpoint, run learned perception and Qwen, update belief again, and
  physically grasp the outside target.
- Decision: reject any rule that maps target pixels anywhere in an image to
  `target_detected` inside the inspected container. Negative evidence is a
  target-to-container relation judgment, not a global detection judgment.
- Decision: simulator generator 3D membership may provide the
  `empty_container` symbol only for this development smoke. Before reserved
  test seeds are opened, replace it with a frozen learned/RGB-D relation
  estimator and evaluate its calibration on episode-disjoint data.
- Decision: retain the `0.65` rendered resolution threshold. Scene runs001,
  run002, and run003 remain failures; run004 is accepted because geometry was
  changed before the held-out final test, not because the gate was relaxed.
- Decision: do not call run002 final ICRA evidence. Its action-conditioned
  observation likelihoods and commitment threshold are development values,
  and its RG6/cover physics remain provisional and non-transfer-ready.

## 2026-08-07 — Gate empty-container updates on learned semantic/geometric agreement

- Decision: for live development, map an inspected container to
  `empty_container` only when Qwen's relation choice and the learned-mask
  RGB-D footprint classifier both return `outside`.
- Decision: if either source is missing, returns `unknown`, or disagrees, fail
  closed; do not use simulator membership as a fallback.
- Decision: treat this agreement rule as a development safety gate, not
  calibrated fusion and not a final-paper method. Freeze its thresholds and
  action-conditioned likelihoods on calibration episodes before reserved
  testing.
- Decision: accept seed-197 run003 as non-oracle integration feasibility, but
  do not retroactively convert earlier oracle-assisted runs into final
  evidence.

## 2026-08-07 — Keep reserved tests closed behind a predeclared protocol

- Decision: do not include spatial-relation answers such as `inside` in the
  VLM instruction. Use identity attributes only and query membership and
  occlusion as separate factorized judgments.
- Decision: reserve seeds 200--209 for both mandatory simulation scenario
  families. Calibration/development uses seeds 165--199; no parameter, prompt,
  threshold, or scene geometry may be tuned from reserved outcomes.
- Decision: evaluate seven required methods and six essential ablations on
  ten seeds in two scenario families, for 260 predeclared method--scenario
  evaluations, plus a smaller predeclared high-fidelity contact-physics
  subset.
- Decision: the protocol file does not itself authorize testing. A fail-closed
  preflight must verify frozen perception calibration, action-conditioned
  likelihoods, task costs, commitment gates, prompt/model hashes, and a
  frozen-parameter artifact before any reserved seed is opened.

## 2026-08-07 — Freeze only calibrated components and count independent cover episodes

- Decision: freeze the identity-only instruction, Qwen checkpoint revision,
  prompt metadata, and target temperature `5.825`; they are backed by the
  20-seed calibration collection and immutable source hashes.
- Decision: do not freeze relation likelihoods merely to start final testing.
  Current cross-validation has zero recall for required unknown/yes subclasses,
  and the old sparse action-conditioned replay lets the proposed method
  underperform the action-agnostic baseline. Relation likelihoods, future
  observation models, task costs, and the commitment gate remain mutable on
  calibration data only.
- Decision: count at most one successful calibration episode for each
  `(outcome family, seed)` pair. Repeated engineering runs of seed 188 or 197
  are retained for debugging but cannot inflate the paper sample count.
- Decision: an episode enters the cover calibration count only if both cover
  removal and final target grasp pass the predeclared `0.15 m` lift, bilateral
  contact, `60 N` force, `3 mm` penetration, `15 mm/10 deg` relative stability,
  finite-joint, no-unexpected-collision, no-attachment, and no-pose-copy gates.
- Decision: accept seed 198 as the second independent negative-evidence
  calibration success, not as a reserved test or final ICRA result.
## 2026-08-07 — Use objective camera-relative geometry for behind evaluation

- Decision: final calibration and testing must not score `behind` from the
  deterministic scene generator's view-intent label. Use simulator world
  bounds, the current camera ray, and rendered projected overlap to generate
  an independent `yes/no/unknown` label after observation capture.
- Decision: the simulator-only measurement is evaluation/calibration metadata
  and must never be exposed to GroundingDINO, SAM2, Qwen, belief update, MPC,
  or action selection.
- Reason: seed 199 demonstrated a concrete disagreement: the legacy center
  intent was `unknown`, while measured geometry placed the target beyond the
  basket far edge with substantial projected overlap (`behind=yes`). Counting
  the legacy label would incorrectly penalize a correct model judgment.
- Boundary: seed 199 is calibration-only evidence. It does not freeze the
  relation likelihood, action-conditioned observation model, task costs, or
  commitment gate, and it is not a final-paper result.

## 2026-08-07 — Freeze the objective hybrid relation observation model

- Decision: freeze relation likelihoods only after adding objective
  camera-relative `behind=unknown` support. Use the 25 calibration seeds
  `165--189`; keep reserved test seeds `200--209` unopened.
- Decision: Qwen remains the instruction-conditioned target-identity source.
  Membership, behind, and reference occlusion use learned masks, metric RGB-D,
  and calibrated categorical observation likelihoods. Qwen relation choices
  remain diagnostics and may not override objective geometry evidence.
- Decision: use the measured basket XY footprint for relation geometry, a
  5--95 percentile candidate-center estimator, and a `0.03 m` behind
  abstention band. These values were selected using calibration data only and
  are now immutable before final testing. For real transfer, the lab basket
  dimensions must replace the simulation values without using final-test
  outcomes for tuning.
- Evidence: five-fold episode-disjoint calibration accepted membership,
  behind, and occluded-by likelihoods. Behind recovered all objective
  yes/no/unknown records (`20/33/10`); membership reached `127/128`; calibrated
  reference occlusion reached `56/63` with `23/28` positive recall.
- Decision: clearing the relation freeze does not authorize final testing.
  Keep the fail-closed preflight blocked until the action-conditioned
  observation model, task costs, and grasp commitment gate are separately
  calibrated and frozen. Training remains unperformed.

## 2026-08-07 — Do not freeze a one-example positive cover observation model

- Decision: retain the `11/11` episode-disjoint view-action result and the
  `10/10` right-after-empty relation result as calibration evidence only.
- Decision: apply the existing minimum-five-per-declared-cell support rule to
  the physical cover-removal observation model. Ten outside/empty outcomes
  pass support, but one inside/target-detected outcome does not; collect four
  more independent inside-cover calibration episodes before freezing it.
- Decision: do not let a Dirichlet-smoothed `11/11` leave-one-episode-out
  diagnostic override missing positive-class support. The full
  action-conditioned model remains unapplied to MPC, and the final-test
  preflight remains closed.
- Decision: keep seeds `200--209` untouched. Additional support must come from
  calibration seeds or new non-reserved calibration scenes, not from final
  test outcomes. No foundation-model training or fine-tuning is introduced.

## 2026-08-07 — Separate the development repository from the lab handoff

- Decision: update the private development repository first. Prepare a
  separate lab-facing repository after the implementation and hardware
  interface are reviewed.
- Decision: the lab handoff must include model revisions, environments,
  schemas, action contracts, commands, hardware measurements, and safety
  assumptions. It must not include model weights, server caches, raw experiment
  outputs, credentials, or machine-specific absolute paths.
- Decision: do not require ROS 2 inside the research method. Expose a stable
  RGB-D/state input and semantic-action/result boundary so the lab can provide
  a ROS 2, RTDE, or other hardware adapter.
