# Decisions

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
