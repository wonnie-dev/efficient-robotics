# Benchmark Environment

## Scenario families

The simulator contains five task families for language-guided retrieval under
partial observability:

1. **Visible/Open:** the target is visible inside or outside the container.
2. **Partially Occluded:** a reachable wrist-camera view resolves target identity.
3. **Covered Container:** cover removal changes the observable scene before replanning.
4. **Ambiguous Inside/Outside:** similar candidates require joint identity and membership reasoning.
5. **Target Absent:** negative evidence should produce a safe deferral rather than a distractor grasp.

## Robot and scene

The environment contains a UR10e, OnRobot RG6, wrist-mounted RGB-D camera,
table, basket or container, target mug, visually similar distractors, and
controlled occluders. The current scanned-basket asset is imported from the
locally installed LIBERO assets; its source and license are recorded in run
metadata.

## Observations

The planner can select reachable semantic wrist views:

- `center`: initial observation;
- `close_high`: closer view of the container interior;
- `right`: lateral view for rim and object occlusion.

An external overview camera is used only for diagnostics and videos. It is not
available to the policy.

Each observation records RGB, metric depth, camera intrinsics and pose,
anonymous instance records, and timestamps. Simulator labels are written to a
separate evaluation record and are not provided to the VLM or planner.

## Scene acceptance checks

Seeded scenes are accepted only when they satisfy the configured checks for:

- stable support on the table or inside the container;
- no initial object-container interpenetration;
- collision clearance for the robot, camera, cover, and gripper;
- the intended visibility and occlusion pattern;
- consistent world and camera-relative relation labels.

Rejected scenes retain a failure reason and are not counted as evaluation
episodes.

## Physics

Cover removal and target retrieval use articulated UR10e motion and bilateral
RG6 contact. Evaluation does not attach objects to the gripper or copy object
poses. A valid physical action must pass lift, contact, force, penetration,
slip, joint-state, tracking, and unexpected-collision checks.

Mass, friction, fingertip, camera, and object geometry remain configurable so
they can be replaced with measured hardware values for real-robot transfer.
