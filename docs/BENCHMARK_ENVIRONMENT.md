# Benchmark Environment

## Scenario families

The simulator contains two task families for language-guided retrieval under
partial observability:

1. **Open-container active view:** similar target candidates are placed inside,
   outside, near, or behind an open basket. The initial view is intentionally
   ambiguous and a wrist-camera action can reveal additional evidence.
2. **Covered-container search:** a removable cover hides either the target or
   an empty container. Cover removal produces a new observation, belief update,
   and another planning step before retrieval.

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
