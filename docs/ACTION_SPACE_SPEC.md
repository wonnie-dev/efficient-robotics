# Action Space

## Scope

The planner selects high-level semantic actions. Deterministic robot code checks reachability, collision, contact, and hardware limits before execution. A VLM may activate an action already present in the library, but it cannot create a new controller or bypass a safety check.

## Common action record

```json
{
  "schema_version": "semantic-action-request-v1",
  "action_id": "viewpoint_right",
  "kind": "observation",
  "parameters": {
    "view_id": "right",
    "reference_id": "basket_01"
  },
  "preconditions": ["ik_feasible", "collision_free"],
  "expected_observations": ["rgb", "depth_m", "camera_pose"],
  "source_belief_id": "belief_000",
  "planner_step": 0
}
```

Execution returns a separate result record containing status, measured motion, observation paths, contact data, failure reason, and the Scene Graph update that may consume the result.

## Implemented semantic actions

| Action | Parameters | Preconditions | Expected result | Cost and risk | Failure result |
| --- | --- | --- | --- | --- | --- |
| `viewpoint_right` | view ID or wrist pose, reference object | reachable IK, joint limits, collision-free path | right-side RGB-D | motion time, collision risk | `ik_failed`, `path_blocked`, or `capture_failed` |
| `viewpoint_close_high` | view ID or wrist pose, reference object | reachable wrist pose, safe camera clearance | higher view of the container interior | motion time, camera clearance | `ik_failed`, `path_blocked`, or `capture_failed` |
| `remove_cover` | cover ID, handle pose, staging pose | cover and handle available, grasp feasible, placement clear | open container and new RGB-D | interaction time, slip, collision, force | `grasp_failed`, `slip`, `force_limit`, `placement_failed`, or `capture_failed` |
| `grasp_inside` | candidate ID, grasp pose | target belief above gate, container open, IK and descent safe | contact-based target lift | wrong commitment, collision, slip | `wrong_target`, `contact_failed`, `slip`, or `lift_failed` |
| `grasp_outside` | candidate ID, grasp pose | target belief above gate, IK and descent safe | contact-based target lift | wrong commitment, collision, slip | `wrong_target`, `contact_failed`, `slip`, or `lift_failed` |
| `defer` | reason | no action passes the commitment and safety gates | no physical motion | noncompletion cost | none |

`viewpoint_center` is the initial observation state, not normally a useful repeated action. `viewpoint_left`, `inspect_container`, and `move_occluder` are extension points and are not part of the frozen evaluation action set.

## Transition semantics

- Viewpoint actions do not change the world state under the nominal model. They change the observation distribution.
- `remove_cover` maps a covered state to an open state when execution succeeds. Failure leaves a covered or uncertain state and produces an explicit failure observation.
- Grasp actions are terminal commitments in the current task. A grasp at an empty or disproven location is a wrong commitment.
- `defer` terminates the current attempt without a physical action.

Every physical action is followed by a measured result. The planner does not assume success from the command alone.

## Observation contract

Viewpoint and interaction actions can return:

```text
target_detected
empty_container
inside_evidence
outside_evidence
unknown_evidence
action_failed
```

The observation model represents `P(observation | state, action)`. The posterior is computed after execution; future image files are never read during root-action selection.

## Safety contract

All physical actions require finite joint states, joint-limit checks, collision checks, and bounded tracking error. Contact actions additionally require bilateral finger contact, force and penetration limits, and an unexpected-contact check. Object attachment and pose copying are forbidden in physical evaluation.

Current simulation evaluation limits are:

- minimum verified lift: `0.15 m`;
- maximum force per finger: `60 N`;
- maximum contact penetration: `0.003 m`;
- maximum object-to-gripper translation: `0.015 m`;
- maximum object-to-gripper rotation: `10 deg`.

These are evaluation gates, not permission to use the same values on the real robot.

## Current and proposed representations

The current implementation uses a semantic viewpoint library backed by fixed poses. The next representation may generate poses from container geometry and filter them by IK and collision. Continuous camera-pose optimization is not implemented and is not claimed.
