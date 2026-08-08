# Research Progress

Status date: 2026-08-07

## Implemented

- deterministic and randomized tabletop scenes with open, occluded, and covered containers;
- UR10e, RG6, wrist RGB-D capture, semantic instance records, and metric depth;
- GroundingDINO proposals, SAM2 masks, Qwen target ranking, and RGB-D localization;
- probabilistic Scene Graph nodes and relation edges;
- positive and negative evidence updates;
- discrete action-conditioned belief-tree planning;
- reachable wrist re-observation actions;
- contact-based cover removal and target grasp without attachment or pose copying;
- cached perception, episode-disjoint calibration, GPU watchdogs, and fail-closed final-test guards.

## Perception calibration

The target-identity temperature is frozen from 20 calibration seeds. Cross-validated target records reached `76/97` accuracy (`78.35%`), Brier score `0.2668`, ECE `0.0946`, and NLL `0.4023`.

Objective relation calibration uses learned masks and metric RGB-D rather than Qwen relation text:

| Relation component | Cross-validated result |
| --- | ---: |
| Membership | `127/128` (`99.22%`) |
| Camera-relative behind | `63/63`; support yes/no/unknown = `20/33/10` |
| Reference occlusion | `56/63` (`88.89%`) |

Qwen remains the instruction-conditioned target selector. The relation geometry uses simulator container dimensions and must be updated from lab measurements before transfer.

## Action-conditioned planning calibration

The first action model was built from scenes that did not create enough causal difference between views. On that collection, the action-conditioned policy did not outperform the action-agnostic baseline. That result is retained as a failed calibration design, not hidden or relabeled.

A replacement collection balanced close-high-only, right-only, either-view, and cover-required outcomes. Leave-one-episode-out calibration selected the designed resolving action in `11/11` episodes. The best fixed action, always selecting right, reached `5/11`. The selected view recovered the target in `8/8` view-action episodes; three episodes selected cover removal.

This is development calibration. It is not final MPC accuracy.

## Closed-loop physics

Seed 188 completed the following sequence in one Isaac Sim process:

```text
covered observation
  -> physical cover grasp and removal
  -> new RGB-D observation
  -> GroundingDINO + SAM2 + Qwen
  -> belief update and replanning
  -> learned-mask RGB-D localization
  -> dynamic grasp IK
  -> bilateral-contact RG6 lift
```

The cover was lifted `0.1599 m`. The target was lifted `0.1800 m`. Both actions passed the configured force, penetration, relative-stability, finite-joint, and unexpected-collision checks. Attachment and pose copying were not used.

The negative-evidence branch completed ten independent calibration episodes for seeds `190-199`. Each run removed the cover, observed an empty container, moved to the right view, updated the belief again, and grasped the outside target. Mean verified target lift was `0.1797 m`.

At the start of this update, the positive covered-target calibration had one accepted independent episode. A four-seed calibration batch for seeds `192`, `196`, `212`, and `216` was running. Final model status must be read from the generated calibration result after that batch finishes.

## Failures that changed the implementation

- Early videos moved an object with a kinematic mount or attachment. Those runs are debug artifacts and do not count as physical grasp evidence.
- Initial UR10e trajectories produced invalid joint states. Accepted runs now monitor finite joint state, tracking error, IK, and collision gates.
- Early cup and occluder placements intersected the basket. Scene generation now checks support and collision geometry before accepting a scene.
- Early lid motion slipped or behaved as an underconstrained body. The current contact controller and placement sequence pass simulation gates, but the physical parameters remain provisional.
- The old observation model underperformed an action-agnostic baseline. Final testing remains blocked rather than tuning on reserved seeds.

## Remaining work before final simulation results

1. Finish positive cover-observation calibration.
2. Freeze the action-conditioned observation model.
3. Select and freeze task-cost weights and the grasp commitment gate using calibration episodes only.
4. Pass the final-test preflight.
5. Run the predeclared methods and ablations on reserved seeds `200-209`.
6. Report paired statistics, failure categories, runtime, and physical-gate results.

The protocol defines 260 policy evaluations plus a smaller high-fidelity contact subset. None of the reserved test outcomes has been used for tuning.

## Real-robot status

Real-robot validation has not started. The lab has reported a UR10, RG6, and a cylindrical container with a lid. Camera, frames, dimensions, mass, fingertip, force, and controller details are still required.
