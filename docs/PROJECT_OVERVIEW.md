# Project Overview

## Research question

How should a robot act when a language instruction identifies a target but the current image does not resolve the target identity, container membership, or occlusion relation?

The robot should not grasp from one uncertain observation. It should compare actions by their expected effect on the task belief and execute only the first action before observing and planning again.

## Scenario

The starting scenario is **Covered Basket / Container Active Re-observation**.

Example instruction:

```text
Find and pick up the red mug with the white logo.
```

The target may be inside the container, outside it, behind it, or hidden by a cover. Similar objects create identity ambiguity.

The benchmark increases partial observability in three stages:

1. Open container: target ranking and `inside`, `outside`, and `near` evidence.
2. Active re-observation: wrist-camera motion for `behind` and `occluded_by` ambiguity.
3. Covered container: cover removal, empty-container evidence, and another planning step.

## Method

The state is a probabilistic Scene Graph. Object nodes hold target-identity belief. Relation edges hold beliefs for action-relevant relations such as `inside`, `behind`, `occluded_by`, and `covered_by`.

For each feasible action, the planner predicts a distribution over the next observation and posterior belief. It compares action sequences using wrong-commitment loss, execution risk, motion or interaction cost, and noncompletion cost. It executes the first action, receives a new RGB-D observation, updates the graph, and replans.

## Method focus

The method is not defined by the use of a VLM or Scene Graph alone. Its distinguishing elements are:

- a calibrated joint belief over target identity, container membership, and target absence;
- one semantic action representation for camera motion, cover interaction, grasp, and safe deferral;
- action-conditioned future-belief prediction;
- task-risk-aware receding-horizon action selection;
- closed-loop positive and negative evidence updates;

These elements are evaluated against fixed-view, confidence-only, information-gain, open-loop, direct-VLM, and component-ablation policies. The primary outcomes are semantic task success, wrong commitment, target-absent safe deferral, and realized task cost.

## Scope and assumptions

The reference embodiment is a UR10e with an RG6 gripper and wrist RGB-D camera. Qwen provides instruction-conditioned evidence over anonymous candidates. Learned masks and RGB-D geometry provide object location and relation evidence. A discrete belief-tree planner selects semantic actions.

Viewpoint actions use a fixed library of reachable wrist poses. Continuous camera-pose optimization is outside the method scope. Robot geometry, tool and camera transforms, contact parameters, and controller limits are configuration inputs that must be replaced by measured values for a different physical system.
