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

Development proceeds in three stages:

1. Open container: target ranking and `inside`, `outside`, and `near` evidence.
2. Active re-observation: wrist-camera motion for `behind` and `occluded_by` ambiguity.
3. Covered container: cover removal, empty-container evidence, and another planning step.

## Method

The state is a probabilistic Scene Graph. Object nodes hold target-identity belief. Relation edges hold beliefs for action-relevant relations such as `inside`, `behind`, `occluded_by`, and `covered_by`.

For each feasible action, the planner predicts a distribution over the next observation and posterior belief. It compares action sequences using wrong-commitment loss, execution risk, motion or interaction cost, and noncompletion cost. It executes the first action, receives a new RGB-D observation, updates the graph, and replans.

## Intended contribution

The contribution is not the use of a VLM or Scene Graph alone. The intended contribution is the action layer:

- one semantic action representation for camera motion, cover interaction, and grasp;
- action-conditioned future-belief prediction;
- task-risk-aware action selection;
- closed-loop positive and negative evidence updates;
- fewer wrong grasps and unnecessary actions than fixed-view or confidence-only policies.

## Current implementation boundary

The simulator uses UR10e, RG6, and a wrist RGB-D camera. Qwen is used for instruction-conditioned target ranking. Learned masks and RGB-D geometry provide relation evidence. A discrete belief-tree planner selects semantic actions.

The current viewpoint library is fixed. The cover and gripper physics use provisional parameters pending lab measurements. Final paper testing and real-robot validation are not complete.
