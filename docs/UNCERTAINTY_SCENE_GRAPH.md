# Uncertainty-Aware Scene Graph

## Interface

The schema defines how task-conditioned object and relation beliefs are stored
and exchanged between perception, calibration, and planning modules. Planner
costs and calibrated likelihood parameters are supplied by experiment
configuration rather than embedded in the graph format.

## Purpose

The graph separates four kinds of information:

1. Direct camera observations such as visibility, 2D box, and depth.
2. Task-conditioned target belief stored on object nodes.
3. Spatial-relation belief stored on object-to-object edges.
4. Graph-level uncertainty and task-risk records consumed later by the action-selection layer.

## Node belief

Every object node has:

- `class_distribution`: belief over semantic classes;
- `target_probability`: probability that the object satisfies the current language instruction;
- `target_uncertainty`: a typed placeholder or computed uncertainty record;
- `source`: the module that produced the belief.

Target probability is task-conditioned. The same physical object may receive a different target probability for a different instruction.

## Relation-edge belief

Every spatial edge has:

- `relation_distribution`: probability distribution over relations such as `inside`, `outside`, `near`, and `behind`;
- `relation_uncertainty`: a typed placeholder or computed uncertainty record;
- `source`: VLM, grounding model, geometry, multi-view fusion, or a test fixture.

The distribution must remain available instead of immediately collapsing to one hard relation label.

## Graph-level belief

`graph_belief` contains the most likely target and separate records for target uncertainty, relation uncertainty, and task-failure risk. Optional values remain `null` when the producing module does not provide them.

For belief-space control with coupled task factors, the optional
`joint_task_state_distribution` stores the exact discrete planner belief
instead of reconstructing it from independent marginals. For example, the
removable-cover scenario uses states such as `inside|covered` and
`outside_near|open`. This is an interface field; its state vocabulary,
transition model, and calibration are supplied by the scenario configuration.

## Provenance and ground truth

Every graph identifies its perception source. `ground_truth_used_for_control` prevents evaluation ground truth from being silently used by the proposed controller. Ground truth may be stored separately for scoring, but it must not be represented as a VLM prediction.

## Probability invariants

- Every probability is in `[0, 1]`.
- Each class or relation distribution should sum to 1 within numerical tolerance.
- An uncertainty value must not be populated without naming its method.
- `calibrated: true` is allowed only after a calibration experiment has been implemented and documented.

## Files

- Schema: `configs/scene_graph/uncertainty_aware_scene_graph.schema.json`
- Illustrative example: `examples/uncertainty_scene_graph.example.json`

The example probabilities are placeholders for interface testing and are not experimental measurements or paper results.
