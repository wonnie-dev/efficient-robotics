# Experiment Protocol

The machine-readable protocol is `configs/research/icra_simulation_evaluation_protocol_v1.json`.

## Data split

- calibration and method development: seeds `165-199` plus explicitly declared non-reserved calibration seeds;
- reserved final test: seeds `200-209`;
- reserved outcomes remain closed until every paper parameter and hash is frozen.

Training is not performed. Calibration adjusts score interpretation and observation likelihoods on labeled calibration episodes. Testing evaluates frozen methods on untouched episodes.

## Scenario families

1. Open-container active view: target identity and relation ambiguity resolved by a reachable wrist view.
2. Covered-container search: cover removal followed by positive or empty-container evidence, replanning, and retrieval.

## Required comparisons

- direct perception followed by immediate grasp;
- deterministic Scene Graph planner;
- object uncertainty without uncertain relation edges;
- fixed-viewpoint policy;
- greedy one-step information gain;
- entropy-only planning without task risk;
- proposed action-conditioned task-risk-aware belief planner.

The protocol also predeclares ablations for relation uncertainty, calibration, action-conditioned prediction, negative evidence, task loss, and planning horizon.

## Metrics

Primary:

- task success rate;
- wrong commitment rate;
- total cost to successful retrieval.

Secondary metrics include calibration error, Brier score, negative-evidence accuracy, action count, runtime, collision rate, belief updates, and replanning count.

## Physical success

A physical grasp counts only when it selects the correct target, lifts at least `0.15 m`, maintains bilateral contact, satisfies force and penetration limits, avoids unexpected collision, and uses neither attachment nor pose copying.

## Current gate

Final testing is blocked until the action-conditioned observation model, task-cost weights, and commitment gate are frozen. Calibration results must not be reported as final paper performance.
