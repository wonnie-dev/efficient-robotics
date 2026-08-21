# Experiment Protocol

The machine-readable protocol is [final_evaluation_protocol.json](../configs/research/final_evaluation_protocol.json). Episode assignments are declared in [reserved_test_episodes.json](../configs/research/reserved_test_episodes.json).

## Data split

- Method development and calibration use non-reserved episodes only.
- The reserved test split contains seeds `1100–1159` and remains closed until the method, calibration models, action costs, and artifact hashes are frozen.
- No reserved test observation may be used for model selection, prompt changes, threshold selection, or failure repair.

Training is not performed. Calibration fits the interpretation of pretrained-model evidence and action-conditioned observation likelihoods. Testing evaluates the frozen system without further adjustment.

## Scenario families

1. **Visible/Open** — the target is visible inside or outside the container.
2. **Partially Occluded** — a right or close-high wrist view resolves target identity.
3. **Covered Container** — the policy removes the cover, observes the changed scene, and replans.
4. **Ambiguous Inside/Outside** — similar candidates require joint identity and membership reasoning.
5. **Target Absent** — negative evidence should lead to safe deferral rather than a distractor grasp.

Each non-absence subcondition has six reserved episodes. Target Absent has twelve, for a total of sixty episodes.

## Comparisons

- immediate grasp;
- fixed right view;
- fixed close-high view;
- confidence-greedy control;
- one-step information gain;
- open-loop belief planning;
- direct VLM action selection;
- random feasible action;
- simulator-state oracle, reported only as an upper bound;
- proposed task-risk-aware joint-belief planner.

The ablations remove joint belief, calibration, negative evidence, action-conditioned future belief, task-risk cost, persistent tracking, or scene-conditioned view prediction one component at a time.

## Metrics

Primary metrics are semantic task success, wrong commitment rate, target-absent safe-deferral rate, and realized task cost. Diagnostic metrics include identity, membership, and absence calibration; proposal recall; selected-candidate accuracy; action count; runtime excluding video encoding; physical execution success; and condition-wise failure causes.

Success-rate intervals use Wilson 95% intervals. Paired binary outcomes use the exact McNemar test, and paired continuous metrics use a seeded bootstrap.

## Physical success

A physical grasp counts only when it selects the correct target, lifts at least `0.15 m`, maintains bilateral contact, satisfies force and penetration limits, avoids unexpected collision, and uses neither object attachment nor pose copying.

## Evaluation gate

Reserved testing begins only after the frozen calibration package and evaluation authorization manifest pass their integrity checks. Calibration or development results must not be reported as final test performance.
