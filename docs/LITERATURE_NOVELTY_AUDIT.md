# Literature and Novelty Audit

**Audit date:** 2026-07-24

## Purpose

This file defines the nearest-work comparison that Codex and paper drafts should use. It is a research-positioning audit, not a guarantee of novelty or acceptance. `docs/FINAL_RESEARCH_SPEC.md` remains authoritative for implementation.

## Fixed embodiment verification

Dinesh and Park report a UR10e collaborative arm, OnRobot RG6 gripper, wrist-mounted Zivid 2 3D camera, ROS 2 modules, and cuRobo-based motion generation. Their framework uses foundation models and a dynamically updated Scene Graph for language-guided long-horizon manipulation.

Primary source: https://arxiv.org/abs/2510.27558

## Nearest-work comparison

### Dinesh and Park: foundation models plus dynamic Scene Graph

What already exists:

- language-guided long-horizon task planning;
- foundation-model perception and reasoning;
- persistent Scene Graph updates;
- conventional collision-aware motion planning and feedback.

Required difference for this project:

- calibrated beliefs over competing target and relation hypotheses;
- action-conditioned prediction of future posterior belief;
- task-loss-aware selection of observation and interaction actions;
- explicit negative-evidence updates after an inspected hypothesis fails.

### RoboEXP: action-conditioned Scene Graph through interactive exploration

What already exists:

- physical interaction used to reveal scene structure;
- action-conditioned Scene Graph representation;
- interactive exploration for manipulation.

Required difference for this project:

- language-conditioned target retrieval rather than general scene exploration;
- probabilistic relation edges tied to a specific decision;
- optimization of wrong commitment and terminal task loss.

Primary source: https://proceedings.mlr.press/v270/jiang25c.html

### RoboRetriever: retrieval plus active and interactive perception

What already exists:

- natural-language object retrieval;
- dynamic hierarchical Scene Graph;
- wrist-camera active viewpoint selection;
- physical interaction and manipulation in a unified system.

Required difference for this project:

- an explicit calibrated belief over target and relation hypotheses;
- action-conditioned posterior prediction rather than VLM-selected camera poses alone;
- a decision-theoretic objective that measures wrong commitment and total retrieval cost;
- controlled node-only, greedy, and entropy-only baselines.

Primary source: https://arxiv.org/abs/2508.12916

### VLMPC: VLM-guided model predictive control

What already exists:

- VLM-conditioned candidate action generation;
- action-conditioned future visual prediction;
- MPC-style candidate evaluation using pixel and VLM costs.

Required difference for this project:

- structured posterior prediction over task-relevant Scene Graph beliefs;
- relation uncertainty and negative-evidence updates;
- terminal task-loss and commitment-risk optimization.

Primary source: https://arxiv.org/abs/2407.09829

### ReKep: relational constraints and optimization

What already exists:

- VLM-generated relational keypoint constraints;
- hierarchical optimization in a closed perception-action loop;
- broad manipulation generality without task-specific demonstrations.

Required difference for this project:

- probabilistic competing scene hypotheses rather than deterministic relational costs;
- information-seeking actions chosen to reduce expected retrieval error;
- calibrated relation belief and posterior replanning.

Primary source: https://arxiv.org/abs/2409.01652

### SCOUT: uncertainty-guided probabilistic Scene Graph exploration

What already exists:

- probabilistic object hypotheses in an online 3D Scene Graph;
- expected semantic certainty gain, coverage gain, and travel cost for viewpoint selection;
- a closed active-perception loop.

Relevant stated limitation:

- the reported uncertainty is attached to object hypotheses, while relational-edge uncertainty is identified as future work.

Required difference for this project:

- calibrated relation-edge beliefs;
- manipulation actions as well as traversal/viewpoints;
- language-conditioned object retrieval and task-loss-aware commitment control.

Primary source: https://arxiv.org/abs/2606.06721

## Defensible contribution boundary

The following broad claim is not sufficient:

> We combine a VLM, a Scene Graph, active re-observation, and MPC.

The defensible claim is:

> We formulate language-guided retrieval as task-loss-aware belief-space control over calibrated object and relation hypotheses. The robot predicts how candidate viewpoint and manipulation actions change the posterior Scene Graph, incorporates positive and negative evidence, and minimizes wrong commitment and total retrieval cost.

## Reviewer stress test

The paper is at high risk of rejection as system integration if any of the following is true:

- a language-model or entropy heuristic is labeled belief-space MPC;
- relation values are raw model confidence without calibration analysis;
- no action-conditioned observation or posterior model is implemented;
- the experiment contains only one container and one camera move;
- opening an empty container does not change posterior hypotheses;
- no object-node-only, fixed/random, greedy-information, entropy-only, or direct-execution baseline is included;
- the paper reports entropy reduction but not task success, wrong commitment, and total cost;
- only simulation demonstrations are provided;
- hardware details do not match the declared UR10e/RG6/Zivid 2 embodiment.

## Current verdict

The direction is defensible only in its narrowed form:

**calibrated relation belief + action-conditioned future posterior + task-loss-aware belief-space planning + negative-evidence replanning on UR10e/RG6/Zivid 2.**

The container scenario is an evaluation vehicle, not the contribution. Acceptance will depend on implementation quality, strong baselines, repeated simulation and real-robot trials, and a clear reduction in wrong commitment or total retrieval cost.
