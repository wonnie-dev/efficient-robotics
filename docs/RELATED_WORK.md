# Related Work and Method Boundary

This repository addresses language-guided retrieval under partial observability. It shares components with recent VLM-based active-perception and belief-space planning systems, but uses a different decision state and objective.

## Closely related systems

**Seeing is Believing: Belief-Space Planning with Foundation Models as Uncertainty Estimators** formulates symbolic belief-space planning with VLM predicate evaluation and information-gathering actions. It uses three-valued symbolic predicates and replanning to support long-horizon mobile manipulation. This repository instead maintains a calibrated probability distribution over coupled target identity, container membership, and target absence, then evaluates sensing, interaction, grasp, and defer actions by expected task cost.

**Scene Exploration by Vision-Language Models** uses an image-aligned 3-D grid and a VLM to propose improved robot viewpoints for semantic queries. This repository uses a fixed, robot-reachable wrist-view library and does not ask the VLM to choose a camera pose. Viewpoint changes, cover removal, grasp, and defer are compared by the same planner using action-conditioned observation models.

## Evaluation boundary

The empirical comparison is designed to separate the contribution of each method component:

- passive or fixed-view perception tests whether active observation is necessary;
- direct-VLM and confidence-greedy policies test whether raw model judgments are sufficient;
- one-step information gain tests whether uncertainty reduction alone matches task-risk planning;
- open-loop belief planning tests the value of observation-conditioned replanning;
- ablations isolate joint belief, calibration, negative evidence, future-belief prediction, task-risk cost, persistent tracking, and scene-conditioned observation models.

No claim of priority follows from component integration alone. The method claim is supported only if the frozen evaluation shows improved task outcomes and reduced wrong commitment across the declared scenario families.

## References

- Linfeng Zhao et al., [Seeing is Believing: Belief-Space Planning with Foundation Models as Uncertainty Estimators](https://arxiv.org/abs/2504.03245), 2025.
- Venkatesh Sripada et al., [Scene Exploration by Vision-Language Models](https://arxiv.org/abs/2409.17641), 2025 revision.
