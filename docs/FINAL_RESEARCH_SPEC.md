# Final Research Specification

## Working title

**Task-Risk-Aware Belief-Space MPC over Relation-Uncertain Scene Graphs for Language-Guided Object Retrieval**

This file is the authoritative current research plan. Older meeting notes are historical context and must not override this specification.

## 1. Research problem

A user asks a robot to retrieve an object from a partially observed tabletop environment. The target may be visually ambiguous, inside an uncertain container, behind another object, partially occluded, or hidden under a removable cover. Acting immediately can produce a wrong grasp, an empty grasp, or an unnecessary environment interaction.

The robot must actively gather task-relevant evidence before committing to an irreversible manipulation action.

## 2. Core research question

Can a UR10e manipulation system maintain calibrated beliefs over target identity and spatial relations in a dynamic Scene Graph, predict how candidate sensing and manipulation actions will change those beliefs, and use receding-horizon belief-space MPC to reduce expected task loss before object retrieval?

## 3. Fixed embodiment and software target

- Simulator: NVIDIA Isaac Sim.
- Manipulator: Universal Robots UR10e.
- Gripper: OnRobot RG6.
- Sensor: wrist-mounted Zivid 2 3D/RGB-D camera.
- Real-system continuity target: the same embodiment used in Dinesh and Park, *Toward Accurate Long-Horizon Robotic Manipulation: Language-to-Action with Foundation Models via Scene Graphs*.
- Expected integration style: ROS 2-compatible modules and a collision-aware motion layer such as cuRobo where practical.

The UR10e, RG6, and Zivid 2 interfaces must be modular, but this does not authorize changing the final embodiment.

## 4. Proposed closed loop

1. Receive a natural-language retrieval instruction and an RGB-D observation.
2. Ground candidate targets and relevant scene entities.
3. Build or update a probabilistic task-conditioned Scene Graph.
4. Maintain beliefs over target identity, target location, and spatial-relation edges.
5. Generate candidate high-level actions: move viewpoint, inspect a container, remove a cover, move an occluder, or grasp.
6. Predict the observation and posterior belief expected after each candidate action sequence.
7. Optimize expected terminal task loss, wrong-commitment risk, execution risk, and motion cost over a receding horizon.
8. Execute only the first action, observe again, update the graph from positive or negative evidence, and replan.
9. Commit to grasp only when the expected risk is below a defined threshold.

## 5. Belief representation

### Object-node beliefs

Relevant nodes may contain:

- probability of being the requested target;
- class or identity belief;
- existence belief;
- pose mean and uncertainty;
- visibility and occlusion state;
- action-relevant affordances.

### Relation-edge beliefs

Relevant edges may contain calibrated probabilities for:

- `inside` / `outside`;
- `behind`;
- `occluded_by`;
- `covered_by`;
- `supported_by`;
- other relations only when they affect the current task decision.

The graph is task-conditioned. It does not need to model every visible object with equal detail.

## 6. Belief update and calibration

Raw VLM self-confidence is not accepted as a calibrated probability.

The implementation should combine controlled evidence such as:

- grounding or classification scores;
- multi-view consistency;
- simulator or calibration-set likelihood models;
- temperature scaling, isotonic calibration, or another validated calibration method;
- Bayesian or Dirichlet-style posterior updates;
- explicit negative evidence when an inspected region does not contain the target.

The first deterministic simulation can use known observation likelihoods. Learned perception should be integrated only after the belief-space loop is validated.

## 7. Belief-space MPC formulation

The planner evaluates action sequences over a short horizon. A conceptual objective is:

`J = expected terminal task loss + wrong-commitment risk + execution/collision risk + motion/interaction cost`.

Task loss should penalize:

- retrieving the wrong object;
- grasping an empty or incorrect location;
- taking an irreversible action while belief is insufficient;
- failing to retrieve the requested target within the action budget.

The planner must model action-conditioned future belief. A fixed viewpoint rule, random exploration, or one-step entropy heuristic is a baseline, not the proposed controller.

Because the action set includes discrete semantic choices and continuous robot motion, the implementation may use a hierarchical or hybrid solver. The paper claim should match the actual solver.

## 8. Core novelty claims

### Novelty 1: Calibrated task-conditioned relation belief

The Scene Graph stores uncertainty not only on object hypotheses but also on action-relevant relation edges. This allows uncertainty such as “target is inside container A” versus “target is behind container A” to directly affect decisions.

### Novelty 2: Action-conditioned future-belief planning

The planner predicts how a camera motion or manipulation action changes the future graph belief. This distinguishes the method from systems that update a graph only after an action has already been chosen by a heuristic or language model.

### Novelty 3: Task-loss-aware commitment control

The controller does not maximize generic information gain. It selects evidence-gathering actions based on their expected effect on retrieval success, wrong commitment, execution risk, and total task cost.

### Novelty 4: Negative-evidence closed-loop replanning

When the robot inspects the most likely location and does not find the target, that failure is treated as informative evidence. The posterior graph changes, and the next action is replanned rather than repeating the original plan.

## 9. How this differs from nearby work

- **Dinesh and Park:** use foundation models, a dynamically updated Scene Graph, and conventional motion planning for long-horizon manipulation. The proposed work adds calibrated relational beliefs and uses expected belief change and task loss to select sensing and manipulation actions.
- **RoboEXP:** builds action-conditioned Scene Graphs through interactive exploration. The proposed work focuses on language-guided retrieval and optimizes task-specific commitment risk rather than scene exploration completeness.
- **RoboRetriever:** already combines dynamic Scene Graph memory, active viewpoint selection, physical interaction, and object retrieval. The proposed work must therefore contribute an explicit calibrated relation-belief model, action-conditioned posterior prediction, and decision-theoretic task-loss objective rather than merely another integrated retrieval pipeline.
- **VLMPC:** predicts future visual outcomes and scores candidate actions with visual and VLM costs. The proposed work predicts future structured beliefs over task-relevant relations and performs posterior updates from negative evidence.
- **ReKep:** optimizes VLM-generated relational keypoint constraints. The proposed work represents uncertainty over competing scene hypotheses and chooses actions to reduce expected wrong commitment rather than treating relational constraints as deterministic costs.
- **SCOUT:** uses uncertainty-guided viewpoints for probabilistic Scene Graph construction, but its reported uncertainty is attached to object hypotheses while relation-edge uncertainty is left as future work, and its objective is semantic scene coverage rather than manipulation task loss.

## 10. Experimental scenarios

### Scenario A: Multi-hypothesis open-container retrieval

The target may be inside container A, inside container B, behind a container, or among visually similar distractors. The robot chooses wrist-camera viewpoints before grasping.

### Scenario B: Removable-cover search with negative evidence

The robot must decide which covered region to inspect. If the target is absent after uncovering the most likely region, the posterior is updated and another region is selected.

### Scenario C: Occluder manipulation

The robot chooses between moving the camera, moving an occluder, inspecting another hypothesis, or committing to grasp.

At least one scenario must require two or more belief updates and replanning cycles.

## 11. Primary metrics

1. **Task Success Rate:** correct target retrieved and task completed.
2. **Wrong Commitment Rate:** wrong-object grasp, empty grasp, premature irreversible interaction, or action based on an incorrect relation belief.
3. **Total Cost to Successful Retrieval:** weighted combination of elapsed time, robot path length, number of observations, environment interactions, and failed commitments.

## 12. Secondary metrics

- Brier score or Expected Calibration Error;
- target-node and relation-edge accuracy;
- negative-evidence update accuracy;
- planning time and control frequency;
- collision and execution-failure rate;
- number of belief updates and replanning cycles.

## 13. Required baselines

1. Direct perception/VLM plus immediate grasp.
2. Deterministic Scene Graph planner.
3. Object-node uncertainty without relation-edge uncertainty.
4. Fixed or random viewpoint policy.
5. Greedy one-step information gain.
6. Entropy-only planner without task-loss or commitment-risk terms.
7. Proposed full method.

## 14. Essential ablations

- remove relation-edge uncertainty;
- remove calibration and use raw scores;
- remove action-conditioned future-belief prediction;
- remove negative-evidence updates;
- remove task-loss or wrong-commitment terms;
- reduce the horizon to a one-step greedy policy.

## 15. Minimum evidence for a credible ICRA submission

### Simulation

- all three scenario families if feasible, at least two mandatory;
- randomized object poses, target hypotheses, occlusion levels, and distractors;
- several hundred total episodes across methods and scenarios;
- multiple random seeds, confidence intervals, and failure-category reporting;
- synchronized logs of observations, beliefs, candidate costs, selected actions, and outcomes.

### Real robot

- UR10e + RG6 + wrist-mounted Zivid 2;
- at least two scenario families;
- repeated trials for the full method and key baselines, not demonstration videos alone;
- disclosed hardware, calibration, latency, safety limits, and failure cases.

## 16. Implementation order

1. Audit Isaac Sim, UR10e assets, RG6 model availability, Zivid 2 approximation, Python environment, GPU, and repository state.
2. Load and control UR10e in a minimal deterministic tabletop scene.
3. Attach and validate the RG6 articulation and grasp frame; use a proxy only as a temporary isolated debug configuration.
4. Attach a wrist RGB-D sensor with documented Zivid 2 geometry/intrinsics assumptions.
5. Implement randomized scenario generation and ground-truth relation labels.
6. Implement the Scene Graph schema and deterministic belief update with known observation likelihoods.
7. Implement camera/viewpoint actions and a greedy information-gain baseline.
8. Implement action-conditioned future-belief prediction and receding-horizon task-loss planning.
9. Add cover and occluder interactions plus negative-evidence updates.
10. Add real perception, grounding, segmentation, and VLM interfaces.
11. Run baselines, ablations, batch experiments, statistical analysis, and video export.
12. Transfer to the matching real robot setup.

## 17. Non-negotiable boundaries

- The contribution is not the container scenario itself.
- The contribution is not merely connecting a VLM, Scene Graph, and MPC.
- The proposed method must use action-conditioned future belief; otherwise label it as a heuristic baseline.
- Relation probabilities must be calibrated or evaluated for calibration.
- The final paper must show that the method reduces wrong commitment or total retrieval cost, not only graph entropy.
- Simulation and real-robot results must use the UR10e/RG6/Zivid 2 embodiment unless the user explicitly changes the project decision.

## 18. One-sentence paper story

A language-guided UR10e robot maintains calibrated beliefs over target objects and spatial relations, predicts how viewpoint and manipulation actions change those beliefs, and uses task-loss-aware belief-space MPC with negative-evidence replanning to avoid wrong commitments during object retrieval.
