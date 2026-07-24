# Efficient Robotics Project - Complete Research and Implementation Handoff

## Document purpose and use

This document is a detailed handoff of the user's robotics research project for use with OpenAI Codex CLI, VS Code, Isaac Sim, and the lab server. It is not a one-paragraph summary. It preserves the evolution of the research direction, the content of the first through fourth research meetings as far as the stored records allow, the current technical topic, the selected scenario, the proposed novelty, the evaluation plan, the simulation and real-robot strategy, and the operational constraints that an implementation agent must follow.

Any older working titles recorded in this historical handoff are historical only. The current working title is defined by `docs/FINAL_RESEARCH_SPEC.md`.

The project should be treated as an ICRA 2027 submission project with an intended submission deadline around September 15, 2026. Some old planning files were named “ICRA 2026,” but they refer to the 2026 research and submission period for the ICRA 2027 paper.

This document is a reconstruction from the user's chat history, uploaded meeting materials, presentation files, research-plan documents, the July 6 meeting transcript, scenario documents, and email screenshots. The fourth meeting section is based on an actual transcript. Some earlier meeting sections are detailed reconstructions from slides and discussion records rather than verbatim transcripts. Do not invent a quotation or claim that a professor said a sentence word-for-word unless that exact wording is present in a transcript or email.

## Instructions for Codex or another coding agent

1. Read this entire file before changing the project architecture.
2. Treat the items marked “current decision” or “hard constraint” as binding until the user explicitly changes them.
3. Use the fixed simulation embodiment defined in `docs/FINAL_RESEARCH_SPEC.md`: UR10e, OnRobot RG6, and a wrist-mounted Zivid 2 3D/RGB-D camera. Do not substitute another final embodiment without explicit user approval.
4. Do not reduce the research to a generic VLM plus MPC demo. The research claim depends on uncertainty-aware relational scene graphs and information-seeking MPC actions.
5. Separate verified implementation facts from proposed research hypotheses.
6. Before editing code, inspect the repository, existing Isaac Sim files, robot assets, Python environment, GPU, and dependency versions.
7. Use Git from the beginning. Commit working checkpoints before major changes.
8. Save commands, experiment configurations, random seeds, logs, metrics, videos, and failure cases so results can be reproduced on the desktop, laptop, and server.
9. Do not assume that Codex chat history will move automatically between machines. The repository and project documentation are the durable source of truth.
10. When a design decision changes, update this file or a dedicated decision log in the repository.

---

# 1. Project identity, people, venue, and operational context

## 1.1 Project name

Working project name: efficient-robotics.

The project began as an attempt to combine an RT-2-style vision-language-action system with model predictive control, then moved toward a more feasible modular architecture using a pretrained vision-language model, a scene graph, and MPC. The direction later narrowed further to uncertainty-aware target and spatial-relation grounding, with the robot acting to reduce uncertainty before executing a potentially wrong manipulation.

## 1.2 Target venue and deadline

- Primary target: ICRA 2027.
- Working submission deadline: September 15, 2026.
- The user strongly wants the paper accepted at ICRA and therefore prioritizes a technically defensible contribution, real-robot validation, clear baselines, and a focused experimental story over a broad but incomplete system.

## 1.3 People and roles

- Wonhee Koh: primary student researcher and likely first author. Responsible for research definition, literature study, Isaac Sim development, implementation, experiment management, result analysis, and paper drafting.
- Professor Eungjoo Lee: primary supervising professor at the University of Arizona. Provides regular advising, narrows scope, requests progress reports, and coordinates research direction and writing speed.
- Professor Shinkyu Park: external research collaborator/advisor at KAUST. Provides robotics, MPC, scene-graph, scenario, experiment, and paper-framing feedback. His team is expected to help with simulation or real-robot validation where appropriate.
- Hansol Ko: senior collaborator involved in research discussion and VLM-related material collection.
- Additional students or collaborators may participate, but their specific implementation responsibility must not be assumed unless explicitly assigned.

## 1.4 Location and computing arrangement

The user works across several locations and machines.

- Local desktop: Windows 11, intended for Isaac Sim GUI, scene creation, interactive debugging, visualization, and initial integration.
- Laptop: used for code editing, documentation, and smaller tests while mobile.
- Lab server: Ubuntu/Linux server accessed through VPN and VS Code Remote SSH. The server currently reports six NVIDIA RTX A6000 GPUs with approximately 49 GB memory each, indexed 0 through 5. It is intended for headless simulation, larger experiment batches, model inference, repeated evaluation, and long-running jobs.
- Real-world validation: planned in Professor Shinkyu Park’s laboratory in Saudi Arabia. The user does not plan to create a new public dataset as the main contribution.
- Local GPU information appeared inconsistently in prior setup records as RTX 4070 Super or RTX 5070. The implementation agent must verify the current machine using `nvidia-smi` instead of relying on old notes.

## 1.5 Robot and simulator constraints

- Simulator: NVIDIA Isaac Sim.
- Fixed simulation embodiment: Universal Robots UR10e with an OnRobot RG6 gripper and wrist-mounted Zivid 2 3D/RGB-D camera.
- This embodiment follows the hardware used in Professor Shinkyu Park's scene-graph manipulation paper and is the required target for the current Isaac Sim project.
- Initial simulation should support tabletop manipulation, object placement, perception views, and active re-observation.
- The final research should be shown in simulation first and then, when feasible, on a real robot.

---

# 2. Full evolution of the research direction

## 2.1 Initial RT-2 and VLA plus MPC direction

The earliest project direction was to combine an RT-2-style vision-language-action model with MPC. The user initially wanted RT-2 to be mandatory and preferred to begin with code before simulation. The intended pipeline was image plus natural-language command to action tokens, followed by an MPC layer that would refine or constrain the resulting action.

The user experimented with unofficial or third-party RT-2-style repositories because the original Google RT-2 system did not provide a directly usable public implementation and checkpoint for the intended robot. The experiments included tokenized action representations, small demonstration training loops, fixed-length joint-action tokens, offline inference, and a simple MPC refinement layer with joint limits, rate limits, clamping, and step-size constraints.

This phase produced several practical lessons.

- A true RT-2 reproduction would require large robot datasets and substantial training infrastructure.
- Public RT-2 artifacts were not directly aligned with the intended laboratory embodiment and research timeline.
- A paper based only on “VLA output plus an MPC correction” risked being technically shallow.
- Environment problems, video decoding problems, TorchCodec or ffmpeg warnings, OpenMP conflicts, and Windows GPU library issues created implementation risk.
- Moving some work to WSL or Linux could solve library problems, but the research contribution itself still required refinement.

## 2.2 Shift from full VLA training to modular VLM plus MPC

After discussion with the supervising professor, the research direction shifted from training or reproducing a full VLA model to a modular VLM-MPC framework.

The reasoning was:

- A pretrained VLM can perform scene understanding, object interpretation, relation reasoning, and high-level target extraction without requiring a large robot-action dataset.
- MPC can provide stable short-horizon corrective control and explicitly handle control constraints.
- A modular system is more feasible within the submission timeline.
- The architecture can be inspected and ablated more clearly than a monolithic VLA model.

The proposed division of responsibility became:

- VLM: determine what the scene contains, what object is relevant, and what semantic relation or goal is requested.
- Scene Graph: preserve object identity, state, relation, containment, and task progress over time.
- MPC: calculate how to move safely and correct execution based on current state, constraints, and predicted short-horizon outcomes.

The original high-level loop was:

User command -> VLM perception -> Scene Graph update -> MPC input generation -> MPC control -> robot execution -> feedback -> Scene Graph correction and replanning.

## 2.3 Scene Graph becomes more than optional memory

The research then focused on whether the Scene Graph should be only a memory store or should directly affect MPC.

Several possible roles were discussed:

1. Memory-only Scene Graph:
   - Store object locations, states, and relations.
   - Tell the planner what has already happened.
   - Weakest technical contribution if used only as a log.

2. State-aware interface:
   - Convert object states, relations, task progress, and VLM outputs into MPC targets, constraints, and costs.
   - Stronger because the graph affects control decisions.

3. Scene Graph as differentiable state:
   - Encode the graph into a state representation consumed by neural MPC.
   - Potentially allow planning loss to influence graph construction.
   - Technically ambitious and high risk for the available timeline.

4. Action-conditioned graph model:
   - Predict how candidate actions would change the Scene Graph.
   - Use future graph states as an internal world model for MPC.
   - Strong contribution but more difficult to implement reliably.

5. Uncertainty-propagating graph:
   - Represent uncertain object grounding and uncertain spatial relations in graph nodes and edges.
   - Use that uncertainty to make the MPC risk-aware and information-seeking.
   - This later became the most practical and promising focus.

## 2.4 Shift from generic efficiency to uncertainty-aware control

The initial proposed novelty included “efficient modular VLM-MPC,” but feedback indicated that efficiency alone should be a motivation rather than the central contribution. Simply using an off-the-shelf VLM instead of training a VLA model would not be enough for ICRA.

The direction therefore narrowed to the interface between uncertain semantic perception and robot control.

The core problem became:

- The VLM or grounding model may be uncertain about which object is the target.
- The system may be uncertain about spatial relations such as inside, outside, near, behind, on, or occluded by.
- If the robot immediately executes the most likely action, it can pick the wrong object, open the wrong container, or manipulate the scene unnecessarily.
- The robot should detect task-relevant uncertainty and choose an action that reduces that uncertainty before final manipulation.

This produced the current direction:

VLM-based target object grounding and spatial-relation grounding uncertainty are represented in a Scene Graph, and MPC selects information-gathering robot actions before final manipulation to prevent wrong actions.

## 2.5 Current research topic

Recommended working title:

Uncertainty-Aware Relational Scene Graph MPC for Language-Guided Object Retrieval under Partial Observability

Alternative stronger title used in discussion:

BeliefGraph-MPC: Information-Seeking Model Predictive Control over Uncertainty-Aware Scene Graphs for Language-Guided Object Retrieval

The exact final title is not fixed. The title must reflect the implemented method rather than promise a capability that is not experimentally demonstrated.

Current one-sentence project definition:

The system stores calibrated uncertainty about the requested target object and spatial relations in Scene Graph nodes and edges, then uses MPC to select re-observation or manipulation actions that reduce task failure risk before retrieving the object.

---

# 3. First research meeting - detailed reconstructed record

## 3.1 Context

The first meeting was the initial serious discussion of whether a VLM plus Scene Graph plus MPC architecture could form an ICRA paper. The presentation framed the project as an efficient modular alternative to a full VLA system and used Professor Park’s scene-graph-based long-horizon manipulation framework as a primary reference.

## 3.2 Material presented

The presentation contrasted four research streams.

### VLA systems

Examples included RT-2 and RT-X. These systems directly connect vision-language input to robot actions. Their strengths are generality and end-to-end action generation. Their weaknesses for this project are large robot-data requirements, training cost, black-box behavior, and implementation risk within a short timeline.

### VLM plus MPC systems

VLMPC and related work were treated as the nearest examples of directly combining semantic vision-language reasoning with MPC. These systems use VLMs for action candidates, cost reasoning, trajectory evaluation, or video-conditioned prediction. The identified gap was that long-horizon object state, containment, relation, and task progress were not necessarily represented as an explicit persistent Scene Graph.

### VLM plus optimization systems

VoxPoser and ReKep were discussed as examples of transforming language and visual reasoning into intermediate robot representations such as value maps, affordances, keypoints, costs, and constraints. This was important because the user’s system also needed an explicit VLM-to-MPC interface rather than sending free-form language directly into low-level control.

### Scene Graph and foundation-model systems

Professor Park’s related framework was used to understand a modular architecture in which an LLM, VLM, Scene Graph, and motion planner play separate roles. The important lesson was that a Scene Graph can act as a world model or memory that preserves object location, object state, relations, and task progress.

## 3.3 Proposed pipeline at the meeting

- A user gives a high-level natural-language command.
- The VLM interprets objects, target locations, relations, affordances, and task-relevant points.
- The Scene Graph stores and updates the world state.
- The system converts the graph and VLM output into MPC target, constraint, and cost terms.
- MPC performs short-horizon control.
- The robot executes.
- Visual and state feedback update the Scene Graph.
- The loop continues until the task is finished or recovery is required.

## 3.4 Proposed novelty at the meeting

The early novelty candidates were:

- Efficient modular VLM-MPC without training a full VLA model.
- State-aware VLM-to-MPC interface using Scene Graph state, relation, and task progress.
- Feedback-based state correction in which visual feedback changes the graph and subsequent control.

## 3.5 Experiment candidates at the meeting

- Basic pick-and-place.
- Cluttered-scene manipulation.
- Dynamic target or execution-feedback correction.
- Long-horizon multi-step manipulation.

Initial ablation candidates:

- VLM-only.
- MPC-only.
- VLM plus MPC.
- VLM plus Scene Graph plus MPC.

Initial metrics:

- Task success rate.
- Collision rate.
- Replanning count.
- Scene Graph update accuracy.

## 3.6 Key questions raised

- Is VLM plus Scene Graph plus MPC sufficiently convincing for ICRA?
- Which task best reveals the benefit of the architecture?
- Should the Scene Graph be memory only or directly determine MPC targets, constraints, and costs?
- Should MPC follow only a VLM-provided target point or become state-aware using relation and progress information?
- How should semantic outputs be converted into numerical control terms?

## 3.7 Outcome of the first meeting phase

The generic framework was considered a useful direction, but it still lacked a precise technical contribution. The project needed to move beyond “connect three modules” and define a specific failure mode, mathematical interface, or control objective. This started the shift toward uncertainty-aware perception and decision making.

---

# 4. Second research meeting - detailed reconstructed record

## 4.1 Context

The second meeting and the study period around it focused on how learning-based perception and MPC should connect. The user studied VLMPC, neural MPC, OpenVLA, VLA pipelines, residual learning plus MPC, learned dynamics versus analytic dynamics, policy warm-starts, and the problem of converting semantic model output into control variables.

## 4.2 Main technical discussion

The system could not simply ask the VLM for a robot action and then call the result MPC. A valid MPC formulation requires:

- A state representation.
- Dynamics or a predictive transition model.
- A finite horizon.
- A cost function.
- Constraints.
- Candidate controls or an optimization method.

The main interface question became:

How can the VLM and Scene Graph provide numerical targets, constraints, and costs that an MPC optimizer can use?

Examples of this conversion included:

- Target object position -> terminal or tracking cost.
- Obstacle and interference objects -> collision constraints.
- Spatial relation such as inside or behind -> geometric relation cost.
- Task progress -> change in active sub-goal.
- Object state such as open or closed -> action feasibility constraint.
- VLM uncertainty -> risk cost, stop condition, re-observation trigger, or information-gain objective.

## 4.3 Scene Graph design discussion

The Scene Graph was considered as a dynamic state representation rather than a static caption.

Possible node fields:

- Object ID.
- Semantic label.
- Target probability.
- Position or pose.
- Visibility.
- Open or closed state.
- Graspability or affordance.
- Observation timestamp.
- Confidence or uncertainty.

Possible edge fields:

- Inside.
- Outside.
- Near.
- On.
- Behind.
- In front of.
- Occluded by.
- Attached to.
- Relation probability or uncertainty.

The graph must be updated after each important observation or physical action.

## 4.4 Method candidates considered

The internal research-plan document listed several technically ambitious ideas:

- Task-conditioned Scene Graph generation.
- Differentiable Scene Graph to MPC state mapping.
- Action-conditioned Scene Graph forward model.
- Adaptive sparse graph attention.
- Temporal graph keyframe compression.
- Uncertainty-propagating Scene Graph.
- Counterfactual reasoning through graph edits.
- Cross-domain transfer to surgical robotics.

At this stage, the differentiable and action-conditioned versions were attractive but carried substantial implementation risk. The uncertainty-propagating direction was initially considered one candidate among several, but later became the core because it aligned better with a concrete scenario and wrong-action prevention.

## 4.5 Advisor-level constraints that emerged

- A generic framework alone would be weak.
- The technical contribution should be visible in the controller or the perception-control interface.
- Efficiency should not be the only novelty.
- A simple pick-and-place demo may be insufficient unless it exposes the intended uncertainty or recovery mechanism.
- Real-robot evidence is important.
- The contribution must be scoped so it can be completed before writing and submission.

## 4.6 Outcome of the second meeting phase

The project retained the modular VLM, Scene Graph, and MPC architecture, but the next step was to identify a specific uncertainty type and a scenario in which robot movement was necessary to resolve that uncertainty.

---

# 5. Third research meeting - detailed reconstructed record

## 5.1 Context

The third meeting phase focused on uncertainty-aware robotic manipulation. The user reviewed papers and concepts such as KnowNo, VLMPC, Traj-VLMPC, conformal prediction, model confidence, re-observation, stop or ask policies, and active sensing.

## 5.2 Scope of uncertainty considered

Several uncertainty categories were distinguished.

### Target object grounding uncertainty

The system is unsure which visible object corresponds to the user’s instruction. Example: two visually similar cups are near the same bowl.

### Spatial-relation grounding uncertainty

The system is unsure whether the target is inside, outside, near, behind, on, or occluded by another object or container.

### Localization uncertainty

The target identity is known, but position or pose is uncertain.

### Grasping uncertainty

The object and pose are known, but grasp success is uncertain.

### Task or language ambiguity

The instruction allows multiple valid interpretations and may require asking the user.

The research direction chose target grounding and spatial-relation grounding as the central uncertainty types. Grasping and localization were recognized as heavily studied. Pure language ambiguity could be handled by asking the user without requiring robot movement, so it did not strongly motivate MPC.

## 5.3 Uncertainty mitigation strategies discussed

- Execute immediately using the maximum-confidence hypothesis.
- Stop when confidence is low.
- Ask the user for clarification.
- Move the camera or robot arm to obtain a new view.
- Open a container.
- Move an occluding object.
- Update the Scene Graph with the new evidence.
- Replan the manipulation.

The key research requirement became that uncertainty reduction should require a robot action. Otherwise, the contribution might remain only an LLM/VLM dialogue policy.

## 5.4 Early metric candidates

A larger list of metrics was initially considered, including perception quality, uncertainty calibration, control efficiency, task success, wrong actions, re-observation count, execution time, and graph consistency. Later feedback recommended reducing this list to the few metrics that directly support the MPC-centered claim.

## 5.5 Outcome of the third meeting phase

The project direction was narrowed to:

- target object grounding uncertainty;
- spatial-relation grounding uncertainty;
- robot action to reduce uncertainty;
- Scene Graph update;
- MPC-based decision before final execution;
- wrong-action prevention.

The remaining missing element was a concrete scenario simple enough to implement but strong enough to demonstrate why robot motion, Scene Graph updates, and MPC were all necessary.

---

# 6. Fourth research meeting - transcript-based detailed record

## 6.1 Scope, datasets, and metrics

The discussion began with the need to reduce the research scope. The presentation contained multiple tasks, models, datasets, and six evaluation metrics. Professor Lee asked what ICRA reviewers typically expect compared with computer-vision conferences, where papers may evaluate on several datasets and many SOTA methods.

Professor Park’s response was that robotics papers increasingly include machine-learning-style SOTA comparisons, but actual physical experiments remain especially important. A simulation-only result often receives the criticism that it was not validated experimentally. His practical recommendation was:

- Use approximately one or two highly relevant datasets or evaluation settings.
- Five datasets would be excessive for this project.
- Do not use all six proposed metrics if the paper’s core is MPC.
- Compress or de-emphasize perception-only metrics if the VLM is not the main contribution.
- Focus on the metrics that demonstrate the robot and MPC behavior.
- Prioritize experiments and comparison with existing methods over a broad benchmark sweep.

## 6.2 Need to define the uncertainty method

Professor Lee emphasized that the presentation had defined what uncertainty the project might address, but not yet how uncertainty would be measured methodologically. The team needed to decide whether the novelty would come from a new uncertainty method or from applying an established uncertainty method to a new robotic problem.

The closest related work, including KnowNo, was discussed. KnowNo reduces uncertainty by asking a human when the model is insufficiently certain. Professor Park pointed out that this type of human intervention is not naturally an MPC contribution.

## 6.3 Key question: how does robot motion reduce uncertainty?

Professor Park directly asked whether the user had considered how MPC could reduce uncertainty. The important distinction was:

- Asking a user can reduce uncertainty without moving the robot.
- Re-observation can require a robot or sensor to move to another angle.
- If sensor data from a different viewpoint is necessary, MPC can plan the robot motion that acquires that information.

This established the required connection:

Robot motion -> new sensor observation -> reduced target or relation uncertainty -> updated Scene Graph -> new MPC decision.

## 6.4 Request for a simple and specific scenario

Professor Park asked for a concrete scenario described without abstract terminology. He explained that in a previous mobile-robot object-retrieval project, defining the task first made it easier for students to formalize the research problem.

He gave several examples.

### Mobile robot object retrieval

A user asks a robot to retrieve a remote object. An LLM selects a navigation plan. When multiple solutions or interpretations exist, the system asks the user. This example showed the value of defining a scenario before formalizing the method, but it relied on human clarification and was less directly connected to MPC-based active uncertainty reduction.

### Factory pick-and-place

A manipulator with a gripper and vision system picks object A from a workbench and places it in a tray. Grasping or localization uncertainty can be formulated to improve pick-and-place success. Professor Park noted that grasping and localization uncertainty have already been studied extensively, so the project should be careful about overlap.

### Basket or container spatial-relation uncertainty

Professor Park proposed a simple spatial-relation example. The robot wants to find an object represented in a Scene Graph, but it is uncertain whether the object is inside or outside a basket.

A more concrete example was:

- The robot initially assumes that the object is inside a basket with a lid.
- MPC opens the lid.
- The object is not inside.
- The system updates the Scene Graph or obtains another observation.
- Another MPC operation moves the robot or continues the search.
- The sequence reduces uncertainty through physical interaction and observation.

Professor Park recommended starting with a simple example, developing a concrete idea, then building a more advanced scenario and experiments.

## 6.5 Public datasets versus custom evaluation

Professor Park stated that for robotics, the first priority is experimental validation, followed by comparison showing improved performance over an existing method. It is not necessary to use five or ten public datasets and many SOTA methods in the style of a large computer-vision benchmark paper.

This supports a project design centered on randomized simulation episodes and real-robot trials rather than creating a large new dataset.

## 6.6 Overleaf collaboration and timeline

The group discussed using a shared Overleaf document continuously rather than waiting for meetings. Professor Park said ideas could be added to Overleaf for asynchronous review and revision.

Professor Lee emphasized the short timeline:

- The project needed visible progress within roughly two to two and a half months.
- Results should be substantially complete by the end of August so the user would have enough time for paper writing.
- The user and Professor Lee should meet more frequently to narrow the uncertainty method and scope.

## 6.7 Model selection discussion

Professor Park suggested selecting a strong model based on relevant performance tables and documenting why it was chosen. The user was advised to search recent evidence and compare candidate models rather than selecting one without justification. Grounding DINO was mentioned as suitable for grounding, and a Qwen vision-language model was discussed as a possible VLM candidate. Model names and versions should be re-verified at implementation time because current leaderboards and releases can change.

## 6.8 Fourth meeting outcome

The meeting did not finalize a complete algorithm. It produced the following decisions and tasks:

- Narrow the research scope.
- Use one or two relevant evaluation settings rather than many datasets.
- Reduce the main metrics.
- Make real experiments a priority.
- Define how uncertainty is measured.
- Design a scenario where robot motion is required to reduce uncertainty.
- Use target and spatial-relation grounding uncertainty as the central problem.
- Start from a basket or container example and develop it into a concrete experiment.
- Use Overleaf for continuous idea development.

---

# 7. Scenario selection after the fourth meeting

## 7.1 Scenario candidates considered

### Ambiguous Tabletop Relation

Setup:

- Two cups, one bowl, and one block are placed on a table.
- Example instruction: “Pick up the cup next to the bowl.”
- Both cups may be near the bowl, creating target ambiguity.

Advantages:

- Easiest physical and simulation setup.
- No lid, drawer, or complex articulation.
- Useful for testing target grounding, relation understanding, stopping, re-observation, and wrong-object pick rate.

Weakness:

- Robot motion and MPC may not appear necessary.
- It can look like a perception benchmark rather than a robotics-control contribution.

### Open Container Relation

Setup:

- An open basket, tray, or box is placed on the table.
- Objects are placed near the boundary so inside, outside, or near can be ambiguous from a single view.
- Example instructions: “Pick up the object inside the basket” or “Pick up the block near the container.”

Advantages:

- Easier than lid opening.
- Directly tests inside, outside, and near relations.
- Natural bridge to the covered-container scenario.

Weakness:

- If the scene is too clear, uncertainty may be too weak.
- The camera and object placement must create meaningful ambiguity.

### Occluded Target Active Viewpoint

Setup:

- A target object is partially hidden behind a bowl, box, or other object.
- The initial view cannot clearly determine whether the object is behind or beside the occluder.
- The robot changes the wrist-camera or arm viewpoint.

Example instruction:

“Pick up the red block behind the bowl.”

Advantages:

- Re-observation is clearly necessary.
- MPC can directly choose where to move for a better view.
- Multi-view consistency can be used as an uncertainty signal.
- Easier than opening an articulated lid.

Weakness:

- Strong overlap with conventional active perception if relation uncertainty is not central.

### Covered Basket or Container Active Re-observation

Setup:

- The target may be inside, outside, behind, near, or partially occluded by a basket or container.
- The initial observation is insufficient to determine the target location or relation.
- The system can change viewpoint, remove a lightweight cover, or open a lid in a later stage.

Example instruction:

“Find and pick up the target object inside or near the basket.”

Advantages:

- Contains target grounding uncertainty and relation uncertainty.
- Makes Scene Graph updating meaningful.
- Makes MPC responsible for information-gathering actions rather than only final grasp trajectory.
- Can begin with an open container and gradually add partial occlusion and cover manipulation.

Weakness:

- A fully articulated lid can consume too much implementation time.
- The scenario alone is not novelty; the algorithm and evaluation must carry the paper.

### Drawer or Shelf Hidden Object Retrieval

Setup:

- The target may be inside a drawer, on a shelf, or behind another object.
- The robot opens a drawer or changes viewpoint and updates the Scene Graph.

Advantages:

- Strong long-horizon household-manipulation story.
- Clear inside, on, and behind relations.

Weakness:

- High implementation difficulty in both Isaac Sim and real robot.
- Risky under the current deadline.

### Relation-Constrained Placement

Setup:

- The robot must place an object inside, next to, or behind another object.

Advantages:

- Spatial relations can become explicit MPC costs and constraints.
- MPC contribution can be visually clear in placement.

Weakness:

- The project shifts from target grounding uncertainty to goal-relation uncertainty.
- It changes the central object-retrieval story.

## 7.2 Selected starting scenario

The selected starting point is:

Covered Basket / Container Active Re-observation

However, implementation should begin with the easiest version:

1. Open container.
2. Ambiguous inside, outside, or near relation.
3. Partial occlusion.
4. Active viewpoint change.
5. Lightweight removable cover.
6. Full lid opening only if time and hardware allow.

## 7.3 Why it was selected

- It directly represents target object and spatial-relation uncertainty.
- It requires physical re-observation and therefore gives MPC a necessary role.
- It supports Scene Graph updates after actions.
- It can be built incrementally.
- It is more technically meaningful than a purely ambiguous tabletop image.
- It is less risky than drawer or shelf manipulation.

---

# 8. Email feedback from Professor Park after scenario selection

## 8.1 Scenario document sent

The user emailed Professor Park a document explaining why Covered Basket / Container Active Re-observation was being considered and included an initial simulation plan. Professor Lee and Hansol were copied.

Professor Park initially replied that he was traveling and would review the material later.

## 8.2 Detailed feedback received

Professor Park later replied that the broad direction appeared to be established and proposed the following project structure.

### Uncertainty-aware Scene Graph formation

When an LLM or VLM forms the Scene Graph, the project should consider how uncertainty can be represented systematically. This means the graph should not only contain a single hard label such as “inside.” It should represent uncertain hypotheses about target identity and spatial relation.

### MPC action based on the uncertain graph

The robot’s MPC action should be defined using the uncertainty-aware Scene Graph. This makes the uncertainty representation relevant to actual control.

### Two phases versus one integrated phase

Professor Park noted that “Active Re-observation” and “Covered Container” could be handled as two phases, but they might also be integrated into one phase if the robot uses MPC to reduce uncertainty.

The stronger interpretation is:

- MPC should not only execute a task after uncertainty is resolved.
- MPC itself should choose an action whose purpose is to reduce uncertainty.

### Autonomous uncertainty reduction

For a scenario in which a user asks the robot to find and deliver a particular object, the implemented framework should let the robot reduce uncertainty on its own and progress until it locates the target correctly.

This feedback strengthens the contribution from a sequential pipeline into an integrated information-seeking control loop.

## 8.3 Professor Lee’s response

Professor Lee replied that integrating the phases could produce a cleaner framework and that Professor Park’s proposed directions were good. He emphasized the limited time before the ICRA deadline and the need to increase development speed.

## 8.4 User’s response

The user replied that the research would be concretized around:

- a Scene Graph that reflects uncertainty; and
- MPC actions that reduce uncertainty.

---

# 9. Current final research direction

> Hardware and method details in `docs/FINAL_RESEARCH_SPEC.md` are authoritative. The fixed embodiment is UR10e + OnRobot RG6 + wrist-mounted Zivid 2.

## 9.1 Research problem

A language-guided manipulation robot operates under partial observability. The user asks for a particular object, but the robot cannot confidently determine:

- which visible object is the target;
- whether the target is inside, outside, behind, near, or occluded by a container or object;
- which action will reveal the missing information with the lowest effort and risk.

A direct VLM-to-action system may execute a wrong pick or unnecessary manipulation. A deterministic Scene Graph may hide uncertainty by committing to one relation. A conventional MPC controller may generate a safe trajectory but does not know that its semantic goal is uncertain.

The proposed system must connect semantic uncertainty to robot action.

## 9.2 Current research question

Can a robot represent calibrated uncertainty over both target objects and spatial relations in a dynamic Scene Graph, then use MPC to choose actions that reduce expected task failure before final object retrieval?

## 9.3 Current hypothesis

Compared with direct execution, deterministic Scene Graph planning, and heuristic re-observation, an uncertainty-aware relational Scene Graph combined with task-risk-aware information-seeking MPC will:

- increase object-retrieval task success;
- decrease wrong-object picks and premature actions;
- reduce unnecessary re-observation or manipulation actions;
- preserve safety and execution efficiency.

## 9.4 Current main contribution

The strongest practical contribution is not the basket scenario itself. It is the following connection:

Target and relation uncertainty -> uncertainty-aware Scene Graph -> expected task-risk reduction -> MPC action -> new observation -> graph update -> final retrieval.

---

# 10. Proposed novelty and its boundaries

## 10.1 Novelty 1 - task-conditioned uncertainty in graph nodes and edges

The Scene Graph should store uncertainty for both object nodes and relation edges.

Example node beliefs:

- Probability that object A is the requested target.
- Probability that object B is a distractor.
- Existence confidence.
- Pose or location uncertainty.
- Visibility or occlusion confidence.

Example relation beliefs:

- Probability that target is inside basket.
- Probability that target is outside basket.
- Probability that target is behind basket.
- Probability that target is occluded by another object.
- Probability that object is near the container.

The graph should be task-conditioned. It does not need to model every possible relation in the room. It should prioritize relations needed for the user’s instruction and the next manipulation decision.

## 10.2 Novelty 2 - calibrated uncertainty rather than raw VLM confidence

Raw VLM confidence or a verbal statement such as “I am 80 percent sure” is not enough. The system should derive a measurable and reproducible uncertainty estimate.

Candidate practical methods include:

- Temperature scaling on a held-out calibration set.
- Ensemble disagreement.
- Multi-view consistency.
- Dirichlet evidence accumulation.
- Bayesian belief update across observations.
- Conformal prediction or prediction sets if feasible.

This is most likely a supporting contribution rather than the entire paper. It is needed to prevent the controller from optimizing meaningless confidence values.

## 10.3 Novelty 3 - task-risk-aware information-seeking MPC

The strongest recommended controller contribution is to make MPC choose actions that reduce the probability of a wrong or failed task, not only reduce generic entropy.

A conceptual objective is:

J = task cost + expected future graph uncertainty + wrong-action risk + collision risk + motion or time cost.

Candidate action types:

- Move the wrist camera to another viewpoint.
- Move the arm without grasping to reveal the scene.
- Remove a lightweight cover.
- Move an occluder.
- Open a lid if the setup is reliable.
- Execute the final grasp.

The action should be selected because it is expected to improve task-relevant belief and reduce failure risk.

## 10.4 Novelty 4 - integrated closed loop

Do not implement two disconnected systems where one module first finishes perception and another later performs manipulation.

The intended loop is:

1. Observe.
2. Build or update target-conditioned graph belief.
3. Estimate target and relation uncertainty.
4. Evaluate candidate actions.
5. MPC selects a control or information-gathering action.
6. Execute the first action.
7. Obtain a new observation.
8. Update graph belief.
9. Repeat until confidence and risk meet the execution condition.
10. Retrieve the target.

## 10.5 Required novelty - action-conditioned future belief prediction

The proposed controller must predict how each candidate action changes future task belief before execution. Otherwise, it is a heuristic or one-step baseline rather than the proposed belief-space MPC. Candidate implementations include:

- geometry-based visibility and observation-likelihood prediction;
- simulated camera rendering in Isaac Sim;
- a learned action-conditioned Scene Graph transition or observation model;
- a calibrated approximate posterior model.

The first implementation can use known simulator likelihoods, but the final paper must include an explicit action-conditioned future-belief model.

## 10.6 Claims that must not be used as the only novelty

- Using a VLM.
- Using a Scene Graph.
- Using MPC.
- Combining VLM and MPC.
- Changing camera viewpoint.
- Opening a container.
- Using an off-the-shelf model for efficiency.
- Performing pick-and-place.

Each of these exists in prior research. The novelty must come from the specific uncertainty representation and its mathematical use in action selection.

---

# 11. Proposed technical architecture

## 11.1 Input

- Natural-language instruction.
- RGB or RGB-D image.
- Robot joint state.
- Camera pose.
- Current dynamic Scene Graph.
- Optional task history.

## 11.2 Perception and grounding layer

Suggested roles, not yet fixed model versions:

- VLM: instruction-conditioned object and relation reasoning.
- Open-vocabulary grounding model such as Grounding DINO: bounding boxes for named objects.
- Segmentation model such as a SAM-family model: masks if needed.
- RGB-D geometry: 3D location, container boundary, occlusion reasoning, and camera-to-world transform.

Do not make one VLM responsible for all localization, segmentation, calibration, and control if specialized components give more reliable outputs.

## 11.3 Scene Graph representation

Recommended graph state:

Object node:

- `id`
- `label_distribution`
- `target_probability`
- `position_mean`
- `position_covariance`
- `bounding_box`
- `mask_reference`
- `visible`
- `occluded_probability`
- `container_state`
- `graspable`
- `last_observed_time`
- `source_view_id`

Relation edge:

- `source_id`
- `target_id`
- `relation_type`
- `probability`
- `uncertainty`
- `geometric_evidence`
- `semantic_evidence`
- `last_updated_time`

Task state:

- `instruction`
- `target_description`
- `active_subgoal`
- `completion_probability`
- `risk_threshold`
- `observation_budget`

## 11.4 Uncertainty calculation

A first implementable version can combine:

- Grounding confidence.
- VLM relation prediction probability or repeated-sampling agreement.
- Geometric relation score from RGB-D.
- Multi-view consistency.
- Calibration parameters learned on a small held-out scenario set.

The output should be a normalized probability distribution for mutually exclusive relations where appropriate.

Example:

Before re-observation:

- inside: 0.48
- behind: 0.37
- outside: 0.15

After re-observation:

- inside: 0.91
- behind: 0.06
- outside: 0.03

The system should log both the probabilities and the evidence used to update them.

## 11.5 MPC layer

The MPC state may include:

- robot joints and velocities;
- end-effector pose;
- camera pose;
- object and container geometric state;
- target and relation belief vector;
- active sub-goal;
- collision geometry.

Candidate cost terms:

- Goal progress.
- Expected task success.
- Expected future target uncertainty.
- Expected future relation uncertainty.
- Wrong-action risk.
- Collision and joint-limit penalties.
- Motion effort.
- Time or path length.
- Number of information-gathering actions.

The initial implementation may use a sampling-based or hierarchical controller available in Isaac Sim. The exact solver should be selected after validating the UR10e, RG6, wrist-camera, collision, and simulation interfaces. The proposed method must eventually predict action-conditioned future belief; a geometric controller alone is not the paper contribution.

## 11.6 Decision threshold

The robot should execute the final grasp only when:

- target probability exceeds a threshold;
- required relation probability exceeds a threshold;
- collision and reachability checks pass;
- predicted wrong-action risk is below a threshold.

Otherwise it should select a re-observation or information-gathering action.

---

# 11A. Final embodiment decision

The current Isaac Sim and planned physical embodiment is fixed as Universal Robots UR10e, OnRobot RG6 gripper, and a wrist-mounted Zivid 2 3D/RGB-D camera. Hardware-specific interfaces should be modular for engineering quality, but Codex must not interpret modularity as permission to change the final embodiment.

# 12. Simulation implementation plan in Isaac Sim

## 12.1 Phase 0 - repository and environment audit

Before building features, Codex should:

- List the current repository files.
- Identify the Isaac Sim version and installation path.
- Verify the Python environment.
- Run `nvidia-smi`.
- Verify CUDA and driver compatibility.
- Identify official or available UR10e URDF/USD assets, RG6 URDF/CAD/mesh assets, wrist-camera mounting geometry, and joint/frame names.
- Confirm whether the robot currently loads correctly.
- Confirm whether articulation control works.
- Confirm camera creation, image capture, and headless execution.
- Create a Git checkpoint.

## 12.2 Phase 1 - minimal deterministic tabletop scene

Build the simplest reliable scene:

- Ground plane.
- Table.
- UR10e with OnRobot RG6.
- Wrist-mounted RGB-D camera configured to approximate Zivid 2 geometry and sensing.
- One open basket or tray.
- Two or three simple objects.
- One target and one distractor.

Initial goals:

- Robot loads with correct scale and joint limits.
- Camera images are saved.
- Object poses are accessible.
- A scripted pick trajectory can be executed.
- The scene can run in GUI and headless modes.

## 12.3 Phase 2 - relation-generation scene variants

Create controlled scene generators for:

- target inside container;
- target outside container;
- target near boundary;
- target behind container;
- target partially occluded;
- target absent from expected container;
- visually similar distractors.

Randomize within bounded ranges:

- object position;
- orientation;
- distractor count;
- occlusion percentage;
- lighting;
- camera pose;
- container pose.

Do not randomize everything at once. Validate each factor separately.

## 12.4 Phase 3 - perception and graph stub

Before integrating a heavy VLM, build a ground-truth or rule-based stub that produces the same Scene Graph interface. This allows the control loop to be tested independently.

Required outputs:

- object node list;
- relation belief distribution;
- target probability;
- uncertainty score;
- graph JSON file;
- image and camera metadata.

Then replace the stub with actual grounding and VLM components.

## 12.5 Phase 4 - active viewpoint actions

Define a finite set of safe camera or arm viewpoints around the table. The first active-perception controller can select among these viewpoints.

Requirements:

- Each viewpoint must be reachable.
- Camera pose must be logged.
- New observation must update the graph.
- The system must compare uncertainty before and after movement.
- A failed or occluded view must not crash the episode.

## 12.6 Phase 5 - lightweight cover interaction

Use a removable lightweight cover before implementing a hinge lid.

- Cover is graspable.
- Lift or move cover to a fixed safe location.
- Re-observe container interior.
- Update inside or absent relation.
- Continue search or grasp.

A hinge lid should be treated as an extension only after the basic closed loop is stable.

## 12.7 Phase 6 - information-seeking MPC

Move from fixed or greedy viewpoint selection to the proposed MPC objective.

The controller should compare candidate actions using:

- predicted reduction in task-relevant uncertainty;
- task progress;
- movement cost;
- collision risk;
- wrong-action risk.

Execute only the first action and replan after the new observation.

## 12.8 Phase 7 - real-robot transfer

The real-robot setup should reproduce the simplest validated simulation scenarios.

Priority real-world scenarios:

1. Open-container relation ambiguity plus active viewpoint change.
2. Lightweight cover removal plus re-observation.

Keep object geometry and lighting controlled initially. Record failures and domain gaps rather than hiding them.

---

# 13. Baselines and ablations

## 13.1 Main baselines

### Direct VLM plus execution

The system chooses the most likely target or relation and acts immediately. This baseline measures the cost of ignoring uncertainty.

### Deterministic Scene Graph plus MPC

The graph keeps one hard relation label without a probability distribution. This isolates the value of probabilistic graph belief.

### Uncertainty-aware graph plus fixed re-observation

The system knows it is uncertain but always uses a predetermined viewpoint. This isolates the value of intelligent action selection.

### Uncertainty-aware graph plus greedy viewpoint

Choose the locally best immediate view without a multi-step or control cost. This isolates the value of MPC.

### MPC without uncertainty or task-risk cost

Use the same controller but remove the uncertainty term. This isolates the proposed objective.

### Proposed full method

Use calibrated target and relation uncertainty, graph updates, and task-risk-aware information-seeking MPC.

## 13.2 Essential ablations

- Remove relation-edge uncertainty.
- Remove target-node uncertainty.
- Remove calibration.
- Remove the uncertainty term from MPC.
- Remove the wrong-action risk term.
- Use only camera re-observation and disable physical interaction.
- Use one observation instead of the closed loop.

## 13.3 Related system comparisons

Full reproduction of large external systems may not be feasible. The paper can compare with implementable strategy-level baselines inspired by direct VLM execution, heuristic active perception, deterministic graph planning, and uncertainty-triggered human clarification. Claims must clearly distinguish a reproduced baseline from a conceptual comparison.

---

# 14. Evaluation metrics

## 14.1 Primary metric 1 - Task Success Rate

Definition:

The fraction of trials in which the robot retrieves the correct requested object and completes the defined terminal condition.

This is the most important robotics metric.

## 14.2 Primary metric 2 - Wrong Commitment Rate

Count or rate of irreversible or task-damaging commitments such as:

- picking the wrong object;
- grasping an empty or disproven location;
- opening or moving an irrelevant object when the decision should have been deferred;
- committing before target or relation belief is sufficiently reliable;
- repeating an action after negative evidence disproves the underlying hypothesis.

Report both episode-level wrong-commitment frequency and the number of wrong commitments per successful retrieval when possible.

## 14.3 Primary metric 3 - Total Cost to Successful Retrieval

Use one predeclared weighted cost that can include:

- elapsed time;
- robot path length or control effort;
- number of re-observations;
- number and cost of environment interactions;
- failed or wrong commitments.

Entropy or information gain should be reported as a diagnostic secondary metric, not as the primary task outcome.

## 14.4 Secondary metrics

- Expected Calibration Error.
- Brier score.
- Target grounding accuracy.
- Relation classification accuracy.
- Scene Graph entropy before and after action.
- Collision rate.
- Planning time per step.
- Total execution time.
- Path length.
- Replanning count.
- Graph update accuracy.
- Success under occlusion severity.
- Success under distractor count.

## 14.5 Statistical reporting

- Use multiple randomized episodes.
- Use at least three random seeds where training or stochastic sampling is involved.
- Report confidence intervals for real-robot success rates where possible.
- Preserve raw episode-level CSV or JSON results.
- Record failures and qualitative examples.

---

# 15. Experimental scale and data strategy

## 15.1 Dataset strategy

The project does not need to create a new large dataset. The main validation can use:

- randomized Isaac Sim episodes generated by the scenario code;
- one or two highly relevant external perception or relation datasets only if they help calibrate or validate a component;
- real-robot trials in Professor Park’s laboratory.

Do not expand to many unrelated benchmarks merely to increase the number of datasets.

## 15.2 Simulation episode target

A practical target is several hundred randomized episodes across the main scenarios, subject to compute and time.

Recommended structure:

- Open container and active viewpoint scenario: main simulation benchmark.
- Covered container or removable cover: second benchmark.
- Occluder removal: optional third scenario or supplementary evaluation.

## 15.3 Real-robot trial target

For each main method and baseline, use repeated trials. A realistic minimum is 10 to 15 trials per method-scenario combination, with 20 preferred when time permits. The final number should reflect available lab time and statistical meaning.

---

# 16. Timeline and current urgency

The research plan worked backward from the September 15 submission deadline.

Major planned stages were:

- May: research direction, literature review, problem definition, and first meeting.
- June: method structure, prototype, Scene Graph representation, and VLM-MPC interface.
- Early July: scope, uncertainty, baseline, and scenario selection.
- Mid to late July: first working simulation, main experiments, and paper skeleton.
- Early August: experiment stabilization, ablations, and result organization.
- Mid to late August: result freeze, figures, tables, full draft.
- Early September: revision, formatting, limitations, supplementary material, and final checks.
- September 14: intended submission buffer.
- September 15: deadline.

Professor Lee stressed that substantial results should be ready by the end of August because the user is writing an ICRA paper for the first time and will need time for drafting and revision.

---

# 17. Recommended repository structure

A repository structure that supports desktop, laptop, and server work:

```text
efficient-robotics/
├─ AGENTS.md
├─ README.md
├─ docs/
│  ├─ PROJECT_CONTEXT.md
│  ├─ DECISIONS.md
│  ├─ MEETING_NOTES/
│  ├─ EXPERIMENT_PROTOCOL.md
│  └─ PAPER_CLAIMS.md
├─ assets/
│  ├─ robots/
│  │  └─ ur10e/
│  ├─ grippers/
│  │  └─ onrobot_rg6/
│  ├─ sensors/
│  │  └─ zivid2/
│  ├─ containers/
│  └─ objects/
├─ configs/
│  ├─ sim/
│  ├─ perception/
│  ├─ scene_graph/
│  ├─ mpc/
│  └─ experiments/
├─ src/
│  ├─ sim/
│  ├─ robot/
│  ├─ perception/
│  ├─ scene_graph/
│  ├─ uncertainty/
│  ├─ mpc/
│  ├─ integration/
│  └─ evaluation/
├─ scripts/
│  ├─ setup/
│  ├─ run_gui.py
│  ├─ run_headless.py
│  ├─ generate_scenarios.py
│  ├─ calibrate_uncertainty.py
│  └─ evaluate.py
├─ tests/
├─ experiments/
│  ├─ manifests/
│  └─ summaries/
├─ results/
│  ├─ raw/
│  ├─ processed/
│  ├─ figures/
│  └─ videos/
└─ paper/
```

The user previously disliked arbitrary `logs` and `results` directories in older unrelated projects, but for this robotics research reproducibility requires a clear experiment-output location. The final repository naming should follow the user’s current preference; the important requirement is that raw outputs are not mixed with source code and are not accidentally committed when too large.

---

# 18. Logging and reproducibility requirements

Every episode should save:

- Git commit hash.
- Machine identifier.
- Isaac Sim version.
- Python and package versions.
- GPU information.
- Scenario seed.
- Object and camera poses.
- Natural-language instruction.
- Grounding outputs.
- Scene Graph before and after each action.
- Target and relation probability distributions.
- Selected action and candidate costs.
- MPC planning time.
- Collision and reachability status.
- Episode success or failure reason.
- Video or image sequence.

Use a manifest file for each experiment batch. Do not rely on terminal scrollback as the only record.

---

# 19. Hard constraints and decisions for Codex

## 19.1 Hard constraints

- Final simulator must use UR10e with OnRobot RG6 and a wrist-mounted Zivid 2-style RGB-D sensor.
- Isaac Sim is the simulator for this project.
- The project is not a generic pick-and-place tutorial.
- The core claim must involve target or relation uncertainty affecting robot action.
- The Scene Graph must influence MPC decisions, not merely display information.
- The robot should reduce uncertainty through action rather than always ask a human.
- Real-robot validation is an important goal.
- Do not create a large new dataset unless the research direction explicitly changes.
- Use a narrow and complete experimental scope rather than many unfinished benchmarks.

## 19.2 Current decisions

- Use UR10e, OnRobot RG6, and a wrist-mounted Zivid 2-style RGB-D sensor in Isaac Sim and planned real-robot validation.
- Start with multi-hypothesis open-container retrieval and active wrist-camera viewpoint selection.
- Add removable-cover search with explicit negative evidence.
- Add occluder manipulation when the core loop is stable.
- Store calibrated target and relation beliefs in graph nodes and edges.
- Require action-conditioned future-belief prediction for the proposed controller.
- Use expected task loss, wrong-commitment risk, execution risk, and motion cost in planning.
- Measure task success, wrong commitment, and total cost to successful retrieval as primary outcomes.

## 19.3 Unresolved decisions

- Exact VLM and model version.
- Exact uncertainty calibration algorithm.
- Exact MPC solver.
- Whether the required action-conditioned belief model should be geometry-based, simulator-rendered, learned, or hybrid.
- Exact Zivid 2 intrinsics, mounting transform, depth-noise approximation, and synchronization settings.
- Exact RG6 asset source, joint configuration, and grasp-frame calibration in Isaac Sim.
- Final paper title.

Codex must not silently treat unresolved items as settled.

---

# 20. Immediate implementation sequence for Codex CLI

## Step 1 - inspect, do not edit

Prompt example:

“Read AGENTS.md and docs/PROJECT_CONTEXT.md. Inspect the repository and Isaac Sim setup. Do not modify files yet. Report the existing robot assets, simulator entry points, Python environments, missing dependencies, and the smallest runnable scene.”

## Step 2 - create a plan and checkpoints

Prompt example:

“Create an implementation plan for a minimal UR10e + RG6 + wrist RGB-D open-container active-viewpoint simulation. Identify files to add or change, tests, expected outputs, and rollback checkpoints. Do not implement until the plan is reviewed.”

## Step 3 - build the minimal scene

Prompt example:

“Implement the smallest deterministic Isaac Sim scene with UR10e, OnRobot RG6, a wrist RGB-D camera, a table, an open container, and two objects. Add a script that saves one RGB-D observation and scene metadata. Run it and report exact commands and outputs.”

## Step 4 - add scenario generation

Prompt example:

“Add controlled inside, outside, near, behind, and partial-occlusion variants. Save the scenario configuration and ground-truth relations for each episode.”

## Step 5 - add graph interface with a stub

Prompt example:

“Implement the Scene Graph JSON schema from PROJECT_CONTEXT.md using ground truth first. Add tests for graph update after viewpoint changes and object movement.”

## Step 6 - add uncertainty and active observation

Prompt example:

“Implement a baseline belief distribution and a fixed set of candidate viewpoints. Select a viewpoint using a simple uncertainty-reduction heuristic. Log uncertainty before and after the action.”

## Step 7 - replace heuristic with MPC

Prompt example:

“Add the task-risk-aware MPC objective. Compare direct execution, fixed viewpoint, greedy viewpoint, MPC without uncertainty cost, and the full method.”

## Step 8 - integrate actual perception models

Prompt example:

“Replace the ground-truth perception stub with the selected grounding and VLM components while preserving the graph interface and test suite. Add calibration and multi-view updates.”

---

# 21. Multi-machine workflow and Codex record handling

## 21.1 What Codex stores locally

Codex CLI stores local state under `CODEX_HOME`, which defaults to `~/.codex`. When history persistence is enabled, local transcripts are stored in files such as `history.jsonl`. A saved local session can be resumed on the same machine with Codex resume commands.

This local history is not a reliable cross-machine project database. The desktop, laptop, and server can each have separate local Codex histories.

## 21.2 What should be synchronized

The durable shared record should be the Git repository, not the chat transcript.

Commit and push:

- Source code.
- Configuration files.
- `AGENTS.md`.
- This project context document.
- Decision log.
- Experiment protocol.
- Lightweight result summaries.
- Environment lock files.

Do not push secrets, access tokens, large checkpoints, or all raw videos directly to a normal Git repository.

## 21.3 Recommended Git workflow

- Use one remote repository accessible from desktop, laptop, and server.
- Pull before beginning work on a machine.
- Create a branch for a coherent task.
- Commit before asking Codex to perform a large edit.
- Commit after a verified working milestone.
- Push before moving to another machine.
- Pull and verify the commit hash on the new machine.
- Use Git LFS, NAS, or an agreed storage location for large assets and checkpoints.

## 21.4 Recommended documentation workflow

Keep these files in the repository:

- `AGENTS.md`: short instructions automatically read by Codex.
- `docs/PROJECT_CONTEXT.md`: detailed research history and architecture.
- `docs/DECISIONS.md`: date, decision, reason, alternatives, and consequences.
- `docs/EXPERIMENT_PROTOCOL.md`: exact episode definitions and metrics.
- `docs/RUNBOOK.md`: setup and run commands for desktop and server.
- `docs/STATUS.md`: current working state and next task.

Before ending a session, ask Codex to update `STATUS.md` with:

- completed work;
- exact commands used;
- files changed;
- tests passed or failed;
- next step;
- known problems;
- current Git commit.

## 21.5 Using AGENTS.md

Codex automatically reads `AGENTS.md` from the project hierarchy at the start of a run. Keep it short and stable. The full project history should remain in this detailed context document because Codex has a default size limit for automatically loaded project instructions.

## 21.6 Optional local history backup

If the user wants to preserve CLI transcripts, back up the relevant `~/.codex` state from each machine. Do not assume that copying only the repository also copies local chat history. Do not commit authentication files such as `auth.json`.

---

# 22. Final current paper story

The paper should tell one focused story.

A user asks a robot to retrieve an object. The object may be ambiguous, partially hidden, or located relative to a basket or container. The VLM and grounding components generate uncertain object and relation hypotheses. The system stores these as a task-conditioned probabilistic Scene Graph. The MPC controller evaluates robot motions and interactions based on expected reduction in task-relevant uncertainty, wrong-action risk, collision risk, and effort. The robot changes viewpoint, removes a cover, or moves an occluder when necessary. Each action creates a new observation, updates the graph, and changes the next MPC decision. The robot grasps only after uncertainty and risk are sufficiently low.

The paper must prove that this loop improves the correct-object retrieval rate and reduces wrong actions compared with direct execution, deterministic graphs, fixed re-observation, greedy re-observation, and MPC without uncertainty cost.

The scenario is a testbed. The contribution is the uncertainty representation and its use in information-seeking MPC.

---

# 23. Source and provenance notes

The detailed reconstruction above was assembled from:

- The “Efficient VLM-MPC Framework for Robotic Manipulation with Scene Graph-Based World State Tracking” presentation.
- The Korean VLM-MPC presentation outline.
- The “Grounded Scene Graphs and Planning-Aware Perception for Neural MPC” internal research plan.
- The detailed ICRA submission schedules.
- The July 6, 2026 meeting transcript.
- The scenario candidate Word documents.
- The “Scenario Selection and Initial Experimental Plan” PDF.
- Email screenshots containing Professor Park’s post-scenario feedback, Professor Lee’s response, and the user’s reply.
- Prior chat records about RT-2/VLA exploration, Isaac Sim, server setup, uncertainty, metrics, hardware decisions, and ICRA positioning.

When an exact professor quotation is needed for a paper or meeting record, return to the original transcript or email screenshot rather than quoting this reconstructed document.
