# AGENTS.md

## Project identity

This repository is the `efficient-robotics` research project targeting an ICRA 2027 submission.

## Mandatory reading order

Before planning, editing, or running experiments, read these files completely:

1. `efficient_robotics_final_project_plan_KO.docx`
2. `docs/FINAL_RESEARCH_SPEC.md`
3. `docs/PROJECT_CONTEXT.md`
4. `docs/LITERATURE_NOVELTY_AUDIT.md`
5. `docs/STATUS.md` and `docs/DECISIONS.md`, if present

`efficient_robotics_final_project_plan_KO.docx` is the authoritative and most
recent project specification. `docs/FINAL_RESEARCH_SPEC.md` is a secondary
implementation-oriented summary and applies only where it does not conflict
with the Word document. If any repository document or older note conflicts
with the Word document, follow the Word document, record the conflict in
`docs/DECISIONS.md`, and synchronize the affected Markdown documentation. Do
not silently fall back to an older specification if the Word document is
missing or unreadable.

## Fixed simulation embodiment

- Simulator: NVIDIA Isaac Sim.
- Manipulator: Universal Robots UR10e.
- Gripper: OnRobot RG6.
- Camera: wrist-mounted Zivid 2 3D/RGB-D camera.
- The implementation should remain modular, but do not substitute a different final embodiment without explicit user approval.
- A temporary proxy gripper or camera may be used only for early debugging and must be clearly labeled, isolated behind configuration, and replaced before final evaluation.

## Core research direction

The project is not a generic VLM, Scene Graph, or MPC integration demo.

The proposed method must:

- maintain calibrated task-conditioned beliefs over target objects and action-relevant spatial relations;
- represent relation uncertainty on Scene Graph edges such as `inside`, `behind`, `occluded_by`, and `covered_by`;
- predict how candidate actions change future task beliefs;
- select viewpoint, uncovering, occluder-removal, or grasp actions by minimizing expected task loss, wrong commitment, execution risk, and motion cost;
- update beliefs from both positive and negative evidence, then replan in a closed loop.

Do not call a controller “belief-space MPC” unless it evaluates action-conditioned future belief or posterior outcomes over a horizon. A one-step entropy heuristic is a baseline, not the proposed method.

## Working rules

- Inspect the repository, Isaac Sim installation, robot assets, Python environment, GPU, and dependency versions before editing.
- Preserve reproducibility with configs, seeds, commands, logs, metrics, graphs, videos, and failure reasons.
- Use Git checkpoints before and after major changes.
- Keep hardware-specific code behind robot, gripper, camera, kinematics, and controller configuration interfaces.
- Do not integrate heavy VLM perception before the deterministic simulation, Scene Graph interface, belief update, and control loop work with ground truth.
- Do not commit credentials, tokens, private data, or large generated assets.
- Update `docs/STATUS.md` after each work session and `docs/DECISIONS.md` after each design change.
- Ask before changing the paper claim, core novelty, primary metrics, baselines, final embodiment, simulator, or scenario scope.
