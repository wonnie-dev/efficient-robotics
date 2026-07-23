# AGENTS.md

## Project identity

This repository is the `efficient-robotics` ICRA 2027 research project.

## Required context

Before planning or editing, read `docs/PROJECT_CONTEXT.md` (or `efficient_robotics_project_context.md` if it has not yet been moved into `docs/`). Treat its hard constraints and current decisions as authoritative.

## Core research direction

- Simulator: NVIDIA Isaac Sim.
- Provisional robot stack: UR10e + OnRobot RG6 + wrist-mounted Zivid 2 RGB-D camera, based on Professor Shinkyu Park's corresponding-author paper. Keep the stack replaceable until availability for this project is confirmed.
- Research focus: task-conditioned target and spatial-relation uncertainty in a dynamic Scene Graph.
- Control focus: MPC selects information-gathering actions that reduce wrong-action or task-failure risk before final object retrieval.
- Starting scenario: open container plus active viewpoint change, then partial occlusion, then removable cover; hinge lid only after the core loop is stable.

## Working rules

- Inspect the repository and environment before editing.
- Keep simulation and real-robot platforms aligned. Do not assume or substitute a robot model before the Professor Park lab platform is confirmed.
- Do not reduce the project to a generic VLM + MPC demo.
- Preserve reproducibility: configs, seeds, commands, metrics, graphs, videos, and failure reasons.
- Use Git checkpoints before and after major changes.
- Do not commit credentials, tokens, or large raw assets.
- Keep `docs/STATUS.md` and `docs/DECISIONS.md` updated.
- Ask before changing a paper claim, metric definition, baseline, robot, simulator, or scenario scope.
