# Simulation Setup

## Tested configuration

- NVIDIA Isaac Sim 6.0.1
- Universal Robots UR10e
- OnRobot RG6
- wrist-mounted RGB-D camera with a Zivid 2 approximation
- headless RTX rendering
- one physical NVIDIA GPU per job

The imported UR10e and RG6 assets are under `assets/robots/`. Scene geometry and observation poses are configured under `configs/sim/`.

## Environment

Use an existing Isaac Sim virtual environment. Do not install Qwen or
GroundingDINO packages into it. Model paths and compute-device assignments are
deployment-specific and must not be hard-coded in the repository.

The episode writes RGB, metric depth, camera metadata, object records, action requests, belief updates, physics diagnostics, and a summary result under `outputs/`. Generated outputs are ignored by Git.

## Physics checks

Contact-based evaluation does not attach an object to the gripper or copy its pose. A successful lift requires bilateral contact, bounded force and penetration, stable object-to-gripper motion, finite joint state, and no unexpected environment collision.

The current RG6 fingertip, friction, lid mass, handle geometry, and drive mapping are provisional simulation values. They are suitable for development but are not real-robot calibration.

## Reproducibility

Each run must record:

- Git commit and dirty-tree state;
- seed and scene variant;
- effective config files;
- Isaac Sim and model versions;
- GPU model and runtime device visibility;
- command line and runtime;
- observation, action, belief, and physics result paths;
- failure stage and reason.
