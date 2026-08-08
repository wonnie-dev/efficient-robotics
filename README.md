# Efficient Robotics

Research code for language-guided object retrieval under partial observability. The robot maintains beliefs over target identity and spatial relations, chooses a camera or manipulation action, updates the Scene Graph from the next RGB-D observation, and replans before committing to a grasp.

The current paper target is ICRA 2027. The main scenario is a covered tabletop container with an uncertain target location.

## System

| Component | Configuration |
| --- | --- |
| Simulator | NVIDIA Isaac Sim 6.0.1 |
| Arm | Universal Robots UR10e |
| Gripper | OnRobot RG6 |
| Camera | Wrist-mounted RGB-D camera; Zivid 2 geometry is approximated in simulation |
| Target model | Qwen/Qwen3-VL-8B-Instruct, revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| Grounding | GroundingDINO-Base and SAM2.1-Large |
| Planner | Discrete receding-horizon belief-tree planner |
| Training | None; pretrained inference and held-out calibration only |

## Pipeline

```text
RGB-D observation
  -> open-vocabulary proposals and masks
  -> Qwen target ranking
  -> RGB-D relation evidence
  -> probabilistic Scene Graph update
  -> action-conditioned belief prediction
  -> task-risk-aware action selection
  -> viewpoint change, cover removal, or grasp
  -> new observation and replanning
```

The current action library contains `viewpoint_right`, `viewpoint_close_high`, `remove_cover`, candidate grasps, and `defer`. Viewpoints are semantic poses backed by fixed calibration poses. They are not continuous viewpoint optimization.

## Repository setup

Use separate environments for Isaac Sim and learned perception. Do not install perception packages into the Isaac Sim environment.

```bash
python3 -m venv .venv-vlm
source .venv-vlm/bin/activate
pip install -r requirements/vlm-qwen3-vl.txt
```

Grounding and segmentation use the pinned environment described in [Model Setup](docs/MODEL_SETUP.md). Model weights are not stored in this repository.

## Documentation

- [Project overview](docs/PROJECT_OVERVIEW.md)
- [Action space](docs/ACTION_SPACE_SPEC.md)
- [Model setup](docs/MODEL_SETUP.md)
- [Simulation setup](docs/SIMULATION_SETUP.md)
- [Real-robot transfer](docs/REAL_ROBOT_TRANSFER.md)
- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Current limitations](docs/CURRENT_LIMITATIONS.md)
- [Code map](docs/CODE_MAP.md)
- [VLM interface](docs/VLM_INTERFACE.md)

Generated observations, videos, model weights, caches, and experiment outputs are excluded from Git.
