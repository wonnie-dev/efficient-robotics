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

Set local paths before running a GPU job:

```bash
export PHYSICAL_GPU=0
export EFFICIENT_ROBOTICS_ISAACSIM_VENV=/path/to/isaacsim_venv
export EFFICIENT_ROBOTICS_VLM_VENV=/path/to/vlm_venv
export EFFICIENT_ROBOTICS_QWEN_MODEL=/path/to/Qwen3-VL-8B-Instruct
```

Only one model instance and one physical GPU are used per job. Inside PyTorch and PhysX, the selected physical GPU is exposed as `cuda:0`.

## Minimal commands

Run the CPU-only Scene Graph and belief-planner contract:

```bash
python scripts/run_cover_search_scene_graph_mpc_integration.py
```

Run one cached perception request:

```bash
PHYSICAL_GPU=0 bash scripts/run_perception_demo.sh path/to/input.json
```

Run one headless simulation episode:

```bash
PHYSICAL_GPU=0 bash scripts/run_simulation_episode.sh 0
```

Replay the cached MPC calibration without launching Isaac Sim:

```bash
bash scripts/replay_cached_mpc.sh
```

The real-robot smoke script performs configuration checks only. It does not command the robot:

```bash
bash scripts/real_robot_smoke_test.sh
```

## Current status

- Qwen target-temperature calibration and RGB-D relation calibration are frozen on calibration episodes.
- Closed-loop simulation has connected cover removal, new perception, belief update, replanning, and contact-based RG6 grasp in one Isaac Sim process.
- The action-conditioned cover observation model is still being calibrated.
- Task-cost weights and the grasp commitment gate are not frozen.
- Reserved final-test seeds `200-209` have not been opened.
- Real-robot validation has not started.
- Lid, fingertip, camera, table, and robot-frame measurements must be replaced with lab measurements before transfer claims are made.

## Results to date

These are calibration and development results, not final paper results.

| Check | Result |
| --- | ---: |
| Objective RGB-D membership calibration | `127/128` |
| Objective camera-relative behind calibration | `63/63` |
| Objective reference-occlusion calibration | `56/63` |
| Scene-conditioned action choice, held-out development episodes | `11/11` |
| Best fixed-action policy on the same episodes | `5/11` |
| Independent empty-container physical episodes | `10` |
| Full cover removal, learned re-perception, replanning, and physical grasp | completed on seed 188 |
| Repository tests | `250/250` |

See [Research Progress](docs/RESEARCH_PROGRESS.md) for scope, failure history, and the remaining path to final evaluation.

## Documentation

- [Project overview](docs/PROJECT_OVERVIEW.md)
- [Action space](docs/ACTION_SPACE_SPEC.md)
- [Model setup](docs/MODEL_SETUP.md)
- [Simulation setup](docs/SIMULATION_SETUP.md)
- [Real-robot transfer](docs/REAL_ROBOT_TRANSFER.md)
- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md)
- [Current limitations](docs/CURRENT_LIMITATIONS.md)
- [Research progress](docs/RESEARCH_PROGRESS.md)
- [Code map](docs/CODE_MAP.md)
- [VLM interface](docs/VLM_INTERFACE.md)

Generated observations, videos, model weights, caches, and experiment outputs are excluded from Git.
