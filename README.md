# Risk-Aware Active Perception for Robotic Retrieval

Research code for language-guided object retrieval under partial observability. The system combines open-vocabulary perception, a probabilistic Scene Graph, and task-risk-aware belief-space planning. Instead of committing to the first plausible object, the robot can acquire another wrist-camera observation, remove a cover, grasp a candidate, or defer when the remaining uncertainty makes manipulation unsafe.

The central question is not only *which object matches the instruction*, but also *whether the target is inside or outside the container, occluded, or absent*, and which action should be executed next to reduce task risk. The method maintains these coupled uncertainties across observations instead of treating a VLM score as a grasp probability.

## Method overview

```text
language instruction + wrist RGB-D observation
  -> GroundingDINO candidate proposals
  -> SAM2.1 candidate masks
  -> Qwen target-identity and relation evidence
  -> RGB-D localization and cross-view candidate association
  -> probabilistic Scene Graph belief update
  -> action-conditioned future-belief prediction
  -> finite-horizon task-cost comparison
  -> execute one action
  -> observe and replan
```

All retained detector proposals, confidence scores, masks, and candidate-selection records are saved. Qwen operates on anonymous candidate IDs and does not receive simulator object identities or ground truth. Its raw choice logits are treated as evidence, not as calibrated probabilities.

## System

| Component | Configuration |
| --- | --- |
| Simulator | NVIDIA Isaac Sim 6.0.1 |
| Robot arm | Universal Robots UR10e |
| Gripper | OnRobot RG6 |
| Camera | Wrist-mounted RGB-D camera; Zivid 2 geometry approximated in simulation |
| Vision-language model | `Qwen/Qwen3-VL-8B-Instruct` |
| Detection | `IDEA-Research/grounding-dino-base` |
| Segmentation | `facebook/sam2.1-hiera-large` |
| Planner | Discrete receding-horizon belief-tree planner |
| Model adaptation | None; pretrained inference with separately fitted calibration models |

Exact model revisions are listed in [Model Setup](docs/MODEL_SETUP.md). Foundation-model weights and generated experiment artifacts are not stored in the repository.

## Module interfaces

| Stage | Input | Output |
| --- | --- | --- |
| RGB-D observation | camera pose and semantic view request | RGB image, metric depth, intrinsics, camera-to-world transform |
| Candidate detection | RGB image and open-vocabulary concepts | all retained boxes, phrases, detector scores, and proposal metadata |
| Candidate segmentation | RGB image and retained boxes | per-candidate masks, mask overlays, and candidate crops |
| VLM evidence | instruction, anonymous candidates, crops or masks, closed relation vocabulary | target-match logits, relation-choice logits, selected candidate, prompt and model provenance |
| RGB-D localization | candidate mask, depth, intrinsics, camera pose | robust 3-D candidate center and extent in world coordinates |
| Candidate association | current 3-D estimates and persistent tracks | cross-view candidate-to-track assignments and association distances |
| Scene Graph update | calibrated observation evidence and previous belief | normalized joint belief over candidate identity, relation, and target absence |
| Belief-space planner | current belief, task state, feasible actions, calibrated observation and execution models | expected cost for every action and the selected first action |
| Physical execution | selected semantic action and safety constraints | execution status, measured trajectory, contact diagnostics, next observation, or explicit failure reason |

The implemented action set is:

```text
viewpoint_right
viewpoint_close_high
remove_cover
grasp(candidate, inside|outside)
defer
```

The planner evaluates these actions in one value table. There is no fixed confidence threshold for grasping. Each decision compares expected wrong-commitment loss, execution risk, sensing or interaction cost, and noncompletion cost. Only the first action is executed before belief update and replanning.

## Evaluation scenarios

The simulation protocol covers five scenario families:

1. **Visible/Open** — grasp a clearly visible target without an unnecessary observation.
2. **Partially Occluded** — select a reachable right or close-high view before grasping.
3. **Covered Container** — remove the cover, observe the changed scene, and replan.
4. **Ambiguous Inside/Outside** — jointly update target identity and container-membership belief for similar candidates.
5. **Target Absent** — use negative evidence and defer instead of committing to a distractor.

Calibration and testing use disjoint episode sets. No VLM fine-tuning, LoRA, or manually annotated training dataset is used. Simulator ground truth is restricted to scene generation, calibration labels, and post-run evaluation; it is not exposed to the online perception or root-action decision.

## Setup

Isaac Sim, Qwen, and grounding dependencies are kept in separate environments. Do not install perception packages into the Isaac Sim environment.

```bash
python3 -m venv /path/to/efficient-robotics-vlm
source /path/to/efficient-robotics-vlm/bin/activate
pip install --upgrade pip
pip install -r requirements/vlm-qwen3-vl.txt
```

Grounding and segmentation installation, checkpoint revisions, and environment details are documented in [Model Setup](docs/MODEL_SETUP.md). Headless Isaac Sim configuration is documented in [Simulation Setup](docs/SIMULATION_SETUP.md).

## Repository layout

```text
configs/       simulation, perception, Scene Graph, planner, and hardware configs
docs/          method, interface, protocol, setup, and transfer documentation
examples/      example Scene Graph records
requirements/  isolated learned-perception dependencies
scripts/       perception, belief update, planning, simulation, and evaluation code
tests/         CPU regression and protocol tests
```

The main implementation files are indexed in [Code Map](docs/CODE_MAP.md). The action definitions and safety contract are in [Action Space](docs/ACTION_SPACE_SPEC.md), the experimental split, comparisons, and metrics are in [Experiment Protocol](docs/EXPERIMENT_PROTOCOL.md), and the distinction from closely related active-perception systems is summarized in [Related Work](docs/RELATED_WORK.md).

Generated RGB-D observations, videos, model weights, inference caches, and experiment outputs are excluded from Git.
