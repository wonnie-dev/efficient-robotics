# Code Map

## Entry points

| Task | Entry point |
| --- | --- |
| Headless scene and RGB-D capture | `scripts/run_simulation_episode.sh` |
| One Qwen perception request | `scripts/run_perception_demo.sh` |
| CPU Scene Graph and belief-planner replay | `scripts/replay_cached_mpc.sh` |
| No-motion hardware configuration check | `scripts/real_robot_smoke_test.sh` |

## Simulation and robot execution

| Area | Files |
| --- | --- |
| Main Isaac scene | `scripts/open_minimal_scene.py` |
| RGB-D capture | `scripts/observation_capture.py` |
| Basket and scenario generation | `scripts/scanned_basket_scene.py` |
| Persistent UR10e and RG6 contact execution | `scripts/persistent_composite_grasp.py` |
| Live RGB-D and replanning loop | `scripts/run_live_single_gpu_pipeline.py` |
| Cover-removal closed loop | `scripts/run_remove_cover_live_smoke.py` |
| Empty-container closed loop | `scripts/run_negative_evidence_live_smoke.py` |

## Perception

| Area | Files |
| --- | --- |
| GroundingDINO and SAM2 stages | `scripts/run_perception_grounding_pilot.py` |
| Anonymous Qwen input export | `scripts/export_grounded_sam2_qwen_inputs.py` |
| Qwen forced-choice logits | `scripts/qwen3_vl_logits.py` |
| Qwen ranking of grounded candidates | `scripts/run_grounded_proposal_qwen_ranking.py` |
| RGB-D selected-mask localization | `scripts/rgbd_target_localization.py` |
| Hybrid relation evidence | `scripts/run_hybrid_rgbd_relation_pilot.py` |

## Scene Graph, belief, and planning

| Area | Files |
| --- | --- |
| Scene Graph and planner integration | `scripts/run_cover_search_scene_graph_mpc_integration.py` |
| Exact discrete belief tree | `scripts/run_cover_search_belief_mpc.py` |
| Scene-conditioned action model | `scripts/run_scene_conditioned_future_belief_calibration.py` |
| Action-observation calibration audit | `scripts/run_full_action_conditioned_observation_calibration.py` |
| Baseline and ablation runner | `scripts/run_offline_full_baseline_ablation.py` |
| Final-test preflight | `scripts/audit_icra_evaluation_protocol.py` |

## Configuration

| Area | Directory or file |
| --- | --- |
| Simulation scene and camera poses | `configs/sim/` |
| Perception models and thresholds | `configs/perception/` |
| VLM schemas | `configs/vlm/` |
| Scene Graph schema | `configs/scene_graph/` |
| Planner, calibration, and evaluation | `configs/research/` |
| Real-robot measurement worksheet | `configs/hardware/rg6_lid_transfer_calibration.json` |

## Tests

Tests are under `tests/`. Run the CPU regression suite with:

```bash
python -m pytest -q
```

Isaac Sim integration runs are separate because they require RTX rendering and contact physics.
