# Code Map

## Entry points

| Task | Entry point |
| --- | --- |
| Headless scene and RGB-D capture | `scripts/run_simulation_episode.sh` |
| One Qwen perception request | `scripts/run_perception_demo.sh` |
| Cached-observation belief-planner evaluation | `scripts/run_offline_action_conditioned_mpc_replay.py` |
| No-motion hardware configuration check | `scripts/validate_real_robot_configuration.sh` |
| One closed-loop simulation episode | `scripts/run_closed_loop_episode.py` |
| Multi-GPU episode batch | `scripts/run_episode_batch.py` |

## Simulation and robot execution

| Area | Files |
| --- | --- |
| Main Isaac scene | `scripts/isaac_sim_server.py` |
| RG6 and composite robot import | `scripts/import_rg6_asset.py`, `scripts/import_ur10e_rg6_asset.py` |
| RGB-D capture | `scripts/observation_capture.py` |
| Basket and scenario generation | `scripts/scanned_basket_scene.py` |
| Persistent UR10e and RG6 contact execution | `scripts/persistent_composite_grasp.py` |
| Live RGB-D and replanning loop | `scripts/run_closed_loop_pipeline.py` |
| Cover-removal closed loop | `scripts/execute_cover_removal.py` |
| Empty-container closed loop | `scripts/run_negative_evidence_episode.py` |

## Perception

| Area | Files |
| --- | --- |
| GroundingDINO and SAM2 stages | `scripts/run_grounded_segmentation.py` |
| Anonymous Qwen input export | `scripts/export_grounded_sam2_qwen_inputs.py` |
| Qwen forced-choice logits | `scripts/qwen3_vl_logits.py` |
| Qwen ranking of grounded candidates | `scripts/run_grounded_proposal_qwen_ranking.py` |
| RGB-D selected-mask localization | `scripts/rgbd_target_localization.py` |
| Hybrid relation evidence | `scripts/estimate_rgbd_relations.py` |

## Scene Graph, belief, and planning

| Area | Files |
| --- | --- |
| Scene Graph and planner integration | `scripts/run_cover_search_scene_graph_mpc_integration.py` |
| Exact discrete belief tree | `scripts/run_cover_search_belief_mpc.py` |
| Scene-conditioned action model | `scripts/run_scene_conditioned_future_belief_calibration.py` |
| Action-observation calibration audit | `scripts/run_full_action_conditioned_observation_calibration.py` |
| Baseline and ablation runner | `scripts/run_offline_full_baseline_ablation.py` |
| Final-test preflight | `scripts/audit_evaluation_protocol.py` |
| Joint observation calibration | `scripts/calibrate_joint_observation_model.py` |
| Frozen baseline and ablation comparison | `scripts/evaluate_policy_comparison.py` |
| Calibration freeze | `scripts/freeze_calibration.py` |

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
python -m unittest discover -s tests -p 'test_*.py'
```

Isaac Sim integration runs are separate because they require RTX rendering and contact physics.
