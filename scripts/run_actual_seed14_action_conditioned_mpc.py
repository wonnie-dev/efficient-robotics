#!/usr/bin/env python3
"""Run the frozen action-conditioned belief planner on actual-motion seed 14."""

import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from run_action_differentiating_mpc_smoke import center_observation
from run_offline_action_conditioned_mpc_replay import (
    build_episode_rows,
    confidence_bin,
    fit_action_model,
    initial_belief,
    select_root_action,
    sigmoid,
    track_belief_observation,
    update_with_observation,
)
from rgbd_target_localization import estimate_mask_center

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/research/offline_action_conditioned_mpc_replay_seed165_184.json"
NESTED_CALIBRATION = ROOT / "outputs/offline_mpc/nested_action_conditioned_mpc_calibration_seed165_184/summary.json"
PERCEPTION = ROOT / "outputs/perception_grounding_pilot/paper_test_actual_multiview_seed014"
OBSERVATIONS = ROOT / "outputs/live_pipeline/paper_test_actual_multiview_seed014/observations"
OUTPUT = ROOT / "outputs/offline_mpc/actual_seed14_action_conditioned/result.json"
ACTION_TO_VIEW = {"viewpoint_right": "right", "viewpoint_close_high": "close_high"}


def load(path):
    return json.loads(Path(path).read_text())


def ranking(view):
    sample = f"seed014_{view}"
    return load(PERCEPTION / "grounded_sam2_qwen_rankings" / sample / "result.json")


def localized_candidates(view, config):
    sample = f"seed014_{view}"
    model_input = load(PERCEPTION / "grounded_sam2_qwen_inputs" / sample / "input.json")
    ranked = ranking(view)
    observation_dir = OBSERVATIONS / view
    depth = np.load(observation_dir / "depth_m.npy")
    calibration = load(observation_dir / "camera_calibration.json")
    raw_logits = dict(zip(ranked["candidate_ids"], ranked["raw_match_logits"]))
    result = {}
    for item in model_input["candidates"]:
        candidate_id = str(item["candidate_id"])
        candidate_mask = np.asarray(Image.open(ROOT / item["mask_path"]).convert("L")) > 0
        try:
            center = estimate_mask_center(depth, candidate_mask, calibration, label=candidate_id)["center_world_m"]
        except ValueError:
            center = None
        probability = sigmoid(float(raw_logits[candidate_id]) / float(config["target_temperature"]))
        result[candidate_id] = {
            "center_world_m": center,
            "identity_bin": confidence_bin(probability, config["identity_confidence_bins"]),
        }
    return result


def euclidean(first, second):
    return float(np.linalg.norm(np.asarray(first) - np.asarray(second)))


def add_track_observation(row, view, center_selected_world, config):
    current = localized_candidates(view, config)
    available = [(euclidean(item["center_world_m"], center_selected_world), candidate_id, item) for candidate_id, item in current.items() if item["center_world_m"] is not None]
    if not available or min(available)[0] > float(config["candidate_tracking"]["maximum_center_distance_m"]):
        agreement, confidence = "missing", "missing"
    else:
        _distance, center_candidate_id, center_candidate = min(available)
        agreement = "same" if row["selected_candidate_id"] == center_candidate_id else "different"
        confidence = center_candidate["identity_bin"]
    row["track_belief_observation"] = track_belief_observation(agreement, confidence)
    row["track_agreement_observation"] = agreement
    row["center_track_confidence_bin"] = confidence
    return row


config = load(CONFIG)
nested = load(NESTED_CALIBRATION)
cost_counts = nested["selected_cost_counts"]["action_conditioned_belief_mpc"]
maximum_count = max(cost_counts.values())
calibration_selected_cost = min(
    float(value) for value, count in cost_counts.items() if count == maximum_count
)
config["task_cost"]["task_noncompletion_cost"] = calibration_selected_cost
calibration = build_episode_rows(config)
model = fit_action_model(calibration, config, action_agnostic=False)
belief = initial_belief(model)
consumed = []
policies = []
view = "center"
center_selected_world = None
root_sensitivity = None
completed_view_actions = set()

for step in range(3):
    # The view file is opened only after the previous policy selected it.
    observation = center_observation(ranking(view), config)
    if step == 0:
        center_candidates = localized_candidates("center", config)
        center_selected_world = center_candidates[observation["selected_candidate_id"]]["center_world_m"]
        if center_selected_world is None:
            raise RuntimeError("Center-selected candidate has no valid RGB-D center")
    else:
        observation = add_track_observation(
            observation, view, center_selected_world, config
        )
    action_name = "initial_observation" if step == 0 else f"viewpoint_{view}"
    belief, selected_probability = update_with_observation(
        belief, observation, model, action=action_name
    )
    consumed.append({"step": step, "view": view, "observation": observation, "belief": belief, "selected_target_correct_probability": selected_probability})
    policy = select_root_action(belief, selected_probability, model, config)
    available_values = [
        item
        for item in policy["action_values"]
        if item["action"] not in completed_view_actions
        and not (
            item["action"] == "grasp"
            and observation["membership_observation"] != "inside"
        )
    ]
    selected_available = min(
        available_values, key=lambda item: item["objective_cost"]
    )
    policy["completed_view_actions_excluded"] = sorted(
        completed_view_actions
    )
    policy["semantic_grasp_gate"] = {
        "required_membership": "inside",
        "observed_membership": observation["membership_observation"],
        "grasp_allowed": observation["membership_observation"] == "inside",
    }
    policy["selected_action"] = selected_available["action"]
    policy["selected_cost"] = selected_available["objective_cost"]
    policies.append(policy)
    if step == 0:
        root_sensitivity = []
        for value in config["task_cost"]["task_noncompletion_cost_sensitivity_grid"]:
            selected_config = copy.deepcopy(config)
            selected_config["task_cost"]["task_noncompletion_cost"] = float(value)
            candidate_policy = select_root_action(
                belief, selected_probability, model, selected_config
            )
            root_sensitivity.append({
                "task_noncompletion_cost": float(value),
                "selected_action": candidate_policy["selected_action"],
                "selected_cost": candidate_policy["selected_cost"],
            })
    selected = policy["selected_action"]
    if selected not in ACTION_TO_VIEW:
        break
    completed_view_actions.add(selected)
    next_view = ACTION_TO_VIEW[selected]
    if next_view in [item["view"] for item in consumed]:
        break
    view = next_view

evaluation = load(PERCEPTION / "evaluation_summary.json")
correct_by_view = {
    row["sample_id"].removeprefix("seed014_"): row["selected_target_correct_at_0_5"]
    for row in evaluation["selected_relation_evaluation"]["rows"]
}
result = {
    "schema_version": "actual-seed14-action-conditioned-mpc-v1",
    "status": "completed",
    "seed": 14,
    "planner": "frozen_discrete_action_conditioned_belief_mpc",
    "horizon": 2,
    "calibration_seeds": sorted(calibration),
    "calibration_selected_task_noncompletion_cost": calibration_selected_cost,
    "calibration_selection_rule": "outer_fold_mode_with_lower_cost_tie_break",
    "consumed_observations": consumed,
    "policies": policies,
    "predeclared_root_cost_sensitivity": root_sensitivity,
    "selected_action_sequence": [p["selected_action"] for p in policies],
    "final_action": policies[-1]["selected_action"],
    "posthoc_selected_target_correct_by_consumed_view": {item["view"]: correct_by_view[item["view"]] for item in consumed},
    "future_view_files_read_before_selection": False,
    "simulator_ground_truth_used_for_action_selection": False,
    "simulator_ground_truth_used_for_posthoc_audit": True,
    "actual_robot_motion_source": "outputs/live_pipeline/paper_test_actual_multiview_seed014/smoke_result.json",
    "training_performed": False,
    "calibration_model_fitting_performed": True,
    "testing_performed": False,
    "valid_for_final_evaluation": False,
    "limitations": [
        "The observation model is fitted on development calibration episodes 165-184.",
        "This single episode is an integration pilot, not a final paper evaluation.",
        "The planner is discrete belief-space MPC; continuous robot trajectory generation remains a separate controller.",
    ],
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"sequence": result["selected_action_sequence"], "final_action": result["final_action"], "consumed_views": [x["view"] for x in consumed], "posthoc_correct": result["posthoc_selected_target_correct_by_consumed_view"]}, indent=2))
