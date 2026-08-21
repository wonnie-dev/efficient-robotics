"""Run two sequential action-conditioned re-observation belief updates."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from run_non_oracle_hybrid_planner import plan
from qwen_belief_adapter import weighted_log_belief_update
from evaluate_occlusion_belief import (
    DEFAULT_CONFIG,
    candidate_centers,
    ranking_belief,
    ranking_paths,
    resolve,
    track_mapping,
    write_json,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERCEPTION = (
    ROOT
    / "configs"
    / "perception"
    / "scanned_basket_occlusion_two_step_seed000.json"
)
DEFAULT_MOTION = (
    ROOT
    / "outputs"
    / "live_pipeline"
    / "reachable_view_capture"
    / "seed000"
    / "run006"
    / "result.json"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs" / "scanned_basket_occlusion_two_step_belief_mpc_seed000"
)


def planner_config(
    base: dict,
    belief: dict,
    *,
    completed_reobservations: int,
    executed_actions: list[str],
    perception_config_path: Path,
) -> dict:
    config = copy.deepcopy(base)
    config["scenario"]["perception_config"] = str(
        perception_config_path.resolve()
    )
    config["initial_belief"]["target"] = belief["target"]
    config["initial_belief"]["relation"] = belief["relation"]
    config["initial_belief"]["source"] = (
        "sequential_grounded_sam2_qwen_rgbd_track_belief"
    )
    config["completed_reobservations"] = completed_reobservations
    if completed_reobservations:
        config["actions"]["viewpoint_center_repeat"]["enabled"] = False
    for action in executed_actions:
        config["actions"][action]["enabled"] = False
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--perception-config", type=Path, default=DEFAULT_PERCEPTION
    )
    parser.add_argument(
        "--executed-motion-result", type=Path, default=DEFAULT_MOTION
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base = json.loads(args.config.read_text(encoding="utf-8"))
    perception_config = json.loads(
        args.perception_config.read_text(encoding="utf-8")
    )
    adapter = base["qwen_belief_adapter"]
    temperature = float(adapter["raw_logit_temperature"])
    observation_weight = float(adapter["observation_log_weight"])
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    center_ranking_path, center_observation_dir = ranking_paths(
        perception_config, "center"
    )
    center_ranking = json.loads(
        center_ranking_path.read_text(encoding="utf-8")
    )
    center_centers = candidate_centers(
        center_ranking, center_observation_dir
    )
    center_mapping = {
        candidate_id: f"track_{index:03d}"
        for index, candidate_id in enumerate(center_centers, start=1)
    }
    center_observation_belief = ranking_belief(
        center_ranking, center_mapping, temperature
    )
    belief = {
        "target": center_observation_belief["target"],
        "relation": center_observation_belief["relation"],
    }
    executed_actions: list[str] = []
    selected_views: list[str] = []
    steps = []

    for step_index in range(2):
        config = planner_config(
            base,
            belief,
            completed_reobservations=step_index,
            executed_actions=executed_actions,
            perception_config_path=args.perception_config,
        )
        current_plan = plan(config)
        plan_path = output_root / f"pre_action_plan_{step_index:03d}.json"
        write_json(
            plan_path,
            {
                **current_plan,
                "step_index": step_index,
                "future_perception_outputs_read_before_plan": [],
                "future_capture_files_read_before_plan": [],
            },
        )
        action = current_plan["action_request"]["type"]
        if not action.startswith("viewpoint_"):
            raise RuntimeError(
                f"Step {step_index} expected viewpoint action, got {action}"
            )
        view = action.removeprefix("viewpoint_")
        if view == "center_repeat":
            view = "center"

        # This selected future output is opened only after the plan is saved.
        ranking_path, observation_dir = ranking_paths(
            perception_config, view
        )
        ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
        centers = candidate_centers(ranking, observation_dir)
        mapping, tracking = track_mapping(center_centers, centers)
        observation_belief = ranking_belief(
            ranking, mapping, temperature
        )
        belief_before = copy.deepcopy(belief)
        belief = weighted_log_belief_update(
            belief,
            {
                "target": observation_belief["target"],
                "relation": observation_belief["relation"],
            },
            observation_weight,
        )
        executed_actions.append(action)
        selected_views.append(view)
        steps.append(
            {
                "step_index": step_index,
                "plan_path": str(plan_path),
                "selected_action": action,
                "selected_view": view,
                "selected_perception_output": str(ranking_path),
                "belief_before_update": belief_before,
                "observation_belief": {
                    "target": observation_belief["target"],
                    "relation": observation_belief["relation"],
                },
                "belief_after_update": belief,
                "rgbd_candidate_tracking": tracking,
            }
        )

    final_config = planner_config(
        base,
        belief,
        completed_reobservations=2,
        executed_actions=executed_actions,
        perception_config_path=args.perception_config,
    )
    final_plan = plan(final_config)
    final_plan_path = output_root / "final_replan.json"
    write_json(final_plan_path, final_plan)

    motion = json.loads(
        args.executed_motion_result.read_text(encoding="utf-8")
    )
    expected_sequence = ["center", *selected_views]
    trajectories = motion.get("trajectories", [])
    motion_valid = (
        motion.get("status") == "completed"
        and motion.get("sequence") == expected_sequence
        and len(trajectories) == len(selected_views)
        and all(
            trajectory.get("collision_checked")
            and not trajectory.get("collision_detected")
            and trajectory.get("actual_robot_motion_executed")
            for trajectory in trajectories
        )
    )
    if not motion_valid:
        raise RuntimeError(
            "Actual motion does not match planned sequence: "
            f"expected={expected_sequence}, result={motion}"
        )

    result = {
        "schema_version": "multiview-belief-evaluation-v1",
        "status": "completed",
        "initial_center_observation_belief": {
            "target": center_observation_belief["target"],
            "relation": center_observation_belief["relation"],
        },
        "steps": steps,
        "planned_and_executed_sequence": expected_sequence,
        "actual_motion_result": str(
            args.executed_motion_result.resolve()
        ),
        "actual_motion_validated": True,
        "final_belief": belief,
        "final_plan_path": str(final_plan_path),
        "final_action": final_plan["action_request"]["type"],
        "final_commitment_gate": final_plan["commitment_gate"],
        "final_action_execution": {
            "status": "blocked_not_executed",
            "blocking_reasons": [
                "scanned_basket_collision_geometry_not_validated",
                "uncalibrated_belief_not_authorized_for_irreversible_action",
            ],
            "planner_request_preserved": final_plan[
                "action_request"
            ]["type"],
        },
        "provenance": {
            "future_capture_files_read_before_each_plan": [],
            "future_perception_outputs_read_before_each_plan": [],
            "selected_outputs_read_only_after_each_plan_was_saved": True,
            "unselected_future_view_output_read": False,
            "future_observation_prediction_source": (
                "pre_action_geometry_informed_hand_specified_likelihood"
            ),
            "simulator_ground_truth_used_for_inference_or_planning": False,
            "oracle": False,
            "actual_mpc_solver": False,
        },
        "training_performed": False,
        "calibration_performed": False,
        "grasp_executed": False,
        "valid_for_final_evaluation": False,
    }
    write_json(output_root / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
