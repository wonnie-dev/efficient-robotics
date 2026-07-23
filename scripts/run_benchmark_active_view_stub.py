"""Select a benchmark re-observation using a provisional one-step objective."""

import argparse
import json
from pathlib import Path

from update_multi_view_belief_stub import entropy, fuse_distributions, rounded


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/policy/benchmark_active_view_stub.json"
POSE_CONFIG = ROOT / "configs/sim/observation_poses.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def relation_distribution(graph: dict) -> dict[str, float]:
    probability = graph["graph_belief"]["required_relation_probability"]
    return {"inside": probability, "unknown": 1.0 - probability}


def mean_joint_motion(source: str, target: str, poses: dict) -> float:
    a, b = poses["poses_rad"][source], poses["poses_rad"][target]
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def task_risk(target: dict, relation: dict, target_id: str) -> float:
    return 1.0 - target[target_id] * relation["inside"]


def evaluate(current: dict, candidate: dict, motion: float, config: dict) -> dict:
    floor = config["probability_floor"]
    current_target = current["graph_belief"]["target_distribution"]
    current_relation = relation_distribution(current)
    predicted_target = fuse_distributions(
        current_target, candidate["graph_belief"]["target_distribution"], floor
    )
    predicted_relation = fuse_distributions(
        current_relation, relation_distribution(candidate), floor
    )
    risk_before = task_risk(current_target, current_relation, config["target_id"])
    risk_after = task_risk(predicted_target, predicted_relation, config["target_id"])
    target_gain = entropy(current_target) - entropy(predicted_target)
    relation_gain = entropy(current_relation) - entropy(predicted_relation)
    weights = config["objective_weights"]
    utility = (
        weights["task_risk_reduction"] * (risk_before - risk_after)
        + weights["target_entropy_reduction"] * target_gain
        + weights["relation_entropy_reduction"] * relation_gain
        - weights["joint_motion_cost"] * motion
    )
    return {
        "predicted_target_distribution": rounded(predicted_target),
        "predicted_relation_distribution": rounded(predicted_relation),
        "task_failure_risk_before": round(risk_before, 8),
        "predicted_task_failure_risk_after": round(risk_after, 8),
        "expected_task_risk_reduction": round(risk_before - risk_after, 8),
        "expected_target_entropy_reduction_nats": round(target_gain, 8),
        "expected_relation_entropy_reduction_nats": round(relation_gain, 8),
        "mean_joint_motion_rad": round(motion, 8),
        "utility": round(utility, 8),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args, _unknown = parser.parse_known_args()
    config = load_json(args.config)
    poses = load_json(POSE_CONFIG)
    root = ROOT / config["observation_root"]
    filename = config["belief_graph_filename"]
    current = load_json(root / config["initial_view"] / filename)
    candidates = {}
    for view in config["candidate_views"]:
        candidates[view] = evaluate(
            current,
            load_json(root / view / filename),
            mean_joint_motion(config["initial_view"], view, poses),
            config,
        )
    selected_view = max(candidates, key=lambda view: candidates[view]["utility"])
    action = {
        "type": "move_to_observation_pose",
        "pose_name": selected_view,
        "reason": "highest_provisional_task_risk_reduction_utility",
    }
    selected = candidates[selected_view]
    gate = config["temporary_execution_gate"]
    result = {
        "status": config["status"],
        "controller_type": "one_step_information_seeking_stub_not_mpc",
        "initial_view": config["initial_view"],
        "candidate_prediction_source": config["candidate_prediction_source"],
        "candidate_evaluations": candidates,
        "selected_action": action,
        "post_action_replay_gate_passed": (
            selected["predicted_target_distribution"][config["target_id"]]
            >= gate["target_probability_minimum"]
            and selected["predicted_relation_distribution"][config["required_relation"]]
            >= gate["relation_probability_minimum"]
        ),
        "provenance": {
            "ground_truth_used": True,
            "actual_robot_motion_executed": False,
            "allowed_for_final_evaluation": config["allowed_for_final_evaluation"],
        },
    }
    output_root = ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "decision.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "action_request.json").write_text(
        json.dumps(action, indent=2) + "\n", encoding="utf-8"
    )
    print(f"SELECTED={selected_view} UTILITY={selected['utility']:.6f}")
    for view, evaluation in candidates.items():
        print(
            f"CANDIDATE={view} UTILITY={evaluation['utility']:.6f} "
            f"RISK_REDUCTION={evaluation['expected_task_risk_reduction']:.6f}"
        )


if __name__ == "__main__":
    main()
