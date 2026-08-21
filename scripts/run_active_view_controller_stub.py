"""Select one active viewpoint using provisional belief-reduction estimates.

Candidate outcomes are replayed from already captured rule-based stub graphs.
This is an offline oracle-style interface test, not online prediction or MPC.
"""

import argparse
import json
from pathlib import Path

from update_multi_view_belief_stub import (
    binary_target_distribution,
    entropy,
    fuse_distributions,
    relation_distribution,
    rounded,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/policy/active_view_controller_stub.json"
POSE_CONFIG = ROOT / "configs/sim/observation_poses.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def graph_path(config: dict, view: str) -> Path:
    return ROOT / config["observation_root"] / view / config["belief_graph_filename"]


def mean_joint_motion(from_view: str, to_view: str, poses: dict) -> float:
    source = poses["poses_rad"][from_view]
    target = poses["poses_rad"][to_view]
    if len(source) != len(target):
        raise ValueError("Observation poses must have the same joint dimension")
    return sum(abs(a - b) for a, b in zip(source, target)) / len(source)


def evaluate_candidate(
    current_target: dict[str, float],
    current_relation: dict[str, float],
    candidate_graph: dict,
    motion_cost: float,
    config: dict,
) -> dict:
    floor = config["fusion"]["probability_floor"]
    predicted_target = fuse_distributions(
        current_target, binary_target_distribution(candidate_graph), floor
    )
    predicted_relation = fuse_distributions(
        current_relation, relation_distribution(candidate_graph), floor
    )
    target_reduction = entropy(current_target) - entropy(predicted_target)
    relation_reduction = entropy(current_relation) - entropy(predicted_relation)
    weights = config["objective_weights"]
    utility = (
        weights["target_entropy_reduction"] * target_reduction
        + weights["relation_entropy_reduction"] * relation_reduction
        - weights["joint_motion_cost"] * motion_cost
    )
    return {
        "predicted_target_distribution": rounded(predicted_target),
        "predicted_relation_distribution": rounded(predicted_relation),
        "expected_target_entropy_reduction_nats": round(target_reduction, 8),
        "expected_relation_entropy_reduction_nats": round(relation_reduction, 8),
        "mean_joint_motion_rad": round(motion_cost, 8),
        "utility": round(utility, 8),
    }


def gate_passed(
    target: dict[str, float],
    relation: dict[str, float],
    config: dict,
) -> bool:
    gate = config["temporary_execution_gate"]
    target_id = max(target, key=target.get)
    return (
        target[target_id] >= gate["target_probability_minimum"]
        and relation[config["required_relation"]] >= gate["relation_probability_minimum"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_json(args.config)
    poses = load_json(POSE_CONFIG)
    initial_view = config["initial_view"]
    initial_graph = load_json(graph_path(config, initial_view))
    current_target = binary_target_distribution(initial_graph)
    current_relation = relation_distribution(initial_graph)
    initial_ready = gate_passed(current_target, current_relation, config)

    candidates = {}
    if not initial_ready:
        for view in config["candidate_views"]:
            candidate_graph = load_json(graph_path(config, view))
            candidates[view] = evaluate_candidate(
                current_target,
                current_relation,
                candidate_graph,
                mean_joint_motion(initial_view, view, poses),
                config,
            )
        selected_view = max(candidates, key=lambda view: candidates[view]["utility"])
        selected = candidates[selected_view]
        final_target = selected["predicted_target_distribution"]
        final_relation = selected["predicted_relation_distribution"]
        action = {
            "type": "move_to_observation_pose",
            "pose_name": selected_view,
            "reason": "highest_predicted_belief_reduction_utility",
        }
    else:
        selected_view = initial_view
        final_target = current_target
        final_relation = current_relation
        action = {
            "type": "proceed_to_retrieval_check",
            "pose_name": initial_view,
            "reason": "initial_belief_passes_temporary_execution_gate",
        }

    result = {
        "status": config["status"],
        "controller_type": "one_step_active_view_rule_stub",
        "initial_view": initial_view,
        "initial_belief": {
            "target_distribution": current_target,
            "relation_distribution": current_relation,
            "target_entropy_nats": round(entropy(current_target), 8),
            "relation_entropy_nats": round(entropy(current_relation), 8),
            "temporary_execution_gate_passed": initial_ready,
        },
        "candidate_prediction_source": config["candidate_prediction_source"],
        "candidate_evaluations": candidates,
        "selected_action": action,
        "post_action_replay_belief": {
            "target_distribution": final_target,
            "relation_distribution": final_relation,
            "temporary_execution_gate_passed": gate_passed(
                final_target, final_relation, config
            ),
        },
        "provenance": {
            "ground_truth_used": True,
            "actual_robot_motion_executed": False,
            "allowed_for_final_evaluation": config["allowed_for_final_evaluation"],
            "notes": (
                "Candidate results are replayed from existing captures. Replace this "
                "oracle-style predictor before online or final evaluation."
            ),
        },
    }
    output_dir = ROOT / config["output_root"]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "decision.json"
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    action_path = output_dir / "action_request.json"
    with action_path.open("w", encoding="utf-8") as stream:
        json.dump(action, stream, indent=2, ensure_ascii=False)
        stream.write("\n")

    print(f"WROTE={output_path}")
    print(f"WROTE={action_path}")
    print(
        f"ACTION={action['type']} VIEW={selected_view} "
        f"POST_GATE={result['post_action_replay_belief']['temporary_execution_gate_passed']}"
    )
    for view, evaluation in candidates.items():
        print(
            f"CANDIDATE={view} UTILITY={evaluation['utility']:.6f} "
            f"TARGET_GAIN={evaluation['expected_target_entropy_reduction_nats']:.6f} "
            f"RELATION_GAIN={evaluation['expected_relation_entropy_reduction_nats']:.6f}"
        )


if __name__ == "__main__":
    main()
