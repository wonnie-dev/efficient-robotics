"""Non-oracle hybrid receding-horizon planning prototype.

Planning reads only the current belief and a pre-action observation-likelihood
model. It never reads future RGB, depth, masks, or captured future graphs.
"""

import argparse
import itertools
import json
from pathlib import Path

from calibrated_belief import (
    bayesian_update,
    binary_detection_likelihood,
    entropy,
    normalize,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/research/initial_method_design.json"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def task_failure_risk(belief: dict, objective: dict) -> float:
    selected_target_probability = max(belief["target"].values())
    return 1.0 - (
        selected_target_probability
        * belief["relation"][objective["required_relation"]]
    )


def observation_branches(
    belief: dict, action_name: str, observation_model: dict
) -> list[dict]:
    model = observation_model[action_name]
    detection_probability = model["target_detection_probability"]
    relation_likelihood = model["relation_likelihood"]
    branches = []
    for detected, relation_outcome in itertools.product(
        (True, False), observation_model["relation_outcomes"]
    ):
        target_likelihood = binary_detection_likelihood(
            detection_probability, detected
        )
        relation_outcome_likelihood = {
            hypothesis: outcomes[relation_outcome]
            for hypothesis, outcomes in relation_likelihood.items()
        }
        target_outcome_probability = sum(
            belief["target"][hypothesis] * target_likelihood[hypothesis]
            for hypothesis in belief["target"]
        )
        relation_outcome_probability = sum(
            belief["relation"][hypothesis]
            * relation_outcome_likelihood[hypothesis]
            for hypothesis in belief["relation"]
        )
        probability = target_outcome_probability * relation_outcome_probability
        branches.append(
            {
                "probability": probability,
                "observation": {
                    "target_detected": detected,
                    "relation_outcome": relation_outcome,
                },
                "posterior": {
                    "target": bayesian_update(belief["target"], target_likelihood),
                    "relation": bayesian_update(
                        belief["relation"], relation_outcome_likelihood
                    ),
                },
            }
        )
    total = sum(branch["probability"] for branch in branches)
    for branch in branches:
        branch["probability"] /= total
    return branches


def expected_view_metrics(
    belief: dict, action_name: str, config: dict
) -> dict:
    objective = config["objective"]
    branches = observation_branches(
        belief, action_name, config["observation_model"]
    )
    prior_entropy = entropy(belief["target"]) + entropy(belief["relation"])
    expected_entropy = sum(
        branch["probability"]
        * (
            entropy(branch["posterior"]["target"])
            + entropy(branch["posterior"]["relation"])
        )
        for branch in branches
    )
    expected_risk = sum(
        branch["probability"]
        * task_failure_risk(branch["posterior"], objective)
        for branch in branches
    )
    return {
        "branches": branches,
        "expected_task_failure_risk": expected_risk,
        "expected_information_gain_nats": prior_entropy - expected_entropy,
    }


def immediate_cost(action_name: str, belief: dict, config: dict) -> dict:
    action = config["actions"][action_name]
    objective = config["objective"]
    if action["kind"] == "grasp":
        expected_risk = task_failure_risk(belief, objective)
        information_gain = 0.0
        risk_cost = objective["task_failure_risk_weight"] * expected_risk
    elif action["kind"] == "viewpoint":
        metrics = expected_view_metrics(belief, action_name, config)
        expected_risk = metrics["expected_task_failure_risk"]
        information_gain = metrics["expected_information_gain_nats"]
        # View actions are intermediate information-gathering actions. Their
        # expected terminal risk is charged by the later grasp action.
        risk_cost = 0.0
    else:
        raise ValueError(f"Enabled action has no model: {action_name}")
    cost = (
        risk_cost
        - objective["task_conditioned_information_gain_weight"] * information_gain
        + objective["motion_cost_weight"] * action.get("motion_cost", 0.0)
        + objective["collision_cost_weight"] * action.get("collision_cost", 0.0)
    )
    return {
        "expected_task_failure_risk": expected_risk,
        "expected_information_gain_nats": information_gain,
        "motion_cost": action.get("motion_cost", 0.0),
        "collision_cost": action.get("collision_cost", 0.0),
        "objective_cost": cost,
    }


def sequence_cost(sequence: list[str], belief: dict, config: dict) -> float:
    action_name = sequence[0]
    action = config["actions"][action_name]
    metrics = immediate_cost(action_name, belief, config)
    if len(sequence) == 1 or action["kind"] == "grasp":
        return metrics["objective_cost"]
    branches = observation_branches(belief, action_name, config["observation_model"])
    future = sum(
        branch["probability"]
        * sequence_cost(sequence[1:], branch["posterior"], config)
        for branch in branches
    )
    return metrics["objective_cost"] + future


def candidate_sequences(config: dict) -> list[list[str]]:
    enabled = [
        name for name, action in config["actions"].items() if action["enabled"]
    ]
    sequences = [["grasp"]]
    for length in range(2, config["horizon"] + 1):
        for sequence in itertools.product(enabled, repeat=length):
            if sequence[-1] != "grasp" or "grasp" in sequence[:-1]:
                continue
            sequences.append(list(sequence))
    return sequences


def plan(config: dict) -> dict:
    belief = {
        "target": normalize(config["initial_belief"]["target"]),
        "relation": normalize(config["initial_belief"]["relation"]),
    }
    evaluations = []
    for sequence in candidate_sequences(config):
        evaluations.append(
            {
                "sequence": sequence,
                "cost": round(sequence_cost(sequence, belief, config), 8),
                "first_action_metrics": {
                    key: round(value, 8)
                    for key, value in immediate_cost(
                        sequence[0], belief, config
                    ).items()
                },
            }
        )
    selected = min(evaluations, key=lambda item: item["cost"])
    return {
        "planner": "hybrid_receding_horizon_engineering_prototype",
        "horizon": config["horizon"],
        "current_belief": belief,
        "current_entropy_nats": {
            "target": round(entropy(belief["target"]), 8),
            "relation": round(entropy(belief["relation"]), 8),
        },
        "candidate_sequences": evaluations,
        "selected_sequence": selected["sequence"],
        "action_request": {
            "type": selected["sequence"][0],
            "reason": "minimum_non_oracle_expected_horizon_objective",
        },
        "disabled_actions": {
            name: action.get("disabled_until", "disabled_for_current_replan")
            for name, action in config["actions"].items()
            if not action["enabled"]
        },
        "provenance": {
            "future_capture_files_read": [],
            "future_observation_source": "pre_action_likelihood_model",
            "oracle": False,
            "actual_mpc_solver": False,
            "calibration_status": config["initial_belief"]["calibrated"],
            "valid_for_final_evaluation": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_json(args.config)
    result = plan(config)
    output_root = ROOT / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    output = output_root / "plan.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"SELECTED_SEQUENCE={' -> '.join(result['selected_sequence'])}")
    print(f"FIRST_ACTION={result['action_request']['type']}")
    print(f"ORACLE={result['provenance']['oracle']}")
    print(f"WROTE={output}")


if __name__ == "__main__":
    main()
