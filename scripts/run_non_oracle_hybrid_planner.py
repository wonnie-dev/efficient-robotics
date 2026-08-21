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
    """Estimate failure as the complement of the best supported task state."""
    selected_target_probability = max(belief["target"].values())
    return 1.0 - (
        selected_target_probability
        * belief["relation"][objective["required_relation"]]
    )


def observation_branches(
    belief: dict, action_name: str, observation_model: dict
) -> list[dict]:
    """Forecast an action's possible observations and Bayesian posteriors.

    The target-detection and relation channels are factored by this prototype,
    so their predictive probabilities multiply at each branch.
    """
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
    # Normalize once more to absorb rounding error in the configured tables.
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
        "prior_entropy_nats": prior_entropy,
        "expected_posterior_entropy_nats": expected_entropy,
        "expected_task_failure_risk": expected_risk,
        "expected_information_gain_nats": prior_entropy - expected_entropy,
    }


def immediate_cost(action_name: str, belief: dict, config: dict) -> dict:
    """Score the current action without peeking at a realized observation."""
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
    """Evaluate a fixed sequence over every predicted observation branch."""
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


def action_feasibility(
    action_name: str, belief: dict, config: dict
) -> dict:
    action = config["actions"][action_name]
    if action["kind"] != "viewpoint":
        return {"feasible": True, "blocking_reasons": []}
    minimum_detection = action.get(
        "minimum_expected_target_detection_probability"
    )
    if minimum_detection is None:
        return {"feasible": True, "blocking_reasons": []}
    probabilities = config["observation_model"][action_name][
        "target_detection_probability"
    ]
    expected_detection = sum(
        belief["target"][hypothesis] * probabilities[hypothesis]
        for hypothesis in belief["target"]
    )
    feasible = expected_detection >= minimum_detection
    return {
        "feasible": feasible,
        "expected_target_detection_probability": expected_detection,
        "minimum_expected_target_detection_probability": minimum_detection,
        "blocking_reasons": (
            [] if feasible else ["predicted_observation_not_usable"]
        ),
    }


def candidate_sequences(config: dict, belief: dict) -> list[list[str]]:
    enabled = [
        name
        for name, action in config["actions"].items()
        if action["enabled"]
        and action_feasibility(name, belief, config)["feasible"]
    ]
    gate = commitment_gate(config["initial_belief"], config)
    sequences = [["grasp"]] if gate["grasp_allowed"] else []
    for length in range(2, config["horizon"] + 1):
        for sequence in itertools.product(enabled, repeat=length):
            if sequence[-1] != "grasp" or "grasp" in sequence[:-1]:
                continue
            sequences.append(list(sequence))
    return sequences


def commitment_gate(
    belief: dict,
    config: dict,
    *,
    completed_reobservations: int | None = None,
) -> dict:
    """Prevent unsafe irreversible commitment during receding-horizon execution.

    This is an execution safety constraint, not the action-selection objective
    and not the paper's claimed contribution. Viewpoint choices are still
    ranked using action-conditioned future observation/posterior branches.
    """
    settings = config.get("commitment_policy", {})
    maximum_risk = float(settings.get("maximum_task_failure_risk", 1.0))
    minimum_reobservations = int(
        settings.get("minimum_completed_reobservations", 0)
    )
    completed = int(
        config.get("completed_reobservations", 0)
        if completed_reobservations is None
        else completed_reobservations
    )
    risk = task_failure_risk(belief, config["objective"])
    reasons = []
    if risk > maximum_risk:
        reasons.append("task_failure_risk_above_threshold")
    if completed < minimum_reobservations:
        reasons.append("insufficient_completed_reobservations")
    return {
        "grasp_allowed": not reasons,
        "current_task_failure_risk": risk,
        "maximum_task_failure_risk": maximum_risk,
        "completed_reobservations": completed,
        "minimum_completed_reobservations": minimum_reobservations,
        "blocking_reasons": reasons,
        "role": "temporary_irreversible_action_safety_constraint",
    }


def belief_tree_action_values(
    belief: dict,
    config: dict,
    *,
    depth: int,
    completed_reobservations: int,
    used_viewpoints: frozenset[str] = frozenset(),
    execution_root: bool = True,
) -> list[dict]:
    """Evaluate a finite-horizon feedback policy over observation branches.

    Unlike the legacy fixed-sequence enumeration, each possible observation
    branch independently chooses its best continuation action.  The current
    irreversible-action gate applies at the execution root and every predicted
    branch.  An unsafe horizon leaf receives a terminal defer value instead of
    pretending that an unsafe grasp will execute.  After the actual
    observation arrives, the MPC replans from the measured posterior.
    """
    if depth < 1:
        return []

    evaluations = []
    gate = commitment_gate(
        belief,
        config,
        completed_reobservations=completed_reobservations,
    )
    grasp = config["actions"].get("grasp")
    if grasp and grasp["enabled"] and gate["grasp_allowed"]:
        metrics = immediate_cost("grasp", belief, config)
        evaluations.append(
            {
                "action": "grasp",
                "kind": "terminal_commitment",
                "cost": metrics["objective_cost"],
                "first_action_metrics": metrics,
                "observation_branches": [],
            }
        )

    if depth == 1:
        if not evaluations and not execution_root:
            risk = task_failure_risk(belief, config["objective"])
            evaluations.append(
                {
                    "action": "defer",
                    "kind": "unsafe_terminal_commitment_avoided",
                    "cost": (
                        config["objective"]["task_failure_risk_weight"]
                        * risk
                    ),
                    "first_action_metrics": {
                        "expected_task_failure_risk": risk,
                        "expected_information_gain_nats": 0.0,
                        "motion_cost": 0.0,
                        "collision_cost": 0.0,
                        "objective_cost": (
                            config["objective"][
                                "task_failure_risk_weight"
                            ]
                            * risk
                        ),
                    },
                    "observation_branches": [],
                }
            )
        return evaluations

    for action_name, action in config["actions"].items():
        if (
            not action["enabled"]
            or action["kind"] != "viewpoint"
            or action_name in used_viewpoints
            or not action_feasibility(action_name, belief, config)["feasible"]
        ):
            continue
        metrics = immediate_cost(action_name, belief, config)
        branch_records = []
        expected_future_cost = 0.0
        feasible = True
        for branch in observation_branches(
            belief, action_name, config["observation_model"]
        ):
            continuations = belief_tree_action_values(
                branch["posterior"],
                config,
                depth=depth - 1,
                completed_reobservations=completed_reobservations + 1,
                used_viewpoints=used_viewpoints | {action_name},
                execution_root=False,
            )
            if not continuations:
                feasible = False
                break
            continuation = min(
                continuations, key=lambda item: item["cost"]
            )
            expected_future_cost += (
                branch["probability"] * continuation["cost"]
            )
            branch_records.append(
                {
                    **branch,
                    "continuation_action": continuation["action"],
                    "continuation_cost": continuation["cost"],
                }
            )
        if not feasible:
            continue
        evaluations.append(
            {
                "action": action_name,
                "kind": "action_conditioned_future_belief",
                "cost": metrics["objective_cost"] + expected_future_cost,
                "first_action_metrics": metrics,
                "expected_future_cost": expected_future_cost,
                "observation_branches": branch_records,
            }
        )
    return evaluations


def belief_tree_policy(belief: dict, config: dict) -> dict:
    """Select the lowest-cost root action from the current belief tree."""
    evaluations = belief_tree_action_values(
        belief,
        config,
        depth=int(config["horizon"]),
        completed_reobservations=int(
            config.get("completed_reobservations", 0)
        ),
    )
    selected = (
        min(evaluations, key=lambda item: item["cost"])
        if evaluations
        else None
    )
    return {
        "method": "exact_discrete_action_observation_belief_tree",
        "feedback_policy": True,
        "replans_after_first_action": True,
        "horizon": int(config["horizon"]),
        "action_values": evaluations,
        "selected": selected,
    }


def action_forecasts(belief: dict, config: dict) -> dict:
    """Expose the pre-action observation and posterior predictions.

    These forecasts are intentionally computed only from the configured
    likelihood model and current belief.  Captured future RGB-D, masks, and
    future scene graphs are not inputs to this function.
    """
    forecasts = {}
    objective = config["objective"]
    for action_name, action in config["actions"].items():
        if not action["enabled"]:
            continue
        if action["kind"] == "grasp":
            forecasts[action_name] = {
                "kind": "terminal_commitment",
                "observation_branches": [],
                "expected_task_failure_risk": task_failure_risk(
                    belief, objective
                ),
                "expected_information_gain_nats": 0.0,
            }
            continue
        metrics = expected_view_metrics(belief, action_name, config)
        forecasts[action_name] = {
            "kind": "action_conditioned_future_observation",
            "observation_branches": [
                {
                    **branch,
                    "posterior_task_failure_risk": task_failure_risk(
                        branch["posterior"], objective
                    ),
                }
                for branch in metrics["branches"]
            ],
            "prior_entropy_nats": metrics["prior_entropy_nats"],
            "expected_posterior_entropy_nats": metrics[
                "expected_posterior_entropy_nats"
            ],
            "expected_task_failure_risk": metrics[
                "expected_task_failure_risk"
            ],
            "expected_information_gain_nats": metrics[
                "expected_information_gain_nats"
            ],
        }
    return forecasts


def plan(config: dict) -> dict:
    """Build a plan from the current belief and pre-action models only."""
    belief = {
        "target": normalize(config["initial_belief"]["target"]),
        "relation": normalize(config["initial_belief"]["relation"]),
    }
    planner_mode = config.get(
        "planner_mode", "legacy_fixed_sequence_prototype"
    )
    belief_tree = None
    if planner_mode == "belief_tree_mpc":
        belief_tree = belief_tree_policy(belief, config)
        evaluations = [
            {
                "sequence": (
                    [item["action"]]
                    if item["action"] == "grasp"
                    else [item["action"], "adaptive_branch_continuation"]
                ),
                "cost": round(item["cost"], 8),
                "first_action_metrics": {
                    key: round(value, 8)
                    for key, value in item[
                        "first_action_metrics"
                    ].items()
                },
            }
            for item in belief_tree["action_values"]
        ]
        if belief_tree["selected"] is None:
            # Keep an unsafe or incomplete planning state visible to the
            # executor instead of silently substituting a grasp.
            selected = {
                "sequence": ["defer"],
                "cost": None,
                "first_action_metrics": {},
            }
        else:
            action = belief_tree["selected"]["action"]
            selected = {
                "sequence": (
                    [action]
                    if action == "grasp"
                    else [action, "adaptive_branch_continuation"]
                ),
                "cost": belief_tree["selected"]["cost"],
                "first_action_metrics": belief_tree["selected"][
                    "first_action_metrics"
                ],
            }
    else:
        evaluations = []
        for sequence in candidate_sequences(config, belief):
            evaluations.append(
                {
                    "sequence": sequence,
                    "cost": round(
                        sequence_cost(sequence, belief, config), 8
                    ),
                    "first_action_metrics": {
                        key: round(value, 8)
                        for key, value in immediate_cost(
                            sequence[0], belief, config
                        ).items()
                    },
                }
            )
        selected = (
            min(evaluations, key=lambda item: item["cost"])
            if evaluations
            else {
                "sequence": ["defer"],
                "cost": None,
                "first_action_metrics": {},
            }
        )
    gate = commitment_gate(belief, config)
    return {
        "planner": (
            "discrete_belief_tree_receding_horizon_mpc"
            if planner_mode == "belief_tree_mpc"
            else "hybrid_receding_horizon_engineering_prototype"
        ),
        "planner_mode": planner_mode,
        "horizon": config["horizon"],
        "current_belief": belief,
        "current_entropy_nats": {
            "target": round(entropy(belief["target"]), 8),
            "relation": round(entropy(belief["relation"]), 8),
        },
        "pre_action_forecasts": action_forecasts(belief, config),
        "action_feasibility": {
            name: action_feasibility(name, belief, config)
            for name, action in config["actions"].items()
            if action["enabled"]
        },
        "commitment_gate": gate,
        "candidate_sequences": evaluations,
        "belief_tree_policy": belief_tree,
        "selected_sequence": selected["sequence"],
        "action_request": {
            "type": selected["sequence"][0],
            "reason": (
                (
                    "minimum_non_oracle_expected_horizon_objective_with_"
                    "irreversible_action_safety_constraint"
                )
                if evaluations
                else "no_safe_or_usable_action_remains"
            ),
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
            "actual_mpc_solver": planner_mode == "belief_tree_mpc",
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
