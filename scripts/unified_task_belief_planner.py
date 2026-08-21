#!/usr/bin/env python3
"""Finite-horizon planner over a semantic belief and observable task state.

The semantic belief contains only latent hypotheses such as persistent target
identity and container membership.  Physical task state, such as whether a
cover is open, is tracked separately when it is directly observed after an
action.  Information actions and terminal actions are compared in one
expected-cost recursion; there is no fixed confidence threshold.
"""

from __future__ import annotations

import math
from typing import Any


FORBIDDEN_FIXED_GATE_KEYS = {
    "commitment_threshold",
    "grasp_confidence_threshold",
    "minimum_grasp_confidence",
    "minimum_grasp_success_probability",
    "sufficient_score_threshold",
}
REQUIRED_INFORMATION_ACTIONS = {
    "remove_cover",
    "viewpoint_close_high",
    "viewpoint_right",
}


def _nested_keys(value: Any) -> set[str]:
    """Collect mapping keys recursively for configuration validation."""
    if isinstance(value, dict):
        return set(value) | {
            key
            for nested in value.values()
            for key in _nested_keys(nested)
        }
    if isinstance(value, list):
        return {key for nested in value for key in _nested_keys(nested)}
    return set()


def normalize(distribution: dict[str, float]) -> dict[str, float]:
    """Normalize non-negative probability mass to a categorical belief."""
    values = {str(key): float(value) for key, value in distribution.items()}
    if not values or any(value < 0.0 for value in values.values()):
        raise ValueError("A belief must contain non-negative probability mass")
    total = sum(values.values())
    if total <= 0.0:
        raise ValueError("A belief cannot have zero total mass")
    return {key: value / total for key, value in values.items()}


def validate_model(model: dict[str, Any]) -> None:
    """Check probability, action, transition, and cost model invariants."""
    hypotheses = tuple(str(value) for value in model["semantic_hypotheses"])
    task_states = set(str(value) for value in model["observable_task_states"])
    if not hypotheses or not task_states:
        raise ValueError("The model requires semantic hypotheses and task states")
    if set(model["initial_semantic_belief"]) != set(hypotheses):
        raise ValueError("Initial semantic belief must cover every hypothesis")
    initial = normalize(model["initial_semantic_belief"])
    if any(
        not math.isclose(initial[key], float(model["initial_semantic_belief"][key]))
        for key in hypotheses
    ):
        raise ValueError("Initial semantic belief must sum to one")
    if model["initial_task_state"] not in task_states:
        raise ValueError("Initial task state is not declared")
    encoded_task_states = {
        hypothesis
        for hypothesis in hypotheses
        if set(hypothesis.split("|")) & task_states
    }
    if encoded_task_states:
        raise ValueError(
            "Observable task state must not be encoded in semantic hypotheses: "
            f"{sorted(encoded_task_states)}"
        )
    forbidden = _nested_keys(model) & FORBIDDEN_FIXED_GATE_KEYS
    if forbidden:
        raise ValueError(
            "Fixed confidence gates are not part of the task-cost planner: "
            f"{sorted(forbidden)}"
        )

    for action_name, action in model["information_actions"].items():
        allowed = set(action["allowed_task_states"])
        if not allowed or not allowed <= task_states:
            raise ValueError(f"{action_name} has invalid allowed task states")
        outcomes = tuple(str(value) for value in action["outcomes"])
        if not outcomes:
            raise ValueError(f"{action_name} has no observation outcomes")
        for task_state in allowed:
            likelihood = action["observation_likelihood"][task_state]
            if set(likelihood) != set(hypotheses):
                raise ValueError(
                    f"{action_name}/{task_state} likelihood must cover hypotheses"
                )
            for hypothesis, row in likelihood.items():
                if set(row) != set(outcomes):
                    raise ValueError(
                        f"{action_name}/{task_state}/{hypothesis} has invalid outcomes"
                    )
                if any(float(value) < 0.0 for value in row.values()) or not math.isclose(
                    sum(float(value) for value in row.values()), 1.0
                ):
                    raise ValueError(
                        f"{action_name}/{task_state}/{hypothesis} must sum to one"
                    )
            transitions = action["next_task_state_by_outcome"][task_state]
            if set(transitions) != set(outcomes):
                raise ValueError(
                    f"{action_name}/{task_state} transitions must cover outcomes"
                )
            if not set(transitions.values()) <= task_states:
                raise ValueError(f"{action_name} transitions to an unknown task state")

    costs = model["costs"]
    for key in ("wrong_commitment", "execution_failure", "defer"):
        if key not in costs or float(costs[key]) < 0.0:
            raise ValueError(f"Missing or invalid task cost: {key}")
    for action_name, action in model["terminal_grasp_actions"].items():
        if action["semantic_hypothesis"] not in hypotheses:
            raise ValueError(f"{action_name} refers to an unknown hypothesis")
        if not set(action["allowed_task_states"]) <= task_states:
            raise ValueError(f"{action_name} has invalid allowed task states")
        execution = float(action["conditional_execution_success_probability"])
        if not 0.0 <= execution <= 1.0:
            raise ValueError(f"{action_name} has invalid execution success")


def validate_unified_method_contract(model: dict[str, Any]) -> None:
    """Validate the paper method contract, not just JSON consistency."""
    validate_model(model)
    hypotheses = set(str(value) for value in model["semantic_hypotheses"])
    if "target_absent|not_applicable" not in hypotheses:
        raise ValueError("The semantic state requires an explicit target-absent hypothesis")
    target_present = hypotheses - {"target_absent|not_applicable"}
    grasp_hypotheses = {
        str(action["semantic_hypothesis"])
        for action in model["terminal_grasp_actions"].values()
    }
    if grasp_hypotheses != target_present:
        raise ValueError(
            "Terminal grasps must cover every and only target-present hypothesis"
        )
    information_actions = set(model["information_actions"])
    missing = REQUIRED_INFORMATION_ACTIONS - information_actions
    if missing:
        raise ValueError(
            "The unified action space is missing required actions: "
            f"{sorted(missing)}"
        )


def observation_distribution(
    belief: dict[str, float],
    task_state: str,
    action: dict[str, Any],
) -> dict[str, float]:
    """Marginalize the action likelihood over the current semantic belief."""
    belief = normalize(belief)
    likelihood = action["observation_likelihood"][task_state]
    return {
        outcome: sum(
            belief[hypothesis] * float(likelihood[hypothesis][outcome])
            for hypothesis in belief
        )
        for outcome in action["outcomes"]
    }


def update_semantic_belief(
    belief: dict[str, float],
    task_state: str,
    action: dict[str, Any],
    outcome: str,
) -> dict[str, float]:
    """Apply a discrete observation likelihood with a Bayesian update."""
    belief = normalize(belief)
    if outcome not in action["outcomes"]:
        raise ValueError(f"Unknown observation outcome: {outcome}")
    likelihood = action["observation_likelihood"][task_state]
    return normalize(
        {
            hypothesis: belief[hypothesis]
            * float(likelihood[hypothesis][outcome])
            for hypothesis in belief
        }
    )


def update_belief_and_task_state(
    belief: dict[str, float],
    task_state: str,
    action: dict[str, Any],
    outcome: str,
) -> tuple[dict[str, float], str]:
    """Apply one observation and its explicitly declared physical transition."""
    posterior = update_semantic_belief(belief, task_state, action, outcome)
    return posterior, next_task_state(action, task_state, outcome)


def next_task_state(
    action: dict[str, Any], task_state: str, outcome: str
) -> str:
    """Return the measured-action transition declared by the model."""
    try:
        return str(action["next_task_state_by_outcome"][task_state][outcome])
    except KeyError as error:
        raise ValueError(
            f"No task-state transition for {task_state}/{outcome}"
        ) from error


def terminal_values(
    belief: dict[str, float], task_state: str, model: dict[str, Any]
) -> list[dict[str, Any]]:
    """Evaluate feasible grasp commitments and the safe defer action."""
    belief = normalize(belief)
    costs = model["costs"]
    values: list[dict[str, Any]] = []
    for action_name, action in model["terminal_grasp_actions"].items():
        if task_state not in action["allowed_task_states"]:
            continue
        hypothesis = str(action["semantic_hypothesis"])
        semantic_success = belief[hypothesis]
        execution_success = float(
            action["conditional_execution_success_probability"]
        )
        wrong_commitment = 1.0 - semantic_success
        execution_failure = semantic_success * (1.0 - execution_success)
        expected_cost = (
            float(action["stage_cost"])
            + float(costs["wrong_commitment"]) * wrong_commitment
            + float(costs["execution_failure"]) * execution_failure
        )
        values.append(
            {
                "action": action_name,
                "kind": "terminal_grasp",
                "expected_cost": expected_cost,
                "semantic_hypothesis": hypothesis,
                "semantic_success_probability": semantic_success,
                "wrong_commitment_risk": wrong_commitment,
                "conditional_execution_success_probability": execution_success,
                "execution_failure_risk": execution_failure,
                "fixed_confidence_threshold_used": False,
            }
        )
    values.append(
        {
            "action": "defer",
            "kind": "terminal_defer",
            "expected_cost": float(costs["defer"]),
            "fixed_confidence_threshold_used": False,
        }
    )
    return values


def _plan(
    belief: dict[str, float],
    task_state: str,
    model: dict[str, Any],
    remaining_actions: tuple[str, ...],
    depth: int,
) -> dict[str, Any]:
    """Evaluate the finite-horizon feedback tree from one belief state."""
    values = terminal_values(belief, task_state, model)
    if depth > 0:
        for action_name in remaining_actions:
            action = model["information_actions"][action_name]
            if task_state not in action["allowed_task_states"]:
                continue
            outcome_probabilities = observation_distribution(
                belief, task_state, action
            )
            branches = []
            expected_future_cost = 0.0
            for outcome, probability in outcome_probabilities.items():
                if probability <= 0.0:
                    continue
                posterior = update_semantic_belief(
                    belief, task_state, action, outcome
                )
                next_task_state = action["next_task_state_by_outcome"][task_state][
                    outcome
                ]
                continuation = _plan(
                    posterior,
                    next_task_state,
                    model,
                    tuple(value for value in remaining_actions if value != action_name),
                    depth - 1,
                )
                expected_future_cost += (
                    float(probability) * continuation["selected_expected_cost"]
                )
                branches.append(
                    {
                        "observation": outcome,
                        "probability": probability,
                        "posterior": posterior,
                        "next_task_state": next_task_state,
                        "continuation_action": continuation["selected_action"],
                        "continuation_cost": continuation[
                            "selected_expected_cost"
                        ],
                    }
                )
            values.append(
                {
                    "action": action_name,
                    "kind": str(action["kind"]),
                    "expected_cost": float(action["stage_cost"])
                    + expected_future_cost,
                    "stage_cost": float(action["stage_cost"]),
                    "expected_future_cost": expected_future_cost,
                    "observation_branches": branches,
                    "branch_probability_sum": sum(
                        branch["probability"] for branch in branches
                    ),
                    "future_held_out_observation_used": False,
                }
            )
    selected = min(values, key=lambda item: (item["expected_cost"], item["action"]))
    return {
        "planner": "unified_semantic_belief_observable_task_state_mpc",
        "semantic_belief": normalize(belief),
        "observable_task_state": task_state,
        "horizon": depth,
        "feedback_policy": True,
        "action_values": values,
        "selected_action": selected["action"],
        "selected_expected_cost": selected["expected_cost"],
        "fixed_confidence_threshold_used": False,
        "marginal_confidence_product_used": False,
        "future_held_out_observation_used_for_action_selection": False,
    }


def plan(
    belief: dict[str, float],
    task_state: str,
    model: dict[str, Any],
    *,
    horizon: int | None = None,
    remaining_actions: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Validate the model and return the minimum expected-cost root action."""
    validate_unified_method_contract(model)
    if task_state not in model["observable_task_states"]:
        raise ValueError(f"Unknown observable task state: {task_state}")
    actions = (
        tuple(model["information_actions"])
        if remaining_actions is None
        else remaining_actions
    )
    unknown = set(actions) - set(model["information_actions"])
    if unknown:
        raise ValueError(f"Unknown information actions: {sorted(unknown)}")
    depth = int(model["horizon"] if horizon is None else horizon)
    if depth < 0:
        raise ValueError("Planning horizon must be non-negative")
    return _plan(normalize(belief), task_state, model, actions, depth)
