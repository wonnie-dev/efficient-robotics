"""Runtime adapters for the calibrated perception--belief--MPC loop."""

from __future__ import annotations

import math
from typing import Any

from run_cover_search_belief_mpc import marginal_location


def sigmoid(value: float) -> float:
    """Compute a numerically stable logistic transform."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def calibrated_target_identity(perception: dict, temperature: float) -> dict:
    """Calibrate the selected candidate's binary identity-match logit."""
    if temperature <= 0.0:
        raise ValueError("Target calibration temperature must be positive")
    ranking = perception["ranking"]
    selected_id = ranking["selected_candidate_id"]
    selected_index = ranking["candidate_ids"].index(selected_id)
    raw_logit = float(ranking["raw_match_logits"][selected_index])
    return {
        "selected_candidate_id": selected_id,
        "raw_match_logit": raw_logit,
        "temperature": float(temperature),
        "probability": sigmoid(raw_logit / temperature),
        "method": "binary_temperature_scaling",
        "calibrated": True,
    }


def relation_observation_from_audit(relation_audit: dict) -> dict:
    """Map learned RGB-D membership evidence to the planner vocabulary.

    RGB-D membership is the calibrated observation channel.  Qwen is kept as
    semantic corroboration and disagreements are recorded instead of being
    silently converted into a confident relation label.
    """
    qwen_label = relation_audit.get("qwen_relation_top_label")
    rgbd_label = (
        relation_audit.get("rgbd_relation", {})
        .get("membership_world_evidence", {})
        .get("label")
    )
    mapping = {
        "inside": "inside_evidence",
        "outside": "outside_evidence",
        "unknown": "unknown_evidence",
    }
    if rgbd_label not in mapping:
        raise RuntimeError(f"Missing RGB-D membership evidence: {rgbd_label!r}")
    return {
        "planner_observation": mapping[rgbd_label],
        "rgbd_membership_label": rgbd_label,
        "qwen_membership_label": qwen_label,
        "qwen_rgbd_agree": qwen_label == rgbd_label,
        "calibrated_channel": "learned_mask_rgbd_membership",
        "simulator_ground_truth_used": False,
    }


def scene_graph_snapshot(
    *,
    step: int,
    view: str,
    identity: dict,
    belief: dict[str, float],
    relation_evidence: dict,
) -> dict[str, Any]:
    """Serialize the object node and probabilistic membership edge."""
    location = marginal_location(belief)
    return {
        "schema_version": "probabilistic-scene-graph-runtime-v1",
        "step": int(step),
        "view": view,
        "nodes": [
            {
                "node_id": identity["selected_candidate_id"],
                "node_type": "target_candidate",
                "target_identity_probability": identity["probability"],
                "target_identity_calibrated": True,
            },
            {"node_id": "container_001", "node_type": "container"},
        ],
        "relation_edges": [
            {
                "source": identity["selected_candidate_id"],
                "target": "container_001",
                "relation": "membership",
                "probabilities": {
                    "inside": location["inside"],
                    "outside": location["outside_near"],
                },
                "latest_observation": relation_evidence,
            }
        ],
    }


def joint_scene_graph_snapshot(
    *,
    step: int,
    view: str,
    joint_belief: dict[str, float],
    candidate_track_ids: list[str],
    observation: dict[str, Any],
) -> dict[str, Any]:
    """Serialize a Scene Graph backed by one normalized joint belief.

    Candidate IDs must be persistent RGB-D track IDs.  Per-view detector IDs
    such as ``candidate_001`` are deliberately rejected because their meaning
    changes whenever GroundingDINO returns proposals in a different order.
    """
    if not candidate_track_ids:
        raise ValueError("At least one persistent candidate track is required")
    if len(candidate_track_ids) != len(set(candidate_track_ids)):
        raise ValueError("Candidate track IDs must be unique")
    if any(not track_id.startswith("track_") for track_id in candidate_track_ids):
        raise ValueError("Joint Scene Graph candidates must use persistent track IDs")

    memberships = ("inside", "outside")
    for track_id in candidate_track_ids:
        for membership in memberships:
            joint_hypothesis_probability(
                joint_belief,
                candidate_id=track_id,
                membership=membership,
            )

    nodes = [
        {"node_id": track_id, "node_type": "target_candidate_track"}
        for track_id in candidate_track_ids
    ]
    nodes.append({"node_id": "container_001", "node_type": "container"})
    edges = []
    candidate_marginals = {}
    for track_id in candidate_track_ids:
        probabilities = {
            membership: joint_belief[f"{track_id}|{membership}"]
            for membership in memberships
        }
        candidate_marginals[track_id] = sum(probabilities.values())
        edges.append(
            {
                "source": track_id,
                "target": "container_001",
                "relation": "target_identity_and_membership",
                "joint_probabilities": probabilities,
            }
        )
    return {
        "schema_version": "probabilistic-scene-graph-joint-runtime-v1",
        "step": int(step),
        "view": view,
        "nodes": nodes,
        "relation_edges": edges,
        "candidate_target_marginals": candidate_marginals,
        "joint_belief": dict(joint_belief),
        "joint_belief_normalized": True,
        "marginal_confidence_product_used": False,
        "candidate_identity_source": "rgbd_persistent_track",
        "latest_observation": observation,
    }


def calibrated_commitment_gate(
    *,
    terminal_action: str,
    posterior: dict[str, float],
    identity: dict,
    minimum_probability: float,
) -> dict:
    """Reproduce the legacy V11 factor-product gate.

    This function is retained only so that historical V11 runs remain
    reproducible.  Multiplying two marginal confidence values is not, by
    itself, a calibrated joint probability.  New experiments must use
    ``expected_cost_commitment_decision`` with a direct joint belief.
    """
    if terminal_action not in {"grasp_inside", "grasp_outside"}:
        return {
            "authorized": False,
            "reason": "planner_did_not_select_grasp",
            "terminal_action": terminal_action,
        }
    location = marginal_location(posterior)
    location_probability = (
        location["inside"]
        if terminal_action == "grasp_inside"
        else location["outside_near"]
    )
    identity_probability = float(identity["probability"])
    joint_probability = identity_probability * location_probability
    return {
        "authorized": joint_probability >= minimum_probability,
        "reason": (
            "joint_commitment_probability_passed"
            if joint_probability >= minimum_probability
            else "joint_commitment_probability_below_threshold"
        ),
        "terminal_action": terminal_action,
        "identity_probability": identity_probability,
        "location_probability": location_probability,
        "joint_commitment_probability": joint_probability,
        "minimum_probability": float(minimum_probability),
        "combination_assumption": "identity_and_location_factor_product",
        "calibrated_inputs_used": True,
        "joint_probability_calibrated": False,
        "calibrated_probability_used": False,
        "protocol_status": "legacy_v11_reproduction_only",
    }


def joint_hypothesis_probability(
    joint_belief: dict[str, float],
    *,
    candidate_id: str,
    membership: str,
    tolerance: float = 1e-6,
) -> float:
    """Return P(candidate, membership | observations) from one joint belief.

    Keys use ``<candidate_id>|<membership>``.  The distribution is validated
    here so a caller cannot silently pass independent marginal scores.
    """
    if not joint_belief:
        raise ValueError("Joint belief must not be empty")
    values = [float(value) for value in joint_belief.values()]
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Joint belief probabilities must lie in [0, 1]")
    if abs(sum(values) - 1.0) > tolerance:
        raise ValueError("Joint belief probabilities must sum to one")
    key = f"{candidate_id}|{membership}"
    if key not in joint_belief:
        raise KeyError(f"Missing joint hypothesis: {key}")
    return float(joint_belief[key])


def update_joint_hypothesis_belief(
    prior: dict[str, float],
    joint_observation_likelihood: dict[str, float],
) -> dict[str, float]:
    """Apply one Bayes update over candidate-by-membership hypotheses.

    The observation model must provide one likelihood per joint hypothesis.
    This interface deliberately does not accept target and relation marginals,
    preventing an unverified independence product inside the belief update.
    """
    if set(prior) != set(joint_observation_likelihood):
        missing = set(prior) - set(joint_observation_likelihood)
        extra = set(joint_observation_likelihood) - set(prior)
        raise ValueError(
            "Joint likelihood keys must match the prior: "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    # Reuse the distribution checks before applying the observation.
    first_key = next(iter(prior))
    candidate_id, membership = first_key.rsplit("|", 1)
    joint_hypothesis_probability(
        prior,
        candidate_id=candidate_id,
        membership=membership,
    )
    likelihood = {
        key: float(value) for key, value in joint_observation_likelihood.items()
    }
    if any(value < 0.0 for value in likelihood.values()):
        raise ValueError("Joint observation likelihoods must be non-negative")
    weights = {key: float(prior[key]) * likelihood[key] for key in prior}
    evidence = sum(weights.values())
    if evidence <= 0.0:
        raise ValueError("Joint observation has zero probability under the prior")
    return {key: value / evidence for key, value in weights.items()}


def expected_cost_commitment_decision(
    *,
    terminal_action: str,
    candidate_id: str,
    membership: str,
    joint_belief: dict[str, float],
    conditional_execution_success_probability: float,
    costs: dict[str, float],
    alternative_action_values: list[dict[str, Any]],
    maximum_wrong_commitment_risk: float | None = None,
) -> dict[str, Any]:
    """Compare grasping with information-gathering and deferral by task cost.

    The semantic success term is read directly from a normalized joint belief
    over candidate identity and membership.  Execution success is explicitly
    conditional on the semantic hypothesis being correct, so its product with
    semantic success follows the probability chain rule rather than an
    independence assumption.
    """
    if terminal_action not in {"grasp_inside", "grasp_outside"}:
        return {
            "authorized": False,
            "selected_action": terminal_action,
            "reason": "planner_did_not_select_grasp",
            "decision_rule": "minimum_expected_task_cost",
        }
    if membership not in {"inside", "outside"}:
        raise ValueError(f"Unsupported terminal membership: {membership}")
    execution_success = float(conditional_execution_success_probability)
    if execution_success < 0.0 or execution_success > 1.0:
        raise ValueError("Execution success probability must lie in [0, 1]")
    required_costs = {
        "grasp",
        "wrong_commitment",
        "execution_failure",
    }
    missing = required_costs - set(costs)
    if missing:
        raise KeyError(f"Missing task costs: {sorted(missing)}")

    semantic_success = joint_hypothesis_probability(
        joint_belief,
        candidate_id=candidate_id,
        membership=membership,
    )
    wrong_commitment_risk = 1.0 - semantic_success
    execution_failure_risk = semantic_success * (1.0 - execution_success)
    grasp_expected_cost = (
        float(costs["grasp"])
        + float(costs["wrong_commitment"]) * wrong_commitment_risk
        + float(costs["execution_failure"]) * execution_failure_risk
    )

    action_values = [
        {
            "action": terminal_action,
            "kind": "terminal_grasp",
            "expected_cost": grasp_expected_cost,
        }
    ]
    for value in alternative_action_values:
        action_values.append(
            {
                "action": str(value["action"]),
                "kind": str(value.get("kind", "alternative")),
                "expected_cost": float(value["expected_cost"]),
            }
        )

    risk_cap_passed = (
        maximum_wrong_commitment_risk is None
        or wrong_commitment_risk <= float(maximum_wrong_commitment_risk)
    )
    feasible_values = (
        action_values
        if risk_cap_passed
        else action_values[1:]
    )
    if not feasible_values:
        raise ValueError("At least one non-grasp alternative is required")
    selected = min(
        feasible_values,
        key=lambda value: (value["expected_cost"], value["action"]),
    )
    authorized = selected["action"] == terminal_action
    return {
        "authorized": authorized,
        "selected_action": selected["action"],
        "reason": (
            "grasp_minimizes_expected_task_cost"
            if authorized
            else (
                "wrong_commitment_risk_cap_exceeded"
                if not risk_cap_passed
                else "alternative_has_lower_expected_task_cost"
            )
        ),
        "decision_rule": "minimum_expected_task_cost",
        "candidate_id": candidate_id,
        "membership": membership,
        "semantic_success_probability": semantic_success,
        "wrong_commitment_risk": wrong_commitment_risk,
        "conditional_execution_success_probability": execution_success,
        "execution_failure_risk": execution_failure_risk,
        "grasp_expected_cost": grasp_expected_cost,
        "maximum_wrong_commitment_risk": maximum_wrong_commitment_risk,
        "risk_cap_passed": risk_cap_passed,
        "action_values": action_values,
        "marginal_product_used": False,
        "joint_belief_required": True,
    }


def next_observation_after_rejected_commitment(policy: dict) -> dict:
    """Select the lowest-cost sensing action after a grasp constraint fails."""
    candidates = [
        value
        for value in policy["action_values"]
        if value.get("kind") == "observation"
    ]
    if not candidates:
        raise RuntimeError("No observation action remains after grasp rejection")
    selected = min(candidates, key=lambda value: float(value["cost"]))
    return {
        "selected_action": selected["action"],
        "selected_cost": float(selected["cost"]),
        "selection_reason": "grasp_infeasible_under_calibrated_commitment_constraint",
        "source_planner": policy["planner"],
        "source_horizon": int(policy["horizon"]),
        "candidate_observation_actions": [
            {"action": value["action"], "cost": float(value["cost"])}
            for value in candidates
        ],
        "future_observation_used_for_selection": False,
    }
