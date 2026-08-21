"""Convert Qwen choice logits into the planner's categorical belief format."""

from __future__ import annotations

import copy
import math

from export_vlm_dataset import ANONYMOUS_IDS


PLANNER_TARGETS = ("target_red", "rear_red_candidate")


def normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(values.values())
    if total <= 0:
        return {key: 1.0 / len(values) for key in values}
    return {key: value / total for key, value in values.items()}


def softmax_temperature(values: list[float], temperature: float) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def qwen_to_planner_belief(output: dict, temperature: float = 4.0) -> dict:
    """Map anonymous candidate and relation logits to planner hypotheses."""
    target_distribution = dict(
        zip(
            output["target"]["candidate_ids"],
            softmax_temperature(output["target"]["raw_logits"], temperature),
        )
    )
    relations = {
        item["query_id"]: dict(
            zip(item["labels"], softmax_temperature(item["raw_logits"], temperature))
        )
        for item in output["relations"]
    }
    epsilon = 1e-9
    target = normalize(
        {
            semantic_id: target_distribution.get(ANONYMOUS_IDS[semantic_id], epsilon)
            for semantic_id in PLANNER_TARGETS
        }
    )
    selected_semantic_id = max(target, key=target.get)
    selected_candidate_id = ANONYMOUS_IDS[selected_semantic_id]
    relation_distribution = relations.get(
        f"{selected_candidate_id}_to_container", {}
    )
    relation = normalize(
        {
            "inside": relation_distribution.get("inside", epsilon),
            "behind": relation_distribution.get("behind", epsilon),
            "unknown": sum(
                relation_distribution.get(label, 0.0)
                for label in ("outside", "near_boundary", "unknown")
            )
            + epsilon,
        }
    )
    return {
        "target": target,
        "relation": relation,
        "selected_relation_query": f"{selected_candidate_id}_to_container",
        "calibrated": False,
        "raw_logit_temperature": temperature,
        "source": "temperature_scaled_qwen_choice_logits",
    }


def weighted_log_belief_update(
    prior: dict, observation: dict, observation_weight: float
) -> dict:
    """Fuse a categorical observation with the prior in log space."""
    if not 0.0 < observation_weight <= 1.0:
        raise ValueError("observation_weight must be in (0, 1]")

    def update_distribution(
        prior_distribution: dict[str, float],
        observation_distribution: dict[str, float],
    ) -> dict[str, float]:
        epsilon = 1e-12
        log_values = {
            key: math.log(max(prior_distribution[key], epsilon))
            + observation_weight * math.log(max(observation_distribution[key], epsilon))
            for key in prior_distribution
        }
        maximum = max(log_values.values())
        return normalize(
            {key: math.exp(value - maximum) for key, value in log_values.items()}
        )

    return {
        "target": update_distribution(prior["target"], observation["target"]),
        "relation": update_distribution(prior["relation"], observation["relation"]),
    }


def prepare_replan_config(
    base_config: dict,
    belief: dict,
    completed_reobservations: int,
    visited_views: set[str],
) -> dict:
    """Insert the current belief and disable already visited viewpoints."""
    config = copy.deepcopy(base_config)
    config["initial_belief"]["target"] = belief["target"]
    config["initial_belief"]["relation"] = belief["relation"]
    config["initial_belief"]["source"] = "qwen_choice_logit_adapter"
    config["completed_reobservations"] = completed_reobservations
    for action_name, action in config["actions"].items():
        if not action_name.startswith("viewpoint_"):
            continue
        view = action_name.removeprefix("viewpoint_")
        enable_after = action.get("enable_after_completed_reobservations")
        if enable_after is not None:
            action["enabled"] = completed_reobservations >= enable_after
        if view in visited_views:
            action["enabled"] = False
            action["disabled_until"] = "new_episode"
    return config
