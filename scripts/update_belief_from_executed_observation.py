"""Update belief after execution using the actual new observation.

This adapter consumes a post-action simulator instance-label result. It is
allowed only after the action has executed and is replaceable by real
VLM/grounding inference.
"""

from calibrated_belief import (
    bayesian_update,
    binary_detection_likelihood,
    entropy,
)


def bbox_center_inside(inner: list[int] | None, outer: list[int] | None) -> bool:
    if inner is None or outer is None:
        return False
    center_x = (inner[0] + inner[2]) / 2.0
    center_y = (inner[1] + inner[3]) / 2.0
    return (
        outer[0] <= center_x <= outer[2]
        and outer[1] <= center_y <= outer[3]
    )


def observed_symbols(objects: dict, adapter: dict) -> dict:
    target = objects[adapter["target_object_id"]]
    container = objects[adapter["container_id"]]
    detected = target["pixel_count"] >= adapter["detection_pixel_threshold"]
    if not detected:
        relation_outcome = "unknown_evidence"
    elif bbox_center_inside(target["bbox_xyxy"], container["bbox_xyxy"]):
        relation_outcome = "inside_evidence"
    else:
        relation_outcome = "outside_evidence"
    return {
        "target_detected": detected,
        "relation_outcome": relation_outcome,
        "target_pixel_count": target["pixel_count"],
    }


def update_from_observation(
    prior: dict,
    action_name: str,
    objects: dict,
    config: dict,
) -> dict:
    symbols = observed_symbols(objects, config["post_action_observation_adapter"])
    model = config["observation_model"][action_name]
    target_likelihood = binary_detection_likelihood(
        model["target_detection_probability"], symbols["target_detected"]
    )
    relation_likelihood = {
        hypothesis: outcomes[symbols["relation_outcome"]]
        for hypothesis, outcomes in model["relation_likelihood"].items()
    }
    posterior = {
        "target": bayesian_update(prior["target"], target_likelihood),
        "relation": bayesian_update(prior["relation"], relation_likelihood),
    }
    return {
        "action": action_name,
        "observation_symbols": symbols,
        "prior": prior,
        "likelihood": {
            "target": target_likelihood,
            "relation": relation_likelihood,
        },
        "posterior": posterior,
        "entropy_nats": {
            "target_before": entropy(prior["target"]),
            "target_after": entropy(posterior["target"]),
            "relation_before": entropy(prior["relation"]),
            "relation_after": entropy(posterior["relation"]),
        },
        "provenance": {
            "observation_timing": "after_action_execution",
            "future_observation_used_during_planning": False,
            "perception_source": "simulator_instance_label_adapter",
            "valid_for_real_robot_or_final_evaluation": False,
        },
    }
