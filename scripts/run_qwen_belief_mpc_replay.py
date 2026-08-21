"""Replay captured RGB-D views through Qwen and the future-belief planner.

This is the perception-interface validation between the simulator-label
integration and a later live Isaac/Qwen loop. Candidate observations are
pre-captured, so this runner is never final evaluation evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

from export_vlm_dataset import ANONYMOUS_IDS, export_view
from run_non_oracle_hybrid_planner import plan
from run_single_gpu_pilot import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_MODEL,
    cached_inference,
    output_belief,
    require_single_gpu_policy,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "first_belief_mpc_integration.json"
)
DEFAULT_OBSERVATIONS = ROOT / "outputs" / "benchmark_observations"
DEFAULT_OUTPUT = ROOT / "outputs" / "qwen_belief_mpc_replay"
PLANNER_TARGETS = ("target_red", "rear_red_candidate")


def normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(values.values())
    if total <= 0:
        return {key: 1.0 / len(values) for key in values}
    return {key: value / total for key, value in values.items()}


def softmax_temperature(
    values: list[float], temperature: float
) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def qwen_to_planner_belief(
    output: dict, temperature: float = 4.0
) -> dict:
    """Adapt anonymous Qwen distributions to the two-hypothesis debug belief."""
    target_distribution = dict(
        zip(
            output["target"]["candidate_ids"],
            softmax_temperature(
                output["target"]["raw_logits"], temperature
            ),
        )
    )
    relations = {
        item["query_id"]: dict(
            zip(
                item["labels"],
                softmax_temperature(item["raw_logits"], temperature),
            )
        )
        for item in output["relations"]
    }
    epsilon = 1e-9
    target = normalize(
        {
            semantic_id: target_distribution.get(
                ANONYMOUS_IDS[semantic_id], epsilon
            )
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
        "selected_relation_query": (
            f"{selected_candidate_id}_to_container"
        ),
        "calibrated": False,
        "raw_logit_temperature": temperature,
        "source": "fixed_temperature_qwen_raw_logits_debug_adapter",
    }


def fuse_planner_beliefs(prior: dict, observation: dict) -> dict:
    return weighted_log_belief_update(prior, observation, 1.0)


def weighted_log_belief_update(
    prior: dict, observation: dict, observation_weight: float
) -> dict:
    if not 0.0 < observation_weight <= 1.0:
        raise ValueError("observation_weight must be in (0, 1]")

    def update_distribution(
        prior_distribution: dict[str, float],
        observation_distribution: dict[str, float],
    ) -> dict[str, float]:
        epsilon = 1e-12
        log_values = {
            key: math.log(max(prior_distribution[key], epsilon))
            + observation_weight
            * math.log(max(observation_distribution[key], epsilon))
            for key in prior_distribution
        }
        maximum = max(log_values.values())
        return normalize(
            {
                key: math.exp(value - maximum)
                for key, value in log_values.items()
            }
        )

    return {
        "target": update_distribution(
            prior["target"], observation["target"]
        ),
        "relation": update_distribution(
            prior["relation"], observation["relation"]
        ),
    }


def prepare_replan_config(
    base_config: dict,
    belief: dict,
    completed_reobservations: int,
    visited_views: set[str],
) -> dict:
    config = copy.deepcopy(base_config)
    config["initial_belief"]["target"] = belief["target"]
    config["initial_belief"]["relation"] = belief["relation"]
    config["initial_belief"]["source"] = "pretrained_qwen3_vl_replay_adapter"
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--observation-root", type=Path, default=DEFAULT_OBSERVATIONS
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--allow-cache-miss-inference", action="store_true")
    args = parser.parse_args()
    require_single_gpu_policy()

    output_root = args.output_root.resolve()
    dataset_root = output_root / "vlm_inputs"
    output_root.mkdir(parents=True, exist_ok=True)
    base_config = json.loads(args.config.read_text(encoding="utf-8"))
    adapter_config = base_config["qwen_belief_adapter"]
    temperature = float(adapter_config["raw_logit_temperature"])
    observation_weight = float(adapter_config["observation_log_weight"])
    input_paths = {}
    for view in ("center", "right", "overhead"):
        input_path, _ = export_view(
            view,
            dataset_root,
            observation_root=args.observation_root.resolve(),
            episode_id="benchmark_seed000_qwen_belief_mpc_replay",
        )
        input_paths[view] = input_path

    started = time.perf_counter()
    current_view = "center"
    visited_views = set()
    fused_belief = {
        "target": normalize(base_config["initial_belief"]["target"]),
        "relation": normalize(base_config["initial_belief"]["relation"]),
    }
    steps = []
    terminal_action = None
    for observation_index in range(3):
        visited_views.add(current_view)
        inference = cached_inference(
            input_paths[current_view],
            args.cache_root.resolve(),
            args.model_path.resolve(),
            args.max_pixels,
            args.allow_cache_miss_inference,
        )
        observation_belief = qwen_to_planner_belief(
            inference["output"], temperature
        )
        prior_belief = copy.deepcopy(fused_belief)
        fused_belief = weighted_log_belief_update(
            fused_belief,
            observation_belief,
            observation_weight,
        )
        completed_reobservations = max(0, observation_index)
        planner_config = prepare_replan_config(
            base_config,
            fused_belief,
            completed_reobservations,
            visited_views,
        )
        planner_result = plan(planner_config)
        action = planner_result["action_request"]["type"]
        step = {
            "index": observation_index,
            "view": current_view,
            "input": str(input_paths[current_view].relative_to(ROOT)),
            "cache_key": inference["cache_key"],
            "cache_dir": inference["cache_dir"],
            "cache_hit": inference["cache_hit"],
            "cache_source": inference["cache_source"],
            "metrics": inference["metrics"],
            "qwen_observation_belief": observation_belief,
            "belief_before_update": prior_belief,
            "fused_belief": fused_belief,
            "belief_update": {
                "method": "weighted_log_space_generalized_bayes_debug_update",
                "raw_logit_temperature": temperature,
                "observation_log_weight": observation_weight,
                "calibrated": False,
            },
            "planner": planner_result,
            "selected_action": action,
        }
        steps.append(step)
        (output_root / f"step_{observation_index:03d}.json").write_text(
            json.dumps(step, indent=2) + "\n", encoding="utf-8"
        )
        if not action.startswith("viewpoint_"):
            terminal_action = action
            break
        next_view = action.removeprefix("viewpoint_")
        if next_view not in input_paths:
            raise RuntimeError(f"Planner requested unavailable replay view: {next_view}")
        current_view = next_view

    result = {
        "schema_version": "qwen-belief-mpc-replay-v1",
        "status": "completed",
        "purpose": "pretrained_qwen_perception_interface_debug_only",
        "steps": steps,
        "terminal_action": terminal_action,
        "runtime_seconds": time.perf_counter() - started,
        "training_performed": False,
        "calibration_performed": False,
        "pre_captured_observation_replay": True,
        "simulator_ground_truth_consumed_during_planning": False,
        "candidate_tracker_mapping_source": (
            "debug_simulator_anonymous_id_adapter_not_ground_truth_labels_in_prompt"
        ),
        "gpu_policy": {
            "physical_gpu": 5,
            "batch_size": 1,
            "distributed": False,
            "parallel_vlm_jobs": False,
        },
        "valid_for_final_evaluation": False,
        "limitations": [
            "Fixed-temperature logits and weighted log updates are not fitted calibration.",
            "Candidate images were captured before this replay run.",
            "The overhead camera pose is synthetic debug geometry.",
            "No UR10e trajectory or RG6 grasp is executed by this runner.",
        ],
    }
    result_path = output_root / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"STATUS={result['status']}")
    print(f"TERMINAL_ACTION={terminal_action}")
    print(f"OBSERVATIONS={len(steps)}")
    print(f"WROTE={result_path}")


if __name__ == "__main__":
    main()
