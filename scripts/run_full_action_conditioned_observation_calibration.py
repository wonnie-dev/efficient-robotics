#!/usr/bin/env python3
"""Validate the saved action-conditioned observation calibration evidence.

This is a calibration-only, CPU-only audit.  It combines the episode-disjoint
view-action cross-validation with physically executed cover-removal outcomes.
Reserved final-test episodes are rejected and never opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "full_action_conditioned_observation_calibration_v1.json"
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fit_likelihood(
    examples: list[dict[str, Any]],
    states: list[str],
    observations: list[str],
    pseudocount: float,
) -> dict[str, dict[str, float]]:
    if pseudocount <= 0:
        raise ValueError("dirichlet_pseudocount must be positive")
    counts = {
        state: {observation: float(pseudocount) for observation in observations}
        for state in states
    }
    for example in examples:
        counts[example["state"]][example["observation"]] += 1.0
    return {
        state: {
            observation: value / sum(counts[state].values())
            for observation, value in counts[state].items()
        }
        for state in states
    }


def posterior(
    observation: str,
    likelihood: dict[str, dict[str, float]],
    states: list[str],
) -> dict[str, float]:
    weights = {state: likelihood[state][observation] for state in states}
    denominator = sum(weights.values())
    return {state: value / denominator for state, value in weights.items()}


def leave_one_episode_out(
    examples: list[dict[str, Any]],
    states: list[str],
    observations: list[str],
    pseudocount: float,
) -> dict[str, Any]:
    folds = []
    for held_out in examples:
        training = [
            example
            for example in examples
            if example["episode_key"] != held_out["episode_key"]
        ]
        likelihood = fit_likelihood(
            training, states, observations, pseudocount
        )
        belief = posterior(held_out["observation"], likelihood, states)
        predicted = max(states, key=lambda state: belief[state])
        folds.append(
            {
                "held_out_episode_key": held_out["episode_key"],
                "held_out_seed": held_out["seed"],
                "state": held_out["state"],
                "observation": held_out["observation"],
                "posterior_with_uniform_state_prior": belief,
                "predicted_state": predicted,
                "correct": predicted == held_out["state"],
            }
        )
    correct = sum(fold["correct"] for fold in folds)
    return {
        "fold_count": len(folds),
        "correct_count": correct,
        "accuracy": correct / len(folds) if folds else None,
        "folds": folds,
    }


def validate_view_model(
    result: dict[str, Any], config: dict[str, Any], reserved: set[int]
) -> dict[str, Any]:
    folds = result["folds"]
    seeds = [int(fold["held_out_seed"]) for fold in folds]
    if len(seeds) != len(set(seeds)):
        raise ValueError("View-model held-out seeds are not episode-disjoint")
    leaked = sorted(set(seeds) & reserved)
    if leaked:
        raise ValueError(f"Reserved final-test seeds used by view model: {leaked}")
    if any(fold["held_out_future_view_used_for_action_selection"] for fold in folds):
        raise ValueError("Held-out future observation leaked into root selection")
    support = Counter(fold["held_out_variant_posthoc"] for fold in folds)
    blockers = []
    if len(folds) < int(config["minimum_view_episode_count"]):
        blockers.append("insufficient_total_view_episodes")
    sparse = {
        variant: count
        for variant, count in support.items()
        if count < int(config["minimum_view_variant_episode_count"])
    }
    if sparse:
        blockers.append("insufficient_view_variant_support")
    correct = sum(bool(fold["correct_action"]) for fold in folds)
    baselines = result["fixed_action_baselines"]
    best_baseline = max(
        (
            {
                "action": action,
                "action_accuracy": float(metrics["action_accuracy"]),
            }
            for action, metrics in baselines.items()
        ),
        key=lambda item: item["action_accuracy"],
    )
    return {
        "episode_count": len(folds),
        "episode_disjoint": True,
        "held_out_future_observation_leakage": False,
        "variant_support": dict(sorted(support.items())),
        "correct_action_count": correct,
        "action_accuracy": correct / len(folds),
        "best_fixed_action_baseline": best_baseline,
        "accuracy_improvement_over_best_fixed_action": (
            correct / len(folds) - best_baseline["action_accuracy"]
        ),
        "selected_view_target_recovery": result[
            "selected_view_target_recovery"
        ],
        "blockers": blockers,
        "candidate_for_freeze": not blockers,
    }


def cover_examples(
    readiness: dict[str, Any], reserved: set[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples = []
    right_after_empty = []
    mapping = {
        "target_inside_after_cover_removal": ("inside", "target_detected"),
        "empty_container_negative_evidence": ("outside", "empty_container"),
    }
    for family, (state, observation) in mapping.items():
        for row in readiness["episodes"].get(family, []):
            seed = int(row["seed"])
            if seed in reserved:
                raise ValueError(f"Reserved final-test seed used: {seed}")
            result_path = resolve_path(row["result_path"])
            result = load_json(result_path)
            if result.get("status") not in {
                "completed",
                "success",
                "successful_closed_loop_negative_evidence_live_development",
            }:
                raise ValueError(
                    f"Calibration result is not successful: {result_path}"
                )
            examples.append(
                {
                    "episode_key": f"{family}:seed{seed}",
                    "seed": seed,
                    "outcome_family": family,
                    "state": state,
                    "observation": observation,
                    "result_path": str(result_path),
                    "runtime_seconds": float(row["runtime_seconds"]),
                }
            )
            if family != "empty_container_negative_evidence":
                continue
            perception = result["learned_post_reobservation_perception"]
            relation = perception["ranking"]["selected_candidate_relation"]
            right_after_empty.append(
                {
                    "episode_key": f"right_after_empty:seed{seed}",
                    "seed": seed,
                    "state": "outside",
                    "observation": relation["top_label"],
                    "selected_candidate_id": perception[
                        "selected_candidate_id"
                    ],
                    "ranking_path": perception["ranking_path"],
                }
            )
    keys = [example["episode_key"] for example in examples]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate cover calibration episode")
    return examples, right_after_empty


def run(config_path: Path) -> dict[str, Any]:
    started = time.monotonic()
    config = load_json(config_path)
    for field in ("training_performed", "testing_performed", "apply_to_mpc"):
        if config.get(field) is not False:
            raise ValueError(f"{field} must remain false during calibration")
    reserved = {int(seed) for seed in config["reserved_test_seeds"]}
    view_path = resolve_path(config["scene_conditioned_view_model_source"])
    readiness_path = resolve_path(config["cover_calibration_readiness_source"])
    view_result = load_json(view_path)
    readiness = load_json(readiness_path)
    view_validation = validate_view_model(
        view_result, config, reserved
    )
    examples, right_examples = cover_examples(readiness, reserved)
    pseudocount = float(config["dirichlet_pseudocount"])
    states = ["inside", "outside"]
    observations = ["target_detected", "empty_container"]
    state_support = Counter(example["state"] for example in examples)
    minimum_cover = int(config["minimum_cover_episode_count_per_outcome"])
    sparse_cover_states = {
        state: state_support.get(state, 0)
        for state in states
        if state_support.get(state, 0) < minimum_cover
    }
    cover_cv = leave_one_episode_out(
        examples, states, observations, pseudocount
    )
    cover_validation = {
        "episode_count": len(examples),
        "state_support": {
            state: state_support.get(state, 0) for state in states
        },
        "observation_support": dict(
            sorted(Counter(x["observation"] for x in examples).items())
        ),
        "minimum_episode_count_per_state": minimum_cover,
        "dirichlet_pseudocount": pseudocount,
        "full_calibration_likelihood_p_observation_given_state": fit_likelihood(
            examples, states, observations, pseudocount
        ),
        "leave_one_episode_out": cover_cv,
        "sparse_states": sparse_cover_states,
        "candidate_for_freeze": not sparse_cover_states,
        "examples": examples,
    }
    right_correct = sum(
        example["observation"] == example["state"]
        for example in right_examples
    )
    right_minimum = int(config["minimum_right_after_empty_episode_count"])
    right_validation = {
        "episode_count": len(right_examples),
        "correct_outside_count": right_correct,
        "outside_relation_accuracy": (
            right_correct / len(right_examples) if right_examples else None
        ),
        "minimum_episode_count": right_minimum,
        "candidate_for_freeze": (
            len(right_examples) >= right_minimum
            and right_correct == len(right_examples)
        ),
        "examples": right_examples,
        "limitation": (
            "This checks saved Qwen relation outputs after the right re-observation; "
            "it is not a calibrated probability or a final-test result."
        ),
    }
    blockers = []
    if not view_validation["candidate_for_freeze"]:
        blockers.extend(view_validation["blockers"])
    if sparse_cover_states:
        blockers.append("insufficient_physical_cover_outcome_support")
    if not right_validation["candidate_for_freeze"]:
        blockers.append("insufficient_right_after_empty_validation")
    result = {
        "schema_version": "full-action-conditioned-observation-calibration-v1",
        "experiment_id": config["experiment_id"],
        "status": "ready_to_freeze" if not blockers else "blocked",
        "source_files": {
            "scene_conditioned_view_model": {
                "path": str(view_path),
                "sha256": sha256(view_path),
            },
            "cover_calibration_readiness": {
                "path": str(readiness_path),
                "sha256": sha256(readiness_path),
            },
        },
        "view_action_validation": view_validation,
        "physical_cover_observation_validation": cover_validation,
        "right_after_empty_validation": right_validation,
        "deployment_gate": {
            "full_action_conditioned_observation_model_frozen": not blockers,
            "apply_to_mpc": False,
            "blockers": blockers,
            "additional_inside_cover_episodes_required": max(
                0, minimum_cover - state_support.get("inside", 0)
            ),
        },
        "gpu_used": False,
        "training_performed": False,
        "foundation_model_weights_changed": False,
        "calibration_performed": True,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
        "runtime_seconds": time.monotonic() - started,
        "limitations": [
            "All inputs are saved calibration artifacts; no new observation was rendered.",
            "The leave-one-episode-out score does not override minimum per-state support.",
            "No score in this file is a calibrated task-success probability.",
            "Reserved final-test seeds 200-209 remain unopened.",
        ],
    }
    write_json_atomic(resolve_path(config["output_path"]), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args()
    result = run(arguments.config.resolve())
    print(json.dumps(result["deployment_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
