"""CPU-only nested calibration for the cached discrete belief-MPC replay.

The outer fold is never used to select the task noncompletion cost or fit the
action model.  Cost selection happens through leave-one-episode-out replay
inside the remaining outer-training episodes.  This is still development
cross-validation over the existing calibration collection, not a frozen test
or a robot motion-MPC experiment.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_offline_action_conditioned_mpc_replay import (  # noqa: E402
    build_episode_rows,
    fit_action_model,
    replay_mpc,
    resolve_path,
    summarize_method,
)


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "nested_action_conditioned_mpc_calibration_seed165_184.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_outer_folds(
    episodes: dict[int, dict[str, dict]],
    folds: list[dict[str, Any]],
) -> None:
    """Require disjoint outer folds that cover every calibration episode once."""
    expected = set(episodes)
    held_out = [
        int(seed)
        for fold in folds
        for seed in fold["held_out_seeds"]
    ]
    if len(held_out) != len(set(held_out)):
        raise ValueError("Outer held-out folds overlap")
    if set(held_out) != expected:
        raise ValueError(
            f"Outer folds cover {sorted(set(held_out))}, expected "
            f"{sorted(expected)}"
        )


def load_fold_target_temperatures(
    nested_config: dict[str, Any],
) -> dict[int, float]:
    """Match each temperature to the exact held-out seeds of its outer fold."""
    calibration_path = nested_config.get("target_calibration_result")
    if calibration_path is None:
        return {}
    calibration = load_json(resolve_path(calibration_path))
    target = calibration["components"]["target_identity"]
    calibration_folds = {
        int(fold["fold_id"]): fold for fold in target["folds"]
    }
    result = {}
    for declared in nested_config["outer_folds"]:
        fold_id = int(declared["fold_id"])
        if fold_id not in calibration_folds:
            raise ValueError(
                f"Target calibration lacks outer fold {fold_id}"
            )
        source = calibration_folds[fold_id]
        declared_seeds = {int(seed) for seed in declared["held_out_seeds"]}
        source_seeds = {int(seed) for seed in source["held_out_seeds"]}
        if declared_seeds != source_seeds:
            raise ValueError(
                f"Target calibration fold {fold_id} held-out seeds differ"
            )
        temperature = float(source["fitted_temperature"])
        if temperature <= 0.0:
            raise ValueError(
                f"Invalid target temperature in fold {fold_id}"
            )
        result[fold_id] = temperature
    return result


def replay_loss(
    row: dict[str, Any],
    loss_config: dict[str, float],
) -> float:
    """Score a replay using task outcomes plus the cost of extra observations."""
    extra_observations = max(0, int(row["observation_count"]) - 1)
    return (
        float(loss_config["wrong_grasp"]) * int(row["wrong_grasp"])
        + float(loss_config["missed_visible_target"])
        * int(row["missed_visible_target"])
        + float(loss_config["safe_defer_fully_hidden"])
        * int(row["safe_defer_fully_hidden"])
        + float(loss_config["correct_grasp"])
        * int(row["correct_grasp"])
        + float(loss_config["extra_observation"]) * extra_observations
    )


def with_noncompletion_cost(
    base_config: dict[str, Any],
    value: float,
) -> dict[str, Any]:
    selected = copy.deepcopy(base_config)
    selected["task_cost"]["task_noncompletion_cost"] = float(value)
    return selected


def tune_noncompletion_cost(
    training_episodes: dict[int, dict[str, dict]],
    base_config: dict[str, Any],
    nested_config: dict[str, Any],
    *,
    action_agnostic: bool,
) -> dict[str, Any]:
    """Tune inside the outer-training split with leave-one-episode-out replay."""
    candidates = []
    training_seeds = sorted(training_episodes)
    method = (
        "action_agnostic_belief_mpc"
        if action_agnostic
        else "action_conditioned_belief_mpc"
    )
    for value in nested_config["candidate_task_noncompletion_costs"]:
        candidate_config = with_noncompletion_cost(base_config, float(value))
        validation_rows = []
        for validation_seed in training_seeds:
            inner_training = {
                seed: episode
                for seed, episode in training_episodes.items()
                if seed != validation_seed
            }
            model = fit_action_model(
                inner_training,
                candidate_config,
                action_agnostic=action_agnostic,
            )
            replay = replay_mpc(
                method,
                validation_seed,
                training_episodes[validation_seed],
                model,
                candidate_config,
            )
            replay["inner_validation_seed"] = validation_seed
            replay["inner_fit_seeds"] = sorted(inner_training)
            replay["inner_validation_seed_used_for_fit"] = (
                validation_seed in inner_training
            )
            replay["selection_loss"] = replay_loss(
                replay,
                nested_config["inner_selection_loss"],
            )
            validation_rows.append(replay)
        total_loss = sum(
            float(row["selection_loss"]) for row in validation_rows
        )
        summary = summarize_method(validation_rows)
        candidates.append(
            {
                "task_noncompletion_cost": float(value),
                "total_selection_loss": total_loss,
                "mean_selection_loss": total_loss / len(validation_rows),
                "summary": summary,
                "validation_rows": validation_rows,
            }
        )
    selected = min(
        candidates,
        key=lambda item: (
            item["mean_selection_loss"],
            item["summary"]["wrong_grasp_count"],
            item["summary"]["missed_visible_target_count"],
            item["summary"]["mean_observation_count"],
            item["task_noncompletion_cost"],
        ),
    )
    return {
        "method": method,
        "inner_training_seeds": training_seeds,
        "candidate_results": candidates,
        "selected_task_noncompletion_cost": selected[
            "task_noncompletion_cost"
        ],
        "selection_key": [
            "mean_selection_loss",
            "wrong_grasp_count",
            "missed_visible_target_count",
            "mean_observation_count",
            "lower_task_noncompletion_cost",
        ],
        "outer_held_out_data_used_for_selection": False,
    }


def action_difference_audit(
    episodes: dict[int, dict[str, dict]],
    nested_config: dict[str, Any],
) -> dict[str, Any]:
    """Separate perceptual variation from physically different action outcomes."""
    fields = [
        str(field)
        for field in nested_config["action_difference_signature_fields"]
    ]
    rows = []
    for seed in sorted(episodes):
        episode = episodes[seed]
        close = episode["viewpoint_close_high"]
        right = episode["viewpoint_right"]
        close_observation = {field: close[field] for field in fields}
        right_observation = {field: right[field] for field in fields}
        close_latent = {
            "perception_state": close["perception_state_posthoc"],
            "world_membership": close[
                "world_membership_state_posthoc"
            ],
        }
        right_latent = {
            "perception_state": right["perception_state_posthoc"],
            "world_membership": right[
                "world_membership_state_posthoc"
            ],
        }
        close_correct = bool(
            close["center_selected_track_available"]
            and close["center_selected_track_correct_posthoc"]
        )
        right_correct = bool(
            right["center_selected_track_available"]
            and right["center_selected_track_correct_posthoc"]
        )
        if close_correct and not right_correct:
            target_preference = "viewpoint_close_high"
        elif right_correct and not close_correct:
            target_preference = "viewpoint_right"
        else:
            target_preference = "tie"
        rows.append(
            {
                "seed": seed,
                "variant": close["variant"],
                "learned_observation_signature_differs": (
                    close_observation != right_observation
                ),
                "posthoc_latent_state_differs": close_latent != right_latent,
                "tracked_target_correctness_differs": (
                    close_correct != right_correct
                ),
                "tracked_target_preference": target_preference,
                "close_high": {
                    "observation": close_observation,
                    "posthoc_latent": close_latent,
                    "tracked_target_correct_posthoc": close_correct,
                },
                "right": {
                    "observation": right_observation,
                    "posthoc_latent": right_latent,
                    "tracked_target_correct_posthoc": right_correct,
                },
            }
        )
    count = len(rows)
    learned_difference = sum(
        row["learned_observation_signature_differs"] for row in rows
    )
    latent_difference = sum(
        row["posthoc_latent_state_differs"] for row in rows
    )
    target_difference = sum(
        row["tracked_target_correctness_differs"] for row in rows
    )
    preference_counts = Counter(
        row["tracked_target_preference"] for row in rows
    )
    return {
        "episode_count": count,
        "signature_fields": fields,
        "learned_observation_signature_difference_count": (
            learned_difference
        ),
        "learned_observation_signature_difference_rate": (
            learned_difference / count
        ),
        "posthoc_latent_state_difference_count": latent_difference,
        "posthoc_latent_state_difference_rate": latent_difference / count,
        "tracked_target_correctness_difference_count": target_difference,
        "tracked_target_correctness_difference_rate": target_difference / count,
        "tracked_target_preference_counts": dict(
            sorted(preference_counts.items())
        ),
        "interpretation": (
            "A learned-output difference without a posthoc latent-state "
            "difference may reflect perception variability rather than a "
            "physically action-differentiating scene."
        ),
        "rows": rows,
    }


def run_experiment(config_path: Path) -> dict[str, Any]:
    """Run nested calibration without opening reserved test episodes."""
    nested_config = load_json(config_path)
    if nested_config.get("training_performed") is not False:
        raise ValueError("Model-weight training is forbidden")
    if nested_config.get("testing_performed") is not False:
        raise ValueError("This is calibration cross-validation, not testing")
    if nested_config.get("reserved_test_seeds_used") is not False:
        raise ValueError("Reserved test seeds must remain unused")
    base_config = load_json(
        resolve_path(nested_config["base_replay_config"])
    )
    episodes = build_episode_rows(base_config)
    folds = nested_config["outer_folds"]
    validate_outer_folds(episodes, folds)
    target_temperatures = load_fold_target_temperatures(nested_config)
    started = time.perf_counter()
    method_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fold_results = []
    held_out_audit_episodes: dict[int, dict[str, dict]] = {}
    for fold in folds:
        fold_id = int(fold["fold_id"])
        fold_temperature = target_temperatures.get(
            fold_id, float(base_config["target_temperature"])
        )
        fold_episodes = build_episode_rows(
            base_config,
            target_temperature=fold_temperature,
        )
        held_out_seeds = {
            int(seed) for seed in fold["held_out_seeds"]
        }
        outer_training = {
            seed: episode
            for seed, episode in fold_episodes.items()
            if seed not in held_out_seeds
        }
        held_out = {
            seed: episode
            for seed, episode in fold_episodes.items()
            if seed in held_out_seeds
        }
        held_out_audit_episodes.update(held_out)
        method_results = {}
        for action_agnostic in (True, False):
            method = (
                "action_agnostic_belief_mpc"
                if action_agnostic
                else "action_conditioned_belief_mpc"
            )
            tuning = tune_noncompletion_cost(
                outer_training,
                base_config,
                nested_config,
                action_agnostic=action_agnostic,
            )
            selected_config = with_noncompletion_cost(
                base_config,
                tuning["selected_task_noncompletion_cost"],
            )
            model = fit_action_model(
                outer_training,
                selected_config,
                action_agnostic=action_agnostic,
            )
            outer_rows = []
            for seed in sorted(held_out):
                replay = replay_mpc(
                    method,
                    seed,
                    held_out[seed],
                    model,
                    selected_config,
                )
                replay["outer_fold_id"] = fold_id
                replay["outer_fit_seeds"] = sorted(outer_training)
                replay["outer_held_out_seeds"] = sorted(held_out)
                replay["outer_held_out_seed_used_for_fit"] = (
                    seed in outer_training
                )
                replay["selected_task_noncompletion_cost"] = tuning[
                    "selected_task_noncompletion_cost"
                ]
                replay["evaluation_loss"] = replay_loss(
                    replay,
                    nested_config["inner_selection_loss"],
                )
                outer_rows.append(replay)
                method_rows[method].append(replay)
            method_results[method] = {
                "tuning": tuning,
                "outer_summary": summarize_method(outer_rows),
                "outer_rows": outer_rows,
            }
        fold_results.append(
            {
                "fold_id": fold_id,
                "target_temperature": fold_temperature,
                "target_temperature_fit_excludes_outer_held_out": (
                    fold_id in target_temperatures
                ),
                "outer_training_seeds": sorted(outer_training),
                "outer_held_out_seeds": sorted(held_out),
                "methods": method_results,
            }
        )
    action_audit = action_difference_audit(
        held_out_audit_episodes, nested_config
    )
    method_summaries = {
        method: summarize_method(rows)
        for method, rows in method_rows.items()
    }
    result = {
        "schema_version": "nested-action-conditioned-mpc-calibration-v2",
        "experiment_id": nested_config["experiment_id"],
        "status": "completed",
        "purpose": (
            "cpu_only_nested_task_cost_calibration_and_view_action_audit"
        ),
        "protocol": nested_config["protocol"],
        "episode_count": len(episodes),
        "seeds": sorted(episodes),
        "outer_fold_count": len(folds),
        "perception_calibration": {
            "target_temperature_source": (
                str(resolve_path(nested_config["target_calibration_result"]))
                if target_temperatures
                else "base_replay_config_single_temperature"
            ),
            "target_temperature_by_outer_fold": {
                str(fold_id): temperature
                for fold_id, temperature in sorted(
                    target_temperatures.items()
                )
            },
            "outer_held_out_target_labels_used_for_temperature_fit": False,
            "relation_evidence": (
                "hybrid_rgbd_membership_and_objective_reference_occlusion"
            ),
            "relation_likelihood_refit_on_each_outer_training_fold": True,
            "outer_held_out_relation_labels_used_for_likelihood_fit": False,
        },
        "inner_selection_loss": nested_config["inner_selection_loss"],
        "fold_results": fold_results,
        "method_summaries": method_summaries,
        "selected_cost_counts": {
            method: dict(
                sorted(
                    Counter(
                        fold["methods"][method]["tuning"][
                            "selected_task_noncompletion_cost"
                        ]
                        for fold in fold_results
                    ).items()
                )
            )
            for method in nested_config["methods"]
        },
        "action_difference_audit": action_audit,
        "runtime_seconds": time.perf_counter() - started,
        "gpu_used": False,
        "gpu_memory_gib": 0.0,
        "isaac_sim_launched": False,
        "vlm_inference_performed": False,
        "cached_vlm_outputs_reused": True,
        "robot_motion_mpc_executed": False,
        "discrete_belief_action_planner_evaluated": True,
        "training_performed": False,
        "calibration_performed": True,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
        "limitations": [
            "This is nested cross-validation over an already used calibration collection.",
            "It is not a frozen unbiased test or final-paper evidence.",
            "The task-risk loss and candidate cost grid are development choices.",
            "The cached scene set has sparse physical differences between close-high and right actions.",
            "Fold-specific Qwen temperatures and hybrid relation likelihoods are calibration candidates, not frozen final parameters.",
            "No continuous UR10e trajectory MPC, Isaac Sim, VLM inference, or grasp was run.",
        ],
    }
    output_root = resolve_path(nested_config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"fold_results", "action_difference_audit"}
    }
    summary["action_difference_summary"] = {
        key: value
        for key, value in action_audit.items()
        if key != "rows"
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"NESTED_MPC_RESULT={result_path}")
    print(f"NESTED_MPC_SUMMARY={summary_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run_experiment(args.config.resolve())


if __name__ == "__main__":
    main()
