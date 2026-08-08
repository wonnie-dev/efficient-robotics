#!/usr/bin/env python3
"""Leak-controlled CPU baseline and ablation replay on cached episodes."""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from run_nested_action_conditioned_mpc_calibration import (
    load_fold_target_temperatures,
    replay_loss,
    validate_outer_folds,
    with_noncompletion_cost,
)
from run_offline_action_conditioned_mpc_replay import (
    MEMBERSHIP_OBSERVATIONS,
    OCCLUSION_OBSERVATIONS,
    build_episode_rows,
    fit_action_model,
    fixed_policy,
    replay_confidence_baseline,
    replay_mpc,
    resolve_path,
    summarize_method,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "offline_full_baseline_ablation_seed165_184.json"
)


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


def configure_relation_observations(
    episodes: dict[int, dict[str, dict]],
    *,
    membership_enabled: bool,
    occlusion_enabled: bool,
) -> dict[int, dict[str, dict]]:
    configured = copy.deepcopy(episodes)
    for episode in configured.values():
        for row in episode.values():
            if not membership_enabled:
                row["membership_observation"] = "missing"
            if not occlusion_enabled:
                row["reference_occlusion_observation"] = "missing"
            row["perception_observation"] = (
                f"{row['identity_bin']}|"
                f"{row['reference_occlusion_observation']}"
            )
    return configured


def remove_hybrid_relation_observations(
    episodes: dict[int, dict[str, dict]],
) -> dict[int, dict[str, dict]]:
    return configure_relation_observations(
        episodes,
        membership_enabled=False,
        occlusion_enabled=False,
    )


def configure_relation_likelihood(
    model: dict[str, Any],
    *,
    membership_enabled: bool,
    occlusion_enabled: bool,
) -> dict[str, Any]:
    """Keep selected calibrated relation components and neutralize the rest."""
    configured = copy.deepcopy(model)
    if not membership_enabled:
        for action, state_tables in configured[
            "membership_observation"
        ].items():
            for state in state_tables:
                state_tables[state] = {
                    outcome: 1.0 if outcome == "missing" else 0.0
                    for outcome in MEMBERSHIP_OBSERVATIONS
                }
    identity_bins = tuple(configured["identity_bins"])
    if not occlusion_enabled:
        for action, state_tables in configured[
            "perception_observation"
        ].items():
            for state, distribution in state_tables.items():
                identity_mass = {
                    identity_bin: sum(
                        distribution[
                            f"{identity_bin}|{occlusion_observation}"
                        ]
                        for occlusion_observation in OCCLUSION_OBSERVATIONS
                    )
                    for identity_bin in identity_bins
                }
                state_tables[state] = {
                    f"{identity_bin}|{occlusion_observation}": (
                        identity_mass[identity_bin]
                        if occlusion_observation == "missing"
                        else 0.0
                    )
                    for identity_bin in identity_bins
                    for occlusion_observation in OCCLUSION_OBSERVATIONS
                }
    configured["hybrid_relation_evidence"] = {
        "membership_enabled": membership_enabled,
        "occlusion_enabled": occlusion_enabled,
    }
    return configured


def make_relation_likelihood_uninformative(
    model: dict[str, Any],
) -> dict[str, Any]:
    return configure_relation_likelihood(
        model,
        membership_enabled=False,
        occlusion_enabled=False,
    )


def risk_neutral_config(base_config: dict[str, Any]) -> dict[str, Any]:
    selected = copy.deepcopy(base_config)
    for key in (
        "wrong_target_weight",
        "hidden_target_weight",
        "reference_occlusion_weight",
        "membership_entropy_weight",
    ):
        selected["task_cost"][key] = 0.0
    selected["commitment_gate"].update(
        {
            "minimum_selected_target_correct_probability": 0.0,
            "maximum_fully_hidden_probability": 1.0,
            "maximum_reference_occlusion_probability": 1.0,
            "status": "disabled_for_no_task_risk_ablation",
        }
    )
    return selected


def model_for_method(
    method: str,
    training: dict[int, dict[str, dict]],
    config: dict[str, Any],
) -> dict[str, Any]:
    action_agnostic = method == "action_agnostic_belief_mpc"
    model = fit_action_model(
        training,
        config,
        action_agnostic=action_agnostic,
    )
    if method == "ablation_no_hybrid_relation_evidence":
        model = make_relation_likelihood_uninformative(model)
    return model


def replay_model_method(
    method: str,
    seed: int,
    episode: dict[str, dict],
    model: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if method == "confidence_only_fixed_reobservation":
        row = replay_confidence_baseline(seed, episode, model, config)
        row["method"] = method
        return row
    return replay_mpc(method, seed, episode, model, config)


def tune_cost(
    method: str,
    outer_training: dict[int, dict[str, dict]],
    base_config: dict[str, Any],
    nested_config: dict[str, Any],
) -> dict[str, Any]:
    candidates = []
    for value in nested_config["candidate_task_noncompletion_costs"]:
        config = with_noncompletion_cost(base_config, float(value))
        rows = []
        for validation_seed in sorted(outer_training):
            inner_training = {
                seed: episode
                for seed, episode in outer_training.items()
                if seed != validation_seed
            }
            model = model_for_method(method, inner_training, config)
            row = replay_model_method(
                method,
                validation_seed,
                outer_training[validation_seed],
                model,
                config,
            )
            row["selection_loss"] = replay_loss(
                row, nested_config["inner_selection_loss"]
            )
            row["inner_validation_seed_used_for_fit"] = (
                validation_seed in model["training_seeds"]
            )
            rows.append(row)
        total = sum(float(row["selection_loss"]) for row in rows)
        candidates.append(
            {
                "task_noncompletion_cost": float(value),
                "mean_selection_loss": total / len(rows),
                "summary": summarize_method(rows),
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
        "selected_task_noncompletion_cost": selected[
            "task_noncompletion_cost"
        ],
        "candidate_results": candidates,
        "outer_held_out_data_used_for_selection": False,
    }


def variant_failure_summary(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["variant"])].append(row)
    return {
        variant: {
            "episode_count": len(items),
            "correct_grasp_count": sum(
                item["correct_grasp"] for item in items
            ),
            "wrong_grasp_count": sum(
                item["wrong_grasp"] for item in items
            ),
            "safe_defer_count": sum(
                item["safe_defer_fully_hidden"] for item in items
            ),
            "missed_visible_target_count": sum(
                item["missed_visible_target"] for item in items
            ),
            "failed_seeds": [
                item["seed"]
                for item in items
                if item["wrong_grasp"]
                or item["missed_visible_target"]
            ],
        }
        for variant, items in sorted(grouped.items())
    }


def paired_loss_counts(
    proposed: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    loss_config: dict[str, float],
) -> dict[str, int]:
    proposed_by_seed = {int(row["seed"]): row for row in proposed}
    comparison_by_seed = {int(row["seed"]): row for row in comparison}
    counts = Counter()
    for seed in sorted(proposed_by_seed):
        proposed_loss = replay_loss(
            proposed_by_seed[seed], loss_config
        )
        comparison_loss = replay_loss(
            comparison_by_seed[seed], loss_config
        )
        if proposed_loss < comparison_loss:
            counts["proposed_better"] += 1
        elif proposed_loss > comparison_loss:
            counts["proposed_worse"] += 1
        else:
            counts["tie"] += 1
    return {
        key: counts[key]
        for key in ("proposed_better", "proposed_worse", "tie")
    }


def run(config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    experiment_config = load_json(config_path)
    if experiment_config["training_performed"] is not False:
        raise ValueError("Model-weight training is forbidden")
    if experiment_config["testing_performed"] is not False:
        raise ValueError("Reserved testing is forbidden")
    nested_config = load_json(
        resolve_path(experiment_config["nested_calibration_config"])
    )
    base_config = load_json(
        resolve_path(experiment_config["base_replay_config"])
    )
    raw_episodes = build_episode_rows(base_config, target_temperature=1.0)
    validate_outer_folds(raw_episodes, nested_config["outer_folds"])
    if set(raw_episodes) & set(range(200, 210)):
        raise ValueError("Reserved test seeds entered calibration replay")
    temperatures = load_fold_target_temperatures(nested_config)
    method_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fold_results = []
    tuned_methods = {
        "confidence_only_fixed_reobservation",
        "action_agnostic_belief_mpc",
        "action_conditioned_belief_mpc",
        "ablation_no_target_temperature",
        "ablation_no_hybrid_relation_evidence",
    }
    fixed_methods = {
        "immediate_grasp": "grasp",
        "fixed_close_high_then_grasp": "viewpoint_close_high",
        "fixed_right_then_grasp": "viewpoint_right",
    }

    for fold in nested_config["outer_folds"]:
        fold_id = int(fold["fold_id"])
        held_out_seeds = {int(seed) for seed in fold["held_out_seeds"]}
        calibrated = build_episode_rows(
            base_config,
            target_temperature=temperatures[fold_id],
        )
        method_episode_sets = {
            method: calibrated for method in experiment_config["methods"]
        }
        method_episode_sets["ablation_no_target_temperature"] = (
            raw_episodes
        )
        method_episode_sets[
            "ablation_no_hybrid_relation_evidence"
        ] = remove_hybrid_relation_observations(calibrated)
        fold_methods = {}
        for method in experiment_config["methods"]:
            episodes = method_episode_sets[method]
            outer_training = {
                seed: episode
                for seed, episode in episodes.items()
                if seed not in held_out_seeds
            }
            held_out = {
                seed: episode
                for seed, episode in episodes.items()
                if seed in held_out_seeds
            }
            if method in fixed_methods:
                tuning = None
                rows = [
                    fixed_policy(
                        method,
                        seed,
                        held_out[seed],
                        fixed_methods[method],
                    )
                    for seed in sorted(held_out)
                ]
            elif method == "ablation_no_task_risk":
                tuning = None
                selected_config = risk_neutral_config(base_config)
                model = model_for_method(
                    method, outer_training, selected_config
                )
                rows = [
                    replay_model_method(
                        method,
                        seed,
                        held_out[seed],
                        model,
                        selected_config,
                    )
                    for seed in sorted(held_out)
                ]
            elif method in tuned_methods:
                tuning = tune_cost(
                    method,
                    outer_training,
                    base_config,
                    nested_config,
                )
                selected_config = with_noncompletion_cost(
                    base_config,
                    tuning["selected_task_noncompletion_cost"],
                )
                model = model_for_method(
                    method, outer_training, selected_config
                )
                rows = [
                    replay_model_method(
                        method,
                        seed,
                        held_out[seed],
                        model,
                        selected_config,
                    )
                    for seed in sorted(held_out)
                ]
            else:
                raise ValueError(f"Unsupported method {method}")
            for row in rows:
                row["outer_fold_id"] = fold_id
                row["outer_held_out_seed_used_for_fit"] = False
                row["target_temperature"] = (
                    1.0
                    if method == "ablation_no_target_temperature"
                    else temperatures[fold_id]
                )
            method_rows[method].extend(rows)
            fold_methods[method] = {
                "summary": summarize_method(rows),
                "tuning": tuning,
                "rows": rows,
            }
        fold_results.append(
            {
                "fold_id": fold_id,
                "held_out_seeds": sorted(held_out_seeds),
                "target_temperature": temperatures[fold_id],
                "methods": fold_methods,
            }
        )

    if set(method_rows) != set(experiment_config["methods"]):
        raise AssertionError("Not every declared method was evaluated")
    summaries = {
        method: summarize_method(rows)
        for method, rows in method_rows.items()
    }
    proposed = method_rows["action_conditioned_belief_mpc"]
    paired = {
        method: paired_loss_counts(
            proposed,
            rows,
            nested_config["inner_selection_loss"],
        )
        for method, rows in method_rows.items()
        if method != "action_conditioned_belief_mpc"
    }
    result = {
        "schema_version": "offline-full-baseline-ablation-result-v1",
        "experiment_id": experiment_config["experiment_id"],
        "status": "completed",
        "protocol": experiment_config["protocol"],
        "episode_count": len(raw_episodes),
        "seeds": sorted(raw_episodes),
        "outer_fold_count": len(nested_config["outer_folds"]),
        "methods": {
            method: {
                "summary": summaries[method],
                "variant_failure_summary": variant_failure_summary(rows),
                "episodes": rows,
            }
            for method, rows in method_rows.items()
        },
        "fold_results": fold_results,
        "paired_loss_vs_proposed": paired,
        "negative_evidence_ablation": experiment_config[
            "negative_evidence_ablation"
        ],
        "claim_gate": {
            "proposed_superiority_supported": False,
            "reasons": [
                "calibration_collection_not_reserved_test",
                "only_one_of_twenty_episodes_has_action_dependent_posthoc_latent_state",
                "proposed_does_not_outperform_all_baselines",
                "no_live_robot_or_physics_execution",
            ],
        },
        "runtime_seconds": time.perf_counter() - started,
        "gpu_used": False,
        "gpu_memory_gib": 0.0,
        "isaac_sim_launched": False,
        "vlm_inference_performed": False,
        "cached_vlm_outputs_reused": True,
        "training_performed": False,
        "calibration_performed": True,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
    }
    output_root = resolve_path(experiment_config["output_root"])
    write_json_atomic(output_root / "result.json", result)
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"methods", "fold_results"}
    }
    summary["method_summaries"] = summaries
    summary["variant_failure_summaries"] = {
        method: variant_failure_summary(rows)
        for method, rows in method_rows.items()
    }
    write_json_atomic(output_root / "summary.json", summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = run(args.config.resolve())
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "runtime_seconds": result["runtime_seconds"],
                "method_summaries": {
                    method: payload["summary"]
                    for method, payload in result["methods"].items()
                },
                "claim_gate": result["claim_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
