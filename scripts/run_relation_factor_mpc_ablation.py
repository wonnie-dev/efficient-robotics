#!/usr/bin/env python3
"""CPU-only downstream ablation of calibrated relation evidence factors."""

from __future__ import annotations

import argparse
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
    build_episode_rows,
    fit_action_model,
    replay_mpc,
    resolve_path,
    summarize_method,
)
from run_offline_full_baseline_ablation import (
    configure_relation_likelihood,
    configure_relation_observations,
    variant_failure_summary,
    write_json_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "relation_factor_mpc_ablation_seed165_184.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fitted_model(
    training: dict[int, dict[str, dict]],
    config: dict[str, Any],
    condition: dict[str, bool],
) -> dict[str, Any]:
    model = fit_action_model(
        training, config, action_agnostic=False
    )
    return configure_relation_likelihood(
        model,
        membership_enabled=bool(condition["membership_enabled"]),
        occlusion_enabled=bool(condition["occlusion_enabled"]),
    )


def tune_cost(
    episodes: dict[int, dict[str, dict]],
    base_config: dict[str, Any],
    nested_config: dict[str, Any],
    condition: dict[str, bool],
) -> dict[str, Any]:
    candidates = []
    for value in nested_config["candidate_task_noncompletion_costs"]:
        config = with_noncompletion_cost(base_config, float(value))
        rows = []
        for validation_seed in sorted(episodes):
            training = {
                seed: episode
                for seed, episode in episodes.items()
                if seed != validation_seed
            }
            model = fitted_model(training, config, condition)
            row = replay_mpc(
                "relation_factor_ablation",
                validation_seed,
                episodes[validation_seed],
                model,
                config,
            )
            row["selection_loss"] = replay_loss(
                row, nested_config["inner_selection_loss"]
            )
            if validation_seed in model["training_seeds"]:
                raise AssertionError("Inner validation seed leaked into fit")
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


def paired_counts(
    full_rows: list[dict[str, Any]],
    ablated_rows: list[dict[str, Any]],
    loss_config: dict[str, float],
) -> dict[str, int]:
    full = {int(row["seed"]): row for row in full_rows}
    ablated = {int(row["seed"]): row for row in ablated_rows}
    counts = Counter()
    for seed in sorted(full):
        full_loss = replay_loss(full[seed], loss_config)
        ablated_loss = replay_loss(ablated[seed], loss_config)
        if full_loss < ablated_loss:
            counts["full_better"] += 1
        elif full_loss > ablated_loss:
            counts["full_worse"] += 1
        else:
            counts["tie"] += 1
    return {
        key: counts[key] for key in ("full_better", "full_worse", "tie")
    }


def run(config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    experiment = load_json(config_path)
    if experiment["training_performed"] is not False:
        raise ValueError("Model-weight training is forbidden")
    if experiment["testing_performed"] is not False:
        raise ValueError("This is calibration replay, not testing")
    nested = load_json(
        resolve_path(experiment["nested_calibration_config"])
    )
    base = load_json(resolve_path(experiment["base_replay_config"]))
    initial = build_episode_rows(base)
    validate_outer_folds(initial, nested["outer_folds"])
    if set(initial) & set(range(200, 210)):
        raise ValueError("Reserved test seeds entered relation ablation")
    temperatures = load_fold_target_temperatures(nested)
    condition_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fold_results = []

    for fold in nested["outer_folds"]:
        fold_id = int(fold["fold_id"])
        held_out_seeds = {int(seed) for seed in fold["held_out_seeds"]}
        calibrated = build_episode_rows(
            base, target_temperature=temperatures[fold_id]
        )
        fold_conditions = {}
        for name, condition in experiment["conditions"].items():
            episodes = configure_relation_observations(
                calibrated,
                membership_enabled=bool(
                    condition["membership_enabled"]
                ),
                occlusion_enabled=bool(
                    condition["occlusion_enabled"]
                ),
            )
            training = {
                seed: episode
                for seed, episode in episodes.items()
                if seed not in held_out_seeds
            }
            held_out = {
                seed: episode
                for seed, episode in episodes.items()
                if seed in held_out_seeds
            }
            tuning = tune_cost(training, base, nested, condition)
            selected_config = with_noncompletion_cost(
                base, tuning["selected_task_noncompletion_cost"]
            )
            model = fitted_model(
                training, selected_config, condition
            )
            rows = []
            for seed in sorted(held_out):
                row = replay_mpc(
                    name,
                    seed,
                    held_out[seed],
                    model,
                    selected_config,
                )
                row["outer_fold_id"] = fold_id
                row["outer_held_out_seed_used_for_fit"] = False
                rows.append(row)
                condition_rows[name].append(row)
            fold_conditions[name] = {
                "condition": condition,
                "tuning": tuning,
                "summary": summarize_method(rows),
                "rows": rows,
            }
        fold_results.append(
            {
                "fold_id": fold_id,
                "held_out_seeds": sorted(held_out_seeds),
                "target_temperature": temperatures[fold_id],
                "conditions": fold_conditions,
            }
        )

    full_name = "target_plus_membership_and_occlusion"
    full_rows = condition_rows[full_name]
    summaries = {
        name: summarize_method(rows)
        for name, rows in condition_rows.items()
    }
    paired = {
        name: paired_counts(
            full_rows,
            rows,
            nested["inner_selection_loss"],
        )
        for name, rows in condition_rows.items()
        if name != full_name
    }
    policy_signatures = {
        name: [
            (
                int(row["seed"]),
                str(row["selected_first_action"]),
                str(row["terminal_action"]),
            )
            for row in rows
        ]
        for name, rows in condition_rows.items()
    }
    factor_diagnostic = {
        "membership_only_safe_outcome_gain_over_target_only": (
            summaries["target_plus_membership"]["safe_outcome_rate"]
            - summaries["target_only"]["safe_outcome_rate"]
        ),
        "occlusion_only_safe_outcome_gain_over_target_only": (
            summaries["target_plus_occlusion"]["safe_outcome_rate"]
            - summaries["target_only"]["safe_outcome_rate"]
        ),
        "full_policy_identical_to_occlusion_only": (
            policy_signatures[full_name]
            == policy_signatures["target_plus_occlusion"]
        ),
        "full_missed_visible_seeds": [
            int(row["seed"])
            for row in full_rows
            if row["missed_visible_target"]
        ],
        "development_recommendation": (
            "retain_membership_candidate_and_block_occlusion_from_"
            "downstream_mpc_until_observation_model_and_gate_are_revised"
        ),
    }
    result = {
        "schema_version": "relation-factor-mpc-ablation-result-v1",
        "experiment_id": experiment["experiment_id"],
        "status": "completed",
        "protocol": experiment["protocol"],
        "episode_count": len(initial),
        "seeds": sorted(initial),
        "conditions": {
            name: {
                "configuration": experiment["conditions"][name],
                "summary": summaries[name],
                "variant_failure_summary": variant_failure_summary(rows),
                "episodes": rows,
            }
            for name, rows in condition_rows.items()
        },
        "fold_results": fold_results,
        "paired_loss_full_vs_ablation": paired,
        "factor_diagnostic": factor_diagnostic,
        "interpretation_gate": {
            "relation_factor_benefit_supported": False,
            "reason": (
                "development_calibration_cache_requires_result_review_"
                "and_lacks_action_differentiating_coverage"
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
        "gpu_used": False,
        "gpu_memory_gib": 0.0,
        "isaac_sim_launched": False,
        "vlm_inference_performed": False,
        "training_performed": False,
        "calibration_performed": True,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
    }
    output_root = resolve_path(experiment["output_root"])
    write_json_atomic(output_root / "result.json", result)
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"conditions", "fold_results"}
    }
    summary["condition_summaries"] = summaries
    summary["variant_failure_summaries"] = {
        name: variant_failure_summary(rows)
        for name, rows in condition_rows.items()
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
                "condition_summaries": {
                    name: payload["summary"]
                    for name, payload in result["conditions"].items()
                },
                "paired_loss_full_vs_ablation": result[
                    "paired_loss_full_vs_ablation"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
