#!/usr/bin/env python3
"""Cross-validate cached Qwen target and relation temperature scaling.

This consumes only saved calibration records. It does not launch a simulator,
perception model, GPU process, training job, or final test evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from calibrated_belief import fit_temperature_grid, softmax_temperature


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "cached_perception_calibration_cv_seed165_184.json"
)
RELATION_LABELS = {
    "membership": ("inside", "outside", "unknown"),
    "behind": ("yes", "no", "unknown"),
    "occluded_by": ("yes", "no"),
}


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


def build_examples(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    examples: dict[str, list[dict[str, Any]]] = {
        "target_identity": [],
        **{factor: [] for factor in RELATION_LABELS},
    }
    for record in records:
        seed = int(record["seed"])
        sample_id = str(record["sample_id"])
        for candidate in record["candidates"]:
            examples["target_identity"].append(
                {
                    "seed": seed,
                    "sample_id": sample_id,
                    "logits": [float(candidate["raw_match_logit"]), 0.0],
                    "label": 0 if candidate["target_label"] else 1,
                }
            )
            if candidate["matched_simulator_entity"] is None:
                continue
            for factor, label_names in RELATION_LABELS.items():
                scores = candidate["factorized_relation_scores"].get(factor)
                ground_truth = candidate["relation_ground_truth"].get(factor)
                if scores is None or ground_truth not in label_names:
                    continue
                if factor == "occluded_by":
                    source = candidate["relation_ground_truth_sources"].get(
                        factor
                    )
                    if source != "rendered_reference_removed_amodal_fraction":
                        continue
                if tuple(scores["labels"]) != label_names:
                    raise ValueError(
                        f"{factor} label order differs in {sample_id}"
                    )
                examples[factor].append(
                    {
                        "seed": seed,
                        "sample_id": sample_id,
                        "logits": [float(value) for value in scores["raw_logits"]],
                        "label": label_names.index(ground_truth),
                    }
                )
    return examples


def expected_calibration_error(
    confidences: list[float],
    correctness: list[bool],
    bins: int,
) -> float:
    if not confidences or bins <= 0:
        raise ValueError("ECE requires examples and a positive bin count")
    total = len(confidences)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            item
            for item, confidence in enumerate(confidences)
            if lower <= confidence < upper
            or (index == bins - 1 and confidence == 1.0)
        ]
        if not members:
            continue
        accuracy = sum(correctness[item] for item in members) / len(members)
        confidence = sum(confidences[item] for item in members) / len(members)
        error += len(members) / total * abs(accuracy - confidence)
    return error


def evaluate(
    examples: list[dict[str, Any]],
    temperatures_by_seed: dict[int, float],
    bins: int,
) -> dict[str, Any]:
    losses: list[float] = []
    brier_terms: list[float] = []
    confidences: list[float] = []
    correctness: list[bool] = []
    labels: list[int] = []
    predictions: list[int] = []
    for example in examples:
        temperature = temperatures_by_seed[int(example["seed"])]
        probabilities = softmax_temperature(example["logits"], temperature)
        label = int(example["label"])
        prediction = max(range(len(probabilities)), key=probabilities.__getitem__)
        losses.append(-math.log(max(1e-12, probabilities[label])))
        brier_terms.append(
            sum(
                (
                    probability
                    - (1.0 if class_index == label else 0.0)
                )
                ** 2
                for class_index, probability in enumerate(probabilities)
            )
        )
        labels.append(label)
        predictions.append(prediction)
        confidences.append(probabilities[prediction])
        correctness.append(prediction == label)
    per_class: dict[str, dict[str, float | int | None]] = {}
    for class_index in sorted(set(labels)):
        support = sum(label == class_index for label in labels)
        correct = sum(
            label == class_index and prediction == class_index
            for label, prediction in zip(labels, predictions)
        )
        per_class[str(class_index)] = {
            "support": support,
            "correct": correct,
            "recall": correct / support if support else None,
        }
    return {
        "record_count": len(examples),
        "negative_log_likelihood": sum(losses) / len(losses),
        "brier_score": sum(brier_terms) / len(brier_terms),
        "accuracy": sum(correctness) / len(correctness),
        "expected_calibration_error": expected_calibration_error(
            confidences, correctness, bins
        ),
        "per_class": per_class,
    }


def validate_folds(
    folds: list[dict[str, Any]], available_seeds: set[int]
) -> dict[int, int]:
    fold_by_seed: dict[int, int] = {}
    for fold in folds:
        fold_id = int(fold["fold_id"])
        for raw_seed in fold["held_out_seeds"]:
            seed = int(raw_seed)
            if seed in fold_by_seed:
                raise ValueError(f"Seed {seed} occurs in multiple folds")
            fold_by_seed[seed] = fold_id
    if set(fold_by_seed) != available_seeds:
        missing = sorted(available_seeds - set(fold_by_seed))
        extra = sorted(set(fold_by_seed) - available_seeds)
        raise ValueError(f"Fold seed mismatch: missing={missing}, extra={extra}")
    return fold_by_seed


def run(config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_json(config_path)
    records_path = resolve_path(config["records_path"])
    records = load_json(records_path)["records"]
    examples_by_component = build_examples(records)
    available_seeds = {int(record["seed"]) for record in records}
    fold_by_seed = validate_folds(config["outer_folds"], available_seeds)
    grid = config["temperature_grid"]
    bins = int(config["ece_bins"])

    component_results: dict[str, Any] = {}
    for component, examples in examples_by_component.items():
        by_fold: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for example in examples:
            by_fold[fold_by_seed[int(example["seed"])]].append(example)
        temperatures_by_seed: dict[int, float] = {
            seed: 1.0 for seed in available_seeds
        }
        fold_results = []
        for fold in config["outer_folds"]:
            fold_id = int(fold["fold_id"])
            train_examples = [
                example
                for candidate_fold, candidate_examples in by_fold.items()
                if candidate_fold != fold_id
                for example in candidate_examples
            ]
            held_out_examples = list(by_fold[fold_id])
            fit = fit_temperature_grid(
                [item["logits"] for item in train_examples],
                [int(item["label"]) for item in train_examples],
                minimum=float(grid["minimum"]),
                maximum=float(grid["maximum"]),
                steps=int(grid["steps"]),
            )
            for seed in fold["held_out_seeds"]:
                temperatures_by_seed[int(seed)] = float(fit["temperature"])
            fold_results.append(
                {
                    "fold_id": fold_id,
                    "train_seed_count": len(
                        available_seeds
                        - {int(seed) for seed in fold["held_out_seeds"]}
                    ),
                    "held_out_seeds": [
                        int(seed) for seed in fold["held_out_seeds"]
                    ],
                    "train_record_count": len(train_examples),
                    "held_out_record_count": len(held_out_examples),
                    "fitted_temperature": fit["temperature"],
                    "held_out_metrics": evaluate(
                        held_out_examples,
                        {
                            int(item["seed"]): float(fit["temperature"])
                            for item in held_out_examples
                        },
                        bins,
                    ),
                }
            )
        raw_temperatures = {seed: 1.0 for seed in available_seeds}
        raw_metrics = evaluate(examples, raw_temperatures, bins)
        cross_validated_metrics = evaluate(
            examples, temperatures_by_seed, bins
        )
        component_results[component] = {
            "label_order": (
                ["target", "not_target"]
                if component == "target_identity"
                else list(RELATION_LABELS[component])
            ),
            "raw_metrics": raw_metrics,
            "cross_validated_temperature_metrics": cross_validated_metrics,
            "delta": {
                "negative_log_likelihood": (
                    cross_validated_metrics["negative_log_likelihood"]
                    - raw_metrics["negative_log_likelihood"]
                ),
                "brier_score": (
                    cross_validated_metrics["brier_score"]
                    - raw_metrics["brier_score"]
                ),
                "expected_calibration_error": (
                    cross_validated_metrics["expected_calibration_error"]
                    - raw_metrics["expected_calibration_error"]
                ),
            },
            "folds": fold_results,
        }

    blockers = []
    for component, result in component_results.items():
        label_order = result["label_order"]
        per_class = result["cross_validated_temperature_metrics"]["per_class"]
        zero_recall = [
            label_order[int(index)]
            for index, diagnostics in per_class.items()
            if diagnostics["support"] and diagnostics["correct"] == 0
        ]
        if zero_recall:
            blockers.append(
                f"{component}:zero_recall_labels:{','.join(zero_recall)}"
            )
        if result["delta"]["negative_log_likelihood"] >= 0:
            blockers.append(f"{component}:held_out_nll_not_improved")

    result = {
        "schema_version": "cached-perception-calibration-cv-result-v1",
        "experiment_id": config["experiment_id"],
        "protocol": "four_fold_episode_disjoint_temperature_cross_validation",
        "source_records": str(records_path),
        "seed_count": len(available_seeds),
        "observation_count": len(records),
        "components": component_results,
        "deployment_decision": {
            "adopt_as_final_calibration": False,
            "blocking_reasons": blockers
            + [
                "calibration_collection_only",
                "task_risk_gate_not_calibrated",
                "action_conditioned_observation_model_not_validated",
                "reserved_test_seeds_not_used",
            ],
        },
        "training_performed": False,
        "calibration_performed": True,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
        "gpu_used": False,
        "gpu_memory_gib": 0.0,
        "runtime_seconds": time.perf_counter() - started,
    }
    output_root = resolve_path(config["output_root"])
    write_json_atomic(output_root / "result.json", result)
    summary = {
        "experiment_id": result["experiment_id"],
        "protocol": result["protocol"],
        "seed_count": result["seed_count"],
        "observation_count": result["observation_count"],
        "component_metrics": {
            component: {
                "raw": value["raw_metrics"],
                "cross_validated": value[
                    "cross_validated_temperature_metrics"
                ],
                "delta": value["delta"],
                "fold_temperatures": [
                    fold["fitted_temperature"] for fold in value["folds"]
                ],
            }
            for component, value in component_results.items()
        },
        "deployment_decision": result["deployment_decision"],
        "training_performed": False,
        "testing_performed": False,
        "valid_for_final_evaluation": False,
        "gpu_used": False,
        "runtime_seconds": result["runtime_seconds"],
    }
    write_json_atomic(output_root / "summary.json", summary)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = run(args.config)
    print(
        json.dumps(
            {
                "experiment_id": result["experiment_id"],
                "runtime_seconds": result["runtime_seconds"],
                "blocking_reasons": result["deployment_decision"][
                    "blocking_reasons"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
