#!/usr/bin/env python3
"""Cross-validate hybrid RGB-D relation observation likelihoods.

The calibration model is a smoothed categorical likelihood P(z | y), where y
is a relation state and z is the hybrid RGB-D evidence label. The script uses
only saved calibration audit rows and never launches perception or simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "hybrid_relation_likelihood_cv_seed165_184.json"
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


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_configured_rows(
    config: dict[str, Any],
) -> tuple[list[Path], list[dict[str, str]]]:
    values = config.get("audit_rows_csvs")
    if values is None:
        values = [config["audit_rows_csv"]]
    paths = [resolve_path(value) for value in values]
    rows = [row for path in paths for row in load_rows(path)]
    return paths, rows


def validate_folds(
    folds: list[dict[str, Any]], available_seeds: set[int]
) -> dict[int, int]:
    fold_by_seed: dict[int, int] = {}
    for fold in folds:
        fold_id = int(fold["fold_id"])
        for raw_seed in fold["held_out_seeds"]:
            seed = int(raw_seed)
            if seed in fold_by_seed:
                raise ValueError(f"Seed {seed} appears in multiple folds")
            fold_by_seed[seed] = fold_id
    if set(fold_by_seed) != available_seeds:
        raise ValueError(
            "Fold seeds differ from audit seeds: "
            f"missing={sorted(available_seeds - set(fold_by_seed))}, "
            f"extra={sorted(set(fold_by_seed) - available_seeds)}"
        )
    return fold_by_seed


def relation_examples(
    rows: list[dict[str, str]], specification: dict[str, Any]
) -> list[dict[str, Any]]:
    true_labels = list(specification["true_labels"])
    observation_labels = list(specification["observation_labels"])
    examples = []
    for row in rows:
        truth = row[specification["truth_column"]]
        observation = row[specification["prediction_column"]]
        if specification.get("require_nonempty_truth") and not truth:
            continue
        if truth not in true_labels:
            raise ValueError(f"Unexpected truth label {truth!r}")
        if observation not in observation_labels:
            raise ValueError(f"Unexpected observation label {observation!r}")
        examples.append(
            {
                "seed": int(row["seed"]),
                "sample_id": row["sample_id"],
                "truth": truth,
                "observation": observation,
            }
        )
    if not examples:
        raise ValueError("Relation specification produced no examples")
    return examples


def missing_true_labels(
    examples: list[dict[str, Any]], true_labels: list[str]
) -> list[str]:
    observed = {example["truth"] for example in examples}
    return sorted(set(true_labels) - observed)


def fit_likelihood(
    examples: list[dict[str, Any]],
    true_labels: list[str],
    observation_labels: list[str],
    pseudocount: float,
) -> dict[str, dict[str, float]]:
    if pseudocount <= 0.0:
        raise ValueError("Dirichlet pseudocount must be positive")
    counts = {
        truth: {
            observation: float(pseudocount)
            for observation in observation_labels
        }
        for truth in true_labels
    }
    for example in examples:
        counts[example["truth"]][example["observation"]] += 1.0
    return {
        truth: {
            observation: value / sum(counts[truth].values())
            for observation, value in counts[truth].items()
        }
        for truth in true_labels
    }


def posterior_from_likelihood(
    observation: str,
    likelihood: dict[str, dict[str, float]],
    true_labels: list[str],
) -> dict[str, float]:
    # A uniform state prior isolates the quality of the calibrated observation
    # model. The live Scene Graph will instead provide its current prior.
    weights = {
        truth: likelihood[truth][observation] / len(true_labels)
        for truth in true_labels
    }
    total = sum(weights.values())
    return {truth: value / total for truth, value in weights.items()}


def uniform_posterior(true_labels: list[str]) -> dict[str, float]:
    return {label: 1.0 / len(true_labels) for label in true_labels}


def decision_metric_summary(
    examples: list[dict[str, Any]],
    true_labels: list[str],
    observation_labels: list[str],
) -> dict[str, Any]:
    """Summarize categorical evidence without inventing probabilities."""
    confusion = {
        truth: {observation: 0 for observation in observation_labels}
        for truth in true_labels
    }
    covered = 0
    covered_correct = 0
    correct = 0
    for example in examples:
        truth = example["truth"]
        observation = example["observation"]
        confusion[truth][observation] += 1
        if observation == truth:
            correct += 1
        if observation in true_labels:
            covered += 1
            if observation == truth:
                covered_correct += 1
    per_class = {}
    for label in true_labels:
        support = sum(example["truth"] == label for example in examples)
        label_correct = sum(
            example["truth"] == label
            and example["observation"] == label
            for example in examples
        )
        per_class[label] = {
            "support": support,
            "correct": label_correct,
            "recall": label_correct / support if support else None,
        }
    return {
        "record_count": len(examples),
        "accuracy_with_abstentions_counted_as_incorrect": (
            correct / len(examples)
        ),
        "coverage": covered / len(examples),
        "selective_accuracy": (
            covered_correct / covered if covered else None
        ),
        "confusion": confusion,
        "per_class": per_class,
        "probabilities_available": False,
    }


def expected_calibration_error(
    confidences: list[float],
    correctness: list[bool],
    bins: int,
) -> float:
    if not confidences or bins <= 0:
        raise ValueError("ECE requires examples and positive bins")
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
        mean_confidence = (
            sum(confidences[item] for item in members) / len(members)
        )
        error += (
            len(members) / total * abs(accuracy - mean_confidence)
        )
    return error


def metric_summary(
    examples: list[dict[str, Any]],
    probabilities: list[dict[str, float]],
    true_labels: list[str],
    bins: int,
) -> dict[str, Any]:
    losses = []
    brier = []
    correctness = []
    confidences = []
    predictions = []
    for example, distribution in zip(examples, probabilities):
        truth = example["truth"]
        maximum = max(distribution.values())
        maximizers = [
            label
            for label, probability in distribution.items()
            if math.isclose(probability, maximum, rel_tol=0.0, abs_tol=1e-12)
        ]
        prediction = maximizers[0] if len(maximizers) == 1 else "__tie__"
        predictions.append(prediction)
        correctness.append(prediction == truth)
        confidences.append(maximum)
        losses.append(-math.log(max(1e-12, distribution[truth])))
        brier.append(
            sum(
                (
                    distribution[label]
                    - (1.0 if label == truth else 0.0)
                )
                ** 2
                for label in true_labels
            )
        )
    per_class = {}
    for label in true_labels:
        support = sum(example["truth"] == label for example in examples)
        correct = sum(
            example["truth"] == label and prediction == label
            for example, prediction in zip(examples, predictions)
        )
        per_class[label] = {
            "support": support,
            "correct": correct,
            "recall": correct / support if support else None,
        }
    return {
        "record_count": len(examples),
        "negative_log_likelihood": sum(losses) / len(losses),
        "brier_score": sum(brier) / len(brier),
        "accuracy": sum(correctness) / len(correctness),
        "expected_calibration_error": expected_calibration_error(
            confidences, correctness, bins
        ),
        "per_class": per_class,
    }


def run(config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_json(config_path)
    rows_paths, rows = load_configured_rows(config)
    available_seeds = {int(row["seed"]) for row in rows}
    fold_by_seed = validate_folds(config["outer_folds"], available_seeds)
    pseudocount = float(config["dirichlet_pseudocount"])
    bins = int(config["ece_bins"])
    relation_results = {}
    blockers = []

    for relation, specification in config["relations"].items():
        true_labels = list(specification["true_labels"])
        observation_labels = list(specification["observation_labels"])
        examples = relation_examples(rows, specification)
        calibrated_probabilities_by_key: dict[
            tuple[str, str], dict[str, float]
        ] = {}
        folds = []
        for fold in config["outer_folds"]:
            fold_id = int(fold["fold_id"])
            train = [
                example
                for example in examples
                if fold_by_seed[example["seed"]] != fold_id
            ]
            held_out = [
                example
                for example in examples
                if fold_by_seed[example["seed"]] == fold_id
            ]
            likelihood = fit_likelihood(
                train, true_labels, observation_labels, pseudocount
            )
            held_out_probabilities = [
                posterior_from_likelihood(
                    example["observation"], likelihood, true_labels
                )
                for example in held_out
            ]
            for example, probabilities in zip(
                held_out, held_out_probabilities
            ):
                calibrated_probabilities_by_key[
                    (example["sample_id"], example["observation"])
                ] = probabilities
            folds.append(
                {
                    "fold_id": fold_id,
                    "held_out_seeds": [
                        int(seed) for seed in fold["held_out_seeds"]
                    ],
                    "train_record_count": len(train),
                    "held_out_record_count": len(held_out),
                    "observation_likelihood_p_z_given_y": likelihood,
                    "held_out_metrics": metric_summary(
                        held_out,
                        held_out_probabilities,
                        true_labels,
                        bins,
                    ),
                }
            )
        calibrated_probabilities = [
            calibrated_probabilities_by_key[
                (example["sample_id"], example["observation"])
            ]
            for example in examples
        ]
        hard_decision_metrics = decision_metric_summary(
            examples, true_labels, observation_labels
        )
        no_evidence_probabilities = [
            uniform_posterior(true_labels) for _ in examples
        ]
        no_evidence_metrics = metric_summary(
            examples, no_evidence_probabilities, true_labels, bins
        )
        calibrated_metrics = metric_summary(
            examples, calibrated_probabilities, true_labels, bins
        )
        full_data_likelihood = fit_likelihood(
            examples,
            true_labels,
            observation_labels,
            pseudocount,
        )
        zero_recall = [
            label
            for label, diagnostics in calibrated_metrics[
                "per_class"
            ].items()
            if diagnostics["support"] and diagnostics["correct"] == 0
        ]
        relation_blockers = []
        missing_labels = missing_true_labels(examples, true_labels)
        if missing_labels:
            relation_blockers.append(
                "missing_truth_labels:" + ",".join(missing_labels)
            )
        if zero_recall:
            relation_blockers.append(
                "zero_recall_labels:" + ",".join(zero_recall)
            )
        if (
            calibrated_metrics["negative_log_likelihood"]
            >= no_evidence_metrics["negative_log_likelihood"]
        ):
            relation_blockers.append(
                "held_out_nll_not_better_than_no_evidence"
            )
        if (
            relation == "behind"
            and specification.get("ground_truth_source")
            != "objective_camera_relative_simulator_geometry"
        ):
            relation_blockers.append(
                "ground_truth_is_legacy_view_intent_not_objective_geometry"
            )
        blockers.extend(
            f"{relation}:{reason}" for reason in relation_blockers
        )
        relation_results[relation] = {
            "true_labels": true_labels,
            "ground_truth_source": specification.get(
                "ground_truth_source", "unspecified"
            ),
            "observation_labels": observation_labels,
            "hard_evidence_decision_metrics": hard_decision_metrics,
            "no_evidence_uniform_baseline": no_evidence_metrics,
            "cross_validated_likelihood_metrics": calibrated_metrics,
            "full_calibration_observation_likelihood_p_z_given_y": (
                full_data_likelihood
            ),
            "delta": {
                "negative_log_likelihood_vs_no_evidence": (
                    calibrated_metrics["negative_log_likelihood"]
                    - no_evidence_metrics["negative_log_likelihood"]
                ),
                "brier_score_vs_no_evidence": (
                    calibrated_metrics["brier_score"]
                    - no_evidence_metrics["brier_score"]
                ),
            },
            "component_candidate_for_scene_graph": not relation_blockers,
            "blocking_reasons": relation_blockers,
            "folds": folds,
        }

    result = {
        "schema_version": "hybrid-relation-likelihood-cv-result-v2",
        "experiment_id": config["experiment_id"],
        "protocol": (
            "four_fold_episode_disjoint_dirichlet_categorical_"
            "observation_likelihood_calibration"
        ),
        "source_audit_rows": [str(path) for path in rows_paths],
        "seed_count": len(available_seeds),
        "relation_results": relation_results,
        "deployment_decision": {
            "apply_to_final_mpc": False,
            "component_candidates": {
                relation: value["component_candidate_for_scene_graph"]
                for relation, value in relation_results.items()
            },
            "blocking_reasons": blockers
            + [
                "calibration_collection_only",
                "action_conditioned_observation_model_not_validated",
                "task_risk_gate_not_calibrated",
                "reserved_test_seeds_not_used",
            ],
        },
        "probability_semantics": {
            "fitted_quantity": "P(hybrid_evidence_label | true_relation)",
            "evaluation_posterior_prior": "uniform",
            "live_posterior_prior": "current_scene_graph_relation_belief",
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
        "relations": {
            relation: {
                "hard_decision": value["hard_evidence_decision_metrics"],
                "no_evidence": value["no_evidence_uniform_baseline"],
                "cross_validated": value[
                    "cross_validated_likelihood_metrics"
                ],
                "delta": value["delta"],
                "component_candidate_for_scene_graph": value[
                    "component_candidate_for_scene_graph"
                ],
                "blocking_reasons": value["blocking_reasons"],
            }
            for relation, value in relation_results.items()
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
                "component_candidates": result["deployment_decision"][
                    "component_candidates"
                ],
                "blocking_reasons": result["deployment_decision"][
                    "blocking_reasons"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
