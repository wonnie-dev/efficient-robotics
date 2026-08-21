#!/usr/bin/env python3
"""Compare cached candidate selectors without using simulator labels as input."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AUDIT_ROOT = (
    ROOT
    / "outputs/development/icra_v19_condition_balanced_development_64episode"
    / "grounding_candidate_audit"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "outputs/development/icra_v20_candidate_selection_comparison"
)
ALPHA_GRID = tuple(index / 20.0 for index in range(21))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def load_samples(audit_root: Path) -> list[dict[str, Any]]:
    audit = load_json(audit_root / "audit.json")
    raw: dict[tuple[str, str], dict[str, str]] = {}
    with (audit_root / "all_raw_proposals.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        for row in csv.DictReader(handle):
            raw[(row["sample_id"], row["detection_id"])] = row

    samples = []
    for sample in audit["samples"]:
        trace = list(sample.get("candidate_selection_trace") or [])
        if int(sample.get("target_visible_pixel_count") or 0) <= 0 or not trace:
            continue
        candidates = []
        for candidate in trace:
            detection_id = str(candidate["proposal_detection_id"])
            proposal = raw.get((str(sample["sample_id"]), detection_id))
            overlap_text = proposal.get("target_visible_mask_iou", "") if proposal else ""
            candidates.append(
                {
                    "candidate_id": candidate.get("candidate_id"),
                    "detection_id": detection_id,
                    "qwen_logit": float(candidate["qwen_raw_match_logit"]),
                    "detector_score": float(candidate["proposal_score"]),
                    "correct_at_mask_iou_0_5": bool(
                        overlap_text and float(overlap_text) >= 0.5
                    ),
                }
            )
        samples.append(
            {
                "sample_id": sample["sample_id"],
                "seed": int(sample["seed"]),
                "family": sample["family"],
                "view_id": sample["view_id"],
                "candidates": candidates,
                "target_candidate_available": any(
                    candidate["correct_at_mask_iou_0_5"]
                    for candidate in candidates
                ),
            }
        )
    return samples


def candidate_probabilities(
    sample: dict[str, Any], alpha: float
) -> list[float]:
    candidates = sample["candidates"]
    qwen = softmax([float(row["qwen_logit"]) for row in candidates])
    detector = softmax(
        [math.log(max(float(row["detector_score"]), 1e-8)) for row in candidates]
    )
    scores = [
        alpha * qwen[index] + (1.0 - alpha) * detector[index]
        for index in range(len(candidates))
    ]
    total = sum(scores)
    return [score / total for score in scores]


def decision(sample: dict[str, Any], alpha: float) -> dict[str, Any]:
    probabilities = candidate_probabilities(sample, alpha)
    index = max(range(len(probabilities)), key=probabilities.__getitem__)
    selected = sample["candidates"][index]
    return {
        "selected_detection_id": selected["detection_id"],
        "correct": bool(selected["correct_at_mask_iou_0_5"]),
        "confidence": float(probabilities[index]),
    }


def accuracy(samples: list[dict[str, Any]], alpha: float) -> float:
    return sum(decision(sample, alpha)["correct"] for sample in samples) / len(samples)


def expected_calibration_error(rows: list[dict[str, Any]], bins: int = 10) -> float:
    total = len(rows)
    error = 0.0
    for bin_index in range(bins):
        low, high = bin_index / bins, (bin_index + 1) / bins
        selected = [
            row
            for row in rows
            if low <= row["confidence"] <= high
            and (bin_index == bins - 1 or row["confidence"] < high)
        ]
        if selected:
            mean_confidence = sum(row["confidence"] for row in selected) / len(selected)
            mean_accuracy = sum(row["correct"] for row in selected) / len(selected)
            error += len(selected) / total * abs(mean_confidence - mean_accuracy)
    return error


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0, "accuracy": None}
    return {
        "sample_count": len(rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "mean_selected_confidence": sum(row["confidence"] for row in rows)
        / len(rows),
        "ece_10_bin": expected_calibration_error(rows),
        "brier_selected_correctness": sum(
            (row["confidence"] - float(row["correct"])) ** 2 for row in rows
        )
        / len(rows),
    }


def grouped_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {name: metrics(values) for name, values in sorted(grouped.items())}


def cross_validated_fusion(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for fold in range(8):
        train = [sample for sample in samples if (sample["seed"] - 1270) % 8 != fold]
        test = [sample for sample in samples if (sample["seed"] - 1270) % 8 == fold]
        scored = [(accuracy(train, alpha), alpha) for alpha in ALPHA_GRID]
        # Prefer the simpler Qwen-only selector when validation accuracy ties.
        selected_alpha = max(scored, key=lambda item: (item[0], item[1]))[1]
        for sample in test:
            result = decision(sample, selected_alpha)
            rows.append(
                {
                    **{key: sample[key] for key in ("sample_id", "seed", "family", "view_id")},
                    **result,
                    "selector": "cross_validated_linear_fusion",
                    "qwen_weight": selected_alpha,
                    "fold": fold,
                    "target_candidate_available": sample["target_candidate_available"],
                }
            )
    return rows


def fixed_selector_rows(
    samples: list[dict[str, Any]], *, name: str, alpha: float
) -> list[dict[str, Any]]:
    rows = []
    for sample in samples:
        rows.append(
            {
                **{key: sample[key] for key in ("sample_id", "seed", "family", "view_id")},
                **decision(sample, alpha),
                "selector": name,
                "qwen_weight": alpha,
                "fold": None,
                "target_candidate_available": sample["target_candidate_available"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    samples = load_samples(args.audit_root.resolve())
    selectors = {
        "qwen_only": fixed_selector_rows(samples, name="qwen_only", alpha=1.0),
        "detector_only": fixed_selector_rows(samples, name="detector_only", alpha=0.0),
        "cross_validated_linear_fusion": cross_validated_fusion(samples),
    }
    comparison = {}
    for name, rows in selectors.items():
        comparison[name] = {
            "overall": metrics(rows),
            "by_view": grouped_metrics(rows, "view_id"),
            "by_family": grouped_metrics(rows, "family"),
        }

    all_rows = [row for rows in selectors.values() for row in rows]
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / "per_sample.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(
        args.output_root / "result.json",
        {
            "schema_version": "icra-v20-candidate-selection-comparison-v1",
            "status": "completed",
            "source": str(args.audit_root.resolve()),
            "sample_count": len(samples),
            "simulator_masks_used_for_calibration_and_evaluation_only": True,
            "simulator_masks_available_to_selector": False,
            "training_performed": False,
            "calibration_performed": True,
            "testing_performed": False,
            "valid_for_final_evaluation": False,
            "comparison": comparison,
            "conclusion": (
                "Cached detector-score fusion does not repair the post-remove "
                "semantic ranking error when cross-validation selects Qwen-only. "
                "The next development experiment must improve post-remove candidate "
                "crops/prompts or rerun semantic scoring; reserved test data remains unopened."
            ),
        },
    )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
