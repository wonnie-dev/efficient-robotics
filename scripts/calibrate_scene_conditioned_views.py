#!/usr/bin/env python3
"""Calibrate post-remove view resolvability from planner-visible features."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = (
    ROOT / "outputs/calibration/calibration_episodes_perception",
    ROOT / "outputs/calibration/supplemental_perception",
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/calibration/scene_conditioned_view_calibration"
    / "result.json"
)
VIEW_MODES = ("close_high", "right", "none")
FEATURE_NAMES = (
    "selected_match_logit",
    "candidate_count",
    "membership_inside_minus_outside",
    "membership_best_minus_unknown",
    "selected_bbox_center_x_fraction",
    "selected_bbox_center_y_fraction",
    "selected_bbox_area_fraction",
    "container_bbox_center_x_fraction",
    "container_bbox_center_y_fraction",
    "container_bbox_area_fraction",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def bbox_features(bbox: list[float], width: float, height: float) -> tuple[float, float, float]:
    x0, y0, x1, y1 = (float(value) for value in bbox)
    return (
        (x0 + x1) / (2.0 * width),
        (y0 + y1) / (2.0 * height),
        max(0.0, x1 - x0) * max(0.0, y1 - y0) / (width * height),
    )


def relation(result: dict[str, Any], candidate_id: str, kind: str) -> dict[str, Any] | None:
    return next(
        (
            row
            for row in result.get("relations", [])
            if row.get("source_id") == candidate_id
            and row.get("relation_type") == kind
        ),
        None,
    )


def view_mode(family: str) -> str:
    mapping = {
        "inside_close_high_resolving": "close_high",
        "inside_right_resolving": "right",
        "target_absent_negative_evidence": "none",
        "remove_cover_failure_or_no_gain": "none",
        "outside_other_target_right_resolving": "right",
        "covered_close_high_resolving": "close_high",
        "covered_right_resolving": "right",
        "target_absent_covered": "none",
    }
    if family not in mapping:
        raise ValueError(f"No post-remove view mode for family {family}")
    return mapping[family]


def extract_features(perception_root: Path, seed: int) -> dict[str, Any]:
    sample_id = f"seed{seed}_post_remove"
    result_path = (
        perception_root / "grounded_sam2_qwen_rankings" / sample_id / "result.json"
    )
    result = load_json(result_path)
    model_input = load_json(Path(result["input_path"]))
    selected = str(result["selected_candidate_id"])
    index = list(result["candidate_ids"]).index(selected)
    candidate = next(
        row for row in model_input["candidates"] if row["candidate_id"] == selected
    )
    width = float(model_input["image"]["width"])
    height = float(model_input["image"]["height"])
    selected_bbox = bbox_features(candidate["bbox_xyxy"], width, height)
    reference = model_input["reference_entities"][0]
    container_bbox = bbox_features(reference["bbox_xyxy"], width, height)
    membership = relation(result, selected, "membership")
    if membership is None:
        membership_scores = {"inside": 0.0, "outside": 0.0, "unknown": 0.0}
    else:
        membership_scores = {
            str(label): float(value)
            for label, value in zip(membership["labels"], membership["raw_logits"])
        }
    values = [
        float(result["raw_match_logits"][index]),
        float(len(result["candidate_ids"])),
        membership_scores.get("inside", 0.0) - membership_scores.get("outside", 0.0),
        max(
            membership_scores.get("inside", 0.0),
            membership_scores.get("outside", 0.0),
        ) - membership_scores.get("unknown", 0.0),
        *selected_bbox,
        *container_bbox,
    ]
    return {
        "sample_id": sample_id,
        "values": values,
        "named_values": dict(zip(FEATURE_NAMES, values)),
        "inputs": [str(result_path.resolve()), str(Path(result["input_path"]).resolve())],
        "simulator_ground_truth_used": False,
    }


def candidate_set_feature_names(max_candidates: int = 4) -> tuple[str, ...]:
    names = ["candidate_count"]
    for index in range(max_candidates):
        prefix = f"candidate_x_sorted_{index}"
        names.extend(
            [
                f"{prefix}_match_logit",
                f"{prefix}_membership_inside_minus_outside",
                f"{prefix}_membership_best_minus_unknown",
                f"{prefix}_bbox_center_x_fraction",
                f"{prefix}_bbox_center_y_fraction",
                f"{prefix}_bbox_area_fraction",
                f"{prefix}_proposal_score",
            ]
        )
    names.extend(
        [
            "container_bbox_center_x_fraction",
            "container_bbox_center_y_fraction",
            "container_bbox_area_fraction",
        ]
    )
    return tuple(names)


def extract_candidate_set_features(
    perception_root: Path, seed: int, *, max_candidates: int = 4
) -> dict[str, Any]:
    """Represent every anonymous candidate instead of only Qwen's top choice."""
    sample_id = f"seed{seed}_post_remove"
    result_path = (
        perception_root / "grounded_sam2_qwen_rankings" / sample_id / "result.json"
    )
    result = load_json(result_path)
    input_path = Path(result["input_path"])
    model_input = load_json(input_path)
    width = float(model_input["image"]["width"])
    height = float(model_input["image"]["height"])
    logits = {
        str(candidate_id): float(logit)
        for candidate_id, logit in zip(
            result["candidate_ids"], result["raw_match_logits"]
        )
    }
    rows = []
    for candidate in model_input["candidates"]:
        candidate_id = str(candidate["candidate_id"])
        membership = relation(result, candidate_id, "membership")
        scores = (
            {
                str(label): float(value)
                for label, value in zip(
                    membership["labels"], membership["raw_logits"]
                )
            }
            if membership is not None
            else {"inside": 0.0, "outside": 0.0, "unknown": 0.0}
        )
        bbox = bbox_features(candidate["bbox_xyxy"], width, height)
        rows.append(
            [
                logits[candidate_id],
                scores.get("inside", 0.0) - scores.get("outside", 0.0),
                max(scores.get("inside", 0.0), scores.get("outside", 0.0))
                - scores.get("unknown", 0.0),
                *bbox,
                float(candidate.get("proposal_score", 0.0)),
            ]
        )
    rows.sort(key=lambda values: (values[3], values[4], values[5]))
    values = [float(len(rows))]
    zero_candidate = [0.0] * 7
    for index in range(max_candidates):
        values.extend(rows[index] if index < len(rows) else zero_candidate)
    reference = model_input["reference_entities"][0]
    values.extend(bbox_features(reference["bbox_xyxy"], width, height))
    names = candidate_set_feature_names(max_candidates)
    return {
        "sample_id": sample_id,
        "values": values,
        "named_values": dict(zip(names, values)),
        "feature_extractor": "candidate_set_x_sorted_v1",
        "max_candidates": max_candidates,
        "inputs": [str(result_path.resolve()), str(input_path.resolve())],
        "simulator_ground_truth_used": False,
    }


def load_rows(perception_roots: tuple[Path, ...]) -> list[dict[str, Any]]:
    rows = []
    for root in perception_roots:
        manifest = load_json(root / "observation_manifest.json")
        for episode in manifest["episodes"]:
            if "post_remove" not in episode["observations"]:
                continue
            family = str(episode["family"])
            rows.append(
                {
                    "seed": int(episode["seed"]),
                    "family": family,
                    "view_mode": view_mode(family),
                    "features": extract_features(root, int(episode["seed"])),
                    "automatic_simulator_label_used_for_calibration_only": True,
                }
            )
    return rows


def statistics(rows: list[dict[str, Any]]) -> tuple[list[float], list[float]]:
    dimension = len(rows[0]["features"]["values"])
    if any(len(row["features"]["values"]) != dimension for row in rows):
        raise ValueError("View-model feature dimensions are inconsistent")
    mean = [
        sum(float(row["features"]["values"][i]) for row in rows) / len(rows)
        for i in range(dimension)
    ]
    std = []
    for i, center in enumerate(mean):
        variance = sum(
            (float(row["features"]["values"][i]) - center) ** 2 for row in rows
        ) / len(rows)
        std.append(math.sqrt(variance) or 1.0)
    return mean, std


def predict(
    query: dict[str, Any], training: list[dict[str, Any]], *, k: int, beta: float
) -> dict[str, Any]:
    mean, std = statistics(training)
    if len(query["values"]) != len(mean):
        raise ValueError("Query and view-model feature dimensions differ")
    neighbors = []
    for row in training:
        distance = math.sqrt(
            sum(
                (
                    (float(query["values"][i]) - float(row["features"]["values"][i]))
                    / std[i]
                ) ** 2
                for i in range(len(mean))
            )
        )
        neighbors.append(
            {"seed": row["seed"], "view_mode": row["view_mode"], "distance": distance}
        )
    neighbors.sort(key=lambda row: (row["distance"], row["seed"]))
    neighbors = neighbors[: min(k, len(neighbors))]
    weights = {mode: float(beta) for mode in VIEW_MODES}
    for neighbor in neighbors:
        weight = 1.0 / (float(neighbor["distance"]) + 0.1)
        neighbor["weight"] = weight
        weights[str(neighbor["view_mode"])] += weight
    total = sum(weights.values())
    return {
        "probabilities": {mode: weights[mode] / total for mode in VIEW_MODES},
        "neighbors": neighbors,
    }


def select_k(rows: list[dict[str, Any]], grid: tuple[int, ...], beta: float) -> dict[str, Any]:
    candidates = []
    for k in grid:
        loss = 0.0
        for held_out in rows:
            training = [row for row in rows if row["seed"] != held_out["seed"]]
            probability = predict(held_out["features"], training, k=k, beta=beta)[
                "probabilities"
            ][held_out["view_mode"]]
            loss -= math.log(max(1e-12, probability))
        candidates.append({"k": k, "leave_one_out_nll": loss / len(rows)})
    selected = min(candidates, key=lambda row: (row["leave_one_out_nll"], row["k"]))
    return {"selected_k": selected["k"], "candidates": candidates}


def expected_calibration_error(folds: list[dict[str, Any]], bins: int = 10) -> float:
    total = len(folds)
    ece = 0.0
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        members = [
            row
            for row in folds
            if lower <= row["confidence"] < upper
            or (index == bins - 1 and row["confidence"] == 1.0)
        ]
        if not members:
            continue
        accuracy = sum(row["correct"] for row in members) / len(members)
        confidence = sum(row["confidence"] for row in members) / len(members)
        ece += len(members) / total * abs(accuracy - confidence)
    return ece


def run(perception_roots: tuple[Path, ...], output: Path) -> dict[str, Any]:
    started = time.perf_counter()
    rows = load_rows(perception_roots)
    k_grid = (1, 3, 5, 7, 9)
    beta = 0.1
    folds = []
    for held_out in rows:
        outer_training = [row for row in rows if row["seed"] != held_out["seed"]]
        inner = select_k(outer_training, k_grid, beta)
        prediction = predict(
            held_out["features"],
            outer_training,
            k=int(inner["selected_k"]),
            beta=beta,
        )
        predicted = max(prediction["probabilities"], key=prediction["probabilities"].get)
        folds.append(
            {
                "seed": held_out["seed"],
                "family": held_out["family"],
                "true_view_mode": held_out["view_mode"],
                "predicted_view_mode": predicted,
                "confidence": prediction["probabilities"][predicted],
                "correct": predicted == held_out["view_mode"],
                "probabilities": prediction["probabilities"],
                "neighbors": prediction["neighbors"],
                "inner_k_selection": inner,
                "held_out_episode_used_for_fit": False,
                "held_out_future_view_used": False,
            }
        )
    full_k = select_k(rows, k_grid, beta)
    accuracy = sum(row["correct"] for row in folds) / len(folds)
    nll = -sum(
        math.log(max(1e-12, row["probabilities"][row["true_view_mode"]]))
        for row in folds
    ) / len(folds)
    result = {
        "schema_version": "scene-conditioned-view-calibration-v1",
        "status": "passed" if accuracy >= 0.8 else "failed",
        "protocol": "nested_leave_one_episode_out_knn_calibration",
        "episode_count": len(rows),
        "feature_names": list(FEATURE_NAMES),
        "view_mode_support": dict(Counter(row["view_mode"] for row in rows)),
        "accuracy": accuracy,
        "negative_log_likelihood": nll,
        "expected_calibration_error": expected_calibration_error(folds),
        "folds": folds,
        "frozen_calibration_candidate": {
            "episodes": rows,
            "neighbor_count": full_k["selected_k"],
            "k_selection": full_k,
            "probability_pseudocount": beta,
            "feature_names": list(FEATURE_NAMES),
        },
        "simulator_ground_truth_used_for_features": False,
        "simulator_ground_truth_used_for_calibration_labels": True,
        "training_performed": False,
        "calibration_performed": True,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
        "gpu_used": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    roots = tuple(path.resolve() for path in args.perception_root) or DEFAULT_ROOTS
    result = run(roots, args.output.resolve())
    print(json.dumps({key: result[key] for key in ("status", "episode_count", "accuracy", "negative_log_likelihood", "expected_calibration_error", "runtime_seconds")}, indent=2))


if __name__ == "__main__":
    main()
