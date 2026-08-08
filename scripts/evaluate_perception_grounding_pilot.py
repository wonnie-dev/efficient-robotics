"""Evaluate saved perception outputs against simulator-only hidden labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from rgbd_target_localization import estimate_mask_center


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/perception/grounding_pilot_seed0_2.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def bbox_from_mask(mask: np.ndarray) -> list[float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def bbox_iou(first: list[float] | None, second: list[float] | None) -> float:
    if first is None or second is None:
        return 0.0
    ax0, ay0, ax1, ay1 = first
    bx0, by0, bx1, by1 = second
    intersection_width = max(0.0, min(ax1, bx1) - max(ax0, bx0) + 1.0)
    intersection_height = max(0.0, min(ay1, by1) - max(ay0, by0) + 1.0)
    intersection = intersection_width * intersection_height
    first_area = max(0.0, ax1 - ax0 + 1.0) * max(0.0, ay1 - ay0 + 1.0)
    second_area = max(0.0, bx1 - bx0 + 1.0) * max(0.0, by1 - by0 + 1.0)
    union = first_area + second_area - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 0.0


def box_mask(shape: tuple[int, int], box: list[float] | None) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if box is None:
        return mask
    x0, y0, x1, y1 = [int(round(value)) for value in box]
    x0 = max(0, min(shape[1] - 1, x0))
    x1 = max(0, min(shape[1] - 1, x1))
    y0 = max(0, min(shape[0] - 1, y0))
    y1 = max(0, min(shape[0] - 1, y1))
    if x1 >= x0 and y1 >= y0:
        mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def semantic_class_masks(observation_dir: Path) -> dict[str, np.ndarray]:
    instance_ids = np.load(observation_dir / "instance_ids.npy")
    labels = json.loads(
        (observation_dir / "instance_labels.json").read_text(encoding="utf-8")
    )
    return {
        value["class"]: instance_ids == int(instance_id)
        for instance_id, value in labels.items()
    }


def normalize_label(label: str, concepts: list[str]) -> str | None:
    lowered = label.lower().strip().rstrip(".")
    exact = [concept for concept in concepts if concept == lowered]
    if exact:
        return exact[0]
    contained = [
        concept
        for concept in concepts
        if concept in lowered or lowered in concept
    ]
    return max(contained, key=len) if contained else None


def center_error_m(
    observation_dir: Path,
    predicted_mask: np.ndarray,
    ground_truth_mask: np.ndarray,
) -> float | None:
    calibration_path = observation_dir / "camera_calibration.json"
    if not calibration_path.is_file():
        return None
    depth = np.load(observation_dir / "depth_m.npy")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    try:
        predicted = estimate_mask_center(
            depth, predicted_mask, calibration, label="predicted"
        )
        expected = estimate_mask_center(
            depth, ground_truth_mask, calibration, label="ground_truth"
        )
    except ValueError:
        return None
    predicted_center = np.asarray(predicted["center_world_m"], dtype=np.float64)
    expected_center = np.asarray(expected["center_world_m"], dtype=np.float64)
    return float(np.linalg.norm(predicted_center - expected_center))


def evaluate_qwen(
    config: dict[str, Any],
    sample: dict[str, Any],
    gt_masks: dict[str, np.ndarray],
) -> dict[str, Any] | None:
    root = resolve_project_path(config["output_root"])
    path = root / "qwen_direct" / sample["sample_id"] / "result.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    expected = gt_masks["target_red"]
    predicted = box_mask(expected.shape, result.get("bbox_xyxy_pixels"))
    return {
        "method": "qwen_direct_bbox",
        "sample_id": sample["sample_id"],
        "semantic_class": "target_red",
        "concept": "relation_conditioned_red_target",
        "matched": result.get("bbox_xyxy_pixels") is not None,
        "bbox_iou": bbox_iou(
            result.get("bbox_xyxy_pixels"), bbox_from_mask(expected)
        ),
        "mask_iou": None,
        "centroid_error_m": center_error_m(
            resolve_project_path(sample["observation_dir"]), predicted, expected
        ),
        "score": None,
        "parse_status": result["parse_status"],
    }


def evaluate_grounded_qwen_selection(
    config: dict[str, Any],
    sample: dict[str, Any],
    gt_masks: dict[str, np.ndarray],
) -> dict[str, Any] | None:
    root = resolve_project_path(config["output_root"])
    ranking_path = (
        root
        / "grounded_sam2_qwen_rankings"
        / sample["sample_id"]
        / "result.json"
    )
    if not ranking_path.is_file():
        return None
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    model_input = json.loads(
        Path(ranking["input_path"]).read_text(encoding="utf-8")
    )
    candidate = next(
        candidate
        for candidate in model_input["candidates"]
        if candidate["candidate_id"] == ranking["selected_candidate_id"]
    )
    mask = np.asarray(
        Image.open(resolve_project_path(candidate["mask_path"])).convert("L")
    ) > 0
    expected = gt_masks["target_red"]
    return {
        "method": "grounded_sam2_qwen_selected_mask",
        "sample_id": sample["sample_id"],
        "semantic_class": "target_red",
        "concept": "anonymous_red_proposals_then_qwen_relation_selection",
        "matched": True,
        "bbox_iou": bbox_iou(
            candidate["bbox_xyxy"], bbox_from_mask(expected)
        ),
        "mask_iou": mask_iou(mask, expected),
        "centroid_error_m": center_error_m(
            resolve_project_path(sample["observation_dir"]), mask, expected
        ),
        "score": max(ranking["raw_match_logits"]),
        "parse_status": "not_applicable",
    }


def evaluate_selected_relation(
    config: dict[str, Any],
    sample: dict[str, Any],
    selected_mask_iou: float,
) -> dict[str, Any] | None:
    root = resolve_project_path(config["output_root"])
    ranking_path = (
        root
        / "grounded_sam2_qwen_rankings"
        / sample["sample_id"]
        / "result.json"
    )
    if not ranking_path.is_file():
        return None
    ranking = json.loads(ranking_path.read_text(encoding="utf-8"))
    relation = next(
        (
            item
            for item in ranking.get("relations", [])
            if item["source_id"] == ranking["selected_candidate_id"]
            and item.get("relation_type") == "membership"
        ),
        ranking.get("selected_candidate_relation"),
    )
    if relation is None:
        return None
    expected = config["evaluation"].get("expected_selected_relation")
    selected_entity = None
    if sample.get("calibration_ground_truth_file"):
        model_input = json.loads(
            Path(ranking["input_path"]).read_text(encoding="utf-8")
        )
        candidate = next(
            item
            for item in model_input["candidates"]
            if item["candidate_id"] == ranking["selected_candidate_id"]
        )
        candidate_mask = np.asarray(
            Image.open(
                resolve_project_path(candidate["mask_path"])
            ).convert("L")
        ) > 0
        observation_dir = resolve_project_path(sample["observation_dir"])
        gt_masks = semantic_class_masks(observation_dir)
        target_iou = mask_iou(candidate_mask, gt_masks["target_red"])
        distractor_iou = mask_iou(
            candidate_mask, gt_masks["rear_red_candidate"]
        )
        if target_iou >= 0.5 and target_iou >= distractor_iou:
            selected_entity = "target_red"
        elif distractor_iou >= 0.5:
            selected_entity = "rear_red_candidate"
        if selected_entity is not None:
            ground_truth = json.loads(
                Path(sample["calibration_ground_truth_file"]).read_text(
                    encoding="utf-8"
                )
            )
            view_id = observation_dir.name
            expected = ground_truth["view_observable_intent"][view_id][
                "entities"
            ][selected_entity]["membership_observable"]
    selection_threshold = float(
        config["evaluation"]["mask_iou_recall_threshold"]
    )
    return {
        "sample_id": sample["sample_id"],
        "selected_candidate_id": ranking["selected_candidate_id"],
        "selected_simulator_entity": selected_entity,
        "selected_target_mask_iou": selected_mask_iou,
        "selected_target_correct_at_0_5": (
            selected_mask_iou >= selection_threshold
        ),
        "labels": relation["labels"],
        "raw_logits": relation["raw_logits"],
        "top_label": relation["top_label"],
        "expected_label": expected,
        "relation_correct": (
            relation["top_label"] == expected
            if expected is not None
            else None
        ),
        "scores_calibrated": False,
    }


def load_specialist_predictions(
    config: dict[str, Any],
    sample: dict[str, Any],
    method: str,
) -> list[dict[str, Any]] | None:
    root = resolve_project_path(config["output_root"])
    if method == "grounded_sam2":
        sample_root = root / "grounded_sam2" / sample["sample_id"]
    elif method == "sam3":
        sample_root = root / "sam3" / sample["sample_id"]
    else:
        raise ValueError(method)
    path = sample_root / "segmentations.json"
    if not path.is_file():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    predictions = []
    concepts = list(config["evaluation"]["semantic_groups"])
    for annotation in result["annotations"]:
        raw_label = annotation.get("label", annotation.get("concept", ""))
        concept = normalize_label(raw_label, concepts)
        mask = np.asarray(
            Image.open(sample_root / annotation["mask_path"]).convert("L")
        ) > 0
        predictions.append(
            {
                "concept": concept,
                "raw_label": raw_label,
                "score": float(annotation["score"]),
                "bbox": annotation["bbox_xyxy_pixels"],
                "mask": mask,
            }
        )
    return predictions


def match_concept_predictions(
    observation_dir: Path,
    method: str,
    sample_id: str,
    concept: str,
    semantic_classes: list[str],
    predictions: list[dict[str, Any]],
    gt_masks: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    concept_predictions = [
        prediction for prediction in predictions if prediction["concept"] == concept
    ]
    rows = []
    if concept_predictions:
        costs = np.zeros((len(semantic_classes), len(concept_predictions)))
        for row_index, semantic_class in enumerate(semantic_classes):
            for column_index, prediction in enumerate(concept_predictions):
                costs[row_index, column_index] = -mask_iou(
                    gt_masks[semantic_class], prediction["mask"]
                )
        matched_rows, matched_columns = linear_sum_assignment(costs)
        matches = dict(zip(matched_rows.tolist(), matched_columns.tolist()))
    else:
        matches = {}

    for class_index, semantic_class in enumerate(semantic_classes):
        expected = gt_masks[semantic_class]
        prediction = (
            concept_predictions[matches[class_index]]
            if class_index in matches
            else None
        )
        rows.append(
            {
                "method": method,
                "sample_id": sample_id,
                "semantic_class": semantic_class,
                "concept": concept,
                "matched": prediction is not None,
                "bbox_iou": (
                    bbox_iou(prediction["bbox"], bbox_from_mask(expected))
                    if prediction is not None
                    else 0.0
                ),
                "mask_iou": (
                    mask_iou(prediction["mask"], expected)
                    if prediction is not None
                    else 0.0
                ),
                "centroid_error_m": (
                    center_error_m(
                        observation_dir, prediction["mask"], expected
                    )
                    if prediction is not None
                    else None
                ),
                "score": prediction["score"] if prediction is not None else None,
                "parse_status": "not_applicable",
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    mask_threshold = float(config["evaluation"]["mask_iou_recall_threshold"])
    box_threshold = float(config["evaluation"]["bbox_iou_recall_threshold"])
    summaries = {}
    for method in sorted({row["method"] for row in rows}):
        method_rows = [row for row in rows if row["method"] == method]
        mask_rows = [row for row in method_rows if row["mask_iou"] is not None]
        center_values = [
            row["centroid_error_m"]
            for row in method_rows
            if row["centroid_error_m"] is not None
        ]
        summaries[method] = {
            "evaluated_instances": len(method_rows),
            "assigned_predictions": sum(bool(row["matched"]) for row in method_rows),
            "bbox_true_positives_at_0_5": sum(
                row["bbox_iou"] >= box_threshold for row in method_rows
            ),
            "mean_bbox_iou": (
                float(np.mean([row["bbox_iou"] for row in method_rows]))
                if method_rows
                else None
            ),
            "bbox_recall_at_0_5": (
                sum(row["bbox_iou"] >= box_threshold for row in method_rows)
                / len(method_rows)
                if method_rows
                else None
            ),
            "mean_mask_iou": (
                float(np.mean([row["mask_iou"] for row in mask_rows]))
                if mask_rows
                else None
            ),
            "mask_recall_at_0_5": (
                sum(row["mask_iou"] >= mask_threshold for row in mask_rows)
                / len(mask_rows)
                if mask_rows
                else None
            ),
            "mean_centroid_error_m": (
                float(np.mean(center_values)) if center_values else None
            ),
            "centroid_error_sample_count": len(center_values),
        }
    return summaries


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    proposal_counts: list[dict[str, Any]] = []
    selected_relation_rows: list[dict[str, Any]] = []
    for sample in config["samples"]:
        observation_dir = resolve_project_path(sample["observation_dir"])
        gt_masks = semantic_class_masks(observation_dir)
        qwen_row = evaluate_qwen(config, sample, gt_masks)
        if qwen_row is not None:
            rows.append(qwen_row)
        grounded_qwen_row = evaluate_grounded_qwen_selection(
            config, sample, gt_masks
        )
        if grounded_qwen_row is not None:
            rows.append(grounded_qwen_row)
            selected_relation = evaluate_selected_relation(
                config,
                sample,
                float(grounded_qwen_row["mask_iou"]),
            )
            if selected_relation is not None:
                selected_relation_rows.append(selected_relation)
        for method in ("grounded_sam2", "sam3"):
            predictions = load_specialist_predictions(config, sample, method)
            if predictions is None:
                continue
            sample_rows = []
            for concept, semantic_classes in config["evaluation"][
                "semantic_groups"
            ].items():
                sample_rows.extend(
                    match_concept_predictions(
                        observation_dir,
                        method,
                        sample["sample_id"],
                        concept,
                        semantic_classes,
                        predictions,
                        gt_masks,
                    )
                )
            rows.extend(sample_rows)
            mask_threshold = float(
                config["evaluation"]["mask_iou_recall_threshold"]
            )
            proposal_counts.append(
                {
                    "method": method,
                    "sample_id": sample["sample_id"],
                    "predicted_proposals": len(predictions),
                    "ground_truth_instances": len(sample_rows),
                    "true_positives_at_mask_iou_0_5": sum(
                        row["mask_iou"] is not None
                        and row["mask_iou"] >= mask_threshold
                        for row in sample_rows
                    ),
                }
            )

    output_root = resolve_project_path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "evaluation_rows.csv"
    fields = [
        "method",
        "sample_id",
        "semantic_class",
        "concept",
        "matched",
        "bbox_iou",
        "mask_iou",
        "centroid_error_m",
        "score",
        "parse_status",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summaries = summarize(rows, config)
    for method in sorted({item["method"] for item in proposal_counts}):
        method_counts = [
            item for item in proposal_counts if item["method"] == method
        ]
        predicted = sum(item["predicted_proposals"] for item in method_counts)
        true_positives = sum(
            item["true_positives_at_mask_iou_0_5"] for item in method_counts
        )
        summaries[method].update(
            {
                "predicted_proposals": predicted,
                "false_positive_or_duplicate_proposals_at_0_5": max(
                    0, predicted - true_positives
                ),
                "proposal_precision_at_mask_iou_0_5": (
                    true_positives / predicted if predicted else None
                ),
            }
        )

    payload = {
        "schema_version": "perception-grounding-pilot-evaluation-v1",
        "experiment_id": config["experiment_id"],
        "summaries": summaries,
        "per_sample_proposal_counts": proposal_counts,
        "selected_relation_evaluation": {
            "sample_count": len(selected_relation_rows),
            "target_selection_correct_count": sum(
                row["selected_target_correct_at_0_5"]
                for row in selected_relation_rows
            ),
            "relation_correct_count": sum(
                row["relation_correct"] is True
                for row in selected_relation_rows
            ),
            "rows": selected_relation_rows,
        },
        "row_count": len(rows),
        "evaluation_rows_csv": str(csv_path),
        "simulator_ground_truth_used_for_inference": False,
        "simulator_ground_truth_used_for_evaluation": True,
        "training_performed": False,
        "calibration_performed": False,
        "valid_for_final_evaluation": False,
        "limitations": config.get(
            "limitations",
            [
                "The pilot contains colored geometric debug objects, not final household objects.",
                "Seed 0 lacks saved camera calibration, so its 3D centroid error is unavailable.",
                "Qwen direct grounding predicts a target box; specialist methods generate candidate masks and are not an apples-to-apples task.",
                "Scores and thresholds are uncalibrated pilot values.",
            ],
        ),
    }
    metrics_path = output_root / "evaluation_summary.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
