#!/usr/bin/env python3
"""Export every GroundingDINO proposal and its downstream selection trace."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERCEPTION_ROOT = (
    ROOT / "outputs/calibration/icra_v16_calibration_perception"
)
DEFAULT_POLICY_RESULT = (
    ROOT
    / "outputs/calibration/icra_v18_persistent_negative_evidence_candidate/result.json"
)
DEFAULT_OUTPUT_ROOT = (
    ROOT / "outputs/analysis/professor_feedback_grounding_candidate_audit_20260819"
)
REFERENCE_CONCEPTS = {"open container", "lid or cover"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def binary_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L")) > 0


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = np.logical_or(first, second).sum()
    if not union:
        return 0.0
    return float(np.logical_and(first, second).sum() / union)


def bbox_iou(first: list[float], second: list[float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = area_first + area_second - intersection
    return float(intersection / union) if union else 0.0


def count_spatial_clusters(rows: list[dict[str, Any]], threshold: float = 0.8) -> int:
    representatives: list[list[float]] = []
    for row in sorted(rows, key=lambda item: float(item["score"]), reverse=True):
        box = [float(value) for value in row["bbox_xyxy_pixels"]]
        if not any(bbox_iou(box, existing) >= threshold for existing in representatives):
            representatives.append(box)
    return len(representatives)


def policy_index(path: Path) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}
    result = load_json(path)
    return {int(row["seed"]): row for row in result.get("episodes", [])}


def corresponding_path(detection_path: Path, stage: str, filename: str) -> Path:
    shard_root = detection_path.parents[2]
    sample_id = detection_path.parent.name
    return shard_root / stage / sample_id / filename


def draw_overlay(
    image_path: Path,
    annotations: list[dict[str, Any]],
    selected_ids: set[str],
    best_target_id: str | None,
    output_path: Path,
) -> None:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for row in annotations:
        detection_id = str(row["detection_id"])
        box = tuple(float(value) for value in row["bbox_xyxy_pixels"])
        if detection_id in selected_ids:
            color, width = "lime", 4
        elif detection_id == best_target_id:
            color, width = "cyan", 3
        elif row["concept"] in REFERENCE_CONCEPTS:
            color, width = "orange", 2
        else:
            color, width = "red", 1
        draw.rectangle(box, outline=color, width=width)
        label = f"{detection_id} {row['concept']} {float(row['score']):.3f}"
        text_box = (
            box[0],
            box[1],
            box[0] + max(1, len(label)) * 6,
            box[1] + 11,
        )
        draw.rectangle(text_box, fill="black")
        draw.text((box[0], box[1]), label, fill=color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def audit_sample(
    detection_path: Path,
    policies: dict[int, dict[str, Any]],
    overlay_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    detections = load_json(detection_path)
    sample_id = str(detections["sample_id"])
    image_path = Path(detections["image_path"])
    observation_dir = image_path.parent
    target_path = observation_dir / "target_visible_mask.png"
    target_mask = binary_mask(target_path) if target_path.exists() else None
    target_visible_pixels = int(target_mask.sum()) if target_mask is not None else 0

    input_path = corresponding_path(
        detection_path, "grounded_sam2_qwen_inputs", "input.json"
    )
    ranking_path = corresponding_path(
        detection_path, "grounded_sam2_qwen_rankings", "result.json"
    )
    vlm_input = load_json(input_path) if input_path.exists() else {}
    ranking = load_json(ranking_path) if ranking_path.exists() else {}
    candidate_rows = vlm_input.get("candidates", [])
    selected_proposal_ids = {
        str(row["proposal_detection_id"])
        for row in candidate_rows
        if row.get("proposal_detection_id")
    }
    selected_candidate = ranking.get("selected_candidate_id")
    selected_detection_id = next(
        (
            str(row["proposal_detection_id"])
            for row in candidate_rows
            if row.get("candidate_id") == selected_candidate
        ),
        None,
    )

    annotations: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for annotation in detections.get("annotations", []):
        row = dict(annotation)
        mask_path = detection_path.parent / str(row["detection_id"] + "_mask.png")
        overlap = None
        if target_mask is not None and target_visible_pixels and mask_path.exists():
            overlap = mask_iou(binary_mask(mask_path), target_mask)
        row["target_visible_mask_iou"] = overlap
        annotations.append(row)

    ranked_objects = sorted(
        [row for row in annotations if row["concept"] not in REFERENCE_CONCEPTS],
        key=lambda item: float(item["score"]),
        reverse=True,
    )
    for rank, row in enumerate(ranked_objects, start=1):
        row["object_score_rank"] = rank

    overlap_rows = [
        row for row in ranked_objects if row["target_visible_mask_iou"] is not None
    ]
    best_target = max(
        overlap_rows,
        key=lambda item: float(item["target_visible_mask_iou"]),
        default=None,
    )
    best_target_id = str(best_target["detection_id"]) if best_target else None
    selected_row = next(
        (
            row
            for row in annotations
            if str(row["detection_id"]) == selected_detection_id
        ),
        None,
    )

    seed = int(sample_id.split("_", 1)[0].replace("seed", ""))
    view_id = sample_id.split("_", 1)[1]
    family = image_path.parents[4].name if len(image_path.parents) > 4 else "unknown"
    policy = policies.get(seed, {})
    for row in annotations:
        raw_rows.append(
            {
                "sample_id": sample_id,
                "seed": seed,
                "family": family,
                "view_id": view_id,
                "detection_id": row["detection_id"],
                "concept": row["concept"],
                "score": float(row["score"]),
                "bbox_xyxy_pixels": row["bbox_xyxy_pixels"],
                "target_visible_mask_iou": row["target_visible_mask_iou"],
                "passed_to_qwen": str(row["detection_id"]) in selected_proposal_ids,
                "qwen_selected": str(row["detection_id"]) == selected_detection_id,
                "object_score_rank": row.get("object_score_rank"),
            }
        )

    overlay_path = overlay_root / f"{sample_id}.png"
    draw_overlay(
        image_path,
        annotations,
        selected_proposal_ids,
        best_target_id,
        overlay_path,
    )
    object_rows = [row for row in annotations if row["concept"] not in REFERENCE_CONCEPTS]
    summary = {
        "sample_id": sample_id,
        "seed": seed,
        "family": family,
        "view_id": view_id,
        "image_path": str(image_path),
        "overlay_path": str(overlay_path),
        "raw_detection_count": len(annotations),
        "raw_object_detection_count": len(object_rows),
        "raw_reference_detection_count": len(annotations) - len(object_rows),
        "unique_object_spatial_cluster_count_at_bbox_iou_0_8": count_spatial_clusters(
            object_rows
        ),
        "qwen_candidate_count": len(candidate_rows),
        "candidate_selection_trace": [
            {
                "candidate_id": row.get("candidate_id"),
                "proposal_detection_id": row.get("proposal_detection_id"),
                "proposal_score": row.get("proposal_score"),
                "qwen_raw_match_logit": (
                    ranking.get("raw_match_logits", [])[index]
                    if index < len(ranking.get("raw_match_logits", []))
                    else None
                ),
                "qwen_selected": row.get("candidate_id") == selected_candidate,
            }
            for index, row in enumerate(candidate_rows)
        ],
        "qwen_selected_candidate_id": selected_candidate,
        "qwen_selected_detection_id": selected_detection_id,
        "selected_detection_score": (
            float(selected_row["score"]) if selected_row else None
        ),
        "selected_target_mask_iou_posthoc": (
            selected_row["target_visible_mask_iou"] if selected_row else None
        ),
        "target_visible_pixel_count": target_visible_pixels,
        "best_target_detection_id_posthoc": best_target_id,
        "best_target_detection_score_posthoc": (
            float(best_target["score"]) if best_target else None
        ),
        "best_target_mask_iou_posthoc": (
            float(best_target["target_visible_mask_iou"]) if best_target else None
        ),
        "target_proposal_recalled_at_mask_iou_0_5": bool(
            best_target
            and float(best_target["target_visible_mask_iou"]) >= 0.5
        ),
        "selected_target_correct_at_mask_iou_0_5": bool(
            selected_row
            and selected_row["target_visible_mask_iou"] is not None
            and float(selected_row["target_visible_mask_iou"]) >= 0.5
        ),
        "semantic_decision_correct": policy.get("semantic_decision_correct"),
        "wrong_commitment": policy.get("wrong_commitment"),
        "action_sequence": policy.get("action_sequence"),
        "posthoc_simulator_masks_used_for_evaluation_only": True,
    }
    return summary, raw_rows


def grouped_summary(samples: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in samples:
        groups[str(row[key])].append(row)
    result = {}
    for name, rows in sorted(groups.items()):
        visible = [row for row in rows if row["target_visible_pixel_count"] > 0]
        result[name] = {
            "sample_count": len(rows),
            "target_visible_sample_count": len(visible),
            "mean_raw_detection_count": float(
                np.mean([row["raw_detection_count"] for row in rows])
            ),
            "mean_unique_object_cluster_count": float(
                np.mean(
                    [
                        row["unique_object_spatial_cluster_count_at_bbox_iou_0_8"]
                        for row in rows
                    ]
                )
            ),
            "mean_qwen_candidate_count": float(
                np.mean([row["qwen_candidate_count"] for row in rows])
            ),
            "target_proposal_recall_at_mask_iou_0_5": (
                float(
                    np.mean(
                        [
                            row["target_proposal_recalled_at_mask_iou_0_5"]
                            for row in visible
                        ]
                    )
                )
                if visible
                else None
            ),
            "selected_target_accuracy_at_mask_iou_0_5": (
                float(
                    np.mean(
                        [
                            row["selected_target_correct_at_mask_iou_0_5"]
                            for row in visible
                        ]
                    )
                )
                if visible
                else None
            ),
        }
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--perception-root", type=Path, default=DEFAULT_PERCEPTION_ROOT)
    parser.add_argument("--policy-result", type=Path, default=DEFAULT_POLICY_RESULT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    detection_paths = sorted(args.perception_root.rglob("detections.json"))
    if not detection_paths:
        raise FileNotFoundError(f"No detections.json below {args.perception_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    policies = policy_index(args.policy_result)
    samples: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for detection_path in detection_paths:
        sample, proposals = audit_sample(
            detection_path, policies, args.output_root / "all_candidate_overlays"
        )
        samples.append(sample)
        raw_rows.extend(proposals)

    visible = [row for row in samples if row["target_visible_pixel_count"] > 0]
    report = {
        "schema_version": "professor-feedback-grounding-candidate-audit-v1",
        "status": "completed_posthoc_development_audit",
        "perception_root": str(args.perception_root.resolve()),
        "sample_count": len(samples),
        "raw_proposal_count": len(raw_rows),
        "target_visible_sample_count": len(visible),
        "overall": {
            "mean_raw_detection_count": float(
                np.mean([row["raw_detection_count"] for row in samples])
            ),
            "mean_unique_object_cluster_count": float(
                np.mean(
                    [
                        row["unique_object_spatial_cluster_count_at_bbox_iou_0_8"]
                        for row in samples
                    ]
                )
            ),
            "mean_qwen_candidate_count": float(
                np.mean([row["qwen_candidate_count"] for row in samples])
            ),
            "target_proposal_recall_at_mask_iou_0_5": float(
                np.mean(
                    [row["target_proposal_recalled_at_mask_iou_0_5"] for row in visible]
                )
            ),
            "selected_target_accuracy_at_mask_iou_0_5": float(
                np.mean(
                    [row["selected_target_correct_at_mask_iou_0_5"] for row in visible]
                )
            ),
        },
        "by_family": grouped_summary(samples, "family"),
        "by_view": grouped_summary(samples, "view_id"),
        "samples": samples,
        "interpretation_limits": [
            "Simulator target masks are used only after inference for evaluation.",
            "Mask IoU 0.5 is an audit threshold, not a calibrated probability.",
            "This calibration/development audit is not a final-test result.",
        ],
    }
    (args.output_root / "audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(args.output_root / "samples.csv", samples)
    write_csv(args.output_root / "all_raw_proposals.csv", raw_rows)
    print(json.dumps({key: report[key] for key in ("status", "sample_count", "raw_proposal_count", "overall", "by_family")}, indent=2))


if __name__ == "__main__":
    main()
