#!/usr/bin/env python3
"""Evaluate cached post-remove joint-choice outputs with simulator masks post hoc."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/research/post_interaction_candidate_choice.json"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"evaluated": 0, "correct": 0, "accuracy": None}
    correct = sum(bool(row["correct"]) for row in rows)
    return {"evaluated": len(rows), "correct": correct, "accuracy": correct / len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    audit_root = resolve_path(config["grounding_audit_root"])
    output_root = resolve_path(config["output_root"])
    audit = load_json(audit_root / "audit.json")
    samples = {row["sample_id"]: row for row in audit["samples"]}
    raw = {}
    with (audit_root / "all_raw_proposals.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw[(row["sample_id"], row["detection_id"])] = row

    rows = []
    result_paths = sorted(output_root.glob("shards/shard*/seed*_post_remove/result.json"))
    for result_path in result_paths:
        result = load_json(result_path)
        sample_id = result["sample_id"]
        audit_sample = samples[sample_id]
        model_input = load_json(Path(result["input_path"]))
        candidate_to_detection = {
            row["candidate_id"]: row["proposal_detection_id"]
            for row in model_input["candidates"]
        }
        visible = int(audit_sample["target_visible_pixel_count"]) > 0
        joint_choice = result["selected_choice"]
        detection_id = candidate_to_detection.get(joint_choice)
        proposal = raw.get((sample_id, detection_id)) if detection_id else None
        overlap_text = proposal.get("target_visible_mask_iou", "") if proposal else ""
        joint_correct = (
            bool(overlap_text and float(overlap_text) >= 0.5)
            if visible
            else joint_choice == "none_of_candidates"
        )
        cached_correct = (
            bool(audit_sample["selected_target_correct_at_mask_iou_0_5"])
            if visible
            else False
        )
        rows.append(
            {
                "sample_id": sample_id,
                "seed": int(audit_sample["seed"]),
                "family": audit_sample["family"],
                "target_visible": visible,
                "target_candidate_available": (
                    any(
                        raw.get((sample_id, row["proposal_detection_id"]), {}).get("target_visible_mask_iou", "")
                        and float(raw[(sample_id, row["proposal_detection_id"])]["target_visible_mask_iou"]) >= 0.5
                        for row in model_input["candidates"]
                    )
                    if visible
                    else False
                ),
                "cached_independent_binary_choice": audit_sample["qwen_selected_candidate_id"],
                "cached_correct": cached_correct,
                "joint_choice": joint_choice,
                "joint_correct": joint_correct,
                "runtime_seconds": result["metrics"]["runtime_seconds"],
                "peak_gpu_memory_gib": result["metrics"]["peak_gpu_memory_gib"],
                "physical_gpu": result["physical_gpu"],
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["family"]].append(row)
    evaluation = {
        "schema_version": "post-interaction-choice-evaluation-v1",
        "status": "completed",
        "sample_count": len(rows),
        "cached_selector": {
            "overall": summarize([{**row, "correct": row["cached_correct"]} for row in rows]),
            "visible": summarize([{**row, "correct": row["cached_correct"]} for row in rows if row["target_visible"]]),
            "absent": summarize([{**row, "correct": row["cached_correct"]} for row in rows if not row["target_visible"]]),
        },
        "joint_choice_selector": {
            "overall": summarize([{**row, "correct": row["joint_correct"]} for row in rows]),
            "visible": summarize([{**row, "correct": row["joint_correct"]} for row in rows if row["target_visible"]]),
            "absent": summarize([{**row, "correct": row["joint_correct"]} for row in rows if not row["target_visible"]]),
            "by_family": {
                family: summarize([{**row, "correct": row["joint_correct"]} for row in values])
                for family, values in sorted(grouped.items())
            },
        },
        "mean_runtime_seconds": sum(row["runtime_seconds"] for row in rows) / len(rows),
        "peak_gpu_memory_gib": max(row["peak_gpu_memory_gib"] for row in rows),
        "simulator_masks_used_for_inference": False,
        "simulator_masks_used_posthoc_for_evaluation": True,
        "training_performed": False,
        "testing_performed": False,
        "valid_for_final_evaluation": False,
        "rows": rows,
    }
    (output_root / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evaluation, indent=2))


if __name__ == "__main__":
    main()
