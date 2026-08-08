#!/usr/bin/env python3
"""Compare cached single-view Qwen evidence with post-action multi-view evidence.

The historical center observation and one already acquired post-action view are
provided together.  The experiment is diagnostic only: a post-action image is
never exposed to root action selection, no reserved test seed is used, and no
simulator ground truth enters model inference.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from qwen3_vl_logits import (
    MODEL_REPOSITORY,
    build_visual_content,
    candidate_identity_question,
    configured_physical_gpu,
    letter_mapping,
    local_hf_revision,
    permutation_debiased_scores,
    relation_question,
    require_single_gpu_only,
    resolve_asset_path,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs/research/qwen_multiview_postaction_ablation_seed165_184.json"
)
PROMPT_VERSION = "qwen3-vl-postaction-multiview-v1-current-relations-only"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--limit",
        type=int,
        help="Run only the first N configured pairs; intended for a preflight.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def prefixed_model_input(model_input: dict[str, Any], prefix: str) -> dict[str, Any]:
    """Return a copy whose anonymous IDs cannot collide across observations."""
    result = copy.deepcopy(model_input)
    candidate_ids = {
        candidate["candidate_id"]: f"{prefix}_{candidate['candidate_id']}"
        for candidate in result["candidates"]
    }
    reference_ids = {
        reference["reference_id"]: f"{prefix}_{reference['reference_id']}"
        for reference in result.get("reference_entities", [])
    }
    for candidate in result["candidates"]:
        candidate["candidate_id"] = candidate_ids[candidate["candidate_id"]]
    for reference in result.get("reference_entities", []):
        reference["reference_id"] = reference_ids[reference["reference_id"]]
    for query in result.get("relation_queries", []):
        original_source = query["source_id"]
        original_target = query["target_id"]
        query["source_id"] = candidate_ids.get(
            original_source, reference_ids.get(original_source, f"{prefix}_{original_source}")
        )
        query["target_id"] = reference_ids.get(
            original_target, candidate_ids.get(original_target, f"{prefix}_{original_target}")
        )
        query["query_id"] = f"{prefix}_{query['query_id']}"
    return result


def build_postaction_multiview_content(
    historical_input: dict[str, Any],
    historical_path: Path,
    current_input: dict[str, Any],
    current_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    historical = prefixed_model_input(historical_input, "historical_center")
    current = prefixed_model_input(current_input, "current")
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "HISTORICAL OBSERVATION: this center view was acquired before the "
                "viewpoint action. Use it only as prior visual evidence about target "
                "identity. Its candidate IDs are local to that historical image."
            ),
        }
    ]
    content.extend(build_visual_content(historical, historical_path))
    content.append(
        {
            "type": "text",
            "text": (
                "CURRENT POST-ACTION OBSERVATION: make the requested decision for "
                "the current_* candidates below. Current candidate IDs are local to "
                "this image and are not guaranteed to match historical IDs. Spatial "
                "relations must describe this current observation only."
            ),
        }
    )
    content.extend(build_visual_content(current, current_path))
    return content, current


def current_relation_question(
    query: dict[str, Any], mapping: list[tuple[str, str]]
) -> tuple[str, list[tuple[str, str]]]:
    question, returned_mapping = relation_question(query, mapping)
    prefix = (
        "\nUse the historical center image only for target identity continuity. "
        "Judge this relation exclusively from the CURRENT POST-ACTION observation. "
        "Do not copy the historical spatial relation into the current answer.\n"
    )
    return prefix + question, returned_mapping


def iter_asset_paths(model_input: dict[str, Any], input_path: Path) -> list[Path]:
    values = [model_input["image"]["rgb_path"]]
    for candidate in model_input["candidates"]:
        values.extend(
            candidate[key]
            for key in ("crop_path", "mask_path", "context_path")
            if candidate.get(key)
        )
    for reference in model_input.get("reference_entities", []):
        values.extend(
            reference[key]
            for key in ("mask_path", "overlay_path")
            if reference.get(key)
        )
    return [resolve_asset_path(value, input_path) for value in values]


def pair_fingerprint(
    historical_input: dict[str, Any],
    historical_path: Path,
    current_input: dict[str, Any],
    current_path: Path,
    model_revision: str,
    max_pixels: int,
    relation_factors: list[str],
) -> str:
    digest = hashlib.sha256()
    contract = {
        "prompt_version": PROMPT_VERSION,
        "model_revision": model_revision,
        "max_pixels": max_pixels,
        "relation_factors": relation_factors,
    }
    digest.update(json.dumps(contract, sort_keys=True).encode("utf-8"))
    for input_file, model_input in (
        (historical_path, historical_input),
        (current_path, current_input),
    ):
        digest.update(input_file.read_bytes())
        for asset in sorted(iter_asset_paths(model_input, input_file)):
            digest.update(str(asset).encode("utf-8"))
            digest.update(asset.read_bytes())
    return digest.hexdigest()


def configured_pairs(config: dict[str, Any]) -> list[dict[str, Any]]:
    historical = config["historical_view"]
    return [
        {"seed": seed, "historical_view": historical, "current_view": current}
        for seed in config["development_seeds"]
        for current in config["post_action_views"]
    ]


def top_label(relation: dict[str, Any]) -> str:
    index = max(range(len(relation["raw_logits"])), key=relation["raw_logits"].__getitem__)
    return relation["labels"][index]


def find_relation(
    relations: list[dict[str, Any]], candidate_id: str, relation_type: str
) -> dict[str, Any] | None:
    return next(
        (
            relation
            for relation in relations
            if relation["candidate_id"] == candidate_id
            and relation["relation_type"] == relation_type
        ),
        None,
    )


def find_cached_relation(
    relations: list[dict[str, Any]], candidate_id: str, relation_type: str
) -> dict[str, Any] | None:
    return next(
        (
            relation
            for relation in relations
            if relation["source_id"] == candidate_id
            and relation["relation_type"] == relation_type
        ),
        None,
    )


def evaluate_results(
    results: list[dict[str, Any]],
    records_path: Path,
    single_view_root: Path,
) -> dict[str, Any]:
    records = json.loads(records_path.read_text(encoding="utf-8"))["records"]
    record_index = {(item["seed"], item["view"]): item for item in records}
    rows = []
    for result in results:
        seed = result["seed"]
        view = result["current_view"]
        record = record_index[(seed, view)]
        candidate_gt = {item["candidate_id"]: item for item in record["candidates"]}
        target_ids = [
            candidate_id
            for candidate_id, candidate in candidate_gt.items()
            if candidate["target_label"]
        ]
        cached = json.loads(
            (single_view_root / f"seed{seed}_{view}" / "result.json").read_text(
                encoding="utf-8"
            )
        )
        target_present = bool(target_ids)
        target_id = target_ids[0] if target_present else None
        multi_selected = result["selected_candidate_id"]
        single_selected = cached["selected_candidate_id"]
        row: dict[str, Any] = {
            "seed": seed,
            "scene_variant": record["calibration_scene_variant"],
            "historical_view": result["historical_view"],
            "current_view": view,
            "target_proposal_present": target_present,
            "target_candidate_id": target_id,
            "single_selected_candidate_id": single_selected,
            "multi_selected_candidate_id": multi_selected,
            "single_target_selection_correct": (
                single_selected == target_id if target_present else None
            ),
            "multi_target_selection_correct": (
                multi_selected == target_id if target_present else None
            ),
        }
        for factor in ("membership", "occluded_by"):
            truth = (
                candidate_gt[target_id]["relation_ground_truth"][factor]
                if target_present
                else None
            )
            multi_relation = (
                find_relation(result["relations"], target_id, factor)
                if target_present
                else None
            )
            single_relation = (
                find_cached_relation(cached["relations"], target_id, factor)
                if target_present
                else None
            )
            multi_prediction = top_label(multi_relation) if multi_relation else None
            single_prediction = single_relation["top_label"] if single_relation else None
            row[f"{factor}_ground_truth"] = truth
            row[f"single_{factor}_prediction"] = single_prediction
            row[f"multi_{factor}_prediction"] = multi_prediction
            row[f"single_{factor}_correct"] = (
                single_prediction == truth if truth is not None else None
            )
            row[f"multi_{factor}_correct"] = (
                multi_prediction == truth if truth is not None else None
            )
        rows.append(row)

    eligible = [row for row in rows if row["target_proposal_present"]]

    def accuracy(field: str) -> dict[str, Any]:
        values = [row[field] for row in eligible if row[field] is not None]
        correct = sum(bool(value) for value in values)
        return {
            "correct": correct,
            "evaluated": len(values),
            "accuracy": correct / len(values) if values else None,
        }

    return {
        "schema_version": "qwen-multiview-postaction-evaluation-v1",
        "pair_count": len(rows),
        "target_visible_pair_count": len(eligible),
        "target_proposal_missing_pair_count": len(rows) - len(eligible),
        "metrics": {
            "single_target_selection": accuracy("single_target_selection_correct"),
            "multi_target_selection": accuracy("multi_target_selection_correct"),
            "single_membership": accuracy("single_membership_correct"),
            "multi_membership": accuracy("multi_membership_correct"),
            "single_occluded_by": accuracy("single_occluded_by_correct"),
            "multi_occluded_by": accuracy("multi_occluded_by_correct"),
        },
        "rows": rows,
        "ground_truth_used_during_inference": False,
        "ground_truth_used_posthoc_for_evaluation": True,
        "valid_for_final_evaluation": False,
    }


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["reserved_test_seeds_used"]:
        raise RuntimeError("Reserved test seeds are forbidden in this diagnostic.")
    require_single_gpu_only()

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA GPU must be visible.")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("Distributed execution is forbidden.")
    torch.cuda.set_device("cuda:0")

    input_root = resolve_path(config["input_root"])
    output_root = resolve_path(config["output_root"])
    result_root = output_root / "results"
    result_root.mkdir(parents=True, exist_ok=True)
    model_path = resolve_path(config["model_path"])
    revision = local_hf_revision(model_path)
    relation_factors = list(config["relation_factors"])
    pairs = configured_pairs(config)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        pairs = pairs[: args.limit]

    pending = []
    cached_results = []
    for pair in pairs:
        seed = pair["seed"]
        historical_path = input_root / f"seed{seed}_{pair['historical_view']}" / "input.json"
        current_path = input_root / f"seed{seed}_{pair['current_view']}" / "input.json"
        historical_input = json.loads(historical_path.read_text(encoding="utf-8"))
        current_input = json.loads(current_path.read_text(encoding="utf-8"))
        fingerprint = pair_fingerprint(
            historical_input,
            historical_path,
            current_input,
            current_path,
            revision,
            config["max_pixels_per_image"],
            relation_factors,
        )
        result_path = result_root / f"seed{seed}_center_plus_{pair['current_view']}" / "result.json"
        if result_path.is_file() and not args.force:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("input_fingerprint_sha256") == fingerprint:
                cached_results.append(existing)
                continue
        pending.append(
            {
                **pair,
                "historical_path": historical_path,
                "current_path": current_path,
                "historical_input": historical_input,
                "current_input": current_input,
                "fingerprint": fingerprint,
                "result_path": result_path,
            }
        )

    model_load_seconds = 0.0
    processor = None
    model = None
    if pending:
        load_started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            min_pixels=4 * 28 * 28,
            max_pixels=config["max_pixels_per_image"],
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to("cuda:0")
        model.eval()
        model_load_seconds = time.perf_counter() - load_started

    fresh_results = []
    for item in pending:
        assert processor is not None and model is not None
        visual_content, current = build_postaction_multiview_content(
            item["historical_input"],
            item["historical_path"],
            item["current_input"],
            item["current_path"],
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats("cuda:0")
        torch.cuda.synchronize("cuda:0")
        started = time.perf_counter()
        candidate_rows = []
        for candidate in current["candidates"]:
            mapping = letter_mapping(
                ["matches_target_description", "does_not_match_target_description"]
            )
            scores = permutation_debiased_scores(
                model,
                processor,
                visual_content,
                mapping,
                lambda current_mapping, candidate_id=candidate[
                    "candidate_id"
                ]: candidate_identity_question(current, candidate_id, current_mapping),
                "cuda:0",
            )
            candidate_rows.append(
                {
                    "candidate_id": candidate["candidate_id"].removeprefix("current_"),
                    "model_candidate_id": candidate["candidate_id"],
                    "raw_match_logit": float(scores[0] - scores[1]),
                }
            )
        relations = []
        for query in current.get("relation_queries", []):
            factor = query.get("relation_type")
            if factor not in relation_factors:
                continue
            mapping = letter_mapping(list(query["label_space"]))
            scores = permutation_debiased_scores(
                model,
                processor,
                visual_content,
                mapping,
                lambda current_mapping, query=query: current_relation_question(
                    query, current_mapping
                ),
                "cuda:0",
            )
            relations.append(
                {
                    "query_id": query["query_id"].removeprefix("current_"),
                    "candidate_id": query["source_id"].removeprefix("current_"),
                    "model_candidate_id": query["source_id"],
                    "relation_type": factor,
                    "labels": [value for _letter, value in mapping],
                    "raw_logits": [float(score) for score in scores],
                }
            )
        torch.cuda.synchronize("cuda:0")
        runtime_seconds = time.perf_counter() - started
        selected = max(candidate_rows, key=lambda row: row["raw_match_logit"])[
            "candidate_id"
        ]
        result = {
            "schema_version": "qwen-multiview-postaction-result-v1",
            "experiment_id": config["experiment_id"],
            "seed": item["seed"],
            "historical_view": item["historical_view"],
            "current_view": item["current_view"],
            "model": {"repository": MODEL_REPOSITORY, "revision": revision},
            "prompt_version": PROMPT_VERSION,
            "input_fingerprint_sha256": item["fingerprint"],
            "historical_input_path": str(item["historical_path"]),
            "current_input_path": str(item["current_path"]),
            "candidate_scores": candidate_rows,
            "selected_candidate_id": selected,
            "relations": relations,
            "metrics": {
                "runtime_seconds": runtime_seconds,
                "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated("cuda:0")),
                "peak_gpu_memory_gib": round(
                    torch.cuda.max_memory_allocated("cuda:0") / (1024**3), 4
                ),
            },
            "physical_gpu": configured_physical_gpu(),
            "logical_gpu": "cuda:0",
            "visible_cuda_device_count": torch.cuda.device_count(),
            "batch_size": 1,
            "training_performed": False,
            "simulator_ground_truth_used_for_inference": False,
            "post_action_image_used_for_root_action_selection": False,
            "valid_for_final_evaluation": False,
        }
        item["result_path"].parent.mkdir(parents=True, exist_ok=True)
        item["result_path"].write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        fresh_results.append(result)

    all_results = sorted(
        cached_results + fresh_results,
        key=lambda row: (row["seed"], row["current_view"]),
    )
    evaluation = evaluate_results(
        all_results,
        resolve_path(config["calibration_records"]),
        resolve_path(config["single_view_result_root"]),
    )
    (output_root / "evaluation.json").write_text(
        json.dumps(evaluation, indent=2) + "\n", encoding="utf-8"
    )
    sample_metrics = [row["metrics"] for row in all_results]
    summary = {
        "schema_version": "qwen-multiview-postaction-summary-v1",
        "experiment_id": config["experiment_id"],
        "configured_pair_count": len(configured_pairs(config)),
        "completed_pair_count": len(all_results),
        "fresh_pair_count": len(fresh_results),
        "cache_hit_count": len(cached_results),
        "model_load_seconds_this_invocation": model_load_seconds,
        "total_inference_seconds": sum(row["runtime_seconds"] for row in sample_metrics),
        "mean_runtime_seconds_per_pair": (
            sum(row["runtime_seconds"] for row in sample_metrics) / len(sample_metrics)
            if sample_metrics
            else 0.0
        ),
        "peak_gpu_memory_gib": max(
            (row["peak_gpu_memory_gib"] for row in sample_metrics), default=0.0
        ),
        "physical_gpu": 5,
        "logical_gpu": "cuda:0",
        "single_model_instance": True,
        "batch_size": 1,
        "training_performed": False,
        "isaac_sim_launched": False,
        "vulkan_used": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
        "evaluation_metrics": evaluation["metrics"],
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
