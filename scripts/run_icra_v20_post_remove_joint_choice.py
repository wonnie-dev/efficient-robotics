#!/usr/bin/env python3
"""Rerank cached post-remove candidates with one joint forced-choice question."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from qwen3_vl_logits import (
    MODEL_REPOSITORY,
    build_visual_content,
    configured_physical_gpu,
    letter_mapping,
    local_hf_revision,
    permutation_debiased_scores,
    require_single_gpu_only,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/research/icra_v20_post_remove_joint_choice_development.json"
PROMPT_VERSION = "icra-v20-joint-candidate-choice-v1"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fingerprint(path: Path, model_revision: str, max_pixels: int) -> str:
    digest = hashlib.sha256(path.read_bytes())
    digest.update(PROMPT_VERSION.encode())
    digest.update(model_revision.encode())
    digest.update(str(max_pixels).encode())
    return digest.hexdigest()


def joint_question(
    model_input: dict[str, Any], mapping: list[tuple[str, str]]
) -> tuple[str, list[tuple[str, str]]]:
    choices = "\n".join(f"{letter}: {value}" for letter, value in mapping)
    question = (
        f"\nTarget appearance description: {model_input['target_description']}\n"
        "Compare all anonymous candidates directly. Select the candidate with the "
        "strongest visible identity evidence for the target. Use color, mug shape, "
        "and especially the white rectangular logo. Ignore inside/outside position "
        "and all spatial relations. Select none_of_candidates when no supplied crop "
        "has adequate visual evidence for the described target.\n"
        f"{choices}\n"
        "Answer with exactly one uppercase choice letter and no other text."
    )
    return question, mapping


def input_paths(source_root: Path, view_id: str) -> list[Path]:
    paths = sorted(
        source_root.glob(
            f"shards/gpu*/grounded_sam2_qwen_inputs/seed*_{view_id}/input.json"
        )
    )
    return sorted(paths, key=lambda path: load_json(path)["sample_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("Invalid shard index/count")
    require_single_gpu_only()

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one CUDA GPU must be visible")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("Distributed inference is forbidden")
    torch.cuda.set_device("cuda:0")

    config = load_json(args.config.resolve())
    source_root = resolve_path(config["source_perception_root"])
    output_root = resolve_path(config["output_root"])
    paths = input_paths(source_root, config["view_id"])
    paths = paths[args.shard_index :: args.shard_count]
    if args.limit is not None:
        paths = paths[: args.limit]
    model_path = resolve_path(config["model_path"])
    revision = local_hf_revision(model_path)
    max_pixels = int(config["max_pixels_per_image"])

    pending = []
    cached = []
    for path in paths:
        model_input = load_json(path)
        result_path = output_root / "shards" / f"shard{args.shard_index}" / model_input["sample_id"] / "result.json"
        expected = fingerprint(path, revision, max_pixels)
        if result_path.is_file() and not args.force:
            result = load_json(result_path)
            if result.get("input_fingerprint_sha256") == expected:
                cached.append(result)
                continue
        pending.append((path, model_input, result_path, expected))

    processor = None
    model = None
    load_seconds = 0.0
    if pending:
        started = time.perf_counter()
        processor = AutoProcessor.from_pretrained(
            model_path,
            local_files_only=True,
            min_pixels=4 * 28 * 28,
            max_pixels=max_pixels,
        )
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to("cuda:0")
        model.eval()
        load_seconds = time.perf_counter() - started

    fresh = []
    for path, model_input, result_path, expected in pending:
        assert processor is not None and model is not None
        candidate_ids = [row["candidate_id"] for row in model_input["candidates"]]
        values = candidate_ids + (["none_of_candidates"] if config["include_none_choice"] else [])
        mapping = letter_mapping(values)
        visual_content = build_visual_content(model_input, path)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats("cuda:0")
        torch.cuda.synchronize("cuda:0")
        started = time.perf_counter()
        scores = permutation_debiased_scores(
            model,
            processor,
            visual_content,
            mapping,
            lambda current_mapping: joint_question(model_input, current_mapping),
            "cuda:0",
        )
        torch.cuda.synchronize("cuda:0")
        runtime = time.perf_counter() - started
        selected_index = max(range(len(scores)), key=scores.__getitem__)
        result = {
            "schema_version": "icra-v20-post-remove-joint-choice-result-v1",
            "sample_id": model_input["sample_id"],
            "model": {"repository": MODEL_REPOSITORY, "revision": revision},
            "prompt_version": PROMPT_VERSION,
            "choice_ids": values,
            "raw_choice_logits": [float(value) for value in scores],
            "selected_choice": values[selected_index],
            "input_path": str(path.resolve()),
            "input_fingerprint_sha256": expected,
            "metrics": {
                "runtime_seconds": runtime,
                "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated("cuda:0") / 1024**3, 4),
            },
            "physical_gpu": configured_physical_gpu(),
            "logical_gpu": "cuda:0",
            "batch_size": 1,
            "training_performed": False,
            "simulator_ground_truth_used_for_inference": False,
            "valid_for_final_evaluation": False
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        fresh.append(result)

    results = cached + fresh
    summary = {
        "schema_version": "icra-v20-post-remove-joint-choice-shard-summary-v1",
        "status": "completed",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "sample_count": len(results),
        "fresh_count": len(fresh),
        "cache_hit_count": len(cached),
        "model_load_seconds": load_seconds,
        "total_inference_seconds": sum(row["metrics"]["runtime_seconds"] for row in results),
        "peak_gpu_memory_gib": max((row["metrics"]["peak_gpu_memory_gib"] for row in results), default=0.0),
        "physical_gpu": configured_physical_gpu(),
        "training_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False
    }
    summary_path = output_root / "shards" / f"shard{args.shard_index}" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
