#!/usr/bin/env python3
"""Score direct next-action choices with Qwen and no belief-space planner."""

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
DEFAULT_CONFIG = ROOT / "configs/research/icra_v20_direct_vlm_action_baseline_development.json"
PROMPT_VERSION = "icra-v20-direct-vlm-next-action-v1"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def action_values(model_input: dict[str, Any]) -> list[str]:
    values = ["viewpoint_right", "viewpoint_close_high", "remove_cover"]
    for candidate in model_input["candidates"]:
        candidate_id = candidate["candidate_id"]
        values.extend(
            [
                f"grasp_{candidate_id}_inside",
                f"grasp_{candidate_id}_outside",
            ]
        )
    values.append("defer")
    return values


def action_question(
    model_input: dict[str, Any], mapping: list[tuple[str, str]]
) -> tuple[str, list[tuple[str, str]]]:
    choices = "\n".join(f"{letter}: {value}" for letter, value in mapping)
    question = (
        f"\nRobot instruction: {model_input['instruction']}\n"
        "Choose the single safest NEXT robot action from the supplied choices. "
        "Use only this current RGB observation and anonymous candidate evidence. "
        "A viewpoint action obtains another wrist-camera observation. remove_cover "
        "interacts with a visible cover. A grasp choice commits to both one candidate "
        "and its inside/outside relation. Choose grasp only when both identity and "
        "relation are visually reliable. Choose defer when the target appears absent "
        "or safe commitment is not justified and no listed information action is useful.\n"
        f"{choices}\n"
        "Answer with exactly one uppercase choice letter and no other text."
    )
    return question, mapping


def fingerprint(path: Path, revision: str, max_pixels: int) -> str:
    digest = hashlib.sha256(path.read_bytes())
    digest.update(PROMPT_VERSION.encode())
    digest.update(revision.encode())
    digest.update(str(max_pixels).encode())
    return digest.hexdigest()


def discover_inputs(root: Path) -> list[Path]:
    paths = root.glob("shards/gpu*/grounded_sam2_qwen_inputs/seed*/input.json")
    return sorted(paths, key=lambda path: load_json(path)["sample_id"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--limit", type=int)
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
    output_root = resolve_path(config["output_root"])
    paths = discover_inputs(resolve_path(config["source_perception_root"]))
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
        if result_path.is_file():
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
        values = action_values(model_input)
        mapping = letter_mapping(values)
        visual_content = build_visual_content(model_input, path)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats("cuda:0")
        torch.cuda.synchronize("cuda:0")
        started = time.perf_counter()
        logits = permutation_debiased_scores(
            model,
            processor,
            visual_content,
            mapping,
            lambda current_mapping: action_question(model_input, current_mapping),
            "cuda:0",
        )
        torch.cuda.synchronize("cuda:0")
        ranked = sorted(
            zip(values, logits), key=lambda pair: (-float(pair[1]), pair[0])
        )
        result = {
            "schema_version": "icra-v20-direct-vlm-action-result-v1",
            "sample_id": model_input["sample_id"],
            "model": {"repository": MODEL_REPOSITORY, "revision": revision},
            "prompt_version": PROMPT_VERSION,
            "action_ids": values,
            "raw_action_logits": [float(value) for value in logits],
            "ranked_actions": [
                {"action": action, "raw_logit": float(logit)}
                for action, logit in ranked
            ],
            "selected_action": ranked[0][0],
            "input_path": str(path.resolve()),
            "input_fingerprint_sha256": expected,
            "metrics": {
                "runtime_seconds": time.perf_counter() - started,
                "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated("cuda:0") / 1024**3, 4),
            },
            "physical_gpu": configured_physical_gpu(),
            "logical_gpu": "cuda:0",
            "batch_size": 1,
            "training_performed": False,
            "simulator_ground_truth_used_for_inference": False,
            "scene_graph_used": False,
            "belief_update_used": False,
            "mpc_used": False,
            "valid_for_final_evaluation": False,
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        fresh.append(result)

    results = cached + fresh
    summary = {
        "schema_version": "icra-v20-direct-vlm-action-shard-summary-v1",
        "status": "completed",
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "sample_count": len(results),
        "fresh_count": len(fresh),
        "cache_hit_count": len(cached),
        "model_load_seconds": load_seconds,
        "total_inference_seconds": sum(row["metrics"]["runtime_seconds"] for row in results),
        "mean_inference_seconds": sum(row["metrics"]["runtime_seconds"] for row in results) / len(results),
        "peak_gpu_memory_gib": max(row["metrics"]["peak_gpu_memory_gib"] for row in results),
        "physical_gpu": configured_physical_gpu(),
        "training_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
    }
    summary_path = output_root / "shards" / f"shard{args.shard_index}" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
