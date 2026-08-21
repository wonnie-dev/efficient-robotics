"""Rank Grounded-SAM2 anonymous red proposals with one Qwen model instance.

Qwen receives proposal IDs, RGB crops, and reference overlays, but never the
simulator entity matched to a proposal. Visual identity and spatial relations
are scored independently so one factor cannot silently stand in for the other.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

from qwen3_vl_logits import (
    DEFAULT_MODEL,
    MODEL_REPOSITORY,
    build_visual_content,
    candidate_identity_question,
    letter_mapping,
    local_hf_revision,
    permutation_debiased_scores,
    relation_question,
    require_single_gpu_only,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/perception/grounded_segmentation.json"
RANKING_PROMPT_VERSION = (
    "grounded-qwen-factorized-identity-relation-v2-conservative-unknown"
)


def resolve_path(value: str | Path) -> Path:
    """Interpret relative artifact paths from the repository root."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    """Return the digest used to bind a ranking result to its input."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_result_matches_input(result_path: Path, input_path: Path) -> bool:
    """Check input bytes and prompt version before accepting a cache hit."""
    if not result_path.is_file():
        return False
    try:
        cached = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        cached.get("input_path") == str(input_path)
        and cached.get("input_sha256") == sha256(input_path)
        and cached.get("prompt_version") == RANKING_PROMPT_VERSION
    )


def main() -> None:
    """Rank anonymous candidates and score their factorized relations."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    require_single_gpu_only()

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one GPU must be visible.")
    torch.cuda.set_device("cuda:0")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    perception_root = resolve_path(config["output_root"])
    input_root = perception_root / "grounded_sam2_qwen_inputs"
    manifest = json.loads(
        (input_root / "manifest.json").read_text(encoding="utf-8")
    )

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        local_files_only=True,
        min_pixels=4 * 28 * 28,
        max_pixels=args.max_pixels,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    model.eval()
    load_seconds = time.perf_counter() - load_started
    revision = local_hf_revision(args.model_path)

    sample_metrics = []
    results = []
    for item in manifest["samples"]:
        input_path = resolve_path(item["input_path"])
        destination = perception_root / "grounded_sam2_qwen_rankings" / item["sample_id"]
        result_path = destination / "result.json"
        if not args.force and cached_result_matches_input(result_path, input_path):
            # Cache hits retain the runtime and memory metrics from their original run.
            result = json.loads(result_path.read_text(encoding="utf-8"))
            results.append(result)
            sample_metrics.append(
                {"sample_id": item["sample_id"], **result["metrics"]}
            )
            continue

        model_input = json.loads(input_path.read_text(encoding="utf-8"))
        visual_content = build_visual_content(model_input, input_path)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats("cuda:0")
        torch.cuda.synchronize("cuda:0")
        started = time.perf_counter()
        candidate_logits = []
        for candidate in model_input["candidates"]:
            mapping = letter_mapping(
                [
                    "matches_target_description",
                    "does_not_match_target_description",
                ]
            )
            scores = permutation_debiased_scores(
                model,
                processor,
                visual_content,
                mapping,
                lambda current_mapping, candidate_id=candidate[
                    "candidate_id"
                ]: candidate_identity_question(
                    model_input, candidate_id, current_mapping
                ),
                "cuda:0",
            )
            # This margin measures appearance only; relation evidence is stored below.
            candidate_logits.append(float(scores[0] - scores[1]))
        relation_results = []
        for query in model_input.get("relation_queries", []):
            relation_mapping = letter_mapping(list(query["label_space"]))
            relation_logits = permutation_debiased_scores(
                model,
                processor,
                visual_content,
                relation_mapping,
                lambda current_mapping, query=query: relation_question(
                    query, current_mapping
                ),
                "cuda:0",
            )
            labels = [
                value for _letter, value in relation_mapping
            ]
            top_index = max(
                range(len(relation_logits)),
                key=relation_logits.__getitem__,
            )
            relation_results.append(
                {
                    "query_id": query["query_id"],
                    "source_id": query["source_id"],
                    "target_id": query["target_id"],
                    "relation_type": query.get(
                        "relation_type",
                        "legacy_mutually_exclusive_relation",
                    ),
                    "labels": labels,
                    "raw_logits": relation_logits,
                    "top_label": labels[top_index],
                }
            )
        torch.cuda.synchronize("cuda:0")
        metrics = {
            "runtime_seconds": time.perf_counter() - started,
            "peak_gpu_memory_bytes": int(
                torch.cuda.max_memory_allocated("cuda:0")
            ),
            "peak_gpu_memory_gib": round(
                torch.cuda.max_memory_allocated("cuda:0") / (1024**3), 4
            ),
        }
        selected_index = max(
            range(len(candidate_logits)), key=candidate_logits.__getitem__
        )
        # Candidate order provides deterministic tie-breaking for equal margins.
        selected_candidate = model_input["candidates"][selected_index][
            "candidate_id"
        ]
        # The singular view takes the first query; the plural view keeps all factors.
        result = {
            "schema_version": "grounded-proposal-qwen-ranking-v1",
            "sample_id": item["sample_id"],
            "model": {
                "repository": MODEL_REPOSITORY,
                "revision": revision,
            },
            "prompt_version": RANKING_PROMPT_VERSION,
            "candidate_ids": [
                candidate["candidate_id"]
                for candidate in model_input["candidates"]
            ],
            "raw_match_logits": candidate_logits,
            "target_score_definition": (
                "factorized_visual_identity_match_minus_nonmatch; "
                "spatial_relation_scored_separately"
            ),
            "selected_candidate_id": selected_candidate,
            "relations": relation_results,
            "selected_candidate_relation": next(
                (
                    relation
                    for relation in relation_results
                    if relation["source_id"] == selected_candidate
                ),
                None,
            ),
            "selected_candidate_relations": [
                relation
                for relation in relation_results
                if relation["source_id"] == selected_candidate
            ],
            "input_path": str(input_path),
            "input_sha256": sha256(input_path),
            "metrics": metrics,
            "training_performed": False,
            "simulator_ground_truth_used_for_inference": False,
            "candidate_semantic_labels_exposed_to_qwen": False,
            "valid_for_final_evaluation": False,
        }
        destination.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        results.append(result)
        sample_metrics.append({"sample_id": item["sample_id"], **metrics})

    summary = {
        "schema_version": "grounded-proposal-qwen-ranking-summary-v1",
        "sample_count": len(results),
        "model_load_seconds": load_seconds,
        "total_inference_seconds": sum(
            item["runtime_seconds"] for item in sample_metrics
        ),
        "mean_runtime_seconds": sum(
            item["runtime_seconds"] for item in sample_metrics
        )
        / len(sample_metrics),
        "peak_gpu_memory_gib": max(
            item["peak_gpu_memory_gib"] for item in sample_metrics
        ),
        "sample_metrics": sample_metrics,
        "physical_gpu": int(os.environ.get("PHYSICAL_GPU", "5")),
        "logical_gpu": "cuda:0",
        "single_model_instance": True,
        "batch_size": 1,
        "training_performed": False,
        "valid_for_final_evaluation": False,
    }
    summary_path = perception_root / "grounded_sam2_qwen_ranking_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
