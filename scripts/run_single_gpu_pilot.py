"""Run a cache-first, inference-only VLM pilot episode.

This runner is intentionally limited to sequential, batch-size-one inference.
It converts uncalibrated Qwen raw logits into pilot-only beliefs, selects one
new viewpoint, fuses the new observation, and replans.  Ground truth is read
only after planning for debugging metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

from qwen3_vl_logits import (
    DEFAULT_MODEL,
    PROMPT_VERSION,
    local_hf_revision,
    resolve_asset_path,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "outputs" / "vlm_dataset" / "manifest.json"
DEFAULT_CACHE_ROOT = ROOT / "outputs" / "pilot_cache" / "qwen3_vl"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "single_gpu_pilot"
VIEW_ORDER = ("center", "close_high", "right", "left")
REQUIRED_RELATION = "inside"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--episode-id", default="benchmark_seed000")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    parser.add_argument(
        "--allow-cache-miss-inference",
        action="store_true",
        help="Run one sequential Qwen subprocess when a cache entry is absent.",
    )
    return parser.parse_args()


def configured_physical_gpu() -> int:
    value = os.environ.get("PHYSICAL_GPU")
    if value is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        value = visible if visible and "," not in visible else "0"
    if not value.isdigit():
        raise RuntimeError(f"PHYSICAL_GPU must be one integer index, got {value!r}")
    return int(value)


def require_single_gpu_policy() -> None:
    expected = str(configured_physical_gpu())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and ("," in visible or not visible.isdigit()):
        raise RuntimeError("Exactly one integer CUDA device index may be visible")
    if os.environ.get("PHYSICAL_GPU") is not None and visible != expected:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES must be exactly {expected} for this run"
        )
    forbidden = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR")
    if any(os.environ.get(name) for name in forbidden):
        raise RuntimeError("Distributed execution variables are forbidden")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_assets(model_input: dict, input_path: Path) -> list[Path]:
    assets = [
        resolve_asset_path(model_input["image"]["rgb_path"], input_path)
    ]
    for candidate in model_input["candidates"]:
        assets.append(resolve_asset_path(candidate["crop_path"], input_path))
        assets.append(resolve_asset_path(candidate["mask_path"], input_path))
        if candidate.get("context_path"):
            assets.append(
                resolve_asset_path(candidate["context_path"], input_path)
            )
    for reference in model_input.get("reference_entities", []):
        for key in ("mask_path", "overlay_path"):
            if reference.get(key):
                assets.append(resolve_asset_path(reference[key], input_path))
    return assets


def cache_request(
    input_path: Path,
    model_path: Path,
    max_pixels: int,
) -> tuple[str, dict]:
    model_input = load_json(input_path)
    assets = input_assets(model_input, input_path)
    asset_hashes = [sha256_file(path) for path in assets]

    def normalize_for_inference_identity(value, key=None):
        if isinstance(value, dict):
            normalized = {}
            for child_key, child_value in value.items():
                if child_key in {"sample_id", "episode_id"}:
                    continue
                normalized[child_key] = normalize_for_inference_identity(
                    child_value, child_key
                )
            return normalized
        if isinstance(value, list):
            return [
                normalize_for_inference_identity(item)
                for item in value
            ]
        if key and key.endswith("_path") and isinstance(value, str):
            path = resolve_asset_path(value, input_path)
            return {"content_sha256": sha256_file(path)}
        return value

    inference_payload = normalize_for_inference_identity(model_input)
    payload_canonical = json.dumps(
        inference_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    request = {
        "schema_version": "qwen3-vl-cache-request-v2",
        "sample_id": model_input["sample_id"],
        "input_sha256": sha256_file(input_path),
        "inference_payload_sha256": hashlib.sha256(
            payload_canonical
        ).hexdigest(),
        "assets": [
            {
                "path": str(path),
                "sha256": asset_hash,
            }
            for path, asset_hash in zip(assets, asset_hashes)
        ],
        "model_revision": local_hf_revision(model_path),
        "prompt_version": PROMPT_VERSION,
        "max_pixels_per_image": max_pixels,
        "inference_policy": {
            "training": False,
            "batch_size": 1,
            "distributed": False,
            "physical_gpu": configured_physical_gpu(),
        },
    }
    cache_identity = {
        "schema_version": "qwen3-vl-cache-identity-v2",
        "inference_payload_sha256": request["inference_payload_sha256"],
        "asset_sha256_ordered": asset_hashes,
        "model_revision": request["model_revision"],
        "prompt_version": request["prompt_version"],
        "max_pixels_per_image": request["max_pixels_per_image"],
        "inference_policy": request["inference_policy"],
    }
    canonical = json.dumps(
        cache_identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), request


def valid_output(
    output: dict, request: dict, *, require_sample_id: bool = True
) -> bool:
    return (
        output.get("schema_version") == "vlm-output-v1"
        and (
            not require_sample_id
            or output.get("sample_id") == request["sample_id"]
        )
        and output.get("provenance", {}).get("prompt_version")
        == request["prompt_version"]
        and request["model_revision"]
        in output.get("model", {}).get("checkpoint", "")
    )


def register_existing_result(
    input_path: Path,
    cache_dir: Path,
    request: dict,
) -> bool:
    source_output = input_path.with_name("qwen3_vl_output.json")
    source_metrics = input_path.with_name("qwen3_vl_metrics.json")
    if not source_output.is_file() or not source_metrics.is_file():
        return False
    output = load_json(source_output)
    if not valid_output(output, request):
        return False
    cache_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_output, cache_dir / "output.json")
    shutil.copy2(source_metrics, cache_dir / "metrics.json")
    return True


def run_cache_miss(
    input_path: Path,
    cache_dir: Path,
    model_path: Path,
    max_pixels: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(ROOT / "scripts" / "run_qwen3_vl_single_gpu.sh"),
        str(input_path),
        "--output",
        str(cache_dir / "output.json"),
        "--metrics-output",
        str(cache_dir / "metrics.json"),
        "--model-path",
        str(model_path),
        "--max-pixels",
        str(max_pixels),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )
    (cache_dir / "inference_stdout.log").write_text(
        completed.stdout, encoding="utf-8"
    )
    (cache_dir / "inference_stderr.log").write_text(
        completed.stderr, encoding="utf-8"
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Qwen inference failed for {input_path}: "
            f"{completed.stderr.strip()}"
        )


def cached_inference(
    input_path: Path,
    cache_root: Path,
    model_path: Path,
    max_pixels: int,
    allow_cache_miss: bool,
) -> dict:
    cache_key, request = cache_request(input_path, model_path, max_pixels)
    cache_dir = cache_root / cache_key
    output_path = cache_dir / "output.json"
    metrics_path = cache_dir / "metrics.json"
    cache_hit = output_path.is_file() and metrics_path.is_file()
    cache_source = "cache"
    if not cache_hit:
        cache_hit = register_existing_result(input_path, cache_dir, request)
        cache_source = "registered_existing_sample_output"
    if not cache_hit:
        if not allow_cache_miss:
            raise FileNotFoundError(
                f"Cache miss for {input_path}; rerun with "
                "--allow-cache-miss-inference"
            )
        run_cache_miss(input_path, cache_dir, model_path, max_pixels)
        cache_source = "fresh_sequential_inference"
    output = load_json(output_path)
    if not valid_output(output, request, require_sample_id=False):
        raise RuntimeError(f"Invalid cached VLM output: {output_path}")
    reused_for_new_sample = output.get("sample_id") != request["sample_id"]
    if reused_for_new_sample:
        output = json.loads(json.dumps(output))
        output["sample_id"] = request["sample_id"]
        if cache_source == "cache":
            cache_source = "content_cache_reused_for_new_sample_id"
    metrics = load_json(metrics_path)
    if reused_for_new_sample:
        metrics = dict(metrics)
        metrics["sample_id"] = request["sample_id"]
        metrics["source_sample_id"] = load_json(output_path).get("sample_id")
        metrics["content_cache_reused"] = True
    request_path = cache_dir / "request.json"
    request_path.write_text(
        json.dumps(request, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "cache_key": cache_key,
        "cache_dir": str(cache_dir),
        "cache_hit": cache_source != "fresh_sequential_inference",
        "cache_source": cache_source,
        "output": output,
        "metrics": metrics,
    }


def softmax(values: list[float]) -> list[float]:
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def output_belief(output: dict) -> dict:
    target_ids = output["target"]["candidate_ids"]
    target_probabilities = softmax(output["target"]["raw_logits"])
    target = dict(zip(target_ids, target_probabilities))
    relations = {}
    for relation in output["relations"]:
        relations[relation["query_id"]] = dict(
            zip(relation["labels"], softmax(relation["raw_logits"]))
        )
    return {
        "target": target,
        "relations": relations,
        "calibrated": False,
        "source": "temperature_1_softmax_of_uncalibrated_qwen_raw_logits",
    }


def normalize(distribution: dict[str, float]) -> dict[str, float]:
    total = sum(distribution.values())
    return {key: value / total for key, value in distribution.items()}


def fuse_distributions(
    prior: dict[str, float],
    observation: dict[str, float],
    epsilon: float = 1e-9,
) -> dict[str, float]:
    keys = sorted(set(prior) | set(observation))
    product = {
        key: max(prior.get(key, epsilon), epsilon)
        * max(observation.get(key, epsilon), epsilon)
        for key in keys
    }
    return normalize(product)


def fuse_beliefs(prior: dict, observation: dict) -> dict:
    relation_keys = set(prior["relations"]) | set(observation["relations"])
    relations = {}
    for query_id in sorted(relation_keys):
        if query_id in prior["relations"] and query_id in observation["relations"]:
            relations[query_id] = fuse_distributions(
                prior["relations"][query_id],
                observation["relations"][query_id],
            )
        else:
            relations[query_id] = (
                prior["relations"].get(query_id)
                or observation["relations"][query_id]
            )
    return {
        "target": fuse_distributions(prior["target"], observation["target"]),
        "relations": relations,
        "calibrated": False,
        "source": (
            "pilot_only_product_fusion_of_uncalibrated_view_probabilities"
        ),
    }


def relation_confidence(belief: dict, candidate_id: str) -> float:
    query_id = f"{candidate_id}_to_container"
    return belief["relations"].get(query_id, {}).get(REQUIRED_RELATION, 0.0)


def select_action(
    belief: dict,
    current_view: str,
    available_views: set[str],
    visited_views: set[str],
    target_threshold: float = 0.75,
    relation_threshold: float = 0.65,
) -> dict:
    candidate_id, target_confidence = max(
        belief["target"].items(), key=lambda item: item[1]
    )
    inside_confidence = relation_confidence(belief, candidate_id)
    if (
        target_confidence >= target_threshold
        and inside_confidence >= relation_threshold
    ):
        action = "grasp"
        reason = "pilot_confidence_thresholds_satisfied"
    else:
        next_view = next(
            (
                view
                for view in VIEW_ORDER
                if view != current_view
                and view in available_views
                and view not in visited_views
            ),
            None,
        )
        action = f"viewpoint_{next_view}" if next_view else "defer"
        reason = (
            "uncalibrated_target_or_relation_confidence_below_threshold"
            if next_view
            else "no_unvisited_pilot_view_available"
        )
    return {
        "type": action,
        "reason": reason,
        "selected_candidate": candidate_id,
        "uncalibrated_target_confidence": target_confidence,
        "uncalibrated_inside_confidence": inside_confidence,
        "valid_for_final_evaluation": False,
    }


def episode_samples(manifest_path: Path, episode_id: str) -> dict[str, Path]:
    manifest = load_json(manifest_path)
    samples = {}
    for item in manifest["samples"]:
        input_path = (ROOT / item["input"]).resolve()
        model_input = load_json(input_path)
        if model_input["episode_id"] == episode_id:
            samples[model_input["view_id"]] = input_path
    return samples


def evaluate_debug_only(output: dict, input_path: Path) -> dict:
    ground_truth_path = input_path.with_name("ground_truth.json")
    if not ground_truth_path.is_file():
        return {"available": False}
    ground_truth = load_json(ground_truth_path)
    belief = output_belief(output)
    predicted_target = max(belief["target"], key=belief["target"].get)
    relation_truth = {
        item["query_id"]: item["label"]
        for item in ground_truth["relations"]
    }
    relation_correct = 0
    relation_total = 0
    for query_id, distribution in belief["relations"].items():
        if query_id in relation_truth:
            relation_total += 1
            relation_correct += (
                max(distribution, key=distribution.get)
                == relation_truth[query_id]
            )
    return {
        "available": True,
        "read_after_planning_only": True,
        "target_correct": (
            predicted_target == ground_truth["target_candidate_id"]
        ),
        "relation_correct": relation_correct,
        "relation_total": relation_total,
        "valid_for_final_evaluation": False,
    }


def main() -> None:
    args = parse_args()
    require_single_gpu_policy()
    manifest_path = args.manifest.resolve()
    model_path = args.model_path.resolve()
    samples = episode_samples(manifest_path, args.episode_id)
    if "center" not in samples:
        raise RuntimeError(f"Episode has no center observation: {args.episode_id}")

    started = time.perf_counter()
    initial = cached_inference(
        samples["center"],
        args.cache_root,
        model_path,
        args.max_pixels,
        args.allow_cache_miss_inference,
    )
    initial_belief = output_belief(initial["output"])
    first_action = select_action(
        initial_belief,
        current_view="center",
        available_views=set(samples),
        visited_views={"center"},
    )
    consumed_inferences = [initial]
    selected_view = None
    post_action = None
    posterior = None
    replan = {
        "type": "not_performed",
        "reason": "initial_action_was_terminal",
        "valid_for_final_evaluation": False,
    }
    final_belief = initial_belief
    final_action = first_action
    second_new_observation = None
    if first_action["type"].startswith("viewpoint_"):
        selected_view = first_action["type"].removeprefix("viewpoint_")
        if selected_view not in samples:
            raise RuntimeError(f"Selected view is unavailable: {selected_view}")
        post_action = cached_inference(
            samples[selected_view],
            args.cache_root,
            model_path,
            args.max_pixels,
            args.allow_cache_miss_inference,
        )
        consumed_inferences.append(post_action)
        posterior = fuse_beliefs(
            initial_belief, output_belief(post_action["output"])
        )
        replan = select_action(
            posterior,
            current_view=selected_view,
            available_views=set(samples),
            visited_views={"center", selected_view},
        )
        final_belief = posterior
        final_action = replan
        if replan["type"].startswith("viewpoint_"):
            second_selected_view = replan["type"].removeprefix("viewpoint_")
            if second_selected_view in samples:
                second_post_action = cached_inference(
                    samples[second_selected_view],
                    args.cache_root,
                    model_path,
                    args.max_pixels,
                    args.allow_cache_miss_inference,
                )
                consumed_inferences.append(second_post_action)
                final_belief = fuse_beliefs(
                    posterior, output_belief(second_post_action["output"])
                )
                final_action = select_action(
                    final_belief,
                    current_view=second_selected_view,
                    available_views=set(samples),
                    visited_views={
                        "center",
                        selected_view,
                        second_selected_view,
                    },
                )
                second_new_observation = {
                    "view": second_selected_view,
                    "sample_id": second_post_action["output"]["sample_id"],
                    "cache_key": second_post_action["cache_key"],
                    "cache_dir": second_post_action["cache_dir"],
                    "cache_hit": second_post_action["cache_hit"],
                    "cache_source": second_post_action["cache_source"],
                    "recorded_inference_metrics": second_post_action["metrics"],
                }
    wall_seconds = time.perf_counter() - started

    episode_result = {
        "schema_version": "single-gpu-pilot-episode-v1",
        "status": "completed",
        "episode_id": args.episode_id,
        "purpose": "pipeline_validation_only_not_final_paper_evidence",
        "execution_mode": (
            "deterministic_pre_captured_observation_replay"
        ),
        "training": {
            "performed": False,
            "fine_tuning": False,
            "lora": False,
        },
        "calibration": {
            "performed": False,
            "description": (
                "Held-out confidence adjustment; deliberately not performed "
                "on this development episode."
            ),
        },
        "testing": {
            "performed": False,
            "description": "Final unbiased evaluation remains pending.",
        },
        "gpu_policy": {
            "physical_gpu": configured_physical_gpu(),
            "cuda_device": "cuda:0",
            "batch_size": 1,
            "distributed": False,
            "parallel_vlm_jobs": False,
        },
        "initial_observation": {
            "view": "center",
            "sample_id": initial["output"]["sample_id"],
            "cache_key": initial["cache_key"],
            "cache_dir": initial["cache_dir"],
            "cache_hit": initial["cache_hit"],
            "cache_source": initial["cache_source"],
            "recorded_inference_metrics": initial["metrics"],
        },
        "initial_belief": initial_belief,
        "first_action": first_action,
        "new_observation": (
            {
                "view": selected_view,
                "sample_id": post_action["output"]["sample_id"],
                "cache_key": post_action["cache_key"],
                "cache_dir": post_action["cache_dir"],
                "cache_hit": post_action["cache_hit"],
                "cache_source": post_action["cache_source"],
                "recorded_inference_metrics": post_action["metrics"],
            }
            if post_action is not None
            else None
        ),
        "posterior_belief": posterior,
        "replan": replan,
        "second_new_observation": second_new_observation,
        "final_belief_after_all_consumed_views": final_belief,
        "final_action": final_action,
        "debug_ground_truth_metrics": {
            "initial": evaluate_debug_only(
                initial["output"], samples["center"]
            ),
            "post_action": (
                evaluate_debug_only(
                    post_action["output"], samples[selected_view]
                )
                if post_action is not None
                else {"available": False}
            ),
            "second_post_action": (
                evaluate_debug_only(
                    consumed_inferences[-1]["output"],
                    samples[second_new_observation["view"]],
                )
                if second_new_observation is not None
                else {"available": False}
            ),
            "leakage_policy": (
                "ground_truth_loaded_only_after_action_selection_and_replanning"
            ),
        },
        "runtime": {
            "current_cache_replay_wall_seconds": wall_seconds,
            "recorded_uncached_model_seconds_for_consumed_observations": sum(
                item["metrics"]["model_load_seconds"]
                + item["metrics"]["inference_seconds"]
                for item in consumed_inferences
            ),
            "recorded_uncached_inference_seconds_per_observation": [
                item["metrics"]["inference_seconds"]
                for item in consumed_inferences
            ],
        },
        "limitations": [
            "Raw-logit softmax and multi-view product fusion are uncalibrated.",
            "Selected views are replayed from deterministic pre-captures.",
            "No live robot or simulator action is executed by this runner.",
            "The result is not a statistical guarantee or final paper result.",
        ],
    }
    output_dir = args.output_root / args.episode_id
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "episode.json"
    output_path.write_text(
        json.dumps(episode_result, indent=2) + "\n", encoding="utf-8"
    )
    print(f"STATUS={episode_result['status']}")
    print(f"FIRST_ACTION={first_action['type']}")
    print(f"REPLAN={replan['type']}")
    print(f"FINAL_ACTION={final_action['type']}")
    print(
        "CACHE_HITS="
        f"{sum(int(item['cache_hit']) for item in consumed_inferences)}"
        f"/{len(consumed_inferences)}"
    )
    print(f"EPISODE_WALL_SECONDS={wall_seconds:.3f}")
    print(f"WROTE={output_path}")


if __name__ == "__main__":
    main()
