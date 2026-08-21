"""Run one-model-at-a-time perception stages on the saved seed 0--2 pilot.

Inference stages read RGB only. Simulator instance IDs, masks, depth, and
ground truth are intentionally reserved for the separate evaluation script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/perception/grounding_pilot_seed0_2.json"


def parse_args() -> argparse.Namespace:
    """Parse the requested perception stage and cache options."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("qwen_direct", "gdino_detect", "sam2_segment", "sam3_segment"),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def require_single_gpu_only() -> None:
    """Keep each stage to one model process on one visible CUDA device."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    expected = os.environ.get("PHYSICAL_GPU") or visible or "0"
    if not expected.isdigit():
        raise RuntimeError(f"PHYSICAL_GPU must be one integer index: {expected!r}")
    if visible is not None and ("," in visible or not visible.isdigit()):
        raise RuntimeError("Exactly one integer CUDA device index may be visible.")
    if os.environ.get("PHYSICAL_GPU") is not None and visible != expected:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES must equal physical GPU {expected}."
        )
    forbidden = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    present = [name for name in forbidden if os.environ.get(name)]
    if present:
        raise RuntimeError(f"Distributed execution variables are forbidden: {present}")

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Exactly one CUDA device must be visible after single-GPU masking."
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must not be initialized.")
    torch.cuda.set_device("cuda:0")


def resolve_project_path(value: str | Path) -> Path:
    """Resolve repository-relative paths while preserving absolute paths."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict[str, Any]:
    """Load the pilot contract and reject any training-enabled variant."""
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("training_performed") is not False:
        raise ValueError("This pilot must remain inference-only.")
    return config


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a formatted JSON artifact, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    """Return the SHA-256 digest used to validate cached image results."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_result_matches_image(result_path: Path, image_path: Path) -> bool:
    """Accept a cached result only when it was computed from the same RGB bytes."""
    if not result_path.is_file():
        return False
    try:
        cached = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        cached.get("image_path") == str(image_path)
        and cached.get("image_sha256") == sha256(image_path)
    )


def sample_items(config: dict[str, Any], limit: int | None) -> list[dict[str, Any]]:
    """Return configured samples, optionally truncated for a smoke test."""
    samples = list(config["samples"])
    return samples if limit is None else samples[:limit]


def output_root(config: dict[str, Any]) -> Path:
    """Resolve the configured artifact directory."""
    return resolve_project_path(config["output_root"])


def rgb_path(sample: dict[str, Any]) -> Path:
    """Return the required RGB path for one observation sample."""
    path = resolve_project_path(sample["observation_dir"]) / "rgb.png"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def cuda_sample_start() -> float:
    """Start a synchronized per-sample timing and peak-memory window."""
    import torch

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats("cuda:0")
    torch.cuda.synchronize("cuda:0")
    return time.perf_counter()


def cuda_sample_metrics(started: float) -> dict[str, Any]:
    """Finish a synchronized timing window and report peak CUDA memory."""
    import torch

    torch.cuda.synchronize("cuda:0")
    return {
        "runtime_seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated("cuda:0")),
        "peak_gpu_memory_gib": round(
            torch.cuda.max_memory_allocated("cuda:0") / (1024**3), 4
        ),
    }


def extract_json_value(text: str) -> dict[str, Any] | list[Any]:
    """Recover the first balanced JSON value from otherwise noisy output."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        value = json.loads(stripped)
        if isinstance(value, (dict, list)):
            return value
    except json.JSONDecodeError:
        pass

    starts = [match.start() for match in re.finditer(r"[\{\[]", stripped)]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(stripped)):
            character = stripped[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
                continue
            if character == '"':
                in_string = True
            elif character in "{[":
                depth += 1
            elif character in "}]":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start : index + 1]
                    try:
                        value = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(value, (dict, list)):
                        return value
    raise ValueError("No valid JSON object or list was found in model output.")


def normalize_qwen_box(
    parsed_value: dict[str, Any] | list[Any], width: int, height: int
) -> list[float] | None:
    """Convert Qwen's 0--1000 ``xyxy`` box to clipped image pixels."""
    if isinstance(parsed_value, list):
        if not parsed_value:
            return None
        if len(parsed_value) != 1 or not isinstance(parsed_value[0], dict):
            raise ValueError(
                "Direct grounding output must contain zero or one JSON object."
            )
        parsed = parsed_value[0]
    else:
        parsed = parsed_value
    box = parsed.get("bbox_2d")
    if box is None:
        return None
    if not isinstance(box, list) or len(box) != 4:
        raise ValueError(f"bbox_2d must be null or length four, got {box!r}")
    values = [float(value) for value in box]
    if any(value < 0.0 or value > 1000.0 for value in values):
        raise ValueError(f"Normalized coordinates escape [0, 1000]: {values}")
    if any(abs(value - round(value)) > 1e-6 for value in values):
        raise ValueError(f"Normalized coordinates must be integers: {values}")
    x0, y0, x1, y1 = values
    values = [
        x0 * width / 1000.0,
        y0 * height / 1000.0,
        x1 * width / 1000.0,
        y1 * height / 1000.0,
    ]
    x0, y0, x1, y1 = values
    clipped = [
        max(0.0, min(float(width - 1), x0)),
        max(0.0, min(float(height - 1), y0)),
        max(0.0, min(float(width - 1), x1)),
        max(0.0, min(float(height - 1), y1)),
    ]
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        raise ValueError(f"Invalid predicted box after clipping: {clipped}")
    return clipped


def run_qwen_direct(
    config: dict[str, Any], samples: list[dict[str, Any]], force: bool
) -> dict[str, Any]:
    """Run the RGB-only direct-grounding baseline with per-sample caching."""
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    model_config = config["models"]["qwen"]
    model_path = Path(model_config["path"])
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        model_path,
        local_files_only=True,
        min_pixels=4 * 28 * 28,
        max_pixels=int(model_config["max_pixels"]),
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    model.eval()
    load_seconds = time.perf_counter() - load_started

    results = []
    for sample in samples:
        destination = output_root(config) / "qwen_direct" / sample["sample_id"]
        result_path = destination / "result.json"
        source_image = rgb_path(sample)
        if not force and cached_result_matches_image(result_path, source_image):
            results.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue

        image = Image.open(source_image).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": config["task"]["qwen_direct_prompt"]},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to("cuda:0")
        started = cuda_sample_start()
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                use_cache=True,
            )
        metrics = cuda_sample_metrics(started)
        generated_tokens = generated[:, inputs["input_ids"].shape[1] :]
        raw_text = processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        destination.mkdir(parents=True, exist_ok=True)
        # Keep the exact response even when the structured parse fails.
        (destination / "raw_response.txt").write_text(
            raw_text + "\n", encoding="utf-8"
        )

        parse_status = "success"
        parse_error = None
        parsed = None
        box_xyxy = None
        try:
            parsed = extract_json_value(raw_text)
            box_xyxy = normalize_qwen_box(parsed, image.width, image.height)
        except (ValueError, TypeError) as error:
            parse_status = "failed"
            parse_error = str(error)

        result = {
            "schema_version": "qwen-direct-grounding-output-v1",
            "sample_id": sample["sample_id"],
            "image_path": str(source_image),
            "image_sha256": sha256(source_image),
            "instruction": config["task"]["instruction"],
            "model": model_config["repository"],
            "parsed_output": parsed,
            "bbox_xyxy_pixels": box_xyxy,
            "parse_status": parse_status,
            "parse_error": parse_error,
            "metrics": metrics,
            "training_performed": False,
            "simulator_ground_truth_used_for_inference": False,
            "valid_for_final_evaluation": False,
        }
        write_json(result_path, result)
        results.append(result)
        del inputs, generated, generated_tokens

    return {
        "stage": "qwen_direct",
        "model_load_seconds": load_seconds,
        "results": results,
    }


def run_gdino_detect(
    config: dict[str, Any], samples: list[dict[str, Any]], force: bool
) -> dict[str, Any]:
    """Generate RGB-only open-vocabulary boxes, reusing saved results by default."""
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    model_config = config["models"]["grounding_dino"]
    model_path = Path(model_config["path"])
    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_path,
        local_files_only=True,
    ).to("cuda:0")
    model.eval()
    load_seconds = time.perf_counter() - load_started

    results_out = []
    for sample in samples:
        destination = output_root(config) / "grounded_sam2" / sample["sample_id"]
        result_path = destination / "detections.json"
        source_image = rgb_path(sample)
        if not force and cached_result_matches_image(result_path, source_image):
            results_out.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue

        image = Image.open(source_image).convert("RGB")
        concepts = list(config["task"]["open_vocabulary_concepts"])
        started = cuda_sample_start()
        annotations = []
        # Separate prompts keep each saved score tied to one requested concept.
        for concept in concepts:
            text_prompt = f"{concept}."
            inputs = processor(
                images=image, text=text_prompt, return_tensors="pt"
            ).to("cuda:0")
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                outputs = model(**inputs)
            processed = processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=float(model_config["box_threshold"]),
                text_threshold=float(model_config["text_threshold"]),
                target_sizes=[image.size[::-1]],
            )[0]
            boxes = processed["boxes"].detach().cpu().float().tolist()
            scores = processed["scores"].detach().cpu().float().tolist()
            labels = processed.get("text_labels", processed.get("labels", []))
            for box, score, raw_label in zip(boxes, scores, labels):
                annotations.append(
                    {
                        "detection_id": f"detection_{len(annotations):03d}",
                        "concept": concept,
                        "raw_text_label": str(raw_label),
                        "label": concept,
                        "score": score,
                        "bbox_xyxy_pixels": box,
                    }
                )
            del inputs, outputs, processed
        metrics = cuda_sample_metrics(started)
        result = {
            "schema_version": "grounding-dino-detections-v1",
            "sample_id": sample["sample_id"],
            "image_path": str(source_image),
            "image_sha256": sha256(source_image),
            "text_prompts": [f"{concept}." for concept in concepts],
            "model": {
                "repository": model_config["repository"],
                "revision": model_config["revision"],
            },
            "thresholds": {
                "box": model_config["box_threshold"],
                "text": model_config["text_threshold"],
            },
            "annotations": annotations,
            "metrics": metrics,
            "training_performed": False,
            "simulator_ground_truth_used_for_inference": False,
            "valid_for_final_evaluation": False,
        }
        write_json(result_path, result)
        results_out.append(result)

    return {
        "stage": "gdino_detect",
        "model_load_seconds": load_seconds,
        "results": results_out,
    }


def run_sam2_segment(
    config: dict[str, Any], samples: list[dict[str, Any]], force: bool
) -> dict[str, Any]:
    """Segment the cached detector boxes without consulting simulator masks."""
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    model_config = config["models"]["sam2"]
    load_started = time.perf_counter()
    model = build_sam2(
        model_config["model_config"],
        model_config["path"],
        device="cuda:0",
    )
    model.eval()
    predictor = SAM2ImagePredictor(model)
    load_seconds = time.perf_counter() - load_started

    results_out = []
    for sample in samples:
        destination = output_root(config) / "grounded_sam2" / sample["sample_id"]
        detection_path = destination / "detections.json"
        if not detection_path.is_file():
            raise FileNotFoundError(
                f"Run gdino_detect before sam2_segment: {detection_path}"
            )
        result_path = destination / "segmentations.json"
        source_image = rgb_path(sample)
        if not force and cached_result_matches_image(result_path, source_image):
            results_out.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue

        detection_result = json.loads(detection_path.read_text(encoding="utf-8"))
        if not cached_result_matches_image(detection_path, source_image):
            raise RuntimeError(
                f"Detector cache does not match current RGB: {detection_path}"
            )
        detections = detection_result["annotations"]
        image = Image.open(source_image).convert("RGB")
        image_array = np.array(image, copy=True)
        boxes = np.asarray(
            [annotation["bbox_xyxy_pixels"] for annotation in detections],
            dtype=np.float32,
        )
        started = cuda_sample_start()
        predictor.set_image(image_array)
        if boxes.size:
            with torch.inference_mode(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16
            ):
                masks, mask_scores, _logits = predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=boxes,
                    multimask_output=False,
                )
            if masks.ndim == 4:
                # One box and one mask are kept in the same detector order.
                masks = masks.squeeze(1)
            masks = masks.astype(bool)
            mask_scores_list = np.asarray(mask_scores).reshape(-1).tolist()
        else:
            masks = np.zeros((0, image.height, image.width), dtype=bool)
            mask_scores_list = []
        metrics = cuda_sample_metrics(started)

        output_annotations = []
        for index, (annotation, mask, mask_score) in enumerate(
            zip(detections, masks, mask_scores_list)
        ):
            mask_name = f"{annotation['detection_id']}_mask.png"
            Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(
                destination / mask_name
            )
            output_annotations.append(
                {
                    **annotation,
                    "sam2_mask_score": float(mask_score),
                    "mask_path": mask_name,
                    "mask_pixel_count": int(mask.sum()),
                }
            )
        result = {
            "schema_version": "grounded-sam2-segmentations-v1",
            "sample_id": sample["sample_id"],
            "image_path": str(source_image),
            "image_sha256": sha256(source_image),
            "models": {
                "detector": config["models"]["grounding_dino"]["repository"],
                "segmenter": model_config["repository"],
                "segmenter_revision": model_config["revision"],
            },
            "annotations": output_annotations,
            "metrics": metrics,
            "training_performed": False,
            "simulator_ground_truth_used_for_inference": False,
            "valid_for_final_evaluation": False,
        }
        write_json(result_path, result)
        results_out.append(result)

    return {
        "stage": "sam2_segment",
        "model_load_seconds": load_seconds,
        "results": results_out,
    }


def run_sam3_segment(
    config: dict[str, Any], samples: list[dict[str, Any]], force: bool
) -> dict[str, Any]:
    """Run text-prompted RGB segmentation with per-sample result caching."""
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model_config = config["models"]["sam3"]
    checkpoint = Path(model_config["path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(
            "SAM3 checkpoint is unavailable. Request access at "
            "https://huggingface.co/facebook/sam3 and download sam3.pt to "
            f"{checkpoint}."
        )
    load_started = time.perf_counter()
    model = build_sam3_image_model(
        checkpoint_path=str(checkpoint),
        load_from_HF=False,
        device="cuda",
        eval_mode=True,
        compile=False,
    )
    processor = Sam3Processor(
        model,
        device="cuda",
        confidence_threshold=float(model_config["confidence_threshold"]),
    )
    load_seconds = time.perf_counter() - load_started

    results_out = []
    for sample in samples:
        destination = output_root(config) / "sam3" / sample["sample_id"]
        result_path = destination / "segmentations.json"
        if result_path.is_file() and not force:
            results_out.append(json.loads(result_path.read_text(encoding="utf-8")))
            continue
        destination.mkdir(parents=True, exist_ok=True)
        image = Image.open(rgb_path(sample)).convert("RGB")
        started = cuda_sample_start()
        state = processor.set_image(image)
        annotations = []
        for concept_index, concept in enumerate(
            config["task"]["open_vocabulary_concepts"]
        ):
            output = processor.set_text_prompt(state=state, prompt=concept)
            masks = output["masks"].detach().cpu().numpy()
            boxes = output["boxes"].detach().cpu().float().numpy()
            scores = output["scores"].detach().cpu().float().numpy()
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            for instance_index, (mask, box, score) in enumerate(
                zip(masks, boxes, scores)
            ):
                mask_name = (
                    f"concept_{concept_index:02d}_instance_{instance_index:03d}_mask.png"
                )
                binary_mask = np.asarray(mask, dtype=bool)
                Image.fromarray(
                    binary_mask.astype(np.uint8) * 255, mode="L"
                ).save(destination / mask_name)
                annotations.append(
                    {
                        "concept": concept,
                        "score": float(score),
                        "bbox_xyxy_pixels": box.tolist(),
                        "mask_path": mask_name,
                        "mask_pixel_count": int(binary_mask.sum()),
                    }
                )
            # A concept must not carry prompt state into the next concept.
            processor.reset_all_prompts(state)
        metrics = cuda_sample_metrics(started)
        result = {
            "schema_version": "sam3-segmentations-v1",
            "sample_id": sample["sample_id"],
            "image_path": str(rgb_path(sample)),
            "model": {
                "repository": model_config["repository"],
                "revision": model_config["revision"],
            },
            "confidence_threshold": model_config["confidence_threshold"],
            "annotations": annotations,
            "metrics": metrics,
            "training_performed": False,
            "simulator_ground_truth_used_for_inference": False,
            "valid_for_final_evaluation": False,
        }
        write_json(result_path, result)
        results_out.append(result)
    return {
        "stage": "sam3_segment",
        "model_load_seconds": load_seconds,
        "results": results_out,
    }


def main() -> None:
    """Run one selected perception stage and save its reproducibility record."""
    args = parse_args()
    require_single_gpu_only()
    config = load_config(args.config.resolve())
    samples = sample_items(config, args.limit)
    stage_functions = {
        "qwen_direct": run_qwen_direct,
        "gdino_detect": run_gdino_detect,
        "sam2_segment": run_sam2_segment,
        "sam3_segment": run_sam3_segment,
    }
    run_started = time.perf_counter()
    stage_result = stage_functions[args.stage](config, samples, args.force)
    summary = {
        "schema_version": "perception-grounding-stage-summary-v1",
        "experiment_id": config["experiment_id"],
        "stage": args.stage,
        "sample_count": len(stage_result["results"]),
        "model_load_seconds": stage_result["model_load_seconds"],
        "total_runtime_seconds": time.perf_counter() - run_started,
        "physical_gpu": int(
            os.environ.get("PHYSICAL_GPU")
            or os.environ.get("CUDA_VISIBLE_DEVICES")
            or "0"
        ),
        "logical_gpu": "cuda:0",
        "single_model_instance": True,
        "batch_size": 1,
        "training_performed": False,
        "calibration_performed": False,
        "valid_for_final_evaluation": False,
        "sample_metrics": [
            {
                "sample_id": result["sample_id"],
                **result["metrics"],
            }
            for result in stage_result["results"]
        ],
    }
    summary_path = output_root(config) / f"{args.stage}_summary.json"
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
