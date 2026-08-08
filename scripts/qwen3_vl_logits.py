"""Produce VLM-contract raw logits with Qwen3-VL-8B-Instruct.

The adapter performs forced-choice scoring. Candidate IDs and relation labels
are mapped to single-token letters, and the corresponding next-token language
model logits are saved before any softmax or calibration.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = Path(
    os.environ.get(
        "EFFICIENT_ROBOTICS_QWEN_MODEL",
        ROOT / "models" / "Qwen3-VL-8B-Instruct",
    )
)
MODEL_REPOSITORY = "Qwen/Qwen3-VL-8B-Instruct"
PROMPT_VERSION = "qwen3-vl-joint-candidate-v6-rgb-box-crop-debiased"
CHOICE_LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="vlm-input-v1 JSON file")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=512 * 28 * 28,
        help="Maximum pixels per image passed to the Qwen visual processor.",
    )
    return parser.parse_args()


def configured_physical_gpu() -> int:
    """Read the host GPU index recorded in inference provenance."""
    value = os.environ.get("PHYSICAL_GPU")
    if value is None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        value = visible if visible and "," not in visible else "0"
    if not value.isdigit():
        raise RuntimeError(f"PHYSICAL_GPU must be one integer index, got {value!r}")
    return int(value)


def require_single_gpu_only() -> None:
    """Enforce the single-device setup used for comparable memory metrics.

    CUDA renumbers a masked physical device to ``cuda:0``. ``PHYSICAL_GPU``
    keeps the host index for provenance while ``CUDA_VISIBLE_DEVICES`` applies
    the mask.
    """
    expected = str(configured_physical_gpu())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and ("," in visible or not visible.isdigit()):
        raise RuntimeError("Exactly one integer CUDA device index may be visible.")
    if os.environ.get("PHYSICAL_GPU") is not None and visible != expected:
        raise RuntimeError(
            f"This adapter requires CUDA_VISIBLE_DEVICES={expected}; "
            f"received {visible!r} for the configured single physical GPU."
        )
    if any(
        os.environ.get(name)
        for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR")
    ):
        raise RuntimeError("Distributed execution environment variables are forbidden.")


def resolve_asset_path(path_text: str, input_path: Path) -> Path:
    """Resolve assets from the contract, repository, or input directory."""
    path = Path(path_text)
    candidates = (
        path,
        ROOT / path,
        input_path.parent / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"VLM asset does not exist: {path_text}")


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def candidate_overlay(scene: Image.Image, candidates: list[dict]) -> Image.Image:
    """Draw anonymous candidate IDs without adding semantic hints."""
    overlay = scene.copy()
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
    except OSError:
        font = ImageFont.load_default()
    colors = (
        "#00FFFF",
        "#FF00FF",
        "#FFFF00",
        "#00FF00",
        "#FF8000",
        "#8080FF",
        "#FFFFFF",
    )
    for index, candidate in enumerate(candidates):
        x0, y0, x1, y1 = candidate["bbox_xyxy"]
        color = colors[index % len(colors)]
        draw.rectangle((x0, y0, x1, y1), outline=color, width=3)
        label = candidate["candidate_id"]
        label_box = draw.textbbox((x0, y0), label, font=font)
        label_width = label_box[2] - label_box[0]
        label_height = label_box[3] - label_box[1]
        label_y = max(0, y0 - label_height - 4)
        draw.rectangle(
            (x0, label_y, x0 + label_width + 4, label_y + label_height + 4),
            fill="black",
        )
        draw.text((x0 + 2, label_y + 2), label, fill=color, font=font)
    return overlay


def build_visual_content(model_input: dict, input_path: Path) -> list[dict]:
    """Build the common anonymous visual evidence used by every question.

    Identity and relation questions see the same scene, overlays, and crops;
    only their final text question changes. Candidate metadata beyond the
    contract fields is never exposed here.
    """
    scene = load_rgb(
        resolve_asset_path(model_input["image"]["rgb_path"], input_path)
    )
    content: list[dict] = [
        {
            "type": "text",
            "text": (
                "Use only the supplied scene and anonymous candidate images. "
                "Candidate IDs do not reveal semantic identity. "
                "The first image is the unmodified full scene. The second image "
                "is the same scene with candidate bounding boxes and anonymous "
                "IDs overlaid."
            ),
        },
        {
            "type": "image",
            "image": scene,
        },
        {
            "type": "image",
            "image": candidate_overlay(scene, model_input["candidates"]),
        },
    ]
    references = model_input.get("reference_entities", [])
    for reference in references:
        if reference.get("overlay_path"):
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"Reference overlay for {reference['reference_id']}. "
                            "Cyan pixels mark visible container material, including "
                            "the floor and walls. The region enclosed by the "
                            "visible inner walls is the container interior:"
                        ),
                    },
                    {
                        "type": "image",
                        "image": load_rgb(
                            resolve_asset_path(
                                reference["overlay_path"], input_path
                            )
                        ),
                    },
                ]
            )
    width = model_input["image"]["width"]
    height = model_input["image"]["height"]
    for candidate in model_input["candidates"]:
        x0, y0, x1, y1 = candidate["bbox_xyxy"]
        # Qwen grounding prompts use a resolution-independent 0--1000 grid.
        normalized_bbox = [
            round(1000 * x0 / width),
            round(1000 * y0 / height),
            round(1000 * x1 / width),
            round(1000 * y1 / height),
        ]
        candidate_content = [
            {
                "type": "text",
                "text": (
                    f"Bounding-box RGB crop for {candidate['candidate_id']}; "
                    f"full-scene bbox on a 0-1000 grid is {normalized_bbox}:"
                ),
            },
            {
                "type": "image",
                "image": load_rgb(
                    resolve_asset_path(candidate["crop_path"], input_path)
                ),
            },
        ]
        if candidate.get("context_path"):
            candidate_content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"Context around {candidate['candidate_id']}; the cyan "
                            "rectangle marks the candidate while surrounding "
                            "container geometry is intentionally preserved:"
                        ),
                    },
                    {
                        "type": "image",
                        "image": load_rgb(
                            resolve_asset_path(
                                candidate["context_path"], input_path
                            )
                        ),
                    },
                ]
            )
        content.extend(candidate_content)
    return content


def letter_mapping(values: list[str]) -> list[tuple[str, str]]:
    """Bind choices to single-letter outputs while preserving input order."""
    if not 2 <= len(values) <= len(CHOICE_LETTERS):
        raise ValueError(f"Forced-choice dimension is unsupported: {len(values)}")
    return list(zip(CHOICE_LETTERS, values))


def target_question(
    model_input: dict,
    mapping: list[tuple[str, str]] | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    candidate_ids = [
        candidate["candidate_id"] for candidate in model_input["candidates"]
    ]
    mapping = mapping or letter_mapping(candidate_ids)
    choices = "\n".join(f"{letter}: {candidate_id}" for letter, candidate_id in mapping)
    question = (
        f"\nRobot instruction: {model_input['instruction']}\n"
        "Which anonymous candidate best matches the requested target?\n"
        "Use the candidate's identity, full-scene location, and spatial relation; "
        "do not decide from crop appearance alone.\n"
        f"{choices}\n"
        "Answer with exactly one uppercase choice letter and no other text."
    )
    return question, mapping


def joint_candidate_question(
    model_input: dict,
    candidate_id: str,
    mapping: list[tuple[str, str]] | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    values = ["matches_instruction", "does_not_match"]
    mapping = mapping or letter_mapping(values)
    choices = "\n".join(f"{letter}: {value}" for letter, value in mapping)
    reference = model_input.get("reference_entities", [{}])[0]
    container_bbox = reference.get("bbox_xyxy", "not_provided")
    question = (
        f"\nRobot instruction: {model_input['instruction']}\n"
        f"Evaluate only candidate {candidate_id} against the complete instruction.\n"
        "A match must satisfy BOTH requirements: (1) it is the requested red "
        "object and (2) it is physically inside the open container. A prominent "
        "red object outside or behind the container does not match. A target "
        "may be tiny or partially occluded, so use its highlighted context and "
        "the visible container walls rather than crop size alone.\n"
        f"Open-container reference bbox in full-image pixels: {container_bbox}.\n"
        f"{choices}\n"
        "Answer with exactly one uppercase choice letter and no other text."
    )
    return question, mapping


def candidate_identity_question(
    model_input: dict,
    candidate_id: str,
    mapping: list[tuple[str, str]] | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    values = ["matches_target_description", "does_not_match_target_description"]
    mapping = mapping or letter_mapping(values)
    choices = "\n".join(f"{letter}: {value}" for letter, value in mapping)
    description = model_input.get(
        "target_description", model_input["instruction"]
    )
    question = (
        f"\nTarget appearance description: {description}\n"
        f"Evaluate only candidate {candidate_id} for visual identity.\n"
        "Use the original scene, bounding-box RGB crop, and candidate context. "
        "Judge color, shape, logo, texture, and other visible appearance cues. "
        "Do not use inside/outside/behind location for this identity score; "
        "spatial relation is evaluated by a separate question.\n"
        f"{choices}\n"
        "Answer with exactly one uppercase choice letter and no other text."
    )
    return question, mapping


def relation_question(
    query: dict,
    mapping: list[tuple[str, str]] | None = None,
) -> tuple[str, list[tuple[str, str]]]:
    """Ask for one relation factor without folding in the other factors."""
    labels = list(query["label_space"])
    mapping = mapping or letter_mapping(labels)
    choices = "\n".join(f"{letter}: {label}" for letter, label in mapping)
    target_note = (
        "container_001 denotes the basket/container visible in the full scene; "
        "it may be open or covered. "
        if query["target_id"] == "container_001"
        else ""
    )
    relation_type = query.get(
        "relation_type", "legacy_mutually_exclusive_relation"
    )
    definitions = {
        "inside": "physically within the open container interior",
        "outside": "physically outside the open container interior",
        "behind": "positioned farther away behind the reference from this camera view",
        "near_boundary": "on, touching, or straddling the container boundary",
        "unknown": "not reliably decidable from this supplied view",
        "occludes": "source blocks the view of at least part of the target",
        "not_occludes": "source does not block the view of the target",
        "yes": "the queried independent relation is visibly present",
        "no": "the queried independent relation is visibly absent",
    }
    applicable_definitions = "; ".join(
        f"{label} = {definitions[label]}"
        for label in labels
        if label in definitions
    )
    factor_instructions = {
        "membership": (
            "Choose only the source object's membership relative to the "
            "container: inside, outside, or unknown. Do not encode behind or "
            "occlusion in this membership choice. Projected overlap with the "
            "basket is not proof of inside. If the source base and the relevant "
            "inner/back boundary are hidden, choose unknown rather than guessing "
            "inside."
        ),
        "behind": (
            "Judge only whether the source is behind the reference from this "
            "current camera viewpoint. Answer yes, no, or unknown independently "
            "of inside/outside membership. If the current view cannot distinguish "
            "an object inside the cavity from an object just beyond the hidden "
            "back wall, choose unknown."
        ),
        "occluded_by": (
            "Judge only whether the source is visually occluded by the "
            "reference container, its rim, or its cover in this image. Answer "
            "yes, no, or unknown independently of membership and behind."
        ),
        "legacy_mutually_exclusive_relation": (
            "Choose the single supplied relation label best supported by the view."
        ),
    }
    question = (
        "\nDetermine one factor of the directly visible spatial relation.\n"
        f"Relation factor: {relation_type}\n"
        f"{factor_instructions.get(relation_type, factor_instructions['legacy_mutually_exclusive_relation'])}\n"
        f"Source: {query['source_id']}\n"
        f"Target: {query['target_id']}\n"
        f"{target_note}"
        f"Label definitions: {applicable_definitions}.\n"
        "Use the full scene, ID overlay, candidate context, and cyan container "
        "surface overlay together. For a membership query on an open basket, "
        "decide relative to the cavity bounded by the visible INNER walls. An "
        "object resting on the container floor is inside. If a cover prevents "
        "the supplied image from revealing membership, use unknown. "
        "near_boundary is only for an object touching or straddling an inner "
        "wall or rim. "
        "Use unknown when the supplied view does not support a reliable decision.\n"
        f"{choices}\n"
        "Answer with exactly one uppercase choice letter and no other text."
    )
    return question, mapping


def prepare_inputs(
    processor: Any,
    visual_content: list[dict],
    question: str,
    device: str,
) -> Any:
    messages = [
        {
            "role": "user",
            "content": visual_content + [{"type": "text", "text": question}],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs.to(device)


def choice_token_ids(tokenizer: Any, mapping: list[tuple[str, str]]) -> list[int]:
    """Validate that each forced choice occupies exactly one LM token."""
    token_ids = []
    for letter, _value in mapping:
        encoded = tokenizer.encode(letter, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(
                f"Choice letter {letter!r} is not one token: {encoded}. "
                "The forced-choice logit contract cannot be applied."
            )
        token_ids.append(encoded[0])
    return token_ids


def score_question(
    model: Any,
    processor: Any,
    visual_content: list[dict],
    question: str,
    mapping: list[tuple[str, str]],
    device: str,
) -> list[float]:
    """Read pre-softmax next-token logits without generating an answer."""
    import torch

    inputs = prepare_inputs(processor, visual_content, question, device)
    token_ids = choice_token_ids(processor.tokenizer, mapping)
    with torch.inference_mode():
        output = model(**inputs, use_cache=False, return_dict=True)
    final_logits = output.logits[0, -1].float()
    scores = [float(final_logits[token_id].item()) for token_id in token_ids]
    del output
    del inputs
    return scores


def cyclic_mappings(
    mapping: list[tuple[str, str]],
) -> list[list[tuple[str, str]]]:
    """Rotate values so every value is scored under every letter once."""
    letters = [letter for letter, _value in mapping]
    values = [value for _letter, value in mapping]
    return [
        [
            (letter, values[(index + offset) % len(values)])
            for index, letter in enumerate(letters)
        ]
        for offset in range(len(values))
    ]


def permutation_debiased_scores(
    model: Any,
    processor: Any,
    visual_content: list[dict],
    base_mapping: list[tuple[str, str]],
    question_builder: Any,
    device: str,
) -> list[float]:
    """Average raw logits across cyclic mappings to reduce letter preference."""
    totals = {value: 0.0 for _letter, value in base_mapping}
    mappings = cyclic_mappings(base_mapping)
    for mapping in mappings:
        question, _ = question_builder(mapping)
        scores = score_question(
            model,
            processor,
            visual_content,
            question,
            mapping,
            device,
        )
        for (_letter, value), score in zip(mapping, scores):
            totals[value] += score
    return [
        totals[value] / len(mappings)
        for _letter, value in base_mapping
    ]


def local_hf_revision(model_path: Path) -> str:
    metadata_root = model_path / ".cache" / "huggingface" / "download"
    revisions = set()
    for metadata_path in metadata_root.glob("*.metadata"):
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
        if lines:
            revisions.add(lines[0])
    if len(revisions) == 1:
        return revisions.pop()
    return "local_revision_unknown"


def run_inference(args: argparse.Namespace) -> tuple[dict, dict]:
    require_single_gpu_only()

    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable after masking to physical GPU "
            f"{configured_physical_gpu()}."
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"Exactly one CUDA device must be visible, got {torch.cuda.device_count()}."
        )
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must not be initialized.")

    device = "cuda:0"
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    model_input = json.loads(args.input.read_text(encoding="utf-8"))
    visual_content = build_visual_content(model_input, args.input)

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
    )
    model.to(device)
    model.eval()
    load_seconds = time.perf_counter() - load_started

    inference_started = time.perf_counter()
    target_mapping = letter_mapping(
        [candidate["candidate_id"] for candidate in model_input["candidates"]]
    )
    target_logits = []
    for _letter, candidate_id in target_mapping:
        match_mapping = letter_mapping(
            ["matches_instruction", "does_not_match"]
        )
        match_scores = permutation_debiased_scores(
            model,
            processor,
            visual_content,
            match_mapping,
            lambda mapping, candidate_id=candidate_id: joint_candidate_question(
                model_input, candidate_id, mapping
            ),
            device,
        )
        # The signed margin is comparable across candidates; no softmax is applied.
        target_logits.append(match_scores[0] - match_scores[1])

    relations = []
    for query in model_input["relation_queries"]:
        _relation_prompt, relation_mapping = relation_question(query)
        relation_logits = permutation_debiased_scores(
            model,
            processor,
            visual_content,
            relation_mapping,
            lambda mapping, query=query: relation_question(query, mapping),
            device,
        )
        relations.append(
            {
                "query_id": query["query_id"],
                "labels": [value for _letter, value in relation_mapping],
                "raw_logits": relation_logits,
            }
        )
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started

    revision = local_hf_revision(args.model_path)
    output = {
        "schema_version": "vlm-output-v1",
        "sample_id": model_input["sample_id"],
        "model": {
            "name": "Qwen3-VL-8B-Instruct",
            "checkpoint": f"{MODEL_REPOSITORY}@{revision}",
        },
        "target": {
            "candidate_ids": [value for _letter, value in target_mapping],
            "raw_logits": target_logits,
        },
        "relations": relations,
        "provenance": {
            "prompt_version": PROMPT_VERSION,
            "weights_hash": f"hf_revision:{revision}",
            "device": (
                "cuda:0 (physical GPU "
                f"{configured_physical_gpu()} via CUDA_VISIBLE_DEVICES="
                f"{configured_physical_gpu()})"
            ),
        },
    }
    metrics = {
        "schema_version": "qwen3-vl-inference-metrics-v1",
        "sample_id": model_input["sample_id"],
        "model_load_seconds": load_seconds,
        "inference_seconds": inference_seconds,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_gpu_memory_gib": round(
            torch.cuda.max_memory_allocated(device) / (1024**3), 3
        ),
        "max_pixels_per_image": args.max_pixels,
        "visible_cuda_device_count": torch.cuda.device_count(),
        "visible_cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "scoring": (
            "candidate-wise match-minus-nonmatch and relation forced-choice "
            "scores from cyclic-letter-averaged pre-softmax LM logits"
        ),
    }
    return output, metrics


def main() -> None:
    args = parse_args()
    args.input = args.input.resolve()
    args.model_path = args.model_path.resolve()
    output_path = args.output or args.input.with_name("qwen3_vl_output.json")
    metrics_path = args.metrics_output or args.input.with_name(
        "qwen3_vl_metrics.json"
    )
    output, metrics = run_inference(args)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE={output_path}")
    print(f"METRICS={metrics_path}")
    print(f"PEAK_GPU_MEMORY_GIB={metrics['peak_gpu_memory_gib']}")
    print(f"INFERENCE_SECONDS={metrics['inference_seconds']:.3f}")


if __name__ == "__main__":
    main()
