"""Export anonymous VLM samples from benchmark simulator observations."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OBSERVATION_ROOT = ROOT / "outputs/benchmark_observations"
DEFAULT_OUTPUT = ROOT / "outputs/vlm_dataset"
VIEWS = ("left", "center", "right")
INSTRUCTION = "Retrieve the red object located inside the open container."

ANONYMOUS_IDS = {
    "target_red": "object_001",
    "occluder_orange": "object_002",
    "distractor_yellow": "object_003",
    "distractor_blue": "object_004",
    "distractor_green": "object_005",
    "boundary_purple": "object_006",
    "rear_red_candidate": "object_007",
}
CONTAINER_ID = "container_001"
CONTAINER_RELATION_LABELS = ["inside", "outside", "behind", "near_boundary", "unknown"]
CONTAINER_RELATION_GROUND_TRUTH = {
    "target_red": "inside",
    "distractor_yellow": "inside",
    "distractor_blue": "outside",
    "distractor_green": "outside",
    "boundary_purple": "near_boundary",
    "rear_red_candidate": "behind",
}


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def save_candidate_assets(
    rgb: np.ndarray,
    instance_ids: np.ndarray,
    instance_id: int,
    bbox: list[int],
    sample_dir: Path,
    candidate_id: str,
) -> tuple[Path, Path, Path]:
    mask = instance_ids == instance_id
    mask_path = sample_dir / f"{candidate_id}_mask.png"
    Image.fromarray((mask * 255).astype(np.uint8), "L").save(mask_path)
    x0, y0, x1, y1 = bbox
    crop = rgb[y0 : y1 + 1, x0 : x1 + 1].copy()
    crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    crop[~crop_mask] = 0
    crop_path = sample_dir / f"{candidate_id}_crop.png"
    Image.fromarray(crop, "RGB").save(crop_path)

    height, width = rgb.shape[:2]
    object_width = x1 - x0 + 1
    object_height = y1 - y0 + 1
    padding = max(40, 2 * max(object_width, object_height))
    context_x0 = max(0, x0 - padding)
    context_y0 = max(0, y0 - padding)
    context_x1 = min(width - 1, x1 + padding)
    context_y1 = min(height - 1, y1 + padding)
    context = Image.fromarray(
        rgb[context_y0 : context_y1 + 1, context_x0 : context_x1 + 1],
        "RGB",
    )
    context_draw = ImageDraw.Draw(context)
    context_draw.rectangle(
        (
            x0 - context_x0,
            y0 - context_y0,
            x1 - context_x0,
            y1 - context_y0,
        ),
        outline=(0, 255, 255),
        width=3,
    )
    context_path = sample_dir / f"{candidate_id}_context.png"
    context.save(context_path)
    return crop_path, mask_path, context_path


def save_reference_assets(
    rgb: np.ndarray,
    instance_ids: np.ndarray,
    instance_id: int,
    sample_dir: Path,
) -> tuple[Path, Path]:
    """Save an anonymous container mask and a human-readable RGB overlay."""
    mask = instance_ids == instance_id
    mask_path = sample_dir / f"{CONTAINER_ID}_mask.png"
    Image.fromarray((mask * 255).astype(np.uint8), "L").save(mask_path)

    overlay = rgb.astype(np.float32).copy()
    cyan = np.asarray([0, 255, 255], dtype=np.float32)
    overlay[mask] = 0.72 * overlay[mask] + 0.28 * cyan
    overlay_image = Image.fromarray(
        np.clip(overlay, 0, 255).astype(np.uint8),
        "RGB",
    )
    overlay_draw = ImageDraw.Draw(overlay_image)
    try:
        overlay_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
    except OSError:
        overlay_font = ImageFont.load_default()
    overlay_draw.rounded_rectangle(
        (12, 12, 282, 44),
        radius=6,
        fill=(0, 0, 0),
    )
    overlay_draw.text(
        (20, 19),
        "cyan = container_001 surface",
        fill=(0, 255, 255),
        font=overlay_font,
    )
    overlay_path = sample_dir / f"{CONTAINER_ID}_overlay.png"
    overlay_image.save(overlay_path)
    return mask_path, overlay_path


def relation_queries() -> list[dict]:
    queries = []
    for semantic_id, candidate_id in ANONYMOUS_IDS.items():
        if semantic_id == "occluder_orange":
            continue
        queries.append(
            {
                "query_id": f"{candidate_id}_to_container",
                "source_id": candidate_id,
                "target_id": CONTAINER_ID,
                "label_space": CONTAINER_RELATION_LABELS,
            }
        )
    queries.append(
        {
            "query_id": "object_002_to_object_001",
            "source_id": "object_002",
            "target_id": "object_001",
            "label_space": ["occludes", "not_occludes", "unknown"],
        }
    )
    return queries


def export_view(
    view: str,
    output_root: Path,
    observation_root: Path = OBSERVATION_ROOT,
    episode_id: str = "benchmark_seed000",
) -> tuple[Path, Path]:
    source_dir = observation_root / view
    objects = json.loads((source_dir / "objects.json").read_text(encoding="utf-8"))
    rgb = np.asarray(Image.open(source_dir / "rgb.png").convert("RGB"))
    instance_ids = np.load(source_dir / "instance_ids.npy")
    sample_id = f"{episode_id}_{view}"
    sample_dir = output_root / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = sample_dir / "rgb.png"
    Image.fromarray(rgb, "RGB").save(rgb_path)

    candidates = []
    for semantic_id, candidate_id in ANONYMOUS_IDS.items():
        observation = objects[semantic_id]
        if not observation["visible"] or observation["bbox_xyxy"] is None:
            continue
        crop_path, mask_path, context_path = save_candidate_assets(
            rgb,
            instance_ids,
            observation["instance_ids"][0],
            observation["bbox_xyxy"],
            sample_dir,
            candidate_id,
        )
        candidates.append(
            {
                "candidate_id": candidate_id,
                "bbox_xyxy": observation["bbox_xyxy"],
                "crop_path": relative(crop_path),
                "mask_path": relative(mask_path),
                "context_path": relative(context_path),
            }
        )

    queries = relation_queries()
    container_observation = objects["container"]
    container_mask_path, container_overlay_path = save_reference_assets(
        rgb,
        instance_ids,
        container_observation["instance_ids"][0],
        sample_dir,
    )
    model_input = {
        "schema_version": "vlm-input-v1",
        "sample_id": sample_id,
        "episode_id": episode_id,
        "view_id": view,
        "instruction": INSTRUCTION,
        "image": {
            "rgb_path": relative(rgb_path),
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
        },
        "candidates": candidates,
        "reference_entities": [
            {
                "reference_id": CONTAINER_ID,
                "bbox_xyxy": container_observation["bbox_xyxy"],
                "description": "open container reference region",
                "mask_path": relative(container_mask_path),
                "overlay_path": relative(container_overlay_path),
            }
        ],
        "relation_queries": queries,
    }
    ground_truth_relations = [
        {
            "query_id": f"{ANONYMOUS_IDS[semantic_id]}_to_container",
            "label": label,
        }
        for semantic_id, label in CONTAINER_RELATION_GROUND_TRUTH.items()
    ]
    ground_truth_relations.append(
        {"query_id": "object_002_to_object_001", "label": "occludes"}
    )
    ground_truth = {
        "schema_version": "vlm-ground-truth-v1",
        "sample_id": sample_id,
        "target_candidate_id": "object_001",
        "relations": ground_truth_relations,
        "leakage_policy": "never_expose_to_model_inference",
    }
    input_path = sample_dir / "input.json"
    ground_truth_path = sample_dir / "ground_truth.json"
    input_path.write_text(json.dumps(model_input, indent=2) + "\n", encoding="utf-8")
    ground_truth_path.write_text(
        json.dumps(ground_truth, indent=2) + "\n", encoding="utf-8"
    )
    return input_path, ground_truth_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--observation-root",
        type=Path,
        default=OBSERVATION_ROOT,
    )
    parser.add_argument("--episode-id", default="benchmark_seed000")
    parser.add_argument(
        "--views",
        nargs="+",
        default=list(VIEWS),
        help="Observation directories to export (default: left center right).",
    )
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    manifest = {"schema_version": "vlm-dataset-manifest-v1", "samples": []}
    for view in args.views:
        input_path, ground_truth_path = export_view(
            view,
            output_root,
            observation_root=args.observation_root.resolve(),
            episode_id=args.episode_id,
        )
        manifest["samples"].append(
            {
                "input": relative(input_path),
                "ground_truth": relative(ground_truth_path),
                "split": "development_only_pending_seeded_dataset",
            }
        )
        print(f"EXPORTED={input_path}")
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"MANIFEST={manifest_path}")


if __name__ == "__main__":
    main()
