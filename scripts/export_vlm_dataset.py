"""Export anonymous VLM samples from benchmark simulator observations."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


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
) -> tuple[Path, Path]:
    mask = instance_ids == instance_id
    mask_path = sample_dir / f"{candidate_id}_mask.png"
    Image.fromarray((mask * 255).astype(np.uint8), "L").save(mask_path)
    x0, y0, x1, y1 = bbox
    crop = rgb[y0 : y1 + 1, x0 : x1 + 1].copy()
    crop_mask = mask[y0 : y1 + 1, x0 : x1 + 1]
    crop[~crop_mask] = 0
    crop_path = sample_dir / f"{candidate_id}_crop.png"
    Image.fromarray(crop, "RGB").save(crop_path)
    return crop_path, mask_path


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


def export_view(view: str, output_root: Path) -> tuple[Path, Path]:
    source_dir = OBSERVATION_ROOT / view
    objects = json.loads((source_dir / "objects.json").read_text(encoding="utf-8"))
    rgb = np.asarray(Image.open(source_dir / "rgb.png").convert("RGB"))
    instance_ids = np.load(source_dir / "instance_ids.npy")
    sample_id = f"benchmark_seed000_{view}"
    sample_dir = output_root / "samples" / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = sample_dir / "rgb.png"
    Image.fromarray(rgb, "RGB").save(rgb_path)

    candidates = []
    for semantic_id, candidate_id in ANONYMOUS_IDS.items():
        observation = objects[semantic_id]
        if not observation["visible"] or observation["bbox_xyxy"] is None:
            continue
        crop_path, mask_path = save_candidate_assets(
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
            }
        )

    queries = relation_queries()
    model_input = {
        "schema_version": "vlm-input-v1",
        "sample_id": sample_id,
        "episode_id": "benchmark_seed000",
        "view_id": view,
        "instruction": INSTRUCTION,
        "image": {
            "rgb_path": relative(rgb_path),
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
        },
        "candidates": candidates,
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
    args = parser.parse_args()
    manifest = {"schema_version": "vlm-dataset-manifest-v1", "samples": []}
    for view in VIEWS:
        input_path, ground_truth_path = export_view(view, args.output_root)
        manifest["samples"].append(
            {
                "input": relative(input_path),
                "ground_truth": relative(ground_truth_path),
                "split": "development_only_pending_seeded_dataset",
            }
        )
        print(f"EXPORTED={input_path}")
    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"MANIFEST={manifest_path}")


if __name__ == "__main__":
    main()
