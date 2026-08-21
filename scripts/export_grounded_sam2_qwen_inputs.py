"""Export anonymous Qwen candidate inputs from Grounded-SAM2 red proposals.

This exporter reads only predicted masks and RGB. It never reads simulator
instance IDs, semantic labels, depth, or ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/perception/grounding_pilot_seed0_2.json"


def resolve_path(value: str | Path) -> Path:
    """Resolve an artifact path against the repository root."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    """Serialize a repository path with platform-independent separators."""
    return str(path.resolve().relative_to(ROOT)).replace("\\", "/")


def sha256(path: Path) -> str:
    """Return a content digest for cache provenance checks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clipped_box(box: list[float], width: int, height: int) -> list[int]:
    """Round and clip a proposal box to valid image coordinates."""
    x0, y0, x1, y1 = [int(round(value)) for value in box]
    x0 = max(0, min(width - 1, x0))
    x1 = max(0, min(width - 1, x1))
    y0 = max(0, min(height - 1, y0))
    y1 = max(0, min(height - 1, y1))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid proposal box: {box}")
    return [x0, y0, x1, y1]


def save_candidate_assets(
    rgb: np.ndarray,
    mask: np.ndarray,
    bbox: list[int],
    sample_dir: Path,
    candidate_id: str,
) -> tuple[Path, Path, Path]:
    """Save the crop, binary mask, and context view for one candidate."""
    mask_path = sample_dir / f"{candidate_id}_mask.png"
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
    x0, y0, x1, y1 = bbox
    crop = rgb[y0 : y1 + 1, x0 : x1 + 1].copy()
    # Preserve RGB attributes inside the detector box. A class-conditioned
    # mask can legitimately omit a contrasting logo or printed label, and
    # blacking out non-mask pixels would remove exactly the semantic evidence
    # Qwen must inspect. The predicted mask remains a separate asset for 3D
    # localization and quantitative evaluation.
    crop_path = sample_dir / f"{candidate_id}_crop.png"
    Image.fromarray(crop, mode="RGB").save(crop_path)

    object_width = x1 - x0 + 1
    object_height = y1 - y0 + 1
    padding = max(40, 2 * max(object_width, object_height))
    context_x0 = max(0, x0 - padding)
    context_y0 = max(0, y0 - padding)
    context_x1 = min(rgb.shape[1] - 1, x1 + padding)
    context_y1 = min(rgb.shape[0] - 1, y1 + padding)
    context = Image.fromarray(
        rgb[context_y0 : context_y1 + 1, context_x0 : context_x1 + 1],
        mode="RGB",
    )
    ImageDraw.Draw(context).rectangle(
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
    mask: np.ndarray,
    sample_dir: Path,
    reference_id: str,
) -> tuple[Path, Path]:
    """Save the reference mask and its RGB overlay."""
    mask_path = sample_dir / f"{reference_id}_mask.png"
    Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(mask_path)
    overlay = rgb.copy()
    cyan = np.asarray([0, 255, 255], dtype=np.float32)
    overlay[mask] = np.clip(
        0.45 * overlay[mask].astype(np.float32) + 0.55 * cyan,
        0,
        255,
    ).astype(np.uint8)
    overlay_path = sample_dir / f"{reference_id}_overlay.png"
    Image.fromarray(overlay, mode="RGB").save(overlay_path)
    return mask_path, overlay_path


def main() -> None:
    """Build anonymous candidate inputs from detector and segmenter outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    pilot_root = resolve_path(config["output_root"])
    export_root = pilot_root / "grounded_sam2_qwen_inputs"
    manifest = {
        "schema_version": "grounded-sam2-qwen-input-manifest-v1",
        "samples": [],
        "simulator_ground_truth_used": False,
        "training_performed": False,
    }
    candidate_concept = config["task"].get(
        "qwen_candidate_concept", "red object"
    )
    reference_concept = config["task"].get(
        "qwen_reference_concept", "open container"
    )
    for sample in config["samples"]:
        sample_id = sample["sample_id"]
        observation_dir = resolve_path(sample["observation_dir"])
        source_root = pilot_root / "grounded_sam2" / sample_id
        segmentations = json.loads(
            (source_root / "segmentations.json").read_text(encoding="utf-8")
        )
        source_rgb_path = observation_dir / "rgb.png"
        source_rgb_sha256 = sha256(source_rgb_path)
        if (
            segmentations.get("image_path") != str(source_rgb_path)
            or segmentations.get("image_sha256") != source_rgb_sha256
        ):
            raise RuntimeError(
                f"Segmentation cache does not match current RGB: {sample_id}"
            )
        candidate_proposals = [
            annotation
            for annotation in segmentations["annotations"]
            if annotation["label"] == candidate_concept
        ]
        minimum_candidates = int(
            config["task"].get("minimum_candidate_proposals", 2)
        )
        if minimum_candidates < 1:
            raise ValueError("minimum_candidate_proposals must be positive")
        if len(candidate_proposals) < minimum_candidates:
            raise ValueError(
                f"{sample_id} has fewer than {minimum_candidates} "
                f"{candidate_concept!r} proposals: {len(candidate_proposals)}"
            )
        reference_proposals = [
            annotation
            for annotation in segmentations["annotations"]
            if annotation["label"] == reference_concept
        ]
        if not reference_proposals:
            raise ValueError(
                f"{sample_id} has no {reference_concept!r} reference proposal"
            )
        sample_dir = export_root / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)
        rgb = np.asarray(
            Image.open(source_rgb_path).convert("RGB")
        ).copy()
        rgb_path = sample_dir / "rgb.png"
        Image.fromarray(rgb, mode="RGB").save(rgb_path)

        candidates = []
        for index, proposal in enumerate(candidate_proposals, start=1):
            candidate_id = f"candidate_{index:03d}"
            mask = np.asarray(
                Image.open(source_root / proposal["mask_path"]).convert("L")
            ) > 0
            bbox = clipped_box(
                proposal["bbox_xyxy_pixels"], rgb.shape[1], rgb.shape[0]
            )
            crop_path, mask_path, context_path = save_candidate_assets(
                rgb, mask, bbox, sample_dir, candidate_id
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "bbox_xyxy": bbox,
                    "crop_path": relative(crop_path),
                    "mask_path": relative(mask_path),
                    "context_path": relative(context_path),
                    "proposal_source": "GroundingDINO-Base + SAM2.1-Large",
                    "proposal_detection_id": proposal["detection_id"],
                    "proposal_score": proposal["score"],
                }
            )
        # Open containers/baskets are frequently split into several slat or
        # wall proposals. Select the largest predicted reference region
        # without consulting simulator masks; the highest confidence proposal
        # is often a single interior object rather than the full cavity.
        reference_proposal = max(
            reference_proposals,
            key=lambda annotation: (
                annotation["bbox_xyxy_pixels"][2]
                - annotation["bbox_xyxy_pixels"][0]
            )
            * (
                annotation["bbox_xyxy_pixels"][3]
                - annotation["bbox_xyxy_pixels"][1]
            ),
        )
        reference_mask = np.asarray(
            Image.open(
                source_root / reference_proposal["mask_path"]
            ).convert("L")
        ) > 0
        reference_mask_path, reference_overlay_path = save_reference_assets(
            rgb,
            reference_mask,
            sample_dir,
            "container_001",
        )
        reference_bbox = clipped_box(
            reference_proposal["bbox_xyxy_pixels"],
            rgb.shape[1],
            rgb.shape[0],
        )
        membership_label_space = config["task"].get(
            "membership_label_space",
            ["inside", "outside", "unknown"],
        )
        independent_relation_label_spaces = config["task"].get(
            "independent_relation_label_spaces",
            {
                "behind": ["yes", "no", "unknown"],
                "occluded_by": ["yes", "no", "unknown"],
            },
        )
        factorized_relations = bool(
            config["task"].get("factorized_relations", False)
        )
        if factorized_relations:
            relation_queries = []
            for candidate in candidates:
                source_id = candidate["candidate_id"]
                relation_queries.append(
                    {
                        "query_id": f"{source_id}_membership",
                        "source_id": source_id,
                        "target_id": "container_001",
                        "relation_type": "membership",
                        "label_space": membership_label_space,
                    }
                )
                relation_queries.extend(
                    {
                        "query_id": f"{source_id}_{relation_type}",
                        "source_id": source_id,
                        "target_id": "container_001",
                        "relation_type": relation_type,
                        "label_space": labels,
                    }
                    for relation_type, labels in (
                        independent_relation_label_spaces.items()
                    )
                )
        else:
            relation_label_space = config["task"].get(
                "relation_label_space",
                ["inside", "outside", "behind", "unknown"],
            )
            relation_queries = [
                {
                    "query_id": f"{candidate['candidate_id']}_to_container",
                    "source_id": candidate["candidate_id"],
                    "target_id": "container_001",
                    "relation_type": "legacy_mutually_exclusive_relation",
                    "label_space": relation_label_space,
                }
                for candidate in candidates
            ]
        model_input = {
            "schema_version": "vlm-input-v1",
            "sample_id": sample_id,
            "episode_id": sample_id.rsplit("_", 1)[0],
            "view_id": sample_id.split("_", 1)[1],
            "instruction": config["task"]["instruction"],
            "target_description": config["task"].get(
                "target_description", config["task"]["instruction"]
            ),
            "image": {
                "rgb_path": relative(rgb_path),
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
            },
            "candidates": candidates,
            "reference_entities": [
                {
                    "reference_id": "container_001",
                    "bbox_xyxy": reference_bbox,
                    "description": (
                        f"the {reference_concept} visible in the full scene"
                    ),
                    "mask_path": relative(reference_mask_path),
                    "overlay_path": relative(reference_overlay_path),
                    "proposal_source": "GroundingDINO-Base + SAM2.1-Large",
                    "proposal_score": reference_proposal["score"],
                }
            ],
            "relation_queries": relation_queries,
            "provenance": {
                "source_rgb_path": str(source_rgb_path),
                "source_rgb_sha256": source_rgb_sha256,
                "simulator_masks_used": False,
                "candidate_semantic_labels_exposed_to_qwen": False,
                "reference_mask_source": (
                    "GroundingDINO-Base + SAM2.1-Large prediction"
                ),
                "factorized_relations": factorized_relations,
            },
        }
        input_path = sample_dir / "input.json"
        input_path.write_text(
            json.dumps(model_input, indent=2) + "\n", encoding="utf-8"
        )
        manifest["samples"].append(
            {
                "sample_id": sample_id,
                "input_path": relative(input_path),
                "input_sha256": sha256(input_path),
                "source_rgb_sha256": source_rgb_sha256,
                "candidate_count": len(candidates),
            }
        )
    manifest_path = export_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
