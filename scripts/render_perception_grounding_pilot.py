"""Render prediction-only diagnostic panels for the grounding pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/perception/grounding_pilot_seed0_2.json"
COLORS = (
    (0, 255, 255),
    (255, 0, 255),
    (255, 255, 0),
    (0, 255, 0),
    (255, 128, 0),
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16
        )
    except OSError:
        return ImageFont.load_default()


def title_panel(image: Image.Image, title: str) -> Image.Image:
    result = Image.new("RGB", (image.width, image.height + 30), "black")
    result.paste(image, (0, 30))
    ImageDraw.Draw(result).text((8, 6), title, fill="white", font=font())
    return result


def qwen_panel(image: Image.Image, result: dict) -> Image.Image:
    panel = image.copy()
    draw = ImageDraw.Draw(panel)
    box = result.get("bbox_xyxy_pixels")
    if box is None:
        draw.text((8, 8), "NO BOX", fill=(255, 64, 64), font=font())
    else:
        draw.rectangle(tuple(box), outline=(255, 0, 255), width=4)
        draw.text(
            (max(0, box[0]), max(0, box[1] - 22)),
            "Qwen target",
            fill=(255, 0, 255),
            font=font(),
        )
    return title_panel(panel, "Qwen3-VL direct target box")


def grounded_panel(
    image: Image.Image,
    result: dict,
    sample_root: Path,
    candidate_concept: str,
) -> Image.Image:
    base = np.asarray(image).astype(np.float32)
    annotations = [
        annotation
        for annotation in result["annotations"]
        if annotation["label"] == candidate_concept
    ]
    for index, annotation in enumerate(annotations):
        mask = np.asarray(
            Image.open(sample_root / annotation["mask_path"]).convert("L")
        ) > 0
        color = np.asarray(COLORS[index % len(COLORS)], dtype=np.float32)
        base[mask] = 0.6 * base[mask] + 0.4 * color
    panel = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(panel)
    for index, annotation in enumerate(annotations):
        color = COLORS[index % len(COLORS)]
        box = annotation["bbox_xyxy_pixels"]
        draw.rectangle(tuple(box), outline=color, width=3)
        draw.text(
            (max(0, box[0]), max(0, box[1] - 18)),
            f"{annotation['score']:.2f}",
            fill=color,
            font=font(),
        )
    return title_panel(
        panel,
        f"GroundingDINO + SAM2 {candidate_concept} proposals "
        f"({len(annotations)})",
    )


def selected_mask_panel(
    image: Image.Image, ranking: dict, input_payload: dict
) -> Image.Image:
    candidate = next(
        candidate
        for candidate in input_payload["candidates"]
        if candidate["candidate_id"] == ranking["selected_candidate_id"]
    )
    mask = np.asarray(
        Image.open(resolve_path(candidate["mask_path"])).convert("L")
    ) > 0
    base = np.asarray(image).astype(np.float32)
    cyan = np.asarray((0, 255, 255), dtype=np.float32)
    base[mask] = 0.55 * base[mask] + 0.45 * cyan
    panel = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(panel)
    draw.rectangle(tuple(candidate["bbox_xyxy"]), outline=(0, 255, 255), width=4)
    draw.text(
        (
            max(0, candidate["bbox_xyxy"][0]),
            max(0, candidate["bbox_xyxy"][1] - 22),
        ),
        ranking["selected_candidate_id"],
        fill=(0, 255, 255),
        font=font(),
    )
    return title_panel(panel, "Qwen-selected anonymous proposal")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_root = resolve_path(config["output_root"])
    visualization_root = output_root / "visualizations"
    visualization_root.mkdir(parents=True, exist_ok=True)

    paths = []
    candidate_concept = config["task"].get(
        "qwen_candidate_concept", "red object"
    )
    for sample in config["samples"]:
        sample_id = sample["sample_id"]
        observation_dir = resolve_path(sample["observation_dir"])
        image = Image.open(observation_dir / "rgb.png").convert("RGB")
        qwen_result_path = (
            output_root / "qwen_direct" / sample_id / "result.json"
        )
        qwen_result = (
            json.loads(qwen_result_path.read_text(encoding="utf-8"))
            if qwen_result_path.is_file()
            else None
        )
        grounded_root = output_root / "grounded_sam2" / sample_id
        grounded_result = json.loads(
            (grounded_root / "segmentations.json").read_text(encoding="utf-8")
        )
        ranking = json.loads(
            (
                output_root
                / "grounded_sam2_qwen_rankings"
                / sample_id
                / "result.json"
            ).read_text(encoding="utf-8")
        )
        ranking_input = json.loads(
            Path(ranking["input_path"]).read_text(encoding="utf-8")
        )
        panels = [title_panel(image.copy(), "Input RGB")]
        if qwen_result is not None:
            panels.append(qwen_panel(image, qwen_result))
        panels.extend(
            [
                grounded_panel(
                image,
                grounded_result,
                grounded_root,
                candidate_concept,
                ),
                selected_mask_panel(image, ranking, ranking_input),
            ]
        )
        canvas = Image.new(
            "RGB",
            (sum(panel.width for panel in panels), max(panel.height for panel in panels)),
            "black",
        )
        offset = 0
        for panel in panels:
            canvas.paste(panel, (offset, 0))
            offset += panel.width
        destination = visualization_root / f"{sample_id}.png"
        canvas.save(destination)
        paths.append(str(destination))

    manifest = {
        "schema_version": "perception-grounding-visualization-manifest-v1",
        "images": paths,
        "simulator_ground_truth_used": False,
        "valid_for_final_evaluation": False,
    }
    (visualization_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
