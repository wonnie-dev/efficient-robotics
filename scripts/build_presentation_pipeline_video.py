"""Build a meeting-ready MP4 from verified stored pilot artifacts."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from build_observation_video import build_frame_sequence_video


ROOT = Path(__file__).resolve().parents[1]
WIDTH = 1920
HEIGHT = 1080
FPS = 5


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            size,
        )
    except OSError:
        return ImageFont.load_default()


def base_slide() -> Image.Image:
    return Image.new("RGB", (WIDTH, HEIGHT), (14, 20, 28))


def draw_centered(draw: ImageDraw.ImageDraw, text: str, y: int, size: int) -> None:
    selected_font = font(size)
    box = draw.textbbox((0, 0), text, font=selected_font)
    draw.text(
        ((WIDTH - (box[2] - box[0])) / 2, y),
        text,
        fill=(245, 248, 252),
        font=selected_font,
    )


def add_held_frame(frames: list[Path], path: Path, seconds: float) -> None:
    frames.extend([path] * int(round(seconds * FPS)))


def paste_panel(
    canvas: Image.Image,
    image_path: Path,
    box: tuple[int, int, int, int],
    title: str,
) -> None:
    image = Image.open(image_path).convert("RGB")
    panel_width = box[2] - box[0]
    panel_height = box[3] - box[1]
    fitted = ImageOps.contain(image, (panel_width, panel_height))
    x = box[0] + (panel_width - fitted.width) // 2
    y = box[1] + (panel_height - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(box, outline=(94, 210, 255), width=4)
    draw.text(
        (box[0] + 12, box[1] + 10),
        title,
        fill=(255, 255, 255),
        stroke_width=3,
        stroke_fill=(0, 0, 0),
        font=font(34),
    )


def main() -> None:
    source_root = ROOT / "outputs" / "candidate_view_demo" / "benchmark_seed000"
    motion_video = source_root / "close_high_candidate_demo.mp4"
    center_rgb = source_root / "observations" / "center" / "rgb.png"
    close_rgb = source_root / "observations" / "close_high" / "rgb.png"
    relation_overlay = (
        source_root
        / "vlm_dataset_v5"
        / "samples"
        / "candidate_seed000_v5_close_high"
        / "container_001_overlay.png"
    )
    qwen_summary_path = source_root / "close_high_qwen_v5_summary.json"
    pilot_report_path = ROOT / "outputs" / "seeded_pilot" / "pilot_report.json"
    required = (
        motion_video,
        center_rgb,
        close_rgb,
        relation_overlay,
        qwen_summary_path,
        pilot_report_path,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing presentation source artifacts: {missing}")

    qwen = json.loads(qwen_summary_path.read_text(encoding="utf-8"))
    pilot = json.loads(pilot_report_path.read_text(encoding="utf-8"))
    output_root = ROOT / "outputs" / "presentation_demo"
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "active_view_pipeline_presentation.mp4"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")

    with tempfile.TemporaryDirectory(
        prefix="efficient_robotics_presentation_"
    ) as temp_text:
        temp_root = Path(temp_text)
        frames: list[Path] = []

        title = base_slide()
        title_draw = ImageDraw.Draw(title)
        draw_centered(title_draw, "Efficient Robotics: Active View Pipeline", 330, 64)
        draw_centered(
            title_draw,
            "RGB-D -> Qwen3-VL -> Reobserve -> Replan",
            440,
            46,
        )
        draw_centered(
            title_draw,
            "Single-GPU pilot | presentation evidence, not final evaluation",
            540,
            30,
        )
        title_path = temp_root / "title.png"
        title.save(title_path)
        add_held_frame(frames, title_path, 2.5)

        before = base_slide()
        paste_panel(before, center_rgb, (90, 180, 920, 850), "Initial center RGB")
        before_draw = ImageDraw.Draw(before)
        before_draw.text(
            (1010, 300),
            "Initial Qwen result",
            fill=(245, 248, 252),
            font=font(48),
        )
        before_draw.text(
            (1010, 410),
            "Selected: object_007 (distractor)",
            fill=(255, 154, 110),
            font=font(36),
        )
        before_draw.text(
            (1010, 500),
            "Policy: acquire a closer/high wrist view",
            fill=(94, 210, 255),
            font=font(34),
        )
        before_draw.text(
            (1010, 590),
            "No grasp is executed from the ambiguous view.",
            fill=(220, 226, 234),
            font=font(28),
        )
        before_path = temp_root / "before.png"
        before.save(before_path)
        add_held_frame(frames, before_path, 3.0)

        extracted_pattern = temp_root / "motion_%04d.png"
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(motion_video),
                "-vf",
                f"fps={FPS}",
                str(extracted_pattern),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"Could not decode motion video: {completed.stderr}")
        motion_frames = sorted(temp_root.glob("motion_*.png"))
        if not motion_frames:
            raise RuntimeError("No motion frames were decoded")
        frames.extend(motion_frames)

        after = base_slide()
        paste_panel(after, close_rgb, (60, 175, 880, 825), "New close/high RGB")
        paste_panel(
            after,
            relation_overlay,
            (1040, 175, 1860, 825),
            "Anonymous container overlay",
        )
        after_draw = ImageDraw.Draw(after)
        target_confidence = qwen["pilot_action"]["uncalibrated_target_confidence"]
        inside_confidence = qwen["pilot_action"]["uncalibrated_inside_confidence"]
        draw_centered(
            after_draw,
            f"Qwen: object_001={target_confidence:.3f}, inside={inside_confidence:.3f}",
            875,
            38,
        )
        after_path = temp_root / "after.png"
        after.save(after_path)
        add_held_frame(frames, after_path, 3.5)

        result = base_slide()
        result_draw = ImageDraw.Draw(result)
        draw_centered(result_draw, "Replanned action: GRASP", 245, 72)
        draw_centered(
            result_draw,
            "Action selected only - physical grasp execution is shown separately",
            360,
            34,
        )
        draw_centered(
            result_draw,
            (
                f"Pilot: {pilot['successful_episode_count']}/"
                f"{pilot['episode_count']} episodes completed"
            ),
            505,
            42,
        )
        draw_centered(
            result_draw,
            (
                f"Active reobservation used in "
                f"{pilot['active_view_episode_count']}/"
                f"{pilot['episode_count']} episodes"
            ),
            585,
            38,
        )
        draw_centered(
            result_draw,
            "Uncalibrated pipeline validation - not a paper accuracy claim",
            720,
            32,
        )
        result_path = temp_root / "result.png"
        result.save(result_path)
        add_held_frame(frames, result_path, 4.0)

        video = build_frame_sequence_video(
            frames,
            output_path,
            fps=FPS,
            crf=16,
            preset="slow",
            purpose="meeting_presentation_pipeline_validation_not_final_evaluation",
        )

    metadata = {
        **video,
        "schema_version": "presentation-pipeline-video-v1",
        "sources": [str(path) for path in required],
        "training_performed": False,
        "fresh_vlm_inference_performed": False,
        "grasp_execution_included": False,
        "valid_for_final_evaluation": False,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"PRESENTATION_VIDEO={output_path}")


if __name__ == "__main__":
    main()
