"""Build a compact MP4 summary from deterministic observation RGB captures."""

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


DEFAULT_POSES = ("left", "center", "right")


def _ffmpeg_concat_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "'\\''")


def build_concat_manifest(
    frame_paths: list[Path], hold_seconds: float
) -> str:
    if not frame_paths:
        raise ValueError("At least one RGB frame is required")
    if hold_seconds <= 0:
        raise ValueError("Frame hold duration must be positive")
    lines = []
    for frame_path in frame_paths:
        lines.append(f"file '{_ffmpeg_concat_path(frame_path)}'")
        lines.append(f"duration {hold_seconds:.6f}")
    lines.append(f"file '{_ffmpeg_concat_path(frame_paths[-1])}'")
    return "\n".join(lines) + "\n"


def build_observation_video(
    output_root: Path,
    output_path: Path,
    poses: tuple[str, ...] = DEFAULT_POSES,
    fps: int = 10,
    hold_seconds: float = 1.0,
    frame_name: str = "rgb.png",
    purpose: str = "headless_visual_verification_not_physics_timing_evidence",
    crf: int = 23,
    preset: str = "medium",
) -> dict:
    if fps <= 0:
        raise ValueError("Video FPS must be positive")
    if not 0 <= crf <= 51:
        raise ValueError("Video CRF must be between 0 and 51")
    if preset not in {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    }:
        raise ValueError(f"Unsupported x264 preset: {preset}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to build the observation MP4")

    if Path(frame_name).name != frame_name:
        raise ValueError("Frame name must be a plain filename")
    frame_paths = [output_root / pose / frame_name for pose in poses]
    missing = [str(path) for path in frame_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing observation RGB frames: {missing}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_concat_manifest(frame_paths, hold_seconds)
    with tempfile.TemporaryDirectory(prefix="efficient_robotics_video_") as temp_dir:
        manifest_path = Path(temp_dir) / "frames.txt"
        manifest_path.write_text(manifest, encoding="utf-8")
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest_path),
                "-vf",
                f"fps={fps},format=yuv420p",
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")
    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("ffmpeg did not produce a non-empty MP4")

    result = {
        "status": "completed",
        "output_path": str(output_path),
        "source_frames": [str(path) for path in frame_paths],
        "poses": list(poses),
        "fps": fps,
        "hold_seconds_per_pose": hold_seconds,
        "duration_seconds": len(poses) * hold_seconds,
        "encoder": "ffmpeg_libx264_cpu",
        "crf": crf,
        "preset": preset,
        "frame_name": frame_name,
        "purpose": purpose,
        "size_bytes": output_path.stat().st_size,
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def build_frame_sequence_video(
    frame_paths: list[Path],
    output_path: Path,
    fps: int = 20,
    crf: int = 18,
    preset: str = "medium",
    purpose: str = "continuous_headless_visual_demo",
) -> dict:
    """Encode an ordered list of already-rendered frames with CPU ffmpeg."""
    if fps <= 0:
        raise ValueError("Video FPS must be positive")
    if not 0 <= crf <= 51:
        raise ValueError("Video CRF must be between 0 and 51")
    if not frame_paths:
        raise ValueError("At least one video frame is required")
    missing = [str(path) for path in frame_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing video frames: {missing}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to build the motion MP4")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_concat_manifest(frame_paths, 1.0 / fps)
    with tempfile.TemporaryDirectory(
        prefix="efficient_robotics_motion_video_"
    ) as temp_dir:
        manifest_path = Path(temp_dir) / "frames.txt"
        manifest_path.write_text(manifest, encoding="utf-8")
        completed = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(manifest_path),
                "-vf",
                f"fps={fps},format=yuv420p",
                "-c:v",
                "libx264",
                "-crf",
                str(crf),
                "-preset",
                preset,
                "-movflags",
                "+faststart",
                str(output_path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")
    result = {
        "status": "completed",
        "output_path": str(output_path),
        "frame_count": len(frame_paths),
        "fps": fps,
        "duration_seconds": len(frame_paths) / fps,
        "encoder": "ffmpeg_libx264_cpu",
        "crf": crf,
        "preset": preset,
        "purpose": purpose,
        "size_bytes": output_path.stat().st_size,
    }
    output_path.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Defaults to OUTPUT_ROOT/minimal_observations.mp4",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--frame-name", default="rgb.png")
    parser.add_argument("--crf", type=int, default=23)
    parser.add_argument("--preset", default="medium")
    parser.add_argument(
        "--purpose",
        default="headless_visual_verification_not_physics_timing_evidence",
    )
    args = parser.parse_args()
    output_path = args.output or args.output_root / "minimal_observations.mp4"
    result = build_observation_video(
        output_root=args.output_root,
        output_path=output_path,
        fps=args.fps,
        hold_seconds=args.hold_seconds,
        frame_name=args.frame_name,
        purpose=args.purpose,
        crf=args.crf,
        preset=args.preset,
    )
    print(f"CAPTURE_VIDEO={result['output_path']}")


if __name__ == "__main__":
    main()
