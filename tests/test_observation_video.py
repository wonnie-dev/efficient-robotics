"""Tests for the headless observation-video manifest."""

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_observation_video import (  # noqa: E402
    build_concat_manifest,
    build_frame_sequence_video,
    build_observation_video,
)


class ObservationVideoTests(unittest.TestCase):
    def test_manifest_preserves_frame_order_and_duration(self) -> None:
        frames = [Path("left.png"), Path("center.png"), Path("right.png")]
        manifest = build_concat_manifest(frames, 1.25)
        self.assertLess(manifest.index("left.png"), manifest.index("center.png"))
        self.assertLess(manifest.index("center.png"), manifest.index("right.png"))
        self.assertEqual(manifest.count("duration 1.250000"), 3)
        self.assertEqual(manifest.count("right.png"), 2)

    def test_missing_capture_is_rejected_before_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(FileNotFoundError):
                build_observation_video(
                    output_root=root,
                    output_path=root / "summary.mp4",
                )

    def test_overview_frame_name_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(FileNotFoundError) as context:
                build_observation_video(
                    output_root=root,
                    output_path=root / "overview.mp4",
                    frame_name="overview_rgb.png",
                )
            self.assertIn("overview_rgb.png", str(context.exception))

    def test_nested_frame_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(ValueError):
                build_observation_video(
                    output_root=root,
                    output_path=root / "summary.mp4",
                    frame_name="../rgb.png",
                )

    def test_invalid_crf_is_rejected_before_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(ValueError):
                build_observation_video(
                    output_root=root,
                    output_path=root / "summary.mp4",
                    crf=52,
                )

    def test_invalid_preset_is_rejected_before_ffmpeg(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(ValueError):
                build_observation_video(
                    output_root=root,
                    output_path=root / "summary.mp4",
                    preset="not-a-preset",
                )

    def test_empty_motion_sequence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(ValueError):
                build_frame_sequence_video([], root / "motion.mp4")

    def test_missing_motion_frame_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(FileNotFoundError):
                build_frame_sequence_video(
                    [root / "missing.png"],
                    root / "motion.mp4",
                )


if __name__ == "__main__":
    unittest.main()
