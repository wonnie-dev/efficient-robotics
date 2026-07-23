"""Reclassify saved benchmark ID passes without relaunching Isaac Sim."""

from pathlib import Path

import numpy as np
from PIL import Image

from observation_capture import classify_benchmark_color_pass, save_capture


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/benchmark_observations"


def main() -> None:
    for view in ("left", "center", "right"):
        view_dir = OUTPUT_ROOT / view
        rgb = np.asarray(Image.open(view_dir / "rgb.png").convert("RGBA"))
        depth = np.load(view_dir / "depth_m.npy")
        color_pass = np.asarray(
            Image.open(view_dir / "instance_color_pass.png").convert("RGB")
        )
        instance_override = (*classify_benchmark_color_pass(color_pass), color_pass)
        save_capture(
            OUTPUT_ROOT,
            view,
            rgb,
            depth,
            instance_override=instance_override,
        )
        print(f"REPROCESSED_BENCHMARK_ID_PASS={view}")


if __name__ == "__main__":
    main()
