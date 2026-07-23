"""Rebuild masks and object statistics from saved RGB/depth without relaunching Isaac Sim."""

from pathlib import Path

import numpy as np
from PIL import Image

from observation_capture import save_capture


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "observations"

for pose_name in ("left", "center", "right"):
    pose_dir = OUTPUT_ROOT / pose_name
    rgb = np.asarray(Image.open(pose_dir / "rgb.png").convert("RGBA"))
    depth = np.load(pose_dir / "depth_m.npy")
    save_capture(OUTPUT_ROOT, pose_name, rgb, depth)
    print(f"REPROCESSED={pose_name}")
