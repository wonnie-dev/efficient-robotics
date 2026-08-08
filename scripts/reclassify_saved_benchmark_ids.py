"""Reclassify saved benchmark color-ID passes without rerunning Isaac Sim."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from observation_capture import (
    BENCHMARK_ID_CHROMATICITY_DISTANCE_THRESHOLD,
    BENCHMARK_SEMANTIC_OBJECTS,
    _colorize_instance_ids,
    _json_safe,
    _object_statistics,
    classify_benchmark_color_pass,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("observation_root", type=Path)
    parser.add_argument(
        "--views",
        nargs="+",
        default=("center", "right"),
    )
    args = parser.parse_args()

    summary = {
        "schema_version": "saved-benchmark-id-reclassification-v1",
        "chromaticity_distance_threshold": (
            BENCHMARK_ID_CHROMATICITY_DISTANCE_THRESHOLD
        ),
        "views": [],
        "inference_outputs_modified": False,
    }
    for view in args.views:
        view_root = args.observation_root / view
        color_path = view_root / "instance_color_pass.png"
        depth_path = view_root / "depth_m.npy"
        if not color_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(f"Missing saved ID inputs for {view_root}")
        color_pass = np.asarray(Image.open(color_path).convert("RGB"))
        depth = np.load(depth_path)
        instance_ids, labels = classify_benchmark_color_pass(color_pass)
        np.save(view_root / "instance_ids.npy", instance_ids)
        Image.fromarray(
            _colorize_instance_ids(instance_ids), "RGB"
        ).save(view_root / "instance_segmentation.png")
        (view_root / "instance_labels.json").write_text(
            json.dumps(_json_safe(labels), indent=2) + "\n",
            encoding="utf-8",
        )
        statistics = _object_statistics(
            instance_ids,
            labels,
            depth,
            expected_objects=BENCHMARK_SEMANTIC_OBJECTS.values(),
        )
        (view_root / "objects.json").write_text(
            json.dumps(statistics, indent=2) + "\n",
            encoding="utf-8",
        )
        summary["views"].append(
            {
                "view_id": view,
                "target_red_pixels": statistics["target_red"][
                    "pixel_count"
                ],
                "rear_red_candidate_pixels": statistics[
                    "rear_red_candidate"
                ]["pixel_count"],
                "container_pixels": statistics["container"]["pixel_count"],
            }
        )
    destination = (
        args.observation_root.parent / "id_reclassification.json"
    )
    destination.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
