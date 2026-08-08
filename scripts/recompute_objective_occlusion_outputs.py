"""Recompute stored objective occlusion JSON after metric-only revisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from observation_capture import objective_occlusion_measurement
from run_live_single_gpu_pipeline import ROOT, write_json_atomic


DEFAULT_ROOT = (
    ROOT
    / "outputs"
    / "live_pipeline"
    / "objective_occlusion_validation_seed156_164_gpu5"
)


def load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 0


def save_mask(path: Path, mask: np.ndarray) -> None:
    Image.fromarray(mask.astype(np.uint8) * 255, "L").save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, nargs="?", default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    allowed_root = (ROOT / "outputs").resolve()
    if not root.is_relative_to(allowed_root):
        raise ValueError(f"Root must be under {allowed_root}: {root}")
    result_paths = sorted(root.rglob("objective_occlusion.json"))
    if not result_paths:
        raise FileNotFoundError(
            f"No objective_occlusion.json files under {root}"
        )
    rows = []
    capture_measurements: dict[Path, dict[str, dict]] = {}
    for result_path in result_paths:
        observation_root = result_path.parent
        visible_path = observation_root / "target_visible_mask.png"
        raw_visible_path = (
            observation_root / "target_visible_mask_raw.png"
        )
        amodal_path = observation_root / "target_amodal_mask.png"
        if raw_visible_path.is_file():
            raw_visible_mask = load_mask(raw_visible_path)
        else:
            raw_visible_mask = load_mask(visible_path)
            save_mask(raw_visible_path, raw_visible_mask)
        amodal_mask = load_mask(amodal_path)
        supported_visible_mask = raw_visible_mask & amodal_mask
        save_mask(visible_path, supported_visible_mask)
        measurement = objective_occlusion_measurement(
            raw_visible_mask, amodal_mask
        )
        write_json_atomic(result_path, measurement)
        capture_root = observation_root.parent.parent
        capture_measurements.setdefault(capture_root, {})[
            observation_root.name
        ] = measurement
        rows.append(
            {
                "path": str(result_path),
                "severity": measurement["severity"],
                "occlusion_fraction": measurement["occlusion_fraction"],
                "out_of_amodal_id_spill_pixels": measurement[
                    "out_of_amodal_id_spill_pixels"
                ],
            }
        )
    for capture_root, measurements in capture_measurements.items():
        ground_truth_path = capture_root / "calibration_ground_truth.json"
        if not ground_truth_path.is_file():
            continue
        ground_truth = json.loads(
            ground_truth_path.read_text(encoding="utf-8")
        )
        ground_truth["objective_occlusion_ground_truth"] = measurements
        for view_id, measurement in measurements.items():
            observed = ground_truth.get(
                "observed_target_visibility", {}
            ).get(view_id)
            if observed is not None:
                observed["objective_occlusion"] = measurement
        write_json_atomic(ground_truth_path, ground_truth)
    write_json_atomic(
        root / "objective_occlusion_recompute.json",
        {
            "schema_version": "objective-occlusion-recompute-v1",
            "status": "completed",
            "result_count": len(rows),
            "metric_revision": (
                "clip temporary color-ID mask to target-only amodal support"
            ),
            "gpu_used": False,
            "rows": rows,
        },
    )
    print(
        "OBJECTIVE_OCCLUSION_RECOMPUTE="
        f"{root / 'objective_occlusion_recompute.json'}"
    )


if __name__ == "__main__":
    main()
