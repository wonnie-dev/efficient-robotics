#!/usr/bin/env python3
"""Export seed-14 target/occluder centers from learned masks and RGB-D."""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from rgbd_target_localization import estimate_mask_center

ROOT = Path(__file__).resolve().parents[1]
OBS = ROOT / "outputs/live_pipeline/paper_test_actual_multiview_seed014/observations/close_high"
PERCEPTION = ROOT / "outputs/perception_grounding_pilot/paper_test_actual_multiview_seed014"
SAMPLE = "seed014_close_high"
OUTPUT = ROOT / "outputs/live_pipeline/paper_test_actual_multiview_seed014/qwen_selected_rgbd_localization.json"


def load(path):
    return json.loads(Path(path).read_text())


ranking = load(PERCEPTION / "grounded_sam2_qwen_rankings" / SAMPLE / "result.json")
model_input = load(PERCEPTION / "grounded_sam2_qwen_inputs" / SAMPLE / "input.json")
segmentations = load(PERCEPTION / "grounded_sam2" / SAMPLE / "segmentations.json")
selected = next(item for item in model_input["candidates"] if item["candidate_id"] == ranking["selected_candidate_id"])
orange = max(
    (item for item in segmentations["annotations"] if item["label"] == "orange object"),
    key=lambda item: float(item["score"]),
)
depth = np.load(OBS / "depth_m.npy")
calibration = load(OBS / "camera_calibration.json")
target_mask = np.asarray(Image.open(ROOT / selected["mask_path"]).convert("L")) > 0
orange_mask = np.asarray(Image.open(PERCEPTION / "grounded_sam2" / SAMPLE / orange["mask_path"]).convert("L")) > 0
target = estimate_mask_center(depth, target_mask, calibration, label="qwen_selected_target")
occluder = estimate_mask_center(depth, orange_mask, calibration, label="grounded_sam2_orange_occluder")
result = {
    "schema_version": "qwen-selected-rgbd-localization-v1",
    "view": "close_high",
    "selection": {"selected_candidate_id": ranking["selected_candidate_id"], "ranking_result": str((PERCEPTION / "grounded_sam2_qwen_rankings" / SAMPLE / "result.json").relative_to(ROOT))},
    "estimates": {"target_red": target, "selected_target": target, "occluder_orange": occluder},
    "simulator_ground_truth_used_for_estimate": False,
    "mask_sources": {"target": selected["mask_path"], "occluder": str((PERCEPTION / "grounded_sam2" / SAMPLE / orange["mask_path"]).relative_to(ROOT))},
    "training_performed": False,
}
OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"output": str(OUTPUT), "target_center_world_m": target["center_world_m"], "occluder_center_world_m": occluder["center_world_m"]}, indent=2))
