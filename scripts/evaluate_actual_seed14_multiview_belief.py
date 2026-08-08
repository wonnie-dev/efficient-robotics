#!/usr/bin/env python3
"""Fuse actual-motion seed-14 Qwen evidence after RGB-D candidate tracking."""

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from rgbd_target_localization import estimate_mask_center

ROOT = Path(__file__).resolve().parents[1]
OBS = ROOT / "outputs/live_pipeline/paper_test_actual_multiview_seed014/observations"
PERCEPTION = ROOT / "outputs/perception_grounding_pilot/paper_test_actual_multiview_seed014"
OUT = ROOT / "outputs/paper_test_actual_multiview_seed014/belief_fusion.json"
VIEWS = ("center", "right", "close_high")
MAX_DISTANCE_M = 0.15


def load(path):
    return json.loads(Path(path).read_text())


def overlap(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def candidates(view):
    sample = f"seed014_{view}"
    inp = load(PERCEPTION / "grounded_sam2_qwen_inputs" / sample / "input.json")
    rank = load(PERCEPTION / "grounded_sam2_qwen_rankings" / sample / "result.json")
    observation = OBS / view
    depth = np.load(observation / "depth_m.npy")
    calibration = load(observation / "camera_calibration.json")
    ids = np.load(observation / "instance_ids.npy")
    labels = load(observation / "instance_labels.json")
    target_id = int(next(k for k, v in labels.items() if v["class"] == "target_red"))
    target = ids == target_id
    logits = dict(zip(rank["candidate_ids"], rank["raw_match_logits"]))
    result = []
    for item in inp["candidates"]:
        candidate_mask = np.asarray(Image.open(ROOT / item["mask_path"]).convert("L")) > 0
        try:
            center = estimate_mask_center(depth, candidate_mask, calibration, label=item["candidate_id"])["center_world_m"]
        except ValueError:
            center = None
        result.append({"candidate_id": item["candidate_id"], "logit": float(logits[item["candidate_id"]]), "center_world_m": center, "target_iou_posthoc": overlap(candidate_mask, target)})
    return result


def dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


tracks = []
trace = []
for view in VIEWS:
    for item in candidates(view):
        matches = []
        if item["center_world_m"] is not None:
            for i, track in enumerate(tracks):
                if view not in track["observations"] and track["center_world_m"] is not None:
                    matches.append((dist(item["center_world_m"], track["center_world_m"]), i))
        if matches and min(matches)[0] <= MAX_DISTANCE_M:
            separation, index = min(matches)
            track = tracks[index]
            item["track_match_distance_m"] = separation
            track["observations"][view] = item
            track["center_world_m"] = np.mean([x["center_world_m"] for x in track["observations"].values() if x["center_world_m"] is not None], axis=0).tolist()
        else:
            tracks.append({"track_id": f"track_{len(tracks):03d}", "center_world_m": item["center_world_m"], "observations": {view: item}})
    scores = {t["track_id"]: sum(x["logit"] for x in t["observations"].values()) / len(t["observations"]) for t in tracks}
    selected = max(scores, key=scores.get)
    selected_track = next(t for t in tracks if t["track_id"] == selected)
    trace.append({"after_view": view, "selected_track": selected, "selected_correct_posthoc": max(x["target_iou_posthoc"] for x in selected_track["observations"].values()) >= 0.5, "mean_logits": scores})

result = {
    "schema_version": "actual-seed14-multiview-belief-fusion-v1",
    "views": list(VIEWS),
    "tracking": {"method": "nearest_rgbd_world_center", "maximum_distance_m": MAX_DISTANCE_M, "simulator_ground_truth_used": False},
    "belief_update": "running_mean_of_uncalibrated_qwen_match_logits",
    "trace": trace,
    "tracks": tracks,
    "simulator_ground_truth_used_after_selection_only": True,
    "training_performed": False,
    "calibration_performed": False,
    "valid_for_final_evaluation": False,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(trace, indent=2))
