#!/usr/bin/env python3
"""Evaluate RGB-D candidate tracking and cached multi-view Qwen fusion.

Simulator instance masks are read only after a policy has selected a track.
They are used solely to score the selected track, never for tracking/fusion.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from rgbd_target_localization import estimate_mask_center


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/paper_test_multiview_rgbd_fusion_seed014_030"
VIEWS = ("center", "close_high", "right")
TRACK_DISTANCE_M = 0.15


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paths(seed: int, view: str) -> tuple[Path, Path, Path]:
    if view == "right":
        observation = ROOT / f"outputs/seeded_pilot/benchmark_seed{seed:03d}/observations/right"
        root = ROOT / "outputs/paper_test_right_perception_seed014_030"
        sample = f"seed{seed:03d}_right"
    else:
        observation = ROOT / f"outputs/seeded_pilot/benchmark_seed{seed:03d}/observations/{view}"
        root = ROOT / "outputs/paper_test_center_close_perception_seed014_030"
        sample = f"seed{seed:03d}_{view}"
    return (
        observation,
        root / "grounded_sam2_qwen_inputs" / sample / "input.json",
        root / "grounded_sam2_qwen_rankings" / sample / "result.json",
    )


def mask(path: str) -> np.ndarray:
    return np.asarray(Image.open(ROOT / path).convert("L")) > 0


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def candidate_rows(seed: int, view: str) -> tuple[list[dict], np.ndarray]:
    observation, input_path, ranking_path = paths(seed, view)
    model_input = load_json(input_path)
    ranking = load_json(ranking_path)
    depth = np.load(observation / "depth_m.npy")
    calibration = load_json(observation / "camera_calibration.json")
    instance_ids = np.load(observation / "instance_ids.npy")
    labels = load_json(observation / "instance_labels.json")
    target_id = int(next(key for key, value in labels.items() if value["class"] == "target_red"))
    target_mask = instance_ids == target_id
    logits = dict(zip(ranking["candidate_ids"], ranking["raw_match_logits"]))
    rows = []
    for candidate in model_input["candidates"]:
        candidate_mask = mask(candidate["mask_path"])
        try:
            center = estimate_mask_center(depth, candidate_mask, calibration, label=candidate["candidate_id"])["center_world_m"]
        except ValueError:
            center = None
        rows.append({
            "candidate_id": candidate["candidate_id"],
            "logit": float(logits[candidate["candidate_id"]]),
            "center_world_m": center,
            "target_mask_iou_posthoc": iou(candidate_mask, target_mask),
        })
    return rows, target_mask


def distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def assign_tracks(per_view: dict[str, list[dict]]) -> list[dict]:
    tracks: list[dict] = []
    for view in VIEWS:
        for candidate in per_view[view]:
            best = None
            if candidate["center_world_m"] is not None:
                choices = []
                for index, track in enumerate(tracks):
                    if view in track["observations"] or track["center_world_m"] is None:
                        continue
                    choices.append((distance(candidate["center_world_m"], track["center_world_m"]), index))
                if choices:
                    separation, index = min(choices)
                    if separation <= TRACK_DISTANCE_M:
                        best = (separation, index)
            if best is None:
                tracks.append({"track_id": f"track_{len(tracks):03d}", "center_world_m": candidate["center_world_m"], "observations": {view: candidate}})
            else:
                separation, index = best
                track = tracks[index]
                track["observations"][view] = candidate
                track["center_world_m"] = np.mean(
                    [item["center_world_m"] for item in track["observations"].values() if item["center_world_m"] is not None], axis=0
                ).tolist()
                candidate["track_match_distance_m"] = separation
    return tracks


def select_by_view(tracks: list[dict], view: str) -> dict:
    eligible = [t for t in tracks if view in t["observations"]]
    return max(eligible, key=lambda t: t["observations"][view]["logit"])


def select_fused(tracks: list[dict]) -> dict:
    return max(tracks, key=lambda t: sum(o["logit"] for o in t["observations"].values()) / len(t["observations"]))


def correct(track: dict) -> bool:
    return max(o["target_mask_iou_posthoc"] for o in track["observations"].values()) >= 0.5


def main() -> None:
    episodes = []
    methods = {"center_only": 0, "close_high_only": 0, "right_only": 0, "last_view_overwrite": 0, "rgbd_multiview_mean_logit": 0}
    for seed in range(14, 31):
        per_view = {view: candidate_rows(seed, view)[0] for view in VIEWS}
        tracks = assign_tracks(per_view)
        selected = {
            "center_only": select_by_view(tracks, "center"),
            "close_high_only": select_by_view(tracks, "close_high"),
            "right_only": select_by_view(tracks, "right"),
            "last_view_overwrite": select_by_view(tracks, "right"),
            "rgbd_multiview_mean_logit": select_fused(tracks),
        }
        results = {}
        for name, track in selected.items():
            outcome = correct(track)
            methods[name] += int(outcome)
            results[name] = {"selected_track": track["track_id"], "correct_posthoc": outcome}
        episodes.append({"seed": seed, "track_count": len(tracks), "tracks": tracks, "methods": results})
    summary = {
        "schema_version": "heldout-multiview-rgbd-fusion-evaluation-v1",
        "seed_range": [14, 30],
        "episode_count": len(episodes),
        "tracking": {"method": "nearest_rgbd_world_center", "maximum_distance_m": TRACK_DISTANCE_M, "simulator_ground_truth_used": False},
        "method_results": {name: {"correct": count, "total": len(episodes), "accuracy": count / len(episodes)} for name, count in methods.items()},
        "episodes": episodes,
        "evaluation": {"target_mask_iou_threshold": 0.5, "simulator_ground_truth_used_after_selection_only": True},
        "training_performed": False,
        "calibration_performed": False,
        "valid_for_final_evaluation": False,
        "limitations": [
            "Cached views were fused offline; no new robot action was executed.",
            "Raw Qwen logits are uncalibrated.",
            "This is a held-out pilot, not a final statistical claim.",
        ],
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["method_results"], indent=2))


if __name__ == "__main__":
    main()
