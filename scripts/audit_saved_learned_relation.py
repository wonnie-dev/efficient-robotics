#!/usr/bin/env python3
"""Audit one saved learned-perception result with RGB-D relation geometry.

This utility never reads simulator instance IDs or relation labels.  It uses
the learned candidate/reference masks, metric depth, and camera calibration
that were available to the perception pipeline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from run_hybrid_rgbd_relation_pilot import (
    load_json,
    load_mask,
    masked_world_points,
    predict_candidate,
    reference_geometry,
    resolve_path,
)
from run_live_single_gpu_pipeline import write_json_atomic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/perception/hybrid_rgbd_relation_pilot.json"


def audit_relation(
    observation_dir: Path,
    model_input_path: Path,
    ranking_path: Path,
    config_path: Path,
) -> dict:
    observation_dir = observation_dir.resolve()
    model_input = load_json(model_input_path.resolve())
    ranking = load_json(ranking_path.resolve())
    config = load_json(config_path.resolve())
    depth_m = np.load(observation_dir / "depth_m.npy")
    calibration = load_json(observation_dir / "camera_calibration.json")

    reference_entity = model_input["reference_entities"][0]
    reference_mask = load_mask(reference_entity["mask_path"])
    reference_points = masked_world_points(
        depth_m,
        reference_mask,
        calibration,
        minimum_valid_pixels=int(
            config["reference_geometry"]["minimum_valid_depth_pixels"]
        ),
    )
    reference = reference_geometry(reference_points, config["reference_geometry"])
    selected_candidate_id = ranking["selected_candidate_id"]
    candidates = {
        candidate["candidate_id"]: candidate
        for candidate in model_input["candidates"]
    }
    if selected_candidate_id not in candidates:
        raise ValueError(
            f"Selected candidate is absent from model input: {selected_candidate_id}"
        )
    selected = predict_candidate(
        candidates[selected_candidate_id],
        reference_entity,
        depth_m,
        calibration,
        reference_mask,
        reference,
        config,
    )
    qwen_relation = ranking.get("selected_candidate_relation") or {}
    return {
        "schema_version": "saved-learned-rgbd-relation-audit-v1",
        "status": "completed",
        "observation_dir": str(observation_dir),
        "model_input_path": str(model_input_path.resolve()),
        "ranking_path": str(ranking_path.resolve()),
        "config_path": str(config_path.resolve()),
        "selected_candidate_id": selected_candidate_id,
        "qwen_relation_top_label": qwen_relation.get("top_label"),
        "rgbd_relation": selected,
        "prediction_inputs": [
            "learned_candidate_mask",
            "learned_reference_mask",
            "metric_depth",
            "camera_calibration",
            "known_reference_and_mug_dimensions",
        ],
        "simulator_ground_truth_used_for_prediction": False,
        "training_performed": False,
        "valid_for_final_evaluation": False,
        "audit_scope": "post_hoc_development_only",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation_dir", type=Path)
    parser.add_argument("model_input", type=Path)
    parser.add_argument("ranking", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = audit_relation(
        args.observation_dir,
        args.model_input,
        args.ranking,
        args.config,
    )
    write_json_atomic(args.output.resolve(), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
