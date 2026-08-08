#!/usr/bin/env python3
"""Validate learned post-remove RGB-D localization on a saved observation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_live_single_gpu_pipeline import write_json_atomic
from run_remove_cover_live_smoke import learned_post_remove_localization
from run_single_gpu_pilot import require_single_gpu_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    require_single_gpu_policy()
    observation_dir = args.observation_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not observation_dir.is_dir():
        parser.error(f"Observation directory does not exist: {observation_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    result = learned_post_remove_localization(
        output_dir,
        observation_dir,
        observation_index=1,
    )
    summary = {
        "schema_version": "saved-post-remove-learned-localization-smoke-v1",
        "status": "completed",
        "observation_dir": str(observation_dir),
        "selected_candidate_id": result["selected_candidate_id"],
        "ranking_path": result["ranking_path"],
        "target_mask_path": result["target_mask_path"],
        "localization_path": result["localization_path"],
        "center_world_m": result["localization"]["estimates"][
            "selected_target"
        ]["center_world_m"],
        "simulator_ground_truth_used_for_control": False,
        "learned_perception_executed": True,
        "training_performed": False,
        "valid_for_final_evaluation": False,
    }
    write_json_atomic(output_dir / "saved_perception_smoke_result.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
