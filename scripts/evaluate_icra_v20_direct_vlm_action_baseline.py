#!/usr/bin/env python3
"""Replay direct Qwen action rankings without Scene Graph, belief, or MPC."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_icra_v16_policy_comparison import (
    exact_mcnemar,
    paired_cost_bootstrap,
    score_episode,
    summarize,
    write_json,
)
from run_icra_v15_joint_calibration_cv import INFORMATION_ACTIONS, build_rows
from run_icra_v15b_integrated_scene_conditioned_mpc_cv import replace_symbols


DEFAULT_CONFIG = ROOT / "configs/research/icra_v20_direct_vlm_action_baseline_development.json"
MODEL = ROOT / "outputs/calibration/icra_v18_persistent_negative_evidence_candidate/calibration_candidate_model.json"


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def result_index(output_root: Path) -> dict[str, dict[str, Any]]:
    return {
        row["sample_id"]: row
        for path in output_root.glob("shards/shard*/seed*/result.json")
        for row in [load_json(path)]
    }


def grasp_action(
    value: str,
    row: dict[str, Any],
    state: str,
    model: dict[str, Any],
) -> str | None:
    if not value.startswith("grasp_candidate_"):
        return None
    body = value.removeprefix("grasp_")
    candidate_id, membership = body.rsplit("_", maxsplit=1)
    if candidate_id not in row["candidate_evidence"]:
        return None
    center = row.get("center_track_candidate_id")
    track = (
        "track_center_selected"
        if center is not None and candidate_id == str(center)
        else "track_other_target"
    )
    action = f"grasp:{track}:{membership}"
    definition = model["terminal_grasp_actions"].get(action)
    if definition is None or state not in definition["allowed_task_states"]:
        return None
    return action


def replay(
    episode: dict[str, Any],
    rankings: dict[str, dict[str, Any]],
    model: dict[str, Any],
) -> dict[str, Any]:
    seed = int(episode["seed"])
    state = str(episode["initial_task_state"])
    current_view = "center"
    remaining = tuple(action for action in INFORMATION_ACTIONS if action in episode["rows"])
    sequence = []
    selected_trace = []
    inference_seconds = 0.0
    for _ in range(len(remaining) + 1):
        sample_id = f"seed{seed}_{current_view}"
        result = rankings[sample_id]
        inference_seconds += float(result.get("metrics", {}).get("runtime_seconds", 0.0))
        row_action = {
            "center": "initial_observation",
            "post_remove": "remove_cover",
            "right": "viewpoint_right",
            "close_high": "viewpoint_close_high",
        }[current_view]
        observation = episode["rows"][row_action]
        selected_value = None
        planner_action = None
        for candidate in result["ranked_actions"]:
            value = str(candidate["action"])
            if value == "defer":
                selected_value, planner_action = value, value
                break
            if value in INFORMATION_ACTIONS:
                if value in remaining and (value != "remove_cover" or state == "covered"):
                    selected_value, planner_action = value, value
                    break
                continue
            converted = grasp_action(value, observation, state, model)
            if converted is not None:
                selected_value, planner_action = value, converted
                break
        if planner_action is None:
            selected_value, planner_action = "defer", "defer"
        selected_trace.append(
            {
                "sample_id": sample_id,
                "direct_vlm_choice": selected_value,
                "executed_action": planner_action,
                "scene_graph_used": False,
                "belief_update_used": False,
                "mpc_used": False,
            }
        )
        sequence.append(planner_action)
        if planner_action.startswith("grasp:") or planner_action == "defer":
            break
        remaining = tuple(action for action in remaining if action != planner_action)
        if planner_action == "remove_cover":
            state = "open"
            current_view = "post_remove"
        elif state == "open" and planner_action.startswith("viewpoint_"):
            current_view = planner_action.removeprefix("viewpoint_")
        # A viewpoint chosen while the opaque cover is still present consumes
        # motion cost but leaves the current visual evidence unchanged.
    return score_episode(
        episode,
        sequence,
        model,
        inference_seconds,
        {
            "direct_vlm_trace": selected_trace,
            "direct_vlm_inference_runtime_seconds": inference_seconds,
            "scene_graph_used": False,
            "belief_update_used": False,
            "mpc_used": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    perception_root = resolve_path(config["source_perception_root"])
    output_root = resolve_path(config["output_root"])
    episodes = replace_symbols(
        build_rows(perception_root, minimum_iou=0.25, maximum_track_distance_m=0.12)
    )
    rankings = result_index(output_root)
    model = load_json(MODEL)
    rows = [replay(episode, rankings, model) for _, episode in sorted(episodes.items())]
    for row in rows:
        write_json(output_root / "episodes" / f"seed{int(row['seed']):04d}.json", row)
    proposed_root = resolve_path(config["proposed_episode_root"])
    proposed = [
        load_json(proposed_root / f"seed{seed:04d}.json")
        for seed in sorted(episodes)
    ]
    result = {
        "schema_version": "icra-v20-direct-vlm-action-baseline-evaluation-v1",
        "status": "completed",
        "method": "direct_vlm_action_selection",
        "summary": summarize(rows),
        "episode_count": len(rows),
        "paired_comparison_against_proposed": {
            "exact_mcnemar": exact_mcnemar(proposed, rows),
            "paired_cost_bootstrap": paired_cost_bootstrap(proposed, rows),
            "paired_same_seed_set_verified": {
                int(row["seed"]) for row in proposed
            }
            == {int(row["seed"]) for row in rows},
        },
        "scene_graph_used": False,
        "belief_update_used": False,
        "mpc_used": False,
        "simulator_ground_truth_used_for_action_selection": False,
        "simulator_ground_truth_used_posthoc_for_evaluation": True,
        "training_performed": False,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
    }
    write_json(output_root / "evaluation.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
