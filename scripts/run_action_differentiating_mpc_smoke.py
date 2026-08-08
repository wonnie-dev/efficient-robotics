"""Apply the frozen calibration MPC to two new causal-view scene centers.

Only each episode's center learned-perception result is available while the
root action is selected.  The rendered resolving-view label is consulted only
after selection for a development audit.  This is calibration transfer, not
reserved testing or a final-paper evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from run_offline_action_conditioned_mpc_replay import (
    build_episode_rows,
    confidence_bin,
    fit_action_model,
    initial_belief,
    perception_observation,
    select_root_action,
    sigmoid,
    update_with_observation,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PERCEPTION_CONFIG = (
    ROOT
    / "configs"
    / "perception"
    / "action_differentiating_scene_pair_seed185_186.json"
)
DEFAULT_REPLAY_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "offline_action_conditioned_mpc_replay_seed165_184.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "offline_mpc"
    / "action_differentiating_seed185_186"
    / "result.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def selected_relation(ranking: dict[str, Any], relation_type: str) -> str:
    selected_id = ranking["selected_candidate_id"]
    matches = [
        item
        for item in ranking["relations"]
        if item["source_id"] == selected_id
        and item["relation_type"] == relation_type
    ]
    if len(matches) != 1:
        return "missing"
    return str(matches[0]["top_label"])


def center_observation(
    ranking: dict[str, Any], replay_config: dict[str, Any]
) -> dict[str, Any]:
    selected_id = str(ranking["selected_candidate_id"])
    candidate_ids = [str(value) for value in ranking["candidate_ids"]]
    selected_index = candidate_ids.index(selected_id)
    raw_logit = float(ranking["raw_match_logits"][selected_index])
    target_score = sigmoid(
        raw_logit / float(replay_config["target_temperature"])
    )
    identity = confidence_bin(
        target_score, replay_config["identity_confidence_bins"]
    )
    membership = selected_relation(ranking, "membership")
    reference_occlusion = selected_relation(ranking, "occluded_by")
    if membership not in ("inside", "outside", "unknown", "missing"):
        membership = "unknown"
    if reference_occlusion not in ("yes", "no", "unknown", "missing"):
        reference_occlusion = "unknown"
    return {
        "selected_candidate_id": selected_id,
        "selected_raw_match_logit": raw_logit,
        "selected_target_score": target_score,
        "target_temperature_applied": float(
            replay_config["target_temperature"]
        ),
        "identity_bin": identity,
        "membership_observation": membership,
        "reference_occlusion_observation": reference_occlusion,
        "perception_observation": perception_observation(
            identity, reference_occlusion
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--perception-config", type=Path, default=DEFAULT_PERCEPTION_CONFIG
    )
    parser.add_argument(
        "--replay-config", type=Path, default=DEFAULT_REPLAY_CONFIG
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    started = time.perf_counter()
    perception_config = load_json(args.perception_config.resolve())
    replay_config = load_json(args.replay_config.resolve())
    calibration_episodes = build_episode_rows(replay_config)
    model = fit_action_model(
        calibration_episodes, replay_config, action_agnostic=False
    )

    perception_root = resolve_path(perception_config["output_root"])
    sample_by_id = {
        str(item["sample_id"]): item
        for item in perception_config["samples"]
    }
    scene_labels = perception_config["evaluation"][
        "scene_labels_not_used_for_inference"
    ]
    expected_actions = {
        "close_high_only": "viewpoint_close_high",
        "right_only": "viewpoint_right",
    }
    decisions = []
    for seed_text, variant in sorted(scene_labels.items()):
        sample_id = f"{seed_text}_center"
        if sample_id not in sample_by_id:
            raise ValueError(f"Missing center sample: {sample_id}")
        ranking_path = (
            perception_root
            / "grounded_sam2_qwen_rankings"
            / sample_id
            / "result.json"
        )
        ranking = load_json(ranking_path)
        observation = center_observation(ranking, replay_config)
        belief, selected_probability = update_with_observation(
            initial_belief(model),
            observation,
            model,
            action="initial_observation",
        )
        policy = select_root_action(
            belief, selected_probability, model, replay_config
        )
        # The expected action is attached only after the policy has returned.
        expected = expected_actions[str(variant)]
        decisions.append(
            {
                "seed": int(seed_text.removeprefix("seed")),
                "variant": variant,
                "center_sample_id": sample_id,
                "center_ranking_path": str(ranking_path),
                "center_observation": observation,
                "belief_after_center": belief,
                "root_policy": policy,
                "selected_action": policy["selected_action"],
                "expected_resolving_action_posthoc": expected,
                "selected_expected_action": (
                    policy["selected_action"] == expected
                ),
                "future_view_result_read_before_selection": False,
            }
        )

    result = {
        "schema_version": "action-differentiating-mpc-smoke-v1",
        "status": "completed",
        "experiment_id": "action_differentiating_seed185_186",
        "planner": "frozen_action_conditioned_belief_mpc",
        "calibration_seeds": sorted(calibration_episodes),
        "calibration_episode_count": len(calibration_episodes),
        "decisions": decisions,
        "correct_root_action_count": sum(
            item["selected_expected_action"] for item in decisions
        ),
        "episode_count": len(decisions),
        "all_expected_actions_selected": all(
            item["selected_expected_action"] for item in decisions
        ),
        "runtime_seconds": time.perf_counter() - started,
        "gpu_used": False,
        "future_observations_used_for_action_selection": False,
        "simulator_ground_truth_used_for_action_selection": False,
        "simulator_scene_label_used_for_posthoc_audit": True,
        "training_performed": False,
        "calibration_model_fitting_performed": True,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
