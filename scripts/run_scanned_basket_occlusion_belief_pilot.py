"""Run the controlled-occlusion action-conditioned belief pilot.

The pre-action plan reads only the center perception result, current belief,
and the configured observation-likelihood model.  The selected future view is
opened only after pre_action_plan.json has been written.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from rgbd_target_localization import estimate_mask_center
from run_non_oracle_hybrid_planner import plan
from run_qwen_belief_mpc_replay import weighted_log_belief_update


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "scanned_basket_occlusion_belief_mpc_pilot.json"
)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def normalize(values: dict[str, float]) -> dict[str, float]:
    total = sum(values.values())
    if total <= 0:
        raise ValueError("Belief has non-positive total mass")
    return {key: value / total for key, value in values.items()}


def softmax(values: list[float], temperature: float) -> list[float]:
    scaled = [value / temperature for value in values]
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def sample_by_view(perception_config: dict, view: str) -> dict:
    suffix = f"_{view}"
    matches = [
        sample
        for sample in perception_config["samples"]
        if sample["sample_id"].endswith(suffix)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one sample for {view}, found {len(matches)}")
    return matches[0]


def ranking_paths(perception_config: dict, view: str) -> tuple[Path, Path]:
    output_root = resolve(perception_config["output_root"])
    sample = sample_by_view(perception_config, view)
    ranking_path = (
        output_root
        / "grounded_sam2_qwen_rankings"
        / sample["sample_id"]
        / "result.json"
    )
    return ranking_path, resolve(sample["observation_dir"])


def candidate_centers(
    ranking: dict, observation_dir: Path
) -> dict[str, list[float]]:
    model_input = json.loads(
        Path(ranking["input_path"]).read_text(encoding="utf-8")
    )
    depth = np.load(observation_dir / "depth_m.npy")
    calibration = json.loads(
        (observation_dir / "camera_calibration.json").read_text(
            encoding="utf-8"
        )
    )
    centers = {}
    for candidate in model_input["candidates"]:
        mask = np.asarray(
            Image.open(resolve(candidate["mask_path"])).convert("L")
        ) > 0
        estimate = estimate_mask_center(
            depth,
            mask,
            calibration,
            label=candidate["candidate_id"],
        )
        centers[candidate["candidate_id"]] = estimate["center_world_m"]
    return centers


def track_mapping(
    reference_centers: dict[str, list[float]],
    current_centers: dict[str, list[float]],
) -> tuple[dict[str, str], dict]:
    reference_ids = list(reference_centers)
    current_ids = list(current_centers)
    if len(reference_ids) != len(current_ids):
        raise ValueError("Candidate count changed between tracked views")
    track_ids = [f"track_{index:03d}" for index in range(1, len(reference_ids) + 1)]
    reference_tracks = dict(zip(reference_ids, track_ids))
    best = None
    for permutation in itertools.permutations(reference_ids):
        pairs = list(zip(current_ids, permutation))
        distances = [
            float(
                np.linalg.norm(
                    np.asarray(current_centers[current_id])
                    - np.asarray(reference_centers[reference_id])
                )
            )
            for current_id, reference_id in pairs
        ]
        total = sum(distances)
        if best is None or total < best[0]:
            best = (total, pairs, distances)
    assert best is not None
    mapping = {
        current_id: reference_tracks[reference_id]
        for current_id, reference_id in best[1]
    }
    return mapping, {
        "method": "minimum_sum_rgbd_world_center_distance",
        "total_distance_m": best[0],
        "pairs": [
            {
                "current_candidate_id": current_id,
                "reference_candidate_id": reference_id,
                "track_id": reference_tracks[reference_id],
                "distance_m": distance,
            }
            for (current_id, reference_id), distance in zip(
                best[1], best[2]
            )
        ],
        "simulator_ground_truth_used": False,
    }


def ranking_belief(
    ranking: dict,
    candidate_to_track: dict[str, str],
    temperature: float,
) -> dict:
    target_scores = softmax(ranking["raw_match_logits"], temperature)
    target = normalize(
        {
            candidate_to_track[candidate_id]: score
            for candidate_id, score in zip(
                ranking["candidate_ids"], target_scores
            )
        }
    )
    relation = ranking["selected_candidate_relation"]
    relation_scores = softmax(relation["raw_logits"], temperature)
    by_label = dict(zip(relation["labels"], relation_scores))
    return {
        "target": target,
        "relation": normalize(
            {
                "inside": by_label.get("inside", 0.0),
                "outside": by_label.get("outside", 0.0),
                "unknown": (
                    by_label.get("behind", 0.0)
                    + by_label.get("unknown", 0.0)
                ),
            }
        ),
        "selected_candidate_id": ranking["selected_candidate_id"],
        "selected_track_id": candidate_to_track[
            ranking["selected_candidate_id"]
        ],
        "raw_logit_temperature": temperature,
        "calibrated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--perception-config",
        type=Path,
        help="Override the perception result set without changing the planner.",
    )
    parser.add_argument(
        "--executed-motion-result",
        type=Path,
        help="Post-plan actual-motion result for the selected viewpoint.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help="Override the debug-pilot output directory.",
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    perception_config_path = (
        args.perception_config
        if args.perception_config is not None
        else resolve(config["scenario"]["perception_config"])
    )
    perception_config = json.loads(
        perception_config_path.read_text(
            encoding="utf-8"
        )
    )
    output_root = (
        args.output_root.resolve()
        if args.output_root is not None
        else resolve(config["output_root"])
    )
    output_root.mkdir(parents=True, exist_ok=True)
    adapter = config["qwen_belief_adapter"]
    temperature = float(adapter["raw_logit_temperature"])
    observation_weight = float(adapter["observation_log_weight"])

    # Pre-action stage: intentionally open only the current center output.
    center_ranking_path, center_observation_dir = ranking_paths(
        perception_config, "center"
    )
    center_ranking = json.loads(
        center_ranking_path.read_text(encoding="utf-8")
    )
    center_centers = candidate_centers(
        center_ranking, center_observation_dir
    )
    center_mapping = {
        candidate_id: f"track_{index:03d}"
        for index, candidate_id in enumerate(center_centers, start=1)
    }
    center_belief = ranking_belief(
        center_ranking, center_mapping, temperature
    )
    planning_config = copy.deepcopy(config)
    planning_config["scenario"]["perception_config"] = str(
        perception_config_path.resolve()
    )
    planning_config["initial_belief"]["target"] = center_belief["target"]
    planning_config["initial_belief"]["relation"] = center_belief["relation"]
    planning_config["initial_belief"]["source"] = (
        "center_grounded_sam2_qwen_rgbd_tracks"
    )
    pre_action_plan = plan(planning_config)
    pre_action_payload = {
        **pre_action_plan,
        "current_observation_input": str(center_ranking_path),
        "current_observation_belief": center_belief,
        "future_perception_outputs_read_before_plan": [],
        "future_capture_files_read_before_plan": [],
        "observation_model_status": planning_config["observation_model"][
            "status"
        ],
    }
    pre_action_path = output_root / "pre_action_plan.json"
    write_json(pre_action_path, pre_action_payload)

    selected_action = pre_action_plan["action_request"]["type"]
    if not selected_action.startswith("viewpoint_"):
        raise RuntimeError(
            f"Pilot expected a re-observation action, got {selected_action}"
        )
    selected_view = selected_action.removeprefix("viewpoint_")
    if selected_view == "center_repeat":
        selected_view = "center"

    # Post-action stage: only now open the selected observation output.
    selected_ranking_path, selected_observation_dir = ranking_paths(
        perception_config, selected_view
    )
    selected_ranking = json.loads(
        selected_ranking_path.read_text(encoding="utf-8")
    )
    selected_centers = candidate_centers(
        selected_ranking, selected_observation_dir
    )
    selected_mapping, tracking = track_mapping(
        center_centers, selected_centers
    )
    observation_belief = ranking_belief(
        selected_ranking, selected_mapping, temperature
    )
    posterior = weighted_log_belief_update(
        {
            "target": center_belief["target"],
            "relation": center_belief["relation"],
        },
        {
            "target": observation_belief["target"],
            "relation": observation_belief["relation"],
        },
        observation_weight,
    )
    post_config = copy.deepcopy(planning_config)
    post_config["initial_belief"]["target"] = posterior["target"]
    post_config["initial_belief"]["relation"] = posterior["relation"]
    post_config["initial_belief"]["source"] = (
        "center_plus_selected_reobservation_weighted_log_update"
    )
    post_config["completed_reobservations"] = 1
    post_config["actions"][selected_action]["enabled"] = False
    post_config["actions"]["viewpoint_center_repeat"]["enabled"] = False
    post_action_plan = plan(post_config)
    executed_motion = None
    if args.executed_motion_result is not None:
        executed_motion = json.loads(
            args.executed_motion_result.read_text(encoding="utf-8")
        )
        expected_sequence = ["center", selected_view]
        if (
            executed_motion.get("status") != "completed"
            or executed_motion.get("sequence") != expected_sequence
            or not all(
                trajectory.get("collision_checked")
                and not trajectory.get("collision_detected")
                and trajectory.get("actual_robot_motion_executed")
                for trajectory in executed_motion.get("trajectories", [])
            )
        ):
            raise RuntimeError(
                "Executed motion does not validate the selected action: "
                f"expected {expected_sequence}, got {executed_motion}"
            )

    result = {
        "schema_version": "scanned-basket-occlusion-belief-pilot-v1",
        "status": "completed",
        "pre_action_plan_path": str(pre_action_path),
        "selected_action": selected_action,
        "selected_view": selected_view,
        "perception_config": str(perception_config_path.resolve()),
        "selected_observation_input": str(selected_ranking_path),
        "rgbd_candidate_tracking": tracking,
        "belief_before_reobservation": {
            "target": center_belief["target"],
            "relation": center_belief["relation"],
        },
        "selected_observation_belief": {
            "target": observation_belief["target"],
            "relation": observation_belief["relation"],
        },
        "belief_after_reobservation": posterior,
        "post_action_plan": post_action_plan,
        "selected_action_execution": {
            "result_path": (
                str(args.executed_motion_result.resolve())
                if args.executed_motion_result is not None
                else None
            ),
            "validated": executed_motion is not None,
            "result": executed_motion,
        },
        "provenance": {
            "future_capture_files_read_before_plan": [],
            "future_perception_outputs_read_before_plan": [],
            "future_observation_prediction_source": (
                "pre_action_geometry_informed_hand_specified_likelihood"
            ),
            "selected_view_output_read_only_after_plan_was_saved": True,
            "unselected_future_view_output_read": False,
            "simulator_ground_truth_used_for_inference_or_planning": False,
            "oracle": False,
            "actual_mpc_solver": False,
        },
        "training_performed": False,
        "calibration_performed": False,
        "valid_for_final_evaluation": False,
    }
    write_json(output_root / "pilot_result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
