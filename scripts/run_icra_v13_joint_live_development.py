#!/usr/bin/env python3
"""Run one live covered-container episode with the V13 joint planner."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from audit_saved_learned_relation import audit_relation
from calibrated_belief import bayesian_update
from core_method_runtime import joint_scene_graph_snapshot
from evaluate_icra_v13_joint_development import (
    plan as joint_plan,
    update_from_row,
)
from run_cover_search_belief_mpc import normalize, plan as cover_plan
from run_live_single_gpu_pipeline import ISAAC_PYTHON, ROOT, wait_for_path, write_json_atomic
from run_negative_evidence_live_smoke import load_json, resolve_path
from run_remove_cover_live_smoke import (
    contact_grasp_success,
    learned_post_remove_localization,
    removal_contact_success,
)
from run_single_gpu_pilot import configured_physical_gpu, require_single_gpu_policy
from unified_task_belief_planner import plan as unified_task_plan
from run_icra_v15_joint_calibration_cv import (
    likelihood_for as v15_likelihood_for,
    likelihood_for_with_source as v15_likelihood_for_with_source,
)
from run_icra_v15b_integrated_scene_conditioned_mpc_cv import semantic_observation
from run_icra_v15b_scene_conditioned_view_calibration import (
    FEATURE_NAMES as V15_VIEW_FEATURE_NAMES,
    bbox_features as v15_bbox_features,
    predict as predict_v15_view_mode,
    relation as v15_relation,
)
from v13_live_joint_runtime import (
    fuse_static_track_localizations,
    initial_joint_observation_row,
    localization_payload,
    tracked_joint_observation_row,
)
from v14_live_development_runtime import (
    condition_joint_model_on_view_resolvability,
    extract_live_post_remove_features,
    predict_view_mode,
)


DEFAULT_CONFIG = ROOT / "configs/research/icra_v13_joint_live_development.json"
OUTPUT_ROOT = ROOT / "outputs/live_pipeline/icra_v13_joint_live_development"
TRACK_IDS = ["track_center_selected", "track_other_target"]


class UnifiedTerminalComplete(Exception):
    """Internal control flow after a unified terminal action completes."""


def v15_live_observation_row(
    perception: dict[str, Any], center_track_candidate_id: str | None
) -> dict[str, Any]:
    """Convert one perception result into the calibrated observation schema."""
    ranking = perception["ranking"]
    evidence = {}
    for candidate_id, raw_logit in zip(
        ranking["candidate_ids"], ranking["raw_match_logits"]
    ):
        membership = v15_relation(ranking, str(candidate_id), "membership")
        evidence[str(candidate_id)] = {
            "raw_match_logit": float(raw_logit),
            "membership": (
                str(membership["top_label"])
                if membership is not None
                else "unknown"
            ),
        }
    row = {
        "center_track_candidate_id": center_track_candidate_id,
        "candidate_evidence": evidence,
        "planner_visible_only": True,
    }
    row["observation_symbol"] = semantic_observation(row)
    return row


def v15_live_view_features(perception: dict[str, Any]) -> dict[str, Any]:
    """Extract scene features used by the frozen view-resolution model."""
    ranking = perception["ranking"]
    model_input = load_json(Path(ranking["input_path"]))
    selected = str(ranking["selected_candidate_id"])
    index = list(ranking["candidate_ids"]).index(selected)
    candidate = next(
        row for row in model_input["candidates"] if row["candidate_id"] == selected
    )
    width = float(model_input["image"]["width"])
    height = float(model_input["image"]["height"])
    selected_bbox = v15_bbox_features(candidate["bbox_xyxy"], width, height)
    reference = model_input["reference_entities"][0]
    container_bbox = v15_bbox_features(reference["bbox_xyxy"], width, height)
    membership = v15_relation(ranking, selected, "membership")
    scores = (
        {
            str(label): float(value)
            for label, value in zip(membership["labels"], membership["raw_logits"])
        }
        if membership is not None
        else {}
    )
    values = [
        float(ranking["raw_match_logits"][index]),
        float(len(ranking["candidate_ids"])),
        scores.get("inside", 0.0) - scores.get("outside", 0.0),
        max(scores.get("inside", 0.0), scores.get("outside", 0.0))
        - scores.get("unknown", 0.0),
        *selected_bbox,
        *container_bbox,
    ]
    return {
        "sample_id": str(model_input["sample_id"]),
        "values": values,
        "named_values": dict(zip(V15_VIEW_FEATURE_NAMES, values)),
        "simulator_ground_truth_used": False,
    }


def update_live_semantic_belief(
    belief: dict[str, float],
    model: dict[str, Any],
    action: str,
    symbol: str,
    *,
    semantic_backoff_model: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Update belief without treating a non-exhaustive miss as absence.

    A wrist view covers only part of the scene.  Therefore a missing target in
    the initial, right, or close-high image is not evidence that the target is
    globally absent.  Empty-container evidence after a successful cover
    removal remains informative and continues through the calibrated Bayesian
    update.
    """
    if action in {
        "initial_observation",
        "viewpoint_close_high",
        "viewpoint_right",
    } and symbol in {"no_target_evidence", "unseen"}:
        return (
            normalize(belief),
            {
                "requested_action": action,
                "requested_symbol": symbol,
                "used_symbol": None,
                "source": "non_exhaustive_view_no_global_absence_update",
                "backoff_actions": [],
            },
        )
    likelihood, source = v15_likelihood_for_with_source(
        model,
        action,
        symbol,
        semantic_backoff_model=semantic_backoff_model,
    )
    return bayesian_update(belief, likelihood), source


def condition_v15_frozen_views(
    model: dict[str, Any],
    frozen_resolution: dict[str, Any],
    probabilities: dict[str, float],
) -> dict[str, Any]:
    """Mix resolved and unresolved likelihoods using predicted view quality."""
    conditioned = copy.deepcopy(model)
    metadata = {}
    for action, mode in (
        ("viewpoint_close_high", "close_high"),
        ("viewpoint_right", "right"),
    ):
        fitted = frozen_resolution["actions"][action]
        vocabulary = tuple(fitted["vocabulary"])
        resolution = float(probabilities[mode])
        likelihood = {
            hypothesis: {
                outcome: (
                    resolution * float(fitted["resolved"][hypothesis][outcome])
                    + (1.0 - resolution) * float(fitted["unresolved"][outcome])
                )
                for outcome in vocabulary
            }
            for hypothesis in conditioned["semantic_hypotheses"]
        }
        conditioned["observation_model"][action] = {
            "outcomes": list(vocabulary),
            "likelihood": likelihood,
        }
        information = conditioned["information_actions"][action]
        information["outcomes"] = list(vocabulary)
        information["observation_likelihood"] = {"open": likelihood}
        information["next_task_state_by_outcome"] = {
            "open": {outcome: "open" for outcome in vocabulary}
        }
        metadata[action] = {
            "view_mode": mode,
            "calibrated_resolution_probability": resolution,
        }
    conditioned["scene_conditioned_sensor_model"] = {
        "method": "frozen_calibrated_resolution_likelihood_mixture",
        "actions": metadata,
        "policy_override_used": False,
        "held_out_future_observation_used": False,
    }
    return conditioned


def add_covered_view_likelihoods(
    model: dict[str, Any], calibration: dict[str, Any]
) -> dict[str, Any]:
    """Add calibrated covered-state view models without changing open-state rows."""
    conditioned = copy.deepcopy(model)
    for action_name in ("viewpoint_close_high", "viewpoint_right"):
        source = calibration["actions"][action_name]
        action = conditioned["information_actions"][action_name]
        outcomes = list(action["outcomes"])
        if list(source["outcomes"]) != outcomes:
            raise ValueError(
                f"Covered-view vocabulary mismatch for {action_name}"
            )
        likelihood = source["likelihood"]
        if set(likelihood) != set(conditioned["semantic_hypotheses"]):
            raise ValueError(
                f"Covered-view hypotheses mismatch for {action_name}"
            )
        allowed = list(action["allowed_task_states"])
        if "covered" not in allowed:
            allowed.append("covered")
        action["allowed_task_states"] = allowed
        action["observation_likelihood"]["covered"] = likelihood
        action["next_task_state_by_outcome"]["covered"] = {
            outcome: "covered" for outcome in outcomes
        }
    conditioned["covered_view_calibration"] = {
        "source": calibration.get("schema_version"),
        "calibration_episode_count": calibration.get("episode_count"),
        "training_performed": False,
    }
    return conditioned


def next_output_dir(seed: int, output_root: Path) -> Path:
    """Allocate the next run directory without overwriting prior results."""
    seed_root = output_root / f"seed{seed:03d}"
    seed_root.mkdir(parents=True, exist_ok=True)
    indices = []
    for path in seed_root.glob("run*"):
        try:
            indices.append(int(path.name.removeprefix("run")))
        except ValueError:
            continue
    return seed_root / f"run{max(indices, default=0) + 1:03d}"


def parse_grasp_action(action: str) -> tuple[str, str]:
    """Parse and validate a candidate-specific terminal grasp action."""
    prefix, track_id, membership = action.split(":", maxsplit=2)
    if prefix != "grasp" or track_id not in TRACK_IDS:
        raise ValueError(f"Invalid V13 grasp action: {action}")
    if membership not in {"inside", "outside"}:
        raise ValueError(f"Invalid V13 membership: {membership}")
    return track_id, membership


def unified_action_kind(action: str) -> str:
    """Map a planner action to the live server's physical dispatch branch."""
    if action == "remove_cover":
        return "remove_cover"
    if action.startswith("viewpoint_"):
        return "viewpoint"
    if action.startswith("grasp:"):
        parse_grasp_action(action)
        return "grasp"
    if action == "defer":
        return "defer"
    raise ValueError(f"Unsupported unified live action: {action}")


def live_request_payload(
    *,
    index: int,
    joint_action: str,
    policy: dict[str, Any],
    belief: dict[str, float],
    localization_path: Path | None = None,
) -> dict[str, Any]:
    """Create one server request without changing the planner's action label."""
    kind = unified_action_kind(joint_action)
    request_type = "grasp" if kind == "grasp" else joint_action
    payload: dict[str, Any] = {
        "schema_version": "icra-v16-unified-live-request-v1",
        "index": int(index),
        "type": request_type,
        "joint_action": joint_action,
        "source_policy": policy,
        "joint_belief": belief,
        "physical_execution_requested": kind != "defer",
    }
    if kind == "grasp":
        if localization_path is None:
            raise ValueError("A unified grasp request requires RGB-D localization")
        payload["rgbd_localization_path"] = str(localization_path)
    elif localization_path is not None:
        raise ValueError(f"{joint_action} must not carry grasp localization")
    return payload


def best_available_view(policy: dict[str, Any], remaining: tuple[str, ...]) -> str | None:
    """Return the lowest-cost unused viewpoint in the current value table."""
    values = [
        item for item in policy["action_values"]
        if item["action"] in remaining
    ]
    if not values:
        return None
    return min(values, key=lambda item: (item["expected_cost"], item["action"]))["action"]


def localization_is_plausible(
    estimate: dict[str, Any], maximum_extent_m: float
) -> bool:
    """Reject masks that merge the target with a large part of the scene."""
    extents = [float(value) for value in estimate.get("robust_extent_m", [])]
    return len(extents) == 3 and max(extents) <= float(maximum_extent_m)


def posthoc_semantic_audit(
    output_dir: Path,
    terminal_joint_action: str | None,
    action_sequence: list[str],
    final_grasp: dict[str, Any],
) -> dict[str, Any]:
    """Score target, membership, and required view choices against hidden truth."""
    scene_path = output_dir / "household_scene.json"
    if not scene_path.is_file() or not terminal_joint_action:
        return {"available": False, "passed": False}
    scene = load_json(scene_path)
    truth = scene.get("calibration_ground_truth", {})
    world = truth.get("world_ground_truth", {})
    # Some open-container scenes explicitly store a null outcome design.
    design = truth.get("action_outcome_design") or {}
    if terminal_joint_action == "defer":
        target_absent = world.get("target_exists") is False
        required_sequence = list(design.get("required_action_sequence") or [])
        required_sequence_correct = (
            not required_sequence or action_sequence == required_sequence
        )
        return {
            "available": True,
            "passed": bool(target_absent and required_sequence_correct),
            "target_absent": target_absent,
            "terminal_action": "defer",
            "safe_absent_deferral": target_absent,
            "required_action_sequence": required_sequence,
            "normalized_executed_sequence": list(action_sequence),
            "required_sequence_correct": required_sequence_correct,
            "ground_truth_used_for_action_selection": False,
        }
    if not terminal_joint_action.startswith("grasp:"):
        return {"available": False, "passed": False}
    _, predicted_membership = parse_grasp_action(terminal_joint_action)
    true_membership = world.get("membership")
    target_path = final_grasp.get("manipulation_target_path")
    target_correct = target_path == "/World/TargetRed"
    membership_correct = predicted_membership == true_membership
    selected_views = [
        action for action in action_sequence if action.startswith("viewpoint_")
    ]
    resolving_views = list(
        design.get("resolving_view_actions_after_remove_cover")
        or design.get("resolving_view_actions")
        or []
    )
    view_choice_correct = not resolving_views or bool(
        set(selected_views).intersection(resolving_views)
    )
    normalized_actions = [
        f"grasp_{predicted_membership}" if action.startswith("grasp:") else action
        for action in action_sequence
    ]
    required_sequence = list(design.get("required_action_sequence") or [])
    required_sequence_correct = (
        not required_sequence or normalized_actions == required_sequence
    )
    passed = bool(
        target_correct
        and membership_correct
        and view_choice_correct
        and required_sequence_correct
    )
    return {
        "available": True,
        "passed": passed,
        "target_identity_correct": target_correct,
        "predicted_membership": predicted_membership,
        "true_membership": true_membership,
        "membership_correct": membership_correct,
        "selected_view_actions": selected_views,
        "resolving_view_actions": resolving_views,
        "view_choice_correct": view_choice_correct,
        "required_action_sequence": required_sequence,
        "normalized_executed_sequence": normalized_actions,
        "required_sequence_correct": required_sequence_correct,
        "ground_truth_used_for_action_selection": False,
    }


def unified_physical_task_success(
    terminal_joint_action: str | None,
    action_sequence: list[str],
    server_result: dict[str, Any],
) -> bool:
    """Apply physical success requirements to the action branch actually used."""
    if server_result.get("status") != "completed" or not terminal_joint_action:
        return False
    removal_was_selected = "remove_cover" in action_sequence
    removal = server_result.get("cover_removal_execution") or {}
    if removal_was_selected and not (
        removal.get("removal_verified") and removal_contact_success(removal)
    ):
        return False
    if terminal_joint_action == "defer":
        return server_result.get("terminal_action") in {"defer", "stop"}
    return bool(
        terminal_joint_action.startswith("grasp:")
        and contact_grasp_success(server_result)
    )


def main() -> None:
    """Run one persistent perception, planning, and manipulation episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--startup-timeout-seconds", type=float, default=1200.0)
    args = parser.parse_args()

    config = load_json(args.config.resolve())
    seed = int(config["seed"] if args.seed is None else args.seed)
    require_single_gpu_policy()
    physical_gpu = configured_physical_gpu()
    joint_config = load_json(resolve_path(config["joint_development_config"]))
    unified_model_value = config.get("unified_joint_model") or config.get(
        "v15_frozen_joint_model"
    )
    v15_mode = bool(unified_model_value)
    initial_task_state = str(config.get("initial_task_state", "covered"))
    if initial_task_state not in {"covered", "open"}:
        raise ValueError(
            "initial_task_state must be either 'covered' or 'open': "
            f"{initial_task_state}"
        )
    joint_model = load_json(
        resolve_path(
            unified_model_value
            or config["joint_observation_model"]
        )
    )
    covered_view_value = config.get("unified_covered_view_likelihoods")
    if covered_view_value:
        joint_model = add_covered_view_likelihoods(
            joint_model,
            load_json(resolve_path(str(covered_view_value))),
        )
    if joint_model.get("valid_for_final_evaluation"):
        raise ValueError("Development runner cannot consume a final-test model")
    cover_config = load_json(resolve_path(config["frozen_cover_planner"]))
    view_model = None
    if v15_mode:
        view_model = load_json(
            resolve_path(
                config.get("unified_view_model")
                or config["v15_frozen_view_model"]
            )
        )
        frozen_resolution = load_json(
            resolve_path(
                config.get("unified_resolution_likelihoods")
                or config["v15_frozen_resolution_likelihoods"]
            )
        )
    elif config.get("scene_conditioned_view_model"):
        view_model = load_json(resolve_path(config["scene_conditioned_view_model"]))
        if view_model.get("valid_for_final_evaluation"):
            raise ValueError("Development runner cannot consume a final-test view model")
    unified_root_model = joint_model if v15_mode else None
    root_policy = None
    if not v15_mode and config.get("unified_root_preflight_model"):
        unified_root_model = load_json(
            resolve_path(config["unified_root_preflight_model"])
        )
        root_policy = unified_task_plan(
            unified_root_model["initial_semantic_belief"],
            unified_root_model["initial_task_state"],
            unified_root_model,
        )
    elif not v15_mode:
        root_policy = cover_plan(
            normalize(cover_config["initial_belief"]), cover_config
        )
    if (
        not v15_mode
        and root_policy is not None
        and root_policy["selected_action"] != "remove_cover"
    ):
        raise RuntimeError(
            "The legacy cover-only development runner requires remove_cover; "
            "use a V16 unified model to dispatch arbitrary root actions: "
            f"{root_policy}"
        )

    output_dir = next_output_dir(seed, args.output_root.resolve())
    output_dir.mkdir(parents=True, exist_ok=False)
    method = load_json(ROOT / "configs/research/first_belief_mpc_integration.json")
    method["viewpoint_execution"]["mode"] = "interpolated_joint_physics"
    method["viewpoint_execution"]["debug_ee_positions_world_m"] = {}
    method_path = output_dir / "effective_method_config.json"
    write_json_atomic(method_path, method)
    write_json_atomic(output_dir / "joint_development_config.json", joint_config)
    if root_policy is not None:
        write_json_atomic(output_dir / "root_action_policy.json", root_policy)

    command = [
        str(ISAAC_PYTHON),
        str(ROOT / "scripts/open_minimal_scene.py"),
        "--scene-profile", "benchmark",
        "--headless",
        "--renderer-gpu", str(physical_gpu),
        "--physics-gpu", "0",
        "--live-pipeline-server",
        "--actual-view-motion",
        "--live-session-dir", str(output_dir),
        "--method-config", str(method_path),
        "--seed", str(seed),
        "--household-perception-pilot",
        "--scanned-basket-perception-pilot",
        "--calibration-scene-variant", str(config["scene_variant"]),
        "--basket-collision-physics-pilot",
    ]
    if initial_task_state == "covered":
        command.extend(
            [
                "--execute-persistent-remove-cover",
                "--continue-after-remove-cover",
                "--rg6-lid-calibration-config",
                str(resolve_path(config["rg6_lid_calibration_config"])),
                "--allow-provisional-rg6-lid-physics",
                "--rg6-coupling-mode",
                str(config["rg6_coupling_mode"]),
                "--coordinated-rg6-total-drive-effort-limit-nm",
                str(config["coordinated_rg6_total_drive_effort_limit_nm"]),
            ]
        )
    else:
        command.append("--execute-persistent-composite-grasp")
    if config.get("record_debug_video") is False:
        command.append("--disable-manipulation-video")
    gpu_text = str(physical_gpu)
    environment = dict(os.environ)
    environment.update(
        {
            "PHYSICAL_GPU": gpu_text,
            "CUDA_VISIBLE_DEVICES": gpu_text,
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "NVIDIA_VISIBLE_DEVICES": gpu_text,
        }
    )
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        environment.pop(name, None)

    stdout = (output_dir / "isaac_stdout.log").open("w", encoding="utf-8")
    stderr = (output_dir / "isaac_stderr.log").open("w", encoding="utf-8")
    started = time.perf_counter()
    server = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, text=True)
    updates = []
    graphs = []
    policies = [root_policy] if root_policy is not None else []
    tracking_audits = []
    action_sequence = []
    terminal_joint_action = None
    terminal_server_action = None
    terminal_localization_path = None
    reference_perception = None
    relation_audits = []
    scene_conditioned_view_prediction = None
    scene_conditioned_view_policy = None
    planning_joint_model = joint_model
    integrated_view_sensor_model = None
    current_perception = None
    center_perception = None
    request_index = 0
    precover_view_actions: list[str] = []
    track_localizations: dict[str, dict[str, Any] | None] = {
        "track_center_selected": None,
        "track_other_target": None,
    }
    track_localization_history: dict[str, list[dict[str, Any]]] = {
        "track_center_selected": [],
        "track_other_target": [],
    }
    try:
        wait_for_path(
            output_dir / "observation_ready_000.json",
            server,
            args.startup_timeout_seconds,
        )
        if v15_mode:
            center_dir = output_dir / "observations/center"
            center_perception = learned_post_remove_localization(
                output_dir,
                center_dir,
                observation_index=0,
                view="center",
                task_overrides=config["perception_task_overrides"],
            )
            center_relation = audit_relation(
                center_dir,
                Path(center_perception["ranking"]["input_path"]),
                Path(center_perception["ranking_path"]),
                resolve_path(config["rgbd_relation_config"]),
            )
            relation_audits.append(center_relation)
            center_selected = str(
                center_perception["ranking"]["selected_candidate_id"]
            )
            center_row = v15_live_observation_row(
                center_perception, center_selected
            )
            current_perception = center_perception
            reference_perception = center_perception
            belief, likelihood_source = update_live_semantic_belief(
                joint_model["initial_semantic_belief"],
                joint_model,
                "initial_observation",
                center_row["observation_symbol"],
                semantic_backoff_model=joint_model,
            )
            updates.append(
                {
                    "action": "initial_observation",
                    "observation": center_row["observation_symbol"],
                    "posterior": belief,
                    "likelihood_source": likelihood_source,
                }
            )
            track_localizations["track_center_selected"] = center_perception[
                "localization"
            ]["estimates"]["selected_target"]
            track_localization_history["track_center_selected"].append(
                track_localizations["track_center_selected"]
            )
            root_policy = unified_task_plan(
                belief,
                initial_task_state,
                planning_joint_model,
                horizon=int(planning_joint_model["horizon"]),
                remaining_actions=tuple(
                    planning_joint_model["information_actions"]
                ),
            )
            policies.append(root_policy)
            write_json_atomic(output_dir / "root_action_policy.json", root_policy)
            graphs.append(
                joint_scene_graph_snapshot(
                    step=0,
                    view="center",
                    joint_belief=belief,
                    candidate_track_ids=TRACK_IDS,
                    observation={"symbol": center_row["observation_symbol"]},
                )
            )
            write_json_atomic(
                output_dir / "probabilistic_scene_graph_joint_step000.json",
                graphs[-1],
            )
            precover_remaining = tuple(
                planning_joint_model["information_actions"]
            )
            while unified_action_kind(str(root_policy["selected_action"])) == "viewpoint":
                selected_action = str(root_policy["selected_action"])
                action_sequence.append(selected_action)
                precover_view_actions.append(selected_action)
                write_json_atomic(
                    output_dir / f"action_request_{request_index:03d}.json",
                    live_request_payload(
                        index=request_index,
                        joint_action=selected_action,
                        policy=root_policy,
                        belief=belief,
                    ),
                )
                observation_index = request_index + 1
                wait_for_path(
                    output_dir / f"observation_ready_{observation_index:03d}.json",
                    server,
                    args.timeout_seconds,
                )
                view = selected_action.removeprefix("viewpoint_")
                view_dir = output_dir / "observations" / view
                current_perception = learned_post_remove_localization(
                    output_dir,
                    view_dir,
                    observation_index=observation_index,
                    view=view,
                    task_overrides=config["perception_task_overrides"],
                )
                relation = audit_relation(
                    view_dir,
                    Path(current_perception["ranking"]["input_path"]),
                    Path(current_perception["ranking_path"]),
                    resolve_path(config["rgbd_relation_config"]),
                )
                relation_audits.append(relation)
                write_json_atomic(
                    output_dir / f"precover_{view}_learned_relation.json",
                    relation,
                )
                _, tracking = tracked_joint_observation_row(
                    center_perception,
                    current_perception,
                    relation,
                    view_dir,
                    action=selected_action,
                    target_temperature=float(config["target_temperature"]),
                    maximum_center_distance_m=float(
                        config["maximum_track_center_distance_m"]
                    ),
                )
                quality_rejections = []
                for track_id, estimate in tracking["track_localizations"].items():
                    if estimate is None:
                        continue
                    maximum_extent = float(
                        config.get("maximum_target_localization_extent_m", 0.18)
                    )
                    if localization_is_plausible(estimate, maximum_extent):
                        track_localizations[track_id] = estimate
                        track_localization_history[track_id].append(estimate)
                    else:
                        quality_rejections.append(
                            {
                                "track_id": track_id,
                                "robust_extent_m": estimate.get("robust_extent_m"),
                                "maximum_allowed_extent_m": maximum_extent,
                                "previous_valid_localization_retained": (
                                    track_localizations.get(track_id) is not None
                                ),
                            }
                        )
                tracking["localization_quality_rejections"] = quality_rejections
                tracking_audits.append(tracking)
                write_json_atomic(
                    output_dir / f"v16_tracking_precover_{view}.json",
                    tracking,
                )
                observation_row = v15_live_observation_row(
                    current_perception,
                    tracking["matched_center_track_candidate_id"],
                )
                belief, likelihood_source = update_live_semantic_belief(
                    belief,
                    planning_joint_model,
                    selected_action,
                    observation_row["observation_symbol"],
                    semantic_backoff_model=joint_model,
                )
                updates.append(
                    {
                        "action": selected_action,
                        "observation": observation_row["observation_symbol"],
                        "posterior": belief,
                        "likelihood_source": likelihood_source,
                    }
                )
                graphs.append(
                    joint_scene_graph_snapshot(
                        step=observation_index,
                        view=view,
                        joint_belief=belief,
                        candidate_track_ids=TRACK_IDS,
                        observation={"symbol": observation_row["observation_symbol"]},
                    )
                )
                write_json_atomic(
                    output_dir
                    / f"probabilistic_scene_graph_joint_step{observation_index:03d}.json",
                    graphs[-1],
                )
                precover_remaining = tuple(
                    action
                    for action in precover_remaining
                    if action != selected_action
                )
                request_index += 1
                root_policy = unified_task_plan(
                    belief,
                    initial_task_state,
                    planning_joint_model,
                    horizon=min(
                        int(planning_joint_model["horizon"]),
                        len(precover_remaining),
                    ),
                    remaining_actions=precover_remaining,
                )
                policies.append(root_policy)
                write_json_atomic(
                    output_dir / f"root_action_policy_{request_index:03d}.json",
                    root_policy,
                )

            selected_root_action = str(root_policy["selected_action"])
            selected_root_kind = unified_action_kind(selected_root_action)
            if selected_root_kind == "grasp":
                track_id, membership = parse_grasp_action(selected_root_action)
                estimate = track_localizations.get(track_id)
                if estimate is None:
                    raise RuntimeError(
                        "Unified planner selected a geometrically unavailable "
                        f"root grasp without localization: {selected_root_action}"
                    )
                terminal_joint_action = selected_root_action
                terminal_server_action = "grasp"
                terminal_localization_path = (
                    output_dir / "v16_terminal_rgbd_localization.json"
                )
                write_json_atomic(
                    terminal_localization_path,
                    localization_payload(
                        fuse_static_track_localizations(
                            track_localization_history[track_id]
                        ),
                        Path(current_perception["localization"]["observation_dir"]),
                        track_id,
                    ),
                )
                action_sequence.append(selected_root_action)
                write_json_atomic(
                    output_dir / f"action_request_{request_index:03d}.json",
                    live_request_payload(
                        index=request_index,
                        joint_action=selected_root_action,
                        policy=root_policy,
                        belief=belief,
                        localization_path=terminal_localization_path,
                    ),
                )
                wait_for_path(output_dir / "server_result.json", server, args.timeout_seconds)
                server.wait(timeout=120.0)
                raise UnifiedTerminalComplete
            if selected_root_kind == "defer":
                terminal_joint_action = "defer"
                terminal_server_action = "defer"
                action_sequence.append("defer")
                write_json_atomic(
                    output_dir / f"action_request_{request_index:03d}.json",
                    live_request_payload(
                        index=request_index,
                        joint_action="defer",
                        policy=root_policy,
                        belief=belief,
                    ),
                )
                wait_for_path(output_dir / "server_result.json", server, args.timeout_seconds)
                server.wait(timeout=120.0)
                raise UnifiedTerminalComplete
            if selected_root_kind != "remove_cover":
                raise RuntimeError(
                    f"Unhandled unified root action: {selected_root_action}"
                )

        post_observation_index = request_index + 1
        action_sequence.append("remove_cover")
        write_json_atomic(
            output_dir / f"action_request_{request_index:03d}.json",
            (
                live_request_payload(
                    index=request_index,
                    joint_action="remove_cover",
                    policy=root_policy,
                    belief=belief,
                )
                if v15_mode
                else {
                    "schema_version": "icra-v13-live-request-v1",
                    "index": request_index,
                    "type": "remove_cover",
                    "source_policy": root_policy,
                    "future_observation_used_for_selection": False,
                }
            ),
        )
        wait_for_path(
            output_dir / f"observation_ready_{post_observation_index:03d}.json",
            server,
            args.timeout_seconds,
        )
        post_dir = output_dir / "observations/post_remove"
        post_perception = learned_post_remove_localization(
            output_dir,
            post_dir,
            observation_index=post_observation_index,
            view="post_remove",
            task_overrides=config["perception_task_overrides"],
        )
        current_perception = post_perception
        reference_perception = center_perception if v15_mode else post_perception
        relation = audit_relation(
            post_dir,
            Path(post_perception["ranking"]["input_path"]),
            Path(post_perception["ranking_path"]),
            resolve_path(config["rgbd_relation_config"]),
        )
        relation_audits.append(relation)
        write_json_atomic(output_dir / "post_remove_learned_relation.json", relation)
        if v15_mode:
            _, post_tracking = tracked_joint_observation_row(
                center_perception,
                post_perception,
                relation,
                post_dir,
                action="remove_cover",
                target_temperature=float(config["target_temperature"]),
                maximum_center_distance_m=float(
                    config["maximum_track_center_distance_m"]
                ),
            )
            post_center_id = post_tracking["matched_center_track_candidate_id"]
            post_row = v15_live_observation_row(post_perception, post_center_id)
            belief, likelihood_source = update_live_semantic_belief(
                belief,
                joint_model,
                "remove_cover",
                post_row["observation_symbol"],
                semantic_backoff_model=joint_model,
            )
            updates.append(
                {
                    "action": "remove_cover",
                    "observation": post_row["observation_symbol"],
                    "posterior": belief,
                    "likelihood_source": likelihood_source,
                }
            )
            for track_id, estimate in post_tracking["track_localizations"].items():
                if estimate is not None:
                    track_localizations[track_id] = estimate
                    track_localization_history[track_id].append(estimate)
            tracking_audits.append(post_tracking)
            write_json_atomic(output_dir / "v15_tracking_post_remove.json", post_tracking)
            view_features = v15_live_view_features(post_perception)
            view_prediction = predict_v15_view_mode(
                view_features,
                list(view_model["episodes"]),
                k=int(view_model["neighbor_count"]),
                beta=float(view_model["probability_pseudocount"]),
            )
            scene_conditioned_view_prediction = {
                **view_prediction,
                "features": view_features,
                "held_out_future_view_used": False,
            }
            planning_joint_model = condition_v15_frozen_views(
                joint_model,
                frozen_resolution,
                view_prediction["probabilities"],
            )
            integrated_view_sensor_model = planning_joint_model[
                "scene_conditioned_sensor_model"
            ]
            write_json_atomic(
                output_dir / "scene_conditioned_view_prediction.json",
                scene_conditioned_view_prediction,
            )
            write_json_atomic(
                output_dir / "integrated_scene_conditioned_sensor_model.json",
                planning_joint_model,
            )
            graphs.append(
                joint_scene_graph_snapshot(
                    step=post_observation_index,
                    view="post_remove",
                    joint_belief=belief,
                    candidate_track_ids=TRACK_IDS,
                    observation={"symbol": post_row["observation_symbol"]},
                )
            )
            write_json_atomic(
                output_dir
                / f"probabilistic_scene_graph_joint_step{post_observation_index:03d}.json",
                graphs[-1],
            )
        elif view_model is not None:
            view_features = extract_live_post_remove_features(
                reference_perception, relation
            )
            scene_conditioned_view_prediction = predict_view_mode(
                view_features, view_model
            )
            planning_joint_model = condition_joint_model_on_view_resolvability(
                joint_model, scene_conditioned_view_prediction
            )
            integrated_view_sensor_model = planning_joint_model[
                "scene_conditioned_sensor_model"
            ]
            write_json_atomic(
                output_dir / "integrated_scene_conditioned_sensor_model.json",
                planning_joint_model,
            )
        if not v15_mode:
            row = initial_joint_observation_row(
                reference_perception,
                relation,
                target_temperature=float(config["target_temperature"]),
            )
            belief, update = update_from_row(
                planning_joint_model["prior"], row, planning_joint_model
            )
            updates.append(update)
            track_localizations["track_center_selected"] = reference_perception[
                "localization"
            ]["estimates"]["selected_target"]
            track_localization_history["track_center_selected"].append(
                track_localizations["track_center_selected"]
            )
            graphs.append(
                joint_scene_graph_snapshot(
                    step=post_observation_index,
                    view="post_remove",
                    joint_belief=belief,
                    candidate_track_ids=TRACK_IDS,
                    observation={"symbol": update["observation"]},
                )
            )
            write_json_atomic(
                output_dir
                / f"probabilistic_scene_graph_joint_step{post_observation_index:03d}.json",
                graphs[-1],
            )

        remaining = (
            tuple(
                action
                for action in ("viewpoint_close_high", "viewpoint_right")
                if action not in precover_view_actions
            )
            if v15_mode
            else tuple(joint_config["actions"])
        )
        request_index = post_observation_index
        maximum_views = int(joint_config["maximum_view_actions"])
        views_executed = len(precover_view_actions)
        while True:
            if v15_mode:
                policy = unified_task_plan(
                    belief,
                    "open",
                    planning_joint_model,
                    horizon=min(
                        int(planning_joint_model["horizon"]), len(remaining)
                    ),
                    remaining_actions=remaining,
                )
            else:
                policy = joint_plan(
                    belief,
                    planning_joint_model,
                    joint_config,
                    remaining,
                    min(len(remaining), maximum_views - views_executed),
                )
            policies.append(policy)
            selected_action = str(policy["selected_action"])
            if selected_action.startswith("grasp:"):
                track_id, membership = parse_grasp_action(selected_action)
                if track_localizations.get(track_id) is None:
                    fallback = best_available_view(policy, remaining)
                    if fallback is None:
                        selected_action = "defer"
                    else:
                        selected_action = fallback
                else:
                    terminal_joint_action = selected_action
                    terminal_server_action = (
                        "grasp" if v15_mode else f"grasp_{membership}"
                    )
                    payload = localization_payload(
                        fuse_static_track_localizations(
                            track_localization_history[track_id]
                        ),
                        Path(current_perception["localization"]["observation_dir"]),
                        track_id,
                    )
                    terminal_localization_path = output_dir / "v13_terminal_rgbd_localization.json"
                    write_json_atomic(terminal_localization_path, payload)
                    action_sequence.append(selected_action)
                    write_json_atomic(
                        output_dir / f"action_request_{request_index:03d}.json",
                        (
                            live_request_payload(
                                index=request_index,
                                joint_action=selected_action,
                                policy=policy,
                                belief=belief,
                                localization_path=terminal_localization_path,
                            )
                            if v15_mode
                            else {
                                "schema_version": "icra-v13-live-request-v1",
                                "index": request_index,
                                "type": terminal_server_action,
                                "joint_action": selected_action,
                                "source_policy": policy,
                                "joint_belief": belief,
                                "rgbd_localization_path": str(
                                    terminal_localization_path
                                ),
                                "physical_execution_requested": True,
                            }
                        ),
                    )
                    wait_for_path(output_dir / "server_result.json", server, args.timeout_seconds)
                    server.wait(timeout=120.0)
                    break
            if selected_action == "defer":
                terminal_joint_action = "defer"
                terminal_server_action = "defer"
                action_sequence.append("defer")
                write_json_atomic(
                    output_dir / f"action_request_{request_index:03d}.json",
                    (
                        live_request_payload(
                            index=request_index,
                            joint_action="defer",
                            policy=policy,
                            belief=belief,
                        )
                        if v15_mode
                        else {
                            "schema_version": "icra-v13-live-request-v1",
                            "index": request_index,
                            "type": "defer",
                            "source_policy": policy,
                            "joint_belief": belief,
                            "physical_execution_requested": False,
                        }
                    ),
                )
                wait_for_path(
                    output_dir / "server_result.json",
                    server,
                    args.timeout_seconds,
                )
                server.wait(timeout=120.0)
                break
            if selected_action not in remaining:
                raise RuntimeError(f"V13 planner returned unavailable action: {selected_action}")
            action_sequence.append(selected_action)
            write_json_atomic(
                output_dir / f"action_request_{request_index:03d}.json",
                {
                    "schema_version": "icra-v13-live-request-v1",
                    "index": request_index,
                    "type": selected_action,
                    "source_policy": policy,
                    "joint_belief": belief,
                    "physical_execution_requested": True,
                },
            )
            observation_index = request_index + 1
            wait_for_path(
                output_dir / f"observation_ready_{observation_index:03d}.json",
                server,
                args.timeout_seconds,
            )
            view = selected_action.removeprefix("viewpoint_")
            view_dir = output_dir / "observations" / view
            current_perception = learned_post_remove_localization(
                output_dir,
                view_dir,
                observation_index=observation_index,
                view=view,
                task_overrides=config["perception_task_overrides"],
            )
            relation = audit_relation(
                view_dir,
                Path(current_perception["ranking"]["input_path"]),
                Path(current_perception["ranking_path"]),
                resolve_path(config["rgbd_relation_config"]),
            )
            relation_audits.append(relation)
            write_json_atomic(output_dir / f"{view}_learned_relation.json", relation)
            row, tracking = tracked_joint_observation_row(
                reference_perception,
                current_perception,
                relation,
                view_dir,
                action=selected_action,
                target_temperature=float(config["target_temperature"]),
                maximum_center_distance_m=float(config["maximum_track_center_distance_m"]),
            )
            quality_rejections = []
            for track_id, estimate in tracking["track_localizations"].items():
                if estimate is not None:
                    maximum_extent = float(
                        config.get("maximum_target_localization_extent_m", 0.18)
                    )
                    if localization_is_plausible(estimate, maximum_extent):
                        track_localizations[track_id] = estimate
                        track_localization_history[track_id].append(estimate)
                    else:
                        quality_rejections.append(
                            {
                                "track_id": track_id,
                                "robust_extent_m": estimate.get("robust_extent_m"),
                                "maximum_allowed_extent_m": maximum_extent,
                                "previous_valid_localization_retained": (
                                    track_localizations.get(track_id) is not None
                                ),
                            }
                        )
            tracking["localization_quality_rejections"] = quality_rejections
            tracking_audits.append(tracking)
            write_json_atomic(output_dir / f"v13_tracking_{view}.json", tracking)
            if v15_mode:
                v15_row = v15_live_observation_row(
                    current_perception,
                    tracking["matched_center_track_candidate_id"],
                )
                belief, likelihood_source = update_live_semantic_belief(
                    belief,
                    planning_joint_model,
                    selected_action,
                    v15_row["observation_symbol"],
                    semantic_backoff_model=joint_model,
                )
                update = {
                    "action": selected_action,
                    "observation": v15_row["observation_symbol"],
                    "posterior": belief,
                    "likelihood_source": likelihood_source,
                }
            else:
                belief, update = update_from_row(
                    belief, row, planning_joint_model
                )
            updates.append(update)
            graphs.append(
                joint_scene_graph_snapshot(
                    step=observation_index,
                    view=view,
                    joint_belief=belief,
                    candidate_track_ids=TRACK_IDS,
                    observation={"symbol": update["observation"]},
                )
            )
            write_json_atomic(
                output_dir / f"probabilistic_scene_graph_joint_step{observation_index:03d}.json",
                graphs[-1],
            )
            remaining = tuple(item for item in remaining if item != selected_action)
            views_executed += 1
            request_index += 1
    except UnifiedTerminalComplete:
        pass
    finally:
        if server.poll() is None:
            server.terminate()
            server.wait(timeout=30.0)
        stdout.close()
        stderr.close()

    server_result = load_json(output_dir / "server_result.json") if (output_dir / "server_result.json").is_file() else {}
    removal = server_result.get("cover_removal_execution") or {}
    removal_was_selected = "remove_cover" in action_sequence
    physical_task_success = unified_physical_task_success(
        terminal_joint_action,
        action_sequence,
        server_result,
    )
    final_grasp = server_result.get("grasp_execution") or {}
    semantic_audit = posthoc_semantic_audit(
        output_dir,
        terminal_joint_action,
        action_sequence,
        final_grasp,
    )
    scientific_success = bool(physical_task_success and semantic_audit["passed"])
    runtime_seconds = time.perf_counter() - started
    manipulation_timings = [
        value.get("timing") or {}
        for value in (removal, final_grasp)
        if isinstance(value, dict)
    ]
    debug_frame_capture_seconds = sum(
        float(value.get("debug_frame_capture_seconds", 0.0))
        for value in manipulation_timings
    )
    video_encoding_seconds = sum(
        float(value.get("video_encoding_seconds", 0.0))
        for value in manipulation_timings
    )
    media_overhead_seconds = (
        debug_frame_capture_seconds + video_encoding_seconds
    )
    result = {
        "schema_version": (
            "icra-v16-unified-live-pretest-smoke-v1"
            if str(joint_model.get("schema_version", "")).startswith("icra-v16")
            else "icra-v15b-frozen-live-pretest-smoke-v1"
            if v15_mode
            else "icra-v13-joint-live-development-result-v1"
        ),
        "status": "completed" if scientific_success else "failed",
        "seed": seed,
        "runtime_seconds": runtime_seconds,
        "runtime_breakdown": {
            "total_wall_seconds": runtime_seconds,
            "debug_frame_capture_seconds": debug_frame_capture_seconds,
            "video_encoding_seconds": video_encoding_seconds,
            "media_overhead_seconds": media_overhead_seconds,
            "online_pipeline_excluding_debug_media_seconds": max(
                0.0, runtime_seconds - media_overhead_seconds
            ),
            "online_runtime_excludes_video_encoding": True,
            "online_runtime_excludes_debug_frame_capture": True,
        },
        "physical_task_success": physical_task_success,
        "posthoc_semantic_protocol_audit": semantic_audit,
        "scientific_episode_success": scientific_success,
        "action_sequence": action_sequence,
        "initial_task_state": initial_task_state,
        "root_action_policy": root_policy,
        "root_action_forced": False if unified_root_model is not None else None,
        "live_smoke_branch_scope": (
            config.get("live_smoke_branch_scope")
            or (
                "unified_root_action_dispatch"
                if v15_mode
                else "legacy_remove_cover_then_replan"
            )
        ),
        "root_action_dispatch": {
            "supported": [
                "remove_cover",
                "viewpoint_close_high",
                "viewpoint_right",
                "grasp",
                "defer",
            ],
            "precover_view_actions": precover_view_actions,
            "remove_cover_selected": removal_was_selected,
            "planner_action_forwarded_without_branch_forcing": bool(v15_mode),
        },
        "root_action_model_status": (
            unified_root_model.get("status")
            if unified_root_model is not None
            else "legacy_cover_belief_planner"
        ),
        "terminal_joint_action": terminal_joint_action,
        "terminal_server_action": terminal_server_action,
        "terminal_localization_path": str(terminal_localization_path) if terminal_localization_path else None,
        "joint_belief_updates": updates,
        "joint_scene_graph_steps": graphs,
        "joint_mpc_policies": policies,
        "scene_conditioned_view_prediction": scene_conditioned_view_prediction,
        "scene_conditioned_view_policy": scene_conditioned_view_policy,
        "integrated_view_sensor_model": integrated_view_sensor_model,
        "first_view_policy_override_used": False,
        "tracking_audits": tracking_audits,
        "relation_audits": relation_audits,
        "cover_removal_execution": removal,
        "final_grasp_execution": final_grasp,
        "candidate_identity": "persistent_rgbd_track",
        "joint_hypothesis": "candidate_track_x_membership",
        "semantic_belief_hypothesis": "persistent_candidate_track_x_membership",
        "observable_task_state": (
            "verified_open_after_remove_cover"
            if removal_was_selected
            else "covered"
        ),
        "task_state_is_part_of_semantic_belief": False,
        "scene_conditioned_view_model": (
            "action_conditioned_sensor_model"
            if integrated_view_sensor_model is not None
            else ("frozen_unconditioned_sensor_model" if view_model else None)
        ),
        "marginal_confidence_product_used": False,
        "gpu_policy": {
            "physical_gpu": physical_gpu,
            "renderer_active_gpu": physical_gpu,
            "physics_cuda_device": 0,
            "multi_gpu": False,
        },
        "debug_video_recording_requested": config.get(
            "record_debug_video", True
        ),
        "training_performed": False,
        "calibration_performed": v15_mode,
        "testing_performed": False,
        "valid_for_final_evaluation": False,
        "limitations": (
            [
                "Nonreserved live smoke using the frozen V15b calibration candidate; this is not a final test episode.",
                "RG6 and cover parameters remain transfer proxies until lab measurements are provided.",
            ]
            if v15_mode
            else [
                "Development integration only; the joint observation model is not the final calibration model.",
                "The optional unified root model uses provisional preflight likelihoods until V15 calibration.",
                "RG6 and cover parameters remain transfer proxies until lab measurements are provided."
            ]
        ),
    }
    result_path = output_dir / "icra_v13_joint_live_result.json"
    write_json_atomic(result_path, result)
    print(f"ICRA_V13_JOINT_LIVE_RESULT={result_path}")
    if not scientific_success and not config.get("collect_development_failures", False):
        raise RuntimeError(f"V13 live development episode failed: {result_path}")


if __name__ == "__main__":
    main()
