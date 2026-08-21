#!/usr/bin/env python3
"""Freeze calibration artifacts before opening the reserved test split."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from calibrate_joint_observation_model import load_json, write_json
from unified_task_belief_planner import validate_unified_method_contract


ROOT = Path(__file__).resolve().parents[1]
CAPTURE_STATUS = ROOT / "outputs/live_pipeline/unified_calibration_episodes/batch_status.json"
PERCEPTION_ROOT = ROOT / "outputs/calibration/perception"
CANDIDATE_ROOT = ROOT / "outputs/calibration/joint_model"
COST_SENSITIVITY = ROOT / "outputs/calibration/cost_sensitivity/result.json"
PHYSICAL_CALIBRATION = ROOT / "outputs/calibration/physical_execution/result.json"
VALIDATION_EPISODE_ROOT = ROOT / "outputs/validation/closed_loop_episode/seed1063"
METHOD_DEFINITIONS = ROOT / "configs/research/method_definitions.json"
PROTOCOL = ROOT / "configs/research/final_evaluation_protocol.json"
TEST_CONFIG = ROOT / "configs/research/reserved_test_episodes.json"
DEFAULT_OUTPUT = ROOT / "outputs/calibration/frozen"
EXPECTED_CALIBRATION_SEEDS = set(range(1064, 1100))
EXPECTED_TEST_SEEDS = set(range(1100, 1160))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def latest_validation_episode(root: Path) -> Path:
    paths = sorted(root.glob("run*/closed_loop_result.json"))
    if not paths:
        raise RuntimeError("The non-reserved closed-loop validation has not run")
    return paths[-1]


def require_calibration_inputs(validation_result_path: Path) -> dict[str, Any]:
    capture = load_json(CAPTURE_STATUS)
    completed = {int(value) for value in capture.get("completed_seeds", [])}
    if capture.get("status") != "completed" or completed != EXPECTED_CALIBRATION_SEEDS:
        raise RuntimeError("The 36-episode calibration capture is incomplete")
    manifest = load_json(PERCEPTION_ROOT / "observation_manifest.json")
    manifest_seeds = {int(row["seed"]) for row in manifest["episodes"]}
    if manifest_seeds != EXPECTED_CALIBRATION_SEEDS:
        raise RuntimeError("The perception manifest is incomplete or changed")
    cv = load_json(CANDIDATE_ROOT / "result.json")
    if int(cv["summary"]["episode_count"]) != 36:
        raise RuntimeError("The joint calibration result does not contain 36 episodes")
    blocking_reasons = list(cv.get("blocking_reasons", []))
    if cv.get("status") != "passed" or blocking_reasons:
        raise RuntimeError(
            "The calibration still has freeze blockers: "
            + ", ".join(blocking_reasons or [str(cv.get("status"))])
        )
    sensitivity = load_json(COST_SENSITIVITY)
    if sensitivity.get("status") != "completed" or int(sensitivity["episode_count"]) != 36:
        raise RuntimeError("The calibration cost sensitivity is incomplete")
    physical = load_json(PHYSICAL_CALIBRATION)
    if physical.get("status") != "passed_calibration":
        raise RuntimeError("The independent physical grasp calibration has not passed")
    validation = load_json(validation_result_path)
    if int(validation["seed"]) in EXPECTED_CALIBRATION_SEEDS | EXPECTED_TEST_SEEDS:
        raise RuntimeError("The validation episode must use a non-calibration, non-test seed")
    if not bool(validation.get("scientific_episode_success")):
        raise RuntimeError("The non-reserved closed-loop validation did not pass")
    if validation.get("root_action_forced") is not False:
        raise RuntimeError("The validation episode forced its root action")
    if validation.get("first_view_policy_override_used") is not False:
        raise RuntimeError("The validation episode used a first-view policy override")
    return {
        "capture": capture,
        "perception_manifest": manifest,
        "cv": cv,
        "sensitivity": sensitivity,
        "physical": physical,
        "validation": validation,
    }


def run(output: Path, validation_result_path: Path | None = None) -> dict[str, Any]:
    capture = load_json(CAPTURE_STATUS)
    completed = {int(value) for value in capture.get("completed_seeds", [])}
    if capture.get("status") != "completed" or completed != EXPECTED_CALIBRATION_SEEDS:
        raise RuntimeError("The 36-episode calibration capture is incomplete")
    validation_path = validation_result_path or latest_validation_episode(
        VALIDATION_EPISODE_ROOT
    )
    inputs = require_calibration_inputs(validation_path)
    joint = load_json(CANDIDATE_ROOT / "calibration_candidate_model.json")
    view = load_json(CANDIDATE_ROOT / "scene_conditioned_view_model_candidate.json")
    resolution = load_json(CANDIDATE_ROOT / "resolution_likelihoods_candidate.json")
    validate_unified_method_contract(joint)
    if dict(inputs["sensitivity"]["nominal_costs_frozen"]) != dict(joint["costs"]):
        raise RuntimeError("The sensitivity report and candidate planner costs differ")
    execution_probability = float(inputs["physical"]["posterior_mean_used_by_planner"])
    for action in joint["terminal_grasp_actions"].values():
        action["conditional_execution_success_probability"] = execution_probability
    joint.update({
        "status": "frozen_before_reserved_test",
        "calibration_episode_count": 36,
        "fixed_confidence_threshold_used": False,
        "policy_override_used": False,
        "training_performed": False,
        "calibration_performed": True,
        "testing_performed": False,
        "valid_for_final_evaluation": False,
    })
    view.update({"status": "frozen_before_reserved_test", "valid_for_final_evaluation": False})
    resolution.update({"status": "frozen_before_reserved_test", "valid_for_final_evaluation": False})

    output.mkdir(parents=True, exist_ok=True)
    joint_path = output / "joint_observation_and_mpc_model.json"
    view_path = output / "scene_conditioned_view_model.json"
    resolution_path = output / "resolution_likelihoods.json"
    write_json(joint_path, joint)
    write_json(view_path, view)
    write_json(resolution_path, resolution)

    definitions = copy.deepcopy(load_json(METHOD_DEFINITIONS))
    definitions.update({
        "status": "frozen_before_untouched_test",
        "calibration_performed": True,
        "testing_performed": False,
        "valid_for_final_evaluation": False,
    })
    definitions_path = output / "frozen_method_definitions.json"
    write_json(definitions_path, definitions)

    protocol = copy.deepcopy(load_json(PROTOCOL))
    protocol.update({
        "status": "frozen_before_untouched_test",
        "method_definitions": str(definitions_path.resolve()),
        "reserved_test_launch_authorized": True,
        "calibration_performed": True,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
    })
    protocol_path = output / "frozen_final_evaluation_protocol.json"
    write_json(protocol_path, protocol)

    capture_config = copy.deepcopy(load_json(TEST_CONFIG))
    capture_config.update({
        "status": "frozen_before_reserved_test_opening",
        "final_evaluation_protocol": str(protocol_path.resolve()),
        "freeze_manifest": str((output / "freeze_manifest.json").resolve()),
        "launch_authorized": True,
        "reserved_test_opened": False,
        "calibration_performed": True,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
    })
    capture_config_path = output / "frozen_reserved_test_capture_config.json"
    write_json(capture_config_path, capture_config)

    input_files = [
        CAPTURE_STATUS,
        PERCEPTION_ROOT / "observation_manifest.json",
        CANDIDATE_ROOT / "result.json",
        CANDIDATE_ROOT / "calibration_candidate_model.json",
        CANDIDATE_ROOT / "scene_conditioned_view_model_candidate.json",
        CANDIDATE_ROOT / "resolution_likelihoods_candidate.json",
        COST_SENSITIVITY,
        PHYSICAL_CALIBRATION,
        validation_path,
        METHOD_DEFINITIONS,
        PROTOCOL,
        TEST_CONFIG,
        ROOT / "scripts/unified_task_belief_planner.py",
        ROOT / "scripts/evaluate_policy_comparison.py",
        ROOT / "scripts/capture_calibration_batch.py",
    ]
    manifest = {
        "schema_version": "calibration-freeze-manifest-v1",
        "status": "frozen_ready_for_reserved_test",
        "calibration_episode_count": 36,
        "calibration_seeds": sorted(EXPECTED_CALIBRATION_SEEDS),
        "reserved_test_seeds": sorted(EXPECTED_TEST_SEEDS),
        "reserved_test_seed_list_sha256": canonical_hash(sorted(EXPECTED_TEST_SEEDS)),
        "reserved_test_opened": False,
        "reserved_test_launch_authorized": True,
        "planner": {
            "costs": dict(joint["costs"]),
            "conditional_execution_success_probability": execution_probability,
            "fixed_confidence_threshold_used": False,
            "first_view_policy_override_used": False,
        },
        "nonreserved_closed_loop_validation": {
            "path": str(validation_path.resolve()),
            "seed": int(inputs["validation"]["seed"]),
            "action_sequence": list(inputs["validation"]["action_sequence"]),
            "scientific_episode_success": True,
        },
        "artifacts": {
            "joint_model": {"path": str(joint_path.resolve()), "sha256": sha256(joint_path)},
            "view_model": {"path": str(view_path.resolve()), "sha256": sha256(view_path)},
            "resolution_likelihoods": {"path": str(resolution_path.resolve()), "sha256": sha256(resolution_path)},
            "method_definitions": {"path": str(definitions_path.resolve()), "sha256": sha256(definitions_path)},
            "final_protocol": {"path": str(protocol_path.resolve()), "sha256": sha256(protocol_path)},
            "test_capture_config": {"path": str(capture_config_path.resolve()), "sha256": sha256(capture_config_path)},
        },
        "input_hashes": {str(path.resolve()): sha256(path) for path in input_files},
        "training_performed": False,
        "calibration_performed": True,
        "testing_performed": False,
        "valid_for_final_evaluation": False,
        "next_required_stage": "open_reserved_test_once_with_frozen_capture_config",
    }
    write_json(output / "freeze_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validation-result", type=Path)
    args = parser.parse_args()
    result = run(
        args.output_root.resolve(),
        args.validation_result.resolve() if args.validation_result else None,
    )
    print(json.dumps({key: result[key] for key in ("status", "calibration_episode_count", "reserved_test_opened", "next_required_stage")}, indent=2))


if __name__ == "__main__":
    main()
