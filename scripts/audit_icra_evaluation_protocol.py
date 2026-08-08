#!/usr/bin/env python3
"""Fail-closed preflight audit for the paper-facing evaluation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "configs" / "research" / "icra_simulation_evaluation_protocol_v1.json"
)
REQUIRED_METHODS = {
    "direct_perception_immediate_grasp",
    "deterministic_scene_graph_planner",
    "object_node_uncertainty_without_relation_edges",
    "fixed_viewpoint_policy",
    "greedy_one_step_information_gain",
    "entropy_only_without_task_risk",
    "proposed_task_risk_aware_action_conditioned_belief_mpc",
}
REQUIRED_ABLATIONS = {
    "no_relation_edge_uncertainty",
    "no_calibration_raw_scores",
    "no_action_conditioned_future_belief",
    "no_negative_evidence_update",
    "no_task_loss_or_wrong_commitment_cost",
    "horizon_one_greedy",
}
RELATION_LEAK_WORDS = {"inside", "outside", "behind", "near", "covered"}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_frozen_artifact(
    config: dict[str, Any], frozen_path: Path
) -> list[str]:
    """Verify every currently frozen item has an immutable source record."""
    unresolved: list[str] = []
    try:
        frozen = load_json(frozen_path)
    except (OSError, json.JSONDecodeError):
        return ["frozen_parameters_file_invalid"]

    requirements = config["calibration_freeze_requirements"]
    instruction = config["instruction_protocol"]["template"]
    identity = frozen.get("instruction_and_model", {})
    if requirements["prompt_and_model_revision_hashes_frozen"]:
        valid = bool(
            identity.get("frozen") is True
            and identity.get("instruction_template") == instruction
            and identity.get("instruction_sha256") == sha256_text(instruction)
            and identity.get("model_repository")
            and identity.get("model_revision")
        )
        # A freeze flag alone is not evidence; its source file must still hash-match.
        replay_path = ROOT / str(identity.get("neutral_replay_config", ""))
        valid = bool(
            valid
            and replay_path.is_file()
            and identity.get("neutral_replay_config_sha256")
            == sha256_file(replay_path)
        )
        if not valid:
            unresolved.append("prompt_and_model_revision_artifact_invalid")

    target = frozen.get("target_identity_calibration", {})
    if requirements["qwen_temperature_frozen"]:
        source_path = ROOT / str(target.get("source", ""))
        valid = bool(
            target.get("frozen") is True
            and float(target.get("temperature", 0.0)) > 0.0
            and int(target.get("calibration_seed_count", 0))
            >= int(requirements["minimum_open_container_calibration_episodes"])
            and source_path.is_file()
            and target.get("source_sha256") == sha256_file(source_path)
        )
        if not valid:
            unresolved.append("qwen_temperature_artifact_invalid")

    relation = frozen.get("relation_calibration", {})
    if requirements["relation_likelihoods_frozen"]:
        source_path = ROOT / str(relation.get("source", ""))
        try:
            source = load_json(source_path)
        except (OSError, json.JSONDecodeError):
            source = {}
        candidates = source.get("deployment_decision", {}).get(
            "component_candidates", {}
        )
        valid = bool(
            relation.get("frozen") is True
            and int(relation.get("calibration_seed_count", 0))
            >= int(requirements["minimum_open_container_calibration_episodes"])
            and source_path.is_file()
            and relation.get("source_sha256") == sha256_file(source_path)
            and candidates
            and all(candidates.values())
        )
        if not valid:
            unresolved.append("relation_likelihoods_frozen_artifact_invalid")

    section_by_requirement = {
        "action_conditioned_observation_models_frozen": (
            "action_conditioned_observation_model"
        ),
    }
    for requirement, section in section_by_requirement.items():
        if requirements[requirement] and (
            frozen.get(section, {}).get("frozen") is not True
        ):
            unresolved.append(f"{requirement}_artifact_invalid")
    if (
        requirements["task_cost_weights_frozen"]
        or requirements["commitment_gate_frozen"]
    ) and frozen.get("task_cost_and_commitment_gate", {}).get("frozen") is not True:
        unresolved.append("task_cost_or_commitment_gate_artifact_invalid")
    return unresolved


def audit(config: dict[str, Any]) -> dict[str, Any]:
    """Return a launch gate that remains closed on any protocol ambiguity."""
    failures: list[str] = []
    calibration = config["data_split"]["perception_and_policy_calibration_seeds"]
    calibration_seeds = set(
        range(
            int(calibration["start_inclusive"]),
            int(calibration["stop_inclusive"]) + 1,
        )
    )
    test_seeds = set(int(seed) for seed in config["data_split"]["reserved_test_seeds"])
    if calibration_seeds & test_seeds:
        failures.append("calibration_and_test_seed_overlap")
    if len(config["scenario_families"]) < 2:
        failures.append("fewer_than_two_scenario_families")
    if set(config["methods"]) != REQUIRED_METHODS:
        failures.append("required_method_set_mismatch")
    if set(config["ablations"]) != REQUIRED_ABLATIONS:
        failures.append("required_ablation_set_mismatch")
    instruction = config["instruction_protocol"]["template"].lower()
    leaked = sorted(word for word in RELATION_LEAK_WORDS if word in instruction)
    if leaked:
        failures.append(f"instruction_leaks_relation:{','.join(leaked)}")
    scale = config["evaluation_scale"]
    expected_main = (
        int(scale["test_seed_count_per_scenario"])
        * int(scale["scenario_count"])
        * int(scale["method_count_including_proposed"])
    )
    expected_ablation = (
        int(scale["test_seed_count_per_scenario"])
        * int(scale["scenario_count"])
        * int(scale["ablation_count"])
    )
    if int(scale["main_method_scenario_episodes"]) != expected_main:
        failures.append("main_evaluation_count_mismatch")
    if int(scale["ablation_scenario_episodes"]) != expected_ablation:
        failures.append("ablation_evaluation_count_mismatch")
    if int(scale["total_policy_evaluations"]) != expected_main + expected_ablation:
        failures.append("total_evaluation_count_mismatch")
    if int(scale["total_policy_evaluations"]) < 200:
        failures.append("paper_scale_below_200_method_scenario_evaluations")
    if config["training_performed"] is not False:
        failures.append("training_must_be_false")
    if config["testing_performed"] or config["reserved_test_seeds_used"]:
        failures.append("reserved_test_marked_used_before_freeze")

    freeze = config["calibration_freeze_requirements"]
    freeze_flags = {
        key: bool(value)
        for key, value in freeze.items()
        if key.endswith("_frozen")
    }
    # Reserved seeds are authorized only when every declared parameter is frozen.
    unresolved = sorted(key for key, value in freeze_flags.items() if not value)
    frozen_path = ROOT / freeze["frozen_parameters_path"]
    if not frozen_path.is_file():
        unresolved.append("frozen_parameters_file_missing")
    else:
        unresolved.extend(audit_frozen_artifact(config, frozen_path))
    protocol_ready = not failures and not unresolved
    return {
        "schema_version": "icra-evaluation-protocol-preflight-v1",
        "status": "passed" if protocol_ready else "blocked_before_reserved_test",
        "structural_failures": failures,
        "unresolved_freeze_requirements": sorted(set(unresolved)),
        "calibration_seed_count": len(calibration_seeds),
        "reserved_test_seed_count": len(test_seeds),
        "scenario_family_count": len(config["scenario_families"]),
        "planned_policy_evaluation_count": int(scale["total_policy_evaluations"]),
        "reserved_test_launch_authorized": protocol_ready,
        "training_performed": False,
        "testing_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "final_evaluation" / "icra_protocol_v1" / "preflight.json",
    )
    args = parser.parse_args()
    result = audit(load_json(args.config.resolve()))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
