#!/usr/bin/env python3
"""Fail-closed preflight audit for the paper-facing evaluation protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
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
    """Load one protocol or frozen-parameter artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(value: str) -> str:
    """Hash a canonical text representation."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash a file for protocol-freeze verification."""
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

    action_model = frozen.get("action_conditioned_observation_model", {})
    if requirements["action_conditioned_observation_models_frozen"]:
        source_fields = (
            ("open_container_source", "open_container_source_sha256"),
            ("scene_conditioned_source", "scene_conditioned_source_sha256"),
            ("full_observation_source", "full_observation_source_sha256"),
            ("joint_freeze_source", "joint_freeze_source_sha256"),
        )
        source_files_valid = True
        for path_field, hash_field in source_fields:
            source_path = ROOT / str(action_model.get(path_field, ""))
            source_files_valid = bool(
                source_files_valid
                and source_path.is_file()
                and action_model.get(hash_field) == sha256_file(source_path)
            )
        full_path = ROOT / str(action_model.get("full_observation_source", ""))
        joint_path = ROOT / str(action_model.get("joint_freeze_source", ""))
        try:
            full_source = load_json(full_path)
            joint_source = load_json(joint_path)
        except (OSError, json.JSONDecodeError):
            full_source = {}
            joint_source = {}
        cover_examples = full_source.get(
            "physical_cover_observation_validation", {}
        ).get("examples", [])
        negative_count = sum(
            str(example.get("state")) == "outside" for example in cover_examples
        )
        valid = bool(
            action_model.get("frozen") is True
            and source_files_valid
            and full_source.get("status") == "ready_to_freeze"
            and not full_source.get("deployment_gate", {}).get("blockers", [])
            and negative_count
            >= int(requirements["minimum_cover_negative_evidence_calibration_episodes"])
            and full_source.get("right_after_empty_validation", {}).get(
                "candidate_for_freeze"
            )
            is True
            and joint_source.get("status") == "ready_to_freeze"
            and not joint_source.get("reserved_seed_overlap", [])
            and joint_source.get("freeze_gate", {}).get(
                "task_cost_and_commitment_gate_candidate_for_freeze"
            )
            is True
        )
        if not valid:
            unresolved.append(
                "action_conditioned_observation_models_frozen_artifact_invalid"
            )
        post_cover_path_value = action_model.get("post_cover_view_source")
        if post_cover_path_value is not None:
            post_cover_path = ROOT / str(post_cover_path_value)
            try:
                post_cover = load_json(post_cover_path)
            except (OSError, json.JSONDecodeError):
                post_cover = {}
            minimum = int(
                requirements.get(
                    "minimum_post_cover_view_calibration_episodes", 1
                )
            )
            post_cover_valid = bool(
                post_cover_path.is_file()
                and action_model.get("post_cover_view_source_sha256")
                == sha256_file(post_cover_path)
                and post_cover.get("status") == "completed"
                and int(post_cover.get("episode_count", 0)) >= minimum
                and post_cover.get("training_performed") is False
                and post_cover.get("testing_performed") is False
                and post_cover.get("reserved_test_seeds_used") is False
                and set(action_model.get("post_cover_reobservation_actions", []))
                == {"viewpoint_close_high", "viewpoint_right"}
            )
            if not post_cover_valid:
                unresolved.append(
                    "post_cover_view_observation_model_artifact_invalid"
                )
    if (
        requirements["task_cost_weights_frozen"]
        or requirements["commitment_gate_frozen"]
    ):
        task = frozen.get("task_cost_and_commitment_gate", {})
        source_path = ROOT / str(task.get("source", ""))
        try:
            source = load_json(source_path)
        except (OSError, json.JSONDecodeError):
            source = {}
        selected = source.get("modal_candidate", {})
        valid = bool(
            task.get("frozen") is True
            and source_path.is_file()
            and task.get("source_sha256") == sha256_file(source_path)
            and source.get("status") == "ready_to_freeze"
            and float(task.get("noncompletion_cost", -1.0))
            == float(selected.get("noncompletion_cost", -2.0))
            and float(task.get("minimum_grasp_success_probability", -1.0))
            == float(selected.get("minimum_grasp_success_probability", -2.0))
            and float(task.get("wrong_commitment_weight", -1.0))
            == float(selected.get("wrong_commitment_weight", -2.0))
        )
        if not valid:
            unresolved.append("task_cost_or_commitment_gate_artifact_invalid")
    if frozen.get("reserved_test_launch_authorized") is not True:
        unresolved.append("frozen_artifact_does_not_authorize_reserved_test")
    return unresolved


def audit_v5(config: dict[str, Any]) -> dict[str, Any]:
    """Audit the task-risk multi-step protocol without v1 schema assumptions."""
    failures: list[str] = []
    split = config.get("data_split", {})
    calibration_seeds = {
        int(seed) for seed in split.get("task_risk_policy_calibration_seeds", [])
    }
    test_seeds = {int(seed) for seed in split.get("reserved_test_seeds", [])}
    if not calibration_seeds:
        failures.append("task_risk_calibration_split_missing")
    if len(test_seeds) != 20:
        failures.append("reserved_test_seed_count_must_be_20")
    if calibration_seeds & test_seeds:
        failures.append("calibration_and_test_seed_overlap")
    if len(config.get("scenario_families", [])) != 2:
        failures.append("scenario_family_count_must_be_two")
    instruction = str(config.get("instruction_protocol", {}).get("template", "")).lower()
    leaked = sorted(word for word in RELATION_LEAK_WORDS if word in instruction)
    if leaked:
        failures.append(f"instruction_leaks_relation:{','.join(leaked)}")
    if config.get("status") != "frozen_before_reserved_test":
        failures.append("protocol_not_frozen_before_test")
    if config.get("reserved_test_launch_authorized") is not True:
        failures.append("protocol_does_not_authorize_reserved_test")
    if config.get("training_performed") is not False:
        failures.append("training_must_be_false")
    if config.get("testing_performed") or config.get("reserved_test_seeds_used"):
        failures.append("reserved_test_marked_used_before_launch")

    freeze = config.get("calibration_freeze_requirements", {})
    unresolved = sorted(
        key
        for key, value in freeze.items()
        if key.endswith("_frozen") and value is not True
    )
    frozen_path = ROOT / str(freeze.get("frozen_parameters_path", ""))
    try:
        frozen = load_json(frozen_path)
    except (OSError, json.JSONDecodeError):
        frozen = {}
        unresolved.append("frozen_parameters_file_invalid")
    if frozen:
        if frozen.get("status") != "frozen_before_reserved_test":
            unresolved.append("frozen_artifact_status_invalid")
        if frozen.get("reserved_test_launch_authorized") is not True:
            unresolved.append("frozen_artifact_does_not_authorize_reserved_test")
        if {int(seed) for seed in frozen.get("reserved_test_seeds", [])} != test_seeds:
            unresolved.append("frozen_reserved_test_seed_set_mismatch")
        method = frozen.get("method", {})
        config_path = ROOT / str(method.get("calibration_config", ""))
        result_path = ROOT / str(method.get("calibration_result", ""))
        if not config_path.is_file() or method.get("calibration_config_sha256") != sha256_file(config_path):
            unresolved.append("task_risk_calibration_config_hash_invalid")
        if not result_path.is_file() or method.get("calibration_result_sha256") != sha256_file(result_path):
            unresolved.append("task_risk_calibration_result_hash_invalid")
        try:
            calibration_config = load_json(config_path)
            calibration_result = load_json(result_path)
        except (OSError, json.JSONDecodeError):
            calibration_config = {}
            calibration_result = {}
        if {
            int(seed) for seed in calibration_config.get("calibration_seeds", [])
        } != calibration_seeds:
            unresolved.append("task_risk_calibration_seed_set_mismatch")
        if calibration_config.get("reserved_test_seeds") != sorted(test_seeds):
            unresolved.append("task_risk_reserved_seed_set_mismatch")
        if not (
            calibration_result.get("status") == "completed"
            and calibration_result.get("calibration_performed") is True
            and calibration_result.get("testing_performed") is False
            and calibration_result.get("reserved_test_seeds_used") is False
            and calibration_result.get("training_performed") is False
        ):
            unresolved.append("task_risk_calibration_result_scope_invalid")
        feature_names = set(method.get("feature_names", []))
        excluded = set(
            config.get("frozen_calibration", {}).get(
                "excluded_nuisance_features", []
            )
        )
        if not feature_names or feature_names & excluded:
            unresolved.append("task_risk_feature_freeze_invalid")
        if int(method.get("maximum_observation_actions", 0)) != 2:
            unresolved.append("task_risk_horizon_not_two")
        if method.get("training_performed") is not False:
            unresolved.append("task_risk_model_training_must_be_false")

    protocol_ready = not failures and not unresolved
    return {
        "schema_version": "icra-v5-evaluation-protocol-preflight-v1",
        "status": "passed" if protocol_ready else "blocked_before_reserved_test",
        "structural_failures": failures,
        "unresolved_freeze_requirements": sorted(set(unresolved)),
        "calibration_seed_count": len(calibration_seeds),
        "reserved_test_seed_count": len(test_seeds),
        "protocol_ready": protocol_ready,
    }


def audit_v11(config: dict[str, Any]) -> dict[str, Any]:
    """Audit the three-family v11 protocol and its frozen sources."""
    failures: list[str] = []
    split = config.get("data_split", {})
    calibration = set(range(
        int(split.get("calibration_seed_first", 0)),
        int(split.get("calibration_seed_last", -1)) + 1,
    ))
    test_seeds = {int(seed) for seed in split.get("reserved_test_seeds", [])}
    if len(calibration) != 24:
        failures.append("calibration_seed_count_must_be_24")
    if len(test_seeds) != 60:
        failures.append("reserved_test_seed_count_must_be_60")
    if calibration & test_seeds:
        failures.append("calibration_and_test_seed_overlap")
    if len(config.get("scenario_families", [])) != 3:
        failures.append("scenario_family_count_must_be_three")
    assignment = config.get("reserved_test_assignment", [])
    assigned_seeds = [int(row["seed"]) for row in assignment]
    if set(assigned_seeds) != test_seeds or len(assigned_seeds) != len(set(assigned_seeds)):
        failures.append("reserved_test_assignment_mismatch")
    family_counts = Counter(str(row["family"]) for row in assignment)
    if set(family_counts.values()) != {20} or len(family_counts) != 3:
        failures.append("reserved_test_families_must_have_20_seeds_each")
    instruction = str(config.get("instruction_protocol", {}).get("template", "")).lower()
    leaked = sorted(word for word in RELATION_LEAK_WORDS if word in instruction)
    if leaked:
        failures.append(f"instruction_leaks_relation:{','.join(leaked)}")
    if config.get("status") != "frozen_before_reserved_test":
        failures.append("protocol_not_frozen_before_test")
    if config.get("reserved_test_launch_authorized") is not True:
        failures.append("protocol_does_not_authorize_reserved_test")
    if config.get("training_performed") is not False:
        failures.append("training_must_be_false")
    if config.get("testing_performed") or config.get("reserved_test_seeds_used"):
        failures.append("reserved_test_marked_used_before_launch")

    unresolved: list[str] = []
    frozen_path = ROOT / str(config.get("frozen_parameters_path", ""))
    try:
        frozen = load_json(frozen_path)
    except (OSError, json.JSONDecodeError):
        frozen = {}
        unresolved.append("frozen_parameters_file_invalid")
    if frozen:
        if frozen.get("status") != "frozen_before_reserved_test":
            unresolved.append("frozen_artifact_status_invalid")
        if frozen.get("reserved_test_launch_authorized") is not True:
            unresolved.append("frozen_artifact_does_not_authorize_reserved_test")
        frozen_range = frozen.get("reserved_test_seeds", {})
        frozen_seeds = set(range(
            int(frozen_range.get("first", 0)),
            int(frozen_range.get("last", -1)) + 1,
        ))
        if frozen_seeds != test_seeds or frozen_range.get("opened") is not False:
            unresolved.append("frozen_reserved_test_seed_set_mismatch")
        for section in (
            "target_identity_calibration",
            "relation_belief_calibration",
            "action_conditioned_future_belief",
            "nested_calibration_result",
            "runtime_integration_replay",
        ):
            item = frozen.get(section, {})
            source = ROOT / str(item.get("source", ""))
            if not source.is_file() or item.get("source_sha256") != sha256_file(source):
                unresolved.append(f"{section}_source_hash_invalid")
        for name, item in frozen.get("implementation", {}).items():
            source = ROOT / str(item.get("path", ""))
            if not source.is_file() or item.get("sha256") != sha256_file(source):
                unresolved.append(f"implementation_hash_invalid:{name}")
        if frozen.get("training_performed") is not False:
            unresolved.append("frozen_training_flag_invalid")
        if frozen.get("testing_performed") or frozen.get("reserved_test_seeds_used"):
            unresolved.append("frozen_test_flag_invalid")

    ready = not failures and not unresolved
    return {
        "schema_version": "icra-v11-evaluation-protocol-preflight-v1",
        "status": "passed" if ready else "blocked_before_reserved_test",
        "structural_failures": failures,
        "unresolved_freeze_requirements": sorted(set(unresolved)),
        "calibration_seed_count": len(calibration),
        "reserved_test_seed_count": len(test_seeds),
        "scenario_family_counts": dict(sorted(family_counts.items())),
        "protocol_ready": ready,
    }


def audit(config: dict[str, Any]) -> dict[str, Any]:
    """Return a launch gate that remains closed on any protocol ambiguity."""
    if config.get("schema_version") == "icra-v15b-final-evaluation-protocol-v1":
        failures: list[str] = []
        split = config.get("data_split", {})
        calibration = {int(seed) for seed in split.get("calibration_seeds", [])}
        reserved = {int(seed) for seed in split.get("reserved_test_seeds", [])}
        if len(calibration) != 48:
            failures.append("calibration_seed_count_must_be_48")
        if len(reserved) != 60:
            failures.append("reserved_test_seed_count_must_be_60")
        if calibration & reserved:
            failures.append("calibration_and_test_seed_overlap")
        if len(config.get("scenario_families", [])) != 6:
            failures.append("scenario_family_count_must_be_six")
        if config.get("status") != "frozen_before_reserved_test":
            failures.append("protocol_not_frozen_before_test")
        if config.get("reserved_test_launch_authorized") is not True:
            failures.append("protocol_does_not_authorize_reserved_test")
        if config.get("training_performed") is not False:
            failures.append("training_must_be_false")
        freeze_path = ROOT / str(
            config.get("calibration_freeze_requirements", {}).get(
                "frozen_parameters_path", ""
            )
        )
        try:
            frozen = load_json(freeze_path)
        except (OSError, json.JSONDecodeError):
            frozen = {}
            failures.append("frozen_manifest_invalid")
        if frozen:
            if frozen.get("status") != "frozen_ready_for_reserved_test":
                failures.append("frozen_manifest_not_ready")
            if {int(seed) for seed in frozen.get("reserved_test_seeds", [])} != reserved:
                failures.append("frozen_reserved_test_seed_set_mismatch")
            if frozen.get("reserved_test_opened") is not False:
                failures.append("frozen_manifest_test_already_opened")
            if frozen.get("training_performed") is not False:
                failures.append("frozen_training_flag_invalid")
            smoke = frozen.get("nonreserved_live_pretest_smoke", {})
            if smoke.get("scientific_episode_success") is not True:
                failures.append("nonreserved_live_smoke_not_passed")
            if frozen.get("planner", {}).get("fixed_confidence_threshold_used") is not False:
                failures.append("fixed_confidence_threshold_not_allowed")
            for artifact in frozen.get("artifacts", {}).values():
                path = Path(str(artifact.get("path", "")))
                if not path.is_file() or artifact.get("sha256") != sha256_file(path):
                    failures.append("frozen_artifact_hash_mismatch")
            for relative, expected in frozen.get("input_hashes", {}).items():
                path = ROOT / relative
                if not path.is_file() or expected != sha256_file(path):
                    failures.append(f"frozen_input_hash_mismatch:{relative}")
        ready = not failures
        return {
            "schema_version": "icra-v15b-final-evaluation-preflight-v1",
            "status": "passed" if ready else "blocked_before_reserved_test",
            "structural_failures": sorted(set(failures)),
            "calibration_seed_count": len(calibration),
            "reserved_test_seed_count": len(reserved),
            "scenario_family_count": len(config.get("scenario_families", [])),
            "reserved_test_launch_authorized": ready,
        }
    if config.get("schema_version") == "icra-simulation-evaluation-protocol-v5":
        return audit_v5(config)
    if config.get("schema_version") == "icra-v11-final-evaluation-protocol-v1":
        return audit_v11(config)
    failures: list[str] = []
    calibration = config["data_split"]["perception_and_policy_calibration_seeds"]
    calibration_seeds = set(
        range(
            int(calibration["start_inclusive"]),
            int(calibration["stop_inclusive"]) + 1,
        )
    )
    test_seeds = set(int(seed) for seed in config["data_split"]["reserved_test_seeds"])
    supplemental_calibration_seeds = {
        int(seed)
        for seed in config["data_split"].get(
            "supplemental_physics_calibration_seeds", []
        )
    }
    if calibration_seeds & test_seeds:
        failures.append("calibration_and_test_seed_overlap")
    if supplemental_calibration_seeds & test_seeds:
        failures.append("supplemental_calibration_and_test_seed_overlap")
    if supplemental_calibration_seeds & calibration_seeds:
        failures.append("supplemental_calibration_duplicates_primary_range")
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
        frozen = load_json(frozen_path)
        frozen_reserved = {
            int(seed) for seed in frozen.get("reserved_test_seeds", test_seeds)
        }
        if frozen_reserved != test_seeds:
            unresolved.append("frozen_reserved_test_seed_set_mismatch")
    protocol_ready = not failures and not unresolved
    return {
        "schema_version": "icra-evaluation-protocol-preflight-v1",
        "status": "passed" if protocol_ready else "blocked_before_reserved_test",
        "structural_failures": failures,
        "unresolved_freeze_requirements": sorted(set(unresolved)),
        "calibration_seed_count": len(calibration_seeds),
        "supplemental_physics_calibration_seed_count": len(
            supplemental_calibration_seeds
        ),
        "reserved_test_seed_count": len(test_seeds),
        "scenario_family_count": len(config["scenario_families"]),
        "planned_policy_evaluation_count": int(scale["total_policy_evaluations"]),
        "reserved_test_launch_authorized": protocol_ready,
        "training_performed": False,
        "testing_performed": False,
    }


def main() -> None:
    """Audit the frozen evaluation protocol before any reserved test run."""
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
