#!/usr/bin/env python3
"""Validate the five-scenario evaluation protocol before reserved testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/research/final_evaluation_protocol.json"
REQUIRED_SCENARIOS = {
    "visible_open",
    "partially_occluded",
    "covered_container",
    "ambiguous_inside_outside",
    "target_absent",
}
REQUIRED_ACTIONS = {
    "viewpoint_right",
    "viewpoint_close_high",
    "remove_cover",
    "grasp",
    "defer",
}
REQUIRED_ABLATIONS = {
    "no_joint_belief",
    "no_calibration",
    "no_negative_evidence",
    "no_action_conditioned_future_belief",
    "no_task_risk_cost",
    "no_persistent_tracking",
    "no_scene_conditioned_view_model",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def reserved_seeds(protocol: dict[str, Any]) -> set[int]:
    split = protocol["data_split"]
    start, stop = [int(value) for value in split["reserved_test_seed_range"]]
    if stop < start:
        raise ValueError("Reserved-test seed range is invalid")
    seeds = set(range(start, stop + 1))
    if len(seeds) != int(split["reserved_test_episode_count"]):
        raise ValueError("Reserved-test count does not match the seed range")
    return seeds


def audit(protocol: dict[str, Any]) -> dict[str, Any]:
    """Return structural errors and the state of the reserved-test gate."""
    failures: list[str] = []
    unresolved: list[str] = []

    if protocol.get("schema_version") != "five-scenario-final-evaluation-protocol-v1":
        failures.append("unsupported_schema_version")
    if set(protocol.get("headline_scenarios", [])) != REQUIRED_SCENARIOS:
        failures.append("scenario_set_mismatch")
    if set(protocol.get("action_set", [])) != REQUIRED_ACTIONS:
        failures.append("action_set_mismatch")
    if set(protocol.get("ablations", [])) != REQUIRED_ABLATIONS:
        failures.append("ablation_set_mismatch")
    if "proposed_task_risk_aware_joint_belief_mpc" not in protocol.get("methods", []):
        failures.append("proposed_method_missing")

    try:
        test_seeds = reserved_seeds(protocol)
    except (KeyError, TypeError, ValueError) as error:
        failures.append(f"reserved_split_invalid:{error}")
        test_seeds = set()
    split = protocol.get("data_split", {})
    development_range = split.get("development_seed_range", [])
    if len(development_range) != 2:
        failures.append("development_split_invalid")
    else:
        development = set(range(int(development_range[0]), int(development_range[1]) + 1))
        if development & test_seeds:
            failures.append("development_and_test_seed_overlap")

    instruction = protocol.get("instruction_protocol", {})
    for key in (
        "scenario_label_hidden_from_policy",
        "ground_truth_hidden_from_policy",
        "required_action_hidden_from_policy",
    ):
        if instruction.get(key) is not True:
            failures.append(f"information_leak_guard_missing:{key}")
    if protocol.get("training_performed") is not False:
        failures.append("training_flag_must_be_false")

    if protocol.get("status") != "frozen_before_untouched_test":
        unresolved.append("protocol_not_frozen")
    if protocol.get("reserved_test_launch_authorized") is not True:
        unresolved.append("reserved_test_not_authorized")
    if protocol.get("testing_performed") is not False:
        failures.append("testing_already_marked_performed")
    if protocol.get("reserved_test_seeds_used") is not False:
        failures.append("reserved_test_already_opened")

    ready = not failures and not unresolved
    return {
        "schema_version": "evaluation-protocol-audit-v1",
        "status": "passed" if ready else "blocked_before_reserved_test",
        "structural_failures": sorted(failures),
        "unresolved_freeze_requirements": sorted(unresolved),
        "reserved_test_episode_count": len(test_seeds),
        "scenario_family_count": len(protocol.get("headline_scenarios", [])),
        "reserved_test_launch_authorized": ready,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = audit(load_json(args.config.resolve()))
    print(json.dumps(result, indent=2))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
