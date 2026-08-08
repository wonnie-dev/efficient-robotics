#!/usr/bin/env python3
"""Count independent covered-container calibration episodes fail-closed.

Repeated engineering runs of one seed are useful for debugging but are not
independent calibration episodes.  This audit therefore keeps at most one
successful integrated episode for each (outcome family, seed) pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = (
    ROOT / "configs" / "research" / "icra_simulation_evaluation_protocol_v1.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "final_evaluation"
    / "icra_protocol_v1"
    / "cover_calibration_readiness.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def physics_gate_passes(
    execution: dict[str, Any],
    gates: dict[str, Any],
    *,
    expected_target_path: str,
) -> bool:
    force = execution.get("contact_force_n", {}).get("maximum", {})
    penetration = execution.get("maximum_contact_penetration_m", {})
    stability = execution.get("target_gripper_relative_stability", {})
    return bool(
        execution.get("status") == "completed"
        and execution.get("manipulation_target_path") == expected_target_path
        and float(execution.get("verified_lift_delta_m", 0.0))
        >= float(gates["minimum_verified_lift_m"])
        and execution.get("bilateral_contact_before_lift") is True
        and execution.get("contact_maintained_after_lift") is True
        and execution.get("contact_maintained_before_release") is True
        and max(float(force.get("left", 1e9)), float(force.get("right", 1e9)))
        <= float(gates["maximum_force_per_finger_n"])
        and max(
            float(penetration.get("left", 1e9)),
            float(penetration.get("right", 1e9)),
        )
        <= float(gates["maximum_penetration_m"])
        and float(stability.get("maximum_translation_m", 1e9))
        <= float(gates["maximum_target_relative_translation_m"])
        and float(stability.get("maximum_rotation_rad", 1e9))
        <= float(gates["maximum_target_relative_rotation_rad"])
        and not execution.get("unexpected_environment_pairs")
        and not execution.get("unexpected_target_environment_pairs")
        and execution.get("target_attachment_used") is False
        and execution.get("target_pose_copying_used") is False
        and execution.get("finite_final_joint_state") is True
    )


def successful_positive(
    path: Path,
    result: dict[str, Any],
    gates: dict[str, Any],
) -> bool:
    return bool(
        path.name == "remove_cover_smoke_result.json"
        and result.get("status") == "completed"
        and result.get("replanning_performed") is True
        and result.get("post_remove_replanned_action") == "grasp_inside"
        and result.get("final_grasp_executed") is True
        and physics_gate_passes(
            result.get("cover_removal_execution") or {},
            gates,
            expected_target_path="/World/OpenContainer/CalibrationCover",
        )
        and physics_gate_passes(
            result.get("final_grasp_execution") or {},
            gates,
            expected_target_path="/World/TargetRed",
        )
    )


def successful_negative(
    path: Path,
    result: dict[str, Any],
    gates: dict[str, Any],
) -> bool:
    return bool(
        path.name == "negative_evidence_live_result.json"
        and result.get("status") == "completed"
        and result.get("post_remove_observation") == "empty_container"
        and int(result.get("belief_update_count", 0)) >= 2
        and result.get("action_sequence")
        == ["remove_cover", "viewpoint_right", "grasp_outside"]
        and physics_gate_passes(
            result.get("cover_removal_execution") or {},
            gates,
            expected_target_path="/World/OpenContainer/CalibrationCover",
        )
        and physics_gate_passes(
            result.get("final_grasp_execution") or {},
            gates,
            expected_target_path="/World/TargetRed",
        )
    )


def collect(
    output_root: Path,
    gates: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    accepted: dict[tuple[str, int], dict[str, Any]] = {}
    patterns = (
        "**/remove_cover_smoke_result.json",
        "**/negative_evidence_live_result.json",
    )
    for pattern in patterns:
        for path in sorted(output_root.glob(pattern)):
            try:
                result = load_json(path)
            except (OSError, json.JSONDecodeError):
                continue
            seed = result.get("seed")
            if not isinstance(seed, int):
                continue
            outcome = None
            if successful_positive(path, result, gates):
                outcome = "target_inside_after_cover_removal"
            elif successful_negative(path, result, gates):
                outcome = "empty_container_negative_evidence"
            if outcome is None:
                continue
            key = (outcome, seed)
            accepted[key] = {
                "seed": seed,
                "outcome_family": outcome,
                "result_path": str(path.resolve()),
                "runtime_seconds": result.get("runtime_seconds"),
                "valid_for_final_evaluation": False,
            }
    grouped = {
        "target_inside_after_cover_removal": [],
        "empty_container_negative_evidence": [],
    }
    for row in accepted.values():
        grouped[row["outcome_family"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["seed"])
    return grouped


def audit(protocol: dict[str, Any], output_root: Path) -> dict[str, Any]:
    grouped = collect(output_root, protocol["physical_success_gates"])
    minimum_negative = int(
        protocol["calibration_freeze_requirements"][
            "minimum_cover_negative_evidence_calibration_episodes"
        ]
    )
    negative_count = len(grouped["empty_container_negative_evidence"])
    return {
        "schema_version": "cover-calibration-readiness-v1",
        "status": (
            "minimum_negative_evidence_count_met"
            if negative_count >= minimum_negative
            else "blocked_insufficient_independent_negative_evidence"
        ),
        "independence_unit": "unique_outcome_family_and_seed",
        "repeated_runs_of_same_seed_counted_once": True,
        "physical_success_gates": protocol["physical_success_gates"],
        "minimum_negative_evidence_episodes": minimum_negative,
        "independent_negative_evidence_episode_count": negative_count,
        "independent_positive_cover_episode_count": len(
            grouped["target_inside_after_cover_removal"]
        ),
        "episodes": grouped,
        "training_performed": False,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument(
        "--live-output-root",
        type=Path,
        default=ROOT / "outputs" / "live_pipeline",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = audit(
        load_json(args.protocol.resolve()),
        args.live_output_root.resolve(),
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
