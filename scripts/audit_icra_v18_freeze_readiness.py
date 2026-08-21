#!/usr/bin/env python3
"""Audit whether the V18 calibration candidate is ready to freeze for testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = (
    ROOT / "outputs/calibration/icra_v18_persistent_negative_evidence_candidate"
)
DEFAULT_PHYSICAL = (
    ROOT / "outputs/calibration/icra_v15_physical_execution/result.json"
)
DEFAULT_DEVELOPMENT = (
    ROOT
    / "outputs/development/icra_v18_persistent_negative_evidence_32episode"
    / "policy_evaluation/result.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def check(name: str, passed: bool, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(passed), "evidence": evidence}


def run(
    candidate_root: Path,
    physical_result_path: Path,
    development_result_path: Path,
    output: Path,
) -> dict[str, Any]:
    calibration = load_json(candidate_root / "result.json")
    model = load_json(candidate_root / "calibration_candidate_model.json")
    cost_sensitivity = load_json(candidate_root / "cost_sensitivity.json")
    physical = load_json(physical_result_path)
    development = load_json(development_result_path)

    summary = calibration["summary"]
    hypothesis_support = summary["joint_hypothesis_support"]
    expected_hypotheses = set(model["semantic_hypotheses"])
    remove_cover = model["information_actions"]["remove_cover"]
    failure_transition = remove_cover["next_task_state_by_outcome"]["covered"].get(
        "removal_failed"
    )
    viewpoints_open_only = all(
        model["information_actions"][action]["allowed_task_states"] == ["open"]
        for action in ("viewpoint_right", "viewpoint_close_high")
    )
    nominal = cost_sensitivity["nominal_setting_validation"]["summary"]
    robust_settings = [
        row
        for row in cost_sensitivity["settings"]
        if row["summary"]["semantic_task_success_count"]
        == cost_sensitivity["episode_count"]
        and row["summary"]["wrong_commitment_count"] == 0
        and row["summary"]["target_absent_safe_deferral_count"]
        == hypothesis_support.get("target_absent|not_applicable", 0)
    ]
    proposed = development["summaries"][
        "proposed_task_risk_aware_joint_belief_mpc"
    ]
    condition_coverage = summary.get(
        "outside_reobservation_condition_coverage", {}
    )
    designed_action_audit_passed = bool(
        summary.get("designed_behavior_pass_count")
        == summary.get("episode_count")
        and condition_coverage.get("resolved_center_episode_count", 0) > 0
        and condition_coverage.get("unresolved_center_episode_count", 0) > 0
        and condition_coverage.get("condition_consistent_episode_count")
        == condition_coverage.get("episode_count")
    )

    passed_checks = [
        check(
            "persistent_semantic_observation_contract",
            calibration.get("persistent_semantic_symbols") is True,
            calibration.get("persistent_semantic_symbols"),
        ),
        check(
            "all_joint_hypotheses_have_calibration_support",
            set(hypothesis_support) == expected_hypotheses
            and min(hypothesis_support.values()) >= 3,
            hypothesis_support,
        ),
        check(
            "episode_disjoint_calibration_replay",
            summary["episode_count"] == 48
            and summary["semantic_decision_correct_count"] == 48
            and summary["wrong_commitment_count"] == 0
            and summary["target_absent_safe_deferral_count"] == 6,
            {
                "episodes": summary["episode_count"],
                "correct": summary["semantic_decision_correct_count"],
                "wrong_commitments": summary["wrong_commitment_count"],
                "safe_absent_deferrals": summary[
                    "target_absent_safe_deferral_count"
                ],
            },
        ),
        check(
            "remove_cover_failure_has_safe_state_transition",
            "removal_failed" in remove_cover["outcomes"]
            and failure_transition == "covered",
            {"outcomes": remove_cover["outcomes"], "failure_next_state": failure_transition},
        ),
        check(
            "covered_state_viewpoint_model_not_required",
            viewpoints_open_only
            and remove_cover["allowed_task_states"] == ["covered"],
            {
                "viewpoints_allowed_in": "open",
                "remove_cover_allowed_in": remove_cover["allowed_task_states"],
            },
        ),
        check(
            "conditional_rg6_physics_calibrated",
            physical["status"] == "passed_calibration"
            and physical["episode_count"] >= 18
            and physical["target_attachment_used"] is False
            and physical["target_pose_copying_used"] is False,
            {
                "episodes": physical["episode_count"],
                "successes": physical["success_count"],
                "posterior_execution_success": physical[
                    "posterior_mean_used_by_planner"
                ],
            },
        ),
        check(
            "task_cost_sensitivity_completed",
            cost_sensitivity["status"] == "completed"
            and nominal["semantic_task_success_count"] == 48
            and nominal["wrong_commitment_count"] == 0
            and len(robust_settings) >= 2,
            {
                "nominal_successes": nominal["semantic_task_success_count"],
                "nominal_wrong_commitments": nominal["wrong_commitment_count"],
                "robust_settings": len(robust_settings),
                "total_settings": len(cost_sensitivity["settings"]),
            },
        ),
        check(
            "balanced_development_policy_diagnostic",
            proposed["episode_count"] == 32
            and proposed["semantic_task_success_count"] == 32
            and proposed["wrong_commitment_count"] == 0,
            {
                "episodes": proposed["episode_count"],
                "successes": proposed["semantic_task_success_count"],
                "wrong_commitments": proposed["wrong_commitment_count"],
                "mean_realized_task_cost": proposed["mean_realized_task_cost"],
            },
        ),
        check(
            "observation_conditioned_action_difficulty_audit",
            designed_action_audit_passed,
            {
                "condition_consistent_episodes": summary.get(
                    "designed_behavior_pass_count"
                ),
                "episodes": summary.get("episode_count"),
                "outside_condition_coverage": condition_coverage,
            },
        ),
        check(
            "reserved_test_remains_unopened",
            calibration["reserved_test_seeds_used"] is False
            and development["valid_for_final_evaluation"] is False,
            {
                "calibration_used_reserved_test": calibration[
                    "reserved_test_seeds_used"
                ],
                "development_valid_for_final_evaluation": development[
                    "valid_for_final_evaluation"
                ],
            },
        ),
    ]

    blockers = [
        {
            "name": "remove_cover_failure_probability_not_empirically_measured",
            "requires_gpu": True,
            "reason": (
                "The planner now handles removal_failed safely, but the calibration "
                "split contains no verified physical removal failure frequency."
            ),
        },
        {
            "name": "v18_candidate_live_end_to_end_smoke_not_run",
            "requires_gpu": True,
            "reason": (
                "The corrected candidate has been validated by cached replay only; "
                "one non-reserved live Isaac Sim episode is required before freezing."
            ),
        },
        {
            "name": "sam_version_selection_not_closed",
            "requires_gpu": True,
            "reason": (
                "The professor's SAM2.1 versus newer SAM comparison must be resolved "
                "on the same held-out perception images before final perception freeze. "
                "The official SAM3.1 checkpoint is gated and the current account does "
                "not yet have download access."
            ),
            "external_access_required": True,
        },
    ]
    failed_checks = [row for row in passed_checks if not row["passed"]]
    for row in failed_checks:
        blockers.append(
            {
                "name": f"failed_check:{row['name']}",
                "requires_gpu": False,
                "reason": row["evidence"],
            }
        )

    result = {
        "schema_version": "icra-v18-freeze-readiness-audit-v1",
        "status": "blocked_pending_validation" if blockers else "ready_to_freeze",
        "candidate_root": str(candidate_root.resolve()),
        "checks": passed_checks,
        "passed_check_count": sum(row["passed"] for row in passed_checks),
        "check_count": len(passed_checks),
        "blocking_reasons": blockers,
        "candidate_frozen": False,
        "reserved_test_opened": False,
        "training_performed": False,
        "calibration_performed": True,
        "testing_performed": False,
        "valid_for_final_evaluation": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--physical-result", type=Path, default=DEFAULT_PHYSICAL)
    parser.add_argument(
        "--development-result", type=Path, default=DEFAULT_DEVELOPMENT
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    candidate_root = args.candidate_root.resolve()
    output = (
        args.output.resolve()
        if args.output
        else candidate_root / "freeze_readiness.json"
    )
    result = run(
        candidate_root,
        args.physical_result.resolve(),
        args.development_result.resolve(),
        output,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "checks": f"{result['passed_check_count']}/{result['check_count']}",
                "blocking_reasons": [
                    row["name"] for row in result["blocking_reasons"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
