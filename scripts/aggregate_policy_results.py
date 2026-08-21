#!/usr/bin/env python3
"""Aggregate Scenario A and B policy replays with paired uncertainty estimates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OPEN_RESULT = ROOT / "outputs/final_evaluation/icra_protocol_v1/policy/open_container/result.json"
COVERED_RESULT = ROOT / "outputs/final_evaluation/icra_protocol_v1/policy/covered_container/result.json"
OUTPUT_ROOT = ROOT / "outputs/final_evaluation/icra_protocol_v1/policy/combined"
PROPOSED = "proposed_task_risk_aware_action_conditioned_belief_mpc"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def wilson(successes: int, count: int, z: float = 1.959963984540054) -> dict[str, float]:
    if count <= 0:
        raise ValueError("Wilson interval requires at least one record")
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
    ) / denominator
    return {"estimate": proportion, "lower": center - half, "upper": center + half}


def exact_mcnemar(first: list[bool], second: list[bool]) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError("Paired results must have equal length")
    first_only = sum(a and not b for a, b in zip(first, second))
    second_only = sum(b and not a for a, b in zip(first, second))
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, k) for k in range(min(first_only, second_only) + 1)
        ) / (2.0 ** discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "first_only": first_only,
        "second_only": second_only,
        "discordant_count": discordant,
        "two_sided_exact_p_value": p_value,
    }


def paired_bootstrap_difference(
    first: list[float],
    second: list[float],
    *,
    resamples: int = 10000,
    seed: int = 0,
) -> dict[str, float | int]:
    if len(first) != len(second) or not first:
        raise ValueError("Paired bootstrap requires equal non-empty inputs")
    differences = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(resamples, len(differences)))
    draws = differences[indices].mean(axis=1)
    return {
        "mean_difference_first_minus_second": float(differences.mean()),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "resamples": resamples,
        "random_seed": seed,
    }


def canonical_rows(open_result: dict[str, Any], covered_result: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    policies = set(open_result["episodes"]) & set(covered_result["episodes"])
    rows: dict[str, list[dict[str, Any]]] = {}
    for policy in sorted(policies):
        combined = []
        for row in open_result["episodes"][policy]:
            combined.append(
                {
                    "scenario": "A_open_container_active_view",
                    "seed": int(row["seed"]),
                    "success": bool(row["joint_target_relation_success"]),
                    "wrong_commitment": bool(row["wrong_commitment"]),
                    "decision_cost": float(row["decision_cost"]),
                }
            )
        for row in covered_result["episodes"][policy]:
            combined.append(
                {
                    "scenario": "B_removable_cover_negative_evidence",
                    "seed": int(row["seed"]),
                    "success": bool(row["task_success_proxy"]),
                    "wrong_commitment": bool(row["wrong_commitment"]),
                    "decision_cost": float(row["decision_cost"]),
                }
            )
        combined.sort(key=lambda item: (item["scenario"], item["seed"]))
        rows[policy] = combined
    return rows


def aggregate(open_path: Path, covered_path: Path, output_root: Path) -> dict[str, Any]:
    open_result = load_json(open_path)
    covered_result = load_json(covered_path)
    if open_result["status"] != "completed" or covered_result["status"] != "completed":
        raise ValueError("Both scenario evaluations must be complete")
    rows = canonical_rows(open_result, covered_result)
    if PROPOSED not in rows:
        raise ValueError("Proposed policy is missing")
    summaries = {}
    for policy, policy_rows in rows.items():
        count = len(policy_rows)
        success_count = sum(row["success"] for row in policy_rows)
        wrong_count = sum(row["wrong_commitment"] for row in policy_rows)
        summaries[policy] = {
            "episode_count": count,
            "success": wilson(success_count, count),
            "wrong_commitment": wilson(wrong_count, count),
            "mean_decision_cost": sum(row["decision_cost"] for row in policy_rows) / count,
        }
    proposed_rows = rows[PROPOSED]
    comparisons = {}
    for policy, policy_rows in rows.items():
        if policy == PROPOSED:
            continue
        proposed_keys = [(row["scenario"], row["seed"]) for row in proposed_rows]
        policy_keys = [(row["scenario"], row["seed"]) for row in policy_rows]
        if proposed_keys != policy_keys:
            raise ValueError(f"Paired episode mismatch for {policy}")
        comparisons[policy] = {
            "success_exact_mcnemar": exact_mcnemar(
                [row["success"] for row in proposed_rows],
                [row["success"] for row in policy_rows],
            ),
            "wrong_commitment_exact_mcnemar": exact_mcnemar(
                [row["wrong_commitment"] for row in proposed_rows],
                [row["wrong_commitment"] for row in policy_rows],
            ),
            "decision_cost_paired_bootstrap": paired_bootstrap_difference(
                [row["decision_cost"] for row in proposed_rows],
                [row["decision_cost"] for row in policy_rows],
            ),
        }
    result = {
        "schema_version": "icra-combined-policy-results-v1",
        "status": "completed",
        "scenario_count": 2,
        "episode_count_per_policy": len(proposed_rows),
        "policy_evaluation_count": sum(len(value) for value in rows.values()),
        "summaries": summaries,
        "paired_comparisons_proposed_vs_each_policy": comparisons,
        "canonical_rows": rows,
        "claim_gate": {
            "decision_level_scenario_A_and_B_complete": True,
            "paper_scale_final_evaluation_complete": False,
            "blocking_reasons": [
                "high_fidelity_counterfactual_physics_subset_not_complete",
                "scenario_B_detailed_baseline_rules_written_after_proposed_runs",
                "twenty_paired_episodes_are_insufficient_for_strong_statistical_guarantees",
                "real_robot_validation_not_performed",
            ],
        },
        "metric_compatibility_note": "Scenario A joint target-relation success and Scenario B accessible target-location success are combined as task-success proxies; neither is a full counterfactual contact-physics result.",
        "training_performed": False,
        "testing_performed": True,
        "valid_for_final_evaluation": True,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_root / "summary.json").write_text(
        json.dumps(
            {key: value for key, value in result.items() if key != "canonical_rows"},
            indent=2,
            sort_keys=True,
        ) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-result", type=Path, default=OPEN_RESULT)
    parser.add_argument("--covered-result", type=Path, default=COVERED_RESULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    result = aggregate(args.open_result.resolve(), args.covered_result.resolve(), args.output_root.resolve())
    print(json.dumps({"status": result["status"], "summaries": result["summaries"], "claim_gate": result["claim_gate"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
