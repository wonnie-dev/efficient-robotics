#!/usr/bin/env python3
"""Replay the frozen belief-space planner on the reserved test cache."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from calibrated_belief import bayesian_update  # noqa: E402
from calibrate_joint_observation_model import (  # noqa: E402
    INFORMATION_ACTIONS,
    build_rows,
    likelihood_for,
)
from evaluate_scene_conditioned_planner import (  # noqa: E402
    replace_symbols,
)
from calibrate_scene_conditioned_views import (  # noqa: E402
    extract_features,
    predict as predict_view_mode,
)
from unified_task_belief_planner import plan  # noqa: E402


DEFAULT_PERCEPTION_ROOT = (
    ROOT / "outputs/final_evaluation/reserved_test/perception"
)
DEFAULT_FREEZE_ROOT = ROOT / "outputs/calibration/frozen"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "outputs/final_evaluation/reserved_test/policy_evaluation"
)
EXPECTED_TEST_SEEDS = set(range(1000, 1060))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_frozen_artifacts(
    freeze_root: Path,
    *,
    expected_test_seeds: set[int] | None = None,
    expected_status: str = "frozen_ready_for_reserved_test",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = freeze_root / "freeze_manifest.json"
    manifest = load_json(manifest_path)
    if manifest.get("status") != expected_status:
        raise RuntimeError("The calibration package was not frozen before test")
    expected = EXPECTED_TEST_SEEDS if expected_test_seeds is None else expected_test_seeds
    if set(int(seed) for seed in manifest["reserved_test_seeds"]) != expected:
        raise RuntimeError("The frozen reserved-test seed list changed")

    artifacts = manifest["artifacts"]
    loaded = []
    for name in ("joint_model", "view_model", "resolution_likelihoods"):
        record = artifacts[name]
        path = Path(record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise RuntimeError(f"Frozen artifact hash mismatch: {name}")
        loaded.append(load_json(path))
    return manifest, loaded[0], loaded[1], loaded[2]


def condition_view_actions(
    model: dict[str, Any],
    resolution_models: dict[str, Any],
    probabilities: dict[str, float],
) -> dict[str, Any]:
    conditioned = copy.deepcopy(model)
    metadata = {}
    for action, mode in (
        ("viewpoint_close_high", "close_high"),
        ("viewpoint_right", "right"),
    ):
        fitted = resolution_models["actions"][action]
        outcomes = tuple(fitted["vocabulary"])
        resolution_probability = float(probabilities[mode])
        likelihood = {
            hypothesis: {
                outcome: (
                    resolution_probability
                    * float(fitted["resolved"][hypothesis][outcome])
                    + (1.0 - resolution_probability)
                    * float(fitted["unresolved"][outcome])
                )
                for outcome in outcomes
            }
            for hypothesis in conditioned["semantic_hypotheses"]
        }
        conditioned["observation_model"][action] = {
            "outcomes": list(outcomes),
            "likelihood": likelihood,
        }
        information = conditioned["information_actions"][action]
        information["outcomes"] = list(outcomes)
        information["observation_likelihood"] = {"open": likelihood}
        information["next_task_state_by_outcome"] = {
            "open": {outcome: "open" for outcome in outcomes}
        }
        metadata[action] = {
            "view_mode": mode,
            "calibrated_resolution_probability": resolution_probability,
        }
    conditioned["scene_conditioned_sensor_model"] = {
        "method": "frozen_calibrated_resolution_likelihood_mixture",
        "actions": metadata,
        "policy_override_used": False,
        "held_out_future_observation_used": False,
    }
    return conditioned


def replay_episode(
    episode: dict[str, Any],
    perception_root: Path,
    frozen_model: dict[str, Any],
    view_model: dict[str, Any],
    resolution_models: dict[str, Any],
) -> dict[str, Any]:
    center = episode["rows"]["initial_observation"]
    belief = bayesian_update(
        frozen_model["initial_semantic_belief"],
        likelihood_for(
            frozen_model, "initial_observation", center["observation_symbol"]
        ),
    )
    state = str(episode["initial_task_state"])
    remaining = tuple(
        action for action in INFORMATION_ACTIONS if action in episode["rows"]
    )
    sequence: list[str] = []
    policies: list[dict[str, Any]] = []
    updates = [
        {
            "action": "initial_observation",
            "observation": center["observation_symbol"],
            "posterior": belief,
        }
    ]
    active_model = frozen_model
    view_prediction = None
    planning_seconds = 0.0

    for _ in range(len(remaining) + 1):
        planning_started = time.perf_counter()
        policy = plan(
            belief,
            state,
            active_model,
            horizon=min(int(active_model["horizon"]), len(remaining)),
            remaining_actions=remaining,
        )
        planning_seconds += time.perf_counter() - planning_started
        policies.append(policy)
        action = str(policy["selected_action"])
        sequence.append(action)
        if action.startswith("grasp:") or action == "defer":
            break
        if action not in episode["rows"]:
            sequence.append("replay_missing_observation")
            break

        row = episode["rows"][action]
        belief = bayesian_update(
            belief,
            likelihood_for(active_model, action, row["observation_symbol"]),
        )
        if action == "remove_cover":
            state = "open"
            features = extract_features(perception_root, int(episode["seed"]))
            view_prediction = predict_view_mode(
                features,
                list(view_model["episodes"]),
                k=int(view_model["neighbor_count"]),
                beta=float(view_model["probability_pseudocount"]),
            )
            view_prediction["features"] = features
            view_prediction["test_episode_used_for_fit"] = False
            view_prediction["held_out_future_view_used"] = False
            active_model = condition_view_actions(
                frozen_model, resolution_models, view_prediction["probabilities"]
            )
        updates.append(
            {
                "action": action,
                "observation": row["observation_symbol"],
                "posterior": belief,
            }
        )
        remaining = tuple(item for item in remaining if item != action)

    terminal = sequence[-1]
    terminal_hypothesis = (
        terminal.removeprefix("grasp:").replace(":", "|")
        if terminal.startswith("grasp:")
        else None
    )
    truth = str(episode["true_joint_hypothesis"])
    retrieval = terminal_hypothesis == truth
    safe_absent = truth == "target_absent|not_applicable" and terminal == "defer"
    return {
        "seed": int(episode["seed"]),
        "family": str(episode["family"]),
        "action_sequence": sequence,
        "terminal_action": terminal,
        "true_joint_hypothesis": truth,
        "retrieval_success": retrieval,
        "target_absent_safe_deferral": safe_absent,
        "semantic_decision_correct": retrieval or safe_absent,
        "wrong_commitment": terminal.startswith("grasp:") and not retrieval,
        "noncompletion": not terminal.startswith("grasp:"),
        "information_action_count": sum(
            action in INFORMATION_ACTIONS for action in sequence
        ),
        "belief_updates": updates,
        "policies": policies,
        "scene_conditioned_view_prediction": view_prediction,
        "planning_runtime_seconds": planning_seconds,
        "fixed_confidence_threshold_used": False,
        "future_test_observation_used_for_action_selection": False,
        "simulator_ground_truth_used_for_action_selection": False,
    }


def wilson(successes: int, count: int) -> list[float]:
    z = 1.959963984540054
    rate = successes / count
    denominator = 1.0 + z * z / count
    center = (rate + z * z / (2.0 * count)) / denominator
    margin = z * math.sqrt(
        rate * (1.0 - rate) / count + z * z / (4.0 * count * count)
    ) / denominator
    return [center - margin, center + margin]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    correct = sum(bool(row["semantic_decision_correct"]) for row in rows)
    return {
        "episode_count": count,
        "semantic_decision_correct_count": correct,
        "semantic_decision_accuracy": correct / count,
        "semantic_decision_accuracy_wilson_ci95": wilson(correct, count),
        "target_present_retrieval_success_count": sum(
            bool(row["retrieval_success"]) for row in rows
        ),
        "target_absent_safe_deferral_count": sum(
            bool(row["target_absent_safe_deferral"]) for row in rows
        ),
        "wrong_commitment_count": sum(bool(row["wrong_commitment"]) for row in rows),
        "wrong_commitment_rate": sum(bool(row["wrong_commitment"]) for row in rows)
        / count,
        "noncompletion_count": sum(bool(row["noncompletion"]) for row in rows),
        "mean_information_action_count": sum(
            int(row["information_action_count"]) for row in rows
        )
        / count,
        "mean_planning_runtime_seconds": sum(
            float(row["planning_runtime_seconds"]) for row in rows
        )
        / count,
        "action_sequence_counts": dict(
            Counter(" -> ".join(row["action_sequence"]) for row in rows)
        ),
        "by_family": {
            family: {
                "episode_count": len(members),
                "semantic_decision_correct_count": sum(
                    bool(row["semantic_decision_correct"]) for row in members
                ),
                "wrong_commitment_count": sum(
                    bool(row["wrong_commitment"]) for row in members
                ),
                "action_sequence_counts": dict(
                    Counter(" -> ".join(row["action_sequence"]) for row in members)
                ),
            }
            for family in sorted({row["family"] for row in rows})
            for members in [[row for row in rows if row["family"] == family]]
        },
    }


def run(perception_root: Path, freeze_root: Path, output_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    manifest, frozen_model, view_model, resolution_models = verify_frozen_artifacts(
        freeze_root
    )
    episodes = replace_symbols(
        build_rows(
            perception_root,
            minimum_iou=0.25,
            maximum_track_distance_m=0.12,
        )
    )
    if set(episodes) != EXPECTED_TEST_SEEDS:
        raise RuntimeError("The perception cache does not contain all reserved seeds")
    if set(int(seed) for seed in manifest["calibration_seeds"]) & set(episodes):
        raise RuntimeError("Calibration and reserved-test seeds overlap")

    rows = [
        replay_episode(
            episode,
            perception_root,
            frozen_model,
            view_model,
            resolution_models,
        )
        for _, episode in sorted(episodes.items())
    ]
    summary = summarize(rows)
    for row in rows:
        write_json(
            output_root / "episodes" / f"seed{int(row['seed']):04d}.json", row
        )
    result = {
        "schema_version": "reserved-test-policy-evaluation-v1",
        "status": "completed",
        "method": "proposed_task_risk_aware_joint_belief_mpc",
        "summary": summary,
        "calibration_freeze_manifest": {
            "path": str((freeze_root / "freeze_manifest.json").resolve()),
            "sha256": sha256(freeze_root / "freeze_manifest.json"),
        },
        "perception_manifest": {
            "path": str((perception_root / "observation_manifest.json").resolve()),
            "sha256": sha256(perception_root / "observation_manifest.json"),
        },
        "test_outcomes_used_for_parameter_selection": False,
        "fixed_confidence_threshold_used": False,
        "action_conditioned_future_belief_used": True,
        "scene_conditioned_view_model_used": True,
        "training_performed": False,
        "calibration_performed": False,
        "testing_performed": True,
        "reserved_test_seeds_used": True,
        "valid_for_frozen_policy_replay": True,
        "physical_terminal_grasps_executed_in_this_replay": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(output_root / "result.json", result)
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "seed",
                "family",
                "semantic_decision_correct",
                "wrong_commitment",
                "noncompletion",
                "information_action_count",
                "action_sequence",
                "planning_runtime_seconds",
            ),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row[key] for key in writer.fieldnames[:-2]},
                    "action_sequence": " -> ".join(row["action_sequence"]),
                    "planning_runtime_seconds": row["planning_runtime_seconds"],
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception-root", type=Path, default=DEFAULT_PERCEPTION_ROOT)
    parser.add_argument("--freeze-root", type=Path, default=DEFAULT_FREEZE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = run(
        args.perception_root.resolve(),
        args.freeze_root.resolve(),
        args.output_root.resolve(),
    )
    print(json.dumps({"status": result["status"], "summary": result["summary"]}, indent=2))


if __name__ == "__main__":
    main()
