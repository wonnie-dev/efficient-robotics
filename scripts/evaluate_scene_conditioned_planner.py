#!/usr/bin/env python3
"""Cross-validate scene-conditioned sensing inside the unified planner.

The replay uses only observations available at the current decision time.  A
post-remove RGB-D/Qwen feature vector predicts which wrist view is likely to
resolve the scene.  That calibrated prediction conditions each view action's
future observation likelihood; it never overrides the planner's action.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

from calibrated_belief import bayesian_update
from calibrate_joint_observation_model import (
    HYPOTHESES,
    INFORMATION_ACTIONS,
    PREFLIGHT_MODEL,
    build_rows,
    fit_model,
    likelihood_for,
    load_json,
    match_label,
    normalize_counts,
    select_alpha_episode_disjoint,
    write_json,
)
from calibrate_scene_conditioned_views import (
    predict as predict_view_mode,
    select_k,
    load_rows as load_view_rows,
    view_mode,
)
from calibrate_candidate_view_model import (
    load_rows as load_candidate_set_view_rows,
)
from unified_task_belief_planner import plan


DEFAULT_ROOTS = (
    ROOT / "outputs/calibration/calibration_episodes_perception",
    ROOT / "outputs/calibration/supplemental_perception",
)
DEFAULT_OUTPUT = (
    ROOT
    / "outputs/calibration/scene_conditioned_planner"
    / "result.json"
)
VIEW_ACTIONS = {
    "viewpoint_close_high": "close_high",
    "viewpoint_right": "right",
}
NO_TARGET_EVIDENCE = "no_target_evidence"


def semantic_observation(row: dict[str, Any]) -> str:
    """Encode persistent candidate identity only when Qwen reports a match."""
    evidence = dict(row["candidate_evidence"])
    center_track = row.get("center_track_candidate_id")
    matches = [
        (candidate_id, item)
        for candidate_id, item in evidence.items()
        if match_label(float(item["raw_match_logit"])) == "match"
    ]
    if not matches:
        return NO_TARGET_EVIDENCE
    candidate_id, item = max(
        matches, key=lambda pair: float(pair[1]["raw_match_logit"])
    )
    identity = "center_target" if candidate_id == center_track else "other_target"
    return f"{identity}|{item['membership']}"


def replace_symbols(episodes: dict[int, dict[str, Any]]) -> dict[int, dict[str, Any]]:
    copied = copy.deepcopy(episodes)
    for episode in copied.values():
        for row in episode["rows"].values():
            row["observation_symbol"] = semantic_observation(row)
    return copied


def resolution_label(episode: dict[str, Any], action: str) -> bool:
    """Return whether the view is designed to resolve this scene family."""
    try:
        required_mode = view_mode(str(episode["family"]))
    except ValueError:
        return False
    return required_mode == VIEW_ACTIONS[action]


def fitted_resolution_likelihoods(
    episodes: dict[int, dict[str, Any]], action: str, alpha: float
) -> dict[str, Any]:
    rows = [episode["rows"][action] for episode in episodes.values() if action in episode["rows"]]
    vocabulary = tuple(sorted({str(row["observation_symbol"]) for row in rows} | {NO_TARGET_EVIDENCE, "unseen"}))
    resolved_counts: dict[str, Counter[str]] = defaultdict(Counter)
    unresolved_counts: Counter[str] = Counter()
    resolved_support = Counter()
    unresolved_support = 0
    for episode in episodes.values():
        if action not in episode["rows"]:
            continue
        row = episode["rows"][action]
        symbol = str(row["observation_symbol"])
        if resolution_label(episode, action):
            truth = str(episode["true_joint_hypothesis"])
            resolved_counts[truth][symbol] += 1
            resolved_support[truth] += 1
        else:
            unresolved_counts[symbol] += 1
            unresolved_support += 1
    resolved = {
        hypothesis: normalize_counts(resolved_counts[hypothesis], vocabulary, alpha)
        for hypothesis in HYPOTHESES
    }
    unresolved = normalize_counts(unresolved_counts, vocabulary, alpha)
    return {
        "vocabulary": vocabulary,
        "resolved": resolved,
        "unresolved": unresolved,
        "resolved_support_by_hypothesis": dict(resolved_support),
        "unresolved_support": unresolved_support,
    }


def condition_view_actions(
    model: dict[str, Any],
    episodes: dict[int, dict[str, Any]],
    view_probabilities: dict[str, float],
    alpha: float,
) -> dict[str, Any]:
    """Condition view likelihoods without selecting an action directly."""
    conditioned = copy.deepcopy(model)
    metadata = {}
    for action, mode in VIEW_ACTIONS.items():
        if action not in conditioned["information_actions"]:
            continue
        fitted = fitted_resolution_likelihoods(episodes, action, alpha)
        probability = float(view_probabilities[mode])
        vocabulary = fitted["vocabulary"]
        likelihood = {}
        for hypothesis in HYPOTHESES:
            likelihood[hypothesis] = {
                outcome: (
                    probability * fitted["resolved"][hypothesis][outcome]
                    + (1.0 - probability) * fitted["unresolved"][outcome]
                )
                for outcome in vocabulary
            }
        conditioned["observation_model"][action] = {
            "outcomes": list(vocabulary),
            "likelihood": likelihood,
            "support_by_hypothesis": fitted["resolved_support_by_hypothesis"],
        }
        action_model = conditioned["information_actions"][action]
        action_model["outcomes"] = list(vocabulary)
        action_model["observation_likelihood"] = {"open": likelihood}
        action_model["next_task_state_by_outcome"] = {
            "open": {outcome: "open" for outcome in vocabulary}
        }
        metadata[action] = {
            "view_mode": mode,
            "calibrated_resolution_probability": probability,
            "resolved_support_by_hypothesis": fitted["resolved_support_by_hypothesis"],
            "unresolved_support": fitted["unresolved_support"],
        }
    conditioned["scene_conditioned_sensor_model"] = {
        "method": "calibrated_resolution_mixture_of_resolved_and_unresolved_likelihoods",
        "actions": metadata,
        "policy_override_used": False,
        "held_out_future_observation_used": False,
    }
    return conditioned


def replay(
    episode: dict[str, Any],
    model: dict[str, Any],
    training_episodes: dict[int, dict[str, Any]],
    view_training: list[dict[str, Any]],
    held_out_view: dict[str, Any] | None,
    *,
    alpha: float,
    k_grid: tuple[int, ...],
    beta: float,
) -> dict[str, Any]:
    center = episode["rows"]["initial_observation"]
    belief = bayesian_update(
        model["initial_semantic_belief"],
        likelihood_for(model, "initial_observation", center["observation_symbol"]),
    )
    state = str(episode["initial_task_state"])
    remaining = tuple(action for action in INFORMATION_ACTIONS if action in episode["rows"])
    sequence: list[str] = []
    updates = [{"action": "initial_observation", "observation": center["observation_symbol"], "posterior": belief}]
    policies = []
    view_prediction = None
    active_model = model
    for _ in range(len(remaining) + 1):
        policy = plan(
            belief,
            state,
            active_model,
            horizon=min(int(active_model["horizon"]), len(remaining)),
            remaining_actions=remaining,
        )
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
            belief, likelihood_for(active_model, action, row["observation_symbol"])
        )
        if action == "remove_cover":
            state = "open"
            if held_out_view is not None:
                inner = select_k(view_training, k_grid, beta)
                prediction = predict_view_mode(
                    held_out_view["features"],
                    view_training,
                    k=int(inner["selected_k"]),
                    beta=beta,
                )
                view_prediction = {
                    **prediction,
                    "inner_k_selection": inner,
                    "held_out_episode_used_for_fit": False,
                    "held_out_future_view_used": False,
                }
                active_model = condition_view_actions(
                    model,
                    training_episodes,
                    prediction["probabilities"],
                    alpha,
                )
        updates.append({"action": action, "observation": row["observation_symbol"], "posterior": belief})
        remaining = tuple(item for item in remaining if item != action)

    terminal = sequence[-1]
    terminal_hypothesis = terminal.removeprefix("grasp:").replace(":", "|") if terminal.startswith("grasp:") else None
    truth = str(episode["true_joint_hypothesis"])
    retrieval = terminal_hypothesis == truth
    safe_absent = truth == "target_absent|not_applicable" and terminal == "defer"
    return {
        "action_sequence": sequence,
        "terminal_action": terminal,
        "retrieval_success": retrieval,
        "target_absent_safe_deferral": safe_absent,
        "semantic_decision_correct": retrieval or safe_absent,
        "wrong_commitment": terminal.startswith("grasp:") and not retrieval,
        "noncompletion": not terminal.startswith("grasp:"),
        "true_joint_hypothesis": truth,
        "belief_updates": updates,
        "policies": policies,
        "scene_conditioned_view_prediction": view_prediction,
        "policy_override_used": False,
        "fixed_confidence_threshold_used": False,
        "held_out_future_observation_used_for_action_selection": False,
    }


def run(
    roots: tuple[Path, ...],
    output: Path,
    *,
    defer_cost: float,
    wrong_commitment_cost: float,
    feature_extractor: str = "selected_candidate_v1",
) -> dict[str, Any]:
    started = time.perf_counter()
    episodes: dict[int, dict[str, Any]] = {}
    for root in roots:
        current = build_rows(root, minimum_iou=0.25, maximum_track_distance_m=0.12)
        overlap = set(episodes) & set(current)
        if overlap:
            raise ValueError(f"Duplicate seeds: {sorted(overlap)}")
        episodes.update(current)
    episodes = replace_symbols(episodes)
    if feature_extractor == "selected_candidate_v1":
        view_rows = load_view_rows(roots)
    elif feature_extractor == "candidate_set_x_sorted_v1":
        view_rows = load_candidate_set_view_rows(roots)
    else:
        raise ValueError(f"Unknown feature extractor: {feature_extractor}")
    view_by_seed = {int(row["seed"]): row for row in view_rows}
    preflight = load_json(PREFLIGHT_MODEL)
    preflight["costs"]["defer"] = float(defer_cost)
    preflight["costs"]["wrong_commitment"] = float(wrong_commitment_cost)
    alpha_grid = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
    k_grid = (1, 3, 5, 7, 9)
    beta = 0.1
    folds = []
    for seed, episode in sorted(episodes.items()):
        training = {key: value for key, value in episodes.items() if key != seed}
        alpha_selection = select_alpha_episode_disjoint(training, alpha_grid, preflight)
        alpha = float(alpha_selection["selected_alpha"])
        model = fit_model(training, alpha, preflight)
        view_training = [row for row in view_rows if int(row["seed"]) != seed]
        result = replay(
            episode,
            model,
            training,
            view_training,
            view_by_seed.get(seed),
            alpha=alpha,
            k_grid=k_grid,
            beta=beta,
        )
        folds.append({
            "seed": seed,
            "family": episode["family"],
            "outer_fold_alpha_selection": alpha_selection,
            **result,
        })
    target_present = [row for row in folds if row["true_joint_hypothesis"] != "target_absent|not_applicable"]
    summary = {
        "episode_count": len(folds),
        "target_present_episode_count": len(target_present),
        "retrieval_success_count": sum(row["retrieval_success"] for row in folds),
        "target_present_retrieval_success_rate": sum(row["retrieval_success"] for row in folds) / len(target_present),
        "target_absent_safe_deferral_count": sum(row["target_absent_safe_deferral"] for row in folds),
        "semantic_decision_correct_count": sum(row["semantic_decision_correct"] for row in folds),
        "semantic_decision_accuracy": sum(row["semantic_decision_correct"] for row in folds) / len(folds),
        "wrong_commitment_count": sum(row["wrong_commitment"] for row in folds),
        "wrong_commitment_rate": sum(row["wrong_commitment"] for row in folds) / len(folds),
        "noncompletion_count": sum(row["noncompletion"] for row in folds),
        "first_action_counts": dict(Counter(row["action_sequence"][0] for row in folds)),
        "action_sequences": dict(Counter(" -> ".join(row["action_sequence"]) for row in folds)),
        "by_family": {
            family: {
                "total": sum(row["family"] == family for row in folds),
                "correct": sum(row["family"] == family and row["semantic_decision_correct"] for row in folds),
                "wrong_commitment": sum(row["family"] == family and row["wrong_commitment"] for row in folds),
            }
            for family in sorted({row["family"] for row in folds})
        },
    }
    result = {
        "schema_version": "scene-conditioned-planner-evaluation-v1",
        "status": "passed_calibration_replay" if summary["wrong_commitment_count"] == 0 and summary["semantic_decision_accuracy"] >= 0.9 else "calibration_replay_failed",
        "protocol": "nested_episode_disjoint_scene_conditioned_sensor_model_and_unified_mpc_replay",
        "view_feature_extractor": feature_extractor,
        "summary": summary,
        "episodes": folds,
        "planner_costs": dict(preflight["costs"]),
        "fixed_confidence_threshold_used": False,
        "policy_override_used": False,
        "action_conditioned_future_belief_used": True,
        "simulator_ground_truth_used_for_inference": False,
        "simulator_ground_truth_used_for_calibration_labels": True,
        "training_performed": False,
        "calibration_performed": True,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
        "gpu_used": False,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--defer-cost", type=float, default=0.8)
    parser.add_argument("--wrong-commitment-cost", type=float, default=1.0)
    parser.add_argument(
        "--view-feature-extractor",
        choices=("selected_candidate_v1", "candidate_set_x_sorted_v1"),
        default="selected_candidate_v1",
    )
    args = parser.parse_args()
    roots = tuple(path.resolve() for path in args.perception_root) or DEFAULT_ROOTS
    result = run(
        roots,
        args.output.resolve(),
        defer_cost=float(args.defer_cost),
        wrong_commitment_cost=float(args.wrong_commitment_cost),
        feature_extractor=args.view_feature_extractor,
    )
    print(json.dumps({"status": result["status"], "summary": result["summary"], "runtime_seconds": result["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
