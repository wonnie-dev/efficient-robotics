#!/usr/bin/env python3
"""Compare frozen policies on a cached perception split."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from calibrated_belief import bayesian_update  # noqa: E402
from evaluate_reserved_test import (  # noqa: E402
    condition_view_actions,
    verify_frozen_artifacts,
    wilson,
    write_json,
)
from calibrate_joint_observation_model import (  # noqa: E402
    HYPOTHESES,
    INFORMATION_ACTIONS,
    build_rows,
    likelihood_for,
)
from evaluate_scene_conditioned_planner import (  # noqa: E402
    NO_TARGET_EVIDENCE,
    replace_symbols,
)
from calibrate_scene_conditioned_views import (  # noqa: E402
    extract_candidate_set_features,
    extract_features,
    predict as predict_view_mode,
)
from unified_task_belief_planner import (  # noqa: E402
    next_task_state,
    observation_distribution,
    plan,
    update_semantic_belief,
    validate_unified_method_contract,
)


DEFAULT_DEFINITIONS = ROOT / "configs/research/method_definitions.json"
DEFAULT_PERCEPTION_ROOT = (
    ROOT / "outputs/final_evaluation/reserved_test/perception"
)
DEFAULT_FREEZE_ROOT = ROOT / "outputs/calibration/frozen"
DEFAULT_OUTPUT_ROOT = (
    ROOT / "outputs/final_evaluation/reserved_test/policy_evaluation"
)
EXPECTED_TEST_SEEDS = set(range(1100, 1160))
VIEW_ORDER = ("viewpoint_right", "viewpoint_close_high")
NATIVE_LABEL_FLOOR = 1e-3


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def entropy(belief: dict[str, float]) -> float:
    return -sum(float(value) * math.log(max(1e-12, float(value))) for value in belief.values())


def sigmoid(value: float) -> float:
    """Convert a native match logit without applying fitted calibration."""
    value = float(value)
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def update_with_native_scores(
    belief: dict[str, float], row: dict[str, Any]
) -> dict[str, float]:
    """Apply raw Qwen match logits and hard relation labels as likelihoods."""
    evidence = dict(row["candidate_evidence"])
    center_candidate = row.get("center_track_candidate_id")
    likelihood = {hypothesis: NATIVE_LABEL_FLOOR for hypothesis in HYPOTHESES}
    nonmatch_probability = 1.0
    for candidate_id, candidate in evidence.items():
        match_probability = sigmoid(float(candidate["raw_match_logit"]))
        nonmatch_probability *= 1.0 - match_probability
        track = (
            "track_center_selected"
            if center_candidate is not None and candidate_id == str(center_candidate)
            else "track_other_target"
        )
        membership = str(candidate["membership"])
        if membership in {"inside", "outside"}:
            likelihood[f"{track}|{membership}"] += match_probability
        else:
            likelihood[f"{track}|inside"] += 0.5 * match_probability
            likelihood[f"{track}|outside"] += 0.5 * match_probability
    likelihood["target_absent|not_applicable"] += nonmatch_probability
    posterior = {
        hypothesis: float(belief[hypothesis]) * likelihood[hypothesis]
        for hypothesis in HYPOTHESES
    }
    total = sum(posterior.values())
    if total <= 0.0:
        raise ValueError("Native-score update produced zero belief mass")
    return {hypothesis: value / total for hypothesis, value in posterior.items()}


def factorize_joint_belief(belief: dict[str, float]) -> dict[str, float]:
    absent = float(belief["target_absent|not_applicable"])
    present = max(0.0, 1.0 - absent)
    if present <= 0.0:
        return dict(belief)
    identity = {
        name: sum(
            float(value)
            for key, value in belief.items()
            if key.startswith(name + "|")
        )
        / present
        for name in ("track_center_selected", "track_other_target")
    }
    membership = {
        name: sum(
            float(value)
            for key, value in belief.items()
            if key.endswith("|" + name)
        )
        / present
        for name in ("inside", "outside")
    }
    result = {
        f"{track}|{relation}": present * identity[track] * membership[relation]
        for track in identity
        for relation in membership
    }
    result["target_absent|not_applicable"] = absent
    total = sum(result.values())
    return {key: value / total for key, value in result.items()}


def terminal_decision(
    belief: dict[str, float], state: str, model: dict[str, Any]
) -> dict[str, Any]:
    return plan(belief, state, model, horizon=0, remaining_actions=())


def forced_grasp(
    belief: dict[str, float], state: str, model: dict[str, Any]
) -> str:
    feasible = [
        (name, action)
        for name, action in model["terminal_grasp_actions"].items()
        if state in action["allowed_task_states"]
    ]
    if not feasible:
        return "defer"
    return max(
        feasible,
        key=lambda pair: (
            float(belief[pair[1]["semantic_hypothesis"]]),
            pair[0],
        ),
    )[0]


def information_gain_action(
    belief: dict[str, float],
    state: str,
    model: dict[str, Any],
    remaining: tuple[str, ...],
) -> tuple[str | None, float]:
    candidates = []
    current_entropy = entropy(belief)
    for action_name in remaining:
        action = model["information_actions"][action_name]
        if state not in action["allowed_task_states"]:
            continue
        distribution = observation_distribution(belief, state, action)
        expected_entropy = 0.0
        for outcome, probability in distribution.items():
            if float(probability) <= 0.0:
                continue
            posterior = update_semantic_belief(belief, state, action, outcome)
            expected_entropy += float(probability) * entropy(posterior)
        candidates.append(
            (
                action_name,
                current_entropy - expected_entropy,
                float(action["stage_cost"]),
            )
        )
    if not candidates:
        return None, 0.0
    selected = min(candidates, key=lambda row: (-row[1], row[2], row[0]))
    return selected[0], selected[1]


def lowest_stage_cost_information_action(
    state: str,
    model: dict[str, Any],
    remaining: tuple[str, ...],
) -> str | None:
    feasible = [
        action_name
        for action_name in remaining
        if state in model["information_actions"][action_name]["allowed_task_states"]
    ]
    if not feasible:
        return None
    return min(
        feasible,
        key=lambda action_name: (
            float(model["information_actions"][action_name]["stage_cost"]),
            action_name,
        ),
    )


def update_belief(
    belief: dict[str, float],
    model: dict[str, Any],
    action: str,
    symbol: str,
    *,
    no_joint: bool = False,
    ignore_negative: bool = False,
    no_tracking: bool = False,
) -> dict[str, float]:
    if ignore_negative and symbol == NO_TARGET_EVIDENCE:
        posterior = dict(belief)
    else:
        used_symbol = (
            symbol.replace("other_target|", "center_target|")
            if no_tracking
            else symbol
        )
        posterior = bayesian_update(
            belief, likelihood_for(model, action, used_symbol)
        )
    return factorize_joint_belief(posterior) if no_joint else posterior


def condition_after_remove(
    episode: dict[str, Any],
    perception_root: Path,
    frozen_model: dict[str, Any],
    view_model: dict[str, Any],
    resolution_models: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    features = (
        extract_candidate_set_features(perception_root, int(episode["seed"]))
        if view_model.get("feature_extractor") == "candidate_set_x_sorted_v1"
        else extract_features(perception_root, int(episode["seed"]))
    )
    prediction = predict_view_mode(
        features,
        list(view_model["episodes"]),
        k=int(view_model["neighbor_count"]),
        beta=float(view_model["probability_pseudocount"]),
    )
    return (
        condition_view_actions(
            frozen_model, resolution_models, prediction["probabilities"]
        ),
        {**prediction, "features": features, "held_out_future_view_used": False},
    )


def score_episode(
    episode: dict[str, Any],
    sequence: list[str],
    model: dict[str, Any],
    planning_seconds: float,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    terminal = sequence[-1]
    terminal_hypothesis = (
        terminal.removeprefix("grasp:").replace(":", "|")
        if terminal.startswith("grasp:")
        else None
    )
    truth = str(episode["true_joint_hypothesis"])
    retrieval = terminal_hypothesis == truth
    safe_absent = truth == "target_absent|not_applicable" and terminal == "defer"
    wrong = terminal.startswith("grasp:") and not retrieval
    costs = model["costs"]
    realized_cost = 0.0
    for action in sequence:
        if action in model["information_actions"]:
            realized_cost += float(model["information_actions"][action]["stage_cost"])
        elif action.startswith("grasp:"):
            terminal_model = model["terminal_grasp_actions"][action]
            realized_cost += float(terminal_model["stage_cost"])
            if wrong:
                realized_cost += float(costs["wrong_commitment"])
            elif retrieval:
                execution = float(
                    terminal_model["conditional_execution_success_probability"]
                )
                realized_cost += float(costs["execution_failure"]) * (1.0 - execution)
        elif action == "defer":
            realized_cost += float(costs["defer"])
    return {
        "seed": int(episode["seed"]),
        "family": str(episode["family"]),
        "action_sequence": sequence,
        "terminal_action": terminal,
        "true_joint_hypothesis": truth,
        "semantic_task_success": retrieval or safe_absent,
        "retrieval_success": retrieval,
        "target_absent_safe_deferral": safe_absent,
        "wrong_commitment": wrong,
        "noncompletion": not terminal.startswith("grasp:"),
        "information_action_count": sum(
            action in INFORMATION_ACTIONS for action in sequence
        ),
        "interaction_action_count": sum(action == "remove_cover" for action in sequence),
        "realized_task_cost": realized_cost,
        "planning_runtime_seconds": planning_seconds,
        "future_test_observation_used_for_action_selection": False,
        "simulator_ground_truth_used_for_action_selection": False,
        **(metadata or {}),
    }


def replay_risk_policy(
    episode: dict[str, Any],
    perception_root: Path,
    model: dict[str, Any],
    view_model: dict[str, Any],
    resolution_models: dict[str, Any],
    *,
    selection: str,
    no_joint: bool = False,
    ignore_negative: bool = False,
    no_tracking: bool = False,
    scene_conditioned: bool = True,
    uncalibrated_current_observations: bool = False,
) -> dict[str, Any]:
    center = episode["rows"]["initial_observation"]
    if uncalibrated_current_observations:
        belief = update_with_native_scores(model["initial_semantic_belief"], center)
    else:
        belief = update_belief(
            model["initial_semantic_belief"],
            model,
            "initial_observation",
            center["observation_symbol"],
            no_joint=no_joint,
            ignore_negative=ignore_negative,
            no_tracking=no_tracking,
        )
    state = str(episode["initial_task_state"])
    remaining = tuple(action for action in INFORMATION_ACTIONS if action in episode["rows"])
    active = model
    sequence: list[str] = []
    view_prediction = None
    planning_seconds = 0.0
    used_views: list[str] = []
    required_fixed_view: str | None = None

    for _ in range(len(remaining) + 1):
        started = time.perf_counter()
        if selection == "proposed":
            decision = plan(
                belief,
                state,
                active,
                horizon=min(int(active["horizon"]), len(remaining)),
                remaining_actions=remaining,
            )
            action = str(decision["selected_action"])
        else:
            decision = terminal_decision(belief, state, active)
            action = str(decision["selected_action"])
            if required_fixed_view is not None and required_fixed_view in remaining:
                action = required_fixed_view
                required_fixed_view = None
            elif action == "defer" and remaining:
                if state == "covered" and "remove_cover" in remaining:
                    action = "remove_cover"
                    if selection == "fixed_close_high":
                        required_fixed_view = "viewpoint_close_high"
                    elif selection == "fixed_right":
                        required_fixed_view = "viewpoint_right"
                elif selection == "fixed_close_high":
                    order = ("viewpoint_close_high", "viewpoint_right")
                    action = next((item for item in order if item in remaining), "defer")
                elif selection == "myopic_information_gain":
                    action, _gain = information_gain_action(
                        belief, state, active, remaining
                    )
                    action = action or "defer"
                elif selection == "lowest_stage_cost_information":
                    action = (
                        lowest_stage_cost_information_action(
                            state, active, remaining
                        )
                        or "defer"
                    )
                else:
                    action = next((item for item in VIEW_ORDER if item in remaining), "defer")
        planning_seconds += time.perf_counter() - started
        sequence.append(action)
        if action.startswith("grasp:") or action == "defer":
            break
        row = episode["rows"][action]
        if uncalibrated_current_observations:
            belief = update_with_native_scores(belief, row)
        else:
            belief = update_belief(
                belief,
                active,
                action,
                row["observation_symbol"],
                no_joint=no_joint,
                ignore_negative=ignore_negative,
                no_tracking=no_tracking,
            )
        transition_symbol = str(row["observation_symbol"])
        if transition_symbol not in active["information_actions"][action]["outcomes"]:
            transition_symbol = "unseen"
        state = next_task_state(
            active["information_actions"][action], state, transition_symbol
        )
        if action == "remove_cover":
            if scene_conditioned and state == "open":
                active, view_prediction = condition_after_remove(
                    episode,
                    perception_root,
                    model,
                    view_model,
                    resolution_models,
                )
        elif action.startswith("viewpoint_"):
            used_views.append(action)
        remaining = tuple(item for item in remaining if item != action)
    return score_episode(
        episode,
        sequence,
        model,
        planning_seconds,
        {
            "scene_conditioned_view_prediction": view_prediction,
            "selected_view_actions": used_views,
        },
    )


def replay_open_loop_belief_planner(
    episode: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    """Plan from the center view and never use later semantic observations."""
    center = episode["rows"]["initial_observation"]
    belief = update_belief(
        model["initial_semantic_belief"],
        model,
        "initial_observation",
        center["observation_symbol"],
    )
    state = str(episode["initial_task_state"])
    remaining = tuple(
        action for action in INFORMATION_ACTIONS if action in episode["rows"]
    )
    sequence: list[str] = []
    planning_seconds = 0.0
    for _ in range(len(remaining) + 1):
        started = time.perf_counter()
        decision = plan(
            belief,
            state,
            model,
            horizon=min(int(model["horizon"]), len(remaining)),
            remaining_actions=remaining,
        )
        planning_seconds += time.perf_counter() - started
        action = str(decision["selected_action"])
        sequence.append(action)
        if action.startswith("grasp:") or action == "defer":
            break
        action_model = model["information_actions"][action]
        predicted = observation_distribution(belief, state, action_model)
        outcome = min(predicted, key=lambda name: (-float(predicted[name]), name))
        belief = update_semantic_belief(belief, state, action_model, outcome)
        state = next_task_state(action_model, state, outcome)
        remaining = tuple(item for item in remaining if item != action)
    return score_episode(
        episode,
        sequence,
        model,
        planning_seconds,
        {
            "replanning_used": False,
            "post_initial_semantic_observations_used": False,
        },
    )


def replay_random_feasible(
    episode: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    """Sample a reproducible feasible action without using semantic evidence."""
    generator = random.Random(7919 + int(episode["seed"]))
    state = str(episode["initial_task_state"])
    remaining = tuple(
        action for action in INFORMATION_ACTIONS if action in episode["rows"]
    )
    sequence: list[str] = []
    for _ in range(len(remaining) + 1):
        information = [
            name
            for name in remaining
            if state in model["information_actions"][name]["allowed_task_states"]
        ]
        terminal = [
            name
            for name, action in model["terminal_grasp_actions"].items()
            if state in action["allowed_task_states"]
        ] + ["defer"]
        action = generator.choice(sorted(information + terminal))
        sequence.append(action)
        if action.startswith("grasp:") or action == "defer":
            break
        row = episode["rows"][action]
        action_model = model["information_actions"][action]
        outcome = str(row["observation_symbol"])
        if outcome not in action_model["outcomes"]:
            outcome = "unseen"
        state = next_task_state(action_model, state, outcome)
        remaining = tuple(item for item in remaining if item != action)
    return score_episode(
        episode,
        sequence,
        model,
        0.0,
        {"random_policy_seed": 7919 + int(episode["seed"])},
    )


def replay_immediate(
    episode: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    center = episode["rows"]["initial_observation"]
    belief = update_belief(
        model["initial_semantic_belief"],
        model,
        "initial_observation",
        center["observation_symbol"],
    )
    action = forced_grasp(belief, str(episode["initial_task_state"]), model)
    return score_episode(episode, [action], model, 0.0)


def replay_native_greedy(
    episode: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    state = str(episode["initial_task_state"])
    actions_by_view = ["initial_observation"]
    if "remove_cover" in episode["rows"]:
        actions_by_view.append("remove_cover")
    actions_by_view.extend(action for action in VIEW_ORDER if action in episode["rows"])
    sequence: list[str] = []
    for observation_action in actions_by_view:
        row = episode["rows"][observation_action]
        if observation_action != "initial_observation":
            # The observation is available only after its camera or interaction
            # action has been executed and charged to the policy.
            sequence.append(observation_action)
            if observation_action in model["information_actions"]:
                transition_symbol = str(row["observation_symbol"])
                action_model = model["information_actions"][observation_action]
                if transition_symbol not in action_model["outcomes"]:
                    transition_symbol = "unseen"
                state = next_task_state(action_model, state, transition_symbol)
        symbol = str(row["observation_symbol"])
        if symbol != NO_TARGET_EVIDENCE and "|" in symbol:
            identity, membership = symbol.split("|", maxsplit=1)
            track = (
                "track_center_selected"
                if identity == "center_target"
                else "track_other_target"
            )
            grasp = f"grasp:{track}:{membership}"
            if grasp in model["terminal_grasp_actions"] and state in model[
                "terminal_grasp_actions"
            ][grasp]["allowed_task_states"]:
                sequence.append(grasp)
                break
    if not sequence or not (sequence[-1].startswith("grasp:") or sequence[-1] == "defer"):
        sequence.append("defer")
    return score_episode(episode, sequence, model, 0.0)


def replay_no_task_risk(
    episode: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    center = episode["rows"]["initial_observation"]
    belief = update_belief(
        model["initial_semantic_belief"], model, "initial_observation", center["observation_symbol"]
    )
    state = str(episode["initial_task_state"])
    remaining = tuple(action for action in INFORMATION_ACTIONS if action in episode["rows"])
    sequence: list[str] = []
    while remaining:
        action, gain = information_gain_action(belief, state, model, remaining)
        if action is None or gain <= 0.0:
            break
        sequence.append(action)
        row = episode["rows"][action]
        belief = update_belief(belief, model, action, row["observation_symbol"])
        transition_symbol = str(row["observation_symbol"])
        action_model = model["information_actions"][action]
        if transition_symbol not in action_model["outcomes"]:
            transition_symbol = "unseen"
        state = next_task_state(action_model, state, transition_symbol)
        remaining = tuple(item for item in remaining if item != action)
    sequence.append(forced_grasp(belief, state, model))
    return score_episode(episode, sequence, model, 0.0)


def replay_oracle(episode: dict[str, Any], model: dict[str, Any]) -> dict[str, Any]:
    truth = str(episode["true_joint_hypothesis"])
    if truth == "target_absent|not_applicable":
        sequence = ["defer"]
    else:
        grasp = "grasp:" + truth.replace("|", ":")
        sequence = []
        if truth.endswith("|inside") and episode["initial_task_state"] == "covered":
            sequence.append("remove_cover")
        sequence.append(grasp)
    row = score_episode(episode, sequence, model, 0.0)
    row["simulator_ground_truth_used_for_action_selection"] = True
    row["oracle_upper_bound_only"] = True
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    successes = sum(bool(row["semantic_task_success"]) for row in rows)
    return {
        "episode_count": count,
        "semantic_task_success_count": successes,
        "semantic_task_success_rate": successes / count,
        "semantic_task_success_wilson_ci95": wilson(successes, count),
        "wrong_commitment_count": sum(bool(row["wrong_commitment"]) for row in rows),
        "wrong_commitment_rate": sum(bool(row["wrong_commitment"]) for row in rows) / count,
        "target_absent_safe_deferral_count": sum(
            bool(row["target_absent_safe_deferral"]) for row in rows
        ),
        "mean_realized_task_cost": sum(float(row["realized_task_cost"]) for row in rows) / count,
        "mean_information_action_count": sum(int(row["information_action_count"]) for row in rows) / count,
        "mean_interaction_action_count": sum(int(row["interaction_action_count"]) for row in rows) / count,
        "mean_planning_runtime_seconds": sum(float(row["planning_runtime_seconds"]) for row in rows) / count,
        "action_sequence_counts": dict(Counter(" -> ".join(row["action_sequence"]) for row in rows)),
        "by_family": {
            family: {
                "episode_count": len(members),
                "semantic_task_success_count": sum(bool(row["semantic_task_success"]) for row in members),
                "wrong_commitment_count": sum(bool(row["wrong_commitment"]) for row in members),
            }
            for family in sorted({row["family"] for row in rows})
            for members in [[row for row in rows if row["family"] == family]]
        },
    }


def exact_mcnemar(proposed: list[dict[str, Any]], comparison: list[dict[str, Any]]) -> dict[str, Any]:
    other = {int(row["seed"]): row for row in comparison}
    proposed_only = sum(
        bool(row["semantic_task_success"])
        and not bool(other[int(row["seed"])]["semantic_task_success"])
        for row in proposed
    )
    comparison_only = sum(
        not bool(row["semantic_task_success"])
        and bool(other[int(row["seed"])]["semantic_task_success"])
        for row in proposed
    )
    discordant = proposed_only + comparison_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = sum(
            math.comb(discordant, index)
            for index in range(min(proposed_only, comparison_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * tail)
    return {
        "proposed_only_successes": proposed_only,
        "comparison_only_successes": comparison_only,
        "discordant_pairs": discordant,
        "two_sided_exact_p_value": p_value,
    }


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_cost_bootstrap(
    proposed: list[dict[str, Any]], comparison: list[dict[str, Any]], samples: int = 10000
) -> dict[str, Any]:
    other = {int(row["seed"]): row for row in comparison}
    pairs = [(row, other[int(row["seed"])]) for row in proposed]
    generator = random.Random(0)
    values = []
    for _ in range(samples):
        selected = [pairs[generator.randrange(len(pairs))] for _ in pairs]
        values.append(
            sum(float(right["realized_task_cost"]) - float(left["realized_task_cost"]) for left, right in selected)
            / len(selected)
        )
    return {"mean_cost_reduction": sum(values) / len(values), "ci95": [quantile(values, 0.025), quantile(values, 0.975)]}


def run(
    definitions_path: Path,
    perception_root: Path,
    freeze_root: Path,
    output_root: Path,
    calibration_candidate_root: Path | None = None,
    view_model_candidate: Path | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    definitions = load_json(definitions_path)
    if calibration_candidate_root is not None:
        if definitions["status"] == "frozen_before_untouched_test":
            raise RuntimeError(
                "An unfrozen calibration candidate cannot evaluate a reserved test"
            )
        if output_root == DEFAULT_OUTPUT_ROOT or "final_evaluation" in output_root.parts:
            raise RuntimeError(
                "An unfrozen calibration candidate can write only development diagnostics"
            )
        model = load_json(
            calibration_candidate_root / "calibration_candidate_model.json"
        )
        view_model = load_json(
            view_model_candidate
            if view_model_candidate is not None
            else calibration_candidate_root
            / "scene_conditioned_view_model_candidate.json"
        )
        resolution_models = load_json(
            calibration_candidate_root / "resolution_likelihoods_candidate.json"
        )
    else:
        expected_frozen_seeds = EXPECTED_TEST_SEEDS
        if definitions["status"] != "frozen_before_untouched_test":
            development_manifest = load_json(freeze_root / "freeze_manifest.json")
            expected_frozen_seeds = {
                int(seed) for seed in development_manifest["reserved_test_seeds"]
            }
        _manifest, model, view_model, resolution_models = verify_frozen_artifacts(
            freeze_root,
            expected_test_seeds=expected_frozen_seeds,
        )
    validate_unified_method_contract(model)
    episodes = replace_symbols(build_rows(perception_root, minimum_iou=0.25, maximum_track_distance_m=0.12))
    if definitions["status"] == "frozen_before_untouched_test":
        if set(episodes) != EXPECTED_TEST_SEEDS:
            raise RuntimeError("The reserved-test episode set is incomplete or changed")
    elif output_root == DEFAULT_OUTPUT_ROOT:
        raise RuntimeError(
            "Unfrozen development diagnostics cannot write into the final-test output root"
        )

    runners: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
        "proposed_task_risk_aware_joint_belief_mpc": lambda ep: replay_risk_policy(ep, perception_root, model, view_model, resolution_models, selection="proposed"),
        "immediate_grasp": lambda ep: replay_immediate(ep, model),
        "fixed_right_view": lambda ep: replay_risk_policy(ep, perception_root, model, view_model, resolution_models, selection="fixed_right"),
        "fixed_close_high_view": lambda ep: replay_risk_policy(ep, perception_root, model, view_model, resolution_models, selection="fixed_close_high"),
        "confidence_greedy": lambda ep: replay_native_greedy(ep, model),
        "myopic_information_gain": lambda ep: replay_risk_policy(ep, perception_root, model, view_model, resolution_models, selection="myopic_information_gain"),
        "open_loop_belief_planner": lambda ep: replay_open_loop_belief_planner(ep, model),
        "random_feasible_action": lambda ep: replay_random_feasible(ep, model),
        "oracle_simulator_ground_truth": lambda ep: replay_oracle(ep, model),
        "no_joint_belief": lambda ep: replay_risk_policy(ep, perception_root, model, view_model, resolution_models, selection="proposed", no_joint=True),
        "no_calibration": lambda ep: replay_risk_policy(ep, perception_root, model, view_model, resolution_models, selection="proposed", uncalibrated_current_observations=True),
        "no_negative_evidence": lambda ep: replay_risk_policy(ep, perception_root, model, view_model, resolution_models, selection="proposed", ignore_negative=True),
        "no_action_conditioned_future_belief": lambda ep: replay_risk_policy(ep, perception_root, model, view_model, resolution_models, selection="lowest_stage_cost_information"),
        "no_task_risk_cost": lambda ep: replay_no_task_risk(ep, model),
        "no_persistent_tracking": lambda ep: replay_risk_policy(ep, perception_root, model, view_model, resolution_models, selection="proposed", no_tracking=True),
        "no_scene_conditioned_view_model": lambda ep: replay_risk_policy(ep, perception_root, model, view_model, resolution_models, selection="proposed", scene_conditioned=False),
    }
    declared = [*definitions["methods"], *definitions["ablations"]]
    if set(declared) != set(runners):
        raise RuntimeError("Declared method set differs from the evaluator")
    rows_by_method = {
        method: [runner(episode) for _, episode in sorted(episodes.items())]
        for method, runner in runners.items()
    }
    expected_method_seeds = set(episodes)
    for method, rows in rows_by_method.items():
        method_seeds = {int(row["seed"]) for row in rows}
        if method_seeds != expected_method_seeds:
            raise RuntimeError(
                f"Method {method} did not evaluate the identical paired seed set"
            )
    for method, rows in rows_by_method.items():
        for row in rows:
            write_json(output_root / "episodes" / method / f"seed{int(row['seed']):04d}.json", {"method": method, **row})
    summaries = {method: summarize(rows) for method, rows in rows_by_method.items()}
    proposed = rows_by_method["proposed_task_risk_aware_joint_belief_mpc"]
    comparisons = {
        method: {
            "exact_mcnemar": exact_mcnemar(proposed, rows),
            "paired_cost_bootstrap": paired_cost_bootstrap(proposed, rows),
        }
        for method, rows in rows_by_method.items()
        if method not in {"proposed_task_risk_aware_joint_belief_mpc", "oracle_simulator_ground_truth"}
    }
    result = {
        "schema_version": "policy-comparison-v1",
        "status": "completed",
        "evaluation_role": "development_diagnostic" if definitions["status"] != "frozen_before_untouched_test" else "reserved_test",
        "method_definitions": str(definitions_path.resolve()),
        "summaries": summaries,
        "paired_comparisons_against_proposed": comparisons,
        "training_performed": False,
        "calibration_performed": False,
        "testing_performed": definitions["status"] == "frozen_before_untouched_test",
        "valid_for_final_evaluation": definitions["status"] == "frozen_before_untouched_test",
        "paired_same_seed_set_verified": True,
        "calibration_candidate_root": (
            str(calibration_candidate_root.resolve())
            if calibration_candidate_root is not None
            else None
        ),
        "unfrozen_calibration_candidate_used": calibration_candidate_root is not None,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(output_root / "result.json", result)
    with (output_root / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["method", "episode_count", "semantic_task_success_count", "semantic_task_success_rate", "wrong_commitment_count", "wrong_commitment_rate", "target_absent_safe_deferral_count", "mean_realized_task_cost", "mean_information_action_count", "mean_interaction_action_count", "mean_planning_runtime_seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for method, summary in summaries.items():
            writer.writerow({"method": method, **{key: summary[key] for key in fields[1:]}})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--definitions", type=Path, default=DEFAULT_DEFINITIONS)
    parser.add_argument("--perception-root", type=Path, default=DEFAULT_PERCEPTION_ROOT)
    parser.add_argument("--freeze-root", type=Path, default=DEFAULT_FREEZE_ROOT)
    parser.add_argument("--calibration-candidate-root", type=Path)
    parser.add_argument("--view-model-candidate", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    result = run(
        args.definitions.resolve(),
        args.perception_root.resolve(),
        args.freeze_root.resolve(),
        args.output_root.resolve(),
        (
            args.calibration_candidate_root.resolve()
            if args.calibration_candidate_root is not None
            else None
        ),
        (
            args.view_model_candidate.resolve()
            if args.view_model_candidate is not None
            else None
        ),
    )
    print(json.dumps({"status": result["status"], "evaluation_role": result["evaluation_role"], "summaries": result["summaries"]}, indent=2))


if __name__ == "__main__":
    main()
