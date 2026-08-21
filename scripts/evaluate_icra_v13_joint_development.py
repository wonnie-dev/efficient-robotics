#!/usr/bin/env python3
"""Run an 18-episode development replay with a direct joint belief.

The hidden hypothesis is the persistent center-view candidate track crossed
with target membership.  Observation likelihoods are estimated directly for
that joint state, rather than by multiplying identity and relation marginals.
Every replayed episode is held out from its own model fit.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from core_method_runtime import (
    expected_cost_commitment_decision,
    joint_scene_graph_snapshot,
    update_joint_hypothesis_belief,
)
from run_offline_action_conditioned_mpc_replay import build_episode_rows


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/research/icra_v13_joint_development_stress_18seed.json"
HYPOTHESES = (
    "track_center_selected|inside",
    "track_center_selected|outside",
    "track_other_target|inside",
    "track_other_target|outside",
)
TRACK_IDS = ["track_center_selected", "track_other_target"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def state_for_episode(episode: dict[str, dict[str, Any]]) -> str:
    center = episode["initial_observation"]
    track = (
        "track_center_selected"
        if center["selected_target_correct_posthoc"]
        else "track_other_target"
    )
    membership = str(center["world_membership_state_posthoc"])
    state = f"{track}|{membership}"
    if state not in HYPOTHESES:
        raise ValueError(f"Unsupported joint state: {state}")
    return state


def membership_symbol(row: dict[str, Any]) -> str:
    value = str(row["membership_observation"])
    return value if value in {"inside", "outside"} else "unknown"


def observation_symbol(row: dict[str, Any]) -> str:
    """Compress only planner-visible tracking and RGB-D evidence."""
    membership = membership_symbol(row)
    if row["action"] == "initial_observation":
        candidate = (
            "selected_high"
            if str(row["identity_bin"]) == "high"
            else "selected_not_high"
        )
    else:
        agreement = str(row["track_agreement_observation"])
        confidence = str(row["center_track_confidence_bin"])
        if agreement == "same":
            candidate = "same_high" if confidence == "high" else "same_not_high"
        elif agreement == "different":
            candidate = "different"
        else:
            candidate = "missing"
    return f"{candidate}|{membership}"


def normalized_counts(counts: Counter[str], keys: tuple[str, ...], alpha: float) -> dict[str, float]:
    denominator = sum(counts.values()) + alpha * len(keys)
    return {key: (counts[key] + alpha) / denominator for key in keys}


def fit_joint_model(
    episodes: dict[int, dict[str, dict[str, Any]]],
    actions: list[str],
    alpha: float,
) -> dict[str, Any]:
    prior_counts = Counter(state_for_episode(episode) for episode in episodes.values())
    vocabularies: dict[str, tuple[str, ...]] = {}
    likelihood: dict[str, dict[str, dict[str, float]]] = {}
    all_actions = ["initial_observation", *actions]
    for action in all_actions:
        symbols = {observation_symbol(episode[action]) for episode in episodes.values()}
        symbols.add("other|unknown")
        vocabulary = tuple(sorted(symbols))
        vocabularies[action] = vocabulary
        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        for episode in episodes.values():
            grouped[state_for_episode(episode)][observation_symbol(episode[action])] += 1
        likelihood[action] = {
            state: normalized_counts(grouped[state], vocabulary, alpha)
            for state in HYPOTHESES
        }
    return {
        "hypotheses": list(HYPOTHESES),
        "prior": normalized_counts(prior_counts, HYPOTHESES, alpha),
        "observation_vocabulary": {key: list(value) for key, value in vocabularies.items()},
        "joint_observation_likelihood": likelihood,
        "fit_episode_count": len(episodes),
        "fit_seeds": sorted(episodes),
        "fit_uses_automatic_simulator_labels": True,
        "marginal_confidence_product_used": False,
    }


def observed_likelihood(model: dict[str, Any], action: str, symbol: str) -> dict[str, float]:
    vocabulary = set(model["observation_vocabulary"][action])
    used = symbol if symbol in vocabulary else "other|unknown"
    return {
        state: float(model["joint_observation_likelihood"][action][state][used])
        for state in HYPOTHESES
    }


def update_from_row(
    belief: dict[str, float], row: dict[str, Any], model: dict[str, Any]
) -> tuple[dict[str, float], dict[str, Any]]:
    action = str(row["action"])
    symbol = observation_symbol(row)
    likelihood = observed_likelihood(model, action, symbol)
    posterior = update_joint_hypothesis_belief(belief, likelihood)
    return posterior, {
        "action": action,
        "observation": symbol,
        "prior": belief,
        "joint_likelihood": likelihood,
        "posterior": posterior,
        "marginal_confidence_product_used": False,
    }


def terminal_values(belief: dict[str, float], config: dict[str, Any]) -> list[dict[str, Any]]:
    costs = config["costs"]
    execution_success = float(config["conditional_execution_success_probability"])
    values = []
    for track_id in TRACK_IDS:
        for membership in ("inside", "outside"):
            action = f"grasp:{track_id}:{membership}"
            probability = belief[f"{track_id}|{membership}"]
            wrong = 1.0 - probability
            execution_failure = probability * (1.0 - execution_success)
            values.append(
                {
                    "action": action,
                    "kind": "terminal_grasp",
                    "candidate_track_id": track_id,
                    "membership": membership,
                    "expected_cost": float(costs["grasp"])
                    + wrong * float(costs["wrong_commitment"])
                    + execution_failure * float(costs["execution_failure"]),
                    "semantic_success_probability": probability,
                }
            )
    values.append(
        {
            "action": "defer",
            "kind": "terminal_defer",
            "expected_cost": float(costs["defer"]),
        }
    )
    return values


def predictive_observations(
    belief: dict[str, float], model: dict[str, Any], action: str
) -> dict[str, float]:
    return {
        symbol: sum(
            belief[state]
            * float(model["joint_observation_likelihood"][action][state][symbol])
            for state in HYPOTHESES
        )
        for symbol in model["observation_vocabulary"][action]
    }


def plan(
    belief: dict[str, float],
    model: dict[str, Any],
    config: dict[str, Any],
    remaining_actions: tuple[str, ...],
    depth: int,
) -> dict[str, Any]:
    values = terminal_values(belief, config)
    if depth > 0:
        for action in remaining_actions:
            branches = []
            expected_future = 0.0
            for symbol, probability in predictive_observations(belief, model, action).items():
                if probability <= 0.0:
                    continue
                posterior = update_joint_hypothesis_belief(
                    belief, observed_likelihood(model, action, symbol)
                )
                continuation = plan(
                    posterior,
                    model,
                    config,
                    tuple(item for item in remaining_actions if item != action),
                    depth - 1,
                )
                expected_future += probability * continuation["selected_expected_cost"]
                branches.append(
                    {
                        "probability": probability,
                        "observation": symbol,
                        "posterior": posterior,
                        "continuation_action": continuation["selected_action"],
                        "continuation_cost": continuation["selected_expected_cost"],
                    }
                )
            values.append(
                {
                    "action": action,
                    "kind": "action_conditioned_future_belief",
                    "expected_cost": float(config["costs"][action]) + expected_future,
                    "stage_cost": float(config["costs"][action]),
                    "expected_future_cost": expected_future,
                    "observation_branches": branches,
                    "future_held_out_observation_used": False,
                }
            )
    selected = min(values, key=lambda item: (item["expected_cost"], item["action"]))
    return {
        "planner": "direct_joint_belief_expected_task_cost_mpc",
        "horizon": depth,
        "feedback_policy": True,
        "action_values": values,
        "selected_action": selected["action"],
        "selected_expected_cost": selected["expected_cost"],
        "marginal_confidence_product_used": False,
        "future_held_out_observation_used_for_action_selection": False,
    }


def greedy_view(belief: dict[str, float], model: dict[str, Any], actions: tuple[str, ...]) -> str:
    values = []
    for action in actions:
        expected_peak = 0.0
        for symbol, probability in predictive_observations(belief, model, action).items():
            posterior = update_joint_hypothesis_belief(
                belief, observed_likelihood(model, action, symbol)
            )
            expected_peak += probability * max(posterior.values())
        values.append((expected_peak, action))
    return max(values, key=lambda item: (item[0], item[1]))[1]


def execute_episode(
    episode: dict[str, dict[str, Any]], model: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    belief, first_update = update_from_row(model["prior"], episode["initial_observation"], model)
    updates = [first_update]
    graphs = [
        joint_scene_graph_snapshot(
            step=0,
            view="center",
            joint_belief=belief,
            candidate_track_ids=TRACK_IDS,
            observation={"symbol": first_update["observation"]},
        )
    ]
    remaining = tuple(config["actions"])
    policies = []
    sequence = []
    first_selected = None
    for step in range(int(config["maximum_view_actions"]) + 1):
        policy = plan(belief, model, config, remaining, min(len(remaining), int(config["maximum_view_actions"]) - step))
        policies.append(policy)
        action = policy["selected_action"]
        sequence.append(action)
        if first_selected is None:
            first_selected = action
        if action.startswith("grasp:") or action == "defer":
            break
        belief, update = update_from_row(belief, episode[action], model)
        updates.append(update)
        graphs.append(
            joint_scene_graph_snapshot(
                step=step + 1,
                view=action.removeprefix("viewpoint_"),
                joint_belief=belief,
                candidate_track_ids=TRACK_IDS,
                observation={"symbol": update["observation"]},
            )
        )
        remaining = tuple(item for item in remaining if item != action)
    terminal = sequence[-1]
    truth = state_for_episode(episode)
    terminal_hypothesis = terminal.removeprefix("grasp:").replace(":", "|")
    success = terminal.startswith("grasp:") and truth == terminal_hypothesis
    wrong = terminal.startswith("grasp:") and not success
    greedy = greedy_view(
        first_update["posterior"], model, tuple(config["actions"])
    )
    return {
        "action_sequence": sequence,
        "first_action": first_selected,
        "greedy_first_view": greedy,
        "proposed_greedy_disagreement": first_selected != greedy,
        "terminal_action": terminal,
        "task_success": success,
        "wrong_commitment": wrong,
        "noncompletion": terminal == "defer",
        "true_joint_state_posthoc": truth,
        "belief_updates": updates,
        "scene_graph_steps": graphs,
        "policies": policies,
        "joint_belief_used": True,
        "marginal_confidence_product_used": False,
    }


def run(config_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    config = load_json(config_path)
    source_config = load_json(resolve_path(config["source_config"]))
    episodes_all = build_episode_rows(source_config)
    seeds = [int(seed) for seed in config["development_seeds"]]
    episodes = {seed: episodes_all[seed] for seed in seeds}
    rows = []
    models = {}
    for seed in seeds:
        training = {key: value for key, value in episodes.items() if key != seed}
        model = fit_joint_model(training, list(config["actions"]), float(config["dirichlet_alpha"]))
        row = execute_episode(episodes[seed], model, config)
        row.update({"seed": seed, "held_out_episode_used_for_fit": False})
        rows.append(row)
        models[str(seed)] = model
    first_counts = Counter(row["first_action"] for row in rows)
    summary = {
        "episode_count": len(rows),
        "task_success_count": sum(row["task_success"] for row in rows),
        "task_success_rate": sum(row["task_success"] for row in rows) / len(rows),
        "wrong_commitment_count": sum(row["wrong_commitment"] for row in rows),
        "wrong_commitment_rate": sum(row["wrong_commitment"] for row in rows) / len(rows),
        "noncompletion_count": sum(row["noncompletion"] for row in rows),
        "first_action_counts": dict(first_counts),
        "proposed_greedy_disagreements": sum(row["proposed_greedy_disagreement"] for row in rows),
    }
    acceptance = config["acceptance"]
    checks = {
        "complete_episode_count": len(rows) >= int(acceptance["minimum_complete_episodes"]),
        "wrong_commitment_rate": summary["wrong_commitment_rate"] <= float(acceptance["maximum_wrong_commitment_rate"]),
        "first_action_diversity": len(first_counts) >= int(acceptance["minimum_distinct_first_actions"]),
        "proposed_greedy_disagreement": summary["proposed_greedy_disagreements"] >= int(acceptance["minimum_proposed_greedy_disagreements"]),
    }
    result = {
        "schema_version": "icra-v13-joint-development-result-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "summary": summary,
        "acceptance_checks": checks,
        "episodes": rows,
        "leave_one_episode_out_models": models,
        "candidate_identity": "persistent_rgbd_center_track",
        "joint_hypothesis": "candidate_track_x_membership",
        "joint_observation_likelihood_fitted_directly": True,
        "marginal_confidence_product_used": False,
        "simulator_ground_truth_used_for_development_fit_only": True,
        "held_out_future_observation_used_for_action_selection": False,
        "runtime_seconds": time.perf_counter() - started,
        "gpu_used": False,
        "cached_perception_reused": True,
        "training_performed": False,
        "calibration_performed": False,
        "testing_performed": False,
        "valid_for_final_evaluation": False,
        "limitations": [
            "Development replay only; the source episodes are not a new calibration or unopened test split.",
            "The joint observation model must be refit on the new V13 calibration split before final evaluation.",
            "Physical grasp execution is not part of this cached stress test."
        ]
    }
    output_root = resolve_path(config["output_root"])
    development_model = fit_joint_model(
        episodes, list(config["actions"]), float(config["dirichlet_alpha"])
    )
    development_model.update(
        {
            "schema_version": "icra-v13-direct-joint-observation-model-development-v1",
            "status": "development_integration_only_not_frozen_calibration",
            "candidate_identity": "persistent_rgbd_track",
            "training_performed": False,
            "calibration_performed": False,
            "testing_performed": False,
            "valid_for_final_evaluation": False,
        }
    )
    write_json(output_root / "result.json", result)
    write_json(output_root / "development_joint_observation_model.json", development_model)
    write_json(output_root / "summary.json", {key: value for key, value in result.items() if key not in {"episodes", "leave_one_episode_out_models"}})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = run(args.config.resolve())
    print(json.dumps({"status": result["status"], "summary": result["summary"], "acceptance_checks": result["acceptance_checks"]}, indent=2))


if __name__ == "__main__":
    main()
