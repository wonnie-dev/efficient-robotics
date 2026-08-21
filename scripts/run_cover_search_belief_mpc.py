"""CPU-only belief-tree MPC for removable-cover search.

The planner models a discrete target-location and cover-state belief. It
forecasts action-conditioned observations for viewpoints and remove_cover,
executes only the first scripted diagnostic action, updates from positive or
negative evidence, and replans. No simulator, robot, learned model, or GPU is
used.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "cover_search_belief_mpc_cpu_pilot.json"
)


def load_json(path: Path) -> dict[str, Any]:
    """Load a planner configuration or observation artifact."""
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    """Resolve a repository-relative path."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def normalize(distribution: dict[str, float]) -> dict[str, float]:
    """Normalize a nonzero discrete probability distribution."""
    total = float(sum(distribution.values()))
    if total <= 0.0:
        raise ValueError("Cannot normalize a zero-mass distribution")
    return {
        state: float(probability) / total
        for state, probability in distribution.items()
    }


def validate_config(config: dict[str, Any]) -> None:
    """Validate state, transition, observation, and evaluation invariants."""
    states = tuple(str(state) for state in config["state_space"])
    if set(config["initial_belief"]) != set(states):
        raise ValueError("Initial belief does not cover state_space")
    if not math.isclose(
        sum(float(value) for value in config["initial_belief"].values()),
        1.0,
    ):
        raise ValueError("Initial belief must sum to one")
    for action_name, model in config["observation_model"].items():
        if set(model["likelihood"]) != set(states):
            raise ValueError(
                f"{action_name} likelihood does not cover state_space"
            )
        outcomes = set(model["outcomes"])
        for state, likelihood in model["likelihood"].items():
            if set(likelihood) != outcomes:
                raise ValueError(
                    f"{action_name}/{state} does not cover outcomes"
                )
            if not math.isclose(sum(likelihood.values()), 1.0):
                raise ValueError(
                    f"{action_name}/{state} likelihood does not sum to one"
                )
    if config.get("training_performed") is not False:
        raise ValueError("Model-weight training must remain disabled")
    if config.get("testing_performed") is not False:
        raise ValueError("This pilot is not testing")


def split_state(state: str) -> tuple[str, str]:
    """Split a joint target-location and cover-state label."""
    location, cover = state.split("|", maxsplit=1)
    if location not in {"inside", "outside_near"}:
        raise ValueError(f"Unknown target location: {location}")
    if cover not in {"covered", "open"}:
        raise ValueError(f"Unknown cover state: {cover}")
    return location, cover


def state_name(location: str, cover: str) -> str:
    """Create and validate a joint state label."""
    value = f"{location}|{cover}"
    split_state(value)
    return value


def entropy(distribution: dict[str, float]) -> float:
    """Return Shannon entropy in nats."""
    return -sum(
        probability * math.log(probability)
        for probability in distribution.values()
        if probability > 0.0
    )


def marginal_location(
    belief: dict[str, float],
) -> dict[str, float]:
    """Marginalize a joint belief over the target-location variable."""
    result = {"inside": 0.0, "outside_near": 0.0}
    for state, probability in belief.items():
        location, _ = split_state(state)
        result[location] += probability
    return result


def marginal_cover(belief: dict[str, float]) -> dict[str, float]:
    """Marginalize a joint belief over the cover-state variable."""
    result = {"covered": 0.0, "open": 0.0}
    for state, probability in belief.items():
        _, cover = split_state(state)
        result[cover] += probability
    return result


def transition_state_distribution(
    state: str,
    action_name: str,
    config: dict[str, Any],
) -> dict[str, float]:
    """Predict the latent-state distribution after one action."""
    transition = config["transition_model"][action_name]
    transition_type = str(transition["type"])
    if transition_type == "identity":
        return {state: 1.0}
    if transition_type != "covered_to_open":
        raise ValueError(f"Unknown transition type: {transition_type}")
    location, cover = split_state(state)
    if cover == "open":
        return {state: 1.0}
    success = float(transition["success_probability"])
    return {
        state_name(location, "open"): success,
        state_name(location, "covered"): 1.0 - success,
    }


def predict_belief(
    belief: dict[str, float],
    action_name: str,
    config: dict[str, Any],
) -> dict[str, float]:
    """Push the current belief through the selected action's transition model."""
    predicted = {state: 0.0 for state in config["state_space"]}
    for state, probability in belief.items():
        for next_state, transition_probability in (
            transition_state_distribution(
                state, action_name, config
            ).items()
        ):
            predicted[next_state] += (
                probability * transition_probability
            )
    return normalize(predicted)


def observation_likelihood(
    action_name: str,
    outcome: str,
    config: dict[str, Any],
    *,
    negative_evidence_enabled: bool,
) -> dict[str, float]:
    """Return the state likelihood of an outcome for the executed action."""
    model = config["observation_model"][action_name]
    if outcome not in model["outcomes"]:
        raise ValueError(
            f"Unknown outcome for {action_name}: {outcome}"
        )
    likelihood = {
        state: float(values[outcome])
        for state, values in model["likelihood"].items()
    }
    if (
        action_name == "remove_cover"
        and outcome == "empty_container"
        and not negative_evidence_enabled
    ):
        # The ablation preserves evidence about the cover transition while
        # removing the empty-container clue about target location.
        open_average = sum(
            likelihood[state]
            for state in likelihood
            if split_state(state)[1] == "open"
        ) / 2.0
        covered_average = sum(
            likelihood[state]
            for state in likelihood
            if split_state(state)[1] == "covered"
        ) / 2.0
        likelihood = {
            state: (
                open_average
                if split_state(state)[1] == "open"
                else covered_average
            )
            for state in likelihood
        }
    return likelihood


def update_belief(
    predicted: dict[str, float],
    action_name: str,
    outcome: str,
    config: dict[str, Any],
    *,
    negative_evidence_enabled: bool,
) -> dict[str, float]:
    """Apply Bayes' rule to the post-transition belief."""
    likelihood = observation_likelihood(
        action_name,
        outcome,
        config,
        negative_evidence_enabled=negative_evidence_enabled,
    )
    return normalize(
        {
            state: predicted[state] * likelihood[state]
            for state in predicted
        }
    )


def predictive_observation_distribution(
    predicted: dict[str, float],
    action_name: str,
    config: dict[str, Any],
    *,
    negative_evidence_enabled: bool,
) -> dict[str, float]:
    """Marginalize latent states into action-conditioned outcome branches."""
    return normalize(
        {
            outcome: sum(
                predicted[state]
                * observation_likelihood(
                    action_name,
                    outcome,
                    config,
                    negative_evidence_enabled=(
                        negative_evidence_enabled
                    ),
                )[state]
                for state in predicted
            )
            for outcome in config["observation_model"][action_name][
                "outcomes"
            ]
        }
    )


def grasp_success_probability(
    action: dict[str, Any],
    belief: dict[str, float],
) -> float:
    """Sum belief mass over the states in which a grasp succeeds."""
    return sum(
        belief[state] for state in action["success_states"]
    )


def terminal_action_values(
    belief: dict[str, float],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Score safe terminal choices, including wrong-commitment risk."""
    objective = config["objective"]
    minimum_success = float(
        objective["minimum_grasp_success_probability"]
    )
    values = []
    for action_name, action in config["actions"].items():
        if not action["enabled"]:
            continue
        kind = action["kind"]
        if kind == "terminal_defer":
            values.append(
                {
                    "action": action_name,
                    "kind": kind,
                    "cost": float(
                        objective["noncompletion_cost"]
                    ),
                    "success_probability": None,
                    "blocking_reasons": [],
                }
            )
        elif kind == "terminal_grasp":
            success = grasp_success_probability(action, belief)
            reasons = (
                []
                if success >= minimum_success
                else ["grasp_success_probability_below_gate"]
            )
            if not reasons:
                values.append(
                    {
                        "action": action_name,
                        "kind": kind,
                        "cost": (
                            float(action["stage_cost"])
                            + float(
                                objective[
                                    "wrong_commitment_weight"
                                ]
                            )
                            * (1.0 - success)
                        ),
                        "success_probability": success,
                        "blocking_reasons": [],
                    }
                )
    return values


def information_action_feasibility(
    action_name: str,
    belief: dict[str, float],
    config: dict[str, Any],
    used_observation_actions: frozenset[str],
) -> dict[str, Any]:
    """Check whether an information-gathering action is currently feasible."""
    action = config["actions"][action_name]
    if action["kind"] == "observation":
        feasible = action_name not in used_observation_actions
        return {
            "feasible": feasible,
            "blocking_reasons": (
                [] if feasible else ["viewpoint_already_used"]
            ),
        }
    if action["kind"] == "interaction_observation":
        covered_probability = marginal_cover(belief)["covered"]
        minimum = float(
            action.get("minimum_covered_probability", 0.0)
        )
        feasible = covered_probability >= minimum
        return {
            "feasible": feasible,
            "covered_probability": covered_probability,
            "minimum_covered_probability": minimum,
            "blocking_reasons": (
                [] if feasible else ["cover_already_open"]
            ),
        }
    return {
        "feasible": False,
        "blocking_reasons": ["not_an_information_action"],
    }


def belief_tree_action_values(
    belief: dict[str, float],
    config: dict[str, Any],
    *,
    depth: int,
    used_observation_actions: frozenset[str] = frozenset(),
    negative_evidence_enabled: bool = True,
) -> list[dict[str, Any]]:
    """Evaluate feedback actions using modeled, not realized, observations."""
    terminal = terminal_action_values(belief, config)
    if depth <= 1:
        return terminal
    values = list(terminal)
    for action_name, action in config["actions"].items():
        if (
            not action["enabled"]
            or action["kind"]
            not in {"observation", "interaction_observation"}
        ):
            continue
        feasibility = information_action_feasibility(
            action_name,
            belief,
            config,
            used_observation_actions,
        )
        if not feasibility["feasible"]:
            continue
        predicted = predict_belief(belief, action_name, config)
        outcome_probabilities = predictive_observation_distribution(
            predicted,
            action_name,
            config,
            negative_evidence_enabled=negative_evidence_enabled,
        )
        branches = []
        expected_future_cost = 0.0
        for outcome, probability in outcome_probabilities.items():
            # Each branch gets its own posterior and continuation; only the
            # probability-weighted continuation cost affects the root choice.
            posterior = update_belief(
                predicted,
                action_name,
                outcome,
                config,
                negative_evidence_enabled=negative_evidence_enabled,
            )
            next_used = used_observation_actions
            if action["kind"] == "observation":
                next_used = next_used | {action_name}
            continuations = belief_tree_action_values(
                posterior,
                config,
                depth=depth - 1,
                used_observation_actions=next_used,
                negative_evidence_enabled=negative_evidence_enabled,
            )
            continuation = min(
                continuations, key=lambda item: item["cost"]
            )
            expected_future_cost += (
                probability * float(continuation["cost"])
            )
            branches.append(
                {
                    "probability": probability,
                    "observation": outcome,
                    "posterior": posterior,
                    "continuation_action": continuation["action"],
                    "continuation_cost": continuation["cost"],
                }
            )
        values.append(
            {
                "action": action_name,
                "kind": action["kind"],
                "cost": (
                    float(action["stage_cost"])
                    + expected_future_cost
                ),
                "stage_cost": float(action["stage_cost"]),
                "expected_future_cost": expected_future_cost,
                "observation_branches": branches,
                "branch_probability_sum": sum(
                    branch["probability"] for branch in branches
                ),
                "future_observation_used_for_selection": False,
            }
        )
    return values


def plan(
    belief: dict[str, float],
    config: dict[str, Any],
    *,
    negative_evidence_enabled: bool = True,
) -> dict[str, Any]:
    """Choose the minimum expected-cost first action from a belief tree."""
    normalized = normalize(belief)
    values = belief_tree_action_values(
        normalized,
        config,
        depth=int(config["horizon"]),
        negative_evidence_enabled=negative_evidence_enabled,
    )
    selected = min(values, key=lambda item: item["cost"])
    return {
        "planner": "exact_discrete_cover_search_belief_tree_mpc",
        "horizon": int(config["horizon"]),
        "feedback_policy": True,
        "replans_after_first_action": True,
        "negative_evidence_enabled": negative_evidence_enabled,
        "current_belief": normalized,
        "location_marginal": marginal_location(normalized),
        "cover_marginal": marginal_cover(normalized),
        "entropy_nats": entropy(normalized),
        "action_values": values,
        "selected_action": selected["action"],
        "selected_cost": selected["cost"],
        "future_observation_used_for_action_selection": False,
    }


def execute_observation_action(
    belief: dict[str, float],
    action_name: str,
    outcome: str,
    config: dict[str, Any],
    *,
    negative_evidence_enabled: bool = True,
) -> dict[str, Any]:
    """Advance belief after the selected action returns its actual outcome."""
    predicted = predict_belief(belief, action_name, config)
    posterior = update_belief(
        predicted,
        action_name,
        outcome,
        config,
        negative_evidence_enabled=negative_evidence_enabled,
    )
    return {
        "action": action_name,
        "observation": outcome,
        "prior": normalize(belief),
        "predicted_after_transition": predicted,
        "posterior": posterior,
        "location_before": marginal_location(normalize(belief)),
        "location_after": marginal_location(posterior),
        "cover_before": marginal_cover(normalize(belief)),
        "cover_after": marginal_cover(posterior),
        "negative_evidence_applied": (
            negative_evidence_enabled
            and action_name == "remove_cover"
            and outcome == "empty_container"
        ),
        "observation_arrived_after_action_selection": True,
    }


def run_scripted_episode(
    episode: dict[str, Any],
    config: dict[str, Any],
    *,
    negative_evidence_enabled: bool = True,
) -> dict[str, Any]:
    """Exercise the replan loop while keeping scripted evidence causally late."""
    belief = normalize(config["initial_belief"])
    steps = []
    observation_index = 0
    terminal_action = None
    status = "running"
    maximum_steps = len(episode["observations"]) + 2
    for step_index in range(maximum_steps):
        policy = plan(
            belief,
            config,
            negative_evidence_enabled=negative_evidence_enabled,
        )
        selected = str(policy["selected_action"])
        action = config["actions"][selected]
        step = {
            "step_index": step_index,
            "policy": policy,
            "selected_action": selected,
        }
        if action["kind"].startswith("terminal_"):
            terminal_action = selected
            status = "completed"
            steps.append(step)
            break
        if observation_index >= len(episode["observations"]):
            # A missing result is a failed episode, not evidence for any state.
            status = "failed_missing_scripted_observation"
            steps.append(step)
            break
        scripted = episode["observations"][observation_index]
        if selected != scripted["expected_action"]:
            # Stop before consuming an outcome that belongs to another action.
            status = "failed_unexpected_planner_action"
            step["expected_action"] = scripted["expected_action"]
            steps.append(step)
            break
        update = execute_observation_action(
            belief,
            selected,
            str(scripted["outcome"]),
            config,
            negative_evidence_enabled=negative_evidence_enabled,
        )
        step["executed_observation"] = update
        belief = update["posterior"]
        observation_index += 1
        steps.append(step)
    expected_terminal = episode["expected_terminal_action"]
    return {
        "episode_id": episode["episode_id"],
        "status": status,
        "negative_evidence_enabled": negative_evidence_enabled,
        "true_state_posthoc": episode["true_state_posthoc"],
        "terminal_action": terminal_action,
        "expected_terminal_action": expected_terminal,
        "terminal_matches_expected": terminal_action == expected_terminal,
        "observation_count": observation_index,
        "steps": steps,
        "final_belief": belief,
        "final_location_marginal": marginal_location(belief),
        "future_scripted_observations_used_during_planning": False,
        "true_state_used_during_planning": False,
    }


def run_experiment(config_path: Path) -> dict[str, Any]:
    """Run a configured CPU belief-tree replay and save its trace."""
    config = load_json(config_path)
    validate_config(config)
    started = time.perf_counter()
    episodes = [
        run_scripted_episode(episode, config)
        for episode in config["scripted_diagnostic_episodes"]
    ]
    negative_episode = next(
        episode
        for episode in config["scripted_diagnostic_episodes"]
        if episode["episode_id"]
        == "target_outside_negative_evidence"
    )
    ablation = run_scripted_episode(
        negative_episode,
        config,
        negative_evidence_enabled=False,
    )
    negative_result = next(
        episode
        for episode in episodes
        if episode["episode_id"]
        == "target_outside_negative_evidence"
    )
    result = {
        "schema_version": "cover-search-belief-mpc-cpu-pilot-v1",
        "experiment_id": config["experiment_id"],
        "status": (
            "completed"
            if all(
                episode["status"] == "completed"
                and episode["terminal_matches_expected"]
                for episode in episodes
            )
            else "partial_failure"
        ),
        "purpose": config["purpose"],
        "episodes": episodes,
        "negative_evidence_ablation": ablation,
        "negative_evidence_comparison": {
            "prior_inside_probability": float(
                config["initial_belief"]["inside|covered"]
            ),
            "inside_probability_after_empty_with_negative_evidence": (
                negative_result["steps"][0][
                    "executed_observation"
                ]["location_after"]["inside"]
            ),
            "inside_probability_after_empty_without_negative_evidence": (
                ablation["steps"][0][
                    "executed_observation"
                ]["location_after"]["inside"]
            ),
            "terminal_with_negative_evidence": negative_result[
                "terminal_action"
            ],
            "terminal_without_negative_evidence": ablation[
                "terminal_action"
            ],
            "actions_with_negative_evidence": [
                step["selected_action"]
                for step in negative_result["steps"]
            ],
            "actions_without_negative_evidence": [
                step["selected_action"]
                for step in ablation["steps"]
            ],
            "observation_count_with_negative_evidence": (
                negative_result["observation_count"]
            ),
            "observation_count_without_negative_evidence": (
                ablation["observation_count"]
            ),
        },
        "runtime_seconds": time.perf_counter() - started,
        "gpu_used": False,
        "gpu_memory_gib": 0.0,
        "isaac_sim_launched": False,
        "vlm_inference_performed": False,
        "robot_motion_mpc_executed": False,
        "cover_manipulation_executed": False,
        "abstract_remove_cover_action_evaluated": True,
        "training_performed": False,
        "calibration_performed": False,
        "testing_performed": False,
        "valid_for_final_evaluation": False,
        "limitations": [
            "Transition, observation, and task-cost values are hand specified.",
            "Scripted observations arrive only after policy action selection.",
            "No continuous UR10e trajectory, collision, RG6 contact, or cover physics was executed.",
            "This validates control flow and negative evidence, not paper performance.",
        ],
    }
    output_root = resolve_path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    result_path = output_root / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        key: value
        for key, value in result.items()
        if key not in {"episodes", "negative_evidence_ablation"}
    }
    summary["episode_summaries"] = [
        {
            "episode_id": episode["episode_id"],
            "status": episode["status"],
            "actions": [
                step["selected_action"] for step in episode["steps"]
            ],
            "terminal_action": episode["terminal_action"],
            "terminal_matches_expected": episode[
                "terminal_matches_expected"
            ],
            "final_location_marginal": episode[
                "final_location_marginal"
            ],
        }
        for episode in episodes
    ]
    summary["ablation_summary"] = {
        "status": ablation["status"],
        "actions": [
            step["selected_action"] for step in ablation["steps"]
        ],
        "terminal_action": ablation["terminal_action"],
        "final_location_marginal": ablation[
            "final_location_marginal"
        ],
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"COVER_MPC_RESULT={result_path}")
    print(f"COVER_MPC_SUMMARY={summary_path}")
    return result


def main() -> None:
    """Run the configured belief-space planning experiment."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run_experiment(args.config.resolve())


if __name__ == "__main__":
    main()
