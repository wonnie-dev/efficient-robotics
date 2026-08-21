"""CPU-only leave-one-episode-out replay for action-conditioned belief MPC.

The runner consumes only saved learned-perception outputs.  For each replayed
episode, the observation and transition model is fitted from the other
episodes.  The held-out future view is read only after the policy has selected
that action.  Simulator labels are used for model fitting and post-hoc audit,
never as the held-out planner observation.

This remains calibration replay.  It does not run Isaac Sim, load a VLM,
train model weights, consume reserved test seeds, or produce final-paper
performance.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "offline_action_conditioned_mpc_replay_seed165_184.json"
)

VISIBILITY_STATES = ("visible", "fully_hidden")
MEMBERSHIP_STATES = ("inside", "outside")
OCCLUSION_STATES = ("yes", "no")
MEMBERSHIP_OBSERVATIONS = ("inside", "outside", "unknown", "missing")
OCCLUSION_OBSERVATIONS = ("yes", "no", "unknown", "missing")
TRACK_STATES = ("correct", "incorrect")
TRACK_AGREEMENT_OBSERVATIONS = ("same", "different", "missing")
PERCEPTION_STATES = tuple(
    f"{visibility}|{occlusion}"
    for visibility in VISIBILITY_STATES
    for occlusion in OCCLUSION_STATES
)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(values: dict[str, float]) -> dict[str, float]:
    total = float(sum(values.values()))
    if total <= 0.0:
        return {key: 1.0 / len(values) for key in values}
    return {key: float(value) / total for key, value in values.items()}


def sigmoid(value: float) -> float:
    if value >= 0.0:
        exponential = math.exp(-value)
        return 1.0 / (1.0 + exponential)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def confidence_bin(
    probability: float,
    bin_settings: list[dict[str, Any]],
) -> str:
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Probability is outside [0, 1]: {probability}")
    for item in bin_settings:
        if probability < float(item["maximum_exclusive"]):
            return str(item["name"])
    raise ValueError(f"No confidence bin contains {probability}")


def perception_state(visibility: str, occlusion: str) -> str:
    value = f"{visibility}|{occlusion}"
    if value not in PERCEPTION_STATES:
        raise ValueError(f"Invalid joint perception state: {value}")
    return value


def perception_observation(
    identity_bin: str,
    occlusion_observation: str,
) -> str:
    if occlusion_observation not in OCCLUSION_OBSERVATIONS:
        raise ValueError(
            f"Invalid reference occlusion observation: "
            f"{occlusion_observation}"
        )
    return f"{identity_bin}|{occlusion_observation}"


def track_belief_observation(
    agreement: str,
    confidence: str,
) -> str:
    if agreement not in (*TRACK_AGREEMENT_OBSERVATIONS, "initial"):
        raise ValueError(f"Invalid track agreement: {agreement}")
    return f"{agreement}|{confidence}"


def split_perception_state(value: str) -> tuple[str, str]:
    visibility, occlusion = value.split("|", maxsplit=1)
    if value not in PERCEPTION_STATES:
        raise ValueError(f"Invalid joint perception state: {value}")
    return visibility, occlusion


def entropy_normalized(distribution: dict[str, float]) -> float:
    if len(distribution) <= 1:
        return 0.0
    value = -sum(
        probability * math.log(probability)
        for probability in distribution.values()
        if probability > 0.0
    )
    return value / math.log(len(distribution))


def categorical_prior(
    rows: list[dict[str, Any]],
    key: str,
    outcomes: tuple[str, ...],
    alpha: float,
) -> dict[str, float]:
    counts = Counter(str(row[key]) for row in rows)
    denominator = len(rows) + alpha * len(outcomes)
    return {
        outcome: (counts[outcome] + alpha) / denominator
        for outcome in outcomes
    }


def categorical_likelihood(
    rows: list[dict[str, Any]],
    *,
    action_values: tuple[str, ...],
    state_key: str,
    states: tuple[str, ...],
    outcome_key: str,
    outcomes: tuple[str, ...],
    alpha: float,
) -> dict[str, dict[str, dict[str, float]]]:
    """Estimate P(outcome | state, action) with symmetric smoothing."""
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    totals: Counter[tuple[str, str]] = Counter()
    for row in rows:
        action = str(row["model_action"])
        state = str(row[state_key])
        outcome = str(row[outcome_key])
        if action not in action_values:
            continue
        if state not in states or outcome not in outcomes:
            raise ValueError(
                f"Invalid likelihood row: {action=} {state=} {outcome=}"
            )
        grouped[(action, state)][outcome] += 1
        totals[(action, state)] += 1
    table = {}
    for action in action_values:
        table[action] = {}
        for state in states:
            denominator = totals[(action, state)] + alpha * len(outcomes)
            table[action][state] = {
                outcome: (
                    grouped[(action, state)][outcome] + alpha
                )
                / denominator
                for outcome in outcomes
            }
    return table


def transition_likelihood(
    transitions: list[dict[str, Any]],
    *,
    action_values: tuple[str, ...],
    current_key: str,
    next_key: str,
    states: tuple[str, ...],
    alpha: float,
) -> dict[str, dict[str, dict[str, float]]]:
    return categorical_likelihood(
        transitions,
        action_values=action_values,
        state_key=current_key,
        states=states,
        outcome_key=next_key,
        outcomes=states,
        alpha=alpha,
    )


def identity_reliability(
    rows: list[dict[str, Any]],
    *,
    action_values: tuple[str, ...],
    identity_bins: tuple[str, ...],
    alpha: float,
    beta: float,
) -> dict[str, dict[str, dict[str, float]]]:
    grouped: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        action = str(row["model_action"])
        identity_bin = str(row["identity_bin"])
        if action not in action_values or identity_bin not in identity_bins:
            continue
        grouped[(action, identity_bin)]["count"] += 1
        grouped[(action, identity_bin)]["correct"] += int(
            bool(row["selected_target_correct_posthoc"])
        )
    result = {}
    for action in action_values:
        result[action] = {}
        for identity_bin in identity_bins:
            count = grouped[(action, identity_bin)]["count"]
            correct = grouped[(action, identity_bin)]["correct"]
            result[action][identity_bin] = {
                "selected_target_correct_probability": (
                    correct + alpha
                )
                / (count + alpha + beta),
                "count": count,
                "correct": correct,
                "beta_alpha": alpha,
                "beta_beta": beta,
            }
    return result


def bayesian_observation_update(
    prior: dict[str, float],
    likelihood: dict[str, dict[str, float]],
    outcome: str,
) -> dict[str, float]:
    """Apply one categorical observation without collapsing the belief to a label."""
    return normalize(
        {
            state: probability * likelihood[state][outcome]
            for state, probability in prior.items()
        }
    )


def predict_state(
    belief: dict[str, float],
    transition: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Propagate a belief through a state transition before observing again."""
    return normalize(
        {
            next_state: sum(
                belief[current_state]
                * transition[current_state][next_state]
                for current_state in belief
            )
            for next_state in belief
        }
    )


def predictive_outcome_distribution(
    belief: dict[str, float],
    likelihood: dict[str, dict[str, float]],
    outcomes: tuple[str, ...],
) -> dict[str, float]:
    """Marginalize hidden state to obtain P(observation | current belief)."""
    return normalize(
        {
            outcome: sum(
                belief[state] * likelihood[state][outcome]
                for state in belief
            )
            for outcome in outcomes
        }
    )


def reference_occlusion_state(ground_truth: dict, view_id: str) -> str:
    measurement = ground_truth[
        "objective_reference_occlusion_ground_truth"
    ][view_id]
    if not measurement["valid"]:
        raise ValueError(f"Invalid reference occlusion GT for {view_id}")
    return "no" if measurement["severity"] == "no" else "yes"


def total_visibility_state(ground_truth: dict, view_id: str) -> str:
    measurement = ground_truth["objective_occlusion_ground_truth"][view_id]
    if not measurement["valid"]:
        raise ValueError(f"Invalid total visibility GT for {view_id}")
    return "fully_hidden" if measurement["fully_hidden"] else "visible"


def euclidean_distance(
    first: list[float],
    second: list[float],
) -> float:
    return math.sqrt(
        sum((float(left) - float(right)) ** 2 for left, right in zip(first, second))
    )


def assign_episode_tracks(
    episode: dict[str, dict[str, Any]],
    *,
    maximum_center_distance_m: float,
) -> None:
    """Assign runtime tracks by nearest learned RGB-D candidate center."""
    center = episode["initial_observation"]
    center_candidates = [
        item
        for item in center["candidate_observations"]
        if item["center_world_m"] is not None
    ]
    track_centers = {
        f"track_{index:03d}": item["center_world_m"]
        for index, item in enumerate(center_candidates, start=1)
    }
    center_candidate_to_track = {
        item["candidate_id"]: track_id
        for track_id, item in zip(track_centers, center_candidates)
    }
    center_selected_track = center_candidate_to_track.get(
        center["selected_candidate_id"]
    )
    for row in episode.values():
        candidate_to_track: dict[str, str] = {}
        track_candidates: dict[str, dict[str, Any]] = {}
        for candidate in row["candidate_observations"]:
            center_world = candidate["center_world_m"]
            if center_world is None or not track_centers:
                continue
            track_id, track_center = min(
                track_centers.items(),
                key=lambda item: euclidean_distance(
                    center_world, item[1]
                ),
            )
            distance = euclidean_distance(center_world, track_center)
            if distance > maximum_center_distance_m:
                continue
            candidate_to_track[candidate["candidate_id"]] = track_id
            track_candidates[track_id] = {
                **candidate,
                "distance_to_center_track_m": distance,
            }
        selected_track = candidate_to_track.get(
            row["selected_candidate_id"]
        )
        if row["action"] == "initial_observation":
            agreement = "initial"
        elif selected_track is None or center_selected_track is None:
            agreement = "missing"
        elif selected_track == center_selected_track:
            agreement = "same"
        else:
            agreement = "different"
        center_track_candidate = track_candidates.get(
            center_selected_track
        )
        if (
            row["action"] != "initial_observation"
            and center_track_candidate is None
        ):
            agreement = "missing"
        center_track_confidence_bin = (
            str(center_track_candidate["identity_bin"])
            if center_track_candidate is not None
            else "missing"
        )
        row.update(
            {
                "candidate_to_track": candidate_to_track,
                "track_candidates": track_candidates,
                "selected_track_id": selected_track,
                "center_selected_track_id": center_selected_track,
                "track_agreement_observation": agreement,
                "center_track_confidence_bin": (
                    center_track_confidence_bin
                ),
                "track_belief_observation": (
                    track_belief_observation(
                        agreement, center_track_confidence_bin
                    )
                ),
                "center_selected_track_available": (
                    center_track_candidate is not None
                ),
                "center_selected_track_correct_posthoc": (
                    bool(center_track_candidate["target_label_posthoc"])
                    if center_track_candidate is not None
                    else False
                ),
                "tracking_uses_simulator_ids": False,
            }
        )
        row["planner_observation_fields"].extend(
            [
                "selected_track_id",
                "center_selected_track_id",
                "track_agreement_observation",
                "center_track_confidence_bin",
                "track_belief_observation",
            ]
        )


def build_episode_rows(
    config: dict[str, Any],
    *,
    target_temperature: float | None = None,
) -> dict[int, dict[str, dict]]:
    """Convert saved perception artifacts into causal replay records by seed."""
    calibration_root = resolve_path(config["calibration_root"])
    records = {
        item["sample_id"]: item
        for item in load_json(
            calibration_root / "calibration_records.json"
        )["records"]
    }
    predictions = {
        item["sample_id"]: item
        for item in load_json(
            calibration_root
            / "hybrid_rgbd_relation"
            / "predictions.json"
        )["samples"]
    }
    perception_config = load_json(
        calibration_root / "perception_config.json"
    )
    temperature = float(
        config["target_temperature"]
        if target_temperature is None
        else target_temperature
    )
    if temperature <= 0.0:
        raise ValueError("Target temperature must be positive")
    view_to_action = {
        str(view): str(action)
        for view, action in config["views"].items()
    }
    episodes: dict[int, dict[str, dict]] = defaultdict(dict)
    for sample in perception_config["samples"]:
        sample_id = str(sample["sample_id"])
        seed = int(sample["seed"])
        view_id = Path(sample["observation_dir"]).name
        action = view_to_action[view_id]
        record = records[sample_id]
        prediction = predictions[sample_id]
        candidates = record["candidates"]
        predicted_candidates = {
            str(item["candidate_id"]): item
            for item in prediction["candidate_predictions"]
        }
        candidate_observations = []
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            predicted = predicted_candidates.get(candidate_id)
            center_world = (
                predicted["candidate_geometry"]["center_world_m"]
                if predicted is not None
                else None
            )
            raw_candidate_logit = float(candidate["raw_match_logit"])
            candidate_observations.append(
                {
                    "candidate_id": candidate_id,
                    "raw_match_logit": raw_candidate_logit,
                    "target_score": sigmoid(
                        raw_candidate_logit / temperature
                    ),
                    "identity_bin": confidence_bin(
                        sigmoid(raw_candidate_logit / temperature),
                        config["identity_confidence_bins"],
                    ),
                    "center_world_m": center_world,
                    "target_label_posthoc": bool(
                        candidate["target_label"]
                    ),
                }
            )
        if not candidates:
            raw_logit = None
            selected_candidate_id = None
            selected_correct = False
            selected_score = 0.0
            membership_observation = "missing"
            occlusion_observation = "missing"
        else:
            selected = max(
                candidates, key=lambda item: item["raw_match_logit"]
            )
            raw_logit = float(selected["raw_match_logit"])
            selected_candidate_id = str(selected["candidate_id"])
            selected_correct = bool(selected["target_label"])
            selected_score = sigmoid(raw_logit / temperature)
            relation = prediction["selected_candidate_relation_evidence"]
            if (
                relation is None
                or relation["candidate_id"] != selected_candidate_id
            ):
                membership_observation = "missing"
                occlusion_observation = "missing"
            else:
                membership_observation = str(
                    relation["membership_world_evidence"]["label"]
                )
                occlusion_observation = str(
                    relation["occluded_by_reference_evidence"]["label"]
                )
        ground_truth = load_json(
            resolve_path(sample["calibration_ground_truth_file"])
        )
        membership_state = str(
            ground_truth["world_ground_truth"]["entities"]["target_red"][
                "membership"
            ]
        )
        identity_bin = confidence_bin(
            selected_score, config["identity_confidence_bins"]
        )
        occlusion_state = reference_occlusion_state(
            ground_truth, view_id
        )
        visibility_state = total_visibility_state(
            ground_truth, view_id
        )
        row = {
            "sample_id": sample_id,
            "seed": seed,
            "view_id": view_id,
            "action": action,
            "model_action": action,
            "variant": str(sample["calibration_scene_variant"]),
            "selected_candidate_id": selected_candidate_id,
            "selected_raw_match_logit": raw_logit,
            "selected_target_score": selected_score,
            "target_temperature_applied": temperature,
            "candidate_observations": candidate_observations,
            "identity_bin": identity_bin,
            "membership_observation": membership_observation,
            "reference_occlusion_observation": occlusion_observation,
            "perception_observation": perception_observation(
                identity_bin, occlusion_observation
            ),
            "selected_target_correct_posthoc": selected_correct,
            "world_membership_state_posthoc": membership_state,
            "reference_occlusion_state_posthoc": occlusion_state,
            "total_visibility_state_posthoc": visibility_state,
            "perception_state_posthoc": perception_state(
                visibility_state, occlusion_state
            ),
            "planner_observation_fields": [
                "identity_bin",
                "membership_observation",
                "reference_occlusion_observation",
                "perception_observation",
                "target_temperature_applied",
            ],
            "posthoc_fields_not_available_to_planner": [
                "selected_target_correct_posthoc",
                "world_membership_state_posthoc",
                "reference_occlusion_state_posthoc",
                "total_visibility_state_posthoc",
                "perception_state_posthoc",
            ],
        }
        episodes[seed][action] = row
    expected_actions = set(view_to_action.values())
    for seed, rows in episodes.items():
        if set(rows) != expected_actions:
            raise ValueError(
                f"Seed {seed} actions {sorted(rows)} != "
                f"{sorted(expected_actions)}"
            )
        assign_episode_tracks(
            rows,
            maximum_center_distance_m=float(
                config["candidate_tracking"][
                    "maximum_center_distance_m"
                ]
            ),
        )
    return dict(episodes)


def pooled_action(action: str, action_agnostic: bool) -> str:
    if action_agnostic and action.startswith("viewpoint_"):
        return "viewpoint_any"
    return action


def fit_action_model(
    training_episodes: dict[int, dict[str, dict]],
    config: dict[str, Any],
    *,
    action_agnostic: bool,
) -> dict[str, Any]:
    """Fit observation and transition tables from calibration episodes only."""
    rows = []
    transitions = []
    for seed, episode in training_episodes.items():
        initial = episode["initial_observation"]
        for row in episode.values():
            copied = dict(row)
            copied["model_action"] = pooled_action(
                str(row["action"]), action_agnostic
            )
            rows.append(copied)
        for action in config["reachable_view_actions"]:
            future = episode[action]
            transitions.append(
                {
                    "seed": seed,
                    "model_action": pooled_action(
                        str(action), action_agnostic
                    ),
                    "current_total_visibility_state": initial[
                        "total_visibility_state_posthoc"
                    ],
                    "next_total_visibility_state": future[
                        "total_visibility_state_posthoc"
                    ],
                    "current_reference_occlusion_state": initial[
                        "reference_occlusion_state_posthoc"
                    ],
                    "next_reference_occlusion_state": future[
                        "reference_occlusion_state_posthoc"
                    ],
                    "current_perception_state": initial[
                        "perception_state_posthoc"
                    ],
                    "next_perception_state": future[
                        "perception_state_posthoc"
                    ],
                    "center_track_state": (
                        "correct"
                        if initial["selected_target_correct_posthoc"]
                        else "incorrect"
                    ),
                    "track_agreement_observation": future[
                        "track_agreement_observation"
                    ],
                    "track_belief_observation": future[
                        "track_belief_observation"
                    ],
                }
            )
    action_values = ["initial_observation"]
    if action_agnostic:
        action_values.append("viewpoint_any")
    else:
        action_values.extend(config["reachable_view_actions"])
    actions = tuple(action_values)
    identity_bins = tuple(
        str(item["name"]) for item in config["identity_confidence_bins"]
    )
    perception_observations = tuple(
        perception_observation(identity_bin, occlusion_observation)
        for identity_bin in identity_bins
        for occlusion_observation in OCCLUSION_OBSERVATIONS
    )
    track_belief_observations = tuple(
        [
            track_belief_observation(agreement, confidence)
            for agreement in ("same", "different")
            for confidence in identity_bins
        ]
        + [track_belief_observation("missing", "missing")]
    )
    alpha = float(config["dirichlet_alpha"])
    beta_alpha = float(config["beta_alpha"])
    beta_beta = float(config["beta_beta"])
    return {
        "action_agnostic": action_agnostic,
        "training_episode_count": len(training_episodes),
        "training_seeds": sorted(training_episodes),
        "actions": actions,
        "identity_bins": identity_bins,
        "perception_observations": perception_observations,
        "track_belief_observations": track_belief_observations,
        "priors": {
            "perception": categorical_prior(
                [
                    episode["initial_observation"]
                    for episode in training_episodes.values()
                ],
                "perception_state_posthoc",
                PERCEPTION_STATES,
                alpha,
            ),
            "membership": categorical_prior(
                [
                    episode["initial_observation"]
                    for episode in training_episodes.values()
                ],
                "world_membership_state_posthoc",
                MEMBERSHIP_STATES,
                alpha,
            ),
        },
        "perception_observation": categorical_likelihood(
            rows,
            action_values=actions,
            state_key="perception_state_posthoc",
            states=PERCEPTION_STATES,
            outcome_key="perception_observation",
            outcomes=perception_observations,
            alpha=alpha,
        ),
        "membership_observation": categorical_likelihood(
            rows,
            action_values=actions,
            state_key="world_membership_state_posthoc",
            states=MEMBERSHIP_STATES,
            outcome_key="membership_observation",
            outcomes=MEMBERSHIP_OBSERVATIONS,
            alpha=alpha,
        ),
        "identity_reliability": identity_reliability(
            rows,
            action_values=actions,
            identity_bins=identity_bins,
            alpha=beta_alpha,
            beta=beta_beta,
        ),
        "perception_transition": transition_likelihood(
            transitions,
            action_values=tuple(
                action for action in actions
                if action != "initial_observation"
            ),
            current_key="current_perception_state",
            next_key="next_perception_state",
            states=PERCEPTION_STATES,
            alpha=alpha,
        ),
        "track_belief_observation": categorical_likelihood(
            transitions,
            action_values=tuple(
                action
                for action in actions
                if action != "initial_observation"
            ),
            state_key="center_track_state",
            states=TRACK_STATES,
            outcome_key="track_belief_observation",
            outcomes=track_belief_observations,
            alpha=alpha,
        ),
        "fit_uses_simulator_ground_truth": True,
        "held_out_episode_used_for_fit": False,
    }


def model_action(action: str, model: dict[str, Any]) -> str:
    return pooled_action(action, bool(model["action_agnostic"]))


def initial_belief(model: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {
        name: dict(distribution)
        for name, distribution in model["priors"].items()
    }


def update_with_observation(
    belief: dict[str, dict[str, float]],
    row: dict[str, Any],
    model: dict[str, Any],
    *,
    action: str,
) -> tuple[dict[str, dict[str, float]], float]:
    selected_action = model_action(action, model)
    updated = {
        "perception": bayesian_observation_update(
            belief["perception"],
            model["perception_observation"][selected_action],
            str(row["perception_observation"]),
        ),
        "membership": bayesian_observation_update(
            belief["membership"],
            model["membership_observation"][selected_action],
            str(row["membership_observation"]),
        ),
    }
    if action == "initial_observation":
        selected_correct_probability = model["identity_reliability"][
            selected_action
        ][str(row["identity_bin"])][
            "selected_target_correct_probability"
        ]
        updated["target_track"] = {
            "correct": float(selected_correct_probability),
            "incorrect": 1.0 - float(selected_correct_probability),
        }
    else:
        updated["target_track"] = bayesian_observation_update(
            belief["target_track"],
            model["track_belief_observation"][selected_action],
            str(row["track_belief_observation"]),
        )
        selected_correct_probability = updated["target_track"]["correct"]
    return updated, float(selected_correct_probability)


def predict_after_action(
    belief: dict[str, dict[str, float]],
    model: dict[str, Any],
    action: str,
) -> dict[str, dict[str, float]]:
    """Fuse the executed action's measured outcome into every belief factor."""
    selected_action = model_action(action, model)
    return {
        "perception": predict_state(
            belief["perception"],
            model["perception_transition"][selected_action],
        ),
        "membership": dict(belief["membership"]),
        "target_track": dict(belief["target_track"]),
    }


def terminal_metrics(
    belief: dict[str, dict[str, float]],
    selected_target_correct_probability: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    costs = config["task_cost"]
    gate = config["commitment_gate"]
    wrong_target_risk = 1.0 - selected_target_correct_probability
    hidden_probability = sum(
        probability
        for state, probability in belief["perception"].items()
        if split_perception_state(state)[0] == "fully_hidden"
    )
    occlusion_probability = sum(
        probability
        for state, probability in belief["perception"].items()
        if split_perception_state(state)[1] == "yes"
    )
    membership_entropy = entropy_normalized(belief["membership"])
    grasp_cost = (
        float(costs["wrong_target_weight"]) * wrong_target_risk
        + float(costs["hidden_target_weight"]) * hidden_probability
        + float(costs["reference_occlusion_weight"])
        * occlusion_probability
        + float(costs["membership_entropy_weight"]) * membership_entropy
        + float(costs["grasp_motion_cost"])
    )
    blocking_reasons = []
    if selected_target_correct_probability < float(
        gate["minimum_selected_target_correct_probability"]
    ):
        blocking_reasons.append("selected_target_risk_above_gate")
    if hidden_probability > float(
        gate["maximum_fully_hidden_probability"]
    ):
        blocking_reasons.append("fully_hidden_risk_above_gate")
    if occlusion_probability > float(
        gate["maximum_reference_occlusion_probability"]
    ):
        blocking_reasons.append("reference_occlusion_risk_above_gate")
    defer_cost = float(costs["task_noncompletion_cost"])
    grasp_allowed = not blocking_reasons
    if grasp_allowed and grasp_cost <= defer_cost:
        selected = "grasp"
        selected_cost = grasp_cost
    else:
        selected = "defer"
        selected_cost = defer_cost
    return {
        "wrong_target_risk": wrong_target_risk,
        "fully_hidden_probability": hidden_probability,
        "reference_occlusion_probability": occlusion_probability,
        "membership_entropy_normalized": membership_entropy,
        "grasp_cost": grasp_cost,
        "defer_cost": defer_cost,
        "grasp_allowed": grasp_allowed,
        "blocking_reasons": blocking_reasons,
        "selected_terminal_action": selected,
        "selected_terminal_cost": selected_cost,
        "selected_target_correct_probability": (
            selected_target_correct_probability
        ),
    }


def forecast_view_action(
    belief: dict[str, dict[str, float]],
    model: dict[str, Any],
    action: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a view by branching over predicted outcomes, never saved future files."""
    selected_action = model_action(action, model)
    predicted = predict_after_action(belief, model, action)
    perception_outcomes = predictive_outcome_distribution(
        predicted["perception"],
        model["perception_observation"][selected_action],
        tuple(model["perception_observations"]),
    )
    membership_outcomes = predictive_outcome_distribution(
        predicted["membership"],
        model["membership_observation"][selected_action],
        MEMBERSHIP_OBSERVATIONS,
    )
    track_outcomes = predictive_outcome_distribution(
        predicted["target_track"],
        model["track_belief_observation"][selected_action],
        tuple(model["track_belief_observations"]),
    )
    branches = []
    expected_terminal_cost = 0.0
    for (
        combined_observation,
        membership_observation,
        track_observation,
    ) in (
        itertools.product(
            model["perception_observations"],
            MEMBERSHIP_OBSERVATIONS,
            model["track_belief_observations"],
        )
    ):
        identity_bin, occlusion_observation = (
            combined_observation.split("|", maxsplit=1)
        )
        probability = (
            perception_outcomes[combined_observation]
            * membership_outcomes[membership_observation]
            * track_outcomes[track_observation]
        )
        posterior = {
            "perception": bayesian_observation_update(
                predicted["perception"],
                model["perception_observation"][selected_action],
                combined_observation,
            ),
            "membership": bayesian_observation_update(
                predicted["membership"],
                model["membership_observation"][selected_action],
                membership_observation,
            ),
            "target_track": bayesian_observation_update(
                predicted["target_track"],
                model["track_belief_observation"][selected_action],
                track_observation,
            ),
        }
        identity_probability = posterior["target_track"]["correct"]
        terminal = terminal_metrics(
            posterior, float(identity_probability), config
        )
        expected_terminal_cost += (
            probability * terminal["selected_terminal_cost"]
        )
        branches.append(
            {
                "probability": probability,
                "observation": {
                    "identity_bin": identity_bin,
                    "membership": membership_observation,
                    "reference_occlusion": occlusion_observation,
                    "track_belief": track_observation,
                },
                "posterior": posterior,
                "terminal": terminal,
            }
        )
    branch_total = sum(branch["probability"] for branch in branches)
    if not math.isclose(branch_total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError(f"Forecast branches sum to {branch_total}")
    stored_branches = sorted(
        branches,
        key=lambda item: item["probability"],
        reverse=True,
    )[:32]
    action_cost = float(config["task_cost"]["view_action_cost"][action])
    return {
        "action": action,
        "kind": "action_conditioned_future_belief",
        "predicted_prior_after_action": predicted,
        "observation_branch_count": len(branches),
        "all_branch_probability_sum": branch_total,
        "stored_top_branch_count": len(stored_branches),
        "stored_top_branch_probability_mass": sum(
            branch["probability"] for branch in stored_branches
        ),
        "observation_branches": stored_branches,
        "expected_terminal_cost": expected_terminal_cost,
        "view_action_cost": action_cost,
        "objective_cost": action_cost + expected_terminal_cost,
        "future_held_out_observation_used_for_forecast": False,
    }


def select_root_action(
    belief: dict[str, dict[str, float]],
    selected_target_correct_probability: float,
    model: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Choose the first receding-horizon action after applying commitment gates."""
    current_terminal = terminal_metrics(
        belief, selected_target_correct_probability, config
    )
    values = [
        {
            "action": "defer",
            "kind": "terminal_noncommitment",
            "objective_cost": current_terminal["defer_cost"],
        }
    ]
    if current_terminal["grasp_allowed"]:
        values.append(
            {
                "action": "grasp",
                "kind": "terminal_commitment",
                "objective_cost": current_terminal["grasp_cost"],
                "terminal": current_terminal,
            }
        )
    for action in config["reachable_view_actions"]:
        values.append(
            forecast_view_action(belief, model, str(action), config)
        )
    selected = min(values, key=lambda item: item["objective_cost"])
    return {
        "planner": "offline_discrete_action_conditioned_belief_mpc",
        "horizon": 2,
        "feedback_policy": True,
        "replans_after_first_action": True,
        "action_values": values,
        "selected_action": selected["action"],
        "selected_cost": selected["objective_cost"],
        "current_terminal": current_terminal,
        "future_held_out_observation_used_for_action_selection": False,
    }


def audit_terminal(
    method: str,
    seed: int,
    variant: str,
    terminal_action: str,
    observation_row: dict[str, Any],
    observation_count: int,
    selected_first_action: str,
    details: dict[str, Any],
    *,
    grasp_target_correct_posthoc: bool | None = None,
) -> dict[str, Any]:
    selected_correct = bool(
        observation_row["selected_target_correct_posthoc"]
        if grasp_target_correct_posthoc is None
        else grasp_target_correct_posthoc
    )
    hidden = (
        observation_row["total_visibility_state_posthoc"]
        == "fully_hidden"
    )
    grasped = terminal_action == "grasp"
    return {
        "method": method,
        "seed": seed,
        "variant": variant,
        "selected_first_action": selected_first_action,
        "terminal_action": terminal_action,
        "terminal_observation_action": observation_row["action"],
        "observation_count": observation_count,
        "correct_grasp": grasped and selected_correct,
        "wrong_grasp": grasped and not selected_correct,
        "deferred": not grasped,
        "safe_defer_fully_hidden": (not grasped) and hidden,
        "missed_visible_target": (not grasped) and not hidden,
        "selected_candidate_correct_posthoc": selected_correct,
        "terminal_target_fully_hidden_posthoc": hidden,
        "planning_details": details,
        "simulator_ground_truth_used_during_action_selection": False,
        "simulator_ground_truth_used_for_posthoc_audit": True,
    }


def replay_mpc(
    method: str,
    seed: int,
    episode: dict[str, dict],
    model: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Replay one held-out episode while revealing observations only after action selection."""
    center = episode["initial_observation"]
    belief, selected_probability = update_with_observation(
        initial_belief(model),
        center,
        model,
        action="initial_observation",
    )
    root_policy = select_root_action(
        belief, selected_probability, model, config
    )
    first_action = str(root_policy["selected_action"])
    terminal_row = center
    terminal_action = first_action
    terminal_policy = root_policy["current_terminal"]
    observations = 1
    if first_action.startswith("viewpoint_"):
        # The held-out cached row is accessed only after the root action is
        # fixed.  It was not used to fit this leave-one-out model.
        terminal_row = episode[first_action]
        predicted_belief = predict_after_action(
            belief, model, first_action
        )
        updated_belief, updated_selected_probability = (
            update_with_observation(
                predicted_belief,
                terminal_row,
                model,
                action=first_action,
            )
        )
        terminal_policy = terminal_metrics(
            updated_belief,
            updated_selected_probability,
            config,
        )
        terminal_action = terminal_policy["selected_terminal_action"]
        observations = 2
    details = {
        "leave_one_episode_out_model_training_seeds": (
            model["training_seeds"]
        ),
        "held_out_seed": seed,
        "action_agnostic_model": model["action_agnostic"],
        "center_observation": {
            key: center[key]
            for key in center["planner_observation_fields"]
        },
        "belief_after_center": belief,
        "root_policy": root_policy,
        "terminal_policy": terminal_policy,
        "held_out_future_observation_applied_only_after_selection": (
            first_action.startswith("viewpoint_")
        ),
    }
    return audit_terminal(
        method,
        seed,
        center["variant"],
        terminal_action,
        terminal_row,
        observations,
        first_action,
        details,
        grasp_target_correct_posthoc=bool(
            terminal_row["center_selected_track_available"]
            and terminal_row[
                "center_selected_track_correct_posthoc"
            ]
        ),
    )


def replay_confidence_baseline(
    seed: int,
    episode: dict[str, dict],
    model: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    center = episode["initial_observation"]
    belief, selected_probability = update_with_observation(
        initial_belief(model),
        center,
        model,
        action="initial_observation",
    )
    center_terminal = terminal_metrics(
        belief, selected_probability, config
    )
    if center_terminal["selected_terminal_action"] == "grasp":
        terminal_row = center
        terminal = center_terminal
        first_action = "grasp"
        observations = 1
    else:
        first_action = "viewpoint_close_high"
        terminal_row = episode[first_action]
        predicted = predict_after_action(belief, model, first_action)
        updated, updated_probability = update_with_observation(
            predicted, terminal_row, model, action=first_action
        )
        terminal = terminal_metrics(
            updated, updated_probability, config
        )
        observations = 2
    return audit_terminal(
        "confidence_only_fixed_reobservation",
        seed,
        center["variant"],
        terminal["selected_terminal_action"],
        terminal_row,
        observations,
        first_action,
        {
            "uses_action_conditioned_future_belief_for_view_selection": False,
            "fixed_reobservation_action": "viewpoint_close_high",
            "center_terminal": center_terminal,
            "terminal_policy": terminal,
        },
    )


def fixed_policy(
    method: str,
    seed: int,
    episode: dict[str, dict],
    action: str,
) -> dict[str, Any]:
    if action == "grasp":
        row = episode["initial_observation"]
        observations = 1
    else:
        row = episode[action]
        observations = 2
    return audit_terminal(
        method,
        seed,
        episode["initial_observation"]["variant"],
        "grasp",
        row,
        observations,
        action,
        {
            "policy": "fixed_no_future_belief",
            "future_cached_observation_used_for_action_selection": False,
        },
    )


def summarize_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    episode_count = len(rows)
    correct = sum(row["correct_grasp"] for row in rows)
    wrong = sum(row["wrong_grasp"] for row in rows)
    deferred = sum(row["deferred"] for row in rows)
    safe_defer = sum(row["safe_defer_fully_hidden"] for row in rows)
    missed = sum(row["missed_visible_target"] for row in rows)
    grasp_count = correct + wrong
    return {
        "episode_count": episode_count,
        "correct_grasp_count": correct,
        "wrong_grasp_count": wrong,
        "defer_count": deferred,
        "safe_defer_fully_hidden_count": safe_defer,
        "missed_visible_target_count": missed,
        "retrieval_success_rate": correct / episode_count,
        "wrong_commitment_rate": wrong / episode_count,
        "wrong_commitment_rate_given_grasp": (
            wrong / grasp_count if grasp_count else None
        ),
        "safe_outcome_rate": (correct + safe_defer) / episode_count,
        "mean_observation_count": sum(
            row["observation_count"] for row in rows
        )
        / episode_count,
        "first_action_counts": dict(
            sorted(Counter(row["selected_first_action"] for row in rows).items())
        ),
        "terminal_action_counts": dict(
            sorted(Counter(row["terminal_action"] for row in rows).items())
        ),
    }


def run_noncompletion_cost_sensitivity(
    episodes: dict[int, dict[str, dict]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    results = []
    for value in config["task_cost"][
        "task_noncompletion_cost_sensitivity_grid"
    ]:
        sensitivity_config = copy.deepcopy(config)
        sensitivity_config["task_cost"]["task_noncompletion_cost"] = float(
            value
        )
        rows = []
        for seed in sorted(episodes):
            training = {
                other_seed: episode
                for other_seed, episode in episodes.items()
                if other_seed != seed
            }
            model = fit_action_model(
                training,
                sensitivity_config,
                action_agnostic=False,
            )
            rows.append(
                replay_mpc(
                    "action_conditioned_belief_mpc",
                    seed,
                    episodes[seed],
                    model,
                    sensitivity_config,
                )
            )
        results.append(
            {
                "task_noncompletion_cost": float(value),
                "summary": summarize_method(rows),
            }
        )
    return results


def run_experiment(config_path: Path) -> dict[str, Any]:
    config = load_json(config_path)
    if config.get("training_performed") is not False:
        raise ValueError("This replay must not train model weights")
    if config.get("testing_performed") is not False:
        raise ValueError("This replay is calibration, not testing")
    if config.get("reserved_test_seeds_used") is not False:
        raise ValueError("Reserved test seeds must remain unused")
    started = time.perf_counter()
    episodes = build_episode_rows(config)
    method_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for seed in sorted(episodes):
        held_out = episodes[seed]
        training = {
            other_seed: episode
            for other_seed, episode in episodes.items()
            if other_seed != seed
        }
        conditioned_model = fit_action_model(
            training, config, action_agnostic=False
        )
        agnostic_model = fit_action_model(
            training, config, action_agnostic=True
        )
        method_rows["immediate_grasp"].append(
            fixed_policy(
                "immediate_grasp",
                seed,
                held_out,
                "grasp",
            )
        )
        method_rows["fixed_close_high_then_grasp"].append(
            fixed_policy(
                "fixed_close_high_then_grasp",
                seed,
                held_out,
                "viewpoint_close_high",
            )
        )
        method_rows["fixed_right_then_grasp"].append(
            fixed_policy(
                "fixed_right_then_grasp",
                seed,
                held_out,
                "viewpoint_right",
            )
        )
        method_rows["confidence_only_fixed_reobservation"].append(
            replay_confidence_baseline(
                seed,
                held_out,
                conditioned_model,
                config,
            )
        )
        method_rows["action_agnostic_belief_mpc"].append(
            replay_mpc(
                "action_agnostic_belief_mpc",
                seed,
                held_out,
                agnostic_model,
                config,
            )
        )
        method_rows["action_conditioned_belief_mpc"].append(
            replay_mpc(
                "action_conditioned_belief_mpc",
                seed,
                held_out,
                conditioned_model,
                config,
            )
        )
    expected_methods = set(config["methods"])
    if set(method_rows) != expected_methods:
        raise AssertionError(
            f"Methods {sorted(method_rows)} != {sorted(expected_methods)}"
        )
    sensitivity = run_noncompletion_cost_sensitivity(episodes, config)
    result = {
        "schema_version": "offline-action-conditioned-mpc-replay-v1",
        "experiment_id": config["experiment_id"],
        "status": "completed",
        "purpose": (
            "cpu_only_action_conditioned_future_belief_integration_pilot"
        ),
        "evaluation_protocol": config["evaluation_protocol"],
        "episode_count": len(episodes),
        "seeds": sorted(episodes),
        "methods": {
            method: {
                "summary": summarize_method(rows),
                "episodes": rows,
            }
            for method, rows in method_rows.items()
        },
        "task_noncompletion_cost_sensitivity": sensitivity,
        "runtime_seconds": time.perf_counter() - started,
        "gpu_used": False,
        "gpu_memory_gib": 0.0,
        "isaac_sim_launched": False,
        "vlm_inference_performed": False,
        "cached_vlm_outputs_reused": True,
        "training_performed": False,
        "calibration_performed": True,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
        "limitations": [
            "The same 20-scene calibration collection is used in leave-one-episode-out folds.",
            "This is calibration replay, not a frozen unbiased test.",
            "Only one re-observation action is executed before terminal replanning.",
            "The discrete scene families do not cover lid opening or object interaction.",
            "Task-cost weights and commitment limits remain provisional.",
            "Simulator ground truth is used for fold fitting and post-hoc audit only.",
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
        if key != "methods"
    }
    summary["method_summaries"] = {
        method: payload["summary"]
        for method, payload in result["methods"].items()
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"OFFLINE_MPC_RESULT={result_path}")
    print(f"OFFLINE_MPC_SUMMARY={summary_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run_experiment(args.config.resolve())


if __name__ == "__main__":
    main()
