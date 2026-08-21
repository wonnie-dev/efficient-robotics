#!/usr/bin/env python3
"""Cross-validate the joint observation model and planner replay.

The script consumes saved calibration RGB-D and cached perception outputs.  It
does not run Isaac Sim or a neural model.  Simulator masks are read only after
inference to assign calibration labels.  Every reported episode is held out
from the observation model used to replay that episode.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from calibrated_belief import bayesian_update  # noqa: E402
from rgbd_target_localization import localize_mask_files  # noqa: E402
from unified_task_belief_planner import plan  # noqa: E402
from unified_task_belief_planner import (  # noqa: E402
    update_belief_and_task_state,
    validate_unified_method_contract,
)


DEFAULT_PERCEPTION_ROOT = (
    ROOT / "outputs/calibration/calibration_episodes_perception"
)
DEFAULT_OUTPUT_ROOT = ROOT / "outputs/calibration/joint_observation_model"
PREFLIGHT_MODEL = ROOT / "configs/research/task_belief_validation.json"
HYPOTHESES = (
    "track_center_selected|inside",
    "track_center_selected|outside",
    "track_other_target|inside",
    "track_other_target|outside",
    "target_absent|not_applicable",
)
INFORMATION_ACTIONS = (
    "remove_cover",
    "viewpoint_close_high",
    "viewpoint_right",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def binary_mask(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 0


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    union = int(np.logical_or(first, second).sum())
    if union == 0:
        return 0.0
    return float(np.logical_and(first, second).sum() / union)


def resolve_input_asset(model_input: dict[str, Any], value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = ROOT / path
    if candidate.exists():
        return candidate
    return Path(model_input["image"]["rgb_path"]).resolve().parent / path


def selected_relation(result: dict[str, Any], relation_type: str) -> str:
    selected = str(result["selected_candidate_id"])
    return candidate_relation(result, selected, relation_type)


def candidate_relation(
    result: dict[str, Any], candidate_id: str, relation_type: str
) -> str:
    item = next(
        (
            row
            for row in result.get("relations", [])
            if row.get("source_id") == candidate_id
            and row.get("relation_type") == relation_type
        ),
        None,
    )
    return str(item["top_label"]) if item is not None else "unknown"


def selected_logit(result: dict[str, Any]) -> float:
    selected = str(result["selected_candidate_id"])
    index = list(result["candidate_ids"]).index(selected)
    return float(result["raw_match_logits"][index])


def candidate_logit(result: dict[str, Any], candidate_id: str) -> float:
    index = list(result["candidate_ids"]).index(candidate_id)
    return float(result["raw_match_logits"][index])


def candidate_localizations(
    result: dict[str, Any], observation_dir: Path
) -> dict[str, dict[str, Any]]:
    model_input = load_json(Path(result["input_path"]))
    masks = {
        str(candidate["candidate_id"]): resolve_input_asset(
            model_input, str(candidate["mask_path"])
        )
        for candidate in model_input["candidates"]
    }
    if not masks:
        return {}
    return dict(localize_mask_files(observation_dir, masks)["estimates"])


def euclidean(first: list[float], second: list[float]) -> float:
    return math.sqrt(
        sum((float(a) - float(b)) ** 2 for a, b in zip(first, second))
    )


def true_candidate(
    result: dict[str, Any], observation_dir: Path, minimum_iou: float
) -> tuple[str | None, float]:
    target = binary_mask(observation_dir / "target_visible_mask.png")
    if not bool(target.any()):
        return None, 0.0
    model_input = load_json(Path(result["input_path"]))
    overlaps = {
        str(candidate["candidate_id"]): mask_iou(
            target,
            binary_mask(resolve_input_asset(model_input, str(candidate["mask_path"]))),
        )
        for candidate in model_input["candidates"]
    }
    if not overlaps:
        return None, 0.0
    candidate_id = max(overlaps, key=overlaps.get)
    overlap = float(overlaps[candidate_id])
    return (candidate_id if overlap >= minimum_iou else None), overlap


def match_label(raw_logit: float) -> str:
    """Use Qwen's native match/non-match decision boundary, not a risk gate."""
    return "match" if float(raw_logit) >= 0.0 else "nonmatch"


def observation_symbol(row: dict[str, Any]) -> str:
    evidence = dict(row["candidate_evidence"])
    center_track = row.get("center_track_candidate_id")
    if center_track in evidence:
        center = evidence[str(center_track)]
        if match_label(float(center["raw_match_logit"])) == "match":
            return f"center_target|{center['membership']}"
    other = [
        (candidate_id, value)
        for candidate_id, value in evidence.items()
        if candidate_id != center_track
    ]
    if other:
        _, best = max(other, key=lambda item: float(item[1]["raw_match_logit"]))
        if match_label(float(best["raw_match_logit"])) == "match":
            return f"other_target|{best['membership']}"
    return "target_unresolved|unknown"


def persistent_observation_symbol(row: dict[str, Any]) -> str:
    """Encode the strongest matched persistent track for the unified MPC."""
    evidence = dict(row["candidate_evidence"])
    center_track = row.get("center_track_candidate_id")
    matches = [
        (candidate_id, value)
        for candidate_id, value in evidence.items()
        if match_label(float(value["raw_match_logit"])) == "match"
    ]
    if not matches:
        return "no_target_evidence"
    candidate_id, value = max(
        matches, key=lambda item: float(item[1]["raw_match_logit"])
    )
    identity = (
        "center_target" if candidate_id == center_track else "other_target"
    )
    return f"{identity}|{value['membership']}"


def normalize_counts(
    counts: Counter[str], keys: tuple[str, ...], alpha: float
) -> dict[str, float]:
    denominator = sum(counts.values()) + alpha * len(keys)
    if denominator <= 0.0:
        raise ValueError("Dirichlet smoothing requires alpha > 0")
    return {key: (counts[key] + alpha) / denominator for key in keys}


def build_rows(
    perception_root: Path,
    *,
    minimum_iou: float,
    maximum_track_distance_m: float,
    persistent_semantic_symbols: bool = False,
) -> dict[int, dict[str, Any]]:
    manifest = load_json(perception_root / "observation_manifest.json")
    episodes: dict[int, dict[str, Any]] = {}
    for episode in manifest["episodes"]:
        seed = int(episode["seed"])
        rows: dict[str, dict[str, Any]] = {}
        center_result = load_json(
            perception_root
            / "grounded_sam2_qwen_rankings"
            / f"seed{seed}_center"
            / "result.json"
        )
        center_dir = Path(episode["observations"]["center"]["rgb.png"]["path"]).parent
        center_selected = str(center_result["selected_candidate_id"])
        center_estimates = candidate_localizations(center_result, center_dir)
        center_reference = center_estimates.get(center_selected)
        center_true, center_iou = true_candidate(
            center_result, center_dir, minimum_iou
        )
        expected_membership = str(episode["expected_membership"])
        if expected_membership == "not_applicable":
            truth = "target_absent|not_applicable"
        else:
            track = (
                "track_center_selected"
                if center_true == center_selected
                else "track_other_target"
            )
            truth = f"{track}|{expected_membership}"
        if truth not in HYPOTHESES:
            raise ValueError(f"Unsupported truth state for seed {seed}: {truth}")

        for view in episode["observations"]:
            action = {
                "center": "initial_observation",
                "post_remove": "remove_cover",
                "close_high": "viewpoint_close_high",
                "right": "viewpoint_right",
            }[view]
            result = load_json(
                perception_root
                / "grounded_sam2_qwen_rankings"
                / f"seed{seed}_{view}"
                / "result.json"
            )
            observation_dir = Path(
                episode["observations"][view]["rgb.png"]["path"]
            ).parent
            selected = str(result["selected_candidate_id"])
            current_true, current_iou = true_candidate(
                result, observation_dir, minimum_iou
            )
            agreement = None
            track_distance = None
            center_track_candidate_id = center_selected
            if action != "initial_observation":
                estimates = candidate_localizations(result, observation_dir)
                matched_id = None
                if center_reference is not None and estimates:
                    matched_id, matched = min(
                        estimates.items(),
                        key=lambda item: euclidean(
                            center_reference["center_world_m"],
                            item[1]["center_world_m"],
                        ),
                    )
                    track_distance = euclidean(
                        center_reference["center_world_m"],
                        matched["center_world_m"],
                    )
                    if track_distance > maximum_track_distance_m:
                        matched_id = None
                if matched_id is None:
                    agreement = "missing"
                else:
                    agreement = "same" if selected == matched_id else "other"
                center_track_candidate_id = matched_id
            candidate_evidence = {
                str(candidate_id): {
                    "raw_match_logit": candidate_logit(result, str(candidate_id)),
                    "membership": candidate_relation(
                        result, str(candidate_id), "membership"
                    ),
                }
                for candidate_id in result["candidate_ids"]
            }
            row = {
                "seed": seed,
                "family": str(episode["family"]),
                "view": view,
                "action": action,
                "true_joint_hypothesis": truth,
                "selected_candidate_id": selected,
                "selected_raw_match_logit": selected_logit(result),
                "membership": selected_relation(result, "membership"),
                "center_track_agreement": agreement,
                "center_track_candidate_id": center_track_candidate_id,
                "center_track_distance_m": track_distance,
                "candidate_evidence": candidate_evidence,
                "true_candidate_id_posthoc": current_true,
                "true_candidate_iou_posthoc": current_iou,
                "selected_target_correct_posthoc": selected == current_true,
                "simulator_ground_truth_used_for_inference": False,
                "simulator_ground_truth_used_for_calibration_label": True,
            }
            row["observation_symbol"] = (
                persistent_observation_symbol(row)
                if persistent_semantic_symbols
                else observation_symbol(row)
            )
            rows[action] = row
        episodes[seed] = {
            "seed": seed,
            "family": str(episode["family"]),
            "collector": (
                "remove_cover_counterfactual"
                if "remove_cover" in rows
                else "static_reachable_views"
            ),
            "initial_task_state": "covered" if "remove_cover" in rows else "open",
            "true_joint_hypothesis": truth,
            "center_true_candidate_iou_posthoc": center_iou,
            "rows": rows,
        }
    return episodes


def fit_model(
    episodes: dict[int, dict[str, Any]], alpha: float, preflight: dict[str, Any]
) -> dict[str, Any]:
    prior_counts = Counter(
        str(episode["true_joint_hypothesis"]) for episode in episodes.values()
    )
    model: dict[str, Any] = {
        "schema_version": "direct-joint-observation-model-v1",
        "planner_template_schema_version": preflight.get("schema_version"),
        "semantic_hypotheses": list(HYPOTHESES),
        "observable_task_states": ["covered", "open"],
        "initial_task_state": "covered",
        "initial_semantic_belief": normalize_counts(
            prior_counts, HYPOTHESES, alpha
        ),
        "horizon": int(preflight["horizon"]),
        "costs": dict(preflight["costs"]),
        "terminal_grasp_actions": dict(preflight["terminal_grasp_actions"]),
        "information_actions": {},
        "observation_model": {},
        "fit_episode_count": len(episodes),
        "fit_seeds": sorted(episodes),
        "dirichlet_alpha": alpha,
        "marginal_confidence_product_used": False,
    }
    all_actions = ("initial_observation", *INFORMATION_ACTIONS)
    for action in all_actions:
        action_rows = [
            episode["rows"][action]
            for episode in episodes.values()
            if action in episode["rows"]
        ]
        observed_outcomes = {
            str(row["observation_symbol"]) for row in action_rows
        }
        persistent_failure_outcomes: set[str] = set()
        if action != "initial_observation":
            template_action = preflight["information_actions"][action]
            for source_state, transitions in template_action[
                "next_task_state_by_outcome"
            ].items():
                persistent_failure_outcomes.update(
                    outcome
                    for outcome, destination in transitions.items()
                    if destination == source_state
                )
        vocabulary = tuple(
            sorted(
                observed_outcomes
                | persistent_failure_outcomes
                | {"unseen"}
            )
        )
        grouped: dict[str, Counter[str]] = defaultdict(Counter)
        support = Counter()
        for row in action_rows:
            truth = str(row["true_joint_hypothesis"])
            grouped[truth][str(row["observation_symbol"])] += 1
            support[truth] += 1
        likelihood = {
            hypothesis: normalize_counts(grouped[hypothesis], vocabulary, alpha)
            for hypothesis in HYPOTHESES
        }
        model["observation_model"][action] = {
            "outcomes": list(vocabulary),
            "likelihood": likelihood,
            "support_by_hypothesis": dict(support),
        }
        if action == "initial_observation":
            continue
        if action == "remove_cover":
            allowed = ["covered"]
            template_transitions = preflight["information_actions"][action][
                "next_task_state_by_outcome"
            ]
            transitions = {}
            for state in allowed:
                transitions[state] = {
                    outcome: template_transitions.get(state, {}).get(
                        outcome, "open"
                    )
                    for outcome in vocabulary
                }
        else:
            allowed = ["open"]
            transitions = {
                "open": {outcome: "open" for outcome in vocabulary}
            }
        stage_cost = float(preflight["information_actions"][action]["stage_cost"])
        model["information_actions"][action] = {
            "kind": preflight["information_actions"][action]["kind"],
            "allowed_task_states": allowed,
            "stage_cost": stage_cost,
            "outcomes": list(vocabulary),
            "next_task_state_by_outcome": transitions,
            "observation_likelihood": {
                state: likelihood for state in allowed
            },
        }
    return model


def likelihood_for_with_source(
    model: dict[str, Any],
    action: str,
    symbol: str,
    *,
    semantic_backoff_model: dict[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Return an action likelihood with calibrated exact-symbol backoff.

    The action-conditioned model remains the first choice.  A valid semantic
    symbol can nevertheless be absent from one action's finite calibration
    support (for example, ``other_target|outside`` was observed from the right
    view but not close-high).  Treating that positive observation as
    ``unseen`` turns target evidence into false negative evidence.  When a
    separate calibration model is supplied, pool the same symbol from the
    actions where it was actually observed.  No simulator label or future
    observation is used at runtime.
    """
    observation_model = model["observation_model"][action]
    if symbol in observation_model["outcomes"]:
        return (
            {
                hypothesis: float(
                    observation_model["likelihood"][hypothesis][symbol]
                )
                for hypothesis in HYPOTHESES
            },
            {
                "requested_action": action,
                "requested_symbol": symbol,
                "used_symbol": symbol,
                "source": "action_conditioned_exact_symbol",
                "backoff_actions": [],
            },
        )

    if semantic_backoff_model is not None:
        sources = [
            (source_action, source_model)
            for source_action, source_model in semantic_backoff_model[
                "observation_model"
            ].items()
            if symbol in source_model["outcomes"]
        ]
        if sources:
            likelihood = {
                hypothesis: sum(
                    float(source["likelihood"][hypothesis][symbol])
                    for _source_action, source in sources
                )
                / len(sources)
                for hypothesis in HYPOTHESES
            }
            return (
                likelihood,
                {
                    "requested_action": action,
                    "requested_symbol": symbol,
                    "used_symbol": symbol,
                    "source": "cross_action_exact_semantic_symbol_backoff",
                    "backoff_actions": [name for name, _source in sources],
                },
            )

    used = "unseen"
    return (
        {
            hypothesis: float(
                observation_model["likelihood"][hypothesis][used]
            )
            for hypothesis in HYPOTHESES
        },
        {
            "requested_action": action,
            "requested_symbol": symbol,
            "used_symbol": used,
            "source": "action_conditioned_unseen_fallback",
            "backoff_actions": [],
        },
    )


def likelihood_for(
    model: dict[str, Any],
    action: str,
    symbol: str,
    *,
    semantic_backoff_model: dict[str, Any] | None = None,
) -> dict[str, float]:
    likelihood, _source = likelihood_for_with_source(
        model,
        action,
        symbol,
        semantic_backoff_model=semantic_backoff_model,
    )
    return likelihood


def held_out_sensor_nll(
    model: dict[str, Any], episode: dict[str, Any]
) -> tuple[float, int]:
    """Score a held-out episode without using its labels during fitting."""
    truth = str(episode["true_joint_hypothesis"])
    loss = -math.log(
        max(1e-12, float(model["initial_semantic_belief"][truth]))
    )
    terms = 1
    for action, row in episode["rows"].items():
        loss -= math.log(
            max(
                1e-12,
                likelihood_for(
                    model, action, str(row["observation_symbol"])
                )[truth],
            )
        )
        terms += 1
    return loss, terms


def select_alpha_episode_disjoint(
    episodes: dict[int, dict[str, Any]],
    alpha_grid: tuple[float, ...],
    preflight: dict[str, Any],
) -> dict[str, Any]:
    if len(episodes) < 3:
        raise ValueError("Nested alpha selection requires at least three episodes")
    candidates = []
    for alpha in alpha_grid:
        total_loss = 0.0
        total_terms = 0
        for held_out_seed, held_out_episode in episodes.items():
            inner_training = {
                seed: episode
                for seed, episode in episodes.items()
                if seed != held_out_seed
            }
            model = fit_model(inner_training, float(alpha), preflight)
            loss, terms = held_out_sensor_nll(model, held_out_episode)
            total_loss += loss
            total_terms += terms
        candidates.append(
            {
                "alpha": float(alpha),
                "held_out_sensor_nll": total_loss / total_terms,
                "held_out_term_count": total_terms,
            }
        )
    selected = min(
        candidates,
        key=lambda row: (row["held_out_sensor_nll"], row["alpha"]),
    )
    return {
        "method": "nested_episode_disjoint_sensor_nll_grid_search",
        "selected_alpha": selected["alpha"],
        "candidates": candidates,
    }


def execute_replay(
    episode: dict[str, Any], model: dict[str, Any]
) -> dict[str, Any]:
    center = episode["rows"]["initial_observation"]
    belief = bayesian_update(
        model["initial_semantic_belief"],
        likelihood_for(model, "initial_observation", center["observation_symbol"]),
    )
    task_state = str(episode["initial_task_state"])
    remaining = tuple(
        action
        for action in INFORMATION_ACTIONS
        if action in episode["rows"]
    )
    sequence: list[str] = []
    updates = [
        {
            "action": "initial_observation",
            "observation": center["observation_symbol"],
            "posterior": belief,
        }
    ]
    policies = []
    for _ in range(len(remaining) + 1):
        policy = plan(
            belief,
            task_state,
            model,
            horizon=min(int(model["horizon"]), len(remaining)),
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
        symbol = str(row["observation_symbol"])
        used_symbol = (
            symbol
            if symbol in model["information_actions"][action]["outcomes"]
            else "unseen"
        )
        belief, task_state = update_belief_and_task_state(
            belief,
            task_state,
            model["information_actions"][action],
            used_symbol,
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
    retrieval_success = terminal_hypothesis == truth
    target_absent_safe_deferral = bool(
        truth == "target_absent|not_applicable" and terminal == "defer"
    )
    semantic_decision_correct = bool(
        retrieval_success or target_absent_safe_deferral
    )
    wrong = terminal.startswith("grasp:") and not retrieval_success
    return {
        "action_sequence": sequence,
        "terminal_action": terminal,
        "retrieval_success": retrieval_success,
        "target_absent_safe_deferral": target_absent_safe_deferral,
        "semantic_decision_correct": semantic_decision_correct,
        "wrong_commitment": wrong,
        "noncompletion": not terminal.startswith("grasp:"),
        "true_joint_hypothesis": truth,
        "belief_updates": updates,
        "policies": policies,
        "fixed_confidence_threshold_used": False,
        "marginal_confidence_product_used": False,
        "held_out_future_observation_used_for_action_selection": False,
    }


def multiclass_metrics(rows: list[dict[str, Any]], bins: int = 10) -> dict[str, Any]:
    nll = 0.0
    brier = 0.0
    correct = 0
    confidence_rows: list[tuple[float, bool]] = []
    for row in rows:
        posterior = row["initial_posterior"]
        truth = row["true_joint_hypothesis"]
        nll -= math.log(max(1e-12, float(posterior[truth])))
        brier += sum(
            (float(posterior[h]) - (1.0 if h == truth else 0.0)) ** 2
            for h in HYPOTHESES
        )
        predicted = max(posterior, key=posterior.get)
        is_correct = predicted == truth
        correct += int(is_correct)
        confidence_rows.append((float(posterior[predicted]), is_correct))
    ece = 0.0
    reliability = []
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        members = [
            item
            for item in confidence_rows
            if low <= item[0] < high or (index == bins - 1 and item[0] == 1.0)
        ]
        if not members:
            continue
        mean_confidence = sum(item[0] for item in members) / len(members)
        accuracy = sum(item[1] for item in members) / len(members)
        ece += len(members) / len(rows) * abs(mean_confidence - accuracy)
        reliability.append(
            {
                "bin_lower": low,
                "bin_upper": high,
                "count": len(members),
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
            }
        )
    return {
        "episode_count": len(rows),
        "negative_log_likelihood": nll / len(rows),
        "brier_score": brier / len(rows),
        "top1_accuracy": correct / len(rows),
        "expected_calibration_error": ece,
        "reliability_bins": reliability,
    }


def run(
    perception_root: Path,
    output_root: Path,
    *,
    additional_perception_roots: tuple[Path, ...] = (),
    alpha: float,
    alpha_grid: tuple[float, ...] = (),
    minimum_iou: float,
    maximum_track_distance_m: float,
    defer_cost: float | None = None,
    wrong_commitment_cost: float | None = None,
    planner_template: Path = PREFLIGHT_MODEL,
    persistent_semantic_symbols: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    preflight = load_json(planner_template)
    validate_unified_method_contract(preflight)
    if defer_cost is not None:
        preflight["costs"]["defer"] = float(defer_cost)
    if wrong_commitment_cost is not None:
        preflight["costs"]["wrong_commitment"] = float(
            wrong_commitment_cost
        )
    perception_roots = (perception_root, *additional_perception_roots)
    episodes: dict[int, dict[str, Any]] = {}
    for source_root in perception_roots:
        source_episodes = build_rows(
            source_root,
            minimum_iou=minimum_iou,
            maximum_track_distance_m=maximum_track_distance_m,
            persistent_semantic_symbols=persistent_semantic_symbols,
        )
        overlap = set(episodes) & set(source_episodes)
        if overlap:
            raise ValueError(f"Duplicate calibration seeds: {sorted(overlap)}")
        episodes.update(source_episodes)
    cv_rows = []
    for seed, episode in sorted(episodes.items()):
        training = {key: value for key, value in episodes.items() if key != seed}
        alpha_selection = (
            select_alpha_episode_disjoint(training, alpha_grid, preflight)
            if alpha_grid
            else {
                "method": "fixed_alpha",
                "selected_alpha": float(alpha),
                "candidates": [],
            }
        )
        model = fit_model(
            training, float(alpha_selection["selected_alpha"]), preflight
        )
        center_symbol = episode["rows"]["initial_observation"]["observation_symbol"]
        initial_posterior = bayesian_update(
            model["initial_semantic_belief"],
            likelihood_for(model, "initial_observation", center_symbol),
        )
        replay = execute_replay(episode, model)
        cv_rows.append(
            {
                "seed": seed,
                "family": episode["family"],
                "true_joint_hypothesis": episode["true_joint_hypothesis"],
                "initial_observation_symbol": center_symbol,
                "initial_posterior": initial_posterior,
                "held_out_episode_used_for_fit": False,
                "outer_fold_alpha_selection": alpha_selection,
                **replay,
            }
        )

    metrics = multiclass_metrics(cv_rows)
    family_support = Counter(episode["family"] for episode in episodes.values())
    truth_support = Counter(
        episode["true_joint_hypothesis"] for episode in episodes.values()
    )
    summary = {
        "episode_count": len(cv_rows),
        "target_present_episode_count": sum(
            row["true_joint_hypothesis"] != "target_absent|not_applicable"
            for row in cv_rows
        ),
        "retrieval_success_count": sum(row["retrieval_success"] for row in cv_rows),
        "target_present_retrieval_success_rate": (
            sum(row["retrieval_success"] for row in cv_rows)
            / sum(
                row["true_joint_hypothesis"] != "target_absent|not_applicable"
                for row in cv_rows
            )
        ),
        "target_absent_safe_deferral_count": sum(
            row["target_absent_safe_deferral"] for row in cv_rows
        ),
        "semantic_decision_correct_count": sum(
            row["semantic_decision_correct"] for row in cv_rows
        ),
        "semantic_decision_accuracy": sum(
            row["semantic_decision_correct"] for row in cv_rows
        ) / len(cv_rows),
        "wrong_commitment_count": sum(row["wrong_commitment"] for row in cv_rows),
        "wrong_commitment_rate": sum(row["wrong_commitment"] for row in cv_rows) / len(cv_rows),
        "noncompletion_count": sum(row["noncompletion"] for row in cv_rows),
        "first_action_counts": dict(Counter(row["action_sequence"][0] for row in cv_rows)),
        "outer_fold_selected_alpha_counts": dict(
            Counter(
                str(row["outer_fold_alpha_selection"]["selected_alpha"])
                for row in cv_rows
            )
        ),
        "initial_joint_calibration_metrics": metrics,
        "family_support": dict(family_support),
        "joint_hypothesis_support": dict(truth_support),
    }
    expected_action_checks = []
    for row in cv_rows:
        sequence = row["action_sequence"]
        family = row["family"]
        if family == "inside_close_high_resolving":
            passed = "viewpoint_close_high" in sequence
            expected = "remove_cover_then_viewpoint_close_high"
        elif family == "inside_right_resolving":
            passed = "viewpoint_right" in sequence
            expected = "remove_cover_then_viewpoint_right"
        elif family == "inside_center_selected_control":
            # This supplemental family is not one of the reserved-test
            # headline scenarios.  Some center renders expose the mug but not
            # the white-logo attribute, so forcing an immediate grasp would
            # reward unsafe commitment.  The valid control is correct
            # retrieval, with re-observation allowed when the calibrated
            # posterior remains uncertain.
            passed = bool(
                row["retrieval_success"]
                and not row["wrong_commitment"]
                and sequence
                and sequence[-1].startswith("grasp:")
            )
            expected = "correct_inside_retrieval_with_optional_reobservation"
        elif family == "outside_other_target_right_resolving":
            passed = bool(
                "remove_cover" in sequence
                and "viewpoint_right" in sequence
                and sequence[-1].startswith("grasp:")
            )
            expected = "remove_cover_then_viewpoint_right_then_outside_grasp"
        elif family == "outside_reobservation_required":
            # The policy is conditioned on the actual center observation, not
            # the family name.  If calibrated center evidence already resolves
            # target identity and outside membership, grasping immediately is
            # valid.  When the target is unresolved, a viewpoint action must
            # precede grasp.  This distinction is the behavior the risk-aware
            # policy is intended to demonstrate.
            center_resolved = (
                row["initial_observation_symbol"] == "center_target|outside"
            )
            if center_resolved:
                passed = bool(
                    sequence
                    and sequence[0].startswith("grasp:")
                    and row["retrieval_success"]
                    and not row["wrong_commitment"]
                )
                expected = "immediate_grasp_when_center_evidence_is_resolved"
            else:
                passed = bool(
                    any(action.startswith("viewpoint_") for action in sequence)
                    and sequence[-1].startswith("grasp:")
                    and row["retrieval_success"]
                    and not row["wrong_commitment"]
                )
                expected = "viewpoint_before_grasp_when_center_is_unresolved"
        elif family == "target_absent_negative_evidence":
            passed = bool(
                "remove_cover" in sequence and row["target_absent_safe_deferral"]
            )
            expected = "remove_cover_then_safe_deferral"
        elif family == "partial_close_high_resolving":
            passed = bool(
                sequence
                and
                "viewpoint_close_high" in sequence
                and sequence[-1].startswith("grasp:")
                and row["retrieval_success"]
                and not row["wrong_commitment"]
            )
            expected = "close_high_before_correct_grasp"
        elif family == "partial_right_semantic_gate":
            passed = bool(
                sequence
                and
                "viewpoint_right" in sequence
                and sequence[-1].startswith("grasp:")
                and row["retrieval_success"]
                and not row["wrong_commitment"]
            )
            expected = "right_view_before_correct_grasp"
        elif family == "partial_right_resolving":
            center_resolved = (
                row["initial_observation_symbol"] == "center_target|inside"
            )
            passed = bool(
                sequence
                and row["retrieval_success"]
                and not row["wrong_commitment"]
                and sequence[-1].startswith("grasp:")
                and (center_resolved or "viewpoint_right" in sequence)
            )
            expected = (
                "immediate_grasp_when_center_evidence_is_resolved"
                if center_resolved
                else "right_view_before_correct_grasp"
            )
        elif family in {
            "covered_close_high_resolving",
            "covered_right_resolving",
        }:
            required_view = (
                "viewpoint_close_high"
                if family == "covered_close_high_resolving"
                else "viewpoint_right"
            )
            passed = bool(
                sequence
                and "remove_cover" in sequence
                and required_view in sequence
                and sequence[-1].startswith("grasp:")
                and row["retrieval_success"]
                and not row["wrong_commitment"]
            )
            expected = f"remove_cover_then_{required_view}_before_correct_grasp"
        elif family == "target_absent_covered":
            passed = bool(
                sequence
                and "remove_cover" in sequence
                and row["target_absent_safe_deferral"]
                and not row["wrong_commitment"]
            )
            expected = "remove_cover_then_safe_deferral"
        elif family in {
            "ambiguous_inside_rim",
            "ambiguous_outside_behind",
        }:
            passed = bool(
                sequence
                and row["retrieval_success"]
                and not row["wrong_commitment"]
                and sequence[-1].startswith("grasp:")
            )
            expected = "correct_relation_aware_retrieval_with_optional_reobservation"
        else:
            passed = bool(sequence and sequence[0].startswith("grasp:"))
            expected = "immediate_grasp"
        expected_action_checks.append(
            {
                "seed": row["seed"],
                "family": family,
                "expected_behavior": expected,
                "passed": passed,
                "action_sequence": sequence,
            }
        )
    summary["designed_behavior_pass_count"] = sum(
        row["passed"] for row in expected_action_checks
    )
    summary["designed_behavior_pass_rate"] = (
        summary["designed_behavior_pass_count"] / len(expected_action_checks)
    )
    summary["designed_behavior_by_family"] = {
        family: {
            "passed": sum(
                row["passed"]
                for row in expected_action_checks
                if row["family"] == family
            ),
            "total": sum(
                row["family"] == family for row in expected_action_checks
            ),
        }
        for family in sorted(family_support)
    }
    outside_condition_rows = [
        row
        for row in expected_action_checks
        if row["family"] == "outside_reobservation_required"
    ]
    summary["outside_reobservation_condition_coverage"] = {
        "resolved_center_episode_count": sum(
            row["expected_behavior"]
            == "immediate_grasp_when_center_evidence_is_resolved"
            for row in outside_condition_rows
        ),
        "unresolved_center_episode_count": sum(
            row["expected_behavior"]
            == "viewpoint_before_grasp_when_center_is_unresolved"
            for row in outside_condition_rows
        ),
        "condition_consistent_episode_count": sum(
            row["passed"] for row in outside_condition_rows
        ),
        "episode_count": len(outside_condition_rows),
    }
    blockers = []
    if set(truth_support) != set(HYPOTHESES):
        blockers.append("not_all_joint_hypotheses_observed_in_calibration")
    if any(value < 3 for value in truth_support.values()):
        blockers.append("fewer_than_three_episodes_for_a_joint_hypothesis")
    blockers.extend(
        [
            "covered_state_viewpoint_likelihood_not_observed",
            "conditional_rg6_grasp_success_not_measured_on_calibration_split",
            "remove_cover_failure_outcome_not_observed",
            "task_costs_not_empirically_frozen",
            "reserved_test_seeds_not_opened",
        ]
    )
    behavior_by_family = summary["designed_behavior_by_family"]
    outside_coverage = summary["outside_reobservation_condition_coverage"]
    if (
        outside_coverage["resolved_center_episode_count"] == 0
        or outside_coverage["unresolved_center_episode_count"] == 0
        or outside_coverage["condition_consistent_episode_count"]
        != outside_coverage["episode_count"]
    ):
        blockers.append(
            "outside_reobservation_condition_coverage_incomplete"
        )
    if behavior_by_family.get("inside_close_high_resolving", {}).get("passed") != behavior_by_family.get("inside_close_high_resolving", {}).get("total"):
        blockers.append("inside_close_high_resolving_planner_skipped_required_view")
    if behavior_by_family.get("inside_center_selected_control", {}).get("passed") != behavior_by_family.get("inside_center_selected_control", {}).get("total"):
        blockers.append("inside_center_selected_control_safe_retrieval_failed")
    full_alpha_selection = (
        select_alpha_episode_disjoint(episodes, alpha_grid, preflight)
        if alpha_grid
        else {
            "method": "fixed_alpha",
            "selected_alpha": float(alpha),
            "candidates": [],
        }
    )
    full_model = fit_model(
        episodes, float(full_alpha_selection["selected_alpha"]), preflight
    )
    full_model.update(
        {
            "status": "calibration_candidate_not_frozen",
            "blocking_reasons": blockers,
            "training_performed": False,
            "calibration_performed": True,
            "testing_performed": False,
            "valid_for_final_evaluation": False,
            "alpha_selection": full_alpha_selection,
        }
    )
    result = {
        "schema_version": "joint-calibration-result-v1",
        "status": "completed_with_freeze_blockers" if blockers else "passed",
        "protocol": "nested_episode_disjoint_direct_joint_observation_calibration_and_mpc_replay",
        "perception_roots": [str(path.resolve()) for path in perception_roots],
        "summary": summary,
        "episodes": cv_rows,
        "designed_behavior_checks": expected_action_checks,
        "blocking_reasons": blockers,
        "fixed_confidence_threshold_used": False,
        "native_qwen_match_boundary_used_only_as_observation_label": True,
        "persistent_semantic_symbols": persistent_semantic_symbols,
        "marginal_confidence_product_used": False,
        "action_conditioned_future_belief_used": True,
        "simulator_ground_truth_used_for_inference": False,
        "simulator_ground_truth_used_for_calibration_labels": True,
        "training_performed": False,
        "calibration_performed": True,
        "testing_performed": False,
        "reserved_test_seeds_used": False,
        "valid_for_final_evaluation": False,
        "gpu_used": False,
        "full_calibration_alpha_selection": full_alpha_selection,
        "planner_costs": dict(preflight["costs"]),
        "planner_template": str(planner_template.resolve()),
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json(output_root / "result.json", result)
    write_json(output_root / "summary.json", {k: v for k, v in result.items() if k != "episodes"})
    write_json(output_root / "calibration_candidate_model.json", full_model)
    flat_rows = []
    for episode in episodes.values():
        flat_rows.extend(episode["rows"].values())
    with (output_root / "posthoc_calibration_rows.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception-root", type=Path, default=DEFAULT_PERCEPTION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--additional-perception-root", type=Path, action="append", default=[]
    )
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5)
    parser.add_argument(
        "--alpha-grid",
        default="0.05,0.1,0.25,0.5,1.0,2.0,4.0",
        help="Comma-separated nested-CV smoothing candidates; empty uses --dirichlet-alpha.",
    )
    parser.add_argument("--minimum-target-iou", type=float, default=0.25)
    parser.add_argument("--maximum-track-distance-m", type=float, default=0.12)
    parser.add_argument("--defer-cost", type=float)
    parser.add_argument("--wrong-commitment-cost", type=float)
    parser.add_argument(
        "--planner-template",
        type=Path,
        default=PREFLIGHT_MODEL,
        help="Unified semantic-belief/task-state planner template.",
    )
    args = parser.parse_args()
    result = run(
        args.perception_root.resolve(),
        args.output_root.resolve(),
        additional_perception_roots=tuple(
            path.resolve() for path in args.additional_perception_root
        ),
        alpha=float(args.dirichlet_alpha),
        alpha_grid=tuple(
            float(value)
            for value in args.alpha_grid.split(",")
            if value.strip()
        ),
        minimum_iou=float(args.minimum_target_iou),
        maximum_track_distance_m=float(args.maximum_track_distance_m),
        defer_cost=args.defer_cost,
        wrong_commitment_cost=args.wrong_commitment_cost,
        planner_template=args.planner_template.resolve(),
    )
    print(json.dumps({"status": result["status"], "summary": result["summary"], "blocking_reasons": result["blocking_reasons"], "runtime_seconds": result["runtime_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
