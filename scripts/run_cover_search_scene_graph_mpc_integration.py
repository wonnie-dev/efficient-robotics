"""CPU-only Scene Graph to cover-search belief-MPC contract integration."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cover_search_belief_mpc import (  # noqa: E402
    entropy,
    execute_observation_action,
    grasp_success_probability,
    marginal_cover,
    marginal_location,
    normalize,
    plan,
    validate_config,
)
from validate_uncertainty_scene_graph import validate as validate_graph  # noqa: E402


DEFAULT_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "cover_search_scene_graph_mpc_integration.json"
)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON configuration or cached observation."""
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    """Resolve a repository-relative path."""
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def canonical_hash(payload: dict[str, Any]) -> str:
    """Hash a JSON payload using stable key ordering."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def uncertainty_record(value: float, method: str) -> dict[str, Any]:
    """Store an uncertainty value together with its estimation method."""
    return {
        "status": "available",
        "method": method,
        "value": float(value),
        "calibrated": False,
        "units": "nats" if "entropy" in method else "probability",
    }


def empty_observation(visible: bool) -> dict[str, Any]:
    """Build the null target observation used by negative-evidence updates."""
    return {
        "visible": visible,
        "visible_fraction": 0.0,
        "bbox_xyxy": None,
        "depth_mean_m": None,
    }


def task_failure_risk(
    belief: dict[str, float],
    planner_config: dict[str, Any],
) -> float:
    """Measure risk against the best currently available terminal grasp."""
    terminal_success = [
        grasp_success_probability(action, belief)
        for action in planner_config["actions"].values()
        if action["kind"] == "terminal_grasp" and action["enabled"]
    ]
    return 1.0 - max(terminal_success)


def build_scene_graph(
    integration_config: dict[str, Any],
    planner_config: dict[str, Any],
    *,
    episode_id: str,
    belief: dict[str, float],
    sequence_index: int,
    observation_symbol: str,
) -> dict[str, Any]:
    """Create and validate one graph revision from the supplied joint belief."""
    joint = normalize(belief)
    location = marginal_location(joint)
    cover = marginal_cover(joint)
    joint_entropy = entropy(joint)
    location_entropy = entropy(location)
    graph = {
        "schema_version": "0.2.0-draft",
        "schema_status": (
            "provisional_pending_overleaf_method_definition"
        ),
        "scene_id": integration_config["scene_id"],
        "episode_id": episode_id,
        "seed": 0,
        "task": {
            "instruction": integration_config["instruction"],
            "target_condition": (
                "red mug with white logo inside or near the basket"
            ),
            "allowed_relations": ["inside", "outside", "near"],
        },
        "observation": {
            "view_id": observation_symbol,
            "sequence_index": sequence_index,
            "rgb_path": "not_available_cpu_symbolic_contract",
            "depth_path": "not_available_cpu_symbolic_contract",
            "camera_prim": (
                "/World/RobotSystem/RG6/Zivid2Camera"
            ),
            "timestamp_seconds": None,
        },
        "nodes": [
            {
                "id": "target_red",
                "type": "object",
                "observation": empty_observation(False),
                "belief": {
                    "class_distribution": {
                        "red_mug_with_white_logo": 0.85,
                        "other": 0.15,
                    },
                    "target_probability": 0.85,
                    "target_uncertainty": uncertainty_record(
                        entropy({"target": 0.85, "other": 0.15}),
                        "categorical_entropy_nats_debug",
                    ),
                    "source": "rule_based_stub",
                },
                "last_observed_view": observation_symbol,
            },
            {
                "id": "basket_01",
                "type": "container",
                "observation": empty_observation(True),
                "belief": {
                    "class_distribution": {
                        "basket": 1.0,
                    },
                    "target_probability": 0.0,
                    "target_uncertainty": uncertainty_record(
                        0.0,
                        "categorical_entropy_nats_debug",
                    ),
                    "source": "rule_based_stub",
                },
                "last_observed_view": observation_symbol,
            },
            {
                "id": "cover_01",
                "type": "cover",
                "observation": empty_observation(True),
                "belief": {
                    "class_distribution": {
                        "covered": cover["covered"],
                        "open": cover["open"],
                    },
                    "target_probability": 0.0,
                    "target_uncertainty": uncertainty_record(
                        entropy(cover),
                        "categorical_entropy_nats_debug",
                    ),
                    "source": "rule_based_stub",
                },
                "last_observed_view": observation_symbol,
            },
        ],
        "edges": [
            {
                "id": "target_red_to_basket_01",
                "source": "target_red",
                "target": "basket_01",
                "type": "spatial_relation_belief",
                "belief": {
                    "relation_distribution": {
                        "inside": location["inside"],
                        "outside": location["outside_near"],
                    },
                    "relation_uncertainty": uncertainty_record(
                        location_entropy,
                        "categorical_entropy_nats_debug",
                    ),
                    "source": "rule_based_stub",
                },
                "last_updated_view": observation_symbol,
            }
        ],
        "graph_belief": {
            "most_likely_target": "target_red",
            "target_uncertainty": uncertainty_record(
                entropy({"target": 0.85, "other": 0.15}),
                "categorical_entropy_nats_debug",
            ),
            "relation_uncertainty": uncertainty_record(
                location_entropy,
                "categorical_entropy_nats_debug",
            ),
            "task_failure_risk": uncertainty_record(
                task_failure_risk(joint, planner_config),
                "maximum_terminal_success_complement_debug",
            ),
            "observation_count": sequence_index + 1,
            "joint_task_state_distribution": joint,
        },
        "provenance": {
            "schema_file": integration_config["scene_graph_schema"],
            "perception_source": (
                "scripted post-action CPU contract symbol; no image model"
            ),
            "ground_truth_used_for_control": False,
            "notes": (
                "Interface validation only. RGB-D paths are explicitly "
                "unavailable and all probabilities are uncalibrated debug "
                "beliefs."
            ),
        },
    }
    validate_graph(graph)
    return graph


def graph_to_planner_belief(graph: dict[str, Any]) -> dict[str, float]:
    """Read the planner state from a validated scene-graph revision."""
    validate_graph(graph)
    joint = graph["graph_belief"].get(
        "joint_task_state_distribution"
    )
    if joint is None:
        raise ValueError(
            "Scene Graph lacks joint_task_state_distribution"
        )
    return normalize(joint)


def action_request(
    graph: dict[str, Any],
    policy: dict[str, Any],
    *,
    step_index: int,
) -> dict[str, Any]:
    """Bind the selected root action to the graph revision that produced it."""
    selected = str(policy["selected_action"])
    selected_value = next(
        item
        for item in policy["action_values"]
        if item["action"] == selected
    )
    branches = [
        {
            "observation": branch["observation"],
            "probability": branch["probability"],
            "continuation_action": branch[
                "continuation_action"
            ],
        }
        for branch in selected_value.get(
            "observation_branches", []
        )
    ]
    graph_revision = canonical_hash(graph)
    request = {
        "schema_version": "cover-search-action-request-v1",
        "request_id": (
            f"{graph['episode_id']}:step{step_index:03d}"
        ),
        "episode_id": graph["episode_id"],
        "step_index": step_index,
        "type": selected,
        "source_scene_graph_sha256": graph_revision,
        "planner": policy["planner"],
        "planner_horizon": policy["horizon"],
        "selected_objective_cost": policy["selected_cost"],
        "predicted_observation_branches": branches,
        "execute_only_first_action": True,
        "future_observation_used_for_selection": False,
        "physical_execution_requested": False,
        "interface_target": (
            "cover_01"
            if selected == "remove_cover"
            else "target_red"
        ),
    }
    return request


def execute_contract_stub(
    request: dict[str, Any],
    *,
    expected_action: str | None,
    observation: str | None,
) -> dict[str, Any]:
    """Check request/result ordering for the CPU-only executor contract."""
    if expected_action is not None and request["type"] != expected_action:
        raise ValueError(
            f"Expected {expected_action}, got {request['type']}"
        )
    terminal = request["type"].startswith("grasp_") or (
        request["type"] == "defer"
    )
    if terminal and observation is not None:
        raise ValueError("Terminal stub result cannot contain observation")
    if not terminal and observation is None:
        # An information action without a result cannot revise the belief.
        raise ValueError("Information action requires post-action observation")
    return {
        "schema_version": "cover-search-action-result-v1",
        "request_id": request["request_id"],
        "episode_id": request["episode_id"],
        "step_index": request["step_index"],
        "type": request["type"],
        "status": "accepted_cpu_contract_stub",
        "observation_symbol": observation,
        "physical_execution": False,
        "collision_checked": False,
        "result_arrived_after_request": True,
        "source_scene_graph_sha256": request[
            "source_scene_graph_sha256"
        ],
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a formatted experiment artifact."""
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def run_episode(
    episode: dict[str, Any],
    integration_config: dict[str, Any],
    planner_config: dict[str, Any],
    episode_root: Path,
) -> dict[str, Any]:
    """Write an auditable graph-request-result chain for one episode."""
    episode_root.mkdir(parents=True, exist_ok=True)
    belief = normalize(planner_config["initial_belief"])
    observation_index = 0
    graph = build_scene_graph(
        integration_config,
        planner_config,
        episode_id=episode["episode_id"],
        belief=belief,
        sequence_index=0,
        observation_symbol="initial_center_symbolic",
    )
    trace = []
    terminal_action = None
    maximum_steps = len(episode["observations"]) + 2
    for step_index in range(maximum_steps):
        graph_path = episode_root / (
            f"scene_graph_{step_index:03d}.json"
        )
        write_json(graph_path, graph)
        adapter_belief = graph_to_planner_belief(graph)
        policy = plan(adapter_belief, planner_config)
        request = action_request(
            graph, policy, step_index=step_index
        )
        request_path = episode_root / (
            f"action_request_{step_index:03d}.json"
        )
        write_json(request_path, request)
        selected = request["type"]
        terminal = selected.startswith("grasp_") or selected == "defer"
        if terminal:
            result = execute_contract_stub(
                request,
                expected_action=episode["expected_terminal_action"],
                observation=None,
            )
            terminal_action = selected
        else:
            if observation_index >= len(episode["observations"]):
                raise RuntimeError(
                    "Planner requested an action without scripted "
                    "post-action evidence"
                )
            scripted = episode["observations"][observation_index]
            result = execute_contract_stub(
                request,
                expected_action=scripted["expected_action"],
                observation=scripted["outcome"],
            )
        result_path = episode_root / (
            f"action_result_{step_index:03d}.json"
        )
        write_json(result_path, result)
        trace.append(
            {
                "step_index": step_index,
                "scene_graph": str(graph_path),
                "scene_graph_sha256": canonical_hash(graph),
                "planner_selected_action": selected,
                "action_request": str(request_path),
                "action_result": str(result_path),
                "observation_symbol": result["observation_symbol"],
                "physical_execution": False,
            }
        )
        if terminal:
            break
        # Results are incorporated only after the request is fixed. The next
        # graph is a new revision built from the resulting posterior.
        update = execute_observation_action(
            adapter_belief,
            selected,
            str(result["observation_symbol"]),
            planner_config,
        )
        observation_index += 1
        graph = build_scene_graph(
            integration_config,
            planner_config,
            episode_id=episode["episode_id"],
            belief=update["posterior"],
            sequence_index=observation_index,
            observation_symbol=str(result["observation_symbol"]),
        )
    if terminal_action is None:
        raise RuntimeError("Episode did not reach a terminal action")
    final_graph_path = episode_root / "scene_graph_final.json"
    write_json(final_graph_path, graph)
    return {
        "episode_id": episode["episode_id"],
        "status": "completed",
        "true_state_posthoc": episode["true_state_posthoc"],
        "true_state_used_for_control": False,
        "actions": [
            item["planner_selected_action"] for item in trace
        ],
        "terminal_action": terminal_action,
        "expected_terminal_action": episode[
            "expected_terminal_action"
        ],
        "terminal_matches_expected": (
            terminal_action
            == episode["expected_terminal_action"]
        ),
        "observation_count": observation_index,
        "final_scene_graph": str(final_graph_path),
        "final_joint_belief": graph[
            "graph_belief"
        ]["joint_task_state_distribution"],
        "trace": trace,
        "future_observation_used_during_planning": False,
        "physical_execution": False,
    }


def run_experiment(config_path: Path) -> dict[str, Any]:
    """Replay one Scene Graph and belief-planning integration experiment."""
    integration_config = load_json(config_path)
    planner_config = load_json(
        resolve_path(integration_config["planner_config"])
    )
    validate_config(planner_config)
    started = time.perf_counter()
    output_root = resolve_path(integration_config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    episodes = [
        run_episode(
            episode,
            integration_config,
            planner_config,
            output_root / episode["episode_id"],
        )
        for episode in planner_config["scripted_diagnostic_episodes"]
    ]
    result = {
        "schema_version": (
            "cover-search-scene-graph-mpc-integration-v1"
        ),
        "experiment_id": integration_config["experiment_id"],
        "status": (
            "completed"
            if all(
                episode["status"] == "completed"
                and episode["terminal_matches_expected"]
                for episode in episodes
            )
            else "partial_failure"
        ),
        "purpose": integration_config["purpose"],
        "episodes": episodes,
        "runtime_seconds": time.perf_counter() - started,
        "scene_graph_schema_validated": True,
        "action_request_result_linkage_validated": True,
        "joint_belief_roundtrip_validated": True,
        "gpu_used": False,
        "gpu_memory_gib": 0.0,
        "isaac_sim_launched": False,
        "vlm_inference_performed": False,
        "robot_motion_mpc_executed": False,
        "cover_manipulation_executed": False,
        "executor_mode": integration_config["executor"]["mode"],
        "physical_execution": False,
        "training_performed": False,
        "calibration_performed": False,
        "testing_performed": False,
        "valid_for_final_evaluation": False,
        "limitations": [
            "The executor and observations are CPU contract stubs.",
            "Scene Graph RGB-D paths explicitly state that images are unavailable.",
            "Planner probabilities and costs remain hand specified and uncalibrated.",
            "No UR10e, RG6, collision, cover physics, or learned perception was executed.",
        ],
    }
    result_path = output_root / "result.json"
    write_json(result_path, result)
    summary = {
        key: value for key, value in result.items() if key != "episodes"
    }
    summary["episode_summaries"] = [
        {
            "episode_id": episode["episode_id"],
            "status": episode["status"],
            "actions": episode["actions"],
            "terminal_action": episode["terminal_action"],
            "terminal_matches_expected": episode[
                "terminal_matches_expected"
            ],
            "observation_count": episode["observation_count"],
        }
        for episode in episodes
    ]
    summary_path = output_root / "summary.json"
    write_json(summary_path, summary)
    print(f"SCENE_GRAPH_MPC_RESULT={result_path}")
    print(f"SCENE_GRAPH_MPC_SUMMARY={summary_path}")
    return result


def main() -> None:
    """Run the integration experiment selected on the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run_experiment(args.config.resolve())


if __name__ == "__main__":
    main()
