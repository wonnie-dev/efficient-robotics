"""Replan from an RGB-D observation produced by physical cover removal.

This adapter closes the causal contract between an already completed Isaac
Sim remove-cover action and the discrete Scene Graph / belief-MPC planner.  It
intentionally uses simulator instance masks as an automatic pilot label.  It
does not claim learned perception or final-evaluation validity.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_cover_search_belief_mpc import (  # noqa: E402
    execute_observation_action,
    normalize,
    plan,
    validate_config,
)
from run_cover_search_scene_graph_mpc_integration import (  # noqa: E402
    action_request,
    build_scene_graph,
    canonical_hash,
    graph_to_planner_belief,
)
from validate_uncertainty_scene_graph import validate as validate_graph  # noqa: E402


DEFAULT_INTEGRATION_CONFIG = (
    ROOT
    / "configs"
    / "research"
    / "cover_search_scene_graph_mpc_integration.json"
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def target_pixels(observation_dir: Path) -> int:
    objects = load_json(observation_dir / "objects.json")
    target = objects.get("target_red")
    if not isinstance(target, dict):
        raise ValueError(
            f"target_red is missing from {observation_dir / 'objects.json'}"
        )
    return int(target.get("pixel_count", 0))


def physical_outcome(
    server_result: dict[str, Any],
    post_remove_dir: Path,
    *,
    minimum_target_pixels: int,
) -> tuple[str, int]:
    removal = server_result.get("cover_removal_execution") or {}
    if not (
        server_result.get("cover_removal_executed")
        and removal.get("status") == "completed"
        and removal.get("removal_verified")
    ):
        return "action_failed", 0
    pixels = target_pixels(post_remove_dir)
    return (
        "target_detected" if pixels >= minimum_target_pixels else "empty_container",
        pixels,
    )


def bind_graph_to_observation(
    graph: dict[str, Any],
    *,
    observation_dir: Path,
    target_pixel_count: int,
    perception_source: str,
) -> dict[str, Any]:
    rgb = (observation_dir / "rgb.png").resolve()
    depth = (observation_dir / "depth_m.npy").resolve()
    if not rgb.is_file() or not depth.is_file():
        raise FileNotFoundError(
            f"RGB-D observation is incomplete: {observation_dir}"
        )
    graph["observation"].update(
        {
            "rgb_path": str(rgb),
            "depth_path": str(depth),
        }
    )
    target_node = next(
        node for node in graph["nodes"] if node["id"] == "target_red"
    )
    target_node["observation"].update(
        {
            "visible": target_pixel_count > 0,
            "visible_fraction": None,
            "bbox_xyxy": None,
            "depth_mean_m": None,
        }
    )
    graph["provenance"].update(
        {
            "perception_source": perception_source,
            "ground_truth_used_for_control": True,
            "notes": (
                "Simulator instance masks were consumed only after the actual "
                "remove-cover action completed. This is an automatic pilot "
                "oracle adapter, not learned perception or final evaluation."
            ),
        }
    )
    validate_graph(graph)
    return graph


def run_replan(
    physical_run_root: Path,
    output_root: Path,
    *,
    integration_config_path: Path = DEFAULT_INTEGRATION_CONFIG,
    minimum_target_pixels: int = 100,
) -> dict[str, Any]:
    started = time.perf_counter()
    physical_run_root = physical_run_root.resolve()
    output_root = output_root.resolve()
    integration_config = load_json(integration_config_path)
    planner_config = load_json(
        resolve_project_path(integration_config["planner_config"])
    )
    validate_config(planner_config)

    server_result_path = physical_run_root / "server_result.json"
    request_path = physical_run_root / "action_request_000.json"
    center_dir = physical_run_root / "observations" / "center"
    post_remove_dir = physical_run_root / "observations" / "post_remove"
    server_result = load_json(server_result_path)
    physical_request = load_json(request_path)
    if physical_request.get("type") != "remove_cover":
        raise ValueError("Physical run did not request remove_cover first")

    episode_id = f"physical_remove_cover_seed{int(server_result['seed']):03d}"
    initial_belief = normalize(planner_config["initial_belief"])
    initial_graph = build_scene_graph(
        integration_config,
        planner_config,
        episode_id=episode_id,
        belief=initial_belief,
        sequence_index=0,
        observation_symbol="center_before_physical_remove",
    )
    initial_graph = bind_graph_to_observation(
        initial_graph,
        observation_dir=center_dir,
        target_pixel_count=target_pixels(center_dir),
        perception_source="simulator_instance_mask_pre_action_pilot_oracle",
    )
    initial_policy = plan(graph_to_planner_belief(initial_graph), planner_config)
    initial_request = action_request(initial_graph, initial_policy, step_index=0)
    if initial_request["type"] != "remove_cover":
        raise RuntimeError(
            "Belief-MPC did not select remove_cover for the physical root action"
        )
    initial_request.update(
        {
            "physical_execution_requested": True,
            "physical_request_source": str(request_path.resolve()),
        }
    )

    outcome, post_pixels = physical_outcome(
        server_result,
        post_remove_dir,
        minimum_target_pixels=minimum_target_pixels,
    )
    update = execute_observation_action(
        initial_belief,
        "remove_cover",
        outcome,
        planner_config,
    )
    post_graph = build_scene_graph(
        integration_config,
        planner_config,
        episode_id=episode_id,
        belief=update["posterior"],
        sequence_index=1,
        observation_symbol=outcome,
    )
    post_graph = bind_graph_to_observation(
        post_graph,
        observation_dir=post_remove_dir,
        target_pixel_count=post_pixels,
        perception_source="simulator_instance_mask_post_action_pilot_oracle",
    )
    post_policy = plan(graph_to_planner_belief(post_graph), planner_config)
    next_request = action_request(post_graph, post_policy, step_index=1)
    next_request["physical_execution_requested"] = False

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "scene_graph_000.json", initial_graph)
    write_json(output_root / "action_request_000.json", initial_request)
    write_json(
        output_root / "action_result_000.json",
        {
            "schema_version": "physical-cover-action-result-v1",
            "request_id": initial_request["request_id"],
            "type": "remove_cover",
            "status": "completed" if outcome != "action_failed" else "failed",
            "observation_symbol": outcome,
            "physical_execution": True,
            "collision_checked": True,
            "result_arrived_after_request": True,
            "source_server_result": str(server_result_path.resolve()),
            "post_remove_observation_dir": str(post_remove_dir.resolve()),
            "target_visible_pixels": post_pixels,
        },
    )
    write_json(output_root / "scene_graph_001.json", post_graph)
    write_json(output_root / "action_request_001.json", next_request)
    result = {
        "schema_version": "physical-remove-cover-replan-pilot-v1",
        "status": "completed",
        "episode_id": episode_id,
        "seed": int(server_result["seed"]),
        "source_physical_run": str(physical_run_root),
        "root_action": initial_request["type"],
        "root_action_physical_execution_verified": outcome != "action_failed",
        "post_action_observation": outcome,
        "post_action_target_visible_pixels": post_pixels,
        "initial_joint_belief": initial_belief,
        "posterior_joint_belief": update["posterior"],
        "next_action": next_request["type"],
        "scene_graph_before_sha256": canonical_hash(initial_graph),
        "scene_graph_after_sha256": canonical_hash(post_graph),
        "runtime_seconds": time.perf_counter() - started,
        "rgbd_consumed_after_physical_action": True,
        "negative_evidence_supported": True,
        "negative_evidence_observed_this_episode": outcome == "empty_container",
        "simulator_ground_truth_used_for_pilot_control": True,
        "learned_perception_executed": False,
        "next_action_physical_execution": False,
        "training_performed": False,
        "calibration_performed": False,
        "testing_performed": False,
        "valid_for_final_evaluation": False,
        "limitations": [
            "The post-action observation symbol comes from simulator instance masks.",
            "The replanned next action is written but not physically executed in this adapter.",
            "Planner observation probabilities and costs remain development values.",
        ],
    }
    write_json(output_root / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("physical_run_root", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument(
        "--integration-config",
        type=Path,
        default=DEFAULT_INTEGRATION_CONFIG,
    )
    parser.add_argument("--minimum-target-pixels", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_replan(
        args.physical_run_root,
        args.output_root,
        integration_config_path=args.integration_config.resolve(),
        minimum_target_pixels=args.minimum_target_pixels,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
